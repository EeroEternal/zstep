# 04 — Phase 0 实施计划：在 `8 × RTX 6000D` (`sm_120`) 上打通 `DeepSeek-V4.1-Flash` 算子栈

在编写完整的调度状态机与 HTTP 服务之前，**Phase 0 的唯一目标是消除所有硬件未知风险**：在 `8 × RTX 6000D` 实机上逐个跑通、对齐数值精度并压测 `DeepSeek-V4.1-Flash` 所需的 **5 大核心算子族 + PCIe 通信原语**，拿到单层与整网 40 层的真实延迟基线。

所有 Phase 0 验证代码统一放在 `worker/tests/phase0/` 下，互不耦合、单脚本秒跑。

---

## Step 0.1 — 机器拓扑摸底与基础工具链锁定（Day 1）

### 执行清单
1. **采集硬件互联拓扑**（决定 8 卡通信算法）：
   ```bash
   nvidia-smi -L
   nvidia-smi topo -m
   nvidia-smi topo -p2p r
   nvidia-smi topo -p2p w
   numactl --hardware
   ```
   * **关键分支判定**：
     * **情况 1（双 PCIe Gen5 Switch 挂载在双路 CPU 下，每 4 卡同 Switch `PIX`/`PXB`，跨组 `SYS`）**：8 卡 AllReduce 必须采用 **Two-Level Hierarchical（Switch 内 4 卡 P2P 直接规约 + 跨 NUMA 仅传 1 份规约值）**，或者对比 `TP=8, EP=8` 与 `TP=4, PP=2` 的实测延迟。
     * **情况 2（8 卡均支持 P2P BAR1 访问）**：直接启用全连接 One-Shot / Two-Shot P2P AllReduce。
     * **情况 3（实测：任意两卡 `nvidia-smi topo -p2p r = CNS`，`can_device_access_peer=False`）**：CUDA IPC / BAR1 P2P 不可用，`b12x.comm.pcie` 也无法建立通道；使用 POSIX `/dev/shm` + `cudaHostRegisterMapped` 的 Host-Mapped Two-Shot AllReduce，NCCL 走主机 SHM 传输。
2. **锁定 Python / CUDA `sm_120` 编译环境**（必须使用独立 `.venv`）：
   * **CUDA Toolkit**：`>= 12.8`（推荐 **CUDA 12.9 / 13.0**，完整支持 `sm_120` `mma.sync.aligned.block_scale` NVFP4/MXFP4/MXFP8 PTX 指令）。
   * **核心包**：PyTorch (`cu128`+)、`nvidia-cutlass-dsl`、`triton` (>= 3.3 支持 `sm_120` TMA 与 FP8/FP4)、`flashinfer-python[b12x]`。
   * **验证门禁**：`worker/tests/phase0/test_00_env_sm120.py` 确认 `torch.cuda.get_device_capability() == (12, 0)` 且 P2P Access 矩阵可用。

---

## Step 0.2 — Dense FP8 / MXFP8 投影算子验证（Day 2）

### `DeepSeek-V4.1-Flash` 真实形状（`TP=8` 单卡切分后）
* `hidden_size = 5120`；权重存储为 `32x32` 或 `1x32` E8M0 块缩放 FP8（MXFP8），还包括 Grouped `wo_a` 投影与 `rmsnorm_quantize_fp8_group32` 前处理。

### 验证动作（`test_01_dense_fp8_gemm.py`）
1. **精度参考**：在 CPU / FP32 GPU 上将 E8M0 Scale 精确展开乘回 FP8 Code 作为 Golden Reference。
2. **`sm_120` 路径打通**：
   * 实现 Triton `rmsnorm_quantize_mxfp8_group32`（融合 RMSNorm + 32 元素一组求 E8M0 Scale + 量化输出 `E4M3`）。
   * 对接 `flashinfer.experimental.b12x` 的 Block-Scaled MXFP8 GEMM（以及 `torch._scaled_mm` `sm_120` cuBLASLt MXFP8 作为对照）。
3. **验收标准**：
   * 相对误差 `cos_sim > 0.9995`，无 NaN/Inf；
   * 测试动态 `M ∈ {6, 48, 192, 2048, 8192}` 不会触发重复 JIT 重编译（固定 Kernel 缓存键）。

---

## Step 0.3 — 384 专家 MXFP4 MoE + `sqrt_softplus` 路由算子（Day 3–5）

这是整网权重占比 >85%、决定 Decode 速度上限的**头号核心算子**。

### `DeepSeek-V4.1-Flash` 真实形状
* 专家总数 `E = 384`，`topk = 6`，`routed_scaling_factor = 1.5`，`normalize_topk_weights = True`；
* **门控函数**：`sqrt_softplus`（注意不是普通 Softmax/Sigmoid！）；
* **专家几何**：`hidden_size = 5120`, `intermediate_size = 2304`，权重为 **MXFP4 (`E2M1` + `E8M0`/32)**；
* **激活截断**：SwiGLU 带限幅 **`swiglu_limit = 10.0`**！
* **并行切分对比**：在 `8 × 6000D` 上实测对比两种切分：
  * **方案 1（`EP=8, TP_moe=1`）**：每张卡持有 `384 / 8 = 48` 个完整专家（`5120 -> 4608 -> 5120`），前后走小包 PCIe Dispatch / Combine；
  * **方案 2（`TP_moe=8, EP=1`）**：每张卡持有全部 384 个专家的 `1/8` 切片（`intermediate = 288`），层末走一次 AllReduce（需测试 `288` 窄切片在 `sm_120` Tensor Core 上的 Tile 利用率，这也是 `EP=8` 通常显著优于 `TP=8` 的原因！）。

### 验证动作（`test_02_moe_mxfp4_sm120.py`）
1. **Triton 编写融合路由 Kernel**：`dsv41_moe_topk_sqrt_softplus`（FP32 输入 logits -> `sqrt_softplus` -> Top-6 选出与归一化 `* 1.5`）。
2. **双后端专家计算对齐**：
   * **Baseline 路径**：W4A16 Marlin / Triton 反量化参考实现，确立精度金标准；
   * **主力高速路径**：接入 `flashinfer[b12x]` 的 `sm_120` Block-Scaled Fused MoE（W4A8 MXFP8/NVFP4 激活或 W4A16），并在 SwiGLU epilogue 中加入 `clamp(..., max=10.0)`。
3. **验收标准**：
   * 在 `M = 6`（单并发 DSpark）、`M = 48`（8 并发 DSpark）、`M = 4096`（Prefill）三档测量单层耗时与等效显存带宽利用率（目标在 `M=48` 下逼近 1.0+ TB/s 有效带宽）。

---

## Step 0.4 — `dsv41` FlatKV 四组编解码、稀疏选块 Indexer 与 Selected Attention（Day 6–8）

### 核心任务
`DeepSeek-V4.1-Flash` 的注意力不使用标准 MHA，而是 **FlatKV 页平面存储 + 128D MXFP4 索引打分选 TopK 块 (`candidate_block_size=8, index_topk=512`) + SWA/CSA `selected_attention`**。

### 验证动作（`test_03_dsv41_attention_sm120.py`）
1. **直接移植并独立化 Triton Portable 实现**：
   * 抽取并精简参考实现中的 Triton 算子（天然兼容 `sm_120`，无 Hopper `wgmma` 依赖）：
     * 页编解码器：`cache_pack` / `cache_scatter`（Global 288B NVFP4、Index 68B MXFP4、SWA 528B MXFP8）；
     * 索引器：`index_key_norm_rope_scatter`、`index_score`（先跑通 Triton 路径，后续可用 `sm_120` 原生 MXFP4 MMA 加速超长上下文打分）、`index_topk`；
     * 稀疏注意力核心：`selected_attention`（同时覆盖 Prefill SWA/CSA、Decode 单行、以及 **DSpark `q_len_per_req = 6` 验证窗口**）。
2. **重点检查 64 位偏移安全（Int64 Offset Guard）**：
   * RTX 6000D 单卡 KV 池可达 **35~40 GB**（远超 `int32` 的 `2 GiB` 字节偏移上限）！所有 `page_id * page_stride` 在 Triton kernel 内必须先转 `tl.int64` 再相乘，并在单测中分配 36 GB 池读写最后几页验证不越界。

---

## Step 0.5 — mHC 残差、Engram 查表与 8 卡 PCIe Gen5 AllReduce（Day 9–10）

### 验证动作
1. **`test_04_mhc_and_engram_sm120.py`**：
   * 验证 `b12x` mHC `post_pre` 多流残差融合算子；
   * 实现 Engram FP8/E8M0 查表：实测对比 **`VRAM Sharded`** 与 **`Host Pinned UVA`** 的延迟，并验证超过 `2^31` 字节偏移与动态 `M` 零重编译。
2. **`test_05_pcie_allreduce_8gpu.py`（决定多卡效率的关键战役）**：
   * 在 `worker/zstep_worker/kernels_sm120/shm_allreduce.py` 内联编译 CUDA Host-Mapped Two-Shot AllReduce，测试 `M ∈ {6,48,192,4096} × 5120 BF16`（`60 KiB ~ 40.96 MiB`）；
   * 对比 `torch.distributed` NCCL 与自定义 SHM AllReduce 的数值、CUDA Graph replay、单次延迟和 80 次通信预算；
   * 显式探测 `b12x.comm.pcie`，在 `CNS` 上记录其 CUDA-IPC 通道不可用，而不是把失败实现计入性能对比。

---

## Phase 0 实测完成结果与 40 层端到端延迟预算（已在 `8 × sm_120` 测试机全部通过）

- [x] `worker/tests/phase0/` 下全部 6 个测试脚本（`test_00` ~ `test_05`）在 8 卡 `sm_120`（Dual-Socket Xeon Gold 6530, PCIe Gen5 x16, 4 SNC NUMA, `P2P=CNS`）测试床 100% 绿灯通过，所有 `>2^31` `int64` 大偏移保护通过；
- [x] 热路径在 Warmup 后跨动态批次 `M ∈ {6, 17, 48, 192, 4096}` **零运行时 JIT 重编译**；
- [x] 原始 JSON 基准归档位于 [`docs/benchmarks/`](./benchmarks/)。

### 1. 单算子 `sm_120` CUDA Graph 实测延迟汇总

| 算子模块 | 精度 (`cos_sim`) | `M = 6` (`B=1` Verify) | `M = 48` (`B=8` Verify) | `M = 192` (`B=32` Verify) | `M = 4096` (Prefill) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `rmsnorm_quantize_mxfp8_group32` (`H=5120`) | `0.99965` | `3.35 µs` | `3.17 µs` | `4.04 µs` | `15.45 µs` |
| `b12x.mxfp8_linear` (`5120 -> 2048` TP8) | `0.99964+` | `15.65 µs` (`696 GB/s`) | `22.27 µs` | `23.45 µs` (`172 TFLOPS`) | `192.52 µs` (`446 TFLOPS`) |
| `dsv41_sqrt_softplus_topk` (`E=384, K=6`) | `1.00000` | `5.74 µs` | `5.79 µs` | `6.89 µs` | `23.88 µs` |
| **`b12x` MXFP4 MoE (`TP=8`, `384×288`)** | **`0.99999`** | **`70.24 µs`** (`1173 GB/s`) | **`435.07 µs`** (`1093 GB/s`) | **`781.73 µs`** (`1141 GB/s`) | **`2414.73 µs`** (`409 GB/s`) |
| `b12x` MXFP4 MoE (`EP=8`, `48×2304` 对照) | `0.99999` | `320.55 µs` | `574.19 µs` | `842.18 µs` | `8865.44 µs` |
| `compressed_sparse_mla` (`deepseek_v41`, 8K–32K) | `0.99960` | `13.52~14.25 µs` | `60.75 µs` | `~185 µs` | `1429.28 µs` (`616 GB/s`) |
| `mHC` (`post_pre`, `streams=4, iters=20`) | `1.00000` | `11.41 µs` | `26.66 µs` | `83.62 µs` | `1846.08 µs` |
| `Engram` (`L1/L14`, VRAM vs Host UVA) | `1.00000` | `4.42 µs` / `6.12 µs` | `3.65 µs` / `20.03 µs` | `5.07 µs` / `68.88 µs` | `48.25 µs` / `2400.12 µs` |
| **8 卡 AllReduce (`HostMappedShm` / `NCCL SHM`)** | **`1.0` / `0.99999`** | `72.53 µs` / **`50.78 µs`** | **`126.31 µs`** / `127.12 µs` | `240.43 µs` / **`154.74 µs`** | **`5922.64 µs`** / `6302.99 µs` |

### 2. `DeepSeek-V4.1-Flash` 40 层端到端单步延迟预算表（`TP=8`）

| 指标项（40 层累加） | 单并发 DSpark (`B=1, M=6`)<br>非 P2P (`CNS`) 实测 | 单并发 DSpark (`B=1, M=6`)<br>`8×RTX 6000D` (P2P Switch 预估) | 8 并发 DSpark (`B=8, M=48`)<br>非 P2P (`CNS`) 实测 |
| :--- | :---: | :---: | :---: |
| Attention 子层（40 × Norm/Proj/SparseMLA） | `1.96 ms` | `1.96 ms` | `4.34 ms` |
| 384 专家 MXFP4 MoE（40 × Router + `TP=8` MoE） | `3.80 ms` | `3.80 ms` | `18.65 ms` |
| `mHC` 多流残差 + `Engram`（第 1、14 层） | `0.47 ms` | `0.47 ms` | `1.07 ms` |
| 80 × 跨 8 卡 AllReduce 通信预算 | `4.06 ms` (`50.8 µs×80`) | `~1.20 ms` (`~15 µs×80`) | `10.10 ms` (`126.3 µs×80`) |
| **40 层总单步耗时 $W$** | **`10.29 ms / step`** | **`~7.43 ms / step`** | **`34.16 ms / step`** |
| **折算吞吐（DSpark 均接受 ~4.2 tok/step）** | **`~408 tok/s`（单流）** | **`~565 tok/s`（单流）** | **`~983 tok/s`（8 并发总计）** |

### 3. 最终锁定的并行与内存驻留决策
* **MoE 与 Attention 并行度**：统一采用 **`TP=8`（全模型对称 8 路张量并行，每层 2 次 AllReduce）**，摒弃 `EP=8`（单流快 `4.56×`，Prefill 快 `3.67×`）。
* **Engram 驻留**：`8 × RTX 6000D (84 GB)` 默认 `"vram_sharded"`；`32 GB` 显存开发卡默认 `"host_uva_pinned"`。
