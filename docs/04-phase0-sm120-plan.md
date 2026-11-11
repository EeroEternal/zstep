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
1. **`test_04_mhc_and_engram.py`**：
   * 用 Triton 实现 `mhc_apply_pre_quantize` 与 `mhc_shifted_post_pre_norm` 多流残差融合算子；
   * 实现 Engram FP8/E8M0 查表：实测对比 **`VRAM Sharded`（分片常驻 8 张 6000D 显存）** 与 **`Host Pinned UVA`（驻留主机内存走 PCIe 5.0 读取）** 的延迟，确定默认策略。
2. **`test_05_pcie_allreduce_8gpu.py`（决定多卡效率的关键战役）**：
   * 编写 `worker/csrc/comm_pcie_sm120.cu`，测试 8 张 RTX 6000D 在张量大小 `16 KB ~ 8 MB`（对应 `1..800` tokens × `5120` BF16）下的：
     * 默认 `torch.distributed` NCCL Ring / Tree AllReduce 延迟（含 CUDA Graph 捕获）；
     * 自定义 PCIe P2P Two-Shot / Hierarchical AllReduce 延迟（含 CUDA Graph 捕获）。

---

## Phase 0 完成里程碑（Exit Criteria）

当以下 4 项指标全部达成，Phase 0 即告收官，立即进入 **Phase 1（组装单层 -> 40 层 `deepseek_v41.py` 显式 Worker 与 `zs-wire` 联调）**：

- [ ] `worker/tests/phase0/` 下全部 6 个测试脚本在 `8 × RTX 6000D` 上绿灯通过，且开启 `TRITON_DEBUG=1` 无 int32 溢出；
- [ ] 热路径在 Warmup 之后执行不同批次大小，**零运行时 JIT 重编译**；
- [ ] 算出精确的 **40 层端到端单步延迟预算表**（例如：`B=1, DSpark T=6` 下，MoE 耗时 $X$ ms + Attention $Y$ ms + 80×AllReduce $Z$ ms = 单步总计 $W$ ms，折算单流吞吐 tokens/s）；
- [ ] 确定最终最优并行组合（`TP_attn=8 + EP_moe=8` vs `TP=4 + PP=2`）与 Engram 驻留位置。
