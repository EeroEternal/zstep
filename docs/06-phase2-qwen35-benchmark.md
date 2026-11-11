# 06 — Phase 2 实测报告：`Qwen3.5-35B-A3B` (`TP=8`) 端到端推理与同机 `vLLM 0.33.0` 对标

本文档记录 `zstep` Phase 2 在同台 `8 × 32 GB sm_120`（无 NVLink、全 `0` P2P、4 个非对称 NUMA 节点）机器上完成真权重端到端模型 **`Qwen/Qwen3.5-35B-A3B`**（`67 GB` BF16 混合架构：30 层 `Qwen3_5MoeGatedDeltaNet` 线性注意力 + 10 层 `PagedGQA` 门控全注意力 + 40 层 `256` 专家 Top-8 + 门控共享专家 MoE）的系统设计、逐 token 贪心对齐验证，以及与 `vLLM 0.33.0`（`TP=8`, `bfloat16`）在两组三档并发（共 6 档）下的正面对标结果。

> **口径约束声明**：
> 1. 本机器物理显存上限为 `8 × 32 GB = 256 GB`，无法容纳 `DeepSeek-V4.1-Flash`（`~510 GB` 权重）；有关 `84 GB` / `96 GB` 大显存卡的容量几何预算在 [`zs-wire`](../crates/zs-wire/src/lib.rs) 中已改为按探测容量计算并显式标记为 **`unverified`**。
> 2. 本机器 `can_device_access_peer` `8 × 8` 全为 `0`，因此文中所有 AllReduce 与端到端 `TP=8` 通信结论均限定在 **无 P2P 的 4-NUMA PCIe 拓扑**（NCCL 亦走降级 SHM 路径），不外推至具备 NVLink 或 PCIe P2P 的拓扑。

---

## 1. 硬件与 NUMA 拓扑钉死

```text
GPU 0, 1, 2 -> PCIe Switch A -> NUMA 0
GPU 3       -> 直连 Root     -> NUMA 1
GPU 4, 5, 6 -> PCIe Switch B -> NUMA 2
GPU 7       -> 直连 Root     -> NUMA 3
NVLink: 无；can_device_access_peer(i, j) == 0 (8x8 全 0)
```

针对该非对称 4-NUMA 拓扑，`zstep` 在 Phase 0.5b 与 Phase 2 落地了三项硬约束：
1. **Worker 启动期自动 NUMA 探测与绑核**（[`numa_pin.py`](../worker/zstep_worker/numa_pin.py)）：通过 `cudaDeviceGetPCIBusId` 反查 `/sys/bus/pci/devices/<bdf>/numa_node` 与 `local_cpulist`，在分配任何主机页锁内存或共享内存页前执行 `os.sched_setaffinity`，并将 `numa_node`（实测 `[0, 0, 0, 1, 2, 2, 2, 3]`）写入共享内存 [`WorkerSlotState`](../crates/zs-wire/src/lib.rs)。
2. **4 KiB 页对齐与本地 NUMA First-Touch 物理页落位**（[`shm_allreduce.py`](../worker/zstep_worker/kernels_sm120/shm_allreduce.py)）：每个 Rank 写入自己分区的首字节触发 Linux 内核在本地 NUMA 节点分配物理页。复测 `8 × GPU` `M = 4096`（`40.96 MiB`）AllReduce（[`step_05b_pcie_allreduce_8gpu.json`](benchmarks/step_05b_pcie_allreduce_8gpu.json)）：
   * `zstep` `HostMappedShmAllReduce`：**`p50 = 4,274.36 µs`，`p95 = 4,289.52 µs`，`p99 = 4,292.75 µs`**（相比 NUMA 优化前 `5,922.64 µs` 下降 **`27.8%`**，`max_abs_error = 0.0`）；
   * 同机降级路径 `NCCL SHM`：`p50 = 6,206.57 µs`，`p95 = 6,227.64 µs`，`p99 = 6,243.15 µs`（在无 P2P 4-NUMA PCIe 拓扑上领先 **`31.1%` / `1.45×`**）。
3. **运行期零驱动锁与零 GC 停顿**（[`runtime_loop.py`](../worker/zstep_worker/runtime_loop.py)）：显存池在初始化末尾一次性采样缓存（消除每步 `cuMemGetInfo_v2` 全局 `nvidia.ko` 驱动锁竞争），并在热循环期间禁用周期性 Python GC（消除任一 Rank GC 暂停通过 `.sys` 屏障放大至 8 卡的尾延迟抖动）。

---

## 2. Phase 2 端到端架构组件

1. **纯 Rust 控制面连续批处理与 `in_flight_depth = 1` 重叠调度**：
   * [`crates/zs-kv/src/radix.rs`](../crates/zs-kv/src/radix.rs)：提供基于 16-token 块粒度的 `PrefixTree`、引用计数 `RadixLease` 与 `KvCacheManager`（LRU 零引用驱逐 + RAII 块回收）。
   * [`crates/zs-scheduler/src/serving.rs`](../crates/zs-scheduler/src/serving.rs)：实现 `SlotPool`（RAII `SlotLease`）与 `ModeBServingEngine`。在 8 张 GPU 执行第 `N` 步解码 CUDA Graph 的同时，Rust 主控在双缓冲的另一槽位预分配第 `N + 1` 步所需的 16-token KV 页并预写 `positions`、`req_slot_ids`、`seq_lens`、`global_kv_page_table` 与 `StepHeader`；待 `eventfd` 唤醒收割第 `N` 步采样出的 token 后，仅原地回填 `token_ids[0..num_seqs]`（`<= 512 B`）即刻发布第 `N + 1` 步。
2. **`sm_120` 显式单路径 `Qwen3.5-35B-A3B` 执行器**：
   * [`worker/zstep_worker/kernels_sm120/qwen35_ops.py`](../worker/zstep_worker/kernels_sm120/qwen35_ops.py)：包含 15 个零运行期重编译的 Triton / CUDA C++ 算子（零中心 `(1 + w)` RMSNorm、融合 64 维部分 RoPE `theta=1e7` + 16-token 分页 KV 写入 prologue、`Split-K=16` 分页门控 GQA 解码与变长 FlashAttention-2 Prefill、`BLOCK_C=32` 因果深度可分离卷积 + `BLOCK_V=4` 寄存器递推 `GatedDeltaNet`、无全局回读的寄存器内 Top-8 路由 + 共享内存 `moe_align_block_size` + Triton Tensor Core 分组专家 GEMM、以及 256 KiB 锁免共享内存 `tp8_argmax_gather`）。
   * [`worker/zstep_worker/models/qwen35_moe.py`](../worker/zstep_worker/models/qwen35_moe.py)：按 Rank 仅从 `safetensors` mmap 切片读取本地 `1/8` 权重分片；预置 `dummy_slot` 与 `dummy_page` 隔离槽位，为解码桶 `(1, 2, 4, 8, 16, 32, 64, 128)` 一次性捕获完整 40 层单图 CUDA Graph（含 80 次跨卡 SHM AllReduce 与跨卡 Argmax 归约，解码期零 CPU 同步）。

---

## 3. 正确性与回归门禁

| 测试项 | 覆盖内容 | 结果 |
| :--- | :--- | :--- |
| `cargo test --workspace` + `clippy -D warnings` | `zs-wire` 动态容量属性断言、`zs-shm`、`zs-kv` Radix 前缀树与 RAII 回收、`zs-scheduler` 连续批处理重叠状态机 | **24 passed, 0 warnings** |
| [`test_qwen35_ops_sm120.py`](../worker/tests/phase2/test_qwen35_ops_sm120.py) | 零中心 RMSNorm + 零重编译断言、`GatedDeltaNet` 变长 Prefill 与单步递推逐位对齐、分页门控 GQA 对齐、**`530,000` 页（`8.6 GB`，偏移 `> 2^31` 元素）`int64` 寻址防溢出测试**、毒化初值下 256 专家 MoE 对齐 | **5 passed** |
| [`test_qwen35_e2e_tp8.py`](../worker/tests/phase2/test_qwen35_e2e_tp8.py) | 8 卡真权重 `Qwen3.5-35B-A3B` 输入 `"The capital of France is"`（`[760, 6511, 314, 9338, 369]`）连续贪心生成 16 个 token，对比 `vLLM 0.33.0` 参考序列 `[11751, 13, 198, 760, 6511, 314, 9338, 369, 11751, 13, 198, 760, 6511, 314, 9338, 369]` + 15 个 Triton kernel 零重编译检查 + `/dev/shm` RAII 清理检查 | **PASSED（16/16 token 精确一致）** |

---

## 4. 同机端到端三档并发对标：`zstep Mode B` vs `vLLM 0.33.0`

* **测试机器**：同一台 `8 × 32 GB sm_120`（无 NVLink，`can_device_access_peer == 0`，4 NUMA `[0,0,0,1,2,2,2,3]`）
* **模型与精度**：`Qwen/Qwen3.5-35B-A3B`，`TP=8`，`bfloat16`，贪心解码（无前缀缓存命中，确定性伪随机非重复 token 序列）
* **原始归档数据**：
  * `vLLM 0.33.0`：[`docs/benchmarks/phase2_vllm_qwen35_35b_a3b_tp8.json`](benchmarks/phase2_vllm_qwen35_35b_a3b_tp8.json)
  * `zstep Mode B`：[`docs/benchmarks/phase2_zstep_qwen35_35b_a3b_tp8.json`](benchmarks/phase2_zstep_qwen35_35b_a3b_tp8.json)

### 4.1 工况 A：`input_len = 512`, `output_len = 256`（`c = 1, 16, 128`）

| 并发度 (`prompts`) | 引擎 | Output 吞吐 (`tok/s`) | Total 吞吐 (`tok/s`) | Median TTFT (`ms`) | Mean TPOT (`ms`) | P99 TPOT (`ms`) | `zstep` 相对 `vLLM` 吞吐增益 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`c = 1`** (`8`) | `vLLM 0.33.0` | `184.72` | `554.16` | `74.51` | `5.09` | `5.38` | — |
| **`c = 1`** (`8`) | **`zstep Mode B`** | **`225.04`** | **`675.11`** | **`66.59`** | **`4.09`** | **`4.11`** | **`+21.8%`** |
| **`c = 16`** (`48`) | `vLLM 0.33.0` | `1,671.63` | `5,014.88` | `392.38` | `8.02` | `9.40` | — |
| **`c = 16`** (`48`) | **`zstep Mode B`** | **`1,768.73`** | **`5,306.19`** | `465.85` *(注1)* | **`7.25`** | **`7.31`** | **`+5.8%`** |
| **`c = 128`** (`256`) | `vLLM 0.33.0` | `3,271.56` | `9,814.67` | `1,240.50` | `33.25` | `37.87` | — |
| **`c = 128`** (`256`) | **`zstep Mode B`** | **`4,540.92`** *(注1)* | **`13,622.75`** | `1,307.87` *(注1)* | **`22.38`** | **`29.57`** | **`+38.8%`** *(注1: `+41.8%`)* |

> *注 1*：表内主数据采用统一的 `--max-step-tokens 8192` 常驻单次连跑 6 档测得（[`phase2_zstep_qwen35_35b_a3b_tp8.json`](benchmarks/phase2_zstep_qwen35_35b_a3b_tp8.json)）；针对 `input=512` 将单步 Prefill 上限设为 `--max-step-tokens 4096` 的完整 6 档实测归档于 [`phase2_zstep_qwen35_35b_a3b_tp8_step4k.json`](benchmarks/phase2_zstep_qwen35_35b_a3b_tp8_step4k.json)，其中 `c=16` `median_ttft` 进一步降至 **`290.93 ms`**（优于 vLLM `25.9%`），`c=128` 输出吞吐达 **`4,638.96 tok/s`**（**`+41.8%`**）、`median_ttft = 822.28 ms`（优于 vLLM `33.7%`）。

### 4.2 工况 B：`input_len = 1024`, `output_len = 256`（`c = 1, 32, 128`）

| 并发度 (`prompts`) | 引擎 | Output 吞吐 (`tok/s`) | Total 吞吐 (`tok/s`) | Median TTFT (`ms`) | Mean TPOT (`ms`) | P99 TPOT (`ms`) | `zstep` 相对 `vLLM` 吞吐增益 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`c = 1`** (`20`) | `vLLM 0.33.0` | `181.96` | `917.22` | `143.35` | `4.86` | — | — |
| **`c = 1`** (`20`) | **`zstep Mode B`** | **`261.72`** | **`1,308.58`** | **`82.73`** | **`3.51`** | **`3.61`** | **`+43.8%`** |
| **`c = 32`** (`200`) | `vLLM 0.33.0` | `1,834.83` | `9,248.90` | `1,091.90` | `11.92` | — | — |
| **`c = 32`** (`200`) | **`zstep Mode B`** | **`2,165.22`** | **`10,826.10`** | **`870.08`** | **`9.81`** | **`12.88`** | **`+18.0%`** |
| **`c = 128`** (`400`) | `vLLM 0.33.0` | `2,939.12` | `14,815.29` | `1,355.88` | `33.77` | — | — |
| **`c = 128`** (`400`) | **`zstep Mode B`** | **`3,024.65`** | **`15,123.25`** | **`1,295.93`** | **`33.19`** | **`39.74`** | **`+2.9%`** |

### 4.3 四阶段渐进优化消融记录（Ablation Study）

完整原始数据归档于 [`phase2_qwen35_optimization_ablation.json`](benchmarks/phase2_qwen35_optimization_ablation.json)（同机亦记录了稠密模型 `Qwen/Qwen3.5-27B` `vLLM 0.33.0` `TP=8` 基线：`c=1` `85.5 tok/s`、`c=128` `1,060 tok/s`）：

| 演进阶段 | 核心改动 | `in512 c=1` (`tok/s` / `p99_itl`) | `in512 c=16` (`tok/s`) | `in512 c=128` (`tok/s`) | `in1024 c=1` (`tok/s`) | `in1024 c=32` (`tok/s`) | `in1024 c=128` (`tok/s`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (`vLLM 0.33.0`)** | `TP=8`, `bfloat16`, 降级 `NCCL SHM` | `184.72` / — | `1,671.63` | `3,271.56` | `181.96` | `1,834.83` | `2,939.12` |
| **Run 1（初始跑通）** | 全 40 层 CUDA Graph + NUMA SHM AllReduce (`blocks=16`, `step=4K`) | `163.82` / `12.53 ms` | `1,421.17` | `4,004.01` | — | — | — |
| **Run 2（消除 GC 与页表冗余）** | 热循环禁用周期 GC + `page_table` 仅拷活跃槽位 + `blocks=8` | `185.69` / `10.18 ms` | `1,552.52` | `3,916.61` | — | — | — |
| **Run 3（消除驱动锁 `ioctl`）** | 启动期静态缓存显存遥测，移除每步 16 次 `cuMemGetInfo_v2` 锁 (`step=4K`) | `224.18` / `5.00 ms` | `1,744.61` | **`4,638.96`** | `257.30` | `2,001.20` | `2,750.47` |
| **Run 4（Prefill Tile 细化 + 8K 批）** | `GatedDeltaNet` `BLOCK_V=4` / `Conv1d` `BLOCK_C=32`（4× SM 占用）+ `step=8K` | **`225.04` / `4.64 ms`** | **`1,768.73`** | `4,540.92` | **`261.72`** | **`2,165.22`** | **`3,024.65`** |

---

## 5. 关键性能归因与后续演进方向

1. **为什么单流（`c=1`）快 `21.8% ~ 43.8%`（解码 TPOT 低至 `3.51 ms`）？**
   * 全 40 层（含 30 层 `GatedDeltaNet` 状态更新、10 层 `PagedGQA`、40 层 `SparseMoE`、80 次跨卡 `HostMappedShmAllReduce` One-Shot 单屏障归约与跨卡 Argmax）完全收敛进 **单张 CUDA Graph**；
   * `4 KiB`（`1 × 2048` BF16）小包走单次 `.sys` 屏障 One-Shot 快路径（`allreduce_blocks = 8`），规避了降级版 `NCCL SHM` 的双跳代理开销，同时 Rust 主控通过 `in_flight_depth = 1` 将下一拍元数据准备 100% 隐藏在 GPU 执行期内。
2. **为什么 `in512_c128` 高并发快 `38.8% ~ 41.8%`（`4,540 ~ 4,639 tok/s`），而 `in1024_c128` 收敛至 `+2.9%`？**
   * 在解码主导的 `in512_c128` 下，`bucket = 128` 的纯解码步中位耗时仅 **`12.73 ms`**（折合纯解码吞吐 `> 10,000 tok/s`）；
   * 在长输入 `in1024_c128`（`400 × 1024 = 40.96 万` 输入 token）下，每批 `8192` token 的 Prefill 需要通过无 P2P 的 4-NUMA 主机内存交换 `80 × 32 MiB = 2.56 GiB` 激活，此时瓶颈转移至主机 DDR5 聚合带宽上限——与硬件评审预判的“撞上主机内存带宽墙后应转向跨层通信隐藏或 chunked prefill 流水化”完全吻合。
