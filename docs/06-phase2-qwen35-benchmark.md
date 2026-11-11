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
   * [`crates/zs-kv/src/radix.rs`](../crates/zs-kv/src/radix.rs)：提供基于 16-token 块粒度的 `PrefixTree`、引用计数 `RadixLease` 与 `KvCacheManager`（LRU 零引用驱逐 + RAII 块回收）。**该组件目前是独立的纯注意力 KV 组件，尚未接入 `Qwen3.5-35B-A3B` serving**；混合模型包含 `GatedDeltaNet`，必须先实现 block 边界的 recurrent state 快照，才能安全复用前缀状态。当前 serving 只使用 `zs_kv::BlockPool` 做物理页 RAII 管理，不声称启用了 Radix 前缀命中。
   * [`crates/zs-scheduler/src/serving.rs`](../crates/zs-scheduler/src/serving.rs)：实现 `SlotPool`（RAII `SlotLease`）与 `ModeBServingEngine`。在 8 张 GPU 执行第 `N` 步解码 CUDA Graph 的同时，Rust 主控在双缓冲的另一槽位预分配第 `N + 1` 步所需的 16-token KV 页并预写 `positions`、`req_slot_ids`、`seq_lens`、`global_kv_page_table` 与 `StepHeader`；待 `eventfd` 唤醒收割第 `N` 步采样出的 token 后，仅原地回填 `token_ids[0..num_seqs]`（`<= 512 B`）即刻发布第 `N + 1` 步。
2. **`sm_120` 显式单路径 `Qwen3.5-35B-A3B` 执行器**：
   * [`worker/zstep_worker/kernels_sm120/qwen35_ops.py`](../worker/zstep_worker/kernels_sm120/qwen35_ops.py)：零中心 `(1 + w)` RMSNorm、融合 64 维部分 RoPE `theta=1e7` + 16-token 分页 KV 写入 prologue、`Split-K=16` 分页门控 GQA 解码与变长 FlashAttention-2 Prefill、`BLOCK_C=32` 因果深度可分离卷积 + `BLOCK_V=4` 寄存器递推 `GatedDeltaNet`、无全局回读的寄存器内 Top-8 路由 + 共享内存 `moe_align_block_size` + Triton Tensor Core 分组专家 GEMM、以及 256 KiB 锁免共享内存 `tp8_argmax_gather`。运行时审计明确监视 **`WATCHED_KERNELS=17`** 个 Triton kernel，要求 **`TRITON_RECOMPILATIONS=0`**；CUDA C++ JIT 扩展不计入这 17 个 Triton kernel。
   * [`worker/zstep_worker/models/qwen35_moe.py`](../worker/zstep_worker/models/qwen35_moe.py)：按 Rank 仅从 `safetensors` mmap 切片读取本地 `1/8` 权重分片；预置 `dummy_slot` 与 `dummy_page` 隔离槽位，为解码桶 `(1, 2, 4, 8, 16, 32, 64, 128)` 一次性捕获完整 40 层单图 CUDA Graph（含 80 次跨卡 SHM AllReduce 与跨卡 Argmax 归约，解码期零 CPU 同步）。

---

## 3. 正确性与回归门禁

| 测试项 | 覆盖内容 | 结果 |
| :--- | :--- | :--- |
| `cargo test --workspace` + `clippy -D warnings` | `zs-wire` 动态容量属性断言、`zs-shm`、`zs-kv` Radix 前缀树与 RAII 回收、`zs-scheduler` 连续批处理重叠状态机及 TTFT 聚合 | **28 passed, 0 warnings** |
| [`test_qwen35_ops_sm120.py`](../worker/tests/phase2/test_qwen35_ops_sm120.py) | 零中心 RMSNorm + 零重编译断言、`GatedDeltaNet` 变长 Prefill 与单步递推逐位对齐、分页门控 GQA 对齐、**`530,000` 页（`8.6 GB`，偏移 `> 2^31` 元素）`int64` 寻址防溢出测试**、毒化初值下 256 专家 MoE 对齐、TP=8 Argmax 容量守卫 | **6 passed**（含 SHM layout 与 TP=8 共 **8 passed**） |
| [`test_qwen35_e2e_tp8.py`](../worker/tests/phase2/test_qwen35_e2e_tp8.py) | 8 卡真权重对 4 个 vLLM 贪心参考执行单序列（含 128-token 多 KV 页）、4 序列批处理与 `--max-prefill-chunk 24`；核对统计和调度不变量，并检查 17 个 Triton kernel 零重编译及 `/dev/shm` RAII 清理 | 通过工程断言；**token 分歧如实保留，见下表** |

### 3.1 多 prompt / 128-token / batched / chunked 贪心结果

参考：[`phase2_vllm_greedy_reference.json`](benchmarks/phase2_vllm_greedy_reference.json)；实测：[`phase2_greedy_parity_report.json`](benchmarks/phase2_greedy_parity_report.json)。`bucket=4` 批处理与 `max_prefill_chunk=24` 的 chunked 调度包含 3 个 prefill steps、128 个 decode steps 和 2 个 partial-prefill/decode steps。全局 `all_matched=false`、`batched_vs_single_exact_match=false`、`chunked_vs_unchunked_exact_match=false`，因此不再宣称全量 100% parity。

| Prompt | 输出长度 | Single 精确前缀 / 全序列匹配率 | Batched | Chunked |
| :--- | ---: | :--- | :--- | :--- |
| `france_capital` | 32 | 32 / 100% | 32 / 100% | 32 / 100% |
| `ieee754_subnormals` | 128 | 50 / 39.06% | 18 / 21.09% | 18 / 20.31% |
| `dijkstra_shortest_path` | 96 | 65 / 67.71% | 65 / 67.71% | 65 / 67.71% |
| `posix_shm_tlb_shootdown` | 64 | 64 / 100% | 64 / 100% | 29 / 50.00% |

运行时审计为 **`TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=17`**，SHM 清理也通过；这些结果与数值 token 分歧是不同维度。pytest 会重算并核对每项统计，但不会把分歧隐藏成“全量对齐成功”。

---

## 4. 单配置、Warm、`n=3` 交错六档对标

* **测试机器**：同一台 `8 × 32 GB sm_120`（无 NVLink，`can_device_access_peer == 0`，4 NUMA `[0,0,0,1,2,2,2,3]`），`Qwen/Qwen3.5-35B-A3B`，`TP=8`，BF16，贪心解码。
* **zstep 唯一配置**：`allreduce_blocks=8`、`num_kv_pages=65536`、`max_seqs=128`、`max_step_tokens=max_decoder_rows=4096`、`max_prefill_chunk=2048`、`max_pages_per_seq=256`。Round 0 前显式执行 **16 个 `512 -> 64` warmup prompts（并发 16）**，随后按 round-major 顺序交错六档，**每档 `n=3`**；没有按档切换配置。
* **vLLM 0.33.0**：32 个 warmup prompts 后同样 round-major 交错六档三轮。两边均为确定性伪随机非重复 prompt，`output_len=256`，`ignore_eos`；各档请求数依次为 `8/48/256/20/200/400`。
* **统计口径**：表中是三轮每档聚合值的 **`mean ± population std`**；每轮 `min..max`、Total 吞吐、mean/median/P99 TTFT、submit-TTFT、TPOT 与 ITL 全量值见 JSON。吞吐判定采用保守规则：相对差 `<3%`，或均值差 `<= 2 × sqrt(zstep_std² + vllm_std²)`，即标为 **统计持平/可比（持平）**。`n=3` 的“持平”只代表现有样本不能区分，不证明严格等价。
* **原始数据**：[`zstep n=3 JSON`](benchmarks/phase2_zstep_qwen35_35b_a3b_tp8.json)、[`vLLM n=3 JSON`](benchmarks/phase2_vllm_qwen35_35b_a3b_tp8.json) 与 [`vLLM 逐轮原始结果`](benchmarks/phase2_vllm_raw/)。zstep 运行时输出 **`TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=17`**。

| 档位 | zstep Output (`tok/s`) | vLLM Output (`tok/s`) | 均值差 / 判定 | Mean TPOT z / v (`ms`) | z Median TTFT / submit-TTFT (`ms`) | v Median TTFT (`ms`) | z P99 ITL (`ms`) | v P99 ITL (`ms`) |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| `in512_c1` | 261.72 ± 19.81 | 204.26 ± 10.53 | +28.14% / **zstep 更快** | 3.66 ± 0.27 / 4.64 ± 0.23 | 51.12 ± 10.68 / 2,999.35 ± 243.52 | 69.13 ± 12.82 | 4.09 ± 0.45 | 6.51 ± 1.24 |
| `in512_c16` | 1,938.14 ± 140.42 | 1,893.04 ± 14.01 | +2.38% / **持平** | 7.29 ± 0.52 / 7.34 ± 0.04 | 229.50 ± 27.42 / 2,378.89 ± 209.47 | 298.17 ± 2.29 | 7.27 ± 0.68 | 10.29 ± 0.52 |
| `in512_c128` | 4,917.46 ± 97.01 | 3,975.90 ± 18.06 | +23.68% / **zstep 更快** | 21.95 ± 0.51 / 27.29 ± 0.10 | 213.40 ± 1.22 / 3,518.72 ± 19.46 | 1,192.91 ± 15.91 | 225.68 ± 2.50 | 112.48 ± 0.18 |
| `in1024_c1` | 252.42 ± 7.16 | 206.05 ± 0.61 | +22.50% / **zstep 更快** | 3.64 ± 0.11 / 4.60 ± 0.02 | 83.95 ± 4.03 / 9,346.27 ± 316.18 | 65.63 ± 1.32 | 3.99 ± 0.11 | 6.66 ± 0.27 |
| `in1024_c32` | 2,152.54 ± 40.35 | 2,678.01 ± 241.68 | -19.62% / **vLLM 更快** | 12.86 ± 0.25 / 9.44 ± 0.62 | 223.14 ± 0.30 / 11,226.25 ± 255.84 | 395.95 ± 65.78 | 230.90 ± 0.32 | 66.08 ± 33.07 |
| `in1024_c128` | 3,025.25 ± 24.35 | 3,666.69 ± 484.51 | -17.49% / **持平（高方差）** | 35.38 ± 0.36 / 28.83 ± 3.80 | 224.81 ± 1.90 / 14,577.91 ± 165.59 | 1,126.07 ± 183.12 | 238.88 ± 2.50 | 117.22 ± 2.22 |

**TTFT 口径**：`ttft_ms` 从请求获得 concurrency permit 时开始计时，与 vLLM `limited_request_func` 的口径一致，因此主表可将 zstep `Median TTFT` 与 vLLM 同列比较。`submit_ttft_ms` 从请求进入本轮 workload 时开始，包含等待 permit 的排队时间；vLLM JSON 没有对应 submit-based 指标，**不能把 zstep submit-TTFT 与 vLLM permit-TTFT 直接相比**。这解释了高并发档中很低的 permit-TTFT 与很高的 submit-TTFT 可以同时成立。

**Warm 与 chunk 结果**：两个单流档的 Mean TPOT 均值分别为 **`3.658 ms`** 和 **`3.644 ms`**，已收敛到约 `3.65 ms`，但三轮范围分别为 `3.431..4.031` 与 `3.552..3.791 ms`，不能省略时钟漂移。统一 chunk 配置将 `in1024_c128` 的 P99 ITL 从旧单轮 **`1,303.57 ms`** 降至 **`238.88 ± 2.50 ms`**；这是显著改善，但仍高于 vLLM 的 `117.22 ± 2.22 ms`，且 `in512_c128` 也出现 `225.68 ms` 的 zstep 尾延迟，不能只报告吞吐优势。

### 4.1 四阶段渐进优化消融记录（历史，非当前统一配置）

以下仅保留工程演进历史；其中旧 vLLM 单轮基线和 zstep `step=8K` 数据不得与上面的 warm `n=3` 主表混用。完整原始数据归档于 [`phase2_qwen35_optimization_ablation.json`](benchmarks/phase2_qwen35_optimization_ablation.json)（同机亦记录了稠密模型 `Qwen/Qwen3.5-27B` `vLLM 0.33.0` `TP=8` 基线：`c=1` `85.5 tok/s`、`c=128` `1,060 tok/s`）：

| 演进阶段 | 核心改动 | `in512 c=1` (`tok/s` / `p99_itl`) | `in512 c=16` (`tok/s`) | `in512 c=128` (`tok/s`) | `in1024 c=1` (`tok/s`) | `in1024 c=32` (`tok/s`) | `in1024 c=128` (`tok/s`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (`vLLM 0.33.0`)** | `TP=8`, `bfloat16`, 降级 `NCCL SHM` | `184.72` / — | `1,671.63` | `3,271.56` | `181.96` | `1,834.83` | `2,939.12` |
| **Run 1（初始跑通）** | 全 40 层 CUDA Graph + NUMA SHM AllReduce (`blocks=16`, `step=4K`) | `163.82` / `12.53 ms` | `1,421.17` | `4,004.01` | — | — | — |
| **Run 2（消除 GC 与页表冗余）** | 热循环禁用周期 GC + `page_table` 仅拷活跃槽位 + `blocks=8` | `185.69` / `10.18 ms` | `1,552.52` | `3,916.61` | — | — | — |
| **Run 3（消除驱动锁 `ioctl`）** | 启动期静态缓存显存遥测，移除每步 16 次 `cuMemGetInfo_v2` 锁 (`step=4K`) | `224.18` / `5.00 ms` | `1,744.61` | **`4,638.96`** | `257.30` | `2,001.20` | `2,750.47` |
| **Run 4（Prefill Tile 细化 + 8K 批）** | `GatedDeltaNet` `BLOCK_V=4` / `Conv1d` `BLOCK_C=32`（4× SM 占用）+ `step=8K` | **`225.04` / `4.64 ms`** | **`1,768.73`** | `4,540.92` | **`261.72`** | **`2,165.22`** | **`3,024.65`** |

---

## 5. `H=2048, blocks=8` 全尺寸 AllReduce 实测

来源：[`phase2_allreduce_h2048_8gpu.json`](benchmarks/phase2_allreduce_h2048_8gpu.json)。8 卡均为 `sm_120`、BF16、无 P2P；每个 Graph 含 10 次调用，预热 300 次、计时 1,000 次。下表不省略中间劣势区间；`80 次` 列是按 Graph `p50` 推算的通信时间，不是独立端到端测量。

| `M` | 载荷 | 算法 | `zstep p50/p95/p99 (µs)` | `NCCL SHM p50/p95/p99 (µs)` | `zstep 80 次 (ms)` |
| ---: | ---: | :--- | :---: | :---: | ---: |
| 1 | 4 KiB | One-Shot | 26.26 / 29.02 / 30.00 | 26.06 / 26.98 / 28.64 | 2.10 |
| 2 | 8 KiB | One-Shot | 27.93 / 30.97 / 32.61 | 26.76 / 27.60 / 29.15 | 2.23 |
| 4 | 16 KiB | One-Shot | 31.34 / 34.12 / 35.49 | 28.43 / 29.31 / 30.73 | 2.51 |
| 8 | 32 KiB | Two-Shot | 48.79 / 51.60 / 54.46 | 37.18 / 40.06 / 42.73 | 3.90 |
| 16 | 64 KiB | Two-Shot | 51.74 / 54.77 / 63.94 | 37.81 / 41.73 / 43.44 | 4.14 |
| 32 | 128 KiB | Two-Shot | 59.76 / 62.52 / 63.93 | **29.63 / 30.58 / 32.14** | 4.78 |
| 64 | 256 KiB | Two-Shot | 82.78 / 85.74 / 90.75 | **46.67 / 69.07 / 102.75** | 6.62 |
| 128 | 512 KiB | Two-Shot | **75.00 / 77.90 / 82.30** | 84.70 / 250.38 / 325.89 | 6.00 |
| 512 | 2 MiB | Two-Shot | 220.45 / 222.91 / 235.74 | **157.34 / 159.65 / 160.83** | 17.64 |
| 4096 | 16 MiB | Two-Shot | 1,834.28 / 1,854.49 / 1,876.96 | **1,637.95 / 1,657.68 / 1,664.29** | 146.74 |
| 8192 | 32 MiB | Two-Shot | **3,818.86 / 3,861.77 / 3,888.71** | 4,738.17 / 4,788.09 / 4,806.43 | 305.51 |

* **Decode 小包**：`M=1..4`（`4..16 KiB`）One-Shot 的 Graph `p50` 为 **`26.26..31.34 µs`**，80 次为 **`2.10..2.51 ms`**，构成约 `3.5 ms` 单流 Decode TPOT 的主要通信预算；不能把它写成未经测量的 `30–37 µs / 2.5–3.0 ms`。Graph 模式下这三档 NCCL `p50` 略低，zstep 的优势不能夸成全尺寸获胜。
* **中间 Two-Shot**：`64 KiB..2 MiB` 中，除 `512 KiB` 外 NCCL `p50` 更低；尤其 `128 KiB` 为 `29.63 vs 59.76 µs`。这些负结果与 `512 KiB` 的 zstep 尾延迟优势同时保留。
* **NUMA First-Touch 大区间的边界**：本组 `16 MiB` 仍是 NCCL 更快；zstep 在 `32 MiB` 取得 `1.24×` `p50` 优势。另一组 `H=5120` 的 `40 MiB` 记录为 `4,274.36 vs 6,206.57 µs`（`1.45×`）。因此只声称已测大 payload 的 First-Touch 优势，不外推为“所有 `>=16 MiB` 均更快”。

`M=48, H=5120` 的 block 扫描还显示，增加 block 会退化而非单调加速：`blocks=8/16/32` 的 `p50` 分别为 **`57.96/66.35/76.44 µs`**，`p99` 为 `60.92/72.86/86.54 µs`。这与更多跨 NUMA `.sys` flag cacheline 流量和屏障开销一致；该微基准没有单独剖分 cacheline 事件，因此这是由测量支持的归因假设，而不是硬件计数器证明。统一端到端配置固定 `blocks=8`。

## 6. 控制面重叠的物理收益与性能边界

Stage 4 最新的 200 步、8 GPU 记录（[`phase1_mode_b_8gpu.json`](benchmarks/phase1_mode_b_8gpu.json)）中，Rust 下一拍准备累计 **`938 µs`**，以全部 200 步为分母得到 **`4.690 µs/step`**，其中 **199 步发生重叠**；GPU 平均步耗时 **`2,832.095 µs`**，准备成本相当于其 **`0.166%`**，实测 **`overlap_speedup = 1.000269×`**（约 **`1.0003×`**）。前一次同口径记录为 **`1,016 µs` 累计 / 200 步 = `5.08 µs/step`**（199 步重叠）、GPU **`3,066.94 µs`**、占比 **`0.166%`**、加速 **`1.000281×`**。`hidden_prepare_us` 只表示这些工作量落在 GPU 窗口内，不代表端到端获得大幅加速；发布/唤醒与 IPC 分位数也不能与准备成本混为一谈。

单流 Decode 将 40 层、80 次 SHM AllReduce 与跨卡 Argmax 收敛进一张 CUDA Graph，消除了逐层 Python 发射；但上表已经说明通信本身仍占主要预算。高并发长输入还受无 P2P 主机内存交换与调度尾延迟约束，chunked prefill 改善长尾并不等于优于 vLLM 的所有延迟指标。
