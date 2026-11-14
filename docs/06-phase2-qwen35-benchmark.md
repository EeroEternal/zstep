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
   * [`worker/zstep_worker/kernels_sm120/qwen35_ops.py`](../worker/zstep_worker/kernels_sm120/qwen35_ops.py)：零中心 `(1 + w)` RMSNorm、融合 64 维部分 RoPE `theta=1e7` + 16-token 分页 KV 写入 prologue、`Split-K=16` 分页门控 GQA 解码与变长 FlashAttention-2 Prefill、`BLOCK_C=32` 因果深度可分离卷积 + `BLOCK_V=4` 寄存器递推 `GatedDeltaNet`、无全局回读的寄存器内 Top-8 路由 + 共享内存 `moe_align_block_size` + Triton Tensor Core 分组专家 GEMM、以及确定性 `local_top2_bf16_kernel`（按 `(logit DESC, token_id ASC)` 打破平局）+ 256 KiB 锁免共享内存 `tp8_argmax_gather`（跨 8 卡归约全局 Top-2 `(logit, token_id)` 并通过零膨胀 wire 槽位回传 Rust 控制面）。运行时审计明确监视 **`WATCHED_KERNELS=17`** 个 Triton kernel，要求 **`TRITON_RECOMPILATIONS=0`**；CUDA C++ JIT 扩展不计入这 17 个 Triton kernel。
   * [`worker/zstep_worker/models/qwen35_moe.py`](../worker/zstep_worker/models/qwen35_moe.py)：按 Rank 仅从 `safetensors` mmap 切片读取本地 `1/8` 权重分片；预置 `dummy_slot` 与 `dummy_page` 隔离槽位，为解码桶 `(1, 2, 4, 8, 16, 32, 64, 128)` 一次性捕获完整 40 层单图 CUDA Graph（含 80 次跨卡 SHM AllReduce 与跨卡 Top-2 归约，解码期零 CPU 同步）。

---

## 3. 正确性与回归门禁

| 测试项 | 覆盖内容 | 结果 |
| :--- | :--- | :--- |
| `cargo test --workspace` + `clippy -D warnings` | `zs-wire` 动态容量属性断言、`zs-shm`、`zs-kv` Radix 前缀树与 RAII 回收、`zs-scheduler` 连续批处理重叠状态机、按 `(round, tier)` 独立 SplitMix64 种子与 TTFT 聚合 | **28 passed, 0 warnings** |
| [`test_qwen35_ops_sm120.py`](../worker/tests/phase2/test_qwen35_ops_sm120.py) | 零中心 RMSNorm + 零重编译断言、`GatedDeltaNet` 变长 Prefill 与单步递推逐位对齐、分页门控 GQA 对齐、**`530,000` 页（`8.6 GB`，偏移 `> 2^31` 元素）`int64` 寻址防溢出测试**、毒化初值下 256 专家 MoE 对齐、TP=8 全局 Top-2 精确平局打破与容量守卫 | **7 passed**（连同 `test_shm_layout_parity.py` 1 项与 `test_qwen35_e2e_tp8.py` 1 项共 **9 passed**） |
| [`test_qwen35_e2e_tp8.py`](../worker/tests/phase2/test_qwen35_e2e_tp8.py) | 8 卡真权重对 5 个 vLLM 贪心参考（含 128-token 无循环结构化算术表）执行单序列、5 序列批处理（`bucket=8`）与 `--max-prefill-chunk 24`；逐分歧步断言双端 Top-2 logit margin `<= 0.25`（`<= 2` 个 BF16 ULP），并核对 17 个 Triton kernel 零重编译及 `/dev/shm` RAII 清理 | **1 passed**；**token 分歧与 Top-2 margin 如实保留，见下表** |

### 3.1 5 用例 / 128-token 无循环 / batched / chunked 贪心与 Top-2 Margin 实测

参考：[`phase2_vllm_greedy_reference.json`](benchmarks/phase2_vllm_greedy_reference.json)（采样参数增加 `logprobs=5, return_tokens_as_token_ids=True` 采集每步 Top-2）；实测：[`phase2_greedy_parity_report.json`](benchmarks/phase2_greedy_parity_report.json)。5 序列批处理命中 decode `bucket=8`（unchunked：1 个 prefill step、127 个 decode steps，`max_active_sequences=5`）；`max_prefill_chunk=24` 的 chunked 调度包含 **4 个 prefill steps（4 个 chunked）、130 个 decode steps（其中 3 个含 partial prefill）**，`max_active_sequences=5`。全局标志仍诚实保留 `all_matched=false`、`batched_vs_single_exact_match=false`、`chunked_vs_unchunked_exact_match=false`。

| Prompt | 输出长度 | Single 精确前缀 / 全序列匹配率 | Batched (`bucket=8`) | Chunked (`chunk=24`) | 最小 `zstep` Top-2 Margin / 首个分歧点诊断 |
| :--- | ---: | :--- | :--- | :--- | :--- |
| `france_capital` | 32 | **32 / 100%** | **32 / 100%** | **32 / 100%** | `min_margin = 0.5625..0.6875`，全模式零分歧 |
| `ieee754_subnormals` | 128 | 18 / 14.84% | 18 / 20.31% | **128 / 100%** | step 18（前缀完全一致）：**vLLM `top1=16661 (' consists')` vs `top2=18091 (' defines')` 为精确 `0.0000` 0-ULP 平局**；`single` (`24.375 vs 24.25`) 与 `batched` (`24.25 vs 24.125`) 为 **`0.125`（1 个 BF16 ULP）** 翻转至 `18091`；`chunked` 为 `24.25 vs 24.25`（`0.0000` 平局取较小 ID `16661`，与 vLLM **128/128 完全一致**）。`single` 与 `batched` 在 step 50 分叉处双端 margin 同为 **`0.125`（1 ULP）** |
| `dijkstra_shortest_path` | 96 | **96 / 100%** | **96 / 100%** | **96 / 100%** | 确定性 `(logit DESC, token_id ASC)` 平局打破上线后，三模式全部 **96/96** 对齐 |
| `posix_shm_tlb_shootdown` | 64 | **64 / 100%** | 54 / 98.44% | **64 / 100%** | 71-token（3 chunks）输入下 `single` 与 `chunked` 均 **64/64**；`batched` 仅在 step 54 翻转 1 个 token（vLLM/`single` margin **`0.25` = 2 个 BF16 ULP**，`batched` 为 `24.875 vs 24.875` **`0.0000` 平局**选 `478 < 594`），随后 step 55..63 立即重收敛（63/64 token 一致） |
| `prime_squares_table` | 128 | **128 / 100%** | **128 / 100%** | **128 / 100%** | 80-token（4 chunks）输入、128-token 素数平方算术表（**113/113 个 16-gram 滑动窗口全唯一**，`min_margin = 0.625..2.125`），三模式全部 **128/128** 精确对齐 |

实测结论：
1. **`chunked`（`max_prefill_chunk=24`）在全部 5 个用例上合计 `448 / 448` tokens（100%）逐位匹配 vLLM**，且在 128-token 无循环长序列 `prime_squares_table`（跨 4 个 prefill chunk、覆盖 9 个 KV 页）上 `single` / `batched` / `chunked` 均无任何跨 chunk 状态漂移。
2. 所有已观测到的 `single` / `batched` 分歧点均发生在 logit `~25`（此时 BF16 指数段 `[16, 32)` 的 **1 ULP = $2^{4-7} = 0.125$**）处的 **`0..2` BF16 ULP 临界平局（`margin <= 0.25`）**，不存在大 margin 逻辑错位。运行时审计保持 **`TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=17`**，`/dev/shm` 退出零残留。

---

## 4. 无前缀缓存（`--no-enable-prefix-caching`）、每轮独立种子、Warm `n=3` 交错六档对标

* **测试机器**：同一台 `8 × 32 GB sm_120`（无 NVLink，`can_device_access_peer == 0`，4 NUMA `[0,0,0,1,2,2,2,3]`），`Qwen/Qwen3.5-35B-A3B`，`TP=8`，BF16，贪心解码。
* **消除 APC 前缀缓存污染**：此前默认 `vllm serve` 未关闭 Automatic Prefix Caching（APC），且每轮共用固定 `--seed 20261111`，导致 `openai-chat` 共享模板前缀及 Round 1/2 整段随机 prompt 命中缓存（例如 `in1024_c128` 曾出现 `3,666.69 ± 484.51 tok/s` 的跨轮剧烈爬升）。本版在 `vLLM 0.33.0` 服务端显式传入 **`--no-enable-prefix-caching`**，并在 **双端同时按 `(round, tier)` 派生独立种子**（`seed = 20261111 + round_idx * 1000 + tier_idx`，warmup 使用独立种子 `20261111 + 900000`；`zstep` 由 Rust 控制面 SplitMix64 生成词表 `[256, 32768)` 内的非重复随机 prompt）。关闭 APC 并隔离每轮种子后，`vLLM` 在 `in1024_c128` 的跨轮标准差从 `±484.51` 收敛 **27×** 至 **`±18.11 tok/s`**。
* **zstep 唯一配置**：`allreduce_blocks=8`、`num_kv_pages=65536`、`max_seqs=128`、`max_step_tokens=max_decoder_rows=4096`、`max_prefill_chunk=2048`、`max_pages_per_seq=256`。Round 0 前显式执行 **16 个 `512 -> 64` warmup prompts（并发 16）**，随后按 round-major 顺序交错六档，**每档 `n=3`**；全程单一进程、零配置切换。
* **vLLM 0.33.0**：`--no-enable-prefix-caching`，32 个 warmup prompts 后同样 round-major 交错六档三轮，`output_len=256`，`ignore_eos`；各档请求数依次为 `8/48/256/20/200/400`。
* **统计口径**：表中是三轮每档聚合值的 **`mean ± population std`**；每轮 `min..max`、Total 吞吐、mean/median/P99 TTFT、submit-TTFT、TPOT 与 ITL 全量值见 JSON。吞吐判定采用保守规则：相对差 `<3%`，或均值差 `<= 2 × sqrt(zstep_std² + vllm_std²)`，即标为 **统计持平**。
* **原始数据**：[`zstep n=3 JSON`](benchmarks/phase2_zstep_qwen35_35b_a3b_tp8.json)、[`vLLM n=3 JSON`](benchmarks/phase2_vllm_qwen35_35b_a3b_tp8.json) 与 [`vLLM 逐轮原始结果`](benchmarks/phase2_vllm_raw/)。zstep 运行时输出 **`TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=17`**。

| 档位 | zstep Output (`tok/s`) | vLLM Output (`tok/s`) | 均值差 / 判定 | Mean TPOT z / v (`ms`) | z Median TTFT / submit-TTFT (`ms`) | v Median TTFT (`ms`) | z P99 ITL (`ms`) | v P99 ITL (`ms`) |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| `in512_c1` | **261.53 ± 24.43** | 192.24 ± 10.22 | +36.05% / **zstep 更快** | 3.68 ± 0.34 / 4.66 ± 0.22 | 50.42 ± 11.10 / 3,016.61 ± 315.79 | 137.34 ± 17.39 | **4.12 ± 0.41** | 6.51 ± 0.93 |
| `in512_c16` | **2,037.92 ± 51.95** | 1,700.98 ± 12.13 | +19.81% / **zstep 更快** | 6.87 ± 0.24 / 7.62 ± 0.02 | 209.37 ± 1.82 / 2,252.67 ± 56.82 | 463.87 ± 17.91 | **6.98 ± 0.14** | 10.40 ± 0.28 |
| `in512_c128` | **5,021.65 ± 104.06** | 3,259.04 ± 5.72 | +54.08% / **zstep 更快** | 21.46 ± 0.50 / 32.66 ± 0.17 | 210.86 ± 1.69 / 3,484.02 ± 21.72 | 832.91 ± 181.55 | 223.50 ± 2.30 | **114.35 ± 1.68** |
| `in1024_c1` | **268.95 ± 1.47** | 197.31 ± 0.59 | +36.31% / **zstep 更快** | 3.42 ± 0.02 / 4.54 ± 0.02 | 79.91 ± 1.33 / 8,659.99 ± 46.85 | 132.48 ± 1.45 | **3.72 ± 0.03** | 6.01 ± 0.10 |
| `in1024_c32` | **2,174.52 ± 13.18** | 1,959.07 ± 29.76 | +11.00% / **zstep 更快** | 12.72 ± 0.07 / **11.29 ± 0.09** | 223.18 ± 1.23 / 11,030.49 ± 27.60 | 1,034.01 ± 33.94 | 230.63 ± 1.34 | **109.76 ± 1.33** |
| `in1024_c128` | **2,982.59 ± 54.67** | 2,565.79 ± 18.11 | +16.24% / **zstep 更快** | 35.97 ± 0.73 / 41.20 ± 0.24 | 226.59 ± 3.82 / 14,742.53 ± 211.99 | 711.86 ± 103.75 | 242.46 ± 3.61 | **113.56 ± 1.40** |

**TTFT 口径**：`ttft_ms` 从请求获得 concurrency permit 时开始计时，与 vLLM `limited_request_func` 的口径一致，因此主表可将 zstep `Median TTFT` 与 vLLM 同列比较。`submit_ttft_ms` 从请求进入本轮 workload 时开始，包含等待 permit 的排队时间；vLLM JSON 没有对应 submit-based 指标，**不能把 zstep submit-TTFT 与 vLLM permit-TTFT 直接相比**。

**吞吐与尾延迟的工程权衡（不可只报单维优势）**：
* 在无前缀缓存口径下，`zstep` 在全部六档的 Output 吞吐均超过 `2σ` 领先 `vLLM`（`+11.00%` 至 `+54.08%`），且两个单流档的 Mean TPOT 分别为 **`3.677 ms`**（三轮范围 `3.376..4.153 ms`）与 **`3.420 ms`**（三轮范围 `3.397..3.433 ms`）。
* 但 `zstep` 当前统一采用 `max_prefill_chunk = 2048`，在中高并发档（`in512_c128`、`in1024_c32`、`in1024_c128`）下当新请求准入并插入 2K-token prefill chunk 时，在途 decode 请求的 **`P99 ITL` 为 `223.50..242.46 ms`，约为 `vLLM`（`109.76..114.35 ms`）的 `2.0×..2.1×`**（尽管已比未分块时的 `1,303.57 ms` 降低 `5.4×`）。
* 此外，**`in1024_c32` 是六档中唯一 `zstep` 稳态解码 `Mean TPOT` 慢于 `vLLM` 的档位（`12.72 ± 0.07 ms` vs `11.29 ± 0.09 ms`）**；`zstep` 在该档的总吞吐领先（`+11.00%`）完全来自更快的分块 Prefill 周转（permit Median TTFT `223.18 ms` vs `1,034.01 ms`），而非 `M=32` 解码更快——其物理根因直接对应下节 `128 KiB` Two-Shot AllReduce 劣势。

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

> **口径与演进说明（`zstep#14`）**：下表归档自 Phase 2 历史单文件 SHM 基线（[`phase2_allreduce_h2048_8gpu.json`](benchmarks/phase2_allreduce_h2048_8gpu.json)，`Host A: 8 × RTX 5090` 4-NUMA，旧 `host_mapped_two_shot_fused_kernel`，`kOneShotMaxVecsLegacy = 1024` 即 `<= 16 KiB` One-Shot）。当前 `main` 生产路径已全面切换至每卡独立 NUMA 页 `shm_v5`（`DEFAULT_ONE_SHOT_MAX_VECS = 4096` 即 `<= 64 KiB` One-Shot、`vecs_per_cta > 256` 无状态自适应 `256 -> 512` 线程、保留 4 槽环形安全握手 `done-wait`），并在 `Qwen35MoeStepRunner` 中对大 `M` Eager Prefill（`numel > 320 * 5120`，即 `H=2048` 下 `M >= 1024`）自动切换至多 SM `NCCL_LOC`（`M <= 800` 及 100% CUDAGraph 解码桶走 `shm_v5`）。在 `8 × RTX 6000D` 2-NUMA 上的最新双端实测见 [`phase37_6000d_comm_audit.json`](benchmarks/phase37_6000d_comm_audit.json) 与 [`07-phase3-dsv41-6000d-benchmark.md` §8](07-phase3-dsv41-6000d-benchmark.md)：`H=2048` 下 `M=1..256` CUDAGraph `shm_v5` 全部领先 `NCCL_LOC`（`14.50..128.69 µs` vs `29.65..342.71 µs`，其中 `M=16` 因升入 `64 KiB` One-Shot + `512` 线程由 `30.61 µs` 降至 **`27.09 µs`**），Eager `M=512` `shm_v5` 仍更快（**`246.69 vs 285.54 µs`**），`M=1024..4096` 路由至 `NCCL_LOC`（**`472.19 / 878.75 / 1,248.16 / 1,639.49 µs`** vs `shm_v5` `478.69 / 943.04 / 1,407.55 / 1,874.43 µs`，4K Prefill 80 次调用每步省约 `18.8 ms`）。

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

* **单流 `c=1`（`4 KiB`）物理归因**：`M=1..4`（`4..16 KiB`）One-Shot 的 Graph `p50` 为 **`26.26..31.34 µs`**（`M=1` 时 `zstep 26.26 µs` vs `NCCL SHM 26.06 µs`，80 次通信均为约 **`2.10 ms`**，甚至 NCCL 略低 `0.2 µs`）。因此 `zstep` 在 `in512_c1` / `in1024_c1` 取得 `+36%` 吞吐领先（Mean TPOT `3.42..3.68 ms` vs `4.54..4.66 ms`）绝非因为 `4 KiB` AllReduce 更快，而是因为**通信之外的 40 层单图整捕获与手写 `GatedDeltaNet` / `PagedGQA` / `MoE` 算子路径将非通信计算压至约 `1.32..1.58 ms`（对比 `vLLM` 的约 `2.46..2.58 ms`）**。
* **中批 `c=32`（`128 KiB`）物理归因**：`64 KiB..2 MiB` 中，除 `512 KiB` 外 NCCL `p50` 更低；尤其 `M=32`（`128 KiB`）处 `zstep` `59.76 µs` 慢于 `NCCL SHM 29.63 µs`，每步 80 次调用引入约 **`+2.41 ms`** 通信劣势——这直接解释了为何上表六档中唯独 `in1024_c32` 出现 `zstep` 稳态 `Mean TPOT` 慢于 `vLLM`（`12.72 ms` vs `11.29 ms`）。这些负结果与 `512 KiB` 的 zstep 尾延迟优势同时保留。
* **NUMA First-Touch 大区间的边界**：本组 `16 MiB` 仍是 NCCL 更快；zstep 在 `32 MiB` 取得 `1.24×` `p50` 优势。另一组 `H=5120` 的 `40 MiB` 记录为 `4,274.36 vs 6,206.57 µs`（`1.45×`）。因此只声称已测大 payload 的 First-Touch 优势，不外推为“所有 `>=16 MiB` 均更快”。

`M=48, H=5120` 的 block 扫描还显示，增加 block 会退化而非单调加速：`blocks=8/16/32` 的 `p50` 分别为 **`57.96/66.35/76.44 µs`**，`p99` 为 `60.92/72.86/86.54 µs`。这与更多跨 NUMA `.sys` flag cacheline 流量和屏障开销一致；该微基准没有单独剖分 cacheline 事件，因此这是由测量支持的归因假设，而不是硬件计数器证明。统一端到端配置固定 `blocks=8`。

## 6. 控制面重叠的物理收益与性能边界

Stage 4 最新的 200 步、8 GPU 记录（[`phase1_mode_b_8gpu.json`](benchmarks/phase1_mode_b_8gpu.json)）中，Rust 下一拍准备累计 **`938 µs`**，以全部 200 步为分母得到 **`4.690 µs/step`**，其中 **199 步发生重叠**；GPU 平均步耗时 **`2,832.095 µs`**，准备成本相当于其 **`0.166%`**，实测 **`overlap_speedup = 1.000269×`**（约 **`1.0003×`**）。前一次同口径记录为 **`1,016 µs` 累计 / 200 步 = `5.08 µs/step`**（199 步重叠）、GPU **`3,066.94 µs`**、占比 **`0.166%`**、加速 **`1.000281×`**。`hidden_prepare_us` 只表示这些工作量落在 GPU 窗口内，不代表端到端获得大幅加速；发布/唤醒与 IPC 分位数也不能与准备成本混为一谈。

单流 Decode 将 40 层、80 次 SHM AllReduce 与跨卡 Argmax 收敛进一张 CUDA Graph，消除了逐层 Python 发射；但上表已经说明通信本身仍占主要预算。高并发长输入还受无 P2P 主机内存交换与调度尾延迟约束，chunked prefill 改善长尾并不等于优于 vLLM 的所有延迟指标。

---

## 7. Phase 3.9（`8 × RTX 6000D` Route A）：`shm_v5` Stage-2 尾融合与 SM120 `[V, K]` `GatedDeltaNet` 审计及 `TP=8` 实测

本节归档 Route A 在 `Host B`（`8 × NVIDIA RTX 6000D`，`sm_120`，`156 SMs`，`85,651 MiB` GDDR7，2-NUMA `[0,0,0,0,1,1,1,1]`，`20 × Mellanox CX-8` 桥 `ACSCtl=0x001d` 强制 CNS 主机内存路径）上针对通用跨卡通信尾融合（Part A1）与 `Qwen3.5-35B-A3B` 30 层线性注意力 `Qwen3_5MoeGatedDeltaNet`（Part A2）的单算子微基准、数值逐位对齐审计及 `TP=8` 真权重三套共 15 档 Warm `n=3` A/B 主结果（单算子延迟与逐位对齐生成脚本：[`bench_sm120_qwen35_route_a_ops.py`](../worker/tests/phase3/bench_sm120_qwen35_route_a_ops.py)；原始数据归档于 [`phase39_6000d_qwen35_route_a_audit.json`](benchmarks/phase39_6000d_qwen35_route_a_audit.json)，基线 `main` `893495c4` vs 候选 `98724b4c`）。

### 7.1 Part A1：`shm_v5` 非交错对端预取与 Stage-2 `AllReduce + Residual Add + RMSNorm` 尾融合

在未融合基线中，40 层解码器每步包含 80 次 `shm_v5` AllReduce，每次紧接一次独立的 `fused_add_zero_centered_rmsnorm` kernel 发射（每步额外 **80 次 kernel 发射 + 80 次 `reduced_buf` 全局显存写回再读回**）。Part A1 在 [`shm_allreduce.py`](../worker/zstep_worker/kernels_sm120/shm_allreduce.py) 中落地两项通用优化，且**完整保留 `shm_v5` 全部四项安全不变量**（入口 `my_done >= step - 4` 4 槽环形反压守卫 `done-wait`、出口 `peer_done` release 写回、`90 s` `clock64()` 超时熔断与 `abort_flag`、以及全模式统一的 `control[0..blocks)` 步进锁步，允许同一会话内任意交错调用 `all_reduce_sum` 与 `all_reduce_add_rmsnorm`）：

1. **对端主机映射页非交错批量预取**（`load_peer_vecs` + `accumulate_peer_vecs`）：先连续发出 7 路远端 `ld.global.cv.v4.u32` PCIe 读取装入寄存器数组，再统一做 FP32 累加，消除旧版 `ld_cv` 与 `__bfloat1622float2` 逐路交错导致的 PCIe 读流水线停顿；本地分片直接由 `input_vec`（VRAM/L2）读取，省去 1/8 主机内存回读。
2. **Stage-2 原地 `Residual Add + [Zero-Centered] RMSNorm` 尾融合**（`all_reduce_add_rmsnorm`，支持 `zero_centered=True/False`）：
   * **`M ∈ {1, 2, 4}`（`M < 8` 且 `8 % M == 0`，覆盖 `< 8` 全部解码桶；`M ∈ {3, 5, 6, 7}` Eager 非整除步自动回退 2-op）走 SM120 硬件线程块簇 `shm_v5_ar_add_rmsnorm_cluster8_kernel`（`__cluster_dims__(8, 1, 1)`）**：保持全部 8 个 CTA 并行承担 `1/8` PCIe 载荷读取，每个 CTA 在寄存器内完成本切片 `residual` BF16 舍入写回与局部平方和，仅通过 SM120 分布式共享内存（DSMEM `cluster.map_shared_rank`）交换 **1 个 `float`**（`4 B`）汇总整行 `rsqrtf(mean(x^2) + eps)`，零全局显存中间缓冲、零二次 kernel 发射。
   * **`8 <= M <= 512` 走行对齐 `shm_v5_ar_add_rmsnorm_kernel`（`fused_1op`）**：Two-Shot Stage-1 按 `(row, col_subslice)` 归约写入 `my_red`，Stage-2 按整行读取 `o_red` 并直接在寄存器内完成 `residual += reduced` 与行内 RMSNorm（对 `H=4096/5120` 配合 `512` 线程双行并行归约）。
   * **逐位精度**：跨 `H ∈ {2048, 5120}` 与 `M ∈ {1..512}`，`residual` 与未融合两算子路径达到 **`0 ULP`（`torch.equal == True`）**，`normed_out` 最大偏差 **`<= 1 BF16 ULP`**。

| 宽度 / 形状 | `base_2op` (`p50 µs`) | `batchload_2op` (`p50 µs`) | `fused_1op` (`p50 µs`) | `fused_cluster8` (`p50 µs`) | 投产融合路径加速比（80 次/步节省） |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `H=2048, M=1` (CUDAGraph) | `15.46` | `9.98` | `11.42` | **`10.20`** | **`1.52×`**（省 **`0.42 ms/step`**） |
| `H=2048, M=2` (CUDAGraph) | `16.14` | `11.22` | `12.65` | **`11.22`** | **`1.44×`**（省 **`0.39 ms/step`**） |
| `H=2048, M=4` (CUDAGraph) | `16.69` | `13.27` | `14.70` | **`13.47`** | **`1.24×`**（省 **`0.26 ms/step`**） |
| `H=2048, M=8` (CUDAGraph) | `18.62` | `18.18` | **`17.98`** | — | `1.04×` |
| `H=2048, M=32` (`128 KiB` Two-Shot) | `35.94` | `32.11` | **`26.58`** | — | **`1.35×`**（省 **`0.75 ms/step`**，直接修复 §5 `M=32` 瓶颈） |
| `H=2048, M=64` (CUDAGraph) | `44.52` | `44.09` | **`41.32`** | — | **`1.08×`**（省 `0.26 ms/step`） |
| `H=5120, M=1` (CUDAGraph) | `17.15` | `12.46` | `20.79` | **`11.57`** | **`1.48×`**（80 次省 **`0.45 ms`**） |
| `H=5120, M=2` (CUDAGraph) | `17.98` | `15.32` | `23.14` | **`14.43`** | **`1.25×`** |
| `H=5120, M=8` (`80 KiB` Two-Shot) | `34.98` | `29.78` | **`21.40`** | — | **`1.63×`**（80 次省 **`1.09 ms`**） |
| `H=5120, M=16` (CUDAGraph) | `39.08` | `36.34` | **`30.61`** | — | **`1.28×`** |
| `H=5120, M=64` (CUDAGraph) | `95.19` | `93.59` | **`85.59`** | — | **`1.11×`**（80 次省 **`0.77 ms`**） |

### 7.2 Part A2：`Qwen3_5MoeGatedDeltaNet` SM120 `[V, K]` 状态转置、自适应三段解码与淘汰项实证

`Qwen3.5-35B-A3B` 在 `TP=8` 下每卡含 30 层 `GatedDeltaNet`（`H_k=2, H_v=4, D_k=128, D_v=128`，每卡因果卷积宽度 `conv_dim=1024, K_conv=4`，`qkvzba` 列 `[1024:1536]` 为旁路 `Z` 门控）。原实现将 `ssm_states` 存为 `[num_slots, H_v=4, K=128, V=128]`，且每层分 3 个 Triton kernel 发射（30 层共 90 次 launch）。Part A2 在 [`qwen35_ops.py`](../worker/zstep_worker/kernels_sm120/qwen35_ops.py) 与 [`qwen35_moe.py`](../worker/zstep_worker/models/qwen35_moe.py) 中完成以下重构：

1. **`ssm_states` 全局转置为 `[num_slots, H_v=4, V=128, K=128]`（`K` 维连续）**：使每个 `v` 通道的 128 个 `float` 状态沿 `K` 维满足 `32 threads × float4 = 512 B` 完美合并访存，同时通过 warp 内 `__shfl_xor_sync` 在寄存器中无共享内存地完成 $\sum_k S_{v,k} k_k$ 与 $\sum_k S'_{v,k} q_k$。
2. **Decode 自适应三段无漂移路由**（`ZSTEP_QWEN_GDN_DECODE_MODE=auto`）：
   * **`batch <= 8` 走 `split_3op_32tile_vk`**：在 `156-SM` 低批解码下铺开 `B × 32` 个 `(v_head, v_tile=4)` CTA 跑满 SM 占用率（`B=1..8` 30 层 `136.77..149.09 µs`）。
   * **`9 <= batch <= 32` 走 `split_2op_vhead512_vk`**：将 `v`-head 递推与 `RMSNorm(128) * SiLU(z)` 尾声融合为 2 次 launch（`B=16/32` 30 层 `298.50 / 493.09 µs`）。
   * **`batch > 32` 走单次发射 `gdn_decode_fused_cluster2_512_vk_kernel`（`__cluster_dims__(2, 1, 1)`，30 层仅 30 次 launch）**：以 `dim3(2 * 2, num_rows, 1)` 为每行发射 **2 个 2-CTA 硬件 Cluster（共 4 个 512 线程 CTA）**；每个 2-CTA Cluster 协同负责 **1 个 `k_head` 与 2 个 `v_head`**（每 CTA 承担 `256` 维因果 `Conv1d + SiLU`，即 `128` 维 `Q` 或 `K` + `128` 维 `V`，Cluster 合计 `512` 维），经 DSMEM `cluster.map_shared_rank` 零全局显存广播归一化 `Q/K` 后原地更新 `[V, K]` 状态并融合输出 `RMSNorm * SiLU(z)`。在 `B=64 / 128` 下将 30 层耗时由 Triton `907.81 / 1,710.66 µs` 压至 **`794.02 / 1,581.60 µs`（`+14.3% / +8.2%`）**。
   * **跨桶 `0 ULP` 严格等价不变量**：`split_3op`、`split_2op`、`cluster2` 与 `auto` 四种模式在 `conv_state`、`ssm_state`、`gated_out` 上实现 **`0 ULP`（`torch.equal == True`）** 逐位一致，杜绝不同解码桶切换引发数值漂移；相对旧 Triton `[K, V]` 路径 `conv_state` `0 ULP`、`ssm_state` `max_abs <= 4.77e-7`、`gated_out` `<= 1..2 BF16 ULP`。
3. **变长 Prefill `cuda_vk_recurrent`**：4 阶段 `[V, K]` 合并访存 CUDA 变长递推在全部实测形状上较旧 Triton 3-op 提速 **`1.69×..3.42×`**（30 层耗时：`1x24` `1,255.07 -> 657.31 µs` **`1.91×`**；`1x512` `19,209.15 -> 10,827.42 µs` **`1.77×`**；`8x512` `26,738.50 -> 11,392.80 µs` **`2.35×`**；`16x256` `23,702.56 -> 6,932.35 µs` **`3.42×`**；`gated_out` 最大绝对偏差 `<= 1.22e-4`，其中 `varlen_3seqs_mixed_pos` `max_ulp = 1`、`single_512_pos0` 近零元素处 `max_abs = 3.05e-5` / `max_ulp = 11` 且 `>1 ULP` 占比仅 `2.29e-5`）。
4. **淘汰实验算子负向实证（不合入生产路径）**：
   * `flashinfer.experimental.gdn_sm120_cute`：存在约 **`14.4..14.6 ms` / 30 层的固定主机端 staging 与发射开销**，在混合步 chunked prefill `1x24`（`14,401.76 vs 657.31 µs`，**慢 `21.9×`**）及 `1x64` / `1x512` / `8x512` / `16x256` 上全面慢于 `cuda_vk_recurrent`；虽在单序列极长 `1x1024` / `1x2048` 裸算子耗时较低（`14.52 / 20.62 ms` vs `21.49 / 42.83 ms`），但其要求 `[K, V]` 状态布局（与 `[V, K]` 解码池不兼容，跨 30 层转置将抵消增益）且 BF16 WY 分块引入更高数值误差（`gated_out max_abs = 3.91e-3` vs `cuda_vk_recurrent` `1.22e-4`），故整条拒绝。
   * `flashinfer.experimental.b12x.sequence.gdn_decode`：公开实验分支符号缺失（`ImportError`）且内部硬编码头数不支持 `TP=8` 下的 `(H_k=2, H_v=4)`，淘汰。

### 7.3 `8 × RTX 6000D` `TP=8` 贪心对齐与零重编译门禁

在 `8 × RTX 6000D`（`vllm/vllm-openai:v0.31.0`，CUDA `13.0`）上对 [`phase2_vllm_greedy_reference.json`](benchmarks/phase2_vllm_greedy_reference.json) 运行单序列、`bucket=8` 批处理与 `--max-prefill-chunk 24` 审计：
* 基线 `893495c4` 与候选 `98724b4c` 均达成 **`TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=20`**（零运行期重编译，退出后 `/dev/shm` 零残留）。
* 结构化无循环算术表 **`prime_squares_table`（128 tokens）** 与 **`france_capital`（32 tokens）** 在 `single` / `batched` / `chunked=24` 三模式下均 **`100%`（`128/128`、`32/32`）逐位精确匹配**；全部 5 条用例的 `single` 精确前缀长度在优化前后保持逐项一致（`32, 18, 96, 64, 128`），且候选版在 `ieee754_subnormals`（`chunked` `128/128`）与 `dijkstra_shortest_path`（`batched` `96/96`）上取得完整匹配（相对 `vLLM` 参考首处分歧点 `zstep` `top1-top2` 裕度全部为 `0.125` = `1` 个 BF16 ULP；跨模式 `single/batched/chunked` 首处分歧最大裕度由基线 `0.500` 收敛至 `0.375` = `3` 个 BF16 ULP）。

### 7.4 `8 × RTX 6000D` `TP=8` 三套共 15 档 Warm `n=3` 端到端 A/B 主结果

三套测试全部在同一台 `8 × RTX 6000D` 上以独立种子、无前缀缓存、16 warmup prompts、round-major 交错 **`n=3`** 采集（Suite 1/2：`num_kv_pages=65536, max_pages_per_seq=256`；Suite 3 `in4096->out128`：`num_kv_pages=65536, max_pages_per_seq=512`）。**Route A（`A1+A2`）在全部 `15 / 15` 档位上同时实现 Output 吞吐提升（`+3.53% .. +23.80%`）、Mean TPOT 下降、Median TTFT 缩短与 P99 ITL 降低**：

#### （1）Suite 1：`in512 -> out128` 解码并发扫描（`c=1..64`，含 `A1-only` 独立消融，Warm `n=3`）

| 档位 | 基线 `893495c` (`tok/s`) | 仅 Part A1 (`tok/s` / 增益) | **完整 Route A (`A1+A2`, `tok/s` / 增益)** | Mean TPOT 基线 -> A1 -> **Full** (`ms`) | Med TTFT 基线 -> **Full** (`ms`) | P99 ITL 基线 -> **Full** (`ms`) |
| :--- | ---: | ---: | ---: | :---: | :---: | :---: |
| `c1` | `248.07 ± 2.82` | `284.25 ± 0.78` (`+14.58%`) | **`307.11 ± 11.03` (`+23.80%`)** | `3.69` -> `3.17` -> **`2.95`** | `47.83` -> **`43.09`** | `3.84` -> **`3.35`** |
| `c2` | `451.49 ± 3.42` | `497.66 ± 1.00` (`+10.23%`) | **`535.96 ± 22.90` (`+18.71%`)** | `3.91` -> `3.51` -> **`3.26`** | `69.77` -> **`64.01`** | `4.10` -> **`3.65`** |
| `c4` | `758.58 ± 7.30` | `805.78 ± 2.18` (`+6.22%`) | **`844.07 ± 26.63` (`+11.27%`)** | `4.44` -> `4.13` -> **`3.96`** | `111.23` -> **`103.72`** | `4.69` -> **`4.33`** |
| `c8` | `1,257.85 ± 11.49` | `1,270.24 ± 2.89` (`+0.99%`) | **`1,318.37 ± 14.91` (`+4.81%`)** | `4.87` -> `4.82` -> **`4.68`** | `196.04` -> **`183.20`** | `5.19` -> **`5.10`** |
| `c16` | `1,690.35 ± 14.64` | `1,707.63 ± 2.49` (`+1.02%`) | **`1,781.42 ± 6.41` (`+5.39%`)** | `8.00` -> `7.91` -> **`7.62`** | `156.26` -> **`144.59`** | `137.52` -> **`125.72`** |
| `c32` | `2,225.78 ± 9.29` | `2,342.80 ± 5.19` (`+5.26%`) | **`2,533.34 ± 7.70` (`+13.82%`)** | `12.33` -> `11.62` -> **`10.80`** | `168.32` -> **`149.71`** | `165.97` -> **`145.06`** |
| `c64` | `2,743.41 ± 6.47` | `2,769.50 ± 8.54` (`+0.95%`) | **`3,033.29 ± 16.27` (`+10.57%`)** | `18.65` -> `18.45` -> **`16.97`** | `340.07` -> **`293.06`** | `170.31` -> **`147.98`** |

* **归因闭环**：`c1` 单流解码吞吐从 `248.07` 跃升 **`+23.80%` 至 `307.11 tok/s`**（稳态 `Mean TPOT` 首次突破 `3 ms` 降至 **`2.95 ms`**），其中 Part A1 独立贡献 **`+14.58%`**（`-0.52 ms/step`），Part A2 在此基础上再叠加 **`+9.22 pct`**（`-0.22 ms/step` + Prefill 提速）。在 `c32` 档位，Part A1 的行对齐 Stage-2 融合单独带来 **`+5.26%`**（TPOT `-0.71 ms`），叠加 Part A2 后总增益达 **`+13.82%`**（TPOT `12.33 -> 10.80 ms`）；在 `c64` 档位，增益主要由 Part A2 的 `cluster2_512_vk` 单次发射解码与 `cuda_vk_recurrent` Prefill 驱动（`+0.95%` -> **`+10.57%`**，Med TTFT 缩短 **`-47.01 ms`**）。

#### （2）Suite 2（标准六档交错，`output=256`）与 Suite 3（`in4096 -> out128` 长输入 Prefill，Warm `n=3`）

| 套件 / 档位 | 基线 Output (`tok/s`) | **Route A Output (`tok/s`)** | 相对增益 | Total 吞吐 基线 -> **Route A** (`tok/s`) | Mean TPOT 基线 -> **Route A** (`ms`) | Med TTFT 基线 -> **Route A** (`ms`) | P99 ITL 基线 -> **Route A** (`ms`) |
| :--- | ---: | ---: | ---: | :---: | :---: | :---: | :---: |
| Suite 2 `in512_c1` | `261.82 ± 0.32` | **`307.14 ± 10.01`** | **`+17.31%`** | `785.45` -> **`921.43`** | `3.65` -> **`3.10`** | `47.34` -> **`43.28`** | `3.81` -> **`3.32`** |
| Suite 2 `in512_c16` | `1,978.11 ± 3.54` | **`2,048.01 ± 38.53`** | **`+3.53%`** | `5,934.31` -> **`6,144.04`** | `7.29` -> **`7.07`** | `155.62` -> **`144.90`** | `7.06` -> **`6.92`** |
| Suite 2 `in512_c128` | `4,447.25 ± 4.78` | **`4,791.36 ± 18.52`** | **`+7.74%`** | `13,341.75` -> **`14,374.07`** | `24.44` -> **`22.89`** | `348.96` -> **`300.43`** | `174.36` -> **`150.07`** |
| Suite 2 `in1024_c1` | `246.33 ± 0.97` | **`288.43 ± 9.29`** | **`+17.09%`** | `1,231.65` -> **`1,442.16`** | `3.74` -> **`3.17`** | `87.02` -> **`79.32`** | `3.86` -> **`3.38`** |
| Suite 2 `in1024_c32` | `1,961.16 ± 6.22` | **`2,166.66 ± 5.43`** | **`+10.48%`** | `9,805.82` -> **`10,833.28`** | `14.38` -> **`12.98`** | `179.39` -> **`159.91`** | `179.24` -> **`159.86`** |
| Suite 2 `in1024_c128` | `2,675.34 ± 0.30` | **`2,956.63 ± 4.68`** | **`+10.51%`** | `13,376.69` -> **`14,783.15`** | `40.28` -> **`36.45`** | `382.16` -> **`328.73`** | `194.18` -> **`164.96`** |
| Suite 3 `in4096_c1` | `148.64 ± 0.33` | **`164.85 ± 0.37`** | **`+10.91%`** | `4,905.04` -> **`5,440.10`** | `4.15` -> **`3.75`** | `333.77` -> **`300.52`** | `4.33` -> **`3.92`** |
| Suite 3 `in4096_c16` | `369.75 ± 0.72` | **`412.22 ± 0.77`** | **`+11.49%`** | `12,201.71` -> **`13,603.36`** | `30.65` -> **`27.68`** | `646.54` -> **`578.35`** | `216.62` -> **`193.23`** |

---

## 8. Phase 3.10（PR `#21` + `#16` + `#19` 合入审计）：设备驻留 Token 跨步流水重叠与并行 `moe_align`（`8 × RTX 6000D`, Warm `n=3`）

在 Phase 3.9 Route A（`#20`）基础上，`main` 依次合入三项控制面/数据面协同变更，并在同台 `8 × NVIDIA RTX 6000D` 上完成完整 Warm `n=3` 消融与 15 档全量归档（原始 JSON：[`phase310_6000d_qwen35_overlap_audit.json`](benchmarks/phase310_6000d_qwen35_overlap_audit.json)）：

1. **`GatedDeltaNet` Triton 回退核 `[V, K]` 状态布局对齐（PR `#21`, `44859cf`）**：将 `ZSTEP_QWEN_ENABLE_FUSED_GDN=0` 的 Triton Prefill / Decode 回退核统一改为 `[num_slots, H_v, V, K]` 状态步幅，并在 [`test_qwen35_ops_sm120.py`](../worker/tests/phase2/test_qwen35_ops_sm120.py) 覆盖 `(CUDA, Triton) × (Prefill, Decode)` 全部 4 种组合混用测试（状态最大偏差 `< 1e-6`，输出 `<= 1 BF16 ULP`）。
2. **`RuntimeLoop` 单次页锁主机拼包与零拷贝 SHM 视图（PR `#16`, `91b3991`）**：通过 `np.frombuffer` 零拷贝读取共享内存切片，并将每步解码元数据搬运合并进单块双缓冲页锁主机缓冲 `dec_io_cpu`（`_dec_io_np[parity]`）做单次异步 H2D 拷贝；同时对 `runtime_loop.py` 中的算子模块改为惰性导入，使无 `torch` 的纯 CPU 控制面单测（`test_shm_layout_parity.py` `9/9`）在标准 CI 容器中直接运行。
3. **设备驻留 Token 跨步流水重叠 + 8-Warp 并行前缀和 `moe_align_block_size_256`（PR `#19`, `cf531bc`）**：
   * **跨步设备内 Token 直传（`STEP_FLAG_DEVICE_TOKENS`）**：当连续纯解码步满足 `decode_continues_for_all` 时，Rust 调度器无需等待上一拍采样结果经 PCIe D2H 回传再写回下一拍 SHM `token_ids`，而是提前在双缓冲另一槽位预发布下一拍 `StepPlan`；GPU 侧由捕获进 CUDA Graph 头部的 `decode_select_tokens_kernel`（`launch_decode_select_tokens`）与尾声的 `decode_publish_tokens_kernel`（`launch_decode_publish_tokens`）直接按 `req_slot_ids` 从设备驻留 `last_token_gpu` 读取输入 Token，并将本拍全局 Top-1 结果同时写回 `last_token_gpu` 与主机页锁输出缓冲 `dec_io_cpu`，实现真正的 `in_flight_depth = 1` 零气泡跨步流水重叠（可通过 `ZSTEP_DECODE_DEVICE_TOKENS=0` 独立关停做严格消融）。
   * **并行 `moe_align_block_size_256_kernel`**：将 256 专家直方图与块偏移前缀和由单 Warp 串行扫描升级为全部 8 个 Warp 协同归约扫描，且全部分支满足 `tp8_argmax_shm_all_ranks_populate_outputs` 非零 Rank 输出写回与越界边界保护。

### 8.1 贪心逐位对齐与零重编译门禁（`OVERLAP=1`）

在 `ZSTEP_DECODE_DEVICE_TOKENS=1` 默认开启下对照 [`phase2_vllm_greedy_reference.json`](benchmarks/phase2_vllm_greedy_reference.json)：
* 运行时审计 **`TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=20`**，退出后 `/dev/shm` 零残留；
* 全部 5 条用例在 `single`、`batched`（`bucket=8`）与 `chunked`（`--max-prefill-chunk 24`）三种模式下的生成序列与 Phase 3.9 Route A（§7.3）**100% 逐 token 完全一致**（`france_capital` `32/32`、`prime_squares_table` `128/128` 三模式精确匹配；跨模式最大首处分歧裕度仍为 `dijkstra_shortest_path` 第 65 token 处的 `0.375` = `3` 个 BF16 ULP）。

### 8.2 Suite 1：`ZSTEP_DECODE_DEVICE_TOKENS=0` vs `=1` 受控消融（`in512 -> out128`, Warm `n=3`）

以下两组测试在同一二进制（`cac7e00`）、同一调度配置（[`README.md` §6](benchmarks/README.md) 标准默认参数：`num_kv_pages=65536`、`max_pages_per_seq=256`、默认单序列 Prefill 穿插即未显式传 `--max-mixed-step-tokens`）下仅切换 `ZSTEP_DECODE_DEVICE_TOKENS=0/1`：

| 档位 | `OVERLAP=0` (`tok/s`) | **`OVERLAP=1` (`tok/s`)** | `OVERLAP=1` vs `OVERLAP=0` | Mean TPOT `OVL=0` -> **`OVL=1`** (`ms`) | Med TTFT `OVL=0` -> **`OVL=1`** (`ms`) | P99 ITL `OVL=0` -> **`OVL=1`** (`ms`) |
| :--- | ---: | ---: | :--- | :---: | :---: | :---: |
| `c1` | `285.99 ± 0.59` | **`335.62 ± 0.11`** | **`+17.35%`** | `3.18` -> **`2.66`** (`-16.4%`) | `43.61` -> **`42.82`** | `3.38` -> **`2.81`** |
| `c2` | `500.55 ± 4.16` | **`584.21 ± 0.83`** | **`+16.71%`** | `3.52` -> **`2.95`** (`-16.2%`) | `64.76` -> **`63.71`** | `3.70` -> **`3.08`** |
| `c4` | `836.46 ± 8.00` | **`945.81 ± 3.52`** | **`+13.07%`** | `4.00` -> **`3.45`** (`-13.8%`) | `104.37` -> **`103.51`** | `4.24` -> **`3.64`** |
| `c8` | `1,315.09 ± 15.85` | **`1,455.36 ± 1.89`** | **`+10.67%`** | `4.68` -> **`4.10`** (`-12.4%`) | `184.13` -> **`182.97`** | `4.99` -> **`4.35`** |
| `c16` | `1,812.25 ± 9.20` | **`1,944.68 ± 11.71`** | **`+7.31%`** | `7.26` -> **`6.66`** (`-8.3%`) | `184.00` -> **`183.09`** | `6.53` -> **`5.98`** |
| `c32` | `2,577.80 ± 0.88` | **`2,715.08 ± 11.04`** | **`+5.33%`** | `10.49` -> **`9.86`** (`-6.0%`) | `183.87` -> **`183.17`** | `190.79` -> **`189.96`** |
| `c64` | `3,083.74 ± 4.12` | **`3,182.99 ± 6.75`** | **`+3.22%`** | `16.76` -> **`16.12`** (`-3.8%`) | `183.49` -> **`183.34`** | `193.46` -> **`193.09`** |

* **物理归因**：开启设备驻留 Token 直传后，每拍消除了 `GPU -> SHM -> Rust eventfd -> SHM -> GPU H2D` 串行控制面空泡（约 **`0.52..0.63 ms/step`**），将单流 `c1` 稳态 `Mean TPOT` 从 `3.18 ms` 进一步压低至 **`2.66 ms`**（输出吞吐达 **`335.62 tok/s`**），且随单拍 GPU 计算时长从 `2.66 ms`（`c1`）增长至 `16.12 ms`（`c64`），固定气泡消除对应的相对吞吐增幅严格符合 Amdahl 定律单调收敛（`+17.35% -> +3.22%`），同时跨轮抖动（`std`）大幅收敛。

### 8.3 Suite 2（标准六档交错）与 Suite 3（`in4096 -> out128`）默认 `OVERLAP=1` 主结果（Warm `n=3`）

> **调度口径注记**：本节严格采用 [`README.md` §6](benchmarks/README.md) 文档化默认 CLI（不额外传入 `--max-mixed-step-tokens 3072`），当存在活跃解码序列时每次混合步仅穿插 1 条新 Prefill，因此 `c16` 稳态尾延迟大幅下降（Suite 1 `c16` `P99 ITL = 5.98 ms`、Suite 2 `in512_c16` `P99 ITL = 6.10 ms`），`c128` 高并发排队首字延迟亦显著降低（`in512_c128` `Med TTFT = 183.52 ms`、`in1024_c128` `Med TTFT = 197.61 ms`）；而全部 **15 / 15** 档位的 Output 吞吐与 `Mean TPOT` 均全面超越 §7.4 Phase 3.9 主表：

| 套件 / 档位 | §7.4 Phase 3.9 (`tok/s`) | **当前 `main` (`OVERLAP=1`, `tok/s`)** | 相对 Phase 3.9 | Total 吞吐 (`tok/s`) | Mean TPOT (`ms`) | Med TTFT (`ms`) | P99 ITL (`ms`) |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| Suite 2 `in512_c1` | `307.14 ± 10.01` | **`353.98 ± 0.24`** | **`+15.25%`** | **`1,061.93`** | **`2.67`** | **`43.13`** | **`2.74`** |
| Suite 2 `in512_c16` | `2,048.01 ± 38.53` | **`2,267.10 ± 16.18`** | **`+10.70%`** | **`6,801.30`** | **`6.24`** | `183.38` | **`6.10`** |
| Suite 2 `in512_c128` | `4,791.36 ± 18.52` | **`5,039.13 ± 6.31`** | **`+5.17%`** | **`15,117.38`** | **`21.82`** | **`183.52`** | `198.82` |
| Suite 2 `in1024_c1` | `288.43 ± 9.29` | **`328.65 ± 0.15`** | **`+13.94%`** | **`1,643.23`** | **`2.74`** | **`79.29`** | **`2.82`** |
| Suite 2 `in1024_c32` | `2,166.66 ± 5.43` | **`2,367.78 ± 1.63`** | **`+9.28%`** | **`11,838.89`** | **`11.82`** | `197.51` | `205.37` |
| Suite 2 `in1024_c128` | `2,956.63 ± 4.68` | **`3,128.09 ± 4.86`** | **`+5.80%`** | **`15,640.43`** | **`34.65`** | **`197.61`** | `214.13` |
| Suite 3 `in4096_c1` | `164.85 ± 0.37` | **`181.72 ± 0.02`** | **`+10.23%`** | **`5,996.72`** | **`3.18`** | **`300.12`** | **`3.38`** |
| Suite 3 `in4096_c16` | `412.22 ± 0.77` | **`453.58 ± 0.67`** | **`+10.03%`** | **`14,968.26`** | **`25.34`** | **`464.48`** | `237.33` |

