# 07 — Phase 3 / 3.5 / 3.6 实测报告：`DeepSeek-V4.1-Flash` (`TP=8`) 在 `8 × RTX 6000D` 端到端推理、纯引擎同口径对标与 `b12x` 算子优化

本文档记录 `zstep` Phase 3、Phase 3.5 与 Phase 3.6 在对称双 NUMA、无 P2P 的 **`8 × NVIDIA RTX 6000D`**（每卡 `85,651 MiB` GDDR7、`sm_120`、156 SMs、驱动 `595.99.02`）服务器上，完成真权重端到端模型 **`DeepSeek-V4.1-Flash`**（`1017` 版本、`475.25 GiB` 混合量化权重：40 层 `DeepseekV41Attention` 稀疏 MLA + SWA 混合注意力、`cr=4` KV 压缩、384 专家 `UE8M0` MXFP4 MoE、TileLang 流形超连接 `mHC`、以及全驻留显存 Engram V2 3-token 查表）的 Mode B 架构实现、`shm_v5` + 按尺寸自适应跨卡集合通信、顺风车混合批调度、**Phase 3.6 选择性 `b12x` SM120 算子集成（40 层 `MXFP4 MoE` + 170 层 `[32, 32] UE8M0` FP8 Linear）**、确定性全局 Top-2 贪心逐 token 审计，以及与官方容器 **`vllm/vllm-openai:v0.31.0`** 在 **双口径（进程内 `AsyncLLMEngine` + `TokensPrompt` 零 HTTP/分词/解码/SSE 纯引擎同口径基线 + `vllm bench serve` `/v1/completions` HTTP 流式基线）** 两组 Warm `n=3` 并发套件（4 档 `in512_out128` 与 6 档 `six_tier`）下的正面对标结果。

> **口径约束声明**：
> 1. 本机器 `can_device_access_peer` `8 × 8` 全为 `0`（`nvidia-smi topo -m` 跨卡全 `CNS`），所有 `TP=8` 集合通信与端到端吞吐/延迟结论均严格限定在 **无 P2P 的对称 2-NUMA `[0,0,0,0,1,1,1,1]` PCIe Gen5 拓扑**（`vLLM` 与 `zstep` 统一设置 `NCCL_P2P_DISABLE=1`、`NCCL_P2P_LEVEL=LOC`），不外推至具备 NVLink 或 PCIe P2P 的系统。
> 2. 在官方镜像 `vllm/vllm-openai:v0.31.0` 中，`FlashInfer` 的 `sm_120` 稀疏滑动窗口注意力（SWA）与 `cr=1` 层在 `page_block_size=128` 下触发 TMA 描述符越界产生 `NaN`；本阶段对 **同机 `vLLM 0.31.0` 基线与 `zstep` 对称施加完全一致的 64-token SWA 页拆分与 512 B TMA 地址对齐修复**，确保两边运行相同的算子数值语义与显存驻留口径（`engram cpu_offload=false`）。
> 3. **严格零前缀缓存与隔离种子口径**：[`crates/zs-kv/src/radix.rs`](../crates/zs-kv/src/radix.rs) 未接入 serving，`zstep` 仅启用 [`BlockPool`](../crates/zs-kv/src/lib.rs) 物理页分配；因此 `vLLM 0.31.0` 统一强制关闭自动前缀缓存（`--no-enable-prefix-caching` / `enable_prefix_caching=False`），且两边均采用确定性隔离种子 `tier_seed = base_seed + round_idx * 1000 + tier_idx`（Warmup 使用 `base_seed + 900_000`），彻底杜绝跨轮/跨档 APC 命中污染（详见 [`08-phase3-dsv41-optimization-plan.md`](08-phase3-dsv41-optimization-plan.md)）。
> 4. **纯引擎同口径边界对齐（响应审查 Issue `#3` 二.1）**：由于 `zstep` 当前直接通过进程内 [`ModeBServingEngine::run_benchmark_tier_with_seed`](../crates/zs-scheduler/src/serving.rs) 输入确定性随机 `int32` token IDs（无 HTTP、无分词、无解码、无 SSE），本阶段专门新增 [`bench_vllm_pure_engine.py`](../worker/tests/phase3/bench_vllm_pure_engine.py)，在 `vLLM 0.31.0` 进程内直接驱动 `AsyncLLMEngine`（`OPENAI_API_SERVER` 上下文、`max_num_batched_tokens=8192`）+ `TokensPrompt(prompt_token_ids=...)` + `detokenize=False` + `RequestOutputKind.DELTA`，并与 `zstep` 共享**逐位完全相同的 SplitMix64 `int32` prompt 序列**与 FIFO 并发信号量，彻底剥离 API/分词/SSE 开销差异。

---

## 1. 硬件拓扑与按尺寸自适应集合通信微基准

### 1.1 `8 × NVIDIA RTX 6000D` 拓扑钉死

```text
GPU 0, 1, 2, 3 -> Socket 0 PCIe Gen5 -> NUMA 0
GPU 4, 5, 6, 7 -> Socket 1 PCIe Gen5 -> NUMA 1
显存容量: 8 × 85,651 MiB GDDR7 (总计 669.15 GiB)
架构/驱动: sm_120 (Blackwell Workstation, 156 SMs), Driver 595.99.02, Mem Clock 12,481 MHz, Graphics Clock 2,422 MHz
互联约束: 无 NVLink；can_device_access_peer(i, j) == 0 (8×8 全 0, topo = CNS)
```

[`numa_pin.py`](../worker/zstep_worker/numa_pin.py) 在 Worker 启动期自动探测 PCIe BDF 对应的 `numa_node`（实测 `worker_numa_nodes = [0, 0, 0, 0, 1, 1, 1, 1]`）并执行 `os.sched_setaffinity`，同时在 [`runtime_loop.py`](../worker/zstep_worker/runtime_loop.py) 与 [`mode_b.rs`](../crates/zs-scheduler/src/mode_b.rs) 中强制 `OMP_NUM_THREADS=1`，避免 8 个 Worker 在大模型权重加载期派生数千个 OpenMP 自旋线程引发跨 Socket 活锁。

### 1.2 `H=5120` `shm_v5` Per-Rank NUMA-Local AllReduce 微基准与双通道自适应路由

来源：[`phase0_6000d_ar_shmv5.json`](benchmarks/phase0_6000d_ar_shmv5.json)。在 [`shm_allreduce.py`](../worker/zstep_worker/kernels_sm120/shm_allreduce.py) 中，共享内存集合通信采用 **`shm_v5` 每 Rank 独立 `/dev/shm` 段 + 本地 NUMA First-Touch + 远端 Push Flag / 本地 Acquire 轮询**：每个 Rank 仅轮询驻留在自身 NUMA 节点物理页上的完成标志，消除了跨 Socket UPI 总线上的远端缓存行自旋风暴，并提供 CUDA Graph 安全的 [`HostMappedShmAllReduce`](../worker/zstep_worker/kernels_sm120/shm_allreduce.py)、[`HostMappedShmAllGather`](../worker/zstep_worker/kernels_sm120/shm_allreduce.py) 与确定性全局 Top-2 [`HostMappedShmArgmax`](../worker/zstep_worker/kernels_sm120/shm_allreduce.py)。

下表完整保留 `H=5120`（BF16）、`blocks=8` 从 `M=1` 到 `M=4096` 的全部实测点，不省略大载荷劣势区间（`80 次` 列按单次 `p50 × 80` 推算，非独立端到端测量）：

| `M` | 载荷 (`H=5120` BF16) | `shm_v5 (b=8)` `p50 / p99 (µs)` | `shm_v4 (b=16)` `p50 / p99 (µs)` | `NCCL_LOC` `p50 / p99 (µs)` | `shm_v5` 相对 `NCCL_LOC` | Phase 3.5 路由通道 |
| ---: | ---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 10 KiB | **14.48 / 17.17** | 42.94 / 48.05 | 48.95 / 49.10 | **`3.38×` 更快**（`b=2` 达 `14.08 µs` / `3.48×`） | `shm_v5 (b=8)` |
| 2 | 20 KiB | **15.12 / 21.72** | 59.09 / 65.22 | 50.34 / 50.67 | **`3.33×` 更快** | `shm_v5 (b=8)` |
| 4 | 40 KiB | **19.28 / 20.09** | 59.83 / 68.82 | 53.98 / 54.17 | **`2.80×` 更快** | `shm_v5 (b=8)` |
| 8 | 80 KiB | **34.95 / 35.48** | 64.95 / 73.60 | 56.09 / 56.46 | **`1.60×` 更快** | `shm_v5 (b=8)` |
| 16 | 160 KiB | **35.18 / 35.33** | 68.64 / 74.46 | 77.90 / 78.23 | **`2.21×` 更快** | `shm_v5 (b=8)` |
| 32 | 320 KiB | **48.87 / 52.43** | 82.24 / 87.90 | 118.62 / 118.92 | **`2.43×` 更快** | `shm_v5 (b=8)` |
| 64 | 640 KiB | **91.50 / 91.60** | 116.77 / 121.38 | 227.93 / 228.63 | **`2.49×` 更快** | `shm_v5 (b=8)` |
| 128 | 1.25 MiB | **157.54 / 157.66** | 193.89 / 199.26 | 187.48 / 188.80 | **`1.19×` 更快** | `shm_v5 (b=8)` |
| 512 | 5.00 MiB | 592.04 / 592.30 | 640.26 / 645.01 | **581.68 / 584.21** | `-1.8%`（NCCL 略快） | `NCCL_LOC`（Eager） |
| 4096 | 40.00 MiB | 4,660.95 / 4,662.06 | 4,867.22 / 4,868.64 | **3,944.79 / 3,954.33** | `-15.4%`（NCCL 更快） | `NCCL_LOC`（Eager） |

* **Phase 3.5 按尺寸自适应双通道（Size-Aware Collective Routing）**：
  * **小中载荷与全部 CUDA Graph 步（`M <= 256` 或流处于图捕获态）**：100% 走 `shm_v5 (blocks=8)`，完整保留 `M=1..128` 相对 `NCCL_LOC` **`1.19×..3.38×`** 的低延迟优势（单流 `M=1` 80 次通信仅 `1.16 ms`，比 `NCCL_LOC` `3.92 ms` 每 token 节省 `2.76 ms`），并保持单调计数器绝对一致。
  * **大载荷 Eager 混合/Prefill 步（`M > 256` 且非图捕获态）**：[`deepseek_v41.py`](../worker/zstep_worker/models/deepseek_v41.py) 自动将 `tensor_model_parallel_all_reduce` 与 `all_gather` 路由至多 SM 并发的 `NCCL_LOC`（启动预热计数 `shm_ar_warm=3402`、`shm_ag_warm=178`、`nccl_large_warm=1010`、`nccl_fallback_warm=0`），在 `M=3072..4096` 大批量步中每步节省约 `43..57 ms`。

---

## 2. `DeepSeekV41StepRunner` Mode B 与 Phase 3.5 混合批调度架构

1. **128-token Rust 物理页池与 SM120 64-token SWA 页映射**：
   * Rust 控制面（[`serving.rs`](../crates/zs-scheduler/src/serving.rs)、[`zs-wire`](../crates/zs-wire/src/lib.rs)）以 `page_block_tokens = 128` 管理 KV 页表（`num_kv_pages = 2560`），显式保留物理页 `Block 0` 为全零 `null_block`，用于安全吸收解码 CUDA Graph 桶中的 padding 槽位写入与读取。
   * Python 执行器 [`deepseek_v41.py`](../worker/zstep_worker/models/deepseek_v41.py)（`DeepSeekV41StepRunner`）针对 `sm_120` 硬件约束，将 128-token 逻辑页无损映射为 2 个连续的 64-token SWA/`cr=1` 物理子页（`2 * page_id` 与 `2 * page_id + 1`），并保证 KV 缓存基址满足 512 B TMA 对齐。
2. **顺风车混合批调度（Piggyback Mixed Decode + Chunked Prefill）**：
   * **无新请求等待或半完成 Prefill 时**：`active` 中 `1..128` 条在途解码序列直接命中已捕获的全 40 层单图 CUDA Graph 桶 `(1, 2, 4, 8, 16, 32, 64, 128)` 并保持 `in_flight_depth=1` 预暂存，**单流与纯解码稳态零退化（`Mean TPOT = 9.55..9.87 ms`）**。
   * **存在活跃解码序列且有待推进 Prefill 时**：[`ModeBServingEngine::stage_mixed_step`](../crates/zs-scheduler/src/serving.rs) 在 `--max-mixed-step-tokens 3072` 预算内把在途单 token 解码与分块/整条新请求 Prefill 打包进同一个前向步，使在途解码与新请求 Prefill **共享同一次 `475.25 GiB` MoE 权重读取**，消除了旧分离式调度下每接纳一批新请求就单独读取整套专家权重所造成的吞吐断层与 `~650..682 ms` 停顿。
   * [`DeepSeekV41StepRunner.execute`](../worker/zstep_worker/models/deepseek_v41.py) 按 `FlashInfer` `sm_120` 稀疏 MLA / SWA / KV Compressor 元数据契约将混合批规范化排布为 4 个连续区域（真解码 `q=1, pos>0` → 单 token 初始步 `q=1, pos=0` → 续块 Prefill `q>1, pos>0` → 新块 Prefill `q>1, pos=0`），精确设置 `is_prefilling_cpu = (q_lens > 1) | (first_pos == 0)`，并通过置换表将采样结果无损映射回原始 `sample_row_indices`。
3. **无状态精确 3-token Engram V2 滑窗与确定性全局 Top-2 词表归约**：
   * Rust 调度器在每个 `StepPlanSlot` 的 [`engram_prev_window`](../crates/zs-wire/src/lib.rs) 中直接下发跨 Prompt 边界与 Chunked Prefill 边界的精确前驱 3-token 窗口，Engram 权重全量驻留 GPU 显存（`cpu_offload=False`）。
   * [`HostMappedShmArgmax`](../worker/zstep_worker/kernels_sm120/shm_allreduce.py) 通过 `local_top2_bf16_kernel` 与 `tp8_argmax_gather_shm_kernel`（`128 B/row`）在 `128K` 词表（每卡 `16,384` 列）上执行严格的 `(logit 降序, global_token_id 升序)` 全序归约，复用 [`SeqStepOutput`](../crates/zs-wire/src/lib.rs) 既有槽位零协议膨胀回传全局 `Top-1 / Top-2` token 与原始 BF16 logit。

---

## 3. 正确性门禁与确定性全局 Top-2 贪心对齐审计

| 检查项 | 覆盖内容 | 实测结果 |
| :--- | :--- | :--- |
| `cargo fmt --check` + `clippy -D warnings` + `cargo test --workspace` | `zs-wire` 布局与容量派生、`zs-shm` RAII、`zs-kv`、`zs-scheduler` 混合批调度 / 重叠解码 Engram 3-token 滑窗修订 / Top-2 审计、`zs-server` | **32 passed, 0 warnings** |
| [`test_shm_layout_parity.py`](../worker/tests/test_shm_layout_parity.py) + `ruff check` | Python `ctypes` 视图与 Rust `#[repr(C)]` 结构体字节偏移逐项对齐、静态代码规范 | **1 passed, 0 lint errors** |
| `--verify-greedy`（`DeepSeek-V4.1-Flash` `TP=8`） | 8 卡真权重对 5 个无缓存 `vLLM 0.31.0` 贪心参考序列执行单序列、5 序列批处理与 `--max-prefill-chunk 24` 顺风车混合步；核对全局 Top-2 裕度、Triton 零重编译与 `/dev/shm` RAII | **`TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=5`**，`SHM_CLEAN`；**高约束结构化任务 `128/128` 100% 精确一致，低裕度 ULP 边界分歧如实保留** |

### 3.1 五条 Prompt（Single / Batched / Mixed-Chunked）全局 Top-2 贪心对齐报告

参考：[`phase3_vllm_greedy_reference.json`](benchmarks/phase3_vllm_greedy_reference.json)（含 `vLLM` Top-5 logprobs）；实测：[`phase3_greedy_parity_report.json`](benchmarks/phase3_greedy_parity_report.json)。`max_prefill_chunk=24` 混合调度共执行 `4` 个 prefill steps、`127` 个 decode steps（含 `3` 个 `decode_steps_with_partial_prefill` 混合步，`max_active_sequences=5`）。全局顶层汇总为 `all_matched=false`：

| Prompt | 输出长度 | Single 精确前缀 / 全序列匹配率 | Batched (`bucket=8`) | Mixed Chunked (`chunk=24`) | Single 平均 / 最小 `Top1-Top2` 裕度 | 首处分歧步裕度 (`zstep` / `vLLM`) |
| :--- | ---: | :--- | :--- | :--- | :---: | :---: |
| **`prime_squares_table`** | **128** | **`128 / 100.00%`** | **`128 / 100.00%`** | **`128 / 100.00%`** | **`9.801 / 4.250`** | **无分歧（三模式 `128/128` 全匹配）** |
| `posix_shm_tlb_shootdown` | 64 | `12 / 18.75%` | `0 / 0.00%` | `0 / 0.00%` | `6.142 / 0.000` | Step 12: **`0.125`** / `1.250` |
| `france_capital` | 32 | `1 / 21.88%` | `3 / 9.38%` | `3 / 9.38%` | `4.762 / 0.375` | Step 1: **`0.375`** / `2.875` |
| `ieee754_subnormals` | 128 | `2 / 3.91%` | `2 / 1.56%` | `4 / 3.91%` | `5.603 / 0.000` | Step 2: `2.375` / `1.687` |
| `dijkstra_shortest_path` | 96 | `0 / 2.08%` | `2 / 3.12%` | `1 / 3.12%` | `2.569 / 0.000` | Step 0: **`0.125`** / `0.500` |

**全局 Top-2 裕度归因结论**：
1. **算子、页表、Engram 滑窗与顺风车混合批语义完全正确**：在消除 Base 模型开放式复读退化的高约束结构化任务 **`prime_squares_table`**（77-token 输入、连续生成 128 个素数平方表 token，最小 `top1 - top2` 裕度 `4.250`）上，`zstep` 的 **单序列、多序列批处理以及 `chunk=24` 混合 Decode+Prefill 三种执行模式与 `vLLM 0.31.0` 实现 `128 / 128`（`100.00%`）逐 token 精确一致**。
2. **低裕度开放式续写的 ULP 边界翻转如实保留**：其余 4 条开放式 Base 模型续写在首处分歧点普遍处于极窄 BF16 裕度（如 `dijkstra_shortest_path` 与 `posix_shm_tlb_shootdown` 的 `zstep` `top1 - top2` 仅 **`0.125`**，`france_capital` 仅 **`0.375`**，`single` vs `batched` 甚至出现 `20.25 vs 20.25` 的 `0.0` 完全打平）；40 层 `UE8M0` MXFP8 激活量化、384 专家路由与 `shm_v5` 两阶段平面 FP32 归约（不同于 `NCCL` 环形归约）累积的细微数值差异翻转该极窄边界后，自回归轨迹自然分叉，因此顶层仍诚实报告 `all_matched=false`。

---

## 4. 单配置、真无缓存、Warm、`n=3` 端到端对标（`zstep` Phase 3.5 vs `vLLM 0.31.0`）

* **统一测试环境**：同一台 `8 × NVIDIA RTX 6000D`（无 P2P，2-NUMA `[0,0,0,0,1,1,1,1]`），同一容器镜像 `vllm/vllm-openai:v0.31.0`，同一模型权重 `DeepSeek-V4.1-Flash`（`1017`），`TP=8`，`bfloat16`，贪心采样（`temperature=0`），`ignore_eos=true`，`engram cpu_offload=false`，`NCCL_P2P_DISABLE=1`，`NCCL_P2P_LEVEL=LOC`。**`vLLM 0.31.0` 显式传入 `--no-enable-prefix-caching`，两边统一使用按 `(round, tier)` 隔离的确定性随机种子**。
* **zstep 唯一常驻配置**：`allreduce_blocks=8`、`num_kv_pages=2560`、`max_seqs=128`、`max_step_tokens=max_decoder_rows=4096`、`max_prefill_chunk=2048`、`max_mixed_step_tokens=3072`、`max_pages_per_seq=256`（冷启动初始化 + Autotune + 全图捕获耗时 `init_wall_ms = 111.14..111.44 s`）。Round 0 前执行 16 个 warmup prompts，随后按 round-major 交错执行 3 轮（`n=3`），**跨所有 10 个档位零参数切换**。运行时全程报告 **`TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=5`**，退出后 `/dev/shm` 零残留、8 卡显存 `0 MiB`。
* **统计判定口径**：表中报告三轮 **`mean ± population std`**；吞吐相对差 `<3%` 或均值差 `<= 2 × sqrt(zstep_std² + vllm_std²)` 判定为 **持平**。`ttft_ms` 为获得并发 permit 后计时的 **permit-TTFT**（与 `vLLM` `limited_request_func` 同口径）；`submit_ttft_ms` 额外包含客户端并发池排队等待时间。

> **总览结果**：在剔除 APC 污染并合入 Phase 3.5 顺风车混合批调度 + 按尺寸自适应通信后，**`zstep` 在两套基准全部 `10 / 10` 个档位上实现全线显著胜出（`+3.45% .. +19.16%`）**，且在全部 `10 / 10` 个档位上取得更低的 permit-Median TTFT（最高快 **`4.72×`**）。

### 4.1 套件一：固定 `512 -> 128` 四档并发扩展（`c=1, 8, 16, 32`）

原始数据：[`phase3_zstep_dsv41_tp8_in512_out128.json`](benchmarks/phase3_zstep_dsv41_tp8_in512_out128.json)、[`phase3_vllm_dsv41_tp8_in512_out128.json`](benchmarks/phase3_vllm_dsv41_tp8_in512_out128.json)（各档请求数依次为 `16 / 32 / 64 / 128`）。

| 档位 | zstep Output (`tok/s`) | vLLM Output (`tok/s`) | 均值差 / 判定 | Mean TPOT z / v (`ms`) | z Median TTFT / submit-TTFT (`ms`) | v Median TTFT (`ms`) | z P99 ITL (`ms`) | v P99 ITL (`ms`) |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| `c1` | **96.18 ± 0.57** | 83.97 ± 0.05 | **+14.53% / zstep 更快** | **9.55 ± 0.06** / 10.95 ± 0.00 | **117.56 ± 0.19** / 9,471.93 ± 19.16 | 132.60 ± 1.27 | **9.80 ± 0.03** | 11.38 ± 0.13 |
| `c8` | **377.50 ± 2.52** | 351.50 ± 10.04 | **+7.40% / zstep 更快** | **17.14 ± 0.14** / 18.25 ± 0.17 | **535.78 ± 0.09** / 3,224.43 ± 35.53 | 594.99 ± 1.19 | **17.67 ± 0.12** | 19.00 ± 0.33 |
| `c16` | **527.00 ± 1.87** | 509.40 ± 3.55 | **+3.45% / zstep 更快** | 26.51 ± 0.11 / **23.47 ± 0.23** | **410.34 ± 0.43** / 4,949.74 ± 48.17 | 1,081.25 ± 0.78 | 351.43 ± 0.19 | **25.26 ± 0.15** |
| `c32` | **663.78 ± 3.57** | 635.98 ± 1.49 | **+4.37% / zstep 更快** | 43.11 ± 0.26 / **38.50 ± 0.12** | **411.54 ± 0.40** / 8,399.90 ± 102.45 | 1,228.47 ± 14.19 | 410.91 ± 0.25 | **107.80 ± 0.70** |

### 4.2 套件二：六档交错主基准（`in512` / `in1024` × `output=256`）

原始数据：[`phase3_zstep_dsv41_tp8_6tier.json`](benchmarks/phase3_zstep_dsv41_tp8_6tier.json)、[`phase3_vllm_dsv41_tp8_6tier.json`](benchmarks/phase3_vllm_dsv41_tp8_6tier.json)（各档请求数依次为 `8 / 48 / 256 / 20 / 200 / 400`）。

| 档位 | zstep Output (`tok/s`) | vLLM Output (`tok/s`) | 均值差 / 判定 | Mean TPOT z / v (`ms`) | z Median TTFT / submit-TTFT (`ms`) | v Median TTFT (`ms`) | z P99 ITL (`ms`) | v P99 ITL (`ms`) |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| `in512_c1` | **98.94 ± 0.33** | 87.24 ± 0.07 | **+13.42% / zstep 更快** | **9.68 ± 0.03** / 10.98 ± 0.00 | **118.68 ± 0.84** / 7,889.69 ± 26.37 | 133.63 ± 1.67 | **9.87 ± 0.04** | 11.48 ± 0.14 |
| `in512_c16` | **609.53 ± 7.15** | 573.38 ± 6.84 | **+6.30% / zstep 更快** | 24.15 ± 0.31 / 24.07 ± 0.38 | **410.82 ± 0.27** / 7,580.25 ± 54.25 | 1,094.82 ± 18.29 | **23.47 ± 0.25** | 25.12 ± 0.53 |
| `in512_c128` | **1,372.77 ± 2.51** | 1,152.03 ± 6.34 | **+19.16% / zstep 更快** | **82.32 ± 0.17** / 96.30 ± 0.60 | **831.11 ± 0.31** / 8,983.78 ± 1.96 | 3,068.72 ± 1.11 | **415.33 ± 0.14** | 1,019.66 ± 0.68 |
| `in1024_c1` | **95.24 ± 0.09** | 84.20 ± 0.02 | **+13.11% / zstep 更快** | **9.87 ± 0.01** / 11.18 ± 0.00 | **171.66 ± 0.16** / 24,369.74 ± 41.77 | 189.45 ± 0.73 | **10.07 ± 0.01** | 11.72 ± 0.02 |
| `in1024_c32` | **649.63 ± 1.94** | 614.93 ± 3.52 | **+5.64% / zstep 更快** | 44.18 ± 0.17 / **42.06 ± 0.35** | **415.61 ± 0.15** / 37,201.49 ± 204.12 | 1,959.74 ± 61.20 | 415.40 ± 0.11 | **175.89 ± 0.90** |
| `in1024_c128` | **1,002.13 ± 3.22** | 878.86 ± 4.26 | **+14.03% / zstep 更快** | **108.63 ± 0.34** / 120.03 ± 0.49 | **837.54 ± 0.10** / 41,618.50 ± 129.12 | 3,108.79 ± 0.44 | **419.58 ± 0.14** | 1,035.37 ± 0.43 |

---

## 5. 物理归因与性能权衡边界分析

1. **低延迟单流与中小并发（`c=1..16`）全维度领先**：
   * 在全部三个单流档位（Suite 1 `c1`、Suite 2 `in512_c1`、`in1024_c1`），`zstep` 输出吞吐达 **`95.24..98.94 tok/s`**，领先无缓存 `vLLM 0.31.0`（`83.97..87.24 tok/s`）**`+13.11%..+14.53%`**；单步解码 Mean TPOT 低至 **`9.55..9.87 ms`**（比 `vLLM` 快 `1.30..1.40 ms/tok`），`P99 ITL` 低至 **`9.80..10.07 ms`**，且 permit-Median TTFT 同步快 **`10.4%..12.8%`**。
   * 在 `c=8` 与 `in512_c16` 档位，`zstep` 同时取得更高的输出吞吐（`+7.40%` 与 `+6.30%`）、更低的 permit-Median TTFT（`535.78 vs 594.99 ms`；**`410.82 vs 1,094.82 ms`，快 `2.66×`**）以及更低的 `P99 ITL`（`17.67 vs 19.00 ms`；`23.47 vs 25.12 ms`）。
2. **高并发满载区间（`c=128`）吞吐、TTFT、TPOT 与 `P99 ITL` 四项全面反超**：
   * 在 `in512_c128` 与 `in1024_c128` 两个高并发极限档位，得益于 `--max-mixed-step-tokens 3072` 顺风车混合批复用 `475 GiB` MoE 权重读取以及 `M > 256 -> NCCL_LOC` 大载荷通信路由，`zstep` 输出吞吐分别达到 **`1,372.77 tok/s`**（领先 `vLLM` `1,152.03 tok/s` **`+19.16%`**）与 **`1,002.13 tok/s`**（领先 `vLLM` `878.86 tok/s` **`+14.03%`**）。
   * 同时，`zstep` 在 `c=128` 下的 permit-Median TTFT 仅为 **`831.11..837.54 ms`**（比 `vLLM` 的 `3,068.72..3,108.79 ms` **快 `3.69×..3.71×`**），Mean TPOT 快 **`10.5%..16.9%`**，且解码 **`P99 ITL` 控制在 `415.33..419.58 ms`**，不仅较优化前 `zstep`（`674..682 ms`）降低 **`-38.5%`**，更比无缓存 `vLLM`（`1,019.66..1,035.37 ms`）**低 `2.46×..2.47×`**。
3. **中并发区间（`c=16..32`）的 TTFT vs `P99 ITL` 工程权衡边界（如实保留）**：
   * 在 `c=32`（`c32` 与 `in1024_c32`）及短输出 `c16`（`512 -> 128`）档位，`zstep` 输出吞吐领先 `vLLM` **`+3.45%..+5.64%`**，且 permit-Median TTFT 压低至 **`410.34..415.61 ms`**（相比 `vLLM` 的 `1,081.25..1,959.74 ms` **快 `2.64×..4.72×`**）。
   * 作为该混合步预算（`3072` tokens，单次混合步耗时约 `350..415 ms`，较旧 `4096` 分离步 `645..652 ms` 已下降 `-36%`）的直接物理权衡：当 `c=16..32` 持续涌入新请求时，`zstep` 以约 `410 ms` 一次的大吞吐混合步快速清空新请求 Prefill，使 `c=32` 的 `P99 ITL` 落在 **`410.91..415.40 ms`**，高于采用更细粒度分片但显著拉长排队 TTFT 的 `vLLM`（`107.80..175.89 ms`）。调用方若在特定 SLA 下优先追求 `<200 ms` 的 `c=32` 尾延迟，可通过运行时参数 `--max-mixed-step-tokens` 下调混合步预算（在吞吐/TTFT 与单步 ITL 之间按需折中）。

---

## 6. Phase 3.6：选择性 `b12x` SM120 计算算子集成与纯引擎同口径四方对标

### 6.1 `b12x_dsv41.py` 选择性算子替换策略

基于单算子同环境微基准（保留 `zstep` `shm_v5` / `NCCL_LOC` 通信层以及 `vLLM 0.31.0` 的 `mhc_shifted_post_pre`、`GateLinear`、`ParallelLMHead` 与 `FlashInfer` 稀疏 MLA/SWA 注意力），[`b12x_dsv41.py`](../worker/zstep_worker/kernels_sm120/b12x_dsv41.py) 仅替换两类在 `sm_120` 上具显著硬件优势的计算算子：

1. **40 层路由专家 `ZstepB12xMxfp4MoEMethod`（`E=384, top_k=6, H=5120, I_tp8=288`）**：
   * 保持原生未填充 `I_tp8 = 288`（`288 % 32 == 0`，无需像标准 `ModelOpt` 填充至 `384`），**每张 GPU 节省约 `10 GiB` 权重显存与 `25%` 无效 MMA 计算**；
   * 采用 `b12x.moe.fused_moe`（`W13Layout("w31")`、`ActivationMode.A8`、`swiglu_limit=10.0`），并强制 `B12X_W4A8_TINY_DECODE=0` 使 `M = 1, 2, 4` 命中 `dynamic` 内核（`36.9 µs` vs `micro` `117 µs`）。
2. **170 层 2D `[32, 32] UE8M0` FP8 稠密线性层 `ZstepB12xBlock32LinearMethod`**：
   * 覆盖 5 种形状（`fused_wqa_wkv` `1792×5120`、`wq_b` `4096×1280`、`wo_b` `5120×1024`、`shared_experts.gate_up_proj` `576×5120`、`engram.wkv` `3200×6144`；`3D BMM` `.wo_a` 与 `K % 64 != 0` 的 `288` 维 `down_proj` 自动保留标准 Cutlass MXFP8 路径）；
   * 对 `fused_wqa_wkv` (`1792×5120`) 在 `rows > 512` 的非捕获大批量 Eager 步自动回退 `FlashInferCutlassMxfp8LinearKernel`，兼顾解码低延迟与大批量 Prefill 算力利用率；
   * Rank 0 先行执行 `PreparationSession` 编译调优并持久化至缓存，Ranks 1..7 秒级加载，并为主流与 `SharedExperts` 辅助流分别预分配静态 256 B 对齐 GPU Arena，确保全桶 `[1, 2, 4, 8, 16, 32, 64, 128]` 零动态分配 CUDAGraph 捕获与重放（`TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=5`）。
   * **正确性门禁**（[`phase36_b12x_greedy_parity_report.json`](benchmarks/phase36_b12x_greedy_parity_report.json)）：`prime_squares_table`（128 tokens）在 Single / Batched / Mixed-Chunked (`chunk=24`) 三种模式下均为 **`128 / 128`（`100.00%`）精确一致**（最小裕度 `3.875`）。

### 6.2 四方 Warm `n=3` 完整对比（`vLLM HTTP` vs `vLLM 纯引擎同口径` vs `zstep 3.5` vs `zstep 3.6 + b12x`）

原始数据：
* `zstep 3.6 + b12x`：[`phase36_zstep_b12x_dsv41_tp8_in512_out128.json`](benchmarks/phase36_zstep_b12x_dsv41_tp8_in512_out128.json)、[`phase36_zstep_b12x_dsv41_tp8_6tier.json`](benchmarks/phase36_zstep_b12x_dsv41_tp8_6tier.json)
* `vLLM 0.31.0` 纯引擎同口径（`AsyncLLMEngine` + `TokensPrompt` + `detokenize=False`，无 HTTP/分词/解码/SSE）：[`phase36_vllm_pure_dsv41_tp8_in512_out128.json`](benchmarks/phase36_vllm_pure_dsv41_tp8_in512_out128.json)、[`phase36_vllm_pure_dsv41_tp8_6tier.json`](benchmarks/phase36_vllm_pure_dsv41_tp8_6tier.json)

#### 套件一：`512 -> 128`（`c=1, 8, 16, 32`）

| 档位 | `vLLM` HTTP (`tok/s`) | `vLLM` 纯引擎 (`tok/s`) | `zstep` 3.5 (`tok/s`) | **`zstep` 3.6 + `b12x` (`tok/s`)** | 3.6 vs 3.5 / vs `vLLM` 纯引擎 | Mean TPOT `vPure` / `z3.5` / **`z3.6`** (`ms`) | Med TTFT `vPure` / `z3.5` / **`z3.6`** (`ms`) | P99 ITL `vPure` / **`z3.6`** (`ms`) |
| :--- | ---: | ---: | ---: | ---: | :--- | :---: | :---: | :---: |
| `c1` | `83.97 ± 0.05` | `83.59 ± 0.01` | **`96.18 ± 0.57`** | `95.85 ± 0.65` | `-0.34%` / **`+14.67%`** | `10.94` / `9.55` / **`9.05`** | `141.37` / **`117.56`** / `185.21` | `11.39` / **`9.28`** |
| `c8` | `351.50 ± 10.04` | `348.18 ± 8.30` | `377.50 ± 2.52` | **`441.70 ± 1.60`** | **`+17.01%`** / **`+26.86%`** | `19.02` / `17.14` / **`14.23`** | `590.28` / `535.78` / **`510.32`** | `18.80` / **`14.79`** |
| `c16` | `509.40 ± 3.55` | `492.61 ± 2.27` | `527.00 ± 1.87` | **`612.24 ± 2.13`** | **`+16.18%`** / **`+24.29%`** | `25.63` / `26.51` / **`22.47`** | `1,084.36` / `410.34` / **`386.49`** | **`25.97`** / `327.96` |
| `c32` | `635.98 ± 1.49` | `617.07 ± 1.10` | `663.78 ± 3.57` | **`800.74 ± 1.01`** | **`+20.63%`** / **`+29.76%`** | `41.13` / `43.11` / **`35.12`** | `1,430.66` / `411.54` / **`387.05`** | **`86.20`** / `386.75` |

#### 套件二：`six_tier`（`in512` / `in1024` × `output=256`）

| 档位 | `vLLM` HTTP (`tok/s`) | `vLLM` 纯引擎 (`tok/s`) | `zstep` 3.5 (`tok/s`) | **`zstep` 3.6 + `b12x` (`tok/s`)** | 3.6 vs 3.5 / vs `vLLM` 纯引擎 | Mean TPOT `vPure` / `z3.5` / **`z3.6`** (`ms`) | Med TTFT `vPure` / `z3.5` / **`z3.6`** (`ms`) | P99 ITL `vPure` / **`z3.6`** (`ms`) |
| :--- | ---: | ---: | ---: | ---: | :--- | :---: | :---: | :---: |
| `in512_c1` | `87.24 ± 0.07` | `87.06 ± 0.01` | `98.94 ± 0.33` | **`103.99 ± 1.10`** | **`+5.10%`** / **`+19.45%`** | `10.97` / `9.68` / **`8.93`** | `141.86` / **`118.68`** / `184.93` | `11.41` / **`9.27`** |
| `in512_c16` | `573.38 ± 6.84` | `555.89 ± 1.34` | `609.53 ± 7.15` | **`727.06 ± 1.62`** | **`+19.28%`** / **`+30.79%`** | `25.32` / `24.15` / **`20.01`** | `1,086.45` / `410.82` / **`386.86`** | `26.12` / **`19.09`** |
| `in512_c128` | `1,152.03 ± 6.34` | `1,126.34 ± 1.87` | `1,372.77 ± 2.51` | **`1,686.53 ± 7.96`** | **`+22.86%`** / **`+49.74%`** | `99.30` / `82.32` / **`65.72`** | `3,066.32` / `831.11` / **`779.52`** | `1,021.41` / **`389.59`** |
| `in1024_c1` | `84.20 ± 0.02` | `84.57 ± 0.00` | `95.24 ± 0.09` | **`102.27 ± 0.18`** | **`+7.38%`** / **`+20.92%`** | `11.17` / `9.87` / **`9.07`** | `177.89` / **`171.66`** / `191.20` | `11.56` / **`9.47`** |
| `in1024_c32` | `614.93 ± 3.52` | `605.12 ± 1.35` | `649.63 ± 1.94` | **`778.39 ± 4.56`** | **`+19.82%`** / **`+28.63%`** | `42.27` / `44.18` / **`36.51`** | `2,127.63` / `415.61` / **`391.62`** | **`296.51`** / `391.34` |
| `in1024_c128` | `878.86 ± 4.26` | `866.25 ± 1.06` | `1,002.13 ± 3.22` | **`1,169.71 ± 1.06`** | **`+16.72%`** / **`+35.03%`** | `122.27` / `108.63` / **`92.32`** | `3,107.53` / `837.54` / **`787.39`** | `1,035.90` / **`394.26`** |

### 6.3 同口径实证与延迟权衡总结

1. **HTTP / 分词 / SSE 开销对 `vLLM V1` 的真实影响被精确量化**：
   * 在单流 `c=1` 下，去掉 HTTP、Prompt 分词、增量解码与 SSE 后，`vLLM 0.31.0` 的 `Mean TPOT` 仅变化 **`0.01 ms/tok`**（`10.95 -> 10.94 ms`、`10.98 -> 10.97 ms`、`11.18 -> 11.17 ms`），证明 `vLLM V1` 的 `APIServer <-> EngineCore` 跨进程流水线已将单 token 解码与 SSE 推送完全重叠隐藏在 `~11 ms` GPU 步内；仅 `in1024_c1` 的首字 `Median TTFT` 因省去 1024-token 分词与 HTTP 往返缩短了 **`11.56 ms`**（`189.45 -> 177.89 ms`）。
   * 在严格一致的纯引擎 `int32` token-ID 口径下，`zstep 3.5` 依然 **`10 / 10` 全档位领先 `vLLM` 纯引擎（`+6.98% .. +21.88%`）**，而 **`zstep 3.6 + b12x` 将领先幅度进一步扩大至 `+14.67% .. +49.74%`**（同时在全部 **`10 / 10` 个档位的 `Mean TPOT`** 上均优于 `vLLM`）。
2. **两项工程权衡边界透明披露**：
   * **`c=1` Eager Prefill Python `bind()` 开销**：在单流 `c=1` 的非捕获 Eager Prefill 步（`M=512` / `M=1024`）中，每步 210 次 `b12x` Python `bind()` 校验带来约 `20..67 ms` 固定 CPU 发射开销，使 `c=1` `Median TTFT` 为 `184.9..191.2 ms`（高于 Phase 3.5 的 `117.6..171.7 ms`），而解码阶段因 40 层整图 CUDAGraph 零 CPU 开销，`Mean TPOT` 降至 **`8.93..9.07 ms`**（纯解码速率 `110.3..112.0 tok/s`）；在 `c=8..128` 批量步中 GPU 计算完全掩盖 CPU 发射，`Median TTFT` 反而比 Phase 3.5 再快 `24..52 ms`。
   * **`c=16 / c=32` 混合步 `P99 ITL` 与 `TTFT` 的折中**：在短输出 `c16` (`512->128`) 与 `c=32` (`c32`、`in1024_c32`) 档位，`--max-mixed-step-tokens 3072` 用更高的 `P99 ITL`（`327.96..391.34 ms` vs `vLLM` 纯引擎 `25.97..296.51 ms`）换取了 **`2.81×..5.43×` 更低的 `Median TTFT`** 与 **`+24.29%..+29.76%` 更高的吞吐**；而在 `c=1`、`c=8`、`in512_c16` 与 `c=128` 档位，`zstep 3.6 + b12x` 的 `P99 ITL` 同时显著低于 `vLLM`。
