# 07 — Phase 3 / 3.5 / 3.6 / 3.7 / 3.8 实测报告：`DeepSeek-V4.1-Flash` (`TP=8`) 在 `8 × RTX 6000D` 端到端推理、纯引擎同口径对标与 SM120 算子优化

本文档记录 `zstep` Phase 3、Phase 3.5、Phase 3.6、Phase 3.7（`main` `d497e85`，PR `#11`）及 Phase 3.8（SM120 `FlashInfer` 实验算子筛选与集成）在对称双 NUMA、无 P2P 的 **`8 × NVIDIA RTX 6000D`**（每卡 `85,651 MiB` GDDR7、`sm_120`、156 SMs、驱动 `595.99.02`）服务器上，完成真权重端到端模型 **`DeepSeek-V4.1-Flash`**（`1017` 版本、`475.25 GiB` 混合量化权重：40 层 `DeepseekV41Attention` 稀疏 MLA + SWA 混合注意力、`cr=4` KV 压缩、384 专家 `UE8M0` MXFP4 MoE、TileLang 流形超连接 `mHC`、以及全驻留显存 Engram V2 3-token 查表）的 Mode B 架构实现、`shm_v5` + 按尺寸自适应跨卡集合通信、顺风车混合批调度、**Phase 3.6 / 3.7 / 3.8 选择性 `b12x` + `sm12x_mega_gate` + `sm120_dsa_indexer` SM120 算子集成与 `156-SM` 生命周期/全覆盖优化（40 层 `MXFP4 MoE` + 210 层 `[32, 32] UE8M0` FP8 Linear + 40 层 `M >= 32` 融合路由门控 + 8 层分页 Decode DSA Indexer）**、确定性全局 Top-2 贪心逐 token 审计，以及与官方容器 **`vllm/vllm-openai:v0.31.0`** 在 **双口径（进程内 `AsyncLLMEngine` + `TokensPrompt` 零 HTTP/分词/解码/SSE 纯引擎同口径基线 + `vllm bench serve` `/v1/completions` HTTP 流式基线）** 两组 Warm `n=3` 并发套件（4 档 `in512_out128` 与 6 档 `six_tier`）下的正面对标结果。

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
| `cargo fmt --check` + `clippy -D warnings` + `cargo test --workspace` | `zs-wire` 全字段偏移与无 `.git` provenance、`zs-shm` RAII、`zs-kv`（含整块前缀零写别名页校验）、`zs-scheduler` 混合批调度 / 看门狗 / Top-2 审计、`zs-server` 贪心门禁 | **41 passed, 0 warnings** |
| [`test_shm_layout_parity.py`](../worker/tests/test_shm_layout_parity.py) + `ruff check worker/` | Python 视图与 Rust `#[repr(C)]` 全结构体字节偏移（含 `ShmSuperblock` `112 vs 116` 哨兵回归测试）、仓库根 [`ruff.toml`](../ruff.toml) 锁定静态规则 | **4 passed, 0 lint errors** |
| `--verify-greedy-json`（`DeepSeek-V4.1-Flash` `TP=8`） | 8 卡真权重对 5 个无缓存 `vLLM 0.31.0` 贪心参考序列执行单序列、5 序列批处理与 `--max-prefill-chunk 24` 顺风车混合步；默认门禁校验 `TRITON_RECOMPILATIONS=0`（`rc=0`），`--require-exact-cases prime_squares_table` 强制锁定结构化算术表 `128/128` 100% 一致（`rc=0`；全量严格门禁 `--require-exact-greedy true` 或 `--max-greedy-mismatch-margin 0.25` 返回 `rc=1`） | **`TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=5`**，`SHM_CLEAN`；**高约束结构化任务 `128/128` 100% 精确一致，低裕度边界分歧如实保留** |

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

### 6.1 `b12x_dsv41.py` 选择性算子替换策略与容器 Overlay Provenance

> **`b12x` 运行环境与复现 Provenance**：官方基础镜像 `vllm/vllm-openai:v0.31.0` 仅包含 `vllm.utils.b12x` 桩模块，不包含 `flashinfer/experimental/b12x` 本体。Phase 3.6 实测在 `vllm/vllm-openai:v0.31.0` 基础上只读挂载公开仓库 [`local-inference-lab/flashinfer`](https://github.com/local-inference-lab/flashinfer) 的标签 **`flashinfer-cu134-sm120-beta-79819b112f8bda7d09709c24bb31de16d7ce59c6`**（commit `79819b112f8bda7d09709c24bb31de16d7ce59c6`）下的 `flashinfer/experimental/b12x -> /usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x:ro`，并设置 `ZSTEP_REQUIRE_B12X=1`（缺失 `b12x` 时立即报错退出，拒绝静默回退 `b12x=off`）与 `B12X_W4A8_TINY_DECODE=0`（完整挂载命令见 [`docs/benchmarks/README.md`](benchmarks/README.md) 第 6 节）。

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

---

## 7. Phase 3.7（`main` `d497e85` / PR `#11`）：`156-SM` `PreparationSession` 生命周期修复、210 层全覆盖与算子边界证明

### 7.1 四项正向优化落地（[`b12x_dsv41.py`](../worker/zstep_worker/kernels_sm120/b12x_dsv41.py) & [`deepseek_v41.py`](../worker/zstep_worker/models/deepseek_v41.py)）

针对 `RTX 6000D`（`sm_120`，`156 SMs`，区别于 `RTX 5090` `170 SMs` 与 `RTX PRO 6000` `188 SMs`），由可复现单算子微基准与数值审计脚本 [`bench_sm120_dsv41_operators.py`](../worker/tests/phase3/bench_sm120_dsv41_operators.py)（归档原始结果 [`phase37_6000d_operator_audit.json`](benchmarks/phase37_6000d_operator_audit.json)，`git_sha = d497e85c91619c0cee8121b691ba31b086d593ea`）逐档验证后，Phase 3.7 落地四项确定性改进：

1. **修复 `PreparationSession.close()` 提前析构导致的调优计划回退**：
   * `b12x` 的 `PreparationSession.prepare()` 内部已排空并关闭编译进程池（`self._pool = None`），而此前在 `prepare()` 后立即调用 `sess.close()` 会清空全部 `plan._prepared`（[`phase37_6000d_operator_audit.json`](benchmarks/phase37_6000d_operator_audit.json) `session_lifecycle_probe`：`prepared_while_session_open=true -> prepared_after_session_close=false`）并执行 `evict_unretained()`，导致运行时每层首次调用触发 `"was not prepared before its first use; preparing its default configuration"` 警告并丢失 `156-SM` `autotune=True` 最优配置；
   * Phase 3.7 将 `PreparationSession` 挂载于 `model._b12x_session` 直至 `DeepSeekV41StepRunner.close()`（在 `decode_graphs.clear()` 之后释放），并将持久化调优缓存升级至 `b12x_zstep_dsv41_v2`，使 **`unprepared` 回退告警降至 `0` 条**，双机实测热缓存准备耗时稳定在 **`~8–9 s`（`7.7..8.8 s`，本批次三轮干净启动 `prep_ms = 8,230..8,267 ms`）**。
2. **解锁全部 40 层 `shared_experts.down_proj`（`5120 × 288`，`in_features % 32 == 0`）**：
   * 将 `ZstepB12xBlock32LinearMethod` 准入条件从 `in_features % 64 == 0` 放宽至硬件原生块对齐 `in_features % 32 == 0`（[`phase37_6000d_operator_audit.json`](benchmarks/phase37_6000d_operator_audit.json) `shared_down_288`：`cos_sim = 1.0`、`max_abs_diff = 0.0`），使 `b12x` 2D FP8 线性层覆盖数从 `170` 提升至 **`210` 层**（`lin=210/6`；单层 CUDAGraph `M=1` 从 Cutlass `14.37 µs` 降至 **`6.16 µs`**，40 层每步节省 `328.5 µs`）。
3. **预解析零校验快速绑定（Zero-Validation Fast Binding）**：
   * 初始化阶段预先解析 `require_prepared(plan, "gemm.block_fp8_linear")`、`require_prepared(eplan, "moe.decode")`、`1024 B` 对齐 workspace 切片及 `rows == cap` 精确 `MXFP8Rows` 视图，前向直调 `BlockFP8LinearBinding` + `state.run_binding()`（单算子 Eager 墙钟在 `M=256..1024` 从标准 `bind()` 的 `136.8..168.3 µs` 降至 `91.8..94.2 µs`），消除每步 250 次 Python `bind()` 重复校验开销。
4. **`156-SM` 按形状实测 Eager Cutlass 混合切换阈值**：
   * 在保留 `PreparationSession` 与 `autotune=True` 后（[`phase37_6000d_operator_audit.json`](benchmarks/phase37_6000d_operator_audit.json) `eager_us`），`fused_wqa_wkv`（`1792 × 5120`，40 层）在 `M <= 2048` 均由 `b12x_fast` 胜出（`M=1024/2048` 为 `93.19 / 133.80 µs` vs Cutlass `154.64 / 169.49 µs`），仅在 `M >= 3072` 由 Cutlass 胜出（`237.91 vs 308.71 µs`）；`engram.wkv`（`3200 × 6144`，2 层）在 `M <= 512` 由 `b12x_fast` 胜出（`M=512` `113.00 vs 154.09 µs`），在 `M=1024..2048` 由 Cutlass 胜出（`172.46 / 319.72 µs` vs `184.45 / 360.15 µs`）。改为按形状精确切换 `_CUTLASS_LARGE_M_THRESHOLDS = {(1792, 5120): 2048, (3200, 6144): 512}`。

### 7.2 硬件实证排除项与在线服务显存余量边界（[`phase37_6000d_operator_audit.json`](benchmarks/phase37_6000d_operator_audit.json)）

| 候选方向 / 运行边界 | `RTX 6000D` 实测数据与根因结论 | 决策 |
| :--- | :--- | :--- |
| **`B12X_W4A8_TINY_DECODE=1`**（`MoETinyDecodeKernelBackend`） | 虽在 `M=1, 2, 4` CUDAGraph 快至 `18.06 / 22.56 / 34.99 µs`（对比 `dynamic` `36.90 / 57.39 / 59.50 µs`），但隔离缓存数值审计（`moe_w4a8_tiny_decode_audit`）证实 `tiny_decode.py` **未实现 `DeepSeek-V4.1-Flash` 必需的 `swiglu_limit=10.0` 截断**且激活采用非量化 FP16 点积，`M=1, 2, 4` 输出严重失真（**`cos_sim = 0.870..0.879`**，`max_abs_diff = 1,476..2,664`） | **严禁开启**（固定 `=0`） |
| **`b12x.gemm.wo_projection`**（融合 `inv_rope + wo_a + wo_b`） | 在 `M=1..128` CUDAGraph 解码桶上（`oproj_lmhead_mhc_audit.o_proj`），现有 `torch.bmm(BF16 wo_a) + b12x(wo_b)` 为 **`12.32..20.52 µs`**，全桶快于 `b12x.wo_projection` 的 `16.18..24.60 µs`（强行切换在 `M=1 / 128` 反退化 `-163.9 / -163.3 µs/step`） | **保持现有路由** |
| **`ParallelLMHead` 替换 `b12x.bf16_vocab_projection`** | `TP=8` 词表投影 `16160 × 5120` BF16（`vocab_tp8 < MIN_TRITON_OUT_FEATURES = 16,384`）：`M=1..64` `cuBLAS` 为 `135.37..143.89 µs`（对比 `b12x` `126.46..143.84 µs`，`M>1` 自动走同一 `torch`/`cuBLAS` 路径），有效显存带宽已达 **`1,169..1,222.7 GB/s`** | **保持 `cuBLAS`** |
| **`b12x.norm.mhc` 替换 `vLLM` TileLang `mhc_shifted_post_pre`** | `vLLM` TileLang `M=1` CUDAGraph 仅 **`9.70 µs/call`**（全模型 80 次合计 `0.78 ms`），已高度逼近寄存器归约下限 | **保持 TileLang** |
| **绕过 ACS 开启跨桥 PCIe P2P（`NCCL_P2P_LEVEL=SYS/PXB`）** | 工作站版 `RTX 6000D` 驱动层 `can_device_access_peer == 1`（`128 GiB` BAR2），但 20 个 Mellanox CX-8 集成 PCIe 桥下行口（`15b3:197c`）默认开启 `ACSCtl = 0x001d`。实测即使通过 `setpci ECAP_ACS+06.w=0000` 清零全部 20 个端口 ACS 并重载 `nvidia.ko`：Copy Engine DMA（`cudaMemcpyPeer`）虽可在同桥 `PXB` 跑通 `107.9 GB/s`，**但 GPU SM 核直接 BAR MMIO（`ld/st`）在全部 4 组同桥 GPU 对上均触发 PCIe `UR`（读回 `0xFFFFFFFF` 毒值），且多对双向并发 `NCCL` P2P 因二级 PCIe 交换网 + 双路 UPI 信用依赖环死锁**（详见 §8.1 与 [`phase37_6000d_comm_audit.json`](benchmarks/phase37_6000d_comm_audit.json)） | **保持 `shm_v5` + `NCCL_LOC`** |
| **在线服务显存水位与余量告警（`num_kv_pages=2560`）** | 在 `engram cpu_offload=false` + 8 档 40 层整图 CUDAGraph 常驻下，`kv_pages=2560`（`327,680` tokens）实测单卡显存占用为 **`84,937..84,959 MiB / 85,651 MiB = 99.2%`（每卡剩余 `< 0.7 GiB`）**。离线基准稳定零 OOM，但长期在线 HTTP 服务（`zstep --serve`）若需为驱动/NCCL/动态大 Prefill 预留安全缓冲，建议下调至 `--num-kv-pages 2048..2304`（释放约 `2.5..5.0 GiB/卡`） | **基准上限 `2560`；长稳在线建议 `2048..2304`** |

### 7.3 Phase 3.7 vs Phase 3.6 vs `vLLM` 纯引擎同口径 Warm `n=3` 主结果（干净 `d497e85` 自证归档）

原始数据（均在未修改的 `main` `d497e85c91619c0cee8121b691ba31b086d593ea` 源码树下独立采集，JSON 内 `provenance.git_commit == "d497e85"` / `git_sha == "d497e85..."`）：
* 单算子微基准与数值审计：[`phase37_6000d_operator_audit.json`](benchmarks/phase37_6000d_operator_audit.json)
* 正确性门禁：[`phase37_b12x_greedy_parity_report.json`](benchmarks/phase37_b12x_greedy_parity_report.json)（`prime_squares_table` 单序列 / 批处理 / `chunk=24` 混合步均 **`128 / 128` 100% 精确一致**，最小裕度 `3.750`；`TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=5`；`b12x=on(moe=40,lin=210/6)`，`unprepared_warnings=0`）
* 套件一（`512 -> 128`）：[`phase37_zstep_b12x_dsv41_tp8_in512_out128.json`](benchmarks/phase37_zstep_b12x_dsv41_tp8_in512_out128.json)（`init_wall_ms = 119.04 s`）
* 套件二（`six_tier`）：[`phase37_zstep_b12x_dsv41_tp8_6tier.json`](benchmarks/phase37_zstep_b12x_dsv41_tp8_6tier.json)（`init_wall_ms = 118.43 s`）

> **统计判定说明（严格遵循 `<3%` 或 `2σ` 持平规则）**：下表中 `c=1` 三个单流档位（`c1`、`in512_c1`、`in1024_c1`）相对 Phase 3.6 的输出吞吐差值（`-0.20%`、`-2.29%`、`-2.87%`）均落在 `<3%` 单轮 GPU 动态 Boost 频率波动噪声带（`2,295..2,422 MHz`）内，统一判定为 **`持平`**（单流真实确定性收益体现在 Eager Fast-Bind 带来的 **`Median TTFT` 全线缩短 `-15.41..-18.71 ms`**）；在并发档位 `c=16..128` 上，输出吞吐呈一致正向推进（`+1.66%..+3.16%`，其中 Suite 1 `c16` `+3.16%` 跨过严格 `3%` 显著阈值），而相对 `vLLM 0.31.0` 纯引擎同口径基线，**全部 `10 / 10` 个档位均显著胜出（`+14.44% .. +52.48%`）**。

#### 套件一：`512 -> 128`（`c=1, 8, 16, 32`）

| 档位 | `vLLM` 纯引擎 (`tok/s`) | `zstep` 3.6 (`tok/s`) | **`zstep` 3.7 (`tok/s`)** | 3.7 vs 3.6 / vs `vLLM` 纯引擎 | Mean TPOT `vPure` / `z3.6` / **`z3.7`** (`ms`) | Med TTFT `vPure` / `z3.6` / **`z3.7`** (`ms`) | P99 ITL `z3.6` / **`z3.7`** (`ms`) |
| :--- | ---: | ---: | ---: | :--- | :---: | :---: | :---: |
| `c1` | `83.59 ± 0.01` | `95.85 ± 0.65` | **`95.66 ± 0.20`** | `-0.20%`（持平） / **`+14.44%`** | `10.94` / `9.05` / **`9.20`** | `141.37` / `185.21` / **`169.42`** (`-15.79 ms`) | `9.28` / **`9.43`** |
| `c8` | `348.18 ± 8.30` | `441.70 ± 1.60` | **`446.37 ± 1.62`** | `+1.06%`（持平） / **`+28.20%`** | `19.02` / `14.23` / **`14.07`** | `590.28` / `510.32` / **`506.43`** | `14.79` / **`14.62`** |
| `c16` | `492.61 ± 2.27` | `612.24 ± 2.13` | **`631.57 ± 0.97`** | **`+3.16%`** / **`+28.21%`** | `25.63` / `22.47` / **`21.71`** | `1,084.36` / `386.49` / **`381.68`** | `327.96` / **`323.60`** |
| `c32` | `617.07 ± 1.10` | `800.74 ± 1.01` | **`820.22 ± 1.82`** | `+2.43%` / **`+32.92%`** | `41.13` / `35.12` / **`34.23`** | `1,430.66` / `387.05` / **`382.36`** | `386.75` / **`382.07`** |

#### 套件二：`six_tier`（`in512` / `in1024` × `output=256`）

| 档位 | `vLLM` 纯引擎 (`tok/s`) | `zstep` 3.6 (`tok/s`) | **`zstep` 3.7 (`tok/s`)** | 3.7 vs 3.6 / vs `vLLM` 纯引擎 | Mean TPOT `vPure` / `z3.6` / **`z3.7`** (`ms`) | Med TTFT `vPure` / `z3.6` / **`z3.7`** (`ms`) | P99 ITL `z3.6` / **`z3.7`** (`ms`) |
| :--- | ---: | ---: | ---: | :--- | :---: | :---: | :---: |
| `in512_c1` | `87.06 ± 0.01` | `103.99 ± 1.10` | **`101.61 ± 0.11`** | `-2.29%`（持平） / **`+16.71%`** | `10.97` / `8.93` / **`9.21`** | `141.86` / `184.93` / **`169.52`** (`-15.41 ms`) | `9.27` / **`9.46`** |
| `in512_c16` | `555.89 ± 1.34` | `727.06 ± 1.62` | **`741.71 ± 5.61`** | `+2.01%` / **`+33.43%`** | `25.32` / `20.01` / **`19.59`** | `1,086.45` / `386.86` / **`382.22`** | `19.09` / **`18.91`** |
| `in512_c128` | `1,126.34 ± 1.87` | `1,686.53 ± 7.96` | **`1,717.41 ± 4.17`** | `+1.83%` / **`+52.48%`** | `99.30` / `65.72` / **`64.47`** | `3,066.32` / `779.52` / **`769.26`** | `389.59` / **`384.61`** |
| `in1024_c1` | `84.57 ± 0.00` | `102.27 ± 0.18` | **`99.33 ± 0.04`** | `-2.87%`（持平） / **`+17.46%`** | `11.17` / `9.07` / **`9.43`** | `177.89` / `191.20` / **`172.49`** (`-18.71 ms`) | `9.47` / **`9.68`** |
| `in1024_c32` | `605.12 ± 1.35` | `778.39 ± 4.56` | **`796.36 ± 0.84`** | `+2.31%` / **`+31.60%`** | `42.27` / `36.51` / **`35.66`** | `2,127.63` / `391.62` / **`386.97`** | `391.34` / **`386.76`** |
| `in1024_c128` | `866.25 ± 1.06` | `1,169.71 ± 1.06` | **`1,189.11 ± 2.74`** | `+1.66%` / **`+37.27%`** | `122.27` / `92.32` / **`90.83`** | `3,107.53` / `787.39` / **`777.10`** | `394.26` / **`389.31`** |

**关键收获**：
* **单流首字延迟确定性改善、输出吞吐保持持平**：`c=1` 输出吞吐维持在 Phase 3.6 的 `±3%` 噪声带内（持平），而 `Median TTFT` 全线缩短 **`15.41..18.71 ms`**，其中 `in1024_c1` 从 `191.20 ms` 降至 **`172.49 ms`**，反超 `vLLM` 纯引擎的 `177.89 ms`（剩余 `c=1` `in=512` 与 Phase 3.5 的差距来自 `b12x` W4A8 MoE 在 `M >= 512` 采用静态专家主序 Prefill 核固定耗时 `~1.97 ms/层 × 40 ≈ 79 ms`，换取了每卡省下 `~10 GiB` 显存与并发解码大幅提速）；
* **中高并发吞吐与尾延迟同步提升（双机独立交叉复现一致）**：`c=16..128` 全部 6 个并发档位的输出吞吐较 Phase 3.6 一致正向上推 **`+1.66%..+3.16%`**（并在另一台同规格 `8 × RTX 6000D` 机器上独立复现，偏差 `<= 0.6%`），峰值输出吞吐达 **`1,717.41 tok/s`**（`in512_c128`，相对 `vLLM` 纯引擎 **`1.52×` / `+52.48%`**），峰值总吞吐达 **`5,945.57 tok/s`**（`in1024_c128`），且所有 `c >= 8` 并发档位的 `Mean TPOT`、`Median TTFT` 与 `P99 ITL` 同步下降。

---

## 8. `zstep#14` 跨卡通信、Mellanox CX-8 PCIe ACS 审计与 `done-wait` 保留实证（[`phase37_6000d_comm_audit.json`](benchmarks/phase37_6000d_comm_audit.json)）

针对 `zstep#14` 提出的 PCIe ACS 绕过、`shm_v5` 同步精简、自适应线程 / One-Shot 扩容及 `shm_v5 + NCCL_LOC` 混合路由方案，由可复现 8 卡审计脚本 [`bench_shmv5_nccl_comm_audit.py`](../worker/tests/phase3/bench_shmv5_nccl_comm_audit.py) 在 `8 × RTX 6000D` 上完成全矩阵实测与数值对齐（全形状 `0 ULP` 误差），归档于 [`phase37_6000d_comm_audit.json`](benchmarks/phase37_6000d_comm_audit.json)。

### 8.1 Mellanox ConnectX-8（`15b3:197c`）PCIe 桥 ACS 清零与 P2P 硬件根因审计

1. **Linux 内核参数核验**：常见文档提到的 `pcie_acs_override=downstream,multifunction` 是社区树外（out-of-tree VFIO）补丁，主线 Ubuntu `6.8` 内核符号表中不存在该解析器；主线内核对应参数为 `pci=disable_acs_redir=pci:15b3:197c`，运行时也可直接通过 `setpci -s <BDF> ECAP_ACS+06.w=0000` 清零桥端口 `ACSCtl`。
2. **主板与桥芯片拓扑事实**：8 个主板 Intel Root Port（`8086:352a`）出厂即 `ACSCtl = 0x0000`、IOMMU 未开启（`iommu_groups = 0`）、每张 `RTX 6000D` 已暴露 `128 GiB` BAR2 且工作站驱动 `can_device_access_peer == 1`；开启 `ACSCtl = 0x001d`（`SrcValid+ ReqRedir+ CmpltRedir+ UpstreamFwd+`）的全部是 **20 个 Mellanox ConnectX-8 集成 PCIe Switch 下行口（`15b3:197c`，共 32 个桥 PCI 功能）**。
3. **清零全部 20 个下行口 `ACSCtl = 0x0000` 并热重载 `nvidia.ko` 后的三维实测**：
   * **Copy Engine DMA（`cudaMemcpyPeer`）**：同颗 CX-8 桥内 `PXB` 对 `(0,1),(2,3),(4,5),(6,7)` 单向带宽达 **`107.9 GB/s`**（Gen5 x16 线速），跨桥同路 `NODE` 达 **`34.9 GB/s`**，跨双路 UPI `SYS` 达 **`31.1 GB/s`**；
   * **GPU SM 计算核直接 BAR MMIO（`ld.global` / `st.global`）**：在全部 4 组同桥 `PXB` GPU 对及全部 `ACSCtl` 掩码（`0x0000 / 0x000c / 0x001c / 0x001d`）下均表现为 **`W0R0`（写入为 `0`，读回 `0xFFFFFFFF` PCIe Unsupported Request 毒值）**——CX-8 网卡内置交换结构针对 NIC<->GPU RDMA 大包 DMA 设计，不转发计算核细粒度 peer BAR MMIO 事务，因此基于 SM 直接读写对端 BAR1 的 `NCCL` P2P / `p2p_allreduce` 内核在物理上不可行；
   * **多对双向并发 `NCCL` P2P**：跨二级 PCIe 桥与双路 UPI 并发双工 BAR1 流量触发 Non-Posted/Posted PCIe 信用依赖环死锁。测试完毕后已将全部 20 个端口恢复出厂 `ACSCtl = 0x001d` 并重载 `nvidia.ko`。

### 8.2 落地生产优化与 `8 × RTX 6000D` 全矩阵微基准（`done-wait` 严格保留）

在不改动 4 槽环形缓冲区入口 `my_done` 等待与出口 `peer_done` 释放写（`done-wait`）的前提下，[`shm_allreduce.py`](../worker/zstep_worker/kernels_sm120/shm_allreduce.py)、[`qwen35_moe.py`](../worker/zstep_worker/models/qwen35_moe.py) 与 [`deepseek_v41.py`](../worker/zstep_worker/models/deepseek_v41.py) 落地三项零风险优化：

1. **无状态自适应线程扩容（`256 -> 512`）+ One-Shot 阈值上调至 `64 KiB`（`DEFAULT_ONE_SHOT_MAX_VECS = 4096`）**：当默认 `threads == 256` 且单 CTA 分片 `vecs_per_cta > 256` 时，发射层自动将 `blockDim.x` 提升至 `512`（`gridDim.x = blocks` 与 `control[0..blocks)` 步进严格不变），彻底消除 `H=5120, M=4 / M=32` 及 `[32, 640] -> [32, 5120]` AllGather 中 `320 > 256` 导致的第二波 64 线程尾延迟空转，并将 `H=2048, M=16`（`64 KiB`）纳入单轮屏障 One-Shot。
2. **`Qwen35MoeStepRunner` Eager 大 `M` 混合 `NCCL_LOC` 路由**：新增可选 `self.nccl_group`（启动期 1 行预热消除首请求懒加载尖峰），`numel <= 320 * 5120`（`H=2048` 下 `M <= 800`）与全部 CUDAGraph 解码桶走 `shm_v5_prod`，`M >= 1024` Eager Prefill 自动走多 SM `NCCL_LOC`。
3. **`DeepSeekV41StepRunner` 混合切换点校准**：`ZSTEP_SHM_MAX_AR_NUMEL` 默认收紧至 `320 * 5120`（`H=5120` 下 `M <= 320` 走 `shm_v5_prod`、`M >= 384` Eager 走 `NCCL_LOC`）；`ZSTEP_SHM_MAX_AG_NUMEL` 默认上调至 `1024 * 4096`（覆盖 `M=1..4096` 全部 `[M, 640] -> [M, 5120]` AllGather，避免 `NCCL` `dim=0` 聚合后的额外 `permute().contiguous()` 显存拷贝）。

#### 表 8.2a：`H = 5120`（`DeepSeek-V4.1-Flash`）8 卡 BF16 AllReduce `p50`（`µs`）

| `M` | 载荷 | 算法 / 线程 | CUDAGraph `NCCL_LOC` | CUDAGraph 旧 `shm_v5` (`t=256, 40K`) | **CUDAGraph `shm_v5_prod` (`done-wait` 保留)** | `nodone` 消融 | Eager `NCCL_LOC` | Eager `shm_v5_prod` | **Eager 混合路由选中 (`p50`)** |
| ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `1` | `10 KiB` | One-Shot / `256` | `45.74` | `14.84` | **`14.91`** | `12.99` | `48.03` | `17.86` | `shm_v5_prod` (**`17.86`**) |
| `2` | `20 KiB` | One-Shot / `256` | `47.35` | `15.52` | **`15.67`** | `13.68` | `49.95` | `17.98` | `shm_v5_prod` (**`17.98`**) |
| `4` | `40 KiB` | One-Shot / **`512`** | `51.17` | `24.58` | **`19.82`** (`-19.4%`) | `17.50` | `52.29` | `22.08` | `shm_v5_prod` (**`22.08`**) |
| `8` | `80 KiB` | Two-Shot / `512` | `53.40` | `30.83` | **`32.11`** | `29.90` | `54.34` | `34.37` | `shm_v5_prod` (**`34.37`**) |
| `16` | `160 KiB` | Two-Shot / `512` | `75.02` | `35.80` | **`36.20`** | `33.96` | `76.67` | `38.46` | `shm_v5_prod` (**`38.46`**) |
| `32` | `320 KiB` | Two-Shot / **`512`** | `114.96` | `60.99` | **`49.33`** (`-19.1%`) | `47.47` | `116.86` | `52.13` | `shm_v5_prod` (**`52.13`**) |
| `64` | `640 KiB` | Two-Shot / `512` | `224.92` | `93.29` | **`92.11`** | `90.07` | `226.24` | `95.10` | `shm_v5_prod` (**`95.10`**) |
| `128` | `1.25 MiB` | Two-Shot / `512` | `431.48` | `160.20` | **`157.98`** | `156.31` | `433.09` | `160.35` | `shm_v5_prod` (**`160.35`**) |
| `256` | `2.50 MiB` | Two-Shot / `512` | `319.15` | `307.30` | **`302.24`** | `300.24` | `321.66` | `304.67` | `shm_v5_prod` (**`304.67`**) |
| `320` | `3.125 MiB` | Two-Shot / `512` | — | — | — | — | `400.38` | `383.90` | `shm_v5_prod` (**`383.90`**) |
| `384` | `3.75 MiB` | Two-Shot / `512` | — | — | — | — | `450.46` | `450.11` | `nccl_loc` (**`450.46`**) |
| `512` | `5.00 MiB` | Two-Shot / `512` | — | — | — | — | **`584.58`** | `594.85` | `nccl_loc` (**`584.58`**) |
| `1024` | `10.0 MiB` | Two-Shot / `512` | — | — | — | — | **`1,067.81`** | `1,175.04` | `nccl_loc` (**`1,067.81`**) |
| `2048` | `20.0 MiB` | Two-Shot / `512` | — | — | — | — | **`2,002.78`** | `2,335.30` | `nccl_loc` (**`2,002.78`**) |
| `4096` | `40.0 MiB` | Two-Shot / `512` | — | — | — | — | **`3,978.82`** | `4,662.85` | `nccl_loc` (**`3,978.82`**) |

#### 表 8.2b：`H = 2048`（`Qwen3.5-35B-A3B`）8 卡 BF16 AllReduce 与 `[M, 640] -> [M, 5120]` AllGather 关键档位 `p50`（`µs`）

* **`H = 2048` AllReduce**：CUDAGraph `M=1, 8, 16, 32, 128, 256` 下 `shm_v5_prod` 为 **`14.50 / 17.30 / 27.09 / 34.56 / 70.82 / 128.69 µs`**，全桶显著快于 `NCCL_LOC`（`29.65 / 34.76 / 36.83 / 55.87 / 174.32 / 342.71 µs`），其中 `M=16`（`64 KiB`）较旧 `40 KiB` 阈值（`30.61 µs`）加速 **`11.5%`**；Eager 模式下 `M=512` 由 `shm_v5_prod` 胜出（**`246.69 vs 285.54 µs`**），`M=1024 / 2048 / 3072 / 4096` 自动切至 `NCCL_LOC`（**`472.19 / 878.75 / 1,248.16 / 1,639.49 µs`** vs `shm_v5_prod` `478.69 / 943.04 / 1,407.55 / 1,874.43 µs`）。
* **`[M, 640] -> [M, 5120]` AllGather**：CUDAGraph `M=1, 16, 32, 128, 256` 下 `shm_ag_prod` 为 **`15.26 / 16.69 / 19.82 / 58.43 / 109.22 µs`**（对比 `NCCL_LOC` `24.95 / 43.16 / 63.75 / 106.87 / 174.15 µs`，其中 `M=32` 借助自适应 `512` 线程从旧 `25.55 µs` 降至 **`19.82 µs`**，加速 **`22.4%`**）；Eager `M=512..4096` 因原生沿 `dim=-1` 一步写出而无需额外转置拷贝，`shm_ag_prod` 全程领先 `NCCL_LOC`（`M=2048`: **`816.74 vs 1,032.48 µs`**；`M=4096`: **`1,623.84 vs 2,061.66 µs`**）。

### 8.3 为什么保留 `done-wait` 并排除 `64-bit` 自同步包（`pkt64`）：端到端 Amdahl 消融实证

1. **排除 `64-bit` `(epoch32 << 32) | payload32` 自同步包（`pkt64`）**：实测 `pkt64` 虽省去独立到达旗标，但将有效载荷带宽开销放大 **`2×`** 并把 `16 B` `uint4` 合并写拆碎为 `7 × 4` 次跨卡 `8 B` 事务，`H=2048, M=1` CUDAGraph 即退化至 `39.69 µs`（慢于 `shm_v5_prod` `14.50 µs` 达 `2.74×`），`M >= 4` 更因 PCIe 写放大退化至 `656..5,832 µs`。
2. **保留 `done-wait`（入口 `my_done` 自旋 + 出口 `peer_done` 释放写）的工程决策**：
   * 从表 8.2a 可见，彻底移除首尾 `done-wait`（`nodone` 消融）在单次 AllReduce 上仅节省 **`~1.7..2.3 µs/call`**（全模型 80 次通信理论上限仅 `~0.14..0.18 ms/step`）；
   * 在 `DeepSeek-V4.1-Flash (TP=8)` Warm `n=3` 端到端实测消融（[`phase37_6000d_comm_audit.json`](benchmarks/phase37_6000d_comm_audit.json) `dsv41_e2e_amdahl_ablation`，`prime_squares_table` `128/128` 100% 精确一致，`TRITON_RECOMPILATIONS=0`）中，`nodone + 自适应线程` 相对 Phase 3.7 `main` 在 `c=1 / c=8 / c=16 / c=128` 的吞吐变化仅为 **`+0.23% .. +2.36%`**（全部落在 `<3%` GPU Boost 频率正常波动带内），唯有 `c=32` 出现 **`+2.39% .. +3.00%`**（Suite 1 `c32`: `820.22 ± 1.82 -> 844.80 ± 3.07 tok/s`，Suite 2 `in1024_c32`: `796.36 ± 0.84 -> 815.37 ± 3.65 tok/s`）的可复现增益——而这部分增益恰恰来自**已保留落地的自适应 `512` 线程扩容**（`M=32` 单次 AR `-11.66 µs` + AG `-5.73 µs`，每步 80 AR + 40 AG 立省 `~1.16 ms`），而非去掉 `done-wait`；
   * 保留 `done-wait` 以不到 `<2%` 的端到端理论代价，换取了 **4 槽环形缓冲区对未来多 CUDA Stream（如通信/计算双流重叠、异步投机采样）的无条件防覆盖安全不变量**，彻底杜绝潜在跨拍静默覆盖风险。

---

## 9. Phase 3.8：SM120 `FlashInfer` 实验算子筛选与 `TP=8` 端到端实测（[`phase38_6000d_flashinfer_ops_audit.json`](benchmarks/phase38_6000d_flashinfer_ops_audit.json)）

基于最新 `FlashInfer` SM120 实验分支（`ae58183fff34c545b3e3020d81d591f2a79658fc`；纯源码包现已内置于 [`worker/third_party/flashinfer_sm120_overlay_ae58183f.tar.xz`](../worker/third_party/flashinfer_sm120_overlay_ae58183f.tar.xz)，单算子筛选生成脚本：[`bench_sm120_flashinfer_exp_ops.py`](../worker/tests/phase3/bench_sm120_flashinfer_exp_ops.py)），在 `RTX 6000D`（`156-SM`, `sm_120`）上对全部 4 组与 `DeepSeek-V4.1-Flash` 匹配的候选算子逐项进行**单算子微基准 + 数值对齐 + `TP=8` 全模型 Warm `n=3` 端到端 A/B 审计**，遵循“只合入实测胜出且数值对齐的算子”原则：

### 9.1 候选算子逐项微基准筛选与去留决策

| 候选算子 / 组件 | `RTX 6000D` (`156-SM`) 实测表现与数值对齐 | 决策 |
| :--- | :--- | :--- |
| **`b12x` 更新（`79819b11 -> ae58183f`）+ 源码指纹缓存隔离** | CUDAGraph `M=1..128` `fused_moe` 持平（`36.95..960.41 µs`）；`block_fp8_linear` 在 `3200×6144`（`kv_b_proj`）大 `M=512 / 2048` 提升 **`+9.8% / +7.7%`**（`135.58 -> 123.47 µs`、`344.43 -> 319.70 µs`）。新增 `_compute_b12x_source_fingerprint()` 对源码树计算 SHA256，自动隔离不同版本 `b12x` 的磁盘 autotune 缓存。 | **保留合入** |
| **`sm12x_mega_gate`（融合 `[M, 5120]×[384, 5120]` GEMM + `bias` + `bias_vl` + `sqrtsoftplus` + `8-group top-4 -> top-6`）** | `M=1024` Top-6 专家精确一致率 **`100.00%`**（权重最大绝对误差 `1.94e-7`）；支持 `vision_n_layers > 0` 的 `bias_vl`（`has_image_bias=True` + 静态全 `False` `img_mask_bool`）、`enable_pdl=False` 及 Rank-0 预编译 TVM-FFI 缓存。因下游 `b12x.moe.fused_moe` 要求 `int32` `topk_ids`（直接传 `int64` 会产生 `max_diff=64.0` 步幅错位），需追加一次 `int64 -> int32` 转换（`0.8..2.0 µs/层`）：单层 `M=1..8` 无净收益，在 `M=16` CUDAGraph Decode 实测轻微拖累 `c16` 吞吐（`635.04 vs 638.15 tok/s`），而在 **`M >= 32` Decode 净省 `80.8..81.1 µs/step`、Eager Prefill `M=256..3072` 加速 `1.35×..2.96×`（40 层净省 `0.81..1.43 ms/step`）**。因此默认门限钉死为 **`DEFAULT_MEGA_GATE_MIN_TOKENS = 32`**。 | **保留合入（`M >= 32`）** |
| **`sm120_dsa_indexer` 分页 Decode（`dsa_indexer_det_topk_paged`）** | 单次融合发射完成分页 FP8 MQA Logits + 确定性 Top-512 选择（`logits_max_diff = 0.0`、`ctx=256..16384` Top-512 集合重合率 **`100.00%`**、零长 padding 行全 `-1`）。**全模型 8 层 Indexer（3×`cr=2` + 1×L20 + 4×L24..36）CUDAGraph 步延迟**：`ctx=640, B=1..128` 从 `125.7..222.1 µs` 降至 **`34.8..51.2 µs`（`3.55×..4.33×`，省 `90.9..170.8 µs`）**；`ctx=4096, B=1..128` 从 `154.1..322.1 µs` 降至 **`44.9..137.2 µs`（`2.35×..3.43×`，省 `99.7..185.0 µs`）**；`ctx=16384, B=1..128` 从 `180.2..641.5 µs` 降至 **`60.2..420.1 µs`（省 `120.0..221.4 µs`）**；同时在 Python 步入口跳过图外 4 次 DeepGEMM `get_paged_mqa_logits_metadata` 调度核发射，并将 `decode_graph_max_seq_len` 收紧至 `16384`（`>16384` 自动回退 Eager 原生双层 `candidate_blocks`）。 | **保留合入（Decode `compressed_ctx <= 16384`）** |
| **独立 `sm120_paged_mqa_logits`** | 内部固定 2 次 kernel launch，在 `ctx=640..4096` 下耗时 `8.23..22.23 µs`，慢于既有 DeepGEMM 单次发射 `fp8_paged_mqa_logits`（`4.06..16.53 µs`）。 | **淘汰不合入** |
| **`sm120_dsa_indexer` Ragged Prefill** | 在 `1×512..4×1024`（`cr=1` 与 `cr=2`）共 9 组常见 Prefill 形状中有 8 组慢于既有 `fp8_fp4_mqa_logits + top_k_per_row_prefill`（`0.50×..0.96×`，如 `1×512` `51.9 vs 25.9 µs`、`1×1024` `67.0 vs 42.0 µs`）。 | **淘汰不合入** |

### 9.2 `TP=8` 全模型 Warm `n=3` 端到端对标（同一代码基线 `main` `71c9814` 同机同日复测）

所有运行均通过 `prime_squares_table` `128/128`（`single` / `batched` / `chunked` 100% 逐 token 一致）、`TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=5`，退出后 8 卡显存 `0 MiB` 零泄漏。

> **门限说明与统计边界**：在套件二（`in512_c1 / c16 / c128`、`in1024_c1 / c32 / c128`）与套件三（`in4096_c1 / c16`）中，落地的 `min_m=32` 与 `min_m=16` 在全部 `c=1`、`c=32`、`c=128` 档位执行**逐算子完全相同的调度路径**（`M=1` 走原生门控、`M >= 32` 走 `sm12x_mega_gate`），仅 `M=16` 解码步由 `min_m=32` 避开 `int64 -> int32` 转换开销（套件一实测 `c16` 达到全部变体最高值 **`638.15 ± 2.95 tok/s`**）。按 `<3%` 单档频率波动持平规则，单档相对增幅（`+0.15% .. +2.70%`）量级符合算子物理节拍账本（每步净省 `~0.10..0.22 ms` / `~9.0 ms` Decode 步与 `~0.8..1.4 ms` Prefill 步），但在**全部 3 套基准共 12 个并发档位上实现 `12 / 12` 全线正向推进、零回退**：

#### 表 9.2a：套件一 `512 -> 128`（`c=1, 8, 16, 32`）门限消融与落地配置（Warm `n=3`）

| 档位 | Base (`main` `71c9814` + 旧 `b12x`) | `Cand (min_m=16)` | `Cand (min_m=257, 仅 Prefill)` | **落地 `Cand (min_m=32)`** | 落地 vs Base (`tok/s`) | Mean TPOT Base / **落地** (`ms`) | Med TTFT Base / **落地** (`ms`) | P99 ITL Base / **落地** (`ms`) |
| :--- | ---: | ---: | ---: | ---: | :--- | :---: | :---: | :---: |
| `c1` | `95.35 ± 0.35` | `97.93 ± 0.03` | `97.60 ± 0.40` | **`97.47 ± 0.50`** | **`+2.22%`** | `9.23` / **`9.03`** | `169.94` / **`166.09`** (`-3.85 ms`) | `9.45` / **`9.37`** |
| `c8` | `442.72 ± 0.26` | `451.14 ± 2.31` | `447.81 ± 1.55` | **`448.96 ± 1.38`** | **`+1.41%`** | `14.23` / **`13.99`** | `505.89` / **`504.19`** (`-1.70 ms`) | `14.83` / **`14.51`** |
| `c16` | `630.91 ± 1.93` | `635.04 ± 3.14` | `637.12 ± 2.14` | **`638.15 ± 2.95`** | **`+1.15%`（全变体最优）** | `21.74` / **`21.46`** | `381.07` / **`379.83`** (`-1.24 ms`) | `323.42` / **`322.86`** |
| `c32` | `841.71 ± 2.47` | `843.04 ± 5.93` | `842.46 ± 4.70` | **`842.79 ± 0.67`** | `+0.13%` | `33.23` / **`33.20`** | `381.58` / **`380.68`** (`-0.90 ms`) | `381.33` / **`380.35`** |

#### 表 9.2b：套件二 `six_tier` 与套件三 `in4096` 长上下文对标（Warm `n=3`）

| 套件 / 档位 | Base (`71c9814`) Output (`tok/s`) | **Phase 3.8 Cand** Output (`tok/s`) | Cand vs Base | Total Throughput Base / **Cand** (`tok/s`) | Mean TPOT Base / **Cand** (`ms`) | Med TTFT Base / **Cand** (`ms`) | P99 ITL Base / **Cand** (`ms`) |
| :--- | ---: | ---: | :--- | :---: | :---: | :---: | :---: |
| Suite 2 `in512_c1` | `101.80 ± 0.06` | **`103.82 ± 0.03`** | **`+1.98%`** | `305.41` / **`311.46`** | `9.18` / **`9.01`** | `172.50` / **`167.72`** (`-4.78 ms`) | `9.40` / **`9.30`** |
| Suite 2 `in512_c16` (`m257` / `m16`) | `750.84 ± 1.29` | **`755.17 ± 7.84`** / `751.24 ± 2.81` | `+0.58%` / `+0.05%` | `2,252.53` / **`2,265.51`** | `19.33` / **`19.22`** | `381.18` / **`380.14`** (`-1.04 ms`) | `18.48` / **`18.23`** |
| Suite 2 `in512_c128` | `1,729.34 ± 6.43` | **`1,731.94 ± 5.14`** | `+0.15%` | `5,188.02` / **`5,195.82`** | `63.99` / **`63.91`** | `767.34` / **`765.41`** (`-1.93 ms`) | `383.44` / **`382.58`** |
| Suite 2 `in1024_c1` | `99.36 ± 0.13` | **`101.55 ± 0.04`** | **`+2.20%`** | `496.80` / **`507.74`** | `9.41` / **`9.22`** | `175.39` / **`170.31`** (`-5.08 ms`) | `9.63` / **`9.46`** |
| Suite 2 `in1024_c32` | `810.75 ± 0.82` | **`820.15 ± 3.00`** | **`+1.16%`** | `4,053.74` / **`4,100.74`** | `34.90` / **`34.51`** | `386.25` / **`384.95`** (`-1.30 ms`) | `386.06` / **`384.78`** |
| Suite 2 `in1024_c128` | `1,193.42 ± 4.09` | **`1,201.05 ± 1.13`** | `+0.64%` | `5,967.11` / **`6,005.25`** | `90.46` / **`89.85`** | `775.50` / **`772.96`** (`-2.54 ms`) | `388.53` / **`387.17`** |
| Suite 3 `in4096_c1` | `73.62 ± 0.07` | **`74.84 ± 0.09`** | **`+1.66%`** | `2,429.32` / **`2,469.67`** | `9.45` / **`9.23`** | `538.28` / **`537.69`** (`-0.59 ms`) | `9.68` / **`9.53`** |
| Suite 3 `in4096_c16` | `195.74 ± 0.58` | **`196.54 ± 0.44`** | `+0.41%` | `6,459.38` / **`6,485.95`** | `58.49` / **`58.22`** | `1,175.59` / **`1,171.84`** (`-3.75 ms`) | `393.11` / **`391.81`** |

---

## 10. Phase 3.10：内置可复现 SM120 实验 Overlay、`ZSTEP_REQUIRE_SM120_EXP=1` 强门禁与合入 `#20`/`#16`/`#19` 后最新 `TP=8` 审计（[`phase310_6000d_dsv41_post_route_a_audit.json`](benchmarks/phase310_6000d_dsv41_post_route_a_audit.json)）

### 10.1 解决外部实验引用不可达与静默回退

在 Phase 3.8（§9）之后，主线先后合入 **PR `#20` Part A1**（`shm_v5` 非交错对端 PCIe 预取，`H=5120, M=1` AllReduce 由 `17.15` 降至 `12.46 µs`，每步 80 次省 `~0.38 ms`）与 **PR `#16`**（单次页锁 staging）。这带来两个可复现性缺口：
1. 若运行环境未挂载 `FlashInfer` `ae58183f` 实验算子，代码会静默回退至 `mega_gate=off dsa_decode=off`，但因 `#20` 通信加速，回退路径本身已跑出高于 §9.2 归档表的吞吐（如 `c1` `~98.60..99.24 tok/s` vs 旧 `cand` `97.47 tok/s`），导致外部复核无法区分“实验算子是否生效”；
2. 公开 fork 未对 `ae58183fff34c545b3e3020d81d591f2a79658fc` 打持久 tag。

为此仓库完成两项固化：
* **内置确定性纯源码 Overlay 包**：将 `ae58183f` 所需的全部 Apache-2.0 纯 Python/CUDA/CuTe-DSL 源码打包进 [`worker/third_party/flashinfer_sm120_overlay_ae58183f.tar.xz`](../worker/third_party/flashinfer_sm120_overlay_ae58183f.tar.xz)（`2.0 MiB`，`SHA256 = 90e4db7f52f725e969de4f9d3c386112122d77db15bafa3a30515d190660cae4`，说明见 [`worker/third_party/README.md`](../worker/third_party/README.md)），并提供一键校验解压与容器挂载参数生成脚本 [`scripts/prepare_sm120_overlay.sh`](../scripts/prepare_sm120_overlay.sh)（自动挂载 `b12x` + `_compat` 顶层别名 + `sm12x_mega_gate` + `sm120_dsa_indexer` + TVM-FFI/CuTe-DSL 头文件而不遮蔽镜像原生 `flashinfer.utils`）。
* **新增 `ZSTEP_REQUIRE_SM120_EXP=1` 强门禁**（[`b12x_dsv41.py`](../worker/zstep_worker/kernels_sm120/b12x_dsv41.py)）：开启后若 `b12x`、`sm12x_mega_gate` 或 `sm120_dsa_indexer` 任一组件缺失或被禁用，初始化期立即抛出 `RuntimeError` 终止，杜绝基准审计中的静默回退（`./scripts/prepare_sm120_overlay.sh --print-docker-args` 默认注入 `-e ZSTEP_REQUIRE_SM120_EXP=1`）。

### 10.2 合入 `#20`/`#16`/`#19` 后 `DeepSeek-V4.1-Flash` (`TP=8`, `8 × RTX 6000D`) Warm `n=3` 最新实测

完全通过仓库内置 `./scripts/prepare_sm120_overlay.sh` 启动 `vllm/vllm-openai:v0.31.0`（日志确认 `b12x=on(moe=40,lin=210/6) mega_gate=on(40/17,min_m=32) dsa_decode=on(cg_max_seq=16384)`、`TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=5`、`prime_squares_table` 单序列/批处理/`chunk=24` 全 `128/128` 100% 匹配）：

#### 表 10.2a：套件一 `512 -> 128`（`c=1, 8, 16, 32`）同二进制受控对照：回退路径 vs 内置 Overlay（Warm `n=3`）

| 档位 | §9.2 Phase 3.8 旧 `Cand` (`tok/s`) | 当前 `main` 仅 `b12x` 回退 (`mega=off, dsa=off`) | **当前 `main` + 内置 Overlay (`mega=on, dsa=on`)** | Overlay vs 回退 / vs Phase 3.8 | Mean TPOT 回退 -> **Overlay** (`ms`) | Med TTFT 回退 -> **Overlay** (`ms`) | P99 ITL 回退 -> **Overlay** (`ms`) |
| :--- | ---: | ---: | ---: | :--- | :---: | :---: | :---: |
| `c1` | `97.47 ± 0.50` | `98.60 ± 0.01` | **`100.44 ± 0.13`** | **`+1.87%` / `+3.05%`** | `8.85` -> **`8.70`** | `173.51` -> **`169.29`** (`-4.22 ms`) | `9.13` -> **`8.99`** |
| `c8` | `448.96 ± 1.38` | `467.33 ± 1.74` | **`470.84 ± 1.08`** | `+0.75%` / **`+4.87%`** | `13.27` -> **`13.15`** | `505.88` -> **`504.12`** (`-1.76 ms`) | `13.79` -> **`13.62`** |
| `c16` | `638.15 ± 2.95` | `644.97 ± 1.39` | **`649.06 ± 1.40`** | `+0.63%` / **`+1.71%`** | `21.18` -> **`21.04`** | `380.48` -> **`379.40`** (`-1.08 ms`) | `323.41` -> **`322.85`** |
| `c32` | `842.79 ± 0.67` | `839.05 ± 1.48` | **`840.34 ± 4.43`** | `+0.15%` / `-0.29%`（持平） | `33.36` -> **`33.32`** | `381.71` -> **`380.33`** (`-1.38 ms`) | `381.32` -> **`380.05`** |

* **正交叠加验证**：`#20` Part A1 的 `shm_v5` 预取将 `b12x`-only 回退路径单流 `TPOT` 从旧 `9.23 ms` 压至 `8.85 ms`（`98.60 tok/s`）；在此之上挂载内置 `ae58183f` Overlay 后，`sm120_dsa_indexer` 与 `sm12x_mega_gate` 继续正交省去 `~0.15 ms/step` 解码与 `~4.22 ms` Prefill 开销，使 `DeepSeek-V4.1-Flash` `TP=8` 在无 P2P PCIe 上的 `512->128` 单流输出吞吐首次突破百字大关达 **`100.44 ± 0.13 tok/s`**（`Mean TPOT = 8.70 ms`，相对 `vLLM 0.31.0` 纯引擎 `83.59 tok/s` 领先 **`+20.16%`**）。

#### 表 10.2b：套件二 `six_tier` 与套件三 `in4096` 内置 Overlay 最新主结果（Warm `n=3`）

| 套件 / 档位 | §9.2 Phase 3.8 (`tok/s`) | **当前 `main` + 内置 Overlay (`tok/s`)** | 相对 Phase 3.8 | Total Throughput (`tok/s`) | Mean TPOT (`ms`) | Med TTFT (`ms`) | P99 ITL (`ms`) |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| Suite 2 `in512_c1` | `103.82 ± 0.03` | **`107.03 ± 0.08`** | **`+3.09%`** | **`321.09`** | **`8.72`** | **`168.39`** | **`8.99`** |
| Suite 2 `in512_c16` | `755.17 ± 7.84` | **`765.98 ± 2.33`** | **`+1.43%`** | **`2,297.93`** | **`18.92`** | `380.23` | **`18.10`** |
| Suite 2 `in512_c128` | `1,731.94 ± 5.14` | **`1,733.34 ± 6.14`** | `+0.08%` | **`5,200.02`** | **`63.86`** | **`764.90`** | **`382.44`** |
| Suite 2 `in1024_c1` | `101.55 ± 0.04` | **`104.97 ± 0.08`** | **`+3.37%`** | **`524.84`** | **`8.89`** | `171.18` | **`9.14`** |
| Suite 2 `in1024_c32` | `820.15 ± 3.00` | **`822.35 ± 3.58`** | `+0.27%` | **`4,111.75`** | **`34.50`** | `384.96` | `384.86` |
| Suite 2 `in1024_c128` | `1,201.05 ± 1.13` | **`1,200.63 ± 1.88`** | `-0.03%` | **`6,003.15`** | `89.99` | **`772.83`** | `387.25` |
| Suite 3 `in4096_c1` | `74.84 ± 0.09` | **`76.63 ± 0.06`** | **`+2.39%`** | **`2,528.76`** | **`8.92`** | **`537.59`** | **`9.18`** |
| Suite 3 `in4096_c16` | `196.54 ± 0.44` | **`197.54 ± 0.84`** | `+0.51%` | **`6,518.90`** | **`57.83`** | **`1,171.48`** | `391.90` |


