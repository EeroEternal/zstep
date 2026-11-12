# `zstep` Master Benchmark Ledger (`sm_120` Non-P2P PCIe: `8×32 GB` & `8×85.6 GB`)

本目录归档 Phase 0（`sm_120` 算子与 PCIe AllReduce）、Phase 1（Mode B Rust 控制面与双缓冲调度）、Phase 2（`8 × RTX 5090 32 GB` 上 `Qwen/Qwen3.5-35B-A3B` 无 APC 真权重 serving 与 `vLLM 0.33.0` 对标）及 **Phase 3 / 3.5（`8 × RTX 6000D 85.6 GB` 上 `DeepSeek-V4.1-Flash` `TP=8` 无 APC 真权重 serving 与 `vLLM 0.31.0` 对标）** 的实测 JSON。所有主结果均为 **warm、单配置、无前缀缓存（`--no-enable-prefix-caching`）、按 `(round, tier)` 独立种子、round-major 交错、`n=3`**；旧单轮与消融数据只作历史记录，不与主表混用。

## 1. 硬件、软件与统计边界

```text
[Host A — Phase 0/1/2]
GPU:       8 × NVIDIA GeForce RTX 5090 (sm_120, 170 SMs, 32 GB GDDR7 per GPU)
Topology:  4-NUMA PCIe Gen5 ([0,0,0,1,2,2,2,3]), no NVLink, can_device_access_peer == 0
Model:     Qwen/Qwen3.5-35B-A3B, BF16, TP=8
Baseline:  vLLM 0.33.0, BF16, TP=8, --no-enable-prefix-caching, isolated per-(round,tier) seeds, degraded NCCL SHM

[Host B — Phase 3 / 3.5]
GPU:       8 × NVIDIA RTX 6000D (sm_120, 156 SMs, 85,651 MiB GDDR7 per GPU, Driver 595.99.02)
Topology:  2-NUMA PCIe Gen5 ([0,0,0,0,1,1,1,1]), no NVLink, can_device_access_peer == 0 (CNS)
Model:     DeepSeek-V4.1-Flash (1017, 475.25 GiB MXFP8/MXFP4/BF16, engram cpu_offload=false), TP=8
Baseline:  vllm/vllm-openai:v0.31.0, TP=8, --no-enable-prefix-caching, isolated per-(round,tier) seeds, NCCL_P2P_DISABLE=1, NCCL_P2P_LEVEL=LOC, symmetric SM120 64-tok SWA shim
```

所有跨卡结论严格限定于对应无 P2P PCIe 拓扑，不外推到 NVLink 或 PCIe P2P。表中 `std` 是三轮重复测量的**总体标准差**；完整 `mean/std/min/max` 以 JSON 为准。吞吐“持平”规则为：相对差 `<3%`，或均值差 `<=2×sqrt(zstep_std²+vllm_std²)`；`n=3` 的持平不是严格等价证明。

## 2. 归档文件总览

共有 **22 份主 JSON + 19 份 vLLM 逐轮/warmup 原始 JSON = 41 份 JSON**。

| 阶段 | 文件 | 内容 |
| :--- | :--- | :--- |
| Phase 0.1 | [`step_00_env_sm120.json`](step_00_env_sm120.json) | 硬件拓扑、P2P 矩阵、算子环境 |
| Phase 0.2 | [`step_01_dense_fp8_gemm.json`](step_01_dense_fp8_gemm.json) | RMSNorm + MXFP8、稠密 GEMM、大偏移与零重编译 |
| Phase 0.3 | [`step_03_moe_mxfp4_sm120.json`](step_03_moe_mxfp4_sm120.json) | 384 专家路由、MXFP4 MoE、`>2^31` 偏移 |
| Phase 0.4 | [`step_04_dsv41_attention_sm120.json`](step_04_dsv41_attention_sm120.json) | 分页稀疏 MLA Decode / Verify / Extend |
| Phase 0.5a | [`step_05a_mhc_engram_sm120.json`](step_05a_mhc_engram_sm120.json) | mHC 与 Engram VRAM / 主机 UVA |
| Phase 0.5b | [`step_05b_pcie_allreduce_8gpu.json`](step_05b_pcie_allreduce_8gpu.json) | `H=5120` NUMA First-Touch AllReduce vs NCCL（4-NUMA 5090） |
| Phase 0.5c | [`phase0_6000d_ar_shmv5.json`](phase0_6000d_ar_shmv5.json) | **`8 × RTX 6000D`** `H=5120, M=1..4096` `shm_v5` Per-Rank NUMA AllReduce vs `shm_v4` / `NCCL_LOC` |
| Phase 1 | [`phase1_mode_b_1gpu.json`](phase1_mode_b_1gpu.json) | 1 GPU、200 步 Mode B IPC 与重叠实测 |
| Phase 1 | [`phase1_mode_b_8gpu.json`](phase1_mode_b_8gpu.json) | 8 GPU、200 步 Mode B IPC 与重叠实测 |
| Phase 2 | [`phase2_allreduce_h2048_8gpu.json`](phase2_allreduce_h2048_8gpu.json) | `H=2048, blocks=8, M=1..8192` 全尺寸 AllReduce 与 block 扫描 |
| Phase 2 | [`phase2_zstep_qwen35_35b_a3b_tp8.json`](phase2_zstep_qwen35_35b_a3b_tp8.json) | **Phase 2 主结果**：`Qwen3.5-35B-A3B` 统一 4K 配置、按 `(round, tier)` 独立种子、六档交错 `n=3` |
| Phase 2 | [`phase2_vllm_qwen35_35b_a3b_tp8.json`](phase2_vllm_qwen35_35b_a3b_tp8.json) | `vLLM 0.33.0`（`--no-enable-prefix-caching`、独立种子）六档交错 `n=3` 基线与硬件 provenance |
| Phase 2 | [`phase2_vllm_raw/`](phase2_vllm_raw/) | 18 份无 APC 逐轮独立种子结果与 1 份 warmup 原始结果 |
| Phase 2 | [`phase2_vllm_greedy_reference.json`](phase2_vllm_greedy_reference.json) | `Qwen3.5-35B-A3B` 5 个多 prompt（含 128-token 无循环算术表）`vLLM` 贪心参考及 Top-2 logprob |
| Phase 2 | [`phase2_greedy_parity_report.json`](phase2_greedy_parity_report.json) | `Qwen3.5-35B-A3B` 单序列/`bucket=8`/`chunk=24` 双端 Top-2 贪心审计及 17-kernel 零重编译 |
| Phase 2 | [`phase2_zstep_qwen35_35b_a3b_tp8_step4k.json`](phase2_zstep_qwen35_35b_a3b_tp8_step4k.json) | 旧单轮 4K 配置，仅作历史对照 |
| Phase 2 | [`phase2_qwen35_optimization_ablation.json`](phase2_qwen35_optimization_ablation.json) | 四阶段工程优化消融与稠密模型参考 |
| Phase 3.5 | [`phase3_zstep_dsv41_tp8_in512_out128.json`](phase3_zstep_dsv41_tp8_in512_out128.json) | **Phase 3.5 套件一主结果**：`DeepSeek-V4.1-Flash` `512->128` 四档（`c=1,8,16,32`）无缓存独立种子 `n=3` |
| Phase 3.5 | [`phase3_vllm_dsv41_tp8_in512_out128.json`](phase3_vllm_dsv41_tp8_in512_out128.json) | `vLLM 0.31.0`（`--no-enable-prefix-caching`）`512->128` 四档独立种子 `n=3` 基线 |
| Phase 3.5 | [`phase3_zstep_dsv41_tp8_6tier.json`](phase3_zstep_dsv41_tp8_6tier.json) | **Phase 3.5 套件二主结果**：`DeepSeek-V4.1-Flash` 六档交错无缓存独立种子 `n=3` |
| Phase 3.5 | [`phase3_vllm_dsv41_tp8_6tier.json`](phase3_vllm_dsv41_tp8_6tier.json) | `vLLM 0.31.0`（`--no-enable-prefix-caching`）六档交错独立种子 `n=3` 基线 |
| Phase 3.5 | [`phase3_vllm_greedy_reference.json`](phase3_vllm_greedy_reference.json) | `DeepSeek-V4.1-Flash` 5 条多 prompt 无缓存 `vLLM 0.31.0` 贪心参考（含 Top-5 logprobs） |
| Phase 3.5 | [`phase3_greedy_parity_report.json`](phase3_greedy_parity_report.json) | `DeepSeek-V4.1-Flash` 单序列/批处理/Mixed-Chunked 全局 Top-2 贪心审计及 5-kernel 零重编译 |

---

## 3. Phase 3.5：`DeepSeek-V4.1-Flash` (`TP=8`, `8 × RTX 6000D`) 无 APC、独立种子、Warm `n=3` 主结果

两套基准共用唯一常驻配置：`allreduce_blocks=8`、`num_kv_pages=2560`、`max_seqs=128`、`max_step_tokens=max_decoder_rows=4096`、`max_prefill_chunk=2048`、`max_mixed_step_tokens=3072`、`max_pages_per_seq=256`，`M<=256` 走 `shm_v5 (b=8)`、`M>256` Eager 步走 `NCCL_LOC`，16 warmup prompts，round-major 交错 `n=3`（与 `--no-enable-prefix-caching` 的 `vLLM 0.31.0` 同享 `(round, tier)` 独立种子）。运行时审计为 **`TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=5`**，退出后 `/dev/shm` 无残留。详细架构与物理归因见 **[`07-phase3-dsv41-6000d-benchmark.md`](../07-phase3-dsv41-6000d-benchmark.md)** 与 **[`08-phase3-dsv41-optimization-plan.md`](../08-phase3-dsv41-optimization-plan.md)**。

### 3.1 套件一：`512 -> 128` 四档并发（`c=1, 8, 16, 32`）

| 档位 | zstep Output (`tok/s`) | vLLM Output (`tok/s`) | 均值差 / 判定 | Mean TPOT z / v (`ms`) | z Median TTFT / submit-TTFT (`ms`) | v Median TTFT (`ms`) | z P99 ITL (`ms`) | v P99 ITL (`ms`) |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| `c1` | **96.18 ± 0.57** | 83.97 ± 0.05 | **+14.53% / zstep 更快** | **9.55 ± 0.06** / 10.95 ± 0.00 | **117.56 ± 0.19** / 9,471.93 ± 19.16 | 132.60 ± 1.27 | **9.80 ± 0.03** | 11.38 ± 0.13 |
| `c8` | **377.50 ± 2.52** | 351.50 ± 10.04 | **+7.40% / zstep 更快** | **17.14 ± 0.14** / 18.25 ± 0.17 | **535.78 ± 0.09** / 3,224.43 ± 35.53 | 594.99 ± 1.19 | **17.67 ± 0.12** | 19.00 ± 0.33 |
| `c16` | **527.00 ± 1.87** | 509.40 ± 3.55 | **+3.45% / zstep 更快** | 26.51 ± 0.11 / **23.47 ± 0.23** | **410.34 ± 0.43** / 4,949.74 ± 48.17 | 1,081.25 ± 0.78 | 351.43 ± 0.19 | **25.26 ± 0.15** |
| `c32` | **663.78 ± 3.57** | 635.98 ± 1.49 | **+4.37% / zstep 更快** | 43.11 ± 0.26 / **38.50 ± 0.12** | **411.54 ± 0.40** / 8,399.90 ± 102.45 | 1,228.47 ± 14.19 | 410.91 ± 0.25 | **107.80 ± 0.70** |

### 3.2 套件二：六档交错主基准（`in512` / `in1024` × `output=256`）

| 档位 | zstep Output (`tok/s`) | vLLM Output (`tok/s`) | 均值差 / 判定 | Mean TPOT z / v (`ms`) | z Median TTFT / submit-TTFT (`ms`) | v Median TTFT (`ms`) | z P99 ITL (`ms`) | v P99 ITL (`ms`) |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| `in512_c1` | **98.94 ± 0.33** | 87.24 ± 0.07 | **+13.42% / zstep 更快** | **9.68 ± 0.03** / 10.98 ± 0.00 | **118.68 ± 0.84** / 7,889.69 ± 26.37 | 133.63 ± 1.67 | **9.87 ± 0.04** | 11.48 ± 0.14 |
| `in512_c16` | **609.53 ± 7.15** | 573.38 ± 6.84 | **+6.30% / zstep 更快** | 24.15 ± 0.31 / 24.07 ± 0.38 | **410.82 ± 0.27** / 7,580.25 ± 54.25 | 1,094.82 ± 18.29 | **23.47 ± 0.25** | 25.12 ± 0.53 |
| `in512_c128` | **1,372.77 ± 2.51** | 1,152.03 ± 6.34 | **+19.16% / zstep 更快** | **82.32 ± 0.17** / 96.30 ± 0.60 | **831.11 ± 0.31** / 8,983.78 ± 1.96 | 3,068.72 ± 1.11 | **415.33 ± 0.14** | 1,019.66 ± 0.68 |
| `in1024_c1` | **95.24 ± 0.09** | 84.20 ± 0.02 | **+13.11% / zstep 更快** | **9.87 ± 0.01** / 11.18 ± 0.00 | **171.66 ± 0.16** / 24,369.74 ± 41.77 | 189.45 ± 0.73 | **10.07 ± 0.01** | 11.72 ± 0.02 |
| `in1024_c32` | **649.63 ± 1.94** | 614.93 ± 3.52 | **+5.64% / zstep 更快** | 44.18 ± 0.17 / **42.06 ± 0.35** | **415.61 ± 0.15** / 37,201.49 ± 204.12 | 1,959.74 ± 61.20 | 415.40 ± 0.11 | **175.89 ± 0.90** |
| `in1024_c128` | **1,002.13 ± 3.22** | 878.86 ± 4.26 | **+14.03% / zstep 更快** | **108.63 ± 0.34** / 120.03 ± 0.49 | **837.54 ± 0.10** / 41,618.50 ± 129.12 | 3,108.79 ± 0.44 | **419.58 ± 0.14** | 1,035.37 ± 0.43 |

### 3.3 `8 × RTX 6000D` `H=5120` `shm_v5` AllReduce 实测与全局 Top-2 贪心审计

`shm_v5`（[`phase0_6000d_ar_shmv5.json`](phase0_6000d_ar_shmv5.json)）在 `M=1..128` 全解码桶领先 `NCCL_LOC` **`1.19×..3.38×`**（`M=1, b=8` 为 **`14.48 vs 48.95 µs`**；`b=2` 达 **`14.08 µs / 3.48×`**），而在大载荷 `M=512`（`592.04 vs 581.68 µs`）与 `M=4096`（`4,660.95 vs 3,944.79 µs`）由 `NCCL_LOC` 取得更低耗时（Phase 3.5 对 `M>256` Eager 步自动路由至 `NCCL_LOC`）。

确定性全局 Top-2 贪心逐 token 审计（[`phase3_greedy_parity_report.json`](phase3_greedy_parity_report.json) 对照 [`phase3_vllm_greedy_reference.json`](phase3_vllm_greedy_reference.json)）：新增结构化非循环任务 **`prime_squares_table`（128 tokens）在单序列、批处理及 `chunk=24` 混合步下均实现 `128/128`（`100.00%` 精确匹配，最小裕度 `4.250`）**；其余 4 条开放式续写在极窄 BF16 ULP 边界（首处分歧 `zstep` `top1-top2` 裕度 `0.125..0.375`）发生分叉，顶层全局标志仍诚实保留 `all_matched=false`。

---

## 4. Phase 2：`Qwen/Qwen3.5-35B-A3B` (`TP=8`, `8 × RTX 5090`) 无 APC、独立种子、Warm `n=3` 主结果

zstep 六档共用：`allreduce_blocks=8`、`num_kv_pages=65536`、`max_seqs=128`、`max_step_tokens=max_decoder_rows=4096`、`max_prefill_chunk=2048`、`max_pages_per_seq=256`。Round 0 前运行 16 个 `512->64` warmup prompts，之后六档按 round-major 交错三轮。`vLLM 0.33.0` 显式传入 `--no-enable-prefix-caching`，双端每轮每档均使用独立种子 `base_seed + round_idx * 1000 + tier_idx`（`vLLM` 在 `in1024_c128` 的跨轮标准差由未关 APC 时的 `±484.51` 降至 `±18.11 tok/s`）。

| 档位 | zstep Output (`tok/s`) | vLLM Output (`tok/s`) | 均值差 / 判定 | Mean TPOT z / v (`ms`) | z Median TTFT / submit-TTFT (`ms`) | v Median TTFT (`ms`) | z P99 ITL (`ms`) | v P99 ITL (`ms`) |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| `in512_c1` | **261.53 ± 24.43** | 192.24 ± 10.22 | +36.05% / **zstep 更快** | 3.68 ± 0.34 / 4.66 ± 0.22 | 50.42 ± 11.10 / 3,016.61 ± 315.79 | 137.34 ± 17.39 | **4.12 ± 0.41** | 6.51 ± 0.93 |
| `in512_c16` | **2,037.92 ± 51.95** | 1,700.98 ± 12.13 | +19.81% / **zstep 更快** | 6.87 ± 0.24 / 7.62 ± 0.02 | 209.37 ± 1.82 / 2,252.67 ± 56.82 | 463.87 ± 17.91 | **6.98 ± 0.14** | 10.40 ± 0.28 |
| `in512_c128` | **5,021.65 ± 104.06** | 3,259.04 ± 5.72 | +54.08% / **zstep 更快** | 21.46 ± 0.50 / 32.66 ± 0.17 | 210.86 ± 1.69 / 3,484.02 ± 21.72 | 832.91 ± 181.55 | 223.50 ± 2.30 | **114.35 ± 1.68** |
| `in1024_c1` | **268.95 ± 1.47** | 197.31 ± 0.59 | +36.31% / **zstep 更快** | 3.42 ± 0.02 / 4.54 ± 0.02 | 79.91 ± 1.33 / 8,659.99 ± 46.85 | 132.48 ± 1.45 | **3.72 ± 0.03** | 6.01 ± 0.10 |
| `in1024_c32` | **2,174.52 ± 13.18** | 1,959.07 ± 29.76 | +11.00% / **zstep 更快** | 12.72 ± 0.07 / **11.29 ± 0.09** | 223.18 ± 1.23 / 11,030.49 ± 27.60 | 1,034.01 ± 33.94 | 230.63 ± 1.34 | **109.76 ± 1.33** |
| `in1024_c128` | **2,982.59 ± 54.67** | 2,565.79 ± 18.11 | +16.24% / **zstep 更快** | 35.97 ± 0.73 / 41.20 ± 0.24 | 226.59 ± 3.82 / 14,742.53 ± 211.99 | 711.86 ± 103.75 | 242.46 ± 3.61 | **113.56 ± 1.40** |

Phase 2 双端 Top-2 贪心审计（[`phase2_greedy_parity_report.json`](phase2_greedy_parity_report.json)）：`chunked` 模式合计 **`448 / 448` tokens（100%）** 与 `vLLM` 逐位一致；所有 `single`/`batched` 翻转均落在 logit `~25` 处的 **`0..2` 个 BF16 ULP（`<= 0.25`）** 边界内（`all_matched=false`，`TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=17`）。

## 5. Phase 2 `H=2048, blocks=8` 全尺寸 AllReduce 与 Phase 1 控制面成本

来源：[`phase2_allreduce_h2048_8gpu.json`](phase2_allreduce_h2048_8gpu.json)、[`phase1_mode_b_1gpu.json`](phase1_mode_b_1gpu.json)、[`phase1_mode_b_8gpu.json`](phase1_mode_b_8gpu.json)。

| `M` (`H=2048`) | 载荷 | 算法 | `zstep p50/p95/p99 (µs)` | `NCCL SHM p50/p95/p99 (µs)` | `zstep 80 次 (ms)` |
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

Phase 1 8-GPU 200 步控制面实测：Rust 下一拍准备 **`4.690 µs/step`**（累计 `938 µs`，199 步重叠），占 GPU 平均步 `2,832.10 µs` 的 **`0.166%`**，`overlap_speedup = 1.000269×`。

## 6. 复现命令

```bash
cargo build --release -p zs-server

# Phase 3.5: DeepSeek-V4.1-Flash TP=8 (8 × RTX 6000D, inside vllm/vllm-openai:v0.31.0)
export ZSTEP_DSV41_MODEL_DIR=/models/DeepSeek-V4.1-Flash
export OMP_NUM_THREADS=1 NCCL_P2P_DISABLE=1 NCCL_P2P_LEVEL=LOC

# 1) 贪心对齐审计（全局 Top-2 Logits）+ 5-kernel 零重编译检查
./target/release/zstep --mode-b --world-size 8 \
  --python /usr/bin/python3 --worker-root ./worker \
  --model-dir "$ZSTEP_DSV41_MODEL_DIR" --allreduce-blocks 8 \
  --num-kv-pages 2560 --max-seqs 16 --max-step-tokens 4096 --max-decoder-rows 4096 \
  --max-pages-per-seq 256 --ready-timeout-secs 900 --step-timeout-secs 180 \
  --verify-greedy docs/benchmarks/phase3_vllm_greedy_reference.json \
  --verify-report docs/benchmarks/phase3_greedy_parity_report.json

# 2) Warm n=3 六档交错主基准（无前缀缓存，独立种子，顺风车混合步 3072）
./target/release/zstep --mode-b --world-size 8 \
  --python /usr/bin/python3 --worker-root ./worker \
  --model-dir "$ZSTEP_DSV41_MODEL_DIR" --allreduce-blocks 8 \
  --num-kv-pages 2560 --max-seqs 128 --max-step-tokens 4096 --max-decoder-rows 4096 \
  --max-prefill-chunk 2048 --max-mixed-step-tokens 3072 --max-pages-per-seq 256 \
  --ready-timeout-secs 900 --step-timeout-secs 180 \
  --warmup-prompts 16 --repeats 3 \
  --bench-tiers "in512_c1:1:8:512:256;in512_c16:16:48:512:256;in512_c128:128:256:512:256;in1024_c1:1:20:1024:256;in1024_c32:32:200:1024:256;in1024_c128:128:400:1024:256" \
  --metrics-output docs/benchmarks/phase3_zstep_dsv41_tp8_6tier.json

# Phase 2: Qwen3.5-35B-A3B TP=8 (8 × RTX 5090, inside vllm/vllm-openai:v0.33.0)
export ZSTEP_QWEN35_MODEL_DIR=/path/to/Qwen3.5-35B-A3B
python3 worker/tests/phase2/collect_vllm_reference.py --mode all --repeats 3 --seed 20261111
./target/release/zstep --mode-b --world-size 8 \
  --python /usr/bin/python3 --worker-root ./worker \
  --model-dir "$ZSTEP_QWEN35_MODEL_DIR" \
  --allreduce-blocks 8 --allreduce-threads 256 --num-kv-pages 65536 \
  --max-seqs 128 --max-step-tokens 4096 --max-decoder-rows 4096 \
  --max-prefill-chunk 2048 --max-pages-per-seq 256 \
  --warmup-prompts 16 --repeats 3 --bench-seed 20261111 \
  --bench-tiers "in512_c1:1:8:512:256;in512_c16:16:48:512:256;in512_c128:128:256:512:256;in1024_c1:1:20:1024:256;in1024_c32:32:200:1024:256;in1024_c128:128:400:1024:256" \
  --metrics-output docs/benchmarks/phase2_zstep_qwen35_35b_a3b_tp8.json
```

Phase 2 `Qwen/Qwen3.5-35B-A3B` 复现使用 `ZSTEP_QWEN35_MODEL_DIR` 与公开镜像 `vllm/vllm-openai:v0.33.0`；Phase 3 / 3.5 `DeepSeek-V4.1-Flash` 复现使用 `ZSTEP_DSV41_MODEL_DIR` 与公开镜像 `vllm/vllm-openai:v0.31.0`（两边 `collect_vllm_reference.py` 均强制 `--no-enable-prefix-caching`），不在归档中记录任何私有主机路径或地址。
