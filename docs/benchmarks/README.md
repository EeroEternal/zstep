# `zstep` Master Benchmark Ledger (`sm_120` Non-P2P PCIe: `8×32 GB` & `8×85.6 GB`)

本目录归档 Phase 0（`sm_120` 算子与 PCIe AllReduce）、Phase 1（Mode B Rust 控制面与双缓冲调度）、Phase 2（`8 × RTX 5090 32 GB` 上 `Qwen/Qwen3.5-35B-A3B` 无 APC 真权重 serving 与 `vLLM 0.33.0` 对标）、**Phase 3 / 3.5 / 3.6 / 3.7 / 3.8（`8 × RTX 6000D 85.6 GB` 上 `DeepSeek-V4.1-Flash` `TP=8` 无 APC 真权重 serving、`b12x` / `sm120_dsa_indexer` `156-SM` 优化与 `vLLM 0.31.0` 对标）**、**Phase 3.9（`8 × RTX 6000D` 上 `Qwen/Qwen3.5-35B-A3B` Route A：`shm_v5` Stage-2 尾融合 + SM120 `[V, K]` `GatedDeltaNet` 全量 A/B 审计）** 及 **Phase 3.10（当前 `main` 合入 `#21/#16/#19` 后 `Qwen3.5-35B-A3B` `OVERLAP=0/1` 消融与 15 档审计 + `DeepSeek-V4.1-Flash` 内置 `ae58183f` Overlay 与 `ZSTEP_REQUIRE_SM120_EXP=1` 最新审计）** 的实测 JSON。所有主结果均为 **warm、单配置、无前缀缓存（`--no-enable-prefix-caching`）、按 `(round, tier)` 独立种子、round-major 交错、`n=3`**；旧单轮与消融数据只作历史记录，不与主表混用。

## 1. 硬件、软件与统计边界

```text
[Host A — Phase 0/1/2]
GPU:       8 × NVIDIA GeForce RTX 5090 (sm_120, 170 SMs, 32 GB GDDR7 per GPU)
Topology:  4-NUMA PCIe Gen5 ([0,0,0,1,2,2,2,3]), no NVLink, can_device_access_peer == 0
Model:     Qwen/Qwen3.5-35B-A3B, BF16, TP=8
Baseline:  vLLM 0.33.0, BF16, TP=8, --no-enable-prefix-caching, isolated per-(round,tier) seeds, degraded NCCL SHM

[Host B — Phase 3 / 3.5 / 3.6 / 3.7 / 3.8 / 3.9 / 3.10]
GPU:       8 × NVIDIA RTX 6000D (sm_120, 156 SMs, 85,651 MiB GDDR7 per GPU, Driver 595.99.02)
Topology:  2-NUMA PCIe Gen5 ([0,0,0,0,1,1,1,1]), no NVLink, 20 × Mellanox CX-8 PCIe bridge ports (`15b3:197c`, `ACSCtl=0x001d`, SM BAR MMIO P2P unsupported -> CNS host-DRAM path)
Models:    DeepSeek-V4.1-Flash (1017, 475.25 GiB MXFP8/MXFP4/BF16, engram cpu_offload=false) & Qwen/Qwen3.5-35B-A3B (BF16), TP=8
Baseline:  vllm/vllm-openai:v0.31.0, TP=8, --no-enable-prefix-caching, isolated per-(round,tier) seeds, NCCL_P2P_DISABLE=1, NCCL_P2P_LEVEL=LOC, symmetric SM120 64-tok SWA shim
SM120 Overlay (Phase 3.8 / 3.10): bundled `worker/third_party/flashinfer_sm120_overlay_ae58183f.tar.xz` (`ae58183fff34c545b3e3020d81d591f2a79658fc`, SHA256 `90e4db7f...`) mounted via `./scripts/prepare_sm120_overlay.sh --print-docker-args` (`ZSTEP_REQUIRE_B12X=1`, `ZSTEP_REQUIRE_SM120_EXP=1`, `B12X_W4A8_TINY_DECODE=0`)
```

所有跨卡结论严格限定于对应无 P2P PCIe 拓扑，不外推到 NVLink 或 PCIe P2P。表中 `std` 是三轮重复测量的**总体标准差**；完整 `mean/std/min/max` 以 JSON 为准。吞吐“持平”规则为：相对差 `<3%`，或均值差 `<=2×sqrt(zstep_std²+vllm_std²)`；`n=3` 的持平不是严格等价证明。

## 2. 归档文件总览

共有 **36 份主 JSON + 19 份 vLLM 逐轮/warmup 原始 JSON = 55 份 JSON**。

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
| Phase 3.6 | [`phase36_zstep_b12x_dsv41_tp8_in512_out128.json`](phase36_zstep_b12x_dsv41_tp8_in512_out128.json) | **Phase 3.6 套件一主结果**：`zstep + b12x` `512->128` 四档无缓存独立种子 `n=3` |
| Phase 3.6 | [`phase36_zstep_b12x_dsv41_tp8_6tier.json`](phase36_zstep_b12x_dsv41_tp8_6tier.json) | **Phase 3.6 套件二主结果**：`zstep + b12x` 六档交错无缓存独立种子 `n=3` |
| Phase 3.6 | [`phase36_vllm_pure_dsv41_tp8_in512_out128.json`](phase36_vllm_pure_dsv41_tp8_in512_out128.json) | `vLLM 0.31.0` 进程内纯引擎（`AsyncLLMEngine` + `TokensPrompt` + `detokenize=False`，无 HTTP/分词/SSE）`512->128` 四档同口径基线 `n=3` |
| Phase 3.6 | [`phase36_vllm_pure_dsv41_tp8_6tier.json`](phase36_vllm_pure_dsv41_tp8_6tier.json) | `vLLM 0.31.0` 进程内纯引擎（无 HTTP/分词/SSE）六档交错同口径基线 `n=3` |
| Phase 3.6 | [`phase36_b12x_greedy_parity_report.json`](phase36_b12x_greedy_parity_report.json) | `zstep + b12x` 单序列/批处理/Mixed-Chunked 全局 Top-2 贪心审计（`prime_squares_table` `128/128` 100% 匹配，零重编译） |
| Phase 3.7 | [`phase37_6000d_operator_audit.json`](phase37_6000d_operator_audit.json) | `RTX 6000D` (`156-SM`, `d497e85`) 单算子微基准与数值审计：`PreparationSession` 生命周期、7 形状 Block32 FP8、`B12X_W4A8_TINY_DECODE=1` 失真证明（`cos_sim=0.870..0.879`）、`o_proj` / `LMHead` / `mHC` |
| Phase 3.7 | [`phase37_zstep_b12x_dsv41_tp8_in512_out128.json`](phase37_zstep_b12x_dsv41_tp8_in512_out128.json) | **Phase 3.7 套件一主结果**（`git_commit=d497e85`）：`156-SM` `PreparationSession` 常驻 + 210 层 Block32 FP8 + 零校验快速绑定 `512->128` 四档 `n=3` |
| Phase 3.7 | [`phase37_zstep_b12x_dsv41_tp8_6tier.json`](phase37_zstep_b12x_dsv41_tp8_6tier.json) | **Phase 3.7 套件二主结果**（`git_commit=d497e85`）：`zstep 3.7 + b12x` 六档交错无缓存独立种子 `n=3`（峰值 `1,717.41 tok/s` 输出 / `5,945.57 tok/s` 总吞吐） |
| Phase 3.7 | [`phase37_b12x_greedy_parity_report.json`](phase37_b12x_greedy_parity_report.json) | `zstep 3.7 + b12x`（`git_sha=d497e85...`，`moe=40, lin=210/6, unprepared_warnings=0`）全局 Top-2 贪心审计（`prime_squares_table` `128/128` 100% 匹配，零重编译） |
| Phase 3.7 | [`phase37_6000d_comm_audit.json`](phase37_6000d_comm_audit.json) | **`zstep#14` 8 卡通信与 PCIe ACS/CX-8（`15b3:197c`）审计**：`H=2048` / `H=5120` AllReduce 与 `[M, 640]->[M, 5120]` AllGather（`shm_v5_prod` 自适应 `256->512` 线程 + `64 KiB` One-Shot vs 旧基线 / `nodone` / `pkt64` / `NCCL_LOC`）、SM BAR MMIO `UR` 证明与 `DeepSeek-V4.1` Amdahl 消融 |
| Phase 3.8 | [`phase38_6000d_flashinfer_ops_audit.json`](phase38_6000d_flashinfer_ops_audit.json) | **SM120 `FlashInfer` 实验算子筛选与 `TP=8` 端到端审计**（`ae58183f`）：`b12x` 新旧对比、`sm12x_mega_gate`（`M >= 32`）门限消融、分页 Decode `sm120_dsa_indexer`（8 层 CUDAGraph `3.55×..4.33×`）、淘汰项负向实证及三套件 12 档 Warm `n=3` 结果（`in1024_c128` 突破 `6,005.25 tok/s`） |
| Phase 3.9 | [`phase39_6000d_qwen35_route_a_audit.json`](phase39_6000d_qwen35_route_a_audit.json) | **Route A（`Qwen3.5-35B-A3B` `TP=8`, `8 × RTX 6000D`）`shm_v5` Stage-2 尾融合 + SM120 `[V, K]` `GatedDeltaNet` 全量审计**（`893495c4 -> 98724b4c`）：`H=2048/5120` `cluster8` DSMEM / `fused_1op` 微基准、30 层 `[V, K]` 自适应解码与 `cuda_vk_recurrent` Prefill 微基准、贪心零重编译门禁及三套件 15 档 Warm `n=3` 结果（`15/15` 全胜，`c1` `+23.80%`、`Mean TPOT = 2.95 ms`，`in1024_c128` 总吞吐 `14,783.15 tok/s`） |
| Phase 3.10 | [`phase310_6000d_qwen35_overlap_audit.json`](phase310_6000d_qwen35_overlap_audit.json) | **`Qwen3.5-35B-A3B` 合入 `#21/#16/#19` 后设备驻留 Token 跨步流水重叠与并行 `moe_align` 全量审计**（`8 × RTX 6000D`, Warm `n=3`）：Suite 1 `ZSTEP_DECODE_DEVICE_TOKENS=0/1` 受控消融（`c1` `285.99 -> 335.62 tok/s` **`+17.35%`**，`Mean TPOT = 2.66 ms`）+ 15 档主结果（`in512_c1` **`353.98 tok/s`**，`in512_c128` 输出 **`5,039.13 tok/s`**，`in1024_c128` 总吞吐 **`15,640.43 tok/s`**）+ 贪心 `0` 重编译门禁 |
| Phase 3.10 | [`phase310_6000d_dsv41_post_route_a_audit.json`](phase310_6000d_dsv41_post_route_a_audit.json) | **`DeepSeek-V4.1-Flash` 合入 `#20/#16/#19` 后内置 `ae58183f` Overlay（`ZSTEP_REQUIRE_SM120_EXP=1`）全量审计**（`8 × RTX 6000D`, Warm `n=3`）：Suite 1 `b12x`-only 回退（`98.60 tok/s`）vs 内置 Overlay（`c1` 突破 **`100.44 ± 0.13 tok/s`**，`Mean TPOT = 8.70 ms`）+ Suite 2/3 最新主结果（`in512_c1` **`107.03 tok/s`**，`in4096_c16` 总吞吐 **`6,518.90 tok/s`**） |

---

## 2.7 Phase 3.10（当前 `main` 最新主结果：`Qwen3.5` 设备内 Token 流水重叠 + `DeepSeek-V4.1` 内置 `ae58183f` Overlay，`8 × RTX 6000D` Warm `n=3`）

> 归档于 [`phase310_6000d_qwen35_overlap_audit.json`](phase310_6000d_qwen35_overlap_audit.json)（详见 **[`06-phase2-qwen35-benchmark.md` §8](../06-phase2-qwen35-benchmark.md)**）与 [`phase310_6000d_dsv41_post_route_a_audit.json`](phase310_6000d_dsv41_post_route_a_audit.json)（详见 **[`07-phase3-dsv41-6000d-benchmark.md` §10](../07-phase3-dsv41-6000d-benchmark.md)**）：
> * **`Qwen3.5-35B-A3B` (`TP=8`)**：开启 `STEP_FLAG_DEVICE_TOKENS` 设备内跨步直传与 8-Warp 并行 `moe_align` 后，Suite 1 `OVERLAP=0 -> 1` 单流 `c1` 由 `285.99 ± 0.59` 提升 **`+17.35%` 至 `335.62 ± 0.11 tok/s`**（`Mean TPOT = 2.66 ms`）；Suite 2 `in512_c1` 达 **`353.98 ± 0.24 tok/s`**、`in512_c128` 输出 **`5,039.13 ± 6.31 tok/s`**（总吞吐 **`15,117.38 tok/s`**）、`in1024_c128` 输出 **`3,128.09 ± 4.86 tok/s`**（总吞吐 **`15,640.43 tok/s`**）；Suite 3 `in4096_c1 / c16` 达 **`181.72 / 453.58 tok/s`**。
> * **`DeepSeek-V4.1-Flash` (`TP=8`)**：通过仓库内置 `./scripts/prepare_sm120_overlay.sh` + `ZSTEP_REQUIRE_SM120_EXP=1` 激活 `b12x + sm12x_mega_gate (min_m=32) + sm120_dsa_indexer` 并叠加 `#20` `shm_v5` 预取后，Suite 1 `c1 / c8 / c16 / c32` 由仅 `b12x` 回退的 `98.60 / 467.33 / 644.97 / 839.05 tok/s` 进一步推进至 **`100.44 ± 0.13 / 470.84 ± 1.08 / 649.06 ± 1.40 / 840.34 ± 4.43 tok/s`**（单流 `Mean TPOT = 8.70 ms`）；Suite 2 `in512_c1` **`107.03 ± 0.08 tok/s`**、`in512_c128` **`1,733.34 ± 6.14 tok/s`**、`in1024_c128` **`1,200.63 ± 1.88 tok/s`**（总吞吐 **`6,003.15 tok/s`**）；Suite 3 `in4096_c1 / c16` **`76.63 ± 0.06 / 197.54 ± 0.84 tok/s`**（总吞吐 **`6,518.90 tok/s`**）。

---

## 2.8 Phase 3.9（Route A：`Qwen3.5-35B-A3B` `shm_v5` Stage-2 尾融合 + SM120 `[V, K]` `GatedDeltaNet`）最新主结果（`8 × RTX 6000D`, Warm `n=3`）

> 归档于 [`phase39_6000d_qwen35_route_a_audit.json`](phase39_6000d_qwen35_route_a_audit.json)，算子设计与物理归因详见 **[`06-phase2-qwen35-benchmark.md` §7](../06-phase2-qwen35-benchmark.md)**。在完整保留 `shm_v5` 4 槽环形 `done-wait`、`90 s` 超时熔断与 `control[0..blocks)` 锁步的前提下，融合 Stage-2 `AllReduce + Residual Add + RMSNorm`（`M ∈ {1, 2, 4}` 解码桶走 `__cluster_dims__(8, 1, 1)` DSMEM `cluster8`，`M >= 8` 走行对齐 `fused_1op`）消除每步 80 次独立 RMSNorm 发射与 `reduced_buf` 往返，并将 30 层 `GatedDeltaNet` 状态池转置为 `float4` 合并访存 `[V, K]`（Decode 自适应 `split_3op` / `split_2op` / `cluster2` 跨桶 `0 ULP` 等价 + 变长 Prefill `cuda_vk_recurrent`）。运行时审计 **`TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=20`**，相对 `main` `893495c4` 在 **全部 `15 / 15` 档位上同时取得吞吐提升（`+3.53% .. +23.80%`）、Mean TPOT 下降、Median TTFT 缩短与 P99 ITL 降低**。

| 套件 / 档位 | 基线 `893495c` (`tok/s`) | 仅 Part A1 (`tok/s`) | **完整 Route A (`tok/s`)** | Route A vs 基线 | Mean TPOT 基线 -> **Route A** (`ms`) | Med TTFT 基线 -> **Route A** (`ms`) | P99 ITL 基线 -> **Route A** (`ms`) |
| :--- | ---: | ---: | ---: | :--- | :---: | :---: | :---: |
| Suite 1 `c1` (`512->128`) | `248.07 ± 2.82` | `284.25 ± 0.78` (`+14.58%`) | **`307.11 ± 11.03`** | **`+23.80%`** | `3.69` -> **`2.95`** | `47.83` -> **`43.09`** | `3.84` -> **`3.35`** |
| Suite 1 `c2` (`512->128`) | `451.49 ± 3.42` | `497.66 ± 1.00` (`+10.23%`) | **`535.96 ± 22.90`** | **`+18.71%`** | `3.91` -> **`3.26`** | `69.77` -> **`64.01`** | `4.10` -> **`3.65`** |
| Suite 1 `c4` (`512->128`) | `758.58 ± 7.30` | `805.78 ± 2.18` (`+6.22%`) | **`844.07 ± 26.63`** | **`+11.27%`** | `4.44` -> **`3.96`** | `111.23` -> **`103.72`** | `4.69` -> **`4.33`** |
| Suite 1 `c8` (`512->128`) | `1,257.85 ± 11.49` | `1,270.24 ± 2.89` (`+0.99%`) | **`1,318.37 ± 14.91`** | **`+4.81%`** | `4.87` -> **`4.68`** | `196.04` -> **`183.20`** | `5.19` -> **`5.10`** |
| Suite 1 `c16` (`512->128`) | `1,690.35 ± 14.64` | `1,707.63 ± 2.49` (`+1.02%`) | **`1,781.42 ± 6.41`** | **`+5.39%`** | `8.00` -> **`7.62`** | `156.26` -> **`144.59`** | `137.52` -> **`125.72`** |
| Suite 1 `c32` (`512->128`) | `2,225.78 ± 9.29` | `2,342.80 ± 5.19` (`+5.26%`) | **`2,533.34 ± 7.70`** | **`+13.82%`** | `12.33` -> **`10.80`** | `168.32` -> **`149.71`** | `165.97` -> **`145.06`** |
| Suite 1 `c64` (`512->128`) | `2,743.41 ± 6.47` | `2,769.50 ± 8.54` (`+0.95%`) | **`3,033.29 ± 16.27`** | **`+10.57%`** | `18.65` -> **`16.97`** | `340.07` -> **`293.06`** | `170.31` -> **`147.98`** |
| Suite 2 `in512_c1` (`out=256`) | `261.82 ± 0.32` | — | **`307.14 ± 10.01`** | **`+17.31%`** | `3.65` -> **`3.10`** | `47.34` -> **`43.28`** | `3.81` -> **`3.32`** |
| Suite 2 `in512_c16` (`out=256`) | `1,978.11 ± 3.54` | — | **`2,048.01 ± 38.53`** | **`+3.53%`** | `7.29` -> **`7.07`** | `155.62` -> **`144.90`** | `7.06` -> **`6.92`** |
| Suite 2 `in512_c128` (`out=256`) | `4,447.25 ± 4.78` | — | **`4,791.36 ± 18.52`** | **`+7.74%`** | `24.44` -> **`22.89`** | `348.96` -> **`300.43`** | `174.36` -> **`150.07`** |
| Suite 2 `in1024_c1` (`out=256`) | `246.33 ± 0.97` | — | **`288.43 ± 9.29`** | **`+17.09%`** | `3.74` -> **`3.17`** | `87.02` -> **`79.32`** | `3.86` -> **`3.38`** |
| Suite 2 `in1024_c32` (`out=256`) | `1,961.16 ± 6.22` | — | **`2,166.66 ± 5.43`** | **`+10.48%`** | `14.38` -> **`12.98`** | `179.39` -> **`159.91`** | `179.24` -> **`159.86`** |
| Suite 2 `in1024_c128` (`out=256`) | `2,675.34 ± 0.30` | — | **`2,956.63 ± 4.68`** | **`+10.51%`** | `40.28` -> **`36.45`** | `382.16` -> **`328.73`** | `194.18` -> **`164.96`** |
| Suite 3 `in4096_c1` (`out=128`) | `148.64 ± 0.33` | — | **`164.85 ± 0.37`** | **`+10.91%`** | `4.15` -> **`3.75`** | `333.77` -> **`300.52`** | `4.33` -> **`3.92`** |
| Suite 3 `in4096_c16` (`out=128`) | `369.75 ± 0.72` | — | **`412.22 ± 0.77`** | **`+11.49%`** | `30.65` -> **`27.68`** | `646.54` -> **`578.35`** | `216.62` -> **`193.23`** |

---

## 2.9 Phase 3.7（`main` `d497e85`：`156-SM` `PreparationSession` 常驻 + 210 层 Block32 FP8 + Fast-Bind）最新主结果（`8 × RTX 6000D`, Warm `n=3`）

> 归档自未修改的 `main` `d497e85`（PR `#11`）源码树；`num_kv_pages=2560` 下稳态显存占用 **`84,937..84,959 / 85,651 MiB = 99.2%`（每卡余量 `< 0.7 GiB`）**。按 `<3%` 持平规则，`c=1` 三档输出吞吐相对 Phase 3.6 均为**持平**（单流确定性收益为 `Median TTFT` 全线缩短 `-15.41..-18.71 ms`），而相对 `vLLM` 纯引擎同口径基线**全部 `10 / 10` 档显著胜出（`+14.44% .. +52.48%`）**。

| 套件 / 档位 | `vLLM` 纯引擎 (`tok/s`) | `zstep` 3.6 (`tok/s`) | **`zstep` 3.7 (`tok/s`)** | 3.7 vs 3.6 / vs `vLLM` 纯引擎 | Mean TPOT `vPure` / **`z3.7`** (`ms`) | Med TTFT `vPure` / `z3.6` / **`z3.7`** (`ms`) | P99 ITL `vPure` / **`z3.7`** (`ms`) |
| :--- | ---: | ---: | ---: | :--- | :---: | :---: | :---: |
| Suite 1 `c1` | `83.59 ± 0.01` | `95.85 ± 0.65` | **`95.66 ± 0.20`** | `-0.20%`（持平） / **`+14.44%`** | `10.94` / **`9.20`** | `141.37` / `185.21` / **`169.42`** | `11.39` / **`9.43`** |
| Suite 1 `c8` | `348.18 ± 8.30` | `441.70 ± 1.60` | **`446.37 ± 1.62`** | `+1.06%`（持平） / **`+28.20%`** | `19.02` / **`14.07`** | `590.28` / `510.32` / **`506.43`** | `18.80` / **`14.62`** |
| Suite 1 `c16` | `492.61 ± 2.27` | `612.24 ± 2.13` | **`631.57 ± 0.97`** | **`+3.16%`** / **`+28.21%`** | `25.63` / **`21.71`** | `1,084.36` / `386.49` / **`381.68`** | **`25.97`** / `323.60` |
| Suite 1 `c32` | `617.07 ± 1.10` | `800.74 ± 1.01` | **`820.22 ± 1.82`** | `+2.43%` / **`+32.92%`** | `41.13` / **`34.23`** | `1,430.66` / `387.05` / **`382.36`** | **`86.20`** / `382.07` |
| Suite 2 `in512_c1` | `87.06 ± 0.01` | `103.99 ± 1.10` | **`101.61 ± 0.11`** | `-2.29%`（持平） / **`+16.71%`** | `10.97` / **`9.21`** | `141.86` / `184.93` / **`169.52`** | `11.41` / **`9.46`** |
| Suite 2 `in512_c16` | `555.89 ± 1.34` | `727.06 ± 1.62` | **`741.71 ± 5.61`** | `+2.01%` / **`+33.43%`** | `25.32` / **`19.59`** | `1,086.45` / `386.86` / **`382.22`** | `26.12` / **`18.91`** |
| Suite 2 `in512_c128` | `1,126.34 ± 1.87` | `1,686.53 ± 7.96` | **`1,717.41 ± 4.17`** | `+1.83%` / **`+52.48%`** | `99.30` / **`64.47`** | `3,066.32` / `779.52` / **`769.26`** | `1,021.41` / **`384.61`** |
| Suite 2 `in1024_c1` | `84.57 ± 0.00` | `102.27 ± 0.18` | **`99.33 ± 0.04`** | `-2.87%`（持平） / **`+17.46%`** | `11.17` / **`9.43`** | `177.89` / `191.20` / **`172.49`** | `11.56` / **`9.68`** |
| Suite 2 `in1024_c32` | `605.12 ± 1.35` | `778.39 ± 4.56` | **`796.36 ± 0.84`** | `+2.31%` / **`+31.60%`** | `42.27` / **`35.66`** | `2,127.63` / `391.62` / **`386.97`** | **`296.51`** / `386.76` |
| Suite 2 `in1024_c128` | `866.25 ± 1.06` | `1,169.71 ± 1.06` | **`1,189.11 ± 2.74`** | `+1.66%` / **`+37.27%`** | `122.27` / **`90.83`** | `3,107.53` / `787.39` / **`777.10`** | `1,035.90` / **`389.31`** |

---

## 3. Phase 3.6（`zstep + b12x`）与纯引擎同口径 `vLLM 0.31.0` 主结果（`8 × RTX 6000D`, Warm `n=3`）

为严格消除 API 层开销偏差（无 HTTP、无分词、无增量解码、无 SSE），[`bench_vllm_pure_engine.py`](../../worker/tests/phase3/bench_vllm_pure_engine.py) 在 `vLLM 0.31.0` 进程内以 `OPENAI_API_SERVER` 上下文（`max_num_batched_tokens=8192`、`enable_prefix_caching=False`、`detokenize=False`、`TokensPrompt`）输入与 `zstep` 逐位一致的 SplitMix64 `int32` prompt 序列。

### 3.0 纯引擎同口径四方总览（`vLLM HTTP` / `vLLM 纯引擎` / `zstep 3.5` / `zstep 3.6 + b12x`）

| 套件 / 档位 | `vLLM` HTTP (`tok/s`) | `vLLM` 纯引擎 (`tok/s`) | `zstep` 3.5 (`tok/s`) | **`zstep` 3.6 + `b12x` (`tok/s`)** | 3.6 vs 3.5 / vs `vLLM` 纯引擎 | Mean TPOT `vPure` / **`z3.6`** (`ms`) | Med TTFT `vPure` / **`z3.6`** (`ms`) | P99 ITL `vPure` / **`z3.6`** (`ms`) |
| :--- | ---: | ---: | ---: | ---: | :--- | :---: | :---: | :---: |
| Suite 1 `c1` | `83.97 ± 0.05` | `83.59 ± 0.01` | **`96.18 ± 0.57`** | `95.85 ± 0.65` | `-0.34%` / **`+14.67%`** | `10.94` / **`9.05`** | **`141.37`** / `185.21` | `11.39` / **`9.28`** |
| Suite 1 `c8` | `351.50 ± 10.04` | `348.18 ± 8.30` | `377.50 ± 2.52` | **`441.70 ± 1.60`** | **`+17.01%`** / **`+26.86%`** | `19.02` / **`14.23`** | `590.28` / **`510.32`** | `18.80` / **`14.79`** |
| Suite 1 `c16` | `509.40 ± 3.55` | `492.61 ± 2.27` | `527.00 ± 1.87` | **`612.24 ± 2.13`** | **`+16.18%`** / **`+24.29%`** | `25.63` / **`22.47`** | `1,084.36` / **`386.49`** | **`25.97`** / `327.96` |
| Suite 1 `c32` | `635.98 ± 1.49` | `617.07 ± 1.10` | `663.78 ± 3.57` | **`800.74 ± 1.01`** | **`+20.63%`** / **`+29.76%`** | `41.13` / **`35.12`** | `1,430.66` / **`387.05`** | **`86.20`** / `386.75` |
| Suite 2 `in512_c1` | `87.24 ± 0.07` | `87.06 ± 0.01` | `98.94 ± 0.33` | **`103.99 ± 1.10`** | **`+5.10%`** / **`+19.45%`** | `10.97` / **`8.93`** | **`141.86`** / `184.93` | `11.41` / **`9.27`** |
| Suite 2 `in512_c16` | `573.38 ± 6.84` | `555.89 ± 1.34` | `609.53 ± 7.15` | **`727.06 ± 1.62`** | **`+19.28%`** / **`+30.79%`** | `25.32` / **`20.01`** | `1,086.45` / **`386.86`** | `26.12` / **`19.09`** |
| Suite 2 `in512_c128` | `1,152.03 ± 6.34` | `1,126.34 ± 1.87` | `1,372.77 ± 2.51` | **`1,686.53 ± 7.96`** | **`+22.86%`** / **`+49.74%`** | `99.30` / **`65.72`** | `3,066.32` / **`779.52`** | `1,021.41` / **`389.59`** |
| Suite 2 `in1024_c1` | `84.20 ± 0.02` | `84.57 ± 0.00` | `95.24 ± 0.09` | **`102.27 ± 0.18`** | **`+7.38%`** / **`+20.92%`** | `11.17` / **`9.07`** | **`177.89`** / `191.20` | `11.56` / **`9.47`** |
| Suite 2 `in1024_c32` | `614.93 ± 3.52` | `605.12 ± 1.35` | `649.63 ± 1.94` | **`778.39 ± 4.56`** | **`+19.82%`** / **`+28.63%`** | `42.27` / **`36.51`** | `2,127.63` / **`391.62`** | **`296.51`** / `391.34` |
| Suite 2 `in1024_c128` | `878.86 ± 4.26` | `866.25 ± 1.06` | `1,002.13 ± 3.22` | **`1,169.71 ± 1.06`** | **`+16.72%`** / **`+35.03%`** | `122.27` / **`92.32`** | `3,107.53` / **`787.39`** | `1,035.90` / **`394.26`** |

---

## 3A. Phase 3.5（标准算子）历史对照：`DeepSeek-V4.1-Flash` (`TP=8`, `8 × RTX 6000D`)

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

来源：[`phase2_allreduce_h2048_8gpu.json`](phase2_allreduce_h2048_8gpu.json)（Phase 2 `Host A` 历史单文件 SHM 基线，`<=16 KiB` One-Shot；当前 `main` 每卡独立 NUMA 页 `shm_v5_prod` + `64 KiB` One-Shot + 自适应 `512` 线程 + 大 `M` `NCCL_LOC` 混合路由在 `8 × RTX 6000D` 上的最新实测见 [`phase37_6000d_comm_audit.json`](phase37_6000d_comm_audit.json)）、[`phase1_mode_b_1gpu.json`](phase1_mode_b_1gpu.json)、[`phase1_mode_b_8gpu.json`](phase1_mode_b_8gpu.json)。

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

## 6. 复现命令（支持含 `.git` 或无 `.git` 的 `git archive` 源码树）

```bash
cargo build --release -p zs-server
# 若在无 .git 的 git archive 目录运行且需精确记录 Git SHA，可传入 export ZSTEP_GIT_SHA=<40-hex-sha>；
# 未设置时 BenchmarkProvenance 与 GreedyParityReport 自动回退编译期 SHA 或 "unknown"，不会因缺少 .git 报错。

# ── Phase 3.8 / 3.10 (+ 内置 SM120 实验 Overlay): DeepSeek-V4.1-Flash TP=8 (8 × RTX 6000D) ─
# 0) 校验解压仓库内置确定性 SM120 overlay（ae58183f）并生成 docker run 挂载与强门禁参数：
#    自动开启 ZSTEP_ENABLE_B12X=1 ZSTEP_REQUIRE_B12X=1 ZSTEP_REQUIRE_SM120_EXP=1 B12X_W4A8_TINY_DECODE=0
./scripts/prepare_sm120_overlay.sh /tmp/zstep-sm120-overlay
DOCKER_OVERLAY_ARGS=($(./scripts/prepare_sm120_overlay.sh --print-docker-args /tmp/zstep-sm120-overlay))
# 在 docker run vllm/vllm-openai:v0.31.0 时直接传入 "${DOCKER_OVERLAY_ARGS[@]}"；
# 若需消融回退到仅 b12x（关闭 mega_gate 与 dsa_decode），追加：
#   -e ZSTEP_ENABLE_SM12X_MEGA_GATE=0 -e ZSTEP_ENABLE_SM120_DSA_DECODE=0 -e ZSTEP_REQUIRE_SM120_EXP=0
export ZSTEP_DSV41_MODEL_DIR=/models/DeepSeek-V4.1-Flash
export OMP_NUM_THREADS=1 NCCL_P2P_DISABLE=1 NCCL_P2P_LEVEL=LOC

# 0b) 156-SM 单算子微基准与 SM120 实验算子筛选（单卡 GPU 0 即可运行）
CUDA_VISIBLE_DEVICES=0 python3 worker/tests/phase3/bench_sm120_dsv41_operators.py \
  --output docs/benchmarks/phase37_6000d_operator_audit.json
CUDA_VISIBLE_DEVICES=0 python3 worker/tests/phase3/bench_sm120_flashinfer_exp_ops.py \
  --output /tmp/sm120_flashinfer_exp_ops.json

# 0c) 8 卡 shm_v5 vs NCCL_LOC 通信审计（zstep#14）及 Route A Stage-2 尾融合 + [V, K] GDN 微基准
python3 worker/tests/phase3/bench_shmv5_nccl_comm_audit.py \
  --output docs/benchmarks/phase37_6000d_comm_audit.json
python3 worker/tests/phase3/bench_sm120_qwen35_route_a_ops.py \
  --part all --output /tmp/sm120_qwen35_route_a_ops.json

# 1) 贪心对齐审计（全局 Top-2 Logits）+ 5-kernel 零重编译检查
#    门禁退出码约定：
#    - 默认（或带 `--require-exact-cases prime_squares_table`）：校验 TRITON_RECOMPILATIONS=0 且结构化算术表 128/128 全匹配，预期 rc=0
#    - `--max-greedy-mismatch-margin 0.25`：纯 BF16 窄裕度门禁（Phase 2 Qwen3.5 预期 rc=0；Phase 3/3.6/3.7 MXFP8/MXFP4 预期 rc=1）
#    - `--require-exact-greedy true`：要求全部开放式续写 100% 逐位一致（`all_matched=false` 时预期 rc=1）
./target/release/zstep --mode-b --world-size 8 \
  --python /usr/bin/python3 --worker-root ./worker \
  --model-dir "$ZSTEP_DSV41_MODEL_DIR" --allreduce-blocks 8 \
  --num-kv-pages 2560 --max-seqs 16 --max-step-tokens 4096 --max-decoder-rows 4096 \
  --max-pages-per-seq 256 --ready-timeout-secs 900 --step-timeout-secs 180 \
  --verify-greedy-json docs/benchmarks/phase3_vllm_greedy_reference.json \
  --verify-output-json docs/benchmarks/phase37_b12x_greedy_parity_report.json \
  --require-exact-cases prime_squares_table

# 2) Warm n=3 六档交错主基准（无前缀缓存，独立种子，顺风车混合步 3072）
#    注：`--num-kv-pages 2560` 在 RTX 6000D (85,651 MiB) 上稳态占用 84,937..84,959 MiB (99.2%)；
#    长期在线 HTTP 服务建议按需下调至 `--num-kv-pages 2048..2304` 预留 ~2.5..5.0 GiB/卡动态缓冲。
./target/release/zstep --mode-b --world-size 8 \
  --python /usr/bin/python3 --worker-root ./worker \
  --model-dir "$ZSTEP_DSV41_MODEL_DIR" --allreduce-blocks 8 \
  --num-kv-pages 2560 --max-seqs 128 --max-step-tokens 4096 --max-decoder-rows 4096 \
  --max-prefill-chunk 2048 --max-mixed-step-tokens 3072 --max-pages-per-seq 256 \
  --ready-timeout-secs 900 --step-timeout-secs 180 \
  --warmup-prompts 16 --repeats 3 \
  --bench-tiers "in512_c1:1:8:512:256;in512_c16:16:48:512:256;in512_c128:128:256:512:256;in1024_c1:1:20:1024:256;in1024_c32:32:200:1024:256;in1024_c128:128:400:1024:256" \
  --metrics-output docs/benchmarks/phase37_zstep_b12x_dsv41_tp8_6tier.json

# ── Phase 2 / Phase 3.9 / Phase 3.10: Qwen3.5-35B-A3B TP=8 (8 × RTX 5090 or 8 × RTX 6000D) ─
export ZSTEP_QWEN35_MODEL_DIR=/path/to/Qwen3.5-35B-A3B
# 默认开启 Part A1 (ZSTEP_QWEN_ENABLE_FUSED_AR_NORM=1)、Part A2 (ZSTEP_QWEN_ENABLE_FUSED_GDN=1)
# 与设备内跨步 Token 流水重叠 (ZSTEP_DECODE_DEVICE_TOKENS=1)；OVERLAP=0 消融传入 ZSTEP_DECODE_DEVICE_TOKENS=0。
# Suite 3 长输入 4096->128 需将 --max-pages-per-seq 设为 512（16 tok/page × 512 = 8192 tokens）。
python3 worker/tests/phase2/collect_vllm_reference.py --mode all --repeats 3 --seed 20261111
./target/release/zstep --mode-b --world-size 8 \
  --python /usr/bin/python3 --worker-root ./worker \
  --model-dir "$ZSTEP_QWEN35_MODEL_DIR" \
  --allreduce-blocks 8 --allreduce-threads 256 --num-kv-pages 65536 \
  --max-seqs 128 --max-step-tokens 4096 --max-decoder-rows 4096 \
  --max-prefill-chunk 2048 --max-pages-per-seq 256 \
  --warmup-prompts 16 --repeats 3 \
  --bench-tiers "in512_c1:1:8:512:256;in512_c16:16:48:512:256;in512_c128:128:256:512:256;in1024_c1:1:20:1024:256;in1024_c32:32:200:1024:256;in1024_c128:128:400:1024:256" \
  --metrics-output docs/benchmarks/phase2_zstep_qwen35_35b_a3b_tp8.json
```

Phase 2 `Qwen/Qwen3.5-35B-A3B` 复现使用 `ZSTEP_QWEN35_MODEL_DIR` 与公开镜像 `vllm/vllm-openai:v0.33.0`；Phase 3 / 3.5 `DeepSeek-V4.1-Flash` 复现使用 `ZSTEP_DSV41_MODEL_DIR` 与公开镜像 `vllm/vllm-openai:v0.31.0`（两边 `collect_vllm_reference.py` 均强制 `--no-enable-prefix-caching`），不在归档中记录任何私有主机路径或地址。
