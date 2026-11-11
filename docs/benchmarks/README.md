# `zstep` Master Benchmark Ledger (`sm_120`, 8×32 GB Non-P2P 4-NUMA PCIe)

本目录归档 Phase 0（`sm_120` 算子与 PCIe AllReduce）、Phase 1（Mode B Rust 控制面与双缓冲调度）及 Phase 2（`Qwen/Qwen3.5-35B-A3B` 真权重 serving 与 vLLM 对标）的实测 JSON。当前统一主结果是 **warm、单配置、round-major 交错、`n=3`**；旧单轮与消融数据只作历史记录，不与主表混用。

## 1. 硬件、软件与统计边界

```text
GPU:       8 × NVIDIA GeForce RTX 5090 (sm_120, 170 SMs, 32 GB GDDR7 per GPU)
CPU/NUMA:  Dual-Socket Xeon Gold 6530, 4 Sub-NUMA Clustering domains
Topology:  GPU [0,1,2] -> NUMA 0; GPU 3 -> NUMA 1
           GPU [4,5,6] -> NUMA 2; GPU 7 -> NUMA 3
Interconnect: PCIe Gen5 x16, no NVLink, can_device_access_peer == 0 for all 8×8 pairs
Model:      Qwen/Qwen3.5-35B-A3B, BF16, TP=8
Baseline:   vLLM 0.33.0, BF16, TP=8, degraded NCCL SHM on this non-P2P host
```

所有跨卡结论严格限定于该无 P2P、4-NUMA PCIe 拓扑，不外推到 NVLink 或 PCIe P2P。`84 GB` / `96 GB` 卡的几何计算仍标记为 `unverified`。表中 `std` 是三档重复测量的**总体标准差**；完整 `mean/std/min/max` 以 JSON 为准。吞吐“持平”规则为：相对差 `<3%`，或均值差 `<=2×sqrt(zstep_std²+vllm_std²)`；`n=3` 的持平不是严格等价证明。

## 2. 归档文件总览

共有 **15 份主 JSON + 19 份 vLLM 逐轮/warmup 原始 JSON = 34 份 JSON**。

| 阶段 | 文件 | 内容 |
| :--- | :--- | :--- |
| Phase 0.1 | [`step_00_env_sm120.json`](step_00_env_sm120.json) | 硬件拓扑、P2P 矩阵、算子环境 |
| Phase 0.2 | [`step_01_dense_fp8_gemm.json`](step_01_dense_fp8_gemm.json) | RMSNorm + MXFP8、稠密 GEMM、大偏移与零重编译 |
| Phase 0.3 | [`step_03_moe_mxfp4_sm120.json`](step_03_moe_mxfp4_sm120.json) | 384 专家路由、MXFP4 MoE、`>2^31` 偏移 |
| Phase 0.4 | [`step_04_dsv41_attention_sm120.json`](step_04_dsv41_attention_sm120.json) | 分页稀疏 MLA Decode / Verify / Extend |
| Phase 0.5a | [`step_05a_mhc_engram_sm120.json`](step_05a_mhc_engram_sm120.json) | mHC 与 Engram VRAM / 主机 UVA |
| Phase 0.5b | [`step_05b_pcie_allreduce_8gpu.json`](step_05b_pcie_allreduce_8gpu.json) | `H=5120` NUMA First-Touch AllReduce vs NCCL |
| Phase 1 | [`phase1_mode_b_1gpu.json`](phase1_mode_b_1gpu.json) | 1 GPU、200 步 Mode B IPC 与重叠实测 |
| Phase 1 | [`phase1_mode_b_8gpu.json`](phase1_mode_b_8gpu.json) | 8 GPU、200 步 Mode B IPC 与重叠实测 |
| Phase 2 | [`phase2_allreduce_h2048_8gpu.json`](phase2_allreduce_h2048_8gpu.json) | `H=2048, blocks=8, M=1..8192` 全尺寸 AllReduce 与 block 扫描 |
| Phase 2 | [`phase2_zstep_qwen35_35b_a3b_tp8.json`](phase2_zstep_qwen35_35b_a3b_tp8.json) | **当前主结果**：统一 4K token 配置、2K prefill chunk、16 warmup、六档交错 `n=3` |
| Phase 2 | [`phase2_vllm_qwen35_35b_a3b_tp8.json`](phase2_vllm_qwen35_35b_a3b_tp8.json) | vLLM 0.33.0 六档交错 `n=3` 基线与硬件 provenance |
| Phase 2 | [`phase2_vllm_raw/`](phase2_vllm_raw/) | 18 份逐轮结果与 1 份 warmup 原始结果 |
| Phase 2 | [`phase2_vllm_greedy_reference.json`](phase2_vllm_greedy_reference.json) | 4 个多 prompt、32/64/96/128-token vLLM 贪心参考 |
| Phase 2 | [`phase2_greedy_parity_report.json`](phase2_greedy_parity_report.json) | 单序列、`bucket=4` 批处理、`chunk=24` 逐 token 对齐及 17-kernel 零重编译审计 |
| Phase 2 | [`phase2_zstep_qwen35_35b_a3b_tp8_step4k.json`](phase2_zstep_qwen35_35b_a3b_tp8_step4k.json) | 旧单轮 4K 配置，仅作历史对照 |
| Phase 2 | [`phase2_qwen35_optimization_ablation.json`](phase2_qwen35_optimization_ablation.json) | 四阶段工程优化消融与稠密模型参考 |

## 3. 当前六档 warm `n=3` 主结果

zstep 六档共用：`allreduce_blocks=8`、`num_kv_pages=65536`、`max_seqs=128`、`max_step_tokens=max_decoder_rows=4096`、`max_prefill_chunk=2048`、`max_pages_per_seq=256`。Round 0 前运行 16 个 `512->64` warmup prompts，之后六档按 round-major 交错三轮。vLLM 使用相同请求数、输入/输出长度与三轮交错口径。

| 档位 | zstep Output (`tok/s`) | vLLM Output (`tok/s`) | 均值差 / 判定 | Mean TPOT z / v (`ms`) | z Median TTFT / submit-TTFT (`ms`) | v Median TTFT (`ms`) | z P99 ITL (`ms`) | v P99 ITL (`ms`) |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| `in512_c1` | 261.72 ± 19.81 | 204.26 ± 10.53 | +28.14% / **zstep 更快** | 3.66 ± 0.27 / 4.64 ± 0.23 | 51.12 ± 10.68 / 2,999.35 ± 243.52 | 69.13 ± 12.82 | 4.09 ± 0.45 | 6.51 ± 1.24 |
| `in512_c16` | 1,938.14 ± 140.42 | 1,893.04 ± 14.01 | +2.38% / **持平** | 7.29 ± 0.52 / 7.34 ± 0.04 | 229.50 ± 27.42 / 2,378.89 ± 209.47 | 298.17 ± 2.29 | 7.27 ± 0.68 | 10.29 ± 0.52 |
| `in512_c128` | 4,917.46 ± 97.01 | 3,975.90 ± 18.06 | +23.68% / **zstep 更快** | 21.95 ± 0.51 / 27.29 ± 0.10 | 213.40 ± 1.22 / 3,518.72 ± 19.46 | 1,192.91 ± 15.91 | 225.68 ± 2.50 | 112.48 ± 0.18 |
| `in1024_c1` | 252.42 ± 7.16 | 206.05 ± 0.61 | +22.50% / **zstep 更快** | 3.64 ± 0.11 / 4.60 ± 0.02 | 83.95 ± 4.03 / 9,346.27 ± 316.18 | 65.63 ± 1.32 | 3.99 ± 0.11 | 6.66 ± 0.27 |
| `in1024_c32` | 2,152.54 ± 40.35 | 2,678.01 ± 241.68 | -19.62% / **vLLM 更快** | 12.86 ± 0.25 / 9.44 ± 0.62 | 223.14 ± 0.30 / 11,226.25 ± 255.84 | 395.95 ± 65.78 | 230.90 ± 0.32 | 66.08 ± 33.07 |
| `in1024_c128` | 3,025.25 ± 24.35 | 3,666.69 ± 484.51 | -17.49% / **持平（高方差）** | 35.38 ± 0.36 / 28.83 ± 3.80 | 224.81 ± 1.90 / 14,577.91 ± 165.59 | 1,126.07 ± 183.12 | 238.88 ± 2.50 | 117.22 ± 2.22 |

`ttft_ms` 是 **permit-based**，匹配 vLLM `limited_request_func`；`submit_ttft_ms` 从 workload 提交开始，额外包含等待 permit 的排队时间。vLLM 归档没有 submit-based TTFT，禁止跨口径直接比较。两个单流档的 zstep Mean TPOT 均值为 `3.658/3.644 ms`，但三轮范围是 `3.431..4.031/3.552..3.791 ms`。统一 chunk 将 `in1024_c128` P99 ITL 从旧单轮 `1,303.57 ms` 降至 `238.88 ± 2.50 ms`，但仍慢于 vLLM；高并发尾延迟不能被吞吐结论掩盖。

## 4. 贪心逐 token 正确性：保留分歧

参考与实测分别见 [`phase2_vllm_greedy_reference.json`](phase2_vllm_greedy_reference.json) 和 [`phase2_greedy_parity_report.json`](phase2_greedy_parity_report.json)。覆盖 4 个不同 prompt、最长 128 tokens、单序列、4 序列批处理（decode `bucket=4`）以及 `max_prefill_chunk=24` 混合 Prefill/Decode。调度记录显示 chunked 执行有 3 个 prefill steps、128 个 decode steps、2 个 partial-prefill/decode steps，最大活跃序列数为 4。

当前报告的全局标志为 `all_matched=false`、`batched_vs_single_exact_match=false`、`chunked_vs_unchunked_exact_match=false`，**不是全量 100% 贪心 parity**：

| Prompt | 输出长度 | Single 精确前缀 / 全序列匹配率 | Batched | Chunked |
| :--- | ---: | :--- | :--- | :--- |
| `france_capital` | 32 | 32 / 100% | 32 / 100% | 32 / 100% |
| `ieee754_subnormals` | 128 | 50 / 39.06% | 18 / 21.09% | 18 / 20.31% |
| `dijkstra_shortest_path` | 96 | 65 / 67.71% | 65 / 67.71% | 65 / 67.71% |
| `posix_shm_tlb_shootdown` | 64 | 64 / 100% | 64 / 100% | 29 / 50.00% |

运行时审计为 **`TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=17`**，且 `/dev/shm` RAII 清理通过；这些工程检查与 token 数值分歧应分别报告。测试会核对统计字段与调度不变量，但不会把上述分歧伪装成通过全量 token 对齐。

## 5. `H=2048, blocks=8` 全尺寸 AllReduce

来源：[`phase2_allreduce_h2048_8gpu.json`](phase2_allreduce_h2048_8gpu.json)。BF16、8 卡、无 P2P；每个 Graph 10 次调用，300 次预热、1,000 次计时。`80 次` 是按 Graph p50 推算，不是独立端到端测量。

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

One-Shot `<=16 KiB` 的 Graph p50 是 `26.26..31.34 µs`，80 次是 `2.10..2.51 ms`，解释约 `3.5 ms` Decode TPOT 的主要通信预算；Graph 模式下 NCCL 这三档 p50 略低。Two-Shot 中间尺寸不能省略：`64 KiB..2 MiB` 除 `512 KiB` 外均由 NCCL 取得更低 p50。NUMA First-Touch 在本组 `16 MiB` 仍未超过 NCCL，在 `32 MiB` 取得 `1.24×`；`H=5120, 40 MiB` 的另一组记录为 `4,274.36 vs 6,206.57 µs`（`1.45×`，`max_abs_error=0`）。

`M=48, H=5120` block 扫描的 p50 为 `blocks=8/16/32: 57.96/66.35/76.44 µs`，p99 为 `60.92/72.86/86.54 µs`。增加 block 的退化与跨 NUMA `.sys` flag cacheline 流量及屏障开销一致；没有硬件 cacheline 计数器剖分，因此该归因明确保留为假设。

## 6. Phase 1：控制面物理成本，而非加速 tautology

来源：[`phase1_mode_b_1gpu.json`](phase1_mode_b_1gpu.json) 与 [`phase1_mode_b_8gpu.json`](phase1_mode_b_8gpu.json)。

| 规模 | 发布/唤醒 (`µs`) | IPC p50/p95/p99 (`µs`) | Rust 准备 (`µs/step`) | 累计准备 / 重叠步 | GPU 平均步 (`µs`) | 准备/GPU | Overlap speedup |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 GPU | 3.110 | 311 / 328 / 346 | 4.505 | 901 / 199 | 2,034.57 | 0.221% | 1.000292× |
| 8 GPU | 23.995 | 325 / 346 / 385 | **4.690** | **938 / 199** | **2,832.10** | **0.166%** | **1.000269×** |

`hidden_prepare_us` 只表示准备工作落在 GPU 执行窗口内；实际等效加速由 `(total_wall_us+hidden_prepare_us)/total_wall_us` 计算，8 GPU 约为 **`1.0003×`**。前一次同口径 8 GPU 记录为 `5.08 µs/step`（`1,016 µs` 累计，以 200 步为分母；199 步重叠）、GPU `3,066.94 µs`、占比 `0.166%`、`1.000281×`；当前 JSON 是上表的新测量。运行期 Triton 重编译为 0，1/8 GPU 回归脚本通过，退出后 `/dev/shm` 无残留。

## 7. Phase 0 算子摘要

| 算子 | 规格 | `M=6` | `M=48` | `M=192` | `M=4096` | 精度 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| RMSNorm + MXFP8 | `H=5120` | 3.40 µs | 3.22 µs | 4.00 µs | 15.52 µs | cos 0.99965 |
| b12x MXFP8 Linear | `5120->2048` | 15.52 µs | 22.17 µs | 23.36 µs | 76.47 µs | cos 0.99964..1.0 |
| TopK Router | 384 experts, top-6 | 5.74 µs | 5.79 µs | 6.89 µs | 23.88 µs | cos 1.0 |
| b12x MXFP4 MoE TP8 | local inter 288 | 70.24 µs | 435.07 µs | 781.73 µs | 2,414.73 µs | cos 0.99999 |
| b12x MXFP4 MoE EP8 | 48 local experts | 320.55 µs | 574.19 µs | 842.18 µs | 8,865.44 µs | cos 0.99999 |

完整 Sparse MLA、mHC、Engram 与各分位数见对应 JSON，不在此重复单值摘要。

## 8. 复现命令

```bash
cargo build --release -p zs-server

# Phase 0.5b AllReduce
torchrun --standalone --nproc_per_node=8 worker/tests/phase0/test_05_pcie_allreduce_8gpu.py

# Phase 1 回归（脚本自身运行 1 GPU / 8 GPU）
python3 worker/tests/test_mode_b_e2e.py

# Phase 2 GPU 正确性与 17-kernel 零重编译审计
export ZSTEP_QWEN35_MODEL_DIR=/path/to/Qwen3.5-35B-A3B
pytest -v worker/tests/test_shm_layout_parity.py \
  worker/tests/phase2/test_qwen35_ops_sm120.py \
  worker/tests/phase2/test_qwen35_e2e_tp8.py

# 当前统一 warm n=3 六档主基准
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

容器复现时通过 `ZSTEP_CONTAINER_IMAGE` 指定镜像，公开默认值为 `vllm/vllm-openai:v0.33.0`；模型目录只通过 `ZSTEP_QWEN35_MODEL_DIR` 传入，不在归档中记录私有主机路径。
