# T-Head Zhenwu 810E × 8 (`TP=8`) DeepSeek-V4.1-Flash (`SAIL 1.7`) Benchmark & Evolution Report

## Overview

`DeepSeekV41SailStepRunner` (`worker/zstep_worker/models/deepseek_v41_sail.py`) and `kernels_sail` (`worker/zstep_worker/kernels_sail/`) provide native `TP=8` inference and `DSpark (K=5)` speculative decoding for **DeepSeek-V4.1-Flash (`476 GiB`)** on **8 × T-Head PPU-ZW810E (`96 GiB` HBM2e, `SAIL 1.7` / `HGGC 12.3` / `triton-for-sail 3.4`)**.

Key architectural components:
- **128-bit Staggered-Burst `ICNLink` P2P Collectives (`PpuIcnCollectives`)**: CUDA-Graph-safe 1-shot / 2-shot BF16 AllReduce (`13.4 µs` at `M=1`, `49.1 µs` at `M=64`, `458.4 µs` at `M=1024`), FP32 Indexer AllReduce, BF16 AllGather, and lock-free Top-2 Argmax over T-Head ICNLink.
- **`triton-for-sail` AIU Tensor-Core & HGGC Compute Kernels**:
  - Grouped MXFP4 MoE (`_mxfp4_moe_grouped_gate_up_kernel` / `_mxfp4_moe_grouped_down_kernel`) for prefill + fused HGGC W4-LUT GEMV (`mxfp4_moe_gemv_*`, `moe_gemv_max_rows=32`) and fused shared-expert clamped SwiGLU+UE8M0 (`shared_swiglu_quant_bf16_kernel`) for decode/verify.
  - Paged `TP=8` sparse MQA (`_triton_sparse_attn_tp8_kernel`, `h_local=8->16`, `d=512`, exact `attn_sink`).
  - Adaptive multi-CU Split-K Hyper-Connections (`mhc_splitk_stage1_bf16_kernel` + `mhc_splitk_reduce_sinkhorn_kernel`, `mhc_split_k=8`, `mhc_fused_max_rows=4096`).
  - Exact stateless Engram V2 4-gram normalized lookup (`engram_compressed_vocab_size=99092`) and cross-NUMA `pin_memory=True` H2D staging for `SAIL 1.7`.
- **Full-40-Layer `CUDAGraph` Decode / Verify + 3-Layer `DSpark (K=5)` `CUDAGraph` Drafter**: `TRITON_RECOMPILATIONS=0` (`WATCHED_KERNELS=4`), `128 / 128` (`100.0%`) exact greedy token match on `prime_squares_table`.

## End-to-End Measured Results (`8 × PPU-ZW810E`, `in=512, out=128`)

### Continuous Batching 5-Tier (`sail_zw810e_dsv41_evolved_5tier.json`)

| Tier (`in=512, out=128`) | Baseline Out (`tok/s`) | Evolved Out (`tok/s`) | Evolved Total (`tok/s`) | Speedup | Baseline Mean TPOT (`ms`) | Evolved Mean TPOT (`ms`) | Median TTFT (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `c = 1` | `32.46` | **`41.70`** | **`208.52`** | **`1.28×`** | `25.14` | **`19.61`** | `580.52` |
| `c = 8` | `112.13` | **`138.57`** | **`692.83`** | **`1.24×`** | `41.25` | **`32.88`** | `3212.30` |
| `c = 16` | `140.01` | **`171.42`** | **`857.09`** | **`1.22×`** | `78.62` | **`64.00`** | `3215.14` |
| `c = 32` | `93.07` | **`197.14`** | **`985.69`** | **`2.12×`** | `290.03` | **`117.29`** (`2.47×`) | `5269.85` |
| `c = 64` | `113.57` | **`149.78`** | **`748.91`** | **`1.32×`** | `469.18` | **`349.96`** (`1.34×`) | `5740.04` |

### Native `DSpark (K=5)` Speculative Decoding (`sail_zw810e_dsv41_evolved_dspark_k5.json`)

| Tier (`in=512, out=128`) | Out (`tok/s`) | Total (`tok/s`) | Mean TPOT (`ms`) | Median TPOT (`ms`) | P99 TPOT (`ms`) | Median TTFT (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `dspark_c1` (`c = 1`) | **`70.01`** (`2.16×` vs non-spec) | **`350.04`** | **`9.81`** (`101.9 tok/s`, `2.56×`) | **`7.22`** (`138.5 tok/s`) | **`10.80`** | **`582.11`** |
| `dspark_c8` (`c = 8`) | **`82.85`** | **`414.26`** | **`74.63`** | **`62.05`** | **`119.44`** | **`769.70`** |
