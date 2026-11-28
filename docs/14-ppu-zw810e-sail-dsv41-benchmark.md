# T-Head Zhenwu 810E × 8 (`TP=8`) DeepSeek-V4.1-Flash (`SAIL 1.7`) Benchmark & Evolution Report

## Overview

`DeepSeekV41SailStepRunner` (`worker/zstep_worker/models/deepseek_v41_sail.py`) and `kernels_sail` (`worker/zstep_worker/kernels_sail/`) provide native `TP=8` inference and zero-host-stall `DSpark (K=5)` speculative decoding for **DeepSeek-V4.1-Flash (`476 GiB`)** on **8 × T-Head PPU-ZW810E (`96 GiB` HBM2e, `SAIL 1.7` / `HGGC 12.3` / `triton-for-sail 3.4`)**.

Key Generation-4 architectural components:
- **Warp `__shfl_sync` Register FP4 LUT + 2-Col Reuse + Flattened `(k, kg)` MXFP4 MoE GEMV (`ozaki_fast_fp4`)**: Eliminates shared-memory LUT bank conflicts, halves `s_x` reads in `gate_up`, lifts `down_combine` SIMD lane utilization from `28.1%` to `96.4%`, and pairs with chunked 128-bit MXFP4 dequant + mixed-precision `acBLAS` (`CUDA_R_16BF -> CUDA_R_32F`) for prefill (`moe_gemv_max_rows=256`).
- **Ozaki `2×BF16 -> FP32` Tensor-Core Projections (`make_ozaki2_bf16` + `launch_gemm_bf16_f32_nt`)**: Lossless 2-term BF16 decomposition on AIU Tensor Cores for MoE Router, `cr=2` Compressor (`compressor_cr2_fused`), and large-batch `mHC`, preserving exact greedy token parity at Tensor-Core throughput.
- **128-bit Vectorized Single-Pass Split-K `mHC` + 16-Lane Warp Sinkhorn (`vec_1pass_ozaki`)**: Self-resetting arrival counter + half-warp `__shfl_xor_sync` 20-iter Sinkhorn (`zero` shared-memory barriers inside Sinkhorn).
- **Pipelined Flash-Decoding Split-K Sparse MQA (`triton_splitk`) & 128-bit Staggered-Burst `ICNLink` P2P Collectives (`PpuIcnCollectives`)**: Branch-free software-pipelined Triton Stage-1 + exact `attn_sink` HGGC reduction, plus CUDA-Graph-safe BF16/FP32 AllReduce and Top-2 Argmax over ICNLink.
- **Zero-Host-Stall Fused `DSpark (K=5)` Post-Verify + Draft `CUDAGraph` (`post_verify_draft_graphs`)**: Device-resident `dspark_sail_verify_prep_kernel` + `fused_mtp_ctx_kv_epilogue_bf16_kernel` + 3-layer MTP + 5-step BF16 Markov chain in a single CUDA Graph (`TRITON_RECOMPILATIONS=0`, `WATCHED_KERNELS=5`, `128 / 128` exact match on `prime_squares_table`).

## End-to-End Measured Results (`8 × PPU-ZW810E`, `in=512, out=128`)

### Continuous Batching 5-Tier (`Gen-0` vs `Gen-3` vs `Gen-4`)

| Tier (`in=512, out=128`) | Gen-0 Out (`tok/s`) | Gen-3 Out (`tok/s`) | **Gen-4 Out (`tok/s`)** | **Gen-4 Total (`tok/s`)** | Speedup vs Gen-3 / Gen-0 | Gen-3 Mean TPOT (`ms`) | **Gen-4 Mean TPOT (`ms`)** | **Gen-4 Median TTFT (`ms`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `c = 1` | `32.46` | `49.61` | **`64.95`** | **`324.74`** | **`1.31×` / `2.00×`** | `17.96` | **`13.34`** | **`276.44`** |
| `c = 8` | `112.13` | `205.94` | **`299.94`** | **`1,499.70`** | **`1.46×` / `2.67×`** | `30.50` | **`19.43`** | **`945.90`** |
| `c = 16` | `140.01` | `279.02` | **`396.04`** | **`1,980.20`** | **`1.42×` / `2.83×`** | `47.09` | **`31.57`** | **`947.61`** |
| `c = 32` | `93.07` | `347.16` | **`501.02`** | **`2,505.10`** | **`1.44×` / `5.38×`** | `76.36` | **`50.43`** | **`1,586.33`** |
| `c = 64` | `113.57` | `395.91` | **`566.41`** | **`2,832.06`** | **`1.43×` / `4.99×`** | `134.21` | **`90.13`** | **`1,685.72`** |

### Native `DSpark (K=5)` Speculative Decoding (`Gen-0` vs `Gen-3` vs `Gen-4`)

| Tier (`in=512, out=128`) | Gen-0 Out (`tok/s`) | Gen-3 Out (`tok/s`) | **Gen-4 Out (`tok/s`)** | **Gen-4 Total (`tok/s`)** | Speedup vs Gen-3 / Gen-0 | **Gen-4 Mean TPOT (`ms`)** | **Gen-4 Median TPOT (`ms`)** | **Gen-4 P99 TPOT (`ms`)** | **Gen-4 Median TTFT (`ms`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `dspark_c1` (`c = 1`) | `47.86` | `75.93` | **`130.46`** | **`652.28`** | **`1.72×` / `2.73×`** (`4.02×` vs Gen-0 non-spec) | **`5.52`** (`181.2 tok/s`) | **`4.12`** (`242.7 tok/s`) | **`4.28`** | **`280.37`** |
| `dspark_c8` (`c = 8`) | `57.94` | `165.84` | **`296.93`** | **`1,484.64`** | **`1.79×` / `5.12×`** | **`19.32`** | **`17.25`** | **`25.97`** | **`433.16`** |
