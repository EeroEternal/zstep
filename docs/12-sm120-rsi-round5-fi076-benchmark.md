# Phase 3.12: SM120 `FlashInfer v0.7.6` Overlay, Dynamic-SMEM `Split-K=32` mHC Fusion & `#8740` 32K Paged DSA Benchmark

- **Date**: `2026-11-27`
- **Hardware**: `8 × NVIDIA RTX 6000D` (`SM120`, `156 SMs`, `99 KB` dynamic SMEM/SM, `112 MiB` L2, `85,651 MiB` GDDR7/GPU, `CUDA 13.0`, Dual-Socket Xeon `4+4` PCIe Gen5 Switch topology at `106.1 GB/s` intra-switch PXB P2P)
- **Model**: `DeepSeek-V4.1-Flash` (`475.25 GiB`, `TP=8`, `kv_cache_dtype=fp8_ds_mla`)
- **Incumbent Baseline**: `main@651395c` (remote Stage-1 clean baseline `ce830f8afc49`)
- **Raw Benchmark Artifacts**:
  - [`docs/benchmarks/round5_stage1_fi076_scout_results.json`](benchmarks/round5_stage1_fi076_scout_results.json)
  - [`docs/benchmarks/round5_stage2_mhc_smem_prescreen.json`](benchmarks/round5_stage2_mhc_smem_prescreen.json)
  - [`docs/benchmarks/phase312_6000d_round5_fi076_smem_mhc_32k_dsa.json`](benchmarks/phase312_6000d_round5_fi076_smem_mhc_32k_dsa.json)

---

## 1. Stage 1: `FlashInfer v0.7.6` (`d0cbb476`) Overlay & 8-GPU Hardware Scout

1. **Deterministic Overlay & SHA256-Keyed Persistent Host JIT Cache**:
   - Upgraded `worker/third_party/flashinfer_sm120_overlay_d0cbb476.tar.xz` (`2,277,180 B`, `SHA256 = e09f7c10503e9df6e7345b355cc3becee8d113cf45702963ae5a3833d56cd72c`) with backwards-compatible comm JIT aliases and surgical `trace/templates/comm.py` file bind-mount (preserving stock `vllm 0.31.0` `gated_act_mxfp8` templates).
   - Added SHA256-keyed persistent host JIT cache (`FI_CACHE_DIR="/tmp/zstep-sm120-fi-cache-${EXPECTED_SHA256:0:12}"`) in `scripts/prepare_sm120_overlay.sh`, cutting cold container `b12x` preparation time from `197 s` to **`15 s`**.
2. **Empirically Excluded Trials (Preserved per `AGENTS.md`)**:
   - **`FlashInfer 0.7.6 pcie_ipc_ar` (`profile="pcieswitch-pairs"`, `memop_supported=true`)**: Measured `5,867 µs` at `M=4096` (`slow_hops=136` across dual-socket NUMA) vs `zstep shm_pxb_bfp1` **`1,168 µs`** (`5.0×` slower). Excluded; kept `shm_pxb_bfp1`.
   - **SM120 `70 MiB` Persisting L2 Cache Window (`cudaStreamAttrAccessPolicyWindow`)**: Pinning `70 MiB` of `fn` weights in SM120's `112 MiB` L2 regressed `M=4..16` sub-layers (`0.984×`) due to non-persisting L2 eviction pressure on activations and KV pages. Excluded.

---

## 2. Stage 2 & Stage 3: Landed Kernel & Long-Context Graph Optimizations

1. **SM120 On-Chip Dynamic-SMEM `Split-K=32` `mHC Prenorm + Sinkhorn` Single-Kernel Fusion (`csrc/shm_allreduce_kernels.cu`, `csrc/shm_allreduce.cpp`, `b12x_dsv41.py`)**:
   - Replaces the 2-kernel `tf32_hc_prenorm_gemm + sm120_mhc_sinkhorn_stats_kernel` chain on `mhc_stream` with `sm120_mhc_prenorm_sinkhorn_smem_kernel<MAX_M>` (`MAX_M in {4, 8, 16}`):
     - Stages `640` BF16s per K-split once in SMEM (`<= 20 KB`), streams `fn [24, 20480]` via `__ldcs` `float4` loads, matches `cvt.rna.tf32.f32` (`max_diff <= 2.47e-5`, bit-exact deterministic across replays), uses compile-time `#pragma unroll` register accumulators (zero local-memory spill), reduces via coherent L2 `__ldcg`, and self-resets arrival counter `done_cnt` before running 20-iteration warp-shuffle Sinkhorn.
   - Standalone microbenchmark (`round5_stage2_mhc_smem_prescreen.json`): **`1.44×–1.83×` faster** at `M=1..8` (`M=1`: `20.06 → 10.98 µs`), saving **`0.33–0.49 ms/step`** in overlapped 8-GPU sub-layers.
   - Dispatched piecewise for `1 <= n <= 12` (`b12x_dsv41.py`) to avoid `32`-CTA auxiliary-stream SM contention against `b12x` main-stream GEMMs at `n >= 16`.
2. **`#8740` Two-Stage `candidate_blocks` 32K Paged DSA Indexer + `long_decode_graphs` (`b12x_dsv41.py`, `deepseek_v41.py`)**:
   - Wires `candidate_out` on writer layer `20` (`self.candidate_write=True`) and `candidate_blocks` on reader layers `24, 28, 32, 36` (`self.candidate_write=False`), pre-sizing `_decode_logits(..., 65536)` and `_decode_long_ws(..., slot=0/1)` to extend SM120 paged sparse decode coverage from `16,384` up to `524,288` tokens while keeping `<= 16,384` untouched.
   - Captures `self.long_decode_graphs` for buckets `[1, 2, 4, 8, 10, 12, 16, 20, 24, 32]` at `max_seq_len = self.max_model_len` sharing `graph_pool` (`deepseek_v41.py`), eliminating eager Python fallback during `> 16K` long-context decode.

---

## 3. Stage 4: 3-Regime 3-Round Interleaved Cold `AB/BA` Verification (`192.0 min`, 36 Cold 8-GPU Containers)

All runs enforce `TRITON_RECOMPILATIONS=0`, `Trace Conformance = CONFORMANT`, Z3 formal resource/soundness checks (`CLOSED`), `DispatchVerifier` (`8,640` pairs, `0` violations), and 100% greedy token parity (`prime_squares_table` single, batched, and chunked):

| Verification Suite | Candidate | Screen Gain | 3-Round Cold `AB/BA` Geomean | `paired_lcb_2se` | Per-Tier 3-Round Mean / Median Deltas (vs `main@651395c`) | Verdict |
|---|---|---:|---:|---:|---|---|
| **`longctx-tp8` (`8K–32K` Target)** | `g0001-01817706` | **`+10.31%`** (`ctx16k_c8 +21.26%`, `ctx8k_c2 +0.34%`) | **`+10.77%`** (`+12.13%` vs `f2c23b1`) | **`+10.55%`** | `ctx32k_c4`: `48.82 → 57.92` out tok/s (**`+18.82%`**, TPOT `42.88 → 30.38 ms` **`-29.2%`**)<br>`ctx16k_c12`: `101.06 → 118.42` out tok/s (**`+17.37%`**, TPOT `75.10 → 65.82 ms` **`-12.4%`**)<br>`ctx32k_c8`: `55.87 → 60.24` out tok/s (**`+7.92%`**, TPOT `86.48 → 77.42 ms` **`-10.5%`**)<br>`ctx8k_c16`: `223.95 → 224.09` out tok/s (**`+0.04%`**, TPOT `49.22 → 49.16 ms`) | **`ACCEPTED`** |
| **`spec-dspark` (`DSpark5` Guard)** | `g0001-882aa453` | **`+0.76%`** (`spec_c1 +0.74%`, `spec_c8 +0.78%`) | **`+0.56%`** | `-0.25%` | `spec_c4`: **`+1.68%`** (`6.66 → 6.54 ms`), `spec_c2`: **`+0.42%`** (`5.69 → 5.67 ms`), `spec_c16`: `-0.42%` | **Guard Positive (`+0.56%`)** |
| **`short-throughput` (`512–1K` Guard)** | `g0001-07bb4dc0` | **`+0.15%`** (`in512_c1 +0.70%`, `in512_c32 -0.40%`) | `-0.42%` (at `n <= 64`) | `-0.47%` | `in512_c1` (`n=1`): **`+0.70%`** (`166.93 → 168.10` tok/s), `in1024_c32`: **`+0.38%`**<br>`in512_c16` (`n=16`): `-1.02%`, `in512_c128`: `-0.31%`, `in1024_c128`: `-0.72%` | **Piecewise Guard Tightened (`1 <= n <= 12`)** |
