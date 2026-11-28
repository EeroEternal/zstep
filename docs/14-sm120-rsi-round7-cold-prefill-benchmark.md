# Phase 3.14: SM120 Lockstep-Safe Cold-Prefill Chunk Scaling, TBO `PXB-4K` Ring Reallocation, Piecewise MLA Prefill & CUDA 13 Green Context Benchmark

- **Date**: `2026-11-28`
- **Hardware**: `8 × NVIDIA RTX 6000D` (`SM120`, `156 SMs`, `99 KB` dynamic SMEM/SM, `112 MiB` L2, `85,651 MiB` GDDR7/GPU, `CUDA 13.0`, Dual-Socket Xeon `4+4` PCIe Gen5 Switch topology at `106.1 GB/s` intra-switch PXB P2P)
- **Model**: `DeepSeek-V4.1-Flash` (`475.25 GiB`, `TP=8`, `kv_cache_dtype=fp8_ds_mla`, `PrecisionTrack.LOSSLESS`)
- **Incumbent Baseline**: `main@3a1e298` (Round 6 `Async Engram + Fused DSpark`, remote SHA `764dec49c8d3`)
- **Raw Benchmark Artifacts**:
  - [`docs/benchmarks/round7_stage1_prefill_green_ctx_scout.json`](benchmarks/round7_stage1_prefill_green_ctx_scout.json)
  - [`docs/benchmarks/round7_multi_regime_abba_results.json`](benchmarks/round7_multi_regime_abba_results.json)

---

## 1. Stage 1: 8-GPU Hardware Scout & Empirically Excluded Hypotheses (`AGENTS.md`)

1. **TBO Chunk Scaling (`2048 -> 8192`) + `shm_ar_tbo` `PXB-4K` Ring Reallocation (`P1` — Landed)**:
   - Scaling pure prefill from `chunk = 2048` (`u = 1024`: `2.6242 µs/tok`, `214.98 ms` / 40L) to `chunk = 8192` (`u = 4096`) initially degraded to `540.11 ms` (`r6_pxb2k_shared_156sm`) because `shm_ar_tbo.pxb_max_elements = 2048 * 5120` caused `u = 4096` (`20,971,520` BF16 elements) to miss the intra-switch PXB P2P ring and fall back to host `/dev/shm`.
   - Reallocating `shm_ar_tbo` to `pxb_max_elements = 4096 * 5120` when `max_step_tokens > 4096` (`pxb4k_shared_156sm`, Z3-verified `245.38 MiB <= 256.0 MiB` on `PP=1`, `320.12 MiB <= 384.0 MiB` on `PP=2`) cut 40-layer `8K` execution to **`287.87 ms` (`0.8785 µs/tok`)** — **`1.88×` faster** than `r6_pxb2k` at `8K` and **`2.99×` higher per-token efficiency** than `2K` chunks.
2. **Piecewise SM120 MG MLA Prefill Crossover (`P1` — Tightened Dispatch)**:
   - Stock `cake_sparse_mla` (`flashinfer_trtllm_batch_decode_sparse_mla_dsv4`) outperforms `b12x` MG prefill (`_run_sm120_compressed_prefill`) on all `32` `swa_only` layers (`0.35×–0.96×`), all `4` `compressed_cr64` layers (`iw_n = 512`, `0.31×–0.97×`), and `full_cr1` at `T = 1024` (`0.835×`).
   - `b12x` MG prefill wins on `full_cr1` (`4` layers, `iw_n = 2048`) at **`rows >= 2048`** (`1.028×` @ `2K`, **`1.097×` @ `4K`**, **`1.125×` @ `8K`**, `cos_sim = 1.0`). Tightened `b12x_dsv41.py` dispatch from `iw_n > 0 and rows >= 1024` to **`iw_n >= 2048 and rows >= 2048`**.
3. **CUDA 13 Green Context SM Partitioning (`P5` — Empirically Falsified & Excluded)**:
   - Disjoint SM splits (`pxb4k_green_split_8_148` and `pxb4k_green_split_16_140`) fail with **`CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE (720)`** because `b12x.moe.fused_moe` launches a `156-SM` persistent cooperative kernel.
   - Asymmetric Green Contexts restricting only `comm_s` (`pxb4k_green_comm8_main156`: `298.94 ms` @ `8K`; `pxb4k_green_comm16_main156`: `309.08 ms` @ `8K`) are **`3.8%–16.7%` slower** than unpartitioned `pxb4k_shared_156sm` (`287.87 ms` @ `8K`) because `shm_ar_tbo` (`24` blocks, `priority=-1`) opportunistically fills `24` SMs during inter-kernel tail gaps when unthrottled. Excluded.
4. **Intra-Layer `shared_experts` Aux-Stream Overlap (`P5` — Empirically Falsified & Excluded)**:
   - Across `M = 8..4096`, overlapping `shared_experts` on an aux stream with `156-SM` persistent `b12x` `routed_experts` regressed **`22.6–41.2 µs/layer` (`0.981×–0.990×`)** due to `100%` SM occupancy plus event synchronization overhead. Excluded.

---

## 2. Landed Lockstep-Safe Architecture & Closed-Loop Guard Fixes (`97` LOC, `g0005-9271a17c`)

1. **Lockstep-Safe Cold-Prefill Burst Scheduling (`crates/zs-scheduler/src/serving.rs`, `tests.rs`)**:
   - Dynamically expands `prefill_chunk_limit` from `max_prefill_chunk` (`2,048`) to `max_step_tokens` (`8,192` on `cold-prefill`, `4,096` default) **if and only if** `decode_active_count == 0 && pending_prefill_seqs <= 1` (`(active.len() - decode_active_count) + waiting_permitted.len() <= 1`).
   - Prevents single-request monopolization when `>= 2` concurrent prompts arrive (`ctx8k_c2`), keeping concurrent prefill requests in lockstep (`p99_ITL = 6.55 ms`, `0%` regression vs `g0002`'s `165.20 ms` spike).
2. **Conditional `4K` TBO Linear Cap & `PXB-4K` Ring Allocation (`deepseek_v41.py`, `b12x_dsv41.py`)**:
   - Enables `shm_ar_tbo` `pxb_max_elements = 4096 * 5120` and `enable_4k_linear_cap=True` (`4096` MoE/linear bucket) strictly when `self.max_step_tokens > 4096`, preserving exact Round 6 cache keys and ring footprints on `max_step_tokens <= 4096` profiles.
3. **Narrow `129..256`-Row Lossless AllReduce Parity Band (`shm_allreduce.py`)**:
   - Sets `pxb_bfp = 0` strictly for `128 * 5120 < input.numel() <= 256 * 5120`, closing the `batched` (`n = 223`) vs `chunked` (`n = 101 <= 128`) prefill precision gap (`max_logit_drift = 2.125 <= 8.00`) while leaving `1..128`-row `DSpark` speculative verification bit-identical.

---

## 3. Full 3-Round Interleaved Cold `AB/BA` Verification & 3-Guard Summary (`g0005-9271a17c`)

All runs passed `RedTeam` (`spec_pins_verified=true`), `DispatchVerifier` (`8,640` pairs, `0` violations), 9/9 Z3 `FormalVerifier` clauses (`CLOSED`, `STATIC_RUNNER_PXB_TOTAL_BUDGET = 245.38 MiB <= 256.0 MiB`), `TRITON_RECOMPILATIONS = 0`, `Trace Conformance = CONFORMANT`, and 100% greedy parity (`prime_squares_table` single, batched, chunked):

| Regime / Phase | Tier / Metric | Baseline (`3a1e298`) | Candidate (`g0005-9271a17c`) | Measured Delta | Gate Verdict |
|---|---|---:|---:|---:|---|
| **`cold-prefill` Screen (`repeats=1`)** | `prefill8k_c1` (`1:1:8192:16`)<br>`prefill16k_c1` (`1:1:16384:16`) | `10,402.0 tot tok/s` (`TTFT 703.68 ms`)<br>`11,020.9 tot tok/s` (`TTFT 1,404.08 ms`) | **`15,099.5 tot tok/s`** (**`TTFT 462.87 ms`**)<br>**`16,243.6 tot tok/s`** (**`TTFT 928.28 ms`**) | **`+45.12%` (`-34.2%` `TTFT`)**<br>**`+47.44%` (`-33.9%` `TTFT`)**<br>**Screen Geomean: `+46.28%`** | **`PASS` (`66,981 MiB/GPU`)** |
| **`cold-prefill` 3-Round Cold `AB/BA`** | `prefill8k_v` (`1:6:8192:32`)<br>`prefill16k_v` (`1:6:16384:32`)<br>`prefill32k_v` (`1:4:32768:32`) | `9,520.9 tot tok/s` (`TTFT 697.25 ms`)<br>`10,595.7 tot tok/s` (`TTFT 1,383.32 ms`)<br>`11,194.3 tot tok/s` (`TTFT 2,762.32 ms`) | **`13,100.7 tot tok/s`** (**`TTFT 462.66 ms`**, `TPOT 5.32 ms`)<br>**`14,996.4 tot tok/s`** (**`TTFT 928.43 ms`**, `TPOT 5.36 ms`)<br>**`15,860.9 tot tok/s`** (**`TTFT 1,901.24 ms`**, peak **`15,864.5`**) | **`+37.60%` med** (**`-33.65%` `TTFT`**)<br>**`+41.55%` med** (**`-32.88%` `TTFT`**)<br>**`+41.67%` med** (**`-31.17%` `TTFT` / `-861 ms`**)<br>**Verify Geomean: `+40.26%`** (`lcb_2se +39.27%`) | **`ACCEPTED` (`67,145 MiB/GPU`)** |
| **`longctx-tp8` Guard** | `ctx8k_c2` / `ctx16k_c8` + VRAM | `99.13` / `61.43 tok/s` (`p99_ITL 6.56 ms`) | **`97.76` / `60.58 tok/s`** (`p99_ITL 6.55 ms`, `66,515 MiB`) | **`-1.38%` geomean** (`0%` `p99_ITL` drift) | **`ACCEPTED`** |
| **`short-throughput` Guard** | `in512_c1` / `in512_c32` + VRAM | `168.06` / `1,105.03 tok/s` (`65,809 MiB`) | **`168.08` / `1,097.96 tok/s`** (`66,133 MiB`) | **`-0.32%` geomean** (`max_drift 2.125 <= 8.00`) | **`ACCEPTED`** |
| **`spec-dspark` Guard** | `spec_c1` / `spec_c8` (`DSpark K=5`) | `270.06` / `743.16 tok/s` (`med_ttft 114.02 ms`) | **`289.92` / `739.79 tok/s`** (`med_ttft 113.57 ms`, `68,845 MiB`) | **`+3.38%` geomean** (**`+7.35%` `spec_c1`**) | **`ACCEPTED`** |
