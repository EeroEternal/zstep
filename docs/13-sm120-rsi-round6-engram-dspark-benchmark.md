# Phase 3.13: SM120 Asynchronous THP `Engram` Offload (`-12 GiB/GPU`), Zero-Host-Stall Fused `DSpark` Pipeline & Boundary-Safe Ragged Verify Benchmark

- **Date**: `2026-11-28`
- **Hardware**: `8 × NVIDIA RTX 6000D` (`SM120`, `156 SMs`, `99 KB` dynamic SMEM/SM, `112 MiB` L2, `85,651 MiB` GDDR7/GPU, `CUDA 13.0`, Dual-Socket Xeon `4+4` PCIe Gen5 Switch topology at `106.1 GB/s` intra-switch PXB P2P)
- **Model**: `DeepSeek-V4.1-Flash` (`475.25 GiB`, `TP=8`, `kv_cache_dtype=fp8_ds_mla`, `PrecisionTrack.LOSSLESS`)
- **Incumbent Baseline**: `main@cfacb76` + `--speculative-method dspark` (`DSpark K=5`, remote SHA `76c36b9929e3`)
- **Raw Benchmark Artifacts**:
  - [`docs/benchmarks/round6_stage1_engram_dspark_scout.json`](benchmarks/round6_stage1_engram_dspark_scout.json)
  - [`docs/benchmarks/round6_stage2_dspark_depth_conf_screen_v3.json`](benchmarks/round6_stage2_dspark_depth_conf_screen_v3.json)
  - [`docs/benchmarks/round6_multi_regime_abba_results.json`](benchmarks/round6_multi_regime_abba_results.json)

---

## 1. Stage 1 & Stage 2: Hardware Scout & Empirical Speculative Calibration

1. **`Engram` VRAM Audit & 8-CTA Background UVA Prefetch (`P3`)**:
   - `DeepSeek-V4.1-Flash` instantiates two `16M × 2048` BF16 `Engram` embedding tables (`layers.1.engram.embed` and `layers.14.engram.embed`), each occupying **`11.802 GiB/GPU`** on `TP=8` (`94.4 GiB` per table across 8 GPUs) while fetching only **`792 B/token/GPU`** (`~18.2 µs` solo PCIe Gen5 x16 UVA gather).
   - Offloading `layers.14.engram.embed` to 2 MiB THP host-pinned memory (`madvise(MADV_HUGEPAGE)`) while keeping `layers.1.engram.embed` on HBM provides a **14-layer hiding window** (`Layers 0..13`, `1.5–4.0 ms`) across decode `CUDAGraph`s, verify `CUDAGraph`s, and TBO prefill.
   - Capping `emb._num_sms = min(int(emb._num_sms), 16)` (**`8` background CTAs** = `5%` of `156` SMs on `_prefetch_stream`, down from `78` CTAs) eliminates Layer-0 SM wave contention and achieves `0.0` `max_diff` bit-exactness with zero non-speculative throughput loss.
   - Guarded automatically by `/proc/meminfo` `MemAvailable >= 320 GiB` (`env_config.py`) so the `475 GiB` safetensors OS page cache is never evicted on RAM-constrained hosts.
2. **Empirically Excluded Speculative Variants (Preserved per `AGENTS.md`)**:
   - **`K=7` (`cand_v1_k7`)**: Regressed `-17.21%` vs `base_dspark_k5` due to `B × 8` verify-row GEMM saturation at `c4..c16` and lower tail acceptance on draft positions `6..7`. Excluded.
   - **`K=6` (`cand_v3_k6`)**: Reached `+33.26%` geomean (`spec_c8: 813.65 tok/s`), trailing `cand_v3_k5` (`+36.10%` geomean; `spec_c2: 517.05 vs 474.64`, `spec_c4: 769.78 vs 727.16`, `spec_c16: 1,256.81 vs 1,228.84`). Excluded.
   - **`K=5, conf=0.12` (`cand_v3_k5_conf12`, `DSparkConfidenceHead`)**: Reached `+33.35%` geomean, outperforming `conf=0.0` at high batch (`spec_c8: 798.86 vs 766.99 tok/s`, `spec_c16: 1,293.40 vs 1,256.81 tok/s`) while trailing at `c1..c4`. Retained behind opt-in `ZSTEP_DSPARK_CONF_THRESHOLD` (`default 0.0`).

---

## 2. Landed Zero-Host-Stall `DSpark` Pipeline & Boundary-Safe Ragged Verify (`P2`)

1. **In-GPU `dspark_verify_prep_kernel` + Captured `post_verify_draft_graphs[v_bucket]`**:
   - Eliminates the inter-stage `runner.finish()` host `synchronize()`, Python `verify_batch` loop, CPU `ctx_slot_mapping` / `_draft_batch` NumPy preparation, and intermediate H2D copies.
   - Computes exact greedy acceptance length `acc_len`, gathers `1..K+1` verified target `aux_hidden_states` + tokens + positions + SWA context slot mappings on GPU, updates per-slot Engram rolling 3-token lookback on GPU, writes verified context KV + `CompressorStateCache`, and runs the 1-pass `K`-token noncausal `DSpark` draft inside a single captured `post_verify_draft_graphs[v_bucket]` replayed immediately after `verify_graphs[v_bucket]`.
   - Rust `ServingEngine::commit_verified_round` (`crates/zs-scheduler/src/serving.rs`) commits accepted tokens, retires unused pre-allocated KV pages, and installs next-round GPU-produced draft tokens without a second worker round-trip.
2. **Eliminating `85–204 ms` `128`-Token Page-Boundary Eager Fallbacks**:
   - Previously, whenever `max_req_len % 128 == 0` (`640, 768, 1280`), the final ragged verify step (`p_max = 638`, `q_len < 6`) failed `(p_max_pad // 128) == (p_max // 128)` and dropped the entire batch into `~95–200 ms` eager Python.
   - Relaxing `_can_use_ragged_verify_graph` to `(p_max_pad // 128) < bt_width` (safe because `CompressorStateCache.CAPACITY = 8 >= width + 1`, `total_128_blocks = num_kv_pages + 2` has permanent zeroed `mla_dummy_vb`/`swa_dummy_vb`, causal rows `0..ql-1` never read past `p_max`, and padded rows carry `slot_mapping = -1`) plus per-sequence `eff_max_ctx` clamping in `dspark_verify_prep_kernel` keeps **100% of speculative steps inside CUDA Graphs**.

---

## 3. Full 3-Round Interleaved Cold `AB/BA` Verification (`50.5 min`, Candidate `g0001-28ab22ed`)

All runs passed `RedTeam` (`spec_pins_verified=true`), `DispatchVerifier` (`8,640` pairs, `0` violations), 9/9 Z3 `FormalVerifier` clauses (`CLOSED`, including `Z3_DSPARK_BT_CACHE_HORIZON=PROVED`), `TRITON_RECOMPILATIONS=0`, `Trace Conformance = CONFORMANT`, and 100% greedy token parity (`prime_squares_table` single, batched, chunked):

| Regime / Phase | Tier / Metric | Baseline (`cfacb76` `DSpark K=5`) | Candidate (`g0001-28ab22ed`) | Measured Delta | Gate Verdict |
|---|---|---:|---:|---:|---|
| **`spec-dspark` Peak VRAM** | `pynvml` Peak GPU Memory | `80,957 MiB/GPU` | **`68,891 MiB/GPU`** | **`-12,066 MiB/GPU` (`-11.78 GiB/GPU`)** | **`PASS` (`<= 74,000 MiB` gate)** |
| **`spec-dspark` Screen (`repeats=1`)** | `spec_c1` (`1:8:512:256`)<br>`spec_c8` (`8:24:512:128`) | `203.91 tok/s` (`TPOT 4.35 ms`, `p99 10.07 ms`)<br>`423.76 tok/s` (`TPOT 14.88 ms`, `p99 166.36 ms`) | **`239.46 tok/s`** (`TPOT 3.62 ms`, `p99 9.85 ms`)<br>**`729.01 tok/s`** (`TPOT 8.53 ms`, **`p99 93.07 ms`**) | **`+17.43%`**<br>**`+72.03%` (`-44.1%` `p99_ITL`)**<br>**Screen Geomean: `+42.14%`** | **`PASS`** |
| **`spec-dspark` 3-Round Cold `AB/BA`** | `spec_c2` (`2:16:512:256`)<br>`spec_c4` (`4:24:512:128`)<br>`spec_c16` (`16:48:1024:256`) | `404.39 tok/s` (`TPOT 4.41 ms`, `p99 36.65 ms`)<br>`499.59 tok/s` (`TPOT 7.38 ms`, `p99 162.53 ms`)<br>`939.60 tok/s` (`TPOT 13.33 ms`, `p99 173.11 ms`) | **`528.42 tok/s`** (`TPOT 3.24 ms`, **`p99 11.60 ms`**)<br>**`728.99 tok/s`** (`TPOT 4.67 ms`, **`p99 14.83 ms`**)<br>**`1,285.54 tok/s`** (`TPOT 10.02 ms`, **`p99 146.76 ms`**, R0 peak **`1,343.70`**) | **`+30.67%` mean** (`-68.4%` `p99_ITL`)<br>**`+45.92%` mean** (**`-90.9%` `p99_ITL`**)<br>**`+36.82%` mean** (`-15.2%` `p99_ITL`)<br>**Verify Geomean: `+34.57%`** (`lcb_2se +27.27%`, `net_gate 1.2303`) | **`ACCEPTED`** |
| **`short-throughput` Guard** | `in512_c1` / `in512_c32` + VRAM | `167.90` / `1,099.24 tok/s` (`78,187 MiB`) | **`168.18` / `1,099.31 tok/s`** (**`66,093 MiB`**) | **`+0.09%` geomean**, **`-12,094 MiB/GPU`** | **`ACCEPTED`** |
| **`longctx-tp8` Guard** | `ctx8k_c2` / `ctx16k_c8` + VRAM | `99.39` / `61.50 tok/s` (`78,549 MiB`) | **`98.93` / `61.42 tok/s`** (**`66,535 MiB`**) | **`-0.30%` geomean**, **`-12,014 MiB/GPU`** | **`ACCEPTED`** |
