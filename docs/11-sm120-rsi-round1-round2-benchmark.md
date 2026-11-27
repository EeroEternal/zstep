# Phase 3.11: SM120 8×RTX 6000D Evolith × Gemini-4-Neon RSI Round 1–2 Benchmark & Analysis

- **Date**: `2026-11-27`
- **Hardware**: `8 × NVIDIA RTX 6000D` (`SM120`, `156 SMs`, `85,651 MiB` GDDR7/GPU, `CUDA 13.0`, Dual-Socket Xeon `4+4` PCIe Gen5 Switch topology, `106.1 GB/s` intra-switch PXB P2P)
- **Model**: `DeepSeek-V4.1-Flash` (`475.25 GiB`, `TP=8`, `kv_cache_dtype=fp8_ds_mla`, `b12x` + `MegaGate` + `SM120 DSA Decode` + `shm_pxb_bfp1`)
- **Incumbent Baseline**: `main@f2c23b1` (`4ba470bc1466`)
- **Raw Benchmark Artifacts**:
  - [`docs/benchmarks/phase311_6000d_scout_comm_and_dsa.json`](benchmarks/phase311_6000d_scout_comm_and_dsa.json)
  - [`docs/benchmarks/phase311_6000d_round1_round2_ab_verify.json`](benchmarks/phase311_6000d_round1_round2_ab_verify.json)
  - [`docs/benchmarks/phase311_6000d_round3_3track_results.json`](benchmarks/phase311_6000d_round3_3track_results.json)

---

## 1. Live Hardware Collective & Operator Scout Audit (`CUDA 13.0`)

Before mutating kernels or collective policies, we ran a direct 8-GPU hardware scout inside `vllm/vllm-openai:v0.31.0` + SM120 overlay (`phase311_6000d_scout_comm_and_dsa.json`):

1. **`TP=8` PCIe Collectives (`shm_pxb_bfp1` vs `NCCL` vs `flashinfer.comm.pcie_ipc_ar`)**:
   - `p2p_probe` confirmed `pxb_shm_enabled=True` with **`106.1 GB/s`** intra-switch bidirectional bandwidth.
   - Stock `vllm:v0.31.0` ships `flashinfer 0.7.0.post1` (only `UNSTAGED`, `STAGED`, `STAGED_RING`, `FLAT_STAGED`; upstream `COPY_ENGINE_*` variants are not present in `0.7.0.post1`), and `pcie_ws.tune()` deadlocks all 8 GPUs at 95W when sweeping cross-socket `UNSTAGED` PCIe BAR writes (`GPU 0 <-> GPU 4`).
   - At `M=4096` (`40 MiB` BF16), `zstep`'s incumbent `shm_pxb_bfp1` completes in **`914.37 µs`** — **`3.27×` faster than `NCCL` (`2,986.6 µs`)** and **`6.06×` faster than `pcie_ipc_ar STAGED_RING` (`5,544.4 µs`)**.
   - **Real Bottleneck Identified**: Even at `914.4 µs`, 80 AllReduces per `M=4096` prefill step cost **`73.15 ms`/step** (~25% of step latency), while `DeepSeekV41StepRunner._run_prefill_tbo` was **never triggered (`_tbo_steps == 0`)** under standard `--max-step-tokens 4096` because `prefill_tbo_min_tokens` defaulted to `6144 > 4096`.

2. **`sm120_dsa_indexer` Ragged Prefill Crossover (`topk=512, H=32`)**:
   - Isolated microbenchmark (`10/10` cases `exact_set_match=True` across `ratio=1` & `ratio=2`, FP8 & MXFP4): stock `fp8_fp4_mqa_logits` wins at `K=512` (`35.0 vs 39.6 µs`), crossover is at `K=1024` (`1.031×`), and `dsa_indexer_det_topk` wins at `K >= 2048` (`1.302×` @ `2K`, `1.117×` @ `8K`, `1.041–1.064×` @ `16K–32K`).
   - **End-to-End Ablation Lesson (`g0006`)**: In full 8-GPU `DeepSeek-V4.1-Flash` execution, DSA Indexer represents `< 0.5%` of total prefill step time; wiring `dsa_indexer_det_topk` into `SparseAttnIndexer.forward_cuda` measured **`-0.31%`** end-to-end (`ctx8k_c2 -0.10%`, `ctx16k_c8 -0.52%`) and was therefore **excluded** from the landed patch.

---

## 2. Four Landed Orthogonal Optimizations (`125 LOC`)

1. **Aligned & Pre-Warmed Pure-Prefill Two-Batch Overlap (`env_config.py`, `deepseek_v41.py` — `52 LOC`)**:
   - Tracks `explicit_prefill_tbo_min_tokens` so `WorkerRuntimeConfig.from_env({}).prefill_tbo_min_tokens == 6144` remains backward-compatible while `DeepSeekV41StepRunner` activates `_effective_tbo_min_tokens = 2048` automatically when `ZSTEP_PREFILL_TBO_MIN_TOKENS` is unset and `max_step_tokens >= 2048`.
   - Allocates 16-byte-aligned `self._tbo_comp_bt_gpu` per microbatch (fixing `dsv41_comp_ring_slot` Triton recompilations caused by `self.comp_bt_gpu` `(ms, 1)` 4-byte row stride when `r_start % 4 != 0`).
   - Gates runtime TBO dispatch to pure 256-token-aligned prefill batches (`min_q_len > 1 and total_tokens % 256 == 0`) and pre-warms `_run_prefill_tbo` inside `_warmup_and_capture_decode_graphs()` (`case_idx > 0`), guaranteeing **`TRITON_RECOMPILATIONS=0`** and zero runtime TileLang JIT stalls.
2. **156-SM `sm12x_mega_gate` Fallback & PDL (`b12x_dsv41.py`)**:
   - Enables Programmatic Dependent Launch (`enable_pdl=True`) on `sm_120*` and falls back to `select_config(..., sms=110)` when `156`-SM heuristic configs exceed `sms` or violate `(hidden // BLOCK_K) % num_split_k == 0`, recovering full MoE gate coverage across `M` buckets.
3. **`shm_allreduce_kernels.cu` 64-Bit Offset, Local `__ldcs`, and 8-Way Binary-Tree FP32 Sum**:
   - Hoists 64-bit `slot_offset`, uses streaming `__ldcs` on local `input_vec` while preserving cache-volatile `ld_cv_uint4` on `peer_shm`, and replaces serial 8-rank FP32 accumulation with balanced pairwise tree summation `((f0+f1)+(f2+f3))+((f4+f5)+(f6+f7))` to halve FADD dependency chains.
4. **Zero-Allocation `DeepSeekV41DSparkDrafter` Step Loop (`deepseek_v41_dspark.py`)**:
   - Pre-caches `markov_w2_shard_t` and pre-allocates `markov_vec_buf` so `torch.index_select(..., out=markov_vec)` + `torch.addmm(..., out=step_logits)` execute with zero per-step tensor allocations or `copy_` launches.

---

## 3. 3-Round Interleaved Cold `AB/BA` Verification Results (`8 × RTX 6000D`)

Every verification cell below averages **3 interleaved cold container pairs (`AB`, `BA`, `AB` = 6 full 8-GPU runs per candidate)** with 100% exact greedy token parity (`prime_squares_table` single, batched, and chunked) and `TRITON_RECOMPILATIONS=0`:

### 3.1 Short-Context Multi-Concurrency (`short-throughput`, Candidate `g0002-e1ad0d86`)
- **3-Round Verify Geomean**: **`+0.45%`** (`paired_lcb_2se = +0.33% > 0`)
  - `in512_c16`: **`+1.03%`** (TPOT `12.56 → 12.42 ms`)
  - `in512_c128`: **`+0.68%`**
  - `in1024_c32`: **`+0.06%`**
  - `in1024_c128`: **`+0.05%`**

### 3.2 Long-Context `8K–32K` (`longctx-tp8`, `ZSTEP_MAX_MODEL_LEN=33792`, `kv_pages=2560`)

| Tier | Baseline (`f2c23b1`) | Pure TBO (`g0010`, `52 LOC`) | Combined 4-Way (`g0009`, `125 LOC`) | Combined Throughput Gain | Combined Mean TTFT Delta |
|---|---:|---:|---:|---:|---:|
| **Screen `ctx8k_c2`** (`8192:64`) | `96.15` out / `12,403.4` tot tok/s (`970.7 ms` TTFT) | `98.72` out (`+2.67%`, `935.7 ms`) | `98.55` out / `12,713.2` tot tok/s (`938.4 ms` TTFT) | **`+2.50%`** | **`-3.32%`** (`-32.3 ms`) |
| **Screen `ctx16k_c8`** (`16384:64`) | `50.35` out / `12,939.5` tot tok/s (`4372.6 ms` TTFT) | `50.70` out (`+0.71%`, `4323.5 ms`) | `50.55` out / `12,992.1` tot tok/s (`4331.2 ms` TTFT) | **`+0.41%`** | **`-0.95%`** (`-41.4 ms`) |
| **Verify `ctx8k_c16`** (`8192:128`, 3-rd mean) | `223.25` out tok/s (`2889.0 ms` TTFT) | `223.42` out (`+0.35%`) | `224.06` out tok/s (`2868.4 ms` TTFT) | **`+0.48%`** | **`-0.71%`** (`-20.6 ms`) |
| **Verify `ctx16k_c12`** (`16384:128`, 3-rd mean) | `100.04` out tok/s (`5515.3 ms` TTFT) | `100.32` out (`+0.16%`) | `100.43` out tok/s (`5478.9 ms` TTFT) | **`+0.37%`** | **`-0.66%`** (`-36.4 ms`) |
| **Verify `ctx32k_c4`** (`32768:128`, 3-rd mean) | `48.12` out tok/s (`5071.2 ms` TTFT) | `48.40` out (`+0.48%`) | `48.52` out tok/s (`4991.1 ms` TTFT) | **`+0.91%`** | **`-1.58%`** (`-80.1 ms`) |
| **Verify `ctx32k_c8`** (`32768:128`, 3-rd mean) | `55.13` out tok/s (`7235.2 ms` TTFT) | `55.40` out (`+0.33%`) | `55.48` out tok/s (`7151.8 ms` TTFT) | **`+0.60%`** | **`-1.15%`** (`-83.4 ms`) |
| **3-Round Verify Geomean / `lcb_2se`** | — | **`+0.33%`** / **`+0.26%`** | **`+0.59%`** (12/12 wins) | **`+0.59%`** | **`paired_lcb_2se = +0.48%`** |

---

## 4. Round 3 Parallel 3-Track Exploration Results (vs `8ad676f`)

1. **Track 1 — Mixed-Step 16-Aligned TBO (`TP=8`, `longctx-tp8`, `17 LOC`)**:
   - Snapping `mid` to 16-token multiples (`((total_tokens // 2) + 8) & ~15`) and enabling TBO across mixed decode+prefill steps (`max_q_len >= 512`) further improved `ctx16k_c8` from `12,996.2 → 13,069.5 tok/s` (**`+0.56%`** over `8ad676f`, **`+1.00%` cumulative** over `f2c23b1`), with `TRITON_RECOMPILATIONS=0` and 100% Greedy parity (`ctx8k_c2`: `-0.02%`, zero mixed steps).
2. **Track 2 — Topology Comparison `PP=2×TP=4` vs `TP=8` (`sm120-8x6000d-pp2tp4`, `46 LOC`)**:
   - Enabling `ar.world_size in (4, 8)` `use_wave_mhc` + double-buffered P2P improved `PP=2×TP=4` `ctx8k_c2` by **`+0.71%`** (`8,586.3 → 8,647.2 tok/s`).
   - Head-to-head topology comparison confirms **`TP=8` (`12,749.5 / 13,069.5 tok/s`) outperforms `PP=2×TP=4` (`8,647.2 / 7,620.1 tok/s`) by `+47.4%` to `+71.5%`** on `zstep` because `shm_pxb_bfp1` (`914 µs`) + TBO already eliminated the `~3 ms` NCCL PCIe AllReduce bottleneck that motivated `PP=2` on stock vLLM.
3. **Track 3 — Speculative Decoding `spec-dspark` (`ZSTEP_SPEC_METHOD=dspark`, `num_spec_tokens=5`, `72 LOC`)**:
   - Baseline `zstep@8ad676f` achieves **`177.61` out tok/s (`5.29 ms` TPOT)** at `spec_c1` and **`820.38` out tok/s (`8.08 ms` TPOT)** at `spec_c8`.
   - Capturing `(b, k)` contiguous draft packing inside `draft_graphs[b]` + single contiguous D2H DMA + zero-`.tolist()` NumPy slicing improved **`spec_c8` to `825.20` out tok/s (`+0.59%`, TPOT `8.08 → 8.02 ms`, `-0.74%`)** with `TRITON_RECOMPILATIONS=0` and 100% Greedy parity.

