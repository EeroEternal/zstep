# zstep

**Zero-magic, step-explicit LLM inference engine (`1` Pure-Rust Master + `8` Explicit `sm_120` GPU Workers over POSIX `/dev/shm` + `eventfd`).**

`zstep` combines:
1. **Pure Rust Master Control Plane**: single-master orchestration over POSIX shared memory + `eventfd`, RAII slot/KV leases, stateless 3-token Engram lookback windows, piggyback mixed decode + chunked-prefill scheduling, `in_flight_depth=1` pure-decode pipelining, and zero `pyo3`/GIL or CUDA coupling in the master.
2. **Explicit Step Data Plane**: flat model structs, explicit tensor shapes, refresh-in-place GPU buffers, full-model 40-layer CUDA Graphs, SM120 `FlashInfer` autotuned tactics, deterministic global Top-2 logit argmax, and runtime zero-recompilation guards.
3. **`sm_120` NUMA-Aware PCIe Collectives & Kernels**: `shm_v5` per-rank NUMA-local host-mapped SHM AllReduce / AllGather / Argmax with size-aware `NCCL_LOC` routing for non-P2P systems, plus Triton / CUDA C++ / TileLang / DeepGEMM operator primitives.

> **Hardware scope**: End-to-end uncached (`--no-enable-prefix-caching`) `TP=8` serving is verified on two non-P2P `sm_120` PCIe Gen5 hosts (`can_device_access_peer == 0` across all `8 × 8` pairs):
> 1. **`8 × NVIDIA RTX 6000D` (`85,651 MiB` GDDR7 per GPU, 2-NUMA `[0,0,0,0,1,1,1,1]`)** running **`DeepSeek-V4.1-Flash` (`1017`, `475.25 GiB` MXFP8/MXFP4/BF16, device-resident Engram)** vs `vllm/vllm-openai:v0.31.0`.
> 2. **`8 × NVIDIA GeForce RTX 5090` (`32 GB` GDDR7 per GPU, 4-NUMA `[0,0,0,1,2,2,2,3]`)** running **`Qwen/Qwen3.5-35B-A3B` (`67 GB` BF16)** vs `vllm/vllm-openai:v0.33.0`.
>
> All collective and serving results are scoped strictly to non-P2P PCIe topologies and are not extrapolated to NVLink or PCIe P2P systems.

## Repository Layout

```text
zstep/
├── crates/
│   ├── zs-wire/                      # #[repr(C)] SHM layout, geometry & result ring
│   ├── zs-shm/                       # RAII shm_open + mmap + eventfd transport
│   ├── zs-kv/                        # BlockPool and standalone pure-attention Radix KV
│   ├── zs-scheduler/                 # WorkerCoordinator + mixed continuous-batching ServingEngine
│   └── zs-server/                    # zstep synthetic/parity/benchmark binary
├── worker/
│   ├── zstep_worker/
│   │   ├── numa_pin.py               # PCIe BDF -> NUMA discovery and CPU pinning
│   │   ├── shm_bridge.py             # Zero-copy mmap + acquire/release atomics
│   │   ├── runtime_loop.py           # Worker loop + zero-recompilation enforcement
│   │   ├── models/
│   │   │   ├── deepseek_v41.py       # Explicit 40-layer DeepSeek-V4.1-Flash TP=8 runner
│   │   │   └── qwen35_moe.py         # Explicit 40-layer Qwen3.5-35B-A3B TP=8 runner
│   │   └── kernels_sm120/            # shm_v5 collectives, b12x SM120 MoE/FP8, GDN/GQA/MoE
│   └── tests/                        # SHM, Coordinator E2E, Phase 0, Phase 2 & Phase 3 harnesses
└── docs/
    ├── 01-advisor-analysis.md
    ├── 02-architecture.md
    ├── 03-shm-protocol.md
    ├── 04-phase0-sm120-plan.md
    ├── 05-phase1-shm-engine.md
    ├── 06-phase2-qwen35-benchmark.md
    ├── 07-phase3-dsv41-6000d-benchmark.md
    ├── 08-phase3-dsv41-optimization-plan.md
    └── benchmarks/                   # 27 primary + 19 raw vLLM JSON artifacts
```

## Current Status

* **Phase 3 / 3.5 / 3.6 — `DeepSeek-V4.1-Flash` (`8 × RTX 6000D`, `TP=8`)**:
  * **Size-Aware `shm_v5` + `NCCL_LOC` Collectives** ([`phase0_6000d_ar_shmv5.json`](docs/benchmarks/phase0_6000d_ar_shmv5.json)): across all decode CUDA Graph buckets (`M=1..128`, `H=5120` BF16), `shm_v5 (blocks=8)` beats `NCCL_LOC` by **`1.19×..3.38×`** (`14.48 µs` vs `48.95 µs` at `M=1`; `14.08 µs` / `3.48×` at `blocks=2`), while eager mixed/prefill payloads (`M > 256`) automatically route to multi-SM `NCCL_LOC` (`3,944.79 µs` vs `4,660.95 µs` at `M=4096`).
  * **Selective `b12x` SM120 Compute Integration (`Phase 3.6`, [`b12x_dsv41.py`](worker/zstep_worker/kernels_sm120/b12x_dsv41.py))**: mounts `flashinfer/experimental/b12x` from [`local-inference-lab/flashinfer`](https://github.com/local-inference-lab/flashinfer) tag `flashinfer-cu134-sm120-beta-79819b112f8bda7d09709c24bb31de16d7ce59c6` (`79819b112f8bda7d09709c24bb31de16d7ce59c6`, enforced via `ZSTEP_REQUIRE_B12X=1`) into `vllm/vllm-openai:v0.31.0`, replacing all 40 routed `MXFP4` MoE layers (`I_tp8=288` unpadded, saving ~10 GiB VRAM/GPU) and 170 2D `[32, 32] UE8M0` FP8 dense linears (`rows > 512` Cutlass fallback on `fused_wqa_wkv`) over static 256 B-aligned GPU arenas for zero-allocation CUDAGraph replay.
  * **Symmetric Pure-Engine & HTTP 10 / 10 Tier Sweep vs No-APC `vLLM 0.31.0`**: Benchmarked against **both** in-process `vLLM 0.31.0` Pure Engine ([`bench_vllm_pure_engine.py`](worker/tests/phase3/bench_vllm_pure_engine.py): `AsyncLLMEngine` + `TokensPrompt` + `detokenize=False`, zero HTTP/tokenizer/detokenizer/SSE, identical SplitMix64 `int32` prompt IDs) and `vllm bench serve` `/v1/completions` HTTP. `zstep 3.6 + b12x` beats pure-engine `vLLM 0.31.0` across **all 10 / 10 tiers (`+14.67%..+49.74%`)** and beats `zstep 3.5` by **`+16.18%..+22.86%`** at `c=8..128` — reaching **`8.93..9.07 ms` single-stream `Mean TPOT`** (`102.27..103.99 tok/s` at `output=256`) and **`1,686.53 tok/s`** (`+49.74%`) / **`1,169.71 tok/s`** (`+35.03%`) at `c=128`. Explicit latency trade-offs: at `c=16` (`512->128`) and `c=32`, `--max-mixed-step-tokens 3072` trades higher `P99 ITL` (`328..391 ms` vs `vLLM` `26..297 ms`) for `2.8×..5.4×` lower `Median TTFT`; at `c=1`, eager `b12x` Python `bind()` adds `~20..67 ms` to `Median TTFT` (`184.9..191.2 ms` vs `117.6..171.7 ms` in Phase 3.5).
  * **Deterministic Global Top-2 Greedy Audit**: Runtime emits **`TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=5`** and clean `/dev/shm` teardown; both [`phase3_greedy_parity_report.json`](docs/benchmarks/phase3_greedy_parity_report.json) and [`phase36_b12x_greedy_parity_report.json`](docs/benchmarks/phase36_b12x_greedy_parity_report.json) achieve **`128 / 128` (`100.00%`) exact token parity** across single, batched, and `chunk=24` mixed-prefill schedules on `prime_squares_table`, while honestly preserving razor-thin BF16 ULP tie divergences (`all_matched=false`) on open-ended base-model prompts.
* **Phase 2 — `Qwen3.5-35B-A3B` (`8 × RTX 5090`, `TP=8`)**:
  * **6 / 6 Uncached Throughput Sweep vs No-APC `vLLM 0.33.0`**: With `--no-enable-prefix-caching` and per-`(round, tier)` isolated seeds, `zstep` leads on uncached output throughput across all 6 tiers (**`+11.00%..+54.08%`**), while explicitly reporting the `max_prefill_chunk=2048` tail-latency tradeoff at `c=32/128`.
  * **Greedy Audit**: Runtime emits **`TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=17`**. `chunked` matches `vLLM` **448/448 tokens (100% across all 5 cases)**; remaining `single`/`batched` flips occur strictly at `<= 2` BF16 ULPs (`margin <= 0.25` at logit `~25`), preserved honestly as `all_matched=false` in [`phase2_greedy_parity_report.json`](docs/benchmarks/phase2_greedy_parity_report.json).
* **Phase 1 / Shared-Memory Control Plane**: 200-step 8-GPU measurement records **`4.690 µs/step`** Rust preparation (`0.166%` of GPU step time, `overlap_speedup=1.000269×`).
* **KV scope**: [`crates/zs-kv/src/radix.rs`](crates/zs-kv/src/radix.rs) (`PrefixTree` / `KvCacheManager`) is a standalone pure-attention component and is **not wired into `DeepSeek-V4.1-Flash` or `Qwen3.5-35B-A3B` serving**; both runners use `BlockPool` physical-page RAII only.

## Phase 3.6 (`+ b12x`) & Phase 3.5 Warm `n=3` Uncached Serving (`DeepSeek-V4.1-Flash`, `8 × RTX 6000D`)

Unified resident config across all tiers: `allreduce_blocks=8`, `num_kv_pages=2560`, `max_seqs=128`, `max_step_tokens=max_decoder_rows=4096`, `max_prefill_chunk=2048`, `max_mixed_step_tokens=3072`, `max_pages_per_seq=256`, 16 warmup prompts, `n=3` round-major interleaved repeats (`enable_prefix_caching=False` on `vLLM 0.31.0`, isolated per-`(round, tier)` seeds). **`vLLM Pure`** runs `AsyncLLMEngine` (`OPENAI_API_SERVER`, `max_num_batched_tokens=8192`) in-process with `TokensPrompt` + `detokenize=False` (no HTTP, no tokenizer, no detokenizer, no SSE — exact same `int32` token-ID boundary as `zstep`), alongside **`vLLM HTTP`** (`vllm bench serve` `/v1/completions`).

### Suite 1: Fixed `512 -> 128` (`c=1, 8, 16, 32`)

Raw artifacts: [`zstep 3.6 + b12x`](docs/benchmarks/phase36_zstep_b12x_dsv41_tp8_in512_out128.json), [`zstep 3.5`](docs/benchmarks/phase3_zstep_dsv41_tp8_in512_out128.json), [`vLLM Pure`](docs/benchmarks/phase36_vllm_pure_dsv41_tp8_in512_out128.json), [`vLLM HTTP`](docs/benchmarks/phase3_vllm_dsv41_tp8_in512_out128.json).

| Tier | `vLLM` HTTP (`tok/s`) | `vLLM` Pure (`tok/s`) | `zstep` 3.5 (`tok/s`) | **`zstep` 3.6 + `b12x` (`tok/s`)** | 3.6 vs `vLLM` Pure | Mean TPOT `vPure` / `z3.5` / **`z3.6`** (`ms`) | Med TTFT `vPure` / `z3.5` / **`z3.6`** (`ms`) | P99 ITL `vPure` / **`z3.6`** (`ms`) |
| :--- | ---: | ---: | ---: | ---: | :--- | :---: | :---: | :---: |
| `c1` | 83.97 ± 0.05 | 83.59 ± 0.01 | **96.18 ± 0.57** | 95.85 ± 0.65 | **+14.67%** | 10.94 / 9.55 / **9.05** | 141.37 / **117.56** / 185.21 | 11.39 / **9.28** |
| `c8` | 351.50 ± 10.04 | 348.18 ± 8.30 | 377.50 ± 2.52 | **441.70 ± 1.60** | **+26.86%** | 19.02 / 17.14 / **14.23** | 590.28 / 535.78 / **510.32** | 18.80 / **14.79** |
| `c16` | 509.40 ± 3.55 | 492.61 ± 2.27 | 527.00 ± 1.87 | **612.24 ± 2.13** | **+24.29%** | 25.63 / 26.51 / **22.47** | 1,084.36 / 410.34 / **386.49** | **25.97** / 327.96 |
| `c32` | 635.98 ± 1.49 | 617.07 ± 1.10 | 663.78 ± 3.57 | **800.74 ± 1.01** | **+29.76%** | 41.13 / 43.11 / **35.12** | 1,430.66 / 411.54 / **387.05** | **86.20** / 386.75 |

### Suite 2: Six-Tier Interleaved (`in512` / `in1024` × `output=256`)

Raw artifacts: [`zstep 3.6 + b12x`](docs/benchmarks/phase36_zstep_b12x_dsv41_tp8_6tier.json), [`zstep 3.5`](docs/benchmarks/phase3_zstep_dsv41_tp8_6tier.json), [`vLLM Pure`](docs/benchmarks/phase36_vllm_pure_dsv41_tp8_6tier.json), [`vLLM HTTP`](docs/benchmarks/phase3_vllm_dsv41_tp8_6tier.json).

| Tier | `vLLM` HTTP (`tok/s`) | `vLLM` Pure (`tok/s`) | `zstep` 3.5 (`tok/s`) | **`zstep` 3.6 + `b12x` (`tok/s`)** | 3.6 vs `vLLM` Pure | Mean TPOT `vPure` / `z3.5` / **`z3.6`** (`ms`) | Med TTFT `vPure` / `z3.5` / **`z3.6`** (`ms`) | P99 ITL `vPure` / **`z3.6`** (`ms`) |
| :--- | ---: | ---: | ---: | ---: | :--- | :---: | :---: | :---: |
| `in512_c1` | 87.24 ± 0.07 | 87.06 ± 0.01 | 98.94 ± 0.33 | **103.99 ± 1.10** | **+19.45%** | 10.97 / 9.68 / **8.93** | 141.86 / **118.68** / 184.93 | 11.41 / **9.27** |
| `in512_c16` | 573.38 ± 6.84 | 555.89 ± 1.34 | 609.53 ± 7.15 | **727.06 ± 1.62** | **+30.79%** | 25.32 / 24.15 / **20.01** | 1,086.45 / 410.82 / **386.86** | 26.12 / **19.09** |
| `in512_c128` | 1,152.03 ± 6.34 | 1,126.34 ± 1.87 | 1,372.77 ± 2.51 | **1,686.53 ± 7.96** | **+49.74%** | 99.30 / 82.32 / **65.72** | 3,066.32 / 831.11 / **779.52** | 1,021.41 / **389.59** |
| `in1024_c1` | 84.20 ± 0.02 | 84.57 ± 0.00 | 95.24 ± 0.09 | **102.27 ± 0.18** | **+20.92%** | 11.17 / 9.87 / **9.07** | 177.89 / **171.66** / 191.20 | 11.56 / **9.47** |
| `in1024_c32` | 614.93 ± 3.52 | 605.12 ± 1.35 | 649.63 ± 1.94 | **778.39 ± 4.56** | **+28.63%** | 42.27 / 44.18 / **36.51** | 2,127.63 / 415.61 / **391.62** | **296.51** / 391.34 |
| `in1024_c128` | 878.86 ± 4.26 | 866.25 ± 1.06 | 1,002.13 ± 3.22 | **1,169.71 ± 1.06** | **+35.03%** | 122.27 / 108.63 / **92.32** | 3,107.53 / 837.54 / **787.39** | 1,035.90 / **394.26** |

## Phase 2 Warm Single-Config `n=3` Uncached Serving (`Qwen3.5-35B-A3B`, `8 × RTX 5090`)

Both engines run **without prefix caching** (`vLLM 0.33.0` launched with `--no-enable-prefix-caching`) and use distinct deterministic random seeds per `(round, tier)`. While `zstep` leads on uncached output throughput across all six tiers (`+11.00%..+54.08%`), at `c=32/128` `zstep`'s `max_prefill_chunk=2048` / `max_step_tokens=4096` trades `~2×` higher `P99 ITL` (`223.5..242.5 ms` vs `vLLM`'s `109.8..114.4 ms`) for higher prefill/total throughput, and at `in1024_c32` `zstep` has slower `Mean TPOT` (`12.72 vs 11.29 ms`). Raw artifacts: [`zstep`](docs/benchmarks/phase2_zstep_qwen35_35b_a3b_tp8.json), [`vLLM 0.33.0`](docs/benchmarks/phase2_vllm_qwen35_35b_a3b_tp8.json), and [`vLLM rounds`](docs/benchmarks/phase2_vllm_raw/).

| Tier | zstep Output (`tok/s`) | vLLM Output (`tok/s`) | Delta / throughput verdict | Mean TPOT z / v (`ms`) | z Median TTFT / submit-TTFT (`ms`) | v Median TTFT (`ms`) | z P99 ITL (`ms`) | v P99 ITL (`ms`) |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| `in512_c1` | **261.53 ± 24.43** | 192.24 ± 10.22 | +36.05% / **zstep faster** | 3.68 ± 0.34 / 4.66 ± 0.22 | 50.42 ± 11.10 / 3,016.61 ± 315.79 | 137.34 ± 17.39 | **4.12 ± 0.41** | 6.51 ± 0.93 |
| `in512_c16` | **2,037.92 ± 51.95** | 1,700.98 ± 12.13 | +19.81% / **zstep faster** | 6.87 ± 0.24 / 7.62 ± 0.02 | 209.37 ± 1.82 / 2,252.67 ± 56.82 | 463.87 ± 17.91 | **6.98 ± 0.14** | 10.40 ± 0.28 |
| `in512_c128` | **5,021.65 ± 104.06** | 3,259.04 ± 5.72 | +54.08% / **zstep faster** | 21.46 ± 0.50 / 32.66 ± 0.17 | 210.86 ± 1.69 / 3,484.02 ± 21.72 | 832.91 ± 181.55 | 223.50 ± 2.30 | **114.35 ± 1.68** |
| `in1024_c1` | **268.95 ± 1.47** | 197.31 ± 0.59 | +36.31% / **zstep faster** | 3.42 ± 0.02 / 4.54 ± 0.02 | 79.91 ± 1.33 / 8,659.99 ± 46.85 | 132.48 ± 1.45 | **3.72 ± 0.03** | 6.01 ± 0.10 |
| `in1024_c32` | **2,174.52 ± 13.18** | 1,959.07 ± 29.76 | +11.00% / **zstep faster** | 12.72 ± 0.07 / **11.29 ± 0.09** | 223.18 ± 1.23 / 11,030.49 ± 27.60 | 1,034.01 ± 33.94 | 230.63 ± 1.34 | **109.76 ± 1.33** |
| `in1024_c128` | **2,982.59 ± 54.67** | 2,565.79 ± 18.11 | +16.24% / **zstep faster** | 35.97 ± 0.73 / 41.20 ± 0.24 | 226.59 ± 3.82 / 14,742.53 ± 211.99 | 711.86 ± 103.75 | 242.46 ± 3.61 | **113.56 ± 1.40** |

## Verification and Reproduction

The verification gate passes `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace` (**41 tests**), `ruff check worker/` (pinned via [`ruff.toml`](ruff.toml) and [`.github/workflows/ci.yml`](.github/workflows/ci.yml)), and `pytest` (`test_shm_layout_parity.py`, `test_07_shmv5_alternating_stress_8gpu.py`, `test_qwen35_ops_sm120.py`, `test_qwen35_e2e_tp8.py`), plus end-to-end Top-2 greedy audits (`TRITON_RECOMPILATIONS=0`, `rc=0` default / `--require-exact-cases prime_squares_table`, opt-in `--max-greedy-mismatch-margin 0.25` and `--require-exact-greedy true`) and `/dev/shm` RAII checks on both `8 × RTX 5090` and `8 × RTX 6000D`.

Unified reproduction commands (including `git archive` source trees without `.git` and exact `b12x` container overlay mounts) and all 46 JSON artifacts are documented in **[`docs/benchmarks/README.md`](docs/benchmarks/README.md)**.

## Documentation Index

* **[Benchmark ledger](docs/benchmarks/README.md)**: all 22 primary JSON files, 19 raw vLLM results, Top-2 greedy divergence tables, collective tables, and reproduction commands.
* **[01 — Technology analysis](docs/01-advisor-analysis.md)**: ZML, TokenSpeed, b12x, and embedded vs shared-memory worker topology comparison.
* **[02 — Architecture](docs/02-architecture.md)**: control/data-plane boundary, non-P2P NUMA topology, overlap loop, and cache design.
* **[03 — SHM protocol](docs/03-shm-protocol.md)**: wire layout, cacheline alignment, and lifecycle contracts.
* **[04 — Phase 0](docs/04-phase0-sm120-plan.md)**: `sm_120` operator and NUMA AllReduce measurements.
* **[05 — Phase 1 SHM Engine](docs/05-phase1-shm-engine.md)**: fresh IPC, preparation-cost, speedup, and RAII measurements.
* **[06 — Phase 2 Qwen3.5](docs/06-phase2-qwen35-benchmark.md)**: `Qwen3.5-35B-A3B` (`8 × RTX 5090`) no-APC serving design, statistical comparison, and Top-2 correctness boundaries.
* **[07 — Phase 3 / 3.5 DeepSeek-V4.1-Flash](docs/07-phase3-dsv41-6000d-benchmark.md)**: `DeepSeek-V4.1-Flash` (`8 × RTX 6000D`) `shm_v5` + size-aware collectives, piggyback mixed batching, SM120 SWA/TMA architecture, and no-APC `vLLM 0.31.0` `10/10` comparison.
* **[08 — Phase 3.5 Optimization Plan & Results](docs/08-phase3-dsv41-optimization-plan.md)**: PR `#1` APC metrology alignment & deterministic Top-2 logit audit, piggyback mixed decode + chunked-prefill scheduling, size-aware collectives, and measured before/after impact.
