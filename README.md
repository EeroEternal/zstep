# zstep

**Zero-magic, step-explicit LLM inference engine built around Mode B (`1` Pure-Rust Master + `8` Explicit `sm_120` GPU Workers over POSIX `/dev/shm` + `eventfd`).**

`zstep` combines:
1. **Pure Rust Master Control Plane**: Mode B over POSIX shared memory + `eventfd`, RAII slot/KV leases, `in_flight_depth=1` scheduling, no `pyo3`/GIL or CUDA coupling in the master.
2. **Explicit Step Data Plane**: flat model structs, explicit tensor shapes, refresh-in-place GPU buffers, full-model CUDA Graphs, and runtime zero-recompilation guards.
3. **`sm_120` NUMA-Aware PCIe Kernels**: host-mapped SHM AllReduce for non-P2P systems, Triton `GatedDeltaNet` / `PagedGQA` / `SparseMoE`, and CUDA C++ operator primitives.

> **Hardware scope**: End-to-end serving is verified on `8 × 32 GB sm_120` GPUs (PCIe Gen5, no NVLink, `can_device_access_peer == 0`, NUMA `[0,0,0,1,2,2,2,3]`) with `Qwen/Qwen3.5-35B-A3B` BF16 `TP=8`. Geometry for `84 GB` / `96 GB` cards and `DeepSeek-V4.1-Flash` remains **`unverified`**.

## Repository Layout

```text
zstep/
├── crates/
│   ├── zs-wire/                   # #[repr(C)] SHM layout, geometry & result ring
│   ├── zs-shm/                    # RAII shm_open + mmap + eventfd transport
│   ├── zs-kv/                     # BlockPool and standalone pure-attention Radix KV
│   ├── zs-scheduler/              # ModeBCoordinator + continuous-batching serving
│   └── zs-server/                 # zstep --mode-b synthetic/parity/benchmark binary
├── worker/
│   ├── zstep_worker/
│   │   ├── numa_pin.py            # PCIe BDF -> NUMA discovery and CPU pinning
│   │   ├── shm_bridge.py          # Zero-copy mmap + acquire/release atomics
│   │   ├── runtime_loop.py        # Worker loop + zero-recompilation enforcement
│   │   ├── models/qwen35_moe.py   # Explicit 40-layer TP=8 runner + CUDA Graphs
│   │   └── kernels_sm120/         # SHM AllReduce, GDN/GQA/MoE, MXFP8/MXFP4
│   └── tests/                     # SHM, Mode B, Phase 0 and Phase 2 GPU tests
└── docs/
    ├── 01-advisor-analysis.md
    ├── 02-architecture.md
    ├── 03-shm-protocol.md
    ├── 04-phase0-sm120-plan.md
    ├── 05-phase1-mode-b-engine.md
    ├── 06-phase2-qwen35-benchmark.md
    └── benchmarks/                # 15 primary + 19 raw vLLM JSON artifacts
```

## Current Status

* **Phase 0 / AllReduce**: Verified on the non-P2P 4-NUMA host. `H=5120, 40 MiB` p50 is `4,274.36 µs` for zstep vs `6,206.57 µs` for degraded NCCL SHM (`1.45×`, `max_abs_error=0`). The complete `H=2048` table below also preserves sizes where NCCL is faster.
* **Phase 1 / Mode B**: Fresh 200-step 8-GPU measurement: Rust next-step preparation is **`4.690 µs/step`** (`938 µs` total, averaged over all 200 steps; 199 steps are overlapped), `0.166%` of the `2,832.095 µs` GPU step, with **`overlap_speedup=1.000269×`**. The prior same-protocol run was `5.08 µs/step`, `1,016 µs`, and `1.000281×`. These are physical costs, not a claim of dramatic end-to-end acceleration. RAII cleanup leaves no `/dev/shm` segments.
* **Phase 2 / Qwen3.5**: Runtime audit reports **`TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=17`**. Greedy verification covers **five prompts** (including a 128-token non-looping arithmetic table with 113/113 unique 16-grams), single/batched (`5` concurrent sequences)/chunked (`max_prefill_chunk=24`) execution, and **exact per-step Top-2 logit margin diagnostics** across all 8 ranks. `chunked` matches `vLLM` **448/448 tokens (100% across all 5 cases)**; the remaining `single`/`batched` flips occur strictly at `<= 2` BF16 ULPs (`margin <= 0.25` at logit `~25`, where `1 BF16 ULP = 0.125`, including exact `0.0000` ties in `vLLM`), so **`all_matched=false`** is preserved honestly alongside the measured Top-2 margins in [`docs/benchmarks/README.md`](docs/benchmarks/README.md).
* **KV scope**: [`crates/zs-kv/src/radix.rs`](crates/zs-kv/src/radix.rs) (`PrefixTree` / `KvCacheManager`) is a standalone pure-attention component. It is **not wired into Qwen3.5 serving**; safe prefix reuse requires future `GatedDeltaNet` block-boundary recurrent-state snapshots. Current serving uses `BlockPool` for physical-page RAII only.

## Warm Single-Config `n=3` Uncached Serving Benchmark

Both engines run **without prefix caching** (`zstep` uses `BlockPool` without Radix prefix reuse; `vLLM 0.33.0` is launched with explicit `--no-enable-prefix-caching`) and use **distinct deterministic random seeds per `(round, tier)`** so later rounds never hit cached prefixes from earlier rounds. All six `zstep` tiers use one resident configuration: `allreduce_blocks=8`, `max_step_tokens=max_decoder_rows=4096`, `max_prefill_chunk=2048`, `num_kv_pages=65536`, `max_seqs=128`. Warmup runs before Round 0; all six tiers then repeat three times in round-major interleaved order. Values are `mean ± population std`; full `min..max` and provenance are in the JSON.

Throughput is called a **statistical tie** when the relative delta is `<3%` or the mean difference is within `2×sqrt(zstep_std²+vllm_std²)`. While `zstep` leads on uncached output throughput across all six tiers (`+11.00%..+54.08%`), **this is not an unconditional all-metric sweep**: at `c=32/128`, `zstep`'s `max_prefill_chunk=2048` / `max_step_tokens=4096` trades **`~2×` higher `P99 ITL` (`223.5..242.5 ms` vs `vLLM`'s `109.8..114.4 ms`)** for higher prefill/total throughput, and at `in1024_c32` `zstep` has slower `Mean TPOT` (`12.72 vs 11.29 ms`).

| Tier | zstep Output (`tok/s`) | vLLM Output (`tok/s`) | Delta / throughput verdict | Mean TPOT z / v (`ms`) | z Median TTFT / submit-TTFT (`ms`) | v Median TTFT (`ms`) | z P99 ITL (`ms`) | v P99 ITL (`ms`) |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| `in512_c1` | 261.53 ± 24.43 | 192.24 ± 10.22 | +36.05% / **zstep faster** | 3.68 ± 0.34 / 4.66 ± 0.22 | 50.42 ± 11.10 / 3,016.61 ± 315.79 | 137.34 ± 17.39 | 4.12 ± 0.41 | 6.51 ± 0.93 |
| `in512_c16` | 2,037.92 ± 51.95 | 1,700.98 ± 12.13 | +19.81% / **zstep faster** | 6.87 ± 0.24 / 7.62 ± 0.02 | 209.37 ± 1.82 / 2,252.67 ± 56.82 | 463.87 ± 17.91 | 6.98 ± 0.14 | 10.40 ± 0.28 |
| `in512_c128` | 5,021.65 ± 104.06 | 3,259.04 ± 5.72 | +54.08% / **zstep faster** | 21.46 ± 0.50 / 32.66 ± 0.17 | 210.86 ± 1.69 / 3,484.02 ± 21.72 | 832.91 ± 181.55 | 223.50 ± 2.30 | **114.35 ± 1.68** |
| `in1024_c1` | 268.95 ± 1.47 | 197.31 ± 0.59 | +36.31% / **zstep faster** | 3.42 ± 0.02 / 4.54 ± 0.02 | 79.91 ± 1.33 / 8,659.99 ± 46.85 | 132.48 ± 1.45 | 3.72 ± 0.03 | 6.01 ± 0.10 |
| `in1024_c32` | 2,174.52 ± 13.18 | 1,959.07 ± 29.76 | +11.00% / **zstep faster** | 12.72 ± 0.07 / **11.29 ± 0.09** | 223.18 ± 1.23 / 11,030.49 ± 27.60 | 1,034.01 ± 33.94 | 230.63 ± 1.34 | **109.76 ± 1.33** |
| `in1024_c128` | 2,982.59 ± 54.67 | 2,565.79 ± 18.11 | +16.24% / **zstep faster** | 35.97 ± 0.73 / 41.20 ± 0.24 | 226.59 ± 3.82 / 14,742.53 ± 211.99 | 711.86 ± 103.75 | 242.46 ± 3.61 | **113.56 ± 1.40** |

`ttft_ms` starts when a request receives its concurrency permit, matching vLLM `limited_request_func`. `submit_ttft_ms` starts at workload submission and includes permit queueing. vLLM has no submit-based measurement in this archive, so cross-engine TTFT comparison uses only the permit-based columns.

Warm single-stream Mean TPOT is `3.677 ms` (`in512_c1`, range `3.376..4.153 ms`) and `3.420 ms` (`in1024_c1`, range `3.397..3.433 ms`). Chunking (`max_prefill_chunk=2048`) reduces `in1024_c128` P99 ITL from the old unchunked `1,303.57 ms` to `242.46 ± 3.61 ms`, which remains ~`2.1×` higher than vLLM (`113.56 ± 1.40 ms`)—a direct throughput-vs-tail-latency tradeoff of the 4,096-token prefill step budget.

Raw artifacts: [`zstep`](docs/benchmarks/phase2_zstep_qwen35_35b_a3b_tp8.json), [`vLLM`](docs/benchmarks/phase2_vllm_qwen35_35b_a3b_tp8.json), and [`vLLM rounds`](docs/benchmarks/phase2_vllm_raw/).

## Complete `H=2048, blocks=8` AllReduce Table

BF16, eight GPUs, no P2P; 10 calls per CUDA Graph, 300 warmup and 1,000 timed replays. The 80-call column is inferred from Graph p50, not a separate end-to-end measurement. Source: [`phase2_allreduce_h2048_8gpu.json`](docs/benchmarks/phase2_allreduce_h2048_8gpu.json).

| `M` | Payload | Algorithm | zstep p50/p95/p99 (`µs`) | NCCL SHM p50/p95/p99 (`µs`) | zstep 80 calls (`ms`) |
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

One-Shot decode packets (`<=16 KiB`) cost `26.26..31.34 µs` Graph p50, or `2.10..2.51 ms` for 80 calls—matching degraded NCCL SHM (`26.06..28.43 µs`, `2.08..2.27 ms`). Consequently, `zstep`'s `+36%` single-stream (`c=1`) decode advantage (`3.42..3.68 ms` vs `4.54..4.66 ms` Mean TPOT) does **not** come from faster 4 KiB AllReduce; subtracting the shared `~2.10 ms` communication floor shows `zstep` spends `~1.32..1.58 ms` in non-communication execution vs `vLLM`'s `~2.46..2.58 ms` thanks to single-graph 40-layer capture and fused `sm_120` kernels. Conversely, in the intermediate Two-Shot range (`64 KiB..2 MiB`), NCCL has lower p50 except at `512 KiB`—most notably at **`M=32` (`128 KiB`: `59.76 µs` vs `29.63 µs`, `+2.41 ms` across 80 calls)**, which directly explains why `in1024_c32` is `zstep`'s smallest throughput margin (`+11.00%`) and the only tier where `zstep` has slower `Mean TPOT` (`12.72 vs 11.29 ms`). NUMA First-Touch does not win at `16 MiB` here; it wins at `32 MiB` (`1.24×`) and in the separate `H=5120, 40 MiB` record (`1.45×`).

For `M=48, H=5120`, increasing `blocks=8 -> 16 -> 32` regresses p50 from `57.96 -> 66.35 -> 76.44 µs`. This is consistent with additional cross-NUMA `.sys` flag cacheline traffic and barrier overhead, but no hardware cacheline counter was collected, so that attribution remains a hypothesis.

## Verification and Reproduction

The final gate runs `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace` (**28 tests**), and GPU pytest across the three test suites (`test_shm_layout_parity.py`: 1, `test_qwen35_ops_sm120.py`: 7, `test_qwen35_e2e_tp8.py`: 1 — **9 passed total**). The benchmark itself also emits `TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=17`.

The unified benchmark command and all artifact links are in **[`docs/benchmarks/README.md`](docs/benchmarks/README.md)**. Model paths are supplied through `ZSTEP_QWEN35_MODEL_DIR`; containerized reproduction uses `ZSTEP_CONTAINER_IMAGE` with public default `vllm/vllm-openai:v0.33.0`.

## Documentation Index

* **[Benchmark ledger](docs/benchmarks/README.md)**: all 15 primary JSON files, 19 raw vLLM results, exact greedy divergences, full tables, and reproduction commands.
* **[01 — Technology analysis](docs/01-advisor-analysis.md)**: ZML, TokenSpeed, b12x, and Mode A/B comparison.
* **[02 — Architecture](docs/02-architecture.md)**: control/data-plane boundary, non-P2P NUMA topology, overlap loop, and cache design.
* **[03 — SHM protocol](docs/03-shm-protocol.md)**: wire layout, cacheline alignment, and lifecycle contracts.
* **[04 — Phase 0](docs/04-phase0-sm120-plan.md)**: `sm_120` operator and NUMA AllReduce measurements.
* **[05 — Phase 1 Mode B](docs/05-phase1-mode-b-engine.md)**: fresh IPC, preparation-cost, speedup, and RAII measurements.
* **[06 — Phase 2 Qwen3.5](docs/06-phase2-qwen35-benchmark.md)**: serving design, statistical comparison, AllReduce ledger, and correctness boundaries.
