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
* **Phase 2 / Qwen3.5**: Runtime audit reports **`TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=17`**. Greedy verification covers four prompts, up to 128 tokens, single/batched/chunked execution; the current report contains numerical divergences and **`all_matched=false`**, so no blanket 100% token-parity claim is made. See the exact per-case table in [`docs/benchmarks/README.md`](docs/benchmarks/README.md).
* **KV scope**: [`crates/zs-kv/src/radix.rs`](crates/zs-kv/src/radix.rs) (`PrefixTree` / `KvCacheManager`) is a standalone pure-attention component. It is **not wired into Qwen3.5 serving**; safe prefix reuse requires future `GatedDeltaNet` block-boundary recurrent-state snapshots. Current serving uses `BlockPool` for physical-page RAII only.

## Warm Single-Config `n=3` Serving Benchmark

All six zstep tiers use one resident configuration: `allreduce_blocks=8`, `max_step_tokens=max_decoder_rows=4096`, `max_prefill_chunk=2048`, `num_kv_pages=65536`, `max_seqs=128`. Sixteen warmup prompts run before Round 0; all tiers then repeat three times in round-major interleaved order. Values are `mean ± population std`; full `min..max` and provenance are in the JSON.

Throughput is called a **statistical tie / comparable (`持平`)** when the relative delta is `<3%` or the mean difference is within `2×sqrt(zstep_std²+vllm_std²)`. This is not a “6/6 wins” claim.

| Tier | zstep Output (`tok/s`) | vLLM Output (`tok/s`) | Delta / verdict | Mean TPOT z / v (`ms`) | z Median TTFT / submit-TTFT (`ms`) | v Median TTFT (`ms`) | z P99 ITL (`ms`) | v P99 ITL (`ms`) |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| `in512_c1` | 261.72 ± 19.81 | 204.26 ± 10.53 | +28.14% / **zstep faster** | 3.66 ± 0.27 / 4.64 ± 0.23 | 51.12 ± 10.68 / 2,999.35 ± 243.52 | 69.13 ± 12.82 | 4.09 ± 0.45 | 6.51 ± 1.24 |
| `in512_c16` | 1,938.14 ± 140.42 | 1,893.04 ± 14.01 | +2.38% / **tie** | 7.29 ± 0.52 / 7.34 ± 0.04 | 229.50 ± 27.42 / 2,378.89 ± 209.47 | 298.17 ± 2.29 | 7.27 ± 0.68 | 10.29 ± 0.52 |
| `in512_c128` | 4,917.46 ± 97.01 | 3,975.90 ± 18.06 | +23.68% / **zstep faster** | 21.95 ± 0.51 / 27.29 ± 0.10 | 213.40 ± 1.22 / 3,518.72 ± 19.46 | 1,192.91 ± 15.91 | 225.68 ± 2.50 | 112.48 ± 0.18 |
| `in1024_c1` | 252.42 ± 7.16 | 206.05 ± 0.61 | +22.50% / **zstep faster** | 3.64 ± 0.11 / 4.60 ± 0.02 | 83.95 ± 4.03 / 9,346.27 ± 316.18 | 65.63 ± 1.32 | 3.99 ± 0.11 | 6.66 ± 0.27 |
| `in1024_c32` | 2,152.54 ± 40.35 | 2,678.01 ± 241.68 | -19.62% / **vLLM faster** | 12.86 ± 0.25 / 9.44 ± 0.62 | 223.14 ± 0.30 / 11,226.25 ± 255.84 | 395.95 ± 65.78 | 230.90 ± 0.32 | 66.08 ± 33.07 |
| `in1024_c128` | 3,025.25 ± 24.35 | 3,666.69 ± 484.51 | -17.49% / **tie, high variance** | 35.38 ± 0.36 / 28.83 ± 3.80 | 224.81 ± 1.90 / 14,577.91 ± 165.59 | 1,126.07 ± 183.12 | 238.88 ± 2.50 | 117.22 ± 2.22 |

`ttft_ms` starts when a request receives its concurrency permit, matching vLLM `limited_request_func`. `submit_ttft_ms` starts at workload submission and includes permit queueing. vLLM has no submit-based measurement in this archive, so cross-engine TTFT comparison uses only the permit-based columns.

Warm single-stream Mean TPOT converges to `3.658 ms` (`in512_c1`) and `3.644 ms` (`in1024_c1`), with repeat ranges `3.431..4.031` and `3.552..3.791 ms`. Chunking reduces `in1024_c128` P99 ITL from the old `1,303.57 ms` to `238.88 ± 2.50 ms`, but this remains slower than vLLM and is reported as such.

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

One-Shot decode packets (`<=16 KiB`) cost `26.26..31.34 µs` Graph p50, or `2.10..2.51 ms` for 80 calls—a major part of the roughly `3.5 ms` decode TPOT. NCCL has slightly lower Graph p50 for those three sizes. In the intermediate Two-Shot range (`64 KiB..2 MiB`), NCCL has lower p50 except at `512 KiB`. NUMA First-Touch does not win at `16 MiB` here; it wins at `32 MiB` (`1.24×`) and in the separate `H=5120, 40 MiB` record (`1.45×`).

For `M=48, H=5120`, increasing `blocks=8 -> 16 -> 32` regresses p50 from `57.96 -> 66.35 -> 76.44 µs`. This is consistent with additional cross-NUMA `.sys` flag cacheline traffic and barrier overhead, but no hardware cacheline counter was collected, so that attribution remains a hypothesis.

## Verification and Reproduction

The final gate runs `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace` (**28 tests**), and GPU pytest for SHM layout, Qwen3.5 operators, and TP=8 end-to-end greedy reporting (**8 passed**). The benchmark itself also emits `TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=17`.

The unified benchmark command and all artifact links are in **[`docs/benchmarks/README.md`](docs/benchmarks/README.md)**. Model paths are supplied through `ZSTEP_QWEN35_MODEL_DIR`; containerized reproduction uses `ZSTEP_CONTAINER_IMAGE` with public default `vllm/vllm-openai:v0.33.0`.

## Documentation Index

* **[Benchmark ledger](docs/benchmarks/README.md)**: all 15 primary JSON files, 19 raw vLLM results, exact greedy divergences, full tables, and reproduction commands.
* **[01 — Technology analysis](docs/01-advisor-analysis.md)**: ZML, TokenSpeed, b12x, and Mode A/B comparison.
* **[02 — Architecture](docs/02-architecture.md)**: control/data-plane boundary, non-P2P NUMA topology, overlap loop, and cache design.
* **[03 — SHM protocol](docs/03-shm-protocol.md)**: wire layout, cacheline alignment, and lifecycle contracts.
* **[04 — Phase 0](docs/04-phase0-sm120-plan.md)**: `sm_120` operator and NUMA AllReduce measurements.
* **[05 — Phase 1 Mode B](docs/05-phase1-mode-b-engine.md)**: fresh IPC, preparation-cost, speedup, and RAII measurements.
* **[06 — Phase 2 Qwen3.5](docs/06-phase2-qwen35-benchmark.md)**: serving design, statistical comparison, AllReduce ledger, and correctness boundaries.
