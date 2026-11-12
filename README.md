# zstep

**Zero-magic, step-explicit LLM inference engine built around Mode B (`1` Pure-Rust Master + `8` Explicit `sm_120` GPU Workers over POSIX `/dev/shm` + `eventfd`).**

`zstep` combines:
1. **Pure Rust Master Control Plane**: Mode B over POSIX shared memory + `eventfd`, RAII slot/KV leases, stateless 3-token Engram lookback windows, piggyback mixed decode + chunked-prefill scheduling, `in_flight_depth=1` pure-decode pipelining, and zero `pyo3`/GIL or CUDA coupling in the master.
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
│   ├── zs-scheduler/                 # ModeBCoordinator + mixed continuous-batching serving
│   └── zs-server/                    # zstep --mode-b synthetic/parity/benchmark binary
├── worker/
│   ├── zstep_worker/
│   │   ├── numa_pin.py               # PCIe BDF -> NUMA discovery and CPU pinning
│   │   ├── shm_bridge.py             # Zero-copy mmap + acquire/release atomics
│   │   ├── runtime_loop.py           # Worker loop + zero-recompilation enforcement
│   │   ├── models/
│   │   │   ├── deepseek_v41.py       # Explicit 40-layer DeepSeek-V4.1-Flash TP=8 runner
│   │   │   └── qwen35_moe.py         # Explicit 40-layer Qwen3.5-35B-A3B TP=8 runner
│   │   └── kernels_sm120/            # shm_v5 AllReduce/AllGather/Argmax, GDN/GQA/MoE
│   └── tests/                        # SHM, Mode B, Phase 0, Phase 2 & Phase 3 harnesses
└── docs/
    ├── 01-advisor-analysis.md
    ├── 02-architecture.md
    ├── 03-shm-protocol.md
    ├── 04-phase0-sm120-plan.md
    ├── 05-phase1-mode-b-engine.md
    ├── 06-phase2-qwen35-benchmark.md
    ├── 07-phase3-dsv41-6000d-benchmark.md
    ├── 08-phase3-dsv41-optimization-plan.md
    └── benchmarks/                   # 22 primary + 19 raw vLLM JSON artifacts
```

## Current Status

* **Phase 3 / 3.5 — `DeepSeek-V4.1-Flash` (`8 × RTX 6000D`, `TP=8`)**:
  * **Size-Aware `shm_v5` + `NCCL_LOC` Collectives** ([`phase0_6000d_ar_shmv5.json`](docs/benchmarks/phase0_6000d_ar_shmv5.json)): across all decode CUDA Graph buckets (`M=1..128`, `H=5120` BF16), `shm_v5 (blocks=8)` beats `NCCL_LOC` by **`1.19×..3.38×`** (`14.48 µs` vs `48.95 µs` at `M=1`; `14.08 µs` / `3.48×` at `blocks=2`), while eager mixed/prefill payloads (`M > 256`) automatically route to multi-SM `NCCL_LOC` (`3,944.79 µs` vs `4,660.95 µs` at `M=4096`).
  * **10 / 10 Tier Sweep vs No-APC `vLLM 0.31.0`**: With `--no-enable-prefix-caching` + isolated per-`(round, tier)` seeds on both engines, piggyback mixed decode + chunked prefill (`--max-mixed-step-tokens 3072`), and size-aware collectives, `zstep` **outperforms `vLLM 0.31.0` across all 10 benchmark tiers (`+3.45%..+19.16%`)** — reaching **`95.24..98.94 tok/s`** single-stream (`Mean TPOT = 9.55..9.87 ms`, **`+13.11%..+14.53%`**) and **`1,372.77 tok/s`** (`+19.16%`) / **`1,002.13 tok/s`** (`+14.03%`) at `c=128` with **`3.69×..3.71×` faster Median TTFT** and **`2.46×..2.47×` lower `P99 ITL`** (`415.33..419.58 ms` vs `1,019.66..1,035.37 ms`).
  * **Deterministic Global Top-2 Greedy Audit**: Runtime emits **`TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=5`** and clean `/dev/shm` teardown; [`phase3_greedy_parity_report.json`](docs/benchmarks/phase3_greedy_parity_report.json) achieves **`128 / 128` (`100.00%`) exact token parity** across single, batched, and `chunk=24` mixed-prefill schedules on `prime_squares_table`, while honestly preserving razor-thin BF16 ULP tie divergences (`top1 - top2 = 0.125..0.375`, `all_matched=false`) on open-ended base-model prompts.
* **Phase 2 — `Qwen3.5-35B-A3B` (`8 × RTX 5090`, `TP=8`)**:
  * **6 / 6 Uncached Throughput Sweep vs No-APC `vLLM 0.33.0`**: With `--no-enable-prefix-caching` and per-`(round, tier)` isolated seeds, `zstep` leads on uncached output throughput across all 6 tiers (**`+11.00%..+54.08%`**), while explicitly reporting the `max_prefill_chunk=2048` tail-latency tradeoff at `c=32/128`.
  * **Greedy Audit**: Runtime emits **`TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=17`**. `chunked` matches `vLLM` **448/448 tokens (100% across all 5 cases)**; remaining `single`/`batched` flips occur strictly at `<= 2` BF16 ULPs (`margin <= 0.25` at logit `~25`), preserved honestly as `all_matched=false` in [`phase2_greedy_parity_report.json`](docs/benchmarks/phase2_greedy_parity_report.json).
* **Phase 1 / Mode B Control Plane**: 200-step 8-GPU measurement records **`4.690 µs/step`** Rust preparation (`0.166%` of GPU step time, `overlap_speedup=1.000269×`).
* **KV scope**: [`crates/zs-kv/src/radix.rs`](crates/zs-kv/src/radix.rs) (`PrefixTree` / `KvCacheManager`) is a standalone pure-attention component and is **not wired into `DeepSeek-V4.1-Flash` or `Qwen3.5-35B-A3B` serving**; both runners use `BlockPool` physical-page RAII only.

## Phase 3.5 Warm Single-Config `n=3` Uncached Serving (`DeepSeek-V4.1-Flash`, `8 × RTX 6000D`)

Unified resident config across all tiers: `allreduce_blocks=8`, `num_kv_pages=2560`, `max_seqs=128`, `max_step_tokens=max_decoder_rows=4096`, `max_prefill_chunk=2048`, `max_mixed_step_tokens=3072`, `max_pages_per_seq=256`, 16 warmup prompts, `n=3` round-major interleaved repeats (`--no-enable-prefix-caching` on `vLLM 0.31.0`, isolated per-`(round, tier)` seeds on both engines). Values are `mean ± population std`; throughput is a **tie** when relative delta is `<3%` or `|delta| <= 2×sqrt(zstep_std²+vllm_std²)`. `ttft_ms` is permit-based (matching `vLLM` `limited_request_func`), whereas `submit_ttft_ms` includes client concurrency-pool queueing.

### Suite 1: Fixed `512 -> 128` (`c=1, 8, 16, 32`)

Raw artifacts: [`zstep`](docs/benchmarks/phase3_zstep_dsv41_tp8_in512_out128.json), [`vLLM 0.31.0`](docs/benchmarks/phase3_vllm_dsv41_tp8_in512_out128.json).

| Tier | zstep Output (`tok/s`) | vLLM Output (`tok/s`) | Delta / verdict | Mean TPOT z / v (`ms`) | z Median TTFT / submit-TTFT (`ms`) | v Median TTFT (`ms`) | z P99 ITL (`ms`) | v P99 ITL (`ms`) |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| `c1` | **96.18 ± 0.57** | 83.97 ± 0.05 | **+14.53% / zstep faster** | **9.55 ± 0.06** / 10.95 ± 0.00 | **117.56 ± 0.19** / 9,471.93 ± 19.16 | 132.60 ± 1.27 | **9.80 ± 0.03** | 11.38 ± 0.13 |
| `c8` | **377.50 ± 2.52** | 351.50 ± 10.04 | **+7.40% / zstep faster** | **17.14 ± 0.14** / 18.25 ± 0.17 | **535.78 ± 0.09** / 3,224.43 ± 35.53 | 594.99 ± 1.19 | **17.67 ± 0.12** | 19.00 ± 0.33 |
| `c16` | **527.00 ± 1.87** | 509.40 ± 3.55 | **+3.45% / zstep faster** | 26.51 ± 0.11 / **23.47 ± 0.23** | **410.34 ± 0.43** / 4,949.74 ± 48.17 | 1,081.25 ± 0.78 | 351.43 ± 0.19 | **25.26 ± 0.15** |
| `c32` | **663.78 ± 3.57** | 635.98 ± 1.49 | **+4.37% / zstep faster** | 43.11 ± 0.26 / **38.50 ± 0.12** | **411.54 ± 0.40** / 8,399.90 ± 102.45 | 1,228.47 ± 14.19 | 410.91 ± 0.25 | **107.80 ± 0.70** |

### Suite 2: Six-Tier Interleaved (`in512` / `in1024` × `output=256`)

Raw artifacts: [`zstep`](docs/benchmarks/phase3_zstep_dsv41_tp8_6tier.json), [`vLLM 0.31.0`](docs/benchmarks/phase3_vllm_dsv41_tp8_6tier.json).

| Tier | zstep Output (`tok/s`) | vLLM Output (`tok/s`) | Delta / verdict | Mean TPOT z / v (`ms`) | z Median TTFT / submit-TTFT (`ms`) | v Median TTFT (`ms`) | z P99 ITL (`ms`) | v P99 ITL (`ms`) |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| `in512_c1` | **98.94 ± 0.33** | 87.24 ± 0.07 | **+13.42% / zstep faster** | **9.68 ± 0.03** / 10.98 ± 0.00 | **118.68 ± 0.84** / 7,889.69 ± 26.37 | 133.63 ± 1.67 | **9.87 ± 0.04** | 11.48 ± 0.14 |
| `in512_c16` | **609.53 ± 7.15** | 573.38 ± 6.84 | **+6.30% / zstep faster** | 24.15 ± 0.31 / 24.07 ± 0.38 | **410.82 ± 0.27** / 7,580.25 ± 54.25 | 1,094.82 ± 18.29 | **23.47 ± 0.25** | 25.12 ± 0.53 |
| `in512_c128` | **1,372.77 ± 2.51** | 1,152.03 ± 6.34 | **+19.16% / zstep faster** | **82.32 ± 0.17** / 96.30 ± 0.60 | **831.11 ± 0.31** / 8,983.78 ± 1.96 | 3,068.72 ± 1.11 | **415.33 ± 0.14** | 1,019.66 ± 0.68 |
| `in1024_c1` | **95.24 ± 0.09** | 84.20 ± 0.02 | **+13.11% / zstep faster** | **9.87 ± 0.01** / 11.18 ± 0.00 | **171.66 ± 0.16** / 24,369.74 ± 41.77 | 189.45 ± 0.73 | **10.07 ± 0.01** | 11.72 ± 0.02 |
| `in1024_c32` | **649.63 ± 1.94** | 614.93 ± 3.52 | **+5.64% / zstep faster** | 44.18 ± 0.17 / **42.06 ± 0.35** | **415.61 ± 0.15** / 37,201.49 ± 204.12 | 1,959.74 ± 61.20 | 415.40 ± 0.11 | **175.89 ± 0.90** |
| `in1024_c128` | **1,002.13 ± 3.22** | 878.86 ± 4.26 | **+14.03% / zstep faster** | **108.63 ± 0.34** / 120.03 ± 0.49 | **837.54 ± 0.10** / 41,618.50 ± 129.12 | 3,108.79 ± 0.44 | **419.58 ± 0.14** | 1,035.37 ± 0.43 |

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

The verification gate passes `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace` (**29 tests**), `ruff check worker/zstep_worker worker/tests/phase2 worker/tests/phase3`, and GPU pytest (`test_shm_layout_parity.py`, `test_qwen35_ops_sm120.py`, `test_qwen35_e2e_tp8.py`), plus end-to-end Top-2 greedy audits (`TRITON_RECOMPILATIONS=0`) and `/dev/shm` RAII checks on both `8 × RTX 5090` and `8 × RTX 6000D`.

Unified reproduction commands and all 41 JSON artifacts are documented in **[`docs/benchmarks/README.md`](docs/benchmarks/README.md)**.

## Documentation Index

* **[Benchmark ledger](docs/benchmarks/README.md)**: all 22 primary JSON files, 19 raw vLLM results, Top-2 greedy divergence tables, collective tables, and reproduction commands.
* **[01 — Technology analysis](docs/01-advisor-analysis.md)**: ZML, TokenSpeed, b12x, and Mode A/B comparison.
* **[02 — Architecture](docs/02-architecture.md)**: control/data-plane boundary, non-P2P NUMA topology, overlap loop, and cache design.
* **[03 — SHM protocol](docs/03-shm-protocol.md)**: wire layout, cacheline alignment, and lifecycle contracts.
* **[04 — Phase 0](docs/04-phase0-sm120-plan.md)**: `sm_120` operator and NUMA AllReduce measurements.
* **[05 — Phase 1 Mode B](docs/05-phase1-mode-b-engine.md)**: fresh IPC, preparation-cost, speedup, and RAII measurements.
* **[06 — Phase 2 Qwen3.5](docs/06-phase2-qwen35-benchmark.md)**: `Qwen3.5-35B-A3B` (`8 × RTX 5090`) no-APC serving design, statistical comparison, and Top-2 correctness boundaries.
* **[07 — Phase 3 / 3.5 DeepSeek-V4.1-Flash](docs/07-phase3-dsv41-6000d-benchmark.md)**: `DeepSeek-V4.1-Flash` (`8 × RTX 6000D`) `shm_v5` + size-aware collectives, piggyback mixed batching, SM120 SWA/TMA architecture, and no-APC `vLLM 0.31.0` `10/10` comparison.
* **[08 — Phase 3.5 Optimization Plan & Results](docs/08-phase3-dsv41-optimization-plan.md)**: PR `#1` APC metrology alignment & deterministic Top-2 logit audit, piggyback mixed decode + chunked-prefill scheduling, size-aware collectives, and measured before/after impact.
