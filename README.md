# zstep

**Zero-magic, step-explicit LLM inference engine built around Mode B (`1` Pure-Rust Master + `8` Explicit `sm_120` GPU Workers over POSIX `/dev/shm` + `eventfd`).**

`zstep` combines three engineering strengths into a compact, single-maintainer-friendly codebase:
1. **Pure Rust Master Control Plane** (inspired by TokenSpeed's control/data plane separation, using **Mode B: 1 Master + 8 Workers over POSIX Shared Memory + `eventfd`** with RAII slot/KV/prefix leases, `in_flight_depth = 1` overlapped scheduling, and zero `pyo3`/GIL or CUDA coupling).
2. **ZML-Style Explicit Step Data Plane** (flat, zero-inheritance model structs with explicit tensor shapes, refresh-in-place GPU buffers, and full-model CUDA Graphs with zero runtime JIT recompilations).
3. **`sm_120`-Native & NUMA-Aware PCIe Kernel Stack** (NUMA-pinned 4-slot lock-free `HostMappedShmAllReduce` for non-P2P PCIe topologies, Triton `GatedDeltaNet` + `PagedGQA` + 256-expert `SparseMoE` kernels, plus `flashinfer[b12x]` NVFP4/MXFP4/MXFP8 operator primitives).

> **Hardware Scope Note**: End-to-end serving and benchmarks are verified on `8 × 32 GB sm_120` GPUs (PCIe Gen5, no NVLink, `can_device_access_peer == 0` across all 8×8 pairs, 4 asymmetric NUMA nodes `[0,0,0,1,2,2,2,3]`) running **`Qwen/Qwen3.5-35B-A3B` (`67 GB` BF16, `TP=8`)**. Geometry calculations for `84 GB` / `96 GB` cards targeting `DeepSeek-V4.1-Flash` (`~510 GB` weights) are dynamically derived and marked **`unverified`** until 84 GB / 96 GB hardware is available.

---

## Repository Layout

```text
zstep/
├── Cargo.toml                     # Rust workspace (zero Python/CUDA dependencies)
├── crates/
│   ├── zs-wire/                   # #[repr(C)] POSIX shared-memory layout, VramGeometrySpec & ResultRing
│   ├── zs-shm/                    # RAII POSIX shm_open + mmap + eventfd transport & typed atomic views
│   ├── zs-kv/                     # RAII BlockPool, Radix PrefixTree (RadixLease) & bounded SWA replay
│   ├── zs-scheduler/              # ModeBCoordinator + ModeBServingEngine continuous batching overlap loop
│   └── zs-server/                 # Master binary (`zstep --mode-b`: synthetic, greedy parity & --bench-tiers)
├── worker/                        # Explicit Step GPU Worker (1 NUMA-pinned process per sm_120 GPU, Rank 0..7)
│   ├── pyproject.toml
│   ├── zstep_worker/
│   │   ├── numa_pin.py            # PCIe BDF -> sysfs NUMA discovery & CPU affinity pinning
│   │   ├── shm_bridge.py          # Zero-copy mmap + libatomic Acquire/Release reader/writer
│   │   ├── runtime_loop.py        # Mode B worker loop + zero-recompilation enforcement
│   │   ├── models/
│   │   │   └── qwen35_moe.py      # Explicit 40-layer Qwen3.5-35B-A3B TP=8 runner + CUDA Graph buckets
│   │   └── kernels_sm120/         # HostMappedShmAllReduce, Qwen3.5 GDN/GQA/MoE kernels, MXFP8/MXFP4 ops
│   └── tests/
│       ├── test_shm_layout_parity.py
│       ├── test_mode_b_e2e.py     # 1-GPU & 8-GPU Rust <-> Python Mode B end-to-end & RAII cleanup suite
│       ├── phase0/                # Phase 0 standalone operator & 8-GPU AllReduce suites (test_00..test_05)
│       └── phase2/                # Phase 2 Qwen3.5 sm_120 op parity (>2^31 int64 KV) & 8-GPU greedy parity
└── docs/
    ├── 01-advisor-analysis.md     # Background research: ZML vs TokenSpeed vs b12x & Mode A/B comparison
    ├── 02-architecture.md         # System architecture, 4-NUMA non-P2P PCIe topology & FlatKV design
    ├── 03-shm-protocol.md         # Shared-memory #[repr(C)] wire specification & overlap synchronization
    ├── 04-phase0-sm120-plan.md    # Phase 0/0.5b verified sm_120 operator & NUMA-pinned AllReduce benchmarks
    ├── 05-phase1-mode-b-engine.md # Phase 1 Mode B Rust Master <-> 8x sm_120 Worker overlap engine report
    ├── 06-phase2-qwen35-benchmark.md # Phase 2 Qwen3.5-35B-A3B TP=8 head-to-head vs vLLM 0.33.0 report
    └── benchmarks/                # Verified JSON benchmark artifacts (step_00..step_05b, phase1, phase2)
```

## Current Status

* **[x] Phase 0 & 0.5b (`sm_120` Operator Stack & NUMA-Pinned 8-GPU PCIe AllReduce)**: Verified across 8 `sm_120` GPUs. On non-P2P 4-NUMA PCIe, `HostMappedShmAllReduce` achieves `M=4096` (`40.96 MiB`) `p50 = 4,274 µs` / `p99 = 4,293 µs` vs degraded `NCCL SHM` `p50 = 6,207 µs` / `p99 = 6,243 µs` (`1.45×` faster, `max_abs_error = 0.0`).
* **[x] Phase 1 (Mode B POSIX SHM + `eventfd` 8-GPU Overlap Engine)**: Verified (`19.3 µs` publish/wake across 8 workers, `100.0%` schedule overlap hiding efficiency, zero `/dev/shm` leaks).
* **[x] Phase 2 (End-to-End `Qwen/Qwen3.5-35B-A3B` `TP=8` BF16 Serving & `vLLM 0.33.0` Head-to-Head)**:
  * **100% Greedy Token Parity** with `vLLM 0.33.0` (`test_qwen35_e2e_tp8.py`) and zero runtime Triton JIT recompilations across all 15 kernels.
  * **Outperforms `vLLM 0.33.0` across all 6 concurrency tiers** on the same `8 × 32 GB sm_120` non-P2P 4-NUMA server:
    * `in=512, out=256`: `c=1` **`225.04` vs `184.72 tok/s` (`+21.8%`)** | `c=16` **`1,768.73` vs `1,671.63 tok/s` (`+5.8%`)** | `c=128` **`4,540.92` (`4,638.96` @ 4K step) vs `3,271.56 tok/s` (`+38.8%` / `+41.8%`)**.
    * `in=1024, out=256`: `c=1` **`261.72` vs `181.96 tok/s` (`+43.8%`)** | `c=32` **`2,165.22` vs `1,834.83 tok/s` (`+18.0%`)** | `c=128` **`3,024.65` vs `2,939.12 tok/s` (`+2.9%`)**.

## Benchmarks (`8 × RTX 5090 sm_120`, Non-P2P 4-NUMA PCIe)

Full raw JSON artifacts (`12` benchmark files across Phase 0, Phase 1, and Phase 2), ablation history, and reproduction commands are indexed in **[`docs/benchmarks/README.md`](docs/benchmarks/README.md)**.

### End-to-End Serving: `Qwen/Qwen3.5-35B-A3B` (`TP=8`, `bfloat16`) vs `vLLM 0.33.0`

| Profile (`in / out`) | Concurrency | `vLLM 0.33.0` (`tok/s`) | `zstep Mode B` (`tok/s`) | Speedup | `vLLM` Median TTFT / Mean TPOT | `zstep` Median TTFT / Mean TPOT |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `512 / 256` | `c = 1` | `184.72` | **`225.04`** | **`+21.8%`** | `74.51 ms` / `5.09 ms` | **`66.59 ms` / `4.09 ms`** (`p99 4.11 ms`) |
| `512 / 256` | `c = 16` | `1,671.63` | **`1,768.73`** | **`+5.8%`** | `392.38 ms` / `8.02 ms` | `465.85 ms` (`290.93 ms` @ 4K) / **`7.25 ms`** |
| `512 / 256` | `c = 128` | `3,271.56` | **`4,540.92`** (`4,638.96` @ 4K) | **`+38.8%` (`+41.8%`)** | `1,240.50 ms` / `33.25 ms` | `1,307.87 ms` (`822.28 ms` @ 4K) / **`22.38 ms`** |
| `1024 / 256` | `c = 1` | `181.96` | **`261.72`** | **`+43.8%`** | `143.35 ms` / `4.86 ms` | **`82.73 ms` / `3.51 ms`** (`p99 3.61 ms`) |
| `1024 / 256` | `c = 32` | `1,834.83` | **`2,165.22`** | **`+18.0%`** | `1,091.90 ms` / `11.92 ms` | **`870.08 ms` / `9.81 ms`** |
| `1024 / 256` | `c = 128` | `2,939.12` | **`3,024.65`** | **`+2.9%`** | `1,355.88 ms` / `33.77 ms` | **`1,295.93 ms` / `33.19 ms`** |

### 8-GPU Non-P2P 4-NUMA PCIe AllReduce (`H = 5120`, `bfloat16`, CUDA Graph Replay)

| Rows `M` (`Bytes`) | `zstep HostMappedShmAllReduce` (`p50 / p99`) | Degraded `NCCL SHM` (`p50 / p99`) | `zstep max_abs_error` |
| :---: | :---: | :---: | :---: |
| `6` (`60 KiB`) | `65.16 µs` / `69.84 µs` | `52.42 µs` / `56.70 µs` | **`0.0`** |
| `48` (`480 KiB`) | `150.19 µs` / **`153.04 µs`** | `106.80 µs` / `155.33 µs` | **`0.0`** |
| `192` (`1.88 MiB`) | `187.49 µs` / `191.47 µs` | `151.01 µs` / `153.15 µs` | **`0.0`** |
| `4096` (`40.0 MiB`) | **`4,274.36 µs` / `4,292.75 µs` (`1.45×`)** | `6,206.57 µs` / `6,243.15 µs` | **`0.0`** |

## Documentation Index

* **[Benchmarks Ledger (`docs/benchmarks/README.md`)](docs/benchmarks/README.md)**：全部 12 份 JSON 基准文件（Phase 0 ~ Phase 2 + 四阶段消融）汇总表与一键复现命令。
* **[01 — 技术选型与深度调研纪要 (`docs/01-advisor-analysis.md`)](docs/01-advisor-analysis.md)**：对 ZML、TokenSpeed、`b12x` 的源码级对比，以及模式 A（内嵌解释器）与模式 B（单主控 + 多 Worker 共享内存 IPC）的详细对比。
* **[02 — 系统架构设计 (`docs/02-architecture.md`)](docs/02-architecture.md)**：`zstep` 的控制面/数据面边界、4-NUMA 非对称无 P2P PCIe 拓扑约束、`in_flight_depth = 1` 双缓冲重叠循环与缓存分层设计。
* **[03 — 共享内存 IPC 协议规范 (`docs/03-shm-protocol.md`)](docs/03-shm-protocol.md)**：单主控与 8 个 GPU Worker 之间的 POSIX `shm_open` + `eventfd` 零拷贝内存布局、Cacheline 对齐协议头与生命周期契约。
* **[04 — Phase 0/0.5b 实测报告与延迟预算 (`docs/04-phase0-sm120-plan.md`)](docs/04-phase0-sm120-plan.md)**：在 8 卡 `sm_120` 上完成核心算子验证及 NUMA First-Touch 优化后的 `p50/p95/p99` AllReduce 实测数据。
* **[05 — Phase 1 Mode B 引擎实测报告 (`docs/05-phase1-mode-b-engine.md`)](docs/05-phase1-mode-b-engine.md)**：纯 Rust `zs-shm` / `ModeBCoordinator` 驱动 8 个 `sm_120` GPU Worker 执行双缓冲重叠推理的设计与实测指标。
* **[06 — Phase 2 `Qwen3.5-35B-A3B` 端到端推理与同机 `vLLM 0.33.0` 对标报告 (`docs/06-phase2-qwen35-benchmark.md`)](docs/06-phase2-qwen35-benchmark.md)**：`zs-kv` Radix 前缀树、`ModeBServingEngine` 连续批处理、40 层混合架构显式执行器设计与 6 档并发同机正面对标数据。
