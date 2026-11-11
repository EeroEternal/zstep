# zstep

**Zero-magic, step-explicit LLM inference engine tailored for `8 × NVIDIA RTX 6000D` (`sm_120`, 84 GB GDDR7, PCIe Gen5) running `DeepSeek-V4.1-Flash`.**

`zstep` combines three engineering strengths into a compact, single-maintainer-friendly codebase:
1. **Pure Rust Master Control Plane** (inspired by TokenSpeed's control/data plane separation, using **Mode B: 1 Master + 8 Workers over POSIX Shared Memory + `eventfd`** with zero `pyo3`/GIL or CUDA coupling).
2. **ZML-Style Explicit Step Data Plane** (flat, zero-inheritance model structs with explicit tensor shapes, refresh-in-place GPU buffers, and split CUDA Graphs).
3. **`sm_120`-Native Kernel Stack** (`flashinfer[b12x]` block-scaled NVFP4/MXFP4/MXFP8 kernels + Triton FlatKV/CSA sparse attention + custom PCIe Gen5 P2P Two-Shot AllReduce).

---

## Repository Layout

```text
zstep/
├── Cargo.toml                     # Rust workspace (zero Python/CUDA dependencies)
├── crates/
│   ├── zs-wire/                   # #[repr(C)] POSIX shared-memory layout, double-buffered StepPlan & ResultRing
│   ├── zs-shm/                    # RAII POSIX shm_open + mmap + eventfd transport & typed atomic views
│   ├── zs-kv/                     # RAII paged allocator for FlatKV 4-group cache + 128-token SWA bounded replay
│   ├── zs-scheduler/              # Continuous batching, CED narrowing, ModeBCoordinator overlap loop
│   └── zs-server/                 # Master binary (`zstep --mode-b`), OpenAI HTTP API & Engram preprocessor
├── worker/                        # Explicit Step GPU Worker (1 process per sm_120 GPU, Rank 0..7)
│   ├── pyproject.toml
│   ├── zstep_worker/
│   │   ├── shm_bridge.py          # Zero-copy mmap + libatomic Acquire/Release reader/writer
│   │   ├── runtime_loop.py        # Mode B worker loop + sm_120 forward micro-step
│   │   └── kernels_sm120/         # Triton RMSNorm+MXFP8, dsv41 router, HostMappedShmAllReduce + b12x ops
│   └── tests/
│       ├── test_shm_layout_parity.py
│       ├── test_mode_b_e2e.py     # 1-GPU & 8-GPU Rust <-> Python Mode B end-to-end & RAII cleanup suite
│       └── phase0/                # Phase 0 standalone operator & 8-GPU AllReduce suites (test_00..test_05)
└── docs/
    ├── 01-advisor-analysis.md     # Background research: ZML vs TokenSpeed vs b12x, RTX 6000D & Mode A/B comparison
    ├── 02-architecture.md         # System architecture: Control/Data split, FlatKV, CED narrowing, TP=8 MoE
    ├── 03-shm-protocol.md         # Shared-memory #[repr(C)] wire specification & overlap synchronization
    ├── 04-phase0-sm120-plan.md    # Phase 0 verified sm_120 operator benchmarks & 40-layer latency budget
    ├── 05-phase1-mode-b-engine.md # Phase 1 Mode B Rust Master <-> 8x sm_120 Worker overlap engine report
    └── benchmarks/                # Verified sm_120 JSON benchmark artifacts (step_00..step_05b, phase1_mode_b)
```

## Current Status

* **[x] Phase 0 (`sm_120` Operator Stack & 8-GPU PCIe AllReduce)**: Completed & verified across 8 `sm_120` GPUs (`TP=8` MXFP4 MoE `70.2 µs` at `M=6`, `deepseek_v41` Sparse MLA `13.5 µs` at 32K context, `mHC` `11.4 µs`, `Engram` VRAM `4.4 µs` / Host UVA `6.1 µs`, 8-GPU single-launch 4-slot ring `HostMappedShmAllReduce` `max_abs_error = 0.0`).
* **[x] Phase 1 (Mode B POSIX SHM + `eventfd` 8-GPU Overlap Engine)**: Completed & verified (`zstep --mode-b --world-size 8 --steps 200`: `19.3 µs` publish/wake across 8 workers, `100.0%` schedule overlap hiding efficiency, zero JIT recompilations, zero `/dev/shm` leaks).
* **[ ] Phase 2 (40-Layer Explicit `DeepSeek-V4.1-Flash` Runner + `zs-kv` Radix Prefix Tree + CED Split CUDA Graphs)**: Next up.

## Documentation Index

* **[01 — 技术选型与深度调研纪要 (`docs/01-advisor-analysis.md`)](docs/01-advisor-analysis.md)**：对 ZML、TokenSpeed、`b12x` 的源码级对比，对 DeepSeek 顾问方案的纠偏，RTX 6000D (`sm_120`) 硬件特性，以及模式 A（内嵌解释器）与模式 B（单主控 + 多 Worker 共享内存 IPC）的详细对比。
* **[02 — 系统架构设计 (`docs/02-architecture.md`)](docs/02-architecture.md)**：`zstep` 的控制面/数据面边界、`in_flight_depth = 1` 双缓冲重叠循环、FlatKV 四组缓存与有界重放、CED 第 20 层窄化分段 CUDA Graph、Engram 与 `TP=8` MoE 实测决策。
* **[03 — 共享内存 IPC 协议规范 (`docs/03-shm-protocol.md`)](docs/03-shm-protocol.md)**：单主控与 8 个 GPU Worker 之间的 POSIX `shm_open` + `eventfd` 零拷贝内存布局、Cacheline 对齐协议头、异常恢复与生命周期契约。
* **[04 — Phase 0 实测报告与 40 层延迟预算 (`docs/04-phase0-sm120-plan.md`)](docs/04-phase0-sm120-plan.md)**：在 8 卡 `sm_120` 上打通并压测 `DeepSeek-V4.1-Flash` 全部核心算子与 8 卡 AllReduce 的实测数据与单步预算表。
* **[05 — Phase 1 Mode B 引擎实测报告 (`docs/05-phase1-mode-b-engine.md`)](docs/05-phase1-mode-b-engine.md)**：纯 Rust `zs-shm` / `ModeBCoordinator` 驱动 8 个 `sm_120` GPU Worker 执行 200 步双缓冲重叠推理的设计与实测指标。
