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
│   ├── zs-kv/                     # RAII paged allocator for FlatKV 4-group cache + 128-token SWA bounded replay
│   ├── zs-scheduler/              # Continuous batching, CED layer-20 row narrowing, DSpark 6-token verify slots
│   └── zs-server/                 # Master control loop, OpenAI HTTP API, Tokenizer & Engram 3-gram preprocessor
├── worker/                        # Explicit Step GPU Worker (1 process per RTX 6000D, Rank 0..7)
│   ├── pyproject.toml
│   ├── csrc/                      # Custom PCIe Gen5 P2P Two-Shot AllReduce & eventfd/shm C helpers
│   ├── zstep_worker/
│   │   ├── shm_bridge.py          # Zero-copy mmap reader -> refresh-in-place GPU input buffers
│   │   ├── comm/                  # TP8/EP8 PCIe collective wrappers (CUDA-graph capturable)
│   │   ├── kernels_sm120/         # flashinfer[b12x] MoE/GEMM + Triton dsv41 FlatKV/SelectedAttention ops
│   │   └── models/                # ZML-style explicit single-file DeepSeek-V4.1-Flash + DSpark step runner
│   └── tests/
│       └── phase0/                # Phase 0 standalone operator correctness & throughput benchmarks on sm_120
└── docs/
    ├── 01-advisor-analysis.md     # Background research: ZML vs TokenSpeed vs b12x, RTX 6000D & Mode A/B comparison
    ├── 02-architecture.md         # System architecture: Control/Data split, FlatKV, CED narrowing, DSpark
    ├── 03-shm-protocol.md         # Shared-memory #[repr(C)] wire specification & overlap synchronization
    └── 04-phase0-sm120-plan.md    # Phase 0 execution plan: bringing up DeepSeek-V4.1-Flash ops on 8 × RTX 6000D
```

## Documentation Index

* **[01 — 技术选型与深度调研纪要 (`docs/01-advisor-analysis.md`)](docs/01-advisor-analysis.md)**：对 ZML、TokenSpeed、`b12x` 的源码级对比，对 DeepSeek 顾问方案的纠偏，RTX 6000D (`sm_120`) 硬件特性，以及模式 A（内嵌解释器）与模式 B（单主控 + 多 Worker 共享内存 IPC）的详细对比。
* **[02 — 系统架构设计 (`docs/02-architecture.md`)](docs/02-architecture.md)**：`zstep` 的控制面/数据面边界、`in_flight_depth = 1` 双缓冲重叠循环、FlatKV 四组缓存与有界重放、CED 第 20 层窄化分段 CUDA Graph、Engram 与 DSpark 投机解码架构。
* **[03 — 共享内存 IPC 协议规范 (`docs/03-shm-protocol.md`)](docs/03-shm-protocol.md)**：单主控与 8 个 GPU Worker 之间的 POSIX `shm_open` + `eventfd` 零拷贝内存布局、Cacheline 对齐协议头、异常恢复与生命周期契约。
* **[04 — Phase 0 实施计划 (`docs/04-phase0-sm120-plan.md`)](docs/04-phase0-sm120-plan.md)**：在 `8 × RTX 6000D` 上逐个打通并压测 `DeepSeek-V4.1-Flash` 五大核心算子与 PCIe AllReduce 的分步路线图。
