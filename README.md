# zstep

**Zero-magic, step-explicit LLM inference engine (`1` Pure-Rust Master + `8` Explicit `sm_120` GPU Workers over POSIX `/dev/shm` + `eventfd`).**

`zstep` combines:
1. **Pure Rust Master Control Plane**: single-master orchestration over POSIX shared memory + `eventfd`, RAII slot/KV leases, Automatic Prefix Caching (APC, including `Qwen3.5` recurrent GatedDeltaNet checkpoints), online Speculative Decoding (`dspark` / `mtp` / `ngram`), piggyback mixed decode + chunked-prefill scheduling, `in_flight_depth=1` decode pipelining, and zero `pyo3`/GIL or CUDA coupling in the master.
2. **Explicit `sm_120` Step Data Plane**: flat model runners, refresh-in-place GPU buffers, full-model CUDA Graphs (decode + speculative verify + draft), SM120 `b12x` MXFP8/MXFP4 + `sm12x_mega_gate` + `sm120_dsa_indexer` + 1-pass `mHC` with side-stream Sinkhorn overlap, deterministic global Top-2 logit argmax, and runtime zero-recompilation guards.
3. **Adaptive Topology-Aware Collectives**: `shm_v5` per-rank NUMA-local host-mapped SHM AllReduce / AllGather / Argmax on non-P2P PCIe hosts (`LOC`), automatically upgrading to `shm_v5_pxb` CX8 intra-switch CUDA-IPC D2D + cross-switch Host-SHM collectives (`PXB`) when live PCIe Switch P2P bandwidth is verified at startup (`p2p_probe.py`).

## Quick Start

### 1. Build the Rust Master & Server CLI

```bash
cargo build --release
```

### 2. Prepare the SM120 GPU Worker Environment (`8 × RTX 6000D` / `RTX 5090`)

Reference container stack: `vllm/vllm-openai:v0.31.0` (`torch==2.13.0+cu130`, CUDA `13.0`) overlaid with the deterministic SHA256-verified SM120 FlashInfer bundle (`b12x`, `sm12x_mega_gate`, `sm120_dsa_indexer`):

```bash
# Unpack and SHA256-verify worker/third_party/flashinfer_sm120_overlay_ae58183f.tar.xz
OVERLAY_FLAGS="$(./scripts/prepare_sm120_overlay.sh --print-docker-args)"
```

### 3. Launch Online OpenAI-Compatible Serving (`127.0.0.1:43537` by default)

```bash
# Localhost-only default (--host 127.0.0.1); APC enabled by default
./target/release/zstep serve \
  --model-dir /models/Qwen3.5-35B-A3B \
  --served-model-name qwen3.5-35b-a3b \
  --world-size 8 \
  --num-kv-pages 32768

# LAN / Gateway deployment: explicit bind + Bearer auth + isolated loopback admin port
ZSTEP_API_KEY="sk-your-secret-token" ./target/release/zstep serve \
  --host 0.0.0.0 \
  --port 43537 \
  --admin-port 43538 \
  --model-dir /models/DeepSeek-V4.1-Flash \
  --served-model-name deepseek-v4.1-flash \
  --speculative-method dspark \
  --num-speculative-tokens 5
```

### 4. Send a Streaming Chat Completion Request

```bash
curl -N http://127.0.0.1:43537/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b-a3b",
    "messages": [{"role": "user", "content": "Explain POSIX shared-memory inference in 3 bullet points."}],
    "max_tokens": 128,
    "stream": true
  }'
```

## Security & Operational Defaults (`zstep serve`)

- **Loopback-by-default bind**: `--host` defaults to `127.0.0.1` (`--port 43537`). Pass `--host 0.0.0.0` explicitly when exposing to a cluster network or gateway.
- **Optional Bearer authentication (`--api-key` / `ZSTEP_API_KEY`)**: Constant-time `Authorization: Bearer <TOKEN>` check on `/v1/models`, `/v1/chat/completions`, `/admin/drain`, and `/metrics` (`SecretToken` redacts secrets in `Debug` logs and benchmark provenance). Liveness/readiness probes (`/livez`, `/health`, `/ready`) remain unauthenticated for orchestrators.
- **Ops surface isolation**: Without `--api-key`, mutating `POST /admin/drain` strictly rejects non-loopback peers (`403 Forbidden`), while read-only `GET /metrics` stays scrapeable on the bound listener for Prometheus / Nebula Docker & K8s scrapers (pass `--no-allow-remote-metrics` or `ZSTEP_ALLOW_REMOTE_METRICS=0` to restrict unauthenticated `/metrics` to loopback as well). Pass `--admin-port <PORT>` to move `/admin/drain` and `/metrics` onto a dedicated `127.0.0.1:<PORT>` listener and remove them from the public `--host:<port>` router completely (`--api-key` still applies on `--admin-port`).

## Repository Layout

```text
zstep/
├── crates/
│   ├── zs-wire/                      # #[repr(C)] SHM layout, geometry & result ring
│   ├── zs-shm/                       # RAII shm_open + mmap + eventfd transport
│   ├── zs-kv/                        # BlockPool, PrefixTree & Automatic Prefix Caching (APC)
│   ├── zs-tokenizer/                 # HF / DeepSeek-V4.1 / ChatML tokenizer & streaming detokenizer
│   ├── zs-scheduler/                 # WorkerCoordinator, ServingEngine & online ServeLoop
│   └── zs-server/                    # zstep / zs-server CLI: OpenAI HTTP/SSE, auth & benchmarks
├── scripts/
│   └── prepare_sm120_overlay.sh      # SHA256-verified SM120 FlashInfer/b12x overlay unpacker
├── worker/
│   ├── pyproject.toml                # CPU core + [test] / [gpu] optional dependency groups
│   ├── zstep_worker/
│   │   ├── env_config.py             # Centralized typed ZSTEP_* runtime configuration
│   │   ├── p2p_probe.py              # Adaptive PCIe Switch P2P + ACS live bandwidth verifier
│   │   ├── numa_pin.py               # PCIe BDF -> NUMA discovery and CPU pinning
│   │   ├── shm_bridge.py             # Zero-copy mmap + acquire/release atomics
│   │   ├── runtime_loop.py           # Worker loop + zero-recompilation enforcement
│   │   ├── spec_decode.py            # Greedy/varlen verification & N-gram suffix drafter
│   │   ├── models/                   # DeepSeek-V4.1 (+ DSpark) & Qwen3.5-MoE (+ MTP) runners
│   │   └── kernels_sm120/            # shm_v5 / shm_v5_pxb, b12x, 1-pass mHC, GDN/GQA/MoE
│   │       └── csrc/                 # Standalone C++/CUDA translation units (.cpp / .cu)
│   └── tests/                        # CPU parity/AST/env/P2P/spec suites & 8-GPU SM120 harnesses
└── docs/
    ├── benchmark.md                  # Master SM120 benchmark report (non-P2P + adaptive P2P)
    └── 01..10-*.md                   # Architecture, SHM wire protocol, phase reports & RSI lessons
```

## Measured Performance Highlights (`8 × SM120`, `TP=8`)

Full multi-tier tables (`Warm n=3`), hardware topology details (`8 × RTX 5090` 4-NUMA non-P2P and `8 × RTX 6000D` 2-NUMA CX8 non-P2P / `PXB` dual-mode), latency trade-offs (`TTFT` / `TPOT` / `P99 ITL`), and reproduction commands live in **[`docs/benchmark.md`](docs/benchmark.md)**.

| Dimension | Workload & Platform (`TP=8`) | Baseline Reference | `zstep` Measured Result | Speedup / Delta |
| :--- | :--- | :--- | :--- | :--- |
| **PCIe AllReduce (`H=5120`)** | `8 × RTX 6000D` (`M=1..512`) | `NCCL_LOC`: `48.95 µs` (`M=1`); `shm_v5`: `571.54 µs` (`M=512`) | `shm_v5`: **`12.46–14.48 µs`** (`M=1`); `shm_v5_pxb`: **`323.95 µs`** (`M=512`) | **`3.38×–3.93×`** (`M=1`); CX8 P2P **`1.40×–1.76×`** faster |
| **Uncached Serving (`Qwen3.5`)** | `Qwen3.5-35B-A3B` @ `8 × RTX 5090` / `6000D` | `vLLM 0.33.0`: `192.2–3,259.0 tok/s` | Non-P2P **`261.5–5,115.9 tok/s`**; Adaptive P2P **`5,877.1 tok/s`** (`17,631.4 tok/s` total), `c=1` **`380.0 tok/s`** | **`+11.0%..+80.3%`** vs `vLLM` |
| **Uncached Serving (`DeepSeek-V4.1`)** | `DeepSeek-V4.1-Flash` @ `8 × RTX 6000D` | `vLLM 0.31.0` Pure: `83.6–1,126.3 tok/s` | Non-P2P **`124.0–1,807.2 tok/s`**; Adaptive P2P + `FP12+BFP7` + `Cluster-8` `mHC` **`2,437.3 tok/s`** (`8,816.9 tok/s` total), `c=1` **`133.5 tok/s`** | **`+42.5%..+116.4%`** vs `vLLM` Pure |
| **Automatic Prefix Caching (APC)** | `Qwen3.5` / `DeepSeek-V4.1` @ `8 × RTX 6000D` | Cold TTFT: `599.2 ms` / `1,049.6 ms` | Warm TTFT: **`36.9–44.4 ms`** / **`202.7–204.3 ms`** | **`13.49×–16.22×`** / **`5.14×–5.18×`** |
| **Speculative Decoding** | `dspark K=5` (`DSV4.1`) / `mtp K=3` (`Qwen3.5`) / `ngram` | Non-spec `zstep`: `118.6` / `314.3 tok/s` | **`168.9–395.1 tok/s`** (`dspark`), **`583.6–762.5 tok/s`** (`mtp`), **`343.8–674.2 tok/s`** (`ngram`) | **`1.42×–3.35×`** / **`1.87×–2.45×`** / **`2.16×–2.92×`** |

## Runtime Environment Configuration

All `ZSTEP_*` worker runtime environment variables are centrally parsed and range-validated in [`worker/zstep_worker/env_config.py`](worker/zstep_worker/env_config.py) (`WorkerRuntimeConfig.from_env()`), logged once on `rank 0` startup (`zstep_worker_config ...`), and recorded (with secret keys redacted) in `BenchmarkProvenance.env_overrides`. Direct `os.environ` reads of `ZSTEP_*` outside `env_config.py` are blocked in CI by [`worker/tests/test_hotpath_ast_guards.py`](worker/tests/test_hotpath_ast_guards.py).

## Verification & CI Gates

Lightweight CPU CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on every push and pull request:
- **Rust (`ubuntu-latest`)**: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace` (**90+ unit & SHM integration tests**, including Bearer auth, loopback/separate-admin-port enforcement, poisoned-mutex recovery, KV prefix caching, and speculative verification).
- **Python (`ubuntu-latest`, no GPU required)**: `pip install -e "./worker[test]"`, `ruff check worker/`, and `pytest` across SHM wire layout parity, PCIe/ACS P2P topology probing, speculative verification, `WorkerRuntimeConfig` validation, and static AST hot-path guards (`.item()` / `.cpu()` / `.tolist()` / `dist.broadcast` / `torch.cuda.synchronize` / `torch.where(..., out=<scalar>)` / `int32` `index_select`).
- **8-GPU `sm_120` Hardware Suites**: `worker/tests/test_p2p_probe.py` (8-GPU `shm_v5_pxb` + 1-pass `mHC` parity), `worker/tests/phase2/test_qwen35_ops_sm120.py`, `worker/tests/phase2/test_qwen35_e2e_tp8.py`, and Top-2 greedy parity audits (`TRITON_RECOMPILATIONS=0`).

## Documentation Index

* **[Master Benchmark Report (`docs/benchmark.md`)](docs/benchmark.md)**: end-to-end non-P2P & adaptive PCIe Switch P2P (`PXB`) benchmarks, APC, Speculative Decoding (`dspark` / `mtp` / `ngram`), Nebula gateway E2E, and workload selection guide.
* **[01 — Technology analysis](docs/01-advisor-analysis.md)**: ZML, TokenSpeed, `b12x`, and embedded vs shared-memory worker topology comparison.
* **[02 — Architecture](docs/02-architecture.md)**: control/data-plane boundary, NUMA topology, overlap loop, and KV cache design.
* **[03 — SHM protocol](docs/03-shm-protocol.md)**: wire layout, cacheline alignment, and lifecycle contracts.
* **[04 — Phase 0](docs/04-phase0-sm120-plan.md)**: `sm_120` operator and NUMA AllReduce microbenchmarks.
* **[05 — Phase 1 SHM Engine](docs/05-phase1-shm-engine.md)**: fresh IPC, `4.69 µs/step` Rust preparation cost, and `/dev/shm` RAII teardown.
* **[06 — Phase 2 Qwen3.5](docs/06-phase2-qwen35-benchmark.md)**: `Qwen3.5-35B-A3B` (`8 × RTX 5090`) serving design and Top-2 correctness boundaries.
* **[07 — Phase 3 / 3.5 DeepSeek-V4.1-Flash](docs/07-phase3-dsv41-6000d-benchmark.md)**: `DeepSeek-V4.1-Flash` (`8 × RTX 6000D`) `shm_v5` + piggyback mixed batching vs `vLLM 0.31.0`.
* **[08 — Phase 3.5 Optimization Plan & Results](docs/08-phase3-dsv41-optimization-plan.md)**: APC metrology alignment, deterministic Top-2 logit audit, and before/after ablations.
* **[09 — Online OpenAI Serving & Nebula Integration](docs/09-online-serving-nebula-plan.md)**: `zstep serve` HTTP/SSE lifecycle, operational probes, step-boundary RAII cancellation, and Nebula `engine_type = "zstep"` contract.
* **[10 — RSI Evolution Lessons](docs/10-rsi-evolution-lessons.md)**: CUDA Graph / SM120 / Triton / collective invariants enforced by `test_hotpath_ast_guards.py`.
