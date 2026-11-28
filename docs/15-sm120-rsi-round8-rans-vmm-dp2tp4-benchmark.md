# Phase 3.15: SM120 Multi-Plane GPU rANS Lossless KV Codec, CUDA 13 VMM Fixed-VA Peer Arena & Dual-Island `DP=2 × TP=4` (`LOSSLESS` vs `CALIBRATED_LOSSY` `NVFP4`) Benchmark

- **Date**: `2026-11-28`
- **Hardware**: `8 × NVIDIA RTX 6000D` (`SM120`, `156 SMs`, `99 KB` dynamic SMEM/SM, `112 MiB` L2, `85,651 MiB` GDDR7/GPU, `CUDA 13.0`, Dual-Socket Xeon `4+4` PCIe Gen5 Switch topology: `GPU 0..3` on NUMA 0, `GPU 4..7` on NUMA 1)
- **Models Evaluated**:
  - `PrecisionTrack.LOSSLESS`: `deepseek-ai/DeepSeek-V4.1-Flash` (`475.25 GiB`, native `MXFP4` experts `268.94 GiB` + BF16 Engram `189.12 GiB`, `kv_cache_dtype=fp8_ds_mla`)
  - `PrecisionTrack.CALIBRATED_LOSSY` (strictly isolated): `LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4` (`399.52 GiB`, ModelOpt `NVFP4` experts `284.76 GiB` + `MXFP4` Engram `97.28 GiB`)
- **Incumbent Baseline**: `main@e9f2b63` (Round 7 `Lockstep-Safe Cold-Prefill + PXB-4K`, local git `a3f7cbddb0b7`)
- **Raw Benchmark Artifacts**:
  - [`docs/benchmarks/round8_stage1_rans_vmm_nvfp4_scout.json`](benchmarks/round8_stage1_rans_vmm_nvfp4_scout.json)
  - [`docs/benchmarks/round8_multi_regime_abba_results.json`](benchmarks/round8_multi_regime_abba_results.json)
  - [`docs/benchmarks/round8_lossy_nvfp4_dp2tp4_results.json`](benchmarks/round8_lossy_nvfp4_dp2tp4_results.json)

---

## 1. Stage 1: 8-GPU Hardware Scout & Capability Boundaries (`AGENTS.md`)

1. **CUDA 13 Virtual Memory Management Fixed-VA Peer Arena (`P6` — Landed)**:
   - `cuMemCreate` (POSIX FD export/import) + `cuMemAddressReserve` (`0x0800_0000_0000 + rank * (1 << 40)`, `2 MiB` granularity) + `cuMemMap` + `cuMemSetAccess` succeeded (`rc = 0`) across all 8 GPUs (`PXB`, `NODE`, and cross-socket `SYS`).
   - 32 MiB peer read achieved **`56.05 GB/s`** over VMM fixed-VA mappings vs **`35.10 GB/s`** over legacy `cudaIpcOpenMemHandle` (**`+59.7%`**).
2. **SM120 Multi-Plane GPU rANS Lossless KV-Cache Codec (`P4` — Landed Opt-In Capacity Tier)**:
   - Splitting `DSv41-584` (`74,752 B/page`) and `Latent576` (`73,728 B/page`) KV pages into homogeneous byte planes (`fp8_latent`: `1.2087×`, `fp16_scales`: `1.3651×`, `bf16_rope_exp`: `2.9049×`, `bf16_rope_mant`: `1.0000×`) achieves **`1.2361×` (`+23.61%` DRAM KV capacity)** on `DSv41-584` and **`1.2344×`** on `Latent576` with **`0 ULP` (`0` byte mismatches)**, **`47.50 GB/s`** SM120 32-way warp rANS GPU encode, and **`81.57 GB/s`** GPU decode.
   - Because uncongested PCIe Gen5 zero-SM Copy-Engine DMA (`cuMemcpyBatchAsync_v2`) completes 256 pages in **`334.22 µs`** vs **`1,228.99 µs`** for `rANS encode + DMA`, `HostDramKvTier` defaults to `codec="none"` (`ZSTEP_KV_DRAM_CODEC=none`) for zero-latency uncongested offload and provides `codec="rans_gpu"` / `codec="lossless"` (with IEEE 802.3 CRC32 verification) for capacity-bound DRAM tiers.
3. **`b12x` `NVFP4` vs `MXFP4` on `TP=4` (`I=576`) & `w4a8_nvfp4` Capability Gap (`P7`)**:
   - On isolated `TP=4` (`I=576`, zero padding), `b12x` `nvfp4_w4a4` outperforms `mxfp4_w4a8` by **`1.345×` at `M=4096` (`1,989.88 vs 2,675.88 µs`)** and **`1.063×` at `M=64` (`1,566.77 vs 1,665.19 µs`)**, but is **`2.64×–6.73×` slower at `M=1..8`** without `a16_max_tokens=16`.
   - **`w4a8_nvfp4` Capability Gap**: FlashInfer `0.7.6` `b12x` fails with `CUDA_ERROR_ILLEGAL_ADDRESS (700)` during `m64` dynamic grouped kernel preparation for `w4a8_nvfp4` on SM120; `ZSTEP_NVFP4_ACT_MODE=a8` fail-stops explicitly rather than silently corrupting memory.

---

## 2. Landed Round 8 Components in `zstep`

1. **Lossless Multi-Plane GPU/CPU rANS KV Codec + CRC32 (`worker/zstep_worker/kernels_sm120/kv_rans_codec.py`, `csrc/kv_rans_codec.{cpp,cu}`, `kv_dram_tier.py`, `env_config.py`)**:
   - 32-way warp-parallel SM120 rANS encoder/decoder with `DSv41-584` & `Latent576` plane split/unsplit kernels and per-block IEEE 802.3 CRC32 integrity verification (`ZSTEP_KV_DRAM_CODEC=none|rans_gpu|lossless`).
2. **CUDA 13 VMM Fixed-VA Peer Arena (`worker/zstep_worker/cuda_driver.py`, `models/deepseek_v41.py`)**:
   - `VmmAllocationHandle`, `vmm_allocation_granularity`, `vmm_alloc_and_map_fixed_va`, and `vmm_free` (`ZSTEP_ENABLE_VMM_PEER_ARENA=1`) with deterministic destructor cleanup.
3. **Single-Socket `TP=4` (`world_size=4`) + Isolated `ModelOpt NVFP4` / `MXFP4 Engram` Support (`crates/zs-server/src/serve/cli.rs`, `models/deepseek_v41.py`, `kernels_sm120/b12x_dsv41.py`)**:
   - Unlocks `--world-size 4` (`GPU 0..3` on NUMA 0, `GPU 4..7` on NUMA 1) for `DeepSeek-V4.1-Flash`, `ZstepModelOptNvFp4B12xMoEMethod` (`a16_max_tokens=16` hybrid `W4A16`/`W4A4`), and `_engram_mxfp4_lookup_kernel` Triton E2M1 nibble dequantization.

---

## 3. Full 8-GPU Verification Summary (Tracks A1, A2 & Track B)

All runs passed `RedTeam` (`spec_pins_verified=true`), `DispatchVerifier` (`8,640` pairs, `0` violations), 9/9 Z3 `FormalVerifier` clauses (`CLOSED`), `cargo fmt --all --check`, `91 passed` worker unit tests, and `triton_recompilations = 0`:

| Track & Precision Regime | Tier / Metric | `LOSSLESS TP=8` Baseline (`e9f2b63`) | Round 8 Candidate | Measured Delta | Verdict |
|---|---|---:|---:|---:|---|
| **Track A2 (`LOSSLESS TP=8 + VmmPeerArena`)** | `in512_c16` / `in512_c64`<br>`pf8k_c1` / `pf8k_c4` / `pf16k_c2` | `1,099.48` / `1,887.38 tok/s`<br>`29.43` / `33.37` / `15.78 tok/s` | **`1,109.22`** / **`1,903.09 tok/s`** (`TPOT 11.20 ms`)<br>**`29.44`** / **`33.36`** / **`15.78 tok/s`** | **`+0.34%` geomean** (`+0.89%` / `+0.83%`) | **`ACCEPTED` (`100%` Greedy Parity)** |
| **Track A2 (`LOSSLESS` GPU rANS DRAM KV Tier)** | 32-Block Live Roundtrip (`2.39 MiB → 2.02 MiB`) | `1.0000×` (uncompressed DMA) | **`1.1834×` live** (**`1.2361×`** 4-plane), **`0 ULP`**, `32/32` CRC32 | **`+18.3%..+23.6%` DRAM density** | **`PASS` (`LosslessKvCodecGateSpec`)** |
| **Track A1 (`LOSSLESS DP=2 × TP=4`, 3-Round Cold `AB/BA`)** | `pf16k_c2` (`16K` Concurrent Cold Prefill)<br>`pf8k_c4`<br>`in512_c16` / `in512_c64` | `15.78 out tok/s` (`TTFT 1,938.59 ms`)<br>`33.37 out tok/s` (`TTFT 1,819.80 ms`)<br>`1,099.48` / `1,887.38 out tok/s` | **`16.42 out tok/s`** (**`TTFT 1,851.14 ms`**, **`p99_ITL 7.39 ms`**)<br>`32.74 out tok/s` (`TTFT 1,839.94 ms`)<br>`1,010.94` / `1,701.56 out tok/s` | **`+4.06%` (`-4.51%` `TTFT`)**<br>`-1.89%`<br>`-8.05%` / `-9.85%` (`geomean -4.09%`) | **Piecewise Win on `>= 16K, C >= 2` Prefill Island; `TP=8` Retained for Decode** |
| **Track B (`CALIBRATED_LOSSY NVFP4 DP=2 × TP=4`)** | `in512_c32_v` / `pf8k_c4` `TTFT`<br>Decode `c16` / `c32` (`84,311 MiB` VRAM) | `TTFT 423.60` / `1,820.89 ms`<br>`1,103.88` / `1,448.66 tok/s` | **`TTFT 347.41`** / **`1,768.28 ms`** (`prime_squares 100%`)<br>`555.46` / `908.23 tok/s` (`KL 0.0388`, `EM 22.45%`) | **`-17.99%` / `-2.89%` `TTFT`**<br>`-49.68%` / `-37.31%` decode (`geomean -32.43%`) | **`REJECTED` (`KL 0.0388 > 0.02`, `W4A16` Decode Fallback, `84.3 GiB` VRAM)** |
