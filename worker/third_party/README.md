# Third-Party SM120 FlashInfer Source Overlay (`ae58183f`)

`flashinfer_sm120_overlay_ae58183f.tar.xz` bundles the pure-source Apache-2.0 SM120 experimental operator subset from `FlashInfer` commit `ae58183fff34c545b3e3020d81d591f2a79658fc` required to reproduce Phase 3.8+ `DeepSeek-V4.1-Flash` `TP=8` results (`b12x` Block32 FP8 + MXFP4 MoE, `sm12x_mega_gate` for `M >= 32`, and `sm120_dsa_indexer` paged decode) on `vllm/vllm-openai:v0.31.0` without external network fetches or unreachable fork refs:

- **Archive**: `flashinfer_sm120_overlay_ae58183f.tar.xz` (`2.0 MiB`)
- **SHA256**: `90e4db7f52f725e969de4f9d3c386112122d77db15bafa3a30515d190660cae4`
- **Upstream Commit**: `ae58183fff34c545b3e3020d81d591f2a79658fc` (Apache License 2.0)
- **Included Subtrees**:
  - `flashinfer/experimental/{b12x,sm120_dsa_indexer,sm12x_mega_gate}`
  - `flashinfer/cute_dsl/`, `flashinfer/{mega_gate,dsa_indexer,api_logging}.py`
  - `flashinfer/jit/{core,cpp_ext,cute_dsl_core,_cute_dsl_source,_cuda_architecture,env}.py`
  - `flashinfer/data/csrc/{tvm_ffi_utils.h,cake_device_helpers,experimental/cake_host_common.cuh}`

## Usage

Unpack & verify SHA256, then pass the mount/strict-enforcement flags directly to `docker run`:

```bash
OVERLAY_ARGS="$(./scripts/prepare_sm120_overlay.sh --print-docker-args /tmp/zstep-sm120-overlay)"
# Mounts only the 16 experimental subpaths into vllm/vllm-openai:v0.31.0 and sets:
#   ZSTEP_REQUIRE_B12X=1 ZSTEP_REQUIRE_SM120_EXP=1 B12X_W4A8_TINY_DECODE=0
```

Setting `ZSTEP_REQUIRE_SM120_EXP=1` turns any missing `b12x`, `sm12x_mega_gate`, or `sm120_dsa_indexer` import/schedule into an immediate `RuntimeError` rather than silently falling back to `mega_gate=off dsa_decode=off`.
