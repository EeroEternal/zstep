# Third-Party SM120 FlashInfer 0.7.6 Source Overlay (`d0cbb476`)

`flashinfer_sm120_overlay_d0cbb476.tar.xz` bundles the pure-source Apache-2.0 SM120 experimental operator and PCIe IPC collective subset from `FlashInfer v0.7.6` commit `d0cbb476cf4041a619523c325109b4d51f2a45f1` required to reproduce `DeepSeek-V4.1-Flash` `TP=8` results (`b12x` Block32 FP8 + MXFP4 MoE, `sm12x_mega_gate` `#8479` `MegaGateSm12xPlan`, `sm120_dsa_indexer` `#8740` `candidate_blocks` / `candidate_out` paged decode up to 32K+, and `pcie_ipc_collectives` `#9112` Copy-Engine & spin timeout) on `vllm/vllm-openai:v0.31.0` without external network fetches:

- **Archive**: `flashinfer_sm120_overlay_d0cbb476.tar.xz` (`2.16 MiB`)
- **SHA256**: `e09f7c10503e9df6e7345b355cc3becee8d113cf45702963ae5a3833d56cd72c`
- **Upstream Commit**: `d0cbb476cf4041a619523c325109b4d51f2a45f1` (`v0.7.6`, Apache License 2.0)
- **Included Subtrees**:
  - `flashinfer/experimental/{b12x,sm120_dsa_indexer,sm12x_mega_gate,deepgemm_sm120_paged_mqa_logits}`
  - `flashinfer/cute_dsl/`, `flashinfer/{mega_gate,dsa_indexer,api_logging}.py`
  - `flashinfer/jit/{core,cpp_ext,comm,cute_dsl_core,_cute_dsl_source,_cuda_architecture,env}.py`
  - `flashinfer/comm/{pcie_ipc_ar,pcie_ipc_policy,pcie_ipc_topology,pcie_ipc_tuning,cuda_ipc}.py`, `flashinfer/comm/pcie_ipc_collectives/`
  - `flashinfer/data/csrc/{tvm_ffi_utils.h,cake_device_helpers,experimental/cake_host_common.cuh,pcie_ipc_all_reduce.cu,pcie_ipc_all_gather.cu,pcie_ipc_reduce_scatter.cu}`
  - `flashinfer/data/include/flashinfer/comm/`

## Usage

Unpack & verify SHA256, then pass the mount/strict-enforcement flags directly to `docker run`:

```bash
OVERLAY_ARGS="$(./scripts/prepare_sm120_overlay.sh --print-docker-args /tmp/zstep-sm120-overlay-d0cbb476)"
# Mounts the SM120 experimental/comm/compat subpaths into vllm/vllm-openai:v0.31.0 and sets:
#   ZSTEP_REQUIRE_B12X=1 ZSTEP_REQUIRE_SM120_EXP=1 B12X_W4A8_TINY_DECODE=0
```

Setting `ZSTEP_REQUIRE_SM120_EXP=1` turns any missing `b12x`, `sm12x_mega_gate`, or `sm120_dsa_indexer` import/schedule into an immediate `RuntimeError` rather than silently falling back to `mega_gate=off dsa_decode=off`.
