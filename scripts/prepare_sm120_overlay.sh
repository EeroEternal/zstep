#!/usr/bin/env bash
# Unpack and verify the deterministic SM120 FlashInfer experimental source overlay
# (`ae58183fff34c545b3e3020d81d591f2a79658fc`: `b12x`, `sm12x_mega_gate`, `sm120_dsa_indexer`)
# for `vllm/vllm-openai:v0.31.0`, and optionally emit ready-to-use `docker run` mount flags.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="${REPO_ROOT}/worker/third_party/flashinfer_sm120_overlay_ae58183f.tar.xz"
EXPECTED_SHA256="90e4db7f52f725e969de4f9d3c386112122d77db15bafa3a30515d190660cae4"
DIST_FLASHINFER="/usr/local/lib/python3.12/dist-packages/flashinfer"

PRINT_DOCKER_ARGS=0
if [[ "${1:-}" == "--print-docker-args" ]]; then
  PRINT_DOCKER_ARGS=1
  shift
fi

OVERLAY_DIR="${1:-${ZSTEP_SM120_OVERLAY_DIR:-/tmp/zstep-sm120-overlay-ae58183f}}"

if [[ ! -f "${ARCHIVE}" ]]; then
  echo "ERROR: overlay archive not found at ${ARCHIVE}" >&2
  exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL_SHA256="$(sha256sum "${ARCHIVE}" | awk '{print $1}')"
else
  ACTUAL_SHA256="$(shasum -a 256 "${ARCHIVE}" | awk '{print $1}')"
fi

if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
  echo "ERROR: SHA256 mismatch for ${ARCHIVE}: expected ${EXPECTED_SHA256}, got ${ACTUAL_SHA256}" >&2
  exit 1
fi

SUBPATHS=(
  "experimental/b12x"
  "experimental/sm120_dsa_indexer"
  "experimental/sm12x_mega_gate"
  "jit/core.py"
  "jit/cpp_ext.py"
  "jit/cute_dsl_core.py"
  "jit/_cute_dsl_source.py"
  "jit/_cuda_architecture.py"
  "jit/env.py"
  "cute_dsl"
  "mega_gate.py"
  "dsa_indexer.py"
  "api_logging.py"
  "data/csrc/tvm_ffi_utils.h"
  "data/csrc/cake_device_helpers"
  "data/csrc/experimental/cake_host_common.cuh"
)
COMPAT_PATHS=(
  "experimental/b12x/_compat/b12x"
  "experimental/b12x/_compat/_flashinfer_filelock.py"
)

STAMP_FILE="${OVERLAY_DIR}/.sha256"
NEED_EXTRACT=0
if [[ ! -f "${STAMP_FILE}" ]] || [[ "$(cat "${STAMP_FILE}")" != "${EXPECTED_SHA256}" ]]; then
  NEED_EXTRACT=1
else
  for rel in "${SUBPATHS[@]}" "${COMPAT_PATHS[@]}"; do
    if [[ ! -e "${OVERLAY_DIR}/flashinfer/${rel}" ]]; then
      NEED_EXTRACT=1
      break
    fi
  done
fi

if [[ "${NEED_EXTRACT}" -eq 1 ]]; then
  rm -rf "${OVERLAY_DIR}/flashinfer" "${STAMP_FILE}"
  mkdir -p "${OVERLAY_DIR}"
  tar -xJf "${ARCHIVE}" -C "${OVERLAY_DIR}"
  for rel in "${SUBPATHS[@]}" "${COMPAT_PATHS[@]}"; do
    if [[ ! -e "${OVERLAY_DIR}/flashinfer/${rel}" ]]; then
      echo "ERROR: extracted overlay missing expected path flashinfer/${rel}" >&2
      exit 1
    fi
  done
  printf '%s\n' "${EXPECTED_SHA256}" > "${STAMP_FILE}"
fi
mkdir -p "${OVERLAY_DIR}/fi-jit-cache"

if [[ "${PRINT_DOCKER_ARGS}" -eq 1 ]]; then
  OVERLAY_ABS="$(cd "${OVERLAY_DIR}" && pwd)"
  ARGS=("-v" "${OVERLAY_ABS}/fi-jit-cache:/fi-cache")
  for rel in "${SUBPATHS[@]}"; do
    ARGS+=("-v" "${OVERLAY_ABS}/flashinfer/${rel}:${DIST_FLASHINFER}/${rel}:ro")
  done
  ARGS+=(
    "-v" "${OVERLAY_ABS}/flashinfer/experimental/b12x/_compat/b12x:/usr/local/lib/python3.12/dist-packages/b12x:ro"
    "-v" "${OVERLAY_ABS}/flashinfer/experimental/b12x/_compat/_flashinfer_filelock.py:/usr/local/lib/python3.12/dist-packages/_flashinfer_filelock.py:ro"
    "-e" "FLASHINFER_WORKSPACE_BASE=/fi-cache"
    "-e" "ZSTEP_ENABLE_B12X=1"
    "-e" "ZSTEP_REQUIRE_B12X=1"
    "-e" "ZSTEP_REQUIRE_SM120_EXP=1"
    "-e" "B12X_W4A8_TINY_DECODE=0"
  )
  printf '%s ' "${ARGS[@]}"
  printf '\n'
else
  echo "${OVERLAY_DIR}"
fi
