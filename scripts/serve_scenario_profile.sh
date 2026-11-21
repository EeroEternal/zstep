#!/usr/bin/env bash
# Launch zs-server serve with near-term fixed-scenario capability presets.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: serve_scenario_profile.sh --profile <agent|coding|content_gen|stress> [options] [-- extra_zs_args...]

Profiles:
  agent        ISL~50k,  OSL~1k, hit~0.7 : 64k ctx, APC+DRAM KV, chunk=1024, piecewise prefill
  coding       ISL~100k, OSL~2k, hit~0.9 : 128k ctx, APC+DRAM KV, chunk=1024, piecewise prefill
  content_gen  ISL~1k,   OSL~8k, hit=0   : 16k ctx, no-APC/no-DRAM, dspark spec decode
  stress       ISL~32k,  OSL~1k, hit=0   : 40k ctx, no-APC/no-DRAM, chunk=2048, step_tokens=4096

Options:
  --profile NAME      Scenario profile name (or set ZSTEP_SCENARIO_PROFILE)
  --model-dir PATH    Model directory (default: $MODEL_DIR)
  --zs-bin PATH       zs-server binary (default: zs-server)
  --port PORT         HTTP serve port (default: 8000)
  --dry-run           Print the resolved command without executing
  -h, --help          Show this help
EOF
}

PROFILE="${ZSTEP_SCENARIO_PROFILE:-}"
MODEL_DIR="${MODEL_DIR:-/data02/models/deepseek-ai_DeepSeek-V4.1-Flash-1017/main}"
ZS_BIN="${ZS_BIN:-zs-server}"
PORT="${PORT:-8000}"
DRY_RUN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="${2:-}"
      shift 2
      ;;
    --model-dir)
      MODEL_DIR="${2:-}"
      shift 2
      ;;
    --zs-bin)
      ZS_BIN="${2:-}"
      shift 2
      ;;
    --port)
      PORT="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$PROFILE" ]]; then
  echo "error: --profile is required (agent | coding | content_gen | stress)" >&2
  exit 2
fi

export NEBULA_KV_HASH_SEED="${NEBULA_KV_HASH_SEED:-nebula-v1}"

COMMON_ARGS=(
  serve
  --model-dir "$MODEL_DIR"
  --port "$PORT"
  --world-size "${WORLD_SIZE:-8}"
)

case "$PROFILE" in
  agent)
    export ZSTEP_ENABLE_SM120_DSA_DECODE="${ZSTEP_ENABLE_SM120_DSA_DECODE:-0}"
    PROFILE_ARGS=(
      --served-model-name "${SERVED_MODEL_NAME:-deepseek-v41-agent}"
      --max-model-len 65536
      --enable-prefix-caching
      --dram-kv-blocks "${DRAM_KV_BLOCKS:-16384}"
      --kv-loading-area-ratio "${KV_LOADING_AREA_RATIO:-0.15}"
      --prefill-policy srpt
      --max-prefill-chunk 1024
      --max-mixed-step-tokens 3072
      --dynamic-prefill-budget
      --speculative-method none
    )
    ;;
  coding)
    export ZSTEP_ENABLE_SM120_DSA_DECODE="${ZSTEP_ENABLE_SM120_DSA_DECODE:-0}"
    PROFILE_ARGS=(
      --served-model-name "${SERVED_MODEL_NAME:-deepseek-v41-coding}"
      --max-model-len 131072
      --enable-prefix-caching
      --dram-kv-blocks "${DRAM_KV_BLOCKS:-32768}"
      --kv-loading-area-ratio "${KV_LOADING_AREA_RATIO:-0.15}"
      --prefill-policy srpt
      --max-prefill-chunk 1024
      --max-mixed-step-tokens 3072
      --dynamic-prefill-budget
      --speculative-method none
    )
    ;;
  content_gen)
    export ZSTEP_ENABLE_SM120_DSA_DECODE="${ZSTEP_ENABLE_SM120_DSA_DECODE:-1}"
    PROFILE_ARGS=(
      --served-model-name "${SERVED_MODEL_NAME:-deepseek-v41-content-gen}"
      --max-model-len 16384
      --no-enable-prefix-caching
      --dram-kv-blocks 0
      --max-prefill-chunk 1024
      --max-mixed-step-tokens 3072
      --speculative-method dspark
    )
    ;;
  stress)
    export ZSTEP_ENABLE_SM120_DSA_DECODE="${ZSTEP_ENABLE_SM120_DSA_DECODE:-0}"
    PROFILE_ARGS=(
      --served-model-name "${SERVED_MODEL_NAME:-deepseek-v41-stress}"
      --max-model-len 40960
      --no-enable-prefix-caching
      --dram-kv-blocks 0
      --prefill-policy srpt
      --prefill-starvation-cap-ms 5000
      --max-prefill-chunk 2048
      --max-step-tokens 4096
      --max-mixed-step-tokens 4096
      --speculative-method none
    )
    ;;
  *)
    echo "error: unknown profile '$PROFILE' (expected agent|coding|content_gen|stress)" >&2
    exit 2
    ;;
esac

CMD=("$ZS_BIN" "${COMMON_ARGS[@]}" "${PROFILE_ARGS[@]}" "${EXTRA_ARGS[@]}")
if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

exec "${CMD[@]}"
