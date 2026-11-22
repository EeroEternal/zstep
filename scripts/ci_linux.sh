#!/usr/bin/env bash
set -euo pipefail

# Sync local zstep tree to a configurable Linux host and run full Rust + Python CI gates
# without pulling from GitHub on the remote host.
#
# Configuration precedence (highest to lowest):
#   1. CLI flags: --host <user@host> --port <port> --dir <remote_dir> --ssh-opts <opts>
#   2. Environment variables: ZSTEP_CI_HOST, ZSTEP_CI_PORT, ZSTEP_CI_DIR, ZSTEP_CI_SSH_OPTS
#   3. Repo-local config: <repo>/.ci_linux.env (gitignored)
#   4. User config: ~/.config/zstep/ci_linux.env

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_HOST="${ZSTEP_CI_HOST-}"
ENV_PORT="${ZSTEP_CI_PORT-}"
ENV_DIR="${ZSTEP_CI_DIR-}"
ENV_SSH_OPTS="${ZSTEP_CI_SSH_OPTS-}"

for cfg in "$HOME/.config/zstep/ci_linux.env" "$ROOT_DIR/.ci_linux.env"; do
  if [ -f "$cfg" ]; then
    # shellcheck disable=SC1090
    source "$cfg"
  fi
done

REMOTE_HOST="${ENV_HOST:-${ZSTEP_CI_HOST-}}"
REMOTE_PORT="${ENV_PORT:-${ZSTEP_CI_PORT:-22}}"
REMOTE_DIR="${ENV_DIR:-${ZSTEP_CI_DIR:-zstep-ci}}"
SSH_OPTS="${ENV_SSH_OPTS:-${ZSTEP_CI_SSH_OPTS:--o BatchMode=yes}}"

usage() {
  cat <<'EOF'
Usage: ./scripts/ci_linux.sh [options]

Options:
  --host <user@host>   Remote Linux SSH target (or set ZSTEP_CI_HOST)
  --port <port>        SSH port (default: 22, or ZSTEP_CI_PORT)
  --dir <remote_dir>   Remote workspace directory (default: zstep-ci, or ZSTEP_CI_DIR)
  --ssh-opts <opts>    Extra ssh options (default: "-o BatchMode=yes", or ZSTEP_CI_SSH_OPTS)
  -h, --help           Show this help message

Config files (optional, lowest precedence):
  ~/.config/zstep/ci_linux.env
  <repo>/.ci_linux.env
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --host)
      REMOTE_HOST="${2:?--host requires a value}"
      shift 2
      ;;
    --port)
      REMOTE_PORT="${2:?--port requires a value}"
      shift 2
      ;;
    --dir)
      REMOTE_DIR="${2:?--dir requires a value}"
      shift 2
      ;;
    --ssh-opts)
      SSH_OPTS="${2:?--ssh-opts requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "$REMOTE_HOST" ]; then
  echo "Error: ZSTEP_CI_HOST is not configured." >&2
  echo "Pass --host <user@host>, export ZSTEP_CI_HOST, or create ~/.config/zstep/ci_linux.env" >&2
  exit 2
fi

echo "==> Syncing $ROOT_DIR -> $REMOTE_HOST:$REMOTE_DIR (port $REMOTE_PORT)"
rsync -az --delete \
  --exclude '.git/' \
  --exclude 'target/' \
  --exclude '.venv/' \
  --exclude '.pytest_cache/' \
  --exclude '__pycache__/' \
  --exclude 'docs/assets/' \
  --exclude '.ci_linux.env' \
  -e "ssh -p $REMOTE_PORT $SSH_OPTS" \
  "$ROOT_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"

echo "==> Running Linux Rust + Python CI on $REMOTE_HOST"
# shellcheck disable=SC2086
ssh -p "$REMOTE_PORT" $SSH_OPTS "$REMOTE_HOST" bash -s -- "$REMOTE_DIR" <<'EOF'
set -euo pipefail
REMOTE_DIR="${1/#\~/$HOME}"
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
cd "$REMOTE_DIR"

echo "--- cargo fmt --all --check"
cargo fmt --all --check

echo "--- cargo clippy --workspace --all-targets -- -D warnings"
cargo clippy --workspace --all-targets -- -D warnings

echo "--- cargo test --workspace"
cargo test --workspace

if [ ! -x .venv/bin/python ]; then
  echo "--- creating Python 3.12 .venv with uv"
  uv venv --python 3.12 .venv
fi
echo "--- syncing worker[test] into .venv"
uv pip install --python .venv/bin/python -e "./worker[test]"

echo "--- ruff check worker/"
.venv/bin/ruff check worker/

echo "--- pytest CPU/SHM/AST suites"
.venv/bin/pytest -q \
  worker/tests/test_shm_layout_parity.py \
  worker/tests/test_p2p_probe.py \
  worker/tests/test_spec_decode.py \
  worker/tests/test_b12x_dsv41_lifecycle.py \
  worker/tests/test_coordinator_e2e.py \
  worker/tests/test_env_config.py \
  worker/tests/test_hotpath_ast_guards.py \
  worker/tests/test_sampler.py \
  worker/tests/test_swa_page_pool.py
EOF
