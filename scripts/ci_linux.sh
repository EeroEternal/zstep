#!/usr/bin/env bash
set -euo pipefail

# Sync local zstep tree to bodesi-102 and run full Linux Rust + Python CI gates
# without pulling from GitHub on the remote host.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${ZSTEP_CI_HOST:-bodesi@117.160.123.99}"
REMOTE_PORT="${ZSTEP_CI_PORT:-60001}"
REMOTE_DIR="${ZSTEP_CI_DIR:-~/zstep-ci}"

echo "==> Syncing $ROOT_DIR -> $REMOTE_HOST:$REMOTE_DIR (port $REMOTE_PORT)"
rsync -az --delete \
  --exclude '.git/' \
  --exclude 'target/' \
  --exclude '.venv/' \
  --exclude '.pytest_cache/' \
  --exclude '__pycache__/' \
  --exclude 'docs/assets/' \
  -e "ssh -p $REMOTE_PORT -o BatchMode=yes" \
  "$ROOT_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"

echo "==> Running Linux Rust + Python CI on $REMOTE_HOST"
ssh -p "$REMOTE_PORT" -o BatchMode=yes "$REMOTE_HOST" bash -s -- "$REMOTE_DIR" <<'EOF'
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
