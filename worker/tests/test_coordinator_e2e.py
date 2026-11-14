# Copyright (c) 2026 zstep Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""End-to-end Phase 1 test: Rust `zstep` Master <-> Python `sm_120` Workers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run_coordinator_e2e(*, zstep_bin: Path, world_size: int, steps: int) -> dict[str, object]:
    worker_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="zstep_coordinator_test_") as tmp_dir:
        metrics_path = Path(tmp_dir) / "metrics.json"
        shm_name = f"zstep_e2e_{os.getpid()}_{world_size}"
        cmd = [
            str(zstep_bin),
            "--world-size",
            str(world_size),
            "--steps",
            str(steps),
            "--token-counts",
            "6,17,48,192",
            "--python",
            sys.executable,
            "--worker-root",
            str(worker_root),
            "--shm-name",
            shm_name,
            "--metrics-output",
            str(metrics_path),
            "--max-seqs",
            "8",
            "--max-step-tokens",
            "192",
            "--max-decoder-rows",
            "192",
            "--allreduce-blocks",
            "32",
            "--allreduce-threads",
            "256",
        ]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        assert metrics_path.is_file(), f"Missing metrics output; stdout={proc.stdout}"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert metrics["engine"] == "zstep"
        assert metrics["world_size"] == world_size
        assert metrics["steps"] == steps
        assert metrics["accepted_tokens"] > 0
        # Verify RAII cleanup: POSIX SHM segment must be unlinked after exit.
        assert not Path(f"/dev/shm/{shm_name}").exists()
        assert not Path(f"/dev/shm/{shm_name}_ar").exists()
        return metrics


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    binary = Path(os.environ.get("ZSTEP_BIN", str(repo_root / "target/release/zstep")))
    m1 = run_coordinator_e2e(zstep_bin=binary, world_size=1, steps=24)
    print("1-GPU Coordinator OK:", json.dumps(m1))
    m8 = run_coordinator_e2e(zstep_bin=binary, world_size=8, steps=64)
    print("8-GPU Coordinator OK:", json.dumps(m8))
