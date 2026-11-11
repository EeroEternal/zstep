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

"""End-to-end 8-GPU Mode B greedy parity test for Qwen/Qwen3.5-35B-A3B (`TP=8`, `bfloat16`)."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys

import pytest
import torch

EXPECTED_PROMPT_IDS = [760, 6511, 314, 9338, 369]  # "The capital of France is"
EXPECTED_GREEDY_16 = [
    11751,
    13,
    198,
    760,
    6511,
    314,
    9338,
    369,
    11751,
    13,
    198,
    760,
    6511,
    314,
    9338,
    369,
]


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="8 CUDA GPUs required for TP=8")
def test_qwen35_35b_a3b_mode_b_tp8_greedy_parity() -> None:
    model_dir_env = os.environ.get("ZSTEP_QWEN35_MODEL_DIR", "")
    if not model_dir_env or not Path(model_dir_env).is_dir():
        pytest.skip("Set ZSTEP_QWEN35_MODEL_DIR to Qwen3.5-35B-A3B checkpoint directory")

    repo_root = Path(__file__).resolve().parents[3]
    zstep_bin = Path(os.environ.get("ZSTEP_BIN", str(repo_root / "target/release/zstep")))
    worker_root = repo_root / "worker"
    shm_name = f"zstep_qwen35_e2e_{os.getpid()}"

    cmd = [
        str(zstep_bin),
        "--mode-b",
        "--world-size",
        "8",
        "--python",
        sys.executable,
        "--worker-root",
        str(worker_root),
        "--shm-name",
        shm_name,
        "--model-dir",
        model_dir_env,
        "--num-kv-pages",
        "16384",
        "--max-seqs",
        "16",
        "--max-step-tokens",
        "512",
        "--max-decoder-rows",
        "512",
        "--max-pages-per-seq",
        "64",
        "--prompt-ids",
        ",".join(str(x) for x in EXPECTED_PROMPT_IDS),
        "--max-new-tokens",
        "16",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"zstep Mode B exited with {proc.returncode}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    match = re.search(r"GENERATED_IDS=\[([0-9,\s-]+)\]", proc.stdout)
    assert match is not None, f"Missing GENERATED_IDS in stdout:\n{proc.stdout}\n{proc.stderr}"
    generated = [int(tok.strip()) for tok in match.group(1).split(",") if tok.strip()]
    assert generated == EXPECTED_GREEDY_16, (
        f"Greedy token mismatch vs reference!\nGot:      {generated}\nExpected: {EXPECTED_GREEDY_16}"
    )

    # Verify RAII cleanup of all SHM segments
    assert not Path(f"/dev/shm/{shm_name}").exists()
    assert not Path(f"/dev/shm/{shm_name}_ar").exists()
    assert not Path(f"/dev/shm/{shm_name}_ar_argmax").exists()
