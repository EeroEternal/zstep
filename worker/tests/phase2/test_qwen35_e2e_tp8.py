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

"""End-to-end 8-GPU Mode B multi-sequence and chunked-prefill greedy parity."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import pytest
import torch


def match_statistics(expected: list[int], generated: list[int]) -> dict[str, Any]:
    total = len(expected)
    prefix = 0
    for expected_token, generated_token in zip(expected, generated, strict=True):
        if expected_token != generated_token:
            break
        prefix += 1
    exact_matches = sum(
        expected_token == generated_token
        for expected_token, generated_token in zip(expected, generated, strict=True)
    )
    return {
        "exact_prefix_match_tokens": prefix,
        "total_tokens": total,
        "exact_token_match_rate": exact_matches / total,
        "all_matched": prefix == total,
    }


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="8 CUDA GPUs required for TP=8")
def test_qwen35_35b_a3b_mode_b_tp8_greedy_parity() -> None:
    model_dir_env = os.environ.get("ZSTEP_QWEN35_MODEL_DIR", "")
    if not model_dir_env or not Path(model_dir_env).is_dir():
        pytest.skip("Set ZSTEP_QWEN35_MODEL_DIR to Qwen3.5-35B-A3B checkpoint directory")

    repo_root = Path(__file__).resolve().parents[3]
    zstep_bin = Path(os.environ.get("ZSTEP_BIN", str(repo_root / "target/release/zstep")))
    worker_root = repo_root / "worker"
    reference_path = repo_root / "docs/benchmarks/phase2_vllm_greedy_reference.json"
    report_path = repo_root / "docs/benchmarks/phase2_greedy_parity_report.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    assert len(reference["cases"]) == 5
    assert 65 <= len(reference["cases"][3]["prompt_token_ids"]) <= 72
    assert reference["cases"][4]["name"] == "prime_squares_table"
    assert reference["cases"][4]["max_tokens"] == 128
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
        "--verify-greedy-json",
        str(reference_path),
        "--verify-output-json",
        str(report_path),
        "--max-prefill-chunk",
        "24",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, (
        f"zstep Mode B exited with {proc.returncode}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    marker = re.search(r"TRITON_RECOMPILATIONS=(\d+) WATCHED_KERNELS=(\d+)", proc.stdout)
    assert marker is not None, f"Missing Triton audit marker:\n{proc.stdout}\n{proc.stderr}"
    assert marker.group(1) == "0"
    assert marker.group(2) == "17"

    for suffix in ("", "_ar", "_ar_argmax"):
        assert not Path(f"/dev/shm/{shm_name}{suffix}").exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert re.fullmatch(r"[0-9a-f]{40}", report["git_sha"])
    assert report["max_prefill_chunk"] == 24
    assert len(report["cases"]) == 5
    assert report["cases"][1]["total_tokens"] == 128
    assert len(report["cases"][1]["single_generated_token_ids"]) == 128
    assert report["cases"][4]["total_tokens"] == 128
    prime_tokens = report["cases"][4]["single_generated_token_ids"]
    assert len({tuple(prime_tokens[i : i + 16]) for i in range(len(prime_tokens) - 15)}) == (
        len(prime_tokens) - 15
    )
    assert report["cases"][4]["single_match"]["all_matched"] is True
    assert report["cases"][4]["batched_match"]["all_matched"] is True
    assert report["cases"][4]["chunked_match"]["all_matched"] is True
    assert report["batched_vs_single_exact_match"] == all(
        case["batched_generated_token_ids"] == case["single_generated_token_ids"]
        for case in report["cases"]
    )
    assert report["chunked_vs_unchunked_exact_match"] == all(
        case["chunked_generated_token_ids"] == case["batched_generated_token_ids"]
        for case in report["cases"]
    )
    assert report["unchunked_schedule"]["prefill_steps"] == 1
    assert report["unchunked_schedule"]["max_active_sequences"] == 5
    assert report["chunked_schedule"]["max_active_sequences"] == 5
    assert report["chunked_schedule"]["prefill_steps"] >= 3
    assert report["chunked_schedule"]["decode_steps_with_partial_prefill"] >= 1

    for reference_case, report_case in zip(reference["cases"], report["cases"], strict=True):
        assert report_case["name"] == reference_case["name"]
        expected = reference_case["expected_greedy_token_ids"]
        for mode in ("single", "batched", "chunked"):
            generated = report_case[f"{mode}_generated_token_ids"]
            assert len(generated) == len(expected)
            expected_stats = match_statistics(expected, generated)
            actual_stats = report_case[f"{mode}_match"]
            for key, value in expected_stats.items():
                if isinstance(value, float):
                    assert actual_stats[key] == pytest.approx(value)
                else:
                    assert actual_stats[key] == value
            assert actual_stats["mean_top1_minus_top2_margin"] >= 0.0
            assert actual_stats["min_top1_minus_top2_margin"] >= 0.0
            if not actual_stats["all_matched"]:
                diag = actual_stats["first_mismatch_diagnostics"]
                assert diag is not None
                assert diag["prefix_identical_at_step"] is True
                assert diag["vllm_top2"] is not None
                # When the prefix is still identical to vLLM (`single` mode, or `batched`/`chunked`
                # if it has not diverged earlier from `single`), any BF16 argmax flip must be a
                # razor-thin near-tie (`margin < 0.25`), never a confident logit mismatch.
                if mode == "single":
                    assert diag["vllm_top2"]["top1_minus_top2_margin"] < 0.25
                    assert diag["zstep_top2"]["top1_minus_top2_margin"] < 0.25
        for pair_key in ("single_vs_batched_first_mismatch", "batched_vs_chunked_first_mismatch"):
            pair_diag = report_case[pair_key]
            if pair_diag is not None:
                assert pair_diag["prefix_identical_at_step"] is True
                assert pair_diag["left_top2"]["top1_minus_top2_margin"] < 0.25
                assert pair_diag["right_top2"]["top1_minus_top2_margin"] < 0.25

    backward_expected = reference["cases"][0]["expected_greedy_token_ids"][:16]
    assert report["cases"][0]["single_generated_token_ids"][:16] == backward_expected

    report["triton_recompilations"] = int(marker.group(1))
    report["watched_kernels"] = int(marker.group(2))
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
