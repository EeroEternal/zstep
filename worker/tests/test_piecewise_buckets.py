"""Unit tests for DeepSeek-V4.1 piecewise prefill bucket coverage."""

from __future__ import annotations

import pytest


def test_piecewise_buckets_cover_full_1_to_1152_without_gaps() -> None:
    try:
        from zstep_worker.models.deepseek_v41 import (
            PREFILL_PIECEWISE_BUCKETS,
            DeepSeekV41StepRunner,
        )
    except ImportError as exc:
        pytest.skip(f"DeepSeekV41StepRunner import skipped: {exc}")

    runner = object.__new__(DeepSeekV41StepRunner)
    runner.piecewise_graphs = {b: object() for b in PREFILL_PIECEWISE_BUCKETS}

    for n in range(1, 1153):
        selected = runner._select_piecewise_bucket(n)
        assert selected is not None, f"blind spot at total_tokens={n}"
        assert selected >= n
        if selected > 128:
            assert selected - n <= 128

    assert runner._select_piecewise_bucket(1153) is None
