"""Unit tests for DeepSeek-V4.1 Prefill Two-Batch Overlap (TBO) gating and slicing."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _load_runner_cls():
    try:
        from zstep_worker.models.deepseek_v41 import DeepSeekV41StepRunner
    except ImportError as exc:
        pytest.skip(f"DeepSeekV41StepRunner import skipped: {exc}")
    return DeepSeekV41StepRunner


def _make_fake_runner(
    *,
    min_tokens: int = 6144,
    min_ctx: int = 32768,
    max_ubatch_tokens: int = 4096,
    fuse_mhc_ar: bool = True,
    use_sp: bool = False,
    mtp_buf: object | None = None,
) -> object:
    cls = _load_runner_cls()
    runner = cls.__new__(cls)
    runner.shm_ar_tbo = object()
    runner._tbo_comm_stream = object()
    runner._tbo_max_ubatch_tokens = max_ubatch_tokens
    runner.runtime_config = SimpleNamespace(
        prefill_tbo_min_tokens=min_tokens,
        prefill_tbo_min_ctx=min_ctx,
    )
    core = SimpleNamespace(
        fuse_mhc_all_reduce=fuse_mhc_ar,
        use_sequence_parallel=use_sp,
        _mtp_hidden_buffer=mtp_buf,
        layers=[],
        embed_input_ids=lambda x: x,
    )
    runner.model = SimpleNamespace(language_model=SimpleNamespace(model=core))
    return runner


def test_can_run_prefill_tbo_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _make_fake_runner()
    import zstep_worker.models.deepseek_v41 as dsv41_mod

    monkeypatch.setattr(dsv41_mod.torch.cuda, "is_current_stream_capturing", lambda: False)

    assert runner._can_run_prefill_tbo(8192, max_seq_len=65536, is_decode=False, is_spec_step=False)
    assert not runner._can_run_prefill_tbo(
        8192, max_seq_len=65536, is_decode=True, is_spec_step=False
    )
    assert not runner._can_run_prefill_tbo(
        8192, max_seq_len=65536, is_decode=False, is_spec_step=True
    )
    assert not runner._can_run_prefill_tbo(
        4096, max_seq_len=65536, is_decode=False, is_spec_step=False
    )
    assert not runner._can_run_prefill_tbo(
        8192, max_seq_len=32768, is_decode=False, is_spec_step=False
    )
    assert not runner._can_run_prefill_tbo(
        8194, max_seq_len=65536, is_decode=False, is_spec_step=False
    )

    runner.runtime_config.prefill_tbo_min_tokens = 0
    assert not runner._can_run_prefill_tbo(
        8192, max_seq_len=65536, is_decode=False, is_spec_step=False
    )


def test_compute_tbo_microbatch_slices_single_and_multi_sequence() -> None:
    cls = _load_runner_cls()

    # Single sequence chunked prefill (tokens 100..108, split at mid=4 -> 100..104 and 104..108).
    q_start_single = np.array([0, 8], dtype=np.int64)
    pos_single = np.arange(100, 108, dtype=np.int64)
    u0, u1 = cls._compute_tbo_microbatch_slices(8, 1, q_start_single, pos_single)
    assert u0 == (0, 4, 0, 1, [0, 4], [104], [True], 4, 104)
    assert u1 == (4, 8, 0, 1, [0, 4], [108], [True], 4, 108)

    # Multi-sequence batch: req 0 has 3 tokens (pos 0..3), req 1 has 2 tokens (pos 10..12),
    # req 2 has 1 token at mid=3..4 boundary (pos 50), req 3 has 2 tokens (pos 0..2).
    q_start_multi = np.array([0, 3, 5, 6, 8], dtype=np.int64)
    pos_multi = np.array([0, 1, 2, 10, 11, 50, 0, 1], dtype=np.int64)
    m0, m1 = cls._compute_tbo_microbatch_slices(8, 4, q_start_multi, pos_multi)
    # u0 covers tokens 0..4: req 0 (3 toks, pos 0->is_pref=True) + req 1 first token (ql=1, pos=10->False)
    assert m0 == (0, 4, 0, 2, [0, 3, 4], [3, 11], [True, False], 3, 11)
    # u1 covers tokens 4..8: req 1 second token (ql=1, pos=11->False), req 2 (ql=1, pos=50->False), req 3 (ql=2->True)
    assert m1 == (4, 8, 1, 4, [0, 1, 2, 4], [12, 51, 2], [False, False, True], 2, 51)
