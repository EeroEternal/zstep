# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from zstep_worker.kernels_sm120.b12x_dsv41 import (  # noqa: E402
    DEFAULT_MEGA_GATE_MIN_TOKENS,
    SM120_DSA_CANDIDATE_COVER_TOKENS,
    B12xPrepareSummary,
    _CUTLASS_LARGE_M_THRESHOLDS,
    _compute_b12x_source_fingerprint,
    _slice_arena_for_specs,
    _specs_nbytes,
    close_b12x_dsv41_model,
    is_sm120_dsa_decode_available,
)


@dataclass(frozen=True)
class _FakeSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype


class _FakeSession:
    def __init__(self) -> None:
        self.close_calls: int = 0

    def close(self) -> None:
        self.close_calls += 1


def test_cutlass_thresholds_cover_156sm_autotuned_crossovers() -> None:
    assert _CUTLASS_LARGE_M_THRESHOLDS == {
        (1792, 5120): 2048,
        (3200, 6144): 512,
    }
    # `shared_experts.down_proj` (`5120 x 288`, `288 % 32 == 0`) stays on `b12x` for all M.
    assert _CUTLASS_LARGE_M_THRESHOLDS.get((5120, 288), 0) == 0


def test_scratch_arena_slices_are_1024_byte_aligned() -> None:
    specs = (
        _FakeSpec(shape=(17,), dtype=torch.uint8),
        _FakeSpec(shape=(8, 4), dtype=torch.float32),
        _FakeSpec(shape=(3, 16), dtype=torch.bfloat16),
    )
    total = _specs_nbytes(specs)
    assert total == 1024 + 1024 + (3 * 16 * 2)

    arena = torch.zeros(total + 1024, dtype=torch.uint8)
    # Shift start to a 1024-byte aligned offset inside `arena` to check relative offsets.
    base_ptr = arena.data_ptr()
    views = _slice_arena_for_specs(arena, specs)
    assert len(views) == 3
    assert (views[0].data_ptr() - base_ptr) == 0
    assert (views[1].data_ptr() - base_ptr) == 1024
    assert (views[2].data_ptr() - base_ptr) == 2048
    assert views[1].shape == (8, 4) and views[1].dtype == torch.float32
    assert views[2].shape == (3, 16) and views[2].dtype == torch.bfloat16


def test_close_b12x_dsv41_model_is_idempotent() -> None:
    close_b12x_dsv41_model(None)

    plain = torch.nn.Linear(4, 4, bias=False)
    close_b12x_dsv41_model(plain)

    sess = _FakeSession()
    plain._b12x_session = sess  # type: ignore[attr-defined]
    plain._b12x_scratch_arenas = (torch.zeros(4), {})  # type: ignore[attr-defined]
    plain._sm120_mega_gate_state = object()  # type: ignore[attr-defined]

    fake_dsa_api = types.ModuleType("flashinfer.experimental.sm120_dsa_indexer._api")
    fake_dsa_api._LOGITS = {0: torch.zeros(4)}  # type: ignore[attr-defined]
    fake_dsa_api._LONG_WS = {0: torch.zeros(4)}  # type: ignore[attr-defined]
    fake_dsa_api._RETIRED = [torch.zeros(4)]  # type: ignore[attr-defined]
    mod_key = "flashinfer.experimental.sm120_dsa_indexer._api"
    prev_mod = sys.modules.get(mod_key)
    sys.modules[mod_key] = fake_dsa_api
    try:
        close_b12x_dsv41_model(plain)
        assert sess.close_calls == 1
        assert getattr(plain, "_b12x_session", None) is None
        assert getattr(plain, "_b12x_scratch_arenas", None) is None
        assert getattr(plain, "_sm120_mega_gate_state", None) is None
        assert fake_dsa_api._LOGITS == {}
        assert fake_dsa_api._LONG_WS == {}
        assert fake_dsa_api._RETIRED == []

        # Second teardown call is a safe no-op.
        close_b12x_dsv41_model(plain)
        assert sess.close_calls == 1
    finally:
        if prev_mod is None:
            sys.modules.pop(mod_key, None)
        else:
            sys.modules[mod_key] = prev_mod


def test_sm120_summary_defaults_and_env_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    assert SM120_DSA_CANDIDATE_COVER_TOKENS == 16384
    assert DEFAULT_MEGA_GATE_MIN_TOKENS == 32

    summary = B12xPrepareSummary(
        enabled=True,
        moe_layers=40,
        linear_layers=210,
        linear_shapes=7,
        moe_scratch_bytes=1024,
        linear_scratch_bytes=2048,
        prepare_ms=10.0,
    )
    assert summary.mega_gate_layers == 0
    assert summary.mega_gate_schedules == 0
    assert summary.mega_gate_min_tokens == 0
    assert summary.dsa_decode_enabled is False

    monkeypatch.setenv("ZSTEP_ENABLE_SM120_DSA_DECODE", "0")
    assert is_sm120_dsa_decode_available() is False


def test_b12x_source_fingerprint_changes_with_source(
    tmp_path: pytest.TempPathFactory,
) -> None:
    pkg = tmp_path / "b12x"  # type: ignore[operator]
    pkg.mkdir()
    init_py = pkg / "__init__.py"
    init_py.write_text("__version__ = '1'\n", encoding="utf-8")

    fake_b12x = types.ModuleType("b12x")
    fake_b12x.__file__ = str(init_py)
    prev = sys.modules.get("b12x")
    sys.modules["b12x"] = fake_b12x
    try:
        fp1 = _compute_b12x_source_fingerprint()
        init_py.write_text("__version__ = '2'\n", encoding="utf-8")
        fp2 = _compute_b12x_source_fingerprint()
        assert len(fp1) == 10
        assert fp1 != fp2
    finally:
        if prev is None:
            sys.modules.pop("b12x", None)
        else:
            sys.modules["b12x"] = prev
