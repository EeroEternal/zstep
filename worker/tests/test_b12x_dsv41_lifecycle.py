# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from zstep_worker.kernels_sm120.b12x_dsv41 import (  # noqa: E402
    _CUTLASS_LARGE_M_THRESHOLDS,
    _slice_arena_for_specs,
    _specs_nbytes,
    close_b12x_dsv41_model,
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
    close_b12x_dsv41_model(plain)
    assert sess.close_calls == 1
    assert getattr(plain, "_b12x_session", None) is None

    # Second teardown call is a safe no-op.
    close_b12x_dsv41_model(plain)
    assert sess.close_calls == 1
