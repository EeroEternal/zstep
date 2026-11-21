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
    _MHC_STATE_BY_DEV,
    _compute_b12x_source_fingerprint,
    _compute_mhc_split_cap,
    _prepare_sm12x_mega_gate,
    _slice_arena_for_specs,
    _specs_nbytes,
    _wrap_mhc_forward_join,
    attach_sm120_mhc_shm_ar,
    close_b12x_dsv41_model,
    flush_sm120_mhc_before_ar,
    is_sm120_dsa_decode_available,
    maybe_enable_b12x_for_dsv41,
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
    plain._sm120_mhc_dev = 77  # type: ignore[attr-defined]
    flushed: list[object] = []

    def _fake_flush(stream: object) -> None:
        flushed.append(stream)
        _MHC_STATE_BY_DEV[77].deferred = None

    _MHC_STATE_BY_DEV[77] = types.SimpleNamespace(
        dev=torch.device("cpu"),
        deferred=("pending_sinkhorn",),
        pending=False,
        flush=_fake_flush,
    )

    fake_dsa_api = types.ModuleType("flashinfer.experimental.sm120_dsa_indexer._api")
    fake_dsa_api._LOGITS = {0: torch.zeros(4)}  # type: ignore[attr-defined]
    fake_dsa_api._LONG_WS = {0: torch.zeros(4)}  # type: ignore[attr-defined]
    fake_dsa_api._RETIRED = [torch.zeros(4)]  # type: ignore[attr-defined]
    mod_key = "flashinfer.experimental.sm120_dsa_indexer._api"
    prev_mod = sys.modules.get(mod_key)
    sys.modules[mod_key] = fake_dsa_api
    try:
        flush_sm120_mhc_before_ar(None)
        flush_sm120_mhc_before_ar(77)
        assert len(flushed) == 1
        assert _MHC_STATE_BY_DEV[77].deferred is None
        # Second flush with `deferred=None` is a no-op.
        flush_sm120_mhc_before_ar(77)
        assert len(flushed) == 1

        close_b12x_dsv41_model(plain)
        assert sess.close_calls == 1
        assert getattr(plain, "_b12x_session", None) is None
        assert getattr(plain, "_b12x_scratch_arenas", None) is None
        assert getattr(plain, "_sm120_mega_gate_state", None) is None
        assert getattr(plain, "_sm120_mhc_dev", None) is None
        assert 77 not in _MHC_STATE_BY_DEV
        assert fake_dsa_api._LOGITS == {}
        assert fake_dsa_api._LONG_WS == {}
        assert fake_dsa_api._RETIRED == []

        # Second teardown call is a safe no-op.
        close_b12x_dsv41_model(plain)
        assert sess.close_calls == 1
    finally:
        _MHC_STATE_BY_DEV.pop(77, None)
        if prev_mod is None:
            sys.modules.pop(mod_key, None)
        else:
            sys.modules[mod_key] = prev_mod


def test_mhc_multi_model_owner_refcount_and_forward_join_guard() -> None:
    assert _compute_mhc_split_cap(lambda _k, n: 16 if n <= 2500 else 2, 4096) == 40000

    target = torch.nn.Linear(4, 4, bias=False)
    draft = torch.nn.Linear(4, 4, bias=False)
    dev_idx = 78
    join_flushes: list[int] = []

    def _fake_flush(_stream: object) -> None:
        join_flushes.append(1)
        _MHC_STATE_BY_DEV[dev_idx].deferred = None
        _MHC_STATE_BY_DEV[dev_idx].pending = True

    _MHC_STATE_BY_DEV[dev_idx] = types.SimpleNamespace(
        dev=torch.device("cpu"),
        mhc_stream=None,
        deferred=None,
        pending=False,
        flush=_fake_flush,
        owners={id(target), id(draft)},
    )
    target._sm120_mhc_dev = dev_idx  # type: ignore[attr-defined]
    draft._sm120_mhc_dev = dev_idx  # type: ignore[attr-defined]
    _wrap_mhc_forward_join(target, dev_idx)
    _wrap_mhc_forward_join(draft, dev_idx)
    try:
        # Normal forward exit flushes/joins any residual `deferred` / `pending` state.
        _MHC_STATE_BY_DEV[dev_idx].deferred = ("unflushed_tail",)
        _ = target(torch.zeros(2, 4))
        assert join_flushes == [1]
        assert _MHC_STATE_BY_DEV[dev_idx].deferred is None
        assert _MHC_STATE_BY_DEV[dev_idx].pending is False

        # Exception mid-forward clears `deferred` and `pending` without re-flushing stale args.
        class _BoomMod(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                _MHC_STATE_BY_DEV[dev_idx].deferred = ("stale_on_err",)
                _MHC_STATE_BY_DEV[dev_idx].pending = True
                raise RuntimeError("boom")

        boom = _BoomMod()
        _wrap_mhc_forward_join(boom, dev_idx)
        with pytest.raises(RuntimeError, match="boom"):
            boom(torch.zeros(2, 4))
        assert _MHC_STATE_BY_DEV[dev_idx].deferred is None
        assert _MHC_STATE_BY_DEV[dev_idx].pending is False
        assert join_flushes == [1]

        # Closing `draft` keeps `_MHC_STATE_BY_DEV[dev_idx]` alive for `target`.
        close_b12x_dsv41_model(draft)
        assert getattr(draft, "_sm120_mhc_dev", None) is None
        assert not getattr(draft, "_zstep_sm120_mhc_fwd_wrapped", False)
        assert dev_idx in _MHC_STATE_BY_DEV

        # Closing last owner `target` evicts `_MHC_STATE_BY_DEV[dev_idx]`.
        close_b12x_dsv41_model(target)
        assert getattr(target, "_sm120_mhc_dev", None) is None
        assert not getattr(target, "_zstep_sm120_mhc_fwd_wrapped", False)
        assert dev_idx not in _MHC_STATE_BY_DEV
    finally:
        _MHC_STATE_BY_DEV.pop(dev_idx, None)


def test_sm120_summary_defaults_and_env_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    assert SM120_DSA_CANDIDATE_COVER_TOKENS == 16384
    assert DEFAULT_MEGA_GATE_MIN_TOKENS == 1

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

    monkeypatch.delenv("ZSTEP_REQUIRE_SM120_EXP", raising=False)
    monkeypatch.setenv("ZSTEP_ENABLE_SM120_DSA_DECODE", "0")
    assert is_sm120_dsa_decode_available() is False


def test_require_sm120_exp_rejects_silent_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZSTEP_REQUIRE_SM120_EXP", "1")

    monkeypatch.setenv("ZSTEP_ENABLE_SM120_DSA_DECODE", "0")
    with pytest.raises(RuntimeError, match="ZSTEP_REQUIRE_SM120_EXP=1"):
        is_sm120_dsa_decode_available()
    monkeypatch.delenv("ZSTEP_ENABLE_SM120_DSA_DECODE", raising=False)

    monkeypatch.setenv("ZSTEP_ENABLE_SM12X_MEGA_GATE", "0")
    with pytest.raises(RuntimeError, match="ZSTEP_REQUIRE_SM120_EXP=1"):
        _prepare_sm12x_mega_gate(
            torch.nn.Linear(4, 4),
            rank=0,
            device=torch.device("cpu"),
            max_tokens=128,
            exchange_group=None,
        )
    monkeypatch.delenv("ZSTEP_ENABLE_SM12X_MEGA_GATE", raising=False)

    monkeypatch.setenv("ZSTEP_ENABLE_B12X", "0")
    with pytest.raises(RuntimeError, match="ZSTEP_REQUIRE_B12X=1"):
        maybe_enable_b12x_for_dsv41()
    monkeypatch.delenv("ZSTEP_ENABLE_B12X", raising=False)

    # Even with flags enabled, missing overlay packages must fail loudly rather than returning False/0.
    saved = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k == "flashinfer.experimental.sm120_dsa_indexer._api"
        or k.startswith("flashinfer.experimental.sm120_dsa_indexer")
        or k.startswith("flashinfer.experimental.sm12x_mega_gate")
    }
    monkeypatch.setitem(sys.modules, "flashinfer.experimental.sm120_dsa_indexer._api", None)
    monkeypatch.setitem(sys.modules, "flashinfer.experimental.sm12x_mega_gate.heuristics", None)
    try:
        with pytest.raises(RuntimeError, match="ZSTEP_REQUIRE_SM120_EXP=1"):
            is_sm120_dsa_decode_available()
        with pytest.raises(RuntimeError, match="ZSTEP_REQUIRE_SM120_EXP=1"):
            _prepare_sm12x_mega_gate(
                torch.nn.Linear(4, 4),
                rank=0,
                device=torch.device("cpu"),
                max_tokens=128,
                exchange_group=None,
            )
    finally:
        sys.modules.update(saved)


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


def test_dsv41_runner_abort_close_survives_early_init_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path
    from zstep_worker.models.deepseek_v41 import DeepSeekV41StepRunner

    closed_channels: list[str] = []

    class _FakeChan:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def signal_abort(self) -> None:
            closed_channels.append(f"abort:{self.tag}")

        def close(self, *, abort: bool = False) -> None:
            closed_channels.append(f"close:{self.tag}:{abort}")

    def _fail_early(self: DeepSeekV41StepRunner, **kwargs: object) -> None:
        self.shm_ar = _FakeChan("ar")  # type: ignore[assignment]
        self.argmax_channel = _FakeChan("argmax")  # type: ignore[assignment]
        raise ValueError("synthetic early _init_impl validation error")

    monkeypatch.setattr(DeepSeekV41StepRunner, "_init_impl", _fail_early)
    with pytest.raises(ValueError, match="synthetic early _init_impl validation error"):
        DeepSeekV41StepRunner(
            model_dir=Path("/nonexistent"),
            rank=0,
            bridge=None,  # type: ignore[arg-type]
            dist_init_file="/tmp/fake",
            allreduce_name="fake",
            allreduce_blocks=8,
            allreduce_threads=256,
            num_kv_pages=16,
        )
    assert "close:argmax:True" in closed_channels
    assert "close:ar:True" in closed_channels


def test_attach_sm120_mhc_shm_ar_uses_weakrefs_and_cleans_up() -> None:
    import gc

    class _Box:
        pass

    dev_idx = 79
    owner = torch.nn.Linear(4, 4, bias=False)
    owner._sm120_mhc_dev = dev_idx  # type: ignore[attr-defined]
    _MHC_STATE_BY_DEV[dev_idx] = types.SimpleNamespace(
        dev=torch.device("cpu"),
        mhc_stream=None,
        deferred=None,
        pending=False,
        flush=lambda _s: None,
        owners={id(owner)},
        shm_ar_ref=None,
        runner_ref=None,
    )
    try:
        attach_sm120_mhc_shm_ar(None, None, None)
        ar_obj = _Box()
        runner_obj = _Box()
        attach_sm120_mhc_shm_ar(dev_idx, ar_obj, runner_obj)
        st = _MHC_STATE_BY_DEV[dev_idx]
        assert callable(st.shm_ar_ref) and st.shm_ar_ref() is ar_obj
        assert callable(st.runner_ref) and st.runner_ref() is runner_obj

        del ar_obj
        del runner_obj
        gc.collect()
        assert st.shm_ar_ref() is None
        assert st.runner_ref() is None

        attach_sm120_mhc_shm_ar(dev_idx, None, None)
        assert st.shm_ar_ref is None
        assert st.runner_ref is None

        close_b12x_dsv41_model(owner)
        assert dev_idx not in _MHC_STATE_BY_DEV
    finally:
        _MHC_STATE_BY_DEV.pop(dev_idx, None)


def test_close_b12x_evicts_csmla_per_owner_independently_of_mhc() -> None:
    from zstep_worker.kernels_sm120.b12x_dsv41 import _CSMLA_STATE_BY_DEV

    main_model = torch.nn.Linear(4, 4)
    dspark_model = torch.nn.Linear(4, 4)
    dev_idx = 777
    sig_dec = (8, 128, 512, True, "decode")
    sig_ext = (8, 128, 512, True, "extend")
    main_model._b12x_csmla_dev = dev_idx  # type: ignore[attr-defined]
    dspark_model._b12x_csmla_dev = dev_idx  # type: ignore[attr-defined]
    # Intentionally leave `_sm120_mhc_dev` unset (None) on both models.
    _CSMLA_STATE_BY_DEV[dev_idx] = {
        sig_dec: (
            (16, "p_dspark", "s_dspark", [], id(dspark_model)),
            (128, "p_main", "s_main", [], id(main_model)),
        ),
        sig_ext: ((3072, "p_ext", "s_ext", "ws", id(main_model)),),
    }
    try:
        close_b12x_dsv41_model(dspark_model)
        assert getattr(dspark_model, "_b12x_csmla_dev", None) is None
        assert dev_idx in _CSMLA_STATE_BY_DEV
        assert [e[4] for e in _CSMLA_STATE_BY_DEV[dev_idx][sig_dec]] == [id(main_model)]
        assert sig_ext in _CSMLA_STATE_BY_DEV[dev_idx]

        close_b12x_dsv41_model(main_model)
        assert getattr(main_model, "_b12x_csmla_dev", None) is None
        assert dev_idx not in _CSMLA_STATE_BY_DEV
    finally:
        _CSMLA_STATE_BY_DEV.pop(dev_idx, None)
