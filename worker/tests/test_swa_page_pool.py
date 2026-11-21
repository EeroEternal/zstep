"""Unit tests for DeepSeek-V4.1 decoupled `_SwaPagePool`."""

from __future__ import annotations

import numpy as np
import pytest


def test_swa_page_pool_pinning_lru_eviction_and_fail_stop() -> None:
    try:
        from zstep_worker.models.deepseek_v41 import _SwaPagePool
    except ImportError as exc:
        pytest.skip(f"_SwaPagePool import skipped: {exc}")

    # 3 physical SWA pages (1..3); global dummy=10 -> swa_dummy_vb=4.
    pool = _SwaPagePool(num_swa_pages=3, mla_dummy_vb=10)

    # Seq 1 prefills tokens 0..255 (global pages 1, 2; active window cols 0..1).
    pt1, alloc1 = pool.map_active_windows(
        np.array([[1, 2, 10, 10]], dtype=np.int64),
        seq_pos_min=np.array([0], dtype=np.int64),
        seq_pos_max=np.array([255], dtype=np.int64),
        allow_alloc_from=np.array([0], dtype=np.int64),
    )
    assert alloc1 is True
    assert np.array_equal(pt1, np.array([[1, 2, 4, 4]], dtype=np.int32))
    assert np.array_equal(pool.lookup(np.array([1, 2, 10])), np.array([1, 2, 4], dtype=np.int32))

    # Seq 2 prefills tokens 0..127 (global page 3) -> fills capacity 3.
    pt2, alloc2 = pool.map_active_windows(
        np.array([[3, 10, 10, 10]], dtype=np.int64),
        seq_pos_min=np.array([0], dtype=np.int64),
        seq_pos_max=np.array([127], dtype=np.int64),
        allow_alloc_from=np.array([0], dtype=np.int64),
    )
    assert alloc2 is True
    assert int(pt2[0, 0]) == 3

    # Seq 1 advances to pos 256..300 (active window cols 1..2: global pages 2, 4),
    # unpinning global page 1 (LRU) while keeping page 2 pinned; Seq 2 stays at pos 127 (page 3).
    pt_next, alloc3 = pool.map_active_windows(
        np.array([[1, 2, 4, 10], [3, 10, 10, 10]], dtype=np.int64),
        seq_pos_min=np.array([256, 127], dtype=np.int64),
        seq_pos_max=np.array([300, 127], dtype=np.int64),
        allow_alloc_from=np.array([256, 127], dtype=np.int64),
    )
    assert alloc3 is True
    # Global page 4 evicted global page 1 and reused physical SWA page 1.
    assert int(pt_next[0, 2]) == 1
    assert int(pt_next[0, 0]) == 0  # evicted global page 1 cleared in LUT

    # Read-only prefix page missing from SWA pool inside active window (col=0 < w_lo=1) must fail loudly.
    with pytest.raises(RuntimeError, match="SWA pool miss on read-only prefix page"):
        pool.map_active_windows(
            np.array([[1, 2, 10, 10]], dtype=np.int64),
            seq_pos_min=np.array([128], dtype=np.int64),
            seq_pos_max=np.array([128], dtype=np.int64),
            allow_alloc_from=np.array([128], dtype=np.int64),
        )

    # Pool exhaustion when all 3 physical pages (2, 4, 3) are pinned in active windows [0, 1] and we need page 5.
    with pytest.raises(RuntimeError, match="SWA pool exhausted"):
        pool.map_active_windows(
            np.array([[2, 4, 10, 10], [3, 5, 10, 10]], dtype=np.int64),
            seq_pos_min=np.array([128, 128], dtype=np.int64),
            seq_pos_max=np.array([255, 255], dtype=np.int64),
            allow_alloc_from=np.array([0, 0], dtype=np.int64),
        )
