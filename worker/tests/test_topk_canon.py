# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT

"""Unit tests for DeepSeek-V4.1 canonical deterministic Top-K (`topk_canon`)."""

from __future__ import annotations

import torch

from zstep_worker.env_config import WorkerRuntimeConfig
from zstep_worker.kernels_sm120.topk_canon import canonicalize_prefill_topk


def test_canonicalize_prefill_topk_breaks_boundary_ties_by_smallest_index() -> None:
    # Row 0: unique top-2 above threshold (`10.0` at col 5, `9.0` at col 2) + 4 ties at `5.0` (cols 0, 3, 4, 7).
    # Adversarial non-canonical prefill output picked cols [5, 2, 7, 4] instead of smallest-index [5, 2, 0, 3].
    # Row 1: short row (`row_len = 3 <= top_k = 4`) -> canonical [0, 1, 2, -1].
    # Row 2: empty row (`ks == ke`) -> [-1, -1, -1, -1].
    logits = torch.tensor(
        [
            [5.0, 1.0, 9.0, 5.0, 5.0, 10.0, 0.0, 5.0],
            [1.0, 2.0, 3.0, 99.0, 99.0, 99.0, 99.0, 99.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    cu_ks = torch.tensor([0, 0, 2], dtype=torch.int32)
    cu_ke = torch.tensor([8, 3, 2], dtype=torch.int32)
    topk_indices = torch.tensor(
        [
            [5, 2, 7, 4],
            [2, 1, 0, -1],
            [0, 1, 2, 3],
        ],
        dtype=torch.int32,
    )

    canonicalize_prefill_topk(logits, cu_ks, cu_ke, topk_indices, top_k=4)

    # Row 0 must contain strict above {2, 5} + smallest 2 tied indices {0, 3}
    assert set(topk_indices[0].tolist()) == {0, 2, 3, 5}
    # Row 1 must be [0, 1, 2, -1]
    assert topk_indices[1].tolist() == [0, 1, 2, -1]
    # Row 2 must be all -1
    assert topk_indices[2].tolist() == [-1, -1, -1, -1]


def test_canonicalize_prefill_topk_no_ties_is_noop() -> None:
    logits = torch.tensor([[1.0, 5.0, 2.0, 8.0, 3.0, 6.0]], dtype=torch.float32)
    cu_ks = torch.tensor([0], dtype=torch.int32)
    cu_ke = torch.tensor([6], dtype=torch.int32)
    # Top-3 strictly are cols {3 (8.0), 5 (6.0), 1 (5.0)}; n_above == 2? No: s_min=5.0, above={3,5} (2 < 3), tie={1} (unique!).
    topk_indices = torch.tensor([[3, 5, 1]], dtype=torch.int32)
    canonicalize_prefill_topk(logits, cu_ks, cu_ke, topk_indices, top_k=3)
    assert set(topk_indices[0].tolist()) == {1, 3, 5}


def test_canonicalize_prefill_topk_nonzero_ks_offset_matches_torch_topk() -> None:
    logits = torch.tensor([[99.0, 99.0, 0.0, 0.0, 5.0, 0.0, 7.0, 0.0]], dtype=torch.float32)
    cu_ks = torch.tensor([2], dtype=torch.int32)
    cu_ke = torch.tensor([8], dtype=torch.int32)
    # Window is [0.0, 0.0, 5.0, 0.0, 7.0, 0.0] (len 6); adversarial topk picked ties at rel cols 3 and 5.
    topk_indices = torch.tensor([[4, 2, 3, 5]], dtype=torch.int32)
    canonicalize_prefill_topk(logits, cu_ks, cu_ke, topk_indices, top_k=4)
    # Canonical top-4 in window: strict above {2 (5.0), 4 (7.0)} + smallest 2 zeros {0, 1}
    assert set(topk_indices[0].tolist()) == {0, 1, 2, 4}


def test_env_config_enable_topk_canon() -> None:
    assert WorkerRuntimeConfig.from_env({}).enable_topk_canon is True
    assert (
        WorkerRuntimeConfig.from_env({"ZSTEP_ENABLE_TOPK_CANON": "0"}).enable_topk_canon
        is False
    )
