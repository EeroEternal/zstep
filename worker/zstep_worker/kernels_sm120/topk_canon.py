# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT

"""Canonical deterministic Top-K tie-breaking (`topk_canon`, Issue #40 Proposal 5).

`ops.top_k_per_row_prefill` (used in DeepSeek-V4.1 sparse attention prefill) can break
floating-point ties non-deterministically depending on warp/block scheduling order,
especially when quantized MQA logits produce repeated values (e.g. ReLU zeros).

`canonicalize_prefill_topk` enforces canonical smallest-key-index tie-breaking matching
`dsa_indexer_det_topk_paged` (decode) and `torch.topk(..., largest=True)`:
1. Fast-path exit in Pass 1 whenever `count(score > k-th threshold) >= top_k` (zero ties).
2. When ties exist on the `k`-th threshold, fills the remaining slots with the smallest
   key indices `j in [0, row_len)` whose score equals the `k`-th threshold.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _canonicalize_topk_ties_kernel(
        logits_ptr,
        cu_ks_ptr,
        cu_ke_ptr,
        topk_ptr,
        logits_stride_row,
        logits_stride_col,
        topk_stride_row,
        TOP_K: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        ks = tl.load(cu_ks_ptr + row).to(tl.int64)
        ke = tl.load(cu_ke_ptr + row).to(tl.int64)
        row_len = ke - ks

        k_offs = tl.arange(0, BLOCK_K)
        k_mask = k_offs < TOP_K
        out_row_ptr = topk_ptr + row * topk_stride_row.to(tl.int64)

        if row_len <= 0:
            tl.store(out_row_ptr + k_offs, -1, mask=k_mask)
            return

        if row_len <= TOP_K:
            canon = tl.where(k_offs < row_len, k_offs.to(tl.int32), -1)
            tl.store(out_row_ptr + k_offs, canon, mask=k_mask)
            return

        row_logits_ptr = logits_ptr + row * logits_stride_row.to(tl.int64)
        sel_idx = tl.load(out_row_ptr + k_offs, mask=k_mask, other=-1).to(tl.int64)
        sel_valid = k_mask & (sel_idx >= 0) & (sel_idx < row_len)
        sel_scores = tl.load(
            row_logits_ptr + (ks + sel_idx) * logits_stride_col.to(tl.int64),
            mask=sel_valid,
            other=float("inf"),
        ).to(tl.float32)
        s_min = tl.min(sel_scores, axis=0)
        # Bail out if s_min is NaN or +inf (no valid selected indices).
        if s_min != s_min or s_min == float("inf"):
            return

        n_above = tl.zeros([], dtype=tl.int32)
        n_eq = tl.zeros([], dtype=tl.int32)
        n_offs = tl.arange(0, BLOCK_N)
        for base in range(0, row_len, BLOCK_N):
            cols = base + n_offs
            c_mask = cols < row_len
            vals = tl.load(
                row_logits_ptr + (ks + cols) * logits_stride_col.to(tl.int64),
                mask=c_mask,
                other=-float("inf"),
            ).to(tl.float32)
            n_above += tl.sum(tl.where(c_mask & (vals > s_min), 1, 0), axis=0)
            n_eq += tl.sum(tl.where(c_mask & (vals == s_min), 1, 0), axis=0)

        if n_above >= TOP_K or n_above + n_eq <= TOP_K:
            return

        above_cursor = tl.zeros([], dtype=tl.int32)
        tie_cursor = tl.zeros([], dtype=tl.int32)
        for base in range(0, row_len, BLOCK_N):
            cols = base + n_offs
            c_mask = cols < row_len
            vals = tl.load(
                row_logits_ptr + (ks + cols) * logits_stride_col.to(tl.int64),
                mask=c_mask,
                other=-float("inf"),
            ).to(tl.float32)
            is_above = c_mask & (vals > s_min)
            is_tie = c_mask & (vals == s_min)
            above_pos = above_cursor + tl.cumsum(is_above.to(tl.int32), axis=0) - 1
            tie_pos = n_above + tie_cursor + tl.cumsum(is_tie.to(tl.int32), axis=0) - 1
            tl.store(
                out_row_ptr + above_pos,
                cols.to(tl.int32),
                mask=is_above & (above_pos < TOP_K),
            )
            tl.store(
                out_row_ptr + tie_pos,
                cols.to(tl.int32),
                mask=is_tie & (tie_pos < TOP_K),
            )
            above_cursor += tl.sum(is_above.to(tl.int32), axis=0)
            tie_cursor += tl.sum(is_tie.to(tl.int32), axis=0)


def canonicalize_prefill_topk(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
    top_k: int,
) -> None:
    """Canonicalize tie-breaking in `topk_indices[:num_rows, :top_k]` in-place."""
    num_rows = int(logits.shape[0])
    if num_rows <= 0 or top_k <= 0:
        return
    if _HAS_TRITON and logits.is_cuda and topk_indices.is_cuda:
        block_k = triton.next_power_of_2(top_k)
        _canonicalize_topk_ties_kernel[(num_rows,)](
            logits,
            cu_seqlen_ks,
            cu_seqlen_ke,
            topk_indices,
            logits.stride(0),
            logits.stride(1),
            topk_indices.stride(0),
            TOP_K=top_k,
            BLOCK_K=block_k,
            BLOCK_N=1024,
            num_warps=4,
        )
        return
    canonicalize_prefill_topk_torch(
        logits, cu_seqlen_ks, cu_seqlen_ke, topk_indices, top_k
    )


def canonicalize_prefill_topk_torch(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
    top_k: int,
) -> None:
    """CPU/PyTorch reference implementation of canonical smallest-index tie-breaking."""
    num_rows = int(logits.shape[0])
    for r in range(num_rows):
        ks = int(cu_seqlen_ks[r].item())
        ke = int(cu_seqlen_ke[r].item())
        row_len = ke - ks
        if row_len <= 0:
            topk_indices[r, :top_k] = -1
            continue
        if row_len <= top_k:
            topk_indices[r, :row_len] = torch.arange(
                row_len, dtype=torch.int32, device=topk_indices.device
            )
            if row_len < top_k:
                topk_indices[r, row_len:top_k] = -1
            continue
        cur = topk_indices[r, :top_k].to(torch.int64)
        valid = (cur >= 0) & (cur < row_len)
        if not bool(valid.any()):
            continue
        row_vals = logits[r, ks:ke].float()
        s_min = float(row_vals[cur[valid]].min().item())
        if s_min != s_min:
            continue
        above_idx = torch.nonzero(row_vals > s_min, as_tuple=False).squeeze(-1)
        n_above = int(above_idx.numel())
        tie_all = torch.nonzero(row_vals == s_min, as_tuple=False).squeeze(-1)
        if n_above >= top_k or n_above + int(tie_all.numel()) <= top_k:
            continue
        need_ties = top_k - n_above
        tie_idx = tie_all[:need_ties]
        canon = torch.cat([above_idx, tie_idx], dim=0).to(torch.int32)
        topk_indices[r, : canon.numel()] = canon
