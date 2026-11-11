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

"""Fused Triton router for DeepSeek-V4.1-Flash (`sqrt_softplus` + bias Top-K)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def dsv41_sqrt_softplus_topk_kernel(
    logits_ptr,
    bias_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    logits_row_stride,
    weights_row_stride,
    ids_row_stride,
    routed_scaling_factor,
    HAS_BIAS: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute `scores = sqrt(softplus(logits))`, select Top-K on `scores + bias`,
    gather unbiased `scores`, optionally renormalize, and multiply by `routed_scaling_factor`.
    """
    row = tl.program_id(0).to(tl.int64)
    l_base = logits_ptr + row * logits_row_stride.to(tl.int64)
    w_base = topk_weights_ptr + row * weights_row_stride.to(tl.int64)
    i_base = topk_ids_ptr + row * ids_row_stride.to(tl.int64)

    offs = tl.arange(0, BLOCK_E)
    mask = offs < NUM_EXPERTS

    x = tl.load(l_base + offs, mask=mask, other=-float("inf")).to(tl.float32)
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    scores = tl.where(mask, tl.sqrt(sp), 0.0)

    if HAS_BIAS:
        b = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sel = tl.where(mask, scores + b, -float("inf"))
    else:
        sel = tl.where(mask, scores, -float("inf"))

    w_sum = 0.0
    for k in tl.static_range(0, TOP_K):
        best_idx = tl.argmax(sel, axis=0)
        best_unbiased = tl.sum(tl.where(offs == best_idx, scores, 0.0), axis=0)
        tl.store(i_base + k, best_idx.to(tl.int32))
        tl.store(w_base + k, best_unbiased)
        w_sum += best_unbiased
        sel = tl.where(offs == best_idx, -float("inf"), sel)

    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < TOP_K
    picked_w = tl.load(w_base + k_offs, mask=k_mask, other=0.0).to(tl.float32)
    if RENORMALIZE:
        denom = tl.maximum(w_sum, 1.17549435e-38)
        picked_w = picked_w / denom
    picked_w = picked_w * routed_scaling_factor
    tl.store(w_base + k_offs, picked_w, mask=k_mask)


def dsv41_sqrt_softplus_topk(
    router_logits: torch.Tensor,
    *,
    top_k: int,
    renormalize: bool,
    routed_scaling_factor: float,
    correction_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused `sqrt_softplus` Top-K routing for `router_logits` `[M, E]`."""
    assert router_logits.ndim == 2 and router_logits.is_cuda
    num_tokens, num_experts = router_logits.shape
    block_e = triton.next_power_of_2(num_experts)
    block_k = triton.next_power_of_2(top_k)

    topk_weights = torch.empty(
        (num_tokens, top_k), device=router_logits.device, dtype=torch.float32
    )
    topk_ids = torch.empty(
        (num_tokens, top_k), device=router_logits.device, dtype=torch.int32
    )

    if num_tokens > 0:
        has_bias = correction_bias is not None
        bias_arg = correction_bias if has_bias else router_logits
        dsv41_sqrt_softplus_topk_kernel[(num_tokens,)](
            router_logits,
            bias_arg,
            topk_weights,
            topk_ids,
            router_logits.stride(0),
            topk_weights.stride(0),
            topk_ids.stride(0),
            float(routed_scaling_factor),
            HAS_BIAS=has_bias,
            RENORMALIZE=renormalize,
            NUM_EXPERTS=num_experts,
            BLOCK_E=block_e,
            TOP_K=top_k,
            BLOCK_K=block_k,
            num_warps=4,
        )
    return topk_weights, topk_ids
