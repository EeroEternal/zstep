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

"""Zero-sync distributed TP GPU sampler (temperature, top_p, top_k, seed).

Invoked only when `STEP_FLAG_SAMPLING` is set in `StepHeader.flags`. Greedy
steps (`temperature == 0.0` for all sequences) bypass this module completely
so the Top-2 argmax fast path stays at zero overhead.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

# Signed int64 two's-complement equivalents of SplitMix64 constants:
# 0x9E37_79B9_7F4A_7C15, 0xBF58_476D_1CE4_E5B9, 0x94D0_49BB_1331_11EB
_SM64_GAMMA: int = -7046029254386353131
_SM64_MUL1: int = -4658895280553007687
_SM64_MUL2: int = -7723592293110705685
_MASK_RSHIFT_30: int = (1 << 34) - 1
_MASK_RSHIFT_27: int = (1 << 37) - 1
_MASK_RSHIFT_31: int = (1 << 33) - 1
_MASK_RSHIFT_41: int = (1 << 23) - 1
_INV_2POW23: float = 1.0 / 8388608.0


def splitmix64_uniform_f32(seeds_i64: torch.Tensor) -> torch.Tensor:
    """Map int64 bit-patterns of u64 seeds to deterministic float32 uniforms in (0, 1).

    Uses pure bitwise GPU ops with no CPU sync and no `torch.Generator` state,
    guaranteeing bit-for-bit identical draws across all TP ranks. Uses 23 bits
    so `(u23 + 0.5)` is exact in IEEE-754 float32 and strictly in `[2^-24, 1 - 2^-24]`.
    """
    x = seeds_i64.to(torch.int64) + _SM64_GAMMA
    x = (x ^ ((x >> 30) & _MASK_RSHIFT_30)) * _SM64_MUL1
    x = (x ^ ((x >> 27) & _MASK_RSHIFT_27)) * _SM64_MUL2
    z = x ^ ((x >> 31) & _MASK_RSHIFT_31)
    u23 = ((z >> 41) & _MASK_RSHIFT_41).to(torch.float32)
    return (u23 + 0.5) * _INV_2POW23


def sample_tp8_logits(
    local_logits: torch.Tensor,
    *,
    temperatures: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    rng_seeds: torch.Tensor,
    tp_group: Any | None = None,
    world_size: int = 1,
    vocab_size: int | None = None,
) -> torch.Tensor:
    """Sample global token IDs from TP-sharded `[M, V_local]` logits on GPU.

    Supports mixed greedy (`temperature == 0.0`) and stochastic (`temperature > 0.0`)
    rows within the same batch with zero host-device synchronization.
    """
    num_rows, local_v = local_logits.shape
    if world_size > 1:
        gathered = torch.empty(
            (world_size, num_rows, local_v),
            dtype=local_logits.dtype,
            device=local_logits.device,
        )
        dist.all_gather_into_tensor(gathered, local_logits.contiguous(), group=tp_group)
        full_logits = gathered.permute(1, 0, 2).reshape(num_rows, world_size * local_v)
    else:
        full_logits = local_logits

    if vocab_size is not None and vocab_size < full_logits.shape[1]:
        full_logits = full_logits[:, :vocab_size]

    vocab_dim = full_logits.shape[1]
    logits_f32 = full_logits.float()
    greedy_tokens = torch.argmax(logits_f32, dim=-1)

    safe_temp = torch.clamp(temperatures.float(), min=1e-6).unsqueeze(-1)
    scaled = logits_f32 / safe_temp

    sorted_logits, sorted_indices = torch.sort(scaled, dim=-1, descending=True)
    rank_idx = torch.arange(vocab_dim, device=logits_f32.device, dtype=torch.int32).unsqueeze(0)

    eff_top_k = torch.where(
        top_k > 0,
        top_k.to(torch.int32),
        torch.full_like(top_k, vocab_dim, dtype=torch.int32),
    )
    keep_k = rank_idx < eff_top_k.unsqueeze(-1)
    masked_logits = sorted_logits.masked_fill(~keep_k, float("-inf"))

    probs = torch.softmax(masked_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)
    prior_cumprobs = cumprobs - probs
    keep_p = (prior_cumprobs < top_p.float().unsqueeze(-1)) | (rank_idx == 0)
    filtered_probs = torch.where(keep_p, probs, torch.zeros_like(probs))

    # Search directly on unnormalized `raw_cdf` with `u * raw_cdf[:, -1:]` and `right=False`:
    # since `raw_cdf[:, k_last] == raw_cdf[:, -1] >= threshold`, searchsorted is mathematically
    # guaranteed to land in `0..=k_last` and never select a zero-probability masked-out token.
    raw_cdf = torch.cumsum(filtered_probs, dim=-1)
    total_prob = raw_cdf[:, -1:]
    u = splitmix64_uniform_f32(rng_seeds).unsqueeze(-1)
    threshold = u * total_prob
    picked_rank = (
        torch.searchsorted(raw_cdf, threshold, right=False)
        .squeeze(-1)
        .clamp(max=vocab_dim - 1)
    )
    sampled_tokens = torch.gather(
        sorted_indices,
        dim=-1,
        index=picked_rank.unsqueeze(-1),
    ).squeeze(-1)

    total_prob_flat = total_prob.squeeze(-1)
    valid_stochastic = (
        (temperatures > 0.0) & torch.isfinite(total_prob_flat) & (total_prob_flat > 0.0)
    )
    final_tokens = torch.where(valid_stochastic, sampled_tokens, greedy_tokens)
    return final_tokens.to(torch.int32)
