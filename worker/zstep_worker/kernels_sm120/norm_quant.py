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

"""Fused RMSNorm + MXFP8 (group-32 E8M0) quantization kernel for sm_120."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_quantize_mxfp8_group32_kernel(
    x_ptr,
    weight_ptr,
    q_out_ptr,
    scale_out_ptr,
    x_row_stride,
    q_row_stride,
    scale_row_stride,
    eps,
    HIDDEN_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    """Fuse RMSNorm(x, w) + MXFP8 (E4M3 values + E8M0 power-of-two scale per 32)."""
    row = tl.program_id(0).to(tl.int64)
    x_base = x_ptr + row * x_row_stride.to(tl.int64)
    q_base = q_out_ptr + row * q_row_stride.to(tl.int64)
    s_base = scale_out_ptr + row * scale_row_stride.to(tl.int64)

    group_offs = tl.arange(0, BLOCK_GROUPS)
    lane_offs = tl.arange(0, 32)
    col_offs = group_offs[:, None] * 32 + lane_offs[None, :]
    group_mask = group_offs < NUM_GROUPS
    elem_mask = group_mask[:, None]

    x = tl.load(x_base + col_offs, mask=elem_mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + col_offs, mask=elem_mask, other=0.0).to(tl.float32)

    sq_sum = tl.sum(tl.sum(x * x, axis=1), axis=0)
    rstd = tl.rsqrt(sq_sum / HIDDEN_SIZE + eps)
    normed = x * rstd * w

    amax = tl.max(tl.abs(normed), axis=1)
    raw_scale = tl.maximum(amax / 448.0, 1.0e-30)
    exp_unbiased = tl.clamp(tl.ceil(tl.log2(raw_scale)), -127.0, 127.0)
    scale_f32 = tl.exp2(exp_unbiased)
    e8m0_biased = (exp_unbiased.to(tl.int32) + 127).to(tl.uint8)

    scaled_vals = tl.clamp(normed / scale_f32[:, None], -448.0, 448.0)
    q_fp8 = scaled_vals.to(tl.float8e4nv)

    tl.store(q_base + col_offs, q_fp8, mask=elem_mask)
    tl.store(s_base + group_offs, e8m0_biased, mask=group_mask)


def rmsnorm_quantize_mxfp8_group32(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused RMSNorm + MXFP8 group-32 quantization on `x` `[M, H]`."""
    assert x.ndim == 2 and x.is_cuda, "x must be a 2D CUDA tensor"
    num_tokens, hidden_size = x.shape
    assert hidden_size % 32 == 0, "hidden_size must be divisible by 32"
    num_groups = hidden_size // 32
    block_groups = triton.next_power_of_2(num_groups)

    q_out = torch.empty((num_tokens, hidden_size), device=x.device, dtype=torch.float8_e4m3fn)
    scale_out = torch.empty((num_tokens, num_groups), device=x.device, dtype=torch.uint8)

    if num_tokens > 0:
        rmsnorm_quantize_mxfp8_group32_kernel[(num_tokens,)](
            x,
            weight,
            q_out,
            scale_out,
            x.stride(0),
            q_out.stride(0),
            scale_out.stride(0),
            eps,
            HIDDEN_SIZE=hidden_size,
            NUM_GROUPS=num_groups,
            BLOCK_GROUPS=block_groups,
            num_warps=8,
        )
    return q_out, scale_out
