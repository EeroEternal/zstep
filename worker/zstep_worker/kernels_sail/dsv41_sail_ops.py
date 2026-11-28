# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT

"""DeepSeek-V4.1-Flash compute kernels for T-Head PPU-ZW810E (`SAIL 1.7` + `Triton-for-SAIL`).

Provides:
1. HGGC C++/CUDA low-latency decode/verify kernels (`rmsnorm_bf16`, `mhc_*`, `rope_inplace`,
   `inplace_act_quant`, `inplace_fp4_quant`, `dsv41_router_topk`, `mxfp4_moe_gemv_*`,
   `engram_lookup_dequant`, `sparse_attn_tp8_hggc`).
2. `triton-for-sail` AIU Tensor-Core kernels for prefill/batched execution:
   - `_mxfp4_moe_grouped_gate_up_kernel` & `_mxfp4_moe_grouped_down_kernel` (branchless integer
     bitshift E2M1+UE8M0 -> IEEE BF16 unpacking directly into `tl.dot` on `810E` AIUs!).
   - `_triton_sparse_attn_tp8_kernel` (paged sparse MQA `h_local=8->16`, `d=512`, `attn_sink`).
3. `flash_mla.flash_mla_sparse_fwd` wrapper with analytical `log2`-LSE `attn_sink` rescaling.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from zstep_worker.kernels_sail.ppu_collectives import get_sail_ops_extension

_HAS_FLASH_MLA: bool | None = None


def is_flash_mla_available() -> bool:
    global _HAS_FLASH_MLA
    if _HAS_FLASH_MLA is None:
        try:
            import vllm  # noqa: F401
            from flash_mla import flash_mla_sparse_fwd  # noqa: F401

            _HAS_FLASH_MLA = True
        except BaseException:
            _HAS_FLASH_MLA = False
    return _HAS_FLASH_MLA


# ---------------------------------------------------------------------------
# Triton-for-SAIL Branchless E2M1 + UE8M0 -> IEEE BF16 Helper & Grouped MoE
# ---------------------------------------------------------------------------


@triton.jit
def _decode_e2m1_ue8m0_to_bf16(nibble_i32, scale_u8_i32):
    """Exact branchless FP32 decode of 4-bit E2M1 + 8-bit UE8M0 to BF16 (no int16 bitcast)."""
    mag = nibble_i32 & 7
    exp_part = tl.exp2((scale_u8_i32 - 127 + (mag >> 1) - 1).to(tl.float32))
    mant_mul = tl.where((mag > 1) & ((mag & 1) != 0), 1.5, 1.0)
    val = tl.where((mag == 0) | (scale_u8_i32 == 0), 0.0, exp_part * mant_mul)
    val = tl.where((nibble_i32 & 8) != 0, -val, val)
    return val.to(tl.bfloat16)


@triton.jit
def _mxfp4_moe_grouped_gate_up_kernel(
    x_ptr,
    w13_packed_ptr,
    w13_scale_ptr,
    sorted_pair_ids_ptr,
    expert_ids_per_tile_ptr,
    num_valid_tiles_ptr,
    topk_weights_ptr,
    out_inter_ptr,
    top_k: tl.constexpr,
    hidden_dim: tl.constexpr,
    inter_per_rank: tl.constexpr,
    swiglu_limit: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile_id = tl.program_id(0)
    n_tile = tl.program_id(1)
    num_valid = tl.load(num_valid_tiles_ptr)
    if tile_id >= num_valid:
        return

    expert_id = tl.load(expert_ids_per_tile_ptr + tile_id)
    offs_m = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
    pair_ids = tl.load(sorted_pair_ids_ptr + offs_m)
    valid_m = pair_ids >= 0
    safe_pair = tl.where(valid_m, pair_ids, 0)
    token_ids = safe_pair // top_k

    offs_n = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_n = offs_n < inter_per_rank
    safe_n = tl.where(valid_n, offs_n, 0)

    expert_rows = 2 * inter_per_rank
    row_gate = expert_id.to(tl.int64) * expert_rows + safe_n.to(tl.int64)
    row_up = row_gate + inter_per_rank
    packed_k = hidden_dim // 2
    scale_k = hidden_dim // 32

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k32 = tl.arange(0, 32)
    for k_base in range(0, hidden_dim, 64):
        x_base = x_ptr + token_ids[:, None].to(tl.int64) * hidden_dim + k_base
        a_even = tl.load(
            x_base + (2 * offs_k32)[None, :],
            mask=valid_m[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        a_odd = tl.load(
            x_base + (2 * offs_k32 + 1)[None, :],
            mask=valid_m[:, None],
            other=0.0,
        ).to(tl.bfloat16)

        pk_off = (k_base // 2) + offs_k32
        sc_off = (k_base // 32) + (offs_k32 // 16)

        bg_u8 = tl.load(
            w13_packed_ptr + row_gate[:, None] * packed_k + pk_off[None, :],
            mask=valid_n[:, None],
            other=0,
        ).to(tl.int32)
        sg_u8 = tl.load(
            w13_scale_ptr + row_gate[:, None] * scale_k + sc_off[None, :],
            mask=valid_n[:, None],
            other=0,
        ).to(tl.int32)
        bg_lo = _decode_e2m1_ue8m0_to_bf16(bg_u8 & 0xF, sg_u8)
        bg_hi = _decode_e2m1_ue8m0_to_bf16((bg_u8 >> 4) & 0xF, sg_u8)
        acc_gate = tl.dot(a_even, tl.trans(bg_lo), acc_gate)
        acc_gate = tl.dot(a_odd, tl.trans(bg_hi), acc_gate)

        bu_u8 = tl.load(
            w13_packed_ptr + row_up[:, None] * packed_k + pk_off[None, :],
            mask=valid_n[:, None],
            other=0,
        ).to(tl.int32)
        su_u8 = tl.load(
            w13_scale_ptr + row_up[:, None] * scale_k + sc_off[None, :],
            mask=valid_n[:, None],
            other=0,
        ).to(tl.int32)
        bu_lo = _decode_e2m1_ue8m0_to_bf16(bu_u8 & 0xF, su_u8)
        bu_hi = _decode_e2m1_ue8m0_to_bf16((bu_u8 >> 4) & 0xF, su_u8)
        acc_up = tl.dot(a_even, tl.trans(bu_lo), acc_up)
        acc_up = tl.dot(a_odd, tl.trans(bu_hi), acc_up)

    if swiglu_limit > 0.0:
        acc_up = tl.minimum(tl.maximum(acc_up, -swiglu_limit), swiglu_limit)
        acc_gate = tl.minimum(acc_gate, swiglu_limit)
    silu_gate = acc_gate * tl.sigmoid(acc_gate)
    route_w = tl.load(topk_weights_ptr + safe_pair, mask=valid_m, other=0.0).to(tl.float32)
    inter_val = (silu_gate * acc_up) * route_w[:, None]

    out_ptrs = (
        out_inter_ptr
        + safe_pair[:, None].to(tl.int64) * inter_per_rank
        + safe_n[None, :].to(tl.int64)
    )
    tl.store(out_ptrs, inter_val.to(tl.bfloat16), mask=valid_m[:, None] & valid_n[None, :])


@triton.jit
def _mxfp4_moe_grouped_down_kernel(
    inter_ptr,
    w2_packed_ptr,
    w2_scale_ptr,
    sorted_pair_ids_ptr,
    expert_ids_per_tile_ptr,
    num_valid_tiles_ptr,
    pair_out_ptr,
    hidden_dim: tl.constexpr,
    inter_per_rank: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile_id = tl.program_id(0)
    n_tile = tl.program_id(1)
    num_valid = tl.load(num_valid_tiles_ptr)
    if tile_id >= num_valid:
        return

    expert_id = tl.load(expert_ids_per_tile_ptr + tile_id)
    offs_m = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
    pair_ids = tl.load(sorted_pair_ids_ptr + offs_m)
    valid_m = pair_ids >= 0
    safe_pair = tl.where(valid_m, pair_ids, 0)

    offs_n = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_n = offs_n < hidden_dim
    safe_n = tl.where(valid_n, offs_n, 0)

    w_row = expert_id.to(tl.int64) * hidden_dim + safe_n.to(tl.int64)
    packed_k = inter_per_rank // 2
    scale_k = inter_per_rank // 32

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k32 = tl.arange(0, 32)
    for k_base in range(0, inter_per_rank, 64):
        valid_k = (k_base + 2 * offs_k32) < inter_per_rank
        a_base = inter_ptr + safe_pair[:, None].to(tl.int64) * inter_per_rank + k_base
        a_even = tl.load(
            a_base + (2 * offs_k32)[None, :],
            mask=valid_m[:, None] & valid_k[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        a_odd = tl.load(
            a_base + (2 * offs_k32 + 1)[None, :],
            mask=valid_m[:, None] & valid_k[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        pk_off = (k_base // 2) + offs_k32
        sc_off = (k_base // 32) + (offs_k32 // 16)
        b_u8 = tl.load(
            w2_packed_ptr + w_row[:, None] * packed_k + pk_off[None, :],
            mask=valid_n[:, None] & valid_k[None, :],
            other=0,
        ).to(tl.int32)
        s_u8 = tl.load(
            w2_scale_ptr + w_row[:, None] * scale_k + sc_off[None, :],
            mask=valid_n[:, None] & valid_k[None, :],
            other=0,
        ).to(tl.int32)
        b_lo = _decode_e2m1_ue8m0_to_bf16(b_u8 & 0xF, s_u8)
        b_hi = _decode_e2m1_ue8m0_to_bf16((b_u8 >> 4) & 0xF, s_u8)
        acc = tl.dot(a_even, tl.trans(b_lo), acc)
        acc = tl.dot(a_odd, tl.trans(b_hi), acc)

    out_ptrs = (
        pair_out_ptr
        + safe_pair[:, None].to(tl.int64) * hidden_dim
        + safe_n[None, :].to(tl.int64)
    )
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=valid_m[:, None] & valid_n[None, :])


@triton.jit
def _moe_combine_pairs_and_shared_kernel(
    pair_out_ptr,
    shared_out_ptr,
    out_ptr,
    top_k: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    d_tile = tl.program_id(1)
    offs_d = d_tile * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < hidden_dim

    acc = tl.load(
        shared_out_ptr + row.to(tl.int64) * hidden_dim + offs_d,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)
    for k in range(top_k):
        p_idx = row * top_k + k
        acc += tl.load(
            pair_out_ptr + p_idx.to(tl.int64) * hidden_dim + offs_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
    tl.store(
        out_ptr + row.to(tl.int64) * hidden_dim + offs_d,
        acc.to(tl.bfloat16),
        mask=mask_d,
    )


# ---------------------------------------------------------------------------
# Triton-for-SAIL Paged Sparse MQA (`h_local=8->16`, `d=512`, Fused `attn_sink`)
# ---------------------------------------------------------------------------


@triton.jit
def _triton_sparse_attn_tp8_kernel(
    q_ptr,
    kv_arena_ptr,
    attn_sink_ptr,
    topk_idxs_ptr,
    out_ptr,
    max_topk,
    sm_scale,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs_h16 = tl.arange(0, 16)
    valid_h = offs_h16 < 8
    safe_h = tl.where(valid_h, offs_h16, 0)
    offs_d = tl.arange(0, 512)

    q_ptrs = q_ptr + (row.to(tl.int64) * 8 + safe_h[:, None]) * 512 + offs_d[None, :]
    q_tile = tl.load(q_ptrs, mask=valid_h[:, None], other=0.0).to(tl.bfloat16)
    sink = tl.load(attn_sink_ptr + safe_h, mask=valid_h, other=-1.0e30).to(tl.float32)

    m_i = tl.full((16,), -1.0e30, dtype=tl.float32)
    l_i = tl.zeros((16,), dtype=tl.float32)
    acc_o = tl.zeros((16, 512), dtype=tl.float32)

    offs_t = tl.arange(0, BLOCK_K)
    for t_base in range(0, max_topk, BLOCK_K):
        t_idx = t_base + offs_t
        in_range = t_idx < max_topk
        slots = tl.load(
            topk_idxs_ptr + row.to(tl.int64) * max_topk + t_idx,
            mask=in_range,
            other=-1,
        )
        valid_t = in_range & (slots >= 0)
        safe_slots = tl.where(valid_t, slots, 0).to(tl.int64)

        kv_tile = tl.load(
            kv_arena_ptr + safe_slots[:, None] * 512 + offs_d[None, :],
            mask=valid_t[:, None],
            other=0.0,
        ).to(tl.bfloat16)

        scores = tl.dot(q_tile, tl.trans(kv_tile)) * sm_scale
        scores = tl.where(valid_t[None, :], scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, block_max)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc_o = acc_o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv_tile)
        m_i = m_new

    denom = l_i + tl.exp(sink - m_i)
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    out_tile = (acc_o * inv_denom[:, None]).to(tl.bfloat16)
    out_ptrs = out_ptr + (row.to(tl.int64) * 8 + safe_h[:, None]) * 512 + offs_d[None, :]
    tl.store(out_ptrs, out_tile, mask=valid_h[:, None])


def watched_sail_triton_kernels() -> list[tuple[str, object]]:
    return [
        ("sail_moe_gate_up", _mxfp4_moe_grouped_gate_up_kernel),
        ("sail_moe_down", _mxfp4_moe_grouped_down_kernel),
        ("sail_moe_combine", _moe_combine_pairs_and_shared_kernel),
        ("sail_sparse_attn_tp8", _triton_sparse_attn_tp8_kernel),
    ]


# ---------------------------------------------------------------------------
# High-Level Operator Wrappers (Zero `.item()` / `.cpu()` / Host Sync)
# ---------------------------------------------------------------------------


def rmsnorm_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1.0e-20,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    ext = get_sail_ops_extension()
    y = torch.empty_like(x) if out is None else out
    stream_handle = torch.cuda.current_stream(x.device).cuda_stream
    ext.launch_rmsnorm_bf16(x, weight, y, float(eps), stream_handle)
    return y


def act_quant_ue8m0(
    src: torch.Tensor, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    ext = get_sail_ops_extension()
    dst = src if out is None else out
    stream_handle = torch.cuda.current_stream(src.device).cuda_stream
    ext.launch_act_quant_ue8m0_bf16(src, dst, stream_handle)
    return dst


def inplace_act_quant_ue8m0(x: torch.Tensor) -> torch.Tensor:
    return act_quant_ue8m0(x, out=x)


def inplace_fp4_quant(
    x: torch.Tensor,
    *,
    block_size: int = 32,
    is_e4m3_scale: bool = False,
) -> torch.Tensor:
    ext = get_sail_ops_extension()
    stream_handle = torch.cuda.current_stream(x.device).cuda_stream
    ext.launch_inplace_fp4_quant_bf16(
        x, block_size, 1 if is_e4m3_scale else 0, stream_handle
    )
    return x


def apply_rope_inplace(
    x: torch.Tensor,
    cos_sin_table: torch.Tensor,
    positions_i32: torch.Tensor,
    *,
    rope_dim: int = 64,
    inverse: bool = False,
) -> torch.Tensor:
    ext = get_sail_ops_extension()
    stream_handle = torch.cuda.current_stream(x.device).cuda_stream
    ext.launch_rope_inplace_bf16(
        x,
        cos_sin_table,
        positions_i32,
        rope_dim,
        1 if inverse else 0,
        stream_handle,
    )
    return x


def mhc_mixes_sinkhorn(
    x: torch.Tensor,
    hc_fn_f32: torch.Tensor,
    hc_fn_bf16: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    pre_out: torch.Tensor,
    post_out: torch.Tensor,
    comb_out: torch.Tensor,
    raw_proj_buf: torch.Tensor | None = None,
    split_k_part_buf: torch.Tensor | None = None,
    split_k: int = 1,
    fused_max_rows: int = 16,
    norm_eps: float = 1.0e-20,
    hc_eps: float = 1.0e-6,
    sinkhorn_iters: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = x.shape[0]
    ext = get_sail_ops_extension()
    stream_handle = torch.cuda.current_stream(x.device).cuda_stream
    if rows <= fused_max_rows:
        eff_sk = split_k
        while eff_sk > 1 and rows * eff_sk > 256:
            eff_sk //= 2
        if (
            eff_sk > 1
            and split_k_part_buf is not None
            and split_k_part_buf.numel() >= rows * eff_sk * 25
        ):
            ext.launch_mhc_fused_mixes_splitk_bf16(
                x,
                hc_fn_f32,
                hc_scale,
                hc_base,
                split_k_part_buf,
                pre_out,
                post_out,
                comb_out,
                eff_sk,
                float(norm_eps),
                float(hc_eps),
                sinkhorn_iters,
                stream_handle,
            )
        else:
            ext.launch_mhc_fused_mixes_bf16(
                x,
                hc_fn_f32,
                hc_scale,
                hc_base,
                pre_out,
                post_out,
                comb_out,
                float(norm_eps),
                float(hc_eps),
                sinkhorn_iters,
                stream_handle,
            )
    else:
        x_flat = x.view(rows, -1)
        if raw_proj_buf is not None and raw_proj_buf.shape[0] >= rows:
            proj = raw_proj_buf[:rows]
            # Chunked FP32 projection (`<= 512` rows = `<= 40 MB` temporary) for exact FP32 parity
            for lo in range(0, rows, 512):
                hi = min(rows, lo + 512)
                torch.matmul(x_flat[lo:hi].float(), hc_fn_f32.t(), out=proj[lo:hi])
        else:
            proj = torch.matmul(x_flat, hc_fn_bf16.t()).float()
        ext.launch_mhc_sinkhorn_from_proj(
            x,
            proj,
            hc_scale,
            hc_base,
            pre_out,
            post_out,
            comb_out,
            float(norm_eps),
            float(hc_eps),
            sinkhorn_iters,
            stream_handle,
        )
    return pre_out, post_out, comb_out


def mhc_pre_rmsnorm(
    x: torch.Tensor,
    pre_mix: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    normed_out: torch.Tensor,
    aux_out: torch.Tensor | None = None,
    norm_eps: float = 1.0e-20,
) -> torch.Tensor:
    ext = get_sail_ops_extension()
    stream_handle = torch.cuda.current_stream(x.device).cuda_stream
    dummy_aux = normed_out if aux_out is None else aux_out
    ext.launch_mhc_pre_rmsnorm_bf16(
        x,
        pre_mix,
        norm_weight,
        normed_out,
        dummy_aux,
        1 if aux_out is not None else 0,
        float(norm_eps),
        stream_handle,
    )
    return normed_out


def mhc_post(
    sub_out: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    ext = get_sail_ops_extension()
    stream_handle = torch.cuda.current_stream(sub_out.device).cuda_stream
    ext.launch_mhc_post_bf16(sub_out, residual, post_mix, comb_mix, out, stream_handle)
    return out


def dsv41_moe_tp8_forward(
    x: torch.Tensor,
    *,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    shared_w13_bf16: torch.Tensor,
    shared_w2_bf16: torch.Tensor,
    experts_w13_packed: torch.Tensor,
    experts_w13_scale: torch.Tensor,
    experts_w2_packed: torch.Tensor,
    experts_w2_scale: torch.Tensor,
    top_k: int,
    route_scale: float,
    swiglu_limit: float,
    router_logits_buf: torch.Tensor,
    topk_weights_buf: torch.Tensor,
    topk_ids_buf: torch.Tensor,
    shared_gu_buf: torch.Tensor,
    shared_out_buf: torch.Tensor,
    moe_inter_buf: torch.Tensor,
    sorted_pair_ids_buf: torch.Tensor,
    expert_ids_per_tile_buf: torch.Tensor,
    num_valid_tiles_buf: torch.Tensor,
    pair_out_buf: torch.Tensor,
    moe_out_buf: torch.Tensor,
    act_quant_x_buf: torch.Tensor | None = None,
    gemv_max_rows: int = 16,
    gemv_threads: int = 256,
    block_m: int = 32,
    num_warps: int = 4,
) -> torch.Tensor:
    """TP=8 routed MXFP4 + shared BF16 MoE forward (returns unreduced local BF16 sum)."""
    rows, hidden_dim = x.shape
    num_experts = gate_weight.shape[0]
    inter_per_rank = shared_w2_bf16.shape[1]
    ext = get_sail_ops_extension()
    stream_handle = torch.cuda.current_stream(x.device).cuda_stream

    # 1. Router logits (FP32 exact match to `linear(x.float(), gate.weight.float())`) + Top-K
    r_logits = router_logits_buf.view(-1)[: rows * num_experts].view(rows, num_experts)
    torch.matmul(x.float(), gate_weight.t(), out=r_logits)
    t_weights = topk_weights_buf.view(-1)[: rows * top_k].view(rows, top_k)
    t_ids = topk_ids_buf.view(-1)[: rows * top_k].view(rows, top_k)
    ext.launch_dsv41_router_topk(
        r_logits, gate_bias, t_weights, t_ids, top_k, float(route_scale), stream_handle
    )

    if act_quant_x_buf is not None:
        x_in = act_quant_ue8m0(x, out=act_quant_x_buf[:rows])
    else:
        x_in = x

    # 2. Shared expert (`[rows, 5120] -> [rows, 2 * 288] -> fused clamped SwiGLU + UE8M0 -> [rows, 5120]`)
    sh_gu = shared_gu_buf[:rows]
    torch.matmul(x_in, shared_w13_bf16.t(), out=sh_gu)
    sh_act = moe_inter_buf.view(-1)[: rows * inter_per_rank].view(rows, inter_per_rank)
    ext.launch_shared_swiglu_quant_bf16(
        sh_gu,
        sh_act,
        float(swiglu_limit),
        1 if act_quant_x_buf is not None else 0,
        stream_handle,
    )
    sh_out = shared_out_buf[:rows]
    torch.matmul(sh_act, shared_w2_bf16.t(), out=sh_out)

    # 3. Routed MXFP4 experts: HGGC fused W4-LUT GEMV for decode/verify vs Triton AIU Tensor-Core GEMM for prefill
    out = moe_out_buf[:rows]
    num_pairs = rows * top_k
    inter = moe_inter_buf[:num_pairs]
    if rows <= gemv_max_rows:
        ext.launch_mxfp4_moe_gemv_gate_up(
            x_in,
            experts_w13_packed,
            experts_w13_scale,
            t_ids,
            t_weights,
            inter,
            float(swiglu_limit),
            gemv_threads,
            stream_handle,
        )
        if act_quant_x_buf is not None:
            inplace_act_quant_ue8m0(inter)
        ext.launch_mxfp4_moe_gemv_down_combine(
            inter,
            experts_w2_packed,
            experts_w2_scale,
            t_ids,
            sh_out,
            out,
            top_k,
            gemv_threads,
            stream_handle,
        )
    else:
        max_tiles = (num_pairs + block_m - 1) // block_m + num_experts
        ext.launch_moe_align_block_size(
            t_ids.view(-1),
            sorted_pair_ids_buf[: max_tiles * block_m],
            expert_ids_per_tile_buf[:max_tiles],
            num_valid_tiles_buf,
            num_experts,
            block_m,
            stream_handle,
        )
        block_n_gu = 64
        _mxfp4_moe_grouped_gate_up_kernel[
            (max_tiles, triton.cdiv(inter_per_rank, block_n_gu))
        ](
            x_in,
            experts_w13_packed,
            experts_w13_scale,
            sorted_pair_ids_buf,
            expert_ids_per_tile_buf,
            num_valid_tiles_buf,
            t_weights.view(-1),
            inter,
            top_k=top_k,
            hidden_dim=hidden_dim,
            inter_per_rank=inter_per_rank,
            swiglu_limit=float(swiglu_limit),
            BLOCK_M=block_m,
            BLOCK_N=block_n_gu,
            num_warps=num_warps,
        )
        if act_quant_x_buf is not None:
            inplace_act_quant_ue8m0(inter)
        block_n_dn = 64
        p_out = pair_out_buf[:num_pairs]
        _mxfp4_moe_grouped_down_kernel[
            (max_tiles, triton.cdiv(hidden_dim, block_n_dn))
        ](
            inter,
            experts_w2_packed,
            experts_w2_scale,
            sorted_pair_ids_buf,
            expert_ids_per_tile_buf,
            num_valid_tiles_buf,
            p_out,
            hidden_dim=hidden_dim,
            inter_per_rank=inter_per_rank,
            BLOCK_M=block_m,
            BLOCK_N=block_n_dn,
            num_warps=num_warps,
        )
        _moe_combine_pairs_and_shared_kernel[(rows, triton.cdiv(hidden_dim, 256))](
            p_out,
            sh_out,
            out,
            top_k=top_k,
            hidden_dim=hidden_dim,
            BLOCK_D=256,
            num_warps=4,
        )
    return out


def sparse_attn_tp8_forward(
    q: torch.Tensor,
    kv_arena: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    *,
    out: torch.Tensor,
    partial_acc: torch.Tensor,
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    q_pad64_buf: torch.Tensor | None = None,
    sm_scale: float,
    backend: str = "auto",
) -> torch.Tensor:
    """Execute paged sparse MQA (`q: [M, 8, 512]`, `kv_arena: [S_tot, 1, 512]`, `topk_idxs: [M, K]`)."""
    rows = q.shape[0]
    max_topk = topk_idxs.shape[1]
    eff_backend = backend
    if eff_backend == "auto":
        # HGGC Split-K MQA (`8` heads native) for decode/verify (`rows <= 32`);
        # FlashMLA AIU Tensor-Core (`topk % 128 == 0`) or Triton TP8 for large prefill (`rows > 32`).
        if rows <= 32:
            eff_backend = "hggc_tp8"
        elif is_flash_mla_available() and (max_topk % 128 == 0):
            eff_backend = "flash_mla"
        else:
            eff_backend = "triton_tp8"

    if (
        eff_backend == "flash_mla"
        and is_flash_mla_available()
        and (max_topk % 128 == 0)
        and not torch.cuda.is_current_stream_capturing()
    ):
        from flash_mla import flash_mla_sparse_fwd

        if q_pad64_buf is not None and q_pad64_buf.shape[0] >= rows:
            q_pad = q_pad64_buf[:rows]
            q_pad[:, :8, :].copy_(q)
            q_pad[:, 8:, :].zero_()
        else:
            q_pad = torch.zeros((rows, 64, 512), dtype=torch.bfloat16, device=q.device)
            q_pad[:, :8, :] = q
        out_pad, _max_logits, lse = flash_mla_sparse_fwd(
            q_pad, kv_arena, topk_idxs.unsqueeze(1), float(sm_scale), d_v=512
        )
        lse_nat = lse[:, :8].float() * math.log(2.0)
        scale_sink = torch.where(
            torch.isfinite(lse_nat),
            torch.sigmoid(lse_nat - attn_sink.view(1, 8)),
            torch.zeros_like(lse_nat),
        )
        out.copy_((out_pad[:, :8, :].float() * scale_sink.unsqueeze(-1)).to(torch.bfloat16))
        return out

    if eff_backend == "triton_tp8":
        _triton_sparse_attn_tp8_kernel[(rows,)](
            q,
            kv_arena,
            attn_sink,
            topk_idxs,
            out,
            max_topk,
            float(sm_scale),
            BLOCK_K=64,
            num_warps=4,
        )
        return out

    ext = get_sail_ops_extension()
    stream_handle = torch.cuda.current_stream(q.device).cuda_stream
    ext.launch_sparse_attn_tp8_bf16(
        q,
        kv_arena,
        attn_sink,
        topk_idxs,
        partial_acc,
        partial_m,
        partial_l,
        out,
        float(sm_scale),
        stream_handle,
    )
    return out
