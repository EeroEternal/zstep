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

"""Phase 2 sm_120 operator parity, zero-recompilation, and int64 offset tests for Qwen3.5."""

from __future__ import annotations

import ctypes
import mmap
import struct
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from zstep_worker.kernels_sm120 import qwen35_ops
from zstep_worker.kernels_sm120.shm_allreduce import get_host_mapped_allreduce_extension
from zstep_worker.models.qwen35_moe import _build_rope_caches

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _ref_zero_centered_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * (1.0 + w.float())).to(torch.bfloat16)


def test_zero_centered_rmsnorm_and_no_recompile() -> None:
    device = torch.device("cuda", 0)
    w = torch.randn(2048, device=device, dtype=torch.bfloat16) * 0.1
    x0 = torch.randn((4, 2048), device=device, dtype=torch.bfloat16)
    out0 = torch.empty_like(x0)
    qwen35_ops.zero_centered_rmsnorm(x0, w, out=out0, eps=1.0e-6)
    torch.testing.assert_close(out0, _ref_zero_centered_rmsnorm(x0, w, 1.0e-6), atol=1e-2, rtol=1e-2)

    with patch.object(
        qwen35_ops._zero_centered_rmsnorm_kernel,
        "_do_compile",
        wraps=qwen35_ops._zero_centered_rmsnorm_kernel._do_compile,
    ) as spy:
        for rows in (1, 13, 64, 257):
            x = torch.randn((rows, 2048), device=device, dtype=torch.bfloat16)
            out = torch.empty_like(x)
            qwen35_ops.zero_centered_rmsnorm(x, w, out=out, eps=1.0e-6)
            torch.testing.assert_close(
                out, _ref_zero_centered_rmsnorm(x, w, 1.0e-6), atol=1e-2, rtol=1e-2
            )
        assert spy.call_count == 0


def test_linear_attn_prefill_matches_decode_recurrence() -> None:
    device = torch.device("cuda", 0)
    torch.manual_seed(1234)
    seq_len = 19
    qkvzba = torch.randn((seq_len, 1544), device=device, dtype=torch.bfloat16) * 0.2
    conv_w = torch.randn((1024, 4), device=device, dtype=torch.bfloat16) * 0.2
    a_log = torch.log(torch.empty(4, device=device, dtype=torch.float32).uniform_(0.5, 4.0))
    dt_bias = torch.ones(4, device=device, dtype=torch.bfloat16) * 0.5
    norm_w = torch.ones(128, device=device, dtype=torch.float32)

    conv_prefill = torch.zeros((2, 1024, 3), device=device, dtype=torch.bfloat16)
    ssm_prefill = torch.zeros((2, 4, 128, 128), device=device, dtype=torch.float32)
    q_start = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    positions = torch.arange(seq_len, device=device, dtype=torch.int32)
    slots = torch.ones(seq_len, device=device, dtype=torch.int32)

    qkv_conv_p = torch.empty((seq_len, 1024), device=device, dtype=torch.bfloat16)
    raw_p = torch.empty((seq_len, 512), device=device, dtype=torch.bfloat16)
    gated_p = torch.empty((seq_len, 512), device=device, dtype=torch.bfloat16)

    qwen35_ops.linear_attn_prefill_varlen(
        qkvzba,
        conv_prefill,
        conv_w,
        a_log,
        dt_bias,
        ssm_prefill,
        norm_w,
        q_start,
        positions,
        slots,
        qkv_conv_buf=qkv_conv_p,
        raw_out_buf=raw_p,
        gated_out_buf=gated_p,
        num_seqs=1,
        eps=1.0e-6,
    )

    conv_dec = torch.zeros_like(conv_prefill)
    ssm_dec = torch.zeros_like(ssm_prefill)
    step_outs = []
    for t in range(seq_len):
        qc = torch.empty((1, 1024), device=device, dtype=torch.bfloat16)
        ro = torch.empty((1, 512), device=device, dtype=torch.bfloat16)
        go = torch.empty((1, 512), device=device, dtype=torch.bfloat16)
        qwen35_ops.linear_attn_decode_step(
            qkvzba[t : t + 1],
            conv_dec,
            conv_w,
            a_log,
            dt_bias,
            ssm_dec,
            norm_w,
            positions[t : t + 1],
            slots[t : t + 1],
            qkv_conv_buf=qc,
            raw_out_buf=ro,
            gated_out_buf=go,
            num_active_rows=1,
            eps=1.0e-6,
        )
        step_outs.append(go)
    stacked_dec = torch.cat(step_outs, dim=0)

    torch.testing.assert_close(conv_prefill, conv_dec, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ssm_prefill, ssm_dec, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(gated_p, stacked_dec, atol=2e-2, rtol=2e-2)


def _run_paged_gqa_parity_check(num_pages: int) -> None:
    device = torch.device("cuda", 0)
    torch.manual_seed(77)
    k_cache = torch.empty((num_pages, 16, 256), device=device, dtype=torch.bfloat16)
    v_cache = torch.empty((num_pages, 16, 256), device=device, dtype=torch.bfloat16)

    seq_len = 21
    far_pages = [num_pages - 2, num_pages - 1]
    page_table = torch.zeros((2, 4), device=device, dtype=torch.int32)
    page_table[1, 0] = far_pages[0]
    page_table[1, 1] = far_pages[1]

    cos_c, sin_c = _build_rope_caches(
        max_positions=128, rotary_dim=64, rope_theta=10_000_000.0, device=device
    )
    q_norm_w = torch.zeros(256, device=device, dtype=torch.bfloat16)
    k_norm_w = torch.zeros(256, device=device, dtype=torch.bfloat16)
    qkv = torch.randn((seq_len, 1536), device=device, dtype=torch.bfloat16) * 0.2
    positions = torch.arange(seq_len, device=device, dtype=torch.int32)
    slots = torch.ones(seq_len, device=device, dtype=torch.int32)
    q_out = torch.empty((seq_len, 512), device=device, dtype=torch.bfloat16)
    gate_out = torch.empty((seq_len, 512), device=device, dtype=torch.bfloat16)

    qwen35_ops.qwen35_attn_prologue(
        qkv,
        positions,
        slots,
        page_table,
        q_norm_w,
        k_norm_w,
        cos_c,
        sin_c,
        q_out=q_out,
        gate_out=gate_out,
        k_cache=k_cache,
        v_cache=v_cache,
        eps=1.0e-6,
    )

    q_start = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
    prefill_out = torch.empty((seq_len, 512), device=device, dtype=torch.bfloat16)
    qwen35_ops.paged_gqa_prefill_varlen_gated(
        q_out,
        gate_out,
        k_cache,
        v_cache,
        page_table,
        q_start,
        seq_lens,
        slots,
        out=prefill_out,
        num_seqs=1,
        max_q_len=seq_len,
        sm_scale=256**-0.5,
    )

    # Compare final query token of prefill against split-K decode at `seq_len`
    p_acc = torch.empty((1, 16, 512), device=device, dtype=torch.float32)
    p_m = torch.empty((1, 16, 2), device=device, dtype=torch.float32)
    p_l = torch.empty((1, 16, 2), device=device, dtype=torch.float32)
    dec_out = torch.empty((1, 512), device=device, dtype=torch.bfloat16)
    qwen35_ops.paged_gqa_decode_gated(
        q_out[-1:],
        gate_out[-1:],
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        slots[-1:],
        partial_acc=p_acc,
        partial_m=p_m,
        partial_l=p_l,
        out=dec_out,
        num_active_seqs=1,
        sm_scale=256**-0.5,
    )
    torch.testing.assert_close(prefill_out[-1:], dec_out, atol=1.5e-2, rtol=1.5e-2)


def test_paged_gqa_prefill_and_decode_parity_small() -> None:
    _run_paged_gqa_parity_check(num_pages=16)


def test_paged_gqa_large_page_int64_offset() -> None:
    # > 525,000 pages so `phys_page * 4096` exceeds 2^31 elements (~8.6 GB BF16 total)
    _run_paged_gqa_parity_check(num_pages=530_000)


def test_sparse_moe_tp8_matches_reference_formula() -> None:
    device = torch.device("cuda", 0)
    torch.manual_seed(99)
    rows = 11
    x = torch.randn((rows, 2048), device=device, dtype=torch.bfloat16) * 0.1
    router_w = torch.randn((256, 2048), device=device, dtype=torch.bfloat16) * 0.1
    sh_gate_w = torch.randn((1, 2048), device=device, dtype=torch.bfloat16) * 0.1
    sh_gate_pad = torch.zeros((8, 2048), device=device, dtype=torch.bfloat16)
    sh_gate_pad[0:1] = sh_gate_w
    sh_gp = torch.randn((64, 2048), device=device, dtype=torch.bfloat16) * 0.1
    sh_up = torch.randn((64, 2048), device=device, dtype=torch.bfloat16) * 0.1
    w_rs = torch.cat([router_w, sh_gate_pad, sh_gp, sh_up], dim=0).contiguous()
    w_sh_down = torch.randn((2048, 64), device=device, dtype=torch.bfloat16) * 0.1
    w_gu = torch.randn((256, 128, 2048), device=device, dtype=torch.bfloat16) * 0.1
    w_dn = torch.randn((256, 2048, 64), device=device, dtype=torch.bfloat16) * 0.1

    max_t = qwen35_ops.max_moe_tiles_for_rows(rows)
    out = qwen35_ops.sparse_moe_tp8_forward(
        x,
        w_rs,
        w_sh_down,
        w_gu,
        w_dn,
        router_shared_buf=torch.empty((rows, 392), device=device, dtype=torch.bfloat16),
        topk_ids_buf=torch.full(
            (rows * 8,), -999_999_999, device=device, dtype=torch.int32
        ),
        topk_weights_buf=torch.empty(rows * 8, device=device, dtype=torch.bfloat16),
        shared_gate_scale_buf=torch.empty(rows, device=device, dtype=torch.float32),
        shared_inter_buf=torch.empty((rows, 64), device=device, dtype=torch.bfloat16),
        shared_out_buf=torch.empty((rows, 2048), device=device, dtype=torch.bfloat16),
        sorted_pair_ids_buf=torch.empty(max_t * 16, device=device, dtype=torch.int32),
        expert_ids_per_tile_buf=torch.empty(max_t, device=device, dtype=torch.int32),
        num_valid_tiles_buf=torch.zeros(1, device=device, dtype=torch.int32),
        moe_inter_buf=torch.empty((rows * 8, 64), device=device, dtype=torch.bfloat16),
        pair_out_buf=torch.empty((rows * 8, 2048), device=device, dtype=torch.bfloat16),
        moe_out_buf=torch.empty((rows, 2048), device=device, dtype=torch.bfloat16),
        num_active_rows=rows,
    )

    # Exact PyTorch reference matching Qwen3_5MoeSparseMoeBlock
    probs = F.softmax(F.linear(x, router_w), dtype=torch.float32, dim=-1)
    topk_v, topk_i = torch.topk(probs, 8, dim=-1)
    topk_v = (topk_v / topk_v.sum(dim=-1, keepdim=True)).to(torch.bfloat16)
    ref = torch.zeros_like(x)
    for r in range(rows):
        acc = torch.zeros(2048, device=device, dtype=torch.float32)
        for k in range(8):
            eid = int(topk_i[r, k].item())
            g, u = F.linear(x[r : r + 1], w_gu[eid]).chunk(2, dim=-1)
            inter = (F.silu(g.float()) * u.float()).to(torch.bfloat16)
            dn = F.linear(inter, w_dn[eid])[0]
            acc += dn.float() * topk_v[r, k].float()
        sh_inter = (F.silu(F.linear(x[r : r + 1], sh_gp).float()) * F.linear(x[r : r + 1], sh_up).float()).to(
            torch.bfloat16
        )
        sh_o = F.linear(sh_inter, w_sh_down)[0]
        sh_scale = torch.sigmoid(F.linear(x[r : r + 1], sh_gate_w).float())[0, 0]
        ref[r] = (acc + sh_scale * sh_o.float()).to(torch.bfloat16)

    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_local_top2_bf16_and_shm_gather_top2_exact_with_ties() -> None:
    device = torch.device("cuda", 0)
    ext = qwen35_ops.get_qwen35_cuda_ext()
    ar_ext = get_host_mapped_allreduce_extension()
    stream = torch.cuda.current_stream(device).cuda_stream

    vocab_per_rank = 31_040
    rows = 5
    torch.manual_seed(20261112)
    logits = torch.randn((rows, vocab_per_rank), device=device, dtype=torch.bfloat16)
    # Row 0: 3-way tie at maximum BF16 value -> smallest two indices (17, 1024) must win
    logits[0].fill_(torch.tensor(-4.0, dtype=torch.bfloat16))
    for tie_col in (9000, 17, 1024):
        logits[0, tie_col] = torch.tensor(19.5, dtype=torch.bfloat16)
    # Row 1: distinct top-1 with 2-way tie for runner-up -> (30001, 42) must win
    logits[1].fill_(torch.tensor(-8.0, dtype=torch.bfloat16))
    logits[1, 30001] = torch.tensor(21.25, dtype=torch.bfloat16)
    logits[1, 555] = torch.tensor(21.125, dtype=torch.bfloat16)
    logits[1, 42] = torch.tensor(21.125, dtype=torch.bfloat16)

    out_vals = torch.empty((rows, 2), dtype=torch.float32, device=device)
    out_ids = torch.empty((rows, 2), dtype=torch.int32, device=device)
    ext.launch_local_top2_bf16(logits, out_vals, out_ids, rows, stream)
    torch.cuda.synchronize(device)

    vals_cpu = out_vals.cpu().tolist()
    ids_cpu = out_ids.cpu().tolist()
    assert ids_cpu[0] == [17, 1024]
    assert vals_cpu[0] == [19.5, 19.5]
    assert ids_cpu[1] == [30001, 42]
    assert vals_cpu[1] == [21.25, 21.125]
    for r in range(2, rows):
        row_list = logits[r].float().cpu().tolist()
        ordered = sorted(range(vocab_per_rank), key=lambda idx: (-row_list[idx], idx))
        assert ids_cpu[r] == [ordered[0], ordered[1]]
        assert vals_cpu[r] == [row_list[ordered[0]], row_list[ordered[1]]]

    # Verify host-mapped SHM ring Top-2 gather (`world_size=1` single-process check)
    max_rows_per_step = 16
    shm_bytes = (256 + 4 * 8 * (max_rows_per_step * 2) * 8 + 4095) & ~4095
    mapping = mmap.mmap(-1, shm_bytes, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    host_addr = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
    ctypes.memset(host_addr, 0, shm_bytes)
    dev_addr = ar_ext.register_mapped_host(host_addr, shm_bytes)
    try:
        out_tokens = torch.zeros((rows, 2), dtype=torch.int32, device=device)
        out_bits = torch.zeros((rows, 2), dtype=torch.int32, device=device)
        step_counter = torch.zeros(1, dtype=torch.int64, device=device)
        for _ in range(6):
            ext.launch_tp8_argmax_gather_shm(
                out_vals,
                out_ids,
                out_tokens,
                out_bits,
                step_counter,
                dev_addr,
                rows,
                max_rows_per_step,
                0,
                1,
                vocab_per_rank,
                stream,
            )
        torch.cuda.synchronize(device)
        assert out_tokens.cpu().tolist() == ids_cpu
        bits_cpu = out_bits.cpu().tolist()
        for r in range(rows):
            for k in range(2):
                unpacked = struct.unpack("<f", struct.pack("<I", bits_cpu[r][k] & 0xFFFF_FFFF))[0]
                assert unpacked == vals_cpu[r][k]
    finally:
        torch.cuda.synchronize(device)
        ar_ext.unregister_mapped_host(host_addr)
        mapping.close()


def test_tp8_argmax_rejects_rows_above_capacity() -> None:
    device = torch.device("cuda", 0)
    ext = qwen35_ops.get_qwen35_cuda_ext()
    local_top2_vals = torch.zeros((1, 2), dtype=torch.float32, device=device)
    local_top2_ids = torch.zeros((1, 2), dtype=torch.int32, device=device)
    out_tokens = torch.zeros((1, 2), dtype=torch.int32, device=device)
    out_logprob_bits = torch.zeros((1, 2), dtype=torch.int32, device=device)
    step_counter = torch.zeros(1, dtype=torch.int64, device=device)
    with pytest.raises(RuntimeError, match="num_rows exceeds max_rows_per_step"):
        ext.launch_tp8_argmax_gather_shm(
            local_top2_vals,
            local_top2_ids,
            out_tokens,
            out_logprob_bits,
            step_counter,
            0,
            513,
            512,
            0,
            8,
            32768,
            torch.cuda.current_stream(device).cuda_stream,
        )
