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

"""Phase 0 Step 0.2: Fused RMSNorm + MXFP8 group-32 quantize & b12x sm_120 MXFP8 GEMM."""

from __future__ import annotations

import json
from unittest.mock import patch

import torch
import torch.nn.functional as F

from zstep_worker.kernels_sm120.norm_quant import (
    rmsnorm_quantize_mxfp8_group32,
    rmsnorm_quantize_mxfp8_group32_kernel,
)


def _dequant_mxfp8(q: torch.Tensor, e8m0: torch.Tensor) -> torch.Tensor:
    m, h = q.shape
    scales = torch.exp2(e8m0.to(torch.float32) - 127.0)
    return (q.to(torch.float32).view(m, h // 32, 32) * scales.unsqueeze(-1)).reshape(m, h)


def _bench_cuda_graph_us(fn, warmup: int = 10, reps: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    g.replay()
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) * 1000.0) / reps


def run_step_02_suite() -> dict[str, object]:
    from flashinfer.experimental.b12x.gemm import mxfp8_linear

    torch.manual_seed(42)
    device = torch.device("cuda:0")
    hidden_size = 5120
    out_features = 2048  # TP8 local projection slice on DeepSeek-V4.1-Flash
    eps = 1e-6

    # 1. Numerical accuracy + zero-recompilation check for Triton RMSNorm+MXFP8
    norm_w = torch.randn(hidden_size, device=device, dtype=torch.bfloat16)
    x0 = torch.randn((6, hidden_size), device=device, dtype=torch.bfloat16)
    rmsnorm_quantize_mxfp8_group32(x0, norm_w, eps=eps)

    cos_sims: dict[int, float] = {}
    norm_latencies_us: dict[int, float] = {}
    with patch.object(
        rmsnorm_quantize_mxfp8_group32_kernel,
        "_do_compile",
        wraps=rmsnorm_quantize_mxfp8_group32_kernel._do_compile,
    ) as compile_spy:
        for m in (6, 17, 48, 192, 4096):
            x = torch.randn((m, hidden_size), device=device, dtype=torch.bfloat16)
            q, s = rmsnorm_quantize_mxfp8_group32(x, norm_w, eps=eps)
            deq = _dequant_mxfp8(q, s)
            xf = x.float()
            ref = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps) * norm_w.float()
            cos = float(F.cosine_similarity(deq.flatten(), ref.flatten(), dim=0).item())
            assert cos > 0.998, f"Cosine similarity too low for M={m}: {cos}"
            cos_sims[m] = round(cos, 6)
            norm_latencies_us[m] = round(
                _bench_cuda_graph_us(
                    lambda x=x: rmsnorm_quantize_mxfp8_group32(x, norm_w, eps=eps)
                ),
                2,
            )
        assert compile_spy.call_count == 0, (
            f"Unexpected Triton JIT recompilation across M: {compile_spy.call_count}"
        )

    # 2. Int64 >2^31 element offset safety check (touching only last 2 rows of 4.6 GiB buffer)
    large_rows = (2_300_000_000 // hidden_size) + 4
    big_x = torch.empty((large_rows, hidden_size), device=device, dtype=torch.bfloat16)
    big_x[-2:].normal_()
    tail_view = big_x[-2:]
    strided_2rows = big_x[:: large_rows - 1]
    big_x[0].copy_(big_x[-2])
    q_big, s_big = rmsnorm_quantize_mxfp8_group32(strided_2rows, norm_w, eps=eps)
    q_ref, s_ref = rmsnorm_quantize_mxfp8_group32(tail_view[1:], norm_w, eps=eps)
    assert torch.equal(q_big[1].view(torch.uint8), q_ref[0].view(torch.uint8))
    assert torch.equal(s_big[1], s_ref[0])
    del big_x

    # 3. b12x sm_120 native MXFP8 Linear accuracy + CUDA Graph latency benchmark
    assert mxfp8_linear.is_supported(device), "b12x mxfp8_linear must support sm_120"
    w_bf16 = torch.randn((out_features, hidden_size), device=device, dtype=torch.bfloat16) * 0.02
    w_grouped = w_bf16.float().view(out_features, hidden_size // 32, 32)
    w_amax = w_grouped.abs().amax(dim=-1).clamp(min=1e-30)
    w_exp = torch.ceil(torch.log2(w_amax / 448.0)).clamp(-127.0, 127.0)
    w_scale_e8m0 = (w_exp.to(torch.int32) + 127).to(torch.uint8)
    w_fp8 = (
        (w_grouped / torch.exp2(w_exp).unsqueeze(-1))
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .reshape(out_features, hidden_size)
    )
    w_deq_ref = _dequant_mxfp8(w_fp8, w_scale_e8m0).to(torch.bfloat16)

    packed_w = mxfp8_linear.pack_weight(w_fp8, w_scale_e8m0)

    gemm_results: dict[int, dict[str, float]] = {}
    for m in (6, 48, 192, 4096):
        lhs = torch.randn((m, hidden_size), device=device, dtype=torch.bfloat16)
        out = mxfp8_linear.mm(lhs, packed_w)
        ref_out = lhs @ w_deq_ref.T
        cos = float(F.cosine_similarity(out.float().flatten(), ref_out.float().flatten(), dim=0))
        assert cos > 0.998, f"b12x mxfp8_linear cos_sim too low at M={m}: {cos}"

        lat_us = _bench_cuda_graph_us(lambda lhs=lhs: mxfp8_linear.mm(lhs, packed_w))
        flops = 2.0 * m * hidden_size * out_features
        tflops = (flops / (lat_us * 1e-6)) / 1e12
        bytes_moved = (
            out_features * hidden_size * (1.0 + 1.0 / 32.0)
            + m * hidden_size * 2.0
            + m * out_features * 2.0
        )
        gbps = (bytes_moved / (lat_us * 1e-6)) / 1e9
        gemm_results[m] = {
            "cos_sim": round(cos, 6),
            "latency_us": round(lat_us, 2),
            "tflops": round(tflops, 2),
            "effective_gbps": round(gbps, 1),
        }

    return {
        "rmsnorm_mxfp8_cos_sim": cos_sims,
        "rmsnorm_mxfp8_latency_us": norm_latencies_us,
        "int64_large_offset_passed": True,
        "zero_jit_recompilation_passed": True,
        "b12x_mxfp8_linear_5120x2048": gemm_results,
    }


if __name__ == "__main__":
    res = run_step_02_suite()
    print(json.dumps(res, indent=2))
