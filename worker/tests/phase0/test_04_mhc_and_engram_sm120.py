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

"""Phase 0 Step 0.5a: mHC and Engram VRAM/pinned-UVA lookup on sm_120."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

HIDDEN_SIZE = 5120
MHC_STREAMS = 4
SINKHORN_ITERS = 20
ENGRAM_HEADS = 24
ENGRAM_HEAD_DIM = 256
ENGRAM_GROUPS = 8
ENGRAM_OUT_DIM = ENGRAM_HEADS * ENGRAM_HEAD_DIM
TOKEN_COUNTS = (6, 48, 192, 4096)


def _bench_cuda_graph_us(fn, *, warmup: int = 5, reps: int = 30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / reps


def _mhc_pre_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = residual.flatten(1).float()
    mixes = F.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + bias[:4]) + hc_eps
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + bias[4:8])
    comb = mixes[:, 8:].view(-1, 4, 4) * scale[2] + bias[8:].view(4, 4)
    comb = torch.softmax(comb, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(SINKHORN_ITERS - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    y = (pre.unsqueeze(-1) * residual.float()).sum(dim=1).to(torch.bfloat16)
    return y, post, comb


def _mhc_post_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    return (
        post.unsqueeze(-1) * x.unsqueeze(1).float()
        + (comb.unsqueeze(-1) * residual.unsqueeze(2).float()).sum(dim=1)
    ).to(torch.bfloat16)


def _cosine(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            actual.float().flatten(), reference.float().flatten(), dim=0
        ).item()
    )


def _run_mhc_case(*, tokens: int, device: torch.device) -> dict[str, float | int]:
    from flashinfer.experimental.b12x.norm import mhc
    from flashinfer.experimental.b12x.norm.mhc import _impl
    from flashinfer.experimental.b12x.preparation import (
        FrozenMapping,
        PreparedCall,
        prepare_default,
        require_prepared,
    )

    torch.manual_seed(6100 + tokens)
    rms_eps = 1e-5
    hc_eps = 1e-6
    x = torch.randn((tokens, HIDDEN_SIZE), device=device, dtype=torch.bfloat16) * 0.1
    residual = (
        torch.randn(
            (tokens, MHC_STREAMS, HIDDEN_SIZE), device=device, dtype=torch.bfloat16
        )
        * 0.1
    )
    fn = torch.randn((24, MHC_STREAMS * HIDDEN_SIZE), device=device) / 64
    hc_scale = torch.ones(3, device=device, dtype=torch.float32)
    hc_base = torch.randn(24, device=device, dtype=torch.float32) * 0.1
    prev_post = torch.rand((tokens, MHC_STREAMS), device=device, dtype=torch.float32)
    prev_comb = torch.softmax(
        torch.randn((tokens, MHC_STREAMS, MHC_STREAMS), device=device), dim=-1
    )
    caps = mhc.Caps(
        device=device,
        max_tokens=tokens,
        hidden_size=HIDDEN_SIZE,
        dtype=torch.bfloat16,
        split_k=MHC_STREAMS * HIDDEN_SIZE // mhc.DEFAULT_BLOCK_K,
    )
    invocation = FrozenMapping(
        {
            "operation": "post_pre",
            "has_norm_weight": False,
            "norm_weight_dtype": "bfloat16",
            "has_fn_bf16": False,
            "collapse_weighted": False,
            "lagged_mix": False,
            "expanded_residual": False,
            "bf16x2_eligible": True,
            "output_mode": "provided",
            "rms_eps": rms_eps,
            "hc_eps": hc_eps,
            "sinkhorn_iters": SINKHORN_ITERS,
            "norm_eps": 0.0,
            "block_k": mhc.DEFAULT_BLOCK_K,
            "block_h": mhc.DEFAULT_BLOCK_H,
        }
    )
    declaration = mhc.plan(caps, invocation=invocation, override=None)

    def allocate_outputs():
        return {
            "y": torch.empty(
                (tokens, HIDDEN_SIZE), device=device, dtype=torch.bfloat16
            ),
            "post": torch.empty(
                (tokens, MHC_STREAMS), device=device, dtype=torch.float32
            ),
            "comb": torch.empty(
                (tokens, MHC_STREAMS, MHC_STREAMS), device=device, dtype=torch.float32
            ),
            "out": torch.empty(
                (tokens, MHC_STREAMS, HIDDEN_SIZE), device=device, dtype=torch.bfloat16
            ),
        }

    outputs = allocate_outputs()

    def prepare_call(state):
        spec = state.scratch_specs()[0]
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        binding = state.bind(
            scratch=scratch, tokens=tokens, expected_m=tokens, **outputs
        )
        run = lambda: _impl._b12x_mhc_post_pre_impl(
            x,
            residual,
            prev_post,
            prev_comb,
            fn,
            hc_scale,
            hc_base,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
            sinkhorn_iters=SINKHORN_ITERS,
            norm_weight=None,
            norm_eps=0.0,
            binding=binding,
            _state=state,
        )
        return PreparedCall(run=run, owners=(scratch, binding, outputs))

    request = declaration.request(
        name=f"phase0-mhc-m{tokens}", prepare_call=prepare_call
    )
    prepare_default(request)
    state = require_prepared(request.plan, "norm.mhc", device)
    spec = state.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
    binding = mhc.bind(
        request.plan,
        scratch=scratch,
        tokens=tokens,
        expected_m=tokens,
        **outputs,
    )

    def run():
        return mhc.run_post_pre(
            x,
            residual,
            prev_post,
            prev_comb,
            fn,
            hc_scale,
            hc_base,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
            sinkhorn_iters=SINKHORN_ITERS,
            norm_weight=None,
            norm_eps=0.0,
            binding=binding,
        )

    run()
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        next_residual = _mhc_post_reference(x, residual, prev_post, prev_comb)
        ref_y, ref_post, ref_comb = _mhc_pre_reference(
            next_residual, fn, hc_scale, hc_base, rms_eps=rms_eps, hc_eps=hc_eps
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
    cosines = {
        "residual": _cosine(outputs["out"], next_residual),
        "y": _cosine(outputs["y"], ref_y),
        "post": _cosine(outputs["post"], ref_post),
        "comb": _cosine(outputs["comb"], ref_comb),
    }
    assert min(cosines.values()) > 0.999, f"mHC mismatch at M={tokens}: {cosines}"
    latency_us = _bench_cuda_graph_us(run)
    return {
        "tokens": tokens,
        "latency_us": round(latency_us, 2),
        **{f"{name}_cos_sim": round(value, 7) for name, value in cosines.items()},
    }


@triton.jit(do_not_specialize=["prepared_tokens"])
def _engram_gather_kernel(
    weight_ptr,
    scales_ptr,
    hash_ids_ptr,
    output_ptr,
    prepared_tokens,
    shard_start,
    shard_end,
    table_rows,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    columns = tl.arange(0, 256).to(tl.int64)
    live = token < prepared_tokens
    row = tl.load(hash_ids_ptr + token * 24 + head, mask=live, other=-1).to(tl.int64)
    local = live & (row >= shard_start) & (row < shard_end) & (row < table_rows)
    local_row = tl.where(local, row - shard_start, 0).to(tl.int64)
    quantized = tl.load(
        weight_ptr + local_row * 256 + columns, mask=local, other=0.0
    ).to(tl.float32)
    exponent = tl.load(
        scales_ptr + local_row * 8 + columns // 32, mask=local, other=127
    ).to(tl.uint32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    scale = tl.where(exponent == 0, 2.0**-127, scale)
    scale = tl.where(exponent == 255, float("nan"), scale)
    values = (quantized * scale).to(tl.bfloat16)
    tl.store(
        output_ptr + token * 6144 + head * 256 + columns,
        tl.where(local, values, 0.0),
        mask=live,
    )


def _engram_gather(
    weight: torch.Tensor,
    scales: torch.Tensor,
    hash_ids: torch.Tensor,
    *,
    shard_start: int,
    shard_end: int,
    table_rows: int,
) -> torch.Tensor:
    tokens = hash_ids.shape[0]
    output = torch.empty(
        (tokens, ENGRAM_OUT_DIM), device=hash_ids.device, dtype=torch.bfloat16
    )
    if tokens:
        _engram_gather_kernel[(tokens, ENGRAM_HEADS, 1)](
            weight,
            scales,
            hash_ids,
            output,
            tokens,
            shard_start,
            shard_end,
            table_rows,
            num_warps=4,
        )
    return output


def _native_engram_check(
    *,
    layer_id: int,
    geometry,
    weight: torch.Tensor,
    scales: torch.Tensor,
    hash_ids: torch.Tensor,
    expected: torch.Tensor,
    device: torch.device,
) -> None:
    from flashinfer.experimental.b12x.preparation import (
        FrozenMapping,
        PreparedCall,
        prepare_default,
    )
    from flashinfer.experimental.b12x.sequence import engram

    token_map = list(range(geometry.compressed_vocab_size))
    token_map.extend([0] * (129280 - len(token_map)))
    caps = engram.Caps(
        device=device,
        max_tokens=hash_ids.shape[0],
        max_seqs=1,
        max_requests=1,
        vocab_size=129280,
        layer_id=layer_id,
        tp_size=8,
        tp_rank=0,
    )
    declaration = engram.plan(
        caps,
        token_map=token_map,
        geometry=geometry,
        invocation=FrozenMapping(
            {"operation": "lookup", "compact_rows": False, "resident_scales": False}
        ),
        override=None,
    )
    request = declaration.request(
        name=f"phase0-engram-native-layer{layer_id}",
        prepare_call=lambda state: PreparedCall(run=lambda: None),
    )
    prepare_default(request)
    output = torch.empty_like(expected)
    binding = engram.bind_lookup(
        request.plan,
        weight=weight,
        scales=scales,
        hash_ids=hash_ids,
        num_tokens=torch.tensor([hash_ids.shape[0]], device=device, dtype=torch.int32),
        out=output,
        disk_table=None,
    )
    engram.run_lookup(binding, token_count=hash_ids.shape[0], clear_tail=False)
    torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)


def _run_engram_layer(
    *, layer_id: int, geometry, device: torch.device
) -> dict[str, object]:
    layer_index = geometry.layer_ids.index(layer_id)
    table_rows = geometry.num_embeddings[layer_index]
    shard_rows = (table_rows + 7) // 8
    shard_start = 0
    shard_end = min(shard_rows, table_rows)
    torch.manual_seed(6200 + layer_id)
    weight = torch.randint(
        0,
        127,
        (shard_rows, ENGRAM_HEAD_DIM),
        device=device,
        dtype=torch.uint8,
    ).view(torch.float8_e4m3fn)
    scales = torch.full(
        (shard_rows, ENGRAM_GROUPS), 127, device=device, dtype=torch.uint8
    )
    pinned_weight = torch.empty_like(weight, device="cpu", pin_memory=True)
    pinned_scales = torch.empty_like(scales, device="cpu", pin_memory=True)
    pinned_weight.copy_(weight)
    pinned_scales.copy_(scales)

    warm_hashes = torch.randint(
        shard_start,
        shard_end,
        (6, ENGRAM_HEADS),
        device=device,
        dtype=torch.int64,
    )
    _engram_gather(
        weight,
        scales,
        warm_hashes,
        shard_start=shard_start,
        shard_end=shard_end,
        table_rows=table_rows,
    )
    expected_native = _engram_gather(
        weight,
        scales,
        warm_hashes,
        shard_start=shard_start,
        shard_end=shard_end,
        table_rows=table_rows,
    )
    _native_engram_check(
        layer_id=layer_id,
        geometry=geometry,
        weight=weight,
        scales=scales,
        hash_ids=warm_hashes,
        expected=expected_native,
        device=device,
    )
    _engram_gather(
        pinned_weight,
        pinned_scales,
        warm_hashes,
        shard_start=shard_start,
        shard_end=shard_end,
        table_rows=table_rows,
    )

    rows: dict[int, dict[str, float]] = {}
    with patch.object(
        _engram_gather_kernel, "_do_compile", wraps=_engram_gather_kernel._do_compile
    ) as compile_spy:
        for tokens in TOKEN_COUNTS:
            hash_ids = torch.randint(
                shard_start,
                shard_end,
                (tokens, ENGRAM_HEADS),
                device=device,
                dtype=torch.int64,
            )
            if tokens >= 2:
                hash_ids[0, 0] = shard_start
                hash_ids[1, 0] = shard_end - 1
            vram = _engram_gather(
                weight,
                scales,
                hash_ids,
                shard_start=shard_start,
                shard_end=shard_end,
                table_rows=table_rows,
            )
            pinned = _engram_gather(
                pinned_weight,
                pinned_scales,
                hash_ids,
                shard_start=shard_start,
                shard_end=shard_end,
                table_rows=table_rows,
            )
            local_ids = hash_ids.long() - shard_start
            dequantized = weight[local_ids].float()
            scale_values = torch.exp2(scales[local_ids].float() - 127.0)
            reference = (
                dequantized.reshape(tokens, ENGRAM_HEADS, ENGRAM_GROUPS, 32)
                * scale_values.unsqueeze(-1)
            ).reshape(tokens, ENGRAM_OUT_DIM)
            vram_cos = _cosine(vram, reference)
            pinned_cos = _cosine(pinned, reference)
            assert vram_cos > 0.9999 and pinned_cos > 0.9999
            vram_us = _bench_cuda_graph_us(
                lambda hash_ids=hash_ids: _engram_gather(
                    weight,
                    scales,
                    hash_ids,
                    shard_start=shard_start,
                    shard_end=shard_end,
                    table_rows=table_rows,
                )
            )
            pinned_us = _bench_cuda_graph_us(
                lambda hash_ids=hash_ids: _engram_gather(
                    pinned_weight,
                    pinned_scales,
                    hash_ids,
                    shard_start=shard_start,
                    shard_end=shard_end,
                    table_rows=table_rows,
                )
            )
            gather_bytes = tokens * ENGRAM_HEADS * (ENGRAM_HEAD_DIM + ENGRAM_GROUPS)
            rows[tokens] = {
                "vram_cos_sim": round(vram_cos, 7),
                "pinned_cos_sim": round(pinned_cos, 7),
                "vram_latency_us": round(vram_us, 2),
                "pinned_latency_us": round(pinned_us, 2),
                "vram_effective_gbps": round(gather_bytes / (vram_us * 1e-6) / 1e9, 1),
                "pinned_effective_gbps": round(
                    gather_bytes / (pinned_us * 1e-6) / 1e9, 1
                ),
                "pinned_over_vram_latency": round(pinned_us / vram_us, 3),
            }
        assert (
            compile_spy.call_count == 0
        ), f"Engram runtime M triggered {compile_spy.call_count} Triton compilations"
    return {
        "layer_id": layer_id,
        "table_rows": table_rows,
        "local_shard_rows": shard_rows,
        "max_local_weight_byte_offset": (shard_rows - 1) * ENGRAM_HEAD_DIM + 255,
        "native_b12x_lookup_exact_match": True,
        "zero_jit_recompilation_passed": True,
        "results": rows,
    }


def run_step_05a_suite() -> dict[str, object]:
    from flashinfer.experimental.b12x.norm import mhc
    from flashinfer.experimental.b12x.sequence import engram

    device = torch.device("cuda:0")
    assert torch.cuda.get_device_capability(device) == (12, 0)
    assert mhc.is_supported(device)
    assert engram.is_supported(device)
    mhc_results = [
        _run_mhc_case(tokens=tokens, device=device) for tokens in TOKEN_COUNTS
    ]
    torch.cuda.empty_cache()
    geometry = engram.build_geometry(
        layer_ids=(1, 14), base_table_size=16000000, compressed_vocab_size=99092
    )
    engram_results = [
        _run_engram_layer(layer_id=layer_id, geometry=geometry, device=device)
        for layer_id in (1, 14)
    ]
    return {
        "step": "0.5a",
        "device": torch.cuda.get_device_name(device),
        "compute_capability": "12.0",
        "torch_version": torch.__version__,
        "mhc": {
            "operation": "post_pre",
            "hidden_size": HIDDEN_SIZE,
            "streams": MHC_STREAMS,
            "sinkhorn_iters": SINKHORN_ITERS,
            "results": mhc_results,
        },
        "engram": {
            "layer_ids": (1, 14),
            "heads_per_token": ENGRAM_HEADS,
            "head_dim": ENGRAM_HEAD_DIM,
            "tp_size": 8,
            "tp_rank": 0,
            "host_memory": "pinned UVA zero-copy",
            "results": engram_results,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run_step_05a_suite()
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
