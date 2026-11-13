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

"""Phase 0 Step 0.3: DSV4.1 router and native MXFP4 MoE on sm_120."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from zstep_worker.kernels_sm120.moe_router import (
    dsv41_sqrt_softplus_topk,
    dsv41_sqrt_softplus_topk_kernel,
)

TOKEN_COUNTS = (6, 48, 192, 4096)
ROUTER_TOKEN_COUNTS = (6, 17, 48, 192, 4096)
NUM_EXPERTS = 384
TOP_K = 6
HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 2304
ROUTED_SCALING_FACTOR = 1.5
SWIGLU_LIMIT = 10.0
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


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


def _router_reference(
    logits: torch.Tensor,
    correction_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.sqrt(F.softplus(logits.float()))
    selection_scores = scores if correction_bias is None else scores + correction_bias
    ids = torch.topk(selection_scores, TOP_K, dim=-1).indices.to(torch.int32)
    weights = scores.gather(1, ids.long())
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.17549435e-38)
    return weights * ROUTED_SCALING_FACTOR, ids


def _run_router_suite(device: torch.device) -> dict[str, object]:
    torch.manual_seed(4103)
    logits = torch.randn((6, NUM_EXPERTS), device=device, dtype=torch.float32)
    bias = torch.randn(NUM_EXPERTS, device=device, dtype=torch.float32) * 0.05
    for bias_arg in (None, bias):
        dsv41_sqrt_softplus_topk(
            logits,
            top_k=TOP_K,
            renormalize=True,
            routed_scaling_factor=ROUTED_SCALING_FACTOR,
            correction_bias=bias_arg,
        )

    results: dict[str, object] = {}
    with patch.object(
        dsv41_sqrt_softplus_topk_kernel,
        "_do_compile",
        wraps=dsv41_sqrt_softplus_topk_kernel._do_compile,
    ) as compile_spy:
        for use_bias in (False, True):
            label = "with_correction_bias" if use_bias else "without_correction_bias"
            bias_arg = bias if use_bias else None
            rows: dict[int, dict[str, float]] = {}
            for m in ROUTER_TOKEN_COUNTS:
                logits = torch.randn(
                    (m, NUM_EXPERTS), device=device, dtype=torch.float32
                )
                weights, ids = dsv41_sqrt_softplus_topk(
                    logits,
                    top_k=TOP_K,
                    renormalize=True,
                    routed_scaling_factor=ROUTED_SCALING_FACTOR,
                    correction_bias=bias_arg,
                )
                ref_weights, ref_ids = _router_reference(logits, bias_arg)
                assert torch.equal(
                    ids, ref_ids
                ), f"Router IDs differ for {label}, M={m}"
                torch.testing.assert_close(weights, ref_weights, rtol=2e-6, atol=2e-6)
                cos = float(
                    F.cosine_similarity(
                        weights.flatten(), ref_weights.flatten(), dim=0
                    ).item()
                )
                latency_us = _bench_cuda_graph_us(
                    lambda logits=logits, bias_arg=bias_arg: dsv41_sqrt_softplus_topk(
                        logits,
                        top_k=TOP_K,
                        renormalize=True,
                        routed_scaling_factor=ROUTED_SCALING_FACTOR,
                        correction_bias=bias_arg,
                    )
                )
                rows[m] = {"cos_sim": round(cos, 8), "latency_us": round(latency_us, 3)}
            results[label] = rows
        assert (
            compile_spy.call_count == 0
        ), f"Runtime batch size triggered {compile_spy.call_count} Triton compilations"

    large_rows = 2_300_000_000 // NUM_EXPERTS + 4
    large_logits = torch.empty(
        (large_rows, NUM_EXPERTS), device=device, dtype=torch.float32
    )
    large_logits[-2:].normal_()
    large_logits[0].copy_(large_logits[-2])
    strided_rows = large_logits[:: large_rows - 1]
    weights, ids = dsv41_sqrt_softplus_topk(
        strided_rows,
        top_k=TOP_K,
        renormalize=True,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        correction_bias=bias,
    )
    ref_weights, ref_ids = _router_reference(large_logits[-1:], bias)
    assert torch.equal(ids[1], ref_ids[0])
    torch.testing.assert_close(weights[1], ref_weights[0], rtol=2e-6, atol=2e-6)
    results["int64_large_offset"] = {
        "passed": True,
        "logit_elements": int(large_logits.numel()),
        "allocation_gib": round(large_logits.numel() * 4 / 1024**3, 3),
    }
    results["zero_jit_recompilation_passed"] = True
    del large_logits, strided_rows, weights, ids, ref_weights, ref_ids
    torch.cuda.empty_cache()
    return results


def _make_mxfp4_weights(
    *,
    num_experts: int,
    intermediate_size: int,
    device: torch.device,
):
    from flashinfer.experimental.b12x.moe import fused_moe

    generator = torch.Generator(device=device).manual_seed(4201 + num_experts)
    w13 = torch.randint(
        0,
        256,
        (num_experts, 2 * intermediate_size, HIDDEN_SIZE // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w2 = torch.randint(
        0,
        256,
        (num_experts, HIDDEN_SIZE, intermediate_size // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w13_scales = torch.full(
        (num_experts, 2 * intermediate_size, HIDDEN_SIZE // 32),
        121,
        dtype=torch.uint8,
        device=device,
    )
    w2_scales = torch.full(
        (num_experts, HIDDEN_SIZE, intermediate_size // 32),
        121,
        dtype=torch.uint8,
        device=device,
    )
    global_scales = torch.ones(num_experts, device=device, dtype=torch.float32)
    source_weights = fused_moe.PackedWeights(
        w13=w13,
        w2=w2,
        w13_block_scales=w13_scales,
        w2_block_scales=w2_scales,
        w13_global_scales=global_scales,
        w2_global_scales=global_scales.clone(),
        input_scale=None,
        intermediate_scale=None,
        immutable_input_scales=False,
    )
    weight_plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(
            format=fused_moe.PackedSourceFormat.MXFP4_E8M0_K32,
            w13_layout=fused_moe.W13Layout.W13,
        ),
        activation=fused_moe.ActivationSpec(
            mode=fused_moe.ActivationMode.A16,
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
            swiglu_limit=SWIGLU_LIMIT,
            swiglu_alpha=None,
            swiglu_beta=None,
            rotation_dtype=None,
            a16_max_tokens=0,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=num_experts,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=intermediate_size,
        ),
        constraints=None,
    )
    native_weights = fused_moe.PackedWeights(
        w13=source_weights.w13.clone(),
        w2=source_weights.w2.clone(),
        w13_block_scales=source_weights.w13_block_scales.clone(),
        w2_block_scales=source_weights.w2_block_scales.clone(),
        w13_global_scales=source_weights.w13_global_scales.clone(),
        w2_global_scales=source_weights.w2_global_scales.clone(),
        input_scale=None,
        intermediate_scale=None,
        immutable_input_scales=False,
    )
    experts = fused_moe.prepare_weights(plan=weight_plan, weights=native_weights)
    return source_weights, experts


def _dequantize_weights(
    source_weights,
    *,
    num_experts: int,
    intermediate_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = source_weights.w13.device
    lut = torch.tensor(
        (*E2M1_VALUES, *(-value for value in E2M1_VALUES)),
        dtype=torch.float32,
        device=device,
    )
    deq13 = torch.empty(
        (num_experts, 2 * intermediate_size, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=device,
    )
    deq2 = torch.empty(
        (num_experts, HIDDEN_SIZE, intermediate_size),
        dtype=torch.bfloat16,
        device=device,
    )
    for expert in range(num_experts):
        for packed, scales, destination in (
            (source_weights.w13, source_weights.w13_block_scales, deq13),
            (source_weights.w2, source_weights.w2_block_scales, deq2),
        ):
            q = packed[expert]
            codes = torch.stack((q & 15, (q >> 4) & 15), dim=-1)
            values = lut[codes.long()].reshape(destination.shape[1:])
            scale = torch.exp2(scales[expert].float() - 127.0).unsqueeze(-1)
            destination[expert].copy_(
                (values.reshape(values.shape[0], -1, 32) * scale).reshape(values.shape)
            )
    return deq13, deq2


def _moe_reference(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    deq13: torch.Tensor,
    deq2: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros(
        (x.shape[0], HIDDEN_SIZE), dtype=torch.float32, device=x.device
    )
    for expert in range(deq13.shape[0]):
        locations = (topk_ids == expert).nonzero(as_tuple=False)
        if locations.numel() == 0:
            continue
        rows = locations[:, 0]
        routes = locations[:, 1]
        fc1 = x[rows].float() @ deq13[expert].float().T
        up, gate = fc1.chunk(2, dim=-1)
        gate = gate.clamp(max=SWIGLU_LIMIT)
        up = up.clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
        intermediate = (F.silu(gate) * up).to(torch.bfloat16).float()
        expert_out = intermediate @ deq2[expert].float().T
        output.index_add_(
            0, rows, expert_out * topk_weights[rows, routes].unsqueeze(-1).float()
        )
    return output


def _prepare_binding(
    *,
    experts,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
):
    from flashinfer.experimental.b12x.moe import fused_moe
    from flashinfer.experimental.b12x.preparation import (
        FrozenMapping,
        PreparedCall,
        prepare_default,
        require_prepared,
    )

    declaration = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=max(x.shape[0], 64),
            top_k=TOP_K,
            warmup_token_counts=(),
            route_num_experts=None,
        ),
        routing=fused_moe.RoutingSpec(
            apply_router_weight_on_input=False,
            logits_dtype=None,
            deterministic_output=True,
            collect_activation_amax=False,
            score_func="softmax",
            renormalize=True,
            has_correction_bias=False,
            has_image_correction_bias=False,
            has_image_mask=False,
            routed_scaling_factor=1.0,
        ),
        invocation=FrozenMapping({"fast_math": True}),
        override=None,
    )

    def prepare_call(state):
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            for spec in state.scratch.scratch_specs()
        )
        binding = state.bind(
            experts=experts,
            scratch=scratch,
            a=x,
            output=output,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            input_scales_static=False,
            fast_math=True,
        )
        return PreparedCall(run=lambda: state.run(binding), owners=(scratch, binding))

    request = declaration.request(
        name=f"phase0-moe-m{x.shape[0]}", prepare_call=prepare_call
    )
    prepare_default(request)
    state = require_prepared(request.plan, "moe.decode", device=x.device)
    scratch = tuple(
        torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        for spec in state.scratch.scratch_specs()
    )
    binding = fused_moe.bind(
        request.plan,
        experts=experts,
        scratch=scratch,
        a=x,
        output=output,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        input_scales_static=False,
        fast_math=True,
    )
    return binding, scratch


def _run_moe_geometry(
    *,
    label: str,
    num_experts: int,
    intermediate_size: int,
    device: torch.device,
) -> dict[str, object]:
    from flashinfer.experimental.b12x.moe import fused_moe

    source_weights, experts = _make_mxfp4_weights(
        num_experts=num_experts, intermediate_size=intermediate_size, device=device
    )
    deq13, deq2 = _dequantize_weights(
        source_weights, num_experts=num_experts, intermediate_size=intermediate_size
    )
    rows: dict[int, dict[str, float | int]] = {}
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for m in TOKEN_COUNTS:
            x = (
                torch.randn((m, HIDDEN_SIZE), device=device, dtype=torch.bfloat16)
                * 0.25
            )
            router_logits = torch.randn(
                (m, num_experts), device=device, dtype=torch.float32
            )
            topk_weights, topk_ids = dsv41_sqrt_softplus_topk(
                router_logits,
                top_k=TOP_K,
                renormalize=True,
                routed_scaling_factor=ROUTED_SCALING_FACTOR,
                correction_bias=None,
            )
            output = torch.empty_like(x)
            binding, scratch = _prepare_binding(
                experts=experts,
                x=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                output=output,
            )
            fused_moe.run(binding=binding)
            reference = _moe_reference(x, topk_weights, topk_ids, deq13, deq2)
            cos = float(
                F.cosine_similarity(
                    output.float().flatten(), reference.flatten(), dim=0
                ).item()
            )
            assert cos > 0.995, f"{label} cosine similarity too low at M={m}: {cos}"
            latency_us = _bench_cuda_graph_us(
                lambda bound=binding: fused_moe.run(binding=bound)
            )
            unique_experts = int(torch.unique(topk_ids).numel())
            weight_values_per_expert = 3 * HIDDEN_SIZE * intermediate_size
            bytes_moved = (
                unique_experts * weight_values_per_expert * (0.5 + 1.0 / 32.0)
                + m * HIDDEN_SIZE * 2
                + m * HIDDEN_SIZE * 2
            )
            effective_gbps = bytes_moved / (latency_us * 1e-6) / 1e9
            rows[m] = {
                "cos_sim": round(cos, 6),
                "latency_us": round(latency_us, 2),
                "effective_gbps": round(effective_gbps, 1),
                "unique_experts": unique_experts,
            }
            del (
                binding,
                scratch,
                output,
                reference,
                router_logits,
                topk_weights,
                topk_ids,
                x,
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
    del deq13, deq2, source_weights, experts
    torch.cuda.empty_cache()
    return {
        "num_local_experts": num_experts,
        "local_intermediate_size": intermediate_size,
        "activation_recipe": "W4A16",
        "communication_included": False,
        "effective_bandwidth_basis": "unique routed expert MXFP4 values and scales plus BF16 input/output",
        "results": rows,
    }


def run_step_03_suite() -> dict[str, object]:
    from flashinfer.experimental.b12x.moe import fused_moe

    device = torch.device("cuda:0")
    assert torch.cuda.get_device_capability(device) == (12, 0)
    assert fused_moe.is_supported(device)
    router = _run_router_suite(device)
    moe = {
        "ep8_local_rank": _run_moe_geometry(
            label="EP8", num_experts=48, intermediate_size=2304, device=device
        ),
        "tp8_local_rank": _run_moe_geometry(
            label="TP8", num_experts=384, intermediate_size=288, device=device
        ),
    }
    return {
        "step": "0.3",
        "device": torch.cuda.get_device_name(device),
        "compute_capability": "12.0",
        "torch_version": torch.__version__,
        "router": router,
        "moe_geometry": {
            "global_experts": NUM_EXPERTS,
            "hidden_size": HIDDEN_SIZE,
            "global_intermediate_size": INTERMEDIATE_SIZE,
            "top_k": TOP_K,
            "source_format": "MXFP4_E8M0_K32",
            "swiglu_limit": SWIGLU_LIMIT,
        },
        "moe": moe,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run_step_03_suite()
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
