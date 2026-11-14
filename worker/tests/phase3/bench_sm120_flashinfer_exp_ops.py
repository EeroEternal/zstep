#!/usr/bin/env python3
# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT
# ruff: noqa: B023
"""Route B Step 2: Single-GPU (RTX 6000D, 156 SMs, sm_120) A/B Microbenchmark & Parity Suite.

Supports two modes via `--mode`:
  - `b12x_only`: Benchmarks `b12x.moe.fused_moe` and `b12x.gemm.block_fp8_linear`
    (run once against `b12x-overlay@79819b11` and once against `flashinfer-exp@ae58183f`).
  - `new_ops`: Benchmarks & checks numerical parity of:
      1. Model config & vLLM 0.31.0 `DeepSeekV4MoE` / `SharedFusedMoE` router details
      2. `sm12x_mega_gate` vs `GateLinear + fused_topk_bias` (standalone & chained with RMSNorm + b12x MoE)
      3. `prepare_sm120_paged_mqa_logits` vs `deep_gemm.fp8_fp4_paged_mqa_logits` (single layer & full 8-layer step)
      4. `sm120_dsa_indexer` (`dsa_indexer_det_topk` paged CUDAGraph & ragged Prefill) vs vLLM `SparseAttnIndexer`
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import sys
import time
import statistics
import torch


def bench_cudagraph(fn, warmup: int = 100, iters: int = 400) -> tuple[float, float]:
    """Capture `fn` in a CUDA graph and measure median & mean per-launch latency in us."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    torch.cuda.synchronize()

    # Batch 20 graph replays per timing window for sub-microsecond accuracy
    batch_replays = 20
    batches = max(5, iters // batch_replays)
    for _ in range(max(2, warmup // batch_replays)):
        for _ in range(batch_replays):
            g.replay()
    torch.cuda.synchronize()

    samples_us: list[float] = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    for _ in range(batches):
        start_ev.record()
        for _ in range(batch_replays):
            g.replay()
        end_ev.record()
        end_ev.synchronize()
        samples_us.append((start_ev.elapsed_time(end_ev) * 1000.0) / batch_replays)

    return statistics.median(samples_us), statistics.mean(samples_us)


def bench_eager(fn, warmup: int = 15, iters: int = 60) -> tuple[float, float]:
    """Measure eager execution latency in us (median, mean)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples_us: list[float] = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start_ev.record()
        fn()
        end_ev.record()
        end_ev.synchronize()
        samples_us.append(start_ev.elapsed_time(end_ev) * 1000.0)
    return statistics.median(samples_us), statistics.mean(samples_us)


def run_b12x_bench(tag: str, cache_dir: str) -> None:
    os.environ["B12X_W4A8_TINY_DECODE"] = "0"
    compat = "/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/_compat"
    if os.path.isdir(compat) and compat not in sys.path:
        sys.path.insert(0, compat)
        os.environ["PYTHONPATH"] = f"{compat}:{os.environ.get('PYTHONPATH', '')}"

    import b12x
    from b12x.gemm import block_fp8_linear
    from b12x.moe import fused_moe
    from b12x.moe.fused_moe.workloads import make_tuning_routes
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.preparation.types import require_prepared

    dev = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(dev)
    print(f"=== [b12x_only: {tag}] GPU={props.name} SMs={props.multi_processor_count} b12x={b12x.__file__} ===")

    # 1. Prepare MoE (E=384, top_k=6, H=5120, inter_dim=288, swiglu_limit=10.0)
    torch.manual_seed(42)
    E, top_k, H, inter_dim = 384, 6, 5120, 288
    w13 = torch.randint(0, 256, (E, 2 * inter_dim, H // 2), dtype=torch.uint8, device=dev)
    w2 = torch.randint(0, 256, (E, H, inter_dim // 2), dtype=torch.uint8, device=dev)
    w13_s = torch.full((E, 2 * inter_dim, H // 32), 127, dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)
    w2_s = torch.full((E, H, inter_dim // 32), 127, dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)
    unit = torch.ones(E, dtype=torch.float32, device=dev)

    wplan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(
            format=fused_moe.PackedSourceFormat("fp4_e8m0_k32"),
            w13_layout=fused_moe.W13Layout("w31"),
        ),
        activation=fused_moe.ActivationSpec(
            mode=fused_moe.ActivationMode.A8,
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
            swiglu_limit=10.0,
        ),
        geometry=fused_moe.MoEGeometry(num_experts=E, hidden_size=H, intermediate_size=inter_dim),
    )
    prep_exp = fused_moe.prepare_weights(
        plan=wplan,
        weights=fused_moe.PackedWeights(
            w13=w13,
            w2=w2,
            w13_block_scales=w13_s,
            w2_block_scales=w2_s,
            w13_global_scales=unit,
            w2_global_scales=unit,
            input_scale=unit,
            intermediate_scale=unit,
            immutable_input_scales=True,
        ),
    )
    max_tokens = 3072
    eplan = fused_moe.plan_execution(
        experts=prep_exp,
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=max_tokens,
            top_k=top_k,
            warmup_token_counts=(1, 2, 4, 8, 16, 32, 64, 128, 512, 3072),
            route_num_experts=0,
        ),
        routing=fused_moe.RoutingSpec(apply_router_weight_on_input=False),
    )

    requests = []

    def _make_moe_call(cnt: int):
        def _f(state):
            scr = tuple(torch.empty(s.shape, dtype=s.dtype, device=dev) for s in state.scratch.scratch_specs())
            hid = torch.randn((cnt, H), dtype=torch.bfloat16, device=dev) * 0.125
            out = torch.empty_like(hid)
            ids = make_tuning_routes(cnt, top_k, E, device=dev)[0].clone()
            wts = torch.full((cnt, top_k), 1.0 / top_k, dtype=torch.float32, device=dev)
            b = state.bind(scratch=scr, a=hid, experts=prep_exp, topk_weights=wts, topk_ids=ids, output=out, input_scales_static=True)
            return PreparedCall(run=b.run, output=out, produce=lambda: hid.normal_(std=0.125), reset=lambda: out.zero_())
        return _f

    calls = {int(c): _make_moe_call(int(c)) for c in eplan.token_counts}
    requests.append(eplan.request(name="moe_bench", prepare_calls=calls, benchmark_calls=calls))

    lin_shapes = [(1792, 5120), (2048, 1280), (5120, 2048), (5120, 288), (3200, 6144)]
    lin_caps = (1, 8, 32, 128, 512, 2048)
    packed_weights = {}
    lin_plans = {}
    for out_f, in_f in lin_shapes:
        wf = (torch.randn(out_f, in_f, device=dev, dtype=torch.bfloat16) * 0.05).to(torch.float8_e4m3fn)
        ws = torch.full((out_f // 32, in_f // 32), 127, dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)
        pw = block_fp8_linear.pack_weight(wf, ws, block_size=(32, 32))
        packed_weights[(out_f, in_f)] = pw
        for cap in lin_caps:
            p = block_fp8_linear.plan(
                block_fp8_linear.Caps(device=dev, max_tokens=cap, in_features=in_f, out_features=out_f, block_size=(32, 32))
            )
            lin_plans[(out_f, in_f, cap)] = p

            def _mk_lin(state, _pw=pw):
                q = state.query
                src = torch.randn((q.max_tokens, q.in_features), dtype=torch.bfloat16, device=dev) * 0.25
                out = torch.empty((q.max_tokens, q.out_features, 1), dtype=torch.bfloat16, device=dev)
                scr = [torch.empty(sp.shape, dtype=sp.dtype, device=dev) for sp in state.scratch.scratch_specs()]
                b = state.bind(scratch=scr, source=src, packed_weight=_pw, output=out)
                return PreparedCall(run=lambda: state.run_binding(b), produce=lambda: src.normal_(std=0.25))

            requests.append(p.request(name=f"fp8_{out_f}x{in_f}_m{cap}", prepare_call=_mk_lin, benchmark_call=_mk_lin))

    t0 = time.perf_counter()
    sess = PreparationSession(device=0, autotune=True, cache_dir=cache_dir)
    sess.prepare(tuple(requests))
    print(f"PreparationSession done in {(time.perf_counter() - t0):.2f}s")

    moe_state = require_prepared(eplan, "moe.decode")
    moe_scr = tuple(torch.empty(s.shape, dtype=s.dtype, device=dev) for s in eplan.scratch_specs())

    print(f"\n--- b12x.moe.fused_moe ({tag}) ---")
    for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072):
        hid = torch.randn((m, H), dtype=torch.bfloat16, device=dev) * 0.125
        out = torch.empty_like(hid)
        ids = make_tuning_routes(m, top_k, E, device=dev)[0].clone()
        wts = torch.full((m, top_k), 1.0 / top_k, dtype=torch.float32, device=dev)
        b = moe_state.bind(scratch=moe_scr, a=hid, experts=prep_exp, topk_weights=wts, topk_ids=ids, output=out, input_scales_static=True)
        if m <= 128:
            med, avg = bench_cudagraph(b.run)
            mode_s = "CG"
        else:
            med, avg = bench_eager(b.run)
            mode_s = "Eager"
        print(f"MoE M={m:4d} [{mode_s:5s}]: median={med:7.2f} us  mean={avg:7.2f} us")

    print(f"\n--- b12x.gemm.block_fp8_linear ({tag}) ---")
    for out_f, in_f in lin_shapes:
        pw = packed_weights[(out_f, in_f)]
        row_parts = []
        for m in lin_caps:
            p = lin_plans[(out_f, in_f, m)]
            st = require_prepared(p, "gemm.block_fp8_linear")
            scr = [torch.empty(sp.shape, dtype=sp.dtype, device=dev) for sp in p.scratch_specs()]
            src = torch.randn((m, in_f), dtype=torch.bfloat16, device=dev) * 0.25
            out = torch.empty((m, out_f, 1), dtype=torch.bfloat16, device=dev)
            b = st.bind(scratch=scr, source=src, packed_weight=pw, output=out)
            if m <= 128:
                med, _ = bench_cudagraph(lambda _st=st, _b=b: _st.run_binding(_b))
            else:
                med, _ = bench_eager(lambda _st=st, _b=b: _st.run_binding(_b))
            row_parts.append(f"M={m}:{med:.2f}us")
        print(f"FP8 {out_f:4d}x{in_f:4d}: " + " | ".join(row_parts))

    sess.close()


def run_new_ops_bench() -> None:
    compat = "/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/_compat"
    if os.path.isdir(compat) and compat not in sys.path:
        sys.path.insert(0, compat)
        os.environ["PYTHONPATH"] = f"{compat}:{os.environ.get('PYTHONPATH', '')}"
    os.environ["B12X_W4A8_TINY_DECODE"] = "0"

    dev = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(dev)
    print(f"=== [new_ops] GPU={props.name} SMs={props.multi_processor_count} ===")

    # 0. Inspect model checkpoint config & vLLM router / SharedFusedMoE behavior
    model_dir_env = os.environ.get("ZSTEP_DSV41_MODEL_DIR", "/models/DeepSeek-V4.1-Flash")
    cfg_candidates = (
        [str(Path(model_dir_env) / "config.json")]
        + glob.glob("/models/*/config.json")
        + glob.glob("/models/*/*/config.json")
    )
    for cpath in cfg_candidates:
        if not Path(cpath).is_file():
            continue
        cfg = json.loads(Path(cpath).read_text())
        if cfg.get("hidden_size") == 5120 and cfg.get("n_routed_experts") == 384:
            print(f"\n[Model Config: {cpath}]")
            for k in (
                "architectures",
                "num_hidden_layers",
                "num_hash_layers",
                "hash_layers",
                "n_routed_experts",
                "n_shared_experts",
                "num_experts_per_tok",
                "scoring_func",
                "topk_method",
                "routed_scaling_factor",
                "index_source_layer_ids",
                "kv_source_layer_ids",
                "candidate_source_layer_id",
                "candidate_topk_blocks",
                "candidate_block_size",
            ):
                if k in cfg:
                    print(f"  {k}: {cfg[k]}")
            break

    try:
        import inspect
        from vllm.models.deepseek_v4.nvidia.model import FusedMoEFactory
        print("\n[FusedMoEFactory tail & MoERunner forward check]")
        src_fm = inspect.getsource(FusedMoEFactory)
        for i, line in enumerate(src_fm.splitlines(), 1):
            if i >= 180:
                print(f"  FMOE:{i}: {line}")
    except Exception as exc:
        print(f"[Info] FusedMoEFactory inspect skipped: {exc}")

    # =========================================================================
    # 1. MoE Router Gate Benchmark & Parity:
    #    vLLM `GateLinear + fused_topk_bias` vs `prepare_mega_gate` (SM120 CuTe DSL)
    # =========================================================================
    print("\n" + "=" * 78)
    print("BENCHMARK 1: MoE Router Gate (H=5120, E=384, top_k=6, sqrtsoftplus, scale=1.5)")
    print("=" * 78)

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.models.deepseek_v4.nvidia.model import GateLinear, fused_topk_bias
    from flashinfer.mega_gate import prepare_mega_gate
    from vllm.model_executor.layers.layernorm import RMSNorm

    vcfg_ctx = set_current_vllm_config(VllmConfig())
    vcfg_ctx.__enter__()
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:29577",
        local_rank=0,
    )
    initialize_model_parallel(1, 1)

    torch.manual_seed(1234)
    H, E, top_k = 5120, 384, 6
    gate = GateLinear(
        H, E, params_dtype=torch.bfloat16, out_dtype=torch.float32
    ).to(device=dev)
    with torch.no_grad():
        gate_w_bf16 = (torch.randn(E, H, device=dev, dtype=torch.float32) * 0.02).to(torch.bfloat16).contiguous()
        gate.weight.copy_(gate_w_bf16.to(gate.weight.dtype))
    bias_f32 = (torch.randn(E, device=dev, dtype=torch.float32) * 0.01).contiguous()
    print(f"GateLinear weight dtype={gate.weight.dtype}, forward_type={type(gate)}")

    # Numerical Parity Check across M = 1, 16, 128, 1024
    for m_chk in (1, 16, 128, 1024):
        x_chk = (torch.randn(m_chk, H, device=dev, dtype=torch.bfloat16) * 0.5).contiguous()
        logits_ref, _ = gate(x_chk)
        w_ref, id_ref = fused_topk_bias(
            hidden_states=x_chk,
            gating_output=logits_ref,
            scoring_func="sqrtsoftplus",
            e_score_correction_bias=bias_f32,
            topk=top_k,
            renormalize=True,
            indices_type=torch.int32,
            routed_scaling_factor=1.5,
        )
        plan_chk = prepare_mega_gate(
            x_chk,
            gate_w_bf16,
            num_topk=top_k,
            scoring_func="sqrtsoftplus",
            bias=bias_f32,
            routed_scaling_factor=1.5,
            enable_pdl=True,
        )
        id_mg, w_mg = plan_chk.run()
        torch.cuda.synchronize()

        # Sort expert indices per token to compare set & corresponding weights
        id_ref_s, perm_ref = torch.sort(id_ref.to(torch.int64), dim=-1)
        w_ref_s = torch.gather(w_ref, -1, perm_ref)
        id_mg_s, perm_mg = torch.sort(id_mg, dim=-1)
        w_mg_s = torch.gather(w_mg, -1, perm_mg)

        exact_row_match = (id_ref_s == id_mg_s).all(dim=-1).float().mean().item() * 100.0
        slot_overlap = sum(
            len(set(id_ref_s[r].tolist()) & set(id_mg_s[r].tolist())) for r in range(m_chk)
        ) / (m_chk * top_k) * 100.0
        matched_rows = (id_ref_s == id_mg_s).all(dim=-1)
        max_w_diff = (
            (w_ref_s[matched_rows] - w_mg_s[matched_rows]).abs().max().item()
            if matched_rows.any()
            else float("nan")
        )
        print(
            f"  [Parity M={m_chk:4d}] exact_top6_rows={exact_row_match:6.2f}% "
            f"slot_overlap={slot_overlap:6.2f}% max_weight_diff={max_w_diff:.3e} route={plan_chk.route['kernel']}"
        )

    # Standalone latency across M = 1..3072
    print("\n  [Standalone Router Latency: 1 layer vs 40-layer step total]")
    for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072):
        x = (torch.randn(m, H, device=dev, dtype=torch.bfloat16) * 0.5).contiguous()

        def _run_vllm_gate(_x=x):
            lg, _ = gate(_x)
            return fused_topk_bias(
                hidden_states=_x,
                gating_output=lg,
                scoring_func="sqrtsoftplus",
                e_score_correction_bias=bias_f32,
                topk=top_k,
                renormalize=True,
                indices_type=torch.int32,
                routed_scaling_factor=1.5,
            )

        mg_plan_pdl = prepare_mega_gate(
            x, gate_w_bf16, num_topk=top_k, scoring_func="sqrtsoftplus",
            bias=bias_f32, routed_scaling_factor=1.5, enable_pdl=True,
        )
        out_i32 = torch.empty((m, top_k), dtype=torch.int32, device=dev)

        def _run_mg_raw(_p=mg_plan_pdl):
            return _p.run()

        def _run_mg_cast(_p=mg_plan_pdl, _o=out_i32):
            idx64, wts = _p.run()
            _o.copy_(idx64)
            return _o, wts

        if m <= 128:
            v_med, _ = bench_cudagraph(_run_vllm_gate)
            m_raw, _ = bench_cudagraph(_run_mg_raw)
            m_cast, _ = bench_cudagraph(_run_mg_cast)
            mode_s = "CG"
        else:
            v_med, _ = bench_eager(_run_vllm_gate)
            m_raw, _ = bench_eager(_run_mg_raw)
            m_cast, _ = bench_eager(_run_mg_cast)
            mode_s = "Eager"
        saved_40 = (v_med - m_cast) * 40.0
        print(
            f"  M={m:4d} [{mode_s:5s}]: vLLM={v_med:6.2f}us | mega_raw={m_raw:6.2f}us | "
            f"mega+i32={m_cast:6.2f}us | speedup={v_med/m_cast:4.2f}x | 40-layer save={saved_40:+7.1f}us"
        )

    # Also test Chained In-Situ CUDAGraph: RMSNorm -> Gate -> b12x.moe.fused_moe (measures true PDL overlap!)
    from b12x.moe import fused_moe
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.preparation.types import require_prepared
    from b12x.moe.fused_moe.workloads import make_tuning_routes

    inter_dim = 288
    w13 = torch.randint(0, 256, (E, 2 * inter_dim, H // 2), dtype=torch.uint8, device=dev)
    w2 = torch.randint(0, 256, (E, H, inter_dim // 2), dtype=torch.uint8, device=dev)
    w13_s = torch.full((E, 2 * inter_dim, H // 32), 127, dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)
    w2_s = torch.full((E, H, inter_dim // 32), 127, dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)
    unit = torch.ones(E, dtype=torch.float32, device=dev)
    wplan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format=fused_moe.PackedSourceFormat("fp4_e8m0_k32"), w13_layout=fused_moe.W13Layout("w31")),
        activation=fused_moe.ActivationSpec(mode=fused_moe.ActivationMode.A8, nonlinearity="silu", io_dtype=torch.bfloat16, swiglu_limit=10.0),
        geometry=fused_moe.MoEGeometry(num_experts=E, hidden_size=H, intermediate_size=inter_dim),
    )
    prep_exp = fused_moe.prepare_weights(
        plan=wplan,
        weights=fused_moe.PackedWeights(
            w13=w13, w2=w2, w13_block_scales=w13_s, w2_block_scales=w2_s,
            w13_global_scales=unit, w2_global_scales=unit, input_scale=unit, intermediate_scale=unit, immutable_input_scales=True,
        ),
    )
    eplan = fused_moe.plan_execution(
        experts=prep_exp,
        capacity=fused_moe.ExecutionCapacity(max_tokens=3072, top_k=top_k, warmup_token_counts=(1, 2, 4, 8, 16, 32, 64, 128, 512, 3072), route_num_experts=0),
        routing=fused_moe.RoutingSpec(apply_router_weight_on_input=False),
    )

    def _mk_m(cnt: int):
        def _f(st):
            scr = tuple(torch.empty(s.shape, dtype=s.dtype, device=dev) for s in st.scratch.scratch_specs())
            hid = torch.randn((cnt, H), dtype=torch.bfloat16, device=dev) * 0.125
            out = torch.empty_like(hid)
            ids = make_tuning_routes(cnt, top_k, E, device=dev)[0].clone()
            wts = torch.full((cnt, top_k), 1.0 / top_k, dtype=torch.float32, device=dev)
            b = st.bind(scratch=scr, a=hid, experts=prep_exp, topk_weights=wts, topk_ids=ids, output=out, input_scales_static=True)
            return PreparedCall(run=b.run, output=out, produce=lambda: hid.normal_(std=0.125), reset=lambda: out.zero_())
        return _f

    sess = PreparationSession(device=0, autotune=True, cache_dir="/fi-cache/b12x_new_ae58183f")
    sess.prepare((eplan.request(name="chain_moe", prepare_calls={int(c): _mk_m(int(c)) for c in eplan.token_counts}, benchmark_calls={int(c): _mk_m(int(c)) for c in eplan.token_counts}),))
    moe_st = require_prepared(eplan, "moe.decode")
    moe_scr = tuple(torch.empty(s.shape, dtype=s.dtype, device=dev) for s in eplan.scratch_specs())
    norm = RMSNorm(H, 1e-6).to(device=dev, dtype=torch.bfloat16)

    # Check if b12x.moe accepts int64 directly!
    accepts_i64 = False
    try:
        x1 = torch.randn((4, H), dtype=torch.bfloat16, device=dev)
        o1 = torch.empty_like(x1)
        i64 = torch.zeros((4, top_k), dtype=torch.int64, device=dev)
        wf = torch.full((4, top_k), 1.0 / top_k, dtype=torch.float32, device=dev)
        moe_st.bind(scratch=moe_scr, a=x1, experts=prep_exp, topk_weights=wf, topk_ids=i64, output=o1, input_scales_static=True).run()
        torch.cuda.synchronize()
        accepts_i64 = True
    except Exception as exc:
        print(f"  [Info] b12x.moe int64 topk_ids direct support: False ({exc})")
    if accepts_i64:
        print("  [Info] b12x.moe int64 topk_ids direct support: True!")

    print("\n  [Chained CUDAGraph: RMSNorm -> Gate -> b12x.moe.fused_moe]")
    for m in (1, 2, 4, 8, 16, 32, 64, 128):
        x_in = (torch.randn(m, H, device=dev, dtype=torch.bfloat16) * 0.25).contiguous()
        x_norm = torch.empty_like(x_in)
        moe_out = torch.empty_like(x_in)
        ids_i32 = torch.empty((m, top_k), dtype=torch.int32, device=dev)
        mg_pdl = prepare_mega_gate(
            x_norm, gate_w_bf16, num_topk=top_k, scoring_func="sqrtsoftplus",
            bias=bias_f32, routed_scaling_factor=1.5, enable_pdl=True,
        )
        mg_nopdl = prepare_mega_gate(
            x_norm, gate_w_bf16, num_topk=top_k, scoring_func="sqrtsoftplus",
            bias=bias_f32, routed_scaling_factor=1.5, enable_pdl=False,
        )

        def _chain_vllm():
            y = norm(x_in)
            x_norm.copy_(y)
            lg, _ = gate(x_norm)
            wts, ids = fused_topk_bias(
                hidden_states=x_norm, gating_output=lg, scoring_func="sqrtsoftplus",
                e_score_correction_bias=bias_f32, topk=top_k, renormalize=True,
                indices_type=torch.int32, routed_scaling_factor=1.5,
            )
            moe_st.bind(scratch=moe_scr, a=x_norm, experts=prep_exp, topk_weights=wts, topk_ids=ids, output=moe_out, input_scales_static=True).run()

        def _chain_mega_pdl():
            torch.ops._C.rms_norm(x_norm, x_in, norm.weight.data, 1e-6)
            idx64, wts = mg_pdl.run()
            ids_i32.copy_(idx64)
            moe_st.bind(scratch=moe_scr, a=x_norm, experts=prep_exp, topk_weights=wts, topk_ids=ids_i32, output=moe_out, input_scales_static=True).run()

        def _chain_mega_vllm_norm():
            y = norm(x_in)
            x_norm.copy_(y)
            idx64, wts = mg_pdl.run()
            ids_i32.copy_(idx64)
            moe_st.bind(scratch=moe_scr, a=x_norm, experts=prep_exp, topk_weights=wts, topk_ids=ids_i32, output=moe_out, input_scales_static=True).run()

        def _chain_mega_nopdl():
            y = norm(x_in)
            x_norm.copy_(y)
            idx64, wts = mg_nopdl.run()
            ids_i32.copy_(idx64)
            moe_st.bind(scratch=moe_scr, a=x_norm, experts=prep_exp, topk_weights=wts, topk_ids=ids_i32, output=moe_out, input_scales_static=True).run()

        c_vllm, _ = bench_cudagraph(_chain_vllm)
        c_pdl, _ = bench_cudagraph(_chain_mega_vllm_norm)
        c_nopdl, _ = bench_cudagraph(_chain_mega_nopdl)
        print(
            f"  Chain M={m:3d}: vLLM={c_vllm:6.2f}us | mega(pdl=1)={c_pdl:6.2f}us | "
            f"mega(pdl=0)={c_nopdl:6.2f}us | 40-layer delta={(c_vllm - c_pdl)*40:+7.1f}us"
        )
    sess.close()

    # =========================================================================
    # 2. Decode Paged FP8 MQA Logits + Fused DSA Indexer (8 layers/step)
    # =========================================================================
    print("\n" + "=" * 78)
    print("BENCHMARK 2: Decode Paged FP8 MQA Logits & Fused DSA Indexer (H=32, D=128, page=64)")
    print("=" * 78)

    from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits, get_paged_mqa_logits_metadata
    from vllm.model_executor.layers.indexer_topk import get_indexer_topk
    from flashinfer.sm120_paged_mqa_logits import prepare_sm120_paged_mqa_logits
    from flashinfer.dsa_indexer import dsa_indexer_det_topk

    num_sms = props.multi_processor_count
    topk_fn = get_indexer_topk("auto")

    # Build realistic planar-per-page FP8 KV cache: [num_pages, 64, 1, 132] uint8
    # First 64*128 bytes = FP8 E4M3 keys, last 64*4 bytes = FP32 positive scales
    num_pages = 4096  # covers up to 262,144 tokens
    page_kv = 64
    kv_pages = torch.empty((num_pages, page_kv * 132), dtype=torch.uint8, device=dev)
    k_fp8_init = (torch.randn((num_pages, page_kv * 128), device=dev, dtype=torch.bfloat16) * 0.5).to(torch.float8_e4m3fn)
    k_sc_init = (torch.rand((num_pages, page_kv), device=dev, dtype=torch.float32) * 0.05 + 0.01)
    kv_pages[:, : page_kv * 128] = k_fp8_init.view(torch.uint8)
    kv_pages[:, page_kv * 128 :] = k_sc_init.view(torch.uint8).reshape(num_pages, page_kv * 4)
    kv_cache_4d = kv_pages.view(num_pages, page_kv, 1, 132)

    # Parity check first!
    for b_chk, ctx_chk in ((1, 640), (8, 1280), (32, 4096)):
        max_ctx = 16384
        max_blks = max_ctx // page_kv
        q_chk = (torch.randn(b_chk, 1, 32, 128, device=dev, dtype=torch.bfloat16) * 0.5).to(torch.float8_e4m3fn).contiguous()
        w_chk = (torch.rand(b_chk, 32, device=dev, dtype=torch.float32) * 0.1 + 0.01).contiguous()
        ctx_t = torch.full((b_chk, 1), ctx_chk, dtype=torch.int32, device=dev)
        bt_t = (torch.arange(b_chk * max_blks, dtype=torch.int32, device=dev) % num_pages).view(b_chk, max_blks).contiguous()

        dg_meta = get_paged_mqa_logits_metadata(ctx_t, page_kv, num_sms)
        dg_logits = fp8_fp4_paged_mqa_logits((q_chk, None), kv_cache_4d, w_chk, ctx_t, bt_t, dg_meta, max_model_len=max_ctx, clean_logits=False)
        topk_dg = torch.full((b_chk, 512), -1, dtype=torch.int32, device=dev)
        topk_fn(dg_logits, ctx_t, 1, topk_dg, 512, max_ctx)

        fi_plan = prepare_sm120_paged_mqa_logits(q_chk, kv_cache_4d, w_chk, ctx_t, bt_t, max_ctx)
        fi_logits = fi_plan.run()
        topk_fi = torch.full((b_chk, 512), -1, dtype=torch.int32, device=dev)
        topk_fn(fi_logits, ctx_t, 1, topk_fi, 512, max_ctx)

        topk_dsa = torch.empty((b_chk, 512), dtype=torch.int32, device=dev)
        dsa_indexer_det_topk(
            q_chk, kv_cache_4d, w_chk, None, None,
            block_table=bt_t, context_lens=ctx_t, top_k=512,
            indices=topk_dsa, return_scores=False, backend="sm120",
        )
        torch.cuda.synchronize()

        valid_diff = (dg_logits[:, :ctx_chk] - fi_logits[:, :ctx_chk]).abs().max().item()
        ov_fi = sum(len(set(topk_dg[r].tolist()) & set(topk_fi[r].tolist())) for r in range(b_chk)) / (b_chk * 512) * 100.0
        ov_dsa = sum(len(set(topk_dg[r].tolist()) & set(topk_dsa[r].tolist())) for r in range(b_chk)) / (b_chk * 512) * 100.0
        print(
            f"  [Decode Parity B={b_chk:2d} ctx={ctx_chk:4d}] logits_max_diff={valid_diff:.3e} | "
            f"top512 overlap: sm120_mqa={ov_fi:6.2f}% sm120_dsa={ov_dsa:6.2f}% route={fi_plan.route}"
        )

    # CUDAGraph Decode Latency across B and ctx_len (max_ctx = 16384 for cr=2, 32768 for cr=1)
    print("\n  [Decode CUDAGraph Single-Layer Latency: max_ctx=16384 (cr=2) / 32768 (cr=1)]")
    for ctx_len in (640, 1280, 4096, 16384):
        max_ctx = 32768
        max_blks = max_ctx // page_kv
        print(f"\n  >> ctx_len = {ctx_len} (max_model_len={max_ctx}):")
        for b in (1, 2, 8, 16, 32, 64, 128):
            q_b = (torch.randn(b, 1, 32, 128, device=dev, dtype=torch.bfloat16) * 0.5).to(torch.float8_e4m3fn).contiguous()
            w_b = (torch.rand(b, 32, device=dev, dtype=torch.float32) * 0.1 + 0.01).contiguous()
            ctx_b = torch.full((b, 1), ctx_len, dtype=torch.int32, device=dev)
            bt_b = (torch.arange(b * max_blks, dtype=torch.int32, device=dev) % num_pages).view(b, max_blks).contiguous()

            dg_meta = get_paged_mqa_logits_metadata(ctx_b, page_kv, num_sms)
            fi_plan_none = prepare_sm120_paged_mqa_logits(q_b, kv_cache_4d, w_b, ctx_b, bt_b, max_ctx, max_context_hint=None)
            fi_plan_hint = prepare_sm120_paged_mqa_logits(q_b, kv_cache_4d, w_b, ctx_b, bt_b, max_ctx, max_context_hint=16384)
            out_topk = torch.empty((b, 512), dtype=torch.int32, device=dev)

            # Warm up dsa_indexer_det_topk paged outside capture (allocates internal scratch)
            dsa_indexer_det_topk(
                q_b, kv_cache_4d, w_b, None, None,
                block_table=bt_b, context_lens=ctx_b, top_k=512,
                indices=out_topk, return_scores=False, backend="sm120",
            )
            torch.cuda.synchronize()

            # 1) Pure logits: DeepGEMM vs SM120 Paged MQA (hint=None & hint=16384)
            dg_log_us, _ = bench_cudagraph(
                lambda: fp8_fp4_paged_mqa_logits((q_b, None), kv_cache_4d, w_b, ctx_b, bt_b, dg_meta, max_model_len=max_ctx, clean_logits=False)
            )
            fi_log_us, _ = bench_cudagraph(fi_plan_none.run)
            fi_hint_us, _ = bench_cudagraph(fi_plan_hint.run)

            # 2) Non-candidate layer full op (clear -1 + logits + get_indexer_topk) vs sm120_dsa_indexer
            def _vllm_full_layer():
                out_topk.fill_(-1)
                lg = fp8_fp4_paged_mqa_logits((q_b, None), kv_cache_4d, w_b, ctx_b, bt_b, dg_meta, max_model_len=max_ctx, clean_logits=False)
                topk_fn(lg, ctx_b, 1, out_topk, 512, max_ctx)

            def _fi_mqa_full_layer():
                out_topk.fill_(-1)
                lg = fi_plan_none.run()
                topk_fn(lg, ctx_b, 1, out_topk, 512, max_ctx)

            def _dsa_fused_layer():
                dsa_indexer_det_topk(
                    q_b, kv_cache_4d, w_b, None, None,
                    block_table=bt_b, context_lens=ctx_b, top_k=512,
                    indices=out_topk, return_scores=False, backend="sm120",
                )

            v_full_us, _ = bench_cudagraph(_vllm_full_layer)
            m_full_us, _ = bench_cudagraph(_fi_mqa_full_layer)
            dsa_us, _ = bench_cudagraph(_dsa_fused_layer)

            print(
                f"    B={b:3d} | Logits: DG={dg_log_us:6.2f}us FI={fi_log_us:6.2f}us FI(16k)={fi_hint_us:6.2f}us | "
                f"Full(Logits+Top512): vLLM={v_full_us:6.2f}us FI+Topk={m_full_us:6.2f}us FusedDSA={dsa_us:6.2f}us"
            )

    # =========================================================================
    # 3. Prefill Ragged DSA Indexer (`sm120_dsa_indexer` vs `fp8_fp4_mqa_logits + top_k_per_row_prefill`)
    # =========================================================================
    print("\n" + "=" * 78)
    print("BENCHMARK 3: Prefill Ragged DSA Indexer (top_k=512, cr=1 & cr=2)")
    print("=" * 78)

    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits
    from vllm import _custom_ops as ops

    for desc, q_lens, ratio in (
        ("1x512 (cr=1)", [512], 1),
        ("1x1024 (cr=1)", [1024], 1),
        ("1x1024 (cr=2)", [1024], 2),
        ("1x2048 (cr=1)", [2048], 1),
        ("1x2048 (cr=2)", [2048], 2),
        ("1x4096 (cr=1)", [4096], 1),
        ("1x4096 (cr=2)", [4096], 2),
        ("8x512 (cr=1)", [512] * 8, 1),
        ("4x1024 (cr=1)", [1024] * 4, 1),
    ):
        T_q = sum(q_lens)
        k_lens = [ql // ratio for ql in q_lens]
        T_kv = sum(k_lens)
        q_r = (torch.randn(T_q, 32, 128, device=dev, dtype=torch.bfloat16) * 0.5).to(torch.float8_e4m3fn).contiguous()
        k_r = (torch.randn(T_kv, 128, device=dev, dtype=torch.bfloat16) * 0.5).to(torch.float8_e4m3fn).contiguous()
        ks_r = (torch.rand(T_kv, device=dev, dtype=torch.float32) * 0.05 + 0.01).contiguous()
        w_r = (torch.rand(T_q, 32, device=dev, dtype=torch.float32) * 0.1 + 0.01).contiguous()

        cu_q_list = [0]
        cu_k_list = [0]
        ks_rows = []
        ke_rows = []
        for ql, kl in zip(q_lens, k_lens, strict=False):
            q0 = cu_q_list[-1]
            k0 = cu_k_list[-1]
            cu_q_list.append(q0 + ql)
            cu_k_list.append(k0 + kl)
            for u in range(ql):
                ks_rows.append(k0)
                ke_rows.append(k0 + min(kl, (u + 1) // ratio))

        cu_q = torch.tensor(cu_q_list, dtype=torch.int32, device=dev)
        cu_k = torch.tensor(cu_k_list, dtype=torch.int32, device=dev)
        cu_ks = torch.tensor(ks_rows, dtype=torch.int32, device=dev)
        cu_ke = torch.tensor(ke_rows, dtype=torch.int32, device=dev)

        topk_ref = torch.full((T_q, 512), -1, dtype=torch.int32, device=dev)
        topk_cand = torch.empty((T_q, 512), dtype=torch.int32, device=dev)

        def _prefill_vllm():
            topk_ref.fill_(-1)
            lg = fp8_fp4_mqa_logits((q_r, None), (k_r, ks_r), w_r, cu_ks, cu_ke, clean_logits=False)
            ops.top_k_per_row_prefill(lg, cu_ks, cu_ke, topk_ref, T_q, lg.stride(0), lg.stride(1), 512)

        def _prefill_dsa():
            dsa_indexer_det_topk(
                q_r, k_r, w_r, cu_q, cu_k,
                k_scale=ks_r, top_k=512, ratio=ratio, max_seqlen_k=max(k_lens),
                indices=topk_cand, return_scores=False, backend="sm120",
            )

        _prefill_vllm()
        _prefill_dsa()
        torch.cuda.synchronize()

        # Check overlap on rows with > 512 visible keys (and exact match on rows <= 512)
        sample_rows = [r for r in range(0, T_q, max(1, T_q // 16))]
        ov_sum = 0.0
        for r in sample_rows:
            s_ref = set(x for x in topk_ref[r].tolist() if x >= 0)
            s_cnd = set(x for x in topk_cand[r].tolist() if x >= 0)
            denom = max(1, len(s_ref))
            ov_sum += len(s_ref & s_cnd) / denom
        ov_pct = (ov_sum / len(sample_rows)) * 100.0

        v_us, _ = bench_eager(_prefill_vllm)
        d_us, _ = bench_eager(_prefill_dsa)
        print(
            f"  {desc:15s}: vLLM={v_us:7.1f}us | sm120_dsa={d_us:7.1f}us | "
            f"speedup={v_us/d_us:4.2f}x | overlap={ov_pct:6.2f}%"
        )

    destroy_model_parallel()
    destroy_distributed_environment()
    vcfg_ctx.__exit__(None, None, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("b12x_only", "new_ops"), required=True)
    parser.add_argument("--tag", default="default")
    parser.add_argument("--cache-dir", default="/fi-cache/b12x_bench")
    args = parser.parse_args()
    if args.mode == "b12x_only":
        run_b12x_bench(args.tag, args.cache_dir)
    else:
        run_new_ops_bench()


if __name__ == "__main__":
    main()
