#!/usr/bin/env python3
# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT

"""Reproducible SM120 (`RTX 6000D`, 156 SMs) operator & numerical parity audit for DeepSeek-V4.1.

Verifies and records raw single-GPU measurements for all Phase 3.7 operator decisions:
1. All 7 Block32 `[32, 32] UE8M0` 2D FP8 linear shapes (including `5120 x 288`
   `shared_experts.down_proj`): numerical parity (`cos_sim`, `max_abs_diff`), CUDAGraph
   latency (`M = 1..128`), and Eager latency (`M = 256..3072`) comparing `b12x` fast-bind,
   `b12x` standard `bind()`, and `FlashInferCutlassMxfp8LinearKernel`.
2. `PreparationSession` premature `close()` vs retained session: verifies that closing
   `PreparationSession` immediately after `prepare()` clears `plan._prepared` and triggers
   runtime fallback to un-autotuned default configurations.
3. `B12X_W4A8_TINY_DECODE=1` vs `0` with isolated fresh autotune caches: proves that
   `MoETinyDecodeKernelBackend` (`tiny_decode.py`) omits `swiglu_limit=10.0` (`cos_sim ~ 0.88`,
   `max_abs_diff > 1000` on `M = 1..4`) despite lower `M = 1` latency.
4. `dsv41_o_proj` (`x40` layers): compares current `deep_gemm_fp8_o_proj` (`BF16 bmm wo_a` +
   `b12x wo_b`) against 2-step `b12x` and fused `b12x.gemm.wo_projection` across `M = 1..128`
   CUDAGraph and `M = 256..3072` Eager.
5. `ParallelLMHead` (`16160 x 5120` BF16 `TP=8` shard) `cuBLAS` vs `b12x.bf16_vocab_projection`,
   and `vLLM` TileLang `mhc_shifted_post_pre` (`x80` calls/step) vs `b12x.norm.mhc`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import torch
import torch.nn.functional as F

DEFAULT_COMPAT_DIR = (
    "/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/_compat"
)

DSV41_TP8_FP8_SHAPES: tuple[tuple[str, int, int, int], ...] = (
    ("fused_wqa_wkv", 1792, 5120, 40),
    ("wq_b", 4096, 1280, 32),
    ("indexer_wq_b", 512, 1280, 8),
    ("wo_b", 5120, 1024, 40),
    ("shared_gate_up", 576, 5120, 40),
    ("shared_down_288", 5120, 288, 40),
    ("engram_wkv", 3200, 6144, 2),
)


def _setup_b12x_path(compat_dir: str) -> None:
    if compat_dir and os.path.isdir(compat_dir) and compat_dir not in sys.path:
        sys.path.insert(0, compat_dir)
    cur = os.environ.get("PYTHONPATH", "")
    parts = [p for p in cur.split(os.pathsep) if p]
    if compat_dir and os.path.isdir(compat_dir) and compat_dir not in parts:
        os.environ["PYTHONPATH"] = f"{compat_dir}{os.pathsep}{cur}" if cur else compat_dir

    import b12x

    b12x_file = getattr(b12x, "__file__", None)
    if b12x_file:
        parent = str(Path(b12x_file).resolve().parent.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        parts2 = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        if parent not in parts2:
            cur2 = os.environ.get("PYTHONPATH", "")
            os.environ["PYTHONPATH"] = f"{parent}{os.pathsep}{cur2}" if cur2 else parent


def _align_up(val: int, align: int = 1024) -> int:
    return ((val + align - 1) // align) * align


def _alloc_scratch(specs: Any, device: torch.device) -> list[torch.Tensor]:
    total = 0
    for sp in specs:
        elem = torch.empty((), dtype=sp.dtype).element_size()
        total = _align_up(total, 1024) + int(math.prod(sp.shape)) * elem
    arena = torch.empty(max(1024, total), dtype=torch.uint8, device=device)
    views: list[torch.Tensor] = []
    off = 0
    for sp in specs:
        off = _align_up(off, 1024)
        elem = torch.empty((), dtype=sp.dtype).element_size()
        nb = int(math.prod(sp.shape)) * elem
        views.append(arena[off : off + nb].view(sp.dtype).reshape(sp.shape))
        off += nb
    return views


def _bench_cudagraph_us(
    fn: Callable[[], Any], warmup: int = 10, iters: int = 100
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
        with torch.cuda.graph(graph):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        graph.replay()
    ev1.record()
    torch.cuda.synchronize()
    res = (ev0.elapsed_time(ev1) * 1000.0) / iters
    del graph
    return res


def _bench_eager_us(
    fn: Callable[[], Any], warmup: int = 8, iters: int = 40
) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    ev0.record()
    for _ in range(iters):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    wall_us = ((time.perf_counter() - t0) * 1e6) / iters
    gpu_us = (ev0.elapsed_time(ev1) * 1000.0) / iters
    return wall_us, gpu_us


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float().flatten()
    bf = b.detach().float().flatten()
    return float(F.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item())


def _make_mxfp8_weight(
    out_f: int, in_f: int, device: torch.device, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w_f32 = torch.randn((out_f, in_f), generator=gen, dtype=torch.float32).to(device) * 0.5
    w_fp8 = w_f32.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
    s_32 = torch.randint(
        122, 129, (out_f // 32, in_f // 32), generator=gen, dtype=torch.int64
    ).to(device=device, dtype=torch.uint8)
    s_row = s_32.repeat_interleave(32, dim=0).contiguous()
    return w_fp8, s_32, s_row


def _query_gpu_provenance(device_idx: int) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device_idx)
    info: dict[str, Any] = {
        "device_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "multi_processor_count": int(props.multi_processor_count),
        "torch_total_memory_mib": round(props.total_memory / (1024 * 1024), 2),
        "torch_version": torch.__version__,
        "git_sha": os.environ.get("ZSTEP_GIT_SHA", "unknown"),
    }
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device_idx}",
                "--query-gpu=driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) == 2:
            info["nvidia_driver_version"] = parts[0]
            info["nvidia_smi_memory_total_mib"] = int(parts[1])
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        info["nvidia_smi_query_error"] = str(exc)
    return info


def audit_block32_fp8_and_session(
    device: torch.device, cache_dir: str, sessions: list[Any]
) -> dict[str, Any]:
    from b12x.gemm import block_fp8_linear
    from b12x.gemm._shared.block_fp8 import (
        _SCRATCH_ALIGN_BYTES,
        BlockFP8LinearBinding,
        _block_fp8_linear_scratch_layout,
        _block_fp8_linear_x_q_from_scratch,
    )
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.preparation.types import require_prepared
    from vllm.config.quantization import QuantSpec
    from vllm.model_executor.layers.quantization.modelopt import (
        CkptCtx,
        ModelOptLinearMethod,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
        kMxfp8Static,
    )

    cg_ms = (1, 8, 32, 64, 128)
    eager_ms = (256, 512, 1024, 2048, 3072)
    all_ms = cg_ms + eager_ms

    # 1. Verify premature PreparationSession.close() clears _prepared on a probe plan.
    probe_w, probe_s32, _ = _make_mxfp8_weight(1792, 5120, device, seed=101)
    probe_pw = block_fp8_linear.pack_weight(
        probe_w, probe_s32.view(torch.float8_e8m0fnu), block_size=(32, 32)
    )
    probe_plan = block_fp8_linear.plan(
        block_fp8_linear.Caps(
            device=device,
            max_tokens=64,
            in_features=5120,
            out_features=1792,
            block_size=(32, 32),
        )
    )

    def _mk_probe(state: Any) -> PreparedCall:
        src = torch.randn((64, 5120), dtype=torch.bfloat16, device=device) * 0.25
        out = torch.empty((64, 1792, 1), dtype=torch.bfloat16, device=device)
        scr = _alloc_scratch(state.scratch.scratch_specs(), device)
        bnd = state.bind(scratch=scr, source=src, packed_weight=probe_pw, output=out)
        return PreparedCall(
            run=lambda: state.run_binding(bnd), produce=lambda: src.normal_(std=0.25)
        )

    probe_sess = PreparationSession(device=0, autotune=True, cache_dir=cache_dir)
    probe_sess.prepare(
        (probe_plan.request(name="probe_close", prepare_call=_mk_probe, benchmark_call=_mk_probe),)
    )
    prepared_while_open = getattr(probe_plan, "_prepared", None) is not None
    probe_sess.close()
    prepared_after_close = getattr(probe_plan, "_prepared", None) is not None

    # 2. Prepare all 7 shapes with retained session + autotune=True.
    packed_weights: dict[str, Any] = {}
    cutlass_layers: dict[str, tuple[torch.nn.Module, Any]] = {}
    plans: dict[str, dict[int, Any]] = {}
    reqs: list[Any] = []

    for idx, (name, out_f, in_f, _cnt) in enumerate(DSV41_TP8_FP8_SHAPES):
        w_fp8, s_32, s_row = _make_mxfp8_weight(out_f, in_f, device, seed=200 + idx)
        pw = block_fp8_linear.pack_weight(
            w_fp8, s_32.view(torch.float8_e8m0fnu), block_size=(32, 32)
        )
        packed_weights[name] = pw

        qm = ModelOptLinearMethod(
            QuantSpec(weight=kMxfp8Static, activation=kMxfp8Dynamic),
            CkptCtx(scale_block_size=(32, 32)),
        )
        clayer = torch.nn.Module()
        qm.create_weights(clayer, in_f, [out_f], in_f, out_f, torch.bfloat16)
        clayer = clayer.to(device)
        clayer.weight.data.copy_(w_fp8)
        if clayer.weight_scale.shape == s_row.shape:
            clayer.weight_scale.data.copy_(s_row)
        else:
            clayer.weight_scale.data.copy_(s_32)
        qm.process_weights_after_loading(clayer)
        cutlass_layers[name] = (clayer, qm.kernel)

        plans[name] = {}
        for cap in all_ms:
            p = block_fp8_linear.plan(
                block_fp8_linear.Caps(
                    device=device,
                    max_tokens=cap,
                    in_features=in_f,
                    out_features=out_f,
                    block_size=(32, 32),
                )
            )
            plans[name][cap] = p

            def _make_call(
                _pw: Any = pw, _cap: int = cap, _in_f: int = in_f, _out_f: int = out_f
            ) -> Callable[[Any], PreparedCall]:
                def _call(state: Any) -> PreparedCall:
                    src = (
                        torch.randn((_cap, _in_f), dtype=torch.bfloat16, device=device)
                        * 0.25
                    )
                    out = torch.empty(
                        (_cap, _out_f, 1), dtype=torch.bfloat16, device=device
                    )
                    scr = _alloc_scratch(state.scratch.scratch_specs(), device)
                    bnd = state.bind(
                        scratch=scr, source=src, packed_weight=_pw, output=out
                    )
                    return PreparedCall(
                        run=lambda: state.run_binding(bnd),
                        produce=lambda: src.normal_(std=0.25),
                    )

                return _call

            cb = _make_call()
            reqs.append(
                p.request(
                    name=f"audit_{name}_m{cap}", prepare_call=cb, benchmark_call=cb
                )
            )

    sess = PreparationSession(device=0, autotune=True, cache_dir=cache_dir)
    sess.prepare(tuple(reqs))
    sessions.append(sess)

    shapes_out: dict[str, Any] = {}
    for name, out_f, in_f, count in DSV41_TP8_FP8_SHAPES:
        pw = packed_weights[name]
        clayer, ck = cutlass_layers[name]
        entry: dict[str, Any] = {
            "out_features": out_f,
            "in_features": in_f,
            "layers": count,
            "cg_us": {},
            "eager_us": {},
        }

        x_p = torch.randn((256, in_f), dtype=torch.bfloat16, device=device) * 0.25
        out_p = torch.empty((256, out_f, 1), dtype=torch.bfloat16, device=device)
        scr_p = _alloc_scratch(plans[name][256].scratch_specs(), device)
        block_fp8_linear.run(
            binding=block_fp8_linear.bind(
                plans[name][256],
                scratch=scr_p,
                source=x_p,
                packed_weight=pw,
                output=out_p,
            )
        )
        ref_p = ck.apply_weights(layer=clayer, x=x_p, bias=None)
        entry["parity_m256"] = {
            "cos_sim": round(_cos_sim(out_p.squeeze(-1), ref_p), 6),
            "max_abs_diff": round(
                float((out_p.squeeze(-1).float() - ref_p.float()).abs().max().item()), 6
            ),
        }

        for m in cg_ms:
            xm = torch.randn((m, in_f), dtype=torch.bfloat16, device=device) * 0.25
            outm = torch.empty((m, out_f, 1), dtype=torch.bfloat16, device=device)
            scrm = _alloc_scratch(plans[name][m].scratch_specs(), device)
            bndm = block_fp8_linear.bind(
                plans[name][m], scratch=scrm, source=xm, packed_weight=pw, output=outm
            )
            b12x_us = _bench_cudagraph_us(
                lambda _b=bndm: block_fp8_linear.run(binding=_b)
            )
            cut_us = _bench_cudagraph_us(
                lambda _c=ck, _l=clayer, _x=xm: _c.apply_weights(
                    layer=_l, x=_x, bias=None
                )
            )
            entry["cg_us"][str(m)] = {
                "b12x_us": round(b12x_us, 2),
                "cutlass_us": round(cut_us, 2),
                "saved_step_us": round((cut_us - b12x_us) * count, 1),
            }

        for m in eager_ms:
            xm = torch.randn((m, in_f), dtype=torch.bfloat16, device=device) * 0.25
            pm = plans[name][m]
            st = require_prepared(pm, "gemm.block_fp8_linear")
            scr = _alloc_scratch(pm.scratch_specs(), device)
            scr_u8 = scr[0]
            splan = st.scratch
            ws = None
            if splan.workspace_nbytes:
                off = _align_up(
                    _block_fp8_linear_scratch_layout(
                        tokens=splan.caps.max_tokens,
                        in_features=splan.caps.in_features,
                        out_features=splan.caps.out_features,
                        output_dtype=splan.caps.output_dtype,
                    ).nbytes,
                    _SCRATCH_ALIGN_BYTES,
                )
                ws = scr_u8.narrow(0, off, splan.workspace_nbytes).view(torch.float32)
            exact_xq = _block_fp8_linear_x_q_from_scratch(
                scr_u8, tokens=m, in_features=in_f, output_dtype=torch.bfloat16
            )

            def _run_fast(
                _m: int = m,
                _out_f: int = out_f,
                _pm: Any = pm,
                _xm: torch.Tensor = xm,
                _pw: Any = pw,
                _xq: Any = exact_xq,
                _ws: Any = ws,
                _tiler: Any = splan.mma_tiler_mn,
                _st: Any = st,
            ) -> torch.Tensor:
                out = torch.empty((_m, _out_f, 1), dtype=torch.bfloat16, device=device)
                bnd = BlockFP8LinearBinding(
                    plan=_pm,
                    source=_xm,
                    packed_weight=_pw,
                    x_q=_xq,
                    output=out,
                    workspace=_ws,
                    bias=None,
                    expected_m=_m,
                    mma_tiler_mn=_tiler,
                )
                _st.run_binding(bnd)
                return out

            def _run_std(
                _m: int = m,
                _out_f: int = out_f,
                _pm: Any = pm,
                _scr: Any = scr,
                _xm: torch.Tensor = xm,
                _pw: Any = pw,
            ) -> torch.Tensor:
                out = torch.empty((_m, _out_f, 1), dtype=torch.bfloat16, device=device)
                bnd = block_fp8_linear.bind(
                    _pm, scratch=_scr, source=_xm, packed_weight=_pw, output=out
                )
                block_fp8_linear.run(binding=bnd)
                return out

            fast_wall, fast_gpu = _bench_eager_us(_run_fast)
            std_wall, _ = _bench_eager_us(_run_std)
            cut_wall, cut_gpu = _bench_eager_us(
                lambda _c=ck, _l=clayer, _x=xm: _c.apply_weights(
                    layer=_l, x=_x, bias=None
                )
            )
            entry["eager_us"][str(m)] = {
                "b12x_fast_wall_us": round(fast_wall, 2),
                "b12x_std_bind_wall_us": round(std_wall, 2),
                "b12x_gpu_us": round(fast_gpu, 2),
                "cutlass_wall_us": round(cut_wall, 2),
                "cutlass_gpu_us": round(cut_gpu, 2),
                "winner": "cutlass" if cut_wall < fast_wall else "b12x_fast",
            }

        shapes_out[name] = entry

    return {
        "session_lifecycle_probe": {
            "prepared_while_session_open": prepared_while_open,
            "prepared_after_session_close": prepared_after_close,
        },
        "shapes": shapes_out,
    }


def audit_moe_tiny_decode_landmine(
    device: torch.device, sessions: list[Any]
) -> dict[str, Any]:
    from b12x.moe import fused_moe
    from b12x.moe.fused_moe.workloads import make_tuning_routes
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.preparation.types import require_prepared

    num_experts, top_k, hidden, inter = 384, 6, 5120, 288
    gen = torch.Generator(device="cpu").manual_seed(999)
    w13 = torch.randint(
        0, 256, (num_experts, 2 * inter, hidden // 2), generator=gen, dtype=torch.int64
    ).to(device=device, dtype=torch.uint8)
    w2 = torch.randint(
        0, 256, (num_experts, hidden, inter // 2), generator=gen, dtype=torch.int64
    ).to(device=device, dtype=torch.uint8)
    w13_s = torch.randint(
        120,
        127,
        (num_experts, 2 * inter, hidden // 32),
        generator=gen,
        dtype=torch.int64,
    ).to(device=device, dtype=torch.uint8)
    w2_s = torch.randint(
        120, 127, (num_experts, hidden, inter // 32), generator=gen, dtype=torch.int64
    ).to(device=device, dtype=torch.uint8)
    unit = torch.ones(num_experts, dtype=torch.float32, device=device)

    warm_counts = (1, 2, 4, 8, 16, 32)
    states: dict[str, Any] = {}
    scratches: dict[str, tuple[torch.Tensor, ...]] = {}
    preps: dict[str, Any] = {}
    out: dict[str, Any] = {
        "swiglu_limit": 10.0,
        "selected_backends": {},
        "cg_us": {},
        "parity_tiny1_vs_tiny0": {},
    }

    tmp_root = Path(tempfile.mkdtemp(prefix="zstep_audit_moe_"))
    prev_env = os.environ.get("B12X_W4A8_TINY_DECODE")
    try:
        for tiny_env in ("0", "1"):
            cdir = str(tmp_root / f"t{tiny_env}")
            os.makedirs(cdir, exist_ok=True)
            os.environ["B12X_W4A8_TINY_DECODE"] = tiny_env

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
                geometry=fused_moe.MoEGeometry(
                    num_experts=num_experts,
                    hidden_size=hidden,
                    intermediate_size=inter,
                ),
            )
            prep = fused_moe.prepare_weights(
                plan=wplan,
                weights=fused_moe.PackedWeights(
                    w13=w13.clone(),
                    w2=w2.clone(),
                    w13_block_scales=w13_s.clone(),
                    w2_block_scales=w2_s.clone(),
                    w13_global_scales=unit,
                    w2_global_scales=unit,
                    input_scale=unit,
                    intermediate_scale=unit,
                    immutable_input_scales=True,
                ),
            )
            eplan = fused_moe.plan_execution(
                experts=prep,
                capacity=fused_moe.ExecutionCapacity(
                    max_tokens=512,
                    top_k=top_k,
                    warmup_token_counts=warm_counts,
                    route_num_experts=0,
                ),
                routing=fused_moe.RoutingSpec(apply_router_weight_on_input=False),
            )

            def _mk_moe(cnt: int, _p: Any = prep) -> Callable[[Any], PreparedCall]:
                def _fac(state: Any) -> PreparedCall:
                    scr = tuple(_alloc_scratch(state.scratch.scratch_specs(), device))
                    hid = (
                        torch.randn((cnt, hidden), dtype=torch.bfloat16, device=device)
                        * 0.125
                    )
                    o = torch.empty_like(hid)
                    ids = make_tuning_routes(cnt, top_k, num_experts, device=device)[
                        0
                    ].clone()
                    wts = torch.softmax(
                        torch.randn((cnt, top_k), dtype=torch.float32, device=device),
                        dim=-1,
                    )
                    bnd = state.bind(
                        scratch=scr,
                        a=hid,
                        experts=_p,
                        topk_weights=wts,
                        topk_ids=ids,
                        output=o,
                        input_scales_static=True,
                    )
                    return PreparedCall(
                        run=bnd.run,
                        output=o,
                        produce=lambda: hid.normal_(std=0.125),
                        reset=lambda: o.zero_(),
                    )

                return _fac

            calls = {c: _mk_moe(int(c)) for c in eplan.token_counts}
            sess = PreparationSession(device=0, autotune=True, cache_dir=cdir)
            sess.prepare(
                (
                    eplan.request(
                        name=f"audit_moe_t{tiny_env}",
                        prepare_calls=calls,
                        benchmark_calls=calls,
                    ),
                )
            )
            sessions.append(sess)
            states[tiny_env] = require_prepared(eplan, "moe.decode")
            scratches[tiny_env] = tuple(_alloc_scratch(eplan.scratch_specs(), device))
            preps[tiny_env] = prep
            sel: dict[str, str] = {}
            for tc, child in eplan.variants.items():
                if child._prepared is not None:
                    sel[str(tc)] = str(getattr(child._prepared.state, "config", ""))
            out["selected_backends"][f"tiny_{tiny_env}"] = {
                k: sel.get(k, "") for k in ("1", "2", "4", "8")
            }

        for tiny_env in ("0", "1"):
            os.environ["B12X_W4A8_TINY_DECODE"] = tiny_env
            cg: dict[str, float] = {}
            for m in (1, 2, 4, 8, 16, 32):
                hid = (
                    torch.randn((m, hidden), dtype=torch.bfloat16, device=device)
                    * 0.125
                )
                o = torch.empty_like(hid)
                ids = make_tuning_routes(m, top_k, num_experts, device=device)[
                    0
                ].clone()
                wts = torch.full(
                    (m, top_k), 1.0 / top_k, dtype=torch.float32, device=device
                )
                bnd = states[tiny_env].bind(
                    scratch=scratches[tiny_env],
                    a=hid,
                    experts=preps[tiny_env],
                    topk_weights=wts,
                    topk_ids=ids,
                    output=o,
                    input_scales_static=True,
                )
                cg[str(m)] = round(_bench_cudagraph_us(bnd.run), 2)
            out["cg_us"][f"tiny_{tiny_env}"] = cg

        gen_in = torch.Generator(device="cpu").manual_seed(20261113)
        for m_test in (1, 2, 3, 4):
            hid = (
                torch.randn(
                    (m_test, hidden), generator=gen_in, dtype=torch.float32
                ).to(device=device, dtype=torch.bfloat16)
                * 0.25
            )
            ids = make_tuning_routes(m_test, top_k, num_experts, device=device)[
                0
            ].clone()
            wts = torch.softmax(
                torch.randn((m_test, top_k), generator=gen_in, dtype=torch.float32).to(
                    device
                ),
                dim=-1,
            )

            os.environ["B12X_W4A8_TINY_DECODE"] = "0"
            out0 = torch.empty_like(hid)
            states["0"].bind(
                scratch=scratches["0"],
                a=hid,
                experts=preps["0"],
                topk_weights=wts,
                topk_ids=ids,
                output=out0,
                input_scales_static=True,
            ).run()

            os.environ["B12X_W4A8_TINY_DECODE"] = "1"
            out1 = torch.empty_like(hid)
            states["1"].bind(
                scratch=scratches["1"],
                a=hid,
                experts=preps["1"],
                topk_weights=wts,
                topk_ids=ids,
                output=out1,
                input_scales_static=True,
            ).run()
            torch.cuda.synchronize()

            cs = _cos_sim(out1, out0)
            md = float((out1.float() - out0.float()).abs().max().item())
            rl2 = float(
                (out1.float() - out0.float()).norm().item()
                / max(out0.float().norm().item(), 1e-8)
            )
            out["parity_tiny1_vs_tiny0"][str(m_test)] = {
                "cos_sim": round(cs, 6),
                "max_abs_diff": round(md, 4),
                "rel_l2": round(rl2, 6),
            }
    finally:
        if prev_env is None:
            os.environ["B12X_W4A8_TINY_DECODE"] = "0"
        else:
            os.environ["B12X_W4A8_TINY_DECODE"] = prev_env
        shutil.rmtree(tmp_root, ignore_errors=True)

    return out


def audit_oproj_lmhead_and_mhc(
    device: torch.device, cache_dir: str, sessions: list[Any]
) -> dict[str, Any]:
    from b12x.gemm import bf16_vocab_projection, block_fp8_linear, wo_projection
    from b12x.norm import mhc as b12x_mhc
    from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall
    from b12x.preparation.types import require_prepared
    from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
        deep_gemm_fp8_o_proj,
        fused_inv_rope_fp8_quant,
    )
    from vllm.models.deepseek_v41.nvidia.model import mhc_shifted_post_pre

    n_groups, heads_per_group, nope_dim, rope_dim = 1, 8, 448, 64
    head_dim = nope_dim + rope_dim
    group_width = heads_per_group * head_dim
    rank, hidden, vocab_tp8 = 1024, 5120, 16160

    wo_a_2d, wo_a_s32_2d, wo_a_srow_2d = _make_realistic_pair(
        rank, group_width, device, seed=777
    )
    wo_b_fp8, wo_b_s32, _ = _make_mxfp8_weight(hidden, rank, device, seed=778)
    angles = torch.arange(4096, dtype=torch.float32, device=device).unsqueeze(1) * (
        1.0
        / (
            10000.0
            ** (
                torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device)
                / rope_dim
            )
        )
    )
    cos_sin_cache = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1).contiguous()

    scale_f32 = (2.0 ** (wo_a_srow_2d.float() - 127.0)).repeat_interleave(32, dim=1)
    wo_a_bf16 = (wo_a_2d.float() * scale_f32).to(torch.bfloat16).unsqueeze(0).contiguous()
    wo_a_mod_bf16 = torch.nn.Module()
    wo_a_mod_bf16.weight = torch.nn.Parameter(wo_a_bf16, requires_grad=False)

    wo_weights = wo_projection.pack_weights(
        wo_a_2d,
        wo_a_s32_2d.view(torch.float8_e8m0fnu),
        wo_b_fp8,
        wo_b_s32.view(torch.float8_e8m0fnu),
        groups=n_groups,
        group_width=group_width,
        rank=rank,
        hidden=hidden,
        block_size=(32, 32),
    )
    wo_a_pw_2d = block_fp8_linear.pack_weight(
        wo_a_2d, wo_a_s32_2d.view(torch.float8_e8m0fnu), block_size=(32, 32)
    )
    wo_b_pw = block_fp8_linear.pack_weight(
        wo_b_fp8, wo_b_s32.view(torch.float8_e8m0fnu), block_size=(32, 32)
    )
    lm_w = (
        torch.randn((vocab_tp8, hidden), dtype=torch.bfloat16, device=device) * 0.02
    ).contiguous()

    cg_ms = (1, 4, 8, 16, 32, 64, 128)
    eager_ms = (512, 1024, 3072)
    all_ms = cg_ms + eager_ms
    wo_plans: dict[int, Any] = {}
    woa_plans: dict[int, Any] = {}
    wob_plans: dict[int, Any] = {}
    lm_plans: dict[int, Any] = {}
    reqs: list[Any] = []

    for cap in all_ms:
        inv = FrozenMapping(
            {
                "operation": "inv_rope",
                "heads_per_group": heads_per_group,
                "nope_dim": nope_dim,
                "rope_dim": rope_dim,
                "positions_dtype": "int64",
                "cos_sin_dtype": "float32",
                "dynamic_tokens": cap > 128,
            }
        )
        wp = wo_projection.plan(
            wo_projection.Caps(
                device=device,
                max_tokens=cap,
                groups=n_groups,
                group_width=group_width,
                rank=rank,
                hidden=hidden,
            ),
            invocation=inv,
        )
        wo_plans[cap] = wp

        def _mk_wo(
            _cap: int = cap, _wp: Any = wp
        ) -> Callable[[Any], PreparedCall]:
            def _cb(state: Any) -> PreparedCall:
                o = (
                    torch.randn(
                        (_cap, heads_per_group, head_dim),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    * 0.2
                )
                pos = torch.arange(_cap, dtype=torch.int64, device=device)
                scr = _alloc_scratch(_wp.scratch_specs(), device)
                bnd = state.bind_inv_rope(
                    scratch=scr,
                    o=o,
                    positions=pos,
                    cos_sin_cache=cos_sin_cache,
                    weights=wo_weights,
                    heads_per_group=heads_per_group,
                    nope_dim=nope_dim,
                    rope_dim=rope_dim,
                )
                return PreparedCall(
                    run=lambda: state.run_inv_rope(bnd),
                    produce=lambda: o.normal_(std=0.2),
                )

            return _cb

        wo_cb = _mk_wo()
        reqs.append(
            wp.request(name=f"audit_wo_{cap}", prepare_call=wo_cb, benchmark_call=wo_cb)
        )

        ap = block_fp8_linear.plan(
            block_fp8_linear.Caps(
                device=device,
                max_tokens=cap,
                in_features=group_width,
                out_features=rank,
                block_size=(32, 32),
            )
        )
        woa_plans[cap] = ap

        def _mk_woa(_cap: int = cap) -> Callable[[Any], PreparedCall]:
            def _cb(state: Any) -> PreparedCall:
                src = (
                    torch.randn(
                        (_cap, group_width), dtype=torch.bfloat16, device=device
                    )
                    * 0.2
                )
                out = torch.empty((_cap, rank, 1), dtype=torch.bfloat16, device=device)
                scr = _alloc_scratch(state.scratch.scratch_specs(), device)
                bnd = state.bind(
                    scratch=scr, source=src, packed_weight=wo_a_pw_2d, output=out
                )
                return PreparedCall(
                    run=lambda: state.run_binding(bnd),
                    produce=lambda: src.normal_(std=0.2),
                )

            return _cb

        woa_cb = _mk_woa()
        reqs.append(
            ap.request(
                name=f"audit_woa_{cap}", prepare_call=woa_cb, benchmark_call=woa_cb
            )
        )

        bp = block_fp8_linear.plan(
            block_fp8_linear.Caps(
                device=device,
                max_tokens=cap,
                in_features=rank,
                out_features=hidden,
                block_size=(32, 32),
            )
        )
        wob_plans[cap] = bp

        def _mk_wob(_cap: int = cap) -> Callable[[Any], PreparedCall]:
            def _cb(state: Any) -> PreparedCall:
                src = (
                    torch.randn((_cap, rank), dtype=torch.bfloat16, device=device)
                    * 0.2
                )
                out = torch.empty(
                    (_cap, hidden, 1), dtype=torch.bfloat16, device=device
                )
                scr = _alloc_scratch(state.scratch.scratch_specs(), device)
                bnd = state.bind(
                    scratch=scr, source=src, packed_weight=wo_b_pw, output=out
                )
                return PreparedCall(
                    run=lambda: state.run_binding(bnd),
                    produce=lambda: src.normal_(std=0.2),
                )

            return _cb

        wob_cb = _mk_wob()
        reqs.append(
            bp.request(
                name=f"audit_wob_{cap}", prepare_call=wob_cb, benchmark_call=wob_cb
            )
        )

    for cap in (1, 8, 32, 64, 128):
        lp = bf16_vocab_projection.plan(
            bf16_vocab_projection.Caps(
                device=device,
                max_tokens=cap,
                in_features=hidden,
                out_features=vocab_tp8,
            )
        )
        lm_plans[cap] = lp

        def _mk_lm(_cap: int = cap) -> Callable[[Any], PreparedCall]:
            def _cb(state: Any) -> PreparedCall:
                src = (
                    torch.randn((_cap, hidden), dtype=torch.bfloat16, device=device)
                    * 0.2
                ).contiguous()
                return PreparedCall(
                    run=lambda: state.run(src, lm_w),
                    produce=lambda: src.normal_(std=0.2),
                )

            return _cb

        lm_cb = _mk_lm()
        reqs.append(
            lp.request(name=f"audit_lm_{cap}", prepare_call=lm_cb, benchmark_call=lm_cb)
        )

    sess = PreparationSession(device=0, autotune=True, cache_dir=cache_dir)
    sess.prepare(tuple(reqs))
    sessions.append(sess)

    oproj_cg: dict[str, Any] = {}
    oproj_parity: dict[str, Any] = {}

    for m in cg_ms:
        o_in = (
            torch.randn(
                (m, heads_per_group, head_dim), dtype=torch.bfloat16, device=device
            )
            * 0.2
        )
        pos = torch.arange(m, dtype=torch.int64, device=device)
        woa_scr = _alloc_scratch(woa_plans[m].scratch_specs(), device)
        wob_scr = _alloc_scratch(wob_plans[m].scratch_specs(), device)
        wo_scr = _alloc_scratch(wo_plans[m].scratch_specs(), device)
        wo_state = require_prepared(wo_plans[m], "gemm.wo_projection", device)
        woa_state = require_prepared(woa_plans[m], "gemm.block_fp8_linear", device)
        wob_state = require_prepared(wob_plans[m], "gemm.block_fp8_linear", device)

        def _wob_cg(
            z: torch.Tensor,
            _m: int = m,
            _st: Any = wob_state,
            _scr: Any = wob_scr,
        ) -> torch.Tensor:
            out = torch.empty((_m, hidden, 1), dtype=torch.bfloat16, device=device)
            _st.run_binding(
                _st.bind(scratch=_scr, source=z, packed_weight=wo_b_pw, output=out)
            )
            return out.squeeze(-1)

        def _route_cur(
            _o: torch.Tensor = o_in,
            _p: torch.Tensor = pos,
            _wob: Callable[[torch.Tensor], torch.Tensor] = _wob_cg,
        ) -> torch.Tensor:
            return deep_gemm_fp8_o_proj(
                _o,
                _p,
                cos_sin_cache,
                wo_a_mod_bf16,
                _wob,
                n_groups=n_groups,
                heads_per_group=heads_per_group,
                nope_dim=nope_dim,
                rope_dim=rope_dim,
                o_lora_rank=rank,
                einsum_recipe=(1, 1, 32),
                tma_aligned_scales=True,
            )

        def _route_2step(
            _m: int = m,
            _o: torch.Tensor = o_in,
            _p: torch.Tensor = pos,
            _st_a: Any = woa_state,
            _scr_a: Any = woa_scr,
            _wob: Callable[[torch.Tensor], torch.Tensor] = _wob_cg,
        ) -> torch.Tensor:
            o_r, _ = fused_inv_rope_fp8_quant(
                _o,
                _p,
                cos_sin_cache,
                n_groups=n_groups,
                heads_per_group=heads_per_group,
                nope_dim=nope_dim,
                rope_dim=rope_dim,
                quant_group_size=32,
                tma_aligned_scales=False,
                quantize=False,
            )
            z = torch.empty((_m, rank, 1), dtype=torch.bfloat16, device=device)
            _st_a.run_binding(
                _st_a.bind(
                    scratch=_scr_a,
                    source=o_r.reshape(_m, group_width),
                    packed_weight=wo_a_pw_2d,
                    output=z,
                )
            )
            return _wob(z.squeeze(-1))

        wo_bnd = wo_state.bind_inv_rope(
            scratch=wo_scr,
            o=o_in,
            positions=pos,
            cos_sin_cache=cos_sin_cache,
            weights=wo_weights,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
        )

        if m in (1, 16):
            ref_cur = _route_cur()
            out_2s = _route_2step()
            out_fu = wo_state.run_inv_rope(wo_bnd).clone()
            oproj_parity[str(m)] = {
                "fused_vs_cur_cos": round(_cos_sim(out_fu, ref_cur), 6),
                "2step_vs_cur_cos": round(_cos_sim(out_2s, ref_cur), 6),
            }

        cur_us = _bench_cudagraph_us(_route_cur)
        twostep_us = _bench_cudagraph_us(_route_2step)
        fused_us = _bench_cudagraph_us(
            lambda _st=wo_state, _b=wo_bnd: _st.run_inv_rope(_b)
        )
        oproj_cg[str(m)] = {
            "cur_bf16_bmm_plus_b12x_wob_us": round(cur_us, 2),
            "2step_b12x_us": round(twostep_us, 2),
            "b12x_wo_projection_us": round(fused_us, 2),
            "delta_step_40layers_us": round((cur_us - fused_us) * 40, 1),
        }

    # ParallelLMHead (`16160 x 5120` BF16)
    lmhead_cg: dict[str, Any] = {}
    weight_bytes = vocab_tp8 * hidden * 2
    for m in (1, 8, 32, 64, 128):
        xm = (
            torch.randn((m, hidden), dtype=torch.bfloat16, device=device) * 0.2
        ).contiguous()
        lm_bnd = bf16_vocab_projection.bind(lm_plans[m], source=xm, weight=lm_w)
        cublas_us = _bench_cudagraph_us(lambda _x=xm: F.linear(_x, lm_w))
        b12x_us = _bench_cudagraph_us(
            lambda _b=lm_bnd: bf16_vocab_projection.run(_b)
        )
        total_bytes = weight_bytes + m * (hidden + vocab_tp8) * 2
        bw_gbps = (total_bytes / 1e9) / (cublas_us / 1e6)
        lmhead_cg[str(m)] = {
            "cublas_us": round(cublas_us, 2),
            "b12x_vocab_proj_us": round(b12x_us, 2),
            "cublas_effective_bw_gbps": round(bw_gbps, 1),
        }

    # vLLM TileLang mHC vs b12x.norm.mhc
    hc_mult, mix_hc = 4, 24
    fn_w = torch.randn((mix_hc, hc_mult * hidden), dtype=torch.float32, device=device) * 0.01
    scale = torch.randn((3,), dtype=torch.float32, device=device) * 0.2
    bias = torch.randn((mix_hc,), dtype=torch.float32, device=device) * 0.1
    norm_w = torch.ones((hidden,), dtype=torch.bfloat16, device=device)
    mhc_cg: dict[str, Any] = {}
    for m in (1, 8, 32, 128):
        xm = torch.randn((m, hidden), dtype=torch.bfloat16, device=device) * 0.2
        rm = torch.randn((m, hc_mult, hidden), dtype=torch.bfloat16, device=device) * 0.2
        pmm = torch.ones((m, hc_mult), dtype=torch.float32, device=device) * 0.25
        cmm = torch.ones((m, hc_mult, hc_mult), dtype=torch.float32, device=device) * 0.25
        prem = torch.ones((m, hc_mult), dtype=torch.float32, device=device) * 0.25
        tl_us = _bench_cudagraph_us(
            lambda _x=xm, _r=rm, _pm=pmm, _cm=cmm, _pr=prem: mhc_shifted_post_pre(
                _x,
                _r,
                _pm,
                _cm,
                fn_w,
                scale,
                bias,
                1e-20,
                1e-6,
                1e-6,
                2.0,
                20,
                pre_mix=_pr,
                norm_weight=norm_w,
                norm_eps=1e-20,
            )
        )
        mhc_cg[str(m)] = {
            "vllm_tilelang_per_call_us": round(tl_us, 2),
            "vllm_tilelang_80calls_us": round(tl_us * 80, 1),
        }

    b12x_mhc_probe: dict[str, Any] = {}
    tmp_mhc = Path(tempfile.mkdtemp(prefix="zstep_audit_mhc_"))
    try:
        mp = b12x_mhc.plan(
            b12x_mhc.Caps(
                device=device,
                max_tokens=1,
                hidden_size=hidden,
                hc_mult=hc_mult,
                sinkhorn_iters=20,
            )
        )

        def _mk_mhc(state: Any) -> PreparedCall:
            xm = torch.randn((1, hidden), dtype=torch.bfloat16, device=device) * 0.2
            rm = torch.randn((1, hc_mult, hidden), dtype=torch.bfloat16, device=device) * 0.2
            pmm = torch.ones((1, hc_mult), dtype=torch.float32, device=device) * 0.25
            cmm = torch.ones((1, hc_mult, hc_mult), dtype=torch.float32, device=device) * 0.25
            prem = torch.ones((1, hc_mult), dtype=torch.float32, device=device) * 0.25
            scr = _alloc_scratch(state.scratch.scratch_specs(), device)
            bnd = state.bind(
                scratch=scr,
                x=xm,
                residual=rm,
                post_mix=pmm,
                comb_mix=cmm,
                fn_weight=fn_w,
                scale=scale,
                bias=bias,
                pre_mix=prem,
                norm_weight=norm_w,
            )
            return PreparedCall(run=bnd.run, produce=lambda: xm.normal_(std=0.2))

        msess = PreparationSession(device=0, autotune=True, cache_dir=str(tmp_mhc))
        try:
            msess.prepare(
                (mp.request(name="audit_mhc_m1", prepare_call=_mk_mhc, benchmark_call=_mk_mhc),)
            )
            b12x_mhc_probe["m1_prepare"] = "ok"
        except Exception as exc:
            b12x_mhc_probe["m1_prepare_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            msess.close()
    except Exception as exc:
        b12x_mhc_probe["plan_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(tmp_mhc, ignore_errors=True)

    return {
        "o_proj": {"parity": oproj_parity, "cg_us": oproj_cg},
        "parallel_lm_head_tp8": lmhead_cg,
        "mhc": {"vllm_tilelang_cg": mhc_cg, "b12x_norm_mhc_m1_probe": b12x_mhc_probe},
    }


def _make_realistic_pair(
    out_f: int, in_f: int, device: torch.device, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _make_mxfp8_weight(out_f, in_f, device, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SM120 DeepSeek-V4.1 operator & parity microbenchmark audit."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON file path.",
    )
    parser.add_argument(
        "--compat-dir",
        type=str,
        default=os.environ.get("ZSTEP_B12X_COMPAT_DIR", DEFAULT_COMPAT_DIR),
        help="Path to `flashinfer/experimental/b12x/_compat`.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(Path.home() / ".cache" / "b12x_zstep_dsv41_operator_audit"),
        help="Persistent cache directory for `b12x` autotuning.",
    )
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    _setup_b12x_path(args.compat_dir)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    vcfg = VllmConfig()
    ctx = set_current_vllm_config(vcfg)
    ctx.__enter__()
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:29691",
        local_rank=args.device,
    )
    initialize_model_parallel(1, 1)
    init_workspace_manager(device)

    sessions: list[Any] = []
    try:
        report: dict[str, Any] = {
            "provenance": _query_gpu_provenance(args.device),
            "block32_fp8_and_session": audit_block32_fp8_and_session(
                device, args.cache_dir, sessions
            ),
            "moe_w4a8_tiny_decode_audit": audit_moe_tiny_decode_landmine(
                device, sessions
            ),
            "oproj_lmhead_mhc_audit": audit_oproj_lmhead_and_mhc(
                device, args.cache_dir, sessions
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote operator audit JSON to {args.output}", flush=True)
    finally:
        for s in reversed(sessions):
            s.close()
        destroy_model_parallel()
        destroy_distributed_environment()
        ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main()
