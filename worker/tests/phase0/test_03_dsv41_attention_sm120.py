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

"""Phase 0 Step 0.4: native DeepSeek-V4.1 compressed sparse MLA on sm_120."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

LOCAL_HEADS = 16
HEAD_DIM = 512
SWA_WIDTH = 128
INDEXED_WIDTH = 512
PAGE_SIZE = 64
SM_SCALE = HEAD_DIM**-0.5


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


def _selected_indices(
    *, rows: int, batch: int, q_len: int, kv_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    row_ids = torch.arange(rows, device=device, dtype=torch.int32)
    query_offsets = row_ids % q_len
    visible = kv_len - (q_len - 1 - query_offsets)
    swa_offsets = torch.arange(SWA_WIDTH, device=device, dtype=torch.int32)
    swa_indices = visible[:, None] - SWA_WIDTH + swa_offsets[None, :]
    indexed_offsets = torch.arange(INDEXED_WIDTH, device=device, dtype=torch.int32)
    indexed_indices = (
        visible[:, None] - SWA_WIDTH - INDEXED_WIDTH + indexed_offsets[None, :]
    )
    assert batch * q_len == rows
    assert int(indexed_indices.min()) >= 0
    swa_lengths = torch.full((rows,), SWA_WIDTH, device=device, dtype=torch.int32)
    indexed_lengths = torch.full(
        (rows,), INDEXED_WIDTH, device=device, dtype=torch.int32
    )
    return swa_indices, swa_lengths, indexed_indices, indexed_lengths


def _reference_attention(
    q: torch.Tensor,
    swa_values: torch.Tensor,
    indexed_values: torch.Tensor,
    swa_indices: torch.Tensor,
    indexed_indices: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty_like(q)
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for start in range(0, q.shape[0], 32):
            end = min(start + 32, q.shape[0])
            swa_k = swa_values[swa_indices[start:end].long()]
            indexed_k = indexed_values[indexed_indices[start:end].long()]
            keys = torch.cat((swa_k, indexed_k), dim=1)
            scores = torch.matmul(q[start:end].float(), keys.transpose(1, 2)) * SM_SCALE
            probabilities = torch.softmax(scores, dim=-1)
            output[start:end].copy_(
                torch.matmul(probabilities, keys).to(torch.bfloat16)
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
    return output


def _run_attention_case(
    *,
    mode: str,
    batch: int,
    q_len: int,
    kv_len: int,
    device: torch.device,
) -> dict[str, float | int | str]:
    from flashinfer.experimental.b12x.attention import compressed_sparse_mla as mla
    from flashinfer.experimental.b12x.attention.compressed_sparse_mla.reference import (
        pack_deepseek_v41_cache_reference,
        unpack_deepseek_v41_cache_reference,
    )
    from flashinfer.experimental.b12x.preparation import PreparedCall, prepare_default

    rows = batch * q_len
    torch.manual_seed(5100 + rows + kv_len)
    q = (
        torch.randn((rows, LOCAL_HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16)
        * 0.2
    )
    kv = torch.randn((kv_len, HEAD_DIM), device=device, dtype=torch.bfloat16) * 0.25
    swa_cache = pack_deepseek_v41_cache_reference(
        kv, page_size=PAGE_SIZE, cache_kind="swa"
    )
    indexed_cache = pack_deepseek_v41_cache_reference(
        kv, page_size=PAGE_SIZE, cache_kind="indexed"
    )
    swa_indices, swa_lengths, indexed_indices, indexed_lengths = _selected_indices(
        rows=rows, batch=batch, q_len=q_len, kv_len=kv_len, device=device
    )
    output = torch.empty_like(q)
    invocation = mla.invocation_from_tensors(
        q=q,
        swa_k_cache=swa_cache,
        indexed_k_cache=indexed_cache,
        attn_sink=None,
        out=output,
        return_lse=False,
        lse_scale="base2",
    )
    caps = mla.Caps(
        device=device,
        num_q_heads=LOCAL_HEADS,
        max_q_rows=rows,
        max_width=SWA_WIDTH + INDEXED_WIDTH,
        max_page_table_width=(kv_len + PAGE_SIZE - 1) // PAGE_SIZE,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        head_dim=HEAD_DIM,
        v_head_dim=HEAD_DIM,
        max_batch=batch,
        max_kv_rows=kv_len,
        max_chunks_per_row=None,
        max_q_chunks=None,
        decode_row_capacity=rows if mode == "decode" else None,
        page_size=PAGE_SIZE,
        layout="compressed_dsv4",
        cache_format="deepseek_v41",
        mode=mode,
        swa_width=SWA_WIDTH,
        indexed_width=INDEXED_WIDTH,
        swa_page_size=PAGE_SIZE,
        indexed_page_size=PAGE_SIZE,
        shared_width_capacity=None,
        use_cuda_graph=True,
    )
    declaration = mla.plan(caps, invocation=invocation, override=None)

    def bind_for(state, scratch):
        return state.bind_for_preparation(
            scratch=scratch,
            q=q,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            indexed_indices=indexed_indices,
            indexed_lengths=indexed_lengths,
            indexed_page_table=None,
        )

    def prepare_call(state):
        scratch = torch.empty(
            state.scratch_plan.scratch_specs()[0].shape,
            dtype=state.scratch_plan.scratch_specs()[0].dtype,
            device=device,
        )
        binding = bind_for(state, scratch)
        run = lambda: state.run(
            binding,
            swa_k_cache=swa_cache,
            indexed_k_cache=indexed_cache,
            swa_page_size=PAGE_SIZE,
            indexed_page_size=PAGE_SIZE,
            sm_scale=SM_SCALE,
            attn_sink=None,
            out=output,
            return_lse=False,
            lse_scale="base2",
            cache_format="deepseek_v41",
        )
        return PreparedCall(run=run, output=output, owners=(scratch, binding))

    request = declaration.request(
        name=f"phase0-mla-{mode}-b{batch}-q{q_len}-kv{kv_len}",
        prepare_call=prepare_call,
    )
    prepare_default(request)
    from flashinfer.experimental.b12x.preparation import require_prepared

    state = require_prepared(request.plan, "attention.compressed_sparse_mla", device)
    spec = state.scratch_plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
    binding = mla.bind(
        request.plan,
        scratch=scratch,
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lengths,
        indexed_page_table=None,
    )

    def run() -> torch.Tensor:
        return mla.run(
            plan=request.plan,
            binding=binding,
            swa_k_cache=swa_cache,
            indexed_k_cache=indexed_cache,
            swa_page_size=PAGE_SIZE,
            indexed_page_size=PAGE_SIZE,
            sm_scale=SM_SCALE,
            attn_sink=None,
            out=output,
            return_lse=False,
            lse_scale="base2",
            cache_format="deepseek_v41",
        )

    run()
    swa_values = unpack_deepseek_v41_cache_reference(
        swa_cache, page_size=PAGE_SIZE, cache_kind="swa"
    )
    indexed_values = unpack_deepseek_v41_cache_reference(
        indexed_cache, page_size=PAGE_SIZE, cache_kind="indexed"
    )
    reference = _reference_attention(
        q, swa_values, indexed_values, swa_indices, indexed_indices
    )
    cos = float(
        F.cosine_similarity(
            output.float().flatten(), reference.float().flatten(), dim=0
        )
    )
    assert torch.isfinite(output).all()
    assert cos > 0.99, (
        f"MLA cosine too low for mode={mode}, batch={batch}, q_len={q_len}, "
        f"kv_len={kv_len}: {cos}"
    )
    latency_us = _bench_cuda_graph_us(run)
    selected_bytes = rows * (SWA_WIDTH * 528 + INDEXED_WIDTH * 288)
    effective_gbps = selected_bytes / (latency_us * 1e-6) / 1e9
    return {
        "mode": mode,
        "batch": batch,
        "q_len": q_len,
        "rows": rows,
        "kv_len": kv_len,
        "cos_sim": round(cos, 6),
        "latency_us": round(latency_us, 2),
        "selected_cache_effective_gbps": round(effective_gbps, 1),
    }


def run_step_04_suite() -> dict[str, object]:
    from flashinfer.experimental.b12x.attention import compressed_sparse_mla as mla

    device = torch.device("cuda:0")
    assert torch.cuda.get_device_capability(device) == (12, 0)
    assert mla.is_supported(device)
    cases: list[dict[str, float | int | str]] = []
    for batch in (1, 8, 32):
        for kv_len in (2048, 8192, 32768):
            cases.append(
                _run_attention_case(
                    mode="decode", batch=batch, q_len=1, kv_len=kv_len, device=device
                )
            )
    cases.append(
        _run_attention_case(mode="verify", batch=1, q_len=6, kv_len=8192, device=device)
    )
    cases.append(
        _run_attention_case(mode="verify", batch=8, q_len=6, kv_len=8192, device=device)
    )
    cases.append(
        _run_attention_case(
            mode="extend", batch=32, q_len=128, kv_len=8192, device=device
        )
    )
    return {
        "step": "0.4",
        "device": torch.cuda.get_device_name(device),
        "compute_capability": "12.0",
        "torch_version": torch.__version__,
        "geometry": {
            "cache_format": "deepseek_v41",
            "local_heads_tp8": LOCAL_HEADS,
            "head_dim": HEAD_DIM,
            "swa_width": SWA_WIDTH,
            "index_topk": INDEXED_WIDTH,
            "page_size": PAGE_SIZE,
            "raw_slot_indexed_decode": True,
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run_step_04_suite()
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
