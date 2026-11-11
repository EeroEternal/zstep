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

"""Phase 0 Step 0.5b: 8-GPU host-mapped SHM, b12x, and NCCL AllReduce."""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import os
import tempfile
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from zstep_worker.kernels_sm120.shm_allreduce import (
    HostMappedShmAllReduce,
    get_host_mapped_allreduce_extension,
)

TOKEN_COUNTS = (6, 48, 192, 4096)
HIDDEN_SIZE = 5120
WORLD_SIZE = 8
WARMUP_CALLS = 10
TIMED_CALLS_PER_GRAPH = 80
TIMED_REPLAYS = 5
SHM_BLOCKS = 32
SHM_THREADS = 256


def _make_case(tokens: int, rank: int, device: torch.device):
    reference = torch.zeros(
        (tokens, HIDDEN_SIZE), dtype=torch.float32, device=device
    )
    local_input = None
    for source_rank in range(WORLD_SIZE):
        generator = torch.Generator(device=device)
        generator.manual_seed(7100 + tokens * WORLD_SIZE + source_rank)
        source = torch.randn(
            (tokens, HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ) * 0.05
        reference += source.float()
        if source_rank == rank:
            local_input = source.clone()
    assert local_input is not None
    return local_input, reference.to(torch.bfloat16)


def _accuracy(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    assert torch.isfinite(actual).all()
    cosine = float(
        F.cosine_similarity(actual.float().flatten(), reference.float().flatten(), dim=0)
        .item()
    )
    max_abs_error = float((actual.float() - reference.float()).abs().max().item())
    assert cosine > 0.9999, f"AllReduce cosine is too low: {cosine}"
    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.02)
    return {
        "cos_sim": round(cosine, 8),
        "max_abs_error": round(max_abs_error, 8),
    }


def _capture_graph(operation, *, calls: int, capture_context=None) -> torch.cuda.CUDAGraph:
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    if capture_context is None:
        with torch.cuda.graph(graph):
            for _ in range(calls):
                operation()
    else:
        with capture_context:
            with torch.cuda.graph(graph):
                for _ in range(calls):
                    operation()
    return graph


def _time_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    reset_operation=None,
) -> dict[str, float]:
    graph.replay()
    torch.cuda.synchronize()
    latencies_us = []
    for _ in range(TIMED_REPLAYS):
        if reset_operation is not None:
            reset_operation()
            torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        latencies_us.append(start.elapsed_time(end) * 1000.0 / TIMED_CALLS_PER_GRAPH)
    per_rank_latencies = [None] * WORLD_SIZE
    dist.all_gather_object(per_rank_latencies, sum(latencies_us) / len(latencies_us))
    slowest_rank_us = max(float(value) for value in per_rank_latencies)
    return {
        "latency_us": round(slowest_rank_us, 3),
        "80_call_budget_ms": round(slowest_rank_us * TIMED_CALLS_PER_GRAPH / 1000.0, 4),
        "rank_latencies_us": [round(float(value), 3) for value in per_rank_latencies],
    }


def _run_shm_case(
    communicator: HostMappedShmAllReduce,
    *,
    tokens: int,
    rank: int,
    device: torch.device,
) -> dict[str, object]:
    input_tensor, reference = _make_case(tokens, rank, device)
    output = torch.empty_like(input_tensor)

    def operation() -> torch.Tensor:
        return communicator.all_reduce(input_tensor, output=output)

    operation()
    torch.cuda.synchronize()
    eager_accuracy = _accuracy(output, reference)

    sequence_before = int(communicator.control[0].item())
    replay_graph = _capture_graph(operation, calls=1)
    assert int(communicator.control[0].item()) == sequence_before
    replay_graph.replay()
    torch.cuda.synchronize()
    assert int(communicator.control[0].item()) == sequence_before + 1
    replay_accuracy = _accuracy(output, reference)
    del replay_graph

    for _ in range(WARMUP_CALLS):
        operation()
    torch.cuda.synchronize()
    timed_graph = _capture_graph(operation, calls=TIMED_CALLS_PER_GRAPH)
    timing = _time_graph(timed_graph)
    del timed_graph
    return {
        "bytes": input_tensor.numel() * input_tensor.element_size(),
        "eager_accuracy": eager_accuracy,
        "graph_replay_accuracy": replay_accuracy,
        "monotonic_graph_counter_passed": True,
        **timing,
    }


def _run_nccl_case(
    nccl_group: dist.ProcessGroup,
    *,
    tokens: int,
    rank: int,
    device: torch.device,
) -> dict[str, object]:
    input_tensor, reference = _make_case(tokens, rank, device)
    reduced = input_tensor.clone()
    dist.all_reduce(reduced, group=nccl_group)
    torch.cuda.synchronize()
    eager_accuracy = _accuracy(reduced, reference)

    def replay_operation() -> None:
        dist.all_reduce(reduced, group=nccl_group)

    replay_graph = _capture_graph(replay_operation, calls=1)
    reduced.copy_(input_tensor)
    replay_graph.replay()
    torch.cuda.synchronize()
    replay_accuracy = _accuracy(reduced, reference)
    del replay_graph

    timed_input = torch.full_like(input_tensor, 1.0e-38)

    def timed_operation() -> None:
        dist.all_reduce(timed_input, group=nccl_group)

    for _ in range(WARMUP_CALLS):
        timed_operation()
    torch.cuda.synchronize()
    timed_graph = _capture_graph(timed_operation, calls=TIMED_CALLS_PER_GRAPH)
    timing = _time_graph(
        timed_graph,
        reset_operation=lambda: timed_input.fill_(1.0e-38),
    )
    del timed_graph
    return {
        "bytes": input_tensor.numel() * input_tensor.element_size(),
        "eager_accuracy": eager_accuracy,
        "graph_replay_accuracy": replay_accuracy,
        **timing,
    }


def _construct_b12x(device: torch.device, exchange_group: dist.ProcessGroup):
    from flashinfer.experimental.b12x.comm import pcie

    runtime = None
    construction_error = None
    try:
        runtime = pcie.AllReduce.from_process_group(
            process_group=exchange_group,
            device=device,
            max_input_bytes=max(TOKEN_COUNTS) * HIDDEN_SIZE * 2,
            eager_buffer_bytes=max(TOKEN_COUNTS) * HIDDEN_SIZE * 2,
            rank_data_bytes=8 * 1024 * 1024,
            single_channel=True,
            max_concurrent_channels=1,
        )
    except Exception as error:
        construction_error = type(error).__name__
    errors = [None] * WORLD_SIZE
    dist.all_gather_object(errors, construction_error)
    if any(error is not None for error in errors):
        return None, {
            "supported": False,
            "reason": "b12x CUDA-IPC peer buffers require direct GPU P2P access, which CNS disables",
            "construction_errors": sorted({error for error in errors if error is not None}),
        }
    return runtime, {"supported": True}


def _run_b12x_case(
    runtime,
    *,
    tokens: int,
    rank: int,
    device: torch.device,
) -> dict[str, object]:
    from flashinfer.experimental.b12x.preparation import PreparedCall, prepare_default

    input_tensor, reference = _make_case(tokens, rank, device)
    output = torch.empty_like(input_tensor)
    plan = runtime.plan(input_tensor)

    def operation() -> torch.Tensor:
        return runtime.all_reduce(input_tensor, plan=plan, out=output)

    request = plan.request(
        name=f"phase0-pcie-allreduce-m{tokens}",
        prepare_call=lambda _state: PreparedCall(
            run=operation, owners=(input_tensor, output)
        ),
    )
    prepare_default(request)
    operation()
    torch.cuda.synchronize()
    eager_accuracy = _accuracy(output, reference)
    replay_graph = _capture_graph(
        operation, calls=1, capture_context=runtime.capture()
    )
    replay_graph.replay()
    torch.cuda.synchronize()
    replay_accuracy = _accuracy(output, reference)
    del replay_graph

    for _ in range(WARMUP_CALLS):
        operation()
    torch.cuda.synchronize()
    timed_graph = _capture_graph(
        operation,
        calls=TIMED_CALLS_PER_GRAPH,
        capture_context=runtime.capture(),
    )
    timing = _time_graph(timed_graph)
    del timed_graph
    return {
        "bytes": input_tensor.numel() * input_tensor.element_size(),
        "eager_accuracy": eager_accuracy,
        "graph_replay_accuracy": replay_accuracy,
        **timing,
    }


def _worker(
    rank: int,
    init_file: str,
    shm_name: str,
    result_queue,
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=WORLD_SIZE,
    )
    nccl_group = dist.new_group(
        ranks=list(range(WORLD_SIZE)), backend="nccl"
    )
    communicator = None
    b12x_runtime = None
    try:
        communicator = HostMappedShmAllReduce(
            name=shm_name,
            rank=rank,
            world_size=WORLD_SIZE,
            capacity_elements=max(TOKEN_COUNTS) * HIDDEN_SIZE,
            blocks=SHM_BLOCKS,
            threads=SHM_THREADS,
            exchange_group=dist.group.WORLD,
        )
        shm_results = {}
        nccl_results = {}
        b12x_results = {}
        for tokens in TOKEN_COUNTS:
            shm_results[tokens] = _run_shm_case(
                communicator, tokens=tokens, rank=rank, device=device
            )
            nccl_results[tokens] = _run_nccl_case(
                nccl_group, tokens=tokens, rank=rank, device=device
            )
        b12x_runtime, b12x_report = _construct_b12x(device, dist.group.WORLD)
        cuda_runtime = ctypes.CDLL("libcudart.so")
        cuda_runtime.cudaGetLastError.restype = ctypes.c_int
        cuda_runtime.cudaGetLastError()
        if b12x_runtime is not None:
            for tokens in TOKEN_COUNTS:
                b12x_results[tokens] = _run_b12x_case(
                    b12x_runtime, tokens=tokens, rank=rank, device=device
                )
        if rank == 0:
            result_queue.put(
                {
                    "step": "0.5b",
                    "device": torch.cuda.get_device_name(device),
                    "compute_capability": "12.0",
                    "torch_version": torch.__version__,
                    "world_size": WORLD_SIZE,
                    "hidden_size": HIDDEN_SIZE,
                    "dtype": "bfloat16",
                    "direct_p2p_available": False,
                    "nccl_transport": "host SHM with NCCL P2P disabled by topology",
                    "host_mapped_shm": {
                        "algorithm": "two-shot reduce-scatter/all-gather",
                        "accumulation": "FP32 with one BF16 rounding",
                        "blocks": SHM_BLOCKS,
                        "threads": SHM_THREADS,
                        "cuda_graph_capturable": True,
                        "results": shm_results,
                    },
                    "b12x_comm_pcie": {**b12x_report, "results": b12x_results},
                    "nccl": {"results": nccl_results},
                }
            )
        dist.barrier()
        if b12x_runtime is not None:
            b12x_runtime.close()
        dist.barrier()
        communicator.close()
    finally:
        if communicator is not None:
            communicator.close()
        dist.destroy_process_group()


def run_step_05b_suite(*, output: Path | None) -> dict[str, object]:
    if torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError(f"Expected {WORLD_SIZE} visible CUDA devices")
    for rank in range(WORLD_SIZE):
        if torch.cuda.get_device_capability(rank) != (12, 0):
            raise RuntimeError(f"GPU {rank} is not sm_120")
        if rank > 0 and torch.cuda.can_device_access_peer(0, rank):
            raise RuntimeError("This test targets the non-P2P CNS topology")
    get_host_mapped_allreduce_extension()

    from torch.multiprocessing import spawn

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    with tempfile.TemporaryDirectory(prefix="zstep_phase0_allreduce_") as directory:
        init_file = str(Path(directory) / "store")
        shm_name = f"zstep_ar_{os.getpid()}_{uuid.uuid4().hex}"
        spawn(
            _worker,
            args=(init_file, shm_name, result_queue),
            nprocs=WORLD_SIZE,
            join=True,
        )
    result = result_queue.get()
    encoded = json.dumps(result, indent=2) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run_step_05b_suite(output=args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
