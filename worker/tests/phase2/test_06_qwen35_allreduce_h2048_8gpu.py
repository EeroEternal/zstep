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

"""Eight-GPU decode AllReduce benchmark for Qwen3.5 H=2048 BF16 on sm_120."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from collections.abc import Callable
import subprocess
import tempfile
import uuid

import torch
import torch.distributed as dist

from zstep_worker.kernels_sm120.shm_allreduce import (
    HostMappedShmAllReduce,
    get_host_mapped_allreduce_extension,
)
from zstep_worker.numa_pin import (
    pin_process_to_device_numa as pin_current_process_to_gpu_numa,
)

WORLD_SIZE = 8
DECODE_HIDDEN_SIZE = 2048
DECODE_ROWS = (1, 2, 4, 8, 16, 32, 64, 128, 512, 4096, 8192)
BLOCK_SWEEP_HIDDEN_SIZE = 5120
BLOCK_SWEEP_ROWS = 48
BLOCK_SWEEP = (8, 16, 32)
SHM_THREADS = 256
LOOP_CALLS_PER_GRAPH = 10
WARMUP_GRAPH_REPLAYS = 300
TIMED_GRAPH_REPLAYS = 1000


def _percentile(samples: list[float], percentile: int) -> float:
    ordered = sorted(samples)
    index = (len(ordered) - 1) * percentile // 100
    return ordered[index]


def _make_case(
    *, rows: int, hidden_size: int, rank: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    reference = torch.zeros((rows, hidden_size), dtype=torch.float32, device=device)
    local_input = None
    for source_rank in range(WORLD_SIZE):
        generator = torch.Generator(device=device)
        generator.manual_seed(
            7100 + rows * WORLD_SIZE + hidden_size * 17 + source_rank
        )
        source = torch.randn(
            (rows, hidden_size),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ) * 0.05
        reference += source.float()
        if source_rank == rank:
            local_input = source.clone()
    assert local_input is not None
    return local_input, reference.to(torch.bfloat16)


def _check_accuracy(
    actual: torch.Tensor, reference: torch.Tensor, *, exact: bool
) -> float:
    assert torch.isfinite(actual).all()
    if exact:
        torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)
    else:
        torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.02)
    return float((actual.float() - reference.float()).abs().max().item())


def _capture_graph(operation: Callable[[], object], *, calls: int) -> torch.cuda.CUDAGraph:
    dist.barrier()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            operation()
    capture_stream.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        for _ in range(calls):
            operation()
    return graph


def _time_graph(graph: torch.cuda.CUDAGraph, *, calls_per_graph: int) -> dict[str, object]:
    dist.barrier()
    for _ in range(WARMUP_GRAPH_REPLAYS):
        graph.replay()
    torch.cuda.synchronize()

    latencies_us: list[float] = []
    for _ in range(TIMED_GRAPH_REPLAYS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        latencies_us.append(start.elapsed_time(end) * 1000.0 / calls_per_graph)

    all_rank_samples: list[list[float]] = [[] for _ in range(WORLD_SIZE)]
    dist.all_gather_object(all_rank_samples, latencies_us)
    slowest_rank_samples = [
        max(all_rank_samples[rank][index] for rank in range(WORLD_SIZE))
        for index in range(TIMED_GRAPH_REPLAYS)
    ]
    rank_means = [sum(samples) / len(samples) for samples in all_rank_samples]
    return {
        "p50_us": round(_percentile(slowest_rank_samples, 50), 4),
        "p95_us": round(_percentile(slowest_rank_samples, 95), 4),
        "p99_us": round(_percentile(slowest_rank_samples, 99), 4),
        "rank_mean_us": [round(value, 4) for value in rank_means],
    }


def _prefixed_metrics(prefix: str, timing: dict[str, object]) -> dict[str, object]:
    return {
        f"{prefix}_p50_us": timing["p50_us"],
        f"{prefix}_p95_us": timing["p95_us"],
        f"{prefix}_p99_us": timing["p99_us"],
        f"{prefix}_rank_mean_us": timing["rank_mean_us"],
    }


def _run_shm_case(
    communicator: HostMappedShmAllReduce,
    *,
    rows: int,
    hidden_size: int,
    rank: int,
    device: torch.device,
) -> dict[str, object]:
    input_tensor, reference = _make_case(
        rows=rows, hidden_size=hidden_size, rank=rank, device=device
    )
    output = torch.empty_like(input_tensor)

    def operation() -> torch.Tensor:
        return communicator.all_reduce(input_tensor, output=output)

    operation()
    torch.cuda.synchronize()
    eager_max_abs_error = _check_accuracy(output, reference, exact=True)

    active_control = communicator.control[: communicator.blocks]
    counters_before = active_control.clone()
    single_graph = _capture_graph(operation, calls=1)
    counters_after_capture = counters_before + 3
    assert torch.equal(active_control, counters_after_capture)
    single_graph.replay()
    torch.cuda.synchronize()
    expected_counters = counters_after_capture + 1
    assert torch.equal(active_control, expected_counters)
    graph_max_abs_error = _check_accuracy(output, reference, exact=True)
    single_timing = _time_graph(single_graph, calls_per_graph=1)
    del single_graph

    loop_graph = _capture_graph(operation, calls=LOOP_CALLS_PER_GRAPH)
    loop_timing = _time_graph(loop_graph, calls_per_graph=LOOP_CALLS_PER_GRAPH)
    del loop_graph
    return {
        "max_abs_error": max(eager_max_abs_error, graph_max_abs_error),
        "eager_max_abs_error": eager_max_abs_error,
        "graph_replay_max_abs_error": graph_max_abs_error,
        "monotonic_graph_counter_passed": True,
        **_prefixed_metrics("shm", loop_timing),
        **_prefixed_metrics("shm_single", single_timing),
    }


def _run_nccl_case(
    nccl_group: dist.ProcessGroup,
    *,
    rows: int,
    hidden_size: int,
    rank: int,
    device: torch.device,
) -> dict[str, object]:
    input_tensor, reference = _make_case(
        rows=rows, hidden_size=hidden_size, rank=rank, device=device
    )
    reduced = input_tensor.clone()

    def accuracy_operation() -> None:
        dist.all_reduce(reduced, group=nccl_group)

    accuracy_operation()
    torch.cuda.synchronize()
    eager_max_abs_error = _check_accuracy(reduced, reference, exact=False)
    accuracy_graph = _capture_graph(accuracy_operation, calls=1)
    reduced.copy_(input_tensor)
    accuracy_graph.replay()
    torch.cuda.synchronize()
    graph_max_abs_error = _check_accuracy(reduced, reference, exact=False)
    del accuracy_graph

    timed_input = torch.full_like(input_tensor, 1.0e-38)

    def timed_operation() -> None:
        dist.all_reduce(timed_input, group=nccl_group)

    single_graph = _capture_graph(timed_operation, calls=1)
    single_timing = _time_graph(single_graph, calls_per_graph=1)
    del single_graph
    loop_graph = _capture_graph(timed_operation, calls=LOOP_CALLS_PER_GRAPH)
    loop_timing = _time_graph(loop_graph, calls_per_graph=LOOP_CALLS_PER_GRAPH)
    del loop_graph
    return {
        "nccl_max_abs_error": max(eager_max_abs_error, graph_max_abs_error),
        "nccl_eager_max_abs_error": eager_max_abs_error,
        "nccl_graph_replay_max_abs_error": graph_max_abs_error,
        **_prefixed_metrics("nccl", loop_timing),
        **_prefixed_metrics("nccl_single", single_timing),
    }


def _run_case(
    communicator: HostMappedShmAllReduce,
    nccl_group: dist.ProcessGroup,
    *,
    rows: int,
    hidden_size: int,
    rank: int,
    device: torch.device,
) -> dict[str, object]:
    if rank == 0:
        print(
            f"BENCHMARK_CASE rows={rows} hidden_size={hidden_size} "
            f"blocks={communicator.blocks}",
            flush=True,
        )
    shm_result = _run_shm_case(
        communicator,
        rows=rows,
        hidden_size=hidden_size,
        rank=rank,
        device=device,
    )
    nccl_result = _run_nccl_case(
        nccl_group,
        rows=rows,
        hidden_size=hidden_size,
        rank=rank,
        device=device,
    )
    shm_p50 = float(shm_result["shm_p50_us"])
    nccl_p50 = float(nccl_result["nccl_p50_us"])
    shm_single_p50 = float(shm_result["shm_single_p50_us"])
    nccl_single_p50 = float(nccl_result["nccl_single_p50_us"])
    payload_bytes = rows * hidden_size * 2
    return {
        "rows": rows,
        "hidden_size": hidden_size,
        "num_blocks": communicator.blocks,
        "num_threads": communicator.threads,
        "algorithm": "one-shot" if payload_bytes <= communicator.one_shot_max_vecs * 16 else "two-shot",
        "payload_bytes": payload_bytes,
        **shm_result,
        **nccl_result,
        "speedup_p50": round(nccl_p50 / shm_p50, 4),
        "speedup_single_p50": round(nccl_single_p50 / shm_single_p50, 4),
        "implied_80_calls_ms": round(shm_p50 * 80.0 / 1000.0, 4),
    }


def _worker(
    rank: int,
    init_file: str,
    shm_name: str,
    git_sha: str,
    result_queue: object,
) -> None:
    topology = pin_current_process_to_gpu_numa(rank)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_SHM_DISABLE"] = "0"
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=WORLD_SIZE,
    )
    nccl_group = dist.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    communicator = None
    try:
        communicator = HostMappedShmAllReduce(
            name=shm_name,
            rank=rank,
            world_size=WORLD_SIZE,
            capacity_elements=max(DECODE_ROWS) * DECODE_HIDDEN_SIZE,
            blocks=8,
            threads=SHM_THREADS,
            exchange_group=dist.group.WORLD,
        )
        decode_results = [
            _run_case(
                communicator,
                nccl_group,
                rows=rows,
                hidden_size=DECODE_HIDDEN_SIZE,
                rank=rank,
                device=device,
            )
            for rows in DECODE_ROWS
        ]
        dist.barrier()
        communicator.close()
        dist.barrier()

        block_sweep_results = []
        for blocks in BLOCK_SWEEP:
            communicator = HostMappedShmAllReduce(
                name=f"{shm_name}_blocks_{blocks}",
                rank=rank,
                world_size=WORLD_SIZE,
                capacity_elements=BLOCK_SWEEP_ROWS * BLOCK_SWEEP_HIDDEN_SIZE,
                blocks=blocks,
                threads=SHM_THREADS,
                exchange_group=dist.group.WORLD,
            )
            block_sweep_results.append(
                _run_case(
                    communicator,
                    nccl_group,
                    rows=BLOCK_SWEEP_ROWS,
                    hidden_size=BLOCK_SWEEP_HIDDEN_SIZE,
                    rank=rank,
                    device=device,
                )
            )
            dist.barrier()
            communicator.close()
            dist.barrier()
        communicator = None

        numa_nodes: list[int | None] = [None] * WORLD_SIZE
        gpu_clocks_mhz: list[int | None] = [None] * WORLD_SIZE
        gpu_names: list[str | None] = [None] * WORLD_SIZE
        dist.all_gather_object(numa_nodes, topology.numa_node)
        dist.all_gather_object(gpu_clocks_mhz, torch.cuda.clock_rate(device))
        dist.all_gather_object(gpu_names, torch.cuda.get_device_name(device))
        if rank == 0:
            driver_version = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ).splitlines()[0].strip()
            result_queue.put(
                {
                    "benchmark": "phase2_allreduce_h2048_8gpu",
                    "world_size": WORLD_SIZE,
                    "dtype": "bfloat16",
                    "direct_p2p_available": False,
                    "nccl_transport": "degraded NCCL SHM with P2P and IB disabled",
                    "worker_numa_nodes": numa_nodes,
                    "warmup_graph_replays": WARMUP_GRAPH_REPLAYS,
                    "timed_graph_replays": TIMED_GRAPH_REPLAYS,
                    "loop_calls_per_graph": LOOP_CALLS_PER_GRAPH,
                    "provenance": {
                        "git_sha": git_sha,
                        "nvidia_driver_version": driver_version,
                        "cuda_version": torch.version.cuda,
                        "torch_version": torch.__version__,
                        "gpu_name": gpu_names,
                        "gpu_clocks_mhz": gpu_clocks_mhz,
                    },
                    "decode_h2048_blocks8": decode_results,
                    "block_sweep_m48_h5120": block_sweep_results,
                }
            )
    finally:
        if communicator is not None:
            communicator.close()
        dist.destroy_process_group()


def _read_git_sha(repo_root: Path) -> str:
    env_sha = os.environ.get("ZSTEP_GIT_SHA", "").strip()
    if env_sha:
        return env_sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def run_phase2_allreduce_suite(*, output: Path | None) -> dict[str, object]:
    if torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError(f"Expected {WORLD_SIZE} visible CUDA devices")
    for rank in range(WORLD_SIZE):
        if torch.cuda.get_device_capability(rank) != (12, 0):
            raise RuntimeError(f"GPU {rank} is not sm_120")
        for peer in range(WORLD_SIZE):
            if rank != peer and torch.cuda.can_device_access_peer(rank, peer):
                raise RuntimeError("This benchmark targets the non-P2P topology")
    get_host_mapped_allreduce_extension()
    repo_root = Path(__file__).resolve().parents[3]
    git_sha = _read_git_sha(repo_root)

    from torch.multiprocessing import spawn

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    with tempfile.TemporaryDirectory(prefix="zstep_phase2_allreduce_") as directory:
        init_file = str(Path(directory) / "store")
        shm_name = f"zstep_h2048_{os.getpid()}_{uuid.uuid4().hex}"
        spawn(
            _worker,
            args=(init_file, shm_name, git_sha, result_queue),
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
    result = run_phase2_allreduce_suite(output=args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
