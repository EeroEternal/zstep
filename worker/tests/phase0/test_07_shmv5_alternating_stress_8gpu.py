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

"""8-GPU stress & latency test for `shm_v5` AllReduce & AllGather under alternating `M`.

Specifically targets Issue #6 items 2.2 and 3.1:
1. Back-to-back 4-slot bursts rapidly alternating `M` across `1..128` (`outer_rows <= 8`
   <-> `> 8` and non-power-of-two `num_vecs`) with zero host synchronization inside bursts
   and **100% per-step output verification** across every single collective launch.
2. Verifies zero element corruption across thousands of ring-buffer wraparounds (`slots=4`),
   uniform per-CTA `step_counter` progression across all `blocks`, and immediate `/dev/shm`
   unlink-after-attach (zero visible `/dev/shm/zstep_*` entries while live).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import tempfile
import time
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from zstep_worker.kernels_sm120.shm_allreduce import (
    HostMappedShmAllGather,
    HostMappedShmAllReduce,
)
from zstep_worker.numa_pin import pin_process_to_device_numa


def _worker_main(
    rank: int,
    world_size: int,
    init_file: str,
    shm_prefix: str,
    steps_per_shape: int,
    blocks: int,
    result_file: str,
) -> None:
    topo = pin_process_to_device_numa(rank)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD
    assert group is not None

    max_rows = 128
    max_hidden = 5120
    ar = HostMappedShmAllReduce(
        name=f"{shm_prefix}_ar",
        rank=rank,
        world_size=world_size,
        capacity_elements=max_rows * max_hidden,
        blocks=blocks,
        threads=256,
        exchange_group=group,
    )
    ag = HostMappedShmAllGather(
        name=f"{shm_prefix}_ag",
        rank=rank,
        world_size=world_size,
        capacity_input_elements=max_rows * 640,
        blocks=blocks,
        threads=256,
        exchange_group=group,
    )

    live_shm_matches = glob.glob(f"/dev/shm/{shm_prefix}*")

    row_pattern = (1, 128, 2, 64, 3, 97, 8, 33, 4, 16, 7, 128, 1, 9, 5, 65)
    ar_max_diff = 0.0
    ar_checked_steps = 0

    # Pre-build deterministic per-step inputs and exact BF16 references on GPU so
    # 4-deep back-to-back collective bursts execute with zero host stalls while
    # verifying 100% of launched steps.
    for hidden in (2048, 4096, 5120):
        base_vals = (
            torch.arange(max_rows * hidden, device=device, dtype=torch.float32)
            .remainder(29.0)
            .sub(14.0)
            .mul(0.0625)
            .reshape(max_rows, hidden)
            .to(torch.bfloat16)
        )
        rank_inputs: list[torch.Tensor] = []
        ref_outputs: list[torch.Tensor] = []
        for s_idx in range(7):
            my_scale = float((s_idx + 1) * (rank + 1)) * 0.125
            rank_inputs.append((base_vals * my_scale).to(torch.bfloat16).contiguous())
            acc = torch.zeros((max_rows, hidden), device=device, dtype=torch.float32)
            for r in range(world_size):
                r_scale = float((s_idx + 1) * (r + 1)) * 0.125
                acc.add_((base_vals * r_scale).to(torch.bfloat16).to(torch.float32))
            ref_outputs.append(acc.to(torch.bfloat16).contiguous())

        out_ring = [
            torch.empty((max_rows, hidden), device=device, dtype=torch.bfloat16)
            for _ in range(4)
        ]
        step_diffs = torch.zeros(steps_per_shape, device=device, dtype=torch.float32)

        for base_step in range(0, steps_per_shape, 4):
            burst_end = min(base_step + 4, steps_per_shape)
            # Launch up to 4 collectives strictly back-to-back with zero intervening kernels.
            for step in range(base_step, burst_end):
                m = row_pattern[step % len(row_pattern)]
                s_idx = step % 7
                slot_idx = step - base_step
                ar.all_reduce(rank_inputs[s_idx][:m], output=out_ring[slot_idx][:m])
            # Enqueue asynchronous GPU diff check for every single step in the burst.
            for step in range(base_step, burst_end):
                m = row_pattern[step % len(row_pattern)]
                s_idx = step % 7
                slot_idx = step - base_step
                step_diffs[step] = (
                    (out_ring[slot_idx][:m].float() - ref_outputs[s_idx][:m].float())
                    .abs()
                    .max()
                )

        shape_max_diff = float(step_diffs.max().item())
        ar_checked_steps += steps_per_shape
        if shape_max_diff > ar_max_diff:
            ar_max_diff = shape_max_diff
        if shape_max_diff > 0.0:
            bad_step = int(step_diffs.argmax().item())
            bad_m = row_pattern[bad_step % len(row_pattern)]
            raise RuntimeError(
                f"[rank {rank}] AllReduce mismatch at hidden={hidden} step={bad_step} "
                f"M={bad_m}: max_diff={shape_max_diff}"
            )

    # Stress HostMappedShmAllGather with the same 4-deep back-to-back burst + 100% step check.
    shard_cols = 640
    ag_base = (
        torch.arange(max_rows * shard_cols, device=device, dtype=torch.float32)
        .remainder(31.0)
        .reshape(max_rows, shard_cols)
        .to(torch.bfloat16)
    )
    ag_shards: list[torch.Tensor] = []
    ag_refs: list[torch.Tensor] = []
    for t_idx in range(13):
        tag = float(t_idx + 1)
        ag_shards.append(
            (ag_base + float(rank * 100) + tag).to(torch.bfloat16).contiguous()
        )
        ag_refs.append(
            torch.cat(
                [
                    (ag_base + float(r * 100) + tag).to(torch.bfloat16)
                    for r in range(world_size)
                ],
                dim=-1,
            ).contiguous()
        )

    ag_out_ring = [
        torch.empty(
            (max_rows, shard_cols * world_size), device=device, dtype=torch.bfloat16
        )
        for _ in range(4)
    ]
    ag_step_diffs = torch.zeros(steps_per_shape, device=device, dtype=torch.float32)
    for base_step in range(0, steps_per_shape, 4):
        burst_end = min(base_step + 4, steps_per_shape)
        for step in range(base_step, burst_end):
            m = row_pattern[step % len(row_pattern)]
            t_idx = step % 13
            slot_idx = step - base_step
            ag.all_gather(ag_shards[t_idx][:m], output=ag_out_ring[slot_idx][:m])
        for step in range(base_step, burst_end):
            m = row_pattern[step % len(row_pattern)]
            t_idx = step % 13
            slot_idx = step - base_step
            ag_step_diffs[step] = (
                (ag_out_ring[slot_idx][:m].float() - ag_refs[t_idx][:m].float())
                .abs()
                .max()
            )

    ag_mismatches = int((ag_step_diffs > 0.0).sum().item())
    ag_checked_steps = steps_per_shape
    if ag_mismatches > 0:
        bad_step = int(ag_step_diffs.argmax().item())
        bad_m = row_pattern[bad_step % len(row_pattern)]
        raise RuntimeError(
            f"[rank {rank}] AllGather mismatch at step={bad_step} M={bad_m} "
            f"(total mismatches={ag_mismatches})"
        )

    torch.cuda.synchronize(device)
    ar_ctrl = ar.control[:blocks].cpu().tolist()
    ag_ctrl = ag.control[:blocks].cpu().tolist()
    if len(set(ar_ctrl)) != 1 or len(set(ag_ctrl)) != 1:
        raise RuntimeError(
            f"[rank {rank}] Non-uniform CTA step counters: ar={ar_ctrl}, ag={ag_ctrl}"
        )

    # Measure M=1..128 latency (H=4096) with fixed gridDim.x + per-block ring reuse barrier.
    latencies_us: dict[str, float] = {}
    bench_x = torch.randn((128, 4096), device=device, dtype=torch.bfloat16)
    bench_y = torch.empty_like(bench_x)
    for m in (1, 2, 4, 8, 16, 32, 64, 128):
        for _ in range(50):
            ar.all_reduce(bench_x[:m], output=bench_y[:m])
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        iters = 400
        start_ev.record()
        for _ in range(iters):
            ar.all_reduce(bench_x[:m], output=bench_y[:m])
        end_ev.record()
        end_ev.synchronize()
        latencies_us[f"m{m}_h4096_us"] = round(
            (start_ev.elapsed_time(end_ev) * 1000.0) / iters, 2
        )

    ar.close()
    ag.close()
    dist.barrier(group=group)
    if rank == 0:
        report = {
            "world_size": world_size,
            "blocks": blocks,
            "steps_per_shape": steps_per_shape,
            "total_ar_launches": ar_ctrl[0],
            "ar_checked_steps": ar_checked_steps,
            "total_ag_launches": ag_ctrl[0],
            "ag_checked_steps": ag_checked_steps,
            "ar_max_diff": ar_max_diff,
            "ag_mismatches": ag_mismatches,
            "live_dev_shm_entries": live_shm_matches,
            "rank0_numa_node": topo.numa_node,
            "latencies_us": latencies_us,
        }
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    dist.destroy_process_group()


def run_alternating_stress(
    *,
    world_size: int = 8,
    steps_per_shape: int = 2048,
    blocks: int = 8,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "dist_init")
        result_file = os.path.join(tmp, "result.json")
        shm_prefix = (
            f"zstep_altstress_{os.getpid()}_{int(time.time_ns() & 0xFFFF_FFFF):08x}"
        )
        mp.spawn(
            _worker_main,
            args=(
                world_size,
                init_file,
                shm_prefix,
                steps_per_shape,
                blocks,
                result_file,
            ),
            nprocs=world_size,
            join=True,
        )
        with open(result_file, encoding="utf-8") as f:
            report: dict[str, Any] = json.load(f)

    post_shm = glob.glob(f"/dev/shm/{shm_prefix}*")
    assert report["ar_max_diff"] == 0.0, report
    assert report["ag_mismatches"] == 0, report
    assert report["ar_checked_steps"] == report["total_ar_launches"], report
    assert report["ag_checked_steps"] == report["total_ag_launches"], report
    assert report["live_dev_shm_entries"] == [], (
        f"Expected immediate unlink after attach, found {report['live_dev_shm_entries']}"
    )
    assert post_shm == [], f"Leaked /dev/shm files after close: {post_shm}"
    return report


def test_shmv5_alternating_rows_8gpu() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        pytest.skip(f"Requires 8 CUDA GPUs, found {torch.cuda.device_count()}")
    cap = torch.cuda.get_device_capability(0)
    if cap != (12, 0):
        pytest.skip(f"Requires sm_120 GPUs, found {cap}")
    report = run_alternating_stress(world_size=8, steps_per_shape=512, blocks=8)
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--steps-per-shape", type=int, default=2048)
    parser.add_argument("--blocks", type=int, default=8)
    args = parser.parse_args()

    if torch.cuda.device_count() < args.world_size:
        raise SystemExit(
            f"Need {args.world_size} GPUs, found {torch.cuda.device_count()}"
        )

    report = run_alternating_stress(
        world_size=args.world_size,
        steps_per_shape=args.steps_per_shape,
        blocks=args.blocks,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
