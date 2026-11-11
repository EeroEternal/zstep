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

"""Mode B GPU worker driven only by POSIX SHM and inherited eventfds."""

from __future__ import annotations

import argparse
import ctypes
import os
import signal
import struct
from datetime import timedelta
from unittest.mock import patch

from zstep_worker.kernels_sm120.moe_router import (
    dsv41_sqrt_softplus_topk,
    dsv41_sqrt_softplus_topk_kernel,
)
from zstep_worker.kernels_sm120.norm_quant import (
    rmsnorm_quantize_mxfp8_group32,
    rmsnorm_quantize_mxfp8_group32_kernel,
)
from zstep_worker.shm_bridge import SeqResult, ShmBridge, StepHeader, StepPlanView

OPCODE_NORMAL_STEP = 1
OPCODE_SHUTDOWN = 99
WORKER_STATUS_READY = 1
WORKER_STATUS_RUNNING = 2
WORKER_STATUS_ERROR = 99


def _set_parent_death_signal() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    pr_set_pdeathsig = 1
    result = libc.prctl(pr_set_pdeathsig, signal.SIGTERM, 0, 0, 0)
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _pack_mxfp8_weight(weight, mxfp8_linear):
    import torch

    out_features, hidden_size = weight.shape
    grouped = weight.float().view(out_features, hidden_size // 32, 32)
    amax = grouped.abs().amax(dim=-1).clamp(min=1.0e-30)
    exponent = torch.ceil(torch.log2(amax / 448.0)).clamp(-127.0, 127.0)
    scales = (exponent.to(torch.int32) + 127).to(torch.uint8)
    quantized = (
        (grouped / torch.exp2(exponent).unsqueeze(-1))
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .reshape(out_features, hidden_size)
    )
    return mxfp8_linear.pack_weight(quantized, scales)


class MicroStep:
    """Fixed-weight sm_120 forward micro-step with explicit runtime dimensions."""

    def __init__(self, args: argparse.Namespace, bridge: ShmBridge) -> None:
        import torch
        import torch.distributed as dist
        from flashinfer.experimental.b12x.gemm import mxfp8_linear

        from zstep_worker.kernels_sm120.shm_allreduce import HostMappedShmAllReduce

        self.args = args
        self.bridge = bridge
        self.rank = args.rank
        self.world_size = bridge.superblock.geometry.world_size
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank is outside the SHM world size")
        if torch.cuda.device_count() < self.world_size:
            raise RuntimeError("fewer visible CUDA devices than the SHM world size")
        torch.cuda.set_device(self.rank)
        self.device = torch.device("cuda", self.rank)
        capability = torch.cuda.get_device_capability(self.device)
        if capability != (12, 0):
            raise RuntimeError(f"Mode B requires sm_120, found {capability}")
        if not mxfp8_linear.is_supported(self.device):
            raise RuntimeError("b12x MXFP8 GEMM does not support this device")

        self.hidden_size = args.hidden_size
        self.output_features = args.output_features
        self.router_experts = args.router_experts
        self.router_top_k = args.router_top_k
        if self.hidden_size <= 0 or self.hidden_size % 32 != 0:
            raise ValueError("hidden_size must be a positive multiple of 32")
        if self.output_features < self.router_experts:
            raise ValueError("output_features must include all router logit columns")
        if not 0 < self.router_top_k <= self.router_experts:
            raise ValueError("router_top_k is invalid")

        generator = torch.Generator(device=self.device)
        generator.manual_seed(2026 + self.rank)
        self.norm_weight = (
            torch.randn(
                self.hidden_size,
                device=self.device,
                dtype=torch.bfloat16,
                generator=generator,
            )
            * 0.02
            + 1.0
        )
        weight = (
            torch.randn(
                (self.output_features, self.hidden_size),
                device=self.device,
                dtype=torch.bfloat16,
                generator=generator,
            )
            * 0.02
        )
        self.packed_weight = _pack_mxfp8_weight(weight, mxfp8_linear)
        self.mxfp8_linear = mxfp8_linear
        self.columns = torch.arange(self.hidden_size, device=self.device, dtype=torch.int32)
        self.reduced_workspace = torch.empty(
            (bridge.superblock.geometry.max_step_tokens, self.output_features),
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

        self.exchange_group = None
        self.communicator = None
        if self.world_size == 8:
            dist.init_process_group(
                backend="gloo",
                init_method=f"file://{args.dist_init_file}",
                rank=self.rank,
                world_size=self.world_size,
                timeout=timedelta(seconds=120),
            )
            self.exchange_group = dist.group.WORLD
            self.communicator = HostMappedShmAllReduce(
                name=args.allreduce_name,
                rank=self.rank,
                world_size=self.world_size,
                capacity_elements=(
                    bridge.superblock.geometry.max_step_tokens * self.output_features
                ),
                blocks=args.allreduce_blocks,
                threads=args.allreduce_threads,
                exchange_group=self.exchange_group,
            )

        max_warm = bridge.superblock.geometry.max_step_tokens
        for warm_m in (6, 17, 48, 192):
            if warm_m <= max_warm:
                warmup_tokens = torch.arange(warm_m, device=self.device, dtype=torch.int32) + 1
                warmup_positions = torch.zeros_like(warmup_tokens)
                warmup_req_slots = torch.zeros_like(warmup_tokens)
                self._forward(warmup_tokens, warmup_positions, warmup_req_slots, None)
        torch.cuda.synchronize(self.device)

    def _forward(
        self,
        token_ids,
        positions,
        req_slot_ids,
        sample_row_indices,
    ) -> tuple[list[SeqResult], int]:
        import torch
        import torch.nn.functional as functional

        self.start_event.record()
        rows = token_ids.shape[0]
        column_offsets = self.columns.unsqueeze(0)
        token_offsets = token_ids.to(torch.int32).unsqueeze(1)
        position_offsets = positions.to(torch.int32).unsqueeze(1)
        request_offsets = req_slot_ids.to(torch.int32).unsqueeze(1)
        activations = (
            (column_offsets + token_offsets + position_offsets + request_offsets) % 1024
        ).to(torch.bfloat16) * (1.0 / 1024.0)

        quantized, scales = rmsnorm_quantize_mxfp8_group32(
            activations,
            self.norm_weight,
            eps=1.0e-6,
        )
        scale_values = torch.exp2(scales.to(torch.float32) - 127.0)
        dequantized = (
            quantized.to(torch.float32).view(rows, self.hidden_size // 32, 32)
            * scale_values.unsqueeze(-1)
        ).reshape(rows, self.hidden_size).to(torch.bfloat16)
        gemm_output = self.mxfp8_linear.mm(dequantized, self.packed_weight)
        router_weights, _router_ids = dsv41_sqrt_softplus_topk(
            gemm_output[:, : self.router_experts].float(),
            top_k=self.router_top_k,
            renormalize=True,
            routed_scaling_factor=1.5,
            correction_bias=None,
        )
        gemm_output = gemm_output + router_weights.sum(dim=1).to(torch.bfloat16).unsqueeze(1)

        if self.communicator is not None:
            reduced = self.reduced_workspace[:rows]
            self.communicator.all_reduce(gemm_output, output=reduced)
        else:
            reduced = gemm_output

        outputs: list[SeqResult] = []
        if self.rank == 0 and sample_row_indices is not None:
            selected = reduced.index_select(0, sample_row_indices)
            token_output = selected.argmax(dim=-1)
            logprob_output = functional.log_softmax(selected.float(), dim=-1).gather(
                1, token_output.unsqueeze(1)
            )
            host_tokens = token_output.cpu().tolist()
            host_logprobs = logprob_output.cpu().tolist()
            outputs = [
                SeqResult(
                    req_slot=req_slot,
                    accepted_token=int(token),
                    accepted_logprob_bits=struct.unpack("<I", struct.pack("<f", logprob[0]))[0],
                )
                for req_slot, token, logprob in zip(
                    range(len(host_tokens)), host_tokens, host_logprobs, strict=True
                )
            ]
        self.end_event.record()
        self.end_event.synchronize()
        gpu_exec_us = max(1, int(self.start_event.elapsed_time(self.end_event) * 1000.0))
        return outputs, gpu_exec_us

    def execute(self, plan: StepPlanView, header: StepHeader) -> tuple[list[SeqResult], int]:
        import torch

        token_count = header.num_encoder_tokens
        if token_count == 0 or token_count != header.num_decoder_rows:
            raise ValueError("micro-step requires equal nonzero encoder and decoder rows")
        token_ids = torch.from_numpy(plan.token_ids(token_count)).to(self.device)
        positions = torch.from_numpy(plan.positions(token_count)).to(self.device)
        req_slot_ids = torch.from_numpy(plan.req_slot_ids(token_count)).to(self.device)
        sample_row_indices = None
        if self.rank == 0:
            sample_row_indices = torch.from_numpy(
                plan.sample_row_indices(header.num_sample_rows)
            ).to(self.device)
        return self._forward(token_ids, positions, req_slot_ids, sample_row_indices)

    def close(self) -> None:
        import torch.distributed as dist

        if self.exchange_group is not None:
            dist.barrier(group=self.exchange_group)
        if self.communicator is not None:
            self.communicator.close()
        if self.exchange_group is not None:
            dist.barrier(group=self.exchange_group)
            dist.destroy_process_group()
            self.exchange_group = None


class RuntimeLoop:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.bridge = ShmBridge(name=args.shm_name)
        self.micro_step = MicroStep(args, self.bridge)
        self.current_step = 0
        self.last_step = 0

    def _write_state(self, *, status: int, error_code: int) -> None:
        import torch

        free_bytes, total_bytes = torch.cuda.mem_get_info(self.micro_step.device)
        self.bridge.write_worker_state(
            rank=self.args.rank,
            ready_step_id=self.last_step,
            status_code=status,
            error_code=error_code,
            device_sm_arch=120,
            free_vram_mb=free_bytes // (1024 * 1024),
            total_vram_mb=total_bytes // (1024 * 1024),
        )

    def run(self) -> None:
        self._write_state(status=WORKER_STATUS_READY, error_code=0)
        os.eventfd_write(self.args.done_fd, 1)
        with (
            patch.object(
                rmsnorm_quantize_mxfp8_group32_kernel,
                "_do_compile",
                wraps=rmsnorm_quantize_mxfp8_group32_kernel._do_compile,
            ) as norm_compile_spy,
            patch.object(
                dsv41_sqrt_softplus_topk_kernel,
                "_do_compile",
                wraps=dsv41_sqrt_softplus_topk_kernel._do_compile,
            ) as router_compile_spy,
        ):
            while True:
                os.eventfd_read(self.args.plan_fd)
                plan = self.bridge.plan(self.current_step + 1)
                header = plan.header()
                expected_step = self.current_step + 1
                if header.step_id != expected_step:
                    raise RuntimeError(
                        f"expected step {expected_step}, acquired {header.step_id}"
                    )
                self.current_step = header.step_id
                self._write_state(status=WORKER_STATUS_RUNNING, error_code=0)
                if header.opcode == OPCODE_SHUTDOWN:
                    if self.args.rank == 0:
                        self.bridge.result(header.step_id).write(
                            step_id=header.step_id,
                            gpu_exec_us=0,
                            status_code=0,
                            outputs=[],
                        )
                    self.last_step = header.step_id
                    self._write_state(status=WORKER_STATUS_READY, error_code=0)
                    os.eventfd_write(self.args.done_fd, 1)
                    break
                if header.opcode != OPCODE_NORMAL_STEP:
                    raise RuntimeError(f"unsupported step opcode {header.opcode}")
                outputs, gpu_exec_us = self.micro_step.execute(plan, header)
                if self.args.rank == 0:
                    self.bridge.result(header.step_id).write(
                        step_id=header.step_id,
                        gpu_exec_us=gpu_exec_us,
                        status_code=0,
                        outputs=outputs,
                    )
                self.last_step = header.step_id
                self._write_state(status=WORKER_STATUS_READY, error_code=0)
                os.eventfd_write(self.args.done_fd, 1)
            if norm_compile_spy.call_count != 0 or router_compile_spy.call_count != 0:
                raise RuntimeError(
                    "runtime token count triggered Triton recompilation: "
                    f"norm={norm_compile_spy.call_count}, router={router_compile_spy.call_count}"
                )

    def report_error(self, error: BaseException) -> None:
        print(f"rank {self.args.rank}: {type(error).__name__}: {error}", flush=True)
        if self.args.rank == 0 and self.current_step > 0:
            self.bridge.result(self.current_step).write(
                step_id=self.current_step,
                gpu_exec_us=0,
                status_code=1,
                outputs=[],
            )
        self.last_step = self.current_step
        self._write_state(status=WORKER_STATUS_ERROR, error_code=1)
        os.eventfd_write(self.args.done_fd, 1)

    def close(self) -> None:
        self.micro_step.close()
        self.bridge.close()
        os.close(self.args.plan_fd)
        os.close(self.args.done_fd)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="zstep Mode B sm_120 GPU worker")
    parser.add_argument("--shm-name", required=True)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--plan-fd", required=True, type=int)
    parser.add_argument("--done-fd", required=True, type=int)
    parser.add_argument("--allreduce-name", required=True)
    parser.add_argument("--dist-init-file", required=True)
    parser.add_argument("--hidden-size", required=True, type=int)
    parser.add_argument("--output-features", required=True, type=int)
    parser.add_argument("--router-experts", required=True, type=int)
    parser.add_argument("--router-top-k", required=True, type=int)
    parser.add_argument("--allreduce-blocks", required=True, type=int)
    parser.add_argument("--allreduce-threads", required=True, type=int)
    return parser.parse_args()


def main() -> int:
    _set_parent_death_signal()
    args = _parse_args()
    runtime = None
    try:
        runtime = RuntimeLoop(args)
        runtime.run()
        return 0
    except BaseException as error:
        if runtime is not None:
            runtime.report_error(error)
        else:
            print(f"worker initialization failed: {type(error).__name__}: {error}", flush=True)
        return 1
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
