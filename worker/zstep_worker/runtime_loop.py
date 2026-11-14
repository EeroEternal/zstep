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
from contextlib import ExitStack
import ctypes
from datetime import timedelta
import gc
import os
from pathlib import Path
import select
import signal
import struct
import threading
import traceback
from unittest.mock import patch

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from zstep_worker.numa_pin import (
    DeviceNumaTopology,
    pin_process_to_device_numa,
)
from zstep_worker.shm_bridge import (
    STEP_FLAG_DECODE_ONLY,
    STEP_FLAG_DEVICE_TOKENS,
    STEP_FLAG_NEXT_PREPUBLISHED,
    SeqResult,
    ShmBridge,
    StepHeader,
    StepPlanView,
)

OPCODE_NORMAL_STEP = 1
OPCODE_SHUTDOWN = 99
WORKER_STATUS_READY = 1
WORKER_STATUS_RUNNING = 2
WORKER_STATUS_ERROR = 99


class _SignalTermination(BaseException):
    """Raised on SIGTERM / SIGINT so `finally: runtime.close(abort=True)` runs cleanly."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"received signal {signum}")
        self.signum = signum


def _install_signal_handlers() -> None:
    def _handler(signum: int, _frame: object) -> None:
        raise _SignalTermination(signum)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


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

        from zstep_worker.kernels_sm120.b12x_dsv41 import maybe_enable_b12x_for_dsv41
        from zstep_worker.kernels_sm120.moe_router import dsv41_sqrt_softplus_topk
        from zstep_worker.kernels_sm120.norm_quant import rmsnorm_quantize_mxfp8_group32
        from zstep_worker.kernels_sm120.shm_allreduce import HostMappedShmAllReduce

        self._dsv41_sqrt_softplus_topk = dsv41_sqrt_softplus_topk
        self._rmsnorm_quantize_mxfp8_group32 = rmsnorm_quantize_mxfp8_group32

        maybe_enable_b12x_for_dsv41()
        from flashinfer.experimental.b12x.gemm import mxfp8_linear

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

        quantized, scales = self._rmsnorm_quantize_mxfp8_group32(
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
        router_weights, _router_ids = self._dsv41_sqrt_softplus_topk(
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
        if self.rank == 0 and sample_row_indices is not None and sample_row_indices.numel() > 0:
            selected = reduced.index_select(0, sample_row_indices)
            token_output = selected.argmax(dim=-1)
            logprob_output = functional.log_softmax(selected.float(), dim=-1).gather(
                1, token_output.unsqueeze(1)
            )
            host_slots = req_slot_ids.index_select(0, sample_row_indices).cpu().tolist()
            host_tokens = token_output.cpu().tolist()
            host_logprobs = logprob_output.cpu().tolist()
            outputs = [
                SeqResult(
                    req_slot=int(req_slot),
                    accepted_token=int(token),
                    accepted_logprob_bits=struct.unpack("<I", struct.pack("<f", logprob[0]))[0],
                    runner_up_token=-1,
                    runner_up_logprob_bits=0,
                )
                for req_slot, token, logprob in zip(
                    host_slots, host_tokens, host_logprobs, strict=True
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

    def signal_abort(self) -> None:
        if self.communicator is not None:
            self.communicator.signal_abort()

    def close(self, *, abort: bool = False) -> None:
        import torch.distributed as dist

        if abort:
            self.signal_abort()
        if not abort and self.exchange_group is not None:
            dist.barrier(group=self.exchange_group)
        if self.communicator is not None:
            self.communicator.close(abort=abort)
        if self.exchange_group is not None:
            if not abort:
                dist.barrier(group=self.exchange_group)
            dist.destroy_process_group()
            self.exchange_group = None


def _probe_model_type(model_dir: str) -> str:
    if not model_dir:
        return ""
    import json

    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.is_file():
        return ""
    return str(json.loads(cfg_path.read_text(encoding="utf-8")).get("model_type", ""))


class RuntimeLoop:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # Pin host CPU affinity to this rank's PCIe NUMA domain BEFORE touching SHM or CUDA.
        self.numa_topology: DeviceNumaTopology = pin_process_to_device_numa(args.rank)
        self.bridge = ShmBridge(name=args.shm_name, shm_fd=args.shm_fd)
        self.runner: object | None = None
        self._watchdog_stop = threading.Event()
        self.bridge.touch_heartbeat(args.rank)
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name=f"zstep-watchdog-r{args.rank}",
            daemon=True,
        )
        self._watchdog_thread.start()
        try:
            self.model_type = _probe_model_type(args.model_dir)
            if args.model_dir and self.model_type == "deepseek_v41":
                from zstep_worker.models.deepseek_v41 import DeepSeekV41StepRunner

                self.runner = DeepSeekV41StepRunner(
                    model_dir=Path(args.model_dir),
                    rank=args.rank,
                    bridge=self.bridge,
                    dist_init_file=args.dist_init_file,
                    allreduce_name=args.allreduce_name,
                    allreduce_blocks=args.allreduce_blocks,
                    allreduce_threads=args.allreduce_threads,
                    num_kv_pages=args.num_kv_pages,
                )
            elif args.model_dir:
                from zstep_worker.models.qwen35_moe import Qwen35MoeStepRunner

                self.runner = Qwen35MoeStepRunner(
                    model_dir=Path(args.model_dir),
                    rank=args.rank,
                    bridge=self.bridge,
                    dist_init_file=args.dist_init_file,
                    allreduce_name=args.allreduce_name,
                    allreduce_blocks=args.allreduce_blocks,
                    allreduce_threads=args.allreduce_threads,
                    num_kv_pages=args.num_kv_pages,
                )
            else:
                self.runner = MicroStep(args, self.bridge)
            import torch

            free_bytes, total_bytes = torch.cuda.mem_get_info(self.runner.device)
            self._free_vram_mb: int = free_bytes // (1024 * 1024)
            self._total_vram_mb: int = total_bytes // (1024 * 1024)
            self.current_step: int = 0
            self.last_step: int = 0
            self._pending_plan_signals: int = 0
        except BaseException:
            self._stop_watchdog()
            self.bridge.close()
            raise

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(0.25):
            if self.bridge._closed:
                break
            self.bridge.touch_heartbeat(self.args.rank)
            if self.bridge.is_aborted():
                runner = self.runner
                if runner is not None:
                    abort_fn = getattr(runner, "signal_abort", None)
                    if callable(abort_fn):
                        abort_fn()
                break

    def _stop_watchdog(self) -> None:
        self._watchdog_stop.set()
        if self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=2.0)

    def _write_state(
        self,
        *,
        status: int,
        error_code: int,
        triton_recompilations: int = 0,
        watched_kernels: int = 0,
    ) -> None:
        self.bridge.write_worker_state(
            rank=self.args.rank,
            ready_step_id=self.last_step,
            status_code=status,
            error_code=error_code,
            device_sm_arch=120,
            free_vram_mb=self._free_vram_mb,
            total_vram_mb=self._total_vram_mb,
            numa_node=self.numa_topology.numa_node,
            triton_recompilations=triton_recompilations,
            watched_kernels=watched_kernels,
        )

    def _wait_for_plan_signal(self, poller: select.poll) -> None:
        # The plan eventfd is not a semaphore: one read drains every publish since the last
        # read, and the master may publish a step ahead of the one this worker is running.
        if self._pending_plan_signals > 0:
            self._pending_plan_signals -= 1
            return
        while True:
            if self.bridge.is_aborted():
                raise RuntimeError("master signaled cluster abort in SHM superblock")
            events = poller.poll(250)
            if events:
                self._pending_plan_signals += os.eventfd_read(self.args.plan_fd) - 1
                return
            self.bridge.touch_heartbeat(self.args.rank)

    def _complete_step(
        self, step_id: int, outputs: list, gpu_exec_us: int, watched_count: int
    ) -> None:
        if self.args.rank == 0:
            self.bridge.result(step_id).write(
                step_id=step_id,
                gpu_exec_us=gpu_exec_us,
                status_code=0,
                outputs=outputs,
            )
        self.last_step = step_id
        self._write_state(
            status=WORKER_STATUS_READY,
            error_code=0,
            watched_kernels=watched_count,
        )
        os.eventfd_write(self.args.done_fd, 1)

    def _finish_pending(self, pending: tuple[int, object], watched_count: int) -> None:
        step_id, launched = pending
        outputs, gpu_exec_us = self.runner.finish(launched)
        self._complete_step(step_id, outputs, gpu_exec_us, watched_count)

    def run(self) -> int:
        self._write_state(status=WORKER_STATUS_READY, error_code=0)
        os.eventfd_write(self.args.done_fd, 1)

        watched_kernels: list[tuple[str, object]] = []
        try:
            from zstep_worker.kernels_sm120.moe_router import dsv41_sqrt_softplus_topk_kernel
            from zstep_worker.kernels_sm120.norm_quant import (
                rmsnorm_quantize_mxfp8_group32_kernel,
            )

            watched_kernels.extend(
                [
                    ("norm_mxfp8", rmsnorm_quantize_mxfp8_group32_kernel),
                    ("router_dsv41", dsv41_sqrt_softplus_topk_kernel),
                ]
            )
        except ModuleNotFoundError:
            if self.args.model_dir or isinstance(self.runner, MicroStep):
                raise
        if self.args.model_dir and self.model_type == "deepseek_v41":
            extra_watched = getattr(self.runner, "watched_triton_kernels", None)
            if callable(extra_watched):
                watched_kernels.extend(extra_watched())
        elif self.args.model_dir:
            from zstep_worker.kernels_sm120 import qwen35_ops

            watched_kernels.extend(
                [
                    ("zc_rmsnorm", qwen35_ops._zero_centered_rmsnorm_kernel),
                    ("fused_add_zc_rmsnorm", qwen35_ops._fused_add_zero_centered_rmsnorm_kernel),
                    ("attn_prologue", qwen35_ops._qwen35_attn_prologue_kernel),
                    ("gqa_decode_splitk", qwen35_ops._paged_gqa_decode_splitk_kernel),
                    ("gqa_decode_reduce", qwen35_ops._paged_gqa_decode_reduce_gated_kernel),
                    ("gqa_prefill", qwen35_ops._paged_gqa_prefill_varlen_gated_kernel),
                    ("conv1d_decode", qwen35_ops._causal_conv1d_decode_kernel),
                    ("conv1d_prefill", qwen35_ops._causal_conv1d_prefill_varlen_kernel),
                    ("gdn_decode", qwen35_ops._gated_delta_decode_kernel),
                    ("gdn_prefill", qwen35_ops._gated_delta_prefill_varlen_kernel),
                    ("rmsnorm_gated", qwen35_ops._rmsnorm_gated_silu_kernel),
                    ("moe_router_shared", qwen35_ops._router_softmax256_topk8_and_shared_act_kernel),
                    ("moe_gate_up", qwen35_ops._fused_moe_gate_up_silu_kernel),
                    ("moe_down", qwen35_ops._fused_moe_down_kernel),
                    ("moe_combine", qwen35_ops._moe_combine_top8_and_shared_kernel),
                    ("moe_gate_up_gemv", qwen35_ops._moe_gate_up_silu_gemv_kernel),
                    ("moe_down_combine_gemv", qwen35_ops._moe_down_combine_gemv_kernel),
                    ("skinny_linear", qwen35_ops._skinny_linear_kernel),
                ]
            )

        poller = select.poll()
        poller.register(self.args.plan_fd, select.POLLIN)

        gc.collect()
        was_gc_enabled = gc.isenabled()
        gc.disable()
        try:
            with ExitStack() as stack:
                spies = [
                    (
                        name,
                        stack.enter_context(
                            patch.object(kernel, "_do_compile", wraps=kernel._do_compile)
                        ),
                    )
                    for name, kernel in watched_kernels
                ]
                watched_count = len(spies)
                pipelined = callable(getattr(self.runner, "launch", None)) and callable(
                    getattr(self.runner, "finish", None)
                )
                # A step launched under STEP_FLAG_NEXT_PREPUBLISHED and not yet completed.
                pending: tuple[int, object] | None = None
                while True:
                    self._wait_for_plan_signal(poller)
                    plan = self.bridge.plan(self.current_step + 1)
                    header = plan.header()
                    expected_step = self.current_step + 1
                    if header.step_id != expected_step:
                        raise RuntimeError(
                            f"expected step {expected_step}, acquired {header.step_id}"
                        )
                    self.current_step = header.step_id
                    self._write_state(
                        status=WORKER_STATUS_RUNNING,
                        error_code=0,
                        watched_kernels=watched_count,
                    )
                    if header.opcode == OPCODE_SHUTDOWN:
                        if pending is not None:
                            raise RuntimeError(
                                f"shutdown step {header.step_id} published while step "
                                f"{pending[0]} was still pending"
                            )
                        recompiled_count = sum(spy.call_count for _, spy in spies)
                        if self.args.rank == 0:
                            self.bridge.result(header.step_id).write(
                                step_id=header.step_id,
                                gpu_exec_us=0,
                                status_code=0,
                                outputs=[],
                            )
                        self.last_step = header.step_id
                        self._write_state(
                            status=WORKER_STATUS_READY,
                            error_code=0,
                            triton_recompilations=recompiled_count,
                            watched_kernels=watched_count,
                        )
                        os.eventfd_write(self.args.done_fd, 1)
                        break
                    if header.opcode != OPCODE_NORMAL_STEP:
                        raise RuntimeError(f"unsupported step opcode {header.opcode}")
                    overlap_flags = header.flags & (
                        STEP_FLAG_DEVICE_TOKENS | STEP_FLAG_NEXT_PREPUBLISHED
                    )
                    if not pipelined:
                        if overlap_flags != 0:
                            raise RuntimeError(
                                f"step {header.step_id} set overlap flags {overlap_flags:#x} "
                                f"on non-pipelined runner {type(self.runner).__name__}"
                            )
                        outputs, gpu_exec_us = self.runner.execute(plan, header)
                        self._complete_step(header.step_id, outputs, gpu_exec_us, watched_count)
                        continue
                    is_decode = (header.flags & STEP_FLAG_DECODE_ONLY) != 0
                    if not is_decode and overlap_flags != 0:
                        raise RuntimeError(
                            f"non-decode step {header.step_id} must not set overlap flags "
                            f"{overlap_flags:#x}"
                        )
                    if pending is not None and not is_decode:
                        # Only decode steps double-buffer their host staging.
                        self._finish_pending(pending, watched_count)
                        pending = None
                    launched = self.runner.launch(plan, header)
                    if pending is not None:
                        # The GPU already has this step queued behind the pending one.
                        self._finish_pending(pending, watched_count)
                        pending = None
                    if header.flags & STEP_FLAG_NEXT_PREPUBLISHED:
                        pending = (header.step_id, launched)
                        continue
                    self._finish_pending((header.step_id, launched), watched_count)
                recompiled = [
                    f"{name}={spy.call_count}" for name, spy in spies if spy.call_count != 0
                ]
                recompiled_total = sum(spy.call_count for _, spy in spies)
                if self.args.rank == 0:
                    print(
                        f"TRITON_RECOMPILATIONS={recompiled_total} WATCHED_KERNELS={watched_count}",
                        flush=True,
                    )
                if recompiled:
                    print(
                        f"rank {self.args.rank}: runtime step triggered unexpected Triton JIT "
                        f"recompilation: {', '.join(recompiled)}",
                        flush=True,
                    )
                    return 1
                return 0
        finally:
            if was_gc_enabled:
                gc.enable()

    def report_error(self, error: BaseException) -> None:
        print(f"rank {self.args.rank}: {type(error).__name__}: {error}", flush=True)
        traceback.print_exc()
        if self.args.rank == 0 and self.current_step > 0 and self.last_step < self.current_step:
            self.bridge.result(self.current_step).write(
                step_id=self.current_step,
                gpu_exec_us=0,
                status_code=1,
                outputs=[],
            )
        self.last_step = self.current_step
        self._write_state(status=WORKER_STATUS_ERROR, error_code=1)
        os.eventfd_write(self.args.done_fd, 1)

    def close(self, *, abort: bool = False) -> None:
        self._stop_watchdog()
        if self.runner is not None:
            try:
                self.runner.close(abort=abort)
            except TypeError:
                self.runner.close()
        self.bridge.close()
        os.close(self.args.plan_fd)
        os.close(self.args.done_fd)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="zstep Mode B sm_120 GPU worker")
    parser.add_argument("--shm-name", required=True)
    parser.add_argument("--shm-fd", default=-1, type=int)
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
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--num-kv-pages", default=32768, type=int)
    return parser.parse_args()


def main() -> int:
    _set_parent_death_signal()
    _install_signal_handlers()
    args = _parse_args()
    runtime = None
    aborted = False
    try:
        runtime = RuntimeLoop(args)
        return runtime.run()
    except _SignalTermination as sig_exc:
        aborted = True
        print(f"rank {args.rank}: {sig_exc}, tearing down cleanly", flush=True)
        return 1
    except BaseException as error:
        aborted = True
        if runtime is not None:
            runtime.report_error(error)
        else:
            print(
                f"rank {args.rank} worker initialization failed: {type(error).__name__}: {error}",
                flush=True,
            )
            traceback.print_exc()
        return 1
    finally:
        if runtime is not None:
            runtime.close(abort=aborted)


if __name__ == "__main__":
    raise SystemExit(main())
