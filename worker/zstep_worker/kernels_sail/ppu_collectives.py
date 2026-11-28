# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT

"""CUDA-Graph-safe T-Head ICNLink P2P IPC collectives (`PPU-ZW810E`, `TP=8`).

Uses direct device-to-device `cudaIpcOpenMemHandle` + `cudaDeviceEnablePeerAccess` over
T-Head ICNLink (`ICN1`/`ICN2`) for sub-15us decode/verify/draft collectives (`BF16` AllReduce,
`FP32` Indexer AllReduce, `BF16` AllGather, and lock-free `Top-2` Argmax), with automatic
size-aware fallback to `PCCL` (`dist.all_reduce`) for large non-graphed prefill chunks.
"""

from __future__ import annotations

import contextlib
import ctypes
import math
import mmap
import os
from pathlib import Path
import struct

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline

from zstep_worker.numa_pin import DeviceNumaTopology, pin_process_to_device_numa

NUM_RING_SLOTS: int = 4
MAX_RANKS: int = 8
MAX_BLOCKS: int = 64
IPC_REGION_HEADER_BYTES: int = 524288  # 256 KiB ready + 256 KiB done flags (64B isolated)
TOP2_HEADER_BYTES: int = 8192

_CSRC_DIR = Path(__file__).resolve().parent / "csrc"
_CPP_SOURCE: str = (_CSRC_DIR / "sail_ops.cpp").read_text(encoding="utf-8")
_CUDA_SOURCE: str = (_CSRC_DIR / "sail_ops_kernels.cu").read_text(encoding="utf-8")

_SAIL_EXT = None


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def get_sail_ops_extension():
    """Compile or load the unified PPU/SAIL C++/CUDA extension (`zstep_sail_ops_ext`)."""
    global _SAIL_EXT
    if _SAIL_EXT is None:
        _SAIL_EXT = load_inline(
            name="zstep_sail_ops_v8_ext",
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            functions=[
                "register_mapped_host",
                "unregister_mapped_host",
                "enable_peer_access",
                "alloc_ipc_buffer",
                "open_ipc_handle",
                "close_ipc_handle",
                "free_ipc_buffer",
                "launch_icn_allreduce_bf16",
                "launch_icn_allreduce_mhc_post_bf16",
                "launch_icn_allreduce_f32",
                "launch_icn_allgather_bf16",
                "launch_local_top2_bf16",
                "launch_icn_tp8_top2_gather",
                "launch_decode_prepare_tokens",
                "launch_decode_commit_tokens",
                "launch_rmsnorm_bf16",
                "launch_mhc_fused_mixes_bf16",
                "launch_mhc_fused_mixes_splitk_bf16",
                "launch_mhc_vec_single_splitk_bf16",
                "launch_mhc_sinkhorn_from_proj",
                "launch_mhc_sinkhorn_ozaki_vec",
                "launch_mhc_pre_rmsnorm_bf16",
                "launch_mhc_pre_rmsnorm_quant_bf16",
                "launch_mhc_post_bf16",
                "launch_inplace_act_quant_ue8m0_bf16",
                "launch_act_quant_ue8m0_bf16",
                "launch_shared_swiglu_quant_bf16",
                "launch_inplace_fp4_quant_bf16",
                "launch_rope_inplace_bf16",
                "launch_fused_qkv_a_epilogue_bf16",
                "launch_fused_q_idx_epilogue_bf16",
                "launch_fused_inv_rope_act_quant_bf16",
                "launch_dsv41_router_topk",
                "launch_mxfp4_moe_gemv_gate_up",
                "launch_mxfp4_moe_gemv_down_combine",
                "launch_moe_align_block_size",
                "launch_moe_gather_tile_inputs_bf16",
                "launch_dequant_mxfp4_tiles_bf16",
                "launch_bmm_bf16_f32_nt",
                "launch_gemm_bf16_f32_nt",
                "launch_moe_tile_swiglu_quant_bf16",
                "launch_moe_scatter_tile_outputs_bf16",
                "launch_moe_combine_pairs_shared_bf16",
                "launch_engram_lookup_dequant",
                "launch_engram_gated_residual_bf16",
                "launch_compressor_cr2_fused",
                "launch_indexer_paged_score_bf16",
                "launch_indexer_topk512_slots",
                "launch_sparse_attn_tp8_bf16",
                "launch_sparse_attn_tp8_reduce",
                "launch_dspark_sail_verify_prep",
                "launch_fused_mtp_ctx_kv_epilogue_bf16",
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            extra_ldflags=["-lcublas"],
            with_cuda=True,
            verbose=False,
        )
    return _SAIL_EXT


class PpuIcnCollectives:
    """Unified CUDA-Graph-safe ICNLink P2P IPC communicator for `TP=8` on `PPU-ZW810E`."""

    def __init__(
        self,
        *,
        name: str,
        rank: int,
        world_size: int,
        device: torch.device,
        exchange_group: dist.ProcessGroup,
        pccl_group: dist.ProcessGroup | None = None,
        blocks: int = 8,
        threads: int = 256,
        oneshot_max_bytes: int = 131072,
        pccl_min_bytes: int = 4194304,
        max_bf16_ar_elements: int = 1024 * 8192,
        max_f32_ar_elements: int = 64 * 16384,
        max_bf16_ag_input_elements: int = 256 * 16160,
        max_top2_rows: int = 4096,
    ) -> None:
        if world_size < 2 or world_size > MAX_RANKS or not 0 <= rank < world_size:
            raise ValueError(f"Invalid rank={rank} / world_size={world_size}")
        if not 1 <= blocks <= MAX_BLOCKS:
            raise ValueError(f"blocks={blocks} outside 1..={MAX_BLOCKS}")
        if threads not in (64, 128, 256, 512):
            raise ValueError(f"threads={threads} must be 64, 128, 256, or 512")

        torch.cuda.set_device(device)
        self.numa_topology: DeviceNumaTopology = pin_process_to_device_numa(device.index)
        self.name: str = name
        self.rank: int = rank
        self.world_size: int = world_size
        self.device: torch.device = device
        self.exchange_group: dist.ProcessGroup = exchange_group
        self.pccl_group: dist.ProcessGroup | None = pccl_group
        self.blocks: int = blocks
        self.threads: int = threads
        self.oneshot_max_bytes: int = oneshot_max_bytes
        self.pccl_min_bytes: int = pccl_min_bytes
        self.max_bf16_ar_elements: int = _align_up(max_bf16_ar_elements, 64)
        self.max_f32_ar_elements: int = _align_up(max_f32_ar_elements, 64)
        self.max_bf16_ag_input_elements: int = _align_up(max_bf16_ag_input_elements, 64)
        self.max_top2_rows: int = max(64, max_top2_rows)

        self._ext = get_sail_ops_extension()
        self._closed: bool = False
        self._abort_shm_path: str = f"/dev/shm/{name}_abort"
        self._abort_mm: mmap.mmap | None = None
        self._abort_host_addr: int = 0
        self._abort_dev_ptr: int = 0
        self._owns_abort_unlink: bool = False
        self._my_ipc_ptr: int = 0
        self._peer_ipc_ptrs_host: list[int] = []

        # Telemetry counters for Evolith workload observation
        self.icn_bf16_ar_calls: int = 0
        self.icn_f32_ar_calls: int = 0
        self.pccl_ar_calls: int = 0

        # Compute non-overlapping region offsets inside each rank's HBM IPC allocation:
        # Each AR slot holds 2 * MAX_RANKS rank inboxes (phase 0 + phase 1).
        bf16_inbox = _align_up(self.max_bf16_ar_elements * 2, 4096)
        self.bf16_ar_slot_stride: int = 2 * MAX_RANKS * bf16_inbox
        self.bf16_ar_region_off: int = 0
        bf16_ar_bytes = _align_up(
            IPC_REGION_HEADER_BYTES + NUM_RING_SLOTS * self.bf16_ar_slot_stride, 65536
        )

        f32_inbox = _align_up(self.max_f32_ar_elements * 4, 4096)
        self.f32_ar_slot_stride: int = 2 * MAX_RANKS * f32_inbox
        self.f32_ar_region_off: int = bf16_ar_bytes
        f32_ar_bytes = _align_up(
            IPC_REGION_HEADER_BYTES + NUM_RING_SLOTS * self.f32_ar_slot_stride, 65536
        )

        ag_inbox = _align_up(self.max_bf16_ag_input_elements * 2, 4096)
        self.bf16_ag_slot_stride: int = MAX_RANKS * ag_inbox
        self.bf16_ag_region_off: int = self.f32_ar_region_off + f32_ar_bytes
        bf16_ag_bytes = _align_up(
            IPC_REGION_HEADER_BYTES + NUM_RING_SLOTS * self.bf16_ag_slot_stride, 65536
        )

        self.top2_region_off: int = self.bf16_ag_region_off + bf16_ag_bytes
        top2_bytes = _align_up(
            TOP2_HEADER_BYTES + NUM_RING_SLOTS * MAX_RANKS * (self.max_top2_rows * 2) * 8,
            65536,
        )
        # Dedicated isolated 4-slot small-payload BF16 AllReduce ring (`<= 64 KiB`, `8 x 256`)
        # so `M <= 6` decode/verify avoids 64-block cross-PPU flag contention with zero slot hazard.
        self.small_ar_max_bytes: int = 65536
        small_inbox = _align_up(self.small_ar_max_bytes, 4096)
        self.bf16_small_ar_slot_stride: int = 2 * MAX_RANKS * small_inbox
        self.bf16_small_ar_region_off: int = self.top2_region_off + top2_bytes
        bf16_small_ar_bytes = _align_up(
            IPC_REGION_HEADER_BYTES + NUM_RING_SLOTS * self.bf16_small_ar_slot_stride, 65536
        )
        self.small_blocks: int = min(self.blocks, 8)
        self.small_threads: int = min(self.threads, 256)
        self.total_ipc_bytes: int = self.bf16_small_ar_region_off + bf16_small_ar_bytes

        try:
            self._init_abort_shm()
            self._init_icn_p2p_ipc()
            self.bf16_ar_control: torch.Tensor = torch.zeros(
                MAX_BLOCKS, dtype=torch.int64, device=device
            )
            self.bf16_small_ar_control: torch.Tensor = torch.zeros(
                MAX_BLOCKS, dtype=torch.int64, device=device
            )
            self.f32_ar_control: torch.Tensor = torch.zeros(
                MAX_BLOCKS, dtype=torch.int64, device=device
            )
            self.bf16_ag_control: torch.Tensor = torch.zeros(
                MAX_BLOCKS, dtype=torch.int64, device=device
            )
            self.top2_step_counter: torch.Tensor = torch.zeros(
                1, dtype=torch.int64, device=device
            )
        except BaseException:
            self.close(abort=True)
            raise

    def _init_abort_shm(self) -> None:
        if self.rank == 0:
            fd = os.open(self._abort_shm_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
            try:
                os.ftruncate(fd, 4096)
                mm = mmap.mmap(
                    fd, 4096, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE
                )
            finally:
                os.close(fd)
            self._owns_abort_unlink = True
            ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(mm)), 0, 4096)
            self._abort_mm = mm
        dist.barrier(group=self.exchange_group)
        if self.rank != 0:
            fd = os.open(self._abort_shm_path, os.O_RDWR)
            try:
                self._abort_mm = mmap.mmap(
                    fd, 4096, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE
                )
            finally:
                os.close(fd)
        dist.barrier(group=self.exchange_group)
        if self.rank == 0 and self._owns_abort_unlink:
            try:
                os.unlink(self._abort_shm_path)
            except FileNotFoundError:
                pass
            self._owns_abort_unlink = False
        assert self._abort_mm is not None
        self._abort_host_addr = ctypes.addressof(ctypes.c_char.from_buffer(self._abort_mm))
        self._abort_dev_ptr = int(
            self._ext.register_mapped_host(self._abort_host_addr, 4096)
        )

    def _init_icn_p2p_ipc(self) -> None:
        for peer in range(self.world_size):
            if peer != self.rank:
                ok = int(self._ext.enable_peer_access(peer))
                if ok != 1:
                    raise RuntimeError(
                        f"PPU rank {self.rank} cannot enable ICNLink P2P access to peer {peer}"
                    )
        own_ptr_t = torch.zeros(1, dtype=torch.int64)
        handle_u8 = self._ext.alloc_ipc_buffer(self.total_ipc_bytes, own_ptr_t)
        self._my_ipc_ptr = int(own_ptr_t[0])
        gathered = [torch.empty_like(handle_u8) for _ in range(self.world_size)]
        dist.all_gather(gathered, handle_u8, group=self.exchange_group)

        base_ptrs: list[int] = []
        for peer in range(self.world_size):
            if peer == self.rank:
                base_ptrs.append(self._my_ipc_ptr)
            else:
                p_ptr = int(self._ext.open_ipc_handle(gathered[peer]))
                base_ptrs.append(p_ptr)
        dist.barrier(group=self.exchange_group)
        self._peer_ipc_ptrs_host = base_ptrs

        def _pinned_i64(vals: list[int]) -> torch.Tensor:
            cpu_p = torch.zeros(len(vals), dtype=torch.int64, pin_memory=True)
            for idx_v, v in enumerate(vals):
                cpu_p[idx_v] = int(v)
            gpu_p = torch.zeros(len(vals), dtype=torch.int64, device=self.device)
            gpu_p.copy_(cpu_p, non_blocking=False)
            return gpu_p

        self.bf16_ar_peer_ptrs: torch.Tensor = _pinned_i64(
            [p + self.bf16_ar_region_off for p in base_ptrs]
        )
        self.bf16_small_ar_peer_ptrs: torch.Tensor = _pinned_i64(
            [p + self.bf16_small_ar_region_off for p in base_ptrs]
        )
        self.f32_ar_peer_ptrs: torch.Tensor = _pinned_i64(
            [p + self.bf16_ar_region_off * 0 + self.f32_ar_region_off for p in base_ptrs]
        )
        self.bf16_ag_peer_ptrs: torch.Tensor = _pinned_i64(
            [p + self.bf16_ag_region_off for p in base_ptrs]
        )
        self.base_peer_ptrs: torch.Tensor = _pinned_i64(base_ptrs)
        torch.cuda.synchronize(self.device)

    def all_reduce_bf16(
        self, input_: torch.Tensor, *, output: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Reduce BF16 `input_` across all ranks in FP32 and write BF16 `output`."""
        if self._closed:
            raise RuntimeError("PpuIcnCollectives is closed")
        out = input_ if output is None else output
        numel = input_.numel()
        if numel == 0:
            return out
        nbytes = numel * 2
        cur_stream = torch.cuda.current_stream(self.device)
        capturing = torch.cuda.is_current_stream_capturing()
        on_default_stream = (
            cur_stream.cuda_stream == torch.cuda.default_stream(self.device).cuda_stream
        )
        if (
            not capturing
            and on_default_stream
            and self.pccl_group is not None
            and (nbytes >= self.pccl_min_bytes or numel > self.max_bf16_ar_elements)
        ):
            if out.data_ptr() != input_.data_ptr():
                out.copy_(input_)
            dist.all_reduce(out, group=self.pccl_group)
            self.pccl_ar_calls += 1
            return out
        if numel > self.max_bf16_ar_elements:
            if not capturing and input_.ndim >= 2:
                flat_in = input_.reshape(-1, input_.shape[-1])
                flat_out = out.reshape(-1, out.shape[-1])
                rows_step = max(1, self.max_bf16_ar_elements // flat_in.shape[1])
                for lo in range(0, flat_in.shape[0], rows_step):
                    hi = min(flat_in.shape[0], lo + rows_step)
                    self.all_reduce_bf16(flat_in[lo:hi], output=flat_out[lo:hi])
                return out
            raise ValueError(
                f"BF16 AllReduce numel={numel} exceeds ICNLink IPC capacity "
                f"{self.max_bf16_ar_elements} without PCCL fallback"
            )
        stream_handle = cur_stream.cuda_stream
        self._ext.launch_icn_allreduce_bf16(
            input_,
            out,
            self.bf16_ar_control,
            self.bf16_ar_peer_ptrs,
            self._abort_dev_ptr,
            self.bf16_ar_slot_stride,
            self.oneshot_max_bytes,
            self.rank,
            self.world_size,
            self.blocks,
            self.threads,
            stream_handle,
        )
        self.icn_bf16_ar_calls += 1
        return out

    def all_reduce_mhc_post_bf16(
        self,
        input_: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        *,
        out: torch.Tensor,
        scratch_red: torch.Tensor,
    ) -> torch.Tensor:
        """Fused TP=8 BF16 AllReduce + 4-stream `mhc_post` update (`CUDA-Graph` & `PCCL` safe)."""
        if self._closed:
            raise RuntimeError("PpuIcnCollectives is closed")
        numel = input_.numel()
        if numel == 0:
            return out
        nbytes = numel * 2
        cur_stream = torch.cuda.current_stream(self.device)
        capturing = torch.cuda.is_current_stream_capturing()
        on_default_stream = (
            cur_stream.cuda_stream == torch.cuda.default_stream(self.device).cuda_stream
        )
        stream_handle = cur_stream.cuda_stream
        if (
            not capturing
            and on_default_stream
            and self.pccl_group is not None
            and (nbytes >= self.pccl_min_bytes or numel > self.max_bf16_ar_elements)
        ):
            red = scratch_red[: input_.shape[0]]
            if red.data_ptr() != input_.data_ptr():
                red.copy_(input_)
            dist.all_reduce(red, group=self.pccl_group)
            self.pccl_ar_calls += 1
            self._ext.launch_mhc_post_bf16(
                red, residual, post_mix, comb_mix, out, stream_handle | 1
            )
            return out
        if nbytes > self.oneshot_max_bytes or numel > self.max_bf16_ar_elements:
            red = scratch_red[: input_.shape[0]]
            self.all_reduce_bf16(input_, output=red)
            self._ext.launch_mhc_post_bf16(
                red, residual, post_mix, comb_mix, out, stream_handle | 1
            )
            return out
        use_small = nbytes <= self.small_ar_max_bytes and self.blocks > 8
        self._ext.launch_icn_allreduce_mhc_post_bf16(
            input_,
            residual,
            post_mix,
            comb_mix,
            out,
            self.bf16_small_ar_control if use_small else self.bf16_ar_control,
            self.bf16_small_ar_peer_ptrs if use_small else self.bf16_ar_peer_ptrs,
            self._abort_dev_ptr,
            self.bf16_small_ar_slot_stride if use_small else self.bf16_ar_slot_stride,
            self.oneshot_max_bytes,
            self.rank,
            self.world_size,
            self.small_blocks if use_small else self.blocks,
            self.small_threads if use_small else self.threads,
            stream_handle,
        )
        self.icn_bf16_ar_calls += 1
        return out

    def all_reduce_f32(
        self, input_: torch.Tensor, *, output: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Reduce FP32 `input_` across all ranks (CUDA-Graph-safe for Indexer scores)."""
        if self._closed:
            raise RuntimeError("PpuIcnCollectives is closed")
        out = input_ if output is None else output
        numel = input_.numel()
        if numel == 0:
            return out
        nbytes = numel * 4
        capturing = torch.cuda.is_current_stream_capturing()
        if (
            not capturing
            and self.pccl_group is not None
            and (nbytes >= self.pccl_min_bytes or numel > self.max_f32_ar_elements)
        ):
            if out.data_ptr() != input_.data_ptr():
                out.copy_(input_)
            dist.all_reduce(out, group=self.pccl_group)
            self.pccl_ar_calls += 1
            return out
        if numel > self.max_f32_ar_elements:
            if not capturing and input_.ndim >= 2:
                flat_in = input_.reshape(-1, input_.shape[-1])
                flat_out = out.reshape(-1, out.shape[-1])
                rows_step = max(1, self.max_f32_ar_elements // flat_in.shape[1])
                for lo in range(0, flat_in.shape[0], rows_step):
                    hi = min(flat_in.shape[0], lo + rows_step)
                    self.all_reduce_f32(flat_in[lo:hi], output=flat_out[lo:hi])
                return out
            raise ValueError(
                f"FP32 AllReduce numel={numel} exceeds ICNLink IPC capacity "
                f"{self.max_f32_ar_elements} without PCCL fallback"
            )
        stream_handle = torch.cuda.current_stream(self.device).cuda_stream
        self._ext.launch_icn_allreduce_f32(
            input_,
            out,
            self.f32_ar_control,
            self.f32_ar_peer_ptrs,
            self._abort_dev_ptr,
            self.f32_ar_slot_stride,
            self.oneshot_max_bytes,
            self.rank,
            self.world_size,
            self.blocks,
            self.threads,
            stream_handle,
        )
        self.icn_f32_ar_calls += 1
        return out

    def all_gather_bf16(
        self,
        input_: torch.Tensor,
        *,
        dim: int = -1,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("PpuIcnCollectives is closed")
        ndim = input_.ndim
        if dim < 0:
            dim += ndim
        if not input_.is_contiguous():
            input_ = input_.contiguous()
        out_shape = list(input_.shape)
        out_shape[dim] *= self.world_size
        out = (
            torch.empty(out_shape, dtype=input_.dtype, device=input_.device)
            if output is None
            else output
        )
        numel = input_.numel()
        if numel == 0:
            return out
        if numel > self.max_bf16_ag_input_elements:
            if self.pccl_group is not None and not torch.cuda.is_current_stream_capturing():
                gathered = torch.empty(
                    (self.world_size, *input_.shape),
                    dtype=input_.dtype,
                    device=input_.device,
                )
                dist.all_gather_into_tensor(gathered, input_, group=self.pccl_group)
                perm = list(range(1, dim + 1)) + [0] + list(range(dim + 1, ndim + 1))
                out.copy_(gathered.permute(*perm).reshape(out_shape))
                return out
            raise ValueError(f"AllGather input numel={numel} exceeds ICNLink capacity")
        outer_rows = math.prod(input_.shape[:dim]) if dim > 0 else 1
        inner_elems = math.prod(input_.shape[dim:])
        stream_handle = torch.cuda.current_stream(self.device).cuda_stream
        self._ext.launch_icn_allgather_bf16(
            input_,
            out,
            self.bf16_ag_control,
            self.bf16_ag_peer_ptrs,
            self._abort_dev_ptr,
            self.bf16_ag_slot_stride,
            outer_rows,
            inner_elems,
            self.rank,
            self.world_size,
            self.blocks,
            self.threads,
            stream_handle,
        )
        return out

    def gather_top2_from_logits(
        self,
        logits: torch.Tensor,
        *,
        local_top2_vals_buf: torch.Tensor,
        local_top2_ids_buf: torch.Tensor,
        out_tokens: torch.Tensor,
        out_logprob_bits: torch.Tensor,
        vocab_per_rank: int,
    ) -> None:
        if self._closed:
            raise RuntimeError("PpuIcnCollectives is closed")
        rows = logits.shape[0]
        if rows <= 0:
            return
        if rows > self.max_top2_rows:
            raise ValueError(
                f"gather_top2_from_logits rows={rows} exceeds max_top2_rows={self.max_top2_rows}"
            )
        stream_handle = torch.cuda.current_stream(self.device).cuda_stream
        self._ext.launch_local_top2_bf16(
            logits,
            local_top2_vals_buf,
            local_top2_ids_buf,
            rows,
            stream_handle,
        )
        self._ext.launch_icn_tp8_top2_gather(
            local_top2_vals_buf,
            local_top2_ids_buf,
            out_tokens,
            out_logprob_bits,
            self.top2_step_counter,
            self.base_peer_ptrs,
            self._abort_dev_ptr,
            self.top2_region_off,
            rows,
            self.max_top2_rows,
            self.rank,
            self.world_size,
            vocab_per_rank,
            stream_handle,
        )

    def signal_abort(self) -> None:
        mm = self._abort_mm
        if mm is not None and not mm.closed and len(mm) >= 8:
            struct.pack_into("<Q", mm, 0, 1)

    def close(self, *, abort: bool = False) -> None:
        if self._closed:
            return
        if abort:
            self.signal_abort()
        with contextlib.suppress(BaseException):
            torch.cuda.synchronize(self.device)
        for peer, ptr in enumerate(self._peer_ipc_ptrs_host):
            if peer != self.rank and ptr != 0:
                self._ext.close_ipc_handle(ptr)
        self._peer_ipc_ptrs_host.clear()
        if not abort and dist.is_initialized():
            with contextlib.suppress(BaseException):
                dist.barrier(group=self.exchange_group)
        if self._my_ipc_ptr != 0:
            self._ext.free_ipc_buffer(self._my_ipc_ptr)
            self._my_ipc_ptr = 0
        if self._abort_dev_ptr != 0 and self._abort_host_addr != 0:
            self._ext.unregister_mapped_host(self._abort_host_addr)
            self._abort_dev_ptr = 0
            self._abort_host_addr = 0
        if self._abort_mm is not None and not self._abort_mm.closed:
            self._abort_mm.close()
            self._abort_mm = None
        if self._owns_abort_unlink:
            try:
                os.unlink(self._abort_shm_path)
            except FileNotFoundError:
                pass
            self._owns_abort_unlink = False
        self._closed = True
