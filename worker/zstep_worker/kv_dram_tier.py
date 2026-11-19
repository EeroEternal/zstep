# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT

"""NUMA-local Pinned Host DRAM KV cache arena and Copy-Engine transfer engine.

Executes `StepOpcode::KvTransfer` (`opcode = 4`) commands issued by the Rust master
to demote HBM LRU victims (`direction = 0`, D2H) into Exclusive Host DRAM and reload
DRAM-resident prefix blocks (`direction = 1`, H2D) into the 10% HBM Loading Area.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging

import numpy as np
import torch

from zstep_worker.cuda_driver import CudaDriverGuard, load_cuda_driver
from zstep_worker.numa_pin import bind_memory_to_numa_node

_LOG = logging.getLogger(__name__)

KV_TRANSFER_DIR_D2H: int = 0
KV_TRANSFER_DIR_H2D: int = 1


@dataclass(frozen=True)
class _LayerSliceSpec:
    """Flattened byte-view layout for one GPU KV cache layer buffer."""

    gpu_flat: torch.Tensor  # shape: [total_hbm_blocks, bytes_per_block], dtype=uint8
    dram_offset: int
    bytes_per_block: int


class HostDramKvTier:
    """NUMA-pinned Host DRAM KV arena backed by a dedicated hardware Copy Engine stream."""

    def __init__(
        self,
        kv_raw_buffers: Sequence[torch.Tensor],
        *,
        total_hbm_blocks: int,
        num_dram_blocks: int,
        device: torch.device,
        numa_node: int = -1,
        enable_driver_batch_memcpy: bool = True,
        cuda_driver: CudaDriverGuard | None = None,
    ) -> None:
        if total_hbm_blocks <= 0:
            raise ValueError(f"total_hbm_blocks must be > 0 (got {total_hbm_blocks})")
        if num_dram_blocks <= 0:
            raise ValueError(f"num_dram_blocks must be > 0 (got {num_dram_blocks})")

        self.total_hbm_blocks: int = int(total_hbm_blocks)
        self.num_dram_blocks: int = int(num_dram_blocks)
        self.device: torch.device = device
        self.numa_node: int = int(numa_node)
        self._layers: list[_LayerSliceSpec] = []

        offset = 0
        for idx, buf in enumerate(kv_raw_buffers):
            if buf.shape[0] % self.total_hbm_blocks != 0:
                raise ValueError(
                    f"KV buffer #{idx} dim0={buf.shape[0]} is not divisible by "
                    f"total_hbm_blocks={self.total_hbm_blocks}"
                )
            if not buf.is_contiguous():
                raise ValueError(
                    f"KV buffer #{idx} must be contiguous so HostDramKvTier aliases live GPU memory"
                )
            flat = buf.view(torch.uint8).view(self.total_hbm_blocks, -1)
            bpb = int(flat.shape[1])
            self._layers.append(
                _LayerSliceSpec(gpu_flat=flat, dram_offset=offset, bytes_per_block=bpb)
            )
            offset += bpb

        self.bytes_per_block: int = offset
        self.total_dram_bytes: int = self.num_dram_blocks * self.bytes_per_block
        pin = device.type == "cuda" and torch.cuda.is_available()
        self.dram_arena: torch.Tensor = torch.empty(
            (self.num_dram_blocks, self.bytes_per_block),
            dtype=torch.uint8,
            pin_memory=pin,
        )
        self.numa_bound: bool = bind_memory_to_numa_node(
            int(self.dram_arena.data_ptr()),
            self.total_dram_bytes,
            self.numa_node,
        )
        self._copy_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream(device=device) if pin else None
        )
        self._cuda_driver: CudaDriverGuard | None = None
        self._compute_ready_event: int = 0
        self._copy_done_event: int = 0
        if enable_driver_batch_memcpy:
            driver = cuda_driver if cuda_driver is not None else (load_cuda_driver() if pin else None)
            if driver is not None and driver.has_event_ops:
                ready_ev = driver.create_event()
                try:
                    done_ev = driver.create_event()
                except BaseException:
                    try:
                        driver.destroy_event(ready_ev)
                    except Exception as cleanup_exc:
                        _LOG.warning(
                            "HostDramKvTier partial-init destroy_event(%s) failed: %s",
                            ready_ev,
                            cleanup_exc,
                        )
                    raise
                self._cuda_driver = driver
                self._compute_ready_event = ready_ev
                self._copy_done_event = done_ev
        self._closed: bool = False
        self.offload_ops: int = 0
        self.reload_ops: int = 0
        self.batch_memcpy_calls: int = 0
        self.batch_memcpy_segments: int = 0
        self.last_batch_mode: str = "torch_copy"
        _LOG.info(
            "Initialized HostDramKvTier: blocks=%d, layers=%d, bytes_per_block=%d "
            "(total=%.2f GiB), pinned=%s, numa_node=%d, numa_bound=%s, driver_batch=%s",
            self.num_dram_blocks,
            len(self._layers),
            self.bytes_per_block,
            self.total_dram_bytes / (1024.0**3),
            pin,
            self.numa_node,
            self.numa_bound,
            self._cuda_driver.batch_memcpy_version if self._cuda_driver is not None else 0,
        )

    def close(self) -> None:
        """Destroy any pre-allocated CUDA driver events safely and idempotently."""
        if self._closed:
            return
        self._closed = True
        driver = self._cuda_driver
        self._cuda_driver = None
        ready_ev = self._compute_ready_event
        done_ev = self._copy_done_event
        self._compute_ready_event = 0
        self._copy_done_event = 0
        if driver is not None:
            for ev in (ready_ev, done_ev):
                if ev:
                    try:
                        driver.destroy_event(ev)
                    except Exception as exc:
                        _LOG.warning("HostDramKvTier.destroy_event(%s) failed: %s", ev, exc)

    def execute_transfers(
        self,
        hbm_page_ids: np.ndarray,
        dram_slot_ids: np.ndarray,
        directions: np.ndarray,
        count: int,
    ) -> None:
        """Execute `count` D2H/H2D KV page transfers on the dedicated Copy Engine stream.

        Orders the Copy Engine stream after the current compute stream (so any prior HBM
        writes are visible to D2H) and orders the current compute stream after the Copy
        Engine stream (so reloaded H2D pages are visible to subsequent prefill/decode steps,
        and `end_event.record()` on the compute stream waits for D2H completion).

        When `CudaDriverGuard` is active, uses `PyDLL` for microsecond `cuEventRecord` /
        `cuStreamWaitEvent` handshakes and `CDLL` `cuMemcpyBatchAsync` (releasing the GIL)
        for all `count * num_layers` page copies in a single driver call.
        """
        if count <= 0 or not self._layers:
            return
        if self._copy_stream is not None and self._cuda_driver is not None:
            compute_stream = torch.cuda.current_stream(self.device)
            self._dispatch_driver_batch(
                hbm_page_ids,
                dram_slot_ids,
                directions,
                count,
                compute_stream_ptr=int(compute_stream.cuda_stream),
                copy_stream_ptr=int(self._copy_stream.cuda_stream),
            )
            return
        if self._copy_stream is not None:
            compute_stream = torch.cuda.current_stream(self.device)
            self._copy_stream.wait_stream(compute_stream)
            with torch.cuda.stream(self._copy_stream):
                self._dispatch_copies(hbm_page_ids, dram_slot_ids, directions, count)
            compute_stream.wait_stream(self._copy_stream)
        else:
            self._dispatch_copies(hbm_page_ids, dram_slot_ids, directions, count)

    def _dispatch_driver_batch(
        self,
        hbm_page_ids: np.ndarray,
        dram_slot_ids: np.ndarray,
        directions: np.ndarray,
        count: int,
        *,
        compute_stream_ptr: int,
        copy_stream_ptr: int,
    ) -> None:
        driver = self._cuda_driver
        assert driver is not None
        dram_base = int(self.dram_arena.data_ptr())
        dsts: list[int] = []
        srcs: list[int] = []
        sizes: list[int] = []
        d2h_ops = 0
        h2d_ops = 0

        for i in range(count):
            hbm_id = int(hbm_page_ids[i])
            dram_id = int(dram_slot_ids[i])
            direction = int(directions[i])
            if not (0 <= hbm_id < self.total_hbm_blocks):
                raise ValueError(
                    f"KvTransfer hbm_page_id={hbm_id} out of bounds [0, {self.total_hbm_blocks})"
                )
            if not (0 <= dram_id < self.num_dram_blocks):
                raise ValueError(
                    f"KvTransfer dram_slot_id={dram_id} out of bounds [0, {self.num_dram_blocks})"
                )
            dram_row_ptr = dram_base + dram_id * self.bytes_per_block
            if direction == KV_TRANSFER_DIR_D2H:
                for layer in self._layers:
                    bpb = layer.bytes_per_block
                    dsts.append(dram_row_ptr + layer.dram_offset)
                    srcs.append(int(layer.gpu_flat.data_ptr()) + hbm_id * bpb)
                    sizes.append(bpb)
                d2h_ops += 1
            elif direction == KV_TRANSFER_DIR_H2D:
                for layer in self._layers:
                    bpb = layer.bytes_per_block
                    dsts.append(int(layer.gpu_flat.data_ptr()) + hbm_id * bpb)
                    srcs.append(dram_row_ptr + layer.dram_offset)
                    sizes.append(bpb)
                h2d_ops += 1
            else:
                raise ValueError(f"Unknown KvTransfer direction={direction}")

        driver.record_event(self._compute_ready_event, compute_stream_ptr)
        driver.stream_wait_event(copy_stream_ptr, self._compute_ready_event)
        mode = driver.memcpy_batch_async(dsts, srcs, sizes, copy_stream_ptr)
        driver.record_event(self._copy_done_event, copy_stream_ptr)
        driver.stream_wait_event(compute_stream_ptr, self._copy_done_event)

        self.offload_ops += d2h_ops
        self.reload_ops += h2d_ops
        self.batch_memcpy_calls += 1
        self.batch_memcpy_segments += len(dsts)
        self.last_batch_mode = mode

    def _dispatch_copies(
        self,
        hbm_page_ids: np.ndarray,
        dram_slot_ids: np.ndarray,
        directions: np.ndarray,
        count: int,
    ) -> None:
        non_blocking = self._copy_stream is not None
        for i in range(count):
            hbm_id = int(hbm_page_ids[i])
            dram_id = int(dram_slot_ids[i])
            direction = int(directions[i])
            if not (0 <= hbm_id < self.total_hbm_blocks):
                raise ValueError(
                    f"KvTransfer hbm_page_id={hbm_id} out of bounds [0, {self.total_hbm_blocks})"
                )
            if not (0 <= dram_id < self.num_dram_blocks):
                raise ValueError(
                    f"KvTransfer dram_slot_id={dram_id} out of bounds [0, {self.num_dram_blocks})"
                )
            dram_row = self.dram_arena[dram_id]
            if direction == KV_TRANSFER_DIR_D2H:
                for layer in self._layers:
                    dst = dram_row[layer.dram_offset : layer.dram_offset + layer.bytes_per_block]
                    dst.copy_(layer.gpu_flat[hbm_id], non_blocking=non_blocking)
                self.offload_ops += 1
            elif direction == KV_TRANSFER_DIR_H2D:
                for layer in self._layers:
                    src = dram_row[layer.dram_offset : layer.dram_offset + layer.bytes_per_block]
                    layer.gpu_flat[hbm_id].copy_(src, non_blocking=non_blocking)
                self.reload_ops += 1
            else:
                raise ValueError(f"Unknown KvTransfer direction={direction}")
