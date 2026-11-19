# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT

"""SM-driven UVA Pinned Host -> GPU H2D metadata copier (`sm_h2d`, Issue #40 Proposal 2).

When Host DRAM KV offload saturates the PCIe hardware Copy Engine (CE), lightweight
per-step metadata copies (`token_ids`, `positions`, `seq_lens`, `slot_mapping`,
`lookback`) queued on `cudaMemcpyAsync` can stall behind multi-megabyte KV transfers.

`SmH2dCopier` launches a minimal Triton kernel on the **Compute Stream** that reads
directly from the UVA-mapped Pinned Host pointer across PCIe using SMs, physically
isolating control-plane metadata from bulk KV Copy Engine traffic.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _sm_h2d_copy_i32_kernel(
        src_ptr_i64,
        dst_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        src_ptr = tl.cast(src_ptr_i64, tl.pointer_type(tl.int32))
        vals = tl.load(src_ptr + offsets, mask=mask)
        tl.store(dst_ptr + offsets, vals, mask=mask)

    @triton.jit
    def _sm_h2d_copy_i64_kernel(
        src_ptr_i64,
        dst_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        src_ptr = tl.cast(src_ptr_i64, tl.pointer_type(tl.int64))
        vals = tl.load(src_ptr + offsets, mask=mask)
        tl.store(dst_ptr + offsets, vals, mask=mask)


class SmH2dCopier:
    """Copy small contiguous `int32`/`int64` metadata slices from Pinned Host -> GPU via SMs.

    Falls back transparently to `dst.copy_(src, non_blocking=True)` when disabled,
    on non-CUDA devices, when Triton is unavailable, or for non-pinned/other-dtype tensors.
    """

    def __init__(self, *, enabled: bool = True, block_size: int = 256) -> None:
        self.enabled: bool = bool(enabled)
        self.block_size: int = int(block_size)
        self.sm_copies: int = 0
        self.fallback_copies: int = 0

    @property
    def active(self) -> bool:
        return self.enabled and _HAS_TRITON and torch.cuda.is_available()

    def copy_async(self, dst_gpu: torch.Tensor, src_cpu: torch.Tensor) -> None:
        """Copy contiguous `int32`/`int64` tensor `src_cpu` into `dst_gpu` via SMs when active."""
        n_elements = int(dst_gpu.numel())
        if n_elements <= 0:
            return
        if (
            self.enabled
            and _HAS_TRITON
            and dst_gpu.is_cuda
            and src_cpu.device.type == "cpu"
            and src_cpu.is_pinned()
            and dst_gpu.dtype == src_cpu.dtype
            and dst_gpu.dtype in (torch.int32, torch.int64)
            and dst_gpu.numel() == src_cpu.numel()
            and dst_gpu.is_contiguous()
            and src_cpu.is_contiguous()
        ):
            grid = ((n_elements + self.block_size - 1) // self.block_size,)
            kernel = (
                _sm_h2d_copy_i32_kernel
                if dst_gpu.dtype == torch.int32
                else _sm_h2d_copy_i64_kernel
            )
            kernel[grid](
                int(src_cpu.data_ptr()),
                dst_gpu,
                n_elements,
                BLOCK_SIZE=self.block_size,
            )
            self.sm_copies += 1
            return
        dst_gpu.copy_(src_cpu, non_blocking=True)
        self.fallback_copies += 1
