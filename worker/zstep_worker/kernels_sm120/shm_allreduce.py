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

"""CUDA-graph-safe host-mapped two-shot BF16 AllReduce for non-P2P systems."""

from __future__ import annotations

import ctypes
import os
from multiprocessing import shared_memory

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline

_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

std::uintptr_t register_mapped_host(std::uintptr_t host_address, int64_t bytes) {
    void* host = reinterpret_cast<void*>(host_address);
    C10_CUDA_CHECK(cudaHostRegister(
        host, static_cast<size_t>(bytes), cudaHostRegisterMapped));
    void* device = nullptr;
    cudaError_t status = cudaHostGetDevicePointer(&device, host, 0);
    if (status != cudaSuccess) {
        cudaHostUnregister(host);
        C10_CUDA_CHECK(status);
    }
    return reinterpret_cast<std::uintptr_t>(device);
}

void unregister_mapped_host(std::uintptr_t host_address) {
    C10_CUDA_CHECK(cudaHostUnregister(reinterpret_cast<void*>(host_address)));
}

void launch_host_mapped_two_shot(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor control,
    std::uintptr_t host_device_address,
    int64_t inputs_offset,
    int64_t reduced_offset,
    int64_t rank_stride_bytes,
    int64_t rank,
    int64_t world_size,
    int64_t blocks,
    int64_t threads);
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ unsigned long long device_load(
    unsigned long long* pointer) {
    return atomicAdd(pointer, 0ULL);
}

__device__ __forceinline__ unsigned long long system_load(
    unsigned long long* pointer) {
    return atomicAdd_system(pointer, 0ULL);
}

__global__ void host_mapped_copy_kernel(
    const __nv_bfloat16* input,
    unsigned long long* control,
    unsigned long long* input_arrivals,
    char* host_base,
    int64_t inputs_offset,
    int64_t rank_stride_bytes,
    int64_t elements,
    int rank,
    int world_size) {
    constexpr int copy_blocks_slot = 1;
    constexpr int copy_ready_slot = 3;
    const unsigned long long target = control[0] + 1;
    const int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t thread_count = static_cast<int64_t>(gridDim.x) * blockDim.x;
    __nv_bfloat16* local_input = reinterpret_cast<__nv_bfloat16*>(
        host_base + inputs_offset + static_cast<int64_t>(rank) * rank_stride_bytes);
    for (int64_t index = thread; index < elements; index += thread_count) {
        local_input[index] = input[index];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        atomicAdd(control + copy_blocks_slot, 1ULL);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        while (device_load(control + copy_blocks_slot) < target * gridDim.x) {
            __nanosleep(128);
        }
        atomicAdd_system(input_arrivals + rank, 1ULL);
        for (int peer = 0; peer < world_size; ++peer) {
            while (system_load(input_arrivals + peer) < target) {
                __nanosleep(128);
            }
        }
        __threadfence_system();
        atomicAdd(control + copy_ready_slot, 1ULL);
    }
}

__global__ void host_mapped_reduce_kernel(
    unsigned long long* control,
    unsigned long long* reduced_arrivals,
    char* host_base,
    int64_t inputs_offset,
    int64_t reduced_offset,
    int64_t rank_stride_bytes,
    int64_t elements,
    int rank,
    int world_size) {
    constexpr int reduce_blocks_slot = 2;
    constexpr int reduce_ready_slot = 4;
    const unsigned long long target = control[0] + 1;
    const int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t thread_count = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t shard_begin = elements * rank / world_size;
    const int64_t shard_end = elements * (rank + 1) / world_size;
    __nv_bfloat16* local_reduced = reinterpret_cast<__nv_bfloat16*>(
        host_base + reduced_offset + static_cast<int64_t>(rank) * rank_stride_bytes);
    for (int64_t index = shard_begin + thread; index < shard_end; index += thread_count) {
        float sum = 0.0f;
        for (int peer = 0; peer < world_size; ++peer) {
            const __nv_bfloat16* peer_input = reinterpret_cast<const __nv_bfloat16*>(
                host_base + inputs_offset + static_cast<int64_t>(peer) * rank_stride_bytes);
            sum += __bfloat162float(peer_input[index]);
        }
        local_reduced[index] = __float2bfloat16_rn(sum);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        atomicAdd(control + reduce_blocks_slot, 1ULL);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        while (device_load(control + reduce_blocks_slot) < target * gridDim.x) {
            __nanosleep(128);
        }
        atomicAdd_system(reduced_arrivals + rank, 1ULL);
        for (int peer = 0; peer < world_size; ++peer) {
            while (system_load(reduced_arrivals + peer) < target) {
                __nanosleep(128);
            }
        }
        __threadfence_system();
        atomicAdd(control + reduce_ready_slot, 1ULL);
    }
}

__global__ void host_mapped_gather_kernel(
    __nv_bfloat16* output,
    unsigned long long* control,
    char* host_base,
    int64_t reduced_offset,
    int64_t rank_stride_bytes,
    int64_t elements,
    int world_size) {
    const int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t thread_count = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int owner = 0; owner < world_size; ++owner) {
        const int64_t owner_begin = elements * owner / world_size;
        const int64_t owner_end = elements * (owner + 1) / world_size;
        const __nv_bfloat16* owner_reduced = reinterpret_cast<const __nv_bfloat16*>(
            host_base + reduced_offset + static_cast<int64_t>(owner) * rank_stride_bytes);
        for (int64_t index = owner_begin + thread; index < owner_end; index += thread_count) {
            output[index] = owner_reduced[index];
        }
    }
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd(control, 1ULL);
    }
}

}  // namespace

void launch_host_mapped_two_shot(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor control,
    std::uintptr_t host_device_address,
    int64_t inputs_offset,
    int64_t reduced_offset,
    int64_t rank_stride_bytes,
    int64_t rank,
    int64_t world_size,
    int64_t blocks,
    int64_t threads) {
    TORCH_CHECK(input.is_cuda() && output.is_cuda(), "input/output must be CUDA tensors");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be BF16");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be BF16");
    TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "input/output must be contiguous");
    TORCH_CHECK(input.numel() == output.numel(), "input/output sizes must match");
    TORCH_CHECK(control.is_cuda() && control.scalar_type() == at::kLong, "control must be int64 CUDA storage");
    TORCH_CHECK(control.numel() >= 5, "control must have five monotonic slots");
    TORCH_CHECK(input.numel() * input.element_size() <= rank_stride_bytes, "input exceeds SHM capacity");
    TORCH_CHECK(world_size >= 2 && rank >= 0 && rank < world_size, "invalid rank/world size");

    auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
    const int grid = static_cast<int>(blocks);
    const int block = static_cast<int>(threads);
    auto* control_pointer = reinterpret_cast<unsigned long long*>(control.data_ptr());
    auto* arrivals = reinterpret_cast<unsigned long long*>(host_device_address);
    auto* reduced_arrivals = reinterpret_cast<unsigned long long*>(host_device_address + 256);
    auto* host_base = reinterpret_cast<char*>(host_device_address);
    host_mapped_copy_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        control_pointer,
        arrivals,
        host_base,
        inputs_offset,
        rank_stride_bytes,
        input.numel(),
        static_cast<int>(rank),
        static_cast<int>(world_size));
    host_mapped_reduce_kernel<<<grid, block, 0, stream>>>(
        control_pointer,
        reduced_arrivals,
        host_base,
        inputs_offset,
        reduced_offset,
        rank_stride_bytes,
        input.numel(),
        static_cast<int>(rank),
        static_cast<int>(world_size));
    host_mapped_gather_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        control_pointer,
        host_base,
        reduced_offset,
        rank_stride_bytes,
        input.numel(),
        static_cast<int>(world_size));
    C10_CUDA_CHECK(cudaGetLastError());
}
"""

_EXTENSION = None


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def get_host_mapped_allreduce_extension():
    """Build or load the inline CUDA extension without initializing a collective."""
    global _EXTENSION
    if _EXTENSION is None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        _EXTENSION = load_inline(
            name="zstep_shm_allreduce_sm120_ext",
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            functions=[
                "register_mapped_host",
                "unregister_mapped_host",
                "launch_host_mapped_two_shot",
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "-gencode=arch=compute_120,code=sm_120",
            ],
            with_cuda=True,
            verbose=False,
        )
    return _EXTENSION


class HostMappedShmAllReduce:
    """One ordered host-mapped AllReduce channel shared by all ranks.

    All launches, including CUDA Graph replays, advance device and shared-host
    monotonic counters. Only block zero waits while other blocks retire, so
    progress does not require simultaneous grid residency. Callers must keep
    collectives in the same order on every rank and use one stream per channel.
    """

    def __init__(
        self,
        *,
        name: str,
        rank: int,
        world_size: int,
        capacity_elements: int,
        blocks: int,
        threads: int,
        exchange_group: dist.ProcessGroup,
    ) -> None:
        if world_size < 2 or not 0 <= rank < world_size:
            raise ValueError("rank/world_size are invalid")
        if capacity_elements <= 0:
            raise ValueError("capacity_elements must be positive")
        if threads <= 0 or threads > 1024 or threads % 32 != 0:
            raise ValueError("threads must be a warp multiple in [32, 1024]")
        device = torch.device("cuda", torch.cuda.current_device())
        if blocks <= 0:
            raise ValueError("blocks must be positive")

        self.name = name
        self.rank = rank
        self.world_size = world_size
        self.capacity_elements = capacity_elements
        self.blocks = blocks
        self.threads = threads
        self.device = device
        self.rank_stride_bytes = _align_up(capacity_elements * 2, 256)
        self.inputs_offset = 4096
        self.reduced_offset = _align_up(
            self.inputs_offset + world_size * self.rank_stride_bytes, 4096
        )
        self.total_bytes = _align_up(
            self.reduced_offset + world_size * self.rank_stride_bytes, 4096
        )
        self._closed = False
        self._registered = False
        self._extension = get_host_mapped_allreduce_extension()

        if rank == 0:
            self._shm = shared_memory.SharedMemory(
                name=name, create=True, size=self.total_bytes
            )
            self._shm.buf[:512] = bytes(512)
        dist.barrier(group=exchange_group)
        if rank != 0:
            self._shm = shared_memory.SharedMemory(name=name, create=False)
        if self._shm.size != self.total_bytes:
            raise RuntimeError("shared-memory geometry differs across ranks")
        dist.barrier(group=exchange_group)

        self._host_address = ctypes.addressof(ctypes.c_char.from_buffer(self._shm.buf))
        ctypes.memset(
            self._host_address + self.inputs_offset + rank * self.rank_stride_bytes,
            0,
            self.rank_stride_bytes,
        )
        ctypes.memset(
            self._host_address + self.reduced_offset + rank * self.rank_stride_bytes,
            0,
            self.rank_stride_bytes,
        )
        dist.barrier(group=exchange_group)
        self._device_address = self._extension.register_mapped_host(
            self._host_address, self.total_bytes
        )
        self._registered = True
        self.control = torch.zeros(5, dtype=torch.int64, device=device)

    def all_reduce(self, input: torch.Tensor, *, output: torch.Tensor) -> torch.Tensor:
        """Reduce all rank inputs into the caller-owned fixed output tensor."""
        if self._closed:
            raise RuntimeError("AllReduce channel is closed")
        if input.shape != output.shape:
            raise ValueError("input/output shapes must match")
        if input.numel() > self.capacity_elements:
            raise ValueError("input exceeds channel capacity")
        self._extension.launch_host_mapped_two_shot(
            input,
            output,
            self.control,
            self._device_address,
            self.inputs_offset,
            self.reduced_offset,
            self.rank_stride_bytes,
            self.rank,
            self.world_size,
            self.blocks,
            self.threads,
        )
        return output

    def close(self) -> None:
        """Release CUDA registration and the POSIX mapping after all ranks drain."""
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        if self._registered:
            self._extension.unregister_mapped_host(self._host_address)
            self._registered = False
        self._shm.close()
        if self.rank == 0:
            self._shm.unlink()
        self._closed = True

    def __enter__(self) -> HostMappedShmAllReduce:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
