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

"""CUDA-Graph-safe host-mapped two-shot BF16 AllReduce for non-P2P (`CNS`) topologies.

Uses a single fused kernel with per-block independent chunk ownership, 16-byte (`uint4` =
8x BF16) vectorized PCIe streaming (`st.global.wt` / `ld.global.cv`), PTX `.sys` scope
release/acquire flag barriers across 8 peers in parallel within warp 0, and a 4-slot
ring buffer so back-to-back CUDA Graph replays never race or deadlock.
"""

from __future__ import annotations

import ctypes
import mmap
import os

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline

NUM_RING_SLOTS = 4
MAX_RANKS = 8
MAX_BLOCKS = 64

_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
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
    int64_t data_base_offset,
    int64_t slot_stride_bytes,
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

constexpr int kNumRingSlots = 4;
constexpr int kMaxRanks = 8;
constexpr int kMaxBlocks = 64;

__device__ __forceinline__ uint64_t sys_load_acquire_u64(const uint64_t* addr) {
    uint64_t val;
    asm volatile("ld.global.acquire.sys.b64 %0, [%1];" : "=l"(val) : "l"(addr) : "memory");
    return val;
}

__device__ __forceinline__ void sys_store_release_u64(uint64_t* addr, uint64_t val) {
    asm volatile("st.global.release.sys.b64 [%0], %1;" :: "l"(addr), "l"(val) : "memory");
}

__device__ __forceinline__ uint4 ld_cv_uint4(const uint4* addr) {
    uint4 v;
    asm volatile(
        "ld.global.cv.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(addr)
        : "memory");
    return v;
}

__device__ __forceinline__ void st_wt_uint4(uint4* addr, uint4 v) {
    asm volatile(
        "st.global.wt.v4.u32 [%0], {%1, %2, %3, %4};"
        :: "l"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
        : "memory");
}

union Pack128Bf16 {
    uint4 u4;
    __nv_bfloat162 pairs[4];
};

__global__ void host_mapped_two_shot_fused_kernel(
    const uint4* __restrict__ input_vec,
    uint4* __restrict__ output_vec,
    uint64_t* __restrict__ control,
    char* __restrict__ host_base,
    int64_t data_base_offset,
    int64_t slot_stride_bytes,
    int64_t rank_stride_bytes,
    int64_t num_vecs,
    int rank,
    int world_size) {
    // Flag layout in first 12 KiB of SHM (each flag cacheline-padded to 8B within 64B per block):
    // [0 .. 4096):   in_flags[kMaxBlocks][kMaxRanks]  (64 * 8 * 8 = 4096 B)
    // [4096 .. 8192): red_flags[kMaxBlocks][kMaxRanks] (64 * 8 * 8 = 4096 B)
    // [8192 .. 12288): done_flags[kMaxBlocks][kMaxRanks] (64 * 8 * 8 = 4096 B)
    uint64_t* in_flags = reinterpret_cast<uint64_t*>(host_base) + blockIdx.x * kMaxRanks;
    uint64_t* red_flags = reinterpret_cast<uint64_t*>(host_base + 4096) + blockIdx.x * kMaxRanks;
    uint64_t* done_flags = reinterpret_cast<uint64_t*>(host_base + 8192) + blockIdx.x * kMaxRanks;

    __shared__ uint64_t s_step;
    if (threadIdx.x == 0) {
        const uint64_t next_step = control[blockIdx.x] + 1ULL;
        control[blockIdx.x] = next_step;
        s_step = next_step;
    }
    __syncthreads();
    const uint64_t step = s_step;
    const int slot = static_cast<int>((step - 1ULL) & (kNumRingSlots - 1));

    // Partition `num_vecs` across blocks so each block `b` synchronizes ONLY with peer block `b`.
    const int64_t chunk_begin = (num_vecs * static_cast<int64_t>(blockIdx.x)) / gridDim.x;
    const int64_t chunk_end = (num_vecs * static_cast<int64_t>(blockIdx.x + 1)) / gridDim.x;
    const int64_t chunk_len = chunk_end - chunk_begin;

    // Ring-slot reuse guard: ensure all peers finished reading `slot` from `step - kNumRingSlots`.
    if (step > static_cast<uint64_t>(kNumRingSlots)) {
        const uint64_t min_done = step - static_cast<uint64_t>(kNumRingSlots);
        if (threadIdx.x < world_size) {
            while (sys_load_acquire_u64(done_flags + threadIdx.x) < min_done) {
                __nanosleep(64);
            }
        }
        __syncthreads();
    }

    char* slot_base = host_base + data_base_offset + static_cast<int64_t>(slot) * slot_stride_bytes;
    char* slot_inputs_base = slot_base;
    char* slot_reduced_base = slot_base + static_cast<int64_t>(world_size) * rank_stride_bytes;

    // Stage 1: Copy local VRAM chunk -> mapped SHM input buffer.
    uint4* local_shm_in = reinterpret_cast<uint4*>(
        slot_inputs_base + static_cast<int64_t>(rank) * rank_stride_bytes);
    for (int64_t idx = chunk_begin + threadIdx.x; idx < chunk_end; idx += blockDim.x) {
        st_wt_uint4(local_shm_in + idx, input_vec[idx]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        sys_store_release_u64(in_flags + rank, step);
    }
    if (threadIdx.x < world_size) {
        while (sys_load_acquire_u64(in_flags + threadIdx.x) < step) {
            __nanosleep(64);
        }
    }
    __syncthreads();

    // Stage 2: Reduce-scatter this rank's 1/world_size slice of [chunk_begin, chunk_end).
    const int64_t sub_begin = chunk_begin + (chunk_len * rank) / world_size;
    const int64_t sub_end = chunk_begin + (chunk_len * (rank + 1)) / world_size;
    uint4* local_shm_red = reinterpret_cast<uint4*>(
        slot_reduced_base + static_cast<int64_t>(rank) * rank_stride_bytes);

    for (int64_t idx = sub_begin + threadIdx.x; idx < sub_end; idx += blockDim.x) {
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int peer = 0; peer < kMaxRanks; ++peer) {
            if (peer < world_size) {
                const uint4* peer_in = reinterpret_cast<const uint4*>(
                    slot_inputs_base + static_cast<int64_t>(peer) * rank_stride_bytes);
                Pack128Bf16 p;
                p.u4 = ld_cv_uint4(peer_in + idx);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f2 = __bfloat1622float2(p.pairs[j]);
                    acc[2 * j] += f2.x;
                    acc[2 * j + 1] += f2.y;
                }
            }
        }
        Pack128Bf16 out_pack;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            out_pack.pairs[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
        }
        st_wt_uint4(local_shm_red + idx, out_pack.u4);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        sys_store_release_u64(red_flags + rank, step);
    }
    if (threadIdx.x < world_size) {
        while (sys_load_acquire_u64(red_flags + threadIdx.x) < step) {
            __nanosleep(64);
        }
    }
    __syncthreads();

    // Stage 3: AllGather all peers' reduced sub-slices directly into VRAM `output_vec`.
    for (int owner = 0; owner < world_size; ++owner) {
        const int64_t owner_begin = chunk_begin + (chunk_len * owner) / world_size;
        const int64_t owner_end = chunk_begin + (chunk_len * (owner + 1)) / world_size;
        const uint4* owner_red = reinterpret_cast<const uint4*>(
            slot_reduced_base + static_cast<int64_t>(owner) * rank_stride_bytes);
        for (int64_t idx = owner_begin + threadIdx.x; idx < owner_end; idx += blockDim.x) {
            output_vec[idx] = ld_cv_uint4(owner_red + idx);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        sys_store_release_u64(done_flags + rank, step);
    }
}

}  // namespace

void launch_host_mapped_two_shot(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor control,
    std::uintptr_t host_device_address,
    int64_t data_base_offset,
    int64_t slot_stride_bytes,
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
    TORCH_CHECK(input.numel() % 8 == 0, "input numel must be divisible by 8 for 128-bit vectorization");
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(input.data_ptr()) % 16 == 0 &&
        reinterpret_cast<std::uintptr_t>(output.data_ptr()) % 16 == 0,
        "input/output pointers must be 16-byte aligned");
    TORCH_CHECK(control.is_cuda() && control.scalar_type() == at::kLong, "control must be int64 CUDA tensor");
    TORCH_CHECK(blocks > 0 && blocks <= kMaxBlocks, "blocks must be in [1, 64]");
    TORCH_CHECK(control.numel() >= blocks, "control tensor must hold at least `blocks` counters");
    TORCH_CHECK(world_size >= 2 && world_size <= kMaxRanks && rank >= 0 && rank < world_size, "invalid rank/world_size");
    TORCH_CHECK(input.numel() * 2 <= rank_stride_bytes, "input exceeds SHM capacity");

    auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
    host_mapped_two_shot_fused_kernel<<<static_cast<int>(blocks), static_cast<int>(threads), 0, stream>>>(
        reinterpret_cast<const uint4*>(input.data_ptr()),
        reinterpret_cast<uint4*>(output.data_ptr()),
        reinterpret_cast<uint64_t*>(control.data_ptr()),
        reinterpret_cast<char*>(host_device_address),
        data_base_offset,
        slot_stride_bytes,
        rank_stride_bytes,
        input.numel() / 8,
        static_cast<int>(rank),
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
            name="zstep_shm_allreduce_sm120_v2_ext",
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
    """Ordered host-mapped 4-slot ring Two-Shot AllReduce channel shared by all ranks."""

    def __init__(
        self,
        *,
        name: str,
        rank: int,
        world_size: int,
        capacity_elements: int,
        blocks: int,
        threads: int,
        exchange_group: dist.ProcessGroup | None,
        create_shm: bool = False,
    ) -> None:
        if world_size < 2 or world_size > MAX_RANKS or not 0 <= rank < world_size:
            raise ValueError(f"rank={rank} / world_size={world_size} invalid (max {MAX_RANKS})")
        if capacity_elements <= 0 or capacity_elements % 8 != 0:
            raise ValueError("capacity_elements must be positive and divisible by 8")
        if not 1 <= blocks <= MAX_BLOCKS:
            raise ValueError(f"blocks must be in [1, {MAX_BLOCKS}]")
        if threads <= 0 or threads > 1024 or threads % 32 != 0:
            raise ValueError("threads must be a warp multiple in [32, 1024]")

        device = torch.device("cuda", torch.cuda.current_device())
        self.name: str = name
        self.rank: int = rank
        self.world_size: int = world_size
        self.capacity_elements: int = capacity_elements
        self.blocks: int = blocks
        self.threads: int = threads
        self.device: torch.device = device
        self.rank_stride_bytes: int = _align_up(capacity_elements * 2, 256)
        self.data_base_offset: int = 16384  # First 16 KiB holds per-block .sys arrival flags
        self.slot_stride_bytes: int = _align_up(
            2 * world_size * self.rank_stride_bytes, 4096
        )
        self.total_bytes: int = _align_up(
            self.data_base_offset + NUM_RING_SLOTS * self.slot_stride_bytes, 4096
        )
        self._closed: bool = False
        self._registered: bool = False
        self._owns_unlink: bool = False
        self._extension = get_host_mapped_allreduce_extension()

        shm_path = f"/dev/shm/{name}"
        if exchange_group is not None:
            if rank == 0:
                fd = os.open(shm_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
                try:
                    os.ftruncate(fd, self.total_bytes)
                    self._mapping = mmap.mmap(
                        fd,
                        self.total_bytes,
                        flags=mmap.MAP_SHARED,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE,
                    )
                finally:
                    os.close(fd)
                self._owns_unlink = True
                ctypes.memset(
                    ctypes.addressof(ctypes.c_char.from_buffer(self._mapping)),
                    0,
                    self.data_base_offset,
                )
            dist.barrier(group=exchange_group)
            if rank != 0:
                fd = os.open(shm_path, os.O_RDWR)
                try:
                    self._mapping = mmap.mmap(
                        fd,
                        self.total_bytes,
                        flags=mmap.MAP_SHARED,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE,
                    )
                finally:
                    os.close(fd)
            dist.barrier(group=exchange_group)
        else:
            flags = (os.O_CREAT | os.O_RDWR | os.O_TRUNC) if create_shm else os.O_RDWR
            fd = os.open(shm_path, flags, 0o600)
            try:
                if create_shm:
                    os.ftruncate(fd, self.total_bytes)
                self._mapping = mmap.mmap(
                    fd,
                    self.total_bytes,
                    flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                )
            finally:
                os.close(fd)
            self._owns_unlink = create_shm
            if create_shm:
                ctypes.memset(
                    ctypes.addressof(ctypes.c_char.from_buffer(self._mapping)),
                    0,
                    self.data_base_offset,
                )

        self._shm_path: str = shm_path
        self._host_address: int = ctypes.addressof(ctypes.c_char.from_buffer(self._mapping))
        self._device_address: int = self._extension.register_mapped_host(
            self._host_address, self.total_bytes
        )
        self._registered = True
        self.control: torch.Tensor = torch.zeros(MAX_BLOCKS, dtype=torch.int64, device=device)

    def all_reduce(self, input: torch.Tensor, *, output: torch.Tensor) -> torch.Tensor:
        """Reduce all rank inputs into `output` in a single CUDA-Graph-safe launch."""
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
            self.data_base_offset,
            self.slot_stride_bytes,
            self.rank_stride_bytes,
            self.rank,
            self.world_size,
            self.blocks,
            self.threads,
        )
        return output

    def close(self) -> None:
        """Release CUDA host mapping and POSIX shared memory handle."""
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        if self._registered:
            self._extension.unregister_mapped_host(self._host_address)
            self._registered = False
        self._mapping.close()
        if self._owns_unlink:
            try:
                os.unlink(self._shm_path)
            except FileNotFoundError:
                pass
        self._closed = True

    def __enter__(self) -> HostMappedShmAllReduce:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
