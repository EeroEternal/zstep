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

"""CUDA-Graph-safe host-mapped BF16 AllReduce & AllGather (`shm_v5`) for non-P2P (`CNS`) PCIe.

In multi-process TP=8 mode (`exchange_group is not None`), each rank allocates and first-touches
its own NUMA-local `/dev/shm/{name}_r{rank}` segment with 64-byte cacheline-isolated `(block, peer)`
arrival flags. Producers push flags via posted PCIe writes (`st.global.release.sys.b64`) into each
peer's local DRAM so consumers spin exclusively on their own local NUMA node (`ld.global.acquire.sys.b64`)
and read their own shard directly from local VRAM.
"""

from __future__ import annotations

import ctypes
import math
import mmap
import os
import struct

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline

from zstep_worker.numa_pin import DeviceNumaTopology, pin_process_to_device_numa

NUM_RING_SLOTS = 4
MAX_RANKS = 8
MAX_BLOCKS = 64
SHM_V5_HEADER_BYTES = 131072
SHM_V5_ABORT_OFFSET = 98304
ARGMAX_SHM_ABORT_OFFSET = 128
DEFAULT_ONE_SHOT_MAX_VECS = 4096  # Up to 64 KiB (e.g. M <= 16 at H=2048 or M <= 6 at H=5120 BF16)

_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <cstdint>
#include <stdexcept>

std::uintptr_t register_mapped_host_cuda(std::uintptr_t host_address, int64_t bytes);
void unregister_mapped_host_cuda(std::uintptr_t host_address);

void launch_host_mapped_two_shot_cuda(
    const void* input_ptr,
    void* output_ptr,
    int64_t* control_ptr,
    std::uintptr_t host_device_address,
    int64_t data_base_offset,
    int64_t slot_stride_bytes,
    int64_t rank_stride_bytes,
    int64_t num_vecs,
    int rank,
    int world_size,
    int blocks,
    int threads,
    std::uintptr_t stream_handle);

void launch_shm_v5_allreduce_cuda(
    const void* input_ptr,
    void* output_ptr,
    uint64_t* control_ptr,
    const std::uintptr_t* rank_shm_dev_ptrs,
    int64_t slot_stride_bytes,
    int64_t num_vecs,
    int rank,
    int world_size,
    int blocks,
    int threads,
    int one_shot_max_vecs,
    std::uintptr_t stream_handle);

void launch_shm_v5_allgather_cuda(
    const void* input_ptr,
    void* output_ptr,
    uint64_t* control_ptr,
    const std::uintptr_t* rank_shm_dev_ptrs,
    int64_t slot_stride_bytes,
    int64_t outer_rows,
    int64_t vecs_per_row,
    int rank,
    int world_size,
    int blocks,
    int threads,
    std::uintptr_t stream_handle);

void launch_local_top2_bf16_cuda(
    const void* logits_bf16,
    float* out_top2_vals,
    int32_t* out_top2_ids,
    int64_t num_rows,
    int64_t stride_row,
    int vocab_per_rank,
    std::uintptr_t stream_handle);

void launch_tp8_argmax_gather_shm_cuda(
    const float* local_top2_vals,
    const int32_t* local_top2_ids,
    int32_t* out_top2_tokens,
    uint32_t* out_top2_logit_bits,
    uint64_t* step_counter,
    std::uintptr_t host_device_address,
    int64_t num_rows,
    int64_t max_rows_per_step,
    int rank,
    int world_size,
    int vocab_per_rank,
    std::uintptr_t stream_handle);

std::uintptr_t register_mapped_host(std::uintptr_t host_address, int64_t bytes) {
    return register_mapped_host_cuda(host_address, bytes);
}

void unregister_mapped_host(std::uintptr_t host_address) {
    unregister_mapped_host_cuda(host_address);
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
    int64_t threads,
    std::uintptr_t stream_handle) {
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
    TORCH_CHECK(blocks > 0 && blocks <= 64, "blocks must be in [1, 64]");
    TORCH_CHECK(control.numel() >= blocks, "control tensor must hold at least `blocks` counters");
    TORCH_CHECK(world_size >= 2 && world_size <= 8 && rank >= 0 && rank < world_size, "invalid rank/world_size");
    TORCH_CHECK(input.numel() * 2 <= rank_stride_bytes, "input exceeds SHM capacity");

    launch_host_mapped_two_shot_cuda(
        input.data_ptr(),
        output.data_ptr(),
        control.data_ptr<int64_t>(),
        host_device_address,
        data_base_offset,
        slot_stride_bytes,
        rank_stride_bytes,
        input.numel() / 8,
        static_cast<int>(rank),
        static_cast<int>(world_size),
        static_cast<int>(blocks),
        static_cast<int>(threads),
        stream_handle);
}

void launch_shm_v5_allreduce(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor control,
    torch::Tensor rank_shm_ptrs_u64,
    int64_t slot_stride_bytes,
    int64_t rank,
    int64_t world_size,
    int64_t blocks,
    int64_t threads,
    int64_t one_shot_max_vecs,
    std::uintptr_t stream_handle) {
    TORCH_CHECK(input.is_cuda() && output.is_cuda(), "input/output must be CUDA tensors");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16, "BF16 required");
    TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "input/output must be contiguous");
    TORCH_CHECK(input.numel() == output.numel(), "input/output sizes must match");
    TORCH_CHECK(input.numel() % 8 == 0, "input numel must be divisible by 8");
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(input.data_ptr()) % 16 == 0 &&
        reinterpret_cast<std::uintptr_t>(output.data_ptr()) % 16 == 0,
        "input/output pointers must be 16-byte aligned");
    TORCH_CHECK(input.numel() * 2 <= slot_stride_bytes / 2, "input exceeds SHM slot capacity");

    launch_shm_v5_allreduce_cuda(
        input.data_ptr(),
        output.data_ptr(),
        reinterpret_cast<uint64_t*>(control.data_ptr<int64_t>()),
        reinterpret_cast<const std::uintptr_t*>(rank_shm_ptrs_u64.data_ptr<int64_t>()),
        slot_stride_bytes,
        input.numel() / 8,
        static_cast<int>(rank),
        static_cast<int>(world_size),
        static_cast<int>(blocks),
        static_cast<int>(threads),
        static_cast<int>(one_shot_max_vecs),
        stream_handle);
}

void launch_shm_v5_allgather(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor control,
    torch::Tensor rank_shm_ptrs_u64,
    int64_t slot_stride_bytes,
    int64_t outer_rows,
    int64_t vecs_per_row,
    int64_t rank,
    int64_t world_size,
    int64_t blocks,
    int64_t threads,
    std::uintptr_t stream_handle) {
    TORCH_CHECK(input.is_cuda() && output.is_cuda(), "input/output must be CUDA tensors");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16, "BF16 required");
    TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "input/output must be contiguous");
    TORCH_CHECK(outer_rows >= 0 && vecs_per_row > 0, "invalid outer_rows/vecs_per_row");
    TORCH_CHECK(input.numel() == outer_rows * vecs_per_row * 8, "input numel mismatch");
    TORCH_CHECK(output.numel() == outer_rows * world_size * vecs_per_row * 8, "output numel mismatch");
    TORCH_CHECK(input.numel() * 2 <= slot_stride_bytes, "input exceeds AllGather SHM slot capacity");

    launch_shm_v5_allgather_cuda(
        input.data_ptr(),
        output.data_ptr(),
        reinterpret_cast<uint64_t*>(control.data_ptr<int64_t>()),
        reinterpret_cast<const std::uintptr_t*>(rank_shm_ptrs_u64.data_ptr<int64_t>()),
        slot_stride_bytes,
        outer_rows,
        vecs_per_row,
        static_cast<int>(rank),
        static_cast<int>(world_size),
        static_cast<int>(blocks),
        static_cast<int>(threads),
        stream_handle);
}

void launch_local_top2_bf16(
    torch::Tensor logits_bf16,
    torch::Tensor out_top2_vals,
    torch::Tensor out_top2_ids,
    int64_t num_rows,
    std::uintptr_t stream_handle) {
    TORCH_CHECK(logits_bf16.scalar_type() == at::kBFloat16, "logits_bf16 must be bfloat16");
    TORCH_CHECK(logits_bf16.dim() == 2, "logits_bf16 must be 2D");
    TORCH_CHECK(logits_bf16.stride(1) == 1, "logits_bf16 inner dimension must be contiguous");
    const int64_t vocab_per_rank = logits_bf16.size(1);
    TORCH_CHECK(vocab_per_rank >= 2, "vocab_per_rank must be at least 2");
    TORCH_CHECK(num_rows >= 0 && num_rows <= logits_bf16.size(0), "num_rows out of bounds");
    TORCH_CHECK(out_top2_vals.is_contiguous() && out_top2_vals.numel() >= num_rows * 2,
                "out_top2_vals must be contiguous with at least num_rows * 2 floats");
    TORCH_CHECK(out_top2_ids.is_contiguous() && out_top2_ids.numel() >= num_rows * 2,
                "out_top2_ids must be contiguous with at least num_rows * 2 int32s");
    if (num_rows == 0) {
        return;
    }
    launch_local_top2_bf16_cuda(
        logits_bf16.data_ptr(),
        out_top2_vals.data_ptr<float>(),
        out_top2_ids.data_ptr<int32_t>(),
        num_rows,
        logits_bf16.stride(0),
        static_cast<int>(vocab_per_rank),
        stream_handle);
}

void launch_tp8_argmax_gather_shm(
    torch::Tensor local_top2_vals,
    torch::Tensor local_top2_ids,
    torch::Tensor out_top2_tokens,
    torch::Tensor out_top2_logit_bits,
    torch::Tensor step_counter,
    std::uintptr_t host_device_address,
    int64_t num_rows,
    int64_t max_rows_per_step,
    int64_t rank,
    int64_t world_size,
    int64_t vocab_per_rank,
    std::uintptr_t stream_handle) {
    TORCH_CHECK(num_rows <= max_rows_per_step, "num_rows exceeds max_rows_per_step");
    TORCH_CHECK(local_top2_vals.is_contiguous() && local_top2_vals.numel() >= num_rows * 2,
                "local_top2_vals must hold at least num_rows * 2 floats");
    TORCH_CHECK(local_top2_ids.is_contiguous() && local_top2_ids.numel() >= num_rows * 2,
                "local_top2_ids must hold at least num_rows * 2 int32s");
    TORCH_CHECK(out_top2_tokens.is_contiguous() && out_top2_tokens.numel() >= num_rows * 2,
                "out_top2_tokens must hold at least num_rows * 2 int32s");
    TORCH_CHECK(out_top2_logit_bits.is_contiguous() && out_top2_logit_bits.numel() >= num_rows * 2,
                "out_top2_logit_bits must hold at least num_rows * 2 int32s");
    launch_tp8_argmax_gather_shm_cuda(
        local_top2_vals.data_ptr<float>(),
        local_top2_ids.data_ptr<int32_t>(),
        out_top2_tokens.data_ptr<int32_t>(),
        reinterpret_cast<uint32_t*>(out_top2_logit_bits.data_ptr<int32_t>()),
        reinterpret_cast<uint64_t*>(step_counter.data_ptr<int64_t>()),
        host_device_address,
        num_rows,
        max_rows_per_step,
        static_cast<int>(rank),
        static_cast<int>(world_size),
        static_cast<int>(vocab_per_rank),
        stream_handle);
}
"""

_CUDA_SOURCE = r"""
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

#define ZSTEP_CUDA_CHECK(expr)                                                      \
    do {                                                                            \
        cudaError_t _err = (expr);                                                  \
        if (_err != cudaSuccess) {                                                  \
            throw std::runtime_error(                                               \
                std::string("CUDA error: ") + cudaGetErrorString(_err));            \
        }                                                                           \
    } while (0)

namespace {

constexpr int kNumRingSlots = 4;
constexpr int kMaxRanks = 8;
constexpr int kOneShotMaxVecsLegacy = 1024;
constexpr int64_t kShmV5AbortOffset = 98304;
constexpr int64_t kArgmaxAbortOffset = 128;
// ~90 seconds at 2.5 GHz boost clock; checked only every 4096 spin iterations.
constexpr uint64_t kSpinTimeoutCycles = 225000000000ULL;

__device__ __forceinline__ uint64_t sys_load_acquire_u64(const uint64_t* addr) {
    uint64_t val;
    asm volatile("ld.global.acquire.sys.b64 %0, [%1];" : "=l"(val) : "l"(addr) : "memory");
    return val;
}

__device__ __forceinline__ void sys_store_release_u64(uint64_t* addr, uint64_t val) {
    asm volatile("st.global.release.sys.b64 [%0], %1;" :: "l"(addr), "l"(val) : "memory");
}

__device__ __forceinline__ bool spin_wait_ge_u64(
    const uint64_t* flag_addr,
    uint64_t target,
    const uint64_t* abort_addr) {
    uint32_t iters = 0;
    uint64_t start_cycle = 0;
    while (sys_load_acquire_u64(flag_addr) < target) {
        __nanosleep(24);
        if ((++iters & 0xFFFU) == 0U) {
            if (sys_load_acquire_u64(abort_addr) != 0ULL) {
                return false;
            }
            const uint64_t now = clock64();
            if (start_cycle == 0ULL) {
                start_cycle = now;
            } else if (now - start_cycle > kSpinTimeoutCycles) {
                return false;
            }
        }
    }
    return true;
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

// Single-file fallback kernel used when `exchange_group is None` (standalone tests).
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

    const int64_t chunk_begin = (num_vecs * static_cast<int64_t>(blockIdx.x)) / gridDim.x;
    const int64_t chunk_end = (num_vecs * static_cast<int64_t>(blockIdx.x + 1)) / gridDim.x;
    const int64_t chunk_len = chunk_end - chunk_begin;

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
            __nanosleep(32);
        }
    }
    __syncthreads();

    if (num_vecs <= kOneShotMaxVecsLegacy) {
        for (int64_t idx = chunk_begin + threadIdx.x; idx < chunk_end; idx += blockDim.x) {
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
            output_vec[idx] = out_pack.u4;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            sys_store_release_u64(done_flags + rank, step);
        }
        return;
    }

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

// Issue every peer's 16-byte load before consuming any of them: the loads are volatile asm, so
// interleaving them with the accumulation would serialize one PCIe round trip per peer.
__device__ __forceinline__ void load_peer_vecs(
    uint4 (&vals)[kMaxRanks],
    const uint4* __restrict__ input_vec,
    char* const* peer_shm,
    int slot,
    int64_t slot_stride_bytes,
    int64_t idx,
    int rank,
    int world_size) {
    #pragma unroll
    for (int p = 0; p < kMaxRanks; ++p) {
        if (p < world_size) {
            const uint4* p_in = reinterpret_cast<const uint4*>(
                peer_shm[p] + 131072 + slot * slot_stride_bytes);
            vals[p] = (p == rank) ? input_vec[idx] : ld_cv_uint4(p_in + idx);
        }
    }
}

__device__ __forceinline__ void accumulate_peer_vecs(
    float (&acc)[8], const uint4 (&vals)[kMaxRanks], int world_size) {
    #pragma unroll
    for (int p = 0; p < kMaxRanks; ++p) {
        if (p < world_size) {
            Pack128Bf16 v;
            v.u4 = vals[p];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __bfloat1622float2(v.pairs[j]);
                acc[2 * j] += f.x;
                acc[2 * j + 1] += f.y;
            }
        }
    }
}

// Per-rank NUMA-local SHM region layout (`shm_v5`):
// [0 .. 32768):     in_flag[block][peer]   (64B per (block, peer) -> zero false sharing & local-NUMA spin)
// [32768 .. 65536): red_flag[block][peer]  (64B per (block, peer))
// [65536 .. 98304): done_flag[block][peer] (64B per (block, peer))
// [98304 .. 98368): abort_flag             (64B cacheline-isolated cluster abort word)
// [131072 .. ):     4 ring slots * (in_data + red_data)
__global__ void shm_v5_allreduce_kernel(
    const uint4* __restrict__ input_vec,
    uint4* __restrict__ output_vec,
    uint64_t* __restrict__ control,
    char* const* __restrict__ rank_shm,
    int64_t slot_stride_bytes,
    int64_t num_vecs,
    int rank,
    int world_size,
    int one_shot_max_vecs) {
    const int b = blockIdx.x;
    __shared__ uint64_t s_step;
    __shared__ int s_aborted;
    __shared__ char* s_peer_shm[kMaxRanks];
    if (threadIdx.x < world_size) {
        s_peer_shm[threadIdx.x] = rank_shm[threadIdx.x];
    }
    if (threadIdx.x == 0) {
        const uint64_t ns = control[b] + 1ULL;
        control[b] = ns;
        s_step = ns;
        s_aborted = 0;
    }
    __syncthreads();
    const uint64_t step = s_step;
    const int slot = static_cast<int>((step - 1ULL) & 3);
    char* my_shm = s_peer_shm[rank];
    const uint64_t* abort_flag = reinterpret_cast<const uint64_t*>(my_shm + kShmV5AbortOffset);

    const int64_t c_begin = (num_vecs * b) / gridDim.x;
    const int64_t c_end = (num_vecs * (b + 1)) / gridDim.x;
    const int64_t c_len = c_end - c_begin;

    if (step > 4ULL) {
        const uint64_t min_done = step - 4ULL;
        if (threadIdx.x < world_size) {
            const uint64_t* my_done = reinterpret_cast<const uint64_t*>(
                my_shm + 65536 + (b * 8 + threadIdx.x) * 64);
            if (!spin_wait_ge_u64(my_done, min_done, abort_flag)) {
                s_aborted = 1;
            }
        }
        __syncthreads();
        if (s_aborted) {
            return;
        }
    }

    const int64_t half_slot = slot_stride_bytes / 2;
    uint4* my_in = reinterpret_cast<uint4*>(my_shm + 131072 + slot * slot_stride_bytes);
    for (int64_t idx = c_begin + threadIdx.x; idx < c_end; idx += blockDim.x) {
        st_wt_uint4(my_in + idx, input_vec[idx]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
    }
    __syncthreads();
    if (threadIdx.x < world_size) {
        uint64_t* peer_in_flag = reinterpret_cast<uint64_t*>(
            s_peer_shm[threadIdx.x] + (b * 8 + rank) * 64);
        sys_store_release_u64(peer_in_flag, step);
        const uint64_t* local_in_flag = reinterpret_cast<const uint64_t*>(
            my_shm + (b * 8 + threadIdx.x) * 64);
        if (!spin_wait_ge_u64(local_in_flag, step, abort_flag)) {
            s_aborted = 1;
        }
    }
    __syncthreads();
    if (s_aborted) {
        return;
    }

    if (num_vecs <= one_shot_max_vecs) {
        for (int64_t idx = c_begin + threadIdx.x; idx < c_end; idx += blockDim.x) {
            uint4 peer_vals[kMaxRanks];
            load_peer_vecs(peer_vals, input_vec, s_peer_shm, slot, slot_stride_bytes, idx, rank, world_size);
            float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            accumulate_peer_vecs(acc, peer_vals, world_size);
            Pack128Bf16 out;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                out.pairs[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
            }
            output_vec[idx] = out.u4;
        }
        __syncthreads();
        if (threadIdx.x < world_size) {
            uint64_t* peer_done = reinterpret_cast<uint64_t*>(
                s_peer_shm[threadIdx.x] + 65536 + (b * 8 + rank) * 64);
            sys_store_release_u64(peer_done, step);
        }
        return;
    }

    const int64_t s_begin = c_begin + (c_len * rank) / world_size;
    const int64_t s_end = c_begin + (c_len * (rank + 1)) / world_size;
    uint4* my_red = reinterpret_cast<uint4*>(my_shm + 131072 + slot * slot_stride_bytes + half_slot);
    for (int64_t idx = s_begin + threadIdx.x; idx < s_end; idx += blockDim.x) {
        uint4 peer_vals[kMaxRanks];
        load_peer_vecs(peer_vals, input_vec, s_peer_shm, slot, slot_stride_bytes, idx, rank, world_size);
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        accumulate_peer_vecs(acc, peer_vals, world_size);
        Pack128Bf16 out;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            out.pairs[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
        }
        st_wt_uint4(my_red + idx, out.u4);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
    }
    __syncthreads();
    if (threadIdx.x < world_size) {
        uint64_t* peer_red_flag = reinterpret_cast<uint64_t*>(
            s_peer_shm[threadIdx.x] + 32768 + (b * 8 + rank) * 64);
        sys_store_release_u64(peer_red_flag, step);
        const uint64_t* local_red_flag = reinterpret_cast<const uint64_t*>(
            my_shm + 32768 + (b * 8 + threadIdx.x) * 64);
        if (!spin_wait_ge_u64(local_red_flag, step, abort_flag)) {
            s_aborted = 1;
        }
    }
    __syncthreads();
    if (s_aborted) {
        return;
    }
    // Walk all owners' reduced shards in lockstep so each thread keeps one remote load per
    // owner in flight instead of paying the PCIe round trip once per owner.
    const int64_t max_sub_len = (c_len + world_size - 1) / world_size;
    for (int64_t j = threadIdx.x; j < max_sub_len; j += blockDim.x) {
        uint4 vals[kMaxRanks];
        #pragma unroll
        for (int owner = 0; owner < kMaxRanks; ++owner) {
            if (owner < world_size) {
                const int64_t o_begin = c_begin + (c_len * owner) / world_size;
                const int64_t o_end = c_begin + (c_len * (owner + 1)) / world_size;
                if (o_begin + j < o_end) {
                    const uint4* o_red = reinterpret_cast<const uint4*>(
                        s_peer_shm[owner] + 131072 + slot * slot_stride_bytes + half_slot);
                    vals[owner] = ld_cv_uint4(o_red + o_begin + j);
                }
            }
        }
        #pragma unroll
        for (int owner = 0; owner < kMaxRanks; ++owner) {
            if (owner < world_size) {
                const int64_t o_begin = c_begin + (c_len * owner) / world_size;
                const int64_t o_end = c_begin + (c_len * (owner + 1)) / world_size;
                if (o_begin + j < o_end) {
                    output_vec[o_begin + j] = vals[owner];
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < world_size) {
        uint64_t* peer_done = reinterpret_cast<uint64_t*>(
            s_peer_shm[threadIdx.x] + 65536 + (b * 8 + rank) * 64);
        sys_store_release_u64(peer_done, step);
    }
}

// 1-shot NUMA-local SHM AllGather for `[outer_rows, vecs_per_row * 8]` ->
// `[outer_rows, world_size * vecs_per_row * 8]` BF16 tensors (handles dim=0 and dim>=1).
__global__ void shm_v5_allgather_kernel(
    const uint4* __restrict__ input_vec,
    uint4* __restrict__ output_vec,
    uint64_t* __restrict__ control,
    char* const* __restrict__ rank_shm,
    int64_t slot_stride_bytes,
    int64_t outer_rows,
    int64_t vecs_per_row,
    int rank,
    int world_size) {
    const int b = blockIdx.x;
    __shared__ uint64_t s_step;
    __shared__ int s_aborted;
    __shared__ char* s_peer_shm[kMaxRanks];
    if (threadIdx.x < world_size) {
        s_peer_shm[threadIdx.x] = rank_shm[threadIdx.x];
    }
    if (threadIdx.x == 0) {
        const uint64_t ns = control[b] + 1ULL;
        control[b] = ns;
        s_step = ns;
        s_aborted = 0;
    }
    __syncthreads();
    const uint64_t step = s_step;
    const int slot = static_cast<int>((step - 1ULL) & 3);
    char* my_shm = s_peer_shm[rank];
    const uint64_t* abort_flag = reinterpret_cast<const uint64_t*>(my_shm + kShmV5AbortOffset);

    const int64_t num_in_vecs = outer_rows * vecs_per_row;
    const int64_t c_begin = (num_in_vecs * b) / gridDim.x;
    const int64_t c_end = (num_in_vecs * (b + 1)) / gridDim.x;

    if (step > 4ULL) {
        const uint64_t min_done = step - 4ULL;
        if (threadIdx.x < world_size) {
            const uint64_t* my_done = reinterpret_cast<const uint64_t*>(
                my_shm + 65536 + (b * 8 + threadIdx.x) * 64);
            if (!spin_wait_ge_u64(my_done, min_done, abort_flag)) {
                s_aborted = 1;
            }
        }
        __syncthreads();
        if (s_aborted) {
            return;
        }
    }

    uint4* my_in = reinterpret_cast<uint4*>(my_shm + 131072 + slot * slot_stride_bytes);
    for (int64_t idx = c_begin + threadIdx.x; idx < c_end; idx += blockDim.x) {
        st_wt_uint4(my_in + idx, input_vec[idx]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
    }
    __syncthreads();
    if (threadIdx.x < world_size) {
        uint64_t* peer_in_flag = reinterpret_cast<uint64_t*>(
            s_peer_shm[threadIdx.x] + (b * 8 + rank) * 64);
        sys_store_release_u64(peer_in_flag, step);
        const uint64_t* local_in_flag = reinterpret_cast<const uint64_t*>(
            my_shm + (b * 8 + threadIdx.x) * 64);
        if (!spin_wait_ge_u64(local_in_flag, step, abort_flag)) {
            s_aborted = 1;
        }
    }
    __syncthreads();
    if (s_aborted) {
        return;
    }

    if (outer_rows == 1) {
        for (int p = 0; p < world_size; ++p) {
            const uint4* p_in = reinterpret_cast<const uint4*>(
                s_peer_shm[p] + 131072 + slot * slot_stride_bytes);
            const int64_t p_base = static_cast<int64_t>(p) * vecs_per_row;
            for (int64_t idx = c_begin + threadIdx.x; idx < c_end; idx += blockDim.x) {
                output_vec[p_base + idx] = (p == rank) ? input_vec[idx] : ld_cv_uint4(p_in + idx);
            }
        }
    } else {
        for (int p = 0; p < world_size; ++p) {
            const uint4* p_in = reinterpret_cast<const uint4*>(
                s_peer_shm[p] + 131072 + slot * slot_stride_bytes);
            for (int64_t idx = c_begin + threadIdx.x; idx < c_end; idx += blockDim.x) {
                const int64_t row = idx / vecs_per_row;
                const int64_t col = idx - row * vecs_per_row;
                const int64_t out_idx = (row * world_size + p) * vecs_per_row + col;
                output_vec[out_idx] = (p == rank) ? input_vec[idx] : ld_cv_uint4(p_in + idx);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < world_size) {
        uint64_t* peer_done = reinterpret_cast<uint64_t*>(
            s_peer_shm[threadIdx.x] + 65536 + (b * 8 + rank) * 64);
        sys_store_release_u64(peer_done, step);
    }
}

}  // namespace

std::uintptr_t register_mapped_host_cuda(std::uintptr_t host_address, int64_t bytes) {
    void* host = reinterpret_cast<void*>(host_address);
    ZSTEP_CUDA_CHECK(cudaHostRegister(
        host, static_cast<size_t>(bytes), cudaHostRegisterMapped));
    void* device = nullptr;
    cudaError_t status = cudaHostGetDevicePointer(&device, host, 0);
    if (status != cudaSuccess) {
        cudaHostUnregister(host);
        ZSTEP_CUDA_CHECK(status);
    }
    return reinterpret_cast<std::uintptr_t>(device);
}

void unregister_mapped_host_cuda(std::uintptr_t host_address) {
    ZSTEP_CUDA_CHECK(cudaHostUnregister(reinterpret_cast<void*>(host_address)));
}

void launch_host_mapped_two_shot_cuda(
    const void* input_ptr,
    void* output_ptr,
    int64_t* control_ptr,
    std::uintptr_t host_device_address,
    int64_t data_base_offset,
    int64_t slot_stride_bytes,
    int64_t rank_stride_bytes,
    int64_t num_vecs,
    int rank,
    int world_size,
    int blocks,
    int threads,
    std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    host_mapped_two_shot_fused_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint4*>(input_ptr),
        reinterpret_cast<uint4*>(output_ptr),
        reinterpret_cast<uint64_t*>(control_ptr),
        reinterpret_cast<char*>(host_device_address),
        data_base_offset,
        slot_stride_bytes,
        rank_stride_bytes,
        num_vecs,
        rank,
        world_size);
    ZSTEP_CUDA_CHECK(cudaGetLastError());
}

void launch_shm_v5_allreduce_cuda(
    const void* input_ptr,
    void* output_ptr,
    uint64_t* control_ptr,
    const std::uintptr_t* rank_shm_dev_ptrs,
    int64_t slot_stride_bytes,
    int64_t num_vecs,
    int rank,
    int world_size,
    int blocks,
    int threads,
    int one_shot_max_vecs,
    std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const int64_t vecs_per_cta = (num_vecs + blocks - 1) / blocks;
    const int launch_threads = (threads == 256 && vecs_per_cta > 256) ? 512 : threads;
    shm_v5_allreduce_kernel<<<blocks, launch_threads, 0, stream>>>(
        reinterpret_cast<const uint4*>(input_ptr),
        reinterpret_cast<uint4*>(output_ptr),
        control_ptr,
        reinterpret_cast<char* const*>(rank_shm_dev_ptrs),
        slot_stride_bytes,
        num_vecs,
        rank,
        world_size,
        one_shot_max_vecs);
    ZSTEP_CUDA_CHECK(cudaGetLastError());
}

void launch_shm_v5_allgather_cuda(
    const void* input_ptr,
    void* output_ptr,
    uint64_t* control_ptr,
    const std::uintptr_t* rank_shm_dev_ptrs,
    int64_t slot_stride_bytes,
    int64_t outer_rows,
    int64_t vecs_per_row,
    int rank,
    int world_size,
    int blocks,
    int threads,
    std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const int64_t num_in_vecs = outer_rows * vecs_per_row;
    const int64_t vecs_per_cta = (num_in_vecs + blocks - 1) / blocks;
    const int launch_threads = (threads == 256 && vecs_per_cta > 256) ? 512 : threads;
    shm_v5_allgather_kernel<<<blocks, launch_threads, 0, stream>>>(
        reinterpret_cast<const uint4*>(input_ptr),
        reinterpret_cast<uint4*>(output_ptr),
        control_ptr,
        reinterpret_cast<char* const*>(rank_shm_dev_ptrs),
        slot_stride_bytes,
        outer_rows,
        vecs_per_row,
        rank,
        world_size);
    ZSTEP_CUDA_CHECK(cudaGetLastError());
}

namespace {

struct alignas(8) ValIdxPair {
    float val;
    int32_t tok;
};

struct alignas(16) Top2Acc {
    float v1;
    int32_t i1;
    float v2;
    int32_t i2;
};

__device__ __forceinline__ void merge_candidate(Top2Acc& acc, float v, int32_t idx) {
    if (idx < 0) {
        return;
    }
    if (acc.i1 < 0 || v > acc.v1 || (v == acc.v1 && idx < acc.i1)) {
        acc.v2 = acc.v1;
        acc.i2 = acc.i1;
        acc.v1 = v;
        acc.i1 = idx;
    } else if (acc.i2 < 0 || v > acc.v2 || (v == acc.v2 && idx < acc.i2)) {
        acc.v2 = v;
        acc.i2 = idx;
    }
}

__device__ __forceinline__ void merge_top2(Top2Acc& acc, const Top2Acc& other) {
    merge_candidate(acc, other.v1, other.i1);
    merge_candidate(acc, other.v2, other.i2);
}

// Deterministic per-row BF16 Top-2 reduction with strict `(val DESC, idx ASC)` tie-breaking.
__global__ void local_top2_bf16_kernel(
    const __nv_bfloat16* __restrict__ logits,
    float* __restrict__ out_top2_vals,
    int32_t* __restrict__ out_top2_ids,
    int64_t stride_row,
    int vocab_per_rank) {
    __shared__ Top2Acc s_acc[256];
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int tid = threadIdx.x;
    const __nv_bfloat16* row_ptr = logits + row * stride_row;

    Top2Acc acc{-1.0e38f, -1, -1.0e38f, -1};
    for (int col = tid; col < vocab_per_rank; col += 256) {
        const float v = __bfloat162float(row_ptr[col]);
        merge_candidate(acc, v, col);
    }
    s_acc[tid] = acc;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            merge_top2(s_acc[tid], s_acc[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_top2_vals[row * 2 + 0] = s_acc[0].v1;
        out_top2_vals[row * 2 + 1] = s_acc[0].v2;
        out_top2_ids[row * 2 + 0] = s_acc[0].i1;
        out_top2_ids[row * 2 + 1] = s_acc[0].i2;
    }
}

__global__ void tp8_argmax_gather_shm_kernel(
    const float* __restrict__ local_top2_vals,
    const int32_t* __restrict__ local_top2_ids,
    int32_t* __restrict__ out_top2_tokens,
    uint32_t* __restrict__ out_top2_logit_bits,
    uint64_t* __restrict__ step_counter,
    char* __restrict__ host_base,
    int64_t num_rows,
    int64_t max_rows_per_step,
    int rank,
    int world_size,
    int vocab_per_rank) {
    uint64_t* ready_flags = reinterpret_cast<uint64_t*>(host_base);
    uint64_t* done_flags = reinterpret_cast<uint64_t*>(host_base + 64);
    const uint64_t* abort_flag =
        reinterpret_cast<const uint64_t*>(host_base + kArgmaxAbortOffset);

    __shared__ uint64_t s_step;
    __shared__ int s_aborted;
    if (threadIdx.x == 0) {
        const uint64_t next = step_counter[0] + 1ULL;
        step_counter[0] = next;
        s_step = next;
        s_aborted = 0;
    }
    __syncthreads();
    const uint64_t step = s_step;
    const int slot = static_cast<int>((step - 1ULL) & (kNumRingSlots - 1));

    if (step > static_cast<uint64_t>(kNumRingSlots)) {
        const uint64_t min_done = step - static_cast<uint64_t>(kNumRingSlots);
        if (threadIdx.x < world_size) {
            if (!spin_wait_ge_u64(done_flags + threadIdx.x, min_done, abort_flag)) {
                s_aborted = 1;
            }
        }
        __syncthreads();
        if (s_aborted) {
            return;
        }
    }

    const int64_t pairs_per_rank = max_rows_per_step * 2;
    ValIdxPair* slot_pairs = reinterpret_cast<ValIdxPair*>(host_base + 256)
        + static_cast<int64_t>(slot) * (kMaxRanks * pairs_per_rank);
    ValIdxPair* my_pairs = slot_pairs + static_cast<int64_t>(rank) * pairs_per_rank;

    const int base_tok = rank * vocab_per_rank;
    for (int64_t r = threadIdx.x; r < num_rows; r += blockDim.x) {
        ValIdxPair p0;
        p0.val = local_top2_vals[r * 2 + 0];
        p0.tok = base_tok + local_top2_ids[r * 2 + 0];
        ValIdxPair p1;
        p1.val = local_top2_vals[r * 2 + 1];
        p1.tok = base_tok + local_top2_ids[r * 2 + 1];
        my_pairs[r * 2 + 0] = p0;
        my_pairs[r * 2 + 1] = p1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        sys_store_release_u64(ready_flags + rank, step);
    }
    if (threadIdx.x < world_size) {
        if (!spin_wait_ge_u64(ready_flags + threadIdx.x, step, abort_flag)) {
            s_aborted = 1;
        }
    }
    __syncthreads();
    if (s_aborted) {
        return;
    }

    if (rank == 0) {
        for (int64_t r = threadIdx.x; r < num_rows; r += blockDim.x) {
            Top2Acc acc{-1.0e38f, -1, -1.0e38f, -1};
            for (int peer = 0; peer < world_size; ++peer) {
                const int64_t peer_row_base =
                    static_cast<int64_t>(peer) * pairs_per_rank + r * 2;
                const ValIdxPair p0 = slot_pairs[peer_row_base + 0];
                const ValIdxPair p1 = slot_pairs[peer_row_base + 1];
                merge_candidate(acc, p0.val, p0.tok);
                merge_candidate(acc, p1.val, p1.tok);
            }
            out_top2_tokens[r * 2 + 0] = acc.i1;
            out_top2_tokens[r * 2 + 1] = acc.i2;
            out_top2_logit_bits[r * 2 + 0] = __float_as_uint(acc.v1);
            out_top2_logit_bits[r * 2 + 1] = __float_as_uint(acc.v2);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        sys_store_release_u64(done_flags + rank, step);
    }
}

}  // namespace

void launch_local_top2_bf16_cuda(
    const void* logits_bf16,
    float* out_top2_vals,
    int32_t* out_top2_ids,
    int64_t num_rows,
    int64_t stride_row,
    int vocab_per_rank,
    std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    local_top2_bf16_kernel<<<static_cast<unsigned int>(num_rows), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits_bf16),
        out_top2_vals,
        out_top2_ids,
        stride_row,
        vocab_per_rank);
    ZSTEP_CUDA_CHECK(cudaGetLastError());
}

void launch_tp8_argmax_gather_shm_cuda(
    const float* local_top2_vals,
    const int32_t* local_top2_ids,
    int32_t* out_top2_tokens,
    uint32_t* out_top2_logit_bits,
    uint64_t* step_counter,
    std::uintptr_t host_device_address,
    int64_t num_rows,
    int64_t max_rows_per_step,
    int rank,
    int world_size,
    int vocab_per_rank,
    std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    tp8_argmax_gather_shm_kernel<<<1, 128, 0, stream>>>(
        local_top2_vals,
        local_top2_ids,
        out_top2_tokens,
        out_top2_logit_bits,
        step_counter,
        reinterpret_cast<char*>(host_device_address),
        num_rows,
        max_rows_per_step,
        rank,
        world_size,
        vocab_per_rank);
    ZSTEP_CUDA_CHECK(cudaGetLastError());
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
            name="zstep_shm_allreduce_sm120_v9_ext",
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            functions=[
                "register_mapped_host",
                "unregister_mapped_host",
                "launch_host_mapped_two_shot",
                "launch_shm_v5_allreduce",
                "launch_shm_v5_allgather",
                "launch_local_top2_bf16",
                "launch_tp8_argmax_gather_shm",
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


class _PerRankShmRingBase:
    """Shared lifecycle helper for per-rank NUMA-local `/dev/shm/{name}_r{r}` rings."""

    def _init_per_rank_shm(
        self,
        *,
        name: str,
        rank: int,
        world_size: int,
        total_bytes: int,
        exchange_group: dist.ProcessGroup,
        device: torch.device,
        extension,
    ) -> None:
        self._per_rank_paths: list[str] = [
            f"/dev/shm/{name}_r{peer}" for peer in range(world_size)
        ]
        self._per_rank_mappings: list[mmap.mmap] = []
        self._per_rank_host_addrs: list[int] = []
        self._own_shm_path: str = self._per_rank_paths[rank]

        own_fd = os.open(
            self._own_shm_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600
        )
        try:
            os.ftruncate(own_fd, total_bytes)
            own_mm = mmap.mmap(
                own_fd,
                total_bytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        finally:
            os.close(own_fd)
        # First-touch every 4 KiB page from this rank's NUMA-pinned thread.
        mv = memoryview(own_mm)
        for off in range(0, total_bytes, 4096):
            mv[off] = 0
        mv.release()
        ctypes.memset(
            ctypes.addressof(ctypes.c_char.from_buffer(own_mm)),
            0,
            min(SHM_V5_HEADER_BYTES, total_bytes),
        )
        dist.barrier(group=exchange_group)

        dev_ptrs: list[int] = []
        for peer in range(world_size):
            if peer == rank:
                mm = own_mm
            else:
                pfd = os.open(self._per_rank_paths[peer], os.O_RDWR)
                try:
                    mm = mmap.mmap(
                        pfd,
                        total_bytes,
                        flags=mmap.MAP_SHARED,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE,
                    )
                finally:
                    os.close(pfd)
            host_addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            dev_ptr = extension.register_mapped_host(host_addr, total_bytes)
            self._per_rank_mappings.append(mm)
            self._per_rank_host_addrs.append(host_addr)
            dev_ptrs.append(dev_ptr)

        dist.barrier(group=exchange_group)
        # All peers have opened and mmapped every `/dev/shm/<name>_r<peer>`; unlink our
        # directory entry immediately so kernel tmpfs reclaims pages even on SIGKILL.
        try:
            os.unlink(self._own_shm_path)
        except FileNotFoundError:
            pass
        dist.barrier(group=exchange_group)
        self._rank_shm_ptrs: torch.Tensor = torch.tensor(
            dev_ptrs, dtype=torch.int64, device=device
        )

    def signal_abort(self) -> None:
        """Set the cacheline-isolated abort flag across all mapped per-rank SHM segments."""
        for mm in getattr(self, "_per_rank_mappings", ()):
            if not mm.closed and len(mm) >= SHM_V5_ABORT_OFFSET + 8:
                struct.pack_into("<Q", mm, SHM_V5_ABORT_OFFSET, 1)

    def _close_per_rank_shm(self, extension) -> None:
        for host_addr in self._per_rank_host_addrs:
            extension.unregister_mapped_host(host_addr)
        self._per_rank_host_addrs.clear()
        for mm in self._per_rank_mappings:
            mm.close()
        self._per_rank_mappings.clear()
        try:
            os.unlink(self._own_shm_path)
        except FileNotFoundError:
            pass


class HostMappedShmAllReduce(_PerRankShmRingBase):
    """Ordered host-mapped 4-slot ring BF16 AllReduce (`shm_v5` when `exchange_group` is set)."""

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
        one_shot_max_vecs: int = DEFAULT_ONE_SHOT_MAX_VECS,
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
        self.numa_topology: DeviceNumaTopology = pin_process_to_device_numa(device.index)
        self.name: str = name
        self.rank: int = rank
        self.world_size: int = world_size
        self.capacity_elements: int = capacity_elements
        self.blocks: int = blocks
        self.threads: int = threads
        self.one_shot_max_vecs: int = one_shot_max_vecs

        self.device: torch.device = device
        self._use_shm_v5: bool = exchange_group is not None
        self._closed: bool = False
        self._registered: bool = False
        self._owns_unlink: bool = False
        self._extension = get_host_mapped_allreduce_extension()

        if self._use_shm_v5:
            assert exchange_group is not None
            self.rank_stride_bytes = _align_up(capacity_elements * 2, 4096)
            self.data_base_offset = SHM_V5_HEADER_BYTES
            self.slot_stride_bytes = 2 * self.rank_stride_bytes
            self.total_bytes = _align_up(
                self.data_base_offset + NUM_RING_SLOTS * self.slot_stride_bytes, 4096
            )
            self._init_per_rank_shm(
                name=name,
                rank=rank,
                world_size=world_size,
                total_bytes=self.total_bytes,
                exchange_group=exchange_group,
                device=device,
                extension=self._extension,
            )
            self._registered = True
        else:
            self.rank_stride_bytes = _align_up(capacity_elements * 2, 4096)
            self.data_base_offset = 16384
            self.slot_stride_bytes = _align_up(
                2 * world_size * self.rank_stride_bytes, 4096
            )
            self.total_bytes = _align_up(
                self.data_base_offset + NUM_RING_SLOTS * self.slot_stride_bytes, 4096
            )
            shm_path = f"/dev/shm/{name}"
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
            self._shm_path = shm_path
            self._host_address = ctypes.addressof(ctypes.c_char.from_buffer(self._mapping))
            self._first_touch_legacy_pages()
            self._device_address = self._extension.register_mapped_host(
                self._host_address, self.total_bytes
            )
            self._registered = True

        self.control: torch.Tensor = torch.zeros(MAX_BLOCKS, dtype=torch.int64, device=device)

    def _first_touch_legacy_pages(self) -> None:
        mv = memoryview(self._mapping)
        for slot in range(NUM_RING_SLOTS):
            slot_base = self.data_base_offset + slot * self.slot_stride_bytes
            in_base = slot_base + self.rank * self.rank_stride_bytes
            red_base = slot_base + (self.world_size + self.rank) * self.rank_stride_bytes
            for page_off in range(0, self.rank_stride_bytes, 4096):
                mv[in_base + page_off] = 0
                mv[red_base + page_off] = 0
        mv.release()

    def all_reduce(self, input: torch.Tensor, *, output: torch.Tensor) -> torch.Tensor:
        """Reduce all rank inputs into `output` in a single CUDA-Graph-safe launch."""
        if self._closed:
            raise RuntimeError("AllReduce channel is closed")
        if input.shape != output.shape:
            raise ValueError("input/output shapes must match")
        if input.numel() > self.capacity_elements:
            raise ValueError("input exceeds channel capacity")
        if input.numel() == 0:
            return output
        stream_handle = torch.cuda.current_stream(self.device).cuda_stream
        if self._use_shm_v5:
            # Keep `gridDim.x == self.blocks` invariant across every payload size so all
            # per-block ring step counters `control[0..blocks)` stay strictly in lockstep.
            self._extension.launch_shm_v5_allreduce(
                input,
                output,
                self.control,
                self._rank_shm_ptrs,
                self.slot_stride_bytes,
                self.rank,
                self.world_size,
                self.blocks,
                self.threads,
                self.one_shot_max_vecs,
                stream_handle,
            )
        else:
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
                stream_handle,
            )
        return output

    def close(self, *, abort: bool = False) -> None:
        """Release CUDA host mapping and POSIX shared memory handle(s)."""
        if self._closed:
            return
        if abort:
            self.signal_abort()
        torch.cuda.synchronize(self.device)
        if self._use_shm_v5:
            if self._registered:
                self._close_per_rank_shm(self._extension)
                self._registered = False
        else:
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


class HostMappedShmAllGather(_PerRankShmRingBase):
    """CUDA-Graph-safe 1-shot NUMA-local SHM BF16 AllGather across `dim`."""

    def __init__(
        self,
        *,
        name: str,
        rank: int,
        world_size: int,
        capacity_input_elements: int,
        blocks: int,
        threads: int,
        exchange_group: dist.ProcessGroup,
    ) -> None:
        if world_size < 2 or world_size > MAX_RANKS or not 0 <= rank < world_size:
            raise ValueError(f"rank={rank} / world_size={world_size} invalid (max {MAX_RANKS})")
        if capacity_input_elements <= 0 or capacity_input_elements % 8 != 0:
            raise ValueError("capacity_input_elements must be positive and divisible by 8")
        if not 1 <= blocks <= MAX_BLOCKS:
            raise ValueError(f"blocks must be in [1, {MAX_BLOCKS}]")
        if threads <= 0 or threads > 1024 or threads % 32 != 0:
            raise ValueError("threads must be a warp multiple in [32, 1024]")

        device = torch.device("cuda", torch.cuda.current_device())
        self.numa_topology: DeviceNumaTopology = pin_process_to_device_numa(device.index)
        self.name: str = name
        self.rank: int = rank
        self.world_size: int = world_size
        self.capacity_input_elements: int = capacity_input_elements
        self.blocks: int = blocks
        self.threads: int = threads
        self.device: torch.device = device
        self.slot_stride_bytes: int = _align_up(capacity_input_elements * 2, 4096)
        self.total_bytes: int = _align_up(
            SHM_V5_HEADER_BYTES + NUM_RING_SLOTS * self.slot_stride_bytes, 4096
        )
        self._closed: bool = False
        self._extension = get_host_mapped_allreduce_extension()
        self._init_per_rank_shm(
            name=name,
            rank=rank,
            world_size=world_size,
            total_bytes=self.total_bytes,
            exchange_group=exchange_group,
            device=device,
            extension=self._extension,
        )
        self._registered: bool = True
        self.control: torch.Tensor = torch.zeros(MAX_BLOCKS, dtype=torch.int64, device=device)

    def all_gather(
        self,
        input: torch.Tensor,
        *,
        dim: int = -1,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """AllGather `input` along `dim` across `world_size` ranks."""
        if self._closed:
            raise RuntimeError("AllGather channel is closed")
        ndim = input.ndim
        if dim < 0:
            dim += ndim
        if not 0 <= dim < ndim:
            raise ValueError(f"dim {dim} out of range for {ndim}D tensor")
        if not input.is_contiguous():
            input = input.contiguous()
        out_shape = list(input.shape)
        out_shape[dim] *= self.world_size
        if output is None:
            output = torch.empty(out_shape, dtype=input.dtype, device=input.device)
        elif list(output.shape) != out_shape:
            raise ValueError(f"output shape {list(output.shape)} != expected {out_shape}")
        if input.numel() == 0:
            return output
        if input.numel() > self.capacity_input_elements:
            raise ValueError("input exceeds AllGather channel capacity")
        outer_rows = math.prod(input.shape[:dim]) if dim > 0 else 1
        inner_elems = math.prod(input.shape[dim:])
        if inner_elems % 8 != 0:
            raise ValueError(f"inner_elems={inner_elems} must be divisible by 8")
        vecs_per_row = inner_elems // 8
        stream_handle = torch.cuda.current_stream(self.device).cuda_stream
        self._extension.launch_shm_v5_allgather(
            input,
            output,
            self.control,
            self._rank_shm_ptrs,
            self.slot_stride_bytes,
            outer_rows,
            vecs_per_row,
            self.rank,
            self.world_size,
            self.blocks,
            self.threads,
            stream_handle,
        )
        return output

    def close(self, *, abort: bool = False) -> None:
        if self._closed:
            return
        if abort:
            self.signal_abort()
        torch.cuda.synchronize(self.device)
        if self._registered:
            self._close_per_rank_shm(self._extension)
            self._registered = False
        self._closed = True

    def __enter__(self) -> HostMappedShmAllGather:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class HostMappedShmArgmax:
    """Host-mapped SHM ring for lock-free TP=8 deterministic Top-2 `(logit, token_id)` gather."""

    def __init__(
        self,
        *,
        name: str,
        rank: int,
        world_size: int,
        device: torch.device,
        exchange_group: dist.ProcessGroup,
        max_rows_per_step: int,
    ) -> None:
        if max_rows_per_step <= 0:
            raise ValueError("max_rows_per_step must be positive")
        self.name: str = name
        self.rank: int = rank
        self.world_size: int = world_size
        self.device: torch.device = device
        self.max_rows_per_step: int = max_rows_per_step
        unaligned_bytes = 256 + NUM_RING_SLOTS * MAX_RANKS * (max_rows_per_step * 2) * 8
        self.total_bytes: int = _align_up(unaligned_bytes, 4096)
        self._closed: bool = False
        self._owns_unlink: bool = False
        self._shm_path: str = f"/dev/shm/{name}"
        self._extension = get_host_mapped_allreduce_extension()

        if rank == 0:
            fd = os.open(self._shm_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
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
                self.total_bytes,
            )
        dist.barrier(group=exchange_group)
        if rank != 0:
            fd = os.open(self._shm_path, os.O_RDWR)
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
        if rank == 0 and self._owns_unlink:
            try:
                os.unlink(self._shm_path)
            except FileNotFoundError:
                pass
            self._owns_unlink = False
        self._host_addr: int = ctypes.addressof(ctypes.c_char.from_buffer(self._mapping))
        self._dev_addr: int = self._extension.register_mapped_host(
            self._host_addr, self.total_bytes
        )
        self._registered: bool = True
        self.step_counter: torch.Tensor = torch.zeros(1, dtype=torch.int64, device=device)

    def gather_top2_from_logits(
        self,
        local_logits_bf16: torch.Tensor,
        *,
        local_top2_vals_buf: torch.Tensor,
        local_top2_ids_buf: torch.Tensor,
        out_tokens: torch.Tensor,
        out_logprob_bits: torch.Tensor,
        vocab_per_rank: int,
    ) -> None:
        num_rows = int(local_logits_bf16.shape[0])
        if num_rows <= 0:
            return
        stream_handle = torch.cuda.current_stream(self.device).cuda_stream
        self._extension.launch_local_top2_bf16(
            local_logits_bf16,
            local_top2_vals_buf,
            local_top2_ids_buf,
            num_rows,
            stream_handle,
        )
        self._extension.launch_tp8_argmax_gather_shm(
            local_top2_vals_buf,
            local_top2_ids_buf,
            out_tokens,
            out_logprob_bits,
            self.step_counter,
            self._dev_addr,
            num_rows,
            self.max_rows_per_step,
            self.rank,
            self.world_size,
            vocab_per_rank,
            stream_handle,
        )

    def signal_abort(self) -> None:
        mm = getattr(self, "_mapping", None)
        if mm is not None and not mm.closed and len(mm) >= ARGMAX_SHM_ABORT_OFFSET + 8:
            struct.pack_into("<Q", mm, ARGMAX_SHM_ABORT_OFFSET, 1)

    def close(self, *, abort: bool = False) -> None:
        if self._closed:
            return
        if abort:
            self.signal_abort()
        torch.cuda.synchronize(self.device)
        if self._registered:
            self._extension.unregister_mapped_host(self._host_addr)
            self._registered = False
        self._mapping.close()
        if self._owns_unlink:
            try:
                os.unlink(self._shm_path)
            except FileNotFoundError:
                pass
        self._closed = True
