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

"""Explicit CUDA-Graph-safe Triton + CUDA kernels for Qwen3.5-35B-A3B (TP=8, sm_120).

All runtime token/sequence/page/expert counts are passed as runtime arguments (never
`tl.constexpr`) so Triton compiles each kernel once at startup with zero runtime JIT
recompilations. Every global memory offset scaled by a runtime index casts to `int64`
before multiplication.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# Inline CUDA C++ helper: GPU-only shared-memory MoE token-to-expert bucketing
# ---------------------------------------------------------------------------

_MOE_ALIGN_CPP = r"""
#include <torch/extension.h>
#include <cstdint>

void launch_moe_align_block_size_cuda(
    const int32_t* topk_ids,
    int32_t* sorted_pair_ids,
    int32_t* expert_ids_per_tile,
    int32_t* num_valid_tiles,
    int64_t num_pairs,
    int64_t max_padded_slots,
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

void launch_moe_align_block_size(
    torch::Tensor topk_ids,
    torch::Tensor sorted_pair_ids,
    torch::Tensor expert_ids_per_tile,
    torch::Tensor num_valid_tiles,
    int64_t num_pairs,
    std::uintptr_t stream_handle) {
    launch_moe_align_block_size_cuda(
        topk_ids.data_ptr<int32_t>(),
        sorted_pair_ids.data_ptr<int32_t>(),
        expert_ids_per_tile.data_ptr<int32_t>(),
        num_valid_tiles.data_ptr<int32_t>(),
        num_pairs,
        sorted_pair_ids.numel(),
        stream_handle);
}

void launch_local_top2_bf16(
    torch::Tensor logits_bf16,
    torch::Tensor out_top2_vals,
    torch::Tensor out_top2_ids,
    int64_t num_rows,
    std::uintptr_t stream_handle) {
    TORCH_CHECK(logits_bf16.scalar_type() == torch::kBFloat16, "logits_bf16 must be bfloat16");
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

_MOE_ALIGN_CUDA = r"""
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

constexpr int kNumExperts = 256;
constexpr int kBlockM = 16;

__global__ void moe_align_block_size_256_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_pair_ids,
    int32_t* __restrict__ expert_ids_per_tile,
    int32_t* __restrict__ num_valid_tiles_out,
    int64_t num_pairs,
    int64_t max_padded_slots) {
    __shared__ int s_counts[kNumExperts];
    __shared__ int s_tiles[kNumExperts];
    __shared__ int s_tile_offsets[kNumExperts + 1];
    __shared__ int s_write_cursor[kNumExperts];

    const int tid = threadIdx.x;
    s_counts[tid] = 0;
    __syncthreads();

    for (int64_t i = tid; i < num_pairs; i += kNumExperts) {
        const int e = topk_ids[i];
        if (static_cast<unsigned>(e) < kNumExperts) {
            atomicAdd(&s_counts[e], 1);
        }
    }
    __syncthreads();

    const int count = s_counts[tid];
    s_tiles[tid] = (count + kBlockM - 1) / kBlockM;
    __syncthreads();

    if (tid == 0) {
        int running = 0;
        for (int e = 0; e < kNumExperts; ++e) {
            s_tile_offsets[e] = running;
            running += s_tiles[e];
        }
        s_tile_offsets[kNumExperts] = running;
        *num_valid_tiles_out = running;
    }
    __syncthreads();

    const int total_tiles = s_tile_offsets[kNumExperts];
    const int64_t total_slots = static_cast<int64_t>(total_tiles) * kBlockM;
    const int64_t clear_limit = total_slots < max_padded_slots ? total_slots : max_padded_slots;
    for (int64_t idx = tid; idx < clear_limit; idx += kNumExperts) {
        sorted_pair_ids[idx] = -1;
    }

    const int tile_start = s_tile_offsets[tid];
    const int tile_end = s_tile_offsets[tid + 1];
    for (int t = tile_start; t < tile_end; ++t) {
        expert_ids_per_tile[t] = tid;
    }
    s_write_cursor[tid] = tile_start * kBlockM;
    __syncthreads();

    for (int64_t i = tid; i < num_pairs; i += kNumExperts) {
        const int e = topk_ids[i];
        if (static_cast<unsigned>(e) < kNumExperts) {
            const int dst = atomicAdd(&s_write_cursor[e], 1);
            if (dst < max_padded_slots) {
                sorted_pair_ids[dst] = static_cast<int32_t>(i);
            }
        }
    }
}

__device__ __forceinline__ uint64_t sys_ld_acq_u64(const uint64_t* addr) {
    uint64_t val;
    asm volatile("ld.global.acquire.sys.b64 %0, [%1];" : "=l"(val) : "l"(addr) : "memory");
    return val;
}

// Returns false if the host raised `abort_flag` before `*flag_addr >= target`. The abort word
// is polled only every 4096 iterations so the fast path adds no extra PCIe loads.
__device__ __forceinline__ bool spin_wait_ge_or_abort_u64(
    const uint64_t* flag_addr,
    uint64_t target,
    const uint64_t* abort_flag) {
    uint32_t iters = 0;
    while (sys_ld_acq_u64(flag_addr) < target) {
        __nanosleep(32);
        if ((++iters & 0xFFFU) == 0U && sys_ld_acq_u64(abort_flag) != 0ULL) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ void sys_st_rel_u64(uint64_t* addr, uint64_t val) {
    asm volatile("st.global.release.sys.b64 [%0], %1;" :: "l"(addr), "l"(val) : "memory");
}

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

// Synchronizes per-rank local Top-2 `(logit, global_token_id)` across 8 PCIe ranks via
// mapped host SHM (128 bytes per row across 8 ranks) using a 4-slot lock-free ring.
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
    constexpr int kSlots = 4;
    constexpr int kMaxRanks = 8;
    // Layout in SHM:
    // [0 .. 64):    ready_flags[8] (uint64_t)
    // [64 .. 128):  done_flags[8]  (uint64_t)
    // [128 .. 136): abort_flag     (uint64_t, set by the host on cluster abort)
    // [256 .. ):    4 slots * 8 ranks * (max_rows_per_step * 2) * sizeof(ValIdxPair)
    uint64_t* ready_flags = reinterpret_cast<uint64_t*>(host_base);
    uint64_t* done_flags = reinterpret_cast<uint64_t*>(host_base + 64);
    const uint64_t* abort_flag = reinterpret_cast<const uint64_t*>(host_base + 128);

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
    const int slot = static_cast<int>((step - 1ULL) & (kSlots - 1));

    if (step > kSlots) {
        const uint64_t min_done = step - kSlots;
        if (threadIdx.x < world_size) {
            if (!spin_wait_ge_or_abort_u64(done_flags + threadIdx.x, min_done, abort_flag)) {
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
        sys_st_rel_u64(ready_flags + rank, step);
    }
    if (threadIdx.x < world_size) {
        if (!spin_wait_ge_or_abort_u64(ready_flags + threadIdx.x, step, abort_flag)) {
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
        sys_st_rel_u64(done_flags + rank, step);
    }
}

}  // namespace

void launch_moe_align_block_size_cuda(
    const int32_t* topk_ids,
    int32_t* sorted_pair_ids,
    int32_t* expert_ids_per_tile,
    int32_t* num_valid_tiles,
    int64_t num_pairs,
    int64_t max_padded_slots,
    std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    moe_align_block_size_256_kernel<<<1, 256, 0, stream>>>(
        topk_ids,
        sorted_pair_ids,
        expert_ids_per_tile,
        num_valid_tiles,
        num_pairs,
        max_padded_slots);
    ZSTEP_CUDA_CHECK(cudaGetLastError());
}

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

_QWEN35_EXT = None


def get_qwen35_cuda_ext():
    """Compile or return the inline CUDA helper extension for MoE align + TP8 Top-2 gather."""
    global _QWEN35_EXT
    if _QWEN35_EXT is None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        _QWEN35_EXT = load_inline(
            name="zstep_qwen35_ops_sm120_v5_ext",
            cpp_sources=[_MOE_ALIGN_CPP],
            cuda_sources=[_MOE_ALIGN_CUDA],
            functions=[
                "launch_moe_align_block_size",
                "launch_local_top2_bf16",
                "launch_tp8_argmax_gather_shm",
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_120,code=sm_120"],
            with_cuda=True,
            verbose=False,
        )
    return _QWEN35_EXT


# ---------------------------------------------------------------------------
# 1. Zero-centered Qwen3.5 RMSNorm (+ optional fused residual addition)
# ---------------------------------------------------------------------------


@triton.jit
def _zero_centered_rmsnorm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    stride_x_row,
    stride_out_row,
    eps,
    HIDDEN_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, HIDDEN_SIZE)
    x = tl.load(x_ptr + row * stride_x_row + cols).to(tl.float32)
    w = tl.load(w_ptr + cols).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) * (1.0 / HIDDEN_SIZE) + eps)
    y = (x * inv_rms) * (1.0 + w)
    tl.store(out_ptr + row * stride_out_row + cols, y.to(tl.bfloat16))


@triton.jit
def _fused_add_zero_centered_rmsnorm_kernel(
    x_ptr,
    residual_ptr,
    w_ptr,
    normed_out_ptr,
    stride_row,
    eps,
    HIDDEN_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, HIDDEN_SIZE)
    off = row * stride_row + cols
    x = tl.load(x_ptr + off).to(tl.float32)
    res = tl.load(residual_ptr + off).to(tl.float32)
    summed = x + res
    tl.store(residual_ptr + off, summed.to(tl.bfloat16))
    summed_bf16_f32 = summed.to(tl.bfloat16).to(tl.float32)
    w = tl.load(w_ptr + cols).to(tl.float32)
    inv_rms = tl.rsqrt(
        tl.sum(summed_bf16_f32 * summed_bf16_f32, axis=0) * (1.0 / HIDDEN_SIZE) + eps
    )
    y = (summed_bf16_f32 * inv_rms) * (1.0 + w)
    tl.store(normed_out_ptr + off, y.to(tl.bfloat16))


def zero_centered_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute `(x.float() * rsqrt(mean(x^2) + eps) * (1 + weight.float())).to(bf16)`."""
    rows, hidden_size = x.shape
    if hidden_size != 2048:
        raise ValueError(f"Expected hidden_size=2048, got {hidden_size}")
    _zero_centered_rmsnorm_kernel[(rows,)](
        x,
        weight,
        out,
        x.stride(0),
        out.stride(0),
        eps,
        HIDDEN_SIZE=2048,
        num_warps=8,
    )
    return out


def fused_add_zero_centered_rmsnorm(
    delta: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    normed_out: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Update `residual += delta` in-place (BF16) and write zero-centered RMSNorm to `normed_out`."""
    rows, hidden_size = delta.shape
    if hidden_size != 2048:
        raise ValueError(f"Expected hidden_size=2048, got {hidden_size}")
    _fused_add_zero_centered_rmsnorm_kernel[(rows,)](
        delta,
        residual,
        weight,
        normed_out,
        residual.stride(0),
        eps,
        HIDDEN_SIZE=2048,
        num_warps=8,
    )
    return normed_out


# ---------------------------------------------------------------------------
# 2. Full-Attention Prologue: QK Zero-Centered Norm + Partial RoPE + KV Cache Write
# ---------------------------------------------------------------------------


@triton.jit
def _qwen35_attn_prologue_kernel(
    qkv_ptr,
    positions_ptr,
    req_slot_ids_ptr,
    page_table_ptr,
    q_norm_w_ptr,
    k_norm_w_ptr,
    cos_cache_ptr,
    sin_cache_ptr,
    q_out_ptr,
    gate_out_ptr,
    k_cache_ptr,
    v_cache_ptr,
    stride_qkv_row,
    stride_pt_slot,
    stride_kv_page,
    eps,
    PAGE_SIZE: tl.constexpr,
):
    # qkv row layout for TP=8 (1536 cols):
    # [0:256]    Q0, [256:512]  Gate0
    # [512:768]  Q1, [768:1024] Gate1
    # [1024:1280] K0, [1280:1536] V0
    row = tl.program_id(0).to(tl.int64)
    pos = tl.load(positions_ptr + row).to(tl.int64)
    slot = tl.load(req_slot_ids_ptr + row).to(tl.int64)

    d_full = tl.arange(0, 256)
    d_half = tl.arange(0, 32)
    qw = 1.0 + tl.load(q_norm_w_ptr + d_full).to(tl.float32)
    kw = 1.0 + tl.load(k_norm_w_ptr + d_full).to(tl.float32)

    cos_h = tl.load(cos_cache_ptr + pos * 32 + d_half).to(tl.bfloat16).to(tl.float32)
    sin_h = tl.load(sin_cache_ptr + pos * 32 + d_half).to(tl.bfloat16).to(tl.float32)

    qkv_base = qkv_ptr + row * stride_qkv_row
    # Gate0 and Gate1 direct copy
    g0 = tl.load(qkv_base + 256 + d_full)
    g1 = tl.load(qkv_base + 768 + d_full)
    tl.store(gate_out_ptr + row * 512 + d_full, g0)
    tl.store(gate_out_ptr + row * 512 + 256 + d_full, g1)

    # Q0 norm + partial RoPE (first 64 dims)
    q0 = tl.load(qkv_base + d_full).to(tl.float32)
    q0_inv = tl.rsqrt(tl.sum(q0 * q0, axis=0) * (1.0 / 256.0) + eps)
    q0_n = (q0 * q0_inv * qw).to(tl.bfloat16)
    tl.store(q_out_ptr + row * 512 + d_full, q0_n)

    q0_lo = tl.load(q_out_ptr + row * 512 + d_half).to(tl.float32)
    q0_hi = tl.load(q_out_ptr + row * 512 + 32 + d_half).to(tl.float32)
    tl.store(q_out_ptr + row * 512 + d_half, (q0_lo * cos_h - q0_hi * sin_h).to(tl.bfloat16))
    tl.store(
        q_out_ptr + row * 512 + 32 + d_half, (q0_hi * cos_h + q0_lo * sin_h).to(tl.bfloat16)
    )

    # Q1 norm + partial RoPE
    q1 = tl.load(qkv_base + 512 + d_full).to(tl.float32)
    q1_inv = tl.rsqrt(tl.sum(q1 * q1, axis=0) * (1.0 / 256.0) + eps)
    q1_n = (q1 * q1_inv * qw).to(tl.bfloat16)
    tl.store(q_out_ptr + row * 512 + 256 + d_full, q1_n)

    q1_lo = tl.load(q_out_ptr + row * 512 + 256 + d_half).to(tl.float32)
    q1_hi = tl.load(q_out_ptr + row * 512 + 288 + d_half).to(tl.float32)
    tl.store(
        q_out_ptr + row * 512 + 256 + d_half, (q1_lo * cos_h - q1_hi * sin_h).to(tl.bfloat16)
    )
    tl.store(
        q_out_ptr + row * 512 + 288 + d_half, (q1_hi * cos_h + q1_lo * sin_h).to(tl.bfloat16)
    )

    # K0 norm + partial RoPE + direct write into paged KV cache
    page_idx = pos // PAGE_SIZE
    page_off = pos % PAGE_SIZE
    phys_page = tl.load(page_table_ptr + slot * stride_pt_slot + page_idx).to(tl.int64)
    kv_dst_off = phys_page * stride_kv_page + page_off * 256

    k0 = tl.load(qkv_base + 1024 + d_full).to(tl.float32)
    k0_inv = tl.rsqrt(tl.sum(k0 * k0, axis=0) * (1.0 / 256.0) + eps)
    k0_n = (k0 * k0_inv * kw).to(tl.bfloat16)
    tl.store(k_cache_ptr + kv_dst_off + d_full, k0_n)

    k0_lo = tl.load(k_cache_ptr + kv_dst_off + d_half).to(tl.float32)
    k0_hi = tl.load(k_cache_ptr + kv_dst_off + 32 + d_half).to(tl.float32)
    tl.store(k_cache_ptr + kv_dst_off + d_half, (k0_lo * cos_h - k0_hi * sin_h).to(tl.bfloat16))
    tl.store(
        k_cache_ptr + kv_dst_off + 32 + d_half, (k0_hi * cos_h + k0_lo * sin_h).to(tl.bfloat16)
    )

    v0 = tl.load(qkv_base + 1280 + d_full)
    tl.store(v_cache_ptr + kv_dst_off + d_full, v0)


def qwen35_attn_prologue(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    req_slot_ids: torch.Tensor,
    page_table: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    *,
    q_out: torch.Tensor,
    gate_out: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    eps: float,
) -> None:
    """Apply QK zero-centered norm + partial RoPE and write K/V into 16-token paged cache."""
    rows = qkv.shape[0]
    _qwen35_attn_prologue_kernel[(rows,)](
        qkv,
        positions,
        req_slot_ids,
        page_table,
        q_norm_weight,
        k_norm_weight,
        cos_cache,
        sin_cache,
        q_out,
        gate_out,
        k_cache,
        v_cache,
        qkv.stride(0),
        page_table.stride(0),
        k_cache.stride(0),
        eps,
        PAGE_SIZE=16,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# 3. Paged GQA Full Attention (H_q=2, H_kv=1, D=256, PAGE_SIZE=16) + Sigmoid Gate
# ---------------------------------------------------------------------------


@triton.jit
def _paged_gqa_decode_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr,
    seq_lens_ptr,
    req_slot_ids_ptr,
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    stride_pt_slot,
    stride_kv_page,
    stride_acc_seq,
    stride_acc_split,
    sm_scale,
    num_active_seqs,
    SPLIT_K: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(0).to(tl.int64)
    split_idx = tl.program_id(1).to(tl.int64)
    d = tl.arange(0, 256)

    if seq_idx >= num_active_seqs:
        return

    seq_len = tl.load(seq_lens_ptr + seq_idx).to(tl.int32)
    if seq_len <= 0:
        m_base = seq_idx * SPLIT_K * 2 + split_idx * 2
        tl.store(partial_m_ptr + m_base + 0, -1.0e30)
        tl.store(partial_m_ptr + m_base + 1, -1.0e30)
        tl.store(partial_l_ptr + m_base + 0, 0.0)
        tl.store(partial_l_ptr + m_base + 1, 0.0)
        return

    slot = tl.load(req_slot_ids_ptr + seq_idx).to(tl.int64)
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    pages_per_split = (num_pages + SPLIT_K - 1) // SPLIT_K
    p_start = split_idx * pages_per_split
    p_end = tl.minimum(num_pages, p_start + pages_per_split)

    m_base = seq_idx * SPLIT_K * 2 + split_idx * 2
    acc_base = seq_idx * stride_acc_seq + split_idx * stride_acc_split
    if p_start >= p_end:
        tl.store(partial_m_ptr + m_base + 0, -1.0e30)
        tl.store(partial_m_ptr + m_base + 1, -1.0e30)
        tl.store(partial_l_ptr + m_base + 0, 0.0)
        tl.store(partial_l_ptr + m_base + 1, 0.0)
        return

    q0 = tl.load(q_ptr + seq_idx * 512 + d).to(tl.float32) * sm_scale
    q1 = tl.load(q_ptr + seq_idx * 512 + 256 + d).to(tl.float32) * sm_scale

    m0 = -1.0e30
    l0 = 0.0
    acc0 = tl.zeros([256], dtype=tl.float32)

    m1 = -1.0e30
    l1 = 0.0
    acc1 = tl.zeros([256], dtype=tl.float32)

    offs_p = tl.arange(0, PAGE_SIZE)
    for page_idx in range(p_start, p_end):
        phys_page = tl.load(page_table_ptr + slot * stride_pt_slot + page_idx).to(tl.int64)
        tok_pos = page_idx * PAGE_SIZE + offs_p
        valid = tok_pos < seq_len

        kv_ptrs = phys_page * stride_kv_page + offs_p[:, None] * 256 + d[None, :]
        k_tile = tl.load(k_cache_ptr + kv_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
        v_tile = tl.load(v_cache_ptr + kv_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)

        s0 = tl.where(valid, tl.sum(k_tile * q0[None, :], axis=1), -1.0e30)
        m0_new = tl.maximum(m0, tl.max(s0, axis=0))
        alpha0 = tl.exp(m0 - m0_new)
        p0 = tl.exp(s0 - m0_new)
        l0 = l0 * alpha0 + tl.sum(p0, axis=0)
        acc0 = acc0 * alpha0 + tl.sum(v_tile * p0[:, None], axis=0)
        m0 = m0_new

        s1 = tl.where(valid, tl.sum(k_tile * q1[None, :], axis=1), -1.0e30)
        m1_new = tl.maximum(m1, tl.max(s1, axis=0))
        alpha1 = tl.exp(m1 - m1_new)
        p1 = tl.exp(s1 - m1_new)
        l1 = l1 * alpha1 + tl.sum(p1, axis=0)
        acc1 = acc1 * alpha1 + tl.sum(v_tile * p1[:, None], axis=0)
        m1 = m1_new

    tl.store(partial_m_ptr + m_base + 0, m0)
    tl.store(partial_m_ptr + m_base + 1, m1)
    tl.store(partial_l_ptr + m_base + 0, l0)
    tl.store(partial_l_ptr + m_base + 1, l1)
    tl.store(partial_acc_ptr + acc_base + d, acc0)
    tl.store(partial_acc_ptr + acc_base + 256 + d, acc1)


@triton.jit
def _paged_gqa_decode_reduce_gated_kernel(
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    gate_ptr,
    out_ptr,
    stride_acc_seq,
    stride_acc_split,
    num_active_seqs,
    SPLIT_K: tl.constexpr,
):
    seq_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    d = tl.arange(0, 256)

    if seq_idx >= num_active_seqs:
        tl.store(out_ptr + seq_idx * 512 + head_idx * 256 + d, tl.zeros([256], dtype=tl.bfloat16))
        return

    splits = tl.arange(0, SPLIT_K)
    m_vals = tl.load(partial_m_ptr + seq_idx * SPLIT_K * 2 + splits * 2 + head_idx)
    l_vals = tl.load(partial_l_ptr + seq_idx * SPLIT_K * 2 + splits * 2 + head_idx)
    m_max = tl.max(m_vals, axis=0)
    has_any = l_vals > 0.0
    scales = tl.where(has_any, tl.exp(m_vals - m_max), 0.0)
    l_sum = tl.sum(l_vals * scales, axis=0)

    acc = tl.zeros([256], dtype=tl.float32)
    for s in range(SPLIT_K):
        l_s = tl.load(partial_l_ptr + seq_idx * SPLIT_K * 2 + s * 2 + head_idx)
        if l_s > 0.0:
            m_s = tl.load(partial_m_ptr + seq_idx * SPLIT_K * 2 + s * 2 + head_idx)
            w_s = tl.exp(m_s - m_max)
            part = tl.load(
                partial_acc_ptr + seq_idx * stride_acc_seq + s * stride_acc_split + head_idx * 256 + d
            )
            acc += part * w_s

    inv_l = tl.where(l_sum > 0.0, 1.0 / l_sum, 0.0)
    attn_bf16 = (acc * inv_l).to(tl.bfloat16)
    gate = tl.load(gate_ptr + seq_idx * 512 + head_idx * 256 + d).to(tl.float32)
    gated = (attn_bf16.to(tl.float32) * tl.sigmoid(gate)).to(tl.bfloat16)
    tl.store(out_ptr + seq_idx * 512 + head_idx * 256 + d, gated)


@triton.jit
def _paged_gqa_prefill_varlen_gated_kernel(
    q_ptr,
    gate_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    req_slot_ids_ptr,
    out_ptr,
    stride_pt_slot,
    stride_kv_page,
    sm_scale,
    BLOCK_M: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    q_blk = tl.program_id(0).to(tl.int64)
    seq_idx = tl.program_id(1).to(tl.int64)
    head_idx = tl.program_id(2).to(tl.int64)

    q_start = tl.load(query_start_loc_ptr + seq_idx).to(tl.int64)
    q_end = tl.load(query_start_loc_ptr + seq_idx + 1).to(tl.int64)
    q_len = q_end - q_start
    m_off = q_blk * BLOCK_M
    if m_off >= q_len:
        return

    seq_len = tl.load(seq_lens_ptr + seq_idx).to(tl.int64)
    start_pos = seq_len - q_len
    slot = tl.load(req_slot_ids_ptr + q_start).to(tl.int64)

    offs_m = m_off + tl.arange(0, BLOCK_M)
    valid_m = offs_m < q_len
    q_abs_pos = start_pos + offs_m

    d_lo = tl.arange(0, 128)
    d_hi = 128 + tl.arange(0, 128)
    q_row_ptrs = q_ptr + (q_start + offs_m[:, None]) * 512 + head_idx * 256
    q_lo = tl.load(q_row_ptrs + d_lo[None, :], mask=valid_m[:, None], other=0.0)
    q_hi = tl.load(q_row_ptrs + d_hi[None, :], mask=valid_m[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], -1.0e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_lo = tl.zeros([BLOCK_M, 128], dtype=tl.float32)
    acc_hi = tl.zeros([BLOCK_M, 128], dtype=tl.float32)

    max_kv_pos = tl.minimum(seq_len, start_pos + m_off + BLOCK_M)
    num_kv_pages = (max_kv_pos + PAGE_SIZE - 1) // PAGE_SIZE
    offs_n = tl.arange(0, PAGE_SIZE)

    for page_idx in range(0, num_kv_pages):
        phys_page = tl.load(page_table_ptr + slot * stride_pt_slot + page_idx).to(tl.int64)
        kv_pos = page_idx * PAGE_SIZE + offs_n
        valid_n = kv_pos < seq_len

        kv_base = phys_page * stride_kv_page + offs_n[:, None] * 256
        k_lo = tl.load(k_cache_ptr + kv_base + d_lo[None, :], mask=valid_n[:, None], other=0.0)
        k_hi = tl.load(k_cache_ptr + kv_base + d_hi[None, :], mask=valid_n[:, None], other=0.0)

        qk = (tl.dot(q_lo, tl.trans(k_lo)) + tl.dot(q_hi, tl.trans(k_hi))) * sm_scale
        causal_mask = (q_abs_pos[:, None] >= kv_pos[None, :]) & valid_n[None, :]
        qk = tl.where(causal_mask, qk, -1.0e30)

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_lo = tl.load(v_cache_ptr + kv_base + d_lo[None, :], mask=valid_n[:, None], other=0.0)
        v_hi = tl.load(v_cache_ptr + kv_base + d_hi[None, :], mask=valid_n[:, None], other=0.0)
        p_bf16 = p.to(tl.bfloat16)
        acc_lo = acc_lo * alpha[:, None] + tl.dot(p_bf16, v_lo)
        acc_hi = acc_hi * alpha[:, None] + tl.dot(p_bf16, v_hi)
        m_i = m_new

    inv_l = tl.where(l_i > 0.0, 1.0 / l_i, 0.0)[:, None]
    out_lo_bf16 = (acc_lo * inv_l).to(tl.bfloat16)
    out_hi_bf16 = (acc_hi * inv_l).to(tl.bfloat16)

    gate_row_ptrs = gate_ptr + (q_start + offs_m[:, None]) * 512 + head_idx * 256
    g_lo = tl.load(gate_row_ptrs + d_lo[None, :], mask=valid_m[:, None], other=0.0).to(tl.float32)
    g_hi = tl.load(gate_row_ptrs + d_hi[None, :], mask=valid_m[:, None], other=0.0).to(tl.float32)

    out_row_ptrs = out_ptr + (q_start + offs_m[:, None]) * 512 + head_idx * 256
    tl.store(
        out_row_ptrs + d_lo[None, :],
        (out_lo_bf16.to(tl.float32) * tl.sigmoid(g_lo)).to(tl.bfloat16),
        mask=valid_m[:, None],
    )
    tl.store(
        out_row_ptrs + d_hi[None, :],
        (out_hi_bf16.to(tl.float32) * tl.sigmoid(g_hi)).to(tl.bfloat16),
        mask=valid_m[:, None],
    )


def paged_gqa_decode_gated(
    q: torch.Tensor,
    gate: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    req_slot_ids: torch.Tensor,
    *,
    partial_acc: torch.Tensor,
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    out: torch.Tensor,
    num_active_seqs: int,
    sm_scale: float,
) -> torch.Tensor:
    """Split-K Paged GQA decode (H_q=2, H_kv=1, D=256) fused with output sigmoid gate."""
    batch_rows = q.shape[0]
    split_k = partial_acc.shape[1]
    _paged_gqa_decode_splitk_kernel[(batch_rows, split_k)](
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        req_slot_ids,
        partial_acc,
        partial_m,
        partial_l,
        page_table.stride(0),
        k_cache.stride(0),
        partial_acc.stride(0),
        partial_acc.stride(1),
        sm_scale,
        num_active_seqs,
        SPLIT_K=16,
        PAGE_SIZE=16,
        num_warps=4,
    )
    _paged_gqa_decode_reduce_gated_kernel[(batch_rows, 2)](
        partial_acc,
        partial_m,
        partial_l,
        gate,
        out,
        partial_acc.stride(0),
        partial_acc.stride(1),
        num_active_seqs,
        SPLIT_K=16,
        num_warps=4,
    )
    return out


def paged_gqa_prefill_varlen_gated(
    q: torch.Tensor,
    gate: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    req_slot_ids: torch.Tensor,
    *,
    out: torch.Tensor,
    num_seqs: int,
    max_q_len: int,
    sm_scale: float,
) -> torch.Tensor:
    """Varlen FlashAttention-2 prefill over 16-token paged KV cache + sigmoid output gate."""
    grid = (triton.cdiv(max_q_len, 64), num_seqs, 2)
    _paged_gqa_prefill_varlen_gated_kernel[grid](
        q,
        gate,
        k_cache,
        v_cache,
        page_table,
        query_start_loc,
        seq_lens,
        req_slot_ids,
        out,
        page_table.stride(0),
        k_cache.stride(0),
        sm_scale,
        BLOCK_M=64,
        PAGE_SIZE=16,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# 4. Linear Attention (Qwen3_5MoeGatedDeltaNet): Causal Conv1d + Gated Delta Rule
# ---------------------------------------------------------------------------


@triton.jit
def _causal_conv1d_decode_kernel(
    qkvzba_ptr,
    conv_state_ptr,
    conv_w_ptr,
    positions_ptr,
    req_slot_ids_ptr,
    qkv_out_ptr,
    stride_in_row,
    stride_state_slot,
    num_active_rows,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    c_blk = tl.program_id(1).to(tl.int64)
    c = c_blk * BLOCK_C + tl.arange(0, BLOCK_C)

    if row >= num_active_rows:
        tl.store(qkv_out_ptr + row * 1024 + c, tl.zeros([BLOCK_C], dtype=tl.bfloat16))
        return

    slot = tl.load(req_slot_ids_ptr + row).to(tl.int64)
    pos = tl.load(positions_ptr + row).to(tl.int32)
    x = tl.load(qkvzba_ptr + row * stride_in_row + c).to(tl.float32)

    state_base = conv_state_ptr + slot * stride_state_slot + c * 3
    if pos == 0:
        s0 = tl.zeros([BLOCK_C], dtype=tl.float32)
        s1 = tl.zeros([BLOCK_C], dtype=tl.float32)
        s2 = tl.zeros([BLOCK_C], dtype=tl.float32)
    else:
        s0 = tl.load(state_base + 0).to(tl.float32)
        s1 = tl.load(state_base + 1).to(tl.float32)
        s2 = tl.load(state_base + 2).to(tl.float32)

    tl.store(state_base + 0, s1.to(tl.bfloat16))
    tl.store(state_base + 1, s2.to(tl.bfloat16))
    tl.store(state_base + 2, x.to(tl.bfloat16))

    w0 = tl.load(conv_w_ptr + c * 4 + 0).to(tl.float32)
    w1 = tl.load(conv_w_ptr + c * 4 + 1).to(tl.float32)
    w2 = tl.load(conv_w_ptr + c * 4 + 2).to(tl.float32)
    w3 = tl.load(conv_w_ptr + c * 4 + 3).to(tl.float32)

    acc = w0 * s0 + w1 * s1 + w2 * s2 + w3 * x
    y = acc * tl.sigmoid(acc)
    tl.store(qkv_out_ptr + row * 1024 + c, y.to(tl.bfloat16))


@triton.jit
def _causal_conv1d_prefill_varlen_kernel(
    qkvzba_ptr,
    conv_state_ptr,
    conv_w_ptr,
    query_start_loc_ptr,
    positions_ptr,
    req_slot_ids_ptr,
    qkv_out_ptr,
    stride_in_row,
    stride_state_slot,
    BLOCK_C: tl.constexpr,
):
    seq_idx = tl.program_id(0).to(tl.int64)
    c_blk = tl.program_id(1).to(tl.int64)
    c = c_blk * BLOCK_C + tl.arange(0, BLOCK_C)

    q_start = tl.load(query_start_loc_ptr + seq_idx).to(tl.int64)
    q_end = tl.load(query_start_loc_ptr + seq_idx + 1).to(tl.int64)
    if q_start >= q_end:
        return

    slot = tl.load(req_slot_ids_ptr + q_start).to(tl.int64)
    start_pos = tl.load(positions_ptr + q_start).to(tl.int32)
    state_base = conv_state_ptr + slot * stride_state_slot + c * 3

    if start_pos == 0:
        s0 = tl.zeros([BLOCK_C], dtype=tl.float32)
        s1 = tl.zeros([BLOCK_C], dtype=tl.float32)
        s2 = tl.zeros([BLOCK_C], dtype=tl.float32)
    else:
        s0 = tl.load(state_base + 0).to(tl.float32)
        s1 = tl.load(state_base + 1).to(tl.float32)
        s2 = tl.load(state_base + 2).to(tl.float32)

    w0 = tl.load(conv_w_ptr + c * 4 + 0).to(tl.float32)
    w1 = tl.load(conv_w_ptr + c * 4 + 1).to(tl.float32)
    w2 = tl.load(conv_w_ptr + c * 4 + 2).to(tl.float32)
    w3 = tl.load(conv_w_ptr + c * 4 + 3).to(tl.float32)

    for row in range(q_start, q_end):
        x = tl.load(qkvzba_ptr + row * stride_in_row + c).to(tl.float32)
        acc = w0 * s0 + w1 * s1 + w2 * s2 + w3 * x
        y = acc * tl.sigmoid(acc)
        tl.store(qkv_out_ptr + row * 1024 + c, y.to(tl.bfloat16))
        s0 = s1
        s1 = s2
        s2 = x

    tl.store(state_base + 0, s0.to(tl.bfloat16))
    tl.store(state_base + 1, s1.to(tl.bfloat16))
    tl.store(state_base + 2, s2.to(tl.bfloat16))


@triton.jit
def _gated_delta_decode_kernel(
    qkv_conv_ptr,
    qkvzba_ptr,
    a_log_ptr,
    dt_bias_ptr,
    ssm_state_ptr,
    positions_ptr,
    req_slot_ids_ptr,
    attn_out_ptr,
    stride_qkvzba_row,
    stride_ssm_slot,
    stride_ssm_head,
    num_active_rows,
    BLOCK_V: tl.constexpr,
):
    # qkv_conv row (1024): [0:256] Q (2 heads * 128), [256:512] K (2 heads * 128), [512:1024] V (4 heads * 128)
    # qkvzba row (1544):   [1024:1536] Z (4 heads * 128), [1536:1540] B (4), [1540:1544] A (4)
    row = tl.program_id(0).to(tl.int64)
    hv_tile = tl.program_id(1).to(tl.int64)
    v_head = hv_tile // (128 // BLOCK_V)
    v_blk = hv_tile % (128 // BLOCK_V)
    k_head = v_head // 2

    offs_k = tl.arange(0, 128)
    offs_v = v_blk * BLOCK_V + tl.arange(0, BLOCK_V)

    if row >= num_active_rows:
        tl.store(attn_out_ptr + row * 512 + v_head * 128 + offs_v, tl.zeros([BLOCK_V], dtype=tl.bfloat16))
        return

    slot = tl.load(req_slot_ids_ptr + row).to(tl.int64)
    pos = tl.load(positions_ptr + row).to(tl.int32)

    q = tl.load(qkv_conv_ptr + row * 1024 + k_head * 128 + offs_k).to(tl.float32)
    k = tl.load(qkv_conv_ptr + row * 1024 + 256 + k_head * 128 + offs_k).to(tl.float32)
    v = tl.load(qkv_conv_ptr + row * 1024 + 512 + v_head * 128 + offs_v).to(tl.float32)

    q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1.0e-6) * 0.08838834764831845  # 128 ** -0.5
    k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1.0e-6)

    b_raw = tl.load(qkvzba_ptr + row * stride_qkvzba_row + 1536 + v_head).to(tl.float32)
    a_raw = tl.load(qkvzba_ptr + row * stride_qkvzba_row + 1540 + v_head).to(tl.float32)
    a_log = tl.load(a_log_ptr + v_head).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + v_head).to(tl.float32)

    beta = tl.sigmoid(b_raw)
    sp_in = a_raw + dt_bias
    softplus = tl.where(sp_in > 20.0, sp_in, tl.log(1.0 + tl.exp(sp_in)))
    decay = tl.exp(-tl.exp(a_log) * softplus)

    ssm_ptrs = (
        ssm_state_ptr
        + slot * stride_ssm_slot
        + v_head * stride_ssm_head
        + offs_k[:, None] * 128
        + offs_v[None, :]
    )
    if pos == 0:
        state = tl.zeros([128, BLOCK_V], dtype=tl.float32)
    else:
        state = tl.load(ssm_ptrs)

    state = state * decay
    kv_mem = tl.sum(state * k[:, None], axis=0)
    delta = (v - kv_mem) * beta
    state = state + k[:, None] * delta[None, :]
    out_v = tl.sum(state * q[:, None], axis=0)

    tl.store(ssm_ptrs, state)
    tl.store(attn_out_ptr + row * 512 + v_head * 128 + offs_v, out_v.to(tl.bfloat16))


@triton.jit
def _gated_delta_prefill_varlen_kernel(
    qkv_conv_ptr,
    qkvzba_ptr,
    a_log_ptr,
    dt_bias_ptr,
    ssm_state_ptr,
    query_start_loc_ptr,
    positions_ptr,
    req_slot_ids_ptr,
    attn_out_ptr,
    stride_qkvzba_row,
    stride_ssm_slot,
    stride_ssm_head,
    BLOCK_V: tl.constexpr,
):
    seq_idx = tl.program_id(0).to(tl.int64)
    hv_tile = tl.program_id(1).to(tl.int64)
    v_head = hv_tile // (128 // BLOCK_V)
    v_blk = hv_tile % (128 // BLOCK_V)
    k_head = v_head // 2

    q_start = tl.load(query_start_loc_ptr + seq_idx).to(tl.int64)
    q_end = tl.load(query_start_loc_ptr + seq_idx + 1).to(tl.int64)
    if q_start >= q_end:
        return

    slot = tl.load(req_slot_ids_ptr + q_start).to(tl.int64)
    start_pos = tl.load(positions_ptr + q_start).to(tl.int32)
    offs_k = tl.arange(0, 128)
    offs_v = v_blk * BLOCK_V + tl.arange(0, BLOCK_V)

    ssm_ptrs = (
        ssm_state_ptr
        + slot * stride_ssm_slot
        + v_head * stride_ssm_head
        + offs_k[:, None] * 128
        + offs_v[None, :]
    )
    if start_pos == 0:
        state = tl.zeros([128, BLOCK_V], dtype=tl.float32)
    else:
        state = tl.load(ssm_ptrs)

    neg_a = -tl.exp(tl.load(a_log_ptr + v_head).to(tl.float32))
    dt_bias = tl.load(dt_bias_ptr + v_head).to(tl.float32)

    for row in range(q_start, q_end):
        q = tl.load(qkv_conv_ptr + row * 1024 + k_head * 128 + offs_k).to(tl.float32)
        k = tl.load(qkv_conv_ptr + row * 1024 + 256 + k_head * 128 + offs_k).to(tl.float32)
        v = tl.load(qkv_conv_ptr + row * 1024 + 512 + v_head * 128 + offs_v).to(tl.float32)

        q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1.0e-6) * 0.08838834764831845
        k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1.0e-6)

        b_raw = tl.load(qkvzba_ptr + row * stride_qkvzba_row + 1536 + v_head).to(tl.float32)
        a_raw = tl.load(qkvzba_ptr + row * stride_qkvzba_row + 1540 + v_head).to(tl.float32)
        beta = tl.sigmoid(b_raw)
        sp_in = a_raw + dt_bias
        softplus = tl.where(sp_in > 20.0, sp_in, tl.log(1.0 + tl.exp(sp_in)))
        decay = tl.exp(neg_a * softplus)

        state = state * decay
        kv_mem = tl.sum(state * k[:, None], axis=0)
        delta = (v - kv_mem) * beta
        state = state + k[:, None] * delta[None, :]
        out_v = tl.sum(state * q[:, None], axis=0)
        tl.store(attn_out_ptr + row * 512 + v_head * 128 + offs_v, out_v.to(tl.bfloat16))

    tl.store(ssm_ptrs, state)


@triton.jit
def _rmsnorm_gated_silu_kernel(
    x_ptr,
    qkvzba_ptr,
    w_ptr,
    out_ptr,
    stride_qkvzba_row,
    eps,
    num_active_rows,
):
    row = tl.program_id(0).to(tl.int64)
    v_head = tl.program_id(1).to(tl.int64)
    d = tl.arange(0, 128)

    if row >= num_active_rows:
        tl.store(out_ptr + row * 512 + v_head * 128 + d, tl.zeros([128], dtype=tl.bfloat16))
        return

    x = tl.load(x_ptr + row * 512 + v_head * 128 + d).to(tl.float32)
    z = tl.load(qkvzba_ptr + row * stride_qkvzba_row + 1024 + v_head * 128 + d).to(tl.float32)
    w = tl.load(w_ptr + d).to(tl.float32)

    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) * (1.0 / 128.0) + eps)
    normed = (x * inv_rms).to(tl.bfloat16).to(tl.float32) * w
    silu_z = z * tl.sigmoid(z)
    tl.store(out_ptr + row * 512 + v_head * 128 + d, (normed * silu_z).to(tl.bfloat16))


def linear_attn_decode_step(
    qkvzba: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    norm_weight: torch.Tensor,
    positions: torch.Tensor,
    req_slot_ids: torch.Tensor,
    *,
    qkv_conv_buf: torch.Tensor,
    raw_out_buf: torch.Tensor,
    gated_out_buf: torch.Tensor,
    num_active_rows: int,
    eps: float,
) -> torch.Tensor:
    """Fused decode step for Qwen3_5MoeGatedDeltaNet (TP=8: 2 K-heads, 4 V-heads)."""
    batch_rows = qkvzba.shape[0]
    _causal_conv1d_decode_kernel[(batch_rows, 1024 // 128)](
        qkvzba,
        conv_state,
        conv_weight,
        positions,
        req_slot_ids,
        qkv_conv_buf,
        qkvzba.stride(0),
        conv_state.stride(0),
        num_active_rows,
        BLOCK_C=128,
        num_warps=4,
    )
    _gated_delta_decode_kernel[(batch_rows, 4 * (128 // 16))](
        qkv_conv_buf,
        qkvzba,
        a_log,
        dt_bias,
        ssm_state,
        positions,
        req_slot_ids,
        raw_out_buf,
        qkvzba.stride(0),
        ssm_state.stride(0),
        ssm_state.stride(1),
        num_active_rows,
        BLOCK_V=16,
        num_warps=4,
    )
    _rmsnorm_gated_silu_kernel[(batch_rows, 4)](
        raw_out_buf,
        qkvzba,
        norm_weight,
        gated_out_buf,
        qkvzba.stride(0),
        eps,
        num_active_rows,
        num_warps=4,
    )
    return gated_out_buf


def linear_attn_prefill_varlen(
    qkvzba: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    norm_weight: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    req_slot_ids: torch.Tensor,
    *,
    qkv_conv_buf: torch.Tensor,
    raw_out_buf: torch.Tensor,
    gated_out_buf: torch.Tensor,
    num_seqs: int,
    eps: float,
) -> torch.Tensor:
    """Varlen prefill for Qwen3_5MoeGatedDeltaNet keeping SSM tiles in registers across tokens."""
    total_rows = qkvzba.shape[0]
    _causal_conv1d_prefill_varlen_kernel[(num_seqs, 1024 // 32)](
        qkvzba,
        conv_state,
        conv_weight,
        query_start_loc,
        positions,
        req_slot_ids,
        qkv_conv_buf,
        qkvzba.stride(0),
        conv_state.stride(0),
        BLOCK_C=32,
        num_warps=2,
    )
    _gated_delta_prefill_varlen_kernel[(num_seqs, 4 * (128 // 4))](
        qkv_conv_buf,
        qkvzba,
        a_log,
        dt_bias,
        ssm_state,
        query_start_loc,
        positions,
        req_slot_ids,
        raw_out_buf,
        qkvzba.stride(0),
        ssm_state.stride(0),
        ssm_state.stride(1),
        BLOCK_V=4,
        num_warps=2,
    )
    _rmsnorm_gated_silu_kernel[(total_rows, 4)](
        raw_out_buf,
        qkvzba,
        norm_weight,
        gated_out_buf,
        qkvzba.stride(0),
        eps,
        total_rows,
        num_warps=4,
    )
    return gated_out_buf


# ---------------------------------------------------------------------------
# 5. Sparse MoE (256 experts, top-8 softmax router + shared expert, TP=8 inter=64)
# ---------------------------------------------------------------------------


@triton.jit
def _router_softmax256_topk8_and_shared_act_kernel(
    router_shared_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    shared_gate_scale_ptr,
    shared_inter_ptr,
    stride_rs_row,
    num_active_rows,
):
    # router_shared row (392 cols):
    # [0:256]   router logits
    # [256]     shared_expert_gate logit (padded to [256:264])
    # [264:328] shared_expert gate_proj (64 cols)
    # [328:392] shared_expert up_proj   (64 cols)
    row = tl.program_id(0).to(tl.int64)
    e = tl.arange(0, 256)
    c64 = tl.arange(0, 64)

    if row >= num_active_rows:
        for k in tl.static_range(0, 8):
            tl.store(topk_ids_ptr + row * 8 + k, 256)
            tl.store(topk_weights_ptr + row * 8 + k, tl.zeros([], dtype=tl.bfloat16))
        tl.store(shared_gate_scale_ptr + row, 0.0)
        tl.store(shared_inter_ptr + row * 64 + c64, tl.zeros([64], dtype=tl.bfloat16))
        return

    base = router_shared_ptr + row * stride_rs_row
    logits = tl.load(base + e).to(tl.float32)
    max_l = tl.max(logits, axis=0)
    exp_l = tl.exp(logits - max_l)
    probs = exp_l / tl.sum(exp_l, axis=0)

    # Extract exact top-8 in registers with deterministic smallest-index tie-break matching torch.topk
    k8 = tl.arange(0, 8)
    top_vals = tl.zeros([8], dtype=tl.float32)
    top_ids = tl.full([8], 256, dtype=tl.int32)
    cur = probs
    w_sum = 0.0
    for k in tl.static_range(0, 8):
        best_val = tl.max(cur, axis=0)
        is_best = cur == best_val
        best_idx = tl.min(tl.where(is_best, e, 256), axis=0).to(tl.int32)
        top_vals = tl.where(k8 == k, best_val, top_vals)
        top_ids = tl.where(k8 == k, best_idx, top_ids)
        w_sum += best_val
        cur = tl.where(e == best_idx, -1.0, cur)

    w_denom = tl.where(w_sum > 0.0, w_sum, 1.0)
    tl.store(topk_ids_ptr + row * 8 + k8, top_ids)
    tl.store(topk_weights_ptr + row * 8 + k8, (top_vals / w_denom).to(tl.bfloat16))

    sg_logit = tl.load(base + 256).to(tl.float32)
    tl.store(shared_gate_scale_ptr + row, tl.sigmoid(sg_logit))

    sh_gate = tl.load(base + 264 + c64).to(tl.float32)
    sh_up = tl.load(base + 328 + c64).to(tl.float32)
    sh_act = (sh_gate * tl.sigmoid(sh_gate)) * sh_up
    tl.store(shared_inter_ptr + row * 64 + c64, sh_act.to(tl.bfloat16))


@triton.jit
def _fused_moe_gate_up_silu_kernel(
    x_ptr,
    w_gate_up_ptr,
    sorted_pair_ids_ptr,
    expert_ids_per_tile_ptr,
    num_valid_tiles_ptr,
    moe_inter_ptr,
    stride_x_row,
    stride_we,
    stride_wn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_id = tl.program_id(0).to(tl.int64)
    col_blk = tl.program_id(1).to(tl.int64)

    num_valid = tl.load(num_valid_tiles_ptr).to(tl.int64)
    if tile_id >= num_valid:
        return

    expert_id = tl.load(expert_ids_per_tile_ptr + tile_id).to(tl.int64)
    offs_m = tl.arange(0, BLOCK_M)
    pair_ids = tl.load(sorted_pair_ids_ptr + tile_id * BLOCK_M + offs_m).to(tl.int64)
    valid_m = pair_ids >= 0
    tok_ids = tl.where(valid_m, pair_ids // 8, 0)

    offs_n = col_blk * BLOCK_N + tl.arange(0, BLOCK_N)
    w_expert_base = w_gate_up_ptr + expert_id * stride_we

    acc_gate = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc_up = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_off in range(0, 2048, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x_ptr + tok_ids[:, None] * stride_x_row + offs_k[None, :],
            mask=valid_m[:, None],
            other=0.0,
        )
        wg_tile = tl.load(w_expert_base + offs_n[:, None] * stride_wn + offs_k[None, :])
        wu_tile = tl.load(w_expert_base + (64 + offs_n[:, None]) * stride_wn + offs_k[None, :])
        acc_gate += tl.dot(x_tile, tl.trans(wg_tile))
        acc_up += tl.dot(x_tile, tl.trans(wu_tile))

    act = (acc_gate * tl.sigmoid(acc_gate)) * acc_up
    tl.store(
        moe_inter_ptr + pair_ids[:, None] * 64 + offs_n[None, :],
        act.to(tl.bfloat16),
        mask=valid_m[:, None],
    )


@triton.jit
def _fused_moe_down_kernel(
    moe_inter_ptr,
    w_down_ptr,
    topk_weights_ptr,
    sorted_pair_ids_ptr,
    expert_ids_per_tile_ptr,
    num_valid_tiles_ptr,
    pair_out_ptr,
    stride_we,
    stride_wn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile_id = tl.program_id(0).to(tl.int64)
    n_blk = tl.program_id(1).to(tl.int64)

    num_valid = tl.load(num_valid_tiles_ptr).to(tl.int64)
    if tile_id >= num_valid:
        return

    expert_id = tl.load(expert_ids_per_tile_ptr + tile_id).to(tl.int64)
    offs_m = tl.arange(0, BLOCK_M)
    pair_ids = tl.load(sorted_pair_ids_ptr + tile_id * BLOCK_M + offs_m).to(tl.int64)
    valid_m = pair_ids >= 0
    safe_pairs = tl.where(valid_m, pair_ids, 0)

    offs_k = tl.arange(0, 64)
    offs_n = n_blk * BLOCK_N + tl.arange(0, BLOCK_N)

    inter_tile = tl.load(
        moe_inter_ptr + safe_pairs[:, None] * 64 + offs_k[None, :],
        mask=valid_m[:, None],
        other=0.0,
    )
    wd_tile = tl.load(
        w_down_ptr + expert_id * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :]
    )
    weights = tl.load(topk_weights_ptr + safe_pairs, mask=valid_m, other=0.0).to(tl.float32)

    out_tile = tl.dot(inter_tile, tl.trans(wd_tile)) * weights[:, None]
    tl.store(
        pair_out_ptr + safe_pairs[:, None] * 2048 + offs_n[None, :],
        out_tile.to(tl.bfloat16),
        mask=valid_m[:, None],
    )


@triton.jit
def _moe_combine_top8_and_shared_kernel(
    pair_out_ptr,
    shared_out_ptr,
    shared_gate_scale_ptr,
    out_ptr,
    num_active_rows,
    BLOCK_COL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col_blk = tl.program_id(1).to(tl.int64)
    cols = col_blk * BLOCK_COL + tl.arange(0, BLOCK_COL)

    if row >= num_active_rows:
        tl.store(out_ptr + row * 2048 + cols, tl.zeros([BLOCK_COL], dtype=tl.bfloat16))
        return

    acc = tl.zeros([BLOCK_COL], dtype=tl.float32)
    pair_base = pair_out_ptr + row * 8 * 2048 + cols
    for k in tl.static_range(0, 8):
        acc += tl.load(pair_base + k * 2048).to(tl.float32)

    sh_scale = tl.load(shared_gate_scale_ptr + row).to(tl.float32)
    sh_val = tl.load(shared_out_ptr + row * 2048 + cols).to(tl.float32)
    total = acc + sh_scale * sh_val
    tl.store(out_ptr + row * 2048 + cols, total.to(tl.bfloat16))


@triton.jit
def _moe_gate_up_silu_gemv_kernel(
    x_ptr,
    w_gate_up_ptr,
    topk_ids_ptr,
    moe_inter_ptr,
    stride_x_row,
    stride_we,
    stride_wn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pair = tl.program_id(0).to(tl.int64)
    n_blk = tl.program_id(1).to(tl.int64)
    tok = pair // 8
    expert_id = tl.load(topk_ids_ptr + pair).to(tl.int64)
    offs_n = n_blk * BLOCK_N + tl.arange(0, BLOCK_N)

    # Inactive rows carry the sentinel expert id 256 from the router kernel.
    if expert_id >= 256:
        tl.store(moe_inter_ptr + pair * 64 + offs_n, tl.zeros([BLOCK_N], dtype=tl.bfloat16))
        return

    w_expert_base = w_gate_up_ptr + expert_id * stride_we
    acc_gate = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
    acc_up = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
    for k_off in range(0, 2048, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + tok * stride_x_row + offs_k).to(tl.float32)
        wg = tl.load(w_expert_base + offs_n[:, None] * stride_wn + offs_k[None, :]).to(tl.float32)
        wu = tl.load(w_expert_base + (64 + offs_n[:, None]) * stride_wn + offs_k[None, :]).to(
            tl.float32
        )
        acc_gate += wg * x[None, :]
        acc_up += wu * x[None, :]
    gate = tl.sum(acc_gate, 1)
    up = tl.sum(acc_up, 1)
    act = (gate * tl.sigmoid(gate)) * up
    tl.store(moe_inter_ptr + pair * 64 + offs_n, act.to(tl.bfloat16))


@triton.jit
def _moe_down_combine_gemv_kernel(
    moe_inter_ptr,
    w_down_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    shared_out_ptr,
    shared_gate_scale_ptr,
    out_ptr,
    stride_we,
    stride_wn,
    num_active_rows,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    n_blk = tl.program_id(1).to(tl.int64)
    offs_n = n_blk * BLOCK_N + tl.arange(0, BLOCK_N)

    if row >= num_active_rows:
        tl.store(out_ptr + row * 2048 + offs_n, tl.zeros([BLOCK_N], dtype=tl.bfloat16))
        return

    offs_k = tl.arange(0, 64)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k in tl.static_range(0, 8):
        pair = row * 8 + k
        expert_id = tl.load(topk_ids_ptr + pair).to(tl.int64)
        valid = expert_id < 256
        safe_expert = tl.where(valid, expert_id, 0)
        weight = tl.where(valid, tl.load(topk_weights_ptr + pair).to(tl.float32), 0.0)
        inter = tl.load(moe_inter_ptr + pair * 64 + offs_k).to(tl.float32)
        wd = tl.load(
            w_down_ptr + safe_expert * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :]
        ).to(tl.float32)
        acc += weight * tl.sum(wd * inter[None, :], 1)

    sh_scale = tl.load(shared_gate_scale_ptr + row).to(tl.float32)
    sh_val = tl.load(shared_out_ptr + row * 2048 + offs_n).to(tl.float32)
    tl.store(out_ptr + row * 2048 + offs_n, (acc + sh_scale * sh_val).to(tl.bfloat16))


@triton.jit
def _skinny_linear_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    num_rows,
    n_out,
    stride_x_row,
    stride_w_row,
    stride_out_row,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    n_blk = tl.program_id(0).to(tl.int64)
    offs_m = tl.arange(0, 16)
    offs_n = n_blk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < num_rows
    mask_n = offs_n < n_out
    acc = tl.zeros([16, BLOCK_N], dtype=tl.float32)
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x_ptr + offs_m[:, None] * stride_x_row + offs_k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        w_tile = tl.load(
            w_ptr + offs_n[:, None] * stride_w_row + offs_k[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )
        acc += tl.dot(x_tile, tl.trans(w_tile))
    tl.store(
        out_ptr + offs_m[:, None] * stride_out_row + offs_n[None, :],
        acc.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


SKINNY_LINEAR_MAX_ROWS = 16
MOE_GEMV_MAX_ROWS = 2


def skinny_linear(x: torch.Tensor, w: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    """`out = x @ w.T` for decode-sized row counts (CUDA-Graph-safe).

    On sm_120 cuBLAS routes 2 <= M <= 16 BF16 GEMMs with K=2048 to sm80 wmma 16x16 tiles
    that run ~2x slower than its own M=1 GEMV; a single 16-row tile keeps these
    projections memory-bound for those buckets. M=1 stays on the cuBLAS GEMV: it is only
    slightly slower and keeps single-sequence greedy decoding within the 2-ULP vLLM
    parity gate of `test_qwen35_e2e_tp8.py`.
    """
    rows = x.shape[0]
    n_out, k = w.shape
    if rows == 1 or rows > SKINNY_LINEAR_MAX_ROWS or k % 256 != 0:
        return torch.matmul(x, w.t(), out=out)
    _skinny_linear_kernel[(triton.cdiv(n_out, 16),)](
        x,
        w,
        out,
        rows,
        n_out,
        x.stride(0),
        w.stride(0),
        out.stride(0),
        K=k,
        BLOCK_N=16,
        BLOCK_K=256,
        num_warps=4,
        num_stages=3,
    )
    return out


def max_moe_tiles_for_rows(rows: int) -> int:
    """Deterministic upper bound on non-empty 16-token expert tiles for `rows * 8` pairs."""
    pairs = rows * 8
    return min(pairs, (pairs // 16) + 256)


def sparse_moe_tp8_forward(
    normed_x: torch.Tensor,
    w_router_shared_gu: torch.Tensor,
    w_shared_down: torch.Tensor,
    w_experts_gate_up: torch.Tensor,
    w_experts_down: torch.Tensor,
    *,
    router_shared_buf: torch.Tensor,
    topk_ids_buf: torch.Tensor,
    topk_weights_buf: torch.Tensor,
    shared_gate_scale_buf: torch.Tensor,
    shared_inter_buf: torch.Tensor,
    shared_out_buf: torch.Tensor,
    sorted_pair_ids_buf: torch.Tensor,
    expert_ids_per_tile_buf: torch.Tensor,
    num_valid_tiles_buf: torch.Tensor,
    moe_inter_buf: torch.Tensor,
    pair_out_buf: torch.Tensor,
    moe_out_buf: torch.Tensor,
    num_active_rows: int,
) -> torch.Tensor:
    """Execute Qwen3_5MoeSparseMoeBlock local TP=8 shard (CUDA-Graph-safe, zero CPU sync)."""
    rows = normed_x.shape[0]
    # cuBLAS is already near-optimal for this N=392 projection except at M=2.
    if rows == 2:
        skinny_linear(normed_x, w_router_shared_gu, out=router_shared_buf[:rows])
    else:
        torch.matmul(normed_x, w_router_shared_gu.t(), out=router_shared_buf[:rows])
    _router_softmax256_topk8_and_shared_act_kernel[(rows,)](
        router_shared_buf,
        topk_ids_buf,
        topk_weights_buf,
        shared_gate_scale_buf,
        shared_inter_buf,
        router_shared_buf.stride(0),
        num_active_rows,
        num_warps=4,
    )
    torch.matmul(shared_inter_buf[:rows], w_shared_down.t(), out=shared_out_buf[:rows])

    if rows <= MOE_GEMV_MAX_ROWS:
        # With <= 16 routed pairs every 16-row expert tile holds one valid row, so the tiled
        # path wastes most of each tile and launches too few CTAs; per-pair GEMVs skip the
        # align pass and fuse down-projection with the top-8 + shared-expert combine.
        _moe_gate_up_silu_gemv_kernel[(rows * 8, 64 // 4)](
            normed_x,
            w_experts_gate_up,
            topk_ids_buf,
            moe_inter_buf,
            normed_x.stride(0),
            w_experts_gate_up.stride(0),
            w_experts_gate_up.stride(1),
            BLOCK_N=4,
            BLOCK_K=512,
            num_warps=4,
        )
        _moe_down_combine_gemv_kernel[(rows, 2048 // 32)](
            moe_inter_buf,
            w_experts_down,
            topk_ids_buf,
            topk_weights_buf,
            shared_out_buf,
            shared_gate_scale_buf,
            moe_out_buf,
            w_experts_down.stride(0),
            w_experts_down.stride(1),
            num_active_rows,
            BLOCK_N=32,
            num_warps=2,
        )
        return moe_out_buf[:rows]

    max_tiles = max_moe_tiles_for_rows(rows)
    ext = get_qwen35_cuda_ext()
    stream_handle = torch.cuda.current_stream(normed_x.device).cuda_stream
    ext.launch_moe_align_block_size(
        topk_ids_buf[: rows * 8],
        sorted_pair_ids_buf[: max_tiles * 16],
        expert_ids_per_tile_buf[:max_tiles],
        num_valid_tiles_buf,
        rows * 8,
        stream_handle,
    )
    _fused_moe_gate_up_silu_kernel[(max_tiles, 64 // 16)](
        normed_x,
        w_experts_gate_up,
        sorted_pair_ids_buf,
        expert_ids_per_tile_buf,
        num_valid_tiles_buf,
        moe_inter_buf,
        normed_x.stride(0),
        w_experts_gate_up.stride(0),
        w_experts_gate_up.stride(1),
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_K=64,
        num_warps=4,
    )
    _fused_moe_down_kernel[(max_tiles, 2048 // 64)](
        moe_inter_buf,
        w_experts_down,
        topk_weights_buf,
        sorted_pair_ids_buf,
        expert_ids_per_tile_buf,
        num_valid_tiles_buf,
        pair_out_buf,
        w_experts_down.stride(0),
        w_experts_down.stride(1),
        BLOCK_M=16,
        BLOCK_N=64,
        num_warps=4,
    )
    _moe_combine_top8_and_shared_kernel[(rows, 2048 // 256)](
        pair_out_buf,
        shared_out_buf,
        shared_gate_scale_buf,
        moe_out_buf,
        num_active_rows,
        BLOCK_COL=256,
        num_warps=4,
    )
    return moe_out_buf[:rows]
