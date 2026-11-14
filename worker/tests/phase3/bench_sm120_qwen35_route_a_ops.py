#!/usr/bin/env python3
# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT
# ruff: noqa: B023
"""Route A comprehensive SM120 microbenchmark + numerical parity suite on 8x RTX 6000D.

Tests and benchmarks:
1. Part A1 (8-GPU TP=8):
   - Baseline 2-op: `shm_v5_allreduce` + `_fused_add_zero_centered_rmsnorm_kernel`
   - Batched-load 2-op: `shm_v5_allreduce` (non-interleaved peer loads) + RMSNorm
   - Fused 1-op: `shm_v5_allreduce_add_rmsnorm` (Stage-2 epilogue fusion, 0-ULP residual)
   Across H=2048 (Qwen3.5) and H=5120 (DSV4.1), CUDAGraph M=1..256 & Eager M=512.
2. Part A2 (Single-GPU SM120 156-SM, 30-layer GatedDeltaNet step):
   - Decode (CUDAGraph B=1,2,4,8,16,32,64,128):
     * Baseline Triton 3-op (`[K,V]` state, 90 launches/step)
     * CUDA C++ Fused 1-CTA (`[V,K]` float4 coalesced, 30 launches/step)
     * CUDA C++ Fused Cluster-4 DSMEM (`__cluster_dims__(4,1,1)`, `[V,K]`, 30 launches/step)
     * CUDA C++ Fused Cluster-8 DSMEM (`__cluster_dims__(8,1,1)`, `[V,K]`, 30 launches/step)
     * b12x.sequence.gdn_decode (if H_k=2, H_v=4 supported)
   - Varlen Prefill (`1x24`, `1x64`, `1x512`, `1x1024`, `1x2048`, `8x512`, `16x256`):
     * Baseline Triton 3-op (`[K,V]` BLOCK_V=4)
     * CUDA C++ `[V,K]` Coalesced Recurrent (3-op / 2-op with fused prologue)
     * FlashInfer `gdn_sm120_cute` (`chunk_gated_delta_rule`, `state_v_first=True`) + fused Conv1d + GatedRMSNorm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.cpp_extension import load_inline
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Inline CUDA C++ Prototype Extension for Route A (A1 + A2)
# ---------------------------------------------------------------------------

_ROUTE_A_CPP = r"""
#include <torch/extension.h>
#include <cstdint>

void launch_shm_v5_ar_legacy_interleaved_cuda(
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

void launch_shm_v5_ar_batchload_cuda(
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

void launch_shm_v5_ar_add_rmsnorm_cuda(
    const void* input_ptr,
    void* residual_ptr,
    const void* weight_ptr,
    void* normed_out_ptr,
    uint64_t* control_ptr,
    const std::uintptr_t* rank_shm_dev_ptrs,
    int64_t slot_stride_bytes,
    int64_t rows,
    int64_t vecs_per_row,
    float inv_hidden,
    float eps,
    float weight_offset,
    int rank,
    int world_size,
    int blocks,
    int threads,
    int one_shot_max_vecs,
    int use_cluster,
    std::uintptr_t stream_handle);

void launch_gdn_decode_fused_vk_cuda(
    const void* qkvzba_ptr,
    void* conv_state_ptr,
    const void* conv_w_ptr,
    const float* a_log_ptr,
    const void* dt_bias_ptr,
    float* ssm_state_vk_ptr,
    const float* norm_w_ptr,
    const int32_t* positions_ptr,
    const int32_t* req_slot_ids_ptr,
    void* gated_out_ptr,
    int64_t stride_qkvzba_row,
    int64_t stride_conv_slot,
    int64_t stride_ssm_slot,
    int64_t stride_ssm_head,
    int num_rows,
    int num_active_rows,
    float eps,
    int variant,
    std::uintptr_t stream_handle);

void launch_gdn_decode_split_vk_cuda(
    const void* qkvzba_ptr,
    void* conv_state_ptr,
    const void* conv_w_ptr,
    const float* a_log_ptr,
    const void* dt_bias_ptr,
    float* ssm_state_vk_ptr,
    const float* norm_w_ptr,
    const int32_t* positions_ptr,
    const int32_t* req_slot_ids_ptr,
    float* qk_norm_scratch_ptr,
    void* v_conv_scratch_ptr,
    void* raw_out_ptr,
    void* gated_out_ptr,
    int64_t stride_qkvzba_row,
    int64_t stride_conv_slot,
    int64_t stride_ssm_slot,
    int64_t stride_ssm_head,
    int num_rows,
    int num_active_rows,
    float eps,
    int mode,
    std::uintptr_t stream_handle);

void launch_gdn_prefill_vk_cuda(
    const void* qkvzba_ptr,
    void* conv_state_ptr,
    const void* conv_w_ptr,
    const float* a_log_ptr,
    const void* dt_bias_ptr,
    float* ssm_state_vk_ptr,
    const float* norm_w_ptr,
    const int32_t* query_start_loc_ptr,
    const int32_t* positions_ptr,
    const int32_t* req_slot_ids_ptr,
    float* qk_norm_scratch_ptr,
    void* v_conv_scratch_ptr,
    void* raw_out_ptr,
    void* gated_out_ptr,
    int64_t stride_qkvzba_row,
    int64_t stride_conv_slot,
    int64_t stride_ssm_slot,
    int64_t stride_ssm_head,
    int num_seqs,
    int total_rows,
    float eps,
    std::uintptr_t stream_handle);

void launch_shm_v5_ar_legacy_interleaved(
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
    launch_shm_v5_ar_legacy_interleaved_cuda(
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

void launch_shm_v5_ar_batchload(
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
    launch_shm_v5_ar_batchload_cuda(
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

void launch_shm_v5_ar_add_rmsnorm(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor normed_out,
    torch::Tensor control,
    torch::Tensor rank_shm_ptrs_u64,
    int64_t slot_stride_bytes,
    double eps,
    double weight_offset,
    int64_t rank,
    int64_t world_size,
    int64_t blocks,
    int64_t threads,
    int64_t one_shot_max_vecs,
    int64_t use_cluster,
    std::uintptr_t stream_handle) {
    const int64_t rows = input.size(0);
    const int64_t hidden = input.size(1);
    launch_shm_v5_ar_add_rmsnorm_cuda(
        input.data_ptr(),
        residual.data_ptr(),
        weight.data_ptr(),
        normed_out.data_ptr(),
        reinterpret_cast<uint64_t*>(control.data_ptr<int64_t>()),
        reinterpret_cast<const std::uintptr_t*>(rank_shm_ptrs_u64.data_ptr<int64_t>()),
        slot_stride_bytes,
        rows,
        hidden / 8,
        1.0f / static_cast<float>(hidden),
        static_cast<float>(eps),
        static_cast<float>(weight_offset),
        static_cast<int>(rank),
        static_cast<int>(world_size),
        static_cast<int>(blocks),
        static_cast<int>(threads),
        static_cast<int>(one_shot_max_vecs),
        static_cast<int>(use_cluster),
        stream_handle);
}

void launch_gdn_decode_fused_vk(
    torch::Tensor qkvzba,
    torch::Tensor conv_state,
    torch::Tensor conv_w,
    torch::Tensor a_log,
    torch::Tensor dt_bias,
    torch::Tensor ssm_state_vk,
    torch::Tensor norm_w,
    torch::Tensor positions,
    torch::Tensor req_slot_ids,
    torch::Tensor gated_out,
    int64_t num_active_rows,
    double eps,
    int64_t variant,
    std::uintptr_t stream_handle) {
    launch_gdn_decode_fused_vk_cuda(
        qkvzba.data_ptr(),
        conv_state.data_ptr(),
        conv_w.data_ptr(),
        a_log.data_ptr<float>(),
        dt_bias.data_ptr(),
        ssm_state_vk.data_ptr<float>(),
        norm_w.data_ptr<float>(),
        positions.data_ptr<int32_t>(),
        req_slot_ids.data_ptr<int32_t>(),
        gated_out.data_ptr(),
        qkvzba.stride(0),
        conv_state.stride(0),
        ssm_state_vk.stride(0),
        ssm_state_vk.stride(1),
        static_cast<int>(qkvzba.size(0)),
        static_cast<int>(num_active_rows),
        static_cast<float>(eps),
        static_cast<int>(variant),
        stream_handle);
}

void launch_gdn_decode_split_vk(
    torch::Tensor qkvzba,
    torch::Tensor conv_state,
    torch::Tensor conv_w,
    torch::Tensor a_log,
    torch::Tensor dt_bias,
    torch::Tensor ssm_state_vk,
    torch::Tensor norm_w,
    torch::Tensor positions,
    torch::Tensor req_slot_ids,
    torch::Tensor qk_norm_scratch,
    torch::Tensor v_conv_scratch,
    torch::Tensor raw_out,
    torch::Tensor gated_out,
    int64_t num_active_rows,
    double eps,
    int64_t mode,
    std::uintptr_t stream_handle) {
    launch_gdn_decode_split_vk_cuda(
        qkvzba.data_ptr(),
        conv_state.data_ptr(),
        conv_w.data_ptr(),
        a_log.data_ptr<float>(),
        dt_bias.data_ptr(),
        ssm_state_vk.data_ptr<float>(),
        norm_w.data_ptr<float>(),
        positions.data_ptr<int32_t>(),
        req_slot_ids.data_ptr<int32_t>(),
        qk_norm_scratch.data_ptr<float>(),
        v_conv_scratch.data_ptr(),
        raw_out.data_ptr(),
        gated_out.data_ptr(),
        qkvzba.stride(0),
        conv_state.stride(0),
        ssm_state_vk.stride(0),
        ssm_state_vk.stride(1),
        static_cast<int>(qkvzba.size(0)),
        static_cast<int>(num_active_rows),
        static_cast<float>(eps),
        static_cast<int>(mode),
        stream_handle);
}

void launch_gdn_prefill_vk(
    torch::Tensor qkvzba,
    torch::Tensor conv_state,
    torch::Tensor conv_w,
    torch::Tensor a_log,
    torch::Tensor dt_bias,
    torch::Tensor ssm_state_vk,
    torch::Tensor norm_w,
    torch::Tensor query_start_loc,
    torch::Tensor positions,
    torch::Tensor req_slot_ids,
    torch::Tensor qk_norm_scratch,
    torch::Tensor v_conv_scratch,
    torch::Tensor raw_out,
    torch::Tensor gated_out,
    int64_t num_seqs,
    double eps,
    std::uintptr_t stream_handle) {
    launch_gdn_prefill_vk_cuda(
        qkvzba.data_ptr(),
        conv_state.data_ptr(),
        conv_w.data_ptr(),
        a_log.data_ptr<float>(),
        dt_bias.data_ptr(),
        ssm_state_vk.data_ptr<float>(),
        norm_w.data_ptr<float>(),
        query_start_loc.data_ptr<int32_t>(),
        positions.data_ptr<int32_t>(),
        req_slot_ids.data_ptr<int32_t>(),
        qk_norm_scratch.data_ptr<float>(),
        v_conv_scratch.data_ptr(),
        raw_out.data_ptr(),
        gated_out.data_ptr(),
        qkvzba.stride(0),
        conv_state.stride(0),
        ssm_state_vk.stride(0),
        ssm_state_vk.stride(1),
        static_cast<int>(num_seqs),
        static_cast<int>(qkvzba.size(0)),
        static_cast<float>(eps),
        stream_handle);
}
"""

_ROUTE_A_CUDA = r"""
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>

namespace cg = cooperative_groups;

#define ZSTEP_CUDA_CHECK(expr)                                                      \
    do {                                                                            \
        cudaError_t _err = (expr);                                                  \
        if (_err != cudaSuccess) {                                                  \
            throw std::runtime_error(                                               \
                std::string("CUDA error: ") + cudaGetErrorString(_err));            \
        }                                                                           \
    } while (0)

namespace {

constexpr int kMaxRanks = 8;
constexpr int64_t kShmV5AbortOffset = 98304;

union Pack128Bf16 {
    uint4 u4;
    __nv_bfloat162 pairs[4];
    __nv_bfloat16 elems[8];
};

__device__ __forceinline__ uint64_t sys_load_acquire_u64(const uint64_t* addr) {
    uint64_t val;
    asm volatile("ld.global.acquire.sys.b64 %0, [%1];" : "=l"(val) : "l"(addr) : "memory");
    return val;
}

__device__ __forceinline__ void sys_store_release_u64(uint64_t* addr, uint64_t val) {
    asm volatile("st.global.release.sys.b64 [%0], %1;" :: "l"(addr), "l"(val) : "memory");
}

__device__ __forceinline__ bool spin_wait_ge_u64(
    const uint64_t* flag, uint64_t target, const uint64_t* abort_flag) {
    uint32_t iters = 0;
    while (sys_load_acquire_u64(flag) < target) {
        __nanosleep(32);
        if ((++iters & 0xFFFU) == 0U && sys_load_acquire_u64(abort_flag) != 0ULL) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ uint4 ld_cv_uint4(const uint4* addr) {
    uint4 v;
    asm volatile("ld.global.cv.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(addr)
                 : "memory");
    return v;
}

__device__ __forceinline__ void st_wt_uint4(uint4* addr, uint4 v) {
    asm volatile("st.global.wt.v4.u32 [%0], {%1, %2, %3, %4};"
                 :: "l"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

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

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFFU, val, mask);
    }
    return val;
}

// ---------------------------------------------------------------------------
// A1-0: Historical pre-PR-#20 (`893495c`) interleaved-load `shm_v5_allreduce_kernel`
//       Kept inline so `"base_2op"` remains a true A/B baseline on post-#20 HEAD.
// ---------------------------------------------------------------------------
__global__ void shm_v5_ar_legacy_interleaved_kernel(
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
                my_shm + 65536 + (b * 8 + rank) * 64);
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
            float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int p = 0; p < kMaxRanks; ++p) {
                if (p < world_size) {
                    const uint4* p_in = reinterpret_cast<const uint4*>(
                        s_peer_shm[p] + 131072 + slot * slot_stride_bytes);
                    Pack128Bf16 v;
                    v.u4 = (p == rank) ? input_vec[idx] : ld_cv_uint4(p_in + idx);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 f = __bfloat1622float2(v.pairs[j]);
                        acc[2 * j] += f.x;
                        acc[2 * j + 1] += f.y;
                    }
                }
            }
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
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int p = 0; p < kMaxRanks; ++p) {
            if (p < world_size) {
                const uint4* p_in = reinterpret_cast<const uint4*>(
                    s_peer_shm[p] + 131072 + slot * slot_stride_bytes);
                Pack128Bf16 v;
                v.u4 = (p == rank) ? input_vec[idx] : ld_cv_uint4(p_in + idx);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __bfloat1622float2(v.pairs[j]);
                    acc[2 * j] += f.x;
                    acc[2 * j + 1] += f.y;
                }
            }
        }
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
    for (int owner = 0; owner < world_size; ++owner) {
        const int64_t o_begin = c_begin + (c_len * owner) / world_size;
        const int64_t o_end = c_begin + (c_len * (owner + 1)) / world_size;
        const uint4* o_red = reinterpret_cast<const uint4*>(
            s_peer_shm[owner] + 131072 + slot * slot_stride_bytes + half_slot);
        for (int64_t idx = o_begin + threadIdx.x; idx < o_end; idx += blockDim.x) {
            output_vec[idx] = ld_cv_uint4(o_red + idx);
        }
    }
    __syncthreads();
    if (threadIdx.x < world_size) {
        uint64_t* peer_done = reinterpret_cast<uint64_t*>(
            s_peer_shm[threadIdx.x] + 65536 + (b * 8 + rank) * 64);
        sys_store_release_u64(peer_done, step);
    }
}

// ---------------------------------------------------------------------------
// A1-a: Batched-load `shm_v5_allreduce_kernel` (unfused reference)
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(512, 1) void shm_v5_ar_batchload_kernel(
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
        if (s_aborted) return;
    }

    const int64_t half_slot = slot_stride_bytes / 2;
    uint4* my_in = reinterpret_cast<uint4*>(my_shm + 131072 + slot * slot_stride_bytes);
    for (int64_t idx = c_begin + threadIdx.x; idx < c_end; idx += blockDim.x) {
        st_wt_uint4(my_in + idx, input_vec[idx]);
    }
    __syncthreads();
    if (threadIdx.x == 0) __threadfence_system();
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
    if (s_aborted) return;

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
    if (threadIdx.x == 0) __threadfence_system();
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
    if (s_aborted) return;

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

// ---------------------------------------------------------------------------
// A1-b: Fused `shm_v5_allreduce_add_rmsnorm_kernel` (Stage-2 Epilogue Fusion)
// Preserves `done-wait` and `control[0..blocks)` lockstep 100%, 0-ULP residual!
// ---------------------------------------------------------------------------
__device__ __forceinline__ void apply_add_rmsnorm_epilogue_single_row(
    int64_t row_base,
    int64_t vecs_per_row,
    uint4* __restrict__ residual_vec,
    const uint4* __restrict__ weight_vec,
    uint4* __restrict__ normed_out_vec,
    const Pack128Bf16 (&red_packs)[3],
    int num_iters,
    float inv_hidden,
    float eps,
    float weight_offset,
    float* s_warp_sq,
    float& s_inv_rms) {
    float s_f32[3][8];
    float thread_sq = 0.0f;

    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        if (k < num_iters) {
            const int64_t col = threadIdx.x + static_cast<int64_t>(k) * blockDim.x;
            if (col < vecs_per_row) {
                const int64_t idx = row_base + col;
                Pack128Bf16 res_pack;
                res_pack.u4 = residual_vec[idx];
                Pack128Bf16 sum_pack;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 rf = __bfloat1622float2(red_packs[k].pairs[j]);
                    const float2 sf = __bfloat1622float2(res_pack.pairs[j]);
                    sum_pack.pairs[j] = __floats2bfloat162_rn(rf.x + sf.x, rf.y + sf.y);
                    const float2 sum_rt = __bfloat1622float2(sum_pack.pairs[j]);
                    s_f32[k][2 * j] = sum_rt.x;
                    s_f32[k][2 * j + 1] = sum_rt.y;
                    thread_sq += sum_rt.x * sum_rt.x + sum_rt.y * sum_rt.y;
                }
                residual_vec[idx] = sum_pack.u4;
            }
        }
    }

    const float warp_sq = warp_reduce_sum(thread_sq);
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int num_warps = blockDim.x >> 5;
    if (lane_id == 0) {
        s_warp_sq[warp_id] = warp_sq;
    }
    __syncthreads();
    if (warp_id == 0) {
        float acc_sq = (lane_id < num_warps) ? s_warp_sq[lane_id] : 0.0f;
        acc_sq = warp_reduce_sum(acc_sq);
        if (lane_id == 0) {
            s_inv_rms = rsqrtf(acc_sq * inv_hidden + eps);
        }
    }
    __syncthreads();
    const float inv_rms = s_inv_rms;

    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        if (k < num_iters) {
            const int64_t col = threadIdx.x + static_cast<int64_t>(k) * blockDim.x;
            if (col < vecs_per_row) {
                Pack128Bf16 w_pack;
                w_pack.u4 = weight_vec[col];
                Pack128Bf16 out_pack;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 wf = __bfloat1622float2(w_pack.pairs[j]);
                    const float y0 = (s_f32[k][2 * j] * inv_rms) * (weight_offset + wf.x);
                    const float y1 = (s_f32[k][2 * j + 1] * inv_rms) * (weight_offset + wf.y);
                    out_pack.pairs[j] = __floats2bfloat162_rn(y0, y1);
                }
                normed_out_vec[row_base + col] = out_pack.u4;
            }
        }
    }
}

__global__ __launch_bounds__(512, 1) void shm_v5_ar_add_rmsnorm_kernel(
    const uint4* __restrict__ input_vec,
    uint4* __restrict__ residual_vec,
    const uint4* __restrict__ weight_vec,
    uint4* __restrict__ normed_out_vec,
    uint64_t* __restrict__ control,
    char* const* __restrict__ rank_shm,
    int64_t slot_stride_bytes,
    int64_t rows,
    int64_t vecs_per_row,
    float inv_hidden,
    float eps,
    float weight_offset,
    int rank,
    int world_size,
    int one_shot_max_vecs) {
    const int b = blockIdx.x;
    const int blocks = gridDim.x;
    __shared__ uint64_t s_step;
    __shared__ int s_aborted;
    __shared__ char* s_peer_shm[kMaxRanks];
    __shared__ float s_warp_sq[16];
    __shared__ float s_inv_rms_arr[2];

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

    // Entry done-wait (100% preserved)
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
        if (s_aborted) return;
    }

    const int64_t num_vecs = rows * vecs_per_row;
    const int64_t half_slot = slot_stride_bytes / 2;
    uint4* my_in = reinterpret_cast<uint4*>(my_shm + 131072 + slot * slot_stride_bytes);

    // Regime A: `rows < blocks` (with `blocks % rows == 0`, e.g. M=1,2,4 at blocks=8)
    if (rows < blocks) {
        const int64_t c_begin = (num_vecs * b) / blocks;
        const int64_t c_end = (num_vecs * (b + 1)) / blocks;
        for (int64_t idx = c_begin + threadIdx.x; idx < c_end; idx += blockDim.x) {
            st_wt_uint4(my_in + idx, input_vec[idx]);
        }
        __syncthreads();
        if (threadIdx.x == 0) __threadfence_system();
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
        if (s_aborted) return;

        const int k_grp = static_cast<int>(blocks / rows);
        if ((b % k_grp) != 0) {
            return;
        }
        // Leader CTA `b = r * k_grp` waits on local-NUMA in_flags for sub-blocks `b+1 .. b+k_grp-1`
        const int extra_flags = (k_grp - 1) * world_size;
        for (int f = threadIdx.x; f < extra_flags; f += blockDim.x) {
            const int b_sub = b + 1 + (f / world_size);
            const int peer = f % world_size;
            const uint64_t* local_in_flag = reinterpret_cast<const uint64_t*>(
                my_shm + (b_sub * 8 + peer) * 64);
            if (!spin_wait_ge_u64(local_in_flag, step, abort_flag)) {
                s_aborted = 1;
            }
        }
        __syncthreads();
        if (s_aborted) return;

        const int64_t r = b / k_grp;
        const int64_t row_base = r * vecs_per_row;
        const int num_iters = static_cast<int>((vecs_per_row + blockDim.x - 1) / blockDim.x);
        Pack128Bf16 red_packs[3];
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            if (k < num_iters) {
                const int64_t col = threadIdx.x + static_cast<int64_t>(k) * blockDim.x;
                if (col < vecs_per_row) {
                    const int64_t idx = row_base + col;
                    uint4 peer_vals[kMaxRanks];
                    load_peer_vecs(peer_vals, input_vec, s_peer_shm, slot, slot_stride_bytes, idx, rank, world_size);
                    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                    accumulate_peer_vecs(acc, peer_vals, world_size);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        red_packs[k].pairs[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
                    }
                }
            }
        }
        apply_add_rmsnorm_epilogue_single_row(
            row_base, vecs_per_row, residual_vec, weight_vec, normed_out_vec,
            red_packs, num_iters, inv_hidden, eps, weight_offset, s_warp_sq, s_inv_rms_arr[0]);
        __syncthreads();
        const int total_done_flags = k_grp * world_size;
        for (int f = threadIdx.x; f < total_done_flags; f += blockDim.x) {
            const int b_sub = b + (f / world_size);
            const int peer = f % world_size;
            uint64_t* peer_done = reinterpret_cast<uint64_t*>(
                s_peer_shm[peer] + 65536 + (b_sub * 8 + rank) * 64);
            sys_store_release_u64(peer_done, step);
        }
        return;
    }

    // Regime B: `rows >= blocks` -> row-aligned CTA partition `[r_begin, r_end)`
    const int64_t r_begin = (rows * b) / blocks;
    const int64_t r_end = (rows * (b + 1)) / blocks;
    const int64_t c_begin = r_begin * vecs_per_row;
    const int64_t c_end = r_end * vecs_per_row;

    for (int64_t idx = c_begin + threadIdx.x; idx < c_end; idx += blockDim.x) {
        st_wt_uint4(my_in + idx, input_vec[idx]);
    }
    __syncthreads();
    if (threadIdx.x == 0) __threadfence_system();
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
    if (s_aborted) return;

    const int num_iters = static_cast<int>((vecs_per_row + blockDim.x - 1) / blockDim.x);

    // Subcase B1: One-Shot (`num_vecs <= one_shot_max_vecs`, e.g. H=2048 M=8, 16)
    if (num_vecs <= one_shot_max_vecs) {
        for (int64_t r = r_begin; r < r_end; ++r) {
            const int64_t row_base = r * vecs_per_row;
            Pack128Bf16 red_packs[3];
            #pragma unroll
            for (int k = 0; k < 3; ++k) {
                if (k < num_iters) {
                    const int64_t col = threadIdx.x + static_cast<int64_t>(k) * blockDim.x;
                    if (col < vecs_per_row) {
                        const int64_t idx = row_base + col;
                        uint4 peer_vals[kMaxRanks];
                        load_peer_vecs(peer_vals, input_vec, s_peer_shm, slot, slot_stride_bytes, idx, rank, world_size);
                        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                        accumulate_peer_vecs(acc, peer_vals, world_size);
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            red_packs[k].pairs[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
                        }
                    }
                }
            }
            apply_add_rmsnorm_epilogue_single_row(
                row_base, vecs_per_row, residual_vec, weight_vec, normed_out_vec,
                red_packs, num_iters, inv_hidden, eps, weight_offset, s_warp_sq, s_inv_rms_arr[0]);
            __syncthreads();
        }
        if (threadIdx.x < world_size) {
            uint64_t* peer_done = reinterpret_cast<uint64_t*>(
                s_peer_shm[threadIdx.x] + 65536 + (b * 8 + rank) * 64);
            sys_store_release_u64(peer_done, step);
        }
        return;
    }

    // Subcase B2: Two-Shot (`num_vecs > one_shot_max_vecs`, e.g. H=2048 M>=32 or H=5120 M>=8)
    // Stripe Stage-1 reduce-scatter across columns of each row (`cols_per_rank = vecs_per_row / world_size`)
    // so Stage-2 row gather reads from ALL 8 peers in parallel across warps!
    const int64_t cols_per_rank = vecs_per_row / world_size;
    const int64_t col_s_begin = cols_per_rank * rank;
    const int64_t total_s_vecs = (r_end - r_begin) * cols_per_rank;
    uint4* my_red = reinterpret_cast<uint4*>(my_shm + 131072 + slot * slot_stride_bytes + half_slot);

    for (int64_t t_idx = threadIdx.x; t_idx < total_s_vecs; t_idx += blockDim.x) {
        const int64_t r = r_begin + (t_idx / cols_per_rank);
        const int64_t col = col_s_begin + (t_idx % cols_per_rank);
        const int64_t idx = r * vecs_per_row + col;
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
    if (threadIdx.x == 0) __threadfence_system();
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
    if (s_aborted) return;

    // Fast path for H=2048 (`vecs_per_row == 256`) with 512 threads: process 2 rows simultaneously!
    if (vecs_per_row == 256 && blockDim.x == 512) {
        const int row_slot = threadIdx.x >> 8;   // 0 or 1
        const int sub_tid = threadIdx.x & 255;   // 0..255 (= col!)
        const int sub_warp = sub_tid >> 5;       // 0..7 (= owner when world_size==8!)
        const int lane_id = sub_tid & 31;
        const int owner = static_cast<int>(sub_tid / cols_per_rank);
        const uint4* o_red = reinterpret_cast<const uint4*>(
            s_peer_shm[owner] + 131072 + slot * slot_stride_bytes + half_slot);
        Pack128Bf16 w_pack;
        w_pack.u4 = weight_vec[sub_tid];
        float2 wf[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            wf[j] = __bfloat1622float2(w_pack.pairs[j]);
        }

        for (int64_t r0 = r_begin; r0 < r_end; r0 += 2) {
            const int64_t r = r0 + row_slot;
            float s_f32[8];
            float thread_sq = 0.0f;
            if (r < r_end) {
                const int64_t idx = r * 256 + sub_tid;
                Pack128Bf16 red_pack;
                red_pack.u4 = ld_cv_uint4(o_red + idx);
                Pack128Bf16 res_pack;
                res_pack.u4 = residual_vec[idx];
                Pack128Bf16 sum_pack;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 rf = __bfloat1622float2(red_pack.pairs[j]);
                    const float2 sf = __bfloat1622float2(res_pack.pairs[j]);
                    sum_pack.pairs[j] = __floats2bfloat162_rn(rf.x + sf.x, rf.y + sf.y);
                    const float2 sum_rt = __bfloat1622float2(sum_pack.pairs[j]);
                    s_f32[2 * j] = sum_rt.x;
                    s_f32[2 * j + 1] = sum_rt.y;
                    thread_sq += sum_rt.x * sum_rt.x + sum_rt.y * sum_rt.y;
                }
                residual_vec[idx] = sum_pack.u4;
            }
            const float warp_sq = warp_reduce_sum(thread_sq);
            if (lane_id == 0) {
                s_warp_sq[row_slot * 8 + sub_warp] = warp_sq;
            }
            __syncthreads();
            if (sub_warp == 0) {
                float acc_sq = (lane_id < 8) ? s_warp_sq[row_slot * 8 + lane_id] : 0.0f;
                acc_sq = warp_reduce_sum(acc_sq);
                if (lane_id == 0) {
                    s_inv_rms_arr[row_slot] = rsqrtf(acc_sq * inv_hidden + eps);
                }
            }
            __syncthreads();
            if (r < r_end) {
                const float inv_rms = s_inv_rms_arr[row_slot];
                Pack128Bf16 out_pack;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float y0 = (s_f32[2 * j] * inv_rms) * (weight_offset + wf[j].x);
                    const float y1 = (s_f32[2 * j + 1] * inv_rms) * (weight_offset + wf[j].y);
                    out_pack.pairs[j] = __floats2bfloat162_rn(y0, y1);
                }
                normed_out_vec[r * 256 + sub_tid] = out_pack.u4;
            }
            __syncthreads();
        }
    } else {
        for (int64_t r = r_begin; r < r_end; ++r) {
            const int64_t row_base = r * vecs_per_row;
            Pack128Bf16 red_packs[3];
            #pragma unroll
            for (int k = 0; k < 3; ++k) {
                if (k < num_iters) {
                    const int64_t col = threadIdx.x + static_cast<int64_t>(k) * blockDim.x;
                    if (col < vecs_per_row) {
                        const int owner = static_cast<int>(col / cols_per_rank);
                        const uint4* o_red = reinterpret_cast<const uint4*>(
                            s_peer_shm[owner] + 131072 + slot * slot_stride_bytes + half_slot);
                        red_packs[k].u4 = ld_cv_uint4(o_red + row_base + col);
                    }
                }
            }
            apply_add_rmsnorm_epilogue_single_row(
                row_base, vecs_per_row, residual_vec, weight_vec, normed_out_vec,
                red_packs, num_iters, inv_hidden, eps, weight_offset, s_warp_sq, s_inv_rms_arr[0]);
            __syncthreads();
        }
    }

    if (threadIdx.x < world_size) {
        uint64_t* peer_done = reinterpret_cast<uint64_t*>(
            s_peer_shm[threadIdx.x] + 65536 + (b * 8 + rank) * 64);
        sys_store_release_u64(peer_done, step);
    }
}

// ===========================================================================
// Part A2: SM120 GatedDeltaNet (`[V, K]` Coalesced + Cluster/DSMEM Decode & Prefill)
// ===========================================================================

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float softplus_f32(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

__device__ __forceinline__ float conv1d_silu_bf16_round(
    const __nv_bfloat16* __restrict__ qkvzba_row,
    __nv_bfloat16* __restrict__ conv_slot_base,
    const __nv_bfloat16* __restrict__ conv_w,
    int c,
    int pos) {
    const float x = __bfloat162float(qkvzba_row[c]);
    __nv_bfloat16* st = conv_slot_base + c * 3;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
    if (pos != 0) {
        s0 = __bfloat162float(st[0]);
        s1 = __bfloat162float(st[1]);
        s2 = __bfloat162float(st[2]);
    }
    st[0] = __float2bfloat16_rn(s1);
    st[1] = __float2bfloat16_rn(s2);
    st[2] = __float2bfloat16_rn(x);

    const __nv_bfloat16* cw = conv_w + c * 4;
    const float w0 = __bfloat162float(cw[0]);
    const float w1 = __bfloat162float(cw[1]);
    const float w2 = __bfloat162float(cw[2]);
    const float w3 = __bfloat162float(cw[3]);
    const float acc = w0 * s0 + w1 * s1 + w2 * s2 + w3 * x;
    const float y = silu_f32(acc);
    return __bfloat162float(__float2bfloat16_rn(y));
}

// ---------------------------------------------------------------------------
// Variant 0: 1-CTA (256 threads = 8 warps) per `(row, k_head)` fused Decode
// No cluster needed; handles all 256 V-channels (`v_head = 2*kh, 2*kh+1`) in 1 CTA.
// ---------------------------------------------------------------------------
__global__ void gdn_decode_fused_1cta_vk_kernel(
    const __nv_bfloat16* __restrict__ qkvzba,
    __nv_bfloat16* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ conv_w,
    const float* __restrict__ a_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    float* __restrict__ ssm_state_vk,
    const float* __restrict__ norm_w,
    const int32_t* __restrict__ positions,
    const int32_t* __restrict__ req_slot_ids,
    __nv_bfloat16* __restrict__ gated_out,
    int64_t stride_qkvzba_row,
    int64_t stride_conv_slot,
    int64_t stride_ssm_slot,
    int64_t stride_ssm_head,
    int num_active_rows,
    float eps) {
    const int kh = blockIdx.x;
    const int64_t row = static_cast<int64_t>(blockIdx.y);
    const int tid = threadIdx.x;  // 0..255
    const int warp_id = tid >> 5; // 0..7
    const int lane = tid & 31;    // 0..31

    if (row >= num_active_rows) {
        gated_out[row * 512 + kh * 256 + tid] = __float2bfloat16_rn(0.0f);
        return;
    }

    __shared__ __align__(16) float s_q[128];
    __shared__ __align__(16) float s_k[128];
    __shared__ float s_v[256];
    __shared__ float s_out[256];
    __shared__ float s_sq[8];
    __shared__ float s_scales[6]; // q_scale, k_scale, beta0, decay0, beta1, decay1

    const int64_t slot = static_cast<int64_t>(req_slot_ids[row]);
    const int pos = positions[row];
    const __nv_bfloat16* row_ptr = qkvzba + row * stride_qkvzba_row;
    __nv_bfloat16* conv_slot = conv_state + slot * stride_conv_slot;

    // 256 threads compute 512 conv1d channels (2 channels per thread):
    // tid < 128 -> Q[tid]; tid >= 128 -> K[tid - 128]; plus every tid computes V[tid]!
    const int c_qk = (tid < 128) ? (kh * 128 + tid) : (256 + kh * 128 + (tid - 128));
    const float qk_val = conv1d_silu_bf16_round(row_ptr, conv_slot, conv_w, c_qk, pos);
    const int c_v = 512 + kh * 256 + tid;
    const float v_val = conv1d_silu_bf16_round(row_ptr, conv_slot, conv_w, c_v, pos);
    s_v[tid] = v_val;

    const float qk_sq = warp_reduce_sum(qk_val * qk_val);
    if (lane == 0) {
        s_sq[warp_id] = qk_sq;
    }
    if (tid < 2) {
        const int vh = 2 * kh + tid;
        const float b_raw = __bfloat162float(row_ptr[1536 + vh]);
        const float a_raw = __bfloat162float(row_ptr[1540 + vh]);
        const float al = a_log[vh];
        const float db = __bfloat162float(dt_bias[vh]);
        s_scales[2 + tid * 2 + 0] = 1.0f / (1.0f + expf(-b_raw));
        s_scales[2 + tid * 2 + 1] = expf(-expf(al) * softplus_f32(a_raw + db));
    }
    __syncthreads();

    if (tid == 0) {
        const float q_sum = s_sq[0] + s_sq[1] + s_sq[2] + s_sq[3];
        const float k_sum = s_sq[4] + s_sq[5] + s_sq[6] + s_sq[7];
        s_scales[0] = rsqrtf(q_sum + 1.0e-6f) * 0.08838834764831845f;
        s_scales[1] = rsqrtf(k_sum + 1.0e-6f);
    }
    __syncthreads();

    if (tid < 128) {
        s_q[tid] = qk_val * s_scales[0];
    } else {
        s_k[tid - 128] = qk_val * s_scales[1];
    }
    __syncthreads();

    const float4 q4 = reinterpret_cast<const float4*>(s_q)[lane];
    const float4 k4 = reinterpret_cast<const float4*>(s_k)[lane];
    const int vh_sub = warp_id >> 2; // 0 for warps 0..3, 1 for warps 4..7
    const int vh = 2 * kh + vh_sub;
    const float beta = s_scales[2 + vh_sub * 2 + 0];
    const float decay = s_scales[2 + vh_sub * 2 + 1];
    const int v_base = (warp_id & 3) * 32; // 32 v-rows per warp
    float4* state_base4 = reinterpret_cast<float4*>(
        ssm_state_vk + slot * stride_ssm_slot + vh * stride_ssm_head);

    #pragma unroll 4
    for (int r = 0; r < 32; ++r) {
        const int vr = v_base + r;
        float4 st4 = (pos == 0) ? make_float4(0.f, 0.f, 0.f, 0.f) : state_base4[vr * 32 + lane];
        st4.x *= decay;
        st4.y *= decay;
        st4.z *= decay;
        st4.w *= decay;
        const float kv_p = st4.x * k4.x + st4.y * k4.y + st4.z * k4.z + st4.w * k4.w;
        const float kv_mem = warp_reduce_sum(kv_p);
        const float delta = (s_v[vh_sub * 128 + vr] - kv_mem) * beta;
        st4.x += k4.x * delta;
        st4.y += k4.y * delta;
        st4.z += k4.z * delta;
        st4.w += k4.w * delta;
        state_base4[vr * 32 + lane] = st4;
        const float out_p = st4.x * q4.x + st4.y * q4.y + st4.z * q4.z + st4.w * q4.w;
        const float out_v = warp_reduce_sum(out_p);
        if (lane == 0) {
            s_out[vh_sub * 128 + vr] = out_v;
        }
    }
    __syncthreads();

    // Fused RMSNorm + SiLU(Z) gate across all 256 V channels (2 V-heads x 128)
    const float x_out = s_out[tid];
    const float out_sq = warp_reduce_sum(x_out * x_out);
    if (lane == 0) {
        s_sq[warp_id] = out_sq;
    }
    __syncthreads();
    if (tid == 0) {
        s_scales[0] = rsqrtf((s_sq[0] + s_sq[1] + s_sq[2] + s_sq[3]) * (1.0f / 128.0f) + eps);
        s_scales[1] = rsqrtf((s_sq[4] + s_sq[5] + s_sq[6] + s_sq[7]) * (1.0f / 128.0f) + eps);
    }
    __syncthreads();

    const int head_local = tid >> 7; // 0 or 1
    const int d = tid & 127;
    const float inv_rms = s_scales[head_local];
    const float normed = __bfloat162float(__float2bfloat16_rn(x_out * inv_rms)) * norm_w[d];
    const float z = __bfloat162float(row_ptr[1024 + kh * 256 + tid]);
    gated_out[row * 512 + kh * 256 + tid] = __float2bfloat16_rn(normed * silu_f32(z));
}

// ---------------------------------------------------------------------------
// Variant 1 & 2: SM120 Thread Block Cluster (`CLUSTER_SIZE = 4` or `8`) + DSMEM!
// `gridDim = dim3(2 * CLUSTER_SIZE, num_rows, 1)`, `blockDim = 128` (4 warps).
// Overlaps 16..32 KiB `ssm_state` global float4 reads with Conv1d + DSMEM Q/K sync!
// ---------------------------------------------------------------------------
template <int CLUSTER_SIZE>
__global__ __cluster_dims__(CLUSTER_SIZE, 1, 1) void gdn_decode_fused_cluster_vk_kernel(
    const __nv_bfloat16* __restrict__ qkvzba,
    __nv_bfloat16* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ conv_w,
    const float* __restrict__ a_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    float* __restrict__ ssm_state_vk,
    const float* __restrict__ norm_w,
    const int32_t* __restrict__ positions,
    const int32_t* __restrict__ req_slot_ids,
    __nv_bfloat16* __restrict__ gated_out,
    int64_t stride_qkvzba_row,
    int64_t stride_conv_slot,
    int64_t stride_ssm_slot,
    int64_t stride_ssm_head,
    int num_active_rows,
    float eps) {
    static_assert(CLUSTER_SIZE == 4 || CLUSTER_SIZE == 8, "CLUSTER_SIZE must be 4 or 8");
    constexpr int kVPerCta = 256 / CLUSTER_SIZE;     // 64 (C=4) or 32 (C=8)
    constexpr int kRowsPerWarp = kVPerCta / 4;       // 16 (C=4) or 8  (C=8)
    constexpr int kCtasPerVHead = CLUSTER_SIZE / 2;  // 2  (C=4) or 4  (C=8)

    cg::cluster_group cluster = cg::this_cluster();
    const int cta_rank = cluster.block_rank();       // 0 .. CLUSTER_SIZE-1
    const int kh = blockIdx.x / CLUSTER_SIZE;        // 0 or 1
    const int64_t row = static_cast<int64_t>(blockIdx.y);
    const int tid = threadIdx.x;                     // 0..127
    const int warp_id = tid >> 5;                    // 0..3
    const int lane = tid & 31;                       // 0..31

    __shared__ __align__(16) float s_q[128];
    __shared__ __align__(16) float s_k[128];
    __shared__ __align__(16) float s_my_qk[64];
    __shared__ float s_my_qk_sq;
    __shared__ float s_my_out_sq;
    __shared__ float s_v[kVPerCta];
    __shared__ float s_out[kVPerCta];
    __shared__ float s_warp[4];
    __shared__ float s_meta[4]; // q_scale, k_scale, beta, decay (or inv_rms)

    const bool active = (row < num_active_rows);
    int64_t slot = 0;
    int pos = 0;
    const __nv_bfloat16* row_ptr = nullptr;
    __nv_bfloat16* conv_slot = nullptr;

    const int vh_sub = cta_rank / kCtasPerVHead;     // 0 or 1
    const int vh_tile = cta_rank % kCtasPerVHead;
    const int vh = 2 * kh + vh_sub;
    const int v_cta_offset = vh_tile * kVPerCta;     // within `vh` (0..127)

    // Pre-load `ssm_state_vk` into registers BEFORE `cluster.sync()` to hide GDDR7 latency!
    float4 st_regs[kRowsPerWarp];
    float4* state_base4 = nullptr;
    if (active) {
        slot = static_cast<int64_t>(req_slot_ids[row]);
        pos = positions[row];
        row_ptr = qkvzba + row * stride_qkvzba_row;
        conv_slot = conv_state + slot * stride_conv_slot;
        state_base4 = reinterpret_cast<float4*>(
            ssm_state_vk + slot * stride_ssm_slot + vh * stride_ssm_head);

        #pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            const int vr = v_cta_offset + warp_id * kRowsPerWarp + r;
            st_regs[r] = (pos == 0) ? make_float4(0.f, 0.f, 0.f, 0.f) : state_base4[vr * 32 + lane];
        }

        // Own V slice conv1d (`kVPerCta` = 64 or 32 channels)
        if (tid < kVPerCta) {
            const int c_v = 512 + vh * 128 + v_cta_offset + tid;
            s_v[tid] = conv1d_silu_bf16_round(row_ptr, conv_slot, conv_w, c_v, pos);
        }

        // CTAs 0..3 also compute Q[0..63], Q[64..127], K[0..63], K[64..127] (64 channels each)
        float qk_val = 0.0f;
        if (cta_rank < 4 && tid < 64) {
            const int c_qk = (cta_rank < 2)
                ? (kh * 128 + cta_rank * 64 + tid)
                : (256 + kh * 128 + (cta_rank - 2) * 64 + tid);
            qk_val = conv1d_silu_bf16_round(row_ptr, conv_slot, conv_w, c_qk, pos);
            s_my_qk[tid] = qk_val;
        }
        if (cta_rank < 4 && warp_id < 2) {
            const float w_sq = warp_reduce_sum(qk_val * qk_val);
            if (lane == 0) s_warp[warp_id] = w_sq;
        }
        if (tid == 0) {
            const float b_raw = __bfloat162float(row_ptr[1536 + vh]);
            const float a_raw = __bfloat162float(row_ptr[1540 + vh]);
            const float al = a_log[vh];
            const float db = __bfloat162float(dt_bias[vh]);
            s_meta[2] = 1.0f / (1.0f + expf(-b_raw));
            s_meta[3] = expf(-expf(al) * softplus_f32(a_raw + db));
        }
        __syncthreads();
        if (cta_rank < 4 && tid == 0) {
            s_my_qk_sq = s_warp[0] + s_warp[1];
        }
    }

    // Cluster Barrier 1: publish `s_my_qk` and `s_my_qk_sq` across DSMEM
    cluster.sync();

    if (active) {
        if (tid == 0) {
            const float q_sq0 = *cluster.map_shared_rank(&s_my_qk_sq, 0);
            const float q_sq1 = *cluster.map_shared_rank(&s_my_qk_sq, 1);
            const float k_sq0 = *cluster.map_shared_rank(&s_my_qk_sq, 2);
            const float k_sq1 = *cluster.map_shared_rank(&s_my_qk_sq, 3);
            s_meta[0] = rsqrtf(q_sq0 + q_sq1 + 1.0e-6f) * 0.08838834764831845f;
            s_meta[1] = rsqrtf(k_sq0 + k_sq1 + 1.0e-6f);
        }
        // Pull Q[0..127] and K[0..127] from CTAs 0..3 via DSMEM
        const int src_half = tid >> 6;   // 0 or 1
        const int half_off = tid & 63;   // 0..63
        const float raw_q = cluster.map_shared_rank(s_my_qk, src_half)[half_off];
        const float raw_k = cluster.map_shared_rank(s_my_qk, 2 + src_half)[half_off];
        __syncthreads();

        s_q[tid] = raw_q * s_meta[0];
        s_k[tid] = raw_k * s_meta[1];
        __syncthreads();

        const float4 q4 = reinterpret_cast<const float4*>(s_q)[lane];
        const float4 k4 = reinterpret_cast<const float4*>(s_k)[lane];
        const float beta = s_meta[2];
        const float decay = s_meta[3];

        #pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            const int local_v = warp_id * kRowsPerWarp + r;
            const int vr = v_cta_offset + local_v;
            float4 st4 = st_regs[r];
            st4.x *= decay;
            st4.y *= decay;
            st4.z *= decay;
            st4.w *= decay;
            const float kv_p = st4.x * k4.x + st4.y * k4.y + st4.z * k4.z + st4.w * k4.w;
            const float kv_mem = warp_reduce_sum(kv_p);
            const float delta = (s_v[local_v] - kv_mem) * beta;
            st4.x += k4.x * delta;
            st4.y += k4.y * delta;
            st4.z += k4.z * delta;
            st4.w += k4.w * delta;
            state_base4[vr * 32 + lane] = st4;
            const float out_p = st4.x * q4.x + st4.y * q4.y + st4.z * q4.z + st4.w * q4.w;
            const float out_v = warp_reduce_sum(out_p);
            if (lane == 0) {
                s_out[local_v] = out_v;
            }
        }
        __syncthreads();

        float my_v = (tid < kVPerCta) ? s_out[tid] : 0.0f;
        const float w_sq = warp_reduce_sum(my_v * my_v);
        if (lane == 0) s_warp[warp_id] = w_sq;
        __syncthreads();
        if (tid == 0) {
            s_my_out_sq = s_warp[0] + s_warp[1];
        }
    }

    // Cluster Barrier 2: publish `s_my_out_sq` for per-V-head RMSNorm across sibling CTAs
    cluster.sync();

    if (active) {
        if (tid == 0) {
            float head_sq = 0.0f;
            const int base_cta = vh_sub * kCtasPerVHead;
            #pragma unroll
            for (int c = 0; c < kCtasPerVHead; ++c) {
                head_sq += *cluster.map_shared_rank(&s_my_out_sq, base_cta + c);
            }
            s_meta[0] = rsqrtf(head_sq * (1.0f / 128.0f) + eps);
        }
        __syncthreads();
        if (tid < kVPerCta) {
            const int d = v_cta_offset + tid;
            const float inv_rms = s_meta[0];
            const float normed = __bfloat162float(__float2bfloat16_rn(s_out[tid] * inv_rms)) * norm_w[d];
            const float z = __bfloat162float(row_ptr[1024 + vh * 128 + d]);
            gated_out[row * 512 + vh * 128 + d] = __float2bfloat16_rn(normed * silu_f32(z));
        }
    } else {
        if (tid < kVPerCta) {
            const int d = v_cta_offset + tid;
            gated_out[row * 512 + vh * 128 + d] = __float2bfloat16_rn(0.0f);
        }
    }

    // Cluster Barrier 3: ensure no CTA exits before siblings read `s_my_out_sq`
    cluster.sync();
}

// ---------------------------------------------------------------------------
// Variant 3: SM120 Cluster-2 (`__cluster_dims__(2, 1, 1)`, 512 threads = 16 warps/CTA)
// 1 CTA per `(v_head, row)`: CTA 0 computes Q_kh + V_{2kh}, CTA 1 computes K_kh + V_{2kh+1},
// ALL 8 `float4` state rows per warp are preloaded BEFORE DSMEM Q/K exchange, and
// GatedRMSNormSiLU is 100% CTA-local (no second DSMEM reduction)!
// ---------------------------------------------------------------------------
__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(512, 1) void gdn_decode_fused_cluster2_512_vk_kernel(
    const __nv_bfloat16* __restrict__ qkvzba,
    __nv_bfloat16* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ conv_w,
    const float* __restrict__ a_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    float* __restrict__ ssm_state_vk,
    const float* __restrict__ norm_w,
    const int32_t* __restrict__ positions,
    const int32_t* __restrict__ req_slot_ids,
    __nv_bfloat16* __restrict__ gated_out,
    int64_t stride_qkvzba_row,
    int64_t stride_conv_slot,
    int64_t stride_ssm_slot,
    int64_t stride_ssm_head,
    int num_active_rows,
    float eps) {
    cg::cluster_group cluster = cg::this_cluster();
    const int cta_rank = cluster.block_rank(); // 0 or 1 (`vh_sub`)
    const int kh = blockIdx.x >> 1;            // 0 or 1
    const int vh = 2 * kh + cta_rank;          // 0..3
    const int64_t row = static_cast<int64_t>(blockIdx.y);
    const int tid = threadIdx.x;               // 0..511
    const int warp_id = tid >> 5;              // 0..15
    const int lane = tid & 31;                 // 0..31

    __shared__ __align__(16) float s_my_qk_norm[128];
    __shared__ __align__(16) float s_peer_qk_norm[128];
    __shared__ float s_v[128];
    __shared__ float s_out[128];
    __shared__ float s_warp[16];
    __shared__ float s_meta[4]; // qk_scale, beta, decay, inv_rms

    const bool active = (row < num_active_rows);
    float4 st_regs[8];
    float4* state_base4 = nullptr;
    const __nv_bfloat16* row_ptr = nullptr;

    if (active) {
        const int64_t slot = static_cast<int64_t>(req_slot_ids[row]);
        const int pos = positions[row];
        row_ptr = qkvzba + row * stride_qkvzba_row;
        __nv_bfloat16* conv_slot = conv_state + slot * stride_conv_slot;
        state_base4 = reinterpret_cast<float4*>(
            ssm_state_vk + slot * stride_ssm_slot + vh * stride_ssm_head);

        // Issue ALL 8 float4 global loads per warp FIRST (`16 warps * 8 = 128` rows = entire V-head!)
        if (pos == 0) {
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                st_regs[r] = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        } else {
            const int v_base = warp_id * 8;
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                st_regs[r] = state_base4[(v_base + r) * 32 + lane];
            }
        }

        // Overlap Conv1d + Q/K L2-norm while GDDR7 `st_regs` loads are in flight:
        // tid 0..127: CTA 0 computes Q_kh[0..127], CTA 1 computes K_kh[0..127]
        // tid 128..255: CTA `cta_rank` computes V_vh[0..127]
        float qk_val = 0.0f;
        if (tid < 128) {
            const int c_qk = (cta_rank == 0) ? (kh * 128 + tid) : (256 + kh * 128 + tid);
            qk_val = conv1d_silu_bf16_round(row_ptr, conv_slot, conv_w, c_qk, pos);
        } else if (tid < 256) {
            const int d_v = tid - 128;
            const int c_v = 512 + vh * 128 + d_v;
            s_v[d_v] = conv1d_silu_bf16_round(row_ptr, conv_slot, conv_w, c_v, pos);
        } else if (tid == 256) {
            const float b_raw = __bfloat162float(row_ptr[1536 + vh]);
            const float a_raw = __bfloat162float(row_ptr[1540 + vh]);
            const float al = a_log[vh];
            const float db = __bfloat162float(dt_bias[vh]);
            s_meta[1] = 1.0f / (1.0f + expf(-b_raw));
            s_meta[2] = expf(-expf(al) * softplus_f32(a_raw + db));
        }

        if (warp_id < 4) {
            const float w_sq = warp_reduce_sum(qk_val * qk_val);
            if (lane == 0) s_warp[warp_id] = w_sq;
        }
        __syncthreads();
        if (tid == 0) {
            const float sq = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
            const float extra = (cta_rank == 0) ? 0.08838834764831845f : 1.0f;
            s_meta[0] = rsqrtf(sq + 1.0e-6f) * extra;
        }
        __syncthreads();
        if (tid < 128) {
            s_my_qk_norm[tid] = qk_val * s_meta[0];
        }
    }

    // Single DSMEM data barrier: exchange normalized Q_kh (CTA 0) <-> K_kh (CTA 1) (512B)!
    cluster.sync();

    if (active) {
        if (tid < 128) {
            s_peer_qk_norm[tid] = cluster.map_shared_rank(s_my_qk_norm, 1 - cta_rank)[tid];
        }
        __syncthreads();

        const float4 q4 = (cta_rank == 0)
            ? reinterpret_cast<const float4*>(s_my_qk_norm)[lane]
            : reinterpret_cast<const float4*>(s_peer_qk_norm)[lane];
        const float4 k4 = (cta_rank == 1)
            ? reinterpret_cast<const float4*>(s_my_qk_norm)[lane]
            : reinterpret_cast<const float4*>(s_peer_qk_norm)[lane];
        const float beta = s_meta[1];
        const float decay = s_meta[2];
        const int v_base = warp_id * 8;

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int vr = v_base + r;
            float4 st4 = st_regs[r];
            st4.x *= decay;
            st4.y *= decay;
            st4.z *= decay;
            st4.w *= decay;
            const float kv_p = st4.x * k4.x + st4.y * k4.y + st4.z * k4.z + st4.w * k4.w;
            const float kv_mem = warp_reduce_sum(kv_p);
            const float delta = (s_v[vr] - kv_mem) * beta;
            st4.x += k4.x * delta;
            st4.y += k4.y * delta;
            st4.z += k4.z * delta;
            st4.w += k4.w * delta;
            state_base4[vr * 32 + lane] = st4;
            const float out_p = st4.x * q4.x + st4.y * q4.y + st4.z * q4.z + st4.w * q4.w;
            const float out_v = warp_reduce_sum(out_p);
            if (lane == 0) {
                s_out[vr] = __bfloat162float(__float2bfloat16_rn(out_v));
            }
        }
        __syncthreads();

        // CTA-local GatedRMSNormSiLU across all 128 channels of `vh`!
        if (tid < 128) {
            const float x_out = s_out[tid];
            const float w_sq = warp_reduce_sum(x_out * x_out);
            if (lane == 0) s_warp[warp_id] = w_sq;
        }
        __syncthreads();
        if (tid == 0) {
            const float h_sq = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
            s_meta[3] = rsqrtf(h_sq * (1.0f / 128.0f) + eps);
        }
        __syncthreads();
        if (tid < 128) {
            const float inv_rms = s_meta[3];
            const float normed = __bfloat162float(__float2bfloat16_rn(s_out[tid] * inv_rms)) * norm_w[tid];
            const float z = __bfloat162float(row_ptr[1024 + vh * 128 + tid]);
            gated_out[row * 512 + vh * 128 + tid] = __float2bfloat16_rn(normed * silu_f32(z));
        }
    } else {
        if (tid < 128) {
            gated_out[row * 512 + vh * 128 + tid] = __float2bfloat16_rn(0.0f);
        }
    }

    cluster.sync();
}

// ---------------------------------------------------------------------------
// Non-cluster `[V, K]` Decode Building Blocks (`Variant 4` 2-op & `Variant 5` 32-tile 3-op)
// Zero cluster barriers, 100% `float4` coalesced `[V, K]` state!
// ---------------------------------------------------------------------------
__global__ void gdn_decode_prologue_conv_qknorm_kernel(
    const __nv_bfloat16* __restrict__ qkvzba,
    __nv_bfloat16* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ conv_w,
    const int32_t* __restrict__ positions,
    const int32_t* __restrict__ req_slot_ids,
    float* __restrict__ qk_norm_out,     // [num_rows, 512] FP32
    __nv_bfloat16* __restrict__ v_out,   // [num_rows, 512] BF16
    int64_t stride_qkvzba_row,
    int64_t stride_conv_slot,
    int num_active_rows) {
    // gridDim = (num_rows, 2), blockDim = 256
    const int64_t row = blockIdx.x;
    const int kh = blockIdx.y; // 0 or 1
    const int tid = threadIdx.x; // 0..255
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    if (row >= num_active_rows) return;

    __shared__ float s_sq[8];
    __shared__ float s_scales[2];

    const int64_t slot = req_slot_ids[row];
    const int pos = positions[row];
    const __nv_bfloat16* row_ptr = qkvzba + row * stride_qkvzba_row;
    __nv_bfloat16* conv_slot = conv_state + slot * stride_conv_slot;

    const int c_qk = (tid < 128) ? (kh * 128 + tid) : (256 + kh * 128 + (tid - 128));
    const float qk_val = conv1d_silu_bf16_round(row_ptr, conv_slot, conv_w, c_qk, pos);
    const int c_v = 512 + kh * 256 + tid;
    const float v_val = conv1d_silu_bf16_round(row_ptr, conv_slot, conv_w, c_v, pos);
    v_out[row * 512 + kh * 256 + tid] = __float2bfloat16_rn(v_val);

    const float w_sq = warp_reduce_sum(qk_val * qk_val);
    if (lane == 0) s_sq[warp_id] = w_sq;
    __syncthreads();
    if (tid == 0) {
        s_scales[0] = rsqrtf(s_sq[0] + s_sq[1] + s_sq[2] + s_sq[3] + 1.0e-6f) * 0.08838834764831845f;
        s_scales[1] = rsqrtf(s_sq[4] + s_sq[5] + s_sq[6] + s_sq[7] + 1.0e-6f);
    }
    __syncthreads();
    if (tid < 128) {
        qk_norm_out[row * 512 + kh * 128 + tid] = qk_val * s_scales[0];
    } else {
        qk_norm_out[row * 512 + 256 + kh * 128 + (tid - 128)] = qk_val * s_scales[1];
    }
}

__global__ __launch_bounds__(512, 1) void gdn_decode_vhead512_epilogue_vk_kernel(
    const float* __restrict__ qk_norm,
    const __nv_bfloat16* __restrict__ v_conv,
    const __nv_bfloat16* __restrict__ qkvzba,
    const float* __restrict__ a_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    float* __restrict__ ssm_state_vk,
    const float* __restrict__ norm_w,
    const int32_t* __restrict__ positions,
    const int32_t* __restrict__ req_slot_ids,
    __nv_bfloat16* __restrict__ gated_out,
    int64_t stride_qkvzba_row,
    int64_t stride_ssm_slot,
    int64_t stride_ssm_head,
    int num_active_rows,
    float eps) {
    // gridDim = (num_rows, 4), blockDim = 512 (16 warps x 8 V-rows/warp, zero cluster!)
    const int64_t row = blockIdx.x;
    const int vh = blockIdx.y; // 0..3
    const int kh = vh >> 1;
    const int tid = threadIdx.x; // 0..511
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    if (row >= num_active_rows) {
        if (tid < 128) gated_out[row * 512 + vh * 128 + tid] = __float2bfloat16_rn(0.0f);
        return;
    }

    __shared__ float s_out[128];
    __shared__ float s_warp[4];
    __shared__ float s_inv_rms;

    const int64_t slot = req_slot_ids[row];
    const int pos = positions[row];
    float4* state_base4 = reinterpret_cast<float4*>(
        ssm_state_vk + slot * stride_ssm_slot + vh * stride_ssm_head);
    const int v_base = warp_id * 8;

    float4 st_regs[8];
    if (pos == 0) {
        #pragma unroll
        for (int r = 0; r < 8; ++r) st_regs[r] = make_float4(0.f, 0.f, 0.f, 0.f);
    } else {
        #pragma unroll
        for (int r = 0; r < 8; ++r) st_regs[r] = state_base4[(v_base + r) * 32 + lane];
    }

    const float4 q4 = reinterpret_cast<const float4*>(qk_norm + row * 512 + kh * 128)[lane];
    const float4 k4 = reinterpret_cast<const float4*>(qk_norm + row * 512 + 256 + kh * 128)[lane];
    const __nv_bfloat16* row_ptr = qkvzba + row * stride_qkvzba_row;
    const float beta = 1.0f / (1.0f + expf(-__bfloat162float(row_ptr[1536 + vh])));
    const float decay = expf(-expf(a_log[vh]) * softplus_f32(__bfloat162float(row_ptr[1540 + vh]) + __bfloat162float(dt_bias[vh])));
    const __nv_bfloat16* v_ptr = v_conv + row * 512 + vh * 128;

    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int vr = v_base + r;
        float4 st4 = st_regs[r];
        st4.x *= decay; st4.y *= decay; st4.z *= decay; st4.w *= decay;
        const float kv_mem = warp_reduce_sum(st4.x * k4.x + st4.y * k4.y + st4.z * k4.z + st4.w * k4.w);
        const float delta = (__bfloat162float(v_ptr[vr]) - kv_mem) * beta;
        st4.x += k4.x * delta; st4.y += k4.y * delta; st4.z += k4.z * delta; st4.w += k4.w * delta;
        state_base4[vr * 32 + lane] = st4;
        const float out_v = warp_reduce_sum(st4.x * q4.x + st4.y * q4.y + st4.z * q4.z + st4.w * q4.w);
        if (lane == 0) s_out[vr] = __bfloat162float(__float2bfloat16_rn(out_v));
    }
    __syncthreads();

    if (tid < 128) {
        const float x_out = s_out[tid];
        const float w_sq = warp_reduce_sum(x_out * x_out);
        if (lane == 0) s_warp[warp_id] = w_sq;
    }
    __syncthreads();
    if (tid == 0) {
        s_inv_rms = rsqrtf((s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3]) * (1.0f / 128.0f) + eps);
    }
    __syncthreads();
    if (tid < 128) {
        const float normed = __bfloat162float(__float2bfloat16_rn(s_out[tid] * s_inv_rms)) * norm_w[tid];
        const float z = __bfloat162float(row_ptr[1024 + vh * 128 + tid]);
        gated_out[row * 512 + vh * 128 + tid] = __float2bfloat16_rn(normed * silu_f32(z));
    }
}

__global__ void gdn_decode_32tile_vk_kernel(
    const float* __restrict__ qk_norm,
    const __nv_bfloat16* __restrict__ v_conv,
    const __nv_bfloat16* __restrict__ qkvzba,
    const float* __restrict__ a_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    float* __restrict__ ssm_state_vk,
    const int32_t* __restrict__ positions,
    const int32_t* __restrict__ req_slot_ids,
    __nv_bfloat16* __restrict__ raw_out,
    int64_t stride_qkvzba_row,
    int64_t stride_ssm_slot,
    int64_t stride_ssm_head,
    int num_active_rows) {
    // gridDim = (num_rows, 32), blockDim = 128 (4 warps x 4 V-rows/warp -> 32*B CTAs across all 156 SMs!)
    const int64_t row = blockIdx.x;
    if (row >= num_active_rows) return;
    const int hv_tile = blockIdx.y; // 0..31
    const int vh = hv_tile >> 3;    // 0..3
    const int v_blk = hv_tile & 7;  // 0..7
    const int kh = vh >> 1;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int64_t slot = req_slot_ids[row];
    const int pos = positions[row];
    const int v_base = v_blk * 16 + warp_id * 4;
    float4* state_base4 = reinterpret_cast<float4*>(
        ssm_state_vk + slot * stride_ssm_slot + vh * stride_ssm_head);

    float4 st_regs[4];
    if (pos == 0) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) st_regs[r] = make_float4(0.f, 0.f, 0.f, 0.f);
    } else {
        #pragma unroll
        for (int r = 0; r < 4; ++r) st_regs[r] = state_base4[(v_base + r) * 32 + lane];
    }

    const float4 q4 = reinterpret_cast<const float4*>(qk_norm + row * 512 + kh * 128)[lane];
    const float4 k4 = reinterpret_cast<const float4*>(qk_norm + row * 512 + 256 + kh * 128)[lane];
    const __nv_bfloat16* row_ptr = qkvzba + row * stride_qkvzba_row;
    const float beta = 1.0f / (1.0f + expf(-__bfloat162float(row_ptr[1536 + vh])));
    const float decay = expf(-expf(a_log[vh]) * softplus_f32(__bfloat162float(row_ptr[1540 + vh]) + __bfloat162float(dt_bias[vh])));
    const __nv_bfloat16* v_ptr = v_conv + row * 512 + vh * 128;
    __nv_bfloat16* out_ptr = raw_out + row * 512 + vh * 128;

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int vr = v_base + r;
        float4 st4 = st_regs[r];
        st4.x *= decay; st4.y *= decay; st4.z *= decay; st4.w *= decay;
        const float kv_mem = warp_reduce_sum(st4.x * k4.x + st4.y * k4.y + st4.z * k4.z + st4.w * k4.w);
        const float delta = (__bfloat162float(v_ptr[vr]) - kv_mem) * beta;
        st4.x += k4.x * delta; st4.y += k4.y * delta; st4.z += k4.z * delta; st4.w += k4.w * delta;
        state_base4[vr * 32 + lane] = st4;
        const float out_v = warp_reduce_sum(st4.x * q4.x + st4.y * q4.y + st4.z * q4.z + st4.w * q4.w);
        if (lane == 0) out_ptr[vr] = __float2bfloat16_rn(out_v);
    }
}

__global__ void gdn_rmsnorm_gated_silu_decode_cuda_kernel(
    const __nv_bfloat16* __restrict__ raw_out,
    const __nv_bfloat16* __restrict__ qkvzba,
    const float* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ gated_out,
    int64_t stride_qkvzba_row,
    int num_active_rows,
    float eps) {
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x; // 0..127
    const int v_head = tid >> 5;
    const int lane = tid & 31;
    const int base_off = v_head * 128 + lane * 4;
    if (row >= num_active_rows) {
        *reinterpret_cast<uint2*>(gated_out + row * 512 + base_off) = make_uint2(0, 0);
        return;
    }
    const uint2 rx = *reinterpret_cast<const uint2*>(raw_out + row * 512 + base_off);
    const uint2 rz = *reinterpret_cast<const uint2*>(qkvzba + row * stride_qkvzba_row + 1024 + base_off);
    const float4 wf = *reinterpret_cast<const float4*>(norm_w + lane * 4);
    const float2 xf01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rx.x));
    const float2 xf23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rx.y));
    const float2 zf01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rz.x));
    const float2 zf23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rz.y));
    const float sq = xf01.x * xf01.x + xf01.y * xf01.y + xf23.x * xf23.x + xf23.y * xf23.y;
    const float inv_rms = rsqrtf(warp_reduce_sum(sq) * (1.0f / 128.0f) + eps);
    const float2 nf01 = __bfloat1622float2(__floats2bfloat162_rn(xf01.x * inv_rms, xf01.y * inv_rms));
    const float2 nf23 = __bfloat1622float2(__floats2bfloat162_rn(xf23.x * inv_rms, xf23.y * inv_rms));
    const __nv_bfloat162 out01 = __floats2bfloat162_rn((nf01.x * wf.x) * silu_f32(zf01.x), (nf01.y * wf.y) * silu_f32(zf01.y));
    const __nv_bfloat162 out23 = __floats2bfloat162_rn((nf23.x * wf.z) * silu_f32(zf23.x), (nf23.y * wf.w) * silu_f32(zf23.y));
    uint2 packed_out;
    packed_out.x = *reinterpret_cast<const uint32_t*>(&out01);
    packed_out.y = *reinterpret_cast<const uint32_t*>(&out23);
    *reinterpret_cast<uint2*>(gated_out + row * 512 + base_off) = packed_out;
}

// ---------------------------------------------------------------------------
// Prefill `[V, K]` Coalesced Kernels:
// 1. `gdn_prefill_conv1d_qknorm_kernel`: computes causal conv1d on Q/K/V, AND
//    normalizes Q/K in-place into FP32 `qk_norm_scratch[total_rows, 512]` so
//    the recurrence kernel NEVER redundantly recomputes Q/K L2-norm!
// 2. `gdn_prefill_recurrent_vk_kernel`: 4 warps per CTA, `BLOCK_V = 16` (`4` v-rows/warp)
//    with `float4` coalesced `[V, K]` state loads/stores + warp-shuffle dot products!
// 3. `gdn_rmsnorm_gated_silu_cuda_kernel`: vectorized per-(row, v_head) epilogue.
// ---------------------------------------------------------------------------
__global__ void gdn_conv1d_prefill_varlen_cuda_kernel(
    const __nv_bfloat16* __restrict__ qkvzba,
    __nv_bfloat16* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ conv_w,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ positions,
    const int32_t* __restrict__ req_slot_ids,
    __nv_bfloat16* __restrict__ v_conv_out, // [total_rows, 1024] (0..511 raw qk_conv bf16, 512..1023 v_conv bf16)
    int64_t stride_in_row,
    int64_t stride_state_slot) {
    const int seq_idx = blockIdx.x;
    const int c = blockIdx.y * 64 + threadIdx.x; // 16 blocks * 64 threads = 1024 channels
    const int64_t q_start = query_start_loc[seq_idx];
    const int64_t q_end = query_start_loc[seq_idx + 1];
    if (q_start >= q_end) return;

    const int64_t slot = req_slot_ids[q_start];
    const int start_pos = positions[q_start];
    __nv_bfloat16* st = conv_state + slot * stride_state_slot + c * 3;

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
    if (start_pos != 0) {
        s0 = __bfloat162float(st[0]);
        s1 = __bfloat162float(st[1]);
        s2 = __bfloat162float(st[2]);
    }
    const __nv_bfloat16* cw = conv_w + c * 4;
    const float w0 = __bfloat162float(cw[0]);
    const float w1 = __bfloat162float(cw[1]);
    const float w2 = __bfloat162float(cw[2]);
    const float w3 = __bfloat162float(cw[3]);

    for (int64_t row = q_start; row < q_end; ++row) {
        const float x = __bfloat162float(qkvzba[row * stride_in_row + c]);
        const float acc = w0 * s0 + w1 * s1 + w2 * s2 + w3 * x;
        const float y = silu_f32(acc);
        v_conv_out[row * 1024 + c] = __float2bfloat16_rn(y);
        s0 = s1;
        s1 = s2;
        s2 = x;
    }
    st[0] = __float2bfloat16_rn(s0);
    st[1] = __float2bfloat16_rn(s1);
    st[2] = __float2bfloat16_rn(s2);
}

// Normalize Q0, Q1, K0, K1 (`4` vectors of 128 per row) ONCE into FP32 `qk_norm[total_rows, 512]`
__global__ void gdn_qk_l2norm_rows_kernel(
    const __nv_bfloat16* __restrict__ qkv_conv,
    float* __restrict__ qk_norm,
    int total_rows) {
    // gridDim = (total_rows, 4), blockDim = 128
    const int64_t row = blockIdx.x;
    if (row >= total_rows) return;
    const int vec_id = blockIdx.y; // 0: Q0, 1: Q1, 2: K0, 3: K1
    const int tid = threadIdx.x;   // 0..127
    __shared__ float s_w[4];
    __shared__ float s_scale;

    const float v = __bfloat162float(qkv_conv[row * 1024 + vec_id * 128 + tid]);
    const float w_sq = warp_reduce_sum(v * v);
    if ((tid & 31) == 0) s_w[tid >> 5] = w_sq;
    __syncthreads();
    if (tid == 0) {
        const float sq = s_w[0] + s_w[1] + s_w[2] + s_w[3];
        const float extra = (vec_id < 2) ? 0.08838834764831845f : 1.0f;
        s_scale = rsqrtf(sq + 1.0e-6f) * extra;
    }
    __syncthreads();
    qk_norm[row * 512 + vec_id * 128 + tid] = v * s_scale;
}

// Coalesced `[V, K]` recurrent prefill: `gridDim = (num_seqs, 4 * (128 / 16))` = `(num_seqs, 32)`, `blockDim = 128` (4 warps x 4 v-rows = 16 v-rows/CTA)
__global__ void gdn_prefill_recurrent_vk_kernel(
    const float* __restrict__ qk_norm,
    const __nv_bfloat16* __restrict__ qkv_conv,
    const __nv_bfloat16* __restrict__ qkvzba,
    const float* __restrict__ a_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    float* __restrict__ ssm_state_vk,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ positions,
    const int32_t* __restrict__ req_slot_ids,
    __nv_bfloat16* __restrict__ raw_out,
    int64_t stride_qkvzba_row,
    int64_t stride_ssm_slot,
    int64_t stride_ssm_head) {
    const int seq_idx = blockIdx.x;
    const int hv_tile = blockIdx.y; // 0..31
    const int v_head = hv_tile >> 3; // 0..3
    const int v_blk = hv_tile & 7;   // 0..7 -> 16 v-rows each
    const int k_head = v_head >> 1;

    const int64_t q_start = query_start_loc[seq_idx];
    const int64_t q_end = query_start_loc[seq_idx + 1];
    if (q_start >= q_end) return;

    const int tid = threadIdx.x;   // 0..127
    const int warp_id = tid >> 5;  // 0..3
    const int lane = tid & 31;     // 0..31

    const int64_t slot = req_slot_ids[q_start];
    const int start_pos = positions[q_start];
    const int v_base = v_blk * 16 + warp_id * 4;
    float4* state_base4 = reinterpret_cast<float4*>(
        ssm_state_vk + slot * stride_ssm_slot + v_head * stride_ssm_head);

    float4 st[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        st[r] = (start_pos == 0)
            ? make_float4(0.f, 0.f, 0.f, 0.f)
            : state_base4[(v_base + r) * 32 + lane];
    }

    const float neg_a = -expf(a_log[v_head]);
    const float db = __bfloat162float(dt_bias[v_head]);

    for (int64_t row = q_start; row < q_end; ++row) {
        const float4 q4 = reinterpret_cast<const float4*>(qk_norm + row * 512 + k_head * 128)[lane];
        const float4 k4 = reinterpret_cast<const float4*>(qk_norm + row * 512 + 256 + k_head * 128)[lane];
        const __nv_bfloat16* zb_row = qkvzba + row * stride_qkvzba_row;
        const float beta = 1.0f / (1.0f + expf(-__bfloat162float(zb_row[1536 + v_head])));
        const float decay = expf(neg_a * softplus_f32(__bfloat162float(zb_row[1540 + v_head]) + db));
        const __nv_bfloat16* v_row = qkv_conv + row * 1024 + 512 + v_head * 128;
        __nv_bfloat16* out_row = raw_out + row * 512 + v_head * 128;

        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int vr = v_base + r;
            st[r].x *= decay;
            st[r].y *= decay;
            st[r].z *= decay;
            st[r].w *= decay;
            const float kv_p = st[r].x * k4.x + st[r].y * k4.y + st[r].z * k4.z + st[r].w * k4.w;
            const float kv_mem = warp_reduce_sum(kv_p);
            const float v_val = __bfloat162float(v_row[vr]);
            const float delta = (v_val - kv_mem) * beta;
            st[r].x += k4.x * delta;
            st[r].y += k4.y * delta;
            st[r].z += k4.z * delta;
            st[r].w += k4.w * delta;
            const float out_p = st[r].x * q4.x + st[r].y * q4.y + st[r].z * q4.z + st[r].w * q4.w;
            const float out_v = warp_reduce_sum(out_p);
            if (lane == 0) {
                out_row[vr] = __float2bfloat16_rn(out_v);
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        state_base4[(v_base + r) * 32 + lane] = st[r];
    }
}

__global__ void gdn_rmsnorm_gated_silu_cuda_kernel(
    const __nv_bfloat16* __restrict__ raw_out,
    const __nv_bfloat16* __restrict__ qkvzba,
    const float* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ gated_out,
    int64_t stride_qkvzba_row,
    int total_rows,
    float eps) {
    // 1 CTA (128 threads = 4 warps, 1 warp per v_head * 4 elems/lane -> ALL 4 V-heads of 1 row in 1 CTA!)
    const int64_t row = blockIdx.x;
    if (row >= total_rows) return;
    const int tid = threadIdx.x;   // 0..127
    const int v_head = tid >> 5;   // 0..3
    const int lane = tid & 31;     // 0..31

    const int base_off = v_head * 128 + lane * 4;
    const uint2 rx = *reinterpret_cast<const uint2*>(raw_out + row * 512 + base_off);
    const uint2 rz = *reinterpret_cast<const uint2*>(qkvzba + row * stride_qkvzba_row + 1024 + base_off);
    const float4 wf = *reinterpret_cast<const float4*>(norm_w + lane * 4);

    const __nv_bfloat162 x01 = *reinterpret_cast<const __nv_bfloat162*>(&rx.x);
    const __nv_bfloat162 x23 = *reinterpret_cast<const __nv_bfloat162*>(&rx.y);
    const __nv_bfloat162 z01 = *reinterpret_cast<const __nv_bfloat162*>(&rz.x);
    const __nv_bfloat162 z23 = *reinterpret_cast<const __nv_bfloat162*>(&rz.y);

    const float2 xf01 = __bfloat1622float2(x01);
    const float2 xf23 = __bfloat1622float2(x23);
    const float2 zf01 = __bfloat1622float2(z01);
    const float2 zf23 = __bfloat1622float2(z23);

    const float sq = xf01.x * xf01.x + xf01.y * xf01.y + xf23.x * xf23.x + xf23.y * xf23.y;
    const float head_sq = warp_reduce_sum(sq);
    const float inv_rms = rsqrtf(head_sq * (1.0f / 128.0f) + eps);

    const __nv_bfloat162 n01 = __floats2bfloat162_rn(xf01.x * inv_rms, xf01.y * inv_rms);
    const __nv_bfloat162 n23 = __floats2bfloat162_rn(xf23.x * inv_rms, xf23.y * inv_rms);
    const float2 nf01 = __bfloat1622float2(n01);
    const float2 nf23 = __bfloat1622float2(n23);

    const __nv_bfloat162 out01 = __floats2bfloat162_rn(
        (nf01.x * wf.x) * silu_f32(zf01.x),
        (nf01.y * wf.y) * silu_f32(zf01.y));
    const __nv_bfloat162 out23 = __floats2bfloat162_rn(
        (nf23.x * wf.z) * silu_f32(zf23.x),
        (nf23.y * wf.w) * silu_f32(zf23.y));

    uint2 packed_out;
    packed_out.x = *reinterpret_cast<const uint32_t*>(&out01);
    packed_out.y = *reinterpret_cast<const uint32_t*>(&out23);
    *reinterpret_cast<uint2*>(gated_out + row * 512 + base_off) = packed_out;
}

// ---------------------------------------------------------------------------
// A1-c: SM120 Cluster-8 DSMEM `shm_v5_ar_add_rmsnorm_cluster8_kernel` for `rows < 8`
// Keeps ALL 8 SMs issuing parallel PCIe `ld_cv_uint4` reads (1/8th slice per CTA),
// exchanges ONLY 1 float (`s_cta_sq`) across DSMEM, and keeps `control[0..7]` in lockstep!
// ---------------------------------------------------------------------------
__global__ __cluster_dims__(8, 1, 1) __launch_bounds__(256, 1) void shm_v5_ar_add_rmsnorm_cluster8_kernel(
    const uint4* __restrict__ input_vec,
    uint4* __restrict__ residual_vec,
    const uint4* __restrict__ weight_vec,
    uint4* __restrict__ normed_out_vec,
    uint64_t* __restrict__ control,
    char* const* __restrict__ rank_shm,
    int64_t slot_stride_bytes,
    int64_t rows,
    int64_t vecs_per_row,
    float inv_hidden,
    float eps,
    float weight_offset,
    int rank,
    int world_size) {
    cg::cluster_group cluster = cg::this_cluster();
    const int b = cluster.block_rank(); // 0..7 == blockIdx.x
    constexpr int blocks = 8;

    __shared__ uint64_t s_step;
    __shared__ int s_aborted_local;
    __shared__ int s_aborted_any;
    __shared__ char* s_peer_shm[kMaxRanks];
    __shared__ float s_warp_sq[16];
    __shared__ float s_cta_sq;
    __shared__ float s_inv_rms;

    if (threadIdx.x < world_size) {
        s_peer_shm[threadIdx.x] = rank_shm[threadIdx.x];
    }
    if (threadIdx.x == 0) {
        const uint64_t ns = control[b] + 1ULL;
        control[b] = ns;
        s_step = ns;
        s_aborted_local = 0;
        s_aborted_any = 0;
    }
    __syncthreads();
    const uint64_t step = s_step;
    const int slot = static_cast<int>((step - 1ULL) & 3);
    char* my_shm = s_peer_shm[rank];
    const uint64_t* abort_flag = reinterpret_cast<const uint64_t*>(my_shm + kShmV5AbortOffset);

    if (step > 4ULL) {
        const uint64_t min_done = step - 4ULL;
        if (threadIdx.x < world_size) {
            const uint64_t* my_done = reinterpret_cast<const uint64_t*>(
                my_shm + 65536 + (b * 8 + threadIdx.x) * 64);
            if (!spin_wait_ge_u64(my_done, min_done, abort_flag)) {
                s_aborted_local = 1;
            }
        }
        __syncthreads();
    }

    const int64_t num_vecs = rows * vecs_per_row;
    const int64_t c_begin = (num_vecs * b) / blocks;
    const int64_t c_end = (num_vecs * (b + 1)) / blocks;
    const int64_t slice_len = c_end - c_begin;
    uint4* my_in = reinterpret_cast<uint4*>(my_shm + 131072 + slot * slot_stride_bytes);

    if (!s_aborted_local) {
        for (int64_t idx = c_begin + threadIdx.x; idx < c_end; idx += blockDim.x) {
            st_wt_uint4(my_in + idx, input_vec[idx]);
        }
        __syncthreads();
        if (threadIdx.x == 0) __threadfence_system();
        __syncthreads();
        if (threadIdx.x < world_size) {
            uint64_t* peer_in_flag = reinterpret_cast<uint64_t*>(
                s_peer_shm[threadIdx.x] + (b * 8 + rank) * 64);
            sys_store_release_u64(peer_in_flag, step);
            const uint64_t* local_in_flag = reinterpret_cast<const uint64_t*>(
                my_shm + (b * 8 + threadIdx.x) * 64);
            if (!spin_wait_ge_u64(local_in_flag, step, abort_flag)) {
                s_aborted_local = 1;
            }
        }
        __syncthreads();
    }

    const int k_grp = static_cast<int>(blocks / rows); // 8 (M=1), 4 (M=2), 2 (M=4)
    const int64_t r = b / k_grp;
    const int64_t row_base = r * vecs_per_row;
    const int num_iters = static_cast<int>((slice_len + blockDim.x - 1) / blockDim.x);

    float s_f32[3][8];
    float thread_sq = 0.0f;

    if (!s_aborted_local) {
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            if (k < num_iters) {
                const int64_t local_off = threadIdx.x + static_cast<int64_t>(k) * blockDim.x;
                if (local_off < slice_len) {
                    const int64_t idx = c_begin + local_off;
                    uint4 peer_vals[kMaxRanks];
                    load_peer_vecs(peer_vals, input_vec, s_peer_shm, slot, slot_stride_bytes, idx, rank, world_size);
                    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                    accumulate_peer_vecs(acc, peer_vals, world_size);
                    Pack128Bf16 red_pack;
                    Pack128Bf16 res_pack;
                    res_pack.u4 = residual_vec[idx];
                    Pack128Bf16 sum_pack;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        red_pack.pairs[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
                        const float2 rf = __bfloat1622float2(red_pack.pairs[j]);
                        const float2 sf = __bfloat1622float2(res_pack.pairs[j]);
                        sum_pack.pairs[j] = __floats2bfloat162_rn(rf.x + sf.x, rf.y + sf.y);
                        const float2 sum_rt = __bfloat1622float2(sum_pack.pairs[j]);
                        s_f32[k][2 * j] = sum_rt.x;
                        s_f32[k][2 * j + 1] = sum_rt.y;
                        thread_sq += sum_rt.x * sum_rt.x + sum_rt.y * sum_rt.y;
                    }
                    residual_vec[idx] = sum_pack.u4;
                }
            }
        }
        const float warp_sq = warp_reduce_sum(thread_sq);
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 31;
        const int num_warps = blockDim.x >> 5;
        if (lane_id == 0) {
            s_warp_sq[warp_id] = warp_sq;
        }
        __syncthreads();
        if (warp_id == 0) {
            float acc_sq = (lane_id < num_warps) ? s_warp_sq[lane_id] : 0.0f;
            acc_sq = warp_reduce_sum(acc_sq);
            if (lane_id == 0) {
                s_cta_sq = acc_sq;
            }
        }
        __syncthreads();
    }

    cluster.sync();

    if (threadIdx.x == 0) {
        int any_ab = 0;
        for (int c = 0; c < blocks; ++c) {
            any_ab |= *cluster.map_shared_rank(&s_aborted_local, c);
        }
        s_aborted_any = any_ab;
        if (!any_ab) {
            float row_sq = 0.0f;
            const int grp_base = static_cast<int>(r) * k_grp;
            for (int c = 0; c < k_grp; ++c) {
                row_sq += *cluster.map_shared_rank(&s_cta_sq, grp_base + c);
            }
            s_inv_rms = rsqrtf(row_sq * inv_hidden + eps);
        }
    }
    __syncthreads();

    if (!s_aborted_any) {
        const float inv_rms = s_inv_rms;
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            if (k < num_iters) {
                const int64_t local_off = threadIdx.x + static_cast<int64_t>(k) * blockDim.x;
                if (local_off < slice_len) {
                    const int64_t idx = c_begin + local_off;
                    const int64_t col = idx - row_base;
                    Pack128Bf16 w_pack;
                    w_pack.u4 = weight_vec[col];
                    Pack128Bf16 out_pack;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 wf = __bfloat1622float2(w_pack.pairs[j]);
                        const float y0 = (s_f32[k][2 * j] * inv_rms) * (weight_offset + wf.x);
                        const float y1 = (s_f32[k][2 * j + 1] * inv_rms) * (weight_offset + wf.y);
                        out_pack.pairs[j] = __floats2bfloat162_rn(y0, y1);
                    }
                    normed_out_vec[idx] = out_pack.u4;
                }
            }
        }
        if (threadIdx.x < world_size) {
            uint64_t* peer_done = reinterpret_cast<uint64_t*>(
                s_peer_shm[threadIdx.x] + 65536 + (b * 8 + rank) * 64);
            sys_store_release_u64(peer_done, step);
        }
    }

    cluster.sync();
}

}  // namespace

void launch_shm_v5_ar_legacy_interleaved_cuda(
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
    shm_v5_ar_legacy_interleaved_kernel<<<blocks, launch_threads, 0, stream>>>(
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

void launch_shm_v5_ar_batchload_cuda(
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
    shm_v5_ar_batchload_kernel<<<blocks, launch_threads, 0, stream>>>(
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

void launch_shm_v5_ar_add_rmsnorm_cuda(
    const void* input_ptr,
    void* residual_ptr,
    const void* weight_ptr,
    void* normed_out_ptr,
    uint64_t* control_ptr,
    const std::uintptr_t* rank_shm_dev_ptrs,
    int64_t slot_stride_bytes,
    int64_t rows,
    int64_t vecs_per_row,
    float inv_hidden,
    float eps,
    float weight_offset,
    int rank,
    int world_size,
    int blocks,
    int threads,
    int one_shot_max_vecs,
    int use_cluster,
    std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const int64_t num_vecs = rows * vecs_per_row;
    const int64_t vecs_per_cta = (num_vecs + blocks - 1) / blocks;

    if (use_cluster != 0 && rows < blocks && blocks == 8 && (blocks % rows) == 0) {
        static bool attr_init = false;
        if (!attr_init) {
            cudaFuncSetAttribute(
                shm_v5_ar_add_rmsnorm_cluster8_kernel,
                cudaFuncAttributeNonPortableClusterSizeAllowed,
                1);
            attr_init = true;
        }
        shm_v5_ar_add_rmsnorm_cluster8_kernel<<<8, 256, 0, stream>>>(
            reinterpret_cast<const uint4*>(input_ptr),
            reinterpret_cast<uint4*>(residual_ptr),
            reinterpret_cast<const uint4*>(weight_ptr),
            reinterpret_cast<uint4*>(normed_out_ptr),
            control_ptr,
            reinterpret_cast<char* const*>(rank_shm_dev_ptrs),
            slot_stride_bytes,
            rows,
            vecs_per_row,
            inv_hidden,
            eps,
            weight_offset,
            rank,
            world_size);
        ZSTEP_CUDA_CHECK(cudaGetLastError());
        return;
    }

    const int launch_threads = (threads == 256 && (vecs_per_cta > 256 || vecs_per_row > 256)) ? 512 : threads;
    shm_v5_ar_add_rmsnorm_kernel<<<blocks, launch_threads, 0, stream>>>(
        reinterpret_cast<const uint4*>(input_ptr),
        reinterpret_cast<uint4*>(residual_ptr),
        reinterpret_cast<const uint4*>(weight_ptr),
        reinterpret_cast<uint4*>(normed_out_ptr),
        control_ptr,
        reinterpret_cast<char* const*>(rank_shm_dev_ptrs),
        slot_stride_bytes,
        rows,
        vecs_per_row,
        inv_hidden,
        eps,
        weight_offset,
        rank,
        world_size,
        one_shot_max_vecs);
    ZSTEP_CUDA_CHECK(cudaGetLastError());
}

void launch_gdn_decode_fused_vk_cuda(
    const void* qkvzba_ptr,
    void* conv_state_ptr,
    const void* conv_w_ptr,
    const float* a_log_ptr,
    const void* dt_bias_ptr,
    float* ssm_state_vk_ptr,
    const float* norm_w_ptr,
    const int32_t* positions_ptr,
    const int32_t* req_slot_ids_ptr,
    void* gated_out_ptr,
    int64_t stride_qkvzba_row,
    int64_t stride_conv_slot,
    int64_t stride_ssm_slot,
    int64_t stride_ssm_head,
    int num_rows,
    int num_active_rows,
    float eps,
    int variant,
    std::uintptr_t stream_handle) {
    if (num_rows <= 0) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    if (variant == 0) {
        dim3 grid(2, static_cast<unsigned int>(num_rows), 1);
        gdn_decode_fused_1cta_vk_kernel<<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(qkvzba_ptr),
            reinterpret_cast<__nv_bfloat16*>(conv_state_ptr),
            reinterpret_cast<const __nv_bfloat16*>(conv_w_ptr),
            a_log_ptr,
            reinterpret_cast<const __nv_bfloat16*>(dt_bias_ptr),
            ssm_state_vk_ptr,
            norm_w_ptr,
            positions_ptr,
            req_slot_ids_ptr,
            reinterpret_cast<__nv_bfloat16*>(gated_out_ptr),
            stride_qkvzba_row,
            stride_conv_slot,
            stride_ssm_slot,
            stride_ssm_head,
            num_active_rows,
            eps);
    } else if (variant == 1) {
        static bool c4_init = false;
        if (!c4_init) {
            cudaFuncSetAttribute(
                gdn_decode_fused_cluster_vk_kernel<4>,
                cudaFuncAttributeNonPortableClusterSizeAllowed,
                1);
            c4_init = true;
        }
        dim3 grid(2 * 4, static_cast<unsigned int>(num_rows), 1);
        gdn_decode_fused_cluster_vk_kernel<4><<<grid, 128, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(qkvzba_ptr),
            reinterpret_cast<__nv_bfloat16*>(conv_state_ptr),
            reinterpret_cast<const __nv_bfloat16*>(conv_w_ptr),
            a_log_ptr,
            reinterpret_cast<const __nv_bfloat16*>(dt_bias_ptr),
            ssm_state_vk_ptr,
            norm_w_ptr,
            positions_ptr,
            req_slot_ids_ptr,
            reinterpret_cast<__nv_bfloat16*>(gated_out_ptr),
            stride_qkvzba_row,
            stride_conv_slot,
            stride_ssm_slot,
            stride_ssm_head,
            num_active_rows,
            eps);
    } else if (variant == 2) {
        static bool c8_init = false;
        if (!c8_init) {
            cudaFuncSetAttribute(
                gdn_decode_fused_cluster_vk_kernel<8>,
                cudaFuncAttributeNonPortableClusterSizeAllowed,
                1);
            c8_init = true;
        }
        dim3 grid(2 * 8, static_cast<unsigned int>(num_rows), 1);
        gdn_decode_fused_cluster_vk_kernel<8><<<grid, 128, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(qkvzba_ptr),
            reinterpret_cast<__nv_bfloat16*>(conv_state_ptr),
            reinterpret_cast<const __nv_bfloat16*>(conv_w_ptr),
            a_log_ptr,
            reinterpret_cast<const __nv_bfloat16*>(dt_bias_ptr),
            ssm_state_vk_ptr,
            norm_w_ptr,
            positions_ptr,
            req_slot_ids_ptr,
            reinterpret_cast<__nv_bfloat16*>(gated_out_ptr),
            stride_qkvzba_row,
            stride_conv_slot,
            stride_ssm_slot,
            stride_ssm_head,
            num_active_rows,
            eps);
    } else {
        dim3 grid(2 * 2, static_cast<unsigned int>(num_rows), 1);
        gdn_decode_fused_cluster2_512_vk_kernel<<<grid, 512, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(qkvzba_ptr),
            reinterpret_cast<__nv_bfloat16*>(conv_state_ptr),
            reinterpret_cast<const __nv_bfloat16*>(conv_w_ptr),
            a_log_ptr,
            reinterpret_cast<const __nv_bfloat16*>(dt_bias_ptr),
            ssm_state_vk_ptr,
            norm_w_ptr,
            positions_ptr,
            req_slot_ids_ptr,
            reinterpret_cast<__nv_bfloat16*>(gated_out_ptr),
            stride_qkvzba_row,
            stride_conv_slot,
            stride_ssm_slot,
            stride_ssm_head,
            num_active_rows,
            eps);
    }
    ZSTEP_CUDA_CHECK(cudaGetLastError());
}

void launch_gdn_decode_split_vk_cuda(
    const void* qkvzba_ptr,
    void* conv_state_ptr,
    const void* conv_w_ptr,
    const float* a_log_ptr,
    const void* dt_bias_ptr,
    float* ssm_state_vk_ptr,
    const float* norm_w_ptr,
    const int32_t* positions_ptr,
    const int32_t* req_slot_ids_ptr,
    float* qk_norm_scratch_ptr,
    void* v_conv_scratch_ptr,
    void* raw_out_ptr,
    void* gated_out_ptr,
    int64_t stride_qkvzba_row,
    int64_t stride_conv_slot,
    int64_t stride_ssm_slot,
    int64_t stride_ssm_head,
    int num_rows,
    int num_active_rows,
    float eps,
    int mode,
    std::uintptr_t stream_handle) {
    if (num_rows <= 0) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    dim3 g_pro(static_cast<unsigned int>(num_rows), 2, 1);
    gdn_decode_prologue_conv_qknorm_kernel<<<g_pro, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkvzba_ptr),
        reinterpret_cast<__nv_bfloat16*>(conv_state_ptr),
        reinterpret_cast<const __nv_bfloat16*>(conv_w_ptr),
        positions_ptr,
        req_slot_ids_ptr,
        qk_norm_scratch_ptr,
        reinterpret_cast<__nv_bfloat16*>(v_conv_scratch_ptr),
        stride_qkvzba_row,
        stride_conv_slot,
        num_active_rows);
    if (mode == 0) {
        // 2-op: Prologue + VHead512 fused recurrence+epilogue
        dim3 g_vh(static_cast<unsigned int>(num_rows), 4, 1);
        gdn_decode_vhead512_epilogue_vk_kernel<<<g_vh, 512, 0, stream>>>(
            qk_norm_scratch_ptr,
            reinterpret_cast<const __nv_bfloat16*>(v_conv_scratch_ptr),
            reinterpret_cast<const __nv_bfloat16*>(qkvzba_ptr),
            a_log_ptr,
            reinterpret_cast<const __nv_bfloat16*>(dt_bias_ptr),
            ssm_state_vk_ptr,
            norm_w_ptr,
            positions_ptr,
            req_slot_ids_ptr,
            reinterpret_cast<__nv_bfloat16*>(gated_out_ptr),
            stride_qkvzba_row,
            stride_ssm_slot,
            stride_ssm_head,
            num_active_rows,
            eps);
    } else {
        // 3-op: Prologue + 32-tile [V,K] recurrence + 1-CTA-per-row GatedRMSNorm
        dim3 g_32(static_cast<unsigned int>(num_rows), 32, 1);
        gdn_decode_32tile_vk_kernel<<<g_32, 128, 0, stream>>>(
            qk_norm_scratch_ptr,
            reinterpret_cast<const __nv_bfloat16*>(v_conv_scratch_ptr),
            reinterpret_cast<const __nv_bfloat16*>(qkvzba_ptr),
            a_log_ptr,
            reinterpret_cast<const __nv_bfloat16*>(dt_bias_ptr),
            ssm_state_vk_ptr,
            positions_ptr,
            req_slot_ids_ptr,
            reinterpret_cast<__nv_bfloat16*>(raw_out_ptr),
            stride_qkvzba_row,
            stride_ssm_slot,
            stride_ssm_head,
            num_active_rows);
        gdn_rmsnorm_gated_silu_decode_cuda_kernel<<<static_cast<unsigned int>(num_rows), 128, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(raw_out_ptr),
            reinterpret_cast<const __nv_bfloat16*>(qkvzba_ptr),
            norm_w_ptr,
            reinterpret_cast<__nv_bfloat16*>(gated_out_ptr),
            stride_qkvzba_row,
            num_active_rows,
            eps);
    }
    ZSTEP_CUDA_CHECK(cudaGetLastError());
}

void launch_gdn_prefill_vk_cuda(
    const void* qkvzba_ptr,
    void* conv_state_ptr,
    const void* conv_w_ptr,
    const float* a_log_ptr,
    const void* dt_bias_ptr,
    float* ssm_state_vk_ptr,
    const float* norm_w_ptr,
    const int32_t* query_start_loc_ptr,
    const int32_t* positions_ptr,
    const int32_t* req_slot_ids_ptr,
    float* qk_norm_scratch_ptr,
    void* v_conv_scratch_ptr,
    void* raw_out_ptr,
    void* gated_out_ptr,
    int64_t stride_qkvzba_row,
    int64_t stride_conv_slot,
    int64_t stride_ssm_slot,
    int64_t stride_ssm_head,
    int num_seqs,
    int total_rows,
    float eps,
    std::uintptr_t stream_handle) {
    if (total_rows <= 0 || num_seqs <= 0) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);

    dim3 grid_conv(static_cast<unsigned int>(num_seqs), 16, 1);
    gdn_conv1d_prefill_varlen_cuda_kernel<<<grid_conv, 64, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkvzba_ptr),
        reinterpret_cast<__nv_bfloat16*>(conv_state_ptr),
        reinterpret_cast<const __nv_bfloat16*>(conv_w_ptr),
        query_start_loc_ptr,
        positions_ptr,
        req_slot_ids_ptr,
        reinterpret_cast<__nv_bfloat16*>(v_conv_scratch_ptr),
        stride_qkvzba_row,
        stride_conv_slot);

    dim3 grid_norm(static_cast<unsigned int>(total_rows), 4, 1);
    gdn_qk_l2norm_rows_kernel<<<grid_norm, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(v_conv_scratch_ptr),
        qk_norm_scratch_ptr,
        total_rows);

    dim3 grid_rec(static_cast<unsigned int>(num_seqs), 32, 1);
    gdn_prefill_recurrent_vk_kernel<<<grid_rec, 128, 0, stream>>>(
        qk_norm_scratch_ptr,
        reinterpret_cast<const __nv_bfloat16*>(v_conv_scratch_ptr),
        reinterpret_cast<const __nv_bfloat16*>(qkvzba_ptr),
        a_log_ptr,
        reinterpret_cast<const __nv_bfloat16*>(dt_bias_ptr),
        ssm_state_vk_ptr,
        query_start_loc_ptr,
        positions_ptr,
        req_slot_ids_ptr,
        reinterpret_cast<__nv_bfloat16*>(raw_out_ptr),
        stride_qkvzba_row,
        stride_ssm_slot,
        stride_ssm_head);

    gdn_rmsnorm_gated_silu_cuda_kernel<<<static_cast<unsigned int>(total_rows), 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(raw_out_ptr),
        reinterpret_cast<const __nv_bfloat16*>(qkvzba_ptr),
        norm_w_ptr,
        reinterpret_cast<__nv_bfloat16*>(gated_out_ptr),
        stride_qkvzba_row,
        total_rows,
        eps);
    ZSTEP_CUDA_CHECK(cudaGetLastError());
}
"""


_EXT = None


def _get_ext():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="zstep_route_a_bench_v7",
            cpp_sources=[_ROUTE_A_CPP],
            cuda_sources=[_ROUTE_A_CUDA],
            functions=[
                "launch_shm_v5_ar_legacy_interleaved",
                "launch_shm_v5_ar_batchload",
                "launch_shm_v5_ar_add_rmsnorm",
                "launch_gdn_decode_fused_vk",
                "launch_gdn_decode_split_vk",
                "launch_gdn_prefill_vk",
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_120"],
            with_cuda=True,
            verbose=False,
        )
    return _EXT


def _bf16_ulp_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    af = a.float()
    bf = b.float()
    abs_diff = (af - bf).abs()
    max_abs = float(abs_diff.max().item()) if abs_diff.numel() else 0.0
    mean_abs = float(abs_diff.mean().item()) if abs_diff.numel() else 0.0
    au = a.view(torch.int16).to(torch.int32)
    bu = b.view(torch.int16).to(torch.int32)
    # Signed magnitude conversion for IEEE/BF16 ULP distance
    au_sm = torch.where(au < 0, 0x8000 - au, au)
    bu_sm = torch.where(bu < 0, 0x8000 - bu, bu)
    ulp = (au_sm - bu_sm).abs()
    # Treat -0.0 and +0.0 as 0 ULP
    both_zero = (af == 0.0) & (bf == 0.0)
    ulp = torch.where(both_zero, torch.zeros_like(ulp), ulp)
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_ulp": int(ulp.max().item()) if ulp.numel() else 0,
        "frac_gt_0_ulp": float((ulp > 0).float().mean().item()) if ulp.numel() else 0.0,
        "frac_gt_1_ulp": float((ulp > 1).float().mean().item()) if ulp.numel() else 0.0,
    }


def _time_fn_us(fn, *, warmup: int = 25, reps: int = 120) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    lat = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends, strict=False))
    n = len(lat)
    return {
        "p50_us": round(lat[n // 2], 2),
        "p90_us": round(lat[min(n - 1, int(n * 0.90))], 2),
        "p99_us": round(lat[min(n - 1, int(n * 0.99))], 2),
        "mean_us": round(sum(lat) / n, 2),
    }


# ===========================================================================
# Part A1: 8-GPU TP=8 `shm_v5` + `fused_add_zero_centered_rmsnorm` Benchmark
# ===========================================================================


def _a1_worker(rank: int, world_size: int, port: int, shm_prefix: str, out_json_path: str):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=5),
    )

    from zstep_worker.kernels_sm120.qwen35_ops import fused_add_zero_centered_rmsnorm
    from zstep_worker.kernels_sm120.shm_allreduce import HostMappedShmAllReduce
    from flashinfer.norm import fused_add_rmsnorm as fi_fused_add_rmsnorm

    ext = _get_ext()
    dist.barrier()

    comm = HostMappedShmAllReduce(
        name=shm_prefix,
        rank=rank,
        world_size=world_size,
        capacity_elements=1024 * 5120,
        blocks=8,
        threads=256,
        exchange_group=dist.group.WORLD,
    )
    dist.barrier()

    def _run_epilogue_2op(delta: torch.Tensor, res: torch.Tensor, w: torch.Tensor, out_buf: torch.Tensor, h_dim: int):
        if h_dim == 2048:
            fused_add_zero_centered_rmsnorm(delta, res, w, normed_out=out_buf, eps=eps)
        else:
            # Standard RMSNorm (`weight_offset = 0.0`) used by DSV4.1 H=5120:
            # `fi_fused_add_rmsnorm` mutates `delta` -> normed_out and `res` -> residual!
            tmp_norm = delta.clone()
            fi_fused_add_rmsnorm(tmp_norm, res, w, eps)
            out_buf.copy_(tmp_norm)

    def _run_epilogue_2op_fast(delta_mut: torch.Tensor, res: torch.Tensor, w: torch.Tensor, out_buf: torch.Tensor, h_dim: int):
        if h_dim == 2048:
            fused_add_zero_centered_rmsnorm(delta_mut, res, w, normed_out=out_buf, eps=eps)
        else:
            fi_fused_add_rmsnorm(delta_mut, res, w, eps)

    eps = 1e-6
    results = {"parity": {}, "lockstep_mixed_m": {}, "latency": {}}

    try:
        for H in (2048, 5120):
            w_offset = 1.0 if H == 2048 else 0.0
            results["parity"][f"H={H}"] = {}
            results["latency"][f"H={H}"] = {}
            g_w = torch.Generator(device="cpu").manual_seed(20261114 + H)
            w_init = torch.randn((H,), generator=g_w, dtype=torch.float32) * 0.1
            if H == 5120:
                w_init = w_init + 1.0
            weight = w_init.to(device=device, dtype=torch.bfloat16)

            # 1. Exact parity check across M=1,2,4,8,16,32,64,128,256,512
            for M in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
                g_in = torch.Generator(device="cpu").manual_seed(1000 * M + rank + H)
                g_res = torch.Generator(device="cpu").manual_seed(9000 * M + H)
                inp = torch.randn((M, H), generator=g_in, dtype=torch.float32).to(
                    device=device, dtype=torch.bfloat16
                )
                res_base = torch.randn((M, H), generator=g_res, dtype=torch.float32).to(
                    device=device, dtype=torch.bfloat16
                )
                res_bl = res_base.clone()
                res_f1 = res_base.clone()
                res_fc8 = res_base.clone()

                ar_tmp = torch.empty_like(inp)
                out_base = torch.empty_like(inp)
                out_bl = torch.empty_like(inp)
                out_f1 = torch.empty_like(inp)
                out_fc8 = torch.empty_like(inp)

                stream_ptr = int(torch.cuda.current_stream(device).cuda_stream)

                # Baseline 2-op (historical pre-PR-#20 interleaved-load `shm_v5_allreduce`)
                ext.launch_shm_v5_ar_legacy_interleaved(
                    inp,
                    ar_tmp,
                    comm.control,
                    comm._rank_shm_ptrs,
                    comm.slot_stride_bytes,
                    rank,
                    world_size,
                    comm.blocks,
                    comm.threads,
                    comm.one_shot_max_vecs,
                    stream_ptr,
                )
                _run_epilogue_2op(ar_tmp, res_base, weight, out_base, H)

                # Batched-load 2-op
                ext.launch_shm_v5_ar_batchload(
                    inp,
                    ar_tmp,
                    comm.control,
                    comm._rank_shm_ptrs,
                    comm.slot_stride_bytes,
                    rank,
                    world_size,
                    comm.blocks,
                    comm.threads,
                    comm.one_shot_max_vecs,
                    stream_ptr,
                )
                _run_epilogue_2op(ar_tmp, res_bl, weight, out_bl, H)

                # Fused 1-op (non-cluster)
                ext.launch_shm_v5_ar_add_rmsnorm(
                    inp,
                    res_f1,
                    weight,
                    out_f1,
                    comm.control,
                    comm._rank_shm_ptrs,
                    comm.slot_stride_bytes,
                    eps,
                    w_offset,
                    rank,
                    world_size,
                    comm.blocks,
                    comm.threads,
                    comm.one_shot_max_vecs,
                    0,
                    stream_ptr,
                )

                # Fused 1-op Cluster-8 (differs for M < 8)
                ext.launch_shm_v5_ar_add_rmsnorm(
                    inp,
                    res_fc8,
                    weight,
                    out_fc8,
                    comm.control,
                    comm._rank_shm_ptrs,
                    comm.slot_stride_bytes,
                    eps,
                    w_offset,
                    rank,
                    world_size,
                    comm.blocks,
                    comm.threads,
                    comm.one_shot_max_vecs,
                    1,
                    stream_ptr,
                )
                torch.cuda.synchronize()
                dist.barrier()

                results["parity"][f"H={H}"][f"M={M}"] = {
                    "batchload_res": _bf16_ulp_stats(res_base, res_bl),
                    "batchload_out": _bf16_ulp_stats(out_base, out_bl),
                    "fused_1op_res": _bf16_ulp_stats(res_base, res_f1),
                    "fused_1op_out": _bf16_ulp_stats(out_base, out_f1),
                    "fused_cluster8_res": _bf16_ulp_stats(res_base, res_fc8),
                    "fused_cluster8_out": _bf16_ulp_stats(out_base, out_fc8),
                }

            # 2. Mixed-M interleaved `all_reduce` + `ar_add_rmsnorm` lockstep test (64 steps)
            mixed_ms = [1, 16, 4, 64, 2, 128, 32, 8] * 8
            ctrl_before = comm.control[: comm.blocks].clone()
            stream_ptr = int(torch.cuda.current_stream(device).cuda_stream)
            for step_i, m_val in enumerate(mixed_ms):
                x_m = torch.ones((m_val, H), device=device, dtype=torch.bfloat16) * ((rank + 1) * 0.125)
                r_m = torch.ones((m_val, H), device=device, dtype=torch.bfloat16) * 0.5
                o_m = torch.empty_like(x_m)
                if step_i % 3 == 0:
                    comm.all_reduce(x_m, output=o_m)
                elif step_i % 3 == 1:
                    ext.launch_shm_v5_ar_add_rmsnorm(
                        x_m, r_m, weight, o_m,
                        comm.control, comm._rank_shm_ptrs, comm.slot_stride_bytes,
                        eps, w_offset, rank, world_size, comm.blocks, comm.threads,
                        comm.one_shot_max_vecs, 0, stream_ptr,
                    )
                else:
                    ext.launch_shm_v5_ar_add_rmsnorm(
                        x_m, r_m, weight, o_m,
                        comm.control, comm._rank_shm_ptrs, comm.slot_stride_bytes,
                        eps, w_offset, rank, world_size, comm.blocks, comm.threads,
                        comm.one_shot_max_vecs, 1, stream_ptr,
                    )
            torch.cuda.synchronize()
            dist.barrier()
            ctrl_after = comm.control[: comm.blocks].cpu().tolist()
            ctrl_delta = (comm.control[: comm.blocks] - ctrl_before).cpu().tolist()
            results["lockstep_mixed_m"][f"H={H}"] = {
                "all_blocks_equal": len(set(ctrl_after)) == 1,
                "expected_delta": len(mixed_ms),
                "actual_deltas": ctrl_delta,
            }

            # 3. Latency benchmark (CUDAGraph 10-op replay to amortize graph launch overhead accurately)
            n_ops_in_graph = 10
            for M in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
                inp = torch.randn((M, H), device=device, dtype=torch.bfloat16) * 0.1
                res = torch.randn((M, H), device=device, dtype=torch.bfloat16) * 0.1
                ar_tmp = torch.empty_like(inp)
                out = torch.empty_like(inp)

                mode_lat = {}
                for mode_name in ("base_2op", "batchload_2op", "fused_1op", "fused_cluster8"):
                    if mode_name == "fused_cluster8" and M >= 8:
                        continue
                    dist.barrier()
                    torch.cuda.synchronize()

                    if M <= 256:
                        g_stream = torch.cuda.Stream(device=device)
                        with torch.cuda.stream(g_stream):
                            s_ptr = int(g_stream.cuda_stream)
                            for _ in range(3):
                                if mode_name == "base_2op":
                                    ext.launch_shm_v5_ar_legacy_interleaved(
                                        inp, ar_tmp, comm.control, comm._rank_shm_ptrs,
                                        comm.slot_stride_bytes, rank, world_size, comm.blocks,
                                        comm.threads, comm.one_shot_max_vecs, s_ptr,
                                    )
                                    _run_epilogue_2op_fast(ar_tmp, res, weight, out, H)
                                elif mode_name == "batchload_2op":
                                    ext.launch_shm_v5_ar_batchload(
                                        inp, ar_tmp, comm.control, comm._rank_shm_ptrs,
                                        comm.slot_stride_bytes, rank, world_size, comm.blocks,
                                        comm.threads, comm.one_shot_max_vecs, s_ptr,
                                    )
                                    _run_epilogue_2op_fast(ar_tmp, res, weight, out, H)
                                elif mode_name == "fused_1op":
                                    ext.launch_shm_v5_ar_add_rmsnorm(
                                        inp, res, weight, out, comm.control, comm._rank_shm_ptrs,
                                        comm.slot_stride_bytes, eps, w_offset, rank, world_size,
                                        comm.blocks, comm.threads, comm.one_shot_max_vecs, 0, s_ptr,
                                    )
                                else:
                                    ext.launch_shm_v5_ar_add_rmsnorm(
                                        inp, res, weight, out, comm.control, comm._rank_shm_ptrs,
                                        comm.slot_stride_bytes, eps, w_offset, rank, world_size,
                                        comm.blocks, comm.threads, comm.one_shot_max_vecs, 1, s_ptr,
                                    )
                        g_stream.synchronize()
                        dist.barrier()

                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=g_stream):
                            s_ptr = int(g_stream.cuda_stream)
                            for _ in range(n_ops_in_graph):
                                if mode_name == "base_2op":
                                    ext.launch_shm_v5_ar_legacy_interleaved(
                                        inp, ar_tmp, comm.control, comm._rank_shm_ptrs,
                                        comm.slot_stride_bytes, rank, world_size, comm.blocks,
                                        comm.threads, comm.one_shot_max_vecs, s_ptr,
                                    )
                                    _run_epilogue_2op_fast(ar_tmp, res, weight, out, H)
                                elif mode_name == "batchload_2op":
                                    ext.launch_shm_v5_ar_batchload(
                                        inp, ar_tmp, comm.control, comm._rank_shm_ptrs,
                                        comm.slot_stride_bytes, rank, world_size, comm.blocks,
                                        comm.threads, comm.one_shot_max_vecs, s_ptr,
                                    )
                                    _run_epilogue_2op_fast(ar_tmp, res, weight, out, H)
                                elif mode_name == "fused_1op":
                                    ext.launch_shm_v5_ar_add_rmsnorm(
                                        inp, res, weight, out, comm.control, comm._rank_shm_ptrs,
                                        comm.slot_stride_bytes, eps, w_offset, rank, world_size,
                                        comm.blocks, comm.threads, comm.one_shot_max_vecs, 0, s_ptr,
                                    )
                                else:
                                    ext.launch_shm_v5_ar_add_rmsnorm(
                                        inp, res, weight, out, comm.control, comm._rank_shm_ptrs,
                                        comm.slot_stride_bytes, eps, w_offset, rank, world_size,
                                        comm.blocks, comm.threads, comm.one_shot_max_vecs, 1, s_ptr,
                                    )
                        torch.cuda.synchronize()
                        dist.barrier()
                        raw_stats = _time_fn_us(lambda: graph.replay(), warmup=30, reps=150)
                        mode_lat[mode_name] = {
                            k: round(v / n_ops_in_graph, 2) for k, v in raw_stats.items()
                        }
                        mode_lat[mode_name]["exec_mode"] = "cudagraph_x10"
                    else:
                        s_ptr = int(torch.cuda.current_stream(device).cuda_stream)

                        def _eager_step():
                            for _ in range(4):
                                if mode_name == "base_2op":
                                    ext.launch_shm_v5_ar_legacy_interleaved(
                                        inp, ar_tmp, comm.control, comm._rank_shm_ptrs,
                                        comm.slot_stride_bytes, rank, world_size, comm.blocks,
                                        comm.threads, comm.one_shot_max_vecs, s_ptr,
                                    )
                                    _run_epilogue_2op_fast(ar_tmp, res, weight, out, H)
                                elif mode_name == "batchload_2op":
                                    ext.launch_shm_v5_ar_batchload(
                                        inp, ar_tmp, comm.control, comm._rank_shm_ptrs,
                                        comm.slot_stride_bytes, rank, world_size, comm.blocks,
                                        comm.threads, comm.one_shot_max_vecs, s_ptr,
                                    )
                                    _run_epilogue_2op_fast(ar_tmp, res, weight, out, H)
                                else:
                                    ext.launch_shm_v5_ar_add_rmsnorm(
                                        inp, res, weight, out, comm.control, comm._rank_shm_ptrs,
                                        comm.slot_stride_bytes, eps, w_offset, rank, world_size,
                                        comm.blocks, comm.threads, comm.one_shot_max_vecs, 0, s_ptr,
                                    )

                        dist.barrier()
                        raw_stats = _time_fn_us(_eager_step, warmup=20, reps=80)
                        mode_lat[mode_name] = {k: round(v / 4.0, 2) for k, v in raw_stats.items()}
                        mode_lat[mode_name]["exec_mode"] = "eager_x4"

                results["latency"][f"H={H}"][f"M={M}"] = mode_lat

        if rank == 0:
            with open(out_json_path, "w") as f:
                json.dump(results, f, indent=2)
    finally:
        dist.barrier()
        comm.close()
        dist.destroy_process_group()


def run_part_a1() -> dict:
    world_size = 8
    port = 29641
    shm_prefix = f"zstep_ra_bench_{os.getpid()}"
    out_json_path = f"/tmp/route_a_part1_{os.getpid()}.json"
    # Pre-compile extension once in parent process before spawning 8 ranks!
    _get_ext()
    mp.spawn(
        _a1_worker,
        args=(world_size, port, shm_prefix, out_json_path),
        nprocs=world_size,
        join=True,
    )
    with open(out_json_path) as f:
        data = json.load(f)
    os.remove(out_json_path)
    return data


# ===========================================================================
# Part A2: Single-GPU SM120 30-Layer GatedDeltaNet Decode & Prefill Benchmark
# ===========================================================================


@triton.jit
def _legacy_gated_delta_decode_kv_kernel(
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
    """Historical pre-PR-#20 (`893495c`) `[K=128, V=128]` Triton decode recurrence."""
    row = tl.program_id(0).to(tl.int64)
    hv_tile = tl.program_id(1).to(tl.int64)
    v_head = hv_tile // (128 // BLOCK_V)
    v_blk = hv_tile % (128 // BLOCK_V)
    k_head = v_head // 2

    offs_k = tl.arange(0, 128)
    offs_v = v_blk * BLOCK_V + tl.arange(0, BLOCK_V)

    if row >= num_active_rows:
        tl.store(
            attn_out_ptr + row * 512 + v_head * 128 + offs_v,
            tl.zeros([BLOCK_V], dtype=tl.bfloat16),
        )
        return

    slot = tl.load(req_slot_ids_ptr + row).to(tl.int64)
    pos = tl.load(positions_ptr + row).to(tl.int32)

    q = tl.load(qkv_conv_ptr + row * 1024 + k_head * 128 + offs_k).to(tl.float32)
    k = tl.load(qkv_conv_ptr + row * 1024 + 256 + k_head * 128 + offs_k).to(tl.float32)
    v = tl.load(qkv_conv_ptr + row * 1024 + 512 + v_head * 128 + offs_v).to(tl.float32)

    q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1.0e-6) * 0.08838834764831845
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
def _legacy_gated_delta_prefill_varlen_kv_kernel(
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
    """Historical pre-PR-#20 (`893495c`) `[K=128, V=128]` Triton varlen prefill recurrence."""
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


def _transpose_ssm_kv_to_vk(ssm_kv: torch.Tensor) -> torch.Tensor:
    # ssm_kv: [..., 4, K=128, V=128] -> ssm_vk: [..., 4, V=128, K=128] contiguous
    return ssm_kv.transpose(-1, -2).contiguous()


def run_part_a2() -> dict:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    ext = _get_ext()

    from zstep_worker.kernels_sm120.qwen35_ops import (
        _causal_conv1d_decode_kernel,
        _causal_conv1d_prefill_varlen_kernel,
        _rmsnorm_gated_silu_kernel,
    )

    def _legacy_triton_gdn_decode_kv(
        qkvzba: torch.Tensor,
        *,
        conv_state: torch.Tensor,
        conv_weight: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_state: torch.Tensor,
        norm_weight: torch.Tensor,
        positions: torch.Tensor,
        req_slot_ids: torch.Tensor,
        qkv_conv_buf: torch.Tensor,
        raw_out_buf: torch.Tensor,
        gated_out_buf: torch.Tensor,
        num_active_rows: int,
        eps: float,
    ) -> torch.Tensor:
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
        _legacy_gated_delta_decode_kv_kernel[(batch_rows, 4 * (128 // 16))](
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

    def _legacy_triton_gdn_prefill_kv(
        qkvzba: torch.Tensor,
        *,
        conv_state: torch.Tensor,
        conv_weight: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_state: torch.Tensor,
        norm_weight: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        req_slot_ids: torch.Tensor,
        qkv_conv_buf: torch.Tensor,
        raw_out_buf: torch.Tensor,
        gated_out_buf: torch.Tensor,
        num_seqs: int,
        eps: float,
    ) -> torch.Tensor:
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
        _legacy_gated_delta_prefill_varlen_kv_kernel[(num_seqs, 4 * (128 // 4))](
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

    def _conv1d_prefill_helper(qkvzba_t, conv_st_t, conv_w_t, q_locs_t, pos_t, slots_t, n_s: int, out_buf_t):
        _causal_conv1d_prefill_varlen_kernel[(n_s, 1024 // 32)](
            qkvzba_t,
            conv_st_t,
            conv_w_t,
            q_locs_t,
            pos_t,
            slots_t,
            out_buf_t,
            qkvzba_t.stride(0),
            conv_st_t.stride(0),
            BLOCK_C=32,
            num_warps=2,
        )
        return out_buf_t

    def _rmsnorm_gated_helper(raw_x_t, qkvzba_t, norm_w_t, eps_val: float, out_buf_t):
        rows_n = raw_x_t.shape[0]
        _rmsnorm_gated_silu_kernel[(rows_n, 4)](
            raw_x_t,
            qkvzba_t,
            norm_w_t,
            out_buf_t,
            qkvzba_t.stride(0),
            eps_val,
            rows_n,
            num_warps=4,
        )
        return out_buf_t

    # Probe optional FlashInfer experimental SM120 GDN modules
    has_gdn_cute = False
    chunk_gated_delta_rule = None
    try:
        from flashinfer.experimental.gdn_sm120_cute._api import chunk_gated_delta_rule as _cgdr

        chunk_gated_delta_rule = _cgdr
        has_gdn_cute = True
    except Exception as exc:
        gdn_cute_err = str(exc)
    else:
        gdn_cute_err = None

    b12x_probe = {}
    try:
        import subprocess as _sp

        _probe_code = (
            "import torch\n"
            "from flashinfer.experimental.b12x.sequence.gdn_decode import gated_delta_net_decode as f\n"
            "d = torch.device('cuda:0')\n"
            "f(a=torch.zeros((1,4),device=d,dtype=torch.bfloat16), b=torch.zeros((1,4),device=d,dtype=torch.bfloat16),"
            " g=torch.zeros((4,),device=d,dtype=torch.float32), dt_bias=torch.zeros((4,),device=d,dtype=torch.bfloat16),"
            " q=torch.randn((1,1,2,128),device=d,dtype=torch.bfloat16), k=torch.randn((1,1,2,128),device=d,dtype=torch.bfloat16),"
            " v=torch.randn((1,1,4,128),device=d,dtype=torch.bfloat16), state=torch.zeros((2,4,128,128),device=d,dtype=torch.float32),"
            " cache_indices=torch.tensor([0],device=d,dtype=torch.int32))\n"
            "torch.cuda.synchronize()\n"
            "print('OK')\n"
        )
        _cp = _sp.run([sys.executable, "-c", _probe_code], capture_output=True, text=True, timeout=45)
        b12x_probe["hk2_hv4_supported"] = (_cp.returncode == 0 and "OK" in _cp.stdout)
        if _cp.returncode != 0:
            b12x_probe["hk2_hv4_error"] = (_cp.stderr or _cp.stdout)[-400:]
    except Exception as exc:
        b12x_probe["hk2_hv4_supported"] = False
        b12x_probe["error"] = str(exc)

    eps = 1e-6
    results = {
        "probes": {
            "has_gdn_sm120_cute": has_gdn_cute,
            "gdn_sm120_cute_error": gdn_cute_err,
            "b12x_gdn_decode": b12x_probe,
        },
        "decode_parity": {},
        "prefill_parity": {},
        "decode_30layers_latency": {},
        "prefill_30layers_latency": {},
    }

    torch.manual_seed(20261114)
    total_slots = 256
    conv_w = (torch.randn((1024, 4), device=device, dtype=torch.float32) * 0.15).to(torch.bfloat16)
    a_log = torch.linspace(-2.0, -0.5, 4, device=device, dtype=torch.float32)
    dt_bias = torch.linspace(-0.2, 0.2, 4, device=device, dtype=torch.bfloat16)
    norm_w = (1.0 + 0.1 * torch.randn((128,), device=device, dtype=torch.float32))

    # 1. Decode numerical parity (`pos=0`, `pos>0`, and padded `num_active_rows < B`)
    for case_name, B, active_B, pos_list in [
        ("B1_pos0", 1, 1, [0]),
        ("B8_mixed_pos", 8, 8, [0, 1, 5, 17, 0, 33, 64, 12]),
        ("B16_padded_active11", 16, 11, [3, 0, 9, 4, 1, 0, 7, 15, 2, 8, 6, 0, 0, 0, 0, 0]),
    ]:
        qkvzba = (torch.randn((B, 1544), device=device, dtype=torch.float32) * 0.25).to(torch.bfloat16)
        positions = torch.tensor(pos_list, device=device, dtype=torch.int32)
        req_slots = torch.arange(10, 10 + B, device=device, dtype=torch.int32)

        conv_init = (torch.randn((total_slots, 1024, 3), device=device, dtype=torch.float32) * 0.2).to(torch.bfloat16)
        ssm_kv_init = torch.randn((total_slots, 4, 128, 128), device=device, dtype=torch.float32) * 0.05
        ssm_vk_init = _transpose_ssm_kv_to_vk(ssm_kv_init)

        conv_ref = conv_init.clone()
        ssm_ref = ssm_kv_init.clone()
        qkv_scr = torch.empty((B, 1024), device=device, dtype=torch.bfloat16)
        raw_scr = torch.empty((B, 512), device=device, dtype=torch.bfloat16)
        out_ref = torch.empty((B, 512), device=device, dtype=torch.bfloat16)

        _legacy_triton_gdn_decode_kv(
            qkvzba,
            conv_state=conv_ref,
            conv_weight=conv_w,
            a_log=a_log,
            dt_bias=dt_bias,
            ssm_state=ssm_ref,
            norm_weight=norm_w,
            positions=positions,
            req_slot_ids=req_slots,
            qkv_conv_buf=qkv_scr,
            raw_out_buf=raw_scr,
            gated_out_buf=out_ref,
            num_active_rows=active_B,
            eps=eps,
        )

        case_res = {}
        s_ptr = int(torch.cuda.current_stream(device).cuda_stream)
        qk_dec_scr = torch.empty((B, 512), device=device, dtype=torch.float32)
        v_dec_scr = torch.empty((B, 512), device=device, dtype=torch.bfloat16)
        for var_id, var_name in [(0, "1cta"), (1, "cluster4"), (2, "cluster8"), (3, "cluster2_512"), (10, "split_2op_vh512"), (11, "split_3op_32tile")]:
            conv_cand = conv_init.clone()
            ssm_cand = ssm_vk_init.clone()
            out_cand = torch.full((B, 512), -99.0, device=device, dtype=torch.bfloat16)
            if var_id < 10:
                ext.launch_gdn_decode_fused_vk(
                    qkvzba, conv_cand, conv_w, a_log, dt_bias, ssm_cand, norm_w,
                    positions, req_slots, out_cand, active_B, eps, var_id, s_ptr,
                )
            else:
                ext.launch_gdn_decode_split_vk(
                    qkvzba, conv_cand, conv_w, a_log, dt_bias, ssm_cand, norm_w,
                    positions, req_slots, qk_dec_scr, v_dec_scr, raw_scr, out_cand,
                    active_B, eps, var_id - 10, s_ptr,
                )
            torch.cuda.synchronize()
            ssm_ref_vk = _transpose_ssm_kv_to_vk(ssm_ref)
            ssm_max_abs = float((ssm_cand - ssm_ref_vk).abs().max().item())
            conv_ulp = _bf16_ulp_stats(conv_ref, conv_cand)
            out_ulp = _bf16_ulp_stats(out_ref, out_cand)
            case_res[var_name] = {
                "ssm_state_max_abs_f32": ssm_max_abs,
                "conv_state": conv_ulp,
                "gated_out": out_ulp,
            }
        results["decode_parity"][case_name] = case_res

    # 2. Prefill numerical parity (including multi-seq varlen + prefill vs stepwise decode!)
    for pf_name, seq_lens, start_positions in [
        ("varlen_3seqs_mixed_pos", [13, 64, 29], [0, 7, 0]),
        ("single_512_pos0", [512], [0]),
    ]:
        total_rows = sum(seq_lens)
        num_seqs = len(seq_lens)
        q_locs = [0]
        pos_flat = []
        slots_flat = []
        for idx, (slen, spos) in enumerate(zip(seq_lens, start_positions, strict=False)):
            q_locs.append(q_locs[-1] + slen)
            pos_flat.extend(range(spos, spos + slen))
            slots_flat.extend([20 + idx] * slen)

        qkvzba = (torch.randn((total_rows, 1544), device=device, dtype=torch.float32) * 0.25).to(torch.bfloat16)
        query_start_loc = torch.tensor(q_locs, device=device, dtype=torch.int32)
        positions = torch.tensor(pos_flat, device=device, dtype=torch.int32)
        req_slots = torch.tensor(slots_flat, device=device, dtype=torch.int32)

        conv_init = (torch.randn((total_slots, 1024, 3), device=device, dtype=torch.float32) * 0.2).to(torch.bfloat16)
        ssm_kv_init = torch.randn((total_slots, 4, 128, 128), device=device, dtype=torch.float32) * 0.05
        ssm_vk_init = _transpose_ssm_kv_to_vk(ssm_kv_init)

        conv_ref = conv_init.clone()
        ssm_ref = ssm_kv_init.clone()
        qkv_ref_buf = torch.empty((total_rows, 1024), device=device, dtype=torch.bfloat16)
        raw_ref_buf = torch.empty((total_rows, 512), device=device, dtype=torch.bfloat16)
        out_ref = torch.empty((total_rows, 512), device=device, dtype=torch.bfloat16)
        _legacy_triton_gdn_prefill_kv(
            qkvzba,
            conv_state=conv_ref,
            conv_weight=conv_w,
            a_log=a_log,
            dt_bias=dt_bias,
            ssm_state=ssm_ref,
            norm_weight=norm_w,
            query_start_loc=query_start_loc,
            positions=positions,
            req_slot_ids=req_slots,
            qkv_conv_buf=qkv_ref_buf,
            raw_out_buf=raw_ref_buf,
            gated_out_buf=out_ref,
            num_seqs=num_seqs,
            eps=eps,
        )

        # Candidate 1: CUDA `[V,K]` Recurrent Prefill
        conv_c1 = conv_init.clone()
        ssm_c1 = ssm_vk_init.clone()
        qk_norm_scr = torch.empty((total_rows, 512), device=device, dtype=torch.float32)
        v_conv_scr = torch.empty((total_rows, 1024), device=device, dtype=torch.bfloat16)
        raw_out_scr = torch.empty((total_rows, 512), device=device, dtype=torch.bfloat16)
        out_c1 = torch.empty((total_rows, 512), device=device, dtype=torch.bfloat16)
        s_ptr = int(torch.cuda.current_stream(device).cuda_stream)
        ext.launch_gdn_prefill_vk(
            qkvzba, conv_c1, conv_w, a_log, dt_bias, ssm_c1, norm_w,
            query_start_loc, positions, req_slots,
            qk_norm_scr, v_conv_scr, raw_out_scr, out_c1,
            num_seqs, eps, s_ptr,
        )
        torch.cuda.synchronize()

        pf_entry = {
            "cuda_vk_recurrent": {
                "ssm_state_max_abs_f32": float((ssm_c1 - _transpose_ssm_kv_to_vk(ssm_ref)).abs().max().item()),
                "conv_state": _bf16_ulp_stats(conv_ref, conv_c1),
                "gated_out": _bf16_ulp_stats(out_ref, out_c1),
            }
        }

        # Stepwise fused decode equivalence check on `varlen_3seqs_mixed_pos`
        if pf_name == "varlen_3seqs_mixed_pos":
            conv_step = conv_init.clone()
            ssm_step = ssm_vk_init.clone()
            step_outs = []
            for r_idx in range(total_rows):
                o_r = torch.empty((1, 512), device=device, dtype=torch.bfloat16)
                ext.launch_gdn_decode_fused_vk(
                    qkvzba[r_idx : r_idx + 1],
                    conv_step, conv_w, a_log, dt_bias, ssm_step, norm_w,
                    positions[r_idx : r_idx + 1],
                    req_slots[r_idx : r_idx + 1],
                    o_r, 1, eps, 0, s_ptr,
                )
                step_outs.append(o_r)
            torch.cuda.synchronize()
            out_step_cat = torch.cat(step_outs, dim=0)
            pf_entry["prefill_vk_vs_stepwise_decode_1cta"] = {
                "ssm_state_max_abs_f32": float((ssm_c1 - ssm_step).abs().max().item()),
                "conv_state": _bf16_ulp_stats(conv_c1, conv_step),
                "gated_out": _bf16_ulp_stats(out_c1, out_step_cat),
            }

        # Candidate 2: FlashInfer `gdn_sm120_cute` (`chunk_gated_delta_rule`)
        if has_gdn_cute:
            conv_c2 = conv_init.clone()
            ssm_c2 = ssm_vk_init.clone()
            qkv_conv = torch.empty((total_rows, 1024), device=device, dtype=torch.bfloat16)
            _conv1d_prefill_helper(
                qkvzba, conv_c2, conv_w, query_start_loc, positions, req_slots, num_seqs, qkv_conv
            )
            seq_slot_ids = [20 + i for i in range(num_seqs)]
            init_st = torch.zeros((num_seqs, 4, 128, 128), device=device, dtype=torch.float32)
            for i, (spos, sl) in enumerate(zip(start_positions, seq_slot_ids, strict=False)):
                if spos != 0:
                    init_st[i].copy_(ssm_c2[sl])
            q_3d = qkv_conv[:, :256].view(1, total_rows, 2, 128)
            k_3d = qkv_conv[:, 256:512].view(1, total_rows, 2, 128)
            v_3d = qkv_conv[:, 512:1024].view(1, total_rows, 4, 128)
            b_raw = qkvzba[:, 1536:1540].float()
            a_raw = qkvzba[:, 1540:1544].float()
            beta_3d = torch.sigmoid(b_raw).unsqueeze(0)
            g_3d = (-torch.exp(a_log) * torch.nn.functional.softplus(a_raw + dt_bias.float())).unsqueeze(0)
            cu_seqlens = query_start_loc.to(torch.int64)
            raw_cute, final_st = chunk_gated_delta_rule(
                q=q_3d,
                k=k_3d,
                v=v_3d,
                g=g_3d,
                beta=beta_3d,
                scale=128**-0.5,
                initial_state=init_st,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
                state_v_first=True,
            )
            for i, sl in enumerate(seq_slot_ids):
                ssm_c2[sl].copy_(final_st[i])
            out_c2 = torch.empty((total_rows, 512), device=device, dtype=torch.bfloat16)
            _rmsnorm_gated_helper(raw_cute.view(total_rows, 512), qkvzba, norm_w, eps, out_c2)
            torch.cuda.synchronize()
            active_slots_tensor = torch.tensor(seq_slot_ids, device=device, dtype=torch.long)
            pf_entry["flashinfer_gdn_sm120_cute"] = {
                "ssm_state_max_abs_f32": float(
                    (ssm_c2[active_slots_tensor] - _transpose_ssm_kv_to_vk(ssm_ref)[active_slots_tensor])
                    .abs()
                    .max()
                    .item()
                ),
                "conv_state": _bf16_ulp_stats(conv_ref, conv_c2),
                "gated_out": _bf16_ulp_stats(out_ref, out_c2),
            }

        results["prefill_parity"][pf_name] = pf_entry

    # 3. 30-Layer CUDAGraph Decode Latency Benchmark (`B = 1, 2, 4, 8, 16, 32, 64, 128`)
    n_layers = 30
    conv_states_30 = torch.randn((n_layers, total_slots, 1024, 3), device=device, dtype=torch.bfloat16) * 0.1
    ssm_states_30 = torch.randn((n_layers, total_slots, 4, 128, 128), device=device, dtype=torch.float32) * 0.02
    conv_ws_30 = torch.randn((n_layers, 1024, 4), device=device, dtype=torch.bfloat16) * 0.1
    a_logs_30 = torch.full((n_layers, 4), -1.0, device=device, dtype=torch.float32)
    dt_biases_30 = torch.zeros((n_layers, 4), device=device, dtype=torch.bfloat16)
    norm_ws_30 = torch.ones((n_layers, 128), device=device, dtype=torch.float32)

    for B in (1, 2, 4, 8, 16, 32, 64, 128):
        qkvzba_b = torch.randn((B, 1544), device=device, dtype=torch.bfloat16) * 0.1
        positions_b = torch.full((B,), 64, device=device, dtype=torch.int32)
        slots_b = torch.arange(0, B, device=device, dtype=torch.int32)
        qkv_scr_b = torch.empty((B, 1024), device=device, dtype=torch.bfloat16)
        qk_norm_b = torch.empty((B, 512), device=device, dtype=torch.float32)
        v_conv_b = torch.empty((B, 512), device=device, dtype=torch.bfloat16)
        raw_scr_b = torch.empty((B, 512), device=device, dtype=torch.bfloat16)
        out_b = torch.empty((B, 512), device=device, dtype=torch.bfloat16)

        b_lat = {}
        for mode_name, var_id in [
            ("triton_3op_kv", -1),
            ("cuda_1cta_vk", 0),
            ("cuda_cluster4_vk", 1),
            ("cuda_cluster8_vk", 2),
            ("cuda_cluster2_512_vk", 3),
            ("cuda_split_2op_vh512_vk", 10),
            ("cuda_split_3op_32tile_vk", 11),
        ]:
            g_stream = torch.cuda.Stream(device=device)

            def _run_30(s_ptr: int):
                for l_idx in range(n_layers):
                    if var_id < 0:
                        _legacy_triton_gdn_decode_kv(
                            qkvzba_b,
                            conv_state=conv_states_30[l_idx],
                            conv_weight=conv_ws_30[l_idx],
                            a_log=a_logs_30[l_idx],
                            dt_bias=dt_biases_30[l_idx],
                            ssm_state=ssm_states_30[l_idx],
                            norm_weight=norm_ws_30[l_idx],
                            positions=positions_b,
                            req_slot_ids=slots_b,
                            qkv_conv_buf=qkv_scr_b,
                            raw_out_buf=raw_scr_b,
                            gated_out_buf=out_b,
                            num_active_rows=B,
                            eps=eps,
                        )
                    elif var_id < 10:
                        ext.launch_gdn_decode_fused_vk(
                            qkvzba_b,
                            conv_states_30[l_idx],
                            conv_ws_30[l_idx],
                            a_logs_30[l_idx],
                            dt_biases_30[l_idx],
                            ssm_states_30[l_idx],
                            norm_ws_30[l_idx],
                            positions_b,
                            slots_b,
                            out_b,
                            B,
                            eps,
                            var_id,
                            s_ptr,
                        )
                    else:
                        ext.launch_gdn_decode_split_vk(
                            qkvzba_b,
                            conv_states_30[l_idx],
                            conv_ws_30[l_idx],
                            a_logs_30[l_idx],
                            dt_biases_30[l_idx],
                            ssm_states_30[l_idx],
                            norm_ws_30[l_idx],
                            positions_b,
                            slots_b,
                            qk_norm_b,
                            v_conv_b,
                            raw_scr_b,
                            out_b,
                            B,
                            eps,
                            var_id - 10,
                            s_ptr,
                        )

            with torch.cuda.stream(g_stream):
                for _ in range(3):
                    _run_30(int(g_stream.cuda_stream))
            g_stream.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=g_stream):
                _run_30(int(g_stream.cuda_stream))
            torch.cuda.synchronize()
            b_lat[mode_name] = _time_fn_us(lambda: graph.replay(), warmup=30, reps=150)

        results["decode_30layers_latency"][f"B={B}"] = b_lat

    # 4. 30-Layer Varlen Prefill Latency Benchmark
    prefill_shapes = [
        ("1x24", 1, 24),
        ("1x64", 1, 64),
        ("1x512", 1, 512),
        ("1x1024", 1, 1024),
        ("1x2048", 1, 2048),
        ("8x512", 8, 512),
        ("16x256", 16, 256),
    ]
    for label, n_seqs, seq_len in prefill_shapes:
        total_rows = n_seqs * seq_len
        qkvzba_p = torch.randn((total_rows, 1544), device=device, dtype=torch.bfloat16) * 0.1
        q_locs_p = torch.arange(0, (n_seqs + 1) * seq_len, seq_len, device=device, dtype=torch.int32)
        pos_p = torch.arange(seq_len, device=device, dtype=torch.int32).repeat(n_seqs)
        slots_p = torch.arange(n_seqs, device=device, dtype=torch.int32).repeat_interleave(seq_len)

        qk_norm_scr = torch.empty((total_rows, 512), device=device, dtype=torch.float32)
        v_conv_scr = torch.empty((total_rows, 1024), device=device, dtype=torch.bfloat16)
        raw_out_scr = torch.empty((total_rows, 512), device=device, dtype=torch.bfloat16)
        out_p = torch.empty((total_rows, 512), device=device, dtype=torch.bfloat16)
        s_ptr = int(torch.cuda.current_stream(device).cuda_stream)

        def _pf_triton():
            for l_idx in range(n_layers):
                _legacy_triton_gdn_prefill_kv(
                    qkvzba_p,
                    conv_state=conv_states_30[l_idx],
                    conv_weight=conv_ws_30[l_idx],
                    a_log=a_logs_30[l_idx],
                    dt_bias=dt_biases_30[l_idx],
                    ssm_state=ssm_states_30[l_idx],
                    norm_weight=norm_ws_30[l_idx],
                    query_start_loc=q_locs_p,
                    positions=pos_p,
                    req_slot_ids=slots_p,
                    qkv_conv_buf=v_conv_scr,
                    raw_out_buf=raw_out_scr,
                    gated_out_buf=out_p,
                    num_seqs=n_seqs,
                    eps=eps,
                )

        def _pf_cuda_vk():
            for l_idx in range(n_layers):
                ext.launch_gdn_prefill_vk(
                    qkvzba_p,
                    conv_states_30[l_idx],
                    conv_ws_30[l_idx],
                    a_logs_30[l_idx],
                    dt_biases_30[l_idx],
                    ssm_states_30[l_idx],
                    norm_ws_30[l_idx],
                    q_locs_p,
                    pos_p,
                    slots_p,
                    qk_norm_scr,
                    v_conv_scr,
                    raw_out_scr,
                    out_p,
                    n_seqs,
                    eps,
                    s_ptr,
                )

        shape_lat = {
            "triton_3op_kv": _time_fn_us(_pf_triton, warmup=10, reps=40),
            "cuda_vk_recurrent": _time_fn_us(_pf_cuda_vk, warmup=10, reps=40),
        }

        if has_gdn_cute:
            cu_seqlens_64 = q_locs_p.to(torch.int64)
            init_st_buf = torch.zeros((n_seqs, 4, 128, 128), device=device, dtype=torch.float32)

            def _pf_cute():
                for l_idx in range(n_layers):
                    _conv1d_prefill_helper(
                        qkvzba_p,
                        conv_states_30[l_idx],
                        conv_ws_30[l_idx],
                        q_locs_p,
                        pos_p,
                        slots_p,
                        n_seqs,
                        v_conv_scr,
                    )
                    b_raw = qkvzba_p[:, 1536:1540].float()
                    a_raw = qkvzba_p[:, 1540:1544].float()
                    beta_3d = torch.sigmoid(b_raw).unsqueeze(0)
                    g_3d = (
                        -torch.exp(a_logs_30[l_idx])
                        * torch.nn.functional.softplus(a_raw + dt_biases_30[l_idx].float())
                    ).unsqueeze(0)
                    raw_c, final_st = chunk_gated_delta_rule(
                        q=v_conv_scr[:, :256].view(1, total_rows, 2, 128),
                        k=v_conv_scr[:, 256:512].view(1, total_rows, 2, 128),
                        v=v_conv_scr[:, 512:1024].view(1, total_rows, 4, 128),
                        g=g_3d,
                        beta=beta_3d,
                        scale=128**-0.5,
                        initial_state=init_st_buf,
                        output_final_state=True,
                        cu_seqlens=cu_seqlens_64,
                        use_qk_l2norm_in_kernel=True,
                        state_v_first=True,
                    )
                    ssm_states_30[l_idx, :n_seqs].copy_(final_st)
                    _rmsnorm_gated_helper(
                        raw_c.view(total_rows, 512),
                        qkvzba_p,
                        norm_ws_30[l_idx],
                        eps,
                        out_p,
                    )

            shape_lat["flashinfer_gdn_sm120_cute"] = _time_fn_us(_pf_cute, warmup=8, reps=30)

        results["prefill_30layers_latency"][label] = shape_lat

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["all", "a1", "a2"], default="all")
    parser.add_argument("--output", default="/tmp/route_a_microbench.json")
    args = parser.parse_args()

    t0 = time.time()
    payload = {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "sm_capability": list(torch.cuda.get_device_capability(0)),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    if args.part in ("all", "a2"):
        print(">>> Running Part A2 (30-layer GatedDeltaNet SM120 Decode & Prefill) ...", flush=True)
        payload["part_a2_gdn"] = run_part_a2()
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(">>> Part A2 complete!", flush=True)
    if args.part in ("all", "a1"):
        print(">>> Running Part A1 (8-GPU TP=8 shm_v5 + RMSNorm epilogue fusion) ...", flush=True)
        payload["part_a1_shm_v5_epilogue"] = run_part_a1()
        print(">>> Part A1 complete!", flush=True)
    payload["elapsed_s"] = round(time.time() - t0, 2)

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f">>> Wrote {args.output} in {payload['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()

