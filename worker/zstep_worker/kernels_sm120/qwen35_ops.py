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

void launch_decode_select_tokens_cuda(
    int32_t* tokens,
    const int32_t* last_tokens,
    const int32_t* req_slot_ids,
    const int32_t* mode,
    int64_t num_rows,
    std::uintptr_t stream_handle);

void launch_decode_publish_tokens_cuda(
    int32_t* last_tokens,
    const int32_t* req_slot_ids,
    const int32_t* top2_tokens,
    int64_t num_rows,
    std::uintptr_t stream_handle);

void launch_decode_select_tokens(
    torch::Tensor tokens,
    torch::Tensor last_tokens,
    torch::Tensor req_slot_ids,
    torch::Tensor mode,
    int64_t num_rows,
    std::uintptr_t stream_handle) {
    TORCH_CHECK(num_rows <= tokens.numel() && num_rows <= req_slot_ids.numel(),
                "num_rows exceeds decode token buffers");
    launch_decode_select_tokens_cuda(
        tokens.data_ptr<int32_t>(),
        last_tokens.data_ptr<int32_t>(),
        req_slot_ids.data_ptr<int32_t>(),
        mode.data_ptr<int32_t>(),
        num_rows,
        stream_handle);
}

void launch_decode_publish_tokens(
    torch::Tensor last_tokens,
    torch::Tensor req_slot_ids,
    torch::Tensor top2_tokens,
    int64_t num_rows,
    std::uintptr_t stream_handle) {
    TORCH_CHECK(num_rows <= req_slot_ids.numel() && num_rows * 2 <= top2_tokens.numel(),
                "num_rows exceeds decode token buffers");
    launch_decode_publish_tokens_cuda(
        last_tokens.data_ptr<int32_t>(),
        req_slot_ids.data_ptr<int32_t>(),
        top2_tokens.data_ptr<int32_t>(),
        num_rows,
        stream_handle);
}

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

void launch_gdn_decode_cluster2_vk_cuda(
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

static void check_gdn_common_tensors(
    const torch::Tensor& qkvzba,
    const torch::Tensor& conv_state,
    const torch::Tensor& conv_w,
    const torch::Tensor& a_log,
    const torch::Tensor& dt_bias,
    const torch::Tensor& ssm_state_vk,
    const torch::Tensor& norm_w,
    const torch::Tensor& positions,
    const torch::Tensor& req_slot_ids,
    const torch::Tensor& gated_out) {
    TORCH_CHECK(
        qkvzba.is_cuda() && conv_state.is_cuda() && conv_w.is_cuda() && a_log.is_cuda() &&
            dt_bias.is_cuda() && ssm_state_vk.is_cuda() && norm_w.is_cuda() &&
            positions.is_cuda() && req_slot_ids.is_cuda() && gated_out.is_cuda(),
        "GDN tensors must be CUDA tensors");
    TORCH_CHECK(
        qkvzba.scalar_type() == torch::kBFloat16 && qkvzba.dim() == 2 &&
            qkvzba.size(1) >= 1544 && qkvzba.stride(1) == 1 &&
            (qkvzba.stride(0) % 4) == 0 &&
            reinterpret_cast<std::uintptr_t>(qkvzba.data_ptr()) % 8 == 0,
        "qkvzba must be 8B-aligned BF16 [rows, >=1544] with stride(1) == 1 and stride(0) % 4 == 0");
    TORCH_CHECK(
        conv_state.scalar_type() == torch::kBFloat16 && conv_state.dim() == 3 &&
            conv_state.size(1) == 1024 && conv_state.size(2) == 3 &&
            conv_state.stride(1) == 3 && conv_state.stride(2) == 1,
        "conv_state must be BF16 [slots, 1024, 3]");
    TORCH_CHECK(
        conv_w.scalar_type() == torch::kBFloat16 && conv_w.is_contiguous() &&
            conv_w.numel() == 4096,
        "conv_w must be contiguous BF16 [1024, 4]");
    TORCH_CHECK(
        a_log.scalar_type() == torch::kFloat32 && a_log.is_contiguous() && a_log.numel() == 4,
        "a_log must be contiguous FP32 [4]");
    TORCH_CHECK(
        dt_bias.scalar_type() == torch::kBFloat16 && dt_bias.is_contiguous() &&
            dt_bias.numel() == 4,
        "dt_bias must be contiguous BF16 [4]");
    TORCH_CHECK(
        ssm_state_vk.scalar_type() == torch::kFloat32 && ssm_state_vk.dim() == 4 &&
            ssm_state_vk.size(1) == 4 && ssm_state_vk.size(2) == 128 &&
            ssm_state_vk.size(3) == 128 && (ssm_state_vk.stride(0) % 4) == 0 &&
            (ssm_state_vk.stride(1) % 4) == 0 && ssm_state_vk.stride(2) == 128 &&
            ssm_state_vk.stride(3) == 1 &&
            reinterpret_cast<std::uintptr_t>(ssm_state_vk.data_ptr()) % 16 == 0,
        "ssm_state_vk must be 16B-aligned FP32 [slots, 4, 128, 128] with 16B-aligned slot/head strides");
    TORCH_CHECK(
        norm_w.scalar_type() == torch::kFloat32 && norm_w.is_contiguous() &&
            norm_w.numel() == 128 &&
            reinterpret_cast<std::uintptr_t>(norm_w.data_ptr()) % 16 == 0,
        "norm_w must be 16B-aligned contiguous FP32 [128]");
    const int64_t rows = qkvzba.size(0);
    TORCH_CHECK(
        positions.scalar_type() == torch::kInt32 && positions.is_contiguous() &&
            positions.numel() >= rows,
        "positions must be contiguous int32 [>=rows]");
    TORCH_CHECK(
        req_slot_ids.scalar_type() == torch::kInt32 && req_slot_ids.is_contiguous() &&
            req_slot_ids.numel() >= rows,
        "req_slot_ids must be contiguous int32 [>=rows]");
    TORCH_CHECK(
        gated_out.scalar_type() == torch::kBFloat16 && gated_out.is_contiguous() &&
            gated_out.dim() == 2 && gated_out.size(0) == rows && gated_out.size(1) == 512 &&
            reinterpret_cast<std::uintptr_t>(gated_out.data_ptr()) % 8 == 0,
        "gated_out must be 8B-aligned contiguous BF16 [rows, 512]");
}

void launch_gdn_decode_cluster2_vk(
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
    std::uintptr_t stream_handle) {
    check_gdn_common_tensors(
        qkvzba, conv_state, conv_w, a_log, dt_bias, ssm_state_vk,
        norm_w, positions, req_slot_ids, gated_out);
    launch_gdn_decode_cluster2_vk_cuda(
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
    check_gdn_common_tensors(
        qkvzba, conv_state, conv_w, a_log, dt_bias, ssm_state_vk,
        norm_w, positions, req_slot_ids, gated_out);
    const int64_t rows = qkvzba.size(0);
    TORCH_CHECK(
        qk_norm_scratch.is_cuda() && qk_norm_scratch.scalar_type() == torch::kFloat32 &&
            qk_norm_scratch.is_contiguous() && qk_norm_scratch.numel() >= rows * 512 &&
            reinterpret_cast<std::uintptr_t>(qk_norm_scratch.data_ptr()) % 16 == 0,
        "qk_norm_scratch must be 16B-aligned contiguous FP32 [>=rows * 512]");
    TORCH_CHECK(
        v_conv_scratch.is_cuda() && v_conv_scratch.scalar_type() == torch::kBFloat16 &&
            v_conv_scratch.is_contiguous() && v_conv_scratch.numel() >= rows * 512,
        "v_conv_scratch must be contiguous BF16 [>=rows * 512]");
    if (mode != 0) {
        TORCH_CHECK(
            raw_out.is_cuda() && raw_out.scalar_type() == torch::kBFloat16 &&
                raw_out.is_contiguous() && raw_out.numel() >= rows * 512 &&
                reinterpret_cast<std::uintptr_t>(raw_out.data_ptr()) % 8 == 0,
            "raw_out must be 8B-aligned contiguous BF16 [>=rows * 512]");
    }
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
        static_cast<int>(rows),
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
    check_gdn_common_tensors(
        qkvzba, conv_state, conv_w, a_log, dt_bias, ssm_state_vk,
        norm_w, positions, req_slot_ids, gated_out);
    const int64_t rows = qkvzba.size(0);
    TORCH_CHECK(
        query_start_loc.is_cuda() && query_start_loc.scalar_type() == torch::kInt32 &&
            query_start_loc.is_contiguous() && query_start_loc.numel() >= num_seqs + 1,
        "query_start_loc must be contiguous int32 [>=num_seqs + 1]");
    TORCH_CHECK(
        qk_norm_scratch.is_cuda() && qk_norm_scratch.scalar_type() == torch::kFloat32 &&
            qk_norm_scratch.is_contiguous() && qk_norm_scratch.numel() >= rows * 512 &&
            reinterpret_cast<std::uintptr_t>(qk_norm_scratch.data_ptr()) % 16 == 0,
        "qk_norm_scratch must be 16B-aligned contiguous FP32 [>=rows * 512]");
    TORCH_CHECK(
        v_conv_scratch.is_cuda() && v_conv_scratch.scalar_type() == torch::kBFloat16 &&
            v_conv_scratch.is_contiguous() && v_conv_scratch.numel() >= rows * 1024,
        "v_conv_scratch must be contiguous BF16 [>=rows * 1024]");
    TORCH_CHECK(
        raw_out.is_cuda() && raw_out.scalar_type() == torch::kBFloat16 &&
            raw_out.is_contiguous() && raw_out.numel() >= rows * 512 &&
            reinterpret_cast<std::uintptr_t>(raw_out.data_ptr()) % 8 == 0,
        "raw_out must be 8B-aligned contiguous BF16 [>=rows * 512]");
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
        static_cast<int>(rows),
        static_cast<float>(eps),
        stream_handle);
}
"""

_MOE_ALIGN_CUDA = r"""
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
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

    // Every rank merges so each GPU holds the sampled tokens for device-resident decode inputs.
    {
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

// ---------------------------------------------------------------------------
// SM120 Coalesced `[V, K]` GatedDeltaNet Decode & Varlen Prefill Kernels
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFFU, val, mask);
    }
    return val;
}

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
    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    if (pos != 0) {
        s0 = __bfloat162float(st[0]);
        s1 = __bfloat162float(st[1]);
        s2 = __bfloat162float(st[2]);
    }
    st[0] = __float2bfloat16_rn(s1);
    st[1] = __float2bfloat16_rn(s2);
    st[2] = __float2bfloat16_rn(x);

    const __nv_bfloat16* cw = conv_w + c * 4;
    const float acc = __bfloat162float(cw[0]) * s0 + __bfloat162float(cw[1]) * s1 +
                      __bfloat162float(cw[2]) * s2 + __bfloat162float(cw[3]) * x;
    return __bfloat162float(__float2bfloat16_rn(silu_f32(acc)));
}

// Single-launch SM120 Cluster-2 (`__cluster_dims__(2, 1, 1)`, 512 threads) `[V, K]` GDN Decode:
// CTA 0 computes Q_kh + V_{2*kh}, CTA 1 computes K_kh + V_{2*kh+1}, all 8 `float4` state rows per
// warp are preloaded before DSMEM Q/K exchange, and GatedRMSNormSiLU runs CTA-locally.
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
    const int cta_rank = cluster.block_rank();
    const int kh = blockIdx.x >> 1;
    const int vh = 2 * kh + cta_rank;
    const int64_t row = static_cast<int64_t>(blockIdx.y);
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    __shared__ __align__(16) float s_my_qk_norm[128];
    __shared__ __align__(16) float s_peer_qk_norm[128];
    __shared__ float s_v[128];
    __shared__ float s_out[128];
    __shared__ float s_warp[16];
    __shared__ float s_meta[4];

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
            if (lane == 0) {
                s_warp[warp_id] = w_sq;
            }
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

        if (tid < 128) {
            const float x_out = s_out[tid];
            const float w_sq = warp_reduce_sum(x_out * x_out);
            if (lane == 0) {
                s_warp[warp_id] = w_sq;
            }
        }
        __syncthreads();
        if (tid == 0) {
            const float h_sq = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
            s_meta[3] = rsqrtf(h_sq * (1.0f / 128.0f) + eps);
        }
        __syncthreads();
        if (tid < 128) {
            const float inv_rms = s_meta[3];
            const float normed =
                __bfloat162float(__float2bfloat16_rn(s_out[tid] * inv_rms)) * norm_w[tid];
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

__global__ void gdn_decode_prologue_conv_qknorm_kernel(
    const __nv_bfloat16* __restrict__ qkvzba,
    __nv_bfloat16* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ conv_w,
    const int32_t* __restrict__ positions,
    const int32_t* __restrict__ req_slot_ids,
    float* __restrict__ qk_norm_out,
    __nv_bfloat16* __restrict__ v_out,
    int64_t stride_qkvzba_row,
    int64_t stride_conv_slot,
    int num_active_rows) {
    const int64_t row = blockIdx.x;
    const int kh = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    if (row >= num_active_rows) {
        return;
    }

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
    if (lane == 0) {
        s_sq[warp_id] = w_sq;
    }
    __syncthreads();
    if (tid == 0) {
        s_scales[0] =
            rsqrtf(s_sq[0] + s_sq[1] + s_sq[2] + s_sq[3] + 1.0e-6f) * 0.08838834764831845f;
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
    const int64_t row = blockIdx.x;
    const int vh = blockIdx.y;
    const int kh = vh >> 1;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    if (row >= num_active_rows) {
        if (tid < 128) {
            gated_out[row * 512 + vh * 128 + tid] = __float2bfloat16_rn(0.0f);
        }
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
        for (int r = 0; r < 8; ++r) {
            st_regs[r] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    } else {
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            st_regs[r] = state_base4[(v_base + r) * 32 + lane];
        }
    }

    const float4 q4 = reinterpret_cast<const float4*>(qk_norm + row * 512 + kh * 128)[lane];
    const float4 k4 = reinterpret_cast<const float4*>(qk_norm + row * 512 + 256 + kh * 128)[lane];
    const __nv_bfloat16* row_ptr = qkvzba + row * stride_qkvzba_row;
    const float beta = 1.0f / (1.0f + expf(-__bfloat162float(row_ptr[1536 + vh])));
    const float decay = expf(
        -expf(a_log[vh]) *
        softplus_f32(__bfloat162float(row_ptr[1540 + vh]) + __bfloat162float(dt_bias[vh])));
    const __nv_bfloat16* v_ptr = v_conv + row * 512 + vh * 128;

    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int vr = v_base + r;
        float4 st4 = st_regs[r];
        st4.x *= decay;
        st4.y *= decay;
        st4.z *= decay;
        st4.w *= decay;
        const float kv_mem =
            warp_reduce_sum(st4.x * k4.x + st4.y * k4.y + st4.z * k4.z + st4.w * k4.w);
        const float delta = (__bfloat162float(v_ptr[vr]) - kv_mem) * beta;
        st4.x += k4.x * delta;
        st4.y += k4.y * delta;
        st4.z += k4.z * delta;
        st4.w += k4.w * delta;
        state_base4[vr * 32 + lane] = st4;
        const float out_v =
            warp_reduce_sum(st4.x * q4.x + st4.y * q4.y + st4.z * q4.z + st4.w * q4.w);
        if (lane == 0) {
            s_out[vr] = __bfloat162float(__float2bfloat16_rn(out_v));
        }
    }
    __syncthreads();

    if (tid < 128) {
        const float x_out = s_out[tid];
        const float w_sq = warp_reduce_sum(x_out * x_out);
        if (lane == 0) {
            s_warp[warp_id] = w_sq;
        }
    }
    __syncthreads();
    if (tid == 0) {
        s_inv_rms =
            rsqrtf((s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3]) * (1.0f / 128.0f) + eps);
    }
    __syncthreads();
    if (tid < 128) {
        const float normed =
            __bfloat162float(__float2bfloat16_rn(s_out[tid] * s_inv_rms)) * norm_w[tid];
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
    const int64_t row = blockIdx.x;
    if (row >= num_active_rows) {
        return;
    }
    const int hv_tile = blockIdx.y;
    const int vh = hv_tile >> 3;
    const int v_blk = hv_tile & 7;
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
        for (int r = 0; r < 4; ++r) {
            st_regs[r] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    } else {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            st_regs[r] = state_base4[(v_base + r) * 32 + lane];
        }
    }

    const float4 q4 = reinterpret_cast<const float4*>(qk_norm + row * 512 + kh * 128)[lane];
    const float4 k4 = reinterpret_cast<const float4*>(qk_norm + row * 512 + 256 + kh * 128)[lane];
    const __nv_bfloat16* row_ptr = qkvzba + row * stride_qkvzba_row;
    const float beta = 1.0f / (1.0f + expf(-__bfloat162float(row_ptr[1536 + vh])));
    const float decay = expf(
        -expf(a_log[vh]) *
        softplus_f32(__bfloat162float(row_ptr[1540 + vh]) + __bfloat162float(dt_bias[vh])));
    const __nv_bfloat16* v_ptr = v_conv + row * 512 + vh * 128;
    __nv_bfloat16* out_ptr = raw_out + row * 512 + vh * 128;

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int vr = v_base + r;
        float4 st4 = st_regs[r];
        st4.x *= decay;
        st4.y *= decay;
        st4.z *= decay;
        st4.w *= decay;
        const float kv_mem =
            warp_reduce_sum(st4.x * k4.x + st4.y * k4.y + st4.z * k4.z + st4.w * k4.w);
        const float delta = (__bfloat162float(v_ptr[vr]) - kv_mem) * beta;
        st4.x += k4.x * delta;
        st4.y += k4.y * delta;
        st4.z += k4.z * delta;
        st4.w += k4.w * delta;
        state_base4[vr * 32 + lane] = st4;
        const float out_v =
            warp_reduce_sum(st4.x * q4.x + st4.y * q4.y + st4.z * q4.z + st4.w * q4.w);
        if (lane == 0) {
            out_ptr[vr] = __float2bfloat16_rn(out_v);
        }
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
    const int tid = threadIdx.x;
    const int v_head = tid >> 5;
    const int lane = tid & 31;
    const int base_off = v_head * 128 + lane * 4;
    if (row >= num_active_rows) {
        *reinterpret_cast<uint2*>(gated_out + row * 512 + base_off) = make_uint2(0, 0);
        return;
    }
    const uint2 rx = *reinterpret_cast<const uint2*>(raw_out + row * 512 + base_off);
    const uint2 rz =
        *reinterpret_cast<const uint2*>(qkvzba + row * stride_qkvzba_row + 1024 + base_off);
    const float4 wf = *reinterpret_cast<const float4*>(norm_w + lane * 4);
    const float2 xf01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rx.x));
    const float2 xf23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rx.y));
    const float2 zf01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rz.x));
    const float2 zf23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rz.y));
    const float sq = xf01.x * xf01.x + xf01.y * xf01.y + xf23.x * xf23.x + xf23.y * xf23.y;
    const float inv_rms = rsqrtf(warp_reduce_sum(sq) * (1.0f / 128.0f) + eps);
    const float2 nf01 =
        __bfloat1622float2(__floats2bfloat162_rn(xf01.x * inv_rms, xf01.y * inv_rms));
    const float2 nf23 =
        __bfloat1622float2(__floats2bfloat162_rn(xf23.x * inv_rms, xf23.y * inv_rms));
    const __nv_bfloat162 out01 = __floats2bfloat162_rn(
        (nf01.x * wf.x) * silu_f32(zf01.x), (nf01.y * wf.y) * silu_f32(zf01.y));
    const __nv_bfloat162 out23 = __floats2bfloat162_rn(
        (nf23.x * wf.z) * silu_f32(zf23.x), (nf23.y * wf.w) * silu_f32(zf23.y));
    uint2 packed_out;
    packed_out.x = *reinterpret_cast<const uint32_t*>(&out01);
    packed_out.y = *reinterpret_cast<const uint32_t*>(&out23);
    *reinterpret_cast<uint2*>(gated_out + row * 512 + base_off) = packed_out;
}

__global__ void gdn_conv1d_prefill_varlen_cuda_kernel(
    const __nv_bfloat16* __restrict__ qkvzba,
    __nv_bfloat16* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ conv_w,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ positions,
    const int32_t* __restrict__ req_slot_ids,
    __nv_bfloat16* __restrict__ v_conv_out,
    int64_t stride_in_row,
    int64_t stride_state_slot) {
    const int seq_idx = blockIdx.x;
    const int c = blockIdx.y * 64 + threadIdx.x;
    const int64_t q_start = query_start_loc[seq_idx];
    const int64_t q_end = query_start_loc[seq_idx + 1];
    if (q_start >= q_end) {
        return;
    }

    const int64_t slot = req_slot_ids[q_start];
    const int start_pos = positions[q_start];
    __nv_bfloat16* st = conv_state + slot * stride_state_slot + c * 3;

    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
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

__global__ void gdn_qk_l2norm_rows_kernel(
    const __nv_bfloat16* __restrict__ qkv_conv,
    float* __restrict__ qk_norm,
    int total_rows) {
    const int64_t row = blockIdx.x;
    if (row >= total_rows) {
        return;
    }
    const int vec_id = blockIdx.y;
    const int tid = threadIdx.x;
    __shared__ float s_w[4];
    __shared__ float s_scale;

    const float v = __bfloat162float(qkv_conv[row * 1024 + vec_id * 128 + tid]);
    const float w_sq = warp_reduce_sum(v * v);
    if ((tid & 31) == 0) {
        s_w[tid >> 5] = w_sq;
    }
    __syncthreads();
    if (tid == 0) {
        const float sq = s_w[0] + s_w[1] + s_w[2] + s_w[3];
        const float extra = (vec_id < 2) ? 0.08838834764831845f : 1.0f;
        s_scale = rsqrtf(sq + 1.0e-6f) * extra;
    }
    __syncthreads();
    qk_norm[row * 512 + vec_id * 128 + tid] = v * s_scale;
}

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
    const int hv_tile = blockIdx.y;
    const int v_head = hv_tile >> 3;
    const int v_blk = hv_tile & 7;
    const int k_head = v_head >> 1;

    const int64_t q_start = query_start_loc[seq_idx];
    const int64_t q_end = query_start_loc[seq_idx + 1];
    if (q_start >= q_end) {
        return;
    }

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

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
        const float4 k4 =
            reinterpret_cast<const float4*>(qk_norm + row * 512 + 256 + k_head * 128)[lane];
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
    const int64_t row = blockIdx.x;
    if (row >= total_rows) {
        return;
    }
    const int tid = threadIdx.x;
    const int v_head = tid >> 5;
    const int lane = tid & 31;

    const int base_off = v_head * 128 + lane * 4;
    const uint2 rx = *reinterpret_cast<const uint2*>(raw_out + row * 512 + base_off);
    const uint2 rz =
        *reinterpret_cast<const uint2*>(qkvzba + row * stride_qkvzba_row + 1024 + base_off);
    const float4 wf = *reinterpret_cast<const float4*>(norm_w + lane * 4);

    const float2 xf01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rx.x));
    const float2 xf23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rx.y));
    const float2 zf01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rz.x));
    const float2 zf23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rz.y));

    const float sq = xf01.x * xf01.x + xf01.y * xf01.y + xf23.x * xf23.x + xf23.y * xf23.y;
    const float head_sq = warp_reduce_sum(sq);
    const float inv_rms = rsqrtf(head_sq * (1.0f / 128.0f) + eps);

    const float2 nf01 =
        __bfloat1622float2(__floats2bfloat162_rn(xf01.x * inv_rms, xf01.y * inv_rms));
    const float2 nf23 =
        __bfloat1622float2(__floats2bfloat162_rn(xf23.x * inv_rms, xf23.y * inv_rms));

    const __nv_bfloat162 out01 = __floats2bfloat162_rn(
        (nf01.x * wf.x) * silu_f32(zf01.x), (nf01.y * wf.y) * silu_f32(zf01.y));
    const __nv_bfloat162 out23 = __floats2bfloat162_rn(
        (nf23.x * wf.z) * silu_f32(zf23.x), (nf23.y * wf.w) * silu_f32(zf23.y));

    uint2 packed_out;
    packed_out.x = *reinterpret_cast<const uint32_t*>(&out01);
    packed_out.y = *reinterpret_cast<const uint32_t*>(&out23);
    *reinterpret_cast<uint2*>(gated_out + row * 512 + base_off) = packed_out;
}

// When `*mode` is non-zero, each decode row takes the token sampled for its request slot in the
// previous decode step instead of the host-provided one.
__global__ void decode_select_tokens_kernel(
    int32_t* __restrict__ tokens,
    const int32_t* __restrict__ last_tokens,
    const int32_t* __restrict__ req_slot_ids,
    const int32_t* __restrict__ mode,
    int64_t num_rows) {
    if (*mode == 0) {
        return;
    }
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < num_rows) {
        tokens[i] = last_tokens[req_slot_ids[i]];
    }
}

__global__ void decode_publish_tokens_kernel(
    int32_t* __restrict__ last_tokens,
    const int32_t* __restrict__ req_slot_ids,
    const int32_t* __restrict__ top2_tokens,
    int64_t num_rows) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < num_rows) {
        last_tokens[req_slot_ids[i]] = top2_tokens[i * 2];
    }
}

}  // namespace

void launch_decode_select_tokens_cuda(
    int32_t* tokens,
    const int32_t* last_tokens,
    const int32_t* req_slot_ids,
    const int32_t* mode,
    int64_t num_rows,
    std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const unsigned int blocks = static_cast<unsigned int>((num_rows + 127) / 128);
    decode_select_tokens_kernel<<<blocks, 128, 0, stream>>>(
        tokens, last_tokens, req_slot_ids, mode, num_rows);
    ZSTEP_CUDA_CHECK(cudaGetLastError());
}

void launch_decode_publish_tokens_cuda(
    int32_t* last_tokens,
    const int32_t* req_slot_ids,
    const int32_t* top2_tokens,
    int64_t num_rows,
    std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const unsigned int blocks = static_cast<unsigned int>((num_rows + 127) / 128);
    decode_publish_tokens_kernel<<<blocks, 128, 0, stream>>>(
        last_tokens, req_slot_ids, top2_tokens, num_rows);
    ZSTEP_CUDA_CHECK(cudaGetLastError());
}

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

void launch_gdn_decode_cluster2_vk_cuda(
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
    std::uintptr_t stream_handle) {
    if (num_rows <= 0) {
        return;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
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
    if (num_rows <= 0) {
        return;
    }
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
    if (total_rows <= 0 || num_seqs <= 0) {
        return;
    }
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

_QWEN35_EXT = None


def get_qwen35_cuda_ext():
    """Compile or return the inline CUDA helper extension for MoE align, Top-2, and SM120 GDN."""
    global _QWEN35_EXT
    if _QWEN35_EXT is None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        _QWEN35_EXT = load_inline(
            name="zstep_qwen35_ops_sm120_v8_ext",
            cpp_sources=[_MOE_ALIGN_CPP],
            cuda_sources=[_MOE_ALIGN_CUDA],
            functions=[
                "launch_moe_align_block_size",
                "launch_local_top2_bf16",
                "launch_tp8_argmax_gather_shm",
                "launch_gdn_decode_cluster2_vk",
                "launch_gdn_decode_split_vk",
                "launch_gdn_prefill_vk",
                "launch_decode_select_tokens",
                "launch_decode_publish_tokens",
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

    # The register tile is [K, V] but ssm_state is stored [V, K], the layout the SM120 CUDA
    # GDN kernels use, so both paths can update the same state tensor.
    ssm_ptrs = (
        ssm_state_ptr
        + slot * stride_ssm_slot
        + v_head * stride_ssm_head
        + offs_v[None, :] * 128
        + offs_k[:, None]
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

    # ssm_state is stored [V, K]; see `_gated_delta_decode_kernel`.
    ssm_ptrs = (
        ssm_state_ptr
        + slot * stride_ssm_slot
        + v_head * stride_ssm_head
        + offs_v[None, :] * 128
        + offs_k[:, None]
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
    qk_norm_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused decode step for Qwen3_5MoeGatedDeltaNet (TP=8: 2 K-heads, 4 V-heads)."""
    batch_rows = qkvzba.shape[0]
    if batch_rows == 0:
        return gated_out_buf
    if os.environ.get("ZSTEP_QWEN_ENABLE_FUSED_GDN", "1") != "0":
        ext = get_qwen35_cuda_ext()
        stream_handle = torch.cuda.current_stream(qkvzba.device).cuda_stream
        mode_cfg = os.environ.get("ZSTEP_QWEN_GDN_DECODE_MODE", "auto")
        if mode_cfg not in ("auto", "cluster2", "split_2op", "split_3op"):
            raise ValueError(
                f"Unsupported ZSTEP_QWEN_GDN_DECODE_MODE={mode_cfg!r}; "
                "expected one of ('auto', 'cluster2', 'split_2op', 'split_3op')"
            )
        if mode_cfg == "cluster2" or (mode_cfg == "auto" and batch_rows > 32):
            ext.launch_gdn_decode_cluster2_vk(
                qkvzba,
                conv_state,
                conv_weight,
                a_log,
                dt_bias,
                ssm_state,
                norm_weight,
                positions,
                req_slot_ids,
                gated_out_buf,
                num_active_rows,
                float(eps),
                stream_handle,
            )
            return gated_out_buf

        split_mode = 0 if (mode_cfg == "split_2op" or (mode_cfg == "auto" and batch_rows > 8)) else 1
        qk_scratch = (
            qk_norm_buf
            if qk_norm_buf is not None
            else torch.empty((batch_rows, 512), dtype=torch.float32, device=qkvzba.device)
        )
        v_conv_scratch = qkv_conv_buf.reshape(-1)[: batch_rows * 512]
        ext.launch_gdn_decode_split_vk(
            qkvzba,
            conv_state,
            conv_weight,
            a_log,
            dt_bias,
            ssm_state,
            norm_weight,
            positions,
            req_slot_ids,
            qk_scratch,
            v_conv_scratch,
            raw_out_buf,
            gated_out_buf,
            num_active_rows,
            float(eps),
            split_mode,
            stream_handle,
        )
        return gated_out_buf

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
    qk_norm_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Varlen prefill for Qwen3_5MoeGatedDeltaNet keeping SSM tiles in registers across tokens."""
    total_rows = qkvzba.shape[0]
    if total_rows == 0 or num_seqs == 0:
        return gated_out_buf
    if os.environ.get("ZSTEP_QWEN_ENABLE_FUSED_GDN", "1") != "0":
        ext = get_qwen35_cuda_ext()
        stream_handle = torch.cuda.current_stream(qkvzba.device).cuda_stream
        qk_scratch = (
            qk_norm_buf
            if qk_norm_buf is not None
            else torch.empty((total_rows, 512), dtype=torch.float32, device=qkvzba.device)
        )
        ext.launch_gdn_prefill_vk(
            qkvzba,
            conv_state,
            conv_weight,
            a_log,
            dt_bias,
            ssm_state,
            norm_weight,
            query_start_loc,
            positions,
            req_slot_ids,
            qk_scratch,
            qkv_conv_buf,
            raw_out_buf,
            gated_out_buf,
            num_seqs,
            float(eps),
            stream_handle,
        )
        return gated_out_buf

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
