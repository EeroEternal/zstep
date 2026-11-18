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
NUM_PXB_RING_SLOTS = 2
MAX_RANKS = 8
MAX_BLOCKS = 64
SHM_V5_HEADER_BYTES = 131072
SHM_V5_ABORT_OFFSET = 98304
ARGMAX_SHM_ABORT_OFFSET = 128
DEFAULT_ONE_SHOT_MAX_VECS = 4096  # Up to 64 KiB (e.g. M <= 16 at H=2048 or M <= 6 at H=5120 BF16)
DEFAULT_PXB_MAX_ELEMENTS = 3072 * 5120  # Up to 30.0 MiB (~122.6 MiB 2-slot dual-regime CUDA-IPC HBM ring)

_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <cstdint>
#include <stdexcept>

std::uintptr_t register_mapped_host_cuda(std::uintptr_t host_address, int64_t bytes);
void unregister_mapped_host_cuda(std::uintptr_t host_address);
torch::Tensor alloc_ipc_buffer_cuda(int64_t bytes, torch::Tensor out_ptr_u64);
std::uintptr_t open_ipc_handle_cuda(torch::Tensor handle_u8);
void close_ipc_handle_cuda(std::uintptr_t ptr);
void free_ipc_buffer_cuda(std::uintptr_t ptr);

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
    std::uintptr_t my_hbm_ptr,
    std::uintptr_t pt_hbm_ptr,
    int64_t slot_stride_bytes,
    int64_t pxb_slot_stride_bytes,
    int64_t num_vecs,
    int rank,
    int world_size,
    int blocks,
    int threads,
    int one_shot_max_vecs,
    int64_t pxb_max_vecs,
    std::uintptr_t stream_handle);

void launch_shm_v5_allreduce_add_rmsnorm_cuda(
    const void* input_ptr,
    void* residual_ptr,
    const void* weight_ptr,
    void* normed_out_ptr,
    uint64_t* control_ptr,
    const std::uintptr_t* rank_shm_dev_ptrs,
    std::uintptr_t my_hbm_ptr,
    std::uintptr_t pt_hbm_ptr,
    int64_t slot_stride_bytes,
    int64_t pxb_slot_stride_bytes,
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
    int64_t pxb_max_vecs,
    int use_cluster,
    std::uintptr_t stream_handle);

void launch_shm_v5_allgather_cuda(
    const void* input_ptr,
    void* output_ptr,
    uint64_t* control_ptr,
    const std::uintptr_t* rank_shm_dev_ptrs,
    std::uintptr_t my_hbm_ptr,
    std::uintptr_t pt_hbm_ptr,
    int64_t slot_stride_bytes,
    int64_t outer_rows,
    int64_t vecs_per_row,
    int rank,
    int world_size,
    int blocks,
    int threads,
    std::uintptr_t stream_handle);
void launch_mhc_post_collapse_rmsnorm_cuda(
    const void* x_ptr,
    void* residual_ptr,
    const float* post_mix_ptr,
    const float* comb_mix_ptr,
    const float* pre_mix_ptr,
    const void* norm_weight_ptr,
    void* layer_input_ptr,
    void* aux_out_ptr,
    int num_tokens,
    float norm_eps,
    int write_aux,
    int post_only,
    std::uintptr_t stream_handle);
void launch_mhc_sinkhorn_stats_cuda(
    const float* gemm_mul_ptr,
    const float* gemm_sqr_ptr,
    const float* hc_scale_ptr,
    const float* hc_base_ptr,
    float* post_mix_ptr,
    float* comb_mix_ptr,
    float* pre_mix_out_ptr,
    int num_tokens,
    int n_splits,
    float inv_rms_numel,
    float rms_eps,
    float hc_pre_eps,
    float hc_sinkhorn_eps,
    float hc_post_mult,
    int sinkhorn_repeat,
    int save_pre_mix,
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
    int write_all_ranks,
    std::uintptr_t stream_handle);

std::uintptr_t register_mapped_host(std::uintptr_t host_address, int64_t bytes) {
    return register_mapped_host_cuda(host_address, bytes);
}

void unregister_mapped_host(std::uintptr_t host_address) {
    unregister_mapped_host_cuda(host_address);
}

torch::Tensor alloc_ipc_buffer(int64_t bytes, torch::Tensor out_ptr_u64) {
    return alloc_ipc_buffer_cuda(bytes, out_ptr_u64);
}

std::uintptr_t open_ipc_handle(torch::Tensor handle_u8) {
    return open_ipc_handle_cuda(handle_u8);
}

void close_ipc_handle(std::uintptr_t ptr) {
    close_ipc_handle_cuda(ptr);
}

void free_ipc_buffer(std::uintptr_t ptr) {
    free_ipc_buffer_cuda(ptr);
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

void launch_shm_v5_pxb_allreduce(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor control,
    torch::Tensor rank_shm_ptrs_u64,
    std::uintptr_t my_hbm_ptr,
    std::uintptr_t pt_hbm_ptr,
    int64_t slot_stride_bytes,
    int64_t pxb_slot_stride_bytes,
    int64_t rank,
    int64_t world_size,
    int64_t blocks,
    int64_t threads,
    int64_t one_shot_max_vecs,
    int64_t pxb_max_vecs,
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
        my_hbm_ptr,
        pt_hbm_ptr,
        slot_stride_bytes,
        pxb_slot_stride_bytes,
        input.numel() / 8,
        static_cast<int>(rank),
        static_cast<int>(world_size),
        static_cast<int>(blocks),
        static_cast<int>(threads),
        static_cast<int>(one_shot_max_vecs),
        pxb_max_vecs,
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
    launch_shm_v5_pxb_allreduce(
        input,
        output,
        control,
        rank_shm_ptrs_u64,
        0,
        0,
        slot_stride_bytes,
        0,
        rank,
        world_size,
        blocks,
        threads,
        one_shot_max_vecs,
        0,
        stream_handle);
}

void launch_shm_v5_pxb_allreduce_add_rmsnorm(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor normed_out,
    torch::Tensor control,
    torch::Tensor rank_shm_ptrs_u64,
    std::uintptr_t my_hbm_ptr,
    std::uintptr_t pt_hbm_ptr,
    int64_t slot_stride_bytes,
    int64_t pxb_slot_stride_bytes,
    double eps,
    double weight_offset,
    int64_t rank,
    int64_t world_size,
    int64_t blocks,
    int64_t threads,
    int64_t one_shot_max_vecs,
    int64_t pxb_max_vecs,
    int64_t use_cluster,
    std::uintptr_t stream_handle) {
    TORCH_CHECK(
        input.is_cuda() && residual.is_cuda() && weight.is_cuda() && normed_out.is_cuda(),
        "input/residual/weight/normed_out must be CUDA tensors");
    TORCH_CHECK(
        input.scalar_type() == at::kBFloat16 && residual.scalar_type() == at::kBFloat16 &&
            weight.scalar_type() == at::kBFloat16 && normed_out.scalar_type() == at::kBFloat16,
        "BF16 required");
    TORCH_CHECK(
        input.is_contiguous() && residual.is_contiguous() && weight.is_contiguous() &&
            normed_out.is_contiguous(),
        "input/residual/weight/normed_out must be contiguous");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [rows, hidden]");
    const int64_t rows = input.size(0);
    const int64_t hidden = input.size(1);
    TORCH_CHECK(rows > 0 && hidden > 0 && hidden % 8 == 0, "hidden must be positive and % 8 == 0");
    TORCH_CHECK(
        residual.sizes() == input.sizes() && normed_out.sizes() == input.sizes(),
        "residual/normed_out shapes must match input");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == hidden, "weight must be 1D [hidden]");
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(input.data_ptr()) % 16 == 0 &&
            reinterpret_cast<std::uintptr_t>(residual.data_ptr()) % 16 == 0 &&
            reinterpret_cast<std::uintptr_t>(weight.data_ptr()) % 16 == 0 &&
            reinterpret_cast<std::uintptr_t>(normed_out.data_ptr()) % 16 == 0,
        "input/residual/weight/normed_out pointers must be 16-byte aligned");
    TORCH_CHECK(input.numel() * 2 <= slot_stride_bytes / 2, "input exceeds SHM slot capacity");
    const int64_t vecs_per_row = hidden / 8;
    TORCH_CHECK(vecs_per_row <= 1536, "hidden exceeds 3-iter epilogue limit (12288)");
    const int64_t eff_blocks = (my_hbm_ptr != 0 && blocks == 32 && rows * vecs_per_row <= 40960) ? 8 : blocks;
    if (rows < eff_blocks) {
        TORCH_CHECK(eff_blocks % rows == 0, "rows < blocks requires blocks % rows == 0");
    }
    if (rows * vecs_per_row > one_shot_max_vecs) {
        TORCH_CHECK(
            vecs_per_row % world_size == 0,
            "two-shot epilogue requires (hidden / 8) % world_size == 0");
    }

    launch_shm_v5_allreduce_add_rmsnorm_cuda(
        input.data_ptr(),
        residual.data_ptr(),
        weight.data_ptr(),
        normed_out.data_ptr(),
        reinterpret_cast<uint64_t*>(control.data_ptr<int64_t>()),
        reinterpret_cast<const std::uintptr_t*>(rank_shm_ptrs_u64.data_ptr<int64_t>()),
        my_hbm_ptr,
        pt_hbm_ptr,
        slot_stride_bytes,
        pxb_slot_stride_bytes,
        rows,
        vecs_per_row,
        1.0f / static_cast<float>(hidden),
        static_cast<float>(eps),
        static_cast<float>(weight_offset),
        static_cast<int>(rank),
        static_cast<int>(world_size),
        static_cast<int>(blocks),
        static_cast<int>(threads),
        static_cast<int>(one_shot_max_vecs),
        pxb_max_vecs,
        static_cast<int>(use_cluster),
        stream_handle);
}

void launch_shm_v5_allreduce_add_rmsnorm(
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
    launch_shm_v5_pxb_allreduce_add_rmsnorm(
        input,
        residual,
        weight,
        normed_out,
        control,
        rank_shm_ptrs_u64,
        0,
        0,
        slot_stride_bytes,
        0,
        eps,
        weight_offset,
        rank,
        world_size,
        blocks,
        threads,
        one_shot_max_vecs,
        0,
        use_cluster,
        stream_handle);
}

void launch_shm_v5_allgather(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor control,
    torch::Tensor rank_shm_ptrs_u64,
    std::uintptr_t my_hbm_ptr,
    std::uintptr_t pt_hbm_ptr,
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
        my_hbm_ptr,
        pt_hbm_ptr,
        slot_stride_bytes,
        outer_rows,
        vecs_per_row,
        static_cast<int>(rank),
        static_cast<int>(world_size),
        static_cast<int>(blocks),
        static_cast<int>(threads),
        stream_handle);
}
void launch_mhc_post_collapse_rmsnorm(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor post_mix,
    torch::Tensor comb_mix,
    torch::Tensor pre_mix,
    torch::Tensor norm_weight,
    torch::Tensor layer_input,
    torch::Tensor aux_out,
    double norm_eps,
    int64_t write_aux,
    int64_t post_only,
    std::uintptr_t stream_handle) {
    launch_mhc_post_collapse_rmsnorm_cuda(
        x.data_ptr(),
        residual.data_ptr(),
        post_mix.data_ptr<float>(),
        comb_mix.data_ptr<float>(),
        pre_mix.data_ptr<float>(),
        post_only ? nullptr : norm_weight.data_ptr(),
        post_only ? nullptr : layer_input.data_ptr(),
        write_aux ? aux_out.data_ptr() : nullptr,
        static_cast<int>(x.size(0)),
        static_cast<float>(norm_eps),
        static_cast<int>(write_aux),
        static_cast<int>(post_only),
        stream_handle);
}

void launch_mhc_sinkhorn_stats(
    torch::Tensor gemm_mul,
    torch::Tensor gemm_sqr,
    torch::Tensor hc_scale,
    torch::Tensor hc_base,
    torch::Tensor post_mix,
    torch::Tensor comb_mix,
    torch::Tensor pre_mix_out,
    int64_t rms_numel,
    double rms_eps,
    double hc_pre_eps,
    double hc_sinkhorn_eps,
    double hc_post_mult,
    int64_t sinkhorn_repeat,
    int64_t save_pre_mix,
    std::uintptr_t stream_handle) {
    launch_mhc_sinkhorn_stats_cuda(
        gemm_mul.data_ptr<float>(),
        gemm_sqr.data_ptr<float>(),
        hc_scale.data_ptr<float>(),
        hc_base.data_ptr<float>(),
        post_mix.data_ptr<float>(),
        comb_mix.data_ptr<float>(),
        pre_mix_out.data_ptr<float>(),
        static_cast<int>(gemm_mul.size(1)),
        static_cast<int>(gemm_mul.size(0)),
        1.0f / static_cast<float>(rms_numel),
        static_cast<float>(rms_eps),
        static_cast<float>(hc_pre_eps),
        static_cast<float>(hc_sinkhorn_eps),
        static_cast<float>(hc_post_mult),
        static_cast<int>(sinkhorn_repeat),
        static_cast<int>(save_pre_mix),
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
    int64_t write_all_ranks,
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
        static_cast<int>(write_all_ranks),
        stream_handle);
}
"""

_CUDA_SOURCE = r"""
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <atomic>
#include <cstdint>
#include <cstring>
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

constexpr int kNumRingSlots = 4;
constexpr int kMaxRanks = 8;
constexpr int kOneShotMaxVecsLegacy = 1024;
constexpr int kPxbOneShotMinVecs = 1024;
constexpr int kPxbTwoShotMinVecs = 4097;
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
    __nv_bfloat16 elems[8];
};

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

// Per-rank NUMA-local SHM region layout (`shm_v5`):
// [0 .. 32768):     in_flag[block][peer]   (64B per (block, peer) -> zero false sharing & local-NUMA spin)
// [32768 .. 65536): red_flag[block][peer]  (64B per (block, peer))
// [65536 .. 98304): done_flag[block][peer] (64B per (block, peer))
// [98304 .. 98368): abort_flag             (64B cacheline-isolated cluster abort word)
// [131072 .. ):     4 ring slots * (in_data + red_data)
__global__ __launch_bounds__(512, 1) void shm_v5_allreduce_kernel(
    const uint4* __restrict__ input_vec,
    uint4* __restrict__ output_vec,
    uint64_t* __restrict__ control,
    char* const* __restrict__ rank_shm,
    char* __restrict__ my_hbm,
    char* __restrict__ pt_hbm,
    int64_t slot_stride_bytes,
    int64_t pxb_slot_stride_bytes,
    int64_t num_vecs,
    int rank,
    int world_size,
    int one_shot_max_vecs,
    int64_t pxb_max_vecs,
    int flag_base) {
    const int b_local = blockIdx.x;
    const int b = flag_base + b_local;
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
    const bool split_regime = (flag_base > 0 || (my_hbm != nullptr && gridDim.x == 32));
    const int slot = split_regime
        ? ((flag_base > 0 ? 2 : 0) + static_cast<int>((step - 1ULL) & 1))
        : static_cast<int>((step - 1ULL) & 3);
    const int pxb_slot = static_cast<int>((step - 1ULL) & 1);
    const uint64_t ring_depth = (split_regime || my_hbm != nullptr) ? 2ULL : 4ULL;
    char* my_shm = s_peer_shm[rank];
    const uint64_t* abort_flag = reinterpret_cast<const uint64_t*>(my_shm + kShmV5AbortOffset);

    const int64_t c_begin = (num_vecs * b_local) / gridDim.x;
    const int64_t c_end = (num_vecs * (b_local + 1)) / gridDim.x;
    const int64_t c_len = c_end - c_begin;

    if (step > ring_depth) {
        const uint64_t min_done = step - ring_depth;
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
    const bool can_pxb =
        (my_hbm != nullptr && pt_hbm != nullptr && world_size == 8 && num_vecs <= pxb_max_vecs && c_len >= 8);
    const int64_t cur_pxb_stride = (flag_base > 0) ? 1310720LL : pxb_slot_stride_bytes;
    const int64_t pxb_base_off = 131072LL + (
        (flag_base > 0)
            ? (2LL * pxb_slot_stride_bytes + pxb_slot * cur_pxb_stride)
            : (pxb_slot * pxb_slot_stride_bytes)
    );

    // Hybrid CX8-partner P2P + 4-pair Host-SHM 1-Shot (`16..64 KiB`)
    if (can_pxb && num_vecs > kPxbOneShotMinVecs && num_vecs <= one_shot_max_vecs) {
        const int parity = rank & 1;
        const int64_t c_mid = c_begin + c_len / 2;
        const int64_t my_hb = (parity == 0) ? c_begin : c_mid;
        const int64_t my_he = (parity == 0) ? c_mid : c_end;
        const int64_t pt_hb = (parity == 0) ? c_mid : c_begin;
        const int64_t pt_he = (parity == 0) ? c_end : c_mid;
        const int64_t half_pxb = cur_pxb_stride / 2;

        uint4* pt_in = reinterpret_cast<uint4*>(pt_hbm + pxb_base_off);
        for (int64_t idx = pt_hb + threadIdx.x; idx < pt_he; idx += blockDim.x) {
            st_wt_uint4(pt_in + idx, input_vec[idx]);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            sys_store_release_u64(reinterpret_cast<uint64_t*>(pt_hbm + b * 64), step);
            if (!spin_wait_ge_u64(reinterpret_cast<const uint64_t*>(my_hbm + b * 64), step, abort_flag)) {
                s_aborted = 1;
            }
        }
        __syncthreads();
        if (s_aborted) {
            return;
        }

        const uint4* from_pt = reinterpret_cast<const uint4*>(my_hbm + pxb_base_off);
        uint4* my_hbm_sum = reinterpret_cast<uint4*>(my_hbm + pxb_base_off + half_pxb);
        for (int64_t idx = my_hb + threadIdx.x; idx < my_he; idx += blockDim.x) {
            Pack128Bf16 a, bp, psum;
            a.u4 = input_vec[idx];
            bp.u4 = ld_cv_uint4(from_pt + idx);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 fa = __bfloat1622float2(a.pairs[j]);
                const float2 fb = __bfloat1622float2(bp.pairs[j]);
                psum.pairs[j] = __floats2bfloat162_rn(fa.x + fb.x, fa.y + fb.y);
            }
            my_hbm_sum[idx] = psum.u4;
            st_wt_uint4(my_in + idx, psum.u4);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
        }
        __syncthreads();
        if (threadIdx.x < 8) {
            sys_store_release_u64(
                reinterpret_cast<uint64_t*>(s_peer_shm[threadIdx.x] + (b * 8 + rank) * 64), step);
            if (!spin_wait_ge_u64(
                    reinterpret_cast<const uint64_t*>(my_shm + (b * 8 + threadIdx.x) * 64), step, abort_flag)) {
                s_aborted = 1;
            }
        }
        __syncthreads();
        if (s_aborted) {
            return;
        }

        for (int64_t idx = c_begin + threadIdx.x; idx < c_end; idx += blockDim.x) {
            const int h_par = (idx < c_mid) ? 0 : 1;
            float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int src_r = q * 2 + h_par;
                Pack128Bf16 pv;
                if (src_r == rank) {
                    pv.u4 = my_hbm_sum[idx];
                } else {
                    const uint4* p_in = reinterpret_cast<const uint4*>(
                        s_peer_shm[src_r] + 131072 + slot * slot_stride_bytes);
                    pv.u4 = ld_cv_uint4(p_in + idx);
                }
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __bfloat1622float2(pv.pairs[j]);
                    acc[2 * j] += f.x;
                    acc[2 * j + 1] += f.y;
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
        if (threadIdx.x < 8) {
            sys_store_release_u64(
                reinterpret_cast<uint64_t*>(s_peer_shm[threadIdx.x] + 65536 + (b * 8 + rank) * 64), step);
        }
        return;
    }

    // Hybrid CX8-partner P2P + 4-pair Host-SHM 2-Shot (`>= 128 KiB`, up to `pxb_max_vecs`)
    if (can_pxb && num_vecs >= kPxbTwoShotMinVecs) {
        const int parity = rank & 1;
        const int pair_id = rank >> 1;
        const int64_t c_mid = c_begin + c_len / 2;
        const int64_t my_hb = (parity == 0) ? c_begin : c_mid;
        const int64_t my_he = (parity == 0) ? c_mid : c_end;
        const int64_t pt_hb = (parity == 0) ? c_mid : c_begin;
        const int64_t pt_he = (parity == 0) ? c_end : c_mid;
        const int64_t my_hlen = my_he - my_hb;
        const int64_t pt_hlen = pt_he - pt_hb;
        const int64_t half_pxb = cur_pxb_stride / 2;

        uint4* pt_in = reinterpret_cast<uint4*>(pt_hbm + pxb_base_off);
        for (int64_t idx = pt_hb + threadIdx.x; idx < pt_he; idx += blockDim.x) {
            st_wt_uint4(pt_in + idx, input_vec[idx]);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            sys_store_release_u64(reinterpret_cast<uint64_t*>(pt_hbm + b * 64), step);
            if (!spin_wait_ge_u64(reinterpret_cast<const uint64_t*>(my_hbm + b * 64), step, abort_flag)) {
                s_aborted = 1;
            }
        }
        __syncthreads();
        if (s_aborted) {
            return;
        }

        const int64_t own_b = my_hb + (my_hlen * pair_id) / 4;
        const int64_t own_e = my_hb + (my_hlen * (pair_id + 1)) / 4;
        const uint4* from_pt = reinterpret_cast<const uint4*>(my_hbm + pxb_base_off);
        uint4* my_hbm_red = reinterpret_cast<uint4*>(my_hbm + pxb_base_off + half_pxb);

        for (int64_t idx = my_hb + threadIdx.x; idx < my_he; idx += blockDim.x) {
            Pack128Bf16 a, bp, psum;
            a.u4 = input_vec[idx];
            bp.u4 = ld_cv_uint4(from_pt + idx);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 fa = __bfloat1622float2(a.pairs[j]);
                const float2 fb = __bfloat1622float2(bp.pairs[j]);
                psum.pairs[j] = __floats2bfloat162_rn(fa.x + fb.x, fa.y + fb.y);
            }
            if (idx >= own_b && idx < own_e) {
                my_hbm_red[idx] = psum.u4;
            } else {
                st_wt_uint4(my_in + idx, psum.u4);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
        }
        __syncthreads();
        if (threadIdx.x < 4 && threadIdx.x != pair_id) {
            const int dst_r = threadIdx.x * 2 + parity;
            sys_store_release_u64(
                reinterpret_cast<uint64_t*>(s_peer_shm[dst_r] + (b * 8 + rank) * 64), step);
            if (!spin_wait_ge_u64(
                    reinterpret_cast<const uint64_t*>(my_shm + (b * 8 + dst_r) * 64), step, abort_flag)) {
                s_aborted = 1;
            }
        }
        __syncthreads();
        if (s_aborted) {
            return;
        }

        uint4* my_shm_red = reinterpret_cast<uint4*>(my_shm + 131072 + slot * slot_stride_bytes + half_slot);
        uint4* pt_hbm_red = reinterpret_cast<uint4*>(pt_hbm + pxb_base_off + half_pxb);
        for (int64_t idx = own_b + threadIdx.x; idx < own_e; idx += blockDim.x) {
            float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int src_r = q * 2 + parity;
                Pack128Bf16 pv;
                if (q == pair_id) {
                    pv.u4 = my_hbm_red[idx];
                } else {
                    const uint4* p_in = reinterpret_cast<const uint4*>(
                        s_peer_shm[src_r] + 131072 + slot * slot_stride_bytes);
                    pv.u4 = ld_cv_uint4(p_in + idx);
                }
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __bfloat1622float2(pv.pairs[j]);
                    acc[2 * j] += f.x;
                    acc[2 * j + 1] += f.y;
                }
            }
            Pack128Bf16 out;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                out.pairs[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
            }
            output_vec[idx] = out.u4;
            st_wt_uint4(pt_hbm_red + idx, out.u4);
            st_wt_uint4(my_shm_red + idx, out.u4);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
        }
        __syncthreads();
        if (threadIdx.x < 4) {
            if (threadIdx.x == pair_id) {
                sys_store_release_u64(reinterpret_cast<uint64_t*>(pt_hbm + 32768 + b * 64), step);
                if (!spin_wait_ge_u64(
                        reinterpret_cast<const uint64_t*>(my_hbm + 32768 + b * 64), step, abort_flag)) {
                    s_aborted = 1;
                }
            } else {
                const int dst_r = threadIdx.x * 2 + parity;
                sys_store_release_u64(
                    reinterpret_cast<uint64_t*>(s_peer_shm[dst_r] + 32768 + (b * 8 + rank) * 64), step);
                if (!spin_wait_ge_u64(
                        reinterpret_cast<const uint64_t*>(my_shm + 32768 + (b * 8 + dst_r) * 64), step, abort_flag)) {
                    s_aborted = 1;
                }
            }
        }
        __syncthreads();
        if (s_aborted) {
            return;
        }

        for (int q = 0; q < 4; ++q) {
            if (q == pair_id) continue;
            const int src_r = q * 2 + parity;
            const int64_t qb = my_hb + (my_hlen * q) / 4;
            const int64_t qe = my_hb + (my_hlen * (q + 1)) / 4;
            const uint4* o_red = reinterpret_cast<const uint4*>(
                s_peer_shm[src_r] + 131072 + slot * slot_stride_bytes + half_slot);
            for (int64_t idx = qb + threadIdx.x; idx < qe; idx += blockDim.x) {
                const uint4 v = ld_cv_uint4(o_red + idx);
                output_vec[idx] = v;
                st_wt_uint4(pt_hbm_red + idx, v);
            }
        }
        const int64_t pt_own_b = pt_hb + (pt_hlen * pair_id) / 4;
        const int64_t pt_own_e = pt_hb + (pt_hlen * (pair_id + 1)) / 4;
        for (int64_t idx = pt_own_b + threadIdx.x; idx < pt_own_e; idx += blockDim.x) {
            output_vec[idx] = ld_cv_uint4(my_hbm_red + idx);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            sys_store_release_u64(reinterpret_cast<uint64_t*>(pt_hbm + 65536 + b * 64), step);
            if (!spin_wait_ge_u64(
                    reinterpret_cast<const uint64_t*>(my_hbm + 65536 + b * 64), step, abort_flag)) {
                s_aborted = 1;
            }
        }
        __syncthreads();
        if (s_aborted) {
            return;
        }

        for (int q = 0; q < 4; ++q) {
            if (q == pair_id) continue;
            const int64_t qb = pt_hb + (pt_hlen * q) / 4;
            const int64_t qe = pt_hb + (pt_hlen * (q + 1)) / 4;
            for (int64_t idx = qb + threadIdx.x; idx < qe; idx += blockDim.x) {
                output_vec[idx] = ld_cv_uint4(my_hbm_red + idx);
            }
        }
        __syncthreads();
        if (threadIdx.x < 8) {
            sys_store_release_u64(
                reinterpret_cast<uint64_t*>(s_peer_shm[threadIdx.x] + 65536 + (b * 8 + rank) * 64), step);
        }
        return;
    }

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
            load_peer_vecs(
                peer_vals, input_vec, s_peer_shm, slot, slot_stride_bytes, idx, rank, world_size);
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
        load_peer_vecs(
            peer_vals, input_vec, s_peer_shm, slot, slot_stride_bytes, idx, rank, world_size);
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        accumulate_peer_vecs(acc, peer_vals, world_size);
        Pack128Bf16 out;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            out.pairs[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
        }
        output_vec[idx] = out.u4;
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
    if (c_len > 0 && (c_len % world_size) == 0) {
        const int64_t sub_len = c_len / world_size;
        for (int64_t off = threadIdx.x; off < c_len; off += blockDim.x) {
            const int owner = static_cast<int>(off / sub_len);
            if (owner != rank) {
                const uint4* o_red = reinterpret_cast<const uint4*>(
                    s_peer_shm[owner] + 131072 + slot * slot_stride_bytes + half_slot);
                output_vec[c_begin + off] = ld_cv_uint4(o_red + c_begin + off);
            }
        }
    } else {
        for (int owner = 0; owner < world_size; ++owner) {
            if (owner == rank) continue;
            const int64_t o_begin = c_begin + (c_len * owner) / world_size;
            const int64_t o_end = c_begin + (c_len * (owner + 1)) / world_size;
            const uint4* o_red = reinterpret_cast<const uint4*>(
                s_peer_shm[owner] + 131072 + slot * slot_stride_bytes + half_slot);
            for (int64_t idx = o_begin + threadIdx.x; idx < o_end; idx += blockDim.x) {
                output_vec[idx] = ld_cv_uint4(o_red + idx);
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

// Fused Stage-2 epilogue (`AllReduce + Residual Add + [Zero-Centered] RMSNorm`) preserving
// `done-wait`, 90s spin timeout, and strict `control[0..blocks)` ring lockstep.
__global__ __launch_bounds__(512, 1) void shm_v5_ar_add_rmsnorm_kernel(
    const uint4* __restrict__ input_vec,
    uint4* __restrict__ residual_vec,
    const uint4* __restrict__ weight_vec,
    uint4* __restrict__ normed_out_vec,
    uint64_t* __restrict__ control,
    char* const* __restrict__ rank_shm,
    char* __restrict__ my_hbm,
    char* __restrict__ pt_hbm,
    int64_t slot_stride_bytes,
    int64_t pxb_slot_stride_bytes,
    int64_t rows,
    int64_t vecs_per_row,
    float inv_hidden,
    float eps,
    float weight_offset,
    int rank,
    int world_size,
    int one_shot_max_vecs,
    int64_t pxb_max_vecs,
    int flag_base) {
    const int b_local = blockIdx.x;
    const int b = flag_base + b_local;
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
    const bool split_regime = (flag_base > 0 || (my_hbm != nullptr && blocks == 32));
    const int slot = split_regime
        ? ((flag_base > 0 ? 2 : 0) + static_cast<int>((step - 1ULL) & 1))
        : static_cast<int>((step - 1ULL) & 3);
    const int pxb_slot = static_cast<int>((step - 1ULL) & 1);
    const uint64_t ring_depth = (split_regime || my_hbm != nullptr) ? 2ULL : 4ULL;
    char* my_shm = s_peer_shm[rank];
    const uint64_t* abort_flag = reinterpret_cast<const uint64_t*>(my_shm + kShmV5AbortOffset);

    if (step > ring_depth) {
        const uint64_t min_done = step - ring_depth;
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

    const int64_t num_vecs = rows * vecs_per_row;
    const int64_t half_slot = slot_stride_bytes / 2;
    uint4* my_in = reinterpret_cast<uint4*>(my_shm + 131072 + slot * slot_stride_bytes);

    if (rows < blocks) {
        const int64_t c_begin = (num_vecs * b_local) / blocks;
        const int64_t c_end = (num_vecs * (b_local + 1)) / blocks;
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

        const int k_grp = static_cast<int>(blocks / rows);
        if ((b_local % k_grp) != 0) {
            return;
        }
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
        if (s_aborted) {
            return;
        }

        const int64_t r = b_local / k_grp;
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
                    load_peer_vecs(
                        peer_vals, input_vec, s_peer_shm, slot, slot_stride_bytes, idx, rank, world_size);
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

    const int64_t r_begin = (rows * b_local) / blocks;
    const int64_t r_end = (rows * (b_local + 1)) / blocks;
    const int64_t c_begin = r_begin * vecs_per_row;
    const int64_t c_end = r_end * vecs_per_row;
    const int num_iters = static_cast<int>((vecs_per_row + blockDim.x - 1) / blockDim.x);
    const bool can_pxb =
        (my_hbm != nullptr && pt_hbm != nullptr && world_size == 8 && num_vecs <= pxb_max_vecs && r_end > r_begin);
    const int64_t cur_pxb_stride = (flag_base > 0) ? 1310720LL : pxb_slot_stride_bytes;
    const int64_t pxb_base_off = 131072LL + (
        (flag_base > 0)
            ? (2LL * pxb_slot_stride_bytes + pxb_slot * cur_pxb_stride)
            : (pxb_slot * pxb_slot_stride_bytes)
    );

    // Hybrid CX8-partner P2P + 4-pair Host-SHM 1-Shot fused epilogue (`16..64 KiB`, e.g. Qwen3.5 M=8, 16)
    if (can_pxb && num_vecs > kPxbOneShotMinVecs && num_vecs <= one_shot_max_vecs && (vecs_per_row % 2) == 0) {
        const int parity = rank & 1;
        const int64_t col_half = vecs_per_row / 2;
        const int64_t my_col_b = (parity == 0) ? 0 : col_half;
        const int64_t pt_col_b = (parity == 0) ? col_half : 0;
        const int64_t r_cnt = r_end - r_begin;
        const int64_t total_half = r_cnt * col_half;
        const int64_t half_pxb = cur_pxb_stride / 2;

        uint4* pt_in = reinterpret_cast<uint4*>(pt_hbm + pxb_base_off);
        for (int64_t t = threadIdx.x; t < total_half; t += blockDim.x) {
            const int64_t r = r_begin + (t / col_half);
            const int64_t col = pt_col_b + (t % col_half);
            const int64_t idx = r * vecs_per_row + col;
            st_wt_uint4(pt_in + idx, input_vec[idx]);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            sys_store_release_u64(reinterpret_cast<uint64_t*>(pt_hbm + b * 64), step);
            if (!spin_wait_ge_u64(reinterpret_cast<const uint64_t*>(my_hbm + b * 64), step, abort_flag)) {
                s_aborted = 1;
            }
        }
        __syncthreads();
        if (s_aborted) {
            return;
        }

        const uint4* from_pt = reinterpret_cast<const uint4*>(my_hbm + pxb_base_off);
        uint4* my_hbm_sum = reinterpret_cast<uint4*>(my_hbm + pxb_base_off + half_pxb);
        for (int64_t t = threadIdx.x; t < total_half; t += blockDim.x) {
            const int64_t r = r_begin + (t / col_half);
            const int64_t col = my_col_b + (t % col_half);
            const int64_t idx = r * vecs_per_row + col;
            Pack128Bf16 a, bp, psum;
            a.u4 = input_vec[idx];
            bp.u4 = ld_cv_uint4(from_pt + idx);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 fa = __bfloat1622float2(a.pairs[j]);
                const float2 fb = __bfloat1622float2(bp.pairs[j]);
                psum.pairs[j] = __floats2bfloat162_rn(fa.x + fb.x, fa.y + fb.y);
            }
            my_hbm_sum[idx] = psum.u4;
            st_wt_uint4(my_in + idx, psum.u4);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
        }
        __syncthreads();
        if (threadIdx.x < 8) {
            sys_store_release_u64(
                reinterpret_cast<uint64_t*>(s_peer_shm[threadIdx.x] + (b * 8 + rank) * 64), step);
            if (!spin_wait_ge_u64(
                    reinterpret_cast<const uint64_t*>(my_shm + (b * 8 + threadIdx.x) * 64), step, abort_flag)) {
                s_aborted = 1;
            }
        }
        __syncthreads();
        if (s_aborted) {
            return;
        }

        for (int64_t r = r_begin; r < r_end; ++r) {
            const int64_t row_base = r * vecs_per_row;
            Pack128Bf16 red_packs[3];
            #pragma unroll
            for (int k = 0; k < 3; ++k) {
                if (k < num_iters) {
                    const int64_t col = threadIdx.x + static_cast<int64_t>(k) * blockDim.x;
                    if (col < vecs_per_row) {
                        const int64_t idx = row_base + col;
                        const int h_par = (col < col_half) ? 0 : 1;
                        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                        #pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const int src_r = q * 2 + h_par;
                            Pack128Bf16 pv;
                            if (src_r == rank) {
                                pv.u4 = my_hbm_sum[idx];
                            } else {
                                const uint4* p_in = reinterpret_cast<const uint4*>(
                                    s_peer_shm[src_r] + 131072 + slot * slot_stride_bytes);
                                pv.u4 = ld_cv_uint4(p_in + idx);
                            }
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const float2 f = __bfloat1622float2(pv.pairs[j]);
                                acc[2 * j] += f.x;
                                acc[2 * j + 1] += f.y;
                            }
                        }
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
        if (threadIdx.x < 8) {
            sys_store_release_u64(
                reinterpret_cast<uint64_t*>(s_peer_shm[threadIdx.x] + 65536 + (b * 8 + rank) * 64), step);
        }
        return;
    }

    // Hybrid CX8-partner P2P + 4-pair Host-SHM 2-Shot fused epilogue (`>= 128 KiB`, e.g. Qwen3.5 M=32..1024)
    if (can_pxb && num_vecs >= kPxbTwoShotMinVecs && (vecs_per_row % 8) == 0) {
        const int parity = rank & 1;
        const int pair_id = rank >> 1;
        const int64_t col_half = vecs_per_row / 2;
        const int64_t cols_per_rank = vecs_per_row / 8;
        const int64_t my_col_b = (parity == 0) ? 0 : col_half;
        const int64_t pt_col_b = (parity == 0) ? col_half : 0;
        const int64_t own_col_b = my_col_b + pair_id * cols_per_rank;
        const int64_t own_col_e = own_col_b + cols_per_rank;
        const int64_t r_cnt = r_end - r_begin;
        const int64_t total_half = r_cnt * col_half;
        const int64_t total_own = r_cnt * cols_per_rank;
        const int64_t half_pxb = cur_pxb_stride / 2;

        uint4* pt_in = reinterpret_cast<uint4*>(pt_hbm + pxb_base_off);
        for (int64_t t = threadIdx.x; t < total_half; t += blockDim.x) {
            const int64_t r = r_begin + (t / col_half);
            const int64_t col = pt_col_b + (t % col_half);
            const int64_t idx = r * vecs_per_row + col;
            st_wt_uint4(pt_in + idx, input_vec[idx]);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            sys_store_release_u64(reinterpret_cast<uint64_t*>(pt_hbm + b * 64), step);
            if (!spin_wait_ge_u64(reinterpret_cast<const uint64_t*>(my_hbm + b * 64), step, abort_flag)) {
                s_aborted = 1;
            }
        }
        __syncthreads();
        if (s_aborted) {
            return;
        }

        const uint4* from_pt = reinterpret_cast<const uint4*>(my_hbm + pxb_base_off);
        uint4* my_hbm_red = reinterpret_cast<uint4*>(my_hbm + pxb_base_off + half_pxb);
        for (int64_t t = threadIdx.x; t < total_half; t += blockDim.x) {
            const int64_t r = r_begin + (t / col_half);
            const int64_t col = my_col_b + (t % col_half);
            const int64_t idx = r * vecs_per_row + col;
            Pack128Bf16 a, bp, psum;
            a.u4 = input_vec[idx];
            bp.u4 = ld_cv_uint4(from_pt + idx);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 fa = __bfloat1622float2(a.pairs[j]);
                const float2 fb = __bfloat1622float2(bp.pairs[j]);
                psum.pairs[j] = __floats2bfloat162_rn(fa.x + fb.x, fa.y + fb.y);
            }
            if (col >= own_col_b && col < own_col_e) {
                my_hbm_red[idx] = psum.u4;
            } else {
                st_wt_uint4(my_in + idx, psum.u4);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
        }
        __syncthreads();
        if (threadIdx.x < 4 && threadIdx.x != pair_id) {
            const int dst_r = threadIdx.x * 2 + parity;
            sys_store_release_u64(
                reinterpret_cast<uint64_t*>(s_peer_shm[dst_r] + (b * 8 + rank) * 64), step);
            if (!spin_wait_ge_u64(
                    reinterpret_cast<const uint64_t*>(my_shm + (b * 8 + dst_r) * 64), step, abort_flag)) {
                s_aborted = 1;
            }
        }
        __syncthreads();
        if (s_aborted) {
            return;
        }

        uint4* my_shm_red = reinterpret_cast<uint4*>(my_shm + 131072 + slot * slot_stride_bytes + half_slot);
        uint4* pt_hbm_red = reinterpret_cast<uint4*>(pt_hbm + pxb_base_off + half_pxb);
        for (int64_t t = threadIdx.x; t < total_own; t += blockDim.x) {
            const int64_t r = r_begin + (t / cols_per_rank);
            const int64_t col = own_col_b + (t % cols_per_rank);
            const int64_t idx = r * vecs_per_row + col;
            float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int src_r = q * 2 + parity;
                Pack128Bf16 pv;
                if (q == pair_id) {
                    pv.u4 = my_hbm_red[idx];
                } else {
                    const uint4* p_in = reinterpret_cast<const uint4*>(
                        s_peer_shm[src_r] + 131072 + slot * slot_stride_bytes);
                    pv.u4 = ld_cv_uint4(p_in + idx);
                }
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __bfloat1622float2(pv.pairs[j]);
                    acc[2 * j] += f.x;
                    acc[2 * j + 1] += f.y;
                }
            }
            Pack128Bf16 out;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                out.pairs[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
            }
            my_hbm_red[idx] = out.u4;
            st_wt_uint4(pt_hbm_red + idx, out.u4);
            st_wt_uint4(my_shm_red + idx, out.u4);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
        }
        __syncthreads();
        if (threadIdx.x < 4) {
            if (threadIdx.x == pair_id) {
                sys_store_release_u64(reinterpret_cast<uint64_t*>(pt_hbm + 32768 + b * 64), step);
                if (!spin_wait_ge_u64(
                        reinterpret_cast<const uint64_t*>(my_hbm + 32768 + b * 64), step, abort_flag)) {
                    s_aborted = 1;
                }
            } else {
                const int dst_r = threadIdx.x * 2 + parity;
                sys_store_release_u64(
                    reinterpret_cast<uint64_t*>(s_peer_shm[dst_r] + 32768 + (b * 8 + rank) * 64), step);
                if (!spin_wait_ge_u64(
                        reinterpret_cast<const uint64_t*>(my_shm + 32768 + (b * 8 + dst_r) * 64), step, abort_flag)) {
                    s_aborted = 1;
                }
            }
        }
        __syncthreads();
        if (s_aborted) {
            return;
        }

        for (int q = 0; q < 4; ++q) {
            if (q == pair_id) continue;
            const int src_r = q * 2 + parity;
            const int64_t qb = my_col_b + q * cols_per_rank;
            const uint4* o_red = reinterpret_cast<const uint4*>(
                s_peer_shm[src_r] + 131072 + slot * slot_stride_bytes + half_slot);
            for (int64_t t = threadIdx.x; t < total_own; t += blockDim.x) {
                const int64_t r = r_begin + (t / cols_per_rank);
                const int64_t col = qb + (t % cols_per_rank);
                const int64_t idx = r * vecs_per_row + col;
                const uint4 v = ld_cv_uint4(o_red + idx);
                my_hbm_red[idx] = v;
                st_wt_uint4(pt_hbm_red + idx, v);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            sys_store_release_u64(reinterpret_cast<uint64_t*>(pt_hbm + 65536 + b * 64), step);
            if (!spin_wait_ge_u64(
                    reinterpret_cast<const uint64_t*>(my_hbm + 65536 + b * 64), step, abort_flag)) {
                s_aborted = 1;
            }
        }
        __syncthreads();
        if (s_aborted) {
            return;
        }

        // All 8/8 columns of `[r_begin, r_end)` now reside in local `my_hbm_red`!
        if (vecs_per_row == 256 && blockDim.x == 512) {
            const int row_slot = threadIdx.x >> 8;
            const int sub_tid = threadIdx.x & 255;
            const int sub_warp = sub_tid >> 5;
            const int lane_id = sub_tid & 31;
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
                    red_pack.u4 = ld_cv_uint4(my_hbm_red + idx);
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
                            red_packs[k].u4 = ld_cv_uint4(my_hbm_red + row_base + col);
                        }
                    }
                }
                apply_add_rmsnorm_epilogue_single_row(
                    row_base, vecs_per_row, residual_vec, weight_vec, normed_out_vec,
                    red_packs, num_iters, inv_hidden, eps, weight_offset, s_warp_sq, s_inv_rms_arr[0]);
                __syncthreads();
            }
        }
        if (threadIdx.x < 8) {
            sys_store_release_u64(
                reinterpret_cast<uint64_t*>(s_peer_shm[threadIdx.x] + 65536 + (b * 8 + rank) * 64), step);
        }
        return;
    }

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
                        load_peer_vecs(
                            peer_vals, input_vec, s_peer_shm, slot, slot_stride_bytes, idx, rank, world_size);
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

    const int64_t cols_per_rank = vecs_per_row / world_size;
    const int64_t col_s_begin = cols_per_rank * rank;
    const int64_t total_s_vecs = (r_end - r_begin) * cols_per_rank;
    uint4* my_red = reinterpret_cast<uint4*>(my_shm + 131072 + slot * slot_stride_bytes + half_slot);

    for (int64_t t_idx = threadIdx.x; t_idx < total_s_vecs; t_idx += blockDim.x) {
        const int64_t r = r_begin + (t_idx / cols_per_rank);
        const int64_t col = col_s_begin + (t_idx % cols_per_rank);
        const int64_t idx = r * vecs_per_row + col;
        uint4 peer_vals[kMaxRanks];
        load_peer_vecs(
            peer_vals, input_vec, s_peer_shm, slot, slot_stride_bytes, idx, rank, world_size);
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

    if (vecs_per_row == 256 && blockDim.x == 512) {
        const int row_slot = threadIdx.x >> 8;
        const int sub_tid = threadIdx.x & 255;
        const int sub_warp = sub_tid >> 5;
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

// SM120 Cluster-8 DSMEM fused `AllReduce + Residual Add + RMSNorm` for `rows < 8`:
// keeps all 8 SMs reading 1/8th slice across PCIe in parallel and exchanges only 1 float (`s_cta_sq`)
// via Distributed Shared Memory (`cluster.map_shared_rank`).
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
    int world_size,
    int flag_base,
    int pxb_active) {
    cg::cluster_group cluster = cg::this_cluster();
    const int b_local = cluster.block_rank();
    const int b = flag_base + b_local;
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
    const int slot = (flag_base > 0)
        ? (2 + static_cast<int>((step - 1ULL) & 1))
        : static_cast<int>((step - 1ULL) & 3);
    const uint64_t ring_depth = (flag_base > 0 || pxb_active != 0) ? 2ULL : 4ULL;
    char* my_shm = s_peer_shm[rank];
    const uint64_t* abort_flag = reinterpret_cast<const uint64_t*>(my_shm + kShmV5AbortOffset);

    if (step > ring_depth) {
        const uint64_t min_done = step - ring_depth;
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
    const int64_t c_begin = (num_vecs * b_local) / blocks;
    const int64_t c_end = (num_vecs * (b_local + 1)) / blocks;
    const int64_t slice_len = c_end - c_begin;
    uint4* my_in = reinterpret_cast<uint4*>(my_shm + 131072 + slot * slot_stride_bytes);

    if (!s_aborted_local) {
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
                s_aborted_local = 1;
            }
        }
        __syncthreads();
    }

    const int k_grp = static_cast<int>(blocks / rows);
    const int64_t r = b_local / k_grp;
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
                    load_peer_vecs(
                        peer_vals, input_vec, s_peer_shm, slot, slot_stride_bytes, idx, rank, world_size);
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

// 1-shot NUMA-local SHM AllGather for `[outer_rows, vecs_per_row * 8]` ->
// `[outer_rows, world_size * vecs_per_row * 8]` BF16 tensors (handles dim=0 and dim>=1).
__device__ __forceinline__ void write_ag_out(
    uint4* out, int64_t outer_rows, int64_t vpr, int p, int64_t idx, uint4 v) {
    const int64_t row = (outer_rows == 1) ? 0 : (idx / vpr);
    out[(outer_rows == 1) ? (static_cast<int64_t>(p) * vpr + idx) : ((row * 8 + p) * vpr + (idx - row * vpr))] = v;
}

__global__ __launch_bounds__(512, 2) void shm_v5_allgather_kernel(
    const uint4* __restrict__ input_vec,
    uint4* __restrict__ output_vec,
    uint64_t* __restrict__ control,
    char* const* __restrict__ rank_shm,
    char* __restrict__ my_hbm,
    char* __restrict__ pt_hbm,
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
    // CX8 PXB partner-relayed AllGather (`[98368..116800)` dedicated header flags):
    // Cuts per-GPU Host-SHM reads across shared PCIe Gen5 x16 switch uplink from 7*B to 3*B (-57.1%).
    if (my_hbm != nullptr && pt_hbm != nullptr && world_size == 8) {
        if (threadIdx.x == 0) {
            sys_store_release_u64(reinterpret_cast<uint64_t*>(pt_hbm + 98368 + b * 64), step);
            if (!spin_wait_ge_u64(reinterpret_cast<const uint64_t*>(my_hbm + 98368 + b * 64), step, abort_flag)) {
                s_aborted = 1;
            }
        }
        __syncthreads();
        if (s_aborted) return;

        const int parity = rank & 1, pair_id = rank >> 1, pt_r = rank ^ 1;
        uint4* pt_buf = reinterpret_cast<uint4*>(pt_hbm + 131072);
        const uint4* my_buf = reinterpret_cast<const uint4*>(my_hbm + 131072);
        for (int64_t idx = c_begin + threadIdx.x; idx < c_end; idx += blockDim.x) {
            const uint4 v = input_vec[idx];
            st_wt_uint4(my_in + idx, v);
            st_wt_uint4(pt_buf + idx, v);
            write_ag_out(output_vec, outer_rows, vecs_per_row, rank, idx, v);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            sys_store_release_u64(reinterpret_cast<uint64_t*>(pt_hbm + 102464 + b * 64), step);
            if (!spin_wait_ge_u64(reinterpret_cast<const uint64_t*>(my_hbm + 102464 + b * 64), step, abort_flag)) {
                s_aborted = 1;
            }
        } else if (threadIdx.x <= 3) {
            const int dst_r = ((pair_id + threadIdx.x) & 3) * 2 + parity;
            sys_store_release_u64(reinterpret_cast<uint64_t*>(s_peer_shm[dst_r] + (b * 8 + rank) * 64), step);
        }
        __syncthreads();
        if (s_aborted) return;

        for (int64_t idx = c_begin + threadIdx.x; idx < c_end; idx += blockDim.x) {
            write_ag_out(output_vec, outer_rows, vecs_per_row, pt_r, idx, ld_cv_uint4(my_buf + idx));
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            sys_store_release_u64(reinterpret_cast<uint64_t*>(pt_hbm + 106560 + b * 64), step);
            if (!spin_wait_ge_u64(reinterpret_cast<const uint64_t*>(my_hbm + 106560 + b * 64), step, abort_flag)) {
                s_aborted = 1;
            }
        } else if (threadIdx.x <= 3) {
            const int src_r = ((pair_id + threadIdx.x) & 3) * 2 + parity;
            if (!spin_wait_ge_u64(reinterpret_cast<const uint64_t*>(my_shm + (b * 8 + src_r) * 64), step, abort_flag)) {
                s_aborted = 1;
            }
        }
        __syncthreads();
        if (s_aborted) return;

        const int r1 = ((pair_id + 1) & 3) * 2 + parity;
        const int r2 = ((pair_id + 2) & 3) * 2 + parity;
        const int r3 = ((pair_id + 3) & 3) * 2 + parity;
        const uint4* s1 = reinterpret_cast<const uint4*>(s_peer_shm[r1] + 131072 + slot * slot_stride_bytes);
        const uint4* s2 = reinterpret_cast<const uint4*>(s_peer_shm[r2] + 131072 + slot * slot_stride_bytes);
        const uint4* s3 = reinterpret_cast<const uint4*>(s_peer_shm[r3] + 131072 + slot * slot_stride_bytes);
        for (int64_t idx = c_begin + threadIdx.x; idx < c_end; idx += blockDim.x) {
            const uint4 v1 = ld_cv_uint4(s1 + idx);
            const uint4 v2 = ld_cv_uint4(s2 + idx);
            const uint4 v3 = ld_cv_uint4(s3 + idx);
            st_wt_uint4(pt_buf + idx, v1);
            st_wt_uint4(pt_buf + num_in_vecs + idx, v2);
            st_wt_uint4(pt_buf + 2 * num_in_vecs + idx, v3);
            write_ag_out(output_vec, outer_rows, vecs_per_row, r1, idx, v1);
            write_ag_out(output_vec, outer_rows, vecs_per_row, r2, idx, v2);
            write_ag_out(output_vec, outer_rows, vecs_per_row, r3, idx, v3);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            sys_store_release_u64(reinterpret_cast<uint64_t*>(pt_hbm + 110656 + b * 64), step);
            if (!spin_wait_ge_u64(reinterpret_cast<const uint64_t*>(my_hbm + 110656 + b * 64), step, abort_flag)) {
                s_aborted = 1;
            }
        }
        __syncthreads();
        if (s_aborted) return;

        for (int64_t idx = c_begin + threadIdx.x; idx < c_end; idx += blockDim.x) {
            write_ag_out(output_vec, outer_rows, vecs_per_row, r1 ^ 1, idx, ld_cv_uint4(my_buf + idx));
            write_ag_out(output_vec, outer_rows, vecs_per_row, r2 ^ 1, idx, ld_cv_uint4(my_buf + num_in_vecs + idx));
            write_ag_out(output_vec, outer_rows, vecs_per_row, r3 ^ 1, idx, ld_cv_uint4(my_buf + 2 * num_in_vecs + idx));
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            sys_store_release_u64(reinterpret_cast<uint64_t*>(pt_hbm + 114752 + b * 64), step);
            if (!spin_wait_ge_u64(reinterpret_cast<const uint64_t*>(my_hbm + 114752 + b * 64), step, abort_flag)) {
                s_aborted = 1;
            }
        }
        if (threadIdx.x < 8) {
            sys_store_release_u64(reinterpret_cast<uint64_t*>(s_peer_shm[threadIdx.x] + 65536 + (b * 8 + rank) * 64), step);
        }
        return;
    }
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

// 1-pass register-fused SM120 mHC post + stream collapse + RMSNorm (`640` threads = `5120` cols,
// updating `residual` in-place to save a 2nd 4*5120 HBM pass and 160 MiB peak HBM at M=4096).
__global__ __launch_bounds__(640, 2) void sm120_mhc_post_collapse_rmsnorm_kernel(
    const uint4* __restrict__ x_vec,
    uint4* __restrict__ res_vec,
    const float* __restrict__ post_mix,
    const float* __restrict__ comb_mix,
    const float* __restrict__ pre_mix,
    const uint4* __restrict__ norm_weight_vec,
    uint4* __restrict__ layer_input_vec,
    uint4* __restrict__ aux_out_vec,
    float norm_eps,
    int write_aux,
    int post_only) {
    const int tok = blockIdx.x, tid = threadIdx.x;
    __shared__ float s_post[4], s_comb[16], s_pre[4], s_warp_sq[20], s_inv_rms;
    if (tid < 4) {
        s_post[tid] = post_mix[tok * 4 + tid];
        if (!post_only) s_pre[tid] = pre_mix[tok * 4 + tid];
    }
    if (tid < 16) s_comb[tid] = comb_mix[tok * 16 + tid];
    __syncthreads();

    const int64_t base_x = static_cast<int64_t>(tok) * 640 + tid;
    const int64_t base_r = static_cast<int64_t>(tok) * 2560 + tid;
    Pack128Bf16 dx, b0, b1, b2, b3, nr0, nr1, nr2, nr3, aux_pack, w_pack, out_pack;
    dx.u4 = x_vec[base_x];
    b0.u4 = res_vec[base_r];
    b1.u4 = res_vec[base_r + 640];
    b2.u4 = res_vec[base_r + 1280];
    b3.u4 = res_vec[base_r + 1920];
    float collapsed[8], thread_sq = 0.0f;

    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float df = __bfloat162float(dx.elems[e]);
        const float r0 = __bfloat162float(b0.elems[e]), r1 = __bfloat162float(b1.elems[e]);
        const float r2 = __bfloat162float(b2.elems[e]), r3 = __bfloat162float(b3.elems[e]);
        const __nv_bfloat16 o0 = __float2bfloat16_rn(s_post[0] * df + s_comb[0] * r0 + s_comb[4] * r1 + s_comb[8] * r2 + s_comb[12] * r3);
        const __nv_bfloat16 o1 = __float2bfloat16_rn(s_post[1] * df + s_comb[1] * r0 + s_comb[5] * r1 + s_comb[9] * r2 + s_comb[13] * r3);
        const __nv_bfloat16 o2 = __float2bfloat16_rn(s_post[2] * df + s_comb[2] * r0 + s_comb[6] * r1 + s_comb[10] * r2 + s_comb[14] * r3);
        const __nv_bfloat16 o3 = __float2bfloat16_rn(s_post[3] * df + s_comb[3] * r0 + s_comb[7] * r1 + s_comb[11] * r2 + s_comb[15] * r3);
        nr0.elems[e] = o0;
        nr1.elems[e] = o1;
        nr2.elems[e] = o2;
        nr3.elems[e] = o3;
        if (!post_only) {
            const float x0 = __bfloat162float(o0), x1 = __bfloat162float(o1);
            const float x2 = __bfloat162float(o2), x3 = __bfloat162float(o3);
            if (write_aux) {
                aux_pack.elems[e] = __float2bfloat16_rn((x0 + x1 + x2 + x3) * 0.25f);
            }
            const float val = __bfloat162float(
                __float2bfloat16_rn(s_pre[0] * x0 + s_pre[1] * x1 + s_pre[2] * x2 + s_pre[3] * x3));
            collapsed[e] = val;
            thread_sq += val * val;
        }
    }
    res_vec[base_r] = nr0.u4;
    res_vec[base_r + 640] = nr1.u4;
    res_vec[base_r + 1280] = nr2.u4;
    res_vec[base_r + 1920] = nr3.u4;
    if (post_only) return;
    if (write_aux) aux_out_vec[base_x] = aux_pack.u4;

    const float w_sq = warp_reduce_sum(thread_sq);
    const int warp_id = tid >> 5, lane = tid & 31;
    if (lane == 0) s_warp_sq[warp_id] = w_sq;
    __syncthreads();
    if (warp_id == 0) {
        float acc = warp_reduce_sum((lane < 20) ? s_warp_sq[lane] : 0.0f);
        if (lane == 0) s_inv_rms = rsqrtf(acc * (1.0f / 5120.0f) + norm_eps);
    }
    __syncthreads();

    w_pack.u4 = norm_weight_vec[tid];
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        out_pack.elems[e] = __float2bfloat16_rn(collapsed[e] * s_inv_rms * __bfloat162float(w_pack.elems[e]));
    }
    layer_input_vec[base_x] = out_pack.u4;
}

// 4-token-per-CTA 16-lane warp-shuffle 4x4 Sinkhorn coefficient generator for SM120 mHC overlap.
__global__ __launch_bounds__(128, 8) void sm120_mhc_sinkhorn_stats_kernel(
    const float* __restrict__ gemm_mul,
    const float* __restrict__ gemm_sqr,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    float* __restrict__ post_mix,
    float* __restrict__ comb_mix,
    float* __restrict__ pre_mix_out,
    int num_tokens,
    int n_splits,
    float inv_rms_numel,
    float rms_eps,
    float hc_pre_eps,
    float hc_sinkhorn_eps,
    float hc_post_mult,
    int sinkhorn_repeat,
    int save_pre_mix) {
    const int lane = threadIdx.x & 31, tok = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (tok >= num_tokens) return;
    float sq = 0.0f, mix_val = 0.0f;
    if (lane == 0) {
        for (int s = 0; s < n_splits; ++s) {
            sq += gemm_sqr[static_cast<int64_t>(s) * num_tokens + tok];
        }
        sq = rsqrtf(sq * inv_rms_numel + rms_eps);
    }
    const float rms = __shfl_sync(0xFFFFFFFFU, sq, 0);
    if (lane < 24) {
        for (int s = 0; s < n_splits; ++s) {
            mix_val += gemm_mul[(static_cast<int64_t>(s) * num_tokens + tok) * 24 + lane];
        }
        mix_val *= rms;
    }
    const float m_post = __shfl_sync(0xFFFFFFFFU, mix_val, (lane < 4) ? (lane + 4) : 0);
    if (lane < 4) {
        if (save_pre_mix) {
            pre_mix_out[tok * 4 + lane] = (1.0f / (1.0f + expf(-(mix_val * hc_scale[0] + hc_base[lane])))) + hc_pre_eps;
        }
        post_mix[tok * 4 + lane] = (1.0f / (1.0f + expf(-(m_post * hc_scale[1] + hc_base[lane + 4])))) * hc_post_mult;
    }
    const float m_cm = __shfl_sync(0xFFFFFFFFU, mix_val, (lane < 16) ? (lane + 8) : 0);
    if (lane < 16) {
        const unsigned m16 = 0x0000FFFFU;
        float cm = m_cm * hc_scale[2] + hc_base[lane + 8];
        float rmax = fmaxf(cm, __shfl_xor_sync(m16, cm, 1));
        rmax = fmaxf(rmax, __shfl_xor_sync(m16, rmax, 2));
        cm = expf(cm - rmax);
        float rsum = cm + __shfl_xor_sync(m16, cm, 1);
        rsum += __shfl_xor_sync(m16, rsum, 2);
        cm = cm / rsum + hc_sinkhorn_eps;
        float csum = cm + __shfl_xor_sync(m16, cm, 4);
        csum += __shfl_xor_sync(m16, csum, 8);
        cm = cm / (csum + hc_sinkhorn_eps);
        for (int it = 0; it < sinkhorn_repeat - 1; ++it) {
            rsum = cm + __shfl_xor_sync(m16, cm, 1);
            rsum += __shfl_xor_sync(m16, rsum, 2);
            cm = cm / (rsum + hc_sinkhorn_eps);
            csum = cm + __shfl_xor_sync(m16, cm, 4);
            csum += __shfl_xor_sync(m16, csum, 8);
            cm = cm / (csum + hc_sinkhorn_eps);
        }
        comb_mix[tok * 16 + lane] = cm;
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

torch::Tensor alloc_ipc_buffer_cuda(int64_t bytes, torch::Tensor out_ptr_u64) {
    TORCH_CHECK(bytes > 0, "IPC buffer size must be positive");
    TORCH_CHECK(
        !out_ptr_u64.is_cuda() && out_ptr_u64.is_contiguous() &&
            out_ptr_u64.scalar_type() == at::kLong && out_ptr_u64.numel() >= 1,
        "out_ptr_u64 must be a contiguous CPU int64 tensor with numel >= 1");
    void* ptr = nullptr;
    ZSTEP_CUDA_CHECK(cudaMalloc(&ptr, static_cast<size_t>(bytes)));
    cudaError_t z_err = cudaMemset(ptr, 0, static_cast<size_t>(bytes));
    if (z_err != cudaSuccess) {
        cudaFree(ptr);
        ZSTEP_CUDA_CHECK(z_err);
    }
    ZSTEP_CUDA_CHECK(cudaDeviceSynchronize());
    out_ptr_u64.data_ptr<int64_t>()[0] =
        static_cast<int64_t>(reinterpret_cast<std::uintptr_t>(ptr));
    cudaIpcMemHandle_t handle;
    cudaError_t h_err = cudaIpcGetMemHandle(&handle, ptr);
    if (h_err != cudaSuccess) {
        cudaFree(ptr);
        out_ptr_u64.data_ptr<int64_t>()[0] = 0;
        ZSTEP_CUDA_CHECK(h_err);
    }
    auto t = torch::empty(
        {static_cast<int64_t>(sizeof(cudaIpcMemHandle_t))},
        torch::dtype(torch::kUInt8));
    std::memcpy(t.data_ptr<uint8_t>(), &handle, sizeof(cudaIpcMemHandle_t));
    return t;
}

std::uintptr_t open_ipc_handle_cuda(torch::Tensor handle_u8) {
    TORCH_CHECK(
        !handle_u8.is_cuda() && handle_u8.is_contiguous() &&
            handle_u8.scalar_type() == at::kByte &&
            handle_u8.numel() == static_cast<int64_t>(sizeof(cudaIpcMemHandle_t)),
        "handle_u8 must be a contiguous CPU uint8 tensor of cudaIpcMemHandle_t size");
    cudaIpcMemHandle_t handle;
    std::memcpy(&handle, handle_u8.data_ptr<uint8_t>(), sizeof(cudaIpcMemHandle_t));
    void* ptr = nullptr;
    ZSTEP_CUDA_CHECK(cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess));
    return reinterpret_cast<std::uintptr_t>(ptr);
}

void close_ipc_handle_cuda(std::uintptr_t ptr) {
    if (ptr != 0) {
        ZSTEP_CUDA_CHECK(cudaIpcCloseMemHandle(reinterpret_cast<void*>(ptr)));
    }
}

void free_ipc_buffer_cuda(std::uintptr_t ptr) {
    if (ptr != 0) {
        ZSTEP_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(ptr)));
    }
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
    std::uintptr_t my_hbm_ptr,
    std::uintptr_t pt_hbm_ptr,
    int64_t slot_stride_bytes,
    int64_t pxb_slot_stride_bytes,
    int64_t num_vecs,
    int rank,
    int world_size,
    int blocks,
    int threads,
    int one_shot_max_vecs,
    int64_t pxb_max_vecs,
    std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const bool small_pxb = (my_hbm_ptr != 0 && blocks == 32 && num_vecs <= 40960);
    const int launch_blocks = small_pxb ? 8 : blocks;
    const int flag_base = small_pxb ? 32 : 0;
    const int64_t vecs_per_cta = (num_vecs + launch_blocks - 1) / launch_blocks;
    const int capped_threads = threads > 512 ? 512 : threads;
    const int launch_threads = small_pxb
        ? 256
        : ((capped_threads == 256 && vecs_per_cta > 256) ? 512 : capped_threads);
    shm_v5_allreduce_kernel<<<launch_blocks, launch_threads, 0, stream>>>(
        reinterpret_cast<const uint4*>(input_ptr),
        reinterpret_cast<uint4*>(output_ptr),
        control_ptr,
        reinterpret_cast<char* const*>(rank_shm_dev_ptrs),
        reinterpret_cast<char*>(my_hbm_ptr),
        reinterpret_cast<char*>(pt_hbm_ptr),
        slot_stride_bytes,
        pxb_slot_stride_bytes,
        num_vecs,
        rank,
        world_size,
        one_shot_max_vecs,
        pxb_max_vecs,
        flag_base);
    ZSTEP_CUDA_CHECK(cudaGetLastError());
}

void launch_shm_v5_allreduce_add_rmsnorm_cuda(
    const void* input_ptr,
    void* residual_ptr,
    const void* weight_ptr,
    void* normed_out_ptr,
    uint64_t* control_ptr,
    const std::uintptr_t* rank_shm_dev_ptrs,
    std::uintptr_t my_hbm_ptr,
    std::uintptr_t pt_hbm_ptr,
    int64_t slot_stride_bytes,
    int64_t pxb_slot_stride_bytes,
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
    int64_t pxb_max_vecs,
    int use_cluster,
    std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const int64_t num_vecs = rows * vecs_per_row;
    const bool small_pxb = (my_hbm_ptr != 0 && blocks == 32 && num_vecs <= 40960);
    const int launch_blocks = small_pxb ? 8 : blocks;
    const int flag_base = small_pxb ? 32 : 0;
    const int64_t vecs_per_cta = (num_vecs + launch_blocks - 1) / launch_blocks;

    if (use_cluster != 0 && rows < launch_blocks && launch_blocks == 8 && (launch_blocks % rows) == 0) {
        int dev = 0;
        ZSTEP_CUDA_CHECK(cudaGetDevice(&dev));
        static std::atomic<uint64_t> attr_init_mask{0ULL};
        const uint64_t bit = (dev >= 0 && dev < 64) ? (1ULL << dev) : 0ULL;
        if (bit == 0ULL || (attr_init_mask.load(std::memory_order_acquire) & bit) == 0ULL) {
            ZSTEP_CUDA_CHECK(cudaFuncSetAttribute(
                shm_v5_ar_add_rmsnorm_cluster8_kernel,
                cudaFuncAttributeNonPortableClusterSizeAllowed,
                1));
            if (bit != 0ULL) {
                attr_init_mask.fetch_or(bit, std::memory_order_release);
            }
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
            world_size,
            flag_base,
            my_hbm_ptr != 0 ? 1 : 0);
        ZSTEP_CUDA_CHECK(cudaGetLastError());
        return;
    }

    const int capped_threads = threads > 512 ? 512 : threads;
    const int eff_threads = small_pxb ? 256 : capped_threads;
    const int launch_threads =
        (eff_threads == 256 && (vecs_per_cta > 256 || vecs_per_row > 256))
            ? 512
            : eff_threads;
    shm_v5_ar_add_rmsnorm_kernel<<<launch_blocks, launch_threads, 0, stream>>>(
        reinterpret_cast<const uint4*>(input_ptr),
        reinterpret_cast<uint4*>(residual_ptr),
        reinterpret_cast<const uint4*>(weight_ptr),
        reinterpret_cast<uint4*>(normed_out_ptr),
        control_ptr,
        reinterpret_cast<char* const*>(rank_shm_dev_ptrs),
        reinterpret_cast<char*>(my_hbm_ptr),
        reinterpret_cast<char*>(pt_hbm_ptr),
        slot_stride_bytes,
        pxb_slot_stride_bytes,
        rows,
        vecs_per_row,
        inv_hidden,
        eps,
        weight_offset,
        rank,
        world_size,
        one_shot_max_vecs,
        pxb_max_vecs,
        flag_base);
    ZSTEP_CUDA_CHECK(cudaGetLastError());
}

void launch_shm_v5_allgather_cuda(
    const void* input_ptr,
    void* output_ptr,
    uint64_t* control_ptr,
    const std::uintptr_t* rank_shm_dev_ptrs,
    std::uintptr_t my_hbm_ptr,
    std::uintptr_t pt_hbm_ptr,
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
    const int capped = threads > 512 ? 512 : threads;
    const int launch_threads = (capped == 256 && vecs_per_cta > 256) ? 512 : capped;
    shm_v5_allgather_kernel<<<blocks, launch_threads, 0, stream>>>(
        reinterpret_cast<const uint4*>(input_ptr),
        reinterpret_cast<uint4*>(output_ptr),
        control_ptr,
        reinterpret_cast<char* const*>(rank_shm_dev_ptrs),
        reinterpret_cast<char*>(my_hbm_ptr),
        reinterpret_cast<char*>(pt_hbm_ptr),
        slot_stride_bytes,
        outer_rows,
        vecs_per_row,
        rank,
        world_size);
    ZSTEP_CUDA_CHECK(cudaGetLastError());
}

void launch_mhc_post_collapse_rmsnorm_cuda(
    const void* x_ptr,
    void* residual_ptr,
    const float* post_mix_ptr,
    const float* comb_mix_ptr,
    const float* pre_mix_ptr,
    const void* norm_weight_ptr,
    void* layer_input_ptr,
    void* aux_out_ptr,
    int num_tokens,
    float norm_eps,
    int write_aux,
    int post_only,
    std::uintptr_t stream_handle) {
    if (num_tokens <= 0) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    sm120_mhc_post_collapse_rmsnorm_kernel<<<num_tokens, 640, 0, stream>>>(
        reinterpret_cast<const uint4*>(x_ptr),
        reinterpret_cast<uint4*>(residual_ptr),
        post_mix_ptr,
        comb_mix_ptr,
        pre_mix_ptr,
        reinterpret_cast<const uint4*>(norm_weight_ptr),
        reinterpret_cast<uint4*>(layer_input_ptr),
        reinterpret_cast<uint4*>(aux_out_ptr),
        norm_eps,
        write_aux,
        post_only);
    ZSTEP_CUDA_CHECK(cudaGetLastError());
}

void launch_mhc_sinkhorn_stats_cuda(
    const float* gemm_mul_ptr,
    const float* gemm_sqr_ptr,
    const float* hc_scale_ptr,
    const float* hc_base_ptr,
    float* post_mix_ptr,
    float* comb_mix_ptr,
    float* pre_mix_out_ptr,
    int num_tokens,
    int n_splits,
    float inv_rms_numel,
    float rms_eps,
    float hc_pre_eps,
    float hc_sinkhorn_eps,
    float hc_post_mult,
    int sinkhorn_repeat,
    int save_pre_mix,
    std::uintptr_t stream_handle) {
    if (num_tokens <= 0) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    sm120_mhc_sinkhorn_stats_kernel<<<(num_tokens + 3) / 4, 128, 0, stream>>>(
        gemm_mul_ptr,
        gemm_sqr_ptr,
        hc_scale_ptr,
        hc_base_ptr,
        post_mix_ptr,
        comb_mix_ptr,
        pre_mix_out_ptr,
        num_tokens,
        n_splits,
        inv_rms_numel,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult,
        sinkhorn_repeat,
        save_pre_mix);
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
    int vocab_per_rank,
    int write_all_ranks) {
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

    if (rank == 0 || write_all_ranks != 0) {
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
    int write_all_ranks,
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
        vocab_per_rank,
        write_all_ranks);
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
            name="zstep_shm_allreduce_sm120_v17_ext",
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            functions=[
                "register_mapped_host",
                "unregister_mapped_host",
                "alloc_ipc_buffer",
                "open_ipc_handle",
                "close_ipc_handle",
                "free_ipc_buffer",
                "launch_host_mapped_two_shot",
                "launch_shm_v5_allreduce",
                "launch_shm_v5_pxb_allreduce",
                "launch_shm_v5_allreduce_add_rmsnorm",
                "launch_shm_v5_pxb_allreduce_add_rmsnorm",
                "launch_shm_v5_allgather",
                "launch_mhc_post_collapse_rmsnorm",
                "launch_mhc_sinkhorn_stats",
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
        self._own_mm: mmap.mmap | None = None

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
            self._own_mm = own_mm
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
            try:
                host_addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
                dev_ptr = extension.register_mapped_host(host_addr, total_bytes)
            except BaseException:
                if peer != rank and not mm.closed:
                    mm.close()
                raise
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
        host_addrs: list[int] = getattr(self, "_per_rank_host_addrs", [])
        for host_addr in host_addrs:
            extension.unregister_mapped_host(host_addr)
        host_addrs.clear()
        mappings: list[mmap.mmap] = getattr(self, "_per_rank_mappings", [])
        for mm in mappings:
            if not mm.closed:
                mm.close()
        mappings.clear()
        own_mm: mmap.mmap | None = getattr(self, "_own_mm", None)
        if own_mm is not None and not own_mm.closed:
            own_mm.close()
        self._own_mm = None
        own_shm_path: str | None = getattr(self, "_own_shm_path", None)
        if own_shm_path is not None:
            try:
                os.unlink(own_shm_path)
            except FileNotFoundError:
                pass


class HostMappedShmAllReduce(_PerRankShmRingBase):
    """Ordered host-mapped 4-slot ring BF16 AllReduce (`shm_v5` + optional CX8 `pxb` IPC)."""

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
        enable_pxb_p2p: bool = False,
        pxb_max_elements: int = DEFAULT_PXB_MAX_ELEMENTS,
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
        self._exchange_group: dist.ProcessGroup | None = exchange_group
        self._closed: bool = False
        self._registered: bool = False
        self._owns_unlink: bool = False
        self._mapping: mmap.mmap | None = None
        self._shm_path: str | None = None
        self._pxb_enabled: bool = False
        self._my_hbm_ptr: int = 0
        self._pt_hbm_ptr: int = 0
        self.pxb_slot_stride_bytes: int = 0
        self.pxb_max_vecs: int = 0
        self._extension = get_host_mapped_allreduce_extension()

        try:
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
                if world_size == 8:
                    self._init_pxb_ipc(
                        exchange_group=exchange_group,
                        want_pxb=enable_pxb_p2p,
                        pxb_max_elements=min(capacity_elements, max(8, pxb_max_elements)),
                    )
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
                self._shm_path = shm_path
                if create_shm:
                    ctypes.memset(
                        ctypes.addressof(ctypes.c_char.from_buffer(self._mapping)),
                        0,
                        self.data_base_offset,
                    )
                self._host_address = ctypes.addressof(ctypes.c_char.from_buffer(self._mapping))
                self._first_touch_legacy_pages()
                self._device_address = self._extension.register_mapped_host(
                    self._host_address, self.total_bytes
                )
                self._registered = True

            self.control: torch.Tensor = torch.zeros(
                MAX_BLOCKS, dtype=torch.int64, device=device
            )
        except BaseException:
            self.close(abort=True)
            raise

    @property
    def pxb_enabled(self) -> bool:
        return self._pxb_enabled

    def _init_pxb_ipc(
        self,
        *,
        exchange_group: dist.ProcessGroup,
        want_pxb: bool,
        pxb_max_elements: int,
    ) -> None:
        aligned_elems = (pxb_max_elements // 8) * 8
        pxb_rank_stride = _align_up(aligned_elems * 2, 4096)
        pxb_slot_stride = 2 * pxb_rank_stride
        small_pxb_stride = 2 * _align_up(40960 * 16, 4096)
        pxb_total_bytes = _align_up(
            SHM_V5_HEADER_BYTES
            + NUM_PXB_RING_SLOTS * (pxb_slot_stride + small_pxb_stride),
            4096,
        )
        own_ptr_t = torch.zeros(1, dtype=torch.int64)
        handle_u8 = torch.zeros(64, dtype=torch.uint8)
        alloc_ok = 0
        if want_pxb:
            try:
                h = self._extension.alloc_ipc_buffer(pxb_total_bytes, own_ptr_t)
                if h.numel() == 64:
                    handle_u8.copy_(h)
                    self._my_hbm_ptr = int(own_ptr_t.item())
                    alloc_ok = 1
            except RuntimeError:
                alloc_ok = 0

        status_t = torch.tensor([alloc_ok], dtype=torch.int32)
        status_list = [torch.zeros(1, dtype=torch.int32) for _ in range(self.world_size)]
        dist.all_gather(status_list, status_t, group=exchange_group)
        if not all(int(s.item()) == 1 for s in status_list):
            self._close_pxb_ipc(sync_peers=True)
            return

        gathered = [torch.empty_like(handle_u8) for _ in range(self.world_size)]
        dist.all_gather(gathered, handle_u8, group=exchange_group)

        open_ok = 0
        try:
            partner = self.rank ^ 1
            self._pt_hbm_ptr = int(self._extension.open_ipc_handle(gathered[partner]))
            open_ok = 1 if self._pt_hbm_ptr != 0 else 0
        except RuntimeError:
            open_ok = 0

        open_t = torch.tensor([open_ok], dtype=torch.int32)
        open_list = [torch.zeros(1, dtype=torch.int32) for _ in range(self.world_size)]
        dist.all_gather(open_list, open_t, group=exchange_group)
        if not all(int(s.item()) == 1 for s in open_list):
            self._close_pxb_ipc(sync_peers=True)
            return

        dist.barrier(group=exchange_group)
        self.pxb_slot_stride_bytes = pxb_slot_stride
        self.pxb_max_vecs = aligned_elems // 8
        self._pxb_enabled = True

    def _close_pxb_ipc(self, *, sync_peers: bool = False) -> None:
        if self._pt_hbm_ptr != 0:
            self._extension.close_ipc_handle(self._pt_hbm_ptr)
            self._pt_hbm_ptr = 0
        if (
            sync_peers
            and self._exchange_group is not None
            and dist.is_initialized()
        ):
            dist.barrier(group=self._exchange_group)
        if self._my_hbm_ptr != 0:
            self._extension.free_ipc_buffer(self._my_hbm_ptr)
            self._my_hbm_ptr = 0
        self._pxb_enabled = False
        self.pxb_slot_stride_bytes = 0
        self.pxb_max_vecs = 0

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
            self._extension.launch_shm_v5_pxb_allreduce(
                input,
                output,
                self.control,
                self._rank_shm_ptrs,
                self._my_hbm_ptr if self._pxb_enabled else 0,
                self._pt_hbm_ptr if self._pxb_enabled else 0,
                self.slot_stride_bytes,
                self.pxb_slot_stride_bytes if self._pxb_enabled else 0,
                self.rank,
                self.world_size,
                self.blocks,
                self.threads,
                self.one_shot_max_vecs,
                self.pxb_max_vecs if self._pxb_enabled else 0,
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

    def can_fuse_add_rmsnorm(self, input: torch.Tensor) -> bool:
        """Return True when `input` (`[rows, hidden]`) can run the 1-op `shm_v5` epilogue."""
        if not self._use_shm_v5 or input.ndim != 2:
            return False
        rows, hidden = int(input.shape[0]), int(input.shape[1])
        if rows <= 0 or hidden <= 0 or hidden % 8 != 0:
            return False
        vecs_per_row = hidden // 8
        num_vecs = rows * vecs_per_row
        small_pxb = self._pxb_enabled and self.blocks == 32 and num_vecs <= 40960
        eff_blocks = 8 if small_pxb else self.blocks
        capped_threads = 512 if self.threads > 512 else self.threads
        eff_threads = 256 if small_pxb else capped_threads
        vecs_per_cta = (num_vecs + eff_blocks - 1) // eff_blocks
        launch_threads = (
            512
            if (eff_threads == 256 and (vecs_per_cta > 256 or vecs_per_row > 256))
            else eff_threads
        )
        if vecs_per_row > 3 * launch_threads or vecs_per_row > 1536:
            return False
        if rows < eff_blocks and (eff_blocks % rows) != 0:
            return False
        if num_vecs > self.one_shot_max_vecs and (vecs_per_row % self.world_size) != 0:
            return False
        return True

    def all_reduce_add_rmsnorm(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        *,
        normed_out: torch.Tensor,
        eps: float,
        zero_centered: bool = True,
        scratch_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fused `AllReduce + Residual Add + [Zero-Centered] RMSNorm` with lockstep `control`."""
        if self._closed:
            raise RuntimeError("AllReduce channel is closed")
        if input.ndim != 2:
            raise ValueError(f"Expected 2D [rows, hidden] input, got shape={tuple(input.shape)}")
        if residual.shape != input.shape or normed_out.shape != input.shape:
            raise ValueError("residual and normed_out shapes must match input")
        if weight.shape != (input.shape[1],):
            raise ValueError(f"weight shape {tuple(weight.shape)} != ({input.shape[1]},)")
        if input.numel() > self.capacity_elements:
            raise ValueError("input exceeds channel capacity")
        if input.numel() == 0:
            return normed_out

        if (
            self.can_fuse_add_rmsnorm(input)
            and input.is_contiguous()
            and residual.is_contiguous()
            and weight.is_contiguous()
            and normed_out.is_contiguous()
        ):
            rows, hidden = int(input.shape[0]), int(input.shape[1])
            vecs_per_row = hidden // 8
            num_vecs = rows * vecs_per_row
            eff_blocks = (
                8
                if (self._pxb_enabled and self.blocks == 32 and num_vecs <= 40960)
                else self.blocks
            )
            use_cluster = (
                1
                if (
                    rows < eff_blocks
                    and eff_blocks == 8
                    and (eff_blocks % rows) == 0
                    and (vecs_per_row // (eff_blocks // rows)) <= 3 * 256
                )
                else 0
            )
            stream_handle = torch.cuda.current_stream(self.device).cuda_stream
            self._extension.launch_shm_v5_pxb_allreduce_add_rmsnorm(
                input,
                residual,
                weight,
                normed_out,
                self.control,
                self._rank_shm_ptrs,
                self._my_hbm_ptr if self._pxb_enabled else 0,
                self._pt_hbm_ptr if self._pxb_enabled else 0,
                self.slot_stride_bytes,
                self.pxb_slot_stride_bytes if self._pxb_enabled else 0,
                float(eps),
                1.0 if zero_centered else 0.0,
                self.rank,
                self.world_size,
                self.blocks,
                self.threads,
                self.one_shot_max_vecs,
                self.pxb_max_vecs if self._pxb_enabled else 0,
                use_cluster,
                stream_handle,
            )
            return normed_out

        scratch = scratch_out if scratch_out is not None else torch.empty_like(input)
        self.all_reduce(input, output=scratch)
        residual.copy_((residual.float() + scratch.float()).to(torch.bfloat16))
        res_f32 = residual.float()
        inv_rms = torch.rsqrt(res_f32.pow(2).mean(dim=-1, keepdim=True) + float(eps))
        w_scale = (1.0 + weight.float()) if zero_centered else weight.float()
        normed_out.copy_((res_f32 * inv_rms * w_scale).to(torch.bfloat16))
        return normed_out

    def close(self, *, abort: bool = False) -> None:
        """Release CUDA host/IPC mapping(s) and POSIX shared memory handle(s)."""
        if self._closed:
            return
        if abort:
            self.signal_abort()
        torch.cuda.synchronize(self.device)
        self._close_pxb_ipc(sync_peers=not abort)
        if self._use_shm_v5:
            if self._registered or hasattr(self, "_own_shm_path"):
                self._close_per_rank_shm(self._extension)
                self._registered = False
        else:
            if self._registered:
                self._extension.unregister_mapped_host(self._host_address)
                self._registered = False
            if self._mapping is not None:
                if not self._mapping.closed:
                    self._mapping.close()
                self._mapping = None
            if self._owns_unlink and self._shm_path is not None:
                try:
                    os.unlink(self._shm_path)
                except FileNotFoundError:
                    pass
                self._owns_unlink = False
        self._closed = True

    def __enter__(self) -> HostMappedShmAllReduce:
        return self

    def __exit__(self, exc_type: object, *_args: object) -> None:
        self.close(abort=exc_type is not None)


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
        self._registered: bool = False
        self._my_hbm_ptr: int = 0
        self._pt_hbm_ptr: int = 0
        self._pxb_cap_bytes: int = 0
        self._extension = get_host_mapped_allreduce_extension()
        try:
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
            self.control: torch.Tensor = torch.zeros(MAX_BLOCKS, dtype=torch.int64, device=device)
        except BaseException:
            self.close(abort=True)
            raise

    def attach_pxb_ipc(self, shm_ar: HostMappedShmAllReduce) -> None:
        """Attach CX8 PXB CUDA-IPC partner ring pointers from `shm_ar` (zero extra HBM)."""
        if shm_ar.pxb_enabled and self.world_size == 8:
            self._my_hbm_ptr = shm_ar._my_hbm_ptr
            self._pt_hbm_ptr = shm_ar._pt_hbm_ptr
            self._pxb_cap_bytes = NUM_PXB_RING_SLOTS * shm_ar.pxb_slot_stride_bytes

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
        use_pxb = self._my_hbm_ptr != 0 and 3 * input.numel() * 2 <= self._pxb_cap_bytes
        self._extension.launch_shm_v5_allgather(
            input,
            output,
            self.control,
            self._rank_shm_ptrs,
            self._my_hbm_ptr if use_pxb else 0,
            self._pt_hbm_ptr if use_pxb else 0,
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
        if self._registered or hasattr(self, "_own_shm_path"):
            self._close_per_rank_shm(self._extension)
            self._registered = False
        self._closed = True

    def __enter__(self) -> HostMappedShmAllGather:
        return self

    def __exit__(self, exc_type: object, *_args: object) -> None:
        self.close(abort=exc_type is not None)


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
        all_ranks: bool = False,
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
            1 if all_ranks else 0,
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
