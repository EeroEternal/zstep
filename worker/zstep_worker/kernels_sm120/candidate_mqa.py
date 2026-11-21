# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT

"""SM120 FP8 Tensor-Core candidate-sparse MQA logits kernel for DeepSeek-V4.1 prefill.

On DeepSeek-V4.1-Flash, indexer layer 20 selects ``candidate_topk_blocks = 2048``
candidate blocks of ``candidate_block_size = 8`` keys (``16,384`` keys per query row),
and the 4 downstream indexer layers (24, 28, 32, 36) only attend to those candidate
keys. Stock vLLM computes dense ``fp8_fp4_mqa_logits`` over all ``K`` keys (up to
128K), masks out non-candidate columns via ``_mask_candidates_kernel``, and runs
``top_k_per_row_prefill`` over all ``K`` columns.

This module computes exact FP8 E4M3 MQA logits *only* over the ``2048 * 8 = 16,384``
candidate keys per query row using ``mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32``
(matching ``Q[32 heads, 128] x K[8 keys, 128]^T = [32, 8]`` with zero wasted lanes),
runs ``top_k_per_row_prefill`` on the compact ``16,384``-wide logits, and remaps the
winning compact indices back to request-local key IDs.
"""

from __future__ import annotations

import logging
import os
import types
from typing import Any

import torch

_LOG = logging.getLogger(__name__)

_EXT: Any | None = None
_SCRATCH: dict[int, dict[str, torch.Tensor]] = {}
_LOGGED_FIRST_FIRE: bool = False

_CUDA_SRC = r"""
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdint>

namespace {

__device__ __forceinline__ void mma_m16n8k32_e4m3(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

template <int BLOCKS_PER_CTA>
__global__ void candidate_fp8_mqa_logits_kernel(
    const uint8_t* __restrict__ q_ptr,          // [M, 32, 128] float8_e4m3fn
    const uint8_t* __restrict__ k_ptr,          // [K_total, 128] float8_e4m3fn
    const float* __restrict__ k_scale_ptr,      // [K_total] fp32
    const float* __restrict__ w_ptr,            // [M, 32] fp32
    const int32_t* __restrict__ cand_ptr,       // [M, num_cand_blocks] int32
    const int32_t* __restrict__ ks_ptr,         // [M] int32
    const int32_t* __restrict__ ke_ptr,         // [M] int32
    float* __restrict__ out_logits,             // [M, num_cand_blocks * 8] fp32
    int32_t num_rows,
    int32_t k_total,
    int32_t num_cand_blocks,
    int64_t cand_stride_row,
    int64_t out_stride_row) {
  const int row = blockIdx.x;
  const int tile = blockIdx.y;
  if (row >= num_rows) return;

  __shared__ uint32_t s_q[32][32];  // [32 heads, 128 bytes = 32 uint32]
  __shared__ float s_w[32];

  const int tid = threadIdx.x;  // 0..127
  const int warp_id = tid >> 5; // 0..3
  const int lane = tid & 31;    // 0..31

  const uint4* q_row_u4 = reinterpret_cast<const uint4*>(q_ptr + (int64_t)row * 4096);
  uint4* sq_u4 = reinterpret_cast<uint4*>(&s_q[0][0]);
  #pragma unroll
  for (int i = tid; i < 256; i += 128) {
    sq_u4[i] = q_row_u4[i];
  }
  if (tid < 32) {
    s_w[tid] = w_ptr[(int64_t)row * 32 + tid];
  }
  __syncthreads();

  const int32_t start = ks_ptr[row];
  const int32_t end = ke_ptr[row];
  const int32_t limit = (end < k_total) ? end : k_total;

  const int g = lane >> 2;  // 0..7
  const int t = lane & 3;   // 0..3

  uint32_t q_reg[2][4][4];
  #pragma unroll
  for (int m_tile = 0; m_tile < 2; ++m_tile) {
    const int h0 = m_tile * 16 + g;
    const int h1 = h0 + 8;
    #pragma unroll
    for (int k_step = 0; k_step < 4; ++k_step) {
      const int k_u32 = k_step * 8 + t;
      q_reg[m_tile][k_step][0] = s_q[h0][k_u32];
      q_reg[m_tile][k_step][1] = s_q[h1][k_u32];
      q_reg[m_tile][k_step][2] = s_q[h0][k_u32 + 4];
      q_reg[m_tile][k_step][3] = s_q[h1][k_u32 + 4];
    }
  }

  const float w0_0 = s_w[g];
  const float w0_1 = s_w[g + 8];
  const float w1_0 = s_w[16 + g];
  const float w1_1 = s_w[24 + g];

  const int b_base = tile * BLOCKS_PER_CTA;
  const int b_end = min(b_base + BLOCKS_PER_CTA, num_cand_blocks);
  float* out_row = out_logits + (int64_t)row * out_stride_row;
  const int32_t* cand_row = cand_ptr + (int64_t)row * cand_stride_row;

  for (int bi = b_base + warp_id; bi < b_end; bi += 4) {
    const int32_t blk = cand_row[bi];
    const int32_t col_base = start + blk * 8;
    if (blk < 0 || col_base >= limit) {
      if (lane < 8) {
        out_row[bi * 8 + lane] = -INFINITY;
      }
      continue;
    }

    const int32_t my_col = col_base + g;
    const bool my_valid = (my_col < limit);
    const uint32_t* k_row_u32 = my_valid
        ? reinterpret_cast<const uint32_t*>(k_ptr + (int64_t)my_col * 128)
        : nullptr;

    float d0_0 = 0.0f, d0_1 = 0.0f, d0_2 = 0.0f, d0_3 = 0.0f;
    float d1_0 = 0.0f, d1_1 = 0.0f, d1_2 = 0.0f, d1_3 = 0.0f;

    #pragma unroll
    for (int k_step = 0; k_step < 4; ++k_step) {
      uint32_t b0 = 0u, b1 = 0u;
      if (my_valid) {
        b0 = k_row_u32[k_step * 8 + t];
        b1 = k_row_u32[k_step * 8 + t + 4];
      }
      mma_m16n8k32_e4m3(
          d0_0, d0_1, d0_2, d0_3,
          q_reg[0][k_step][0], q_reg[0][k_step][1], q_reg[0][k_step][2], q_reg[0][k_step][3],
          b0, b1,
          d0_0, d0_1, d0_2, d0_3);
      mma_m16n8k32_e4m3(
          d1_0, d1_1, d1_2, d1_3,
          q_reg[1][k_step][0], q_reg[1][k_step][1], q_reg[1][k_step][2], q_reg[1][k_step][3],
          b0, b1,
          d1_0, d1_1, d1_2, d1_3);
    }

    float sum_even = fmaxf(d0_0, 0.0f) * w0_0 + fmaxf(d0_2, 0.0f) * w0_1
                   + fmaxf(d1_0, 0.0f) * w1_0 + fmaxf(d1_2, 0.0f) * w1_1;
    float sum_odd  = fmaxf(d0_1, 0.0f) * w0_0 + fmaxf(d0_3, 0.0f) * w0_1
                   + fmaxf(d1_1, 0.0f) * w1_0 + fmaxf(d1_3, 0.0f) * w1_1;

    sum_even += __shfl_xor_sync(0xffffffff, sum_even, 4);
    sum_even += __shfl_xor_sync(0xffffffff, sum_even, 8);
    sum_even += __shfl_xor_sync(0xffffffff, sum_even, 16);

    sum_odd += __shfl_xor_sync(0xffffffff, sum_odd, 4);
    sum_odd += __shfl_xor_sync(0xffffffff, sum_odd, 8);
    sum_odd += __shfl_xor_sync(0xffffffff, sum_odd, 16);

    if (g == 0) {
      const int32_t c_even = col_base + t * 2;
      const int32_t c_odd  = c_even + 1;
      out_row[bi * 8 + t * 2]     = (c_even < limit) ? (sum_even * k_scale_ptr[c_even]) : -INFINITY;
      out_row[bi * 8 + t * 2 + 1] = (c_odd  < limit) ? (sum_odd  * k_scale_ptr[c_odd])  : -INFINITY;
    }
  }
}

__global__ void remap_candidate_topk_kernel(
    const int32_t* __restrict__ compact_topk,   // [M, topk] in 0..(num_cand*8 - 1) or -1
    const int32_t* __restrict__ cand_ptr,       // [M, num_cand_blocks]
    const int32_t* __restrict__ ks_ptr,         // [M] int32
    const int32_t* __restrict__ ke_ptr,         // [M] int32
    int32_t* __restrict__ out_topk,             // [M, topk] request-local key ids or -1
    int32_t num_rows,
    int32_t topk,
    int32_t k_total,
    int64_t cand_stride_row,
    int64_t out_stride_row) {
  const int row = blockIdx.x;
  const int col = threadIdx.x + blockIdx.y * blockDim.x;
  if (row >= num_rows || col >= topk) return;
  const int32_t idx = compact_topk[(int64_t)row * topk + col];
  if (idx < 0) {
    out_topk[(int64_t)row * out_stride_row + col] = -1;
    return;
  }
  const int32_t bi = idx >> 3;
  const int32_t o  = idx & 7;
  const int32_t blk = cand_ptr[(int64_t)row * cand_stride_row + bi];
  const int32_t start = ks_ptr[row];
  const int32_t end = ke_ptr[row];
  const int32_t limit = (end < k_total) ? end : k_total;
  const int32_t local_key = blk * 8 + o;
  out_topk[(int64_t)row * out_stride_row + col] =
      (blk >= 0 && (start + local_key) < limit) ? local_key : -1;
}

}  // namespace

void candidate_fp8_mqa_logits(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor k_scale,
    torch::Tensor w,
    torch::Tensor cand_blocks,
    torch::Tensor ks,
    torch::Tensor ke,
    torch::Tensor out_logits) {
  const int32_t num_rows = static_cast<int32_t>(q.size(0));
  const int32_t k_total = static_cast<int32_t>(k.size(0));
  const int32_t num_cand_blocks = static_cast<int32_t>(cand_blocks.size(1));
  if (num_rows <= 0 || num_cand_blocks <= 0) return;
  constexpr int BLOCKS_PER_CTA = 64;
  const int num_tiles = (num_cand_blocks + BLOCKS_PER_CTA - 1) / BLOCKS_PER_CTA;
  dim3 grid(num_rows, num_tiles);
  dim3 block(128);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  candidate_fp8_mqa_logits_kernel<BLOCKS_PER_CTA><<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(q.data_ptr()),
      reinterpret_cast<const uint8_t*>(k.data_ptr()),
      k_scale.data_ptr<float>(),
      w.data_ptr<float>(),
      cand_blocks.data_ptr<int32_t>(),
      ks.data_ptr<int32_t>(),
      ke.data_ptr<int32_t>(),
      out_logits.data_ptr<float>(),
      num_rows,
      k_total,
      num_cand_blocks,
      cand_blocks.stride(0),
      out_logits.stride(0));
}

void remap_candidate_topk(
    torch::Tensor compact_topk,
    torch::Tensor cand_blocks,
    torch::Tensor ks,
    torch::Tensor ke,
    torch::Tensor out_topk,
    int64_t k_total) {
  const int32_t num_rows = static_cast<int32_t>(compact_topk.size(0));
  const int32_t topk = static_cast<int32_t>(compact_topk.size(1));
  if (num_rows <= 0 || topk <= 0) return;
  dim3 block(256);
  dim3 grid(num_rows, (topk + 255) / 256);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  remap_candidate_topk_kernel<<<grid, block, 0, stream>>>(
      compact_topk.data_ptr<int32_t>(),
      cand_blocks.data_ptr<int32_t>(),
      ks.data_ptr<int32_t>(),
      ke.data_ptr<int32_t>(),
      out_topk.data_ptr<int32_t>(),
      num_rows,
      topk,
      static_cast<int32_t>(k_total),
      cand_blocks.stride(0),
      out_topk.stride(0));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("candidate_fp8_mqa_logits", &candidate_fp8_mqa_logits);
  m.def("remap_candidate_topk", &remap_candidate_topk);
}
"""


def is_sm120_candidate_mqa_enabled() -> bool:
    from zstep_worker.env_config import WorkerRuntimeConfig

    if not WorkerRuntimeConfig.from_env().enable_sm120_candidate_mqa:
        return False
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (12, 0)


def get_sm120_candidate_mqa_extension() -> Any | None:
    global _EXT
    if _EXT is not None:
        return _EXT
    if not is_sm120_candidate_mqa_enabled():
        return None
    from torch.utils.cpp_extension import load_inline

    _EXT = load_inline(
        name="zstep_sm120_candidate_mqa_v1",
        cpp_sources="",
        cuda_sources=_CUDA_SRC,
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"],
        verbose=False,
    )
    return _EXT


def _get_scratch(
    device: torch.device,
    num_rows: int,
    compact_width: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    entry = _SCRATCH.get(dev_idx)
    if (
        entry is None
        or entry["logits"].shape[0] < num_rows
        or entry["logits"].shape[1] < compact_width
        or entry["topk"].shape[1] < topk
    ):
        cap_rows = max(num_rows, entry["logits"].shape[0] if entry else 0)
        cap_w = max(compact_width, entry["logits"].shape[1] if entry else 0)
        cap_k = max(topk, entry["topk"].shape[1] if entry else 0)
        entry = {
            "logits": torch.empty((cap_rows, cap_w), device=device, dtype=torch.float32),
            "ks": torch.zeros((cap_rows,), device=device, dtype=torch.int32),
            "ke": torch.full((cap_rows,), cap_w, device=device, dtype=torch.int32),
            "topk": torch.empty((cap_rows, cap_k), device=device, dtype=torch.int32),
            "ke_val": torch.tensor([cap_w], dtype=torch.int32),
        }
        _SCRATCH[dev_idx] = entry
    if int(entry["ke_val"][0]) != compact_width:
        entry["ke"].fill_(compact_width)
        entry["ke_val"][0] = compact_width
    return (
        entry["logits"][:num_rows, :compact_width],
        entry["ks"][:num_rows],
        entry["ke"][:num_rows],
        entry["topk"][:num_rows, :topk],
    )


def run_candidate_sparse_prefill_topk(
    *,
    q_slice: torch.Tensor,
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    chunk_candidates: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    prefill_topk: torch.Tensor,
    topk_tokens: int,
    candidate_block_size: int,
    local_total_seq_lens: int,
) -> bool:
    """Attempt the SM120 FP8 TC candidate-sparse prefill path.

    Returns True if executed, False if ineligible (caller should run stock path).
    """
    ext = get_sm120_candidate_mqa_extension()
    if ext is None:
        return False
    if candidate_block_size != 8:
        return False
    if q_slice.ndim != 3 or q_slice.shape[1] != 32 or q_slice.shape[2] != 128:
        return False
    if q_slice.dtype != torch.float8_e4m3fn or k_quant.dtype != torch.float8_e4m3fn:
        return False
    num_cand_blocks = int(chunk_candidates.shape[1])
    compact_width = num_cand_blocks * 8
    if local_total_seq_lens <= compact_width:
        return False

    from vllm import _custom_ops as ops

    num_rows = int(q_slice.shape[0])
    q_contig = q_slice if q_slice.is_contiguous() else q_slice.contiguous()
    k_contig = k_quant if k_quant.is_contiguous() else k_quant.contiguous()
    ks_f32 = k_scale if (k_scale.dtype == torch.float32 and k_scale.is_contiguous()) else k_scale.contiguous().view(torch.float32).squeeze(-1)
    w_contig = weights if (weights.dtype == torch.float32 and weights.is_contiguous()) else weights.to(torch.float32).contiguous()
    cand_i32 = chunk_candidates if chunk_candidates.dtype == torch.int32 else chunk_candidates.to(torch.int32)
    ks_i32 = cu_seqlen_ks if cu_seqlen_ks.dtype == torch.int32 else cu_seqlen_ks.to(torch.int32)
    ke_i32 = cu_seqlen_ke if cu_seqlen_ke.dtype == torch.int32 else cu_seqlen_ke.to(torch.int32)

    compact_logits, compact_ks, compact_ke, compact_topk = _get_scratch(
        q_contig.device, num_rows, compact_width, topk_tokens
    )
    ext.candidate_fp8_mqa_logits(
        q_contig,
        k_contig,
        ks_f32,
        w_contig,
        cand_i32,
        ks_i32,
        ke_i32,
        compact_logits,
    )
    ops.top_k_per_row_prefill(
        compact_logits,
        compact_ks,
        compact_ke,
        compact_topk,
        num_rows,
        compact_logits.stride(0),
        compact_logits.stride(1),
        topk_tokens,
    )
    ext.remap_candidate_topk(
        compact_topk,
        cand_i32,
        ks_i32,
        ke_i32,
        prefill_topk,
        int(k_contig.shape[0]),
    )
    global _LOGGED_FIRST_FIRE
    if not _LOGGED_FIRST_FIRE:
        _LOGGED_FIRST_FIRE = True
        _LOG.info(
            "zstep_sm120_candidate_mqa: FIRED (rows=%d, k_total=%d, cand_blocks=%d, topk=%d)",
            num_rows,
            int(k_contig.shape[0]),
            num_cand_blocks,
            topk_tokens,
        )
    return True


def install_sm120_candidate_mqa_prefill_patch(model: Any | None = None) -> bool:
    """Patch vLLM's stock ``sparse_attn_indexer`` prefill loop for SM120 candidate-sparse FP8 MQA.

    Safe to call regardless of ``ZSTEP_ENABLE_SM120_DSA_DECODE``: when ``DSA_DECODE=1``,
    ``b12x_dsv41.py`` also calls ``run_candidate_sparse_prefill_topk`` directly.
    """
    if get_sm120_candidate_mqa_extension() is None:
        return False

    import vllm.model_executor.layers.sparse_attn_indexer as sai

    if getattr(sai, "_zstep_sm120_candidate_mqa_patched", False):
        return True

    orig_sparse_attn_indexer = sai.sparse_attn_indexer

    def _patched_sparse_attn_indexer(
        hidden_states: torch.Tensor,
        k_cache_prefix: Any,
        kv_cache: torch.Tensor,
        q_quant: torch.Tensor,
        q_scale: torch.Tensor | None,
        k: torch.Tensor | None,
        weights: torch.Tensor,
        quant_block_size: int,
        scale_fmt: str | None,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        total_seq_lens: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool,
        use_pcp: bool,
        dense_mha_metadata_layer_name: Any,
        use_fp4_cache: bool = False,
        dcp_rank: int = 0,
        dcp_world_size: int = 1,
        cp_kv_cache_interleave_size: int = 1,
        skip_topk_buffer_clear: bool = False,
        candidate_blocks: torch.Tensor | None = None,
        candidate_block_size: int = 0,
        candidate_write: bool = False,
        topk_backend: str = "auto",
    ) -> torch.Tensor:
        # Only intercept FP8 candidate-masked (non-write) prefill with dcp=1, use_pcp=False.
        if (
            use_fp4_cache
            or use_pcp
            or dcp_world_size != 1
            or candidate_blocks is None
            or candidate_write
            or candidate_block_size != 8
            or head_dim != 128
            or torch.cuda.is_current_stream_capturing()
        ):
            return orig_sparse_attn_indexer(
                hidden_states,
                k_cache_prefix,
                kv_cache,
                q_quant,
                q_scale,
                k,
                weights,
                quant_block_size,
                scale_fmt,
                topk_tokens,
                head_dim,
                max_model_len,
                total_seq_lens,
                topk_indices_buffer,
                skip_k_cache_insert,
                use_pcp,
                dense_mha_metadata_layer_name,
                use_fp4_cache,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
                skip_topk_buffer_clear,
                candidate_blocks,
                candidate_block_size,
                candidate_write,
                topk_backend,
            )

        from vllm import _custom_ops as ops
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import get_forward_context
        from vllm.platforms import current_platform
        from vllm.utils.torch_utils import _resolve_layer_name
        from vllm.v1.worker.workspace import current_workspace_manager

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if not isinstance(attn_metadata, dict):
            return orig_sparse_attn_indexer(
                hidden_states,
                k_cache_prefix,
                kv_cache,
                q_quant,
                q_scale,
                k,
                weights,
                quant_block_size,
                scale_fmt,
                topk_tokens,
                head_dim,
                max_model_len,
                total_seq_lens,
                topk_indices_buffer,
                skip_k_cache_insert,
                use_pcp,
                dense_mha_metadata_layer_name,
                use_fp4_cache,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
                skip_topk_buffer_clear,
                candidate_blocks,
                candidate_block_size,
                candidate_write,
                topk_backend,
            )

        resolved_prefix = _resolve_layer_name(k_cache_prefix)
        attn_meta = attn_metadata.get(resolved_prefix)
        if attn_meta is None or getattr(attn_meta, "prefill", None) is None:
            return orig_sparse_attn_indexer(
                hidden_states,
                k_cache_prefix,
                kv_cache,
                q_quant,
                q_scale,
                k,
                weights,
                quant_block_size,
                scale_fmt,
                topk_tokens,
                head_dim,
                max_model_len,
                total_seq_lens,
                topk_indices_buffer,
                skip_k_cache_insert,
                use_pcp,
                dense_mha_metadata_layer_name,
                use_fp4_cache,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
                skip_topk_buffer_clear,
                candidate_blocks,
                candidate_block_size,
                candidate_write,
                topk_backend,
            )

        prefill_metadata = attn_meta.prefill
        has_decode = getattr(attn_meta, "decode", None) is not None
        if not any(
            c.local_total_seq_lens > candidate_blocks.shape[1] * 8
            for c in prefill_metadata.chunks
        ):
            return orig_sparse_attn_indexer(
                hidden_states,
                k_cache_prefix,
                kv_cache,
                q_quant,
                q_scale,
                k,
                weights,
                quant_block_size,
                scale_fmt,
                topk_tokens,
                head_dim,
                max_model_len,
                total_seq_lens,
                topk_indices_buffer,
                skip_k_cache_insert,
                use_pcp,
                dense_mha_metadata_layer_name,
                use_fp4_cache,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
                skip_topk_buffer_clear,
                candidate_blocks,
                candidate_block_size,
                candidate_write,
                topk_backend,
            )

        # Prefill (and mixed prefill+decode) on candidate-masked FP8 layer with seq_len > 16384!
        slot_mapping = attn_meta.slot_mapping
        if not skip_k_cache_insert:
            assert k is not None and scale_fmt is not None
            ops.indexer_k_quant_and_cache(
                k[: slot_mapping.shape[0]],
                kv_cache,
                slot_mapping,
                quant_block_size,
                scale_fmt,
            )

        if forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
            dense_mha_layer = _resolve_layer_name(dense_mha_metadata_layer_name)
            if dense_mha_layer:
                mla_metadata = attn_metadata.get(dense_mha_layer)
                mla_prefill = getattr(mla_metadata, "prefill", None)
                if (
                    getattr(mla_prefill, "use_dense_mha", False)
                    and getattr(mla_metadata, "num_decode_tokens", -1) == 0
                ):
                    return topk_indices_buffer

        if not skip_topk_buffer_clear:
            topk_indices_buffer[: hidden_states.shape[0]] = -1

        fp8_dtype = current_platform.fp8_dtype()
        values_spec, scales_spec = sai._gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, False
        )
        k_quant_full, k_scale_full = current_workspace_manager().get_simultaneous(
            values_spec, scales_spec
        )
        from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
            apply_candidate_mask as _apply_candidate_mask,
        )
        from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

        for chunk in prefill_metadata.chunks:
            cu_seqlen_ks = chunk.cu_seqlen_ks
            cu_seqlen_ke = chunk.cu_seqlen_ke
            k_quant = k_quant_full[: chunk.max_local_total_seq_lens]
            k_scale = k_scale_full[: chunk.max_local_total_seq_lens]
            if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.local_cu_seq_lens,
                )
            q_slice = q_quant[chunk.token_start : chunk.token_end]
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            if chunk.local_total_seq_lens == 0 or q_slice.shape[0] == 0:
                topk_indices.fill_(-1)
                continue
            chunk_candidates = candidate_blocks[chunk.token_start : chunk.token_end]
            k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
            used_fast = run_candidate_sparse_prefill_topk(
                q_slice=q_slice,
                k_quant=k_quant,
                k_scale=k_scale_cast,
                weights=weights[chunk.token_start : chunk.token_end],
                chunk_candidates=chunk_candidates,
                cu_seqlen_ks=cu_seqlen_ks,
                cu_seqlen_ke=cu_seqlen_ke,
                prefill_topk=topk_indices,
                topk_tokens=topk_tokens,
                candidate_block_size=candidate_block_size,
                local_total_seq_lens=int(chunk.local_total_seq_lens),
            )
            if not used_fast:
                logits = fp8_fp4_mqa_logits(
                    (q_slice, None),
                    (k_quant, k_scale_cast),
                    weights[chunk.token_start : chunk.token_end],
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    clean_logits=False,
                )
                _apply_candidate_mask(
                    logits,
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    chunk_candidates,
                    candidate_block_size,
                )
                ops.top_k_per_row_prefill(
                    logits,
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    topk_indices,
                    logits.shape[0],
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
        if has_decode:
            saved_prefill = attn_meta.prefill
            saved_num_prefills = attn_meta.num_prefills
            attn_meta.prefill = None
            attn_meta.num_prefills = 0
            try:
                return orig_sparse_attn_indexer(
                    hidden_states,
                    k_cache_prefix,
                    kv_cache,
                    q_quant,
                    q_scale,
                    k,
                    weights,
                    quant_block_size,
                    scale_fmt,
                    topk_tokens,
                    head_dim,
                    max_model_len,
                    total_seq_lens,
                    topk_indices_buffer,
                    True,  # skip_k_cache_insert (already inserted above)
                    use_pcp,
                    dense_mha_metadata_layer_name,
                    use_fp4_cache,
                    dcp_rank,
                    dcp_world_size,
                    cp_kv_cache_interleave_size,
                    True,  # skip_topk_buffer_clear (prefill rows already filled above)
                    candidate_blocks,
                    candidate_block_size,
                    candidate_write,
                    topk_backend,
                )
            finally:
                attn_meta.prefill = saved_prefill
                attn_meta.num_prefills = saved_num_prefills
        return topk_indices_buffer

    sai.sparse_attn_indexer = _patched_sparse_attn_indexer
    # Also update the registered custom op implementation so torch.ops.vllm.sparse_attn_indexer uses it
    try:
        orig_fwd = sai.SparseAttnIndexer.forward_cuda

        def _fwd_cuda_with_cand_mqa(
            self,
            hidden_states: torch.Tensor,
            q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
            k: torch.Tensor | None,
            weights: torch.Tensor,
        ) -> torch.Tensor:
            if (
                not self.use_fp4_cache
                and not self.use_pcp
                and self.dcp_world_size == 1
                and self.candidate_blocks is not None
                and not self.candidate_write
                and self.candidate_block_size == 8
                and self.head_dim == 128
                and not torch.cuda.is_current_stream_capturing()
            ):
                q_vals = q_quant[0] if isinstance(q_quant, tuple) else q_quant
                return _patched_sparse_attn_indexer(
                    hidden_states,
                    self.k_cache.prefix,
                    self.k_cache.kv_cache,
                    q_vals,
                    None,
                    k,
                    weights,
                    self.quant_block_size,
                    self.scale_fmt,
                    self.topk_tokens,
                    self.head_dim,
                    self.max_model_len,
                    self.max_total_seq_len,
                    self.topk_indices_buffer,
                    self.skip_k_cache_insert,
                    self.use_pcp,
                    self.dense_mha_metadata_layer_name,
                    False,
                    self.dcp_rank,
                    self.dcp_world_size,
                    self.cp_kv_cache_interleave_size,
                    getattr(self, "skip_topk_buffer_clear", False),
                    self.candidate_blocks,
                    self.candidate_block_size,
                    False,
                    getattr(self, "topk_backend", "auto"),
                )
            return orig_fwd(self, hidden_states, q_quant, k, weights)

        if not getattr(sai.SparseAttnIndexer, "_zstep_sm120_dsa_patched", False):
            sai.SparseAttnIndexer.forward_cuda = _fwd_cuda_with_cand_mqa
            if model is not None:
                rebound = 0
                for mod in model.modules():
                    if isinstance(mod, sai.SparseAttnIndexer):
                        mod._forward_method = types.MethodType(_fwd_cuda_with_cand_mqa, mod)
                        rebound += 1
                _LOG.info(
                    "zstep_sm120_candidate_mqa: rebound %d SparseAttnIndexer modules",
                    rebound,
                )
    except Exception as exc:
        _LOG.warning("Failed to hook SparseAttnIndexer.forward_cuda for candidate MQA: %s", exc)

    sai._zstep_sm120_candidate_mqa_patched = True
    return True
