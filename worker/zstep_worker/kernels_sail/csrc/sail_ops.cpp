// Copyright (c) 2026 zstep Contributors
// SPDX-License-Identifier: MIT

#include <torch/extension.h>
#include <cstdint>
#include <stdexcept>

std::uintptr_t register_mapped_host_cuda(std::uintptr_t host_address, int64_t bytes);
void unregister_mapped_host_cuda(std::uintptr_t host_address);
int64_t enable_peer_access_cuda(int peer_device);
torch::Tensor alloc_ipc_buffer_cuda(int64_t bytes, torch::Tensor out_ptr_u64);
std::uintptr_t open_ipc_handle_cuda(torch::Tensor handle_u8);
void close_ipc_handle_cuda(std::uintptr_t ptr);
void free_ipc_buffer_cuda(std::uintptr_t ptr);
void launch_bmm_bf16_f32_nt_cuda(
    const void* a_bf16_ptr,
    const void* b_bf16_ptr,
    float* c_f32_ptr,
    int batch,
    int m,
    int n,
    int k);

void launch_icn_allreduce_bf16_cuda(
    const void* input_ptr,
    void* output_ptr,
    uint64_t* control_ptr,
    const std::uintptr_t* peer_ipc_ptrs,
    std::uintptr_t abort_dev_ptr,
    int64_t slot_stride_bytes,
    int64_t oneshot_max_bytes,
    int64_t num_vecs,
    int rank,
    int world_size,
    int blocks,
    int threads,
    std::uintptr_t stream_handle);

void launch_icn_allreduce_f32_cuda(
    const float* input_ptr,
    float* output_ptr,
    uint64_t* control_ptr,
    const std::uintptr_t* peer_ipc_ptrs,
    std::uintptr_t abort_dev_ptr,
    int64_t slot_stride_bytes,
    int64_t oneshot_max_bytes,
    int64_t numel,
    int rank,
    int world_size,
    int blocks,
    int threads,
    std::uintptr_t stream_handle);

void launch_icn_allgather_bf16_cuda(
    const void* input_ptr,
    void* output_ptr,
    uint64_t* control_ptr,
    const std::uintptr_t* peer_ipc_ptrs,
    std::uintptr_t abort_dev_ptr,
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

void launch_icn_tp8_top2_gather_cuda(
    const float* local_top2_vals,
    const int32_t* local_top2_ids,
    int32_t* out_top2_tokens,
    uint32_t* out_top2_logit_bits,
    uint64_t* step_counter,
    const std::uintptr_t* peer_ipc_ptrs,
    std::uintptr_t abort_dev_ptr,
    int64_t ipc_base_offset,
    int64_t num_rows,
    int64_t max_rows_per_step,
    int rank,
    int world_size,
    int vocab_per_rank,
    std::uintptr_t stream_handle);

void launch_decode_prepare_tokens_cuda(
    int32_t* dec_tokens,
    int32_t* dec_lookback,
    const int32_t* slot_last_tok,
    const int32_t* slot_lookback,
    const int32_t* dec_req_slots,
    const int32_t* dec_mode,
    int bucket,
    int max_seqs,
    std::uintptr_t stream_handle);

void launch_decode_commit_tokens_cuda(
    int32_t* slot_last_tok,
    int32_t* slot_lookback,
    const int32_t* dec_tokens,
    const int32_t* dec_lookback,
    const int32_t* dec_req_slots,
    const int32_t* sampled_tokens,
    int num_active,
    int max_seqs,
    std::uintptr_t stream_handle);

void launch_rmsnorm_bf16_cuda(
    const void* x_ptr,
    const void* weight_ptr,
    void* out_ptr,
    int64_t rows,
    int dim,
    float eps,
    std::uintptr_t stream_handle);

void launch_mhc_fused_mixes_bf16_cuda(
    const void* x_ptr,
    const float* hc_fn_ptr,
    const float* hc_scale_ptr,
    const float* hc_base_ptr,
    float* pre_out_ptr,
    float* post_out_ptr,
    float* comb_out_ptr,
    int64_t rows,
    float norm_eps,
    float hc_eps,
    int sinkhorn_iters,
    std::uintptr_t stream_handle);

void launch_mhc_fused_mixes_splitk_bf16_cuda(
    const void* x_ptr,
    const float* hc_fn_ptr,
    const float* hc_scale_ptr,
    const float* hc_base_ptr,
    float* partial_out_ptr,
    float* pre_out_ptr,
    float* post_out_ptr,
    float* comb_out_ptr,
    int64_t rows,
    int split_k,
    float norm_eps,
    float hc_eps,
    int sinkhorn_iters,
    std::uintptr_t stream_handle);

void launch_mhc_sinkhorn_from_proj_cuda(
    const void* x_ptr,
    const float* raw_proj_ptr,
    const float* hc_scale_ptr,
    const float* hc_base_ptr,
    float* pre_out_ptr,
    float* post_out_ptr,
    float* comb_out_ptr,
    int64_t rows,
    float norm_eps,
    float hc_eps,
    int sinkhorn_iters,
    std::uintptr_t stream_handle);

void launch_mhc_pre_rmsnorm_bf16_cuda(
    const void* x_ptr,
    const float* pre_mix_ptr,
    const void* norm_weight_ptr,
    void* normed_out_ptr,
    void* aux_out_ptr,
    int64_t rows,
    int dim,
    int write_aux,
    float norm_eps,
    std::uintptr_t stream_handle);

void launch_mhc_post_bf16_cuda(
    const void* sub_out_ptr,
    const void* residual_ptr,
    const float* post_mix_ptr,
    const float* comb_mix_ptr,
    void* out_ptr,
    int64_t rows,
    int dim,
    std::uintptr_t stream_handle);

void launch_inplace_act_quant_ue8m0_bf16_cuda(
    void* x_ptr,
    int64_t num_groups,
    std::uintptr_t stream_handle);

void launch_act_quant_ue8m0_bf16_cuda(
    const void* src_ptr,
    void* dst_ptr,
    int64_t num_groups,
    std::uintptr_t stream_handle);

void launch_shared_swiglu_quant_bf16_cuda(
    const void* sh_gu_ptr,
    void* sh_act_out_ptr,
    int64_t rows,
    int inter_dim,
    float swiglu_limit,
    int do_ue8m0,
    std::uintptr_t stream_handle);

void launch_inplace_fp4_quant_bf16_cuda(
    void* x_ptr,
    int64_t num_groups,
    int block_size,
    int is_e4m3_scale,
    std::uintptr_t stream_handle);

void launch_rope_inplace_bf16_cuda(
    void* x_ptr,
    const float* cos_sin_ptr,
    const int32_t* positions_ptr,
    int64_t rows,
    int num_heads,
    int head_dim,
    int rope_dim,
    int inverse,
    std::uintptr_t stream_handle);

void launch_dsv41_router_topk_cuda(
    const void* logits_ptr,
    const float* bias_ptr,
    float* out_weights_ptr,
    int32_t* out_indices_ptr,
    int64_t rows,
    int num_experts,
    int top_k,
    float route_scale,
    int logits_is_bf16,
    std::uintptr_t stream_handle);

void launch_mxfp4_moe_gemv_gate_up_cuda(
    const void* x_ptr,
    const uint8_t* w13_packed_ptr,
    const uint8_t* w13_scale_ptr,
    const int32_t* topk_ids_ptr,
    const float* topk_weights_ptr,
    void* out_inter_ptr,
    int rows,
    int top_k,
    int inter_per_rank,
    int hidden_dim,
    float swiglu_limit,
    int threads,
    std::uintptr_t stream_handle);

void launch_mxfp4_moe_gemv_down_combine_cuda(
    const void* inter_ptr,
    const uint8_t* w2_packed_ptr,
    const uint8_t* w2_scale_ptr,
    const int32_t* topk_ids_ptr,
    const void* shared_out_ptr,
    void* out_ptr,
    int rows,
    int top_k,
    int hidden_dim,
    int inter_per_rank,
    int threads,
    std::uintptr_t stream_handle);

void launch_moe_align_block_size_cuda(
    const int32_t* topk_ids_ptr,
    int32_t* sorted_pair_ids_ptr,
    int32_t* expert_ids_per_tile_ptr,
    int32_t* num_valid_tiles_ptr,
    int64_t num_pairs,
    int64_t max_padded_slots,
    int num_experts,
    int block_m,
    std::uintptr_t stream_handle);

void launch_engram_lookup_dequant_cuda(
    const int32_t* token_ids_ptr,
    const int32_t* prev_window_ptr,
    const int32_t* query_start_loc_ptr,
    const int64_t* token_map_ptr,
    const int64_t* multipliers_ptr,
    const int64_t* primes_ptr,
    const int64_t* offsets_ptr,
    const uint8_t* weight_fp8_ptr,
    const uint8_t* scale_u8_ptr,
    void* out_bf16_ptr,
    int num_seqs,
    int total_tokens,
    int64_t vocab_start_idx,
    int64_t vocab_end_idx,
    int64_t pad_compressed_id,
    std::uintptr_t stream_handle);

void launch_mhc_pre_rmsnorm_quant_bf16_cuda(
    const void* x_ptr,
    const float* pre_mix_ptr,
    const void* norm_weight_ptr,
    void* normed_out_ptr,
    void* quant_out_ptr,
    int64_t rows,
    int dim,
    int write_normed,
    float norm_eps,
    std::uintptr_t stream_handle);

void launch_fused_qkv_a_epilogue_bf16_cuda(
    const void* qkv_a_ptr,
    const void* q_norm_w_ptr,
    const void* kv_norm_w_ptr,
    const float* cos_sin_ptr,
    const int32_t* positions_ptr,
    const int64_t* swa_write_slots_ptr,
    void* qr_unquant_out_ptr,
    void* qr_quant_out_ptr,
    void* kv_arena_ptr,
    int64_t rows,
    int q_rank,
    int kv_dim,
    int rope_dim,
    int write_qr_unquant,
    int64_t layer_swa_base,
    float eps,
    std::uintptr_t stream_handle);

void launch_fused_q_idx_epilogue_bf16_cuda(
    const void* q_and_idx_ptr,
    const float* cos_sin_ptr,
    const int32_t* positions_ptr,
    void* q_out_ptr,
    void* q_idx_out_ptr,
    int64_t rows,
    std::uintptr_t stream_handle);

void launch_fused_inv_rope_act_quant_bf16_cuda(
    void* attn_o_ptr,
    const float* cos_sin_ptr,
    const int32_t* positions_ptr,
    int64_t rows,
    int num_heads,
    int head_dim,
    int rope_dim,
    std::uintptr_t stream_handle);

void launch_moe_gather_tile_inputs_bf16_cuda(
    const void* x_ptr,
    const int32_t* sorted_pair_ids_ptr,
    const int32_t* num_valid_tiles_ptr,
    void* tile_a_ptr,
    int tile_offset,
    int chunk_tiles,
    int block_m,
    int top_k,
    int hidden_dim,
    std::uintptr_t stream_handle);

void launch_dequant_mxfp4_tiles_bf16_cuda(
    const uint8_t* w_packed_ptr,
    const uint8_t* w_scale_ptr,
    const int32_t* expert_ids_ptr,
    const int32_t* num_valid_tiles_ptr,
    void* out_bf16_ptr,
    int tile_offset,
    int chunk_tiles,
    int n_rows,
    int k_cols,
    std::uintptr_t stream_handle);

void launch_moe_tile_swiglu_quant_bf16_cuda(
    const void* tile_gu_ptr,
    const int32_t* sorted_pair_ids_ptr,
    const int32_t* num_valid_tiles_ptr,
    const float* topk_weights_ptr,
    void* tile_inter_out_ptr,
    int tile_offset,
    int chunk_tiles,
    int block_m,
    int inter_dim,
    float swiglu_limit,
    int do_ue8m0,
    std::uintptr_t stream_handle);

void launch_moe_scatter_tile_outputs_bf16_cuda(
    const void* tile_dn_ptr,
    const int32_t* sorted_pair_ids_ptr,
    const int32_t* num_valid_tiles_ptr,
    void* pair_out_ptr,
    int tile_offset,
    int chunk_tiles,
    int block_m,
    int hidden_dim,
    std::uintptr_t stream_handle);

void launch_moe_combine_pairs_shared_bf16_cuda(
    const void* pair_out_ptr,
    const void* shared_out_ptr,
    void* out_ptr,
    int rows,
    int top_k,
    int hidden_dim,
    std::uintptr_t stream_handle);

void launch_engram_gated_residual_bf16_cuda(
    void* h_stream_ptr,
    const void* kv_proj_ptr,
    const float* qk_prod_ptr,
    int rows,
    int hidden_size,
    float norm_eps,
    std::uintptr_t stream_handle);

void launch_indexer_paged_score_bf16_cuda(
    const void* q_idx_ptr,
    const void* w_idx_ptr,
    const void* k_cache_ptr,
    const int32_t* mla_bt_ptr,
    const int64_t* row_seq_ids_ptr,
    const int32_t* positions_ptr,
    float* scores_out_ptr,
    int rows,
    float weight_scale,
    int compress_ratio,
    int lat_pp,
    int bt_width,
    int eff_max_c,
    std::uintptr_t stream_handle);

void launch_indexer_topk512_slots_cuda(
    const float* scores_red_ptr,
    const int32_t* mla_bt_ptr,
    const int64_t* row_seq_ids_ptr,
    const int32_t* positions_ptr,
    int32_t* shared_full_topk_ptr,
    int rows,
    int compress_ratio,
    int lat_pp,
    int bt_width,
    int eff_max_c,
    int mla_arena_base,
    std::uintptr_t stream_handle);

void launch_sparse_attn_tp8_bf16_cuda(
    const void* q_ptr,
    const void* kv_arena_ptr,
    const float* attn_sink_ptr,
    const int32_t* topk_idxs_ptr,
    float* partial_acc_ptr,
    float* partial_m_ptr,
    float* partial_l_ptr,
    void* out_ptr,
    int rows,
    int max_topk,
    float sm_scale,
    std::uintptr_t stream_handle);

// ---------------------------------------------------------------------------
// PyBind11 / Torch Entrypoints
// ---------------------------------------------------------------------------

std::uintptr_t register_mapped_host(std::uintptr_t host_address, int64_t bytes) {
    return register_mapped_host_cuda(host_address, bytes);
}

void unregister_mapped_host(std::uintptr_t host_address) {
    unregister_mapped_host_cuda(host_address);
}

int64_t enable_peer_access(int64_t peer_device) {
    return enable_peer_access_cuda(static_cast<int>(peer_device));
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

void launch_icn_allreduce_bf16(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor control,
    torch::Tensor peer_ipc_ptrs,
    std::uintptr_t abort_dev_ptr,
    int64_t slot_stride_bytes,
    int64_t oneshot_max_bytes,
    int64_t rank,
    int64_t world_size,
    int64_t blocks,
    int64_t threads,
    std::uintptr_t stream_handle) {
    TORCH_CHECK(input.is_cuda() && output.is_cuda(), "input/output must be CUDA tensors");
    TORCH_CHECK(
        input.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16,
        "BF16 required");
    TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "contiguous required");
    TORCH_CHECK(input.numel() == output.numel(), "numel mismatch");
    TORCH_CHECK(input.numel() % 8 == 0, "numel must be divisible by 8");
    TORCH_CHECK(blocks >= 1 && blocks <= 64, "blocks out of range");
    launch_icn_allreduce_bf16_cuda(
        input.data_ptr(),
        output.data_ptr(),
        reinterpret_cast<uint64_t*>(control.data_ptr<int64_t>()),
        reinterpret_cast<const std::uintptr_t*>(peer_ipc_ptrs.data_ptr<int64_t>()),
        abort_dev_ptr,
        slot_stride_bytes,
        oneshot_max_bytes,
        input.numel() / 8,
        static_cast<int>(rank),
        static_cast<int>(world_size),
        static_cast<int>(blocks),
        static_cast<int>(threads),
        stream_handle);
}

void launch_icn_allreduce_f32(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor control,
    torch::Tensor peer_ipc_ptrs,
    std::uintptr_t abort_dev_ptr,
    int64_t slot_stride_bytes,
    int64_t oneshot_max_bytes,
    int64_t rank,
    int64_t world_size,
    int64_t blocks,
    int64_t threads,
    std::uintptr_t stream_handle) {
    TORCH_CHECK(input.is_cuda() && output.is_cuda(), "input/output must be CUDA tensors");
    TORCH_CHECK(
        input.scalar_type() == at::kFloat && output.scalar_type() == at::kFloat,
        "FP32 required");
    TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "contiguous required");
    TORCH_CHECK(input.numel() == output.numel(), "numel mismatch");
    TORCH_CHECK(blocks >= 1 && blocks <= 64, "blocks out of range");
    launch_icn_allreduce_f32_cuda(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        reinterpret_cast<uint64_t*>(control.data_ptr<int64_t>()),
        reinterpret_cast<const std::uintptr_t*>(peer_ipc_ptrs.data_ptr<int64_t>()),
        abort_dev_ptr,
        slot_stride_bytes,
        oneshot_max_bytes,
        input.numel(),
        static_cast<int>(rank),
        static_cast<int>(world_size),
        static_cast<int>(blocks),
        static_cast<int>(threads),
        stream_handle);
}

void launch_icn_allgather_bf16(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor control,
    torch::Tensor peer_ipc_ptrs,
    std::uintptr_t abort_dev_ptr,
    int64_t slot_stride_bytes,
    int64_t outer_rows,
    int64_t inner_elems,
    int64_t rank,
    int64_t world_size,
    int64_t blocks,
    int64_t threads,
    std::uintptr_t stream_handle) {
    TORCH_CHECK(input.is_cuda() && output.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(
        input.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16,
        "BF16 required");
    TORCH_CHECK(inner_elems % 8 == 0, "inner_elems must be divisible by 8");
    launch_icn_allgather_bf16_cuda(
        input.data_ptr(),
        output.data_ptr(),
        reinterpret_cast<uint64_t*>(control.data_ptr<int64_t>()),
        reinterpret_cast<const std::uintptr_t*>(peer_ipc_ptrs.data_ptr<int64_t>()),
        abort_dev_ptr,
        slot_stride_bytes,
        outer_rows,
        inner_elems / 8,
        static_cast<int>(rank),
        static_cast<int>(world_size),
        static_cast<int>(blocks),
        static_cast<int>(threads),
        stream_handle);
}

void launch_local_top2_bf16(
    torch::Tensor logits,
    torch::Tensor out_vals,
    torch::Tensor out_ids,
    int64_t num_rows,
    std::uintptr_t stream_handle) {
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kBFloat16, "BF16 CUDA logits");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
    if (num_rows <= 0) return;
    launch_local_top2_bf16_cuda(
        logits.data_ptr(),
        out_vals.data_ptr<float>(),
        out_ids.data_ptr<int32_t>(),
        num_rows,
        logits.stride(0),
        static_cast<int>(logits.size(1)),
        stream_handle);
}

void launch_icn_tp8_top2_gather(
    torch::Tensor local_vals,
    torch::Tensor local_ids,
    torch::Tensor out_tokens,
    torch::Tensor out_logit_bits,
    torch::Tensor step_counter,
    torch::Tensor peer_ipc_ptrs,
    std::uintptr_t abort_dev_ptr,
    int64_t ipc_base_offset,
    int64_t num_rows,
    int64_t max_rows_per_step,
    int64_t rank,
    int64_t world_size,
    int64_t vocab_per_rank,
    std::uintptr_t stream_handle) {
    if (num_rows <= 0) return;
    launch_icn_tp8_top2_gather_cuda(
        local_vals.data_ptr<float>(),
        local_ids.data_ptr<int32_t>(),
        out_tokens.data_ptr<int32_t>(),
        reinterpret_cast<uint32_t*>(out_logit_bits.data_ptr<int32_t>()),
        reinterpret_cast<uint64_t*>(step_counter.data_ptr<int64_t>()),
        reinterpret_cast<const std::uintptr_t*>(peer_ipc_ptrs.data_ptr<int64_t>()),
        abort_dev_ptr,
        ipc_base_offset,
        num_rows,
        max_rows_per_step,
        static_cast<int>(rank),
        static_cast<int>(world_size),
        static_cast<int>(vocab_per_rank),
        stream_handle);
}

void launch_decode_prepare_tokens(
    torch::Tensor dec_tokens,
    torch::Tensor dec_lookback,
    torch::Tensor slot_last_tok,
    torch::Tensor slot_lookback,
    torch::Tensor dec_req_slots,
    torch::Tensor dec_mode,
    int64_t bucket,
    int64_t max_seqs,
    std::uintptr_t stream_handle) {
    launch_decode_prepare_tokens_cuda(
        dec_tokens.data_ptr<int32_t>(),
        dec_lookback.data_ptr<int32_t>(),
        slot_last_tok.data_ptr<int32_t>(),
        slot_lookback.data_ptr<int32_t>(),
        dec_req_slots.data_ptr<int32_t>(),
        dec_mode.data_ptr<int32_t>(),
        static_cast<int>(bucket),
        static_cast<int>(max_seqs),
        stream_handle);
}

void launch_decode_commit_tokens(
    torch::Tensor slot_last_tok,
    torch::Tensor slot_lookback,
    torch::Tensor dec_tokens,
    torch::Tensor dec_lookback,
    torch::Tensor dec_req_slots,
    torch::Tensor sampled_tokens,
    int64_t num_active,
    int64_t max_seqs,
    std::uintptr_t stream_handle) {
    launch_decode_commit_tokens_cuda(
        slot_last_tok.data_ptr<int32_t>(),
        slot_lookback.data_ptr<int32_t>(),
        dec_tokens.data_ptr<int32_t>(),
        dec_lookback.data_ptr<int32_t>(),
        dec_req_slots.data_ptr<int32_t>(),
        sampled_tokens.data_ptr<int32_t>(),
        static_cast<int>(num_active),
        static_cast<int>(max_seqs),
        stream_handle);
}

void launch_rmsnorm_bf16(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out,
    double eps,
    std::uintptr_t stream_handle) {
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && out.is_contiguous(), "contiguous");
    const int dim = static_cast<int>(x.size(-1));
    const int64_t rows = x.numel() / dim;
    if (rows <= 0) return;
    launch_rmsnorm_bf16_cuda(
        x.data_ptr(),
        weight.data_ptr(),
        out.data_ptr(),
        rows,
        dim,
        static_cast<float>(eps),
        stream_handle);
}

void launch_mhc_fused_mixes_bf16(
    torch::Tensor x,
    torch::Tensor hc_fn,
    torch::Tensor hc_scale,
    torch::Tensor hc_base,
    torch::Tensor pre_out,
    torch::Tensor post_out,
    torch::Tensor comb_out,
    double norm_eps,
    double hc_eps,
    int64_t sinkhorn_iters,
    std::uintptr_t stream_handle) {
    const int64_t rows = x.size(0);
    if (rows <= 0) return;
    launch_mhc_fused_mixes_bf16_cuda(
        x.data_ptr(),
        hc_fn.data_ptr<float>(),
        hc_scale.data_ptr<float>(),
        hc_base.data_ptr<float>(),
        pre_out.data_ptr<float>(),
        post_out.data_ptr<float>(),
        comb_out.data_ptr<float>(),
        rows,
        static_cast<float>(norm_eps),
        static_cast<float>(hc_eps),
        static_cast<int>(sinkhorn_iters),
        stream_handle);
}

void launch_mhc_fused_mixes_splitk_bf16(
    torch::Tensor x,
    torch::Tensor hc_fn,
    torch::Tensor hc_scale,
    torch::Tensor hc_base,
    torch::Tensor partial_out,
    torch::Tensor pre_out,
    torch::Tensor post_out,
    torch::Tensor comb_out,
    int64_t split_k,
    double norm_eps,
    double hc_eps,
    int64_t sinkhorn_iters,
    std::uintptr_t stream_handle) {
    const int64_t rows = x.size(0);
    if (rows <= 0) return;
    TORCH_CHECK(
        split_k == 1 || split_k == 2 || split_k == 4 || split_k == 8 || split_k == 16,
        "split_k must be 1, 2, 4, 8, or 16");
    if (split_k > 1) {
        TORCH_CHECK(partial_out.numel() >= rows * split_k * 25, "partial_out buffer too small");
    }
    launch_mhc_fused_mixes_splitk_bf16_cuda(
        x.data_ptr(),
        hc_fn.data_ptr<float>(),
        hc_scale.data_ptr<float>(),
        hc_base.data_ptr<float>(),
        partial_out.data_ptr<float>(),
        pre_out.data_ptr<float>(),
        post_out.data_ptr<float>(),
        comb_out.data_ptr<float>(),
        rows,
        static_cast<int>(split_k),
        static_cast<float>(norm_eps),
        static_cast<float>(hc_eps),
        static_cast<int>(sinkhorn_iters),
        stream_handle);
}

void launch_mhc_sinkhorn_from_proj(
    torch::Tensor x,
    torch::Tensor raw_proj,
    torch::Tensor hc_scale,
    torch::Tensor hc_base,
    torch::Tensor pre_out,
    torch::Tensor post_out,
    torch::Tensor comb_out,
    double norm_eps,
    double hc_eps,
    int64_t sinkhorn_iters,
    std::uintptr_t stream_handle) {
    const int64_t rows = x.size(0);
    if (rows <= 0) return;
    launch_mhc_sinkhorn_from_proj_cuda(
        x.data_ptr(),
        raw_proj.data_ptr<float>(),
        hc_scale.data_ptr<float>(),
        hc_base.data_ptr<float>(),
        pre_out.data_ptr<float>(),
        post_out.data_ptr<float>(),
        comb_out.data_ptr<float>(),
        rows,
        static_cast<float>(norm_eps),
        static_cast<float>(hc_eps),
        static_cast<int>(sinkhorn_iters),
        stream_handle);
}

void launch_mhc_pre_rmsnorm_bf16(
    torch::Tensor x,
    torch::Tensor pre_mix,
    torch::Tensor norm_weight,
    torch::Tensor normed_out,
    torch::Tensor aux_out,
    int64_t write_aux,
    double norm_eps,
    std::uintptr_t stream_handle) {
    const int64_t rows = x.size(0);
    const int dim = static_cast<int>(x.size(-1));
    if (rows <= 0) return;
    launch_mhc_pre_rmsnorm_bf16_cuda(
        x.data_ptr(),
        pre_mix.data_ptr<float>(),
        norm_weight.data_ptr(),
        normed_out.data_ptr(),
        write_aux ? aux_out.data_ptr() : nullptr,
        rows,
        dim,
        static_cast<int>(write_aux),
        static_cast<float>(norm_eps),
        stream_handle);
}

void launch_mhc_post_bf16(
    torch::Tensor sub_out,
    torch::Tensor residual,
    torch::Tensor post_mix,
    torch::Tensor comb_mix,
    torch::Tensor out,
    std::uintptr_t stream_handle) {
    const int64_t rows = sub_out.size(0);
    const int dim = static_cast<int>(sub_out.size(-1));
    if (rows <= 0) return;
    launch_mhc_post_bf16_cuda(
        sub_out.data_ptr(),
        residual.data_ptr(),
        post_mix.data_ptr<float>(),
        comb_mix.data_ptr<float>(),
        out.data_ptr(),
        rows,
        dim,
        stream_handle);
}

void launch_inplace_act_quant_ue8m0_bf16(
    torch::Tensor x,
    std::uintptr_t stream_handle) {
    TORCH_CHECK(x.is_contiguous() && x.scalar_type() == at::kBFloat16, "contiguous BF16 required");
    TORCH_CHECK(x.numel() % 32 == 0, "numel must be divisible by 32");
    if (x.numel() == 0) return;
    launch_inplace_act_quant_ue8m0_bf16_cuda(x.data_ptr(), x.numel() / 32, stream_handle);
}

void launch_act_quant_ue8m0_bf16(
    torch::Tensor src,
    torch::Tensor dst,
    std::uintptr_t stream_handle) {
    TORCH_CHECK(
        src.is_contiguous() && dst.is_contiguous()
            && src.scalar_type() == at::kBFloat16 && dst.scalar_type() == at::kBFloat16,
        "contiguous BF16 src/dst required");
    TORCH_CHECK(src.numel() == dst.numel() && src.numel() % 32 == 0, "numel mismatch or %32 != 0");
    if (src.numel() == 0) return;
    launch_act_quant_ue8m0_bf16_cuda(src.data_ptr(), dst.data_ptr(), src.numel() / 32, stream_handle);
}

void launch_shared_swiglu_quant_bf16(
    torch::Tensor sh_gu,
    torch::Tensor sh_act_out,
    double swiglu_limit,
    int64_t do_ue8m0,
    std::uintptr_t stream_handle) {
    TORCH_CHECK(
        sh_gu.is_contiguous() && sh_act_out.is_contiguous()
            && sh_gu.scalar_type() == at::kBFloat16
            && sh_act_out.scalar_type() == at::kBFloat16,
        "contiguous BF16 sh_gu/sh_act_out required");
    const int64_t rows = sh_act_out.size(0);
    if (rows <= 0) return;
    const int inter_dim = static_cast<int>(sh_act_out.size(1));
    TORCH_CHECK(inter_dim % 32 == 0 && sh_gu.size(1) == 2 * inter_dim, "invalid shared SwiGLU dims");
    launch_shared_swiglu_quant_bf16_cuda(
        sh_gu.data_ptr(),
        sh_act_out.data_ptr(),
        rows,
        inter_dim,
        static_cast<float>(swiglu_limit),
        static_cast<int>(do_ue8m0),
        stream_handle);
}

void launch_inplace_fp4_quant_bf16(
    torch::Tensor x,
    int64_t block_size,
    int64_t is_e4m3_scale,
    std::uintptr_t stream_handle) {
    TORCH_CHECK(x.is_contiguous() && x.scalar_type() == at::kBFloat16, "contiguous BF16 required");
    TORCH_CHECK(block_size == 16 || block_size == 32, "block_size must be 16 or 32");
    TORCH_CHECK(x.numel() % block_size == 0, "numel must be divisible by block_size");
    if (x.numel() == 0) return;
    launch_inplace_fp4_quant_bf16_cuda(
        x.data_ptr(),
        x.numel() / block_size,
        static_cast<int>(block_size),
        static_cast<int>(is_e4m3_scale),
        stream_handle);
}

void launch_rope_inplace_bf16(
    torch::Tensor x,
    torch::Tensor cos_sin_table,
    torch::Tensor positions,
    int64_t rope_dim,
    int64_t inverse,
    std::uintptr_t stream_handle) {
    TORCH_CHECK(x.is_contiguous() && x.scalar_type() == at::kBFloat16, "contiguous BF16 x");
    const int64_t rows = x.size(0);
    if (rows <= 0) return;
    int num_heads = 1;
    int head_dim = static_cast<int>(x.size(-1));
    if (x.dim() == 3) {
        num_heads = static_cast<int>(x.size(1));
    }
    launch_rope_inplace_bf16_cuda(
        x.data_ptr(),
        cos_sin_table.data_ptr<float>(),
        positions.data_ptr<int32_t>(),
        rows,
        num_heads,
        head_dim,
        static_cast<int>(rope_dim),
        static_cast<int>(inverse),
        stream_handle);
}

void launch_dsv41_router_topk(
    torch::Tensor logits,
    torch::Tensor bias,
    torch::Tensor out_weights,
    torch::Tensor out_indices,
    int64_t top_k,
    double route_scale,
    std::uintptr_t stream_handle) {
    const int64_t rows = logits.size(0);
    const int num_experts = static_cast<int>(logits.size(1));
    if (rows <= 0) return;
    const int is_bf16 = (logits.scalar_type() == at::kBFloat16) ? 1 : 0;
    launch_dsv41_router_topk_cuda(
        logits.data_ptr(),
        bias.data_ptr<float>(),
        out_weights.data_ptr<float>(),
        out_indices.data_ptr<int32_t>(),
        rows,
        num_experts,
        static_cast<int>(top_k),
        static_cast<float>(route_scale),
        is_bf16,
        stream_handle);
}

void launch_mxfp4_moe_gemv_gate_up(
    torch::Tensor x,
    torch::Tensor w13_packed,
    torch::Tensor w13_scale,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor out_inter,
    double swiglu_limit,
    int64_t threads,
    std::uintptr_t stream_handle) {
    const int rows = static_cast<int>(x.size(0));
    if (rows <= 0) return;
    const int hidden_dim = static_cast<int>(x.size(1));
    const int top_k = static_cast<int>(topk_ids.size(1));
    const int inter_per_rank = static_cast<int>(w13_packed.size(1) / 2);
    launch_mxfp4_moe_gemv_gate_up_cuda(
        x.data_ptr(),
        w13_packed.data_ptr<uint8_t>(),
        w13_scale.data_ptr<uint8_t>(),
        topk_ids.data_ptr<int32_t>(),
        topk_weights.data_ptr<float>(),
        out_inter.data_ptr(),
        rows,
        top_k,
        inter_per_rank,
        hidden_dim,
        static_cast<float>(swiglu_limit),
        static_cast<int>(threads),
        stream_handle);
}

void launch_mxfp4_moe_gemv_down_combine(
    torch::Tensor inter,
    torch::Tensor w2_packed,
    torch::Tensor w2_scale,
    torch::Tensor topk_ids,
    torch::Tensor shared_out,
    torch::Tensor out,
    int64_t top_k,
    int64_t threads,
    std::uintptr_t stream_handle) {
    const int rows = static_cast<int>(out.size(0));
    if (rows <= 0) return;
    const int hidden_dim = static_cast<int>(out.size(1));
    const int inter_per_rank = static_cast<int>(inter.size(-1));
    launch_mxfp4_moe_gemv_down_combine_cuda(
        inter.data_ptr(),
        w2_packed.data_ptr<uint8_t>(),
        w2_scale.data_ptr<uint8_t>(),
        topk_ids.data_ptr<int32_t>(),
        shared_out.data_ptr(),
        out.data_ptr(),
        rows,
        static_cast<int>(top_k),
        hidden_dim,
        inter_per_rank,
        static_cast<int>(threads),
        stream_handle);
}

void launch_moe_align_block_size(
    torch::Tensor topk_ids,
    torch::Tensor sorted_pair_ids,
    torch::Tensor expert_ids_per_tile,
    torch::Tensor num_valid_tiles_out,
    int64_t num_experts,
    int64_t block_m,
    std::uintptr_t stream_handle) {
    launch_moe_align_block_size_cuda(
        topk_ids.data_ptr<int32_t>(),
        sorted_pair_ids.data_ptr<int32_t>(),
        expert_ids_per_tile.data_ptr<int32_t>(),
        num_valid_tiles_out.data_ptr<int32_t>(),
        topk_ids.numel(),
        sorted_pair_ids.numel(),
        static_cast<int>(num_experts),
        static_cast<int>(block_m),
        stream_handle);
}

void launch_engram_lookup_dequant(
    torch::Tensor token_ids,
    torch::Tensor prev_window,
    torch::Tensor query_start_loc,
    torch::Tensor token_map,
    torch::Tensor multipliers,
    torch::Tensor primes,
    torch::Tensor offsets,
    torch::Tensor weight_fp8,
    torch::Tensor scale_u8,
    torch::Tensor out_bf16,
    int64_t num_seqs,
    int64_t vocab_start_idx,
    int64_t vocab_end_idx,
    int64_t pad_compressed_id,
    std::uintptr_t stream_handle) {
    const int total_tokens = static_cast<int>(token_ids.numel());
    if (total_tokens <= 0 || num_seqs <= 0) return;
    launch_engram_lookup_dequant_cuda(
        token_ids.data_ptr<int32_t>(),
        prev_window.data_ptr<int32_t>(),
        query_start_loc.data_ptr<int32_t>(),
        token_map.data_ptr<int64_t>(),
        multipliers.data_ptr<int64_t>(),
        primes.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>(),
        reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr()),
        reinterpret_cast<const uint8_t*>(scale_u8.data_ptr()),
        out_bf16.data_ptr(),
        static_cast<int>(num_seqs),
        total_tokens,
        vocab_start_idx,
        vocab_end_idx,
        pad_compressed_id,
        stream_handle);
}

void launch_sparse_attn_tp8_bf16(
    torch::Tensor q,
    torch::Tensor kv_arena,
    torch::Tensor attn_sink,
    torch::Tensor topk_idxs,
    torch::Tensor partial_acc,
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor out,
    double sm_scale,
    std::uintptr_t stream_handle) {
    const int rows = static_cast<int>(q.size(0));
    if (rows <= 0) return;
    const int max_topk = static_cast<int>(topk_idxs.size(-1));
    launch_sparse_attn_tp8_bf16_cuda(
        q.data_ptr(),
        kv_arena.data_ptr(),
        attn_sink.data_ptr<float>(),
        topk_idxs.data_ptr<int32_t>(),
        partial_acc.data_ptr<float>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        out.data_ptr(),
        rows,
        max_topk,
        static_cast<float>(sm_scale),
        stream_handle);
}

void launch_mhc_pre_rmsnorm_quant_bf16(
    torch::Tensor x,
    torch::Tensor pre_mix,
    torch::Tensor norm_weight,
    torch::Tensor normed_out,
    torch::Tensor quant_out,
    int64_t write_normed,
    double norm_eps,
    std::uintptr_t stream_handle) {
    const int64_t rows = x.size(0);
    const int dim = static_cast<int>(x.size(-1));
    if (rows <= 0) return;
    TORCH_CHECK(dim % 32 == 0, "dim must be divisible by 32");
    launch_mhc_pre_rmsnorm_quant_bf16_cuda(
        x.data_ptr(),
        pre_mix.data_ptr<float>(),
        norm_weight.data_ptr(),
        write_normed ? normed_out.data_ptr() : nullptr,
        quant_out.data_ptr(),
        rows,
        dim,
        static_cast<int>(write_normed),
        static_cast<float>(norm_eps),
        stream_handle);
}

void launch_fused_qkv_a_epilogue_bf16(
    torch::Tensor qkv_a,
    torch::Tensor q_norm_w,
    torch::Tensor kv_norm_w,
    torch::Tensor cos_sin,
    torch::Tensor positions,
    torch::Tensor swa_write_slots,
    torch::Tensor qr_unquant_out,
    torch::Tensor qr_quant_out,
    torch::Tensor kv_arena,
    int64_t rope_dim,
    int64_t write_qr_unquant,
    int64_t layer_swa_base,
    double eps,
    std::uintptr_t stream_handle) {
    const int64_t rows = qkv_a.size(0);
    if (rows <= 0) return;
    const int q_rank = static_cast<int>(q_norm_w.numel());
    const int kv_dim = static_cast<int>(kv_norm_w.numel());
    TORCH_CHECK(qkv_a.size(1) == q_rank + kv_dim && kv_dim == 512, "invalid qkv_a dims");
    launch_fused_qkv_a_epilogue_bf16_cuda(
        qkv_a.data_ptr(),
        q_norm_w.data_ptr(),
        kv_norm_w.data_ptr(),
        cos_sin.data_ptr<float>(),
        positions.data_ptr<int32_t>(),
        swa_write_slots.data_ptr<int64_t>(),
        write_qr_unquant ? qr_unquant_out.data_ptr() : nullptr,
        qr_quant_out.data_ptr(),
        kv_arena.data_ptr(),
        rows,
        q_rank,
        kv_dim,
        static_cast<int>(rope_dim),
        static_cast<int>(write_qr_unquant),
        layer_swa_base,
        static_cast<float>(eps),
        stream_handle);
}

void launch_fused_q_idx_epilogue_bf16(
    torch::Tensor q_and_idx,
    torch::Tensor cos_sin,
    torch::Tensor positions,
    torch::Tensor q_out,
    torch::Tensor q_idx_out,
    std::uintptr_t stream_handle) {
    const int64_t rows = q_and_idx.size(0);
    if (rows <= 0) return;
    TORCH_CHECK(q_and_idx.size(1) == 4608, "q_and_idx must have 4608 cols");
    launch_fused_q_idx_epilogue_bf16_cuda(
        q_and_idx.data_ptr(),
        cos_sin.data_ptr<float>(),
        positions.data_ptr<int32_t>(),
        q_out.data_ptr(),
        q_idx_out.data_ptr(),
        rows,
        stream_handle);
}

void launch_fused_inv_rope_act_quant_bf16(
    torch::Tensor attn_o,
    torch::Tensor cos_sin,
    torch::Tensor positions,
    int64_t rope_dim,
    std::uintptr_t stream_handle) {
    const int64_t rows = attn_o.size(0);
    if (rows <= 0) return;
    TORCH_CHECK(attn_o.dim() == 3 && attn_o.is_contiguous(), "contiguous 3D attn_o required");
    launch_fused_inv_rope_act_quant_bf16_cuda(
        attn_o.data_ptr(),
        cos_sin.data_ptr<float>(),
        positions.data_ptr<int32_t>(),
        rows,
        static_cast<int>(attn_o.size(1)),
        static_cast<int>(attn_o.size(2)),
        static_cast<int>(rope_dim),
        stream_handle);
}

void launch_moe_gather_tile_inputs_bf16(
    torch::Tensor x,
    torch::Tensor sorted_pair_ids,
    torch::Tensor num_valid_tiles,
    torch::Tensor tile_a_out,
    int64_t tile_offset,
    int64_t chunk_tiles,
    int64_t block_m,
    int64_t top_k,
    std::uintptr_t stream_handle) {
    if (chunk_tiles <= 0) return;
    const int hidden_dim = static_cast<int>(x.size(1));
    TORCH_CHECK(hidden_dim % 8 == 0, "hidden_dim must be divisible by 8");
    launch_moe_gather_tile_inputs_bf16_cuda(
        x.data_ptr(),
        sorted_pair_ids.data_ptr<int32_t>(),
        num_valid_tiles.data_ptr<int32_t>(),
        tile_a_out.data_ptr(),
        static_cast<int>(tile_offset),
        static_cast<int>(chunk_tiles),
        static_cast<int>(block_m),
        static_cast<int>(top_k),
        hidden_dim,
        stream_handle);
}

void launch_dequant_mxfp4_tiles_bf16(
    torch::Tensor w_packed,
    torch::Tensor w_scale,
    torch::Tensor expert_ids,
    torch::Tensor num_valid_tiles,
    torch::Tensor out_bf16,
    int64_t tile_offset,
    int64_t chunk_tiles,
    std::uintptr_t stream_handle) {
    if (chunk_tiles <= 0) return;
    const int n_rows = static_cast<int>(w_packed.size(1));
    const int k_cols = static_cast<int>(w_packed.size(2) * 2);
    launch_dequant_mxfp4_tiles_bf16_cuda(
        w_packed.data_ptr<uint8_t>(),
        w_scale.data_ptr<uint8_t>(),
        expert_ids.data_ptr<int32_t>(),
        num_valid_tiles.data_ptr<int32_t>(),
        out_bf16.data_ptr(),
        static_cast<int>(tile_offset),
        static_cast<int>(chunk_tiles),
        n_rows,
        k_cols,
        stream_handle);
}

void launch_moe_tile_swiglu_quant_bf16(
    torch::Tensor tile_gu,
    torch::Tensor sorted_pair_ids,
    torch::Tensor num_valid_tiles,
    torch::Tensor topk_weights,
    torch::Tensor tile_inter_out,
    int64_t tile_offset,
    int64_t chunk_tiles,
    int64_t block_m,
    double swiglu_limit,
    int64_t do_ue8m0,
    std::uintptr_t stream_handle) {
    if (chunk_tiles <= 0) return;
    TORCH_CHECK(tile_gu.scalar_type() == at::kFloat, "tile_gu must be float32");
    const int inter_dim = static_cast<int>(tile_inter_out.size(-1));
    launch_moe_tile_swiglu_quant_bf16_cuda(
        tile_gu.data_ptr<float>(),
        sorted_pair_ids.data_ptr<int32_t>(),
        num_valid_tiles.data_ptr<int32_t>(),
        topk_weights.data_ptr<float>(),
        tile_inter_out.data_ptr(),
        static_cast<int>(tile_offset),
        static_cast<int>(chunk_tiles),
        static_cast<int>(block_m),
        inter_dim,
        static_cast<float>(swiglu_limit),
        static_cast<int>(do_ue8m0),
        stream_handle);
}

void launch_bmm_bf16_f32_nt(
    torch::Tensor a_bf16,
    torch::Tensor b_bf16,
    torch::Tensor c_f32) {
    TORCH_CHECK(
        a_bf16.dim() == 3 && b_bf16.dim() == 3 && c_f32.dim() == 3 &&
            a_bf16.is_contiguous() && b_bf16.is_contiguous() && c_f32.is_contiguous(),
        "contiguous 3D tensors required for launch_bmm_bf16_f32_nt");
    TORCH_CHECK(
        a_bf16.scalar_type() == at::kBFloat16 &&
            b_bf16.scalar_type() == at::kBFloat16 &&
            c_f32.scalar_type() == at::kFloat,
        "expected BF16 inputs and FP32 output");
    const int batch = static_cast<int>(a_bf16.size(0));
    if (batch <= 0) return;
    const int m = static_cast<int>(a_bf16.size(1));
    const int k = static_cast<int>(a_bf16.size(2));
    const int n = static_cast<int>(b_bf16.size(1));
    TORCH_CHECK(
        b_bf16.size(0) == batch && b_bf16.size(2) == k &&
            c_f32.size(0) == batch && c_f32.size(1) == m && c_f32.size(2) == n,
        "shape mismatch in launch_bmm_bf16_f32_nt");
    launch_bmm_bf16_f32_nt_cuda(
        a_bf16.data_ptr(),
        b_bf16.data_ptr(),
        c_f32.data_ptr<float>(),
        batch,
        m,
        n,
        k);
}

void launch_moe_scatter_tile_outputs_bf16(
    torch::Tensor tile_dn,
    torch::Tensor sorted_pair_ids,
    torch::Tensor num_valid_tiles,
    torch::Tensor pair_out,
    int64_t tile_offset,
    int64_t chunk_tiles,
    int64_t block_m,
    std::uintptr_t stream_handle) {
    if (chunk_tiles <= 0) return;
    const int hidden_dim = static_cast<int>(pair_out.size(1));
    launch_moe_scatter_tile_outputs_bf16_cuda(
        tile_dn.data_ptr(),
        sorted_pair_ids.data_ptr<int32_t>(),
        num_valid_tiles.data_ptr<int32_t>(),
        pair_out.data_ptr(),
        static_cast<int>(tile_offset),
        static_cast<int>(chunk_tiles),
        static_cast<int>(block_m),
        hidden_dim,
        stream_handle);
}

void launch_moe_combine_pairs_shared_bf16(
    torch::Tensor pair_out,
    torch::Tensor shared_out,
    torch::Tensor out,
    int64_t top_k,
    std::uintptr_t stream_handle) {
    const int rows = static_cast<int>(out.size(0));
    if (rows <= 0) return;
    const int hidden_dim = static_cast<int>(out.size(1));
    launch_moe_combine_pairs_shared_bf16_cuda(
        pair_out.data_ptr(),
        shared_out.data_ptr(),
        out.data_ptr(),
        rows,
        static_cast<int>(top_k),
        hidden_dim,
        stream_handle);
}

void launch_engram_gated_residual_bf16(
    torch::Tensor h_stream,
    torch::Tensor kv_proj,
    torch::Tensor qk_prod,
    double norm_eps,
    std::uintptr_t stream_handle) {
    const int rows = static_cast<int>(h_stream.size(0));
    if (rows <= 0) return;
    const int hidden_size = static_cast<int>(h_stream.size(2));
    launch_engram_gated_residual_bf16_cuda(
        h_stream.data_ptr(),
        kv_proj.data_ptr(),
        qk_prod.data_ptr<float>(),
        rows,
        hidden_size,
        static_cast<float>(norm_eps),
        stream_handle);
}

void launch_indexer_paged_score_bf16(
    torch::Tensor q_idx,
    torch::Tensor w_idx,
    torch::Tensor k_cache,
    torch::Tensor mla_bt,
    torch::Tensor row_seq_ids,
    torch::Tensor positions,
    torch::Tensor scores_out,
    double weight_scale,
    int64_t compress_ratio,
    int64_t lat_pp,
    int64_t bt_width,
    int64_t eff_max_c,
    std::uintptr_t stream_handle) {
    const int rows = static_cast<int>(q_idx.size(0));
    if (rows <= 0 || eff_max_c <= 0) return;
    launch_indexer_paged_score_bf16_cuda(
        q_idx.data_ptr(),
        w_idx.data_ptr(),
        k_cache.data_ptr(),
        mla_bt.data_ptr<int32_t>(),
        row_seq_ids.data_ptr<int64_t>(),
        positions.data_ptr<int32_t>(),
        scores_out.data_ptr<float>(),
        rows,
        static_cast<float>(weight_scale),
        static_cast<int>(compress_ratio),
        static_cast<int>(lat_pp),
        static_cast<int>(bt_width),
        static_cast<int>(eff_max_c),
        stream_handle);
}

void launch_indexer_topk512_slots(
    torch::Tensor scores_red,
    torch::Tensor mla_bt,
    torch::Tensor row_seq_ids,
    torch::Tensor positions,
    torch::Tensor shared_full_topk,
    int64_t compress_ratio,
    int64_t lat_pp,
    int64_t bt_width,
    int64_t eff_max_c,
    int64_t mla_arena_base,
    std::uintptr_t stream_handle) {
    const int rows = static_cast<int>(scores_red.size(0));
    if (rows <= 0) return;
    TORCH_CHECK(eff_max_c >= 0 && eff_max_c <= 2048, "eff_max_c must be <= 2048");
    launch_indexer_topk512_slots_cuda(
        scores_red.data_ptr<float>(),
        mla_bt.data_ptr<int32_t>(),
        row_seq_ids.data_ptr<int64_t>(),
        positions.data_ptr<int32_t>(),
        shared_full_topk.data_ptr<int32_t>(),
        rows,
        static_cast<int>(compress_ratio),
        static_cast<int>(lat_pp),
        static_cast<int>(bt_width),
        static_cast<int>(eff_max_c),
        static_cast<int>(mla_arena_base),
        stream_handle);
}
