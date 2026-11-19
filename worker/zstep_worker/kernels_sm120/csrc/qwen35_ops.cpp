
#include <torch/extension.h>
#include <cstdint>
#include <optional>

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
    int64_t active_slots,
    std::uintptr_t stream_handle);

void launch_decode_publish_tokens_cuda(
    int32_t* last_tokens,
    const int32_t* req_slot_ids,
    const int32_t* top2_tokens,
    int64_t num_rows,
    int64_t active_slots,
    std::uintptr_t stream_handle);

void launch_decode_select_tokens(
    torch::Tensor tokens,
    torch::Tensor last_tokens,
    torch::Tensor req_slot_ids,
    torch::Tensor mode,
    int64_t num_rows,
    int64_t active_slots,
    std::uintptr_t stream_handle) {
    TORCH_CHECK(
        tokens.is_cuda() && last_tokens.is_cuda() && req_slot_ids.is_cuda() && mode.is_cuda(),
        "decode_select_tokens tensors must be CUDA");
    TORCH_CHECK(
        tokens.is_contiguous() && last_tokens.is_contiguous()
            && req_slot_ids.is_contiguous() && mode.is_contiguous(),
        "decode_select_tokens tensors must be contiguous");
    TORCH_CHECK(
        tokens.scalar_type() == torch::kInt32 && last_tokens.scalar_type() == torch::kInt32
            && req_slot_ids.scalar_type() == torch::kInt32 && mode.scalar_type() == torch::kInt32,
        "decode_select_tokens tensors must be int32");
    TORCH_CHECK(mode.numel() >= 1, "decode_select_tokens mode tensor must be non-empty");
    TORCH_CHECK(
        num_rows >= 0 && num_rows <= tokens.numel() && num_rows <= req_slot_ids.numel(),
        "num_rows out of range for decode token buffers");
    TORCH_CHECK(
        active_slots >= 0 && active_slots <= last_tokens.numel(),
        "active_slots out of range for last_tokens");
    launch_decode_select_tokens_cuda(
        tokens.data_ptr<int32_t>(),
        last_tokens.data_ptr<int32_t>(),
        req_slot_ids.data_ptr<int32_t>(),
        mode.data_ptr<int32_t>(),
        num_rows,
        active_slots,
        stream_handle);
}

void launch_decode_publish_tokens(
    torch::Tensor last_tokens,
    torch::Tensor req_slot_ids,
    torch::Tensor top2_tokens,
    int64_t num_rows,
    int64_t active_slots,
    std::uintptr_t stream_handle) {
    TORCH_CHECK(
        last_tokens.is_cuda() && req_slot_ids.is_cuda() && top2_tokens.is_cuda(),
        "decode_publish_tokens tensors must be CUDA");
    TORCH_CHECK(
        last_tokens.is_contiguous() && req_slot_ids.is_contiguous() && top2_tokens.is_contiguous(),
        "decode_publish_tokens tensors must be contiguous");
    TORCH_CHECK(
        last_tokens.scalar_type() == torch::kInt32 && req_slot_ids.scalar_type() == torch::kInt32
            && top2_tokens.scalar_type() == torch::kInt32,
        "decode_publish_tokens tensors must be int32");
    TORCH_CHECK(
        num_rows >= 0 && num_rows <= req_slot_ids.numel() && num_rows * 2 <= top2_tokens.numel(),
        "num_rows out of range for decode token buffers");
    TORCH_CHECK(
        active_slots >= 0 && active_slots <= last_tokens.numel(),
        "active_slots out of range for last_tokens");
    launch_decode_publish_tokens_cuda(
        last_tokens.data_ptr<int32_t>(),
        req_slot_ids.data_ptr<int32_t>(),
        top2_tokens.data_ptr<int32_t>(),
        num_rows,
        active_slots,
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
    void* cached_conv_state_ptr,
    float* cached_ssm_state_vk_ptr,
    const int32_t* restore_slot_ids_ptr,
    const int32_t* save_slot_ids_ptr,
    int64_t stride_qkvzba_row,
    int64_t stride_conv_slot,
    int64_t stride_ssm_slot,
    int64_t stride_ssm_head,
    int64_t stride_cached_conv_slot,
    int64_t stride_cached_ssm_slot,
    int64_t stride_cached_ssm_head,
    int num_cached_slots,
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
    std::optional<torch::Tensor> cached_conv_state,
    std::optional<torch::Tensor> cached_ssm_state_vk,
    std::optional<torch::Tensor> restore_slot_ids,
    std::optional<torch::Tensor> save_slot_ids,
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

    void* cached_conv_ptr = nullptr;
    float* cached_ssm_ptr = nullptr;
    const int32_t* restore_ids_ptr = nullptr;
    const int32_t* save_ids_ptr = nullptr;
    int64_t stride_cached_conv_slot = 0;
    int64_t stride_cached_ssm_slot = 0;
    int64_t stride_cached_ssm_head = 0;
    int num_cached_slots = 0;

    const bool has_ckpt = cached_conv_state.has_value();
    TORCH_CHECK(
        has_ckpt == cached_ssm_state_vk.has_value() &&
            has_ckpt == restore_slot_ids.has_value() &&
            has_ckpt == save_slot_ids.has_value(),
        "cached_conv_state, cached_ssm_state_vk, restore_slot_ids, and save_slot_ids must all be provided or all omitted");
    if (has_ckpt) {
        const auto& c_conv = *cached_conv_state;
        const auto& c_ssm = *cached_ssm_state_vk;
        const auto& r_ids = *restore_slot_ids;
        const auto& s_ids = *save_slot_ids;
        TORCH_CHECK(
            c_conv.is_cuda() && c_conv.scalar_type() == torch::kBFloat16 &&
                c_conv.dim() == 3 && c_conv.size(1) == 1024 && c_conv.size(2) == 3 &&
                c_conv.stride(1) == 3 && c_conv.stride(2) == 1,
            "cached_conv_state must be CUDA BF16 [N, 1024, 3]");
        TORCH_CHECK(
            c_ssm.is_cuda() && c_ssm.scalar_type() == torch::kFloat32 &&
                c_ssm.dim() == 4 && c_ssm.size(0) == c_conv.size(0) &&
                c_ssm.size(1) == 4 && c_ssm.size(2) == 128 && c_ssm.size(3) == 128 &&
                c_ssm.stride(2) == 128 && c_ssm.stride(3) == 1 &&
                reinterpret_cast<std::uintptr_t>(c_ssm.data_ptr()) % 16 == 0,
            "cached_ssm_state_vk must be 16B-aligned CUDA FP32 [N, 4, 128, 128]");
        TORCH_CHECK(
            r_ids.is_cuda() && r_ids.scalar_type() == torch::kInt32 &&
                r_ids.is_contiguous() && r_ids.numel() >= num_seqs,
            "restore_slot_ids must be contiguous CUDA int32 [>=num_seqs]");
        TORCH_CHECK(
            s_ids.is_cuda() && s_ids.scalar_type() == torch::kInt32 &&
                s_ids.is_contiguous() && s_ids.numel() >= rows,
            "save_slot_ids must be contiguous CUDA int32 [>=rows]");
        cached_conv_ptr = c_conv.data_ptr();
        cached_ssm_ptr = c_ssm.data_ptr<float>();
        restore_ids_ptr = r_ids.data_ptr<int32_t>();
        save_ids_ptr = s_ids.data_ptr<int32_t>();
        stride_cached_conv_slot = c_conv.stride(0);
        stride_cached_ssm_slot = c_ssm.stride(0);
        stride_cached_ssm_head = c_ssm.stride(1);
        num_cached_slots = static_cast<int>(c_conv.size(0));
    }

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
        cached_conv_ptr,
        cached_ssm_ptr,
        restore_ids_ptr,
        save_ids_ptr,
        qkvzba.stride(0),
        conv_state.stride(0),
        ssm_state_vk.stride(0),
        ssm_state_vk.stride(1),
        stride_cached_conv_slot,
        stride_cached_ssm_slot,
        stride_cached_ssm_head,
        num_cached_slots,
        static_cast<int>(num_seqs),
        static_cast<int>(rows),
        static_cast<float>(eps),
        stream_handle);
}
