
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
    int pxb_bfp,
    std::uintptr_t ar_cta_done_ptr,
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
    int pxb_bfp,
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
    int ar_blocks,
    std::uintptr_t ar_cta_done_ptr,
    std::uintptr_t ar_cta_wait_ptr,
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
void launch_ar_mhc_cluster8_cuda(
    const void* x_ptr,
    void* residual_ptr,
    const float* post_mix_ptr,
    const float* comb_mix_ptr,
    const float* pre_mix_ptr,
    const void* norm_weight_ptr,
    void* layer_input_ptr,
    void* aux_out_ptr,
    uint64_t* control_ptr,
    const std::uintptr_t* rank_shm_dev_ptrs,
    std::uintptr_t my_hbm_ptr,
    std::uintptr_t pt_hbm_ptr,
    int64_t slot_stride_bytes,
    int64_t pxb_slot_stride_bytes,
    int rows,
    float norm_eps,
    int write_aux,
    int rank,
    int flag_base,
    std::uintptr_t stream_handle);
void launch_ar_mhc_decode_cuda(
    const void* input_ptr,
    void* residual_ptr,
    const float* post_mix_ptr,
    const float* comb_mix_ptr,
    const float* pre_mix_ptr,
    const void* norm_weight_ptr,
    void* layer_input_ptr,
    void* aux_out_ptr,
    uint64_t* control_ptr,
    const std::uintptr_t* rank_shm_ptrs,
    std::uintptr_t my_hbm_ptr,
    std::uintptr_t pt_hbm_ptr,
    int64_t slot_stride_bytes,
    int64_t pxb_slot_stride_bytes,
    int rows,
    float norm_eps,
    int write_aux,
    int rank,
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
    int64_t pxb_bfp,
    std::uintptr_t ar_cta_done_ptr,
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
    TORCH_CHECK(pxb_bfp >= 0 && pxb_bfp <= 2, "pxb_bfp must be 0, 1, or 2");

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
        static_cast<int>(pxb_bfp),
        ar_cta_done_ptr,
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
        0,
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
    const int64_t eff_blocks =
        (my_hbm_ptr != 0 && blocks >= 32 && blocks <= 56 && rows * vecs_per_row <= 40960) ? 8 : blocks;
    if (rows < eff_blocks) {
        TORCH_CHECK(eff_blocks % rows == 0, "rows < eff_blocks requires eff_blocks % rows == 0");
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
    int64_t pxb_bfp,
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
        static_cast<int>(pxb_bfp),
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
    int64_t ar_blocks,
    std::uintptr_t ar_cta_done_ptr,
    std::uintptr_t ar_cta_wait_ptr,
    std::uintptr_t stream_handle) {
    TORCH_CHECK(x.is_cuda() && residual.is_cuda(), "x/residual must be CUDA tensors");
    TORCH_CHECK(
        x.scalar_type() == at::kBFloat16 && residual.scalar_type() == at::kBFloat16,
        "x/residual must be BF16");
    TORCH_CHECK(x.is_contiguous() && residual.is_contiguous(), "x/residual must be contiguous");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == 5120, "x must have shape [num_tokens, 5120]");
    const int64_t num_tokens = x.size(0);
    TORCH_CHECK(
        residual.dim() == 3 && residual.size(0) == num_tokens &&
        residual.size(1) == 4 && residual.size(2) == 5120,
        "residual must have shape [num_tokens, 4, 5120]");
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(x.data_ptr()) % 16 == 0 &&
        reinterpret_cast<std::uintptr_t>(residual.data_ptr()) % 16 == 0,
        "x/residual pointers must be 16-byte aligned");
    TORCH_CHECK(
        post_mix.is_cuda() && post_mix.scalar_type() == at::kFloat &&
        post_mix.is_contiguous() && post_mix.numel() == num_tokens * 4,
        "post_mix must be contiguous FP32 CUDA tensor of numel num_tokens * 4");
    TORCH_CHECK(
        comb_mix.is_cuda() && comb_mix.scalar_type() == at::kFloat &&
        comb_mix.is_contiguous() && comb_mix.numel() == num_tokens * 16,
        "comb_mix must be contiguous FP32 CUDA tensor of numel num_tokens * 16");
    if (!post_only) {
        TORCH_CHECK(
            pre_mix.is_cuda() && pre_mix.scalar_type() == at::kFloat &&
            pre_mix.is_contiguous() && pre_mix.numel() == num_tokens * 4,
            "pre_mix must be contiguous FP32 CUDA tensor of numel num_tokens * 4");
        TORCH_CHECK(
            norm_weight.is_cuda() && norm_weight.scalar_type() == at::kBFloat16 &&
            norm_weight.is_contiguous() && norm_weight.numel() == 5120 &&
            reinterpret_cast<std::uintptr_t>(norm_weight.data_ptr()) % 16 == 0,
            "norm_weight must be 16B-aligned contiguous BF16 CUDA tensor of numel 5120");
        TORCH_CHECK(
            layer_input.is_cuda() && layer_input.scalar_type() == at::kBFloat16 &&
            layer_input.is_contiguous() && layer_input.dim() == 2 &&
            layer_input.size(0) == num_tokens && layer_input.size(1) == 5120 &&
            reinterpret_cast<std::uintptr_t>(layer_input.data_ptr()) % 16 == 0,
            "layer_input must be 16B-aligned contiguous BF16 CUDA tensor [num_tokens, 5120]");
        if (write_aux) {
            TORCH_CHECK(
                aux_out.is_cuda() && aux_out.scalar_type() == at::kBFloat16 &&
                aux_out.is_contiguous() && aux_out.dim() == 2 &&
                aux_out.size(0) == num_tokens && aux_out.size(1) == 5120 &&
                reinterpret_cast<std::uintptr_t>(aux_out.data_ptr()) % 16 == 0,
                "aux_out must be 16B-aligned contiguous BF16 CUDA tensor [num_tokens, 5120]");
        }
    }
    launch_mhc_post_collapse_rmsnorm_cuda(
        x.data_ptr(),
        residual.data_ptr(),
        post_mix.data_ptr<float>(),
        comb_mix.data_ptr<float>(),
        post_only ? nullptr : pre_mix.data_ptr<float>(),
        post_only ? nullptr : norm_weight.data_ptr(),
        post_only ? nullptr : layer_input.data_ptr(),
        (!post_only && write_aux) ? aux_out.data_ptr() : nullptr,
        static_cast<int>(num_tokens),
        static_cast<float>(norm_eps),
        static_cast<int>(write_aux),
        static_cast<int>(post_only),
        static_cast<int>(ar_blocks),
        ar_cta_done_ptr,
        ar_cta_wait_ptr,
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
    TORCH_CHECK(
        gemm_mul.is_cuda() && gemm_mul.scalar_type() == at::kFloat &&
        gemm_mul.is_contiguous() && gemm_mul.dim() == 3 && gemm_mul.size(2) == 24,
        "gemm_mul must be contiguous FP32 CUDA tensor [n_splits, num_tokens, 24]");
    const int64_t n_splits = gemm_mul.size(0);
    const int64_t num_tokens = gemm_mul.size(1);
    TORCH_CHECK(
        n_splits > 0 && rms_numel > 0 && sinkhorn_repeat >= 1,
        "n_splits, rms_numel, and sinkhorn_repeat must be >= 1");
    TORCH_CHECK(
        gemm_sqr.is_cuda() && gemm_sqr.scalar_type() == at::kFloat &&
        gemm_sqr.is_contiguous() && gemm_sqr.dim() == 2 &&
        gemm_sqr.size(0) == n_splits && gemm_sqr.size(1) == num_tokens,
        "gemm_sqr must be contiguous FP32 CUDA tensor [n_splits, num_tokens]");
    TORCH_CHECK(
        hc_scale.is_cuda() && hc_scale.scalar_type() == at::kFloat &&
        hc_scale.is_contiguous() && hc_scale.numel() >= 3,
        "hc_scale must be contiguous FP32 CUDA tensor of numel >= 3");
    TORCH_CHECK(
        hc_base.is_cuda() && hc_base.scalar_type() == at::kFloat &&
        hc_base.is_contiguous() && hc_base.numel() >= 24,
        "hc_base must be contiguous FP32 CUDA tensor of numel >= 24");
    TORCH_CHECK(
        post_mix.is_cuda() && post_mix.scalar_type() == at::kFloat &&
        post_mix.is_contiguous() && post_mix.numel() == num_tokens * 4,
        "post_mix must be contiguous FP32 CUDA tensor of numel num_tokens * 4");
    TORCH_CHECK(
        comb_mix.is_cuda() && comb_mix.scalar_type() == at::kFloat &&
        comb_mix.is_contiguous() && comb_mix.numel() == num_tokens * 16,
        "comb_mix must be contiguous FP32 CUDA tensor of numel num_tokens * 16");
    if (save_pre_mix) {
        TORCH_CHECK(
            pre_mix_out.is_cuda() && pre_mix_out.scalar_type() == at::kFloat &&
            pre_mix_out.is_contiguous() && pre_mix_out.numel() == num_tokens * 4,
            "pre_mix_out must be contiguous FP32 CUDA tensor of numel num_tokens * 4");
    }
    launch_mhc_sinkhorn_stats_cuda(
        gemm_mul.data_ptr<float>(),
        gemm_sqr.data_ptr<float>(),
        hc_scale.data_ptr<float>(),
        hc_base.data_ptr<float>(),
        post_mix.data_ptr<float>(),
        comb_mix.data_ptr<float>(),
        save_pre_mix ? pre_mix_out.data_ptr<float>() : nullptr,
        static_cast<int>(num_tokens),
        static_cast<int>(n_splits),
        1.0f / static_cast<float>(rms_numel),
        static_cast<float>(rms_eps),
        static_cast<float>(hc_pre_eps),
        static_cast<float>(hc_sinkhorn_eps),
        static_cast<float>(hc_post_mult),
        static_cast<int>(sinkhorn_repeat),
        static_cast<int>(save_pre_mix),
        stream_handle);
}

void launch_ar_mhc_cluster8(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor post_mix,
    torch::Tensor comb_mix,
    torch::Tensor pre_mix,
    torch::Tensor norm_weight,
    torch::Tensor layer_input,
    torch::Tensor aux_out,
    torch::Tensor control,
    torch::Tensor rank_shm_ptrs_u64,
    std::uintptr_t my_hbm_ptr,
    std::uintptr_t pt_hbm_ptr,
    int64_t slot_stride_bytes,
    int64_t pxb_slot_stride_bytes,
    double norm_eps,
    int64_t write_aux,
    int64_t rank,
    int64_t flag_base,
    std::uintptr_t stream_handle) {
    TORCH_CHECK(x.is_cuda() && residual.is_cuda(), "x/residual must be CUDA tensors");
    TORCH_CHECK(
        x.scalar_type() == at::kBFloat16 && residual.scalar_type() == at::kBFloat16,
        "x/residual must be BF16");
    TORCH_CHECK(x.is_contiguous() && residual.is_contiguous(), "x/residual must be contiguous");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == 5120, "x must have shape [rows, 5120]");
    const int64_t rows = x.size(0);
    TORCH_CHECK(rows == 1 || rows == 2 || rows == 4, "cluster8 AR+mHC requires rows in {1, 2, 4}");
    TORCH_CHECK(
        residual.dim() == 3 && residual.size(0) == rows &&
        residual.size(1) == 4 && residual.size(2) == 5120,
        "residual must have shape [rows, 4, 5120]");
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(x.data_ptr()) % 16 == 0 &&
        reinterpret_cast<std::uintptr_t>(residual.data_ptr()) % 16 == 0,
        "x/residual pointers must be 16-byte aligned");
    TORCH_CHECK(
        control.is_cuda() && control.scalar_type() == at::kLong &&
        control.is_contiguous() && flag_base >= 0 && flag_base + 8 <= control.numel(),
        "control must hold at least flag_base + 8 int64 counters");
    TORCH_CHECK(
        rank_shm_ptrs_u64.is_cuda() && rank_shm_ptrs_u64.scalar_type() == at::kLong &&
        rank_shm_ptrs_u64.is_contiguous() && rank_shm_ptrs_u64.numel() >= 8 &&
        rank >= 0 && rank < 8,
        "rank_shm_ptrs_u64 must have 8 int64 device pointers and rank in [0, 7]");
    TORCH_CHECK(x.numel() * 2 <= slot_stride_bytes / 2, "x exceeds SHM slot capacity");
    TORCH_CHECK(
        post_mix.is_cuda() && post_mix.scalar_type() == at::kFloat &&
        post_mix.is_contiguous() && post_mix.numel() == rows * 4,
        "post_mix must be contiguous FP32 CUDA tensor of numel rows * 4");
    TORCH_CHECK(
        comb_mix.is_cuda() && comb_mix.scalar_type() == at::kFloat &&
        comb_mix.is_contiguous() && comb_mix.numel() == rows * 16,
        "comb_mix must be contiguous FP32 CUDA tensor of numel rows * 16");
    TORCH_CHECK(
        pre_mix.is_cuda() && pre_mix.scalar_type() == at::kFloat &&
        pre_mix.is_contiguous() && pre_mix.numel() == rows * 4,
        "pre_mix must be contiguous FP32 CUDA tensor of numel rows * 4");
    TORCH_CHECK(
        norm_weight.is_cuda() && norm_weight.scalar_type() == at::kBFloat16 &&
        norm_weight.is_contiguous() && norm_weight.numel() == 5120 &&
        reinterpret_cast<std::uintptr_t>(norm_weight.data_ptr()) % 16 == 0,
        "norm_weight must be 16B-aligned contiguous BF16 CUDA tensor of numel 5120");
    TORCH_CHECK(
        layer_input.is_cuda() && layer_input.scalar_type() == at::kBFloat16 &&
        layer_input.is_contiguous() && layer_input.dim() == 2 &&
        layer_input.size(0) == rows && layer_input.size(1) == 5120 &&
        reinterpret_cast<std::uintptr_t>(layer_input.data_ptr()) % 16 == 0,
        "layer_input must be 16B-aligned contiguous BF16 CUDA tensor [rows, 5120]");
    if (write_aux) {
        TORCH_CHECK(
            aux_out.is_cuda() && aux_out.scalar_type() == at::kBFloat16 &&
            aux_out.is_contiguous() && aux_out.dim() == 2 &&
            aux_out.size(0) == rows && aux_out.size(1) == 5120 &&
            reinterpret_cast<std::uintptr_t>(aux_out.data_ptr()) % 16 == 0,
            "aux_out must be 16B-aligned contiguous BF16 CUDA tensor [rows, 5120]");
    }
    launch_ar_mhc_cluster8_cuda(
        x.data_ptr(),
        residual.data_ptr(),
        post_mix.data_ptr<float>(),
        comb_mix.data_ptr<float>(),
        pre_mix.data_ptr<float>(),
        norm_weight.data_ptr(),
        layer_input.data_ptr(),
        write_aux ? aux_out.data_ptr() : nullptr,
        reinterpret_cast<uint64_t*>(control.data_ptr<int64_t>()),
        reinterpret_cast<const std::uintptr_t*>(rank_shm_ptrs_u64.data_ptr<int64_t>()),
        my_hbm_ptr,
        pt_hbm_ptr,
        slot_stride_bytes,
        pxb_slot_stride_bytes,
        static_cast<int>(rows),
        static_cast<float>(norm_eps),
        static_cast<int>(write_aux),
        static_cast<int>(rank),
        static_cast<int>(flag_base),
        stream_handle);
}
void launch_ar_mhc_decode(
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor pm,
    torch::Tensor cm,
    torch::Tensor pr,
    torch::Tensor nw,
    torch::Tensor li,
    torch::Tensor ax,
    torch::Tensor ct,
    torch::Tensor rs,
    std::uintptr_t mh,
    std::uintptr_t ph,
    int64_t ss,
    int64_t ps,
    double eps,
    int64_t wa,
    int64_t rk,
    std::uintptr_t st) {
    TORCH_CHECK(
        x.is_cuda() && r.is_cuda() && x.is_contiguous() && r.is_contiguous()
            && x.size(0) >= 5 && x.size(0) <= 128 && x.size(1) == 5120
            && r.size(0) == x.size(0) && mh != 0 && ph != 0 && ct.numel() >= 64
            && rk >= 0 && rk < 8,
        "invalid decode AR+mHC args");
    launch_ar_mhc_decode_cuda(
        x.data_ptr(),
        r.data_ptr(),
        pm.data_ptr<float>(),
        cm.data_ptr<float>(),
        pr.data_ptr<float>(),
        nw.data_ptr(),
        li.data_ptr(),
        wa ? ax.data_ptr() : nullptr,
        reinterpret_cast<uint64_t*>(ct.data_ptr<int64_t>()),
        reinterpret_cast<const std::uintptr_t*>(rs.data_ptr<int64_t>()),
        mh,
        ph,
        ss,
        ps,
        static_cast<int>(x.size(0)),
        static_cast<float>(eps),
        static_cast<int>(wa),
        static_cast<int>(rk),
        st);
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
