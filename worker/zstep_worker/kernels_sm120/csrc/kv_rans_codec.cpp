// Copyright (c) 2026 zstep Contributors
// SPDX-License-Identifier: MIT

#include <torch/extension.h>
#include <cstdint>

int64_t kv_rans_slot_stride_cuda();

void launch_kv_page_split_cuda(
    void* dst_ptr,
    const void* src_ptr,
    int rows_per_page,
    int row_stride,
    int num_pages,
    int decode,
    std::uintptr_t stream_handle);

void launch_kv_rans32_encode_cuda(
    const void* src_ptr,
    int64_t chunk_bytes,
    void* slots_ptr,
    int32_t* comp_lens_ptr,
    uint32_t* crc32_ptr,
    int num_chunks,
    std::uintptr_t stream_handle);

void launch_kv_rans32_decode_cuda(
    const void* slots_ptr,
    const int32_t* comp_lens_ptr,
    const uint32_t* expected_crc32_ptr,
    void* dst_ptr,
    int64_t chunk_bytes,
    int num_chunks,
    int32_t* err_flag_ptr,
    std::uintptr_t stream_handle);

int64_t kv_rans_slot_stride() {
    return kv_rans_slot_stride_cuda();
}

void launch_kv_page_split(
    torch::Tensor dst,
    torch::Tensor src,
    int64_t rows_per_page,
    int64_t row_stride,
    int64_t num_pages,
    int64_t decode,
    int64_t stream_handle) {
    TORCH_CHECK(dst.is_cuda() && src.is_cuda(), "dst and src must be CUDA tensors");
    TORCH_CHECK(dst.is_contiguous() && src.is_contiguous(), "dst and src must be contiguous");
    const int64_t total = rows_per_page * row_stride * num_pages;
    TORCH_CHECK(dst.nbytes() >= static_cast<size_t>(total) && src.nbytes() >= static_cast<size_t>(total),
                "buffer smaller than rows_per_page * row_stride * num_pages");
    launch_kv_page_split_cuda(
        dst.data_ptr(),
        src.data_ptr(),
        static_cast<int>(rows_per_page),
        static_cast<int>(row_stride),
        static_cast<int>(num_pages),
        static_cast<int>(decode),
        static_cast<std::uintptr_t>(stream_handle));
}

void launch_kv_rans32_encode(
    torch::Tensor src,
    int64_t chunk_bytes,
    torch::Tensor slots,
    torch::Tensor comp_lens,
    torch::Tensor crc32_out,
    int64_t num_chunks,
    int64_t stream_handle) {
    TORCH_CHECK(src.is_cuda() && slots.is_cuda() && comp_lens.is_cuda() && crc32_out.is_cuda(),
                "all rANS encode tensors must be CUDA");
    TORCH_CHECK(chunk_bytes > 0 && chunk_bytes <= 74752, "chunk_bytes must be in (0, 74752]");
    TORCH_CHECK(src.nbytes() >= static_cast<size_t>(chunk_bytes * num_chunks), "src too small");
    TORCH_CHECK(slots.nbytes() >= static_cast<size_t>(kv_rans_slot_stride_cuda() * num_chunks),
                "slots buffer too small");
    TORCH_CHECK(comp_lens.numel() >= num_chunks && crc32_out.numel() >= num_chunks,
                "comp_lens/crc32_out too small");
    launch_kv_rans32_encode_cuda(
        src.data_ptr(),
        chunk_bytes,
        slots.data_ptr(),
        comp_lens.data_ptr<int32_t>(),
        reinterpret_cast<uint32_t*>(crc32_out.data_ptr<int32_t>()),
        static_cast<int>(num_chunks),
        static_cast<std::uintptr_t>(stream_handle));
}

void launch_kv_rans32_decode(
    torch::Tensor slots,
    torch::Tensor comp_lens,
    torch::Tensor expected_crc32,
    torch::Tensor dst,
    int64_t chunk_bytes,
    int64_t num_chunks,
    torch::Tensor err_flag,
    int64_t stream_handle) {
    TORCH_CHECK(slots.is_cuda() && comp_lens.is_cuda() && expected_crc32.is_cuda() &&
                dst.is_cuda() && err_flag.is_cuda(),
                "all rANS decode tensors must be CUDA");
    TORCH_CHECK(chunk_bytes > 0 && chunk_bytes <= 74752, "chunk_bytes must be in (0, 74752]");
    TORCH_CHECK(slots.nbytes() >= static_cast<size_t>(kv_rans_slot_stride_cuda() * num_chunks),
                "slots buffer too small");
    TORCH_CHECK(dst.nbytes() >= static_cast<size_t>(chunk_bytes * num_chunks), "dst too small");
    TORCH_CHECK(comp_lens.numel() >= num_chunks && expected_crc32.numel() >= num_chunks &&
                err_flag.numel() >= 1,
                "metadata tensor too small");
    launch_kv_rans32_decode_cuda(
        slots.data_ptr(),
        comp_lens.data_ptr<int32_t>(),
        reinterpret_cast<const uint32_t*>(expected_crc32.data_ptr<int32_t>()),
        dst.data_ptr(),
        chunk_bytes,
        static_cast<int>(num_chunks),
        err_flag.data_ptr<int32_t>(),
        static_cast<std::uintptr_t>(stream_handle));
}
