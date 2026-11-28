// Copyright (c) 2026 zstep Contributors
// SPDX-License-Identifier: MIT
//
// Portable HGGC / CUDA kernels for T-Head Zhenwu 810E (`ppu001`).
// Uses zero NVIDIA-only PTX inline assembly so HGGC 12.3 compiles cleanly to native AIU code.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#define ZSTEP_SAIL_CUDA_CHECK(expr)                                                 \
    do {                                                                            \
        cudaError_t _err = (expr);                                                  \
        if (_err != cudaSuccess) {                                                  \
            throw std::runtime_error(                                               \
                std::string("SAIL CUDA error: ") + cudaGetErrorString(_err));       \
        }                                                                           \
    } while (0)

namespace {

constexpr int kRingSlots = 4;
constexpr int kMaxBlocks = 64;
constexpr int kMaxRanks = 8;
// 256 KiB arrival flags + 256 KiB completion flags, 64-byte cacheline isolated per flag.
constexpr int64_t kReadyFlagsBase = 0;
constexpr int64_t kDoneFlagsBase = 262144;
constexpr int64_t kDataBaseOffset = 524288;

__device__ __forceinline__ unsigned long long* flag_slot_ptr(
    char* ipc_base,
    int64_t region_base,
    int slot,
    int block_idx,
    int phase,
    int peer) {
    const int64_t idx =
        (((static_cast<int64_t>(slot) * kMaxBlocks + block_idx) * 2 + phase) * kMaxRanks + peer);
    return reinterpret_cast<unsigned long long*>(ipc_base + region_base + idx * 64);
}

__device__ __forceinline__ bool spin_wait_flag(
    unsigned long long* local_flag,
    unsigned long long target,
    const volatile uint64_t* abort_ptr) {
    uint32_t iters = 0;
    while (atomicAdd(local_flag, 0ULL) < target) {
        if ((++iters & 0xFFFU) == 0U && abort_ptr != nullptr && *abort_ptr != 0ULL) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ uint4 ld_volatile_u4(const uint4* ptr) {
    const volatile uint32_t* p = reinterpret_cast<const volatile uint32_t*>(ptr);
    uint4 out;
    out.x = p[0];
    out.y = p[1];
    out.z = p[2];
    out.w = p[3];
    return out;
}

__device__ __forceinline__ void st_volatile_u4(uint4* ptr, const uint4& v) {
    volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(ptr);
    p[0] = v.x;
    p[1] = v.y;
    p[2] = v.z;
    p[3] = v.w;
}

__device__ __forceinline__ void unpack_bf16x8(const uint4& u, float out[8]) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(p[i]);
        out[2 * i + 0] = f.x;
        out[2 * i + 1] = f.y;
    }
}

__device__ __forceinline__ uint4 pack_bf16x8(const float in[8]) {
    uint4 u;
    __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&u);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        p[i] = __floats2bfloat162_rn(in[2 * i + 0], in[2 * i + 1]);
    }
    return u;
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFU, v, mask);
    }
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFU, v, mask));
    }
    return v;
}

__device__ __forceinline__ float ue8m0_to_f32(uint8_t u) {
    return u ? __uint_as_float(static_cast<uint32_t>(u) << 23) : 5.877471754111438e-39f;
}

__device__ __forceinline__ float fast_round_pow2_scale(float amax, float inv_max) {
    const float x = amax * inv_max;
    const uint32_t bits = __float_as_uint(x);
    const int exp_x = static_cast<int>((bits >> 23) & 0xFFU);
    const uint32_t man = bits & 0x7FFFFFU;
    const int ceil_log2 = exp_x - 127 + (man != 0U ? 1 : 0);
    const int biased = max(1, min(254, ceil_log2 + 127));
    return __uint_as_float(static_cast<uint32_t>(biased) << 23);
}

// Exact IEEE-style E4M3fn (1-4-3, bias 7, max 448.0, no inf) round-to-nearest-even -> float32.
__device__ __forceinline__ float round_via_e4m3fn(float x) {
    if (!(x == x)) return 0.0f;
    const uint32_t sign = __float_as_uint(x) & 0x80000000U;
    float ax = fabsf(x);
    if (ax >= 448.0f) {
        return __uint_as_float(sign | __float_as_uint(448.0f));
    }
    if (ax < 9.765625e-4f) {  // < 2^-10 -> subnormal in E4M3fn (step 2^-9 = 0.001953125)
        const float scaled = ax * 512.0f;
        int q = __float2int_rn(scaled);
        q = min(max(q, 0), 8);
        return __uint_as_float(sign | __float_as_uint(static_cast<float>(q) * 0.001953125f));
    }
    const uint32_t bits = __float_as_uint(ax);
    int f_exp = static_cast<int>((bits >> 23) & 0xFFU) - 127;
    f_exp = max(-6, min(8, f_exp));
    const float step = __uint_as_float(static_cast<uint32_t>((f_exp - 3) + 127) << 23);
    float rounded = static_cast<float>(__float2int_rn(ax / step)) * step;
    if (rounded > 448.0f) rounded = 448.0f;
    return __uint_as_float(sign | __float_as_uint(rounded));
}

// Exact E2M1fn round-to-nearest-even -> float32 in [-6.0, 6.0].
__device__ __forceinline__ float round_via_e2m1fn(float x) {
    const float sgn = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 6.0f);
    float mag;
    if (ax <= 0.25f) {
        mag = 0.0f;
    } else if (ax < 0.75f) {
        mag = 0.5f;
    } else if (ax <= 1.25f) {
        mag = 1.0f;
    } else if (ax < 1.75f) {
        mag = 1.5f;
    } else if (ax <= 2.5f) {
        mag = 2.0f;
    } else if (ax < 3.5f) {
        mag = 3.0f;
    } else if (ax <= 5.0f) {
        mag = 4.0f;
    } else {
        mag = 6.0f;
    }
    return mag == 0.0f ? 0.0f : copysignf(mag, sgn);
}

// ---------------------------------------------------------------------------
// 1. ICNLink P2P IPC Collectives (One-Shot + Two-Shot Push, CUDA-Graph-Safe)
// ---------------------------------------------------------------------------

__global__ void icn_allreduce_bf16_kernel(
    const uint4* __restrict__ input,
    uint4* __restrict__ output,
    uint64_t* __restrict__ control,
    const std::uintptr_t* __restrict__ peer_ptrs,
    const volatile uint64_t* __restrict__ abort_ptr,
    int64_t slot_stride_bytes,
    int64_t oneshot_max_vecs,
    int64_t num_vecs,
    int rank,
    int world_size) {
    __shared__ unsigned long long s_step;
    __shared__ int s_abort;
    __shared__ char* s_peers[kMaxRanks];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gdim = gridDim.x;
    if (tid < world_size) {
        s_peers[tid] = reinterpret_cast<char*>(peer_ptrs[tid]);
    }
    if (tid == 0) {
        const unsigned long long next = static_cast<unsigned long long>(control[bid] + 1ULL);
        control[bid] = next;
        s_step = next;
        s_abort = 0;
    }
    __syncthreads();

    const unsigned long long step = s_step;
    const int slot = static_cast<int>((step - 1ULL) & (kRingSlots - 1));
    char* my_base = s_peers[rank];

    // Ring backpressure across > kRingSlots steps
    if (step > kRingSlots) {
        const unsigned long long min_done = step - kRingSlots;
        if (tid < world_size) {
            if (!spin_wait_flag(
                    flag_slot_ptr(my_base, kDoneFlagsBase, slot, bid, 0, tid),
                    min_done,
                    abort_ptr)) {
                s_abort = 1;
            }
        }
        __syncthreads();
        if (s_abort) return;
    }

    const int64_t slot_off = kDataBaseOffset + static_cast<int64_t>(slot) * slot_stride_bytes;
    const int64_t rank_inbox_stride = slot_stride_bytes / (2 * kMaxRanks);

    if (num_vecs <= oneshot_max_vecs) {
        // One-Shot Push: contiguous 128-bit burst per peer (staggered order avoids incast).
        for (int step_p = 0; step_p < world_size; ++step_p) {
            int p = rank + step_p;
            if (p >= world_size) p -= world_size;
            uint4* __restrict__ dst = reinterpret_cast<uint4*>(
                s_peers[p] + slot_off + static_cast<int64_t>(rank) * rank_inbox_stride);
            for (int64_t idx = static_cast<int64_t>(bid) * blockDim.x + tid;
                 idx < num_vecs;
                 idx += static_cast<int64_t>(gdim) * blockDim.x) {
                dst[idx] = input[idx];
            }
        }
        __syncthreads();
        __threadfence_system();
        if (tid < world_size) {
            atomicExch(flag_slot_ptr(s_peers[tid], kReadyFlagsBase, slot, bid, 0, rank), step);
            if (!spin_wait_flag(
                    flag_slot_ptr(my_base, kReadyFlagsBase, slot, bid, 0, tid),
                    step,
                    abort_ptr)) {
                s_abort = 1;
            }
        }
        __syncthreads();
        if (s_abort) return;
        __threadfence_system();

        for (int64_t idx = static_cast<int64_t>(bid) * blockDim.x + tid;
             idx < num_vecs;
             idx += static_cast<int64_t>(gdim) * blockDim.x) {
            float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            for (int src = 0; src < world_size; ++src) {
                const uint4* src_buf = reinterpret_cast<const uint4*>(
                    my_base + slot_off + static_cast<int64_t>(src) * rank_inbox_stride);
                float vals[8];
                unpack_bf16x8(__ldcv(src_buf + idx), vals);
                #pragma unroll
                for (int k = 0; k < 8; ++k) acc[k] += vals[k];
            }
            output[idx] = pack_bf16x8(acc);
        }
    } else {
        // Two-Shot Staggered 128-Bit Burst Reduce-Scatter + AllGather over ICNLink.
        const int64_t chunk_vecs = (num_vecs + world_size - 1) / world_size;
        for (int step_p = 0; step_p < world_size; ++step_p) {
            int p = rank + step_p;
            if (p >= world_size) p -= world_size;
            const int64_t p_start = static_cast<int64_t>(p) * chunk_vecs;
            const int64_t p_end = min(num_vecs, p_start + chunk_vecs);
            uint4* __restrict__ dst = reinterpret_cast<uint4*>(
                s_peers[p] + slot_off + static_cast<int64_t>(rank) * rank_inbox_stride);
            for (int64_t local_i = static_cast<int64_t>(bid) * blockDim.x + tid;
                 p_start + local_i < p_end;
                 local_i += static_cast<int64_t>(gdim) * blockDim.x) {
                dst[local_i] = input[p_start + local_i];
            }
        }
        __syncthreads();
        __threadfence_system();
        if (tid < world_size) {
            atomicExch(flag_slot_ptr(s_peers[tid], kReadyFlagsBase, slot, bid, 0, rank), step);
            if (!spin_wait_flag(
                    flag_slot_ptr(my_base, kReadyFlagsBase, slot, bid, 0, tid),
                    step,
                    abort_ptr)) {
                s_abort = 1;
            }
        }
        __syncthreads();
        if (s_abort) return;
        __threadfence_system();

        // Reduce our slice `rank` in FP32 into `output[my_start .. my_end]`
        const int64_t my_start = static_cast<int64_t>(rank) * chunk_vecs;
        const int64_t my_end = min(num_vecs, my_start + chunk_vecs);
        for (int64_t local_i = static_cast<int64_t>(bid) * blockDim.x + tid;
             my_start + local_i < my_end;
             local_i += static_cast<int64_t>(gdim) * blockDim.x) {
            float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            for (int src = 0; src < world_size; ++src) {
                const uint4* src_buf = reinterpret_cast<const uint4*>(
                    my_base + slot_off + static_cast<int64_t>(src) * rank_inbox_stride);
                float vals[8];
                unpack_bf16x8(__ldcv(src_buf + local_i), vals);
                #pragma unroll
                for (int k = 0; k < 8; ++k) acc[k] += vals[k];
            }
            output[my_start + local_i] = pack_bf16x8(acc);
        }

        // Shot 2: contiguous 128-bit burst write of reduced slice to each peer (staggered order)
        const int64_t ag_base_off = slot_off + slot_stride_bytes / 2;
        for (int step_p = 1; step_p < world_size; ++step_p) {
            int p = rank + step_p;
            if (p >= world_size) p -= world_size;
            uint4* __restrict__ ag_dst = reinterpret_cast<uint4*>(
                s_peers[p] + ag_base_off + static_cast<int64_t>(rank) * rank_inbox_stride);
            for (int64_t local_i = static_cast<int64_t>(bid) * blockDim.x + tid;
                 my_start + local_i < my_end;
                 local_i += static_cast<int64_t>(gdim) * blockDim.x) {
                ag_dst[local_i] = output[my_start + local_i];
            }
        }
        __syncthreads();
        __threadfence_system();
        if (tid < world_size) {
            atomicExch(flag_slot_ptr(s_peers[tid], kReadyFlagsBase, slot, bid, 1, rank), step);
            if (!spin_wait_flag(
                    flag_slot_ptr(my_base, kReadyFlagsBase, slot, bid, 1, tid),
                    step,
                    abort_ptr)) {
                s_abort = 1;
            }
        }
        __syncthreads();
        if (s_abort) return;
        __threadfence_system();

        for (int step_s = 1; step_s < world_size; ++step_s) {
            int src = rank + step_s;
            if (src >= world_size) src -= world_size;
            const int64_t s_start = static_cast<int64_t>(src) * chunk_vecs;
            const int64_t s_end = min(num_vecs, s_start + chunk_vecs);
            const uint4* ag_src = reinterpret_cast<const uint4*>(
                my_base + ag_base_off + static_cast<int64_t>(src) * rank_inbox_stride);
            for (int64_t local_i = static_cast<int64_t>(bid) * blockDim.x + tid;
                 s_start + local_i < s_end;
                 local_i += static_cast<int64_t>(gdim) * blockDim.x) {
                output[s_start + local_i] = __ldcv(ag_src + local_i);
            }
        }
    }

    __syncthreads();
    __threadfence_system();
    if (tid < world_size) {
        atomicExch(flag_slot_ptr(s_peers[tid], kDoneFlagsBase, slot, bid, 0, rank), step);
    }
}

__global__ void icn_allreduce_f32_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    uint64_t* __restrict__ control,
    const std::uintptr_t* __restrict__ peer_ptrs,
    const volatile uint64_t* __restrict__ abort_ptr,
    int64_t slot_stride_bytes,
    int64_t oneshot_max_floats,
    int64_t numel,
    int rank,
    int world_size) {
    __shared__ unsigned long long s_step;
    __shared__ int s_abort;
    __shared__ char* s_peers[kMaxRanks];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gdim = gridDim.x;
    if (tid < world_size) {
        s_peers[tid] = reinterpret_cast<char*>(peer_ptrs[tid]);
    }
    if (tid == 0) {
        const unsigned long long next = static_cast<unsigned long long>(control[bid] + 1ULL);
        control[bid] = next;
        s_step = next;
        s_abort = 0;
    }
    __syncthreads();

    const unsigned long long step = s_step;
    const int slot = static_cast<int>((step - 1ULL) & (kRingSlots - 1));
    char* my_base = s_peers[rank];

    if (step > kRingSlots) {
        const unsigned long long min_done = step - kRingSlots;
        if (tid < world_size) {
            if (!spin_wait_flag(
                    flag_slot_ptr(my_base, kDoneFlagsBase, slot, bid, 0, tid),
                    min_done,
                    abort_ptr)) {
                s_abort = 1;
            }
        }
        __syncthreads();
        if (s_abort) return;
    }

    const int64_t slot_off = kDataBaseOffset + static_cast<int64_t>(slot) * slot_stride_bytes;
    const int64_t rank_inbox_stride = slot_stride_bytes / (2 * kMaxRanks);

    if ((numel & 3LL) == 0LL) {
        const int64_t num_vecs = numel >> 2;
        const int64_t oneshot_max_vecs = oneshot_max_floats >> 2;
        const uint4* __restrict__ in_u4 = reinterpret_cast<const uint4*>(input);
        uint4* __restrict__ out_u4 = reinterpret_cast<uint4*>(output);

        if (num_vecs <= oneshot_max_vecs) {
            for (int step_p = 0; step_p < world_size; ++step_p) {
                int p = rank + step_p;
                if (p >= world_size) p -= world_size;
                uint4* __restrict__ dst = reinterpret_cast<uint4*>(
                    s_peers[p] + slot_off + static_cast<int64_t>(rank) * rank_inbox_stride);
                for (int64_t idx = static_cast<int64_t>(bid) * blockDim.x + tid;
                     idx < num_vecs;
                     idx += static_cast<int64_t>(gdim) * blockDim.x) {
                    dst[idx] = in_u4[idx];
                }
            }
            __syncthreads();
            __threadfence_system();
            if (tid < world_size) {
                atomicExch(flag_slot_ptr(s_peers[tid], kReadyFlagsBase, slot, bid, 0, rank), step);
                if (!spin_wait_flag(
                        flag_slot_ptr(my_base, kReadyFlagsBase, slot, bid, 0, tid),
                        step,
                        abort_ptr)) {
                    s_abort = 1;
                }
            }
            __syncthreads();
            if (s_abort) return;
            __threadfence_system();

            for (int64_t idx = static_cast<int64_t>(bid) * blockDim.x + tid;
                 idx < num_vecs;
                 idx += static_cast<int64_t>(gdim) * blockDim.x) {
                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
                for (int src = 0; src < world_size; ++src) {
                    const uint4* src_buf = reinterpret_cast<const uint4*>(
                        my_base + slot_off + static_cast<int64_t>(src) * rank_inbox_stride);
                    const uint4 v = __ldcv(src_buf + idx);
                    acc0 += __uint_as_float(v.x);
                    acc1 += __uint_as_float(v.y);
                    acc2 += __uint_as_float(v.z);
                    acc3 += __uint_as_float(v.w);
                }
                uint4 packed;
                packed.x = __float_as_uint(acc0);
                packed.y = __float_as_uint(acc1);
                packed.z = __float_as_uint(acc2);
                packed.w = __float_as_uint(acc3);
                out_u4[idx] = packed;
            }
        } else {
            const int64_t chunk_vecs = (num_vecs + world_size - 1) / world_size;
            for (int step_p = 0; step_p < world_size; ++step_p) {
                int p = rank + step_p;
                if (p >= world_size) p -= world_size;
                const int64_t p_start = static_cast<int64_t>(p) * chunk_vecs;
                const int64_t p_end = min(num_vecs, p_start + chunk_vecs);
                uint4* __restrict__ dst = reinterpret_cast<uint4*>(
                    s_peers[p] + slot_off + static_cast<int64_t>(rank) * rank_inbox_stride);
                for (int64_t local_i = static_cast<int64_t>(bid) * blockDim.x + tid;
                     p_start + local_i < p_end;
                     local_i += static_cast<int64_t>(gdim) * blockDim.x) {
                    dst[local_i] = in_u4[p_start + local_i];
                }
            }
            __syncthreads();
            __threadfence_system();
            if (tid < world_size) {
                atomicExch(flag_slot_ptr(s_peers[tid], kReadyFlagsBase, slot, bid, 0, rank), step);
                if (!spin_wait_flag(
                        flag_slot_ptr(my_base, kReadyFlagsBase, slot, bid, 0, tid),
                        step,
                        abort_ptr)) {
                    s_abort = 1;
                }
            }
            __syncthreads();
            if (s_abort) return;
            __threadfence_system();

            const int64_t my_start = static_cast<int64_t>(rank) * chunk_vecs;
            const int64_t my_end = min(num_vecs, my_start + chunk_vecs);
            for (int64_t local_i = static_cast<int64_t>(bid) * blockDim.x + tid;
                 my_start + local_i < my_end;
                 local_i += static_cast<int64_t>(gdim) * blockDim.x) {
                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
                for (int src = 0; src < world_size; ++src) {
                    const uint4* src_buf = reinterpret_cast<const uint4*>(
                        my_base + slot_off + static_cast<int64_t>(src) * rank_inbox_stride);
                    const uint4 v = __ldcv(src_buf + local_i);
                    acc0 += __uint_as_float(v.x);
                    acc1 += __uint_as_float(v.y);
                    acc2 += __uint_as_float(v.z);
                    acc3 += __uint_as_float(v.w);
                }
                uint4 packed;
                packed.x = __float_as_uint(acc0);
                packed.y = __float_as_uint(acc1);
                packed.z = __float_as_uint(acc2);
                packed.w = __float_as_uint(acc3);
                out_u4[my_start + local_i] = packed;
            }

            const int64_t ag_base_off = slot_off + slot_stride_bytes / 2;
            for (int step_p = 1; step_p < world_size; ++step_p) {
                int p = rank + step_p;
                if (p >= world_size) p -= world_size;
                uint4* __restrict__ ag_dst = reinterpret_cast<uint4*>(
                    s_peers[p] + ag_base_off + static_cast<int64_t>(rank) * rank_inbox_stride);
                for (int64_t local_i = static_cast<int64_t>(bid) * blockDim.x + tid;
                     my_start + local_i < my_end;
                     local_i += static_cast<int64_t>(gdim) * blockDim.x) {
                    ag_dst[local_i] = out_u4[my_start + local_i];
                }
            }
            __syncthreads();
            __threadfence_system();
            if (tid < world_size) {
                atomicExch(flag_slot_ptr(s_peers[tid], kReadyFlagsBase, slot, bid, 1, rank), step);
                if (!spin_wait_flag(
                        flag_slot_ptr(my_base, kReadyFlagsBase, slot, bid, 1, tid),
                        step,
                        abort_ptr)) {
                    s_abort = 1;
                }
            }
            __syncthreads();
            if (s_abort) return;
            __threadfence_system();

            for (int step_s = 1; step_s < world_size; ++step_s) {
                int src = rank + step_s;
                if (src >= world_size) src -= world_size;
                const int64_t s_start = static_cast<int64_t>(src) * chunk_vecs;
                const int64_t s_end = min(num_vecs, s_start + chunk_vecs);
                const uint4* ag_src = reinterpret_cast<const uint4*>(
                    my_base + ag_base_off + static_cast<int64_t>(src) * rank_inbox_stride);
                for (int64_t local_i = static_cast<int64_t>(bid) * blockDim.x + tid;
                     s_start + local_i < s_end;
                     local_i += static_cast<int64_t>(gdim) * blockDim.x) {
                    out_u4[s_start + local_i] = __ldcv(ag_src + local_i);
                }
            }
        }
    } else if (numel <= oneshot_max_floats) {
        for (int64_t idx = static_cast<int64_t>(bid) * blockDim.x + tid;
             idx < numel;
             idx += static_cast<int64_t>(gdim) * blockDim.x) {
            const float v = input[idx];
            for (int p = 0; p < world_size; ++p) {
                volatile float* dst = reinterpret_cast<volatile float*>(
                    s_peers[p] + slot_off + static_cast<int64_t>(rank) * rank_inbox_stride);
                dst[idx] = v;
            }
        }
        __syncthreads();
        __threadfence_system();
        if (tid < world_size) {
            atomicExch(flag_slot_ptr(s_peers[tid], kReadyFlagsBase, slot, bid, 0, rank), step);
            if (!spin_wait_flag(
                    flag_slot_ptr(my_base, kReadyFlagsBase, slot, bid, 0, tid),
                    step,
                    abort_ptr)) {
                s_abort = 1;
            }
        }
        __syncthreads();
        if (s_abort) return;

        for (int64_t idx = static_cast<int64_t>(bid) * blockDim.x + tid;
             idx < numel;
             idx += static_cast<int64_t>(gdim) * blockDim.x) {
            float acc = 0.0f;
            for (int src = 0; src < world_size; ++src) {
                const volatile float* src_buf = reinterpret_cast<const volatile float*>(
                    my_base + slot_off + static_cast<int64_t>(src) * rank_inbox_stride);
                acc += src_buf[idx];
            }
            output[idx] = acc;
        }
    } else {
        const int64_t chunk = (numel + world_size - 1) / world_size;
        for (int p = 0; p < world_size; ++p) {
            const int64_t p_start = static_cast<int64_t>(p) * chunk;
            const int64_t p_end = min(numel, p_start + chunk);
            volatile float* dst = reinterpret_cast<volatile float*>(
                s_peers[p] + slot_off + static_cast<int64_t>(rank) * rank_inbox_stride);
            for (int64_t local_i = static_cast<int64_t>(bid) * blockDim.x + tid;
                 p_start + local_i < p_end;
                 local_i += static_cast<int64_t>(gdim) * blockDim.x) {
                dst[local_i] = input[p_start + local_i];
            }
        }
        __syncthreads();
        __threadfence_system();
        if (tid < world_size) {
            atomicExch(flag_slot_ptr(s_peers[tid], kReadyFlagsBase, slot, bid, 0, rank), step);
            if (!spin_wait_flag(
                    flag_slot_ptr(my_base, kReadyFlagsBase, slot, bid, 0, tid),
                    step,
                    abort_ptr)) {
                s_abort = 1;
            }
        }
        __syncthreads();
        if (s_abort) return;

        const int64_t my_start = static_cast<int64_t>(rank) * chunk;
        const int64_t my_end = min(numel, my_start + chunk);
        const int64_t ag_base_off = slot_off + slot_stride_bytes / 2;
        for (int64_t local_i = static_cast<int64_t>(bid) * blockDim.x + tid;
             my_start + local_i < my_end;
             local_i += static_cast<int64_t>(gdim) * blockDim.x) {
            float acc = 0.0f;
            for (int src = 0; src < world_size; ++src) {
                const volatile float* src_buf = reinterpret_cast<const volatile float*>(
                    my_base + slot_off + static_cast<int64_t>(src) * rank_inbox_stride);
                acc += src_buf[local_i];
            }
            output[my_start + local_i] = acc;
            for (int p = 0; p < world_size; ++p) {
                if (p != rank) {
                    volatile float* ag_dst = reinterpret_cast<volatile float*>(
                        s_peers[p] + ag_base_off + static_cast<int64_t>(rank) * rank_inbox_stride);
                    ag_dst[local_i] = acc;
                }
            }
        }
        __syncthreads();
        __threadfence_system();
        if (tid < world_size) {
            atomicExch(flag_slot_ptr(s_peers[tid], kReadyFlagsBase, slot, bid, 1, rank), step);
            if (!spin_wait_flag(
                    flag_slot_ptr(my_base, kReadyFlagsBase, slot, bid, 1, tid),
                    step,
                    abort_ptr)) {
                s_abort = 1;
            }
        }
        __syncthreads();
        if (s_abort) return;

        for (int src = 0; src < world_size; ++src) {
            if (src == rank) continue;
            const int64_t s_start = static_cast<int64_t>(src) * chunk;
            const int64_t s_end = min(numel, s_start + chunk);
            const volatile float* ag_src = reinterpret_cast<const volatile float*>(
                my_base + ag_base_off + static_cast<int64_t>(src) * rank_inbox_stride);
            for (int64_t local_i = static_cast<int64_t>(bid) * blockDim.x + tid;
                 s_start + local_i < s_end;
                 local_i += static_cast<int64_t>(gdim) * blockDim.x) {
                output[s_start + local_i] = ag_src[local_i];
            }
        }
    }

    __syncthreads();
    __threadfence_system();
    if (tid < world_size) {
        atomicExch(flag_slot_ptr(s_peers[tid], kDoneFlagsBase, slot, bid, 0, rank), step);
    }
}

__global__ void icn_allgather_bf16_kernel(
    const uint4* __restrict__ input,
    uint4* __restrict__ output,
    uint64_t* __restrict__ control,
    const std::uintptr_t* __restrict__ peer_ptrs,
    const volatile uint64_t* __restrict__ abort_ptr,
    int64_t slot_stride_bytes,
    int64_t outer_rows,
    int64_t vecs_per_row,
    int rank,
    int world_size) {
    __shared__ unsigned long long s_step;
    __shared__ int s_abort;
    __shared__ char* s_peers[kMaxRanks];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gdim = gridDim.x;
    if (tid < world_size) {
        s_peers[tid] = reinterpret_cast<char*>(peer_ptrs[tid]);
    }
    if (tid == 0) {
        const unsigned long long next = static_cast<unsigned long long>(control[bid] + 1ULL);
        control[bid] = next;
        s_step = next;
        s_abort = 0;
    }
    __syncthreads();

    const unsigned long long step = s_step;
    const int slot = static_cast<int>((step - 1ULL) & (kRingSlots - 1));
    char* my_base = s_peers[rank];

    if (step > kRingSlots) {
        const unsigned long long min_done = step - kRingSlots;
        if (tid < world_size) {
            if (!spin_wait_flag(
                    flag_slot_ptr(my_base, kDoneFlagsBase, slot, bid, 0, tid),
                    min_done,
                    abort_ptr)) {
                s_abort = 1;
            }
        }
        __syncthreads();
        if (s_abort) return;
    }

    const int64_t slot_off = kDataBaseOffset + static_cast<int64_t>(slot) * slot_stride_bytes;
    const int64_t rank_inbox_stride = slot_stride_bytes / kMaxRanks;
    const int64_t total_in_vecs = outer_rows * vecs_per_row;

    for (int64_t idx = static_cast<int64_t>(bid) * blockDim.x + tid;
         idx < total_in_vecs;
         idx += static_cast<int64_t>(gdim) * blockDim.x) {
        const uint4 v = input[idx];
        const int64_t r = idx / vecs_per_row;
        const int64_t c = idx - r * vecs_per_row;
        output[(r * world_size + rank) * vecs_per_row + c] = v;
        for (int p = 0; p < world_size; ++p) {
            if (p != rank) {
                uint4* dst = reinterpret_cast<uint4*>(
                    s_peers[p] + slot_off + static_cast<int64_t>(rank) * rank_inbox_stride);
                st_volatile_u4(dst + idx, v);
            }
        }
    }
    __syncthreads();
    __threadfence_system();
    if (tid < world_size) {
        atomicExch(flag_slot_ptr(s_peers[tid], kReadyFlagsBase, slot, bid, 0, rank), step);
        if (!spin_wait_flag(
                flag_slot_ptr(my_base, kReadyFlagsBase, slot, bid, 0, tid),
                step,
                abort_ptr)) {
            s_abort = 1;
        }
    }
    __syncthreads();
    if (s_abort) return;

    for (int src = 0; src < world_size; ++src) {
        if (src == rank) continue;
        const uint4* src_buf = reinterpret_cast<const uint4*>(
            my_base + slot_off + static_cast<int64_t>(src) * rank_inbox_stride);
        for (int64_t idx = static_cast<int64_t>(bid) * blockDim.x + tid;
             idx < total_in_vecs;
             idx += static_cast<int64_t>(gdim) * blockDim.x) {
            const int64_t r = idx / vecs_per_row;
            const int64_t c = idx - r * vecs_per_row;
            output[(r * world_size + src) * vecs_per_row + c] = ld_volatile_u4(src_buf + idx);
        }
    }

    __syncthreads();
    __threadfence_system();
    if (tid < world_size) {
        atomicExch(flag_slot_ptr(s_peers[tid], kDoneFlagsBase, slot, bid, 0, rank), step);
    }
}

// ---------------------------------------------------------------------------
// 2. Deterministic Local Top-2 + ICNLink P2P Top-2 Gather + Device Token Ops
// ---------------------------------------------------------------------------

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
    if (idx < 0) return;
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

__global__ void icn_tp8_top2_gather_kernel(
    const float* __restrict__ local_top2_vals,
    const int32_t* __restrict__ local_top2_ids,
    int32_t* __restrict__ out_top2_tokens,
    uint32_t* __restrict__ out_top2_logit_bits,
    uint64_t* __restrict__ step_counter,
    const std::uintptr_t* __restrict__ peer_ptrs,
    const volatile uint64_t* __restrict__ abort_ptr,
    int64_t ipc_base_offset,
    int64_t num_rows,
    int64_t max_rows_per_step,
    int rank,
    int world_size,
    int vocab_per_rank) {
    // Dedicated Top-2 region at `peer_ptrs[p] + ipc_base_offset`:
    // [0..4096): ready flags (4 slots * 8 peers * 64B)
    // [4096..8192): done flags (4 slots * 8 peers * 64B)
    // [8192..): 4 slots * 8 ranks * (max_rows_per_step * 2) * sizeof(ValIdxPair)
    __shared__ unsigned long long s_step;
    __shared__ int s_abort;
    __shared__ char* s_peers[kMaxRanks];

    const int tid = threadIdx.x;
    if (tid < world_size) {
        s_peers[tid] = reinterpret_cast<char*>(peer_ptrs[tid]) + ipc_base_offset;
    }
    if (tid == 0) {
        const unsigned long long next = static_cast<unsigned long long>(step_counter[0] + 1ULL);
        step_counter[0] = next;
        s_step = next;
        s_abort = 0;
    }
    __syncthreads();

    const unsigned long long step = s_step;
    const int slot = static_cast<int>((step - 1ULL) & (kRingSlots - 1));
    char* my_top2_base = s_peers[rank];

    auto top2_ready_ptr = [](char* base, int sl, int pr) {
        return reinterpret_cast<unsigned long long*>(base + (sl * kMaxRanks + pr) * 64);
    };
    auto top2_done_ptr = [](char* base, int sl, int pr) {
        return reinterpret_cast<unsigned long long*>(base + 4096 + (sl * kMaxRanks + pr) * 64);
    };

    if (step > kRingSlots) {
        const unsigned long long min_done = step - kRingSlots;
        if (tid < world_size) {
            if (!spin_wait_flag(top2_done_ptr(my_top2_base, slot, tid), min_done, abort_ptr)) {
                s_abort = 1;
            }
        }
        __syncthreads();
        if (s_abort) return;
    }

    const int64_t pairs_per_rank = max_rows_per_step * 2;
    const int64_t slot_pair_off = static_cast<int64_t>(slot) * (kMaxRanks * pairs_per_rank);
    const int base_tok = rank * vocab_per_rank;

    for (int64_t r = tid; r < num_rows; r += blockDim.x) {
        const float v0 = local_top2_vals[r * 2 + 0];
        const int32_t t0 = base_tok + local_top2_ids[r * 2 + 0];
        const float v1 = local_top2_vals[r * 2 + 1];
        const int32_t t1 = base_tok + local_top2_ids[r * 2 + 1];
        for (int p = 0; p < world_size; ++p) {
            volatile ValIdxPair* dst =
                reinterpret_cast<volatile ValIdxPair*>(s_peers[p] + 8192) + slot_pair_off
                + static_cast<int64_t>(rank) * pairs_per_rank + r * 2;
            dst[0].val = v0;
            dst[0].tok = t0;
            dst[1].val = v1;
            dst[1].tok = t1;
        }
    }
    __syncthreads();
    __threadfence_system();
    if (tid < world_size) {
        atomicExch(top2_ready_ptr(s_peers[tid], slot, rank), step);
        if (!spin_wait_flag(top2_ready_ptr(my_top2_base, slot, tid), step, abort_ptr)) {
            s_abort = 1;
        }
    }
    __syncthreads();
    if (s_abort) return;

    const volatile ValIdxPair* my_slot_pairs =
        reinterpret_cast<const volatile ValIdxPair*>(my_top2_base + 8192) + slot_pair_off;
    for (int64_t r = tid; r < num_rows; r += blockDim.x) {
        Top2Acc acc{-1.0e38f, -1, -1.0e38f, -1};
        for (int src = 0; src < world_size; ++src) {
            const int64_t base_idx = static_cast<int64_t>(src) * pairs_per_rank + r * 2;
            merge_candidate(acc, my_slot_pairs[base_idx + 0].val, my_slot_pairs[base_idx + 0].tok);
            merge_candidate(acc, my_slot_pairs[base_idx + 1].val, my_slot_pairs[base_idx + 1].tok);
        }
        out_top2_tokens[r * 2 + 0] = acc.i1;
        out_top2_tokens[r * 2 + 1] = acc.i2;
        out_top2_logit_bits[r * 2 + 0] = __float_as_uint(acc.v1);
        out_top2_logit_bits[r * 2 + 1] = __float_as_uint(acc.v2);
    }
    __syncthreads();
    __threadfence_system();
    if (tid < world_size) {
        atomicExch(top2_done_ptr(s_peers[tid], slot, rank), step);
    }
}

__global__ void decode_prepare_tokens_kernel(
    int32_t* __restrict__ dec_tokens,
    int32_t* __restrict__ dec_lookback,
    const int32_t* __restrict__ slot_last_tok,
    const int32_t* __restrict__ slot_lookback,
    const int32_t* __restrict__ dec_req_slots,
    const int32_t* __restrict__ dec_mode,
    int bucket,
    int max_seqs) {
    if (dec_mode[0] == 0) return;
    const int i = threadIdx.x;
    if (i < bucket) {
        const int slot = dec_req_slots[i];
        if (slot >= 0 && slot < max_seqs) {
            dec_tokens[i] = slot_last_tok[slot];
            dec_lookback[i * 3 + 0] = slot_lookback[slot * 3 + 0];
            dec_lookback[i * 3 + 1] = slot_lookback[slot * 3 + 1];
            dec_lookback[i * 3 + 2] = slot_lookback[slot * 3 + 2];
        }
    }
}

__global__ void decode_commit_tokens_kernel(
    int32_t* __restrict__ slot_last_tok,
    int32_t* __restrict__ slot_lookback,
    const int32_t* __restrict__ dec_tokens,
    const int32_t* __restrict__ dec_lookback,
    const int32_t* __restrict__ dec_req_slots,
    const int32_t* __restrict__ sampled_tokens,
    int num_active,
    int max_seqs) {
    const int i = threadIdx.x;
    if (i < num_active) {
        const int slot = dec_req_slots[i];
        if (slot >= 0 && slot < max_seqs) {
            const int32_t cur_tok = dec_tokens[i];
            const int32_t lb1 = dec_lookback[i * 3 + 1];
            const int32_t lb2 = dec_lookback[i * 3 + 2];
            slot_lookback[slot * 3 + 0] = lb1;
            slot_lookback[slot * 3 + 1] = lb2;
            slot_lookback[slot * 3 + 2] = cur_tok;
            slot_last_tok[slot] = sampled_tokens[i * 2 + 0];
        }
    }
}

// ---------------------------------------------------------------------------
// 3. RMSNorm + Hyper-Connections (mHC) + RoPE + Inplace Quant Kernels
// ---------------------------------------------------------------------------

__global__ void rmsnorm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int dim,
    float eps) {
    __shared__ float s_warp[8];
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const __nv_bfloat16* row_x = x + row * dim;
    __nv_bfloat16* row_out = out + row * dim;

    float sq_sum = 0.0f;
    for (int j = tid; j < dim; j += blockDim.x) {
        const float v = __bfloat162float(row_x[j]);
        sq_sum += v * v;
    }
    sq_sum = warp_sum(sq_sum);
    if (lane == 0) s_warp[warp] = sq_sum;
    __syncthreads();
    if (warp == 0) {
        float block_sq = (lane < (blockDim.x >> 5)) ? s_warp[lane] : 0.0f;
        block_sq = warp_sum(block_sq);
        if (lane == 0) s_warp[0] = rsqrtf(block_sq / static_cast<float>(dim) + eps);
    }
    __syncthreads();
    const float inv_rms = s_warp[0];
    for (int j = tid; j < dim; j += blockDim.x) {
        const float v = __bfloat162float(row_x[j]) * inv_rms * __bfloat162float(weight[j]);
        row_out[j] = __float2bfloat16(v);
    }
}

__device__ __forceinline__ void run_sinkhorn_row(
    const float raw_mixes[24],
    const float* hc_scale,
    const float* hc_base,
    float* pre_out,
    float* post_out,
    float* comb_out,
    float hc_eps,
    int sinkhorn_iters) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        pre_out[j] = (1.0f / (1.0f + expf(-(raw_mixes[j] * hc_scale[0] + hc_base[j])))) + hc_eps;
        post_out[j] = 2.0f / (1.0f + expf(-(raw_mixes[j + 4] * hc_scale[1] + hc_base[j + 4])));
    }
    float comb[4][4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float rmax = -1.0e38f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int idx = 8 + j * 4 + k;
            comb[j][k] = raw_mixes[idx] * hc_scale[2] + hc_base[idx];
            rmax = fmaxf(rmax, comb[j][k]);
        }
        float rsum = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            comb[j][k] = expf(comb[j][k] - rmax);
            rsum += comb[j][k];
        }
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            comb[j][k] = comb[j][k] / rsum + hc_eps;
        }
    }
    // Initial column normalization
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float csum = comb[0][k] + comb[1][k] + comb[2][k] + comb[3][k];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            comb[j][k] /= (csum + hc_eps);
        }
    }
    for (int iter = 0; iter < sinkhorn_iters - 1; ++iter) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float rsum = comb[j][0] + comb[j][1] + comb[j][2] + comb[j][3];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                comb[j][k] /= (rsum + hc_eps);
            }
        }
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float csum = comb[0][k] + comb[1][k] + comb[2][k] + comb[3][k];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                comb[j][k] /= (csum + hc_eps);
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            comb_out[j * 4 + k] = comb[j][k];
        }
    }
}

// Fused decode/verify mHC mixes + Sinkhorn (`x: [rows, 20480] BF16`, `hc_fn: [24, 20480] FP32`).
__global__ void mhc_fused_mixes_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ hc_fn,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    float* __restrict__ pre_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    float norm_eps,
    float hc_eps,
    int sinkhorn_iters) {
    constexpr int kHcDim = 20480;
    constexpr int kMixHc = 24;
    __shared__ float s_warp_sq[8];
    __shared__ float s_warp_dot[8][kMixHc];

    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const __nv_bfloat16* row_x = x + row * kHcDim;

    float sq = 0.0f;
    float dots[kMixHc];
    #pragma unroll
    for (int m = 0; m < kMixHc; ++m) dots[m] = 0.0f;

    for (int j = tid; j < kHcDim; j += 256) {
        const float xv = __bfloat162float(row_x[j]);
        sq += xv * xv;
        #pragma unroll
        for (int m = 0; m < kMixHc; ++m) {
            dots[m] += xv * hc_fn[m * kHcDim + j];
        }
    }
    sq = warp_sum(sq);
    if (lane == 0) s_warp_sq[warp] = sq;
    #pragma unroll
    for (int m = 0; m < kMixHc; ++m) {
        const float ms = warp_sum(dots[m]);
        if (lane == 0) s_warp_dot[warp][m] = ms;
    }
    __syncthreads();

    if (tid == 0) {
        float total_sq = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) total_sq += s_warp_sq[w];
        const float inv_rms = rsqrtf(total_sq * (1.0f / 20480.0f) + norm_eps);
        float mixes[kMixHc];
        #pragma unroll
        for (int m = 0; m < kMixHc; ++m) {
            float s = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) s += s_warp_dot[w][m];
            mixes[m] = s * inv_rms;
        }
        run_sinkhorn_row(
            mixes,
            hc_scale,
            hc_base,
            pre_out + row * 4,
            post_out + row * 4,
            comb_out + row * 16,
            hc_eps,
            sinkhorn_iters);
    }
}

// Multi-CU Split-K Stage-1 for mHC (`grid = (rows, split_k)`, `block = 256`).
// Each block reduces `20480 / split_k` columns into `[sq, dots[0..23]]` (`25` FP32 floats).
__global__ void mhc_splitk_stage1_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ hc_fn,
    float* __restrict__ partial_out,
    int split_k) {
    constexpr int kHcDim = 20480;
    constexpr int kMixHc = 24;
    __shared__ float s_warp[8][kMixHc + 1];

    const int64_t row = blockIdx.x;
    const int sk = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int cols_per_split = kHcDim / split_k;
    const int j_start = sk * cols_per_split;
    const int j_end = j_start + cols_per_split;
    const __nv_bfloat16* row_x = x + row * kHcDim;

    float sq = 0.0f;
    float dots[kMixHc];
    #pragma unroll
    for (int m = 0; m < kMixHc; ++m) dots[m] = 0.0f;

    for (int j = j_start + tid; j < j_end; j += 256) {
        const float xv = __bfloat162float(row_x[j]);
        sq += xv * xv;
        #pragma unroll
        for (int m = 0; m < kMixHc; ++m) {
            dots[m] += xv * hc_fn[m * kHcDim + j];
        }
    }

    sq = warp_sum(sq);
    if (lane == 0) s_warp[warp][0] = sq;
    #pragma unroll
    for (int m = 0; m < kMixHc; ++m) {
        const float ms = warp_sum(dots[m]);
        if (lane == 0) s_warp[warp][1 + m] = ms;
    }
    __syncthreads();

    if (tid < (kMixHc + 1)) {
        float sum_v = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            sum_v += s_warp[w][tid];
        }
        partial_out[(row * split_k + sk) * (kMixHc + 1) + tid] = sum_v;
    }
}

__global__ void mhc_splitk_reduce_sinkhorn_kernel(
    const float* __restrict__ partial_out,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    float* __restrict__ pre_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    int split_k,
    float norm_eps,
    float hc_eps,
    int sinkhorn_iters) {
    constexpr int kMixHc = 24;
    __shared__ float s_tot[kMixHc + 1];
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;

    if (tid < (kMixHc + 1)) {
        float acc = 0.0f;
        for (int sk = 0; sk < split_k; ++sk) {
            acc += partial_out[(row * split_k + sk) * (kMixHc + 1) + tid];
        }
        s_tot[tid] = acc;
    }
    __syncthreads();

    if (tid == 0) {
        const float inv_rms = rsqrtf(s_tot[0] * (1.0f / 20480.0f) + norm_eps);
        float mixes[kMixHc];
        #pragma unroll
        for (int m = 0; m < kMixHc; ++m) {
            mixes[m] = s_tot[1 + m] * inv_rms;
        }
        run_sinkhorn_row(
            mixes,
            hc_scale,
            hc_base,
            pre_out + row * 4,
            post_out + row * 4,
            comb_out + row * 16,
            hc_eps,
            sinkhorn_iters);
    }
}

__global__ void mhc_sinkhorn_from_proj_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ raw_proj,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    float* __restrict__ pre_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    float norm_eps,
    float hc_eps,
    int sinkhorn_iters) {
    constexpr int kHcDim = 20480;
    __shared__ float s_warp_sq[8];
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const __nv_bfloat16* row_x = x + row * kHcDim;

    float sq = 0.0f;
    for (int j = tid; j < kHcDim; j += 256) {
        const float xv = __bfloat162float(row_x[j]);
        sq += xv * xv;
    }
    sq = warp_sum(sq);
    if (lane == 0) s_warp_sq[warp] = sq;
    __syncthreads();

    if (tid == 0) {
        float total_sq = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) total_sq += s_warp_sq[w];
        const float inv_rms = rsqrtf(total_sq * (1.0f / 20480.0f) + norm_eps);
        float mixes[24];
        #pragma unroll
        for (int m = 0; m < 24; ++m) {
            mixes[m] = raw_proj[row * 24 + m] * inv_rms;
        }
        run_sinkhorn_row(
            mixes,
            hc_scale,
            hc_base,
            pre_out + row * 4,
            post_out + row * 4,
            comb_out + row * 16,
            hc_eps,
            sinkhorn_iters);
    }
}

// Fuses `hc_pre` (`[rows, 4, dim]` -> BF16 `[rows, dim]`) + optional `aux_out = mean(dim=1)` + `RMSNorm`.
__global__ void mhc_pre_rmsnorm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ pre_mix,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ normed_out,
    __nv_bfloat16* __restrict__ aux_out,
    int dim,
    int write_aux,
    float norm_eps) {
    __shared__ float s_warp[8];
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const float p0 = pre_mix[row * 4 + 0];
    const float p1 = pre_mix[row * 4 + 1];
    const float p2 = pre_mix[row * 4 + 2];
    const float p3 = pre_mix[row * 4 + 3];
    const __nv_bfloat16* rx0 = x + (row * 4 + 0) * dim;
    const __nv_bfloat16* rx1 = x + (row * 4 + 1) * dim;
    const __nv_bfloat16* rx2 = x + (row * 4 + 2) * dim;
    const __nv_bfloat16* rx3 = x + (row * 4 + 3) * dim;

    float sq_sum = 0.0f;
    for (int j = tid; j < dim; j += blockDim.x) {
        const float x0 = __bfloat162float(rx0[j]);
        const float x1 = __bfloat162float(rx1[j]);
        const float x2 = __bfloat162float(rx2[j]);
        const float x3 = __bfloat162float(rx3[j]);
        if (write_aux && aux_out != nullptr) {
            aux_out[row * dim + j] = __float2bfloat16(0.25f * (x0 + x1 + x2 + x3));
        }
        // Match `hc_pre` casting to BF16 before `RMSNorm`
        const float c_bf16 = __bfloat162float(
            __float2bfloat16(p0 * x0 + p1 * x1 + p2 * x2 + p3 * x3));
        sq_sum += c_bf16 * c_bf16;
    }
    sq_sum = warp_sum(sq_sum);
    if (lane == 0) s_warp[warp] = sq_sum;
    __syncthreads();
    if (warp == 0) {
        float bsq = (lane < (blockDim.x >> 5)) ? s_warp[lane] : 0.0f;
        bsq = warp_sum(bsq);
        if (lane == 0) s_warp[0] = rsqrtf(bsq / static_cast<float>(dim) + norm_eps);
    }
    __syncthreads();
    const float inv_rms = s_warp[0];

    __nv_bfloat16* out_row = normed_out + row * dim;
    for (int j = tid; j < dim; j += blockDim.x) {
        const float x0 = __bfloat162float(rx0[j]);
        const float x1 = __bfloat162float(rx1[j]);
        const float x2 = __bfloat162float(rx2[j]);
        const float x3 = __bfloat162float(rx3[j]);
        const float c_bf16 = __bfloat162float(
            __float2bfloat16(p0 * x0 + p1 * x1 + p2 * x2 + p3 * x3));
        out_row[j] = __float2bfloat16(c_bf16 * inv_rms * __bfloat162float(norm_weight[j]));
    }
}

// Fuses `hc_post`: `out[m, c, d] = post[m, c] * sub[m, d] + sum_{i=0..3} comb[m, i, c] * res[m, i, d]`.
__global__ void mhc_post_bf16_kernel(
    const __nv_bfloat16* __restrict__ sub_out,
    const __nv_bfloat16* __restrict__ residual,
    const float* __restrict__ post_mix,
    const float* __restrict__ comb_mix,
    __nv_bfloat16* __restrict__ out,
    int dim) {
    const int64_t row = blockIdx.x;
    const float ps0 = post_mix[row * 4 + 0];
    const float ps1 = post_mix[row * 4 + 1];
    const float ps2 = post_mix[row * 4 + 2];
    const float ps3 = post_mix[row * 4 + 3];
    const float* cb = comb_mix + row * 16;

    const __nv_bfloat16* r0 = residual + (row * 4 + 0) * dim;
    const __nv_bfloat16* r1 = residual + (row * 4 + 1) * dim;
    const __nv_bfloat16* r2 = residual + (row * 4 + 2) * dim;
    const __nv_bfloat16* r3 = residual + (row * 4 + 3) * dim;
    __nv_bfloat16* o0 = out + (row * 4 + 0) * dim;
    __nv_bfloat16* o1 = out + (row * 4 + 1) * dim;
    __nv_bfloat16* o2 = out + (row * 4 + 2) * dim;
    __nv_bfloat16* o3 = out + (row * 4 + 3) * dim;
    const __nv_bfloat16* sub = sub_out + row * dim;

    for (int j = blockIdx.y * blockDim.x + threadIdx.x; j < dim; j += gridDim.y * blockDim.x) {
        const float sv = __bfloat162float(sub[j]);
        const float rv0 = __bfloat162float(r0[j]);
        const float rv1 = __bfloat162float(r1[j]);
        const float rv2 = __bfloat162float(r2[j]);
        const float rv3 = __bfloat162float(r3[j]);
        o0[j] = __float2bfloat16(ps0 * sv + cb[0] * rv0 + cb[4] * rv1 + cb[8] * rv2 + cb[12] * rv3);
        o1[j] = __float2bfloat16(ps1 * sv + cb[1] * rv0 + cb[5] * rv1 + cb[9] * rv2 + cb[13] * rv3);
        o2[j] = __float2bfloat16(ps2 * sv + cb[2] * rv0 + cb[6] * rv1 + cb[10] * rv2 + cb[14] * rv3);
        o3[j] = __float2bfloat16(ps3 * sv + cb[3] * rv0 + cb[7] * rv1 + cb[11] * rv2 + cb[15] * rv3);
    }
}

__global__ void act_quant_ue8m0_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int64_t num_groups) {
    const int64_t g = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    if (g >= num_groups) return;
    const int lane = threadIdx.x & 31;
    const float v = __bfloat162float(src[g * 32 + lane]);
    const float amax = fmaxf(warp_max(fabsf(v)), 1.0e-4f);
    const float s = fast_round_pow2_scale(amax, 1.0f / 448.0f);
    const float q = round_via_e4m3fn(fminf(fmaxf(v / s, -448.0f), 448.0f));
    dst[g * 32 + lane] = __float2bfloat16(q * s);
}

__global__ void shared_swiglu_quant_bf16_kernel(
    const __nv_bfloat16* __restrict__ sh_gu,
    __nv_bfloat16* __restrict__ sh_act_out,
    int64_t total_groups,
    int groups_per_row,
    int inter_dim,
    float swiglu_limit,
    int do_ue8m0) {
    const int64_t g_idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    if (g_idx >= total_groups) return;
    const int lane = threadIdx.x & 31;
    const int64_t row = g_idx / groups_per_row;
    const int col = static_cast<int>((g_idx - row * groups_per_row) * 32 + lane);
    const __nv_bfloat16* row_gu = sh_gu + row * (2 * inter_dim);

    float g = __bfloat162float(row_gu[col]);
    float u = __bfloat162float(row_gu[inter_dim + col]);
    if (swiglu_limit > 0.0f) {
        g = fminf(g, swiglu_limit);
        u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
    }
    const __nv_bfloat16 bf = __float2bfloat16((g / (1.0f + expf(-g))) * u);
    if (!do_ue8m0) {
        sh_act_out[row * inter_dim + col] = bf;
        return;
    }
    const float v = __bfloat162float(bf);
    const float amax = fmaxf(warp_max(fabsf(v)), 1.0e-4f);
    const float s = fast_round_pow2_scale(amax, 1.0f / 448.0f);
    const float q = round_via_e4m3fn(fminf(fmaxf(v / s, -448.0f), 448.0f));
    sh_act_out[row * inter_dim + col] = __float2bfloat16(q * s);
}

__global__ void inplace_fp4_quant_bf16_kernel(
    __nv_bfloat16* __restrict__ x,
    int64_t num_groups,
    int block_size,
    int is_e4m3_scale) {
    const int lane = threadIdx.x & 31;
    if (block_size == 32) {
        const int64_t g = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
        if (g >= num_groups) return;
        __nv_bfloat16* grp = x + g * 32;
        const float v = __bfloat162float(grp[lane]);
        float amax = warp_max(fabsf(v));
        float s;
        if (is_e4m3_scale) {
            amax = fmaxf(amax, 6.0f * 0.001953125f);
            s = round_via_e4m3fn(amax * (1.0f / 6.0f));
        } else {
            amax = fmaxf(amax, 6.0f * 1.1754943508222875e-38f);
            s = fast_round_pow2_scale(amax, 1.0f / 6.0f);
        }
        const float q = round_via_e2m1fn(fminf(fmaxf(v / s, -6.0f), 6.0f));
        grp[lane] = __float2bfloat16(q * s);
    } else {
        // block_size == 16: two groups per 32-lane warp
        const int half = lane >> 4;
        const int sub = lane & 15;
        const int64_t g = ((static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5) * 2 + half;
        if (g >= num_groups) return;
        __nv_bfloat16* grp = x + g * 16;
        const float v = __bfloat162float(grp[sub]);
        float amax = fabsf(v);
        #pragma unroll
        for (int mask = 8; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFU, amax, mask));
        }
        float s;
        if (is_e4m3_scale) {
            amax = fmaxf(amax, 6.0f * 0.001953125f);
            s = round_via_e4m3fn(amax * (1.0f / 6.0f));
        } else {
            amax = fmaxf(amax, 6.0f * 1.1754943508222875e-38f);
            s = fast_round_pow2_scale(amax, 1.0f / 6.0f);
        }
        const float q = round_via_e2m1fn(fminf(fmaxf(v / s, -6.0f), 6.0f));
        grp[sub] = __float2bfloat16(q * s);
    }
}

// `cos_sin` has shape `[max_pos, rope_dim]` storing `[cos_0, sin_0, cos_1, sin_1, ...]`.
__global__ void rope_inplace_bf16_kernel(
    __nv_bfloat16* __restrict__ x,
    const float* __restrict__ cos_sin,
    const int32_t* __restrict__ positions,
    int num_heads,
    int head_dim,
    int rope_dim,
    int inverse) {
    const int64_t row = blockIdx.x;
    const int pos = positions[row];
    if (pos < 0) return;
    const float* cs = cos_sin + static_cast<int64_t>(pos) * rope_dim;
    const int half_rd = rope_dim >> 1;
    const int total_pairs = num_heads * half_rd;
    const int nope_dim = head_dim - rope_dim;

    for (int idx = threadIdx.x; idx < total_pairs; idx += blockDim.x) {
        const int h = idx / half_rd;
        const int p = idx - h * half_rd;
        __nv_bfloat16* pair_ptr = x + (row * num_heads + h) * head_dim + nope_dim + 2 * p;
        const float re = __bfloat162float(pair_ptr[0]);
        const float im = __bfloat162float(pair_ptr[1]);
        const float c = cs[2 * p + 0];
        const float s = inverse ? -cs[2 * p + 1] : cs[2 * p + 1];
        pair_ptr[0] = __float2bfloat16(re * c - im * s);
        pair_ptr[1] = __float2bfloat16(re * s + im * c);
    }
}

// ---------------------------------------------------------------------------
// 4. DeepSeek-V4.1 Router (`sqrtsoftplus` + unbiased gather) & MXFP4 MoE Kernels
// ---------------------------------------------------------------------------

__global__ void dsv41_router_topk_kernel(
    const void* __restrict__ logits,
    const float* __restrict__ bias,
    float* __restrict__ out_weights,
    int32_t* __restrict__ out_indices,
    int num_experts,
    int top_k,
    float route_scale,
    int logits_is_bf16) {
    __shared__ float s_raw[384];
    __shared__ float s_biased[384];
    __shared__ float s_sel_val[8];
    __shared__ int32_t s_sel_idx[8];

    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;

    for (int e = tid; e < num_experts; e += blockDim.x) {
        const float x = logits_is_bf16
            ? __bfloat162float(static_cast<const __nv_bfloat16*>(logits)[row * num_experts + e])
            : static_cast<const float*>(logits)[row * num_experts + e];
        // Numerically stable sqrt(softplus(x)) = sqrt(log(1 + exp(x)))
        const float sp = (x > 20.0f) ? x : log1pf(expf(x));
        const float score = sqrtf(fmaxf(sp, 0.0f));
        s_raw[e] = score;
        s_biased[e] = score + bias[e];
    }
    __syncthreads();

    // Thread-0 / single-warp exact top-k over <= 384 experts (`(val DESC, idx ASC)`).
    if (tid < 32) {
        const int lane = tid;
        for (int k = 0; k < top_k; ++k) {
            float best_v = -1.0e38f;
            int32_t best_i = -1;
            for (int e = lane; e < num_experts; e += 32) {
                const float v = s_biased[e];
                if (best_i < 0 || v > best_v || (v == best_v && e < best_i)) {
                    best_v = v;
                    best_i = e;
                }
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_xor_sync(0xFFFFFFFFU, best_v, off);
                const int32_t oi = __shfl_xor_sync(0xFFFFFFFFU, best_i, off);
                if (oi >= 0 && (best_i < 0 || ov > best_v || (ov == best_v && oi < best_i))) {
                    best_v = ov;
                    best_i = oi;
                }
            }
            if (lane == 0) {
                s_sel_idx[k] = best_i;
                s_sel_val[k] = s_raw[best_i];
                s_biased[best_i] = -1.0e38f;
            }
            __syncwarp();
        }
        if (lane == 0) {
            float w_sum = 0.0f;
            for (int k = 0; k < top_k; ++k) w_sum += s_sel_val[k];
            const float inv = (top_k > 1) ? (route_scale / (w_sum + 1.0e-20f)) : route_scale;
            for (int k = 0; k < top_k; ++k) {
                out_indices[row * top_k + k] = s_sel_idx[k];
                out_weights[row * top_k + k] = s_sel_val[k] * inv;
            }
        }
    }
}

__constant__ float kFp4Lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ float dot32_fp4_u4(const uint4& w_u4, const float* s_x32) {
    const uint32_t words[4] = {w_u4.x, w_u4.y, w_u4.z, w_u4.w};
    float acc = 0.0f;
    #pragma unroll
    for (int wi = 0; wi < 4; ++wi) {
        uint32_t w = words[wi];
        const float* x8 = s_x32 + wi * 8;
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t byte_v = (w >> (b * 8)) & 0xFFU;
            acc += kFp4Lut[byte_v & 0xFU] * x8[2 * b + 0];
            acc += kFp4Lut[byte_v >> 4] * x8[2 * b + 1];
        }
    }
    return acc;
}

// `w13_packed`: `[num_experts, 2 * inter_per_rank, hidden_dim / 2]` uint8
// `w13_scale`:  `[num_experts, 2 * inter_per_rank, hidden_dim / 32]` uint8 (UE8M0)
// Grid: `(rows * top_k, cdiv(inter_per_rank, warps_per_block))`, 1 warp per `(gate, up)` row pair!
__global__ void mxfp4_moe_gemv_gate_up_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w13_packed,
    const uint8_t* __restrict__ w13_scale,
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    __nv_bfloat16* __restrict__ out_inter,
    int top_k,
    int inter_per_rank,
    int hidden_dim,
    float swiglu_limit) {
    extern __shared__ float s_x[];
    const int pair_idx = blockIdx.x;
    const int token_idx = pair_idx / top_k;
    const int expert_id = topk_ids[pair_idx];
    if (expert_id < 0) return;

    const __nv_bfloat16* x_row = x + static_cast<int64_t>(token_idx) * hidden_dim;
    for (int j = threadIdx.x; j < hidden_dim; j += blockDim.x) {
        s_x[j] = __bfloat162float(x_row[j]);
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int out_col = blockIdx.y * warps_per_block + warp;
    if (out_col >= inter_per_rank) return;

    const int num_k_groups = hidden_dim >> 5;  // 5120 / 32 = 160
    const int64_t expert_rows = 2 * static_cast<int64_t>(inter_per_rank);
    const int64_t row_gate = static_cast<int64_t>(expert_id) * expert_rows + out_col;
    const int64_t row_up = row_gate + inter_per_rank;

    const uint4* w_gate_u4 = reinterpret_cast<const uint4*>(
        w13_packed + row_gate * (hidden_dim >> 1));
    const uint4* w_up_u4 = reinterpret_cast<const uint4*>(
        w13_packed + row_up * (hidden_dim >> 1));
    const uint8_t* s_gate = w13_scale + row_gate * num_k_groups;
    const uint8_t* s_up = w13_scale + row_up * num_k_groups;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (int kg = lane; kg < num_k_groups; kg += 32) {
        const float* x32 = s_x + kg * 32;
        gate_acc += dot32_fp4_u4(w_gate_u4[kg], x32) * ue8m0_to_f32(s_gate[kg]);
        up_acc += dot32_fp4_u4(w_up_u4[kg], x32) * ue8m0_to_f32(s_up[kg]);
    }
    gate_acc = warp_sum(gate_acc);
    up_acc = warp_sum(up_acc);

    if (lane == 0) {
        if (swiglu_limit > 0.0f) {
            up_acc = fminf(fmaxf(up_acc, -swiglu_limit), swiglu_limit);
            gate_acc = fminf(gate_acc, swiglu_limit);
        }
        const float silu_g = gate_acc / (1.0f + expf(-gate_acc));
        const float val = (silu_g * up_acc) * topk_weights[pair_idx];
        out_inter[static_cast<int64_t>(pair_idx) * inter_per_rank + out_col] =
            __float2bfloat16(val);
    }
}

// `inter`: `[rows * top_k, inter_per_rank]` BF16 (`inter_per_rank = 288 = 9 * 32`)
// `w2_packed`: `[num_experts, hidden_dim, inter_per_rank / 2]` uint8
// `w2_scale`:  `[num_experts, hidden_dim, inter_per_rank / 32]` uint8
// Each 8-lane subwarp reduces `9` groups of `32` elements across all `top_k` experts for 1 output dim!
__global__ void mxfp4_moe_gemv_down_combine_kernel(
    const __nv_bfloat16* __restrict__ inter,
    const uint8_t* __restrict__ w2_packed,
    const uint8_t* __restrict__ w2_scale,
    const int32_t* __restrict__ topk_ids,
    const __nv_bfloat16* __restrict__ shared_out,
    __nv_bfloat16* __restrict__ out,
    int top_k,
    int hidden_dim,
    int inter_per_rank) {
    extern __shared__ float s_inter[];
    const int token_idx = blockIdx.x;
    const int total_inter = top_k * inter_per_rank;
    const __nv_bfloat16* inter_base = inter + static_cast<int64_t>(token_idx) * total_inter;
    for (int j = threadIdx.x; j < total_inter; j += blockDim.x) {
        s_inter[j] = __bfloat162float(inter_base[j]);
    }
    __syncthreads();

    // 8 threads per output row (`sub_lane = tid & 7`, `sub_id = tid >> 3`)
    const int sub_lane = threadIdx.x & 7;
    const int sub_id = threadIdx.x >> 3;
    const int subs_per_block = blockDim.x >> 3;
    const int out_col = blockIdx.y * subs_per_block + sub_id;
    if (out_col >= hidden_dim) return;

    const int num_k_groups = inter_per_rank >> 5;  // 288 / 32 = 9
    float total_acc = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        const int expert_id = topk_ids[token_idx * top_k + k];
        if (expert_id < 0) continue;
        const int64_t w_row = static_cast<int64_t>(expert_id) * hidden_dim + out_col;
        const uint4* w_u4 = reinterpret_cast<const uint4*>(
            w2_packed + w_row * (inter_per_rank >> 1));
        const uint8_t* s_u8 = w2_scale + w_row * num_k_groups;
        const float* x_k = s_inter + k * inter_per_rank;
        for (int kg = sub_lane; kg < num_k_groups; kg += 8) {
            total_acc += dot32_fp4_u4(w_u4[kg], x_k + kg * 32) * ue8m0_to_f32(s_u8[kg]);
        }
    }
    #pragma unroll
    for (int mask = 4; mask > 0; mask >>= 1) {
        total_acc += __shfl_xor_sync(0xFFFFFFFFU, total_acc, mask);
    }
    if (sub_lane == 0) {
        const int64_t out_idx = static_cast<int64_t>(token_idx) * hidden_dim + out_col;
        const float sh = __bfloat162float(shared_out[out_idx]);
        out[out_idx] = __float2bfloat16(total_acc + sh);
    }
}

__global__ void moe_align_block_size_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_pair_ids,
    int32_t* __restrict__ expert_ids_per_tile,
    int32_t* __restrict__ num_valid_tiles_out,
    int64_t num_pairs,
    int64_t max_padded_slots,
    int num_experts,
    int block_m) {
    __shared__ int s_counts[512];
    __shared__ int s_warp_sums[16];
    __shared__ int s_tile_offsets[513];
    __shared__ int s_cursor[512];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    s_counts[tid] = 0;
    __syncthreads();

    for (int64_t i = tid; i < num_pairs; i += 512) {
        const int e = topk_ids[i];
        if (e >= 0 && e < num_experts) {
            atomicAdd(&s_counts[e], 1);
        }
    }
    __syncthreads();

    const int tiles = (tid < num_experts) ? ((s_counts[tid] + block_m - 1) / block_m) : 0;
    int incl = tiles;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int up = __shfl_up_sync(0xFFFFFFFFU, incl, off);
        if (lane >= off) incl += up;
    }
    if (lane == 31) s_warp_sums[warp] = incl;
    __syncthreads();

    if (warp == 0) {
        int w = (lane < 16) ? s_warp_sums[lane] : 0;
        #pragma unroll
        for (int off = 1; off < 16; off <<= 1) {
            const int up = __shfl_up_sync(0xFFFFFFFFU, w, off);
            if (lane >= off) w += up;
        }
        if (lane < 16) s_warp_sums[lane] = w;
    }
    __syncthreads();

    const int excl = incl - tiles + (warp > 0 ? s_warp_sums[warp - 1] : 0);
    if (tid < num_experts) {
        s_tile_offsets[tid] = excl;
    }
    if (tid == 511) {
        const int total_tiles = excl + tiles;
        s_tile_offsets[num_experts] = total_tiles;
        num_valid_tiles_out[0] = total_tiles;
    }
    __syncthreads();

    const int total_tiles = s_tile_offsets[num_experts];
    const int64_t clear_slots = min(max_padded_slots, static_cast<int64_t>(total_tiles) * block_m);
    for (int64_t idx = tid; idx < clear_slots; idx += 512) {
        sorted_pair_ids[idx] = -1;
    }
    if (tid < num_experts) {
        const int t_lo = s_tile_offsets[tid];
        const int t_hi = s_tile_offsets[tid + 1];
        for (int t = t_lo; t < t_hi; ++t) {
            expert_ids_per_tile[t] = tid;
        }
        s_cursor[tid] = t_lo * block_m;
    }
    __syncthreads();

    for (int64_t i = tid; i < num_pairs; i += 512) {
        const int e = topk_ids[i];
        if (e >= 0 && e < num_experts) {
            const int dst = atomicAdd(&s_cursor[e], 1);
            if (dst < max_padded_slots) {
                sorted_pair_ids[dst] = static_cast<int32_t>(i);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Stateless Engram V2 4-Gram Hash + Sharded FP8+UE8M0 Lookup Kernel
// ---------------------------------------------------------------------------

__device__ __forceinline__ float e4m3fn_byte_to_f32(uint8_t b) {
    const uint32_t sign = static_cast<uint32_t>(b & 0x80U) << 24;
    const uint32_t exp = (b >> 3) & 0xFU;
    const uint32_t mant = b & 0x7U;
    if (exp == 0U) {
        if (mant == 0U) return __uint_as_float(sign);
        const float val = static_cast<float>(mant) * 0.001953125f;  // 2^-9
        return __uint_as_float(sign | __float_as_uint(val));
    }
    if (exp == 15U && mant == 7U) {
        return 0.0f;
    }
    const uint32_t f_bits = sign | ((exp + 120U) << 23) | (mant << 20);
    return __uint_as_float(f_bits);
}

// 1 block per sequence or per token: binary-searches `query_start_loc` for each token `t`,
// forms exact 4-gram lookback from `prev_window[seq, 0..2]` + `token_ids[seq_start..t]`,
// computes 24 hash IDs (`3` n-gram orders * `8` heads), and dequantizes owned rows to BF16!
__global__ void engram_lookup_dequant_kernel(
    const int32_t* __restrict__ token_ids,
    const int32_t* __restrict__ prev_window,
    const int32_t* __restrict__ query_start_loc,
    const int64_t* __restrict__ token_map,
    const int64_t* __restrict__ multipliers,
    const int64_t* __restrict__ primes,
    const int64_t* __restrict__ offsets,
    const uint8_t* __restrict__ weight_fp8,
    const uint8_t* __restrict__ scale_u8,
    __nv_bfloat16* __restrict__ out_bf16,
    int num_seqs,
    int64_t vocab_start_idx,
    int64_t vocab_end_idx,
    int64_t pad_compressed_id) {
    __shared__ int64_t s_hashes[24];
    const int tok_row = blockIdx.x;
    const int tid = threadIdx.x;

    if (tid == 0) {
        // Binary search `seq_idx` such that `qsl[seq_idx] <= tok_row < qsl[seq_idx + 1]`
        int lo = 0;
        int hi = num_seqs - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (query_start_loc[mid] <= tok_row) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        const int seq_idx = lo;
        const int seq_begin = query_start_loc[seq_idx];
        const int rel = tok_row - seq_begin;

        int64_t comp_toks[4];
        bool blocked = false;
        #pragma unroll
        for (int shift = 0; shift < 4; ++shift) {
            int32_t raw_tok;
            if (rel - shift >= 0) {
                raw_tok = token_ids[tok_row - shift];
            } else {
                const int win_idx = 3 + (rel - shift);  // rel - shift in {-3, -2, -1} -> {0, 1, 2}
                raw_tok = (win_idx >= 0) ? prev_window[seq_idx * 3 + win_idx] : -1;
            }
            int64_t src = (raw_tok < 0) ? -1LL : token_map[raw_tok];
            if (src < 0) blocked = true;
            comp_toks[shift] = blocked ? pad_compressed_id : src;
        }

        uint64_t rolling = static_cast<uint64_t>(comp_toks[0] * multipliers[0]);
        #pragma unroll
        for (int ng = 1; ng < 4; ++ng) {
            const uint64_t prod = static_cast<uint64_t>(comp_toks[ng] * multipliers[ng]);
            rolling ^= prod;
            // Python `%` on signed int64 in `engram.py`: note `tokens >= 0` and `multipliers > 0`
            // bounded by `(INT64_MAX // vocab_size) // 2`, so `prod` and `rolling` are strictly >= 0!
            const int64_t r_pos = static_cast<int64_t>(rolling & 0x7FFFFFFFFFFFFFFFULL);
            #pragma unroll
            for (int h = 0; h < 8; ++h) {
                const int col = (ng - 1) * 8 + h;
                const int64_t p = primes[col];
                s_hashes[col] = (r_pos % p) + offsets[col];
            }
        }
    }
    __syncthreads();

    // 24 cols * 256 head_dim = 6144 outputs per token
    __nv_bfloat16* row_out = out_bf16 + static_cast<int64_t>(tok_row) * 6144;
    for (int idx = tid; idx < 6144; idx += blockDim.x) {
        const int col = idx >> 8;     // / 256
        const int d = idx & 255;      // % 256
        const int64_t h_id = s_hashes[col];
        if (h_id >= vocab_start_idx && h_id < vocab_end_idx) {
            const int64_t local_r = h_id - vocab_start_idx;
            const float v = e4m3fn_byte_to_f32(weight_fp8[local_r * 256 + d]);
            const float s = ue8m0_to_f32(scale_u8[local_r * 8 + (d >> 5)]);
            row_out[idx] = __float2bfloat16(v * s);
        } else {
            row_out[idx] = __float2bfloat16(0.0f);
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Native TP=8 Sparse MQA (`h_local = 8, d = 512`, Fused `attn_sink`, Split-K)
// ---------------------------------------------------------------------------

// Stage 1 (or single-pass when `split_k == 1`): 256 threads per block, each thread owns `2` dims
// (`2*tid, 2*tid + 1`) across all `8` heads in registers!
__global__ void sparse_attn_tp8_stage1_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ kv_arena,
    const float* __restrict__ attn_sink,
    const int32_t* __restrict__ topk_idxs,
    float* __restrict__ partial_acc,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    __nv_bfloat16* __restrict__ out,
    int max_topk,
    int split_k,
    float sm_scale) {
    __shared__ float s_warp_dots[8][8];
    __shared__ float s_alpha[8];
    __shared__ float s_weight[8];
    __shared__ float s_m[8];
    __shared__ float s_l[8];
    __shared__ int32_t s_slot;

    const int row = blockIdx.x;
    const int sp = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    float q0[8], q1[8];
    float acc0[8], acc1[8];
    #pragma unroll
    for (int h = 0; h < 8; ++h) {
        const __nv_bfloat162 qv = *reinterpret_cast<const __nv_bfloat162*>(
            q + (static_cast<int64_t>(row) * 8 + h) * 512 + 2 * tid);
        const float2 qf = __bfloat1622float2(qv);
        q0[h] = qf.x;
        q1[h] = qf.y;
        acc0[h] = 0.0f;
        acc1[h] = 0.0f;
    }
    if (tid < 8) {
        s_m[tid] = -1.0e30f;
        s_l[tid] = 0.0f;
    }
    __syncthreads();

    const int chunk = (max_topk + split_k - 1) / split_k;
    const int t_start = sp * chunk;
    const int t_end = min(max_topk, t_start + chunk);
    const int32_t* row_idxs = topk_idxs + static_cast<int64_t>(row) * max_topk;

    for (int t = t_start; t < t_end; ++t) {
        if (tid == 0) {
            s_slot = row_idxs[t];
        }
        __syncthreads();
        const int32_t slot = s_slot;
        if (slot < 0) continue;

        const __nv_bfloat162 kv_v = *reinterpret_cast<const __nv_bfloat162*>(
            kv_arena + static_cast<int64_t>(slot) * 512 + 2 * tid);
        const float2 kvf = __bfloat1622float2(kv_v);

        #pragma unroll
        for (int h = 0; h < 8; ++h) {
            const float ws = warp_sum(q0[h] * kvf.x + q1[h] * kvf.y);
            if (lane == 0) s_warp_dots[warp][h] = ws;
        }
        __syncthreads();

        if (tid < 8) {
            const float score =
                (s_warp_dots[0][tid] + s_warp_dots[1][tid] + s_warp_dots[2][tid] + s_warp_dots[3][tid]
                 + s_warp_dots[4][tid] + s_warp_dots[5][tid] + s_warp_dots[6][tid] + s_warp_dots[7][tid])
                * sm_scale;
            const float m_prev = s_m[tid];
            const float m_new = fmaxf(m_prev, score);
            const float alpha = expf(m_prev - m_new);
            const float w = expf(score - m_new);
            s_m[tid] = m_new;
            s_l[tid] = s_l[tid] * alpha + w;
            s_alpha[tid] = alpha;
            s_weight[tid] = w;
        }
        __syncthreads();

        #pragma unroll
        for (int h = 0; h < 8; ++h) {
            const float a = s_alpha[h];
            const float w = s_weight[h];
            acc0[h] = acc0[h] * a + w * kvf.x;
            acc1[h] = acc1[h] * a + w * kvf.y;
        }
        __syncthreads();
    }

    if (split_k == 1) {
        if (tid < 8) {
            const float denom = s_l[tid] + expf(attn_sink[tid] - s_m[tid]);
            s_alpha[tid] = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int h = 0; h < 8; ++h) {
            const float inv = s_alpha[h];
            *reinterpret_cast<__nv_bfloat162*>(
                out + (static_cast<int64_t>(row) * 8 + h) * 512 + 2 * tid) =
                __floats2bfloat162_rn(acc0[h] * inv, acc1[h] * inv);
        }
    } else {
        const int64_t base_part = (static_cast<int64_t>(row) * split_k + sp) * 8;
        if (tid < 8) {
            partial_m[base_part + tid] = s_m[tid];
            partial_l[base_part + tid] = s_l[tid];
        }
        #pragma unroll
        for (int h = 0; h < 8; ++h) {
            float* p_dst = partial_acc + (base_part + h) * 512 + 2 * tid;
            p_dst[0] = acc0[h];
            p_dst[1] = acc1[h];
        }
    }
}

__global__ void sparse_attn_tp8_reduce_kernel(
    const float* __restrict__ partial_acc,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ attn_sink,
    __nv_bfloat16* __restrict__ out,
    int split_k) {
    __shared__ float s_m_max[8];
    __shared__ float s_inv_denom[8];
    __shared__ float s_scales[8][8];  // [sp][h], split_k <= 8

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (tid < 8) {
        const int h = tid;
        float mmax = -1.0e30f;
        for (int sp = 0; sp < split_k; ++sp) {
            mmax = fmaxf(mmax, partial_m[(static_cast<int64_t>(row) * split_k + sp) * 8 + h]);
        }
        s_m_max[h] = mmax;
        float l_sum = 0.0f;
        for (int sp = 0; sp < split_k; ++sp) {
            const int64_t idx = (static_cast<int64_t>(row) * split_k + sp) * 8 + h;
            const float sc = expf(partial_m[idx] - mmax);
            s_scales[sp][h] = sc;
            l_sum += partial_l[idx] * sc;
        }
        l_sum += expf(attn_sink[h] - mmax);
        s_inv_denom[h] = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    }
    __syncthreads();

    #pragma unroll
    for (int h = 0; h < 8; ++h) {
        float a0 = 0.0f;
        float a1 = 0.0f;
        for (int sp = 0; sp < split_k; ++sp) {
            const float sc = s_scales[sp][h];
            const float* p_src = partial_acc
                + ((static_cast<int64_t>(row) * split_k + sp) * 8 + h) * 512 + 2 * tid;
            a0 += p_src[0] * sc;
            a1 += p_src[1] * sc;
        }
        const float inv = s_inv_denom[h];
        *reinterpret_cast<__nv_bfloat162*>(
            out + (static_cast<int64_t>(row) * 8 + h) * 512 + 2 * tid) =
            __floats2bfloat162_rn(a0 * inv, a1 * inv);
    }
}

// ---------------------------------------------------------------------------
// 7. Fused QKV/O Epilogues, Chunked MXFP4->BF16 Dequant, Fused Engram & Paged Indexer
// ---------------------------------------------------------------------------

// Fuses `mhc_pre` + `RMSNorm` + `UE8M0(block=32)` activation quantization in one pass.
__global__ void mhc_pre_rmsnorm_quant_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ pre_mix,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ normed_out,
    __nv_bfloat16* __restrict__ quant_out,
    int dim,
    int write_normed,
    float norm_eps) {
    __shared__ float s_warp[8];
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const float p0 = pre_mix[row * 4 + 0];
    const float p1 = pre_mix[row * 4 + 1];
    const float p2 = pre_mix[row * 4 + 2];
    const float p3 = pre_mix[row * 4 + 3];
    const __nv_bfloat16* rx0 = x + (row * 4 + 0) * dim;
    const __nv_bfloat16* rx1 = x + (row * 4 + 1) * dim;
    const __nv_bfloat16* rx2 = x + (row * 4 + 2) * dim;
    const __nv_bfloat16* rx3 = x + (row * 4 + 3) * dim;

    float sq_sum = 0.0f;
    for (int j = tid; j < dim; j += 256) {
        const float x0 = __bfloat162float(rx0[j]);
        const float x1 = __bfloat162float(rx1[j]);
        const float x2 = __bfloat162float(rx2[j]);
        const float x3 = __bfloat162float(rx3[j]);
        const float c_bf16 = __bfloat162float(
            __float2bfloat16(p0 * x0 + p1 * x1 + p2 * x2 + p3 * x3));
        sq_sum += c_bf16 * c_bf16;
    }
    sq_sum = warp_sum(sq_sum);
    if (lane == 0) s_warp[warp] = sq_sum;
    __syncthreads();
    if (warp == 0) {
        float bsq = (lane < 8) ? s_warp[lane] : 0.0f;
        bsq = warp_sum(bsq);
        if (lane == 0) s_warp[0] = rsqrtf(bsq / static_cast<float>(dim) + norm_eps);
    }
    __syncthreads();
    const float inv_rms = s_warp[0];

    __nv_bfloat16* n_row = write_normed ? (normed_out + row * dim) : nullptr;
    __nv_bfloat16* q_row = quant_out + row * dim;
    for (int base = 0; base < dim; base += 256) {
        const int j = base + tid;
        if (j < dim) {
            const float x0 = __bfloat162float(rx0[j]);
            const float x1 = __bfloat162float(rx1[j]);
            const float x2 = __bfloat162float(rx2[j]);
            const float x3 = __bfloat162float(rx3[j]);
            const float c_bf16 = __bfloat162float(
                __float2bfloat16(p0 * x0 + p1 * x1 + p2 * x2 + p3 * x3));
            const __nv_bfloat16 n_bf =
                __float2bfloat16(c_bf16 * inv_rms * __bfloat162float(norm_weight[j]));
            if (n_row != nullptr) {
                n_row[j] = n_bf;
            }
            const float v = __bfloat162float(n_bf);
            const float amax = fmaxf(warp_max(fabsf(v)), 1.0e-4f);
            const float s = fast_round_pow2_scale(amax, 1.0f / 448.0f);
            const float q = round_via_e4m3fn(fminf(fmaxf(v / s, -448.0f), 448.0f));
            q_row[j] = __float2bfloat16(q * s);
        }
    }
}

// Fuses `RMSNorm(qr)` + `UE8M0(qr)` + `RMSNorm(wkv)` + `RoPE(wkv)` + `UE8M0(wkv)` + `kv_arena` store!
__global__ void fused_qkv_a_epilogue_bf16_kernel(
    const __nv_bfloat16* __restrict__ qkv_a,
    const __nv_bfloat16* __restrict__ q_norm_w,
    const __nv_bfloat16* __restrict__ kv_norm_w,
    const float* __restrict__ cos_sin,
    const int32_t* __restrict__ positions,
    const int64_t* __restrict__ swa_write_slots,
    __nv_bfloat16* __restrict__ qr_unquant_out,
    __nv_bfloat16* __restrict__ qr_quant_out,
    __nv_bfloat16* __restrict__ kv_arena,
    int q_rank,
    int kv_dim,
    int rope_dim,
    int write_qr_unquant,
    int64_t layer_swa_base,
    float eps) {
    __shared__ float s_warp_q[8];
    __shared__ float s_warp_kv[8];
    __shared__ __nv_bfloat16 s_kv[512];

    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int stride = q_rank + kv_dim;
    const __nv_bfloat16* row_in = qkv_a + row * stride;

    float sq_q = 0.0f;
    for (int j = tid; j < q_rank; j += 256) {
        const float v = __bfloat162float(row_in[j]);
        sq_q += v * v;
    }
    float sq_kv = 0.0f;
    for (int j = tid; j < kv_dim; j += 256) {
        const float v = __bfloat162float(row_in[q_rank + j]);
        sq_kv += v * v;
    }
    sq_q = warp_sum(sq_q);
    sq_kv = warp_sum(sq_kv);
    if (lane == 0) {
        s_warp_q[warp] = sq_q;
        s_warp_kv[warp] = sq_kv;
    }
    __syncthreads();
    if (warp == 0) {
        float bq = (lane < 8) ? s_warp_q[lane] : 0.0f;
        float bkv = (lane < 8) ? s_warp_kv[lane] : 0.0f;
        bq = warp_sum(bq);
        bkv = warp_sum(bkv);
        if (lane == 0) {
            s_warp_q[0] = rsqrtf(bq / static_cast<float>(q_rank) + eps);
            s_warp_kv[0] = rsqrtf(bkv / static_cast<float>(kv_dim) + eps);
        }
    }
    __syncthreads();
    const float inv_q = s_warp_q[0];
    const float inv_kv = s_warp_kv[0];

    __nv_bfloat16* qr_u = write_qr_unquant ? (qr_unquant_out + row * q_rank) : nullptr;
    __nv_bfloat16* qr_q = qr_quant_out + row * q_rank;
    for (int base = 0; base < q_rank; base += 256) {
        const int j = base + tid;
        if (j < q_rank) {
            const __nv_bfloat16 bf = __float2bfloat16(
                __bfloat162float(row_in[j]) * inv_q * __bfloat162float(q_norm_w[j]));
            if (qr_u != nullptr) {
                qr_u[j] = bf;
            }
            const float v = __bfloat162float(bf);
            const float amax = fmaxf(warp_max(fabsf(v)), 1.0e-4f);
            const float s = fast_round_pow2_scale(amax, 1.0f / 448.0f);
            const float q = round_via_e4m3fn(fminf(fmaxf(v / s, -448.0f), 448.0f));
            qr_q[j] = __float2bfloat16(q * s);
        }
    }

    for (int j = tid; j < kv_dim; j += 256) {
        s_kv[j] = __float2bfloat16(
            __bfloat162float(row_in[q_rank + j]) * inv_kv * __bfloat162float(kv_norm_w[j]));
    }
    __syncthreads();

    const int pos = positions[row];
    const int half_rd = rope_dim >> 1;
    const int nope_kv = kv_dim - rope_dim;
    if (pos >= 0 && tid < half_rd) {
        const float* cs = cos_sin + static_cast<int64_t>(pos) * rope_dim;
        const float re = __bfloat162float(s_kv[nope_kv + 2 * tid + 0]);
        const float im = __bfloat162float(s_kv[nope_kv + 2 * tid + 1]);
        const float c = cs[2 * tid + 0];
        const float s = cs[2 * tid + 1];
        s_kv[nope_kv + 2 * tid + 0] = __float2bfloat16(re * c - im * s);
        s_kv[nope_kv + 2 * tid + 1] = __float2bfloat16(re * s + im * c);
    }
    __syncthreads();

    const int64_t dst_slot = layer_swa_base + swa_write_slots[row];
    __nv_bfloat16* kv_dst = kv_arena + dst_slot * kv_dim;
    for (int base = 0; base < kv_dim; base += 256) {
        const int j = base + tid;
        if (j < kv_dim) {
            const float v = __bfloat162float(s_kv[j]);
            const float amax = fmaxf(warp_max(fabsf(v)), 1.0e-4f);
            const float s = fast_round_pow2_scale(amax, 1.0f / 448.0f);
            const float q = round_via_e4m3fn(fminf(fmaxf(v / s, -448.0f), 448.0f));
            kv_dst[j] = __float2bfloat16(q * s);
        }
    }
}

// Splits `q_and_idx` (`[rows, 4608]` = `8*512 + 4*128`), applies RoPE(64) to both,
// and FP4(block=32, UE8M0) quantizes `q_idx` into `q_idx_out`!
__global__ void fused_q_idx_epilogue_bf16_kernel(
    const __nv_bfloat16* __restrict__ q_and_idx,
    const float* __restrict__ cos_sin,
    const int32_t* __restrict__ positions,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ q_idx_out) {
    __shared__ __nv_bfloat16 s_qi[512];
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* in_row = q_and_idx + row * 4608;
    __nv_bfloat16* out_q = q_out + row * 4096;
    const int pos = positions[row];
    const float* cs = (pos >= 0) ? (cos_sin + static_cast<int64_t>(pos) * 64) : nullptr;

    // 1. Copy `q` nope (`8 * 448 = 3584` BF16 elements)
    for (int idx = tid; idx < 3584; idx += 256) {
        const int h = idx / 448;
        const int d = idx - h * 448;
        out_q[h * 512 + d] = in_row[h * 512 + d];
    }
    // 2. Rotate `q` rope (`8 * 32 = 256` pairs -> 1 pair per thread!)
    {
        const int h = tid >> 5;
        const int p = tid & 31;
        const int off = h * 512 + 448 + 2 * p;
        const __nv_bfloat16 b0 = in_row[off + 0];
        const __nv_bfloat16 b1 = in_row[off + 1];
        if (cs != nullptr) {
            const float re = __bfloat162float(b0);
            const float im = __bfloat162float(b1);
            const float c = cs[2 * p + 0];
            const float s = cs[2 * p + 1];
            out_q[off + 0] = __float2bfloat16(re * c - im * s);
            out_q[off + 1] = __float2bfloat16(re * s + im * c);
        } else {
            out_q[off + 0] = b0;
            out_q[off + 1] = b1;
        }
    }

    // 3. Load & rotate `q_idx` (`4 * 128 = 512` BF16 elements at `in_row + 4096`)
    {
        const int h_n = tid >> 6;
        const int d_n = tid & 63;
        s_qi[h_n * 128 + d_n] = in_row[4096 + h_n * 128 + d_n];
    }
    if (tid < 128) {
        const int h_r = tid >> 5;
        const int p_r = tid & 31;
        const int local_off = h_r * 128 + 64 + 2 * p_r;
        const __nv_bfloat16 b0 = in_row[4096 + local_off + 0];
        const __nv_bfloat16 b1 = in_row[4096 + local_off + 1];
        if (cs != nullptr) {
            const float re = __bfloat162float(b0);
            const float im = __bfloat162float(b1);
            const float c = cs[2 * p_r + 0];
            const float s = cs[2 * p_r + 1];
            s_qi[local_off + 0] = __float2bfloat16(re * c - im * s);
            s_qi[local_off + 1] = __float2bfloat16(re * s + im * c);
        } else {
            s_qi[local_off + 0] = b0;
            s_qi[local_off + 1] = b1;
        }
    }
    __syncthreads();

    // 4. FP4(block=32, UE8M0) quantize `s_qi` -> `q_idx_out`
    __nv_bfloat16* out_qi = q_idx_out + row * 512;
    for (int base = 0; base < 512; base += 256) {
        const int j = base + tid;
        const float v = __bfloat162float(s_qi[j]);
        const float amax = fmaxf(warp_max(fabsf(v)), 6.0f * 1.1754943508222875e-38f);
        const float s = fast_round_pow2_scale(amax, 1.0f / 6.0f);
        const float q = round_via_e2m1fn(fminf(fmaxf(v / s, -6.0f), 6.0f));
        out_qi[j] = __float2bfloat16(q * s);
    }
}

// In-place inverse RoPE(64) + UE8M0(block=32) quantization on `attn_o` (`[rows, 8, 512]` BF16).
__global__ void fused_inv_rope_act_quant_bf16_kernel(
    __nv_bfloat16* __restrict__ attn_o,
    const float* __restrict__ cos_sin,
    const int32_t* __restrict__ positions,
    int num_heads,
    int head_dim,
    int rope_dim) {
    __shared__ __nv_bfloat16 s_h[512];
    const int64_t row = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    __nv_bfloat16* h_ptr = attn_o + (row * num_heads + h) * head_dim;

    for (int j = tid; j < head_dim; j += 256) {
        s_h[j] = h_ptr[j];
    }
    __syncthreads();

    const int pos = positions[row];
    const int half_rd = rope_dim >> 1;
    const int nope_dim = head_dim - rope_dim;
    if (pos >= 0 && tid < half_rd) {
        const float* cs = cos_sin + static_cast<int64_t>(pos) * rope_dim;
        const float re = __bfloat162float(s_h[nope_dim + 2 * tid + 0]);
        const float im = __bfloat162float(s_h[nope_dim + 2 * tid + 1]);
        const float c = cs[2 * tid + 0];
        const float s = -cs[2 * tid + 1];
        s_h[nope_dim + 2 * tid + 0] = __float2bfloat16(re * c - im * s);
        s_h[nope_dim + 2 * tid + 1] = __float2bfloat16(re * s + im * c);
    }
    __syncthreads();

    for (int base = 0; base < head_dim; base += 256) {
        const int j = base + tid;
        if (j < head_dim) {
            const float v = __bfloat162float(s_h[j]);
            const float amax = fmaxf(warp_max(fabsf(v)), 1.0e-4f);
            const float s = fast_round_pow2_scale(amax, 1.0f / 448.0f);
            const float q = round_via_e4m3fn(fminf(fmaxf(v / s, -448.0f), 448.0f));
            h_ptr[j] = __float2bfloat16(q * s);
        }
    }
}

// Device-gated tile input gather for `dequant_acblas` prefill MoE (zero `.item()`).
__global__ void moe_gather_tile_inputs_bf16_kernel(
    const uint4* __restrict__ x_u4,
    const int32_t* __restrict__ sorted_pair_ids,
    const int32_t* __restrict__ num_valid_tiles,
    uint4* __restrict__ tile_a_u4,
    int tile_offset,
    int block_m,
    int top_k,
    int hidden_u4) {
    const int local_t = blockIdx.x;
    const int m = blockIdx.y;
    const int tile_id = tile_offset + local_t;
    uint4* dst = tile_a_u4 + (static_cast<int64_t>(local_t) * block_m + m) * hidden_u4;

    int32_t pair = -1;
    if (tile_id < num_valid_tiles[0]) {
        pair = sorted_pair_ids[tile_id * block_m + m];
    }
    if (pair < 0) {
        const uint4 zero_u4{0U, 0U, 0U, 0U};
        for (int j = threadIdx.x; j < hidden_u4; j += blockDim.x) {
            dst[j] = zero_u4;
        }
    } else {
        const int tok = pair / top_k;
        const uint4* src = x_u4 + static_cast<int64_t>(tok) * hidden_u4;
        for (int j = threadIdx.x; j < hidden_u4; j += blockDim.x) {
            dst[j] = src[j];
        }
    }
}

// Vectorized 128-bit MXFP4 (`E2M1` + `UE8M0`) -> BF16 batch dequantization for active tiles.
__global__ void dequant_mxfp4_tiles_bf16_kernel(
    const uint8_t* __restrict__ w_packed,
    const uint8_t* __restrict__ w_scale,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_valid_tiles,
    __nv_bfloat16* __restrict__ out_bf16,
    int tile_offset,
    int chunk_tiles,
    int n_rows,
    int k_cols) {
    const int local_t = blockIdx.y;
    const int tile_id = tile_offset + local_t;
    if (local_t >= chunk_tiles || tile_id >= num_valid_tiles[0]) return;

    const int expert_id = expert_ids[tile_id];
    const int k_groups = k_cols >> 5;
    const int total_groups = n_rows * k_groups;

    const uint4* exp_w_u4 = reinterpret_cast<const uint4*>(
        w_packed + static_cast<int64_t>(expert_id) * n_rows * (k_cols >> 1));
    const uint8_t* exp_s = w_scale + static_cast<int64_t>(expert_id) * total_groups;
    uint4* dst_u4 = reinterpret_cast<uint4*>(
        out_bf16 + static_cast<int64_t>(local_t) * n_rows * k_cols);

    for (int g = blockIdx.x * blockDim.x + threadIdx.x;
         g < total_groups;
         g += gridDim.x * blockDim.x) {
        const uint4 pk = exp_w_u4[g];
        const float sc = ue8m0_to_f32(exp_s[g]);
        const uint32_t words[4] = {pk.x, pk.y, pk.z, pk.w};
        #pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            uint4 out_vec;
            __nv_bfloat162* p2 = reinterpret_cast<__nv_bfloat162*>(&out_vec);
            const uint32_t w = words[wi];
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const uint32_t bv = (w >> (b * 8)) & 0xFFU;
                p2[b] = __floats2bfloat162_rn(
                    kFp4Lut[bv & 0xFU] * sc,
                    kFp4Lut[bv >> 4] * sc);
            }
            dst_u4[g * 4 + wi] = out_vec;
        }
    }
}

// Fuses clamped SwiGLU + routed expert weight + `UE8M0(block=32)` on `[chunk_tiles, block_m, 2*inter]` FP32.
__global__ void moe_tile_swiglu_quant_bf16_kernel(
    const float* __restrict__ tile_gu,
    const int32_t* __restrict__ sorted_pair_ids,
    const int32_t* __restrict__ num_valid_tiles,
    const float* __restrict__ topk_weights,
    __nv_bfloat16* __restrict__ tile_inter_out,
    int tile_offset,
    int block_m,
    int inter_dim,
    float swiglu_limit,
    int do_ue8m0) {
    const int tm = blockIdx.x;
    const int local_t = tm / block_m;
    const int m = tm - local_t * block_m;
    const int tile_id = tile_offset + local_t;
    const int groups_per_row = inter_dim >> 5;
    const int grp = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (grp >= groups_per_row) return;
    const int lane = threadIdx.x & 31;
    const int col = grp * 32 + lane;

    __nv_bfloat16* dst_row = tile_inter_out + static_cast<int64_t>(tm) * inter_dim;
    int32_t pair = -1;
    if (tile_id < num_valid_tiles[0]) {
        pair = sorted_pair_ids[tile_id * block_m + m];
    }
    if (pair < 0) {
        dst_row[col] = __float2bfloat16(0.0f);
        return;
    }

    const float* gu_row = tile_gu + static_cast<int64_t>(tm) * (2 * inter_dim);
    float g = gu_row[col];
    float u = gu_row[inter_dim + col];
    if (swiglu_limit > 0.0f) {
        g = fminf(g, swiglu_limit);
        u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
    }
    const float wt = topk_weights[pair];
    const __nv_bfloat16 bf = __float2bfloat16(((g / (1.0f + expf(-g))) * u) * wt);
    if (!do_ue8m0) {
        dst_row[col] = bf;
        return;
    }
    const float v = __bfloat162float(bf);
    const float amax = fmaxf(warp_max(fabsf(v)), 1.0e-4f);
    const float s = fast_round_pow2_scale(amax, 1.0f / 448.0f);
    const float q = round_via_e4m3fn(fminf(fmaxf(v / s, -448.0f), 448.0f));
    dst_row[col] = __float2bfloat16(q * s);
}

// Scatters valid `[chunk_tiles, block_m, hidden_dim]` down-projection rows into `pair_out`.
__global__ void moe_scatter_tile_outputs_bf16_kernel(
    const uint4* __restrict__ tile_dn_u4,
    const int32_t* __restrict__ sorted_pair_ids,
    const int32_t* __restrict__ num_valid_tiles,
    uint4* __restrict__ pair_out_u4,
    int tile_offset,
    int block_m,
    int hidden_u4) {
    const int local_t = blockIdx.x;
    const int m = blockIdx.y;
    const int tile_id = tile_offset + local_t;
    if (tile_id >= num_valid_tiles[0]) return;
    const int32_t pair = sorted_pair_ids[tile_id * block_m + m];
    if (pair < 0) return;

    const uint4* src = tile_dn_u4 + (static_cast<int64_t>(local_t) * block_m + m) * hidden_u4;
    uint4* dst = pair_out_u4 + static_cast<int64_t>(pair) * hidden_u4;
    for (int j = threadIdx.x; j < hidden_u4; j += blockDim.x) {
        dst[j] = src[j];
    }
}

// Vectorized `uint4` (`8` BF16 elements/thread) combine of `shared_out + sum_{k} pair_out[row*top_k+k]`.
__global__ void moe_combine_pairs_shared_bf16_kernel(
    const uint4* __restrict__ pair_out_u4,
    const uint4* __restrict__ shared_out_u4,
    uint4* __restrict__ out_u4,
    int top_k,
    int hidden_u4) {
    const int64_t row = blockIdx.x;
    const int col_u4 = blockIdx.y * blockDim.x + threadIdx.x;
    if (col_u4 >= hidden_u4) return;

    float acc[8];
    unpack_bf16x8(shared_out_u4[row * hidden_u4 + col_u4], acc);
    for (int k = 0; k < top_k; ++k) {
        float pv[8];
        unpack_bf16x8(pair_out_u4[(row * top_k + k) * hidden_u4 + col_u4], pv);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[i] += pv[i];
        }
    }
    out_u4[row * hidden_u4 + col_u4] = pack_bf16x8(acc);
}

// Fuses 14 PyTorch Engram post-projection RMS/dot/sqrt-gate/residual ops into 1 HGGC pass (`45x` faster).
__global__ void engram_gated_residual_bf16_kernel(
    __nv_bfloat16* __restrict__ h_stream,
    const __nv_bfloat16* __restrict__ kv_proj,
    const float* __restrict__ qk_prod,
    int hidden_size,
    float norm_eps) {
    __shared__ float s_h_sq[8];
    __shared__ float s_k_sq[8];
    __shared__ float s_dot[8];
    __shared__ float s_gate;

    const int64_t row = blockIdx.x;
    const int c = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    __nv_bfloat16* h_row = h_stream + (row * 4 + c) * hidden_size;
    const __nv_bfloat16* k_row = kv_proj + row * (5 * hidden_size) + c * hidden_size;
    const __nv_bfloat16* v_row = kv_proj + row * (5 * hidden_size) + 4 * hidden_size;
    const float* qk_row = qk_prod + c * hidden_size;

    float h_sq = 0.0f, k_sq = 0.0f, dot = 0.0f;
    for (int d = tid; d < hidden_size; d += 256) {
        const float hv = __bfloat162float(h_row[d]);
        const float kv = __bfloat162float(k_row[d]);
        h_sq += hv * hv;
        k_sq += kv * kv;
        dot += hv * qk_row[d] * kv;
    }
    h_sq = warp_sum(h_sq);
    k_sq = warp_sum(k_sq);
    dot = warp_sum(dot);
    if (lane == 0) {
        s_h_sq[warp] = h_sq;
        s_k_sq[warp] = k_sq;
        s_dot[warp] = dot;
    }
    __syncthreads();
    if (tid == 0) {
        float th = 0.0f, tk = 0.0f, td = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            th += s_h_sq[w];
            tk += s_k_sq[w];
            td += s_dot[w];
        }
        const float inv_d = 1.0f / static_cast<float>(hidden_size);
        const float rstd = rsqrtf(th * inv_d + norm_eps) * rsqrtf(tk * inv_d + norm_eps);
        const float sc_dot = td * rstd * sqrtf(inv_d);
        const float signed_sqrt = copysignf(sqrtf(fmaxf(fabsf(sc_dot), 1.0e-6f)), sc_dot);
        s_gate = 1.0f / (1.0f + expf(-signed_sqrt));
    }
    __syncthreads();
    const float g = s_gate;
    for (int d = tid; d < hidden_size; d += 256) {
        h_row[d] = __float2bfloat16(__bfloat162float(h_row[d]) + g * __bfloat162float(v_row[d]));
    }
}

// Paged Indexer scoring directly over `indexer_k_cache` (`0` `index_select`, early-exit past `comp_len`!).
// `q_idx`: `[rows, 4, 128]` BF16, `w_idx`: `[rows, 4]` BF16, `k_cache`: `[num_pages, lat_pp, 128]` BF16.
__global__ void indexer_paged_score_bf16_kernel(
    const __nv_bfloat16* __restrict__ q_idx,
    const __nv_bfloat16* __restrict__ w_idx,
    const __nv_bfloat16* __restrict__ k_cache,
    const int32_t* __restrict__ mla_bt,
    const int64_t* __restrict__ row_seq_ids,
    const int32_t* __restrict__ positions,
    float* __restrict__ scores_out,
    float weight_scale,
    int compress_ratio,
    int lat_pp,
    int bt_width,
    int eff_max_c) {
    __shared__ float s_q[4][128];
    __shared__ float s_w[4];

    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int c_base = blockIdx.y * 8;
    const int c = c_base + warp;
    const int comp_len = (positions[row] + 1) / compress_ratio;

    if (c_base >= comp_len) {
        if (lane == 0 && c < eff_max_c) {
            scores_out[row * eff_max_c + c] = 0.0f;
        }
        return;
    }

    const __nv_bfloat16* q_row = q_idx + row * 512;
    for (int i = tid; i < 512; i += 256) {
        s_q[i >> 7][i & 127] = __bfloat162float(q_row[i]);
    }
    if (tid < 4) {
        s_w[tid] = __bfloat162float(w_idx[row * 4 + tid]) * weight_scale;
    }
    __syncthreads();

    if (c >= eff_max_c) return;
    if (c >= comp_len) {
        if (lane == 0) scores_out[row * eff_max_c + c] = 0.0f;
        return;
    }

    const int64_t seq_id = row_seq_ids[row];
    const int page_col = c / lat_pp;
    const int32_t phys_page =
        (page_col < bt_width) ? mla_bt[seq_id * bt_width + page_col] : 0;
    if (phys_page <= 0) {
        if (lane == 0) scores_out[row * eff_max_c + c] = 0.0f;
        return;
    }

    const int in_page = c - page_col * lat_pp;
    const __nv_bfloat162* k2 = reinterpret_cast<const __nv_bfloat162*>(
        k_cache + (static_cast<int64_t>(phys_page) * lat_pp + in_page) * 128);
    const float2 kv01 = __bfloat1622float2(k2[2 * lane + 0]);
    const float2 kv23 = __bfloat1622float2(k2[2 * lane + 1]);
    const int d0 = 4 * lane;

    float d_h0 = warp_sum(
        s_q[0][d0 + 0] * kv01.x + s_q[0][d0 + 1] * kv01.y
        + s_q[0][d0 + 2] * kv23.x + s_q[0][d0 + 3] * kv23.y);
    float d_h1 = warp_sum(
        s_q[1][d0 + 0] * kv01.x + s_q[1][d0 + 1] * kv01.y
        + s_q[1][d0 + 2] * kv23.x + s_q[1][d0 + 3] * kv23.y);
    float d_h2 = warp_sum(
        s_q[2][d0 + 0] * kv01.x + s_q[2][d0 + 1] * kv01.y
        + s_q[2][d0 + 2] * kv23.x + s_q[2][d0 + 3] * kv23.y);
    float d_h3 = warp_sum(
        s_q[3][d0 + 0] * kv01.x + s_q[3][d0 + 1] * kv01.y
        + s_q[3][d0 + 2] * kv23.x + s_q[3][d0 + 3] * kv23.y);

    if (lane == 0) {
        // Match BF16 BMM output rounding before FP32 ReLU * scaled weight sum
        const float r0 = fmaxf(__bfloat162float(__float2bfloat16(d_h0)), 0.0f);
        const float r1 = fmaxf(__bfloat162float(__float2bfloat16(d_h1)), 0.0f);
        const float r2 = fmaxf(__bfloat162float(__float2bfloat16(d_h2)), 0.0f);
        const float r3 = fmaxf(__bfloat162float(__float2bfloat16(d_h3)), 0.0f);
        scores_out[row * eff_max_c + c] =
            r0 * s_w[0] + r1 * s_w[1] + r2 * s_w[2] + r3 * s_w[3];
    }
}

// Fast exact Top-512 + ascending sort + MLA arena slot gather for `eff_max_c <= 2048` (`block = 512`).
__global__ void indexer_topk512_slots_kernel(
    const float* __restrict__ scores_red,
    const int32_t* __restrict__ mla_bt,
    const int64_t* __restrict__ row_seq_ids,
    const int32_t* __restrict__ positions,
    int32_t* __restrict__ shared_full_topk,
    int compress_ratio,
    int lat_pp,
    int bt_width,
    int eff_max_c,
    int mla_arena_base) {
    __shared__ float s_val[2048];
    __shared__ int32_t s_idx[2048];

    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;  // 0..511
    const int comp_len = (positions[row] + 1) / compress_ratio;
    const int64_t seq_id = row_seq_ids[row];
    int32_t* out_topk = shared_full_topk + row * 640 + 128;

    // Exact O(1) fast path when `comp_len <= 512`: all reachable indices `0..comp_len-1` are in Top-512!
    if (comp_len <= 512) {
        int32_t slot = -1;
        if (tid < comp_len && tid < eff_max_c) {
            const float sc = scores_red[row * eff_max_c + tid];
            if (sc > -1.0e19f) {
                const int p_col = tid / lat_pp;
                const int32_t phys_p =
                    (p_col < bt_width) ? mla_bt[seq_id * bt_width + p_col] : 0;
                if (phys_p > 0) {
                    slot = mla_arena_base + phys_p * lat_pp + (tid - p_col * lat_pp);
                }
            }
        }
        out_topk[tid] = slot;
        return;
    }

    // General exact 2048-element parallel bitonic Top-512 + 512-element ascending index sort
    const float* row_sc = scores_red + row * eff_max_c;
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        const int c = u * 512 + tid;
        s_val[c] = (c < eff_max_c && c < comp_len) ? row_sc[c] : -1.0e30f;
        s_idx[c] = c;
    }
    __syncthreads();

    for (int k = 2; k <= 2048; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            #pragma unroll
            for (int pass = 0; pass < 2; ++pass) {
                const int pair_id = pass * 512 + tid;
                const int low = pair_id & (j - 1);
                const int i = ((pair_id - low) << 1) + low;
                const int ixj = i ^ j;
                const float vi = s_val[i];
                const float vx = s_val[ixj];
                const int32_t ii = s_idx[i];
                const int32_t ix = s_idx[ixj];
                // Primary direction (`(i & k) == 0`): `(val DESC, idx ASC)`
                const bool i_worse = (vi < vx) || (vi == vx && ii > ix);
                const bool x_worse = (vx < vi) || (vx == vi && ix > ii);
                const bool do_swap = ((i & k) == 0) ? i_worse : x_worse;
                if (do_swap) {
                    s_val[i] = vx;
                    s_idx[i] = ix;
                    s_val[ixj] = vi;
                    s_idx[ixj] = ii;
                }
            }
            __syncthreads();
        }
    }

    // Sort top 512 selected entries ascending by `s_idx`
    for (int k = 2; k <= 512; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            if (tid < 256) {
                const int low = tid & (j - 1);
                const int i = ((tid - low) << 1) + low;
                const int ixj = i ^ j;
                const int32_t ii = s_idx[i];
                const int32_t ix = s_idx[ixj];
                const bool do_swap = ((i & k) == 0) ? (ii > ix) : (ii < ix);
                if (do_swap) {
                    const float vi = s_val[i];
                    s_val[i] = s_val[ixj];
                    s_val[ixj] = vi;
                    s_idx[i] = ix;
                    s_idx[ixj] = ii;
                }
            }
            __syncthreads();
        }
    }

    const int k_pick = min(512, eff_max_c);
    int32_t slot = -1;
    if (tid < k_pick) {
        const int32_t c = s_idx[tid];
        const float v = s_val[tid];
        if (c >= 0 && c < comp_len && v > -1.0e19f) {
            const int p_col = c / lat_pp;
            const int32_t phys_p =
                (p_col < bt_width) ? mla_bt[seq_id * bt_width + p_col] : 0;
            if (phys_p > 0) {
                slot = mla_arena_base + phys_p * lat_pp + (c - p_col * lat_pp);
            }
        }
    }
    out_topk[tid] = slot;
}

}  // namespace

// ---------------------------------------------------------------------------
// Host Launch Wrappers
// ---------------------------------------------------------------------------

std::uintptr_t register_mapped_host_cuda(std::uintptr_t host_address, int64_t bytes) {
    void* host_ptr = reinterpret_cast<void*>(host_address);
    ZSTEP_SAIL_CUDA_CHECK(cudaHostRegister(
        host_ptr,
        static_cast<size_t>(bytes),
        cudaHostRegisterMapped | cudaHostRegisterPortable));
    void* dev_ptr = nullptr;
    ZSTEP_SAIL_CUDA_CHECK(cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0));
    return reinterpret_cast<std::uintptr_t>(dev_ptr);
}

void unregister_mapped_host_cuda(std::uintptr_t host_address) {
    (void)cudaHostUnregister(reinterpret_cast<void*>(host_address));
}

int64_t enable_peer_access_cuda(int peer_device) {
    int cur = -1;
    ZSTEP_SAIL_CUDA_CHECK(cudaGetDevice(&cur));
    if (cur == peer_device) return 1;
    int can = 0;
    ZSTEP_SAIL_CUDA_CHECK(cudaDeviceCanAccessPeer(&can, cur, peer_device));
    if (!can) return 0;
    cudaError_t err = cudaDeviceEnablePeerAccess(peer_device, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        (void)cudaGetLastError();
        return 1;
    }
    ZSTEP_SAIL_CUDA_CHECK(err);
    return 1;
}

torch::Tensor alloc_ipc_buffer_cuda(int64_t bytes, torch::Tensor out_ptr_u64) {
    TORCH_CHECK(out_ptr_u64.scalar_type() == at::kLong && out_ptr_u64.numel() >= 1, "out_ptr_u64");
    void* dev_ptr = nullptr;
    ZSTEP_SAIL_CUDA_CHECK(cudaMalloc(&dev_ptr, static_cast<size_t>(bytes)));
    ZSTEP_SAIL_CUDA_CHECK(cudaMemset(dev_ptr, 0, static_cast<size_t>(bytes)));
    ZSTEP_SAIL_CUDA_CHECK(cudaDeviceSynchronize());
    cudaIpcMemHandle_t handle{};
    cudaError_t err = cudaIpcGetMemHandle(&handle, dev_ptr);
    if (err != cudaSuccess) {
        (void)cudaFree(dev_ptr);
        ZSTEP_SAIL_CUDA_CHECK(err);
    }
    auto handle_t = torch::empty({static_cast<int64_t>(sizeof(cudaIpcMemHandle_t))}, torch::kUInt8);
    std::memcpy(handle_t.data_ptr<uint8_t>(), &handle, sizeof(cudaIpcMemHandle_t));
    out_ptr_u64.data_ptr<int64_t>()[0] =
        static_cast<int64_t>(reinterpret_cast<std::uintptr_t>(dev_ptr));
    return handle_t;
}

std::uintptr_t open_ipc_handle_cuda(torch::Tensor handle_u8) {
    TORCH_CHECK(
        handle_u8.scalar_type() == at::kByte &&
            handle_u8.numel() == static_cast<int64_t>(sizeof(cudaIpcMemHandle_t)),
        "invalid handle_u8");
    cudaIpcMemHandle_t handle{};
    std::memcpy(&handle, handle_u8.data_ptr<uint8_t>(), sizeof(cudaIpcMemHandle_t));
    void* peer_ptr = nullptr;
    ZSTEP_SAIL_CUDA_CHECK(
        cudaIpcOpenMemHandle(&peer_ptr, handle, cudaIpcMemLazyEnablePeerAccess));
    return reinterpret_cast<std::uintptr_t>(peer_ptr);
}

void close_ipc_handle_cuda(std::uintptr_t ptr) {
    if (ptr != 0) {
        (void)cudaIpcCloseMemHandle(reinterpret_cast<void*>(ptr));
    }
}

void free_ipc_buffer_cuda(std::uintptr_t ptr) {
    if (ptr != 0) {
        (void)cudaFree(reinterpret_cast<void*>(ptr));
    }
}

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
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    icn_allreduce_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint4*>(input_ptr),
        reinterpret_cast<uint4*>(output_ptr),
        control_ptr,
        peer_ipc_ptrs,
        reinterpret_cast<const volatile uint64_t*>(abort_dev_ptr),
        slot_stride_bytes,
        oneshot_max_bytes / 16,
        num_vecs,
        rank,
        world_size);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

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
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    icn_allreduce_f32_kernel<<<blocks, threads, 0, stream>>>(
        input_ptr,
        output_ptr,
        control_ptr,
        peer_ipc_ptrs,
        reinterpret_cast<const volatile uint64_t*>(abort_dev_ptr),
        slot_stride_bytes,
        oneshot_max_bytes / 4,
        numel,
        rank,
        world_size);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

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
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    icn_allgather_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint4*>(input_ptr),
        reinterpret_cast<uint4*>(output_ptr),
        control_ptr,
        peer_ipc_ptrs,
        reinterpret_cast<const volatile uint64_t*>(abort_dev_ptr),
        slot_stride_bytes,
        outer_rows,
        vecs_per_row,
        rank,
        world_size);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

void launch_local_top2_bf16_cuda(
    const void* logits_bf16,
    float* out_top2_vals,
    int32_t* out_top2_ids,
    int64_t num_rows,
    int64_t stride_row,
    int vocab_per_rank,
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    local_top2_bf16_kernel<<<static_cast<unsigned>(num_rows), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits_bf16),
        out_top2_vals,
        out_top2_ids,
        stride_row,
        vocab_per_rank);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

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
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    icn_tp8_top2_gather_kernel<<<1, 256, 0, stream>>>(
        local_top2_vals,
        local_top2_ids,
        out_top2_tokens,
        out_top2_logit_bits,
        step_counter,
        peer_ipc_ptrs,
        reinterpret_cast<const volatile uint64_t*>(abort_dev_ptr),
        ipc_base_offset,
        num_rows,
        max_rows_per_step,
        rank,
        world_size,
        vocab_per_rank);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

void launch_decode_prepare_tokens_cuda(
    int32_t* dec_tokens,
    int32_t* dec_lookback,
    const int32_t* slot_last_tok,
    const int32_t* slot_lookback,
    const int32_t* dec_req_slots,
    const int32_t* dec_mode,
    int bucket,
    int max_seqs,
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    decode_prepare_tokens_kernel<<<1, 128, 0, stream>>>(
        dec_tokens,
        dec_lookback,
        slot_last_tok,
        slot_lookback,
        dec_req_slots,
        dec_mode,
        bucket,
        max_seqs);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

void launch_decode_commit_tokens_cuda(
    int32_t* slot_last_tok,
    int32_t* slot_lookback,
    const int32_t* dec_tokens,
    const int32_t* dec_lookback,
    const int32_t* dec_req_slots,
    const int32_t* sampled_tokens,
    int num_active,
    int max_seqs,
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    decode_commit_tokens_kernel<<<1, 128, 0, stream>>>(
        slot_last_tok,
        slot_lookback,
        dec_tokens,
        dec_lookback,
        dec_req_slots,
        sampled_tokens,
        num_active,
        max_seqs);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

void launch_rmsnorm_bf16_cuda(
    const void* x_ptr,
    const void* weight_ptr,
    void* out_ptr,
    int64_t rows,
    int dim,
    float eps,
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    rmsnorm_bf16_kernel<<<static_cast<unsigned>(rows), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_ptr),
        reinterpret_cast<const __nv_bfloat16*>(weight_ptr),
        reinterpret_cast<__nv_bfloat16*>(out_ptr),
        dim,
        eps);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

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
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    mhc_fused_mixes_bf16_kernel<<<static_cast<unsigned>(rows), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_ptr),
        hc_fn_ptr,
        hc_scale_ptr,
        hc_base_ptr,
        pre_out_ptr,
        post_out_ptr,
        comb_out_ptr,
        norm_eps,
        hc_eps,
        sinkhorn_iters);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

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
    std::uintptr_t stream_handle) {
    if (split_k <= 1) {
        launch_mhc_fused_mixes_bf16_cuda(
            x_ptr,
            hc_fn_ptr,
            hc_scale_ptr,
            hc_base_ptr,
            pre_out_ptr,
            post_out_ptr,
            comb_out_ptr,
            rows,
            norm_eps,
            hc_eps,
            sinkhorn_iters,
            stream_handle);
        return;
    }
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(split_k));
    mhc_splitk_stage1_bf16_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_ptr),
        hc_fn_ptr,
        partial_out_ptr,
        split_k);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
    mhc_splitk_reduce_sinkhorn_kernel<<<static_cast<unsigned>(rows), 32, 0, stream>>>(
        partial_out_ptr,
        hc_scale_ptr,
        hc_base_ptr,
        pre_out_ptr,
        post_out_ptr,
        comb_out_ptr,
        split_k,
        norm_eps,
        hc_eps,
        sinkhorn_iters);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

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
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    mhc_sinkhorn_from_proj_kernel<<<static_cast<unsigned>(rows), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_ptr),
        raw_proj_ptr,
        hc_scale_ptr,
        hc_base_ptr,
        pre_out_ptr,
        post_out_ptr,
        comb_out_ptr,
        norm_eps,
        hc_eps,
        sinkhorn_iters);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

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
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    mhc_pre_rmsnorm_bf16_kernel<<<static_cast<unsigned>(rows), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_ptr),
        pre_mix_ptr,
        reinterpret_cast<const __nv_bfloat16*>(norm_weight_ptr),
        reinterpret_cast<__nv_bfloat16*>(normed_out_ptr),
        reinterpret_cast<__nv_bfloat16*>(aux_out_ptr),
        dim,
        write_aux,
        norm_eps);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

void launch_mhc_post_bf16_cuda(
    const void* sub_out_ptr,
    const void* residual_ptr,
    const float* post_mix_ptr,
    const float* comb_mix_ptr,
    void* out_ptr,
    int64_t rows,
    int dim,
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const unsigned grid_y = static_cast<unsigned>((dim + 255) / 256);
    dim3 grid(static_cast<unsigned>(rows), grid_y);
    mhc_post_bf16_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(sub_out_ptr),
        reinterpret_cast<const __nv_bfloat16*>(residual_ptr),
        post_mix_ptr,
        comb_mix_ptr,
        reinterpret_cast<__nv_bfloat16*>(out_ptr),
        dim);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

void launch_act_quant_ue8m0_bf16_cuda(
    const void* src_ptr,
    void* dst_ptr,
    int64_t num_groups,
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    constexpr int kWarpsPerBlock = 8;
    const unsigned blocks = static_cast<unsigned>((num_groups + kWarpsPerBlock - 1) / kWarpsPerBlock);
    act_quant_ue8m0_bf16_kernel<<<blocks, kWarpsPerBlock * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src_ptr),
        reinterpret_cast<__nv_bfloat16*>(dst_ptr),
        num_groups);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

void launch_inplace_act_quant_ue8m0_bf16_cuda(
    void* x_ptr,
    int64_t num_groups,
    std::uintptr_t stream_handle) {
    launch_act_quant_ue8m0_bf16_cuda(x_ptr, x_ptr, num_groups, stream_handle);
}

void launch_shared_swiglu_quant_bf16_cuda(
    const void* sh_gu_ptr,
    void* sh_act_out_ptr,
    int64_t rows,
    int inter_dim,
    float swiglu_limit,
    int do_ue8m0,
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const int groups_per_row = inter_dim / 32;
    const int64_t total_groups = rows * groups_per_row;
    constexpr int kWarpsPerBlock = 8;
    const unsigned blocks =
        static_cast<unsigned>((total_groups + kWarpsPerBlock - 1) / kWarpsPerBlock);
    shared_swiglu_quant_bf16_kernel<<<blocks, kWarpsPerBlock * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(sh_gu_ptr),
        reinterpret_cast<__nv_bfloat16*>(sh_act_out_ptr),
        total_groups,
        groups_per_row,
        inter_dim,
        swiglu_limit,
        do_ue8m0);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

void launch_inplace_fp4_quant_bf16_cuda(
    void* x_ptr,
    int64_t num_groups,
    int block_size,
    int is_e4m3_scale,
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const int groups_per_warp = (block_size == 16) ? 2 : 1;
    constexpr int kWarpsPerBlock = 8;
    const int groups_per_block = kWarpsPerBlock * groups_per_warp;
    const unsigned blocks = static_cast<unsigned>((num_groups + groups_per_block - 1) / groups_per_block);
    inplace_fp4_quant_bf16_kernel<<<blocks, kWarpsPerBlock * 32, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(x_ptr),
        num_groups,
        block_size,
        is_e4m3_scale);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

void launch_rope_inplace_bf16_cuda(
    void* x_ptr,
    const float* cos_sin_ptr,
    const int32_t* positions_ptr,
    int64_t rows,
    int num_heads,
    int head_dim,
    int rope_dim,
    int inverse,
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const int threads = min(256, max(32, ((num_heads * (rope_dim / 2) + 31) / 32) * 32));
    rope_inplace_bf16_kernel<<<static_cast<unsigned>(rows), threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(x_ptr),
        cos_sin_ptr,
        positions_ptr,
        num_heads,
        head_dim,
        rope_dim,
        inverse);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

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
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    dsv41_router_topk_kernel<<<static_cast<unsigned>(rows), 128, 0, stream>>>(
        logits_ptr,
        bias_ptr,
        out_weights_ptr,
        out_indices_ptr,
        num_experts,
        top_k,
        route_scale,
        logits_is_bf16);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

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
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const int warps = threads >> 5;
    const unsigned grid_y = static_cast<unsigned>((inter_per_rank + warps - 1) / warps);
    dim3 grid(static_cast<unsigned>(rows * top_k), grid_y);
    const size_t smem = static_cast<size_t>(hidden_dim) * sizeof(float);
    mxfp4_moe_gemv_gate_up_kernel<<<grid, threads, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_ptr),
        w13_packed_ptr,
        w13_scale_ptr,
        topk_ids_ptr,
        topk_weights_ptr,
        reinterpret_cast<__nv_bfloat16*>(out_inter_ptr),
        top_k,
        inter_per_rank,
        hidden_dim,
        swiglu_limit);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

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
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const int subs = threads >> 3;
    const unsigned grid_y = static_cast<unsigned>((hidden_dim + subs - 1) / subs);
    dim3 grid(static_cast<unsigned>(rows), grid_y);
    const size_t smem = static_cast<size_t>(top_k * inter_per_rank) * sizeof(float);
    mxfp4_moe_gemv_down_combine_kernel<<<grid, threads, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(inter_ptr),
        w2_packed_ptr,
        w2_scale_ptr,
        topk_ids_ptr,
        reinterpret_cast<const __nv_bfloat16*>(shared_out_ptr),
        reinterpret_cast<__nv_bfloat16*>(out_ptr),
        top_k,
        hidden_dim,
        inter_per_rank);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

void launch_moe_align_block_size_cuda(
    const int32_t* topk_ids_ptr,
    int32_t* sorted_pair_ids_ptr,
    int32_t* expert_ids_per_tile_ptr,
    int32_t* num_valid_tiles_ptr,
    int64_t num_pairs,
    int64_t max_padded_slots,
    int num_experts,
    int block_m,
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    moe_align_block_size_kernel<<<1, 512, 0, stream>>>(
        topk_ids_ptr,
        sorted_pair_ids_ptr,
        expert_ids_per_tile_ptr,
        num_valid_tiles_ptr,
        num_pairs,
        max_padded_slots,
        num_experts,
        block_m);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

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
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    engram_lookup_dequant_kernel<<<static_cast<unsigned>(total_tokens), 256, 0, stream>>>(
        token_ids_ptr,
        prev_window_ptr,
        query_start_loc_ptr,
        token_map_ptr,
        multipliers_ptr,
        primes_ptr,
        offsets_ptr,
        weight_fp8_ptr,
        scale_u8_ptr,
        reinterpret_cast<__nv_bfloat16*>(out_bf16_ptr),
        num_seqs,
        vocab_start_idx,
        vocab_end_idx,
        pad_compressed_id);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

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
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    int split_k = 1;
    if (rows <= 8 && max_topk >= 256) {
        split_k = 8;
    } else if (rows <= 32 && max_topk >= 256) {
        split_k = 4;
    }
    dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(split_k));
    sparse_attn_tp8_stage1_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_ptr),
        reinterpret_cast<const __nv_bfloat16*>(kv_arena_ptr),
        attn_sink_ptr,
        topk_idxs_ptr,
        partial_acc_ptr,
        partial_m_ptr,
        partial_l_ptr,
        reinterpret_cast<__nv_bfloat16*>(out_ptr),
        max_topk,
        split_k,
        sm_scale);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
    if (split_k > 1) {
        sparse_attn_tp8_reduce_kernel<<<static_cast<unsigned>(rows), 256, 0, stream>>>(
            partial_acc_ptr,
            partial_m_ptr,
            partial_l_ptr,
            attn_sink_ptr,
            reinterpret_cast<__nv_bfloat16*>(out_ptr),
            split_k);
        ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
    }
}

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
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    mhc_pre_rmsnorm_quant_bf16_kernel<<<static_cast<unsigned>(rows), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_ptr),
        pre_mix_ptr,
        reinterpret_cast<const __nv_bfloat16*>(norm_weight_ptr),
        reinterpret_cast<__nv_bfloat16*>(normed_out_ptr),
        reinterpret_cast<__nv_bfloat16*>(quant_out_ptr),
        dim,
        write_normed,
        norm_eps);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

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
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    fused_qkv_a_epilogue_bf16_kernel<<<static_cast<unsigned>(rows), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv_a_ptr),
        reinterpret_cast<const __nv_bfloat16*>(q_norm_w_ptr),
        reinterpret_cast<const __nv_bfloat16*>(kv_norm_w_ptr),
        cos_sin_ptr,
        positions_ptr,
        swa_write_slots_ptr,
        reinterpret_cast<__nv_bfloat16*>(qr_unquant_out_ptr),
        reinterpret_cast<__nv_bfloat16*>(qr_quant_out_ptr),
        reinterpret_cast<__nv_bfloat16*>(kv_arena_ptr),
        q_rank,
        kv_dim,
        rope_dim,
        write_qr_unquant,
        layer_swa_base,
        eps);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

void launch_fused_q_idx_epilogue_bf16_cuda(
    const void* q_and_idx_ptr,
    const float* cos_sin_ptr,
    const int32_t* positions_ptr,
    void* q_out_ptr,
    void* q_idx_out_ptr,
    int64_t rows,
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    fused_q_idx_epilogue_bf16_kernel<<<static_cast<unsigned>(rows), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_and_idx_ptr),
        cos_sin_ptr,
        positions_ptr,
        reinterpret_cast<__nv_bfloat16*>(q_out_ptr),
        reinterpret_cast<__nv_bfloat16*>(q_idx_out_ptr));
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

void launch_fused_inv_rope_act_quant_bf16_cuda(
    void* attn_o_ptr,
    const float* cos_sin_ptr,
    const int32_t* positions_ptr,
    int64_t rows,
    int num_heads,
    int head_dim,
    int rope_dim,
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(num_heads));
    fused_inv_rope_act_quant_bf16_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(attn_o_ptr),
        cos_sin_ptr,
        positions_ptr,
        num_heads,
        head_dim,
        rope_dim);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

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
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    dim3 grid(static_cast<unsigned>(chunk_tiles), static_cast<unsigned>(block_m));
    moe_gather_tile_inputs_bf16_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const uint4*>(x_ptr),
        sorted_pair_ids_ptr,
        num_valid_tiles_ptr,
        reinterpret_cast<uint4*>(tile_a_ptr),
        tile_offset,
        block_m,
        top_k,
        hidden_dim / 8);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

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
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const int total_groups = n_rows * (k_cols >> 5);
    const unsigned blocks_x = static_cast<unsigned>(min(256, (total_groups + 255) / 256));
    dim3 grid(blocks_x, static_cast<unsigned>(chunk_tiles));
    dequant_mxfp4_tiles_bf16_kernel<<<grid, 256, 0, stream>>>(
        w_packed_ptr,
        w_scale_ptr,
        expert_ids_ptr,
        num_valid_tiles_ptr,
        reinterpret_cast<__nv_bfloat16*>(out_bf16_ptr),
        tile_offset,
        chunk_tiles,
        n_rows,
        k_cols);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

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
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const unsigned grid_y = static_cast<unsigned>(((inter_dim >> 5) + 7) / 8);
    dim3 grid(static_cast<unsigned>(chunk_tiles * block_m), grid_y);
    moe_tile_swiglu_quant_bf16_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const float*>(tile_gu_ptr),
        sorted_pair_ids_ptr,
        num_valid_tiles_ptr,
        topk_weights_ptr,
        reinterpret_cast<__nv_bfloat16*>(tile_inter_out_ptr),
        tile_offset,
        block_m,
        inter_dim,
        swiglu_limit,
        do_ue8m0);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

void launch_bmm_bf16_f32_nt_cuda(
    const void* a_bf16_ptr,
    const void* b_bf16_ptr,
    float* c_f32_ptr,
    int batch,
    int m,
    int n,
    int k) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasStatus_t st = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        b_bf16_ptr,
        CUDA_R_16BF,
        k,
        static_cast<long long>(n) * k,
        a_bf16_ptr,
        CUDA_R_16BF,
        k,
        static_cast<long long>(m) * k,
        &beta,
        c_f32_ptr,
        CUDA_R_32F,
        n,
        static_cast<long long>(m) * n,
        batch,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CHECK(
        st == CUBLAS_STATUS_SUCCESS,
        "SAIL cublasGemmStridedBatchedEx failed with status ",
        static_cast<int>(st));
}

void launch_moe_scatter_tile_outputs_bf16_cuda(
    const void* tile_dn_ptr,
    const int32_t* sorted_pair_ids_ptr,
    const int32_t* num_valid_tiles_ptr,
    void* pair_out_ptr,
    int tile_offset,
    int chunk_tiles,
    int block_m,
    int hidden_dim,
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    dim3 grid(static_cast<unsigned>(chunk_tiles), static_cast<unsigned>(block_m));
    moe_scatter_tile_outputs_bf16_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const uint4*>(tile_dn_ptr),
        sorted_pair_ids_ptr,
        num_valid_tiles_ptr,
        reinterpret_cast<uint4*>(pair_out_ptr),
        tile_offset,
        block_m,
        hidden_dim / 8);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

void launch_moe_combine_pairs_shared_bf16_cuda(
    const void* pair_out_ptr,
    const void* shared_out_ptr,
    void* out_ptr,
    int rows,
    int top_k,
    int hidden_dim,
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const int hidden_u4 = hidden_dim / 8;
    const unsigned grid_y = static_cast<unsigned>((hidden_u4 + 255) / 256);
    dim3 grid(static_cast<unsigned>(rows), grid_y);
    moe_combine_pairs_shared_bf16_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const uint4*>(pair_out_ptr),
        reinterpret_cast<const uint4*>(shared_out_ptr),
        reinterpret_cast<uint4*>(out_ptr),
        top_k,
        hidden_u4);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

void launch_engram_gated_residual_bf16_cuda(
    void* h_stream_ptr,
    const void* kv_proj_ptr,
    const float* qk_prod_ptr,
    int rows,
    int hidden_size,
    float norm_eps,
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    dim3 grid(static_cast<unsigned>(rows), 4U);
    engram_gated_residual_bf16_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(h_stream_ptr),
        reinterpret_cast<const __nv_bfloat16*>(kv_proj_ptr),
        qk_prod_ptr,
        hidden_size,
        norm_eps);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

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
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const unsigned grid_y = static_cast<unsigned>((eff_max_c + 7) / 8);
    dim3 grid(static_cast<unsigned>(rows), grid_y);
    indexer_paged_score_bf16_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_idx_ptr),
        reinterpret_cast<const __nv_bfloat16*>(w_idx_ptr),
        reinterpret_cast<const __nv_bfloat16*>(k_cache_ptr),
        mla_bt_ptr,
        row_seq_ids_ptr,
        positions_ptr,
        scores_out_ptr,
        weight_scale,
        compress_ratio,
        lat_pp,
        bt_width,
        eff_max_c);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}

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
    std::uintptr_t stream_handle) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    indexer_topk512_slots_kernel<<<static_cast<unsigned>(rows), 512, 0, stream>>>(
        scores_red_ptr,
        mla_bt_ptr,
        row_seq_ids_ptr,
        positions_ptr,
        shared_full_topk_ptr,
        compress_ratio,
        lat_pp,
        bt_width,
        eff_max_c,
        mla_arena_base);
    ZSTEP_SAIL_CUDA_CHECK(cudaGetLastError());
}
