
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

__device__ __forceinline__ void gpu_store_release_u64(uint64_t* addr, uint64_t val) {
    asm volatile("st.global.release.gpu.b64 [%0], %1;" :: "l"(addr), "l"(val) : "memory");
}
__device__ __forceinline__ bool gpu_spin_wait_ge_u64(const uint64_t* flag_addr, uint64_t target, const uint64_t* abort_addr) {
    uint32_t iters = 0;
    uint64_t start_cycle = 0, val = 0;
    for (;; __nanosleep(48)) {
        asm volatile("ld.global.acquire.gpu.b64 %0, [%1];" : "=l"(val) : "l"(flag_addr) : "memory");
        if (val >= target) return true;
        if ((++iters & 0xFFFU) == 0U) {
            if (sys_load_acquire_u64(abort_addr) != 0ULL) return false;
            const uint64_t now = clock64();
            if (start_cycle == 0ULL) start_cycle = now;
            else if (now - start_cycle > kSpinTimeoutCycles) return false;
        }
    }
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
// Inline 128-bit `FP12 + BFP7` packet (`1x uint4` = 16B per 16 BF16s = 50% cross-SHM size):
// - `hdr16 = (h1_big << 15) | (fp12 << 3) | (sh & 7)`
//   * `fp12` (`16..0xFD0`): 8-bit FP32 exponent + 4-bit FP32 mantissa rounded up from `big_max / 63.0f`
//   * `fp12 == 0xFFF`: reserved non-finite escape header (`q[i]`: 0=0.0f, 1=+Inf, 2=-Inf, 3=qNaN)
//   * `sh`:   3-bit `4^sh` relative shift between sub-blocks `v[0..7]` and `v[8..15]` (up to 16,384x)
// - `payload`: 16x 7-bit two's-complement signed integers `[-63..+63]`
__device__ __forceinline__ uint4 pack_fp12_bfp7_words(
    uint32_t hdr16,
    const uint32_t (&q)[16]) {
    const uint32_t w0 = hdr16 | (q[0] << 16) | (q[1] << 23) | ((q[2] & 0x3U) << 30);
    const uint32_t w1 =
        (q[2] >> 2) | (q[3] << 5) | (q[4] << 12) | (q[5] << 19) | ((q[6] & 0x3FU) << 26);
    const uint32_t w2 = (q[6] >> 6) | (q[7] << 1) | (q[8] << 8) | (q[9] << 15) |
        (q[10] << 22) | ((q[11] & 0x7U) << 29);
    const uint32_t w3 =
        (q[11] >> 3) | (q[12] << 4) | (q[13] << 11) | (q[14] << 18) | (q[15] << 25);
    return make_uint4(w0, w1, w2, w3);
}

__device__ __forceinline__ uint4 pack_fp12_bfp7(const float (&v)[16]) {
    float amax0 = 0.0f;
    float amax1 = 0.0f;
    uint32_t has_non_finite = 0U;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t b0 = __float_as_uint(v[i]) & 0x7FFFFFFFU;
        const uint32_t b1 = __float_as_uint(v[i + 8]) & 0x7FFFFFFFU;
        if (b0 >= 0x7F800000U || b1 >= 0x7F800000U) {
            has_non_finite = 1U;
        }
        amax0 = fmaxf(amax0, __uint_as_float(b0));
        amax1 = fmaxf(amax1, __uint_as_float(b1));
    }
    if (__builtin_expect(has_non_finite != 0U, 0)) {
        uint32_t q_nf[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint32_t bits = __float_as_uint(v[i]);
            const uint32_t abs_bits = bits & 0x7FFFFFFFU;
            if (abs_bits > 0x7F800000U) {
                q_nf[i] = 3U;
            } else if (abs_bits == 0x7F800000U) {
                q_nf[i] = (bits & 0x80000000U) ? 2U : 1U;
            } else {
                q_nf[i] = 0U;
            }
        }
        return pack_fp12_bfp7_words(0xFFFU << 3, q_nf);
    }
    const uint32_t h1_big = (amax1 > amax0) ? 1U : 0U;
    const float big_max = h1_big ? amax1 : amax0;
    const float sml_max = h1_big ? amax0 : amax1;
    if (big_max < 1.0e-36f) {
        return make_uint4(0U, 0U, 0U, 0U);
    }
    const uint32_t raw_bits = __float_as_uint(big_max * (1.0f / 63.0f));
    uint32_t fp12 = min(0xFD0U, (raw_bits + 0x0007FFFFU) >> 19);
    if (fp12 < 16U) {
        return make_uint4(0U, 0U, 0U, 0U);
    }
    const float inv_big = 1.0f / __uint_as_float(fp12 << 19);
    uint32_t sh = 0U;
    float inv_sml = 0.0f;
    if (sml_max > 1.0e-36f) {
        const int log2_r =
            static_cast<int>((__float_as_uint(big_max / sml_max) >> 23) & 0xFFU) - 127;
        sh = static_cast<uint32_t>(min(7, max(0, log2_r >> 1)));
        inv_sml = inv_big * __uint_as_float((127U + 2U * sh) << 23);
    }
    const float inv0 = h1_big ? inv_sml : inv_big;
    const float inv1 = h1_big ? inv_big : inv_sml;

    uint32_t q[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        q[i] = static_cast<uint32_t>(max(-63, min(63, __float2int_rn(v[i] * inv0)))) & 0x7FU;
        q[i + 8] =
            static_cast<uint32_t>(max(-63, min(63, __float2int_rn(v[i + 8] * inv1)))) & 0x7FU;
    }
    const uint32_t hdr16 = (h1_big << 15) | ((fp12 & 0xFFFU) << 3) | (sh & 0x7U);
    return pack_fp12_bfp7_words(hdr16, q);
}

__device__ __forceinline__ int sign_ext7(uint32_t u) {
    return static_cast<int>((u ^ 0x40U) - 0x40U);
}

__device__ __forceinline__ void unpack_accum_fp12_bfp7(uint4 pkt, float (&acc)[16]) {
    const uint32_t hdr16 = pkt.x & 0xFFFFU;
    const uint32_t fp12 = (hdr16 >> 3) & 0xFFFU;
    if (fp12 == 0U) {
        return;
    }
    const uint32_t q[16] = {
        (pkt.x >> 16) & 0x7FU,
        (pkt.x >> 23) & 0x7FU,
        ((pkt.x >> 30) | (pkt.y << 2)) & 0x7FU,
        (pkt.y >> 5) & 0x7FU,
        (pkt.y >> 12) & 0x7FU,
        (pkt.y >> 19) & 0x7FU,
        ((pkt.y >> 26) | (pkt.z << 6)) & 0x7FU,
        (pkt.z >> 1) & 0x7FU,
        (pkt.z >> 8) & 0x7FU,
        (pkt.z >> 15) & 0x7FU,
        (pkt.z >> 22) & 0x7FU,
        ((pkt.z >> 29) | (pkt.w << 3)) & 0x7FU,
        (pkt.w >> 4) & 0x7FU,
        (pkt.w >> 11) & 0x7FU,
        (pkt.w >> 18) & 0x7FU,
        (pkt.w >> 25) & 0x7FU,
    };
    if (__builtin_expect(fp12 == 0xFFFU, 0)) {
        const float pos_inf = __uint_as_float(0x7F800000U);
        const float neg_inf = __uint_as_float(0xFF800000U);
        const float quiet_nan = __uint_as_float(0x7FC00000U);
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            if (q[i] == 1U) {
                acc[i] += pos_inf;
            } else if (q[i] == 2U) {
                acc[i] += neg_inf;
            } else if (q[i] == 3U) {
                acc[i] += quiet_nan;
            }
        }
        return;
    }
    const float s_big = __uint_as_float(fp12 << 19);
    const float s_sml = s_big * __uint_as_float((127U - 2U * (hdr16 & 0x7U)) << 23);
    const float s0 = (hdr16 & 0x8000U) ? s_sml : s_big;
    const float s1 = (hdr16 & 0x8000U) ? s_big : s_sml;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc[i] += static_cast<float>(sign_ext7(q[i])) * s0;
        acc[i + 8] += static_cast<float>(sign_ext7(q[i + 8])) * s1;
    }
}

__device__ __forceinline__ void unpack_to_bf16_pair(uint4 pkt, uint4& out0, uint4& out1) {
    float tmp[16] = {0.f};
    unpack_accum_fp12_bfp7(pkt, tmp);
    Pack128Bf16 p0, p1;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        p0.pairs[j] = __floats2bfloat162_rn(tmp[2 * j], tmp[2 * j + 1]);
        p1.pairs[j] = __floats2bfloat162_rn(tmp[8 + 2 * j], tmp[8 + 2 * j + 1]);
    }
    out0 = p0.u4;
    out1 = p1.u4;
}

__device__ __forceinline__ uint4 pack_bf16_pair(const uint4* __restrict__ vec, int64_t idx0) {
    Pack128Bf16 a0, a1;
    a0.u4 = vec[idx0];
    a1.u4 = vec[idx0 + 1];
    float v[16];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = __bfloat162float(a0.elems[j]);
        v[8 + j] = __bfloat162float(a1.elems[j]);
    }
    return pack_fp12_bfp7(v);
}

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
    int flag_base,
    int pxb_bfp,
    uint64_t* __restrict__ ar_cta_done = nullptr) {
    const int b_local = blockIdx.x;
    const int b = flag_base + b_local;
    auto signal_ar_cta_done = [&]() {
        if (threadIdx.x == 0 && ar_cta_done != nullptr) {
            __threadfence();
            gpu_store_release_u64(ar_cta_done + b_local, ar_cta_done[b_local] + 1ULL);
        }
    };
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
    const bool split_regime = (flag_base > 0 || (my_hbm != nullptr && gridDim.x >= 32 && gridDim.x <= 56));
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
            signal_ar_cta_done();
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

    if (can_pxb && pxb_bfp == 1 && num_vecs > kPxbOneShotMinVecs && num_vecs <= 15360 && (num_vecs & 15) == 0 && (num_vecs >> 4) >= gridDim.x) {
        const int parity = rank & 1, pair_id = rank >> 1, tid = threadIdx.x, sh = ((num_vecs & 127) == 0 && (num_vecs >> 7) >= gridDim.x) ? 7 : 4;
        const int64_t g16 = num_vecs >> sh, fb = ((g16 * b_local) / gridDim.x) << sh, fe = (b_local + 1 == gridDim.x) ? num_vecs : (((g16 * (b_local + 1)) / gridDim.x) << sh);
        const int64_t fm = fb + (((fe - fb) >> 2) << 1), p_fb = fb >> 1, p_fm = fm >> 1, p_fe = fe >> 1;
        const int64_t my_pb = parity ? p_fm : p_fb, my_pe = parity ? p_fe : p_fm, pt_pb = parity ? p_fb : p_fm, pt_pe = parity ? p_fm : p_fe;
        uint4* pt_in = reinterpret_cast<uint4*>(pt_hbm + pxb_base_off);
        for (int64_t p = pt_pb + tid; p < pt_pe; p += blockDim.x) st_wt_uint4(pt_in + p, pack_bf16_pair(input_vec, p << 1));
        __syncthreads();
        if (tid == 0) {
            __threadfence_system();
            sys_store_release_u64(reinterpret_cast<uint64_t*>(pt_hbm + b * 64), step);
            if (!spin_wait_ge_u64(reinterpret_cast<const uint64_t*>(my_hbm + b * 64), step, abort_flag)) s_aborted = 1;
        }
        __syncthreads();
        if (s_aborted) { signal_ar_cta_done(); return; }
        const uint4* from_pt = reinterpret_cast<const uint4*>(my_hbm + pxb_base_off);
        uint4* my_hbm_sum = reinterpret_cast<uint4*>(my_hbm + pxb_base_off + cur_pxb_stride / 2);
        for (int64_t p = my_pb + tid; p < my_pe; p += blockDim.x) {
            Pack128Bf16 a0{input_vec[p << 1]}, a1{input_vec[(p << 1) + 1]};
            float v[16];
            #pragma unroll
            for (int j = 0; j < 8; ++j) { v[j] = __bfloat162float(a0.elems[j]); v[8 + j] = __bfloat162float(a1.elems[j]); }
            unpack_accum_fp12_bfp7(ld_cv_uint4(from_pt + p), v);
            const uint4 rq = pack_fp12_bfp7(v);
            my_hbm_sum[p] = rq;
            st_wt_uint4(my_in + p, rq);
        }
        __syncthreads();
        if (tid == 0) __threadfence_system();
        __syncthreads();
        if (tid < 8) {
            const int dst = (rank + tid) & 7;
            sys_store_release_u64(reinterpret_cast<uint64_t*>(s_peer_shm[dst] + (b * 8 + rank) * 64), step);
            if (!spin_wait_ge_u64(reinterpret_cast<const uint64_t*>(my_shm + (b * 8 + dst) * 64), step, abort_flag)) s_aborted = 1;
        }
        __syncthreads();
        if (s_aborted) { signal_ar_cta_done(); return; }
        for (int64_t p = p_fb + tid; p < p_fe; p += blockDim.x) {
            const int h_par = (p < p_fm) ? 0 : 1;
            uint4 pkts[4];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int q = (pair_id + k) & 3, src_r = q * 2 + h_par;
                pkts[q] = (src_r == rank) ? my_hbm_sum[p] : ld_cv_uint4(reinterpret_cast<const uint4*>(s_peer_shm[src_r] + 131072 + slot * slot_stride_bytes) + p);
            }
            float acc[16] = {0.f};
            #pragma unroll
            for (int q = 0; q < 4; ++q) unpack_accum_fp12_bfp7(pkts[q], acc);
            Pack128Bf16 o0, o1;
            #pragma unroll
            for (int j = 0; j < 4; ++j) { o0.pairs[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]); o1.pairs[j] = __floats2bfloat162_rn(acc[8 + 2 * j], acc[8 + 2 * j + 1]); }
            output_vec[p << 1] = o0.u4;
            output_vec[(p << 1) + 1] = o1.u4;
        }
        __syncthreads();
        if (tid < 8) sys_store_release_u64(reinterpret_cast<uint64_t*>(s_peer_shm[tid] + 65536 + (b * 8 + rank) * 64), step);
        return;
    }

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
            signal_ar_cta_done();
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
            signal_ar_cta_done();
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
        if (pxb_bfp != 0 && (num_vecs & 15) == 0 && (flag_base == 0 || num_vecs >= 15360)) {
            const int bfp_mode = (pxb_bfp == 2) ? 2 : 1;
            const int parity = rank & 1;
            const int pair_id = rank >> 1;
            const int tid = threadIdx.x;
            const int sh = ((num_vecs & 127) == 0 && (num_vecs >> 7) >= gridDim.x) ? 7 : 4;
            const int64_t g16 = num_vecs >> sh;
            const int64_t n_toks = num_vecs / 640;
            const bool tok_align = (ar_cta_done != nullptr && (num_vecs % 640) == 0 && n_toks >= gridDim.x);
            const int64_t fb = tok_align ? (((n_toks * b_local) / gridDim.x) * 640) : (((g16 * b_local) / gridDim.x) << sh);
            const int64_t fe = tok_align
                ? (((n_toks * (b_local + 1)) / gridDim.x) * 640)
                : ((b_local + 1 == gridDim.x) ? num_vecs : (((g16 * (b_local + 1)) / gridDim.x) << sh));
            const int64_t fm = fb + (((fe - fb) >> 2) << 1);
            const int64_t my_hb = (parity == 0) ? fb : fm;
            const int64_t my_he = (parity == 0) ? fm : fe;
            const int64_t pt_hb = (parity == 0) ? fm : fb;
            const int64_t pt_he = (parity == 0) ? fe : fm;
            const int64_t my_pairs = (my_he - my_hb) >> 1;
            const int64_t pt_pairs = (pt_he - pt_hb) >> 1;
            const int64_t my_pb = my_hb >> 1;
            const int64_t own_pb = my_pb + (my_pairs * pair_id) / 4;
            const int64_t own_pe = my_pb + (my_pairs * (pair_id + 1)) / 4;
            const int64_t half_pxb = cur_pxb_stride / 2;
            const bool do_pipe = (bfp_mode == 1 && flag_base == 0 && gridDim.x == 56 && num_vecs >= 81920), three_stage = (num_vecs <= 327680);
            const int wave_stride = (num_vecs >= 1310720) ? 7 : 14;
            uint64_t *g1_self = reinterpret_cast<uint64_t*>(my_hbm + 118848 + b * 64), *g2_self = reinterpret_cast<uint64_t*>(my_hbm + 122432 + b * 64);
            if (do_pipe && b_local >= wave_stride) {
                if (tid == 0 && !gpu_spin_wait_ge_u64(reinterpret_cast<const uint64_t*>(my_hbm + (three_stage ? 118848 : 122432) + (b - wave_stride) * 64), step, abort_flag)) s_aborted = 1;
                __syncthreads();
                if (s_aborted) {
                    if (tid == 0 && b_local + wave_stride < gridDim.x) { gpu_store_release_u64(g1_self, step); gpu_store_release_u64(g2_self, step); }
                    signal_ar_cta_done();
                    return;
                }
            }

            uint4* pt_in = reinterpret_cast<uint4*>(pt_hbm + pxb_base_off);
            if (bfp_mode == 1) {
                for (int64_t p = (pt_hb >> 1) + tid; p < (pt_he >> 1); p += blockDim.x) {
                    st_wt_uint4(pt_in + p, pack_bf16_pair(input_vec, p << 1));
                }
            } else {
                for (int64_t idx = pt_hb + tid; idx < pt_he; idx += blockDim.x) {
                    st_wt_uint4(pt_in + idx, input_vec[idx]);
                }
            }
            __syncthreads();
            if (tid == 0) {
                __threadfence_system();
                if (do_pipe && three_stage && b_local + wave_stride < gridDim.x) gpu_store_release_u64(g1_self, step);
                sys_store_release_u64(reinterpret_cast<uint64_t*>(pt_hbm + b * 64), step);
                if (!spin_wait_ge_u64(
                        reinterpret_cast<const uint64_t*>(my_hbm + b * 64), step, abort_flag)) {
                    s_aborted = 1;
                }
                if (do_pipe && three_stage && !s_aborted && b_local >= wave_stride &&
                    !gpu_spin_wait_ge_u64(reinterpret_cast<const uint64_t*>(my_hbm + 122432 + (b - wave_stride) * 64), step, abort_flag)) s_aborted = 1;
            }
            __syncthreads();
            if (s_aborted) {
                if (do_pipe && tid == 0 && b_local + wave_stride < gridDim.x) { gpu_store_release_u64(g1_self, step); gpu_store_release_u64(g2_self, step); }
                signal_ar_cta_done();
                return;
            }

            const uint4* from_pt = reinterpret_cast<const uint4*>(my_hbm + pxb_base_off);
            uint4* my_hbm_red = reinterpret_cast<uint4*>(my_hbm + pxb_base_off + half_pxb);
            uint4* pt_hbm_red = reinterpret_cast<uint4*>(pt_hbm + pxb_base_off + half_pxb);
            uint4* my_shm_red =
                reinterpret_cast<uint4*>(my_shm + 131072 + slot * slot_stride_bytes + half_slot);

            for (int64_t p = my_pb + tid; p < my_pb + my_pairs; p += blockDim.x) {
                const int64_t idx0 = p << 1;
                Pack128Bf16 a0, a1, b0, b1;
                a0.u4 = input_vec[idx0];
                a1.u4 = input_vec[idx0 + 1];
                float v[16];
                if (bfp_mode == 1) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        v[j] = __bfloat162float(a0.elems[j]);
                        v[8 + j] = __bfloat162float(a1.elems[j]);
                    }
                    unpack_accum_fp12_bfp7(ld_cv_uint4(from_pt + p), v);
                } else {
                    b0.u4 = ld_cv_uint4(from_pt + idx0);
                    b1.u4 = ld_cv_uint4(from_pt + idx0 + 1);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 fa0 = __bfloat1622float2(a0.pairs[j]);
                        const float2 fb0 = __bfloat1622float2(b0.pairs[j]);
                        const float2 fa1 = __bfloat1622float2(a1.pairs[j]);
                        const float2 fb1 = __bfloat1622float2(b1.pairs[j]);
                        v[2 * j] = fa0.x + fb0.x;
                        v[2 * j + 1] = fa0.y + fb0.y;
                        v[8 + 2 * j] = fa1.x + fb1.x;
                        v[8 + 2 * j + 1] = fa1.y + fb1.y;
                    }
                }
                if (p >= own_pb && p < own_pe) {
                    Pack128Bf16 s0, s1;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        s0.pairs[j] = __floats2bfloat162_rn(v[2 * j], v[2 * j + 1]);
                        s1.pairs[j] = __floats2bfloat162_rn(v[8 + 2 * j], v[8 + 2 * j + 1]);
                    }
                    my_hbm_red[idx0] = s0.u4;
                    my_hbm_red[idx0 + 1] = s1.u4;
                } else {
                    st_wt_uint4(my_in + p, pack_fp12_bfp7(v));
                }
            }
            __syncthreads();
            if (tid == 0) {
                __threadfence_system();
                if (do_pipe && b_local + wave_stride < gridDim.x) gpu_store_release_u64(g2_self, step);
            }
            __syncthreads();
            if (tid < 4 && tid != pair_id) {
                const int dst_r = tid * 2 + parity;
                sys_store_release_u64(
                    reinterpret_cast<uint64_t*>(s_peer_shm[dst_r] + (b * 8 + rank) * 64), step);
                if (!spin_wait_ge_u64(
                        reinterpret_cast<const uint64_t*>(my_shm + (b * 8 + dst_r) * 64),
                        step,
                        abort_flag)) {
                    s_aborted = 1;
                }
            }
            __syncthreads();
            if (s_aborted) {
                signal_ar_cta_done();
                return;
            }

            for (int64_t p = own_pb + tid; p < own_pe; p += blockDim.x) {
                const int64_t idx0 = p << 1;
                Pack128Bf16 own0, own1;
                own0.u4 = my_hbm_red[idx0];
                own1.u4 = my_hbm_red[idx0 + 1];
                float acc[16];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f0 = __bfloat1622float2(own0.pairs[j]);
                    const float2 f1 = __bfloat1622float2(own1.pairs[j]);
                    acc[2 * j] = f0.x;
                    acc[2 * j + 1] = f0.y;
                    acc[8 + 2 * j] = f1.x;
                    acc[8 + 2 * j + 1] = f1.y;
                }
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    if (q == pair_id) continue;
                    const uint4* p_in = reinterpret_cast<const uint4*>(
                        s_peer_shm[q * 2 + parity] + 131072 + slot * slot_stride_bytes);
                    unpack_accum_fp12_bfp7(ld_cv_uint4(p_in + p), acc);
                }
                if (bfp_mode == 1) {
                    const uint4 rq = pack_fp12_bfp7(acc);
                    st_wt_uint4(my_shm_red + p, rq);
                    st_wt_uint4(pt_hbm_red + (my_hb + (p - my_pb)), rq);
                    // Reconstruct owner's output from `rq` for 0-ULP cross-rank bit identity.
                    unpack_to_bf16_pair(rq, output_vec[idx0], output_vec[idx0 + 1]);
                } else {
                    Pack128Bf16 out0, out1;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        out0.pairs[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
                        out1.pairs[j] =
                            __floats2bfloat162_rn(acc[8 + 2 * j], acc[8 + 2 * j + 1]);
                    }
                    output_vec[idx0] = out0.u4;
                    output_vec[idx0 + 1] = out1.u4;
                    st_wt_uint4(pt_hbm_red + idx0, out0.u4);
                    st_wt_uint4(pt_hbm_red + idx0 + 1, out1.u4);
                    st_wt_uint4(my_shm_red + idx0, out0.u4);
                    st_wt_uint4(my_shm_red + idx0 + 1, out1.u4);
                }
            }
            __syncthreads();
            if (tid == 0) {
                __threadfence_system();
            }
            __syncthreads();
            if (tid < 4) {
                if (tid == pair_id) {
                    sys_store_release_u64(
                        reinterpret_cast<uint64_t*>(pt_hbm + 32768 + b * 64), step);
                    if (!spin_wait_ge_u64(
                            reinterpret_cast<const uint64_t*>(my_hbm + 32768 + b * 64),
                            step,
                            abort_flag)) {
                        s_aborted = 1;
                    }
                } else {
                    const int dst_r = tid * 2 + parity;
                    sys_store_release_u64(
                        reinterpret_cast<uint64_t*>(
                            s_peer_shm[dst_r] + 32768 + (b * 8 + rank) * 64),
                        step);
                    if (!spin_wait_ge_u64(
                            reinterpret_cast<const uint64_t*>(
                                my_shm + 32768 + (b * 8 + dst_r) * 64),
                            step,
                            abort_flag)) {
                        s_aborted = 1;
                    }
                }
            }
            __syncthreads();
            if (s_aborted) {
                signal_ar_cta_done();
                return;
            }

            for (int k = 1; k < 4; ++k) {
                const int q = (pair_id + k) & 3;
                const uint4* o_red = reinterpret_cast<const uint4*>(
                    s_peer_shm[q * 2 + parity] + 131072 + slot * slot_stride_bytes + half_slot);
                if (bfp_mode == 1) {
                    const int64_t qb = my_pb + (my_pairs * q) / 4;
                    const int64_t qe = my_pb + (my_pairs * (q + 1)) / 4;
                    for (int64_t p = qb + tid; p < qe; p += blockDim.x) {
                        const uint4 rq = ld_cv_uint4(o_red + p);
                        st_wt_uint4(pt_hbm_red + (my_hb + (p - my_pb)), rq);
                        unpack_to_bf16_pair(rq, output_vec[p << 1], output_vec[(p << 1) + 1]);
                    }
                } else {
                    const int64_t qb = (my_pb + (my_pairs * q) / 4) << 1;
                    const int64_t qe = (my_pb + (my_pairs * (q + 1)) / 4) << 1;
                    for (int64_t idx = qb + tid; idx < qe; idx += blockDim.x) {
                        const uint4 v = ld_cv_uint4(o_red + idx);
                        output_vec[idx] = v;
                        st_wt_uint4(pt_hbm_red + idx, v);
                    }
                }
            }
            const int64_t pt_pb = pt_hb >> 1;
            const int64_t pt_own_b = (pt_pb + (pt_pairs * pair_id) / 4) << 1;
            const int64_t pt_own_e = (pt_pb + (pt_pairs * (pair_id + 1)) / 4) << 1;
            if (bfp_mode == 1) {
                for (int64_t p = (pt_own_b >> 1) + tid; p < (pt_own_e >> 1); p += blockDim.x) {
                    unpack_to_bf16_pair(
                        ld_cv_uint4(my_hbm_red + (pt_hb + (p - pt_pb))),
                        output_vec[p << 1],
                        output_vec[(p << 1) + 1]);
                }
            } else {
                for (int64_t idx = pt_own_b + tid; idx < pt_own_e; idx += blockDim.x) {
                    output_vec[idx] = ld_cv_uint4(my_hbm_red + idx);
                }
            }
            __syncthreads();
            if (tid == 0) {
                __threadfence_system();
                sys_store_release_u64(
                    reinterpret_cast<uint64_t*>(pt_hbm + 65536 + b * 64), step);
                if (!spin_wait_ge_u64(
                        reinterpret_cast<const uint64_t*>(my_hbm + 65536 + b * 64),
                        step,
                        abort_flag)) {
                    s_aborted = 1;
                }
            }
            __syncthreads();
            if (s_aborted) {
                signal_ar_cta_done();
                return;
            }

            for (int k = 1; k < 4; ++k) {
                const int q = (pair_id + k) & 3;
                const int64_t qb = (pt_pb + (pt_pairs * q) / 4) << 1;
                const int64_t qe = (pt_pb + (pt_pairs * (q + 1)) / 4) << 1;
                if (bfp_mode == 1) {
                    for (int64_t p = (qb >> 1) + tid; p < (qe >> 1); p += blockDim.x) {
                        unpack_to_bf16_pair(
                            ld_cv_uint4(my_hbm_red + (pt_hb + (p - pt_pb))),
                            output_vec[p << 1],
                            output_vec[(p << 1) + 1]);
                    }
                } else {
                    for (int64_t idx = qb + tid; idx < qe; idx += blockDim.x) {
                        output_vec[idx] = ld_cv_uint4(my_hbm_red + idx);
                    }
                }
            }
            __syncthreads();
            if (tid < 8) {
                sys_store_release_u64(
                    reinterpret_cast<uint64_t*>(
                        s_peer_shm[tid] + 65536 + (b * 8 + rank) * 64),
                    step);
            }
            signal_ar_cta_done();
            return;
        }
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
            signal_ar_cta_done();
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
    const bool split_regime = (flag_base > 0 || (my_hbm != nullptr && blocks >= 32 && blocks <= 56));
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
    int world_size,
    int pxb_bfp = 0) {
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
    // CX8 PXB partner-relayed AllGather (`[98368..118848)` dedicated header flags across <=64 CTAs):
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
        if (pxb_bfp == 1 && (num_in_vecs & 1) == 0 && (vecs_per_row & 1) == 0 && num_in_vecs >= 4096) {
            const int64_t np = num_in_vecs >> 1, pb = (np * b) / gridDim.x, pe = (np * (b + 1)) / gridDim.x;
            auto wr = [&](int r, int64_t p, uint4 rq) {
                uint4 o0, o1; unpack_to_bf16_pair(rq, o0, o1);
                write_ag_out(output_vec, outer_rows, vecs_per_row, r, p << 1, o0);
                write_ag_out(output_vec, outer_rows, vecs_per_row, r, (p << 1) + 1, o1);
            };
            for (int64_t p = pb + threadIdx.x; p < pe; p += blockDim.x) {
                const uint4 rq = pack_bf16_pair(input_vec, p << 1);
                st_wt_uint4(my_in + p, rq); st_wt_uint4(pt_buf + p, rq); wr(rank, p, rq);
            }
            __syncthreads(); if (threadIdx.x == 0) __threadfence_system(); __syncthreads();
            if (threadIdx.x == 0) {
                sys_store_release_u64(reinterpret_cast<uint64_t*>(pt_hbm + 102464 + b * 64), step);
                if (!spin_wait_ge_u64(reinterpret_cast<const uint64_t*>(my_hbm + 102464 + b * 64), step, abort_flag)) s_aborted = 1;
            } else if (threadIdx.x <= 3) {
                sys_store_release_u64(reinterpret_cast<uint64_t*>(s_peer_shm[((pair_id + threadIdx.x) & 3) * 2 + parity] + (b * 8 + rank) * 64), step);
            }
            __syncthreads(); if (s_aborted) return;
            for (int64_t p = pb + threadIdx.x; p < pe; p += blockDim.x) wr(pt_r, p, ld_cv_uint4(my_buf + p));
            __syncthreads();
            if (threadIdx.x == 0) {
                sys_store_release_u64(reinterpret_cast<uint64_t*>(pt_hbm + 106560 + b * 64), step);
                if (!spin_wait_ge_u64(reinterpret_cast<const uint64_t*>(my_hbm + 106560 + b * 64), step, abort_flag)) s_aborted = 1;
            } else if (threadIdx.x <= 3) {
                if (!spin_wait_ge_u64(reinterpret_cast<const uint64_t*>(my_shm + (b * 8 + ((pair_id + threadIdx.x) & 3) * 2 + parity) * 64), step, abort_flag)) s_aborted = 1;
            }
            __syncthreads(); if (s_aborted) return;
            const int r1 = ((pair_id + 1) & 3) * 2 + parity, r2 = ((pair_id + 2) & 3) * 2 + parity, r3 = ((pair_id + 3) & 3) * 2 + parity;
            const uint4 *s1 = reinterpret_cast<const uint4*>(s_peer_shm[r1] + 131072 + slot * slot_stride_bytes),
                        *s2 = reinterpret_cast<const uint4*>(s_peer_shm[r2] + 131072 + slot * slot_stride_bytes),
                        *s3 = reinterpret_cast<const uint4*>(s_peer_shm[r3] + 131072 + slot * slot_stride_bytes);
            for (int64_t p = pb + threadIdx.x; p < pe; p += blockDim.x) {
                const uint4 q1 = ld_cv_uint4(s1 + p), q2 = ld_cv_uint4(s2 + p), q3 = ld_cv_uint4(s3 + p);
                st_wt_uint4(pt_buf + p, q1); st_wt_uint4(pt_buf + np + p, q2); st_wt_uint4(pt_buf + 2 * np + p, q3);
                wr(r1, p, q1); wr(r2, p, q2); wr(r3, p, q3);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                __threadfence_system();
                sys_store_release_u64(reinterpret_cast<uint64_t*>(pt_hbm + 110656 + b * 64), step);
                if (!spin_wait_ge_u64(reinterpret_cast<const uint64_t*>(my_hbm + 110656 + b * 64), step, abort_flag)) s_aborted = 1;
            }
            __syncthreads(); if (s_aborted) return;
            for (int64_t p = pb + threadIdx.x; p < pe; p += blockDim.x) {
                wr(r1 ^ 1, p, ld_cv_uint4(my_buf + p)); wr(r2 ^ 1, p, ld_cv_uint4(my_buf + np + p)); wr(r3 ^ 1, p, ld_cv_uint4(my_buf + 2 * np + p));
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                sys_store_release_u64(reinterpret_cast<uint64_t*>(pt_hbm + 114752 + b * 64), step);
                if (!spin_wait_ge_u64(reinterpret_cast<const uint64_t*>(my_hbm + 114752 + b * 64), step, abort_flag)) s_aborted = 1;
            }
            if (threadIdx.x < 8) sys_store_release_u64(reinterpret_cast<uint64_t*>(s_peer_shm[threadIdx.x] + 65536 + (b * 8 + rank) * 64), step);
            return;
        }
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
__global__ __launch_bounds__(640, 1) void sm120_mhc_post_collapse_rmsnorm_kernel(
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
    int post_only,
    int num_tokens = 0,
    int ar_blocks = 0,
    const uint64_t* __restrict__ ar_cta_done = nullptr,
    uint64_t* __restrict__ ar_cta_wait = nullptr) {
    const int tid = threadIdx.x;
    const int total_toks = (num_tokens > 0) ? num_tokens : gridDim.x;
    __shared__ float s_post[4], s_comb[16], s_pre[4], s_warp_sq[20], s_inv_rms;
    __shared__ uint64_t s_tgt;
    int last_ob = -1;
    if (ar_cta_done != nullptr && tid == 0) {
        const uint64_t t = ar_cta_wait[blockIdx.x] + 1ULL;
        ar_cta_wait[blockIdx.x] = t;
        s_tgt = t;
    }
    if (ar_cta_done != nullptr) __syncthreads();
    for (int tok = blockIdx.x; tok < total_toks; tok += gridDim.x) {
    if (ar_cta_done != nullptr) {
        const int ob = static_cast<int>((static_cast<int64_t>(tok + 1) * ar_blocks - 1) / total_toks);
        if (ob != last_ob) {
            if (tid == 0) {
                uint64_t cur = 0ULL;
                for (;;) {
                    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(ar_cta_done + ob) : "memory");
                    if (cur >= s_tgt) break;
                    __nanosleep(64);
                }
            }
            __syncthreads();
            last_ob = ob;
        }
    }
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
    if (!post_only) {
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
    if (ar_cta_done != nullptr) __syncthreads();
    }
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

// SM120 Cluster-8 DSMEM fused `AllReduce + in-place mHC post/comb + collapse + RMSNorm`
// for low-concurrency decode / speculative verify (`rows in {1, 2, 4}`, `H = 5120`).
__global__ __cluster_dims__(8, 1, 1) __launch_bounds__(320, 1) void shm_v5_ar_mhc_cluster8_kernel(
    const uint4* __restrict__ input_vec,
    uint4* __restrict__ res_vec,
    const float* __restrict__ post_mix,
    const float* __restrict__ comb_mix,
    const float* __restrict__ pre_mix,
    const uint4* __restrict__ norm_weight_vec,
    uint4* __restrict__ layer_input_vec,
    uint4* __restrict__ aux_out_vec,
    uint64_t* __restrict__ control,
    char* const* __restrict__ rank_shm,
    char* __restrict__ my_hbm,
    char* __restrict__ pt_hbm,
    int64_t slot_stride_bytes,
    int64_t pxb_slot_stride_bytes,
    int rows,
    float norm_eps,
    int write_aux,
    int rank,
    int flag_base) {
    cg::cluster_group cluster = cg::this_cluster();
    const int b_local = cluster.block_rank();
    const int b = flag_base + b_local;
    const int tid = threadIdx.x;
    const int k_grp = 8 / rows;
    const int tok = b_local / k_grp;
    const int grp_sub = b_local % k_grp;

    __shared__ uint64_t s_step;
    __shared__ int s_aborted_local;
    __shared__ int s_aborted_any;
    __shared__ char* s_peer_shm[kMaxRanks];
    __shared__ float s_post[4];
    __shared__ float s_comb[16];
    __shared__ float s_pre[4];
    __shared__ float s_warp_sq[10];
    __shared__ float s_cta_sq;
    __shared__ float s_inv_rms;

    if (tid < kMaxRanks) {
        s_peer_shm[tid] = rank_shm[tid];
    }
    if (tid == 0) {
        const uint64_t ns = control[b] + 1ULL;
        control[b] = ns;
        s_step = ns;
        s_aborted_local = 0;
        s_aborted_any = 0;
    }
    if (tid < 4) {
        s_post[tid] = post_mix[tok * 4 + tid];
        s_pre[tid] = pre_mix[tok * 4 + tid];
    }
    if (tid < 16) {
        s_comb[tid] = comb_mix[tok * 16 + tid];
    }
    __syncthreads();

    const uint64_t step = s_step;
    const uint64_t ring_depth = (flag_base > 0 || my_hbm != nullptr) ? 2ULL : 4ULL;
    const int slot = (flag_base > 0)
        ? (2 + static_cast<int>((step - 1ULL) & 1))
        : static_cast<int>((step - 1ULL) & 3);
    char* my_shm = s_peer_shm[rank];
    const uint64_t* abort_flag = reinterpret_cast<const uint64_t*>(my_shm + kShmV5AbortOffset);

    if (step > ring_depth) {
        if (tid < kMaxRanks &&
            !spin_wait_ge_u64(
                reinterpret_cast<const uint64_t*>(my_shm + 65536 + (b * 8 + tid) * 64),
                step - ring_depth,
                abort_flag)) {
            s_aborted_local = 1;
        }
        __syncthreads();
    }

    const int64_t num_vecs = static_cast<int64_t>(rows) * 640;
    const int64_t c_begin = (num_vecs * b_local) / 8;
    const int64_t c_end = (num_vecs * (b_local + 1)) / 8;
    const int64_t slice_len = c_end - c_begin;
    const int col = grp_sub * static_cast<int>(slice_len) + tid;

    uint4* my_in = reinterpret_cast<uint4*>(my_shm + 131072 + slot * slot_stride_bytes);
    float collapsed[8] = {0.f};
    float thread_sq = 0.0f;

    if (!s_aborted_local) {
        Pack128Bf16 dx;
        if (my_hbm != nullptr && pt_hbm != nullptr && num_vecs > kPxbOneShotMinVecs) {
            const int parity = rank & 1;
            const int p_slot = static_cast<int>((step - 1ULL) & 1);
            const int64_t c_mid = c_begin + slice_len / 2;
            const int64_t my_hb = (parity == 0) ? c_begin : c_mid;
            const int64_t my_he = (parity == 0) ? c_mid : c_end;
            const int64_t pt_hb = (parity == 0) ? c_mid : c_begin;
            const int64_t pt_he = (parity == 0) ? c_end : c_mid;
            const int64_t cur_pxb_stride = (flag_base > 0) ? 1310720LL : pxb_slot_stride_bytes;
            const int64_t pxb_base_off = 131072LL + (
                (flag_base > 0)
                    ? (2LL * pxb_slot_stride_bytes + p_slot * cur_pxb_stride)
                    : (p_slot * pxb_slot_stride_bytes)
            );
            const int64_t half_pxb = cur_pxb_stride / 2;

            uint4* pt_in = reinterpret_cast<uint4*>(pt_hbm + pxb_base_off);
            if (pt_hb + tid < pt_he) {
                st_wt_uint4(pt_in + pt_hb + tid, input_vec[pt_hb + tid]);
            }
            __syncthreads();
            if (tid == 0) {
                __threadfence_system();
                sys_store_release_u64(reinterpret_cast<uint64_t*>(pt_hbm + b * 64), step);
                if (!spin_wait_ge_u64(
                        reinterpret_cast<const uint64_t*>(my_hbm + b * 64), step, abort_flag)) {
                    s_aborted_local = 1;
                }
            }
            __syncthreads();
            if (!s_aborted_local) {
                const uint4* from_pt = reinterpret_cast<const uint4*>(my_hbm + pxb_base_off);
                uint4* my_hbm_sum = reinterpret_cast<uint4*>(my_hbm + pxb_base_off + half_pxb);
                const int64_t idx_my = my_hb + tid;
                if (idx_my < my_he) {
                    Pack128Bf16 a, bp, psum;
                    a.u4 = input_vec[idx_my];
                    bp.u4 = ld_cv_uint4(from_pt + idx_my);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 fa = __bfloat1622float2(a.pairs[j]);
                        const float2 fb = __bfloat1622float2(bp.pairs[j]);
                        psum.pairs[j] = __floats2bfloat162_rn(fa.x + fb.x, fa.y + fb.y);
                    }
                    my_hbm_sum[idx_my] = psum.u4;
                    st_wt_uint4(my_in + idx_my, psum.u4);
                }
                __syncthreads();
                if (tid == 0) {
                    __threadfence_system();
                }
                __syncthreads();
                if (tid < 8) {
                    sys_store_release_u64(
                        reinterpret_cast<uint64_t*>(s_peer_shm[tid] + (b * 8 + rank) * 64), step);
                    if (!spin_wait_ge_u64(
                            reinterpret_cast<const uint64_t*>(my_shm + (b * 8 + tid) * 64),
                            step,
                            abort_flag)) {
                        s_aborted_local = 1;
                    }
                }
                __syncthreads();
                if (!s_aborted_local && tid < slice_len) {
                    const int64_t idx = c_begin + tid;
                    const int h_par = (idx < c_mid) ? 0 : 1;
                    float acc[8] = {0.f};
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int src_r = q * 2 + h_par;
                        Pack128Bf16 pv;
                        pv.u4 = (src_r == rank)
                            ? my_hbm_sum[idx]
                            : ld_cv_uint4(
                                  reinterpret_cast<const uint4*>(
                                      s_peer_shm[src_r] + 131072 + slot * slot_stride_bytes) +
                                  idx);
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const float2 f = __bfloat1622float2(pv.pairs[j]);
                            acc[2 * j] += f.x;
                            acc[2 * j + 1] += f.y;
                        }
                    }
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        dx.pairs[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
                    }
                }
            }
        } else {
            if (tid < slice_len) {
                st_wt_uint4(my_in + c_begin + tid, input_vec[c_begin + tid]);
            }
            __syncthreads();
            if (tid == 0) {
                __threadfence_system();
            }
            __syncthreads();
            if (tid < kMaxRanks) {
                sys_store_release_u64(
                    reinterpret_cast<uint64_t*>(s_peer_shm[tid] + (b * 8 + rank) * 64), step);
                if (!spin_wait_ge_u64(
                        reinterpret_cast<const uint64_t*>(my_shm + (b * 8 + tid) * 64),
                        step,
                        abort_flag)) {
                    s_aborted_local = 1;
                }
            }
            __syncthreads();
            if (!s_aborted_local && tid < slice_len) {
                const int64_t idx = c_begin + tid;
                float acc[8] = {0.f};
                #pragma unroll
                for (int p = 0; p < kMaxRanks; ++p) {
                    Pack128Bf16 pv;
                    pv.u4 = (p == rank)
                        ? input_vec[idx]
                        : ld_cv_uint4(
                              reinterpret_cast<const uint4*>(
                                  s_peer_shm[p] + 131072 + slot * slot_stride_bytes) +
                              idx);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 f = __bfloat1622float2(pv.pairs[j]);
                        acc[2 * j] += f.x;
                        acc[2 * j + 1] += f.y;
                    }
                }
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    dx.pairs[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
                }
            }
        }

        if (!s_aborted_local && tid < slice_len) {
            const int64_t base_r = static_cast<int64_t>(tok) * 2560 + col;
            const int64_t base_x = static_cast<int64_t>(tok) * 640 + col;
            Pack128Bf16 b0, b1, b2, b3, nr0, nr1, nr2, nr3, aux_pack;
            b0.u4 = res_vec[base_r];
            b1.u4 = res_vec[base_r + 640];
            b2.u4 = res_vec[base_r + 1280];
            b3.u4 = res_vec[base_r + 1920];
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float df = __bfloat162float(dx.elems[e]);
                const float r0 = __bfloat162float(b0.elems[e]);
                const float r1 = __bfloat162float(b1.elems[e]);
                const float r2 = __bfloat162float(b2.elems[e]);
                const float r3 = __bfloat162float(b3.elems[e]);
                const __nv_bfloat16 o0 = __float2bfloat16_rn(
                    s_post[0] * df + s_comb[0] * r0 + s_comb[4] * r1 + s_comb[8] * r2 +
                    s_comb[12] * r3);
                const __nv_bfloat16 o1 = __float2bfloat16_rn(
                    s_post[1] * df + s_comb[1] * r0 + s_comb[5] * r1 + s_comb[9] * r2 +
                    s_comb[13] * r3);
                const __nv_bfloat16 o2 = __float2bfloat16_rn(
                    s_post[2] * df + s_comb[2] * r0 + s_comb[6] * r1 + s_comb[10] * r2 +
                    s_comb[14] * r3);
                const __nv_bfloat16 o3 = __float2bfloat16_rn(
                    s_post[3] * df + s_comb[3] * r0 + s_comb[7] * r1 + s_comb[11] * r2 +
                    s_comb[15] * r3);
                nr0.elems[e] = o0;
                nr1.elems[e] = o1;
                nr2.elems[e] = o2;
                nr3.elems[e] = o3;
                const float x0 = __bfloat162float(o0);
                const float x1 = __bfloat162float(o1);
                const float x2 = __bfloat162float(o2);
                const float x3 = __bfloat162float(o3);
                if (write_aux) {
                    aux_pack.elems[e] = __float2bfloat16_rn((x0 + x1 + x2 + x3) * 0.25f);
                }
                const float val = __bfloat162float(
                    __float2bfloat16_rn(
                        s_pre[0] * x0 + s_pre[1] * x1 + s_pre[2] * x2 + s_pre[3] * x3));
                collapsed[e] = val;
                thread_sq += val * val;
            }
            res_vec[base_r] = nr0.u4;
            res_vec[base_r + 640] = nr1.u4;
            res_vec[base_r + 1280] = nr2.u4;
            res_vec[base_r + 1920] = nr3.u4;
            if (write_aux) {
                aux_out_vec[base_x] = aux_pack.u4;
            }
        }

        if (!s_aborted_local) {
            const float w_sq = warp_reduce_sum(thread_sq);
            const int warp_id = tid >> 5;
            const int lane = tid & 31;
            if (lane == 0) {
                s_warp_sq[warp_id] = w_sq;
            }
            __syncthreads();
            if (warp_id == 0) {
                const float acc = warp_reduce_sum((lane < 10) ? s_warp_sq[lane] : 0.0f);
                if (lane == 0) {
                    s_cta_sq = acc;
                }
            }
            __syncthreads();
        }
    }

    cluster.sync();
    if (tid == 0) {
        int any_ab = 0;
        for (int c = 0; c < 8; ++c) {
            any_ab |= *cluster.map_shared_rank(&s_aborted_local, c);
        }
        s_aborted_any = any_ab;
        if (!any_ab) {
            float row_sq = 0.0f;
            const int grp_base = tok * k_grp;
            for (int c = 0; c < k_grp; ++c) {
                row_sq += *cluster.map_shared_rank(&s_cta_sq, grp_base + c);
            }
            s_inv_rms = rsqrtf(row_sq * (1.0f / 5120.0f) + norm_eps);
        }
    }
    __syncthreads();
    cluster.sync();
    if (s_aborted_any) {
        return;
    }

    if (tid < slice_len) {
        Pack128Bf16 w_pack, out_pack;
        w_pack.u4 = norm_weight_vec[col];
        const float inv_rms = s_inv_rms;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            out_pack.elems[e] =
                __float2bfloat16_rn(collapsed[e] * inv_rms * __bfloat162float(w_pack.elems[e]));
        }
        layer_input_vec[static_cast<int64_t>(tok) * 640 + col] = out_pack.u4;
    }
    __syncthreads();
    if (tid < kMaxRanks) {
        sys_store_release_u64(
            reinterpret_cast<uint64_t*>(s_peer_shm[tid] + 65536 + (b * 8 + rank) * 64), step);
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
    int pxb_bfp,
    std::uintptr_t ar_cta_done_ptr,
    std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const int64_t small_lim = (pxb_bfp == 1) ? 20480 : 40960;
    const bool small_pxb = (ar_cta_done_ptr == 0 && my_hbm_ptr != 0 && blocks >= 32 && blocks <= 56 && num_vecs <= small_lim);
    const int launch_blocks = small_pxb ? 8 : blocks;
    const int flag_base = small_pxb ? blocks : 0;
    const int64_t vecs_per_cta = (num_vecs + launch_blocks - 1) / launch_blocks;
    const int capped_threads = threads > 512 ? 512 : threads;
    const int launch_threads = small_pxb
        ? ((pxb_bfp == 1 && num_vecs > 2560 && num_vecs <= 15360) ? 512 : 256)
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
        flag_base,
        pxb_bfp,
        reinterpret_cast<uint64_t*>(ar_cta_done_ptr));
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
    const bool small_pxb = (my_hbm_ptr != 0 && blocks >= 32 && blocks <= 56 && num_vecs <= 40960);
    const int launch_blocks = small_pxb ? 8 : blocks;
    const int flag_base = small_pxb ? blocks : 0;
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
    int pxb_bfp,
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
        world_size,
        pxb_bfp);
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
    int ar_blocks,
    std::uintptr_t ar_cta_done_ptr,
    std::uintptr_t ar_cta_wait_ptr,
    std::uintptr_t stream_handle) {
    if (num_tokens <= 0) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const int grid = (ar_cta_done_ptr != 0) ? 120 : num_tokens;
    sm120_mhc_post_collapse_rmsnorm_kernel<<<grid, 640, 0, stream>>>(
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
        post_only,
        num_tokens,
        ar_blocks,
        reinterpret_cast<const uint64_t*>(ar_cta_done_ptr),
        reinterpret_cast<uint64_t*>(ar_cta_wait_ptr));
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
    std::uintptr_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    int dev = 0;
    ZSTEP_CUDA_CHECK(cudaGetDevice(&dev));
    static std::atomic<uint64_t> attr_init_mask{0ULL};
    const uint64_t bit = (dev >= 0 && dev < 64) ? (1ULL << dev) : 0ULL;
    if (bit == 0ULL || (attr_init_mask.load(std::memory_order_acquire) & bit) == 0ULL) {
        ZSTEP_CUDA_CHECK(cudaFuncSetAttribute(
            shm_v5_ar_mhc_cluster8_kernel,
            cudaFuncAttributeNonPortableClusterSizeAllowed,
            1));
        if (bit != 0ULL) {
            attr_init_mask.fetch_or(bit, std::memory_order_release);
        }
    }
    shm_v5_ar_mhc_cluster8_kernel<<<8, 320, 0, stream>>>(
        reinterpret_cast<const uint4*>(x_ptr),
        reinterpret_cast<uint4*>(residual_ptr),
        post_mix_ptr,
        comb_mix_ptr,
        pre_mix_ptr,
        reinterpret_cast<const uint4*>(norm_weight_ptr),
        reinterpret_cast<uint4*>(layer_input_ptr),
        reinterpret_cast<uint4*>(aux_out_ptr),
        control_ptr,
        reinterpret_cast<char* const*>(rank_shm_dev_ptrs),
        reinterpret_cast<char*>(my_hbm_ptr),
        reinterpret_cast<char*>(pt_hbm_ptr),
        slot_stride_bytes,
        pxb_slot_stride_bytes,
        rows,
        norm_eps,
        write_aux,
        rank,
        flag_base);
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
