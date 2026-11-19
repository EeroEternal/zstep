
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
    constexpr int kNumWarps = kNumExperts / 32;
    __shared__ int s_counts[kNumExperts];
    __shared__ int s_warp_tiles[kNumWarps];
    __shared__ int s_tile_offsets[kNumExperts + 1];
    __shared__ int s_write_cursor[kNumExperts];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    s_counts[tid] = 0;
    __syncthreads();

    for (int64_t i = tid; i < num_pairs; i += kNumExperts) {
        const int e = topk_ids[i];
        if (static_cast<unsigned>(e) < kNumExperts) {
            atomicAdd(&s_counts[e], 1);
        }
    }
    __syncthreads();

    // Block-wide exclusive scan of per-expert tile counts (one expert per thread).
    const int tiles = (s_counts[tid] + kBlockM - 1) / kBlockM;
    int incl = tiles;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int up = __shfl_up_sync(0xffffffffu, incl, off);
        if (lane >= off) incl += up;
    }
    if (lane == 31) s_warp_tiles[warp] = incl;
    __syncthreads();
    if (warp == 0) {
        int w = lane < kNumWarps ? s_warp_tiles[lane] : 0;
#pragma unroll
        for (int off = 1; off < kNumWarps; off <<= 1) {
            const int up = __shfl_up_sync(0xffffffffu, w, off);
            if (lane >= off) w += up;
        }
        if (lane < kNumWarps) s_warp_tiles[lane] = w;
    }
    __syncthreads();
    const int excl = incl - tiles + (warp > 0 ? s_warp_tiles[warp - 1] : 0);
    s_tile_offsets[tid] = excl;
    if (tid == kNumExperts - 1) {
        s_tile_offsets[kNumExperts] = excl + tiles;
        *num_valid_tiles_out = excl + tiles;
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
    __nv_bfloat16* __restrict__ cached_conv_state,
    const int32_t* __restrict__ restore_slot_ids,
    const int32_t* __restrict__ save_slot_ids,
    int64_t stride_in_row,
    int64_t stride_state_slot,
    int64_t stride_cached_conv_slot,
    int num_cached_slots) {
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
    const int restore_slot =
        (restore_slot_ids != nullptr && cached_conv_state != nullptr)
            ? restore_slot_ids[seq_idx]
            : -1;
    if (restore_slot >= 0 && restore_slot < num_cached_slots) {
        const __nv_bfloat16* cst =
            cached_conv_state + static_cast<int64_t>(restore_slot) * stride_cached_conv_slot + c * 3;
        s0 = __bfloat162float(cst[0]);
        s1 = __bfloat162float(cst[1]);
        s2 = __bfloat162float(cst[2]);
    } else if (start_pos != 0) {
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
        if (save_slot_ids != nullptr && cached_conv_state != nullptr) {
            const int32_t save_slot = save_slot_ids[row];
            if (save_slot >= 0 && save_slot < num_cached_slots) {
                __nv_bfloat16* cst =
                    cached_conv_state + static_cast<int64_t>(save_slot) * stride_cached_conv_slot + c * 3;
                cst[0] = __float2bfloat16_rn(s0);
                cst[1] = __float2bfloat16_rn(s1);
                cst[2] = __float2bfloat16_rn(s2);
            }
        }
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
    float* __restrict__ cached_ssm_state_vk,
    const int32_t* __restrict__ restore_slot_ids,
    const int32_t* __restrict__ save_slot_ids,
    int64_t stride_qkvzba_row,
    int64_t stride_ssm_slot,
    int64_t stride_ssm_head,
    int64_t stride_cached_ssm_slot,
    int64_t stride_cached_ssm_head,
    int num_cached_slots) {
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

    const int restore_slot =
        (restore_slot_ids != nullptr && cached_ssm_state_vk != nullptr)
            ? restore_slot_ids[seq_idx]
            : -1;
    float4 st[4];
    if (restore_slot >= 0 && restore_slot < num_cached_slots) {
        const float4* cached_base4 = reinterpret_cast<const float4*>(
            cached_ssm_state_vk
            + static_cast<int64_t>(restore_slot) * stride_cached_ssm_slot
            + v_head * stride_cached_ssm_head);
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            st[r] = cached_base4[(v_base + r) * 32 + lane];
        }
    } else {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            st[r] = (start_pos == 0)
                ? make_float4(0.f, 0.f, 0.f, 0.f)
                : state_base4[(v_base + r) * 32 + lane];
        }
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

        if (save_slot_ids != nullptr && cached_ssm_state_vk != nullptr) {
            const int32_t save_slot = save_slot_ids[row];
            if (save_slot >= 0 && save_slot < num_cached_slots) {
                float4* cached_base4 = reinterpret_cast<float4*>(
                    cached_ssm_state_vk
                    + static_cast<int64_t>(save_slot) * stride_cached_ssm_slot
                    + v_head * stride_cached_ssm_head);
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    cached_base4[(v_base + r) * 32 + lane] = st[r];
                }
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
// previous decode step instead of the host-provided one. Padding rows (`slot >= active_slots`)
// receive token 0 rather than reading or racing on `dummy_slot`.
__global__ void decode_select_tokens_kernel(
    int32_t* __restrict__ tokens,
    const int32_t* __restrict__ last_tokens,
    const int32_t* __restrict__ req_slot_ids,
    const int32_t* __restrict__ mode,
    int64_t num_rows,
    int64_t active_slots) {
    if (*mode == 0) {
        return;
    }
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < num_rows) {
        const int32_t slot = req_slot_ids[i];
        tokens[i] = (slot >= 0 && slot < active_slots) ? last_tokens[slot] : 0;
    }
}

__global__ void decode_publish_tokens_kernel(
    int32_t* __restrict__ last_tokens,
    const int32_t* __restrict__ req_slot_ids,
    const int32_t* __restrict__ top2_tokens,
    int64_t num_rows,
    int64_t active_slots) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < num_rows) {
        const int32_t slot = req_slot_ids[i];
        const int32_t tok = top2_tokens[i * 2];
        if (slot >= 0 && slot < active_slots && tok >= 0) {
            last_tokens[slot] = tok;
        }
    }
}

}  // namespace

void launch_decode_select_tokens_cuda(
    int32_t* tokens,
    const int32_t* last_tokens,
    const int32_t* req_slot_ids,
    const int32_t* mode,
    int64_t num_rows,
    int64_t active_slots,
    std::uintptr_t stream_handle) {
    if (num_rows == 0) {
        return;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const unsigned int blocks = static_cast<unsigned int>((num_rows + 127) / 128);
    decode_select_tokens_kernel<<<blocks, 128, 0, stream>>>(
        tokens, last_tokens, req_slot_ids, mode, num_rows, active_slots);
    ZSTEP_CUDA_CHECK(cudaGetLastError());
}

void launch_decode_publish_tokens_cuda(
    int32_t* last_tokens,
    const int32_t* req_slot_ids,
    const int32_t* top2_tokens,
    int64_t num_rows,
    int64_t active_slots,
    std::uintptr_t stream_handle) {
    if (num_rows == 0) {
        return;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const unsigned int blocks = static_cast<unsigned int>((num_rows + 127) / 128);
    decode_publish_tokens_kernel<<<blocks, 128, 0, stream>>>(
        last_tokens, req_slot_ids, top2_tokens, num_rows, active_slots);
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
        reinterpret_cast<__nv_bfloat16*>(cached_conv_state_ptr),
        restore_slot_ids_ptr,
        save_slot_ids_ptr,
        stride_qkvzba_row,
        stride_conv_slot,
        stride_cached_conv_slot,
        num_cached_slots);

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
        cached_ssm_state_vk_ptr,
        restore_slot_ids_ptr,
        save_slot_ids_ptr,
        stride_qkvzba_row,
        stride_ssm_slot,
        stride_ssm_head,
        stride_cached_ssm_slot,
        stride_cached_ssm_head,
        num_cached_slots);

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
