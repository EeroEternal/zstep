// Copyright (c) 2026 zstep Contributors
// SPDX-License-Identifier: MIT
//
// SM120 Lossless Page-Split (`DSv41-584` & `Latent576`) + 32-Way Warp-Interleaved
// 12-bit Static rANS + IEEE 802.3 CRC32 KV-cache codec (`0 ULP` bit-exact roundtrip).

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

using U = uint32_t;
using B = uint8_t;

static constexpr int RANS_LANES = 32;
static constexpr int RANS_PROB_BITS = 12;
static constexpr int RANS_PROB_SCALE = 1 << RANS_PROB_BITS; // 4096
static constexpr U RANS_L = 1u << 16;
static constexpr U RAW_FLAG = 1u << 31;
static constexpr int RANS_HDR_RESERVE = 640;
static constexpr int RANS_LANE_CAP_U16 = 1280; // 2560 B/lane (> 74752 / 32 = 2336 B)
static constexpr int RANS_LANE_CAP_BYTES = RANS_LANE_CAP_U16 * 2;
static constexpr int RANS_SLOT_STRIDE = RANS_HDR_RESERVE + RANS_LANES * RANS_LANE_CAP_BYTES; // 82560 B

static __device__ __forceinline__ uint16_t load_u16_le(const B* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

static __device__ __forceinline__ void store_u16_le(B* p, uint16_t v) {
    p[0] = static_cast<B>(v & 0xFFu);
    p[1] = static_cast<B>(v >> 8);
}

static __device__ __forceinline__ void store_u32_le(B* p, U v) {
    p[0] = static_cast<B>(v & 0xFFu);
    p[1] = static_cast<B>((v >> 8) & 0xFFu);
    p[2] = static_cast<B>((v >> 16) & 0xFFu);
    p[3] = static_cast<B>((v >> 24) & 0xFFu);
}

static __device__ __forceinline__ void prep_div_u32(U d, U& m_out, U& sh_out) {
    if (d <= 1u) {
        m_out = 0u;
        sh_out = 0u;
        return;
    }
    U l = 32u - static_cast<U>(__clz(d - 1u));
    uint64_t num = ((1ull << l) - static_cast<uint64_t>(d)) << 32;
    m_out = static_cast<U>(num / static_cast<uint64_t>(d)) + 1u;
    sh_out = l;
}

static __device__ __forceinline__ U div_u32_fast(U x, U m, U sh) {
    if (sh == 0u) return x;
    U t = __umulhi(x, m);
    return (t + ((x - t) >> 1)) >> (sh - 1u);
}

static __device__ __forceinline__ U crc32_byte_entry(U b) {
    U c = b;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        U mask = 0u - (c & 1u);
        asm("" : "+r"(mask));
        c = (c >> 1) ^ (0xEDB88320u & mask);
    }
    return c;
}

static __device__ __forceinline__ U gf2_matrix_times(const U* mat, U vec) {
    U sum = 0u;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        if ((vec >> i) & 1u) sum ^= mat[i];
    }
    return sum;
}

static __device__ __forceinline__ U crc32_combine_dev(
    const U gf2_pow[][32], int levels, U crc1, U crc2, uint64_t len2
) {
    int bit = 0;
    while (len2 && bit < levels) {
        if (len2 & 1ull) crc1 = gf2_matrix_times(gf2_pow[bit], crc1);
        len2 >>= 1;
        bit++;
    }
    return crc1 ^ crc2;
}

static __device__ __forceinline__ U compute_cta_crc32(const B* data, size_t len) {
    int tid = threadIdx.x; // 0..255
    __shared__ U crc_tab[256];
    __shared__ U gf2_pow[18][32];
    __shared__ U sh_crc[256];
    __shared__ uint64_t sh_len[256];

    crc_tab[tid] = crc32_byte_entry(static_cast<U>(tid));
    if (tid < 32) {
        gf2_pow[0][tid] = (tid < 8) ? crc32_byte_entry(1u << tid) : (1u << (tid - 8));
        __syncwarp();
        for (int k = 1; k < 18; ++k) {
            gf2_pow[k][tid] = gf2_matrix_times(gf2_pow[k - 1], gf2_pow[k - 1][tid]);
            __syncwarp();
        }
    }
    __syncthreads();

    size_t t_start = (len * static_cast<size_t>(tid)) / 256u;
    size_t t_end = (len * static_cast<size_t>(tid + 1)) / 256u;
    U c = 0xFFFFFFFFu;
    for (size_t i = t_start; i < t_end; ++i) {
        c = crc_tab[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    }
    sh_crc[tid] = c ^ 0xFFFFFFFFu;
    sh_len[tid] = static_cast<uint64_t>(t_end - t_start);
    __syncthreads();

    for (int stride = 1; stride < 256; stride <<= 1) {
        if ((tid & (2 * stride - 1)) == 0) {
            sh_crc[tid] = crc32_combine_dev(
                gf2_pow, 18, sh_crc[tid], sh_crc[tid + stride], sh_len[tid + stride]
            );
            sh_len[tid] += sh_len[tid + stride];
        }
        __syncthreads();
    }
    return sh_crc[0];
}

// Separates compound KV pages into homogeneous planes:
// - row_stride == 584 (`DSv41` `fp8_ds_mla`): 512B FP8 latent | 32B BF16 RoPE exp | 32B BF16 RoPE mant | 8B scale
// - row_stride == 576 (`Latent576`): 512B FP8 latent | 32B BF16 RoPE exp | 32B BF16 RoPE mant
// - any other even stride: 2-byte BF16 high/low byte-lane shuffle
__global__ void kv_page_split_kernel(
    B* dst_base,
    const B* src_base,
    int rows_per_page,
    int row_stride,
    int num_pages,
    int decode
) {
    int p = blockIdx.x;
    if (p >= num_pages) return;
    size_t page_bytes = static_cast<size_t>(rows_per_page) * static_cast<size_t>(row_stride);
    B* dst = dst_base + static_cast<size_t>(p) * page_bytes;
    const B* src = src_base + static_cast<size_t>(p) * page_bytes;

    if (row_stride == 584 || row_stride == 576) {
        int scale_bytes = row_stride - 576;
        size_t fp8_plane = static_cast<size_t>(rows_per_page) * 512u;
        size_t rope_half = static_cast<size_t>(rows_per_page) * 32u;
        size_t scale_base = fp8_plane + 2u * rope_half;
        for (size_t idx = threadIdx.x; idx < page_bytes; idx += blockDim.x) {
            size_t r = idx / static_cast<size_t>(row_stride);
            size_t c = idx % static_cast<size_t>(row_stride);
            size_t split_pos;
            if (c < 512u) {
                split_pos = r * 512u + c;
            } else if (scale_bytes > 0 && c < 512u + static_cast<size_t>(scale_bytes)) {
                split_pos = scale_base + r * static_cast<size_t>(scale_bytes) + (c - 512u);
            } else {
                size_t rc = c - (512u + static_cast<size_t>(scale_bytes));
                size_t bf_idx = r * 32u + (rc >> 1);
                split_pos = ((rc & 1u) == 1u) ? (fp8_plane + bf_idx) : (fp8_plane + rope_half + bf_idx);
            }
            if (!decode) dst[split_pos] = src[idx];
            else dst[idx] = src[split_pos];
        }
    } else if ((page_bytes & 1u) == 0u) {
        size_t half = page_bytes >> 1;
        for (size_t idx = threadIdx.x; idx < page_bytes; idx += blockDim.x) {
            size_t elem = idx >> 1;
            size_t split_pos = ((idx & 1u) == 1u) ? elem : (half + elem);
            if (!decode) dst[split_pos] = src[idx];
            else dst[idx] = src[split_pos];
        }
    } else {
        for (size_t idx = threadIdx.x; idx < page_bytes; idx += blockDim.x) {
            dst[idx] = src[idx];
        }
    }
}

__global__ void kv_rans32_encode_kernel(
    const B* src_base,
    size_t chunk_bytes,
    B* slot_base,
    int32_t* comp_lens,
    U* crc32_out,
    int num_chunks
) {
    int p = blockIdx.x;
    if (p >= num_chunks) return;
    int tid = threadIdx.x;
    const B* src = src_base + static_cast<size_t>(p) * chunk_bytes;
    B* slot = slot_base + static_cast<size_t>(p) * static_cast<size_t>(RANS_SLOT_STRIDE);

    U crc = compute_cta_crc32(src, chunk_bytes);
    if (tid == 0) crc32_out[p] = crc;
    __syncthreads();

    __shared__ union {
        U hist4[4][256];
        uint16_t wbuf[RANS_LANES][64];
        B lane_tmp[4][RANS_LANE_CAP_BYTES];
    } sh;
    __shared__ U freq_sh[256];
    __shared__ U cdf_sh[256];
    __shared__ U rcp_m_sh[256];
    __shared__ U rcp_sh_sh[256];
    __shared__ U warp_tmp_sh[8];
    __shared__ U warp_max_h_sh[8];
    __shared__ int warp_max_s_sh[8];
    __shared__ B hdr_sh[RANS_HDR_RESERVE];
    __shared__ U states_sh[RANS_LANES];
    __shared__ uint16_t lane_words_sh[RANS_LANES];
    __shared__ U lane_dst_offs_sh[RANS_LANES];
    __shared__ int active_sh, max_sym_sh, overflow_sh, total_len_sh;

    sh.hist4[0][tid] = sh.hist4[1][tid] = sh.hist4[2][tid] = sh.hist4[3][tid] = 0u;
    __syncthreads();
    int bank = tid & 3;
    for (size_t i = tid; i < chunk_bytes; i += 256) {
        atomicAdd(&sh.hist4[bank][src[i]], 1u);
    }
    __syncthreads();

    U h = sh.hist4[0][tid] + sh.hist4[1][tid] + sh.hist4[2][tid] + sh.hist4[3][tid];
    int wid = tid >> 5, lane = tid & 31;
    unsigned lane_lt_mask = (1u << lane) - 1u;
    unsigned nz_mask = __ballot_sync(0xFFFFFFFFu, h > 0u);
    U mh = h;
    int ms = tid;
    for (int off = 16; off > 0; off >>= 1) {
        U oh = __shfl_down_sync(0xFFFFFFFFu, mh, off);
        int os = __shfl_down_sync(0xFFFFFFFFu, ms, off);
        if (oh > mh || (oh == mh && os < ms)) {
            mh = oh;
            ms = os;
        }
    }
    if (lane == 0) {
        warp_tmp_sh[wid] = static_cast<U>(__popc(nz_mask));
        warp_max_h_sh[wid] = mh;
        warp_max_s_sh[wid] = ms;
    }
    __syncthreads();
    if (tid == 0) {
        int active = 0;
        U best_h = 0u;
        int best_s = 0;
        for (int w = 0; w < 8; ++w) {
            active += static_cast<int>(warp_tmp_sh[w]);
            if (warp_max_h_sh[w] > best_h || (warp_max_h_sh[w] == best_h && warp_max_s_sh[w] < best_s)) {
                best_h = warp_max_h_sh[w];
                best_s = warp_max_s_sh[w];
            }
        }
        active_sh = active;
        max_sym_sh = best_s;
    }
    __syncthreads();
    if (active_sh == 1 && chunk_bytes > 4) {
        if (tid == 0) {
            slot[0] = 1u;
            slot[1] = static_cast<B>(max_sym_sh);
            comp_lens[p] = 2;
        }
        return;
    }
    if (active_sh <= 1 || chunk_bytes <= 256) {
        for (size_t i = tid; i < chunk_bytes; i += 256) slot[i] = src[i];
        if (tid == 0) comp_lens[p] = static_cast<int32_t>(RAW_FLAG | static_cast<U>(chunk_bytes));
        return;
    }

    U f = 0u, rem = 0u;
    if (h > 0u) {
        uint64_t budget = static_cast<uint64_t>(RANS_PROB_SCALE - active_sh);
        uint64_t prod = static_cast<uint64_t>(h) * budget;
        f = 1u + static_cast<U>(prod / static_cast<uint64_t>(chunk_bytes));
        rem = static_cast<U>(prod % static_cast<uint64_t>(chunk_bytes));
    }
    U wf = f;
    for (int off = 16; off > 0; off >>= 1) wf += __shfl_down_sync(0xFFFFFFFFu, wf, off);
    if (lane == 0) warp_tmp_sh[wid] = wf;
    __syncthreads();
    U init_sum = 0u;
    for (int w = 0; w < 8; ++w) init_sum += warp_tmp_sh[w];
    __syncthreads();
    U left = static_cast<U>(RANS_PROB_SCALE) - init_sum;
    if (left > 0u) {
        bool elig = (h > 0u && f < 4095u);
        U lo = 0u, hi = static_cast<U>(chunk_bytes);
        while (lo < hi) {
            U mid = (lo + hi) >> 1;
            unsigned m = __ballot_sync(0xFFFFFFFFu, elig && rem >= mid);
            if (lane == 0) warp_tmp_sh[wid] = static_cast<U>(__popc(m));
            __syncthreads();
            U cnt = 0u;
            for (int w = 0; w < 8; ++w) cnt += warp_tmp_sh[w];
            __syncthreads();
            if (cnt > left) lo = mid + 1u;
            else hi = mid;
        }
        unsigned ge_m = __ballot_sync(0xFFFFFFFFu, elig && rem >= lo);
        if (lane == 0) warp_tmp_sh[wid] = static_cast<U>(__popc(ge_m));
        __syncthreads();
        U ge_base = 0u, ge_total = 0u;
        for (int w = 0; w < 8; ++w) {
            if (w < wid) ge_base += warp_tmp_sh[w];
            ge_total += warp_tmp_sh[w];
        }
        __syncthreads();
        if (elig && rem >= lo && (ge_base + static_cast<U>(__popc(ge_m & lane_lt_mask))) < left) f++;
        left = (ge_total >= left) ? 0u : (left - ge_total);
        if (left > 0u && lo > 0u) {
            unsigned eq_m = __ballot_sync(0xFFFFFFFFu, elig && rem == (lo - 1u));
            if (lane == 0) warp_tmp_sh[wid] = static_cast<U>(__popc(eq_m));
            __syncthreads();
            U eq_base = 0u, eq_total = 0u;
            for (int w = 0; w < 8; ++w) {
                if (w < wid) eq_base += warp_tmp_sh[w];
                eq_total += warp_tmp_sh[w];
            }
            __syncthreads();
            if (elig && rem == (lo - 1u) && (eq_base + static_cast<U>(__popc(eq_m & lane_lt_mask))) < left) f++;
            left = (eq_total >= left) ? 0u : (left - eq_total);
        }
        if (left > 0u && tid == max_sym_sh) f += left;
    }
    freq_sh[tid] = f;
    U scan = f;
    for (int off = 1; off < 32; off <<= 1) {
        U prev = __shfl_up_sync(0xFFFFFFFFu, scan, off);
        if (lane >= off) scan += prev;
    }
    if (lane == 31) warp_tmp_sh[wid] = scan;
    __syncthreads();
    U cdf_base = 0u;
    for (int w = 0; w < wid; ++w) cdf_base += warp_tmp_sh[w];
    __syncthreads();
    cdf_sh[tid] = cdf_base + scan - f;
    if (f > 0u) prep_div_u32(f, rcp_m_sh[tid], rcp_sh_sh[tid]);
    else {
        rcp_m_sh[tid] = 0u;
        rcp_sh_sh[tid] = 0u;
    }

    if (tid < 128) {
        U a = freq_sh[tid * 2], b = freq_sh[tid * 2 + 1];
        int pos = 1 + tid * 3;
        hdr_sh[pos] = static_cast<B>(a & 0xFFu);
        hdr_sh[pos + 1] = static_cast<B>(((a >> 8) & 0x0Fu) | ((b & 0x0Fu) << 4));
        hdr_sh[pos + 2] = static_cast<B>((b >> 4) & 0xFFu);
    }
    if (tid == 0) hdr_sh[0] = 3u;
    __syncthreads();

    if (tid < RANS_LANES) {
        int l = tid;
        int cnt = static_cast<int>((chunk_bytes - 1u - static_cast<size_t>(l)) / 32u + 1u);
        uint16_t* lane_end_u16 = reinterpret_cast<uint16_t*>(
            slot + RANS_HDR_RESERVE + static_cast<size_t>(l + 1) * RANS_LANE_CAP_BYTES
        );
        U x = RANS_L;
        int nwords = 0, bcnt = 0, ovf = 0;
        for (int k = cnt - 1; k >= 0; --k) {
            B s = src[static_cast<size_t>(k) * 32u + static_cast<size_t>(l)];
            U fs = freq_sh[s], cs = cdf_sh[s];
            if (x >= (fs << 20)) {
                if (nwords >= RANS_LANE_CAP_U16) {
                    ovf = 1;
                    break;
                }
                sh.wbuf[l][63 - bcnt] = static_cast<uint16_t>(x & 0xFFFFu);
                bcnt++;
                nwords++;
                if (bcnt == 64) {
                    uint16_t* dw = lane_end_u16 - nwords;
                    for (int j = 0; j < 64; ++j) dw[j] = sh.wbuf[l][j];
                    bcnt = 0;
                }
                x >>= 16;
            }
            U q = div_u32_fast(x, rcp_m_sh[s], rcp_sh_sh[s]);
            x = (q << RANS_PROB_BITS) + (x - q * fs) + cs;
        }
        if (!ovf && bcnt > 0) {
            uint16_t* dw = lane_end_u16 - nwords;
            for (int j = 0; j < bcnt; ++j) dw[j] = sh.wbuf[l][64 - bcnt + j];
        }
        states_sh[l] = x;
        lane_words_sh[l] = static_cast<uint16_t>(nwords);
        unsigned any_ovf = __any_sync(0xFFFFFFFFu, ovf != 0);
        if (l == 0) overflow_sh = any_ovf ? 1 : 0;
    }
    __syncthreads();
    const int tbl_bytes = 385;
    if (tid < RANS_LANES) {
        int l = tid;
        U len_b = static_cast<U>(lane_words_sh[l]) * 2u, scan_b = len_b;
        for (int off = 1; off < 32; off <<= 1) {
            U prev = __shfl_up_sync(0xFFFFFFFFu, scan_b, off);
            if (l >= off) scan_b += prev;
        }
        lane_dst_offs_sh[l] = scan_b - len_b;
        store_u32_le(hdr_sh + tbl_bytes + l * 4, states_sh[l]);
        store_u16_le(hdr_sh + tbl_bytes + RANS_LANES * 4 + l * 2, lane_words_sh[l]);
        if (l == 31) {
            int tot = tbl_bytes + RANS_LANES * 6 + static_cast<int>(scan_b);
            total_len_sh = (overflow_sh || static_cast<size_t>(tot) >= chunk_bytes) ? -1 : tot;
        }
    }
    __syncthreads();
    if (total_len_sh < 0) {
        for (size_t i = tid; i < chunk_bytes; i += 256) slot[i] = src[i];
        if (tid == 0) comp_lens[p] = static_cast<int32_t>(RAW_FLAG | static_cast<U>(chunk_bytes));
        return;
    }
    int hdr_len = tbl_bytes + RANS_LANES * 6; // 577 B
    for (int i = tid; i < hdr_len; i += 256) slot[i] = hdr_sh[i];
    int sub_lane = tid & 63, grp_lane = tid >> 6;
    for (int g = 0; g < 8; ++g) {
        int l = g * 4 + grp_lane;
        int len_l = static_cast<int>(lane_words_sh[l]) * 2;
        if (len_l > 0) {
            const B* src_l = slot + RANS_HDR_RESERVE + static_cast<size_t>(l + 1) * RANS_LANE_CAP_BYTES - len_l;
            for (int i = sub_lane; i < len_l; i += 64) sh.lane_tmp[grp_lane][i] = src_l[i];
        }
        __syncthreads();
        if (len_l > 0) {
            B* dst_l = slot + hdr_len + lane_dst_offs_sh[l];
            for (int i = sub_lane; i < len_l; i += 64) dst_l[i] = sh.lane_tmp[grp_lane][i];
        }
        __syncthreads();
    }
    if (tid == 0) comp_lens[p] = total_len_sh;
}

__global__ void kv_rans32_decode_kernel(
    const B* slot_base,
    const int32_t* comp_lens,
    const U* expected_crc32,
    B* dst_base,
    size_t chunk_bytes,
    int num_chunks,
    int32_t* err_flag
) {
    int p = blockIdx.x;
    if (p >= num_chunks) return;
    int tid = threadIdx.x;
    const B* in = slot_base + static_cast<size_t>(p) * static_cast<size_t>(RANS_SLOT_STRIDE);
    B* dst = dst_base + static_cast<size_t>(p) * chunk_bytes;
    U word = static_cast<U>(comp_lens[p]);

    if ((word & RAW_FLAG) != 0u) {
        if ((word & ~RAW_FLAG) != static_cast<U>(chunk_bytes)) {
            if (tid == 0) atomicOr(err_flag, 1);
            return;
        }
        for (size_t i = tid; i < chunk_bytes; i += 256) dst[i] = in[i];
    } else if (word == 2u && in[0] == 1u) {
        B sym = in[1];
        for (size_t i = tid; i < chunk_bytes; i += 256) dst[i] = sym;
    } else {
        if (word < 577u || in[0] != 3u) {
            if (tid == 0) atomicOr(err_flag, 1);
            return;
        }
        __shared__ U freq_sh[256];
        __shared__ U cdf_sh[256];
        __shared__ U lut_sh[RANS_PROB_SCALE];
        __shared__ U states_sh[RANS_LANES];
        __shared__ uint16_t lane_words_sh[RANS_LANES];
        __shared__ U lane_offs_sh[RANS_LANES];
        __shared__ int bad_hdr_sh;

        if (tid < 128) {
            const B* fp = in + 1;
            size_t bi = static_cast<size_t>(tid) * 3u;
            freq_sh[tid * 2] = static_cast<U>(fp[bi]) | ((static_cast<U>(fp[bi + 1]) & 0x0Fu) << 8);
            freq_sh[tid * 2 + 1] = (static_cast<U>(fp[bi + 1]) >> 4) | (static_cast<U>(fp[bi + 2]) << 4);
        }
        __syncthreads();
        if (tid == 0) {
            bad_hdr_sh = 0;
            U sum = 0u;
            for (int s = 0; s < 256; ++s) {
                cdf_sh[s] = sum;
                sum += freq_sh[s];
            }
            if (sum != static_cast<U>(RANS_PROB_SCALE)) bad_hdr_sh = 1;
            size_t pos = 385;
            for (int l = 0; l < RANS_LANES; ++l) {
                U st = static_cast<U>(in[pos]) | (static_cast<U>(in[pos + 1]) << 8)
                     | (static_cast<U>(in[pos + 2]) << 16) | (static_cast<U>(in[pos + 3]) << 24);
                if (st < RANS_L) bad_hdr_sh = 1;
                states_sh[l] = st;
                pos += 4;
            }
            U off = 0u;
            for (int l = 0; l < RANS_LANES; ++l) {
                uint16_t nw = load_u16_le(in + pos);
                lane_words_sh[l] = nw;
                lane_offs_sh[l] = off;
                off += static_cast<U>(nw) * 2u;
                pos += 2;
            }
            if (577u + off != word) bad_hdr_sh = 1;
            if (bad_hdr_sh) atomicOr(err_flag, 1);
        }
        __syncthreads();
        if (bad_hdr_sh) return;

        U f = freq_sh[tid];
        if (f > 0u) {
            U c = cdf_sh[tid];
            U entry = static_cast<U>(tid) | (f << 8) | (c << 20);
            for (U i = 0; i < f; ++i) lut_sh[c + i] = entry;
        }
        __syncthreads();
        if (tid < RANS_LANES) {
            int l = tid;
            int cnt = static_cast<int>((chunk_bytes - 1u - static_cast<size_t>(l)) / 32u + 1u);
            U x = states_sh[l];
            int nw = static_cast<int>(lane_words_sh[l]);
            const B* lp = in + 577 + lane_offs_sh[l];
            int wpos = 0, bad = 0;
            for (int k = 0; k < cnt; ++k) {
                U s_slot = x & 4095u;
                U entry = lut_sh[s_slot];
                dst[static_cast<size_t>(k) * 32u + static_cast<size_t>(l)] = static_cast<B>(entry & 0xFFu);
                U fs = (entry >> 8) & 0xFFFu;
                U cs = entry >> 20;
                x = fs * (x >> RANS_PROB_BITS) + (s_slot - cs);
                if (x < RANS_L) {
                    if (wpos >= nw) {
                        bad = 1;
                        break;
                    }
                    x = (x << 16) | static_cast<U>(load_u16_le(lp + static_cast<size_t>(wpos) * 2u));
                    wpos++;
                }
            }
            if (bad || x != RANS_L || wpos != nw) atomicOr(err_flag, 2);
        }
    }
    __syncthreads();
    U actual_crc = compute_cta_crc32(dst, chunk_bytes);
    if (tid == 0 && actual_crc != expected_crc32[p]) {
        atomicOr(err_flag, 4);
    }
}

int64_t kv_rans_slot_stride_cuda() {
    return static_cast<int64_t>(RANS_SLOT_STRIDE);
}

void launch_kv_page_split_cuda(
    void* dst_ptr,
    const void* src_ptr,
    int rows_per_page,
    int row_stride,
    int num_pages,
    int decode,
    std::uintptr_t stream_handle
) {
    if (num_pages <= 0) return;
    kv_page_split_kernel<<<num_pages, 256, 0, reinterpret_cast<cudaStream_t>(stream_handle)>>>(
        static_cast<B*>(dst_ptr),
        static_cast<const B*>(src_ptr),
        rows_per_page,
        row_stride,
        num_pages,
        decode
    );
}

void launch_kv_rans32_encode_cuda(
    const void* src_ptr,
    int64_t chunk_bytes,
    void* slots_ptr,
    int32_t* comp_lens_ptr,
    uint32_t* crc32_ptr,
    int num_chunks,
    std::uintptr_t stream_handle
) {
    if (num_chunks <= 0) return;
    kv_rans32_encode_kernel<<<num_chunks, 256, 0, reinterpret_cast<cudaStream_t>(stream_handle)>>>(
        static_cast<const B*>(src_ptr),
        static_cast<size_t>(chunk_bytes),
        static_cast<B*>(slots_ptr),
        comp_lens_ptr,
        crc32_ptr,
        num_chunks
    );
}

void launch_kv_rans32_decode_cuda(
    const void* slots_ptr,
    const int32_t* comp_lens_ptr,
    const uint32_t* expected_crc32_ptr,
    void* dst_ptr,
    int64_t chunk_bytes,
    int num_chunks,
    int32_t* err_flag_ptr,
    std::uintptr_t stream_handle
) {
    if (num_chunks <= 0) return;
    kv_rans32_decode_kernel<<<num_chunks, 256, 0, reinterpret_cast<cudaStream_t>(stream_handle)>>>(
        static_cast<const B*>(slots_ptr),
        comp_lens_ptr,
        expected_crc32_ptr,
        static_cast<B*>(dst_ptr),
        static_cast<size_t>(chunk_bytes),
        num_chunks,
        err_flag_ptr
    );
}
