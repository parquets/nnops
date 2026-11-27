#pragma once
/// @file attention_blocked.cuh
/// @brief Register-blocked prefill — the f32 GEMM's FFMA tile shape with the
///        online softmax between its two products.
///
/// Why. The tiled kernel (attention_kernel.cuh) gives each thread one query row
/// and two keys, so every FMA is fed by its own shared read: the block moves
/// about 2.5 KB of shared per 512 FMAs, five times the 1 B/FMA the shared port
/// can carry at full FFMA rate, and it measures 3% of the FFMA peak. This
/// kernel gives a lane 8 rows x 4 keys of the score tile and 8 rows x 4 value
/// columns -- 32 accumulators each, the ratio matmul_f32.cuh uses -- so one
/// shared read feeds 32 FMAs.
///
/// Both products run on the same lane grid: the QK product's key axis and the
/// PV product's value axis are the same 4-wide lane group, so nothing is
/// transposed or shuffled between them beyond the softmax, which writes the
/// probability tile into shared and reads it back as the PV product's A panel.
///
/// Geometry. One block is 64 query rows x 128 keys, 256 threads as 8 warps
/// along the key axis. Lane (lrow, lcol) = (lane & 7, lane >> 3) owns rows
/// lrow + 8r (r < 8) and the key / value-column group warp*16 + lcol*4.
///
/// Eight warps is one block per SM, and two blocks of this shape cannot fit:
/// each needs its own 74 KB of query panel and K ring against a 96 KB budget.
/// A 512-thread form with half the rows per lane was measured and is slower
/// (the nest's shared reads per FMA rise by a third and ptxas runs into the
/// 128-register cap a 512-thread block implies), so the warp count stays here
/// and the nests lean on their 32 accumulators for instruction-level overlap.
///
/// Shared. The query block stays resident (64 x Dqk, padded 4). The key tile
/// arrives in 32-wide slices of Dqk through a two-deep ring, and the
/// probability tile (64 x 128, padded 4) aliases that ring once QK has finished
/// reading it -- which is what sizes the tile at 128 keys rather than wider.
///
/// V is staged through shared too, in slices of `kVs` keys. Read straight from
/// global inside the PV nest, V cost 40% of the kernel: that nest is only 128
/// FMAs deep, so every iteration waited an L2 round trip no resident warp could
/// cover, and the register file had no room to prefetch it. `kVs` is the
/// deepest slice the shared budget holds -- 32 keys (16.9 KB at Dv 128) beside a
/// 128-wide query panel, 8 keys beside a 192-wide one -- and it is a template
/// parameter so the two sizes are separate instantiations rather than a runtime
/// bound in the nest. Barriers measured at ~0.4% of a tile each, so the shallow
/// slice a 192 head forces is the cheaper trade against a global read.
///
/// Grid order. blockIdx.x is the head and blockIdx.y the query block, reversed,
/// so every head's longest-run block is dispatched before any short one. The
/// work per block runs from one KV tile to the run's length; with the query
/// block on the fast axis the last wave would be all short blocks with the long
/// ones still queued, which measured 9% at 512 keys and 19% on the 192-wide MLA
/// rows. A warp whose whole 16-key strip is past the last visible key skips its
/// QK nest outright (half the warps on a diagonal tile).
///
/// Scope. This is the fast path, not the only path: a float or 16-bit cache (no
/// s8 dequantization), no additive bias and no keep mask (both are per-score
/// gathers that would sit inside the nest), no softcap, a head width of at most
/// 192 (the query panel is what costs shared; 192 is the widest the resident
/// panel, the ring and the shallowest V slice fit together), and Dqk, Dv
/// multiples of 4 with Dv at most 128 (32 lane column groups of 4). Everything
/// else, and every decode shape, stays on the tiled and decode kernels.

#include "attention_common.cuh"

namespace nnops::backend::cuda::attn {
namespace blocked {

constexpr int kBlockM = 64;                    ///< query rows one block owns
constexpr int kBlockN = 128;                   ///< keys one tile carries
constexpr int kBlockK = 32;                    ///< key-width slice of one ring stage
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kRowsPerLane = 8;                ///< rows one lane owns
constexpr int kColsPerLane = 4;                ///< keys / value columns one lane owns
constexpr int kKeysPerWarp = 4 * kColsPerLane; ///< four lane columns
/// Keys one staged V slice carries: deepest the shared budget holds, so the
/// widest heads take 8 and the 128s take 32.
constexpr int kVsMax = 32;
constexpr int kVsMid = 16;
constexpr int kVsMin = 8;

static_assert(kWarps * kKeysPerWarp == kBlockN, "the key axis is the warp axis");
static_assert(kBlockM == 8 * kRowsPerLane, "eight lane rows cover the block");
static_assert(kBlockN % 4 == 0 && kBlockK % 4 == 0, "16 B loads");

/// Row strides, each a multiple of 4 so a 16 B read stays a float4 and 4 floats
/// of skew keep a warp's eight lane rows on distinct bank quads.
__host__ __device__ inline int64_t q_stride(int64_t dqk) { return dqk + 4; }
constexpr int64_t kKStride = kBlockK + 4;
constexpr int64_t kPStride = kBlockN + 4;
/// The probability tile has to fit inside the ring it aliases.
static_assert(kBlockM * kPStride <= 2 * kBlockN * kKStride, "P aliases the K ring");

/// Value columns one staged V slice holds, padded four for the 16 B reads and
/// the bank skew.
__host__ __device__ inline int64_t v_stride(int64_t dv) { return dv + 4; }

inline int64_t smem_bytes(int64_t dqk, int64_t dv, int kvs)
{
    const int64_t ring = 2 * static_cast<int64_t>(kBlockN) * kKStride;
    const int64_t vst = static_cast<int64_t>(kvs) * v_stride(dv);
    const int64_t red = 2 * static_cast<int64_t>(kWarps) * kBlockM + 5 * kBlockM;
    return static_cast<int64_t>(sizeof(float)) *
           (static_cast<int64_t>(kBlockM) * q_stride(dqk) + ring + vst + red);
}

/// The two 16-bit elements one 32-bit word holds, as their float pair.
template <class T> __device__ __forceinline__ float2 unpack_pair(uint32_t w)
{
    if constexpr (std::is_same_v<T, __half>) {
        return __half22float2(__halves2half2(__ushort_as_half(static_cast<unsigned short>(w)),
                                             __ushort_as_half(static_cast<unsigned short>(w >> 16))));
    } else {
        return __bfloat1622float2(__halves2bfloat162(
            __ushort_as_bfloat16(static_cast<unsigned short>(w)),
            __ushort_as_bfloat16(static_cast<unsigned short>(w >> 16))));
    }
}

/// `W` consecutive floats of a cache row, from column `d0`. The staged copies
/// are the tile's only global reads, so they take the widest load the row's
/// alignment allows: float4 for an f32 cache, uint4 / uint2 of pairs (with the
/// pair conversions) for a 16-bit one. A row that starts off the bound, or a
/// window that runs past the head width, falls back to the scalar path.
template <class T, int W>
__device__ __forceinline__ void fetchw(const T* row, int d0, int dqk, float (&dst)[W])
{
    constexpr bool is_f32 = std::is_same_v<T, float>;
    constexpr uintptr_t align = is_f32 ? 15u : (W == 4 ? 7u : 15u);
    const bool vec = row != nullptr && d0 + W <= dqk &&
                     (reinterpret_cast<uintptr_t>(row) & align) == 0;
    if (vec) {
        if constexpr (is_f32) {
#pragma unroll
            for (int i = 0; i < W / 4; ++i) {
                const float4 v = reinterpret_cast<const float4*>(row + d0)[i];
                dst[4 * i + 0] = v.x;
                dst[4 * i + 1] = v.y;
                dst[4 * i + 2] = v.z;
                dst[4 * i + 3] = v.w;
            }
        } else if constexpr (W == 16) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const uint4 v = reinterpret_cast<const uint4*>(row + d0)[i];
                const float2 a = unpack_pair<T>(v.x);
                const float2 b = unpack_pair<T>(v.y);
                const float2 c = unpack_pair<T>(v.z);
                const float2 d = unpack_pair<T>(v.w);
                dst[8 * i + 0] = a.x;
                dst[8 * i + 1] = a.y;
                dst[8 * i + 2] = b.x;
                dst[8 * i + 3] = b.y;
                dst[8 * i + 4] = c.x;
                dst[8 * i + 5] = c.y;
                dst[8 * i + 6] = d.x;
                dst[8 * i + 7] = d.y;
            }
        } else if constexpr (W == 8) {
            const uint4 v = reinterpret_cast<const uint4*>(row + d0)[0];
            const float2 a = unpack_pair<T>(v.x);
            const float2 b = unpack_pair<T>(v.y);
            const float2 c = unpack_pair<T>(v.z);
            const float2 d = unpack_pair<T>(v.w);
            dst[0] = a.x;
            dst[1] = a.y;
            dst[2] = b.x;
            dst[3] = b.y;
            dst[4] = c.x;
            dst[5] = c.y;
            dst[6] = d.x;
            dst[7] = d.y;
        } else {
            const uint2 v = reinterpret_cast<const uint2*>(row + d0)[0];
            const float2 a = unpack_pair<T>(v.x);
            const float2 b = unpack_pair<T>(v.y);
            dst[0] = a.x;
            dst[1] = a.y;
            dst[2] = b.x;
            dst[3] = b.y;
        }
    } else {
#pragma unroll
        for (int i = 0; i < W; ++i) {
            const int d = d0 + i;
            dst[i] = (row != nullptr && d < dqk) ? s_load(row + d) : 0.0f;
        }
    }
}

template <class T>
__device__ __forceinline__ void fetch16(const T* row, int d0, int dqk, float (&dst)[16])
{
    fetchw<T, 16>(row, d0, dqk, dst);
}


// ============================================================
// The kernel
// ============================================================

template <class T, int kVs>
__global__ void __launch_bounds__(kThreads) attention_blocked_kernel(AttnParams<T, false> p)
{
    using P = AttnParams<T, false>;
    using KVT = typename P::KVT;

    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int lrow = lane & 7;                    ///< this lane's row; then +8 each
    const int lcol = lane >> 3;                   ///< its 4-wide key / column group

    const int64_t b = blockIdx.z;
    const int64_t h = blockIdx.x;
    const int64_t hkv = h / p.group;
    // Reversed, and the head on the fast axis: block y owns the query rows from
    // the far end, so every head's full-run block is dispatched before any
    // short one. The work per block ranges from 1 KV tile to the run's length,
    // and with the query block on the fast axis the last wave would be all
    // short blocks with the long ones still to come.
    const int64_t i0 =
        static_cast<int64_t>(gridDim.y - 1 - blockIdx.y) * kBlockM;
    const int rows = static_cast<int>(std::min<int64_t>(kBlockM, p.Sq - i0));

    const int64_t dq = q_stride(p.Dqk);
    const int64_t vs = v_stride(p.Dv);
    float* const Qs = smem;                                     // kBlockM x dq
    float* const ring = Qs + static_cast<int64_t>(kBlockM) * dq; // 2 x kBlockN x kKStride
    float* const Vs = ring + 2 * static_cast<int64_t>(kBlockN) * kKStride; // kVs x vs
    float* const mred = Vs + static_cast<int64_t>(kVs) * vs;
    float* const lred = mred + static_cast<int64_t>(kWarps) * kBlockM;
    float* const rM = lred + static_cast<int64_t>(kWarps) * kBlockM;
    float* const rL = rM + kBlockM;
    float* const rS = rL + kBlockM;
    int* const hilo = reinterpret_cast<int*>(rS + kBlockM); ///< per row: hi, lo
    float* const Ps = ring; ///< the probability tile, once QK has read the ring

    // ---- the batch's run and where this block's rows sit ----
    int run = static_cast<int>(p.Sk);
    if (p.run_ptr != nullptr) {
        const int64_t v = p.run_ptr[b] + (p.run_plus_one ? 1 : 0);
        run = static_cast<int>(v <= 0 ? 0 : (v < p.Sk ? v : p.Sk));
    }
    const int64_t pos0 = p.pos_ids != nullptr ? p.pos_ids[b] : p.position_offset;
    const int q0 = static_cast<int>(pos0 >= 0 ? pos0 : (run - p.Sq));
    const int sink_n = (p.sink_tokens > 0 && p.window > 0)
                           ? static_cast<int>(p.sink_tokens < run ? p.sink_tokens : run)
                           : 0;

    if (tid < kBlockM) {
        rM[tid] = kNegInf;
        rL[tid] = 0.0f;
        // The row's causal / window bounds. They live in shared because only
        // the diagonal tile's mask reads them, and a lane that held all sixteen
        // in registers had none left to software-pipeline the nest's loads.
        const int pos = q0 + static_cast<int>(i0) + tid;
        hilo[tid] = p.causal ? std::max(0, std::min(run, pos + 1)) : run;
        hilo[kBlockM + tid] =
            (p.window > 0 && p.causal) ? std::max(0, pos - static_cast<int>(p.window) + 1) : 0;
    }

    // ---- the query block, staged once, pre-scaled ----
    // The softmax scale rides on Q: a key the mask leaves visible then needs no
    // per-score multiply at all. Lanes 0-7 copy seven different rows at one
    // column, so a warp's stores land on eight distinct bank quads (the 132
    // float row stride steps four quads a row) instead of four lanes a quad.
    // The four pad floats of the row are zeroed so a ragged last key chunk (a
    // head width the 32-wide slices do not divide) multiplies zeros, not
    // whatever the panel last held.
    {
        constexpr int kQPieces = kThreads / kBlockM;
        // scale * log2(e): a score then exponentiates with one MUFU.EX2 and no
        // multiply, and the running max, the sum and the sink all live in the
        // same log2 units.
        const float qscale = p.scale * 1.4426950408889634f;
        const int qrow = tid & (kBlockM - 1);
        const int nd4 = static_cast<int>(p.Dqk >> 2);
        const int64_t qbase = b * p.q_b + h * p.q_h;
        const T* const src =
            qrow < rows ? p.q + qbase + (i0 + qrow) * p.q_s : nullptr;
        float v[16];
        for (int d0 = (tid / kBlockM) * 16; d0 < nd4 * 4; d0 += 16 * kQPieces) {
            fetch16<T>(src, d0, static_cast<int>(p.Dqk), v);
#pragma unroll
            for (int i = 0; i < 16; i += 4) {
                if (d0 + i < nd4 * 4) {
                    *reinterpret_cast<float4*>(
                        &Qs[static_cast<int64_t>(qrow) * dq + d0 + i]) =
                        make_float4(v[i] * qscale, v[i + 1] * qscale, v[i + 2] * qscale,
                                    v[i + 3] * qscale);
                }
            }
        }
        if (tid < kBlockM) {
            *reinterpret_cast<float4*>(&Qs[static_cast<int64_t>(tid) * dq + nd4 * 4]) =
                make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    // ---- each lane row's causal / window bounds, fixed for the block ----
    // Only the extremes stay in registers: a whole 4-key strip tests against
    // them once per chunk, and a strip inside every row's window needs no mask
    // and no scale, which is every tile but the diagonal one. `live_all` folds
    // the ragged last block's dead rows into the same test.
    int min_hi = run, max_lo = 0;
    bool live_all = true;
#pragma unroll
    for (int r = 0; r < kRowsPerLane; ++r) {
        const int row = lrow + 8 * r;
        const int pos = q0 + static_cast<int>(i0) + row;
        const int hi = p.causal ? std::max(0, std::min(run, pos + 1)) : run;
        const int lo = (p.window > 0 && p.causal)
                           ? std::max(0, pos - static_cast<int>(p.window) + 1)
                           : 0;
        min_hi = std::min(min_hi, hi);
        max_lo = std::max(max_lo, lo);
        live_all = live_all && (row < rows);
    }

    // The staging slot: thread `tid` copies 16 floats of key (tid >> 1) at
    // column offset 16 * (tid & 1) -- one 32-wide slice exactly, from the first
    // eight warps.
    const int st_key = tid >> 1;
    const int st_d = (tid & 1) * 16;
    float* const st_dst = ring + static_cast<int64_t>(st_key) * kKStride + st_d;

    const int key0 = warp * kKeysPerWarp + lcol * kColsPerLane;
    const bool has_col = key0 < p.Dv; ///< V columns this lane owns are in range

    float o[kRowsPerLane][kColsPerLane];
#pragma unroll
    for (int r = 0; r < kRowsPerLane; ++r) {
#pragma unroll
        for (int c = 0; c < kColsPerLane; ++c) {
            o[r][c] = 0.0f;
        }
    }

    const int kv_end =
        p.causal ? std::max(0, std::min(run, q0 + static_cast<int>(i0) + rows)) : run;
    const int nch = static_cast<int>((p.Dqk + kBlockK - 1) / kBlockK);

    for (int j0 = 0; j0 < kv_end; j0 += kBlockN) {
        const KVT* const st_row = k_row(p, b, hkv, j0 + st_key);

        float acc[kRowsPerLane][kColsPerLane];
#pragma unroll
        for (int r = 0; r < kRowsPerLane; ++r) {
#pragma unroll
            for (int c = 0; c < kColsPerLane; ++c) {
                acc[r][c] = 0.0f;
            }
        }

        // ---- Q.K over the head width, one 32-wide slice at a time ----
        // Chunk c is read from ring[c & 1] while chunk c+1's global loads are
        // already in flight, so the ring needs no cp.async to stay busy.
        float nxt[16];
        fetch16<T>(st_row, st_d, static_cast<int>(p.Dqk), nxt);
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            *reinterpret_cast<float4*>(st_dst + i) =
                make_float4(nxt[i], nxt[i + 1], nxt[i + 2], nxt[i + 3]);
        }
        __syncthreads();

        for (int c = 0; c < nch; ++c) {
            // The slice is always walked 32 wide: a head width the slices do not
            // divide reads zeros there (the ring is zero-filled past Dqk), which
            // keeps the nest's trip count a constant ptxas can unroll and
            // software-pipeline instead of a runtime bound it must leave rolled.
            const int kbase = c * kBlockK;
            const float* const Ks =
                ring + static_cast<int64_t>(c & 1) * kBlockN * kKStride;

            // chunk c+1's rows: issued before the nest so their latency lands
            // under it, stored after.
            if (c + 1 < nch) {
                fetch16<T>(st_row, (c + 1) * kBlockK + st_d, static_cast<int>(p.Dqk), nxt);
            }

            // A warp whose whole 16-key group sits past the last visible key
            // owns an all-masked column strip: its scores stay zero and the
            // epilogue masks them to -inf, so the nest is skipped outright. On
            // the diagonal tile that is half the warps and a quarter of the
            // block's QK work.
            if (j0 + key0 < kv_end) {
#pragma unroll
            for (int ks = 0; ks < kBlockK; ks += 4) {
                const int kv = kbase + ks;
                float ak[kRowsPerLane][4];
                float bk[4][kColsPerLane];
#pragma unroll
                for (int r = 0; r < kRowsPerLane; ++r) {
                    const float4 v = *reinterpret_cast<const float4*>(
                        &Qs[static_cast<int64_t>(lrow + 8 * r) * dq + kv]);
                    ak[r][0] = v.x;
                    ak[r][1] = v.y;
                    ak[r][2] = v.z;
                    ak[r][3] = v.w;
                }
#pragma unroll
                for (int cc = 0; cc < kColsPerLane; ++cc) {
                    const float4 v = *reinterpret_cast<const float4*>(
                        &Ks[static_cast<int64_t>(key0 + cc) * kKStride + ks]);
                    bk[0][cc] = v.x;
                    bk[1][cc] = v.y;
                    bk[2][cc] = v.z;
                    bk[3][cc] = v.w;
                }
                // t outermost: an accumulator is written once per t, so the
                // thirty-two of them sit between two updates of any one chain.
#pragma unroll
                for (int t = 0; t < 4; ++t) {
#pragma unroll
                    for (int r = 0; r < kRowsPerLane; ++r) {
                        const float a = ak[r][t];
#pragma unroll
                        for (int cc = 0; cc < kColsPerLane; ++cc) {
                            acc[r][cc] = fmaf(a, bk[t][cc], acc[r][cc]);
                        }
                    }
                }
            }
            }

            if (c + 1 < nch) {
                float* const dst =
                    ring + static_cast<int64_t>((c + 1) & 1) * kBlockN * kKStride +
                    static_cast<int64_t>(st_key) * kKStride + st_d;
#pragma unroll
                for (int i = 0; i < 16; i += 4) {
                    *reinterpret_cast<float4*>(dst + i) =
                        make_float4(nxt[i], nxt[i + 1], nxt[i + 2], nxt[i + 3]);
                }
            }
            __syncthreads();
        }

        // ---- the score tile: the visible-key test, online softmax ----
        // Q carries the scale, so a strip whose four keys sit inside every one
        // of the lane's rows needs nothing here. The diagonal tile takes the
        // per-score path: the masked scores go to the online softmax's identity,
        // -inf, which the exp turns into zero.
        {
            const int jb = static_cast<int>(j0) + key0;
            if (!(live_all && jb >= max_lo && jb + kColsPerLane - 1 < min_hi)) {
#pragma unroll
                for (int r = 0; r < kRowsPerLane; ++r) {
                    const int row = lrow + 8 * r;
                    const bool live = row < rows;
                    const int hi = hilo[row];
                    const int lo = hilo[kBlockM + row];
#pragma unroll
                    for (int c = 0; c < kColsPerLane; ++c) {
                        const int j = jb + c;
                        const bool vis = (j < hi) && (j >= lo || j < sink_n);
                        acc[r][c] = (live && vis) ? acc[r][c] : kNegInf;
                    }
                }
            }
        }

        // Row max: this lane's four keys, then the four lane columns, then the
        // key octants. -FLT_MAX is the identity, so a wholly masked row keeps
        // the sentinel.
        {
            float m4[kRowsPerLane];
#pragma unroll
            for (int r = 0; r < kRowsPerLane; ++r) {
                m4[r] = fmaxf(fmaxf(acc[r][0], acc[r][1]), fmaxf(acc[r][2], acc[r][3]));
                m4[r] = fmaxf(m4[r], __shfl_xor_sync(0xffffffffu, m4[r], 8));
                m4[r] = fmaxf(m4[r], __shfl_xor_sync(0xffffffffu, m4[r], 16));
            }
            if (lcol == 0) {
#pragma unroll
                for (int r = 0; r < kRowsPerLane; ++r) {
                    mred[warp * kBlockM + lrow + 8 * r] = m4[r];
                }
            }
            __syncthreads();
            if (tid < kBlockM) {
                float mt = mred[tid];
#pragma unroll
                for (int w = 1; w < kWarps; ++w) {
                    mt = fmaxf(mt, mred[w * kBlockM + tid]);
                }
                const float mo = rM[tid];
                const float mn = fmaxf(mo, mt);
                rS[tid] = (mo == kNegInf) ? 0.0f : exp2f(mo - mn);
                rM[tid] = mn;
            }
            __syncthreads();
        }

        // p = exp(s - m_new), and the row sum, in place.
        {
            float s4[kRowsPerLane];
#pragma unroll
            for (int r = 0; r < kRowsPerLane; ++r) {
                s4[r] = 0.0f;
                const float mn = rM[lrow + 8 * r];
                const bool ok = mn != kNegInf;
#pragma unroll
                for (int c = 0; c < kColsPerLane; ++c) {
                    const float pv = ok ? exp2f(acc[r][c] - mn) : 0.0f;
                    acc[r][c] = pv;
                    s4[r] += pv;
                }
                s4[r] += __shfl_xor_sync(0xffffffffu, s4[r], 8);
                s4[r] += __shfl_xor_sync(0xffffffffu, s4[r], 16);
            }
            if (lcol == 0) {
#pragma unroll
                for (int r = 0; r < kRowsPerLane; ++r) {
                    lred[warp * kBlockM + lrow + 8 * r] = s4[r];
                }
            }
            __syncthreads();
            if (tid < kBlockM) {
                float st = lred[tid];
#pragma unroll
                for (int w = 1; w < kWarps; ++w) {
                    st += lred[w * kBlockM + tid];
                }
                rL[tid] = rL[tid] * rS[tid] + st;
            }
        }

        // The ring is free: write the probability tile into it and rescale the
        // running output by this row's max step.
#pragma unroll
        for (int r = 0; r < kRowsPerLane; ++r) {
            const int row = lrow + 8 * r;
            const float sc = rS[row];
            float4 pv;
            pv.x = acc[r][0];
            pv.y = acc[r][1];
            pv.z = acc[r][2];
            pv.w = acc[r][3];
            *reinterpret_cast<float4*>(
                &Ps[static_cast<int64_t>(row) * kPStride + key0]) = pv;
#pragma unroll
            for (int c = 0; c < kColsPerLane; ++c) {
                o[r][c] *= sc;
            }
        }
        __syncthreads();

        // ---- P.V, one 4-key unit at a time, V staged in kVs-key slices ----
        // Lanes 0-7 copy seven different staged keys at one column, so a warp's
        // stores take four wavefronts -- the least a full warp of 16 B stores
        // can. A thread copies kVF floats of its key, and the nest reads a
        // lane's four value columns as one 16 B vector per key. The next slice
        // is fetched under this one's nest, so the copy costs no latency --
        // only the two barriers a single buffer needs, which is why the slice
        // is as deep as the budget allows.
        {
            // As many floats a thread as the thread count divides the slice by,
            // so every thread copies once: 64 B a piece at the deep slice (two
            // whole sectors in global), 16 B at the shallow one.
            constexpr int kVF = (128 * kVs) / kThreads;
            const int vsk = tid & (kVs - 1);
            const int vcol = (tid / kVs) * kVF;
            const bool vcopies = vcol < p.Dv;
            float vrow[kVF];
            {
                const KVT* const sr = v_row(p, b, hkv, j0 + vsk);
                fetchw<T, kVF>(sr, vcol, static_cast<int>(p.Dv), vrow);
            }
            for (int s0 = 0; s0 < kBlockN; s0 += kVs) {
                const bool more = s0 + kVs < kBlockN;
                if (vcopies) {
                    float* const vdst = &Vs[static_cast<int64_t>(vsk) * vs + vcol];
#pragma unroll
                    for (int i = 0; i < kVF; i += 4) {
                        if (vcol + i < p.Dv) {
                            *reinterpret_cast<float4*>(vdst + i) =
                                make_float4(vrow[i], vrow[i + 1], vrow[i + 2], vrow[i + 3]);
                        }
                    }
                }
                __syncthreads();
                if (more) {
                    const KVT* const sr = v_row(p, b, hkv, j0 + s0 + kVs + vsk);
                    fetchw<T, kVF>(sr, vcol, static_cast<int>(p.Dv), vrow);
                }
#pragma unroll
                for (int k0 = 0; k0 < kVs; k0 += 4) {
                    float pa[kRowsPerLane][4];
                    float vb[4][kColsPerLane];
#pragma unroll
                    for (int r = 0; r < kRowsPerLane; ++r) {
                        const float4 v = *reinterpret_cast<const float4*>(
                            &Ps[(static_cast<int64_t>(lrow + 8 * r)) * kPStride + s0 + k0]);
                        pa[r][0] = v.x;
                        pa[r][1] = v.y;
                        pa[r][2] = v.z;
                        pa[r][3] = v.w;
                    }
                    if (has_col) {
#pragma unroll
                        for (int t = 0; t < 4; ++t) {
                            const float4 v = *reinterpret_cast<const float4*>(
                                &Vs[static_cast<int64_t>(k0 + t) * vs + key0]);
                            vb[t][0] = v.x;
                            vb[t][1] = v.y;
                            vb[t][2] = v.z;
                            vb[t][3] = v.w;
                        }
                    } else {
#pragma unroll
                        for (int t = 0; t < 4; ++t) {
#pragma unroll
                            for (int c = 0; c < kColsPerLane; ++c) {
                                vb[t][c] = 0.0f;
                            }
                        }
                    }
#pragma unroll
                    for (int t = 0; t < 4; ++t) {
#pragma unroll
                        for (int r = 0; r < kRowsPerLane; ++r) {
                            const float a = pa[r][t];
#pragma unroll
                            for (int c = 0; c < kColsPerLane; ++c) {
                                o[r][c] = fmaf(a, vb[t][c], o[r][c]);
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
        // The last slice's barrier already holds the next tile's K staging off
        // the ring that P aliases.
    }

    // ---- head sink and normalization ----
    if (tid < kBlockM) {
        if (p.sink != nullptr) {
            const float mo = rM[tid];
            const float sv = sink_at(p, h) * 1.4426950408889634f;
            const float mn = fmaxf(mo, sv);
            const float sc = (mo == kNegInf) ? 0.0f : exp2f(mo - mn);
            rL[tid] = rL[tid] * sc + exp2f(sv - mn);
            rS[tid] = sc;
            rM[tid] = mn;
        } else {
            rS[tid] = 1.0f;
        }
    }
    __syncthreads();

    if (has_col) {
        const int64_t obase = b * p.o_b + h * p.o_h;
#pragma unroll
        for (int r = 0; r < kRowsPerLane; ++r) {
            const int row = lrow + 8 * r;
            if (row >= rows) {
                break;
            }
            const float l = rL[row];
            const float norm = l > 0.0f ? (1.0f / l) : 0.0f;
            const float scl = rS[row];
            const int64_t qrow = i0 + row;
            T* const dst = p.o + obase + qrow * p.o_s;
            const T* const grow =
                p.has_gate ? p.gate + b * p.g_b + h * p.g_h + qrow * p.g_s : nullptr;
#pragma unroll
            for (int c = 0; c < kColsPerLane; ++c) {
                float val = o[r][c] * scl * norm;
                if (grow != nullptr) {
                    val *= gate_act_value(p.gate_act, s_load(grow + key0 + c));
                }
                s_store(dst + key0 + c, val);
            }
        }
    }
}

} // namespace blocked
} // namespace nnops::backend::cuda::attn
