#pragma once
/// @file attention_mla.cuh
/// @brief MultiLatentAttention's absorbed form as its own tile — FlashMLA's
///        structure on the Ampere m16n8k16 path.
///
/// Why a separate tile. MLA's absorbed form has a shape the general attention
/// tiles are the wrong fit for: the key row *is* the value row (v_head_dim
/// leading columns of the latent), the key width is the DeepSeek 576 (512
/// latent + 64 rope), the value width is the 512 latent, and the query heads
/// vastly outnumber the key heads (one latent row serves every head, 128:1 in
/// DeepSeek). The m16n8k16 tile stops at a value width of 128 because one warp
/// holds the whole per-row accumulator; here the value width is split over the
/// block's warps and the query axis is the head group, which is FlashMLA's
/// answer to the same shape and the one thing that makes the 512-wide value fit
/// at all.
///
/// Geometry. One block is sixteen query heads of one KV head at one query
/// position — the sixteen rows of one m16n8k16 tile — against kBlockN = 32 keys.
/// Sixteen warps:
///
///   Q.K     warp (g, q) owns keys [8g, 8g + 8) over depth quarter q of the
///           head width, and the quarters are summed by the softmax pass
///           through four score buffers. Four 8-key groups is what a 32-key
///           tile cuts into, so the sixteen warps are 4 groups x 4 quarters
///           with no warp idle and a nine-step mma chain apiece.
///   softmax warp w owns row w with its 32 lanes on the tile's keys, exactly
///           the tiled kernel's row pass, and leaves the probabilities in
///           shared as 16-bit A fragments.
///   P.V     warp w owns value columns [32w, 32w + 32) for all sixteen rows —
///           four accumulator tiles, 16 registers a lane.
///
/// Because the value is the key row's prefix, the key tile the Q.K warps read
/// is also the value panel the P.V warps read: 512 of the 576 columns are one
/// staging, one read of DRAM per tile — FlashMLA's central saving, and the
/// reason a 512-wide value fits a 99 KB shared slab. The query block itself is
/// staged into the K ring's first stage and read into A fragments at once, so
/// it costs no shared memory of its own. That is what pays for the warps: at
/// one CTA a multiprocessor they are the only occupancy there is, and a warp's
/// Q.K chain is nine mma instead of eighteen with twice as many warps to
/// overlap it. Measured, the same tile at eight warps read 0.547 ms on the
/// 512-row prefill and 0.139 on the 4096-token decode at 128 heads; at sixteen
/// they read 0.44-0.51 and 0.123-0.129 (the bench drifts ~15% row to row).
///
/// Where the time goes, per ncu on this card: DRAM 6-9% and an L2 hit rate of
/// 94%, so the latent is not the cost; SM throughput 48% with 0.55 eligible
/// warps a scheduler, i.e. latency, not bandwidth. The stalls are `wait` (the
/// mma chain, 2.5 warps an issue), `barrier` (the three a tile, 1.9),
/// `math_pipe_throttle` (address arithmetic, 1.9 — the staging loop's divide
/// and remainder were the largest single customer and are now a shift) and
/// `mio_throttle` (ldmatrix/st.shared, 1.5). Two accumulator chains for Q.K
/// measured 4-5% *slower* at 114 registers; sixteen warps already interleave
/// the chain. The next real step is a producer/consumer warp split, which is
/// what FlashMLA does and what this stall mix asks for.
///
/// Decode. The grid is (heads / 16, query rows, batch) and the whole run is
/// streamed per block, so a decode step (one query row) leaves the machine with
/// heads/16 blocks. The run is therefore cut into `MlaSplits` contiguous spans,
/// each block keeping its own online-softmax state, and a second pass merges
/// them — the same split the GQA decode kernel makes, for the same reason.
///
/// Scope. The 16-bit absorbed form only (the mma is 16-bit): no additive bias,
/// keep mask or softcap, a value width a multiple of 64 and no wider than the
/// key width, and a head group of at least sixteen. One instantiation, the
/// DeepSeek 576 x 512: a narrow latent (192 x 128) is the m16n8k16 tile's own
/// ground, where it wins by 2x, and everything else stays on that tile or the
/// tiled kernel.

#include "attention_common.cuh"
#include "backend/cuda/matmul/ptx.h"

namespace nnops::backend::cuda::attn {
namespace mla {

constexpr int kRows = 16;    ///< query heads one block serves (one m16 tile)
constexpr int kBlockN = 32;  ///< keys one tile carries
constexpr int kWarps = 16;
constexpr int kThreads = kWarps * 32;
constexpr int kKeyGroups = 4;                       ///< 8-key groups in a tile
constexpr int kDepthSplits = kWarps / kKeyGroups;   ///< head width split per group
static_assert(kBlockN == 32, "one lane a key in the softmax pass");
static_assert(kKeyGroups * kDepthSplits == kWarps, "every warp has a Q.K role");
static_assert(kBlockN % 16 == 0, "whole 16-key P.V steps");
static_assert(kRows == kWarps, "one softmax row a warp");

constexpr int kPStride = kBlockN + 8; ///< probability panel: 16 rows x 32 keys
constexpr int kSStride = kBlockN + 1; ///< score buffer: odd, so rows scatter banks

/// Row stride of a 16-bit query/key panel. Eight halfs of skew put the eight
/// rows one ldmatrix reads at 16 B offsets of 16 mod 128 bytes (a 576-wide row
/// is 1152 B, a multiple of every bank period, so without it every row lands on
/// the same banks).
__host__ __device__ inline constexpr int panel_stride(int64_t d)
{
    return static_cast<int>(d) + 8;
}

/// The query panel is *not* counted: it is staged into the K ring's first
/// stage, read into the A fragments, and the ring takes the region back — the
/// m16n8k16 tile's trick, and it is what buys the sixteen warps.
inline int64_t smem_bytes(int64_t dqk)
{
    const int64_t qs = panel_stride(dqk);
    return static_cast<int64_t>(sizeof(half)) *
               (2 * static_cast<int64_t>(kBlockN) * qs + static_cast<int64_t>(kRows) * kPStride) +
           static_cast<int64_t>(sizeof(float)) *
               (kDepthSplits * static_cast<int64_t>(kRows) * kSStride + 2 * kRows);
}

/// The per-split partial state. `m`/`l` are one float per (row, split), `o` is
/// DV floats per (row, split); all null when the call fits in one span.
struct MlaSplits {
    int splits = 1;
    float* m = nullptr;
    float* l = nullptr;
    float* o = nullptr;
};

/// One block's rows stage a panel: eight halfs a step, async when the run row
/// is 16 B aligned and straight copies otherwise, zeros past the run. `Rows`
/// is a template argument so the thread→(row, chunk) map is a shift and a mask
/// instead of the divide-and-remainder the flat odometer needed — the staging
/// loop was the ALU pipe's largest customer (27% of active cycles, ncu).
template <int Rows, class P, int DQK, class SrcRow>
__device__ __forceinline__ void stage_rows(SrcRow src_row, half* panel, int tid)
{
    constexpr int kChunks = DQK / 8;
    constexpr int kStride = DQK + 8;
    constexpr int kTPR = kThreads / Rows; ///< threads a row
    constexpr int kIters = (kChunks + kTPR - 1) / kTPR;
    static_assert(kThreads % Rows == 0 && (kTPR & (kTPR - 1)) == 0, "shift a row, not a divide");
    const int r = tid / kTPR;
    const int c0 = tid - r * kTPR;
    const typename P::KVT* const src = src_row(r);
    half* const dst = panel + static_cast<int64_t>(r) * kStride;
#pragma unroll
    for (int i = 0; i < kIters; ++i) {
        const int c = c0 + i * kTPR;
        if (c >= kChunks) {
            break;
        }
        if (src == nullptr) {
            *reinterpret_cast<uint4*>(dst + c * 8) = make_uint4(0, 0, 0, 0);
        } else if ((reinterpret_cast<uintptr_t>(src) & 15u) == 0) {
            // .cg: the tile is read once and consumed from shared, so leaving it
            // out of L1 keeps the cache for what actually re-reads.
            cp_async_cg(dst + c * 8, reinterpret_cast<const uint4*>(src) + c);
        } else {
            for (int j = 0; j < 8; ++j) {
                dst[c * 8 + j] = static_cast<half>(src[c * 8 + j]);
            }
        }
    }
}

template <class T, int DQK, int DV>
__global__ void __launch_bounds__(kThreads, 1) attention_mla_kernel(AttnParams<T, false> p,
                                                                   MlaSplits s)
{
    using P = AttnParams<T, false>;
    constexpr int kqs = DQK + 8;
    constexpr int kDepth = DQK / kDepthSplits; ///< head width one warp's group walks
    constexpr int kSteps = kDepth / 16;
    constexpr int kWt = DV / kWarps / 8; ///< 8-column accumulator tiles per warp
    constexpr int kCols = DV / kWarps;   ///< value columns per warp
    static_assert(DQK % (kDepthSplits * 16) == 0, "whole 16-deep steps a warp");
    static_assert(DV % (kWarps * 16) == 0, "an even number of 8-column tiles a warp");

    extern __shared__ half smem[];
    half* const Kb = smem;
    half* const Pt = Kb + 2 * static_cast<int64_t>(kBlockN) * kqs;
    float* const Sb = reinterpret_cast<float*>(Pt + static_cast<int64_t>(kRows) * kPStride);
    float* const rowS = Sb + static_cast<int64_t>(kDepthSplits) * kRows * kSStride;
    float* const rowN = rowS + kRows;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = warp & (kKeyGroups - 1); ///< keys [8g, 8g + 8) of the tile
    const int dq = warp / kKeyGroups;      ///< this warp's depth slice
    float* const Sc = Sb + static_cast<int64_t>(dq) * kRows * kSStride;

    const int64_t b = blockIdx.z / s.splits;
    const int sid = s.splits > 1 ? static_cast<int>(blockIdx.z % s.splits) : 0;
    const int64_t i = blockIdx.y;
    const int64_t hb = static_cast<int64_t>(blockIdx.x) * kRows;
    const int64_t hkv = hb / p.group;

    // The run and where this query row sits in it (vector > scalar >
    // bottom-right). Every row of the block is the same query position, so the
    // visible range is one pair of bounds for the whole tile.
    int64_t run = p.Sk;
    if (p.run_ptr != nullptr) {
        const int64_t v = p.run_ptr[b];
        run = p.run_plus_one ? v + 1 : v;
        run = run <= 0 ? 0 : (run < p.Sk ? run : p.Sk);
    }
    const int64_t pos0 = p.pos_ids != nullptr ? p.pos_ids[b] : p.position_offset;
    const int64_t q0 = pos0 >= 0 ? pos0 : (run - p.Sq);
    const int64_t pos = q0 + i;
    const int64_t hi = p.causal ? std::max<int64_t>(0, std::min<int64_t>(run, pos + 1)) : run;
    const int64_t lo = (p.window > 0 && p.causal) ? std::max<int64_t>(0, pos - p.window + 1) : 0;
    const int sink_n = (p.sink_tokens > 0 && p.window > 0)
                           ? static_cast<int>(std::min<int64_t>(p.sink_tokens, run))
                           : 0;

    // This block's key span: its split's slice of the run, then the sliding
    // window's skip of a fully masked prefix.
    int64_t jbeg = 0;
    int64_t jend = hi;
    if (s.splits > 1) {
        const int64_t span =
            ((p.Sk + s.splits - 1) / s.splits + kBlockN - 1) / kBlockN * kBlockN;
        jbeg = std::min<int64_t>(sid * span, hi);
        jend = std::min<int64_t>(hi, (sid + 1) * span);
    }
    if (p.causal && p.window > 0 && sink_n == 0 && lo > kBlockN) {
        jbeg = std::max(jbeg, ((lo - 1) / kBlockN) * kBlockN);
    }

    // The query panel is staged into the K ring's first stage and read into the
    // A fragments at once; the ring then takes the region back for its first
    // key tile. The panel therefore costs no shared memory of its own — the
    // m16n8k16 tile's trick, and what pays for sixteen warps — and one ldmatrix
    // a step, once, replaces the per-tile re-read the panel used to take.
    stage_rows<kRows, P, DQK>(
        [&](int r) { return p.q + b * p.q_b + (hb + r) * p.q_h + i * p.q_s; }, Kb, tid);
    cp_async_commit_group();
    cp_async_wait_group<0>();
    __syncthreads();

    // The A operand of Q.K: sixteen rows of the panel, 16 B a row group.
    const uint32_t q_off =
        smem_addr_u32(Kb) + static_cast<uint32_t>((lane & 7) + ((lane >> 3) & 1) * 8) *
                                static_cast<uint32_t>(kqs) * 2u +
        static_cast<uint32_t>(lane >> 4) * 16u;
    uint32_t qf[kSteps][4];
    {
        const uint32_t qb = q_off + static_cast<uint32_t>(dq * kDepth) * 2u;
#pragma unroll
        for (int ks = 0; ks < kSteps; ++ks) {
            ldmatrix_x4_at(&qf[ks][0], &qf[ks][1], &qf[ks][2], &qf[ks][3],
                           qb + static_cast<uint32_t>(ks * 32));
        }
    }
    __syncthreads();

    if (jbeg < jend) {
        stage_rows<kBlockN, P, DQK>([&](int r) { return k_row(p, b, hkv, jbeg + r); }, Kb, tid);
    }
    cp_async_commit_group();

    // The B operand of Q.K: eight keys a warp, the two 8-half column halves.
    const uint32_t k_off =
        smem_addr_u32(Kb) +
        static_cast<uint32_t>(g * 8 + (lane & 7)) * static_cast<uint32_t>(kqs) * 2u +
        static_cast<uint32_t>((lane >> 3) & 1) * 16u;
    const uint32_t stage_bytes = static_cast<uint32_t>(kBlockN * kqs * 2);
    const uint32_t depth_off = static_cast<uint32_t>(dq * kDepth) * 2u;

    float o[kWt][4];
#pragma unroll
    for (int t = 0; t < kWt; ++t) {
        o[t][0] = 0.0f;
        o[t][1] = 0.0f;
        o[t][2] = 0.0f;
        o[t][3] = 0.0f;
    }

    // The softmax state of this warp's row.
    const int sr = warp;
    float mrow = kNegInf, lrow = 0.0f;

    const float sc = p.scale * 1.4426950408889634f; // log2(e): the exp is an ex2
    const int r0 = lane >> 2;
    const int r1 = r0 + 8;
    const int c0 = 2 * (lane & 3);

    int stage = 0;
    for (int64_t j0 = jbeg; j0 < jend; j0 += kBlockN) {
        cp_async_wait_group<0>();
        __syncthreads();

        // The next tile is issued here and has the whole tile to arrive: the
        // panel it targets is the other stage, which nothing reads until the
        // next iteration's wait.
        if (j0 + kBlockN < jend) {
            const int64_t jn = j0 + kBlockN;
            stage_rows<kBlockN, P, DQK>([&](int r) { return k_row(p, b, hkv, jn + r); },
                                        Kb + static_cast<int64_t>(stage ^ 1) * kBlockN * kqs,
                                        tid);
        }
        cp_async_commit_group();

        // ---- Q.K ----
        // Every warp walks its own depth slice: the score is the sum of the
        // quarters' partial products, folded in the softmax pass. (Two
        // accumulator chains were measured here and are 4-5% slower at 114
        // registers: with sixteen warps to interleave, the single nine-step
        // chain already has enough neighbours to hide behind.)
        float s4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const uint32_t kb = k_off + depth_off + static_cast<uint32_t>(stage) * stage_bytes;
#pragma unroll
        for (int ks = 0; ks < kSteps; ++ks) {
            uint32_t kf[2];
            ldmatrix_x2_at(&kf[0], &kf[1], kb + static_cast<uint32_t>(ks * 32));
            hmma<T>(s4, qf[ks], kf[0], kf[1]);
        }
        {
            const int kc = c0 + 8 * g;
            Sc[r0 * kSStride + kc] = s4[0];
            Sc[r0 * kSStride + kc + 1] = s4[1];
            Sc[r1 * kSStride + kc] = s4[2];
            Sc[r1 * kSStride + kc + 1] = s4[3];
        }
        __syncthreads();

        // ---- softmax: this warp's row, its 32 lanes on the tile's keys ----
        {
            // `jend` rather than `hi`: a split's last tile runs past its span,
            // and those keys belong to the next split's softmax.
            const int j = static_cast<int>(j0) + lane;
            const bool vis = j < jend && (j >= lo || j < sink_n);
            float sv = 0.0f;
#pragma unroll
            for (int dd = 0; dd < kDepthSplits; ++dd) {
                sv += Sb[static_cast<int64_t>(dd) * kRows * kSStride + sr * kSStride + lane];
            }
            sv = vis ? sv * sc : kNegInf;
            const float mt = __shfl_sync(0xffffffffu, warp_reduce_max(sv), 0);
            const float mn = fmaxf(mrow, mt);
            const float rs = (mrow == kNegInf) ? 0.0f : ex2(mrow - mn);
            mrow = mn;
            const float pv = (mn != kNegInf) ? ex2(sv - mn) : 0.0f;
            const float sum = warp_reduce_sum(pv);
            lrow = lrow * rs + sum;
            Pt[sr * kPStride + lane] = __float2half(pv);
            if (lane == 0) {
                rowS[sr] = rs;
            }
        }
        __syncthreads();

        // ---- P.V ----
        {
            const float rs0 = rowS[r0];
            const float rs1 = rowS[r1];
#pragma unroll
            for (int t = 0; t < kWt; ++t) {
                o[t][0] *= rs0;
                o[t][1] *= rs0;
                o[t][2] *= rs1;
                o[t][3] *= rs1;
            }
            const uint32_t pb = smem_addr_u32(Pt) +
                                static_cast<uint32_t>((lane & 7) + ((lane >> 3) & 1) * 8) *
                                    static_cast<uint32_t>(kPStride) * 2u +
                                static_cast<uint32_t>(lane >> 4) * 16u;
            const uint32_t vb =
                smem_addr_u32(Kb) + static_cast<uint32_t>(stage) * stage_bytes +
                static_cast<uint32_t>((lane & 7) + ((lane >> 3) & 1) * 8) *
                    static_cast<uint32_t>(kqs) * 2u +
                static_cast<uint32_t>(lane >> 4) * 16u +
                static_cast<uint32_t>(warp * kCols) * 2u;
#pragma unroll
            for (int kk = 0; kk < kBlockN / 16; ++kk) {
                uint32_t pa[4];
                ldmatrix_x4_at(&pa[0], &pa[1], &pa[2], &pa[3], pb + static_cast<uint32_t>(kk * 32));
                uint32_t vf[kWt / 2][4];
#pragma unroll
                for (int t = 0; t < kWt / 2; ++t) {
                    ldmatrix_x4_trans_at(&vf[t][0], &vf[t][1], &vf[t][2], &vf[t][3],
                                         vb + static_cast<uint32_t>(kk * 16) *
                                                  static_cast<uint32_t>(kqs * 2) +
                                             static_cast<uint32_t>(t * 32));
                }
#pragma unroll
                for (int t = 0; t < kWt / 2; ++t) {
                    hmma<T>(o[2 * t], pa, vf[t][0], vf[t][1]);
                    hmma<T>(o[2 * t + 1], pa, vf[t][2], vf[t][3]);
                }
            }
        }
        stage ^= 1;
    }

    // ---- the head sink, normalization and the gate ----
    // A split call leaves the sink to the merge, which applies it to the folded
    // denominator exactly once.
    float scl = 1.0f;
    if (p.sink != nullptr && s.splits == 1) {
        const float sinkv = sink_at(p, hb + sr) * 1.4426950408889634f;
        const float a = fmaxf(mrow, sinkv);
        scl = (mrow == kNegInf) ? 0.0f : ex2(mrow - a);
        lrow = lrow * scl + ex2(sinkv - a);
    }

    if (s.splits > 1) {
        // Partial state for the merge pass: this block's own softmax state and
        // its value accumulators, both row-major by (batch, head).
        if (lane == 0) {
            const int64_t rowr = (b * p.H + hb + sr) * s.splits + sid;
            s.m[rowr] = mrow;
            s.l[rowr] = lrow;
        }
        const int64_t obase = b * p.H + hb;
#pragma unroll
        for (int t = 0; t < kWt; ++t) {
            const int col = warp * kCols + 8 * t + c0;
            float* const d0 = s.o + ((obase + r0) * s.splits + sid) * p.Dv + col;
            float* const d1 = s.o + ((obase + r1) * s.splits + sid) * p.Dv + col;
            d0[0] = o[t][0];
            d0[1] = o[t][1];
            d1[0] = o[t][2];
            d1[1] = o[t][3];
        }
        return;
    }

    __syncthreads();
    if (lane == 0) {
        rowS[sr] = scl;
        rowN[sr] = lrow > 0.0f ? 1.0f / lrow : 0.0f;
    }
    __syncthreads();

    {
        const float f0 = rowS[r0] * rowN[r0];
        const float f1 = rowS[r1] * rowN[r1];
        const int64_t rbase = b * p.o_b + i * p.o_s;
        T* const d0 = p.o + rbase + (hb + r0) * p.o_h;
        T* const d1 = p.o + rbase + (hb + r1) * p.o_h;
        const int64_t gbase = b * p.g_b + i * p.g_s;
        const T* const g0 = p.has_gate ? p.gate + gbase + (hb + r0) * p.g_h : nullptr;
        const T* const g1 = p.has_gate ? p.gate + gbase + (hb + r1) * p.g_h : nullptr;
#pragma unroll
        for (int t = 0; t < kWt; ++t) {
            const int col = warp * kCols + 8 * t + c0;
            float v0 = o[t][0] * f0;
            float v1 = o[t][1] * f0;
            float v2 = o[t][2] * f1;
            float v3 = o[t][3] * f1;
            if (g0 != nullptr) {
                v0 *= gate_act_value(p.gate_act, s_load(g0 + col));
                v1 *= gate_act_value(p.gate_act, s_load(g0 + col + 1));
                v2 *= gate_act_value(p.gate_act, s_load(g1 + col));
                v3 *= gate_act_value(p.gate_act, s_load(g1 + col + 1));
            }
            s_store(d0 + col, v0);
            s_store(d0 + col + 1, v1);
            s_store(d1 + col, v2);
            s_store(d1 + col + 1, v3);
        }
    }
}

/// The split merge: one block a (batch, head group), sixteen rows and the value
/// width. Every row's spans fold in the online-softmax weights, then the head
/// sink, the normalization and the gate — the decode merge's arithmetic over
/// this tile's partial layout. Its own block size: the tile's sixteen warps are
/// for the mma pipeline, and nothing here computes.
constexpr int kMergeThreads = 256;

template <class T>
__global__ void __launch_bounds__(kMergeThreads) attention_mla_merge_kernel(AttnParams<T, false> p,
                                                                            MlaSplits s)
{
    __shared__ float w[32];
    __shared__ float norm;
    const int64_t b = blockIdx.y;
    const int64_t hb = static_cast<int64_t>(blockIdx.x) * kRows;
    const int64_t i = blockIdx.z;

    for (int r = 0; r < kRows; ++r) {
        const int64_t row = (b * p.H + hb + r) * s.splits;
        __syncthreads();
        if (threadIdx.x == 0) {
            float m = kNegInf;
            for (int sp = 0; sp < s.splits; ++sp) {
                m = fmaxf(m, s.m[row + sp]);
            }
            if (p.sink != nullptr) {
                m = fmaxf(m, sink_at(p, hb + r) * 1.4426950408889634f);
            }
            float l = 0.0f;
            for (int sp = 0; sp < s.splits; ++sp) {
                w[sp] = (s.m[row + sp] == kNegInf) ? 0.0f : ex2(s.m[row + sp] - m);
                l += w[sp] * s.l[row + sp];
            }
            if (p.sink != nullptr) {
                l += ex2(sink_at(p, hb + r) * 1.4426950408889634f - m);
            }
            norm = l > 0.0f ? 1.0f / l : 0.0f;
        }
        __syncthreads();
        const int64_t rbase = b * p.o_b + i * p.o_s + (hb + r) * p.o_h;
        const int64_t gbase = b * p.g_b + i * p.g_s + (hb + r) * p.g_h;
        for (int64_t c = threadIdx.x; c < p.Dv; c += blockDim.x) {
            float acc = 0.0f;
            for (int sp = 0; sp < s.splits; ++sp) {
                acc += w[sp] * s.o[(row + sp) * p.Dv + c];
            }
            float v = acc * norm;
            if (p.has_gate) {
                v *= gate_act_value(p.gate_act, s_load(p.gate + gbase + c));
            }
            s_store(p.o + rbase + c, v);
        }
    }
}

} // namespace mla
} // namespace nnops::backend::cuda::attn
