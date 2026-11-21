/// @file conv2d.cu
/// @brief CUDA Conv2D — one implicit-GEMM kernel on NHWC activations.
///
/// The convolution is posed as a GEMM: M is the output pixel axis, N the output
/// channels of one group, K the reduction over IC/G × KH × KW. The activation is
/// the gathered operand and therefore lives on A, whose shared-memory tile is
/// filled tap by tap with a bounds check that zeroes the padding ring; the filter
/// is B, read straight out of global memory. An NHWC activation carries the
/// channels innermost, so the filter is HWIO and the k-flattening is
/// channel-major inside a tap (k = tap·ICG + ic); the filter's fill and the
/// gather walk are both built for that order.
///
/// Shape indices follow the library's convention: an NHWC activation's dims are
/// the storage order, [N, H, W, C]. The epilogue writes through the operator's
/// own address arithmetic rather than a strided C tensor, because an NHWC row is
/// padded to a 32 B pitch and the pixel axis is only uniform inside one row. add_to reads the old output at that same
/// address before overwriting it, so the residual lands after the activation as
/// it does on the CPU.
///
/// The grid is one block per 64-pixel by 64-channel tile per batch·group slice,
/// which starves the machine once the plane shrinks: 14x14x256 is 16 blocks
/// against 114 resident slots (three blocks a SM at 30 KB of f32 tile or 32 KB
/// of f16). Those calls cut the reduction instead — blockIdx.z then carries
/// (batch·group)·splits + split, each block owns a proportional k range and
/// writes an f32 accumulator plane, and a second kernel sums the planes and runs
/// the epilogue once. That is what takes 14x14x256 k3 from 0.21 to 0.066 ms and
/// the 7x7x512 tail from 0.059 to 0.023 ms.
///
/// A 1x1 filter at unit stride is never gathered at all: it is the dense GEMM's
/// `[pixels][IC] x [IC][OC]` with the activation row as A, so
/// conv_pointwise_gemm hands that pose to the matmul engine — its shallow tile,
/// which is the shape a short-K product wants — once the reduction is deep
/// enough to beat this kernel. Everything below that threshold stays here.
///
/// f32, f16 and bf16 in the dense kernels; a Winograd weight never reaches them
/// (the dispatch refuses it) and a prepacked filter is opaque to them. The
/// quantized family (an s8 filter against a pre-quantized u8/s8 activation) is
/// conv2d_i8.cu: its own kernel pair, its own split-K reduce and its own
/// prepacked filter. What both families walk — the parameter block, the A-tile
/// row walk, the tap decode, the output addressing and the split-K planner —
/// is conv2d_common.cuh.

#include "conv2d_common.cuh" // params, A-tile row walk, tap decode, split-K planner
#include "conv2d_i8.cuh"      // the quantized route's entry
#include "../matmul/gemm_common.cuh"
#include "../matmul/matmul_op.cuh" // the dense GEMM the 1x1 route borrows
#include "../matmul/matmul_f16.cuh"
#include "../matmul/matmul_f32.cuh"
#include "conv_panel.cuh"
#include "nnops/ops/conv2d.hpp"

namespace nnops::backend::cuda {
namespace {

/// Element (n, k) of one group's filter, walked through its own strides so a
/// padded or sliced filter tensor is read correctly. The k-flattening is
/// channel-major inside a tap (HWIO): k = tap·ICG + ic.
template <class Elem>
__device__ inline Elem conv_weight(const ConvParams& p, const void* base, int n, int k)
{
    const int tap = k / p.icg;
    const int ic = k - tap * p.icg;
    const int kh = tap / p.kw;
    const int kw = tap - kh * p.kw;
    return static_cast<const Elem*>(base)[int64_t(n) * p.w_n + int64_t(ic) * p.w_ic +
                                          int64_t(kh) * p.w_kh + int64_t(kw) * p.w_kw];
}



/// The vector fill, one 16 B vector per (row, k-group): k is the innermost
/// axis, so a group of `kGroup` consecutive k is a run of whole channels
/// inside a single tap -- contiguous in the activation and contiguous in the
/// shared tile, so both the load and the store are one 16 B move. This is the
/// im2col pack the reference kernels do; element-at-a-time it cost 40-70% of the
/// kernel, since a 2-byte scalar load per element halves the traffic a request
/// can carry.
template <class Elem, int bK, class SmemA>
__device__ inline void conv_fill_a_vec(const ConvParams& p, SmemA sA, int stage, const Elem* in,
                                       const ConvRows& rows, int k0, ConvTapWalk& walk)
{
    constexpr int kGroup = 16 / int(sizeof(Elem)); // one 16 B vector of k
    using Vec = std::conditional_t<sizeof(Elem) == 2, uint4, float4>;

    const int K = p.icg * p.kh * p.kw;
    const int kg = int(threadIdx.x) % (bK / kGroup);
    const int k = k0 + kg * kGroup;
    conv_tap_seek(p, walk, k);
    const int ic = walk.ic, kh = walk.kh, kw = walk.kw;
    const int inc = int(p.in_c), inh = int(p.in_h), inw = int(p.in_w);
    const int koff = ic * inc + (kh * p.dh) * inh + (kw * p.dw) * inw;
    const int dh = kh * p.dh, dw = kw * p.dw;
    const bool k_ok = k < K;
    const Vec zero = {};

#pragma unroll
    for (int i = 0; i < ConvRows::kMax; ++i) {
        if (i >= rows.count) {
            break;
        }
        Vec v = zero;
        if (k_ok && rows.ok[i]) {
            const int ih = rows.oh0[i] + dh;
            const int iw = rows.ow0[i] + dw;
            if (unsigned(ih) < unsigned(p.ih) && unsigned(iw) < unsigned(p.iw)) {
                v = *reinterpret_cast<const Vec*>(in + koff + rows.off[i]);
            }
        }
        *reinterpret_cast<Vec*>(&sA(rows.row[i], kg * kGroup, stage)) = v;
    }
}

/// Fill one shared-memory A tile with the gathered activation. Threads spread
/// over the k axis in channel order (consecutive threads read consecutive
/// channels).
///
/// The walk decodes its starting index once and then steps it, and every
/// address is a 32-bit offset from the block's already-shifted input pointer.
/// That matters more than it sounds: a division by a runtime value is a
/// ~20-instruction sequence on this ISA, and the element-at-a-time form spent two
/// of them on every element of the tile — 45-79% of the kernel's time sat in
/// this fill. `ic` steps channel-fastest inside a tap, matching the HWIO
/// k-flattening.
template <class Elem, int Threads, int bM, int bK, class SmemA>
__device__ inline void conv_fill_a(const ConvParams& p, SmemA sA, int stage, const Elem* in,
                                   int pixel0, int k0)
{
    const int M = p.oh * p.ow;
    const int K = p.icg * p.kh * p.kw;
    const int tid = int(threadIdx.x);
    // Narrow, so the address arithmetic below stays 32-bit IMAD.
    const int inc = int(p.in_c), inh = int(p.in_h), inw = int(p.in_w);

    static_assert(Threads % bK == 0, "a thread has to own one k column");
    constexpr int per = Threads / bK; // pixels per round
    const int col = tid % bK;
    const int k = k0 + col;
    const int tap = k / p.icg;
    int ic = k - tap * p.icg;
    int kh = tap / p.kw;
    int kw = tap - kh * p.kw;
    int pixel = pixel0 + tid / bK;
    int oh = pixel / p.ow;
    int ow = pixel - oh * p.ow;
    for (int row = tid / bK; row < bM; row += per) {
        const int ih = oh * p.sh - p.ph + kh * p.dh;
        const int iw = ow * p.sw - p.pw + kw * p.dw;
        Elem val = Elem(0);
        if (pixel < M && k < K && unsigned(ih) < unsigned(p.ih) &&
            unsigned(iw) < unsigned(p.iw)) {
            val = in[ic * inc + ih * inh + iw * inw];
        }
        sA(row, col, stage) = val;
        pixel += per;
        ow += per;
        while (ow >= p.ow) {
            ow -= p.ow;
            ++oh;
        }
    }
}


/// The activation tags a conv kernel is compiled for.
///
/// `EpilogueActivateType` has ten kinds, and a kernel per kind would be ten
/// instantiations a dtype -- but a CNN's graph carries two of them. Those two
/// are dispatched at compile time, so their kernel holds no switch at all, and
/// the other eight share one run-time switch. The tail is out of the hot path
/// either way; what this buys is a store site of a couple of instructions
/// instead of a copy of every arm.
constexpr int kConvEpiNone = 0;
constexpr int kConvEpiRelu = 1;
constexpr int kConvEpiGeneral = 2;

template <int Epi> __device__ inline float conv_activate(const ConvParams& p, float v)
{
    if constexpr (Epi == kConvEpiNone) {
        return v;
    } else if constexpr (Epi == kConvEpiRelu) {
        return fmaxf(v, 0.0f);
    } else {
        return gemm_apply_epilogue(p.epilogue, p.ep_alpha, p.ep_beta, v);
    }
}

/// Which of the three a call needs.
inline int conv_epi_tag(const ConvParams& p)
{
    if (p.epilogue == EpilogueActivateType::None) {
        return kConvEpiNone;
    }
    if (p.epilogue == EpilogueActivateType::Relu) {
        return kConvEpiRelu;
    }
    return kConvEpiGeneral;
}


// ============================================================
// f16 tensor-core kernel
// ============================================================

/// The f16/bf16 conv tile: 64 pixels by 64 channels, 64 halves of k, two stages,
/// eight warps each holding a 32x16 strip.
///
/// The output tile and the k depth were swept on the CNN layer set in 2026-11-11
/// and have not moved: 32 KB is what fits three blocks a SM, and a deeper k panel
/// or a wider n tile each cost more residency than they return.
///
/// What did move (2026-11-20, on the hardware counters) is how many warps sit in
/// that tile. Four warps of 32x32 put 12 on an SM — 25% occupancy, 2.6 active
/// warps a scheduler, and the counters named it: 72% of cycles had no eligible
/// warp, the tensor pipe was 5.6% active, and per warp *every* MMA carried 11
/// non-MMA instructions. Splitting the same 64x64 of output eight ways instead —
/// 32x16 per warp, four atoms of A and two of B, 16 accumulator registers — uses
/// 80 registers and 32 KB, so three blocks still fit, and the SM then holds 24
/// warps. Stalls move from "waiting on latency" to "waiting on the integer
/// pipes", which is the better side of that trade: the wide-plane set goes
/// 0.940 -> 0.865 ms and the CNN layer set 0.450 -> 0.402 ms, with the small
/// planes up most (7x7x512 k1 -34%, the 224 7x7 stem -22%). The one loss is the
/// deepest k the set has, 56x56x256 k3, at +6%, because 36 k tiles already hid
/// the latency the extra warps are there to hide.
using ConvShapeH = F16TileShape<32, 16, 2, 4, 64, 2, 1, false, 3>;

/// The four-warp tile the eight-warp one replaced, kept for the deep k axis
/// (see `kConvDeepKTiles`): 32x32 per warp, two stages, the same 64x64 of output.
using ConvShapeHDeep = F16TileShape<32, 32, 2, 2, 64, 2, 1, false, 1>;

/// The gather's (row, col, stage) indexer over the k-major swizzled panel.
template <class S, class Elem> using ConvHPanel = conv::KMajorPanel<S, Elem>;

template <class Elem, class S, int Epi>
__global__ __launch_bounds__(S::threads, S::MinBlocks)
void conv_gemm_h_kernel(ConvParams p)
{
    constexpr int BM = S::BM;
    constexpr int BN = S::BN;
    constexpr int BK = S::BK;
    constexpr int ST = S::stages;

    extern __shared__ __align__(16) unsigned char conv_smem[];
    Elem* const sAbase = reinterpret_cast<Elem*>(conv_smem);
    Elem* const sBbase = sAbase + ST * S::kAMax;

    const int M = p.oh * p.ow;
    const int N = p.ocg;
    const int K = p.icg * p.kh * p.kw;
    const int m0 = int(blockIdx.x) * BM;
    const int n0 = int(blockIdx.y) * BN;
    const int tid = int(threadIdx.x);
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / S::wn;
    const int warp_n = warp - warp_m * S::wn;

    const ConvBlock blk = conv_block(p);
    const int zb = blk.zb;
    const int g = blk.g;

    const int64_t in_base = int64_t(zb) * p.in_batch + int64_t(g) * p.icg * p.in_c;
    const int64_t out_base = int64_t(zb) * p.out_batch + int64_t(g) * p.ocg * p.out_c;

    Elem const* W = static_cast<Elem const*>(p.weight) + int64_t(g) * p.ocg * p.w_n;
    // Shifted once so the gather's own arithmetic stays 32-bit (the launch gate
    // keeps every offset it forms inside int).
    Elem const* in = static_cast<Elem const*>(p.input) + in_base;
    const ConvRows rows = conv_rows<Elem, S::threads, BM, BK>(p, m0);

    int kt_begin = 0, kt_count = 0;
    conv_split_range(K, BK, p.k_splits, blk.split, kt_begin, kt_count);
    const int kt_end = kt_begin + kt_count;

    const bool n_full = (n0 + BN <= N);
    constexpr int kNVec = BN / 8;
    // The fill's k-region odometer: the ring asks for monotonic k, so the decode
    // can be carried instead of divided.
    ConvTapWalk tap_walk;

    // One call is one committed group, whether or not it copies anything: the
    // wait's group count has to stay uniform when a split runs dry near its end.
    auto fill = [&](int kt, int stage) {
        if (kt >= kt_end) {
            cp_async_commit_group();
            return;
        }
        const int k0 = kt * BK;
        Elem* const sA = sAbase + stage * S::kAMax;
        Elem* const sB = sBbase + stage * S::kBMax;
        if (p.fill_vec) {
            conv_fill_a_vec<Elem, BK>(p, ConvHPanel<S, Elem>{sA}, stage, in, rows, k0, tap_walk);
        } else {
            conv_fill_a<Elem, S::threads, BM, BK>(p, ConvHPanel<S, Elem>{sA}, stage, in, m0, k0);
        }
        // The filter is HWIO with the channel innermost, so a whole 16 B vector
        // is eight consecutive output channels of one k, and the k stride is the
        // group's channel count.
        if (p.weight_vec && n_full && k0 + BK <= K) {
            for (int i = tid; i < BK * kNVec; i += S::threads) {
                const int kr = i / kNVec;
                const int nv = i - kr * kNVec;
                cp_async_ca_16B(&sB[f16_swz_n<S>(kr, nv * 8)],
                                W + int64_t(n0 + nv * 8) * p.w_n + int64_t(k0 + kr) * p.w_ic);
            }
        } else {
            for (int i = tid; i < BN * BK; i += S::threads) {
                const int row = i / BK;
                const int col = i - row * BK;
                const int n = n0 + row;
                const int k = k0 + col;
                sB[f16_swz_n<S>(col, row)] =
                    (n < N && k < K) ? conv_weight<Elem>(p, W, n, k) : gemm_zero<Elem>();
            }
        }
        cp_async_commit_group();
    };

    float acc[S::MA][S::NA][4];
#pragma unroll
    for (int i = 0; i < S::MA; ++i)
#pragma unroll
        for (int j = 0; j < S::NA; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                acc[i][j][t] = 0.0f;
            }

    // Prime every stage but one; the loop then keeps exactly ST groups in
    // flight, so `wait_group<ST-2>` retires precisely the tile about to be read.
#pragma unroll
    for (int t = 0; t < ST - 1; ++t) {
        fill(kt_begin + t, (kt_begin + t) % ST);
    }
    for (int kt = kt_begin; kt < kt_end; ++kt) {
        const int stage = kt % ST;
        // One barrier a tile: the wait retires this tile's group, the barrier
        // then publishes the gather's plain stores and retires every warp's
        // reads of the tile before it -- whose stage the fill now overwrites.
        cp_async_wait_group<ST - 2>();
        __syncthreads();
        fill(kt + ST - 1, (kt + ST - 1) % ST);
        f16_gemm_tile<S, Elem, false, false>(sAbase + stage * S::kAMax,
                                                  sBbase + stage * S::kBMax, warp_m, warp_n, lane,
                                                  acc);
    }

    Elem const* bias =
        p.bias != nullptr ? static_cast<Elem const*>(p.bias) + int64_t(g) * p.ocg : nullptr;

    // A split's block stops at its accumulator: the sum, the bias and the
    // activation all belong to the reduce kernel, which sees every plane.
    float* part = nullptr;
    if (p.k_splits > 1) {
        // blockIdx.z already is zg·splits + split, which is the plane index.
        part = p.partial + int64_t(blockIdx.z) * int64_t(M) * N;
    }

    // C: lane l holds (m = l/4, n = 2*(l%4)), plus the +1 column and the +8 row.
    const int cm = lane / 4;
    const int cn = 2 * (lane % 4);
#pragma unroll
    for (int i = 0; i < S::MA; ++i) {
        const int gm = m0 + warp_m * S::WM + i * 16 + cm;
#pragma unroll
        for (int j = 0; j < S::NA; ++j) {
            const int gn = n0 + warp_n * S::WN + j * 8 + cn;
#pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                const int m = gm + rr * 8;
                if (m >= M) {
                    continue;
                }
                const int64_t row_base = conv_pixel_base(p, out_base, m);
#pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    const int n = gn + cc;
                    if (n >= N) {
                        continue;
                    }
                    float v = acc[i][j][rr * 2 + cc];
                    if (part != nullptr) {
                        part[int64_t(m) * N + n] = v;
                        continue;
                    }
                    if (bias != nullptr) {
                        v += float(bias[n]);
                    }
                    v = fminf(fmaxf(v, p.min_clip), p.max_clip);
                    v = conv_activate<Epi>(p, v);
                    conv_row_store<Elem>(p, row_base, n, v);
                }
            }
        }
    }
}

// ============================================================
// f32 SIMT kernel
// ============================================================

/// The f32 tile: 64 pixels by 64 channels, 8x8 per lane, and 16 elements of k a
/// stage.
///
/// Unlike the f16 twin the warp count is not the lever here. Eight warps of a
/// 32x16 lane tile do buy 24 warps and 50% occupancy, but the 4x4 lane tile
/// under them costs a quarter of the per-FMA reuse (2 loads a k step against 2.7)
/// and the sum comes out even: 112x112x64 k1 gains 28% and 56x56x256 k3 loses 9%.
/// The 4x8 lane tile keeps its reuse and takes a second stage instead of a third,
/// which is 20 KB and four resident blocks rather than 30 KB and three.
using ConvShapeF = F32TileShape<4, 8, 2, 2, 16, 2, 4, 8, 4, 4>;

/// The three-stage arm, kept for the deep k axis: 20 KB becomes 30 KB and three
/// resident blocks, which is the wrong trade until the pipeline has enough k to
/// run, and then it is the right one.
using ConvShapeFDeep = F32TileShape<4, 8, 2, 2, 16, 3, 4, 8, 4, 1>;

/// The A panel in float units: rows of `BK + Pad`, the padding being what keeps
/// a 16 B gather vector 16 B aligned on every row.
template <class S> using ConvFPanel = conv::PaddedPanel<S::LDA>;

template <class S, int Epi>
__global__ __launch_bounds__(S::threads, S::MinBlocks)
void conv_gemm_f_kernel(ConvParams p)
{
    constexpr int BM = S::BM;
    constexpr int BN = S::BN;
    constexpr int BK = S::BK;
    constexpr int ST = S::stages;
    constexpr int KU = S::KU;

    extern __shared__ __align__(16) unsigned char conv_smem[];
    float* const sAbase = reinterpret_cast<float*>(conv_smem);
    float* const sBbase = sAbase + ST * S::a_panel(false);

    const int M = p.oh * p.ow;
    const int N = p.ocg;
    const int K = p.icg * p.kh * p.kw;
    const int m0 = int(blockIdx.x) * BM;
    const int n0 = int(blockIdx.y) * BN;
    const int tid = int(threadIdx.x);
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / S::wn;
    const int warp_n = warp - warp_m * S::wn;

    const ConvBlock blk = conv_block(p);
    const int zb = blk.zb;
    const int g = blk.g;

    const int64_t in_base = int64_t(zb) * p.in_batch + int64_t(g) * p.icg * p.in_c;
    const int64_t out_base = int64_t(zb) * p.out_batch + int64_t(g) * p.ocg * p.out_c;

    float const* W = static_cast<float const*>(p.weight) + int64_t(g) * p.ocg * p.w_n;
    float const* in = static_cast<float const*>(p.input) + in_base;
    const ConvRows rows = conv_rows<float, S::threads, BM, BK>(p, m0);

    int kt_begin = 0, kt_count = 0;
    conv_split_range(K, BK, p.k_splits, blk.split, kt_begin, kt_count);
    const int kt_end = kt_begin + kt_count;

    const bool n_full = (n0 + BN <= N);
    // Column fills: one 16 B vector of every `kBStep`-th k row, so the row index
    // is a base plus a constant step instead of a divide per tile.
    constexpr int kBSvecPerRow = BN / KU;
    constexpr int kBStep = S::threads / kBSvecPerRow;
    const int bColOff = (tid % kBSvecPerRow) * KU;
    const int bRow0 = tid / kBSvecPerRow;
    ConvTapWalk tap_walk;

    auto fill = [&](int kt, int stage) {
        if (kt >= kt_end) {
            cp_async_commit_group();
            return;
        }
        const int k0 = kt * BK;
        float* const sA = sAbase + stage * S::a_panel(false);
        float* const sB = sBbase + stage * S::b_panel(false);
        if (p.fill_vec) {
            conv_fill_a_vec<float, BK>(p, ConvFPanel<S>{sA}, stage, in, rows, k0, tap_walk);
        } else {
            conv_fill_a<float, S::threads, BM, BK>(p, ConvFPanel<S>{sA}, stage, in, m0, k0);
        }
        // The filter is HWIO with the channel innermost, so a vector is four
        // consecutive output channels of one k and the k stride is the group's
        // channel count. The B panel keeps the [k][n] order the loop reads.
        if (p.weight_vec && n_full && k0 + BK <= K) {
            for (int kr = bRow0; kr < BK; kr += kBStep) {
                cp_async_ca_16B(&sB[kr * S::LDN + bColOff],
                                W + int64_t(n0 + bColOff) * p.w_n + int64_t(k0 + kr) * p.w_ic);
            }
        } else {
            for (int i = tid; i < BN * BK; i += S::threads) {
                const int row = i / BK;
                const int col = i - row * BK;
                const int n = n0 + row;
                const int k = k0 + col;
                sB[col * S::LDN + row] =
                    (n < N && k < K) ? conv_weight<float>(p, W, n, k) : 0.0f;
            }
        }
        cp_async_commit_group();
    };

    float acc[S::LaneRow][S::LaneCol];
#pragma unroll
    for (int r = 0; r < S::LaneRow; ++r)
#pragma unroll
        for (int c = 0; c < S::LaneCol; ++c) {
            acc[r][c] = 0.0f;
        }

#pragma unroll
    for (int t = 0; t < ST - 1; ++t) {
        fill(kt_begin + t, (kt_begin + t) % ST);
    }
    for (int kt = kt_begin; kt < kt_end; ++kt) {
        const int stage = kt % ST;
        cp_async_wait_group<ST - 2>();
        __syncthreads();
        fill(kt + ST - 1, (kt + ST - 1) % ST);
        f32_gemm_tile<S, false, false>(sAbase + stage * S::a_panel(false),
                                            sBbase + stage * S::b_panel(false), warp_m, warp_n,
                                            lane, acc);
    }

    float const* bias =
        p.bias != nullptr ? static_cast<float const*>(p.bias) + int64_t(g) * p.ocg : nullptr;

    float* part = nullptr;
    if (p.k_splits > 1) {
        part = p.partial + int64_t(blockIdx.z) * int64_t(M) * N;
    }

    // A lane is eight lane-rows by four lane-cols; its accumulator row r is the
    // row eight apart, its column c the next column of the lane's row.
    const int lrow = lane % 8;
    const int lcol = lane / 8;
    const int row0 = warp_m * S::warpRows + lrow;
    const int col0 = warp_n * S::warpCols + lcol * S::LaneCol;
#pragma unroll
    for (int r = 0; r < S::LaneRow; ++r) {
        const int m = m0 + row0 + r * 8;
        if (m >= M) {
            continue;
        }
        const int64_t row_base = conv_pixel_base(p, out_base, m);
#pragma unroll
        for (int c = 0; c < S::LaneCol; ++c) {
            const int n = n0 + col0 + c;
            if (n >= N) {
                continue;
            }
            float v = acc[r][c];
            if (part != nullptr) {
                part[int64_t(m) * N + n] = v;
                continue;
            }
            if (bias != nullptr) {
                v += bias[n];
            }
            v = fminf(fmaxf(v, p.min_clip), p.max_clip);
            v = conv_activate<Epi>(p, v);
            conv_row_store<float>(p, row_base, n, v);
        }
    }
}

// ============================================================
// split-K reduction
// ============================================================

/// Sum the k-split planes of one conv and run the epilogue once, exactly as the
/// single-pass kernel would have. One thread per (pixel, channel), planes in
/// split order so a split count is deterministic. The output goes through the
/// operator's own addressing, since an NHWC row is pitch-padded.
template <class Elem> __global__ void conv_splitk_reduce_kernel(ConvParams p)
{
    const int M = p.oh * p.ow;
    const int N = p.ocg;
    const int64_t plane = int64_t(M) * N;
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= plane) {
        return;
    }

    const int pixel = int(idx / N);
    const int n = int(idx - int64_t(pixel) * N);
    const int zg = int(blockIdx.y);

    const float* src = p.partial + int64_t(zg) * p.k_splits * plane + idx;
    float sum = 0.0f;
    for (int s = 0; s < p.k_splits; ++s) {
        sum += src[int64_t(s) * plane];
    }

    const int zb = zg / p.groups;
    const int g = zg - zb * p.groups;
    const int64_t out_base = int64_t(zb) * p.out_batch + int64_t(g) * p.ocg * p.out_c;

    float v = sum;
    if (p.bias != nullptr) {
        v += gemm_to_float(static_cast<const Elem*>(p.bias)[int64_t(g) * p.ocg + n]);
    }
    v = fminf(fmaxf(v, p.min_clip), p.max_clip);
    v = gemm_apply_epilogue(p.epilogue, p.ep_alpha, p.ep_beta, v);
    conv_store<Elem>(p, out_base, pixel, n, v);
}

// ============================================================
// Launch
// ============================================================


/// Sum the split planes and hand the scratch back. Shared by every tile and
/// both element types.
template <class Elem>
inline void conv_reduce(ConvParams& p, detail::GemmWorkspace ws, cudaStream_t stream)
{
    constexpr int kReduceThreads = 256;
    const int64_t plane = int64_t(p.oh) * p.ow * p.ocg;
    dim3 rgrid(unsigned((plane + kReduceThreads - 1) / kReduceThreads),
               unsigned(p.batch * p.groups), 1);
    conv_splitk_reduce_kernel<Elem><<<rgrid, kReduceThreads, 0, stream>>>(p);
    detail::gemm_workspace_free(ws, stream);
}

template <class Elem, class S, int Epi>
inline void launch_conv_h_as(ConvParams& p, cudaStream_t stream)
{
    constexpr int smem = S::kSmemBytes;
    auto kernel = conv_gemm_h_kernel<Elem, S, Epi>;
    static const cudaError_t cfg =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    (void)cfg;

    // Quarter wave: the 2-byte block is 2.3x shorter than the f32 one, so its
    // shallow cuts need a deeper grid to cover the same fixed cost.
    const detail::GemmWorkspace ws = conv_split_workspace(p, S::BM, S::BN, S::BK, smem, 4, stream);
    dim3 grid(ceil_div(p.oh * p.ow, S::BM), ceil_div(p.ocg, S::BN),
              p.batch * p.groups * p.k_splits);
    kernel<<<grid, S::threads, smem, stream>>>(p);
    if (p.k_splits > 1) {
        conv_reduce<Elem>(p, ws, stream);
    }
}

/// The reduction depth past which the other arm wins its turn.
///
/// Eight warps of 32x16 are there to hide the gather's latency with more
/// resident warps, and a deep k axis does that by itself: the prologue is the
/// first tile and nothing after it, so once the pipeline is running the latency
/// is gone and what is left is instructions per MMA -- where four warps of 32x32
/// carry a quarter fewer ldmatrix per MMA. The f32 twin's lever is the third
/// stage rather than the warp count, and it turns at the same place.
///
/// The line is drawn on K, not on k tiles, because the two dtypes use different
/// k depths (64 halves against 16 floats) and the crossover is the same shape in
/// both: 112x112x128 3x3 (K = 1152) still wants the shallow arm, 56x56x256 3x3
/// (K = 2304) and 28x28x128 5x5 (K = 3200) want the deep one.
constexpr int kConvDeepK = 1536;

inline int conv_k_depth(const ConvParams& p)
{
    return p.icg * p.kh * p.kw;
}

inline bool conv_wants_deep(const ConvParams& p)
{
    return conv_k_depth(p) >= kConvDeepK;
}

template <class Elem> void launch_conv_h(ConvParams& p, cudaStream_t stream)
{
    if (conv_wants_deep(p)) {
        launch_conv_h_as<Elem, ConvShapeHDeep, kConvEpiGeneral>(p, stream);
        return;
    }
    switch (conv_epi_tag(p)) {
    case kConvEpiNone: launch_conv_h_as<Elem, ConvShapeH, kConvEpiNone>(p, stream); break;
    case kConvEpiRelu: launch_conv_h_as<Elem, ConvShapeH, kConvEpiRelu>(p, stream); break;
    default: launch_conv_h_as<Elem, ConvShapeH, kConvEpiGeneral>(p, stream); break;
    }
}

template <class S, int Epi> inline void launch_conv_f_as(ConvParams& p, cudaStream_t stream)
{
    constexpr int smem = S::panel_bytes(false, false);
    auto kernel = conv_gemm_f_kernel<S, Epi>;
    static const cudaError_t cfg =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    (void)cfg;

    const detail::GemmWorkspace ws = conv_split_workspace(p, S::BM, S::BN, S::BK, smem, 2, stream);
    dim3 grid(ceil_div(p.oh * p.ow, S::BM), ceil_div(p.ocg, S::BN),
              p.batch * p.groups * p.k_splits);
    kernel<<<grid, S::threads, smem, stream>>>(p);
    if (p.k_splits > 1) {
        conv_reduce<float>(p, ws, stream);
    }
}

inline void launch_conv_f(ConvParams& p, cudaStream_t stream)
{
    if (conv_wants_deep(p)) {
        launch_conv_f_as<ConvShapeFDeep, kConvEpiGeneral>(p, stream);
        return;
    }
    switch (conv_epi_tag(p)) {
    case kConvEpiNone: launch_conv_f_as<ConvShapeF, kConvEpiNone>(p, stream); break;
    case kConvEpiRelu: launch_conv_f_as<ConvShapeF, kConvEpiRelu>(p, stream); break;
    default: launch_conv_f_as<ConvShapeF, kConvEpiGeneral>(p, stream); break;
    }
}

/// Where the two engines cross, measured on the pointwise shapes in both
/// dtypes (`bench_cuda_conv2d pointwise`). The dense GEMM wants a plane wide
/// enough to fill its grid in either dtype; past that f32 is a win at any
/// reduction depth (112x112x64 1.69x, 112x112x256 1.39x, 224x224x256 1.27x),
/// while f16 still needs a deep one — 112x112x256 is 0.92x and only 112x112x512
/// (1.21x) and 224x224x256 (1.11x) pay. Below 4096 pixels the conv kernel's
/// split-K covers a starved grid with work the GEMM's own cut does not.
constexpr int64_t kConvPointwiseMinPixels = 4096;
constexpr int64_t kConvPointwiseF16DeepIc = 512;
constexpr int64_t kConvPointwiseF16WidePixels = 16384;

/// A 1x1 filter at unit stride, no padding and one group is a plain GEMM, the
/// same reduction the CPU route takes as its pointwise plan: the pixel axis is
/// M, the channels are K, and the HWIO filter's `[IC][OC]` rows are B n-major.
///
/// The conv kernel would still gather that A row tap by tap through a
/// shared-memory panel and carry the split-K planner; the dense engine reads the
/// activation row itself, with its own tile and its own k cut. Only a dense NHWC
/// activation and output can be posed as one matrix, which is what the strides
/// below test — an NHWC view is dense by contract, but a caller can still hand a
/// sliced one.
///
/// Returns false when the call is outside the pose; the conv kernel runs then.
bool conv_pointwise_gemm(const Conv2DAttributes& attrs, const TensorView& act,
                         const TensorView& wgt, TensorView& output,
                         const TensorView* bias, cudaStream_t stream)
{
    if (attrs.kernel_size[0] != 1 || attrs.kernel_size[1] != 1 ||
        attrs.stride[0] != 1 || attrs.stride[1] != 1 ||
        attrs.dilation[0] != 1 || attrs.dilation[1] != 1 ||
        attrs.padding[0] != 0 || attrs.padding[1] != 0 || attrs.groups != 1) {
        return false;
    }
    const int knob = ::nnops::detail::cuda_conv_pw.load(std::memory_order_relaxed);
    if (knob == 0) {
        return false;
    }

    const int64_t ic = act.shape(3);
    const int64_t oc = output.shape(3);
    const int64_t iw = act.shape(2);
    const int64_t ih = act.shape(1);
    const int64_t ow = output.shape(2);
    const int64_t oh = output.shape(1);
    // A: pixels x channels, one row per pixel; C: the same rows of the output.
    if (act.stride_elems(3) != 1 || act.stride_elems(2) != ic ||
        act.stride_elems(1) != iw * ic || act.stride_elems(0) != ih * iw * ic) {
        return false;
    }
    if (output.stride_elems(3) != 1 || output.stride_elems(2) != oc ||
        output.stride_elems(1) != ow * oc || output.stride_elems(0) != oh * ow * oc) {
        return false;
    }
    // B n-major: HWIO's oc axis contiguous and its ic row OC long.
    if (wgt.stride_elems(3) != 1 || wgt.stride_elems(2) != oc) {
        return false;
    }

    const int64_t m = act.shape(0) * ih * iw;
    if (m <= 0 || m > (int64_t(1) << 30) || ic <= 0 || oc <= 0) {
        return false;
    }
    // The knob's 1 forces the arm through this gate so a bench can price both
    // engines on a shape the rule declines; the automatic route stops here.
    if (knob < 0) {
        const bool fp32 = act.data_type() == DataType::f32;
        const bool take = m >= kConvPointwiseMinPixels &&
                          (fp32 || ic >= kConvPointwiseF16DeepIc ||
                           (ic >= 256 && m >= kConvPointwiseF16WidePixels));
        if (!take) {
            return false;
        }
    }

    GemmParams p;
    p.a = act.ptr<void>();
    p.b = wgt.ptr<void>();
    p.c = output.ptr<void>();
    p.m = static_cast<int>(m);
    p.n = static_cast<int>(oc);
    p.k = static_cast<int>(ic);
    p.lda = ic;
    p.ldb = oc;
    p.ldc = oc;
    p.alpha = 1.0f;
    p.beta = 0.0f;
    p.bias = bias != nullptr ? bias->ptr<void>() : nullptr;
    p.bias_row = false;
    p.epilogue = attrs.epilogue.type;
    p.ep_alpha = attrs.epilogue.alpha;
    p.ep_beta = attrs.epilogue.beta;
    p.min_clip = attrs.epilogue.min_clip;
    p.max_clip = attrs.epilogue.max_clip;
    // add_to reads the destination back after the activation, which is exactly
    // what the dense epilogue's residual pointer does (CUDA refuses the
    // pre-activation form, so there is no other order to honour).
    if (attrs.add_to) {
        p.residual = output.ptr<void>();
        p.ldr = oc;
    }

    return dense_gemm_status(p, act.data_type(), /*transpose_a=*/false,
                             /*transpose_b=*/false, stream) == Status::ok;
}

} // namespace

Status conv2d_cuda(const Conv2DAttributes& attrs, TensorView& output,
                   std::span<const TensorView> inputs, const ComputeContext& ctx, void*)
{
    const TensorView& act = inputs[0];
    const TensorView& wgt = inputs[1];
    const TensorView* bias = inputs.size() > 2 ? &inputs[2] : nullptr;

    // The quantized family is an s8 filter against a pre-quantized u8/s8
    // activation; it owns its own kernel pair (conv2d_i8.cu).
    if (is_quantized_conv2d(wgt.data_type())) {
        if (output.numel() == 0) {
            return Status::ok;
        }
        return conv2d_i8_cuda(attrs, output, inputs, ctx);
    }

    const DataType dt = act.data_type();
    if ((dt != DataType::f32 && dt != DataType::f16 && dt != DataType::bf16) ||
        wgt.data_type() != dt || output.data_type() != dt) {
        return Status::unsupported;
    }
    if (output.numel() == 0) {
        return Status::ok;
    }
    if (attrs.algorithm == ConvAlgorithm::Winograd) {
        return Status::unsupported;
    }
    // The int8 blob is the quantized kernel's, and this route would read its
    // descriptor as an HWIO filter.
    if (is_packed_conv_weight(wgt.layout()) || is_wino_weight(wgt.layout()) ||
        is_cuda_conv_weight(wgt.layout())) {
        return Status::unsupported;
    }

    const cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    if (conv_pointwise_gemm(attrs, act, wgt, output, bias, stream)) {
        return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
    }

    ConvParams p;
    p.batch = static_cast<int>(act.shape(0));
    p.ic = static_cast<int>(act.shape(3));
    p.ih = static_cast<int>(act.shape(1));
    p.iw = static_cast<int>(act.shape(2));
    p.oc = static_cast<int>(output.shape(3));
    p.oh = static_cast<int>(output.shape(1));
    p.ow = static_cast<int>(output.shape(2));
    p.icg = static_cast<int>(wgt.shape(2));
    p.groups = static_cast<int>(attrs.groups);
    p.ocg = p.oc / p.groups;

    p.kh = static_cast<int>(attrs.kernel_size[0]);
    p.kw = static_cast<int>(attrs.kernel_size[1]);
    p.sh = static_cast<int>(attrs.stride[0]);
    p.sw = static_cast<int>(attrs.stride[1]);
    p.dh = static_cast<int>(attrs.dilation[0]);
    p.dw = static_cast<int>(attrs.dilation[1]);
    p.ph = static_cast<int>(attrs.padding[0]);
    p.pw = static_cast<int>(attrs.padding[1]);

    p.in_batch = act.stride_elems(0);
    p.in_c = act.stride_elems(3);
    p.in_h = act.stride_elems(1);
    p.in_w = act.stride_elems(2);
    p.out_batch = output.stride_elems(0);
    p.out_c = output.stride_elems(3);
    p.out_h = output.stride_elems(1);
    p.out_w = output.stride_elems(2);

    p.input = act.ptr<void>();
    p.weight = wgt.ptr<void>();
    p.output = output.ptr<void>();
    p.bias = bias != nullptr ? bias->ptr<void>() : nullptr;

    p.epilogue = attrs.epilogue.type;
    p.ep_alpha = attrs.epilogue.alpha;
    p.ep_beta = attrs.epilogue.beta;
    // The epilogue contract is activate(clamp(x, min_clip, max_clip)) on every
    // backend; the descriptor's bounds go through as-is.
    p.min_clip = attrs.epilogue.min_clip;
    p.max_clip = attrs.epilogue.max_clip;
    p.add_to = attrs.add_to;

    // The filter's own strides. `w_k_dense` says the k axis (IC/G·KH·KW) is
    // contiguous, which is what lets the kernel pose the filter as a plain
    // [OC/G][K] matrix; a padded filter row is not, and then B is gathered
    // element by element instead.
    // HWIO [KH][KW][IC/G][OC]
    p.w_kh = wgt.stride_elems(0);
    p.w_kw = wgt.stride_elems(1);
    p.w_ic = wgt.stride_elems(2);
    p.w_n = wgt.stride_elems(3);
    p.w_k_dense = p.w_n == 1 && p.w_ic == p.oc && p.w_kw == int64_t(p.icg) * p.oc &&
                  p.w_kh == int64_t(p.kw) * p.icg * p.oc;

    // A 16 B cp.async down the filter's contiguous axis: HWIO rows are OC long,
    // and a group's base moves by a whole number of rows.
    const int64_t esz = static_cast<int64_t>(data_type_size(dt));
    const int64_t row = int64_t(p.oc) * esz;
    p.weight_vec =
        p.w_k_dense && (reinterpret_cast<uintptr_t>(p.weight) % 16 == 0) && (row % 16 == 0);

    // The 16 B A-fill needs four (or eight) whole channels of one
    // tap on a dense channel axis: in_c == 1 puts consecutive channels next to
    // each other, IC/G a multiple of the vector makes a k-group never straddle
    // a tap, and the channel stride is what has to be 16 B aligned.
    const int kGroup = esz == 2 ? 8 : 4;
    p.fill_vec = p.in_c == 1 && p.icg % kGroup == 0 && (p.in_w * esz) % 16 == 0 &&
                 (reinterpret_cast<uintptr_t>(p.input) % 16 == 0);

    // The fill addresses the activation with 32-bit offsets from its block's
    // slice, so it is the slice's span -- not the pixel count -- that has to fit
    // the int range. `conv_offset_span` is that span.
    if (conv_offset_span(p) >= (int64_t(1) << 30) || p.batch * p.groups > 65535 ||
        p.ocg <= 0 || act.numel() > (int64_t(1) << 30)) {
        return Status::unsupported;
    }

    if (dt == DataType::bf16) {
        launch_conv_h<__nv_bfloat16>(p, stream);
    } else if (esz == 2) {
        launch_conv_h<__half>(p, stream);
    } else {
        launch_conv_f(p, stream);
    }
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

} // namespace nnops::backend::cuda
