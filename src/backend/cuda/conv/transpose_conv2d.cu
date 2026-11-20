/// @file transpose_conv2d.cu
/// @brief CUDA TransposeConv2D — Conv2D's implicit GEMM, cut by output residue
///        class.
///
/// A transposed convolution scatters: every input pixel contributes its whole
/// filter to a neighbourhood of outputs, so posed head-on it is not a GEMM. The
/// obvious repair — walk the output pixels and treat the KH·KW taps as the
/// reduction, zeroing the K positions a stride skips — costs SH·SW times the
/// MACs (4x on the stride-2 upsamplers that are this op's main use).
///
/// The outputs instead sort into residue classes. With
/// c = (oh + PH) mod SH, every output of one class is hit by the same subset of
/// kh, and that subset is an arithmetic progression: kh = k0 + t·step with
/// step = SH/gcd(DH, SH) (for DH = 1 that is just kh ≡ c (mod SH)). Writing
/// oh = c − PH + SH·m turns the class into an ordinary *correlation* with a
/// flipped, cropped kernel — ih = m − j0 − t·(DH/gcd) — which is exactly
/// Conv2D's gather with a negative dilation, a shifted origin and a strided
/// weight tap. So this file poses the same M = pixels, N = OC/G,
/// K = IC/G·Jc_h·Jc_w GEMM as conv2d.cu, one class per blockIdx.z slice,
/// and reuses its tile shapes: the exact MAC count, no padding ring, and a
/// block grid SH·SW times deeper than the undivided one.
///
/// The filter is [IC, OC/G, KH, KW] — the output channel sits *inside* the
/// reduction — so no k-flattening of it is contiguous and B is always gathered
/// element by element; only the activation keeps the 16 B lane, and an NHWC
/// activation has a channel axis dense enough to use it. That also rules out the
/// vectorized filter fill Conv2D's B side has.
///
/// Two geometries do not reach the GEMM. A group one channel wide on both sides
/// — a depthwise transposed convolution, where a 64-wide n tile would run
/// against a 1-wide reduction — has its own route in
/// depthwise_transpose_conv.cu, the same split the CPU backend makes. A grid
/// too deep for blockIdx.z goes to an output-per-thread script instead.
///
/// f32, f16 and bf16; the quantized route (an s8 weight) returns
/// unsupported.

#include "../cuda_common.cuh"
#include "../matmul/matmul_f16.cuh"
#include "../matmul/matmul_f32.cuh"
#include "conv_panel.cuh"
#include "transpose_conv_common.cuh"
#include "nnops/ops/transpose_conv2d.hpp"

namespace nnops::backend::cuda {

using tconv::ConvParams;
using tconv::TClass;
using tconv::tconv_class;
using tconv::tconv_store;

namespace {

/// Element (n, k) of one group's filter, walked through its own strides. The k
/// flattening is channel-fastest inside a tap.
template <class Elem>
__device__ inline Elem tconv_weight(const ConvParams& p, const Elem* base, int n, int k)
{
    const int tap = k / p.icg;
    const int ic = k - tap * p.icg;
    const int kh = tap / p.kw;
    const int kw = tap - kh * p.kw;
    return base[int64_t(n) * p.w_n + int64_t(ic) * p.w_ic + int64_t(kh) * p.w_kh +
                int64_t(kw) * p.w_kw];
}

/// The rows of an A tile one thread feeds on the channel-vector walk, and
/// the two pixel coordinates each of them starts from. The thread-to-(row,
/// k-group) mapping is fixed for the whole kernel, so the division that turns a
/// pixel into (oh, ow) is paid once, and `oh0`/`ow0` already carry the class's
/// input origin: a tap is then one add.
struct ConvRows {
    static constexpr int kMax = 4;
    int count = 0;
    int row[kMax] = {};
    int off[kMax] = {}; // (oh·sh − ph)·in_h + (ow·sw − pw)·in_w
    int oh0[kMax] = {};
    int ow0[kMax] = {};
    int ok[kMax] = {};
};

template <class Elem, int Threads, int bM, int bK>
__device__ inline ConvRows conv_rows(const ConvParams& p, int pixel0)
{
    constexpr int kGroup = 16 / int(sizeof(Elem)); // one 16 B vector of k
    constexpr int groups = bK / kGroup;
    constexpr int step = Threads / groups;
    static_assert(bM % step == 0 && bM / step <= ConvRows::kMax,
                  "the row walk has to fit ConvRows");
    const int M = p.oh * p.ow;
    const int row0 = int(threadIdx.x) / groups;

    ConvRows r;
    r.count = bM / step;
#pragma unroll
    for (int i = 0; i < ConvRows::kMax; ++i) {
        if (i >= r.count) {
            break;
        }
        const int row = row0 + i * step;
        const int pixel = pixel0 + row;
        const int oh = pixel / p.ow;
        const int ow = pixel - oh * p.ow;
        r.row[i] = row;
        r.oh0[i] = oh * p.sh - p.ph;
        r.ow0[i] = ow * p.sw - p.pw;
        r.off[i] = r.oh0[i] * int(p.in_h) + r.ow0[i] * int(p.in_w);
        r.ok[i] = pixel < M;
    }
    return r;
}

/// The vector fill, one 16 B vector per (row, k-group): k is the innermost
/// axis, so a group of `kGroup` consecutive k is a run of whole channels
/// inside a single tap -- contiguous in the activation and in the shared tile,
/// so both the load and the store are one 16 B move.
template <class Elem, int bK, class SmemA>
__device__ inline void conv_fill_a_vec(const ConvParams& p, SmemA sA, int stage, const Elem* in,
                                       const ConvRows& rows, int k0)
{
    constexpr int kGroup = 16 / int(sizeof(Elem)); // one 16 B vector of k
    using Vec = std::conditional_t<sizeof(Elem) == 2, uint4, float4>;

    const int K = p.icg * p.kh * p.kw;
    const int kg = int(threadIdx.x) % (bK / kGroup);
    const int k = k0 + kg * kGroup;
    const int tap = k / p.icg;
    const int ic = k - tap * p.icg;
    const int kh = tap / p.kw;
    const int kw = tap - kh * p.kw;
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
/// Every address is a 32-bit offset from the block's already-shifted input
/// pointer, and the walk decodes its starting index once and then steps it: a
/// division by a runtime value is a ~20-instruction sequence on this ISA, and
/// the element-at-a-time form spent two of them on every element of the tile.
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

template <class Elem>
__device__ inline Elem* tconv_out_addr(const ConvParams& p, int64_t base, int pixel, int n)
{
    const int oh = pixel / p.ow;
    const int ow = pixel - oh * p.ow;
    return static_cast<Elem*>(p.output) + base + int64_t(oh) * p.out_h + int64_t(ow) * p.out_w +
           int64_t(n) * p.out_c;
}

// ============================================================
// f16 tensor-core kernel
// ============================================================

/// Conv2D's tile, for its reasons: 64 pixels by 64 channels, 64 halves of k,
/// two stages — the depth that keeps three blocks resident per SM, which is
/// what the shallow upsampler grids need. A wider n tile is the tempting lever
/// and it is worse here for the same reason it is worse there.
using ConvShapeH = F16TileShape<32, 32, 2, 2, 64, 2, 1, false, 1>;

/// The gather's (row, col, stage) indexer over the k-major swizzled panel.
template <class S, class Elem> using TConvHPanel = conv::KMajorPanel<S, Elem>;

template <class Elem>
__global__ __launch_bounds__(ConvShapeH::threads, ConvShapeH::MinBlocks)
void tconv_gemm_h_kernel(ConvParams pin)
{
    using S = ConvShapeH;
    constexpr int BM = S::BM;
    constexpr int BN = S::BN;
    constexpr int BK = S::BK;
    constexpr int ST = S::stages;

    const int cls = int(blockIdx.z) % pin.classes;
    const int zg = int(blockIdx.z) / pin.classes;
    const TClass tc = tconv_class(pin, cls);
    if (tc.empty) {
        return;
    }

    const ConvParams p = tc.q;
    const int M = p.oh * p.ow;
    const int N = p.ocg;
    const int K = p.icg * p.kh * p.kw;
    const int m0 = int(blockIdx.x) * BM;
    if (m0 >= M) {
        return;
    }
    const int n0 = int(blockIdx.y) * BN;

    const int zb = zg / p.groups;
    const int g = zg - zb * p.groups;

    const int64_t in_base = int64_t(zb) * p.in_batch + int64_t(g) * p.icg * p.in_c;
    const int64_t out_base = int64_t(zb) * p.out_batch + int64_t(g) * p.ocg * p.out_c + tc.out_off;

    // A group owns IC/G consecutive input channels, so its filter slice starts
    // at g·IC/G·w_ic -- the input axis, not the output one.
    Elem const* W = static_cast<Elem const*>(p.weight) + int64_t(g) * p.icg * p.w_ic + tc.w_off;
    // Shifted once so the gather's own arithmetic stays 32-bit (the launch gate
    // keeps every offset it forms inside int).
    Elem const* in = static_cast<Elem const*>(p.input) + in_base;
    const ConvRows rows = conv_rows<Elem, S::threads, BM, BK>(p, m0);

    const int tid = int(threadIdx.x);
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / S::wn;
    const int warp_n = warp - warp_m * S::wn;

    extern __shared__ __align__(16) unsigned char tconv_smem[];
    Elem* const sAbase = reinterpret_cast<Elem*>(tconv_smem);
    Elem* const sBbase = sAbase + ST * S::kAMax;

    const int k_tiles = (K + BK - 1) / BK;
    constexpr int per = S::threads / BK; // filter rows per round

    // One call is one committed group, whether or not it copies anything: the
    // wait's group count has to stay uniform when the reduction runs dry.
    auto fill = [&](int kt, int stage) {
        if (kt >= k_tiles) {
            cp_async_commit_group();
            return;
        }
        const int k0 = kt * BK;
        Elem* const sA = sAbase + stage * S::kAMax;
        Elem* const sB = sBbase + stage * S::kBMax;
        if (p.fill_vec) {
            conv_fill_a_vec<Elem, BK>(p, TConvHPanel<S, Elem>{sA}, stage, in, rows, k0);
        } else {
            conv_fill_a<Elem, S::threads, BM, BK>(p, TConvHPanel<S, Elem>{sA}, stage, in, m0, k0);
        }
        // One filter column per thread into the n-major panel. The output
        // channel sits inside the reduction, so no vector is contiguous here.
        const int col = tid % BK;
        const int k = k0 + col;
        const int tap = k / p.icg;
        const int ic = k - tap * p.icg;
        const int kh = tap / p.kw;
        const int kw = tap - kh * p.kw;
        const bool k_ok = k < K;
        const Elem* wc = W + int64_t(ic) * p.w_ic + int64_t(kh) * p.w_kh + int64_t(kw) * p.w_kw;
        for (int row = tid / BK; row < BN; row += per) {
            const int n = n0 + row;
            sB[f16_swz_n<S>(col, row)] =
                (k_ok && n < N) ? wc[int64_t(n) * p.w_n] : gemm_zero<Elem>();
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
        fill(t, t % ST);
    }
    for (int kt = 0; kt < k_tiles; ++kt) {
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
#pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    const int n = gn + cc;
                    if (n >= N) {
                        continue;
                    }
                    Elem* dst = tconv_out_addr<Elem>(p, out_base, m, n);
                    tconv_store<Elem>(p, dst, acc[i][j][rr * 2 + cc], n, bias);
                }
            }
        }
    }
}

// ============================================================
// f32 SIMT kernel
// ============================================================

using ConvShapeF = F32TileShape<4, 8, 2, 2, 16, 3>;

/// The A panel in float units, padded rows of `BK + Pad`.
using TConvFPanel = conv::PaddedPanel<ConvShapeF::LDA>;

__global__ __launch_bounds__(ConvShapeF::threads, ConvShapeF::MinBlocks)
void tconv_gemm_f_kernel(ConvParams pin)
{
    using S = ConvShapeF;
    using Elem = float;
    constexpr int BM = S::BM;
    constexpr int BN = S::BN;
    constexpr int BK = S::BK;
    constexpr int ST = S::stages;

    const int cls = int(blockIdx.z) % pin.classes;
    const int zg = int(blockIdx.z) / pin.classes;
    const TClass tc = tconv_class(pin, cls);
    if (tc.empty) {
        return;
    }

    const ConvParams p = tc.q;
    const int M = p.oh * p.ow;
    const int N = p.ocg;
    const int K = p.icg * p.kh * p.kw;
    const int m0 = int(blockIdx.x) * BM;
    if (m0 >= M) {
        return;
    }
    const int n0 = int(blockIdx.y) * BN;

    const int zb = zg / p.groups;
    const int g = zg - zb * p.groups;

    const int64_t in_base = int64_t(zb) * p.in_batch + int64_t(g) * p.icg * p.in_c;
    const int64_t out_base = int64_t(zb) * p.out_batch + int64_t(g) * p.ocg * p.out_c + tc.out_off;

    // A group owns IC/G consecutive input channels, so its filter slice starts
    // at g·IC/G·w_ic -- the input axis, not the output one.
    Elem const* W = static_cast<Elem const*>(p.weight) + int64_t(g) * p.icg * p.w_ic + tc.w_off;
    // Shifted once so the gather's own arithmetic stays 32-bit (the launch gate
    // keeps every offset it forms inside int).
    Elem const* in = static_cast<Elem const*>(p.input) + in_base;
    const ConvRows rows = conv_rows<Elem, S::threads, BM, BK>(p, m0);

    const int tid = int(threadIdx.x);
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / S::wn;
    const int warp_n = warp - warp_m * S::wn;

    extern __shared__ __align__(16) unsigned char tconv_smem[];
    float* const sAbase = reinterpret_cast<float*>(tconv_smem);
    float* const sBbase = sAbase + ST * S::kAMax;

    const int k_tiles = (K + BK - 1) / BK;
    constexpr int per = S::threads / BK; // filter rows per round

    auto fill = [&](int kt, int stage) {
        if (kt >= k_tiles) {
            cp_async_commit_group();
            return;
        }
        const int k0 = kt * BK;
        float* const sA = sAbase + stage * S::kAMax;
        float* const sB = sBbase + stage * S::kBMax;
        if (p.fill_vec) {
            conv_fill_a_vec<Elem, BK>(p, TConvFPanel{sA}, stage, in, rows, k0);
        } else {
            conv_fill_a<Elem, S::threads, BM, BK>(p, TConvFPanel{sA}, stage, in, m0, k0);
        }
        // One filter column per thread into the [k][n] panel.
        const int col = tid % BK;
        const int k = k0 + col;
        const int tap = k / p.icg;
        const int ic = k - tap * p.icg;
        const int kh = tap / p.kw;
        const int kw = tap - kh * p.kw;
        const bool k_ok = k < K;
        const Elem* wc = W + int64_t(ic) * p.w_ic + int64_t(kh) * p.w_kh + int64_t(kw) * p.w_kw;
        for (int row = tid / BK; row < BN; row += per) {
            const int n = n0 + row;
            sB[col * S::LDN + row] = (k_ok && n < N) ? wc[int64_t(n) * p.w_n] : 0.0f;
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
        fill(t, t % ST);
    }
    for (int kt = 0; kt < k_tiles; ++kt) {
        const int stage = kt % ST;
        cp_async_wait_group<ST - 2>();
        __syncthreads();
        fill(kt + ST - 1, (kt + ST - 1) % ST);
        f32_gemm_tile<S, false, false>(sAbase + stage * S::kAMax,
                                            sBbase + stage * S::kBMax, warp_m, warp_n, lane, acc);
    }

    Elem const* bias =
        p.bias != nullptr ? static_cast<Elem const*>(p.bias) + int64_t(g) * p.ocg : nullptr;

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
#pragma unroll
        for (int c = 0; c < S::LaneCol; ++c) {
            const int n = n0 + col0 + c;
            if (n >= N) {
                continue;
            }
            Elem* dst = tconv_out_addr<Elem>(p, out_base, m, n);
            tconv_store<Elem>(p, dst, acc[r][c], n, bias);
        }
    }
}

// ============================================================
// Output-per-thread scripts
// ============================================================

/// Fetch one input channel of one tap with the stride/dilation arithmetic the
/// class decomposition does not need but a general geometry does.
template <class Elem>
__device__ inline bool tconv_tap(const ConvParams& p, const Elem* in, const Elem* w, int ic, int n,
                                 int oh, int ow, int kh, int kw, float& acc)
{
    const int t = oh + p.ph - kh * p.dh;
    if (t < 0 || t % p.sh != 0) {
        return false;
    }
    const int ih = t / p.sh;
    if (ih >= p.ih) {
        return false;
    }
    const int u = ow + p.pw - kw * p.dw;
    if (u < 0 || u % p.sw != 0) {
        return false;
    }
    const int iw = u / p.sw;
    if (iw >= p.iw) {
        return false;
    }
    acc += float(in[int64_t(ih) * p.in_h + int64_t(iw) * p.in_w + int64_t(ic) * p.in_c]) *
           float(w[int64_t(ic) * p.w_ic + int64_t(n) * p.w_n + int64_t(kh) * p.w_kh +
                   int64_t(kw) * p.w_kw]);
    return true;
}

/// One output element per thread, arbitrary layout, stride, dilation and group
/// geometry. blockIdx.x walks (plane, ow tile) flat so the (image, channel) pair
/// needs one division per block rather than one per element.
template <class Elem> __global__ void tconv_plane_kernel(ConvParams p, int ow_tiles)
{
    const int plane = int(blockIdx.x) / ow_tiles;
    const int ow = (int(blockIdx.x) - plane * ow_tiles) * int(blockDim.x) + int(threadIdx.x);
    const int oh = int(blockIdx.y) * int(blockDim.y) + int(threadIdx.y);
    if (ow >= p.ow || oh >= p.oh) {
        return;
    }

    const int oc = plane % p.oc;
    const int zb = plane / p.oc;
    const int g = oc / p.ocg;
    const int n = oc - g * p.ocg;

    const Elem* in =
        static_cast<const Elem*>(p.input) + int64_t(zb) * p.in_batch + int64_t(g) * p.icg * p.in_c;
    const Elem* w = static_cast<const Elem*>(p.weight) + int64_t(g) * p.icg * p.w_ic;

    float acc = 0.0f;
    for (int kh = 0; kh < p.kh; ++kh) {
        for (int kw = 0; kw < p.kw; ++kw) {
            for (int ic = 0; ic < p.icg; ++ic) {
                tconv_tap<Elem>(p, in, w, ic, n, oh, ow, kh, kw, acc);
            }
        }
    }

    const Elem* bias =
        p.bias != nullptr ? static_cast<const Elem*>(p.bias) + int64_t(g) * p.ocg : nullptr;
    Elem* dst = static_cast<Elem*>(p.output) + int64_t(zb) * p.out_batch + int64_t(oc) * p.out_c +
                int64_t(oh) * p.out_h + int64_t(ow) * p.out_w;
    tconv_store<Elem>(p, dst, acc, n, bias);
}

// ============================================================
// Launch
// ============================================================

template <class Elem>
void launch_tconv_gemm_h(const ConvParams& p, int m_max, cudaStream_t stream)
{
    using S = ConvShapeH;
    constexpr int smem = S::kSmemBytes;
    auto kernel = tconv_gemm_h_kernel<Elem>;
    static const cudaError_t cfg =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    (void)cfg;

    dim3 grid(ceil_div(m_max, S::BM), ceil_div(p.ocg, S::BN), p.batch * p.groups * p.classes);
    kernel<<<grid, S::threads, smem, stream>>>(p);
}

inline void launch_tconv_gemm_f(const ConvParams& p, int m_max, cudaStream_t stream)
{
    using S = ConvShapeF;
    constexpr int smem = S::stages * (S::kAMax + S::kBMax) * int(sizeof(float));
    auto kernel = tconv_gemm_f_kernel;
    static const cudaError_t cfg =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    (void)cfg;

    dim3 grid(ceil_div(m_max, S::BM), ceil_div(p.ocg, S::BN), p.batch * p.groups * p.classes);
    kernel<<<grid, S::threads, smem, stream>>>(p);
}

template <class Elem> void launch_tconv_plane(const ConvParams& p, cudaStream_t stream)
{
    constexpr int bx = 32, by = 8;
    const int ow_tiles = (p.ow + bx - 1) / bx;
    dim3 grid(unsigned(ow_tiles) * unsigned(p.batch) * unsigned(p.oc),
              unsigned((p.oh + by - 1) / by), 1);
    tconv_plane_kernel<Elem><<<grid, dim3(bx, by), 0, stream>>>(p, ow_tiles);
}

} // namespace

Status transpose_conv2d_cuda(const TransposeConv2DAttributes& attrs, TensorView& output,
                             std::span<const TensorView> inputs, const ComputeContext& ctx, void*)
{
    const TensorView& act = inputs[0];
    const TensorView& wgt = inputs[1];
    const TensorView* bias = inputs.size() > 2 ? &inputs[2] : nullptr;

    const DataType dt = act.data_type();
    if ((dt != DataType::f32 && dt != DataType::f16 && dt != DataType::bf16) ||
        wgt.data_type() != dt || output.data_type() != dt) {
        return Status::unsupported;
    }
    if (is_quantized_transpose_conv2d(wgt.data_type())) {
        return Status::unsupported;
    }
    if (output.numel() == 0) {
        return Status::ok;
    }
    if (wgt.rank() != 4) {
        return Status::unsupported;
    }
    ConvParams p;
    p.batch = static_cast<int>(act.shape(0));
    p.ic = static_cast<int>(act.shape(3));
    p.ih = static_cast<int>(act.shape(1));
    p.iw = static_cast<int>(act.shape(2));
    p.oc = static_cast<int>(output.shape(3));
    p.oh = static_cast<int>(output.shape(1));
    p.ow = static_cast<int>(output.shape(2));
    p.groups = static_cast<int>(attrs.groups);
    p.icg = p.ic / p.groups;
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
    p.min_clip = attrs.epilogue.min_clip;
    p.max_clip = attrs.epilogue.max_clip;
    p.add_to = attrs.add_to;
    p.pre_act = attrs.add_to && attrs.residual_pre_activation;

    // The filter is [IC][OC/G][KH][KW].
    p.w_ic = wgt.stride_elems(0);
    p.w_n = wgt.stride_elems(1);
    p.w_kh = wgt.stride_elems(2);
    p.w_kw = wgt.stride_elems(3);

    // The 16 B A-fill needs whole channels of one tap on a dense
    // channel axis: in_c == 1 puts consecutive channels next to each other, IC/G a
    // multiple of the vector makes a k-group never straddle a tap, and the
    // channel stride is what has to be 16 B aligned.
    const int64_t esz = static_cast<int64_t>(data_type_size(dt));
    const int kGroup = esz == 2 ? 8 : 4;
    p.fill_vec = p.in_c == 1 && p.icg % kGroup == 0 && (p.in_w * esz) % 16 == 0 &&
                 (reinterpret_cast<uintptr_t>(p.input) % 16 == 0);

    p.classes = p.sh * p.sw;
    // Launch gates, in the order the dispatch below reads them: M per class is
    // at most ceil(OH/SH)·ceil(OW/SW), blockIdx.z is 16-bit, and the fills form
    // 32-bit offsets inside one activation plane.
    const int m_max =
        int((int64_t(p.oh) + p.sh - 1) / p.sh) * int((int64_t(p.ow) + p.sw - 1) / p.sw);
    const int64_t zg = int64_t(p.batch) * p.groups * p.classes;
    const bool gemm = p.ocg >= 16 && p.classes <= 16 && zg <= 65535 && m_max <= 65535 &&
                      act.numel() <= (int64_t(1) << 30);
    // A depthwise geometry — one channel on both sides of a group — has no
    // channel axis to tile, so it takes the class-view correlation kernels of
    // depthwise_transpose_conv.cu instead of anything here.
    const int th_max = (p.oh + p.sh - 1) / p.sh;
    const int tw_max = (p.ow + p.sw - 1) / p.sw;
    const bool plane_ok = int64_t(((p.ow + 31) / 32)) * p.batch * p.oc < (int64_t(1) << 31);

    const cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    if (gemm) {
        if (dt == DataType::bf16) {
            launch_tconv_gemm_h<__nv_bfloat16>(p, m_max, stream);
        } else if (esz == 2) {
            launch_tconv_gemm_h<__half>(p, m_max, stream);
        } else {
            launch_tconv_gemm_f(p, m_max, stream);
        }
    } else if (tconv::transpose_conv2d_depthwise_cuda(p, th_max, tw_max, dt, stream)) {
        // launched by the depthwise route
    } else if (plane_ok) {
        if (dt == DataType::bf16) {
            launch_tconv_plane<__nv_bfloat16>(p, stream);
        } else if (esz == 2) {
            launch_tconv_plane<__half>(p, stream);
        } else {
            launch_tconv_plane<float>(p, stream);
        }
    } else {
        return Status::unsupported;
    }
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

} // namespace nnops::backend::cuda
