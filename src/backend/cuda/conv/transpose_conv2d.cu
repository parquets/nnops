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
/// The filter is [IC, OC/G, KH, KW] in both memory orders — the output channel
/// sits *inside* the reduction — so no k-flattening of it is contiguous and B
/// is always gathered element by element; only the activation keeps the 16 B
/// lane, and only channel-last has a channel axis dense enough to use it. That
/// also rules out the dense-filter copy atom Conv2D's B side has.
///
/// Two geometries do not reach the GEMM. A group one channel wide on both sides
/// — a depthwise transposed convolution, where a 64-wide n tile would run
/// against a 1-wide reduction — has its own route in
/// depthwise_transpose_conv.cu, the same split the CPU backend makes. A grid
/// too deep for blockIdx.z goes to an output-per-thread script instead.
///
/// f32 and f16, channel-first and channel-last; the quantized route (an s8
/// weight) returns unsupported.

#include "../cuda_common.cuh"
#include "../gemm/cute_gemm.cuh"
#include "transpose_conv_common.cuh"
#include "nnops/ops/transpose_conv2d.hpp"

namespace nnops::backend::cuda {

using tconv::ConvParams;
using tconv::TClass;
using tconv::tconv_class;
using tconv::tconv_store;

namespace {

constexpr int kThreads = detail::kGemmThreads;

/// Element (n, k) of one group's filter, walked through its own strides. The k
/// flattening follows the memory order: tap-fastest inside an input channel for
/// channel-first, channel-fastest inside a tap for channel-last.
template <class Elem, bool ChannelsLast>
__device__ inline Elem tconv_weight(const ConvParams& p, const Elem* base, int n, int k)
{
    const int karea = p.kh * p.kw;
    int ic, tap;
    if constexpr (ChannelsLast) {
        tap = k / p.icg;
        ic = k - tap * p.icg;
    } else {
        ic = k / karea;
        tap = k - ic * karea;
    }
    const int kh = tap / p.kw;
    const int kw = tap - kh * p.kw;
    return base[int64_t(n) * p.w_n + int64_t(ic) * p.w_ic +
                int64_t(kh) * p.w_kh + int64_t(kw) * p.w_kw];
}

/// The rows of an A tile one thread feeds on the channel-last vector walk, and
/// the two pixel coordinates each of them starts from. The thread-to-(row,
/// k-group) mapping is fixed for the whole kernel, so the division that turns a
/// pixel into (oh, ow) is paid once, and `oh0`/`ow0` already carry the class's
/// input origin: a tap is then one add.
struct ConvRows {
    static constexpr int kMax = 4;
    int count = 0;
    int row[kMax] = {};
    int off[kMax] = {};   // (oh·sh − ph)·in_h + (ow·sw − pw)·in_w
    int oh0[kMax] = {};
    int ow0[kMax] = {};
    int ok[kMax] = {};
};

template <class Elem, int Threads, int bM, int bK>
__device__ inline ConvRows conv_rows(const ConvParams& p, int pixel0)
{
    constexpr int kGroup = 16 / int(sizeof(Elem));   // one 16 B vector of k
    constexpr int groups = bK / kGroup;
    constexpr int step = Threads / groups;
    static_assert(bM % step == 0 && bM / step <= ConvRows::kMax,
                  "the row walk has to fit ConvRows");
    const int M = p.oh * p.ow;
    const int row0 = int(threadIdx.x) / groups;

    ConvRows r;
    r.count = bM / step;
    CUTE_UNROLL
    for (int i = 0; i < ConvRows::kMax; ++i) {
        if (i >= r.count) { break; }
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

/// Channel-last fill, one 16 B vector per (row, k-group): k is the innermost
/// axis there, so a group of `kGroup` consecutive k is a run of whole channels
/// inside a single tap -- contiguous in the activation and in the shared tile,
/// so both the load and the store are one 16 B move.
template <class Elem, int bK, class SmemA>
__device__ inline void conv_fill_a_vec(const ConvParams& p, SmemA sA, int stage,
                                       const Elem* in, const ConvRows& rows,
                                       int k0)
{
    constexpr int kGroup = 16 / int(sizeof(Elem));   // one 16 B vector of k
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

    CUTE_UNROLL
    for (int i = 0; i < ConvRows::kMax; ++i) {
        if (i >= rows.count) { break; }
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
/// over the k axis in the channel-last order (consecutive threads read
/// consecutive channels) and over the pixel axis in the channel-first one
/// (consecutive threads read consecutive columns of a row).
///
/// Every address is a 32-bit offset from the block's already-shifted input
/// pointer, and both walks decode their starting index once and then step it: a
/// division by a runtime value is a ~20-instruction sequence on this ISA, and
/// the element-at-a-time form spent two of them on every element of the tile.
template <class Elem, bool ChannelsLast, int Threads, int bM, int bK, class SmemA>
__device__ inline void conv_fill_a(const ConvParams& p, SmemA sA, int stage,
                                   const Elem* in, int pixel0, int k0)
{
    const int M = p.oh * p.ow;
    const int K = p.icg * p.kh * p.kw;
    const int karea = p.kh * p.kw;
    const int tid = int(threadIdx.x);
    // Narrow, so the address arithmetic below stays 32-bit IMAD.
    const int inc = int(p.in_c), inh = int(p.in_h), inw = int(p.in_w);

    if constexpr (ChannelsLast) {
        static_assert(Threads % bK == 0, "a thread has to own one k column");
        constexpr int per = Threads / bK;   // pixels per round
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
            while (ow >= p.ow) { ow -= p.ow; ++oh; }
        }
    } else {
        static_assert(Threads % bM == 0 && bK % (Threads / bM) == 0,
                      "the k column walk has to divide the tile");
        constexpr int per = Threads / bM;   // k columns per thread
        const int row = tid % bM;
        const int pixel = pixel0 + row;
        const bool m_ok = pixel < M;
        const int oh = pixel / p.ow;
        const int ow = pixel - oh * p.ow;
        const int col0 = (tid / bM) * (bK / per);
        int k = k0 + col0;
        int ic = k / karea;
        int tap = k - ic * karea;
        int kh = tap / p.kw;
        int kw = tap - kh * p.kw;
        for (int j = 0; j < bK / per; ++j) {
            const int ih = oh * p.sh - p.ph + kh * p.dh;
            const int iw = ow * p.sw - p.pw + kw * p.dw;
            Elem val = Elem(0);
            if (m_ok && k < K && unsigned(ih) < unsigned(p.ih) &&
                unsigned(iw) < unsigned(p.iw)) {
                val = in[ic * inc + ih * inh + iw * inw];
            }
            sA(row, col0 + j, stage) = val;
            ++k;
            if (++kw == p.kw) {
                kw = 0;
                if (++kh == p.kh) { kh = 0; ++ic; }
            }
        }
    }
}

template <class Elem>
__device__ inline Elem* tconv_out_addr(const ConvParams& p, int64_t base,
                                       int pixel, int n)
{
    const int oh = pixel / p.ow;
    const int ow = pixel - oh * p.ow;
    return static_cast<Elem*>(p.output) + base + int64_t(oh) * p.out_h +
           int64_t(ow) * p.out_w + int64_t(n) * p.out_c;
}

// ============================================================
// f16 tensor-core kernel
// ============================================================

/// Conv2D's tile, for its reasons: 64 pixels by 64 channels, 64 halves of k,
/// two stages — the depth that keeps three blocks resident per SM, which is
/// what the shallow upsampler grids need. A wider n tile is the tempting lever
/// and it is worse here for the same reason it is worse there.
struct ConvTileH
{
    static constexpr int bM = 64;
    static constexpr int bN = 64;
    static constexpr int bK = 64;   // the swizzle atom is 64 halves wide in k
    static constexpr int stages = 2;
    static constexpr int wm = 2;
    static constexpr int wn = 2;
    static constexpr int mmaM = 32;
    static constexpr int mmaN = 32;
    static constexpr int threads = 32 * wm * wn;
};

template <bool ChannelsLast>
__global__ __launch_bounds__(ConvTileH::threads)
void tconv_gemm_h_kernel(ConvParams pin)
{
    using namespace cute;
    using Elem = cute::half_t;

    constexpr int bM = ConvTileH::bM;
    constexpr int bN = ConvTileH::bN;
    constexpr int bK = ConvTileH::bK;
    constexpr int kStages = ConvTileH::stages;

    const int cls = int(blockIdx.z) % pin.classes;
    const int zg = int(blockIdx.z) / pin.classes;
    const TClass tc = tconv_class(pin, cls);
    if (tc.empty) { return; }

    const ConvParams p = tc.q;
    const int M = p.oh * p.ow;
    const int N = p.ocg;
    const int K = p.icg * p.kh * p.kw;
    const int m0 = int(blockIdx.x) * bM;
    if (m0 >= M) { return; }
    const int n0 = int(blockIdx.y) * bN;

    const int zb = zg / p.groups;
    const int g = zg - zb * p.groups;

    const int64_t in_base = int64_t(zb) * p.in_batch + int64_t(g) * p.icg * p.in_c;
    const int64_t out_base = int64_t(zb) * p.out_batch + int64_t(g) * p.ocg * p.out_c +
                             tc.out_off;

    // A group owns IC/G consecutive input channels, so its filter slice starts
    // at g·IC/G·w_ic — the input axis, not the output one.
    Elem const* W = static_cast<Elem const*>(p.weight) +
                    int64_t(g) * p.icg * p.w_ic + tc.w_off;
    // Shifted once so the fill's own arithmetic stays 32-bit (the launch gate
    // keeps every offset it forms inside int).
    Elem const* in = static_cast<Elem const*>(p.input) + in_base;
    const ConvRows rows = conv_rows<Elem, ConvTileH::threads, bM, bK>(p, m0);

    auto cta_tiler = make_shape(Int<bM>{}, Int<bN>{}, Int<bK>{});
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);

    constexpr auto sA_layout = detail::gemm_smem_layout<true>(
        Int<bM>{}, Int<bK>{}, Int<kStages>{});
    constexpr auto sB_layout = detail::gemm_smem_layout<!ChannelsLast>(
        Int<bN>{}, Int<bK>{}, Int<kStages>{});
    constexpr int kSmemA = int(cosize(sA_layout));

    extern __shared__ char gemm_smem[];
    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<Elem*>(gemm_smem)), sA_layout);
    Tensor sB = make_tensor(
        make_smem_ptr(reinterpret_cast<Elem*>(gemm_smem + kSmemA * int(sizeof(Elem)))),
        sB_layout);

    TiledMMA mma = make_tiled_mma(
        SM80_16x8x16_F32F16F16F32_TN{},
        Layout<Shape<Int<ConvTileH::wm>, Int<ConvTileH::wn>, _1>>{},
        cute::Tile<Int<ConvTileH::mmaM>, Int<ConvTileH::mmaN>, _16>{});

    auto load_a = [&](int kt, int stage) {
        if constexpr (ChannelsLast) {
            if (p.fill_vec) {
                conv_fill_a_vec<Elem, bK>(p, sA, stage, in, rows, kt * bK);
                return;
            }
        }
        conv_fill_a<Elem, ChannelsLast, ConvTileH::threads, bM, bK>(
            p, sA, stage, in, m0, kt * bK);
    };
    // One column of B per thread, so the tap is decoded once and the row walk is
    // a stride. The filter's k axis is never contiguous here (the output channel
    // sits inside the reduction), so there is no copy atom to fall back on.
    auto load_b = [&](int kt, int stage) {
        constexpr int per = ConvTileH::threads / bK;   // rows per round
        const int col = int(threadIdx.x) % bK;
        const int k = kt * bK + col;
        const int karea = p.kh * p.kw;
        int ic, tap;
        if constexpr (ChannelsLast) {
            tap = k / p.icg;
            ic = k - tap * p.icg;
        } else {
            ic = k / karea;
            tap = k - ic * karea;
        }
        const int kh = tap / p.kw;
        const int kw = tap - kh * p.kw;
        const bool k_ok = k < K;
        const Elem* wc = W + int64_t(ic) * p.w_ic + int64_t(kh) * p.w_kh +
                         int64_t(kw) * p.w_kw;
        for (int row = int(threadIdx.x) / bK; row < bN; row += per) {
            const int n = n0 + row;
            sB(row, col, stage) =
                (k_ok && n < N) ? wc[int64_t(n) * p.w_n] : Elem(0);
        }
    };

    const int k_tile_count = (K + bK - 1) / bK;
    int k_tile_next = 0;
    int tiles_left = k_tile_count;

    CUTE_UNROLL
    for (int pipe = 0; pipe < kStages - 1; ++pipe) {
        load_a(k_tile_next, pipe);
        load_b(k_tile_next, pipe);
        cp_async_fence();
        --tiles_left;
        if (tiles_left > 0) { ++k_tile_next; }
    }

    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));
    // The identity tile carries the (m, n) coordinates the epilogue needs and
    // gives the fragment its static extent; there is no strided C tensor to
    // partition because the pixel axis is only uniform inside one output row.
    Tensor gCc = local_tile(make_identity_tensor(make_shape(M, N)), cta_tiler,
                            cta_coord, Step<_1, _1, X>{});
    Tensor tCcC = thr_mma.partition_C(gCc);
    Tensor tCrC = thr_mma.make_fragment_C(tCcC);
    clear(tCrC);

    Copy_Atom<SM75_U32x4_LDSM_N, Elem> s2r_atom_a;
    auto s2r_atom_b = [&] {
        if constexpr (ChannelsLast) {
            return Copy_Atom<SM75_U16x8_LDSM_T, Elem>{};
        } else {
            return Copy_Atom<SM75_U32x4_LDSM_N, Elem>{};
        }
    }();

    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
    ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
    Tensor tXsA = s2r_thr_copy_a.partition_S(sA);
    Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);

    TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
    ThrCopy s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
    Tensor tXsB = s2r_thr_copy_b.partition_S(sB);
    Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);

    int smem_pipe_read = 0;
    int smem_pipe_write = kStages - 1;
    Tensor tXsA_p = tXsA(_, _, _, smem_pipe_read);
    Tensor tXsB_p = tXsB(_, _, _, smem_pipe_read);
    constexpr int kBlocks = bK / 16;

    cp_async_wait<kStages - 2>();
    __syncthreads();
    copy(s2r_atom_a, tXsA_p(_, _, Int<0>{}), tXrA(_, _, Int<0>{}));
    copy(s2r_atom_b, tXsB_p(_, _, Int<0>{}), tXrB(_, _, Int<0>{}));

    CUTE_NO_UNROLL
    while (tiles_left > -(kStages - 1)) {
        CUTE_UNROLL
        for (int k_block = 0; k_block < kBlocks; ++k_block) {
            if (k_block == kBlocks - 1) {
                tXsA_p = tXsA(_, _, _, smem_pipe_read);
                tXsB_p = tXsB(_, _, _, smem_pipe_read);
                cp_async_wait<kStages - 2>();
                __syncthreads();
            }
            const int k_block_next = (k_block + 1) % kBlocks;
            copy(s2r_atom_a, tXsA_p(_, _, k_block_next), tXrA(_, _, k_block_next));
            copy(s2r_atom_b, tXsB_p(_, _, k_block_next), tXrB(_, _, k_block_next));
            if (k_block == 0) {
                load_a(k_tile_next, smem_pipe_write);
                load_b(k_tile_next, smem_pipe_write);
                cp_async_fence();
                --tiles_left;
                if (tiles_left > 0) { ++k_tile_next; }
                smem_pipe_write = smem_pipe_read;
                smem_pipe_read =
                    (smem_pipe_read == kStages - 1) ? 0 : smem_pipe_read + 1;
            }
            gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }
    }

    Elem const* bias = p.bias != nullptr
        ? static_cast<Elem const*>(p.bias) + int64_t(g) * p.ocg : nullptr;

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        auto coord = tCcC(i);
        const int m = int(get<0>(coord));
        const int n = int(get<1>(coord));
        if (m >= M || n >= N) { continue; }
        Elem* dst = tconv_out_addr<Elem>(p, out_base, m, n);
        tconv_store<Elem>(p, dst, float(tCrC(i)), n, bias);
    }
}

// ============================================================
// f32 SIMT kernel
// ============================================================

template <bool ChannelsLast>
__global__ __launch_bounds__(kThreads)
void tconv_gemm_f_kernel(ConvParams pin)
{
    using namespace cute;
    using Elem = float;

    constexpr int bM = 64;
    constexpr int bN = 64;
    constexpr int bK = 16;
    constexpr int kStages = 3;
    constexpr int kPad = 4;   // 16 B: a cp.async destination row start must be 16 B aligned

    const int cls = int(blockIdx.z) % pin.classes;
    const int zg = int(blockIdx.z) / pin.classes;
    const TClass tc = tconv_class(pin, cls);
    if (tc.empty) { return; }

    const ConvParams p = tc.q;
    const int M = p.oh * p.ow;
    const int N = p.ocg;
    const int K = p.icg * p.kh * p.kw;
    const int m0 = int(blockIdx.x) * bM;
    if (m0 >= M) { return; }
    const int n0 = int(blockIdx.y) * bN;

    const int zb = zg / p.groups;
    const int g = zg - zb * p.groups;

    const int64_t in_base = int64_t(zb) * p.in_batch + int64_t(g) * p.icg * p.in_c;
    const int64_t out_base = int64_t(zb) * p.out_batch + int64_t(g) * p.ocg * p.out_c +
                             tc.out_off;

    // A group owns IC/G consecutive input channels, so its filter slice starts
    // at g·IC/G·w_ic — the input axis, not the output one.
    Elem const* W = static_cast<Elem const*>(p.weight) +
                    int64_t(g) * p.icg * p.w_ic + tc.w_off;
    // Shifted once so the fill's own arithmetic stays 32-bit (the launch gate
    // keeps every offset it forms inside int).
    Elem const* in = static_cast<Elem const*>(p.input) + in_base;
    const ConvRows rows = conv_rows<Elem, kThreads, bM, bK>(p, m0);

    auto cta_tiler = make_shape(Int<bM>{}, Int<bN>{}, Int<bK>{});
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);

    constexpr auto sA_layout = make_layout(
        make_shape(Int<bM>{}, Int<bK>{}, Int<kStages>{}),
        make_stride(Int<bK + kPad>{}, Int<1>{}, Int<(bK + kPad) * bM>{}));
    constexpr auto sB_layout = [&] {
        if constexpr (ChannelsLast) {
            return make_layout(make_shape(Int<bN>{}, Int<bK>{}, Int<kStages>{}),
                               make_stride(Int<1>{}, Int<bN + kPad>{},
                                           Int<(bN + kPad) * bK>{}));
        } else {
            return make_layout(make_shape(Int<bN>{}, Int<bK>{}, Int<kStages>{}),
                               make_stride(Int<bK + kPad>{}, Int<1>{},
                                           Int<(bK + kPad) * bN>{}));
        }
    }();
    constexpr int kSmemA = int(cosize(sA_layout));

    extern __shared__ char gemm_smem[];
    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<Elem*>(gemm_smem)), sA_layout);
    Tensor sB = make_tensor(
        make_smem_ptr(reinterpret_cast<Elem*>(gemm_smem + kSmemA * int(sizeof(Elem)))),
        sB_layout);

    // One-thread atom: the atom layout has to cover every launched thread, or
    // the extra threads alias thread 0's slice and race with it (the epilogue
    // reads the destination for add_to).
    TiledMMA mma = make_tiled_mma(UniversalFMA<float, float, float>{},
                                  Layout<Shape<_8, _16, _1>>{},
                                  Tile<_64, _64, _1>{});

    auto load_a = [&](int kt, int stage) {
        if constexpr (ChannelsLast) {
            if (p.fill_vec) {
                conv_fill_a_vec<Elem, bK>(p, sA, stage, in, rows, kt * bK);
                return;
            }
        }
        conv_fill_a<Elem, ChannelsLast, kThreads, bM, bK>(
            p, sA, stage, in, m0, kt * bK);
    };
    auto load_b = [&](int kt, int stage) {
        constexpr int per = kThreads / bK;   // rows per round
        const int col = int(threadIdx.x) % bK;
        const int k = kt * bK + col;
        const int karea = p.kh * p.kw;
        int ic, tap;
        if constexpr (ChannelsLast) {
            tap = k / p.icg;
            ic = k - tap * p.icg;
        } else {
            ic = k / karea;
            tap = k - ic * karea;
        }
        const int kh = tap / p.kw;
        const int kw = tap - kh * p.kw;
        const bool k_ok = k < K;
        const Elem* wc = W + int64_t(ic) * p.w_ic + int64_t(kh) * p.w_kh +
                         int64_t(kw) * p.w_kw;
        for (int row = int(threadIdx.x) / bK; row < bN; row += per) {
            const int n = n0 + row;
            sB(row, col, stage) =
                (k_ok && n < N) ? wc[int64_t(n) * p.w_n] : Elem(0);
        }
    };

    const int k_tile_count = (K + bK - 1) / bK;
    int k_tile_next = 0;
    int tiles_left = k_tile_count;

    CUTE_UNROLL
    for (int pipe = 0; pipe < kStages - 1; ++pipe) {
        load_a(k_tile_next, pipe);
        load_b(k_tile_next, pipe);
        cp_async_fence();
        --tiles_left;
        if (tiles_left > 0) { ++k_tile_next; }
    }

    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCrA = thr_mma.make_fragment_A(tCsA(_, _, _, 0));
    Tensor tCrB = thr_mma.make_fragment_B(tCsB(_, _, _, 0));
    Tensor gCc = local_tile(make_identity_tensor(make_shape(M, N)), cta_tiler,
                            cta_coord, Step<_1, _1, X>{});
    Tensor tCcC = thr_mma.partition_C(gCc);
    Tensor tCrC = thr_mma.make_fragment_C(tCcC);
    clear(tCrC);

    int smem_pipe_read = 0;
    int smem_pipe_write = kStages - 1;

    CUTE_NO_UNROLL
    while (tiles_left > -(kStages - 1)) {
        cp_async_wait<kStages - 2>();
        __syncthreads();

        load_a(k_tile_next, smem_pipe_write);
        load_b(k_tile_next, smem_pipe_write);
        cp_async_fence();

        copy(tCsA(_, _, _, smem_pipe_read), tCrA);
        copy(tCsB(_, _, _, smem_pipe_read), tCrB);
        gemm(mma, tCrA, tCrB, tCrC);

        --tiles_left;
        if (tiles_left > 0) { ++k_tile_next; }
        smem_pipe_write = smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == kStages - 1) ? 0 : smem_pipe_read + 1;
    }

    Elem const* bias = p.bias != nullptr
        ? static_cast<Elem const*>(p.bias) + int64_t(g) * p.ocg : nullptr;

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        auto coord = tCcC(i);
        const int m = int(get<0>(coord));
        const int n = int(get<1>(coord));
        if (m >= M || n >= N) { continue; }
        Elem* dst = tconv_out_addr<Elem>(p, out_base, m, n);
        tconv_store<Elem>(p, dst, tCrC(i), n, bias);
    }
}

// ============================================================
// Output-per-thread scripts
// ============================================================

/// Fetch one input channel of one tap with the stride/dilation arithmetic the
/// class decomposition does not need but a general geometry does.
template <class Elem>
__device__ inline bool tconv_tap(const ConvParams& p, const Elem* in, const Elem* w,
                                 int ic, int n, int oh, int ow, int kh, int kw,
                                 float& acc)
{
    const int t = oh + p.ph - kh * p.dh;
    if (t < 0 || t % p.sh != 0) { return false; }
    const int ih = t / p.sh;
    if (ih >= p.ih) { return false; }
    const int u = ow + p.pw - kw * p.dw;
    if (u < 0 || u % p.sw != 0) { return false; }
    const int iw = u / p.sw;
    if (iw >= p.iw) { return false; }
    acc += float(in[int64_t(ih) * p.in_h + int64_t(iw) * p.in_w + int64_t(ic) * p.in_c]) *
           float(w[int64_t(ic) * p.w_ic + int64_t(n) * p.w_n +
                   int64_t(kh) * p.w_kh + int64_t(kw) * p.w_kw]);
    return true;
}

/// One output element per thread, arbitrary layout, stride, dilation and group
/// geometry. blockIdx.x walks (plane, ow tile) flat so the (image, channel) pair
/// needs one division per block rather than one per element.
template <class Elem>
__global__ void tconv_plane_kernel(ConvParams p, int ow_tiles)
{
    const int plane = int(blockIdx.x) / ow_tiles;
    const int ow = (int(blockIdx.x) - plane * ow_tiles) * int(blockDim.x) + int(threadIdx.x);
    const int oh = int(blockIdx.y) * int(blockDim.y) + int(threadIdx.y);
    if (ow >= p.ow || oh >= p.oh) { return; }

    const int oc = plane % p.oc;
    const int zb = plane / p.oc;
    const int g = oc / p.ocg;
    const int n = oc - g * p.ocg;

    const Elem* in = static_cast<const Elem*>(p.input) +
                     int64_t(zb) * p.in_batch + int64_t(g) * p.icg * p.in_c;
    const Elem* w = static_cast<const Elem*>(p.weight) +
                     int64_t(g) * p.icg * p.w_ic;

    float acc = 0.0f;
    for (int kh = 0; kh < p.kh; ++kh) {
        for (int kw = 0; kw < p.kw; ++kw) {
            for (int ic = 0; ic < p.icg; ++ic) {
                tconv_tap<Elem>(p, in, w, ic, n, oh, ow, kh, kw, acc);
            }
        }
    }

    const Elem* bias = p.bias != nullptr
        ? static_cast<const Elem*>(p.bias) + int64_t(g) * p.ocg : nullptr;
    Elem* dst = static_cast<Elem*>(p.output) + int64_t(zb) * p.out_batch +
                int64_t(oc) * p.out_c + int64_t(oh) * p.out_h + int64_t(ow) * p.out_w;
    tconv_store<Elem>(p, dst, acc, n, bias);
}

// ============================================================
// Launch
// ============================================================

template <bool ChannelsLast>
void launch_tconv_gemm_h(const ConvParams& p, int m_max, cudaStream_t stream)
{
    using namespace cute;
    constexpr int bM = ConvTileH::bM, bN = ConvTileH::bN, bK = ConvTileH::bK;
    constexpr auto la = detail::gemm_smem_layout<true>(
        Int<bM>{}, Int<bK>{}, Int<ConvTileH::stages>{});
    constexpr auto lb = detail::gemm_smem_layout<!ChannelsLast>(
        Int<bN>{}, Int<bK>{}, Int<ConvTileH::stages>{});
    constexpr int smem = int(cosize(la) + cosize(lb)) * int(sizeof(half_t));

    auto kernel = tconv_gemm_h_kernel<ChannelsLast>;
    static const cudaError_t cfg = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    (void)cfg;

    dim3 grid(ceil_div(m_max, bM), ceil_div(p.ocg, bN),
              p.batch * p.groups * p.classes);
    kernel<<<grid, ConvTileH::threads, smem, stream>>>(p);
}

template <bool ChannelsLast>
void launch_tconv_gemm_f(const ConvParams& p, int m_max, cudaStream_t stream)
{
    using namespace cute;   // ceil_div
    constexpr int bM = 64, bN = 64, bK = 16, kStages = 3, kPad = 4;
    constexpr int smem =
        (bM * (bK + kPad) + bN * (bK + kPad)) * kStages * int(sizeof(float));

    auto kernel = tconv_gemm_f_kernel<ChannelsLast>;
    static const cudaError_t cfg = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    (void)cfg;

    dim3 grid(ceil_div(m_max, bM), ceil_div(p.ocg, bN),
              p.batch * p.groups * p.classes);
    kernel<<<grid, kThreads, smem, stream>>>(p);
}

template <class Elem>
void launch_tconv_plane(const ConvParams& p, cudaStream_t stream)
{
    constexpr int bx = 32, by = 8;
    const int ow_tiles = (p.ow + bx - 1) / bx;
    dim3 grid(unsigned(ow_tiles) * unsigned(p.batch) * unsigned(p.oc),
              unsigned((p.oh + by - 1) / by), 1);
    tconv_plane_kernel<Elem><<<grid, dim3(bx, by), 0, stream>>>(p, ow_tiles);
}

}  // namespace

Status transpose_conv2d_cuda(const TransposeConv2DAttributes& attrs, TensorView& output,
                             std::span<const TensorView> inputs,
                             const ComputeContext& ctx, void*)
{
    const TensorView& act = inputs[0];
    const TensorView& wgt = inputs[1];
    const TensorView* bias = inputs.size() > 2 ? &inputs[2] : nullptr;

    const DataType dt = act.data_type();
    if ((dt != DataType::f32 && dt != DataType::f16) || wgt.data_type() != dt ||
        output.data_type() != dt) {
        return Status::unsupported;
    }
    if (is_quantized_transpose_conv2d(wgt.data_type())) { return Status::unsupported; }
    if (output.numel() == 0) { return Status::ok; }
    if (!is_channel_first_spatial(wgt.layout())) { return Status::unsupported; }
    const bool last = is_channel_last(act.layout());
    if (last != is_channel_last(output.layout())) { return Status::unsupported; }

    ConvParams p;
    p.batch = static_cast<int>(act.shape(0));
    p.ic = static_cast<int>(act.shape(1));
    p.ih = static_cast<int>(act.shape(2));
    p.iw = static_cast<int>(act.shape(3));
    p.oc = static_cast<int>(output.shape(1));
    p.oh = static_cast<int>(output.shape(2));
    p.ow = static_cast<int>(output.shape(3));
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
    p.in_c = act.stride_elems(1);
    p.in_h = act.stride_elems(2);
    p.in_w = act.stride_elems(3);
    p.out_batch = output.stride_elems(0);
    p.out_c = output.stride_elems(1);
    p.out_h = output.stride_elems(2);
    p.out_w = output.stride_elems(3);

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
    p.channels_last = last;

    // The filter is [IC][OC/G][KH][KW] in both memory orders.
    p.w_ic = wgt.stride_elems(0);
    p.w_n = wgt.stride_elems(1);
    p.w_kh = wgt.stride_elems(2);
    p.w_kw = wgt.stride_elems(3);

    // The channel-last 16 B A-fill needs whole channels of one tap on a dense
    // channel axis: in_c == 1 puts consecutive channels next to each other, IC/G a
    // multiple of the vector makes a k-group never straddle a tap, and the
    // channel stride is what has to be 16 B aligned.
    const int64_t esz = static_cast<int64_t>(data_type_size(dt));
    const int kGroup = esz == 2 ? 8 : 4;
    p.fill_vec = last && p.in_c == 1 && p.icg % kGroup == 0 &&
                 (p.in_w * esz) % 16 == 0 &&
                 (reinterpret_cast<uintptr_t>(p.input) % 16 == 0);

    p.classes = p.sh * p.sw;
    // Launch gates, in the order the dispatch below reads them: M per class is
    // at most ceil(OH/SH)·ceil(OW/SW), blockIdx.z is 16-bit, and the fills form
    // 32-bit offsets inside one activation plane.
    const int m_max = int((int64_t(p.oh) + p.sh - 1) / p.sh) *
                      int((int64_t(p.ow) + p.sw - 1) / p.sw);
    const int64_t zg = int64_t(p.batch) * p.groups * p.classes;
    const bool gemm = p.ocg >= 16 && p.classes <= 16 && zg <= 65535 &&
                      m_max <= 65535 && act.numel() <= (int64_t(1) << 30);
    // A depthwise geometry — one channel on both sides of a group — has no
    // channel axis to tile, so it takes the class-view correlation kernels of
    // depthwise_transpose_conv.cu instead of anything here.
    const int th_max = (p.oh + p.sh - 1) / p.sh;
    const int tw_max = (p.ow + p.sw - 1) / p.sw;
    const bool plane_ok = int64_t(((p.ow + 31) / 32)) * p.batch * p.oc < (int64_t(1) << 31);

    const cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    if (gemm) {
        if (esz == 2) {
            if (last) { launch_tconv_gemm_h<true>(p, m_max, stream); }
            else      { launch_tconv_gemm_h<false>(p, m_max, stream); }
        } else {
            if (last) { launch_tconv_gemm_f<true>(p, m_max, stream); }
            else      { launch_tconv_gemm_f<false>(p, m_max, stream); }
        }
    } else if (tconv::transpose_conv2d_depthwise_cuda(p, th_max, tw_max, last,
                                                     esz == 2, stream)) {
        // launched by the depthwise route
    } else if (plane_ok) {
        if (esz == 2) { launch_tconv_plane<cute::half_t>(p, stream); }
        else          { launch_tconv_plane<float>(p, stream); }
    } else {
        return Status::unsupported;
    }
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

}  // namespace nnops::backend::cuda
