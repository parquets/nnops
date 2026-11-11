/// @file conv2d_cuda.cu
/// @brief CUDA Conv2D — one implicit-GEMM kernel for both memory orders.
///
/// The convolution is posed the same way for channel-first and channel-last: M is
/// the output pixel axis, N the output channels of one group, K the reduction over
/// IC/G × KH × KW. The activation is the gathered operand and therefore lives
/// on A, whose shared-memory tile is filled tap by tap with a bounds check that
/// zeroes the padding ring; the filter is B, read straight out of global memory.
/// Only B changes shape between the two orders — OIHW flattens to a k-contiguous
/// [OC/G][K] tile, HWIO to an n-contiguous [K][OC] — so both routes run the same
/// pipeline and differ in the filter's copy atom and in which of the two
/// k-flattenings the gather walks. The k order has to match the filter's:
/// tap-major inside an output channel for OIHW (k = ic·karea + tap), channel-
/// major for HWIO (k = tap·ICG + ic).
///
/// Shape indices follow the library's convention: they are the logical channel-first
/// ones even for a channel-last tensor, and the layout tag only changes the
/// element strides. The epilogue therefore writes through the operator's own
/// address arithmetic rather than a strided C tensor, because a channel-first row is
/// padded to a 32 B pitch and the pixel axis is only uniform inside one row.
/// add_to reads the old output at that same address before overwriting it, so
/// the residual lands after the activation as it does on the CPU.
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
/// f32 and f16; a Winograd weight never reaches this kernel (the dispatch
/// refuses it), a prepacked filter is opaque to it, and quantized weights
/// return unsupported.

#include "../gemm/cute_gemm.cuh"
#include "nnops/ops/conv2d.hpp"

namespace nnops::backend::cuda {
namespace {

struct ConvParams {
    const void* input = nullptr;
    const void* weight = nullptr;
    const void* bias = nullptr;
    void* output = nullptr;

    int batch = 0;
    int ic = 0, ih = 0, iw = 0;
    int oc = 0, oh = 0, ow = 0;
    int kh = 0, kw = 0;
    int groups = 0, icg = 0, ocg = 0;
    int sh = 0, sw = 0, dh = 0, dw = 0, ph = 0, pw = 0;

    int64_t in_batch = 0, in_c = 0, in_h = 0, in_w = 0;
    int64_t out_batch = 0, out_c = 0, out_h = 0, out_w = 0;
    // Filter strides in elements: n is the output channel, ic the input channel
    // within the group, kh/kw the tap. A padded filter is not k-contiguous, so
    // the k-flattened (N, K) view only exists when w_k_dense holds.
    int64_t w_n = 0, w_ic = 0, w_kh = 0, w_kw = 0;
    bool w_k_dense = false;

    EpilogueActivateType epilogue = EpilogueActivateType::None;
    float ep_alpha = 0.0f;
    float ep_beta = 1.0f;
    float min_clip = -std::numeric_limits<float>::infinity();
    float max_clip = std::numeric_limits<float>::infinity();
    bool add_to = false;
    bool channels_last = false;
    bool weight_vec = false;
    bool fill_vec = false;

    // ---- split-K ----
    // Same form as the dense GEMM's: blockIdx.z carries (batch·group)·splits +
    // split, each block owns a proportional range of the k tiles and writes its
    // f32 accumulator tile to `partial` instead of running the epilogue, and a
    // second kernel sums the planes and epilogues once.
    int k_splits = 1;
    float* partial = nullptr;   ///< [(batch·group)][split][M][OC/G], row-major
};

constexpr int kConvThreads = detail::kGemmThreads;

/// Element (n, k) of one group's filter, walked through its own strides so a
/// padded or sliced filter tensor is read correctly.
template <class Elem, bool ChannelsLast>
__device__ inline Elem conv_weight(const ConvParams& p, const void* base,
                                   int n, int k)
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
    return static_cast<const Elem*>(base)[int64_t(n) * p.w_n +
                                          int64_t(ic) * p.w_ic +
                                          int64_t(kh) * p.w_kh +
                                          int64_t(kw) * p.w_kw];
}

/// The rows of an A tile one thread feeds on the channel-last vector walk, and
/// the two pixel coordinates each of them starts from. The thread-to-(row,
/// k-group) mapping is fixed for the whole kernel, so the division that turns a
/// pixel into (oh, ow) is paid once instead of once per k tile, and `oh0`/`ow0`
/// already carry the stride and the negative padding: a tap is then one add.
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

/// The (batch, group, split) triple one blockIdx.z stands for. Split varies
/// fastest, exactly as in the dense GEMM, so the k ranges of one product land on
/// different waves of the scheduler.
struct ConvBlock {
    int zb = 0;
    int g = 0;
    int split = 0;
};

__device__ inline ConvBlock conv_block(const ConvParams& p)
{
    const int zg = int(blockIdx.z);
    const int bg = p.k_splits > 1 ? zg / p.k_splits : zg;
    ConvBlock b;
    b.split = p.k_splits > 1 ? zg - bg * p.k_splits : 0;
    b.zb = bg / p.groups;
    b.g = bg - b.zb * p.groups;
    return b;
}

/// The k-tile range a split owns, as [begin, begin + count). Proportional cuts
/// of the tile count, so with `splits <= k_tiles` no split is empty.
__device__ inline void conv_split_range(int K, int bK, int splits, int split,
                                        int& kt_begin, int& kt_count)
{
    const int k_tiles = (K + bK - 1) / bK;
    if (splits <= 1) {
        kt_begin = 0;
        kt_count = k_tiles;
        return;
    }
    kt_begin = int(int64_t(k_tiles) * split / splits);
    kt_count = int(int64_t(k_tiles) * (split + 1) / splits) - kt_begin;
}

/// Channel-last fill, one 16 B vector per (row, k-group): k is the innermost
/// axis there, so a group of `kGroup` consecutive k is a run of whole channels
/// inside a single tap -- contiguous in the activation and contiguous in the
/// shared tile, so both the load and the store are one 16 B move. This is the
/// im2col pack the reference kernels do; element-at-a-time it cost 40-70% of the
/// kernel, since a 2-byte scalar load per element halves the traffic a request
/// can carry.
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
/// consecutive channels) and over the pixel axis in the channel-first one (consecutive
/// threads read consecutive columns of a row).
///
/// Both walks decode their starting index once and then step it, and every
/// address is a 32-bit offset from the block's already-shifted input pointer.
/// That matters more than it sounds: a division by a runtime value is a
/// ~20-instruction sequence on this ISA, and the element-at-a-time form spent two
/// of them on every element of the tile — 45-79% of the kernel's time sat in
/// this fill. `ic`/`tap` step tap-fastest inside a channel for the channel-first
/// order and channel-fastest inside a tap for the other, matching each order's
/// k-flattening.
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

/// Output element (pixel, channel) of the order the op was asked for: channel-first
/// gives the channel axis a whole plane, channel-last the innermost slot.
template <class Elem>
__device__ inline Elem* conv_out_addr(const ConvParams& p, int64_t base,
                                      int pixel, int n)
{
    const int oh = pixel / p.ow;
    const int ow = pixel - oh * p.ow;
    return static_cast<Elem*>(p.output) + base + int64_t(oh) * p.out_h +
           int64_t(ow) * p.out_w + int64_t(n) * p.out_c;
}

/// Post-accumulation store, including the residual an add_to conv adds after
/// the activation.
template <class Elem>
__device__ inline void conv_store(const ConvParams& p, int64_t base, int pixel,
                                  int n, float v)
{
    Elem* dst = conv_out_addr<Elem>(p, base, pixel, n);
    if (p.add_to) { v += float(*dst); }
    *dst = Elem(v);
}

// ============================================================
// f16 tensor-core kernel
// ============================================================

/// The f16 conv tile: 64 pixels by 64 channels, 64 halves of k, two stages.
///
/// Every axis was swept on the CNN layer set (2026-11-11), and the tile is 32 KB
/// because that is what fits three blocks a SM where three stages fit two: the
/// deeper pipeline only wins on the one 56x56 plane whose grid already hides the
/// gather, and loses by up to 1.7x on the stem and the 1x1s, where the extra
/// resident block is the whole margin (sum over the layer set 1.58 -> 1.20 ms).
///
/// A wider n tile is the tempting lever — every block re-gathers its own n slice
/// of the activation, so bN halves the total A traffic — and it loses anyway:
/// 1.1-6.9x on every layer whose OC/G does not cover the tile, because the
/// filter then falls back to its scalar gather, and no better than noise on the
/// ones where it does, because the wider tile halves the block count of planes
/// that were already too short to fill the machine. Wider m tiles and deeper k
/// panels lose the same way, on resident blocks.
struct ConvTileH
{
    static constexpr int bM = 64;
    static constexpr int bN = 64;
    static constexpr int bK = 64;   // the swizzle atom is 64 halves wide in k
    static constexpr int stages = 2;
    // 2x2 warps of a 32x32 atom. Wider warp grids break the LDSM copy atoms,
    // which are built off the per-warp tile.
    static constexpr int wm = 2;
    static constexpr int wn = 2;
    static constexpr int mmaM = 32;
    static constexpr int mmaN = 32;
    static constexpr int threads = 32 * wm * wn;
};

template <bool ChannelsLast>
__global__ __launch_bounds__(ConvTileH::threads)
void conv_gemm_h_kernel(ConvParams p)
{
    using namespace cute;
    using Elem = cute::half_t;

    constexpr int bM = ConvTileH::bM;
    constexpr int bN = ConvTileH::bN;
    constexpr int bK = ConvTileH::bK;
    constexpr int kStages = ConvTileH::stages;

    const int M = p.oh * p.ow;
    const int N = p.ocg;
    const int K = p.icg * p.kh * p.kw;
    const int m0 = int(blockIdx.x) * bM;
    const int n0 = int(blockIdx.y) * bN;

    const ConvBlock blk = conv_block(p);
    const int zb = blk.zb;
    const int g = blk.g;

    const int64_t in_base = int64_t(zb) * p.in_batch + int64_t(g) * p.icg * p.in_c;
    const int64_t out_base = int64_t(zb) * p.out_batch + int64_t(g) * p.ocg * p.out_c;

    Elem const* W = static_cast<Elem const*>(p.weight) +
                    int64_t(g) * p.ocg * p.w_n;
    // Shifted once so the fill's own arithmetic stays 32-bit (the launch gate
    // keeps every offset it forms inside int).
    Elem const* in = static_cast<Elem const*>(p.input) + in_base;
    const ConvRows rows = conv_rows<Elem, ConvTileH::threads, bM, bK>(p, m0);

    Tensor mB = [&] {
        if constexpr (ChannelsLast) {
            return make_tensor(make_gmem_ptr(W), make_shape(N, K),
                               make_stride(Int<1>{}, int(p.w_ic)));
        } else {
            return make_tensor(make_gmem_ptr(W), make_shape(N, K),
                               make_stride(int(p.w_n), Int<1>{}));
        }
    }();

    auto cta_tiler = make_shape(Int<bM>{}, Int<bN>{}, Int<bK>{});
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});

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

    TiledCopy copy_b = detail::gemm_copy_half<ConvTileH::threads, bN, bK, !ChannelsLast>();

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);
    Tensor tBsB = thr_copy_b.partition_D(sB);

    const bool n_full = (n0 + bN <= N);

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
    auto load_b = [&](int kt, int stage) {
        if (p.weight_vec && n_full && (kt + 1) * bK <= K) {
            copy(copy_b, tBgB(_, _, _, kt), tBsB(_, _, _, stage));
            return;
        }
        const int k0 = kt * bK;
        for (int i = threadIdx.x; i < bN * bK; i += blockDim.x) {
            const int row = i / bK;
            const int col = i - row * bK;
            const int n = n0 + row;
            const int k = k0 + col;
            sB(row, col, stage) =
                (n < N && k < K) ? conv_weight<Elem, ChannelsLast>(p, W, n, k)
                                 : Elem(0);
        }
    };

    int k_tile_count = 0;
    int k_tile_next = 0;
    conv_split_range(K, bK, p.k_splits, blk.split, k_tile_next, k_tile_count);

    CUTE_UNROLL
    for (int pipe = 0; pipe < kStages - 1; ++pipe) {
        load_a(k_tile_next, pipe);
        load_b(k_tile_next, pipe);
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }
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
    while (k_tile_count > -(kStages - 1)) {
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
                --k_tile_count;
                if (k_tile_count > 0) { ++k_tile_next; }
                smem_pipe_write = smem_pipe_read;
                smem_pipe_read =
                    (smem_pipe_read == kStages - 1) ? 0 : smem_pipe_read + 1;
            }
            gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }
    }

    Elem const* bias = p.bias != nullptr
        ? static_cast<Elem const*>(p.bias) + int64_t(g) * p.ocg : nullptr;

    // A split's block stops at its accumulator: the sum, the bias and the
    // activation all belong to the reduce kernel, which sees every plane.
    float* part = nullptr;
    if (p.k_splits > 1) {
        // blockIdx.z already is zg·splits + split, which is the plane index.
        part = p.partial + int64_t(blockIdx.z) * int64_t(M) * N;
    }

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        auto coord = tCcC(i);
        const int m = int(get<0>(coord));
        const int n = int(get<1>(coord));
        if (m >= M || n >= N) { continue; }
        if (part != nullptr) {
            part[int64_t(m) * N + n] = float(tCrC(i));
            continue;
        }

        float v = float(tCrC(i));
        if (bias != nullptr) { v += gemm_to_float(bias[n]); }
        v = fminf(fmaxf(v, p.min_clip), p.max_clip);
        v = gemm_apply_epilogue(p.epilogue, p.ep_alpha, p.ep_beta, v);
        conv_store<Elem>(p, out_base, m, n, v);
    }
}

// ============================================================
// f32 SIMT kernel
// ============================================================

template <bool ChannelsLast>
__global__ __launch_bounds__(kConvThreads)
void conv_gemm_f_kernel(ConvParams p)
{
    using namespace cute;
    using Elem = float;

    constexpr int bM = 64;
    constexpr int bN = 64;
    constexpr int bK = 16;
    constexpr int kStages = 3;
    constexpr int kPad = 4;   // 16 B: a cp.async destination row start must be 16 B aligned

    const int M = p.oh * p.ow;
    const int N = p.ocg;
    const int K = p.icg * p.kh * p.kw;
    const int m0 = int(blockIdx.x) * bM;
    const int n0 = int(blockIdx.y) * bN;

    const ConvBlock blk = conv_block(p);
    const int zb = blk.zb;
    const int g = blk.g;

    const int64_t in_base = int64_t(zb) * p.in_batch + int64_t(g) * p.icg * p.in_c;
    const int64_t out_base = int64_t(zb) * p.out_batch + int64_t(g) * p.ocg * p.out_c;

    Elem const* W = static_cast<Elem const*>(p.weight) +
                    int64_t(g) * p.ocg * p.w_n;
    // Shifted once so the fill's own arithmetic stays 32-bit (the launch gate
    // keeps every offset it forms inside int).
    Elem const* in = static_cast<Elem const*>(p.input) + in_base;
    const ConvRows rows = conv_rows<Elem, kConvThreads, bM, bK>(p, m0);

    Tensor mB = [&] {
        if constexpr (ChannelsLast) {
            return make_tensor(make_gmem_ptr(W), make_shape(N, K),
                               make_stride(Int<1>{}, int(p.w_ic)));
        } else {
            return make_tensor(make_gmem_ptr(W), make_shape(N, K),
                               make_stride(int(p.w_n), Int<1>{}));
        }
    }();

    auto cta_tiler = make_shape(Int<bM>{}, Int<bN>{}, Int<bK>{});
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});

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

    TiledCopy copy_b = [&] {
        if constexpr (ChannelsLast) {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Elem>{},
                Layout<Shape<_16, _8>>{},
                Layout<Shape<_4, _1>>{});
        } else {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Elem>{},
                Layout<Shape<_32, _4>, Stride<_4, _1>>{},
                Layout<Shape<_1, _4>>{});
        }
    }();

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);
    Tensor tBsB = thr_copy_b.partition_D(sB);

    const bool n_full = (n0 + bN <= N);

    auto load_a = [&](int kt, int stage) {
        if constexpr (ChannelsLast) {
            if (p.fill_vec) {
                conv_fill_a_vec<Elem, bK>(p, sA, stage, in, rows, kt * bK);
                return;
            }
        }
        conv_fill_a<Elem, ChannelsLast, kConvThreads, bM, bK>(
            p, sA, stage, in, m0, kt * bK);
    };
    auto load_b = [&](int kt, int stage) {
        if (p.weight_vec && n_full && (kt + 1) * bK <= K) {
            copy(copy_b, tBgB(_, _, _, kt), tBsB(_, _, _, stage));
            return;
        }
        const int k0 = kt * bK;
        for (int i = threadIdx.x; i < bN * bK; i += blockDim.x) {
            const int row = i / bK;
            const int col = i - row * bK;
            const int n = n0 + row;
            const int k = k0 + col;
            sB(row, col, stage) =
                (n < N && k < K) ? conv_weight<Elem, ChannelsLast>(p, W, n, k)
                                 : Elem(0);
        }
    };

    int k_tile_count = 0;
    int k_tile_next = 0;
    conv_split_range(K, bK, p.k_splits, blk.split, k_tile_next, k_tile_count);

    CUTE_UNROLL
    for (int pipe = 0; pipe < kStages - 1; ++pipe) {
        load_a(k_tile_next, pipe);
        load_b(k_tile_next, pipe);
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }
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
    while (k_tile_count > -(kStages - 1)) {
        cp_async_wait<kStages - 2>();
        __syncthreads();

        load_a(k_tile_next, smem_pipe_write);
        load_b(k_tile_next, smem_pipe_write);
        cp_async_fence();

        copy(tCsA(_, _, _, smem_pipe_read), tCrA);
        copy(tCsB(_, _, _, smem_pipe_read), tCrB);
        gemm(mma, tCrA, tCrB, tCrC);

        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }
        smem_pipe_write = smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == kStages - 1) ? 0 : smem_pipe_read + 1;
    }

    Elem const* bias = p.bias != nullptr
        ? static_cast<Elem const*>(p.bias) + int64_t(g) * p.ocg : nullptr;

    // A split's block stops at its accumulator: the sum, the bias and the
    // activation all belong to the reduce kernel, which sees every plane.
    float* part = nullptr;
    if (p.k_splits > 1) {
        // blockIdx.z already is zg·splits + split, which is the plane index.
        part = p.partial + int64_t(blockIdx.z) * int64_t(M) * N;
    }

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        auto coord = tCcC(i);
        const int m = int(get<0>(coord));
        const int n = int(get<1>(coord));
        if (m >= M || n >= N) { continue; }
        if (part != nullptr) {
            part[int64_t(m) * N + n] = tCrC(i);
            continue;
        }

        float v = tCrC(i);
        if (bias != nullptr) { v += bias[n]; }
        v = fminf(fmaxf(v, p.min_clip), p.max_clip);
        v = gemm_apply_epilogue(p.epilogue, p.ep_alpha, p.ep_beta, v);
        conv_store<Elem>(p, out_base, m, n, v);
    }
}

// ============================================================
// split-K reduction
// ============================================================

/// Sum the k-split planes of one conv and run the epilogue once, exactly as the
/// single-pass kernel would have. One thread per (pixel, channel), planes in
/// split order so a split count is deterministic. The output goes through the
/// operator's own addressing, since a channel-first row is pitch-padded.
template <class Elem>
__global__ void conv_splitk_reduce_kernel(ConvParams p)
{
    const int M = p.oh * p.ow;
    const int N = p.ocg;
    const int64_t plane = int64_t(M) * N;
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= plane) { return; }

    const int pixel = int(idx / N);
    const int n = int(idx - int64_t(pixel) * N);
    const int zg = int(blockIdx.y);

    const float* src = p.partial + int64_t(zg) * p.k_splits * plane + idx;
    float sum = 0.0f;
    for (int s = 0; s < p.k_splits; ++s) { sum += src[int64_t(s) * plane]; }

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

/// How many k ranges the block grid should be cut into.
///
/// The dense GEMM's rule prices the cut against a streamed A and declines the
/// ones whose f32 planes outweigh the operands; a conv block instead gathers its
/// A tile tap by tap, and the shapes that lose here are the ones that repeat
/// that gather. Measured on this part, the cut pays exactly when
///  - the grid is under `1/wave_div` of a wave, and
///  - each split keeps four k tiles, so the two-deep pipeline prologue is at
///    most a quarter of the split's work (f32 56x56x64 k1 -- 4 k tiles, 2 splits
///    measured 1.7x slower; 9 k tiles at 2 splits, 1.3x faster),
/// and then the count is the largest that still fits one wave, `capacity/tiles`
/// (rounding up instead was measured at -25% on the dense twins). 14x14x256 k3
/// f16 goes 0.075 -> 0.028 ms at four splits, 28x28x128 k5 f32 0.34 -> 0.14 ms.
///
/// Half a wave is too eager at two splits for f16: the 56x56x64 3x3 plane has
/// 49 blocks against 114 resident and a shallow cut already costs it 13%
/// channel-last, because two blocks per k range repeat the gather while the
/// plane was wide enough to hide it. A quarter-wave gate leaves every deeper
/// case untouched (7x7x512 has 8 blocks, 14x14x256 16) and only declines that
/// one.
///
/// `wave_div` is dtype-scaled because that margin is a time, not a shape: the
/// cut costs a fixed two launches and a stream-ordered workspace round trip, so
/// a plane only pays for it when one undivided wave lasts longer than that. The
/// same 49-block f16 plane takes 0.031 ms and loses 13% at two splits, while the
/// f32 twin takes 0.086 ms and gains 20% (0.087 -> 0.070 channel-last), and the
/// gate is the only thing that tells them apart. The f16 kernel therefore needs
/// the quarter-wave gate and f32 the half.
///
/// The smallest cut is left in place at the very shallow end. f16 7x7x512 is
/// 12.8M MACs against a fixed cut cost of about 20 us, and channel-last lands at
/// 0.0184 ms in five runs out of six and 0.0417 ms in the sixth -- the arm where
/// the cut does not cover itself. It still stays: the same layer channel-first
/// takes 0.0440 ms cut against 0.0705 ms whole, every run, and a MAC floor high
/// enough to decline them both (which is what a first attempt did) costs that
/// 1.6x to buy back nothing but the sixth run.
inline int conv_pick_splits(int m, int n, int K, int64_t zg, int bM, int bN, int bK,
                            int capacity, int wave_div)
{
    const int forced = detail::splitk_env();
    const int k_tiles = (K + bK - 1) / bK;
    if (forced == 0) { return 1; }
    if (forced > 1) {
        return std::max(1, std::min({forced, k_tiles, detail::kSplitKMaxSplits}));
    }
    if (capacity <= 0) { return 1; }

    const int64_t tiles = int64_t((m + bM - 1) / bM) * ((n + bN - 1) / bN) * zg;
    if (tiles * wave_div > capacity) { return 1; }

    int64_t splits = std::min<int64_t>(detail::kSplitKMaxSplits, capacity / tiles);
    splits = std::min<int64_t>(splits, k_tiles / 4);
    return splits >= 2 ? int(splits) : 1;
}

/// Cut the reduction into k ranges when the grid is shallower than the resident
/// block count wants, and carve the scratch the split blocks write into. A null
/// workspace means single-pass: either the rule declined or the carve failed
/// (`k_splits` stays 1).
inline detail::GemmWorkspace conv_split_workspace(ConvParams& p, int bM, int bN,
                                                  int bK, int smem_bytes,
                                                  int wave_div, cudaStream_t stream)
{
    const int64_t zg = int64_t(p.batch) * p.groups;
    const int blocks = std::max(1, int(detail::kSmemPerSm / smem_bytes));
    const int64_t plane = int64_t(p.oh) * p.ow * p.ocg;
    int splits = conv_pick_splits(p.oh * p.ow, p.ocg, p.icg * p.kh * p.kw, zg,
                                  bM, bN, bK, detail::device_sm_count() * blocks,
                                  wave_div);

    // gridDim.z is 16-bit, and every split costs one f32 plane of stores.
    const int64_t bytes_per_split = zg * plane * int64_t(sizeof(float));
    while (splits > 1 && (zg * splits > 65535 ||
                          bytes_per_split * splits > detail::kSplitKMaxBytes)) {
        splits /= 2;
    }
    if (splits <= 1) { return {}; }

    const detail::GemmWorkspace ws =
        detail::gemm_workspace_alloc(bytes_per_split * splits, stream);
    if (ws.ptr == nullptr) { return {}; }
    p.k_splits = splits;
    p.partial = static_cast<float*>(ws.ptr);
    return ws;
}

/// Sum the split planes and hand the scratch back. Shared by every tile and
/// both element types.
template <class Elem>
inline void conv_reduce(ConvParams& p, detail::GemmWorkspace ws,
                        cudaStream_t stream)
{
    constexpr int kReduceThreads = 256;
    const int64_t plane = int64_t(p.oh) * p.ow * p.ocg;
    dim3 rgrid(unsigned((plane + kReduceThreads - 1) / kReduceThreads),
               unsigned(p.batch * p.groups), 1);
    conv_splitk_reduce_kernel<Elem><<<rgrid, kReduceThreads, 0, stream>>>(p);
    detail::gemm_workspace_free(ws, stream);
}

template <bool ChannelsLast>
void launch_conv_h(ConvParams& p, cudaStream_t stream)
{
    using namespace cute;
    constexpr int bM = ConvTileH::bM, bN = ConvTileH::bN, bK = ConvTileH::bK;
    constexpr auto la = detail::gemm_smem_layout<true>(
        Int<bM>{}, Int<bK>{}, Int<ConvTileH::stages>{});
    constexpr auto lb = detail::gemm_smem_layout<!ChannelsLast>(
        Int<bN>{}, Int<bK>{}, Int<ConvTileH::stages>{});
    constexpr int smem = int(cosize(la) + cosize(lb)) * int(sizeof(half_t));

    auto kernel = conv_gemm_h_kernel<ChannelsLast>;
    static const cudaError_t cfg = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    (void)cfg;

    // Quarter wave: the f16 block is 2.3x shorter than the f32 one, so its
    // shallow cuts need a deeper grid to cover the same fixed cost.
    const detail::GemmWorkspace ws = conv_split_workspace(
        p, bM, bN, bK, smem, 4, stream);
    dim3 grid(ceil_div(p.oh * p.ow, bM), ceil_div(p.ocg, bN),
              p.batch * p.groups * p.k_splits);
    kernel<<<grid, ConvTileH::threads, smem, stream>>>(p);
    if (p.k_splits > 1) { conv_reduce<half_t>(p, ws, stream); }
}

template <bool ChannelsLast>
void launch_conv_f(ConvParams& p, cudaStream_t stream)
{
    using namespace cute;   // ceil_div
    constexpr int bM = 64, bN = 64, bK = 16, kStages = 3, kPad = 4;
    constexpr int smem =
        (bM * (bK + kPad) + bN * (bK + kPad)) * kStages * int(sizeof(float));

    auto kernel = conv_gemm_f_kernel<ChannelsLast>;
    static const cudaError_t cfg = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    (void)cfg;

    const detail::GemmWorkspace ws =
        conv_split_workspace(p, bM, bN, bK, smem, 2, stream);
    dim3 grid(ceil_div(p.oh * p.ow, bM), ceil_div(p.ocg, bN),
              p.batch * p.groups * p.k_splits);
    kernel<<<grid, kConvThreads, smem, stream>>>(p);
    if (p.k_splits > 1) { conv_reduce<float>(p, ws, stream); }
}

}  // namespace

Status conv2d_cuda(const Conv2DAttributes& attrs, TensorView& output,
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
    if (output.numel() == 0) { return Status::ok; }
    if (attrs.algorithm == ConvAlgorithm::Winograd) { return Status::unsupported; }
    if (is_packed_conv_weight(wgt.layout()) || is_wino_weight(wgt.layout())) {
        return Status::unsupported;
    }

    const bool last = is_channel_last(act.layout());
    if (last != is_channel_last_weight(wgt.layout())) { return Status::unsupported; }

    ConvParams p;
    p.batch = static_cast<int>(act.shape(0));
    p.ic = static_cast<int>(act.shape(1));
    p.ih = static_cast<int>(act.shape(2));
    p.iw = static_cast<int>(act.shape(3));
    p.oc = static_cast<int>(output.shape(1));
    p.oh = static_cast<int>(output.shape(2));
    p.ow = static_cast<int>(output.shape(3));
    p.icg = static_cast<int>(last ? wgt.shape(2) : wgt.shape(1));
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
    // The epilogue contract is activate(clamp(x, min_clip, max_clip)) on every
    // backend; the descriptor's bounds go through as-is.
    p.min_clip = attrs.epilogue.min_clip;
    p.max_clip = attrs.epilogue.max_clip;
    p.add_to = attrs.add_to;

    // The filter's own strides. `w_k_dense` says the k axis (IC/G·KH·KW) is
    // contiguous, which is what lets the kernel pose the filter as a plain
    // [OC/G][K] or [K][OC] matrix; a padded filter row is not, and then B is
    // gathered element by element instead.
    const int64_t karea = int64_t(p.kh) * p.kw;
    if (last) {   // HWIO [KH][KW][IC/G][OC]
        p.w_kh = wgt.stride_elems(0);
        p.w_kw = wgt.stride_elems(1);
        p.w_ic = wgt.stride_elems(2);
        p.w_n = wgt.stride_elems(3);
        p.w_k_dense = p.w_n == 1 && p.w_ic == p.oc &&
                      p.w_kw == int64_t(p.icg) * p.oc &&
                      p.w_kh == int64_t(p.kw) * p.icg * p.oc;
    } else {      // OIHW [OC][IC/G][KH][KW]
        p.w_n = wgt.stride_elems(0);
        p.w_ic = wgt.stride_elems(1);
        p.w_kh = wgt.stride_elems(2);
        p.w_kw = wgt.stride_elems(3);
        p.w_k_dense = p.w_kw == 1 && p.w_kh == p.kw && p.w_ic == karea &&
                      p.w_n == int64_t(p.icg) * karea;
    }

    // A 16 B cp.async down the filter's contiguous axis: OIHW rows are K long,
    // HWIO rows OC, and a group's base moves by a whole number of rows.
    const int64_t esz = static_cast<int64_t>(data_type_size(dt));
    const int64_t row = last ? int64_t(p.oc) * esz : int64_t(p.icg) * karea * esz;
    p.weight_vec = p.w_k_dense &&
                   (reinterpret_cast<uintptr_t>(p.weight) % 16 == 0) &&
                   (row % 16 == 0);

    // The channel-last 16 B A-fill needs four (or eight) whole channels of one
    // tap on a dense channel axis: in_c == 1 puts consecutive channels next to
    // each other, IC/G a multiple of the vector makes a k-group never straddle
    // a tap, and the channel stride is what has to be 16 B aligned.
    const int kGroup = esz == 2 ? 8 : 4;
    p.fill_vec = last && p.in_c == 1 && p.icg % kGroup == 0 &&
                 (p.in_w * esz) % 16 == 0 &&
                 (reinterpret_cast<uintptr_t>(p.input) % 16 == 0);

    // The fill addresses the activation with 32-bit offsets from its block's
    // slice; a tensor past the halfway point of the int range is refused rather
    // than silently wrapping.
    if (p.oh * p.ow > 65535 || p.batch * p.groups > 65535 || p.ocg <= 0 ||
        act.numel() > (int64_t(1) << 30)) {
        return Status::unsupported;
    }

    const cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    if (esz == 2) {
        if (last) { launch_conv_h<true>(p, stream); }
        else      { launch_conv_h<false>(p, stream); }
    } else {
        if (last) { launch_conv_f<true>(p, stream); }
        else      { launch_conv_f<false>(p, stream); }
    }
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

}  // namespace nnops::backend::cuda
