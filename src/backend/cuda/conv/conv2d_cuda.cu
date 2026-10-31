/// @file conv2d_cuda.cu
/// @brief CUDA Conv2D — one implicit-GEMM kernel for both memory orders.
///
/// The convolution is posed the same way for NCHW and NHWC: M is the output
/// pixel axis, N the output channels of one group, K the reduction over
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
/// Shape indices follow the library's convention: they are the logical NCHW
/// ones even for a channel-last tensor, and the layout tag only changes the
/// element strides. The epilogue therefore writes through the operator's own
/// address arithmetic rather than a strided C tensor, because a planar row is
/// padded to a 32 B pitch and the pixel axis is only uniform inside one row.
/// add_to reads the old output at that same address before overwriting it, so
/// the residual lands after the activation as it does on the CPU.
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
};

constexpr int kConvThreads = detail::kGemmThreads;

/// Element (pixel, k) of the gathered activation, zero outside the input.
template <class Elem, bool ChannelsLast>
__device__ inline Elem conv_gather(const ConvParams& p, int64_t base, int k,
                                   int oh, int ow)
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
    const int ih = oh * p.sh - p.ph + kh * p.dh;
    const int iw = ow * p.sw - p.pw + kw * p.dw;
    if (ih < 0 || ih >= p.ih || iw < 0 || iw >= p.iw) { return Elem(0); }
    return static_cast<const Elem*>(p.input)[base + int64_t(ic) * p.in_c +
                                             int64_t(ih) * p.in_h +
                                             int64_t(iw) * p.in_w];
}

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

/// Fill one shared-memory A tile with the gathered activation. Threads spread
/// over the k axis in the channel-last order (consecutive threads read
/// consecutive channels) and over the pixel axis in the planar one (consecutive
/// threads read consecutive columns of a row).
template <class Elem, bool ChannelsLast, class SmemA>
__device__ inline void conv_fill_a(const ConvParams& p, SmemA sA, int stage,
                                   int64_t base, int pixel0, int k0)
{
    constexpr int bM = 64;
    constexpr int bK = (sizeof(Elem) == 2) ? 64 : 16;
    const int M = p.oh * p.ow;
    const int K = p.icg * p.kh * p.kw;
    const int tid = int(threadIdx.x);

    if constexpr (ChannelsLast) {
        constexpr int per = kConvThreads / bK;   // pixels per round
        const int col = tid % bK;
        const int k = k0 + col;
        for (int row = tid / bK; row < bM; row += per) {
            const int pixel = pixel0 + row;
            Elem val = Elem(0);
            if (pixel < M && k < K) {
                const int oh = pixel / p.ow;
                const int ow = pixel - oh * p.ow;
                val = conv_gather<Elem, ChannelsLast>(p, base, k, oh, ow);
            }
            sA(row, col, stage) = val;
        }
    } else {
        constexpr int per = kConvThreads / bM;   // k columns per thread
        const int row = tid % bM;
        const int pixel = pixel0 + row;
        const int oh = pixel / p.ow;
        const int ow = pixel - oh * p.ow;
        const int col0 = (tid / bM) * (bK / per);
        for (int j = 0; j < bK / per; ++j) {
            const int k = k0 + col0 + j;
            Elem val = Elem(0);
            if (pixel < M && k < K) {
                val = conv_gather<Elem, ChannelsLast>(p, base, k, oh, ow);
            }
            sA(row, col0 + j, stage) = val;
        }
    }
}

/// Output element (pixel, channel) of the order the op was asked for: planar
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

template <bool ChannelsLast>
__global__ __launch_bounds__(kConvThreads)
void conv_gemm_h_kernel(ConvParams p)
{
    using namespace cute;
    using Elem = cute::half_t;

    constexpr int bM = 64;
    constexpr int bN = 64;
    constexpr int bK = 64;
    constexpr int kStages = 3;

    const int M = p.oh * p.ow;
    const int N = p.ocg;
    const int K = p.icg * p.kh * p.kw;
    const int m0 = int(blockIdx.x) * bM;
    const int n0 = int(blockIdx.y) * bN;

    const int zb = int(blockIdx.z) / p.groups;
    const int g = int(blockIdx.z) - zb * p.groups;

    const int64_t in_base = int64_t(zb) * p.in_batch + int64_t(g) * p.icg * p.in_c;
    const int64_t out_base = int64_t(zb) * p.out_batch + int64_t(g) * p.ocg * p.out_c;

    Elem const* W = static_cast<Elem const*>(p.weight) +
                    int64_t(g) * p.ocg * p.w_n;

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

    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                                  Layout<Shape<_2, _2, _1>>{},
                                  Tile<_32, _32, _16>{});

    TiledCopy copy_b = [&] {
        if constexpr (ChannelsLast) {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Elem>{},
                Layout<Shape<_8, _16>>{},
                Layout<Shape<_8, _1>>{});
        } else {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Elem>{},
                Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                Layout<Shape<_1, _8>>{});
        }
    }();

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);
    Tensor tBsB = thr_copy_b.partition_D(sB);

    const bool n_full = (n0 + bN <= N);

    auto load_a = [&](int kt, int stage) {
        conv_fill_a<Elem, ChannelsLast>(p, sA, stage, in_base, m0, kt * bK);
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

    int k_tile_count = (K + bK - 1) / bK;
    int k_tile_next = 0;

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

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        auto coord = tCcC(i);
        const int m = int(get<0>(coord));
        const int n = int(get<1>(coord));
        if (m >= M || n >= N) { continue; }

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

    const int zb = int(blockIdx.z) / p.groups;
    const int g = int(blockIdx.z) - zb * p.groups;

    const int64_t in_base = int64_t(zb) * p.in_batch + int64_t(g) * p.icg * p.in_c;
    const int64_t out_base = int64_t(zb) * p.out_batch + int64_t(g) * p.ocg * p.out_c;

    Elem const* W = static_cast<Elem const*>(p.weight) +
                    int64_t(g) * p.ocg * p.w_n;

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
        conv_fill_a<Elem, ChannelsLast>(p, sA, stage, in_base, m0, kt * bK);
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

    int k_tile_count = (K + bK - 1) / bK;
    int k_tile_next = 0;

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

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        auto coord = tCcC(i);
        const int m = int(get<0>(coord));
        const int n = int(get<1>(coord));
        if (m >= M || n >= N) { continue; }

        float v = tCrC(i);
        if (bias != nullptr) { v += bias[n]; }
        v = fminf(fmaxf(v, p.min_clip), p.max_clip);
        v = gemm_apply_epilogue(p.epilogue, p.ep_alpha, p.ep_beta, v);
        conv_store<Elem>(p, out_base, m, n, v);
    }
}

// ============================================================
// Launch
// ============================================================

template <bool ChannelsLast>
void launch_conv_h(const ConvParams& p, cudaStream_t stream)
{
    using namespace cute;
    constexpr int bM = 64, bN = 64, bK = 64, kStages = 3;
    constexpr auto la = detail::gemm_smem_layout<true>(
        Int<bM>{}, Int<bK>{}, Int<kStages>{});
    constexpr auto lb = detail::gemm_smem_layout<!ChannelsLast>(
        Int<bN>{}, Int<bK>{}, Int<kStages>{});
    constexpr int smem = int(cosize(la) + cosize(lb)) * int(sizeof(half_t));

    auto kernel = conv_gemm_h_kernel<ChannelsLast>;
    static const cudaError_t cfg = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    (void)cfg;

    dim3 grid(ceil_div(p.oh * p.ow, bM), ceil_div(p.ocg, bN), p.batch * p.groups);
    kernel<<<grid, kConvThreads, smem, stream>>>(p);
}

template <bool ChannelsLast>
void launch_conv_f(const ConvParams& p, cudaStream_t stream)
{
    using namespace cute;   // ceil_div
    constexpr int bM = 64, bN = 64, bK = 16, kStages = 3, kPad = 4;
    constexpr int smem =
        (bM * (bK + kPad) + bN * (bK + kPad)) * kStages * int(sizeof(float));

    auto kernel = conv_gemm_f_kernel<ChannelsLast>;
    static const cudaError_t cfg = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    (void)cfg;

    dim3 grid(ceil_div(p.oh * p.ow, bM), ceil_div(p.ocg, bN), p.batch * p.groups);
    kernel<<<grid, kConvThreads, smem, stream>>>(p);
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
    p.add_to = attrs.add_to;

    // Conv folds Relu/Relu6 into the store clamp and ignores epilogue's own
    // min_clip/max_clip (they are MatMul's); the CPU conv does the same.
    switch (attrs.epilogue.type) {
    case EpilogueActivateType::Relu:  p.min_clip = 0.0f; break;
    case EpilogueActivateType::Relu6: p.min_clip = 0.0f; p.max_clip = 6.0f; break;
    default: break;
    }

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

    if (p.oh * p.ow > 65535 || p.batch * p.groups > 65535 || p.ocg <= 0) {
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
