/// @file conv2d.cu
/// @brief CUDA Conv2D — one implicit-GEMM kernel for channel-last activations.
///
/// The convolution is posed as a GEMM: M is the output pixel axis, N the output
/// channels of one group, K the reduction over IC/G × KH × KW. The activation is
/// the gathered operand and therefore lives on A, whose shared-memory tile is
/// filled tap by tap with a bounds check that zeroes the padding ring; the filter
/// is B, read straight out of global memory. A channel-last activation carries
/// the channels innermost, so the filter is HWIO and the k-flattening is
/// channel-major inside a tap (k = tap·ICG + ic); the filter's copy atom and the
/// gather walk are both built for that order.
///
/// Shape indices follow the library's convention: they are the logical [N, C, H,
/// W] ones, and the layout tag only changes the element strides. The epilogue
/// writes through the operator's own address arithmetic rather than a strided C
/// tensor, because a channel-last row is padded to a 32 B pitch and the pixel
/// axis is only uniform inside one row. add_to reads the old output at that same
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
/// f32, f16 and bf16 in the dense kernels; a Winograd weight never reaches them
/// (the dispatch refuses it) and a prepacked filter is opaque to them. The
/// quantized family (an s8 filter against a pre-quantized u8/s8 activation) is
/// its own kernel at the bottom of this file: 16x8x32 s8 MMA over the same CTA
/// tile, channel-last only, MatMulInteger compensation with the per-pixel
/// activation sums folded in during the gather.

#include "../cuda_common.cuh"
#include "../matmul/gemm_common.cuh"
#include "../matmul/matmul_f16.cuh"
#include "../matmul/matmul_f32.cuh"
#include "../matmul/matmul_i8_common.cuh" // I8OutKind + the shared int8 epilogue pieces
#include "../matmul/matmul_i8_ptx.cuh"
#include "conv_panel.cuh"
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
    bool weight_vec = false;
    bool fill_vec = false;

    // ---- split-K ----
    // Same form as the dense GEMM's: blockIdx.z carries (batch·group)·splits +
    // split, each block owns a proportional range of the k tiles and writes its
    // f32 accumulator tile to `partial` instead of running the epilogue, and a
    // second kernel sums the planes and epilogues once.
    int k_splits = 1;
    float* partial = nullptr; ///< [(batch·group)][split][M][OC/G], row-major

    // ---- int8 route (channel-last, pre-quantized activation) --------------
    // The activation's parameters are per-tensor; the filter's per-channel
    // arrays are device copies staged by the host entry. R_b is the per-output-
    // channel filter sum the compensation needs, also device-resident.
    int32_t zp_x = 0;
    float sc_x = 1.0f;
    const int32_t* zp_w = nullptr;
    int32_t zp_w_scalar = 0;
    const float* sc_w = nullptr;
    float sc_w_scalar = 1.0f;
    int32_t* r_b = nullptr;
    float out_scale = 1.0f;
    int32_t out_zp = 0;
    int out_kind = 0;  ///< I8OutKind
    bool relu = false; ///< integer destinations: Relu before the store
};

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

/// The rows of an A tile one thread feeds on the channel-last vector walk, and
/// the two pixel coordinates each of them starts from. The thread-to-(row,
/// k-group) mapping is fixed for the whole kernel, so the division that turns a
/// pixel into (oh, ow) is paid once instead of once per k tile, and `oh0`/`ow0`
/// already carry the stride and the negative padding: a tap is then one add.
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
__device__ inline void conv_split_range(int K, int bK, int splits, int split, int& kt_begin,
                                        int& kt_count)
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
/// over the k axis in the channel-last order (consecutive threads read
/// consecutive channels).
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

/// Output element (pixel, channel): channel-last puts the channel in the
/// innermost slot.
template <class Elem>
__device__ inline Elem* conv_out_addr(const ConvParams& p, int64_t base, int pixel, int n)
{
    const int oh = pixel / p.ow;
    const int ow = pixel - oh * p.ow;
    return static_cast<Elem*>(p.output) + base + int64_t(oh) * p.out_h + int64_t(ow) * p.out_w +
           int64_t(n) * p.out_c;
}

/// Post-accumulation store, including the residual an add_to conv adds after
/// the activation.
template <class Elem>
__device__ inline void conv_store(const ConvParams& p, int64_t base, int pixel, int n, float v)
{
    Elem* dst = conv_out_addr<Elem>(p, base, pixel, n);
    if (p.add_to) {
        v += float(*dst);
    }
    *dst = Elem(v);
}

// ============================================================
// f16 tensor-core kernel
// ============================================================

/// The f16/bf16 conv tile: 64 pixels by 64 channels, 64 halves of k, two stages.
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
using ConvShapeH = ManualHShape<32, 32, 2, 2, 64, 2, 1, false, 1>;

/// The gather's (row, col, stage) indexer over the k-major swizzled panel.
template <class S, class Elem> using ConvHPanel = conv::KMajorPanel<S, Elem>;

template <class Elem>
__global__ __launch_bounds__(ConvShapeH::threads, ConvShapeH::MinBlocks)
void conv_gemm_h_kernel(ConvParams p)
{
    using S = ConvShapeH;
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
            conv_fill_a_vec<Elem, BK>(p, ConvHPanel<S, Elem>{sA}, stage, in, rows, k0);
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
                cp_async_ca_16B(&sB[manual_swz_n<S>(kr, nv * 8)],
                                W + int64_t(n0 + nv * 8) * p.w_n + int64_t(k0 + kr) * p.w_ic);
            }
        } else {
            for (int i = tid; i < BN * BK; i += S::threads) {
                const int row = i / BK;
                const int col = i - row * BK;
                const int n = n0 + row;
                const int k = k0 + col;
                sB[manual_swz_n<S>(col, row)] =
                    (n < N && k < K) ? conv_weight<Elem>(p, W, n, k) : manual_zero<Elem>();
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
        manual_gemm_h_tile<S, Elem, false, false>(sAbase + stage * S::kAMax,
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
                    v = gemm_apply_epilogue(p.epilogue, p.ep_alpha, p.ep_beta, v);
                    conv_store<Elem>(p, out_base, m, n, v);
                }
            }
        }
    }
}

// ============================================================
// f32 SIMT kernel
// ============================================================

/// The f32 tile: 8x8 per lane, a three-deep ring, and 16 elements of k a stage.
using ConvShapeF = ManualFShape<4, 8, 2, 2, 16, 3>;

/// The A panel in float units: rows of `BK + Pad`, the padding being what keeps
/// a 16 B gather vector 16 B aligned on every row.
using ConvFPanel = conv::PaddedPanel<ConvShapeF::LDA>;

__global__ __launch_bounds__(ConvShapeF::threads, ConvShapeF::MinBlocks)
void conv_gemm_f_kernel(ConvParams p)
{
    using S = ConvShapeF;
    constexpr int BM = S::BM;
    constexpr int BN = S::BN;
    constexpr int BK = S::BK;
    constexpr int ST = S::stages;
    constexpr int KU = S::KU;

    extern __shared__ __align__(16) unsigned char conv_smem[];
    float* const sAbase = reinterpret_cast<float*>(conv_smem);
    float* const sBbase = sAbase + ST * S::kAMax;

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

    auto fill = [&](int kt, int stage) {
        if (kt >= kt_end) {
            cp_async_commit_group();
            return;
        }
        const int k0 = kt * BK;
        float* const sA = sAbase + stage * S::kAMax;
        float* const sB = sBbase + stage * S::kBMax;
        if (p.fill_vec) {
            conv_fill_a_vec<float, BK>(p, ConvFPanel{sA}, stage, in, rows, k0);
        } else {
            conv_fill_a<float, S::threads, BM, BK>(p, ConvFPanel{sA}, stage, in, m0, k0);
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
        manual_gemm_f_tile<S, false, false>(sAbase + stage * S::kAMax,
                                            sBbase + stage * S::kBMax, warp_m, warp_n, lane, acc);
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
            v = gemm_apply_epilogue(p.epilogue, p.ep_alpha, p.ep_beta, v);
            conv_store<float>(p, out_base, m, n, v);
        }
    }
}

// ============================================================
// split-K reduction
// ============================================================

/// Sum the k-split planes of one conv and run the epilogue once, exactly as the
/// single-pass kernel would have. One thread per (pixel, channel), planes in
/// split order so a split count is deterministic. The output goes through the
/// operator's own addressing, since a channel-last row is pitch-padded.
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
/// the cut does not cover itself. It still stays: the same layer in the previous
/// channel-first order took 0.0440 ms cut against 0.0705 ms whole, every run,
/// and a MAC floor high enough to decline them both (which is what a first
/// attempt did) costs that 1.6x to buy back nothing but the sixth run.
inline int conv_pick_splits(int m, int n, int K, int64_t zg, int bM, int bN, int bK, int capacity,
                            int wave_div)
{
    const int forced = detail::splitk_env();
    const int k_tiles = (K + bK - 1) / bK;
    if (forced == 0) {
        return 1;
    }
    if (forced > 1) {
        return std::max(1, std::min({forced, k_tiles, detail::kSplitKMaxSplits}));
    }
    if (capacity <= 0) {
        return 1;
    }

    const int64_t tiles = int64_t((m + bM - 1) / bM) * ((n + bN - 1) / bN) * zg;
    if (tiles * wave_div > capacity) {
        return 1;
    }

    int64_t splits = std::min<int64_t>(detail::kSplitKMaxSplits, capacity / tiles);
    splits = std::min<int64_t>(splits, k_tiles / 4);
    return splits >= 2 ? int(splits) : 1;
}

/// Cut the reduction into k ranges when the grid is shallower than the resident
/// block count wants, and carve the scratch the split blocks write into. A null
/// workspace means single-pass: either the rule declined or the carve failed
/// (`k_splits` stays 1).
inline detail::GemmWorkspace conv_split_workspace(ConvParams& p, int bM, int bN, int bK,
                                                  int smem_bytes, int wave_div, cudaStream_t stream)
{
    const int64_t zg = int64_t(p.batch) * p.groups;
    const int blocks = std::max(1, int(detail::kSmemPerSm / smem_bytes));
    const int64_t plane = int64_t(p.oh) * p.ow * p.ocg;
    int splits = conv_pick_splits(p.oh * p.ow, p.ocg, p.icg * p.kh * p.kw, zg, bM, bN, bK,
                                  detail::device_sm_count() * blocks, wave_div);

    // gridDim.z is 16-bit, and every split costs one f32 plane of stores.
    const int64_t bytes_per_split = zg * plane * int64_t(sizeof(float));
    while (splits > 1 &&
           (zg * splits > 65535 || bytes_per_split * splits > detail::kSplitKMaxBytes)) {
        splits /= 2;
    }
    if (splits <= 1) {
        return {};
    }

    const detail::GemmWorkspace ws = detail::gemm_workspace_alloc(bytes_per_split * splits, stream);
    if (ws.ptr == nullptr) {
        return {};
    }
    p.k_splits = splits;
    p.partial = static_cast<float*>(ws.ptr);
    return ws;
}

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

template <class Elem> void launch_conv_h(ConvParams& p, cudaStream_t stream)
{
    using S = ConvShapeH;
    constexpr int smem = S::kSmemBytes;
    auto kernel = conv_gemm_h_kernel<Elem>;
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

inline void launch_conv_f(ConvParams& p, cudaStream_t stream)
{
    using S = ConvShapeF;
    constexpr int smem = S::stages * (S::kAMax + S::kBMax) * int(sizeof(float));
    auto kernel = conv_gemm_f_kernel;
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

// ============================================================
// int8 tensor-core kernel
// ============================================================

/// The int8 tile: 64 pixels by 64 channels, 64 bytes of k, three stages. The s8
/// panels are half the fp16 footprint, so a third stage still leaves four blocks
/// a SM.
using ConvShapeI8 = ManualI8Shape<32, 32, 2, 2, 64, 3>;

/// Either k-major s8 panel, as the gather's (row, col, stage) indexer. The base
/// already carries the stage; rows are `LD` bytes apart.
template <class Elem>
struct ConvI8Panel {
    Elem* base;
    __device__ __forceinline__ Elem& operator()(int row, int col, int) const
    {
        return base[row * ConvShapeI8::LD + col];
    }
};

/// Channel-last gather for the int8 route. Each thread owns one output pixel's
/// tile row, so the same walk returns that thread's share of R_a[pixel]; the
/// pixel a padded or out-of-range tap contributes is the real value 0, which is
/// also what it adds to the sum.
template <class Elem, int Threads, int bM, int bK, class SmemA>
__device__ inline int32_t conv_fill_a_i8(const ConvParams& p, SmemA sA, int stage, const Elem* in,
                                         int pixel0, int k0)
{
    const int M = p.oh * p.ow;
    const int K = p.icg * p.kh * p.kw;
    const int tid = int(threadIdx.x);
    const int inc = int(p.in_c), inh = int(p.in_h), inw = int(p.in_w);

    constexpr int per = Threads / bM; // k columns per thread
    const int row = tid % bM;
    const int pixel = pixel0 + row;
    const bool m_ok = pixel < M;
    const int oh = pixel / p.ow;
    const int ow = pixel - oh * p.ow;
    const int col0 = (tid / bM) * (bK / per);
    int k = k0 + col0;
    // k = tap*IC/G + ic: the channels are innermost inside one tap.
    const int tap = k / p.icg;
    int ic = k - tap * p.icg;
    int kh = tap / p.kw;
    int kw = tap - kh * p.kw;

    int32_t sum = 0;
    for (int j = 0; j < bK / per; ++j) {
        const int ih = oh * p.sh - p.ph + kh * p.dh;
        const int iw = ow * p.sw - p.pw + kw * p.dw;
        // A window-padded tap is the real value 0, stored as the activation's
        // zero point -- what the CPU im2col does. Filling 0 instead would leave
        // -(zp_x)*(qw - zp_w) per padded tap in the compensated sum.
        Elem val = Elem(0);
        if (m_ok && k < K) {
            val = (unsigned(ih) < unsigned(p.ih) && unsigned(iw) < unsigned(p.iw)) ?
                      in[ic * inc + ih * inh + iw * inw] :
                      Elem(p.zp_x);
        }
        sA(row, col0 + j, stage) = val;
        sum += int32_t(val);
        ++k;
        if (++ic == p.icg) {
            ic = 0;
            if (++kw == p.kw) {
                kw = 0;
                ++kh;
            }
        }
    }
    return sum;
}

/// Channel-last gather, one 16 B vector per (row, k-group): 16 consecutive k are
/// 16 consecutive channels inside one tap, contiguous in the activation and in
/// the shared tile. A thread owns `rows.count` pixels, so it keeps one partial
/// per owned row and folds them into shared memory once, after the k loop.
template <class Elem, int bK, class SmemA>
__device__ inline void conv_fill_a_vec_i8(const ConvParams& p, SmemA sA, int stage, const Elem* in,
                                          const ConvRows& rows, int k0, int32_t* sums, bool add)
{
    constexpr int kGroup = 16 / int(sizeof(Elem));
    using Vec = uint4;

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
    // A padded tap carries the activation's zero point, so the compensation
    // cancels it exactly (see conv_fill_a_i8).
    Vec zpv = zero;
    {
        Elem* e = reinterpret_cast<Elem*>(&zpv);
    #pragma unroll
        for (int t = 0; t < kGroup; ++t) {
            e[t] = Elem(p.zp_x);
        }
    }

#pragma unroll
    for (int i = 0; i < ConvRows::kMax; ++i) {
        if (i >= rows.count) {
            break;
        }
        Vec v = zero;
        if (k_ok && rows.ok[i]) {
            const int ih = rows.oh0[i] + dh;
            const int iw = rows.ow0[i] + dw;
            v = (unsigned(ih) < unsigned(p.ih) && unsigned(iw) < unsigned(p.iw)) ?
                    *reinterpret_cast<const Vec*>(in + koff + rows.off[i]) :
                    zpv;
        }
        *reinterpret_cast<Vec*>(&sA(rows.row[i], kg * kGroup, stage)) = v;
        if (add) {
            const Elem* e = reinterpret_cast<const Elem*>(&v);
            int32_t sum = 0;
        #pragma unroll
            for (int t = 0; t < kGroup; ++t) {
                sum += int32_t(e[t]);
            }
            sums[i] += sum;
        }
    }
}

/// R_b[oc] = Σ_k filter[oc,k] over the whole reduction, per global channel.
__global__ void conv_i8_col_sums_kernel(ConvParams p)
{
    const int oc = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (oc >= p.oc) {
        return;
    }
    const int g = oc / p.ocg;
    const int ocl = oc - g * p.ocg;
    const int8_t* w =
        static_cast<const int8_t*>(p.weight) + int64_t(g) * p.ocg * p.w_n + int64_t(ocl) * p.w_n;
    int32_t s = 0;
    for (int ic = 0; ic < p.icg; ++ic) {
        for (int kh = 0; kh < p.kh; ++kh) {
            for (int kw = 0; kw < p.kw; ++kw) {
                s += int32_t(w[ic * p.w_ic + kh * p.w_kh + kw * p.w_kw]);
            }
        }
    }
    p.r_b[oc] = s;
}

template <class AElem>
__global__ __launch_bounds__(ConvShapeI8::threads, ConvShapeI8::MinBlocks)
void conv_gemm_i8_kernel(ConvParams p)
{
    using S = ConvShapeI8;
    using BElem = int8_t;
    constexpr int BM = S::BM;
    constexpr int BN = S::BN;
    constexpr int BK = S::BK;
    constexpr int ST = S::stages;
    constexpr int LD = S::LD;

    extern __shared__ __align__(16) unsigned char conv_smem[];
    int8_t* const sAbase = reinterpret_cast<int8_t*>(conv_smem);
    int8_t* const sBbase = sAbase + ST * S::kABytes;

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
    const int gi = g * p.ocg;

    const int64_t in_base = int64_t(zb) * p.in_batch + int64_t(g) * p.icg * p.in_c;
    const int64_t out_base = int64_t(zb) * p.out_batch + int64_t(g) * p.ocg * p.out_c;

    BElem const* W = static_cast<BElem const*>(p.weight) + int64_t(g) * p.ocg * p.w_n;
    AElem const* in = static_cast<AElem const*>(p.input) + in_base;

    // The per-pixel activation sums, folded into shared memory after the k
    // loop. The scalar fill gives one partial per thread (each owns one row);
    // the vector fill owns several rows, so it keeps an array.
    __shared__ int32_t s_ra[BM];
    if (tid < BM) {
        s_ra[tid] = 0;
    }
    __syncthreads();
    const ConvRows rows = conv_rows<AElem, S::threads, BM, BK>(p, m0);
    int32_t ra_single = 0;
    int32_t ra_vec[ConvRows::kMax] = {};
    int ra_kt = -1;

    const int k_tiles = (K + BK - 1) / BK;

    auto fill = [&](int kt, int stage) {
        if (kt >= k_tiles) {
            cp_async_commit_group();
            return;
        }
        // A k tile is gathered once; the guard keeps the fold that way if the
        // ring is ever allowed to re-issue a tile.
        const bool add = (kt != ra_kt);
        if (add) {
            ra_kt = kt;
        }
        const int k0 = kt * BK;
        int8_t* const sA = sAbase + stage * S::kABytes;
        int8_t* const sB = sBbase + stage * S::kBBytes;
        if (p.fill_vec) {
            conv_fill_a_vec_i8<AElem, BK>(p, ConvI8Panel<AElem>{reinterpret_cast<AElem*>(sA)},
                                          stage, in, rows, k0, ra_vec, add);
        } else {
            const int32_t s =
                conv_fill_a_i8<AElem, S::threads, BM, BK>(
                    p, ConvI8Panel<AElem>{reinterpret_cast<AElem*>(sA)}, stage, in, m0, k0);
            if (add) {
                ra_single += s;
            }
        }
        // The filter is HWIO with the channel innermost, so its k axis is not
        // contiguous and B is gathered byte-wise into a k-major panel.
        for (int i = tid; i < BN * BK; i += S::threads) {
            const int row = i / BK;
            const int col = i - row * BK;
            const int n = n0 + row;
            const int k = k0 + col;
            sB[row * LD + col] = (n < N && k < K) ? conv_weight<BElem>(p, W, n, k) : BElem(0);
        }
        cp_async_commit_group();
    };

    int32_t acc[S::MA][S::NA][4];
#pragma unroll
    for (int i = 0; i < S::MA; ++i)
#pragma unroll
        for (int j = 0; j < S::NA; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                acc[i][j][t] = 0;
            }

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
        manual_gemm_i8_tile<S, AElem>(sAbase + stage * S::kABytes, sBbase + stage * S::kBBytes,
                                      warp_m, warp_n, lane, acc);
    }

    if (p.fill_vec) {
        for (int i = 0; i < rows.count; ++i) {
            atomicAdd(&s_ra[rows.row[i]], ra_vec[i]);
        }
    } else {
        atomicAdd(&s_ra[tid % BM], ra_single);
    }
    __syncthreads();

    // ---- output ---------------------------------------------------------
    const I8OutKind kind = static_cast<I8OutKind>(p.out_kind);
    const bool int_out = kind == I8OutKind::S32 || kind == I8OutKind::S8 || kind == I8OutKind::U8;
    const int32_t* bias32 =
        int_out && p.bias != nullptr ? static_cast<const int32_t*>(p.bias) + gi : nullptr;
    const float* biasf32 = kind == I8OutKind::F32 && p.bias != nullptr ?
                               static_cast<const float*>(p.bias) + gi :
                               nullptr;
    const __half* biasf16 = kind == I8OutKind::F16 && p.bias != nullptr ?
                                static_cast<const __half*>(p.bias) + gi :
                                nullptr;

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
                    const int32_t zpw = p.zp_w != nullptr ? p.zp_w[gi + n] : p.zp_w_scalar;
                    int32_t v = acc[i][j][rr * 2 + cc];
                    v -= p.zp_x * p.r_b[gi + n];
                    // `m` is the global pixel; the sums are indexed by tile row.
                    v -= (s_ra[m - m0] - K * p.zp_x) * zpw;

                    switch (kind) {
                    case I8OutKind::S32:
                        // Refused by the entry: the raw int32 destination is
                        // MatMul's.
                        break;
                    case I8OutKind::S8:
                    case I8OutKind::U8: {
                        if (bias32 != nullptr) {
                            v += bias32[n];
                        }
                        if (p.relu) {
                            v = v < 0 ? 0 : v;
                        }
                        const float s_w = p.sc_w != nullptr ? p.sc_w[gi + n] : p.sc_w_scalar;
                        const float row_k = s_w / p.out_scale;
                        if (kind == I8OutKind::S8) {
                            *conv_out_addr<int8_t>(p, out_base, m, n) = detail::i8_requant<int8_t>(
                                v, row_k, p.sc_x, float(p.out_zp));
                        } else {
                            *conv_out_addr<uint8_t>(p, out_base, m, n) = detail::i8_requant<uint8_t>(
                                v, row_k, p.sc_x, float(p.out_zp));
                        }
                        break;
                    }
                    case I8OutKind::F32:
                    case I8OutKind::F16: {
                        const float s_w = p.sc_w != nullptr ? p.sc_w[gi + n] : p.sc_w_scalar;
                        float f = float(v) * (s_w * p.sc_x);
                        float b = 0.0f;
                        if (biasf32 != nullptr) {
                            b = biasf32[n];
                        } else if (biasf16 != nullptr) {
                            b = __half2float(biasf16[n]);
                        }
                        if (b != 0.0f) {
                            f += b;
                        }
                        f = fminf(fmaxf(f, p.min_clip), p.max_clip);
                        f = gemm_apply_epilogue(p.epilogue, p.ep_alpha, p.ep_beta, f);
                        if (kind == I8OutKind::F32) {
                            *conv_out_addr<float>(p, out_base, m, n) = f;
                        } else {
                            *conv_out_addr<__half>(p, out_base, m, n) = __float2half(f);
                        }
                        break;
                    }
                    }
                }
            }
        }
    }
}

template <class AElem> void launch_conv_i8(ConvParams& p, cudaStream_t stream)
{
    using S = ConvShapeI8;
    constexpr int smem = S::kSmemBytes;
    auto kernel = conv_gemm_i8_kernel<AElem>;
    static const cudaError_t cfg =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    (void)cfg;

    dim3 grid(unsigned((p.oh * p.ow + S::BM - 1) / S::BM),
              unsigned((p.ocg + S::BN - 1) / S::BN), unsigned(p.batch * p.groups));
    kernel<<<grid, S::threads, smem, stream>>>(p);
}

/// The quantized route's CUDA entry: pre-quantized u8/s8 activation against an
/// s8 HWIO filter, channel-last, MatMulInteger-shaped output. The weight's
/// per-channel parameters are host arrays by contract and staged here, exactly
/// as the dense GEMM does.
Status conv2d_i8_cuda(const Conv2DAttributes& attrs, TensorView& output,
                      std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const TensorView& act = inputs[0];
    const TensorView& wgt = inputs[1];
    const TensorView* bias = inputs.size() > 2 ? &inputs[2] : nullptr;

    const DataType adt = act.data_type();
    if (adt != DataType::u8 && adt != DataType::s8) {
        return Status::unsupported;
    }
    if (wgt.data_type() != DataType::s8) {
        return Status::unsupported;
    }
    if (attrs.algorithm == ConvAlgorithm::Winograd) {
        return Status::unsupported;
    }
    if (is_packed_conv_weight(wgt.layout()) || is_wino_weight(wgt.layout())) {
        return Status::unsupported;
    }
    // The channel-last route reads 16 whole channels per vector and gathers the
    // HWIO filter byte-wise (its k axis is not contiguous).
    if (attrs.add_to) {
        return Status::unsupported;
    }

    I8OutKind kind;
    if (!i8_out_kind(output.data_type(), kind) || kind == I8OutKind::S32) {
        return Status::unsupported;
    }

    ConvParams p;
    p.batch = static_cast<int>(act.shape(0));
    p.ic = static_cast<int>(act.shape(1));
    p.ih = static_cast<int>(act.shape(2));
    p.iw = static_cast<int>(act.shape(3));
    p.oc = static_cast<int>(output.shape(1));
    p.oh = static_cast<int>(output.shape(2));
    p.ow = static_cast<int>(output.shape(3));
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
    p.in_c = act.stride_elems(1);
    p.in_h = act.stride_elems(2);
    p.in_w = act.stride_elems(3);
    const int64_t esz = static_cast<int64_t>(data_type_size(output.data_type()));
    (void)esz;
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
    p.add_to = false;

    // HWIO [KH][KW][IC/G][OC]: n is innermost, k is not contiguous.
    p.w_kh = wgt.stride_elems(0);
    p.w_kw = wgt.stride_elems(1);
    p.w_ic = wgt.stride_elems(2);
    p.w_n = wgt.stride_elems(3);
    p.w_k_dense = p.w_n == 1 && p.w_ic == p.oc && p.w_kw == int64_t(p.icg) * p.oc &&
                  p.w_kh == int64_t(p.kw) * p.icg * p.oc;
    p.weight_vec = false;

    // The channel-last 16 B A-fill needs a dense channel axis and 16 channels of
    // one tap per vector (k/groups is tap-major there).
    const int kGroup = 16; // one 16 B vector of s8
    p.fill_vec = p.in_c == 1 && p.icg % kGroup == 0 && (p.in_w % 16 == 0) &&
                 (reinterpret_cast<uintptr_t>(p.input) % 16 == 0);

    if (p.oh * p.ow > 65535 || p.batch * p.groups > 65535 || (p.ocg + 63) / 64 > 65535 ||
        p.ocg <= 0) {
        return Status::unsupported;
    }

    const QuantParams& qa = act.quant_params();
    const QuantParams& qw = wgt.quant_params();
    p.zp_x = qa.zero_point_data != nullptr ? qa.zero_point_data[0] : qa.zero_point;
    p.sc_x = qa.scale_data != nullptr ? qa.scale_data[0] : qa.scale;
    p.zp_w_scalar = qw.zero_point;
    p.sc_w_scalar = qw.scale;

    const bool int_out = kind == I8OutKind::S32 || kind == I8OutKind::S8 || kind == I8OutKind::U8;
    p.relu = int_out && attrs.epilogue.type == EpilogueActivateType::Relu;
    if (int_out) {
        const QuantParams& qc = output.quant_params();
        p.out_scale = qc.scale_data != nullptr ? qc.scale_data[0] : qc.scale;
        p.out_zp = qc.zero_point_data != nullptr ? qc.zero_point_data[0] : qc.zero_point;
    }
    p.out_kind = static_cast<int>(kind);

    // ---- workspace: R_b and the staged per-channel arrays -----------------
    const int64_t rb_off = 0;
    int64_t cursor = int64_t(p.oc) * 4;
    int64_t off_zpw = -1, off_scw = -1;
    if (qw.zero_point_data != nullptr) {
        off_zpw = cursor;
        cursor += (int64_t(p.oc) * 4 + 15) & ~int64_t(15);
    }
    if (qw.scale_data != nullptr) {
        off_scw = cursor;
        cursor += (int64_t(p.oc) * 4 + 15) & ~int64_t(15);
    }

    const cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    const detail::GemmWorkspace ws = detail::gemm_workspace_alloc(cursor, stream);
    if (ws.ptr == nullptr) {
        return Status::internal;
    }
    char* base = static_cast<char*>(ws.ptr);
    p.r_b = reinterpret_cast<int32_t*>(base + rb_off);
    const auto copy_in = [&](const void* src, int64_t off) {
        if (off < 0) {
            return true;
        }
        return cudaMemcpyAsync(base + off, src, size_t(int64_t(p.oc) * 4), cudaMemcpyHostToDevice,
                               stream) == cudaSuccess;
    };
    if (!copy_in(qw.zero_point_data, off_zpw) || !copy_in(qw.scale_data, off_scw)) {
        detail::gemm_workspace_free(ws, stream);
        cudaGetLastError();
        return Status::internal;
    }
    p.zp_w = off_zpw >= 0 ? reinterpret_cast<const int32_t*>(base + off_zpw) : nullptr;
    p.sc_w = off_scw >= 0 ? reinterpret_cast<const float*>(base + off_scw) : nullptr;

    constexpr int kSumThreads = 128;
    const dim3 sgrid(unsigned((p.oc + kSumThreads - 1) / kSumThreads), 1, 1);
    conv_i8_col_sums_kernel<<<sgrid, kSumThreads, 0, stream>>>(p);

    if (adt == DataType::u8) {
        launch_conv_i8<uint8_t>(p, stream);
    } else {
        launch_conv_i8<int8_t>(p, stream);
    }

    detail::gemm_workspace_free(ws, stream);
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

} // namespace

Status conv2d_cuda(const Conv2DAttributes& attrs, TensorView& output,
                   std::span<const TensorView> inputs, const ComputeContext& ctx, void*)
{
    const TensorView& act = inputs[0];
    const TensorView& wgt = inputs[1];
    const TensorView* bias = inputs.size() > 2 ? &inputs[2] : nullptr;

    // The quantized family is an s8 filter against a pre-quantized u8/s8
    // activation; it owns its own kernel pair below.
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
    if (is_packed_conv_weight(wgt.layout()) || is_wino_weight(wgt.layout())) {
        return Status::unsupported;
    }

    ConvParams p;
    p.batch = static_cast<int>(act.shape(0));
    p.ic = static_cast<int>(act.shape(1));
    p.ih = static_cast<int>(act.shape(2));
    p.iw = static_cast<int>(act.shape(3));
    p.oc = static_cast<int>(output.shape(1));
    p.oh = static_cast<int>(output.shape(2));
    p.ow = static_cast<int>(output.shape(3));
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

    // The channel-last 16 B A-fill needs four (or eight) whole channels of one
    // tap on a dense channel axis: in_c == 1 puts consecutive channels next to
    // each other, IC/G a multiple of the vector makes a k-group never straddle
    // a tap, and the channel stride is what has to be 16 B aligned.
    const int kGroup = esz == 2 ? 8 : 4;
    p.fill_vec = p.in_c == 1 && p.icg % kGroup == 0 && (p.in_w * esz) % 16 == 0 &&
                 (reinterpret_cast<uintptr_t>(p.input) % 16 == 0);

    // The fill addresses the activation with 32-bit offsets from its block's
    // slice; a tensor past the halfway point of the int range is refused rather
    // than silently wrapping.
    if (p.oh * p.ow > 65535 || p.batch * p.groups > 65535 || p.ocg <= 0 ||
        act.numel() > (int64_t(1) << 30)) {
        return Status::unsupported;
    }

    const cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);
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
