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
/// f32, f16 and bf16 in the dense kernels; a Winograd weight never reaches them
/// (the dispatch refuses it) and a prepacked filter is opaque to them. The
/// quantized family (an s8 filter against a pre-quantized u8/s8 activation) is
/// its own kernel at the bottom of this file: 16x8x32 s8 MMA over the same CTA
/// tile, MatMulInteger compensation with the per-pixel
/// activation sums folded in during the gather. That kernel is also the one
/// route with a prepacked filter of its own — `PackedHWIOCuda`, the k-major
/// panels cut at the tile's own geometry plus the per-channel sums, which turns
/// the fill's sixteen strided byte gathers into one `cp.async` (see
/// conv2d_pack.h).

#include "../cuda_common.cuh"
#include "../matmul/gemm_common.cuh"
#include "../matmul/matmul_f16.cuh"
#include "../matmul/matmul_f32.cuh"
#include "../matmul/matmul_i8_common.cuh" // I8OutKind + the shared int8 epilogue pieces
#include "../matmul/matmul_i8_ptx.cuh"
#include "conv_panel.cuh"
#include "conv2d_pack.h"
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

    // ---- int8 route (pre-quantized activation) ----------------------------
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
    /// A PackedHWIOCuda blob, or null for a raw HWIO filter. When set, the
    /// k-major panel is a plain copy out of it and the per-channel sums come
    /// from its tail instead of the separate reduction kernel.
    const void* w_panel = nullptr;
    int w_nblk = 0; ///< oc blocks per k tile
    float out_scale = 1.0f;
    int32_t out_zp = 0;
    int out_kind = 0;  ///< I8OutKind
    bool relu = false; ///< integer destinations: Relu before the store
};

/// The largest element offset the fills form inside one group slice of the
/// activation: a tap reaches KH rows and KW columns past the output's own
/// extent, and the innermost axis is the full channel count (the slice's base is
/// pre-shifted, so the strides are the tensor's, not the group's).
///
/// The fills keep that offset in 32 bits -- a runtime 64-bit offset costs an
/// IADD.X on every vector -- so a slice wider than the int range is refused
/// rather than silently wrapped. It is the span that matters, not the pixel
/// count: 448x448 of output is 13M, while the 65535-pixel limit that used to
/// stand here refused anything past 255x255.
inline int64_t conv_offset_span(const ConvParams& p)
{
    return (int64_t(p.ih) + p.kh * p.dh) * (int64_t(p.iw) + p.kw * p.dw) * p.ic;
}

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

/// The rows of an A tile one thread feeds on the channel-vector walk, and
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

/// The A tile row a thread feeds, given its place in the k-group walk.
///
/// A 16 B store from an eight-lane group has to land on eight distinct bank
/// quads. With four k-groups (the int8 vector is 16 B) one group spans two
/// rows, and two adjacent rows start 5 quads apart, so the second row's run of
/// four overlaps the first's by one: an LD of 80 bytes costs 2x the
/// wavefronts. Rows four apart are 20 quads apart, i.e. 4 mod 8, which puts the
/// two runs on disjoint quads. Swapping bits 0 and 2 of the row does that
/// (0,1,2,3 -> 0,4,2,6) without moving a lane off its four-lane load run.
/// Sixteen k-groups (f32) and eight (f16) never needed it.
template <int Groups>
__device__ inline int conv_fill_row(int p)
{
    if constexpr (Groups == 4) {
        p = (p & ~5) | ((p & 4) >> 2) | ((p & 1) << 2);
    }
    return p;
}

template <class Elem, int Threads, int bM, int bK>
__device__ inline ConvRows conv_rows(const ConvParams& p, int pixel0)
{
    constexpr int kGroup = 16 / int(sizeof(Elem)); // one 16 B vector of k
    constexpr int groups = bK / kGroup;
    constexpr int step = Threads / groups;
    static_assert(bM % step == 0 && bM / step <= ConvRows::kMax,
                  "the row walk has to fit ConvRows");
    const int M = p.oh * p.ow;
    const int base = int(threadIdx.x) / groups;

    ConvRows r;
    r.count = bM / step;
#pragma unroll
    for (int i = 0; i < ConvRows::kMax; ++i) {
        if (i >= r.count) {
            break;
        }
        const int row = conv_fill_row<groups>(base + i * step);
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

/// The tap and the channel a k index lands on, as state that can be stepped.
///
/// The vector fill asks for this once per k tile per thread, and the two runtime
/// divisions that answer it -- `k / icg` then `tap / kw` -- compile to a
/// reciprocal-multiply each. The fill is called with monotonic k (the ring only
/// ever looks one tile ahead), so it can be carried instead: one division on the
/// first tile, then a compare and a subtract per tap the tile crosses. Measured
/// interleaved against the dividing arm, 112x112x128 3x3 f32 0.656 -> 0.581 ms
/// and 224x224x64 3x3 f16 0.202 -> 0.192 ms.
struct ConvTapWalk {
    int k = -1;
    int ic = 0;
    int kh = 0;
    int kw = 0;
};

__device__ inline ConvTapWalk conv_tap_at(const ConvParams& p, int k)
{
    ConvTapWalk w;
    w.k = k;
    const int tap = k / p.icg;
    w.ic = k - tap * p.icg;
    w.kh = tap / p.kw;
    w.kw = tap - w.kh * p.kw;
    return w;
}

/// Advance to `k` (>= the current position) by stepping the carry, which costs a
/// compare and a subtract per tap crossed instead of two divisions.
__device__ inline void conv_tap_seek(const ConvParams& p, ConvTapWalk& w, int k)
{
    if (w.k < 0) {
        w = conv_tap_at(p, k);
        return;
    }
    w.ic += k - w.k;
    w.k = k;
    while (w.ic >= p.icg) {
        w.ic -= p.icg;
        if (++w.kw == p.kw) {
            w.kw = 0;
            ++w.kh;
        }
    }
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

/// Output element (pixel, channel): the channel is the innermost slot.
template <class Elem>
__device__ inline Elem* conv_out_addr(const ConvParams& p, int64_t base, int pixel, int n)
{
    const int oh = pixel / p.ow;
    const int ow = pixel - oh * p.ow;
    return static_cast<Elem*>(p.output) + base + int64_t(oh) * p.out_h + int64_t(ow) * p.out_w +
           int64_t(n) * p.out_c;
}

/// The same address when the caller already decoded the pixel: an epilogue
/// walking several channels of one output pixel pays the division once.
template <class Elem>
__device__ inline Elem* conv_row_addr(const ConvParams& p, int64_t row_base, int n)
{
    return static_cast<Elem*>(p.output) + row_base + int64_t(n) * p.out_c;
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

/// The same store when the caller already decoded the pixel to a row base. An
/// epilogue writes a run of channels per pixel, and `conv_out_addr`'s division
/// is what it pays for the privilege: the f16 tile wrote 32 elements a block,
/// 32 divisions of which 28 were redundant.
template <class Elem>
__device__ inline void conv_row_store(const ConvParams& p, int64_t row_base, int n, float v)
{
    Elem* dst = conv_row_addr<Elem>(p, row_base, n);
    if (p.add_to) {
        v += float(*dst);
    }
    *dst = Elem(v);
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

/// Output row base for `pixel`: the address `conv_row_addr` adds the channel to.
__device__ inline int64_t conv_pixel_base(const ConvParams& p, int64_t base, int pixel)
{
    const int oh = pixel / p.ow;
    return base + int64_t(oh) * p.out_h + int64_t(pixel - oh * p.ow) * p.out_w;
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
    ConvTapWalk tap_walk;

    auto fill = [&](int kt, int stage) {
        if (kt >= kt_end) {
            cp_async_commit_group();
            return;
        }
        const int k0 = kt * BK;
        float* const sA = sAbase + stage * S::kAMax;
        float* const sB = sBbase + stage * S::kBMax;
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
        f32_gemm_tile<S, false, false>(sAbase + stage * S::kAMax,
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
/// 49 blocks against 114 resident and a shallow cut already costs it 13%,
/// because two blocks per k range repeat the gather while the plane was wide
/// enough to hide it. A quarter-wave gate leaves every deeper
/// case untouched (7x7x512 has 8 blocks, 14x14x256 16) and only declines that
/// one.
///
/// `wave_div` is dtype-scaled because that margin is a time, not a shape: the
/// cut costs a fixed two launches and a stream-ordered workspace round trip, so
/// a plane only pays for it when one undivided wave lasts longer than that. The
/// same 49-block f16 plane takes 0.031 ms and loses 13% at two splits, while the
/// f32 twin takes 0.086 ms and gains 20% (0.087 -> 0.070), and the
/// gate is the only thing that tells them apart. The f16 kernel therefore needs
/// the quarter-wave gate and f32 the half.
///
/// The smallest cut is left in place at the very shallow end. f16 7x7x512 is
/// 12.8M MACs against a fixed cut cost of about 20 us, and it lands at 0.0184 ms
/// in five runs out of six and 0.0417 ms in the sixth -- the arm where the cut
/// does not cover itself. It still stays: the same layer in the earlier NCHW
/// order took 0.0440 ms cut against 0.0705 ms whole, every run,
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
    constexpr int smem = S::stages * (S::kAMax + S::kBMax) * int(sizeof(float));
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

// ============================================================
// int8 tensor-core kernel
// ============================================================

/// The int8 tile: 64 pixels by 64 channels, 64 bytes of k, three stages, and the
/// same eight-warp split as the f16 tile's [[ConvShapeH]] (32x16 per warp) rather
/// than four warps of 32x32. The s8 panels are half the fp16 footprint, so a
/// third stage still leaves three blocks a SM and 24 warps.
using ConvShapeI8 = I8TileShape<32, 16, 2, 4, 64, 3, 3>;

/// The same tile with 128 channels a CTA, for a group that has them. The A fill
/// is the int8 kernel's dominant cost and its total traffic is M*N*BK/BN, so
/// doubling BN halves it; the warp keeps a square-ish 32x32 fragment, which also
/// takes the A `ldmatrix` from one m16n8k32 operand set (LDSM:IMMA 1:1) to two.
using ConvShapeI8Wide = I8TileShape<32, 32, 2, 4, 64, 3, 2>;

/// The epilogue staging tile's row pitch, in bytes: the smallest pitch at least
/// `BN * esz` that is 80 mod 128. Four banks a row (80 bytes is 20 words) is
/// what the staging pass needs -- eight consecutive rows then land on eight
/// distinct banks -- and the copy-out reads 16 B at a time, where the pitch
/// only has to be a multiple of 16.
template <class S>
constexpr int conv_i8_stage_pitch(int esz)
{
    const int need = S::BN * esz;
    return need + (80 - need % 128 + 128) % 128;
}

static_assert(ConvShapeI8::BM * conv_i8_stage_pitch<ConvShapeI8>(4) <= ConvShapeI8::kSmemBytes,
              "the staging tile has to fit the pipeline's shared memory");
static_assert(ConvShapeI8Wide::BM * conv_i8_stage_pitch<ConvShapeI8Wide>(4) <=
                  ConvShapeI8Wide::kSmemBytes,
              "the staging tile has to fit the pipeline's shared memory");

/// Either k-major s8 panel, as the gather's (row, col, stage) indexer. The base
/// already carries the stage; rows are `LD` bytes apart.
template <class S, class Elem>
struct ConvI8Panel {
    Elem* base;
    __device__ __forceinline__ Elem& operator()(int row, int col, int) const
    {
        return base[row * S::LD + col];
    }
};

/// The int8 gather. Each thread owns one output pixel's
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

/// The int8 vector fill, one 16 B vector per (row, k-group): 16 consecutive k are
/// 16 consecutive channels inside one tap, contiguous in the activation and in
/// the shared tile. A thread owns `rows.count` pixels, so it keeps one partial
/// per owned row and folds them into shared memory once, after the k loop.
template <class Elem, int bK, class SmemA>
__device__ inline void conv_fill_a_vec_i8(const ConvParams& p, SmemA sA, int stage, const Elem* in,
                                          const ConvRows& rows, int k0, ConvTapWalk& walk,
                                          int32_t* sums, bool add)
{
    constexpr int kGroup = 16 / int(sizeof(Elem));
    using Vec = uint4;

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
            // Four bytes a step: the scalar fold was a shift and an add per
            // element -- 32 of the fill's ~143 instructions per k tile.
            const uint32_t* const w = reinterpret_cast<const uint32_t*>(&v);
            int32_t sum = 0;
        #pragma unroll
            for (int t = 0; t < kGroup / 4; ++t) {
                if constexpr (std::is_same_v<Elem, uint8_t>) {
                    sum = int(__dp4a(w[t], 0x01010101u, unsigned(sum)));
                } else {
                    sum = __dp4a(int(w[t]), 0x01010101, sum);
                }
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

// ============================================================
// int8 packed filter (PackedHWIOCuda)
// ============================================================

/// The blob's geometry, derived from the filter's shape and the attributes so
/// the descriptor query, the pack pass and the kernel cannot disagree. Bytes:
/// `[group][k tile][oc block]` panels of BN rows at LD bytes, then the
/// per-channel sums as `[group][OC/G]` int32.
struct ConvI8PackGeom {
    bool ok = false;
    int groups = 0, icg = 0, ocg = 0, oc = 0;
    int kh = 0, kw = 0;
    int K = 0, ktiles = 0, nblk = 0;
    int64_t panel = 0;    ///< bytes of one panel
    int64_t data = 0;     ///< bytes of every panel
    int64_t sums_off = 0; ///< where the int32 sums start
    int64_t total = 0;
};

inline ConvI8PackGeom conv_i8_pack_geom(int64_t groups, int64_t icg, int64_t ocg, int64_t kh,
                                        int64_t kw)
{
    ConvI8PackGeom g;
    if (groups <= 0 || icg <= 0 || ocg <= 0 || kh <= 0 || kw <= 0) {
        return g;
    }
    const int64_t K = icg * kh * kw;
    const int64_t ktiles = (K + kConvI8PackBK - 1) / kConvI8PackBK;
    const int64_t nblk = (ocg + kConvI8PackBN - 1) / kConvI8PackBN;
    const int64_t oc = groups * ocg;
    g.groups = int(groups);
    g.icg = int(icg);
    g.ocg = int(ocg);
    g.oc = int(oc);
    g.kh = int(kh);
    g.kw = int(kw);
    g.K = int(K);
    g.ktiles = int(ktiles);
    g.nblk = int(nblk);
    g.panel = int64_t(kConvI8PackBN) * kConvI8PackLD;
    g.data = groups * ktiles * nblk * g.panel;
    g.sums_off = (g.data + 15) & ~int64_t(15);
    g.total = g.sums_off + oc * 4;
    g.ok = groups * ocg <= 65535 && K <= (int64_t(1) << 24) &&
           g.total <= (int64_t(1) << 40);
    return g;
}

struct ConvI8PackParams {
    const void* weight;
    void* blob;
    ConvI8PackGeom g;
    int32_t w_n, w_ic, w_kh, w_kw;
};

/// One launch builds the whole blob: the panels first, then the sums. The
/// gather is the one the kernel used to run once per block per k tile, so
/// paying it here once per call — or never, when the caller caches the blob —
/// is the entire point. Byte-wise, because that is what a pack pass can afford
/// and the kernel's fill could not.
__global__ void conv_i8_pack_kernel(ConvI8PackParams q)
{
    constexpr int BN = kConvI8PackBN, BK = kConvI8PackBK, LD = kConvI8PackLD;
    constexpr int kVec = BK / 16;
    const int64_t per_panel = int64_t(BN) * kVec;
    const int G = q.g.groups, K = q.g.K, kt = q.g.ktiles, nb = q.g.nblk;
    const int64_t chunks = int64_t(G) * kt * nb * per_panel;
    int8_t* const blob = static_cast<int8_t*>(q.blob);
    const int8_t* const W = static_cast<const int8_t*>(q.weight);
    const int32_t w_n = q.w_n, w_ic = q.w_ic, w_kh = q.w_kh, w_kw = q.w_kw;
    const int32_t icg = q.g.icg, kw_end = q.g.kw;

    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < chunks;
         i += int64_t(gridDim.x) * blockDim.x) {
        const int64_t panel = i / per_panel;
        const int64_t r = i - panel * per_panel;
        const int row = int(r / kVec);
        const int c = int(r - int64_t(row) * kVec);
        const int nbi = int(panel % nb);
        const int kti = int((panel / nb) % kt);
        const int g = int(panel / (int64_t(nb) * kt));
        const int n = nbi * BN + row;
        const int k0 = kti * BK + c * 16;
        int8_t* const d = blob + panel * q.g.panel + row * LD + c * 16;
        if (n >= q.g.ocg) {
            *reinterpret_cast<uint4*>(d) = uint4{};
            continue;
        }
        const int8_t* const row_w = W + int64_t(g) * q.g.ocg * w_n + int64_t(n) * w_n;
        const int32_t tap = k0 / icg;
        int32_t ic = k0 - tap * icg;
        int32_t kh = tap / kw_end;
        int32_t kw = tap - kh * kw_end;
        int32_t off = ic * w_ic + kh * w_kh + kw * w_kw;
        int k = k0;
        uint4 v;
        int8_t* const vb = reinterpret_cast<int8_t*>(&v);
#pragma unroll 4
        for (int j = 0; j < 16; ++j) {
            vb[j] = k < K ? row_w[off] : int8_t(0);
            ++k;
            off += w_ic;
            if (++ic == icg) {
                ic = 0;
                off -= icg * w_ic;
                off += w_kw;
                if (++kw == kw_end) {
                    kw = 0;
                    off -= kw_end * w_kw;
                    off += w_kh;
                }
            }
        }
        *reinterpret_cast<uint4*>(d) = v;
    }

    int32_t* const sums = reinterpret_cast<int32_t*>(blob + q.g.sums_off);
    for (int oc = int(blockIdx.x) * blockDim.x + threadIdx.x; oc < q.g.oc;
         oc += int(gridDim.x) * blockDim.x) {
        const int8_t* const row_w = W + int64_t(oc) * w_n;
        int32_t s = 0;
        for (int32_t kh = 0; kh < q.g.kh; ++kh) {
            for (int32_t kw = 0; kw < q.g.kw; ++kw) {
                for (int32_t ic = 0; ic < icg; ++ic) {
                    s += int32_t(row_w[ic * w_ic + kh * w_kh + kw * w_kw]);
                }
            }
        }
        sums[oc] = s;
    }
}

template <class S, class AElem>
__global__ __launch_bounds__(S::threads, S::MinBlocks)
void conv_gemm_i8_kernel(ConvParams p)
{
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
    // The fill's k-region odometer, as on the fp tiles: the ring asks for
    // monotonic k, so the tap decode is carried instead of divided.
    ConvTapWalk tap_walk;

    const int k_tiles = (K + BK - 1) / BK;
    // The filter gather's bytes per thread per k tile, and whether such a run
    // can cross a tap. It cannot when icg is a multiple of the run, which the
    // tail check below is what makes sufficient.
    constexpr int kBFillK = BK / (S::threads / BN);
    constexpr int kNVec = kBFillK / 16;
    static_assert(kBFillK % 16 == 0, "the filter gather writes whole 16 B vectors");
    // A packed-filter panel is kConvI8PackBN rows of the format's own stride, so
    // a tile wider than the blob reads one panel per block of channels.
    constexpr int kNPanels = BN / kConvI8PackBN;
    constexpr int kPanelBytes = kConvI8PackBN * kConvI8PackLD;
    static_assert(BN % kConvI8PackBN == 0, "a tile has to read whole blob panels");
    const bool w_tap_aligned = (p.icg % kBFillK) == 0;
    ConvTapWalk w_walk;

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
            conv_fill_a_vec_i8<AElem, BK>(p, ConvI8Panel<S, AElem>{reinterpret_cast<AElem*>(sA)},
                                          stage, in, rows, k0, tap_walk, ra_vec, add);
        } else {
            const int32_t s =
                conv_fill_a_i8<AElem, S::threads, BM, BK>(
                    p, ConvI8Panel<S, AElem>{reinterpret_cast<AElem*>(sA)}, stage, in, m0, k0);
            if (add) {
                ra_single += s;
            }
        }
        // The filter is HWIO with the channel innermost, so its k axis is not
        // contiguous in memory and B is gathered byte-wise into a k-major panel.
        // The loads are the same either way; the store is what the tile pays
        // for. Thirty-two lanes storing one byte each sit on thirty-two rows at
        // a 16 B stride, which reaches two of the eight bank quads: 4 wavefronts
        // a byte, so the sixteen a thread owns cost sixty-four where one 16 B
        // vector costs four. ncu on 448x448x64: 12.4M of the kernel's 16.9M
        // store wavefronts were conflicts, against 79,799 on all the ldmatrix
        // reads together.
        //
        // A prepacked filter removes the gather altogether — and the sixteen
        // strided loads with it, which the vector store above leaves alone.
        {
            const int row = tid % BN;
            const int col0 = (tid / BN) * kBFillK;
            int8_t* const d = sB + row * LD + col0;
            const int n = n0 + row;
            const int8_t* const row_w = W + int64_t(n) * p.w_n;
            const int k = k0 + col0;
            const bool n_ok = n < N;
            const int32_t w_ic = int32_t(p.w_ic), w_kh = int32_t(p.w_kh), w_kw = int32_t(p.w_kw);
            if (p.w_panel != nullptr) {
                // The blob already holds this panel row for row: no predicate
                // (its tails are zeroed), no odometer, one asynchronous copy.
                // A blob panel is kConvI8PackBN rows, so a tile wider than that
                // reads one panel per block of channels; a block past the last
                // panel is left alone, and the epilogue skips those channels.
                const int half = row / kConvI8PackBN;
                const int64_t pan = int64_t(blockIdx.y) * kNPanels + half;
                if (pan < p.w_nblk) {
                    const int8_t* const src = static_cast<const int8_t*>(p.w_panel) +
                                              ((int64_t(g) * k_tiles + kt) * p.w_nblk + pan) *
                                                  kPanelBytes +
                                              (row - half * kConvI8PackBN) * LD + col0;
#pragma unroll
                    for (int c = 0; c < kNVec; ++c) {
                        cp_async_ca_16B(d + c * 16, src + c * 16);
                    }
                }
            } else if (n_ok && k + kBFillK <= K && w_tap_aligned) {
                // Inside one tap the run is a fixed `w_ic` stride, so the
                // odometer the fallback steps is not needed at all.
                conv_tap_seek(p, w_walk, k);
                const int32_t off0 = w_walk.ic * w_ic + w_walk.kh * w_kh + w_walk.kw * w_kw;
#pragma unroll
                for (int c = 0; c < kNVec; ++c) {
                    uint4 v;
                    int8_t* const vb = reinterpret_cast<int8_t*>(&v);
                    const int32_t off = off0 + c * 16 * w_ic;
#pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        vb[j] = row_w[off + j * w_ic];
                    }
                    *reinterpret_cast<uint4*>(d + c * 16) = v;
                }
            } else {
                const int32_t icg = int32_t(p.icg), kw_end = int32_t(p.kw);
                int kk = k;
                const int32_t tap0 = kk / icg;
                int32_t ic = kk - tap0 * icg;
                int32_t kh = tap0 / kw_end;
                int32_t kw = tap0 - kh * kw_end;
                int32_t off = ic * w_ic + kh * w_kh + kw * w_kw;
#pragma unroll 4
                for (int j = 0; j < kBFillK; ++j) {
                    d[j] = (n_ok && kk < K) ? row_w[off] : BElem(0);
                    ++kk;
                    off += w_ic;
                    if (++ic == icg) {
                        ic = 0;
                        off -= icg * w_ic;
                        off += w_kw;
                        if (++kw == kw_end) {
                            kw = 0;
                            off -= kw_end * w_kw;
                            off += w_kh;
                        }
                    }
                }
            }
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
        i8_gemm_tile<S, AElem>(sAbase + stage * S::kABytes, sBbase + stage * S::kBBytes,
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

    // A lane's fragment carries the cc pair of one pixel's channel run, so a
    // direct store moves four useful bytes per sector it touches -- eight rows
    // an instruction. ncu on 448x448x64 read 3.21M store sectors for a tile
    // whose bytes need 0.40M. The tile is staged in the pipeline's shared
    // memory instead (dead behind the barrier above) and copied out as whole
    // 16 B vectors, which is a sector's worth of one pixel's row.
    const int esz = kind == I8OutKind::F32 ? 4 : (kind == I8OutKind::F16 ? 2 : 1);
    const int pitch = conv_i8_stage_pitch<S>(esz);
    unsigned char* const sC = conv_smem;

    // C: lane l holds (m = l/4, n = 2*(l%4)), plus the +1 column and the +8 row.
    const int cm = lane / 4;
    const int cn = 2 * (lane % 4);
#pragma unroll
    for (int i = 0; i < S::MA; ++i) {
        const int gm = m0 + warp_m * S::WM + i * 16 + cm;
#pragma unroll
        for (int j = 0; j < S::NA; ++j) {
            const int gn = n0 + warp_n * S::WN + j * 8 + cn;
            const int cs = warp_n * S::WN + j * 8 + cn;
#pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                const int m = gm + rr * 8;
                if (m >= M) {
                    continue;
                }
                unsigned char* const srow = sC + (m - m0) * pitch;
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
                            reinterpret_cast<int8_t*>(srow)[cs + cc] = detail::i8_requant<int8_t>(
                                v, row_k, p.sc_x, float(p.out_zp));
                        } else {
                            reinterpret_cast<uint8_t*>(srow)[cs + cc] = detail::i8_requant<uint8_t>(
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
                            reinterpret_cast<float*>(srow)[cs + cc] = f;
                        } else {
                            reinterpret_cast<__half*>(srow)[cs + cc] = __float2half(f);
                        }
                        break;
                    }
                    }
                }
            }
        }
    }
    __syncthreads();

    // ---- copy-out --------------------------------------------------------
    // Phase two moves bytes: the staged tile already holds the destination's
    // bytes, row-major by channel. The widest run a row allows is the unit: 16
    // elements of s8 need the pixel pitch to be a multiple of 16, and a channel
    // count like 24 drops to a 4 B unit instead of falling all the way back.
    const auto fits = [&](int u) {
        return p.out_c == 1 && (p.out_w % u) == 0 && (p.out_h % u) == 0;
    };
    const int unit = fits(16 / esz) ? 16 / esz : (fits(4 / esz) ? 4 / esz : 1);
    const int chunks = S::BN / unit;
    const int width = unit * esz;
    unsigned char* const dst = static_cast<unsigned char*>(p.output);
    for (int v = tid; v < S::BM * chunks; v += S::threads) {
        const int row = v / chunks;
        const int ch = (v - row * chunks) * unit;
        const int m = m0 + row;
        if (m >= M || n0 + ch >= N) {
            continue;
        }
        const int oh = m / p.ow;
        const int ow = m - oh * p.ow;
        const int64_t px = out_base + int64_t(oh) * p.out_h + int64_t(ow) * p.out_w;
        const unsigned char* const src = sC + row * pitch + ch * esz;
        unsigned char* const d = dst + (px + n0 + ch) * esz;
        const uintptr_t a = reinterpret_cast<uintptr_t>(d);
        const bool vec = n0 + ch + unit <= N &&
                         (width < 4 || (a & (width == 16 ? 15u : 3u)) == 0);
        if (vec) {
            if (width == 16) {
                *reinterpret_cast<uint4*>(d) = *reinterpret_cast<const uint4*>(src);
            } else if (width == 4) {
                *reinterpret_cast<uint32_t*>(d) = *reinterpret_cast<const uint32_t*>(src);
            } else {
                for (int b = 0; b < width; ++b) {
                    d[b] = src[b];
                }
            }
            continue;
        }
        for (int c = 0; c < unit; ++c) {
            const int n = n0 + ch + c;
            if (n >= N) {
                break;
            }
            const unsigned char* const s = src + c * esz;
            unsigned char* const o = dst + (px + int64_t(n) * p.out_c) * esz;
            for (int b = 0; b < esz; ++b) {
                o[b] = s[b];
            }
        }
    }
}

template <class AElem> void launch_conv_i8(ConvParams& p, cudaStream_t stream)
{
    // A group narrower than the wide tile would waste half its MMA work, so the
    // 64-channel shape keeps the layers that have exactly that.
    if (p.ocg >= ConvShapeI8Wide::BN) {
        using S = ConvShapeI8Wide;
        constexpr int smem = S::kSmemBytes;
        auto kernel = conv_gemm_i8_kernel<S, AElem>;
        static const cudaError_t cfg =
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        (void)cfg;
        dim3 grid(unsigned((p.oh * p.ow + S::BM - 1) / S::BM),
                  unsigned((p.ocg + S::BN - 1) / S::BN), unsigned(p.batch * p.groups));
        kernel<<<grid, S::threads, smem, stream>>>(p);
        return;
    }
    using S = ConvShapeI8;
    constexpr int smem = S::kSmemBytes;
    auto kernel = conv_gemm_i8_kernel<S, AElem>;
    static const cudaError_t cfg =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    (void)cfg;

    dim3 grid(unsigned((p.oh * p.ow + S::BM - 1) / S::BM),
              unsigned((p.ocg + S::BN - 1) / S::BN), unsigned(p.batch * p.groups));
    kernel<<<grid, S::threads, smem, stream>>>(p);
}

/// The quantized route's CUDA entry: pre-quantized u8/s8 activation against an
/// s8 HWIO filter, MatMulInteger-shaped output. The weight's
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
    // PackedHWIO and the SME blob are the CPU routes' products and the Winograd
    // U has no CUDA consumer, so only this route's own blob is admitted here.
    const bool w_packed = is_cuda_conv_weight(wgt.layout());
    if (is_packed_conv_weight(wgt.layout()) || is_wino_weight(wgt.layout()) ||
        (w_packed && wgt.rank() != 2)) {
        return Status::unsupported;
    }
    // The int8 store has no residual stage, so an add_to conv is refused.
    if (attrs.add_to) {
        return Status::unsupported;
    }

    I8OutKind kind;
    if (!i8_out_kind(output.data_type(), kind) || kind == I8OutKind::S32) {
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
    // A blob's descriptor leads with OC, so the reduction's channel count comes
    // from the activation; a raw filter carries it on its third axis.
    p.icg = w_packed ? p.ic / p.groups : static_cast<int>(wgt.shape(2));
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
    const int64_t esz = static_cast<int64_t>(data_type_size(output.data_type()));
    (void)esz;
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
    p.add_to = false;

    // HWIO [KH][KW][IC/G][OC]: n is innermost, k is not contiguous. A blob has
    // no such axes -- its descriptor is [OC, ...] -- so the gather's strides are
    // only meaningful on the raw arm.
    if (!w_packed) {
        p.w_kh = wgt.stride_elems(0);
        p.w_kw = wgt.stride_elems(1);
        p.w_ic = wgt.stride_elems(2);
        p.w_n = wgt.stride_elems(3);
        p.w_k_dense = p.w_n == 1 && p.w_ic == p.oc && p.w_kw == int64_t(p.icg) * p.oc &&
                      p.w_kh == int64_t(p.kw) * p.icg * p.oc;
    }
    p.weight_vec = false;

    if (w_packed) {
        // The blob is what the panel would have gathered, so the geometry has to
        // agree to the byte: same groups, same channel count, same tiles.
        const ConvI8PackGeom pg = conv_i8_pack_geom(p.groups, p.icg, p.ocg, p.kh, p.kw);
        const char* const blob = wgt.ptr<const char>();
        if (!pg.ok || wgt.shape(0) != p.oc || wgt.numel() < pg.total ||
            (reinterpret_cast<uintptr_t>(blob) % 16 != 0)) {
            return Status::unsupported;
        }
        p.w_panel = blob;
        p.w_nblk = pg.nblk;
        p.r_b = reinterpret_cast<int32_t*>(const_cast<char*>(blob + pg.sums_off));
    }

    // The 16 B A-fill needs a dense channel axis and 16 channels of one tap per
    // vector (k/groups is tap-major there).
    const int kGroup = 16; // one 16 B vector of s8
    p.fill_vec = p.in_c == 1 && p.icg % kGroup == 0 && (p.in_w % 16 == 0) &&
                 (reinterpret_cast<uintptr_t>(p.input) % 16 == 0);

    if (conv_offset_span(p) >= (int64_t(1) << 30) || p.batch * p.groups > 65535 ||
        (p.ocg + 63) / 64 > 65535 || p.ocg <= 0) {
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
    // A packed filter brings its own sums, so it needs no R_b plane.
    const int64_t rb_off = 0;
    int64_t cursor = w_packed ? 0 : int64_t(p.oc) * 4;
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
    const detail::GemmWorkspace ws =
        cursor > 0 ? detail::gemm_workspace_alloc(cursor, stream) : detail::GemmWorkspace{};
    if (cursor > 0 && ws.ptr == nullptr) {
        return Status::internal;
    }
    char* base = static_cast<char*>(ws.ptr);
    if (!w_packed) {
        p.r_b = reinterpret_cast<int32_t*>(base + rb_off);
    }
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

    // A blob carries the sums its pack pass wrote, so the reduction — and its
    // grid — is skipped rather than run and overwritten.
    if (!w_packed) {
        constexpr int kSumThreads = 128;
        const dim3 sgrid(unsigned((p.oc + kSumThreads - 1) / kSumThreads), 1, 1);
        conv_i8_col_sums_kernel<<<sgrid, kSumThreads, 0, stream>>>(p);
    }

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
    // The int8 blob is the quantized kernel's, and this route would read its
    // descriptor as an HWIO filter.
    if (is_packed_conv_weight(wgt.layout()) || is_wino_weight(wgt.layout()) ||
        is_cuda_conv_weight(wgt.layout())) {
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

// ============================================================
// the int8 route's prepacked filter, for Conv2D::prepackWeights
// ============================================================

static_assert(kConvI8PackBN == ConvShapeI8::BN && kConvI8PackBK == ConvShapeI8::BK &&
                  kConvI8PackLD == ConvShapeI8::LD,
              "the blob's panels are the kernel's shared-memory panels");

TensorDesc conv_i8_packed_weight_desc(const Conv2DAttributes& attrs, const TensorDesc& weight)
{
    TensorDesc d;
    if (weight.rank != 4 || weight.dtype != DataType::s8 ||
        weight.layout != TensorLayout::HWIO || attrs.algorithm == ConvAlgorithm::Winograd ||
        attrs.groups <= 0 || weight.dims[3] % attrs.groups != 0) {
        return d;
    }
    const int64_t oc = weight.dims[3];
    const ConvI8PackGeom g =
        conv_i8_pack_geom(attrs.groups, weight.dims[2], oc / attrs.groups, weight.dims[0],
                          weight.dims[1]);
    if (!g.ok) {
        return d;
    }
    d.rank = 2;
    d.layout = TensorLayout::PackedHWIOCuda;
    d.dtype = DataType::s8;
    d.dims[0] = oc;
    d.dims[1] = (g.total + oc - 1) / oc;
    d.row_stride_elems = d.dims[1];
    return d;
}

Status conv_i8_packed_weight_pack(const Conv2DAttributes& attrs, const TensorView& weight,
                                  TensorView& out, const ComputeContext& ctx)
{
    if (weight.rank() != 4 || weight.data_type() != DataType::s8 ||
        weight.layout() != TensorLayout::HWIO || attrs.groups <= 0 ||
        weight.shape(3) % attrs.groups != 0) {
        return Status::unsupported;
    }
    const int64_t oc = weight.shape(3);
    const ConvI8PackGeom g =
        conv_i8_pack_geom(attrs.groups, weight.shape(2), oc / attrs.groups, weight.shape(0),
                          weight.shape(1));
    if (!g.ok || out.rank() != 2 || out.data_type() != DataType::s8 ||
        out.layout() != TensorLayout::PackedHWIOCuda || out.shape(0) != oc ||
        out.numel() < g.total ||
        (reinterpret_cast<uintptr_t>(out.ptr<const void>()) % 16 != 0)) {
        return Status::unsupported;
    }

    ConvI8PackParams q;
    q.weight = weight.ptr<const void>();
    q.blob = out.ptr<void>();
    q.g = g;
    q.w_kh = int32_t(weight.stride_elems(0));
    q.w_kw = int32_t(weight.stride_elems(1));
    q.w_ic = int32_t(weight.stride_elems(2));
    q.w_n = int32_t(weight.stride_elems(3));

    const int64_t chunks = int64_t(g.groups) * g.ktiles * g.nblk * kConvI8PackBN *
                           (kConvI8PackBK / 16);
    const int64_t work = chunks > oc ? chunks : oc;
    const int threads = 256;
    const int blocks = int((work + threads - 1) / threads);
    const cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    conv_i8_pack_kernel<<<blocks, threads, 0, stream>>>(q);
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

} // namespace nnops::backend::cuda
