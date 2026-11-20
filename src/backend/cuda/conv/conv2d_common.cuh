#pragma once
/// @file conv2d_common.cuh
/// @brief CUDA Conv2D internals shared by the two translation units.
///
/// The route is split one file per arithmetic family, as the CPU backend's conv
/// files are:
///
///   - conv2d.cu     — the f32/f16/bf16 tensor-core kernels, their launchers
///                     and the operator entry (conv2d_cuda);
///   - conv2d_i8.cu  — the quantized kernel pair (16x8x32 s8 MMA), its split-K
///                     reduce, and its own prepacked filter (PackedHWIOCuda).
///
/// Both walk the same activation geometry and cut the same reduction, so what
/// is genuinely common lives here: the parameter block, the A-tile row walk,
/// the tap decode, the output addressing and the split-K planner. Everything is
/// device-inline in an internal namespace, so each TU instantiates exactly what
/// its kernels call.

#include "../cuda_common.cuh"
#include "../matmul/gemm_common.cuh"
#include "nnops/ops/conv2d.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>

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
    int32_t* partial_i32 = nullptr; ///< the same planes for the int8 route

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

/// Output row base for `pixel`: the address `conv_row_addr` adds the channel to.
__device__ inline int64_t conv_pixel_base(const ConvParams& p, int64_t base, int pixel)
{
    const int oh = pixel / p.ow;
    return base + int64_t(oh) * p.out_h + int64_t(pixel - oh * p.ow) * p.out_w;
}

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
                            int wave_div, int min_tiles_per_split = 4)
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
    splits = std::min<int64_t>(splits, k_tiles / min_tiles_per_split);
    return splits >= 2 ? int(splits) : 1;
}

/// `NNOPS_SPLITK_LOG=1` reports the split count each launch chose, against the
/// geometry it was chosen from. Read once, so a timed launch pays nothing.
inline bool conv_splitk_log()
{
    static const bool on = std::getenv("NNOPS_SPLITK_LOG") != nullptr;
    return on;
}

/// Cut the reduction into k ranges when the grid is shallower than the resident
/// block count wants, and carve the scratch the split blocks write into. A null
/// workspace means single-pass: either the rule declined or the carve failed
/// (`k_splits` stays 1).
inline detail::GemmWorkspace conv_split_workspace(ConvParams& p, int bM, int bN, int bK,
                                                  int smem_bytes, int wave_div, cudaStream_t stream,
                                                  int min_tiles_per_split = 4)
{
    const int64_t zg = int64_t(p.batch) * p.groups;
    const int blocks = std::max(1, int(detail::kSmemPerSm / smem_bytes));
    const int64_t plane = int64_t(p.oh) * p.ow * p.ocg;
    int splits = conv_pick_splits(p.oh * p.ow, p.ocg, p.icg * p.kh * p.kw, zg, bM, bN, bK,
                                  detail::device_sm_count() * blocks, wave_div,
                                  min_tiles_per_split);
    if (conv_splitk_log()) {
        std::fprintf(stderr,
                     "conv splitk: %dx%d k%lld m=%lld n=%d K=%d tile=%dx%dx%d blocks/SM=%d "
                     "capacity=%d -> %d split(s)\n",
                     int(p.oh), int(p.ow), (long long)p.kh, (long long)p.oh * p.ow, p.ocg,
                     p.icg * p.kh * p.kw, bM, bN, bK, blocks,
                     detail::device_sm_count() * blocks, splits);
    }

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

} // namespace
} // namespace nnops::backend::cuda
