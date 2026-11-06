/// @file depthwise_transpose_conv.cpp
/// @brief SIMD CPU depthwise transposed 2D convolution, channel-first and
///        channel-last.
///
/// Gather form, the exact inverse of depthwise conv's indexing: output (oh, ow)
/// reads input (ih, iw) where oh + PH - kh·DH == ih·SH, so a tap contributes
/// only to one residue of the output coordinate and every output pixel is
/// accumulated once — bias and epilogue included — without the zero-fill and
/// save/restore an output-side scatter would need for add_to.
///
/// Two per-class facts are hoisted out of the pixel loop: the (kh, ih) pairs
/// that land on an output row, and the (kw, shift) pairs that land on one
/// output-column residue, where a residue step t reads input column shift + t.
/// A row therefore splits into min(SW, OW) residue classes, and inside each one
/// the valid window [t_lo, t_hi) — every tap in range — runs free of bounds
/// tests, with the two edges left to the scalar tap loop.
///
/// The lane axis follows the layout:
///
///   channel-first   the lane spans the residue step, so a tap is a contiguous
///                   input run against a broadcast scalar weight. The store is
///                   contiguous when SW == 1 and strided otherwise, so the
///                   strided case stages the vector through a lane buffer.
///   channel-last    the lane spans channels and the HWCM filter is read
///                   directly. Four residue steps share one weight vector per
///                   tap, and 16 channels (two vectors) are swept per pass so
///                   a pixel's 64-byte line is touched once.
///
/// A kernel taller than the tap-list cap falls back to the plain scalar
/// gather, which is correct for any shape.

#include "nnops/ops/depthwise_transpose_conv.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"
#include "backend/cpu/simd_kernel/simd_int8_lane.hpp"
#include "backend/cpu/simd_kernel/simd_quant.hpp"
#include "backend/cpu/common/memory_pool.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

struct DwtConvGeom {
    int64_t N, C, IH, IW;          // input
    int64_t OH, OW;                // output
    int64_t KH, KW;                // kernel
    int64_t SH, SW;                // stride
    int64_t DH, DW;                // dilation
    int64_t PH, PW;                // padding
    Epilogue epilogue;
    bool add_to;
};

/// The tap lists hold at most ceil(K) entries; a taller kernel takes the scalar
/// route.
constexpr int64_t kMaxTaps = 16;

struct TapList {
    int64_t n = 0;
    int64_t a[kMaxTaps] = {};   // kernel index (kh or kw)
    int64_t b[kMaxTaps] = {};   // input index (ih) or column shift
};

/// The (kh, ih) pairs that land on output row @p oh.
TapList row_taps(const DwtConvGeom& gm, int64_t oh) noexcept
{
    TapList t;
    for (int64_t kh = 0; kh < gm.KH && t.n < kMaxTaps; ++kh) {
        const int64_t num = oh + gm.PH - kh * gm.DH;
        if (num < 0 || num % gm.SH != 0) { continue; }
        const int64_t ih = num / gm.SH;
        if (ih >= gm.IH) { continue; }
        t.a[t.n] = kh;
        t.b[t.n] = ih;
        ++t.n;
    }
    return t;
}

/// The (kw, shift) pairs that land on output-column residue @p r: the class
/// step t reads input column shift + t. Pairs that no step of the class can
/// reach are dropped.
TapList col_taps(const DwtConvGeom& gm, int64_t r, int64_t mic) noexcept
{
    TapList t;
    for (int64_t kw = 0; kw < gm.KW && t.n < kMaxTaps; ++kw) {
        const int64_t num = r + gm.PW - kw * gm.DW;
        if (num % gm.SW != 0) { continue; }
        const int64_t shift = num / gm.SW;
        if (shift >= gm.IW || shift <= -mic) { continue; }
        t.a[t.n] = kw;
        t.b[t.n] = shift;
        ++t.n;
    }
    return t;
}

/// The channel-first task grid: N·C planes × a block of output rows × a strip
/// of a row's residue steps.
///
/// A row is only OW pixels, which is too little to amortize a claimed item once
/// the pool over-subscribes the grid (OpenMP switches to claimed items past the
/// fast-core count). A task therefore walks a block of rows, and when even the
/// row grid cannot fill the pool, it splits a row's residue steps into strips.
struct DwtRowGrid {
    int64_t rows_per_task = 1;
    int64_t row_blocks = 1;
    int64_t strips = 1;
    int64_t total = 0;
};

DwtRowGrid dwt_row_grid(const DwtConvGeom& gm, int64_t threads) noexcept
{
    DwtRowGrid g;
    const int64_t slots = threads > 0 ? threads : 1;
    const int64_t total_rows = gm.N * gm.C * gm.OH;
    const int64_t planes = gm.N * gm.C;
    if (total_rows <= 0 || planes <= 0) { return g; }

    const int64_t want = 8 * slots;              // eight claims per item slot
    g.rows_per_task = gm.OH;
    if (total_rows >= want) {
        g.row_blocks = std::min<int64_t>(gm.OH, (want + planes - 1) / planes);
        g.rows_per_task = (gm.OH + g.row_blocks - 1) / g.row_blocks;
        g.row_blocks = (gm.OH + g.rows_per_task - 1) / g.rows_per_task;
    } else {
        const int64_t max_strips = std::max<int64_t>(1, (gm.OW + 3) / 4);
        g.strips = std::min<int64_t>((want + total_rows - 1) / total_rows, max_strips);
    }
    g.total = planes * g.row_blocks * g.strips;
    return g;
}

// ============================================================
// Scalar gather — the fallback for kernels past the tap-list cap
// ============================================================

template <typename T>
void dwtconv_scalar_first(const DwtConvGeom& gm, TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const T* in_ptr  = input.ptr<T>();
    const T* w_ptr   = weight.ptr<T>();
    const T* b_ptr   = has_bias ? inputs[2].ptr<T>() : nullptr;
    T* out_ptr = output.ptr<T>();

    const int64_t in_n   = input.stride_elems(0);
    const int64_t in_row = input.row_stride_elems();
    const int64_t in_ch  = input.channel_block_stride_elems();
    const int64_t out_n   = output.stride_elems(0);
    const int64_t out_row = output.row_stride_elems();
    const int64_t out_ch  = output.channel_block_stride_elems();
    const int64_t w_ch = weight.stride_elems(0);
    const int64_t w_kh = weight.stride_elems(2);
    const int64_t w_kw = weight.stride_elems(3);

    ctx.cpu.run(0, gm.N * gm.C, [&](int64_t task) {
        const int64_t n = task / gm.C;
        const int64_t c = task % gm.C;
        const T* in_c  = in_ptr + n * in_n + c * in_ch;
        T* out_c = out_ptr + n * out_n + c * out_ch;
        const T* w_c = w_ptr + c * w_ch;
        const float bias = has_bias ? s_load(&b_ptr[c]) : 0.0f;

        for (int64_t oh = 0; oh < gm.OH; ++oh) {
            T* orow = out_c + oh * out_row;
            for (int64_t ow = 0; ow < gm.OW; ++ow) {
                float acc = 0.0f;
                for (int64_t kh = 0; kh < gm.KH; ++kh) {
                    const int64_t num_h = oh + gm.PH - kh * gm.DH;
                    if (num_h < 0 || num_h % gm.SH != 0) { continue; }
                    const int64_t ih = num_h / gm.SH;
                    if (ih >= gm.IH) { continue; }
                    const T* xrow = in_c + ih * in_row;
                    for (int64_t kw = 0; kw < gm.KW; ++kw) {
                        const int64_t num_w = ow + gm.PW - kw * gm.DW;
                        if (num_w < 0 || num_w % gm.SW != 0) { continue; }
                        const int64_t iw = num_w / gm.SW;
                        if (iw >= gm.IW) { continue; }
                        acc += s_load(&xrow[iw]) *
                               s_load(&w_c[kh * w_kh + kw * w_kw]);
                    }
                }
                acc += bias;
                acc = nnops::apply_epilogue(gm.epilogue, acc);
                if (gm.add_to) { acc += s_load(&orow[ow]); }
                s_store(&orow[ow], acc);
            }
        }
    });
}

template <typename T>
void dwtconv_scalar_last(const DwtConvGeom& gm, TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const T* in_ptr  = input.ptr<T>();
    const T* w_ptr   = weight.ptr<T>();
    const T* b_ptr   = has_bias ? inputs[2].ptr<T>() : nullptr;
    T* out_ptr = output.ptr<T>();

    const int64_t in_n   = input.stride_elems(0);
    const int64_t in_row = input.row_stride_elems();
    const int64_t out_n   = output.stride_elems(0);
    const int64_t out_row = output.row_stride_elems();
    const int64_t w_kh = weight.stride_elems(0);
    const int64_t w_kw = weight.stride_elems(1);
    const int64_t C = gm.C;

    ctx.cpu.run(0, gm.N * gm.OH, [&](int64_t task) {
        const int64_t n  = task / gm.OH;
        const int64_t oh = task % gm.OH;
        const T* in_np = in_ptr + n * in_n;
        T* orow = out_ptr + n * out_n + oh * out_row;

        for (int64_t ow = 0; ow < gm.OW; ++ow) {
            T* dst = orow + ow * C;
            for (int64_t c = 0; c < C; ++c) {
                float acc = 0.0f;
                for (int64_t kh = 0; kh < gm.KH; ++kh) {
                    const int64_t num_h = oh + gm.PH - kh * gm.DH;
                    if (num_h < 0 || num_h % gm.SH != 0) { continue; }
                    const int64_t ih = num_h / gm.SH;
                    if (ih >= gm.IH) { continue; }
                    const T* xrow = in_np + ih * in_row;
                    for (int64_t kw = 0; kw < gm.KW; ++kw) {
                        const int64_t num_w = ow + gm.PW - kw * gm.DW;
                        if (num_w < 0 || num_w % gm.SW != 0) { continue; }
                        const int64_t iw = num_w / gm.SW;
                        if (iw >= gm.IW) { continue; }
                        acc += s_load(&xrow[iw * C + c]) *
                               s_load(&w_ptr[kh * w_kh + kw * w_kw + c]);
                    }
                }
                if (has_bias) { acc += s_load(&b_ptr[c]); }
                acc = nnops::apply_epilogue(gm.epilogue, acc);
                if (gm.add_to) { acc += s_load(&dst[c]); }
                s_store(&dst[c], acc);
            }
        }
    });
}

// ============================================================
// Channel-first — the lane spans the residue step
// ============================================================

template <typename T>
void dwtconv_channel_first(const DwtConvGeom& gm, TensorView& output,
                           std::span<const TensorView> inputs,
                           const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const T* in_ptr  = input.ptr<T>();
    const T* w_ptr   = weight.ptr<T>();
    const T* b_ptr   = has_bias ? inputs[2].ptr<T>() : nullptr;
    T* out_ptr = output.ptr<T>();

    const int64_t in_n   = input.stride_elems(0);
    const int64_t in_row = input.row_stride_elems();
    const int64_t in_ch  = input.channel_block_stride_elems();
    const int64_t out_n   = output.stride_elems(0);
    const int64_t out_row = output.row_stride_elems();
    const int64_t out_ch  = output.channel_block_stride_elems();
    // The raw [C, 1, KH, KW] filter: one tap block per channel.
    const int64_t w_ch = weight.stride_elems(0);
    const int64_t w_kh = weight.stride_elems(2);
    const int64_t w_kw = weight.stride_elems(3);

    const T* tag = nullptr;
    constexpr int L = simd_lane_for<T>;

    const DwtRowGrid grid = dwt_row_grid(gm, ctx.cpu.thread_count());
    if (grid.total <= 0) { return; }

    ctx.cpu.run(0, grid.total, [&](int64_t task) {
        const int64_t st = task % grid.strips;
        const int64_t t1 = task / grid.strips;
        const int64_t rb = t1 % grid.row_blocks;
        const int64_t t2 = t1 / grid.row_blocks;
        const int64_t c  = t2 % gm.C;
        const int64_t n  = t2 / gm.C;

        const int64_t oh_begin = rb * grid.rows_per_task;
        const int64_t oh_end   = std::min(gm.OH, oh_begin + grid.rows_per_task);

        const T* in_c = in_ptr + n * in_n + c * in_ch;
        T* och = out_ptr + n * out_n + c * out_ch;
        const T* w_c = w_ptr + c * w_ch;
        const float bias = has_bias ? s_load(&b_ptr[c]) : 0.0f;

        const int64_t nc = std::min(gm.SW, gm.OW);

        alignas(32) T lanes[L];

        for (int64_t oh = oh_begin; oh < oh_end; ++oh) {
            const TapList rt = row_taps(gm, oh);
            T* orow = och + oh * out_row;

            for (int64_t r = 0; r < nc; ++r) {
                const int64_t mic = (gm.OW - r + gm.SW - 1) / gm.SW;
                const int64_t per = (mic + grid.strips - 1) / grid.strips;
                const int64_t ts = std::min(mic, st * per);
                const int64_t te = std::min(mic, ts + per);
                if (ts >= te) { continue; }

                const TapList ct = col_taps(gm, r, mic);

                int64_t t_lo = 0;
                int64_t t_hi = mic;
                for (int64_t k = 0; k < ct.n; ++k) {
                    t_lo = std::max(t_lo, -ct.b[k]);
                    t_hi = std::min(t_hi, gm.IW - ct.b[k]);
                }
                if (t_lo > t_hi) { t_lo = t_hi; }   // no tap is in range throughout

                const auto pixel = [&](int64_t t) {
                    float acc = 0.0f;
                    for (int64_t a = 0; a < rt.n; ++a) {
                        const T* xrow = in_c + rt.b[a] * in_row;
                        for (int64_t k = 0; k < ct.n; ++k) {
                            const int64_t iw = ct.b[k] + t;
                            if (iw < 0 || iw >= gm.IW) { continue; }
                            acc += s_load(&xrow[iw]) *
                                   s_load(&w_c[rt.a[a] * w_kh + ct.a[k] * w_kw]);
                        }
                    }
                    acc += bias;
                    acc = nnops::apply_epilogue(gm.epilogue, acc);
                    T* dst = orow + r + t * gm.SW;
                    if (gm.add_to) { acc += s_load(dst); }
                    s_store(dst, acc);
                };

                const int64_t ilo = std::min(std::max(t_lo, ts), te);
                const int64_t ihi = std::min(t_hi, te);
                int64_t t = ts;
                for (; t < ilo; ++t) { pixel(t); }

                for (; t + L <= ihi; t += L) {
                    vec_for<T> acc = v_zero(tag);
                    for (int64_t a = 0; a < rt.n; ++a) {
                        const T* xrow = in_c + rt.b[a] * in_row;
                        for (int64_t k = 0; k < ct.n; ++k) {
                            const float wv =
                                s_load(&w_c[rt.a[a] * w_kh + ct.a[k] * w_kw]);
                            acc = v_fmadd(v_load(xrow + ct.b[k] + t),
                                          v_set1(tag, wv), acc);
                        }
                    }
                    if (has_bias) { acc = v_add(acc, v_set1(tag, bias)); }
                    acc = apply_epilogue_vec<T>(gm.epilogue, tag, acc);

                    if (gm.SW == 1) {
                        if (gm.add_to) { acc = v_add(acc, v_load(orow + t)); }
                        v_store(orow + t, acc);
                    } else {
                        // A residue step lands SW pixels apart: stage the lanes and
                        // store one by one, reading the destination the same way.
                        if (gm.add_to) {
                            for (int64_t j = 0; j < L; ++j) {
                                s_store(&lanes[j], s_load(&orow[r + (t + j) * gm.SW]));
                            }
                            acc = v_add(acc, v_load(lanes));
                        }
                        v_store(lanes, acc);
                        for (int64_t j = 0; j < L; ++j) {
                            s_store(&orow[r + (t + j) * gm.SW], s_load(&lanes[j]));
                        }
                    }
                }

                for (; t < ihi; ++t) { pixel(t); }
                for (; t < te; ++t) { pixel(t); }
            }
            }
    });
}

// ============================================================
// Channel-last — the lane spans channels
// ============================================================

template <typename T>
void dwtconv_channel_last(const DwtConvGeom& gm, TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const T* in_ptr  = input.ptr<T>();
    const T* w_ptr   = weight.ptr<T>();
    const T* b_ptr   = has_bias ? inputs[2].ptr<T>() : nullptr;
    T* out_ptr = output.ptr<T>();

    const int64_t in_n   = input.stride_elems(0);
    const int64_t in_row = input.row_stride_elems();
    const int64_t out_n   = output.stride_elems(0);
    const int64_t out_row = output.row_stride_elems();
    // The raw HWCM [KH][KW][C] filter: one tap is C contiguous channels.
    const int64_t w_kh = weight.stride_elems(0);
    const int64_t w_kw = weight.stride_elems(1);
    const int64_t C = gm.C;

    const T* tag = nullptr;
    constexpr int L = simd_lane_for<T>;
    // Sixteen channels a chunk: a pixel's 16 f32 is one 64-byte line, so the
    // sweep touches every line once.
    constexpr int64_t kChunk = 16;

    // The task grid is N×OH rows, and a row holds only OW pixels: split each
    // row's residue steps into strips when the grid cannot fill the pool.
    const int64_t rows = gm.N * gm.OH;
    const int64_t nt = std::max(1, ctx.cpu.thread_count());
    int64_t strips = 1;
    if (rows < nt) {
        const int64_t max_strips = std::max<int64_t>(1, (gm.OW + 3) / 4);
        strips = std::min<int64_t>((nt + rows - 1) / rows, max_strips);
        strips = std::max<int64_t>(strips, 1);
    }

    const int64_t nc = std::min(gm.SW, gm.OW);

    ctx.cpu.run(0, rows * strips, [&](int64_t task) {
        const int64_t st  = task % strips;
        const int64_t row = task / strips;
        const int64_t oh  = row % gm.OH;
        const int64_t n   = row / gm.OH;

        const T* in_np = in_ptr + n * in_n;
        T* orow = out_ptr + n * out_n + oh * out_row;
        const TapList rt = row_taps(gm, oh);

        for (int64_t r = 0; r < nc; ++r) {
            const int64_t mic = (gm.OW - r + gm.SW - 1) / gm.SW;
            const int64_t per = (mic + strips - 1) / strips;
            const int64_t ts = std::min(mic, st * per);
            const int64_t te = std::min(mic, ts + per);
            if (ts >= te) { continue; }

            const TapList ct = col_taps(gm, r, mic);
            int64_t t_lo = 0;
            int64_t t_hi = mic;
            for (int64_t k = 0; k < ct.n; ++k) {
                t_lo = std::max(t_lo, -ct.b[k]);
                t_hi = std::min(t_hi, gm.IW - ct.b[k]);
            }
            if (t_lo > t_hi) { t_lo = t_hi; }

            const auto scalar_one = [&](int64_t t, int64_t c) {
                float acc = 0.0f;
                for (int64_t a = 0; a < rt.n; ++a) {
                    const T* xrow = in_np + rt.b[a] * in_row;
                    for (int64_t k = 0; k < ct.n; ++k) {
                        const int64_t iw = ct.b[k] + t;
                        if (iw < 0 || iw >= gm.IW) { continue; }
                        acc += s_load(&xrow[iw * C + c]) *
                               s_load(&w_ptr[rt.a[a] * w_kh + ct.a[k] * w_kw + c]);
                    }
                }
                if (has_bias) { acc += s_load(&b_ptr[c]); }
                acc = nnops::apply_epilogue(gm.epilogue, acc);
                T* dst = orow + (r + t * gm.SW) * C + c;
                if (gm.add_to) { acc += s_load(dst); }
                s_store(dst, acc);
            };

            const auto pixel = [&](int64_t t) {
                for (int64_t c0 = 0; c0 < C; c0 += kChunk) {
                    const int64_t cend = std::min(C, c0 + kChunk);
                    int64_t c = c0;
                    for (; c + L <= cend; c += L) {
                        vec_for<T> acc = v_zero(tag);
                        for (int64_t a = 0; a < rt.n; ++a) {
                            const T* xrow = in_np + rt.b[a] * in_row;
                            for (int64_t k = 0; k < ct.n; ++k) {
                                const int64_t iw = ct.b[k] + t;
                                if (iw < 0 || iw >= gm.IW) { continue; }
                                acc = v_fmadd(
                                    v_load(xrow + iw * C + c),
                                    v_load(w_ptr + rt.a[a] * w_kh +
                                           ct.a[k] * w_kw + c),
                                    acc);
                            }
                        }
                        if (has_bias) { acc = v_add(acc, v_load(b_ptr + c)); }
                        acc = apply_epilogue_vec<T>(gm.epilogue, tag, acc);
                        T* dst = orow + (r + t * gm.SW) * C + c;
                        if (gm.add_to) { acc = v_add(acc, v_load(dst)); }
                        v_store(dst, acc);
                    }
                    for (; c < cend; ++c) { scalar_one(t, c); }
                }
            };

            // Four residue steps share each tap's weight vector; the window is
            // tap-complete, so the tile takes no bounds test.
            const auto tile4 = [&](int64_t t) {
                for (int64_t c0 = 0; c0 < C; c0 += kChunk) {
                    const int64_t cend = std::min(C, c0 + kChunk);
                    int64_t c = c0;
                    for (; c + L <= cend; c += L) {
                        vec_for<T> a0 = v_zero(tag);
                        vec_for<T> a1 = v_zero(tag);
                        vec_for<T> a2 = v_zero(tag);
                        vec_for<T> a3 = v_zero(tag);
                        for (int64_t a = 0; a < rt.n; ++a) {
                            const T* xrow = in_np + rt.b[a] * in_row + c;
                            for (int64_t k = 0; k < ct.n; ++k) {
                                const vec_for<T> wv =
                                    v_load(w_ptr + rt.a[a] * w_kh +
                                           ct.a[k] * w_kw + c);
                                const T* src = xrow + (ct.b[k] + t) * C;
                                a0 = v_fmadd(v_load(src), wv, a0);
                                a1 = v_fmadd(v_load(src + C), wv, a1);
                                a2 = v_fmadd(v_load(src + 2 * C), wv, a2);
                                a3 = v_fmadd(v_load(src + 3 * C), wv, a3);
                            }
                        }
                        if (has_bias) {
                            const vec_for<T> bv = v_load(b_ptr + c);
                            a0 = v_add(a0, bv);
                            a1 = v_add(a1, bv);
                            a2 = v_add(a2, bv);
                            a3 = v_add(a3, bv);
                        }
                        a0 = apply_epilogue_vec<T>(gm.epilogue, tag, a0);
                        a1 = apply_epilogue_vec<T>(gm.epilogue, tag, a1);
                        a2 = apply_epilogue_vec<T>(gm.epilogue, tag, a2);
                        a3 = apply_epilogue_vec<T>(gm.epilogue, tag, a3);
                        T* d0 = orow + (r + t * gm.SW) * C + c;
                        T* d1 = d0 + gm.SW * C;
                        T* d2 = d1 + gm.SW * C;
                        T* d3 = d2 + gm.SW * C;
                        if (gm.add_to) {
                            a0 = v_add(a0, v_load(d0));
                            a1 = v_add(a1, v_load(d1));
                            a2 = v_add(a2, v_load(d2));
                            a3 = v_add(a3, v_load(d3));
                        }
                        v_store(d0, a0);
                        v_store(d1, a1);
                        v_store(d2, a2);
                        v_store(d3, a3);
                    }
                    for (; c < cend; ++c) {
                        for (int64_t j = 0; j < 4; ++j) { scalar_one(t + j, c); }
                    }
                }
            };

            const int64_t ilo = std::max(t_lo, ts);
            const int64_t ihi = std::min(t_hi, te);
            int64_t t = ts;
            for (; t < ilo; ++t) { pixel(t); }
            for (; t + 4 <= ihi; t += 4) { tile4(t); }
            for (; t < ihi; ++t) { pixel(t); }
            for (; t < te; ++t) { pixel(t); }
        }
    });
}

// ============================================================
// Quantized — s8 weight against a u8/s8/f16/f32 activation
// ============================================================
//
// The tap lists drive both int8 layouts unchanged: the reduction is still over
// taps, so the same per-row (kh, ih) and per-residue (kw, shift) classes apply
// and only the arithmetic inside the window differs. Every term is
// (w − zp_w)·(x − zp_x) accumulated in int32, the reference's own expression;
// integer sums are exact and order-independent over these magnitudes, so the
// vector lanes and the scalar edges cannot disagree on a bit.
//
// The activation is normalised to one s8 plane per call (dwtconv_i8_plane), so
// every storage type meets one zero-point convention. The weight's zero point
// folds into the product — a channel-first filter broadcasts (w − zp_w) as a
// scalar, a channel-last HWCM filter is staged as int16 lanes once per call.
// The activation's cannot: (x − zp_x) does not fit a byte, so it becomes the
// accumulator's seed, −zp_x·Σ(w − zp_w) over the taps live across the whole
// window, and the boundary pixels, whose tap set is filtered, take the explicit
// per-tap form instead.

/// The s8 plane the int8 kernels read: one byte per element, the activation's
/// zero point shifted to the s8 convention.
struct DwtI8Plane {
    const int8_t* ptr = nullptr;
    int64_t n_stride = 0;    ///< elements between batches
    int64_t row_stride = 0;  ///< elements between rows of one plane
    int64_t ch_stride = 0;   ///< channel-first: elements between channels
    float scale = 1.0f;
    int32_t zero = 0;
};

/// Normalise one activation to a dense s8 plane. A pre-quantized s8 activation
/// with dense rows is read in place; anything else is copied, u8 through a bit
/// shift and fp through the arch quantizer at `zp − 128`, so the byte difference
/// against the shifted zero point is still the reference's (x − zp_x) — the
/// clamp windows map onto each other under the same −128.
template <class TAct>
Status dwtconv_i8_plane(const TensorView& in, bool channel_last, bool act_u8,
                        DwtI8Plane& pl, PoolArray<int8_t>& buf,
                        const ComputeContext& ctx)
{
    constexpr bool kFpAct = !std::is_same_v<TAct, int8_t> && !std::is_same_v<TAct, uint8_t>;

    const int64_t N  = in.shape(0);
    const int64_t C  = in.shape(1);
    const int64_t IH = in.shape(2);
    const int64_t IW = in.shape(3);
    const int64_t row_len = channel_last ? IW * C : IW;

    pl.scale = quant_scale_at(in.quant_params(), 0);
    pl.zero = quant_zp_at(in.quant_params(), 0) - (act_u8 ? 128 : 0);

    const int64_t in_ns = in.stride_elems(0);
    const int64_t in_rs = in.row_stride_elems();
    const int64_t in_cs = in.channel_block_stride_elems();

    if (!kFpAct && !act_u8 && in_rs == row_len) {
        pl.ptr = in.ptr<int8_t>();
        pl.n_stride = in_ns;
        pl.row_stride = in_rs;
        pl.ch_stride = channel_last ? 0 : in_cs;
        return Status::ok;
    }

    const int64_t rows = channel_last ? N * IH : N * C * IH;
    const int64_t total = rows * row_len;
    if (!buf.reset(static_cast<size_t>(total))) { return Status::out_of_memory; }
    if (total <= 0) { return Status::ok; }

    pl.ptr = buf.data();
    pl.n_stride = IH * IW * C;
    pl.row_stride = row_len;
    pl.ch_stride = channel_last ? 0 : IH * IW;

    const TAct* src = in.ptr<TAct>();
    int8_t* dst = buf.data();
    const float s = pl.scale;
    const float z = static_cast<float>(pl.zero);

    const auto run = [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            const TAct* sp;
            if (channel_last) {
                sp = src + (r / IH) * in_ns + (r % IH) * in_rs;
            } else {
                const int64_t n = r / (C * IH);
                const int64_t t = r % (C * IH);
                sp = src + n * in_ns + (t / IH) * in_cs + (t % IH) * in_rs;
            }
            int8_t* dp = dst + r * row_len;
            if constexpr (kFpAct) {
                quant_kernel::quantization<int8_t>(1, static_cast<int>(row_len), dp, 0,
                                                   sp, 0, &s, &z);
            } else if constexpr (std::is_same_v<TAct, uint8_t>) {
                for (int64_t k = 0; k < row_len; ++k) {
                    dp[k] = static_cast<int8_t>(static_cast<uint8_t>(sp[k] ^ 0x80u));
                }
            } else {
                std::memcpy(dp, sp, static_cast<size_t>(row_len));
            }
        }
    };

    // The pass is memory-bound and embarrassingly parallel over rows; below a
    // MiB the pool's own dispatch costs more than it saves.
    constexpr int64_t kInlineRows = int64_t{1} << 12;
    const int64_t nt = std::max<int64_t>(1, ctx.cpu.thread_count());
    if (nt == 1 || rows <= kInlineRows) {
        run(0, rows);
        return Status::ok;
    }
    const int64_t slab = (rows + nt - 1) / nt;
    const int64_t nslab = (rows + slab - 1) / slab;
    ctx.cpu.run(0, nslab, [&](int64_t i) {
        run(i * slab, std::min(i * slab + slab, rows));
    });
    return Status::ok;
}

/// (w − zp_w) as int16 lanes: [KH·KW][stride], the channels padded with zeros so
/// a 16-lane load never reaches into the next tap's storage.
Status dwtconv_i8_weight(const TensorView& weight, int64_t KH, int64_t KW,
                         int64_t C, PoolArray<int16_t>& buf, int64_t& stride)
{
    stride = (C + kI8Lane - 1) / kI8Lane * kI8Lane;
    if (C <= 0) { return Status::ok; }
    if (!buf.reset(static_cast<size_t>(KH * KW * stride))) {
        return Status::out_of_memory;
    }

    const int8_t* w_ptr = weight.ptr<int8_t>();
    const int64_t w_kh = weight.stride_elems(0);
    const int64_t w_kw = weight.stride_elems(1);
    const QuantParams& qw = weight.quant_params();

    for (int64_t kh = 0; kh < KH; ++kh) {
        for (int64_t kw = 0; kw < KW; ++kw) {
            int16_t* d = buf.data() + (kh * KW + kw) * stride;
            const int8_t* s = w_ptr + kh * w_kh + kw * w_kw;
            for (int64_t c = 0; c < C; ++c) {
                d[c] = static_cast<int16_t>(
                    static_cast<int32_t>(s[c]) - quant_zp_at(qw, c));
            }
            for (int64_t c = C; c < stride; ++c) { d[c] = 0; }
        }
    }
    return Status::ok;
}

/// Channel-first int8: the lane spans the residue step, as in the fp route.
template <typename TOut>
Status dwtconv_i8_first(const DwtConvGeom& gm, TensorView& output,
                        std::span<const TensorView> inputs,
                        const DwtI8Plane& pl, const ComputeContext& ctx)
{
    constexpr bool kIntOut = std::is_same_v<TOut, int8_t> || std::is_same_v<TOut, uint8_t>;

    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int8_t* w_ptr = weight.ptr<int8_t>();
    const int64_t w_ch = weight.stride_elems(0);
    const int64_t w_kh = weight.stride_elems(2);
    const int64_t w_kw = weight.stride_elems(3);
    const QuantParams& qw = weight.quant_params();

    const int32_t* b_i32 = (has_bias && kIntOut) ? inputs[2].ptr<int32_t>() : nullptr;
    const TOut* b_out = (has_bias && !kIntOut) ? inputs[2].ptr<TOut>() : nullptr;

    TOut* out_ptr = output.ptr<TOut>();
    const int64_t out_n = output.stride_elems(0);
    const int64_t out_row = output.row_stride_elems();
    const int64_t out_ch = output.channel_block_stride_elems();

    const QuantParams& qo = output.quant_params();
    const float out_scale = kIntOut ? quant_scale_at(qo, 0) : 0.0f;
    const int32_t zp_out = kIntOut ? quant_zp_at(qo, 0) : 0;

    const float s_x = pl.scale;
    const int32_t zp_x = pl.zero;
    const int64_t in_row = pl.row_stride;
    const int64_t in_ch = pl.ch_stride;
    const float min_clip = gm.epilogue.min_clip;
    const float max_clip = gm.epilogue.max_clip;
    const bool relu = gm.epilogue.type == EpilogueActivateType::Relu;

    const DwtRowGrid grid = dwt_row_grid(gm, ctx.cpu.thread_count());
    if (grid.total <= 0) { return Status::ok; }
    const int64_t nc = std::min(gm.SW, gm.OW);

    ctx.cpu.run(0, grid.total, [&](int64_t task) {
        const int64_t st = task % grid.strips;
        const int64_t t1 = task / grid.strips;
        const int64_t rb = t1 % grid.row_blocks;
        const int64_t t2 = t1 / grid.row_blocks;
        const int64_t c  = t2 % gm.C;
        const int64_t n  = t2 / gm.C;

        const int64_t oh_begin = rb * grid.rows_per_task;
        const int64_t oh_end   = std::min(gm.OH, oh_begin + grid.rows_per_task);

        const int8_t* in_c = pl.ptr + n * pl.n_stride + c * in_ch;
        TOut* och = out_ptr + n * out_n + c * out_ch;
        const int8_t* w_c = w_ptr + c * w_ch;

        const float s_w = quant_scale_at(qw, c);
        const int32_t zp_w = quant_zp_at(qw, c);
        const float swsx = s_w * s_x;
        const float req = kIntOut ? matmul_int8_requant_scale(s_w, s_x, out_scale) : 0.0f;
        const float bias_f = [&]() -> float {
            if constexpr (kIntOut) {
                return 0.0f;
            } else {
                return has_bias ? s_load(&b_out[c]) : 0.0f;
            }
        }();
        const int32_t bias_i = (has_bias && kIntOut) ? b_i32[c] : 0;

        alignas(32) int32_t lanes[kI8Lane];

        const auto emit = [&](TOut* dst, int32_t acc) {
            if constexpr (kIntOut) {
                int32_t v = acc + bias_i;
                if (relu) { v = std::max(v, 0); }
                *dst = matmul_int8_requant<TOut>(v, req, zp_out);
            } else {
                float f = static_cast<float>(acc) * swsx;
                if (has_bias) { f += bias_f; }
                f = std::min(std::max(f, min_clip), max_clip);
                f = nnops::apply_epilogue(gm.epilogue, f, c);
                s_store(dst, f);
            }
        };

        // A whole 16-lane block at once: the int destination takes the shared row
        // epilogue with the requant scale folded into s_b, so s_a/out_scale is
        // exactly 1 and the expression is the reference's v·req.
        const auto emit_block = [&](TOut* dst) {
            if constexpr (kIntOut) {
                Int8EpilogueRow p{};
                p.s_a = 1.0;
                p.out_scale = 1.0;
                p.s_b_scalar = req;
                p.zp_out = zp_out;
                p.relu = relu;
                p.bias = bias_i;
                int8_epilogue_row_quant<TOut>(lanes, dst, 0, kI8Lane, p);
            } else {
                for (int64_t j = 0; j < kI8Lane; j += 8) {
                    v_f32x8 f = v_cvt_i32_to_f32(lanes + j);
                    f = v_mul(f, v_set1_f32x8(swsx));
                    if (has_bias) { f = v_add(f, v_set1_f32x8(bias_f)); }
                    f = v_max(f, v_set1_f32x8(min_clip));
                    f = v_min(f, v_set1_f32x8(max_clip));
                    f = apply_epilogue_vec<float>(gm.epilogue,
                                                  static_cast<const float*>(nullptr), f);
                    if constexpr (std::is_same_v<TOut, half>) {
                        v_store(dst + j, v_cvt_f32_to_f16(f));
                    } else {
                        v_store(dst + j, f);
                    }
                }
            }
        };

        for (int64_t oh = oh_begin; oh < oh_end; ++oh) {
            const TapList rt = row_taps(gm, oh);
            TOut* orow = och + oh * out_row;
            const int8_t* xrow[kMaxTaps];
            for (int64_t a = 0; a < rt.n; ++a) {
                xrow[a] = in_c + rt.b[a] * in_row;
            }

            for (int64_t r = 0; r < nc; ++r) {
                const int64_t mic = (gm.OW - r + gm.SW - 1) / gm.SW;
                const int64_t per = (mic + grid.strips - 1) / grid.strips;
                const int64_t ts = std::min(mic, st * per);
                const int64_t te = std::min(mic, ts + per);
                if (ts >= te) { continue; }

                const TapList ct = col_taps(gm, r, mic);
                int64_t t_lo = 0;
                int64_t t_hi = mic;
                for (int64_t k = 0; k < ct.n; ++k) {
                    t_lo = std::max(t_lo, -ct.b[k]);
                    t_hi = std::min(t_hi, gm.IW - ct.b[k]);
                }
                if (t_lo > t_hi) { t_lo = t_hi; }

                const auto pixel = [&](int64_t t) {
                    int32_t acc = 0;
                    for (int64_t a = 0; a < rt.n; ++a) {
                        for (int64_t k = 0; k < ct.n; ++k) {
                            const int64_t iw = ct.b[k] + t;
                            if (iw < 0 || iw >= gm.IW) { continue; }
                            const int32_t we = static_cast<int32_t>(
                                w_c[rt.a[a] * w_kh + ct.a[k] * w_kw]) - zp_w;
                            acc += we * (static_cast<int32_t>(xrow[a][iw]) - zp_x);
                        }
                    }
                    emit(orow + r + t * gm.SW, acc);
                };

                // Inside the window every tap is live, so the activation zero
                // point folds once into the seed instead of costing a subtract
                // per lane.
                int32_t sum_w = 0;
                for (int64_t a = 0; a < rt.n; ++a) {
                    for (int64_t k = 0; k < ct.n; ++k) {
                        sum_w += static_cast<int32_t>(
                                     w_c[rt.a[a] * w_kh + ct.a[k] * w_kw]) - zp_w;
                    }
                }
                const int32_t seed = -zp_x * sum_w;

                const int64_t ilo = std::min(std::max(t_lo, ts), te);
                const int64_t ihi = std::min(t_hi, te);
                int64_t t = ts;
                for (; t < ilo; ++t) { pixel(t); }

                for (; t + kI8Lane <= ihi; t += kI8Lane) {
                    I8LaneAcc acc;
                    acc.set1(seed);
                    for (int64_t a = 0; a < rt.n; ++a) {
                        for (int64_t k = 0; k < ct.n; ++k) {
                            acc.madd_s(xrow[a] + ct.b[k] + t,
                                       static_cast<int32_t>(
                                           w_c[rt.a[a] * w_kh + ct.a[k] * w_kw]) - zp_w);
                        }
                    }
                    acc.store(lanes);
                    if (gm.SW == 1) {
                        emit_block(orow + r + t);
                    } else {
                        for (int64_t j = 0; j < kI8Lane; ++j) {
                            emit(orow + r + (t + j) * gm.SW, lanes[j]);
                        }
                    }
                }

                for (; t < ihi; ++t) { pixel(t); }
                for (; t < te; ++t) { pixel(t); }
            }
        }
    });
    return Status::ok;
}

/// Channel-last int8: the lane spans channels, one 16-channel block of one pixel
/// per accumulate step, the filter read from the staged int16 copy.
template <typename TOut>
Status dwtconv_i8_last(const DwtConvGeom& gm, TensorView& output,
                       std::span<const TensorView> inputs, const DwtI8Plane& pl,
                       const int16_t* wq, int64_t wq_c, const ComputeContext& ctx)
{
    constexpr bool kIntOut = std::is_same_v<TOut, int8_t> || std::is_same_v<TOut, uint8_t>;

    const int64_t C = gm.C;
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int32_t* b_i32 = (has_bias && kIntOut) ? inputs[2].ptr<int32_t>() : nullptr;
    const TOut* b_out = (has_bias && !kIntOut) ? inputs[2].ptr<TOut>() : nullptr;
    const QuantParams& qw = weight.quant_params();

    TOut* out_ptr = output.ptr<TOut>();
    const int64_t out_n = output.stride_elems(0);
    const int64_t out_row = output.row_stride_elems();

    const QuantParams& qo = output.quant_params();
    const float out_scale = kIntOut ? quant_scale_at(qo, 0) : 0.0f;
    const int32_t zp_out = kIntOut ? quant_zp_at(qo, 0) : 0;

    const float s_x = pl.scale;
    const int32_t zp_x = pl.zero;
    const int64_t in_row = pl.row_stride;
    const float min_clip = gm.epilogue.min_clip;
    const float max_clip = gm.epilogue.max_clip;
    const bool relu = gm.epilogue.type == EpilogueActivateType::Relu;

    PoolArray<float> req_arr;
    const float* req = nullptr;
    if constexpr (kIntOut) {
        if (!req_arr.reset(static_cast<size_t>(C))) { return Status::out_of_memory; }
        for (int64_t c = 0; c < C; ++c) {
            req_arr[static_cast<size_t>(c)] =
                matmul_int8_requant_scale(quant_scale_at(qw, c), s_x, out_scale);
        }
        req = req_arr.data();
    }

    const int64_t rows = gm.N * gm.OH;
    const int64_t nt = std::max(1, ctx.cpu.thread_count());
    int64_t strips = 1;
    if (rows < nt) {
        const int64_t max_strips = std::max<int64_t>(1, (gm.OW + 3) / 4);
        strips = std::max<int64_t>(1,
                                   std::min<int64_t>((nt + rows - 1) / rows, max_strips));
    }
    const int64_t nc = std::min(gm.SW, gm.OW);

    ctx.cpu.run(0, rows * strips, [&](int64_t task) {
        const int64_t st  = task % strips;
        const int64_t row = task / strips;
        const int64_t oh  = row % gm.OH;
        const int64_t n   = row / gm.OH;

        const int8_t* in_n = pl.ptr + n * pl.n_stride;
        TOut* orow = out_ptr + n * out_n + oh * out_row;
        const TapList rt = row_taps(gm, oh);
        const int8_t* xrow[kMaxTaps];
        for (int64_t a = 0; a < rt.n; ++a) { xrow[a] = in_n + rt.b[a] * in_row; }

        alignas(32) int32_t lanes[kI8Lane];
        alignas(32) float swsx[kI8Lane];

        for (int64_t r = 0; r < nc; ++r) {
            const int64_t mic = (gm.OW - r + gm.SW - 1) / gm.SW;
            const int64_t per = (mic + strips - 1) / strips;
            const int64_t ts = std::min(mic, st * per);
            const int64_t te = std::min(mic, ts + per);
            if (ts >= te) { continue; }

            const TapList ct = col_taps(gm, r, mic);
            int64_t t_lo = 0;
            int64_t t_hi = mic;
            for (int64_t k = 0; k < ct.n; ++k) {
                t_lo = std::max(t_lo, -ct.b[k]);
                t_hi = std::min(t_hi, gm.IW - ct.b[k]);
            }
            if (t_lo > t_hi) { t_lo = t_hi; }

            // One channel at a time, for the fewer-than-16 tail.
            const auto scalar_one = [&](int64_t t, int64_t c) {
                int32_t acc = 0;
                for (int64_t a = 0; a < rt.n; ++a) {
                    for (int64_t k = 0; k < ct.n; ++k) {
                        const int64_t iw = ct.b[k] + t;
                        if (iw < 0 || iw >= gm.IW) { continue; }
                        acc += static_cast<int32_t>(
                                   wq[(rt.a[a] * gm.KW + ct.a[k]) * wq_c + c]) *
                               (static_cast<int32_t>(xrow[a][iw * C + c]) - zp_x);
                    }
                }
                TOut* dst = orow + (r + t * gm.SW) * C + c;
                if constexpr (kIntOut) {
                    int32_t v = acc + (has_bias ? b_i32[c] : 0);
                    if (relu) { v = std::max(v, 0); }
                    *dst = matmul_int8_requant<TOut>(v, req[c], zp_out);
                } else {
                    float f = static_cast<float>(acc) * (quant_scale_at(qw, c) * s_x);
                    if (has_bias) { f += s_load(&b_out[c]); }
                    f = std::min(std::max(f, min_clip), max_clip);
                    f = nnops::apply_epilogue(gm.epilogue, f, c);
                    s_store(dst, f);
                }
            };

            int64_t c0 = 0;
            if (C >= kI8Lane) {
                c0 = (C / kI8Lane) * kI8Lane;
            }

            for (int64_t cb = 0; cb < c0; cb += kI8Lane) {
                for (int64_t j = 0; j < kI8Lane; ++j) {
                    swsx[j] = quant_scale_at(qw, cb + j) * s_x;
                }

                // The tap-sum seed, built once per block: Σ(w − zp_w) over the
                // taps the whole window keeps.
                I8LaneAcc base;
                if (zp_x != 0) {
                    I8LaneAcc sum;
                    sum.set1(0);
                    for (int64_t a = 0; a < rt.n; ++a) {
                        for (int64_t k = 0; k < ct.n; ++k) {
                            sum.add_v(wq + (rt.a[a] * gm.KW + ct.a[k]) * wq_c + cb);
                        }
                    }
                    base.set1(0);
                    base.sub_scaled(sum, zp_x);
                } else {
                    base.set1(0);
                }

                const auto emit_block = [&](int64_t t) {
                    TOut* dst = orow + (r + t * gm.SW) * C + cb;
                    if constexpr (kIntOut) {
                        Int8EpilogueRow p{};
                        p.s_a = 1.0;
                        p.out_scale = 1.0;
                        p.s_b = req + cb;
                        p.zp_out = zp_out;
                        p.relu = relu;
                        p.col_bias = has_bias ? b_i32 + cb : nullptr;
                        int8_epilogue_row_quant<TOut>(lanes, dst, 0, kI8Lane, p);
                    } else {
                        for (int64_t j = 0; j < kI8Lane; j += 8) {
                            v_f32x8 f = v_cvt_i32_to_f32(lanes + j);
                            f = v_mul(f, v_load(swsx + j));
                            if (has_bias) {
                                if constexpr (std::is_same_v<TOut, half>) {
                                    f = v_add(f, v_cvt_f16_to_f32(b_out + cb + j));
                                } else {
                                    f = v_add(f, v_load(b_out + cb + j));
                                }
                            }
                            f = v_max(f, v_set1_f32x8(min_clip));
                            f = v_min(f, v_set1_f32x8(max_clip));
                            f = apply_epilogue_vec<float>(
                                gm.epilogue, static_cast<const float*>(nullptr), f);
                            if constexpr (std::is_same_v<TOut, half>) {
                                v_store(dst + j, v_cvt_f32_to_f16(f));
                            } else {
                                v_store(dst + j, f);
                            }
                        }
                    }
                };

                const auto edge_pixel = [&](int64_t t) {
                    I8LaneAcc acc;
                    acc.set1(0);
                    I8LaneAcc sum;
                    if (zp_x != 0) { sum.set1(0); }
                    for (int64_t a = 0; a < rt.n; ++a) {
                        for (int64_t k = 0; k < ct.n; ++k) {
                            const int64_t iw = ct.b[k] + t;
                            if (iw < 0 || iw >= gm.IW) { continue; }
                            const int16_t* wv =
                                wq + (rt.a[a] * gm.KW + ct.a[k]) * wq_c + cb;
                            acc.madd_v(xrow[a] + iw * C + cb, wv);
                            if (zp_x != 0) { sum.add_v(wv); }
                        }
                    }
                    if (zp_x != 0) { acc.sub_scaled(sum, zp_x); }
                    acc.store(lanes);
                    emit_block(t);
                };

                const int64_t ilo = std::min(std::max(t_lo, ts), te);
                const int64_t ihi = std::min(t_hi, te);
                int64_t t = ts;
                for (; t < ilo; ++t) { edge_pixel(t); }
                for (; t < ihi; ++t) {
                    I8LaneAcc acc = base;
                    for (int64_t a = 0; a < rt.n; ++a) {
                        for (int64_t k = 0; k < ct.n; ++k) {
                            acc.madd_v(xrow[a] + (ct.b[k] + t) * C + cb,
                                       wq + (rt.a[a] * gm.KW + ct.a[k]) * wq_c + cb);
                        }
                    }
                    acc.store(lanes);
                    emit_block(t);
                }
                for (; t < te; ++t) { edge_pixel(t); }
            }

            for (int64_t c = c0; c < C; ++c) {
                for (int64_t t = ts; t < te; ++t) { scalar_one(t, c); }
            }
        }
    });
    return Status::ok;
}

template <typename TOut>
Status dwtconv_i8_layout(const DwtConvGeom& gm, TensorView& output,
                         std::span<const TensorView> inputs,
                         const DwtI8Plane& pl, const ComputeContext& ctx)
{
    if (is_channel_last(inputs[0].layout())) {
        PoolArray<int16_t> wq;
        int64_t wq_c = 0;
        const Status st = dwtconv_i8_weight(inputs[1], gm.KH, gm.KW, gm.C, wq, wq_c);
        if (!is_ok(st)) { return st; }
        return dwtconv_i8_last<TOut>(gm, output, inputs, pl, wq.data(), wq_c, ctx);
    }
    return dwtconv_i8_first<TOut>(gm, output, inputs, pl, ctx);
}

template <class TAct>
Status dwtconv_i8_act(const DwtConvGeom& gm, TensorView& output,
                      std::span<const TensorView> inputs, bool act_u8,
                      const ComputeContext& ctx)
{
    DwtI8Plane pl;
    PoolArray<int8_t> plane;
    const Status st = dwtconv_i8_plane<TAct>(inputs[0],
                                             is_channel_last(inputs[0].layout()),
                                             act_u8, pl, plane, ctx);
    if (!is_ok(st)) { return st; }

    switch (output.data_type()) {
    case DataType::f32: return dwtconv_i8_layout<float>(gm, output, inputs, pl, ctx);
    case DataType::f16: return dwtconv_i8_layout<half>(gm, output, inputs, pl, ctx);
    case DataType::s8:  return dwtconv_i8_layout<int8_t>(gm, output, inputs, pl, ctx);
    default:            return dwtconv_i8_layout<uint8_t>(gm, output, inputs, pl, ctx);
    }
}

/// The quantized gate: which activation/weight/output/bias combinations have an
/// arithmetic here. add_to is refused — a residual on a quantized destination
/// would have to requantize the existing value.
Status depthwise_transpose_conv_int8_cpu(const DepthwiseTransposeConv2DAttributes& attrs,
                                         const DwtConvGeom& gm, TensorView& output,
                                         std::span<const TensorView> inputs,
                                         const ComputeContext& ctx)
{
    const DataType adt = inputs[0].data_type();
    const DataType odt = output.data_type();
    const bool fp_act = adt == DataType::f32 || adt == DataType::f16;
    const bool int_out = odt == DataType::s8 || odt == DataType::u8;

    if (!fp_act && adt != DataType::s8 && adt != DataType::u8) {
        return Status::unsupported;
    }
    if (!int_out && odt != DataType::f32 && odt != DataType::f16) {
        return Status::unsupported;
    }
    if (!int_out && fp_act && odt != adt) { return Status::unsupported; }
    if (inputs.size() > 2 &&
        inputs[2].data_type() != (int_out ? DataType::s32 : odt)) {
        return Status::unsupported;
    }
    if (gm.add_to) { return Status::unsupported; }

    if (gm.N == 0 || gm.C == 0 || gm.OH == 0 || gm.OW == 0) { return Status::ok; }

    // The plane's storage follows the reference: an fp activation is quantized
    // to act_quant_dtype, a pre-quantized one keeps its own.
    const bool act_u8 = fp_act ? (attrs.act_quant_dtype == DataType::u8)
                               : (adt == DataType::u8);

    switch (adt) {
    case DataType::f32: return dwtconv_i8_act<float>(gm, output, inputs, act_u8, ctx);
    case DataType::f16: return dwtconv_i8_act<half>(gm, output, inputs, act_u8, ctx);
    case DataType::u8:  return dwtconv_i8_act<uint8_t>(gm, output, inputs, act_u8, ctx);
    default:            return dwtconv_i8_act<int8_t>(gm, output, inputs, act_u8, ctx);
    }
}

}  // anonymous namespace

// ============================================================
// Entry point with dtype and layout dispatch
// ============================================================

Status depthwise_transpose_conv_cpu(const DepthwiseTransposeConv2DAttributes& attrs,
                                    TensorView& output,
                                    std::span<const TensorView> inputs,
                                    const ComputeContext& ctx,
                                    void* /*workspace*/)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];

    DwtConvGeom gm{};
    gm.N  = input.shape(0);
    gm.C  = input.shape(1);
    gm.IH = input.shape(2);
    gm.IW = input.shape(3);
    gm.OH = output.shape(2);
    gm.OW = output.shape(3);
    gm.KH = attrs.kernel_size[0];
    gm.KW = attrs.kernel_size[1];
    gm.SH = attrs.stride[0];
    gm.SW = attrs.stride[1];
    gm.DH = attrs.dilation[0];
    gm.DW = attrs.dilation[1];
    gm.PH = attrs.padding[0];
    gm.PW = attrs.padding[1];
    gm.epilogue = attrs.epilogue;
    gm.add_to = attrs.add_to;

    const bool last = is_channel_last(input.layout());
    if (!last && !is_channel_first_spatial(input.layout())) {
        return Status::unsupported;
    }

    if (is_quantized_depthwise_transpose_conv2d(weight.data_type())) {
        return depthwise_transpose_conv_int8_cpu(attrs, gm, output, inputs, ctx);
    }
    if (input.data_type() != weight.data_type() ||
        output.data_type() != input.data_type()) {
        return Status::unsupported;
    }

    switch (input.data_type()) {
    case DataType::f32:
        if (last) {
            if (gm.KH > kMaxTaps || gm.KW > kMaxTaps) {
                dwtconv_scalar_last<float>(gm, output, inputs, ctx);
            } else {
                dwtconv_channel_last<float>(gm, output, inputs, ctx);
            }
        } else {
            if (gm.KH > kMaxTaps || gm.KW > kMaxTaps) {
                dwtconv_scalar_first<float>(gm, output, inputs, ctx);
            } else {
                dwtconv_channel_first<float>(gm, output, inputs, ctx);
            }
        }
        return Status::ok;
    case DataType::f16:
        if (last) {
            if (gm.KH > kMaxTaps || gm.KW > kMaxTaps) {
                dwtconv_scalar_last<half>(gm, output, inputs, ctx);
            } else {
                dwtconv_channel_last<half>(gm, output, inputs, ctx);
            }
        } else {
            if (gm.KH > kMaxTaps || gm.KW > kMaxTaps) {
                dwtconv_scalar_first<half>(gm, output, inputs, ctx);
            } else {
                dwtconv_channel_first<half>(gm, output, inputs, ctx);
            }
        }
        return Status::ok;
    default:
        return Status::unsupported;
    }
}

}  // namespace nnops::backend::cpu
