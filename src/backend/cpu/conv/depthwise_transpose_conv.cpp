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

#include <algorithm>
#include <cstdint>

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

    // A row is only OW pixels, which is too little to amortize a claimed item
    // once the pool over-subscribes the grid (OpenMP switches to claimed items
    // past the fast-core count). Grow a task to a block of rows, and when even
    // the row grid cannot fill the pool, split a row's residue steps into strips.
    const int64_t slots = std::max(1, ctx.cpu.thread_count());
    const int64_t total_rows = gm.N * gm.C * gm.OH;
    const int64_t planes = gm.N * gm.C;
    if (total_rows <= 0 || planes <= 0) { return; }
    const int64_t want = 8 * slots;              // eight claims per item slot
    int64_t row_blocks = 1;
    int64_t rows_per_task = gm.OH;
    int64_t strips = 1;
    if (total_rows >= want) {
        row_blocks = std::min<int64_t>(gm.OH, (want + planes - 1) / planes);
        rows_per_task = (gm.OH + row_blocks - 1) / row_blocks;
        row_blocks = (gm.OH + rows_per_task - 1) / rows_per_task;
    } else {
        const int64_t max_strips = std::max<int64_t>(1, (gm.OW + 3) / 4);
        strips = std::min<int64_t>((want + total_rows - 1) / total_rows, max_strips);
    }

    ctx.cpu.run(0, gm.N * gm.C * row_blocks * strips, [&](int64_t task) {
        const int64_t st = task % strips;
        const int64_t t1 = task / strips;
        const int64_t rb = t1 % row_blocks;
        const int64_t t2 = t1 / row_blocks;
        const int64_t c  = t2 % gm.C;
        const int64_t n  = t2 / gm.C;

        const int64_t oh_begin = rb * rows_per_task;
        const int64_t oh_end   = std::min(gm.OH, oh_begin + rows_per_task);

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

    if (input.data_type() != weight.data_type() ||
        output.data_type() != input.data_type()) {
        return Status::unsupported;
    }

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
