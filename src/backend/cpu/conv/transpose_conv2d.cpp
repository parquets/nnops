/// @file transpose_conv2d.cpp
/// @brief CPU implementation of 2D transposed convolution, channel-first and
///        channel-last.
///
/// Gather formulation (output-driven), the exact inverse of regular conv's
/// indexing: output (oh, ow) reads input (ih, iw) where
/// oh + PH - kh·DH == ih·SH (and likewise for w), so every output pixel is
/// accumulated independently and add_to needs no save/restore pass — in either
/// order: residual_pre_activation reads the destination into the sum before
/// the epilogue at every merge site. Weights are
/// Channel-first [IC, OC/G, KH, KW] either way; the image tensor is channel-first or
/// channel-last.
///
/// Channel-first runs the onnxruntime structure: for each (n, group) the group's
/// filter slice is a [OC/G·KH·KW] × [IC/G] matrix and the whole input plane is
/// [IC/G] × [IH·IW], so one GEMM produces the col matrix — every (oc, kh, kw)
/// row over every input pixel, the channel reduction already done. A scatter
/// then adds each tap's col row into the output plane, which is one contiguous
/// add per tap once the output row is walked by stride residue. That scatter is
/// where the layout bites: a pixel of residue r sits at r + t·SW, so the row is
/// accumulated class-major in a scratch row and merged with the epilogue at the
/// end — the merge is the only place that writes output, which also keeps
/// add_to's original value intact. The GEMM counts the input-tap pairs, not the
/// output-tap ones, so a stride-2 layer costs what it should.
///
/// Channel-last has three routes. The pixel kernel repacks the weight once to
/// [G][KH][KW][IC/G][OC/G] (so consecutive output channels are contiguous) and
/// dispatches per (n, oh row), accumulating the whole OC/G vector with SIMD
/// since an input pixel's channels are contiguous too. Two per-class facts are
/// hoisted out of the pixel loop: which (kh, ih) rows land on this output row,
/// and which (kw, shift) taps land on a given output-column residue. That
/// leaves a valid window [t_lo, t_hi) in which every tap of the class is in
/// range, and there the row is walked four pixels at a time against two oc
/// vectors — one weight vector then serves four pixels. Outside the window a
/// single pixel goes through the general path.
///
/// That reuse only survives while the filter does: a large filter is re-read
/// from DRAM on every tile, and a stride-1 filter has many taps per output
/// pixel, so both go through the col matrix instead — the same one GEMM per
/// (n, group) as the channel-first route, with the image as [npix][IC/G] and the
/// filter repacked to [IC/G][KH·KW·OC/G], followed by a stride-residue scatter
/// that adds whole oc vectors. Depthwise (one channel per group) has its own
/// route: group g's input channel *is* output channel g, so the vector axis is
/// the channel and every tap is one contiguous vector add over C.
///
/// Both GEMM routes are guarded on dense operands (a flat matrix view needs
/// both rows to be their logical width); anything else — and any kernel too
/// tall for the per-row tap list — falls back to the plain scalar gather, which
/// is correct for every shape.

#include "nnops/ops/transpose_conv2d.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"
#include "backend/cpu/simd_kernel/simd_quant.hpp"   // quant_kernel::quantization
#include "backend/cpu/matmul/matmul.h"
#include "backend/cpu/common/dtype_dispatch.hpp"
#include "backend/cpu/common/memory_pool.hpp"

#include <algorithm>
#include <cstring>
#include <type_traits>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

/// One transposed convolution's geometry, read off the tensors once.
struct Geom {
    int64_t N, IC, IH, IW;
    int64_t OC_per_G, IC_per_G, OC, G;
    int64_t KH, KW, OH, OW;
    int64_t SH, SW, DH, DW, PH, PW;
    Epilogue epilogue;
    bool add_to;
    bool pre_act;   ///< fold add_to's old value in before the epilogue
};

/// Largest col one GEMM call may materialise; past it the M rows are split.
constexpr int64_t kColCapBytes = 8 << 20;

/// Tallest per-row tap list the channel-first scatter keeps on the stack.
constexpr int64_t kMaxRowTaps = 16;

/// row[0, n) += src[0, n), element-wise.
template <typename T>
void row_add(T* dst, const T* src, int64_t n)
{
    constexpr int L = simd_lane_for<T>;
    int64_t i = 0;
    for (; i + L <= n; i += L) {
        v_store(dst + i, v_add(v_load(dst + i), v_load(src + i)));
    }
    for (; i < n; ++i) { s_store(dst + i, s_load(dst + i) + s_load(src + i)); }
}

/// row[0, n) += src[0, n) · w[0, n).
template <typename T>
void row_fma(T* dst, const T* src, const T* w, int64_t n)
{
    constexpr int L = simd_lane_for<T>;
    int64_t i = 0;
    for (; i + L <= n; i += L) {
        v_store(dst + i, v_fmadd(v_load(src + i), v_load(w + i), v_load(dst + i)));
    }
    for (; i < n; ++i) { s_store(dst + i, s_load(dst + i) + s_load(src + i) * s_load(w + i)); }
}

/// row[0, n) = v.
template <typename T>
void row_fill(T* dst, int64_t n, float v)
{
    constexpr int L = simd_lane_for<T>;
    const T* tag = nullptr;
    const vec_for<T> fill = v_set1(tag, v);
    int64_t i = 0;
    for (; i + L <= n; i += L) { v_store(dst + i, fill); }
    for (; i < n; ++i) { s_store(dst + i, v); }
}

/// Channel-first through a col matrix — the onnxruntime route. Returns false when the
/// operands cannot be read as flat matrices, leaving the caller its scalar path.
template <typename T>
bool tconv2d_channel_first_gemm(const Geom& gm,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    if (weight.row_stride_elems() != gm.KW) { return false; }
    if (input.row_stride_elems() != gm.IW) { return false; }
    if (gm.KH > kMaxRowTaps) { return false; }

    const int64_t karea = gm.KH * gm.KW;
    const int64_t npix  = gm.IH * gm.IW;
    const int64_t icg   = gm.IC_per_G;
    const int64_t ocg   = gm.OC_per_G;

    const int64_t in_n_stride    = input.stride_elems(0);
    const int64_t out_n_stride   = output.stride_elems(0);
    const int64_t out_c_stride   = output.stride_elems(1);
    const int64_t out_row_stride = output.row_stride_elems();

    const auto* in_ptr = input.ptr<T>();
    const auto* w_ptr  = weight.ptr<T>();
    const auto* b_ptr  = has_bias ? inputs[2].ptr<T>() : nullptr;
    auto* out_ptr = output.ptr<T>();

    // A stride-SW output row splits into SW residue classes of `mic` pixels;
    // each class's taps read a contiguous run of the col.
    const int64_t mic     = (gm.OW + gm.SW - 1) / gm.SW;
    const int64_t row_len = gm.SW * mic;
    const int64_t slots   = std::max(1, ctx.cpu.worker_count());

    const int64_t elems_per_oc = karea * npix;
    const int64_t cap = kColCapBytes / static_cast<int64_t>(sizeof(T));
    const int64_t chunk = std::max<int64_t>(
        1, std::min(ocg, cap / std::max<int64_t>(elems_per_oc, 1)));

    PoolArray<T> col(static_cast<size_t>(chunk * elems_per_oc));
    PoolArray<T> rows(static_cast<size_t>(slots * row_len));
    if (!col || !rows) { return false; }

    const DataType vdt = input.data_type();

    MatMulAttributes mm{};
    mm.transpose_a = true;

    for (int64_t n = 0; n < gm.N; ++n) {
        const T* in_n = in_ptr + n * in_n_stride;
        T* out_n = out_ptr + n * out_n_stride;

        for (int64_t g = 0; g < gm.G; ++g) {
            const T* w_g = w_ptr + g * icg * (ocg * karea);
            const T* b_g = b_ptr ? b_ptr + g * ocg : nullptr;

            // B: the group's whole input plane, [IC/G] × [IH·IW].
            const int64_t bshape[] = {icg, npix};
            TensorView bm(std::span<const int64_t>(bshape, 2), vdt,
                          const_cast<T*>(in_n + g * icg * npix),
                          static_cast<int64_t>(npix * sizeof(T)), TensorLayout::NCHW);

            for (int64_t oc0 = 0; oc0 < ocg; oc0 += chunk) {
                const int64_t cc = std::min(chunk, ocg - oc0);
                const int64_t m  = cc * karea;

                // A: the filter rows of this oc slice, [IC/G] × [OC/G·KH·KW],
                // read through the group's own row stride.
                const int64_t ashape[] = {icg, m};
                TensorView am(std::span<const int64_t>(ashape, 2), vdt,
                              const_cast<T*>(w_g + oc0 * karea),
                              static_cast<int64_t>(ocg * karea * sizeof(T)),
                              TensorLayout::NCHW);
                const int64_t cshape[] = {m, npix};
                TensorView cm(std::span<const int64_t>(cshape, 2), vdt,
                              col.data(), static_cast<int64_t>(npix * sizeof(T)),
                              TensorLayout::NCHW);

                const TensorView mins[] = {am, bm};
                matmul_cpu(mm, cm, mins, ctx, nullptr);

                ctx.cpu.run(0, cc * gm.OH, [&](int64_t task) {
                    const int64_t oc_l = task / gm.OH;
                    const int64_t oh   = task % gm.OH;
                    T* row = rows.data() +
                             static_cast<int64_t>(ctx.cpu.current_thread_id()) * row_len;

                    row_fill(row, row_len, b_g ? s_load(&b_g[oc_l]) : 0.0f);

                    // Which kernel rows land on this output row.
                    int64_t kh_of[kMaxRowTaps];
                    int64_t ih_of[kMaxRowTaps];
                    int64_t taps = 0;
                    for (int64_t kh = 0; kh < gm.KH; ++kh) {
                        const int64_t nh = oh + gm.PH - kh * gm.DH;
                        if (nh < 0 || nh % gm.SH != 0) { continue; }
                        const int64_t ih = nh / gm.SH;
                        if (ih >= gm.IH) { continue; }
                        kh_of[taps] = kh;
                        ih_of[taps] = ih;
                        ++taps;
                    }

                    for (int64_t ti = 0; ti < taps; ++ti) {
                        const T* base = col.data() +
                            (oc_l * karea + kh_of[ti] * gm.KW) * npix +
                            ih_of[ti] * gm.IW;
                        for (int64_t kw = 0; kw < gm.KW; ++kw) {
                            const int64_t d = gm.PW - kw * gm.DW;
                            for (int64_t r = 0; r < gm.SW; ++r) {
                                if (r >= gm.OW) { break; }
                                const int64_t nd = r + d;
                                // A negative offset is not a reason to drop the
                                // tap: pixel t of the class adds t·SW, so it
                                // starts to apply at t = -shift, and tlo is
                                // exactly that.
                                if (nd % gm.SW != 0) { continue; }
                                const int64_t shift = nd / gm.SW;
                                const int64_t nr = (gm.OW - r + gm.SW - 1) / gm.SW;
                                const int64_t tlo = std::max<int64_t>(0, -shift);
                                const int64_t thi = std::min<int64_t>(nr, gm.IW - shift);
                                if (thi <= tlo) { continue; }
                                row_add(row + r * mic + tlo,
                                        base + kw * npix + shift + tlo,
                                        thi - tlo);
                            }
                        }
                    }

                    const int64_t oc = g * ocg + oc0 + oc_l;
                    T* orow = out_n + oc * out_c_stride + oh * out_row_stride;
                    for (int64_t r = 0; r < gm.SW; ++r) {
                        if (r >= gm.OW) { break; }
                        const int64_t nr = (gm.OW - r + gm.SW - 1) / gm.SW;
                        const T* rp = row + r * mic;
                        for (int64_t t = 0; t < nr; ++t) {
                            const int64_t ow = r + t * gm.SW;
                            float a = s_load(&rp[t]);
                            if (gm.pre_act) { a += s_load(&orow[ow]); }
                            float v = nnops::apply_epilogue(gm.epilogue, a, oc);
                            if (gm.add_to && !gm.pre_act) { v += s_load(&orow[ow]); }
                            s_store(&orow[ow], v);
                        }
                    }
                });
            }
        }
    }
    return true;
}

/// Depthwise channel-last. One channel per group means group g's input channel
/// is output channel g, so the vector axis is the channel itself: an output
/// pixel accumulates the whole C-wide tap vector at once. The filter is
/// repacked to [KH·KW][C] so a tap is one contiguous vector. Returns false for
/// anything that is not depthwise, or when the row scratch would not fit.
template <typename T>
bool tconv2d_channel_last_depthwise(const Geom& gm,
                                    TensorView& output,
                                    std::span<const TensorView> inputs,
                                    const ComputeContext& ctx)
{
    if (gm.IC_per_G != 1 || gm.OC_per_G != 1) { return false; }

    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    if (weight.row_stride_elems() != gm.KW) { return false; }
    if (input.row_stride_elems() != gm.IW * gm.IC) { return false; }

    const int64_t karea   = gm.KH * gm.KW;
    const int64_t C       = gm.IC;
    const int64_t slots   = std::max(1, ctx.cpu.worker_count());
    const int64_t row_len = gm.OW * C;

    const int64_t elem_cap = kColCapBytes / static_cast<int64_t>(sizeof(T));
    if (slots * row_len > elem_cap) { return false; }

    PoolArray<T> wd(static_cast<size_t>(karea * C));
    PoolArray<T> rows(static_cast<size_t>(slots * row_len));
    if (!wd || !rows) { return false; }

    const auto* w_ptr = weight.ptr<T>();
    for (int64_t c = 0; c < C; ++c) {
        for (int64_t tap = 0; tap < karea; ++tap) {
            wd[static_cast<size_t>(tap * C + c)] = w_ptr[c * karea + tap];
        }
    }

    const auto* in_ptr = input.ptr<T>();
    const auto* b_ptr  = has_bias ? inputs[2].ptr<T>() : nullptr;
    auto* out_ptr = output.ptr<T>();
    const int64_t in_n_stride    = input.stride_elems(0);
    const int64_t in_npitch      = input.row_stride_elems();
    const int64_t out_n_stride   = output.stride_elems(0);
    const int64_t out_row_stride = output.row_stride_elems();

    const auto epilogue = gm.epilogue;
    const T* type_tag = nullptr;
    constexpr int L = simd_lane_for<T>;

    ctx.cpu.run(0, gm.N * gm.OH, [&](int64_t task) {
        const int64_t n  = task / gm.OH;
        const int64_t oh = task % gm.OH;
        T* row = rows.data() +
                 static_cast<int64_t>(ctx.cpu.current_thread_id()) * row_len;
        const T* in_n = in_ptr + n * in_n_stride;

        row_fill(row, row_len, 0.0f);

        for (int64_t kh = 0; kh < gm.KH; ++kh) {
            const int64_t num_h = oh + gm.PH - kh * gm.DH;
            if (num_h < 0 || num_h % gm.SH != 0) { continue; }
            const int64_t ih = num_h / gm.SH;
            if (ih >= gm.IH) { continue; }
            const T* xrow = in_n + ih * in_npitch;

            for (int64_t kw = 0; kw < gm.KW; ++kw) {
                const T* wrow = wd.data() + (kh * gm.KW + kw) * C;
                for (int64_t r = 0; r < gm.SW && r < gm.OW; ++r) {
                    const int64_t num_w = r + gm.PW - kw * gm.DW;
                    if (num_w % gm.SW != 0) { continue; }
                    const int64_t sh  = num_w / gm.SW;
                    const int64_t mic = (gm.OW - r + gm.SW - 1) / gm.SW;
                    const int64_t tlo = std::max<int64_t>(0, -sh);
                    const int64_t thi = std::min<int64_t>(mic, gm.IW - sh);
                    for (int64_t t = tlo; t < thi; ++t) {
                        row_fma(row + (r + t * gm.SW) * C, xrow + (t + sh) * gm.IC,
                                wrow, C);
                    }
                }
            }
        }

        T* orow = out_ptr + n * out_n_stride + oh * out_row_stride;
        for (int64_t ow = 0; ow < gm.OW; ++ow) {
            const T* rp = row + ow * C;
            T* dp = orow + ow * gm.OC;
            int64_t c = 0;
            for (; c + L <= C; c += L) {
                vec_for<T> acc = v_load(rp + c);
                if (b_ptr) { acc = v_add(acc, v_load(b_ptr + c)); }
                if (gm.pre_act) { acc = v_add(acc, v_load(dp + c)); }
                acc = apply_epilogue_vec<T>(epilogue, type_tag, acc);
                if (gm.add_to && !gm.pre_act) { acc = v_add(acc, v_load(dp + c)); }
                v_store(dp + c, acc);
            }
            for (; c < C; ++c) {
                float v = s_load(rp + c);
                if (b_ptr) { v += s_load(b_ptr + c); }
                if (gm.pre_act) { v += s_load(dp + c); }
                v = nnops::apply_epilogue(epilogue, v, c);
                if (gm.add_to && !gm.pre_act) { v += s_load(dp + c); }
                s_store(dp + c, v);
            }
        }
    });
    return true;
}

/// Channel-last through a col matrix — onnxruntime's channels-last structure,
/// and what MNN's NC4HW4 deconvolution does: one GEMM per (batch, group) does
/// the channel reduction for the whole input plane, so the filter is walked
/// once per group instead of once per pixel tile, and a stride-residue scatter
/// then adds whole oc vectors out of the col. That pass over the filter is the
/// entire point on a layer where the pixel kernel's re-read goes to DRAM.
/// Returns false when the operands cannot be read as flat matrices.
template <typename T>
bool tconv2d_channel_last_gemm(const Geom& gm,
                               TensorView& output,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    if (weight.row_stride_elems() != gm.KW) { return false; }
    if (input.row_stride_elems() != gm.IW * gm.IC) { return false; }

    const int64_t karea = gm.KH * gm.KW;
    const int64_t npix  = gm.IH * gm.IW;
    const int64_t icg   = gm.IC_per_G;
    const int64_t ocg   = gm.OC_per_G;
    if (npix <= 0 || karea <= 0 || icg <= 0) { return false; }

    const int64_t in_n_stride    = input.stride_elems(0);
    const int64_t out_n_stride   = output.stride_elems(0);
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t slots          = std::max(1, ctx.cpu.worker_count());

    // Two col orientations. Ordered [tap][oc] the GEMM needs the filter
    // permuted to [IC/G][KH·KW·OC/G], but the col feeds the scatter as it is.
    // Ordered [oc][tap] the raw filter slice *is* B — oc runs faster than tap
    // in the channel-first weight already — and the col pays a transpose instead, per
    // batch item. One of the two is walked twice either way, so walk the
    // smaller: the filter once, or the col per batch item.
    const bool direct_w = npix * gm.N < icg;

    // The col and the per-worker row scratch both scale with the oc chunk.
    const int64_t elem_cap = kColCapBytes / static_cast<int64_t>(sizeof(T));
    const int64_t cap = direct_w ? elem_cap / 2 : elem_cap;
    int64_t cc = ocg;
    cc = std::min(cc, std::max<int64_t>(1, cap / (npix * karea)));
    cc = std::min(cc, std::max<int64_t>(1, cap / (slots * gm.OW)));

    PoolArray<T> col(static_cast<size_t>(npix * karea * cc));
    PoolArray<T> aux(static_cast<size_t>((direct_w ? npix : icg) * karea * cc));
    PoolArray<T> rows(static_cast<size_t>(slots * gm.OW * cc));
    if (!col || !aux || !rows) { return false; }

    const auto* in_ptr = input.ptr<T>();
    const auto* w_ptr  = weight.ptr<T>();
    const auto* b_ptr  = has_bias ? inputs[2].ptr<T>() : nullptr;
    auto* out_ptr = output.ptr<T>();
    const DataType vdt = input.data_type();
    const auto epilogue = gm.epilogue;
    const T* type_tag = nullptr;
    constexpr int L = simd_lane_for<T>;

    MatMulAttributes mm{};

    for (int64_t g = 0; g < gm.G; ++g) {
        for (int64_t oc0 = 0; oc0 < ocg; oc0 += cc) {
            const int64_t nc = std::min(cc, ocg - oc0);
            if (nc <= 0) { continue; }
            const int64_t m = nc * karea;

            if (!direct_w) {
                // B: the group's filter slice as [IC/G] × [KH·KW·nc], a blocked
                // transpose of the channel-first [IC/G][OC/G][KH·KW] rows.
                for (int64_t ic = 0; ic < icg; ++ic) {
                    const T* src = w_ptr + ((g * icg + ic) * ocg + oc0) * karea;
                    T* dst = aux.data() + ic * karea * nc;
                    constexpr int64_t kB = 8;
                    for (int64_t o0 = 0; o0 < nc; o0 += kB) {
                        const int64_t o1 = std::min(o0 + kB, nc);
                        for (int64_t t0 = 0; t0 < karea; t0 += kB) {
                            const int64_t t1 = std::min(t0 + kB, karea);
                            for (int64_t o = o0; o < o1; ++o) {
                                T* d = dst + t0 * nc + o;
                                const T* s = src + o * karea + t0;
                                for (int64_t t = 0; t < t1 - t0; ++t) { d[t * nc] = s[t]; }
                            }
                        }
                    }
                }
            }

            // B is the filter slice as [IC/G] × [KH·KW·nc]: permuted, or the
            // Channel-first rows read in place with oc still the faster axis.
            const int64_t bshape[] = {icg, m};
            TensorView bm(std::span<const int64_t>(bshape, 2), vdt,
                          direct_w
                              ? const_cast<T*>(w_ptr + (g * icg * ocg + oc0) * karea)
                              : aux.data(),
                          (direct_w ? ocg * karea : m) * sizeof(T),
                          TensorLayout::NCHW);

            for (int64_t n = 0; n < gm.N; ++n) {
                const T* in_n = in_ptr + n * in_n_stride;
                T* out_n = out_ptr + n * out_n_stride;

                // A: the channel-last image as [IH·IW] × [IC/G] pixels — one pixel
                // stride of IC, so the group offset is g·IC/G elements in.
                const int64_t ashape[] = {npix, icg};
                TensorView am(std::span<const int64_t>(ashape, 2), vdt,
                              const_cast<T*>(in_n + g * icg),
                              static_cast<int64_t>(gm.IC * sizeof(T)),
                              TensorLayout::NCHW);
                const int64_t cshape[] = {npix, m};
                TensorView cm(std::span<const int64_t>(cshape, 2), vdt,
                              col.data(), static_cast<int64_t>(m * sizeof(T)),
                              TensorLayout::NCHW);

                const TensorView mins[] = {am, bm};
                matmul_cpu(mm, cm, mins, ctx, nullptr);

                const T* cbase = col.data();
                if (direct_w) {
                    // Per pixel, [oc][tap] → [tap][oc], so the scatter sees the
                    // same col either way.
                    constexpr int64_t kB = 8;
                    for (int64_t p = 0; p < npix; ++p) {
                        const T* sp = col.data() + p * m;
                        T* dp = aux.data() + p * m;
                        for (int64_t o0 = 0; o0 < nc; o0 += kB) {
                            const int64_t o1 = std::min(o0 + kB, nc);
                            for (int64_t t0 = 0; t0 < karea; t0 += kB) {
                                const int64_t t1 = std::min(t0 + kB, karea);
                                for (int64_t o = o0; o < o1; ++o) {
                                    T* d = dp + t0 * nc + o;
                                    const T* s = sp + o * karea + t0;
                                    for (int64_t t = 0; t < t1 - t0; ++t) {
                                        d[t * nc] = s[t];
                                    }
                                }
                            }
                        }
                    }
                    cbase = aux.data();
                }

                ctx.cpu.run(0, gm.OH, [&](int64_t oh) {
                    T* row = rows.data() +
                             static_cast<int64_t>(ctx.cpu.current_thread_id()) *
                             (gm.OW * nc);
                    row_fill(row, gm.OW * nc, 0.0f);

                    for (int64_t kh = 0; kh < gm.KH; ++kh) {
                        const int64_t num_h = oh + gm.PH - kh * gm.DH;
                        if (num_h < 0 || num_h % gm.SH != 0) { continue; }
                        const int64_t ih = num_h / gm.SH;
                        if (ih >= gm.IH) { continue; }
                        const T* ihrow = cbase + ih * gm.IW * m;

                        for (int64_t kw = 0; kw < gm.KW; ++kw) {
                            const T* colw = ihrow + (kh * gm.KW + kw) * nc;
                            for (int64_t r = 0; r < gm.SW && r < gm.OW; ++r) {
                                const int64_t num_w = r + gm.PW - kw * gm.DW;
                                if (num_w % gm.SW != 0) { continue; }
                                const int64_t sh  = num_w / gm.SW;
                                const int64_t mic = (gm.OW - r + gm.SW - 1) / gm.SW;
                                const int64_t tlo = std::max<int64_t>(0, -sh);
                                const int64_t thi = std::min<int64_t>(mic, gm.IW - sh);
                                for (int64_t t = tlo; t < thi; ++t) {
                                    row_add(row + (r + t * gm.SW) * nc,
                                            colw + (t + sh) * m, nc);
                                }
                            }
                        }
                    }

                    // Merge: the only writer of this output row, which is
                    // also what keeps add_to's original value.
                    T* orow = out_n + oh * out_row_stride + g * ocg + oc0;
                    for (int64_t ow = 0; ow < gm.OW; ++ow) {
                        const T* rp = row + ow * nc;
                        T* dp = orow + ow * gm.OC;
                        int64_t c = 0;
                        for (; c + L <= nc; c += L) {
                            vec_for<T> acc = v_load(rp + c);
                            if (b_ptr) {
                                acc = v_add(acc, v_load(b_ptr + g * ocg + oc0 + c));
                            }
                            if (gm.pre_act) { acc = v_add(acc, v_load(dp + c)); }
                            acc = apply_epilogue_vec<T>(epilogue, type_tag, acc);
                            if (gm.add_to && !gm.pre_act) {
                                acc = v_add(acc, v_load(dp + c));
                            }
                            v_store(dp + c, acc);
                        }
                        for (; c < nc; ++c) {
                            float v = s_load(rp + c);
                            if (b_ptr) { v += s_load(b_ptr + g * ocg + oc0 + c); }
                            if (gm.pre_act) { v += s_load(dp + c); }
                            v = nnops::apply_epilogue(epilogue, v,
                                                      g * ocg + oc0 + c);
                            if (gm.add_to && !gm.pre_act) { v += s_load(dp + c); }
                            s_store(dp + c, v);
                        }
                    }
                });
            }
        }
    }
    return true;
}

// =========================================================================
//  Quantized route — s8 weight (per-channel), u8/s8/f32/f16 activation
// =========================================================================
//
// The two col-matrix routes above, with the int32 accumulate as the col: the
// quantized GEMM (MatMulInteger) reduces the input channels and withdraws both
// zero points, so the scatter adds raw int32 and the merge does what the fp
// merge did — bias, clip, epilogue — under a dequantize or a requantize. The
// geometry does not change, which is why each scatter below is the fp scatter
// element for element.
//
// The activation is normalised to one s8 plane per call so the GEMM sees a
// single zero-point convention: an fp activation is quantized into it (its
// act_quant_dtype picks the clamp range), and a u8 activation is shifted to s8
// with its zero point moved by -128 — the same value, one convention. A
// pre-quantized s8 activation whose rows are dense is read in place.
//
// The weight keeps its per-channel parameters. The GEMM's reduction axis is the
// input channel, so an output channel owns karea consecutive rows (channel-
// first) or columns (channel-last) and its (scale, zero point) has to be
// replicated across them — the one thing the operand views need that the raw
// tensors do not have. Replication is cheap and per chunk: the taps of one
// channel always sit together.

namespace k = nnops::kernel;

/// row[0, n) += src[0, n) over the int32 col (simd has no int32 vector).
void row_add_i32(int32_t* dst, const int32_t* src, int64_t n)
{
    for (int64_t i = 0; i < n; ++i) { dst[i] += src[i]; }
}

/// One (scale, zero point) for the whole tensor.
inline QuantParams qp_per_tensor(float scale, int32_t zero)
{
    QuantParams q;
    q.scale = scale;
    q.zero_point = zero;
    return q;
}

/// One (scale, zero point) per row (channel-first) or column (channel-last).
inline QuantParams qp_per_channel(const float* scale, const int32_t* zero, int64_t n)
{
    QuantParams q;
    q.granularity = QuantGranularity::PerChannel;
    q.scale_data = scale;
    q.zero_point_data = zero;
    q.num_scales = n;
    return q;
}

/// The activation the int8 GEMM reads: an s8 plane (or the input itself when it
/// is already s8 and dense) and the single (scale, zero point) it carries.
struct ActPlane {
    const int8_t* ptr = nullptr;
    int64_t n_stride = 0;   ///< elements per batch item
    float scale = 1.0f;
    int32_t zero = 0;
};

/// Quantize/shift the activation into a dense s8 plane in the layout the route
/// reads: channel-first is channel-major [N][IC][IH·IW], channel-last is
/// pixel-major [N][IH·IW][IC]. @p zero is the plane's zero point (-128 from the
/// stored one when the storage was unsigned).
template <class TAct>
void build_activation_plane(const TensorView& in, int8_t* dst, float scale,
                            int32_t zero, bool channel_last, int64_t IC,
                            int64_t IH, int64_t IW, const ComputeContext& ctx)
{
    const int64_t N  = in.shape(0);
    const int64_t rs = in.row_stride_elems();
    const int64_t ns = in.stride_elems(0);
    const int64_t cs = channel_last ? 0 : in.stride_elems(1);
    const int64_t row_len = channel_last ? IW * IC : IW;
    const int64_t rows_n  = channel_last ? IH : IC * IH;
    const int64_t total   = N * rows_n;
    const TAct* src = in.ptr<TAct>();
    const float s = scale;
    const float z = static_cast<float>(zero);

    const auto run_rows = [&](int64_t beg, int64_t end) {
        for (int64_t r = beg; r < end; ++r) {
            const int64_t nb = r / rows_n;
            const int64_t i  = r - nb * rows_n;
            const int64_t h  = channel_last ? i : i % IH;
            const int64_t c  = channel_last ? 0 : i / IH;
            const TAct* sp = src + nb * ns + c * cs + h * rs;
            int8_t* dp = dst + (nb * rows_n + i) * row_len;
            if constexpr (std::is_same_v<TAct, int8_t>) {
                std::memcpy(dp, sp, static_cast<size_t>(row_len));
            } else if constexpr (std::is_same_v<TAct, uint8_t>) {
                for (int64_t k = 0; k < row_len; ++k) {
                    dp[k] = static_cast<int8_t>(static_cast<uint8_t>(sp[k]) ^ 0x80u);
                }
            } else {
                k::quant_kernel::quantization<int8_t>(
                    static_cast<int>(1), static_cast<int>(row_len), dp, 1, sp, 1,
                    &s, &z);
            }
        }
    };

    // Below this a dispatch costs more than the pass (see quantize_plane).
    constexpr int64_t kInlineMax = int64_t{1} << 19;
    const int64_t nt = std::max<int64_t>(ctx.cpu.thread_count(), 1);
    if (nt == 1 || total <= kInlineMax) {
        run_rows(0, total);
        return;
    }
    const int64_t slab = (total + nt - 1) / nt;
    const int64_t nslab = (total + slab - 1) / slab;
    ctx.cpu.run(0, nslab, [&](int64_t i) {
        run_rows(i * slab, std::min(i * slab + slab, total));
    });
}

/// Channel-first quantized route: one int8 GEMM per (n, group, oc chunk) gives
/// every (oc, tap) row over every input pixel, and the fp route's scatter adds
/// those int32 rows into a scratch row before the merge requantizes.
template <typename TOut>
bool tconv2d_int8_channel_first(const Geom& gm, TensorView& output,
                                std::span<const TensorView> inputs,
                                const ActPlane& act, const ComputeContext& ctx)
{
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;
    constexpr bool kIntOut =
        std::is_same_v<TOut, int8_t> || std::is_same_v<TOut, uint8_t>;

    if (weight.row_stride_elems() != gm.KW) { return false; }
    if (gm.KH > kMaxRowTaps) { return false; }

    const int64_t karea = gm.KH * gm.KW;
    const int64_t npix  = gm.IH * gm.IW;
    const int64_t icg   = gm.IC_per_G;
    const int64_t ocg   = gm.OC_per_G;
    if (npix <= 0 || icg <= 0 || ocg <= 0) { return false; }

    const int64_t out_n_stride   = output.stride_elems(0);
    const int64_t out_c_stride   = output.stride_elems(1);
    const int64_t out_row_stride = output.row_stride_elems();

    const int64_t mic     = (gm.OW + gm.SW - 1) / gm.SW;
    const int64_t row_len = gm.SW * mic;
    const int64_t slots   = std::max(1, ctx.cpu.worker_count());

    const int64_t elems_per_oc = karea * npix;
    const int64_t cap = kColCapBytes / static_cast<int64_t>(sizeof(int32_t));
    const int64_t chunk = std::max<int64_t>(
        1, std::min(ocg, cap / std::max<int64_t>(elems_per_oc, 1)));

    PoolArray<int32_t> col(static_cast<size_t>(chunk * elems_per_oc));
    PoolArray<int32_t> rows(static_cast<size_t>(slots * row_len));
    PoolArray<float> w_scale(static_cast<size_t>(chunk * karea));
    PoolArray<int32_t> w_zp(static_cast<size_t>(chunk * karea));
    if (!col || !rows || !w_scale || !w_zp) { return false; }

    const auto* w_ptr = weight.ptr<int8_t>();
    const QuantParams& qw = weight.quant_params();
    const auto* b_i32 = (has_bias && kIntOut) ? inputs[2].ptr<int32_t>() : nullptr;
    const TOut* b_fp = (has_bias && !kIntOut) ? inputs[2].ptr<TOut>() : nullptr;
    const float s_x = act.scale;
    const QuantParams act_qp = qp_per_tensor(s_x, act.zero);

    const QuantParams& qo = output.quant_params();
    const float out_scale = kIntOut ? quant_scale_at(qo, 0) : 0.0f;
    const int32_t zp_out = kIntOut ? quant_zp_at(qo, 0) : 0;
    const bool relu = gm.epilogue.type == EpilogueActivateType::Relu;

    auto* out_ptr = output.ptr<TOut>();

    MatMulAttributes mm{};
    mm.transpose_a = true;

    for (int64_t n = 0; n < gm.N; ++n) {
        const int8_t* in_n = act.ptr + n * act.n_stride;
        TOut* out_n = out_ptr + n * out_n_stride;

        for (int64_t g = 0; g < gm.G; ++g) {
            const int8_t* w_g = w_ptr + g * icg * (ocg * karea);

            // B: the group's whole input plane, [IC/G] × [IH·IW].
            const int64_t bshape[] = {icg, npix};
            TensorView bm(std::span<const int64_t>(bshape, 2), DataType::s8,
                          const_cast<int8_t*>(in_n + g * icg * npix),
                          static_cast<int64_t>(npix), TensorLayout::NCHW, act_qp);

            for (int64_t oc0 = 0; oc0 < ocg; oc0 += chunk) {
                const int64_t cc = std::min(chunk, ocg - oc0);
                const int64_t m  = cc * karea;

                // A channel's karea taps are consecutive GEMM rows, so its
                // (scale, zero point) fills karea entries.
                for (int64_t o = 0; o < cc; ++o) {
                    const int64_t oc = g * ocg + oc0 + o;
                    const float s = quant_scale_at(qw, oc);
                    const int32_t z = quant_zp_at(qw, oc);
                    for (int64_t t = 0; t < karea; ++t) {
                        w_scale.data()[o * karea + t] = s;
                        w_zp.data()[o * karea + t] = z;
                    }
                }

                const int64_t ashape[] = {icg, m};
                TensorView am(std::span<const int64_t>(ashape, 2), DataType::s8,
                              const_cast<int8_t*>(w_g + oc0 * karea),
                              static_cast<int64_t>(ocg * karea), TensorLayout::NCHW,
                              qp_per_channel(w_scale.data(), w_zp.data(), m));
                const int64_t cshape[] = {m, npix};
                TensorView cm(std::span<const int64_t>(cshape, 2), DataType::s32,
                              col.data(),
                              static_cast<int64_t>(npix * sizeof(int32_t)),
                              TensorLayout::NCHW);

                const TensorView mins[] = {am, bm};
                matmul_cpu(mm, cm, mins, ctx, nullptr);

                ctx.cpu.run(0, cc * gm.OH, [&](int64_t task) {
                    const int64_t oc_l = task / gm.OH;
                    const int64_t oh   = task % gm.OH;
                    int32_t* row = rows.data() +
                        static_cast<int64_t>(ctx.cpu.current_thread_id()) * row_len;
                    std::fill_n(row, row_len, int32_t{0});

                    // Which kernel rows land on this output row.
                    int64_t kh_of[kMaxRowTaps];
                    int64_t ih_of[kMaxRowTaps];
                    int64_t taps = 0;
                    for (int64_t kh = 0; kh < gm.KH; ++kh) {
                        const int64_t nh = oh + gm.PH - kh * gm.DH;
                        if (nh < 0 || nh % gm.SH != 0) { continue; }
                        const int64_t ih = nh / gm.SH;
                        if (ih >= gm.IH) { continue; }
                        kh_of[taps] = kh;
                        ih_of[taps] = ih;
                        ++taps;
                    }

                    for (int64_t ti = 0; ti < taps; ++ti) {
                        const int32_t* base = col.data() +
                            (oc_l * karea + kh_of[ti] * gm.KW) * npix +
                            ih_of[ti] * gm.IW;
                        for (int64_t kw = 0; kw < gm.KW; ++kw) {
                            const int64_t d = gm.PW - kw * gm.DW;
                            for (int64_t r = 0; r < gm.SW; ++r) {
                                if (r >= gm.OW) { break; }
                                const int64_t nd = r + d;
                                if (nd % gm.SW != 0) { continue; }
                                const int64_t shift = nd / gm.SW;
                                const int64_t nr = (gm.OW - r + gm.SW - 1) / gm.SW;
                                const int64_t tlo = std::max<int64_t>(0, -shift);
                                const int64_t thi = std::min<int64_t>(nr, gm.IW - shift);
                                if (thi <= tlo) { continue; }
                                row_add_i32(row + r * mic + tlo,
                                            base + kw * npix + shift + tlo,
                                            thi - tlo);
                            }
                        }
                    }

                    const int64_t oc = g * ocg + oc0 + oc_l;
                    const float s_w = quant_scale_at(qw, oc);
                    const float req = kIntOut
                        ? matmul_int8_requant_scale(s_w, s_x, out_scale) : 0.0f;
                    TOut* orow = out_n + oc * out_c_stride + oh * out_row_stride;
                    for (int64_t r = 0; r < gm.SW; ++r) {
                        if (r >= gm.OW) { break; }
                        const int64_t nr = (gm.OW - r + gm.SW - 1) / gm.SW;
                        const int32_t* rp = row + r * mic;
                        for (int64_t t = 0; t < nr; ++t) {
                            const int64_t ow = r + t * gm.SW;
                            if constexpr (kIntOut) {
                                int32_t v = rp[t];
                                if (b_i32) { v += b_i32[oc]; }
                                if (relu) { v = std::max(v, int32_t{0}); }
                                orow[ow] = matmul_int8_requant<TOut>(v, req, zp_out);
                            } else {
                                float f = static_cast<float>(rp[t]) * (s_w * s_x);
                                if (b_fp) { f += s_load(&b_fp[oc]); }
                                f = std::min(std::max(f, gm.epilogue.min_clip),
                                             gm.epilogue.max_clip);
                                f = nnops::apply_epilogue(gm.epilogue, f, oc);
                                s_store(&orow[ow], f);
                            }
                        }
                    }
                });
            }
        }
    }
    return true;
}

/// Channel-last quantized route: one int8 GEMM per (group, oc chunk, batch)
/// gives [pixel] × [tap·oc], and the fp route's stride-residue scatter adds the
/// int32 columns before the merge. The filter is always permuted here, unlike
/// the fp route's in-place orientation: its columns carry the per-channel
/// parameters, which the raw [oc][tap] order does not line up with.
template <typename TOut>
bool tconv2d_int8_channel_last(const Geom& gm, TensorView& output,
                               std::span<const TensorView> inputs,
                               const ActPlane& act, const ComputeContext& ctx)
{
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;
    constexpr bool kIntOut =
        std::is_same_v<TOut, int8_t> || std::is_same_v<TOut, uint8_t>;

    if (weight.row_stride_elems() != gm.KW) { return false; }

    const int64_t karea = gm.KH * gm.KW;
    const int64_t npix  = gm.IH * gm.IW;
    const int64_t icg   = gm.IC_per_G;
    const int64_t ocg   = gm.OC_per_G;
    if (npix <= 0 || karea <= 0 || icg <= 0) { return false; }

    const int64_t out_n_stride   = output.stride_elems(0);
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t slots          = std::max(1, ctx.cpu.worker_count());

    // The col and the per-worker row scratch both scale with the oc chunk.
    const int64_t elem_cap = kColCapBytes / static_cast<int64_t>(sizeof(int32_t));
    int64_t cc = ocg;
    cc = std::min(cc, std::max<int64_t>(1, elem_cap / (npix * karea)));
    cc = std::min(cc, std::max<int64_t>(1, elem_cap / (slots * gm.OW)));

    PoolArray<int32_t> col(static_cast<size_t>(npix * karea * cc));
    PoolArray<int8_t> aux(static_cast<size_t>(icg * karea * cc));
    PoolArray<int32_t> rows(static_cast<size_t>(slots * gm.OW * cc));
    PoolArray<float> b_scale(static_cast<size_t>(karea * cc));
    PoolArray<int32_t> b_zp(static_cast<size_t>(karea * cc));
    if (!col || !aux || !rows || !b_scale || !b_zp) { return false; }

    const auto* w_ptr = weight.ptr<int8_t>();
    const QuantParams& qw = weight.quant_params();
    const auto* b_i32 = (has_bias && kIntOut) ? inputs[2].ptr<int32_t>() : nullptr;
    const TOut* b_fp = (has_bias && !kIntOut) ? inputs[2].ptr<TOut>() : nullptr;
    const float s_x = act.scale;
    const QuantParams act_qp = qp_per_tensor(s_x, act.zero);

    const QuantParams& qo = output.quant_params();
    const float out_scale = kIntOut ? quant_scale_at(qo, 0) : 0.0f;
    const int32_t zp_out = kIntOut ? quant_zp_at(qo, 0) : 0;
    const bool relu = gm.epilogue.type == EpilogueActivateType::Relu;

    auto* out_ptr = output.ptr<TOut>();

    MatMulAttributes mm{};

    for (int64_t g = 0; g < gm.G; ++g) {
        for (int64_t oc0 = 0; oc0 < ocg; oc0 += cc) {
            const int64_t nc = std::min(cc, ocg - oc0);
            if (nc <= 0) { continue; }
            const int64_t m = nc * karea;

            // Per-column (tap, oc) parameters: B's columns are the
            // (kh, kw, oc) triples of one output channel.
            for (int64_t o = 0; o < nc; ++o) {
                const int64_t oc = g * ocg + oc0 + o;
                const float s = quant_scale_at(qw, oc);
                const int32_t z = quant_zp_at(qw, oc);
                for (int64_t t = 0; t < karea; ++t) {
                    b_scale.data()[t * nc + o] = s;
                    b_zp.data()[t * nc + o] = z;
                }
            }

            // B: the group's filter slice as [IC/G] × [KH·KW·nc], a blocked
            // transpose of the channel-first [IC/G][OC/G][KH·KW] rows.
            for (int64_t ic = 0; ic < icg; ++ic) {
                const int8_t* src = w_ptr + ((g * icg + ic) * ocg + oc0) * karea;
                int8_t* dst = aux.data() + ic * m;
                constexpr int64_t kB = 8;
                for (int64_t o0 = 0; o0 < nc; o0 += kB) {
                    const int64_t o1 = std::min(o0 + kB, nc);
                    for (int64_t t0 = 0; t0 < karea; t0 += kB) {
                        const int64_t t1 = std::min(t0 + kB, karea);
                        for (int64_t o = o0; o < o1; ++o) {
                            int8_t* d = dst + t0 * nc + o;
                            const int8_t* s = src + o * karea + t0;
                            for (int64_t t = 0; t < t1 - t0; ++t) { d[t * nc] = s[t]; }
                        }
                    }
                }
            }

            const int64_t bshape[] = {icg, m};
            TensorView bm(std::span<const int64_t>(bshape, 2), DataType::s8,
                          aux.data(), static_cast<int64_t>(m), TensorLayout::NCHW,
                          qp_per_channel(b_scale.data(), b_zp.data(), m));

            for (int64_t n = 0; n < gm.N; ++n) {
                const int8_t* in_n = act.ptr + n * act.n_stride;
                TOut* out_n = out_ptr + n * out_n_stride;

                // A: the channel-last image as [IH·IW] × [IC/G] pixels — one
                // pixel stride of IC, so the group offset is g·IC/G elements in.
                const int64_t ashape[] = {npix, icg};
                TensorView am(std::span<const int64_t>(ashape, 2), DataType::s8,
                              const_cast<int8_t*>(in_n + g * icg),
                              static_cast<int64_t>(gm.IC), TensorLayout::NCHW,
                              act_qp);
                const int64_t cshape[] = {npix, m};
                TensorView cm(std::span<const int64_t>(cshape, 2), DataType::s32,
                              col.data(),
                              static_cast<int64_t>(m * sizeof(int32_t)),
                              TensorLayout::NCHW);

                const TensorView mins[] = {am, bm};
                matmul_cpu(mm, cm, mins, ctx, nullptr);

                ctx.cpu.run(0, gm.OH, [&](int64_t oh) {
                    int32_t* row = rows.data() +
                        static_cast<int64_t>(ctx.cpu.current_thread_id()) *
                            (gm.OW * nc);
                    std::fill_n(row, gm.OW * nc, int32_t{0});

                    for (int64_t kh = 0; kh < gm.KH; ++kh) {
                        const int64_t num_h = oh + gm.PH - kh * gm.DH;
                        if (num_h < 0 || num_h % gm.SH != 0) { continue; }
                        const int64_t ih = num_h / gm.SH;
                        if (ih >= gm.IH) { continue; }
                        const int32_t* ihrow = col.data() + ih * gm.IW * m;

                        for (int64_t kw = 0; kw < gm.KW; ++kw) {
                            const int32_t* colw = ihrow + (kh * gm.KW + kw) * nc;
                            for (int64_t r = 0; r < gm.SW && r < gm.OW; ++r) {
                                const int64_t num_w = r + gm.PW - kw * gm.DW;
                                if (num_w % gm.SW != 0) { continue; }
                                const int64_t sh  = num_w / gm.SW;
                                const int64_t mic = (gm.OW - r + gm.SW - 1) / gm.SW;
                                const int64_t tlo = std::max<int64_t>(0, -sh);
                                const int64_t thi = std::min<int64_t>(mic, gm.IW - sh);
                                for (int64_t t = tlo; t < thi; ++t) {
                                    row_add_i32(row + (r + t * gm.SW) * nc,
                                                colw + (t + sh) * m, nc);
                                }
                            }
                        }
                    }

                    TOut* orow = out_n + oh * out_row_stride + g * ocg + oc0;
                    for (int64_t ow = 0; ow < gm.OW; ++ow) {
                        const int32_t* rp = row + ow * nc;
                        TOut* dp = orow + ow * gm.OC;
                        for (int64_t c = 0; c < nc; ++c) {
                            const int64_t oc = g * ocg + oc0 + c;
                            const float s_w = quant_scale_at(qw, oc);
                            if constexpr (kIntOut) {
                                int32_t v = rp[c];
                                if (b_i32) { v += b_i32[oc]; }
                                if (relu) { v = std::max(v, int32_t{0}); }
                                dp[c] = matmul_int8_requant<TOut>(
                                    v, matmul_int8_requant_scale(s_w, s_x, out_scale),
                                    zp_out);
                            } else {
                                float f = static_cast<float>(rp[c]) * (s_w * s_x);
                                if (b_fp) { f += s_load(&b_fp[oc]); }
                                f = std::min(std::max(f, gm.epilogue.min_clip),
                                             gm.epilogue.max_clip);
                                f = nnops::apply_epilogue(gm.epilogue, f, oc);
                                s_store(&dp[c], f);
                            }
                        }
                    }
                });
            }
        }
    }
    return true;
}

/// Scalar quantized gather — the total route. Each term is
/// (w - zp_w)·(x - zp_x) in int32, the reference's own arithmetic; used when the
/// operands cannot be read as flat matrices or the kernel is too tall for the
/// per-row tap list.
template <typename TOut>
void tconv2d_int8_scalar(const Geom& gm, TensorView& output,
                         std::span<const TensorView> inputs, const ActPlane& act,
                         bool channel_last, const ComputeContext& ctx)
{
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;
    constexpr bool kIntOut =
        std::is_same_v<TOut, int8_t> || std::is_same_v<TOut, uint8_t>;

    const int64_t npix = gm.IH * gm.IW;
    const auto* w_ptr = weight.ptr<int8_t>();
    const QuantParams& qw = weight.quant_params();
    const auto* b_i32 = (has_bias && kIntOut) ? inputs[2].ptr<int32_t>() : nullptr;
    const TOut* b_fp = (has_bias && !kIntOut) ? inputs[2].ptr<TOut>() : nullptr;
    const float s_x = act.scale;
    const QuantParams& qo = output.quant_params();
    const float out_scale = kIntOut ? quant_scale_at(qo, 0) : 0.0f;
    const int32_t zp_out = kIntOut ? quant_zp_at(qo, 0) : 0;
    const bool relu = gm.epilogue.type == EpilogueActivateType::Relu;

    const int64_t in_c_stride  = channel_last ? 1 : npix;
    const int64_t in_row_stride = channel_last ? gm.IC : gm.IW;
    const int64_t out_c_stride = output.stride_elems(1);
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_n_stride = output.stride_elems(0);
    auto* out_ptr = output.ptr<TOut>();

    const auto process = [&](int64_t task) {
        const int64_t n  = task / gm.OC;
        const int64_t oc = task % gm.OC;
        const int64_t g = oc / gm.OC_per_G;
        const int64_t oc_l = oc % gm.OC_per_G;
        const float s_w = quant_scale_at(qw, oc);
        const int32_t zp_w = quant_zp_at(qw, oc);
        const int8_t* in_n = act.ptr + n * act.n_stride;
        TOut* out_plane = out_ptr + n * out_n_stride + oc * out_c_stride;

        for (int64_t oh = 0; oh < gm.OH; ++oh) {
            TOut* orow = out_plane + oh * out_row_stride;
            for (int64_t ow = 0; ow < gm.OW; ++ow) {
                int32_t acc = 0;
                for (int64_t kh = 0; kh < gm.KH; ++kh) {
                    const int64_t num_h = oh + gm.PH - kh * gm.DH;
                    if (num_h < 0 || num_h % gm.SH != 0) { continue; }
                    const int64_t ih = num_h / gm.SH;
                    if (ih >= gm.IH) { continue; }
                    for (int64_t kw = 0; kw < gm.KW; ++kw) {
                        const int64_t num_w = ow + gm.PW - kw * gm.DW;
                        if (num_w < 0 || num_w % gm.SW != 0) { continue; }
                        const int64_t iw = num_w / gm.SW;
                        if (iw >= gm.IW) { continue; }
                        const int8_t* xp = in_n +
                            (channel_last ? (ih * gm.IW + iw) * gm.IC + g * gm.IC_per_G
                                          : (g * gm.IC_per_G) * npix + ih * in_row_stride + iw);
                        for (int64_t ic_l = 0; ic_l < gm.IC_per_G; ++ic_l) {
                            const int64_t ic = g * gm.IC_per_G + ic_l;
                            const int32_t xv = static_cast<int32_t>(xp[ic_l * in_c_stride]);
                            const int64_t wi =
                                ((ic * gm.OC_per_G + oc_l) * gm.KH + kh) * gm.KW + kw;
                            acc += (static_cast<int32_t>(w_ptr[wi]) - zp_w) * (xv - act.zero);
                        }
                    }
                }

                if constexpr (kIntOut) {
                    int32_t v = acc;
                    if (b_i32) { v += b_i32[oc]; }
                    if (relu) { v = std::max(v, int32_t{0}); }
                    orow[ow] = matmul_int8_requant<TOut>(
                        v, matmul_int8_requant_scale(s_w, s_x, out_scale), zp_out);
                } else {
                    float f = static_cast<float>(acc) * (s_w * s_x);
                    if (b_fp) { f += s_load(&b_fp[oc]); }
                    f = std::min(std::max(f, gm.epilogue.min_clip),
                                 gm.epilogue.max_clip);
                    f = nnops::apply_epilogue(gm.epilogue, f, oc);
                    s_store(&orow[ow], f);
                }
            }
        }
    };

    ctx.cpu.run(0, gm.N * gm.OC, process);
}

template <typename TOut>
void tconv2d_int8_layout(const Geom& gm, TensorView& output,
                         std::span<const TensorView> inputs, const ActPlane& act,
                         const ComputeContext& ctx)
{
    if (is_channel_last(inputs[0].layout())) {
        if (tconv2d_int8_channel_last<TOut>(gm, output, inputs, act, ctx)) { return; }
    } else if (tconv2d_int8_channel_first<TOut>(gm, output, inputs, act, ctx)) {
        return;
    }
    tconv2d_int8_scalar<TOut>(gm, output, inputs, act,
                              is_channel_last(inputs[0].layout()), ctx);
}

/// One activation storage type of the quantized route: build the plane, then
/// dispatch the output dtype.
template <class TAct>
Status tconv2d_int8_act(const TransposeConv2DAttributes& attrs, const Geom& gm,
                        TensorView& output, std::span<const TensorView> inputs,
                        const ComputeContext& ctx)
{
    constexpr bool kFpAct =
        !std::is_same_v<TAct, int8_t> && !std::is_same_v<TAct, uint8_t>;
    const auto& input = inputs[0];
    const DataType adt = input.data_type();
    const bool target_u8 = kFpAct ? (attrs.act_quant_dtype == DataType::u8)
                                  : (adt == DataType::u8);
    const QuantParams& qa = input.quant_params();
    const float s_x    = quant_scale_at(qa, 0);
    const int32_t zp_x = quant_zp_at(qa, 0);

    const bool channel_last = is_channel_last(input.layout());
    const int64_t npix = gm.IH * gm.IW;
    const bool dense = channel_last
        ? (input.row_stride_elems() == gm.IW * gm.IC)
        : (input.row_stride_elems() == gm.IW);
    // A pre-quantized s8 activation with dense rows needs no plane at all.
    const bool in_place = !kFpAct && adt == DataType::s8 && dense;

    ActPlane act;
    act.scale = s_x;
    act.zero = zp_x - (target_u8 ? 128 : 0);
    // An empty plane (IH or IW 0) leaves the pointer on the input: no pixel of
    // it is ever read, but the batch walk still forms in_n.
    act.ptr = reinterpret_cast<const int8_t*>(input.ptr<TAct>());

    PoolArray<int8_t> plane;
    if (in_place) {
        act.ptr = input.ptr<int8_t>();
        act.n_stride = input.stride_elems(0);
    } else if (npix > 0) {
        const int64_t total = gm.N * gm.IC * npix;
        plane.reset(static_cast<size_t>(total));
        if (!plane) { return Status::out_of_memory; }
        build_activation_plane<TAct>(input, plane.data(), s_x, act.zero,
                                     channel_last, gm.IC, gm.IH, gm.IW, ctx);
        act.ptr = plane.data();
        act.n_stride = gm.IC * npix;
    }

    switch (output.data_type()) {
    case DataType::f32:
        tconv2d_int8_layout<float>(gm, output, inputs, act, ctx);
        break;
    case DataType::f16:
        tconv2d_int8_layout<half>(gm, output, inputs, act, ctx);
        break;
    case DataType::s8:
        tconv2d_int8_layout<int8_t>(gm, output, inputs, act, ctx);
        break;
    default:
        tconv2d_int8_layout<uint8_t>(gm, output, inputs, act, ctx);
        break;
    }
    return Status::ok;
}

/// dtype gate of the quantized route, then the activation dispatch.
Status transpose_conv2d_int8_cpu(const TransposeConv2DAttributes& attrs,
                                 const Geom& gm, TensorView& output,
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
    // An fp activation keeps its own dtype on an fp destination; a pre-quantized
    // one is dequantized to whatever output_dtype asked for.
    if (!int_out && fp_act && odt != adt) { return Status::unsupported; }
    if (inputs.size() > 2) {
        const DataType bdt = inputs[2].data_type();
        if (bdt != (int_out ? DataType::s32 : odt)) { return Status::unsupported; }
    }
    // add_to would have to keep the pre-existing destination out of the
    // scatter; the integer accumulate has no fold for it.
    if (attrs.add_to) { return Status::unsupported; }

    if (gm.N == 0 || gm.OC == 0 || gm.OH == 0 || gm.OW == 0) { return Status::ok; }

    switch (adt) {
    case DataType::f32:
        return tconv2d_int8_act<float>(attrs, gm, output, inputs, ctx);
    case DataType::f16:
        return tconv2d_int8_act<half>(attrs, gm, output, inputs, ctx);
    case DataType::u8:
        return tconv2d_int8_act<uint8_t>(attrs, gm, output, inputs, ctx);
    default:
        return tconv2d_int8_act<int8_t>(attrs, gm, output, inputs, ctx);
    }
}

template <typename T>
void tconv2d_impl(const Geom& gm,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t N  = gm.N;
    const int64_t IH = gm.IH;
    const int64_t IW = gm.IW;

    const int64_t IC       = gm.IC;
    const int64_t OC_per_G = gm.OC_per_G;
    const int64_t KH = gm.KH;
    const int64_t KW = gm.KW;

    const int64_t G = gm.G;
    const int64_t OC = gm.OC;
    const int64_t IC_per_G = gm.IC_per_G;

    const int64_t OH = gm.OH;
    const int64_t OW = gm.OW;

    const int64_t SH = gm.SH;
    const int64_t SW = gm.SW;
    const int64_t DH = gm.DH;
    const int64_t DW = gm.DW;
    const int64_t PH = gm.PH;
    const int64_t PW = gm.PW;

    if (N == 0 || OC == 0 || OH == 0 || OW == 0) { return; }

    auto* out_ptr = output.ptr<T>();
    const auto* in_ptr = input.ptr<T>();
    const auto* w_ptr  = weight.ptr<T>();
    const auto* b_ptr  = has_bias ? inputs[2].ptr<T>() : nullptr;

    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t in_n_stride    = input.stride_elems(0);
    const int64_t out_n_stride   = output.stride_elems(0);

    const auto epilogue = gm.epilogue;
    const bool add_to = gm.add_to && !gm.pre_act;
    const T* type_tag = nullptr;

    if (!is_channel_last(input.layout())) {
        if (tconv2d_channel_first_gemm<T>(gm, output, inputs, ctx)) { return; }

        const int64_t in_c_stride  = input.stride_elems(1);
        const int64_t out_c_stride = output.stride_elems(1);

        const auto process = [&](int64_t task) {
            const int64_t n  = task / OC;
            const int64_t oc = task % OC;
            const int64_t g = oc / OC_per_G;
            const int64_t oc_l = oc % OC_per_G;

            const T* in_n = in_ptr + n * in_n_stride;
            T* out_plane = out_ptr + n * out_n_stride + oc * out_c_stride;

            for (int64_t oh = 0; oh < OH; ++oh) {
                T* orow = out_plane + oh * out_row_stride;

                for (int64_t ow = 0; ow < OW; ++ow) {
                    float acc = 0.0f;

                    for (int64_t kh = 0; kh < KH; ++kh) {
                        const int64_t num_h = oh + PH - kh * DH;
                        if (num_h < 0 || num_h % SH != 0) { continue; }
                        const int64_t ih = num_h / SH;
                        if (ih >= IH) { continue; }

                        for (int64_t kw = 0; kw < KW; ++kw) {
                            const int64_t num_w = ow + PW - kw * DW;
                            if (num_w < 0 || num_w % SW != 0) { continue; }
                            const int64_t iw = num_w / SW;
                            if (iw >= IW) { continue; }

                            for (int64_t ic_l = 0; ic_l < IC_per_G; ++ic_l) {
                                const int64_t ic = g * IC_per_G + ic_l;
                                const T* xp = in_n + ic * in_c_stride
                                            + ih * in_row_stride + iw;
                                const int64_t wi =
                                    ((ic * OC_per_G + oc_l) * KH + kh) * KW + kw;
                                acc += s_load(xp) * s_load(&w_ptr[wi]);
                            }
                        }
                    }

                    if (b_ptr) { acc += s_load(&b_ptr[oc]); }
                    if (gm.pre_act) { acc += s_load(&orow[ow]); }
                    acc = nnops::apply_epilogue(epilogue, acc, oc);
                    if (add_to) { acc += s_load(&orow[ow]); }
                    s_store(&orow[ow], acc);
                }
            }
        };

        ctx.cpu.run(0, N * OC, process);
        return;
    }

    // ---- channel-last ----
    constexpr int L = simd_lane_for<T>;
    constexpr int64_t kPix = 4;
    const int64_t OC_span = OC_per_G;

    // Route. The pixel kernel's one trick is reusing a weight vector across
    // four pixels, which pays only when the filter stays resident between
    // tiles and few taps land on each output pixel; past that it re-reads the
    // filter from DRAM. Both other cases go through a col matrix, which walks
    // the filter once per (n, group) whatever the geometry. Depthwise fits
    // neither — its channels, not its pixels, are the vector axis.
    if (tconv2d_channel_last_depthwise<T>(gm, output, inputs, ctx)) { return; }

    // In f16 every vector FMA is a convert pair, so the pixel kernel's
    // broadcast-and-reuse does not pay for itself the way it does in f32; the
    // GEMM kernels convert whole panels and win on every shape.
    const int64_t wbytes = gm.IC * OC_span * gm.KH * gm.KW *
                           static_cast<int64_t>(sizeof(T));
    const bool pixel_form = std::is_same_v<T, float> &&
                            gm.KH * gm.KW <= 3 * gm.SH * gm.SW &&
                            wbytes <= (512 << 10);
    if (!pixel_form && tconv2d_channel_last_gemm<T>(gm, output, inputs, ctx)) {
        return;
    }

    // Weight repack: [G][KH][KW][IC/G][OC/G].
    const int64_t wt_g_stride = KH * KW * IC_per_G * OC_span;
    PoolArray<T> wt(static_cast<size_t>(G * wt_g_stride));

    // Per-worker tap lists: the output row's (kh, ih) pairs, then one output
    // column residue's (kw, iw shift) pairs. Both are per-*class* facts, so the
    // divisibility tests leave the pixel loop entirely.
    const int64_t slots   = std::max(1, ctx.cpu.worker_count());
    const int64_t tap_cap = KH + SW * KW;
    PoolArray<int64_t> tap_keys(static_cast<size_t>(slots * tap_cap));
    PoolArray<int64_t> tap_vals(static_cast<size_t>(slots * tap_cap));
    if (!wt || !tap_keys || !tap_vals) { return; }

    // Group-relative index into one [KH][KW][IC/G][OC/G] block.
    const auto wt_index = [&](int64_t kh, int64_t kw, int64_t ic_l, int64_t oc_l) {
        return ((kh * KW + kw) * IC_per_G + ic_l) * OC_span + oc_l;
    };

    for (int64_t ic = 0; ic < IC; ++ic) {
        const int64_t g    = ic / IC_per_G;
        const int64_t ic_l = ic % IC_per_G;
        for (int64_t oc_l = 0; oc_l < OC_span; ++oc_l) {
            const int64_t w_base = (ic * OC_span + oc_l) * KH * KW;
            for (int64_t kh = 0; kh < KH; ++kh) {
                for (int64_t kw = 0; kw < KW; ++kw) {
                    wt[static_cast<size_t>(g * wt_g_stride +
                                           wt_index(kh, kw, ic_l, oc_l))] =
                        w_ptr[w_base + kh * KW + kw];
                }
            }
        }
    }

    const auto process = [&](int64_t task) {
        const int64_t n  = task / OH;
        const int64_t oh = task % OH;
        const int64_t slot = static_cast<int64_t>(ctx.cpu.current_thread_id());

        int64_t* kh_of = tap_keys.data() + slot * tap_cap;
        int64_t* kw_of = kh_of + KH;
        int64_t* ih_of = tap_vals.data() + slot * tap_cap;
        int64_t* sh_of = ih_of + KH;

        const T* in_n = in_ptr + n * in_n_stride;
        T* orow_base = out_ptr + n * out_n_stride + oh * out_row_stride;

        int64_t nt = 0;
        for (int64_t kh = 0; kh < KH; ++kh) {
            const int64_t num_h = oh + PH - kh * DH;
            if (num_h < 0 || num_h % SH != 0) { continue; }
            const int64_t ih = num_h / SH;
            if (ih >= IH) { continue; }
            kh_of[nt] = kh;
            ih_of[nt] = ih;
            ++nt;
        }

        for (int64_t g = 0; g < G; ++g) {
            const T* wg = wt.data() + g * wt_g_stride;
            const T* bg = b_ptr ? b_ptr + g * OC_span : nullptr;
            const T* in_g = in_n + g * IC_per_G;

            // An output column belongs to residue class ow % SW, and inside a
            // class the taps are the same for every column — only the validity
            // window moves, so it becomes one [t_lo, t_hi) interval.
            for (int64_t r = 0; r < SW && r < OW; ++r) {
                const int64_t mic_r = (OW - r + SW - 1) / SW;

                int64_t nk = 0;
                int64_t t_lo = 0;
                int64_t t_hi = mic_r;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    // Only divisibility is a class fact. A negative offset is
                    // not: pixel t of the class sees t·SW more, so the tap
                    // starts to apply at t = -sh, which is what t_lo is.
                    const int64_t num_w = r + PW - kw * DW;
                    if (num_w % SW != 0) { continue; }
                    const int64_t sh = num_w / SW;
                    kw_of[nk] = kw;
                    sh_of[nk] = sh;
                    ++nk;
                    t_lo = std::max<int64_t>(t_lo, -sh);
                    t_hi = std::min<int64_t>(t_hi, IW - sh);
                }

                // One output pixel, channels [c_begin, OC_span). The edge pixels
                // and the oc tail come through here, where nothing is reusable.
                const auto pixel = [&](int64_t t, int64_t c_begin) {
                    T* dst = orow_base + (r + t * SW) * OC + g * OC_span;
                    int64_t c0 = c_begin;
                    for (; c0 + L <= OC_span; c0 += L) {
                        vec_for<T> acc = v_zero(type_tag);

                        for (int64_t a = 0; a < nt; ++a) {
                            const T* row = in_g + ih_of[a] * in_row_stride;
                            for (int64_t k = 0; k < nk; ++k) {
                                const int64_t iw = t + sh_of[k];
                                if (iw < 0 || iw >= IW) { continue; }
                                const T* pix = row + iw * IC;
                                const T* wrow = wg + ((kh_of[a] * KW + kw_of[k])
                                                    * IC_per_G * OC_span + c0);
                                for (int64_t ic = 0; ic < IC_per_G; ++ic) {
                                    acc = v_fmadd(v_set1(type_tag, s_load(&pix[ic])),
                                                  v_load(wrow + ic * OC_span), acc);
                                }
                            }
                        }

                        acc = v_add(acc, bg ? v_load(bg + c0) : v_zero(type_tag));
                        if (gm.pre_act) { acc = v_add(acc, v_load(dst + c0)); }
                        acc = apply_epilogue_vec<T>(epilogue, type_tag, acc);
                        if (add_to) { acc = v_add(acc, v_load(dst + c0)); }
                        v_store(dst + c0, acc);
                    }

                    for (; c0 < OC_span; ++c0) {
                        float acc = 0.0f;

                        for (int64_t a = 0; a < nt; ++a) {
                            const T* row = in_g + ih_of[a] * in_row_stride;
                            for (int64_t k = 0; k < nk; ++k) {
                                const int64_t iw = t + sh_of[k];
                                if (iw < 0 || iw >= IW) { continue; }
                                const T* pix = row + iw * IC;
                                for (int64_t ic = 0; ic < IC_per_G; ++ic) {
                                    acc += s_load(&pix[ic]) *
                                           s_load(&wg[wt_index(kh_of[a], kw_of[k],
                                                               ic, c0)]);
                                }
                            }
                        }

                        if (bg) { acc += s_load(&bg[c0]); }
                        if (gm.pre_act) { acc += s_load(&dst[c0]); }
                        acc = nnops::apply_epilogue(epilogue, acc,
                                                    g * OC_span + c0);
                        if (add_to) { acc += s_load(&dst[c0]); }
                        s_store(&dst[c0], acc);
                    }
                };

                // Four pixels of one class against two oc vectors: the weight
                // vector is then loaded once per four pixels and the broadcast
                // once per pixel, where the pixel loop pays both per pixel.
                const auto tile = [&](int64_t t0) {
                    T* d0 = orow_base + (r + t0 * SW) * OC + g * OC_span;
                    int64_t c0 = 0;
                    for (; c0 + 2 * L <= OC_span; c0 += 2 * L) {
                        vec_for<T> a0[kPix];
                        vec_for<T> a1[kPix];
                        for (int64_t j = 0; j < kPix; ++j) {
                            a0[j] = v_zero(type_tag);
                            a1[j] = v_zero(type_tag);
                        }

                        for (int64_t a = 0; a < nt; ++a) {
                            const T* row = in_g + ih_of[a] * in_row_stride;
                            for (int64_t k = 0; k < nk; ++k) {
                                const T* pix = row + (t0 + sh_of[k]) * IC;
                                const T* wrow = wg + ((kh_of[a] * KW + kw_of[k])
                                                    * IC_per_G * OC_span + c0);
                                for (int64_t ic = 0; ic < IC_per_G; ++ic) {
                                    const auto w0 = v_load(wrow + ic * OC_span);
                                    const auto w1 = v_load(wrow + ic * OC_span + L);
                                    for (int64_t j = 0; j < kPix; ++j) {
                                        const auto x = v_set1(
                                            type_tag, s_load(pix + j * IC + ic));
                                        a0[j] = v_fmadd(x, w0, a0[j]);
                                        a1[j] = v_fmadd(x, w1, a1[j]);
                                    }
                                }
                            }
                        }

                        const auto b0 = bg ? v_load(bg + c0) : v_zero(type_tag);
                        const auto b1 = bg ? v_load(bg + c0 + L) : v_zero(type_tag);
                        for (int64_t j = 0; j < kPix; ++j) {
                            T* dst = d0 + j * SW * OC;
                            vec_for<T> s0 = v_add(a0[j], b0);
                            vec_for<T> s1 = v_add(a1[j], b1);
                            if (gm.pre_act) {
                                s0 = v_add(s0, v_load(dst + c0));
                                s1 = v_add(s1, v_load(dst + c0 + L));
                            }
                            vec_for<T> v0 = apply_epilogue_vec<T>(
                                epilogue, type_tag, s0);
                            vec_for<T> v1 = apply_epilogue_vec<T>(
                                epilogue, type_tag, s1);
                            if (add_to) {
                                v0 = v_add(v0, v_load(dst + c0));
                                v1 = v_add(v1, v_load(dst + c0 + L));
                            }
                            v_store(dst + c0, v0);
                            v_store(dst + c0 + L, v1);
                        }
                    }
                    if (c0 < OC_span) {
                        for (int64_t j = 0; j < kPix; ++j) { pixel(t0 + j, c0); }
                    }
                };

                // The edge pixels first, then tiles aligned to the valid window
                // — not to zero. Aligning to zero would throw the whole tile
                // away whenever the window does not start on a multiple of
                // kPix, which on a short row is the difference between reusing
                // the weights and not.
                int64_t t0 = 0;
                for (; t0 < t_lo && t0 < mic_r; ++t0) { pixel(t0, 0); }
                for (; t0 + kPix <= t_hi; t0 += kPix) { tile(t0); }
                for (; t0 < mic_r; ++t0) { pixel(t0, 0); }
            }
        }
    };

    ctx.cpu.run(0, N * OH, process);
}

}  // anonymous namespace

Status transpose_conv2d_cpu(const TransposeConv2DAttributes& attrs,
                            TensorView& output,
                            std::span<const TensorView> inputs,
                            const ComputeContext& ctx,
                            void* /*workspace*/)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];

    Geom gm{};
    gm.N  = input.shape(0);
    gm.IH = input.shape(input.rank() - 2);
    gm.IW = input.shape(input.rank() - 1);
    gm.IC = weight.shape(0);
    gm.OC_per_G = weight.shape(1);
    gm.KH = attrs.kernel_size[0];
    gm.KW = attrs.kernel_size[1];
    gm.G  = attrs.groups;
    gm.OC = gm.OC_per_G * gm.G;
    gm.IC_per_G = gm.IC / gm.G;
    gm.OH = output.shape(2);
    gm.OW = output.shape(3);
    gm.SH = attrs.stride[0];
    gm.SW = attrs.stride[1];
    gm.DH = attrs.dilation[0];
    gm.DW = attrs.dilation[1];
    gm.PH = attrs.padding[0];
    gm.PW = attrs.padding[1];
    gm.epilogue = attrs.epilogue;
    gm.add_to = attrs.add_to;
    gm.pre_act = attrs.add_to && attrs.residual_pre_activation;

    if (is_quantized_transpose_conv2d(weight.data_type())) {
        return transpose_conv2d_int8_cpu(attrs, gm, output, inputs, ctx);
    }

    dispatch_f32_f16(input.data_type(), "transpose_conv2d_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        tconv2d_impl<T>(gm, output, inputs, ctx);
    });
    return Status::ok;
}

}  // namespace nnops::backend::cpu
