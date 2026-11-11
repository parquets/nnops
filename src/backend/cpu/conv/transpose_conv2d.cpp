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
#include "backend/cpu/conv/tconv_row.h"
#include "backend/cpu/conv/transpose_conv2d_col2im.h"
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

    /// The col2im module's view of the same numbers.
    Tconv2dColGeom col_geom() const
    {
        return make_col_geom(IH, IW, OW, KH, KW, SH, SW, DH, DW, PH, PW);
    }
};

/// Largest col one GEMM call may materialise; past it the M rows are split.
constexpr int64_t kColCapBytes = 8 << 20;

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
    const Tconv2dColGeom cg = gm.col_geom();
    const int64_t row_len = cg.row_len;
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

                    tconv2d_col2im_channel_major<T>(
                        cg, col.data() + oc_l * karea * npix, row, oh,
                        [](T* dst, const T* src, int64_t len, int64_t) {
                            row_add(dst, src, len);
                        }, npix);

                    const int64_t oc = g * ocg + oc0 + oc_l;
                    T* orow = out_n + oc * out_c_stride + oh * out_row_stride;
                    for (int64_t r = 0; r < gm.SW; ++r) {
                        if (r >= gm.OW) { break; }
                        const int64_t nr = (gm.OW - r + gm.SW - 1) / gm.SW;
                        const T* rp = row + r * cg.mic;
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
    permute_taps(w_ptr, wd.data(), C, karea);

    const auto* in_ptr = input.ptr<T>();
    const auto* b_ptr  = has_bias ? inputs[2].ptr<T>() : nullptr;
    auto* out_ptr = output.ptr<T>();
    const int64_t in_n_stride    = input.stride_elems(0);
    const int64_t out_n_stride   = output.stride_elems(0);
    const int64_t out_row_stride = output.row_stride_elems();
    const Tconv2dColGeom cg      = gm.col_geom();

    ctx.cpu.run(0, gm.N * gm.OH, [&](int64_t task) {
        const int64_t n  = task / gm.OH;
        const int64_t oh = task % gm.OH;
        T* row = rows.data() +
                 static_cast<int64_t>(ctx.cpu.current_thread_id()) * row_len;
        const T* in_n = in_ptr + n * in_n_stride;

        row_fill(row, row_len, 0.0f);

        // The "col" is the input plane itself: one channel per group means the
        // tap indexes the repacked filter, not a source stride.
        tconv2d_col2im_pixel_major<T>(
            cg, in_n, gm.IC, C, 0, row, oh,
            [&](T* dst, const T* src, int64_t len, int64_t tap) {
                row_fma(dst, src, wd.data() + tap * C, len);
            });

        T* orow = out_ptr + n * out_n_stride + oh * out_row_stride;
        merge_oc_row(orow, row, b_ptr, 0, gm, C);
    });
    return true;
}

/// One channel-last col GEMM call, resolved to flat operands: the chunk walk
/// below varies only the group and the oc chunk.
template <typename T>
struct ClCall {
    const Geom& gm;
    Tconv2dColGeom cg;
    const T *in, *w, *b;
    T *out, *col, *aux, *rows;
    DataType vdt;
    int64_t icg, ocg, npix, karea;
    int64_t in_n_stride, out_n_stride, out_row_stride;
    bool direct_w;
};

/// One oc chunk: B for the chunk — the filter as [IC/G] × [KH·KW·nc], permuted,
/// or the channel-first rows read in place with oc still the faster axis — then
/// per batch item the GEMM, the direct-weight col transpose, and the scatter and
/// merge of each output row.
template <typename T>
void tconv2d_cl_chunk(const ClCall<T>& c, const ComputeContext& ctx,
                      int64_t g, int64_t oc0, int64_t nc)
{
    const int64_t m = nc * c.karea;

    if (!c.direct_w) {
        // A blocked transpose of the channel-first [IC/G][OC/G][KH·KW] rows.
        for (int64_t ic = 0; ic < c.icg; ++ic) {
            const T* src = c.w + ((g * c.icg + ic) * c.ocg + oc0) * c.karea;
            permute_taps(src, c.aux + ic * c.karea * nc, nc, c.karea);
        }
    }

    const int64_t bshape[] = {c.icg, m};
    TensorView bm(std::span<const int64_t>(bshape, 2), c.vdt,
                  c.direct_w
                      ? const_cast<T*>(c.w + (g * c.icg * c.ocg + oc0) * c.karea)
                      : c.aux,
                  (c.direct_w ? c.ocg * c.karea : m) * sizeof(T),
                  TensorLayout::NCHW);

    for (int64_t n = 0; n < c.gm.N; ++n) {
        const T* in_n = c.in + n * c.in_n_stride;
        T* out_n = c.out + n * c.out_n_stride;

        // A: the channel-last image as [IH·IW] × [IC/G] pixels — one pixel
        // stride of IC, so the group offset is g·IC/G elements in.
        const int64_t ashape[] = {c.npix, c.icg};
        TensorView am(std::span<const int64_t>(ashape, 2), c.vdt,
                      const_cast<T*>(in_n + g * c.icg),
                      static_cast<int64_t>(c.gm.IC * sizeof(T)),
                      TensorLayout::NCHW);
        const int64_t cshape[] = {c.npix, m};
        TensorView cm(std::span<const int64_t>(cshape, 2), c.vdt, c.col,
                      static_cast<int64_t>(m * sizeof(T)), TensorLayout::NCHW);

        const TensorView mins[] = {am, bm};
        const MatMulAttributes mm{};
        matmul_cpu(mm, cm, mins, ctx, nullptr);

        const T* cbase = c.col;
        if (c.direct_w) {
            // Per pixel, [oc][tap] → [tap][oc], so the scatter sees the same
            // col either way.
            for (int64_t p = 0; p < c.npix; ++p) {
                permute_taps(c.col + p * m, c.aux + p * m, nc, c.karea);
            }
            cbase = c.aux;
        }

        ctx.cpu.run(0, c.gm.OH, [&](int64_t oh) {
            T* row = c.rows + static_cast<int64_t>(ctx.cpu.current_thread_id()) *
                                   (c.gm.OW * nc);
            row_fill(row, c.gm.OW * nc, 0.0f);

            tconv2d_col2im_pixel_major<T>(
                c.cg, cbase, m, nc, nc, row, oh,
                [](T* dst, const T* src, int64_t len, int64_t) {
                    row_add(dst, src, len);
                });

            T* orow = out_n + oh * c.out_row_stride + g * c.ocg + oc0;
            merge_oc_row(orow, row, c.b, g * c.ocg + oc0, c.gm, nc);
        });
    }
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
    const Tconv2dColGeom cg      = gm.col_geom();

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

    const ClCall<T> call{gm,         cg,       in_ptr,   w_ptr,
                         b_ptr,      out_ptr,  col.data(), aux.data(),
                         rows.data(), vdt,     icg,      ocg,
                         npix,       karea,    in_n_stride, out_n_stride,
                         out_row_stride, direct_w};

    for (int64_t g = 0; g < gm.G; ++g) {
        for (int64_t oc0 = 0; oc0 < ocg; oc0 += cc) {
            const int64_t nc = std::min(cc, ocg - oc0);
            if (nc > 0) { tconv2d_cl_chunk(call, ctx, g, oc0, nc); }
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

/// The int32 col row, whether the GEMM wrote it or the depthwise FMA pass did.
/// The depthwise sum is exact in f32 (|q − zp| ≤ 255 on both operands), so the
/// two forms are the same integer and the passes are shared.
inline v_f32x8 v_acc_to_f32(const int32_t* p)
{
#if defined(NNOPS_ARCH_X86_64)
    return v_f32x8(_mm256_cvtepi32_ps(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))));
#elif defined(NNOPS_ARCH_AARCH64)
    return v_f32x8(vcvtq_f32_s32(vld1q_s32(p)),
                   vcvtq_f32_s32(vld1q_s32(p + 4)));
#endif
}
inline v_f32x8 v_acc_to_f32(const float* p) { return v_load(p); }

inline float acc_at(int32_t v) { return static_cast<float>(v); }
inline float acc_at(float v) { return v; }

#if defined(NNOPS_ARCH_X86_64)
/// Rounded int32 lanes: a float row is exact here, so the convert is a no-op.
template <class Src>
inline __m256i v_acc_to_i32(const Src* p)
{
    if constexpr (std::is_same_v<Src, float>) {
        return _mm256_cvtps_epi32(_mm256_loadu_ps(p));
    } else {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
}
inline int32_t acc_to_i32(int32_t v) { return v; }
inline int32_t acc_to_i32(float v) { return static_cast<int32_t>(v); }
#elif defined(NNOPS_ARCH_AARCH64)
template <class Src>
inline int32x4_t v_acc_to_i32(const Src* p)
{
    if constexpr (std::is_same_v<Src, float>) {
        return vcvtnq_s32_f32(vld1q_f32(p));
    } else {
        return vld1q_s32(p);
    }
}
inline int32_t acc_to_i32(int32_t v) { return v; }
inline int32_t acc_to_i32(float v) { return static_cast<int32_t>(v); }
#endif

/// One pixel's `nc` integer output channels: bias, Relu, then round(acc·req) +
/// zp_out, `req` carrying the per-lane requant scale. Eight lanes at a time,
/// matching matmul_int8_requant's clamp-round-then-bias order.
template <typename TOut, class Src>
void requant_oc_row_i8(TOut* dp, const Src* rp, int64_t nc,
                       const int32_t* bias, bool relu, const float* req,
                       int32_t zp_out)
{
    constexpr bool kU8 = std::is_same_v<TOut, uint8_t>;
    constexpr float kMin = kU8 ? 0.0f : -128.0f;
    constexpr float kMax = kU8 ? 255.0f : 127.0f;
    const float zpf = static_cast<float>(zp_out);
    int64_t c = 0;
#if defined(NNOPS_ARCH_X86_64)
    const __m256i vzp  = _mm256_set1_epi32(zp_out);
    const __m256 vmin  = _mm256_set1_ps(kMin - zpf);
    const __m256 vmax  = _mm256_set1_ps(kMax - zpf);
    const __m256 vsign = _mm256_set1_ps(-0.0f);
    const __m256 vhalf = _mm256_set1_ps(0.5f);
    for (; c + 8 <= nc; c += 8) {
        __m256i v = v_acc_to_i32(rp + c);
        if (bias) {
            v = _mm256_add_epi32(
                v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + c)));
        }
        if (relu) { v = _mm256_max_epi32(v, _mm256_setzero_si256()); }
        __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(v), _mm256_loadu_ps(req + c));
        f = _mm256_min_ps(_mm256_max_ps(f, vmin), vmax);
        f = _mm256_add_ps(f, _mm256_or_ps(_mm256_and_ps(f, vsign), vhalf));
        const __m256i q = _mm256_add_epi32(_mm256_cvttps_epi32(f), vzp);
        const __m128i i16 = _mm_packs_epi32(_mm256_castsi256_si128(q),
                                            _mm256_extracti128_si256(q, 1));
        const __m128i v8 = kU8 ? _mm_packus_epi16(i16, i16)
                               : _mm_packs_epi16(i16, i16);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dp + c), v8);
    }
#elif defined(NNOPS_ARCH_AARCH64)
    const int32x4_t vzp = vdupq_n_s32(zp_out);
    const float32x4_t vmin = vdupq_n_f32(kMin - zpf);
    const float32x4_t vmax = vdupq_n_f32(kMax - zpf);
    for (; c + 4 <= nc; c += 4) {
        int32x4_t v = v_acc_to_i32(rp + c);
        if (bias) { v = vaddq_s32(v, vld1q_s32(bias + c)); }
        if (relu) { v = vmaxq_s32(v, vdupq_n_s32(0)); }
        float32x4_t f = vmulq_f32(vcvtq_f32_s32(v), vld1q_f32(req + c));
        f = vminq_f32(vmaxq_f32(f, vmin), vmax);
        const int32x4_t q = vaddq_s32(vcvtaq_s32_f32(f), vzp);
        const int16x4_t i16 = vqmovn_s32(q);
        if constexpr (kU8) {
            const uint8x8_t v8 = vqmovun_s16(vcombine_s16(i16, i16));
            vst1_lane_u32(reinterpret_cast<uint32_t*>(dp + c),
                          vreinterpret_u32_u8(v8), 0);
        } else {
            const int8x8_t v8 = vqmovn_s16(vcombine_s16(i16, i16));
            vst1_lane_s32(reinterpret_cast<int32_t*>(dp + c),
                          vreinterpret_s32_s8(v8), 0);
        }
    }
#endif
    for (; c < nc; ++c) {
        int32_t v = acc_to_i32(rp[c]);
        if (bias) { v += bias[c]; }
        if (relu) { v = std::max(v, 0); }
        dp[c] = matmul_int8_requant<TOut>(v, req[c], zp_out);
    }
}

/// The same row, element at a time: for an f16 destination the wrapper has no
/// f32→f16 row store, and the quant stage indexes its table by channel, which
/// the vector passes do not carry.
template <typename TOut, class Src>
void dequant_oc_row_scalar(TOut* orow, const Src* row, int64_t OW, int64_t nc,
                           int64_t OC, const TOut* bo, const QuantParams& qw,
                           int64_t oc_base, float s_x, const Epilogue& ep)
{
    for (int64_t ow = 0; ow < OW; ++ow) {
        const Src* rp = row + ow * nc;
        TOut* dp = orow + ow * OC;
        for (int64_t c = 0; c < nc; ++c) {
            const int64_t oc = oc_base + c;
            float f = acc_at(rp[c]) * (quant_scale_at(qw, oc) * s_x);
            if (bo) { f += s_load(&bo[c]); }
            f = nnops::apply_epilogue(ep, f, oc);
            s_store(&dp[c], f);
        }
    }
}

/// Vector-exact epilogues: the ones that decompose into min/max alone, so the
/// vector pass matches the scalar expression bit for bit. A transcendental
/// activation (Sigmoid, Tanh, Gelu, Silu, Elu, HardSwish) is a different function
/// in its vector approximation, and LeakyRelu's blend turns a −0 into a +0, where
/// the scalar `x > 0 ? x : alpha·x` keeps the sign. The integer routes are
/// compared byte for byte against the scalar reference, so those keep the element
/// loop.
inline bool epilogue_is_vector_exact(const Epilogue& ep) noexcept
{
    switch (ep.type) {
    case EpilogueActivateType::None:
    case EpilogueActivateType::Relu:
    case EpilogueActivateType::Relu6:
        return ep.quant == EpilogueQuantType::None;
    default:
        return false;
    }
}

/// The fp form: dequantize with the per-lane s_w·s_x, add the bias, then the
/// clip and the activation the fp route applies.
template <class Src>
void dequant_oc_row_f32(float* dp, const Src* rp, int64_t nc,
                        const float* w_sx, const float* bias,
                        const Epilogue& ep)
{
    constexpr int L = simd_default_lane_f32;
    const float* tag = nullptr;
    int64_t c = 0;
    for (; c + L <= nc; c += L) {
        vec_for<float> acc = v_mul(v_acc_to_f32(rp + c), v_load(w_sx + c));
        if (bias) { acc = v_add(acc, v_load(bias + c)); }
        acc = apply_epilogue_vec<float>(ep, tag, acc);
        v_store(dp + c, acc);
    }
    for (; c < nc; ++c) {
        float f = acc_at(rp[c]) * w_sx[c];
        if (bias) { f += bias[c]; }
        dp[c] = nnops::apply_epilogue(ep, f);
    }
}

/// Depthwise quantized, channel-last. One channel per group, so group g is
/// channel g and the vector axis is the channel — the fp depthwise route's shape,
/// with the col being the input plane itself. The GEMM route computes
/// Σ (q_x − zp_x)·(q_w − zp_w) in int32; here the same sum is accumulated in f32
/// over operands whose magnitude is at most 255, so every product and partial sum
/// is an integer below 2^24 and the conversion back is exact. The plane is
/// therefore held as (q_x − zp_x) and the encode stage is one row pass, no col
/// and no GEMM.
template <typename TOut>
bool tconv2d_int8_channel_last_depthwise(const Geom& gm, TensorView& output,
                                         std::span<const TensorView> inputs,
                                         const ActPlane& act,
                                         const ComputeContext& ctx)
{
    constexpr bool kIntOut =
        std::is_same_v<TOut, int8_t> || std::is_same_v<TOut, uint8_t>;

    if (gm.IC_per_G != 1 || gm.OC_per_G != 1) { return false; }

    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;
    if (weight.row_stride_elems() != gm.KW) { return false; }

    const int64_t karea = gm.KH * gm.KW;
    const int64_t C     = gm.IC;
    const int64_t npix  = gm.IH * gm.IW;
    const int64_t slots = std::max(1, ctx.cpu.worker_count());
    const int64_t row_len = gm.OW * C;
    const int64_t plane_elems = gm.N * npix * C;
    // 255·255·karea has to stay an exact integer in f32.
    if (karea > 256) { return false; }

    const int64_t elem_cap = kColCapBytes / static_cast<int64_t>(sizeof(float));
    if (C <= 0 || npix <= 0 || slots * row_len + plane_elems > elem_cap) {
        return false;
    }

    PoolArray<float> wd(static_cast<size_t>(karea * C));
    PoolArray<float> rows(static_cast<size_t>(slots * row_len));
    PoolArray<float> plane(static_cast<size_t>(plane_elems));
    PoolArray<float> lane_scale(static_cast<size_t>(C));
    if (!wd || !rows || !plane || !lane_scale) { return false; }

    const auto* w_ptr = weight.ptr<int8_t>();
    const QuantParams& qw = weight.quant_params();
    const auto* b_i32 = (has_bias && kIntOut) ? inputs[2].ptr<int32_t>() : nullptr;
    const TOut* b_fp = (has_bias && !kIntOut) ? inputs[2].ptr<TOut>() : nullptr;
    const float s_x = act.scale;
    const QuantParams& qo = output.quant_params();
    const float out_scale = kIntOut ? quant_scale_at(qo, 0) : 0.0f;
    const int32_t zp_out = kIntOut ? quant_zp_at(qo, 0) : 0;
    const bool relu = gm.epilogue.type == EpilogueActivateType::Relu;

    // [KH·KW][C] of (q_w − zp_w): the tap vector an output row is FMA'd with.
    for (int64_t c = 0; c < C; ++c) {
        const float s_w = quant_scale_at(qw, c);
        const float zw = static_cast<float>(quant_zp_at(qw, c));
        lane_scale.data()[c] = kIntOut
            ? matmul_int8_requant_scale(s_w, s_x, out_scale) : s_w * s_x;
        for (int64_t t = 0; t < karea; ++t) {
            wd.data()[t * C + c] = static_cast<float>(w_ptr[c * karea + t]) - zw;
        }
    }

    // The plane: (q_x − zp_x) channel-last, which is the layout the pixel-major
    // walk reads; one pass, shared by every tap of every output row.
    {
        const int8_t* src = act.ptr;
        const float zx = static_cast<float>(act.zero);
        const float* tag = nullptr;
        const vec_for<float> zvec = v_set1(tag, zx);
        const int64_t row_width = gm.IW * C;
        ctx.cpu.run(0, gm.N * gm.IH, [&](int64_t r) {
            const int64_t nb = r / gm.IH;
            const int8_t* sp = src + nb * act.n_stride + (r % gm.IH) * row_width;
            float* dp = plane.data() + r * row_width;
            int64_t i = 0;
            for (; i + simd_default_lane_f32 <= row_width;
                 i += simd_default_lane_f32) {
                v_store(dp + i, v_sub(v_cvt_s8_to_f32(sp + i), zvec));
            }
            for (; i < row_width; ++i) {
                dp[i] = static_cast<float>(sp[i]) - zx;
            }
        });
    }

    auto* out_ptr = output.ptr<TOut>();
    const int64_t out_n_stride   = output.stride_elems(0);
    const int64_t out_row_stride = output.row_stride_elems();
    const Tconv2dColGeom cg      = gm.col_geom();

    ctx.cpu.run(0, gm.N * gm.OH, [&](int64_t task) {
        const int64_t n  = task / gm.OH;
        const int64_t oh = task % gm.OH;
        float* row = rows.data() +
                     static_cast<int64_t>(ctx.cpu.current_thread_id()) * row_len;
        const float* in_n = plane.data() + n * act.n_stride;

        row_fill(row, row_len, 0.0f);

        tconv2d_col2im_pixel_major<float>(
            cg, in_n, C, C, 0, row, oh,
            [&](float* dst, const float* src, int64_t len, int64_t tap) {
                row_fma(dst, src, wd.data() + tap * C, len);
            });

        TOut* orow = out_ptr + n * out_n_stride + oh * out_row_stride;
        for (int64_t ow = 0; ow < gm.OW; ++ow) {
            TOut* dp = orow + ow * gm.OC;
            const float* rp = row + ow * C;
            if constexpr (kIntOut) {
                requant_oc_row_i8<TOut>(dp, rp, C, b_i32, relu,
                                        lane_scale.data(), zp_out);
            } else if constexpr (std::is_same_v<TOut, float>) {
                if (epilogue_is_vector_exact(gm.epilogue)) {
                    dequant_oc_row_f32(dp, rp, C, lane_scale.data(), b_fp,
                                       gm.epilogue);
                } else {
                    dequant_oc_row_scalar(dp, rp, 1, C, gm.OC, b_fp, qw,
                                          /*oc_base=*/0, s_x, gm.epilogue);
                }
            } else {
                dequant_oc_row_scalar(dp, rp, 1, C, gm.OC, b_fp, qw, /*oc_base=*/0,
                                      s_x, gm.epilogue);
            }
        }
    });
    return true;
}

/// Channel-first depthwise quantized route: the same exact-integer argument as
/// the channel-last one, but here the vector axis is the pixel, so a tap's
/// weight is a scalar over the run and the "col" is the group's own input plane
/// — one FMA per tap, no col, no GEMM. The row comes out in stride-residue
/// order, so the merge is per element, as in the channel-first GEMM route.
template <typename TOut>
bool tconv2d_int8_channel_first_depthwise(const Geom& gm, TensorView& output,
                                          std::span<const TensorView> inputs,
                                          const ActPlane& act,
                                          const ComputeContext& ctx)
{
    constexpr bool kIntOut =
        std::is_same_v<TOut, int8_t> || std::is_same_v<TOut, uint8_t>;

    if (gm.IC_per_G != 1 || gm.OC_per_G != 1) { return false; }

    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;
    if (weight.row_stride_elems() != gm.KW) { return false; }

    const int64_t karea = gm.KH * gm.KW;
    const int64_t C     = gm.IC;
    const int64_t npix  = gm.IH * gm.IW;
    const int64_t slots = std::max(1, ctx.cpu.worker_count());
    const Tconv2dColGeom cg = gm.col_geom();
    const int64_t row_len = cg.row_len;
    const int64_t plane_elems = gm.N * C * npix;
    // 255·255·karea has to stay an exact integer in f32.
    if (karea > 256) { return false; }

    const int64_t elem_cap = kColCapBytes / static_cast<int64_t>(sizeof(float));
    if (C <= 0 || npix <= 0 || slots * row_len + plane_elems > elem_cap) {
        return false;
    }

    PoolArray<float> wd(static_cast<size_t>(C * karea));
    PoolArray<float> rows(static_cast<size_t>(slots * row_len));
    PoolArray<float> plane(static_cast<size_t>(plane_elems));
    PoolArray<float> lane_scale(static_cast<size_t>(C));
    if (!wd || !rows || !plane || !lane_scale) { return false; }

    const auto* w_ptr = weight.ptr<int8_t>();
    const QuantParams& qw = weight.quant_params();
    const auto* b_i32 = (has_bias && kIntOut) ? inputs[2].ptr<int32_t>() : nullptr;
    const TOut* b_fp = (has_bias && !kIntOut) ? inputs[2].ptr<TOut>() : nullptr;
    const float s_x = act.scale;
    const QuantParams& qo = output.quant_params();
    const float out_scale = kIntOut ? quant_scale_at(qo, 0) : 0.0f;
    const int32_t zp_out = kIntOut ? quant_zp_at(qo, 0) : 0;
    const bool relu = gm.epilogue.type == EpilogueActivateType::Relu;

    // [c][karea] of (q_w − zp_w): one tap weight per output row this channel
    // touches, which the scalar FMA broadcasts.
    for (int64_t c = 0; c < C; ++c) {
        const float s_w = quant_scale_at(qw, c);
        const float zw = static_cast<float>(quant_zp_at(qw, c));
        lane_scale.data()[c] = kIntOut
            ? matmul_int8_requant_scale(s_w, s_x, out_scale) : s_w * s_x;
        for (int64_t t = 0; t < karea; ++t) {
            wd.data()[c * karea + t] = static_cast<float>(w_ptr[c * karea + t]) - zw;
        }
    }

    // The plane: (q_x − zp_x) channel-major, i.e. the input plane itself, one
    // pass shared by every tap of every output row.
    {
        const float zx = static_cast<float>(act.zero);
        const float* tag = nullptr;
        const vec_for<float> zvec = v_set1(tag, zx);
        ctx.cpu.run(0, gm.N * C, [&](int64_t i) {
            const int64_t n = i / C;
            const int64_t c = i % C;
            const int8_t* sp = act.ptr + n * act.n_stride + c * npix;
            float* dp = plane.data() + i * npix;
            int64_t k = 0;
            for (; k + simd_default_lane_f32 <= npix; k += simd_default_lane_f32) {
                v_store(dp + k, v_sub(v_cvt_s8_to_f32(sp + k), zvec));
            }
            for (; k < npix; ++k) { dp[k] = static_cast<float>(sp[k]) - zx; }
        });
    }

    auto* out_ptr = output.ptr<TOut>();
    const int64_t out_n_stride   = output.stride_elems(0);
    const int64_t out_c_stride   = output.stride_elems(1);
    const int64_t out_row_stride = output.row_stride_elems();

    ctx.cpu.run(0, gm.N * C * gm.OH, [&](int64_t task) {
        const int64_t oh = task % gm.OH;
        const int64_t c  = (task / gm.OH) % C;
        const int64_t n  = task / (gm.OH * C);
        float* row = rows.data() +
                     static_cast<int64_t>(ctx.cpu.current_thread_id()) * row_len;
        const float* w_c = wd.data() + c * karea;

        row_fill(row, row_len, 0.0f);

        tconv2d_col2im_channel_major<float>(
            cg, plane.data() + (n * C + c) * npix, row, oh,
            [&](float* dst, const float* src, int64_t len, int64_t tap) {
                row_fma_scalar(dst, src, w_c[tap], len);
            }, /*tap_stride=*/0);

        // Groups == channels: the output channel is the input channel.
        const int64_t oc = c;
        const float s_w = quant_scale_at(qw, oc);
        const float req = kIntOut
            ? matmul_int8_requant_scale(s_w, s_x, out_scale) : 0.0f;
        TOut* orow = out_ptr + n * out_n_stride + oc * out_c_stride +
                     oh * out_row_stride;
        for (int64_t r = 0; r < gm.SW; ++r) {
            if (r >= gm.OW) { break; }
            const int64_t nr = (gm.OW - r + gm.SW - 1) / gm.SW;
            const float* rp = row + r * cg.mic;
            for (int64_t t = 0; t < nr; ++t) {
                const int64_t ow = r + t * gm.SW;
                if constexpr (kIntOut) {
                    int32_t v = static_cast<int32_t>(rp[t]);
                    if (b_i32) { v += b_i32[oc]; }
                    if (relu) { v = std::max(v, int32_t{0}); }
                    orow[ow] = matmul_int8_requant<TOut>(v, req, zp_out);
                } else {
                    float f = rp[t] * (s_w * s_x);
                    if (b_fp) { f += s_load(&b_fp[oc]); }
                    f = std::min(std::max(f, gm.epilogue.min_clip),
                                 gm.epilogue.max_clip);
                    f = nnops::apply_epilogue(gm.epilogue, f, oc);
                    s_store(&orow[ow], f);
                }
            }
        }
    });
    return true;
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

    const Tconv2dColGeom cg = gm.col_geom();
    const int64_t row_len = cg.row_len;
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

                    tconv2d_col2im_channel_major<int32_t>(
                        cg, col.data() + oc_l * karea * npix, row, oh,
                        [](int32_t* dst, const int32_t* src, int64_t len, int64_t) {
                            row_add_i32(dst, src, len);
                        }, npix);

                    const int64_t oc = g * ocg + oc0 + oc_l;
                    const float s_w = quant_scale_at(qw, oc);
                    const float req = kIntOut
                        ? matmul_int8_requant_scale(s_w, s_x, out_scale) : 0.0f;
                    TOut* orow = out_n + oc * out_c_stride + oh * out_row_stride;
                    for (int64_t r = 0; r < gm.SW; ++r) {
                        if (r >= gm.OW) { break; }
                        const int64_t nr = (gm.OW - r + gm.SW - 1) / gm.SW;
                        const int32_t* rp = row + r * cg.mic;
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
    const Tconv2dColGeom cg      = gm.col_geom();

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
    PoolArray<float> lane_scale(static_cast<size_t>(cc));
    if (!col || !aux || !rows || !b_scale || !b_zp || !lane_scale) { return false; }

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
            // (kh, kw, oc) triples of one output channel. The merge's own
            // per-lane factor is the same scale folded once per channel.
            for (int64_t o = 0; o < nc; ++o) {
                const int64_t oc = g * ocg + oc0 + o;
                const float s = quant_scale_at(qw, oc);
                const int32_t z = quant_zp_at(qw, oc);
                lane_scale.data()[o] = kIntOut
                    ? matmul_int8_requant_scale(s, s_x, out_scale) : s * s_x;
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

                    tconv2d_col2im_pixel_major<int32_t>(
                        cg, col.data(), m, nc, nc, row, oh,
                        [](int32_t* dst, const int32_t* src, int64_t len, int64_t) {
                            row_add_i32(dst, src, len);
                        });

                    TOut* orow = out_n + oh * out_row_stride + g * ocg + oc0;
                    const TOut* bo_fp = b_fp ? b_fp + g * ocg + oc0 : nullptr;
                    const int64_t oc_base = g * ocg + oc0;
                    if constexpr (kIntOut) {
                        const int32_t* bo = b_i32 ? b_i32 + oc_base : nullptr;
                        for (int64_t ow = 0; ow < gm.OW; ++ow) {
                            requant_oc_row_i8<TOut>(orow + ow * gm.OC, row + ow * nc,
                                                    nc, bo, relu,
                                                    lane_scale.data(), zp_out);
                        }
                    } else if constexpr (std::is_same_v<TOut, float>) {
                        if (epilogue_is_vector_exact(gm.epilogue)) {
                            for (int64_t ow = 0; ow < gm.OW; ++ow) {
                                dequant_oc_row_f32(orow + ow * gm.OC,
                                                   row + ow * nc, nc,
                                                   lane_scale.data(), bo_fp,
                                                   gm.epilogue);
                            }
                        } else {
                            dequant_oc_row_scalar(orow, row, gm.OW, nc, gm.OC,
                                                  bo_fp, qw, oc_base, s_x,
                                                  gm.epilogue);
                        }
                    } else {
                        dequant_oc_row_scalar(orow, row, gm.OW, nc, gm.OC, bo_fp,
                                              qw, oc_base, s_x, gm.epilogue);
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
        if (tconv2d_int8_channel_last_depthwise<TOut>(gm, output, inputs, act, ctx)) {
            return;
        }
        if (tconv2d_int8_channel_last<TOut>(gm, output, inputs, act, ctx)) { return; }
    } else if (tconv2d_int8_channel_first_depthwise<TOut>(gm, output, inputs, act,
                                                           ctx)) {
        return;
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
