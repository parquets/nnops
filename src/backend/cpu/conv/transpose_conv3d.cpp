/// @file transpose_conv3d.cpp
/// @brief CPU TransposeConv3D kernel (NCDHW and NDHWC) — the 2-D col routes
///        with a third tap axis.
///
/// Same gather formulation as TransposeConv2D, one dimension further out. Either
/// layout runs a col GEMM per (n, group): the channel reduction over IC/G, then a
/// scatter that folds the (oc, kd, kh, kw) rows back into the volume
/// (transpose_conv3d_col2im.h). The two layouts differ only in which operand is
/// turned:
///
///   channel-first — the filter is [OC/G·KD·KH·KW] × [IC/G] and the volume
///     [IC/G] × [ID·IH·IW], so the GEMM is `transpose_a` and reads both operands
///     where they lie. The scatter is per (oc, od, oh) row and class-major along
///     W: the depth and height taps only select which input slice row a tap
///     reads.
///   channel-last — the volume is [ID·IH·IW] × [IC/G] voxels (one voxel stride of
///     IC) and the filter is permuted to [IC/G] × [KD·KH·KW·nc], so the col comes
///     out pixel-major and the scatter adds whole oc vectors. Depthwise, whose
///     channels are the vector axis, takes the input volume as its own col and
///     weights each tap instead.
///
/// Both col routes chunk the M side on the same 8 MiB budget, which 3-D reaches
/// sooner: the pixel axis is one dimension larger. The merge that ends a scatter
/// is the only writer of an output element, which is what keeps add_to's original
/// value.
///
/// The fallback is the plain output-driven gather, for the shapes the flat
/// operand views cannot describe: a pitched volume or a kernel taller than the
/// per-row tap list. It walks whichever layout it is given.
///
/// The weight stays channel-first [IC, OC/G, KD, KH, KW] in both cases.

#include "nnops/ops/transpose_conv3d.hpp"
#include "backend/cpu/conv/tconv_row.h"
#include "backend/cpu/conv/transpose_conv3d_col2im.h"
#include "nnops/core/parallel_for.hpp"
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"
#include "backend/cpu/matmul/matmul.h"
#include "backend/cpu/common/dtype_dispatch.hpp"
#include "backend/cpu/common/memory_pool.hpp"

#include <algorithm>

namespace nnops::backend::cpu {

namespace {

/// One transposed convolution's geometry, read off the tensors once.
struct Geom3 {
    int64_t N, IC, ID, IH, IW;
    int64_t OC_per_G, IC_per_G, OC, G;
    int64_t KD, KH, KW, OD, OH, OW;
    int64_t SD, SH, SW, DD, DH, DW, PD, PH, PW;
    Epilogue epilogue;
    bool add_to;
    bool pre_act;   ///< fold add_to's old value in before the epilogue

    /// The col2im module's view of the same numbers.
    Tconv3dColGeom col_geom() const
    {
        return make_col3_geom(ID, IH, IW, OW, KD, KH, KW, SD, SH, SW,
                              DD, DH, DW, PD, PH, PW);
    }
};

/// Largest col one GEMM call may materialise; past it the M rows are split.
constexpr int64_t kColCapBytes = 8 << 20;

/// The col route. Returns false when the operands cannot be read as flat
/// matrices, leaving the caller its gather.
template <typename T>
bool tconv3d_gemm(const Geom3& gm,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    if (weight.row_stride_elems() != gm.KW) { return false; }
    if (input.row_stride_elems() != gm.IW) { return false; }
    if (gm.KD > kMaxRowTaps || gm.KH > kMaxRowTaps) { return false; }

    const int64_t karea = gm.KD * gm.KH * gm.KW;
    const int64_t npix  = gm.ID * gm.IH * gm.IW;
    const int64_t icg   = gm.IC_per_G;
    const int64_t ocg   = gm.OC_per_G;
    if (npix <= 0 || karea <= 0 || icg <= 0) { return false; }
    // The volume view is [IC/G] × [ID·IH·IW], so a channel has to be one whole
    // contiguous volume.
    if (input.stride_elems(1) != npix) { return false; }

    const int64_t in_n_stride    = input.stride_elems(0);
    const int64_t out_n_stride   = output.stride_elems(0);
    const int64_t out_c_stride   = output.stride_elems(1);
    const int64_t out_d_stride   = output.stride_elems(2);
    const int64_t out_row_stride = output.row_stride_elems();

    const Tconv3dColGeom cg = gm.col_geom();
    const int64_t row_len = cg.w.row_len;
    const int64_t slots   = std::max(1, ctx.cpu.worker_count());

    const int64_t elems_per_oc = karea * npix;
    const int64_t cap = kColCapBytes / static_cast<int64_t>(sizeof(T));
    const int64_t chunk = std::max<int64_t>(
        1, std::min(ocg, cap / std::max<int64_t>(elems_per_oc, 1)));

    PoolArray<T> col(static_cast<size_t>(chunk * elems_per_oc));
    PoolArray<T> rows(static_cast<size_t>(slots * row_len));
    if (!col || !rows) { return false; }

    const auto* in_ptr = input.ptr<T>();
    const auto* w_ptr  = weight.ptr<T>();
    const auto* b_ptr  = has_bias ? inputs[2].ptr<T>() : nullptr;
    auto* out_ptr = output.ptr<T>();
    const DataType vdt = input.data_type();

    MatMulAttributes mm{};
    mm.transpose_a = true;

    for (int64_t n = 0; n < gm.N; ++n) {
        const T* in_n = in_ptr + n * in_n_stride;
        T* out_n = out_ptr + n * out_n_stride;

        for (int64_t g = 0; g < gm.G; ++g) {
            const T* w_g = w_ptr + g * icg * (ocg * karea);
            const T* b_g = b_ptr ? b_ptr + g * ocg : nullptr;

            // B: the group's whole input volume, [IC/G] × [ID·IH·IW].
            const int64_t bshape[] = {icg, npix};
            TensorView bm(std::span<const int64_t>(bshape, 2), vdt,
                          const_cast<T*>(in_n + g * icg * npix),
                          static_cast<int64_t>(npix * sizeof(T)), TensorLayout::NCHW);

            for (int64_t oc0 = 0; oc0 < ocg; oc0 += chunk) {
                const int64_t cc = std::min(chunk, ocg - oc0);
                const int64_t m  = cc * karea;

                // A: the filter rows of this oc slice, [IC/G] × [OC/G·KD·KH·KW],
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

                // One task per output row: the taps that land on it, then the
                // residue walk along W, then the merge that owns the row.
                ctx.cpu.run(0, cc * gm.OD * gm.OH, [&](int64_t task) {
                    const int64_t oc_l = task / (gm.OD * gm.OH);
                    const int64_t rest = task % (gm.OD * gm.OH);
                    const int64_t od   = rest / gm.OH;
                    const int64_t oh   = rest % gm.OH;
                    T* row = rows.data() +
                             static_cast<int64_t>(ctx.cpu.current_thread_id()) * row_len;

                    row_fill(row, row_len, b_g ? s_load(&b_g[oc_l]) : 0.0f);

                    tconv3d_col2im_channel_major<T>(
                        cg, col.data() + oc_l * karea * npix, row, od, oh,
                        [](T* dst, const T* src, int64_t len, int64_t) {
                            row_add(dst, src, len);
                        });

                    const int64_t oc = g * ocg + oc0 + oc_l;
                    T* orow = out_n + oc * out_c_stride + od * out_d_stride +
                              oh * out_row_stride;
                    for (int64_t r = 0; r < gm.SW && r < gm.OW; ++r) {
                        const int64_t nr = (gm.OW - r + gm.SW - 1) / gm.SW;
                        const T* rp = row + r * cg.w.mic;
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

// =========================================================================
//  Channel-last (NDHWC)
// =========================================================================
//
// The same col GEMM as the channel-first route, with the operands turned:
// A is the volume as [ID·IH·IW] × [IC/G] voxels (a voxel's channels are
// contiguous), B is the filter as [IC/G] × [KD·KH·KW·nc] taps, and the scatter
// adds whole oc vectors out of the pixel-major col. It is the 2-D channel-last
// structure (transpose_conv2d.cpp) one axis deeper — see that file for why the
// col beats the pixel kernel once the filter stops staying resident.
//
// Depthwise has its own route, as in 2-D: one channel per group makes the
// channel the vector axis, so the "col" is the input volume itself and each tap
// is one weighted vector add per voxel.

/// One channel-last col GEMM call, resolved to flat operands: the chunk walk
/// below varies only the group and the oc chunk.
template <typename T>
struct ClCall3 {
    const Geom3& gm;
    Tconv3dColGeom cg;
    const T *in, *w, *b;
    T *out, *col, *aux, *rows;
    DataType vdt;
    int64_t icg, ocg, npix, karea;
    int64_t in_n_stride, out_n_stride, out_row_stride;
    bool direct_w;
};

/// One oc chunk: B for the chunk — the filter as [IC/G] × [KD·KH·KW·nc],
/// permuted, or the channel-first rows read in place with oc still the faster
/// axis — then per batch item the GEMM, the direct-weight col transpose, and the
/// scatter and merge of each output row.
template <typename T>
void tconv3d_cl_chunk(const ClCall3<T>& c, const ComputeContext& ctx,
                      int64_t g, int64_t oc0, int64_t nc)
{
    const int64_t m = nc * c.karea;

    if (!c.direct_w) {
        // A blocked transpose of the channel-first [IC/G][OC/G][KD·KH·KW] rows.
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

        // A: the channel-last volume as [ID·IH·IW] × [IC/G] voxels — one voxel
        // stride of IC, so the group offset is g·IC/G elements in.
        const int64_t ashape[] = {c.npix, c.icg};
        TensorView am(std::span<const int64_t>(ashape, 2), c.vdt,
                      const_cast<T*>(in_n + g * c.icg),
                      static_cast<int64_t>(c.gm.IC * sizeof(T)), TensorLayout::NCHW);
        const int64_t cshape[] = {c.npix, m};
        TensorView cm(std::span<const int64_t>(cshape, 2), c.vdt, c.col,
                      static_cast<int64_t>(m * sizeof(T)), TensorLayout::NCHW);

        const TensorView mins[] = {am, bm};
        const MatMulAttributes mm{};
        matmul_cpu(mm, cm, mins, ctx, nullptr);

        const T* cbase = c.col;
        if (c.direct_w) {
            // Per voxel, [oc][tap] → [tap][oc], so the scatter sees the same col
            // either way.
            for (int64_t p = 0; p < c.npix; ++p) {
                permute_taps(c.col + p * m, c.aux + p * m, nc, c.karea);
            }
            cbase = c.aux;
        }

        const int64_t row_len = c.gm.OW * nc;
        ctx.cpu.run(0, c.gm.OD * c.gm.OH, [&](int64_t task) {
            const int64_t od = task / c.gm.OH;
            const int64_t oh = task % c.gm.OH;
            T* row = c.rows +
                     static_cast<int64_t>(ctx.cpu.current_thread_id()) * row_len;
            row_fill(row, row_len, 0.0f);

            tconv3d_col2im_pixel_major<T>(
                c.cg, cbase, m, nc, nc, row, od, oh,
                [](T* dst, const T* src, int64_t len, int64_t) {
                    row_add(dst, src, len);
                });

            T* orow = out_n + (od * c.gm.OH + oh) * c.out_row_stride +
                      g * c.ocg + oc0;
            merge_oc_row(orow, row, c.b, g * c.ocg + oc0, c.gm, nc);
        });
    }
}

/// Channel-last through a col matrix — the 2-D route one axis deeper, and what
/// MNN's NC4HW4 deconvolution does: one GEMM per (batch, group) does the channel
/// reduction for the whole volume, and a depth/height tap list plus a
/// stride-residue scatter fold the col back. Returns false when the operands
/// cannot be read as flat matrices.
template <typename T>
bool tconv3d_channel_last_gemm(const Geom3& gm,
                               TensorView& output,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    if (weight.row_stride_elems() != gm.KW) { return false; }
    if (input.row_stride_elems() != gm.IW * gm.IC) { return false; }
    // The volume view is [ID·IH·IW] × [IC/G] voxels, so a depth slice has to be
    // one contiguous plane whose stride the H rows already imply.
    if (input.stride_elems(2) != gm.IH * gm.IW * gm.IC) { return false; }
    if (gm.KD > kMaxRowTaps || gm.KH > kMaxRowTaps) { return false; }

    const int64_t karea = gm.KD * gm.KH * gm.KW;
    const int64_t npix  = gm.ID * gm.IH * gm.IW;
    const int64_t icg   = gm.IC_per_G;
    const int64_t ocg   = gm.OC_per_G;
    if (npix <= 0 || karea <= 0 || icg <= 0 || ocg <= 0) { return false; }

    const int64_t in_n_stride    = input.stride_elems(0);
    const int64_t out_n_stride   = output.stride_elems(0);
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t slots          = std::max(1, ctx.cpu.worker_count());
    const Tconv3dColGeom cg      = gm.col_geom();

    // Two col orientations, exactly as in 2-D: [tap][oc] needs the filter
    // permuted; [oc][tap] lets the raw filter slice be B and pays a transpose
    // per voxel instead. Walk the smaller of the filter and the col.
    const bool direct_w = npix * gm.N < icg;

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

    const ClCall3<T> call{gm,         cg,          in_ptr,       w_ptr,
                          b_ptr,      out_ptr,     col.data(),   aux.data(),
                          rows.data(), vdt,        icg,          ocg,
                          npix,       karea,       in_n_stride,  out_n_stride,
                          out_row_stride, direct_w};

    for (int64_t g = 0; g < gm.G; ++g) {
        for (int64_t oc0 = 0; oc0 < ocg; oc0 += cc) {
            const int64_t nc = std::min(cc, ocg - oc0);
            if (nc > 0) { tconv3d_cl_chunk(call, ctx, g, oc0, nc); }
        }
    }
    return true;
}

/// Depthwise channel-last: one channel per group means group g's input channel
/// is output channel g, so the vector axis is the channel itself and an output
/// voxel accumulates the whole C-wide tap vector at once. The filter is
/// repacked to [KD·KH·KW][C] so a tap is one contiguous vector. Returns false
/// for anything that is not depthwise, or when the row scratch would not fit.
template <typename T>
bool tconv3d_channel_last_depthwise(const Geom3& gm,
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
    if (input.stride_elems(2) != gm.IH * gm.IW * gm.IC) { return false; }

    const int64_t karea   = gm.KD * gm.KH * gm.KW;
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
    const Tconv3dColGeom cg      = gm.col_geom();
    const int64_t rows_per_n     = gm.OD * gm.OH;

    ctx.cpu.run(0, gm.N * rows_per_n, [&](int64_t task) {
        const int64_t n    = task / rows_per_n;
        const int64_t rest = task % rows_per_n;
        const int64_t od   = rest / gm.OH;
        const int64_t oh   = rest % gm.OH;
        T* row = rows.data() +
                 static_cast<int64_t>(ctx.cpu.current_thread_id()) * row_len;
        const T* in_n = in_ptr + n * in_n_stride;

        row_fill(row, row_len, 0.0f);

        // The "col" is the input volume itself: one channel per group means the
        // tap indexes the repacked filter, not a source stride.
        tconv3d_col2im_pixel_major<T>(
            cg, in_n, gm.IC, C, 0, row, od, oh,
            [&](T* dst, const T* src, int64_t len, int64_t tap) {
                row_fma(dst, src, wd.data() + tap * C, len);
            });

        T* orow = out_ptr + n * out_n_stride + rest * out_row_stride;
        merge_oc_row(orow, row, b_ptr, 0, gm, C);
    });
    return true;
}

/// The output-driven gather: every output voxel walks its own taps. Correct for
/// every geometry and layout, and the route a shape too odd for the col takes.
template <typename T>
void tconv3d_gather(const Geom3& gm,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t in_c_stride    = input.stride_elems(1);
    const int64_t in_d_stride    = input.stride_elems(2);
    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t in_n_stride    = input.stride_elems(0);
    const int64_t out_c_stride   = output.stride_elems(1);
    const int64_t out_d_stride   = output.stride_elems(2);
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_n_stride   = output.stride_elems(0);

    const auto* in_ptr = input.ptr<T>();
    const auto* w_ptr  = weight.ptr<T>();
    const auto* b_ptr  = has_bias ? inputs[2].ptr<T>() : nullptr;
    auto* out_ptr = output.ptr<T>();

    // One voxel step is one element channel-first and one channel channel-last;
    // the row and slice strides already carry the rest of either layout.
    const int64_t in_px  = is_channel_last(input.layout()) ? gm.IC : 1;
    const int64_t out_px = is_channel_last(output.layout()) ? gm.OC : 1;

    const int64_t karea = gm.KD * gm.KH * gm.KW;

    ctx.cpu.run(0, gm.N * gm.OC, [&](int64_t task) {
        const int64_t n  = task / gm.OC;
        const int64_t oc = task % gm.OC;
        const int64_t g  = oc / gm.OC_per_G;
        const int64_t oc_l = oc % gm.OC_per_G;

        const T* in_n = in_ptr + n * in_n_stride;
        T* out_c = out_ptr + n * out_n_stride + oc * out_c_stride;

        for (int64_t od = 0; od < gm.OD; ++od) {
            for (int64_t oh = 0; oh < gm.OH; ++oh) {
                T* orow = out_c + od * out_d_stride + oh * out_row_stride;

                for (int64_t ow = 0; ow < gm.OW; ++ow) {
                    T* op = orow + ow * out_px;
                    float acc = 0.0f;

                    for (int64_t kd = 0; kd < gm.KD; ++kd) {
                        const int64_t num_d = od + gm.PD - kd * gm.DD;
                        if (num_d < 0 || num_d % gm.SD != 0) { continue; }
                        const int64_t id = num_d / gm.SD;
                        if (id >= gm.ID) { continue; }

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

                                const T* xp = in_n + (g * gm.IC_per_G) * in_c_stride +
                                              id * in_d_stride + ih * in_row_stride +
                                              iw * in_px;
                                const T* wp = w_ptr +
                                    ((g * gm.IC_per_G) * gm.OC_per_G + oc_l) * karea +
                                    (kd * gm.KH + kh) * gm.KW + kw;
                                for (int64_t ic_l = 0; ic_l < gm.IC_per_G; ++ic_l) {
                                    acc += s_load(xp + ic_l * in_c_stride) *
                                           s_load(wp + ic_l * (gm.OC_per_G * karea));
                                }
                            }
                        }
                    }

                    if (b_ptr) { acc += s_load(&b_ptr[oc]); }
                    if (gm.pre_act) { acc += s_load(op); }
                    acc = nnops::apply_epilogue(gm.epilogue, acc, oc);
                    if (gm.add_to && !gm.pre_act) { acc += s_load(op); }
                    s_store(op, acc);
                }
            }
        }
    });
}

template <typename T>
void tconv3d_impl(const Geom3& gm,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx)
{
    if (gm.N == 0 || gm.OC == 0 || gm.OD == 0 || gm.OH == 0 || gm.OW == 0) { return; }
    if (is_channel_last(inputs[0].layout())) {
        if (tconv3d_channel_last_depthwise<T>(gm, output, inputs, ctx)) { return; }
        if (tconv3d_channel_last_gemm<T>(gm, output, inputs, ctx)) { return; }
    } else if (tconv3d_gemm<T>(gm, output, inputs, ctx)) {
        return;
    }
    tconv3d_gather<T>(gm, output, inputs, ctx);
}

}  // anonymous namespace

Status transpose_conv3d_cpu(const TransposeConv3DAttributes& attrs,
                            TensorView& output,
                            std::span<const TensorView> inputs,
                            const ComputeContext& ctx,
                            void* /*workspace*/)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];

    Geom3 gm{};
    gm.N  = input.shape(0);
    gm.IC = input.shape(1);
    gm.ID = input.shape(2);
    gm.IH = input.shape(3);
    gm.IW = input.shape(4);
    gm.OC_per_G = weight.shape(1);
    gm.KD = attrs.kernel_size[0];
    gm.KH = attrs.kernel_size[1];
    gm.KW = attrs.kernel_size[2];
    gm.G  = attrs.groups;
    gm.OC = gm.OC_per_G * gm.G;
    gm.IC_per_G = gm.IC / gm.G;
    gm.OD = output.shape(2);
    gm.OH = output.shape(3);
    gm.OW = output.shape(4);
    gm.SD = attrs.stride[0];
    gm.SH = attrs.stride[1];
    gm.SW = attrs.stride[2];
    gm.DD = attrs.dilation[0];
    gm.DH = attrs.dilation[1];
    gm.DW = attrs.dilation[2];
    gm.PD = attrs.padding[0];
    gm.PH = attrs.padding[1];
    gm.PW = attrs.padding[2];
    gm.epilogue = attrs.epilogue;
    gm.add_to = attrs.add_to;
    gm.pre_act = attrs.add_to && attrs.residual_pre_activation;

    dispatch_f32_f16(input.data_type(), "transpose_conv3d_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        tconv3d_impl<T>(gm, output, inputs, ctx);
    });
    return Status::ok;
}

}  // namespace nnops::backend::cpu
