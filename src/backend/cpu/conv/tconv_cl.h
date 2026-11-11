#pragma once
/// @file tconv_cl.h
/// @brief The channel-last route of transposed convolution, shared by the 2-D
///        and 3-D operators: the col chunk driver and the depthwise driver.
///
/// Both ranks run the same structure — one GEMM per (batch, group, oc chunk) over
/// the channel-last image, then a stride-residue scatter of the col into a
/// scratch row and a merge that owns that row — and differ only in the geometry
/// and in which walk folds the col back (simd_col2im.hpp). So the rank supplies
/// a walk and everything else lives here: the filter permutation, the operand
/// views, the oc chunk split, the row scratch and the merge.
///
/// `rows_per_n` counts the output rows of one image — OH in 2-D, OD·OH in 3-D —
/// and the flat row index names which one a task scatters, so the walk turns it
/// back into (oh) or (od, oh). One row-slice of `nc` channels is what a task
/// owns, and the merge is the only writer of an output element, which is what
/// lets add_to read the old value without a save/restore pass.
///
/// The depthwise driver is the same skeleton with the input standing in for the
/// col: one channel per group makes the channel the vector axis, so each tap is
/// one weighted vector add per pixel and there is no GEMM at all.
///
/// This is route glue, not a kernel — it needs TensorView, matmul_cpu and the
/// pool — so it sits beside conv2d_implicit_gemm.h in conv/ rather than in
/// simd_kernel/, which holds only code that needs nothing but the SIMD wrapper.

#include "nnops/core/compute_context.hpp"
#include "nnops/core/epilogue.hpp"
#include "nnops/core/tensor_view.hpp"
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"
#include "backend/cpu/simd_kernel/simd_tconv_row.hpp"
#include "backend/cpu/common/memory_pool.hpp"
#include "backend/cpu/matmul/matmul.h"

#include <algorithm>
#include <span>

namespace nnops::backend::cpu {

using namespace nnops::simd;

/// Merge one channel-last scratch row into the output: bias, then the
/// pre-activation residual, the epilogue and add_to, over `nc` of the row's
/// channels starting at @p oc_base. The caller has this row to itself, which is
/// what keeps add_to's original value — this is the only read and the only write.
///
/// `gm` is duck-typed on OW, OC, epilogue, pre_act and add_to, so the 2-D and
/// 3-D geometries both fit without either kernel learning the other's type.
template <typename T, typename G>
void merge_oc_row(T* orow, const T* row, const T* bias, int64_t oc_base,
                  const G& gm, int64_t nc)
{
    const T* type_tag = nullptr;
    constexpr int L = simd_lane_for<T>;
    for (int64_t ow = 0; ow < gm.OW; ++ow) {
        const T* rp = row + ow * nc;
        T* dp = orow + ow * gm.OC;
        int64_t c = 0;
        for (; c + L <= nc; c += L) {
            vec_for<T> acc = v_load(rp + c);
            if (bias) { acc = v_add(acc, v_load(bias + oc_base + c)); }
            if (gm.pre_act) { acc = v_add(acc, v_load(dp + c)); }
            acc = apply_epilogue_vec<T>(gm.epilogue, type_tag, acc);
            if (gm.add_to && !gm.pre_act) { acc = v_add(acc, v_load(dp + c)); }
            v_store(dp + c, acc);
        }
        for (; c < nc; ++c) {
            float v = s_load(rp + c);
            if (bias) { v += s_load(bias + oc_base + c); }
            if (gm.pre_act) { v += s_load(dp + c); }
            v = nnops::apply_epilogue(gm.epilogue, v, oc_base + c);
            if (gm.add_to && !gm.pre_act) { v += s_load(dp + c); }
            s_store(dp + c, v);
        }
    }
}

/// Largest col one GEMM call may materialise; past it the oc chunk shrinks.
inline constexpr int64_t kColCapBytes = 8 << 20;

/// One channel-last col GEMM call, resolved to flat operands: the chunk walk
/// varies only the group and the oc chunk.
template <typename T, typename G>
struct TconvClCall {
    const G& gm;
    const T *in, *w, *b;
    T *out, *col, *aux, *rows;
    DataType vdt;
    int64_t icg, ocg, npix, karea, rows_per_n;
    int64_t in_n_stride, out_n_stride, out_row_stride;
    bool direct_w;
};

/// The oc chunk: how many output channels one GEMM plus one per-worker row
/// scratch fit the col cap — half of it when the filter is walked per call
/// (direct_w), because `aux` then also holds a col copy.
inline int64_t tconv_cl_oc_chunk(int64_t ocg, int64_t npix, int64_t karea,
                                 int64_t OW, int64_t slots, int64_t esz,
                                 bool direct_w)
{
    const int64_t elem_cap = kColCapBytes / esz;
    const int64_t cap = direct_w ? elem_cap / 2 : elem_cap;
    int64_t cc = ocg;
    cc = std::min(cc, std::max<int64_t>(1, cap / (npix * karea)));
    cc = std::min(cc, std::max<int64_t>(1, cap / (slots * OW)));
    return cc;
}

/// One oc chunk: B for the chunk — the filter as [IC/G] × [KH·KW·nc], permuted,
/// or the channel-first rows read in place with oc still the faster axis — then
/// per batch item the GEMM, the direct-weight col transpose, and the scatter and
/// merge of each output row.
template <typename T, typename G, class Walk>
void tconv_cl_chunk(const TconvClCall<T, G>& c, Walk&& walk,
                    const ComputeContext& ctx, int64_t g, int64_t oc0, int64_t nc)
{
    const int64_t m = nc * c.karea;

    if (!c.direct_w) {
        // A blocked transpose of the channel-first [IC/G][OC/G][taps] rows.
        for (int64_t ic = 0; ic < c.icg; ++ic) {
            const T* src = c.w + ((g * c.icg + ic) * c.ocg + oc0) * c.karea;
            nnops::kernel::permute_taps(src, c.aux + ic * c.karea * nc, nc, c.karea);
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

        // A: the channel-last image as [pixels] × [IC/G] — one pixel stride of
        // IC, so the group offset is g·IC/G elements in.
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
            // Per pixel, [oc][tap] → [tap][oc], so the scatter sees the same col
            // either way.
            for (int64_t p = 0; p < c.npix; ++p) {
                nnops::kernel::permute_taps(c.col + p * m, c.aux + p * m, nc, c.karea);
            }
            cbase = c.aux;
        }

        const int64_t row_len = c.gm.OW * nc;
        ctx.cpu.run(0, c.rows_per_n, [&](int64_t row_index) {
            T* row = c.rows +
                     static_cast<int64_t>(ctx.cpu.current_thread_id()) * row_len;
            nnops::kernel::row_fill(row, row_len, 0.0f);

            walk(cbase, row, row_index, m, nc);

            T* orow = out_n + row_index * c.out_row_stride + g * c.ocg + oc0;
            merge_oc_row(orow, row, c.b, g * c.ocg + oc0, c.gm, nc);
        });
    }
}

/// The whole channel-last col route: the col, the filter staging and the row
/// scratch are pooled once, then one chunk call per (group, oc chunk).
template <typename T, typename G, class Walk>
bool tconv_cl_route(const G& gm, TensorView& output,
                    std::span<const TensorView> inputs, const ComputeContext& ctx,
                    int64_t npix, int64_t rows_per_n, int64_t karea,
                    int64_t direct_w, Walk&& walk)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t icg = gm.IC_per_G;
    const int64_t ocg = gm.OC_per_G;
    if (npix <= 0 || karea <= 0 || icg <= 0 || ocg <= 0) { return false; }

    const int64_t slots = std::max(1, ctx.cpu.worker_count());
    const int64_t cc = tconv_cl_oc_chunk(ocg, npix, karea, gm.OW, slots,
                                         static_cast<int64_t>(sizeof(T)),
                                         direct_w != 0);

    PoolArray<T> col(static_cast<size_t>(npix * karea * cc));
    PoolArray<T> aux(static_cast<size_t>((direct_w ? npix : icg) * karea * cc));
    PoolArray<T> rows(static_cast<size_t>(slots * gm.OW * cc));
    if (!col || !aux || !rows) { return false; }

    const TconvClCall<T, G> call{gm,
                                 input.ptr<T>(),
                                 weight.ptr<T>(),
                                 has_bias ? inputs[2].ptr<T>() : nullptr,
                                 output.ptr<T>(),
                                 col.data(),
                                 aux.data(),
                                 rows.data(),
                                 input.data_type(),
                                 icg,
                                 ocg,
                                 npix,
                                 karea,
                                 rows_per_n,
                                 input.stride_elems(0),
                                 output.stride_elems(0),
                                 output.row_stride_elems(),
                                 direct_w != 0};

    for (int64_t g = 0; g < gm.G; ++g) {
        for (int64_t oc0 = 0; oc0 < ocg; oc0 += cc) {
            const int64_t nc = std::min(cc, ocg - oc0);
            if (nc > 0) { tconv_cl_chunk(call, walk, ctx, g, oc0, nc); }
        }
    }
    return true;
}

/// The channel-last depthwise route: the input itself is the col, so there is no
/// GEMM — the filter is permuted to [taps][C] and each tap is one weighted vector
/// add over the image. `walk(src, row, row_index, wd, m, nc)` gets the permuted
/// filter so its tap indexes `wd` rather than the source. Returns false when the
/// row scratch would not fit.
template <typename T, typename G, class Walk>
bool tconv_cl_depthwise(const G& gm, TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx, int64_t rows_per_n,
                        int64_t karea, int64_t C, Walk&& walk)
{
    if (gm.IC_per_G != 1 || gm.OC_per_G != 1) { return false; }

    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t slots   = std::max(1, ctx.cpu.worker_count());
    const int64_t row_len = gm.OW * C;
    const int64_t elem_cap = kColCapBytes / static_cast<int64_t>(sizeof(T));
    if (C <= 0 || slots * row_len > elem_cap) { return false; }

    PoolArray<T> wd(static_cast<size_t>(karea * C));
    PoolArray<T> rows(static_cast<size_t>(slots * row_len));
    if (!wd || !rows) { return false; }

    nnops::kernel::permute_taps(weight.ptr<T>(), wd.data(), C, karea);

    const T* in_ptr  = input.ptr<T>();
    const T* b_ptr   = has_bias ? inputs[2].ptr<T>() : nullptr;
    T* out_ptr       = output.ptr<T>();
    const int64_t in_n_stride    = input.stride_elems(0);
    const int64_t out_n_stride   = output.stride_elems(0);
    const int64_t out_row_stride = output.row_stride_elems();

    ctx.cpu.run(0, gm.N * rows_per_n, [&](int64_t task) {
        const int64_t n = task / rows_per_n;
        const int64_t row_index = task % rows_per_n;
        T* row = rows.data() +
                 static_cast<int64_t>(ctx.cpu.current_thread_id()) * row_len;

        nnops::kernel::row_fill(row, row_len, 0.0f);
        walk(in_ptr + n * in_n_stride, row, row_index, wd.data(), C, C);

        T* orow = out_ptr + n * out_n_stride + row_index * out_row_stride;
        merge_oc_row(orow, row, b_ptr, 0, gm, C);
    });
    return true;
}

}  // namespace nnops::backend::cpu
