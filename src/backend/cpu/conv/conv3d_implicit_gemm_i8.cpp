/// @file conv3d_implicit_gemm_i8.cpp
/// @brief CPU Conv3D — the quantized route.
///
/// An s8 filter against a pre-quantized u8/s8 activation (or an fp one
/// quantized inside the GEMM's A pack), MatMulInteger shaped: the same
/// im2col → GEMM the float route runs, with the int32 accumulate and the
/// dequantize/requantize left to MatMul's own output pass. That pass is the
/// same shared code Conv2D's quantized kernel uses (simd_epilogue's int8 tile),
/// so the two operators agree on the compensation order, the requant rounding
/// and where the int32 bias enters.
///
/// The activation reaches the GEMM exactly as stored: a pre-quantized one is
/// gathered byte for byte with the zero point as the padding fill (a padded tap
/// is a real 0, which is the zero point in the quantized domain), and an fp one
/// is gathered as fp with 0 padding and quantized by MatMul's fused A pack with
/// `act_quant_dtype`'s clamp — 0 quantizes to the zero point either way.
///
/// The filter is DHWIO [KD][KH][KW][IC/G][OC] — TensorFlow's order, which is
/// the GEMM's B already; each group's per-channel parameters are handed to the
/// MatMul view as a slice, so the GEMM's per-column scale, zero point and
/// compensation all index from 0. The bias is fp on an fp destination and
/// int32 on s8/u8, as on Conv2D.
///
/// The prepacked blob (Conv3D::prepackWeights) holds each group's dense
/// k-major matrix, not MatMul's segmented s8 panels: those fold one activation
/// convention's shift into the blob, and this route accepts u8, s8 and fp
/// activations alike.
///
/// What is refused rather than approximated: add_to (the integer path has no
/// residual stage), an fp activation against an output of the other float type,
/// an unsupported destination dtype, and any filter whose k axis is not dense
/// (the scalar gather that backs the float route has no quantized twin).

#include "backend/cpu/conv/conv3d_implicit_gemm_internal.h"
#include "backend/cpu/simd_kernel/simd_im2col.hpp"
#include "backend/cpu/common/memory_pool.hpp"
#include "backend/cpu/matmul/matmul.h"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <span>

namespace nnops::backend::cpu {

namespace {

using nnops::kernel::tiled_im2col_3d;
using nnops::kernel::tiled_im2col_3d_i8;

/// A conv activation is per-tensor: the arrays collapse to element 0, which is
/// what the gather and the compensation read (Conv2D does the same).
QuantParams per_tensor(const QuantParams& q)
{
    QuantParams r;
    r.scale = q.scale_data != nullptr ? q.scale_data[0] : q.scale;
    r.zero_point = q.zero_point_data != nullptr ? q.zero_point_data[0] : q.zero_point;
    return r;
}

/// A group's slice of the filter's per-channel parameters.
QuantParams group_params(const QuantParams& q, int64_t off, int64_t n)
{
    QuantParams r = q;
    if (q.scale_data != nullptr) {
        r.scale_data = q.scale_data + off;
        r.num_scales = n;
    }
    if (q.zero_point_data != nullptr) {
        r.zero_point_data = q.zero_point_data + off;
        r.num_scales = n;
    }
    return r;
}

/// The quantized im2col → MatMulInteger route. False when the operands have no
/// flat form (the caller refuses).
bool conv3d_i8_col_gemm(const Conv3DAttributes& attrs, const ConvChannels& gm, TensorView& output,
                        std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& act = inputs[0];
    const auto& wgt = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const DataType adt = act.data_type();
    const DataType odt = output.data_type();
    const bool fp_act = adt == DataType::f32 || adt == DataType::f16;

    const int64_t kdim = gm.IC_per_G * gm.karea();
    const int64_t npix = gm.npix();
    if (kdim <= 0 || npix <= 0 || gm.OC_per_G <= 0) {
        return false;
    }
    // The filter must read as one dense [IC/G·KD·KH·KW][OC/G] matrix per group.
    if (!conv3d_filter_is_matrix(wgt, gm)) {
        return false;
    }

    const int64_t esz_a = static_cast<int64_t>(data_type_size(adt));
    const int64_t esz_c = static_cast<int64_t>(data_type_size(odt));
    const int64_t esz_b = has_bias ? static_cast<int64_t>(data_type_size(inputs[2].data_type())) : 0;

    const int64_t in_n_stride = act.stride_elems(0);
    const int64_t in_d_stride = act.stride_elems(1);
    const int64_t in_row_stride = act.row_stride_elems();
    const int64_t in_pix = act.shape(4);
    const int64_t out_n_stride = output.stride_elems(0);
    const int64_t out_d_stride = output.stride_elems(1);
    const int64_t out_row_stride = output.row_stride_elems();

    const bool in_dense =
        (in_row_stride == gm.IW * in_pix) && (in_d_stride == gm.IH * in_row_stride);
    const bool pointwise = gm.KD == 1 && gm.KH == 1 && gm.KW == 1 && gm.SD == 1 && gm.SH == 1 &&
                           gm.SW == 1 && gm.PD == 0 && gm.PH == 0 && gm.PW == 0 && in_dense;

    // add_to is refused upstream, so a back-to-back destination is written in
    // place; anything else goes through a staged tile the scatter copies out.
    const bool direct = output.row_stride_elems() == gm.OW * gm.OC;

    const QuantParams qa = per_tensor(act.quant_params());
    const QuantParams qw = wgt.quant_params();
    const QuantParams qc = output.quant_params();
    const int32_t zp_a = qa.zero_point;

    const auto* in_bytes = static_cast<const uint8_t*>(act.ptr<void>());
    const auto* b_bytes = has_bias ? static_cast<const uint8_t*>(inputs[2].ptr<void>()) : nullptr;
    auto* out_bytes = static_cast<uint8_t*>(output.ptr<void>());

    // One pixel block: gather A, run the quantized GEMM, scatter what C cannot
    // hold. @p ictx carries the pool: the real context in the serial arm, an
    // empty one in the task grid. @p wg is the group's B, shared read-only.
    const auto run_block = [&](int64_t n, int64_t g, int64_t od0, int64_t oh0, int64_t od_chunk,
                               int64_t roi_h, uint8_t* col, uint8_t* stage,
                               const TensorView& wg, const ComputeContext& ictx) {
        const int64_t nd = std::min(od_chunk, gm.OD - od0);
        const int64_t nh = std::min(roi_h, gm.OH - oh0);
        const int64_t nblk = nd * nh * gm.OW;
        const int64_t p0 = (od0 * gm.OH + oh0) * gm.OW;

        const int64_t in_off = n * in_n_stride + g * gm.IC_per_G;
        const int64_t out_off = n * out_n_stride + g * gm.OC_per_G;
        const uint8_t* in_g = in_bytes + in_off * esz_a;
        uint8_t* out_g = out_bytes + out_off * esz_c;
        const void* b_g = has_bias ? b_bytes + g * gm.OC_per_G * esz_b : nullptr;

        const int icnc = static_cast<int>(gm.IC_per_G);
        void* a_ptr = nullptr;
        if (pointwise) {
            a_ptr = const_cast<uint8_t*>(in_g) + p0 * in_pix * esz_a;
        } else {
            // One (od, oh) row at a time so the serial arm puts the (byte,
            // bandwidth-bound) gather on the pool too.
            const int64_t rows = nd * nh;
            const auto gather_row = [&](int64_t r) {
                uint8_t* const dst = col + r * gm.OW * kdim * esz_a;
                const int od = static_cast<int>(od0 + r / nh);
                const int oh = static_cast<int>(oh0 + r % nh);
                if (fp_act) {
                    if (adt == DataType::f32) {
                        tiled_im2col_3d(reinterpret_cast<float*>(dst), static_cast<int>(kdim),
                                        reinterpret_cast<const float*>(in_g), in_pix,
                                        in_d_stride, in_row_stride, icnc, od, oh, 1, 1,
                                        static_cast<int>(gm.OW), static_cast<int>(gm.ID),
                                        static_cast<int>(gm.IH), static_cast<int>(gm.IW),
                                        static_cast<int>(gm.KD), static_cast<int>(gm.KH),
                                        static_cast<int>(gm.KW), static_cast<int>(gm.SD),
                                        static_cast<int>(gm.SH), static_cast<int>(gm.SW),
                                        static_cast<int>(gm.DD), static_cast<int>(gm.DH),
                                        static_cast<int>(gm.DW), static_cast<int>(gm.PD),
                                        static_cast<int>(gm.PH), static_cast<int>(gm.PW));
                    } else {
                        tiled_im2col_3d(reinterpret_cast<half*>(dst), static_cast<int>(kdim),
                                        reinterpret_cast<const half*>(in_g), in_pix,
                                        in_d_stride, in_row_stride, icnc, od, oh, 1, 1,
                                        static_cast<int>(gm.OW), static_cast<int>(gm.ID),
                                        static_cast<int>(gm.IH), static_cast<int>(gm.IW),
                                        static_cast<int>(gm.KD), static_cast<int>(gm.KH),
                                        static_cast<int>(gm.KW), static_cast<int>(gm.SD),
                                        static_cast<int>(gm.SH), static_cast<int>(gm.SW),
                                        static_cast<int>(gm.DD), static_cast<int>(gm.DH),
                                        static_cast<int>(gm.DW), static_cast<int>(gm.PD),
                                        static_cast<int>(gm.PH), static_cast<int>(gm.PW));
                    }
                } else if (adt == DataType::u8) {
                    tiled_im2col_3d_i8(dst, static_cast<int>(kdim), in_g, in_pix, in_d_stride,
                                       in_row_stride, icnc, od, oh, 1, 1,
                                       static_cast<int>(gm.OW), static_cast<int>(gm.ID),
                                       static_cast<int>(gm.IH), static_cast<int>(gm.IW),
                                       static_cast<int>(gm.KD), static_cast<int>(gm.KH),
                                       static_cast<int>(gm.KW), static_cast<int>(gm.SD),
                                       static_cast<int>(gm.SH), static_cast<int>(gm.SW),
                                       static_cast<int>(gm.DD), static_cast<int>(gm.DH),
                                       static_cast<int>(gm.DW), static_cast<int>(gm.PD),
                                       static_cast<int>(gm.PH), static_cast<int>(gm.PW),
                                       static_cast<uint8_t>(zp_a));
                } else {
                    tiled_im2col_3d_i8(reinterpret_cast<int8_t*>(dst), static_cast<int>(kdim),
                                       reinterpret_cast<const int8_t*>(in_g), in_pix,
                                       in_d_stride, in_row_stride, icnc, od, oh, 1, 1,
                                       static_cast<int>(gm.OW), static_cast<int>(gm.ID),
                                       static_cast<int>(gm.IH), static_cast<int>(gm.IW),
                                       static_cast<int>(gm.KD), static_cast<int>(gm.KH),
                                       static_cast<int>(gm.KW), static_cast<int>(gm.SD),
                                       static_cast<int>(gm.SH), static_cast<int>(gm.SW),
                                       static_cast<int>(gm.DD), static_cast<int>(gm.DH),
                                       static_cast<int>(gm.DW), static_cast<int>(gm.PD),
                                       static_cast<int>(gm.PH), static_cast<int>(gm.PW),
                                       static_cast<int8_t>(zp_a));
                }
            };
            ictx.cpu.run(0, rows, gather_row);
            a_ptr = col;
        }

        const int64_t ashape[] = {nblk, kdim};
        TensorView am(std::span<const int64_t>(ashape, 2), adt, a_ptr,
                      (pointwise ? in_pix : kdim) * esz_a, TensorLayout::Dense, qa);

        // The group's own per-channel parameters, sliced so the GEMM indexes
        // from 0.
        const TensorView bm(wg.shape_span(), wg.data_type(),
                            const_cast<void*>(wg.ptr<void>()), wg.pitch(), wg.layout(),
                            group_params(qw, g * gm.OC_per_G, gm.OC_per_G));

        uint8_t* ctile = direct ? out_g + p0 * gm.OC * esz_c : stage;
        const int64_t cshape[] = {nblk, gm.OC_per_G};
        TensorView cm(std::span<const int64_t>(cshape, 2), odt, ctile,
                      (direct ? gm.OC : gm.OC_per_G) * esz_c, TensorLayout::Dense, qc);

        TensorView ins[3] = {am, bm};
        int ni = 2;
        if (has_bias) {
            const int64_t bshape1[] = {gm.OC_per_G};
            ins[2] = TensorView(std::span<const int64_t>(bshape1, 1), inputs[2].data_type(),
                                const_cast<void*>(b_g), gm.OC_per_G * esz_b, TensorLayout::Dense);
            ni = 3;
        }

        MatMulAttributes mm{};
        mm.output_dtype = odt;
        mm.act_quant_dtype = attrs.act_quant_dtype;
        mm.epilogue = attrs.epilogue;
        matmul_cpu(mm, cm, std::span<const TensorView>(ins, static_cast<size_t>(ni)), ictx,
                   nullptr);

        if (direct) {
            return;
        }
        // The GEMM has already dequantized/requantized into the staged tile, so
        // the scatter is a row copy into the padded or offset destination.
        const int64_t row_bytes = gm.OC_per_G * esz_c;
        ictx.cpu.run(0, nblk, [&](int64_t p) {
            const int64_t od = p / (nh * gm.OW);
            const int64_t rem = p % (nh * gm.OW);
            const int64_t oh = rem / gm.OW;
            const int64_t ow = rem % gm.OW;
            uint8_t* dst = out_g + ((od0 + od) * out_d_stride + (oh0 + oh) * out_row_stride +
                                    ow * gm.OC) *
                                       esz_c;
            std::memcpy(dst, stage + p * row_bytes, static_cast<size_t>(row_bytes));
        });
    };

    const int64_t workers = std::max<int64_t>(ctx.cpu.worker_count(), 1);
    const int64_t ng = gm.N * gm.G;
    const int grant = conv3d_grid_override();
    // The GEMM's own block grid, roughly (see the float route): a per-group
    // GEMM narrower than the pool runs as a task instead.
    const int64_t gemm_blocks =
        ((npix + 47) / 48) * std::max<int64_t>(1, (gm.OC_per_G + 31) / 32);
    const bool task_grid = (grant == 1) || (grant != 0 && gemm_blocks < workers);
    const int64_t esz_blk = std::max(esz_a, esz_c);

    const Conv3DBlock blk = conv3d_choose_block(
        gm, kdim, esz_blk,
        task_grid ? std::max<int64_t>(kConv3dColCapBytes / workers, kConv3dMinTaskColBytes)
                  : kConv3dColCapBytes);

    if (!task_grid) {
        const int64_t nblk_max = blk.od_chunk * blk.roi_h * gm.OW;
        PoolArray<uint8_t> col(pointwise ? 0 : static_cast<size_t>(nblk_max * kdim * esz_a));
        PoolArray<uint8_t> stage(direct ? 0 : static_cast<size_t>(nblk_max * gm.OC_per_G * esz_c));
        if ((!pointwise && !col) || (!direct && !stage)) {
            return false;
        }
        for (int64_t n = 0; n < gm.N; ++n) {
            for (int64_t g = 0; g < gm.G; ++g) {
                const TensorView wg = conv3d_weight_group(wgt, gm, g);
                for (int64_t od0 = 0; od0 < gm.OD; od0 += blk.od_chunk) {
                    for (int64_t oh0 = 0; oh0 < gm.OH; oh0 += blk.roi_h) {
                        run_block(n, g, od0, oh0, blk.od_chunk, blk.roi_h, col.data(),
                                  stage.data(), wg, ctx);
                    }
                }
            }
        }
        return true;
    }

    const int64_t blocks_d = (gm.OD + blk.od_chunk - 1) / blk.od_chunk;
    const int64_t blocks_h = (gm.OH + blk.roi_h - 1) / blk.roi_h;
    const int64_t blocks = blocks_d * blocks_h;
    const int64_t tasks = ng * blocks;
    const int64_t slots = std::min<int64_t>(tasks, workers);
    const int64_t nblk_max = blk.od_chunk * blk.roi_h * gm.OW;

    PoolArray<uint8_t> col(pointwise ? 0 : static_cast<size_t>(nblk_max * kdim * esz_a * slots));
    PoolArray<uint8_t> stage(direct ? 0 :
                                           static_cast<size_t>(nblk_max * gm.OC_per_G * esz_c *
                                                               slots));
    if ((!pointwise && !col) || (!direct && !stage)) {
        return false;
    }

    // The group's B is read-only, so every task shares it: the raw filter is
    // addressed in place and a prepacked blob is fixed at load time.
    const ComputeContext serial_ctx{};
    ctx.cpu.run(0, tasks, [&](int64_t t) {
        const int64_t slot = (slots < tasks) ? ctx.cpu.current_thread_id() : t;
        const int64_t bi = t % blocks;
        const int64_t gi = t / blocks;
        const int64_t n = gi / gm.G;
        const int64_t g = gi % gm.G;
        const int64_t od0 = (bi / blocks_h) * blk.od_chunk;
        const int64_t oh0 = (bi % blocks_h) * blk.roi_h;

        run_block(n, g, od0, oh0, blk.od_chunk, blk.roi_h,
                  pointwise ? nullptr : col.data() + slot * nblk_max * kdim * esz_a,
                  direct ? nullptr : stage.data() + slot * nblk_max * gm.OC_per_G * esz_c,
                  conv3d_weight_group(wgt, gm, g), serial_ctx);
    });
    return true;
}

} // anonymous namespace

Status conv3d_i8_cpu(const Conv3DAttributes& attrs, TensorView& output,
                     std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& act = inputs[0];
    const auto& wgt = inputs[1];
    const DataType adt = act.data_type();
    const DataType odt = output.data_type();

    if (wgt.data_type() != DataType::s8) {
        return Status::unsupported;
    }
    if (adt != DataType::u8 && adt != DataType::s8 && adt != DataType::f32 &&
        adt != DataType::f16) {
        return Status::unsupported;
    }
    if (odt != DataType::f32 && odt != DataType::f16 && odt != DataType::s8 &&
        odt != DataType::u8) {
        return Status::unsupported;
    }
    // The integer path has no residual stage.
    if (attrs.add_to) {
        return Status::unsupported;
    }

    const bool fp_act = adt == DataType::f32 || adt == DataType::f16;
    const bool int_out = odt == DataType::s8 || odt == DataType::u8;
    if (fp_act) {
        // One float type end to end: the GEMM quantizes A and dequantizes back.
        if (odt != adt) {
            return Status::unsupported;
        }
        const QuantParams& qa = act.quant_params();
        if (!(qa.is_active() && (qa.granularity == QuantGranularity::PerTensor ||
                                 qa.granularity == QuantGranularity::PerToken))) {
            return Status::invalid_argument;
        }
        if (attrs.act_quant_dtype != DataType::s8 && attrs.act_quant_dtype != DataType::u8) {
            return Status::unsupported;
        }
    }
    if (inputs.size() > 2) {
        const DataType bdt = inputs[2].data_type();
        if (int_out ? bdt != DataType::s32 : bdt != odt) {
            return Status::unsupported;
        }
    }

    ConvChannels gm{};
    fill_conv_axes<3>(gm, attrs, act, output);
    gm.G = attrs.groups;
    if (!conv3d_fill_channels(gm, act, wgt)) {
        return Status::unsupported;
    }
    // The blob must describe this call's K, or MatMul would read a group packed
    // for another depth.
    if (is_packed_conv3d_weight(wgt.layout()) && !conv3d_packed_blob_matches(wgt, gm)) {
        return Status::invalid_argument;
    }

    if (!conv3d_i8_col_gemm(attrs, gm, output, inputs, ctx)) {
        return Status::unsupported;
    }
    return Status::ok;
}

void conv3d_packed_weight_pack_i8(const Conv3DAttributes& attrs, const TensorView& weight,
                                 TensorView& packed, const ComputeContext& ctx)
{
    const int64_t OC = weight.shape(4);
    const int64_t ICG = weight.shape(3);
    const int64_t G = attrs.groups;
    const int64_t OCG = OC / G;
    const int64_t karea = attrs.kernel_size[0] * attrs.kernel_size[1] * attrs.kernel_size[2];
    const int64_t kdim = ICG * karea;
    const Conv3DPackedWeight pg =
        conv3d_packed_weight_geom(karea, ICG, OCG, G, DataType::s8);
    NNOPS_ASSERT(pg.total > 0 && packed.numel() >= pg.total &&
                 "conv3d_packed_weight_pack_i8: the buffer does not match the geometry");

    // The blob is the raw filter with each group's OC/G column run compacted
    // out of the pitch: row k of a group's B is the filter's (tap, ic) row, so
    // one row copy is the whole pack.
    const int8_t* const src = weight.ptr<int8_t>();
    int8_t* const dst = packed.ptr<int8_t>();
    ctx.cpu.run(0, G * kdim, [&](int64_t i) {
        const int64_t g = i / kdim;
        const int64_t k = i % kdim;
        std::memcpy(dst + g * pg.group_elems + k * OCG, src + k * OC + g * OCG,
                    static_cast<size_t>(OCG));
    });
}

} // namespace nnops::backend::cpu
