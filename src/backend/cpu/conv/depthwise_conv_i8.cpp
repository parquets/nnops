/// @file depthwise_conv_i8.cpp
/// @brief CPU depthwise 2D/3D convolution on NHWC/NDHWC — the quantized route.
///
/// s8/u8 activation against an s8/u8 per-channel weight, the destination picking
/// the requantize. The task grid is the fp route's — a strip of four rows cut to
/// fit L2, the filter axis splitting a strip into channel chunks when the grid
/// cannot fill the pool — and the raw HWCM filter is dequantized per row per
/// block into a staging vector (see kernel::dwconv_block_quant).
///
/// The fp route and the entry point's dtype dispatch are depthwise_conv.cpp.

#include "depthwise_conv_internal.h"
#include "conv_util.h"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "backend/cpu/simd_kernel/simd_depthwise_conv.hpp"

#include <algorithm>
#include <cstdint>

namespace nnops::backend::cpu {

using namespace nnops::simd;
namespace k = nnops::kernel;

namespace {

// ============================================================
// Spatial depthwise, quantized (s8/u8 activation, s8/u8 weight)
// ============================================================
//
// Same task grid as the fp route; the HWCM weight is dequantized per row per
// block into a staging vector (see dwconv_block_quant). The activation is
// PerTensor and the weight PerChannel.

template <typename InT, typename OutT>
Status dwconv_quant_impl(const DepthwiseConvAttributes& attrs, TensorView& output,
                                      std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;
    NNOPS_ASSERT(!has_bias || inputs[2].data_type() == DataType::f32);
    // At most one of the two: the pre-activation order folds the destination
    // into the accumulator, so nothing is left for the post-epilogue add.
    const bool pre_act = attrs.add_to && attrs.residual_pre_activation;
    const bool add_to = attrs.add_to && !pre_act;

    const int64_t srank = DepthwiseConvAttributes::spatial_rank(input.rank());

    const int64_t N = input.shape(0);
    const int64_t C = input.shape(input.rank() - 1);
    const int64_t nblocks = (C + 7) / 8;

    const int64_t OD = (srank == 3) ? output.shape(1) : 1;
    const int64_t OH = output.shape(srank - 1);
    const int64_t OW = output.shape(srank);

    if (attrs.kernel_size[0] * attrs.kernel_size[1] * attrs.kernel_size[2] > k::kMaxQuantTaps) {
        return Status::unsupported; // the staging buffer is sized for 9x9x9
    }

    const auto& in_qp = input.quant_params();
    const auto& out_qp = output.quant_params();
    const auto& w_qp = weight.quant_params();
    if (w_qp.granularity != QuantGranularity::PerChannel || w_qp.scale_data == nullptr) {
        return Status::invalid_argument;
    }
    const float in_scale = in_qp.scale;
    const float in_zero = static_cast<float>(in_qp.zero_point);
    const float out_scale = out_qp.scale;
    const float out_zero = static_cast<float>(out_qp.zero_point);

    const k::DwConvParams p = dwconv_params(attrs, input, output);

    auto* out_ptr = output.ptr<OutT>();
    const auto* in_ptr = input.ptr<InT>();
    const auto* w_ptr = weight.ptr<InT>();
    const auto* b_ptr = has_bias ? inputs[2].ptr<float>() : nullptr;

    const int64_t out_d_stride = OH * p.out_row_stride;

    const int64_t hgroups = (OH + 3) / 4;
    const int64_t in_rows = 4 * p.SH + (p.KH - 1) * p.DH; // rows a group reads
    const int64_t strip = conv_strip_cols(in_rows, C, static_cast<int64_t>(sizeof(OutT)), OW,
                                          p.KW, p.DW, p.SW);
    const int64_t strips = (OW + strip - 1) / strip;
    const int64_t groups = N * OD * hgroups * strips;
    const int64_t cchunks = conv_channel_chunks(groups, nblocks, ctx.cpu.thread_count());
    const int64_t blocks_per = (nblocks + cchunks - 1) / cchunks;

    ctx.cpu.run(0, groups * cchunks, [&](int64_t tid) {
        const ConvGridPos gp = conv_grid_pos(tid, strips, hgroups, OD, cchunks);

        k::DwConvParams pc = p;
        pc.od = gp.od;
        pc.oh = gp.hg * 4;
        pc.ow0 = gp.st * strip;
        pc.ow_end = std::min(pc.ow0 + strip, OW);

        const InT* in_img = in_ptr + gp.n * p.ID * p.in_d_stride;
        OutT* out_rows = out_ptr + gp.n * OD * out_d_stride + gp.od * out_d_stride +
                         pc.oh * p.out_row_stride;

        const int64_t rows_here = std::min<int64_t>(4, OH - pc.oh);
        const int64_t blk0 = gp.cc * blocks_per;
        const int64_t blk1 = std::min(blk0 + blocks_per, nblocks);
        for (int64_t blk = blk0; blk < blk1; ++blk) {
            const int64_t c0 = blk * 8;
            const int64_t lanes = std::min<int64_t>(8, C - c0);
            const InT* in_blk = in_img + c0;
            const InT* w_blk = w_ptr + c0;
            const float* b_blk = b_ptr ? b_ptr + c0 : nullptr;
            const float* ws = w_qp.scale_data + c0;
            const int32_t* wz = w_qp.zero_point_data ? w_qp.zero_point_data + c0 : nullptr;
            // One call at the short group's own height, as in the f32 driver.
            if (rows_here == 4) {
                k::dwconv_block_quant<4, InT, OutT>(
                    out_rows + c0, in_blk, w_blk, ws, wz, b_blk, pc, attrs.epilogue, add_to,
                    pre_act, lanes, in_scale, in_zero, out_scale, out_zero);
            } else if (rows_here == 3) {
                k::dwconv_block_quant<3, InT, OutT>(
                    out_rows + c0, in_blk, w_blk, ws, wz, b_blk, pc, attrs.epilogue, add_to,
                    pre_act, lanes, in_scale, in_zero, out_scale, out_zero);
            } else if (rows_here == 2) {
                k::dwconv_block_quant<2, InT, OutT>(
                    out_rows + c0, in_blk, w_blk, ws, wz, b_blk, pc, attrs.epilogue, add_to,
                    pre_act, lanes, in_scale, in_zero, out_scale, out_zero);
            } else {
                k::dwconv_block_quant<1, InT, OutT>(
                    out_rows + c0, in_blk, w_blk, ws, wz, b_blk, pc, attrs.epilogue, add_to,
                    pre_act, lanes, in_scale, in_zero, out_scale, out_zero);
            }
        }
    });
    return Status::ok;
}

} // anonymous namespace

/// The quantized ladder: the activation's dtype, then the destination's.
Status dwconv_quant_run(const DepthwiseConvAttributes& attrs, TensorView& output,
                        std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const DataType in = inputs[0].data_type();
    const DataType out = output.data_type();
    if (in == DataType::s8) {
        return out == DataType::s8 ? dwconv_quant_impl<int8_t, int8_t>(attrs, output, inputs, ctx) :
                                     dwconv_quant_impl<int8_t, uint8_t>(attrs, output, inputs, ctx);
    }
    if (in == DataType::u8) {
        return out == DataType::s8 ? dwconv_quant_impl<uint8_t, int8_t>(attrs, output, inputs, ctx) :
                                     dwconv_quant_impl<uint8_t, uint8_t>(attrs, output, inputs, ctx);
    }
    return Status::unsupported;
}

} // namespace nnops::backend::cpu
