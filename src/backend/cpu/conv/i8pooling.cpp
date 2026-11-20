/// @file i8pooling.cpp
/// @brief Quantized CPU pooling (s8/u8 input and output) on NHWC/NDHWC.
///
/// PerTensor activation: values are dequantized to float, pooled, and requantized
/// on store (round-to-nearest-even). MaxPooling takes a raw-integer fast path
/// when input and output share dtype, scale and zero_point (and !add_to) — then
/// `max` is exact in the integer domain.
///
/// The grid is the fp route's four-row group without the width strip, swept over
/// dequantize/requantize kernels on the same channel lane. The float routes, the
/// Pool1D lift and the entry point are pooling.cpp.

#include "pooling_internal.h"
#include "nnops/core/parallel_for.hpp"
#include "backend/cpu/simd_kernel/simd_pooling.hpp"

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace nnops::backend::cpu {

using namespace nnops::simd;
namespace k = nnops::kernel;

namespace {

// ============================================================
// Quantized pooling (s8/u8 input/output)
// ============================================================
//
// PerTensor activation: values are dequantized to float, pooled, and requantized
// on store (round-to-nearest-even). MaxPooling takes a raw-integer fast path
// when input and output share dtype, scale and zero_point (and !add_to) — then
// `max` is exact in the integer domain.

/// Spatial quantized pooling: the fp four-row grid without the width
/// strip, over dequantize/requantize kernels; full blocks keep the raw-integer
/// max (exact when scales and zero points match).
template <bool IsMax, typename InT, typename OutT>
void pooling_quant_impl(const PoolingAttributes& attrs, TensorView& output,
                                     std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t srank = PoolingAttributes::spatial_rank(input.rank());

    const int64_t N = input.shape(0);
    const int64_t C = input.shape(input.rank() - 1);
    const int64_t nblocks = (C + 7) / 8;

    const int64_t ID = (srank == 3) ? input.shape(1) : 1;
    const int64_t IH = input.shape(srank - 1);
    const int64_t IW = input.shape(srank);
    const int64_t OD = (srank == 3) ? output.shape(1) : 1;
    const int64_t OH = output.shape(srank - 1);
    const int64_t OW = output.shape(srank);

    const auto& in_qp = input.quant_params();
    const auto& out_qp = output.quant_params();
    const float in_scale = in_qp.scale;
    const float in_zero = static_cast<float>(in_qp.zero_point);
    const float out_scale = out_qp.scale;
    const float out_zero = static_cast<float>(out_qp.zero_point);

    k::PoolingKernelParams p{};
    p.ID = ID;
    p.IH = IH;
    p.IW = IW;
    p.OH = OH;
    p.OW = OW;
    p.KD = (srank == 3) ? attrs.kernel_shape[0] : 1;
    p.KH = attrs.kernel_shape[1];
    p.KW = attrs.kernel_shape[2];
    p.SD = (srank == 3) ? attrs.stride[0] : 1;
    p.SH = attrs.stride[1];
    p.SW = attrs.stride[2];
    p.DD = (srank == 3) ? attrs.dilation[0] : 1;
    p.DH = attrs.dilation[1];
    p.DW = attrs.dilation[2];
    p.PD = (srank == 3) ? attrs.padding[0] : 0;
    p.PH = attrs.padding[1];
    p.PW = attrs.padding[2];
    p.out_row_stride = output.row_stride_elems();
    p.in_row_stride = input.row_stride_elems();
    p.in_d_stride = IH * p.in_row_stride;
    p.pix_stride = C;
    p.exclude_pad = attrs.exclude_pad;

    const float avg_scale = 1.0f / static_cast<float>(p.KD * p.KH * p.KW);

    const bool use_fast = IsMax && std::is_same_v<InT, OutT> && !attrs.add_to &&
                          (in_qp.scale == out_qp.scale) && (in_qp.zero_point == out_qp.zero_point);

    auto* out_ptr = output.ptr<OutT>();
    const auto* in_ptr = input.ptr<InT>();

    const int64_t hgroups = (OH + 3) / 4;
    const int64_t groups = N * OD * hgroups;
    const int64_t nt = ctx.cpu.thread_count();
    int64_t cchunks = 1;
    if (groups < nt && nblocks > 1) {
        cchunks = std::min<int64_t>((nt + groups / 2) / groups, nblocks);
    }
    const int64_t blocks_per = (nblocks + cchunks - 1) / cchunks;

    ctx.cpu.run(0, groups * cchunks, [&](int64_t tid) {
        const int64_t cc = tid % cchunks;
        const int64_t g = tid / cchunks;
        const int64_t hg = g % hgroups;
        const int64_t od = (g / hgroups) % OD;
        const int64_t n = g / (hgroups * OD);

        k::PoolingKernelParams pc = p;
        pc.od = od;
        pc.oh = hg * 4;

        const InT* in_img = in_ptr + n * ID * p.in_d_stride;
        OutT* out_row = out_ptr + n * OD * OH * p.out_row_stride + od * OH * p.out_row_stride +
                        pc.oh * p.out_row_stride;

        const int64_t blk0 = cc * blocks_per;
        const int64_t blk1 = std::min(blk0 + blocks_per, nblocks);
        for (int64_t blk = blk0; blk < blk1; ++blk) {
            const int64_t c0 = blk * 8;
            const int64_t lanes = std::min<int64_t>(8, C - c0);
            if constexpr (IsMax) {
                if constexpr (std::is_same_v<InT, OutT>) {
                    if (use_fast && lanes == k::kChannelLanes) {
                        k::maxpool_quant_fast<4>(out_row + c0, in_img + c0, pc,
                                                              out_zero);
                        continue;
                    }
                }
                k::pool_block_quant<true, InT, OutT>(out_row + c0, in_img + c0, pc, 0.0f,
                                                            in_scale, in_zero, out_scale, out_zero,
                                                            attrs.add_to, lanes);
            } else {
                k::pool_block_quant<false, InT, OutT>(
                    out_row + c0, in_img + c0, pc, avg_scale, in_scale, in_zero, out_scale,
                    out_zero, attrs.add_to, lanes);
            }
        }
    });
}

/// Dispatch the one quantized route over max/avg.
template <typename InT, typename OutT>
void pooling_quant(const PoolingAttributes& attrs, TensorView& output,
                   std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    if (attrs.type == PoolingType::Max) {
        pooling_quant_impl<true, InT, OutT>(attrs, output, inputs, ctx);
    } else {
        pooling_quant_impl<false, InT, OutT>(attrs, output, inputs, ctx);
    }
}

} // anonymous namespace

/// The quantized ladder: max/avg, then the (input, output) dtype pair.
void pooling_quant_run(const PoolingAttributes& attrs, TensorView& output,
                       std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const bool in_s8 = inputs[0].data_type() == DataType::s8;
    const bool out_s8 = output.data_type() == DataType::s8;
    if (in_s8) {
        if (out_s8) {
            pooling_quant<int8_t, int8_t>(attrs, output, inputs, ctx);
        } else {
            pooling_quant<int8_t, uint8_t>(attrs, output, inputs, ctx);
        }
    } else {
        if (out_s8) {
            pooling_quant<uint8_t, int8_t>(attrs, output, inputs, ctx);
        } else {
            pooling_quant<uint8_t, uint8_t>(attrs, output, inputs, ctx);
        }
    }
}

} // namespace nnops::backend::cpu
