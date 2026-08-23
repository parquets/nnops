/// @file depthwise_conv.cpp
/// @brief SIMD-optimized CPU implementation of 2D/3D depthwise convolution (NCHWC8/NCDHWC8 only).
///
/// Input/output are always channel-packed (NCHWC8 or NCDHWC8). Weights are prepacked
/// by the operator dispatch layer into dense [C8, (KD,) KH, KW, 8] format before
/// reaching this backend.
///
/// Spatial rank is auto-detected from input tensor rank (4→2D, 5→3D).
/// A single unified kernel handles both 2D and 3D: the KD loop is always present;
/// for 2D, KD=1 so it degenerates to a single no-op iteration.
///
/// Key design decisions (following the pooling.cpp pattern):
///   1. Single DwConvParams struct with full 3D fields — 2D uses KD=1/OD=1/ID=1.
///   2. Single set of h4/h1 SIMD kernels with built-in KD loop.
///   3. Dispatch has od loop; for 2D, OD=1 so it runs once.
///   4. All C8 blocks use SIMD — pad channels are zero.
///   5. N*C8 parallel dispatch.
///   6. Weights loaded as vectors (8 channels at once) from prepacked buffer.

#include "nnops/ops/depthwise_conv.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/core/tensor_layout.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "simd_kernel/simd_depthwise_conv.hpp"
#include "simd_kernel/simd_epilogue.hpp"

#include <cstdint>
#include <type_traits>


namespace nnops::backend::cpu {

using namespace nnops::simd;
namespace k = nnops::kernel;

// ============================================================
// Main depthwise conv implementation (unified 2D/3D)
// ============================================================

template <typename T>
void dwconv_impl_nchwc8(
    const DepthwiseConvAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& packed_weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t rank  = input.rank();
    const int64_t srank = DepthwiseConvAttributes::spatial_rank(rank);

    const int64_t N  = input.shape(0);
    const int64_t C8 = input.num_channel_blocks();

    // Spatial dims (auto-detect 2D vs 3D)
    const int64_t ID  = (srank == 3) ? input.shape(2) : 1;
    const int64_t IH  = input.shape(srank);
    const int64_t IW  = input.shape(srank + 1);

    const int64_t KD  = attrs.kernel_size[0];
    const int64_t KH  = attrs.kernel_size[1];
    const int64_t KW  = attrs.kernel_size[2];

    const int64_t OD  = (srank == 3) ? output.shape(2) : 1;
    const int64_t OH  = output.shape(srank);
    const int64_t OW  = output.shape(srank + 1);

    const int64_t SD  = attrs.stride[0];
    const int64_t SH  = attrs.stride[1];
    const int64_t SW  = attrs.stride[2];
    const int64_t DD  = attrs.dilation[0];
    const int64_t DH  = attrs.dilation[1];
    const int64_t DW  = attrs.dilation[2];
    const int64_t PD  = attrs.padding[0];
    const int64_t PH  = attrs.padding[1];
    const int64_t PW  = attrs.padding[2];

    auto* out_ptr = output.ptr<T>();
    const auto* in_ptr  = input.ptr<T>();
    const auto* pw_ptr  = packed_weight.ptr<T>();
    const auto* pb_ptr  = has_bias ? inputs[2].ptr<T>() : nullptr;

    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t in_d_stride    = IH * in_row_stride;
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_d_stride   = OH * out_row_stride;
    const int64_t in_ch_stride   = input.channel_block_stride_elems();
    const int64_t out_ch_stride  = output.channel_block_stride_elems();

    // Kernel params (od/oh updated per call)
    k::DwConvParams p{};
    p.ID = ID; p.IH = IH; p.IW = IW;
    p.OH = OH; p.OW = OW;
    p.KD = KD; p.KH = KH; p.KW = KW;
    p.SD = SD; p.SH = SH; p.SW = SW;
    p.DD = DD; p.DH = DH; p.DW = DW;
    p.PD = PD; p.PH = PH; p.PW = PW;
    p.out_row_stride = out_row_stride;
    p.in_d_stride    = in_d_stride;
    p.in_row_stride  = in_row_stride;

    // Per-C8-block compute lambda (all C8 blocks use SIMD — pad channels are zero)
    const auto compute_c8 = [&](int64_t n, int64_t c8) {
        const T* in_base  = in_ptr  + n * C8 * in_ch_stride  + c8 * in_ch_stride;
        T*       out_base = out_ptr + n * C8 * out_ch_stride + c8 * out_ch_stride;
        const T* w_base   = pw_ptr + c8 * KD * KH * KW * 8;
        const T* bias_vec = pb_ptr ? pb_ptr + c8 * 8 : nullptr;

        for (int64_t od = 0; od < OD; ++od) {
            T* out_d = out_base + od * out_d_stride;

            int64_t oh = 0;
            for (; oh + 3 < OH; oh += 4) {
                p.od = od;
                p.oh = oh;
                k::dwconv_h4(out_d + oh * p.out_row_stride, in_base, w_base, bias_vec,
                               p, attrs.epilogue, attrs.add_to);
            }
            for (; oh < OH; ++oh) {
                p.od = od;
                p.oh = oh;
                k::dwconv_h1(out_d + oh * p.out_row_stride, in_base, w_base, bias_vec,
                               p, attrs.epilogue, attrs.add_to);
            }
        }
    };

    // Parallel dispatch (N * C8)
    const int64_t N_C8 = N * C8;
    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, N_C8,
            [&](int64_t tid) { compute_c8(tid / C8, tid % C8); });
    } else {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c8 = 0; c8 < C8; ++c8) {
                compute_c8(n, c8);
            }
        }
    }
}

// ============================================================
// Quantized depthwise conv (s8/u8 input/output, f32 bias)
// ============================================================
//
// The activation is PerTensor-quantized (broadcast scale/zero_point); the weight
// is PerChannel-quantized (one scale/zero_point per channel, expanded to the 8
// lanes of each C8 block — pad lanes are neutralized with scale == 0). Bias is
// f32 and prepacked to [C8, 8]. Accumulation is done in float with the same h4/h1
// SIMD kernels as the f32 path, then requantized on store (round-to-nearest-even).
//
// Quant-only constraint: the precomputed dequantized-weight block is a fixed
// 9×9×9×8 fp32 stack buffer, shared between 2D and 3D. The total number of
// kernel taps (KD×KH×KW) must not exceed 9×9×9; larger kernels error out.
// The f32/f16 path has no such limit.

constexpr int64_t kMaxQuantKernelTaps = 9 * 9 * 9;  // 729

template <typename InT, typename OutT>
void dwconv_impl_nchwc8_quant(
    const DepthwiseConvAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& packed_weight = inputs[1];
    const bool has_bias = inputs.size() > 2;
    NNOPS_ASSERT(!has_bias || inputs[2].data_type() == DataType::f32);

    const int64_t rank  = input.rank();
    const int64_t srank = DepthwiseConvAttributes::spatial_rank(rank);

    const int64_t N  = input.shape(0);
    const int64_t C  = input.shape(1);   // logical channels (== weight channels)
    const int64_t C8 = input.num_channel_blocks();

    const int64_t ID  = (srank == 3) ? input.shape(2) : 1;
    const int64_t IH  = input.shape(srank);
    const int64_t IW  = input.shape(srank + 1);

    const int64_t KD  = attrs.kernel_size[0];
    const int64_t KH  = attrs.kernel_size[1];
    const int64_t KW  = attrs.kernel_size[2];

    // Number of kernel taps per C8 block (KD×KH×KW), used to size/precompute the
    // dequantized-weight buffer below.
    const int64_t num_taps = KD * KH * KW;
    if (num_taps > kMaxQuantKernelTaps) {
        NNOPS_ASSERT(!"depthwise_conv quant: kernel taps exceed 9x9x9");
        return;
    }

    const int64_t OD  = (srank == 3) ? output.shape(2) : 1;
    const int64_t OH  = output.shape(srank);
    const int64_t OW  = output.shape(srank + 1);

    const int64_t SD  = attrs.stride[0];
    const int64_t SH  = attrs.stride[1];
    const int64_t SW  = attrs.stride[2];
    const int64_t DD  = attrs.dilation[0];
    const int64_t DH  = attrs.dilation[1];
    const int64_t DW  = attrs.dilation[2];
    const int64_t PD  = attrs.padding[0];
    const int64_t PH  = attrs.padding[1];
    const int64_t PW  = attrs.padding[2];

    // Quantization parameters: activation PerTensor, weight PerChannel.
    // Depthwise weights are always PerChannel-quantized (one scale/zero_point
    // per channel); no other granularity is supported.
    const auto& in_qp  = input.quant_params();
    const auto& out_qp = output.quant_params();
    const auto& w_qp   = packed_weight.quant_params();
    NNOPS_ASSERT(w_qp.granularity == QuantGranularity::PerChannel);
    const float in_scale  = in_qp.scale;
    const float in_zero   = static_cast<float>(in_qp.zero_point);
    const float out_scale = out_qp.scale;
    const float out_zero  = static_cast<float>(out_qp.zero_point);

    const bool w_has_zp = (w_qp.zero_point_data != nullptr);

    auto* out_ptr = output.ptr<OutT>();
    const auto* in_ptr  = input.ptr<InT>();
    const auto* pw_ptr  = packed_weight.ptr<InT>();
    const auto* pb_ptr  = has_bias ? inputs[2].ptr<float>() : nullptr;

    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t in_d_stride    = IH * in_row_stride;
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_d_stride   = OH * out_row_stride;
    const int64_t in_ch_stride   = input.channel_block_stride_elems();
    const int64_t out_ch_stride  = output.channel_block_stride_elems();

    k::DwConvParams p{};
    p.ID = ID; p.IH = IH; p.IW = IW;
    p.OH = OH; p.OW = OW;
    p.KD = KD; p.KH = KH; p.KW = KW;
    p.SD = SD; p.SH = SH; p.SW = SW;
    p.DD = DD; p.DH = DH; p.DW = DW;
    p.PD = PD; p.PH = PH; p.PW = PW;
    p.out_row_stride = out_row_stride;
    p.in_d_stride    = in_d_stride;
    p.in_row_stride  = in_row_stride;

    const auto compute_c8 = [&](int64_t n, int64_t c8) {
        const InT* in_base  = in_ptr  + n * C8 * in_ch_stride  + c8 * in_ch_stride;
        OutT*      out_base = out_ptr + n * C8 * out_ch_stride + c8 * out_ch_stride;
        const InT* w_base   = pw_ptr + c8 * KD * KH * KW * 8;
        const float* bias_vec = pb_ptr ? pb_ptr + c8 * 8 : nullptr;

        // Expand the PerChannel weight scale/zero_point to the 8 lanes of this
        // C8 block; pad lanes (c >= C) are neutralized with scale == 0.
        float w_scale8[8];
        float w_zero8[8];
        const int64_t c_base = c8 * 8;
        for (int lane = 0; lane < 8; ++lane) {
            const int64_t c = c_base + lane;
            if (c < C) {
                w_scale8[lane] = w_qp.scale_data[c];
                w_zero8[lane] = w_has_zp
                    ? static_cast<float>(w_qp.zero_point_data[c])
                    : 0.0f;
            } else {
                w_scale8[lane] = 0.0f;
                w_zero8[lane] = 0.0f;
            }
        }

        // Precompute the dequantized weights for this C8 block once. They depend
        // only on (kd,kh,kw) and the fixed per-lane scale/zero_point — not on the
        // output position — so this hoists dequantize_channel out of the ow/oh
        // loops and turns the inner weight load into a plain v_load.
        alignas(32) float wq[kMaxQuantKernelTaps * 8];
        for (int64_t k = 0; k < num_taps; ++k) {
            v_store(&wq[k * 8], k::dequantize_channel(&w_base[k * 8], w_scale8, w_zero8));
        }

        for (int64_t od = 0; od < OD; ++od) {
            OutT* out_d = out_base + od * out_d_stride;

            int64_t oh = 0;
            for (; oh + 3 < OH; oh += 4) {
                p.od = od;
                p.oh = oh;
                k::dwconv_h4_quant<InT, OutT>(out_d + oh * p.out_row_stride, in_base, wq,
                                           bias_vec, p, attrs.epilogue, attrs.add_to,
                                           in_scale, in_zero,
                                           out_scale, out_zero);
            }
            for (; oh < OH; ++oh) {
                p.od = od;
                p.oh = oh;
                k::dwconv_h1_quant<InT, OutT>(out_d + oh * p.out_row_stride, in_base, wq,
                                           bias_vec, p, attrs.epilogue, attrs.add_to,
                                           in_scale, in_zero,
                                           out_scale, out_zero);
            }
        }
    };

    const int64_t N_C8 = N * C8;
    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, N_C8,
            [&](int64_t tid) { compute_c8(tid / C8, tid % C8); });
    } else {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c8 = 0; c8 < C8; ++c8) {
                compute_c8(n, c8);
            }
        }
    }
}

// ============================================================
// Entry point with dtype dispatch
// ============================================================

void depthwise_conv_cpu(const DepthwiseConvAttributes& attrs,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    const auto odtype = output.data_type();
    switch (dtype) {
    case DataType::f32:
        dwconv_impl_nchwc8<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        dwconv_impl_nchwc8<half>(attrs, output, inputs, ctx);
        return;
    case DataType::s8:
        if (odtype == DataType::s8) {
            dwconv_impl_nchwc8_quant<int8_t, int8_t>(attrs, output, inputs, ctx);
        } else if (odtype == DataType::u8) {
            dwconv_impl_nchwc8_quant<int8_t, uint8_t>(attrs, output, inputs, ctx);
        } else {
            NNOPS_ASSERT(!"depthwise_conv_cpu: s8 input requires s8/u8 output");
        }
        return;
    case DataType::u8:
        if (odtype == DataType::s8) {
            dwconv_impl_nchwc8_quant<uint8_t, int8_t>(attrs, output, inputs, ctx);
        } else if (odtype == DataType::u8) {
            dwconv_impl_nchwc8_quant<uint8_t, uint8_t>(attrs, output, inputs, ctx);
        } else {
            NNOPS_ASSERT(!"depthwise_conv_cpu: u8 input requires s8/u8 output");
        }
        return;
    default:
        NNOPS_ASSERT(!"depthwise_conv_cpu: unsupported data type (only f32, f16, s8, u8)");
    }
}

}  // namespace nnops::backend::cpu
