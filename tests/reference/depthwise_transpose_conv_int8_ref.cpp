/// @file depthwise_transpose_conv_int8_ref.cpp
/// @brief Naive CPU reference for the quantized DepthwiseTransposeConv2D route
///        — s8 per-channel weight × u8/s8/f16/f32 activation → f32/f16 or s8/u8
///        output (channel-first).
///
/// The correctness baseline for the quantized kernel in both layouts. Gather
/// form, matching the forward route's output-driven indexing: an output pixel
/// (oh, ow) reads input (ih, iw) where oh + PH - kh·DH == ih·SH and likewise
/// for w, so no scatter and no save/restore of the destination. Loops over
/// (c, oh, ow, kh, kw) with one task per (batch, channel).
///
/// Integer sums are order-independent and exact over these magnitudes, so this
/// reference's per-tap order and the kernel's dot-product order agree bit for
/// bit. Every step after the sum matches the kernel's expression for
/// expression: for an fp destination the dequantize, bias-then-clip order and
/// activation; for an integer destination the int32 bias, Relu and requantize
/// by the output's QuantParams. Activation quantization reproduces
/// quant_kernel::quantization<T>: x·(1/scale) + zero through fmaf, round to
/// nearest-even, clamp to the storage range.
///
/// The weight carries per-channel QuantParams keyed by channel; the activation
/// and the output carry per-tensor ones.

#include "nnops/ops/depthwise_transpose_conv.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "simd_kernel/simd_epilogue.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace nnops::backend::cpu::reference {

using nnops::simd::s_load;
using nnops::simd::s_store;

namespace {

/// float → integer storage, the same arithmetic as quant_kernel::quantization.
template <class QInt>
inline QInt quantize_value(float x, float inv_scale, float zero) noexcept {
    constexpr int32_t qmin = std::is_same_v<QInt, uint8_t> ? 0 : -128;
    constexpr int32_t qmax = std::is_same_v<QInt, uint8_t> ? 255 : 127;
    const float v = std::fmaf(x, inv_scale, zero);
    int32_t q = static_cast<int32_t>(std::nearbyintf(v));
    q = std::min(std::max(q, qmin), qmax);
    return static_cast<QInt>(q);
}

/// @p TDst is the integer storage the kernel's plane would hold: the
/// activation's own type when it arrives quantized, `act_quant_dtype` when it
/// arrives as fp.
template <class TAct, class TDst, class TOut>
void depthwise_transpose_conv_int8_ref_impl(
    const DepthwiseTransposeConv2DAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    constexpr bool kFpAct = !std::is_same_v<TAct, int8_t> && !std::is_same_v<TAct, uint8_t>;
    constexpr bool kIntOut = std::is_same_v<TOut, int8_t> || std::is_same_v<TOut, uint8_t>;

    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    // Input: [N, C, IH, IW]; a channel-last call reaches this through the
    // caller's permutation, so only the channel-first storage is read here.
    const int64_t N  = input.shape(0);
    const int64_t C  = input.shape(1);
    const int64_t IH = input.shape(2);
    const int64_t IW = input.shape(3);

    const int64_t OH = output.shape(2);
    const int64_t OW = output.shape(3);

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

    const int64_t KH = attrs.kernel_size[0];
    const int64_t KW = attrs.kernel_size[1];
    const int64_t SH = attrs.stride[0];
    const int64_t SW = attrs.stride[1];
    const int64_t DH = attrs.dilation[0];
    const int64_t DW = attrs.dilation[1];
    const int64_t PH = attrs.padding[0];
    const int64_t PW = attrs.padding[1];

    const QuantParams& qa = input.quant_params();
    const QuantParams& qw = weight.quant_params();

    const float   s_x   = quant_scale_at(qa, 0);
    const int32_t zp_x  = quant_zp_at(qa, 0);
    const float   inv_x = 1.0f / s_x;

    const auto* in_ptr = input.ptr<TAct>();
    const auto* w_ptr  = weight.ptr<int8_t>();
    [[maybe_unused]] const TOut* b_ptr =
        (has_bias && !kIntOut) ? inputs[2].ptr<TOut>() : nullptr;
    [[maybe_unused]] const int32_t* b_i32 =
        (has_bias && kIntOut) ? inputs[2].ptr<int32_t>() : nullptr;
    auto* out_ptr = output.ptr<TOut>();
    [[maybe_unused]] const QuantParams& qo = output.quant_params();
    [[maybe_unused]] const float out_scale = quant_scale_at(qo, 0);
    [[maybe_unused]] const int32_t zp_out = quant_zp_at(qo, 0);

    ctx.cpu.run(0, N * C, [&](int64_t task) {
        const int64_t n = task / C;
        const int64_t c = task % C;

        const auto* in_c = in_ptr + n * in_n + c * in_ch;
        auto* out_c = out_ptr + n * out_n + c * out_ch;
        const auto* w_c = w_ptr + c * w_ch;

        const float s_w = quant_scale_at(qw, c);
        const int32_t zp_w = quant_zp_at(qw, c);

        for (int64_t oh = 0; oh < OH; ++oh) {
            for (int64_t ow = 0; ow < OW; ++ow) {
                int32_t acc = 0;
                for (int64_t kh = 0; kh < KH; ++kh) {
                    const int64_t nh = oh + PH - kh * DH;
                    if (nh < 0 || nh % SH != 0) { continue; }
                    const int64_t ih = nh / SH;
                    if (ih >= IH) { continue; }
                    const auto* xrow = in_c + ih * in_row;
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        const int64_t nw = ow + PW - kw * DW;
                        if (nw < 0 || nw % SW != 0) { continue; }
                        const int64_t iw = nw / SW;
                        if (iw >= IW) { continue; }
                        int32_t qa_val;
                        if constexpr (kFpAct) {
                            qa_val = static_cast<int32_t>(quantize_value<TDst>(
                                s_load(&xrow[iw]), inv_x, static_cast<float>(zp_x)));
                        } else {
                            qa_val = static_cast<int32_t>(xrow[iw]);
                        }
                        const int32_t w_val =
                            static_cast<int32_t>(w_c[kh * w_kh + kw * w_kw]);
                        acc += (w_val - zp_w) * (qa_val - zp_x);
                    }
                }

                const int64_t out_idx = oh * out_row + ow;
                if constexpr (kIntOut) {
                    int32_t v = acc;
                    if (has_bias) { v += b_i32[c]; }
                    if (attrs.epilogue.type == EpilogueActivateType::Relu) {
                        v = std::max(v, 0);
                    }
                    const float req = matmul_int8_requant_scale(s_w, s_x, out_scale);
                    out_c[out_idx] = matmul_int8_requant<TOut>(v, req, zp_out);
                } else {
                    float f = static_cast<float>(acc) * (s_w * s_x);
                    if (has_bias) { f += s_load(&b_ptr[c]); }
                    f = std::min(std::max(f, attrs.epilogue.min_clip),
                                 attrs.epilogue.max_clip);
                    f = apply_epilogue(attrs.epilogue, f, c);
                    s_store(&out_c[out_idx], f);
                }
            }
        }
    });
}

template <class TAct, class TDst>
void depthwise_transpose_conv_int8_ref_dst(
    const DepthwiseTransposeConv2DAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    switch (output.data_type()) {
    case DataType::f16:
        depthwise_transpose_conv_int8_ref_impl<TAct, TDst, half>(attrs, output, inputs, ctx);
        break;
    case DataType::s8:
        depthwise_transpose_conv_int8_ref_impl<TAct, TDst, int8_t>(attrs, output, inputs, ctx);
        break;
    case DataType::u8:
        depthwise_transpose_conv_int8_ref_impl<TAct, TDst, uint8_t>(attrs, output, inputs, ctx);
        break;
    default:
        depthwise_transpose_conv_int8_ref_impl<TAct, TDst, float>(attrs, output, inputs, ctx);
        break;
    }
}

template <class TAct>
void depthwise_transpose_conv_int8_ref_act(
    const DepthwiseTransposeConv2DAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    // Same storage choice as the kernel: an fp activation is quantized to
    // act_quant_dtype, a pre-quantized one is read as stored.
    const DataType act_dt = inputs[0].data_type();
    const bool u8 = (act_dt == DataType::u8)
                 || ((act_dt == DataType::f32 || act_dt == DataType::f16) &&
                     attrs.act_quant_dtype == DataType::u8);
    if (u8) {
        depthwise_transpose_conv_int8_ref_dst<TAct, uint8_t>(attrs, output, inputs, ctx);
    } else {
        depthwise_transpose_conv_int8_ref_dst<TAct, int8_t>(attrs, output, inputs, ctx);
    }
}

}  // anonymous namespace

void depthwise_transpose_conv_int8_ref(const DepthwiseTransposeConv2DAttributes& attrs,
                                       TensorView& output,
                                       std::span<const TensorView> inputs,
                                       const ComputeContext& ctx,
                                       void* /*workspace*/)
{
    switch (inputs[0].data_type()) {
    case DataType::u8:  depthwise_transpose_conv_int8_ref_act<uint8_t>(attrs, output, inputs, ctx); break;
    case DataType::s8:  depthwise_transpose_conv_int8_ref_act<int8_t>(attrs, output, inputs, ctx);  break;
    case DataType::f16: depthwise_transpose_conv_int8_ref_act<half>(attrs, output, inputs, ctx);    break;
    default:            depthwise_transpose_conv_int8_ref_act<float>(attrs, output, inputs, ctx);   break;
    }
}

}  // namespace nnops::backend::cpu::reference
