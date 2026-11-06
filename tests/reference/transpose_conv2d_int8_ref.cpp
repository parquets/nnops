/// @file transpose_conv2d_int8_ref.cpp
/// @brief Naive CPU reference for the quantized TransposeConv2D route — s8
///        weight × u8/s8/f16/f32 activation → f32/f16 or s8/u8 output
///        (channel-first).
///
/// The correctness baseline for the quantized routes in the CPU backend. Gather
/// form, matching the forward route's output-driven indexing: an output pixel
/// (oh, ow) reads input (ih, iw) where oh + PH - kh·DH == ih·SH and likewise
/// for w, so no scatter and no save/restore of the destination. Loops over
/// (g, oc, oh, ow, ic, kh, kw) with per-sample parallelism.
///
/// Integer sums are order-independent and exact over these magnitudes, so the
/// reference's ic-outer order and the kernel's GEMM order agree bit for bit.
/// Every step after the dot product matches the kernel's expression for
/// expression: for an fp destination the dequantize, bias-then-clip order and
/// activation; for an integer destination the int32 bias, Relu and requantize by
/// the output's QuantParams. Activation quantization reproduces
/// quant_kernel::quantization<T>: x·(1/scale) + zero through fmaf, round to
/// nearest-even, clamp to the storage range.
///
/// The weight carries per-channel QuantParams; the activation and the output
/// carry per-tensor ones.

#include "nnops/ops/transpose_conv2d.hpp"
#include "nnops/core/parallel_for.hpp"
#include "simd_kernel/simd_epilogue.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace nnops::backend::cpu::reference {

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
void transpose_conv2d_int8_ref_impl(const TransposeConv2DAttributes& attrs,
                                    TensorView& output,
                                    std::span<const TensorView> inputs,
                                    const ComputeContext& ctx)
{
    constexpr bool kFpAct = !std::is_same_v<TAct, int8_t> && !std::is_same_v<TAct, uint8_t>;
    constexpr bool kIntOut = std::is_same_v<TOut, int8_t> || std::is_same_v<TOut, uint8_t>;

    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    // Input: [N, IC, IH, IW]; the channel-last plane is handled by the caller,
    // this reference is channel-first.
    const int64_t N  = input.shape(0);
    const int64_t IC = input.shape(1);
    const int64_t IH = input.shape(2);
    const int64_t IW = input.shape(3);
    const int64_t in_row_stride = input.row_stride_elems();

    // Weight: [IC, OC/G, KH, KW]
    const int64_t OC_per_G = weight.shape(1);

    // Output: [N, OC, OH, OW]
    const int64_t OC = output.shape(1);
    const int64_t OH = output.shape(2);
    const int64_t OW = output.shape(3);
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_n_stride   = output.stride_elems(0);

    const int64_t G = attrs.groups;
    const int64_t IC_per_G = IC / G;

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
    [[maybe_unused]] const QuantParams& qc = output.quant_params();
    [[maybe_unused]] const float out_scale = quant_scale_at(qc, 0);
    [[maybe_unused]] const int32_t zp_out = quant_zp_at(qc, 0);

    const auto compute_sample = [&](int64_t n) {
        for (int64_t oc = 0; oc < OC; ++oc) {
            const int64_t g = oc / OC_per_G;
            const int64_t oc_l = oc % OC_per_G;
            const float s_w = quant_scale_at(qw, oc);
            const int32_t zp_w = quant_zp_at(qw, oc);
            const int64_t ic_start = g * IC_per_G;

            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    int32_t acc = 0;
                    for (int64_t ic_l = 0; ic_l < IC_per_G; ++ic_l) {
                        const int64_t ic = ic_start + ic_l;
                        for (int64_t kh = 0; kh < KH; ++kh) {
                            const int64_t nh = oh + PH - kh * DH;
                            if (nh < 0 || nh % SH != 0) { continue; }
                            const int64_t ih = nh / SH;
                            if (ih >= IH) { continue; }
                            for (int64_t kw = 0; kw < KW; ++kw) {
                                const int64_t nw = ow + PW - kw * DW;
                                if (nw < 0 || nw % SW != 0) { continue; }
                                const int64_t iw = nw / SW;
                                if (iw >= IW) { continue; }
                                const int64_t in_idx =
                                    ((n * IC + ic) * IH + ih) * in_row_stride + iw;
                                int32_t qa_val;
                                if constexpr (kFpAct) {
                                    qa_val = static_cast<int32_t>(quantize_value<TDst>(
                                        s_load(&in_ptr[in_idx]), inv_x,
                                        static_cast<float>(zp_x)));
                                } else {
                                    qa_val = static_cast<int32_t>(
                                        in_ptr[in_idx]);
                                }
                                const int64_t w_idx =
                                    ((ic * OC_per_G + oc_l) * KH + kh) * KW + kw;
                                const int32_t w_val = static_cast<int32_t>(w_ptr[w_idx]);
                                acc += (w_val - zp_w) * (qa_val - zp_x);
                            }
                        }
                    }

                    const int64_t out_idx =
                        n * out_n_stride + (oc * OH + oh) * out_row_stride + ow;
                    if constexpr (kIntOut) {
                        int32_t v = acc;
                        if (has_bias) { v += b_i32[oc]; }
                        if (attrs.epilogue.type == EpilogueActivateType::Relu) {
                            v = std::max(v, 0);
                        }
                        const float req = matmul_int8_requant_scale(s_w, s_x, out_scale);
                        out_ptr[out_idx] = matmul_int8_requant<TOut>(v, req, zp_out);
                    } else {
                        float f = static_cast<float>(acc) * (s_w * s_x);
                        if (has_bias) { f += s_load(&b_ptr[oc]); }
                        f = std::min(std::max(f, attrs.epilogue.min_clip),
                                     attrs.epilogue.max_clip);
                        f = apply_epilogue(attrs.epilogue, f, oc);
                        s_store(&out_ptr[out_idx], f);
                    }
                }
            }
        }
    };

    ctx.cpu.run(0, N, [&](int64_t n) { compute_sample(n); });
}

template <class TAct, class TDst>
void transpose_conv2d_int8_ref_dst(const TransposeConv2DAttributes& attrs,
                                   TensorView& output,
                                   std::span<const TensorView> inputs,
                                   const ComputeContext& ctx)
{
    switch (output.data_type()) {
    case DataType::f16:
        transpose_conv2d_int8_ref_impl<TAct, TDst, half>(attrs, output, inputs, ctx);
        break;
    case DataType::s8:
        transpose_conv2d_int8_ref_impl<TAct, TDst, int8_t>(attrs, output, inputs, ctx);
        break;
    case DataType::u8:
        transpose_conv2d_int8_ref_impl<TAct, TDst, uint8_t>(attrs, output, inputs, ctx);
        break;
    default:
        transpose_conv2d_int8_ref_impl<TAct, TDst, float>(attrs, output, inputs, ctx);
        break;
    }
}

template <class TAct>
void transpose_conv2d_int8_ref_act(const TransposeConv2DAttributes& attrs,
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
        transpose_conv2d_int8_ref_dst<TAct, uint8_t>(attrs, output, inputs, ctx);
    } else {
        transpose_conv2d_int8_ref_dst<TAct, int8_t>(attrs, output, inputs, ctx);
    }
}

}  // anonymous namespace

void transpose_conv2d_int8_ref(const TransposeConv2DAttributes& attrs,
                               TensorView& output,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* /*workspace*/)
{
    switch (inputs[0].data_type()) {
    case DataType::u8:  transpose_conv2d_int8_ref_act<uint8_t>(attrs, output, inputs, ctx); break;
    case DataType::s8:  transpose_conv2d_int8_ref_act<int8_t>(attrs, output, inputs, ctx);  break;
    case DataType::f16: transpose_conv2d_int8_ref_act<half>(attrs, output, inputs, ctx);    break;
    default:            transpose_conv2d_int8_ref_act<float>(attrs, output, inputs, ctx);   break;
    }
}

}  // namespace nnops::backend::cpu::reference
