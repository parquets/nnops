/// @file conv2d_int8_ref.cpp
/// @brief Naive CPU reference for the quantized Conv2D route — s8 weight ×
///        u8/s8/f16/f32 activation → f32/f16 or s8/u8 output (planar NCHW).
///
/// The correctness baseline for the quantized route in conv2d_implicit_gemm.cpp,
/// and the only implementation of `add_to` on it (the tiled route would have to
/// fold the pre-existing output into its accumulator, so it refuses the plan).
/// Loops over (g, oc, oh, ow, ic, kh, kw) with per-sample parallelism.
///
/// Every step after the integer dot product is written to match the tiled
/// kernel's epilogue expression for expression, so the two agree bit for bit:
/// for an fp destination dq_epilogue_row (same dequantize, same bias-then-clip
/// order, same activation), for an integer destination int8_epilogue_row_quant
/// (int32 bias, Relu, requant by the output's QuantParams). The activation
/// quantization reproduces quant_kernel::quantization<T>: x·(1/scale) + zero,
/// round to nearest-even, clamp to the storage range.

#include "nnops/ops/conv2d.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/assert.hpp"
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

/// @p TDst is the integer storage the GEMM would consume: the activation's own
/// type when it arrives quantized, `act_quant_dtype` when it arrives as fp.
template <class TAct, class TDst, class TOut>
void conv2d_int8_ref_impl(const Conv2DAttributes& attrs,
                          TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx)
{
    constexpr bool kFpAct = !std::is_same_v<TAct, int8_t> && !std::is_same_v<TAct, uint8_t>;
    constexpr bool kU8    = std::is_same_v<TDst, uint8_t>;
    /// Integer destination: requantize by the output's QuantParams, int32 bias.
    constexpr bool kIntOut = std::is_same_v<TOut, int8_t> || std::is_same_v<TOut, uint8_t>;

    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    // Input: [N, IC, IH, IW]
    const int64_t N  = input.shape(0);
    const int64_t IC = input.shape(1);
    const int64_t IH = input.shape(2);
    const int64_t IW = input.shape(3);
    const int64_t in_row_stride = input.row_stride_elems();

    // Weight: [OC, IC/G, KH, KW]
    const int64_t OC = weight.shape(0);
    const int64_t KC = weight.shape(1);  // IC / groups

    // Output: [N, OC, OH, OW]
    const int64_t OH = output.shape(2);
    const int64_t OW = output.shape(3);
    const int64_t out_row_stride = output.row_stride_elems();

    const int64_t G = attrs.groups;
    const int64_t OC_per_G = OC / G;

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

    const float   s_x    = quant_scale_at(qa, 0);
    const int32_t zp_x   = quant_zp_at(qa, 0);
    const float   inv_x  = 1.0f / s_x;

    const auto* in_ptr  = input.ptr<TAct>();
    const auto* w_ptr   = weight.ptr<int8_t>();
    [[maybe_unused]] const TOut* b_ptr =
        (has_bias && !kIntOut) ? inputs[2].ptr<TOut>() : nullptr;
    [[maybe_unused]] const int32_t* b_i32 =
        (has_bias && kIntOut) ? inputs[2].ptr<int32_t>() : nullptr;
    auto*       out_ptr = output.ptr<TOut>();
    [[maybe_unused]] const QuantParams& qc = output.quant_params();
    [[maybe_unused]] const float out_scale = quant_scale_at(qc, 0);
    [[maybe_unused]] const int32_t zp_out = quant_zp_at(qc, 0);
    NNOPS_ASSERT(!attrs.add_to || !kIntOut);

    const auto compute_sample = [&](int64_t n) {
        for (int64_t g = 0; g < G; ++g) {
            for (int64_t oc = 0; oc < OC_per_G; ++oc) {
                const int64_t oc_global = g * OC_per_G + oc;
                const float   s_w  = quant_scale_at(qw, oc_global);
                const int32_t zp_w = quant_zp_at(qw, oc_global);

                for (int64_t oh = 0; oh < OH; ++oh) {
                    for (int64_t ow = 0; ow < OW; ++ow) {
                        int32_t acc = 0;
                        const int64_t ic_start = g * KC;
                        const int64_t ic_end   = ic_start + KC;
                        for (int64_t ic = ic_start; ic < ic_end; ++ic) {
                            for (int64_t kh = 0; kh < KH; ++kh) {
                                for (int64_t kw = 0; kw < KW; ++kw) {
                                    const int64_t ih = oh * SH + kh * DH - PH;
                                    const int64_t iw = ow * SW + kw * DW - PW;
                                    if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                        const int64_t in_idx =
                                            ((n * IC + ic) * IH + ih) * in_row_stride + iw;
                                        int32_t qa_val;
                                        if constexpr (kFpAct) {
                                            qa_val = static_cast<int32_t>(
                                                quantize_value<TDst>(
                                                    s_load(&in_ptr[in_idx]), inv_x,
                                                    static_cast<float>(zp_x)));
                                        } else {
                                            // sign- or zero-extended, per TDst
                                            qa_val = static_cast<int32_t>(
                                                reinterpret_cast<const TDst*>(in_ptr)[in_idx]);
                                        }
                                        const int64_t w_idx =
                                            ((oc_global * KC + (ic - ic_start)) * KH + kh) * KW + kw;
                                        const int32_t w_val =
                                            static_cast<int32_t>(w_ptr[w_idx]);
                                        acc += (w_val - zp_w) * (qa_val - zp_x);
                                    }
                                }
                            }
                        }

                        const int64_t out_idx =
                            ((n * OC + oc_global) * OH + oh) * out_row_stride + ow;
                        if constexpr (kIntOut) {
                            // ---- same tail as int8_epilogue_row_quant --------
                            int32_t v = acc;
                            if (has_bias) { v += b_i32[oc_global]; }
                            if (attrs.epilogue.type == EpilogueActivateType::Relu) {
                                v = std::max(v, 0);
                            }
                            const float req = matmul_int8_requant_scale(s_w, s_x, out_scale);
                            out_ptr[out_idx] = matmul_int8_requant<TOut>(v, req, zp_out);
                        } else {
                            // ---- same tail as dq_epilogue_row -----------------
                            float f = static_cast<float>(acc) * (s_w * s_x);
                            if (has_bias) {
                                f += s_load(&b_ptr[oc_global]);
                            }
                            f = std::min(std::max(f, attrs.epilogue.min_clip),
                                         attrs.epilogue.max_clip);
                            f = apply_epilogue(attrs.epilogue, f, oc_global);

                            if (attrs.add_to) {
                                f += s_load(&out_ptr[out_idx]);
                            }
                            s_store(&out_ptr[out_idx], f);
                        }
                    }
                }
            }
        }
    };

    ctx.cpu.run(0, N, [&](int64_t n) { compute_sample(n); });
}

template <class TAct, class TDst>
void conv2d_int8_ref_dst(const Conv2DAttributes& attrs,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx)
{
    switch (output.data_type()) {
    case DataType::f16: conv2d_int8_ref_impl<TAct, TDst, half>(attrs, output, inputs, ctx);    break;
    case DataType::s8:  conv2d_int8_ref_impl<TAct, TDst, int8_t>(attrs, output, inputs, ctx);  break;
    case DataType::u8:  conv2d_int8_ref_impl<TAct, TDst, uint8_t>(attrs, output, inputs, ctx); break;
    default:            conv2d_int8_ref_impl<TAct, TDst, float>(attrs, output, inputs, ctx);   break;
    }
}

template <class TAct>
void conv2d_int8_ref_act(const Conv2DAttributes& attrs,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx)
{
    // Same storage choice as the tiled route: fp activations go to
    // act_quant_dtype, pre-quantized ones are used as stored.
    const DataType act_dt = inputs[0].data_type();
    const bool u8 = (act_dt == DataType::u8)
                 || ((act_dt == DataType::f32 || act_dt == DataType::f16) &&
                     attrs.act_quant_dtype == DataType::u8);
    if (u8) {
        conv2d_int8_ref_dst<TAct, uint8_t>(attrs, output, inputs, ctx);
    } else {
        conv2d_int8_ref_dst<TAct, int8_t>(attrs, output, inputs, ctx);
    }
}

}  // anonymous namespace

void conv2d_int8_ref(const Conv2DAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* /*workspace*/)
{
    const DataType act_dt = inputs[0].data_type();

#define NNOPS_CONV2D_I8_REF_ACT(TAct)                                          \
    conv2d_int8_ref_act<TAct>(attrs, output, inputs, ctx)

    switch (act_dt) {
    case DataType::u8:  NNOPS_CONV2D_I8_REF_ACT(uint8_t); break;
    case DataType::s8:  NNOPS_CONV2D_I8_REF_ACT(int8_t);  break;
    case DataType::f16: NNOPS_CONV2D_I8_REF_ACT(half);    break;
    default:            NNOPS_CONV2D_I8_REF_ACT(float);   break;
    }

#undef NNOPS_CONV2D_I8_REF_ACT
}

}  // namespace nnops::backend::cpu::reference
