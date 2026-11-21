/// @file conv3d_int8_ref.cpp
/// @brief Naive CPU reference for the quantized Conv3D route — s8 weight ×
///        u8/s8/f16/f32 activation → f32/f16 or s8/u8 output, NDHWC.
///
/// The correctness baseline for conv3d_implicit_gemm_i8.cpp. Loops over
/// (oc, od, oh, ow, ic, kd, kh, kw) with per-sample parallelism, accumulating
/// (w − zp_w)·(a − zp_a) directly, so the compensation sum is never formed.
///
/// The tail matches the shared int8 output pass expression for expression —
/// for an fp destination the dequantize (acc·s_a·s_b + bias), the clip and the
/// activation; for an integer destination the int32 bias, Relu and the requant
/// by the output's QuantParams — because both the kernel and this file call the
/// same helpers (matmul_int8_requant, apply_epilogue). An fp activation is
/// quantized with quant_kernel's arithmetic: x·(1/scale) + zero, round to
/// nearest-even, clamp to the storage range.

#include "nnops/ops/conv3d.hpp"
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
inline QInt quantize_value(float x, float inv_scale, float zero) noexcept
{
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
void conv3d_int8_ref_impl(const Conv3DAttributes& attrs, TensorView& output,
                          std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    constexpr bool kFpAct = !std::is_same_v<TAct, int8_t> && !std::is_same_v<TAct, uint8_t>;
    constexpr bool kU8 = std::is_same_v<TDst, uint8_t>;
    constexpr bool kIntOut = std::is_same_v<TOut, int8_t> || std::is_same_v<TOut, uint8_t>;

    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    // Activation: [N, ID, IH, IW, IC] (NDHWC).
    const int64_t N = input.shape(0);
    const int64_t ID = input.shape(1);
    const int64_t IH = input.shape(2);
    const int64_t IW = input.shape(3);
    const int64_t IC = input.shape(4);
    const int64_t in_row_stride = input.row_stride_elems();

    // Weight: DHWIO [KD, KH, KW, IC/G, OC].
    const int64_t OC = weight.shape(4);
    const int64_t KC = weight.shape(3);
    const int64_t KD = attrs.kernel_size[0];
    const int64_t KH = attrs.kernel_size[1];
    const int64_t KW = attrs.kernel_size[2];

    // Output: [N, OD, OH, OW, OC].
    const int64_t OD = output.shape(1);
    const int64_t OH = output.shape(2);
    const int64_t OW = output.shape(3);
    const int64_t out_row_stride = output.row_stride_elems();

    const int64_t G = attrs.groups;
    const int64_t OC_per_G = OC / G;
    const int64_t SD = attrs.stride[0], SH = attrs.stride[1], SW = attrs.stride[2];
    const int64_t DD = attrs.dilation[0], DH = attrs.dilation[1], DW = attrs.dilation[2];
    const int64_t PD = attrs.padding[0], PH = attrs.padding[1], PW = attrs.padding[2];

    const QuantParams& qa = input.quant_params();
    const QuantParams& qw = weight.quant_params();

    const float s_x = quant_scale_at(qa, 0);
    const int32_t zp_x = quant_zp_at(qa, 0);
    const float inv_x = 1.0f / s_x;

    const auto* in_ptr = input.ptr<TAct>();
    const auto* w_ptr = weight.ptr<int8_t>();
    [[maybe_unused]] const TOut* b_ptr = (has_bias && !kIntOut) ? inputs[2].ptr<TOut>() : nullptr;
    [[maybe_unused]] const int32_t* b_i32 =
        (has_bias && kIntOut) ? inputs[2].ptr<int32_t>() : nullptr;
    auto* out_ptr = output.ptr<TOut>();
    [[maybe_unused]] const QuantParams& qc = output.quant_params();
    [[maybe_unused]] const float out_scale = quant_scale_at(qc, 0);
    [[maybe_unused]] const int32_t zp_out = quant_zp_at(qc, 0);
    NNOPS_ASSERT(!attrs.add_to || !kIntOut);

    const auto compute_sample = [&](int64_t n) {
        for (int64_t g = 0; g < G; ++g) {
            for (int64_t oc = 0; oc < OC_per_G; ++oc) {
                const int64_t oc_global = g * OC_per_G + oc;
                const float s_w = quant_scale_at(qw, oc_global);
                const int32_t zp_w = quant_zp_at(qw, oc_global);
                for (int64_t od = 0; od < OD; ++od) {
                    for (int64_t oh = 0; oh < OH; ++oh) {
                        for (int64_t ow = 0; ow < OW; ++ow) {
                            int32_t acc = 0;
                            const int64_t ic_start = g * KC;
                            const int64_t ic_end = ic_start + KC;
                            for (int64_t ic = ic_start; ic < ic_end; ++ic) {
                                for (int64_t kd = 0; kd < KD; ++kd) {
                                    const int64_t id = od * SD + kd * DD - PD;
                                    if (id < 0 || id >= ID) {
                                        continue;
                                    }
                                    for (int64_t kh = 0; kh < KH; ++kh) {
                                        const int64_t ih = oh * SH + kh * DH - PH;
                                        if (ih < 0 || ih >= IH) {
                                            continue;
                                        }
                                        for (int64_t kw = 0; kw < KW; ++kw) {
                                            const int64_t iw = ow * SW + kw * DW - PW;
                                            if (iw < 0 || iw >= IW) {
                                                continue;
                                            }
                                            const int64_t in_idx =
                                                ((n * ID + id) * IH + ih) * in_row_stride +
                                                iw * IC + ic;
                                            int32_t qa_val;
                                            if constexpr (kFpAct) {
                                                qa_val = static_cast<int32_t>(quantize_value<TDst>(
                                                    s_load(&in_ptr[in_idx]), inv_x,
                                                    static_cast<float>(zp_x)));
                                            } else {
                                                qa_val = static_cast<int32_t>(
                                                    reinterpret_cast<const TDst*>(in_ptr)[in_idx]);
                                            }
                                            const int64_t w_idx =
                                                ((((kd * KH + kh) * KW + kw) * KC +
                                                  (ic - ic_start)) *
                                                     OC +
                                                 oc_global);
                                            const int32_t w_val = static_cast<int32_t>(w_ptr[w_idx]);
                                            acc += (w_val - zp_w) * (qa_val - zp_x);
                                        }
                                    }
                                }
                            }

                            const int64_t out_idx =
                                ((n * OD + od) * OH + oh) * out_row_stride + ow * OC + oc_global;
                            if constexpr (kIntOut) {
                                int32_t v = acc;
                                if (has_bias) {
                                    v += b_i32[oc_global];
                                }
                                if (attrs.epilogue.type == EpilogueActivateType::Relu) {
                                    v = std::max(v, 0);
                                }
                                const float req = matmul_int8_requant_scale(s_w, s_x, out_scale);
                                out_ptr[out_idx] = matmul_int8_requant<TOut>(v, req, zp_out);
                            } else {
                                float f = static_cast<float>(acc) * (s_w * s_x);
                                if (has_bias) {
                                    f += s_load(&b_ptr[oc_global]);
                                }
                                f = std::min(std::max(f, attrs.epilogue.min_clip),
                                             attrs.epilogue.max_clip);
                                f = apply_epilogue(attrs.epilogue, f, oc_global);
                                s_store(&out_ptr[out_idx], f);
                            }
                        }
                    }
                }
            }
        }
    };

    ctx.cpu.run(0, N, compute_sample);
}

template <class TAct, class TDst>
void conv3d_int8_ref_dst(const Conv3DAttributes& attrs, TensorView& output,
                         std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    switch (output.data_type()) {
    case DataType::f16:
        conv3d_int8_ref_impl<TAct, TDst, half>(attrs, output, inputs, ctx);
        break;
    case DataType::s8:
        conv3d_int8_ref_impl<TAct, TDst, int8_t>(attrs, output, inputs, ctx);
        break;
    case DataType::u8:
        conv3d_int8_ref_impl<TAct, TDst, uint8_t>(attrs, output, inputs, ctx);
        break;
    default:
        conv3d_int8_ref_impl<TAct, TDst, float>(attrs, output, inputs, ctx);
        break;
    }
}

template <class TAct>
void conv3d_int8_ref_act(const Conv3DAttributes& attrs, TensorView& output,
                         std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    // Same storage choice as the tiled route: fp activations go to
    // act_quant_dtype, pre-quantized ones are used as stored.
    const DataType act_dt = inputs[0].data_type();
    const bool u8 = (act_dt == DataType::u8) ||
                    ((act_dt == DataType::f32 || act_dt == DataType::f16) &&
                     attrs.act_quant_dtype == DataType::u8);
    if (u8) {
        conv3d_int8_ref_dst<TAct, uint8_t>(attrs, output, inputs, ctx);
    } else {
        conv3d_int8_ref_dst<TAct, int8_t>(attrs, output, inputs, ctx);
    }
}

} // anonymous namespace

void conv3d_int8_ref(const Conv3DAttributes& attrs, TensorView& output,
                     std::span<const TensorView> inputs, const ComputeContext& ctx,
                     void* /*workspace*/)
{
    switch (inputs[0].data_type()) {
    case DataType::f32:
        conv3d_int8_ref_act<float>(attrs, output, inputs, ctx);
        break;
    case DataType::f16:
        conv3d_int8_ref_act<half>(attrs, output, inputs, ctx);
        break;
    case DataType::u8:
        conv3d_int8_ref_act<uint8_t>(attrs, output, inputs, ctx);
        break;
    default:
        conv3d_int8_ref_act<int8_t>(attrs, output, inputs, ctx);
        break;
    }
}

} // namespace nnops::backend::cpu::reference
