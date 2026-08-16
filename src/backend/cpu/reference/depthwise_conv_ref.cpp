/// @file depthwise_conv_ref.cpp
/// @brief Naive CPU reference implementation of 2D/3D depthwise convolution (NCHW/NCDHW layout).
///
/// Spatial rank is auto-detected from input tensor rank (4→2D, 5→3D).
/// Supports both f32 and f16 data types. Computation always done in fp32;
/// dtype-specific load/store handled via overloaded helpers.
/// This is the correctness baseline.

#include "nnops/ops/depthwise_conv.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "../epilogue_impl.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace nnops::backend::cpu::reference {

using nnops::simd::s_load;
using nnops::simd::s_store;

namespace {

/// Quantize a float value to int8/uint8 with round-to-nearest-even and
/// saturation, matching the arch `requantize_8` / `quantization<T>` semantics.
template <typename OutT>
inline OutT quantize_float(float val, float out_scale, float out_zero) {
    constexpr bool Q_U8 = std::is_same_v<OutT, uint8_t>;
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;
    int32_t qi = static_cast<int32_t>(
        std::nearbyintf(val / out_scale + out_zero));
    qi = std::min(std::max(qi, qmin), qmax);
    return static_cast<OutT>(qi);
}

}  // anonymous namespace

// ============================================================
// 2D reference (NCHW)
// ============================================================

template <typename T>
void dwconv_impl_2d_ref(const DepthwiseConvAttributes& attrs,
                          TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t N  = input.shape(0);
    const int64_t C  = input.shape(1);
    const int64_t IH = input.shape(2);
    const int64_t IW = input.shape(3);
    const int64_t in_row_stride  = input.row_stride_elems();

    const int64_t KH = attrs.kernel_size[1];
    const int64_t KW = attrs.kernel_size[2];

    const int64_t OH = output.shape(2);
    const int64_t OW = output.shape(3);
    const int64_t out_row_stride = output.row_stride_elems();

    const int64_t SH = attrs.stride[1];
    const int64_t SW = attrs.stride[2];
    const int64_t DH = attrs.dilation[1];
    const int64_t DW = attrs.dilation[2];
    const int64_t PH = attrs.padding[1];
    const int64_t PW = attrs.padding[2];

    auto* out_ptr = output.ptr<T>();
    const auto* in_ptr  = input.ptr<T>();
    const auto* w_ptr   = weight.ptr<T>();
    const auto* b_ptr   = has_bias ? inputs[2].ptr<T>() : nullptr;

    // Per-channel compute lambda (N*C parallel)
    const auto compute_channel = [&](int64_t n, int64_t c) {
        for (int64_t oh = 0; oh < OH; ++oh) {
            for (int64_t ow = 0; ow < OW; ++ow) {
                float sum = 0.0f;

                for (int64_t kh = 0; kh < KH; ++kh) {
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        const int64_t ih = static_cast<int64_t>(oh) * SH
                                         + static_cast<int64_t>(kh) * DH - PH;
                        const int64_t iw = static_cast<int64_t>(ow) * SW
                                         + static_cast<int64_t>(kw) * DW - PW;

                        if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                            const int64_t in_idx =
                                ((n * C + c) * IH + ih) * in_row_stride + iw;
                            const int64_t w_idx =
                                (c * KH + kh) * KW + kw;
                            sum += s_load(&in_ptr[in_idx]) * s_load(&w_ptr[w_idx]);
                        }
                    }
                }

                if (has_bias) {
                    sum += s_load(&b_ptr[c]);
                }

                const int64_t out_idx =
                    ((n * C + c) * OH + oh) * out_row_stride + ow;
                float val = apply_epilogue(attrs.epilogue, sum, c);
                if (attrs.add_to) {
                    s_store(&out_ptr[out_idx], s_load(&out_ptr[out_idx]) + val);
                } else {
                    s_store(&out_ptr[out_idx], val);
                }
            }
        }
    };

    const int64_t NC = N * C;
    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, NC,
            [&](int64_t tid) {
                int64_t n = tid / C;
                int64_t c = tid % C;
                compute_channel(n, c);
            });
    } else {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                compute_channel(n, c);
            }
        }
    }
}

// ============================================================
// 3D reference (NCDHW)
// ============================================================

template <typename T>
void dwconv_impl_3d_ref(const DepthwiseConvAttributes& attrs,
                          TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t N  = input.shape(0);
    const int64_t C  = input.shape(1);
    const int64_t ID = input.shape(2);
    const int64_t IH = input.shape(3);
    const int64_t IW = input.shape(4);
    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t in_d_stride    = IH * in_row_stride;

    const int64_t KD = attrs.kernel_size[0];
    const int64_t KH = attrs.kernel_size[1];
    const int64_t KW = attrs.kernel_size[2];

    const int64_t OD = output.shape(2);
    const int64_t OH = output.shape(3);
    const int64_t OW = output.shape(4);
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_d_stride   = OH * out_row_stride;

    const int64_t SD = attrs.stride[0];
    const int64_t SH = attrs.stride[1];
    const int64_t SW = attrs.stride[2];
    const int64_t DD = attrs.dilation[0];
    const int64_t DH = attrs.dilation[1];
    const int64_t DW = attrs.dilation[2];
    const int64_t PD = attrs.padding[0];
    const int64_t PH = attrs.padding[1];
    const int64_t PW = attrs.padding[2];

    auto* out_ptr = output.ptr<T>();
    const auto* in_ptr  = input.ptr<T>();
    const auto* w_ptr   = weight.ptr<T>();
    const auto* b_ptr   = has_bias ? inputs[2].ptr<T>() : nullptr;

    // Per-channel compute lambda (N*C parallel)
    const auto compute_channel = [&](int64_t n, int64_t c) {
        for (int64_t od = 0; od < OD; ++od) {
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    float sum = 0.0f;

                    for (int64_t kd = 0; kd < KD; ++kd) {
                        const int64_t id = static_cast<int64_t>(od) * SD
                                         + static_cast<int64_t>(kd) * DD - PD;
                        if (id < 0 || id >= ID) {
                            continue;
                        }

                        for (int64_t kh = 0; kh < KH; ++kh) {
                            const int64_t ih = static_cast<int64_t>(oh) * SH
                                             + static_cast<int64_t>(kh) * DH - PH;
                            if (ih < 0 || ih >= IH) {
                                continue;
                            }

                            for (int64_t kw = 0; kw < KW; ++kw) {
                                const int64_t iw = static_cast<int64_t>(ow) * SW
                                                 + static_cast<int64_t>(kw) * DW - PW;
                                if (iw < 0 || iw >= IW) {
                                    continue;
                                }

                                const int64_t in_idx =
                                    ((n * C + c) * ID + id) * in_d_stride
                                    + ih * in_row_stride + iw;
                                const int64_t w_idx =
                                    (c * KD + kd) * KH * KW + kh * KW + kw;
                                sum += s_load(&in_ptr[in_idx]) * s_load(&w_ptr[w_idx]);
                            }
                        }
                    }

                    if (has_bias) {
                        sum += s_load(&b_ptr[c]);
                    }

                    const int64_t out_idx =
                        ((n * C + c) * OD + od) * out_d_stride
                        + oh * out_row_stride + ow;
                    float val = apply_epilogue(attrs.epilogue, sum, c);
                    if (attrs.add_to) {
                        s_store(&out_ptr[out_idx], s_load(&out_ptr[out_idx]) + val);
                    } else {
                        s_store(&out_ptr[out_idx], val);
                    }
                }
            }
        }
    };

    const int64_t NC = N * C;
    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, NC,
            [&](int64_t tid) {
                int64_t n = tid / C;
                int64_t c = tid % C;
                compute_channel(n, c);
            });
    } else {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                compute_channel(n, c);
            }
        }
    }
}

// ============================================================
// Quantized reference (s8/u8 input/output, f32 bias)
// ============================================================
//
// Dequantize-on-load (activation PerTensor, weight PerChannel), accumulate in
// float, epilogue in float, requantize-on-store with round-to-nearest-even.
// Mirrors the SIMD backend's float-accumulate strategy for exact cross-check.

template <typename InT, typename OutT>
void dwconv_impl_2d_ref_quant(const DepthwiseConvAttributes& attrs,
                          TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t N  = input.shape(0);
    const int64_t C  = input.shape(1);
    const int64_t IH = input.shape(2);
    const int64_t IW = input.shape(3);
    const int64_t in_row_stride  = input.row_stride_elems();

    const int64_t KH = attrs.kernel_size[1];
    const int64_t KW = attrs.kernel_size[2];

    const int64_t OH = output.shape(2);
    const int64_t OW = output.shape(3);
    const int64_t out_row_stride = output.row_stride_elems();

    const int64_t SH = attrs.stride[1];
    const int64_t SW = attrs.stride[2];
    const int64_t DH = attrs.dilation[1];
    const int64_t DW = attrs.dilation[2];
    const int64_t PH = attrs.padding[1];
    const int64_t PW = attrs.padding[2];

    const auto& in_qp  = input.quant_params();
    const auto& out_qp = output.quant_params();
    const auto& w_qp   = weight.quant_params();
    NNOPS_ASSERT(w_qp.granularity == QuantGranularity::PerChannel);
    const float in_scale  = in_qp.scale;
    const float in_zero   = static_cast<float>(in_qp.zero_point);
    const float out_scale = out_qp.scale;
    const float out_zero  = static_cast<float>(out_qp.zero_point);

    const bool w_has_zp = (w_qp.zero_point_data != nullptr);

    auto* out_ptr = output.ptr<OutT>();
    const auto* in_ptr  = input.ptr<InT>();
    const auto* w_ptr   = weight.ptr<InT>();
    const auto* b_ptr   = has_bias ? inputs[2].ptr<float>() : nullptr;

    const auto compute_channel = [&](int64_t n, int64_t c) {
        const float w_scale = w_qp.scale_data[c];
        const float w_zero  = w_has_zp
            ? static_cast<float>(w_qp.zero_point_data[c])
            : 0.0f;

        for (int64_t oh = 0; oh < OH; ++oh) {
            for (int64_t ow = 0; ow < OW; ++ow) {
                float sum = 0.0f;

                for (int64_t kh = 0; kh < KH; ++kh) {
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        const int64_t ih = oh * SH + kh * DH - PH;
                        const int64_t iw = ow * SW + kw * DW - PW;
                        if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                            const int64_t in_idx =
                                ((n * C + c) * IH + ih) * in_row_stride + iw;
                            const int64_t w_idx = (c * KH + kh) * KW + kw;
                            const float x =
                                (static_cast<float>(in_ptr[in_idx]) - in_zero) * in_scale;
                            const float wv =
                                (static_cast<float>(w_ptr[w_idx]) - w_zero) * w_scale;
                            sum += x * wv;
                        }
                    }
                }

                if (has_bias) {
                    sum += b_ptr[c];
                }

                const int64_t out_idx =
                    ((n * C + c) * OH + oh) * out_row_stride + ow;
                float val = apply_epilogue(attrs.epilogue, sum, c);
                if (attrs.add_to) {
                    const float existing =
                        (static_cast<float>(out_ptr[out_idx]) - out_zero) * out_scale;
                    val += existing;
                }
                out_ptr[out_idx] = quantize_float<OutT>(val, out_scale, out_zero);
            }
        }
    };

    const int64_t NC = N * C;
    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, NC,
            [&](int64_t tid) {
                int64_t n = tid / C;
                int64_t c = tid % C;
                compute_channel(n, c);
            });
    } else {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                compute_channel(n, c);
            }
        }
    }
}

template <typename InT, typename OutT>
void dwconv_impl_3d_ref_quant(const DepthwiseConvAttributes& attrs,
                          TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t N  = input.shape(0);
    const int64_t C  = input.shape(1);
    const int64_t ID = input.shape(2);
    const int64_t IH = input.shape(3);
    const int64_t IW = input.shape(4);
    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t in_d_stride    = IH * in_row_stride;

    const int64_t KD = attrs.kernel_size[0];
    const int64_t KH = attrs.kernel_size[1];
    const int64_t KW = attrs.kernel_size[2];

    const int64_t OD = output.shape(2);
    const int64_t OH = output.shape(3);
    const int64_t OW = output.shape(4);
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_d_stride   = OH * out_row_stride;

    const int64_t SD = attrs.stride[0];
    const int64_t SH = attrs.stride[1];
    const int64_t SW = attrs.stride[2];
    const int64_t DD = attrs.dilation[0];
    const int64_t DH = attrs.dilation[1];
    const int64_t DW = attrs.dilation[2];
    const int64_t PD = attrs.padding[0];
    const int64_t PH = attrs.padding[1];
    const int64_t PW = attrs.padding[2];

    const auto& in_qp  = input.quant_params();
    const auto& out_qp = output.quant_params();
    const auto& w_qp   = weight.quant_params();
    NNOPS_ASSERT(w_qp.granularity == QuantGranularity::PerChannel);
    const float in_scale  = in_qp.scale;
    const float in_zero   = static_cast<float>(in_qp.zero_point);
    const float out_scale = out_qp.scale;
    const float out_zero  = static_cast<float>(out_qp.zero_point);

    const bool w_has_zp = (w_qp.zero_point_data != nullptr);

    auto* out_ptr = output.ptr<OutT>();
    const auto* in_ptr  = input.ptr<InT>();
    const auto* w_ptr   = weight.ptr<InT>();
    const auto* b_ptr   = has_bias ? inputs[2].ptr<float>() : nullptr;

    const auto compute_channel = [&](int64_t n, int64_t c) {
        const float w_scale = w_qp.scale_data[c];
        const float w_zero  = w_has_zp
            ? static_cast<float>(w_qp.zero_point_data[c])
            : 0.0f;

        for (int64_t od = 0; od < OD; ++od) {
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    float sum = 0.0f;

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
                                    ((n * C + c) * ID + id) * in_d_stride
                                    + ih * in_row_stride + iw;
                                const int64_t w_idx =
                                    (c * KD + kd) * KH * KW + kh * KW + kw;
                                const float x =
                                    (static_cast<float>(in_ptr[in_idx]) - in_zero) * in_scale;
                                const float wv =
                                    (static_cast<float>(w_ptr[w_idx]) - w_zero) * w_scale;
                                sum += x * wv;
                            }
                        }
                    }

                    if (has_bias) {
                        sum += b_ptr[c];
                    }

                    const int64_t out_idx =
                        ((n * C + c) * OD + od) * out_d_stride
                        + oh * out_row_stride + ow;
                    float val = apply_epilogue(attrs.epilogue, sum, c);
                    if (attrs.add_to) {
                        const float existing =
                            (static_cast<float>(out_ptr[out_idx]) - out_zero) * out_scale;
                        val += existing;
                    }
                    out_ptr[out_idx] = quantize_float<OutT>(val, out_scale, out_zero);
                }
            }
        }
    };

    const int64_t NC = N * C;
    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, NC,
            [&](int64_t tid) {
                int64_t n = tid / C;
                int64_t c = tid % C;
                compute_channel(n, c);
            });
    } else {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                compute_channel(n, c);
            }
        }
    }
}

// ============================================================
// Unified entry point
// ============================================================

void depthwise_conv_ref(const DepthwiseConvAttributes& attrs,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         void* /*workspace*/)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();
    const int64_t srank = DepthwiseConvAttributes::spatial_rank(rank);

    const auto dtype = input.data_type();
    const auto odtype = output.data_type();
    switch (dtype) {
    case DataType::f32:
        if (srank == 3) {
            dwconv_impl_3d_ref<float>(attrs, output, inputs, ctx);
        }
        else {
            dwconv_impl_2d_ref<float>(attrs, output, inputs, ctx);
        }
        return;
    case DataType::f16:
        if (srank == 3) {
            dwconv_impl_3d_ref<half>(attrs, output, inputs, ctx);
        }
        else {
            dwconv_impl_2d_ref<half>(attrs, output, inputs, ctx);
        }
        return;
    case DataType::s8:
        if (odtype == DataType::s8) {
            if (srank == 3) {
                dwconv_impl_3d_ref_quant<int8_t, int8_t>(attrs, output, inputs, ctx);
            } else {
                dwconv_impl_2d_ref_quant<int8_t, int8_t>(attrs, output, inputs, ctx);
            }
        } else if (odtype == DataType::u8) {
            if (srank == 3) {
                dwconv_impl_3d_ref_quant<int8_t, uint8_t>(attrs, output, inputs, ctx);
            } else {
                dwconv_impl_2d_ref_quant<int8_t, uint8_t>(attrs, output, inputs, ctx);
            }
        } else {
            NNOPS_ASSERT(!"depthwise_conv_ref: s8 input requires s8/u8 output");
        }
        return;
    case DataType::u8:
        if (odtype == DataType::s8) {
            if (srank == 3) {
                dwconv_impl_3d_ref_quant<uint8_t, int8_t>(attrs, output, inputs, ctx);
            } else {
                dwconv_impl_2d_ref_quant<uint8_t, int8_t>(attrs, output, inputs, ctx);
            }
        } else if (odtype == DataType::u8) {
            if (srank == 3) {
                dwconv_impl_3d_ref_quant<uint8_t, uint8_t>(attrs, output, inputs, ctx);
            } else {
                dwconv_impl_2d_ref_quant<uint8_t, uint8_t>(attrs, output, inputs, ctx);
            }
        } else {
            NNOPS_ASSERT(!"depthwise_conv_ref: u8 input requires s8/u8 output");
        }
        return;
    default:
        NNOPS_ASSERT(!"depthwise_conv_ref: unsupported data type (only f32, f16, s8, u8)");
    }
}

}  // namespace nnops::backend::cpu::reference
