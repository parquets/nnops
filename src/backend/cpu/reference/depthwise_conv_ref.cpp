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

namespace nnops::backend::cpu::reference {

using nnops::simd::s_load;
using nnops::simd::s_store;

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
                        if (id < 0 || id >= ID) continue;

                        for (int64_t kh = 0; kh < KH; ++kh) {
                            const int64_t ih = static_cast<int64_t>(oh) * SH
                                             + static_cast<int64_t>(kh) * DH - PH;
                            if (ih < 0 || ih >= IH) continue;

                            for (int64_t kw = 0; kw < KW; ++kw) {
                                const int64_t iw = static_cast<int64_t>(ow) * SW
                                                 + static_cast<int64_t>(kw) * DW - PW;
                                if (iw < 0 || iw >= IW) continue;

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
    switch (dtype) {
    case DataType::f32:
        if (srank == 3)
            dwconv_impl_3d_ref<float>(attrs, output, inputs, ctx);
        else
            dwconv_impl_2d_ref<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        if (srank == 3)
            dwconv_impl_3d_ref<half>(attrs, output, inputs, ctx);
        else
            dwconv_impl_2d_ref<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"depthwise_conv_ref: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu::reference
