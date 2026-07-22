/// @file depthwise_conv2d_ref.cpp
/// @brief Naive CPU reference implementation of 2D depthwise convolution (NCHW layout).
///
/// Supports both f32 and f16 data types for all tensors (input, weight, bias, output).
/// Weight and bias dtype match the input dtype. Computation always done in fp32;
/// dtype-specific load/store handled via overloaded helpers.
/// Depthwise convolution applies a separate filter to each input channel independently.
/// This is the correctness baseline. Uses a straightforward 4-level nested loop
/// with per-(sample,channel) parallelism (N*C tasks) via the external parallel_for hook.

#include "nnops/ops/depthwise_conv2d.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"

namespace nnops::backend::cpu::reference {

using nnops::simd::s_load;
using nnops::simd::s_store;

template <typename T>
void dwconv_impl_ref(const DepthwiseConv2DAttributes& attrs,
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

    const int64_t KH = attrs.kernel_size[0];
    const int64_t KW = attrs.kernel_size[1];

    const int64_t OH = output.shape(2);
    const int64_t OW = output.shape(3);
    const int64_t out_row_stride = output.row_stride_elems();

    const int64_t SH = attrs.stride[0];
    const int64_t SW = attrs.stride[1];
    const int64_t DH = attrs.dilation[0];
    const int64_t DW = attrs.dilation[1];
    const int64_t PH = attrs.padding[0];
    const int64_t PW = attrs.padding[1];

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

void depthwise_conv2d_ref(const DepthwiseConv2DAttributes& attrs,
                           TensorView& output,
                           std::span<const TensorView> inputs,
                           const ComputeContext& ctx,
                           void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        dwconv_impl_ref<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        dwconv_impl_ref<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"depthwise_conv2d_ref: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu::reference
