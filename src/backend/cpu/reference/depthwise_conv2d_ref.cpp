/// @file depthwise_conv2d_ref.cpp
/// @brief Naive CPU reference implementation of 2D depthwise convolution (NCHW layout).
///
/// Depthwise convolution applies a separate filter to each input channel independently.
/// This is the correctness baseline. Uses a straightforward 6-level nested loop
/// with per-sample parallelism via the external parallel_for hook.

#include "nnops/ops/depthwise_conv2d.hpp"
#include "nnops/core/parallel_for.hpp"

namespace nnops::backend::cpu::reference {

void depthwise_conv2d_ref(const DepthwiseConv2DAttributes& attrs,
                           const TensorView& output,
                           std::span<const TensorView> inputs,
                           const ComputeContext& ctx,
                           void* /*workspace*/)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    // Input: [N, C, IH, IW]
    const int64_t N  = input.shape(0);
    const int64_t C  = input.shape(1);
    const int64_t IH = input.shape(2);
    const int64_t IW = input.shape(3);

    // Weight: [C, 1, KH, KW]
    const int64_t KH = attrs.kernel_size[0];
    const int64_t KW = attrs.kernel_size[1];

    // Output: [N, C, OH, OW]
    const int64_t OH = output.shape(2);
    const int64_t OW = output.shape(3);

    const int64_t SH = attrs.stride[0];
    const int64_t SW = attrs.stride[1];
    const int64_t DH = attrs.dilation[0];
    const int64_t DW = attrs.dilation[1];
    const int64_t PH = attrs.padding[0];
    const int64_t PW = attrs.padding[1];

    auto* out_ptr = output.data_as<float>();
    const auto* in_ptr  = input.data_as<float>();
    const auto* w_ptr   = weight.data_as<float>();
    const auto* b_ptr   = has_bias ? inputs[2].data_as<float>() : nullptr;

    // Per-sample compute lambda
    const auto compute_sample = [&](int64_t n) {
        for (int64_t c = 0; c < C; ++c) {
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
                                // input: [N, C, IH, IW]
                                const int64_t in_idx =
                                    ((n * C + c) * IH + ih) * IW + iw;
                                // weight: [C, 1, KH, KW] → per-channel weight
                                const int64_t w_idx =
                                    (c * KH + kh) * KW + kw;
                                sum += in_ptr[in_idx] * w_ptr[w_idx];
                            }
                        }
                    }

                    if (has_bias) {
                        sum += b_ptr[c];
                    }

                    const int64_t out_idx =
                        ((n * C + c) * OH + oh) * OW + ow;
                    float val = apply_epilogue(attrs.epilogue, sum, c);
                    out_ptr[out_idx] = attrs.add_to
                        ? out_ptr[out_idx] + val
                        : val;
                }
            }
        }
    };

    // Parallel dispatch
    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, N,
            [&](int64_t n) { compute_sample(n); });
    } else {
        for (int64_t n = 0; n < N; ++n) {
            compute_sample(n);
        }
    }
}

}  // namespace nnops::backend::cpu::reference
