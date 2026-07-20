/// @file conv2d_ref.cpp
/// @brief Naive CPU reference implementation of 2D convolution (NCHW layout).
///
/// This is the correctness baseline. Uses a straightforward 7-level nested loop
/// with per-sample parallelism via the external parallel_for hook.

#include "nnops/ops/conv2d.hpp"
#include "nnops/core/parallel_for.hpp"

namespace nnops::backend::cpu::reference {

void conv2d_ref(const Conv2DAttributes& attrs,
                const TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* /*workspace*/)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    // Input: [N, IC, IH, IW]
    const int64_t N  = input.shape(0);
    const int64_t IC = input.shape(1);
    const int64_t IH = input.shape(2);
    const int64_t IW = input.shape(3);

    // Weight: [OC, IC/G, KH, KW]
    const int64_t OC = weight.shape(0);
    const int64_t KC = weight.shape(1);  // IC / groups
    const int64_t KH = weight.shape(2);
    const int64_t KW = weight.shape(3);

    // Output: [N, OC, OH, OW]
    const int64_t OH = output.shape(2);
    const int64_t OW = output.shape(3);

    const int64_t G  = attrs.groups;
    const int64_t OC_per_G = OC / G;

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
        for (int64_t g = 0; g < G; ++g) {
            for (int64_t oc = 0; oc < OC_per_G; ++oc) {
                const int64_t oc_global = g * OC_per_G + oc;
                for (int64_t oh = 0; oh < OH; ++oh) {
                    for (int64_t ow = 0; ow < OW; ++ow) {
                        float sum = 0.0f;
                        const int64_t ic_start = g * KC;
                        const int64_t ic_end   = ic_start + KC;
                        for (int64_t ic = ic_start; ic < ic_end; ++ic) {
                            for (int64_t kh = 0; kh < KH; ++kh) {
                                for (int64_t kw = 0; kw < KW; ++kw) {
                                    const int64_t ih = oh * SH + kh * DH - PH;
                                    const int64_t iw = ow * SW + kw * DW - PW;
                                    if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                        const int64_t in_idx =
                                            ((n * IC + ic) * IH + ih) * IW + iw;
                                        const int64_t w_idx =
                                            (((oc_global * KC) + (ic - ic_start))
                                             * KH + kh) * KW + kw;
                                        sum += in_ptr[in_idx] * w_ptr[w_idx];
                                    }
                                }
                            }
                        }
                        if (has_bias) {
                            sum += b_ptr[oc_global];
                        }
                        const int64_t out_idx =
                            ((n * OC + oc_global) * OH + oh) * OW + ow;
                        out_ptr[out_idx] = sum;
                    }
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
