/// @file conv3d_ref.cpp
/// @brief Naive CPU reference implementation of 3D convolution (NCDHW layout).
///
/// This is the correctness baseline. Uses a straightforward 9-level nested loop
/// with per-sample parallelism via the external parallel_for hook.

#include "nnops/ops/conv3d.hpp"
#include "nnops/core/parallel_for.hpp"
#include "../epilogue_impl.hpp"

namespace nnops::backend::cpu::reference {

void conv3d_ref(const Conv3DAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* /*workspace*/)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    // Input: [N, IC, ID, IH, IW]
    const int64_t N  = input.shape(0);
    const int64_t IC = input.shape(1);
    const int64_t ID = input.shape(2);
    const int64_t IH = input.shape(3);
    const int64_t IW = input.shape(4);
    const int64_t in_row_stride = input.row_stride_elems();   // elements per row (>= IW)
    const int64_t in_d_stride   = IH * in_row_stride;          // elements per depth slice

    // Weight: [OC, IC/G, KD, KH, KW]
    const int64_t OC = weight.shape(0);
    const int64_t KC = weight.shape(1);  // IC / groups
    const int64_t KD = attrs.kernel_size[0];
    const int64_t KH = attrs.kernel_size[1];
    const int64_t KW = attrs.kernel_size[2];

    // Output: [N, OC, OD, OH, OW]
    const int64_t OD = output.shape(2);
    const int64_t OH = output.shape(3);
    const int64_t OW = output.shape(4);
    const int64_t out_row_stride = output.row_stride_elems();  // elements per row (>= OW)
    const int64_t out_d_stride   = OH * out_row_stride;         // elements per depth slice

    const int64_t G  = attrs.groups;
    const int64_t OC_per_G = OC / G;

    const int64_t SD = attrs.stride[0];
    const int64_t SH = attrs.stride[1];
    const int64_t SW = attrs.stride[2];
    const int64_t DD = attrs.dilation[0];
    const int64_t DH = attrs.dilation[1];
    const int64_t DW = attrs.dilation[2];
    const int64_t PD = attrs.padding[0];
    const int64_t PH = attrs.padding[1];
    const int64_t PW = attrs.padding[2];

    auto* out_ptr = output.ptr<float>();
    const auto* in_ptr  = input.ptr<float>();
    const auto* w_ptr   = weight.ptr<float>();
    const auto* b_ptr   = has_bias ? inputs[2].ptr<float>() : nullptr;

    // Per-sample compute lambda
    const auto compute_sample = [&](int64_t n) {
        for (int64_t g = 0; g < G; ++g) {
            for (int64_t oc = 0; oc < OC_per_G; ++oc) {
                const int64_t oc_global = g * OC_per_G + oc;
                for (int64_t od = 0; od < OD; ++od) {
                    for (int64_t oh = 0; oh < OH; ++oh) {
                        for (int64_t ow = 0; ow < OW; ++ow) {
                            float sum = 0.0f;
                            const int64_t ic_start = g * KC;
                            const int64_t ic_end   = ic_start + KC;
                            for (int64_t ic = ic_start; ic < ic_end; ++ic) {
                                for (int64_t kd = 0; kd < KD; ++kd) {
                                    for (int64_t kh = 0; kh < KH; ++kh) {
                                        for (int64_t kw = 0; kw < KW; ++kw) {
                                            const int64_t id = od * SD + kd * DD - PD;
                                            const int64_t ih = oh * SH + kh * DH - PH;
                                            const int64_t iw = ow * SW + kw * DW - PW;
                                            if (id >= 0 && id < ID &&
                                                ih >= 0 && ih < IH &&
                                                iw >= 0 && iw < IW) {
                                                const int64_t in_idx =
                                                    (((n * IC + ic) * ID + id)
                                                     * IH + ih) * in_row_stride + iw;
                                                const int64_t w_idx =
                                                    ((((oc_global * KC) + (ic - ic_start))
                                                      * KD + kd) * KH + kh) * KW + kw;
                                                sum += in_ptr[in_idx] * w_ptr[w_idx];
                                            }
                                        }
                                    }
                                }
                            }
                            if (has_bias) {
                                sum += b_ptr[oc_global];
                            }
                            const int64_t out_idx =
                                (((n * OC + oc_global) * OD + od)
                                 * OH + oh) * out_row_stride + ow;
                            float val = apply_epilogue(attrs.epilogue, sum, oc_global);
                            out_ptr[out_idx] = attrs.add_to ? out_ptr[out_idx] + val : val;
                        }
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
