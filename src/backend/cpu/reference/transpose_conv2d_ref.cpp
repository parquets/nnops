/// @file transpose_conv2d_ref.cpp
/// @brief NCHW planar scalar reference — used as correctness baseline for tests.
///
/// Implements the scatter-add (input-driven) approach from ncnn Deconvolution:
/// each input pixel, scaled by kernel weights, is accumulated into the
/// appropriate output region.
///
/// This kernel operates on NCHW planar tensors. It serves as the golden
/// reference for verifying the NCHWC8 SIMD implementation.
///
/// Weight layout: planar NCHW [IC, OC/G, KH, KW].
/// Input:  NCHW [N, IC, IH, IW]
/// Output: NCHW [N, OC, OH, OW]

#include "nnops/ops/transpose_conv2d.hpp"
#include "nnops/core/parallel_for.hpp"
#include "../simd_kernel/simd_epilogue.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

namespace nnops::backend::cpu::reference {

void transpose_conv2d_nchw_ref(const TransposeConv2DAttributes& attrs,
                                TensorView& output,
                                std::span<const TensorView> inputs,
                                const ComputeContext& ctx,
                                void* /*workspace*/)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t N  = input.shape(0);
    const int64_t IC = input.shape(1);
    const int64_t IH = input.shape(2);
    const int64_t IW = input.shape(3);
    const int64_t in_row_stride = input.row_stride_elems();

    const int64_t OC_per_G = weight.shape(1);
    const int64_t KH = attrs.kernel_size[0];
    const int64_t KW = attrs.kernel_size[1];

    const int64_t OC = output.shape(1);
    const int64_t OH = output.shape(2);
    const int64_t OW = output.shape(3);
    const int64_t out_row_stride = output.row_stride_elems();

    const int64_t G = attrs.groups;
    const int64_t IC_per_G = IC / G;

    const int64_t SH = attrs.stride[0];
    const int64_t SW = attrs.stride[1];
    const int64_t DH = attrs.dilation[0];
    const int64_t DW = attrs.dilation[1];
    const int64_t PH = attrs.padding[0];
    const int64_t PW = attrs.padding[1];

    auto* out_ptr = output.ptr<float>();
    const auto* in_ptr  = input.ptr<float>();
    const auto* w_ptr   = weight.ptr<float>();
    const auto* b_ptr   = has_bias ? inputs[2].ptr<float>() : nullptr;

    const int64_t out_sample_size = OC * OH * out_row_stride;

    const auto compute_sample = [&](int64_t n) {
        const int64_t n_offset = n * OC * OH * out_row_stride;

        // For add_to: save existing output before zero/scatter
        std::vector<float> saved;
        if (attrs.add_to) {
            saved.assign(out_ptr + n_offset, out_ptr + n_offset + out_sample_size);
        }

        std::fill_n(out_ptr + n_offset, out_sample_size, 0.0f);

        for (int64_t g = 0; g < G; ++g) {
            const int64_t ic_start = g * IC_per_G;
            const int64_t oc_start = g * OC_per_G;

            for (int64_t ic = ic_start; ic < ic_start + IC_per_G; ++ic) {
                for (int64_t oc = oc_start; oc < oc_start + OC_per_G; ++oc) {
                    const int64_t oc_local = oc - oc_start;
                    const int64_t w_base = ((ic * OC_per_G + oc_local) * KH) * KW;

                    for (int64_t ih = 0; ih < IH; ++ih) {
                        for (int64_t iw = 0; iw < IW; ++iw) {
                            const float in_val = in_ptr[
                                ((n * IC + ic) * IH + ih) * in_row_stride + iw];
                            if (in_val == 0.0f) { continue; }

                            for (int64_t kh = 0; kh < KH; ++kh) {
                                const int64_t oh = ih * SH + kh * DH - PH;
                                if (oh < 0 || oh >= OH) { continue; }

                                for (int64_t kw = 0; kw < KW; ++kw) {
                                    const int64_t ow = iw * SW + kw * DW - PW;
                                    if (ow < 0 || ow >= OW) { continue; }

                                    const float w_val = w_ptr[w_base + kh * KW + kw];
                                    const int64_t out_idx =
                                        (oc * OH + oh) * out_row_stride + ow;
                                    out_ptr[n_offset + out_idx] += in_val * w_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        for (int64_t oc = 0; oc < OC; ++oc) {
            const float bias_val = b_ptr ? b_ptr[oc] : 0.0f;
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    const int64_t out_idx =
                        (oc * OH + oh) * out_row_stride + ow;
                    float val = out_ptr[n_offset + out_idx] + bias_val;
                    val = apply_epilogue(attrs.epilogue, val, oc);
                    if (attrs.add_to) {
                        out_ptr[n_offset + out_idx] = saved[out_idx] + val;
                    } else {
                        out_ptr[n_offset + out_idx] = val;
                    }
                }
            }
        }
    };

    // Per-sample dispatch
    ctx.cpu.run(0, N,
            [&](int64_t n) { compute_sample(n); });
}

}  // namespace nnops::backend::cpu::reference
