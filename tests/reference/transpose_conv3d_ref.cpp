/// @file transpose_conv3d_ref.cpp
/// @brief channel-first scalar reference — used as correctness baseline for tests.
///
/// The scatter-add (input-driven) form, one axis further out than the 2-D
/// reference: each input voxel, scaled by kernel weights, is accumulated into the
/// appropriate output region. It never reads the output before the epilogue pass,
/// so it is independent of the routes the kernel under test takes.
///
/// Weight layout: channel-first [IC, OC/G, KD, KH, KW].
/// Input:  channel-first [N, IC, ID, IH, IW]
/// Output: channel-first [N, OC, OD, OH, OW]

#include "nnops/ops/transpose_conv3d.hpp"
#include "nnops/core/parallel_for.hpp"
#include "simd_kernel/simd_epilogue.hpp"

#include <algorithm>
#include <vector>

namespace nnops::backend::cpu::reference {

void transpose_conv3d_channel_first_ref(const TransposeConv3DAttributes& attrs,
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
    const int64_t ID = input.shape(2);
    const int64_t IH = input.shape(3);
    const int64_t IW = input.shape(4);
    const int64_t in_row_stride = input.row_stride_elems();
    const int64_t in_d_stride   = IH * in_row_stride;

    const int64_t OC_per_G = weight.shape(1);
    const int64_t KD = attrs.kernel_size[0];
    const int64_t KH = attrs.kernel_size[1];
    const int64_t KW = attrs.kernel_size[2];

    const int64_t OC = output.shape(1);
    const int64_t OD = output.shape(2);
    const int64_t OH = output.shape(3);
    const int64_t OW = output.shape(4);
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_d_stride   = OH * out_row_stride;

    const int64_t G = attrs.groups;
    const int64_t IC_per_G = IC / G;

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

    const int64_t out_sample_size = OC * OD * out_d_stride;

    const auto compute_sample = [&](int64_t n) {
        const int64_t n_offset = n * out_sample_size;

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
                    const int64_t w_base =
                        ((ic * OC_per_G + oc_local) * KD * KH) * KW;

                    for (int64_t id = 0; id < ID; ++id) {
                        for (int64_t ih = 0; ih < IH; ++ih) {
                            for (int64_t iw = 0; iw < IW; ++iw) {
                                const float in_val = in_ptr[
                                    ((n * IC + ic) * ID + id) * in_d_stride +
                                    ih * in_row_stride + iw];

                                for (int64_t kd = 0; kd < KD; ++kd) {
                                    const int64_t od = id * SD + kd * DD - PD;
                                    if (od < 0 || od >= OD) { continue; }

                                    for (int64_t kh = 0; kh < KH; ++kh) {
                                        const int64_t oh = ih * SH + kh * DH - PH;
                                        if (oh < 0 || oh >= OH) { continue; }

                                        for (int64_t kw = 0; kw < KW; ++kw) {
                                            const int64_t ow = iw * SW + kw * DW - PW;
                                            if (ow < 0 || ow >= OW) { continue; }

                                            const float w_val = w_ptr[
                                                w_base + (kd * KH + kh) * KW + kw];
                                            const int64_t out_idx =
                                                (oc * OD + od) * out_d_stride +
                                                oh * out_row_stride + ow;
                                            out_ptr[n_offset + out_idx] += in_val * w_val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for (int64_t oc = 0; oc < OC; ++oc) {
            const float bias_val = b_ptr ? b_ptr[oc] : 0.0f;
            for (int64_t od = 0; od < OD; ++od) {
                for (int64_t oh = 0; oh < OH; ++oh) {
                    for (int64_t ow = 0; ow < OW; ++ow) {
                        const int64_t out_idx =
                            (oc * OD + od) * out_d_stride + oh * out_row_stride + ow;
                        float val = out_ptr[n_offset + out_idx] + bias_val;
                        if (attrs.add_to && attrs.residual_pre_activation) {
                            val += saved[out_idx];
                        }
                        val = apply_epilogue(attrs.epilogue, val, oc);
                        if (attrs.add_to && !attrs.residual_pre_activation) {
                            out_ptr[n_offset + out_idx] = saved[out_idx] + val;
                        } else {
                            out_ptr[n_offset + out_idx] = val;
                        }
                    }
                }
            }
        }
    };

    ctx.cpu.run(0, N, compute_sample);
}

}  // namespace nnops::backend::cpu::reference
