/// @file conv3d_ref.cpp
/// @brief channel-first scalar reference — used as correctness baseline for tests.
///
/// The input-driven (scatter) form, which is the inverse of the gather every
/// kernel route takes: each input voxel, scaled by a kernel weight, lands on the
/// output voxels that read it back, oh = (ih + PH − kh·DH) / SH only where SH
/// divides the numerator. So the two derivations share no index arithmetic, and
/// a stride or padding sign error in one cannot cancel in the other.
///
/// Weight layout: channel-first [OC, IC/G, KD, KH, KW].
/// Input:  channel-first [N, IC, ID, IH, IW]
/// Output: channel-first [N, OC, OD, OH, OW]

#include "nnops/ops/conv3d.hpp"
#include "nnops/core/parallel_for.hpp"
#include "simd_kernel/simd_epilogue.hpp"

#include <vector>

namespace nnops::backend::cpu::reference {

void conv3d_channel_first_ref(const Conv3DAttributes& attrs,
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
    const int64_t in_n_stride = input.stride_elems(0);
    const int64_t in_c_stride = input.stride_elems(1);
    const int64_t in_d_stride = input.stride_elems(2);
    const int64_t in_row_stride = input.row_stride_elems();

    const int64_t OC = weight.shape(0);
    const int64_t IC_per_G = weight.shape(1);
    const int64_t KD = attrs.kernel_size[0];
    const int64_t KH = attrs.kernel_size[1];
    const int64_t KW = attrs.kernel_size[2];

    const int64_t OD = output.shape(2);
    const int64_t OH = output.shape(3);
    const int64_t OW = output.shape(4);
    const int64_t out_n_stride   = output.stride_elems(0);
    const int64_t out_c_stride   = output.stride_elems(1);
    const int64_t out_d_stride   = output.stride_elems(2);
    const int64_t out_row_stride = output.row_stride_elems();

    const int64_t G = attrs.groups;
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
    const auto* in_ptr = input.ptr<float>();
    const auto* w_ptr  = weight.ptr<float>();
    const auto* b_ptr  = has_bias ? inputs[2].ptr<float>() : nullptr;

    const int64_t kdim = IC_per_G * KD * KH * KW;

    const auto out_idx = [&](int64_t oc, int64_t od, int64_t oh, int64_t ow) {
        return oc * out_c_stride + od * out_d_stride + oh * out_row_stride + ow;
    };

    ctx.cpu.run(0, N, [&](int64_t n) {
        float* o = out_ptr + n * out_n_stride;

        // add_to reads the destination once, after the activation — Conv3D has
        // no pre-activation form — so the old values have to survive the zeroing.
        std::vector<float> saved;
        if (attrs.add_to) {
            saved.resize(static_cast<size_t>(OC * OD * OH * OW));
            for (int64_t oc = 0; oc < OC; ++oc) {
                for (int64_t od = 0; od < OD; ++od) {
                    for (int64_t oh = 0; oh < OH; ++oh) {
                        for (int64_t ow = 0; ow < OW; ++ow) {
                            saved[static_cast<size_t>(((oc * OD + od) * OH + oh) * OW + ow)] =
                                o[out_idx(oc, od, oh, ow)];
                        }
                    }
                }
            }
        }

        for (int64_t oc = 0; oc < OC; ++oc) {
            for (int64_t od = 0; od < OD; ++od) {
                for (int64_t oh = 0; oh < OH; ++oh) {
                    float* row = o + out_idx(oc, od, oh, 0);
                    for (int64_t ow = 0; ow < OW; ++ow) { row[ow] = 0.0f; }
                }
            }
        }

        for (int64_t g = 0; g < G; ++g) {
            for (int64_t oc_l = 0; oc_l < OC_per_G; ++oc_l) {
                const int64_t oc = g * OC_per_G + oc_l;
                for (int64_t ic_l = 0; ic_l < IC_per_G; ++ic_l) {
                    const int64_t ic = g * IC_per_G + ic_l;
                    const float* w_ic = w_ptr + oc * kdim + ic_l * (KD * KH * KW);
                    const float* in_c = in_ptr + n * in_n_stride + ic * in_c_stride;

                    for (int64_t id = 0; id < ID; ++id) {
                        for (int64_t ih = 0; ih < IH; ++ih) {
                            const float* in_row = in_c + id * in_d_stride +
                                                  ih * in_row_stride;
                            for (int64_t iw = 0; iw < IW; ++iw) {
                                const float v = in_row[iw];
                                for (int64_t kd = 0; kd < KD; ++kd) {
                                    const int64_t nd = id + PD - kd * DD;
                                    if (nd < 0 || nd % SD != 0) { continue; }
                                    const int64_t od = nd / SD;
                                    if (od >= OD) { continue; }
                                    for (int64_t kh = 0; kh < KH; ++kh) {
                                        const int64_t nh = ih + PH - kh * DH;
                                        if (nh < 0 || nh % SH != 0) { continue; }
                                        const int64_t oh = nh / SH;
                                        if (oh >= OH) { continue; }
                                        for (int64_t kw = 0; kw < KW; ++kw) {
                                            const int64_t nw = iw + PW - kw * DW;
                                            if (nw < 0 || nw % SW != 0) { continue; }
                                            const int64_t ow = nw / SW;
                                            if (ow >= OW) { continue; }
                                            o[out_idx(oc, od, oh, ow)] +=
                                                v * w_ic[(kd * KH + kh) * KW + kw];
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
                    float* row = o + out_idx(oc, od, oh, 0);
                    for (int64_t ow = 0; ow < OW; ++ow) {
                        float val = apply_epilogue(attrs.epilogue, row[ow] + bias_val, oc);
                        if (attrs.add_to) {
                            val += saved[static_cast<size_t>(
                                ((oc * OD + od) * OH + oh) * OW + ow)];
                        }
                        row[ow] = val;
                    }
                }
            }
        }
    });
}

}  // namespace nnops::backend::cpu::reference
