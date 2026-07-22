/// @file depthwise_conv2d.cpp
/// @brief SIMD-optimized CPU implementation of 2D depthwise convolution (NCHW layout).
///
/// Supports both f32 and f16 data types for all tensors (input, weight, bias, output).
/// Weight and bias dtype match the input dtype. For f16 data, the kernel uses v_f16x8
/// SIMD throughout (native NEON on ARM, convert→compute→convert on x86 F16C).
/// For f32 data, v_f32x8 AVX2/NEON.
///
/// Design informed by:
///   - nn_compute depthwise_conv (height-4 blocking + width-8 SIMD)
///   - onnxruntime MLAS sconv_nchw_depthwise (3x3 kernel specialization, edge handling)
///   - ARM ComputeLibrary NEDepthwiseConvolutionLayer (stride/dilation handling)
///
/// Key optimizations:
///   1. Height blocking: process 4 output rows simultaneously to reuse kernel weights
///   2. Width SIMD: process 8 output columns with v_f32x8/v_f16x8 within each height block
///   3. Region splitting: pad-top/bottom → scalar, pad-left/right → scalar,
///      interior → SIMD (h4 for aligned rows, h1 for remainder)
///   4. Kernel weight pre-load into registers within inner loop (reused across 4 rows)
///   5. N*C parallel: per-(sample,channel) tasks for better thread utilization at N=1
///
/// Falls back to reference implementation when SIMD is unavailable.

#include "nnops/ops/depthwise_conv2d.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <algorithm>
#include <cmath>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {
// ============================================================
// Helper: compute a single output element (scalar)
// ============================================================
template <typename T>
inline float depthwise_elem(const T* input,
                             const T* weight,
                             float bias_val,
                             int64_t IH, int64_t IW,
                             int64_t oh, int64_t ow,
                             int64_t KH, int64_t KW,
                             int64_t SH, int64_t SW,
                             int64_t DH, int64_t DW,
                             int64_t PH, int64_t PW,
                             int64_t ih_step, int64_t iw_step)
{
    float sum = bias_val;
    for (int64_t kh = 0; kh < KH; ++kh) {
        const int64_t ih = oh * SH + kh * DH - PH;
        if (ih < 0 || ih >= IH) continue;
        for (int64_t kw = 0; kw < KW; ++kw) {
            const int64_t iw = ow * SW + kw * DW - PW;
            if (iw < 0 || iw >= IW) continue;
            sum += s_load(&input[ih * ih_step + iw * iw_step]) * s_load(&weight[kh * KW + kw]);
        }
    }
    return sum;
}

// ============================================================
// SIMD h4 kernel: process 4 output rows × 8 output columns.
// Uses type-deduced vector V = v_f32x8 or v_f16x8.
// ============================================================
template <typename T>
inline void dwconv_h4_simd(T* output,
                            const T* input,
                            const T* weight,
                            float bias_val,
                            int64_t IH, int64_t IW,
                            int64_t oh_start, int64_t ow_start, int64_t ow_end,
                            int64_t KH, int64_t KW,
                            int64_t SH, int64_t SW,
                            int64_t DH, int64_t DW,
                            int64_t PH, int64_t PW,
                            int64_t out_row_stride, int64_t out_w_stride,
                            int64_t in_row_stride,  int64_t in_w_stride,
                            bool add_to)
{


    for (int64_t ow = ow_start; ow < ow_end; ow += 8) {
        auto vacc0 = v_set1(input, bias_val);
        auto vacc1 = v_set1(input, bias_val);
        auto vacc2 = v_set1(input, bias_val);
        auto vacc3 = v_set1(input, bias_val);

        for (int64_t kh = 0; kh < KH; ++kh) {
            const int64_t ih0 = (oh_start + 0) * SH + kh * DH - PH;
            const int64_t ih1 = (oh_start + 1) * SH + kh * DH - PH;
            const int64_t ih2 = (oh_start + 2) * SH + kh * DH - PH;
            const int64_t ih3 = (oh_start + 3) * SH + kh * DH - PH;

            const bool valid0 = (ih0 >= 0 && ih0 < IH);
            const bool valid1 = (ih1 >= 0 && ih1 < IH);
            const bool valid2 = (ih2 >= 0 && ih2 < IH);
            const bool valid3 = (ih3 >= 0 && ih3 < IH);

            for (int64_t kw = 0; kw < KW; ++kw) {
                const float kval = s_load(&weight[kh * KW + kw]);
                const auto vk = v_set1(input, kval);

                const int64_t iw_base = ow * SW + kw * DW - PW;

                if (valid0) {
                    const auto vin0 = v_load(input + ih0 * in_row_stride + iw_base * in_w_stride);
                    vacc0 = v_fmadd(vin0, vk, vacc0);
                }
                if (valid1) {
                    const auto vin1 = v_load(input + ih1 * in_row_stride + iw_base * in_w_stride);
                    vacc1 = v_fmadd(vin1, vk, vacc1);
                }
                if (valid2) {
                    const auto vin2 = v_load(input + ih2 * in_row_stride + iw_base * in_w_stride);
                    vacc2 = v_fmadd(vin2, vk, vacc2);
                }
                if (valid3) {
                    const auto vin3 = v_load(input + ih3 * in_row_stride + iw_base * in_w_stride);
                    vacc3 = v_fmadd(vin3, vk, vacc3);
                }
            }
        }

        T* out_row0 = output + (oh_start + 0) * out_row_stride + ow * out_w_stride;
        T* out_row1 = output + (oh_start + 1) * out_row_stride + ow * out_w_stride;
        T* out_row2 = output + (oh_start + 2) * out_row_stride + ow * out_w_stride;
        T* out_row3 = output + (oh_start + 3) * out_row_stride + ow * out_w_stride;

        if (add_to) {
            vacc0 = v_add(vacc0, v_load(out_row0));
            vacc1 = v_add(vacc1, v_load(out_row1));
            vacc2 = v_add(vacc2, v_load(out_row2));
            vacc3 = v_add(vacc3, v_load(out_row3));
        }

        v_store(out_row0, vacc0);
        v_store(out_row1, vacc1);
        v_store(out_row2, vacc2);
        v_store(out_row3, vacc3);
    }
}

// ============================================================
// SIMD h1 kernel: process 1 output row × 8 output columns.
// ============================================================
template <typename T>
inline void dwconv_h1_simd(T* output,
                            const T* input,
                            const T* weight,
                            float bias_val,
                            int64_t IH, int64_t IW,
                            int64_t oh, int64_t ow_start, int64_t ow_end,
                            int64_t KH, int64_t KW,
                            int64_t SH, int64_t SW,
                            int64_t DH, int64_t DW,
                            int64_t PH, int64_t PW,
                            int64_t out_row_stride, int64_t out_w_stride,
                            int64_t in_row_stride,  int64_t in_w_stride,
                            bool add_to)
{


    for (int64_t ow = ow_start; ow < ow_end; ow += 8) {
        auto vacc = v_set1(input, bias_val);

        for (int64_t kh = 0; kh < KH; ++kh) {
            const int64_t ih = oh * SH + kh * DH - PH;
            if (ih < 0 || ih >= IH) continue;

            for (int64_t kw = 0; kw < KW; ++kw) {
                const float kval = s_load(&weight[kh * KW + kw]);
                const auto vk = v_set1(input, kval);
                const int64_t iw_base = ow * SW + kw * DW - PW;

                const auto vin = v_load(input + ih * in_row_stride + iw_base * in_w_stride);
                vacc = v_fmadd(vin, vk, vacc);
            }
        }

        T* out_row = output + oh * out_row_stride + ow * out_w_stride;

        if (add_to) {
            vacc = v_add(vacc, v_load(out_row));
        }

        v_store(out_row, vacc);
    }
}

// ============================================================
// Scalar row processor
// ============================================================
template <typename T>
inline void dwconv_scalar_row(T* output,
                               const T* input,
                               const T* weight,
                               float bias_val,
                               int64_t IH, int64_t IW,
                               int64_t oh, int64_t ow_start, int64_t ow_end,
                               int64_t KH, int64_t KW,
                               int64_t SH, int64_t SW,
                               int64_t DH, int64_t DW,
                               int64_t PH, int64_t PW,
                               int64_t out_row_stride, int64_t out_w_stride,
                               int64_t in_row_stride,  int64_t in_w_stride,
                               bool add_to,
                               const Epilogue& epilogue, int64_t channel)
{
    for (int64_t ow = ow_start; ow < ow_end; ++ow) {
        float sum = depthwise_elem(input, weight, bias_val,
                                    IH, IW, oh, ow,
                                    KH, KW, SH, SW, DH, DW, PH, PW,
                                    in_row_stride, in_w_stride);
        float val = apply_epilogue(epilogue, sum, channel);
        const int64_t out_idx = oh * out_row_stride + ow * out_w_stride;
        if (add_to) {
            s_store(&output[out_idx], s_load(&output[out_idx]) + val);
        } else {
            s_store(&output[out_idx], val);
        }
    }
}

}  // anonymous namespace

// ============================================================
// Main depthwise conv implementation (templated on data type T)
// ============================================================

template <typename T>
void dwconv_impl(const DepthwiseConv2DAttributes& attrs,
                  const TensorView& output,
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

    const int64_t KH = attrs.kernel_size[0];
    const int64_t KW = attrs.kernel_size[1];

    const int64_t OH = output.shape(2);
    const int64_t OW = output.shape(3);

    const int64_t SH = attrs.stride[0];
    const int64_t SW = attrs.stride[1];
    const int64_t DH = attrs.dilation[0];
    const int64_t DW = attrs.dilation[1];
    const int64_t PH = attrs.padding[0];
    const int64_t PW = attrs.padding[1];

    auto* out_ptr = output.data_as<T>();
    const auto* in_ptr  = input.data_as<T>();
    const auto* w_ptr   = weight.data_as<T>();
    const auto* b_ptr   = has_bias ? inputs[2].data_as<T>() : nullptr;

    const int64_t in_row_stride   = input.row_stride_elems();
    const int64_t in_w_stride     = 1;
    const int64_t in_ch_stride    = IH * in_row_stride;
    const int64_t out_row_stride  = output.row_stride_elems();
    const int64_t out_w_stride    = 1;
    const int64_t out_ch_stride   = OH * out_row_stride;

    const bool use_simd = (SW == 1);

    const int64_t oh_beg = static_cast<int64_t>(
        std::ceil(static_cast<float>(PH) / static_cast<float>(SH)));
    const int64_t oh_end = static_cast<int64_t>(
        std::max(static_cast<int64_t>(
            std::ceil((static_cast<float>(IH) + static_cast<float>(PH) -
                       static_cast<float>((KH - 1) * DH + 1)) / static_cast<float>(SH))),
            oh_beg));

    const int64_t ow_beg = static_cast<int64_t>(
        std::ceil(static_cast<float>(PW) / static_cast<float>(SW)));
    const int64_t ow_end = static_cast<int64_t>(
        std::max(static_cast<int64_t>(
            std::ceil((static_cast<float>(IW) + static_cast<float>(PW) -
                       static_cast<float>((KW - 1) * DW + 1)) / static_cast<float>(SW))),
            ow_beg));

    const int64_t ow_simd_beg = ow_beg;
    const int64_t ow_simd_end = use_simd
        ? ow_simd_beg + ((ow_end - ow_simd_beg) / 8) * 8
        : ow_beg;

    // Per-channel compute lambda (N*C parallel)
    const auto compute_channel = [&](int64_t n, int64_t c) {
        const T* in_ch = in_ptr + n * C * in_ch_stride + c * in_ch_stride;
        const T* w_ch  = w_ptr + c * KH * KW;
        T* out_ch = out_ptr + n * C * out_ch_stride + c * out_ch_stride;
        const float bias_val = b_ptr ? s_load(&b_ptr[c]) : 0.0f;

        for (int64_t oh = 0; oh < oh_beg && oh < OH; ++oh) {
            dwconv_scalar_row(out_ch, in_ch, w_ch, bias_val,
                               IH, IW, oh, 0, OW,
                               KH, KW, SH, SW, DH, DW, PH, PW,
                               out_row_stride, out_w_stride,
                               in_row_stride, in_w_stride,
                               attrs.add_to, attrs.epilogue, c);
        }

        if (use_simd) {
            int64_t oh = oh_beg;
            for (; oh + 3 < oh_end; oh += 4) {
                if (ow_beg > 0) {
                    for (int64_t d = 0; d < 4; ++d) {
                        dwconv_scalar_row(out_ch, in_ch, w_ch, bias_val,
                                           IH, IW, oh + d, 0, ow_beg,
                                           KH, KW, SH, SW, DH, DW, PH, PW,
                                           out_row_stride, out_w_stride,
                                           in_row_stride, in_w_stride,
                                           attrs.add_to, attrs.epilogue, c);
                    }
                }

                dwconv_h4_simd(out_ch, in_ch, w_ch, bias_val,
                                IH, IW,
                                oh, ow_simd_beg, ow_simd_end,
                                KH, KW, SH, SW, DH, DW, PH, PW,
                                out_row_stride, out_w_stride,
                                in_row_stride, in_w_stride,
                                attrs.add_to);

                if (ow_simd_end < ow_end) {
                    for (int64_t d = 0; d < 4; ++d) {
                        dwconv_scalar_row(out_ch, in_ch, w_ch, bias_val,
                                           IH, IW, oh + d, ow_simd_end, ow_end,
                                           KH, KW, SH, SW, DH, DW, PH, PW,
                                           out_row_stride, out_w_stride,
                                           in_row_stride, in_w_stride,
                                           attrs.add_to, attrs.epilogue, c);
                    }
                }

                if (ow_end < OW) {
                    for (int64_t d = 0; d < 4; ++d) {
                        dwconv_scalar_row(out_ch, in_ch, w_ch, bias_val,
                                           IH, IW, oh + d, ow_end, OW,
                                           KH, KW, SH, SW, DH, DW, PH, PW,
                                           out_row_stride, out_w_stride,
                                           in_row_stride, in_w_stride,
                                           attrs.add_to, attrs.epilogue, c);
                    }
                }
            }

            for (; oh < oh_end; ++oh) {
                if (ow_beg > 0) {
                    dwconv_scalar_row(out_ch, in_ch, w_ch, bias_val,
                                       IH, IW, oh, 0, ow_beg,
                                       KH, KW, SH, SW, DH, DW, PH, PW,
                                       out_row_stride, out_w_stride,
                                       in_row_stride, in_w_stride,
                                       attrs.add_to, attrs.epilogue, c);
                }

                dwconv_h1_simd(out_ch, in_ch, w_ch, bias_val,
                                IH, IW, oh, ow_simd_beg, ow_simd_end,
                                KH, KW, SH, SW, DH, DW, PH, PW,
                                out_row_stride, out_w_stride,
                                in_row_stride, in_w_stride,
                                attrs.add_to);

                if (ow_simd_end < ow_end) {
                    dwconv_scalar_row(out_ch, in_ch, w_ch, bias_val,
                                       IH, IW, oh, ow_simd_end, ow_end,
                                       KH, KW, SH, SW, DH, DW, PH, PW,
                                       out_row_stride, out_w_stride,
                                       in_row_stride, in_w_stride,
                                       attrs.add_to, attrs.epilogue, c);
                }

                if (ow_end < OW) {
                    dwconv_scalar_row(out_ch, in_ch, w_ch, bias_val,
                                       IH, IW, oh, ow_end, OW,
                                       KH, KW, SH, SW, DH, DW, PH, PW,
                                       out_row_stride, out_w_stride,
                                       in_row_stride, in_w_stride,
                                       attrs.add_to, attrs.epilogue, c);
                }
            }
        } else {
            for (int64_t oh = oh_beg; oh < oh_end; ++oh) {
                dwconv_scalar_row(out_ch, in_ch, w_ch, bias_val,
                                   IH, IW, oh, 0, OW,
                                   KH, KW, SH, SW, DH, DW, PH, PW,
                                   out_row_stride, out_w_stride,
                                   in_row_stride, in_w_stride,
                                   attrs.add_to, attrs.epilogue, c);
            }
        }

        for (int64_t oh = oh_end; oh < OH; ++oh) {
            dwconv_scalar_row(out_ch, in_ch, w_ch, bias_val,
                               IH, IW, oh, 0, OW,
                               KH, KW, SH, SW, DH, DW, PH, PW,
                               out_row_stride, out_w_stride,
                               in_row_stride, in_w_stride,
                               attrs.add_to, attrs.epilogue, c);
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
// Entry point with dtype dispatch
// ============================================================

void depthwise_conv2d_cpu(const DepthwiseConv2DAttributes& attrs,
                           const TensorView& output,
                           std::span<const TensorView> inputs,
                           const ComputeContext& ctx,
                           void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        dwconv_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        dwconv_impl<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"depthwise_conv2d_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
