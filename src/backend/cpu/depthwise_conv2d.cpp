/// @file depthwise_conv2d.cpp
/// @brief SIMD-optimized CPU implementation of 2D depthwise convolution (NCHW layout).
///
/// Design informed by:
///   - nn_compute depthwise_conv (height-4 blocking + width-8 SIMD)
///   - onnxruntime MLAS sconv_nchw_depthwise (3x3 kernel specialization, edge handling)
///   - ARM ComputeLibrary NEDepthwiseConvolutionLayer (stride/dilation handling)
///
/// Key optimizations:
///   1. Height blocking: process 4 output rows simultaneously to reuse kernel weights
///   2. Width SIMD: process 8 output columns with v_f32x8 within each height block
///   3. Region splitting: pad-top/bottom → scalar, pad-left/right → scalar,
///      interior → SIMD (h4 for aligned rows, h1 for remainder)
///   4. Kernel weight pre-load into registers within inner loop (reused across 4 rows)
///
/// Falls back to reference implementation when SIMD is unavailable.

#include "nnops/ops/depthwise_conv2d.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <algorithm>
#include <cmath>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

// ============================================================
// Helper: compute a single output element (scalar, used for
// pad regions where SIMD boundaries don't apply cleanly).
// ============================================================
inline float depthwise_elem(const float* input,
                             const float* weight,
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
            sum += input[ih * ih_step + iw * iw_step] * weight[kh * KW + kw];
        }
    }
    return sum;
}

// ============================================================
// SIMD h4 kernel: process 4 output rows × 8 output columns.
//
// Layout notes (all tensors are NCHW dense):
//   ih_step = IH * IW (pixels per input channel)
//   iw_step = IW
//   oh_step = OH * OW (pixels per output channel)
//   ow_step = OW
//
// For each (oh_start, ow_start), we process:
//   - 4 consecutive output rows: oh, oh+1, oh+2, oh+3
//   - 8 consecutive output cols: ow, ow+1, ..., ow+7
//
// We take advantage of the fact that all 4 rows share the same
// kernel weights and each row reads from adjacent input rows.
// ============================================================
inline void dwconv_h4_simd(float* output,
                            const float* input,
                            const float* weight,
                            float bias_val,
                            int64_t IH, int64_t IW,
                            int64_t oh_start, int64_t ow_start, int64_t ow_end,
                            int64_t KH, int64_t KW,
                            int64_t SH, int64_t SW,
                            int64_t DH, int64_t DW,
                            int64_t PH, int64_t PW,
                            int64_t out_h_stride, int64_t out_w_stride,
                            int64_t in_h_stride,  int64_t in_w_stride,
                            bool add_to)
{
    for (int64_t ow = ow_start; ow < ow_end; ow += 8) {
        // Accumulators for 4 rows × 8 columns
        v_f32x8 vacc0 = v_set1_f32x8(bias_val);
        v_f32x8 vacc1 = v_set1_f32x8(bias_val);
        v_f32x8 vacc2 = v_set1_f32x8(bias_val);
        v_f32x8 vacc3 = v_set1_f32x8(bias_val);

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
                // Broadcast kernel weight to all 8 lanes
                const float kval = weight[kh * KW + kw];
                const v_f32x8 vk = v_set1_f32x8(kval);

                const int64_t iw_base = ow * SW + kw * DW - PW;

                if (valid0) {
                    const v_f32x8 vin0 = v_load_f32x8(input + ih0 * in_h_stride + iw_base * in_w_stride);
                    vacc0 = v_fmadd(vin0, vk, vacc0);
                }
                if (valid1) {
                    const v_f32x8 vin1 = v_load_f32x8(input + ih1 * in_h_stride + iw_base * in_w_stride);
                    vacc1 = v_fmadd(vin1, vk, vacc1);
                }
                if (valid2) {
                    const v_f32x8 vin2 = v_load_f32x8(input + ih2 * in_h_stride + iw_base * in_w_stride);
                    vacc2 = v_fmadd(vin2, vk, vacc2);
                }
                if (valid3) {
                    const v_f32x8 vin3 = v_load_f32x8(input + ih3 * in_h_stride + iw_base * in_w_stride);
                    vacc3 = v_fmadd(vin3, vk, vacc3);
                }
            }
        }

        // Write back 4 rows × 8 columns
        float* out_row0 = output + (oh_start + 0) * out_h_stride + ow * out_w_stride;
        float* out_row1 = output + (oh_start + 1) * out_h_stride + ow * out_w_stride;
        float* out_row2 = output + (oh_start + 2) * out_h_stride + ow * out_w_stride;
        float* out_row3 = output + (oh_start + 3) * out_h_stride + ow * out_w_stride;

        if (add_to) {
            vacc0 = v_add(vacc0, v_load_f32x8(out_row0));
            vacc1 = v_add(vacc1, v_load_f32x8(out_row1));
            vacc2 = v_add(vacc2, v_load_f32x8(out_row2));
            vacc3 = v_add(vacc3, v_load_f32x8(out_row3));
        }

        v_store(out_row0, vacc0);
        v_store(out_row1, vacc1);
        v_store(out_row2, vacc2);
        v_store(out_row3, vacc3);
    }
}

// ============================================================
// SIMD h1 kernel: process 1 output row × 8 output columns.
// Used for rows that can't be grouped into blocks of 4.
// ============================================================
inline void dwconv_h1_simd(float* output,
                            const float* input,
                            const float* weight,
                            float bias_val,
                            int64_t IH, int64_t IW,
                            int64_t oh, int64_t ow_start, int64_t ow_end,
                            int64_t KH, int64_t KW,
                            int64_t SH, int64_t SW,
                            int64_t DH, int64_t DW,
                            int64_t PH, int64_t PW,
                            int64_t out_h_stride, int64_t out_w_stride,
                            int64_t in_h_stride,  int64_t in_w_stride,
                            bool add_to)
{
    for (int64_t ow = ow_start; ow < ow_end; ow += 8) {
        v_f32x8 vacc = v_set1_f32x8(bias_val);

        for (int64_t kh = 0; kh < KH; ++kh) {
            const int64_t ih = oh * SH + kh * DH - PH;
            if (ih < 0 || ih >= IH) continue;

            for (int64_t kw = 0; kw < KW; ++kw) {
                const float kval = weight[kh * KW + kw];
                const v_f32x8 vk = v_set1_f32x8(kval);
                const int64_t iw_base = ow * SW + kw * DW - PW;

                const v_f32x8 vin = v_load_f32x8(input + ih * in_h_stride + iw_base * in_w_stride);
                vacc = v_fmadd(vin, vk, vacc);
            }
        }

        float* out_row = output + oh * out_h_stride + ow * out_w_stride;

        if (add_to) {
            vacc = v_add(vacc, v_load_f32x8(out_row));
        }

        v_store(out_row, vacc);
    }
}

// ============================================================
// Scalar row processor: handles elements that didn't fit in
// 8-wide SIMD or are in pad regions.
// ============================================================
inline void dwconv_scalar_row(float* output,
                               const float* input,
                               const float* weight,
                               float bias_val,
                               int64_t IH, int64_t IW,
                               int64_t oh, int64_t ow_start, int64_t ow_end,
                               int64_t KH, int64_t KW,
                               int64_t SH, int64_t SW,
                               int64_t DH, int64_t DW,
                               int64_t PH, int64_t PW,
                               int64_t out_h_stride, int64_t out_w_stride,
                               int64_t in_h_stride,  int64_t in_w_stride,
                               bool add_to,
                               const Epilogue& epilogue, int64_t channel)
{
    for (int64_t ow = ow_start; ow < ow_end; ++ow) {
        float sum = depthwise_elem(input, weight, bias_val,
                                    IH, IW, oh, ow,
                                    KH, KW, SH, SW, DH, DW, PH, PW,
                                    in_h_stride, in_w_stride);
        float val = apply_epilogue(epilogue, sum, channel);
        const int64_t out_idx = oh * out_h_stride + ow * out_w_stride;
        output[out_idx] = add_to ? output[out_idx] + val : val;
    }
}

}  // anonymous namespace

// ============================================================
// Main entry point
// ============================================================
void depthwise_conv2d_cpu(const DepthwiseConv2DAttributes& attrs,
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

    // Strides for dense NCHW layout
    const int64_t in_ch_stride  = IH * IW;
    const int64_t in_h_stride   = IW;
    const int64_t in_w_stride   = 1;
    const int64_t out_ch_stride = OH * OW;
    const int64_t out_h_stride  = OW;
    const int64_t out_w_stride  = 1;

    // SIMD is only valid when stride_w == 1:
    //   For stride > 1, consecutive output columns map to strided input columns,
    //   and we cannot use contiguous v_load_f32x8. Use scalar path instead.
    //   (Dilation is fine — it affects spacing between kernel iterations, not
    //    within a single SIMD load.)
    const bool use_simd = (SW == 1);

    // Compute valid output height range (where all kernel rows see valid input pixels)
    // oh such that: 0 <= oh*SH + kh*DH - PH < IH for all kh in [0, KH)
    const int64_t oh_beg = static_cast<int64_t>(
        std::ceil(static_cast<float>(PH) / static_cast<float>(SH)));
    const int64_t oh_end = static_cast<int64_t>(
        std::max(static_cast<int64_t>(
            std::ceil((static_cast<float>(IH) + static_cast<float>(PH) -
                       static_cast<float>((KH - 1) * DH + 1)) / static_cast<float>(SH))),
            oh_beg));

    // Compute valid output width range (where all kernel cols see valid input pixels)
    const int64_t ow_beg = static_cast<int64_t>(
        std::ceil(static_cast<float>(PW) / static_cast<float>(SW)));
    const int64_t ow_end = static_cast<int64_t>(
        std::max(static_cast<int64_t>(
            std::ceil((static_cast<float>(IW) + static_cast<float>(PW) -
                       static_cast<float>((KW - 1) * DW + 1)) / static_cast<float>(SW))),
            ow_beg));

    // Round interior OW range to SIMD alignment
    const int64_t ow_simd_beg = ow_beg;
    const int64_t ow_simd_end = use_simd
        ? ow_simd_beg + ((ow_end - ow_simd_beg) / 8) * 8
        : ow_beg;  // no SIMD: collapse to empty range

    // Per-sample compute lambda
    const auto compute_sample = [&](int64_t n) {
        for (int64_t c = 0; c < C; ++c) {
            const float* in_ch = in_ptr + n * C * in_ch_stride + c * in_ch_stride;
            const float* w_ch  = w_ptr + c * KH * KW;
            float* out_ch = out_ptr + n * C * out_ch_stride + c * out_ch_stride;
            const float bias_val = b_ptr ? b_ptr[c] : 0.0f;

            // ---- Pad-top region: scalar ----
            for (int64_t oh = 0; oh < oh_beg && oh < OH; ++oh) {
                dwconv_scalar_row(out_ch, in_ch, w_ch, bias_val,
                                   IH, IW, oh, 0, OW,
                                   KH, KW, SH, SW, DH, DW, PH, PW,
                                   out_h_stride, out_w_stride,
                                   in_h_stride, in_w_stride,
                                   attrs.add_to, attrs.epilogue, c);
            }

            if (use_simd) {
                // ---- Interior region: SIMD h4 + h1 ----
                int64_t oh = oh_beg;
                for (; oh + 3 < oh_end; oh += 4) {
                    // Pad-left: scalar
                    if (ow_beg > 0) {
                        for (int64_t d = 0; d < 4; ++d) {
                            dwconv_scalar_row(out_ch, in_ch, w_ch, bias_val,
                                               IH, IW, oh + d, 0, ow_beg,
                                               KH, KW, SH, SW, DH, DW, PH, PW,
                                               out_h_stride, out_w_stride,
                                               in_h_stride, in_w_stride,
                                               attrs.add_to, attrs.epilogue, c);
                        }
                    }

                    // Interior: SIMD h4
                    dwconv_h4_simd(out_ch, in_ch, w_ch, bias_val,
                                    IH, IW,
                                    oh, ow_simd_beg, ow_simd_end,
                                    KH, KW, SH, SW, DH, DW, PH, PW,
                                    out_h_stride, out_w_stride,
                                    in_h_stride, in_w_stride,
                                    attrs.add_to);

                    // SIMD scalar tail: remaining elements after SIMD blocks
                    if (ow_simd_end < ow_end) {
                        for (int64_t d = 0; d < 4; ++d) {
                            dwconv_scalar_row(out_ch, in_ch, w_ch, bias_val,
                                               IH, IW, oh + d, ow_simd_end, ow_end,
                                               KH, KW, SH, SW, DH, DW, PH, PW,
                                               out_h_stride, out_w_stride,
                                               in_h_stride, in_w_stride,
                                               attrs.add_to, attrs.epilogue, c);
                        }
                    }

                    // Pad-right: scalar
                    if (ow_end < OW) {
                        for (int64_t d = 0; d < 4; ++d) {
                            dwconv_scalar_row(out_ch, in_ch, w_ch, bias_val,
                                               IH, IW, oh + d, ow_end, OW,
                                               KH, KW, SH, SW, DH, DW, PH, PW,
                                               out_h_stride, out_w_stride,
                                               in_h_stride, in_w_stride,
                                               attrs.add_to, attrs.epilogue, c);
                        }
                    }
                }

                // ---- Remaining interior rows: SIMD h1 ----
                for (; oh < oh_end; ++oh) {
                    // Pad-left: scalar
                    if (ow_beg > 0) {
                        dwconv_scalar_row(out_ch, in_ch, w_ch, bias_val,
                                           IH, IW, oh, 0, ow_beg,
                                           KH, KW, SH, SW, DH, DW, PH, PW,
                                           out_h_stride, out_w_stride,
                                           in_h_stride, in_w_stride,
                                           attrs.add_to, attrs.epilogue, c);
                    }

                    // Interior: SIMD h1
                    dwconv_h1_simd(out_ch, in_ch, w_ch, bias_val,
                                    IH, IW, oh, ow_simd_beg, ow_simd_end,
                                    KH, KW, SH, SW, DH, DW, PH, PW,
                                    out_h_stride, out_w_stride,
                                    in_h_stride, in_w_stride,
                                    attrs.add_to);

                    // SIMD scalar tail
                    if (ow_simd_end < ow_end) {
                        dwconv_scalar_row(out_ch, in_ch, w_ch, bias_val,
                                           IH, IW, oh, ow_simd_end, ow_end,
                                           KH, KW, SH, SW, DH, DW, PH, PW,
                                           out_h_stride, out_w_stride,
                                           in_h_stride, in_w_stride,
                                           attrs.add_to, attrs.epilogue, c);
                    }

                    // Pad-right: scalar
                    if (ow_end < OW) {
                        dwconv_scalar_row(out_ch, in_ch, w_ch, bias_val,
                                           IH, IW, oh, ow_end, OW,
                                           KH, KW, SH, SW, DH, DW, PH, PW,
                                           out_h_stride, out_w_stride,
                                           in_h_stride, in_w_stride,
                                           attrs.add_to, attrs.epilogue, c);
                    }
                }
            } else {
                // ---- stride > 1: all-scalar interior ----
                for (int64_t oh = oh_beg; oh < oh_end; ++oh) {
                    dwconv_scalar_row(out_ch, in_ch, w_ch, bias_val,
                                       IH, IW, oh, 0, OW,
                                       KH, KW, SH, SW, DH, DW, PH, PW,
                                       out_h_stride, out_w_stride,
                                       in_h_stride, in_w_stride,
                                       attrs.add_to, attrs.epilogue, c);
                }
            }

            // ---- Pad-bottom region: scalar ----
            for (int64_t oh = oh_end; oh < OH; ++oh) {
                dwconv_scalar_row(out_ch, in_ch, w_ch, bias_val,
                                   IH, IW, oh, 0, OW,
                                   KH, KW, SH, SW, DH, DW, PH, PW,
                                   out_h_stride, out_w_stride,
                                   in_h_stride, in_w_stride,
                                   attrs.add_to, attrs.epilogue, c);
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

}  // namespace nnops::backend::cpu
