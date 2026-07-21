/// @file pooling.cpp
/// @brief SIMD-optimized CPU implementation of 2D/3D spatial pooling (NCHW/NCDHW layout).
///
/// Design combines best practices from two reference codebases:
///
///   nn_compute (d:\vscode\nn_compute\src\cpu\kernel\pooling\):
///     - Height-4 blocking: process 4 output rows simultaneously for Max/Avg
///     - Pre-positioned pointer passing: caller pre-positions input+output pointers,
///       SIMD kernels use only relative offsets — eliminates positional bugs
///     - Trust boundary: SIMD kernels called ONLY for interior valid region;
///       NO per-element bounds checks inside SIMD kernels
///     - Region splitting: pad-depth/top/bottom/front/back → scalar; interior → SIMD
///
///   onnxruntime MLAS (d:\git\onnxruntime\onnxruntime\core\mlas\lib\pooling.cpp):
///     - Two-phase reduction (vertical then horizontal) — noted for future reference
///     - Identity pre-fill for edge handling (reduction buffer approach)
///     - Global pooling fast path (dedicated kernel when kernel == input spatial shape)
///
/// Key design decisions for nnops:
///   1. SIMD kernels accept pre-positioned pointers, use only relative offsets
///   2. No per-element validity checks in SIMD — caller guarantees valid interior region
///   3. Width-8 SIMD via v_fp32x8 (AVX2 native, SSE emulated, NEON emulated)
///   4. SIMD gated on stride_w == 1 (nnops SIMD has no gather; contiguous loads required)
///   5. AverageExcludePad and Lp pooling: scalar-only (per-element counting needed)
///   6. Unified 2D/3D code path: KD=1/SD=1/DD=1/PD=0/ID=1/OD=1 for 2D
///   7. add_to support: branch-hoisted; one extra load per store when enabled
///   8. Per-sample parallel dispatch via parallel_for

#include "nnops/ops/pooling.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

// ============================================================
// Unified scalar row processor (all pool types, 2D + 3D, pad + interior)
// ============================================================

/// Process a range of output columns for one (od, oh) position using scalar code.
/// Handles all 4 pool types with full 3D bounds checking.
/// Called for: pad-depth regions, pad-height regions, pad-width edges,
///             non-SIMD types (AvgExcludePad, Lp), and stride_w != 1 fallback.
inline void pooling_scalar_row(
    float* output,          // output row pointer: out_d + oh * out_h_stride
    const float* input,     // channel base pointer: in_ch
    int64_t ID, int64_t IH, int64_t IW,
    int64_t od, int64_t oh,
    int64_t ow_start, int64_t ow_end,
    int64_t KD, int64_t KH, int64_t KW,
    int64_t SD, int64_t SH, int64_t SW,
    int64_t DD, int64_t DH, int64_t DW,
    int64_t PD, int64_t PH, int64_t PW,
    int64_t out_w_stride,
    int64_t in_d_stride, int64_t in_h_stride,
    PoolingType type, bool add_to, int64_t p_norm)
{
    for (int64_t ow = ow_start; ow < ow_end; ++ow) {
        float result = 0.0f;

        switch (type) {
        // ---- MaxPooling ----
        case PoolingType::Max: {
            float max_val = -std::numeric_limits<float>::infinity();
            bool any = false;
            for (int64_t kd = 0; kd < KD; ++kd) {
                const int64_t id = od * SD + kd * DD - PD;
                if (id < 0 || id >= ID) continue;
                for (int64_t kh = 0; kh < KH; ++kh) {
                    const int64_t ih = oh * SH + kh * DH - PH;
                    if (ih < 0 || ih >= IH) continue;
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        const int64_t iw = ow * SW + kw * DW - PW;
                        if (iw < 0 || iw >= IW) continue;
                        const float val = input[id * in_d_stride + ih * in_h_stride + iw];
                        if (val > max_val) max_val = val;
                        any = true;
                    }
                }
            }
            result = any ? max_val : 0.0f;
            break;
        }

        // ---- Average / AverageExcludePad ----
        case PoolingType::Average:
        case PoolingType::AverageExcludePad: {
            float sum = 0.0f;
            int64_t pad_count = 0;
            const int64_t K_total = KD * KH * KW;
            for (int64_t kd = 0; kd < KD; ++kd) {
                const int64_t id = od * SD + kd * DD - PD;
                for (int64_t kh = 0; kh < KH; ++kh) {
                    const int64_t ih = oh * SH + kh * DH - PH;
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        const int64_t iw = ow * SW + kw * DW - PW;
                        if (id < 0 || id >= ID || ih < 0 || ih >= IH ||
                            iw < 0 || iw >= IW) {
                            ++pad_count;
                            continue;
                        }
                        sum += input[id * in_d_stride + ih * in_h_stride + iw];
                    }
                }
            }
            if (type == PoolingType::AverageExcludePad) {
                const int64_t valid = K_total - pad_count;
                result = (valid > 0) ? sum / static_cast<float>(valid) : 0.0f;
            } else {
                result = sum / static_cast<float>(K_total);
            }
            break;
        }

        // ---- Lp Pooling ----
        case PoolingType::Lp: {
            float sum = 0.0f;
            const float fp = static_cast<float>(p_norm);
            for (int64_t kd = 0; kd < KD; ++kd) {
                const int64_t id = od * SD + kd * DD - PD;
                if (id < 0 || id >= ID) continue;
                for (int64_t kh = 0; kh < KH; ++kh) {
                    const int64_t ih = oh * SH + kh * DH - PH;
                    if (ih < 0 || ih >= IH) continue;
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        const int64_t iw = ow * SW + kw * DW - PW;
                        if (iw < 0 || iw >= IW) continue;
                        sum += std::pow(
                            std::abs(input[id * in_d_stride + ih * in_h_stride + iw]), fp);
                    }
                }
            }
            result = std::pow(sum, 1.0f / fp);
            break;
        }
        }

        // Write back
        const int64_t out_idx = ow * out_w_stride;
        output[out_idx] = add_to ? output[out_idx] + result : result;
    }
}

// ============================================================
// MaxPooling SIMD kernels
// ============================================================
//
// Pointer conventions (PRE-POSITIONED by caller):
//   output: out_d + oh_start * out_h_stride + ow_start * out_w_stride
//   input:  in_ch
//           + (od * SD - PD) * in_d_stride     ← base depth position
//           + (oh_start * SH - PH) * in_h_stride ← base height position
//           + (ow_start - PW)                   ← base width position (SW == 1)
//
// Relative offsets used inside the kernel (caller guarantees validity):
//   Depth:  kd * DD * in_d_stride
//   Height: (r * SH + kh * DH) * in_h_stride    (r = 0..3 for h4, 0 for h1)
//   Width:  (ow - ow_start) + kw * DW           (SW == 1, in_w_stride == 1, ow loops)

/// Process 4 output rows × up to `ow_simd_count * 8` output columns with element-wise max.
/// Called only for interior valid region (SW == 1 required).
inline void maxpool_h4_simd(
    float* output, const float* input,
    int64_t KD, int64_t KH, int64_t KW,
    int64_t SH,
    int64_t DD, int64_t DH, int64_t DW,
    int64_t ow_simd_count,
    int64_t out_h_stride, int64_t out_w_stride,
    int64_t in_d_stride, int64_t in_h_stride,
    bool add_to)
{
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();

    for (int64_t blk = 0; blk < ow_simd_count; ++blk) {
        const int64_t w_off = blk * 8;  // relative width offset from ow_start

        v_fp32x8 vacc0 = set1_fp32x8(neg_inf);
        v_fp32x8 vacc1 = set1_fp32x8(neg_inf);
        v_fp32x8 vacc2 = set1_fp32x8(neg_inf);
        v_fp32x8 vacc3 = set1_fp32x8(neg_inf);

        for (int64_t kd = 0; kd < KD; ++kd) {
            const int64_t d_off = kd * DD * in_d_stride;
            for (int64_t kh = 0; kh < KH; ++kh) {
                const int64_t h_off0 = (0 * SH + kh * DH) * in_h_stride;
                const int64_t h_off1 = (1 * SH + kh * DH) * in_h_stride;
                const int64_t h_off2 = (2 * SH + kh * DH) * in_h_stride;
                const int64_t h_off3 = (3 * SH + kh * DH) * in_h_stride;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    const int64_t kw_off = w_off + kw * DW;
                    vacc0 = max(vacc0, load_fp32x8(input + d_off + h_off0 + kw_off));
                    vacc1 = max(vacc1, load_fp32x8(input + d_off + h_off1 + kw_off));
                    vacc2 = max(vacc2, load_fp32x8(input + d_off + h_off2 + kw_off));
                    vacc3 = max(vacc3, load_fp32x8(input + d_off + h_off3 + kw_off));
                }
            }
        }

        // Write back
        float* out0 = output + 0 * out_h_stride + w_off * out_w_stride;
        float* out1 = output + 1 * out_h_stride + w_off * out_w_stride;
        float* out2 = output + 2 * out_h_stride + w_off * out_w_stride;
        float* out3 = output + 3 * out_h_stride + w_off * out_w_stride;

        if (add_to) {
            vacc0 = add(vacc0, load_fp32x8(out0));
            vacc1 = add(vacc1, load_fp32x8(out1));
            vacc2 = add(vacc2, load_fp32x8(out2));
            vacc3 = add(vacc3, load_fp32x8(out3));
        }

        store(out0, vacc0);
        store(out1, vacc1);
        store(out2, vacc2);
        store(out3, vacc3);
    }
}

/// Process 1 output row × up to `ow_simd_count * 8` output columns with element-wise max.
/// Called only for interior valid region (SW == 1 required).
inline void maxpool_h1_simd(
    float* output, const float* input,
    int64_t KD, int64_t KH, int64_t KW,
    int64_t SH,
    int64_t DD, int64_t DH, int64_t DW,
    int64_t ow_simd_count,
    int64_t out_h_stride, int64_t out_w_stride,
    int64_t in_d_stride, int64_t in_h_stride,
    bool add_to)
{
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();

    for (int64_t blk = 0; blk < ow_simd_count; ++blk) {
        const int64_t w_off = blk * 8;
        v_fp32x8 vacc = set1_fp32x8(neg_inf);

        for (int64_t kd = 0; kd < KD; ++kd) {
            const int64_t d_off = kd * DD * in_d_stride;
            for (int64_t kh = 0; kh < KH; ++kh) {
                const int64_t h_off = kh * DH * in_h_stride;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    const int64_t kw_off = w_off + kw * DW;
                    vacc = max(vacc, load_fp32x8(input + d_off + h_off + kw_off));
                }
            }
        }

        float* out_r = output + w_off * out_w_stride;

        if (add_to) {
            vacc = add(vacc, load_fp32x8(out_r));
        }

        store(out_r, vacc);
    }
}

// ============================================================
// AvgPooling SIMD kernels (Average — includes padding in denominator)
// ============================================================

/// Process 4 output rows × 8 output columns with FMA accumulation + scale by 1/kernel_area.
/// Called only for interior valid region (SW == 1 required).
inline void avgpool_h4_simd(
    float* output, const float* input,
    int64_t KD, int64_t KH, int64_t KW,
    int64_t SH,
    int64_t DD, int64_t DH, int64_t DW,
    int64_t ow_simd_count,
    int64_t out_h_stride, int64_t out_w_stride,
    int64_t in_d_stride, int64_t in_h_stride,
    float scale, bool add_to)
{
    const v_fp32x8 vscale = set1_fp32x8(scale);

    for (int64_t blk = 0; blk < ow_simd_count; ++blk) {
        const int64_t w_off = blk * 8;

        v_fp32x8 vacc0 = zero_fp32x8();
        v_fp32x8 vacc1 = zero_fp32x8();
        v_fp32x8 vacc2 = zero_fp32x8();
        v_fp32x8 vacc3 = zero_fp32x8();

        for (int64_t kd = 0; kd < KD; ++kd) {
            const int64_t d_off = kd * DD * in_d_stride;
            for (int64_t kh = 0; kh < KH; ++kh) {
                const int64_t h_off0 = (0 * SH + kh * DH) * in_h_stride;
                const int64_t h_off1 = (1 * SH + kh * DH) * in_h_stride;
                const int64_t h_off2 = (2 * SH + kh * DH) * in_h_stride;
                const int64_t h_off3 = (3 * SH + kh * DH) * in_h_stride;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    const int64_t kw_off = w_off + kw * DW;
                    vacc0 = fmadd(load_fp32x8(input + d_off + h_off0 + kw_off), vscale, vacc0);
                    vacc1 = fmadd(load_fp32x8(input + d_off + h_off1 + kw_off), vscale, vacc1);
                    vacc2 = fmadd(load_fp32x8(input + d_off + h_off2 + kw_off), vscale, vacc2);
                    vacc3 = fmadd(load_fp32x8(input + d_off + h_off3 + kw_off), vscale, vacc3);
                }
            }
        }

        float* out0 = output + 0 * out_h_stride + w_off * out_w_stride;
        float* out1 = output + 1 * out_h_stride + w_off * out_w_stride;
        float* out2 = output + 2 * out_h_stride + w_off * out_w_stride;
        float* out3 = output + 3 * out_h_stride + w_off * out_w_stride;

        if (add_to) {
            vacc0 = add(vacc0, load_fp32x8(out0));
            vacc1 = add(vacc1, load_fp32x8(out1));
            vacc2 = add(vacc2, load_fp32x8(out2));
            vacc3 = add(vacc3, load_fp32x8(out3));
        }

        store(out0, vacc0);
        store(out1, vacc1);
        store(out2, vacc2);
        store(out3, vacc3);
    }
}

/// Process 1 output row × 8 output columns with FMA accumulation + scale.
/// Called only for interior valid region (SW == 1 required).
inline void avgpool_h1_simd(
    float* output, const float* input,
    int64_t KD, int64_t KH, int64_t KW,
    int64_t SH,
    int64_t DD, int64_t DH, int64_t DW,
    int64_t ow_simd_count,
    int64_t out_h_stride, int64_t out_w_stride,
    int64_t in_d_stride, int64_t in_h_stride,
    float scale, bool add_to)
{
    const v_fp32x8 vscale = set1_fp32x8(scale);

    for (int64_t blk = 0; blk < ow_simd_count; ++blk) {
        const int64_t w_off = blk * 8;
        v_fp32x8 vacc = zero_fp32x8();

        for (int64_t kd = 0; kd < KD; ++kd) {
            const int64_t d_off = kd * DD * in_d_stride;
            for (int64_t kh = 0; kh < KH; ++kh) {
                const int64_t h_off = kh * DH * in_h_stride;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    const int64_t kw_off = w_off + kw * DW;
                    vacc = fmadd(load_fp32x8(input + d_off + h_off + kw_off), vscale, vacc);
                }
            }
        }

        float* out_r = output + w_off * out_w_stride;

        if (add_to) {
            vacc = add(vacc, load_fp32x8(out_r));
        }

        store(out_r, vacc);
    }
}

}  // anonymous namespace

// ============================================================
// Main entry point
// ============================================================

void pooling_cpu(const PoolingAttributes& attrs,
                  const TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx,
                  void* /*workspace*/)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();
    const int64_t srank = PoolingAttributes::spatial_rank(rank);  // 2 or 3

    // ----- Common dimensions -----
    const int64_t N = input.shape(0);
    const int64_t C = input.shape(1);

    // ----- Spatial input dimensions -----
    const int64_t ID = (srank == 3) ? input.shape(2) : 1;
    const int64_t IH = input.shape(srank);
    const int64_t IW = input.shape(srank + 1);

    // ----- Spatial output dimensions -----
    const int64_t OD = (srank == 3) ? output.shape(2) : 1;
    const int64_t OH = output.shape(srank);
    const int64_t OW = output.shape(srank + 1);

    // ----- Kernel / stride / padding / dilation -----
    // Layout [KD, KH, KW]; KH always at [1], KW always at [2].
    const int64_t KD = (srank == 3) ? attrs.kernel_shape[0] : 1;
    const int64_t KH = attrs.kernel_shape[1];
    const int64_t KW = attrs.kernel_shape[2];

    const int64_t SD = (srank == 3) ? attrs.stride[0] : 1;
    const int64_t SH = attrs.stride[1];
    const int64_t SW = attrs.stride[2];

    const int64_t DD_ = (srank == 3) ? attrs.dilation[0] : 1;
    const int64_t DH = attrs.dilation[1];
    const int64_t DW = attrs.dilation[2];

    const int64_t PD = (srank == 3) ? attrs.padding[0] : 0;
    const int64_t PH = attrs.padding[1];
    const int64_t PW = attrs.padding[2];

    auto* out_ptr = output.data_as<float>();
    const auto* in_ptr = input.data_as<float>();

    // ----- Strides for dense NCHW/NCDHW layout -----
    const int64_t in_ch_stride  = ID * IH * IW;
    const int64_t in_d_stride   = IH * IW;
    const int64_t in_h_stride   = IW;
    const int64_t out_ch_stride = OD * OH * OW;
    const int64_t out_d_stride  = OH * OW;
    const int64_t out_h_stride  = OW;
    const int64_t out_w_stride  = 1;

    // ----- SIMD gating -----
    // Max and Average (includes pad) support SIMD via contiguous load.
    // AverageExcludePad and Lp require per-element counting → scalar only.
    // stride_w == 1 is required for contiguous input loads (nnops has no gather ops).
    const bool pool_supports_simd =
        (attrs.type == PoolingType::Max || attrs.type == PoolingType::Average);
    const bool use_simd = pool_supports_simd && (SW == 1);

    // ----- Interior valid region computation -----
    // The "interior" output region is where every kernel element maps to a valid
    // input position. Pad regions (where some kernel elements fall outside input)
    // are handled by the scalar path.
    //
    // Formula: first output position where kernel start is >= 0 (after padding)
    //   beg = ceil(P / S)
    // Last output position where kernel end is within input
    //   end = max(ceil((I + P - ((K-1)*D + 1)) / S), beg)

    auto compute_beg = [](int64_t pad, int64_t stride) -> int64_t {
        if (stride <= 0) return 0;
        return static_cast<int64_t>(
            std::ceil(static_cast<float>(pad) / static_cast<float>(stride)));
    };

    auto compute_end = [](int64_t input_extent, int64_t pad, int64_t kernel_extent,
                           int64_t stride, int64_t beg) -> int64_t {
        if (stride <= 0) return input_extent;
        const float kernel_span = static_cast<float>(kernel_extent);
        const float num = static_cast<float>(input_extent) + static_cast<float>(pad) - kernel_span;
        return std::max(static_cast<int64_t>(
            std::ceil(num / static_cast<float>(stride))), beg);
    };

    // Valid output depth range (3D only; for 2D: od_beg=0, od_end=1)
    const int64_t od_beg = compute_beg(PD, SD);
    const int64_t od_end = compute_end(ID, PD, (KD - 1) * DD_ + 1, SD, od_beg);

    // Valid output height range
    const int64_t oh_beg = compute_beg(PH, SH);
    const int64_t oh_end = compute_end(IH, PH, (KH - 1) * DH + 1, SH, oh_beg);

    // Valid output width range
    const int64_t ow_beg = compute_beg(PW, SW);
    const int64_t ow_end = compute_end(IW, PW, (KW - 1) * DW + 1, SW, ow_beg);

    // Clamp to output bounds
    const int64_t od_beg_c = std::max<int64_t>(0, std::min(od_beg, OD));
    const int64_t od_end_c = std::max<int64_t>(od_beg_c, std::min(od_end, OD));
    const int64_t oh_beg_c = std::max<int64_t>(0, std::min(oh_beg, OH));
    const int64_t oh_end_c = std::max<int64_t>(oh_beg_c, std::min(oh_end, OH));
    const int64_t ow_beg_c = std::max<int64_t>(0, std::min(ow_beg, OW));
    const int64_t ow_end_c = std::max<int64_t>(ow_beg_c, std::min(ow_end, OW));

    // SIMD-aligned interior OW range (8-wide)
    const int64_t ow_simd_beg = ow_beg_c;
    const int64_t ow_simd_end = use_simd
        ? ow_simd_beg + ((ow_end_c - ow_simd_beg) / 8) * 8
        : ow_beg_c;

    // AvgPool scale (pre-computed for SIMD kernel)
    const float avg_scale = 1.0f / static_cast<float>(KD * KH * KW);

    // ----- Per-sample compute lambda -----
    const auto compute_sample = [&](int64_t n) {
        for (int64_t c = 0; c < C; ++c) {
            const float* in_ch  = in_ptr + n * C * in_ch_stride + c * in_ch_stride;
            float* out_ch = out_ptr + n * C * out_ch_stride + c * out_ch_stride;

            // ================================================
            // Pad-Depth Front (3D only; empty for 2D)
            // ================================================
            for (int64_t od = 0; od < od_beg_c; ++od) {
                float* out_d = out_ch + od * out_d_stride;
                for (int64_t oh = 0; oh < OH; ++oh) {
                    pooling_scalar_row(
                        out_d + oh * out_h_stride, in_ch,
                        ID, IH, IW, od, oh, 0, OW,
                        KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                        out_w_stride, in_d_stride, in_h_stride,
                        attrs.type, attrs.add_to, attrs.p_norm);
                }
            }

            // ================================================
            // Interior Depth (SIMD-eligible for Max/Avg with SW==1)
            // ================================================
            for (int64_t od = od_beg_c; od < od_end_c; ++od) {
                float* out_d = out_ch + od * out_d_stride;

                // Base depth position in input (valid in interior region)
                const int64_t id_base = od * SD - PD;

                // ---- Pad-Top (scalar) ----
                for (int64_t oh = 0; oh < oh_beg_c; ++oh) {
                    pooling_scalar_row(
                        out_d + oh * out_h_stride, in_ch,
                        ID, IH, IW, od, oh, 0, OW,
                        KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                        out_w_stride, in_d_stride, in_h_stride,
                        attrs.type, attrs.add_to, attrs.p_norm);
                }

                // ---- Interior Height ----
                if (use_simd) {
                    // --- h4 blocks (4 rows at a time) ---
                    int64_t oh = oh_beg_c;
                    for (; oh + 3 < oh_end_c; oh += 4) {
                        const int64_t ih_base = oh * SH - PH;  // valid in interior

                        // Pad-left (per-row scalar)
                        if (ow_beg_c > 0) {
                            for (int64_t r = 0; r < 4; ++r) {
                                pooling_scalar_row(
                                    out_d + (oh + r) * out_h_stride, in_ch,
                                    ID, IH, IW, od, oh + r, 0, ow_beg_c,
                                    KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                                    out_w_stride, in_d_stride, in_h_stride,
                                    attrs.type, attrs.add_to, attrs.p_norm);
                            }
                        }

                        // SIMD h4 interior (8-wide blocks)
                        const int64_t ow_simd_cnt = (ow_simd_end - ow_simd_beg) / 8;
                        if (ow_simd_cnt > 0) {
                            float* out_simd = out_d + oh * out_h_stride
                                            + ow_simd_beg * out_w_stride;
                            const float* in_simd = in_ch
                                + id_base * in_d_stride
                                + ih_base * in_h_stride
                                + (ow_simd_beg - PW);

                            if (attrs.type == PoolingType::Max) {
                                maxpool_h4_simd(out_simd, in_simd,
                                                KD, KH, KW, SH,
                                                DD_, DH, DW,
                                                ow_simd_cnt,
                                                out_h_stride, out_w_stride,
                                                in_d_stride, in_h_stride,
                                                attrs.add_to);
                            } else {
                                avgpool_h4_simd(out_simd, in_simd,
                                                KD, KH, KW, SH,
                                                DD_, DH, DW,
                                                ow_simd_cnt,
                                                out_h_stride, out_w_stride,
                                                in_d_stride, in_h_stride,
                                                avg_scale, attrs.add_to);
                            }
                        }

                        // SIMD tail (interior columns not 8-aligned)
                        if (ow_simd_end < ow_end_c) {
                            for (int64_t r = 0; r < 4; ++r) {
                                pooling_scalar_row(
                                    out_d + (oh + r) * out_h_stride, in_ch,
                                    ID, IH, IW, od, oh + r, ow_simd_end, ow_end_c,
                                    KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                                    out_w_stride, in_d_stride, in_h_stride,
                                    attrs.type, attrs.add_to, attrs.p_norm);
                            }
                        }

                        // Pad-right (per-row scalar)
                        if (ow_end_c < OW) {
                            for (int64_t r = 0; r < 4; ++r) {
                                pooling_scalar_row(
                                    out_d + (oh + r) * out_h_stride, in_ch,
                                    ID, IH, IW, od, oh + r, ow_end_c, OW,
                                    KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                                    out_w_stride, in_d_stride, in_h_stride,
                                    attrs.type, attrs.add_to, attrs.p_norm);
                            }
                        }
                    }

                    // --- h1 blocks (remaining interior rows) ---
                    for (; oh < oh_end_c; ++oh) {
                        const int64_t ih_base = oh * SH - PH;

                        // Pad-left
                        if (ow_beg_c > 0) {
                            pooling_scalar_row(
                                out_d + oh * out_h_stride, in_ch,
                                ID, IH, IW, od, oh, 0, ow_beg_c,
                                KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                                out_w_stride, in_d_stride, in_h_stride,
                                attrs.type, attrs.add_to, attrs.p_norm);
                        }

                        // SIMD h1 interior
                        const int64_t ow_simd_cnt = (ow_simd_end - ow_simd_beg) / 8;
                        if (ow_simd_cnt > 0) {
                            float* out_simd = out_d + oh * out_h_stride
                                            + ow_simd_beg * out_w_stride;
                            const float* in_simd = in_ch
                                + id_base * in_d_stride
                                + ih_base * in_h_stride
                                + (ow_simd_beg - PW);

                            if (attrs.type == PoolingType::Max) {
                                maxpool_h1_simd(out_simd, in_simd,
                                                KD, KH, KW, SH,
                                                DD_, DH, DW,
                                                ow_simd_cnt,
                                                out_h_stride, out_w_stride,
                                                in_d_stride, in_h_stride,
                                                attrs.add_to);
                            } else {
                                avgpool_h1_simd(out_simd, in_simd,
                                                KD, KH, KW, SH,
                                                DD_, DH, DW,
                                                ow_simd_cnt,
                                                out_h_stride, out_w_stride,
                                                in_d_stride, in_h_stride,
                                                avg_scale, attrs.add_to);
                            }
                        }

                        // SIMD tail
                        if (ow_simd_end < ow_end_c) {
                            pooling_scalar_row(
                                out_d + oh * out_h_stride, in_ch,
                                ID, IH, IW, od, oh, ow_simd_end, ow_end_c,
                                KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                                out_w_stride, in_d_stride, in_h_stride,
                                attrs.type, attrs.add_to, attrs.p_norm);
                        }

                        // Pad-right
                        if (ow_end_c < OW) {
                            pooling_scalar_row(
                                out_d + oh * out_h_stride, in_ch,
                                ID, IH, IW, od, oh, ow_end_c, OW,
                                KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                                out_w_stride, in_d_stride, in_h_stride,
                                attrs.type, attrs.add_to, attrs.p_norm);
                        }
                    }
                } else {
                    // ---- Non-SIMD interior: all-scalar fallback ----
                    for (int64_t oh = oh_beg_c; oh < oh_end_c; ++oh) {
                        pooling_scalar_row(
                            out_d + oh * out_h_stride, in_ch,
                            ID, IH, IW, od, oh, 0, OW,
                            KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                            out_w_stride, in_d_stride, in_h_stride,
                            attrs.type, attrs.add_to, attrs.p_norm);
                    }
                }

                // ---- Pad-Bottom (scalar) ----
                for (int64_t oh = oh_end_c; oh < OH; ++oh) {
                    pooling_scalar_row(
                        out_d + oh * out_h_stride, in_ch,
                        ID, IH, IW, od, oh, 0, OW,
                        KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                        out_w_stride, in_d_stride, in_h_stride,
                        attrs.type, attrs.add_to, attrs.p_norm);
                }
            }

            // ================================================
            // Pad-Depth Back (3D only; empty for 2D)
            // ================================================
            for (int64_t od = od_end_c; od < OD; ++od) {
                float* out_d = out_ch + od * out_d_stride;
                for (int64_t oh = 0; oh < OH; ++oh) {
                    pooling_scalar_row(
                        out_d + oh * out_h_stride, in_ch,
                        ID, IH, IW, od, oh, 0, OW,
                        KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                        out_w_stride, in_d_stride, in_h_stride,
                        attrs.type, attrs.add_to, attrs.p_norm);
                }
            }
        }
    };

    // ----- Parallel dispatch (per batch sample) -----
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
