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
///   3. Width-SIMD via v_f32x8/v_f16x8 (simd_lane_for<T>-wide; AVX2 native, SSE/NEON emulated)
///   4. SW == 1: v_load (contiguous). SW == 2: v_load_even (stride-2 gather via LD2/UNPCK).
///   5. AverageExcludePad and Lp pooling: scalar-only (per-element counting needed)
///   6. Unified 2D/3D code path: KD=1/SD=1/DD=1/PD=0/ID=1/OD=1 for 2D
///   7. add_to support: branch-hoisted; one extra load per v_store when enabled
///   8. Per-(sample,channel) parallel dispatch via parallel_for (N*C tasks)

#include "nnops/ops/pooling.hpp"
#include "nnops/detail/assert.hpp"
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
/// Templated on T (float for f32, half for f16).
template <typename T>
inline void pooling_scalar_row(
    T* output,              // output row pointer: out_d + oh * out_row_stride
    const T* input,         // channel base pointer: in_ch
    int64_t ID, int64_t IH, int64_t IW,
    int64_t od, int64_t oh,
    int64_t ow_start, int64_t ow_end,
    int64_t KD, int64_t KH, int64_t KW,
    int64_t SD, int64_t SH, int64_t SW,
    int64_t DD, int64_t DH, int64_t DW,
    int64_t PD, int64_t PH, int64_t PW,
    int64_t out_w_stride,
    int64_t in_d_stride, int64_t in_row_stride,
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
                if (id < 0 || id >= ID) { continue; }
                for (int64_t kh = 0; kh < KH; ++kh) {
                    const int64_t ih = oh * SH + kh * DH - PH;
                    if (ih < 0 || ih >= IH) { continue; }
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        const int64_t iw = ow * SW + kw * DW - PW;
                        if (iw < 0 || iw >= IW) { continue; }
                        const float val = s_load(&input[id * in_d_stride + ih * in_row_stride + iw]);
                        if (val > max_val) { max_val = val; }
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
                        sum += s_load(&input[id * in_d_stride + ih * in_row_stride + iw]);
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
                if (id < 0 || id >= ID) { continue; }
                for (int64_t kh = 0; kh < KH; ++kh) {
                    const int64_t ih = oh * SH + kh * DH - PH;
                    if (ih < 0 || ih >= IH) { continue; }
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        const int64_t iw = ow * SW + kw * DW - PW;
                        if (iw < 0 || iw >= IW) { continue; }
                        sum += std::pow(
                            std::abs(s_load(&input[id * in_d_stride + ih * in_row_stride + iw])), fp);
                    }
                }
            }
            result = std::pow(sum, 1.0f / fp);
            break;
        }
        }

        // Write back
        const int64_t out_idx = ow * out_w_stride;
        if (add_to) {
            s_store(&output[out_idx], s_load(&output[out_idx]) + result);
        } else {
            s_store(&output[out_idx], result);
        }
    }
}

// ============================================================
// MaxPooling SIMD kernels
// ============================================================
//
// Pointer conventions (PRE-POSITIONED by caller):
//   output: out_d + oh_start * out_row_stride + ow_start * out_w_stride
//   input:  in_ch
//           + (od * SD - PD) * in_d_stride     ← base depth position
//           + (oh_start * SH - PH) * in_row_stride ← base height position
//           + (ow_start - PW)                   ← base width position (SW == 1)
//
// Relative offsets used inside the kernel (caller guarantees validity):
//   Depth:  kd * DD * in_d_stride
//   Height: (r * SH + kh * DH) * in_row_stride    (r = 0..3 for h4, 0 for h1)
//   Width:  (ow - ow_start) + kw * DW           (SW == 1, in_w_stride == 1, ow loops)

/// Process 4 output rows × ow_count output columns with element-wise v_max.
/// ow_count must be a multiple of 8 (caller guarantees SIMD-aligned interior region).
/// SW_val: 1 = v_load contiguous; 2 = v_load_even stride-2 gather.
template <typename T, int SW_val>
inline void maxpool_h4_simd(
    T* output, const T* input,
    int64_t KD, int64_t KH, int64_t KW,
    int64_t SH,
    int64_t DD, int64_t DH, int64_t DW,
    int64_t ow_count,
    int64_t out_row_stride, int64_t out_w_stride,
    int64_t in_d_stride, int64_t in_row_stride,
    float /*scale*/,
    bool add_to)
{

    constexpr float neg_inf = -std::numeric_limits<float>::infinity();
    const auto vinit = v_set1(input, neg_inf);

    for (int64_t ow = 0; ow < ow_count; ow += simd_lane_for<T>) {
        auto vacc0 = vinit;
        auto vacc1 = vinit;
        auto vacc2 = vinit;
        auto vacc3 = vinit;

        for (int64_t kd = 0; kd < KD; ++kd) {
            const int64_t d_off = kd * DD * in_d_stride;
            for (int64_t kh = 0; kh < KH; ++kh) {
                const int64_t h_off0 = (0 * SH + kh * DH) * in_row_stride;
                const int64_t h_off1 = (1 * SH + kh * DH) * in_row_stride;
                const int64_t h_off2 = (2 * SH + kh * DH) * in_row_stride;
                const int64_t h_off3 = (3 * SH + kh * DH) * in_row_stride;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    const int64_t kw_off = ow * SW_val + kw * DW;
                    if constexpr (SW_val == 2) {
                        vacc0 = v_max(vacc0, v_load_even(input + d_off + h_off0 + kw_off));
                        vacc1 = v_max(vacc1, v_load_even(input + d_off + h_off1 + kw_off));
                        vacc2 = v_max(vacc2, v_load_even(input + d_off + h_off2 + kw_off));
                        vacc3 = v_max(vacc3, v_load_even(input + d_off + h_off3 + kw_off));
                    } else {
                        vacc0 = v_max(vacc0, v_load(input + d_off + h_off0 + kw_off));
                        vacc1 = v_max(vacc1, v_load(input + d_off + h_off1 + kw_off));
                        vacc2 = v_max(vacc2, v_load(input + d_off + h_off2 + kw_off));
                        vacc3 = v_max(vacc3, v_load(input + d_off + h_off3 + kw_off));
                    }
                }
            }
        }

        T* out0 = output + 0 * out_row_stride + ow * out_w_stride;
        T* out1 = output + 1 * out_row_stride + ow * out_w_stride;
        T* out2 = output + 2 * out_row_stride + ow * out_w_stride;
        T* out3 = output + 3 * out_row_stride + ow * out_w_stride;

        if (add_to) {
            vacc0 = v_add(vacc0, v_load(out0));
            vacc1 = v_add(vacc1, v_load(out1));
            vacc2 = v_add(vacc2, v_load(out2));
            vacc3 = v_add(vacc3, v_load(out3));
        }

        v_store(out0, vacc0);
        v_store(out1, vacc1);
        v_store(out2, vacc2);
        v_store(out3, vacc3);
    }
}

/// Process 1 output row × ow_count output columns with element-wise v_max.
/// ow_count must be a multiple of 8 (caller guarantees SIMD-aligned interior region).
/// SW_val: 1 = v_load contiguous; 2 = v_load_even stride-2 gather.
template <typename T, int SW_val>
inline void maxpool_h1_simd(
    T* output, const T* input,
    int64_t KD, int64_t KH, int64_t KW,
    int64_t SH,
    int64_t DD, int64_t DH, int64_t DW,
    int64_t ow_count,
    int64_t out_row_stride, int64_t out_w_stride,
    int64_t in_d_stride, int64_t in_row_stride,
    float /*scale*/,
    bool add_to)
{

    constexpr float neg_inf = -std::numeric_limits<float>::infinity();
    const auto vinit = v_set1(input, neg_inf);

    for (int64_t ow = 0; ow < ow_count; ow += simd_lane_for<T>) {
        auto vacc = vinit;

        for (int64_t kd = 0; kd < KD; ++kd) {
            const int64_t d_off = kd * DD * in_d_stride;
            for (int64_t kh = 0; kh < KH; ++kh) {
                const int64_t h_off = kh * DH * in_row_stride;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    const int64_t kw_off = ow * SW_val + kw * DW;
                    if constexpr (SW_val == 2) {
                        vacc = v_max(vacc, v_load_even(input + d_off + h_off + kw_off));
                    } else {
                        vacc = v_max(vacc, v_load(input + d_off + h_off + kw_off));
                    }
                }
            }
        }

        T* out_r = output + ow * out_w_stride;

        if (add_to) {
            vacc = v_add(vacc, v_load(out_r));
        }

        v_store(out_r, vacc);
    }
}

// ============================================================
// AvgPooling SIMD kernels (Average — includes padding in denominator)
// ============================================================

/// Process 4 output rows × ow_count output columns with FMA accumulation + scale by 1/kernel_area.
/// ow_count must be a multiple of 8 (caller guarantees SIMD-aligned interior region).
/// SW_val: 1 = v_load contiguous; 2 = v_load_even stride-2 gather.
template <typename T, int SW_val>
inline void avgpool_h4_simd(
    T* output, const T* input,
    int64_t KD, int64_t KH, int64_t KW,
    int64_t SH,
    int64_t DD, int64_t DH, int64_t DW,
    int64_t ow_count,
    int64_t out_row_stride, int64_t out_w_stride,
    int64_t in_d_stride, int64_t in_row_stride,
    float scale, bool add_to)
{

    const auto vscale = v_set1(input, scale);
    const auto vzero = v_set1(input, 0.0f);

    for (int64_t ow = 0; ow < ow_count; ow += simd_lane_for<T>) {
        auto vacc0 = vzero;
        auto vacc1 = vzero;
        auto vacc2 = vzero;
        auto vacc3 = vzero;

        for (int64_t kd = 0; kd < KD; ++kd) {
            const int64_t d_off = kd * DD * in_d_stride;
            for (int64_t kh = 0; kh < KH; ++kh) {
                const int64_t h_off0 = (0 * SH + kh * DH) * in_row_stride;
                const int64_t h_off1 = (1 * SH + kh * DH) * in_row_stride;
                const int64_t h_off2 = (2 * SH + kh * DH) * in_row_stride;
                const int64_t h_off3 = (3 * SH + kh * DH) * in_row_stride;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    const int64_t kw_off = ow * SW_val + kw * DW;
                    if constexpr (SW_val == 2) {
                        vacc0 = v_fmadd(v_load_even(input + d_off + h_off0 + kw_off), vscale, vacc0);
                        vacc1 = v_fmadd(v_load_even(input + d_off + h_off1 + kw_off), vscale, vacc1);
                        vacc2 = v_fmadd(v_load_even(input + d_off + h_off2 + kw_off), vscale, vacc2);
                        vacc3 = v_fmadd(v_load_even(input + d_off + h_off3 + kw_off), vscale, vacc3);
                    } else {
                        vacc0 = v_fmadd(v_load(input + d_off + h_off0 + kw_off), vscale, vacc0);
                        vacc1 = v_fmadd(v_load(input + d_off + h_off1 + kw_off), vscale, vacc1);
                        vacc2 = v_fmadd(v_load(input + d_off + h_off2 + kw_off), vscale, vacc2);
                        vacc3 = v_fmadd(v_load(input + d_off + h_off3 + kw_off), vscale, vacc3);
                    }
                }
            }
        }

        T* out0 = output + 0 * out_row_stride + ow * out_w_stride;
        T* out1 = output + 1 * out_row_stride + ow * out_w_stride;
        T* out2 = output + 2 * out_row_stride + ow * out_w_stride;
        T* out3 = output + 3 * out_row_stride + ow * out_w_stride;

        if (add_to) {
            vacc0 = v_add(vacc0, v_load(out0));
            vacc1 = v_add(vacc1, v_load(out1));
            vacc2 = v_add(vacc2, v_load(out2));
            vacc3 = v_add(vacc3, v_load(out3));
        }

        v_store(out0, vacc0);
        v_store(out1, vacc1);
        v_store(out2, vacc2);
        v_store(out3, vacc3);
    }
}

/// Process 1 output row × ow_count output columns with FMA accumulation + scale.
/// ow_count must be a multiple of 8 (caller guarantees SIMD-aligned interior region).
/// SW_val: 1 = v_load contiguous; 2 = v_load_even stride-2 gather.
template <typename T, int SW_val>
inline void avgpool_h1_simd(
    T* output, const T* input,
    int64_t KD, int64_t KH, int64_t KW,
    int64_t SH,
    int64_t DD, int64_t DH, int64_t DW,
    int64_t ow_count,
    int64_t out_row_stride, int64_t out_w_stride,
    int64_t in_d_stride, int64_t in_row_stride,
    float scale, bool add_to)
{

    const auto vscale = v_set1(input, scale);
    const auto vzero = v_set1(input, 0.0f);

    for (int64_t ow = 0; ow < ow_count; ow += simd_lane_for<T>) {
        auto vacc = vzero;

        for (int64_t kd = 0; kd < KD; ++kd) {
            const int64_t d_off = kd * DD * in_d_stride;
            for (int64_t kh = 0; kh < KH; ++kh) {
                const int64_t h_off = kh * DH * in_row_stride;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    const int64_t kw_off = ow * SW_val + kw * DW;
                    if constexpr (SW_val == 2) {
                        vacc = v_fmadd(v_load_even(input + d_off + h_off + kw_off), vscale, vacc);
                    } else {
                        vacc = v_fmadd(v_load(input + d_off + h_off + kw_off), vscale, vacc);
                    }
                }
            }
        }

        T* out_r = output + ow * out_w_stride;

        if (add_to) {
            vacc = v_add(vacc, v_load(out_r));
        }

        v_store(out_r, vacc);
    }
}

}  // anonymous namespace

// ============================================================
// Main pooling implementation (templated on data type T)
// ============================================================

template <typename T>
void pooling_impl(const PoolingAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx)
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

    auto* out_ptr = output.ptr<T>();
    const auto* in_ptr = input.ptr<T>();

    // ----- Strides derived from pitch (row pitch in bytes → element stride) -----
    const int64_t in_row_stride   = input.row_stride_elems();   // elements per input row (>= IW)
    const int64_t in_d_stride     = IH * in_row_stride;          // elements per depth slice
    const int64_t in_ch_stride    = ID * in_d_stride;            // elements per channel
    const int64_t out_row_stride  = output.row_stride_elems();   // elements per output row (>= OW)
    const int64_t out_d_stride    = OH * out_row_stride;         // elements per output depth slice
    const int64_t out_ch_stride   = OD * out_d_stride;           // elements per output channel
    const int64_t out_w_stride    = 1;

    // ----- SIMD gating -----
    // Max and Average (includes pad) support SIMD via contiguous load.
    // AverageExcludePad and Lp require per-element counting → scalar only.
    // SW == 1: v_load reads 8 consecutive input elements.
    // SW == 2: v_load_even reads 8 even-indexed input elements (16-span, stride-2).
    const bool pool_supports_simd =
        (attrs.type == PoolingType::Max || attrs.type == PoolingType::Average);
    const bool use_simd = pool_supports_simd && (SW == 1 || SW == 2);

    // ----- Interior valid region computation -----
    // The "interior" output region is where every kernel element maps to a valid
    // input position. Pad regions (where some kernel elements fall outside input)
    // are handled by the scalar path.
    //
    // Formula: first output position where kernel start is >= 0 (after padding)
    //   beg = ceil(P / S)
    // Last output position where kernel end is within input
    //   end = v_max(ceil((I + P - ((K-1)*D + 1)) / S), beg)

    auto compute_beg = [](int64_t pad, int64_t stride) -> int64_t {
        if (stride <= 0) { return 0; }
        return static_cast<int64_t>(
            std::ceil(static_cast<float>(pad) / static_cast<float>(stride)));
    };

    auto compute_end = [](int64_t input_extent, int64_t pad, int64_t kernel_extent,
                           int64_t stride, int64_t beg) -> int64_t {
        if (stride <= 0) { return input_extent; }
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

    // SIMD-aligned interior OW range (simd_lane_for<T>-wide)
    const int64_t ow_simd_beg = ow_beg_c;
    const int64_t ow_simd_end = use_simd
        ? ow_simd_beg + ((ow_end_c - ow_simd_beg) / simd_lane_for<T>) * simd_lane_for<T>
        : ow_beg_c;

    // AvgPool scale (pre-computed for SIMD kernel)
    const float avg_scale = 1.0f / static_cast<float>(KD * KH * KW);

    // ----- Function pointer selection: hoist SW+type dispatch out of hot loops -----
    // Max and Avg kernels share the same signature (Max ignores the scale parameter).
    using PoolingFn = void (*)(T*, const T*,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
        int64_t, int64_t, int64_t, int64_t, int64_t,
        float, bool);

    PoolingFn pool_h4_fn, pool_h1_fn;
    if (attrs.type == PoolingType::Max) {
        pool_h4_fn = (SW == 2) ? maxpool_h4_simd<T, 2> : maxpool_h4_simd<T, 1>;
        pool_h1_fn = (SW == 2) ? maxpool_h1_simd<T, 2> : maxpool_h1_simd<T, 1>;
    } else {
        pool_h4_fn = (SW == 2) ? avgpool_h4_simd<T, 2> : avgpool_h4_simd<T, 1>;
        pool_h1_fn = (SW == 2) ? avgpool_h1_simd<T, 2> : avgpool_h1_simd<T, 1>;
    }

    // ----- Per-channel compute lambda (N*C parallel) -----
    const auto compute_channel = [&](int64_t n, int64_t c) {
        const T* in_ch  = in_ptr + n * C * in_ch_stride + c * in_ch_stride;
        T* out_ch = out_ptr + n * C * out_ch_stride + c * out_ch_stride;

        // ================================================
        // Pad-Depth Front (3D only; empty for 2D)
        // ================================================
        for (int64_t od = 0; od < od_beg_c; ++od) {
            T* out_d = out_ch + od * out_d_stride;
            for (int64_t oh = 0; oh < OH; ++oh) {
                pooling_scalar_row(
                    out_d + oh * out_row_stride, in_ch,
                    ID, IH, IW, od, oh, 0, OW,
                    KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                    out_w_stride, in_d_stride, in_row_stride,
                    attrs.type, attrs.add_to, attrs.p_norm);
            }
        }

        // ================================================
        // Interior Depth (SIMD-eligible for Max/Avg with SW==1)
        // ================================================
        for (int64_t od = od_beg_c; od < od_end_c; ++od) {
            T* out_d = out_ch + od * out_d_stride;

            // Base depth position in input (valid in interior region)
            const int64_t id_base = od * SD - PD;

            // ---- Pad-Top (scalar) ----
            for (int64_t oh = 0; oh < oh_beg_c; ++oh) {
                pooling_scalar_row(
                    out_d + oh * out_row_stride, in_ch,
                    ID, IH, IW, od, oh, 0, OW,
                    KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                    out_w_stride, in_d_stride, in_row_stride,
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
                                out_d + (oh + r) * out_row_stride, in_ch,
                                ID, IH, IW, od, oh + r, 0, ow_beg_c,
                                KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                                out_w_stride, in_d_stride, in_row_stride,
                                attrs.type, attrs.add_to, attrs.p_norm);
                        }
                    }

                    // SIMD h4 interior (simd_lane_for<T>-wide blocks)
                    const int64_t ow_simd_elems = ow_simd_end - ow_simd_beg;
                    if (ow_simd_elems > 0) {
                        T* out_simd = out_d + oh * out_row_stride
                                    + ow_simd_beg * out_w_stride;
                        const T* in_simd = in_ch
                            + id_base * in_d_stride
                            + ih_base * in_row_stride
                            + (ow_simd_beg * SW - PW);   // ow * SW for general stride

                        pool_h4_fn(out_simd, in_simd,
                                   KD, KH, KW, SH,
                                   DD_, DH, DW,
                                   ow_simd_elems,
                                   out_row_stride, out_w_stride,
                                   in_d_stride, in_row_stride,
                                   avg_scale, attrs.add_to);
                    }

                    // SIMD tail (interior columns not 8-aligned)
                    if (ow_simd_end < ow_end_c) {
                        for (int64_t r = 0; r < 4; ++r) {
                            pooling_scalar_row(
                                out_d + (oh + r) * out_row_stride, in_ch,
                                ID, IH, IW, od, oh + r, ow_simd_end, ow_end_c,
                                KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                                out_w_stride, in_d_stride, in_row_stride,
                                attrs.type, attrs.add_to, attrs.p_norm);
                        }
                    }

                    // Pad-right (per-row scalar)
                    if (ow_end_c < OW) {
                        for (int64_t r = 0; r < 4; ++r) {
                            pooling_scalar_row(
                                out_d + (oh + r) * out_row_stride, in_ch,
                                ID, IH, IW, od, oh + r, ow_end_c, OW,
                                KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                                out_w_stride, in_d_stride, in_row_stride,
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
                            out_d + oh * out_row_stride, in_ch,
                            ID, IH, IW, od, oh, 0, ow_beg_c,
                            KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                            out_w_stride, in_d_stride, in_row_stride,
                            attrs.type, attrs.add_to, attrs.p_norm);
                    }

                    // SIMD h1 interior
                    const int64_t ow_simd_elems = ow_simd_end - ow_simd_beg;
                    if (ow_simd_elems > 0) {
                        T* out_simd = out_d + oh * out_row_stride
                                    + ow_simd_beg * out_w_stride;
                        const T* in_simd = in_ch
                            + id_base * in_d_stride
                            + ih_base * in_row_stride
                            + (ow_simd_beg * SW - PW);   // ow * SW for general stride

                        pool_h1_fn(out_simd, in_simd,
                                   KD, KH, KW, SH,
                                   DD_, DH, DW,
                                   ow_simd_elems,
                                   out_row_stride, out_w_stride,
                                   in_d_stride, in_row_stride,
                                   avg_scale, attrs.add_to);
                    }

                    // SIMD tail
                    if (ow_simd_end < ow_end_c) {
                        pooling_scalar_row(
                            out_d + oh * out_row_stride, in_ch,
                            ID, IH, IW, od, oh, ow_simd_end, ow_end_c,
                            KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                            out_w_stride, in_d_stride, in_row_stride,
                            attrs.type, attrs.add_to, attrs.p_norm);
                    }

                    // Pad-right
                    if (ow_end_c < OW) {
                        pooling_scalar_row(
                            out_d + oh * out_row_stride, in_ch,
                            ID, IH, IW, od, oh, ow_end_c, OW,
                            KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                            out_w_stride, in_d_stride, in_row_stride,
                            attrs.type, attrs.add_to, attrs.p_norm);
                    }
                }
            } else {
                // ---- Non-SIMD interior: all-scalar fallback ----
                for (int64_t oh = oh_beg_c; oh < oh_end_c; ++oh) {
                    pooling_scalar_row(
                        out_d + oh * out_row_stride, in_ch,
                        ID, IH, IW, od, oh, 0, OW,
                        KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                        out_w_stride, in_d_stride, in_row_stride,
                        attrs.type, attrs.add_to, attrs.p_norm);
                }
            }

            // ---- Pad-Bottom (scalar) ----
            for (int64_t oh = oh_end_c; oh < OH; ++oh) {
                pooling_scalar_row(
                    out_d + oh * out_row_stride, in_ch,
                    ID, IH, IW, od, oh, 0, OW,
                    KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                    out_w_stride, in_d_stride, in_row_stride,
                    attrs.type, attrs.add_to, attrs.p_norm);
            }
        }

        // ================================================
        // Pad-Depth Back (3D only; empty for 2D)
        // ================================================
        for (int64_t od = od_end_c; od < OD; ++od) {
            T* out_d = out_ch + od * out_d_stride;
            for (int64_t oh = 0; oh < OH; ++oh) {
                pooling_scalar_row(
                    out_d + oh * out_row_stride, in_ch,
                    ID, IH, IW, od, oh, 0, OW,
                    KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                    out_w_stride, in_d_stride, in_row_stride,
                    attrs.type, attrs.add_to, attrs.p_norm);
            }
        }
    };

    // ----- Parallel dispatch (N*C parallel — per sample+channel) -----
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

void pooling_cpu(const PoolingAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx,
                  void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        pooling_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        pooling_impl<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"pooling_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
