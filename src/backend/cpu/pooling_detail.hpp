#pragma once
/// @file pooling_impl.h
/// @brief Unified SIMD pooling kernels — NCHW (Packed=false) and NCHWC8 (Packed=true).
///
/// When Packed=false (NCHW): v_load loads simd_lane_for<T> W-positions of 1 channel,
///   ow steps by simd_lane_for<T>, SW==2 uses v_load_even for stride-2 gather.
/// When Packed=true (NCHWC8): v_load loads 8 channels at 1 W-position,
///   ow steps by 1, w_off is multiplied by 8, always uses v_load (channels contiguous).
///
/// The Packed template parameter collapses 8 kernel functions (4 NCHW + 4 NCHWC8)
/// into 4, and the scalar row processors from 2 into 1.

#include "nnops/detail/simd/simd.hpp"
#include "nnops/ops/pooling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nnops::backend::cpu {
namespace pooling_detail {

using namespace nnops::simd;

// ============================================================
// Unified scalar row processor — NCHW (Packed=false) and
// NCHWC8 (Packed=true).
// ============================================================

/// Process a range of output columns for one (od, oh) position using scalar code.
/// Packed=false: single-channel, w_stride=1, valid_lanes ignored (always 1).
/// Packed=true:  multi-channel, w_stride=8, valid_lanes = 1–8.
template <typename T, bool Packed>
inline void pooling_scalar_row(
    T* output, const T* input,
    int64_t ID, int64_t IH, int64_t IW,
    int64_t od, int64_t oh,
    int64_t ow_start, int64_t ow_end,
    int64_t KD, int64_t KH, int64_t KW,
    int64_t SD, int64_t SH, int64_t SW,
    int64_t DD, int64_t DH, int64_t DW,
    int64_t PD, int64_t PH, int64_t PW,
    int64_t in_d_stride, int64_t in_row_stride,
    int64_t valid_lanes,
    PoolingType type, bool add_to, int64_t p_norm)
{
    constexpr int64_t w_stride = Packed ? 8 : 1;
    const int64_t num_lanes = Packed ? valid_lanes : 1;

    for (int64_t ow = ow_start; ow < ow_end; ++ow) {
        for (int64_t lane = 0; lane < num_lanes; ++lane) {
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
                            const float val = s_load(
                                &input[id * in_d_stride + ih * in_row_stride + iw * w_stride + lane]);
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
                            sum += s_load(&input[id * in_d_stride + ih * in_row_stride + iw * w_stride + lane]);
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
                                std::abs(s_load(
                                    &input[id * in_d_stride + ih * in_row_stride + iw * w_stride + lane])), fp);
                        }
                    }
                }
                result = std::pow(sum, 1.0f / fp);
                break;
            }
            }

            // Write back
            const int64_t out_idx = ow * w_stride + lane;
            if (add_to) {
                s_store(&output[out_idx], s_load(&output[out_idx]) + result);
            } else {
                s_store(&output[out_idx], result);
            }
        }
    }
}

// ============================================================
// Unified MaxPooling SIMD kernels
// ============================================================

/// Process 4 output rows × ow_count output columns with element-wise v_max.
/// Packed=false: ow_count must be multiple of simd_lane_for<T>;
///   SW==2 uses v_load_even for stride-2 gather.
/// Packed=true: ow steps by 1; w_off multiplied by 8; always v_load.
template <typename T, int SW_val, bool Packed>
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
    constexpr int64_t ow_step = Packed ? 1 : simd_lane_for<T>;
    const int64_t ow_mul = Packed ? 8 : out_w_stride;

    for (int64_t ow = 0; ow < ow_count; ow += ow_step) {
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
                    const int64_t w_off = Packed
                        ? (ow * SW_val + kw * DW) * 8
                        : (ow * SW_val + kw * DW);

                    if constexpr (!Packed && SW_val == 2) {
                        vacc0 = v_max(vacc0, v_load_even(input + d_off + h_off0 + w_off));
                        vacc1 = v_max(vacc1, v_load_even(input + d_off + h_off1 + w_off));
                        vacc2 = v_max(vacc2, v_load_even(input + d_off + h_off2 + w_off));
                        vacc3 = v_max(vacc3, v_load_even(input + d_off + h_off3 + w_off));
                    } else {
                        vacc0 = v_max(vacc0, v_load(input + d_off + h_off0 + w_off));
                        vacc1 = v_max(vacc1, v_load(input + d_off + h_off1 + w_off));
                        vacc2 = v_max(vacc2, v_load(input + d_off + h_off2 + w_off));
                        vacc3 = v_max(vacc3, v_load(input + d_off + h_off3 + w_off));
                    }
                }
            }
        }

        T* out0 = output + 0 * out_row_stride + ow * ow_mul;
        T* out1 = output + 1 * out_row_stride + ow * ow_mul;
        T* out2 = output + 2 * out_row_stride + ow * ow_mul;
        T* out3 = output + 3 * out_row_stride + ow * ow_mul;

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
template <typename T, int SW_val, bool Packed>
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
    constexpr int64_t ow_step = Packed ? 1 : simd_lane_for<T>;
    const int64_t ow_mul = Packed ? 8 : out_w_stride;

    for (int64_t ow = 0; ow < ow_count; ow += ow_step) {
        auto vacc = vinit;

        for (int64_t kd = 0; kd < KD; ++kd) {
            const int64_t d_off = kd * DD * in_d_stride;
            for (int64_t kh = 0; kh < KH; ++kh) {
                const int64_t h_off = kh * DH * in_row_stride;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    const int64_t w_off = Packed
                        ? (ow * SW_val + kw * DW) * 8
                        : (ow * SW_val + kw * DW);

                    if constexpr (!Packed && SW_val == 2) {
                        vacc = v_max(vacc, v_load_even(input + d_off + h_off + w_off));
                    } else {
                        vacc = v_max(vacc, v_load(input + d_off + h_off + w_off));
                    }
                }
            }
        }

        T* out_r = output + ow * ow_mul;

        if (add_to) {
            vacc = v_add(vacc, v_load(out_r));
        }

        v_store(out_r, vacc);
    }
}

// ============================================================
// Unified AvgPooling SIMD kernels
// ============================================================

/// Process 4 output rows × ow_count output columns with FMA accumulation.
template <typename T, int SW_val, bool Packed>
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
    const auto vzero  = v_set1(input, 0.0f);
    constexpr int64_t ow_step = Packed ? 1 : simd_lane_for<T>;
    const int64_t ow_mul = Packed ? 8 : out_w_stride;

    for (int64_t ow = 0; ow < ow_count; ow += ow_step) {
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
                    const int64_t w_off = Packed
                        ? (ow * SW_val + kw * DW) * 8
                        : (ow * SW_val + kw * DW);

                    if constexpr (!Packed && SW_val == 2) {
                        vacc0 = v_fmadd(v_load_even(input + d_off + h_off0 + w_off), vscale, vacc0);
                        vacc1 = v_fmadd(v_load_even(input + d_off + h_off1 + w_off), vscale, vacc1);
                        vacc2 = v_fmadd(v_load_even(input + d_off + h_off2 + w_off), vscale, vacc2);
                        vacc3 = v_fmadd(v_load_even(input + d_off + h_off3 + w_off), vscale, vacc3);
                    } else {
                        vacc0 = v_fmadd(v_load(input + d_off + h_off0 + w_off), vscale, vacc0);
                        vacc1 = v_fmadd(v_load(input + d_off + h_off1 + w_off), vscale, vacc1);
                        vacc2 = v_fmadd(v_load(input + d_off + h_off2 + w_off), vscale, vacc2);
                        vacc3 = v_fmadd(v_load(input + d_off + h_off3 + w_off), vscale, vacc3);
                    }
                }
            }
        }

        T* out0 = output + 0 * out_row_stride + ow * ow_mul;
        T* out1 = output + 1 * out_row_stride + ow * ow_mul;
        T* out2 = output + 2 * out_row_stride + ow * ow_mul;
        T* out3 = output + 3 * out_row_stride + ow * ow_mul;

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

/// Process 1 output row × ow_count output columns with FMA accumulation.
template <typename T, int SW_val, bool Packed>
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
    const auto vzero  = v_set1(input, 0.0f);
    constexpr int64_t ow_step = Packed ? 1 : simd_lane_for<T>;
    const int64_t ow_mul = Packed ? 8 : out_w_stride;

    for (int64_t ow = 0; ow < ow_count; ow += ow_step) {
        auto vacc = vzero;

        for (int64_t kd = 0; kd < KD; ++kd) {
            const int64_t d_off = kd * DD * in_d_stride;
            for (int64_t kh = 0; kh < KH; ++kh) {
                const int64_t h_off = kh * DH * in_row_stride;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    const int64_t w_off = Packed
                        ? (ow * SW_val + kw * DW) * 8
                        : (ow * SW_val + kw * DW);

                    if constexpr (!Packed && SW_val == 2) {
                        vacc = v_fmadd(v_load_even(input + d_off + h_off + w_off), vscale, vacc);
                    } else {
                        vacc = v_fmadd(v_load(input + d_off + h_off + w_off), vscale, vacc);
                    }
                }
            }
        }

        T* out_r = output + ow * ow_mul;

        if (add_to) {
            vacc = v_add(vacc, v_load(out_r));
        }

        v_store(out_r, vacc);
    }
}

}  // namespace pooling_detail
}  // namespace nnops::backend::cpu
