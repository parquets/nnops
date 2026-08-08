#pragma once
/// @file simd_pooling.hpp
/// @brief SIMD kernel functions for 2D/3D spatial pooling (NCHWC8 only).
///
/// Key design:
///   1. All C8 blocks (including partial) use SIMD h4/h1 kernels — pad channels in
///      NCHWC8 input are zero, so SIMD operates correctly on all 8 lanes
///      (max ignores pad zeros, avg accumulates zero = no contribution).
///   2. SIMD kernels handle exclude_pad via in-kernel per-position valid_count.
///   3. Height-4 blocking: process 4 output rows at once (h4), remainder with h1.
///   4. In-kernel bounds checking for all kernel positions — no pre-splitting.
///   5. All SW values supported — offset math is just integer arithmetic.
///   6. Params passed by const reference (unified across all kernels).

#include "nnops/detail/simd/simd.hpp"

#include <limits>

namespace nnops::kernel {

using namespace simd;

// ============================================================
// Kernel parameters — set once, od/oh updated per call
// ============================================================

struct PoolingKernelParams {
    int64_t ID, IH, IW;          // input spatial dims
    int64_t OH, OW;              // output spatial dims (for store bounds)
    int64_t od, oh;              // current output position (set before call)
    int64_t KD, KH, KW;          // kernel shape
    int64_t SD, SH, SW;          // stride
    int64_t DD, DH, DW;          // dilation
    int64_t PD, PH, PW;          // padding
    int64_t out_row_stride;      // output row stride (elems)
    int64_t in_d_stride;         // input depth stride (elems)
    int64_t in_row_stride;       // input row stride (elems)
    bool exclude_pad = false;    // AvgPool: divide by valid count instead of K_total
};

// ============================================================
// MaxPooling SIMD kernels (bounds-checked, any SW)
// ============================================================

template <typename T>
inline void maxpool_h4(
    T* output, const T* input,
    const PoolingKernelParams& p,
    float /*scale*/, bool add_to)
{
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();
    const auto vinit = v_set1(input, neg_inf);
    const auto vzero = v_set1(input, 0.0f);

    for (int64_t ow = 0; ow < p.OW; ++ow) {
        auto vacc0 = vinit, vacc1 = vinit, vacc2 = vinit, vacc3 = vinit;
        bool any0 = false, any1 = false, any2 = false, any3 = false;

        for (int64_t kd = 0; kd < p.KD; ++kd) {
            int64_t id = p.od * p.SD + kd * p.DD - p.PD;
            if (id < 0 || id >= p.ID) {
                continue;
            }
            int64_t d_off = id * p.in_d_stride;

            for (int64_t kh = 0; kh < p.KH; ++kh) {
                int64_t ih0 = (p.oh + 0) * p.SH + kh * p.DH - p.PH;
                int64_t ih1 = (p.oh + 1) * p.SH + kh * p.DH - p.PH;
                int64_t ih2 = (p.oh + 2) * p.SH + kh * p.DH - p.PH;
                int64_t ih3 = (p.oh + 3) * p.SH + kh * p.DH - p.PH;
                bool v0 = (p.oh + 0 < p.OH) && ih0 >= 0 && ih0 < p.IH;
                bool v1 = (p.oh + 1 < p.OH) && ih1 >= 0 && ih1 < p.IH;
                bool v2 = (p.oh + 2 < p.OH) && ih2 >= 0 && ih2 < p.IH;
                bool v3 = (p.oh + 3 < p.OH) && ih3 >= 0 && ih3 < p.IH;

                for (int64_t kw = 0; kw < p.KW; ++kw) {
                    int64_t iw = ow * p.SW + kw * p.DW - p.PW;
                    if (iw < 0 || iw >= p.IW) {
                        continue;
                    }
                    int64_t w_off = iw * 8;

                    if (v0) { vacc0 = v_max(vacc0, v_load(input + d_off + ih0 * p.in_row_stride + w_off)); any0 = true; }
                    if (v1) { vacc1 = v_max(vacc1, v_load(input + d_off + ih1 * p.in_row_stride + w_off)); any1 = true; }
                    if (v2) { vacc2 = v_max(vacc2, v_load(input + d_off + ih2 * p.in_row_stride + w_off)); any2 = true; }
                    if (v3) { vacc3 = v_max(vacc3, v_load(input + d_off + ih3 * p.in_row_stride + w_off)); any3 = true; }
                }
            }
        }

        if (!any0) {
            vacc0 = vzero;
        }
        if (!any1) {
            vacc1 = vzero;
        }
        if (!any2) {
            vacc2 = vzero;
        }
        if (!any3) {
            vacc3 = vzero;
        }

        if (p.oh + 0 < p.OH) {
            T* out0 = output + 0 * p.out_row_stride + ow * 8;
            if (add_to) {
                vacc0 = v_add(vacc0, v_load(out0));
            }
            v_store(out0, vacc0);
        }
        if (p.oh + 1 < p.OH) {
            T* out1 = output + 1 * p.out_row_stride + ow * 8;
            if (add_to) {
                vacc1 = v_add(vacc1, v_load(out1));
            }
            v_store(out1, vacc1);
        }
        if (p.oh + 2 < p.OH) {
            T* out2 = output + 2 * p.out_row_stride + ow * 8;
            if (add_to) {
                vacc2 = v_add(vacc2, v_load(out2));
            }
            v_store(out2, vacc2);
        }
        if (p.oh + 3 < p.OH) {
            T* out3 = output + 3 * p.out_row_stride + ow * 8;
            if (add_to) {
                vacc3 = v_add(vacc3, v_load(out3));
            }
            v_store(out3, vacc3);
        }
    }
}

template <typename T>
inline void maxpool_h1(
    T* output, const T* input,
    const PoolingKernelParams& p,
    float /*scale*/, bool add_to)
{
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();
    const auto vinit = v_set1(input, neg_inf);
    const auto vzero = v_set1(input, 0.0f);

    for (int64_t ow = 0; ow < p.OW; ++ow) {
        auto vacc = vinit;
        bool any = false;

        for (int64_t kd = 0; kd < p.KD; ++kd) {
            int64_t id = p.od * p.SD + kd * p.DD - p.PD;
            if (id < 0 || id >= p.ID) {
                continue;
            }
            int64_t d_off = id * p.in_d_stride;

            for (int64_t kh = 0; kh < p.KH; ++kh) {
                int64_t ih = p.oh * p.SH + kh * p.DH - p.PH;
                if (ih < 0 || ih >= p.IH) {
                    continue;
                }

                for (int64_t kw = 0; kw < p.KW; ++kw) {
                    int64_t iw = ow * p.SW + kw * p.DW - p.PW;
                    if (iw < 0 || iw >= p.IW) {
                        continue;
                    }
                    vacc = v_max(vacc, v_load(input + d_off + ih * p.in_row_stride + iw * 8));
                    any = true;
                }
            }
        }

        if (!any) {
            vacc = vzero;
        }

        T* out_r = output + ow * 8;
        if (add_to) {
            vacc = v_add(vacc, v_load(out_r));
        }
        v_store(out_r, vacc);
    }
}

// ============================================================
// AvgPooling SIMD kernels (bounds-checked, any SW)
// ============================================================

template <typename T>
inline void avgpool_h4(
    T* output, const T* input,
    const PoolingKernelParams& p,
    float scale, bool add_to)
{
    const auto vscale = v_set1(input, scale);
    const auto vzero  = v_set1(input, 0.0f);

    for (int64_t ow = 0; ow < p.OW; ++ow) {
        auto vacc0 = vzero, vacc1 = vzero, vacc2 = vzero, vacc3 = vzero;
        int64_t cnt0 = 0, cnt1 = 0, cnt2 = 0, cnt3 = 0;

        for (int64_t kd = 0; kd < p.KD; ++kd) {
            int64_t id = p.od * p.SD + kd * p.DD - p.PD;
            if (id < 0 || id >= p.ID) {
                continue;
            }
            int64_t d_off = id * p.in_d_stride;

            for (int64_t kh = 0; kh < p.KH; ++kh) {
                int64_t ih0 = (p.oh + 0) * p.SH + kh * p.DH - p.PH;
                int64_t ih1 = (p.oh + 1) * p.SH + kh * p.DH - p.PH;
                int64_t ih2 = (p.oh + 2) * p.SH + kh * p.DH - p.PH;
                int64_t ih3 = (p.oh + 3) * p.SH + kh * p.DH - p.PH;
                bool v0 = (p.oh + 0 < p.OH) && ih0 >= 0 && ih0 < p.IH;
                bool v1 = (p.oh + 1 < p.OH) && ih1 >= 0 && ih1 < p.IH;
                bool v2 = (p.oh + 2 < p.OH) && ih2 >= 0 && ih2 < p.IH;
                bool v3 = (p.oh + 3 < p.OH) && ih3 >= 0 && ih3 < p.IH;

                for (int64_t kw = 0; kw < p.KW; ++kw) {
                    int64_t iw = ow * p.SW + kw * p.DW - p.PW;
                    if (iw < 0 || iw >= p.IW) {
                        continue;
                    }
                    int64_t w_off = iw * 8;

                    if (v0) { vacc0 = v_add(vacc0, v_load(input + d_off + ih0 * p.in_row_stride + w_off)); cnt0++; }
                    if (v1) { vacc1 = v_add(vacc1, v_load(input + d_off + ih1 * p.in_row_stride + w_off)); cnt1++; }
                    if (v2) { vacc2 = v_add(vacc2, v_load(input + d_off + ih2 * p.in_row_stride + w_off)); cnt2++; }
                    if (v3) { vacc3 = v_add(vacc3, v_load(input + d_off + ih3 * p.in_row_stride + w_off)); cnt3++; }
                }
            }
        }

        // Apply scale: exclude_pad → divide by valid count; else use fixed scale
        if (p.exclude_pad) {
            if (cnt0 > 0) {
                vacc0 = v_mul(vacc0, v_set1(input, 1.0f / static_cast<float>(cnt0)));
            }
            if (cnt1 > 0) {
                vacc1 = v_mul(vacc1, v_set1(input, 1.0f / static_cast<float>(cnt1)));
            }
            if (cnt2 > 0) {
                vacc2 = v_mul(vacc2, v_set1(input, 1.0f / static_cast<float>(cnt2)));
            }
            if (cnt3 > 0) {
                vacc3 = v_mul(vacc3, v_set1(input, 1.0f / static_cast<float>(cnt3)));
            }
        } else {
            vacc0 = v_mul(vacc0, vscale);
            vacc1 = v_mul(vacc1, vscale);
            vacc2 = v_mul(vacc2, vscale);
            vacc3 = v_mul(vacc3, vscale);
        }

        if (p.oh + 0 < p.OH) {
            T* out0 = output + 0 * p.out_row_stride + ow * 8;
            if (add_to) {
                vacc0 = v_add(vacc0, v_load(out0));
            }
            v_store(out0, vacc0);
        }
        if (p.oh + 1 < p.OH) {
            T* out1 = output + 1 * p.out_row_stride + ow * 8;
            if (add_to) {
                vacc1 = v_add(vacc1, v_load(out1));
            }
            v_store(out1, vacc1);
        }
        if (p.oh + 2 < p.OH) {
            T* out2 = output + 2 * p.out_row_stride + ow * 8;
            if (add_to) {
                vacc2 = v_add(vacc2, v_load(out2));
            }
            v_store(out2, vacc2);
        }
        if (p.oh + 3 < p.OH) {
            T* out3 = output + 3 * p.out_row_stride + ow * 8;
            if (add_to) {
                vacc3 = v_add(vacc3, v_load(out3));
            }
            v_store(out3, vacc3);
        }
    }
}

template <typename T>
inline void avgpool_h1(
    T* output, const T* input,
    const PoolingKernelParams& p,
    float scale, bool add_to)
{
    const auto vscale = v_set1(input, scale);
    const auto vzero  = v_set1(input, 0.0f);

    for (int64_t ow = 0; ow < p.OW; ++ow) {
        auto vacc = vzero;
        int64_t valid_count = 0;

        for (int64_t kd = 0; kd < p.KD; ++kd) {
            int64_t id = p.od * p.SD + kd * p.DD - p.PD;
            if (id < 0 || id >= p.ID) {
                continue;
            }
            int64_t d_off = id * p.in_d_stride;

            for (int64_t kh = 0; kh < p.KH; ++kh) {
                int64_t ih = p.oh * p.SH + kh * p.DH - p.PH;
                if (ih < 0 || ih >= p.IH) {
                    continue;
                }

                for (int64_t kw = 0; kw < p.KW; ++kw) {
                    int64_t iw = ow * p.SW + kw * p.DW - p.PW;
                    if (iw < 0 || iw >= p.IW) {
                        continue;
                    }
                    vacc = v_add(vacc, v_load(input + d_off + ih * p.in_row_stride + iw * 8));
                    valid_count++;
                }
            }
        }

        if (p.exclude_pad && valid_count > 0) {
            vacc = v_mul(vacc, v_set1(input, 1.0f / static_cast<float>(valid_count)));
        } else if (!p.exclude_pad) {
            vacc = v_mul(vacc, vscale);
        }

        T* out_r = output + ow * 8;
        if (add_to) {
            vacc = v_add(vacc, v_load(out_r));
        }
        v_store(out_r, vacc);
    }
}

}  // namespace nnops::kernel
