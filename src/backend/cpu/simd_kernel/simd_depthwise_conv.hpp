#pragma once
/// @file simd_depthwise_conv.hpp
/// @brief SIMD kernel functions for 2D/3D depthwise convolution (NCHWC8/NCDHWC8 only).
///
/// Weights are prepacked into dense [C8, (KD,) KH, KW, 8] format.
/// Spatial rank is auto-detected: 2D uses KD=1/OD=1/ID=1 (loop degenerates).
///
/// Key design (following the pooling.cpp pattern):
///   1. Single DwConvParams struct with full 3D fields — 2D uses KD=1/OD=1/ID=1.
///   2. Single set of h4/h1 SIMD kernels with built-in KD loop.
///   3. All C8 blocks use SIMD — pad channels are zero.
///   4. Weights loaded as vectors (8 channels at once) from prepacked buffer.
///   5. Epilogue activation fused in-kernel via apply_epilogue_vec.

#include "nnops/detail/simd/simd.hpp"
#include "nnops/core/epilogue.hpp"
#include "../epilogue_impl.hpp"

namespace nnops::kernel {

using namespace simd;

// ============================================================
// Kernel parameters — set once per C8 block, od/oh updated per call
// ============================================================

struct DwConvParams {
    int64_t ID, IH, IW;           // input spatial dims
    int64_t OH, OW;               // output spatial dims (for store bounds)
    int64_t od, oh;               // current output position (set before call)
    int64_t KD, KH, KW;           // kernel shape
    int64_t SD, SH, SW;           // stride
    int64_t DD, DH, DW;           // dilation
    int64_t PD, PH, PW;           // padding
    int64_t out_row_stride;       // output row stride (elems)
    int64_t in_d_stride;          // input depth stride (elems)
    int64_t in_row_stride;        // input row stride (elems)
};

// ============================================================
// H4 SIMD kernel: process 4 output rows x all OW columns
// ============================================================

template <typename T>
inline void dwconv_h4(
    T* output, const T* input, const T* w_base,
    const T* bias_vec,            // nullptr or pointer to 8 bias values
    const DwConvParams& p,
    const Epilogue& epilogue,
    bool add_to)
{
    const T* type_tag = output;
    const auto vbias = bias_vec ? v_load(bias_vec) : v_zero(type_tag);

    for (int64_t ow = 0; ow < p.OW; ++ow) {
        auto vacc0 = vbias;
        auto vacc1 = vbias;
        auto vacc2 = vbias;
        auto vacc3 = vbias;

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

                    auto vk = v_load(&w_base[(kd * p.KH * p.KW + kh * p.KW + kw) * 8]);

                    if (v0) {
                        vacc0 = v_fmadd(v_load(input + d_off + ih0 * p.in_row_stride + w_off), vk, vacc0);
                    }
                    if (v1) {
                        vacc1 = v_fmadd(v_load(input + d_off + ih1 * p.in_row_stride + w_off), vk, vacc1);
                    }
                    if (v2) {
                        vacc2 = v_fmadd(v_load(input + d_off + ih2 * p.in_row_stride + w_off), vk, vacc2);
                    }
                    if (v3) {
                        vacc3 = v_fmadd(v_load(input + d_off + ih3 * p.in_row_stride + w_off), vk, vacc3);
                    }
                }
            }
        }

        // SIMD epilogue
        vacc0 = nnops::backend::cpu::apply_epilogue_vec(epilogue, type_tag, vacc0);
        vacc1 = nnops::backend::cpu::apply_epilogue_vec(epilogue, type_tag, vacc1);
        vacc2 = nnops::backend::cpu::apply_epilogue_vec(epilogue, type_tag, vacc2);
        vacc3 = nnops::backend::cpu::apply_epilogue_vec(epilogue, type_tag, vacc3);

        // Store
        if (p.oh + 0 < p.OH) {
            v_store_add(output + 0 * p.out_row_stride + ow * 8, vacc0, add_to);
        }
        if (p.oh + 1 < p.OH) {
            v_store_add(output + 1 * p.out_row_stride + ow * 8, vacc1, add_to);
        }
        if (p.oh + 2 < p.OH) {
            v_store_add(output + 2 * p.out_row_stride + ow * 8, vacc2, add_to);
        }
        if (p.oh + 3 < p.OH) {
            v_store_add(output + 3 * p.out_row_stride + ow * 8, vacc3, add_to);
        }
    }
}

// ============================================================
// H1 SIMD kernel: process 1 output row x all OW columns
// ============================================================

template <typename T>
inline void dwconv_h1(
    T* output, const T* input, const T* w_base,
    const T* bias_vec,
    const DwConvParams& p,
    const Epilogue& epilogue,
    bool add_to)
{
    const T* type_tag = output;
    const auto vbias = bias_vec ? v_load(bias_vec) : v_zero(type_tag);

    for (int64_t ow = 0; ow < p.OW; ++ow) {
        auto vacc = vbias;

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
                    int64_t w_off = iw * 8;

                    auto vk = v_load(&w_base[(kd * p.KH * p.KW + kh * p.KW + kw) * 8]);
                    vacc = v_fmadd(v_load(input + d_off + ih * p.in_row_stride + w_off), vk, vacc);
                }
            }
        }

        vacc = nnops::backend::cpu::apply_epilogue_vec(epilogue, type_tag, vacc);
        v_store_add(output + ow * 8, vacc, add_to);
    }
}

}  // namespace nnops::kernel
