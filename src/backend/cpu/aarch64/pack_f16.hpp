#pragma once
/// @file pack_f16.hpp
/// @brief AArch64 NEON float16 pack kernels for GEMM LHS/RHS.
///
/// Mirrors pack_f32.hpp on float16 (binary16) and needs the NEON fp16
/// arithmetic extension — there is no scalar fallback. A 128-bit f16 vector is
/// 8 columns wide (4 for the half-width loads), so the n8 and wider transpose
/// blocks step 8 in k where the f32 pack steps 4.
///
/// Two families:
///   - pack_trans_nN (LHS/A): reads n rows (rows at stride ir_step, k
///     contiguous within a row), scales by `scale`, writes k-major. Rows past
///     `mr` alias the last live row — the MMA kernels' convention — so a
///     partial-M panel needs no padding.
///   - pack_copy_nN (RHS/B): rows are contiguous; the vectorised widths take
///     the plain loads at full width, a narrow panel (`nr` < width) reads only
///     its live columns and zero-fills the rest in the same k loop.
///
/// Reference: nn_compute/src/cpu/kernel/pack/aarch64/pack_fp16.hpp

#include <arm_neon.h>
#include <cstring>
#include "backend/cpu/common/common_utils.hpp"
#include "transpose.hpp"

namespace nnops::backend::cpu::aarch64 {

// =========================================================================
//  LHS Transpose pack (f16)
// =========================================================================

inline void pack_trans_n1_f16(float16_t* NNOPS_RESTRICT output,
                              const float16_t* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int mr = 1) noexcept {
    const float16_t s = static_cast<float16_t>(scale);
    for (int k = 0; k < K; ++k) {
        *output++ = input[0] * s;
        input += 1;  // consecutive k within the M×K source row
    }
}

inline void pack_trans_n4_f16(float16_t* NNOPS_RESTRICT output,
                              const float16_t* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int mr = 4) noexcept {
    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;
    const float16_t* NNOPS_RESTRICT p0 = input + 0 * ir_step;
    const float16_t* NNOPS_RESTRICT p1 = input + r1 * ir_step;
    const float16_t* NNOPS_RESTRICT p2 = input + r2 * ir_step;
    const float16_t* NNOPS_RESTRICT p3 = input + r3 * ir_step;

    const float16_t s = static_cast<float16_t>(scale);
    int k = 0;
    for (; k <= K - 4; k += 4) {
        float16x4_t v0 = vmul_n_f16(vld1_f16(p0), s); p0 += 4;
        float16x4_t v1 = vmul_n_f16(vld1_f16(p1), s); p1 += 4;
        float16x4_t v2 = vmul_n_f16(vld1_f16(p2), s); p2 += 4;
        float16x4_t v3 = vmul_n_f16(vld1_f16(p3), s); p3 += 4;

        transpose_4x4_f16(v0, v1, v2, v3);

        vst1_f16(output + 0 * 4, v0);
        vst1_f16(output + 1 * 4, v1);
        vst1_f16(output + 2 * 4, v2);
        vst1_f16(output + 3 * 4, v3);

        output += 4 * 4;
    }
    for (; k < K; ++k) {
        *output++ = *(p0++) * s;
        *output++ = *(p1++) * s;
        *output++ = *(p2++) * s;
        *output++ = *(p3++) * s;
    }
}

inline void pack_trans_n8_f16(float16_t* NNOPS_RESTRICT output,
                              const float16_t* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int mr = 8) noexcept {
    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;
    const int r4 = (mr > 4) ? 4 : r3;
    const int r5 = (mr > 5) ? 5 : r4;
    const int r6 = (mr > 6) ? 6 : r5;
    const int r7 = (mr > 7) ? 7 : r6;
    const float16_t* NNOPS_RESTRICT p0 = input + 0 * ir_step;
    const float16_t* NNOPS_RESTRICT p1 = input + r1 * ir_step;
    const float16_t* NNOPS_RESTRICT p2 = input + r2 * ir_step;
    const float16_t* NNOPS_RESTRICT p3 = input + r3 * ir_step;
    const float16_t* NNOPS_RESTRICT p4 = input + r4 * ir_step;
    const float16_t* NNOPS_RESTRICT p5 = input + r5 * ir_step;
    const float16_t* NNOPS_RESTRICT p6 = input + r6 * ir_step;
    const float16_t* NNOPS_RESTRICT p7 = input + r7 * ir_step;

    const float16_t s = static_cast<float16_t>(scale);
    int k = 0;
    for (; k <= K - 8; k += 8) {
        float16x8_t v0 = vmulq_n_f16(vld1q_f16(p0), s); p0 += 8;
        float16x8_t v1 = vmulq_n_f16(vld1q_f16(p1), s); p1 += 8;
        float16x8_t v2 = vmulq_n_f16(vld1q_f16(p2), s); p2 += 8;
        float16x8_t v3 = vmulq_n_f16(vld1q_f16(p3), s); p3 += 8;
        float16x8_t v4 = vmulq_n_f16(vld1q_f16(p4), s); p4 += 8;
        float16x8_t v5 = vmulq_n_f16(vld1q_f16(p5), s); p5 += 8;
        float16x8_t v6 = vmulq_n_f16(vld1q_f16(p6), s); p6 += 8;
        float16x8_t v7 = vmulq_n_f16(vld1q_f16(p7), s); p7 += 8;

        transpose_8x8_f16(v0, v1, v2, v3, v4, v5, v6, v7);

        vst1q_f16(output + 0 * 8, v0);
        vst1q_f16(output + 1 * 8, v1);
        vst1q_f16(output + 2 * 8, v2);
        vst1q_f16(output + 3 * 8, v3);
        vst1q_f16(output + 4 * 8, v4);
        vst1q_f16(output + 5 * 8, v5);
        vst1q_f16(output + 6 * 8, v6);
        vst1q_f16(output + 7 * 8, v7);

        output += 8 * 8;
    }
    for (; k < K; ++k) {
        *output++ = *(p0++) * s;
        *output++ = *(p1++) * s;
        *output++ = *(p2++) * s;
        *output++ = *(p3++) * s;
        *output++ = *(p4++) * s;
        *output++ = *(p5++) * s;
        *output++ = *(p6++) * s;
        *output++ = *(p7++) * s;
    }
}

inline void pack_trans_n12_f16(float16_t* NNOPS_RESTRICT output,
                               const float16_t* NNOPS_RESTRICT input,
                               int ir_step, int K, float scale, int mr = 12) noexcept {
    const int last = (mr > 1) ? mr - 1 : 0;
    const float16_t* NNOPS_RESTRICT p[12];
    for (int i = 0; i < 12; ++i) {
        p[i] = input + ((i < mr) ? i : last) * ir_step;
    }

    const float16_t s = static_cast<float16_t>(scale);
    int k = 0;
    for (; k <= K - 8; k += 8) {
        float16x8_t v[12];
        for (int i = 0; i < 12; ++i) {
            v[i] = vmulq_n_f16(vld1q_f16(p[i]), s); p[i] += 8;
        }

        transpose_12x8_f16(v[0], v[1], v[2], v[3], v[4], v[5],
                           v[6], v[7], v[8], v[9], v[10], v[11]);

        for (int i = 0; i < 12; ++i) {
            vst1q_f16(output + i * 8, v[i]);
        }

        output += 12 * 8;
    }
    for (; k < K; ++k) {
        for (int i = 0; i < 12; ++i) {
            *output++ = *(p[i]++) * s;
        }
    }
}

inline void pack_trans_n16_f16(float16_t* NNOPS_RESTRICT output,
                               const float16_t* NNOPS_RESTRICT input,
                               int ir_step, int K, float scale, int mr = 16) noexcept {
    const int last = (mr > 1) ? mr - 1 : 0;
    const float16_t* NNOPS_RESTRICT p[16];
    for (int i = 0; i < 16; ++i) {
        p[i] = input + ((i < mr) ? i : last) * ir_step;
    }

    const float16_t s = static_cast<float16_t>(scale);
    int k = 0;
    for (; k <= K - 8; k += 8) {
        float16x8_t v[16];
        for (int i = 0; i < 16; ++i) {
            v[i] = vmulq_n_f16(vld1q_f16(p[i]), s); p[i] += 8;
        }

        transpose_16x8_f16(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                           v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);

        for (int i = 0; i < 16; ++i) {
            vst1q_f16(output + i * 8, v[i]);
        }

        output += 16 * 8;
    }
    for (; k < K; ++k) {
        for (int i = 0; i < 16; ++i) {
            *output++ = *(p[i]++) * s;
        }
    }
}

// =========================================================================
//  RHS copy pack (f16)  —  contiguous rows, panels by N.
// =========================================================================

inline void pack_copy_n1_f16(float16_t* NNOPS_RESTRICT output,
                             const float16_t* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale, int nr = 1) noexcept {
    const float16_t s = static_cast<float16_t>(scale);
    for (int k = 0; k < K; ++k) {
        *output++ = (nr > 0) ? input[0] * s : static_cast<float16_t>(0.0f);
        input += ir_step;
    }
}

inline void pack_copy_n4_f16(float16_t* NNOPS_RESTRICT output,
                             const float16_t* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale, int nr = 4) noexcept {
    const float16_t s = static_cast<float16_t>(scale);
    int k = 0;
    for (; k <= K - 4; k += 4) {
        if (nr >= 4) {
            const float16x4_t v0 = vmul_n_f16(vld1_f16(input + 0 * ir_step), s);
            const float16x4_t v1 = vmul_n_f16(vld1_f16(input + 1 * ir_step), s);
            const float16x4_t v2 = vmul_n_f16(vld1_f16(input + 2 * ir_step), s);
            const float16x4_t v3 = vmul_n_f16(vld1_f16(input + 3 * ir_step), s);

            vst1_f16(output + 0 * 4, v0);
            vst1_f16(output + 1 * 4, v1);
            vst1_f16(output + 2 * 4, v2);
            vst1_f16(output + 3 * 4, v3);

            output += 4 * 4;
        } else {
            for (int j = 0; j < 4; ++j) {
                for (int i = 0; i < 4; ++i) {
                    output[i] = (i < nr) ? input[j * ir_step + i] * s
                                         : static_cast<float16_t>(0.0f);
                }
                output += 4;
            }
        }
        input += 4 * ir_step;
    }
    for (; k < K; ++k) {
        for (int i = 0; i < 4; ++i) {
            *output++ = (i < nr) ? input[i] * s : static_cast<float16_t>(0.0f);
        }
        input += ir_step;
    }
}

inline void pack_copy_n6_f16(float16_t* NNOPS_RESTRICT output,
                             const float16_t* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale, int nr = 6) noexcept {
    const float16_t s = static_cast<float16_t>(scale);
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < 6; ++i) {
            *output++ = (i < nr) ? input[i * ir_step] * s
                                 : static_cast<float16_t>(0.0f);
        }
        input += 1;
    }
}

inline void pack_copy_n8_f16(float16_t* NNOPS_RESTRICT output,
                             const float16_t* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale, int nr = 8) noexcept {
    const float16_t s = static_cast<float16_t>(scale);
    for (int k = 0; k < K; ++k) {
        if (nr >= 8) {
            vst1q_f16(output, vmulq_n_f16(vld1q_f16(input), s));
        } else {
            for (int i = 0; i < 8; ++i) {
                output[i] = (i < nr) ? input[i] * s : static_cast<float16_t>(0.0f);
            }
        }
        output += 8;
        input  += ir_step;
    }
}

inline void pack_copy_n12_f16(float16_t* NNOPS_RESTRICT output,
                              const float16_t* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int nr = 12) noexcept {
    const float16_t s = static_cast<float16_t>(scale);
    for (int k = 0; k < K; ++k) {
        if (nr >= 12) {
            vst1q_f16(output + 0 * 8, vmulq_n_f16(vld1q_f16(input + 0 * 8), s));
            vst1_f16(output + 1 * 8, vmul_n_f16(vld1_f16(input + 1 * 8), s));
        } else {
            for (int i = 0; i < 12; ++i) {
                output[i] = (i < nr) ? input[i] * s : static_cast<float16_t>(0.0f);
            }
        }
        output += 12;
        input  += ir_step;
    }
}

inline void pack_copy_n16_f16(float16_t* NNOPS_RESTRICT output,
                              const float16_t* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int nr = 16) noexcept {
    const float16_t s = static_cast<float16_t>(scale);
    for (int k = 0; k < K; ++k) {
        if (nr >= 16) {
            vst1q_f16(output + 0 * 8, vmulq_n_f16(vld1q_f16(input + 0 * 8), s));
            vst1q_f16(output + 1 * 8, vmulq_n_f16(vld1q_f16(input + 1 * 8), s));
        } else {
            for (int i = 0; i < 16; ++i) {
                output[i] = (i < nr) ? input[i] * s : static_cast<float16_t>(0.0f);
            }
        }
        output += 16;
        input  += ir_step;
    }
}

// =========================================================================
//  Micro-panel size menu (f16)  —  mr = LHS rows, nr = RHS columns.
// =========================================================================

inline constexpr int mr_f16[3] = {8, 4, 1};

inline constexpr int nr_f16[3] = {16, 8, 1};

}  // namespace nnops::backend::cpu::aarch64
