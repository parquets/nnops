#pragma once
/// @file pack_f16.hpp
/// @brief AArch64 NEON float16 pack kernels for GEMM LHS/RHS.
///
/// Mirrors pack_f32.hpp but operates on float16 (binary16). The NEON
/// fp16 vector arithmetic extension (ARMv8.2-A+) is required for SIMD
/// paths; scalar fallback is used otherwise.
///
/// Reference: nn_compute/src/cpu/kernel/pack/aarch64/pack_fp16.hpp

#include <arm_neon.h>
#include <cstring>
#include "backend/cpu/common/restrict.hpp"
#include "transpose.hpp"

namespace nnops::backend::cpu::aarch64 {

// =========================================================================
//  LHS Transpose pack (f16)
// =========================================================================

inline void pack_trans_n1_f16(float16_t* NNOPS_RESTRICT output,
                              const float16_t* NNOPS_RESTRICT input,
                              int ir_step, int K, float /*scale*/) noexcept {
    for (int k = 0; k < K; ++k) {
        *output++ = input[0];
        input += ir_step;
    }
}

inline void pack_trans_n4_f16(float16_t* NNOPS_RESTRICT output,
                              const float16_t* NNOPS_RESTRICT input,
                              int ir_step, int K, float /*scale*/) noexcept {
    int k = 0;
    for (; k <= K - 4; k += 4) {
        float16x4_t v0 = vld1_f16(input + 0 * ir_step);
        float16x4_t v1 = vld1_f16(input + 1 * ir_step);
        float16x4_t v2 = vld1_f16(input + 2 * ir_step);
        float16x4_t v3 = vld1_f16(input + 3 * ir_step);

        transpose_4x4_f16(v0, v1, v2, v3);

        vst1_f16(output + 0 * 4, v0);
        vst1_f16(output + 1 * 4, v1);
        vst1_f16(output + 2 * 4, v2);
        vst1_f16(output + 3 * 4, v3);

        output += 4 * 4;
        input  += 4;
    }
    for (; k < K; ++k) {
        output[0] = input[0 * ir_step];
        output[1] = input[1 * ir_step];
        output[2] = input[2 * ir_step];
        output[3] = input[3 * ir_step];
        output += 4;
        input  += 1;
    }
}

inline void pack_trans_n8_f16(float16_t* NNOPS_RESTRICT output,
                              const float16_t* NNOPS_RESTRICT input,
                              int ir_step, int K, float /*scale*/) noexcept {
    int k = 0;
    for (; k <= K - 4; k += 4) {
        float16x8_t v0 = vld1q_f16(input + 0 * ir_step);
        float16x8_t v1 = vld1q_f16(input + 1 * ir_step);
        float16x8_t v2 = vld1q_f16(input + 2 * ir_step);
        float16x8_t v3 = vld1q_f16(input + 3 * ir_step);
        float16x8_t v4 = vld1q_f16(input + 4 * ir_step);
        float16x8_t v5 = vld1q_f16(input + 5 * ir_step);
        float16x8_t v6 = vld1q_f16(input + 6 * ir_step);
        float16x8_t v7 = vld1q_f16(input + 7 * ir_step);

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
        input  += 4;
    }
    for (; k < K; ++k) {
        for (int i = 0; i < 8; ++i) {
            *output++ = input[i * ir_step];
        }
        input += 1;
    }
}

inline void pack_trans_n12_f16(float16_t* NNOPS_RESTRICT output,
                               const float16_t* NNOPS_RESTRICT input,
                               int ir_step, int K, float /*scale*/) noexcept {
    int k = 0;
    for (; k <= K - 8; k += 8) {
        float16x8_t v0  = vld1q_f16(input + 0 * ir_step);
        float16x8_t v1  = vld1q_f16(input + 1 * ir_step);
        float16x8_t v2  = vld1q_f16(input + 2 * ir_step);
        float16x8_t v3  = vld1q_f16(input + 3 * ir_step);
        float16x8_t v4  = vld1q_f16(input + 4 * ir_step);
        float16x8_t v5  = vld1q_f16(input + 5 * ir_step);
        float16x8_t v6  = vld1q_f16(input + 6 * ir_step);
        float16x8_t v7  = vld1q_f16(input + 7 * ir_step);
        float16x8_t v8  = vld1q_f16(input + 8 * ir_step);
        float16x8_t v9  = vld1q_f16(input + 9 * ir_step);
        float16x8_t v10 = vld1q_f16(input + 10 * ir_step);
        float16x8_t v11 = vld1q_f16(input + 11 * ir_step);

        transpose_12x8_f16(v0, v1, v2, v3, v4, v5,
                           v6, v7, v8, v9, v10, v11);

        vst1q_f16(output + 0 * 8, v0);
        vst1q_f16(output + 1 * 8, v1);
        vst1q_f16(output + 2 * 8, v2);
        vst1q_f16(output + 3 * 8, v3);
        vst1q_f16(output + 4 * 8, v4);
        vst1q_f16(output + 5 * 8, v5);
        vst1q_f16(output + 6 * 8, v6);
        vst1q_f16(output + 7 * 8, v7);
        vst1q_f16(output + 8 * 8, v8);
        vst1q_f16(output + 9 * 8, v9);
        vst1q_f16(output + 10 * 8, v10);
        vst1q_f16(output + 11 * 8, v11);

        output += 12 * 8;
        input  += 8;
    }
    for (; k < K; ++k) {
        *output++ = input[0 * ir_step];
        *output++ = input[1 * ir_step];
        *output++ = input[2 * ir_step];
        *output++ = input[3 * ir_step];
        *output++ = input[4 * ir_step];
        *output++ = input[5 * ir_step];
        *output++ = input[6 * ir_step];
        *output++ = input[7 * ir_step];
        *output++ = input[8 * ir_step];
        *output++ = input[9 * ir_step];
        *output++ = input[10 * ir_step];
        *output++ = input[11 * ir_step];
        input += 1;
    }
}

inline void pack_trans_n16_f16(float16_t* NNOPS_RESTRICT output,
                               const float16_t* NNOPS_RESTRICT input,
                               int ir_step, int K, float /*scale*/) noexcept {
    int k = 0;
    for (; k <= K - 8; k += 8) {
        float16x8_t v0  = vld1q_f16(input + 0 * ir_step);
        float16x8_t v1  = vld1q_f16(input + 1 * ir_step);
        float16x8_t v2  = vld1q_f16(input + 2 * ir_step);
        float16x8_t v3  = vld1q_f16(input + 3 * ir_step);
        float16x8_t v4  = vld1q_f16(input + 4 * ir_step);
        float16x8_t v5  = vld1q_f16(input + 5 * ir_step);
        float16x8_t v6  = vld1q_f16(input + 6 * ir_step);
        float16x8_t v7  = vld1q_f16(input + 7 * ir_step);
        float16x8_t v8  = vld1q_f16(input + 8 * ir_step);
        float16x8_t v9  = vld1q_f16(input + 9 * ir_step);
        float16x8_t v10 = vld1q_f16(input + 10 * ir_step);
        float16x8_t v11 = vld1q_f16(input + 11 * ir_step);
        float16x8_t v12 = vld1q_f16(input + 12 * ir_step);
        float16x8_t v13 = vld1q_f16(input + 13 * ir_step);
        float16x8_t v14 = vld1q_f16(input + 14 * ir_step);
        float16x8_t v15 = vld1q_f16(input + 15 * ir_step);

        transpose_16x8_f16(v0, v1, v2, v3, v4, v5, v6, v7,
                           v8, v9, v10, v11, v12, v13, v14, v15);

        vst1q_f16(output + 0 * 8,  v0);
        vst1q_f16(output + 1 * 8,  v1);
        vst1q_f16(output + 2 * 8,  v2);
        vst1q_f16(output + 3 * 8,  v3);
        vst1q_f16(output + 4 * 8,  v4);
        vst1q_f16(output + 5 * 8,  v5);
        vst1q_f16(output + 6 * 8,  v6);
        vst1q_f16(output + 7 * 8,  v7);
        vst1q_f16(output + 8 * 8,  v8);
        vst1q_f16(output + 9 * 8,  v9);
        vst1q_f16(output + 10 * 8, v10);
        vst1q_f16(output + 11 * 8, v11);
        vst1q_f16(output + 12 * 8, v12);
        vst1q_f16(output + 13 * 8, v13);
        vst1q_f16(output + 14 * 8, v14);
        vst1q_f16(output + 15 * 8, v15);

        output += 16 * 8;
        input  += 8;
    }
    for (; k < K; ++k) {
        *output++ = input[0 * ir_step];
        *output++ = input[1 * ir_step];
        *output++ = input[2 * ir_step];
        *output++ = input[3 * ir_step];
        *output++ = input[4 * ir_step];
        *output++ = input[5 * ir_step];
        *output++ = input[6 * ir_step];
        *output++ = input[7 * ir_step];
        *output++ = input[8 * ir_step];
        *output++ = input[9 * ir_step];
        *output++ = input[10 * ir_step];
        *output++ = input[11 * ir_step];
        *output++ = input[12 * ir_step];
        *output++ = input[13 * ir_step];
        *output++ = input[14 * ir_step];
        *output++ = input[15 * ir_step];
        input += 1;
    }
}

// =========================================================================
//  RHS Copy pack (f16)  —  contiguous-row copy, panels by N.
// =========================================================================

inline void pack_copy_n1_f16(float16_t* NNOPS_RESTRICT output,
                             const float16_t* NNOPS_RESTRICT input,
                             int ir_step, int K, float /*scale*/) noexcept {
    for (int k = 0; k < K; ++k) {
        *output++ = input[0];
        input += ir_step;
    }
}

inline void pack_copy_n4_f16(float16_t* NNOPS_RESTRICT output,
                             const float16_t* NNOPS_RESTRICT input,
                             int ir_step, int K, float /*scale*/) noexcept {
    int k = 0;
    for (; k <= K - 4; k += 4) {
        float16x4_t v0 = vld1_f16(input + 0 * ir_step + 0 * 4);
        float16x4_t v1 = vld1_f16(input + 1 * ir_step + 0 * 4);
        float16x4_t v2 = vld1_f16(input + 2 * ir_step + 0 * 4);
        float16x4_t v3 = vld1_f16(input + 3 * ir_step + 0 * 4);

        vst1_f16(output + 0 * 4, v0);
        vst1_f16(output + 1 * 4, v1);
        vst1_f16(output + 2 * 4, v2);
        vst1_f16(output + 3 * 4, v3);

        output += 4 * 4;
        input  += 4 * ir_step;
    }
    for (; k < K; ++k) {
        output[0] = input[0];
        output[1] = input[1];
        output[2] = input[2];
        output[3] = input[3];
        output += 4;
        input  += ir_step;
    }
}

inline void pack_copy_n6_f16(float16_t* NNOPS_RESTRICT output,
                             const float16_t* NNOPS_RESTRICT input,
                             int ir_step, int K, float /*scale*/) noexcept {
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < 6; ++i) {
            *output++ = input[i];
        }
        input += ir_step;
    }
}

inline void pack_copy_n8_f16(float16_t* NNOPS_RESTRICT output,
                             const float16_t* NNOPS_RESTRICT input,
                             int ir_step, int K, float /*scale*/) noexcept {
    for (int k = 0; k < K; ++k) {
        float16x8_t v0 = vld1q_f16(input);
        vst1q_f16(output, v0);
        output += 8;
        input  += ir_step;
    }
}

inline void pack_copy_n12_f16(float16_t* NNOPS_RESTRICT output,
                              const float16_t* NNOPS_RESTRICT input,
                              int ir_step, int K, float /*scale*/) noexcept {
    for (int k = 0; k < K; ++k) {
        float16x8_t v0 = vld1q_f16(input + 0 * 8);
        float16x4_t v1 = vld1_f16(input + 1 * 8);
        vst1q_f16(output + 0 * 8, v0);
        vst1_f16(output + 1 * 8, v1);
        output += 12;
        input  += ir_step;
    }
}

inline void pack_copy_n16_f16(float16_t* NNOPS_RESTRICT output,
                              const float16_t* NNOPS_RESTRICT input,
                              int ir_step, int K, float /*scale*/) noexcept {
    for (int k = 0; k < K; ++k) {
        float16x8_t v0 = vld1q_f16(input + 0 * 8);
        float16x8_t v1 = vld1q_f16(input + 1 * 8);
        vst1q_f16(output + 0 * 8, v0);
        vst1q_f16(output + 1 * 8, v1);
        output += 16;
        input  += ir_step;
    }
}

// =========================================================================
//  Panel size constants (f16)  —  tuned for AArch64 NEON.
// =========================================================================

/// @brief f16 LHS micro-panel heights.
inline constexpr int mr_f16[3] = {8, 4, 1};

/// @brief f16 RHS micro-panel widths.
inline constexpr int nr_f16[3] = {16, 8, 1};

}  // namespace nnops::backend::cpu::aarch64
