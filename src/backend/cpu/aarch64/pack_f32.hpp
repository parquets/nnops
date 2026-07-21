#pragma once
/// @file pack_f32.hpp
/// @brief AArch64 NEON float32 pack kernels for GEMM LHS/RHS.
///
/// Pack operations reorder matrix rows/columns into micro-kernel-friendly
/// layouts. Two fundamental transforms:
///   - TRANSPOSE (LHS): each row is read contiguously; output is N×K packed
///     (row striding across input, contiguous across K within each row block).
///   - COPY (RHS): rows are contiguous in input; output keeps row-major.
///
/// Reference: nn_compute/src/cpu/kernel/pack/aarch64/pack_fp32.hpp

#include <arm_neon.h>
#include <cstring>
#include "backend/cpu/common/restrict.hpp"
#include "transpose.hpp"

namespace nnops::backend::cpu::aarch64 {

// =========================================================================
//  LHS Transpose pack  —  panelled transpose for packed-A in GEMM
//  pack_trans_nN_f32: reads N rows, transposes N×4 blocks, scales by `scale`.
// =========================================================================

inline void pack_trans_n1_f32(float* NNOPS_RESTRICT output,
                              const float* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale) noexcept {
    for (int k = 0; k < K; ++k) {
        *output++ = input[0] * scale;
        input += ir_step;
    }
}

inline void pack_trans_n4_f32(float* NNOPS_RESTRICT output,
                              const float* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale) noexcept {
    int k = 0;
    for (; k <= K - 4; k += 4) {
        float32x4_t v0 = vld1q_f32(input + 0 * ir_step);
        float32x4_t v1 = vld1q_f32(input + 1 * ir_step);
        float32x4_t v2 = vld1q_f32(input + 2 * ir_step);
        float32x4_t v3 = vld1q_f32(input + 3 * ir_step);

        transpose_4x4_f32(v0, v1, v2, v3);

        const float32x4_t s = vdupq_n_f32(scale);
        vst1q_f32(output + 0 * 4, vmulq_f32(v0, s));
        vst1q_f32(output + 1 * 4, vmulq_f32(v1, s));
        vst1q_f32(output + 2 * 4, vmulq_f32(v2, s));
        vst1q_f32(output + 3 * 4, vmulq_f32(v3, s));

        output += 4 * 4;
        input  += 4;
    }
    // scalar tail
    for (; k < K; ++k) {
        output[0] = input[0 * ir_step] * scale;
        output[1] = input[1 * ir_step] * scale;
        output[2] = input[2 * ir_step] * scale;
        output[3] = input[3 * ir_step] * scale;
        output += 4;
        input  += 1;
    }
}

inline void pack_trans_n8_f32(float* NNOPS_RESTRICT output,
                              const float* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale) noexcept {
    int k = 0;
    for (; k <= K - 4; k += 4) {
        float32x4_t v0 = vld1q_f32(input + 0 * ir_step);
        float32x4_t v1 = vld1q_f32(input + 1 * ir_step);
        float32x4_t v2 = vld1q_f32(input + 2 * ir_step);
        float32x4_t v3 = vld1q_f32(input + 3 * ir_step);
        float32x4_t v4 = vld1q_f32(input + 4 * ir_step);
        float32x4_t v5 = vld1q_f32(input + 5 * ir_step);
        float32x4_t v6 = vld1q_f32(input + 6 * ir_step);
        float32x4_t v7 = vld1q_f32(input + 7 * ir_step);

        transpose_8x4_f32(v0, v1, v2, v3, v4, v5, v6, v7);

        const float32x4_t s = vdupq_n_f32(scale);
        vst1q_f32(output + 0 * 4, vmulq_f32(v0, s));
        vst1q_f32(output + 1 * 4, vmulq_f32(v1, s));
        vst1q_f32(output + 2 * 4, vmulq_f32(v2, s));
        vst1q_f32(output + 3 * 4, vmulq_f32(v3, s));
        vst1q_f32(output + 4 * 4, vmulq_f32(v4, s));
        vst1q_f32(output + 5 * 4, vmulq_f32(v5, s));
        vst1q_f32(output + 6 * 4, vmulq_f32(v6, s));
        vst1q_f32(output + 7 * 4, vmulq_f32(v7, s));

        output += 4 * 8;
        input  += 4;
    }
    // scalar tail
    for (; k < K; ++k) {
        for (int i = 0; i < 8; ++i) {
            *output++ = input[i * ir_step] * scale;
        }
        input += 1;
    }
}

inline void pack_trans_n12_f32(float* NNOPS_RESTRICT output,
                               const float* NNOPS_RESTRICT input,
                               int ir_step, int K, float scale) noexcept {
    int k = 0;
    for (; k <= K - 4; k += 4) {
        float32x4_t v[12];
        for (int i = 0; i < 12; ++i) {
            v[i] = vld1q_f32(input + i * ir_step);
        }

        transpose_12x4_f32(v[0], v[1], v[2], v[3], v[4], v[5],
                           v[6], v[7], v[8], v[9], v[10], v[11]);

        const float32x4_t s = vdupq_n_f32(scale);
        for (int i = 0; i < 12; ++i) {
            vst1q_f32(output + i * 4, vmulq_f32(v[i], s));
        }

        output += 12 * 4;
        input  += 4;
    }
    for (; k < K; ++k) {
        for (int i = 0; i < 12; ++i) {
            *output++ = input[i * ir_step] * scale;
        }
        input += 1;
    }
}

inline void pack_trans_n16_f32(float* NNOPS_RESTRICT output,
                               const float* NNOPS_RESTRICT input,
                               int ir_step, int K, float scale) noexcept {
    int k = 0;
    for (; k <= K - 4; k += 4) {
        float32x4_t v[16];
        for (int i = 0; i < 16; ++i) {
            v[i] = vld1q_f32(input + i * ir_step);
        }

        transpose_16x4_f32(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                           v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);

        const float32x4_t s = vdupq_n_f32(scale);
        for (int i = 0; i < 16; ++i) {
            vst1q_f32(output + i * 4, vmulq_f32(v[i], s));
        }

        output += 16 * 4;
        input  += 4;
    }
    for (; k < K; ++k) {
        for (int i = 0; i < 16; ++i) {
            *output++ = input[i * ir_step] * scale;
        }
        input += 1;
    }
}

// =========================================================================
//  RHS Copy pack  —  contiguous-row copy (no transpose), panels by N.
//  pack_copy_nN_f32: reads N-element rows, copies with optional scale.
// =========================================================================

inline void pack_copy_n1_f32(float* NNOPS_RESTRICT output,
                             const float* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale) noexcept {
    for (int k = 0; k < K; ++k) {
        *output++ = input[0] * scale;
        input += ir_step;
    }
}

inline void pack_copy_n4_f32(float* NNOPS_RESTRICT output,
                             const float* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale) noexcept {
    int k = 0;
    for (; k <= K - 4; k += 4) {
        float32x4_t v0 = vld1q_f32(input + 0 * ir_step);
        float32x4_t v1 = vld1q_f32(input + 1 * ir_step);
        float32x4_t v2 = vld1q_f32(input + 2 * ir_step);
        float32x4_t v3 = vld1q_f32(input + 3 * ir_step);

        const float32x4_t s = vdupq_n_f32(scale);
        vst1q_f32(output + 0 * 4, vmulq_f32(v0, s));
        vst1q_f32(output + 1 * 4, vmulq_f32(v1, s));
        vst1q_f32(output + 2 * 4, vmulq_f32(v2, s));
        vst1q_f32(output + 3 * 4, vmulq_f32(v3, s));

        output += 4 * 4;
        input  += 4 * ir_step;
    }
    for (; k < K; ++k) {
        output[0] = input[0] * scale;
        output[1] = input[1] * scale;
        output[2] = input[2] * scale;
        output[3] = input[3] * scale;
        output += 4;
        input  += ir_step;
    }
}

inline void pack_copy_n6_f32(float* NNOPS_RESTRICT output,
                             const float* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale) noexcept {
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < 6; ++i) {
            *output++ = input[i * ir_step] * scale;
        }
        input += 1;
    }
}

inline void pack_copy_n8_f32(float* NNOPS_RESTRICT output,
                             const float* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale) noexcept {
    int k = 0;
    for (; k <= K - 4; k += 4) {
        float32x4_t v0 = vld1q_f32(input + 0 * ir_step + 0 * 4);
        float32x4_t v1 = vld1q_f32(input + 0 * ir_step + 1 * 4);
        float32x4_t v2 = vld1q_f32(input + 1 * ir_step + 0 * 4);
        float32x4_t v3 = vld1q_f32(input + 1 * ir_step + 1 * 4);

        const float32x4_t s = vdupq_n_f32(scale);
        vst1q_f32(output + 0 * 4, vmulq_f32(v0, s));
        vst1q_f32(output + 1 * 4, vmulq_f32(v1, s));
        vst1q_f32(output + 2 * 4, vmulq_f32(v2, s));
        vst1q_f32(output + 3 * 4, vmulq_f32(v3, s));

        v0 = vld1q_f32(input + 2 * ir_step + 0 * 4);
        v1 = vld1q_f32(input + 2 * ir_step + 1 * 4);
        v2 = vld1q_f32(input + 3 * ir_step + 0 * 4);
        v3 = vld1q_f32(input + 3 * ir_step + 1 * 4);

        vst1q_f32(output + 4 * 4, vmulq_f32(v0, s));
        vst1q_f32(output + 5 * 4, vmulq_f32(v1, s));
        vst1q_f32(output + 6 * 4, vmulq_f32(v2, s));
        vst1q_f32(output + 7 * 4, vmulq_f32(v3, s));

        output += 4 * 8;
        input  += 4 * ir_step;
    }
    for (; k < K; ++k) {
        for (int i = 0; i < 8; ++i) {
            *output++ = input[i] * scale;
        }
        input += ir_step;
    }
}

inline void pack_copy_n12_f32(float* NNOPS_RESTRICT output,
                              const float* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale) noexcept {
    for (int k = 0; k < K; ++k) {
        float32x4_t v0 = vld1q_f32(input + 0 * 4);
        float32x4_t v1 = vld1q_f32(input + 1 * 4);
        float32x4_t v2 = vld1q_f32(input + 2 * 4);

        const float32x4_t s = vdupq_n_f32(scale);
        vst1q_f32(output + 0 * 4, vmulq_f32(v0, s));
        vst1q_f32(output + 1 * 4, vmulq_f32(v1, s));
        vst1q_f32(output + 2 * 4, vmulq_f32(v2, s));

        output += 12;
        input  += ir_step;
    }
}

inline void pack_copy_n16_f32(float* NNOPS_RESTRICT output,
                              const float* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale) noexcept {
    for (int k = 0; k < K; ++k) {
        float32x4_t v0 = vld1q_f32(input + 0 * 4);
        float32x4_t v1 = vld1q_f32(input + 1 * 4);
        float32x4_t v2 = vld1q_f32(input + 2 * 4);
        float32x4_t v3 = vld1q_f32(input + 3 * 4);

        const float32x4_t s = vdupq_n_f32(scale);
        vst1q_f32(output + 0 * 4, vmulq_f32(v0, s));
        vst1q_f32(output + 1 * 4, vmulq_f32(v1, s));
        vst1q_f32(output + 2 * 4, vmulq_f32(v2, s));
        vst1q_f32(output + 3 * 4, vmulq_f32(v3, s));

        output += 16;
        input  += ir_step;
    }
}

// =========================================================================
//  Transposed "copy" pack  —  for RHS when B is pre-transposed.
//  These read K elements from each of N input rows (stride = ir_step).
// =========================================================================

inline void pack_trans_n6_f32(float* NNOPS_RESTRICT output,
                              const float* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale) noexcept {
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < 6; ++i) {
            *output++ = input[i] * scale;
        }
        input += ir_step;
    }
}

// =========================================================================
//  Tiled pack dispatchers  —  decompose M/N into micro-panel sizes.
// =========================================================================

/// @brief Panel sizes for f32 LHS (A matrix rows).
/// Maps {large, medium, small} micro-panel heights indexed as [m_idx].
inline constexpr int mr_f32[3] = {8, 4, 1};

/// @brief Panel sizes for f32 RHS (B matrix columns).
inline constexpr int nr_f32[3] = {12, 4, 1};

}  // namespace nnops::backend::cpu::aarch64
