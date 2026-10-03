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
        input += 1;  // consecutive k within the M×K source row
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
        float32x4_t v0  = vld1q_f32(input + 0 * ir_step);
        float32x4_t v1  = vld1q_f32(input + 1 * ir_step);
        float32x4_t v2  = vld1q_f32(input + 2 * ir_step);
        float32x4_t v3  = vld1q_f32(input + 3 * ir_step);
        float32x4_t v4  = vld1q_f32(input + 4 * ir_step);
        float32x4_t v5  = vld1q_f32(input + 5 * ir_step);
        float32x4_t v6  = vld1q_f32(input + 6 * ir_step);
        float32x4_t v7  = vld1q_f32(input + 7 * ir_step);
        float32x4_t v8  = vld1q_f32(input + 8 * ir_step);
        float32x4_t v9  = vld1q_f32(input + 9 * ir_step);
        float32x4_t v10 = vld1q_f32(input + 10 * ir_step);
        float32x4_t v11 = vld1q_f32(input + 11 * ir_step);

        transpose_12x4_f32(v0, v1, v2, v3, v4, v5,
                           v6, v7, v8, v9, v10, v11);

        const float32x4_t s = vdupq_n_f32(scale);
        vst1q_f32(output + 0 * 4,  vmulq_f32(v0,  s));
        vst1q_f32(output + 1 * 4,  vmulq_f32(v1,  s));
        vst1q_f32(output + 2 * 4,  vmulq_f32(v2,  s));
        vst1q_f32(output + 3 * 4,  vmulq_f32(v3,  s));
        vst1q_f32(output + 4 * 4,  vmulq_f32(v4,  s));
        vst1q_f32(output + 5 * 4,  vmulq_f32(v5,  s));
        vst1q_f32(output + 6 * 4,  vmulq_f32(v6,  s));
        vst1q_f32(output + 7 * 4,  vmulq_f32(v7,  s));
        vst1q_f32(output + 8 * 4,  vmulq_f32(v8,  s));
        vst1q_f32(output + 9 * 4,  vmulq_f32(v9,  s));
        vst1q_f32(output + 10 * 4, vmulq_f32(v10, s));
        vst1q_f32(output + 11 * 4, vmulq_f32(v11, s));

        output += 12 * 4;
        input  += 4;
    }
    for (; k < K; ++k) {
        *output++ = input[0 * ir_step] * scale;
        *output++ = input[1 * ir_step] * scale;
        *output++ = input[2 * ir_step] * scale;
        *output++ = input[3 * ir_step] * scale;
        *output++ = input[4 * ir_step] * scale;
        *output++ = input[5 * ir_step] * scale;
        *output++ = input[6 * ir_step] * scale;
        *output++ = input[7 * ir_step] * scale;
        *output++ = input[8 * ir_step] * scale;
        *output++ = input[9 * ir_step] * scale;
        *output++ = input[10 * ir_step] * scale;
        *output++ = input[11 * ir_step] * scale;
        input += 1;
    }
}

inline void pack_trans_n16_f32(float* NNOPS_RESTRICT output,
                               const float* NNOPS_RESTRICT input,
                               int ir_step, int K, float scale) noexcept {
    int k = 0;
    for (; k <= K - 4; k += 4) {
        float32x4_t v0  = vld1q_f32(input + 0 * ir_step);
        float32x4_t v1  = vld1q_f32(input + 1 * ir_step);
        float32x4_t v2  = vld1q_f32(input + 2 * ir_step);
        float32x4_t v3  = vld1q_f32(input + 3 * ir_step);
        float32x4_t v4  = vld1q_f32(input + 4 * ir_step);
        float32x4_t v5  = vld1q_f32(input + 5 * ir_step);
        float32x4_t v6  = vld1q_f32(input + 6 * ir_step);
        float32x4_t v7  = vld1q_f32(input + 7 * ir_step);
        float32x4_t v8  = vld1q_f32(input + 8 * ir_step);
        float32x4_t v9  = vld1q_f32(input + 9 * ir_step);
        float32x4_t v10 = vld1q_f32(input + 10 * ir_step);
        float32x4_t v11 = vld1q_f32(input + 11 * ir_step);
        float32x4_t v12 = vld1q_f32(input + 12 * ir_step);
        float32x4_t v13 = vld1q_f32(input + 13 * ir_step);
        float32x4_t v14 = vld1q_f32(input + 14 * ir_step);
        float32x4_t v15 = vld1q_f32(input + 15 * ir_step);

        transpose_16x4_f32(v0, v1, v2, v3, v4, v5, v6, v7,
                           v8, v9, v10, v11, v12, v13, v14, v15);

        const float32x4_t s = vdupq_n_f32(scale);
        vst1q_f32(output + 0 * 4,  vmulq_f32(v0,  s));
        vst1q_f32(output + 1 * 4,  vmulq_f32(v1,  s));
        vst1q_f32(output + 2 * 4,  vmulq_f32(v2,  s));
        vst1q_f32(output + 3 * 4,  vmulq_f32(v3,  s));
        vst1q_f32(output + 4 * 4,  vmulq_f32(v4,  s));
        vst1q_f32(output + 5 * 4,  vmulq_f32(v5,  s));
        vst1q_f32(output + 6 * 4,  vmulq_f32(v6,  s));
        vst1q_f32(output + 7 * 4,  vmulq_f32(v7,  s));
        vst1q_f32(output + 8 * 4,  vmulq_f32(v8,  s));
        vst1q_f32(output + 9 * 4,  vmulq_f32(v9,  s));
        vst1q_f32(output + 10 * 4, vmulq_f32(v10, s));
        vst1q_f32(output + 11 * 4, vmulq_f32(v11, s));
        vst1q_f32(output + 12 * 4, vmulq_f32(v12, s));
        vst1q_f32(output + 13 * 4, vmulq_f32(v13, s));
        vst1q_f32(output + 14 * 4, vmulq_f32(v14, s));
        vst1q_f32(output + 15 * 4, vmulq_f32(v15, s));

        output += 16 * 4;
        input  += 4;
    }
    for (; k < K; ++k) {
        *output++ = input[0 * ir_step] * scale;
        *output++ = input[1 * ir_step] * scale;
        *output++ = input[2 * ir_step] * scale;
        *output++ = input[3 * ir_step] * scale;
        *output++ = input[4 * ir_step] * scale;
        *output++ = input[5 * ir_step] * scale;
        *output++ = input[6 * ir_step] * scale;
        *output++ = input[7 * ir_step] * scale;
        *output++ = input[8 * ir_step] * scale;
        *output++ = input[9 * ir_step] * scale;
        *output++ = input[10 * ir_step] * scale;
        *output++ = input[11 * ir_step] * scale;
        *output++ = input[12 * ir_step] * scale;
        *output++ = input[13 * ir_step] * scale;
        *output++ = input[14 * ir_step] * scale;
        *output++ = input[15 * ir_step] * scale;
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

/// @brief LHS panel heights for the unpacked-A (direct) path.
/// The direct kernels read A row-major, so each row's 4-k slice is held live
/// across the whole 4-k unroll: the register set is mr*nr/4 + mr + nr/4. At
/// nr=12 that reaches 35 > 32 NEON registers for mr=8 and the kernel spills on
/// every k, but 27 for mr=6, which is the largest spill-free height. The pack
/// path keeps mr=8 — its interleaved A layout carries four rows per vector and
/// never exceeds the budget.
inline constexpr int mr_f32_direct[3] = {6, 4, 1};

/// @brief Panel sizes for f32 RHS (B matrix columns).
inline constexpr int nr_f32[3] = {12, 4, 1};

}  // namespace nnops::backend::cpu::aarch64
