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
#include "backend/cpu/common/common_utils.hpp"
#include "transpose.hpp"

namespace nnops::backend::cpu::aarch64 {

// =========================================================================
//  LHS Transpose pack  —  panelled transpose for packed-A in GEMM
//  pack_trans_nN_f32: reads N rows, transposes N×4 blocks, scales by `scale`.
// =========================================================================

inline void pack_trans_n1_f32(float* NNOPS_RESTRICT output,
                              const float* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int mr = 1) noexcept {
    for (int k = 0; k < K; ++k) {
        *output++ = input[0] * scale;
        input += 1;  // consecutive k within the M×K source row
    }
}

inline void pack_trans_n4_f32(float* NNOPS_RESTRICT output,
                              const float* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int mr = 4) noexcept {
    // Rows past `mr` alias the last live row — the same convention the
    // MMA kernels use — so a partial-M panel needs no padding.
    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;
    const float* NNOPS_RESTRICT p0 = input + 0 * ir_step;
    const float* NNOPS_RESTRICT p1 = input + r1 * ir_step;
    const float* NNOPS_RESTRICT p2 = input + r2 * ir_step;
    const float* NNOPS_RESTRICT p3 = input + r3 * ir_step;

    const float32x4_t s = vdupq_n_f32(scale);
    int k = 0;
    for (; k <= K - 4; k += 4) {
        float32x4_t v0 = vld1q_f32(p0); p0 += 4;
        float32x4_t v1 = vld1q_f32(p1); p1 += 4;
        float32x4_t v2 = vld1q_f32(p2); p2 += 4;
        float32x4_t v3 = vld1q_f32(p3); p3 += 4;

        transpose_4x4_f32(v0, v1, v2, v3);

        vst1q_f32(output + 0 * 4, vmulq_f32(v0, s));
        vst1q_f32(output + 1 * 4, vmulq_f32(v1, s));
        vst1q_f32(output + 2 * 4, vmulq_f32(v2, s));
        vst1q_f32(output + 3 * 4, vmulq_f32(v3, s));

        output += 4 * 4;
    }
    // scalar tail
    for (; k < K; ++k) {
        *output++ = *(p0++) * scale;
        *output++ = *(p1++) * scale;
        *output++ = *(p2++) * scale;
        *output++ = *(p3++) * scale;
    }
}

inline void pack_trans_n8_f32(float* NNOPS_RESTRICT output,
                              const float* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int mr = 8) noexcept {
    // Rows past `mr` alias the last live row — the same convention the
    // MMA kernels use — so a partial-M panel needs no padding.
    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;
    const int r4 = (mr > 4) ? 4 : r3;
    const int r5 = (mr > 5) ? 5 : r4;
    const int r6 = (mr > 6) ? 6 : r5;
    const int r7 = (mr > 7) ? 7 : r6;
    const float* NNOPS_RESTRICT p0 = input + 0 * ir_step;
    const float* NNOPS_RESTRICT p1 = input + r1 * ir_step;
    const float* NNOPS_RESTRICT p2 = input + r2 * ir_step;
    const float* NNOPS_RESTRICT p3 = input + r3 * ir_step;
    const float* NNOPS_RESTRICT p4 = input + r4 * ir_step;
    const float* NNOPS_RESTRICT p5 = input + r5 * ir_step;
    const float* NNOPS_RESTRICT p6 = input + r6 * ir_step;
    const float* NNOPS_RESTRICT p7 = input + r7 * ir_step;

    const float32x4_t s = vdupq_n_f32(scale);
    int k = 0;
    for (; k <= K - 4; k += 4) {
        float32x4_t v0 = vld1q_f32(p0); p0 += 4;
        float32x4_t v1 = vld1q_f32(p1); p1 += 4;
        float32x4_t v2 = vld1q_f32(p2); p2 += 4;
        float32x4_t v3 = vld1q_f32(p3); p3 += 4;
        float32x4_t v4 = vld1q_f32(p4); p4 += 4;
        float32x4_t v5 = vld1q_f32(p5); p5 += 4;
        float32x4_t v6 = vld1q_f32(p6); p6 += 4;
        float32x4_t v7 = vld1q_f32(p7); p7 += 4;

        transpose_8x4_f32(v0, v1, v2, v3, v4, v5, v6, v7);

        vst1q_f32(output + 0 * 4, vmulq_f32(v0, s));
        vst1q_f32(output + 1 * 4, vmulq_f32(v1, s));
        vst1q_f32(output + 2 * 4, vmulq_f32(v2, s));
        vst1q_f32(output + 3 * 4, vmulq_f32(v3, s));
        vst1q_f32(output + 4 * 4, vmulq_f32(v4, s));
        vst1q_f32(output + 5 * 4, vmulq_f32(v5, s));
        vst1q_f32(output + 6 * 4, vmulq_f32(v6, s));
        vst1q_f32(output + 7 * 4, vmulq_f32(v7, s));

        output += 4 * 8;
    }
    // scalar tail
    for (; k < K; ++k) {
        *output++ = *(p0++) * scale;
        *output++ = *(p1++) * scale;
        *output++ = *(p2++) * scale;
        *output++ = *(p3++) * scale;
        *output++ = *(p4++) * scale;
        *output++ = *(p5++) * scale;
        *output++ = *(p6++) * scale;
        *output++ = *(p7++) * scale;
    }
}

inline void pack_trans_n12_f32(float* NNOPS_RESTRICT output,
                               const float* NNOPS_RESTRICT input,
                               int ir_step, int K, float scale, int mr = 12) noexcept {
    // Rows past `mr` alias the last live row — the same convention the
    // MMA kernels use — so a partial-M panel needs no padding.
    const int last = (mr > 1) ? mr - 1 : 0;
    const float* NNOPS_RESTRICT p[12];
    for (int i = 0; i < 12; ++i) {
        p[i] = input + ((i < mr) ? i : last) * ir_step;
    }

    const float32x4_t s = vdupq_n_f32(scale);
    int k = 0;
    for (; k <= K - 4; k += 4) {
        float32x4_t v[12];
        for (int i = 0; i < 12; ++i) {
            v[i] = vld1q_f32(p[i]); p[i] += 4;
        }

        transpose_12x4_f32(v[0], v[1], v[2], v[3], v[4], v[5],
                           v[6], v[7], v[8], v[9], v[10], v[11]);

        for (int i = 0; i < 12; ++i) {
            vst1q_f32(output + i * 4, vmulq_f32(v[i], s));
        }

        output += 12 * 4;
    }
    for (; k < K; ++k) {
        for (int i = 0; i < 12; ++i) {
            *output++ = *(p[i]++) * scale;
        }
    }
}

inline void pack_trans_n16_f32(float* NNOPS_RESTRICT output,
                               const float* NNOPS_RESTRICT input,
                               int ir_step, int K, float scale, int mr = 16) noexcept {
    // Rows past `mr` alias the last live row — the same convention the
    // MMA kernels use — so a partial-M panel needs no padding.
    const int last = (mr > 1) ? mr - 1 : 0;
    const float* NNOPS_RESTRICT p[16];
    for (int i = 0; i < 16; ++i) {
        p[i] = input + ((i < mr) ? i : last) * ir_step;
    }

    const float32x4_t s = vdupq_n_f32(scale);
    int k = 0;
    for (; k <= K - 4; k += 4) {
        float32x4_t v[16];
        for (int i = 0; i < 16; ++i) {
            v[i] = vld1q_f32(p[i]); p[i] += 4;
        }

        transpose_16x4_f32(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                           v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);

        for (int i = 0; i < 16; ++i) {
            vst1q_f32(output + i * 4, vmulq_f32(v[i], s));
        }

        output += 16 * 4;
    }
    for (; k < K; ++k) {
        for (int i = 0; i < 16; ++i) {
            *output++ = *(p[i]++) * scale;
        }
    }
}

// =========================================================================
//  RHS Copy pack  —  contiguous-row copy (no transpose), panels by N.
//  pack_copy_nN_f32: reads N-element rows, copies with optional scale.
// =========================================================================

inline void pack_copy_n1_f32(float* NNOPS_RESTRICT output,
                             const float* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale, int nr = 1) noexcept {
    for (int k = 0; k < K; ++k) {
        *output++ = (nr > 0) ? input[0] * scale : 0.0f;
        input += ir_step;
    }
}

inline void pack_copy_n4_f32(float* NNOPS_RESTRICT output,
                             const float* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale, int nr = 4) noexcept {
    // One k loop serves every width: a full panel takes the plain loads, a
    // narrow one reads only its live columns and zero-fills the rest.
    const float32x4_t s = vdupq_n_f32(scale);
    int k = 0;
    for (; k <= K - 4; k += 4) {
        if (nr >= 4) {
            const float32x4_t v0 = vld1q_f32(input + 0 * ir_step);
            const float32x4_t v1 = vld1q_f32(input + 1 * ir_step);
            const float32x4_t v2 = vld1q_f32(input + 2 * ir_step);
            const float32x4_t v3 = vld1q_f32(input + 3 * ir_step);

            vst1q_f32(output + 0 * 4, vmulq_f32(v0, s));
            vst1q_f32(output + 1 * 4, vmulq_f32(v1, s));
            vst1q_f32(output + 2 * 4, vmulq_f32(v2, s));
            vst1q_f32(output + 3 * 4, vmulq_f32(v3, s));

            output += 4 * 4;
        } else {
            for (int j = 0; j < 4; ++j) {
                for (int i = 0; i < 4; ++i) {
                    output[i] = (i < nr) ? input[j * ir_step + i] * scale : 0.0f;
                }
                output += 4;
            }
        }
        input += 4 * ir_step;
    }
    for (; k < K; ++k) {
        for (int i = 0; i < 4; ++i) {
            *output++ = (i < nr) ? input[i] * scale : 0.0f;
        }
        input += ir_step;
    }
}

inline void pack_copy_n6_f32(float* NNOPS_RESTRICT output,
                             const float* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale, int nr = 6) noexcept {
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < 6; ++i) {
            *output++ = (i < nr) ? input[i * ir_step] * scale : 0.0f;
        }
        input += 1;
    }
}

inline void pack_copy_n8_f32(float* NNOPS_RESTRICT output,
                             const float* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale, int nr = 8) noexcept {
    // One k loop serves every width: a full panel takes the plain loads, a
    // narrow one reads only its live columns and zero-fills the rest.
    const float32x4_t s = vdupq_n_f32(scale);
    int k = 0;
    for (; k <= K - 4; k += 4) {
        if (nr >= 8) {
            const float32x4_t a0 = vld1q_f32(input + 0 * ir_step + 0 * 4);
            const float32x4_t a1 = vld1q_f32(input + 0 * ir_step + 1 * 4);
            const float32x4_t a2 = vld1q_f32(input + 1 * ir_step + 0 * 4);
            const float32x4_t a3 = vld1q_f32(input + 1 * ir_step + 1 * 4);

            vst1q_f32(output + 0 * 4, vmulq_f32(a0, s));
            vst1q_f32(output + 1 * 4, vmulq_f32(a1, s));
            vst1q_f32(output + 2 * 4, vmulq_f32(a2, s));
            vst1q_f32(output + 3 * 4, vmulq_f32(a3, s));

            const float32x4_t b0 = vld1q_f32(input + 2 * ir_step + 0 * 4);
            const float32x4_t b1 = vld1q_f32(input + 2 * ir_step + 1 * 4);
            const float32x4_t b2 = vld1q_f32(input + 3 * ir_step + 0 * 4);
            const float32x4_t b3 = vld1q_f32(input + 3 * ir_step + 1 * 4);

            vst1q_f32(output + 4 * 4, vmulq_f32(b0, s));
            vst1q_f32(output + 5 * 4, vmulq_f32(b1, s));
            vst1q_f32(output + 6 * 4, vmulq_f32(b2, s));
            vst1q_f32(output + 7 * 4, vmulq_f32(b3, s));

            output += 4 * 8;
        } else {
            for (int j = 0; j < 4; ++j) {
                for (int i = 0; i < 8; ++i) {
                    output[i] = (i < nr) ? input[j * ir_step + i] * scale : 0.0f;
                }
                output += 8;
            }
        }
        input += 4 * ir_step;
    }
    for (; k < K; ++k) {
        for (int i = 0; i < 8; ++i) {
            *output++ = (i < nr) ? input[i] * scale : 0.0f;
        }
        input += ir_step;
    }
}

inline void pack_copy_n12_f32(float* NNOPS_RESTRICT output,
                              const float* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int nr = 12) noexcept {
    // One k loop serves every width: a full panel takes the plain loads, a
    // narrow one reads only its live columns and zero-fills the rest.
    const float32x4_t s = vdupq_n_f32(scale);
    for (int k = 0; k < K; ++k) {
        if (nr >= 12) {
            const float32x4_t v0 = vld1q_f32(input + 0 * 4);
            const float32x4_t v1 = vld1q_f32(input + 1 * 4);
            const float32x4_t v2 = vld1q_f32(input + 2 * 4);

            vst1q_f32(output + 0 * 4, vmulq_f32(v0, s));
            vst1q_f32(output + 1 * 4, vmulq_f32(v1, s));
            vst1q_f32(output + 2 * 4, vmulq_f32(v2, s));
        } else {
            for (int i = 0; i < 12; ++i) {
                output[i] = (i < nr) ? input[i] * scale : 0.0f;
            }
        }
        output += 12;
        input  += ir_step;
    }
}

inline void pack_copy_n16_f32(float* NNOPS_RESTRICT output,
                              const float* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int nr = 16) noexcept {
    // One k loop serves every width: a full panel takes the plain loads, a
    // narrow one reads only its live columns and zero-fills the rest.
    const float32x4_t s = vdupq_n_f32(scale);
    for (int k = 0; k < K; ++k) {
        if (nr >= 16) {
            const float32x4_t v0 = vld1q_f32(input + 0 * 4);
            const float32x4_t v1 = vld1q_f32(input + 1 * 4);
            const float32x4_t v2 = vld1q_f32(input + 2 * 4);
            const float32x4_t v3 = vld1q_f32(input + 3 * 4);

            vst1q_f32(output + 0 * 4, vmulq_f32(v0, s));
            vst1q_f32(output + 1 * 4, vmulq_f32(v1, s));
            vst1q_f32(output + 2 * 4, vmulq_f32(v2, s));
            vst1q_f32(output + 3 * 4, vmulq_f32(v3, s));
        } else {
            for (int i = 0; i < 16; ++i) {
                output[i] = (i < nr) ? input[i] * scale : 0.0f;
            }
        }
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
                              int ir_step, int K, float scale, int nr = 6) noexcept {
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < 6; ++i) {
            *output++ = (i < nr) ? input[i] * scale : 0.0f;
        }
        input += ir_step;
    }
}

// =========================================================================
//  Tiled pack dispatchers  —  decompose M/N into micro-panel sizes.
// =========================================================================

/// @brief Panel sizes for f32 LHS (A matrix rows), shared by both routes.
/// Maps {large, medium, small} micro-panel heights indexed as [m_idx].
/// The packed route interleaves four rows per vector, so it never leaves the
/// register file. The unpacked-A ("direct") route reads A row-major, and a 4-k
/// unroll there holds each row's slice live across the whole unroll: the
/// register set becomes mr*nr/4 + mr + nr/4, which at nr=12 is 35 > 32 NEON
/// registers for mr=8 and spills on every k. The 8x12 kernel therefore runs
/// un-unrolled, which is what lets both routes share mr=8 instead of the direct
/// path dropping to mr=6.
inline constexpr int mr_f32[3] = {8, 4, 1};

/// @brief Panel sizes for f32 RHS (B matrix columns).
inline constexpr int nr_f32[3] = {12, 4, 1};

}  // namespace nnops::backend::cpu::aarch64
