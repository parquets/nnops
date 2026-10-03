#pragma once
/// @file pack_f16.hpp
/// @brief x86_64 F16C pack kernels for GEMM LHS/RHS (float16).
///
/// Data is stored as uint16_t (half) in memory. F16C intrinsics
/// (_mm256_cvtph_ps / _mm256_cvtps_ph) convert on the fly between the
/// fp16 storage format and fp32 accumulators.
///
/// Reference: nn_compute/src/cpu/kernel/pack/x86_64/pack_f16.hpp

#include <immintrin.h>
#include "nnops/detail/half.hpp"
#include "backend/cpu/common/restrict.hpp"
#include "transpose.hpp"

namespace nnops::backend::cpu::x86_64 {

using nnops::backend::cpu::half;
using nnops::backend::cpu::half_to_float;
using nnops::backend::cpu::float_to_half;

// Helper: load 8 fp16 values and convert to __m256 (fp32)
namespace {
inline __m256 load_f16x8(const half* NNOPS_RESTRICT p) noexcept {
    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

// Helper: convert __m256 (fp32) to 8 fp16 values and store
inline void store_f16x8(half* NNOPS_RESTRICT p, __m256 v) noexcept {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p),
                     _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
}
}  // anonymous namespace

// =========================================================================
//  Transpose pack (f16)
//
//  The vector loops convert to f32 and run the SAME transpose_Nx8_f32 the
//  f32 packs use, then round back to f16 on store — no extra conversions,
//  since the scale multiply already round-trips through f32. This matters
//  because the pack stores its registers linearly and the mma kernels read
//  the panel as [K][n] (n contiguous per k): only the f32 transposes
//  produce that order. A 16-bit-lane butterfly cannot express it (mr=6
//  panels interleave two k's per register); an earlier __m128i transpose
//  family got this wrong and produced a 2x2-blocked order.
// =========================================================================

inline void pack_trans_n1_f16(half* NNOPS_RESTRICT output,
                              const half* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale) noexcept {
    for (int k = 0; k < K; ++k) {
        *output++ = float_to_half(half_to_float(input[0]) * scale);
        input += 1;  // consecutive k within the M×K source row
    }
}

inline void pack_trans_n4_f16(half* NNOPS_RESTRICT output,
                              const half* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale) noexcept {
    const half* NNOPS_RESTRICT p0 = input + 0 * ir_step;
    const half* NNOPS_RESTRICT p1 = input + 1 * ir_step;
    const half* NNOPS_RESTRICT p2 = input + 2 * ir_step;
    const half* NNOPS_RESTRICT p3 = input + 3 * ir_step;

    const __m256 v_scale = _mm256_set1_ps(scale);
    int k = 0;
    for (; k <= K - 8; k += 8) {
        __m256 v0 = load_f16x8(p0); p0 += 8;
        __m256 v1 = load_f16x8(p1); p1 += 8;
        __m256 v2 = load_f16x8(p2); p2 += 8;
        __m256 v3 = load_f16x8(p3); p3 += 8;

        transpose_4x8_f32(v0, v1, v2, v3);

        v0 = _mm256_mul_ps(v0, v_scale);
        v1 = _mm256_mul_ps(v1, v_scale);
        v2 = _mm256_mul_ps(v2, v_scale);
        v3 = _mm256_mul_ps(v3, v_scale);

        store_f16x8(output + 0 * 8, v0);
        store_f16x8(output + 1 * 8, v1);
        store_f16x8(output + 2 * 8, v2);
        store_f16x8(output + 3 * 8, v3);

        output += 4 * 8;
    }
    // scalar tail (p0..p3 already point at k = 8*nblocks in each row)
    for (; k < K; ++k) {
        *output++ = float_to_half(half_to_float(*p0++) * scale);
        *output++ = float_to_half(half_to_float(*p1++) * scale);
        *output++ = float_to_half(half_to_float(*p2++) * scale);
        *output++ = float_to_half(half_to_float(*p3++) * scale);
    }
}

inline void pack_trans_n6_f16(half* NNOPS_RESTRICT output,
                              const half* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale) noexcept {
    const half* NNOPS_RESTRICT p0 = input + 0 * ir_step;
    const half* NNOPS_RESTRICT p1 = input + 1 * ir_step;
    const half* NNOPS_RESTRICT p2 = input + 2 * ir_step;
    const half* NNOPS_RESTRICT p3 = input + 3 * ir_step;
    const half* NNOPS_RESTRICT p4 = input + 4 * ir_step;
    const half* NNOPS_RESTRICT p5 = input + 5 * ir_step;

    const __m256 v_scale = _mm256_set1_ps(scale);
    int k = 0;
    for (; k <= K - 8; k += 8) {
        __m256 v0 = load_f16x8(p0); p0 += 8;
        __m256 v1 = load_f16x8(p1); p1 += 8;
        __m256 v2 = load_f16x8(p2); p2 += 8;
        __m256 v3 = load_f16x8(p3); p3 += 8;
        __m256 v4 = load_f16x8(p4); p4 += 8;
        __m256 v5 = load_f16x8(p5); p5 += 8;

        transpose_6x8_f32(v0, v1, v2, v3, v4, v5);

        v0 = _mm256_mul_ps(v0, v_scale);
        v1 = _mm256_mul_ps(v1, v_scale);
        v2 = _mm256_mul_ps(v2, v_scale);
        v3 = _mm256_mul_ps(v3, v_scale);
        v4 = _mm256_mul_ps(v4, v_scale);
        v5 = _mm256_mul_ps(v5, v_scale);

        store_f16x8(output + 0 * 8, v0);
        store_f16x8(output + 1 * 8, v1);
        store_f16x8(output + 2 * 8, v2);
        store_f16x8(output + 3 * 8, v3);
        store_f16x8(output + 4 * 8, v4);
        store_f16x8(output + 5 * 8, v5);

        output += 6 * 8;
    }
    for (; k < K; ++k) {
        *output++ = float_to_half(half_to_float(*p0++) * scale);
        *output++ = float_to_half(half_to_float(*p1++) * scale);
        *output++ = float_to_half(half_to_float(*p2++) * scale);
        *output++ = float_to_half(half_to_float(*p3++) * scale);
        *output++ = float_to_half(half_to_float(*p4++) * scale);
        *output++ = float_to_half(half_to_float(*p5++) * scale);
    }
}

inline void pack_trans_n8_f16(half* NNOPS_RESTRICT output,
                              const half* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale) noexcept {
    const half* p[8];
    for (int i = 0; i < 8; ++i) {
        p[i] = input + i * ir_step;
    }

    const __m256 v_scale = _mm256_set1_ps(scale);
    int k = 0;
    for (; k <= K - 8; k += 8) {
        __m256 v[8];
        for (int i = 0; i < 8; ++i) {
            v[i] = load_f16x8(p[i]); p[i] += 8;
        }

        transpose_8x8_f32(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);

        for (int i = 0; i < 8; ++i) {
            v[i] = _mm256_mul_ps(v[i], v_scale);
            store_f16x8(output + i * 8, v[i]);
        }

        output += 8 * 8;
    }
    for (; k < K; ++k) {
        for (int i = 0; i < 8; ++i) {
            *output++ = float_to_half(half_to_float(*p[i]++) * scale);
        }
    }
}

inline void pack_trans_n16_f16(half* NNOPS_RESTRICT output,
                               const half* NNOPS_RESTRICT input,
                               int ir_step, int K, float scale) noexcept {
    const half* p[16];
    for (int i = 0; i < 16; ++i) {
        p[i] = input + i * ir_step;
    }

    const __m256 v_scale = _mm256_set1_ps(scale);
    int k = 0;
    for (; k <= K - 8; k += 8) {
        __m256 v[16];
        for (int i = 0; i < 16; ++i) {
            v[i] = load_f16x8(p[i]); p[i] += 8;
        }

        transpose_16x8_f32(
            v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
            v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);

        for (int i = 0; i < 16; ++i) {
            v[i] = _mm256_mul_ps(v[i], v_scale);
            store_f16x8(output + i * 8, v[i]);
        }

        output += 16 * 8;
    }
    for (; k < K; ++k) {
        for (int i = 0; i < 16; ++i) {
            *output++ = float_to_half(half_to_float(*p[i]++) * scale);
        }
    }
}

inline void pack_trans_n24_f16(half* NNOPS_RESTRICT output,
                               const half* NNOPS_RESTRICT input,
                               int ir_step, int K, float scale) noexcept {
    // Pure scalar path for wide N=24 panel — rarely called.
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < 24; ++i) {
            *output++ = float_to_half(half_to_float(input[i * ir_step]) * scale);
        }
        input += 1;
    }
}

// =========================================================================
//  RHS Copy pack (f16)
// =========================================================================

inline void pack_copy_n1_f16(half* NNOPS_RESTRICT output,
                             const half* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale) noexcept {
    for (int k = 0; k < K; ++k) {
        *output++ = float_to_half(half_to_float(input[0]) * scale);
        input += ir_step;
    }
}

inline void pack_copy_n4_f16(half* NNOPS_RESTRICT output,
                             const half* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale) noexcept {
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < 4; ++i) {
            *output++ = float_to_half(half_to_float(input[i]) * scale);
        }
        input += ir_step;
    }
}

inline void pack_copy_n6_f16(half* NNOPS_RESTRICT output,
                             const half* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale) noexcept {
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < 6; ++i) {
            *output++ = float_to_half(half_to_float(input[i]) * scale);
        }
        input += ir_step;
    }
}

inline void pack_copy_n8_f16(half* NNOPS_RESTRICT output,
                             const half* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale) noexcept {
    const __m256 v_scale = _mm256_set1_ps(scale);
    int k = 0;
    for (; k <= K - 2; k += 2) {
        __m256 v0 = load_f16x8(input + 0 * ir_step);
        __m256 v1 = load_f16x8(input + 1 * ir_step);
        v0 = _mm256_mul_ps(v0, v_scale);
        v1 = _mm256_mul_ps(v1, v_scale);
        store_f16x8(output + 0 * 8, v0);
        store_f16x8(output + 1 * 8, v1);
        output += 2 * 8;
        input  += 2 * ir_step;
    }
    for (; k < K; ++k) {
        __m256 v0 = load_f16x8(input);
        v0 = _mm256_mul_ps(v0, v_scale);
        store_f16x8(output, v0);
        output += 8;
        input  += ir_step;
    }
}

inline void pack_copy_n16_f16(half* NNOPS_RESTRICT output,
                              const half* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale) noexcept {
    const __m256 v_scale = _mm256_set1_ps(scale);
    for (int k = 0; k < K; ++k) {
        __m256 v0 = load_f16x8(input + 0);
        __m256 v1 = load_f16x8(input + 8);

        v0 = _mm256_mul_ps(v0, v_scale);
        v1 = _mm256_mul_ps(v1, v_scale);

        store_f16x8(output + 0 * 8, v0);
        store_f16x8(output + 1 * 8, v1);

        output += 16;
        input  += ir_step;
    }
}

inline void pack_copy_n24_f16(half* NNOPS_RESTRICT output,
                              const half* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale) noexcept {
    const __m256 v_scale = _mm256_set1_ps(scale);
    for (int k = 0; k < K; ++k) {
        __m256 v0 = load_f16x8(input + 0 * 8);
        __m256 v1 = load_f16x8(input + 1 * 8);
        __m256 v2 = load_f16x8(input + 2 * 8);

        v0 = _mm256_mul_ps(v0, v_scale);
        v1 = _mm256_mul_ps(v1, v_scale);
        v2 = _mm256_mul_ps(v2, v_scale);

        store_f16x8(output + 0 * 8, v0);
        store_f16x8(output + 1 * 8, v1);
        store_f16x8(output + 2 * 8, v2);

        output += 24;
        input  += ir_step;
    }
}

// =========================================================================
//  Panel size constants (f16)  —  tuned for x86_64 AVX2/F16C.
// =========================================================================

/// @brief f16 LHS micro-panel heights.
inline constexpr int mr_f16[3] = {6, 4, 1};

/// @brief f16 RHS micro-panel widths.
inline constexpr int nr_f16[3] = {16, 8, 1};

}  // namespace nnops::backend::cpu::x86_64
