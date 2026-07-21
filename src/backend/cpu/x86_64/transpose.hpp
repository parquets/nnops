#pragma once
/// @file transpose.hpp
/// @brief x86_64 SSE/AVX transpose primitives for GEMM micro-kernels.
///
/// Two register widths are used:
///   - __m128  (4×float32 or 8×float16) — packed float/intrinsics
///   - __m256  (8×float32) — packed float intrinsics
///
/// For fp16, data is held in __m128i and transposed as 16-bit lanes;
/// the caller is responsible for half <-> float conversion.
///
/// Reference: nn_compute/src/cpu/kernel/transpose/x86_64/transpose_x86.hpp

#include <immintrin.h>
#include "backend/cpu/common/restrict.hpp"

namespace nnops::backend::cpu::x86_64 {

// =========================================================================
//  f32  —  __m128 (4-element)  transposes
// =========================================================================

inline void transpose_2x4_f32(__m128& v_r0, __m128& v_r1) noexcept {
    const __m128 tmp0 = _mm_unpacklo_ps(v_r0, v_r1);
    const __m128 tmp1 = _mm_unpackhi_ps(v_r0, v_r1);
    v_r0 = _mm_shuffle_ps(tmp0, tmp1, _MM_SHUFFLE(1, 0, 1, 0));
    v_r1 = _mm_shuffle_ps(tmp0, tmp1, _MM_SHUFFLE(3, 2, 3, 2));
}

inline void transpose_4x4_f32(__m128& v_r0, __m128& v_r1, __m128& v_r2, __m128& v_r3) noexcept {
    const __m128 tmp0 = _mm_unpacklo_ps(v_r0, v_r1);
    const __m128 tmp1 = _mm_unpackhi_ps(v_r0, v_r1);
    const __m128 tmp2 = _mm_unpacklo_ps(v_r2, v_r3);
    const __m128 tmp3 = _mm_unpackhi_ps(v_r2, v_r3);

    v_r0 = _mm_movelh_ps(tmp0, tmp2);
    v_r1 = _mm_movehl_ps(tmp2, tmp0);
    v_r2 = _mm_movelh_ps(tmp1, tmp3);
    v_r3 = _mm_movehl_ps(tmp3, tmp1);
}

inline void transpose_6x4_f32(__m128& v_a, __m128& v_b, __m128& v_c,
                               __m128& v_d, __m128& v_e, __m128& v_f) noexcept {
    const __m128 tmp0 = _mm_unpacklo_ps(v_a, v_b);
    const __m128 tmp1 = _mm_unpacklo_ps(v_c, v_d);
    const __m128 tmp2 = _mm_unpacklo_ps(v_e, v_f);
    const __m128 tmp3 = _mm_unpackhi_ps(v_a, v_b);
    const __m128 tmp4 = _mm_unpackhi_ps(v_c, v_d);
    const __m128 tmp5 = _mm_unpackhi_ps(v_e, v_f);

    v_a = _mm_shuffle_ps(tmp0, tmp1, _MM_SHUFFLE(1, 0, 1, 0));
    v_b = _mm_shuffle_ps(tmp2, tmp3, _MM_SHUFFLE(1, 0, 1, 0));
    v_c = _mm_shuffle_ps(tmp4, tmp5, _MM_SHUFFLE(1, 0, 1, 0));
    v_d = _mm_shuffle_ps(tmp0, tmp1, _MM_SHUFFLE(3, 2, 3, 2));
    v_e = _mm_shuffle_ps(tmp2, tmp3, _MM_SHUFFLE(3, 2, 3, 2));
    v_f = _mm_shuffle_ps(tmp4, tmp5, _MM_SHUFFLE(3, 2, 3, 2));
}

inline void transpose_8x4_f32(__m128& v_a, __m128& v_b, __m128& v_c, __m128& v_d,
                               __m128& v_e, __m128& v_f, __m128& v_g, __m128& v_h) noexcept {
    const __m128 t0 = _mm_unpacklo_ps(v_a, v_b);
    const __m128 t1 = _mm_unpackhi_ps(v_a, v_b);
    const __m128 t2 = _mm_unpacklo_ps(v_c, v_d);
    const __m128 t3 = _mm_unpackhi_ps(v_c, v_d);
    const __m128 t4 = _mm_unpacklo_ps(v_e, v_f);
    const __m128 t5 = _mm_unpackhi_ps(v_e, v_f);
    const __m128 t6 = _mm_unpacklo_ps(v_g, v_h);
    const __m128 t7 = _mm_unpackhi_ps(v_g, v_h);

    v_a = _mm_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    v_b = _mm_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    v_c = _mm_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    v_d = _mm_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    v_e = _mm_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    v_f = _mm_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    v_g = _mm_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    v_h = _mm_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
}

inline void transpose_12x4_f32(
    __m128& v_r0, __m128& v_r1, __m128& v_r2, __m128& v_r3,
    __m128& v_r4, __m128& v_r5, __m128& v_r6, __m128& v_r7,
    __m128& v_r8, __m128& v_r9, __m128& v_ra, __m128& v_rb) noexcept {

    const __m128 t0 = _mm_unpacklo_ps(v_r0, v_r1);
    const __m128 t1 = _mm_unpacklo_ps(v_r2, v_r3);
    const __m128 t2 = _mm_unpacklo_ps(v_r4, v_r5);
    const __m128 t3 = _mm_unpacklo_ps(v_r6, v_r7);
    const __m128 t4 = _mm_unpacklo_ps(v_r8, v_r9);
    const __m128 t5 = _mm_unpacklo_ps(v_ra, v_rb);

    const __m128 t6 = _mm_unpackhi_ps(v_r0, v_r1);
    const __m128 t7 = _mm_unpackhi_ps(v_r2, v_r3);
    const __m128 t8 = _mm_unpackhi_ps(v_r4, v_r5);
    const __m128 t9 = _mm_unpackhi_ps(v_r6, v_r7);
    const __m128 ta = _mm_unpackhi_ps(v_r8, v_r9);
    const __m128 tb = _mm_unpackhi_ps(v_ra, v_rb);

    v_r0 = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(1, 0, 1, 0));
    v_r1 = _mm_shuffle_ps(t2, t3, _MM_SHUFFLE(1, 0, 1, 0));
    v_r2 = _mm_shuffle_ps(t4, t5, _MM_SHUFFLE(1, 0, 1, 0));
    v_r3 = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(3, 2, 3, 2));
    v_r4 = _mm_shuffle_ps(t2, t3, _MM_SHUFFLE(3, 2, 3, 2));
    v_r5 = _mm_shuffle_ps(t4, t5, _MM_SHUFFLE(3, 2, 3, 2));
    v_r6 = _mm_shuffle_ps(t6, t7, _MM_SHUFFLE(1, 0, 1, 0));
    v_r7 = _mm_shuffle_ps(t8, t9, _MM_SHUFFLE(1, 0, 1, 0));
    v_r8 = _mm_shuffle_ps(ta, tb, _MM_SHUFFLE(1, 0, 1, 0));
    v_r9 = _mm_shuffle_ps(t6, t7, _MM_SHUFFLE(3, 2, 3, 2));
    v_ra = _mm_shuffle_ps(t8, t9, _MM_SHUFFLE(3, 2, 3, 2));
    v_rb = _mm_shuffle_ps(ta, tb, _MM_SHUFFLE(3, 2, 3, 2));
}

inline void transpose_16x4_f32(
    __m128& v_r0, __m128& v_r1, __m128& v_r2, __m128& v_r3,
    __m128& v_r4, __m128& v_r5, __m128& v_r6, __m128& v_r7,
    __m128& v_r8, __m128& v_r9, __m128& v_ra, __m128& v_rb,
    __m128& v_rc, __m128& v_rd, __m128& v_re, __m128& v_rf) noexcept {

    const __m128 t0 = _mm_unpacklo_ps(v_r0, v_r1);
    const __m128 t1 = _mm_unpacklo_ps(v_r2, v_r3);
    const __m128 t2 = _mm_unpacklo_ps(v_r4, v_r5);
    const __m128 t3 = _mm_unpacklo_ps(v_r6, v_r7);
    const __m128 t4 = _mm_unpacklo_ps(v_r8, v_r9);
    const __m128 t5 = _mm_unpacklo_ps(v_ra, v_rb);
    const __m128 t6 = _mm_unpacklo_ps(v_rc, v_rd);
    const __m128 t7 = _mm_unpacklo_ps(v_re, v_rf);

    const __m128 t8  = _mm_unpackhi_ps(v_r0, v_r1);
    const __m128 t9  = _mm_unpackhi_ps(v_r2, v_r3);
    const __m128 ta  = _mm_unpackhi_ps(v_r4, v_r5);
    const __m128 tb  = _mm_unpackhi_ps(v_r6, v_r7);
    const __m128 tc  = _mm_unpackhi_ps(v_r8, v_r9);
    const __m128 td  = _mm_unpackhi_ps(v_ra, v_rb);
    const __m128 te  = _mm_unpackhi_ps(v_rc, v_rd);
    const __m128 tf  = _mm_unpackhi_ps(v_re, v_rf);

    v_r0 = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(1, 0, 1, 0));
    v_r1 = _mm_shuffle_ps(t2, t3, _MM_SHUFFLE(1, 0, 1, 0));
    v_r2 = _mm_shuffle_ps(t4, t5, _MM_SHUFFLE(1, 0, 1, 0));
    v_r3 = _mm_shuffle_ps(t6, t7, _MM_SHUFFLE(1, 0, 1, 0));
    v_r4 = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(3, 2, 3, 2));
    v_r5 = _mm_shuffle_ps(t2, t3, _MM_SHUFFLE(3, 2, 3, 2));
    v_r6 = _mm_shuffle_ps(t4, t5, _MM_SHUFFLE(3, 2, 3, 2));
    v_r7 = _mm_shuffle_ps(t6, t7, _MM_SHUFFLE(3, 2, 3, 2));
    v_r8 = _mm_shuffle_ps(t8, t9, _MM_SHUFFLE(1, 0, 1, 0));
    v_r9 = _mm_shuffle_ps(ta, tb, _MM_SHUFFLE(1, 0, 1, 0));
    v_ra = _mm_shuffle_ps(tc, td, _MM_SHUFFLE(1, 0, 1, 0));
    v_rb = _mm_shuffle_ps(te, tf, _MM_SHUFFLE(1, 0, 1, 0));
    v_rc = _mm_shuffle_ps(t8, t9, _MM_SHUFFLE(3, 2, 3, 2));
    v_rd = _mm_shuffle_ps(ta, tb, _MM_SHUFFLE(3, 2, 3, 2));
    v_re = _mm_shuffle_ps(tc, td, _MM_SHUFFLE(3, 2, 3, 2));
    v_rf = _mm_shuffle_ps(te, tf, _MM_SHUFFLE(3, 2, 3, 2));
}

// =========================================================================
//  f32  —  __m256 (8-element)  transposes
// =========================================================================

inline void transpose_2x8_f32(__m256& v_r0, __m256& v_r1) noexcept {
    const __m256 tmp0 = _mm256_unpacklo_ps(v_r0, v_r1);
    const __m256 tmp1 = _mm256_unpackhi_ps(v_r0, v_r1);
    v_r0 = _mm256_permute2f128_ps(tmp0, tmp1, _MM_SHUFFLE(0, 2, 0, 0));
    v_r1 = _mm256_permute2f128_ps(tmp0, tmp1, _MM_SHUFFLE(0, 3, 0, 1));
}

inline void transpose_4x8_f32(__m256& v_r0, __m256& v_r1, __m256& v_r2, __m256& v_r3) noexcept {
    const __m256 tmp0 = _mm256_unpacklo_ps(v_r0, v_r1);
    const __m256 tmp1 = _mm256_unpackhi_ps(v_r0, v_r1);
    const __m256 tmp2 = _mm256_unpacklo_ps(v_r2, v_r3);
    const __m256 tmp3 = _mm256_unpackhi_ps(v_r2, v_r3);

    const __m256 tmp4 = _mm256_shuffle_ps(tmp0, tmp2, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 tmp5 = _mm256_shuffle_ps(tmp0, tmp2, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 tmp6 = _mm256_shuffle_ps(tmp1, tmp3, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 tmp7 = _mm256_shuffle_ps(tmp1, tmp3, _MM_SHUFFLE(3, 2, 3, 2));

    v_r0 = _mm256_permute2f128_ps(tmp4, tmp5, _MM_SHUFFLE(0, 2, 0, 0));
    v_r1 = _mm256_permute2f128_ps(tmp6, tmp7, _MM_SHUFFLE(0, 2, 0, 0));
    v_r2 = _mm256_permute2f128_ps(tmp4, tmp5, _MM_SHUFFLE(0, 3, 0, 1));
    v_r3 = _mm256_permute2f128_ps(tmp6, tmp7, _MM_SHUFFLE(0, 3, 0, 1));
}

inline void transpose_6x8_f32(__m256& v_a, __m256& v_b, __m256& v_c,
                               __m256& v_d, __m256& v_e, __m256& v_f) noexcept {
    const __m256 t0 = _mm256_unpacklo_ps(v_a, v_b);
    const __m256 t1 = _mm256_unpackhi_ps(v_a, v_b);
    const __m256 t2 = _mm256_unpacklo_ps(v_c, v_d);
    const __m256 t3 = _mm256_unpackhi_ps(v_c, v_d);
    const __m256 t4 = _mm256_unpacklo_ps(v_e, v_f);
    const __m256 t5 = _mm256_unpackhi_ps(v_e, v_f);

    const __m256 u0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u1 = _mm256_shuffle_ps(t4, t0, _MM_SHUFFLE(3, 2, 1, 0));
    const __m256 u2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u3 = _mm256_shuffle_ps(t5, t1, _MM_SHUFFLE(3, 2, 1, 0));
    const __m256 u4 = _mm256_shuffle_ps(t2, t4, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u5 = _mm256_shuffle_ps(t3, t5, _MM_SHUFFLE(3, 2, 3, 2));

    v_a = _mm256_permute2f128_ps(u0, u1, _MM_SHUFFLE(0, 2, 0, 0));
    v_b = _mm256_permute2f128_ps(u4, u2, _MM_SHUFFLE(0, 2, 0, 0));
    v_c = _mm256_permute2f128_ps(u3, u5, _MM_SHUFFLE(0, 2, 0, 0));
    v_d = _mm256_permute2f128_ps(u0, u1, _MM_SHUFFLE(0, 3, 0, 1));
    v_e = _mm256_permute2f128_ps(u4, u2, _MM_SHUFFLE(0, 3, 0, 1));
    v_f = _mm256_permute2f128_ps(u3, u5, _MM_SHUFFLE(0, 3, 0, 1));
}

inline void transpose_8x8_f32(__m256& v_a, __m256& v_b, __m256& v_c, __m256& v_d,
                               __m256& v_e, __m256& v_f, __m256& v_g, __m256& v_h) noexcept {
    const __m256 t0 = _mm256_unpacklo_ps(v_a, v_b);
    const __m256 t1 = _mm256_unpackhi_ps(v_a, v_b);
    const __m256 t2 = _mm256_unpacklo_ps(v_c, v_d);
    const __m256 t3 = _mm256_unpackhi_ps(v_c, v_d);
    const __m256 t4 = _mm256_unpacklo_ps(v_e, v_f);
    const __m256 t5 = _mm256_unpackhi_ps(v_e, v_f);
    const __m256 t6 = _mm256_unpacklo_ps(v_g, v_h);
    const __m256 t7 = _mm256_unpackhi_ps(v_g, v_h);

    const __m256 u0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

    v_a = _mm256_permute2f128_ps(u0, u4, _MM_SHUFFLE(0, 2, 0, 0));
    v_b = _mm256_permute2f128_ps(u1, u5, _MM_SHUFFLE(0, 2, 0, 0));
    v_c = _mm256_permute2f128_ps(u2, u6, _MM_SHUFFLE(0, 2, 0, 0));
    v_d = _mm256_permute2f128_ps(u3, u7, _MM_SHUFFLE(0, 2, 0, 0));
    v_e = _mm256_permute2f128_ps(u0, u4, _MM_SHUFFLE(0, 3, 0, 1));
    v_f = _mm256_permute2f128_ps(u1, u5, _MM_SHUFFLE(0, 3, 0, 1));
    v_g = _mm256_permute2f128_ps(u2, u6, _MM_SHUFFLE(0, 3, 0, 1));
    v_h = _mm256_permute2f128_ps(u3, u7, _MM_SHUFFLE(0, 3, 0, 1));
}

inline void transpose_12x8_f32(
    __m256& v_r0, __m256& v_r1, __m256& v_r2, __m256& v_r3,
    __m256& v_r4, __m256& v_r5, __m256& v_r6, __m256& v_r7,
    __m256& v_r8, __m256& v_r9, __m256& v_ra, __m256& v_rb) noexcept {

    const __m256 t0 = _mm256_unpacklo_ps(v_r0, v_r1);
    const __m256 t1 = _mm256_unpackhi_ps(v_r0, v_r1);
    const __m256 t2 = _mm256_unpacklo_ps(v_r2, v_r3);
    const __m256 t3 = _mm256_unpackhi_ps(v_r2, v_r3);
    const __m256 t4 = _mm256_unpacklo_ps(v_r4, v_r5);
    const __m256 t5 = _mm256_unpackhi_ps(v_r4, v_r5);
    const __m256 t6 = _mm256_unpacklo_ps(v_r6, v_r7);
    const __m256 t7 = _mm256_unpackhi_ps(v_r6, v_r7);
    const __m256 t8 = _mm256_unpacklo_ps(v_r8, v_r9);
    const __m256 t9 = _mm256_unpackhi_ps(v_r8, v_r9);
    const __m256 ta = _mm256_unpacklo_ps(v_ra, v_rb);
    const __m256 tb = _mm256_unpackhi_ps(v_ra, v_rb);

    const __m256 u0  = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u1  = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u2  = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u3  = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u4  = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u5  = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u6  = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u7  = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u8  = _mm256_shuffle_ps(t8, ta, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u9  = _mm256_shuffle_ps(t8, ta, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u10 = _mm256_shuffle_ps(t9, tb, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u11 = _mm256_shuffle_ps(t9, tb, _MM_SHUFFLE(3, 2, 3, 2));

    v_r0 = _mm256_permute2f128_ps(u0, u4, _MM_SHUFFLE(0, 2, 0, 0));
    v_r1 = _mm256_permute2f128_ps(u8, u1, _MM_SHUFFLE(0, 2, 0, 0));
    v_r2 = _mm256_permute2f128_ps(u5, u9, _MM_SHUFFLE(0, 2, 0, 0));
    v_r3 = _mm256_permute2f128_ps(u2, u6, _MM_SHUFFLE(0, 2, 0, 0));
    v_r4 = _mm256_permute2f128_ps(u10, u3, _MM_SHUFFLE(0, 2, 0, 0));
    v_r5 = _mm256_permute2f128_ps(u7, u11, _MM_SHUFFLE(0, 2, 0, 0));
    v_r6 = _mm256_permute2f128_ps(u0, u4, _MM_SHUFFLE(0, 3, 0, 1));
    v_r7 = _mm256_permute2f128_ps(u8, u1, _MM_SHUFFLE(0, 3, 0, 1));
    v_r8 = _mm256_permute2f128_ps(u5, u9, _MM_SHUFFLE(0, 3, 0, 1));
    v_r9 = _mm256_permute2f128_ps(u2, u6, _MM_SHUFFLE(0, 3, 0, 1));
    v_ra = _mm256_permute2f128_ps(u10, u3, _MM_SHUFFLE(0, 3, 0, 1));
    v_rb = _mm256_permute2f128_ps(u7, u11, _MM_SHUFFLE(0, 3, 0, 1));
}

inline void transpose_16x8_f32(
    __m256& v_r0, __m256& v_r1, __m256& v_r2, __m256& v_r3,
    __m256& v_r4, __m256& v_r5, __m256& v_r6, __m256& v_r7,
    __m256& v_r8, __m256& v_r9, __m256& v_ra, __m256& v_rb,
    __m256& v_rc, __m256& v_rd, __m256& v_re, __m256& v_rf) noexcept {

    const __m256 t0  = _mm256_unpacklo_ps(v_r0, v_r1);
    const __m256 t1  = _mm256_unpackhi_ps(v_r0, v_r1);
    const __m256 t2  = _mm256_unpacklo_ps(v_r2, v_r3);
    const __m256 t3  = _mm256_unpackhi_ps(v_r2, v_r3);
    const __m256 t4  = _mm256_unpacklo_ps(v_r4, v_r5);
    const __m256 t5  = _mm256_unpackhi_ps(v_r4, v_r5);
    const __m256 t6  = _mm256_unpacklo_ps(v_r6, v_r7);
    const __m256 t7  = _mm256_unpackhi_ps(v_r6, v_r7);
    const __m256 t8  = _mm256_unpacklo_ps(v_r8, v_r9);
    const __m256 t9  = _mm256_unpackhi_ps(v_r8, v_r9);
    const __m256 ta  = _mm256_unpacklo_ps(v_ra, v_rb);
    const __m256 tb  = _mm256_unpackhi_ps(v_ra, v_rb);
    const __m256 tc  = _mm256_unpacklo_ps(v_rc, v_rd);
    const __m256 td  = _mm256_unpackhi_ps(v_rc, v_rd);
    const __m256 te  = _mm256_unpacklo_ps(v_re, v_rf);
    const __m256 tf  = _mm256_unpackhi_ps(v_re, v_rf);

    const __m256 u0  = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u1  = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u2  = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u3  = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u4  = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u5  = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u6  = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u7  = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u8  = _mm256_shuffle_ps(t8, ta, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u9  = _mm256_shuffle_ps(t8, ta, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u10 = _mm256_shuffle_ps(t9, tb, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u11 = _mm256_shuffle_ps(t9, tb, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u12 = _mm256_shuffle_ps(tc, te, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u13 = _mm256_shuffle_ps(tc, te, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u14 = _mm256_shuffle_ps(td, tf, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u15 = _mm256_shuffle_ps(td, tf, _MM_SHUFFLE(3, 2, 3, 2));

    v_r0 = _mm256_permute2f128_ps(u0, u4, _MM_SHUFFLE(0, 2, 0, 0));
    v_r1 = _mm256_permute2f128_ps(u8, u12, _MM_SHUFFLE(0, 2, 0, 0));
    v_r2 = _mm256_permute2f128_ps(u1, u5, _MM_SHUFFLE(0, 2, 0, 0));
    v_r3 = _mm256_permute2f128_ps(u9, u13, _MM_SHUFFLE(0, 2, 0, 0));
    v_r4 = _mm256_permute2f128_ps(u2, u6, _MM_SHUFFLE(0, 2, 0, 0));
    v_r5 = _mm256_permute2f128_ps(u10, u14, _MM_SHUFFLE(0, 2, 0, 0));
    v_r6 = _mm256_permute2f128_ps(u3, u7, _MM_SHUFFLE(0, 2, 0, 0));
    v_r7 = _mm256_permute2f128_ps(u11, u15, _MM_SHUFFLE(0, 2, 0, 0));
    v_r8 = _mm256_permute2f128_ps(u0, u4, _MM_SHUFFLE(0, 3, 0, 1));
    v_r9 = _mm256_permute2f128_ps(u8, u12, _MM_SHUFFLE(0, 3, 0, 1));
    v_ra = _mm256_permute2f128_ps(u1, u5, _MM_SHUFFLE(0, 3, 0, 1));
    v_rb = _mm256_permute2f128_ps(u9, u13, _MM_SHUFFLE(0, 3, 0, 1));
    v_rc = _mm256_permute2f128_ps(u2, u6, _MM_SHUFFLE(0, 3, 0, 1));
    v_rd = _mm256_permute2f128_ps(u10, u14, _MM_SHUFFLE(0, 3, 0, 1));
    v_re = _mm256_permute2f128_ps(u3, u7, _MM_SHUFFLE(0, 3, 0, 1));
    v_rf = _mm256_permute2f128_ps(u11, u15, _MM_SHUFFLE(0, 3, 0, 1));
}

// =========================================================================
//  i16 (fp16 storage)  —  __m128i (8-element)  transposes
//  Used by the floating-point f16 pack pipelines.
// =========================================================================

inline void transpose_4x8_i16(__m128i& v_r0, __m128i& v_r1, __m128i& v_r2, __m128i& v_r3) noexcept {
    const __m128i tmp0 = _mm_unpacklo_epi16(v_r0, v_r1);
    const __m128i tmp1 = _mm_unpackhi_epi16(v_r0, v_r1);
    const __m128i tmp2 = _mm_unpacklo_epi16(v_r2, v_r3);
    const __m128i tmp3 = _mm_unpackhi_epi16(v_r2, v_r3);

    v_r0 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(tmp0), _mm_castsi128_ps(tmp2)));
    v_r1 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp2), _mm_castsi128_ps(tmp0)));
    v_r2 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(tmp1), _mm_castsi128_ps(tmp3)));
    v_r3 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(tmp3), _mm_castsi128_ps(tmp1)));
}

inline void transpose_6x8_i16(__m128i& v_a, __m128i& v_b, __m128i& v_c,
                               __m128i& v_d, __m128i& v_e, __m128i& v_f) noexcept {
    const __m128i t0 = _mm_unpacklo_epi16(v_a, v_b);
    const __m128i t1 = _mm_unpackhi_epi16(v_a, v_b);
    const __m128i t2 = _mm_unpacklo_epi16(v_c, v_d);
    const __m128i t3 = _mm_unpackhi_epi16(v_c, v_d);
    const __m128i t4 = _mm_unpacklo_epi16(v_e, v_f);
    const __m128i t5 = _mm_unpackhi_epi16(v_e, v_f);

    v_a = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t0), _mm_castsi128_ps(t1), _MM_SHUFFLE(1, 0, 1, 0)));
    v_b = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t2), _mm_castsi128_ps(t3), _MM_SHUFFLE(1, 0, 1, 0)));
    v_c = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t4), _mm_castsi128_ps(t5), _MM_SHUFFLE(1, 0, 1, 0)));
    v_d = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t0), _mm_castsi128_ps(t1), _MM_SHUFFLE(3, 2, 3, 2)));
    v_e = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t2), _mm_castsi128_ps(t3), _MM_SHUFFLE(3, 2, 3, 2)));
    v_f = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t4), _mm_castsi128_ps(t5), _MM_SHUFFLE(3, 2, 3, 2)));
}

inline void transpose_8x8_i16(__m128i& v_a, __m128i& v_b, __m128i& v_c, __m128i& v_d,
                               __m128i& v_e, __m128i& v_f, __m128i& v_g, __m128i& v_h) noexcept {
    const __m128i t0 = _mm_unpacklo_epi16(v_a, v_b);
    const __m128i t1 = _mm_unpackhi_epi16(v_a, v_b);
    const __m128i t2 = _mm_unpacklo_epi16(v_c, v_d);
    const __m128i t3 = _mm_unpackhi_epi16(v_c, v_d);
    const __m128i t4 = _mm_unpacklo_epi16(v_e, v_f);
    const __m128i t5 = _mm_unpackhi_epi16(v_e, v_f);
    const __m128i t6 = _mm_unpacklo_epi16(v_g, v_h);
    const __m128i t7 = _mm_unpackhi_epi16(v_g, v_h);

    v_a = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t0), _mm_castsi128_ps(t2), _MM_SHUFFLE(1, 0, 1, 0)));
    v_b = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t4), _mm_castsi128_ps(t6), _MM_SHUFFLE(1, 0, 1, 0)));
    v_c = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t0), _mm_castsi128_ps(t2), _MM_SHUFFLE(3, 2, 3, 2)));
    v_d = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t4), _mm_castsi128_ps(t6), _MM_SHUFFLE(3, 2, 3, 2)));
    v_e = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t1), _mm_castsi128_ps(t3), _MM_SHUFFLE(1, 0, 1, 0)));
    v_f = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t5), _mm_castsi128_ps(t7), _MM_SHUFFLE(1, 0, 1, 0)));
    v_g = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t1), _mm_castsi128_ps(t3), _MM_SHUFFLE(3, 2, 3, 2)));
    v_h = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t5), _mm_castsi128_ps(t7), _MM_SHUFFLE(3, 2, 3, 2)));
}

inline void transpose_12x8_i16(
    __m128i& v_r0, __m128i& v_r1, __m128i& v_r2, __m128i& v_r3,
    __m128i& v_r4, __m128i& v_r5, __m128i& v_r6, __m128i& v_r7,
    __m128i& v_r8, __m128i& v_r9, __m128i& v_ra, __m128i& v_rb) noexcept {

    const __m128i t0 = _mm_unpacklo_epi16(v_r0, v_r1);
    const __m128i t1 = _mm_unpackhi_epi16(v_r0, v_r1);
    const __m128i t2 = _mm_unpacklo_epi16(v_r2, v_r3);
    const __m128i t3 = _mm_unpackhi_epi16(v_r2, v_r3);
    const __m128i t4 = _mm_unpacklo_epi16(v_r4, v_r5);
    const __m128i t5 = _mm_unpackhi_epi16(v_r4, v_r5);
    const __m128i t6 = _mm_unpacklo_epi16(v_r6, v_r7);
    const __m128i t7 = _mm_unpackhi_epi16(v_r6, v_r7);
    const __m128i t8 = _mm_unpacklo_epi16(v_r8, v_r9);
    const __m128i t9 = _mm_unpackhi_epi16(v_r8, v_r9);
    const __m128i ta = _mm_unpacklo_epi16(v_ra, v_rb);
    const __m128i tb = _mm_unpackhi_epi16(v_ra, v_rb);

    v_r0 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t0), _mm_castsi128_ps(t1), _MM_SHUFFLE(1, 0, 1, 0)));
    v_r1 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t2), _mm_castsi128_ps(t3), _MM_SHUFFLE(1, 0, 1, 0)));
    v_r2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t4), _mm_castsi128_ps(t5), _MM_SHUFFLE(1, 0, 1, 0)));
    v_r3 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t0), _mm_castsi128_ps(t1), _MM_SHUFFLE(3, 2, 3, 2)));
    v_r4 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t2), _mm_castsi128_ps(t3), _MM_SHUFFLE(3, 2, 3, 2)));
    v_r5 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t4), _mm_castsi128_ps(t5), _MM_SHUFFLE(3, 2, 3, 2)));
    v_r6 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t6), _mm_castsi128_ps(t7), _MM_SHUFFLE(1, 0, 1, 0)));
    v_r7 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t8), _mm_castsi128_ps(t9), _MM_SHUFFLE(1, 0, 1, 0)));
    v_r8 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(ta), _mm_castsi128_ps(tb), _MM_SHUFFLE(1, 0, 1, 0)));
    v_r9 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t6), _mm_castsi128_ps(t7), _MM_SHUFFLE(3, 2, 3, 2)));
    v_ra = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t8), _mm_castsi128_ps(t9), _MM_SHUFFLE(3, 2, 3, 2)));
    v_rb = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(ta), _mm_castsi128_ps(tb), _MM_SHUFFLE(3, 2, 3, 2)));
}

inline void transpose_16x8_i16(
    __m128i& v_r0, __m128i& v_r1, __m128i& v_r2, __m128i& v_r3,
    __m128i& v_r4, __m128i& v_r5, __m128i& v_r6, __m128i& v_r7,
    __m128i& v_r8, __m128i& v_r9, __m128i& v_ra, __m128i& v_rb,
    __m128i& v_rc, __m128i& v_rd, __m128i& v_re, __m128i& v_rf) noexcept {

    const __m128i t0  = _mm_unpacklo_epi16(v_r0, v_r1);
    const __m128i t1  = _mm_unpackhi_epi16(v_r0, v_r1);
    const __m128i t2  = _mm_unpacklo_epi16(v_r2, v_r3);
    const __m128i t3  = _mm_unpackhi_epi16(v_r2, v_r3);
    const __m128i t4  = _mm_unpacklo_epi16(v_r4, v_r5);
    const __m128i t5  = _mm_unpackhi_epi16(v_r4, v_r5);
    const __m128i t6  = _mm_unpacklo_epi16(v_r6, v_r7);
    const __m128i t7  = _mm_unpackhi_epi16(v_r6, v_r7);
    const __m128i t8  = _mm_unpacklo_epi16(v_r8, v_r9);
    const __m128i t9  = _mm_unpackhi_epi16(v_r8, v_r9);
    const __m128i ta  = _mm_unpacklo_epi16(v_ra, v_rb);
    const __m128i tb  = _mm_unpackhi_epi16(v_ra, v_rb);
    const __m128i tc  = _mm_unpacklo_epi16(v_rc, v_rd);
    const __m128i td  = _mm_unpackhi_epi16(v_rc, v_rd);
    const __m128i te  = _mm_unpacklo_epi16(v_re, v_rf);
    const __m128i tf  = _mm_unpackhi_epi16(v_re, v_rf);

    v_r0 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t0), _mm_castsi128_ps(t1), _MM_SHUFFLE(1, 0, 1, 0)));
    v_r1 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t2), _mm_castsi128_ps(t3), _MM_SHUFFLE(1, 0, 1, 0)));
    v_r2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t4), _mm_castsi128_ps(t5), _MM_SHUFFLE(1, 0, 1, 0)));
    v_r3 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t6), _mm_castsi128_ps(t7), _MM_SHUFFLE(1, 0, 1, 0)));
    v_r4 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t0), _mm_castsi128_ps(t1), _MM_SHUFFLE(3, 2, 3, 2)));
    v_r5 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t2), _mm_castsi128_ps(t3), _MM_SHUFFLE(3, 2, 3, 2)));
    v_r6 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t4), _mm_castsi128_ps(t5), _MM_SHUFFLE(3, 2, 3, 2)));
    v_r7 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t6), _mm_castsi128_ps(t7), _MM_SHUFFLE(3, 2, 3, 2)));
    v_r8 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t8), _mm_castsi128_ps(t9), _MM_SHUFFLE(1, 0, 1, 0)));
    v_r9 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(ta), _mm_castsi128_ps(tb), _MM_SHUFFLE(1, 0, 1, 0)));
    v_ra = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tc), _mm_castsi128_ps(td), _MM_SHUFFLE(1, 0, 1, 0)));
    v_rb = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(te), _mm_castsi128_ps(tf), _MM_SHUFFLE(1, 0, 1, 0)));
    v_rc = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t8), _mm_castsi128_ps(t9), _MM_SHUFFLE(3, 2, 3, 2)));
    v_rd = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(ta), _mm_castsi128_ps(tb), _MM_SHUFFLE(3, 2, 3, 2)));
    v_re = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(tc), _mm_castsi128_ps(td), _MM_SHUFFLE(3, 2, 3, 2)));
    v_rf = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(te), _mm_castsi128_ps(tf), _MM_SHUFFLE(3, 2, 3, 2)));
}

}  // namespace nnops::backend::cpu::x86_64
