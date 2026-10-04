#pragma once
/// @file avx2.hpp
/// @brief AVX2 + FMA backend for 256-bit float32 vector operations (v_f32x8).
///
/// Requires compile-time AVX2 support (built with /arch:AVX2 or -mavx2 -mfma).
/// The runtime dispatch (via CpuFeatures) ensures this code only executes on
/// AVX2-capable hardware, but the code itself must be compiled with AVX2 flags.
/// For runtime multi-ISA, compile this as a separate translation unit with custom flags.
///
/// v_f32x4 is also provided on this platform (using the low 128 bits).

#if !defined(NNOPS_ARCH_X86_64)
  #error "avx2.hpp requires x86_64 architecture"
#endif

#include <immintrin.h>
#include <cstdint>
#include "sse_mathfunc.hpp"
#include "avx2_mathfunc.hpp"

// v_cvt_f32_to_f16 returns arch::sse::v_f16x8 (the canonical 128-bit fp16 type
// on x86_64), so pull in sse.hpp to make that type visible on the F16C path.
#if defined(__F16C__)
#include "sse.hpp"
#endif

namespace nnops {
namespace simd {
namespace arch {
namespace avx2 {

// ============================================================
// v_f32x4: 128-bit float vector, backed by __m128
// ============================================================
struct v_f32x4 {
    __m128 val;

    v_f32x4() = default;
    explicit v_f32x4(__m128 v) : val(v) {}
    explicit v_f32x4(float s) : val(_mm_set1_ps(s)) {}
    v_f32x4(float v0, float v1, float v2, float v3)
        : val(_mm_setr_ps(v0, v1, v2, v3)) {}

    float operator[](int i) const {
        float tmp[4];
        _mm_storeu_ps(tmp, val);
        return tmp[i];
    }
};

// ============================================================
// v_f32x8 — 256-bit float vector, backed by __m256
// ============================================================
struct v_f32x8 {
    __m256 val;

    v_f32x8() = default;
    explicit v_f32x8(__m256 v) : val(v) {}
    explicit v_f32x8(float s) : val(_mm256_set1_ps(s)) {}
    v_f32x8(float v0, float v1, float v2, float v3,
             float v4, float v5, float v6, float v7)
        : val(_mm256_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7)) {}
};

// ============================================================
// v_f32x4 operations (same API, 128-bit under the hood)
// ============================================================
inline v_f32x4 v_load_f32x4(const float* p) {
    return v_f32x4(_mm_loadu_ps(p));
}
inline void v_store(float* p, const v_f32x4& a) {
    _mm_storeu_ps(p, a.val);
}
inline v_f32x4 v_set1_f32x4(float s)  { return v_f32x4(_mm_set1_ps(s)); }
inline v_f32x4 v_zero_f32x4()         { return v_f32x4(_mm_setzero_ps()); }

inline v_f32x4 v_add(const v_f32x4& a, const v_f32x4& b)
    { return v_f32x4(_mm_add_ps(a.val, b.val)); }
inline v_f32x4 v_sub(const v_f32x4& a, const v_f32x4& b)
    { return v_f32x4(_mm_sub_ps(a.val, b.val)); }
inline v_f32x4 v_mul(const v_f32x4& a, const v_f32x4& b)
    { return v_f32x4(_mm_mul_ps(a.val, b.val)); }
inline v_f32x4 v_div(const v_f32x4& a, const v_f32x4& b)
    { return v_f32x4(_mm_div_ps(a.val, b.val)); }
inline v_f32x4 v_fmadd(const v_f32x4& a, const v_f32x4& b, const v_f32x4& c)
    { return v_f32x4(_mm_fmadd_ps(a.val, b.val, c.val)); }

inline v_f32x4 v_min(const v_f32x4& a, const v_f32x4& b)
    { return v_f32x4(_mm_min_ps(a.val, b.val)); }
inline v_f32x4 v_max(const v_f32x4& a, const v_f32x4& b)
    { return v_f32x4(_mm_max_ps(a.val, b.val)); }
inline v_f32x4 v_abs(const v_f32x4& a)
    { return v_f32x4(_mm_andnot_ps(_mm_set1_ps(-0.0f), a.val)); }
inline v_f32x4 v_neg(const v_f32x4& a)
    { return v_f32x4(_mm_xor_ps(a.val, _mm_set1_ps(-0.0f))); }

inline v_f32x4 v_cmplt(const v_f32x4& a, const v_f32x4& b)
    { return v_f32x4(_mm_cmplt_ps(a.val, b.val)); }
inline v_f32x4 v_cmpgt(const v_f32x4& a, const v_f32x4& b)
    { return v_f32x4(_mm_cmpgt_ps(a.val, b.val)); }

inline v_f32x4 v_and(const v_f32x4& a, const v_f32x4& b)
    { return v_f32x4(_mm_and_ps(a.val, b.val)); }
inline v_f32x4 v_sqrt(const v_f32x4& a)
    { return v_f32x4(_mm_sqrt_ps(a.val)); }

// Transcendental math for v_f32x4 (delegated to sse_mathfunc)
inline v_f32x4 v_exp(const v_f32x4& a) {
    return v_f32x4(sse::exp_ps(a.val));
}
inline v_f32x4 v_log(const v_f32x4& a) {
    return v_f32x4(sse::log_ps(a.val));
}
inline v_f32x4 v_sin(const v_f32x4& a) {
    return v_f32x4(sse::sin_ps(a.val));
}
inline v_f32x4 v_cos(const v_f32x4& a) {
    return v_f32x4(sse::cos_ps(a.val));
}
inline v_f32x4 v_tan(const v_f32x4& a) {
    return v_f32x4(sse::tan_ps(a.val));
}
inline v_f32x4 v_tanh(const v_f32x4& a) {
    return v_f32x4(sse::tanh_ps(a.val));
}

inline float v_reduce_sum(const v_f32x4& a) {
    __m128 t = _mm_add_ps(a.val, _mm_movehl_ps(a.val, a.val));
    t = _mm_add_ps(t, _mm_shuffle_ps(t, t, 1));
    return _mm_cvtss_f32(t);
}

// Horizontal max/min — same pairwise shuffle reduction as v_reduce_sum
inline float v_reduce_max(const v_f32x4& a) {
    __m128 t = _mm_max_ps(a.val, _mm_movehl_ps(a.val, a.val));
    t = _mm_max_ps(t, _mm_shuffle_ps(t, t, 1));
    return _mm_cvtss_f32(t);
}
inline float v_reduce_min(const v_f32x4& a) {
    __m128 t = _mm_min_ps(a.val, _mm_movehl_ps(a.val, a.val));
    t = _mm_min_ps(t, _mm_shuffle_ps(t, t, 1));
    return _mm_cvtss_f32(t);
}

// ============================================================
// v_f32x8 operations (native 256-bit AVX2, FMA3)
// ============================================================
inline v_f32x8 v_load_f32x8(const float* p) {
    return v_f32x8(_mm256_loadu_ps(p));
}
inline void v_store(float* p, const v_f32x8& a) {
    _mm256_storeu_ps(p, a.val);
}
inline v_f32x8 v_set1_f32x8(float s)  { return v_f32x8(_mm256_set1_ps(s)); }
inline v_f32x8 v_zero_f32x8()         { return v_f32x8(_mm256_setzero_ps()); }

inline v_f32x8 v_add(const v_f32x8& a, const v_f32x8& b)
    { return v_f32x8(_mm256_add_ps(a.val, b.val)); }
inline v_f32x8 v_sub(const v_f32x8& a, const v_f32x8& b)
    { return v_f32x8(_mm256_sub_ps(a.val, b.val)); }
inline v_f32x8 v_mul(const v_f32x8& a, const v_f32x8& b)
    { return v_f32x8(_mm256_mul_ps(a.val, b.val)); }
inline v_f32x8 v_div(const v_f32x8& a, const v_f32x8& b)
    { return v_f32x8(_mm256_div_ps(a.val, b.val)); }

// FMA: true fused multiply-add via FMA3
inline v_f32x8 v_fmadd(const v_f32x8& a, const v_f32x8& b, const v_f32x8& c)
    { return v_f32x8(_mm256_fmadd_ps(a.val, b.val, c.val)); }
inline v_f32x8 v_fmsub(const v_f32x8& a, const v_f32x8& b, const v_f32x8& c)
    { return v_f32x8(_mm256_fmsub_ps(a.val, b.val, c.val)); }

inline v_f32x8 v_min(const v_f32x8& a, const v_f32x8& b)
    { return v_f32x8(_mm256_min_ps(a.val, b.val)); }
inline v_f32x8 v_max(const v_f32x8& a, const v_f32x8& b)
    { return v_f32x8(_mm256_max_ps(a.val, b.val)); }
inline v_f32x8 v_abs(const v_f32x8& a)
    { return v_f32x8(_mm256_andnot_ps(_mm256_set1_ps(-0.0f), a.val)); }
inline v_f32x8 v_neg(const v_f32x8& a)
    { return v_f32x8(_mm256_xor_ps(a.val, _mm256_set1_ps(-0.0f))); }

inline v_f32x8 v_cmplt(const v_f32x8& a, const v_f32x8& b)
    { return v_f32x8(_mm256_cmp_ps(a.val, b.val, _CMP_LT_OS)); }
inline v_f32x8 v_cmple(const v_f32x8& a, const v_f32x8& b)
    { return v_f32x8(_mm256_cmp_ps(a.val, b.val, _CMP_LE_OS)); }
inline v_f32x8 v_cmpgt(const v_f32x8& a, const v_f32x8& b)
    { return v_f32x8(_mm256_cmp_ps(a.val, b.val, _CMP_GT_OS)); }
inline v_f32x8 v_ceq(const v_f32x8& a, const v_f32x8& b)
    { return v_f32x8(_mm256_cmp_ps(a.val, b.val, _CMP_EQ_OS)); }

inline v_f32x8 v_and(const v_f32x8& a, const v_f32x8& b)
    { return v_f32x8(_mm256_and_ps(a.val, b.val)); }
inline v_f32x8 v_or(const v_f32x8& a, const v_f32x8& b)
    { return v_f32x8(_mm256_or_ps(a.val, b.val)); }

inline v_f32x8 v_sqrt(const v_f32x8& a)
    { return v_f32x8(_mm256_sqrt_ps(a.val)); }
inline v_f32x8 v_rcp(const v_f32x8& a)
    { return v_f32x8(_mm256_rcp_ps(a.val)); }
inline v_f32x8 v_rsqrt(const v_f32x8& a)
    { return v_f32x8(_mm256_rsqrt_ps(a.val)); }

// Transcendental math functions for v_f32x8 (delegated to avx2_mathfunc)
inline v_f32x8 v_exp(const v_f32x8& a) {
    return v_f32x8(exp256_ps(a.val));
}
inline v_f32x8 v_log(const v_f32x8& a) {
    return v_f32x8(log256_ps(a.val));
}
inline v_f32x8 v_sin(const v_f32x8& a) {
    return v_f32x8(sin256_ps(a.val));
}
inline v_f32x8 v_cos(const v_f32x8& a) {
    return v_f32x8(cos256_ps(a.val));
}
inline v_f32x8 v_tan(const v_f32x8& a) {
    return v_f32x8(tan256_ps(a.val));
}
inline v_f32x8 v_tanh(const v_f32x8& a) {
    return v_f32x8(tanh256_ps(a.val));
}

// Horizontal sum: fold the two 128-bit halves, then pairwise-reduce
inline float v_reduce_sum(const v_f32x8& a) {
    __m128 lo = _mm256_castps256_ps128(a.val);
    __m128 hi = _mm256_extractf128_ps(a.val, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    return v_reduce_sum(v_f32x4(sum128));
}

// Horizontal max — pairwise shuffle reduction
inline float v_reduce_max(const v_f32x8& a) {
    __m128 lo = _mm256_castps256_ps128(a.val);
    __m128 hi = _mm256_extractf128_ps(a.val, 1);
    __m128 max128 = _mm_max_ps(lo, hi);
    return v_reduce_max(v_f32x4(max128));
}

// Horizontal min — pairwise shuffle reduction
inline float v_reduce_min(const v_f32x8& a) {
    __m128 lo = _mm256_castps256_ps128(a.val);
    __m128 hi = _mm256_extractf128_ps(a.val, 1);
    __m128 min128 = _mm_min_ps(lo, hi);
    return v_reduce_min(v_f32x4(min128));
}

// ============================================================
// f32 → f16 conversion (F16C)
// ============================================================
#if defined(__F16C__)

/// @brief Narrow 8 f32 lanes (__m256) to 8 f16 lanes (sse::v_f16x8).
inline sse::v_f16x8 v_cvt_f32_to_f16(const v_f32x8& a) {
    return sse::v_f16x8(_mm256_cvtps_ph(a.val, 0));
}

#endif  // defined(__F16C__)

// ============================================================
// Deinterleave (stride-2 gather) — pair types
// ============================================================

/// @brief Pair of v_f32x4 registers: even- and odd-indexed elements.
struct v_f32x4x2_t { v_f32x4 even; v_f32x4 odd; };

/// @brief Pair of v_f32x8 registers: even- and odd-indexed elements.
struct v_f32x8x2_t { v_f32x8 even; v_f32x8 odd; };

// ============================================================
// 方案 A: v_deinterleave_* — load 2×N elements, return {even, odd}
// ============================================================

/// @brief Load 8 contiguous f32s, return {a0,a2,a4,a6}, {a1,a3,a5,a7}.
inline v_f32x4x2_t v_deinterleave_f32x4(const float* src) {
    __m128 lo   = _mm_loadu_ps(src);       // a0, a1, a2, a3
    __m128 hi   = _mm_loadu_ps(src + 4);   // a4, a5, a6, a7
    __m128 t0   = _mm_unpacklo_ps(lo, hi); // a0, a4, a1, a5
    __m128 t1   = _mm_unpackhi_ps(lo, hi); // a2, a6, a3, a7
    return {v_f32x4(_mm_unpacklo_ps(t0, t1)),   // a0, a2, a4, a6
            v_f32x4(_mm_unpackhi_ps(t0, t1))};  // a1, a3, a5, a7
}

/// @brief Load 16 contiguous f32s, return {even}, {odd} — each v_f32x8.
inline v_f32x8x2_t v_deinterleave_f32x8(const float* src) {
    v_f32x4x2_t lo = v_deinterleave_f32x4(src);
    v_f32x4x2_t hi = v_deinterleave_f32x4(src + 8);
    __m256 even = _mm256_insertf128_ps(
        _mm256_castps128_ps256(lo.even.val), hi.even.val, 1);
    __m256 odd  = _mm256_insertf128_ps(
        _mm256_castps128_ps256(lo.odd.val),  hi.odd.val,  1);
    return {v_f32x8(even), v_f32x8(odd)};
}

// ============================================================
// 方案 B: v_load_even_* / v_load_odd_* — single-result convenience
// ============================================================

inline v_f32x4 v_load_even_f32x4(const float* src) {
    return v_deinterleave_f32x4(src).even;
}
inline v_f32x4 v_load_odd_f32x4(const float* src) {
    return v_deinterleave_f32x4(src).odd;
}
inline v_f32x8 v_load_even_f32x8(const float* src) {
    return v_deinterleave_f32x8(src).even;
}
inline v_f32x8 v_load_odd_f32x8(const float* src) {
    return v_deinterleave_f32x8(src).odd;
}

// ============================================================
// 方案 C: v_load_stride2_even_* / v_load_stride2_odd_* — explicit names
// ============================================================

inline v_f32x4 v_load_stride2_even_f32x4(const float* src) {
    return v_deinterleave_f32x4(src).even;
}
inline v_f32x4 v_load_stride2_odd_f32x4(const float* src) {
    return v_deinterleave_f32x4(src).odd;
}
inline v_f32x8 v_load_stride2_even_f32x8(const float* src) {
    return v_deinterleave_f32x8(src).even;
}
inline v_f32x8 v_load_stride2_odd_f32x8(const float* src) {
    return v_deinterleave_f32x8(src).odd;
}

// ============================================================
// 8×8 f32 transpose — 8 v_f32x8 rows → 8 v_f32x8 columns
// ============================================================

/// @brief Transpose an 8×8 matrix of f32 held in 8 v_f32x8 registers.
/// On entry, r0..r7 are rows 0..7 of the matrix. On exit, they are
/// the transposed rows (= original columns).
inline void v_transpose_8x8(v_f32x8& r0, v_f32x8& r1, v_f32x8& r2, v_f32x8& r3,
                             v_f32x8& r4, v_f32x8& r5, v_f32x8& r6, v_f32x8& r7) {
    __m256 t0, t1, t2, t3, t4, t5, t6, t7;
    __m256 s0, s1, s2, s3, s4, s5, s6, s7;

    // Phase 1: interleave 32-bit lanes across pairs
    t0 = _mm256_unpacklo_ps(r0.val, r1.val);
    t1 = _mm256_unpackhi_ps(r0.val, r1.val);
    t2 = _mm256_unpacklo_ps(r2.val, r3.val);
    t3 = _mm256_unpackhi_ps(r2.val, r3.val);
    t4 = _mm256_unpacklo_ps(r4.val, r5.val);
    t5 = _mm256_unpackhi_ps(r4.val, r5.val);
    t6 = _mm256_unpacklo_ps(r6.val, r7.val);
    t7 = _mm256_unpackhi_ps(r6.val, r7.val);

    // Phase 2: shuffle within 128-bit lanes
    s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1,0,1,0));  // col 0 | col 4
    s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3,2,3,2));  // col 1 | col 5
    s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1,0,1,0));
    s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3,2,3,2));
    s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1,0,1,0));
    s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3,2,3,2));
    s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1,0,1,0));
    s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3,2,3,2));

    // Phase 3: permute 128-bit lanes
    r0.val = _mm256_permute2f128_ps(s0, s4, 0x20);
    r1.val = _mm256_permute2f128_ps(s1, s5, 0x20);
    r2.val = _mm256_permute2f128_ps(s2, s6, 0x20);
    r3.val = _mm256_permute2f128_ps(s3, s7, 0x20);
    r4.val = _mm256_permute2f128_ps(s0, s4, 0x31);
    r5.val = _mm256_permute2f128_ps(s1, s5, 0x31);
    r6.val = _mm256_permute2f128_ps(s2, s6, 0x31);
    r7.val = _mm256_permute2f128_ps(s3, s7, 0x31);
}

} // namespace avx2
} // namespace arch
} // namespace simd
} // namespace nnops
