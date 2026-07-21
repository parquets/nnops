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

namespace nnops {
namespace simd {
namespace arch {
namespace avx2 {

// ============================================================
// v_f32x4 — 128-bit, via __m128 (low 128 bits of 256-bit)
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

// FMA: true fused multiply-v_add via FMA3
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

// Horizontal sum via hadd + permute
inline float v_reduce_sum(const v_f32x8& a) {
    __m128 lo = _mm256_castps256_ps128(a.val);
    __m128 hi = _mm256_extractf128_ps(a.val, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    return v_reduce_sum(v_f32x4(sum128));
}

} // namespace avx2
} // namespace arch
} // namespace simd
} // namespace nnops
