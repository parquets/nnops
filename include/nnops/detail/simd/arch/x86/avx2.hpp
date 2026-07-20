#pragma once
/// @file avx2.hpp
/// @brief AVX2 + FMA backend for 256-bit float32 vector operations (v_fp32x8).
///
/// Requires compile-time AVX2 support (built with /arch:AVX2 or -mavx2 -mfma).
/// The runtime dispatch (via CpuFeatures) ensures this code only executes on
/// AVX2-capable hardware, but the code itself must be compiled with AVX2 flags.
/// For runtime multi-ISA, compile this as a separate translation unit with custom flags.
///
/// v_fp32x4 is also provided on this platform (using the low 128 bits).

#if !defined(NNOPS_ARCH_X86_64)
  #error "avx2.hpp requires x86_64 architecture"
#endif

#include <immintrin.h>
#include <cstdint>

namespace nnops {
namespace simd {
namespace arch {
namespace avx2 {

// ============================================================
// v_fp32x4 — 128-bit, via __m128 (low 128 bits of 256-bit)
// ============================================================
struct v_fp32x4 {
    __m128 val;

    v_fp32x4() = default;
    explicit v_fp32x4(__m128 v) : val(v) {}
    explicit v_fp32x4(float s) : val(_mm_set1_ps(s)) {}
    v_fp32x4(float v0, float v1, float v2, float v3)
        : val(_mm_setr_ps(v0, v1, v2, v3)) {}

    float operator[](int i) const {
        float tmp[4];
        _mm_storeu_ps(tmp, val);
        return tmp[i];
    }
};

// ============================================================
// v_fp32x8 — 256-bit float vector, backed by __m256
// ============================================================
struct v_fp32x8 {
    __m256 val;

    v_fp32x8() = default;
    explicit v_fp32x8(__m256 v) : val(v) {}
    explicit v_fp32x8(float s) : val(_mm256_set1_ps(s)) {}
    v_fp32x8(float v0, float v1, float v2, float v3,
             float v4, float v5, float v6, float v7)
        : val(_mm256_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7)) {}
};

// ============================================================
// v_fp32x4 operations (same API, 128-bit under the hood)
// ============================================================
inline v_fp32x4 load_fp32x4(const float* p) {
    return v_fp32x4(_mm_loadu_ps(p));
}
inline void store(float* p, const v_fp32x4& a) {
    _mm_storeu_ps(p, a.val);
}
inline v_fp32x4 set1_fp32x4(float s)  { return v_fp32x4(_mm_set1_ps(s)); }
inline v_fp32x4 zero_fp32x4()         { return v_fp32x4(_mm_setzero_ps()); }

inline v_fp32x4 add(const v_fp32x4& a, const v_fp32x4& b)
    { return v_fp32x4(_mm_add_ps(a.val, b.val)); }
inline v_fp32x4 sub(const v_fp32x4& a, const v_fp32x4& b)
    { return v_fp32x4(_mm_sub_ps(a.val, b.val)); }
inline v_fp32x4 mul(const v_fp32x4& a, const v_fp32x4& b)
    { return v_fp32x4(_mm_mul_ps(a.val, b.val)); }
inline v_fp32x4 div(const v_fp32x4& a, const v_fp32x4& b)
    { return v_fp32x4(_mm_div_ps(a.val, b.val)); }
inline v_fp32x4 fmadd(const v_fp32x4& a, const v_fp32x4& b, const v_fp32x4& c)
    { return v_fp32x4(_mm_fmadd_ps(a.val, b.val, c.val)); }

inline v_fp32x4 min(const v_fp32x4& a, const v_fp32x4& b)
    { return v_fp32x4(_mm_min_ps(a.val, b.val)); }
inline v_fp32x4 max(const v_fp32x4& a, const v_fp32x4& b)
    { return v_fp32x4(_mm_max_ps(a.val, b.val)); }
inline v_fp32x4 abs(const v_fp32x4& a)
    { return v_fp32x4(_mm_andnot_ps(_mm_set1_ps(-0.0f), a.val)); }
inline v_fp32x4 neg(const v_fp32x4& a)
    { return v_fp32x4(_mm_xor_ps(a.val, _mm_set1_ps(-0.0f))); }

inline v_fp32x4 cmplt(const v_fp32x4& a, const v_fp32x4& b)
    { return v_fp32x4(_mm_cmplt_ps(a.val, b.val)); }
inline v_fp32x4 cmpgt(const v_fp32x4& a, const v_fp32x4& b)
    { return v_fp32x4(_mm_cmpgt_ps(a.val, b.val)); }

inline v_fp32x4 and_(const v_fp32x4& a, const v_fp32x4& b)
    { return v_fp32x4(_mm_and_ps(a.val, b.val)); }
inline v_fp32x4 sqrt(const v_fp32x4& a)
    { return v_fp32x4(_mm_sqrt_ps(a.val)); }

inline float reduce_sum(const v_fp32x4& a) {
    __m128 t = _mm_add_ps(a.val, _mm_movehl_ps(a.val, a.val));
    t = _mm_add_ps(t, _mm_shuffle_ps(t, t, 1));
    return _mm_cvtss_f32(t);
}

// ============================================================
// v_fp32x8 operations (native 256-bit AVX2, FMA3)
// ============================================================
inline v_fp32x8 load_fp32x8(const float* p) {
    return v_fp32x8(_mm256_loadu_ps(p));
}
inline void store(float* p, const v_fp32x8& a) {
    _mm256_storeu_ps(p, a.val);
}
inline v_fp32x8 set1_fp32x8(float s)  { return v_fp32x8(_mm256_set1_ps(s)); }
inline v_fp32x8 zero_fp32x8()         { return v_fp32x8(_mm256_setzero_ps()); }

inline v_fp32x8 add(const v_fp32x8& a, const v_fp32x8& b)
    { return v_fp32x8(_mm256_add_ps(a.val, b.val)); }
inline v_fp32x8 sub(const v_fp32x8& a, const v_fp32x8& b)
    { return v_fp32x8(_mm256_sub_ps(a.val, b.val)); }
inline v_fp32x8 mul(const v_fp32x8& a, const v_fp32x8& b)
    { return v_fp32x8(_mm256_mul_ps(a.val, b.val)); }
inline v_fp32x8 div(const v_fp32x8& a, const v_fp32x8& b)
    { return v_fp32x8(_mm256_div_ps(a.val, b.val)); }

// FMA: true fused multiply-add via FMA3
inline v_fp32x8 fmadd(const v_fp32x8& a, const v_fp32x8& b, const v_fp32x8& c)
    { return v_fp32x8(_mm256_fmadd_ps(a.val, b.val, c.val)); }
inline v_fp32x8 fmsub(const v_fp32x8& a, const v_fp32x8& b, const v_fp32x8& c)
    { return v_fp32x8(_mm256_fmsub_ps(a.val, b.val, c.val)); }

inline v_fp32x8 min(const v_fp32x8& a, const v_fp32x8& b)
    { return v_fp32x8(_mm256_min_ps(a.val, b.val)); }
inline v_fp32x8 max(const v_fp32x8& a, const v_fp32x8& b)
    { return v_fp32x8(_mm256_max_ps(a.val, b.val)); }
inline v_fp32x8 abs(const v_fp32x8& a)
    { return v_fp32x8(_mm256_andnot_ps(_mm256_set1_ps(-0.0f), a.val)); }
inline v_fp32x8 neg(const v_fp32x8& a)
    { return v_fp32x8(_mm256_xor_ps(a.val, _mm256_set1_ps(-0.0f))); }

inline v_fp32x8 cmplt(const v_fp32x8& a, const v_fp32x8& b)
    { return v_fp32x8(_mm256_cmp_ps(a.val, b.val, _CMP_LT_OS)); }
inline v_fp32x8 cmple(const v_fp32x8& a, const v_fp32x8& b)
    { return v_fp32x8(_mm256_cmp_ps(a.val, b.val, _CMP_LE_OS)); }
inline v_fp32x8 cmpgt(const v_fp32x8& a, const v_fp32x8& b)
    { return v_fp32x8(_mm256_cmp_ps(a.val, b.val, _CMP_GT_OS)); }
inline v_fp32x8 ceq(const v_fp32x8& a, const v_fp32x8& b)
    { return v_fp32x8(_mm256_cmp_ps(a.val, b.val, _CMP_EQ_OS)); }

inline v_fp32x8 and_(const v_fp32x8& a, const v_fp32x8& b)
    { return v_fp32x8(_mm256_and_ps(a.val, b.val)); }
inline v_fp32x8 or_(const v_fp32x8& a, const v_fp32x8& b)
    { return v_fp32x8(_mm256_or_ps(a.val, b.val)); }

inline v_fp32x8 sqrt(const v_fp32x8& a)
    { return v_fp32x8(_mm256_sqrt_ps(a.val)); }
inline v_fp32x8 rcp(const v_fp32x8& a)
    { return v_fp32x8(_mm256_rcp_ps(a.val)); }
inline v_fp32x8 rsqrt(const v_fp32x8& a)
    { return v_fp32x8(_mm256_rsqrt_ps(a.val)); }

// Horizontal sum via hadd + permute
inline float reduce_sum(const v_fp32x8& a) {
    __m128 lo = _mm256_castps256_ps128(a.val);
    __m128 hi = _mm256_extractf128_ps(a.val, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    return reduce_sum(v_fp32x4(sum128));
}

} // namespace avx2
} // namespace arch
} // namespace simd
} // namespace nnops
