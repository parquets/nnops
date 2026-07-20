#pragma once
/// @file avx2.hpp
/// @brief AVX2 + FMA backend for 256-bit float32 vector operations (VecF32x8).
///
/// Requires compile-time AVX2 support (built with /arch:AVX2 or -mavx2 -mfma).
/// The runtime dispatch (via CpuFeatures) ensures this code only executes on
/// AVX2-capable hardware, but the code itself must be compiled with AVX2 flags.
/// For runtime multi-ISA, compile this as a separate translation unit with custom flags.
///
/// VecF32x4 is also provided on this platform (using the low 128 bits).

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
// VecF32x4 — 128-bit, via __m128 (low 128 bits of 256-bit)
// ============================================================
struct VecF32x4 {
    __m128 val;

    VecF32x4() = default;
    explicit VecF32x4(__m128 v) : val(v) {}
    explicit VecF32x4(float s) : val(_mm_set1_ps(s)) {}
    VecF32x4(float v0, float v1, float v2, float v3)
        : val(_mm_setr_ps(v0, v1, v2, v3)) {}

    float operator[](int i) const {
        float tmp[4];
        _mm_storeu_ps(tmp, val);
        return tmp[i];
    }
};

// ============================================================
// VecF32x8 — 256-bit float vector, backed by __m256
// ============================================================
struct VecF32x8 {
    __m256 val;

    VecF32x8() = default;
    explicit VecF32x8(__m256 v) : val(v) {}
    explicit VecF32x8(float s) : val(_mm256_set1_ps(s)) {}
    VecF32x8(float v0, float v1, float v2, float v3,
             float v4, float v5, float v6, float v7)
        : val(_mm256_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7)) {}
};

// ============================================================
// VecF32x4 operations (same API, 128-bit under the hood)
// ============================================================
inline VecF32x4 vec_load_f32x4(const float* p) {
    return VecF32x4(_mm_loadu_ps(p));
}
inline void vec_store_f32x4(float* p, const VecF32x4& a) {
    _mm_storeu_ps(p, a.val);
}
inline VecF32x4 vec_set1_f32x4(float s)  { return VecF32x4(_mm_set1_ps(s)); }
inline VecF32x4 vec_zero_f32x4()         { return VecF32x4(_mm_setzero_ps()); }

inline VecF32x4 vec_add_f32x4(const VecF32x4& a, const VecF32x4& b)
    { return VecF32x4(_mm_add_ps(a.val, b.val)); }
inline VecF32x4 vec_sub_f32x4(const VecF32x4& a, const VecF32x4& b)
    { return VecF32x4(_mm_sub_ps(a.val, b.val)); }
inline VecF32x4 vec_mul_f32x4(const VecF32x4& a, const VecF32x4& b)
    { return VecF32x4(_mm_mul_ps(a.val, b.val)); }
inline VecF32x4 vec_div_f32x4(const VecF32x4& a, const VecF32x4& b)
    { return VecF32x4(_mm_div_ps(a.val, b.val)); }
inline VecF32x4 vec_fmadd_f32x4(const VecF32x4& a, const VecF32x4& b, const VecF32x4& c)
    { return VecF32x4(_mm_fmadd_ps(a.val, b.val, c.val)); }

inline VecF32x4 vec_min_f32x4(const VecF32x4& a, const VecF32x4& b)
    { return VecF32x4(_mm_min_ps(a.val, b.val)); }
inline VecF32x4 vec_max_f32x4(const VecF32x4& a, const VecF32x4& b)
    { return VecF32x4(_mm_max_ps(a.val, b.val)); }
inline VecF32x4 vec_abs_f32x4(const VecF32x4& a)
    { return VecF32x4(_mm_andnot_ps(_mm_set1_ps(-0.0f), a.val)); }
inline VecF32x4 vec_neg_f32x4(const VecF32x4& a)
    { return VecF32x4(_mm_xor_ps(a.val, _mm_set1_ps(-0.0f))); }

inline VecF32x4 vec_cmplt_f32x4(const VecF32x4& a, const VecF32x4& b)
    { return VecF32x4(_mm_cmplt_ps(a.val, b.val)); }
inline VecF32x4 vec_cmpgt_f32x4(const VecF32x4& a, const VecF32x4& b)
    { return VecF32x4(_mm_cmpgt_ps(a.val, b.val)); }

inline VecF32x4 vec_and_f32x4(const VecF32x4& a, const VecF32x4& b)
    { return VecF32x4(_mm_and_ps(a.val, b.val)); }
inline VecF32x4 vec_sqrt_f32x4(const VecF32x4& a)
    { return VecF32x4(_mm_sqrt_ps(a.val)); }

inline float vec_reduce_sum_f32x4(const VecF32x4& a) {
    __m128 t = _mm_add_ps(a.val, _mm_movehl_ps(a.val, a.val));
    t = _mm_add_ps(t, _mm_shuffle_ps(t, t, 1));
    return _mm_cvtss_f32(t);
}

// ============================================================
// VecF32x8 operations (native 256-bit AVX2, FMA3)
// ============================================================
inline VecF32x8 vec_load_f32x8(const float* p) {
    return VecF32x8(_mm256_loadu_ps(p));
}
inline void vec_store_f32x8(float* p, const VecF32x8& a) {
    _mm256_storeu_ps(p, a.val);
}
inline VecF32x8 vec_set1_f32x8(float s)  { return VecF32x8(_mm256_set1_ps(s)); }
inline VecF32x8 vec_zero_f32x8()         { return VecF32x8(_mm256_setzero_ps()); }

inline VecF32x8 vec_add_f32x8(const VecF32x8& a, const VecF32x8& b)
    { return VecF32x8(_mm256_add_ps(a.val, b.val)); }
inline VecF32x8 vec_sub_f32x8(const VecF32x8& a, const VecF32x8& b)
    { return VecF32x8(_mm256_sub_ps(a.val, b.val)); }
inline VecF32x8 vec_mul_f32x8(const VecF32x8& a, const VecF32x8& b)
    { return VecF32x8(_mm256_mul_ps(a.val, b.val)); }
inline VecF32x8 vec_div_f32x8(const VecF32x8& a, const VecF32x8& b)
    { return VecF32x8(_mm256_div_ps(a.val, b.val)); }

// FMA: true fused multiply-add via FMA3
inline VecF32x8 vec_fmadd_f32x8(const VecF32x8& a, const VecF32x8& b, const VecF32x8& c)
    { return VecF32x8(_mm256_fmadd_ps(a.val, b.val, c.val)); }
inline VecF32x8 vec_fmsub_f32x8(const VecF32x8& a, const VecF32x8& b, const VecF32x8& c)
    { return VecF32x8(_mm256_fmsub_ps(a.val, b.val, c.val)); }

inline VecF32x8 vec_min_f32x8(const VecF32x8& a, const VecF32x8& b)
    { return VecF32x8(_mm256_min_ps(a.val, b.val)); }
inline VecF32x8 vec_max_f32x8(const VecF32x8& a, const VecF32x8& b)
    { return VecF32x8(_mm256_max_ps(a.val, b.val)); }
inline VecF32x8 vec_abs_f32x8(const VecF32x8& a)
    { return VecF32x8(_mm256_andnot_ps(_mm256_set1_ps(-0.0f), a.val)); }
inline VecF32x8 vec_neg_f32x8(const VecF32x8& a)
    { return VecF32x8(_mm256_xor_ps(a.val, _mm256_set1_ps(-0.0f))); }

inline VecF32x8 vec_cmplt_f32x8(const VecF32x8& a, const VecF32x8& b)
    { return VecF32x8(_mm256_cmp_ps(a.val, b.val, _CMP_LT_OS)); }
inline VecF32x8 vec_cmple_f32x8(const VecF32x8& a, const VecF32x8& b)
    { return VecF32x8(_mm256_cmp_ps(a.val, b.val, _CMP_LE_OS)); }
inline VecF32x8 vec_cmpgt_f32x8(const VecF32x8& a, const VecF32x8& b)
    { return VecF32x8(_mm256_cmp_ps(a.val, b.val, _CMP_GT_OS)); }
inline VecF32x8 vec_ceq_f32x8(const VecF32x8& a, const VecF32x8& b)
    { return VecF32x8(_mm256_cmp_ps(a.val, b.val, _CMP_EQ_OS)); }

inline VecF32x8 vec_and_f32x8(const VecF32x8& a, const VecF32x8& b)
    { return VecF32x8(_mm256_and_ps(a.val, b.val)); }
inline VecF32x8 vec_or_f32x8(const VecF32x8& a, const VecF32x8& b)
    { return VecF32x8(_mm256_or_ps(a.val, b.val)); }

inline VecF32x8 vec_sqrt_f32x8(const VecF32x8& a)
    { return VecF32x8(_mm256_sqrt_ps(a.val)); }
inline VecF32x8 vec_rcp_f32x8(const VecF32x8& a)
    { return VecF32x8(_mm256_rcp_ps(a.val)); }
inline VecF32x8 vec_rsqrt_f32x8(const VecF32x8& a)
    { return VecF32x8(_mm256_rsqrt_ps(a.val)); }

// Horizontal sum via hadd + permute
inline float vec_reduce_sum_f32x8(const VecF32x8& a) {
    __m128 lo = _mm256_castps256_ps128(a.val);
    __m128 hi = _mm256_extractf128_ps(a.val, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    return vec_reduce_sum_f32x4(VecF32x4(sum128));
}

} // namespace avx2
} // namespace arch
} // namespace simd
} // namespace nnops
