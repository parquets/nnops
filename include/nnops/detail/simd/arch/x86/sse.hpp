#pragma once
/// @file sse.hpp
/// @brief SSE4.1 backend for 128-bit float32 vector operations (VecF32x4).
///
/// Maps VecF32x4 to __m128 and implements all operations using SSE intrinsics.
/// Always available on x86_64 (SSE4.1 is baseline on all modern CPUs).

#if !defined(NNOPS_ARCH_X86_64)
  #error "sse.hpp requires x86_64 architecture"
#endif

#include <smmintrin.h>  // SSE4.1
#include <cstdint>

namespace nnops {
namespace simd {
namespace arch {
namespace sse {

// ============================================================
// VecF32x4 — 128-bit float vector, backed by __m128
// ============================================================
struct VecF32x4 {
    __m128 val;

    VecF32x4() = default;
    explicit VecF32x4(__m128 v) : val(v) {}
    explicit VecF32x4(float s) : val(_mm_set1_ps(s)) {}
    VecF32x4(float v0, float v1, float v2, float v3)
        : val(_mm_setr_ps(v0, v1, v2, v3)) {}

    float operator[](int i) const {
        // MSVC-compatible: extract lane via _mm_shuffle_ps + _mm_cvtss_f32
        float tmp[4];
        _mm_storeu_ps(tmp, val);
        return tmp[i];
    }
};

// ============================================================
// VecF32x8 — 256-bit, emulated with two __m128
// ============================================================
struct VecF32x8 {
    __m128 lo, hi;

    VecF32x8() = default;
    explicit VecF32x8(float s) : lo(_mm_set1_ps(s)), hi(_mm_set1_ps(s)) {}
    VecF32x8(__m128 lo_, __m128 hi_) : lo(lo_), hi(hi_) {}
    VecF32x8(float v0, float v1, float v2, float v3,
             float v4, float v5, float v6, float v7)
        : lo(_mm_setr_ps(v0, v1, v2, v3))
        , hi(_mm_setr_ps(v4, v5, v6, v7)) {}
};

// ============================================================
// VecF32x4 operations
// ============================================================
inline VecF32x4 vec_load_f32x4(const float* p) {
    return VecF32x4(_mm_loadu_ps(p));
}
inline void vec_store_f32x4(float* p, const VecF32x4& a) {
    _mm_storeu_ps(p, a.val);
}
inline VecF32x4 vec_set1_f32x4(float s)  { return VecF32x4(_mm_set1_ps(s)); }
inline VecF32x4 vec_zero_f32x4()         { return VecF32x4(_mm_setzero_ps()); }

inline VecF32x4 vec_add_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(_mm_add_ps(a.val, b.val));
}
inline VecF32x4 vec_sub_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(_mm_sub_ps(a.val, b.val));
}
inline VecF32x4 vec_mul_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(_mm_mul_ps(a.val, b.val));
}
inline VecF32x4 vec_div_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(_mm_div_ps(a.val, b.val));
}
inline VecF32x4 vec_fmadd_f32x4(const VecF32x4& a, const VecF32x4& b, const VecF32x4& c) {
    // SSE has no FMA — use mul+add. The AVX/FMA3 path provides true FMA.
    return VecF32x4(_mm_add_ps(_mm_mul_ps(a.val, b.val), c.val));
}

inline VecF32x4 vec_min_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(_mm_min_ps(a.val, b.val));
}
inline VecF32x4 vec_max_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(_mm_max_ps(a.val, b.val));
}
inline VecF32x4 vec_abs_f32x4(const VecF32x4& a) {
    return VecF32x4(_mm_andnot_ps(_mm_set1_ps(-0.0f), a.val));
}
inline VecF32x4 vec_neg_f32x4(const VecF32x4& a) {
    return VecF32x4(_mm_xor_ps(a.val, _mm_set1_ps(-0.0f)));
}

inline VecF32x4 vec_cmplt_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(_mm_cmplt_ps(a.val, b.val));
}
inline VecF32x4 vec_cmple_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(_mm_cmple_ps(a.val, b.val));
}
inline VecF32x4 vec_cmpgt_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(_mm_cmpgt_ps(a.val, b.val));
}
inline VecF32x4 vec_cmpge_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(_mm_cmpge_ps(a.val, b.val));
}
inline VecF32x4 vec_ceq_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(_mm_cmpeq_ps(a.val, b.val));
}

inline VecF32x4 vec_and_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(_mm_and_ps(a.val, b.val));
}
inline VecF32x4 vec_or_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(_mm_or_ps(a.val, b.val));
}

inline VecF32x4 vec_sqrt_f32x4(const VecF32x4& a) {
    return VecF32x4(_mm_sqrt_ps(a.val));
}

// RCP and RSQRT approximations
inline VecF32x4 vec_rcp_f32x4(const VecF32x4& a) {
    return VecF32x4(_mm_rcp_ps(a.val));
}
inline VecF32x4 vec_rsqrt_f32x4(const VecF32x4& a) {
    return VecF32x4(_mm_rsqrt_ps(a.val));
}

// Horizontal sum
inline float vec_reduce_sum_f32x4(const VecF32x4& a) {
    __m128 t = _mm_add_ps(a.val, _mm_movehl_ps(a.val, a.val));
    t = _mm_add_ps(t, _mm_shuffle_ps(t, t, 1));
    return _mm_cvtss_f32(t);
}

// ============================================================
// VecF32x8 operations (emulated with two 128-bit lanes)
// ============================================================
inline VecF32x8 vec_load_f32x8(const float* p) {
    return VecF32x8(_mm_loadu_ps(p), _mm_loadu_ps(p + 4));
}
inline void vec_store_f32x8(float* p, const VecF32x8& a) {
    _mm_storeu_ps(p, a.lo);
    _mm_storeu_ps(p + 4, a.hi);
}
inline VecF32x8 vec_set1_f32x8(float s) {
    __m128 v = _mm_set1_ps(s);
    return VecF32x8(v, v);
}
inline VecF32x8 vec_zero_f32x8() {
    __m128 z = _mm_setzero_ps();
    return VecF32x8(z, z);
}

inline VecF32x8 vec_add_f32x8(const VecF32x8& a, const VecF32x8& b) {
    return VecF32x8(_mm_add_ps(a.lo, b.lo), _mm_add_ps(a.hi, b.hi));
}
inline VecF32x8 vec_sub_f32x8(const VecF32x8& a, const VecF32x8& b) {
    return VecF32x8(_mm_sub_ps(a.lo, b.lo), _mm_sub_ps(a.hi, b.hi));
}
inline VecF32x8 vec_mul_f32x8(const VecF32x8& a, const VecF32x8& b) {
    return VecF32x8(_mm_mul_ps(a.lo, b.lo), _mm_mul_ps(a.hi, b.hi));
}
inline VecF32x8 vec_div_f32x8(const VecF32x8& a, const VecF32x8& b) {
    return VecF32x8(_mm_div_ps(a.lo, b.lo), _mm_div_ps(a.hi, b.hi));
}
inline VecF32x8 vec_fmadd_f32x8(const VecF32x8& a, const VecF32x8& b, const VecF32x8& c) {
    return VecF32x8(
        _mm_add_ps(_mm_mul_ps(a.lo, b.lo), c.lo),
        _mm_add_ps(_mm_mul_ps(a.hi, b.hi), c.hi));
}

inline VecF32x8 vec_min_f32x8(const VecF32x8& a, const VecF32x8& b) {
    return VecF32x8(_mm_min_ps(a.lo, b.lo), _mm_min_ps(a.hi, b.hi));
}
inline VecF32x8 vec_max_f32x8(const VecF32x8& a, const VecF32x8& b) {
    return VecF32x8(_mm_max_ps(a.lo, b.lo), _mm_max_ps(a.hi, b.hi));
}
inline VecF32x8 vec_sqrt_f32x8(const VecF32x8& a) {
    return VecF32x8(_mm_sqrt_ps(a.lo), _mm_sqrt_ps(a.hi));
}

inline float vec_reduce_sum_f32x8(const VecF32x8& a) {
    VecF32x4 lo(a.lo), hi(a.hi);
    return vec_reduce_sum_f32x4(lo) + vec_reduce_sum_f32x4(hi);
}

} // namespace sse
} // namespace arch
} // namespace simd
} // namespace nnops
