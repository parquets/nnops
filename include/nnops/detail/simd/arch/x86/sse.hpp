#pragma once
/// @file sse.hpp
/// @brief SSE4.1 backend for 128-bit float32 vector operations (v_fp32x4).
///
/// Maps v_fp32x4 to __m128 and implements all operations using SSE intrinsics.
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
// v_fp32x4 — 128-bit float vector, backed by __m128
// ============================================================
struct v_fp32x4 {
    __m128 val;

    v_fp32x4() = default;
    explicit v_fp32x4(__m128 v) : val(v) {}
    explicit v_fp32x4(float s) : val(_mm_set1_ps(s)) {}
    v_fp32x4(float v0, float v1, float v2, float v3)
        : val(_mm_setr_ps(v0, v1, v2, v3)) {}

    float operator[](int i) const {
        // MSVC-compatible: extract lane via _mm_shuffle_ps + _mm_cvtss_f32
        float tmp[4];
        _mm_storeu_ps(tmp, val);
        return tmp[i];
    }
};

// ============================================================
// v_fp32x8 — 256-bit, emulated with two __m128
// ============================================================
struct v_fp32x8 {
    __m128 lo, hi;

    v_fp32x8() = default;
    explicit v_fp32x8(float s) : lo(_mm_set1_ps(s)), hi(_mm_set1_ps(s)) {}
    v_fp32x8(__m128 lo_, __m128 hi_) : lo(lo_), hi(hi_) {}
    v_fp32x8(float v0, float v1, float v2, float v3,
             float v4, float v5, float v6, float v7)
        : lo(_mm_setr_ps(v0, v1, v2, v3))
        , hi(_mm_setr_ps(v4, v5, v6, v7)) {}
};

// ============================================================
// v_fp32x4 operations
// ============================================================
inline v_fp32x4 load_fp32x4(const float* p) {
    return v_fp32x4(_mm_loadu_ps(p));
}
inline void store(float* p, const v_fp32x4& a) {
    _mm_storeu_ps(p, a.val);
}
inline v_fp32x4 set1_fp32x4(float s)  { return v_fp32x4(_mm_set1_ps(s)); }
inline v_fp32x4 zero_fp32x4()         { return v_fp32x4(_mm_setzero_ps()); }

inline v_fp32x4 add(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(_mm_add_ps(a.val, b.val));
}
inline v_fp32x4 sub(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(_mm_sub_ps(a.val, b.val));
}
inline v_fp32x4 mul(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(_mm_mul_ps(a.val, b.val));
}
inline v_fp32x4 div(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(_mm_div_ps(a.val, b.val));
}
inline v_fp32x4 fmadd(const v_fp32x4& a, const v_fp32x4& b, const v_fp32x4& c) {
    // SSE has no FMA — use mul+add. The AVX/FMA3 path provides true FMA.
    return v_fp32x4(_mm_add_ps(_mm_mul_ps(a.val, b.val), c.val));
}

inline v_fp32x4 min(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(_mm_min_ps(a.val, b.val));
}
inline v_fp32x4 max(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(_mm_max_ps(a.val, b.val));
}
inline v_fp32x4 abs(const v_fp32x4& a) {
    return v_fp32x4(_mm_andnot_ps(_mm_set1_ps(-0.0f), a.val));
}
inline v_fp32x4 neg(const v_fp32x4& a) {
    return v_fp32x4(_mm_xor_ps(a.val, _mm_set1_ps(-0.0f)));
}

inline v_fp32x4 cmplt(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(_mm_cmplt_ps(a.val, b.val));
}
inline v_fp32x4 cmple(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(_mm_cmple_ps(a.val, b.val));
}
inline v_fp32x4 cmpgt(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(_mm_cmpgt_ps(a.val, b.val));
}
inline v_fp32x4 cmpge(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(_mm_cmpge_ps(a.val, b.val));
}
inline v_fp32x4 ceq(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(_mm_cmpeq_ps(a.val, b.val));
}

inline v_fp32x4 and_(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(_mm_and_ps(a.val, b.val));
}
inline v_fp32x4 or_(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(_mm_or_ps(a.val, b.val));
}

inline v_fp32x4 sqrt(const v_fp32x4& a) {
    return v_fp32x4(_mm_sqrt_ps(a.val));
}

// RCP and RSQRT approximations
inline v_fp32x4 rcp(const v_fp32x4& a) {
    return v_fp32x4(_mm_rcp_ps(a.val));
}
inline v_fp32x4 rsqrt(const v_fp32x4& a) {
    return v_fp32x4(_mm_rsqrt_ps(a.val));
}

// Horizontal sum
inline float reduce_sum(const v_fp32x4& a) {
    __m128 t = _mm_add_ps(a.val, _mm_movehl_ps(a.val, a.val));
    t = _mm_add_ps(t, _mm_shuffle_ps(t, t, 1));
    return _mm_cvtss_f32(t);
}

// ============================================================
// v_fp32x8 operations (emulated with two 128-bit lanes)
// ============================================================
inline v_fp32x8 load_fp32x8(const float* p) {
    return v_fp32x8(_mm_loadu_ps(p), _mm_loadu_ps(p + 4));
}
inline void store(float* p, const v_fp32x8& a) {
    _mm_storeu_ps(p, a.lo);
    _mm_storeu_ps(p + 4, a.hi);
}
inline v_fp32x8 set1_fp32x8(float s) {
    __m128 v = _mm_set1_ps(s);
    return v_fp32x8(v, v);
}
inline v_fp32x8 zero_fp32x8() {
    __m128 z = _mm_setzero_ps();
    return v_fp32x8(z, z);
}

inline v_fp32x8 add(const v_fp32x8& a, const v_fp32x8& b) {
    return v_fp32x8(_mm_add_ps(a.lo, b.lo), _mm_add_ps(a.hi, b.hi));
}
inline v_fp32x8 sub(const v_fp32x8& a, const v_fp32x8& b) {
    return v_fp32x8(_mm_sub_ps(a.lo, b.lo), _mm_sub_ps(a.hi, b.hi));
}
inline v_fp32x8 mul(const v_fp32x8& a, const v_fp32x8& b) {
    return v_fp32x8(_mm_mul_ps(a.lo, b.lo), _mm_mul_ps(a.hi, b.hi));
}
inline v_fp32x8 div(const v_fp32x8& a, const v_fp32x8& b) {
    return v_fp32x8(_mm_div_ps(a.lo, b.lo), _mm_div_ps(a.hi, b.hi));
}
inline v_fp32x8 fmadd(const v_fp32x8& a, const v_fp32x8& b, const v_fp32x8& c) {
    return v_fp32x8(
        _mm_add_ps(_mm_mul_ps(a.lo, b.lo), c.lo),
        _mm_add_ps(_mm_mul_ps(a.hi, b.hi), c.hi));
}

inline v_fp32x8 min(const v_fp32x8& a, const v_fp32x8& b) {
    return v_fp32x8(_mm_min_ps(a.lo, b.lo), _mm_min_ps(a.hi, b.hi));
}
inline v_fp32x8 max(const v_fp32x8& a, const v_fp32x8& b) {
    return v_fp32x8(_mm_max_ps(a.lo, b.lo), _mm_max_ps(a.hi, b.hi));
}
inline v_fp32x8 sqrt(const v_fp32x8& a) {
    return v_fp32x8(_mm_sqrt_ps(a.lo), _mm_sqrt_ps(a.hi));
}

inline float reduce_sum(const v_fp32x8& a) {
    v_fp32x4 lo(a.lo), hi(a.hi);
    return reduce_sum(lo) + reduce_sum(hi);
}

} // namespace sse
} // namespace arch
} // namespace simd
} // namespace nnops
