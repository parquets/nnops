#pragma once
/// @file sse.hpp
/// @brief SSE4.1 backend for 128-bit float32 vector operations (v_f32x4).
///
/// Maps v_f32x4 to __m128 and implements all operations using SSE intrinsics.
/// Always available on x86_64 (SSE4.1 is baseline on all modern CPUs).

#if !defined(NNOPS_ARCH_X86_64)
  #error "sse.hpp requires x86_64 architecture"
#endif

#include <smmintrin.h>  // SSE4.1
#include <cstdint>
#include <algorithm>
#include "sse_mathfunc.hpp"

namespace nnops {
namespace simd {
namespace arch {
namespace sse {

// ============================================================
// v_f32x4 — 128-bit float vector, backed by __m128
// ============================================================
struct v_f32x4 {
    __m128 val;

    v_f32x4() = default;
    explicit v_f32x4(__m128 v) : val(v) {}
    explicit v_f32x4(float s) : val(_mm_set1_ps(s)) {}
    v_f32x4(float v0, float v1, float v2, float v3)
        : val(_mm_setr_ps(v0, v1, v2, v3)) {}

    float operator[](int i) const {
        // MSVC-compatible: extract lane via _mm_shuffle_ps + _mm_cvtss_f32
        float tmp[4];
        _mm_storeu_ps(tmp, val);
        return tmp[i];
    }
};

// ============================================================
// v_f32x8 — 256-bit, emulated with two __m128
// ============================================================
struct v_f32x8 {
    __m128 lo, hi;

    v_f32x8() = default;
    explicit v_f32x8(float s) : lo(_mm_set1_ps(s)), hi(_mm_set1_ps(s)) {}
    v_f32x8(__m128 lo_, __m128 hi_) : lo(lo_), hi(hi_) {}
    v_f32x8(float v0, float v1, float v2, float v3,
             float v4, float v5, float v6, float v7)
        : lo(_mm_setr_ps(v0, v1, v2, v3))
        , hi(_mm_setr_ps(v4, v5, v6, v7)) {}
};

// ============================================================
// v_f32x4 operations
// ============================================================
inline v_f32x4 v_load_f32x4(const float* p) {
    return v_f32x4(_mm_loadu_ps(p));
}
inline void v_store(float* p, const v_f32x4& a) {
    _mm_storeu_ps(p, a.val);
}
inline v_f32x4 v_set1_f32x4(float s)  { return v_f32x4(_mm_set1_ps(s)); }
inline v_f32x4 v_zero_f32x4()         { return v_f32x4(_mm_setzero_ps()); }

inline v_f32x4 v_add(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(_mm_add_ps(a.val, b.val));
}
inline v_f32x4 v_sub(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(_mm_sub_ps(a.val, b.val));
}
inline v_f32x4 v_mul(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(_mm_mul_ps(a.val, b.val));
}
inline v_f32x4 v_div(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(_mm_div_ps(a.val, b.val));
}
inline v_f32x4 v_fmadd(const v_f32x4& a, const v_f32x4& b, const v_f32x4& c) {
    // SSE has no FMA — use v_mul+v_add. The AVX/FMA3 path provides true FMA.
    return v_f32x4(_mm_add_ps(_mm_mul_ps(a.val, b.val), c.val));
}

inline v_f32x4 v_min(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(_mm_min_ps(a.val, b.val));
}
inline v_f32x4 v_max(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(_mm_max_ps(a.val, b.val));
}
inline v_f32x4 v_abs(const v_f32x4& a) {
    return v_f32x4(_mm_andnot_ps(_mm_set1_ps(-0.0f), a.val));
}
inline v_f32x4 v_neg(const v_f32x4& a) {
    return v_f32x4(_mm_xor_ps(a.val, _mm_set1_ps(-0.0f)));
}

inline v_f32x4 v_cmplt(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(_mm_cmplt_ps(a.val, b.val));
}
inline v_f32x4 v_cmple(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(_mm_cmple_ps(a.val, b.val));
}
inline v_f32x4 v_cmpgt(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(_mm_cmpgt_ps(a.val, b.val));
}
inline v_f32x4 v_cmpge(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(_mm_cmpge_ps(a.val, b.val));
}
inline v_f32x4 v_ceq(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(_mm_cmpeq_ps(a.val, b.val));
}

inline v_f32x4 v_and(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(_mm_and_ps(a.val, b.val));
}
inline v_f32x4 v_or(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(_mm_or_ps(a.val, b.val));
}

inline v_f32x4 v_sqrt(const v_f32x4& a) {
    return v_f32x4(_mm_sqrt_ps(a.val));
}

// RCP and RSQRT approximations
inline v_f32x4 v_rcp(const v_f32x4& a) {
    return v_f32x4(_mm_rcp_ps(a.val));
}
inline v_f32x4 v_rsqrt(const v_f32x4& a) {
    return v_f32x4(_mm_rsqrt_ps(a.val));
}

// Transcendental math functions (delegated to sse_mathfunc.hpp)
inline v_f32x4 v_exp(const v_f32x4& a) {
    return v_f32x4(exp_ps(a.val));
}
inline v_f32x4 v_log(const v_f32x4& a) {
    return v_f32x4(log_ps(a.val));
}
inline v_f32x4 v_sin(const v_f32x4& a) {
    return v_f32x4(sin_ps(a.val));
}
inline v_f32x4 v_cos(const v_f32x4& a) {
    return v_f32x4(cos_ps(a.val));
}
inline v_f32x4 v_tan(const v_f32x4& a) {
    return v_f32x4(tan_ps(a.val));
}
inline v_f32x4 v_tanh(const v_f32x4& a) {
    return v_f32x4(tanh_ps(a.val));
}

// Horizontal sum
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
// v_f32x8 operations (emulated with two 128-bit lanes)
// ============================================================
inline v_f32x8 v_load_f32x8(const float* p) {
    return v_f32x8(_mm_loadu_ps(p), _mm_loadu_ps(p + 4));
}
inline void v_store(float* p, const v_f32x8& a) {
    _mm_storeu_ps(p, a.lo);
    _mm_storeu_ps(p + 4, a.hi);
}
inline v_f32x8 v_set1_f32x8(float s) {
    __m128 v = _mm_set1_ps(s);
    return v_f32x8(v, v);
}
inline v_f32x8 v_zero_f32x8() {
    __m128 z = _mm_setzero_ps();
    return v_f32x8(z, z);
}

inline v_f32x8 v_add(const v_f32x8& a, const v_f32x8& b) {
    return v_f32x8(_mm_add_ps(a.lo, b.lo), _mm_add_ps(a.hi, b.hi));
}
inline v_f32x8 v_sub(const v_f32x8& a, const v_f32x8& b) {
    return v_f32x8(_mm_sub_ps(a.lo, b.lo), _mm_sub_ps(a.hi, b.hi));
}
inline v_f32x8 v_mul(const v_f32x8& a, const v_f32x8& b) {
    return v_f32x8(_mm_mul_ps(a.lo, b.lo), _mm_mul_ps(a.hi, b.hi));
}
inline v_f32x8 v_div(const v_f32x8& a, const v_f32x8& b) {
    return v_f32x8(_mm_div_ps(a.lo, b.lo), _mm_div_ps(a.hi, b.hi));
}
inline v_f32x8 v_fmadd(const v_f32x8& a, const v_f32x8& b, const v_f32x8& c) {
    return v_f32x8(
        _mm_add_ps(_mm_mul_ps(a.lo, b.lo), c.lo),
        _mm_add_ps(_mm_mul_ps(a.hi, b.hi), c.hi));
}

inline v_f32x8 v_min(const v_f32x8& a, const v_f32x8& b) {
    return v_f32x8(_mm_min_ps(a.lo, b.lo), _mm_min_ps(a.hi, b.hi));
}
inline v_f32x8 v_max(const v_f32x8& a, const v_f32x8& b) {
    return v_f32x8(_mm_max_ps(a.lo, b.lo), _mm_max_ps(a.hi, b.hi));
}
inline v_f32x8 v_abs(const v_f32x8& a) {
    return v_f32x8(_mm_andnot_ps(_mm_set1_ps(-0.0f), a.lo),
                    _mm_andnot_ps(_mm_set1_ps(-0.0f), a.hi));
}
inline v_f32x8 v_neg(const v_f32x8& a) {
    return v_f32x8(_mm_xor_ps(a.lo, _mm_set1_ps(-0.0f)),
                    _mm_xor_ps(a.hi, _mm_set1_ps(-0.0f)));
}
inline v_f32x8 v_rcp(const v_f32x8& a) {
    return v_f32x8(_mm_rcp_ps(a.lo), _mm_rcp_ps(a.hi));
}
inline v_f32x8 v_sqrt(const v_f32x8& a) {
    return v_f32x8(_mm_sqrt_ps(a.lo), _mm_sqrt_ps(a.hi));
}

// Compare operations for v_f32x8 (emulated)
inline v_f32x8 v_cmplt(const v_f32x8& a, const v_f32x8& b) {
    return v_f32x8(_mm_cmplt_ps(a.lo, b.lo), _mm_cmplt_ps(a.hi, b.hi));
}
inline v_f32x8 v_cmpgt(const v_f32x8& a, const v_f32x8& b) {
    return v_f32x8(_mm_cmpgt_ps(a.lo, b.lo), _mm_cmpgt_ps(a.hi, b.hi));
}

// Bitwise operations for v_f32x8 (emulated)
inline v_f32x8 v_and(const v_f32x8& a, const v_f32x8& b) {
    return v_f32x8(_mm_and_ps(a.lo, b.lo), _mm_and_ps(a.hi, b.hi));
}
inline v_f32x8 v_or(const v_f32x8& a, const v_f32x8& b) {
    return v_f32x8(_mm_or_ps(a.lo, b.lo), _mm_or_ps(a.hi, b.hi));
}

// Transcendental math functions for v_f32x8 (lane-wise via v_f32x4)
inline v_f32x8 v_exp(const v_f32x8& a) {
    return v_f32x8(exp_ps(a.lo), exp_ps(a.hi));
}
inline v_f32x8 v_log(const v_f32x8& a) {
    return v_f32x8(log_ps(a.lo), log_ps(a.hi));
}
inline v_f32x8 v_sin(const v_f32x8& a) {
    return v_f32x8(sin_ps(a.lo), sin_ps(a.hi));
}
inline v_f32x8 v_cos(const v_f32x8& a) {
    return v_f32x8(cos_ps(a.lo), cos_ps(a.hi));
}
inline v_f32x8 v_tan(const v_f32x8& a) {
    return v_f32x8(tan_ps(a.lo), tan_ps(a.hi));
}
inline v_f32x8 v_tanh(const v_f32x8& a) {
    return v_f32x8(tanh_ps(a.lo), tanh_ps(a.hi));
}

inline float v_reduce_sum(const v_f32x8& a) {
    v_f32x4 lo(a.lo), hi(a.hi);
    return v_reduce_sum(lo) + v_reduce_sum(hi);
}
inline float v_reduce_max(const v_f32x8& a) {
    v_f32x4 lo(a.lo), hi(a.hi);
    return std::max(v_reduce_max(lo), v_reduce_max(hi));
}
inline float v_reduce_min(const v_f32x8& a) {
    v_f32x4 lo(a.lo), hi(a.hi);
    return std::min(v_reduce_min(lo), v_reduce_min(hi));
}

// ============================================================
// v_f16x8 — 128-bit fp16 vector, backed by __m128i
//
// Uses F16C intrinsics (_mm256_cvtph_ps / _mm256_cvtps_ph) for
// convert→compute→convert: widen fp16→fp32, compute with AVX/AVX2,
// narrow fp32→fp16. Assumes target CPU supports F16C, AVX2, FMA3.
// ============================================================
#if defined(__F16C__)

#include <immintrin.h>
#include "avx2_mathfunc.hpp"

struct v_f16x8 {
    __m128i val;  // 8 × fp16 (binary16) values

    v_f16x8() = default;
    explicit v_f16x8(__m128i v) : val(v) {}
    explicit v_f16x8(float s) : val(_mm256_cvtps_ph(_mm256_set1_ps(s), 0)) {}
};

// Load / v_store
inline v_f16x8 v_load_f16x8(const uint16_t* p) {
    return v_f16x8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}
inline void v_store(uint16_t* p, const v_f16x8& a) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), a.val);
}

// Broadcast / zero
inline v_f16x8 v_set1_f16x8(float s) {
    return v_f16x8(_mm256_cvtps_ph(_mm256_set1_ps(s), 0));
}
inline v_f16x8 v_zero_f16x8() {
    return v_f16x8(_mm_setzero_si128());
}

// Helper: extract __m256 from v_f32x8
inline __m256 to_m256(const v_f32x8& a) {
    return _mm256_insertf128_ps(
        _mm256_castps128_ps256(a.lo), a.hi, 1);
}

// Conversion: v_f16x8 ↔ v_f32x8
inline v_f32x8 v_cvt_f16_to_f32(const v_f16x8& a) {
    __m256 f32 = _mm256_cvtph_ps(a.val);
    return v_f32x8(
        _mm256_castps256_ps128(f32),
        _mm256_extractf128_ps(f32, 1));
}
inline v_f16x8 v_cvt_f32_to_f16(const v_f32x8& a) {
    __m256 f32 = to_m256(a);
    return v_f16x8(_mm256_cvtps_ph(f32, 0));
}

// Helper: convert both args, apply __m256 op, convert back
#define NNOPS_F16X8_BINOP(name, op) \
inline v_f16x8 name(const v_f16x8& a, const v_f16x8& b) { \
    __m256 fa = _mm256_cvtph_ps(a.val); \
    __m256 fb = _mm256_cvtph_ps(b.val); \
    return v_f16x8(_mm256_cvtps_ph(op(fa, fb), 0)); \
}

NNOPS_F16X8_BINOP(v_add, _mm256_add_ps)
NNOPS_F16X8_BINOP(v_sub, _mm256_sub_ps)
NNOPS_F16X8_BINOP(v_mul, _mm256_mul_ps)
NNOPS_F16X8_BINOP(v_div, _mm256_div_ps)
NNOPS_F16X8_BINOP(v_min, _mm256_min_ps)
NNOPS_F16X8_BINOP(v_max, _mm256_max_ps)

#undef NNOPS_F16X8_BINOP

// v_fmadd: a*b + c (fused multiply-v_add via FMA3)
inline v_f16x8 v_fmadd(const v_f16x8& a, const v_f16x8& b, const v_f16x8& c) {
    __m256 fa = _mm256_cvtph_ps(a.val);
    __m256 fb = _mm256_cvtph_ps(b.val);
    __m256 fc = _mm256_cvtph_ps(c.val);
    return v_f16x8(_mm256_cvtps_ph(_mm256_fmadd_ps(fa, fb, fc), 0));
}

// Sign operations (bitwise on fp16 bits)
inline v_f16x8 v_abs(const v_f16x8& a) {
    // Clear sign bit: ~0x8000 & val
    return v_f16x8(_mm_andnot_si128(_mm_set1_epi16(0x8000), a.val));
}
inline v_f16x8 v_neg(const v_f16x8& a) {
    // Flip sign bit
    return v_f16x8(_mm_xor_si128(a.val, _mm_set1_epi16(0x8000)));
}

// Comparisons — returns 0xFFFF per lane (true) or 0x0000 (false)
// Strategy: convert to fp32, compare → 0xFFFFFFFF/0x00000000 mask,
// shift right by 16, then pack 32-bit lanes to 16-bit.
inline v_f16x8 v_cmplt(const v_f16x8& a, const v_f16x8& b) {
    __m256 fa = _mm256_cvtph_ps(a.val);
    __m256 fb = _mm256_cvtph_ps(b.val);
    __m256i mask = _mm256_castps_si256(_mm256_cmp_ps(fa, fb, _CMP_LT_OS));
    mask = _mm256_srli_epi32(mask, 16);
    __m128i lo = _mm256_castsi256_si128(mask);
    __m128i hi = _mm256_extracti128_si256(mask, 1);
    return v_f16x8(_mm_packus_epi32(lo, hi));
}
inline v_f16x8 v_cmpgt(const v_f16x8& a, const v_f16x8& b) {
    return v_cmplt(b, a);
}

// v_sqrt
inline v_f16x8 v_sqrt(const v_f16x8& a) {
    __m256 f32 = _mm256_cvtph_ps(a.val);
    return v_f16x8(_mm256_cvtps_ph(_mm256_sqrt_ps(f32), 0));
}

// Transcendental math — widen to fp32, use AVX2 math functions, narrow back
inline v_f16x8 v_exp(const v_f16x8& a) {
    __m256 f32 = _mm256_cvtph_ps(a.val);
    return v_f16x8(_mm256_cvtps_ph(avx2::exp256_ps(f32), 0));
}
inline v_f16x8 v_log(const v_f16x8& a) {
    __m256 f32 = _mm256_cvtph_ps(a.val);
    return v_f16x8(_mm256_cvtps_ph(avx2::log256_ps(f32), 0));
}
inline v_f16x8 v_sin(const v_f16x8& a) {
    __m256 f32 = _mm256_cvtph_ps(a.val);
    return v_f16x8(_mm256_cvtps_ph(avx2::sin256_ps(f32), 0));
}
inline v_f16x8 v_cos(const v_f16x8& a) {
    __m256 f32 = _mm256_cvtph_ps(a.val);
    return v_f16x8(_mm256_cvtps_ph(avx2::cos256_ps(f32), 0));
}
inline v_f16x8 v_tan(const v_f16x8& a) {
    __m256 f32 = _mm256_cvtph_ps(a.val);
    return v_f16x8(_mm256_cvtps_ph(avx2::tan256_ps(f32), 0));
}
inline v_f16x8 v_tanh(const v_f16x8& a) {
    __m256 f32 = _mm256_cvtph_ps(a.val);
    return v_f16x8(_mm256_cvtps_ph(avx2::tanh256_ps(f32), 0));
}

// Horizontal reductions — widen to fp32, then pairwise reduce
inline float v_reduce_sum(const v_f16x8& a) {
    __m256 f32 = _mm256_cvtph_ps(a.val);
    __m128 lo = _mm256_castps256_ps128(f32);
    __m128 hi = _mm256_extractf128_ps(f32, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    __m128 t = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    t = _mm_add_ps(t, _mm_shuffle_ps(t, t, 1));
    return _mm_cvtss_f32(t);
}
inline float v_reduce_max(const v_f16x8& a) {
    __m256 f32 = _mm256_cvtph_ps(a.val);
    __m128 lo = _mm256_castps256_ps128(f32);
    __m128 hi = _mm256_extractf128_ps(f32, 1);
    __m128 max128 = _mm_max_ps(lo, hi);
    __m128 t = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
    t = _mm_max_ps(t, _mm_shuffle_ps(t, t, 1));
    return _mm_cvtss_f32(t);
}
inline float v_reduce_min(const v_f16x8& a) {
    __m256 f32 = _mm256_cvtph_ps(a.val);
    __m128 lo = _mm256_castps256_ps128(f32);
    __m128 hi = _mm256_extractf128_ps(f32, 1);
    __m128 min128 = _mm_min_ps(lo, hi);
    __m128 t = _mm_min_ps(min128, _mm_movehl_ps(min128, min128));
    t = _mm_min_ps(t, _mm_shuffle_ps(t, t, 1));
    return _mm_cvtss_f32(t);
}

#endif  // defined(__F16C__)

// ============================================================
// Deinterleave (stride-2 gather) — pair types
// ============================================================

struct v_f32x4x2_t { v_f32x4 even; v_f32x4 odd; };
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
    return {v_f32x8(lo.even.val, hi.even.val),
            v_f32x8(lo.odd.val,  hi.odd.val)};
}

// ============================================================
// 方案 B: v_load_even_* / v_load_odd_*
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
// 方案 C: v_load_stride2_even_* / v_load_stride2_odd_*
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
// FP16 deinterleave (F16C path)
// ============================================================
#if defined(__F16C__)

/// @brief Pair of v_f16x8 registers: even- and odd-indexed half elements.
struct v_f16x8x2_t { v_f16x8 even; v_f16x8 odd; };

/// @brief Load 16 contiguous f16s, return {h0,h2,...,h14}, {h1,h3,...,h15}.
///
/// Strategy: convert to f32, deinterleave with UNPCK cascade, convert back.
/// The stack buffer (16 f32 = 64 B) stays in L1; overhead is negligible.
inline v_f16x8x2_t v_deinterleave_f16x8(const uint16_t* src) {
    // Load 2×8 f16 → convert to 2×8 f32
    __m256 flo = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src)));
    __m256 fhi = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 8)));

    // Deinterleave each 128-bit half of flo/fhi via UNPCK cascade
    __m128 flo_lo = _mm256_castps256_ps128(flo);       // f0,f1,f2,f3
    __m128 flo_hi = _mm256_extractf128_ps(flo, 1);      // f4,f5,f6,f7
    __m128 fhi_lo = _mm256_castps256_ps128(fhi);        // f8,f9,f10,f11
    __m128 fhi_hi = _mm256_extractf128_ps(fhi, 1);       // f12,f13,f14,f15

    // Low pair: deinterleave flo_lo (f0..f3) × fhi_lo (f8..f11)
    __m128 t0 = _mm_unpacklo_ps(flo_lo, fhi_lo);  // f0,f8, f1,f9
    __m128 t1 = _mm_unpackhi_ps(flo_lo, fhi_lo);  // f2,f10,f3,f11
    __m128 even_lo = _mm_unpacklo_ps(t0, t1);      // f0,f2, f8,f10
    __m128 odd_lo  = _mm_unpackhi_ps(t0, t1);      // f1,f3, f9,f11

    // High pair: deinterleave flo_hi (f4..f7) × fhi_hi (f12..f15)
    t0 = _mm_unpacklo_ps(flo_hi, fhi_hi);  // f4,f12, f5,f13
    t1 = _mm_unpackhi_ps(flo_hi, fhi_hi);  // f6,f14, f7,f15
    __m128 even_hi = _mm_unpacklo_ps(t0, t1);  // f4,f6, f12,f14
    __m128 odd_hi  = _mm_unpackhi_ps(t0, t1);  // f5,f7, f13,f15

    // Recombine: even = f0,f2,f4,f6, f8,f10,f12,f14
    __m128 ev_lo = _mm_shuffle_ps(even_lo, even_hi, _MM_SHUFFLE(1,0,1,0));
    __m128 ev_hi = _mm_shuffle_ps(even_lo, even_hi, _MM_SHUFFLE(3,2,3,2));
    __m256 even_f = _mm256_insertf128_ps(
        _mm256_castps128_ps256(ev_lo), ev_hi, 1);

    // Recombine: odd = f1,f3,f5,f7, f9,f11,f13,f15
    __m128 od_lo = _mm_shuffle_ps(odd_lo, odd_hi, _MM_SHUFFLE(1,0,1,0));
    __m128 od_hi = _mm_shuffle_ps(odd_lo, odd_hi, _MM_SHUFFLE(3,2,3,2));
    __m256 odd_f = _mm256_insertf128_ps(
        _mm256_castps128_ps256(od_lo), od_hi, 1);

    return {v_f16x8(_mm256_cvtps_ph(even_f, 0)),
            v_f16x8(_mm256_cvtps_ph(odd_f,  0))};
}

// 方案 B — f16
inline v_f16x8 v_load_even_f16x8(const uint16_t* src) {
    return v_deinterleave_f16x8(src).even;
}
inline v_f16x8 v_load_odd_f16x8(const uint16_t* src) {
    return v_deinterleave_f16x8(src).odd;
}

// 方案 C — f16
inline v_f16x8 v_load_stride2_even_f16x8(const uint16_t* src) {
    return v_deinterleave_f16x8(src).even;
}
inline v_f16x8 v_load_stride2_odd_f16x8(const uint16_t* src) {
    return v_deinterleave_f16x8(src).odd;
}

#endif  // defined(__F16C__)

} // namespace sse
} // namespace arch
} // namespace simd
} // namespace nnops
