#pragma once
/// @file neon.hpp
/// @brief ARM NEON backend for 128-bit float32 vector operations (v_f32x4).
///
/// NEON (ASIMD) is mandatory on AArch64. Provides native 128-bit float operations.
/// For 256-bit v_f32x8, we emulate with two 128-bit NEON registers since NEON
/// does not natively have 256-bit vectors.
///
/// Reference: OpenCV intrin_neon.hpp, onnxruntime MLAS NEON kernels.

#if !defined(NNOPS_ARCH_AARCH64)
  #error "neon.hpp requires AArch64 architecture"
#endif

#include <arm_neon.h>
#include <cmath>  // for std::abs in vec_abs emulation
#include "neon_mathfunc.hpp"

namespace nnops {
namespace simd {
namespace arch {
namespace neon {

// ============================================================
// v_f32x4 — 128-bit float vector, backed by float32x4_t
// ============================================================
struct v_f32x4 {
    float32x4_t val;

    v_f32x4() = default;
    explicit v_f32x4(float32x4_t v) : val(v) {}
    explicit v_f32x4(float s) : val(vdupq_n_f32(s)) {}
    v_f32x4(float v0, float v1, float v2, float v3) {
        float tmp[4] = {v0, v1, v2, v3};
        val = vld1q_f32(tmp);
    }

    float operator[](int i) const { return vgetq_lane_f32(val, i); }
};

// ============================================================
// v_f32x8 — 256-bit, emulated with two float32x4_t
// ============================================================
struct v_f32x8 {
    float32x4_t lo, hi;

    v_f32x8() = default;
    explicit v_f32x8(float s) : lo(vdupq_n_f32(s)), hi(vdupq_n_f32(s)) {}
    v_f32x8(float32x4_t lo_, float32x4_t hi_) : lo(lo_), hi(hi_) {}
    v_f32x8(float v0, float v1, float v2, float v3,
             float v4, float v5, float v6, float v7) {
        float tmp_lo[4] = {v0, v1, v2, v3};
        float tmp_hi[4] = {v4, v5, v6, v7};
        lo = vld1q_f32(tmp_lo);
        hi = vld1q_f32(tmp_hi);
    }
};

// ============================================================
// v_f32x4 operations
// ============================================================
inline v_f32x4 v_load_f32x4(const float* p) {
    return v_f32x4(vld1q_f32(p));
}
inline void v_store(float* p, const v_f32x4& a) {
    vst1q_f32(p, a.val);
}
inline v_f32x4 v_set1_f32x4(float s)  { return v_f32x4(vdupq_n_f32(s)); }
inline v_f32x4 v_zero_f32x4() {
    return v_f32x4(vdupq_n_f32(0.0f));
}

inline v_f32x4 v_add(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(vaddq_f32(a.val, b.val));
}
inline v_f32x4 v_sub(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(vsubq_f32(a.val, b.val));
}
inline v_f32x4 v_mul(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(vmulq_f32(a.val, b.val));
}

// Division: NEON doesn't have native float v_div, use reciprocal + NR step.
// For simplicity use vrecpe + vrecps (Newton-Raphson) or just the scalar fallback.
// Actually, GCC/Clang provide __builtin_vdivq_f32 (compiler synthesizes).
inline v_f32x4 v_div(const v_f32x4& a, const v_f32x4& b) {
    // Use reciprocal approximation with Newton-Raphson refinement
    float32x4_t recip = vrecpeq_f32(b.val);          // initial estimate ~1/b
    recip = vmulq_f32(vrecpsq_f32(b.val, recip), recip);  // Newton-Raphson step
    recip = vmulq_f32(vrecpsq_f32(b.val, recip), recip);  // second NR step for full precision
    return v_f32x4(vmulq_f32(a.val, recip));
}

inline v_f32x4 v_fmadd(const v_f32x4& a, const v_f32x4& b, const v_f32x4& c) {
    // vfmaq_f32: c + a * b (results stored in first arg convention)
    return v_f32x4(vfmaq_f32(c.val, a.val, b.val));
}

inline v_f32x4 v_min(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(vminq_f32(a.val, b.val));
}
inline v_f32x4 v_max(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(vmaxq_f32(a.val, b.val));
}
inline v_f32x4 v_abs(const v_f32x4& a) {
    return v_f32x4(vabsq_f32(a.val));
}
inline v_f32x4 v_neg(const v_f32x4& a) {
    return v_f32x4(vnegq_f32(a.val));
}

inline v_f32x4 v_cmplt(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(vreinterpretq_f32_u32(vcltq_f32(a.val, b.val)));
}
inline v_f32x4 v_cmple(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(vreinterpretq_f32_u32(vcleq_f32(a.val, b.val)));
}
inline v_f32x4 v_cmpgt(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(vreinterpretq_f32_u32(vcgtq_f32(a.val, b.val)));
}
inline v_f32x4 v_cmpge(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(vreinterpretq_f32_u32(vcgeq_f32(a.val, b.val)));
}
inline v_f32x4 v_ceq(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(vreinterpretq_f32_u32(vceqq_f32(a.val, b.val)));
}

inline v_f32x4 v_and(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(vreinterpretq_f32_u32(
        vandq_u32(vreinterpretq_u32_f32(a.val), vreinterpretq_u32_f32(b.val))));
}
inline v_f32x4 v_or(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(vreinterpretq_f32_u32(
        vorrq_u32(vreinterpretq_u32_f32(a.val), vreinterpretq_u32_f32(b.val))));
}

inline v_f32x4 v_sqrt(const v_f32x4& a) {
    // NEON: vsqrt is not always available. Use vrsqrte + NR + v_mul for portability.
    float32x4_t est = vrsqrteq_f32(a.val);          // 1/v_sqrt(a) estimate
    est = vmulq_f32(vrsqrtsq_f32(a.val, vmulq_f32(est, est)), est);  // NR step
    est = vmulq_f32(vrsqrtsq_f32(a.val, vmulq_f32(est, est)), est);  // second NR
    return v_f32x4(vmulq_f32(a.val, est));          // a * 1/v_sqrt(a) = v_sqrt(a)
}

// Reciprocal with Newton-Raphson refinement
inline v_f32x4 v_rcp(const v_f32x4& a) {
    float32x4_t est = vrecpeq_f32(a.val);
    est = vmulq_f32(vrecpsq_f32(a.val, est), est);
    return v_f32x4(vmulq_f32(vrecpsq_f32(a.val, est), est));
}

// Transcendental math functions (delegated to neon_mathfunc.hpp)
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

inline float v_reduce_sum(const v_f32x4& a) {
    // vpadd (pairwise v_add) — two steps reduces from 4 to 1
    float32x2_t sum2 = vadd_f32(vget_low_f32(a.val), vget_high_f32(a.val));
    sum2 = vpadd_f32(sum2, sum2);
    return vget_lane_f32(sum2, 0);
}

// ============================================================
// v_f32x8 operations (emulated with two 128-bit lanes)
// ============================================================
inline v_f32x8 v_load_f32x8(const float* p) {
    return v_f32x8(vld1q_f32(p), vld1q_f32(p + 4));
}
inline void v_store(float* p, const v_f32x8& a) {
    vst1q_f32(p, a.lo);
    vst1q_f32(p + 4, a.hi);
}
inline v_f32x8 v_set1_f32x8(float s) {
    auto v = vdupq_n_f32(s);
    return v_f32x8(v, v);
}
inline v_f32x8 v_zero_f32x8() {
    auto z = vdupq_n_f32(0.0f);
    return v_f32x8(z, z);
}

inline v_f32x8 v_add(const v_f32x8& a, const v_f32x8& b)
    { return v_f32x8(vaddq_f32(a.lo, b.lo), vaddq_f32(a.hi, b.hi)); }
inline v_f32x8 v_sub(const v_f32x8& a, const v_f32x8& b)
    { return v_f32x8(vsubq_f32(a.lo, b.lo), vsubq_f32(a.hi, b.hi)); }
inline v_f32x8 v_mul(const v_f32x8& a, const v_f32x8& b)
    { return v_f32x8(vmulq_f32(a.lo, b.lo), vmulq_f32(a.hi, b.hi)); }
inline v_f32x8 v_fmadd(const v_f32x8& a, const v_f32x8& b, const v_f32x8& c)
    { return v_f32x8(vfmaq_f32(c.lo, a.lo, b.lo), vfmaq_f32(c.hi, a.hi, b.hi)); }

inline v_f32x8 v_min(const v_f32x8& a, const v_f32x8& b)
    { return v_f32x8(vminq_f32(a.lo, b.lo), vminq_f32(a.hi, b.hi)); }
inline v_f32x8 v_max(const v_f32x8& a, const v_f32x8& b)
    { return v_f32x8(vmaxq_f32(a.lo, b.lo), vmaxq_f32(a.hi, b.hi)); }
inline v_f32x8 v_abs(const v_f32x8& a)
    { return v_f32x8(vabsq_f32(a.lo), vabsq_f32(a.hi)); }
inline v_f32x8 v_neg(const v_f32x8& a)
    { return v_f32x8(vnegq_f32(a.lo), vnegq_f32(a.hi)); }

// Division for v_f32x8 (emulated via v_f32x4)
inline v_f32x8 v_div(const v_f32x8& a, const v_f32x8& b) {
    v_f32x4 lo_ret = v_div(v_f32x4(a.lo), v_f32x4(b.lo));
    v_f32x4 hi_ret = v_div(v_f32x4(a.hi), v_f32x4(b.hi));
    return v_f32x8(lo_ret.val, hi_ret.val);
}

// Reciprocal for v_f32x8 (emulated via v_f32x4)
inline v_f32x8 v_rcp(const v_f32x8& a) {
    v_f32x4 lo_ret = v_rcp(v_f32x4(a.lo));
    v_f32x4 hi_ret = v_rcp(v_f32x4(a.hi));
    return v_f32x8(lo_ret.val, hi_ret.val);
}

// Compare operations for v_f32x8 (emulated)
inline v_f32x8 v_cmplt(const v_f32x8& a, const v_f32x8& b) {
    return v_f32x8(vreinterpretq_f32_u32(vcltq_f32(a.lo, b.lo)),
                    vreinterpretq_f32_u32(vcltq_f32(a.hi, b.hi)));
}
inline v_f32x8 v_cmpgt(const v_f32x8& a, const v_f32x8& b) {
    return v_f32x8(vreinterpretq_f32_u32(vcgtq_f32(a.lo, b.lo)),
                    vreinterpretq_f32_u32(vcgtq_f32(a.hi, b.hi)));
}

// Bitwise operations for v_f32x8 (emulated)
inline v_f32x8 v_and(const v_f32x8& a, const v_f32x8& b) {
    return v_f32x8(vreinterpretq_f32_u32(
                        vandq_u32(vreinterpretq_u32_f32(a.lo), vreinterpretq_u32_f32(b.lo))),
                    vreinterpretq_f32_u32(
                        vandq_u32(vreinterpretq_u32_f32(a.hi), vreinterpretq_u32_f32(b.hi))));
}
inline v_f32x8 v_or(const v_f32x8& a, const v_f32x8& b) {
    return v_f32x8(vreinterpretq_f32_u32(
                        vorrq_u32(vreinterpretq_u32_f32(a.lo), vreinterpretq_u32_f32(b.lo))),
                    vreinterpretq_f32_u32(
                        vorrq_u32(vreinterpretq_u32_f32(a.hi), vreinterpretq_u32_f32(b.hi))));
}

inline v_f32x8 v_sqrt(const v_f32x8& a) {
    v_f32x4 lo(a.lo), hi(a.hi);
    return v_f32x8(v_sqrt(lo).val, v_sqrt(hi).val);
}

// Transcendental math functions for v_f32x8 (lane-wise via v_f32x4 wrappers)
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
    return v_reduce_sum(v_f32x4(a.lo))
         + v_reduce_sum(v_f32x4(a.hi));
}

// ============================================================
// FP16 support — always available on AArch64
// ============================================================

// ============================================================
// v_f16x8 — 128-bit float16 vector, backed by float16x8_t
// ============================================================
struct v_f16x8 {
    float16x8_t val;

    v_f16x8() = default;
    explicit v_f16x8(float16x8_t v) : val(v) {}
    explicit v_f16x8(float s) : val(vdupq_n_f16(static_cast<float16_t>(s))) {}
    v_f16x8(uint16_t v0, uint16_t v1, uint16_t v2, uint16_t v3,
             uint16_t v4, uint16_t v5, uint16_t v6, uint16_t v7) {
        uint16_t tmp[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
        val = vreinterpretq_f16_u16(vld1q_u16(tmp));
    }
};

// ============================================================
// v_f16x8 operations
// ============================================================
inline v_f16x8 v_load_f16x8(const uint16_t* p) {
    return v_f16x8(vreinterpretq_f16_u16(vld1q_u16(p)));
}
inline void v_store(uint16_t* p, const v_f16x8& a) {
    vst1q_u16(p, vreinterpretq_u16_f16(a.val));
}
inline v_f16x8 v_set1_f16x8(float s) {
    return v_f16x8(vdupq_n_f16(static_cast<float16_t>(s)));
}
inline v_f16x8 v_zero_f16x8() {
    return v_f16x8(vdupq_n_f16(static_cast<float16_t>(0.0f)));
}

// FP16 <-> FP32 conversion
inline v_f32x8 v_cvt_f16_to_f32(const v_f16x8& a) {
    // vcvt_f32_f16 on float16x8_t → two float32x4_t results
    float32x4_t lo = vcvt_f32_f16(vget_low_f16(a.val));
    float32x4_t hi = vcvt_high_f32_f16(lo, a.val);  // overwrites lo, computes from upper half
    // Actually vcvt_high_f32_f16 takes two inputs and returns different format:
    //   vcvt_high_f32_f16(float32x4_t, float16x8_t) → float32x4_t (upper 4 lanes)
    // So the correct usage is:
    //   lo = vcvt_f32_f16(vget_low_f16(a.val));
    //   hi = vcvt_high_f32_f16(lo, a.val);  // but this clobbers...
    // The ARM convention: vcvt_high_f32_f16 returns a new float32x4_t for the upper half
    // and takes a "passthru" arg that it ignores. Better to be explicit:

    // Correct approach for NEON f16x8 → f32x8:
    float32x4_t f32_lo = vcvt_f32_f16(vget_low_f16(a.val));
    float32x4_t f32_hi = vcvt_f32_f16(vget_high_f16(a.val));

    return v_f32x8(f32_lo, f32_hi);
}
inline v_f16x8 v_cvt_f32_to_f16(const v_f32x8& a) {
    float16x4_t lo = vcvt_f16_f32(a.lo);
    float16x4_t hi = vcvt_f16_f32(a.hi);
    return v_f16x8(vcombine_f16(lo, hi));
}

// Arithmetic
inline v_f16x8 v_add(const v_f16x8& a, const v_f16x8& b) {
    return v_f16x8(vaddq_f16(a.val, b.val));
}
inline v_f16x8 v_sub(const v_f16x8& a, const v_f16x8& b) {
    return v_f16x8(vsubq_f16(a.val, b.val));
}
inline v_f16x8 v_mul(const v_f16x8& a, const v_f16x8& b) {
    return v_f16x8(vmulq_f16(a.val, b.val));
}
inline v_f16x8 v_div(const v_f16x8& a, const v_f16x8& b) {
    return v_f16x8(vdivq_f16(a.val, b.val));
}
inline v_f16x8 v_fmadd(const v_f16x8& a, const v_f16x8& b, const v_f16x8& c) {
    return v_f16x8(vfmaq_f16(c.val, a.val, b.val));
}

inline v_f16x8 v_min(const v_f16x8& a, const v_f16x8& b) {
    return v_f16x8(vminq_f16(a.val, b.val));
}
inline v_f16x8 v_max(const v_f16x8& a, const v_f16x8& b) {
    return v_f16x8(vmaxq_f16(a.val, b.val));
}
inline v_f16x8 v_abs(const v_f16x8& a) {
    return v_f16x8(vabsq_f16(a.val));
}
inline v_f16x8 v_neg(const v_f16x8& a) {
    return v_f16x8(vnegq_f16(a.val));
}

// Comparison
inline v_f16x8 v_cmplt(const v_f16x8& a, const v_f16x8& b) {
    return v_f16x8(vreinterpretq_f16_u16(vcltq_f16(a.val, b.val)));
}
inline v_f16x8 v_cmpgt(const v_f16x8& a, const v_f16x8& b) {
    return v_f16x8(vreinterpretq_f16_u16(vcgtq_f16(a.val, b.val)));
}

inline v_f16x8 v_sqrt(const v_f16x8& a) {
    return v_f16x8(vsqrtq_f16(a.val));
}

inline float v_reduce_sum(const v_f16x8& a) {
    // Horizontal v_add across all 8 lanes
    // Approach: reduce low and high halves, accumulate as float32
    float sum = static_cast<float>(vaddvq_f16(a.val));
    return sum;
}

// ============================================================
// v_f16x8 transcendental math (convert→compute→convert)
//
// fp16 has too little range/precision for meaningful polynomial
// approximations. Instead, widen to fp32, call the Cephes math
// functions, and narrow back. The conversion overhead is negligible
// compared to the cost of computing the transcendental.
// ============================================================
inline v_f16x8 v_exp(const v_f16x8& a) {
    v_f32x8 fa = v_cvt_f16_to_f32(a);
    v_f32x8 fr = v_f32x8(exp_ps(fa.lo), exp_ps(fa.hi));
    return v_cvt_f32_to_f16(fr);
}
inline v_f16x8 v_log(const v_f16x8& a) {
    v_f32x8 fa = v_cvt_f16_to_f32(a);
    v_f32x8 fr = v_f32x8(log_ps(fa.lo), log_ps(fa.hi));
    return v_cvt_f32_to_f16(fr);
}
inline v_f16x8 v_sin(const v_f16x8& a) {
    v_f32x8 fa = v_cvt_f16_to_f32(a);
    v_f32x8 fr = v_f32x8(sin_ps(fa.lo), sin_ps(fa.hi));
    return v_cvt_f32_to_f16(fr);
}
inline v_f16x8 v_cos(const v_f16x8& a) {
    v_f32x8 fa = v_cvt_f16_to_f32(a);
    v_f32x8 fr = v_f32x8(cos_ps(fa.lo), cos_ps(fa.hi));
    return v_cvt_f32_to_f16(fr);
}
inline v_f16x8 v_tan(const v_f16x8& a) {
    v_f32x8 fa = v_cvt_f16_to_f32(a);
    v_f32x8 fr = v_f32x8(tan_ps(fa.lo), tan_ps(fa.hi));
    return v_cvt_f32_to_f16(fr);
}
inline v_f16x8 v_tanh(const v_f16x8& a) {
    v_f32x8 fa = v_cvt_f16_to_f32(a);
    v_f32x8 fr = v_f32x8(tanh_ps(fa.lo), tanh_ps(fa.hi));
    return v_cvt_f32_to_f16(fr);
}

} // namespace neon
} // namespace arch
} // namespace simd
} // namespace nnops
