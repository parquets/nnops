#pragma once
/// @file neon.hpp
/// @brief ARM NEON backend for 128-bit float32 vector operations (v_fp32x4).
///
/// NEON (ASIMD) is mandatory on AArch64. Provides native 128-bit float operations.
/// For 256-bit v_fp32x8, we emulate with two 128-bit NEON registers since NEON
/// does not natively have 256-bit vectors.
///
/// Reference: OpenCV intrin_neon.hpp, onnxruntime MLAS NEON kernels.

#if !defined(NNOPS_ARCH_AARCH64)
  #error "neon.hpp requires AArch64 architecture"
#endif

#include <arm_neon.h>
#include <cmath>  // for std::abs in vec_abs emulation

namespace nnops {
namespace simd {
namespace arch {
namespace neon {

// ============================================================
// v_fp32x4 — 128-bit float vector, backed by float32x4_t
// ============================================================
struct v_fp32x4 {
    float32x4_t val;

    v_fp32x4() = default;
    explicit v_fp32x4(float32x4_t v) : val(v) {}
    explicit v_fp32x4(float s) : val(vdupq_n_f32(s)) {}
    v_fp32x4(float v0, float v1, float v2, float v3) {
        float tmp[4] = {v0, v1, v2, v3};
        val = vld1q_f32(tmp);
    }

    float operator[](int i) const { return vgetq_lane_f32(val, i); }
};

// ============================================================
// v_fp32x8 — 256-bit, emulated with two float32x4_t
// ============================================================
struct v_fp32x8 {
    float32x4_t lo, hi;

    v_fp32x8() = default;
    explicit v_fp32x8(float s) : lo(vdupq_n_f32(s)), hi(vdupq_n_f32(s)) {}
    v_fp32x8(float32x4_t lo_, float32x4_t hi_) : lo(lo_), hi(hi_) {}
    v_fp32x8(float v0, float v1, float v2, float v3,
             float v4, float v5, float v6, float v7) {
        float tmp_lo[4] = {v0, v1, v2, v3};
        float tmp_hi[4] = {v4, v5, v6, v7};
        lo = vld1q_f32(tmp_lo);
        hi = vld1q_f32(tmp_hi);
    }
};

// ============================================================
// v_fp32x4 operations
// ============================================================
inline v_fp32x4 load_fp32x4(const float* p) {
    return v_fp32x4(vld1q_f32(p));
}
inline void store(float* p, const v_fp32x4& a) {
    vst1q_f32(p, a.val);
}
inline v_fp32x4 set1_fp32x4(float s)  { return v_fp32x4(vdupq_n_f32(s)); }
inline v_fp32x4 zero_fp32x4() {
    return v_fp32x4(vdupq_n_f32(0.0f));
}

inline v_fp32x4 add(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(vaddq_f32(a.val, b.val));
}
inline v_fp32x4 sub(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(vsubq_f32(a.val, b.val));
}
inline v_fp32x4 mul(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(vmulq_f32(a.val, b.val));
}

// Division: NEON doesn't have native float div, use reciprocal + NR step.
// For simplicity use vrecpe + vrecps (Newton-Raphson) or just the scalar fallback.
// Actually, GCC/Clang provide __builtin_vdivq_f32 (compiler synthesizes).
inline v_fp32x4 div(const v_fp32x4& a, const v_fp32x4& b) {
    // Use reciprocal approximation with Newton-Raphson refinement
    float32x4_t recip = vrecpeq_f32(b.val);          // initial estimate ~1/b
    recip = vmulq_f32(vrecpsq_f32(b.val, recip), recip);  // Newton-Raphson step
    recip = vmulq_f32(vrecpsq_f32(b.val, recip), recip);  // second NR step for full precision
    return v_fp32x4(vmulq_f32(a.val, recip));
}

inline v_fp32x4 fmadd(const v_fp32x4& a, const v_fp32x4& b, const v_fp32x4& c) {
    // vfmaq_f32: c + a * b (results stored in first arg convention)
    return v_fp32x4(vfmaq_f32(c.val, a.val, b.val));
}

inline v_fp32x4 min(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(vminq_f32(a.val, b.val));
}
inline v_fp32x4 max(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(vmaxq_f32(a.val, b.val));
}
inline v_fp32x4 abs(const v_fp32x4& a) {
    return v_fp32x4(vabsq_f32(a.val));
}
inline v_fp32x4 neg(const v_fp32x4& a) {
    return v_fp32x4(vnegq_f32(a.val));
}

inline v_fp32x4 cmplt(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(vreinterpretq_f32_u32(vcltq_f32(a.val, b.val)));
}
inline v_fp32x4 cmple(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(vreinterpretq_f32_u32(vcleq_f32(a.val, b.val)));
}
inline v_fp32x4 cmpgt(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(vreinterpretq_f32_u32(vcgtq_f32(a.val, b.val)));
}
inline v_fp32x4 cmpge(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(vreinterpretq_f32_u32(vcgeq_f32(a.val, b.val)));
}
inline v_fp32x4 ceq(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(vreinterpretq_f32_u32(vceqq_f32(a.val, b.val)));
}

inline v_fp32x4 and_(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(vreinterpretq_f32_u32(
        vandq_u32(vreinterpretq_u32_f32(a.val), vreinterpretq_u32_f32(b.val))));
}
inline v_fp32x4 or_(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(vreinterpretq_f32_u32(
        vorrq_u32(vreinterpretq_u32_f32(a.val), vreinterpretq_u32_f32(b.val))));
}

inline v_fp32x4 sqrt(const v_fp32x4& a) {
    // NEON: vsqrt is not always available. Use vrsqrte + NR + mul for portability.
    float32x4_t est = vrsqrteq_f32(a.val);          // 1/sqrt(a) estimate
    est = vmulq_f32(vrsqrtsq_f32(a.val, vmulq_f32(est, est)), est);  // NR step
    est = vmulq_f32(vrsqrtsq_f32(a.val, vmulq_f32(est, est)), est);  // second NR
    return v_fp32x4(vmulq_f32(a.val, est));          // a * 1/sqrt(a) = sqrt(a)
}

// Reciprocal with Newton-Raphson refinement
inline v_fp32x4 rcp(const v_fp32x4& a) {
    float32x4_t est = vrecpeq_f32(a.val);
    est = vmulq_f32(vrecpsq_f32(a.val, est), est);
    return v_fp32x4(vmulq_f32(vrecpsq_f32(a.val, est), est));
}

inline float reduce_sum(const v_fp32x4& a) {
    // vpadd (pairwise add) — two steps reduces from 4 to 1
    float32x2_t sum2 = vadd_f32(vget_low_f32(a.val), vget_high_f32(a.val));
    sum2 = vpadd_f32(sum2, sum2);
    return vget_lane_f32(sum2, 0);
}

// ============================================================
// v_fp32x8 operations (emulated with two 128-bit lanes)
// ============================================================
inline v_fp32x8 load_fp32x8(const float* p) {
    return v_fp32x8(vld1q_f32(p), vld1q_f32(p + 4));
}
inline void store(float* p, const v_fp32x8& a) {
    vst1q_f32(p, a.lo);
    vst1q_f32(p + 4, a.hi);
}
inline v_fp32x8 set1_fp32x8(float s) {
    auto v = vdupq_n_f32(s);
    return v_fp32x8(v, v);
}
inline v_fp32x8 zero_fp32x8() {
    auto z = vdupq_n_f32(0.0f);
    return v_fp32x8(z, z);
}

inline v_fp32x8 add(const v_fp32x8& a, const v_fp32x8& b)
    { return v_fp32x8(vaddq_f32(a.lo, b.lo), vaddq_f32(a.hi, b.hi)); }
inline v_fp32x8 sub(const v_fp32x8& a, const v_fp32x8& b)
    { return v_fp32x8(vsubq_f32(a.lo, b.lo), vsubq_f32(a.hi, b.hi)); }
inline v_fp32x8 mul(const v_fp32x8& a, const v_fp32x8& b)
    { return v_fp32x8(vmulq_f32(a.lo, b.lo), vmulq_f32(a.hi, b.hi)); }
inline v_fp32x8 fmadd(const v_fp32x8& a, const v_fp32x8& b, const v_fp32x8& c)
    { return v_fp32x8(vfmaq_f32(c.lo, a.lo, b.lo), vfmaq_f32(c.hi, a.hi, b.hi)); }

inline v_fp32x8 min(const v_fp32x8& a, const v_fp32x8& b)
    { return v_fp32x8(vminq_f32(a.lo, b.lo), vminq_f32(a.hi, b.hi)); }
inline v_fp32x8 max(const v_fp32x8& a, const v_fp32x8& b)
    { return v_fp32x8(vmaxq_f32(a.lo, b.lo), vmaxq_f32(a.hi, b.hi)); }

inline v_fp32x8 sqrt(const v_fp32x8& a) {
    v_fp32x4 lo(a.lo), hi(a.hi);
    return v_fp32x8(sqrt(lo).val, sqrt(hi).val);
}

inline float reduce_sum(const v_fp32x8& a) {
    return reduce_sum(v_fp32x4(a.lo))
         + reduce_sum(v_fp32x4(a.hi));
}

// ============================================================
// FP16 support (ARMv8.2-A+ with __ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
//
// On AArch64 with the FP16 vector arithmetic feature, we get native
// float16x4_t (64-bit) and float16x8_t (128-bit) operations.
// When the feature is not available at compile time, vec_f16x4.hpp
// and vec_f16x8.hpp fall back to the scalar emulation.
// ============================================================
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

// ============================================================
// v_fp16x4 — 64-bit float16 vector, backed by float16x4_t
// ============================================================
struct v_fp16x4 {
    float16x4_t val;

    v_fp16x4() = default;
    explicit v_fp16x4(float16x4_t v) : val(v) {}
    explicit v_fp16x4(float s) : val(vdup_n_f16(static_cast<float16_t>(s))) {}
    v_fp16x4(uint16_t v0, uint16_t v1, uint16_t v2, uint16_t v3) {
        uint16_t tmp[4] = {v0, v1, v2, v3};
        val = vreinterpret_f16_u16(vld1_u16(tmp));
    }
};

// ============================================================
// v_fp16x8 — 128-bit float16 vector, backed by float16x8_t
// ============================================================
struct v_fp16x8 {
    float16x8_t val;

    v_fp16x8() = default;
    explicit v_fp16x8(float16x8_t v) : val(v) {}
    explicit v_fp16x8(float s) : val(vdupq_n_f16(static_cast<float16_t>(s))) {}
    v_fp16x8(uint16_t v0, uint16_t v1, uint16_t v2, uint16_t v3,
             uint16_t v4, uint16_t v5, uint16_t v6, uint16_t v7) {
        uint16_t tmp[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
        val = vreinterpretq_f16_u16(vld1q_u16(tmp));
    }
};

// ============================================================
// v_fp16x4 operations
// ============================================================
inline v_fp16x4 load_fp16x4(const uint16_t* p) {
    return v_fp16x4(vreinterpret_f16_u16(vld1_u16(p)));
}
inline void store(uint16_t* p, const v_fp16x4& a) {
    vst1_u16(p, vreinterpret_u16_f16(a.val));
}
inline v_fp16x4 set1_fp16x4(float s) {
    return v_fp16x4(vdup_n_f16(static_cast<float16_t>(s)));
}
inline v_fp16x4 zero_fp16x4() {
    return v_fp16x4(vdup_n_f16(static_cast<float16_t>(0.0f)));
}

// FP16 <-> FP32 conversion
inline v_fp32x4 cvt_f16_to_f32(const v_fp16x4& a) {
    return v_fp32x4(vcvt_f32_f16(a.val));
}
inline v_fp16x4 cvt_f32_to_f16(const v_fp32x4& a) {
    return v_fp16x4(vcvt_f16_f32(a.val));
}

// Arithmetic
inline v_fp16x4 add(const v_fp16x4& a, const v_fp16x4& b) {
    return v_fp16x4(vadd_f16(a.val, b.val));
}
inline v_fp16x4 sub(const v_fp16x4& a, const v_fp16x4& b) {
    return v_fp16x4(vsub_f16(a.val, b.val));
}
inline v_fp16x4 mul(const v_fp16x4& a, const v_fp16x4& b) {
    return v_fp16x4(vmul_f16(a.val, b.val));
}
inline v_fp16x4 div(const v_fp16x4& a, const v_fp16x4& b) {
    return v_fp16x4(vdiv_f16(a.val, b.val));
}
inline v_fp16x4 fmadd(const v_fp16x4& a, const v_fp16x4& b, const v_fp16x4& c) {
    return v_fp16x4(vfma_f16(c.val, a.val, b.val));
}

inline v_fp16x4 min(const v_fp16x4& a, const v_fp16x4& b) {
    return v_fp16x4(vmin_f16(a.val, b.val));
}
inline v_fp16x4 max(const v_fp16x4& a, const v_fp16x4& b) {
    return v_fp16x4(vmax_f16(a.val, b.val));
}
inline v_fp16x4 abs(const v_fp16x4& a) {
    return v_fp16x4(vabs_f16(a.val));
}
inline v_fp16x4 neg(const v_fp16x4& a) {
    return v_fp16x4(vneg_f16(a.val));
}

// Comparison
inline v_fp16x4 cmplt(const v_fp16x4& a, const v_fp16x4& b) {
    return v_fp16x4(vreinterpret_f16_u16(vclt_f16(a.val, b.val)));
}
inline v_fp16x4 cmpgt(const v_fp16x4& a, const v_fp16x4& b) {
    return v_fp16x4(vreinterpret_f16_u16(vcgt_f16(a.val, b.val)));
}

inline v_fp16x4 sqrt(const v_fp16x4& a) {
    return v_fp16x4(vsqrt_f16(a.val));
}

inline float reduce_sum(const v_fp16x4& a) {
    // vaddv_f16: horizontal add across vector → float16_t
    return static_cast<float>(vaddv_f16(a.val));
}

// ============================================================
// v_fp16x8 operations
// ============================================================
inline v_fp16x8 load_fp16x8(const uint16_t* p) {
    return v_fp16x8(vreinterpretq_f16_u16(vld1q_u16(p)));
}
inline void store(uint16_t* p, const v_fp16x8& a) {
    vst1q_u16(p, vreinterpretq_u16_f16(a.val));
}
inline v_fp16x8 set1_fp16x8(float s) {
    return v_fp16x8(vdupq_n_f16(static_cast<float16_t>(s)));
}
inline v_fp16x8 zero_fp16x8() {
    return v_fp16x8(vdupq_n_f16(static_cast<float16_t>(0.0f)));
}

// FP16 <-> FP32 conversion
inline v_fp32x8 cvt_f16_to_f32(const v_fp16x8& a) {
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

    return v_fp32x8(f32_lo, f32_hi);
}
inline v_fp16x8 cvt_f32_to_f16(const v_fp32x8& a) {
    float16x4_t lo = vcvt_f16_f32(a.lo);
    float16x4_t hi = vcvt_f16_f32(a.hi);
    return v_fp16x8(vcombine_f16(lo, hi));
}

// Arithmetic
inline v_fp16x8 add(const v_fp16x8& a, const v_fp16x8& b) {
    return v_fp16x8(vaddq_f16(a.val, b.val));
}
inline v_fp16x8 sub(const v_fp16x8& a, const v_fp16x8& b) {
    return v_fp16x8(vsubq_f16(a.val, b.val));
}
inline v_fp16x8 mul(const v_fp16x8& a, const v_fp16x8& b) {
    return v_fp16x8(vmulq_f16(a.val, b.val));
}
inline v_fp16x8 div(const v_fp16x8& a, const v_fp16x8& b) {
    return v_fp16x8(vdivq_f16(a.val, b.val));
}
inline v_fp16x8 fmadd(const v_fp16x8& a, const v_fp16x8& b, const v_fp16x8& c) {
    return v_fp16x8(vfmaq_f16(c.val, a.val, b.val));
}

inline v_fp16x8 min(const v_fp16x8& a, const v_fp16x8& b) {
    return v_fp16x8(vminq_f16(a.val, b.val));
}
inline v_fp16x8 max(const v_fp16x8& a, const v_fp16x8& b) {
    return v_fp16x8(vmaxq_f16(a.val, b.val));
}
inline v_fp16x8 abs(const v_fp16x8& a) {
    return v_fp16x8(vabsq_f16(a.val));
}
inline v_fp16x8 neg(const v_fp16x8& a) {
    return v_fp16x8(vnegq_f16(a.val));
}

// Comparison
inline v_fp16x8 cmplt(const v_fp16x8& a, const v_fp16x8& b) {
    return v_fp16x8(vreinterpretq_f16_u16(vcltq_f16(a.val, b.val)));
}
inline v_fp16x8 cmpgt(const v_fp16x8& a, const v_fp16x8& b) {
    return v_fp16x8(vreinterpretq_f16_u16(vcgtq_f16(a.val, b.val)));
}

inline v_fp16x8 sqrt(const v_fp16x8& a) {
    return v_fp16x8(vsqrtq_f16(a.val));
}

inline float reduce_sum(const v_fp16x8& a) {
    // Horizontal add across all 8 lanes
    // Approach: reduce low and high halves, accumulate as float32
    float sum = static_cast<float>(vaddvq_f16(a.val));
    return sum;
}

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace neon
} // namespace arch
} // namespace simd
} // namespace nnops
