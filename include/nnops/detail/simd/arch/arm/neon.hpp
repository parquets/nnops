#pragma once
/// @file neon.hpp
/// @brief ARM NEON backend for 128-bit float32 vector operations (VecF32x4).
///
/// NEON (ASIMD) is mandatory on AArch64. Provides native 128-bit float operations.
/// For 256-bit VecF32x8, we emulate with two 128-bit NEON registers since NEON
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
// VecF32x4 — 128-bit float vector, backed by float32x4_t
// ============================================================
struct VecF32x4 {
    float32x4_t val;

    VecF32x4() = default;
    explicit VecF32x4(float32x4_t v) : val(v) {}
    explicit VecF32x4(float s) : val(vdupq_n_f32(s)) {}
    VecF32x4(float v0, float v1, float v2, float v3) {
        float tmp[4] = {v0, v1, v2, v3};
        val = vld1q_f32(tmp);
    }

    float operator[](int i) const { return vgetq_lane_f32(val, i); }
};

// ============================================================
// VecF32x8 — 256-bit, emulated with two float32x4_t
// ============================================================
struct VecF32x8 {
    float32x4_t lo, hi;

    VecF32x8() = default;
    explicit VecF32x8(float s) : lo(vdupq_n_f32(s)), hi(vdupq_n_f32(s)) {}
    VecF32x8(float32x4_t lo_, float32x4_t hi_) : lo(lo_), hi(hi_) {}
    VecF32x8(float v0, float v1, float v2, float v3,
             float v4, float v5, float v6, float v7) {
        float tmp_lo[4] = {v0, v1, v2, v3};
        float tmp_hi[4] = {v4, v5, v6, v7};
        lo = vld1q_f32(tmp_lo);
        hi = vld1q_f32(tmp_hi);
    }
};

// ============================================================
// VecF32x4 operations
// ============================================================
inline VecF32x4 vec_load_f32x4(const float* p) {
    return VecF32x4(vld1q_f32(p));
}
inline void vec_store_f32x4(float* p, const VecF32x4& a) {
    vst1q_f32(p, a.val);
}
inline VecF32x4 vec_set1_f32x4(float s)  { return VecF32x4(vdupq_n_f32(s)); }
inline VecF32x4 vec_zero_f32x4() {
    return VecF32x4(vdupq_n_f32(0.0f));
}

inline VecF32x4 vec_add_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(vaddq_f32(a.val, b.val));
}
inline VecF32x4 vec_sub_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(vsubq_f32(a.val, b.val));
}
inline VecF32x4 vec_mul_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(vmulq_f32(a.val, b.val));
}

// Division: NEON doesn't have native float div, use reciprocal + NR step.
// For simplicity use vrecpe + vrecps (Newton-Raphson) or just the scalar fallback.
// Actually, GCC/Clang provide __builtin_vdivq_f32 (compiler synthesizes).
inline VecF32x4 vec_div_f32x4(const VecF32x4& a, const VecF32x4& b) {
    // Use reciprocal approximation with Newton-Raphson refinement
    float32x4_t recip = vrecpeq_f32(b.val);          // initial estimate ~1/b
    recip = vmulq_f32(vrecpsq_f32(b.val, recip), recip);  // Newton-Raphson step
    recip = vmulq_f32(vrecpsq_f32(b.val, recip), recip);  // second NR step for full precision
    return VecF32x4(vmulq_f32(a.val, recip));
}

inline VecF32x4 vec_fmadd_f32x4(const VecF32x4& a, const VecF32x4& b, const VecF32x4& c) {
    // vfmaq_f32: c + a * b (results stored in first arg convention)
    return VecF32x4(vfmaq_f32(c.val, a.val, b.val));
}

inline VecF32x4 vec_min_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(vminq_f32(a.val, b.val));
}
inline VecF32x4 vec_max_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(vmaxq_f32(a.val, b.val));
}
inline VecF32x4 vec_abs_f32x4(const VecF32x4& a) {
    return VecF32x4(vabsq_f32(a.val));
}
inline VecF32x4 vec_neg_f32x4(const VecF32x4& a) {
    return VecF32x4(vnegq_f32(a.val));
}

inline VecF32x4 vec_cmplt_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(vreinterpretq_f32_u32(vcltq_f32(a.val, b.val)));
}
inline VecF32x4 vec_cmple_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(vreinterpretq_f32_u32(vcleq_f32(a.val, b.val)));
}
inline VecF32x4 vec_cmpgt_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(vreinterpretq_f32_u32(vcgtq_f32(a.val, b.val)));
}
inline VecF32x4 vec_cmpge_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(vreinterpretq_f32_u32(vcgeq_f32(a.val, b.val)));
}
inline VecF32x4 vec_ceq_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(vreinterpretq_f32_u32(vceqq_f32(a.val, b.val)));
}

inline VecF32x4 vec_and_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(vreinterpretq_f32_u32(
        vandq_u32(vreinterpretq_u32_f32(a.val), vreinterpretq_u32_f32(b.val))));
}
inline VecF32x4 vec_or_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(vreinterpretq_f32_u32(
        vorrq_u32(vreinterpretq_u32_f32(a.val), vreinterpretq_u32_f32(b.val))));
}

inline VecF32x4 vec_sqrt_f32x4(const VecF32x4& a) {
    // NEON: vsqrt is not always available. Use vrsqrte + NR + mul for portability.
    float32x4_t est = vrsqrteq_f32(a.val);          // 1/sqrt(a) estimate
    est = vmulq_f32(vrsqrtsq_f32(a.val, vmulq_f32(est, est)), est);  // NR step
    est = vmulq_f32(vrsqrtsq_f32(a.val, vmulq_f32(est, est)), est);  // second NR
    return VecF32x4(vmulq_f32(a.val, est));          // a * 1/sqrt(a) = sqrt(a)
}

// Reciprocal with Newton-Raphson refinement
inline VecF32x4 vec_rcp_f32x4(const VecF32x4& a) {
    float32x4_t est = vrecpeq_f32(a.val);
    est = vmulq_f32(vrecpsq_f32(a.val, est), est);
    return VecF32x4(vmulq_f32(vrecpsq_f32(a.val, est), est));
}

inline float vec_reduce_sum_f32x4(const VecF32x4& a) {
    // vpadd (pairwise add) — two steps reduces from 4 to 1
    float32x2_t sum2 = vadd_f32(vget_low_f32(a.val), vget_high_f32(a.val));
    sum2 = vpadd_f32(sum2, sum2);
    return vget_lane_f32(sum2, 0);
}

// ============================================================
// VecF32x8 operations (emulated with two 128-bit lanes)
// ============================================================
inline VecF32x8 vec_load_f32x8(const float* p) {
    return VecF32x8(vld1q_f32(p), vld1q_f32(p + 4));
}
inline void vec_store_f32x8(float* p, const VecF32x8& a) {
    vst1q_f32(p, a.lo);
    vst1q_f32(p + 4, a.hi);
}
inline VecF32x8 vec_set1_f32x8(float s) {
    auto v = vdupq_n_f32(s);
    return VecF32x8(v, v);
}
inline VecF32x8 vec_zero_f32x8() {
    auto z = vdupq_n_f32(0.0f);
    return VecF32x8(z, z);
}

inline VecF32x8 vec_add_f32x8(const VecF32x8& a, const VecF32x8& b)
    { return VecF32x8(vaddq_f32(a.lo, b.lo), vaddq_f32(a.hi, b.hi)); }
inline VecF32x8 vec_sub_f32x8(const VecF32x8& a, const VecF32x8& b)
    { return VecF32x8(vsubq_f32(a.lo, b.lo), vsubq_f32(a.hi, b.hi)); }
inline VecF32x8 vec_mul_f32x8(const VecF32x8& a, const VecF32x8& b)
    { return VecF32x8(vmulq_f32(a.lo, b.lo), vmulq_f32(a.hi, b.hi)); }
inline VecF32x8 vec_fmadd_f32x8(const VecF32x8& a, const VecF32x8& b, const VecF32x8& c)
    { return VecF32x8(vfmaq_f32(c.lo, a.lo, b.lo), vfmaq_f32(c.hi, a.hi, b.hi)); }

inline VecF32x8 vec_min_f32x8(const VecF32x8& a, const VecF32x8& b)
    { return VecF32x8(vminq_f32(a.lo, b.lo), vminq_f32(a.hi, b.hi)); }
inline VecF32x8 vec_max_f32x8(const VecF32x8& a, const VecF32x8& b)
    { return VecF32x8(vmaxq_f32(a.lo, b.lo), vmaxq_f32(a.hi, b.hi)); }

inline VecF32x8 vec_sqrt_f32x8(const VecF32x8& a) {
    VecF32x4 lo(a.lo), hi(a.hi);
    return VecF32x8(vec_sqrt_f32x4(lo).val, vec_sqrt_f32x4(hi).val);
}

inline float vec_reduce_sum_f32x8(const VecF32x8& a) {
    return vec_reduce_sum_f32x4(VecF32x4(a.lo))
         + vec_reduce_sum_f32x4(VecF32x4(a.hi));
}

} // namespace neon
} // namespace arch
} // namespace simd
} // namespace nnops
