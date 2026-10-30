#pragma once
/// @file neon.hpp
/// @brief ARM NEON backend for 128-bit f32 (v_f32x4), f16 (v_f16x8) and int8 vectors.
///
/// NEON (ASIMD) is mandatory on AArch64; there is no 256-bit register, so
/// v_f32x8 is emulated as two float32x4_t halves driven together. v_f16x8 is
/// native, but its transcendentals widen to f32, run the Cephes polynomials in
/// neon_mathfunc.hpp, and narrow back -- fp16 cannot carry the polynomial.
///
/// Numerical contract: the f32 v_div and v_sqrt are the correctly-rounded
/// AArch64 FDIV/FSQRT, so they match the scalar reference and IEEE-754
/// (x/0 -> ±inf, 0/0 -> NaN). Do not swap in vrecpeq/vrsqrteq +
/// Newton-Raphson: they are not exact, and the rsqrt form returns NaN for
/// sqrt(0) (0 * inf). The f32 v_rcp is 1/a for the same reason (x86 keeps a
/// 12-bit _mm_rcp_ps, so NEON is the more accurate side here).
///
/// v_transpose_8x8 (f32) is four 4x4 transposes on the lo/hi halves, with the
/// bottom-left quadrant snapshotted because it and the top-right are each
/// other's destinations. The f16 variant interleaves with widening granularity
/// -- ZIP f16 -> u32 -> u64, as on x86 -- since an all-f16 ZIP produces a
/// bit-reversal, not a transpose. v_deinterleave_* use native LD2; the
/// v_load_even_* / v_load_stride2_* families are equivalent aliases.
///
/// Reference: OpenCV intrin_neon.hpp, onnxruntime MLAS NEON kernels.

#if !defined(NNOPS_ARCH_AARCH64)
  #error "neon.hpp requires AArch64 architecture"
#endif

#include <arm_neon.h>
#include <cmath>
#include "neon_mathfunc.hpp"

namespace nnops {
namespace simd {
namespace arch {
namespace neon {

struct v_f32x4 {
    float32x4_t val;

    v_f32x4() = default;
    explicit v_f32x4(float32x4_t v) : val(v) {}
    explicit v_f32x4(float s) : val(vdupq_n_f32(s)) {}
    v_f32x4(float v0, float v1, float v2, float v3) {
        float tmp[4] = {v0, v1, v2, v3};
        val = vld1q_f32(tmp);
    }

    float operator[](int i) const {
        // vgetq_lane_f32 needs a compile-time lane index; clang rejects a runtime one.
        switch (i) {
            case 0:  return vgetq_lane_f32(val, 0);
            case 1:  return vgetq_lane_f32(val, 1);
            case 2:  return vgetq_lane_f32(val, 2);
            default: return vgetq_lane_f32(val, 3);
        }
    }
};

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

inline v_f32x4 v_div(const v_f32x4& a, const v_f32x4& b) {
    return v_f32x4(vdivq_f32(a.val, b.val));
}

inline v_f32x4 v_fmadd(const v_f32x4& a, const v_f32x4& b, const v_f32x4& c) {
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
    return v_f32x4(vsqrtq_f32(a.val));
}

inline v_f32x4 v_rcp(const v_f32x4& a) {
    return v_f32x4(vdivq_f32(vdupq_n_f32(1.0f), a.val));
}

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
    float32x2_t sum2 = vadd_f32(vget_low_f32(a.val), vget_high_f32(a.val));
    sum2 = vpadd_f32(sum2, sum2);
    return vget_lane_f32(sum2, 0);
}
inline float v_reduce_max(const v_f32x4& a) {
    float32x2_t max2 = vpmax_f32(vget_low_f32(a.val), vget_high_f32(a.val));
    max2 = vpmax_f32(max2, max2);
    return vget_lane_f32(max2, 0);
}
inline float v_reduce_min(const v_f32x4& a) {
    float32x2_t min2 = vpmin_f32(vget_low_f32(a.val), vget_high_f32(a.val));
    min2 = vpmin_f32(min2, min2);
    return vget_lane_f32(min2, 0);
}

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

inline v_f32x8 v_div(const v_f32x8& a, const v_f32x8& b) {
    v_f32x4 lo_ret = v_div(v_f32x4(a.lo), v_f32x4(b.lo));
    v_f32x4 hi_ret = v_div(v_f32x4(a.hi), v_f32x4(b.hi));
    return v_f32x8(lo_ret.val, hi_ret.val);
}

inline v_f32x8 v_rcp(const v_f32x8& a) {
    v_f32x4 lo_ret = v_rcp(v_f32x4(a.lo));
    v_f32x4 hi_ret = v_rcp(v_f32x4(a.hi));
    return v_f32x8(lo_ret.val, hi_ret.val);
}

inline v_f32x8 v_cmplt(const v_f32x8& a, const v_f32x8& b) {
    return v_f32x8(vreinterpretq_f32_u32(vcltq_f32(a.lo, b.lo)),
                    vreinterpretq_f32_u32(vcltq_f32(a.hi, b.hi)));
}
inline v_f32x8 v_cmpgt(const v_f32x8& a, const v_f32x8& b) {
    return v_f32x8(vreinterpretq_f32_u32(vcgtq_f32(a.lo, b.lo)),
                    vreinterpretq_f32_u32(vcgtq_f32(a.hi, b.hi)));
}

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
inline float v_reduce_max(const v_f32x8& a) {
    float lo = v_reduce_max(v_f32x4(a.lo));
    float hi = v_reduce_max(v_f32x4(a.hi));
    return lo > hi ? lo : hi;
}
inline float v_reduce_min(const v_f32x8& a) {
    float lo = v_reduce_min(v_f32x4(a.lo));
    float hi = v_reduce_min(v_f32x4(a.hi));
    return lo < hi ? lo : hi;
}

// ---------------------------------------------------------------- float16

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

inline v_f32x8 v_cvt_f16_to_f32(const v_f16x8& a) {
    float32x4_t f32_lo = vcvt_f32_f16(vget_low_f16(a.val));
    float32x4_t f32_hi = vcvt_f32_f16(vget_high_f16(a.val));

    return v_f32x8(f32_lo, f32_hi);
}
inline v_f16x8 v_cvt_f32_to_f16(const v_f32x8& a) {
    float16x4_t lo = vcvt_f16_f32(a.lo);
    float16x4_t hi = vcvt_f16_f32(a.hi);
    return v_f16x8(vcombine_f16(lo, hi));
}

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

inline v_f16x8 v_cmplt(const v_f16x8& a, const v_f16x8& b) {
    return v_f16x8(vreinterpretq_f16_u16(vcltq_f16(a.val, b.val)));
}
inline v_f16x8 v_cmpgt(const v_f16x8& a, const v_f16x8& b) {
    return v_f16x8(vreinterpretq_f16_u16(vcgtq_f16(a.val, b.val)));
}

inline v_f16x8 v_sqrt(const v_f16x8& a) {
    return v_f16x8(vsqrtq_f16(a.val));
}
inline v_f16x8 v_rcp(const v_f16x8& a) {
    return v_f16x8(vrecpeq_f16(a.val));
}

inline float v_reduce_sum(const v_f16x8& a) {
    // vaddvq_f16 is GCC-only, so mirror the f32 pairwise reduction.
    float16x4_t sum4 = vpadd_f16(vget_low_f16(a.val), vget_high_f16(a.val));
    sum4 = vpadd_f16(sum4, sum4);
    sum4 = vpadd_f16(sum4, sum4);
    return static_cast<float>(vget_lane_f16(sum4, 0));
}

inline float v_reduce_max(const v_f16x8& a) {
    float16x4_t lo = vget_low_f16(a.val);
    float16x4_t hi = vget_high_f16(a.val);
    float16x4_t max4 = vpmax_f16(lo, hi);
    max4 = vpmax_f16(max4, max4);
    max4 = vpmax_f16(max4, max4);
    return static_cast<float>(vget_lane_f16(max4, 0));
}
inline float v_reduce_min(const v_f16x8& a) {
    float16x4_t lo = vget_low_f16(a.val);
    float16x4_t hi = vget_high_f16(a.val);
    float16x4_t min4 = vpmin_f16(lo, hi);
    min4 = vpmin_f16(min4, min4);
    min4 = vpmin_f16(min4, min4);
    return static_cast<float>(vget_lane_f16(min4, 0));
}

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

// ---------------------------------------------------------------- deinterleave

struct v_f32x4x2_t { v_f32x4 even; v_f32x4 odd; };
struct v_f32x8x2_t { v_f32x8 even; v_f32x8 odd; };
struct v_f16x8x2_t { v_f16x8 even; v_f16x8 odd; };

/// Load 8 f32s: even lanes {a0,a2,a4,a6}, odd {a1,a3,a5,a7}.
inline v_f32x4x2_t v_deinterleave_f32x4(const float* src) {
    float32x4x2_t v = vld2q_f32(src);
    return {v_f32x4(v.val[0]), v_f32x4(v.val[1])};
}

/// Load 16 f32s: two v_f32x8 results, {even} and {odd}.
inline v_f32x8x2_t v_deinterleave_f32x8(const float* src) {
    v_f32x4x2_t lo = v_deinterleave_f32x4(src);
    v_f32x4x2_t hi = v_deinterleave_f32x4(src + 8);
    return {v_f32x8(lo.even.val, hi.even.val),
            v_f32x8(lo.odd.val,  hi.odd.val)};
}

/// Load 16 f16s: two v_f16x8 results, {h0,h2,...,h14} and {h1,h3,...,h15}.
inline v_f16x8x2_t v_deinterleave_f16x8(const uint16_t* src) {
    float16x8x2_t v = vld2q_f16(
        reinterpret_cast<const float16_t*>(src));
    return {v_f16x8(v.val[0]), v_f16x8(v.val[1])};
}

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
inline v_f16x8 v_load_even_f16x8(const uint16_t* src) {
    return v_deinterleave_f16x8(src).even;
}
inline v_f16x8 v_load_odd_f16x8(const uint16_t* src) {
    return v_deinterleave_f16x8(src).odd;
}

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
inline v_f16x8 v_load_stride2_even_f16x8(const uint16_t* src) {
    return v_deinterleave_f16x8(src).even;
}
inline v_f16x8 v_load_stride2_odd_f16x8(const uint16_t* src) {
    return v_deinterleave_f16x8(src).odd;
}

// ---------------------------------------------------------------- 8x8 transpose

namespace {

/// In-place 4x4 transpose of four float32x4_t.
inline void transpose_4x4_f32(float32x4_t& r0, float32x4_t& r1,
                               float32x4_t& r2, float32x4_t& r3) {
    float32x4x2_t t01 = vtrnq_f32(r0, r1);   // {a0,b0,a2,b2}, {a1,b1,a3,b3}
    float32x4x2_t t23 = vtrnq_f32(r2, r3);   // {c0,d0,c2,d2}, {c1,d1,c3,d3}
    r0 = vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
    r1 = vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]));
    r2 = vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]));
    r3 = vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]));
}

}  // anonymous namespace

inline void v_transpose_8x8(v_f32x8& r0, v_f32x8& r1, v_f32x8& r2, v_f32x8& r3,
                             v_f32x8& r4, v_f32x8& r5, v_f32x8& r6, v_f32x8& r7) {
    // Bottom-left must be saved: the next block writes it from the top-right,
    // which is where this quadrant goes.
    float32x4_t t0, t1, t2, t3;
    float32x4_t u0 = r4.lo, u1 = r5.lo, u2 = r6.lo, u3 = r7.lo;

    t0 = r0.lo; t1 = r1.lo; t2 = r2.lo; t3 = r3.lo;
    transpose_4x4_f32(t0, t1, t2, t3);
    r0.lo = t0; r1.lo = t1; r2.lo = t2; r3.lo = t3;

    t0 = r0.hi; t1 = r1.hi; t2 = r2.hi; t3 = r3.hi;
    transpose_4x4_f32(t0, t1, t2, t3);
    r4.lo = t0; r5.lo = t1; r6.lo = t2; r7.lo = t3;

    transpose_4x4_f32(u0, u1, u2, u3);
    r0.hi = u0; r1.hi = u1; r2.hi = u2; r3.hi = u3;

    t0 = r4.hi; t1 = r5.hi; t2 = r6.hi; t3 = r7.hi;
    transpose_4x4_f32(t0, t1, t2, t3);
    r4.hi = t0; r5.hi = t1; r6.hi = t2; r7.hi = t3;
}

/// 8x8 f16 transpose via ZIP with widening lanes: f16, then u32, then u64.
inline void v_transpose_8x8(v_f16x8& r0, v_f16x8& r1, v_f16x8& r2, v_f16x8& r3,
                             v_f16x8& r4, v_f16x8& r5, v_f16x8& r6, v_f16x8& r7) {
    float16x8_t t0 = vzip1q_f16(r0.val, r1.val);
    float16x8_t t1 = vzip2q_f16(r0.val, r1.val);
    float16x8_t t2 = vzip1q_f16(r2.val, r3.val);
    float16x8_t t3 = vzip2q_f16(r2.val, r3.val);
    float16x8_t t4 = vzip1q_f16(r4.val, r5.val);
    float16x8_t t5 = vzip2q_f16(r4.val, r5.val);
    float16x8_t t6 = vzip1q_f16(r6.val, r7.val);
    float16x8_t t7 = vzip2q_f16(r6.val, r7.val);

    uint32x4_t u0 = vzip1q_u32(vreinterpretq_u32_f16(t0), vreinterpretq_u32_f16(t2));  // rows 0-3, cols 0-1
    uint32x4_t u1 = vzip2q_u32(vreinterpretq_u32_f16(t0), vreinterpretq_u32_f16(t2));  // rows 0-3, cols 2-3
    uint32x4_t u2 = vzip1q_u32(vreinterpretq_u32_f16(t1), vreinterpretq_u32_f16(t3));  // rows 0-3, cols 4-5
    uint32x4_t u3 = vzip2q_u32(vreinterpretq_u32_f16(t1), vreinterpretq_u32_f16(t3));  // rows 0-3, cols 6-7
    uint32x4_t u4 = vzip1q_u32(vreinterpretq_u32_f16(t4), vreinterpretq_u32_f16(t6));  // rows 4-7, cols 0-1
    uint32x4_t u5 = vzip2q_u32(vreinterpretq_u32_f16(t4), vreinterpretq_u32_f16(t6));  // rows 4-7, cols 2-3
    uint32x4_t u6 = vzip1q_u32(vreinterpretq_u32_f16(t5), vreinterpretq_u32_f16(t7));  // rows 4-7, cols 4-5
    uint32x4_t u7 = vzip2q_u32(vreinterpretq_u32_f16(t5), vreinterpretq_u32_f16(t7));  // rows 4-7, cols 6-7

    r0.val = vreinterpretq_f16_u64(vzip1q_u64(vreinterpretq_u64_u32(u0), vreinterpretq_u64_u32(u4)));
    r1.val = vreinterpretq_f16_u64(vzip2q_u64(vreinterpretq_u64_u32(u0), vreinterpretq_u64_u32(u4)));
    r2.val = vreinterpretq_f16_u64(vzip1q_u64(vreinterpretq_u64_u32(u1), vreinterpretq_u64_u32(u5)));
    r3.val = vreinterpretq_f16_u64(vzip2q_u64(vreinterpretq_u64_u32(u1), vreinterpretq_u64_u32(u5)));
    r4.val = vreinterpretq_f16_u64(vzip1q_u64(vreinterpretq_u64_u32(u2), vreinterpretq_u64_u32(u6)));
    r5.val = vreinterpretq_f16_u64(vzip2q_u64(vreinterpretq_u64_u32(u2), vreinterpretq_u64_u32(u6)));
    r6.val = vreinterpretq_f16_u64(vzip1q_u64(vreinterpretq_u64_u32(u3), vreinterpretq_u64_u32(u7)));
    r7.val = vreinterpretq_f16_u64(vzip2q_u64(vreinterpretq_u64_u32(u3), vreinterpretq_u64_u32(u7)));
}

// ---------------------------------------------------------------- int8

struct v_s8x16 {
    int8x16_t val;

    v_s8x16() = default;
    explicit v_s8x16(int8x16_t v) : val(v) {}
    explicit v_s8x16(int8_t s) : val(vdupq_n_s8(s)) {}
};

inline v_s8x16 v_load_s8x16(const int8_t* p) {
    return v_s8x16(vld1q_s8(p));
}
inline void v_store(int8_t* p, const v_s8x16& a) {
    vst1q_s8(p, a.val);
}
inline v_s8x16 v_set1_s8x16(int8_t s)  { return v_s8x16(vdupq_n_s8(s)); }
inline v_s8x16 v_zero_s8x16()         { return v_s8x16(vdupq_n_s8(0)); }

struct v_u8x16 {
    uint8x16_t val;

    v_u8x16() = default;
    explicit v_u8x16(uint8x16_t v) : val(v) {}
    explicit v_u8x16(uint8_t s) : val(vdupq_n_u8(s)) {}
};

inline v_u8x16 v_load_u8x16(const uint8_t* p) {
    return v_u8x16(vld1q_u8(p));
}
inline void v_store(uint8_t* p, const v_u8x16& a) {
    vst1q_u8(p, a.val);
}
inline v_u8x16 v_set1_u8x16(uint8_t s)  { return v_u8x16(vdupq_n_u8(s)); }
inline v_u8x16 v_zero_u8x16()         { return v_u8x16(vdupq_n_u8(0)); }

} // namespace neon
} // namespace arch
} // namespace simd
} // namespace nnops
