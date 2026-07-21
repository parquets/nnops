#pragma once
/// @file neon_mathfunc.hpp
/// @brief ARM NEON implementation of sin, cos, tan, exp, log, and tanh.
///
/// Based on neon_mathfun by Julien Pommier (zlib license):
///   http://gruntthepeon.free.fr/ssemath/
///
/// tanh_ps is derived from ncnn (Tencent, BSD 3-Clause).
///
/// All functions use the Cephes polynomial approximation algorithms
/// adapted for ARM NEON float32x4_t intrinsics.

#if !defined(NNOPS_ARCH_AARCH64)
  #error "neon_mathfunc.hpp requires AArch64 architecture"
#endif

#include <arm_neon.h>
#include <cmath>

namespace nnops {
namespace simd {
namespace arch {
namespace neon {

// ============================================================
// Constants (macro-defined for Cephes algorithms)
// ============================================================
#define c_inv_mant_mask ~0x7f800000u
#define c_cephes_SQRTHF 0.707106781186547524
#define c_cephes_log_p0 7.0376836292E-2
#define c_cephes_log_p1 -1.1514610310E-1
#define c_cephes_log_p2 1.1676998740E-1
#define c_cephes_log_p3 -1.2420140846E-1
#define c_cephes_log_p4 +1.4249322787E-1
#define c_cephes_log_p5 -1.6668057665E-1
#define c_cephes_log_p6 +2.0000714765E-1
#define c_cephes_log_p7 -2.4999993993E-1
#define c_cephes_log_p8 +3.3333331174E-1
#define c_cephes_log_q1 -2.12194440e-4
#define c_cephes_log_q2 0.693359375

// ============================================================
// log_ps — natural logarithm, 4-wide
// ============================================================
static inline float32x4_t log_ps(float32x4_t x)
{
    float32x4_t one = vdupq_n_f32(1);
    x = vmaxq_f32(x, vdupq_n_f32(0)); /* force flush to zero on denormal values */
    uint32x4_t invalid_mask = vcleq_f32(x, vdupq_n_f32(0));
    int32x4_t ux = vreinterpretq_s32_f32(x);
    int32x4_t emm0 = vshrq_n_s32(ux, 23);
    /* keep only the fractional part */
    ux = vandq_s32(ux, vdupq_n_s32(c_inv_mant_mask));
    ux = vorrq_s32(ux, vreinterpretq_s32_f32(vdupq_n_f32(0.5f)));
    x = vreinterpretq_f32_s32(ux);
    emm0 = vsubq_s32(emm0, vdupq_n_s32(0x7f));
    float32x4_t e = vcvtq_f32_s32(emm0);
    e = vaddq_f32(e, one);
    /* part2: if( x < SQRTHF ) { e -= 1; x = x + x - 1.0; } else { x = x - 1.0; } */
    uint32x4_t mask = vcltq_f32(x, vdupq_n_f32(c_cephes_SQRTHF));
    float32x4_t tmp = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), mask));
    x = vsubq_f32(x, one);
    e = vsubq_f32(e, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(one), mask)));
    x = vaddq_f32(x, tmp);
    float32x4_t z = vmulq_f32(x, x);
    float32x4_t y = vdupq_n_f32(c_cephes_log_p0);
    y = vmlaq_f32(vdupq_n_f32(c_cephes_log_p1), y, x);
    y = vmlaq_f32(vdupq_n_f32(c_cephes_log_p2), y, x);
    y = vmlaq_f32(vdupq_n_f32(c_cephes_log_p3), y, x);
    y = vmlaq_f32(vdupq_n_f32(c_cephes_log_p4), y, x);
    y = vmlaq_f32(vdupq_n_f32(c_cephes_log_p5), y, x);
    y = vmlaq_f32(vdupq_n_f32(c_cephes_log_p6), y, x);
    y = vmlaq_f32(vdupq_n_f32(c_cephes_log_p7), y, x);
    y = vmlaq_f32(vdupq_n_f32(c_cephes_log_p8), y, x);
    y = vmulq_f32(y, x);
    y = vmulq_f32(y, z);
    y = vmlaq_f32(y, e, vdupq_n_f32(c_cephes_log_q1));
    y = vmlsq_f32(y, z, vdupq_n_f32(0.5f));
    x = vaddq_f32(x, y);
    x = vmlaq_f32(x, e, vdupq_n_f32(c_cephes_log_q2));
    x = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x), invalid_mask)); // negative arg will be NAN
    return x;
}

// ============================================================
// exp_ps — exponential, 4-wide
// ============================================================
#define c_exp_hi 88.3762626647949f
#define c_exp_lo -88.3762626647949f
#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4
#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

static inline float32x4_t exp_ps(float32x4_t x)
{
    float32x4_t tmp, fx;
    float32x4_t one = vdupq_n_f32(1);
    x = vminq_f32(x, vdupq_n_f32(c_exp_hi));
    x = vmaxq_f32(x, vdupq_n_f32(c_exp_lo));
    /* express exp(x) as exp(g + n*log(2)) */
    fx = vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(c_cephes_LOG2EF));
    /* perform a floorf */
    tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));
    /* if greater, subtract 1 */
    uint32x4_t mask = vcgtq_f32(tmp, fx);
    mask = vandq_u32(mask, vreinterpretq_u32_f32(one));
    fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));
    tmp = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C1));
    float32x4_t z = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C2));
    x = vsubq_f32(x, tmp);
    x = vsubq_f32(x, z);
    z = vmulq_f32(x, x);
    float32x4_t y = vdupq_n_f32(c_cephes_exp_p0);
    y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p1), y, x);
    y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p2), y, x);
    y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p3), y, x);
    y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p4), y, x);
    y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p5), y, x);
    y = vmlaq_f32(x, y, z);
    y = vaddq_f32(y, one);
    /* build 2^n */
    int32x4_t mm;
    mm = vcvtq_s32_f32(fx);
    mm = vaddq_s32(mm, vdupq_n_s32(0x7f));
    mm = vshlq_n_s32(mm, 23);
    float32x4_t pow2n = vreinterpretq_f32_s32(mm);
    y = vmulq_f32(y, pow2n);
    return y;
}

// ============================================================
// sincos_ps — simultaneous sin/cos, 4-wide (Cody-Waite range reduction)
// ============================================================
#define c_minus_cephes_DP1 -0.78515625
#define c_minus_cephes_DP2 -2.4187564849853515625e-4
#define c_minus_cephes_DP3 -3.77489497744594108e-8
#define c_sincof_p0        -1.9515295891E-4
#define c_sincof_p1        8.3321608736E-3
#define c_sincof_p2        -1.6666654611E-1
#define c_coscof_p0        2.443315711809948E-005
#define c_coscof_p1        -1.388731625493765E-003
#define c_coscof_p2        4.166664568298827E-002
#define c_cephes_FOPI      1.27323954473516 // 4 / M_PI

static inline void sincos_ps(float32x4_t x, float32x4_t* ysin, float32x4_t* ycos)
{
    float32x4_t y;
    uint32x4_t emm2;
    uint32x4_t sign_mask_sin, sign_mask_cos;
    sign_mask_sin = vcltq_f32(x, vdupq_n_f32(0));
    x = vabsq_f32(x);
    /* scale by 4/Pi */
    y = vmulq_f32(x, vdupq_n_f32(c_cephes_FOPI));
    /* store the integer part of y in mm0 */
    emm2 = vcvtq_u32_f32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    emm2 = vaddq_u32(emm2, vdupq_n_u32(1));
    emm2 = vandq_u32(emm2, vdupq_n_u32(~1));
    y = vcvtq_f32_u32(emm2);
    uint32x4_t poly_mask = vtstq_u32(emm2, vdupq_n_u32(2));
    /* The magic pass: "Extended precision modular arithmetic" */
    x = vmlaq_f32(x, y, vdupq_n_f32(c_minus_cephes_DP1));
    x = vmlaq_f32(x, y, vdupq_n_f32(c_minus_cephes_DP2));
    x = vmlaq_f32(x, y, vdupq_n_f32(c_minus_cephes_DP3));
    sign_mask_sin = veorq_u32(sign_mask_sin, vtstq_u32(emm2, vdupq_n_u32(4)));
    sign_mask_cos = vtstq_u32(vsubq_u32(emm2, vdupq_n_u32(2)), vdupq_n_u32(4));
    /* Evaluate the two polynoms */
    float32x4_t z = vmulq_f32(x, x);
    float32x4_t y1, y2;
    y1 = vmlaq_f32(vdupq_n_f32(c_coscof_p1), z, vdupq_n_f32(c_coscof_p0));
    y2 = vmlaq_f32(vdupq_n_f32(c_sincof_p1), z, vdupq_n_f32(c_sincof_p0));
    y1 = vmlaq_f32(vdupq_n_f32(c_coscof_p2), y1, z);
    y2 = vmlaq_f32(vdupq_n_f32(c_sincof_p2), y2, z);
    y1 = vmulq_f32(y1, z);
    y2 = vmulq_f32(y2, z);
    y1 = vmulq_f32(y1, z);
    y1 = vmlsq_f32(y1, z, vdupq_n_f32(0.5f));
    y2 = vmlaq_f32(x, y2, x);
    y1 = vaddq_f32(y1, vdupq_n_f32(1));
    /* select the correct result */
    float32x4_t ys = vbslq_f32(poly_mask, y1, y2);
    float32x4_t yc = vbslq_f32(poly_mask, y2, y1);
    *ysin = vbslq_f32(sign_mask_sin, vnegq_f32(ys), ys);
    *ycos = vbslq_f32(sign_mask_cos, yc, vnegq_f32(yc));
}

static inline float32x4_t sin_ps(float32x4_t x)
{
    float32x4_t ysin, ycos;
    sincos_ps(x, &ysin, &ycos);
    return ysin;
}

static inline float32x4_t cos_ps(float32x4_t x)
{
    float32x4_t ysin, ycos;
    sincos_ps(x, &ysin, &ycos);
    return ycos;
}

// ============================================================
// tan_ps — tangent, 4-wide (sin/cos with epsilon guard)
// ============================================================
static inline float32x4_t tan_ps(float32x4_t x)
{
    float32x4_t ysin, ycos;
    sincos_ps(x, &ysin, &ycos);
    // Guard against division by zero: add tiny epsilon to zero cosine lanes
    uint32x4_t zero_mask = vceqq_f32(ycos, vdupq_n_f32(0.0f));
    float32x4_t eps = vdupq_n_f32(1E-8f);
    ycos = vaddq_f32(ycos, vreinterpretq_f32_u32(vandq_u32(zero_mask, vreinterpretq_u32_f32(eps))));
    return vdivq_f32(ysin, ycos);
}

// ============================================================
// tanh_ps — hyperbolic tangent, 4-wide (ncnn-derived rational polynomial)
// ============================================================
#define c_tanh_tiny 1e-4f
#define c_tanh_hi   9.0f
#define c_tanh_alpha_1  4.89352455891786e-3f
#define c_tanh_alpha_3  6.37261928875436e-4f
#define c_tanh_alpha_5  1.48572235717979e-5f
#define c_tanh_alpha_7  5.12229709037114e-8f
#define c_tanh_alpha_9  -8.60467152213735e-11f
#define c_tanh_alpha_11 2.00018790482477e-13f
#define c_tanh_alpha_13 -2.76076847742355e-16f
#define c_tanh_beta_0 4.89352518554385e-3f
#define c_tanh_beta_2 2.26843463243900e-3f
#define c_tanh_beta_4 1.18534705686654e-4f
#define c_tanh_beta_6 1.19825839466702e-6f

static inline float32x4_t tanh_ps(float32x4_t x)
{
    float32x4_t x2 = vabsq_f32(x);
    uint32x4_t tiny_mask = vcgeq_f32(x2, vdupq_n_f32(c_tanh_tiny));
    // clamp the inputs to the range [-9, 9]
    x2 = vreinterpretq_f32_u32(vbslq_u32(
        vcgeq_f32(vdupq_n_f32(c_tanh_hi), x2),
        vreinterpretq_u32_f32(x2),
        vreinterpretq_u32_f32(vdupq_n_f32(c_tanh_hi))));
    // since the polynomials are odd/even, we need x**2
    float32x4_t z = vmulq_f32(x2, x2);
    // evaluate the numerator polynomial y (odd)
    float32x4_t y = vdupq_n_f32(c_tanh_alpha_13);
    y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_11), y, z);
    y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_9), y, z);
    y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_7), y, z);
    y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_5), y, z);
    y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_3), y, z);
    y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_1), y, z);
    y = vmulq_f32(y, x2);
    // evaluate the denominator polynomial w (even)
    float32x4_t w = vdupq_n_f32(c_tanh_beta_6);
    w = vmlaq_f32(vdupq_n_f32(c_tanh_beta_4), w, z);
    w = vmlaq_f32(vdupq_n_f32(c_tanh_beta_2), w, z);
    w = vmlaq_f32(vdupq_n_f32(c_tanh_beta_0), w, z);
    // divide the numerator by the denominator
    y = vdivq_f32(y, w);
    // reinstate the sign
    y = vreinterpretq_f32_u32(vbslq_u32(
        vdupq_n_u32(1u << 31),
        vreinterpretq_u32_f32(x),
        vreinterpretq_u32_f32(y)));
    // when the argument is very small in magnitude it's more accurate to just return it
    y = vreinterpretq_f32_u32(vbslq_u32(
        tiny_mask,
        vreinterpretq_u32_f32(y),
        vreinterpretq_u32_f32(x)));
    return y;
}

// Undefine local constants to avoid polluting the namespace
#undef c_inv_mant_mask
#undef c_cephes_SQRTHF
#undef c_cephes_log_p0
#undef c_cephes_log_p1
#undef c_cephes_log_p2
#undef c_cephes_log_p3
#undef c_cephes_log_p4
#undef c_cephes_log_p5
#undef c_cephes_log_p6
#undef c_cephes_log_p7
#undef c_cephes_log_p8
#undef c_cephes_log_q1
#undef c_cephes_log_q2
#undef c_exp_hi
#undef c_exp_lo
#undef c_cephes_LOG2EF
#undef c_cephes_exp_C1
#undef c_cephes_exp_C2
#undef c_cephes_exp_p0
#undef c_cephes_exp_p1
#undef c_cephes_exp_p2
#undef c_cephes_exp_p3
#undef c_cephes_exp_p4
#undef c_cephes_exp_p5
#undef c_minus_cephes_DP1
#undef c_minus_cephes_DP2
#undef c_minus_cephes_DP3
#undef c_sincof_p0
#undef c_sincof_p1
#undef c_sincof_p2
#undef c_coscof_p0
#undef c_coscof_p1
#undef c_coscof_p2
#undef c_cephes_FOPI
#undef c_tanh_tiny
#undef c_tanh_hi
#undef c_tanh_alpha_1
#undef c_tanh_alpha_3
#undef c_tanh_alpha_5
#undef c_tanh_alpha_7
#undef c_tanh_alpha_9
#undef c_tanh_alpha_11
#undef c_tanh_alpha_13
#undef c_tanh_beta_0
#undef c_tanh_beta_2
#undef c_tanh_beta_4
#undef c_tanh_beta_6

} // namespace neon
} // namespace arch
} // namespace simd
} // namespace nnops
