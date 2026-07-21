#pragma once
/// @file sse_mathfunc.hpp
/// @brief SSE1+MMX/SSE2 implementation of sin, cos, tan, exp, log, and tanh.
///
/// Based on sse_mathfun by Julien Pommier (zlib license):
///   http://gruntthepeon.free.fr/ssemath/
///
/// On MSVC, delegates to SVML (_mm_exp_ps, _mm_sin_ps, etc.).
/// On other compilers, uses the Cephes polynomial approximation algorithms.

#if !defined(NNOPS_ARCH_X86_64)
  #error "sse_mathfunc.hpp requires x86_64 architecture"
#endif

#include <smmintrin.h>
#include <emmintrin.h>

namespace nnops {
namespace simd {
namespace arch {
namespace sse {

#ifndef _MSC_VER

#define ALIGN16_BEG
#define ALIGN16_END __attribute__((aligned(16)))

#define _PS_CONST(Name, Val) \
    static const ALIGN16_BEG float _ps_##Name[4] ALIGN16_END = {Val, Val, Val, Val}
#define _PI32_CONST(Name, Val) \
    static const ALIGN16_BEG int _pi32_##Name[4] ALIGN16_END = {Val, Val, Val, Val}
#define _PS_CONST_TYPE(Name, Type, Val) \
    static const ALIGN16_BEG Type _ps_##Name[4] ALIGN16_END = {Val, Val, Val, Val}

_PS_CONST(1, 1.0f);
_PS_CONST(0p5, 0.5f);
_PS_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);
_PS_CONST_TYPE(sign_mask, int, (int)0x80000000);
_PS_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST(1, 1);
_PI32_CONST(inv1, ~1);
_PI32_CONST(2, 2);
_PI32_CONST(4, 4);
_PI32_CONST(0x7f, 0x7f);

_PS_CONST(cephes_SQRTHF, 0.707106781186547524f);
_PS_CONST(cephes_log_p0, 7.0376836292E-2f);
_PS_CONST(cephes_log_p1, -1.1514610310E-1f);
_PS_CONST(cephes_log_p2, 1.1676998740E-1f);
_PS_CONST(cephes_log_p3, -1.2420140846E-1f);
_PS_CONST(cephes_log_p4, +1.4249322787E-1f);
_PS_CONST(cephes_log_p5, -1.6668057665E-1f);
_PS_CONST(cephes_log_p6, +2.0000714765E-1f);
_PS_CONST(cephes_log_p7, -2.4999993993E-1f);
_PS_CONST(cephes_log_p8, +3.3333331174E-1f);
_PS_CONST(cephes_log_q1, -2.12194440e-4f);
_PS_CONST(cephes_log_q2, 0.693359375f);

static inline __m128 log_ps(__m128 x)
{
    __m128i emm0;
    __m128 one = *(__m128*)_ps_1;
    __m128 invalid_mask = _mm_cmple_ps(x, _mm_setzero_ps());
    x = _mm_max_ps(x, *(__m128*)_ps_min_norm_pos); /* cut off denormalized stuff */
    emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);
    /* keep only the fractional part */
    x = _mm_and_ps(x, *(__m128*)_ps_inv_mant_mask);
    x = _mm_or_ps(x, *(__m128*)_ps_0p5);
    emm0 = _mm_sub_epi32(emm0, *(__m128i*)_pi32_0x7f);
    __m128 e = _mm_cvtepi32_ps(emm0);
    e = _mm_add_ps(e, one);
    /* part2: if( x < SQRTHF ) { e -= 1; x = x + x - 1.0; } else { x = x - 1.0; } */
    __m128 mask = _mm_cmplt_ps(x, *(__m128*)_ps_cephes_SQRTHF);
    __m128 tmp = _mm_and_ps(x, mask);
    x = _mm_sub_ps(x, one);
    e = _mm_sub_ps(e, _mm_and_ps(one, mask));
    x = _mm_add_ps(x, tmp);
    __m128 z = _mm_mul_ps(x, x);
    __m128 y = *(__m128*)_ps_cephes_log_p0;
    y = _mm_fmadd_ps(y, x, *(__m128*)_ps_cephes_log_p1);
    y = _mm_fmadd_ps(y, x, *(__m128*)_ps_cephes_log_p2);
    y = _mm_fmadd_ps(y, x, *(__m128*)_ps_cephes_log_p3);
    y = _mm_fmadd_ps(y, x, *(__m128*)_ps_cephes_log_p4);
    y = _mm_fmadd_ps(y, x, *(__m128*)_ps_cephes_log_p5);
    y = _mm_fmadd_ps(y, x, *(__m128*)_ps_cephes_log_p6);
    y = _mm_fmadd_ps(y, x, *(__m128*)_ps_cephes_log_p7);
    y = _mm_fmadd_ps(y, x, *(__m128*)_ps_cephes_log_p8);
    y = _mm_mul_ps(y, x);
    y = _mm_mul_ps(y, z);
    y = _mm_fmadd_ps(e, *(__m128*)_ps_cephes_log_q1, y);
    y = _mm_fnmadd_ps(z, *(__m128*)_ps_0p5, y);
    x = _mm_add_ps(x, y);
    x = _mm_fmadd_ps(e, *(__m128*)_ps_cephes_log_q2, x);
    x = _mm_or_ps(x, invalid_mask); // negative arg will be NAN
    return x;
}

_PS_CONST(exp_hi, 88.3762626647949f);
_PS_CONST(exp_lo, -88.3762626647949f);
_PS_CONST(cephes_LOG2EF, 1.44269504088896341f);
_PS_CONST(cephes_exp_C1, 0.693359375f);
_PS_CONST(cephes_exp_C2, -2.12194440e-4f);
_PS_CONST(cephes_exp_p0, 1.9875691500E-4f);
_PS_CONST(cephes_exp_p1, 1.3981999507E-3f);
_PS_CONST(cephes_exp_p2, 8.3334519073E-3f);
_PS_CONST(cephes_exp_p3, 4.1665795894E-2f);
_PS_CONST(cephes_exp_p4, 1.6666665459E-1f);
_PS_CONST(cephes_exp_p5, 5.0000001201E-1f);

static inline __m128 exp_ps(__m128 x)
{
    __m128 tmp = _mm_setzero_ps(), fx;
    __m128i emm0;
    __m128 one = *(__m128*)_ps_1;
    x = _mm_min_ps(x, *(__m128*)_ps_exp_hi);
    x = _mm_max_ps(x, *(__m128*)_ps_exp_lo);
    /* express exp(x) as exp(g + n*log(2)) */
    fx = _mm_mul_ps(x, *(__m128*)_ps_cephes_LOG2EF);
    fx = _mm_add_ps(fx, *(__m128*)_ps_0p5);
    /* how to perform a floorf with SSE: just below */
    emm0 = _mm_cvttps_epi32(fx);
    tmp = _mm_cvtepi32_ps(emm0);
    /* if greater, subtract 1 */
    __m128 mask = _mm_cmpgt_ps(tmp, fx);
    mask = _mm_and_ps(mask, one);
    fx = _mm_sub_ps(tmp, mask);
    x = _mm_fnmadd_ps(fx, *(__m128*)_ps_cephes_exp_C1, x);
    x = _mm_fnmadd_ps(fx, *(__m128*)_ps_cephes_exp_C2, x);
    tmp = _mm_mul_ps(x, x);
    __m128 y = *(__m128*)_ps_cephes_exp_p0;
    y = _mm_fmadd_ps(y, x, *(__m128*)_ps_cephes_exp_p1);
    y = _mm_fmadd_ps(y, x, *(__m128*)_ps_cephes_exp_p2);
    y = _mm_fmadd_ps(y, x, *(__m128*)_ps_cephes_exp_p3);
    y = _mm_fmadd_ps(y, x, *(__m128*)_ps_cephes_exp_p4);
    y = _mm_fmadd_ps(y, x, *(__m128*)_ps_cephes_exp_p5);
    y = _mm_fmadd_ps(y, tmp, x);
    y = _mm_add_ps(y, one);
    /* build 2^n */
    emm0 = _mm_cvttps_epi32(fx);
    emm0 = _mm_add_epi32(emm0, *(__m128i*)_pi32_0x7f);
    emm0 = _mm_slli_epi32(emm0, 23);
    __m128 pow2n = _mm_castsi128_ps(emm0);
    y = _mm_mul_ps(y, pow2n);
    return y;
}

// tanh: rational polynomial approximation (Pade-like), valid on [-9, 9]
_PS_CONST(tanh_hi, 9.0f);
_PS_CONST(tanh_lo, -9.0f);
_PS_CONST(cephes_tanh_p0, -2.76076847742355E-16f);
_PS_CONST(cephes_tanh_p1, 2.00018790482477E-13f);
_PS_CONST(cephes_tanh_p2, -8.60467152213735E-11f);
_PS_CONST(cephes_tanh_p3, 5.12229709037114E-08f);
_PS_CONST(cephes_tanh_p4, 1.48572235717979E-05f);
_PS_CONST(cephes_tanh_p5, 6.37261928875436E-04f);
_PS_CONST(cephes_tanh_p6, 4.89352455891786E-03f);
_PS_CONST(cephes_tanh_p7, 1.19825839466702e-06f);
_PS_CONST(cephes_tanh_p8, 1.18534705686654e-04f);
_PS_CONST(cephes_tanh_p9, 2.26843463243900e-03f);

static inline __m128 tanh_ps(const __m128 x)
{
    __m128 value = x;
    value = _mm_max_ps(*(__m128*)_ps_tanh_lo, value);
    value = _mm_min_ps(*(__m128*)_ps_tanh_hi, value);
    __m128 value_squared = _mm_mul_ps(value, value);
    __m128 p;
    p = _mm_fmadd_ps(value_squared, *(__m128*)_ps_cephes_tanh_p0, *(__m128*)_ps_cephes_tanh_p1);
    p = _mm_fmadd_ps(p, value_squared, *(__m128*)_ps_cephes_tanh_p2);
    p = _mm_fmadd_ps(p, value_squared, *(__m128*)_ps_cephes_tanh_p3);
    p = _mm_fmadd_ps(p, value_squared, *(__m128*)_ps_cephes_tanh_p4);
    p = _mm_fmadd_ps(p, value_squared, *(__m128*)_ps_cephes_tanh_p5);
    p = _mm_fmadd_ps(p, value_squared, *(__m128*)_ps_cephes_tanh_p6);
    p = _mm_mul_ps(p, value);
    __m128 q;
    q = _mm_fmadd_ps(value_squared, *(__m128*)_ps_cephes_tanh_p7, *(__m128*)_ps_cephes_tanh_p8);
    q = _mm_fmadd_ps(q, value_squared, *(__m128*)_ps_cephes_tanh_p9);
    q = _mm_fmadd_ps(q, value_squared, *(__m128*)_ps_cephes_tanh_p6);
    __m128 dst = _mm_div_ps(p, q);
    return dst;
}

// sin/cos constants and functions (Cephes Cody-Waite range reduction)
_PS_CONST(minus_cephes_DP1, -0.78515625f);
_PS_CONST(minus_cephes_DP2, -2.4187564849853515625e-4f);
_PS_CONST(minus_cephes_DP3, -3.77489497744594108e-8f);
_PS_CONST(sincof_p0, -1.9515295891E-4f);
_PS_CONST(sincof_p1, 8.3321608736E-3f);
_PS_CONST(sincof_p2, -1.6666654611E-1f);
_PS_CONST(coscof_p0, 2.443315711809948E-005f);
_PS_CONST(coscof_p1, -1.388731625493765E-003f);
_PS_CONST(coscof_p2, 4.166664568298827E-002f);
_PS_CONST(cephes_FOPI, 1.27323954473516f); // 4 / M_PI

static inline __m128 sin_ps(__m128 x)
{
    __m128 xmm1, xmm2 = _mm_setzero_ps(), xmm3, sign_bit, y;
    __m128i emm0, emm2;
    sign_bit = x;
    x = _mm_and_ps(x, *(__m128*)_ps_inv_sign_mask);
    sign_bit = _mm_and_ps(sign_bit, *(__m128*)_ps_sign_mask);
    y = _mm_mul_ps(x, *(__m128*)_ps_cephes_FOPI);
    emm2 = _mm_cvttps_epi32(y);
    emm2 = _mm_add_epi32(emm2, *(__m128i*)_pi32_1);
    emm2 = _mm_and_si128(emm2, *(__m128i*)_pi32_inv1);
    y = _mm_cvtepi32_ps(emm2);
    emm0 = _mm_and_si128(emm2, *(__m128i*)_pi32_4);
    emm0 = _mm_slli_epi32(emm0, 29);
    emm2 = _mm_and_si128(emm2, *(__m128i*)_pi32_2);
    emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());
    __m128 swap_sign_bit = _mm_castsi128_ps(emm0);
    __m128 poly_mask = _mm_castsi128_ps(emm2);
    sign_bit = _mm_xor_ps(sign_bit, swap_sign_bit);
    xmm1 = *(__m128*)_ps_minus_cephes_DP1;
    xmm2 = *(__m128*)_ps_minus_cephes_DP2;
    xmm3 = *(__m128*)_ps_minus_cephes_DP3;
    x = _mm_fmadd_ps(y, xmm1, x);
    x = _mm_fmadd_ps(y, xmm2, x);
    x = _mm_fmadd_ps(y, xmm3, x);
    y = *(__m128*)_ps_coscof_p0;
    __m128 z = _mm_mul_ps(x, x);
    y = _mm_fmadd_ps(y, z, *(__m128*)_ps_coscof_p1);
    y = _mm_fmadd_ps(y, z, *(__m128*)_ps_coscof_p2);
    y = _mm_mul_ps(y, z);
    y = _mm_mul_ps(y, z);
    y = _mm_fnmadd_ps(z, *(__m128*)_ps_0p5, y);
    y = _mm_add_ps(y, *(__m128*)_ps_1);
    __m128 y2 = *(__m128*)_ps_sincof_p0;
    y2 = _mm_fmadd_ps(y2, z, *(__m128*)_ps_sincof_p1);
    y2 = _mm_fmadd_ps(y2, z, *(__m128*)_ps_sincof_p2);
    y2 = _mm_mul_ps(y2, z);
    y2 = _mm_fmadd_ps(y2, x, x);
    xmm3 = poly_mask;
    y2 = _mm_and_ps(xmm3, y2);
    y = _mm_andnot_ps(xmm3, y);
    y = _mm_add_ps(y, y2);
    y = _mm_xor_ps(y, sign_bit);
    return y;
}

static inline __m128 cos_ps(__m128 x)
{
    __m128 xmm1, xmm2 = _mm_setzero_ps(), xmm3, y;
    __m128i emm0, emm2;
    x = _mm_and_ps(x, *(__m128*)_ps_inv_sign_mask);
    y = _mm_mul_ps(x, *(__m128*)_ps_cephes_FOPI);
    emm2 = _mm_cvttps_epi32(y);
    emm2 = _mm_add_epi32(emm2, *(__m128i*)_pi32_1);
    emm2 = _mm_and_si128(emm2, *(__m128i*)_pi32_inv1);
    y = _mm_cvtepi32_ps(emm2);
    emm2 = _mm_sub_epi32(emm2, *(__m128i*)_pi32_2);
    emm0 = _mm_andnot_si128(emm2, *(__m128i*)_pi32_4);
    emm0 = _mm_slli_epi32(emm0, 29);
    emm2 = _mm_and_si128(emm2, *(__m128i*)_pi32_2);
    emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());
    __m128 sign_bit = _mm_castsi128_ps(emm0);
    __m128 poly_mask = _mm_castsi128_ps(emm2);
    xmm1 = *(__m128*)_ps_minus_cephes_DP1;
    xmm2 = *(__m128*)_ps_minus_cephes_DP2;
    xmm3 = *(__m128*)_ps_minus_cephes_DP3;
    x = _mm_fmadd_ps(y, xmm1, x);
    x = _mm_fmadd_ps(y, xmm2, x);
    x = _mm_fmadd_ps(y, xmm3, x);
    y = *(__m128*)_ps_coscof_p0;
    __m128 z = _mm_mul_ps(x, x);
    y = _mm_fmadd_ps(y, z, *(__m128*)_ps_coscof_p1);
    y = _mm_fmadd_ps(y, z, *(__m128*)_ps_coscof_p2);
    y = _mm_mul_ps(y, z);
    y = _mm_mul_ps(y, z);
    y = _mm_fnmadd_ps(z, *(__m128*)_ps_0p5, y);
    y = _mm_add_ps(y, *(__m128*)_ps_1);
    __m128 y2 = *(__m128*)_ps_sincof_p0;
    y2 = _mm_fmadd_ps(y2, z, *(__m128*)_ps_sincof_p1);
    y2 = _mm_fmadd_ps(y2, z, *(__m128*)_ps_sincof_p2);
    y2 = _mm_mul_ps(y2, z);
    y2 = _mm_fmadd_ps(y2, x, x);
    xmm3 = poly_mask;
    y2 = _mm_and_ps(xmm3, y2);
    y = _mm_andnot_ps(xmm3, y);
    y = _mm_add_ps(y, y2);
    y = _mm_xor_ps(y, sign_bit);
    return y;
}

static inline void sincos_ps(__m128 x, __m128* s, __m128* c)
{
    __m128 xmm1, xmm2, xmm3 = _mm_setzero_ps(), sign_bit_sin, y;
    __m128i emm0, emm2, emm4;
    sign_bit_sin = x;
    x = _mm_and_ps(x, *(__m128*)_ps_inv_sign_mask);
    sign_bit_sin = _mm_and_ps(sign_bit_sin, *(__m128*)_ps_sign_mask);
    y = _mm_mul_ps(x, *(__m128*)_ps_cephes_FOPI);
    emm2 = _mm_cvttps_epi32(y);
    emm2 = _mm_add_epi32(emm2, *(__m128i*)_pi32_1);
    emm2 = _mm_and_si128(emm2, *(__m128i*)_pi32_inv1);
    y = _mm_cvtepi32_ps(emm2);
    emm4 = emm2;
    emm0 = _mm_and_si128(emm2, *(__m128i*)_pi32_4);
    emm0 = _mm_slli_epi32(emm0, 29);
    __m128 swap_sign_bit_sin = _mm_castsi128_ps(emm0);
    emm2 = _mm_and_si128(emm2, *(__m128i*)_pi32_2);
    emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());
    __m128 poly_mask = _mm_castsi128_ps(emm2);
    xmm1 = *(__m128*)_ps_minus_cephes_DP1;
    xmm2 = *(__m128*)_ps_minus_cephes_DP2;
    xmm3 = *(__m128*)_ps_minus_cephes_DP3;
    x = _mm_fmadd_ps(y, xmm1, x);
    x = _mm_fmadd_ps(y, xmm2, x);
    x = _mm_fmadd_ps(y, xmm3, x);
    emm4 = _mm_sub_epi32(emm4, *(__m128i*)_pi32_2);
    emm4 = _mm_andnot_si128(emm4, *(__m128i*)_pi32_4);
    emm4 = _mm_slli_epi32(emm4, 29);
    __m128 sign_bit_cos = _mm_castsi128_ps(emm4);
    sign_bit_sin = _mm_xor_ps(sign_bit_sin, swap_sign_bit_sin);
    __m128 z = _mm_mul_ps(x, x);
    y = *(__m128*)_ps_coscof_p0;
    y = _mm_fmadd_ps(y, z, *(__m128*)_ps_coscof_p1);
    y = _mm_fmadd_ps(y, z, *(__m128*)_ps_coscof_p2);
    y = _mm_mul_ps(y, z);
    y = _mm_mul_ps(y, z);
    y = _mm_fnmadd_ps(z, *(__m128*)_ps_0p5, y);
    y = _mm_add_ps(y, *(__m128*)_ps_1);
    __m128 y2 = *(__m128*)_ps_sincof_p0;
    y2 = _mm_fmadd_ps(y2, z, *(__m128*)_ps_sincof_p1);
    y2 = _mm_fmadd_ps(y2, z, *(__m128*)_ps_sincof_p2);
    y2 = _mm_mul_ps(y2, z);
    y2 = _mm_fmadd_ps(y2, x, x);
    xmm3 = poly_mask;
    __m128 ysin2 = _mm_and_ps(xmm3, y2);
    __m128 ysin1 = _mm_andnot_ps(xmm3, y);
    y2 = _mm_sub_ps(y2, ysin2);
    y = _mm_sub_ps(y, ysin1);
    xmm1 = _mm_add_ps(ysin1, ysin2);
    xmm2 = _mm_add_ps(y, y2);
    *s = _mm_xor_ps(xmm1, sign_bit_sin);
    *c = _mm_xor_ps(xmm2, sign_bit_cos);
}

static inline __m128 tan_ps(__m128 x)
{
    __m128 ysin, ycos;
    sincos_ps(x, &ysin, &ycos);
    __m128 mask = _mm_cmpeq_ps(ycos, _mm_setzero_ps());
    __m128 eps = _mm_set1_ps(1E-8f);
    ycos = _mm_add_ps(ycos, _mm_and_ps(eps, mask));
    return _mm_div_ps(ysin, ycos);
}

#else  // _MSC_VER: use Intel SVML

static inline __m128 log_ps(__m128 x)   { return _mm_log_ps(x); }
static inline __m128 exp_ps(__m128 x)   { return _mm_exp_ps(x); }
static inline __m128 sin_ps(__m128 x)   { return _mm_sin_ps(x); }
static inline __m128 cos_ps(__m128 x)   { return _mm_cos_ps(x); }
static inline __m128 tan_ps(__m128 x)   { return _mm_tan_ps(x); }
static inline __m128 tanh_ps(const __m128 x) { return _mm_tanh_ps(x); }

#endif  // _MSC_VER

} // namespace sse
} // namespace arch
} // namespace simd
} // namespace nnops
