#pragma once
/// @file avx2_mathfunc.hpp
/// @brief AVX2 implementation of sin, cos, tan, exp, log, and tanh.
///
/// Based on avx_mathfun by Giovanni Garberoglio (zlib license), which is
/// itself derived from sse_mathfun by Julien Pommier.
///
/// On MSVC, delegates to SVML (_mm256_exp_ps, _mm256_sin_ps, etc.).
/// On other compilers, uses the Cephes polynomial approximation algorithms.

#if !defined(NNOPS_ARCH_X86_64)
  #error "avx2_mathfunc.hpp requires x86_64 architecture"
#endif

#include <immintrin.h>

namespace nnops {
namespace simd {
namespace arch {
namespace avx2 {

#ifndef _MSC_VER

#define ALIGN32_BEG
#define ALIGN32_END __attribute__((aligned(32)))

#define _PS256_CONST(Name, Val) \
    static const ALIGN32_BEG float _ps256_##Name[8] ALIGN32_END = {Val, Val, Val, Val, Val, Val, Val, Val}
#define _PI32_CONST256(Name, Val) \
    static const ALIGN32_BEG int _pi32_256_##Name[8] ALIGN32_END = {Val, Val, Val, Val, Val, Val, Val, Val}
#define _PS256_CONST_TYPE(Name, Type, Val) \
    static const ALIGN32_BEG Type _ps256_##Name[8] ALIGN32_END = {Val, Val, Val, Val, Val, Val, Val, Val}

_PS256_CONST(1, 1.0f);
_PS256_CONST(0p5, 0.5f);
_PS256_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS256_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS256_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);
_PS256_CONST_TYPE(sign_mask, int, (int)0x80000000);
_PS256_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST256(0, 0);
_PI32_CONST256(1, 1);
_PI32_CONST256(inv1, ~1);
_PI32_CONST256(2, 2);
_PI32_CONST256(4, 4);
_PI32_CONST256(0x7f, 0x7f);

_PS256_CONST(cephes_SQRTHF, 0.707106781186547524f);
_PS256_CONST(cephes_log_p0, 7.0376836292E-2f);
_PS256_CONST(cephes_log_p1, -1.1514610310E-1f);
_PS256_CONST(cephes_log_p2, 1.1676998740E-1f);
_PS256_CONST(cephes_log_p3, -1.2420140846E-1f);
_PS256_CONST(cephes_log_p4, +1.4249322787E-1f);
_PS256_CONST(cephes_log_p5, -1.6668057665E-1f);
_PS256_CONST(cephes_log_p6, +2.0000714765E-1f);
_PS256_CONST(cephes_log_p7, -2.4999993993E-1f);
_PS256_CONST(cephes_log_p8, +3.3333331174E-1f);
_PS256_CONST(cephes_log_q1, -2.12194440e-4f);
_PS256_CONST(cephes_log_q2, 0.693359375f);

static inline __m256 log256_ps(__m256 x)
{
    __m256i imm0;
    __m256 one = *(__m256*)_ps256_1;
    __m256 invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);
    x = _mm256_max_ps(x, *(__m256*)_ps256_min_norm_pos);
    imm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);
    x = _mm256_and_ps(x, *(__m256*)_ps256_inv_mant_mask);
    x = _mm256_or_ps(x, *(__m256*)_ps256_0p5);
    imm0 = _mm256_sub_epi32(imm0, *(__m256i*)_pi32_256_0x7f);
    __m256 e = _mm256_cvtepi32_ps(imm0);
    e = _mm256_add_ps(e, one);
    __m256 mask = _mm256_cmp_ps(x, *(__m256*)_ps256_cephes_SQRTHF, _CMP_LT_OS);
    __m256 tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, one);
    e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
    x = _mm256_add_ps(x, tmp);
    __m256 z = _mm256_mul_ps(x, x);
    __m256 y = *(__m256*)_ps256_cephes_log_p0;
    y = _mm256_fmadd_ps(y, x, *(__m256*)_ps256_cephes_log_p1);
    y = _mm256_fmadd_ps(y, x, *(__m256*)_ps256_cephes_log_p2);
    y = _mm256_fmadd_ps(y, x, *(__m256*)_ps256_cephes_log_p3);
    y = _mm256_fmadd_ps(y, x, *(__m256*)_ps256_cephes_log_p4);
    y = _mm256_fmadd_ps(y, x, *(__m256*)_ps256_cephes_log_p5);
    y = _mm256_fmadd_ps(y, x, *(__m256*)_ps256_cephes_log_p6);
    y = _mm256_fmadd_ps(y, x, *(__m256*)_ps256_cephes_log_p7);
    y = _mm256_fmadd_ps(y, x, *(__m256*)_ps256_cephes_log_p8);
    y = _mm256_mul_ps(y, x);
    y = _mm256_mul_ps(y, z);
    y = _mm256_fmadd_ps(e, *(__m256*)_ps256_cephes_log_q1, y);
    y = _mm256_fnmadd_ps(z, *(__m256*)_ps256_0p5, y);
    x = _mm256_add_ps(x, y);
    x = _mm256_fmadd_ps(e, *(__m256*)_ps256_cephes_log_q2, x);
    x = _mm256_or_ps(x, invalid_mask);
    return x;
}

_PS256_CONST(exp_hi, 88.3762626647949f);
_PS256_CONST(exp_lo, -88.3762626647949f);
_PS256_CONST(cephes_LOG2EF, 1.44269504088896341f);
_PS256_CONST(cephes_exp_C1, 0.693359375f);
_PS256_CONST(cephes_exp_C2, -2.12194440e-4f);
_PS256_CONST(cephes_exp_p0, 1.9875691500E-4f);
_PS256_CONST(cephes_exp_p1, 1.3981999507E-3f);
_PS256_CONST(cephes_exp_p2, 8.3334519073E-3f);
_PS256_CONST(cephes_exp_p3, 4.1665795894E-2f);
_PS256_CONST(cephes_exp_p4, 1.6666665459E-1f);
_PS256_CONST(cephes_exp_p5, 5.0000001201E-1f);

static inline __m256 exp256_ps(__m256 x)
{
    __m256 tmp = _mm256_setzero_ps(), fx;
    __m256i imm0;
    __m256 one = *(__m256*)_ps256_1;
    x = _mm256_min_ps(x, *(__m256*)_ps256_exp_hi);
    x = _mm256_max_ps(x, *(__m256*)_ps256_exp_lo);
    fx = _mm256_fmadd_ps(x, *(__m256*)_ps256_cephes_LOG2EF, *(__m256*)_ps256_0p5);
    tmp = _mm256_floor_ps(fx);
    __m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
    mask = _mm256_and_ps(mask, one);
    fx = _mm256_sub_ps(tmp, mask);
    x = _mm256_fnmadd_ps(fx, *(__m256*)_ps256_cephes_exp_C1, x);
    x = _mm256_fnmadd_ps(fx, *(__m256*)_ps256_cephes_exp_C2, x);
    tmp = _mm256_mul_ps(x, x);
    __m256 y = *(__m256*)_ps256_cephes_exp_p0;
    y = _mm256_fmadd_ps(y, x, *(__m256*)_ps256_cephes_exp_p1);
    y = _mm256_fmadd_ps(y, x, *(__m256*)_ps256_cephes_exp_p2);
    y = _mm256_fmadd_ps(y, x, *(__m256*)_ps256_cephes_exp_p3);
    y = _mm256_fmadd_ps(y, x, *(__m256*)_ps256_cephes_exp_p4);
    y = _mm256_fmadd_ps(y, x, *(__m256*)_ps256_cephes_exp_p5);
    y = _mm256_fmadd_ps(y, tmp, x);
    y = _mm256_add_ps(y, one);
    imm0 = _mm256_cvttps_epi32(fx);
    imm0 = _mm256_add_epi32(imm0, *(__m256i*)_pi32_256_0x7f);
    imm0 = _mm256_slli_epi32(imm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(imm0);
    y = _mm256_mul_ps(y, pow2n);
    return y;
}

_PS256_CONST(tanh_hi, 9.0f);
_PS256_CONST(tanh_lo, -9.0f);
_PS256_CONST(cephes_tanh_p0, -2.76076847742355E-16f);
_PS256_CONST(cephes_tanh_p1, 2.00018790482477E-13f);
_PS256_CONST(cephes_tanh_p2, -8.60467152213735E-11f);
_PS256_CONST(cephes_tanh_p3, 5.12229709037114E-08f);
_PS256_CONST(cephes_tanh_p4, 1.48572235717979E-05f);
_PS256_CONST(cephes_tanh_p5, 6.37261928875436E-04f);
_PS256_CONST(cephes_tanh_p6, 4.89352455891786E-03f);
_PS256_CONST(cephes_tanh_p7, 1.19825839466702e-06f);
_PS256_CONST(cephes_tanh_p8, 1.18534705686654e-04f);
_PS256_CONST(cephes_tanh_p9, 2.26843463243900e-03f);

static inline __m256 tanh256_ps(__m256 x)
{
    __m256 value = x;
    value = _mm256_max_ps(*(__m256*)_ps256_tanh_lo, value);
    value = _mm256_min_ps(*(__m256*)_ps256_tanh_hi, value);
    __m256 value_squared = _mm256_mul_ps(value, value);
    __m256 p;
    p = _mm256_fmadd_ps(value_squared, *(__m256*)_ps256_cephes_tanh_p0, *(__m256*)_ps256_cephes_tanh_p1);
    p = _mm256_fmadd_ps(p, value_squared, *(__m256*)_ps256_cephes_tanh_p2);
    p = _mm256_fmadd_ps(p, value_squared, *(__m256*)_ps256_cephes_tanh_p3);
    p = _mm256_fmadd_ps(p, value_squared, *(__m256*)_ps256_cephes_tanh_p4);
    p = _mm256_fmadd_ps(p, value_squared, *(__m256*)_ps256_cephes_tanh_p5);
    p = _mm256_fmadd_ps(p, value_squared, *(__m256*)_ps256_cephes_tanh_p6);
    p = _mm256_mul_ps(p, value);
    __m256 q;
    q = _mm256_fmadd_ps(value_squared, *(__m256*)_ps256_cephes_tanh_p7, *(__m256*)_ps256_cephes_tanh_p8);
    q = _mm256_fmadd_ps(q, value_squared, *(__m256*)_ps256_cephes_tanh_p9);
    q = _mm256_fmadd_ps(q, value_squared, *(__m256*)_ps256_cephes_tanh_p6);
    __m256 dst = _mm256_div_ps(p, q);
    return dst;
}

_PS256_CONST(minus_cephes_DP1, -0.78515625f);
_PS256_CONST(minus_cephes_DP2, -2.4187564849853515625e-4f);
_PS256_CONST(minus_cephes_DP3, -3.77489497744594108e-8f);
_PS256_CONST(sincof_p0, -1.9515295891E-4f);
_PS256_CONST(sincof_p1, 8.3321608736E-3f);
_PS256_CONST(sincof_p2, -1.6666654611E-1f);
_PS256_CONST(coscof_p0, 2.443315711809948E-005f);
_PS256_CONST(coscof_p1, -1.388731625493765E-003f);
_PS256_CONST(coscof_p2, 4.166664568298827E-002f);
_PS256_CONST(cephes_FOPI, 1.27323954473516f); // 4 / M_PI

static inline __m256 sin256_ps(__m256 x)
{
    __m256 xmm1, xmm2 = _mm256_setzero_ps(), xmm3, sign_bit, y;
    __m256i imm0, imm2;
    sign_bit = x;
    x = _mm256_and_ps(x, *(__m256*)_ps256_inv_sign_mask);
    sign_bit = _mm256_and_ps(sign_bit, *(__m256*)_ps256_sign_mask);
    y = _mm256_mul_ps(x, *(__m256*)_ps256_cephes_FOPI);
    imm2 = _mm256_cvttps_epi32(y);
    imm2 = _mm256_add_epi32(imm2, *(__m256i*)_pi32_256_1);
    imm2 = _mm256_and_si256(imm2, *(__m256i*)_pi32_256_inv1);
    y = _mm256_cvtepi32_ps(imm2);
    imm0 = _mm256_and_si256(imm2, *(__m256i*)_pi32_256_4);
    imm0 = _mm256_slli_epi32(imm0, 29);
    imm2 = _mm256_and_si256(imm2, *(__m256i*)_pi32_256_2);
    imm2 = _mm256_cmpeq_epi32(imm2, *(__m256i*)_pi32_256_0);
    __m256 swap_sign_bit = _mm256_castsi256_ps(imm0);
    __m256 poly_mask = _mm256_castsi256_ps(imm2);
    sign_bit = _mm256_xor_ps(sign_bit, swap_sign_bit);
    xmm1 = *(__m256*)_ps256_minus_cephes_DP1;
    xmm2 = *(__m256*)_ps256_minus_cephes_DP2;
    xmm3 = *(__m256*)_ps256_minus_cephes_DP3;
    x = _mm256_fmadd_ps(y, xmm1, x);
    x = _mm256_fmadd_ps(y, xmm2, x);
    x = _mm256_fmadd_ps(y, xmm3, x);
    y = *(__m256*)_ps256_coscof_p0;
    __m256 z = _mm256_mul_ps(x, x);
    y = _mm256_fmadd_ps(y, z, *(__m256*)_ps256_coscof_p1);
    y = _mm256_fmadd_ps(y, z, *(__m256*)_ps256_coscof_p2);
    y = _mm256_mul_ps(y, z);
    y = _mm256_mul_ps(y, z);
    y = _mm256_fnmadd_ps(z, *(__m256*)_ps256_0p5, y);
    y = _mm256_add_ps(y, *(__m256*)_ps256_1);
    __m256 y2 = *(__m256*)_ps256_sincof_p0;
    y2 = _mm256_fmadd_ps(y2, z, *(__m256*)_ps256_sincof_p1);
    y2 = _mm256_fmadd_ps(y2, z, *(__m256*)_ps256_sincof_p2);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_fmadd_ps(y2, x, x);
    xmm3 = poly_mask;
    y2 = _mm256_and_ps(xmm3, y2);
    y = _mm256_andnot_ps(xmm3, y);
    y = _mm256_add_ps(y, y2);
    y = _mm256_xor_ps(y, sign_bit);
    return y;
}

static inline __m256 cos256_ps(__m256 x)
{
    __m256 xmm1, xmm2 = _mm256_setzero_ps(), xmm3, y;
    __m256i imm0, imm2;
    x = _mm256_and_ps(x, *(__m256*)_ps256_inv_sign_mask);
    y = _mm256_mul_ps(x, *(__m256*)_ps256_cephes_FOPI);
    imm2 = _mm256_cvttps_epi32(y);
    imm2 = _mm256_add_epi32(imm2, *(__m256i*)_pi32_256_1);
    imm2 = _mm256_and_si256(imm2, *(__m256i*)_pi32_256_inv1);
    y = _mm256_cvtepi32_ps(imm2);
    imm2 = _mm256_sub_epi32(imm2, *(__m256i*)_pi32_256_2);
    imm0 = _mm256_andnot_si256(imm2, *(__m256i*)_pi32_256_4);
    imm0 = _mm256_slli_epi32(imm0, 29);
    imm2 = _mm256_and_si256(imm2, *(__m256i*)_pi32_256_2);
    imm2 = _mm256_cmpeq_epi32(imm2, *(__m256i*)_pi32_256_0);
    __m256 sign_bit = _mm256_castsi256_ps(imm0);
    __m256 poly_mask = _mm256_castsi256_ps(imm2);
    xmm1 = *(__m256*)_ps256_minus_cephes_DP1;
    xmm2 = *(__m256*)_ps256_minus_cephes_DP2;
    xmm3 = *(__m256*)_ps256_minus_cephes_DP3;
    x = _mm256_fmadd_ps(y, xmm1, x);
    x = _mm256_fmadd_ps(y, xmm2, x);
    x = _mm256_fmadd_ps(y, xmm3, x);
    y = *(__m256*)_ps256_coscof_p0;
    __m256 z = _mm256_mul_ps(x, x);
    y = _mm256_fmadd_ps(y, z, *(__m256*)_ps256_coscof_p1);
    y = _mm256_fmadd_ps(y, z, *(__m256*)_ps256_coscof_p2);
    y = _mm256_mul_ps(y, z);
    y = _mm256_mul_ps(y, z);
    y = _mm256_fnmadd_ps(z, *(__m256*)_ps256_0p5, y);
    y = _mm256_add_ps(y, *(__m256*)_ps256_1);
    __m256 y2 = *(__m256*)_ps256_sincof_p0;
    y2 = _mm256_fmadd_ps(y2, z, *(__m256*)_ps256_sincof_p1);
    y2 = _mm256_fmadd_ps(y2, z, *(__m256*)_ps256_sincof_p2);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_fmadd_ps(y2, x, x);
    xmm3 = poly_mask;
    y2 = _mm256_and_ps(xmm3, y2);
    y = _mm256_andnot_ps(xmm3, y);
    y = _mm256_add_ps(y, y2);
    y = _mm256_xor_ps(y, sign_bit);
    return y;
}

static inline void sincos256_ps(__m256 x, __m256* s, __m256* c)
{
    __m256 xmm1, xmm2, xmm3 = _mm256_setzero_ps(), sign_bit_sin, y;
    __m256i imm0, imm2, imm4;
    sign_bit_sin = x;
    x = _mm256_and_ps(x, *(__m256*)_ps256_inv_sign_mask);
    sign_bit_sin = _mm256_and_ps(sign_bit_sin, *(__m256*)_ps256_sign_mask);
    y = _mm256_mul_ps(x, *(__m256*)_ps256_cephes_FOPI);
    imm2 = _mm256_cvttps_epi32(y);
    imm2 = _mm256_add_epi32(imm2, *(__m256i*)_pi32_256_1);
    imm2 = _mm256_and_si256(imm2, *(__m256i*)_pi32_256_inv1);
    y = _mm256_cvtepi32_ps(imm2);
    imm4 = imm2;
    imm0 = _mm256_and_si256(imm2, *(__m256i*)_pi32_256_4);
    imm0 = _mm256_slli_epi32(imm0, 29);
    __m256 swap_sign_bit_sin = _mm256_castsi256_ps(imm0);
    imm2 = _mm256_and_si256(imm2, *(__m256i*)_pi32_256_2);
    imm2 = _mm256_cmpeq_epi32(imm2, *(__m256i*)_pi32_256_0);
    __m256 poly_mask = _mm256_castsi256_ps(imm2);
    xmm1 = *(__m256*)_ps256_minus_cephes_DP1;
    xmm2 = *(__m256*)_ps256_minus_cephes_DP2;
    xmm3 = *(__m256*)_ps256_minus_cephes_DP3;
    x = _mm256_fmadd_ps(y, xmm1, x);
    x = _mm256_fmadd_ps(y, xmm2, x);
    x = _mm256_fmadd_ps(y, xmm3, x);
    imm4 = _mm256_sub_epi32(imm4, *(__m256i*)_pi32_256_2);
    imm4 = _mm256_andnot_si256(imm4, *(__m256i*)_pi32_256_4);
    imm4 = _mm256_slli_epi32(imm4, 29);
    __m256 sign_bit_cos = _mm256_castsi256_ps(imm4);
    sign_bit_sin = _mm256_xor_ps(sign_bit_sin, swap_sign_bit_sin);
    __m256 z = _mm256_mul_ps(x, x);
    y = *(__m256*)_ps256_coscof_p0;
    y = _mm256_fmadd_ps(y, z, *(__m256*)_ps256_coscof_p1);
    y = _mm256_fmadd_ps(y, z, *(__m256*)_ps256_coscof_p2);
    y = _mm256_mul_ps(y, z);
    y = _mm256_mul_ps(y, z);
    y = _mm256_fnmadd_ps(z, *(__m256*)_ps256_0p5, y);
    y = _mm256_add_ps(y, *(__m256*)_ps256_1);
    __m256 y2 = *(__m256*)_ps256_sincof_p0;
    y2 = _mm256_fmadd_ps(y2, z, *(__m256*)_ps256_sincof_p1);
    y2 = _mm256_fmadd_ps(y2, z, *(__m256*)_ps256_sincof_p2);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_fmadd_ps(y2, x, x);
    xmm3 = poly_mask;
    __m256 ysin2 = _mm256_and_ps(xmm3, y2);
    __m256 ysin1 = _mm256_andnot_ps(xmm3, y);
    y2 = _mm256_sub_ps(y2, ysin2);
    y = _mm256_sub_ps(y, ysin1);
    xmm1 = _mm256_add_ps(ysin1, ysin2);
    xmm2 = _mm256_add_ps(y, y2);
    *s = _mm256_xor_ps(xmm1, sign_bit_sin);
    *c = _mm256_xor_ps(xmm2, sign_bit_cos);
}

static inline __m256 tan256_ps(__m256 x)
{
    __m256 ysin, ycos;
    sincos256_ps(x, &ysin, &ycos);
    __m256 mask = _mm256_cmp_ps(ycos, _mm256_setzero_ps(), _CMP_EQ_OS);
    __m256 eps = _mm256_set1_ps(1E-8f);
    ycos = _mm256_add_ps(ycos, _mm256_and_ps(eps, mask));
    return _mm256_div_ps(ysin, ycos);
}

#else  // _MSC_VER: use Intel SVML

static inline __m256 log256_ps(__m256 x)   { return _mm256_log_ps(x); }
static inline __m256 exp256_ps(__m256 x)   { return _mm256_exp_ps(x); }
static inline __m256 sin256_ps(__m256 x)   { return _mm256_sin_ps(x); }
static inline __m256 cos256_ps(__m256 x)   { return _mm256_cos_ps(x); }
static inline __m256 tan256_ps(__m256 x)   { return _mm256_tan_ps(x); }
static inline __m256 tanh256_ps(const __m256 x) { return _mm256_tanh_ps(x); }

#endif  // _MSC_VER

} // namespace avx2
} // namespace arch
} // namespace simd
} // namespace nnops
