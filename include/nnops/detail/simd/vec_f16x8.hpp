#pragma once
/// @file vec_f16x8.hpp
/// @brief Unified 128-bit float16 vector type and operations (8 half floats).
///
/// At compile time, selects the best available backend:
///   - AArch64:   native NEON (float16x8_t)
///   - x86_64 + __F16C__:                               F16C/AVX2 (__m128i + convert→compute→convert)
///   - everything else:                                 scalar fallback (uint16_t[8])
///
/// The public API uses uint16_t for load/v_store (raw IEEE 754 binary16 bits)
/// and float for scalar interchange.
///
/// Usage:
///   #include "nnops/detail/simd/vec_f16x8.hpp"
///   using namespace nnops::simd;
///   v_f16x8 a = v_load_f16x8(ptr);
///   v_f32x8 fa = v_cvt_f16_to_f32(a);   // widen to f32x8 for compute
///   v_f16x8 r = v_cvt_f32_to_f16(fa);   // narrow back to f16

#if defined(NNOPS_ARCH_AARCH64)
  #include "nnops/detail/simd/arch/arm/neon.hpp"
  namespace nnops { namespace simd {
  using v_f16x8 = arch::neon::v_f16x8;
  using arch::neon::v_load_f16x8;
  using arch::neon::v_store;
  using arch::neon::v_set1_f16x8;
  using arch::neon::v_zero_f16x8;
  using arch::neon::v_cvt_f16_to_f32;
  using arch::neon::v_cvt_f32_to_f16;
  using arch::neon::v_add;
  using arch::neon::v_sub;
  using arch::neon::v_mul;
  using arch::neon::v_div;
  using arch::neon::v_fmadd;
  using arch::neon::v_min;
  using arch::neon::v_max;
  using arch::neon::v_abs;
  using arch::neon::v_neg;
  using arch::neon::v_cmplt;
  using arch::neon::v_cmpgt;
  using arch::neon::v_sqrt;
  using arch::neon::v_rcp;
  using arch::neon::v_exp;
  using arch::neon::v_log;
  using arch::neon::v_sin;
  using arch::neon::v_cos;
  using arch::neon::v_tan;
  using arch::neon::v_tanh;
  using arch::neon::v_reduce_sum;
  using arch::neon::v_reduce_max;
  using arch::neon::v_reduce_min;
  // deinterleave
  using arch::neon::v_f16x8x2_t;
  using arch::neon::v_deinterleave_f16x8;
  using arch::neon::v_load_even_f16x8;
  using arch::neon::v_load_odd_f16x8;
  using arch::neon::v_load_stride2_even_f16x8;
  using arch::neon::v_load_stride2_odd_f16x8;
  // transpose
  using arch::neon::v_transpose_8x8;
  }} // namespace nnops::simd
#elif defined(NNOPS_ARCH_X86_64) && defined(__F16C__)
  // x86_64: F16C + AVX2 (__m128i storage, convert→compute→convert via _mm256_cvtph_ps / _mm256_cvtps_ph)
  #include "nnops/detail/simd/arch/x86/sse.hpp"
  namespace nnops { namespace simd {
  using v_f16x8 = arch::sse::v_f16x8;
  using arch::sse::v_load_f16x8;
  using arch::sse::v_store;
  using arch::sse::v_set1_f16x8;
  using arch::sse::v_zero_f16x8;
  using arch::sse::v_cvt_f16_to_f32;
  using arch::sse::v_cvt_f32_to_f16;
  using arch::sse::v_add;
  using arch::sse::v_sub;
  using arch::sse::v_mul;
  using arch::sse::v_div;
  using arch::sse::v_fmadd;
  using arch::sse::v_min;
  using arch::sse::v_max;
  using arch::sse::v_abs;
  using arch::sse::v_neg;
  using arch::sse::v_cmplt;
  using arch::sse::v_cmpgt;
  using arch::sse::v_sqrt;
  using arch::sse::v_rcp;
  using arch::sse::v_exp;
  using arch::sse::v_log;
  using arch::sse::v_sin;
  using arch::sse::v_cos;
  using arch::sse::v_tan;
  using arch::sse::v_tanh;
  using arch::sse::v_reduce_sum;
  using arch::sse::v_reduce_max;
  using arch::sse::v_reduce_min;
  // deinterleave
  using arch::sse::v_f16x8x2_t;
  using arch::sse::v_deinterleave_f16x8;
  using arch::sse::v_load_even_f16x8;
  using arch::sse::v_load_odd_f16x8;
  using arch::sse::v_load_stride2_even_f16x8;
  using arch::sse::v_load_stride2_odd_f16x8;
  // transpose
  using arch::sse::v_transpose_8x8;
  }} // namespace nnops::simd
#else
  // Scalar fallback
  #include "nnops/detail/simd/arch/scalar.hpp"
  namespace nnops { namespace simd {
  using v_f16x8 = arch::scalar::v_f16x8;
  using arch::scalar::v_load_f16x8;
  using arch::scalar::v_store;
  using arch::scalar::v_set1_f16x8;
  using arch::scalar::v_zero_f16x8;
  using arch::scalar::v_cvt_f16_to_f32;
  using arch::scalar::v_cvt_f32_to_f16;
  using arch::scalar::v_add;
  using arch::scalar::v_sub;
  using arch::scalar::v_mul;
  using arch::scalar::v_div;
  using arch::scalar::v_fmadd;
  using arch::scalar::v_min;
  using arch::scalar::v_max;
  using arch::scalar::v_abs;
  using arch::scalar::v_neg;
  using arch::scalar::v_sqrt;
  using arch::scalar::v_rcp;
  using arch::scalar::v_exp;
  using arch::scalar::v_log;
  using arch::scalar::v_sin;
  using arch::scalar::v_cos;
  using arch::scalar::v_tan;
  using arch::scalar::v_tanh;
  using arch::scalar::v_reduce_sum;
  using arch::scalar::v_reduce_max;
  using arch::scalar::v_reduce_min;
  // deinterleave
  using arch::scalar::v_f16x8x2_t;
  using arch::scalar::v_deinterleave_f16x8;
  using arch::scalar::v_load_even_f16x8;
  using arch::scalar::v_load_odd_f16x8;
  using arch::scalar::v_load_stride2_even_f16x8;
  using arch::scalar::v_load_stride2_odd_f16x8;
  // transpose
  using arch::scalar::v_transpose_8x8;
  }} // namespace nnops::simd
#endif
