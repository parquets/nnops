#pragma once
/// @file vec_f32x4.hpp
/// @brief Unified 128-bit float32 vector type and operations.
///
/// At compile time, selects the best available backend:
///   - x86_64:         SSE4.1 (always available, __m128)
///   - AArch64:        NEON (always available, float32x4_t)
///   - everything else: scalar fallback
///
/// Usage:
///   #include "nnops/detail/simd/vec_f32x4.hpp"
///   using namespace nnops::simd;
///   v_f32x4 a = v_load_f32x4(ptr);
///   v_f32x4 r = v_fmadd(a, b, c);
///   v_store(out, r);

#if defined(NNOPS_ARCH_X86_64)
  #include "nnops/detail/simd/arch/x86/sse.hpp"
  namespace nnops { namespace simd {
  using v_f32x4 = arch::sse::v_f32x4;
  using arch::sse::v_load_f32x4;
  using arch::sse::v_store;
  using arch::sse::v_set1_f32x4;
  using arch::sse::v_zero_f32x4;
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
  using arch::sse::v_cmple;
  using arch::sse::v_cmpgt;
  using arch::sse::v_cmpge;
  using arch::sse::v_ceq;
  using arch::sse::v_and;
  using arch::sse::v_or;
  using arch::sse::v_sqrt;
  using arch::sse::v_rcp;
  using arch::sse::v_rsqrt;
  using arch::sse::v_exp;
  using arch::sse::v_log;
  using arch::sse::v_sin;
  using arch::sse::v_cos;
  using arch::sse::v_tan;
  using arch::sse::v_tanh;
  using arch::sse::v_reduce_sum;
  using arch::sse::v_reduce_max;
  using arch::sse::v_reduce_min;
  }} // namespace nnops::simd

#elif defined(NNOPS_ARCH_AARCH64)
  #include "nnops/detail/simd/arch/arm/neon.hpp"
  namespace nnops { namespace simd {
  using v_f32x4 = arch::neon::v_f32x4;
  using arch::neon::v_load_f32x4;
  using arch::neon::v_store;
  using arch::neon::v_set1_f32x4;
  using arch::neon::v_zero_f32x4;
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
  using arch::neon::v_cmple;
  using arch::neon::v_cmpgt;
  using arch::neon::v_cmpge;
  using arch::neon::v_ceq;
  using arch::neon::v_and;
  using arch::neon::v_or;
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
  }} // namespace nnops::simd

#else
  // Fallback: architecture with no known SIMD
  #include "nnops/detail/simd/arch/scalar.hpp"
  namespace nnops { namespace simd {
  using v_f32x4 = arch::scalar::v_f32x4;
  using arch::scalar::v_load_f32x4;
  using arch::scalar::v_store;
  using arch::scalar::v_set1_f32x4;
  using arch::scalar::v_zero_f32x4;
  using arch::scalar::v_add;
  using arch::scalar::v_sub;
  using arch::scalar::v_mul;
  using arch::scalar::v_div;
  using arch::scalar::v_fmadd;
  using arch::scalar::v_min;
  using arch::scalar::v_max;
  using arch::scalar::v_abs;
  using arch::scalar::v_neg;
  using arch::scalar::v_cmplt;
  using arch::scalar::v_cmpgt;
  using arch::scalar::v_and;
  using arch::scalar::v_sqrt;
  using arch::scalar::v_exp;
  using arch::scalar::v_log;
  using arch::scalar::v_rcp;
  using arch::scalar::v_sin;
  using arch::scalar::v_cos;
  using arch::scalar::v_tan;
  using arch::scalar::v_tanh;
  using arch::scalar::v_reduce_sum;
  using arch::scalar::v_reduce_max;
  using arch::scalar::v_reduce_min;
  }} // namespace nnops::simd
#endif
