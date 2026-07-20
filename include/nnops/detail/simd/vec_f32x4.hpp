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
///   VecF32x4 a = vec_load_f32x4(ptr);
///   VecF32x4 r = vec_fmadd_f32x4(a, b, c);
///   vec_store_f32x4(out, r);

#if defined(NNOPS_ARCH_X86_64)
  #include "nnops/detail/simd/arch/x86/sse.hpp"
  namespace nnops { namespace simd {
  using VecF32x4 = arch::sse::VecF32x4;
  using arch::sse::vec_load_f32x4;
  using arch::sse::vec_store_f32x4;
  using arch::sse::vec_set1_f32x4;
  using arch::sse::vec_zero_f32x4;
  using arch::sse::vec_add_f32x4;
  using arch::sse::vec_sub_f32x4;
  using arch::sse::vec_mul_f32x4;
  using arch::sse::vec_div_f32x4;
  using arch::sse::vec_fmadd_f32x4;
  using arch::sse::vec_min_f32x4;
  using arch::sse::vec_max_f32x4;
  using arch::sse::vec_abs_f32x4;
  using arch::sse::vec_neg_f32x4;
  using arch::sse::vec_cmplt_f32x4;
  using arch::sse::vec_cmple_f32x4;
  using arch::sse::vec_cmpgt_f32x4;
  using arch::sse::vec_cmpge_f32x4;
  using arch::sse::vec_ceq_f32x4;
  using arch::sse::vec_and_f32x4;
  using arch::sse::vec_or_f32x4;
  using arch::sse::vec_sqrt_f32x4;
  using arch::sse::vec_rcp_f32x4;
  using arch::sse::vec_rsqrt_f32x4;
  using arch::sse::vec_reduce_sum_f32x4;
  }} // namespace nnops::simd

#elif defined(NNOPS_ARCH_AARCH64)
  #include "nnops/detail/simd/arch/arm/neon.hpp"
  namespace nnops { namespace simd {
  using VecF32x4 = arch::neon::VecF32x4;
  using arch::neon::vec_load_f32x4;
  using arch::neon::vec_store_f32x4;
  using arch::neon::vec_set1_f32x4;
  using arch::neon::vec_zero_f32x4;
  using arch::neon::vec_add_f32x4;
  using arch::neon::vec_sub_f32x4;
  using arch::neon::vec_mul_f32x4;
  using arch::neon::vec_div_f32x4;
  using arch::neon::vec_fmadd_f32x4;
  using arch::neon::vec_min_f32x4;
  using arch::neon::vec_max_f32x4;
  using arch::neon::vec_abs_f32x4;
  using arch::neon::vec_neg_f32x4;
  using arch::neon::vec_cmplt_f32x4;
  using arch::neon::vec_cmple_f32x4;
  using arch::neon::vec_cmpgt_f32x4;
  using arch::neon::vec_cmpge_f32x4;
  using arch::neon::vec_ceq_f32x4;
  using arch::neon::vec_and_f32x4;
  using arch::neon::vec_or_f32x4;
  using arch::neon::vec_sqrt_f32x4;
  using arch::neon::vec_rcp_f32x4;
  using arch::neon::vec_reduce_sum_f32x4;
  }} // namespace nnops::simd

#else
  // Fallback: architecture with no known SIMD
  #include "nnops/detail/simd/arch/scalar.hpp"
  namespace nnops { namespace simd {
  using VecF32x4 = arch::scalar::VecF32x4;
  using arch::scalar::vec_load_f32x4;
  using arch::scalar::vec_store_f32x4;
  using arch::scalar::vec_set1_f32x4;
  using arch::scalar::vec_zero_f32x4;
  using arch::scalar::vec_add_f32x4;
  using arch::scalar::vec_sub_f32x4;
  using arch::scalar::vec_mul_f32x4;
  using arch::scalar::vec_div_f32x4;
  using arch::scalar::vec_fmadd_f32x4;
  using arch::scalar::vec_min_f32x4;
  using arch::scalar::vec_max_f32x4;
  using arch::scalar::vec_abs_f32x4;
  using arch::scalar::vec_neg_f32x4;
  using arch::scalar::vec_cmplt_f32x4;
  using arch::scalar::vec_cmpgt_f32x4;
  using arch::scalar::vec_and_f32x4;
  using arch::scalar::vec_sqrt_f32x4;
  using arch::scalar::vec_exp_f32x4;
  using arch::scalar::vec_reduce_sum_f32x4;
  }} // namespace nnops::simd
#endif
