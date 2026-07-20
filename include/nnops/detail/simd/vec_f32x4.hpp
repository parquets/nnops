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
///   v_fp32x4 a = load_fp32x4(ptr);
///   v_fp32x4 r = fmadd(a, b, c);
///   store(out, r);

#if defined(NNOPS_ARCH_X86_64)
  #include "nnops/detail/simd/arch/x86/sse.hpp"
  namespace nnops { namespace simd {
  using v_fp32x4 = arch::sse::v_fp32x4;
  using arch::sse::load_fp32x4;
  using arch::sse::store;
  using arch::sse::set1_fp32x4;
  using arch::sse::zero_fp32x4;
  using arch::sse::add;
  using arch::sse::sub;
  using arch::sse::mul;
  using arch::sse::div;
  using arch::sse::fmadd;
  using arch::sse::min;
  using arch::sse::max;
  using arch::sse::abs;
  using arch::sse::neg;
  using arch::sse::cmplt;
  using arch::sse::cmple;
  using arch::sse::cmpgt;
  using arch::sse::cmpge;
  using arch::sse::ceq;
  using arch::sse::and_;
  using arch::sse::or_;
  using arch::sse::sqrt;
  using arch::sse::rcp;
  using arch::sse::rsqrt;
  using arch::sse::reduce_sum;
  }} // namespace nnops::simd

#elif defined(NNOPS_ARCH_AARCH64)
  #include "nnops/detail/simd/arch/arm/neon.hpp"
  namespace nnops { namespace simd {
  using v_fp32x4 = arch::neon::v_fp32x4;
  using arch::neon::load_fp32x4;
  using arch::neon::store;
  using arch::neon::set1_fp32x4;
  using arch::neon::zero_fp32x4;
  using arch::neon::add;
  using arch::neon::sub;
  using arch::neon::mul;
  using arch::neon::div;
  using arch::neon::fmadd;
  using arch::neon::min;
  using arch::neon::max;
  using arch::neon::abs;
  using arch::neon::neg;
  using arch::neon::cmplt;
  using arch::neon::cmple;
  using arch::neon::cmpgt;
  using arch::neon::cmpge;
  using arch::neon::ceq;
  using arch::neon::and_;
  using arch::neon::or_;
  using arch::neon::sqrt;
  using arch::neon::rcp;
  using arch::neon::reduce_sum;
  }} // namespace nnops::simd

#else
  // Fallback: architecture with no known SIMD
  #include "nnops/detail/simd/arch/scalar.hpp"
  namespace nnops { namespace simd {
  using v_fp32x4 = arch::scalar::v_fp32x4;
  using arch::scalar::load_fp32x4;
  using arch::scalar::store;
  using arch::scalar::set1_fp32x4;
  using arch::scalar::zero_fp32x4;
  using arch::scalar::add;
  using arch::scalar::sub;
  using arch::scalar::mul;
  using arch::scalar::div;
  using arch::scalar::fmadd;
  using arch::scalar::min;
  using arch::scalar::max;
  using arch::scalar::abs;
  using arch::scalar::neg;
  using arch::scalar::cmplt;
  using arch::scalar::cmpgt;
  using arch::scalar::and_;
  using arch::scalar::sqrt;
  using arch::scalar::exp;
  using arch::scalar::reduce_sum;
  }} // namespace nnops::simd
#endif
