#pragma once
/// @file vec_s8x16.hpp
/// @brief Unified 128-bit signed int8 vector type and operations (16 int8s).
///
/// At compile time, selects the best available backend:
///   - x86_64:         SSE4.1 (always available, __m128i)
///   - AArch64:        NEON (always available, int8x16_t)
///   - everything else: scalar fallback
///
/// Usage:
///   #include "nnops/detail/simd/vec_s8x16.hpp"
///   using namespace nnops::simd;
///   v_s8x16 a = v_load_s8x16(ptr);
///   v_store(out, a);

#if defined(NNOPS_ARCH_X86_64)
  #include "nnops/detail/simd/arch/x86/sse.hpp"
  namespace nnops { namespace simd {
  using v_s8x16 = arch::sse::v_s8x16;
  using arch::sse::v_load_s8x16;
  using arch::sse::v_set1_s8x16;
  using arch::sse::v_zero_s8x16;
  }} // namespace nnops::simd

#elif defined(NNOPS_ARCH_AARCH64)
  #include "nnops/detail/simd/arch/arm/neon.hpp"
  namespace nnops { namespace simd {
  using v_s8x16 = arch::neon::v_s8x16;
  using arch::neon::v_load_s8x16;
  using arch::neon::v_set1_s8x16;
  using arch::neon::v_zero_s8x16;
  }} // namespace nnops::simd

#else
  // Fallback: architecture with no known SIMD
  #include "nnops/detail/simd/arch/scalar.hpp"
  namespace nnops { namespace simd {
  using v_s8x16 = arch::scalar::v_s8x16;
  using arch::scalar::v_load_s8x16;
  using arch::scalar::v_set1_s8x16;
  using arch::scalar::v_zero_s8x16;
  }} // namespace nnops::simd
#endif
