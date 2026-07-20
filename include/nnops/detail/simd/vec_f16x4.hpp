#pragma once
/// @file vec_f16x4.hpp
/// @brief Unified 64-bit float16 vector type and operations.
///
/// At compile time, selects the best available backend:
///   - AArch64 + __ARM_FEATURE_FP16_VECTOR_ARITHMETIC:  native NEON (float16x4_t)
///   - everything else:                                scalar fallback (uint16_t)
///
/// The public API uses uint16_t for load/store (raw IEEE 754 binary16 bits)
/// and float for scalar interchange (set1, reduce_sum return types).
///
/// Usage:
///   #include "nnops/detail/simd/vec_f16x4.hpp"
///   using namespace nnops::simd;
///   v_fp16x4 a = load_fp16x4(ptr);
///   v_fp32x4 fa = cvt_f16_to_f32(a);   // widen to f32 for compute
///   v_fp16x4 r = cvt_f32_to_f16(fa);   // narrow back to f16

#if defined(NNOPS_ARCH_AARCH64) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  #include "nnops/detail/simd/arch/arm/neon.hpp"
  namespace nnops { namespace simd {
  using v_fp16x4 = arch::neon::v_fp16x4;
  using arch::neon::load_fp16x4;
  using arch::neon::store;
  using arch::neon::set1_fp16x4;
  using arch::neon::zero_fp16x4;
  using arch::neon::cvt_f16_to_f32;
  using arch::neon::cvt_f32_to_f16;
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
  using arch::neon::cmpgt;
  using arch::neon::sqrt;
  using arch::neon::reduce_sum;
  }} // namespace nnops::simd
#else
  #include "nnops/detail/simd/arch/scalar.hpp"
  namespace nnops { namespace simd {
  using v_fp16x4 = arch::scalar::v_fp16x4;
  using arch::scalar::load_fp16x4;
  using arch::scalar::store;
  using arch::scalar::set1_fp16x4;
  using arch::scalar::zero_fp16x4;
  using arch::scalar::cvt_f16_to_f32;
  using arch::scalar::cvt_f32_to_f16;
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
  using arch::scalar::sqrt;
  using arch::scalar::reduce_sum;
  }} // namespace nnops::simd
#endif
