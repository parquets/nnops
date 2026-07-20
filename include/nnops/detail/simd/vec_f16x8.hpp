#pragma once
/// @file vec_f16x8.hpp
/// @brief Unified 128-bit float16 vector type and operations (8 half floats).
///
/// At compile time, selects the best available backend:
///   - AArch64 + __ARM_FEATURE_FP16_VECTOR_ARITHMETIC:  native NEON (float16x8_t)
///   - everything else:                                scalar fallback (uint16_t[8])
///
/// The public API uses uint16_t for load/store (raw IEEE 754 binary16 bits)
/// and float for scalar interchange.
///
/// Usage:
///   #include "nnops/detail/simd/vec_f16x8.hpp"
///   using namespace nnops::simd;
///   v_fp16x8 a = load_fp16x8(ptr);
///   v_fp32x8 fa = cvt_f16_to_f32(a);   // widen to f32x8 for compute
///   v_fp16x8 r = cvt_f32_to_f16(fa);   // narrow back to f16

#if defined(NNOPS_ARCH_AARCH64) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  #include "nnops/detail/simd/arch/arm/neon.hpp"
  namespace nnops { namespace simd {
  using v_fp16x8 = arch::neon::v_fp16x8;
  using arch::neon::load_fp16x8;
  using arch::neon::store;
  using arch::neon::set1_fp16x8;
  using arch::neon::zero_fp16x8;
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
  using v_fp16x8 = arch::scalar::v_fp16x8;
  using arch::scalar::load_fp16x8;
  using arch::scalar::store;
  using arch::scalar::set1_fp16x8;
  using arch::scalar::zero_fp16x8;
  using arch::scalar::cvt_f16_to_f32;
  using arch::scalar::cvt_f32_to_f16;
  using arch::scalar::add;
  using arch::scalar::sub;
  using arch::scalar::mul;
  using arch::scalar::div;
  using arch::scalar::fmadd;
  using arch::scalar::min;
  using arch::scalar::max;
  using arch::scalar::sqrt;
  using arch::scalar::reduce_sum;
  }} // namespace nnops::simd
#endif
