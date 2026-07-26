#pragma once
/// @file vec_f32x8.hpp
/// @brief Unified 256-bit float32 vector type and operations.
///
/// Selection strategy (compile-time):
///   - x86_64 + __AVX__ defined: AVX2+FMA native (__m256), v_f32x4 via low 128 bits
///   - x86_64 without AVX:     SSE4.1 emulated (two __m128), v_f32x4 via sse
///   - AArch64:                NEON emulated (two float32x4_t)
///   - otherwise:              scalar fallback
///
/// For runtime dispatch of AVX2 kernels, compile the consuming .cpp with /arch:AVX2
/// (or -mavx2 -mfma) and guard the call site with CpuFeatures::get().has(CpuIsa::AVX2).

#if defined(NNOPS_ARCH_X86_64)

  #if defined(__AVX__)  // /arch:AVX2 or -mavx2 was passed
    #include "nnops/detail/simd/arch/x86/avx2.hpp"
    namespace nnops { namespace simd {
    using v_f32x8 = arch::avx2::v_f32x8;
    using arch::avx2::v_load_f32x8;
    using arch::avx2::v_store;
    using arch::avx2::v_set1_f32x8;
    using arch::avx2::v_zero_f32x8;
    using arch::avx2::v_add;
    using arch::avx2::v_sub;
    using arch::avx2::v_mul;
    using arch::avx2::v_div;
    using arch::avx2::v_fmadd;
    using arch::avx2::v_fmsub;
    using arch::avx2::v_min;
    using arch::avx2::v_max;
    using arch::avx2::v_abs;
    using arch::avx2::v_neg;
    using arch::avx2::v_cmplt;
    using arch::avx2::v_cmple;
    using arch::avx2::v_cmpgt;
    using arch::avx2::v_ceq;
    using arch::avx2::v_and;
    using arch::avx2::v_or;
    using arch::avx2::v_sqrt;
    using arch::avx2::v_rcp;
    using arch::avx2::v_rsqrt;
    using arch::avx2::v_exp;
    using arch::avx2::v_log;
    using arch::avx2::v_sin;
    using arch::avx2::v_cos;
    using arch::avx2::v_tan;
    using arch::avx2::v_tanh;
    using arch::avx2::v_reduce_sum;
    using arch::avx2::v_reduce_max;
    using arch::avx2::v_reduce_min;
    }} // namespace nnops::simd
  #else
    // x86_64 without AVX: emulate with two SSE registers
    #include "nnops/detail/simd/arch/x86/sse.hpp"
    namespace nnops { namespace simd {
    using v_f32x8 = arch::sse::v_f32x8;
    using arch::sse::v_load_f32x8;
    using arch::sse::v_store;
    using arch::sse::v_set1_f32x8;
    using arch::sse::v_zero_f32x8;
    using arch::sse::v_add;
    using arch::sse::v_sub;
    using arch::sse::v_mul;
    using arch::sse::v_div;
    using arch::sse::v_fmadd;
    using arch::sse::v_min;
    using arch::sse::v_max;
    using arch::sse::v_sqrt;
    using arch::sse::v_abs;
    using arch::sse::v_neg;
    using arch::sse::v_rcp;
    using arch::sse::v_cmplt;
    using arch::sse::v_cmpgt;
    using arch::sse::v_and;
    using arch::sse::v_or;
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
  #endif  // __AVX__

#elif defined(NNOPS_ARCH_AARCH64)
  #include "nnops/detail/simd/arch/arm/neon.hpp"
  namespace nnops { namespace simd {
  using v_f32x8 = arch::neon::v_f32x8;
  using arch::neon::v_load_f32x8;
  using arch::neon::v_store;
  using arch::neon::v_set1_f32x8;
  using arch::neon::v_zero_f32x8;
  using arch::neon::v_add;
  using arch::neon::v_sub;
  using arch::neon::v_mul;
  using arch::neon::v_fmadd;
  using arch::neon::v_min;
  using arch::neon::v_max;
  using arch::neon::v_sqrt;
  using arch::neon::v_abs;
  using arch::neon::v_neg;
  using arch::neon::v_div;
  using arch::neon::v_rcp;
  using arch::neon::v_cmplt;
  using arch::neon::v_cmpgt;
  using arch::neon::v_and;
  using arch::neon::v_or;
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
  #include "nnops/detail/simd/arch/scalar.hpp"
  namespace nnops { namespace simd {
  using v_f32x8 = arch::scalar::v_f32x8;
  using arch::scalar::v_load_f32x8;
  using arch::scalar::v_store;
  using arch::scalar::v_set1_f32x8;
  using arch::scalar::v_zero_f32x8;
  using arch::scalar::v_add;
  using arch::scalar::v_sub;
  using arch::scalar::v_mul;
  using arch::scalar::v_div;
  using arch::scalar::v_fmadd;
  using arch::scalar::v_min;
  using arch::scalar::v_max;
  using arch::scalar::v_sqrt;
  using arch::scalar::v_exp;
  using arch::scalar::v_log;
  using arch::scalar::v_sin;
  using arch::scalar::v_cos;
  using arch::scalar::v_tan;
  using arch::scalar::v_tanh;
  using arch::scalar::v_abs;
  using arch::scalar::v_neg;
  using arch::scalar::v_rcp;
  using arch::scalar::v_and;
  using arch::scalar::v_or;
  using arch::scalar::v_reduce_sum;
  using arch::scalar::v_reduce_max;
  using arch::scalar::v_reduce_min;
  }} // namespace nnops::simd
#endif
