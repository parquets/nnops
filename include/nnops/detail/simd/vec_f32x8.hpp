#pragma once
/// @file vec_f32x8.hpp
/// @brief Unified 256-bit float32 vector type and operations.
///
/// Selection strategy (compile-time):
///   - x86_64 + __AVX__ defined: AVX2+FMA native (__m256), v_fp32x4 via low 128 bits
///   - x86_64 without AVX:     SSE4.1 emulated (two __m128), v_fp32x4 via sse
///   - AArch64:                NEON emulated (two float32x4_t)
///   - otherwise:              scalar fallback
///
/// For runtime dispatch of AVX2 kernels, compile the consuming .cpp with /arch:AVX2
/// (or -mavx2 -mfma) and guard the call site with CpuFeatures::get().has(CpuIsa::AVX2).

#if defined(NNOPS_ARCH_X86_64)

  #if defined(__AVX__)  // /arch:AVX2 or -mavx2 was passed
    #include "nnops/detail/simd/arch/x86/avx2.hpp"
    namespace nnops { namespace simd {
    using v_fp32x8 = arch::avx2::v_fp32x8;
    using arch::avx2::load_fp32x8;
    using arch::avx2::store;
    using arch::avx2::set1_fp32x8;
    using arch::avx2::zero_fp32x8;
    using arch::avx2::add;
    using arch::avx2::sub;
    using arch::avx2::mul;
    using arch::avx2::div;
    using arch::avx2::fmadd;
    using arch::avx2::fmsub;
    using arch::avx2::min;
    using arch::avx2::max;
    using arch::avx2::abs;
    using arch::avx2::neg;
    using arch::avx2::cmplt;
    using arch::avx2::cmple;
    using arch::avx2::cmpgt;
    using arch::avx2::ceq;
    using arch::avx2::and_;
    using arch::avx2::or_;
    using arch::avx2::sqrt;
    using arch::avx2::rcp;
    using arch::avx2::rsqrt;
    using arch::avx2::reduce_sum;
    }} // namespace nnops::simd
  #else
    // x86_64 without AVX: emulate with two SSE registers
    #include "nnops/detail/simd/arch/x86/sse.hpp"
    namespace nnops { namespace simd {
    using v_fp32x8 = arch::sse::v_fp32x8;
    using arch::sse::load_fp32x8;
    using arch::sse::store;
    using arch::sse::set1_fp32x8;
    using arch::sse::zero_fp32x8;
    using arch::sse::add;
    using arch::sse::sub;
    using arch::sse::mul;
    using arch::sse::div;
    using arch::sse::fmadd;
    using arch::sse::min;
    using arch::sse::max;
    using arch::sse::sqrt;
    using arch::sse::reduce_sum;
    }} // namespace nnops::simd
  #endif  // __AVX__

#elif defined(NNOPS_ARCH_AARCH64)
  #include "nnops/detail/simd/arch/arm/neon.hpp"
  namespace nnops { namespace simd {
  using v_fp32x8 = arch::neon::v_fp32x8;
  using arch::neon::load_fp32x8;
  using arch::neon::store;
  using arch::neon::set1_fp32x8;
  using arch::neon::zero_fp32x8;
  using arch::neon::add;
  using arch::neon::sub;
  using arch::neon::mul;
  using arch::neon::fmadd;
  using arch::neon::min;
  using arch::neon::max;
  using arch::neon::sqrt;
  using arch::neon::reduce_sum;
  }} // namespace nnops::simd

#else
  #include "nnops/detail/simd/arch/scalar.hpp"
  namespace nnops { namespace simd {
  using v_fp32x8 = arch::scalar::v_fp32x8;
  using arch::scalar::load_fp32x8;
  using arch::scalar::store;
  using arch::scalar::set1_fp32x8;
  using arch::scalar::zero_fp32x8;
  using arch::scalar::add;
  using arch::scalar::sub;
  using arch::scalar::mul;
  using arch::scalar::div;
  using arch::scalar::fmadd;
  using arch::scalar::min;
  using arch::scalar::max;
  using arch::scalar::sqrt;
  using arch::scalar::exp;
  using arch::scalar::reduce_sum;
  }} // namespace nnops::simd
#endif
