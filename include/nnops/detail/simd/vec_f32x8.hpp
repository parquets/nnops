#pragma once
/// @file vec_f32x8.hpp
/// @brief Unified 256-bit float32 vector type and operations.
///
/// Selection strategy (compile-time):
///   - x86_64 + __AVX__ defined: AVX2+FMA native (__m256), VecF32x4 via low 128 bits
///   - x86_64 without AVX:     SSE4.1 emulated (two __m128), VecF32x4 via sse
///   - AArch64:                NEON emulated (two float32x4_t)
///   - otherwise:              scalar fallback
///
/// For runtime dispatch of AVX2 kernels, compile the consuming .cpp with /arch:AVX2
/// (or -mavx2 -mfma) and guard the call site with CpuFeatures::get().has(CpuIsa::AVX2).

#if defined(NNOPS_ARCH_X86_64)

  #if defined(__AVX__)  // /arch:AVX2 or -mavx2 was passed
    #include "nnops/detail/simd/arch/x86/avx2.hpp"
    namespace nnops { namespace simd {
    using VecF32x8 = arch::avx2::VecF32x8;
    using arch::avx2::vec_load_f32x8;
    using arch::avx2::vec_store_f32x8;
    using arch::avx2::vec_set1_f32x8;
    using arch::avx2::vec_zero_f32x8;
    using arch::avx2::vec_add_f32x8;
    using arch::avx2::vec_sub_f32x8;
    using arch::avx2::vec_mul_f32x8;
    using arch::avx2::vec_div_f32x8;
    using arch::avx2::vec_fmadd_f32x8;
    using arch::avx2::vec_fmsub_f32x8;
    using arch::avx2::vec_min_f32x8;
    using arch::avx2::vec_max_f32x8;
    using arch::avx2::vec_abs_f32x8;
    using arch::avx2::vec_neg_f32x8;
    using arch::avx2::vec_cmplt_f32x8;
    using arch::avx2::vec_cmple_f32x8;
    using arch::avx2::vec_cmpgt_f32x8;
    using arch::avx2::vec_ceq_f32x8;
    using arch::avx2::vec_and_f32x8;
    using arch::avx2::vec_or_f32x8;
    using arch::avx2::vec_sqrt_f32x8;
    using arch::avx2::vec_rcp_f32x8;
    using arch::avx2::vec_rsqrt_f32x8;
    using arch::avx2::vec_reduce_sum_f32x8;
    }} // namespace nnops::simd
  #else
    // x86_64 without AVX: emulate with two SSE registers
    #include "nnops/detail/simd/arch/x86/sse.hpp"
    namespace nnops { namespace simd {
    using VecF32x8 = arch::sse::VecF32x8;
    using arch::sse::vec_load_f32x8;
    using arch::sse::vec_store_f32x8;
    using arch::sse::vec_set1_f32x8;
    using arch::sse::vec_zero_f32x8;
    using arch::sse::vec_add_f32x8;
    using arch::sse::vec_sub_f32x8;
    using arch::sse::vec_mul_f32x8;
    using arch::sse::vec_div_f32x8;
    using arch::sse::vec_fmadd_f32x8;
    using arch::sse::vec_min_f32x8;
    using arch::sse::vec_max_f32x8;
    using arch::sse::vec_sqrt_f32x8;
    using arch::sse::vec_reduce_sum_f32x8;
    }} // namespace nnops::simd
  #endif  // __AVX__

#elif defined(NNOPS_ARCH_AARCH64)
  #include "nnops/detail/simd/arch/arm/neon.hpp"
  namespace nnops { namespace simd {
  using VecF32x8 = arch::neon::VecF32x8;
  using arch::neon::vec_load_f32x8;
  using arch::neon::vec_store_f32x8;
  using arch::neon::vec_set1_f32x8;
  using arch::neon::vec_zero_f32x8;
  using arch::neon::vec_add_f32x8;
  using arch::neon::vec_sub_f32x8;
  using arch::neon::vec_mul_f32x8;
  using arch::neon::vec_fmadd_f32x8;
  using arch::neon::vec_min_f32x8;
  using arch::neon::vec_max_f32x8;
  using arch::neon::vec_sqrt_f32x8;
  using arch::neon::vec_reduce_sum_f32x8;
  }} // namespace nnops::simd

#else
  #include "nnops/detail/simd/arch/scalar.hpp"
  namespace nnops { namespace simd {
  using VecF32x8 = arch::scalar::VecF32x8;
  using arch::scalar::vec_load_f32x8;
  using arch::scalar::vec_store_f32x8;
  using arch::scalar::vec_set1_f32x8;
  using arch::scalar::vec_zero_f32x8;
  using arch::scalar::vec_add_f32x8;
  using arch::scalar::vec_sub_f32x8;
  using arch::scalar::vec_mul_f32x8;
  using arch::scalar::vec_div_f32x8;
  using arch::scalar::vec_fmadd_f32x8;
  using arch::scalar::vec_min_f32x8;
  using arch::scalar::vec_max_f32x8;
  using arch::scalar::vec_sqrt_f32x8;
  using arch::scalar::vec_exp_f32x8;
  using arch::scalar::vec_reduce_sum_f32x8;
  }} // namespace nnops::simd
#endif
