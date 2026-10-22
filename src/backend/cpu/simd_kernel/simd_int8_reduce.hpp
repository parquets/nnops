#pragma once
/// @file simd_int8_reduce.hpp
/// @brief int8 row / column sums — the reductions behind the int8 GEMM's
///        zero-point compensation.
///
///   R_a[m] = Σ_k A[m,k]   (a row sum; a column sum when transpose_a)
///   R_b[n] = Σ_k B[k,n]   (a column sum; a row sum when transpose_b)
///
/// Integer, int32-accumulated and bit-identical to the scalar reference, so it
/// does not fit `row_reduce<T, Op>` in simd_reduce_primitive.hpp (f32/f16 with
/// float accumulators). The kernels here are thread-free — the caller owns the
/// slab split, exactly like `epilogue_inplace`.
///
/// Widening on x86 is the madd trick: `_mm256_maddubs_epi16` against a vector
/// of 1s sums byte pairs into i16 (|Σ| ≤ 254, no saturation), then
/// `_mm256_madd_epi16` against 1s sums those into i32. An unsigned operand
/// uses `_mm256_sad_epu8` instead, which sums 8 bytes straight into i64 lanes.

#include "nnops/detail/simd/simd.hpp"

#include <cstdint>
#include <type_traits>

#if defined(NNOPS_ARCH_X86_64)
#include <immintrin.h>
#elif defined(NNOPS_ARCH_AARCH64)
#include <arm_neon.h>
#endif

namespace nnops::kernel {

/// Σ p[0..n) of int8 (or uint8) values, accumulated in int32.
template <typename AT>
inline int32_t int8_row_sum(const AT* p, int64_t n) noexcept {
    constexpr bool kU8 = std::is_same_v<AT, uint8_t>;
    int64_t i = 0;
    int32_t s = 0;

#if defined(NNOPS_ARCH_X86_64)
    if constexpr (kU8) {
        __m256i acc = _mm256_setzero_si256();
        const __m256i zero = _mm256_setzero_si256();
        for (; i + 32 <= n; i += 32) {
            acc = _mm256_add_epi64(
                acc, _mm256_sad_epu8(
                         _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i)),
                         zero));
        }
        alignas(32) int64_t lane[4];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lane), acc);
        s = static_cast<int32_t>(lane[0] + lane[1] + lane[2] + lane[3]);
    } else {
        const __m256i ones8 = _mm256_set1_epi8(1);
        const __m256i ones16 = _mm256_set1_epi16(1);
        __m256i acc = _mm256_setzero_si256();
        for (; i + 32 <= n; i += 32) {
            const __m256i x =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i));
            acc = _mm256_add_epi32(
                acc, _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, x), ones16));
        }
        alignas(32) int32_t lane[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lane), acc);
        for (int l = 0; l < 8; ++l) { s += lane[l]; }
    }
#elif defined(NNOPS_ARCH_AARCH64)
    int32x4_t acc = vdupq_n_s32(0);
    for (; i + 16 <= n; i += 16) {
        if constexpr (kU8) {
            const uint8x16_t x = vld1q_u8(reinterpret_cast<const uint8_t*>(p + i));
            acc = vaddq_s32(
                acc, vpaddlq_s16(vreinterpretq_s16_u16(vpaddlq_u8(x))));
        } else {
            const int8x16_t x = vld1q_s8(reinterpret_cast<const int8_t*>(p + i));
            acc = vaddq_s32(acc, vpaddlq_s16(vpaddlq_s8(x)));
        }
    }
    s = vaddvq_s32(acc);
#endif

    for (; i < n; ++i) { s += static_cast<int32_t>(p[i]); }
    return s;
}

/// r[i] += Σ_j p[j*ld + i] for i in [i0, i1): a column slab of a row-major
/// matrix. The caller zeroes the slab (or feeds it a previous partial sum).
template <typename AT>
inline void int8_col_sum_slab(const AT* p, int64_t ld, int64_t i0, int64_t i1,
                              int64_t n_in, int32_t* r) noexcept {
    constexpr bool kU8 = std::is_same_v<AT, uint8_t>;
    for (int64_t j = 0; j < n_in; ++j) {
        const AT* row = p + j * ld;
        int64_t i = i0;
#if defined(NNOPS_ARCH_X86_64)
        for (; i + 8 <= i1; i += 8) {
            const __m128i b = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(row + i));
            __m256i v;
            if constexpr (kU8) {
                v = _mm256_cvtepu8_epi32(b);
            } else {
                v = _mm256_cvtepi8_epi32(b);
            }
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(r + i),
                _mm256_add_epi32(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r + i)), v));
        }
#elif defined(NNOPS_ARCH_AARCH64)
        for (; i + 8 <= i1; i += 8) {
            int32x4_t v0, v1;
            if constexpr (kU8) {
                const uint8x8_t b = vld1_u8(reinterpret_cast<const uint8_t*>(row + i));
                const uint16x8_t h = vmovl_u8(b);
                v0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(h)));
                v1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(h)));
            } else {
                const int8x8_t b = vld1_s8(reinterpret_cast<const int8_t*>(row + i));
                const int16x8_t h = vmovl_s8(b);
                v0 = vmovl_s16(vget_low_s16(h));
                v1 = vmovl_s16(vget_high_s16(h));
            }
            vst1q_s32(r + i, vaddq_s32(vld1q_s32(r + i), v0));
            vst1q_s32(r + i + 4, vaddq_s32(vld1q_s32(r + i + 4), v1));
        }
#endif
        for (; i < i1; ++i) { r[i] += static_cast<int32_t>(row[i]); }
    }
}

}  // namespace nnops::kernel
