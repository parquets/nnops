#pragma once
/// @file simd_int8_lane.hpp
/// @brief Element-wise int8 × int16 multiply-accumulate into int32 lanes.
///
/// The depthwise family reduces over taps, never along the lane: one tap's
/// weight multiplies the lane's own value, and successive taps accumulate in
/// place. That is the opposite shape from an int8 GEMM, whose madd/dot kernels
/// sum *adjacent bytes*, so none of the kernels in simd_int8_reduce.hpp apply
/// here — there is nothing to reduce along the lane, only per-lane widening
/// arithmetic.
///
/// The product is taken in int16 and widened on the way into the accumulator: a
/// byte times an effective weight (w − zp_w, |·| ≤ 255) is at most 255·128 =
/// 32640, inside the signed range, where an int32 multiply costs two uops and
/// a ten-cycle latency chain. One tap is then a multiply plus a pair of
/// sign-extends per 16 lanes.
///
/// Two weight forms: `madd_s` broadcasts one scalar per tap (a channel-first
/// filter is [C, 1, KH, KW], one weight per tap), `madd_v` reads a staged
/// 16-lane int16 vector (a channel-last HWCM filter is one weight per channel).
/// The activation zero point is not applied per lane — it would not fit the
/// byte — so the caller folds it into the accumulator's seed, which `add_v` and
/// `sub_scaled` build from the tap weights.

#include "nnops/detail/simd/simd.hpp"

#include <cstdint>

#if defined(NNOPS_ARCH_X86_64)
#include <immintrin.h>
#elif defined(NNOPS_ARCH_AARCH64)
#include <arm_neon.h>
#endif

namespace nnops::kernel {

/// Lanes per accumulate step, matching the int16 vector width.
inline constexpr int64_t kI8Lane = 16;

/// 16 int32 accumulators, one per lane.
struct I8LaneAcc {
#if defined(NNOPS_ARCH_X86_64)
    __m256i lo, hi;

    void set1(int32_t v) noexcept {
        lo = _mm256_set1_epi32(v);
        hi = lo;
    }
    void madd(const __m256i x16, const __m256i w16) noexcept {
        const __m256i p = _mm256_mullo_epi16(x16, w16);
        lo = _mm256_add_epi32(lo, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(p)));
        hi = _mm256_add_epi32(hi,
                              _mm256_cvtepi16_epi32(_mm256_extracti128_si256(p, 1)));
    }
    void madd_s(const int8_t* x, int32_t w) noexcept {
        madd(_mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(x))),
             _mm256_set1_epi16(static_cast<int16_t>(w)));
    }
    void madd_v(const int8_t* x, const int16_t* w) noexcept {
        madd(_mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(x))),
             _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w)));
    }
    void add_v(const int16_t* w) noexcept {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w));
        lo = _mm256_add_epi32(lo, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v)));
        hi = _mm256_add_epi32(hi, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1)));
    }
    void sub_scaled(const I8LaneAcc& b, int32_t k) noexcept {
        const __m256i vk = _mm256_set1_epi32(k);
        lo = _mm256_sub_epi32(lo, _mm256_mullo_epi32(b.lo, vk));
        hi = _mm256_sub_epi32(hi, _mm256_mullo_epi32(b.hi, vk));
    }
    void store(int32_t* dst) const noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), lo);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 8), hi);
    }

#elif defined(NNOPS_ARCH_AARCH64)
    int32x4_t lo, hi, lo2, hi2;

    void set1(int32_t v) noexcept {
        lo = hi = lo2 = hi2 = vdupq_n_s32(v);
    }
    void madd(const int16x8_t x0, const int16x8_t x1, const int16x8_t w0,
              const int16x8_t w1) noexcept {
        lo  = vmlal_s16(lo,  vget_low_s16(x0),  vget_low_s16(w0));
        hi  = vmlal_s16(hi,  vget_high_s16(x0), vget_high_s16(w0));
        lo2 = vmlal_s16(lo2, vget_low_s16(x1),  vget_low_s16(w1));
        hi2 = vmlal_s16(hi2, vget_high_s16(x1), vget_high_s16(w1));
    }
    void madd_s(const int8_t* x, int32_t w) noexcept {
        madd(vmovl_s8(vld1_s8(x)), vmovl_s8(vld1_s8(x + 8)), vdupq_n_s16(w),
             vdupq_n_s16(w));
    }
    void madd_v(const int8_t* x, const int16_t* w) noexcept {
        madd(vmovl_s8(vld1_s8(x)), vmovl_s8(vld1_s8(x + 8)), vld1q_s16(w),
             vld1q_s16(w + 8));
    }
    void add_v(const int16_t* w) noexcept {
        const int16x8_t w0 = vld1q_s16(w);
        const int16x8_t w1 = vld1q_s16(w + 8);
        lo  = vaddq_s32(lo,  vmovl_s16(vget_low_s16(w0)));
        hi  = vaddq_s32(hi,  vmovl_s16(vget_high_s16(w0)));
        lo2 = vaddq_s32(lo2, vmovl_s16(vget_low_s16(w1)));
        hi2 = vaddq_s32(hi2, vmovl_s16(vget_high_s16(w1)));
    }
    void sub_scaled(const I8LaneAcc& b, int32_t k) noexcept {
        const int32x4_t vk = vdupq_n_s32(k);
        lo  = vsubq_s32(lo,  vmulq_s32(b.lo, vk));
        hi  = vsubq_s32(hi,  vmulq_s32(b.hi, vk));
        lo2 = vsubq_s32(lo2, vmulq_s32(b.lo2, vk));
        hi2 = vsubq_s32(hi2, vmulq_s32(b.hi2, vk));
    }
    void store(int32_t* dst) const noexcept {
        vst1q_s32(dst, lo);
        vst1q_s32(dst + 4, hi);
        vst1q_s32(dst + 8, lo2);
        vst1q_s32(dst + 12, hi2);
    }

#else
    int32_t a[kI8Lane];

    void set1(int32_t v) noexcept {
        for (int64_t l = 0; l < kI8Lane; ++l) { a[l] = v; }
    }
    void madd_s(const int8_t* x, int32_t w) noexcept {
        for (int64_t l = 0; l < kI8Lane; ++l) { a[l] += w * static_cast<int32_t>(x[l]); }
    }
    void madd_v(const int8_t* x, const int16_t* w) noexcept {
        for (int64_t l = 0; l < kI8Lane; ++l) {
            a[l] += static_cast<int32_t>(w[l]) * static_cast<int32_t>(x[l]);
        }
    }
    void add_v(const int16_t* w) noexcept {
        for (int64_t l = 0; l < kI8Lane; ++l) { a[l] += static_cast<int32_t>(w[l]); }
    }
    void sub_scaled(const I8LaneAcc& b, int32_t k) noexcept {
        for (int64_t l = 0; l < kI8Lane; ++l) { a[l] -= k * b.a[l]; }
    }
    void store(int32_t* dst) const noexcept {
        for (int64_t l = 0; l < kI8Lane; ++l) { dst[l] = a[l]; }
    }
#endif
};

/// Eight int32 lanes as the build's f32 vector (the epilogue works in f32).
inline v_f32x8 v_cvt_i32_to_f32(const int32_t* p) {
#if defined(NNOPS_ARCH_X86_64)
    return v_f32x8(_mm256_cvtepi32_ps(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))));
#elif defined(NNOPS_ARCH_AARCH64)
    return v_f32x8(vcvtq_f32_s32(vld1q_s32(p)), vcvtq_f32_s32(vld1q_s32(p + 4)));
#else
    float buf[8];
    for (int i = 0; i < 8; ++i) { buf[i] = static_cast<float>(p[i]); }
    return v_load(buf);
#endif
}

}  // namespace nnops::kernel
