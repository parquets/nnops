#pragma once
/// @file simd_epilogue.hpp
/// @brief SIMD epilogue kernels — scalar and vector apply_epilogue, plus
///        epilogue_inplace for fused bias + activation output write-back.
///
/// The scalar overloads declared in nnops/core/epilogue.hpp are defined here.
/// Formulas match activation_ref.cpp exactly.

#include "nnops/core/epilogue.hpp"
#include "nnops/core/quant_params.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "activation_kernels.hpp"

#include <cmath>
#include <cstdint>
#include <algorithm>

// The int8 output pass below needs int32 vectors, which the portable wrapper
// (f32/f16/s8/u8) does not expose, so it drops to the native intrinsics.
#if defined(NNOPS_ARCH_X86_64)
#include <immintrin.h>
#elif defined(NNOPS_ARCH_AARCH64)
#include <arm_neon.h>
#endif

namespace nnops {

inline float apply_epilogue(const Epilogue& ep, float x) {
    switch (ep.type) {
    case EpilogueActivateType::None:
        return x;
    case EpilogueActivateType::Relu:
        return x > 0.0f ? x : 0.0f;
    case EpilogueActivateType::Relu6:
        return std::min(std::max(x, 0.0f), 6.0f);
    case EpilogueActivateType::LeakyRelu:
        return x > 0.0f ? x : ep.alpha * x;
    case EpilogueActivateType::Sigmoid:
        return 1.0f / (1.0f + std::exp(-x));
    case EpilogueActivateType::Tanh:
        return std::tanh(x);
    case EpilogueActivateType::Gelu: {
        constexpr float c = 0.7978845608028654f;  // sqrt(2/pi)
        return 0.5f * x * (1.0f + std::tanh(c * (x + 0.044715f * x * x * x)));
    }
    case EpilogueActivateType::Silu:
        return x / (1.0f + std::exp(-x));  // x * sigmoid(x)
    case EpilogueActivateType::HardSwish: {
        float relu6 = std::min(std::max(x + 3.0f, 0.0f), 6.0f);
        return x * relu6 * (ep.beta / 6.0f);
    }
    case EpilogueActivateType::Elu:
        return x > 0.0f ? x : ep.alpha * (std::exp(x) - 1.0f);
    }
    return x;
}

inline float apply_epilogue(const Epilogue& ep, float x, int64_t /*channel*/) {
    // Every live epilogue type is per-element, so the channel index is unused;
    // a future Dequantize/Requantize would index quant_scales/quant_zero_points
    // with it here.
    return apply_epilogue(ep, x);
}

}  // namespace nnops

namespace nnops::backend::cpu {

using namespace simd;  // v_load, v_zero, v_set1, etc.
using namespace nnops::kernel;  // pure SIMD kernels: v_relu, v_gelu, v_sigmoid, ...

/// Index a quantization parameter by output row / channel: the per-token or
/// per-channel buffer when present, else the per-tensor scalar. Shared by the
/// MatMul and Conv2D quantization paths, which both resolve parameters this way.
inline int32_t quant_zp_at(const QuantParams& q, int64_t idx) noexcept {
    return (q.zero_point_data != nullptr) ? q.zero_point_data[idx] : q.zero_point;
}
inline float quant_scale_at(const QuantParams& q, int64_t idx) noexcept {
    return (q.scale_data != nullptr) ? q.scale_data[idx] : q.scale;
}

// The GEMM accumulates (in int32) the raw dot-product
//
//     raw[m,n] = Σ_k (qa[m,k] + u8_offset) · qb[k,n]
//
// where `u8_offset` is the two's-complement bias the x86 VNNI path bakes into
// the A pack (128 to turn s8 into u8), and 0 everywhere else (SDOT / scalar /
// reference compute s8×s8 directly). The MatMulInteger value — the dot product
// of the *dezero-pointed* operands — is recovered as
//
//     out[m,n] = raw − (u8_offset + zp_a[m])·R_b[n] − zp_b[n]·R_a[m]
//                     + K·zp_a[m]·zp_b[n]
//
// with the per-token / per-channel row reductions R_a[m] = Σ_k qa[m,k] and
// R_b[n] = Σ_k qb[k,n]. For an s8 output the compensated int32 is requantized
// with the output scale / zero-point. These are shared by the tiled kernel and
// the reference so they agree bit-for-bit.

/// Recover the MatMulInteger value (Σ (qa − zp_a)·(qb − zp_b)) from a raw int32
/// accumulate. All arithmetic is exact for the int8 operand range.
inline int32_t matmul_int8_compensate(int32_t raw, int32_t u8_offset,
                                      int32_t zp_a, int32_t zp_b,
                                      int32_t r_a, int32_t r_b, int32_t k) noexcept {
    return raw - (u8_offset + zp_a) * r_b - zp_b * r_a + k * zp_a * zp_b;
}

/// Requantize a compensated int32 accumulate to int8 or uint8.
///
///   q = clamp(round(acc · requant_scale) + zp_out, −128, 127)   [s8]
///   q = clamp(round(acc · requant_scale) + zp_out,   0, 255)    [u8]
///
/// where requant_scale = scale_a[m]·scale_b[n] / scale_out (built by
/// matmul_int8_requant_scale).
///
/// The multiply and the rounding are f32, like MLAS's requantize: the product
/// is exact below 2^24 and, above it, the shorter mantissa only moves the
/// result by a fraction of one output step — never across a clamp bound. Ties
/// round away from zero, here and in the vector pass (the tests compare exact
/// bytes). The clamp runs before the truncating convert, since an f32 outside
/// int32 turns into the "integer indefinite" value.
template <class TOut>
inline TOut matmul_int8_requant(int32_t acc, float requant_scale, int32_t zp_out) noexcept {
    constexpr float kMin = std::is_same_v<TOut, uint8_t> ? 0.0f : -128.0f;
    constexpr float kMax = std::is_same_v<TOut, uint8_t> ? 255.0f : 127.0f;
    const float zpf = static_cast<float>(zp_out);
    const float x = std::min(std::max(static_cast<float>(acc) * requant_scale, kMin - zpf),
                             kMax - zpf);
    const float r = std::trunc(x + std::copysign(0.5f, x));
    return static_cast<TOut>(static_cast<int32_t>(r) + zp_out);
}

/// The s8 form, kept as a plain overload so existing callers read as before.
inline int8_t matmul_int8_requant(int32_t acc, float requant_scale, int32_t zp_out) noexcept {
    return matmul_int8_requant<int8_t>(acc, requant_scale, zp_out);
}

/// The requant factor, (s_a / s_out) · s_b, in f32. Folding the output scale
/// into s_a leaves one multiply per element even when the weight scale is
/// per-channel, and keeps the tiled pass, the reference and Conv2D on one
/// identical expression.
inline float matmul_int8_requant_scale(float s_a, float s_b, float out_scale) noexcept {
    return (s_a / out_scale) * s_b;
}

// =========================================================================
//  int8 output pass — a whole output row, vectorized
// =========================================================================
//
// Per column n of one output row,
//
//   v[n]  = raw[n] − (u8_offset + zp_a)·R_b[n] − (R_a − K·zp_a)·zp_b[n]
//   v[n] += bias            (int32, per row and/or per column; integer outputs)
//   out   = clamp(round(v·s_a·s_b[n]/s_out) + zp_out, −128, 127)   [s8 output]
//   out   = clamp(round(v·s_a·s_b[n]/s_out) + zp_out,    0, 255)   [u8 output]
//   out   = v                                                       [s32 output]
//
// The bias is an int32 in the same units as the compensated accumulate, so it
// lands between the compensation and the requant (and before Relu, which is part
// of the pre-activation value).
//
// The column-varying part is exactly two vectors — R_b and zp_b — scaled by
// row constants, and both disappear when the compensation is the identity.
// The requant then stays in f32 end to end: convert, one multiply (the row's
// s_a/s_out, times the per-channel s_b when there is one), clamp inside the
// output range, round, add zp_out, saturate. f64 would double the vector width
// needed and cost ~2x per element for no output it can tell apart.

/// The row-constant half of the int8 epilogue.
struct Int8EpilogueRow {
    int32_t a_bias_offset;  ///< u8 A shift the packer left in the accumulator
    int32_t zp_a;           ///< A's zero point for this row
    int32_t r_a;            ///< Σ_k A[m,k] (true values, see reductions)
    const int32_t* r_b;     ///< Σ_k B[k,n], indexed by column
    const int32_t* zp_b;    ///< per-channel B zero points, or nullptr
    int32_t zp_b_scalar;    ///< B zero point when zp_b is null
    double s_a;             ///< A scale for this row (s8 output only)
    const float* s_b;       ///< per-channel B scales, or nullptr
    float s_b_scalar;       ///< B scale when s_b is null
    double out_scale;       ///< output scale (s8 output only)
    int32_t zp_out;         ///< output zero point (s8 output only)
    int32_t k;              ///< reduction depth
    bool relu;              ///< fuse Relu before the (optional) requant
    int32_t bias;           ///< row-constant int32 bias (Conv2D's [OC]), or 0
    const int32_t* col_bias;///< per-column int32 bias, or nullptr
    bool compensate;        ///< the −(r_a − k·zp_a)·zp_b[n] term is live
    bool col_term;          ///< the −(a_bias_offset + zp_a)·r_b[n] term is live
};

#if defined(NNOPS_ARCH_X86_64)
/// std::round semantics — nearest, ties away from zero — in f32.
///
/// AVX2 only offers ties-to-even, so round by adding ±0.5 (rebuilt from the
/// sign bit) and letting the caller truncate: the sum is exact for every
/// |x| < 2^23, and above that x carries no fractional part to round.
inline __m256 int8_round_away_f32(__m256 x) noexcept {
    const __m256 half = _mm256_or_ps(_mm256_and_ps(x, _mm256_set1_ps(-0.0f)),
                                     _mm256_set1_ps(0.5f));
    return _mm256_add_ps(x, half);
}
#endif

/// Compensate one output row of the raw int32 accumulator, in place.
inline void int8_epilogue_row_s32(int32_t* c, int64_t n0, int64_t n1,
                                  const Int8EpilogueRow& p) noexcept {
    const int32_t coef_rb = p.a_bias_offset + p.zp_a;
    const int32_t coef_zp = p.r_a - p.k * p.zp_a;
    int64_t n = n0;

    // The compensation is v = raw − coef_rb·R_b[n] − coef_zp·zp_b[n]: two
    // vector FMAs' worth of integer work per element, no float involved. The
    // two halves are independent — a caller that folded one out sets only the
    // other's flag.
    const bool biased = (p.bias != 0);
    const bool col_biased = (p.col_bias != nullptr);
#if defined(NNOPS_ARCH_X86_64)
    const __m256i vcrb = _mm256_set1_epi32(coef_rb);
    const __m256i vczp = _mm256_set1_epi32(coef_zp);
    const __m256i vbias = _mm256_set1_epi32(p.bias);
    for (; n + 8 <= n1; n += 8) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c + n));
        if (p.col_term) {
            v = _mm256_sub_epi32(
                v, _mm256_mullo_epi32(vcrb,
                                      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                          p.r_b + n))));
        }
        if (p.compensate) {
            const __m256i zps =
                p.zp_b ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p.zp_b + n))
                       : _mm256_set1_epi32(p.zp_b_scalar);
            v = _mm256_sub_epi32(v, _mm256_mullo_epi32(vczp, zps));
        }
        if (biased) { v = _mm256_add_epi32(v, vbias); }
        if (col_biased) {
            v = _mm256_add_epi32(
                v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p.col_bias + n)));
        }
        if (p.relu) { v = _mm256_max_epi32(v, _mm256_setzero_si256()); }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + n), v);
    }
#elif defined(NNOPS_ARCH_AARCH64)
    const int32x4_t vcrb = vdupq_n_s32(coef_rb);
    const int32x4_t vczp = vdupq_n_s32(coef_zp);
    const int32x4_t vbias = vdupq_n_s32(p.bias);
    for (; n + 4 <= n1; n += 4) {
        int32x4_t v = vld1q_s32(c + n);
        if (p.col_term) { v = vsubq_s32(v, vmulq_s32(vcrb, vld1q_s32(p.r_b + n))); }
        if (p.compensate) {
            const int32x4_t zps = p.zp_b ? vld1q_s32(p.zp_b + n) : vdupq_n_s32(p.zp_b_scalar);
            v = vsubq_s32(v, vmulq_s32(vczp, zps));
        }
        if (biased) { v = vaddq_s32(v, vbias); }
        if (col_biased) { v = vaddq_s32(v, vld1q_s32(p.col_bias + n)); }
        if (p.relu) { v = vmaxq_s32(v, vdupq_n_s32(0)); }
        vst1q_s32(c + n, v);
    }
#endif

    for (; n < n1; ++n) {
        int32_t v = c[n];
        if (p.col_term) { v -= coef_rb * p.r_b[n]; }
        if (p.compensate) {
            v -= coef_zp * (p.zp_b ? p.zp_b[n] : p.zp_b_scalar);
        }
        if (p.bias != 0) { v += p.bias; }
        if (p.col_bias) { v += p.col_bias[n]; }
        if (p.relu) { v = std::max(v, 0); }
        c[n] = v;
    }
}

/// Bias-only s32 output: the compensation is the identity, so no reductions
/// exist and the row is raw + bias [+ relu], in place.
inline void int8_epilogue_row_bias(int32_t* c, int64_t n0, int64_t n1,
                                   const Int8EpilogueRow& p) noexcept {
    const bool biased = (p.bias != 0);
    const bool col_biased = (p.col_bias != nullptr);
    int64_t n = n0;
#if defined(NNOPS_ARCH_X86_64)
    const __m256i vbias = _mm256_set1_epi32(p.bias);
    for (; n + 8 <= n1; n += 8) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c + n));
        if (biased) { v = _mm256_add_epi32(v, vbias); }
        if (col_biased) {
            v = _mm256_add_epi32(
                v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p.col_bias + n)));
        }
        if (p.relu) { v = _mm256_max_epi32(v, _mm256_setzero_si256()); }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + n), v);
    }
#elif defined(NNOPS_ARCH_AARCH64)
    const int32x4_t vbias = vdupq_n_s32(p.bias);
    for (; n + 4 <= n1; n += 4) {
        int32x4_t v = vld1q_s32(c + n);
        if (biased) { v = vaddq_s32(v, vbias); }
        if (col_biased) { v = vaddq_s32(v, vld1q_s32(p.col_bias + n)); }
        if (p.relu) { v = vmaxq_s32(v, vdupq_n_s32(0)); }
        vst1q_s32(c + n, v);
    }
#endif

    for (; n < n1; ++n) {
        int32_t v = c[n];
        if (p.bias != 0) { v += p.bias; }
        if (p.col_bias) { v += p.col_bias[n]; }
        if (p.relu) { v = std::max(v, 0); }
        c[n] = v;
    }
}

/// Compensate, bias and requantize one output row into an s8 or u8 row.
///
/// TOut picks the clamp range and the saturating pack: [-128, 127] + packs_epi16
/// for s8, [0, 255] + packus_epi16 for u8. Everything before the clamp is the
/// same expression, so the two forms stay bit-identical to each other.
template <class TOut>
inline void int8_epilogue_row_quant(const int32_t* c, TOut* out, int64_t n0, int64_t n1,
                                    const Int8EpilogueRow& p) noexcept {
    constexpr bool kU8 = std::is_same_v<TOut, uint8_t>;
    constexpr float kMin = kU8 ? 0.0f : -128.0f;
    constexpr float kMax = kU8 ? 255.0f : 127.0f;
    const int32_t coef_rb = p.a_bias_offset + p.zp_a;
    const int32_t coef_zp = p.r_a - p.k * p.zp_a;
    const bool biased = (p.bias != 0);
    const bool col_biased = (p.col_bias != nullptr);
    // s_a / s_out once per row: a per-channel weight scale is then one multiply.
    const float row_k = static_cast<float>(p.s_a) / static_cast<float>(p.out_scale);
    const float zpf = static_cast<float>(p.zp_out);
    int64_t n = n0;
#if defined(NNOPS_ARCH_X86_64)
    const __m256i vcrb = _mm256_set1_epi32(coef_rb);
    const __m256i vczp = _mm256_set1_epi32(coef_zp);
    const __m256i vbias = _mm256_set1_epi32(p.bias);
    const __m256i vzp = _mm256_set1_epi32(p.zp_out);
    const __m256 vk = _mm256_set1_ps(row_k);
    const __m256 vreq = _mm256_set1_ps(row_k * p.s_b_scalar);
    const __m256 vmin = _mm256_set1_ps(kMin - zpf);
    const __m256 vmax = _mm256_set1_ps(kMax - zpf);
    for (; n + 8 <= n1; n += 8) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c + n));
        if (p.col_term) {
            v = _mm256_sub_epi32(
                v, _mm256_mullo_epi32(vcrb,
                                      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                          p.r_b + n))));
        }
        if (p.compensate) {
            const __m256i zps =
                p.zp_b ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p.zp_b + n))
                       : _mm256_set1_epi32(p.zp_b_scalar);
            v = _mm256_sub_epi32(v, _mm256_mullo_epi32(vczp, zps));
        }
        if (biased) { v = _mm256_add_epi32(v, vbias); }
        if (col_biased) {
            v = _mm256_add_epi32(
                v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p.col_bias + n)));
        }
        if (p.relu) { v = _mm256_max_epi32(v, _mm256_setzero_si256()); }

        __m256 f = _mm256_cvtepi32_ps(v);
        f = _mm256_mul_ps(f, p.s_b ? _mm256_mul_ps(vk, _mm256_loadu_ps(p.s_b + n)) : vreq);
        // Clamped into [kMin − zp, kMax − zp]: the truncating convert below is
        // then always in range, so its indefinite value never appears.
        f = _mm256_min_ps(_mm256_max_ps(f, vmin), vmax);
        const __m256i q =
            _mm256_add_epi32(_mm256_cvttps_epi32(int8_round_away_f32(f)), vzp);
        const __m128i i16 =
            _mm_packs_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
        __m128i v8;
        if constexpr (kU8) {
            v8 = _mm_packus_epi16(i16, i16);
        } else {
            v8 = _mm_packs_epi16(i16, i16);
        }
        _mm_storel_epi64(reinterpret_cast<__m128i*>(out + n), v8);
    }
#elif defined(NNOPS_ARCH_AARCH64)
    const int32x4_t vcrb = vdupq_n_s32(coef_rb);
    const int32x4_t vczp = vdupq_n_s32(coef_zp);
    const int32x4_t vbias = vdupq_n_s32(p.bias);
    const int32x4_t vzp = vdupq_n_s32(p.zp_out);
    const float32x4_t vk = vdupq_n_f32(row_k);
    const float32x4_t vreq = vdupq_n_f32(row_k * p.s_b_scalar);
    const float32x4_t vmin = vdupq_n_f32(kMin - zpf);
    const float32x4_t vmax = vdupq_n_f32(kMax - zpf);
    for (; n + 4 <= n1; n += 4) {
        int32x4_t v = vld1q_s32(c + n);
        if (p.col_term) { v = vsubq_s32(v, vmulq_s32(vcrb, vld1q_s32(p.r_b + n))); }
        if (p.compensate) {
            const int32x4_t zps = p.zp_b ? vld1q_s32(p.zp_b + n) : vdupq_n_s32(p.zp_b_scalar);
            v = vsubq_s32(v, vmulq_s32(vczp, zps));
        }
        if (biased) { v = vaddq_s32(v, vbias); }
        if (col_biased) { v = vaddq_s32(v, vld1q_s32(p.col_bias + n)); }
        if (p.relu) { v = vmaxq_s32(v, vdupq_n_s32(0)); }

        float32x4_t f = vcvtq_f32_s32(v);
        f = vmulq_f32(f, p.s_b ? vmulq_f32(vk, vld1q_f32(p.s_b + n)) : vreq);
        f = vminq_f32(vmaxq_f32(f, vmin), vmax);
        // FCVTAS: nearest, ties away from zero — exactly std::round.
        const int32x4_t q = vaddq_s32(vcvtaq_s32_f32(f), vzp);
        const int16x4_t i16 = vqmovn_s32(q);
        if constexpr (kU8) {
            const uint8x8_t v8 = vqmovun_s16(vcombine_s16(i16, i16));
            vst1_lane_u32(reinterpret_cast<uint32_t*>(out + n),
                          vreinterpret_u32_u8(v8), 0);
        } else {
            const int8x8_t v8 = vqmovn_s16(vcombine_s16(i16, i16));
            vst1_lane_s32(reinterpret_cast<int32_t*>(out + n),
                          vreinterpret_s32_s8(v8), 0);
        }
    }
#endif

    for (; n < n1; ++n) {
        int32_t v = c[n];
        if (p.col_term) { v -= coef_rb * p.r_b[n]; }
        if (p.compensate) { v -= coef_zp * (p.zp_b ? p.zp_b[n] : p.zp_b_scalar); }
        if (p.bias != 0) { v += p.bias; }
        if (p.col_bias) { v += p.col_bias[n]; }
        if (p.relu) { v = std::max(v, 0); }
        const float req = row_k * (p.s_b ? p.s_b[n] : p.s_b_scalar);
        out[n] = matmul_int8_requant<TOut>(v, req, p.zp_out);
    }
}

/// Deduce the SIMD vector type for a given data pointer type T*.
/// v_load(float*) → v_f32x8, v_load(half*) → v_f16x8.
template <typename T>
using vec_for = decltype(v_load(std::declval<const T*>()));

/// Apply epilogue to a SIMD vector (unified f32/f16 via type tag dispatch).
///
/// The type_tag pointer is used for v_set1/v_zero overload resolution,
/// matching the pattern in activation.cpp. No per-type duplication needed.
///
/// All activation types use min/max decomposition (no comparison+blend),
/// so the kernel works uniformly across f32x8, f16x8, and all backends.
template <typename T>
inline vec_for<T> apply_epilogue_vec(const Epilogue& ep,
                                      const T* type_tag,
                                      vec_for<T> vx) {
    switch (ep.type) {
    case EpilogueActivateType::None:
        return vx;
    case EpilogueActivateType::Relu:
        return v_relu(vx, v_zero(type_tag));
    case EpilogueActivateType::Relu6:
        return v_min(v_max(vx, v_zero(type_tag)), v_set1(type_tag, 6.0f));
    case EpilogueActivateType::LeakyRelu:
        return v_leaky_relu(vx, v_zero(type_tag),
                                 v_set1(type_tag, ep.alpha));
    case EpilogueActivateType::Sigmoid:
        return v_sigmoid(vx, v_set1(type_tag, 1.0f));
    case EpilogueActivateType::Tanh:
        return v_tanh(vx);
    case EpilogueActivateType::Gelu: {
        constexpr float c = 0.7978845608028654f;
        return v_gelu(vx,
            v_set1(type_tag, 0.5f),
            v_set1(type_tag, 1.0f),
            v_set1(type_tag, c),
            v_set1(type_tag, 0.044715f));
    }
    case EpilogueActivateType::Silu:
        return v_silu(vx, v_set1(type_tag, 1.0f));
    case EpilogueActivateType::HardSwish:
        return v_hard_swish(vx,
            v_zero(type_tag),
            v_set1(type_tag, 3.0f),
            v_set1(type_tag, 6.0f),
            v_set1(type_tag, ep.beta / 6.0f));
    case EpilogueActivateType::Elu:
        return v_elu(vx,
            v_zero(type_tag),
            v_set1(type_tag, ep.alpha),
            v_set1(type_tag, 1.0f));
    }
    return vx;
}

/// Apply bias addition, epilogue activation, and an optional residual add-back
/// to a MatMul/Conv output matrix.
///
/// Processes output in-place: for each of the M rows, applies per-column
/// bias (if non-null), the epilogue activation function, and — when @p add_to —
/// adds back @p add_src. This fuses the Conv2D residual (add_to) path into a
/// single write-back pass instead of a separate epilogue + add kernel.
///
/// Uses SIMD for the bulk of each row; scalar for the tail.
///
/// @tparam T      Data type (float or half)
/// @param M        Number of rows in the output matrix
/// @param N        Number of columns in the output matrix
/// @param data     Output matrix data (row-major, ld stride between rows)
/// @param ld       Leading dimension (stride in elements between consecutive rows)
/// @param bias     Per-column bias array of length N (may be nullptr)
/// @param epilogue Epilogue descriptor (None = identity, no-op)
/// @param add_src  Dense M×N residual buffer (row stride N), added after the
///                 epilogue when @p add_to. May be nullptr.
/// @param add_to   When true, adds @p add_src back into @p data after the
///                 epilogue: data = epilogue(data + bias) + add_src.
template <class T>
void epilogue_inplace(int M, int N, T* data, int ld, const T* bias, const Epilogue& epilogue,
                      const T* add_src = nullptr, bool add_to = false) {
    constexpr int L = simd_lane_for<T>;

    for (int m = 0; m < M; ++m) {
        T* row = data + m * ld;
        const T* add_row = (add_to && add_src) ? add_src + m * N : nullptr;
        int n = 0;

        // ---- SIMD loop ----
        for (; n + L <= N; n += L) {
            auto v_data = v_load(row + n);

            // Fused bias addition
            if (bias) {
                auto v_bias = v_load(bias + n);
                v_data = v_add(v_data, v_bias);
            }

            // Epilogue activation (type dispatched via pointer overload)
            v_data = apply_epilogue_vec(epilogue, row + n, v_data);

            // Fused residual add-back (add_to residual connection)
            if (add_row) {
                v_data = v_add(v_data, v_load(add_row + n));
            }

            v_store(row + n, v_data);
        }

        // ---- Scalar tail ----
        for (; n < N; ++n) {
            float val = s_load(&row[n]);
            if (bias) {
                val += s_load(&bias[n]);
            }
            val = nnops::apply_epilogue(epilogue, val);
            if (add_row) {
                val += s_load(&add_row[n]);
            }
            s_store(&row[n], val);
        }
    }
}

/// Everything the tile epilogue needs beyond the tile itself: the two
/// reductions, the quantization parameters, the int32 bias, and the integer
/// destination (s8 / u8 / s32-in-place). The per-token / per-channel arrays are
/// indexed by the *global* output row / column, so a tile can be handled on its
/// own; the per-row bias tile offsets are applied by the caller, the per-column
/// ones are indexed absolutely like r_b.
struct Int8EpilogueParams {
    const int32_t* r_a;      ///< Σ_k A[m,k]
    const float* s_a;        ///< per-token A scales, or nullptr
    const int32_t* zp_a;     ///< per-token A zero points, or nullptr
    const int32_t* r_b;      ///< Σ_k B[k,n]
    const float* s_b;        ///< per-channel B scales, or nullptr
    const int32_t* zp_b;     ///< per-channel B zero points, or nullptr
    float s_a_scalar, s_b_scalar;
    int32_t zp_a_scalar, zp_b_scalar;
    double out_scale;
    int32_t zp_out;
    int32_t a_bias_offset;
    int32_t k;
    bool relu;
    /// The compensation's two halves, applied independently:
    ///   compensate → subtract (r_a − k·zp_a)·zp_b[n] per element (r_a is read)
    ///   col_term   → subtract (a_bias_offset + zp_a)·r_b[n]      (r_b is read)
    /// Both false when the compensation is the identity, i.e. both coefficients
    /// are zero for every row and every zp_b is zero: r_a / r_b are null and the
    /// reduction sweep was skipped, leaving only the bias, relu and requant per
    /// element. An integer destination does not by itself force the sweep — a
    /// symmetric s8 W8A8 setup requantizes from the raw accumulate.
    /// A caller may fold one half out instead: `bias` (the per-row vector) takes
    /// the per-element half as a precomputed int32, which is what Conv2D's and
    /// MatMul's per-tensor-B folds carry.
    bool compensate;
    bool col_term;
    const int32_t* bias;     ///< per-row int32 bias (Conv2D [OC]), or nullptr
    const int32_t* col_bias; ///< per-column int32 bias (MatMul [N]), or nullptr
    int8_t* out;             ///< s8 destination; nullptr → s32 or u8 (see out_u8)
    uint8_t* out_u8;         ///< u8 destination; takes precedence over out
    int ldc_out;
};

inline int32_t int8_zp_at(const int32_t* data, int32_t scalar, int64_t i) noexcept {
    return data ? data[i] : scalar;
}
inline float int8_scale_at(const float* data, float scalar, int64_t i) noexcept {
    return data ? data[i] : scalar;
}

/// Apply the int8 output pass to one C tile — rows [m0, m0+mc) × columns
/// [n0, n0+nc) — the way fp applies its epilogue: at last_k, on the tile the
/// MMA just produced, so C is never swept again.
inline void int8_epilogue_tile(const Int8EpilogueParams& p, int32_t* c, int ldc,
                               int64_t m0, int n0, int64_t mc, int64_t nc) noexcept {
    for (int64_t r = 0; r < mc; ++r) {
        const int64_t m = m0 + r;
        const Int8EpilogueRow row{
            p.a_bias_offset,
            int8_zp_at(p.zp_a, p.zp_a_scalar, m),
            p.compensate ? p.r_a[m] : 0,
            p.r_b,
            p.zp_b,
            p.zp_b_scalar,
            (p.out || p.out_u8)
                ? static_cast<double>(int8_scale_at(p.s_a, p.s_a_scalar, m)) : 0.0,
            p.s_b,
            p.s_b_scalar,
            p.out_scale,
            p.zp_out,
            p.k,
            p.relu,
            p.bias ? p.bias[m] : 0,
            p.col_bias,
            p.compensate,
            p.col_term,
        };
        if (p.out_u8 != nullptr) {
            int8_epilogue_row_quant<uint8_t>(c + r * ldc, p.out_u8 + m * p.ldc_out,
                                             n0, n0 + nc, row);
        } else if (p.out != nullptr) {
            int8_epilogue_row_quant<int8_t>(c + r * ldc, p.out + m * p.ldc_out,
                                            n0, n0 + nc, row);
        } else if (p.compensate || p.col_term) {
            int8_epilogue_row_s32(c + r * ldc, n0, n0 + nc, row);
        } else {
            int8_epilogue_row_bias(c + r * ldc, n0, n0 + nc, row);
        }
    }
}

// =========================================================================
//  fp output pass — compensate, dequantize, clip, activate
// =========================================================================
//
// The offline-quantized-activation route writes fp output:
//
//   acc   = matmul_int8_compensate(raw, u8_offset, zp_a, zp_b[n], R_a, R_b[n], K)
//   f     = float(acc) · (s_a · s_b[n])
//   out   = ep(clamp(f, min_clip, max_clip))
//
// fp's own route clips inside the last-k MMA and activates in place afterwards;
// here the MMA clamps int32, so the fp clip happens after the dequant, before
// the activation. The float expression is identical to matmul_dq_ref's, so the
// tests compare output bits.

/// The row-constant half of the fp output pass.
template <class TOut>
struct DqEpilogueRow {
    int32_t coef_rb;        ///< a_bias_offset + zp_a
    int32_t coef_zp;        ///< r_a − k·zp_a
    /// The compensation's two halves, applied independently (see
    /// DqEpilogueParams): the per-element one −coef_zp·zp_b[n] is live here, the
    /// column one −coef_rb·r_b[n] is live when col_term is set.
    bool compensate;
    bool col_term;
    /// −zp_b·(r_a − k·zp_a) for this row, precomputed by the caller: the
    /// per-row half of the compensation applied once per row instead of once
    /// per element. 0 when the caller folded nothing (the usual case).
    int32_t row_off;
    int32_t a_bias_offset;
    int32_t zp_a;
    int32_t r_a;
    const int32_t* r_b;     ///< Σ_k B[k,n], indexed by column
    const int32_t* zp_b;    ///< per-channel B zero points, or nullptr
    int32_t zp_b_scalar;
    float s_a;              ///< A scale for this row
    const float* s_b;       ///< per-channel B scales, or nullptr
    float s_b_scalar;
    float bias;             ///< output-row bias, added before the clip (0 if none)
    const TOut* col_bias;   ///< per-column bias, indexed by absolute column, or null
    int32_t k;
    const Epilogue* ep;     ///< clip bounds + activation
};

template <class TOut>
inline void dq_epilogue_row(const int32_t* c, TOut* out, int64_t n0, int64_t n1,
                            const DqEpilogueRow<TOut>& p) noexcept {
    const Epilogue& ep = *p.ep;
    const bool plain = (ep.type == EpilogueActivateType::None);
    int64_t n = n0;

#if defined(NNOPS_ARCH_X86_64)
    const __m256i vcrb = _mm256_set1_epi32(p.coef_rb);
    const __m256i vczp = _mm256_set1_epi32(p.coef_zp);
    const __m256i vrow = _mm256_set1_epi32(p.row_off);
    const __m256 vsa  = _mm256_set1_ps(p.s_a);
    const __m256 vsbs = _mm256_set1_ps(p.s_b_scalar);
    const __m256 vmin = _mm256_set1_ps(ep.min_clip);
    const __m256 vmax = _mm256_set1_ps(ep.max_clip);
    const __m256 vbias = _mm256_set1_ps(p.bias);
    // Skipping a zero bias keeps x + 0 from turning -0 into +0: MatMul compares
    // these outputs by bit pattern, and it never has a bias.
    const bool biased = (p.bias != 0.0f);
    const bool col_biased = (p.col_bias != nullptr);
    alignas(32) float fbuf[8];
    for (; n + 8 <= n1; n += 8) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c + n));
        if (p.col_term) {
            v = _mm256_sub_epi32(
                v, _mm256_mullo_epi32(vcrb,
                                      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                          p.r_b + n))));
        }
        if (p.compensate) {
            const __m256i zps =
                p.zp_b ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p.zp_b + n))
                       : _mm256_set1_epi32(p.zp_b_scalar);
            v = _mm256_sub_epi32(v, _mm256_mullo_epi32(vczp, zps));
        }
        if (p.row_off != 0) { v = _mm256_add_epi32(v, vrow); }

        const __m256 sb = p.s_b ? _mm256_loadu_ps(p.s_b + n) : vsbs;
        __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(v), _mm256_mul_ps(vsa, sb));
        if (biased) { f = _mm256_add_ps(f, vbias); }
        if (col_biased) {
            if constexpr (std::is_same_v<TOut, half>) {
                f = _mm256_add_ps(f, _mm256_cvtph_ps(_mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(p.col_bias + n))));
            } else {
                f = _mm256_add_ps(f, _mm256_loadu_ps(p.col_bias + n));
            }
        }
        f = _mm256_min_ps(_mm256_max_ps(f, vmin), vmax);

        if (plain) {
            if constexpr (std::is_same_v<TOut, half>) {
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out + n),
                                 _mm256_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT));
            } else {
                _mm256_storeu_ps(out + n, f);
            }
        } else {
            // The scalar epilogue is the reference's, lane for lane.
            _mm256_store_ps(fbuf, f);
            for (int i = 0; i < 8; ++i) {
                s_store(&out[n + i], nnops::apply_epilogue(ep, fbuf[i]));
            }
        }
    }
#elif defined(NNOPS_ARCH_AARCH64)
    const int32x4_t vcrb = vdupq_n_s32(p.coef_rb);
    const int32x4_t vczp = vdupq_n_s32(p.coef_zp);
    const int32x4_t vrow = vdupq_n_s32(p.row_off);
    const float32x4_t vsa = vdupq_n_f32(p.s_a);
    const float32x4_t vsbs = vdupq_n_f32(p.s_b_scalar);
    const float32x4_t vmin = vdupq_n_f32(ep.min_clip);
    const float32x4_t vmax = vdupq_n_f32(ep.max_clip);
    const float32x4_t vbias = vdupq_n_f32(p.bias);
    // Skipping a zero bias keeps x + 0 from turning -0 into +0: MatMul compares
    // these outputs by bit pattern, and it never has a bias.
    const bool biased = (p.bias != 0.0f);
    const bool col_biased = (p.col_bias != nullptr);
    alignas(16) float fbuf[4];
    for (; n + 4 <= n1; n += 4) {
        int32x4_t v = vld1q_s32(c + n);
        if (p.col_term) { v = vsubq_s32(v, vmulq_s32(vcrb, vld1q_s32(p.r_b + n))); }
        if (p.compensate) {
            const int32x4_t zps =
                p.zp_b ? vld1q_s32(p.zp_b + n) : vdupq_n_s32(p.zp_b_scalar);
            v = vsubq_s32(v, vmulq_s32(vczp, zps));
        }
        if (p.row_off != 0) { v = vaddq_s32(v, vrow); }

        const float32x4_t sb = p.s_b ? vld1q_f32(p.s_b + n) : vsbs;
        float32x4_t f = vmulq_f32(vcvtq_f32_s32(v), vmulq_f32(vsa, sb));
        if (biased) { f = vaddq_f32(f, vbias); }
        if (col_biased) {
            if constexpr (std::is_same_v<TOut, half>) {
                f = vaddq_f32(f, vcvt_f32_f16(vld1_f16(
                    reinterpret_cast<const __fp16*>(p.col_bias + n))));
            } else {
                f = vaddq_f32(f, vld1q_f32(p.col_bias + n));
            }
        }
        f = vminq_f32(vmaxq_f32(f, vmin), vmax);

        if (plain) {
            if constexpr (std::is_same_v<TOut, half>) {
                vst1_f16(reinterpret_cast<__fp16*>(out + n), vcvt_f16_f32(f));
            } else {
                vst1q_f32(out + n, f);
            }
        } else {
            vst1q_f32(fbuf, f);
            for (int i = 0; i < 4; ++i) {
                s_store(&out[n + i], nnops::apply_epilogue(ep, fbuf[i]));
            }
        }
    }
#endif

    for (; n < n1; ++n) {
        const int32_t zp_b = p.zp_b ? p.zp_b[n] : p.zp_b_scalar;
        int32_t acc = c[n];
        if (p.col_term) { acc -= p.coef_rb * p.r_b[n]; }
        if (p.compensate) { acc -= p.coef_zp * zp_b; }
        acc += p.row_off;
        const float scale = p.s_a * (p.s_b ? p.s_b[n] : p.s_b_scalar);
        float f = static_cast<float>(acc) * scale;
        if (p.bias != 0.0f) { f += p.bias; }
        if (p.col_bias != nullptr) { f += s_load(&p.col_bias[n]); }
        f = std::min(std::max(f, ep.min_clip), ep.max_clip);
        f = nnops::apply_epilogue(ep, f);
        s_store(&out[n], f);
    }
}

/// Everything the fp output pass needs beyond the tile itself.
template <class TOut>
struct DqEpilogueParams {
    const int32_t* r_a;      ///< Σ_k A[m,k], indexed by global row
    const float* s_a;        ///< per-token A scales, or nullptr
    const int32_t* zp_a;     ///< per-token A zero points, or nullptr
    const int32_t* r_b;      ///< Σ_k B[k,n], indexed by column
    const float* s_b;        ///< per-channel B scales, or nullptr
    const int32_t* zp_b;     ///< per-channel B zero points, or nullptr
    /// Per-row half of the compensation, already computed: −zp_b·(r_a − k·zp_a)
    /// for row m, added to the accumulate before the dequantize. nullptr → the
    /// row pass applies that term itself, per element.
    const int32_t* row_off;
    float s_a_scalar, s_b_scalar;
    int32_t zp_a_scalar, zp_b_scalar;
    int32_t a_bias_offset;
    int32_t k;
    /// The compensation's two halves, applied independently, so a caller can
    /// keep whichever ones it did not fold away:
    ///   compensate → subtract (r_a − k·zp_a)·zp_b[n] per element (r_a is read)
    ///   col_term   → subtract (a_bias_offset + zp_a)·r_b[n]      (r_b is read)
    /// Both false when the compensation is the identity for every row and
    /// column (a_bias_offset + zp_a == 0 and all zp_b == 0): the reductions are
    /// then skipped, r_a / r_b are null, and the row pass drops both
    /// subtractions. The per-element half is a row constant whenever zp_b is
    /// per-tensor, so MatMul folds it into row_off and leaves compensate false,
    /// which also retires the r_a read; the column half has no n-constant fold
    /// and is only absent when a_bias_offset + zp_a == 0 for every row.
    bool compensate;
    bool col_term;
    /// A is quantised inside the A pack from the fp activations rather than
    /// materialised as an int8 tensor; this is the target's storage type.
    bool act_u8;
    const Epilogue* ep;
    /// Per-row bias, added after the dequantize and before the clip — the same
    /// place fp's accumulator seed puts it. nullptr → no bias.
    const TOut* bias;
    /// Per-column bias over the full [N] output width, added in the same place.
    /// MatMul passes its optional bias here; Conv2D has none and leaves it null.
    const TOut* col_bias;
    TOut* out;               ///< fp destination
    int ldc_out;
};

/// Apply the fp output pass to one C tile — rows [m0, m0+mc) × columns
/// [n0, n0+nc) — at last_k, like the int8 and fp epilogues.
template <class TOut>
inline void dq_epilogue_tile(const DqEpilogueParams<TOut>& p, const int32_t* c, int ldc,
                             int64_t m0, int n0, int64_t mc, int64_t nc) noexcept {
    for (int64_t r = 0; r < mc; ++r) {
        const int64_t m = m0 + r;
        const int32_t zp_a = int8_zp_at(p.zp_a, p.zp_a_scalar, m);
        const DqEpilogueRow<TOut> row{
            p.a_bias_offset + zp_a,
            p.compensate ? p.r_a[m] - p.k * zp_a : 0,
            p.compensate,
            p.col_term,
            p.row_off ? p.row_off[m] : 0,
            p.a_bias_offset,
            zp_a,
            p.compensate ? p.r_a[m] : 0,
            p.r_b,
            p.zp_b,
            p.zp_b_scalar,
            int8_scale_at(p.s_a, p.s_a_scalar, m),
            p.s_b,
            p.s_b_scalar,
            p.bias ? s_load(&p.bias[m]) : 0.0f,
            p.col_bias,
            p.k,
            p.ep,
        };
        dq_epilogue_row(c + r * ldc, p.out + m * p.ldc_out, n0, n0 + nc, row);
    }
}

}  // namespace nnops::backend::cpu