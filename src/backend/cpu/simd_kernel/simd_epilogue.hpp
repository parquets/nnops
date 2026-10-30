#pragma once
/// @file simd_epilogue.hpp
/// @brief Epilogue kernels — scalar and vector apply_epilogue, epilogue_inplace,
///        and the int8 / fp tile output passes of MatMul and Conv2D. The scalar
///        overloads declared in nnops/core/epilogue.hpp are defined here.
///
/// The int8 routes accumulate (qa + a_bias_offset)·qb in int32, so the passes
/// here recover the MatMulInteger value before requantizing:
///
///     v[n] = raw[n] − coef_rb·R_b[n] − coef_zp·zp_b[n]
///     coef_rb = a_bias_offset + zp_a[m],  coef_zp = R_a[m] − K·zp_a[m]
///
/// with R_a[m] = Σ_k qa[m,k] and R_b[n] = Σ_k qb[k,n]. An int32 bias is in those
/// same units, so it sits after both subtractions and before the requant/Relu.
/// `compensate` / `col_term` gate one subtraction each, because a caller may fold
/// one away: a per-tensor zp_b (MatMul's or Conv2D's) folds into the per-row
/// `bias` (integer output) or `row_off` (fp), clearing `compensate`; the column
/// half has no n-constant fold. Both false with nothing folded means the
/// reductions were never swept and r_a / r_b are null.
///
/// a_bias_offset is the A-pack shift the kernels did not withdraw themselves —
/// the quantity matmul_int8_compensate takes as `u8_offset`: 0 on x86 (the B
/// panel's correction block takes it), −128 for a u8 A on aarch64. The int8
/// passes need int32 vectors, which the portable wrapper (f32/f16/s8/u8) does
/// not expose, so they use the native intrinsics.
///
/// The fp route dequantizes, clips, then activates: the int8 MMA clamps int32,
/// so the fp clip cannot ride the k loop the way fp's own MMA clamp does.
///
/// Rounding is nearest, ties away from zero, in the scalar and both vector
/// passes, so integer outputs match the reference byte for byte. The requant
/// multiplies in f32 like MLAS's requantize — exact below 2^24, never off by
/// enough to cross a clamp bound — and clamps before the truncating convert,
/// whose out-of-range value is the integer indefinite. dq_epilogue_row skips a
/// bias of exactly 0: x + 0 turns -0 into +0, and MatMul outputs are compared
/// bitwise.

#include "nnops/core/epilogue.hpp"
#include "nnops/core/quant_params.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "activation_kernels.hpp"

#include <cmath>
#include <cstdint>
#include <algorithm>

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
    // Every live type is per-element; the channel index is for the reserved
    // per-channel quant fields (Epilogue::quant_scales / quant_zero_points).
    return apply_epilogue(ep, x);
}

}  // namespace nnops

namespace nnops::backend::cpu {

using namespace simd;
using namespace nnops::kernel;

/// The per-row / per-channel entry when present, else the per-tensor scalar.
inline int32_t quant_zp_at(const QuantParams& q, int64_t idx) noexcept {
    return (q.zero_point_data != nullptr) ? q.zero_point_data[idx] : q.zero_point;
}
inline float quant_scale_at(const QuantParams& q, int64_t idx) noexcept {
    return (q.scale_data != nullptr) ? q.scale_data[idx] : q.scale;
}

/// Recover Σ (qa − zp_a)·(qb − zp_b) from a raw int32 accumulate; exact over the
/// int8 range. Shared with the reference so the two agree bit-for-bit.
inline int32_t matmul_int8_compensate(int32_t raw, int32_t u8_offset,
                                      int32_t zp_a, int32_t zp_b,
                                      int32_t r_a, int32_t r_b, int32_t k) noexcept {
    return raw - (u8_offset + zp_a) * r_b - zp_b * r_a + k * zp_a * zp_b;
}

/// Requantize a compensated int32 accumulate: clamp(round(acc·scale) + zp_out,
/// lo, hi), lo/hi being −128..127 for s8 and 0..255 for u8.
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

/// The s8 form, so a call site needs no template argument.
inline int8_t matmul_int8_requant(int32_t acc, float requant_scale, int32_t zp_out) noexcept {
    return matmul_int8_requant<int8_t>(acc, requant_scale, zp_out);
}

/// (s_a / s_out) · s_b: folding the output scale into s_a leaves one multiply per
/// element even when the weight scale is per-channel.
inline float matmul_int8_requant_scale(float s_a, float s_b, float out_scale) noexcept {
    return (s_a / out_scale) * s_b;
}

// =========================================================================
//  int8 output pass — a whole output row, vectorized
// =========================================================================

/// The row-constant half of the int8 epilogue.
struct Int8EpilogueRow {
    int32_t a_bias_offset;  ///< A-pack shift the kernels left in the accumulate
    int32_t zp_a;           ///< A zero point for this row
    int32_t r_a;            ///< Σ_k qa[m,k]
    const int32_t* r_b;     ///< Σ_k qb[k,n], indexed by column
    const int32_t* zp_b;    ///< per-channel B zero points, or nullptr
    int32_t zp_b_scalar;    ///< B zero point when zp_b is null
    double s_a;             ///< A scale for this row
    const float* s_b;       ///< per-channel B scales, or nullptr
    float s_b_scalar;       ///< B scale when s_b is null
    double out_scale;       ///< output scale
    int32_t zp_out;         ///< output zero point
    int32_t k;              ///< reduction depth
    bool relu;              ///< fuse Relu before the requant
    int32_t bias;           ///< row-constant int32 bias (Conv2D's [OC]), or 0
    const int32_t* col_bias;///< per-column int32 bias, or nullptr
    bool compensate;        ///< the −coef_zp·zp_b[n] half is live
    bool col_term;          ///< the −coef_rb·r_b[n] half is live
};

#if defined(NNOPS_ARCH_X86_64)
/// std::round semantics in f32 — nearest, ties away from zero. AVX2 only offers
/// ties-to-even: add ±0.5, rebuilt from the sign bit, and let the caller
/// truncate. Exact for |x| <= 255, the span the caller's clamp leaves.
inline __m256 int8_round_away_f32(__m256 x) noexcept {
    const __m256 half = _mm256_or_ps(_mm256_and_ps(x, _mm256_set1_ps(-0.0f)),
                                     _mm256_set1_ps(0.5f));
    return _mm256_add_ps(x, half);
}
#endif

/// Compensate, bias and relu one output row of the raw int32 accumulator, in place.
inline void int8_epilogue_row_s32(int32_t* c, int64_t n0, int64_t n1,
                                  const Int8EpilogueRow& p) noexcept {
    const int32_t coef_rb = p.a_bias_offset + p.zp_a;
    const int32_t coef_zp = p.r_a - p.k * p.zp_a;
    int64_t n = n0;

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

/// The identity-compensation s32 output: raw + bias [+ relu], in place.
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

/// Compensate, bias and requantize one output row into s8 or u8. TOut picks the
/// clamp range and the saturating pack, the expression before it being shared.
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

/// The SIMD vector for element type T: v_load(float*) → v_f32x8, v_load(half*)
/// → v_f16x8.
template <typename T>
using vec_for = decltype(v_load(std::declval<const T*>()));

/// Apply an epilogue to one SIMD vector. `type_tag` only picks the f32 / f16
/// overloads of v_set1 and v_zero; every type is a min/max decomposition, so one
/// kernel serves f32x8, f16x8 and all backends.
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

/// data = epilogue(data + bias) + add_src, in place over an M×N row-major tile.
/// `add_src` is dense M×N (row stride N), added after the activation — how the
/// Conv2D residual fuses into the same write-back pass.
template <class T>
void epilogue_inplace(int M, int N, T* data, int ld, const T* bias, const Epilogue& epilogue,
                      const T* add_src = nullptr, bool add_to = false) {
    constexpr int L = simd_lane_for<T>;

    for (int m = 0; m < M; ++m) {
        T* row = data + m * ld;
        const T* add_row = (add_to && add_src) ? add_src + m * N : nullptr;
        int n = 0;

        for (; n + L <= N; n += L) {
            auto v_data = v_load(row + n);

            if (bias) {
                auto v_bias = v_load(bias + n);
                v_data = v_add(v_data, v_bias);
            }

            v_data = apply_epilogue_vec(epilogue, row + n, v_data);

            if (add_row) {
                v_data = v_add(v_data, v_load(add_row + n));
            }

            v_store(row + n, v_data);
        }

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

/// Everything the int8 tile epilogue needs beyond the tile itself. The per-token /
/// per-channel arrays are indexed by the *global* output row / column, so a tile
/// can be handled on its own; the per-row bias offsets are applied by the caller.
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
    /// The two halves, independently gated: compensate → per element, reads r_a;
    /// col_term → per column, reads r_b. An integer destination does not itself
    /// force the sweep: a symmetric s8 W8A8 setup requantizes from the raw
    /// accumulate with both flags clear.
    bool compensate;
    bool col_term;
    const int32_t* bias;     ///< row int32 bias (Conv2D's [OC] or a folded row term)
    const int32_t* col_bias; ///< column int32 bias (MatMul [N], Conv2D channel-last)
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
/// [n0, n0+nc) — at last_k, on the tile the MMA just produced, so C is never
/// swept again.
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
// Per column n of one output row:
//   acc = raw[n] − coef_rb·R_b[n] − coef_zp·zp_b[n] + row_off
//   out = ep(clamp(float(acc)·(s_a·s_b[n]) + bias, min_clip, max_clip))

/// The row-constant half of the fp output pass.
template <class TOut>
struct DqEpilogueRow {
    int32_t coef_rb;        ///< a_bias_offset + zp_a
    int32_t coef_zp;        ///< r_a − k·zp_a
    bool compensate;        ///< the −coef_zp·zp_b[n] half is live
    bool col_term;          ///< the −coef_rb·r_b[n] half is live
    /// −zp_b·(r_a − k·zp_a) for this row, precomputed by the caller; 0 when the
    /// caller folded nothing.
    int32_t row_off;
    int32_t a_bias_offset;
    int32_t zp_a;
    int32_t r_a;
    const int32_t* r_b;     ///< Σ_k qb[k,n], indexed by column
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
    // A zero bias is skipped: x + 0 turns -0 into +0, and MatMul's outputs are
    // compared bitwise.
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
    // A zero bias is skipped: x + 0 turns -0 into +0, and MatMul's outputs are
    // compared bitwise.
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

/// Everything the fp tile output pass needs beyond the tile itself.
template <class TOut>
struct DqEpilogueParams {
    const int32_t* r_a;      ///< Σ_k A[m,k], indexed by global row
    const float* s_a;        ///< per-token A scales, or nullptr
    const int32_t* zp_a;     ///< per-token A zero points, or nullptr
    const int32_t* r_b;      ///< Σ_k B[k,n], indexed by column
    const float* s_b;        ///< per-channel B scales, or nullptr
    const int32_t* zp_b;     ///< per-channel B zero points, or nullptr
    /// Per-row half of the compensation, already computed: −zp_b·(r_a − k·zp_a)
    /// for row m. nullptr → the row pass applies that term itself, per element.
    const int32_t* row_off;
    float s_a_scalar, s_b_scalar;
    int32_t zp_a_scalar, zp_b_scalar;
    int32_t a_bias_offset;
    int32_t k;
    /// The two halves, independently gated (see Int8EpilogueParams): compensate
    /// → per element, reads r_a; col_term → per column, reads r_b.
    bool compensate;
    bool col_term;
    /// The fused A pack's target storage; read by the driver, not by the row pass.
    bool act_u8;
    const Epilogue* ep;
    /// Per-row bias, added after the dequantize and before the clip — the same
    /// place fp's accumulator seed puts it. nullptr → no bias.
    const TOut* bias;
    /// Per-column bias over the full [N] output width, added in the same place:
    /// MatMul's [N] bias, or Conv2D's channel-last one. nullptr → none.
    const TOut* col_bias;
    TOut* out;               ///< fp destination
    int ldc_out;
};

/// Apply the fp output pass to one C tile — rows [m0, m0+mc) × columns
/// [n0, n0+nc) — at last_k, like the int8 pass.
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