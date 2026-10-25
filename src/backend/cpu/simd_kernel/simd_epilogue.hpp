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

/// Requantize a compensated int32 accumulate to int8.
///
///   q = clamp(round(acc · requant_scale) + zp_out, −128, 127)
///
/// where requant_scale = scale_a[m]·scale_b[n] / scale_out. The multiply is done
/// in double (exact for the int32 range) and rounded half-away-from-zero.
inline int8_t matmul_int8_requant(int32_t acc, double requant_scale, int32_t zp_out) noexcept {
    const double d = std::round(static_cast<double>(acc) * requant_scale)
                   + static_cast<double>(zp_out);
    const int64_t v = static_cast<int64_t>(d);
    return static_cast<int8_t>(std::max<int64_t>(-128, std::min<int64_t>(127, v)));
}

// =========================================================================
//  int8 output pass — a whole output row, vectorized
// =========================================================================
//
// Per column n of one output row,
//
//   v[n]  = raw[n] − (u8_offset + zp_a)·R_b[n] − (R_a − K·zp_a)·zp_b[n]
//   out   = clamp(round(v·s_a·s_b[n]/s_out) + zp_out, −128, 127)   [s8 output]
//
// so the column-varying part is exactly two vectors — R_b (always) and zp_b
// (only when per-channel) — scaled by row constants. The scalar form ran a
// libm round per element; the vector form keeps f64 so it stays bit-identical
// to matmul_int8_requant (the tests compare exact bytes).

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
};

#if defined(NNOPS_ARCH_X86_64)
/// std::round semantics — nearest, ties away from zero — in f64.
///
/// AVX2 only offers ties-to-even, so this rounds by truncating and adding the
/// sign when the fractional part reaches a half: t = trunc(x), f = x − t is
/// exact for every |x| < 2^52 (and f = 0 above it), so the test is exact.
inline __m256d int8_round_away(__m256d x) noexcept {
    const __m256d t = _mm256_round_pd(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    const __m256d f = _mm256_sub_pd(x, t);
    const __m256d af = _mm256_andnot_pd(_mm256_set1_pd(-0.0), f);
    const __m256d half = _mm256_cmp_pd(af, _mm256_set1_pd(0.5), _CMP_GE_OQ);
    const __m256d dir = _mm256_and_pd(
        half, _mm256_or_pd(_mm256_and_pd(x, _mm256_set1_pd(-0.0)), _mm256_set1_pd(1.0)));
    return _mm256_add_pd(t, dir);
}
#endif

/// Compensate one output row of the raw int32 accumulator, in place.
inline void int8_epilogue_row_s32(int32_t* c, int64_t n0, int64_t n1,
                                  const Int8EpilogueRow& p) noexcept {
    const int32_t coef_rb = p.a_bias_offset + p.zp_a;
    const int32_t coef_zp = p.r_a - p.k * p.zp_a;
    int64_t n = n0;

    // The compensation is v = raw − coef_rb·R_b[n] − coef_zp·zp_b[n]: two
    // vector FMAs' worth of integer work per element, no float involved.
#if defined(NNOPS_ARCH_X86_64)
    const __m256i vcrb = _mm256_set1_epi32(coef_rb);
    const __m256i vczp = _mm256_set1_epi32(coef_zp);
    for (; n + 8 <= n1; n += 8) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c + n));
        v = _mm256_sub_epi32(
            v, _mm256_mullo_epi32(vcrb,
                                  _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                      p.r_b + n))));
        const __m256i zps =
            p.zp_b ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p.zp_b + n))
                   : _mm256_set1_epi32(p.zp_b_scalar);
        v = _mm256_sub_epi32(v, _mm256_mullo_epi32(vczp, zps));
        if (p.relu) { v = _mm256_max_epi32(v, _mm256_setzero_si256()); }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + n), v);
    }
#elif defined(NNOPS_ARCH_AARCH64)
    const int32x4_t vcrb = vdupq_n_s32(coef_rb);
    const int32x4_t vczp = vdupq_n_s32(coef_zp);
    for (; n + 4 <= n1; n += 4) {
        int32x4_t v = vld1q_s32(c + n);
        v = vsubq_s32(v, vmulq_s32(vcrb, vld1q_s32(p.r_b + n)));
        const int32x4_t zps = p.zp_b ? vld1q_s32(p.zp_b + n) : vdupq_n_s32(p.zp_b_scalar);
        v = vsubq_s32(v, vmulq_s32(vczp, zps));
        if (p.relu) { v = vmaxq_s32(v, vdupq_n_s32(0)); }
        vst1q_s32(c + n, v);
    }
#endif

    for (; n < n1; ++n) {
        const int32_t zp_bn = p.zp_b ? p.zp_b[n] : p.zp_b_scalar;
        int32_t v = matmul_int8_compensate(c[n], p.a_bias_offset, p.zp_a, zp_bn,
                                           p.r_a, p.r_b[n], p.k);
        if (p.relu) { v = std::max(v, 0); }
        c[n] = v;
    }
}

/// Compensate and requantize one output row into an s8 row.
inline void int8_epilogue_row_s8(const int32_t* c, int8_t* out, int64_t n0, int64_t n1,
                                 const Int8EpilogueRow& p) noexcept {
    const int32_t coef_rb = p.a_bias_offset + p.zp_a;
    const int32_t coef_zp = p.r_a - p.k * p.zp_a;
    int64_t n = n0;
#if defined(NNOPS_ARCH_X86_64)
    const __m256i vcrb = _mm256_set1_epi32(coef_rb);
    const __m256i vczp = _mm256_set1_epi32(coef_zp);
    const __m256d vsa = _mm256_set1_pd(p.s_a);
    const __m256d vsout = _mm256_set1_pd(p.out_scale);
    const __m256d vzp = _mm256_set1_pd(static_cast<double>(p.zp_out));
    const __m256d vmin = _mm256_set1_pd(-128.0);
    const __m256d vmax = _mm256_set1_pd(127.0);
    // Per-tensor B scale: the scalar's exact expression, once per row.
    const __m256d vreq = _mm256_set1_pd(
        p.s_a * static_cast<double>(p.s_b_scalar) / p.out_scale);
    for (; n + 8 <= n1; n += 8) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c + n));
        v = _mm256_sub_epi32(
            v, _mm256_mullo_epi32(vcrb,
                                  _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                      p.r_b + n))));
        const __m256i zps =
            p.zp_b ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p.zp_b + n))
                   : _mm256_set1_epi32(p.zp_b_scalar);
        v = _mm256_sub_epi32(v, _mm256_mullo_epi32(vczp, zps));
        if (p.relu) { v = _mm256_max_epi32(v, _mm256_setzero_si256()); }

        __m256d a = _mm256_cvtepi32_pd(_mm256_castsi256_si128(v));
        __m256d b = _mm256_cvtepi32_pd(_mm256_extracti128_si256(v, 1));
        if (p.s_b) {
            a = _mm256_mul_pd(a, _mm256_div_pd(
                _mm256_mul_pd(vsa, _mm256_cvtps_pd(_mm_loadu_ps(p.s_b + n))), vsout));
            b = _mm256_mul_pd(b, _mm256_div_pd(
                _mm256_mul_pd(vsa, _mm256_cvtps_pd(_mm_loadu_ps(p.s_b + n + 4))), vsout));
        } else {
            a = _mm256_mul_pd(a, vreq);
            b = _mm256_mul_pd(b, vreq);
        }
        a = _mm256_add_pd(int8_round_away(a), vzp);
        b = _mm256_add_pd(int8_round_away(b), vzp);
        a = _mm256_min_pd(_mm256_max_pd(a, vmin), vmax);
        b = _mm256_min_pd(_mm256_max_pd(b, vmin), vmax);
        // In range and integral, so the truncating convert and the saturating
        // packs are both exact.
        const __m128i i16 = _mm_packs_epi32(_mm256_cvttpd_epi32(a),
                                            _mm256_cvttpd_epi32(b));
        const __m128i i8 = _mm_packs_epi16(i16, i16);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(out + n), i8);
    }
#elif defined(NNOPS_ARCH_AARCH64)
    const int32x4_t vcrb = vdupq_n_s32(coef_rb);
    const int32x4_t vczp = vdupq_n_s32(coef_zp);
    const float64x2_t vsa = vdupq_n_f64(p.s_a);
    const float64x2_t vsout = vdupq_n_f64(p.out_scale);
    const float64x2_t vzp = vdupq_n_f64(static_cast<double>(p.zp_out));
    const float64x2_t vmin = vdupq_n_f64(-128.0);
    const float64x2_t vmax = vdupq_n_f64(127.0);
    const float64x2_t vreq = vdupq_n_f64(
        p.s_a * static_cast<double>(p.s_b_scalar) / p.out_scale);
    for (; n + 4 <= n1; n += 4) {
        int32x4_t v = vld1q_s32(c + n);
        v = vsubq_s32(v, vmulq_s32(vcrb, vld1q_s32(p.r_b + n)));
        const int32x4_t zps = p.zp_b ? vld1q_s32(p.zp_b + n) : vdupq_n_s32(p.zp_b_scalar);
        v = vsubq_s32(v, vmulq_s32(vczp, zps));
        if (p.relu) { v = vmaxq_s32(v, vdupq_n_s32(0)); }

        // int32 → int64 → f64: exact, and the only widening clang's arm_neon.h
        // offers from a 2-lane int32 vector.
        float64x2_t a0 = vcvtq_f64_s64(vmovl_s32(vget_low_s32(v)));
        float64x2_t a1 = vcvtq_f64_s64(vmovl_s32(vget_high_s32(v)));
        if (p.s_b) {
            const float32x4_t sb = vld1q_f32(p.s_b + n);
            a0 = vmulq_f64(a0, vdivq_f64(vmulq_f64(vsa, vcvt_f64_f32(vget_low_f32(sb))),
                                         vsout));
            a1 = vmulq_f64(a1, vdivq_f64(vmulq_f64(vsa, vcvt_f64_f32(vget_high_f32(sb))),
                                         vsout));
        } else {
            a0 = vmulq_f64(a0, vreq);
            a1 = vmulq_f64(a1, vreq);
        }
        // FRINTA: nearest, ties away from zero — exactly std::round.
        a0 = vaddq_f64(vrndaq_f64(a0), vzp);
        a1 = vaddq_f64(vrndaq_f64(a1), vzp);
        a0 = vminq_f64(vmaxq_f64(a0, vmin), vmax);
        a1 = vminq_f64(vmaxq_f64(a1, vmin), vmax);
        const int32x4_t vi = vcombine_s32(vmovn_s64(vcvtq_s64_f64(a0)),
                                          vmovn_s64(vcvtq_s64_f64(a1)));
        const int16x4_t i16 = vqmovn_s32(vi);
        const int8x8_t i8 = vqmovn_s16(vcombine_s16(i16, i16));
        vst1_lane_s32(reinterpret_cast<int32_t*>(out + n),
                      vreinterpret_s32_s8(i8), 0);
    }
#endif

    for (; n < n1; ++n) {
        const int32_t zp_bn = p.zp_b ? p.zp_b[n] : p.zp_b_scalar;
        int32_t v = matmul_int8_compensate(c[n], p.a_bias_offset, p.zp_a, zp_bn,
                                           p.r_a, p.r_b[n], p.k);
        if (p.relu) { v = std::max(v, 0); }
        const double req = p.s_a * static_cast<double>(p.s_b ? p.s_b[n] : p.s_b_scalar)
                         / p.out_scale;
        out[n] = matmul_int8_requant(v, req, p.zp_out);
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
/// reductions, the quantization parameters, and (for an s8 output) the real
/// destination. The per-token / per-channel arrays are indexed by the *global*
/// output row / column, so a tile can be handled on its own.
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
    int8_t* out;             ///< s8 destination; nullptr → s32, in place
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
            p.r_a[m],
            p.r_b,
            p.zp_b,
            p.zp_b_scalar,
            p.out ? static_cast<double>(int8_scale_at(p.s_a, p.s_a_scalar, m)) : 0.0,
            p.s_b,
            p.s_b_scalar,
            p.out_scale,
            p.zp_out,
            p.k,
            p.relu,
        };
        if (p.out != nullptr) {
            int8_epilogue_row_s8(c + r * ldc, p.out + m * p.ldc_out, n0, n0 + nc, row);
        } else {
            int8_epilogue_row_s32(c + r * ldc, n0, n0 + nc, row);
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
    /// False when both coefficients are zero for every row and column, i.e.
    /// compensate() is the identity. r_a / r_b are null then and the dispatch
    /// skipped the reduction sweep entirely.
    bool compensate;
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
        if (p.compensate) {
            v = _mm256_sub_epi32(
                v, _mm256_mullo_epi32(vcrb,
                                      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                          p.r_b + n))));
            const __m256i zps =
                p.zp_b ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p.zp_b + n))
                       : _mm256_set1_epi32(p.zp_b_scalar);
            v = _mm256_sub_epi32(v, _mm256_mullo_epi32(vczp, zps));
        }

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
        if (p.compensate) {
            v = vsubq_s32(v, vmulq_s32(vcrb, vld1q_s32(p.r_b + n)));
            const int32x4_t zps =
                p.zp_b ? vld1q_s32(p.zp_b + n) : vdupq_n_s32(p.zp_b_scalar);
            v = vsubq_s32(v, vmulq_s32(vczp, zps));
        }

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
        const int32_t acc =
            p.compensate
                ? matmul_int8_compensate(c[n], p.a_bias_offset, p.zp_a, zp_b,
                                         p.r_a, p.r_b[n], p.k)
                : c[n];
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
    float s_a_scalar, s_b_scalar;
    int32_t zp_a_scalar, zp_b_scalar;
    int32_t a_bias_offset;
    int32_t k;
    /// False when the compensation is the identity for every row and column
    /// (a_bias_offset + zp_a == 0 and all zp_b == 0). The reductions are then
    /// skipped, r_a / r_b are null, and the row pass drops both subtractions.
    bool compensate;
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