#pragma once
/// @file simd_quant.hpp
/// @brief SIMD kernel functions for QuantizeLinear / DequantizeLinear, plus the
/// shared dequant/requant helpers for quantized conv/pooling kernels.
///
/// All arithmetic is in f32 (v_f32x8) or f16 (v_f16x8) depending on the
/// output type. For f16, the full pipeline runs in f16 vectors when possible.
/// int8/uint8 → f32 uses v_cvt_s8_to_f32 / v_cvt_u8_to_f32.
/// int8/uint8 → f16 uses v_cvt_s8_to_f16 / v_cvt_u8_to_f16.
///
/// Key interface (overloaded on pointer type so callers stay type-agnostic):
///   load_i8_to_f32 / load_i8_to_f16     — widen int8/uint8 to SIMD vectors
///   quant_store_f32 / quant_load_f32    — f32/f16 boundary load/store
///   dequant_i8_store / dequant_i8_scalar — full load→dequant→store pipeline
///
/// The quantized conv/pooling kernels (depthwise conv, pooling) share a second
/// group of helpers at the bottom of this file: dequantize_tensor (PerTensor),
/// dequantize_channel (PerChannel weight), and requantize_store
/// (round-to-nearest-even via the arch `requantize_8`). The `quant_kernel`
/// alias resolves to the active arch quant primitives.
///
/// Reference: include/nnops/detail/simd/simd.hpp — v_load/v_store for half*

#include "nnops/detail/simd/simd.hpp"
#include "nnops/detail/simd/cpu_features.hpp"
#include "nnops/detail/half.hpp"

#if defined(NNOPS_ARCH_X86_64)
#include "../x86_64/quant.hpp"
#elif defined(NNOPS_ARCH_AARCH64)
#include "../aarch64/quant.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace nnops::kernel {

using namespace simd;

#if defined(NNOPS_ARCH_X86_64)
namespace quant_kernel = nnops::backend::cpu::x86_64;
#elif defined(NNOPS_ARCH_AARCH64)
namespace quant_kernel = nnops::backend::cpu::aarch64;
#endif

constexpr int kQuantLane = 8;  // v_f32x8 / v_f16x8 lane width

// ============================================================
// QuantizeLinear helpers
// ============================================================

inline int32_t quant_int_min(DataType dt) { return (dt == DataType::s8) ? -128 : 0; }
inline int32_t quant_int_max(DataType dt) { return (dt == DataType::s8) ? 127 : 255; }

inline void quant_write_int8(void* ptr, DataType dt, int64_t off, int32_t val) {
    int32_t lo = quant_int_min(dt), hi = quant_int_max(dt);
    val = std::max(lo, std::min(hi, val));
    if (dt == DataType::s8) {
        static_cast<int8_t*>(ptr)[off] = static_cast<int8_t>(val);
    }
    else {
        static_cast<uint8_t*>(ptr)[off] = static_cast<uint8_t>(val);
    }
}

/// Quantize a v_f32x8 vector to int8 and write to output.
inline void quant_store_int8(v_f32x8 vr, void* y_row, int64_t off,
                              int64_t pack, DataType out_dtype) {
    alignas(32) float ftmp[8];
    v_store(ftmp, vr);
    for (int lane = 0; lane < pack; ++lane) {
        int32_t q = static_cast<int32_t>(std::lround(ftmp[lane]));
        quant_write_int8(y_row, out_dtype, off + lane, q);
    }
}

// ============================================================
// DequantizeLinear: int8/uint8 → floating-point SIMD vectors
// ============================================================

/// Load 8 int8/uint8 values and widen to v_f32x8.
inline v_f32x8 load_i8_to_f32(const void* p, bool in_is_i8) {
    if (in_is_i8) {
        return v_cvt_s8_to_f32(static_cast<const int8_t*>(p));
    }
    else {
        return v_cvt_u8_to_f32(static_cast<const uint8_t*>(p));
    }
}

/// Load 8 int8/uint8 values and widen to v_f16x8.
inline v_f16x8 load_i8_to_f16(const void* p, bool in_is_i8) {
    if (in_is_i8) {
        return v_cvt_s8_to_f16(static_cast<const int8_t*>(p));
    }
    else {
        return v_cvt_u8_to_f16(static_cast<const uint8_t*>(p));
    }
}

// ============================================================
// Store v_f32x8 to output (overloaded on pointer type)
// ============================================================

/// Store v_f32x8 directly to float* — no conversion needed.
inline void quant_store_f32(float* dest, v_f32x8 vy) {
    v_store(dest, vy);
}

/// Store v_f32x8 to half*: extract to f32 buffer, convert each lane
/// to half, then SIMD store as v_f16x8 (v_load + v_store).
inline void quant_store_f32(half* dest, v_f32x8 vy) {
    alignas(32) float fbuf[8];
    v_store(fbuf, vy);
    half hbuf[8];
    for (int i = 0; i < 8; ++i) {
        hbuf[i] = ::nnops::backend::cpu::float_to_half(fbuf[i]);
    }
    v_store(dest, v_load(hbuf));  // SIMD store of 8 half values
}

// ============================================================
// Load f32/f16 → v_f32x8 (overloaded on pointer type)
// ============================================================

/// Load float* → v_f32x8 directly — no conversion needed.
inline v_f32x8 quant_load_f32(const float* p) {
    return v_load(p);
}

/// Load half* → v_f32x8: SIMD load as v_f16x8, extract each lane,
/// convert to float, then load back as v_f32x8.
inline v_f32x8 quant_load_f32(const half* p) {
    v_f16x8 vh = v_load(p);      // SIMD load of 8 half values
    half hbuf[8];
    v_store(hbuf, vh);
    float fbuf[8];
    for (int i = 0; i < 8; ++i) {
        fbuf[i] = ::nnops::backend::cpu::half_to_float(hbuf[i]);
    }
    return v_load(fbuf);          // SIMD load of 8 float values
}

// ============================================================
// Combined dequant+store pipeline: load s8, dequantize, store
// ============================================================
// Overloaded on output pointer type so callers (e.g. embed) can
// be type-agnostic — no SIMD vector types exposed in the caller.

/// Load 8 int8/uint8, dequantize in f32, store to float*.
inline void dequant_i8_store(const void* src, bool is_i8,
                              float scale, float zp, float* dst) {
    auto vw = load_i8_to_f32(src, is_i8);
    auto vs = v_set1_f32x8(scale);
    auto vz = v_set1_f32x8(zp);
    v_store(dst, v_mul(v_sub(vw, vz), vs));
}

/// Load 8 int8/uint8, dequantize in f16, store to half*.
inline void dequant_i8_store(const void* src, bool is_i8,
                              float scale, float zp, half* dst) {
    auto vw = load_i8_to_f16(src, is_i8);
    auto vs = v_set1_f16x8(scale);
    auto vz = v_set1_f16x8(zp);
    v_store(dst, v_mul(v_sub(vw, vz), vs));
}

// ============================================================
// Scalar dequant+store — for tail elements (< kQuantLane)
// ============================================================

/// Scalar dequantize and store to float*.
inline void dequant_i8_scalar(const void* src, bool is_i8,
                               float scale, float zp, float* dst) {
    float w = is_i8
        ? static_cast<float>(*static_cast<const int8_t*>(src))
        : static_cast<float>(*static_cast<const uint8_t*>(src));
    *dst = (w - zp) * scale;
}

/// Scalar dequantize and store to half*.
inline void dequant_i8_scalar(const void* src, bool is_i8,
                               float scale, float zp, half* dst) {
    float w = is_i8
        ? static_cast<float>(*static_cast<const int8_t*>(src))
        : static_cast<float>(*static_cast<const uint8_t*>(src));
    *dst = ::nnops::backend::cpu::float_to_half((w - zp) * scale);
}

// ============================================================
// Shared dequant/requant helpers for quantized conv/pooling kernels
// ============================================================
//
// Depthwise conv and pooling dequantize s8/u8 inputs to float on load,
// accumulate in float, then requantize on store. These helpers are the shared
// glue between the two operators' SIMD kernels. The activation is PerTensor
// (broadcast scale/zero_point); the depthwise weight is PerChannel (per-lane
// scale/zero_point). Rounding is round-to-nearest-even via the arch
// `requantize_8`.

/// Dequantize 8 activation values (PerTensor: broadcast scale/zero_point).
template <typename InT>
inline v_f32x8 dequantize_tensor(const InT* p, float scale, float zero) {
    v_f32x8 v;
    if constexpr (std::is_same_v<InT, int8_t>) {
        v = v_cvt_s8_to_f32(p);
    } else {
        v = v_cvt_u8_to_f32(p);
    }
    v = v_sub(v, v_set1_f32x8(zero));
    return v_mul(v, v_set1_f32x8(scale));
}

/// Dequantize 8 weight values (PerChannel: per-lane scale/zero_point).
template <typename InT>
inline v_f32x8 dequantize_channel(const InT* p, const float* scale8, const float* zero8) {
    v_f32x8 v;
    if constexpr (std::is_same_v<InT, int8_t>) {
        v = v_cvt_s8_to_f32(p);
    } else {
        v = v_cvt_u8_to_f32(p);
    }
    v = v_sub(v, v_load(zero8));
    return v_mul(v, v_load(scale8));
}

/// Requantize a float accumulator and store to 8 int8/uint8 output lanes.
/// `add_to` dequantizes the existing output, adds in float, then requantizes.
template <typename OutT>
inline void requantize_store(OutT* dst, const v_f32x8& acc,
                             float out_scale, float out_zero, bool add_to) {
    float buf[8];
    v_store(buf, acc);
    if (add_to) {
        for (int i = 0; i < 8; ++i) {
            const float q = static_cast<float>(dst[i]);
            buf[i] += (q - out_zero) * out_scale;
        }
    }
    quant_kernel::requantize_8<OutT>(dst, buf, 1.0f / out_scale, out_zero);
}

}  // namespace nnops::kernel
