#pragma once
/// @file simd_quant.hpp
/// @brief SIMD QuantizeLinear / DequantizeLinear kernels and the shared
/// dequant/requant helpers for quantized conv and pooling.
///
/// Arithmetic is f32 (v_f32x8) or f16 (v_f16x8) by output type; f16 keeps the
/// pipeline in f16 vectors where it can. int8/uint8 widen through
/// v_cvt_s8_to_f32 / v_cvt_u8_to_f32 (or the _to_f16 pair). The helpers are
/// overloaded on pointer type so callers stay type-agnostic.
///
/// The bottom group is shared with the quantized conv/pooling kernels: the
/// activation is PerTensor (dequantize_tensor broadcasts scale/zero_point), the
/// depthwise weight is PerChannel (dequantize_channel takes per-lane scale/
/// zero_point), and requantize_store rounds to nearest even through the arch
/// `requantize_8`. `quant_kernel` resolves to the active arch's primitives.

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

// --- QuantizeLinear helpers ---

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

inline void quant_store_int8(v_f32x8 vr, void* y_row, int64_t off,
                              int64_t pack, DataType out_dtype) {
    alignas(32) float ftmp[8];
    v_store(ftmp, vr);
    for (int lane = 0; lane < pack; ++lane) {
        int32_t q = static_cast<int32_t>(std::lround(ftmp[lane]));
        quant_write_int8(y_row, out_dtype, off + lane, q);
    }
}

// --- DequantizeLinear: int8/uint8 → floating-point SIMD vectors ---

inline v_f32x8 load_i8_to_f32(const void* p, bool in_is_i8) {
    if (in_is_i8) {
        return v_cvt_s8_to_f32(static_cast<const int8_t*>(p));
    }
    else {
        return v_cvt_u8_to_f32(static_cast<const uint8_t*>(p));
    }
}

inline v_f16x8 load_i8_to_f16(const void* p, bool in_is_i8) {
    if (in_is_i8) {
        return v_cvt_s8_to_f16(static_cast<const int8_t*>(p));
    }
    else {
        return v_cvt_u8_to_f16(static_cast<const uint8_t*>(p));
    }
}

// --- Store v_f32x8 to output (overloaded on pointer type) ---

inline void quant_store_f32(float* dest, v_f32x8 vy) {
    v_store(dest, vy);
}

inline void quant_store_f32(half* dest, v_f32x8 vy) {
    alignas(32) float fbuf[8];
    v_store(fbuf, vy);
    half hbuf[8];
    for (int i = 0; i < 8; ++i) {
        hbuf[i] = ::nnops::backend::cpu::float_to_half(fbuf[i]);
    }
    v_store(dest, v_load(hbuf));
}

// --- Load f32/f16 → v_f32x8 (overloaded on pointer type) ---

inline v_f32x8 quant_load_f32(const float* p) {
    return v_load(p);
}

inline v_f32x8 quant_load_f32(const half* p) {
    v_f16x8 vh = v_load(p);
    half hbuf[8];
    v_store(hbuf, vh);
    float fbuf[8];
    for (int i = 0; i < 8; ++i) {
        fbuf[i] = ::nnops::backend::cpu::half_to_float(hbuf[i]);
    }
    return v_load(fbuf);
}

// --- Combined dequant+store pipeline: load s8, dequantize, store ---

inline void dequant_i8_store(const void* src, bool is_i8,
                              float scale, float zp, float* dst) {
    auto vw = load_i8_to_f32(src, is_i8);
    auto vs = v_set1_f32x8(scale);
    auto vz = v_set1_f32x8(zp);
    v_store(dst, v_mul(v_sub(vw, vz), vs));
}

inline void dequant_i8_store(const void* src, bool is_i8,
                              float scale, float zp, half* dst) {
    auto vw = load_i8_to_f16(src, is_i8);
    auto vs = v_set1_f16x8(scale);
    auto vz = v_set1_f16x8(zp);
    v_store(dst, v_mul(v_sub(vw, vz), vs));
}

// --- Scalar dequant+store, for tail elements (< kQuantLane) ---

inline void dequant_i8_scalar(const void* src, bool is_i8,
                               float scale, float zp, float* dst) {
    float w = is_i8
        ? static_cast<float>(*static_cast<const int8_t*>(src))
        : static_cast<float>(*static_cast<const uint8_t*>(src));
    *dst = (w - zp) * scale;
}

inline void dequant_i8_scalar(const void* src, bool is_i8,
                               float scale, float zp, half* dst) {
    float w = is_i8
        ? static_cast<float>(*static_cast<const int8_t*>(src))
        : static_cast<float>(*static_cast<const uint8_t*>(src));
    *dst = ::nnops::backend::cpu::float_to_half((w - zp) * scale);
}

// --- Shared dequant/requant helpers for quantized conv/pooling kernels ---

/// Dequantize eight activation values (PerTensor: broadcast scale/zero_point).
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

/// Dequantize eight weight values (PerChannel: per-lane scale/zero_point).
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

/// Requantize eight lanes; @p add_to dequantizes the existing output, adds in
/// float, then requantizes.
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
