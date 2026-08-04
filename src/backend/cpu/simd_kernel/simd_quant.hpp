#pragma once
/// @file simd_quant.hpp
/// @brief SIMD kernel functions for QuantizeLinear / DequantizeLinear.
///
/// All arithmetic is in f32 (v_f32x8). For f16 input/output, conversion
/// happens at the boundary. int8/uint8 → f32 uses v_cvt_i8_to_f32 / v_cvt_u8_to_f32.

#include "nnops/detail/simd/simd.hpp"
#include "nnops/detail/half.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace nnops::kernel {

using namespace simd;

constexpr int kQuantLane = 8;  // v_f32x8 lane width

// ============================================================
// QuantizeLinear helpers
// ============================================================

inline int32_t quant_int_min(DataType dt) { return (dt == DataType::i8) ? -128 : 0; }
inline int32_t quant_int_max(DataType dt) { return (dt == DataType::i8) ? 127 : 255; }

inline void quant_write_int8(void* ptr, DataType dt, int64_t off, int32_t val) {
    int32_t lo = quant_int_min(dt), hi = quant_int_max(dt);
    val = std::max(lo, std::min(hi, val));
    if (dt == DataType::i8)
        static_cast<int8_t*>(ptr)[off] = static_cast<int8_t>(val);
    else
        static_cast<uint8_t*>(ptr)[off] = static_cast<uint8_t>(val);
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
// DequantizeLinear helpers
// ============================================================

/// Load 8 int8/uint8 values and widen to v_f32x8.
inline v_f32x8 quant_load_int8_to_f32(const void* p, bool in_is_i8) {
    if (in_is_i8)
        return v_cvt_i8_to_f32(static_cast<const int8_t*>(p));
    else
        return v_cvt_u8_to_f32(static_cast<const uint8_t*>(p));
}

/// Store v_f32x8 to output (f32 or f16).
template <typename T>
inline void quant_store_f32(T* dest, v_f32x8 vy) {
    if constexpr (std::is_same_v<T, half>) {
        alignas(32) float fbuf[8];
        v_store(fbuf, vy);
        for (int i = 0; i < 8; ++i) dest[i] = ::nnops::backend::cpu::float_to_half(fbuf[i]);
    } else {
        v_store(dest, vy);
    }
}

/// Load f32/f16 → v_f32x8.
template <typename T>
inline v_f32x8 quant_load_f32(const T* p) {
    if constexpr (std::is_same_v<T, half>) {
        alignas(16) ::nnops::backend::cpu::half htmp[8];
        v_store(htmp, v_load(p));
        float fbuf[8];
        for (int i = 0; i < 8; ++i) fbuf[i] = ::nnops::backend::cpu::half_to_float(htmp[i]);
        return v_load(fbuf);
    } else {
        return v_load(p);
    }
}

}  // namespace nnops::kernel
