#pragma once
/// @file simd_access.hpp
/// @brief Load/store policies shared by the spatial kernels (pooling, depthwise).
///
/// Fp*    — the element type is the accumulator (f32 or f16).
/// Quant* — s8/u8 activations dequantize to f32 on load, requantize on store.
/// A kernel that takes these as template parameters needs no per-type copy.

#include "nnops/detail/simd/simd.hpp"
#include "simd_quant.hpp"

#include <cstdint>

namespace nnops::kernel {

using namespace simd;

template <typename T>
struct FpSrc {
    const T* in;    // also picks the v_set1/v_zero overload

    auto load_at(int64_t off) const { return v_load(in + off); }
    auto zero() const { return v_zero(in); }
    auto set1(float s) const { return v_set1(in, s); }
};

template <typename InT>
struct QuantSrc {
    const InT* in;
    float scale, zero_pt;

    v_f32x8 load_at(int64_t off) const { return dequantize_tensor(in + off, scale, zero_pt); }
    v_f32x8 zero() const { return v_zero_f32x8(); }
    v_f32x8 set1(float s) const { return v_set1_f32x8(s); }
};

/// Store R rows below the current one — r == 0 is the current output row.
template <typename T>
struct FpDst {
    T* out;
    int64_t row_stride;
    bool add_to;

    template <typename V>
    void store(int64_t r, int64_t ow, const V& v) const {
        v_store_add(out + r * row_stride + ow * 8, v, add_to);
    }
};

template <typename OutT>
struct QuantDst {
    OutT* out;
    int64_t row_stride;
    float scale, zero_pt;
    bool add_to;

    template <typename V>
    void store(int64_t r, int64_t ow, const V& v) const {
        requantize_store(out + r * row_stride + ow * 8, v, scale, zero_pt, add_to);
    }
};

}  // namespace nnops::kernel
