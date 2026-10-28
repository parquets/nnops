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

/// Load one input row into a vector whose lanes are consecutive output
/// positions — lane l reads input column iw0 + l*sw. Used by the planar
/// (NCHW/NCDHW) spatial kernels, where the lane spans output width instead of
/// channels, so an out-of-range column has to be filled with the accumulate's
/// identity (@p fill) rather than skipped. Unit and doubled stride take a
/// plain (or deinterleaving) load; any other stride gathers scalar-wise.
template <typename T>
inline auto planar_load(const T* row, int64_t iw0, int64_t sw, int64_t iw, T fill)
{
    constexpr int64_t L = simd_lane_for<T>;
    if (iw0 >= 0) {
        if (sw == 1 && iw0 + L <= iw) { return v_load(row + iw0); }
        if (sw == 2 && iw0 + 2 * L <= iw) { return v_load_even(row + iw0); }
    }
    T tmp[L];
    for (int64_t l = 0; l < L; ++l) {
        const int64_t i = iw0 + l * sw;
        tmp[l] = (i >= 0 && i < iw) ? row[i] : fill;
    }
    return v_load(tmp);
}

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
