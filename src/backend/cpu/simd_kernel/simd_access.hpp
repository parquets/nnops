#pragma once
/// @file simd_access.hpp
/// @brief Load/store policies shared by the spatial kernels (pooling, depthwise).
///
/// Fp*    — the element type is the accumulator (f32 or f16).
/// Quant* — s8/u8 activations dequantize to f32 on load, requantize on store.
/// A kernel that takes these as template parameters needs no per-type copy.
///
/// @c Full says whether the channel block is a whole vector. A channel-packed
/// block always is (NCHWC8 pads C to a block boundary), but a channel-last
/// plane's last block holds C % 8 channels, and an 8-wide load there would take
/// in the neighbouring pixel's channels — in bounds, so no fault, but three
/// lanes of garbage. The short form fills those lanes from a stack copy on
/// load and writes element by element on store, and it is a separate
/// instantiation, so the common path pays nothing: C % 8 == 0 never reaches it.

#include "nnops/detail/simd/simd.hpp"
#include "simd_quant.hpp"

#include <cstdint>
#include <cstring>

namespace nnops::kernel {

using namespace simd;

/// Full SIMD width for both fp vector types (v_f32x8 / v_f16x8).
constexpr int64_t kChannelLanes = 8;

/// Load up to @p lanes elements, zero-filling the rest — a channel vector whose
/// block is short. Used for the bias and by the policies' short form.
template <typename T>
inline auto load_lanes(const T* p, int64_t lanes)
{
    if (lanes == kChannelLanes) { return v_load(p); }
    T tmp[kChannelLanes] = {};
    std::memcpy(tmp, p, static_cast<size_t>(lanes) * sizeof(T));
    return v_load(tmp);
}

template <typename T, bool Full = true>
struct FpSrc {
    using elem = T;
    const T* in;    // also picks the v_set1/v_zero overload
    int64_t lanes = kChannelLanes;

    const T* ptr_at(int64_t off) const { return in + off; }
    auto load_at(int64_t off) const { return load_ptr(in + off); }
    /// Load through an address the caller already built — the spatial kernels
    /// keep a per-tap row pointer so the width loop does no address arithmetic.
    auto load_ptr(const T* p) const {
        if constexpr (Full) {
            return v_load(p);
        } else {
            return load_lanes(p, lanes);
        }
    }
    auto zero() const { return v_zero(in); }
    auto set1(float s) const { return v_set1(in, s); }
};

template <typename InT, bool Full = true>
struct QuantSrc {
    using elem = InT;
    const InT* in;
    float scale, zero_pt;
    int64_t lanes = kChannelLanes;

    const InT* ptr_at(int64_t off) const { return in + off; }
    v_f32x8 load_at(int64_t off) const { return load_ptr(in + off); }
    v_f32x8 load_ptr(const InT* p) const {
        if constexpr (Full) {
            return dequantize_tensor(p, scale, zero_pt);
        } else {
            InT tmp[kChannelLanes] = {};
            std::memcpy(tmp, p, static_cast<size_t>(lanes) * sizeof(InT));
            return dequantize_tensor(tmp, scale, zero_pt);
        }
    }
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
/// @c pix is the element distance between two adjacent output positions: 8 for
/// a channel-packed block (consecutive pixels of one block are a vector apart)
/// and C for a channel-last plane (they are a whole pixel apart).
template <typename T, bool Full = true>
struct FpDst {
    T* out;
    int64_t row_stride;
    bool add_to;
    int64_t pix = kChannelLanes;
    int64_t lanes = kChannelLanes;

    template <typename V>
    void store(int64_t r, int64_t ow, const V& v) const {
        T* p = out + r * row_stride + ow * pix;
        if constexpr (Full) {
            v_store_add(p, v, add_to);
        } else {
            T tmp[kChannelLanes];
            v_store(tmp, v);
            for (int64_t l = 0; l < lanes; ++l) {
                s_store_add(p + l, s_load(tmp + l), add_to);
            }
        }
    }
};

template <typename OutT, bool Full = true>
struct QuantDst {
    OutT* out;
    int64_t row_stride;
    float scale, zero_pt;
    bool add_to;
    int64_t pix = kChannelLanes;
    int64_t lanes = kChannelLanes;

    template <typename V>
    void store(int64_t r, int64_t ow, const V& v) const {
        OutT* p = out + r * row_stride + ow * pix;
        if constexpr (Full) {
            requantize_store(p, v, scale, zero_pt, add_to);
        } else {
            // Requantize into a staging vector and copy back the live bytes
            // only: the dead lanes read and write nothing that survives, so
            // the neighbouring pixel's channels are untouched.
            OutT tmp[kChannelLanes] = {};
            if (add_to) {
                std::memcpy(tmp, p, static_cast<size_t>(lanes) * sizeof(OutT));
            }
            requantize_store(tmp, v, scale, zero_pt, add_to);
            std::memcpy(p, tmp, static_cast<size_t>(lanes) * sizeof(OutT));
        }
    }
};

}  // namespace nnops::kernel
