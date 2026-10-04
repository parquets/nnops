#pragma once
/// @file tiled_map.hpp
/// @brief Generic row-tiled element-wise mapper — single source of truth for
///        the SIMD + scalar-tail row loop shared by every element-wise op.
///
/// The only difference between unary and binary (and future n-ary) element-wise
/// kernels is the arity, captured here by the compile-time `Nin` (number of inputs).
///
/// The op-specific work is supplied as two functors over the loaded values:
///   - `simd_kernel(v0, v1, ...)  -> Vec`  for the vectorized (lane=8) path
///   - `scalar_kernel(s0, s1, ...) -> float` for the scalar tail
///
/// Both functors receive exactly `Nin` arguments; `std::array` + an index
/// sequence expand the loaded inputs into the right call shape. `add_to`
/// semantics are handled here (v_store_add / s_store_add), so callers never
/// branch on it.

#include "nnops/detail/simd/simd.hpp"

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

namespace nnops::kernel {

using namespace simd;

/// Vector type produced by `v_load` for element type `T` (v_f32x8 / v_f16x8).
template <typename T>
using simd_vec_t = decltype(v_load(std::declval<const T*>()));

/// Invoke `fn` with the elements of `arr` expanded as separate arguments.
template <typename Fn, typename Arr, std::size_t... I>
inline auto apply_array(Fn& fn, const Arr& arr, std::index_sequence<I...>) -> decltype(auto) {
    return fn(std::get<I>(arr)...);
}

/// Row-tiled mapper with a SIMD inner loop and scalar tail, for a `Nin`-ary op.
///
/// `in` holds the `Nin` input base pointers; `in_pitch` the corresponding row
/// strides (elements). The output is written at `out` with row stride
/// `out_pitch`. When `add_to`, the result is accumulated onto the output.
template <typename T, int Nin, typename SimdK, typename ScalarK>
inline void tiled_map_simd(const std::array<const T*, Nin>& in,
                           const std::array<int64_t, Nin>& in_pitch,
                           T* out, int64_t out_pitch,
                           int64_t m, int64_t n, bool add_to,
                           SimdK&& simd_kernel, ScalarK&& scalar_kernel)
{
    constexpr int L = simd_lane_for<T>;  // 8 for both f32 and f16
    for (int64_t r = 0; r < m; ++r) {
        T* out_row = out + r * out_pitch;
        int64_t i = 0;
        for (; i + L <= n; i += L) {
            std::array<simd_vec_t<T>, Nin> v;
            for (int j = 0; j < Nin; ++j) {
                v[j] = v_load(in[j] + r * in_pitch[j] + i);
            }
            v_store_add(out_row + i,
                        apply_array(simd_kernel, v, std::make_index_sequence<Nin>{}),
                        add_to);
        }
        for (; i < n; ++i) {
            std::array<float, Nin> s;
            for (int j = 0; j < Nin; ++j) {
                s[j] = s_load(&in[j][r * in_pitch[j] + i]);
            }
            s_store_add(&out_row[i],
                        apply_array(scalar_kernel, s, std::make_index_sequence<Nin>{}),
                        add_to);
        }
    }
}

/// Scalar-only row-tiled mapper for a `Nin`-ary op without SIMD intrinsics.
template <typename T, int Nin, typename ScalarK>
inline void tiled_map_scalar(const std::array<const T*, Nin>& in,
                             const std::array<int64_t, Nin>& in_pitch,
                             T* out, int64_t out_pitch,
                             int64_t m, int64_t n, bool add_to,
                             ScalarK&& scalar_kernel)
{
    for (int64_t r = 0; r < m; ++r) {
        T* out_row = out + r * out_pitch;
        for (int64_t i = 0; i < n; ++i) {
            std::array<float, Nin> s;
            for (int j = 0; j < Nin; ++j) {
                s[j] = s_load(&in[j][r * in_pitch[j] + i]);
            }
            s_store_add(&out_row[i],
                        apply_array(scalar_kernel, s, std::make_index_sequence<Nin>{}),
                        add_to);
        }
    }
}

}  // namespace nnops::kernel
