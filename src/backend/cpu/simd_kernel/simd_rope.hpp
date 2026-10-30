#pragma once
/// @file simd_rope.hpp
/// @brief SIMD RoPE rotation kernels for a single position row.
///
/// Split-half (!interleaved, GPT-NeoX): full SIMD, 8 elements per iteration.
/// A pair sits at a fixed offset head_dim/2, so one v_load per half plus the
/// cos/sin vectors covers it. Only f32 takes this path: the tables are float,
/// and v_mul has no mixed v_f16x8 / v_f32x8 overload. f16 falls back to scalar
/// through s_load/s_store, which convert f16↔f32 transparently.
///
/// Interleaved (LLaMA): scalar. A SIMD form would need a per-pair shuffle
/// (v_interleave / v_swap_adjacent) the abstraction does not provide.
///
/// Angles are computed by the CPU backend and passed as cos_table / sin_table,
/// head_dim/2 entries per position: for split-half, index d is the angle of the
/// pair (d, d+half); for interleaved, index k is the pair (2k, 2k+1).

#include "nnops/detail/simd/simd.hpp"

#include <type_traits>

namespace nnops::kernel {

using namespace simd;

template <typename T>
inline void rope_process_row_splithalf(
    T* y, const T* x,
    int64_t head_dim,
    const float* cos_table,
    const float* sin_table)
{
    const int64_t half = head_dim / 2;

    if constexpr (std::is_same_v<T, float>) {
        constexpr int L = simd_lane_for<T>;
        int64_t d = 0;
        for (; d + L <= half; d += L) {
            auto x0   = v_load(x + d);
            auto x1   = v_load(x + d + half);
            auto c    = v_load(cos_table + d);
            auto s    = v_load(sin_table + d);

            auto x0c  = v_mul(x0, c);
            auto x0s  = v_mul(x0, s);
            auto x1c  = v_mul(x1, c);
            auto x1s  = v_mul(x1, s);

            v_store(y + d,       v_sub(x0c, x1s));
            v_store(y + d + half, v_add(x1c, x0s));
        }

        for (; d < half; ++d) {
            float x0v = s_load(&x[d]);
            float x1v = s_load(&x[d + half]);
            float cv  = cos_table[d];
            float sv  = sin_table[d];
            s_store(&y[d],        x0v * cv - x1v * sv);
            s_store(&y[d + half], x1v * cv + x0v * sv);
        }
    } else {
        for (int64_t d = 0; d < half; ++d) {
            float x0v = s_load(&x[d]);
            float x1v = s_load(&x[d + half]);
            float cv  = cos_table[d];
            float sv  = sin_table[d];
            s_store(&y[d],        x0v * cv - x1v * sv);
            s_store(&y[d + half], x1v * cv + x0v * sv);
        }
    }
}

template <typename T>
inline void rope_process_row_interleaved(
    T* y, const T* x,
    int64_t head_dim,
    const float* cos_table,
    const float* sin_table)
{
    for (int64_t d = 0; d < head_dim; d += 2) {
        float x0 = s_load(&x[d]);
        float x1 = s_load(&x[d + 1]);
        float cv = cos_table[d / 2];
        float sv = sin_table[d / 2];

        float y0 = x0 * cv - x1 * sv;
        float y1 = x1 * cv + x0 * sv;

        s_store(&y[d],     y0);
        s_store(&y[d + 1], y1);
    }
}

template <typename T>
inline void rope_process_row(
    T* y, const T* x,
    int64_t head_dim,
    const float* cos_table,
    const float* sin_table,
    bool interleaved)
{
    if (interleaved) {
        rope_process_row_interleaved<T>(y, x, head_dim, cos_table, sin_table);
    } else {
        rope_process_row_splithalf<T>(y, x, head_dim, cos_table, sin_table);
    }
}

}  // namespace nnops::kernel
