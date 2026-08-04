#pragma once
/// @file simd_rope.hpp
/// @brief SIMD-accelerated RoPE rotation kernels for a single position row.
///
/// Two paths:
///   - Split-half (!interleaved): full SIMD — 8 elements per iteration.
///     First-half and second-half pairs are at a fixed offset (head_dim/2),
///     so a single v_load of each half + cos/sin vectors is sufficient.
///   - Interleaved (interleaved): scalar fallback for now.
///     Full SIMD would require a per-pair shuffle not yet in the SIMD
///     abstraction (v_interleave / v_swap_adjacent).  Scalar via s_load/
///     s_store is correct and transparently handles f16<->f32 conversion.
///
/// All angles are computed externally (by the CPU backend) and passed as
/// pre-computed cos_table / sin_table arrays.  Each table has head_dim/2
/// entries for the current position.  For split-half, indices 0..half-1
/// correspond to the rotation angle for pair (d, d+half).  For interleaved,
/// index k corresponds to the pair (2k, 2k+1).

#include "nnops/detail/simd/simd.hpp"

#include <type_traits>

namespace nnops::kernel {

using namespace simd;

// ============================================================
// Split-half SIMD rotation (GPT-NeoX convention)
// ============================================================

template <typename T>
inline void rope_process_row_splithalf(
    T* y, const T* x,
    int64_t head_dim,
    const float* cos_table,
    const float* sin_table,
    bool add_to)
{
    const int64_t half = head_dim / 2;

    // SIMD path only for f32 (cos/sin tables are always float, so mixing
    // v_f16x8 with v_f32x8 cos/sin vectors is not type-safe).
    if constexpr (std::is_same_v<T, float>) {
        constexpr int L = simd_lane_for<T>;  // 8
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

            v_store_add(y + d,       v_sub(x0c, x1s), add_to);
            v_store_add(y + d + half, v_add(x1c, x0s), add_to);
        }

        // Scalar tail
        for (; d < half; ++d) {
            float x0v = s_load(&x[d]);
            float x1v = s_load(&x[d + half]);
            float cv  = cos_table[d];
            float sv  = sin_table[d];
            s_store_add(&y[d],        x0v * cv - x1v * sv, add_to);
            s_store_add(&y[d + half], x1v * cv + x0v * sv, add_to);
        }
    } else {
        // f16: all-scalar (s_load transparently converts f16→f32→f16)
        for (int64_t d = 0; d < half; ++d) {
            float x0v = s_load(&x[d]);
            float x1v = s_load(&x[d + half]);
            float cv  = cos_table[d];
            float sv  = sin_table[d];
            s_store_add(&y[d],        x0v * cv - x1v * sv, add_to);
            s_store_add(&y[d + half], x1v * cv + x0v * sv, add_to);
        }
    }
}

// ============================================================
// Interleaved scalar rotation (LLaMA convention)
// ============================================================

template <typename T>
inline void rope_process_row_interleaved(
    T* y, const T* x,
    int64_t head_dim,
    const float* cos_table,
    const float* sin_table,
    bool add_to)
{
    for (int64_t d = 0; d < head_dim; d += 2) {
        float x0 = s_load(&x[d]);
        float x1 = s_load(&x[d + 1]);
        float cv = cos_table[d / 2];
        float sv = sin_table[d / 2];

        float y0 = x0 * cv - x1 * sv;
        float y1 = x1 * cv + x0 * sv;

        s_store_add(&y[d],     y0, add_to);
        s_store_add(&y[d + 1], y1, add_to);
    }
}

// ============================================================
// Unified dispatch
// ============================================================

template <typename T>
inline void rope_process_row(
    T* y, const T* x,
    int64_t head_dim,
    const float* cos_table,
    const float* sin_table,
    bool interleaved,
    bool add_to)
{
    if (interleaved) {
        rope_process_row_interleaved<T>(y, x, head_dim, cos_table, sin_table, add_to);
    } else {
        rope_process_row_splithalf<T>(y, x, head_dim, cos_table, sin_table, add_to);
    }
}

}  // namespace nnops::kernel
