#pragma once
/// @file simd_mul_add_reduce.hpp
/// @brief Scaled residual add fused with a row's sum and sum-of-squares:
/// out[i] = a[i] + b[i] * scale, accumulating both statistics in the same pass
/// so a following normalize can derive mean/var (LayerNorm) or rms (RMSNorm)
/// without re-reading the row.
///
/// 4-wide multi-accumulator unrolling for ILP, then 2-wide → 1-wide SIMD →
/// scalar tail. Both statistics are returned as float regardless of T, though
/// a half input accumulates in half lanes; the output is written back in T.

#include "nnops/detail/simd/simd.hpp"

#include <utility>

namespace nnops::kernel {

using namespace simd;

/// Returns {sum, sum_sq} of `out` over n contiguous elements, both float.
template <typename T>
inline std::pair<float, float> mul_add_reduce_square_sum(
    const T* a, const T* b, float scale, T* out, int64_t n)
{
    constexpr int L = simd_lane_for<T>;
    const auto v_scale = v_set1(a, scale);
    int64_t i = 0;
    float sum = 0.0f;
    float sum_sq = 0.0f;

    {
        auto v_sum0 = v_zero(a);
        auto v_sum_sq0 = v_zero(a);
        auto v_sum1 = v_zero(a);
        auto v_sum_sq1 = v_zero(a);
        auto v_sum2 = v_zero(a);
        auto v_sum_sq2 = v_zero(a);
        auto v_sum3 = v_zero(a);
        auto v_sum_sq3 = v_zero(a);

        for (; i + 4 * L <= n; i += 4 * L) {
            auto v0 = v_fmadd(v_load(b + i), v_scale, v_load(a + i));
            auto v1 = v_fmadd(v_load(b + i + L), v_scale, v_load(a + i + L));
            auto v2 = v_fmadd(v_load(b + i + 2 * L), v_scale, v_load(a + i + 2 * L));
            auto v3 = v_fmadd(v_load(b + i + 3 * L), v_scale, v_load(a + i + 3 * L));
            v_store(out + i, v0);
            v_store(out + i + L, v1);
            v_store(out + i + 2 * L, v2);
            v_store(out + i + 3 * L, v3);
            v_sum0 = v_add(v_sum0, v0);
            v_sum_sq0 = v_fmadd(v0, v0, v_sum_sq0);
            v_sum1 = v_add(v_sum1, v1);
            v_sum_sq1 = v_fmadd(v1, v1, v_sum_sq1);
            v_sum2 = v_add(v_sum2, v2);
            v_sum_sq2 = v_fmadd(v2, v2, v_sum_sq2);
            v_sum3 = v_add(v_sum3, v3);
            v_sum_sq3 = v_fmadd(v3, v3, v_sum_sq3);
        }
        auto v_sum = v_add(v_add(v_sum0, v_sum1), v_add(v_sum2, v_sum3));
        auto v_sum_sq = v_add(v_add(v_sum_sq0, v_sum_sq1),
                              v_add(v_sum_sq2, v_sum_sq3));

        auto v_sum4 = v_zero(a);
        auto v_sum_sq4 = v_zero(a);
        auto v_sum5 = v_zero(a);
        auto v_sum_sq5 = v_zero(a);
        for (; i + 2 * L <= n; i += 2 * L) {
            auto v4 = v_fmadd(v_load(b + i), v_scale, v_load(a + i));
            auto v5 = v_fmadd(v_load(b + i + L), v_scale, v_load(a + i + L));
            v_store(out + i, v4);
            v_store(out + i + L, v5);
            v_sum4 = v_add(v_sum4, v4);
            v_sum_sq4 = v_fmadd(v4, v4, v_sum_sq4);
            v_sum5 = v_add(v_sum5, v5);
            v_sum_sq5 = v_fmadd(v5, v5, v_sum_sq5);
        }
        v_sum = v_add(v_sum, v_add(v_sum4, v_sum5));
        v_sum_sq = v_add(v_sum_sq, v_add(v_sum_sq4, v_sum_sq5));

        for (; i + L <= n; i += L) {
            auto v = v_fmadd(v_load(b + i), v_scale, v_load(a + i));
            v_store(out + i, v);
            v_sum = v_add(v_sum, v);
            v_sum_sq = v_fmadd(v, v, v_sum_sq);
        }

        sum = v_reduce_sum(v_sum);
        sum_sq = v_reduce_sum(v_sum_sq);
    }

    for (; i < n; ++i) {
        float s = s_load(&a[i]) + s_load(&b[i]) * scale;
        s_store(&out[i], s);
        sum += s;
        sum_sq += s * s;
    }

    return {sum, sum_sq};
}

}  // namespace nnops::kernel