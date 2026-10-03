#pragma once
/// @file simd_add_fuse_norm.hpp
/// @brief Fused multiply-add + sum / sum-of-squares primitives for the
/// AddFuseNorm operator (scaled residual add fused with normalization
/// statistics).
///
/// Pass 1 of AddFuseNorm: for one contiguous row, write
/// out[i] = a[i] + b[i] * scale while simultaneously accumulating that row's
/// sum and sum-of-squares, so the subsequent normalize pass can derive
/// mean/var (LayerNorm) or rms (RMSNorm) without re-reading the row.

#include "nnops/detail/simd/simd.hpp"

#include <utility>

namespace nnops::kernel {

using namespace simd;

/// Fused multiply-add + reduce over n contiguous elements.
///
/// Computes `out[i] = a[i] + b[i] * scale` for i in [0, n) and returns
/// `{sum, sum_sq}` where sum = Σ out[i], sum_sq = Σ out[i]².
///
/// Both statistics are returned as float regardless of T (fp16 inputs are
/// promoted); the output is written back in T. Uses 4-wide multi-accumulator
/// unrolling for ILP, then 2-wide → 1-wide SIMD → scalar tail — the same
/// shape as row_reduce_sum_sq in simd_reduce_primitive.hpp.
template <typename T>
inline std::pair<float, float> mul_add_reduce_square_sum(
    const T* a, const T* b, float scale, T* out, int64_t n)
{
    constexpr int L = simd_lane_for<T>;
    const auto v_scale = v_set1(a, scale);
    int64_t i = 0;
    float sum = 0.0f;
    float sum_sq = 0.0f;

    // ---- Stage 1: 4-wide SIMD (ILP ×4) ----
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

        // ---- Stage 2: 2-wide SIMD ----
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

        // ---- Stage 3: 1-wide SIMD ----
        for (; i + L <= n; i += L) {
            auto v = v_fmadd(v_load(b + i), v_scale, v_load(a + i));
            v_store(out + i, v);
            v_sum = v_add(v_sum, v);
            v_sum_sq = v_fmadd(v, v, v_sum_sq);
        }

        sum = v_reduce_sum(v_sum);
        sum_sq = v_reduce_sum(v_sum_sq);
    }

    // ---- Stage 4: scalar tail ----
    for (; i < n; ++i) {
        float s = s_load(&a[i]) + s_load(&b[i]) * scale;
        s_store(&out[i], s);
        sum += s;
        sum_sq += s * s;
    }

    return {sum, sum_sq};
}

}  // namespace nnops::kernel