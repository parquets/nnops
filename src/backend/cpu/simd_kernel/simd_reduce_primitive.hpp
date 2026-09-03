#pragma once
/// @file simd_reduce_primitive.hpp
/// @brief Shared reduction primitives for contiguous rows.
///
/// Two of the per-operator kernels — Reduce, Softmax and Norm — independently
/// re-implemented the same "multi-accumulator reduce a contiguous row to a
/// scalar" pattern (4-wide ILP, 2-wide, 1-wide, scalar tail) and the same
/// horizontal max/min/sum reductions. This header is the single source of
/// truth for that pattern:
///
///   - ReduceAdd / ReduceMax / ReduceMin: tag types capturing a reduction's
///     identity, vector combine, horizontal reduction and scalar combine.
///   - row_reduce<T, Op>: generic 4→2→1 multi-accumulator reduce to scalar.
///   - row_reduce_sum_sq<T>: fused sum + sum-of-squares reduce (moved from
///     simd_norm.hpp so all reduction primitives live in one place).
///
/// Only lane=8 SIMD types (v_f32x8 / v_f16x8), matching simd_lane_for<T>.

#include "nnops/detail/simd/simd.hpp"

#include <limits>
#include <utility>

namespace nnops::kernel {

using namespace simd;

// ============================================================
// Reduction op tags
// ============================================================

/// The vector-level members are templates so a single tag works for both
/// v_f32x8 and v_f16x8; the compiler instantiates only the width it needs.
struct ReduceAdd {
    template <typename P>
    static auto vec_identity(const P* p) { return v_zero(p); }
    template <typename V>
    static V combine(V a, V b) { return v_add(a, b); }
    template <typename V>
    static float horizontal(V v) { return v_reduce_sum(v); }
    static float combine_scalar(float a, float b) { return a + b; }
};

struct ReduceMax {
    template <typename P>
    static auto vec_identity(const P* p) {
        return v_set1(p, -std::numeric_limits<float>::infinity());
    }
    template <typename V>
    static V combine(V a, V b) { return v_max(a, b); }
    template <typename V>
    static float horizontal(V v) { return v_reduce_max(v); }
    static float combine_scalar(float a, float b) { return a > b ? a : b; }
};

struct ReduceMin {
    template <typename P>
    static auto vec_identity(const P* p) {
        return v_set1(p, std::numeric_limits<float>::infinity());
    }
    template <typename V>
    static V combine(V a, V b) { return v_min(a, b); }
    template <typename V>
    static float horizontal(V v) { return v_reduce_min(v); }
    static float combine_scalar(float a, float b) { return a < b ? a : b; }
};

// ============================================================
// Contiguous row reduction
// ============================================================

/// Reduce one contiguous row to a scalar via multi-accumulator unrolling:
/// 4-wide ILP → 2-wide → 1-wide SIMD → scalar tail. `Op` supplies the
/// reduction semantics (ReduceAdd/ReduceMax/ReduceMin); the accumulate order
/// is fixed so sum results are bit-identical across all users.
template <typename T, typename Op>
inline float row_reduce(const T* x, int64_t n)
{
    constexpr int L = simd_lane_for<T>;
    int64_t i = 0;

    // ---- Stage 1: 4-wide ILP ----
    auto a0 = Op::vec_identity(x);
    auto a1 = Op::vec_identity(x);
    auto a2 = Op::vec_identity(x);
    auto a3 = Op::vec_identity(x);
    for (; i + 4 * L <= n; i += 4 * L) {
        a0 = Op::combine(a0, v_load(x + i));
        a1 = Op::combine(a1, v_load(x + i + L));
        a2 = Op::combine(a2, v_load(x + i + 2 * L));
        a3 = Op::combine(a3, v_load(x + i + 3 * L));
    }
    auto acc = Op::combine(Op::combine(a0, a1), Op::combine(a2, a3));

    // ---- Stage 2: 2-wide ----
    auto a4 = Op::vec_identity(x);
    auto a5 = Op::vec_identity(x);
    for (; i + 2 * L <= n; i += 2 * L) {
        a4 = Op::combine(a4, v_load(x + i));
        a5 = Op::combine(a5, v_load(x + i + L));
    }
    acc = Op::combine(acc, Op::combine(a4, a5));

    // ---- Stage 3: 1-wide ----
    for (; i + L <= n; i += L) {
        acc = Op::combine(acc, v_load(x + i));
    }
    float s = Op::horizontal(acc);

    // ---- Stage 4: scalar tail ----
    for (; i < n; ++i) {
        s = Op::combine_scalar(s, s_load(&x[i]));
    }
    return s;
}

/// SIMD sum + sum-of-squares reduction over n contiguous elements.
///
/// Uses 4-wide multi-accumulator unrolling for ILP, then falls back
/// through 2-wide → 1-wide SIMD → scalar tail.
///
/// Returns {sum, sum_sq} as float regardless of T (fp16 values are promoted).
template <typename T>
inline std::pair<float, float> row_reduce_sum_sq(const T* x, int64_t n)
{
    constexpr int L = simd_lane_for<T>;
    int64_t i = 0;
    float sum = 0.0f;
    float sum_sq = 0.0f;

    // ---- Stage 1: 4-wide SIMD reduction (ILP ×4) ----
    {
        auto v_sum0 = v_zero(x);
        auto v_sum_sq0 = v_zero(x);
        auto v_sum1 = v_zero(x);
        auto v_sum_sq1 = v_zero(x);
        auto v_sum2 = v_zero(x);
        auto v_sum_sq2 = v_zero(x);
        auto v_sum3 = v_zero(x);
        auto v_sum_sq3 = v_zero(x);

        for (; i + 4 * L <= n; i += 4 * L) {
            auto v0 = v_load(x + i);
            auto v1 = v_load(x + i + L);
            auto v2 = v_load(x + i + 2 * L);
            auto v3 = v_load(x + i + 3 * L);
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
        auto v_sum4 = v_zero(x);
        auto v_sum_sq4 = v_zero(x);
        auto v_sum5 = v_zero(x);
        auto v_sum_sq5 = v_zero(x);
        for (; i + 2 * L <= n; i += 2 * L) {
            auto v4 = v_load(x + i);
            auto v5 = v_load(x + i + L);
            v_sum4 = v_add(v_sum4, v4);
            v_sum_sq4 = v_fmadd(v4, v4, v_sum_sq4);
            v_sum5 = v_add(v_sum5, v5);
            v_sum_sq5 = v_fmadd(v5, v5, v_sum_sq5);
        }
        v_sum = v_add(v_sum, v_add(v_sum4, v_sum5));
        v_sum_sq = v_add(v_sum_sq, v_add(v_sum_sq4, v_sum_sq5));

        // ---- Stage 3: 1-wide SIMD ----
        for (; i + L <= n; i += L) {
            auto v = v_load(x + i);
            v_sum = v_add(v_sum, v);
            v_sum_sq = v_fmadd(v, v, v_sum_sq);
        }

        sum = v_reduce_sum(v_sum);
        sum_sq = v_reduce_sum(v_sum_sq);
    }

    // ---- Stage 4: scalar tail ----
    for (; i < n; ++i) {
        float xv = s_load(&x[i]);
        sum += xv;
        sum_sq += xv * xv;
    }

    return {sum, sum_sq};
}

}  // namespace nnops::kernel
