#pragma once
/// @file simd_reduce.hpp
/// @brief SIMD kernels for the Reduce operator.
///
/// Two dispatch paths:
///   1. Contiguous: multi-accumulator unrolling per row, delegated to row_reduce.
///   2. Inner-contiguous: SIMD across the inner (non-reduced) axes.
///
/// Only lane-8 vector types (v_f32x8 / v_f16x8), matching simd_lane_for<T>.

#include "nnops/detail/simd/simd.hpp"
#include "nnops/ops/reduce.hpp"
#include "simd_reduce_primitive.hpp"

#include <cfloat>

namespace nnops::kernel {

using namespace simd;

/// Contiguous: one row's reduction through the shared row_reduce primitive.
template <typename T>
inline void reduce_process_contiguous_row(
    const T* x_row, float* out_scalar,
    int64_t norm_size, ReduceType type)
{
    switch (type) {
    case ReduceType::Sum:
    case ReduceType::Mean: {
        const float sum = row_reduce<T, ReduceAdd>(x_row, norm_size);
        *out_scalar = (type == ReduceType::Mean)
            ? sum / static_cast<float>(norm_size) : sum;
        break;
    }
    case ReduceType::Max:
        *out_scalar = row_reduce<T, ReduceMax>(x_row, norm_size);
        break;
    case ReduceType::Min:
        *out_scalar = row_reduce<T, ReduceMin>(x_row, norm_size);
        break;
    }
}

/// Inner-contiguous: SIMD across the inner axis, L inner elements per block.
template <typename T>
inline void reduce_process_inner_contiguous_block(
    const T* x_base, T* y_base,
    int64_t inner_start, int64_t inner_end,
    int64_t reduce_size, int64_t axis_stride,
    ReduceType type, float inv_reduce)
{
    constexpr int L = simd_lane_for<T>;
    int64_t inner = inner_start;

    switch (type) {
    case ReduceType::Sum:
    case ReduceType::Mean: {
        for (; inner + L <= inner_end; inner += L) {
            auto v_sum0 = v_zero(x_base);
            auto v_sum1 = v_zero(x_base);
            auto v_sum2 = v_zero(x_base);
            auto v_sum3 = v_zero(x_base);

            int64_t k = 0;
            for (; k + 4 <= reduce_size; k += 4) {
                v_sum0 = v_add(v_sum0, v_load(x_base + (k + 0) * axis_stride + inner));
                v_sum1 = v_add(v_sum1, v_load(x_base + (k + 1) * axis_stride + inner));
                v_sum2 = v_add(v_sum2, v_load(x_base + (k + 2) * axis_stride + inner));
                v_sum3 = v_add(v_sum3, v_load(x_base + (k + 3) * axis_stride + inner));
            }
            auto v_sum = v_add(v_add(v_sum0, v_sum1), v_add(v_sum2, v_sum3));

            auto v_sum4 = v_zero(x_base);
            auto v_sum5 = v_zero(x_base);
            for (; k + 2 <= reduce_size; k += 2) {
                v_sum4 = v_add(v_sum4, v_load(x_base + (k + 0) * axis_stride + inner));
                v_sum5 = v_add(v_sum5, v_load(x_base + (k + 1) * axis_stride + inner));
            }
            v_sum = v_add(v_sum, v_add(v_sum4, v_sum5));

            for (; k < reduce_size; ++k) {
                v_sum = v_add(v_sum, v_load(x_base + k * axis_stride + inner));
            }

            if (type == ReduceType::Mean) {
                auto v_inv = v_set1(x_base, inv_reduce);
                v_store(y_base + inner, v_mul(v_sum, v_inv));
            } else {
                v_store(y_base + inner, v_sum);
            }
        }
        break;
    }

    case ReduceType::Max: {
        for (; inner + L <= inner_end; inner += L) {
            auto v_best0 = v_set1(x_base, -std::numeric_limits<float>::infinity());
            auto v_best1 = v_set1(x_base, -std::numeric_limits<float>::infinity());

            int64_t k = 0;
            for (; k + 2 <= reduce_size; k += 2) {
                v_best0 = v_max(v_best0, v_load(x_base + (k + 0) * axis_stride + inner));
                v_best1 = v_max(v_best1, v_load(x_base + (k + 1) * axis_stride + inner));
            }
            auto v_best = v_max(v_best0, v_best1);

            for (; k < reduce_size; ++k) {
                v_best = v_max(v_best, v_load(x_base + k * axis_stride + inner));
            }

            v_store(y_base + inner, v_best);
        }
        break;
    }

    case ReduceType::Min: {
        for (; inner + L <= inner_end; inner += L) {
            auto v_best0 = v_set1(x_base, std::numeric_limits<float>::infinity());
            auto v_best1 = v_set1(x_base, std::numeric_limits<float>::infinity());

            int64_t k = 0;
            for (; k + 2 <= reduce_size; k += 2) {
                v_best0 = v_min(v_best0, v_load(x_base + (k + 0) * axis_stride + inner));
                v_best1 = v_min(v_best1, v_load(x_base + (k + 1) * axis_stride + inner));
            }
            auto v_best = v_min(v_best0, v_best1);

            for (; k < reduce_size; ++k) {
                v_best = v_min(v_best, v_load(x_base + k * axis_stride + inner));
            }

            v_store(y_base + inner, v_best);
        }
        break;
    }
    }
}

}  // namespace nnops::kernel
