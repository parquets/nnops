#pragma once
/// @file simd_reduce.hpp
/// @brief SIMD kernel functions for the Reduce operator.
///
/// Three SIMD dispatch paths:
///   1. Packed: per-lane SIMD reduction.
///   2. Contiguous: multi-accumulator unrolling per row.
///   3. Inner-contiguous: SIMD across axis for L inner elements.
///
/// Only lane=8 SIMD types (v_f32x8 / v_f16x8), matching simd_lane_for<T>.

#include "nnops/detail/simd/simd.hpp"
#include "nnops/ops/reduce.hpp"

#include <cfloat>

namespace nnops::kernel {

using namespace simd;

/// Packed SIMD: per-lane reduction over D spatial positions.
template <typename T>
inline void reduce_process_packed_row(
    const T* x_row, T* y_pos,
    int64_t D, int64_t pack,
    ReduceType type, float inv_D)
{
    switch (type) {
    case ReduceType::Sum:
    case ReduceType::Mean: {
        auto v_sum = v_zero(x_row);
        for (int64_t d = 0; d < D; ++d)
            v_sum = v_add(v_sum, v_load(x_row + d * pack));
        if (type == ReduceType::Mean) {
            auto v_inv = v_set1(x_row, inv_D);
            v_sum = v_mul(v_sum, v_inv);
        }
        v_store(y_pos, v_sum);
        break;
    }
    case ReduceType::Max: {
        auto v_best = v_set1(x_row, -std::numeric_limits<float>::infinity());
        for (int64_t d = 0; d < D; ++d)
            v_best = v_max(v_best, v_load(x_row + d * pack));
        v_store(y_pos, v_best);
        break;
    }
    case ReduceType::Min: {
        auto v_best = v_set1(x_row, std::numeric_limits<float>::infinity());
        for (int64_t d = 0; d < D; ++d)
            v_best = v_min(v_best, v_load(x_row + d * pack));
        v_store(y_pos, v_best);
        break;
    }
    }
}

/// Packed channel SIMD: per-lane reduction over D C8 blocks at a single
/// spatial position.  x_chan_stride is the element stride between consecutive
/// C8 blocks (= stride_elems(1) in NCHWC8).
template <typename T>
inline void reduce_process_packed_channel(
    const T* x_chan, T* y_pos,
    int64_t x_chan_stride,
    int64_t D, int64_t pack,
    ReduceType type, float inv_D)
{
    switch (type) {
    case ReduceType::Sum:
    case ReduceType::Mean: {
        auto v_sum = v_zero(x_chan);
        for (int64_t c = 0; c < D; ++c)
            v_sum = v_add(v_sum, v_load(x_chan + c * x_chan_stride));
        if (type == ReduceType::Mean) {
            auto v_inv = v_set1(x_chan, inv_D);
            v_sum = v_mul(v_sum, v_inv);
        }
        v_store(y_pos, v_sum);
        break;
    }
    case ReduceType::Max: {
        auto v_best = v_set1(x_chan, -std::numeric_limits<float>::infinity());
        for (int64_t c = 0; c < D; ++c)
            v_best = v_max(v_best, v_load(x_chan + c * x_chan_stride));
        v_store(y_pos, v_best);
        break;
    }
    case ReduceType::Min: {
        auto v_best = v_set1(x_chan, std::numeric_limits<float>::infinity());
        for (int64_t c = 0; c < D; ++c)
            v_best = v_min(v_best, v_load(x_chan + c * x_chan_stride));
        v_store(y_pos, v_best);
        break;
    }
    }
}

/// Contiguous tail: multi-accumulator unrolling per row.
template <typename T>
inline void reduce_process_contiguous_row(
    const T* x_row, float* out_scalar,
    int64_t norm_size, ReduceType type)
{
    constexpr int L = simd_lane_for<T>;
    int64_t i = 0;

    switch (type) {
    case ReduceType::Sum:
    case ReduceType::Mean: {
        auto v_sum0 = v_zero(x_row);
        auto v_sum1 = v_zero(x_row);
        auto v_sum2 = v_zero(x_row);
        auto v_sum3 = v_zero(x_row);

        for (; i + 4 * L <= norm_size; i += 4 * L) {
            v_sum0 = v_add(v_sum0, v_load(x_row + i));
            v_sum1 = v_add(v_sum1, v_load(x_row + i + L));
            v_sum2 = v_add(v_sum2, v_load(x_row + i + 2 * L));
            v_sum3 = v_add(v_sum3, v_load(x_row + i + 3 * L));
        }
        auto v_sum = v_add(v_add(v_sum0, v_sum1), v_add(v_sum2, v_sum3));

        auto v_sum4 = v_zero(x_row);
        auto v_sum5 = v_zero(x_row);
        for (; i + 2 * L <= norm_size; i += 2 * L) {
            v_sum4 = v_add(v_sum4, v_load(x_row + i));
            v_sum5 = v_add(v_sum5, v_load(x_row + i + L));
        }
        v_sum = v_add(v_sum, v_add(v_sum4, v_sum5));

        for (; i + L <= norm_size; i += L) {
            v_sum = v_add(v_sum, v_load(x_row + i));
        }
        float sum = v_reduce_sum(v_sum);

        for (; i < norm_size; ++i) {
            sum += s_load(&x_row[i]);
        }
        *out_scalar = (type == ReduceType::Mean)
            ? sum / static_cast<float>(norm_size) : sum;
        break;
    }

    case ReduceType::Max: {
        auto v_best0 = v_set1(x_row, -std::numeric_limits<float>::infinity());
        auto v_best1 = v_set1(x_row, -std::numeric_limits<float>::infinity());

        for (; i + 2 * L <= norm_size; i += 2 * L) {
            v_best0 = v_max(v_best0, v_load(x_row + i));
            v_best1 = v_max(v_best1, v_load(x_row + i + L));
        }
        auto v_best = v_max(v_best0, v_best1);

        for (; i + L <= norm_size; i += L) {
            v_best = v_max(v_best, v_load(x_row + i));
        }
        float best = v_reduce_max(v_best);

        for (; i < norm_size; ++i) {
            float xv = s_load(&x_row[i]);
            if (xv > best) { best = xv; }
        }
        *out_scalar = best;
        break;
    }

    case ReduceType::Min: {
        auto v_best0 = v_set1(x_row, std::numeric_limits<float>::infinity());
        auto v_best1 = v_set1(x_row, std::numeric_limits<float>::infinity());

        for (; i + 2 * L <= norm_size; i += 2 * L) {
            v_best0 = v_min(v_best0, v_load(x_row + i));
            v_best1 = v_min(v_best1, v_load(x_row + i + L));
        }
        auto v_best = v_min(v_best0, v_best1);

        for (; i + L <= norm_size; i += L) {
            v_best = v_min(v_best, v_load(x_row + i));
        }
        float best = v_reduce_min(v_best);

        for (; i < norm_size; ++i) {
            float xv = s_load(&x_row[i]);
            if (xv < best) { best = xv; }
        }
        *out_scalar = best;
        break;
    }
    }
}

/// Inner-contiguous: SIMD across axis for L inner elements.
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
