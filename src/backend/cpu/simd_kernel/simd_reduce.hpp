#pragma once
/// @file simd_reduce.hpp
/// @brief SIMD kernels for the Reduce operator.
///
/// Three dispatch paths:
///   1. Packed: per-lane SIMD reduction, contiguous (axis == rank-1) or
///      pitch-strided (axis < rank-1), plus a channel merge to one scalar.
///   2. Contiguous: multi-accumulator unrolling per row, delegated to row_reduce.
///   3. Inner-contiguous: SIMD across the inner (non-reduced) axes.
///
/// Only lane-8 vector types (v_f32x8 / v_f16x8), matching simd_lane_for<T>.

#include "nnops/detail/simd/simd.hpp"
#include "nnops/ops/reduce.hpp"
#include "simd_reduce_primitive.hpp"

#include <cfloat>

namespace nnops::kernel {

using namespace simd;

/// Packed SIMD (axis == rank-1): per-lane reduction over D positions `pack` apart.
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
        for (int64_t d = 0; d < D; ++d) {
            v_sum = v_add(v_sum, v_load(x_row + d * pack));
        }
        if (type == ReduceType::Mean) {
            auto v_inv = v_set1(x_row, inv_D);
            v_sum = v_mul(v_sum, v_inv);
        }
        v_store(y_pos, v_sum);
        break;
    }
    case ReduceType::Max: {
        auto v_best = v_set1(x_row, -std::numeric_limits<float>::infinity());
        for (int64_t d = 0; d < D; ++d) {
            v_best = v_max(v_best, v_load(x_row + d * pack));
        }
        v_store(y_pos, v_best);
        break;
    }
    case ReduceType::Min: {
        auto v_best = v_set1(x_row, std::numeric_limits<float>::infinity());
        for (int64_t d = 0; d < D; ++d) {
            v_best = v_min(v_best, v_load(x_row + d * pack));
        }
        v_store(y_pos, v_best);
        break;
    }
    }
}

/// Packed column SIMD (axis < rank-1, pack > 1): per-lane reduction over D rows
/// `x_pitch` apart, output pack-wide.
template <typename T>
inline void reduce_process_packed_col(
    const T* x_col, T* y_pos,
    int64_t x_pitch,
    int64_t D, int64_t pack,
    ReduceType type, float inv_D)
{
    switch (type) {
    case ReduceType::Sum:
    case ReduceType::Mean: {
        auto v_sum = v_zero(x_col);
        for (int64_t d = 0; d < D; ++d) {
            v_sum = v_add(v_sum, v_load(x_col + d * x_pitch));
        }
        if (type == ReduceType::Mean) {
            auto v_inv = v_set1(x_col, inv_D);
            v_sum = v_mul(v_sum, v_inv);
        }
        v_store(y_pos, v_sum);
        break;
    }
    case ReduceType::Max: {
        auto v_best = v_set1(x_col, -std::numeric_limits<float>::infinity());
        for (int64_t d = 0; d < D; ++d) {
            v_best = v_max(v_best, v_load(x_col + d * x_pitch));
        }
        v_store(y_pos, v_best);
        break;
    }
    case ReduceType::Min: {
        auto v_best = v_set1(x_col, std::numeric_limits<float>::infinity());
        for (int64_t d = 0; d < D; ++d) {
            v_best = v_min(v_best, v_load(x_col + d * x_pitch));
        }
        v_store(y_pos, v_best);
        break;
    }
    }
}

/// Packed channel SIMD: reduce D C8 blocks at one spatial position and merge all
/// channels to one scalar. Full blocks accumulate per-lane, then v_reduce; a
/// partial last block (valid_lanes < pack) is added lane by lane. @p inv_total is
/// 1/C and used only by Mean.
template <typename T>
inline void reduce_process_packed_channel(
    const T* x_chan, T* y_pos,
    int64_t x_chan_stride,
    int64_t D, int64_t pack, int64_t valid_lanes,
    ReduceType type, float inv_total)
{
    // Full blocks: all lanes contain real channel data
    int64_t full_blocks = (valid_lanes == pack || D == 0) ? D : D - 1;

    switch (type) {
    case ReduceType::Sum:
    case ReduceType::Mean: {
        auto v_sum = v_zero(x_chan);
        for (int64_t c = 0; c < full_blocks; ++c) {
            v_sum = v_add(v_sum, v_load(x_chan + c * x_chan_stride));
        }

        float sum = v_reduce_sum(v_sum);

        // Partial last block: scalar accumulate valid lanes only
        if (valid_lanes < pack) {
            const T* last_row = x_chan + (D - 1) * x_chan_stride;
            for (int64_t l = 0; l < valid_lanes; ++l) {
                sum += s_load(&last_row[l]);
            }
        }

        float result = (type == ReduceType::Mean) ? sum * inv_total : sum;
        s_store(y_pos, result);
        break;
    }
    case ReduceType::Max: {
        auto v_best = v_set1(x_chan, -std::numeric_limits<float>::infinity());
        for (int64_t c = 0; c < full_blocks; ++c) {
            v_best = v_max(v_best, v_load(x_chan + c * x_chan_stride));
        }

        float best = v_reduce_max(v_best);

        if (valid_lanes < pack) {
            const T* last_row = x_chan + (D - 1) * x_chan_stride;
            for (int64_t l = 0; l < valid_lanes; ++l) {
                float v = s_load(&last_row[l]);
                if (v > best) {
                    best = v;
                }
            }
        }

        s_store(y_pos, best);
        break;
    }
    case ReduceType::Min: {
        auto v_best = v_set1(x_chan, std::numeric_limits<float>::infinity());
        for (int64_t c = 0; c < full_blocks; ++c) {
            v_best = v_min(v_best, v_load(x_chan + c * x_chan_stride));
        }

        float best = v_reduce_min(v_best);

        if (valid_lanes < pack) {
            const T* last_row = x_chan + (D - 1) * x_chan_stride;
            for (int64_t l = 0; l < valid_lanes; ++l) {
                float v = s_load(&last_row[l]);
                if (v < best) {
                    best = v;
                }
            }
        }

        s_store(y_pos, best);
        break;
    }
    }
}

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
