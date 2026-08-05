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

/// Packed SIMD: per-lane reduction over D spatial positions,
/// contiguous pack-wide strides (axis == rank-1).
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

/// Packed column SIMD: per-lane reduction over D pitch-strided rows
/// (axis < rank-1, pack > 1).  Each row holds `pack` SIMD lanes at a
/// fixed (outer, inner) position; x_pitch strides to the next row
/// along the reduce axis.
/// Output is pack-wide (v_store) — each lane independently reduced.
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
        for (int64_t d = 0; d < D; ++d)
            v_sum = v_add(v_sum, v_load(x_col + d * x_pitch));
        if (type == ReduceType::Mean) {
            auto v_inv = v_set1(x_col, inv_D);
            v_sum = v_mul(v_sum, v_inv);
        }
        v_store(y_pos, v_sum);
        break;
    }
    case ReduceType::Max: {
        auto v_best = v_set1(x_col, -std::numeric_limits<float>::infinity());
        for (int64_t d = 0; d < D; ++d)
            v_best = v_max(v_best, v_load(x_col + d * x_pitch));
        v_store(y_pos, v_best);
        break;
    }
    case ReduceType::Min: {
        auto v_best = v_set1(x_col, std::numeric_limits<float>::infinity());
        for (int64_t d = 0; d < D; ++d)
            v_best = v_min(v_best, v_load(x_col + d * x_pitch));
        v_store(y_pos, v_best);
        break;
    }
    }
}

/// Packed channel SIMD: reduction over D C8 blocks at a single spatial
/// position, with a horizontal reduction across SIMD lanes to produce
/// a single scalar.
///
/// Unlike per-lane packed_row (each lane = different channel, independent
/// reduction), channel reduction merges ALL channels:
///   1. Per-lane SIMD accumulation over full C8 blocks (0..D-2, or 0..D-1
///      when valid_lanes == pack).
///   2. Scalar accumulation for the partial last block (valid_lanes < pack).
///   3. Horizontal reduction: v_reduce on the SIMD accumulator, then
///      combine with partial scalar contributions via std::max/std::min/+.
///   4. s_store a single scalar to y_pos.
///
/// valid_lanes: number of valid channels in the last C8 block
///   (= C % pack, or pack when C % pack == 0).
/// inv_total: 1.0f / logical_C (used only by Mean).
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
        for (int64_t c = 0; c < full_blocks; ++c)
            v_sum = v_add(v_sum, v_load(x_chan + c * x_chan_stride));

        float sum = v_reduce_sum(v_sum);

        // Partial last block: scalar accumulate valid lanes only
        if (valid_lanes < pack) {
            const T* last_row = x_chan + (D - 1) * x_chan_stride;
            for (int64_t l = 0; l < valid_lanes; ++l)
                sum += s_load(&last_row[l]);
        }

        float result = (type == ReduceType::Mean) ? sum * inv_total : sum;
        s_store(y_pos, result);
        break;
    }
    case ReduceType::Max: {
        auto v_best = v_set1(x_chan, -std::numeric_limits<float>::infinity());
        for (int64_t c = 0; c < full_blocks; ++c)
            v_best = v_max(v_best, v_load(x_chan + c * x_chan_stride));

        float best = v_reduce_max(v_best);

        if (valid_lanes < pack) {
            const T* last_row = x_chan + (D - 1) * x_chan_stride;
            for (int64_t l = 0; l < valid_lanes; ++l) {
                float v = s_load(&last_row[l]);
                if (v > best) best = v;
            }
        }

        s_store(y_pos, best);
        break;
    }
    case ReduceType::Min: {
        auto v_best = v_set1(x_chan, std::numeric_limits<float>::infinity());
        for (int64_t c = 0; c < full_blocks; ++c)
            v_best = v_min(v_best, v_load(x_chan + c * x_chan_stride));

        float best = v_reduce_min(v_best);

        if (valid_lanes < pack) {
            const T* last_row = x_chan + (D - 1) * x_chan_stride;
            for (int64_t l = 0; l < valid_lanes; ++l) {
                float v = s_load(&last_row[l]);
                if (v < best) best = v;
            }
        }

        s_store(y_pos, best);
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
