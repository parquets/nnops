/// @file softmax.cpp
/// @brief SIMD-optimized CPU implementation of softmax / log-softmax.
///
/// Supports both f32 and f16 via a single templated implementation.
///
/// Supports planar (NCHW/NCDHW) and packed (NCHWC8/NCDHWC8) layouts.
///
/// Reference: onnxruntime MlasComputeSoftmax + MlasReduceMaximumF32Kernel +
///            MlasComputeSumExpF32Kernel + MlasComputeSoftmaxOutputF32Kernel.
///
/// Three paths:
///   - Packed SIMD (axis == rank-1, pack > 1): per-lane SIMD reduction
///     within each physical row. Delegates to kernel::softmax_process_packed_row.
///   - Standard SIMD fast path (axis == rank-1, pack == 1): contiguous
///     tail. Delegates to kernel::softmax_process_standard_row.
///   - General scalar path: reference-style outer/D/inner decomposition
///     with strided access, correct for all layouts and axes.
///
/// Pitch-aware via stride_elems() / row_stride_elems().

#include "nnops/ops/softmax.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "simd_kernel/simd_softmax.hpp"

#include <cmath>
#include <cfloat>
#include <vector>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

// ============================================================
// General-axis scalar normalization (reference-style decomposition)
// ============================================================

template <typename T>
void softmax_general_scalar(
    const T* x_ptr, T* y_ptr,
    const TensorView& X,
    int64_t axis, int64_t outer_size, int64_t D, int64_t inner_total,
    bool log_softmax,
    const ComputeContext& ctx)
{
    const int64_t rank = X.rank();
    const int64_t pack = X.channel_pack_size();

    const int64_t axis_stride = X.stride_elems(axis) * pack;

    // Precompute offsets for inner dimensions (dims after axis)
    std::vector<int64_t> inner_offsets(static_cast<size_t>(inner_total));
    for (int64_t s = 0; s < inner_total; ++s) {
        int64_t off = 0, rem = s;
        for (int64_t d = rank - 1; d > axis; --d) {
            int64_t dim = X.shape(d);
            off += (rem % dim) * X.stride_elems(d) * pack;
            rem /= dim;
        }
        inner_offsets[static_cast<size_t>(s)] = off;
    }

    const auto process_row = [&](int64_t outer) {
        int64_t base = outer * D * axis_stride;

        for (int64_t s = 0; s < inner_total; ++s) {
            int64_t inner_off = inner_offsets[static_cast<size_t>(s)];

            // ---- Pass 1: find max ----
            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t k = 0; k < D; ++k) {
                float v = s_load(&x_ptr[base + inner_off + k * axis_stride]);
                if (v > max_val) max_val = v;
            }

            // ---- Pass 2: sum of exp(x - max) ----
            float sum_exp = 0.0f;
            for (int64_t k = 0; k < D; ++k) {
                sum_exp += std::exp(s_load(&x_ptr[base + inner_off + k * axis_stride]) - max_val);
            }

            // ---- Pass 3: normalize ----
            if (log_softmax) {
                float log_sum = std::log(sum_exp);
                for (int64_t k = 0; k < D; ++k) {
                    float val = s_load(&x_ptr[base + inner_off + k * axis_stride]) - max_val - log_sum;
                    s_store(&y_ptr[base + inner_off + k * axis_stride], val);
                }
            } else {
                float inv_sum = 1.0f / sum_exp;
                for (int64_t k = 0; k < D; ++k) {
                    float val = std::exp(s_load(&x_ptr[base + inner_off + k * axis_stride]) - max_val) * inv_sum;
                    s_store(&y_ptr[base + inner_off + k * axis_stride], val);
                }
            }
        }
    };

    if (ctx.cpu_parallel_for)
        ctx.cpu_parallel_for(0, outer_size, process_row);
    else
        for (int64_t i = 0; i < outer_size; ++i) process_row(i);
}


}  // anonymous namespace

// ============================================================
// Templated implementation (f32 and f16)
// ============================================================

template <typename T>
void softmax_impl(const SoftmaxAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx)
{
    const auto& X = inputs[0];

    const int64_t rank = X.rank();
    NNOPS_ASSERT(rank >= 1);

    const bool log_softmax = attrs.log_softmax;

    // Normalize axis
    int64_t axis = attrs.axis;
    if (axis < 0) { axis += rank; }
    NNOPS_ASSERT(axis >= 0 && axis < rank);

    const auto* x_ptr = X.ptr<T>();
    auto* y_ptr = output.ptr<T>();

    const int64_t pack = X.channel_pack_size();

    // ============================================================
    // Path 1 — Packed SIMD: axis == rank-1, pack > 1
    // ============================================================

    if (axis == rank - 1 && pack > 1) {
        const int64_t D = X.shape(axis);
        const int64_t num_rows = X.total_rows();
        const int64_t x_rs = X.row_stride_elems();
        const int64_t y_rs = output.row_stride_elems();

        const auto process_row = [&](int64_t r) {
            kernel::softmax_process_packed_row<T>(
                x_ptr + r * x_rs, y_ptr + r * y_rs, D, pack, log_softmax);
        };

        if (ctx.cpu_parallel_for)
            ctx.cpu_parallel_for(0, num_rows, process_row);
        else
            for (int64_t r = 0; r < num_rows; ++r) process_row(r);

        return;
    }

    // ============================================================
    // Path 1b — Packed channel SIMD: axis == 1, pack > 1 (NCHWC8)
    // ============================================================

    if (axis == 1 && pack > 1) {
        const int64_t C = X.shape(1);                    // logical channel count
        const int64_t C8 = X.num_channel_blocks();        // ceil(C / pack)
        const int64_t chan_stride = X.stride_elems(1);    // stride between C8 blocks
        const int64_t valid_lanes = (C % pack == 0) ? pack : (C % pack);

        // Count spatial positions: product of all dims except C8 (axis=1)
        int64_t num_spatial = 1;
        for (int64_t d = 0; d < rank; ++d)
            if (d != 1) num_spatial *= X.shape(d);

        const auto process_pos = [&](int64_t s) {
            // Map flat spatial index → tensor offset (skipping C8 dim)
            int64_t off = 0;
            int64_t rem = s;
            for (int64_t d = rank - 1; d >= 0; --d) {
                if (d == 1) continue;
                int64_t dim = X.shape(d);
                off += (rem % dim) * X.stride_elems(d);
                rem /= dim;
            }
            kernel::softmax_process_packed_channel<T>(
                x_ptr + off, y_ptr + off,
                chan_stride, chan_stride,
                C8, pack, valid_lanes, log_softmax);
        };

        if (ctx.cpu_parallel_for)
            ctx.cpu_parallel_for(0, num_spatial, process_pos);
        else
            for (int64_t i = 0; i < num_spatial; ++i) process_pos(i);

        return;
    }

    // ============================================================
    // Path 1c — Packed column SIMD (pack > 1, general non-channel axis)
    //           Covers NCHWC8 H, NCDHWC8 D/H, and axis=0 for all.
    // ============================================================

    if (pack > 1) {
        const int64_t D = X.shape(axis);
        const int64_t axis_stride = X.stride_elems(axis);

        int64_t num_positions = 1;
        for (int64_t d = 0; d < rank; ++d)
            if (d != axis) num_positions *= X.shape(d);

        const auto process_pos = [&](int64_t s) {
            int64_t off = 0;
            int64_t rem = s;
            for (int64_t d = rank - 1; d >= 0; --d) {
                if (d == axis) continue;
                int64_t dim = X.shape(d);
                off += (rem % dim) * X.stride_elems(d);
                rem /= dim;
            }
            kernel::softmax_process_packed_col<T>(
                x_ptr + off, y_ptr + off,
                axis_stride, axis_stride,
                D, pack, log_softmax);
        };

        if (ctx.cpu_parallel_for)
            ctx.cpu_parallel_for(0, num_positions, process_pos);
        else
            for (int64_t i = 0; i < num_positions; ++i) process_pos(i);

        return;
    }

    // ============================================================
    // Path 2 — Standard SIMD fast path: axis == rank-1, pack == 1
    // ============================================================

    if (axis == rank - 1) {
        int64_t num_rows = 1;
        for (int64_t i = 0; i < axis; ++i)
            num_rows *= X.shape(i);

        const int64_t D = X.shape(axis);
        const int64_t group_stride = (axis > 0) ? X.stride_elems(axis - 1) : D;

        const auto process_row = [&](int64_t row) {
            kernel::softmax_process_standard_row<T>(
                x_ptr + row * group_stride, y_ptr + row * group_stride, D, log_softmax);
        };

        if (ctx.cpu_parallel_for)
            ctx.cpu_parallel_for(0, num_rows, process_row);
        else
            for (int64_t row = 0; row < num_rows; ++row) process_row(row);

        return;
    }

    // ============================================================
    // Path 3 — General axis: scalar, reference-style decomposition
    // ============================================================

    {
        int64_t outer_size = 1;
        for (int64_t i = 0; i < axis; ++i)
            outer_size *= X.shape(i);

        const int64_t D = X.shape(axis);

        int64_t inner_total = 1;
        for (int64_t i = axis + 1; i < rank; ++i)
            inner_total *= X.shape(i);

        softmax_general_scalar<T>(
            x_ptr, y_ptr, X,
            axis, outer_size, D, inner_total,
            log_softmax, ctx);
    }
}

// ============================================================
// Entry point with dtype dispatch
// ============================================================

void softmax_cpu(const SoftmaxAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx,
                  void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        softmax_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        softmax_impl<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"softmax_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
