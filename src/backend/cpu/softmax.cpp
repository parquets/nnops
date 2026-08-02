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
/// Algorithm (per normalization group):
///   1. ReduceMax — find max value for numerical stability.
///   2. ComputeSumExp — exp(x - max) + reduce sum; for softmax (non-log)
///      store exp values to the output buffer to avoid recomputation.
///   3. Normalize — softmax: multiply stored exp by 1/sum;
///      log_softmax: y = (x - max) - log(sum).
///
/// Three paths:
///   - Packed SIMD (axis == rank-1, pack > 1): per-lane SIMD reduction
///     within each physical row. For NCHWC8 [N,C8,H,W], processes all
///     8 C lanes simultaneously over the W spatial dimension.
///   - Standard SIMD fast path (axis == rank-1, pack == 1): contiguous
///     tail, scalar max/sum reduction via SIMD.
///   - General scalar path: reference-style outer/D/inner decomposition
///     with strided access, correct for all layouts and axes.
///
/// Pitch-aware via stride_elems() / row_stride_elems().

#include "nnops/ops/softmax.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <cmath>
#include <cfloat>
#include <vector>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

// ============================================================
// General-axis scalar normalization (reference-style decomposition)
//
// Separates the tensor into outer (dims before axis), D (axis dim),
// and inner (dims after axis). For each (outer, inner) pair,
// normalizes D elements spaced by axis_stride.
//
// Unlike the flat inner_offsets approach, this correctly handles
// packed layouts where dimensions after the axis may be interleaved.
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

    // stride_elems() returns logical strides. For packed layouts,
    // physical stride = logical stride * pack (the C8 interleave
    // multiplies the stride for every dimension).
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

    constexpr int L = simd_lane_for<T>;
    const int64_t pack = X.channel_pack_size();

    // ============================================================
    // Path 1 — Packed SIMD: axis == rank-1, pack > 1
    //
    // For NCHWC8 [N,C8,H,W] with axis=W (the innermost dim), the
    // row layout is: [w0_l0..w0_l7, w1_l0..w1_l7, ...].
    // Each of the `pack` C lanes is an independent softmax over
    // D=W spatial positions. We process all lanes simultaneously
    // with per-lane SIMD vector max/sum reductions.
    // ============================================================

    if (axis == rank - 1 && pack > 1) {
        const int64_t D = X.shape(axis);       // spatial dim (e.g. W)
        const int64_t num_rows = X.total_rows();
        const int64_t x_rs = X.row_stride_elems();
        const int64_t y_rs = output.row_stride_elems();

        const auto process_row = [&](int64_t r) {
            const T* x_row = x_ptr + r * x_rs;
            T* y_row = y_ptr + r * y_rs;

            // ---- Pass 1: per-lane max ----
            auto v_max_vec = v_set1(x_ptr, -std::numeric_limits<float>::infinity());
            for (int64_t w = 0; w < D; ++w)
                v_max_vec = v_max(v_max_vec, v_load(x_row + w * pack));

            // ---- Pass 2: per-lane sum of exp(x - max) ----
            auto v_sum_vec = v_zero(x_ptr);
            auto v_neg_max = v_neg(v_max_vec);
            for (int64_t w = 0; w < D; ++w) {
                auto v = v_add(v_load(x_row + w * pack), v_neg_max);
                v = v_exp(v);
                if (!log_softmax) { v_store(y_row + w * pack, v); }
                v_sum_vec = v_add(v_sum_vec, v);
            }

            // ---- Pass 3: per-lane normalize ----
            if (log_softmax) {
                auto v_bias = v_sub(v_neg(v_max_vec), v_log(v_sum_vec));
                for (int64_t w = 0; w < D; ++w)
                    v_store(y_row + w * pack, v_add(v_load(x_row + w * pack), v_bias));
            } else {
                auto v_inv = v_div(v_set1(x_ptr, 1.0f), v_sum_vec);
                for (int64_t w = 0; w < D; ++w)
                    v_store(y_row + w * pack, v_mul(v_load(y_row + w * pack), v_inv));
            }
        };

        if (ctx.cpu_parallel_for)
            ctx.cpu_parallel_for(0, num_rows, process_row);
        else
            for (int64_t r = 0; r < num_rows; ++r) process_row(r);

        return;
    }

    // ============================================================
    // Path 2 — Standard SIMD fast path: axis == rank-1, pack == 1
    //
    // Planar layouts (NCHW, NCDHW): normalization groups are
    // contiguous in memory. Each group has D elements.
    // stride_elems(axis-1) gives the stride between consecutive
    // groups (equals row_stride_elems for planar).
    // ============================================================

    if (axis == rank - 1) {
        int64_t num_rows = 1;
        for (int64_t i = 0; i < axis; ++i)
            num_rows *= X.shape(i);

        const int64_t D = X.shape(axis);
        const int64_t group_stride = (axis > 0) ? X.stride_elems(axis - 1) : D;

        const auto process_row = [&](int64_t row) {
            const int64_t off = row * group_stride;
            int64_t i = 0;

            // ---- Pass 1: Max reduction ----
            float max_val = -std::numeric_limits<float>::infinity();
            {
                auto v_max_val = v_set1(x_ptr, max_val);
                for (; i + L <= D; i += L)
                    v_max_val = v_max(v_max_val, v_load(x_ptr + off + i));
                max_val = v_reduce_max(v_max_val);
            }
            for (; i < D; ++i) {
                float x = s_load(&x_ptr[off + i]);
                if (x > max_val) { max_val = x; }
            }

            const float neg_max = -max_val;

            // ---- Pass 2: Sum of exp(x - max) ----
            float sum_exp = 0.0f;
            i = 0;
            const bool store_exp = !log_softmax;

            {
                auto v_sum = v_zero(x_ptr);
                auto v_neg_max = v_set1(x_ptr, neg_max);
                for (; i + L <= D; i += L) {
                    auto v = v_add(v_load(x_ptr + off + i), v_neg_max);
                    v = v_exp(v);
                    if (store_exp) { v_store(y_ptr + off + i, v); }
                    v_sum = v_add(v_sum, v);
                }
                sum_exp = v_reduce_sum(v_sum);
            }
            for (; i < D; ++i) {
                float val = std::exp(s_load(&x_ptr[off + i]) - max_val);
                if (store_exp) { s_store(&y_ptr[off + i], val); }
                sum_exp += val;
            }

            // ---- Pass 3: Normalize ----
            i = 0;
            if (log_softmax) {
                const float bias = neg_max - std::log(sum_exp);
                auto v_bias = v_set1(x_ptr, bias);
                for (; i + L <= D; i += L)
                    v_store(y_ptr + off + i, v_add(v_load(x_ptr + off + i), v_bias));
                for (; i < D; ++i)
                    s_store(&y_ptr[off + i], s_load(&x_ptr[off + i]) + bias);
            } else {
                auto v_inv = v_set1(x_ptr, 1.0f / sum_exp);
                for (; i + L <= D; i += L)
                    v_store(y_ptr + off + i, v_mul(v_load(y_ptr + off + i), v_inv));
                for (; i < D; ++i)
                    s_store(&y_ptr[off + i], s_load(&y_ptr[off + i]) / sum_exp);
            }
        };

        if (ctx.cpu_parallel_for)
            ctx.cpu_parallel_for(0, num_rows, process_row);
        else
            for (int64_t row = 0; row < num_rows; ++row) process_row(row);

        return;
    }

    // ============================================================
    // Path 3 — General axis: scalar, reference-style decomposition
    //
    // Splits the tensor into outer (dims before axis), D (axis dim),
    // and inner (dims after axis). For each (outer, inner) pair,
    // normalizes D elements spaced by axis_stride.
    // Correct for all layouts including packed.
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
