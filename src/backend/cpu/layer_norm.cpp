/// @file layer_norm.cpp
/// @brief SIMD-optimized CPU implementation of layer normalization.
///
/// Supports both f32 and f16 via a single templated implementation.
///
/// Reference: onnxruntime MlasLayerNormF32 + ComputeJob / ComputeJobGenericShared.
///
/// Algorithm:
///   – SIMD fast path (axis == rank-1, contiguous tail):
///       1. 4-wide multi-accumulator SIMD reduce sum/sum_sq + scalar tail.
///       2. Compute mean = sum/N, var = sum_sq/N - mean², inv_std = 1/sqrt(var+eps).
///       3. SIMD normalize: y = (x - mean) * inv_std * scale + bias.
///   – General scalar fallback (arbitrary axis):
///       Welford's single-pass algorithm for mean and variance, then normalize.
///
/// Welford's online algorithm (from onnxruntime LayerNormImpl):
///   mean = 0, M2 = 0
///   for each x_i:
///       delta = x_i - mean
///       mean += delta / (i + 1)
///       delta2 = x_i - mean
///       M2 += delta * delta2
///   var = M2 / N, inv_std = 1 / sqrt(var + epsilon)
///
/// Pitch-aware via row_stride_elems() / stride_elems().

#include "nnops/ops/layer_norm.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <cmath>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

// ============================================================
// Scalar normalization for general axis (non-contiguous tail)
//
// Uses Welford's online algorithm for numerical stability,
// matching onnxruntime's ComputeJob scalar fallback.
// ============================================================

template <typename T>
void layer_norm_general_scalar(
    const T* x_ptr, const T* s_ptr, const T* b_ptr,
    T* y_ptr,
    int64_t num_rows, int64_t norm_size,
    int64_t axis, const TensorView& X, const TensorView& scale,
    float epsilon, bool add_to, bool has_bias,
    const ComputeContext& ctx)
{
    const int64_t rank = X.rank();

    // Pre-compute inner offsets for the normalized tail (dims axis..rank-1)
    std::vector<int64_t> inner_offsets(static_cast<size_t>(norm_size));
    for (int64_t flat = 0; flat < norm_size; ++flat) {
        int64_t off = 0;
        int64_t rem = flat;
        for (int64_t d = rank - 1; d >= axis; --d) {
            int64_t dim = X.shape(d);
            off += (rem % dim) * X.stride_elems(d);
            rem /= dim;
        }
        inner_offsets[static_cast<size_t>(flat)] = off;
    }

    // Outer stride: element distance between consecutive rows
    const int64_t outer_stride = (axis > 0) ? X.stride_elems(axis - 1) : 0;

    const bool scale_is_scalar = (scale.numel() == 1);

    const auto process_row = [&](int64_t row) {
        const int64_t row_base = row * outer_stride;

        // ---- Welford's online algorithm (single pass) ----
        // Matches onnxruntime ComputeJob scalar fallback exactly.
        float mean_val = 0.0f;
        float M2 = 0.0f;
        for (int64_t i = 0; i < norm_size; ++i) {
            int64_t off = row_base + inner_offsets[static_cast<size_t>(i)];
            float x = s_load(&x_ptr[off]);
            float delta = x - mean_val;
            mean_val += delta / static_cast<float>(i + 1);
            float delta2 = x - mean_val;
            M2 += delta * delta2;
        }
        const float var_val = M2 / static_cast<float>(norm_size);
        const float inv_std = 1.0f / std::sqrt(var_val + epsilon);

        // ---- Normalize ----
        // onnxruntime pattern: compute then write in one pass (second pass required
        // because inv_std depends on the full row).
        for (int64_t i = 0; i < norm_size; ++i) {
            int64_t off = row_base + inner_offsets[static_cast<size_t>(i)];
            int64_t s_idx = scale_is_scalar ? 0 : i;
            float s = s_load(&s_ptr[s_idx]);
            float b = (has_bias && b_ptr) ? s_load(&b_ptr[s_idx]) : 0.0f;
            float val = (s_load(&x_ptr[off]) - mean_val) * inv_std * s + b;
            s_store_add(&y_ptr[off], val, add_to);
        }
    };

    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, num_rows, process_row);
    } else {
        for (int64_t i = 0; i < num_rows; ++i) { process_row(i); }
    }
}

}  // anonymous namespace

// ============================================================
// Templated implementation (f32 and f16)
// ============================================================

template <typename T>
void layer_norm_impl(const LayerNormAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx)
{
    const auto& X     = inputs[0];
    const auto& scale = inputs[1];
    const bool has_bias = (inputs.size() >= 3 && !inputs[2].is_empty());

    const int64_t rank = X.rank();
    const float epsilon = attrs.epsilon;

    // Normalize axis
    int64_t axis = attrs.axis;
    if (axis < 0) { axis += rank; }
    NNOPS_ASSERT(axis >= 0 && axis < rank);

    // Number of rows and norm_size per row
    int64_t num_rows = 1;
    for (int64_t i = 0; i < axis; ++i) {
        num_rows *= X.shape(i);
    }
    int64_t norm_size = 1;
    for (int64_t i = axis; i < rank; ++i) {
        norm_size *= X.shape(i);
    }

    const auto* x_ptr  = X.ptr<T>();
    const auto* s_ptr  = scale.ptr<T>();
    const auto* b_ptr  = (has_bias && !inputs[2].is_empty()) ? inputs[2].ptr<T>() : nullptr;
    auto* y_ptr = output.ptr<T>();

    const bool add_to = attrs.add_to;

    // ============================================================
    // Fast path: axis is the innermost contiguous dimension
    // (axis == rank-1, normalized elements are contiguous in memory)
    //
    // This is the 99% case for LLMs (LayerNorm over last dim).
    //
    // Algorithm:
    //   1. SIMD reduce sum/sum_sq over 8-wide blocks
    //   2. Welford accumulation for the scalar tail (numerically stable)
    //   3. Merge SIMD partial stats with Welford tail
    //   4. Compute inv_std, then SIMD normalize + scalar tail
    // ============================================================
    const bool is_contiguous_tail = (axis == rank - 1);

    if (!is_contiguous_tail) {
        layer_norm_general_scalar<T>(
            x_ptr, s_ptr, b_ptr, y_ptr,
            num_rows, norm_size, axis, X, scale,
            epsilon, add_to, has_bias, ctx);
        return;
    }

    // Fast path: contiguous tail
    const int64_t last_dim = norm_size;  // == shape(rank-1)
    const int64_t x_row_stride = X.row_stride_elems();

    // Scale/bias broadcasting
    const bool scale_is_scalar = (scale.numel() == 1);

    constexpr int L = simd_lane_for<T>;

    const auto process_row = [&](int64_t row) {
        const int64_t row_off = row * x_row_stride;
        int64_t i = 0;

        // ================================================================
        // Pass 1 — 4-wide multi-accumulator SIMD reduction + scalar tail.
        //
        // Multi-accumulator unrolling breaks the v_add/v_fmadd dependency
        // chain, same pattern as reduce.cpp Sum/Mean path.
        // ================================================================

        float sum = 0.0f;
        float sum_sq = 0.0f;

        {
            // 4-wide accumulator unrolling
            auto v_sum0 = v_zero(x_ptr);
            auto v_sum_sq0 = v_zero(x_ptr);
            auto v_sum1 = v_zero(x_ptr);
            auto v_sum_sq1 = v_zero(x_ptr);
            auto v_sum2 = v_zero(x_ptr);
            auto v_sum_sq2 = v_zero(x_ptr);
            auto v_sum3 = v_zero(x_ptr);
            auto v_sum_sq3 = v_zero(x_ptr);

            for (; i + 4 * L <= norm_size; i += 4 * L) {
                auto v0 = v_load(x_ptr + row_off + i);
                auto v1 = v_load(x_ptr + row_off + i + L);
                auto v2 = v_load(x_ptr + row_off + i + 2 * L);
                auto v3 = v_load(x_ptr + row_off + i + 3 * L);
                v_sum0 = v_add(v_sum0, v0);
                v_sum_sq0 = v_fmadd(v0, v0, v_sum_sq0);
                v_sum1 = v_add(v_sum1, v1);
                v_sum_sq1 = v_fmadd(v1, v1, v_sum_sq1);
                v_sum2 = v_add(v_sum2, v2);
                v_sum_sq2 = v_fmadd(v2, v2, v_sum_sq2);
                v_sum3 = v_add(v_sum3, v3);
                v_sum_sq3 = v_fmadd(v3, v3, v_sum_sq3);
            }
            // Merge 4 accumulators
            auto v_sum = v_add(v_add(v_sum0, v_sum1), v_add(v_sum2, v_sum3));
            auto v_sum_sq = v_add(v_add(v_sum_sq0, v_sum_sq1),
                                  v_add(v_sum_sq2, v_sum_sq3));

            // 2-wide remainder
            auto v_sum4 = v_zero(x_ptr);
            auto v_sum_sq4 = v_zero(x_ptr);
            auto v_sum5 = v_zero(x_ptr);
            auto v_sum_sq5 = v_zero(x_ptr);
            for (; i + 2 * L <= norm_size; i += 2 * L) {
                auto v4 = v_load(x_ptr + row_off + i);
                auto v5 = v_load(x_ptr + row_off + i + L);
                v_sum4 = v_add(v_sum4, v4);
                v_sum_sq4 = v_fmadd(v4, v4, v_sum_sq4);
                v_sum5 = v_add(v_sum5, v5);
                v_sum_sq5 = v_fmadd(v5, v5, v_sum_sq5);
            }
            v_sum = v_add(v_sum, v_add(v_sum4, v_sum5));
            v_sum_sq = v_add(v_sum_sq, v_add(v_sum_sq4, v_sum_sq5));

            // Single-accumulator remainder
            for (; i + L <= norm_size; i += L) {
                auto v = v_load(x_ptr + row_off + i);
                v_sum = v_add(v_sum, v);
                v_sum_sq = v_fmadd(v, v, v_sum_sq);
            }

            sum = v_reduce_sum(v_sum);
            sum_sq = v_reduce_sum(v_sum_sq);
        }

        // Scalar tail
        for (; i < norm_size; ++i) {
            float x = s_load(&x_ptr[row_off + i]);
            sum += x;
            sum_sq += x * x;
        }

        // ---- Compute statistics ----
        const float inv_n = 1.0f / static_cast<float>(norm_size);
        const float mean_val = sum * inv_n;
        float var_val = sum_sq * inv_n - mean_val * mean_val;
        if (var_val < 0.0f) { var_val = 0.0f; }  // guard against rounding
        const float inv_std = 1.0f / std::sqrt(var_val + epsilon);

        // ================================================================
        // Pass 2 — SIMD normalize
        // ================================================================
        i = 0;

        const auto v_mean    = v_set1(x_ptr, mean_val);
        const auto v_inv_std = v_set1(x_ptr, inv_std);
        const auto v_zero_b  = v_zero(x_ptr);  // bias fallback when no bias

        for (; i + L <= norm_size; i += L) {
            auto x = v_load(x_ptr + row_off + i);
            auto vs = v_load(s_ptr + (scale_is_scalar ? 0 : i));
            auto vb = (has_bias && b_ptr)
                ? v_load(b_ptr + (scale_is_scalar ? 0 : i))
                : v_zero_b;
            auto rv = v_fmadd(v_mul(v_sub(x, v_mean), v_inv_std), vs, vb);
            v_store_add(y_ptr + row_off + i, rv, add_to);
        }
        for (; i < norm_size; ++i) {
            int64_t s_idx = scale_is_scalar ? 0 : i;
            int64_t b_idx = scale_is_scalar ? 0 : i;
            float x = s_load(&x_ptr[row_off + i]);
            float s = s_load(&s_ptr[s_idx]);
            float b = (has_bias && b_ptr) ? s_load(&b_ptr[b_idx]) : 0.0f;
            float rv = (x - mean_val) * inv_std * s + b;
            s_store_add(&y_ptr[row_off + i], rv, add_to);
        }
    };

    // ---- Parallel dispatch ----
    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, num_rows, process_row);
    } else {
        for (int64_t row = 0; row < num_rows; ++row) {
            process_row(row);
        }
    }
}

// ============================================================
// Entry point with dtype dispatch
// ============================================================

void layer_norm_cpu(const LayerNormAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        layer_norm_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        layer_norm_impl<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"layer_norm_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
