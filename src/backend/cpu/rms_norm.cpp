/// @file rms_norm.cpp
/// @brief SIMD-optimized CPU implementation of RMS normalization.
///
/// Supports both f32 and f16 via a single templated implementation.
///
/// Reference: onnxruntime ComputeJob with simplified=true (RMSNorm).
///
/// Algorithm (mirrors onnxruntime):
///   – SIMD fast path (axis == rank-1, contiguous tail):
///       1. SIMD reduce sum_sq in typed vectors.
///       2. Scalar tail accumulation (no Welford needed — RMSNorm has no mean).
///       3. Compute rms = sqrt(sum_sq / N + eps), inv_rms = 1 / rms.
///       4. Normalize with typed SIMD + scalar tail: y = x * inv_rms * scale.
///   – General scalar fallback (arbitrary axis):
///       Simple sum of squares accumulation, then normalize.
///
/// Unlike LayerNorm, RMSNorm does NOT subtract the mean and has no bias,
/// so there is no catastrophic cancellation concern — simple sum_sq
/// accumulation is numerically adequate (matches onnxruntime's approach).
///
/// Pitch-aware via row_stride_elems() / stride_elems().

#include "nnops/ops/rms_norm.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <cmath>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

// ============================================================
// Scalar two-pass normalization for general axis (non-contiguous tail)
// ============================================================

template <typename T>
void rms_norm_general_scalar(
    const T* x_ptr, const T* s_ptr,
    T* y_ptr,
    int64_t num_rows, int64_t norm_size,
    int64_t axis, const TensorView& X, const TensorView& scale,
    float epsilon, bool add_to,
    const ComputeContext& ctx)
{
    const int64_t rank = X.rank();

    // Pre-compute inner offsets for the normalized tail
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

    const int64_t outer_stride = (axis > 0) ? X.stride_elems(axis - 1) : 0;
    const bool scale_is_scalar = (scale.numel() == 1);

    const auto process_row = [&](int64_t row) {
        const int64_t row_base = row * outer_stride;

        // Pass 1: accumulate sum of squares
        float sum_sq = 0.0f;
        for (int64_t i = 0; i < norm_size; ++i) {
            int64_t off = row_base + inner_offsets[static_cast<size_t>(i)];
            float x = s_load(&x_ptr[off]);
            sum_sq += x * x;
        }

        // Compute RMS
        const float rms = std::sqrt(sum_sq / static_cast<float>(norm_size) + epsilon);
        const float inv_rms = 1.0f / rms;

        // Pass 2: normalize
        for (int64_t i = 0; i < norm_size; ++i) {
            int64_t off = row_base + inner_offsets[static_cast<size_t>(i)];
            int64_t s_idx = scale_is_scalar ? 0 : i;
            float s = s_load(&s_ptr[s_idx]);
            float val = s_load(&x_ptr[off]) * inv_rms * s;
            if (add_to) {
                s_store(&y_ptr[off], s_load(&y_ptr[off]) + val);
            } else {
                s_store(&y_ptr[off], val);
            }
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
void rms_norm_impl(const RMSNormAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx)
{
    const auto& X     = inputs[0];
    const auto& scale = inputs[1];

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

    const auto* x_ptr = X.ptr<T>();
    const auto* s_ptr = scale.ptr<T>();
    auto* y_ptr = output.ptr<T>();

    const bool add_to = attrs.add_to;

    // ============================================================
    // Fast path: axis is the innermost contiguous dimension
    // ============================================================
    const bool is_contiguous_tail = (axis == rank - 1);

    if (!is_contiguous_tail) {
        rms_norm_general_scalar<T>(
            x_ptr, s_ptr, y_ptr,
            num_rows, norm_size, axis, X, scale,
            epsilon, add_to, ctx);
        return;
    }

    // Fast path: contiguous tail
    const int64_t last_dim = norm_size;
    const int64_t x_row_stride = X.row_stride_elems();

    const bool scale_is_scalar = (scale.numel() == 1);

    constexpr int L = simd_lane_for<T>;

    const auto process_row = [&](int64_t row) {
        const int64_t row_off = row * x_row_stride;

        // ---- Pass 1: SIMD reduction (typed API, f32-accumulated internally) ----
        float sum_sq = 0.0f;
        int64_t i = 0;

        // Use typed SIMD API: v_f32x8 for float, v_f16x8 for half.
        // Both accumulate in f32 internally (x86 converts f16→f32→compute→f16).
        {
            auto v_sum_sq = v_zero(x_ptr);

            for (; i + L <= norm_size; i += L) {
                auto v = v_load(x_ptr + row_off + i);
                v_sum_sq = v_fmadd(v, v, v_sum_sq);
            }

            sum_sq = v_reduce_sum(v_sum_sq);
        }

        // Scalar tail for reduction
        for (; i < norm_size; ++i) {
            float x = s_load(&x_ptr[row_off + i]);
            sum_sq += x * x;
        }

        // ---- Compute RMS ----
        const float rms = std::sqrt(sum_sq / static_cast<float>(norm_size) + epsilon);
        const float inv_rms = 1.0f / rms;

        // ---- Pass 2: SIMD normalize ----
        i = 0;

        const auto v_inv_rms = v_set1(x_ptr, inv_rms);

        for (; i + L <= norm_size; i += L) {
            auto x = v_load(x_ptr + row_off + i);
            auto vs = v_load(s_ptr + (scale_is_scalar ? 0 : i));
            auto rv = v_mul(v_mul(x, v_inv_rms), vs);
            if (add_to) {
                v_store(y_ptr + row_off + i,
                        v_add(v_load(y_ptr + row_off + i), rv));
            } else {
                v_store(y_ptr + row_off + i, rv);
            }
        }
        for (; i < norm_size; ++i) {
            int64_t s_idx = scale_is_scalar ? 0 : i;
            float x = s_load(&x_ptr[row_off + i]);
            float s = s_load(&s_ptr[s_idx]);
            float rv = x * inv_rms * s;
            if (add_to) {
                s_store(&y_ptr[row_off + i],
                        s_load(&y_ptr[row_off + i]) + rv);
            } else {
                s_store(&y_ptr[row_off + i], rv);
            }
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

void rms_norm_cpu(const RMSNormAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx,
                   void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        rms_norm_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        rms_norm_impl<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"rms_norm_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
