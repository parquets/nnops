/// @file layer_norm.cpp
/// @brief SIMD-optimized CPU implementation of layer normalization.
///
/// Supports both f32 and f16 via a single templated implementation.
///
/// Reference: onnxruntime MlasLayerNormF32 + ComputeJob / ComputeJobGenericShared.
///
/// Algorithm (mirrors onnxruntime):
///   – SIMD fast path (axis == rank-1, contiguous tail):
///       1. SIMD reduce sum/sum_sq in typed vectors (v_f32x8 or v_f16x8).
///       2. Scalar tail uses Welford's online algorithm for numerical stability.
///       3. Merge SIMD partial stats with Welford tail via parallel Welford formula.
///       4. Compute inv_std and normalize with typed SIMD + scalar tail.
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
// Welford merge: combine two partial statistics.
//
// Given (n1, mean1, M2_1) and (n2, mean2, M2_2), compute the
// merged (n, mean, M2).  This is Chan et al.'s parallel formula.
// ============================================================
struct WelfordStats {
    int64_t n = 0;
    float mean = 0.0f;
    float M2 = 0.0f;  // sum of squared differences from mean
};

inline void welford_merge(WelfordStats& a, const WelfordStats& b) {
    if (b.n == 0) return;
    if (a.n == 0) { a = b; return; }
    const int64_t total = a.n + b.n;
    const float delta = b.mean - a.mean;
    const float na = static_cast<float>(a.n);
    const float nb = static_cast<float>(b.n);
    a.mean = (na * a.mean + nb * b.mean) / static_cast<float>(total);
    a.M2 = a.M2 + b.M2 + delta * delta * na * nb / static_cast<float>(total);
    a.n = total;
}

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
        for (int64_t i = 0; i < num_rows; ++i) process_row(i);
    }
}

}  // anonymous namespace

// ============================================================
// Templated implementation (f32 and f16)
// ============================================================

template <typename T>
void layer_norm_impl(const LayerNormAttributes& attrs,
                      const TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx)
{
    const auto& X     = inputs[0];
    const auto& scale = inputs[1];
    const bool has_bias = (inputs.size() >= 3 && inputs[2].data() != nullptr);

    const int64_t rank = X.rank();
    const float epsilon = attrs.epsilon;

    // Normalize axis
    int64_t axis = attrs.axis;
    if (axis < 0) axis += rank;
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

    const auto* x_ptr  = X.data_as<T>();
    const auto* s_ptr  = scale.data_as<T>();
    const auto* b_ptr  = (has_bias && inputs[2].data()) ? inputs[2].data_as<T>() : nullptr;
    auto* y_ptr = output.data_as<T>();

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
        // Pass 1 — Hybrid statistics: SIMD reduce + Welford scalar tail
        //
        // SIMD portion: accumulate sum and sum_sq in typed vectors.
        // Welford tail:  handle remaining elements with online algorithm.
        // Then merge both partial results via parallel Welford formula.
        //
        // This mirrors onnxruntime: MlasLayerNormF32 (SIMD) for the bulk,
        // Welford fallback for correctness on the tail.
        // ================================================================

        WelfordStats stats;

        // -- SIMD block: reduce sum and sum_sq --
        {
            float simd_sum = 0.0f;
            float simd_sum_sq = 0.0f;
            int64_t simd_count = 0;

            {
                auto v_sum = v_zero(x_ptr);
                auto v_sum_sq = v_zero(x_ptr);

                for (; i + L <= norm_size; i += L) {
                    auto v = v_load(x_ptr + row_off + i);
                    v_sum = v_add(v_sum, v);
                    v_sum_sq = v_fmadd(v, v, v_sum_sq);
                    simd_count += L;
                }

                simd_sum = v_reduce_sum(v_sum);
                simd_sum_sq = v_reduce_sum(v_sum_sq);
            }

            // Convert SIMD (count, sum, sum_sq) to WelfordStats
            if (simd_count > 0) {
                stats.n = simd_count;
                stats.mean = simd_sum / static_cast<float>(simd_count);
                stats.M2 = simd_sum_sq - simd_sum * simd_sum / static_cast<float>(simd_count);
                // Guard against tiny negative M2 from rounding
                if (stats.M2 < 0.0f) stats.M2 = 0.0f;
            }
        }

        // -- Scalar tail: Welford's online algorithm --
        {
            WelfordStats tail;
            tail.n = norm_size - i;  // remaining elements
            if (tail.n > 0) {
                float mean_val = 0.0f;
                float M2_val = 0.0f;
                int64_t j = 0;
                for (; i < norm_size; ++i, ++j) {
                    float x = s_load(&x_ptr[row_off + i]);
                    float delta = x - mean_val;
                    mean_val += delta / static_cast<float>(j + 1);
                    float delta2 = x - mean_val;
                    M2_val += delta * delta2;
                }
                tail.mean = mean_val;
                tail.M2 = M2_val;
            }
            welford_merge(stats, tail);
        }

        // ---- Compute statistics from merged Welford result ----
        const float var_val = stats.M2 / static_cast<float>(stats.n);
        const float inv_std = 1.0f / std::sqrt(var_val + epsilon);
        const float mean_val = stats.mean;

        // ================================================================
        // Pass 2 — SIMD normalize
        // ================================================================
        i = 0;

        const auto v_mean    = v_set1(x_ptr, mean_val);
        const auto v_inv_std = v_set1(x_ptr, inv_std);

        if (has_bias && b_ptr) {
            if (add_to) {
                for (; i + L <= norm_size; i += L) {
                    auto x = v_load(x_ptr + row_off + i);
                    auto vs = v_load(s_ptr + (scale_is_scalar ? 0 : i));
                    auto vb = v_load(b_ptr + (scale_is_scalar ? 0 : i));
                    // y = (x - mean) * inv_std * scale + bias
                    auto result = v_fmadd(v_mul(v_sub(x, v_mean), v_inv_std), vs, vb);
                    v_store(y_ptr + row_off + i, v_add(v_load(y_ptr + row_off + i), result));
                }
                for (; i < norm_size; ++i) {
                    int64_t s_idx = scale_is_scalar ? 0 : i;
                    int64_t b_idx = scale_is_scalar ? 0 : i;
                    float x = s_load(&x_ptr[row_off + i]);
                    float s = s_load(&s_ptr[s_idx]);
                    float b = s_load(&b_ptr[b_idx]);
                    float val = (x - mean_val) * inv_std * s + b;
                    s_store(&y_ptr[row_off + i], s_load(&y_ptr[row_off + i]) + val);
                }
            } else {
                for (; i + L <= norm_size; i += L) {
                    auto x = v_load(x_ptr + row_off + i);
                    auto vs = v_load(s_ptr + (scale_is_scalar ? 0 : i));
                    auto vb = v_load(b_ptr + (scale_is_scalar ? 0 : i));
                    v_store(y_ptr + row_off + i,
                            v_fmadd(v_mul(v_sub(x, v_mean), v_inv_std), vs, vb));
                }
                for (; i < norm_size; ++i) {
                    int64_t s_idx = scale_is_scalar ? 0 : i;
                    int64_t b_idx = scale_is_scalar ? 0 : i;
                    float x = s_load(&x_ptr[row_off + i]);
                    float s = s_load(&s_ptr[s_idx]);
                    float b = s_load(&b_ptr[b_idx]);
                    s_store(&y_ptr[row_off + i], (x - mean_val) * inv_std * s + b);
                }
            }
        } else {
            // No bias
            if (add_to) {
                for (; i + L <= norm_size; i += L) {
                    auto x = v_load(x_ptr + row_off + i);
                    auto vs = v_load(s_ptr + (scale_is_scalar ? 0 : i));
                    auto result = v_mul(v_mul(v_sub(x, v_mean), v_inv_std), vs);
                    v_store(y_ptr + row_off + i, v_add(v_load(y_ptr + row_off + i), result));
                }
                for (; i < norm_size; ++i) {
                    int64_t s_idx = scale_is_scalar ? 0 : i;
                    float x = s_load(&x_ptr[row_off + i]);
                    float s = s_load(&s_ptr[s_idx]);
                    float val = (x - mean_val) * inv_std * s;
                    s_store(&y_ptr[row_off + i], s_load(&y_ptr[row_off + i]) + val);
                }
            } else {
                for (; i + L <= norm_size; i += L) {
                    auto x = v_load(x_ptr + row_off + i);
                    auto vs = v_load(s_ptr + (scale_is_scalar ? 0 : i));
                    v_store(y_ptr + row_off + i,
                            v_mul(v_mul(v_sub(x, v_mean), v_inv_std), vs));
                }
                for (; i < norm_size; ++i) {
                    int64_t s_idx = scale_is_scalar ? 0 : i;
                    float x = s_load(&x_ptr[row_off + i]);
                    float s = s_load(&s_ptr[s_idx]);
                    s_store(&y_ptr[row_off + i], (x - mean_val) * inv_std * s);
                }
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

void layer_norm_cpu(const LayerNormAttributes& attrs,
                     const TensorView& output,
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
