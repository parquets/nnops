/// @file softmax.cpp
/// @brief SIMD-optimized CPU implementation of softmax / log-softmax.
///
/// Supports both f32 and f16 via a single templated implementation.
///
/// Reference: onnxruntime MlasComputeSoftmax + MlasReduceMaximumF32Kernel +
///            MlasComputeSumExpF32Kernel + MlasComputeSoftmaxOutputF32Kernel.
///
/// Algorithm (per row, mirrors onnxruntime):
///   1. ReduceMax — SIMD max reduction + scalar tail → max_val.
///   2. ComputeSumExp — SIMD exp(x - max) + reduce sum; for softmax (non-log,
///      non-add_to) exp values are stored to the output buffer to avoid
///      recomputing exp in the normalization pass.
///   3. Normalize — softmax: multiply stored exp by 1/sum (or recompute exp
///      for add_to); log_softmax: y = (x - max) - log(sum).
///
/// The fast path handles axis == rank-1 (contiguous last dim, the 99% case
/// for LLM attention softmax). General axis falls back to a scalar path with
/// pre-computed inner offsets — same as the reference.
///
/// Pitch-aware via row_stride_elems() / stride_elems().

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

/// @brief Reduce a typed SIMD vector (v_f32x8 or v_f16x8) to its maximum
/// scalar value.  The SIMD layer has v_reduce_sum but not v_reduce_max, so
/// we store to a temp buffer and scan.
template <typename T>
float reduce_max_vec(const T* type_tag, const auto& vmax) {
    T tmp[simd_lane_for<T>];
    v_store(tmp, vmax);
    float best = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < simd_lane_for<T>; ++k) {
        float x = s_load(&tmp[k]);
        if (x > best) best = x;
    }
    (void)type_tag; // used only for overload resolution
    return best;
}

// ============================================================
// Scalar normalization for general axis (non-contiguous tail)
// ============================================================

template <typename T>
void softmax_general_scalar(
    const T* x_ptr,
    T* y_ptr,
    int64_t num_rows, int64_t norm_size,
    int64_t axis, const TensorView& X,
    bool log_softmax, bool add_to,
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

    const int64_t outer_stride = (axis > 0) ? X.stride_elems(axis - 1) : 0;

    const auto process_row = [&](int64_t row) {
        const int64_t row_base = row * outer_stride;

        // ---- Pass 1: find max ----
        float max_val = -std::numeric_limits<float>::infinity();
        for (int64_t i = 0; i < norm_size; ++i) {
            int64_t off = row_base + inner_offsets[static_cast<size_t>(i)];
            float x = s_load(&x_ptr[off]);
            if (x > max_val) max_val = x;
        }

        // ---- Pass 2: sum of exp(x - max) ----
        float sum_exp = 0.0f;
        for (int64_t i = 0; i < norm_size; ++i) {
            int64_t off = row_base + inner_offsets[static_cast<size_t>(i)];
            sum_exp += std::exp(s_load(&x_ptr[off]) - max_val);
        }

        // ---- Pass 3: normalize ----
        if (log_softmax) {
            // log_softmax = (x - max) - log(sum)
            float log_sum = std::log(sum_exp);
            for (int64_t i = 0; i < norm_size; ++i) {
                int64_t off = row_base + inner_offsets[static_cast<size_t>(i)];
                float val = s_load(&x_ptr[off]) - max_val - log_sum;
                if (add_to) {
                    s_store(&y_ptr[off], s_load(&y_ptr[off]) + val);
                } else {
                    s_store(&y_ptr[off], val);
                }
            }
        } else {
            float inv_sum = 1.0f / sum_exp;
            for (int64_t i = 0; i < norm_size; ++i) {
                int64_t off = row_base + inner_offsets[static_cast<size_t>(i)];
                float val = std::exp(s_load(&x_ptr[off]) - max_val) * inv_sum;
                if (add_to) {
                    s_store(&y_ptr[off], s_load(&y_ptr[off]) + val);
                } else {
                    s_store(&y_ptr[off], val);
                }
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
void softmax_impl(const SoftmaxAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx)
{
    const auto& X = inputs[0];

    const int64_t rank = X.rank();
    NNOPS_ASSERT(rank >= 1);

    const bool log_softmax = attrs.log_softmax;
    const bool add_to = attrs.add_to;

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

    const auto* x_ptr = X.ptr<T>();
    auto* y_ptr = output.ptr<T>();

    // ============================================================
    // Fast path: axis is the innermost contiguous dimension
    // (axis == rank-1, normalized elements are contiguous in memory)
    //
    // This is the 99% case for LLMs (softmax over last dim).
    //
    // Algorithm (mirrors onnxruntime MlasComputeSoftmax):
    //   1. SIMD reduce max + scalar tail → max_val
    //   2. SIMD exp(x - max) + accumulate sum; for non-log non-add_to
    //      store exp values to output buffer to avoid recomputation
    //   3. Normalize: softmax = exp/sum or log_softmax = (x-max)-log(sum)
    // ============================================================
    const bool is_contiguous_tail = (axis == rank - 1);

    if (!is_contiguous_tail) {
        softmax_general_scalar<T>(
            x_ptr, y_ptr,
            num_rows, norm_size, axis, X,
            log_softmax, add_to, ctx);
        return;
    }

    // Fast path: contiguous tail
    const int64_t D = norm_size;          // elements per row
    const int64_t x_row_stride = X.row_stride_elems();

    constexpr int L = simd_lane_for<T>;

    const auto process_row = [&](int64_t row) {
        const int64_t row_off = row * x_row_stride;
        int64_t i = 0;

        // ============================================================
        // Pass 1 — Max reduction
        //
        // Mirror: onnxruntime MlasReduceMaximumF32Kernel
        // ============================================================

        float max_val = -std::numeric_limits<float>::infinity();

        {
            auto v_max_val = v_set1(x_ptr, max_val);

            for (; i + L <= D; i += L) {
                v_max_val = v_max(v_max_val, v_load(x_ptr + row_off + i));
            }

            max_val = reduce_max_vec(x_ptr, v_max_val);
        }

        // Scalar tail for max reduction
        for (; i < D; ++i) {
            float x = s_load(&x_ptr[row_off + i]);
            if (x > max_val) max_val = x;
        }

        const float neg_max = -max_val;

        // ============================================================
        // Pass 2 — Compute sum of exp(x - max)
        //
        // For non-log_softmax without add_to we can store the exp
        // values directly to the output buffer and normalize them
        // in pass 3 with a cheap multiply — avoids recomputing exp.
        //
        // For log_softmax we don't need the intermediate exp values
        // at all (the output pass only uses the original input).
        //
        // Mirror: onnxruntime MlasComputeSumExpF32Kernel
        // ============================================================

        float sum_exp = 0.0f;
        i = 0;

        const bool store_exp_to_output = !log_softmax && !add_to;

        {
            auto v_sum = v_zero(x_ptr);
            auto v_neg_max = v_set1(x_ptr, neg_max);

            for (; i + L <= D; i += L) {
                auto v = v_load(x_ptr + row_off + i);
                v = v_add(v, v_neg_max);      // x - max
                v = v_exp(v);
                if (store_exp_to_output) {
                    v_store(y_ptr + row_off + i, v);
                }
                v_sum = v_add(v_sum, v);
            }

            sum_exp = v_reduce_sum(v_sum);
        }

        // Scalar tail for sum
        for (; i < D; ++i) {
            float val = std::exp(s_load(&x_ptr[row_off + i]) - max_val);
            if (store_exp_to_output) {
                s_store(&y_ptr[row_off + i], val);
            }
            sum_exp += val;
        }

        // ============================================================
        // Pass 3 — Normalize
        //
        // Mirror: onnxruntime MlasComputeSoftmaxOutputF32Kernel
        //         / MlasComputeLogSoftmaxOutputF32Kernel
        // ============================================================

        i = 0;

        if (log_softmax) {
            // log_softmax = (x - max) - log(sum_exp)
            const float log_sum = std::log(sum_exp);
            const float bias = neg_max - log_sum;  // -max - log(sum)

            auto v_bias = v_set1(x_ptr, bias);

            for (; i + L <= D; i += L) {
                auto rv = v_add(v_load(x_ptr + row_off + i), v_bias);
                if (add_to) {
                    v_store(y_ptr + row_off + i,
                            v_add(v_load(y_ptr + row_off + i), rv));
                } else {
                    v_store(y_ptr + row_off + i, rv);
                }
            }
            for (; i < D; ++i) {
                float rv = s_load(&x_ptr[row_off + i]) + bias;
                if (add_to) {
                    s_store(&y_ptr[row_off + i],
                            s_load(&y_ptr[row_off + i]) + rv);
                } else {
                    s_store(&y_ptr[row_off + i], rv);
                }
            }
        } else {
            // softmax = exp(x - max) / sum_exp
            const float inv_sum = 1.0f / sum_exp;

            if (store_exp_to_output) {
                // Exp values already stored in output — just multiply by 1/sum
                auto v_inv = v_set1(x_ptr, inv_sum);

                for (; i + L <= D; i += L) {
                    auto v = v_load(y_ptr + row_off + i);
                    v_store(y_ptr + row_off + i, v_mul(v, v_inv));
                }
                for (; i < D; ++i) {
                    s_store(&y_ptr[row_off + i],
                            s_load(&y_ptr[row_off + i]) * inv_sum);
                }
            } else {
                // add_to path: recompute exp(x - max), multiply by inv_sum,
                // and add to existing output
                auto v_inv = v_set1(x_ptr, inv_sum);
                auto v_neg_max = v_set1(x_ptr, neg_max);

                for (; i + L <= D; i += L) {
                    auto v = v_load(x_ptr + row_off + i);
                    v = v_add(v, v_neg_max);      // x - max
                    v = v_exp(v);
                    v = v_mul(v, v_inv);          // exp / sum
                    v_store(y_ptr + row_off + i,
                            v_add(v_load(y_ptr + row_off + i), v));
                }
                for (; i < D; ++i) {
                    float val = std::exp(s_load(&x_ptr[row_off + i]) - max_val) * inv_sum;
                    s_store(&y_ptr[row_off + i],
                            s_load(&y_ptr[row_off + i]) + val);
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
