/// @file reduce.cpp
/// @brief SIMD-optimized CPU implementation of the Reduce operator.
///
/// Supports both f32 and f16 via a single templated implementation.
///
/// Three dispatch paths:
///   1. Contiguous tail (axis == rank-1): SIMD reduction per row with
///      multi-accumulator unrolling (4-wide for Sum/Mean, 2-wide for
///      Max/Min). Each row produces one scalar output after horizontal
///      reduction (v_reduce_sum / v_reduce_max / v_reduce_min).
///   2. Inner-contiguous general axis (axis < rank-1, inner dims
///      contiguous): SIMD reduction across the axis dimension for groups
///      of L inner elements simultaneously. Each SIMD lane holds one
///      inner position's accumulated result — no horizontal reduction
///      needed, making this path more efficient per element than path 1.
///      Multi-accumulator unrolling in the reduce direction (4-wide for
///      Sum/Mean, 2-wide for Max/Min).
///   3. General scalar (non-contiguous inner dims): pre-computed offset
///      vectors for both input and output, matching the softmax/layer_norm
///      pattern.
///
/// Pitch-aware via stride_elems() / row_stride_elems().

#include "nnops/ops/reduce.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <algorithm>
#include <cfloat>
#include <vector>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

// ============================================================
// Path 1: Contiguous tail (axis == rank-1)
//
// Fast path for the common LLM case: reduce over the last dimension.
// Multi-accumulator unrolling breaks the v_add/v_max/v_min dependency
// chain, allowing the CPU to pipeline multiple independent operations.
// ============================================================

template <typename T>
void reduce_contiguous_simd(const ReduceAttributes& attrs,
                            TensorView& output,
                            const TensorView& input,
                            int64_t axis,
                            const ComputeContext& ctx)
{
    const int64_t rank = input.rank();
    NNOPS_ASSERT(axis == rank - 1);

    const int64_t num_rows = [&]() {
        int64_t n = 1;
        for (int64_t d = 0; d < axis; ++d) { n *= input.shape(d); }
        return n;
    }();
    const int64_t norm_size = input.shape(axis);
    const int64_t row_stride = input.row_stride_elems();

    const T* x_ptr = input.ptr<T>();
    T* y_ptr = output.ptr<T>();
    constexpr int L = simd_lane_for<T>;

    auto process_row = [&](int64_t row) {
        const T* x_row = x_ptr + row * row_stride;
        int64_t i = 0;

        switch (attrs.type) {
        case ReduceType::Sum:
        case ReduceType::Mean: {
            // 4-wide accumulator unrolling
            auto v_sum0 = v_zero(x_ptr);
            auto v_sum1 = v_zero(x_ptr);
            auto v_sum2 = v_zero(x_ptr);
            auto v_sum3 = v_zero(x_ptr);

            for (; i + 4 * L <= norm_size; i += 4 * L) {
                v_sum0 = v_add(v_sum0, v_load(x_row + i));
                v_sum1 = v_add(v_sum1, v_load(x_row + i + L));
                v_sum2 = v_add(v_sum2, v_load(x_row + i + 2 * L));
                v_sum3 = v_add(v_sum3, v_load(x_row + i + 3 * L));
            }
            auto v_sum = v_add(v_add(v_sum0, v_sum1), v_add(v_sum2, v_sum3));

            // 2-wide remainder
            auto v_sum4 = v_zero(x_ptr);
            auto v_sum5 = v_zero(x_ptr);
            for (; i + 2 * L <= norm_size; i += 2 * L) {
                v_sum4 = v_add(v_sum4, v_load(x_row + i));
                v_sum5 = v_add(v_sum5, v_load(x_row + i + L));
            }
            v_sum = v_add(v_sum, v_add(v_sum4, v_sum5));

            // Single-accumulator remainder
            for (; i + L <= norm_size; i += L) {
                v_sum = v_add(v_sum, v_load(x_row + i));
            }
            float sum = v_reduce_sum(v_sum);

            // Scalar tail
            for (; i < norm_size; ++i) {
                sum += s_load(&x_row[i]);
            }
            if (attrs.type == ReduceType::Mean) {
                sum /= static_cast<float>(norm_size);
            }
            s_store(&y_ptr[row], sum);
            break;
        }

        case ReduceType::Max: {
            // 2-wide accumulator unrolling
            auto v_best0 = v_set1(x_ptr, -std::numeric_limits<float>::infinity());
            auto v_best1 = v_set1(x_ptr, -std::numeric_limits<float>::infinity());

            for (; i + 2 * L <= norm_size; i += 2 * L) {
                v_best0 = v_max(v_best0, v_load(x_row + i));
                v_best1 = v_max(v_best1, v_load(x_row + i + L));
            }
            auto v_best = v_max(v_best0, v_best1);

            // Single-accumulator remainder
            for (; i + L <= norm_size; i += L) {
                v_best = v_max(v_best, v_load(x_row + i));
            }
            float best = v_reduce_max(v_best);

            // Scalar tail
            for (; i < norm_size; ++i) {
                float x = s_load(&x_row[i]);
                if (x > best) { best = x; }
            }
            s_store(&y_ptr[row], best);
            break;
        }

        case ReduceType::Min: {
            // 2-wide accumulator unrolling
            auto v_best0 = v_set1(x_ptr, std::numeric_limits<float>::infinity());
            auto v_best1 = v_set1(x_ptr, std::numeric_limits<float>::infinity());

            for (; i + 2 * L <= norm_size; i += 2 * L) {
                v_best0 = v_min(v_best0, v_load(x_row + i));
                v_best1 = v_min(v_best1, v_load(x_row + i + L));
            }
            auto v_best = v_min(v_best0, v_best1);

            // Single-accumulator remainder
            for (; i + L <= norm_size; i += L) {
                v_best = v_min(v_best, v_load(x_row + i));
            }
            float best = v_reduce_min(v_best);

            // Scalar tail
            for (; i < norm_size; ++i) {
                float x = s_load(&x_row[i]);
                if (x < best) { best = x; }
            }
            s_store(&y_ptr[row], best);
            break;
        }
        }
    };

    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, num_rows, process_row);
    } else {
        for (int64_t row = 0; row < num_rows; ++row) { process_row(row); }
    }
}

// ============================================================
// Path 2: Inner-contiguous general axis (axis < rank-1, inner
// dims contiguous in both input and output).
//
// For each outer position, process L inner elements simultaneously.
// The reduction loop accumulates across reduce_size steps using SIMD,
// producing L results per inner block — no horizontal reduction needed.
// This makes the path more efficient per element than path 1 for
// multi-dim inner blocks (L results per SIMD operation vs 1).
//
// Multi-accumulator unrolling in the reduce direction (the hot loop)
// breaks the dependency chain, matching the GEMM micro-kernel pattern.
// ============================================================

template <typename T>
void reduce_inner_contiguous_simd(const ReduceAttributes& attrs,
                                   TensorView& output,
                                   const TensorView& input,
                                   int64_t axis,
                                   int64_t num_outer,
                                   int64_t reduce_size,
                                   int64_t num_inner,
                                   const ComputeContext& ctx)
{
    const int64_t axis_stride = input.stride_elems(axis);
    const int64_t x_outer_stride = (axis > 0) ? input.stride_elems(axis - 1) : 0;

    // For axis > 0: output stride between consecutive outer positions.
    // For axis == 0: output is purely inner (no outer dim), so the
    // stride is num_inner (outer * stride = 0 for the single position).
    const int64_t y_outer_stride = (axis > 0)
        ? output.stride_elems(axis - 1) : num_inner;

    const T* x_ptr = input.ptr<T>();
    T* y_ptr = output.ptr<T>();
    constexpr int L = simd_lane_for<T>;
    const float inv_reduce = 1.0f / static_cast<float>(reduce_size);

    auto process_outer = [&](int64_t outer) {
        const T* x_base = x_ptr + outer * x_outer_stride;
        T* y_base = y_ptr + outer * y_outer_stride;

        int64_t inner = 0;

        switch (attrs.type) {
        case ReduceType::Sum:
        case ReduceType::Mean: {
            // ---- SIMD inner blocks ----
            for (; inner + L <= num_inner; inner += L) {
                // 4-wide accumulator unrolling in reduce direction
                auto v_sum0 = v_zero(x_ptr);
                auto v_sum1 = v_zero(x_ptr);
                auto v_sum2 = v_zero(x_ptr);
                auto v_sum3 = v_zero(x_ptr);

                int64_t k = 0;
                for (; k + 4 <= reduce_size; k += 4) {
                    v_sum0 = v_add(v_sum0, v_load(x_base + (k + 0) * axis_stride + inner));
                    v_sum1 = v_add(v_sum1, v_load(x_base + (k + 1) * axis_stride + inner));
                    v_sum2 = v_add(v_sum2, v_load(x_base + (k + 2) * axis_stride + inner));
                    v_sum3 = v_add(v_sum3, v_load(x_base + (k + 3) * axis_stride + inner));
                }
                // Merge 4 accumulators
                auto v_sum = v_add(v_add(v_sum0, v_sum1), v_add(v_sum2, v_sum3));

                // 2-wide remainder
                auto v_sum4 = v_zero(x_ptr);
                auto v_sum5 = v_zero(x_ptr);
                for (; k + 2 <= reduce_size; k += 2) {
                    v_sum4 = v_add(v_sum4, v_load(x_base + (k + 0) * axis_stride + inner));
                    v_sum5 = v_add(v_sum5, v_load(x_base + (k + 1) * axis_stride + inner));
                }
                v_sum = v_add(v_sum, v_add(v_sum4, v_sum5));

                // Single-step remainder
                for (; k < reduce_size; ++k) {
                    v_sum = v_add(v_sum, v_load(x_base + k * axis_stride + inner));
                }

                if (attrs.type == ReduceType::Mean) {
                    auto v_inv = v_set1(x_ptr, inv_reduce);
                    v_store(y_base + inner, v_mul(v_sum, v_inv));
                } else {
                    v_store(y_base + inner, v_sum);
                }
            }

            // ---- Scalar tail for inner elements ----
            for (; inner < num_inner; ++inner) {
                float sum = 0.0f;
                for (int64_t k = 0; k < reduce_size; ++k) {
                    sum += s_load(&x_base[k * axis_stride + inner]);
                }
                float result = (attrs.type == ReduceType::Mean)
                    ? sum * inv_reduce : sum;
                s_store(&y_base[inner], result);
            }
            break;
        }

        case ReduceType::Max: {
            // ---- SIMD inner blocks ----
            for (; inner + L <= num_inner; inner += L) {
                // 2-wide accumulator unrolling in reduce direction
                auto v_best0 = v_set1(x_ptr, -std::numeric_limits<float>::infinity());
                auto v_best1 = v_set1(x_ptr, -std::numeric_limits<float>::infinity());

                int64_t k = 0;
                for (; k + 2 <= reduce_size; k += 2) {
                    v_best0 = v_max(v_best0, v_load(x_base + (k + 0) * axis_stride + inner));
                    v_best1 = v_max(v_best1, v_load(x_base + (k + 1) * axis_stride + inner));
                }
                auto v_best = v_max(v_best0, v_best1);

                // Single-step remainder
                for (; k < reduce_size; ++k) {
                    v_best = v_max(v_best, v_load(x_base + k * axis_stride + inner));
                }

                v_store(y_base + inner, v_best);
            }

            // ---- Scalar tail ----
            for (; inner < num_inner; ++inner) {
                float best = -std::numeric_limits<float>::infinity();
                for (int64_t k = 0; k < reduce_size; ++k) {
                    float x = s_load(&x_base[k * axis_stride + inner]);
                    if (x > best) { best = x; }
                }
                s_store(&y_base[inner], best);
            }
            break;
        }

        case ReduceType::Min: {
            // ---- SIMD inner blocks ----
            for (; inner + L <= num_inner; inner += L) {
                // 2-wide accumulator unrolling in reduce direction
                auto v_best0 = v_set1(x_ptr, std::numeric_limits<float>::infinity());
                auto v_best1 = v_set1(x_ptr, std::numeric_limits<float>::infinity());

                int64_t k = 0;
                for (; k + 2 <= reduce_size; k += 2) {
                    v_best0 = v_min(v_best0, v_load(x_base + (k + 0) * axis_stride + inner));
                    v_best1 = v_min(v_best1, v_load(x_base + (k + 1) * axis_stride + inner));
                }
                auto v_best = v_min(v_best0, v_best1);

                // Single-step remainder
                for (; k < reduce_size; ++k) {
                    v_best = v_min(v_best, v_load(x_base + k * axis_stride + inner));
                }

                v_store(y_base + inner, v_best);
            }

            // ---- Scalar tail ----
            for (; inner < num_inner; ++inner) {
                float best = std::numeric_limits<float>::infinity();
                for (int64_t k = 0; k < reduce_size; ++k) {
                    float x = s_load(&x_base[k * axis_stride + inner]);
                    if (x < best) { best = x; }
                }
                s_store(&y_base[inner], best);
            }
            break;
        }
        }
    };

    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, num_outer, process_outer);
    } else {
        for (int64_t i = 0; i < num_outer; ++i) { process_outer(i); }
    }
}

// ============================================================
// Path 3: General scalar (non-contiguous inner dims)
//
// Pre-computed offset vectors for both input and output, matching
// the softmax/layer_norm pattern. Handles arbitrary strides and
// pitch padding. Supports f32 and f16 (the reference only handles
// f32, so this path closes the f16 gap).
// ============================================================

template <typename T>
void reduce_general_scalar(const ReduceAttributes& attrs,
                           TensorView& output,
                           const TensorView& input,
                           int64_t axis,
                           int64_t num_outer,
                           int64_t reduce_size,
                           int64_t num_inner,
                           const ComputeContext& ctx)
{
    const int64_t rank = input.rank();
    const int64_t rank_out = output.rank();

    // Pre-compute inner offsets for input (dims axis+1..rank-1)
    std::vector<int64_t> inner_offsets(static_cast<size_t>(num_inner));
    for (int64_t flat = 0; flat < num_inner; ++flat) {
        int64_t off = 0;
        int64_t rem = flat;
        for (int64_t d = rank - 1; d >= axis + 1; --d) {
            off += (rem % input.shape(d)) * input.stride_elems(d);
            rem /= input.shape(d);
        }
        inner_offsets[static_cast<size_t>(flat)] = off;
    }

    // Pre-compute inner offsets for output.
    // Without keepdims: inner dims start at axis (reduced dim removed).
    // With keepdims: inner dims start at axis+1 (reduced dim kept as size 1).
    const int64_t y_inner_start = attrs.keepdims ? axis + 1 : axis;
    std::vector<int64_t> y_inner_offsets(static_cast<size_t>(num_inner));
    for (int64_t flat = 0; flat < num_inner; ++flat) {
        int64_t off = 0;
        int64_t rem = flat;
        for (int64_t d = rank_out - 1; d >= y_inner_start; --d) {
            off += (rem % output.shape(d)) * output.stride_elems(d);
            rem /= output.shape(d);
        }
        y_inner_offsets[static_cast<size_t>(flat)] = off;
    }

    const int64_t x_outer_stride = (axis > 0) ? input.stride_elems(axis - 1) : 0;
    const int64_t y_outer_stride = (axis > 0) ? output.stride_elems(axis - 1) : 0;
    const int64_t axis_stride = input.stride_elems(axis);

    const T* x_ptr = input.ptr<T>();
    T* y_ptr = output.ptr<T>();
    const float inv_reduce = 1.0f / static_cast<float>(reduce_size);

    auto process_outer = [&](int64_t outer) {
        const int64_t x_row_base = outer * x_outer_stride;
        const int64_t y_row_base = outer * y_outer_stride;

        for (int64_t inner = 0; inner < num_inner; ++inner) {
            const int64_t in_off = x_row_base + inner_offsets[static_cast<size_t>(inner)];
            const int64_t out_off = y_row_base + y_inner_offsets[static_cast<size_t>(inner)];

            switch (attrs.type) {
            case ReduceType::Sum:
            case ReduceType::Mean: {
                float sum = 0.0f;
                for (int64_t k = 0; k < reduce_size; ++k) {
                    sum += s_load(&x_ptr[in_off + k * axis_stride]);
                }
                float result = (attrs.type == ReduceType::Mean)
                    ? sum * inv_reduce : sum;
                s_store(&y_ptr[out_off], result);
                break;
            }
            case ReduceType::Max: {
                float best = -std::numeric_limits<float>::infinity();
                for (int64_t k = 0; k < reduce_size; ++k) {
                    float x = s_load(&x_ptr[in_off + k * axis_stride]);
                    if (x > best) { best = x; }
                }
                s_store(&y_ptr[out_off], best);
                break;
            }
            case ReduceType::Min: {
                float best = std::numeric_limits<float>::infinity();
                for (int64_t k = 0; k < reduce_size; ++k) {
                    float x = s_load(&x_ptr[in_off + k * axis_stride]);
                    if (x < best) { best = x; }
                }
                s_store(&y_ptr[out_off], best);
                break;
            }
            }
        }
    };

    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, num_outer, process_outer);
    } else {
        for (int64_t i = 0; i < num_outer; ++i) { process_outer(i); }
    }
}

}  // anonymous namespace

// ============================================================
// Templated dispatch (f32 and f16)
// ============================================================

template <typename T>
void reduce_impl(const ReduceAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();

    // Normalize axis
    int64_t axis = attrs.axis;
    if (axis < 0) { axis += rank; }
    NNOPS_ASSERT(axis >= 0 && axis < rank);

    // ---- Path 1: Contiguous tail (axis == rank-1) ----
    if (axis == rank - 1) {
        reduce_contiguous_simd<T>(attrs, output, input, axis, ctx);
        return;
    }

    // Compute dimensions for general axis
    int64_t num_outer = 1;
    for (int64_t d = 0; d < axis; ++d) { num_outer *= input.shape(d); }

    const int64_t reduce_size = input.shape(axis);

    int64_t num_inner = 1;
    for (int64_t d = axis + 1; d < rank; ++d) { num_inner *= input.shape(d); }

    // ---- Path 2: Inner-contiguous SIMD ----
    // Inner dims are contiguous iff stride along the axis equals the
    // inner block size (stride_elems(axis) == num_inner).
    // Output inner dims are contiguous iff the output row stride matches
    // the innermost dim size (no pitch padding in output).
    const bool input_inner_contiguous = (input.stride_elems(axis) == num_inner);
    const bool output_inner_contiguous = (output.rank() <= 1) ||
        (output.row_stride_elems() == output.shape(output.rank() - 1));

    if (input_inner_contiguous && output_inner_contiguous) {
        reduce_inner_contiguous_simd<T>(attrs, output, input, axis,
                                         num_outer, reduce_size, num_inner, ctx);
        return;
    }

    // ---- Path 3: General scalar ----
    reduce_general_scalar<T>(attrs, output, input, axis,
                              num_outer, reduce_size, num_inner, ctx);
}

// ============================================================
// Entry point
// ============================================================

void reduce_cpu(const ReduceAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        reduce_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        reduce_impl<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"reduce_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
