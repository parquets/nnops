/// @file reduce.cpp
/// @brief SIMD-optimized CPU implementation of the Reduce operator.
///
/// Supports both f32 and f16 via a single templated implementation.
/// Handles the fast path (contiguous tail, axis == rank-1) with SIMD.
/// All other cases (non-contiguous axis, multi-axis) delegate to reference.
///
/// Algorithm (fast path, per row):
///   - Sum/Mean: v_zero → v_add accumulate → v_reduce_sum + scalar tail
///   - Max: v_set1(-inf) → v_max accumulate → store-scan + scalar tail
///   - Min: v_set1(+inf) → v_min accumulate → store-scan + scalar tail
///
/// Pitch-aware via row_stride_elems().

#include "nnops/ops/reduce.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <algorithm>
#include <cfloat>
#include <vector>

namespace nnops::backend::cpu {

using namespace nnops::simd;

// Forward-declare reference kernel
namespace reference {
    void reduce_ref(const ReduceAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

namespace {

/// @brief Reduce a typed SIMD vector to its maximum scalar value.
template <typename T>
float reduce_max_vec(const T* /*type_tag*/, const auto& vmax) {
    T tmp[simd_lane_for<T>];
    v_store(tmp, vmax);
    float best = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < simd_lane_for<T>; ++k) {
        float x = s_load(&tmp[k]);
        if (x > best) best = x;
    }
    return best;
}

/// @brief Reduce a typed SIMD vector to its minimum scalar value.
template <typename T>
float reduce_min_vec(const T* /*type_tag*/, const auto& vmin) {
    T tmp[simd_lane_for<T>];
    v_store(tmp, vmin);
    float best = std::numeric_limits<float>::infinity();
    for (int k = 0; k < simd_lane_for<T>; ++k) {
        float x = s_load(&tmp[k]);
        if (x < best) best = x;
    }
    return best;
}

/// @brief Contiguous-tail SIMD reduction (axis == rank-1 only).
template <typename T>
void reduce_contiguous_simd(const ReduceAttributes& attrs,
                            TensorView& output,
                            const TensorView& input,
                            const ComputeContext& ctx)
{
    const int64_t rank = input.rank();
    const int64_t axis = rank - 1;

    const int64_t num_rows = [&]() {
        int64_t n = 1;
        for (int64_t d = 0; d < axis; ++d) n *= input.shape(d);
        return n;
    }();
    const int64_t norm_size = input.shape(axis);

    const T* x_ptr = input.ptr<T>();
    T* y_ptr = output.ptr<T>();
    constexpr int L = simd_lane_for<T>;

    // Row stride: distance between outermost loops (before axis).
    // For axis==rank-1 and keepdims=false, output is [outer_dims...].
    const int64_t row_stride = (axis > 0) ? input.stride_elems(axis - 1) : input.numel();

    auto process_row = [&](int64_t row) {
        const T* x_row = x_ptr + row * row_stride;
        int64_t i = 0;

        switch (attrs.type) {
        case ReduceType::Sum:
        case ReduceType::Mean: {
            auto v_sum = v_zero(x_ptr);
            for (; i + L <= norm_size; i += L)
                v_sum = v_add(v_sum, v_load(x_row + i));
            float sum = v_reduce_sum(v_sum);
            for (; i < norm_size; ++i)
                sum += s_load(&x_row[i]);
            if (attrs.type == ReduceType::Mean)
                sum /= static_cast<float>(norm_size);
            s_store(&y_ptr[row], sum);
            break;
        }
        case ReduceType::Max: {
            auto v_best = v_set1(x_ptr, -std::numeric_limits<float>::infinity());
            for (; i + L <= norm_size; i += L)
                v_best = v_max(v_best, v_load(x_row + i));
            float best = reduce_max_vec(x_ptr, v_best);
            for (; i < norm_size; ++i) {
                float x = s_load(&x_row[i]);
                if (x > best) best = x;
            }
            s_store(&y_ptr[row], best);
            break;
        }
        case ReduceType::Min: {
            auto v_best = v_set1(x_ptr, std::numeric_limits<float>::infinity());
            for (; i + L <= norm_size; i += L)
                v_best = v_min(v_best, v_load(x_row + i));
            float best = reduce_min_vec(x_ptr, v_best);
            for (; i < norm_size; ++i) {
                float x = s_load(&x_row[i]);
                if (x < best) best = x;
            }
            s_store(&y_ptr[row], best);
            break;
        }
        }
    };

    if (ctx.cpu_parallel_for)
        ctx.cpu_parallel_for(0, num_rows, process_row);
    else
        for (int64_t row = 0; row < num_rows; ++row) process_row(row);
}

}  // anonymous namespace

// ============================================================
// Entry point
// ============================================================

void reduce_cpu(const ReduceAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* workspace)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();

    // Only handle single-axis, contiguous tail (axis == rank-1) in SIMD.
    // Everything else (multi-axis, non-contiguous axis) → reference.
    if (attrs.axes.size() != 1) {
        reference::reduce_ref(attrs, output, inputs, ctx, workspace);
        return;
    }

    int64_t axis = attrs.axes[0];
    if (axis < 0) axis += rank;

    if (axis != rank - 1) {
        reference::reduce_ref(attrs, output, inputs, ctx, workspace);
        return;
    }

    // Fast path: contiguous tail
    switch (input.data_type()) {
    case DataType::f32:
        return reduce_contiguous_simd<float>(attrs, output, input, ctx);
    case DataType::f16:
        return reduce_contiguous_simd<half>(attrs, output, input, ctx);
    default:
        reference::reduce_ref(attrs, output, inputs, ctx, workspace);
        break;
    }
}

}  // namespace nnops::backend::cpu
