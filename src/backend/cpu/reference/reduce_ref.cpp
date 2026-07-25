/// @file reduce_ref.cpp
/// @brief Reference (scalar) CPU implementation of the Reduce operator.
///
/// Handles single-axis reduction only.
/// Supports Sum, Min, Max, and Mean reduction types.
/// Serves as the correctness baseline for SIMD and CUDA kernels.

#include "nnops/ops/reduce.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"

#include <algorithm>
#include <cfloat>
#include <vector>

namespace nnops::backend::cpu::reference {

void reduce_ref(const ReduceAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* /*workspace*/)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();
    const auto& in_shape = input.shape_span();

    // Normalize axis
    int64_t axis = attrs.axis;
    if (axis < 0) { axis += rank; }
    NNOPS_ASSERT(axis >= 0 && axis < rank);

    const int64_t norm_size = in_shape[axis];
    const float* in_ptr = input.ptr<float>();
    float* out_ptr = output.ptr<float>();

    // Number of independent output elements (outer dims only)
    const int64_t num_outer = [&]() {
        int64_t n = 1;
        for (int64_t d = 0; d < axis; ++d) { n *= in_shape[d]; }
        return n;
    }();
    const int64_t num_inner = [&]() {
        int64_t n = 1;
        for (int64_t d = axis + 1; d < rank; ++d) { n *= in_shape[d]; }
        return n;
    }();

    const int64_t outer_stride = input.stride_elems(axis > 0 ? axis - 1 : 0);

    // For non-contiguous axis where axis > 0: outer_stride accounts for axis dim.
    // Inner elements are contiguous (axis is a single dim), so inner_stride = 1 for
    // the elements within the inner dimension group.

    auto process_outer = [&](int64_t outer) {
        const float* x_outer = in_ptr + outer * outer_stride;
        float* y_outer = out_ptr + outer * num_inner;

        for (int64_t inner = 0; inner < num_inner; ++inner) {
            // Stride along the reduction axis: distance between consecutive
            // elements being reduced. For axis == rank-1 this is 1.
            // For non-contiguous axes: inner dims stride = 1, axis dim stride
            // needs to skip inner dims.
            const int64_t axis_stride = num_inner;  // stride along the reduction axis

            float result;
            switch (attrs.type) {
            case ReduceType::Sum:
            case ReduceType::Mean: {
                double sum = 0.0;
                for (int64_t k = 0; k < norm_size; ++k) {
                    sum += static_cast<double>(x_outer[inner + k * axis_stride]);
                }
                result = static_cast<float>(
                    attrs.type == ReduceType::Mean ? sum / static_cast<double>(norm_size) : sum);
                break;
            }
            case ReduceType::Max: {
                float best = -std::numeric_limits<float>::infinity();
                for (int64_t k = 0; k < norm_size; ++k) {
                    float v = x_outer[inner + k * axis_stride];
                    if (v > best) { best = v; }
                }
                result = best;
                break;
            }
            case ReduceType::Min: {
                float best = std::numeric_limits<float>::infinity();
                for (int64_t k = 0; k < norm_size; ++k) {
                    float v = x_outer[inner + k * axis_stride];
                    if (v < best) { best = v; }
                }
                result = best;
                break;
            }
            }

            y_outer[inner] = result;
        }
    };

    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, num_outer, process_outer);
    } else {
        for (int64_t i = 0; i < num_outer; ++i) { process_outer(i); }
    }
}

}  // namespace nnops::backend::cpu::reference
