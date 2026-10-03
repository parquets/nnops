/// @file reduce_ref.cpp
/// @brief Reference (scalar) CPU implementation of the Reduce operator.
///
/// Handles single-axis reduction only.
/// Supports Sum, Min, Max, and Mean reduction types.
/// Supports planar (NCHW/NCDHW) and packed (NCHWC8/NCDHWC8) layouts.
/// Serves as the correctness baseline for SIMD and CUDA kernels.

#include "nnops/ops/reduce.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"

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

    // Normalize axis
    int64_t axis = attrs.axis;
    if (axis < 0) { axis += rank; }
    NNOPS_ASSERT(axis >= 0 && axis < rank);

    const int64_t norm_size = input.shape(axis);
    const float* in_ptr = input.ptr<float>();
    float* out_ptr = output.ptr<float>();
    const int64_t rank_out = output.rank();

    // Number of independent output elements (outer dims only)
    const int64_t num_outer = [&]() {
        int64_t n = 1;
        for (int64_t d = 0; d < axis; ++d) { n *= input.shape(d); }
        return n;
    }();
    const int64_t num_inner = [&]() {
        int64_t n = 1;
        for (int64_t d = axis + 1; d < rank; ++d) { n *= input.shape(d); }
        return n;
    }();

    // stride_elems() returns physical strides (pack-aware).
    const int64_t x_outer_stride = (axis > 0) ? input.stride_elems(axis - 1) : 0;
    const int64_t axis_stride = input.stride_elems(axis);

    // Precompute inner offsets for input (physical, pack-aware)
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

    // Precompute inner offsets for output
    const int64_t y_inner_start = attrs.keepdims ? axis + 1 : axis;
    const int64_t y_outer_stride = (axis > 0) ? output.stride_elems(axis - 1) : 0;
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

    auto process_outer = [&](int64_t outer) {
        const float* x_base = in_ptr + outer * x_outer_stride;
        float* y_base = out_ptr + outer * y_outer_stride;

        for (int64_t inner = 0; inner < num_inner; ++inner) {
            const int64_t in_off  = inner_offsets[static_cast<size_t>(inner)];
            const int64_t out_off = y_inner_offsets[static_cast<size_t>(inner)];

            float result;
            switch (attrs.type) {
            case ReduceType::Sum:
            case ReduceType::Mean: {
                double sum = 0.0;
                for (int64_t k = 0; k < norm_size; ++k) {
                    sum += static_cast<double>(x_base[in_off + k * axis_stride]);
                }
                result = static_cast<float>(
                    attrs.type == ReduceType::Mean ? sum / static_cast<double>(norm_size) : sum);
                break;
            }
            case ReduceType::Max: {
                float best = -std::numeric_limits<float>::infinity();
                for (int64_t k = 0; k < norm_size; ++k) {
                    float v = x_base[in_off + k * axis_stride];
                    if (v > best) { best = v; }
                }
                result = best;
                break;
            }
            case ReduceType::Min: {
                float best = std::numeric_limits<float>::infinity();
                for (int64_t k = 0; k < norm_size; ++k) {
                    float v = x_base[in_off + k * axis_stride];
                    if (v < best) { best = v; }
                }
                result = best;
                break;
            }
            }

            y_base[out_off] = result;
        }
    };

    ctx.cpu.run(0, num_outer, process_outer);
}

}  // namespace nnops::backend::cpu::reference
