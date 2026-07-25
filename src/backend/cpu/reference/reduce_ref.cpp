/// @file reduce_ref.cpp
/// @brief Reference (scalar) CPU implementation of the Reduce operator.
///
/// Handles any number of axes (including empty = reduce all).
/// Supports Sum, Min, Max, and Mean reduction types.
/// Serves as the correctness baseline for SIMD and CUDA kernels.

#include "nnops/ops/reduce.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"

#include <algorithm>
#include <cfloat>
#include <vector>

namespace nnops::backend::cpu::reference {

namespace {

/// Normalize negative axes to positive, sort ascending, and validate + deduplicate.
void normalize_axes(detail::SmallVector<int64_t, 4>& axes, int64_t rank)
{
    for (auto& a : axes) {
        if (a < 0) a += rank;
        NNOPS_ASSERT(a >= 0 && a < rank);
    }
    std::sort(axes.begin(), axes.end());
    // Deduplicate
    size_t j = 0;
    for (size_t i = 0; i < axes.size(); ++i) {
        if (i == 0 || axes[i] != axes[i - 1])
            axes[j++] = axes[i];
    }
    axes.resize(j);
}

/// Count total number of elements being reduced (for Mean).
int64_t count_reduced_elements(std::span<const int64_t> input_shape,
                                const detail::SmallVector<int64_t, 4>& axes)
{
    int64_t count = 1;
    for (auto a : axes) count *= input_shape[a];
    return count;
}

}  // anonymous namespace

void reduce_ref(const ReduceAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* /*workspace*/)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();
    const auto& in_shape = input.shape_span();
    const float* in_ptr = input.ptr<float>();
    float* out_ptr = output.ptr<float>();

    // Normalize axes
    auto axes = attrs.axes;
    if (axes.empty()) {
        axes.resize(rank);
        for (int64_t d = 0; d < rank; ++d) axes[d] = d;
    }
    normalize_axes(axes, rank);

    const int64_t num_reduced = static_cast<int64_t>(axes.size());
    const int64_t reduced_count = count_reduced_elements(in_shape, axes);

    // Compute the output shape and dimension mapping.
    // Output layout: for each output flat index, expand to full-rank base
    // index (reduced dims set to 0), then iterate over reduced dims.
    const auto& out_shape = output.shape_span();
    const int64_t out_rank = static_cast<int64_t>(out_shape.size());
    const int64_t out_total = [&]() {
        int64_t n = 1;
        for (int64_t i = 0; i < out_rank; ++i) n *= out_shape[i];
        return n;
    }();

    // Build input strides
    std::vector<int64_t> in_strides(rank);
    for (int64_t d = 0; d < rank; ++d)
        in_strides[d] = input.stride_elems(d);

    // Build "keepdims" mask: for each axis, is it reduced?
    std::vector<bool> is_reduced(rank, false);
    for (auto a : axes) is_reduced[a] = true;

    // Build normalized strides (treat reduced dims as stride 1 for
    // keepdims output) and shapes for the full-rank output representation.
    std::vector<int64_t> full_out_strides(rank);
    std::vector<int64_t> full_out_shape(rank);
    for (int64_t d = 0; d < rank; ++d) {
        full_out_shape[d] = is_reduced[d] ? int64_t(1) : in_shape[d];
    }
    full_out_strides[rank - 1] = 1;
    for (int64_t d = rank - 2; d >= 0; --d)
        full_out_strides[d] = full_out_strides[d + 1] * full_out_shape[d + 1];

    // For each output element (expanded to full rank with reduced dims = size 1),
    // iterate over the Cartesian product of reduced axes.
    auto process_out = [&](int64_t out_linear) {
        // Decompose out_linear to full-rank index
        std::vector<int64_t> base_idx(rank);
        int64_t rem = out_linear;
        for (int64_t d = 0; d < rank; ++d) {
            base_idx[d] = (is_reduced[d]) ? int64_t(0) : rem / full_out_strides[d];
            rem %= full_out_strides[d];
        }

        // Iterate over Cartesian product of reduced dims
        std::vector<int64_t> counter(num_reduced, 0);
        bool done = (num_reduced == 0);

        float result;
        switch (attrs.type) {
        case ReduceType::Sum:
        case ReduceType::Mean: {
            double sum = 0.0;
            while (!done) {
                for (int64_t i = 0; i < num_reduced; ++i)
                    base_idx[axes[i]] = counter[i];
                int64_t off = 0;
                for (int64_t d = 0; d < rank; ++d)
                    off += base_idx[d] * in_strides[d];
                sum += static_cast<double>(in_ptr[off]);
                // Increment
                done = true;
                for (int64_t i = 0; i < num_reduced; ++i) {
                    if (++counter[i] < in_shape[axes[i]]) { done = false; break; }
                    counter[i] = 0;
                }
            }
            result = static_cast<float>(
                attrs.type == ReduceType::Mean ? sum / static_cast<double>(reduced_count) : sum);
            break;
        }
        case ReduceType::Max: {
            float best = -std::numeric_limits<float>::infinity();
            while (!done) {
                for (int64_t i = 0; i < num_reduced; ++i)
                    base_idx[axes[i]] = counter[i];
                int64_t off = 0;
                for (int64_t d = 0; d < rank; ++d)
                    off += base_idx[d] * in_strides[d];
                float v = in_ptr[off];
                if (v > best) best = v;
                done = true;
                for (int64_t i = 0; i < num_reduced; ++i) {
                    if (++counter[i] < in_shape[axes[i]]) { done = false; break; }
                    counter[i] = 0;
                }
            }
            result = best;
            break;
        }
        case ReduceType::Min: {
            float best = std::numeric_limits<float>::infinity();
            while (!done) {
                for (int64_t i = 0; i < num_reduced; ++i)
                    base_idx[axes[i]] = counter[i];
                int64_t off = 0;
                for (int64_t d = 0; d < rank; ++d)
                    off += base_idx[d] * in_strides[d];
                float v = in_ptr[off];
                if (v < best) best = v;
                done = true;
                for (int64_t i = 0; i < num_reduced; ++i) {
                    if (++counter[i] < in_shape[axes[i]]) { done = false; break; }
                    counter[i] = 0;
                }
            }
            result = best;
            break;
        }
        }

        out_ptr[out_linear] = result;
    };

    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, out_total, process_out);
    } else {
        for (int64_t i = 0; i < out_total; ++i) process_out(i);
    }
}

}  // namespace nnops::backend::cpu::reference
