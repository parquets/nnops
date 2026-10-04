/// @file topk.cpp
/// @brief SIMD-optimized CPU implementation of TopK.
///
/// Supports both f32 and f16 via templated implementation.
/// Uses scalar selection (top-k is inherently sequential along the axis).
///
/// Supports planar layouts (NCHW, NCDHW). Not for packed layouts.

#include "nnops/ops/topk.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "common/dtype_dispatch.hpp"

#include <algorithm>
#include <vector>

namespace nnops::backend::cpu {

using nnops::simd::s_load;
using nnops::simd::s_store;

template <typename T>
void topk_impl(const TopKAttributes& attrs,
                TensorView& values_out,
                TensorView& indices_out,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx)
{
    const auto& in = inputs[0];
    const int64_t rank = in.rank();
    const int64_t ax = (attrs.axis < 0) ? attrs.axis + rank : attrs.axis;
    const int64_t axis_dim = in.shape(ax);
    const int64_t k = attrs.k;

    NNOPS_ASSERT(k >= 1);
    NNOPS_ASSERT(k <= axis_dim);
    NNOPS_ASSERT(ax >= 0 && ax < rank);

    const bool is_max = (attrs.type == TopKType::Max);
    const bool sorted_opt = attrs.sorted;

    int64_t outer_dims = 1;
    for (int64_t d = 0; d < ax; ++d) {
        outer_dims *= in.shape(d);
    }
    int64_t inner_size = 1;
    for (int64_t d = ax + 1; d < rank; ++d) {
        inner_size *= in.shape(d);
    }

    const auto* i_ptr = in.ptr<T>();
    auto* v_ptr = values_out.ptr<T>();
    auto* idx_ptr = indices_out.ptr<int64_t>();

    const int64_t axis_stride = in.stride_elems(ax);
    const int64_t v_outer_stride = k * inner_size;
    const int64_t v_inner_stride = k;

    struct Pair {
        float value;
        int64_t index;
    };

    auto worst_cmp = [is_max](float a, float b) {
        return is_max ? (a < b) : (a > b);
    };

    // Process each (outer, inner) position. Parallelize over outer dims
    // when there are enough positions.
    auto process_outer = [&](int64_t outer) {
        // Per-thread heap to avoid allocation in hot loop
        std::vector<Pair> heap(static_cast<size_t>(k));

        for (int64_t inner = 0; inner < inner_size; ++inner) {
            int64_t i_base = outer * axis_stride * axis_dim + inner;

            for (int64_t j = 0; j < k; ++j) {
                heap[static_cast<size_t>(j)] = {
                    s_load(i_ptr + i_base + j * axis_stride), j};
            }

            std::sort(heap.begin(), heap.end(),
                [&](const Pair& a, const Pair& b) {
                    return worst_cmp(a.value, b.value);
                });

            for (int64_t j = k; j < axis_dim; ++j) {
                float val = s_load(i_ptr + i_base + j * axis_stride);
                if (worst_cmp(heap[0].value, val)) {
                    heap[0] = {val, j};
                    // Re-sort: heap[0] is the worst candidate (easiest to replace).
                    // After replacement the heap may be out of order — re-sort.
                    std::sort(heap.begin(), heap.end(),
                        [&](const Pair& a, const Pair& b) {
                            return worst_cmp(a.value, b.value);
                        });
                }
            }

            if (sorted_opt) {
                if (is_max) {
                    std::sort(heap.begin(), heap.end(),
                        [](const Pair& a, const Pair& b) {
                            return a.value > b.value;
                        });
                } else {
                    std::sort(heap.begin(), heap.end(),
                        [](const Pair& a, const Pair& b) {
                            return a.value < b.value;
                        });
                }
            }

            int64_t v_base = outer * v_outer_stride + inner * v_inner_stride;
            for (int64_t j = 0; j < k; ++j) {
                s_store(v_ptr + v_base + j, heap[static_cast<size_t>(j)].value);
                idx_ptr[v_base + j] = heap[static_cast<size_t>(j)].index;
            }
        }
    };

    ctx.cpu.run(0, outer_dims, process_outer);
}

void topk_cpu(const TopKAttributes& attrs,
               TensorView& values,
               TensorView& indices,
               std::span<const TensorView> inputs,
               const ComputeContext& ctx,
               void* /*workspace*/)
{
    dispatch_f32_f16(inputs[0].data_type(), "topk_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        topk_impl<T>(attrs, values, indices, inputs, ctx);
    });
}

}  // namespace nnops::backend::cpu
