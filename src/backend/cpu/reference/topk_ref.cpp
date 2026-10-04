/// @file topk_ref.cpp
/// @brief Naive CPU reference implementation of TopK.
///
/// Correctness baseline. Scans along the specified axis to find the k largest
/// (or smallest) values and their indices. Uses insertion into a small sorted
/// buffer, which is efficient for small k typical in practice.
///
/// Supports planar layouts (NCHW, NCDHW). F32 and F16 data types.

#include "nnops/ops/topk.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <algorithm>
#include <vector>

namespace nnops::backend::cpu::reference {

using nnops::simd::s_load;
using nnops::simd::s_store;

template <typename T>
void topk_impl_ref(const TopKAttributes& attrs,
                    TensorView& values_out,
                    TensorView& indices_out,
                    std::span<const TensorView> inputs)
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

    // Output shape = [d0, ..., d_{ax-1}, k, d_{ax+1}, ...]
    const int64_t v_outer_stride = k * inner_size;
    const int64_t v_inner_stride = k;

    const int64_t total = outer_dims * inner_size;
    if (total == 0) {
        return;
    }

    // Per-position: maintain a small array of top-k (value, index) pairs.
    struct Pair {
        float value;
        int64_t index;
    };

    std::vector<Pair> heap(static_cast<size_t>(k));

    // Comparison: for Max we want to easily replace the smallest of the top-k;
    // for Min we want to easily replace the largest of the top-k.
    // So we keep the heap sorted with the "worst" candidate at index 0.
    auto worst_cmp = [is_max](float a, float b) {
        return is_max ? (a < b) : (a > b);
    };

    for (int64_t outer = 0; outer < outer_dims; ++outer) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            int64_t i_base = outer * axis_stride * axis_dim + inner;

            for (int64_t j = 0; j < k; ++j) {
                heap[static_cast<size_t>(j)] = {
                    s_load(i_ptr + i_base + j * axis_stride), j};
            }

            // Sort so heap[0] is the worst (easiest to replace)
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
                    // k is small so this is cheap.
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
    }
}

void topk_ref(const TopKAttributes& attrs,
               TensorView& values,
               TensorView& indices,
               std::span<const TensorView> inputs,
               const ComputeContext& /*ctx*/,
               void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        topk_impl_ref<float>(attrs, values, indices, inputs);
        return;
    case DataType::f16:
        topk_impl_ref<half>(attrs, values, indices, inputs);
        return;
    default:
        NNOPS_ASSERT(!"topk_ref: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu::reference
