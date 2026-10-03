/// @file argminmax_ref.cpp
/// @brief Naive CPU reference implementation of ArgMax / ArgMin.
///
/// Correctness baseline. Scans along the specified axis to find
/// the index of the max/min value. Handles all layouts uniformly.

#include "nnops/ops/argminmax.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/simd/simd.hpp"

namespace nnops::backend::cpu::reference {

using nnops::simd::s_load;

template <typename T>
void argmax_impl_ref(const ArgMinMaxAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs)
{
    const auto& in = inputs[0];
    const int64_t rank = in.rank();
    const int64_t ax = (attrs.axis < 0) ? attrs.axis + rank : attrs.axis;
    const int64_t axis_dim = in.shape(ax);
    const bool is_max = (attrs.type == ArgMinMaxType::Max);

    // Compute outer_dims (product before ax) and inner_size (after ax)
    int64_t outer_dims = 1;
    for (int64_t d = 0; d < ax; ++d) {
        outer_dims *= in.shape(d);
    }
    int64_t inner_size = 1;
    for (int64_t d = ax + 1; d < rank; ++d) {
        inner_size *= in.shape(d);
    }

    // Physical strides
    const int64_t axis_stride = in.stride_elems(ax);  // elements between consecutive axis positions
    const auto* i_ptr = in.ptr<T>();
    auto* o_ptr = output.ptr<int64_t>();

    // Output strides
    int64_t o_outer_stride = 0, o_axis_stride = 0;
    if (attrs.keepdims) {
        o_outer_stride = output.stride_elems(ax);  // stride along outer→axis transition
        (void)o_axis_stride; // unused when keepdims (axis dim is 1)
    } else {
        // output rank = rank - 1
        // outer stride: product of output dims from 0..ax-1 (same as input outer dims)
        o_outer_stride = inner_size;
        for (int64_t d = ax + 1; d < rank; ++d) {
            // o_outer_stride already includes all post-axis dims
        }
    }

    for (int64_t outer = 0; outer < outer_dims; ++outer) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            // Base position in input
            int64_t i_base = outer * axis_stride * axis_dim + inner;

            // Find argmax/argmin along axis
            int64_t best_idx = 0;
            float   best_val = s_load(i_ptr + i_base);

            for (int64_t k = 1; k < axis_dim; ++k) {
                float val = s_load(i_ptr + i_base + k * axis_stride);
                bool better = is_max ? (val > best_val) : (val < best_val);
                if (better) {
                    best_val = val;
                    best_idx = k;
                }
            }

            // Output position
            int64_t o_pos;
            if (attrs.keepdims) {
                o_pos = outer * o_outer_stride + inner;  // axis dim is 1, skip it
            } else {
                o_pos = outer * o_outer_stride + inner;
            }
            o_ptr[o_pos] = best_idx;
        }
    }
}

void argmax_ref(const ArgMinMaxAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& /*ctx*/,
                 void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        argmax_impl_ref<float>(attrs, output, inputs);
        return;
    case DataType::f16:
        argmax_impl_ref<half>(attrs, output, inputs);
        return;
    default:
        NNOPS_ASSERT(!"argmax_ref: unsupported data type (only f32 and f16)");
    }
}

void argmin_ref(const ArgMinMaxAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx,
                 void* workspace)
{
    ArgMinMaxAttributes a = attrs;
    a.type = ArgMinMaxType::Min;
    argmax_ref(a, output, inputs, ctx, workspace);
}

}  // namespace nnops::backend::cpu::reference
