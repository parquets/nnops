/// @file concat_ref.cpp
/// @brief Naive CPU reference implementation of Concat.
///
/// Correctness baseline. Copies elements row-by-row from each input
/// to the output at the correct axis offset.
/// Handles all layouts (NCHW, NCDHW, NCHWC8, NCDHWC8) uniformly.

#include "nnops/ops/concat.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/simd/simd.hpp"
#include <cstring>

namespace nnops::backend::cpu::reference {

using nnops::simd::s_load;
using nnops::simd::s_store;

template <typename T>
void concat_impl_ref(const ConcatAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs)
{
    const int64_t N = static_cast<int64_t>(inputs.size());
    if (N == 0) return;

    const int64_t rank = inputs[0].rank();
    const int64_t ax = (attrs.axis < 0) ? attrs.axis + rank : attrs.axis;

    // Compute output axis offsets for each input
    std::vector<int64_t> axis_offset(static_cast<size_t>(N + 1), 0);
    for (int64_t i = 0; i < N; ++i) {
        axis_offset[static_cast<size_t>(i + 1)] =
            axis_offset[static_cast<size_t>(i)] + inputs[static_cast<size_t>(i)].shape(ax);
    }
    const int64_t total_axis = axis_offset[static_cast<size_t>(N)];

    const bool is_rank1 = (rank < 2);
    const int64_t num_rows = is_rank1 ? 1 : output.total_rows();
    const int64_t last_dim = is_rank1
        ? output.numel()
        : output.shape(rank - 1) * output.channel_pack_size();
    const int64_t o_row_stride = is_rank1 ? 1 : output.row_stride_elems();

    // For the innermost axis (axis == rank-1 for planar, rank-1 for packed):
    // we need to interleave segments within each row.
    // For packed, the actual innermost element dimension is W*pack.
    const bool axis_splits_row = (ax >= rank - 1);

    auto* o_ptr = output.ptr<T>();

    if (axis_splits_row) {
        // Concatenating within each row: each input provides a segment
        for (int64_t r = 0; r < num_rows; ++r) {
            T* o_row = o_ptr + r * o_row_stride;
            int64_t o_col = 0;
            for (int64_t n = 0; n < N; ++n) {
                const auto& in = inputs[static_cast<size_t>(n)];
                const int64_t in_row_stride = is_rank1 ? 1 : in.row_stride_elems();
                const int64_t in_last_dim = is_rank1
                    ? in.numel()
                    : in.shape(rank - 1) * in.channel_pack_size();
                const T* i_row = in.ptr<T>() + r * in_row_stride;
                for (int64_t c = 0; c < in_last_dim; ++c) {
                    s_store(&o_row[o_col++], s_load(&i_row[c]));
                }
            }
        }
    } else {
        // Concatenating on an outer axis: each input contributes entire rows
        // The output is organized by outer dimensions.
        // For each input, we need to find the right output rows.

        // Compute outer_dims (product before ax) and inner_size (elements per axis-step)
        int64_t outer_dims = 1;
        for (int64_t d = 0; d < ax; ++d) {
            outer_dims *= output.shape(d);
        }
        int64_t inner_size = 1;
        for (int64_t d = ax + 1; d < rank; ++d) {
            inner_size *= inputs[0].shape(d);
        }
        // For packed, inner_size needs to be adjusted
        if (!is_rank1) {
            // inner_size is actually "elements per axis-step",
            // but for the packed layout we need the physical element count
            const int64_t pack = inputs[0].channel_pack_size();
            if (pack > 1) {
                // inner_size already accounts for the packed last dim
                // The actual elements: inner_size = dims[ax+1] * ... * dims[rank-1] * pack
                // And the last dim includes pack implicitly
            }
        }

        // Physical inner elements = stride along axis
        const int64_t phys_inner = is_rank1 ? 1 : inputs[0].stride_elems(ax);
        const int64_t o_stride_ax = is_rank1 ? total_axis : output.stride_elems(ax);

        for (int64_t n = 0; n < N; ++n) {
            const auto& in = inputs[static_cast<size_t>(n)];
            const int64_t in_axis_dim = in.shape(ax);
            const auto* i_ptr = in.ptr<T>();
            const int64_t i_stride_ax = is_rank1 ? in_axis_dim : in.stride_elems(ax);

            // For each outer position, copy in_axis_dim blocks of inner_size elements
            for (int64_t outer = 0; outer < outer_dims; ++outer) {
                int64_t o_base = outer * o_stride_ax
                    + axis_offset[static_cast<size_t>(n)] * phys_inner;
                int64_t i_base = outer * i_stride_ax;

                for (int64_t k = 0; k < in_axis_dim; ++k) {
                    // Copy inner_size elements
                    int64_t o_pos = o_base + k * phys_inner;
                    int64_t i_pos = i_base + k * phys_inner;
                    std::memcpy(o_ptr + o_pos, i_ptr + i_pos,
                                static_cast<size_t>(phys_inner) * sizeof(T));
                }
            }
        }
    }
}

void concat_ref(const ConcatAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& /*ctx*/,
                 void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        concat_impl_ref<float>(attrs, output, inputs);
        return;
    case DataType::f16:
        concat_impl_ref<half>(attrs, output, inputs);
        return;
    default:
        NNOPS_ASSERT(!"concat_ref: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu::reference
