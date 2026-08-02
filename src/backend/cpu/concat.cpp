/// @file concat.cpp
/// @brief SIMD-optimized CPU implementation of Concat.
///
/// Supports both f32 and f16 via a single templated implementation.
/// Row-by-row pitch-aware copying with SIMD inner loop + scalar tail.
///
/// Two code paths:
///   1. Axis splitting rows (axis >= rank-1): SIMD copy segments within each row
///   2. Outer axis (axis < rank-1): memcpy per contiguous block

#include "nnops/ops/concat.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <cstring>
#include <vector>

namespace nnops::backend::cpu {

using namespace nnops::simd;

template <typename T>
void concat_impl(const ConcatAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs)
{
    const int64_t N = static_cast<int64_t>(inputs.size());
    if (N == 0) return;

    const int64_t rank = inputs[0].rank();
    const int64_t ax = (attrs.axis < 0) ? attrs.axis + rank : attrs.axis;

    // Compute output axis offsets
    std::vector<int64_t> axis_offset(static_cast<size_t>(N + 1), 0);
    for (int64_t i = 0; i < N; ++i) {
        axis_offset[static_cast<size_t>(i + 1)] =
            axis_offset[static_cast<size_t>(i)] + inputs[static_cast<size_t>(i)].shape(ax);
    }

    const bool is_rank1 = (rank < 2);
    const int64_t num_rows = is_rank1 ? 1 : output.total_rows();
    const int64_t last_dim = is_rank1
        ? output.numel()
        : output.shape(rank - 1) * output.channel_pack_size();
    const int64_t o_row_stride = is_rank1 ? 1 : output.row_stride_elems();

    const bool axis_splits_row = (ax >= rank - 1);
    auto* o_ptr = output.ptr<T>();

    if (axis_splits_row) {
        // Axis splits within each row: SIMD-copy segments
        constexpr int L = simd_lane_for<T>;

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

                int64_t c = 0;
                for (; c + L <= in_last_dim; c += L) {
                    v_store(o_row + o_col + c, v_load(i_row + c));
                }
                for (; c < in_last_dim; ++c) {
                    s_store(&o_row[o_col + c], s_load(&i_row[c]));
                }
                o_col += in_last_dim;
            }
        }
    } else {
        // Outer axis: each input contributes complete blocks.
        // Copy using memcpy for contiguous physical blocks.
        const int64_t phys_inner = is_rank1 ? 1 : inputs[0].stride_elems(ax);
        const int64_t o_stride_ax = is_rank1
            ? output.numel()
            : output.stride_elems(ax);

        int64_t outer_dims = 1;
        for (int64_t d = 0; d < ax; ++d) {
            outer_dims *= output.shape(d);
        }

        for (int64_t n = 0; n < N; ++n) {
            const auto& in = inputs[static_cast<size_t>(n)];
            const int64_t in_axis_dim = in.shape(ax);
            const auto* i_ptr = in.ptr<T>();
            const int64_t i_stride_ax = is_rank1 ? in_axis_dim : in.stride_elems(ax);

            for (int64_t outer = 0; outer < outer_dims; ++outer) {
                int64_t o_base = outer * o_stride_ax
                    + axis_offset[static_cast<size_t>(n)] * phys_inner;
                int64_t i_base = outer * i_stride_ax;

                for (int64_t k = 0; k < in_axis_dim; ++k) {
                    int64_t o_pos = o_base + k * phys_inner;
                    int64_t i_pos = i_base + k * phys_inner;
                    std::memcpy(o_ptr + o_pos, i_ptr + i_pos,
                                static_cast<size_t>(phys_inner) * sizeof(T));
                }
            }
        }
    }
}

void concat_cpu(const ConcatAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& /*ctx*/,
                 void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        concat_impl<float>(attrs, output, inputs);
        return;
    case DataType::f16:
        concat_impl<half>(attrs, output, inputs);
        return;
    default:
        NNOPS_ASSERT(!"concat_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
