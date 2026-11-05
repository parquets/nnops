/// @file concat_ref.cpp
/// @brief Naive CPU reference for Concat; the correctness baseline.
///
/// One element at a time: the output's logical flat index is decomposed to an
/// index vector, the axis entry selects the source input, and both physical
/// offsets come from stride_elems(). That covers channel-first and channel-last
/// alike, 2D and 3D, without a layout-specific branch — a channel-last row
/// interleaves C, so a contiguous-copy formulation would be wrong there.

#include "nnops/ops/concat.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/simd/simd.hpp"
#include <vector>

namespace nnops::backend::cpu::reference {

using nnops::simd::s_load;
using nnops::simd::s_store;

template <typename T>
void concat_impl_ref(const ConcatAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs)
{
    const int64_t N = static_cast<int64_t>(inputs.size());
    if (N == 0) {
        return;
    }

    const int64_t rank = inputs[0].rank();
    const int64_t ax = (attrs.axis < 0) ? attrs.axis + rank : attrs.axis;

    std::vector<int64_t> axis_offset(static_cast<size_t>(N + 1), 0);
    for (int64_t i = 0; i < N; ++i) {
        axis_offset[static_cast<size_t>(i + 1)] =
            axis_offset[static_cast<size_t>(i)] + inputs[static_cast<size_t>(i)].shape(ax);
    }

    const int64_t total = output.numel();
    auto* o_ptr = output.ptr<T>();

    std::vector<int64_t> idx(static_cast<size_t>(rank), 0);
    int64_t o_off = 0;

    for (int64_t flat = 0; flat < total; ++flat) {
        const int64_t o_ax = (rank > 0) ? idx[static_cast<size_t>(ax)] : 0;

        // Select the input whose axis span contains o_ax.
        int64_t in_idx = 0;
        while (in_idx < N - 1 &&
               o_ax >= axis_offset[static_cast<size_t>(in_idx + 1)]) {
            ++in_idx;
        }
        const auto& in = inputs[static_cast<size_t>(in_idx)];
        const int64_t i_ax = o_ax - axis_offset[static_cast<size_t>(in_idx)];

        int64_t i_off = 0;
        for (int64_t d = 0; d < rank; ++d) {
            const int64_t d_idx = (d == ax) ? i_ax : idx[static_cast<size_t>(d)];
            i_off += d_idx * in.stride_elems(d);
        }

        s_store(&o_ptr[o_off], s_load(&in.ptr<T>()[i_off]));

        // Advance the odometer and the output offset.
        for (int64_t d = rank - 1; d >= 0; --d) {
            if (++idx[static_cast<size_t>(d)] < output.shape(d)) {
                o_off += output.stride_elems(d);
                break;
            }
            idx[static_cast<size_t>(d)] = 0;
            o_off -= output.stride_elems(d) * (output.shape(d) - 1);
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
