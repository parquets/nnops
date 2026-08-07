/// @file slice_ref.cpp
/// @brief Naive CPU reference implementation of Slice.
///
/// Correctness baseline. Handles all planar layouts uniformly.
/// Uses flat-index decomposition: for each output element, compute the
/// corresponding input flat index via starts/steps/axes mapping.

#include "nnops/ops/slice.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"

#include <algorithm>
#include <cstring>

namespace nnops::backend::cpu::reference {

namespace {

template <typename T>
void slice_impl_ref(const SliceAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();
    const int64_t total = output.numel();
    if (total == 0) return;

    const size_t n_axes = attrs.axes.size();

    // Resolve effective start/step per axis (default: 0, input dim, 1)
    int64_t eff_start[TensorDesc::kMaxRank];
    int64_t eff_step[TensorDesc::kMaxRank];
    for (int64_t d = 0; d < rank; ++d) {
        eff_start[d] = 0;
        eff_step[d]  = 1;
    }
    for (size_t a = 0; a < n_axes; ++a) {
        int64_t ax = attrs.axes[a];
        if (ax < 0) ax += rank;
        eff_start[static_cast<size_t>(ax)] = attrs.starts[a];
        eff_step[static_cast<size_t>(ax)]  =
            (a < attrs.steps.size()) ? attrs.steps[a] : int64_t(1);
    }

    // Input strides in elements (planar → contiguous innermost)
    int64_t in_strides[TensorDesc::kMaxRank];
    in_strides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; --i)
        in_strides[i] = in_strides[i + 1] * input.shape(i + 1);

    // Output strides in elements (planar)
    int64_t out_strides[TensorDesc::kMaxRank];
    out_strides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; --i)
        out_strides[i] = out_strides[i + 1] * output.shape(i + 1);

    const T* in_ptr  = input.ptr<T>();
    T*       out_ptr = output.ptr<T>();

    for (int64_t out_idx = 0; out_idx < total; ++out_idx) {
        // Decompose flat output index → multi-index
        int64_t tmp = out_idx;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < rank; ++d) {
            int64_t coord = tmp / out_strides[d];
            tmp %= out_strides[d];
            // Map output coord to input coord via slice parameters
            int64_t in_coord = eff_start[static_cast<size_t>(d)]
                             + coord * eff_step[static_cast<size_t>(d)];
            in_idx += in_coord * in_strides[d];
        }
        out_ptr[out_idx] = in_ptr[in_idx];
    }
}

}  // anonymous namespace

void slice_ref(const SliceAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& /*ctx*/,
                void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        slice_impl_ref<float>(attrs, output, inputs);
        return;
    case DataType::f16:
        slice_impl_ref<half>(attrs, output, inputs);
        return;
    default:
        NNOPS_ASSERT(!"slice_ref: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu::reference
