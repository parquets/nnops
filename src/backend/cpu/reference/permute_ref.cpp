/// @file permute_ref.cpp
/// @brief Scalar CPU reference implementation of dimension permute (transpose).
///
/// For planar data, computes input strides and output strides in elements,
/// then copies each element from the permuted input position.

#include "nnops/ops/permute.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"

#include <algorithm>
#include <cstring>

namespace nnops::backend::cpu::reference {

template <typename T>
void permute_impl_ref(const PermuteAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();
    const int64_t total = output.numel();
    const auto& perm = attrs.perm;

    // Input strides in elements (planar → contiguous, so last dim stride = 1)
    int64_t in_strides[TensorDesc::kMaxRank];
    in_strides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * input.shape(i + 1);
    }

    // Output strides in elements
    int64_t out_strides[TensorDesc::kMaxRank];
    out_strides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; --i) {
        out_strides[i] = out_strides[i + 1] * output.shape(i + 1);
    }

    const T* in_ptr  = input.ptr<T>();
    T*       out_ptr = output.ptr<T>();

    for (int64_t out_idx = 0; out_idx < total; ++out_idx) {
        // Decompose flat output index into multi-dimensional coordinates
        int64_t tmp = out_idx;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < rank; ++d) {
            int64_t coord = tmp / out_strides[d];
            tmp %= out_strides[d];
            // perm[d] is the source dimension in input
            in_idx += coord * in_strides[perm[static_cast<size_t>(d)]];
        }
        out_ptr[out_idx] = in_ptr[in_idx];
    }
}

void permute_ref(const PermuteAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& /*ctx*/,
                 void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        permute_impl_ref<float>(attrs, output, inputs);
        return;
    case DataType::f16:
        permute_impl_ref<half>(attrs, output, inputs);
        return;
    default:
        NNOPS_ASSERT(!"permute_ref: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu::reference
