/// @file embed_ref.cpp
/// @brief Scalar CPU reference implementation of embedding table lookup.
///
/// For each index in the indices tensor, copies the corresponding row
/// from the weight table to the output. Out-of-range indices are clamped
/// to [0, vocab_size - 1].

#include "nnops/ops/embed.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"

#include <algorithm>
#include <cstring>

namespace nnops::backend::cpu::reference {

template <typename T>
void embed_impl_ref(const EmbedAttributes& /*attrs*/,
                     TensorView& output,
                     std::span<const TensorView> inputs)
{
    const auto& weight  = inputs[0];
    const auto& indices = inputs[1];

    const int64_t V   = weight.shape(0);   // vocab size
    const int64_t dim = weight.shape(1);   // embedding dim
    const int64_t num_indices = indices.numel();

    const T* weight_ptr = weight.ptr<T>();
    const int64_t w_row_stride = (weight.rank() == 2)
        ? dim
        : weight.row_stride_elems();

    T* out_ptr = output.ptr<T>();
    const int64_t out_row_stride = output.row_stride_elems();

    // Handle both int64 and int32 indices
    const bool is_i64 = (indices.data_type() == DataType::i64);

    for (int64_t n = 0; n < num_indices; ++n) {
        int64_t idx;
        if (is_i64) {
            idx = static_cast<const int64_t*>(indices.ptr<void>())[n];
        } else {
            idx = static_cast<const int32_t*>(indices.ptr<void>())[n];
        }

        // Clamp to valid range
        if (idx < 0) idx = 0;
        if (idx >= V) idx = V - 1;

        // Copy weight[idx, :] to output[n, :]
        std::memcpy(out_ptr + n * out_row_stride,
                    weight_ptr + idx * w_row_stride,
                    static_cast<size_t>(dim) * sizeof(T));
    }
}

void embed_ref(const EmbedAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& /*ctx*/,
                 void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        embed_impl_ref<float>(attrs, output, inputs);
        return;
    case DataType::f16:
        embed_impl_ref<half>(attrs, output, inputs);
        return;
    default:
        NNOPS_ASSERT(!"embed_ref: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu::reference
