/// @file embed.cpp
/// @brief SIMD-optimized CPU implementation of embedding table lookup.
///
/// The core operation is memcpy of embedding rows, parallelized over
/// indices via cpu_parallel_for.

#include "nnops/ops/embed.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"

#include <algorithm>
#include <cstring>

namespace nnops::backend::cpu {

template <typename T>
void embed_impl(const EmbedAttributes& /*attrs*/,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx)
{
    const auto& weight  = inputs[0];
    const auto& indices = inputs[1];

    const int64_t V   = weight.shape(0);
    const int64_t dim = weight.shape(1);
    const int64_t num_indices = indices.numel();

    const T* weight_ptr = weight.ptr<T>();
    const int64_t w_row_stride = (weight.rank() == 2)
        ? dim
        : weight.row_stride_elems();

    T* out_ptr = output.ptr<T>();
    const int64_t out_row_stride = output.row_stride_elems();
    const bool is_i64 = (indices.data_type() == DataType::i64);
    const size_t row_bytes = static_cast<size_t>(dim) * sizeof(T);

    auto body = [&](int64_t n) {
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
                    row_bytes);
    };

    if (ctx.cpu_parallel_for)
        ctx.cpu_parallel_for(0, num_indices, body);
    else
        for (int64_t n = 0; n < num_indices; ++n) body(n);
}

void embed_cpu(const EmbedAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx,
                 void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        embed_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        embed_impl<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"embed_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
