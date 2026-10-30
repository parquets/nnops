/// @file flatten_ref.cpp
/// @brief Naive CPU reference implementation of Flatten.
///
/// Reads any input layout (NCHW, NCHWC8, NCDHW, NCDHWC8, with or without pitch
/// padding) and writes dense contiguous planar output: the output row stride is
/// its innermost extent, never a pitched one.
///
/// One output logical row at a time. The row index decomposes into
/// (dim[rank-2], ..., dim[0]) and each index maps to a physical input offset
/// through the input's stride/pitch model. The logical rows are the same for
/// planar and packed inputs; only that mapping differs, and a packed channel
/// dimension needs the block/lane split.

#include "nnops/ops/flatten.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/simd/simd.hpp"

namespace nnops::backend::cpu::reference {

using nnops::simd::s_load;
using nnops::simd::s_store;

template <typename T>
void flatten_impl_ref(const FlattenAttributes& /*attrs*/,
                       TensorView& output,
                       std::span<const TensorView> inputs)
{
    const auto& input = inputs[0];
    const int64_t rank  = input.rank();
    const int64_t pack  = input.channel_pack_size();

    const int64_t oW = output.shape(rank - 1);
    const int64_t o_row_stride = oW;

    auto* o_ptr = output.ptr<T>();
    const auto* i_ptr = input.ptr<T>();

    int64_t num_logical_rows = 1;
    for (int64_t d = 0; d < rank - 1; ++d) {
        num_logical_rows *= output.shape(d);
    }

    const int64_t i_w_stride = input.stride_elems(rank - 1);  // 1 planar, pack when packed

    for (int64_t row = 0; row < num_logical_rows; ++row) {
        T* o_row = o_ptr + row * o_row_stride;

        int64_t i_off = 0;
        int64_t rem = row;

        for (int64_t d = rank - 2; d >= 0; --d) {
            int64_t dim = input.shape(d);
            int64_t idx = rem % dim;
            rem /= dim;

            if (d == 1 && pack > 1) {
                // Packed channel: logical c → C8 block + lane inside it.
                int64_t c8_block = idx / pack;
                int64_t lane     = idx % pack;
                i_off += c8_block * input.channel_block_stride_elems() + lane;
            } else {
                i_off += idx * input.stride_elems(d);
            }
        }

        for (int64_t w = 0; w < oW; ++w) {
            s_store(&o_row[w], s_load(&i_ptr[i_off + w * i_w_stride]));
        }
    }
}

void flatten_ref(const FlattenAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& /*ctx*/,
                  void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        flatten_impl_ref<float>(attrs, output, inputs);
        return;
    case DataType::f16:
        flatten_impl_ref<half>(attrs, output, inputs);
        return;
    default:
        NNOPS_ASSERT(!"flatten_ref: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu::reference
