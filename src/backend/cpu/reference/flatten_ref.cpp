/// @file flatten_ref.cpp
/// @brief Naive CPU reference implementation of Flatten.
///
/// Converts any input layout (NCHW, NCHWC8, NCDHW, NCDHWC8, with or
/// without pitch padding) to dense contiguous planar output.
///
/// Row-by-row processing: decompose each output logical row into
/// multi-dimensional indices, then compute the physical input offset
/// using the input's stride/pitch model.

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

    // Output is always dense planar: row_stride == last_dim
    const int64_t oW = output.shape(rank - 1);
    const int64_t o_row_stride = oW;  // dense: no pitch padding

    auto* o_ptr = output.ptr<T>();
    const auto* i_ptr = input.ptr<T>();

    // Total logical rows in the output (product of all dims except innermost):
    // N * C * (D) * H  — same for planar and packed logical view.
    int64_t num_logical_rows = 1;
    for (int64_t d = 0; d < rank - 1; ++d) {
        num_logical_rows *= output.shape(d);
    }

    // Within-row element stride for the input.
    const int64_t i_w_stride = input.stride_elems(rank - 1);  // 1 planar, pack for packed

    for (int64_t row = 0; row < num_logical_rows; ++row) {
        T* o_row = o_ptr + row * o_row_stride;

        // Decompose logical row into (dim[rank-2], ..., dim[1], dim[0])
        // Row index = ((n * C + c) * D + d) * H + h  for 3D
        //            =  (n * C + c) * H + h           for 2D
        int64_t i_off = 0;
        int64_t rem = row;

        // Iterate from innermost spatial dim outward to N
        for (int64_t d = rank - 2; d >= 0; --d) {
            int64_t dim = input.shape(d);
            int64_t idx = rem % dim;
            rem /= dim;

            if (d == 1 && pack > 1) {
                // Channel dimension with packed layout:
                // logical channel → C8 block + lane within the block
                int64_t c8_block = idx / pack;
                int64_t lane     = idx % pack;
                i_off += c8_block * input.channel_block_stride_elems() + lane;
            } else {
                i_off += idx * input.stride_elems(d);
            }
        }

        // Copy this row: o_row[w] ← input[row_start + w * i_w_stride]
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
