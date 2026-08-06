/// @file flatten.cpp
/// @brief SIMD-optimized CPU implementation of Flatten.
///
/// Row-by-row SIMD copy from any input layout to dense planar output.
/// Uses the generic SIMD API (v_load/v_store) to handle both f32 and f16.
///
/// The key optimization: detect when the input is already dense planar
/// (pack == 1, row_stride == oW) → single memcpy of the entire buffer.

#include "nnops/ops/flatten.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <cstring>

namespace nnops::backend::cpu {

using namespace nnops::simd;

template <typename T>
void flatten_impl(const FlattenAttributes& /*attrs*/,
                   TensorView& output,
                   std::span<const TensorView> inputs)
{
    const auto& input = inputs[0];
    const int64_t rank  = input.rank();
    const int64_t pack  = input.channel_pack_size();
    constexpr int L = simd_lane_for<T>;

    // Output is always dense planar
    const int64_t oW = output.shape(rank - 1);
    const int64_t o_row_stride = oW;

    auto* o_ptr = output.ptr<T>();
    const auto* i_ptr = input.ptr<T>();
    const int64_t i_w_stride = input.stride_elems(rank - 1);

    // ---- Fast path: already dense planar → single memcpy ----
    if (pack == 1 && input.row_stride_elems() == oW) {
        size_t total_bytes = static_cast<size_t>(output.numel()) * sizeof(T);
        std::memcpy(o_ptr, i_ptr, total_bytes);
        return;
    }

    // ---- General path: row-by-row SIMD copy ----

    int64_t num_logical_rows = 1;
    for (int64_t d = 0; d < rank - 1; ++d)
        num_logical_rows *= output.shape(d);

    for (int64_t row = 0; row < num_logical_rows; ++row) {
        T* o_row = o_ptr + row * o_row_stride;

        // Decompose logical row into multi-dimensional indices
        int64_t i_off = 0;
        int64_t rem = row;
        for (int64_t d = rank - 2; d >= 0; --d) {
            int64_t dim = input.shape(d);
            int64_t idx = rem % dim;
            rem /= dim;

            if (d == 1 && pack > 1) {
                int64_t c8_block = idx / pack;
                int64_t lane     = idx % pack;
                i_off += c8_block * input.channel_block_stride_elems() + lane;
            } else {
                i_off += idx * input.stride_elems(d);
            }
        }

        // SIMD copy within the row
        int64_t w = 0;
        if (i_w_stride == 1 && pack == 1) {
            // Planar input: contiguous row → SIMD vector copy
            for (; w + L <= oW; w += L)
                v_store(o_row + w, v_load(i_ptr + i_off + w));
        } else {
            // Packed input: strided element access
            for (; w + L <= oW; w += L) {
                T tmp[16];  // max simd lane width
                for (int k = 0; k < L; ++k)
                    s_store(&tmp[k], s_load(&i_ptr[i_off + (w + k) * i_w_stride]));
                v_store(o_row + w, v_load(tmp));
            }
        }
        // Scalar tail
        for (; w < oW; ++w)
            s_store(&o_row[w], s_load(&i_ptr[i_off + w * i_w_stride]));
    }
}

void flatten_cpu(const FlattenAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& /*ctx*/,
                  void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        flatten_impl<float>(attrs, output, inputs);
        return;
    case DataType::f16:
        flatten_impl<half>(attrs, output, inputs);
        return;
    default:
        NNOPS_ASSERT(!"flatten_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
