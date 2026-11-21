/// @file flatten.cpp
/// @brief CPU Flatten — a storage copy into dense row-major order.
///
/// The input is read through its own strides, so a channel-last tensor
/// [N,H,W,C] flattens onto dense [N,H,W,C] storage as one memcpy of the whole
/// buffer; a strided view copies row by row (memcpy when the row is dense,
/// element by element when it is not). Both are byte copies, so every
/// whole-element dtype is carried — f16 bits and int64 indices included.

#include "nnops/ops/flatten.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "common/dtype_dispatch.hpp"

#include <algorithm>
#include <cstring>

namespace nnops::backend::cpu {

using namespace nnops::simd;

template <typename T>
void flatten_impl(const FlattenAttributes& /*attrs*/, TensorView& output,
                  std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();

    const int64_t oW = output.shape(rank - 1);
    // The output's row step is its pitch: a padded destination is a legal view.
    const int64_t o_row_stride = output.row_stride_elems();

    auto* o_ptr = output.ptr<T>();
    const auto* i_ptr = input.ptr<T>();
    const int64_t i_w_stride = input.stride_elems(rank - 1);

    // Already in output order: the input's axes are the storage order with no
    // row pitch between them, and the output's rows are unpadded too — the whole
    // op is one memcpy. A channel-last [N,H,W,C] read as dense is this case even
    // though its own row stride is only W*C.
    if (output.row_stride_elems() == oW) {
        bool compact = true;
        int64_t want = 1;
        for (int64_t d = rank - 1; d >= 0; --d) {
            if (input.shape(d) != 1 && input.stride_elems(d) != want) {
                compact = false;
                break;
            }
            want *= input.shape(d);
        }
        if (compact) {
            // One copy, split across the pool: a 12 MB feature map is a
            // bandwidth-bound memcpy and one core reaches a fraction of it.
            constexpr int64_t kChunk = int64_t{1} << 16;
            const int64_t total = output.numel();
            const int64_t chunks = (total + kChunk - 1) / kChunk;
            ctx.cpu.run(0, chunks, [&](int64_t c) {
                const int64_t off = c * kChunk;
                const int64_t n = std::min(kChunk, total - off);
                std::memcpy(o_ptr + off, i_ptr + off, static_cast<size_t>(n) * sizeof(T));
            });
            return;
        }
    }

    int64_t num_logical_rows = 1;
    for (int64_t d = 0; d < rank - 1; ++d) {
        num_logical_rows *= output.shape(d);
    }

    for (int64_t row = 0; row < num_logical_rows; ++row) {
        T* o_row = o_ptr + row * o_row_stride;

        // Decompose logical row into multi-dimensional indices
        int64_t i_off = 0;
        int64_t rem = row;
        for (int64_t d = rank - 2; d >= 0; --d) {
            int64_t dim = input.shape(d);
            int64_t idx = rem % dim;
            rem /= dim;
            i_off += idx * input.stride_elems(d);
        }

        // A dense row is one memcpy whatever the element type is (this is a
        // storage copy, not arithmetic); a strided row walks elements.
        if (i_w_stride == 1) {
            std::memcpy(o_row, i_ptr + i_off, static_cast<size_t>(oW) * sizeof(T));
        } else {
            for (int64_t w = 0; w < oW; ++w) {
                o_row[w] = i_ptr[i_off + w * i_w_stride];
            }
        }
    }
}

Status flatten_cpu(const FlattenAttributes& attrs, TensorView& output,
                   std::span<const TensorView> inputs, const ComputeContext& ctx,
                   void* /*workspace*/)
{
    if (!for_each_data_type(inputs[0].data_type(), [&](auto tag) {
            using T = typename decltype(tag)::type;
            flatten_impl<T>(attrs, output, inputs, ctx);
        })) {
        return Status::unsupported;
    }
    return Status::ok;
}

} // namespace nnops::backend::cpu
