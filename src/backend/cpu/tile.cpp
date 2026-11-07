/// @file tile.cpp
/// @brief CPU Tile / Expand — repeat axes by element copy.
///
/// Tile maps out index o to in index o % in_dim on every axis; Expand maps a
/// right-aligned o to the input index, 0 where the input axis is 1 or absent.
/// Both are the same chunked logical walk as the gather family, with the source
/// offset computed from the input's own strides.

#include "nnops/ops/tile.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "common/strided_walk.hpp"
#include "common/dtype_dispatch.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <cstdint>
#include <type_traits>
#include <utility>

namespace nnops::backend::cpu {

namespace {

template <typename T>
void tile_impl(const TileAttributes& /*attrs*/, TensorView& output,
               const TensorView& input, const ComputeContext& ctx)
{
    const int64_t rank = input.rank();
    int64_t odims[8], ostr[8], istr[8], idims[8];
    logical_strides(input, istr);
    logical_strides(output, ostr);
    for (int64_t d = 0; d < rank; ++d) {
        odims[d] = output.shape(d);
        idims[d] = input.shape(d);
    }

    const T* ip = input.ptr<T>();
    T* op = output.ptr<T>();

    walk_logical(odims, rank, output.numel(), ctx, [&](const int64_t* o) {
        int64_t is = 0, os = 0;
        for (int64_t d = 0; d < rank; ++d) {
            is += (idims[d] > 0 ? o[d] % idims[d] : 0) * istr[d];
            os += o[d] * ostr[d];
        }
        op[os] = ip[is];
    });
}

template <typename T>
void expand_impl(const ExpandAttributes& /*attrs*/, TensorView& output,
                 const TensorView& input, const ComputeContext& ctx)
{
    const int64_t orank = output.rank();
    const int64_t irank = input.rank();
    int64_t odims[8], ostr[8], istr[8], idims[8];
    logical_strides(input, istr);
    logical_strides(output, ostr);
    for (int64_t d = 0; d < orank; ++d) { odims[d] = output.shape(d); }
    for (int64_t d = 0; d < irank; ++d) { idims[d] = input.shape(d); }

    const T* ip = input.ptr<T>();
    T* op = output.ptr<T>();

    walk_logical(odims, orank, output.numel(), ctx, [&](const int64_t* o) {
        int64_t is = 0, os = 0;
        for (int64_t d = 0; d < orank; ++d) {
            os += o[d] * ostr[d];
            const int64_t ii = irank - orank + d;
            if (ii >= 0 && idims[ii] != 1) { is += o[d] * istr[ii]; }
        }
        op[os] = ip[is];
    });
}

}  // anonymous namespace

Status tile_cpu(const TileAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* /*workspace*/)
{
    for_each_data_type(inputs[0].data_type(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        tile_impl<T>(attrs, output, inputs[0], ctx);
    });
    return Status::ok;
}

Status expand_cpu(const ExpandAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx,
                  void* /*workspace*/)
{
    for_each_data_type(inputs[0].data_type(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        expand_impl<T>(attrs, output, inputs[0], ctx);
    });
    return Status::ok;
}

}  // namespace nnops::backend::cpu
