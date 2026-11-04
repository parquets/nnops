/// @file split.cpp
/// @brief CPU Split — one logical walk per slice.
///
/// Output j covers the axis range [off, off + len); its walk adds off to the
/// axis coordinate of every element. The outputs run one after another, each
/// with the full thread pool (the walks already parallelize).

#include "nnops/ops/split.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "common/strided_walk.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <cstdint>
#include <type_traits>
#include <utility>

namespace nnops::backend::cpu {

namespace {

template <class F>
void for_each_data_type(DataType t, F&& f)
{
    switch (t) {
    case DataType::f32: f(std::type_identity<float>{}); break;
    case DataType::f16: f(std::type_identity<half>{}); break;
    case DataType::s8:  f(std::type_identity<int8_t>{}); break;
    case DataType::u8:  f(std::type_identity<uint8_t>{}); break;
    case DataType::s32: f(std::type_identity<int32_t>{}); break;
    case DataType::s64: f(std::type_identity<int64_t>{}); break;
    default: break;
    }
}

template <typename T>
void split_impl(const TensorView& input, std::span<TensorView> outputs,
                const SplitAttributes& attrs, const ComputeContext& ctx)
{
    const int64_t axis = resolve_axis(attrs.axis, input.rank());
    const int64_t n = static_cast<int64_t>(outputs.size());
    const int64_t dim = input.shape(axis);

    int64_t istr[8];
    logical_strides(input, istr);

    const T* ip = input.ptr<T>();

    int64_t off = 0;
    for (int64_t j = 0; j < n; ++j) {
        TensorView& out = outputs[static_cast<size_t>(j)];
        const int64_t len = attrs.split.empty() ? dim / n
                                                : attrs.split[static_cast<size_t>(j)];

        int64_t odims[8], ostr[8];
        logical_strides(out, ostr);
        for (int64_t d = 0; d < out.rank(); ++d) { odims[d] = out.shape(d); }

        T* op = out.ptr<T>();
        const int64_t base = off * istr[axis];

        walk_logical(odims, out.rank(), out.numel(), ctx, [&](const int64_t* o) {
            int64_t is = base, os = 0;
            for (int64_t d = 0; d < out.rank(); ++d) {
                is += o[d] * istr[d];
                os += o[d] * ostr[d];
            }
            op[os] = ip[is];
        });

        off += len;
    }
}

}  // anonymous namespace

Status split_cpu(const SplitAttributes& attrs,
                 std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx,
                 void* /*workspace*/)
{
    for_each_data_type(inputs[0].data_type(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        split_impl<T>(inputs[0], outputs, attrs, ctx);
    });
    return Status::ok;
}

}  // namespace nnops::backend::cpu
