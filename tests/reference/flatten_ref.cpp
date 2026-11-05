/// @file flatten_ref.cpp
/// @brief Naive CPU reference implementation of Flatten.
///
/// Reads any input layout (channel-first or channel-last, 2D or 3D, with or
/// without pitch padding) and writes dense contiguous output in logical
/// (channel-first) order. Each logical flat index maps to a physical offset
/// through stride_elems(), which is what makes the same walk cover
/// channel-first and channel-last inputs.

#include "nnops/ops/flatten.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/simd/simd.hpp"

namespace nnops::backend::cpu::reference {

using nnops::simd::s_load;
using nnops::simd::s_store;

namespace {

/// Offset of logical flat index @p flat over all dims, row-major.
int64_t flat_offset(const TensorView& t, int64_t flat) {
    int64_t off = 0;
    int64_t rem = flat;
    for (int64_t d = t.rank() - 1; d >= 0; --d) {
        const int64_t dim = t.shape(d);
        off += (rem % dim) * t.stride_elems(d);
        rem /= dim;
    }
    return off;
}

}  // anonymous namespace

template <typename T>
void flatten_impl_ref(const FlattenAttributes& /*attrs*/,
                      TensorView& output,
                      std::span<const TensorView> inputs)
{
    const auto& input = inputs[0];
    const int64_t total = output.numel();

    auto* o_ptr = output.ptr<T>();
    const auto* i_ptr = input.ptr<T>();

    for (int64_t flat = 0; flat < total; ++flat) {
        s_store(&o_ptr[flat], s_load(&i_ptr[flat_offset(input, flat)]));
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
