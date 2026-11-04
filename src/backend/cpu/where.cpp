/// @file where.cpp
/// @brief CPU Where — a three-operand broadcast select.
///
/// The condition is read as raw bytes: a signed byte is nonzero exactly when
/// its bits are, so s8 and u8 share one test. Only the two branches are typed.

#include "nnops/ops/where.hpp"
#include "nnops/detail/assert.hpp"
#include "common/broadcast_walk.hpp"
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
void where_impl(TensorView& out, std::span<const TensorView> inputs,
                const ComputeContext& ctx)
{
    const auto* cond = inputs[0].ptr<uint8_t>();
    const auto* xp = inputs[1].ptr<T>();
    const auto* yp = inputs[2].ptr<T>();
    auto* dst = out.ptr<T>();

    broadcast_walk<3>(out, inputs.data(), ctx, [&](int64_t oo, const int64_t* off) {
        dst[oo] = cond[off[0]] ? xp[off[1]] : yp[off[2]];
    });
}

}  // anonymous namespace

Status where_cpu(const WhereAttributes&,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx,
                 void* /*workspace*/)
{
    for_each_data_type(inputs[1].data_type(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        where_impl<T>(output, inputs, ctx);
    });
    return Status::ok;
}

}  // namespace nnops::backend::cpu
