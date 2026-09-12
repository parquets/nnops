#pragma once
/// @file dtype_dispatch.hpp
/// @brief Shared f32/f16 dtype dispatch for CPU operator entry points.
///
/// Every `*_cpu()` entry point repeats the same switch that instantiates a
/// templated `*_impl` for f32 and f16 and asserts on anything else. This header
/// collapses that boilerplate into one call:
///
///   dispatch_f32_f16(dtype, "foo_cpu", [&](auto tag) {
///       using T = typename decltype(tag)::type;   // float or half
///       foo_impl<T>(attrs, output, inputs, ctx);
///   });

#include "nnops/core/data_type.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"

#include <type_traits>

namespace nnops::backend::cpu {

/// Invoke `fn` with `std::type_identity<float>{}` or `std::type_identity<half>{}`
/// selected by `dt`. Asserts (with `op_name` in the message) on any other dtype.
template <typename Fn>
void dispatch_f32_f16(DataType dt, const char* op_name, Fn&& fn)
{
    switch (dt) {
    case DataType::f32:
        fn(std::type_identity<float>{});
        return;
    case DataType::f16:
        fn(std::type_identity<half>{});
        return;
    default:
        break;
    }
    NNOPS_ASSERT_MSG(!"unsupported data type (only f32 and f16)", op_name);
}

}  // namespace nnops::backend::cpu
