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
#include "nnops/detail/bf16.hpp"

#include <type_traits>

namespace nnops::backend::cpu {

/// Invoke `fn` with `std::type_identity<T>{}` for each dtype the indexing and
/// data-movement kernels carry (f32, f16, bf16, s8, u8, s32, s64); an unknown
/// dtype calls nothing.
template <typename Fn>
void for_each_data_type(DataType t, Fn&& fn)
{
    switch (t) {
    case DataType::f32:  fn(std::type_identity<float>{}); break;
    case DataType::f16:  fn(std::type_identity<half>{}); break;
    case DataType::bf16: fn(std::type_identity<bfloat16>{}); break;
    case DataType::s8:  fn(std::type_identity<int8_t>{}); break;
    case DataType::u8:  fn(std::type_identity<uint8_t>{}); break;
    case DataType::s32: fn(std::type_identity<int32_t>{}); break;
    case DataType::s64: fn(std::type_identity<int64_t>{}); break;
    default: break;
    }
}

/// Invoke `fn` with `std::type_identity<a float type>{}` selected by `dt`:
/// float, half, or bfloat16 where the backend carries bf16 kernels. Asserts
/// (with `op_name` in the message) on any other dtype — reachable only past
/// validation, which rejects bf16 on an architecture without those kernels.
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
    case DataType::bf16:
        if constexpr (nnops::detail::bf16_cpu_enabled) {
            fn(std::type_identity<bfloat16>{});
            return;
        }
        break;
    default:
        break;
    }
    NNOPS_ASSERT_MSG(!"unsupported data type (only f32, f16 and bf16)", op_name);
}

/// The fp dtypes, as one predicate: wherever a CPU route is templated over the
/// float type, this is the set it carries. bf16 is in the set on aarch64 only,
/// which validate() enforces before a route is chosen.
constexpr bool is_float_dtype(DataType d) noexcept {
    return d == DataType::f32 || d == DataType::f16 || d == DataType::bf16;
}

/// The dtype a T-templated fp route carries — the inverse of the dispatch
/// above, for building a view or an output desc from the type.
template <class T>
constexpr DataType flt_data_type() noexcept {
    if constexpr (std::is_same_v<T, float>) { return DataType::f32; }
    else if constexpr (std::is_same_v<T, half>) { return DataType::f16; }
    else {
        static_assert(std::is_same_v<T, bfloat16>, "not an fp element type");
        return DataType::bf16;
    }
}

}  // namespace nnops::backend::cpu
