#pragma once
/// @file dtype_dispatch.hpp
/// @brief Shared dtype dispatch for CPU operator entry points.
///
/// Every `*_cpu()` entry point repeats the same switch that instantiates a
/// templated `*_impl` for each dtype it carries. This header collapses that
/// boilerplate into one call, and the call *reports* whether it dispatched:
///
///   if (!dispatch_f32_f16(dtype, "foo_cpu", [&](auto tag) {
///           using T = typename decltype(tag)::type;   // float or half
///           foo_impl<T>(attrs, output, inputs, ctx);
///       })) {
///       return Status::unsupported;
///   }
///
/// The bool is the operator's last line of defence. validate() is the
/// descriptor-only twin that rejects an unsupported dtype first, but a caller
/// may skip it, and a dtype gate that is missing or too wide must not turn into
/// "compute() returned ok and wrote nothing" — which is what a void dispatch
/// plus an NDEBUG-compiled-out assert produced.

#include "nnops/core/data_type.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/bf16.hpp"

#include <type_traits>

namespace nnops::backend::cpu {

/// Invoke `fn` with `std::type_identity<T>{}` for each dtype the indexing and
/// data-movement kernels carry (f32, f16, bf16, s8, u8, s32, s64). Returns
/// false, calling nothing, for any other dtype (f8, s4/u4, q4_1).
template <typename Fn> bool for_each_data_type(DataType t, Fn&& fn)
{
    switch (t) {
    case DataType::f32:
        fn(std::type_identity<float>{});
        return true;
    case DataType::f16:
        fn(std::type_identity<half>{});
        return true;
    case DataType::bf16:
        fn(std::type_identity<bfloat16>{});
        return true;
    case DataType::s8:
        fn(std::type_identity<int8_t>{});
        return true;
    case DataType::u8:
        fn(std::type_identity<uint8_t>{});
        return true;
    case DataType::s32:
        fn(std::type_identity<int32_t>{});
        return true;
    case DataType::s64:
        fn(std::type_identity<int64_t>{});
        return true;
    default:
        break;
    }
    return false;
}

/// Invoke `fn` with `std::type_identity<a float type>{}` selected by `dt`:
/// float, half, or bfloat16 where the backend carries bf16 kernels. Returns
/// false for any other dtype (and for bf16 on a build without bf16 kernels), so
/// the caller can report `Status::unsupported` instead of returning ok after
/// doing nothing. The assert stays as the debug-only invariant: a call site
/// that ignores the bool is a bug in nnops.
template <typename Fn> bool dispatch_f32_f16(DataType dt, const char* op_name, Fn&& fn)
{
    switch (dt) {
    case DataType::f32:
        fn(std::type_identity<float>{});
        return true;
    case DataType::f16:
        fn(std::type_identity<half>{});
        return true;
    case DataType::bf16:
        if constexpr (nnops::detail::bf16_cpu_enabled) {
            fn(std::type_identity<bfloat16>{});
            return true;
        }
        break;
    default:
        break;
    }
    NNOPS_ASSERT_MSG(!"unsupported data type (only f32, f16 and bf16)", op_name);
    return false;
}

/// The fp dtypes, as one predicate: wherever a CPU route is templated over the
/// float type, this is the set it carries. bf16 is in the set on aarch64 only,
/// which validate() enforces before a route is chosen.
constexpr bool is_float_dtype(DataType d) noexcept
{
    return d == DataType::f32 || d == DataType::f16 || d == DataType::bf16;
}

/// The dtype a T-templated fp route carries — the inverse of the dispatch
/// above, for building a view or an output desc from the type.
template <class T> constexpr DataType flt_data_type() noexcept
{
    if constexpr (std::is_same_v<T, float>) {
        return DataType::f32;
    } else if constexpr (std::is_same_v<T, half>) {
        return DataType::f16;
    } else {
        static_assert(std::is_same_v<T, bfloat16>, "not an fp element type");
        return DataType::bf16;
    }
}

} // namespace nnops::backend::cpu
