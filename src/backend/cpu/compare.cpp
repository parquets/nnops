/// @file compare.cpp
/// @brief CPU Compare and Logical — one broadcast walk, u8 results.
///
/// Compare reads f16 through s_load so the relation runs on the float value,
/// never on the bit pattern. A NaN compares false against everything, including
/// itself, which is what IEEE does and what every relation here inherits.
/// Logical treats a nonzero word as true; the byte types are read as raw bytes.

#include "nnops/ops/compare.hpp"
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

/// Logical accepts only the integer types; keeping them in their own dispatch
/// stops the f16 branch from being instantiated, where `half != 0` has no
/// meaning.
template <class F>
void for_each_int_type(DataType t, F&& f)
{
    switch (t) {
    case DataType::s8:  f(std::type_identity<int8_t>{}); break;
    case DataType::u8:  f(std::type_identity<uint8_t>{}); break;
    case DataType::s32: f(std::type_identity<int32_t>{}); break;
    case DataType::s64: f(std::type_identity<int64_t>{}); break;
    default: break;
    }
}

template <typename T>
inline T value_at(const T* p, int64_t off) { return p[off]; }
inline float value_at(const half* p, int64_t off) { return simd::s_load(p + off); }

template <typename T>
inline bool relation(CompareType type, T a, T b)
{
    switch (type) {
    case CompareType::Equal:        return a == b;
    case CompareType::Greater:      return a > b;
    case CompareType::Less:         return a < b;
    case CompareType::GreaterOrEqual: return a >= b;
    case CompareType::LessOrEqual:  return a <= b;
    }
    return false;
}

template <typename T>
void compare_impl(TensorView& out, std::span<const TensorView> inputs,
                  CompareType type, const ComputeContext& ctx)
{
    const auto* ap = inputs[0].ptr<T>();
    const auto* bp = inputs[1].ptr<T>();
    auto* dst = out.ptr<uint8_t>();

    broadcast_walk<2>(out, inputs.data(), ctx, [&](int64_t oo, const int64_t* off) {
        dst[oo] = relation(type, value_at(ap, off[0]), value_at(bp, off[1])) ? 1 : 0;
    });
}

template <typename T>
void logical_binary_impl(TensorView& out, std::span<const TensorView> inputs,
                         LogicalType type, const ComputeContext& ctx)
{
    const auto* ap = inputs[0].ptr<T>();
    const auto* bp = inputs[1].ptr<T>();
    auto* dst = out.ptr<uint8_t>();

    broadcast_walk<2>(out, inputs.data(), ctx, [&](int64_t oo, const int64_t* off) {
        const bool a = ap[off[0]] != 0;
        const bool b = bp[off[1]] != 0;
        switch (type) {
        case LogicalType::And: dst[oo] = (a && b) ? 1 : 0; break;
        case LogicalType::Or:  dst[oo] = (a || b) ? 1 : 0; break;
        case LogicalType::Xor: dst[oo] = (a != b) ? 1 : 0; break;
        default: dst[oo] = 0; break;
        }
    });
}

template <typename T>
void logical_not_impl(TensorView& out, std::span<const TensorView> inputs,
                      const ComputeContext& ctx)
{
    const auto* ap = inputs[0].ptr<T>();
    auto* dst = out.ptr<uint8_t>();

    broadcast_walk<1>(out, inputs.data(), ctx, [&](int64_t oo, const int64_t* off) {
        dst[oo] = (ap[off[0]] == 0) ? 1 : 0;
    });
}

}  // anonymous namespace

Status compare_cpu(const CompareAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx,
                   void* /*workspace*/)
{
    for_each_data_type(inputs[0].data_type(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        compare_impl<T>(output, inputs, attrs.type, ctx);
    });
    return Status::ok;
}

Status logical_cpu(const LogicalAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx,
                   void* /*workspace*/)
{
    if (attrs.type == LogicalType::Not) {
        for_each_int_type(inputs[0].data_type(), [&](auto tag) {
            using T = typename decltype(tag)::type;
            logical_not_impl<T>(output, inputs, ctx);
        });
    } else {
        for_each_int_type(inputs[0].data_type(), [&](auto tag) {
            using T = typename decltype(tag)::type;
            logical_binary_impl<T>(output, inputs, attrs.type, ctx);
        });
    }
    return Status::ok;
}

}  // namespace nnops::backend::cpu
