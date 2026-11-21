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
#include "common/dtype_dispatch.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace nnops::backend::cpu {

namespace {

/// Logical accepts only the integer types; keeping them in their own dispatch
/// stops the f16 branch from being instantiated, where `half != 0` has no
/// meaning.
template <class F> void for_each_int_type(DataType t, F&& f)
{
    switch (t) {
    case DataType::s8:
        f(std::type_identity<int8_t>{});
        break;
    case DataType::u8:
        f(std::type_identity<uint8_t>{});
        break;
    case DataType::s32:
        f(std::type_identity<int32_t>{});
        break;
    case DataType::s64:
        f(std::type_identity<int64_t>{});
        break;
    default:
        break;
    }
}

template <typename T> inline T value_at(const T* p, int64_t off) { return p[off]; }
inline float value_at(const half* p, int64_t off) { return simd::s_load(p + off); }

/// The relation as a functor: picked once per call rather than switched on for
/// every element. `f16` is read through s_load so the test is on the value, and
/// the operands keep their own type so an s64 compare stays exact.
struct Relation {
    CompareType type;
    template <typename V> bool operator()(V a, V b) const
    {
        switch (type) {
        case CompareType::Equal:
            return a == b;
        case CompareType::Greater:
            return a > b;
        case CompareType::Less:
            return a < b;
        case CompareType::GreaterOrEqual:
            return a >= b;
        case CompareType::LessOrEqual:
            return a <= b;
        }
        return false;
    }
};

/// `dst = op(a, b)` over contiguous output runs.
template <typename T, typename Op>
void compare_loop(TensorView& out, const TensorView* ins, const ComputeContext& ctx, Op&& op)
{
    const auto* ap = ins[0].template ptr<T>();
    const auto* bp = ins[1].template ptr<T>();
    auto* dst = out.ptr<uint8_t>();

    broadcast_walk_run<2>(out, ins, ctx,
                          [&](int64_t oo, const int64_t* off, const int64_t* step, int64_t run) {
                              const T* a_row = ap + off[0];
                              const T* b_row = bp + off[1];
                              uint8_t* o_row = dst + oo;
                              for (int64_t i = 0; i < run; ++i) {
                                  o_row[i] = op(value_at(a_row, i * step[0]),
                                                value_at(b_row, i * step[1])) ? 1 : 0;
                              }
                          });
}

template <typename T>
void compare_impl(TensorView& out, std::span<const TensorView> inputs, CompareType type,
                  const ComputeContext& ctx)
{
    compare_loop<T>(out, inputs.data(), ctx, Relation{type});
}

template <typename T, typename Op>
void logical_loop(TensorView& out, const TensorView* ins, const ComputeContext& ctx, Op&& op)
{
    const auto* ap = ins[0].template ptr<T>();
    const auto* bp = ins[1].template ptr<T>();
    auto* dst = out.ptr<uint8_t>();

    broadcast_walk_run<2>(out, ins, ctx,
                          [&](int64_t oo, const int64_t* off, const int64_t* step, int64_t run) {
                              const T* a_row = ap + off[0];
                              const T* b_row = bp + off[1];
                              uint8_t* o_row = dst + oo;
                              for (int64_t i = 0; i < run; ++i) {
                                  o_row[i] = op(a_row[i * step[0]], b_row[i * step[1]]) ? 1 : 0;
                              }
                          });
}

struct LogicalAnd {
    bool operator()(auto a, auto b) const { return a != 0 && b != 0; }
};
struct LogicalOr {
    bool operator()(auto a, auto b) const { return a != 0 || b != 0; }
};
struct LogicalXor {
    bool operator()(auto a, auto b) const { return (a != 0) != (b != 0); }
};

template <typename T>
void logical_binary_impl(TensorView& out, std::span<const TensorView> inputs, LogicalType type,
                         const ComputeContext& ctx)
{
    switch (type) {
    case LogicalType::And:
        logical_loop<T>(out, inputs.data(), ctx, LogicalAnd{});
        return;
    case LogicalType::Or:
        logical_loop<T>(out, inputs.data(), ctx, LogicalOr{});
        return;
    case LogicalType::Xor:
        logical_loop<T>(out, inputs.data(), ctx, LogicalXor{});
        return;
    default:
        std::fill_n(out.ptr<uint8_t>(), static_cast<size_t>(out.numel()), uint8_t{0});
        return;
    }
}

template <typename T>
void logical_not_impl(TensorView& out, std::span<const TensorView> inputs,
                      const ComputeContext& ctx)
{
    const auto* ap = inputs[0].ptr<T>();
    auto* dst = out.ptr<uint8_t>();

    broadcast_walk_run<1>(out, inputs.data(), ctx,
                          [&](int64_t oo, const int64_t* off, const int64_t* step, int64_t run) {
                              const T* a_row = ap + off[0];
                              uint8_t* o_row = dst + oo;
                              for (int64_t i = 0; i < run; ++i) {
                                  o_row[i] = (a_row[i * step[0]] == 0) ? 1 : 0;
                              }
                          });
}

} // anonymous namespace

Status compare_cpu(const CompareAttributes& attrs, TensorView& output,
                   std::span<const TensorView> inputs, const ComputeContext& ctx,
                   void* /*workspace*/)
{
    if (!for_each_data_type(inputs[0].data_type(), [&](auto tag) {
            using T = typename decltype(tag)::type;
            compare_impl<T>(output, inputs, attrs.type, ctx);
        })) {
        return Status::unsupported;
    }
    return Status::ok;
}

Status logical_cpu(const LogicalAttributes& attrs, TensorView& output,
                   std::span<const TensorView> inputs, const ComputeContext& ctx,
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

} // namespace nnops::backend::cpu
