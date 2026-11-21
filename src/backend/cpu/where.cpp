/// @file where.cpp
/// @brief CPU Where — a three-operand broadcast select.
///
/// The condition is read as raw bytes: a signed byte is nonzero exactly when
/// its bits are, so s8 and u8 share one test. Only the two branches are typed.
///
/// The walk hands out one contiguous output run at a time, so a broadcast
/// condition (a mask row) or a broadcast branch is one load per run instead of
/// one per element.

#include "nnops/ops/where.hpp"
#include "nnops/detail/assert.hpp"
#include "common/broadcast_walk.hpp"
#include "common/dtype_dispatch.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <cstdint>
#include <type_traits>
#include <utility>

namespace nnops::backend::cpu {

namespace {

template <typename T>
void where_impl(TensorView& out, std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto* cond = inputs[0].ptr<uint8_t>();
    const auto* xp = inputs[1].ptr<T>();
    const auto* yp = inputs[2].ptr<T>();
    auto* dst = out.ptr<T>();

    // One contiguous run per body call: the select itself stays scalar (the
    // abstraction carries no blend), but the per-element odometer is gone and
    // the branch operands are read at a stride the compiler can see.
    broadcast_walk_run<3>(out, inputs.data(), ctx,
                          [&](int64_t oo, const int64_t* off, const int64_t* step, int64_t run) {
                              const uint8_t* c_row = cond + off[0];
                              const T* x_row = xp + off[1];
                              const T* y_row = yp + off[2];
                              T* o_row = dst + oo;
                              for (int64_t i = 0; i < run; ++i) {
                                  o_row[i] = c_row[i * step[0]] ? x_row[i * step[1]] :
                                                                  y_row[i * step[2]];
                              }
                          });
}

} // anonymous namespace

Status where_cpu(const WhereAttributes&, TensorView& output, std::span<const TensorView> inputs,
                 const ComputeContext& ctx, void* /*workspace*/)
{
    if (!for_each_data_type(inputs[1].data_type(), [&](auto tag) {
            using T = typename decltype(tag)::type;
            where_impl<T>(output, inputs, ctx);
        })) {
        return Status::unsupported;
    }
    return Status::ok;
}

} // namespace nnops::backend::cpu
