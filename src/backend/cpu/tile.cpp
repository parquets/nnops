/// @file tile.cpp
/// @brief CPU Tile / Expand — repeat axes by block copy.
///
/// Tile maps out index o to in index o % in_dim on every axis; Expand maps a
/// right-aligned o to the input index, 0 where the input axis is 1 or absent.
///
/// Both move a *run* of adjacent elements at a time: the trailing axes whose
/// extent agrees and whose step is unit on both sides are one memcpy, and only
/// the axes above the run go through an odometer. A [1,H,1,D] -> [B,H,S,D]
/// expand (repeat_kv) is one row copy per (b,h,s) instead of one element copy
/// with a modulo per axis, and a shape-preserving Tile is a single memcpy.
/// The walk is chunked across the pool.

#include "nnops/ops/tile.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "common/strided_walk.hpp"
#include "common/dtype_dispatch.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

namespace nnops::backend::cpu {

namespace {

/// Elements copied per task, at least one run.
constexpr int64_t kElementsPerTask = 1 << 16;

/// Shared driver: `body(out_off, in_off)` moves one run of `run` elements per
/// call, `in_off` coming from the per-axis source index the walk maintains.
template <typename T, class F>
void run_walk(const TensorView& input, TensorView& output, const ComputeContext& ctx, F&& body)
{
    const DenseTail tail = shared_dense_tail(input, output);
    const int64_t run = tail.run;
    const int64_t num_runs = output.numel() / run;
    if (num_runs <= 0) {
        return;
    }

    const int64_t rank = output.rank();
    const int64_t odim_end = tail.from_dim;
    const int64_t i_shift = rank - input.rank();

    int64_t odims[8], ostr[8], idims[8], istr[8];
    for (int64_t d = 0; d < rank; ++d) {
        odims[d] = output.shape(d);
        ostr[d] = output.stride_elems(d);
        const int64_t i = d - i_shift;
        idims[d] = (i < 0) ? 1 : input.shape(i);
        istr[d] = (i < 0 || idims[d] == 1) ? 0 : input.stride_elems(i);
    }

    const T* ip = input.ptr<T>();
    T* op = output.ptr<T>();
    const size_t run_bytes = static_cast<size_t>(run) * sizeof(T);

    const int64_t runs_per_task = std::max<int64_t>(1, kElementsPerTask / run);
    const int64_t tasks = (num_runs + runs_per_task - 1) / runs_per_task;

    ctx.cpu.run(0, tasks, [&](int64_t ti) {
        const int64_t begin = ti * runs_per_task;
        const int64_t end = std::min(begin + runs_per_task, num_runs);

        int64_t idx[8] = {};
        int64_t in_idx[8] = {};
        int64_t rem = begin;
        for (int64_t d = odim_end - 1; d >= 0; --d) {
            idx[d] = (odims[d] > 0) ? rem % odims[d] : 0;
            rem = (odims[d] > 0) ? rem / odims[d] : 0;
            in_idx[d] = (idims[d] > 0) ? idx[d] % idims[d] : 0;
        }

        int64_t oo = 0;
        int64_t io = 0;
        for (int64_t d = 0; d < odim_end; ++d) {
            oo += idx[d] * ostr[d];
            io += in_idx[d] * istr[d];
        }

        for (int64_t n = begin; n < end; ++n) {
            body(op + oo, ip + io, run_bytes);

            for (int64_t d = odim_end - 1; d >= 0; --d) {
                ++idx[d];
                oo += ostr[d];
                if (++in_idx[d] < idims[d]) {
                    io += istr[d];
                } else {
                    io -= (in_idx[d] - 1) * istr[d]; // wrapped back to coordinate 0
                    in_idx[d] = 0;
                }
                if (idx[d] < odims[d]) {
                    break;
                }
                idx[d] = 0;
                oo -= ostr[d] * odims[d];
                if (in_idx[d] != 0) {
                    io -= in_idx[d] * istr[d];
                    in_idx[d] = 0;
                }
            }
        }
    });
}

template <typename T>
void tile_impl(const TileAttributes& /*attrs*/, TensorView& output, const TensorView& input,
               const ComputeContext& ctx)
{
    run_walk<T>(input, output, ctx, [](T* dst, const T* src, size_t bytes) {
        std::memcpy(dst, src, bytes);
    });
}

template <typename T>
void expand_impl(const ExpandAttributes& /*attrs*/, TensorView& output, const TensorView& input,
                 const ComputeContext& ctx)
{
    run_walk<T>(input, output, ctx, [](T* dst, const T* src, size_t bytes) {
        std::memcpy(dst, src, bytes);
    });
}

} // anonymous namespace

Status tile_cpu(const TileAttributes& attrs, TensorView& output, std::span<const TensorView> inputs,
                const ComputeContext& ctx, void* /*workspace*/)
{
    for_each_data_type(inputs[0].data_type(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        tile_impl<T>(attrs, output, inputs[0], ctx);
    });
    return Status::ok;
}

Status expand_cpu(const ExpandAttributes& attrs, TensorView& output,
                  std::span<const TensorView> inputs, const ComputeContext& ctx,
                  void* /*workspace*/)
{
    for_each_data_type(inputs[0].data_type(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        expand_impl<T>(attrs, output, inputs[0], ctx);
    });
    return Status::ok;
}

} // namespace nnops::backend::cpu
