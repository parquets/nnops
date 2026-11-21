/// @file split.cpp
/// @brief CPU Split — one block copy per output position.
///
/// Output j covers the axis range [off, off + len) and adds off to the axis
/// coordinate of every element. The axes after the split axis are one contiguous
/// run on both sides (the value head of a split QKV is a row copy), so the walk
/// moves a run per step and only the axes up to and including the split axis go
/// through the odometer. The outputs run one after another, each with the full
/// thread pool.

#include "nnops/ops/split.hpp"
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
        const int64_t len = attrs.split.empty() ? dim / n : attrs.split[static_cast<size_t>(j)];

        // The run is the trailing set of axes whose element stride agrees on
        // both sides — the extents need not, because a slice of a trailing axis
        // (a C split) stays contiguous and is still one copy per pixel. The
        // axes above the run are the ones the odometer walks.
        int64_t run = 1;
        int64_t from_dim = input.rank();
        for (int64_t d = input.rank() - 1; d >= 0; --d) {
            if (out.shape(d) == 1) {
                continue;
            }
            if (out.stride_elems(d) != run || input.stride_elems(d) != run) {
                break;
            }
            if (input.shape(d) < out.shape(d)) {
                break;
            }
            run *= out.shape(d);
            from_dim = d;
        }
        const int64_t steps = out.numel() / run;
        if (steps <= 0) {
            off += len;
            continue;
        }

        const int64_t wnd = from_dim;
        int64_t wdims[8], wistr[8], wostr[8];
        for (int64_t d = 0; d < wnd; ++d) {
            wdims[d] = out.shape(d);
            wistr[d] = input.stride_elems(d);
            wostr[d] = out.stride_elems(d);
        }

        T* op = out.ptr<T>();
        const int64_t base = off * istr[axis];
        const size_t run_bytes = static_cast<size_t>(run) * sizeof(T);

        const int64_t steps_per_task = std::max<int64_t>(1, (int64_t{1} << 16) / run);
        const int64_t tasks = (steps + steps_per_task - 1) / steps_per_task;

        ctx.cpu.run(0, tasks, [&](int64_t ti) {
            const int64_t begin = ti * steps_per_task;
            const int64_t end = std::min(begin + steps_per_task, steps);

            int64_t idx[8] = {};
            int64_t rem = begin;
            for (int64_t d = wnd - 1; d >= 0; --d) {
                idx[d] = (wdims[d] > 0) ? rem % wdims[d] : 0;
                rem = (wdims[d] > 0) ? rem / wdims[d] : 0;
            }

            int64_t io = base, oo = 0;
            for (int64_t d = 0; d < wnd; ++d) {
                io += idx[d] * wistr[d];
                oo += idx[d] * wostr[d];
            }

            for (int64_t s = begin; s < end; ++s) {
                std::memcpy(op + oo, ip + io, run_bytes);

                for (int64_t d = wnd - 1; d >= 0; --d) {
                    ++idx[d];
                    io += wistr[d];
                    oo += wostr[d];
                    if (idx[d] < wdims[d]) {
                        break;
                    }
                    idx[d] = 0;
                    io -= wistr[d] * wdims[d];
                    oo -= wostr[d] * wdims[d];
                }
            }
        });

        off += len;
    }
}

} // anonymous namespace

Status split_cpu(const SplitAttributes& attrs, std::span<TensorView> outputs,
                 std::span<const TensorView> inputs, const ComputeContext& ctx, void* /*workspace*/)
{
    for_each_data_type(inputs[0].data_type(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        split_impl<T>(inputs[0], outputs, attrs, ctx);
    });
    return Status::ok;
}

} // namespace nnops::backend::cpu
