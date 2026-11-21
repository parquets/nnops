#pragma once
/// @file broadcast_walk.hpp
/// @brief N-operand NumPy broadcast walks over one output index space.
///
/// Every operand is right-aligned to the output; a size-1 axis (or one the
/// operand does not have) has stride 0, so the same odometer serves all of them.
/// The chunk start is decomposed once into running offsets, and each step adds
/// the per-axis strides, so the inner loop has no division.
///
/// `broadcast_walk` walks one element per step (the general shape).
/// `broadcast_walk_run` walks one *contiguous output run* per step: every
/// operand's innermost stride is 0 (broadcast) or 1 (aligned), so the inner loop
/// can load whole vectors and the per-element odometer disappears. That is the
/// shape of every bias/mask broadcast a graph actually runs, and it is what
/// makes the trailing dimension vectorizable.

#include "nnops/core/compute_context.hpp"
#include "nnops/core/tensor_view.hpp"

#include <algorithm>
#include <cstdint>

namespace nnops::backend::cpu {

/// body(out_offset, in_offsets) for every output element.
template <int N, class F>
inline void broadcast_walk(const TensorView& out, const TensorView* ins, const ComputeContext& ctx,
                           F&& body)
{
    const int64_t rank = out.rank();
    const int64_t numel = out.numel();
    if (numel <= 0) {
        return;
    }

    int64_t dims[8], ostr[8];
    for (int64_t d = 0; d < rank; ++d) {
        dims[d] = out.shape(d);
        ostr[d] = out.stride_elems(d);
    }

    int64_t istr[N][8];
    for (int k = 0; k < N; ++k) {
        const int64_t r = ins[k].rank();
        for (int64_t d = 0; d < rank; ++d) {
            const int64_t i = d - (rank - r);
            istr[k][d] = (i < 0 || ins[k].shape(i) == 1) ? 0 : ins[k].stride_elems(i);
        }
    }

    constexpr int64_t kChunk = 1 << 16;
    const int64_t chunks = (numel + kChunk - 1) / kChunk;

    ctx.cpu.run(0, chunks, [&](int64_t ci) {
        const int64_t begin = ci * kChunk;
        const int64_t end = std::min(begin + kChunk, numel);

        int64_t idx[8] = {};
        int64_t rem = begin;
        for (int64_t d = rank - 1; d >= 0; --d) {
            idx[d] = (dims[d] > 0) ? rem % dims[d] : 0;
            rem = (dims[d] > 0) ? rem / dims[d] : 0;
        }

        int64_t off[N] = {};
        int64_t oo = 0;
        for (int64_t d = 0; d < rank; ++d) {
            for (int k = 0; k < N; ++k) {
                off[k] += idx[d] * istr[k][d];
            }
            oo += idx[d] * ostr[d];
        }

        for (int64_t n = begin; n < end; ++n) {
            body(oo, static_cast<const int64_t*>(off));

            for (int64_t d = rank - 1; d >= 0; --d) {
                ++idx[d];
                for (int k = 0; k < N; ++k) {
                    off[k] += istr[k][d];
                }
                oo += ostr[d];
                if (idx[d] < dims[d]) {
                    break;
                }
                idx[d] = 0;
                for (int k = 0; k < N; ++k) {
                    off[k] -= istr[k][d] * dims[d];
                }
                oo -= ostr[d] * dims[d];
            }
        }
    });
}

/// Run-walking broadcast: `body(out_off, in_offs, in_steps, run)` once per
/// contiguous output run, `in_steps[k]` being 0 (the operand broadcasts along
/// the run) or 1 (it advances with it).
template <int N, class F>
inline void broadcast_walk_run(const TensorView& out, const TensorView* ins,
                               const ComputeContext& ctx, F&& body)
{
    const int64_t rank = out.rank();
    const int64_t numel = out.numel();
    if (numel <= 0) {
        return;
    }

    // The run is the innermost axis when it is one contiguous physical step; a
    // view whose last axis steps by more than one falls back to one element per
    // step, with that axis folded into the walked index space.
    const bool tail_unit = (rank >= 1) && (out.stride_elems(rank - 1) == 1);
    const int64_t run = tail_unit ? out.shape(rank - 1) : 1;
    const int64_t odim_end = tail_unit ? rank - 1 : rank;
    const int64_t num_outer = (run > 0) ? numel / run : 0;
    if (num_outer <= 0) {
        return;
    }

    int64_t odims[8], ostr[8];
    for (int64_t d = 0; d < rank; ++d) {
        odims[d] = out.shape(d);
        ostr[d] = out.stride_elems(d);
    }

    int64_t istr[N][8];
    int64_t istep[N];
    for (int k = 0; k < N; ++k) {
        const int64_t r = ins[k].rank();
        for (int64_t d = 0; d < rank; ++d) {
            const int64_t i = d - (rank - r);
            istr[k][d] = (i < 0 || ins[k].shape(i) == 1) ? 0 : ins[k].stride_elems(i);
        }
        // Right-aligned, so the output's innermost axis maps to the operand's
        // innermost one whenever the operand has it — stride 1 in either layout.
        istep[k] = tail_unit ? istr[k][rank - 1] : 0;
    }

    // One task per ~64k elements of the run walk (at least one run).
    const int64_t rows_per_task = std::max<int64_t>(1, std::min<int64_t>(256, (1 << 16) / run));
    const int64_t tasks = (num_outer + rows_per_task - 1) / rows_per_task;

    ctx.cpu.run(0, tasks, [&](int64_t ti) {
        const int64_t begin = ti * rows_per_task;
        const int64_t end = std::min(begin + rows_per_task, num_outer);

        int64_t idx[8] = {};
        int64_t rem = begin;
        for (int64_t d = odim_end - 1; d >= 0; --d) {
            idx[d] = (odims[d] > 0) ? rem % odims[d] : 0;
            rem = (odims[d] > 0) ? rem / odims[d] : 0;
        }

        int64_t off[N] = {};
        int64_t oo = 0;
        for (int64_t d = 0; d < odim_end; ++d) {
            for (int k = 0; k < N; ++k) {
                off[k] += idx[d] * istr[k][d];
            }
            oo += idx[d] * ostr[d];
        }

        for (int64_t n = begin; n < end; ++n) {
            body(oo, static_cast<const int64_t*>(off), static_cast<const int64_t*>(istep), run);

            for (int64_t d = odim_end - 1; d >= 0; --d) {
                ++idx[d];
                for (int k = 0; k < N; ++k) {
                    off[k] += istr[k][d];
                }
                oo += ostr[d];
                if (idx[d] < odims[d]) {
                    break;
                }
                idx[d] = 0;
                for (int k = 0; k < N; ++k) {
                    off[k] -= istr[k][d] * odims[d];
                }
                oo -= ostr[d] * odims[d];
            }
        }
    });
}

} // namespace nnops::backend::cpu
