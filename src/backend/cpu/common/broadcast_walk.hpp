#pragma once
/// @file broadcast_walk.hpp
/// @brief N-operand NumPy broadcast walk over one output index space.
///
/// Every operand is right-aligned to the output; a size-1 axis (or one the
/// operand does not have) has stride 0, so the same odometer serves all of them.
/// The chunk start is decomposed once into running offsets, and each step adds
/// the per-axis strides, so the inner loop has no division.

#include "nnops/core/compute_context.hpp"
#include "nnops/core/tensor_view.hpp"

#include <algorithm>
#include <cstdint>

namespace nnops::backend::cpu {

/// body(out_offset, in_offsets) for every output element.
template <int N, class F>
inline void broadcast_walk(const TensorView& out, const TensorView* ins,
                           const ComputeContext& ctx, F&& body)
{
    const int64_t rank  = out.rank();
    const int64_t numel = out.numel();
    if (numel <= 0) { return; }

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
            istr[k][d] = (i < 0 || ins[k].shape(i) == 1)
                ? 0 : ins[k].stride_elems(i);
        }
    }

    constexpr int64_t kChunk = 1 << 16;
    const int64_t chunks = (numel + kChunk - 1) / kChunk;

    ctx.cpu.run(0, chunks, [&](int64_t ci) {
        const int64_t begin = ci * kChunk;
        const int64_t end   = std::min(begin + kChunk, numel);

        int64_t idx[8] = {};
        int64_t rem = begin;
        for (int64_t d = rank - 1; d >= 0; --d) {
            idx[d] = (dims[d] > 0) ? rem % dims[d] : 0;
            rem    = (dims[d] > 0) ? rem / dims[d] : 0;
        }

        int64_t off[N] = {};
        int64_t oo = 0;
        for (int64_t d = 0; d < rank; ++d) {
            for (int k = 0; k < N; ++k) { off[k] += idx[d] * istr[k][d]; }
            oo += idx[d] * ostr[d];
        }

        for (int64_t n = begin; n < end; ++n) {
            body(oo, static_cast<const int64_t*>(off));

            for (int64_t d = rank - 1; d >= 0; --d) {
                ++idx[d];
                for (int k = 0; k < N; ++k) { off[k] += istr[k][d]; }
                oo += ostr[d];
                if (idx[d] < dims[d]) { break; }
                idx[d] = 0;
                for (int k = 0; k < N; ++k) { off[k] -= istr[k][d] * dims[d]; }
                oo -= ostr[d] * dims[d];
            }
        }
    });
}

}  // namespace nnops::backend::cpu
