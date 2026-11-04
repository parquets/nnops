#pragma once
/// @file strided_walk.hpp
/// @brief The chunked logical-index odometer shared by the gather/scatter family.
///
/// Each of these ops is a permutation of element addresses: the output (or the
/// index space) is walked in logical row-major order and the body maps that
/// tuple to one or more storage offsets. Doing the mapping per element keeps one
/// implementation correct for any rank, any layout (stride_elems carries
/// channel-last) and a pitched view; the chunk start is decomposed once, so the
/// increment inside the loop is amortized O(1).

#include "nnops/core/compute_context.hpp"
#include "nnops/core/tensor_view.hpp"

#include <algorithm>
#include <cstdint>

namespace nnops::backend::cpu {

/// Logical element strides of a view, in elements.
inline void logical_strides(const TensorView& v, int64_t (&s)[8])
{
    for (int64_t d = 0; d < v.rank(); ++d) { s[d] = v.stride_elems(d); }
}

/// Call `body(idx)` for every logical index of `rank` dims, in row-major order,
/// chunked across the thread pool. Chunks are disjoint index ranges, so a body
/// that writes the output element(s) it visits needs no synchronization.
template <class F>
inline void walk_logical(const int64_t* dims, int64_t rank, int64_t numel,
                         const ComputeContext& ctx, F&& body)
{
    if (numel <= 0) { return; }

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

        for (int64_t n = begin; n < end; ++n) {
            body(idx);
            for (int64_t d = rank - 1; d >= 0; --d) {
                ++idx[d];
                if (idx[d] < dims[d]) { break; }
                idx[d] = 0;
            }
        }
    });
}

/// Linear offset of a logical index under `strides` for the first `n` axes.
inline int64_t flat_offset(const int64_t* idx, const int64_t* strides, int64_t n)
{
    int64_t off = 0;
    for (int64_t d = 0; d < n; ++d) { off += idx[d] * strides[d]; }
    return off;
}

}  // namespace nnops::backend::cpu
