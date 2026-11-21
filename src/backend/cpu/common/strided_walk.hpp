#pragma once
/// @file strided_walk.hpp
/// @brief The chunked logical-index odometer shared by the gather/scatter family.
///
/// Each of these ops is a permutation of element addresses: the output (or the
/// index space) is walked in logical row-major order and the body maps that
/// tuple to one or more storage offsets. Doing the mapping per element keeps one
/// implementation correct for any rank, any layout (stride_elems carries
/// NHWC) and a pitched view; the chunk start is decomposed once, so the
/// increment inside the loop is amortized O(1).

#include "nnops/core/compute_context.hpp"
#include "nnops/core/tensor_view.hpp"

#include <algorithm>
#include <cstdint>

namespace nnops::backend::cpu {

/// Logical element strides of a view, in elements.
inline void logical_strides(const TensorView& v, int64_t (&s)[8])
{
    for (int64_t d = 0; d < v.rank(); ++d) {
        s[d] = v.stride_elems(d);
    }
}

/// Call `body(idx)` for every logical index of `rank` dims, in row-major order,
/// chunked across the thread pool. Chunks are disjoint index ranges, so a body
/// that writes the output element(s) it visits needs no synchronization.
template <class F>
inline void walk_logical(const int64_t* dims, int64_t rank, int64_t numel,
                         const ComputeContext& ctx, F&& body)
{
    if (numel <= 0) {
        return;
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

        for (int64_t n = begin; n < end; ++n) {
            body(idx);
            for (int64_t d = rank - 1; d >= 0; --d) {
                ++idx[d];
                if (idx[d] < dims[d]) {
                    break;
                }
                idx[d] = 0;
            }
        }
    });
}

/// Linear offset of a logical index under `strides` for the first `n` axes.
inline int64_t flat_offset(const int64_t* idx, const int64_t* strides, int64_t n)
{
    int64_t off = 0;
    for (int64_t d = 0; d < n; ++d) {
        off += idx[d] * strides[d];
    }
    return off;
}

/// The densest trailing run two views can move with one memcpy: the axes from
/// `from_dim` to rank-1 agree in extent and are physically contiguous in both,
/// so a run moves as `run` adjacent elements. Everything before `from_dim` is
/// walked by an odometer.
struct DenseTail {
    int64_t run = 1;      ///< elements in one run
    int64_t from_dim = 0; ///< first axis of the run (== rank when run is 1)
};

inline DenseTail shared_dense_tail(const TensorView& a, const TensorView& b)
{
    const int64_t ra = a.rank();
    const int64_t rb = b.rank();
    const int64_t rank = (ra > rb) ? ra : rb;
    DenseTail t{1, rank};
    for (int64_t d = rank - 1; d >= 0; --d) {
        // Right-aligned: an axis the view does not have is a size-1 axis, so it
        // contributes one element and no address step.
        const int64_t ia = d - (rank - ra);
        const int64_t ib = d - (rank - rb);
        const int64_t dim_a = (ia < 0) ? 1 : a.shape(ia);
        const int64_t dim_b = (ib < 0) ? 1 : b.shape(ib);
        if (dim_a != dim_b) {
            break;
        }
        if (dim_a == 1) {
            continue;
        }
        // Both sides step by exactly the run length: contiguous, and with no
        // row pitch between them.
        const int64_t str_a = (ia < 0) ? 0 : a.stride_elems(ia);
        const int64_t str_b = (ib < 0) ? 0 : b.stride_elems(ib);
        if (str_a != t.run || str_b != t.run) {
            break;
        }
        t.run *= dim_a;
        t.from_dim = d;
    }
    return t;
}

} // namespace nnops::backend::cpu
