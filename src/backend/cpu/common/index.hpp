#pragma once
/// @file index.hpp
/// @brief Flat-index → strided-offset decomposition shared by the axis-wise
///        operators (Norm, Softmax, Reduce).
///
/// Several operators independently re-implement the same "odometer": turn a
/// row-major flat index into a multi-dimensional element offset using
/// `shape(d)` × `stride_elems(d)` over a contiguous run of dimensions,
/// optionally skipping one axis. This header is the single source of truth
/// for that pattern.

#include "nnops/core/tensor_view.hpp"

#include <cstdint>
#include <vector>

namespace nnops::backend::cpu {

/// Decompose a row-major flat index into a strided element offset.
///
/// `flat` is a linear index over the dimensions [dim_lo, dim_hi] (inclusive),
/// each `t.shape(d)` wide, row-major with `dim_hi` innermost. `skip_dim`
/// (default -1 = none) excludes one dimension from the walk — used by
/// axis-wise operators that flatten "all dims except `axis`".
///
/// Returns sum over d of coord(d) * t.stride_elems(d).
inline int64_t decompose_flat_offset(int64_t flat, const TensorView& t,
                                     int64_t dim_lo, int64_t dim_hi,
                                     int64_t skip_dim = -1)
{
    int64_t off = 0;
    int64_t rem = flat;
    for (int64_t d = dim_hi; d >= dim_lo; --d) {
        if (d == skip_dim) {
            continue;
        }
        const int64_t dim = t.shape(d);
        off += (rem % dim) * t.stride_elems(d);
        rem /= dim;
    }
    return off;
}

/// Build the offset table for `count` consecutive flat indices over
/// [dim_lo, dim_hi] (see decompose_flat_offset). Used by the general
/// (non-contiguous) paths to map inner flat positions to strided offsets once.
inline std::vector<int64_t> build_offsets(const TensorView& t,
                                          int64_t dim_lo, int64_t dim_hi,
                                          int64_t count)
{
    std::vector<int64_t> offs(static_cast<size_t>(count));
    for (int64_t flat = 0; flat < count; ++flat) {
        offs[static_cast<size_t>(flat)] =
            decompose_flat_offset(flat, t, dim_lo, dim_hi);
    }
    return offs;
}

}  // namespace nnops::backend::cpu
