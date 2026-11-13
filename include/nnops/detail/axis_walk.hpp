#pragma once
/// @file axis_walk.hpp
/// @brief Layout-free striding for a one-axis reduction.
///
/// The walk wants one thing from the view: the element stride of every dim.
/// Nothing else about the layout enters it — planar and channel-last are the
/// same problem, and so is a pitched row — so each side of the reduced axis
/// reduces to either a single flat step, when the dims nest dense, or one
/// (extent, step) term per dim that a mixed-radix decode walks. Which of the
/// two a side gets is decided by the strides, never by a layout flag.

#include "nnops/core/tensor_view.hpp"

#ifdef __CUDACC__
#define NNOPS_AXIS_HD __host__ __device__
#else
#define NNOPS_AXIS_HD
#endif

namespace nnops::detail {

/// The dims on one side of the axis, innermost dim first. `n == 0` is the dense
/// case: the flat index times one step is the whole offset.
struct AxisGroup {
    int64_t flat_step = 0;
    int n = 0;
    int64_t ext[TensorView::kMaxRank] = {};
    int64_t step[TensorView::kMaxRank] = {};
};

struct AxisWalk {
    int64_t axis_dim = 0;
    int64_t axis_stride = 0;
    int64_t outer = 0;       ///< product of the dims before the axis
    int64_t inner = 0;       ///< product of the dims after it
    AxisGroup pre;           ///< input dims before the axis
    AxisGroup post;          ///< input dims after it
    AxisGroup out_pre;       ///< the same two for the output
    AxisGroup out_post;
};

/// Offset of flat index `t` inside its group.
NNOPS_AXIS_HD inline int64_t axis_group_offset(int64_t t, const AxisGroup& g)
{
    if (g.n == 0) {
        return t * g.flat_step;
    }
    int64_t off = 0;
    for (int i = 0; i < g.n; ++i) {
        const int64_t q = t / g.ext[i];
        off += (t - q * g.ext[i]) * g.step[i];
        t = q;
    }
    return off;
}

/// True when every dim in [lo, hi) steps by the extent of the dim inside it, so
/// the group flattens to `base_step`.
inline bool axis_group_dense(const TensorView& v, int64_t lo, int64_t hi,
                             int64_t base_step)
{
    int64_t acc = base_step;
    for (int64_t d = hi - 2; d >= lo; --d) {
        acc *= v.shape(d + 1);
        if (v.stride_elems(d) != acc) {
            return false;
        }
    }
    return true;
}

/// Build one side. `base_step` is the step of the innermost dim, and `dense`
/// says whether the flat form is exact; the caller has already checked the
/// dims outside [lo, hi) that the flat form also has to agree with.
inline void axis_make_group(AxisGroup& g, const TensorView& v, int64_t lo,
                            int64_t hi, int64_t base_step, bool dense)
{
    g.n = 0;
    if (dense) {
        g.flat_step = base_step;
        return;
    }
    g.flat_step = 0;
    for (int64_t d = hi - 1; d >= lo; --d) {
        if (v.shape(d) == 1) {
            continue;
        }
        g.ext[g.n] = v.shape(d);
        g.step[g.n] = v.stride_elems(d);
        ++g.n;
    }
}

/// Build one side of a view whose dims [lo, hi) sit on that side of the axis.
/// `inner_step` is the step the flat form assumes for the innermost dim.
inline void axis_fill_group(AxisGroup& g, const TensorView& v, int64_t lo,
                            int64_t hi, int64_t inner_step)
{
    if (hi <= lo) {
        axis_make_group(g, v, lo, hi, 0, true);
        return;
    }
    const bool dense = axis_group_dense(v, lo, hi, inner_step);
    axis_make_group(g, v, lo, hi, inner_step, dense);
}

/// Build a group whose flat form assumes a unit innermost step — the trailing
/// dims, which only nest densely when the innermost one is one element wide.
inline void axis_fill_group_unit(AxisGroup& g, const TensorView& v, int64_t lo,
                                 int64_t hi)
{
    if (hi <= lo) {
        axis_make_group(g, v, lo, hi, 0, true);
        return;
    }
    const bool dense = v.stride_elems(hi - 1) == 1 &&
                       axis_group_dense(v, lo, hi, 1);
    axis_make_group(g, v, lo, hi, 1, dense);
}

/// The geometry of one call. The axis is resolved (negative wrapped) by the
/// caller; `oax` is where the axis sits in the output, which drops it unless
/// keepdims holds it at one.
inline AxisWalk make_axis_walk(const TensorView& in, const TensorView& out,
                               int64_t ax, bool keepdims)
{
    AxisWalk w;
    const int64_t rank = in.rank();
    w.axis_dim = in.shape(ax);
    w.axis_stride = in.stride_elems(ax);
    w.outer = 1;
    for (int64_t d = 0; d < ax; ++d) {
        w.outer *= in.shape(d);
    }
    w.inner = 1;
    for (int64_t d = ax + 1; d < rank; ++d) {
        w.inner *= in.shape(d);
    }

    // Before the axis the flat step is the axis's own row, stride(ax - 1).
    if (ax > 0) {
        axis_fill_group(w.pre, in, 0, ax, in.stride_elems(ax - 1));
    } else {
        axis_fill_group(w.pre, in, 0, 0, 0);
    }
    // After it the flat step is one element, so the tail must also be dense.
    axis_fill_group_unit(w.post, in, ax + 1, rank);

    const int64_t orank = out.rank();
    if (ax > 0) {
        axis_fill_group(w.out_pre, out, 0, ax, out.stride_elems(ax - 1));
    } else {
        axis_fill_group(w.out_pre, out, 0, 0, 0);
    }
    const int64_t post_lo = keepdims ? ax + 1 : ax;
    axis_fill_group_unit(w.out_post, out, post_lo, orank);
    return w;
}

}  // namespace nnops::detail

#undef NNOPS_AXIS_HD
