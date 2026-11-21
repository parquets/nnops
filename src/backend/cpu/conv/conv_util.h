#pragma once
/// @file conv_util.h
/// @brief The two things every spatial conv route shares without owning: the
///        window it reads, and the way a plane is cut into tasks.
///
/// **The window.** Four families carry their own geometry struct (Conv2D's
/// im2col route, TransposeConv2D, DepthwiseTransposeConv2D, Conv3D /
/// TransposeConv3D), and the four declared the same eighteen axis members and
/// filled them with the same twenty assignments. They also differ in ways that
/// matter — groups and per-group channels for the dense routes, one channel for
/// the depthwise one, a third spatial axis for the 3-D routes — so the shared
/// part is the *axes* and not the whole shape:
///
///     struct ConvChannels : ConvAxes { int64_t N, IC, IC_per_G, OC_per_G, OC, G; };
///     struct DwtConvGeom  : ConvAxes { int64_t N, C; };
///
/// The grouped dense form is the same six members in every route that has them
/// (Conv3D, TransposeConv2D, TransposeConv3D), so it is one `ConvChannels`
/// here; the depthwise form has a channel count and no split, so it stays its
/// own struct. Field access is unchanged (`gm.IH`, `gm.KH`, …), so the route
/// bodies and the duck-typed chunk drivers in transpose_conv_common.h read the
/// derived structs exactly as before.
///
/// This is the *problem*, not the execution plan: the tile sizes, the task grid
/// and the scratch layout stay in each route's own plan struct
/// (Conv2DGemmPlan, Conv2DWinogradPlan, …), because those differ per algorithm
/// and a union of their knobs would be worse than four honest ones.
///
/// **The grid.** Pooling and DepthwiseConv cut a plane the same way, at either
/// spatial rank and in both arithmetic families: a task owns a strip of four
/// output rows (one row's window is a dependent chain; four are independent),
/// the strip is cut to fit L2 so every channel chunk sweeping it stays
/// resident, and when that grid cannot fill the pool the channel axis splits a
/// strip into chunks. The families declared the same arithmetic four times; it
/// lives here. These are execution heuristics, not geometry — the plan knobs
/// own their own routes.
///
/// **The rank lift.** Conv1D and Pool1D own no kernel: a rank-3 channel-last
/// signal lifts to the 2D kernels' [N, 1, L, C] view, so the gather, the
/// window walk and the im2col N axis all see the signal as one row.

#include "backend/cpu/simd_kernel/simd_col2im.hpp" // Tconv2dColGeom / Tconv3dColGeom
#include "nnops/core/epilogue.hpp"
#include "nnops/core/tensor_view.hpp"

#include <algorithm>
#include <cstdint>
#include <span>

namespace nnops::backend::cpu {

// =========================================================================
//  The window
// =========================================================================

/// The spatial window and epilogue flags of one convolution. A 2-D operator
/// leaves the D axis at its defaults (1 wide, no pad, unit stride/dilation).
struct ConvAxes {
    int64_t ID = 1, IH = 0, IW = 0;
    int64_t OD = 1, OH = 0, OW = 0;
    int64_t KD = 1, KH = 0, KW = 0;
    int64_t SD = 1, SH = 1, SW = 1;
    int64_t DD = 1, DH = 1, DW = 1;
    int64_t PD = 0, PH = 0, PW = 0;
    Epilogue epilogue;
    bool add_to = false;
    bool pre_act = false; ///< fold add_to's old value in before the epilogue

    /// Filter taps: one output value reduces KD·KH·KW input positions.
    int64_t karea() const noexcept { return KD * KH * KW; }
    /// Output positions one image produces.
    int64_t npix() const noexcept { return OD * OH * OW; }

    /// The col2im module's view of the H/W pair — shared by the 2-D scatter and
    /// (as the `.w` member) by the 3-D one.
    kernel::Tconv2dColGeom col_geom() const
    {
        return kernel::make_col_geom(IH, IW, OW, KH, KW, SH, SW, DH, DW, PH, PW);
    }

    /// The same view one axis deeper.
    kernel::Tconv3dColGeom col3_geom() const
    {
        return kernel::make_col3_geom(ID, IH, IW, OW, KD, KH, KW, SD, SH, SW, DD, DH, DW, PD, PH,
                                      PW);
    }
};

/// A grouped dense convolution's channel counts on top of the shared axes. The
/// routes differ only in where they read them from — Conv3D's filter is DHWIO
/// (OC last), the transposed forms lead with IC — so the members are shared and
/// each entry does its own read.
struct ConvChannels : ConvAxes {
    int64_t N = 0, IC = 0, IC_per_G = 0, OC_per_G = 0, OC = 0, G = 1;
};

/// Fill the axes from the operator's attributes and the input/output views.
/// @p Srank is the spatial rank: 2 for NHWC, 3 for NDHWC — the D axis then
/// takes the attribute arrays' first entry instead of its defaults.
template <int Srank, class Attrs>
inline void fill_conv_axes(ConvAxes& a, const Attrs& attrs, const TensorView& in,
                           const TensorView& out) noexcept
{
    static_assert(Srank == 2 || Srank == 3, "a spatial convolution is 2- or 3-D");
    // Channel-last: C is the last axis, W the one before it, H before that (and
    // D before that at rank 5) — whatever the spatial rank.
    const int64_t ir = in.rank();
    const int64_t orr = out.rank();
    a.IH = in.shape(ir - 3);
    a.IW = in.shape(ir - 2);
    a.OH = out.shape(orr - 3);
    a.OW = out.shape(orr - 2);
    if constexpr (Srank == 3) {
        a.ID = in.shape(ir - 4);
        a.OD = out.shape(orr - 4);
        a.KD = attrs.kernel_size[0];
        a.KH = attrs.kernel_size[1];
        a.KW = attrs.kernel_size[2];
        a.SD = attrs.stride[0];
        a.SH = attrs.stride[1];
        a.SW = attrs.stride[2];
        a.DD = attrs.dilation[0];
        a.DH = attrs.dilation[1];
        a.DW = attrs.dilation[2];
        a.PD = attrs.padding[0];
        a.PH = attrs.padding[1];
        a.PW = attrs.padding[2];
    } else {
        a.KH = attrs.kernel_size[0];
        a.KW = attrs.kernel_size[1];
        a.SH = attrs.stride[0];
        a.SW = attrs.stride[1];
        a.DH = attrs.dilation[0];
        a.DW = attrs.dilation[1];
        a.PH = attrs.padding[0];
        a.PW = attrs.padding[1];
    }
    a.epilogue = attrs.epilogue;
    a.add_to = attrs.add_to;
    if constexpr (requires { attrs.residual_pre_activation; }) {
        a.pre_act = attrs.add_to && attrs.residual_pre_activation;
    }
}

// =========================================================================
//  The grid
// =========================================================================

/// Width-strip cap: a four-row group's window is 4·S + (K-1)·D input rows of
/// IW·C elements, and every channel chunk walks them again, so the strip is cut
/// to keep that working set here.
inline constexpr int64_t kConvStripBytes = 128 * 1024;

/// Output columns per strip for a window @p kw wide stepping @p sw, from the
/// rows one four-row group reads (@p in_rows) and the plane's @p channels, in
/// elements of @p esz bytes. Rounded down to whole 16-lane blocks, floored at 16.
inline int64_t conv_strip_cols(int64_t in_rows, int64_t channels, int64_t esz, int64_t ow,
                               int64_t kw, int64_t dw, int64_t sw) noexcept
{
    const int64_t per_col = in_rows * channels * esz;
    const int64_t cols = per_col > 0 ? kConvStripBytes / per_col : ow;
    if (cols >= sw * ow) {
        return ow;
    }
    const int64_t w = (cols - (kw - 1) * dw) / sw;
    return std::clamp<int64_t>(w & ~int64_t{15}, 16, ow);
}

/// Channel chunks over the filter axis: the split that minimizes
/// waves × passes. A count the pool cannot divide evenly costs a whole extra
/// wave (21 tasks on 20 workers run in two), so the choice is by makespan, not
/// by pool width. One chunk whenever the grid already fills the pool.
inline int64_t conv_channel_chunks(int64_t groups, int64_t nblocks, int64_t workers) noexcept
{
    if (groups >= workers || nblocks <= 1) {
        return 1;
    }
    int64_t chunks = 1;
    int64_t best = 0;
    for (int64_t c = 1; c <= nblocks; ++c) {
        const int64_t passes = (nblocks + c - 1) / c;
        const int64_t waves = (groups * c + workers - 1) / workers;
        const int64_t cost = waves * passes;
        if (best == 0 || cost < best) {
            best = cost;
            chunks = c;
        }
    }
    return chunks;
}

/// Where flat task @p tid lands: the channel chunk is the fast axis, then the
/// width strip, the four-row group, the depth slice and the sample.
struct ConvGridPos {
    int64_t n = 0, od = 0, hg = 0, st = 0, cc = 0;
};

inline ConvGridPos conv_grid_pos(int64_t tid, int64_t strips, int64_t hgroups, int64_t od,
                                 int64_t chunks) noexcept
{
    ConvGridPos p;
    p.cc = tid % chunks;
    const int64_t g = tid / chunks;
    p.st = g % strips;
    p.hg = (g / strips) % hgroups;
    p.od = (g / (strips * hgroups)) % od;
    p.n = g / (strips * hgroups * od);
    return p;
}

// =========================================================================
//  The rank lift (Conv1D / Pool1D)
// =========================================================================

/// A view's data pointer as a writable handle; TensorView has no data().
inline void* view_data(const TensorView& v) noexcept
{
    return const_cast<void*>(static_cast<const void*>(v.ptr<uint8_t>()));
}

/// [N, L, C] → [N, 1, L, C], pitch and layout preserved. The 2D kernels read H
/// from axis 1 and W from axis 2, so the signal becomes one row; keeping the
/// caller's pitch is what keeps a sub-view that starts mid-row addressable.
inline TensorView lift_wide(const TensorView& v)
{
    const int64_t s[4] = {v.shape(0), 1, v.shape(1), v.shape(2)};
    return TensorView(std::span<const int64_t>(s, 4), v.data_type(), view_data(v), v.pitch(),
                      v.layout(), v.quant_params());
}

/// An NHWC conv filter (rank 3) → its 2D order:
/// HWIO [K, IC/G, OC] → [1, K, IC/G, OC].
///
/// The rank-3 row is IC/G·OC, which is not the rank-4 row (K·OC), so the lifted
/// view is built dense — a filter is addressed by its dims anyway, never by its
/// pitch.
inline TensorView lift_filter_wide(const TensorView& w)
{
    const int64_t s[4] = {1, w.shape(0), w.shape(1), w.shape(2)};
    return TensorView(std::span<const int64_t>(s, 4), w.data_type(), view_data(w), w.layout(),
                      w.quant_params());
}

} // namespace nnops::backend::cpu
