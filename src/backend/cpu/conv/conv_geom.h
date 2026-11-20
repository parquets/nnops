#pragma once
/// @file conv_geom.h
/// @brief The window one convolution reads — the axis block every spatial conv
/// family shares.
///
/// Four families carry their own geometry struct (Conv2D's im2col route,
/// TransposeConv2D, DepthwiseTransposeConv2D, Conv3D / TransposeConv3D), and
/// the four declared the same eighteen axis members and filled them with the
/// same twenty assignments. They also differ in ways that matter — groups and
/// per-group channels for the dense routes, one channel for the depthwise one,
/// a third spatial axis for the 3-D routes — so the shared part is the *axes*
/// and not the whole shape:
///
///     struct Geom : ConvAxes { int64_t N, IC, IC_per_G, OC_per_G, OC, G; };
///     struct DwtConvGeom : ConvAxes { int64_t N, C; };
///
/// Field access is unchanged (`gm.IH`, `gm.KH`, …), so the route bodies and the
/// duck-typed chunk drivers in transpose_conv_common.h read the derived structs
/// exactly as before.
///
/// This is the *problem*, not the execution plan: the tile sizes, the task grid
/// and the scratch layout stay in each route's own plan struct
/// (Conv2DGemmPlan, Conv2DWinogradPlan, …), because those differ per algorithm
/// and a union of their knobs would be worse than four honest ones.

#include "backend/cpu/simd_kernel/simd_col2im.hpp" // Tconv2dColGeom / Tconv3dColGeom
#include "nnops/core/epilogue.hpp"
#include "nnops/core/tensor_view.hpp"

#include <cstdint>

namespace nnops::backend::cpu {

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

} // namespace nnops::backend::cpu
