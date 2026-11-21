#pragma once
/// @file pooling_internal.h
/// @brief Internals shared by the two Pooling translation units.
///
/// Split one file per arithmetic family:
///
///   - pooling.cpp    — the float routes (spatial, global, adaptive), the Pool1D
///                      lift and the entry point;
///   - pooling_i8.cpp — quantized pooling (s8/u8 in and out), which runs the
///                      same four-row grid over dequantize/requantize kernels.
///
/// Both take the same PoolingAttributes; the quantized family has no
/// whole-plane form, so the driver turns a global request into the full-extent
/// window walk before calling in.
///
/// The extents and the window params are the same in both families (the
/// arithmetic, not the dtype, is what differs), so they are filled here.

#include "backend/cpu/simd_kernel/simd_pooling.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/ops/pooling.hpp"

#include <span>

namespace nnops::backend::cpu {

/// Channel-last pooling's extents: the spatial rank comes from the activation
/// (rank 4 or 5), and a 2-D call leaves the D axis at 1.
struct PoolShape {
    int64_t srank = 2, N = 0, C = 0;
    int64_t ID = 1, IH = 0, IW = 0;
    int64_t OD = 1, OH = 0, OW = 0;
};

inline PoolShape pool_shape(const TensorView& in, const TensorView& out) noexcept
{
    PoolShape s;
    const int64_t rank = in.rank();
    s.srank = PoolingAttributes::spatial_rank(rank);
    s.N = in.shape(0);
    s.C = in.shape(rank - 1);
    s.ID = (s.srank == 3) ? in.shape(1) : 1;
    s.IH = in.shape(s.srank - 1);
    s.IW = in.shape(s.srank);
    s.OD = (s.srank == 3) ? out.shape(1) : 1;
    s.OH = out.shape(s.srank - 1);
    s.OW = out.shape(s.srank);
    return s;
}

/// The window geometry both families walk. A 2-D call leaves the D axis at the
/// attribute defaults, which is what makes the per-axis spans carry a leading D
/// entry only at rank 5.
inline kernel::PoolingKernelParams pooling_params(const PoolingAttributes& attrs,
                                                  const PoolShape& s, const TensorView& in,
                                                  const TensorView& out) noexcept
{
    kernel::PoolingKernelParams p{};
    p.ID = s.ID;
    p.IH = s.IH;
    p.IW = s.IW;
    p.OH = s.OH;
    p.OW = s.OW;
    p.KD = (s.srank == 3) ? attrs.kernel_shape[0] : 1;
    p.KH = attrs.kernel_shape[1];
    p.KW = attrs.kernel_shape[2];
    p.SD = (s.srank == 3) ? attrs.stride[0] : 1;
    p.SH = attrs.stride[1];
    p.SW = attrs.stride[2];
    p.DD = (s.srank == 3) ? attrs.dilation[0] : 1;
    p.DH = attrs.dilation[1];
    p.DW = attrs.dilation[2];
    p.PD = (s.srank == 3) ? attrs.padding[0] : 0;
    p.PH = attrs.padding[1];
    p.PW = attrs.padding[2];
    p.out_row_stride = out.row_stride_elems();
    p.in_row_stride = in.row_stride_elems();
    p.in_d_stride = s.IH * p.in_row_stride;
    p.pix_stride = s.C;
    p.exclude_pad = attrs.exclude_pad;
    return p;
}

/// The quantized family's entry (pooling_i8.cpp): s8/u8 input, s8/u8 output.
void pooling_quant_run(const PoolingAttributes& attrs, TensorView& output,
                       std::span<const TensorView> inputs, const ComputeContext& ctx);

} // namespace nnops::backend::cpu
