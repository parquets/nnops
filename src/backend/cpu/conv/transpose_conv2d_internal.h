#pragma once
/// @file transpose_conv2d_internal.h
/// @brief Internals shared by the two TransposeConv2D translation units.
///
/// The operator is split one file per arithmetic family, the split Conv2D and
/// MatMul use:
///
///   - transpose_conv2d.cpp    — the float routes (pixel gather, col-matrix
///                               GEMM, depthwise) and the entry point;
///   - i8transpose_conv2d.cpp  — the quantized routes (s8 weight, u8/s8/f32/f16
///                               activation, int32 accumulate) and their
///                               dequant/requant merges.
///
/// The geometry is one struct for both: the routes read the same numbers and
/// only disagree on what an element is.

#include "backend/cpu/conv/tconv_common.h"
#include "backend/cpu/simd_kernel/simd_col2im.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/ops/transpose_conv2d.hpp"

#include <cstdint>
#include <span>

namespace nnops::backend::cpu {

/// One transposed convolution's geometry, read off the tensors once.
struct Geom {
    int64_t N, IC, IH, IW;
    int64_t OC_per_G, IC_per_G, OC, G;
    int64_t KH, KW, OH, OW;
    int64_t SH, SW, DH, DW, PH, PW;
    Epilogue epilogue;
    bool add_to;
    bool pre_act; ///< fold add_to's old value in before the epilogue

    /// The col2im module's view of the same numbers.
    Tconv2dColGeom col_geom() const
    {
        return make_col_geom(IH, IW, OW, KH, KW, SH, SW, DH, DW, PH, PW);
    }
};

/// The quantized family's entry (i8transpose_conv2d.cpp): s8 weights only; the
/// activation ladder and the output ladder are the route's own.
Status transpose_conv2d_int8_cpu(const TransposeConv2DAttributes& attrs, const Geom& gm,
                                 TensorView& output, std::span<const TensorView> inputs,
                                 const ComputeContext& ctx);

} // namespace nnops::backend::cpu
