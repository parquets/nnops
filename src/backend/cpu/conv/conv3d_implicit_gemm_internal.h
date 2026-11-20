#pragma once
/// @file conv3d_implicit_gemm_internal.h
/// @brief Internals shared by the two Conv3D translation units.
///
/// Conv3D is im2col → GEMM at both ranks of "dtype", split one file per
/// arithmetic family as Conv2D is:
///
///   - conv3d_implicit_gemm.cpp    — the float route (f32/f16), its scalar
///                                   gather fallback and the entry point;
///   - conv3d_implicit_gemm_i8.cpp — the quantized route: an s8 filter against
///                                   a pre-quantized u8/s8 (or per-call
///                                   quantized fp) activation, int32 accumulate.
///
/// What they share is here: the geometry, the filter repack (identical bytes
/// either way — the difference is only what an element *is*), and the block
/// decomposition the col budget picks. The epilogue is where they part: the fp
/// route scatters with the shared epilogue, the quantized one hands the whole
/// dequantize/requantize to MatMul's output pass.

#include "backend/cpu/conv/conv_geom.h"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/ops/conv3d.hpp"

#include <algorithm>
#include <cstdint>
#include <span>

namespace nnops::backend::cpu {

/// One convolution's geometry, read off the tensors once. The axes, the
/// epilogue flags and karea()/npix() are conv_geom.h's; the channels are this
/// operator's.
struct Conv3DGeom : ConvAxes {
    int64_t N = 0, IC = 0, IC_per_G = 0, OC_per_G = 0, OC = 0, G = 1;
};

/// Repack one group's filter [OC/G][IC/G][KD·KH·KW] into the GEMM's B,
/// [tap·ic + ic][oc]. Disjoint writes per output channel, so the split is free.
template <typename T>
void conv3d_pack_weight(const T* w, int64_t oc_per_g, int64_t ic_per_g, int64_t karea, T* out,
                        const ComputeContext& ctx)
{
    ctx.cpu.run(0, oc_per_g, [&](int64_t oc) {
        const T* src = w + oc * ic_per_g * karea;
        for (int64_t ic = 0; ic < ic_per_g; ++ic) {
            for (int64_t tap = 0; tap < karea; ++tap) {
                out[(tap * ic_per_g + ic) * oc_per_g + oc] = src[ic * karea + tap];
            }
        }
    });
}

/// The pixel block a col budget allows: whole depth slices while they fit,
/// otherwise a run of rows inside one slice (the col's row stride is the
/// block's own volume, so the rectangle is the unit).
struct Conv3DBlock {
    int64_t od_chunk = 1;
    int64_t roi_h = 1;
};

inline Conv3DBlock conv3d_choose_block(const Conv3DGeom& g, int64_t kdim, int64_t esz,
                                       int64_t budget_bytes) noexcept
{
    const int64_t cap = std::max<int64_t>(1, budget_bytes / std::max<int64_t>(esz, 1));
    const int64_t slice = g.OH * g.OW;
    Conv3DBlock b;
    b.roi_h = g.OH;
    if (kdim <= 0 || slice <= 0) {
        return b;
    }
    if (kdim * slice <= cap) {
        b.od_chunk = std::max<int64_t>(1, std::min(g.OD, cap / (kdim * slice)));
    } else {
        b.roi_h = std::max<int64_t>(1, std::min(g.OH, cap / (kdim * g.OW)));
    }
    return b;
}

/// The quantized family's entry (conv3d_implicit_gemm_i8.cpp).
Status conv3d_i8_cpu(const Conv3DAttributes& attrs, TensorView& output,
                     std::span<const TensorView> inputs, const ComputeContext& ctx);

} // namespace nnops::backend::cpu
