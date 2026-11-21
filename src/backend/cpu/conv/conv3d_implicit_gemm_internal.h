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
/// What they share is here: the geometry, the filter's B reading (DHWIO is
/// already the GEMM's k-major order — the difference per group is only what an
/// element *is*), the prepacked-filter blob, and the block decomposition the
/// col budget picks. The epilogue is where they part: the fp route scatters
/// with the shared epilogue, the quantized one hands the whole
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

/// True when the filter reads as one dense [IC/G·KD·KH·KW][OC/G] matrix per
/// group — the GEMM's B. DHWIO already stores a tap's channels-in/outputs as
/// one contiguous row, so the raw filter needs no repack: element
/// (tap, ic, oc) is at (tap·IC/G + ic)·OC + oc, which is exactly row
/// tap·IC/G + ic with a pitch of OC. A filter that departs from that (a padded
/// pitch, a foreign layout) has no B reading and takes the scalar gather.
inline bool conv3d_filter_is_matrix(const TensorView& w, const Conv3DGeom& gm) noexcept
{
    if (is_packed_conv3d_weight(w.layout())) {
        return true;
    }
    if (!is_channel_last_weight_3d(w.layout()) || w.rank() != 5) {
        return false;
    }
    return w.stride_elems(4) == 1 && w.stride_elems(3) == gm.OC &&
           w.stride_elems(2) == gm.IC_per_G * gm.OC;
}

/// One group's sub-blob in the prepacked filter (PackedDHWIO): MatMul's
/// whole-K PackedB panel set for an fp filter, the group's dense k-major
/// matrix for an s8 one.
struct Conv3DPackedWeight {
    int64_t group_elems = 0; ///< elements one group occupies
    int64_t groups = 0;
    int64_t total = 0;       ///< group_elems × groups
};

/// The blob's geometry. Empty for a dtype with no packed form, which is
/// anything but f32/f16/s8.
Conv3DPackedWeight conv3d_packed_weight_geom(int64_t karea, int64_t ic_per_group,
                                             int64_t oc_per_group, int64_t groups,
                                             DataType weight_dtype) noexcept;

/// The blob the filter prepacks into — the query half of the two-step pack,
/// what Conv3D::prepackWeights returns. Shape [OC, ⌈elems / OC⌉] with an
/// explicit pitch, like the other conv blobs; rank 0 for a filter with no
/// packed form.
TensorDesc conv3d_packed_weight_desc(const Conv3DAttributes& attrs, const TensorDesc& weight);

/// Fill @p packed, sized from conv3d_packed_weight_desc, from the raw filter.
void conv3d_packed_weight_pack(const Conv3DAttributes& attrs, const TensorView& weight,
                               TensorView& packed, const ComputeContext& ctx);

/// The s8 half (conv3d_implicit_gemm_i8.cpp): each group's dense matrix.
void conv3d_packed_weight_pack_i8(const Conv3DAttributes& attrs, const TensorView& weight,
                                  TensorView& packed, const ComputeContext& ctx);

/// Group @p g's B, [kdim][OC/G], from either the raw filter (a pitched view
/// into the caller's tensor — no copy) or the prepacked blob.
TensorView conv3d_weight_group(const TensorView& weight, const Conv3DGeom& gm, int64_t g);

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
