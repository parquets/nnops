/// @file conv2d_pack.h
/// @brief The CUDA int8 conv's prepacked filter: its descriptor query and its
///        pack pass, the pair Conv2D::prepackWeights drives.
///
/// The blob is PackedHWIOCuda — per group, one 64-row by 64-byte panel per (k
/// tile, oc block), rows `LD = BK + 16` bytes apart, so a row of it is exactly
/// the row the shared-memory panel wants and the fill is a 16 B copy per thread.
/// The tail rows (an oc block past OC/G) and the tail k of the last tile are
/// written as zeros, which is what the panel would have gathered anyway, so the
/// kernel needs no predicate on either axis. After the panels comes the
/// per-channel filter sum `Σ_k W[oc, k]` in `[group][OC/G]` order: the raw
/// route recomputes those on the device on every call, and the requant's
/// zero-point compensation reads them.
///
/// Two calls, as the operator's contract wants — an empty output is the query,
/// and `desc().numel()` is the allocation. Only this route reads the blob, so
/// both functions answer an unsupported geometry with a rank-0 descriptor and a
/// refused status rather than a blob some other route would misread.

#pragma once

#include "nnops/core/compute_context.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/ops/conv2d.hpp"

namespace nnops::backend::cuda {

/// The blob's descriptor for this filter, or a rank-0 one when the geometry has
/// no blob (anything but an s8 HWIO filter on the implicit-GEMM route).
TensorDesc conv_i8_packed_weight_desc(const Conv2DAttributes& attrs, const TensorDesc& weight);

/// Fill `out`, a buffer from the query above, from the raw filter.
Status conv_i8_packed_weight_pack(const Conv2DAttributes& attrs, const TensorView& weight,
                                  TensorView& out, const ComputeContext& ctx);

/// The geometry both functions agree on, and the kernel reads: the panel row
/// stride and the tile the panels are cut on. Compile-time here so the pack pass
/// and the kernel cannot drift apart.
constexpr int kConvI8PackBN = 64; ///< oc rows per panel
constexpr int kConvI8PackBK = 64; ///< k bytes per panel
constexpr int kConvI8PackLD = 80; ///< panel row stride in bytes (BK + the pad)

} // namespace nnops::backend::cuda
