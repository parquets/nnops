#pragma once
/// @file layout_convert.hpp
/// @brief Layout conversion utilities — pack/unpack between NCHW and NCHWC8.
///
/// Internal header — not part of the public API. Used by the operator
/// dispatch layer (src/ops/) and the LayoutDispatch utility.
///
/// Unified implementation: 2D (NCHW↔NCHWC8) and 3D (NCDHW↔NCDHWC8)
/// share a single kernel templated on direction. The only difference
/// is num_spatial_rows (H for 2D, D*H for 3D).
///
/// All functions support f32 and f16, with optional parallel dispatch
/// via ctx.cpu.run.

#include "nnops/core/tensor_view.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstddef>

namespace nnops {

// ============================================================
// 2D: NCHW <-> NCHWC8
// ============================================================

/// Pack NCHW → NCHWC8 (2D).
/// src: dense NCHW [N, C, H, W].
/// dst: pre-allocated NCHWC8 [N, ceil(C/8), H, W, 8].
void pack_nchw_to_nchwc8(const TensorView& src, TensorView& dst,
                          const ComputeContext& ctx = {});

/// Unpack NCHWC8 → NCHW (2D).
void unpack_nchwc8_to_nchw(const TensorView& src, TensorView& dst,
                            const ComputeContext& ctx = {});

// ============================================================
// 3D: NCDHW <-> NCDHWC8
// ============================================================

/// Pack NCDHW → NCDHWC8 (3D).
void pack_ncdhw_to_ncdhwc8(const TensorView& src, TensorView& dst,
                            const ComputeContext& ctx = {});

/// Unpack NCDHWC8 → NCDHW (3D).
void unpack_ncdhwc8_to_ncdhw(const TensorView& src, TensorView& dst,
                              const ComputeContext& ctx = {});

size_t nchwc8_storage_bytes(const TensorDesc& logical_desc,
                             int64_t alignment = 32);

}  // namespace nnops
