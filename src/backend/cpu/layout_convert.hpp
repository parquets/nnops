#pragma once
/// @file layout_convert.hpp
/// @brief Internal NCHW↔NCHWC8 (2D) and NCDHW↔NCDHWC8 (3D) pack/unpack.
///
/// Not part of the public API — used by the operator dispatch layer (src/ops/)
/// and LayoutDispatch. 2D and 3D run the same pack and unpack kernels; only
/// num_spatial_rows differs (H vs D*H). f32 and f16 go through an 8×8 SIMD
/// transpose; s8/u8 go through a byte copy. A partial trailing C8 block is
/// zero-padded on pack for the fp path, and padded with the source zero_point
/// for s8/u8 so the extra lanes dequantize to nothing; unpack drops it. Every
/// row is an independent task on ctx.cpu.run.

#include "nnops/core/tensor_view.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstddef>

namespace nnops {

// ============================================================
// 2D: NCHW <-> NCHWC8
// ============================================================

/// Dst must be pre-allocated as NCHWC8 [N, ceil(C/8), H, W, 8].
void pack_nchw_to_nchwc8(const TensorView& src, TensorView& dst,
                          const ComputeContext& ctx = {});

void unpack_nchwc8_to_nchw(const TensorView& src, TensorView& dst,
                            const ComputeContext& ctx = {});

// ============================================================
// 3D: NCDHW <-> NCDHWC8
// ============================================================

/// Dst must be pre-allocated as NCDHWC8 [N, ceil(C/8), D, H, W, 8].
void pack_ncdhw_to_ncdhwc8(const TensorView& src, TensorView& dst,
                            const ComputeContext& ctx = {});

void unpack_ncdhwc8_to_ncdhw(const TensorView& src, TensorView& dst,
                              const ComputeContext& ctx = {});

/// Bytes for an aligned NCHWC8 buffer of a logical NCHW desc.
size_t nchwc8_storage_bytes(const TensorDesc& logical_desc,
                             int64_t alignment = 32);

}  // namespace nnops
