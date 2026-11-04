#pragma once
/// @file layout_convert.hpp
/// @brief Internal planar ↔ channel-last conversion.
///
/// Source and destination describe the same logical shape (NCHW dim order);
/// only the physical layout differs. 2D and 3D run the same kernel — a 3D
/// tensor is just more spatial rows. f32/f16 transpose 8 pixels × 8 channels
/// with SIMD; s8/u8 fall back to scalar copies. Every spatial row is an
/// independent task on ctx.cpu.run.

#include "nnops/core/tensor_view.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstddef>

namespace nnops {

/// NCHW/NCDHW → NHWC/NDHWC. `dst` carries the channel-last layout.
void convert_planar_to_channel_last(const TensorView& src, TensorView& dst,
                                    const ComputeContext& ctx = {});

/// NHWC/NDHWC → NCHW/NCDHW.
void convert_channel_last_to_planar(const TensorView& src, TensorView& dst,
                                    const ComputeContext& ctx = {});

/// Bytes for an aligned channel-last buffer of a logical desc.
size_t channel_last_storage_bytes(const TensorDesc& logical_desc,
                                  int64_t alignment = 32);

}  // namespace nnops
