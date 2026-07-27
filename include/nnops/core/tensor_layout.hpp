#pragma once
/// @file tensor_layout.hpp
/// @brief TensorLayout enum for describing data arrangement in memory.
///
/// Layout naming convention:
///   NC  - batch + channels prefix
///   D   - optional depth dimension (3D: NCDHW, 2D: NCHW)
///   HW  - spatial dimensions (height, width)
///   Cx  - channel packing: x channels blocked at innermost position
///
/// Memory layout examples (logical dims stored in TensorDesc::dims):
///   NCHW:     [N, C, H, W]                  — planar 2D
///   NCDHW:    [N, C, D, H, W]               — planar 3D
///   NCHWC8:   [N, ceil(C/8), H, W, 8]       — 2D, 8 chan/lane (f32×8=256b, f16×8=128b)
///   NCHWC16:  [N, ceil(C/16), H, W, 16]     — 2D, 16 chan/lane (i8×16=128b SSE/NEON)
///   NCHWC32:  [N, ceil(C/32), H, W, 32]     — 2D, 32 chan/lane (AVX-512)
///   NCDHWC8:  [N, ceil(C/8), D, H, W, 8]    — 3D packed variant
///
/// The [W][Cx] row is always a single contiguous block; pitch = align_up(W * Cx * elem_size, 32).

#include <cstdint>

namespace nnops {

/// Memory layout of tensor data.
enum class TensorLayout : uint8_t {
    // ---- 2D planar ----
    NCHW      = 0,  ///< [N, C, H, W]

    // ---- 2D channel-packed ----
    NCHWC8    = 1,  ///< [N, ceil(C/8),  H, W, 8]   (f32/f16 SIMD: 8×4=256b / 8×2=128b)
    NCHWC16   = 2,  ///< [N, ceil(C/16), H, W, 16]  (int8 quant: 16×1=128b SSE/NEON)
    NCHWC32   = 3,  ///< [N, ceil(C/32), H, W, 32]  (AVX-512)

    // ---- 3D planar ----
    NCDHW     = 4,  ///< [N, C, D, H, W]

    // ---- 3D channel-packed ----
    NCDHWC8   = 5,  ///< [N, ceil(C/8),  D, H, W, 8]
    NCDHWC16  = 6,  ///< [N, ceil(C/16), D, H, W, 16]  (int8 quant)
    NCDHWC32  = 7,  ///< [N, ceil(C/32), D, H, W, 32]  (AVX-512)
};

/// Returns the channel pack size for packed layouts, or 1 for planar layouts.
constexpr int64_t layout_channel_pack(TensorLayout layout) noexcept {
    switch (layout) {
    case TensorLayout::NCHW:     return 1;
    case TensorLayout::NCDHW:    return 1;
    case TensorLayout::NCHWC8:   return 8;
    case TensorLayout::NCDHWC8:  return 8;
    case TensorLayout::NCHWC16:  return 16;
    case TensorLayout::NCDHWC16: return 16;
    case TensorLayout::NCHWC32:  return 32;
    case TensorLayout::NCDHWC32: return 32;
    }
    return 1;
}

/// Returns true if the layout uses channel packing (NCHWCx / NCDHWCx).
constexpr bool is_channel_packed(TensorLayout layout) noexcept {
    return layout_channel_pack(layout) > 1;
}

/// Returns the expected spatial rank encoded in the layout name.
/// NCHW* → 2 (H, W), NCDHW* → 3 (D, H, W).
constexpr int64_t layout_spatial_rank(TensorLayout layout) noexcept {
    switch (layout) {
    case TensorLayout::NCHW:
    case TensorLayout::NCHWC8:
    case TensorLayout::NCHWC16:
    case TensorLayout::NCHWC32:
        return 2;
    case TensorLayout::NCDHW:
    case TensorLayout::NCDHWC8:
    case TensorLayout::NCDHWC16:
    case TensorLayout::NCDHWC32:
        return 3;
    }
    return 2;
}

// Backward-compatibility alias (use layout_channel_pack going forward).
constexpr int64_t layout_channel_block(TensorLayout layout) noexcept {
    return layout_channel_pack(layout);
}

}  // namespace nnops
