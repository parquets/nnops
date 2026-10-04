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
/// A [W][Cx] row is a single contiguous block. Channel-packed tensors use
/// pitch = align_up(W * pack * elem_size, 32) — see TensorDesc::row_pitch().
/// A TensorView built with an explicit pitch may use any larger value.

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

    // ---- Prepacked weight (dense, last dim = 8 lanes) ----
    PackedWeight = 8,  ///< [C8, (KD,) KH, KW, 8] — prepacked depthwise/conv weights
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
    case TensorLayout::PackedWeight: return 1;
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
    case TensorLayout::PackedWeight:
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

// ============================================================
// LayoutSupport — per-operator layout capability declaration
// ============================================================

/// Categories of layout support for an operator.
///
/// Mirrors OpenCV DNN's C/A/B classification:
///   PlanarOnly — C operations: can only handle non-packed layouts (NCHW, NCDHW)
///   PackedOnly — B operations: can only handle channel-packed layouts (NCHWC8, etc.)
///   Any       — A operations: pitch-aware, handles any layout transparently
enum class LayoutSupport : uint8_t {
    PlanarOnly,   ///< Only planar (non-packed) layouts: NCHW, NCDHW
    PackedOnly,   ///< Only channel-packed layouts: NCHWC8/16/32, NCDHWC8/16/32
    Any,          ///< All layouts supported (pitch-aware row processing)
};

/// Check whether a single TensorLayout satisfies a LayoutSupport constraint.
inline bool is_layout_supported(TensorLayout layout, LayoutSupport support) noexcept {
    switch (support) {
    case LayoutSupport::PlanarOnly:
        return !is_channel_packed(layout);
    case LayoutSupport::PackedOnly:
        return is_channel_packed(layout);
    case LayoutSupport::Any:
        return true;
    }
    return false;
}

}  // namespace nnops
