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
///   NHWC:     [N, H, W, C]                  — 2D channel-last (pack = 1)
///   NDHWC:    [N, D, H, W, C]               — 3D channel-last (pack = 1)
///
/// Channel-last layouts keep the *logical* [N, C, ...] dims and in the descriptor
/// exactly like the packed ones; only the physical order changes. There the
/// "row" of the pitch model is one spatial line — (n, h) over all W pixels and
/// all C channels — so pitch = W * C * elem_size and the channel axis is the
/// contiguous one.
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

    // ---- Winograd-transformed weight ----
    PackedWinoWeight = 9,  ///< [16, OC, IC/G] — F(2,3) Winograd-transformed weight U
                           ///< (one 4×4 transform-domain kernel per position p ∈ [0,16))

    // ---- Channel-last (NHWC family) ----
    NHWC      = 10,  ///< logical [N, C, H, W], stored [N, H, W, C]
    NDHWC     = 11,  ///< logical [N, C, D, H, W], stored [N, D, H, W, C]

    // ---- Channel-last conv weight ----
    HWIO      = 12,  ///< [KH][KW][IC/G][OC] — TensorFlow's filter order for
                     ///< channel-last activations, so the tensor's shape *is*
                     ///< H×W×I×O (unlike NHWC, which keeps the logical NCHW
                     ///< dims and changes only the storage order). The output
                     ///< channel is the contiguous axis, which is what the
                     ///< channel-last kernel's B panels want: a filter tap for
                     ///< one input channel is one contiguous OC-length run.
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
    case TensorLayout::PackedWinoWeight: return 1;
    case TensorLayout::NHWC:     return 1;
    case TensorLayout::NDHWC:    return 1;
    case TensorLayout::HWIO:     return 1;
    }
    return 1;
}

/// Returns true if the layout uses channel packing (NCHWCx / NCDHWCx).
constexpr bool is_channel_packed(TensorLayout layout) noexcept {
    return layout_channel_pack(layout) > 1;
}

/// Returns true for the channel-last layouts (NHWC, NDHWC) — the channel is the
/// innermost, contiguous axis; the spatial dims are the outer ones.
constexpr bool is_channel_last(TensorLayout layout) noexcept {
    return layout == TensorLayout::NHWC || layout == TensorLayout::NDHWC;
}

/// Returns true for the planar spatial layouts (NCHW, NCDHW) — the activations
/// a graph arrives in, as opposed to weight layouts.
constexpr bool is_planar_spatial(TensorLayout layout) noexcept {
    return layout == TensorLayout::NCHW || layout == TensorLayout::NCDHW;
}

/// Returns true for the channel-last conv weight (HWIO), the weight order that
/// pairs with channel-last activations: it is TensorFlow's [KH][KW][IC/G][OC]
/// filter, so the output-channel axis is the contiguous one.
constexpr bool is_channel_last_weight(TensorLayout layout) noexcept {
    return layout == TensorLayout::HWIO;
}

/// Returns true for the activation layouts with no channel blocking: planar
/// (NCHW, NCDHW) and channel-last (NHWC, NDHWC). Weight layouts are planar.
constexpr bool is_unpacked(TensorLayout layout) noexcept {
    return !is_channel_packed(layout);
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
    case TensorLayout::PackedWinoWeight:
    case TensorLayout::NHWC:
    case TensorLayout::HWIO:
        return 2;
    case TensorLayout::NCDHW:
    case TensorLayout::NCDHWC8:
    case TensorLayout::NCDHWC16:
    case TensorLayout::NCDHWC32:
    case TensorLayout::NDHWC:
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
///   PlanarOnly — C operations: can only handle planar layouts (NCHW, NCDHW)
///   PackedOnly — B operations: can only handle channel-packed layouts (NCHWC8, etc.)
///   Any       — A operations: pitch-aware, handles any layout transparently
enum class LayoutSupport : uint8_t {
    PlanarOnly,   ///< Only planar (non-packed, non-channel-last) layouts: NCHW, NCDHW
                  ///< (OIHW and prepacked weight layouts, which are not
                  ///< channel-last, pass too; HWIO does not)
    PackedOnly,   ///< Only channel-packed layouts: NCHWC8/16/32, NCDHWC8/16/32
    Any,          ///< All layouts supported (pitch-aware row processing)
    /// Planar and channel-last, but no channel blocking: NCHW, NCDHW, NHWC,
    /// NDHWC. The ops with per-layout fast paths (Conv2D today) — they must
    /// handle the spatial axis being either the row or the outer axis.
    PlanarOrChannelLast,
};

/// Check whether a single TensorLayout satisfies a LayoutSupport constraint.
inline bool is_layout_supported(TensorLayout layout, LayoutSupport support) noexcept {
    switch (support) {
    case LayoutSupport::PlanarOnly:
        return is_unpacked(layout) && !is_channel_last(layout);
    case LayoutSupport::PackedOnly:
        return is_channel_packed(layout);
    case LayoutSupport::Any:
        return true;
    case LayoutSupport::PlanarOrChannelLast:
        return is_unpacked(layout);
    }
    return false;
}

}  // namespace nnops
