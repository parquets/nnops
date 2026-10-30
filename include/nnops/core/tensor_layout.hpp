#pragma once
/// @file tensor_layout.hpp
/// @brief TensorLayout enum for describing data arrangement in memory.
///
/// Naming: NC + optional D (3D) + HW, with an optional Cx channel-pack suffix
/// (x channels blocked at the innermost position).
///
///   NCHW / NCDHW       — planar 2D / 3D
///   NCHWC8 / 16 / 32   — 2D, C blocked into 8/16/32-channel lanes
///   NCDHWC8 / 16 / 32  — 3D packed
///   NHWC / NDHWC       — channel-last 2D / 3D (pack = 1)
///
/// Channel-last keeps the logical [N, C, ...] dims in the descriptor exactly like
/// the packed layouts; only the physical order changes, so the "row" of the pitch
/// model is one spatial line — (n, h) over all W pixels and all C channels — and
/// the channel axis is contiguous. A [W][Cx] row is one contiguous block;
/// channel-packed tensors use pitch = align_up(W * pack * elem_size, 32) (see
/// TensorDesc::row_pitch()), and a TensorView built with an explicit pitch may
/// use any larger value.

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
    PackedWinoWeight = 9,  ///< [16, OC, IC/G] — F(2,3) Winograd U for a planar
                           ///< activation, one transform-domain kernel per p ∈ [0,16).

    // ---- Channel-last (NHWC family) ----
    NHWC      = 10,  ///< logical [N, C, H, W], stored [N, H, W, C]
    NDHWC     = 11,  ///< logical [N, C, D, H, W], stored [N, D, H, W, C]

    // ---- Channel-last conv weight ----
    HWIO      = 12,  ///< [KH][KW][IC/G][OC] — TF filter order for channel-last
                     ///< activations: the shape *is* H×W×I×O, OC contiguous.

    // ---- Winograd-transformed weight, channel-last ----
    PackedWinoWeightLast = 13,  ///< the same U for a channel-last activation, as
                                ///< the GEMM's B panels [16][⌈IC/G ÷ KC⌉][G·⌈OC/G ÷ nr⌉][ldd],
                                ///< each [KC][nr] k-major, OC zero-padded to panels;
                                ///< KC divides IC/G, nr is the arch's panel width.
                                ///< Opaque outside the Winograd kernel.

    // ---- Channel-last depthwise weight ----
    HWCM = 14,  ///< [KH][KW][C] (3D: [KD][KH][KW][C]) — TF depthwise filter with
                ///< multiplier 1 (OC == IC); one tap's 8 channels load as a vector.

    // ---- Prepacked conv weight (Conv2D::prepackWeights, im2col route) ----
    //
    // The tags name which route's operand a blob holds; the two routes swap their
    // GEMM's operands, so they pack along different axes:
    //   PackedOIHW — planar (OIHW) filter, planar activation: the filter is the A
    //                operand (fp: [OC][IC/G·KH·KW] copy, quant: packed-A panels).
    //   PackedHWIO — channel-last (HWIO) filter, channel-last activation: the
    //                filter is the B operand (one k-major panel of nr OC per k block).
    //
    // Both are opaque outside the im2col kernel: the descriptor's shape is
    // [OC, ⌈elements / OC⌉] so shape inference still reads the output-channel
    // count; panel geometry is resolved from the attributes and the filter's
    // dims, never indexed by hand.
    PackedOIHW = 15,  ///< planar activation's prepacked filter (see above)
    PackedHWIO = 16,  ///< channel-last activation's prepacked filter (see above)

    // ---- Prepacked matmul RHS (MatMul / Linear ::prepackWeights) ----
    //
    // The GEMM's B operand, packed over its *whole* K once at load time instead of
    // into pooled scratch every call: ceil(N / nr) panels of nr columns, each
    // [K][nr] k-major with the tail zero-filled, panels `ldd` elements apart.
    // Packing at full K makes `ldd` depend on K, not the call's k-block, so the
    // plan's kc stays free (it depends on M, N and the worker count); the MMA
    // walks one panel at a time (tile_mma_pack_ldb). alpha cannot ride the pack
    // (it is per call), so a prepacked B requires alpha == 1. Shape is
    // [N, ⌈elements / N⌉], N first so shape inference reads the output's last dim.
    PackedB = 17,
};


/// Channel pack size for packed layouts, 1 for planar.
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
    case TensorLayout::PackedWinoWeightLast: return 1;
    case TensorLayout::HWCM:     return 1;
    case TensorLayout::NHWC:     return 1;
    case TensorLayout::NDHWC:    return 1;
    case TensorLayout::HWIO:     return 1;
    case TensorLayout::PackedOIHW: return 1;
    case TensorLayout::PackedHWIO: return 1;
    case TensorLayout::PackedB:  return 1;
    }
    return 1;
}

/// True if the layout uses channel packing (NCHWCx / NCDHWCx).
constexpr bool is_channel_packed(TensorLayout layout) noexcept {
    return layout_channel_pack(layout) > 1;
}

/// True for NHWC, NDHWC — channel is the innermost, contiguous axis.
constexpr bool is_channel_last(TensorLayout layout) noexcept {
    return layout == TensorLayout::NHWC || layout == TensorLayout::NDHWC;
}

/// True for NCHW, NCDHW — the activation layouts a graph arrives in.
constexpr bool is_planar_spatial(TensorLayout layout) noexcept {
    return layout == TensorLayout::NCHW || layout == TensorLayout::NCDHW;
}

/// True for HWIO, the channel-last conv weight order.
constexpr bool is_channel_last_weight(TensorLayout layout) noexcept {
    return layout == TensorLayout::HWIO;
}

/// True for HWCM, the channel-last depthwise filter.
constexpr bool is_depthwise_weight_last(TensorLayout layout) noexcept {
    return layout == TensorLayout::HWCM;
}

/// True for anything not channel-packed — activations and weight tags alike.
constexpr bool is_unpacked(TensorLayout layout) noexcept {
    return !is_channel_packed(layout);
}

/// True for either Winograd weight form; the two pair with NCHW and NHWC.
constexpr bool is_wino_weight(TensorLayout layout) noexcept {
    return layout == TensorLayout::PackedWinoWeight ||
           layout == TensorLayout::PackedWinoWeightLast;
}

/// True for either prepacked im2col conv filter; opaque blobs.
constexpr bool is_packed_conv_weight(TensorLayout layout) noexcept {
    return layout == TensorLayout::PackedOIHW ||
           layout == TensorLayout::PackedHWIO;
}

/// True for the prepacked matmul RHS; opaque, addressed by matmul_packed_b_geom.
constexpr bool is_packed_matmul_b(TensorLayout layout) noexcept {
    return layout == TensorLayout::PackedB;
}

/// Spatial rank encoded in the layout: NCHW* → 2, NCDHW* → 3.
constexpr int64_t layout_spatial_rank(TensorLayout layout) noexcept {
    switch (layout) {
    case TensorLayout::NCHW:
    case TensorLayout::NCHWC8:
    case TensorLayout::NCHWC16:
    case TensorLayout::NCHWC32:
    case TensorLayout::PackedWeight:
    case TensorLayout::HWCM:
    case TensorLayout::PackedWinoWeight:
    case TensorLayout::PackedWinoWeightLast:
    case TensorLayout::NHWC:
    case TensorLayout::HWIO:
    case TensorLayout::PackedOIHW:
    case TensorLayout::PackedHWIO:
    case TensorLayout::PackedB:
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

// Backward-compatibility alias (use layout_channel_pack).
constexpr int64_t layout_channel_block(TensorLayout layout) noexcept {
    return layout_channel_pack(layout);
}

// ============================================================
// LayoutSupport — per-operator layout capability declaration
// ============================================================

/// Categories of layout support for an operator.
///
/// Mirrors OpenCV DNN's C/A/B classification:
///   PlanarOnly — C operations: only planar layouts
///   PackedOnly — B operations: only channel-packed layouts
///   Any        — A operations: pitch-aware, handles any layout
enum class LayoutSupport : uint8_t {
    PlanarOnly,   ///< is_unpacked && !is_channel_last; HWIO and other weight tags pass
    PackedOnly,   ///< NCHWC8/16/32, NCDHWC8/16/32
    Any,          ///< all layouts (pitch-aware row processing)
    /// Anything without channel blocking (is_unpacked): activations plus every
    /// weight tag. Ops with per-layout fast paths (Conv2D today), which must
    /// handle the spatial axis as either the row or the outer axis.
    PlanarOrChannelLast,
};

/// True when @p layout satisfies the @p support constraint.
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
