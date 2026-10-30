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
                           ///< for a *planar* activation, where OC is the GEMM's M.
                           ///< The channel-last form of the same U is
                           ///< PackedWinoWeightLast.

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

    // ---- Winograd-transformed weight, channel-last ----
    PackedWinoWeightLast = 13,  ///< the same U for a channel-last activation,
                                ///< stored as the GEMM's own B panels:
                                ///< [16][⌈IC/G ÷ KC⌉][G·⌈OC/G ÷ nr⌉][ldd], each
                                ///< panel [KC][nr] k-major with OC zero-padded to
                                ///< whole panels (there the *pixels* are the
                                ///< GEMM's M and the output channels its N, so
                                ///< the transform is tile-major). KC divides IC/G
                                ///< and is at most 128; nr is the arch's panel
                                ///< width. Prepacked weights are opaque blobs to
                                ///< every consumer outside the Winograd kernel —
                                ///< this tag is a promise about *which* kernel,
                                ///< not a layout anyone else may index.

    // ---- Channel-last depthwise weight ----
    HWCM = 14,  ///< [KH][KW][C] (3D: [KD][KH][KW][C]) — TensorFlow's depthwise
                ///< filter [KH, KW, C, M] with the multiplier M == 1, which is
                ///< what this operator requires (OC == IC). Like HWIO for a
                ///< dense conv, the tensor's *shape* is the filter itself, and
                ///< it pairs with a channel-last activation: one vector per tap
                ///< is 8 consecutive channels of that tap, which is exactly
                ///< what the channel-lane kernel loads. The planar filter
                ///< [C, 1, (KD,) KH, KW] stores a tap's channels C elements
                ///< apart, so it cannot serve that kernel.

    // ---- Prepacked conv weight (Conv2D::prepackWeights, im2col route) ----
    //
    // The two tags name which *route*'s operand a blob holds, because the two
    // routes swap their GEMM's operands and therefore pack the filter along
    // different axes:
    //
    //   PackedOIHW — from a planar (OIHW) filter, for a planar activation. The
    //                planar GEMM makes the filter its A operand, so the fp form
    //                is the filter's own [OC][IC/G·KH·KW] matrix (nothing to
    //                transpose, which is why the fp prepack is only a densifying
    //                copy) and the quantized form is the MMA's packed-A panels,
    //                which the kernel otherwise rebuilds on every task.
    //   PackedHWIO — from a channel-last (HWIO) filter, for a channel-last
    //                activation. There the filter is the B operand: the blob is
    //                one k-major panel of `nr` output channels per k block,
    //                which the kernel otherwise repacks on every compute().
    //
    // Both are opaque to everything outside the im2col kernel: the descriptor's
    // shape is [OC, ⌈elements / OC⌉] — the output-channel count, so shape
    // inference still reads the right number, followed by whatever the panel
    // padding needs — and the panel geometry is resolved from the attributes
    // plus the filter's own dims, never indexed by hand.
    PackedOIHW = 15,  ///< planar activation's prepacked filter (see above)
    PackedHWIO = 16,  ///< channel-last activation's prepacked filter (see above)

    // ---- Prepacked matmul RHS (MatMul / Linear ::prepackWeights) ----
    //
    // The GEMM's B operand, packed over its *whole* K once at load time
    // instead of into pooled scratch on every call. It is the same uniform
    // panel layout the packers write — ceil(N / nr) panels of nr columns, each
    // [K][nr] k-major with the columns past N zero-filled, panels `ldd`
    // elements apart — where the RHS pack builds the same bytes for a k-block
    // of kc. Two things follow from packing at full K, and both are the point:
    //
    //   * `ldd` is a function of K, not of the call's k-block, so the plan's
    //     kc stays free (it depends on M, N and the worker count) and nothing
    //     has to agree with it. The MMA is handed the stride and walks one
    //     panel at a time; see tile_mma_pack_ldb.
    //   * alpha cannot ride the RHS pack any more (the pack is load-time and
    //     alpha is per call), so a prepacked B requires alpha == 1.
    //
    // The descriptor's shape is [N, ⌈elements / N⌉]: N first, because shape
    // inference reads the output's last dim from it, then whatever the panel
    // padding needs — the same convention as the conv blobs above. The panel
    // geometry is resolved from (N, K, dtype), never indexed by hand.
    PackedB = 17,
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

/// Returns true for the channel-last *depthwise* filter (HWCM) — [KH][KW][C],
/// the weight order that pairs with a channel-last activation.
constexpr bool is_depthwise_weight_last(TensorLayout layout) noexcept {
    return layout == TensorLayout::HWCM;
}

/// Returns true for the activation layouts with no channel blocking: planar
/// (NCHW, NCDHW) and channel-last (NHWC, NDHWC). Weight layouts are planar.
constexpr bool is_unpacked(TensorLayout layout) noexcept {
    return !is_channel_packed(layout);
}

/// Returns true for either form of the Winograd-transformed weight — the
/// prepacked U a Winograd kernel consumes. The two forms pair with the two
/// activation layouts (see Conv2D::validate): planar U with NCHW, the panel
/// form with NHWC.
constexpr bool is_wino_weight(TensorLayout layout) noexcept {
    return layout == TensorLayout::PackedWinoWeight ||
           layout == TensorLayout::PackedWinoWeightLast;
}

/// Returns true for either form of the prepacked im2col conv filter — the
/// operand Conv2D::prepackWeights builds for the route its source layout pairs
/// with (PackedOIHW for a planar activation, PackedHWIO for a channel-last
/// one). Both are opaque blobs; only the kernel that packed them indexes them.
constexpr bool is_packed_conv_weight(TensorLayout layout) noexcept {
    return layout == TensorLayout::PackedOIHW ||
           layout == TensorLayout::PackedHWIO;
}

/// Returns true for the prepacked matmul RHS — the operand MatMul/Linear
/// prepackWeights builds over the whole K. Opaque: only the matmul kernel that
/// packed it indexes it, and it is addressed by the geometry matmul_packed_b_geom
/// resolves from the descriptor's own N and the activation's K.
constexpr bool is_packed_matmul_b(TensorLayout layout) noexcept {
    return layout == TensorLayout::PackedB;
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
