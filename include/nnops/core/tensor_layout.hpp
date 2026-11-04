#pragma once
/// @file tensor_layout.hpp
/// @brief TensorLayout enum for describing data arrangement in memory.
///
/// Two activation families:
///   NCHW / NCDHW  — planar 2D / 3D
///   NHWC / NDHWC  — channel-last 2D / 3D
///
/// Channel-last keeps the logical [N, C, ...] dims in the descriptor exactly as
/// planar does; only the physical order changes, so the "row" of the pitch model
/// is one spatial line — (n, h) over all W pixels and all C channels — and the
/// channel axis is contiguous. A [W][C] row is one contiguous block, with
/// pitch = align_up(W * C * elem_size, 32) (see TensorDesc::row_pitch()); a
/// TensorView built with an explicit pitch may use any larger value.
///
/// The packed weight tags at the end describe filter blobs produced by
/// prepackWeights, never activations.

#include <cstdint>

namespace nnops {

/// Memory layout of tensor data.
enum class TensorLayout : uint8_t {
    // ---- 2D planar ----
    NCHW      = 0,  ///< [N, C, H, W]

    // ---- 3D planar ----
    NCDHW     = 1,  ///< [N, C, D, H, W]

    // ---- Channel-last (NHWC family) ----
    NHWC      = 2,  ///< logical [N, C, H, W], stored [N, H, W, C]
    NDHWC     = 3,  ///< logical [N, C, D, H, W], stored [N, D, H, W, C]

    // ---- Winograd-transformed weight ----
    PackedWinoWeight = 4,  ///< [16, OC, IC/G] — F(2,3) Winograd U for a planar
                           ///< activation, one transform-domain kernel per p ∈ [0,16).

    // ---- Channel-last conv weight ----
    HWIO      = 5,  ///< [KH][KW][IC/G][OC] — TF filter order for channel-last
                    ///< activations: the shape *is* H×W×I×O, OC contiguous.

    // ---- Winograd-transformed weight, channel-last ----
    PackedWinoWeightLast = 6,  ///< the same U for a channel-last activation, as
                               ///< the GEMM's B panels [16][⌈IC/G ÷ KC⌉][G·⌈OC/G ÷ nr⌉][ldd],
                               ///< each [KC][nr] k-major, OC zero-padded to panels;
                               ///< KC divides IC/G, nr is the arch's panel width.
                               ///< Opaque outside the Winograd kernel.

    // ---- Channel-last depthwise weight ----
    HWCM = 7,  ///< [KH][KW][C] (3D: [KD][KH][KW][C]) — TF depthwise filter with
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
    PackedOIHW = 8,   ///< planar activation's prepacked filter (see above)
    PackedHWIO = 9,   ///< channel-last activation's prepacked filter (see above)

    // ---- Prepacked matmul RHS (MatMul / Linear ::prepackWeights) ----
    //
    // The GEMM's B operand, packed over its *whole* K once at load time instead of
    // into pooled scratch every call: ceil(N / nr) panels of nr columns, each
    // [K][nr] k-major with the tail zero-filled, panels `ldd` elements apart.
    // Packing at full K makes `ldd` depend on K, not the call's k-block, so the
    // plan's kc stays free (it depends on M, N and the worker count); the MMA
    // walks one panel at a time (tile_mma_pack at the blob's stride). alpha
    // cannot ride the pack (it is per call), so a prepacked B requires
    // alpha == 1. Shape is
    // [N, ⌈elements / N⌉], N first so shape inference reads the output's last dim.
    PackedB = 10,

    // ---- Prepacked recurrent weights (Rnn / Gru / Lstm ::prepackWeights) ----
    //
    // Everything the step loop would otherwise rebuild on every call, for every
    // direction: W and R interleaved into the gate-block order and packed into
    // the MMA's nr-wide B panels, plus the per-phase bias vectors (Rbh kept
    // apart under linear_before_reset). Opaque and self-describing — the blob
    // leads with its own header, so a compute() handed one reads it in place
    // instead of running the interleave and the two pack passes.
    PackedRecurrent = 11,
};

/// True for NHWC, NDHWC — channel is the innermost, contiguous axis.
constexpr bool is_channel_last(TensorLayout layout) noexcept {
    return layout == TensorLayout::NHWC || layout == TensorLayout::NDHWC;
}

/// True for NCHW, NCDHW — planar activations.
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

/// True for the recurrent family's prepacked weights; opaque, self-describing.
constexpr bool is_packed_recurrent(TensorLayout layout) noexcept {
    return layout == TensorLayout::PackedRecurrent;
}

/// True for any layout a conv-family filter may carry: a raw planar or
/// channel-last filter, or one of the prepacked blobs.
constexpr bool is_conv_filter_layout(TensorLayout layout) noexcept {
    return is_planar_spatial(layout) || is_channel_last(layout) ||
           is_channel_last_weight(layout) || is_packed_conv_weight(layout) ||
           is_wino_weight(layout);
}

/// Spatial rank encoded in the layout: NCHW/NHWC → 2, NCDHW/NDHWC → 3.
constexpr int64_t layout_spatial_rank(TensorLayout layout) noexcept {
    switch (layout) {
    case TensorLayout::NCDHW:
    case TensorLayout::NDHWC:
        return 3;
    default:
        return 2;
    }
}

// ============================================================
// LayoutSupport — per-operator layout capability declaration
// ============================================================

/// Categories of layout support for an operator.
///
///   PlanarOnly — planar activations (plus every weight tag)
///   Any        — planar or channel-last, pitch-aware
///   PlanarOrChannelLast — both activation families (same as Any today; kept
///                          so an op can declare its spatial fast path)
enum class LayoutSupport : uint8_t {
    PlanarOnly,          ///< NCHW/NCDHW; channel-last and weight tags rejected
    Any,                 ///< every layout (pitch-aware processing)
    PlanarOrChannelLast, ///< NCHW/NCDHW/NHWC/NDHWC
};

/// True when @p layout satisfies the @p support constraint.
inline bool is_layout_supported(TensorLayout layout, LayoutSupport support) noexcept {
    switch (support) {
    case LayoutSupport::PlanarOnly:
        return is_planar_spatial(layout);
    case LayoutSupport::Any:
        return true;
    case LayoutSupport::PlanarOrChannelLast:
        return is_planar_spatial(layout) || is_channel_last(layout);
    }
    return false;
}

}  // namespace nnops
