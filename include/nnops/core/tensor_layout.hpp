#pragma once
/// @file tensor_layout.hpp
/// @brief TensorLayout enum for describing data arrangement in memory.
///
/// Spatial activations are NHWC/NDHWC or dense (Dense). A spatial layout keeps
/// the logical [N, C, ...] dims in the descriptor; only the physical order
/// changes, so the "row" of the pitch model is one spatial line — (n, h) over
/// all W pixels and all C channels — and the channel axis is contiguous.
/// A [W][C] row is one contiguous block, with pitch =
/// align_up(W * C * elem_size, 32) (see TensorDesc::row_pitch()); a TensorView
/// built with an explicit pitch may use any larger value.
///
/// Dense is plain row-major by the logical dims, with no channel-axis claim:
/// it is the default for layout-agnostic operators, GEMM operands, scratch and
/// artifact buffers.
///
/// The packed weight tags describe filter blobs produced by prepackWeights,
/// never activations.

#include <cstdint>

namespace nnops {

/// Memory layout of tensor data.
enum class TensorLayout : uint8_t {
    // ---- spatial activations, 2D / 3D ----
    NHWC      = 2,  ///< logical [N, C, H, W], stored [N, H, W, C]
    NDHWC     = 3,  ///< logical [N, C, D, H, W], stored [N, D, H, W, C]

    // ---- Spatial conv weight ----
    HWIO      = 5,  ///< [KH][KW][IC/G][OC] — TF filter order for spatial
                    ///< activations: the shape *is* H×W×I×O, OC contiguous.

    // ---- Winograd-transformed weight, spatial ----
    PackedWinoWeightLast = 6,  ///< the U for a spatial activation, as the
                               ///< GEMM's B panels [16][⌈IC/G ÷ KC⌉][G·⌈OC/G ÷ nr⌉][ldd],
                               ///< each [KC][nr] k-major, OC zero-padded to panels;
                               ///< KC divides IC/G, nr is the arch's panel width.
                               ///< Opaque outside the Winograd kernel.

    // ---- Spatial depthwise weight ----
    HWCM = 7,  ///< [KH][KW][C] (3D: [KD][KH][KW][C]) — TF depthwise filter with
               ///< multiplier 1 (OC == IC); one tap's 8 channels load as a vector.

    // ---- Prepacked conv weight (Conv2D::prepackWeights, im2col route) ----
    //
    // PackedHWIO — the HWIO filter: the filter is the B operand (one
    // k-major panel of nr OC per k block). Opaque outside the im2col kernel: the
    // descriptor's shape is [OC, ⌈elements / OC⌉] so shape inference still reads
    // the output-channel count; panel geometry is resolved from the attributes
    // and the filter's dims, never indexed by hand.
    PackedHWIO = 9,

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

    // ---- Prepacked matmul RHS, AArch64 SME (f32) ----
    //
    // The same idea as PackedB, in the layout the ZA kernels want: 64 contiguous
    // output columns per k, `[⌈N/64⌉][K][64]`, tail columns zero-filled. Packed
    // once at load time, it needs no per-call RHS pack. Small M reads it too —
    // inside one 64-column block the blob is a plain [K][64] row-major matrix,
    // which is exactly what the GEMV wants (ldb = 64, one call per block).
    PackedBSme = 11,

    // ---- Prepacked recurrent weights (Rnn / Gru / Lstm ::prepackWeights) ----
    //
    // Everything the step loop would otherwise rebuild on every call, for every
    // direction: W and R interleaved into the gate-block order and packed into
    // the MMA's nr-wide B panels, plus the per-phase bias vectors (Rbh kept
    // apart under linear_before_reset). Opaque and self-describing — the blob
    // leads with its own header, so a compute() handed one reads it in place
    // instead of running the interleave and the two pack passes.
    PackedRecurrent = 11,

    // ---- dense row-major, no channel-axis semantics ----
    //
    // The logical dims are the memory order: dim rank-1 is contiguous, and the
    // second-innermost step is the row pitch. There is no claim about which
    // axis is the channel, so a kernel that only walks rows or applies an axis
    // attribute treats this like any dense tensor. It is the default for
    // layout-agnostic and flat tensors (GEMM operands, norm/resize scratch,
    // artifact buffers, reference data).
    Dense = 12,
};

/// True for the spatial layouts, NHWC and NDHWC.
constexpr bool is_channel_last(TensorLayout layout) noexcept {
    return layout == TensorLayout::NHWC || layout == TensorLayout::NDHWC;
}

/// True for HWIO, the conv weight order of a spatial activation.
constexpr bool is_channel_last_weight(TensorLayout layout) noexcept {
    return layout == TensorLayout::HWIO;
}

/// True for HWCM, the depthwise filter with the channel innermost.
constexpr bool is_depthwise_weight_last(TensorLayout layout) noexcept {
    return layout == TensorLayout::HWCM;
}

/// True for the Winograd U weight form; opaque outside its kernel.
constexpr bool is_wino_weight(TensorLayout layout) noexcept {
    return layout == TensorLayout::PackedWinoWeightLast;
}

/// True for the prepacked im2col conv filter; an opaque blob.
constexpr bool is_packed_conv_weight(TensorLayout layout) noexcept {
    return layout == TensorLayout::PackedHWIO;
}

/// True for the prepacked matmul RHS; opaque, addressed by matmul_packed_b_geom.
constexpr bool is_packed_matmul_b(TensorLayout layout) noexcept {
    return layout == TensorLayout::PackedB || layout == TensorLayout::PackedBSme;
}

/// True for the recurrent family's prepacked weights; opaque, self-describing.
constexpr bool is_packed_recurrent(TensorLayout layout) noexcept {
    return layout == TensorLayout::PackedRecurrent;
}

/// Spatial rank encoded in the layout: NDHWC → 3, everything else → 2.
constexpr int64_t layout_spatial_rank(TensorLayout layout) noexcept {
    return layout == TensorLayout::NDHWC ? 3 : 2;
}

// ============================================================
// LayoutSupport — per-operator layout capability declaration
// ============================================================

/// Categories of layout support for an operator.
///
///   ChannelLastOnly  — NHWC/NDHWC activations (plus weight tags)
///   DenseOnly        — dense row-major only
///   Any              — layout-agnostic: the data is walked as flat rows, so
///                      neither family is interpreted
enum class LayoutSupport : uint8_t {
    ChannelLastOnly,  ///< NHWC/NDHWC only; other layouts and weight tags rejected
    DenseOnly,        ///< Dense only: the logical order is the memory order
    Any,              ///< every layout (layout-agnostic, pitch-aware processing)
};

/// True when @p layout satisfies the @p support constraint.
inline bool is_layout_supported(TensorLayout layout, LayoutSupport support) noexcept {
    switch (support) {
    case LayoutSupport::ChannelLastOnly:
        return is_channel_last(layout);
    case LayoutSupport::DenseOnly:
        return layout == TensorLayout::Dense;
    case LayoutSupport::Any:
        return true;
    }
    return false;
}

}  // namespace nnops
