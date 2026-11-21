#pragma once
/// @file tensor_layout.hpp
/// @brief TensorLayout enum for describing data arrangement in memory.
///
/// Spatial activations are NHWC/NDHWC or dense (Dense). A channel-last layout
/// keeps the channel-last logical dims in the descriptor — [N, H, W, C] and
/// [N, D, H, W, C] — so the dims *are* the physical order, the "row" of the
/// pitch model is one spatial line — (n, h) over all W pixels and all C
/// channels — and the channel axis is contiguous.
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

#include <cstddef>
#include <cstdint>

namespace nnops {

/// Memory layout of tensor data.
enum class TensorLayout : uint8_t {
    // ---- spatial activations, 2D / 3D ----
    NHWC      = 2,  ///< logical and stored [N, H, W, C]
    NDHWC     = 3,  ///< logical and stored [N, D, H, W, C]

    // ---- Spatial conv weight ----
    HWIO      = 5,  ///< [KH][KW][IC/G][OC] — TF filter order for spatial
                    ///< activations: the shape *is* H×W×I×O, OC contiguous.

    DHWIO = 16,     ///< [KD][KH][KW][IC/G][OC] — the 3-D twin of HWIO, TF's
                    ///< conv3d filter: the shape *is* D×H×W×I×O, OC contiguous.

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

    // PackedHWIOSme — the same filter for the AArch64 SME tiles: per group,
    // ic block and 64-column oc block, the ZA B layout ([oc block][k][64] with
    // f16 K pairs interleaved), whole-K so compute() needs no per-call B pack.
    // Opaque, like PackedHWIO; only the im2col route reads it.
    PackedHWIOSme = 13,

    // PackedHWIOCuda — the same filter for the CUDA int8 conv: per group, one
    // 64-row by 64-byte panel per (k tile, oc block), each row `LD = BK + 16`
    // bytes so the kernel copies it into shared memory verbatim (the pad lane is
    // never read), then the per-channel filter sums the requant's compensation
    // needs. Whole-K and whole-OC, so compute() gathers nothing. Opaque, like
    // PackedHWIO, and refused by every route but that one.
    PackedHWIOCuda = 14,

    // PackedDHWIO — the 3-D filter for Conv3D::prepackWeights: one sub-blob per
    // group, each that group's [IC/G·KD·KH·KW][OC/G] B matrix. An fp filter is
    // stored as MatMul's whole-K PackedB panels; an s8 one as the group's dense
    // k-major matrix (a u8 or fp activation reads it too, which MatMul's
    // segmented s8 panels — packed for one activation convention — cannot).
    // Opaque outside the Conv3D kernel; 17 because 15 and 16 are taken.
    PackedDHWIO = 17,

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

    // ---- Prepacked matmul RHS, AArch64 SME (f32 / f16 / s8) ----
    //
    // The same idea as PackedB, in the layout the ZA kernels want: 64 contiguous
    // output columns per k, `[⌈N/64⌉][K][64]` for f32 (`[⌈N/64⌉][⌈K/2⌉][128]`
    // for f16's interleaved K pairs, `[⌈N/64⌉][⌈K/4⌉][256]` for s8's four-K
    // groups), tail columns zero-filled. Packed once at load time, it needs no
    // per-call RHS pack. Small M reads the fp forms too — inside one 64-column
    // block the f32 blob is a plain [K][64] row-major matrix, which is exactly
    // what the GEMV wants (ldb = 64, one call per block). The s8 blob is only
    // produced for an integral destination (see MatMul::prepackWeights).
    PackedBSme = 11,

    // ---- Prepacked recurrent weights (Rnn / Gru / Lstm ::prepackWeights) ----
    //
    // Everything the step loop would otherwise rebuild on every call, for every
    // direction: W and R interleaved into the gate-block order and packed into
    // the MMA's nr-wide B panels, plus the per-phase bias vectors (Rbh kept
    // apart under linear_before_reset). Opaque and self-describing — the blob
    // leads with its own header, so a compute() handed one reads it in place
    // instead of running the interleave and the two pack passes.
    //
    // 15, not 11: 11 is PackedBSme, and this tag held the same value until the
    // guard below was added. Numbering is append-only — a new tag takes the next
    // free value and an old one is never renumbered, so a value that outlives a
    // build still means what it meant.
    PackedRecurrent = 15,

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

/// Every tag above, so the check below cannot be silently outgrown: a value
/// left out of this list is a value the collision guard does not see.
constexpr TensorLayout kAllTensorLayouts[] = {
    TensorLayout::NHWC,
    TensorLayout::NDHWC,
    TensorLayout::HWIO,
    TensorLayout::PackedWinoWeightLast,
    TensorLayout::HWCM,
    TensorLayout::PackedHWIO,
    TensorLayout::PackedHWIOSme,
    TensorLayout::PackedHWIOCuda,
    TensorLayout::DHWIO,
    TensorLayout::PackedDHWIO,
    TensorLayout::PackedB,
    TensorLayout::PackedBSme,
    TensorLayout::PackedRecurrent,
    TensorLayout::Dense,
};

/// Two tags sharing a value makes every predicate on them wrong at once —
/// PackedBSme and PackedRecurrent both read 11, which quietly made
/// is_packed_recurrent() true of an SME matmul panel. No rank, no arithmetic
/// and no wire format depends on what these numbers are, only on their being
/// distinct.
static_assert([] {
                  constexpr size_t n = sizeof(kAllTensorLayouts) / sizeof(kAllTensorLayouts[0]);
                  for (size_t i = 0; i < n; ++i) {
                      for (size_t j = i + 1; j < n; ++j) {
                          if (kAllTensorLayouts[i] == kAllTensorLayouts[j]) {
                              return false;
                          }
                      }
                  }
                  return true;
              }(),
              "TensorLayout values must be distinct");

/// True for the spatial layouts, NHWC and NDHWC.
constexpr bool is_channel_last(TensorLayout layout) noexcept {
    return layout == TensorLayout::NHWC || layout == TensorLayout::NDHWC;
}

/// True for HWIO, the conv weight order of a spatial activation.
constexpr bool is_channel_last_weight(TensorLayout layout) noexcept {
    return layout == TensorLayout::HWIO;
}

/// True for DHWIO, the 3-D conv weight order of an NDHWC activation.
constexpr bool is_channel_last_weight_3d(TensorLayout layout) noexcept {
    return layout == TensorLayout::DHWIO;
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
    return layout == TensorLayout::PackedHWIO || layout == TensorLayout::PackedHWIOSme;
}

/// True for the AArch64 SME filter blob (the ZA tiles' B layout).
constexpr bool is_sme_conv_weight(TensorLayout layout) noexcept {
    return layout == TensorLayout::PackedHWIOSme;
}

/// True for the CUDA int8 conv's filter blob. Deliberately outside
/// is_packed_conv_weight: that predicate is what the CPU im2col route accepts,
/// and this blob's bytes are a CUDA tile's, not that route's panels.
constexpr bool is_cuda_conv_weight(TensorLayout layout) noexcept {
    return layout == TensorLayout::PackedHWIOCuda;
}

/// True for Conv3D's prepacked filter; an opaque blob.
constexpr bool is_packed_conv3d_weight(TensorLayout layout) noexcept {
    return layout == TensorLayout::PackedDHWIO;
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
