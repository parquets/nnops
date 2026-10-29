#pragma once
/// @file conv2d_implicit_gemm.h
/// @brief Conv2D as implicit GEMM — both memory layouts, one plan.
///
/// Conv2D over an im2col matrix is a GEMM either way; what the layout decides
/// is which operand is which, and that is the only thing the two routes differ
/// in:
///
///   planar (NCHW) — the `planar` route, the file's first half:
///     M = oc_per_group, N = pixels of the oh-block, K = ic_per_group × karea
///     A = the weight, handed to the direct-A MMA as it lies (no pack)
///     B = the activations, gathered straight into uniform packed panels
///         (tiled_im2col_2d_packed — im2col and pack fused into one pass)
///
///   channel-last (NHWC) — the `channel_last` route, the second half:
///     M = pixels, N = oc block, K = karea × ic block, tap-major
///     A = the activation tile (a gather, or the input row itself for 1x1)
///     B = the weight, packed from TensorFlow's HWIO filter
///
/// The swap is what makes channel-last fast: a pixel's channels are contiguous,
/// so a tap is one copy in both the input and the A row, and the weight's
/// contiguous axis is the output channel, so a B panel's k row is one OC-length
/// run with no transposing pack. A planar weight is OIHW, so its k slice is a
/// karea-strided scatter and the pixel axis is what has to be gathered.
///
/// Both routes share everything around the GEMM, and that is what this header
/// holds:
///
///   * one plan (Conv2DGemmPlan) — block sizes, the parallel decomposition and
///     the scratch geometry, resolved once per call by
///     get_conv2d_implicit_gemm_plan. The fields a route does not use stay zero,
///     so the two plans are one struct with a route tag rather than two;
///   * one scratch layout (Conv2DGemmScratch / gemm_scratch_layout) — the
///     per-slot regions both routes address the same way;
///   * one entry (conv2d_implicit_gemm_kernel) that dispatches on the
///     activation's layout, then on the weight dtype (a quantized weight sends
///     either route to its int8 driver).
///
/// Work is split at (batch, group, oh-chunk, oc-chunk) granularity on both
/// routes, and both fold Relu/Relu6 into the MMA clamp when there is no bias and
/// no add_to, so neither pays an epilogue pass for the MobileNet activations.
///
/// Scratch geometry — one arena, one shape for both routes (offsets are bytes,
/// sizes are in {@p esz}-sized elements except where noted):
///
///   [quant plane][shared packed B][slot 0 | slot 1 | …][int32 acc][packed A]
///
/// where a slot is [col | A][add_to residual][weight staging][packed B], and
///   quant plane     the shared fp→int8 activation plane (quant_size), written
///                   once per call on the calling thread before the task loop;
///   shared packed B the channel-last prepack (shared_elems): the whole packed
///                   weight, written once and read in place by every task;
///   col             the planar int8 im2col buffer (col_size);
///   A               the channel-last A region (a_size): the packed-A panels
///                   first (a_pack_stride), then the row-major gather;
///   residual        the add_to original tile (orig_size);
///   staging         the channel-last tap-major weight staging tile;
///   packed B        the per-slot B panels, when not shared (pack_size);
///   int32 acc       the quantized routes' accumulators;
///   packed A        the planar int8 packed weight (pack_a_size).
///
/// Every region is 64-byte aligned, and the offsets are computed by
/// gemm_scratch_layout alone — the plan's workspace_size is its `total`, so the
/// sizing and the kernels' pointers cannot drift.

#include "backend/cpu/matmul/matmul_helper.h"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/ops/conv2d.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace nnops::backend::cpu {

/// Which route a plan was resolved for — the tag the kernel dispatches on.
enum class Conv2DGemmRoute : uint8_t {
    Planar,       ///< activation stored NCHW, weight OIHW [OC][IC/G][KH][KW]
    ChannelLast,  ///< activation stored NHWC, weight HWIO [KH][KW][IC/G][OC]
};

/// Single source of truth for both routes' block sizes, parallel decomposition
/// and scratch geometry. The fields a route does not use stay zero.
struct Conv2DGemmPlan {
    Conv2DGemmRoute route = Conv2DGemmRoute::Planar;

    // ---- both routes ------------------------------------------------------
    int64_t icn_block = 0;    ///< input-channel block (planar: kdim = icn × karea,
                              ///< channel-last: kdim = icn × karea as well, but
                              ///< tap-major)
    int64_t ocn_block = 0;    ///< output-channel block: the planar GEMM's M tile,
                              ///< the channel-last GEMM's N tile
    int64_t oh_chunks = 1;    ///< parallel row chunks per (batch, group)
    int64_t oc_chunks = 1;    ///< oc-chunks per (batch, group), each a task, so
                              ///< the grid is N·G × oh_chunks × oc_chunks
    int64_t num_slots = 0;    ///< scratch slots (worker threads, else tasks)
    int64_t pack_size = 0;    ///< per-slot packed-B elements
    int64_t orig_size = 0;    ///< per-slot add_to residual-tile elements
    int64_t acc_size = 0;     ///< per-slot int32 accumulator elements (both
                              ///< quantized routes; 0 on the fp ones)
    int64_t quant_size = 0;   ///< bytes of the shared quantized-activation plane
                              ///< at the front of the arena (0 when the
                              ///< activation already arrives quantized)
    int64_t shared_elems = 0; ///< shared packed-B elements at the arena head
                              ///< (channel-last prepack; 0 → each task packs its
                              ///< own panels)
    bool pointwise = false;   ///< k1s1p0: the GEMM reads the input tensor itself
                              ///< instead of a gathered tile, so there is no
                              ///< gather and icn_block spans all of ic_per_group
                              ///< (planar: only for a dense input plane)
    size_t workspace_size = 0;///< pooled scratch bytes. 0 means either "nothing to
                              ///< pool" or "unsupported dtype → the kernel tells
                              ///< it apart by the plan's icn_block, never by this"

    // ---- planar only ------------------------------------------------------
    int64_t oh_block = 0;          ///< output-row block (the GEMM's N tile)
    int64_t num_oh_blocks = 1;     ///< ceil(OH / oh_block)
    int64_t col_size = 0;          ///< per-slot im2col elements — the quantized
                                   ///< route only: the fp route's gather is
                                   ///< fused with the pack, and pointwise needs
                                   ///< none (0)
    int64_t pack_a_size = 0;       ///< per-slot packed-weight elements (the
                                   ///< quantized route's A operand)

    // ---- channel-last only ------------------------------------------------
    int64_t m_block = 0;           ///< pixels per GEMM tile (the GEMM's M)
    int64_t ph_block = 1;          ///< rows one accumulator pass covers inside a
                                   ///< task (bounds the live accumulator set)
    int64_t a_size = 0;            ///< per-slot A region elements
    int64_t a_pack_stride = 0;     ///< elements of the packed-A panels at the
                                   ///< front of the A region (0 → fp route)
    int64_t stage_size = 0;        ///< per-slot tap-major weight staging tile
                                   ///< (0 → a 1x1 filter, which packs in place)
};

constexpr int64_t gemm_align64(int64_t v) noexcept { return (v + 63) & ~int64_t{63}; }

/// Elements of one packed-B panel at k-length @p kc. The MMA computes this
/// stride itself from the kc it is handed (mma_pack_align(nr_max·kc·esz), plus
/// the int8 correction block on x86), so every pack must agree with it exactly.
inline int64_t gemm_panel_slot(int64_t kc, int64_t nr_max, int64_t esz,
                               bool quant) noexcept
{
    if (quant) { return int8_panel_stride(kc); }
    return static_cast<int64_t>(mma_pack_align(
        static_cast<size_t>(nr_max * kc) * static_cast<size_t>(esz))) / esz;
}

/// Packed-B panel-set elements for @p panels_per_oc panels at k-length
/// @p kdim — the planar route's whole oh-block, or the channel-last route's one
/// (ic block, oc block) at the whole filter's kdim.
inline int64_t gemm_set_elems(int64_t panels_per_oc, int64_t kdim,
                              int64_t nr_max, int64_t esz, bool quant) noexcept
{
    return panels_per_oc * gemm_panel_slot(kdim, nr_max, esz, quant);
}

/// Scratch geometry — one source of truth for the plan's sizing and both
/// routes' pointers. All offsets are bytes, and the per-slot ones are into the
/// slot. Layout:
///   [quant plane][shared packed B][slot 0 | slot 1 | …][int32 acc][packed A]
/// where a slot is [col | A][add_to residual][weight staging][packed B].
struct Conv2DGemmScratch {
    int64_t quant_off = 0;   ///< the shared quantized-activation plane (at 0)
    int64_t panel_off = 0;   ///< absolute: the shared packed-B region
    int64_t head = 0;        ///< absolute: the per-slot region
    int64_t off_col = 0;     ///< per-slot im2col col matrix (planar quantized)
    int64_t off_a = 0;       ///< per-slot A: the packed-A panels, then the gather
    int64_t off_orig = 0;    ///< per-slot add_to residual region
    int64_t off_stage = 0;   ///< per-slot weight staging tile
    int64_t off_pack = 0;    ///< per-slot packed-B region (when not shared)
    int64_t slot_stride = 0; ///< bytes per slot
    int64_t off_acc = 0;     ///< int32 accumulator region (absolute, after slots)
    int64_t off_pack_a = 0;  ///< packed-weight region (absolute, planar quantized)
    int64_t total = 0;       ///< total arena bytes
};

inline Conv2DGemmScratch gemm_scratch_layout(const Conv2DGemmPlan& p,
                                             int64_t esz) noexcept
{
    Conv2DGemmScratch s;
    s.quant_off = 0;
    s.panel_off = gemm_align64(p.quant_size);
    s.head = gemm_align64(s.panel_off + p.shared_elems * esz);
    s.off_col = 0;
    s.off_a = gemm_align64(p.col_size * esz);
    s.off_orig = gemm_align64(s.off_a + p.a_size * esz);
    s.off_stage = gemm_align64(s.off_orig + p.orig_size * esz);
    s.off_pack = gemm_align64(s.off_stage + p.stage_size * esz);
    s.slot_stride = gemm_align64(s.off_pack + p.pack_size * esz);
    s.off_acc = s.head + p.num_slots * s.slot_stride;
    s.off_pack_a = gemm_align64(
        s.off_acc + p.num_slots * p.acc_size * static_cast<int64_t>(sizeof(int32_t)));
    s.total = s.off_pack_a + p.num_slots * p.pack_a_size;
    return s;
}

/// Resolve the plan (block sizes, parallel decomposition, workspace bytes) from
/// the descriptors. The route follows the activation's layout: NCHW → the
/// planar route, NHWC → the channel-last one (which additionally requires the
/// HWIO filter). Returns a zeroed plan — `icn_block == 0`, i.e. workspace_size
/// is not the signal — for an unsupported dtype combination, which routes the
/// kernel to the reference or to Status::unsupported. Supported on both routes:
///   * f32/f16 on every tensor (the floating-point route);
///   * the quantized route — an s8 weight, a u8/s8/f16/f32 activation, an
///     f32/f16/s8/u8 output and a matching bias (see is_quantized_conv2d: the
///     bias is int32 on an s8/u8 destination, the output dtype otherwise), with
///     add_to supported on the planar route only.
/// @p num_threads drives the oh-chunk split; @p use_thread_slots sizes the
/// scratch per worker thread (thread_id-indexed) instead of per task.
Conv2DGemmPlan get_conv2d_implicit_gemm_plan(const Conv2DAttributes& attrs,
                                             std::span<const TensorDesc> inputs,
                                             std::span<const TensorDesc> outputs,
                                             int num_threads,
                                             bool use_thread_slots = false);

/// Conv2D through the implicit-GEMM route of the activation's layout: f32 /
/// f16, or a quantized weight (s8, per-channel QuantParams) × a u8/s8/f16/f32
/// activation (per-tensor QuantParams) → f32/f16/s8/u8. Scratch is pooled
/// internally (sized by the plan); @p workspace is vestigial.
///
/// Every unsupported combination returns Status::unsupported, which the op turns
/// into the reference kernel.
Status conv2d_implicit_gemm_kernel(const Conv2DAttributes& attrs,
                                   TensorView& output,
                                   std::span<const TensorView> inputs,
                                   const ComputeContext& ctx,
                                   void* workspace);

}  // namespace nnops::backend::cpu
