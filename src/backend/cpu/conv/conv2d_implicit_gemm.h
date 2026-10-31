#pragma once
/// @file conv2d_implicit_gemm.h
/// @brief Conv2D as implicit GEMM — both memory layouts, one plan.
///
/// Conv2D over an im2col matrix is a GEMM either way; the layout decides the
/// operand roles:
///
///   planar (NCHW):
///     M = oc_per_group, N = pixels of the oh-block, K = ic_per_group × karea
///     A = the weight (direct-A on fp, packed into A panels on int8)
///     B = the activations, gathered into uniform packed panels (on fp the
///         gather and the pack are one fused pass)
///
///   channel-last (NHWC):
///     M = pixels, N = oc block, K = karea × ic block, tap-major
///     A = the activation tile (the input row itself for a 1x1)
///     B = the weight, packed from TensorFlow's HWIO filter
///
/// The swap is what makes channel-last fast: a pixel's channels are contiguous,
/// so a tap is one copy in both the input and the A row, and the weight's
/// contiguous axis is the output channel, so a B panel's k row is one OC-length
/// run with no transposing pack. A planar OIHW weight's k slice is instead a
/// karea-strided scatter, so the pixel axis is what has to be gathered.
///
/// Shared here:
///   * one plan (Conv2DGemmPlan), resolved once per call by
///     get_conv2d_implicit_gemm_plan; a route's unused fields stay zero, so the
///     two plans are one struct with a route tag;
///   * one scratch layout (Conv2DGemmScratch / gemm_scratch_layout): every
///     region 64-byte aligned, all offsets computed in that one function;
///   * the prepacked-filter geometry (ConvPackedWeightLast / _Planar) — a
///     function of the attributes and the channel counts alone, no activation
///     and no thread count. The plan adopts its block sizes, so the panels
///     prepackWeights builds and the panels the kernel addresses are one
///     decision;
///   * one entry (conv2d_implicit_gemm_kernel) that dispatches on the
///     activation's layout, then on the weight dtype.
///
/// Both routes split work at (batch, group, oh-chunk, oc-chunk) granularity and
/// fold Relu/Relu6 into the MMA clamp when there is no add_to (add_to adds the
/// residual after the activation, so it keeps an epilogue pass).
///
/// Scratch arena, one shape for both routes (offsets are bytes, sizes are in
/// esz-sized elements unless noted):
///   [quant plane][shared packed B][slot 0 | slot 1 | …][int32 acc][packed A]
/// A slot is [col | A][add_to residual][weight staging][packed B]: the planar
/// int8 col matrix, the channel-last A region (packed-A panels, then the
/// row-major gather), the add_to original tile, the channel-last staging tile,
/// then the per-slot B panels. Before the slots sit the quantized activation
/// plane and the channel-last shared prepack.
///
/// gemm_scratch_layout alone computes the offsets, and its `total` is the plan's
/// workspace_size, so sizing and the kernels' pointers cannot drift.

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

/// Both routes' block sizes, parallel decomposition and scratch geometry;
/// unused fields stay zero.
struct Conv2DGemmPlan {
    Conv2DGemmRoute route = Conv2DGemmRoute::Planar;

    // ---- both routes ------------------------------------------------------
    int64_t icn_block = 0;    ///< input-channel block (kdim = icn_block × karea on
                              ///< both routes)
    int64_t ocn_block = 0;    ///< output-channel block: the planar GEMM's M tile,
                              ///< the channel-last GEMM's N tile
    int64_t oh_chunks = 1;    ///< parallel row chunks per (batch, group)
    int64_t oc_chunks = 1;    ///< filter chunks per (batch, group); the grid is
                              ///< N·G × oh_chunks × oc_chunks
    int64_t num_slots = 0;    ///< scratch slots (worker threads, else tasks)
    int64_t pack_size = 0;    ///< per-slot packed-B elements
    int64_t orig_size = 0;    ///< per-slot add_to residual-tile elements
    int64_t acc_size = 0;     ///< per-slot int32 accumulator (0 on fp)
    int64_t quant_size = 0;   ///< bytes of the shared activation plane at the
                              ///< arena head (0 when no per-call plane is
                              ///< materialized)
    int64_t shared_elems = 0; ///< shared packed-B elements at the arena head
                              ///< (channel-last prepack; 0 → the filter is read
                              ///< from the caller's blob, or each task packs its
                              ///< own panels)
    bool pointwise = false;   ///< k1s1p0: the GEMM reads the input tensor itself,
                              ///< so no gather and icn_block spans all of
                              ///< ic_per_group (planar: only for a dense plane)
    size_t workspace_size = 0;///< pooled scratch bytes; 0 is "nothing to pool",
                              ///< never "unsupported" — that is icn_block == 0

    // ---- planar only ------------------------------------------------------
    int64_t oh_block = 0;          ///< output-row block (the planar GEMM's N tile);
                                   ///< 1 when the destination's rows are not
                                   ///< back to back (a padded pitch)
    int64_t num_oh_blocks = 1;     ///< ceil(OH / oh_block)
    int64_t col_size = 0;          ///< per-slot im2col elements — int8 only (fp
                                   ///< fuses the gather with the pack; pointwise
                                   ///< needs none)
    int64_t pack_a_size = 0;       ///< per-slot packed-weight elements (the int8
                                   ///< route's A operand)

    // ---- channel-last only ------------------------------------------------
    int64_t m_block = 0;           ///< pixels per GEMM tile (the channel-last M)
    int64_t ph_block = 1;          ///< rows per accumulator pass inside a task
                                   ///< (bounds the live accumulator set)
    int64_t a_size = 0;            ///< per-slot A region elements
    int64_t a_pack_stride = 0;     ///< packed-A panels at the front of the A
                                   ///< region (0 on fp)
    int64_t stage_size = 0;        ///< per-slot tap-major weight staging tile
                                   ///< (0 when no task packs: a 1x1 packs in
                                   ///< place, and a filter packed once — shared
                                   ///< or prepacked — never restages)
};

constexpr int64_t gemm_align64(int64_t v) noexcept { return (v + 63) & ~int64_t{63}; }

/// Elements of one packed-B panel at k-length @p kc. The MMA derives this
/// stride itself from kc (mma_pack_align(nr_max·kc·esz), plus the int8
/// correction block on x86), so every pack must agree with it exactly.
inline int64_t gemm_panel_slot(int64_t kc, int64_t nr_max, int64_t esz,
                               bool quant) noexcept
{
    if (quant) { return int8_panel_stride(kc); }
    return static_cast<int64_t>(mma_pack_align(
        static_cast<size_t>(nr_max * kc) * static_cast<size_t>(esz))) / esz;
}

/// Packed-B panel-set elements: @p panels_per_oc panels at k-length @p kdim —
/// the planar route's whole oh-block, or the channel-last route's one
/// (ic block, oc block) at the whole filter's kdim.
inline int64_t gemm_set_elems(int64_t panels_per_oc, int64_t kdim,
                              int64_t nr_max, int64_t esz, bool quant) noexcept
{
    return panels_per_oc * gemm_panel_slot(kdim, nr_max, esz, quant);
}

/// Geometry of the channel-last prepacked filter (PackedHWIO): one panel column
/// per k block, panels of `nr` output channels tiled across the group's oc axis.
/// The trailing k block packs a shorter k, hence a shorter panel stride, and the
/// columns are laid out group by group.
struct ConvPackedWeightLast {
    int64_t nr = 0;          ///< panel width (output channels per panel)
    int64_t icn_block = 0;   ///< k block
    int64_t ocn_block = 0;   ///< oc block (the plan's N tile)
    int64_t kdim_max = 0;    ///< icn_block × karea
    int64_t panels_oc = 0;   ///< ⌈oc_per_group / nr⌉ — panels in one k column
    int64_t slot_full = 0;   ///< elements of one panel at kdim_max
    int64_t slot_tail = 0;   ///< ... at the trailing k block (0 when none)
    int64_t nfull_ic = 0;    ///< whole k blocks per group
    int64_t tail_ic = 0;     ///< input channels in the trailing k block
    int64_t group_elems = 0; ///< one group's panels
    int64_t sums_off = 0;    ///< element offset of the sums table
    int64_t total = 0;       ///< all groups' panels, then the sums table

    // A quantized blob carries Σ_k weight[oc, k] per output channel — the
    // epilogue's R_b, which the raw path re-derives from the filter on every
    // call. It is a filter property, so prepacking is where it belongs, and it
    // is layout-free: a plain int32[OC] table after the panels, computed from
    // the raw filter, never from the packed bytes (whose format differs per
    // arch).

    /// Element offset of k block @p ic_block in group @p g's panel column.
    int64_t offset(int64_t g, int64_t ic_block) const noexcept
    {
        const int64_t col = ic_block >= nfull_ic
            ? nfull_ic * panels_oc * slot_full +
                  (ic_block - nfull_ic) * panels_oc * slot_tail
            : ic_block * panels_oc * slot_full;
        return g * group_elems + col;
    }

    /// Elements of one panel at k block @p ic_block (neighbouring panels in that
    /// column are this far apart).
    int64_t slot(int64_t ic_block) const noexcept
    {
        return ic_block >= nfull_ic ? slot_tail : slot_full;
    }
};

/// Resolve the channel-last prepacked geometry; it is also the channel-last
/// plan's block sizes (the plan reads icn_block/ocn_block from here), so the
/// panels it sizes and the panels prepackWeights builds cannot disagree.
ConvPackedWeightLast conv_packed_weight_last(const Conv2DAttributes& attrs,
                                             int64_t ic_per_group,
                                             int64_t oc_per_group,
                                             DataType weight_dtype);

/// Geometry of the planar prepacked filter (PackedOIHW). Two forms, one tag: an
/// int8 filter packs into the MMA's A panels (the per-task pack the kernel
/// otherwise repeats); a float one is already the GEMM's A matrix and only gets
/// a densifying copy (`dense`, and every field besides `total` is then unused).
struct ConvPackedWeightPlanar {
    bool dense = false;      ///< the blob is the filter's own [OC][K] matrix
    int64_t icn_block = 0;   ///< k block
    int64_t ocn_block = 0;   ///< oc block
    int64_t kdim_max = 0;    ///< icn_block × karea
    int64_t num_oc_blocks = 0;
    int64_t nfull_ic = 0;
    int64_t tail_ic = 0;
    int64_t set_full = 0;    ///< elements of one (k block, oc block) panel set
    int64_t set_tail = 0;
    int64_t group_elems = 0;
    int64_t sums_off = 0;    ///< element offset of the sums table
    int64_t total = 0;       ///< all groups' panel sets, then the sums table

    // A quantized blob carries Σ_k weight[oc, k] per output channel — the
    // epilogue's R_a — for the same reason the channel-last one carries R_b: a
    // filter property, and a plain int32[OC] table is the only form both arches
    // can read without knowing a panel format.

    /// Element offset of the panel set for (@p ic_block, @p oc_block) in group
    /// @p g.
    int64_t offset(int64_t g, int64_t ic_block, int64_t oc_block) const noexcept;
};

ConvPackedWeightPlanar conv_packed_weight_planar(const Conv2DAttributes& attrs,
                                                 int64_t ic_per_group,
                                                 int64_t oc_per_group,
                                                 DataType weight_dtype);

/// The blob @p weight prepacks into — the query half of the two-step prepack,
/// what Conv2D::prepackWeights returns. Shape [OC, ⌈elems / OC⌉] with an
/// explicit pitch, so numel() is exactly the buffer the pack writes; rank 0 for
/// a filter with no prepacked form.
TensorDesc conv_packed_weight_desc(const Conv2DAttributes& attrs,
                                   const TensorDesc& weight);

/// Fill @p packed — sized from conv_packed_weight_desc — from the raw filter.
void conv_packed_weight_pack(const Conv2DAttributes& attrs,
                             const TensorView& weight,
                             TensorView& packed,
                             const ComputeContext& ctx);

/// Byte offsets into the arena (see the file header); the per-slot ones are
/// relative to the slot.
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
/// the descriptors. The route follows the activation's layout: NCHW → planar,
/// NHWC → channel-last (which additionally needs the HWIO filter or its
/// prepacked blob). A zeroed plan — `icn_block == 0`, not workspace_size — is
/// the unsupported-dtype answer, which the kernel turns into
/// Status::unsupported. Supported on both routes: f32/f16 on every tensor, or
/// the int8 route — an s8 weight, a u8/s8/f16/f32 activation, an f32/f16/s8/u8
/// output and a matching bias (int32 on an s8/u8 destination, the output dtype
/// otherwise), with no add_to.
/// @p num_threads drives the oh-chunk split; @p use_thread_slots sizes the
/// scratch per worker thread instead of per task. @p num_slots is how many
/// distinct ids `current_thread_id()` can report (0 → num_threads); with thread
/// slots the scratch is min(num_slots, tasks). `num_threads` may legitimately
/// exceed the worker count, so the scratch must not follow it.
Conv2DGemmPlan get_conv2d_implicit_gemm_plan(const Conv2DAttributes& attrs,
                                             std::span<const TensorDesc> inputs,
                                             std::span<const TensorDesc> outputs,
                                             int num_threads,
                                             bool use_thread_slots = false,
                                             int64_t num_slots = 0);

/// Conv2D through the implicit-GEMM route of the activation's layout: f32/f16,
/// or an s8 weight × a u8/s8/f16/f32 activation → f32/f16/s8/u8. Scratch is
/// pooled internally; @p workspace is vestigial. Every unsupported combination
/// returns Status::unsupported, which the op turns into the reference kernel.
Status conv2d_implicit_gemm_kernel(const Conv2DAttributes& attrs,
                                   TensorView& output,
                                   std::span<const TensorView> inputs,
                                   const ComputeContext& ctx,
                                   void* workspace);

}  // namespace nnops::backend::cpu
