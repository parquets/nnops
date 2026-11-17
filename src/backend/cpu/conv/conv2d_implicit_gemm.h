#pragma once
/// @file conv2d_implicit_gemm.h
/// @brief Conv2D as implicit GEMM over NHWC activations.
///
/// Conv2D over an im2col matrix is a GEMM with the operands laid out for the
/// NHWC activation:
///
///     M = pixels, N = oc block, K = karea × ic block, tap-major
///     A = the activation tile (the input row itself for a 1x1)
///     B = the weight, packed from TensorFlow's HWIO filter
///
/// A pixel's channels are contiguous, so a tap is one copy in both the input and
/// the A row, and the weight's contiguous axis is the output channel, so a B
/// panel's k row is one OC-length run with no transposing pack.
///
/// Shared here:
///   * one plan (Conv2DGemmPlan), resolved once per call by
///     get_conv2d_implicit_gemm_plan;
///   * one scratch layout (Conv2DGemmScratch / gemm_scratch_layout): every
///     region 64-byte aligned, all offsets computed in that one function;
///   * the prepacked-filter geometry (ConvPackedWeight) — a function of the
///     attributes and the channel counts alone, no activation and no thread
///     count. The plan adopts its block sizes, so the panels prepackWeights
///     builds and the panels the kernel addresses are one decision;
///   * one entry (conv2d_implicit_gemm_kernel) that dispatches on the weight
///     dtype.
///
/// The route splits work at (batch, group, oh-chunk, oc-chunk) granularity and
/// folds Relu/Relu6 into the MMA clamp when there is no add_to (add_to adds the
/// residual after the activation, so it keeps an epilogue pass).
///
/// Scratch arena (offsets are bytes, sizes are in esz-sized elements unless
/// noted):
///   [quant plane][shared packed B][slot 0 | slot 1 | …][int32 acc]
/// A slot is [A][add_to residual][weight staging][packed B]: the A region
/// (packed-A panels, then the row-major gather), the add_to original tile, the
/// staging tile, then the per-slot B panels. Before the slots sit the quantized
/// activation plane and the shared prepack.
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

/// Block sizes, parallel decomposition and scratch geometry.
struct Conv2DGemmPlan {
    // ---- plan -------------------------------------------------------------
    int64_t icn_block = 0;     ///< input-channel block (kdim = icn_block × karea)
    int64_t ocn_block = 0;     ///< output-channel block (the GEMM's N tile)
    int64_t oh_chunks = 1;     ///< parallel row chunks per (batch, group)
    int64_t oc_chunks = 1;     ///< filter chunks per (batch, group); the grid is
                               ///< N·G × oh_chunks × oc_chunks
    int64_t num_slots = 0;     ///< scratch slots (worker threads, else tasks)
    int64_t pack_size = 0;     ///< per-slot packed-B elements
    int64_t orig_size = 0;     ///< per-slot add_to residual-tile elements
    int64_t acc_size = 0;      ///< per-slot int32 accumulator (0 on fp)
    int64_t quant_size = 0;    ///< bytes of the shared activation plane at the
                               ///< arena head (0 when no per-call plane is
                               ///< materialized)
    int64_t shared_elems = 0;  ///< shared packed-B elements at the arena head
                               ///< (0 → the filter is read from the caller's
                               ///< blob, or each task packs its own panels)
    bool pointwise = false;    ///< k1s1p0: the GEMM reads the input tensor itself,
                               ///< so no gather and icn_block spans all of
                               ///< ic_per_group
    size_t workspace_size = 0; ///< pooled scratch bytes; 0 is "nothing to pool",
                               ///< never "unsupported" — that is icn_block == 0

    // ---- GEMM tiles and per-slot regions ----------------------------------
    int64_t m_block = 0;       ///< pixels per GEMM tile (M)
    int64_t ph_block = 1;      ///< rows per accumulator pass inside a task
                               ///< (bounds the live accumulator set)
    int64_t a_size = 0;        ///< per-slot A region elements
    int64_t a_pack_stride = 0; ///< packed-A panels at the front of the A
                               ///< region (0 on fp)
    int64_t stage_size = 0;    ///< per-slot tap-major weight staging tile
                               ///< (0 when no task packs: a 1x1 packs in
                               ///< place, and a filter packed once — shared
                               ///< or prepacked — never restages)
};

constexpr int64_t gemm_align64(int64_t v) noexcept { return (v + 63) & ~int64_t{63}; }

/// Elements of one packed-B panel at k-length @p kc. The MMA derives this
/// stride itself from kc (mma_pack_align(nr_max·kc·esz), plus the int8
/// correction block on x86), so every pack must agree with it exactly.
///
/// bf16's panel is BFMMLA's k-group layout rather than a dense nr_max×kc
/// rectangle: ⌈kc/4⌉ groups of [colpair][8] for a 16-wide panel. Its k rounds
/// up to the group, so the panel is wider than nr_max·kc when kc % 4 ≠ 0 —
/// passing @p bf16 is what keeps the packer and the kernel's own advance equal.
inline int64_t gemm_panel_slot(int64_t kc, int64_t nr_max, int64_t esz, bool quant,
                               bool bf16 = false) noexcept
{
    if (quant) {
        return int8_panel_stride(kc);
    }
#ifdef NNOPS_ARCH_AARCH64
    if (bf16) {
        (void)nr_max;
        (void)esz;
        return arch::bf16_b_panel_stride(kc);
    }
#else
    (void)bf16;
#endif
    return static_cast<int64_t>(
               mma_pack_align(static_cast<size_t>(nr_max * kc) * static_cast<size_t>(esz))) /
           esz;
}

/// Packed-B panel-set elements: @p panels_per_oc panels at k-length @p kdim —
/// one (ic block, oc block) at the whole filter's kdim.
inline int64_t gemm_set_elems(int64_t panels_per_oc, int64_t kdim, int64_t nr_max, int64_t esz,
                              bool quant, bool bf16 = false) noexcept
{
    return panels_per_oc * gemm_panel_slot(kdim, nr_max, esz, quant, bf16);
}

/// Geometry of the prepacked filter (PackedHWIO): the blob is
/// [group][oc block][k block][panel][k][nr], panels of `nr` output channels.
/// The two outer axes are the reader's own loop nest — the driver walks oc block
/// → pixel pass → k block → panel ([conv2d_implicit_gemm.cpp], the
/// task) — so one oc block's whole filter is one contiguous run taken in order,
/// and a cold filter streams once instead of in `panels_oc`-strided slices.
/// The trailing k block packs a shorter k, hence a shorter panel stride.
struct ConvPackedWeight {
    int64_t nr = 0;           ///< panel width (output channels per panel)
    int64_t icn_block = 0;    ///< k block
    int64_t ocn_block = 0;    ///< oc block (the plan's N tile)
    int64_t kdim_max = 0;     ///< icn_block × karea
    int64_t panels_oc = 0;    ///< oc blocks × block_panels — the group's panels
    int64_t slot_full = 0;    ///< elements of one panel at kdim_max
    int64_t slot_tail = 0;    ///< ... at the trailing k block (0 when none)
    int64_t nfull_ic = 0;     ///< whole k blocks per group
    int64_t tail_ic = 0;      ///< input channels in the trailing k block
    int64_t col_elems = 0;    ///< one panel's whole k column
    int64_t block_panels = 0; ///< ⌈ocn_block / nr⌉ — panels in an oc block
    int64_t group_elems = 0;  ///< one group's panels
    int64_t sums_off = 0;     ///< element offset of the sums table
    int64_t total = 0;        ///< all groups' panels, then the sums table

    // A quantized blob carries Σ_k weight[oc, k] per output channel — the
    // epilogue's R_b, which the raw path re-derives from the filter on every
    // call. It is a filter property, so prepacking is where it belongs, and it
    // is layout-free: a plain int32[OC] table after the panels, computed from
    // the raw filter, never from the packed bytes (whose format differs per
    // arch).

    /// Element offset of the panel holding output channel @p oc (panel-aligned)
    /// at k block @p ic_block, in group @p g. Panels of one oc block at one k
    /// block are `slot` apart, and the next k block starts `block_panels·slot`
    /// later — so a whole oc block is read straight through.
    int64_t offset(int64_t g, int64_t ic_block, int64_t oc) const noexcept
    {
        const int64_t koff = ic_block >= nfull_ic ?
                                 nfull_ic * slot_full + (ic_block - nfull_ic) * slot_tail :
                                 ic_block * slot_full;
        const int64_t blk = oc / ocn_block;
        const int64_t pin = (oc - blk * ocn_block) / nr;
        return g * group_elems + blk * block_panels * col_elems + block_panels * koff +
               pin * slot(ic_block);
    }

    /// Elements of one panel at k block @p ic_block (neighbouring panels in that
    /// column are this far apart).
    int64_t slot(int64_t ic_block) const noexcept
    {
        return ic_block >= nfull_ic ? slot_tail : slot_full;
    }
};

/// Resolve the prepacked geometry; it is also the im2col plan's block sizes
/// (the plan reads icn_block/ocn_block from here), so the
/// panels it sizes and the panels prepackWeights builds cannot disagree.
ConvPackedWeight conv_packed_weight(const Conv2DAttributes& attrs, int64_t ic_per_group,
                                             int64_t oc_per_group, DataType weight_dtype);

/// The blob @p weight prepacks into — the query half of the two-step prepack,
/// what Conv2D::prepackWeights returns. Shape [OC, ⌈elems / OC⌉] with an
/// explicit pitch, so numel() is exactly the buffer the pack writes; rank 0 for
/// a filter with no prepacked form.
TensorDesc conv_packed_weight_desc(const Conv2DAttributes& attrs, const TensorDesc& weight);

/// Fill @p packed — sized from conv_packed_weight_desc — from the raw filter.
void conv_packed_weight_pack(const Conv2DAttributes& attrs, const TensorView& weight,
                             TensorView& packed, const ComputeContext& ctx);

/// Byte offsets into the arena (see the file header); the per-slot ones are
/// relative to the slot.
struct Conv2DGemmScratch {
    int64_t quant_off = 0;   ///< the shared quantized-activation plane (at 0)
    int64_t panel_off = 0;   ///< absolute: the shared packed-B region
    int64_t head = 0;        ///< absolute: the per-slot region
    int64_t off_a = 0;       ///< per-slot A: the packed-A panels, then the gather
    int64_t off_orig = 0;    ///< per-slot add_to residual region
    int64_t off_stage = 0;   ///< per-slot weight staging tile
    int64_t off_pack = 0;    ///< per-slot packed-B region (when not shared)
    int64_t slot_stride = 0; ///< bytes per slot
    int64_t off_acc = 0;     ///< int32 accumulator region (absolute, after slots)
    int64_t total = 0;       ///< total arena bytes
};

inline Conv2DGemmScratch gemm_scratch_layout(const Conv2DGemmPlan& p, int64_t esz) noexcept
{
    Conv2DGemmScratch s;
    s.quant_off = 0;
    s.panel_off = gemm_align64(p.quant_size);
    s.head = gemm_align64(s.panel_off + p.shared_elems * esz);
    s.off_a = 0;
    s.off_orig = gemm_align64(s.off_a + p.a_size * esz);
    s.off_stage = gemm_align64(s.off_orig + p.orig_size * esz);
    s.off_pack = gemm_align64(s.off_stage + p.stage_size * esz);
    s.slot_stride = gemm_align64(s.off_pack + p.pack_size * esz);
    s.off_acc = s.head + p.num_slots * s.slot_stride;
    s.total = gemm_align64(s.off_acc + p.num_slots * p.acc_size *
                                                static_cast<int64_t>(sizeof(int32_t)));
    return s;
}

/// Resolve the plan (block sizes, parallel decomposition, workspace bytes) from
/// the descriptors. The im2col route needs the HWIO filter or its
/// prepacked blob. A zeroed plan — `icn_block == 0`, not workspace_size — is the
/// unsupported-dtype answer, which the kernel turns into Status::unsupported.
/// Supported: f32/f16 on every tensor, or the int8 route — an s8 weight, a
/// u8/s8/f16/f32 activation, an f32/f16/s8/u8 output and a matching bias (int32
/// on an s8/u8 destination, the output dtype otherwise), with no add_to.
/// @p num_threads drives the oh-chunk split; @p use_thread_slots sizes the
/// scratch per worker thread instead of per task. @p num_slots is how many
/// distinct ids `current_thread_id()` can report (0 → num_threads); with thread
/// slots the scratch is min(num_slots, tasks). `num_threads` may legitimately
/// exceed the worker count, so the scratch must not follow it.
Conv2DGemmPlan get_conv2d_implicit_gemm_plan(const Conv2DAttributes& attrs,
                                             std::span<const TensorDesc> inputs,
                                             std::span<const TensorDesc> outputs, int num_threads,
                                             bool use_thread_slots = false, int64_t num_slots = 0);

/// Conv2D through the implicit-GEMM route: f32/f16, or an s8 weight × a
/// u8/s8/f16/f32 activation → f32/f16/s8/u8. Scratch is
/// pooled internally; @p workspace is vestigial. Every unsupported combination
/// returns Status::unsupported, which the op turns into the reference kernel.
Status conv2d_implicit_gemm_kernel(const Conv2DAttributes& attrs, TensorView& output,
                                   std::span<const TensorView> inputs, const ComputeContext& ctx,
                                   void* workspace);

} // namespace nnops::backend::cpu
