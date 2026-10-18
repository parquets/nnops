#pragma once
/// @file conv2d_winograd.h
/// @brief Winograd F(2×2, 3×3) Conv2D — k3s1 fast path (planar NCHW, f32/f16).
///
/// Reference: 高洋《"远超"理论浮点峰值》. The 2-D F(2,3) algorithm is 16 batched
/// GEMMs over the transform domain: for each transform position p ∈ [0,16),
///
///     M_p[oc][t] = Σ_ic  U_p[oc][ic] · V_p[ic][t]        t = tile index
///
/// with M = OC, N = tiles, K = IC. U is the prepacked weight (G g Gᵀ, one 4×4
/// kernel per position — [16, OC, IC/G], TensorLayout::PackedWinoWeight); V is
/// the transformed input (Bᵀ d B, one 4×4 window per 2×2 output tile, fully
/// materialised). For each transform position the V slice [K=ic][N=tiles] is
/// packed into uniform nr_max-wide panels (tile_pack_rhs) and the GEMM
/// runs through tile_mma_direct, the same route MatMul uses.
///
/// Structure: two parallel phases per (batch, group).
///   Phase A — input transform, parallel over (tile row, 8-tile column group),
///             writing disjoint V slices.
///   Phase B — parallel over (oc block, tile row, tile block): accumulate the 16
///             GEMMs into M, then transform + write back immediately. M never
///             holds all of OC or all tiles. (V is fully materialised as the
///             reference does; M is blocked because its full size —
///             16·ocn_block·num_tiles, ~3.6 MB for 1×64×56×56 — would blow past
///             the 16 MiB scratch pool and fall back to per-call aligned_alloc.)
///
/// The phases are separate because phase B accumulates over *all* input
/// channels: a phase-B task loops ic inside itself, so it reads V for every ic
/// and V must already be complete.
///
/// Numeric order (each one is load-bearing):
///   * bias is NOT folded into M — M is a transform-domain accumulator and
///     Aᵀ(·)A would distort a folded bias. M starts at zero (zero_mode on the
///     first ic-block), then bias[oc] is added per output element during
///     write-back.
///   * the epilogue runs after the output transform, on the real 2×2 outputs.
///   * clamp is ±inf and zero_mode=false on every accumulating GEMM.
///
/// add_to needs no orig scratch: every output element is written exactly once,
/// so the residual simply reads the pre-existing output in place during
/// write-back.
///
/// This kernel must never route to reference::conv2d_ref — U is rank-3 with
/// dims[0] == 16, and the reference would read it as an OC of 16 and silently
/// compute garbage. A dtype mismatch is a hard assert.
///
/// Winograd is NOT always faster, and this kernel makes no attempt to hide
/// that: the input/output transforms cost a fixed amount per tile per channel,
/// so with few channels the 2.25x multiply saving never pays for them. Measured
/// on this tree (bench_conv2d, interleaved A/B with a null control): f32 gains
/// ~1.1-1.45x at 64 channels and ~1.9x at 256; f16 is a wash at 64 and ~1.8x at
/// 256; and it is *slower* at 32 channels and below — down to ~2.8x slower at 8.
/// Selecting Winograd is therefore the caller's explicit choice
/// (ConvAlgorithm::Winograd — there is no Auto), never something this path
/// decides for them.

#include "nnops/core/tensor_view.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/ops/conv2d.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace nnops::backend::cpu {

/// Single source of truth for the Winograd block sizes, the two-phase parallel
/// decomposition, and the scratch layout.
struct Conv2DWinogradPlan {
    // ---- transform-domain geometry ----
    int64_t num_tile_rows = 0;      ///< TH  = ceil(OH / 2)
    int64_t num_tile_cols = 0;      ///< TW  = ceil(OW / 2)
    int64_t num_tile_cols_pad = 0;  ///< TW_pad = ceil(TW / 8) · 8 (vector width)
    int64_t num_tiles = 0;          ///< TH · TW_pad — the GEMM's N
    int64_t halo_h = 0;             ///< padded input rows  = 2·TH + 2
    int64_t halo_w = 0;             ///< padded input cols  = 2·TW_pad + 2

    // ---- blocking ----
    int64_t icn_block = 0;   ///< input-channel (GEMM K) block
    int64_t ocn_block = 0;   ///< output-channel (GEMM M) block
    int64_t tile_block = 0;  ///< tiles per phase-B task within one tile row (×8)

    int64_t num_oc_blocks = 1;     ///< ceil(oc_per_group / ocn_block)
    int64_t num_tile_blocks = 1;   ///< ceil(TW_pad / tile_block)
    int64_t num_a_tasks = 0;       ///< TH · (TW_pad / 8)
    int64_t num_b_tasks = 0;       ///< num_oc_blocks · TH · num_tile_blocks

    // ---- scratch (elements, not bytes) ----
    int64_t num_slots = 0;   ///< M slots: worker threads, else tasks
    int64_t halo_size = 0;   ///< ic_per_group · halo_h · halo_w
    int64_t v_size = 0;      ///< 16 · ic_per_group · num_tiles
    int64_t m_size = 0;      ///< 16 · ocn_block · tile_block
    int64_t b_size = 0;      ///< uniform packed-B elements per slot
    size_t workspace_size = 0;
};

/// Resolve the Winograd plan (tile geometry, block sizes, two-phase task counts,
/// scratch bytes) from the descriptors. Returns a zeroed plan (workspace_size 0)
/// for unsupported cases (non-f32/f16 dtypes, non-k3s1, groups that the U layout
/// cannot express), which routes the kernel to its assert.
Conv2DWinogradPlan get_conv2d_winograd_plan(const Conv2DAttributes& attrs,
                                            std::span<const TensorDesc> inputs,
                                            std::span<const TensorDesc> outputs,
                                            int num_threads,
                                            bool use_thread_slots = false);

/// Winograd F(2×2, 3×3) Conv2D kernel (f32 / f16, planar NCHW, k3s1).
///
/// inputs[1] must be the prepacked U produced by Conv2D::prepackWeights. Scratch
/// is pooled internally (sized by get_conv2d_winograd_plan); @p workspace is
/// vestigial.
void conv2d_winograd_kernel(const Conv2DAttributes& attrs,
                            TensorView& output,
                            std::span<const TensorView> inputs,
                            const ComputeContext& ctx,
                            void* workspace);

}  // namespace nnops::backend::cpu
