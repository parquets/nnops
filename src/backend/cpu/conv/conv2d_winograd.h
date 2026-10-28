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
/// the transformed input (Bᵀ d B, one 4×4 window per 2×2 output tile). For each
/// transform position the V slice [K=ic][N=tiles] is packed into uniform
/// nr_max-wide panels (tile_pack_rhs) and the GEMM runs through
/// tile_mma_direct, the same route MatMul uses.
///
/// Structure: ONE dispatch over (batch, group, tile row, tile block). Each task
/// is self-contained — it builds the padded halo strip its tiles read, transforms
/// it into that task's V, runs the 16 GEMMs for every oc block of the group into
/// M, and writes those tiles back. Tile rows and blocks are disjoint, so no task
/// reads another's scratch and the phases need no barrier between them.
///
/// Task-local scratch is what makes that possible: the halo strip, V and M all
/// live in one per-slot block, instead of the kernel-wide halo + V planes the
/// three-phase form materialised. The cost is deliberate duplication at the task
/// edges — the two padded columns neighbouring tile blocks share (~6% of a
/// block's transform) and the two window rows neighbouring tile rows share
/// (~2x on the halo copy, which is a pure copy of the input plane).
///
/// Why not keep the phases and fence them inside one dispatch instead: CpuBackend
/// has no barrier — `run` is just a forwarder to parallel_for, and a ParallelForFn
/// gives no guarantee that every worker visits the body, nor in what order (a
/// pool that partitions statically would never release a spinning worker). A
/// completion counter in the body is therefore a deadlock waiting for a caller.
///
/// Why not keep the phases and fuse only the halo: folding phase A into phase B
/// means every oc-block task repeats the whole input transform for its tiles. At
/// 256 channels that is 4x the transform work, more than a dispatch costs. Rows
/// and blocks are the only axes that partition tiles without repeating work, so
/// they are the only axes in the grid; the oc axis stays inside a task and the
/// batch/group axes join the grid.
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
/// This kernel must never route to conv2d_cpu — U is rank-3 with
/// dims[0] == 16, and the reference would read it as an OC of 16 and silently
/// compute garbage. A dtype mismatch is a hard assert.
///
/// Winograd is NOT always faster, and this kernel makes no attempt to hide
/// that: the input/output transforms cost a fixed amount per tile per channel,
/// so with few channels the 2.25x multiply saving never pays for them. Measured
/// on this tree (bench_conv2d, interleaved A/B against im2col with a null
/// control, medians of 3): f32 is ahead from 32 channels up — ~1.2x at 32,
/// ~1.3-1.45x at 64 and ~1.55-1.6x at 256 — and a wash at 8; f16 reaches
/// ~1.35x at 64 and ~1.6-1.7x at 256. A batch-4 quad is ~1.35-1.45x, because
/// the batch axis is now in the task grid. Selecting Winograd is therefore the
/// caller's explicit choice
/// (ConvAlgorithm::Winograd — there is no Auto), never something this path
/// decides for them.

#include "nnops/core/tensor_view.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/ops/conv2d.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace nnops::backend::cpu {

/// Single source of truth for the Winograd block sizes, the parallel
/// decomposition, and the scratch layout.
struct Conv2DWinogradPlan {
    // ---- transform-domain geometry ----
    int64_t num_tile_rows = 0;      ///< TH  = ceil(OH / 2)
    int64_t num_tile_cols = 0;      ///< TW  = ceil(OW / 2)
    int64_t num_tile_cols_pad = 0;  ///< TW_pad = ceil(TW / 8) · 8 (vector width)
    int64_t num_tiles = 0;          ///< TH · TW_pad — the GEMM's N

    // ---- blocking ----
    int64_t icn_block = 0;   ///< input-channel (GEMM K) block
    int64_t ocn_block = 0;   ///< output-channel (GEMM M) block
    int64_t tile_block = 0;  ///< tiles per task within one tile row (8 or 16)

    int64_t num_oc_blocks = 1;     ///< ceil(oc_per_group / ocn_block)
    int64_t num_tile_blocks = 1;   ///< ceil(TW_pad / tile_block)
    int64_t num_tasks = 0;         ///< N · G · TH · num_tile_blocks

    // ---- scratch (elements, not bytes), all per slot ----
    int64_t num_slots = 0;    ///< worker threads, else tasks
    int64_t halo_ld = 0;      ///< padded columns of a halo strip = 2·tile_block + 2
    int64_t halo_rows = 0;    ///< padded rows of a halo strip = 4 (the window)
    int64_t halo_size = 0;    ///< ic_per_group · halo_rows · halo_ld
    int64_t v_p_stride = 0;   ///< elements between transform positions
    int64_t v_size = 0;       ///< 16 · v_p_stride
    int64_t m_size = 0;       ///< 16 · ocn_block · tile_block
    int64_t b_size = 0;       ///< uniform packed-B elements
    int64_t slot_size = 0;    ///< halo + V + M + B
    size_t workspace_size = 0;
};

/// Resolve the Winograd plan (tile geometry, block sizes, task grid, per-slot
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
Status conv2d_winograd_kernel(const Conv2DAttributes& attrs,
                            TensorView& output,
                            std::span<const TensorView> inputs,
                            const ComputeContext& ctx,
                            void* workspace);

}  // namespace nnops::backend::cpu
