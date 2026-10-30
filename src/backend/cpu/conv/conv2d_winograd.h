#pragma once
/// @file conv2d_winograd.h
/// @brief Winograd F(2×2, 3×3) Conv2D — k3s1 fast path, both layouts (f32/f16).
///
/// The transform algebra is the same for both layouts and only the GEMM's
/// operands swap, exactly as Conv2D's implicit-GEMM routes swap them:
///
///   planar (NCHW) — A = U plane [oc][ic] handed to the direct-A MMA as it
///     lies, B = V [ic][tiles] packed per call, M = [oc][tiles];
///   channel-last (NHWC) — tile-major: M = tiles, N = oc block, K = ic, so
///     A = V [tiles][ic] (the transform writes it tile-major, channel-
///     contiguous, which is what the gather-free layout gives for free), B =
///     U panels [ic][nr], prepacked once at load, and M = [tiles][oc] so the
///     output transform reads whole channel vectors instead of striding down
///     a column.
///
/// Reference: 高洋《"远超"理论浮点峰值》. The 2-D F(2,3) algorithm is 16 batched
/// GEMMs over the transform domain: for each transform position p ∈ [0,16),
///
///     M_p[oc][t] = Σ_ic  U_p[oc][ic] · V_p[ic][t]        t = tile index
///
/// with M = OC, N = tiles, K = IC on the planar route — and the same product
/// transposed, M_p[t][oc] = Σ_ic V_p[t][ic] · U_p[ic][oc], channel-last. U is
/// the prepacked weight (G g Gᵀ, one 4×4 kernel per position): [16, OC, IC/G]
/// as TensorLayout::PackedWinoWeight for planar, the GEMM's own B panels as
/// PackedWinoWeightLast for channel-last. V is the transformed input (Bᵀ d B,
/// one 4×4 window per 2×2 output tile). Every GEMM runs through
/// tile_mma_direct — raw A, packed B — the same route MatMul uses.
///
/// Structure: ONE dispatch over (batch, group, tile row, tile block, oc-chunk).
/// Each task is self-contained — it builds the padded halo strip its tiles read,
/// transforms it into that task's V, runs the 16 GEMMs for its slice of the
/// filter into M, and writes those tiles back. Tile rows, blocks and oc ranges
/// are disjoint, so no task reads another's scratch and the phases need no
/// barrier between them.
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
/// means every oc-block task repeats the whole input transform for its tiles.
/// Rows and blocks are the axes that partition tiles without repeating any work,
/// so they lead the grid, and the batch/group axes join them.
///
/// The oc axis is the exception, taken only when the pool would otherwise idle:
/// it re-does the halo and the input transform per chunk (the two things the
/// task-local scratch exists to do once), so it is spent last and stops at two
/// M-panels of work per task. It is what a 14x14 plane needs — 7 tile rows and
/// one 8-wide tile block leave the tile axis nothing to give, and on this host
/// 20 threads ran ResNet-18's layer3.x faster at 4 threads than at 20 until the
/// filter rows were split (bench_conv2d conv2d_wino_f32_14x14_256ch). The
/// duplicated transform measures ~1% of such a task, not the 4x the first
/// estimate feared; the reason to spend it last is that it is the only axis that
/// repeats anything at all.
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
/// the batch axis is now in the task grid. So whether a conv is worth
/// transforming is the caller's decision, taken once when it prepacks —
/// winograd_applicable() is that policy (the caller allows it, k3s1, at least
/// 32 input channels per group, at least a 14x14 input); ConvAlgorithm::Auto
/// then only follows the weight it is given.
///
/// ---- the channel-last route -----------------------------------------------
///
/// A channel-last activation turns both transforms into channel-vector work: a
/// 4×4 window is sixteen loads of one channel vector each, and an output tile
/// is four stores of one channel vector — where the planar route needs a
/// deinterleaving load (two adjacent tiles per vector, never strided rows) and
/// scalar strided column stores. So the transforms here run across the channel
/// axis, every tile is independent, and the GEMM flips to tile-major. That flip
/// is also why U is prepacked into its B panels: a per-call pack is a pass over
/// 16·IC·OC, which for the wide layers that make Winograd worth it is a large
/// fraction of the GEMM it feeds, while at load time it is free.
///
/// Tiles are flattened, so a task's chunk may cross a tile row: nothing here
/// needs horizontally adjacent tiles (that is the planar transform's
/// deinterleave constraint), and flattening is what keeps a 14x14 plane — 49
/// tiles — from ending in a straggler chunk.
///
/// The scratch is V and M only: no halo strip (a window is read straight from
/// the input; the two columns neighbouring tiles share are already in L1), and
/// no packed-B block (the prepacked U is read in place, shared by every task).
///
/// Numeric order is the planar route's: M starts at zero, bias and epilogue are
/// applied to the real 2×2 outputs during write-back, and every accumulating
/// GEMM clamps at ±inf.
///
/// Measured against the channel-last implicit GEMM on the same layer
/// (bench_conv2d_nhwc, interleaved A/B with a null control, medians of 3), this
/// route is ahead from 32 channels up by more than the planar one is: 0.47-0.73x
/// of its time at 1T and 0.26-0.84x at 4T/20T for f32, 0.46-0.63x and
/// 0.32-0.75x for f16, over 7x7 to 224x224. The reason it beats the planar
/// route's margin rather than trailing it is that the two costs the planar
/// thresholds are about — the deinterleaving load and the 8-lane tile packing —
/// do not exist here, so the crossover sits lower. Only a 3-channel stem
/// still loses (2.1-3.3x), which winograd_applicable() never lets through.

#include "nnops/core/tensor_view.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/ops/conv2d.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace nnops::backend::cpu {

/// Which layout a plan was resolved for — the tag the kernel dispatches on.
enum class Conv2DWinogradRoute : uint8_t {
    Planar,       ///< activation stored NCHW, U [16][OC][IC/G]
    ChannelLast,  ///< activation stored NHWC, U as B panels [16][nb][G·jb][ldd]
};

/// Geometry of the channel-last U (TensorLayout::PackedWinoWeightLast).
///
/// Resolved by the same function at prepack time and inside the kernel, so the
/// producer and the consumer of those panels cannot drift. One panel is
/// [kc][nr] k-major — exactly what the packer would have written — and the
/// output channels run zero-padded to whole panels, so a partial panel's dead
/// columns are computed and dropped.
struct WinoChannelLastU {
    int64_t nr = 0;     ///< panel width: the arch's nr_max for the dtype
    int64_t kc = 0;     ///< k-block length: the largest divisor of IC/G <= 128
    int64_t nb = 0;     ///< k-blocks per position = IC/G / kc
    int64_t jb = 0;     ///< oc panels per group = ceil(oc_per_group / nr)
    int64_t jbg = 0;    ///< oc panels over all groups = groups · jb
    int64_t ldd = 0;    ///< elements per panel (kc·nr, padded to the pack align)
    int64_t plane = 0;  ///< elements per transform position = nb · jbg · ldd
    int64_t total = 0;  ///< 16 · plane — the whole U
};

/// Resolve the channel-last U geometry for a dtype (see WinoChannelLastU).
WinoChannelLastU wino_channel_last_u(int64_t ic_per_group, int64_t oc_per_group,
                                     int64_t groups, DataType dt) noexcept;

/// Single source of truth for the Winograd block sizes, the parallel
/// decomposition, and the scratch layout. The fields a route does not use stay
/// zero.
struct Conv2DWinogradPlan {
    Conv2DWinogradRoute route = Conv2DWinogradRoute::Planar;

    // ---- transform-domain geometry ----
    int64_t num_tile_rows = 0;      ///< TH  = ceil(OH / 2)
    int64_t num_tile_cols = 0;      ///< TW  = ceil(OW / 2)
    int64_t num_tile_cols_pad = 0;  ///< planar: TW_pad = ceil(TW / 8) · 8
                                    ///< channel-last: the total tile count
                                    ///< TH · TW (they are flattened there)
    int64_t num_tiles = 0;          ///< planar: TH · TW_pad — the GEMM's N

    // ---- blocking ----
    int64_t icn_block = 0;   ///< planar: input-channel (GEMM K) block
                             ///< channel-last: the U's k-block (kc)
    int64_t ocn_block = 0;   ///< planar: output-channel (GEMM M) block
                             ///< channel-last: oc block (GEMM N) — a multiple
                             ///< of the panel width
    int64_t tile_block = 0;  ///< tiles per task: 8 or 16 within one tile row
                             ///< (planar), up to 24 flattened (channel-last)

    int64_t num_tile_blocks = 1;   ///< ceil(tiles / tile_block)
    int64_t oc_chunks = 1;         ///< parallel filter-row chunks per plane, i.e.
                                   ///< the task grid is N·G ×
                                   ///< num_tile_blocks × oc_chunks. 1 unless
                                   ///< the plane is too small to fill the pool
                                   ///< (see the plan)
    int64_t num_tasks = 0;         ///< N · G · num_tile_blocks · oc_chunks

    // ---- channel-last only ----
    WinoChannelLastU u;       ///< the prepacked U's panel geometry
    int64_t v_lda = 0;        ///< elements per tile row of one V position =
                              ///< IC/G rounded up to the vector (the GEMM's lda)
    int64_t m_oc_stride = 0;  ///< channel-last: M's row (tile) stride in oc

    // ---- scratch (elements, not bytes), all per slot ----
    int64_t num_slots = 0;    ///< worker threads, else tasks
    int64_t halo_ld = 0;      ///< planar: padded columns = 2·tile_block + 2
    int64_t halo_rows = 0;    ///< planar: padded rows = 4 (the window)
    int64_t halo_size = 0;    ///< planar: ic_per_group · halo_rows · halo_ld
    int64_t v_p_stride = 0;   ///< elements between transform positions
    int64_t v_size = 0;       ///< 16 · v_p_stride
    int64_t m_size = 0;       ///< 16 · tile_block · ocn_block
    int64_t b_size = 0;       ///< planar: uniform packed-B elements
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
                                            bool use_thread_slots = false,
                                            int64_t num_slots = 0);

/// Winograd F(2×2, 3×3) Conv2D kernel (f32 / f16, k3s1, NCHW or NHWC).
///
/// inputs[1] must be the prepacked U produced by Conv2D::prepackWeights, in the
/// form that pairs with the activation's layout. Scratch is pooled internally
/// (sized by get_conv2d_winograd_plan); @p workspace is vestigial.
Status conv2d_winograd_kernel(const Conv2DAttributes& attrs,
                            TensorView& output,
                            std::span<const TensorView> inputs,
                            const ComputeContext& ctx,
                            void* workspace);

}  // namespace nnops::backend::cpu
