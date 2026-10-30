#pragma once
/// @file conv2d_winograd.h
/// @brief Winograd F(2×2, 3×3) Conv2D — the k3s1 fast path, both layouts (f32/f16).
///
/// F(2,3) replaces the 3×3 convolution with 16 batched GEMMs in the transform
/// domain: for each transform position p ∈ [0,16),
///
///     M_p[oc][t] = Σ_ic U_p[oc][ic] · V_p[ic][t]        t = tile index
///
/// where U is the prepacked weight (G g Gᵀ, one 4×4 kernel per position) and V
/// the transformed input (Bᵀ d B, one 4×4 window per 2×2 output tile). Every one
/// of those GEMMs runs through tile_mma_direct — raw A, packed B — the same
/// route MatMul uses.
///
/// The algebra is the same for both layouts and only the tuple swaps, exactly as
/// Conv2D's implicit-GEMM routes swap it:
///   * planar (NCHW) — M = oc, N = tiles (the tile count padded to a multiple of
///     8), K = ic; A = U [16][OC][IC/G] (PackedWinoWeight) as it lies, B = V
///     [ic][tiles] packed per call.
///   * channel-last (NHWC) — M = tiles, N = oc block, K = ic; A = V [tiles][ic],
///     the tile-major, channel-contiguous layout the gather-free transform gives
///     for free, and B = the U panels prepacked once at load
///     ([16][nb][G·jb][ldd], PackedWinoWeightLast). The output transform then
///     reads whole channel vectors instead of striding down a column.
///
/// Structure: ONE dispatch over (batch, group, tile row, tile block, oc-chunk) —
/// the channel-last grid flattens the tile rows into tile blocks. Each task
/// transforms its tiles into V (the planar route from a halo strip it builds),
/// runs the 16 GEMMs for its slice of the filter, and writes those tiles back.
/// Tile rows, blocks and oc ranges are disjoint, so no task reads another's
/// scratch and the phases need no barrier between them.
///
/// A barrier is not available anyway: CpuBackend has no such thing — `run` is
/// just a forwarder to parallel_for, and a ParallelForFn gives no guarantee that
/// every worker visits the body, nor in what order (a pool that partitions
/// statically would never release a spinning worker). A completion counter in
/// the body is therefore a deadlock waiting for a caller.
///
/// Task-local scratch is what makes one dispatch possible: V and M (and, planar
/// only, the halo strip) live in one per-slot block, and on the planar route the
/// price is deliberate duplication at the task edges — the two padded columns
/// neighbouring tile blocks share, and the two window rows neighbouring tile rows
/// share (the halo copy is 2x the input plane, being a pure copy of it). Fusing
/// only the halo would instead make every oc-block task repeat the whole input
/// transform for its tiles, so the tile axes lead the grid and the batch/group
/// axes join them.
///
/// The oc axis is the exception, spent only when the pool would otherwise idle:
/// it redoes the halo and the input transform per chunk (under 1% of such a
/// task) and stops at two M-panels of work per task. It is what a 14x14 plane
/// needs — 7 tile rows and one 8-wide tile block is 7 tasks, and on this host 20
/// threads ran ResNet-18's layer3.x slower than 4 (0.59 vs 0.45 ms) until the
/// filter rows were split.
///
/// Numeric order (each load-bearing):
///   * bias is NOT folded into M — M is a transform-domain accumulator and
///     Aᵀ(·)A would distort a folded bias. M starts at zero (zero_mode on the
///     first k-block), then bias[oc] is added per output element during
///     write-back.
///   * the epilogue runs after the output transform, on the real 2×2 outputs,
///     and every accumulating GEMM clamps at ±inf with zero_mode=false.
///   * add_to needs no orig scratch: every output element is written exactly
///     once, so the residual reads the pre-existing output in place.
///
/// This kernel must never fall back to the reference conv2d: U is not an OIHW
/// filter (planar's is rank-3 with dims[0] == 16), so the reference would read it
/// as a weight of 16 output channels and silently compute garbage. A dtype
/// mismatch is a hard assert.
///
/// Winograd is NOT always faster, and this kernel makes no attempt to hide that:
/// the transforms cost a fixed amount per tile per channel, so with few channels
/// the 2.25x multiply saving never pays for them. Measured on this tree
/// (bench_conv2d, interleaved A/B against im2col with a null control, median
/// over rounds): planar f32 is ahead from 32 channels up — ~1.2x at 32,
/// ~1.3-1.45x at 64 and ~1.55-1.6x at 256 — and a wash at 8; f16 reaches
/// ~1.35x at 64 and ~1.6-1.7x at 256. So whether a conv is worth transforming
/// is the caller's decision, taken once when it prepacks —
/// winograd_applicable() is that policy (the caller allows it, k3s1, at least
/// 32 input channels per group, at least a 14x14 input) — and
/// ConvAlgorithm::Auto then only follows the weight it is given.
///
/// Channel-last is the wider win, because the two costs the planar thresholds
/// are about — a deinterleaving load and 8-lane tile packing — do not exist
/// there: a window is sixteen channel-vector loads, an output tile four
/// channel-vector stores, and flattened tiles need no neighbour constraint. Its
/// scratch is V and M alone — a window is read straight from the input, and the
/// prepacked U is read in place by every task.
///
/// Against the channel-last implicit GEMM (bench_conv2d_nhwc, same method) this
/// route is 0.47-0.73x of its time at 1T and 0.26-0.84x at 4T/20T for f32,
/// 0.46-0.63x and 0.32-0.75x for f16, over 7x7 to 224x224. Only a 3-channel stem
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
    int64_t kc = 0;     ///< k-block length: the largest divisor of IC/G that is
                        ///< <= 128 (f16) or 256 (f32)
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
/// decomposition and the scratch layout; the fields a route does not use stay
/// zero.
struct Conv2DWinogradPlan {
    Conv2DWinogradRoute route = Conv2DWinogradRoute::Planar;

    // ---- transform-domain geometry ----
    int64_t num_tile_rows = 0;      ///< TH  = ceil(OH / 2)
    int64_t num_tile_cols = 0;      ///< TW  = ceil(OW / 2)
    int64_t num_tile_cols_pad = 0;  ///< planar: TW_pad = ceil(TW / 8) · 8
                                    ///< channel-last: the flat tile count TH · TW
    int64_t num_tiles = 0;          ///< planar: TH · TW_pad — the GEMM's N

    // ---- blocking ----
    int64_t icn_block = 0;   ///< planar: input-channel (GEMM K) block
                             ///< channel-last: the U's k-block (kc)
    int64_t ocn_block = 0;   ///< planar: output-channel (GEMM M) block
                             ///< channel-last: oc block (GEMM N), a multiple of nr
    int64_t tile_block = 0;  ///< tiles per task: 8 or 16 within one tile row
                             ///< (planar), up to 24 flattened (channel-last)

    int64_t num_tile_blocks = 1;   ///< ceil(tiles / tile_block)
    int64_t oc_chunks = 1;         ///< parallel filter-row chunks per plane; 1
                                   ///< unless the plane is too small for the pool
    int64_t num_tasks = 0;         ///< N · G · num_tile_blocks · oc_chunks, with
                                   ///< TH also in the product on the planar route

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
    int64_t slot_size = 0;    ///< halo + V + M + B; channel-last rounds to 16
    size_t workspace_size = 0;
};

/// Resolve the plan (tile geometry, block sizes, task grid, per-slot scratch
/// bytes) from the descriptors. Returns a zeroed plan (workspace_size 0) — which
/// routes the kernel to its assert — for unsupported cases: non-f32/f16 dtypes,
/// non-k3s1, or groups the U layout cannot express.
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
