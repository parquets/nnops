#pragma once
/// @file conv2d_winograd.h
/// @brief Winograd Conv2D — the k3s1 fast path on NHWC, two tile sizes.
///
/// Winograd replaces the 3×3 convolution with `pos` batched GEMMs in the
/// transform domain: for each transform position p,
///
///     M_p[oc][t] = Σ_ic U_p[oc][ic] · V_p[ic][t]        t = tile index
///
/// where U is the prepacked weight (G g Gᵀ) and V the transformed input
/// (Bᵀ d B). Every one of those GEMMs runs through tile_mma_direct — raw A,
/// packed B — the same route MatMul uses.
///
/// Two orders, told apart by the U's leading dimension because they are not
/// interchangeable (WinoOrder):
///   * F(2×2, 3×3) — 16 positions, a 4×4 window per 2×2 output tile. 2.25x the
///     multiplies of a direct 3×3.
///   * F(4×4, 3×3) — 36 positions, a 6×6 window per 4×4 output tile. 4x the
///     multiplies of a direct 3×3 (1.78x better than F(2,3)) but ~4.5x the
///     transform work per tile and 2.25x the U to stream. wino_pick_order()
///     decides; see there.
///
/// M = tiles, N = oc block, K = ic; A = V [tiles][ic], the tile-major,
/// channel-contiguous layout the gather-free transform gives for free, and B =
/// the U panels prepacked once at load ([pos][nb][G·jb][ldd],
/// PackedWinoWeightLast). The output transform reads whole channel vectors
/// instead of striding down a column.
///
/// Structure: ONE dispatch over (batch, group, tile row, tile block, oc-chunk) —
/// the grid flattens the tile rows into tile blocks. Each task transforms its
/// tiles into V, runs the pos GEMMs for its slice of the filter, and writes
/// those tiles back. Tile rows, blocks and oc ranges are disjoint, so no task
/// reads another's scratch and the phases need no barrier between them.
///
/// The tile block is the one axis a task pays U for twice: every block streams
/// the whole weight blob, while an oc chunk reads a disjoint slice of it for the
/// same bytes. So the block count is taken as the fewest that fill the pool and
/// keep the arena inside one pooled allocation — at 1 thread that is the whole
/// plane whenever it fits (1.10x on a 36 MB U, flat where the U is already
/// L3-resident), bounded by an L2-sized working set because V and M spilling
/// past it costs more than the passes it saves.
///
/// A barrier is not available anyway: CpuBackend has no such thing — `run` is
/// just a forwarder to parallel_for, and a ParallelForFn gives no guarantee that
/// every worker visits the body, nor in what order (a pool that partitions
/// statically would never release a spinning worker). A completion counter in
/// the body is therefore a deadlock waiting for a caller.
///
/// Task-local scratch is what makes one dispatch possible: V and M live in one
/// per-slot block.
///
/// The oc axis is the exception, spent only when the pool would otherwise idle:
/// it redoes the input transform per chunk (under 1% of such a task) and stops
/// at two M-panels of work per task. It is what a 14x14 plane needs — 7 tile
/// rows and one 8-wide tile block is 7 tasks, and on this host 20 threads ran
/// ResNet-18's layer3.x slower than 4 (0.59 vs 0.45 ms) until the filter rows
/// were split.
///
/// Numeric order (each load-bearing):
///   * bias is NOT folded into M — M is a transform-domain accumulator and
///     Aᵀ(·)A would distort a folded bias. M starts at zero (zero_mode on the
///     first k-block), then bias[oc] is added per output element during
///     write-back.
///   * the epilogue runs after the output transform, on the real tile outputs,
///     and every accumulating GEMM clamps at ±inf with zero_mode=false.
///   * add_to needs no orig scratch: every output element is written exactly
///     once, so the residual reads the pre-existing output in place.
///
/// This kernel must never fall back to the reference conv2d: U is not an HWIO
/// filter, so the reference would silently compute garbage. A dtype mismatch is
/// a hard assert.
///
/// Winograd is NOT always faster, and this kernel makes no attempt to hide that:
/// the transforms cost a fixed amount per tile per channel, so with few channels
/// the multiply saving never pays for them. So whether a conv is worth
/// transforming is the caller's decision, taken once when it prepacks —
/// winograd_applicable() is that policy (the caller allows it, k3s1, at least
/// 32 input channels per group, and a 7x7 plane) — and ConvAlgorithm::Auto then
/// only follows the weight it is given.
///
/// A window is sixteen channel-vector loads, an output tile four channel-vector
/// stores, and flattened tiles need no neighbour constraint. The scratch is V
/// and M alone — a window is read straight from the input, and the prepacked U
/// is read in place by every task.
///
/// Against the implicit GEMM (bench_conv2d, same method)
/// this route is 0.47-0.73x of its time at 1T and 0.26-0.84x at 4T/20T for f32,
/// 0.46-0.63x and 0.32-0.75x for f16, over 7x7 to 224x224. Only a 3-channel stem
/// still loses (2.1-3.3x), which winograd_applicable() never lets through.

#include "nnops/core/tensor_view.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/ops/conv2d.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace nnops::backend::cpu {

/// Which transform order the U was built with — F(2×2,3×3) or F(4×4,3×3). The
/// two are not interchangeable, so the U's leading dimension has to say which:
/// 16 positions or 36. Prepack picks one; compute reads it back off the blob.
enum class WinoOrder : uint8_t {
    F23, ///< 16 positions, a 4×4 window per 2×2 output tile
    F43, ///< 36 positions, a 6×6 window per 4×4 output tile
};

constexpr int64_t wino_pos(WinoOrder o) noexcept { return o == WinoOrder::F43 ? 36 : 16; }
/// Output pixels per side of one tile: 4 on F(4,3), 2 on F(2,3).
constexpr int64_t wino_tile(WinoOrder o) noexcept { return o == WinoOrder::F43 ? 4 : 2; }
/// Input window per side: one tile plus the two kernel taps.
constexpr int64_t wino_alpha(WinoOrder o) noexcept { return wino_tile(o) + 2; }

/// Geometry of the Winograd U (TensorLayout::PackedWinoWeightLast).
///
/// Resolved by the same function at prepack time and inside the kernel, so the
/// producer and the consumer of those panels cannot drift. One panel is
/// [kc][nr] k-major — exactly what the packer would have written — and the
/// output channels run zero-padded to whole panels, so a partial panel's dead
/// columns are computed and dropped.
struct WinoU {
    WinoOrder order = WinoOrder::F23; ///< which transform produced these panels
    int64_t pos = 0;                  ///< transform positions = wino_pos(order)
    int64_t nr = 0;                   ///< panel width: the arch's nr_max for the dtype
    int64_t kc = 0;                   ///< k-block length: the largest divisor of IC/G that is
                                      ///< <= 128 (f16) or 256 (f32)
    int64_t nb = 0;                   ///< k-blocks per position = IC/G / kc
    int64_t jb = 0;                   ///< oc panels per group = ceil(oc_per_group / nr)
    int64_t jbg = 0;                  ///< oc panels over all groups = groups · jb
    int64_t ldd = 0;                  ///< elements per panel (kc·nr, padded to the pack align)
    int64_t plane = 0;                ///< elements per transform position = nb · jbg · ldd
    int64_t total = 0;                ///< pos · plane — the whole U
};

/// Resolve the U geometry for a dtype and transform order (see
/// WinoU).
WinoU wino_u(int64_t ic_per_group, int64_t oc_per_group, int64_t groups,
                                     DataType dt, WinoOrder order = WinoOrder::F23) noexcept;

/// Which order a prepack should build. The trade is F(4,3)'s 1.78x fewer
/// multiplies per output against ~4.5x the transform work per tile, which the
/// channel count amortizes. The windows are read in place, so the plane alone
/// decides: F(4,3) from 28x28 up, at any team size, subject to the U still
/// fitting the last-level cache.
///
/// f16 gets the same choice; its GEMM accumulates in f16, and F(4,3)'s Aᵀ/Bᵀ
/// amplify that rounding ~8x with the error growing as √IC (measured: 4% of full
/// scale at 32 channels per group, 8% at 128). The order is still picked by the
/// shape and cost fences; a caller that cannot take the wider error pins 2.
///
/// @p num_threads is the *worker* count (CpuBackend::thread_slot_count, not the
/// item grid) — the grid an over-subscribed pool hands out says nothing about
/// how many cores the transforms run on.
///
/// Pass the output extent when the caller knows it — a filter alone picks the
/// safe order. The policy is what `ConvAlgorithm::Winograd` means for a filter:
/// compute never re-decides it, it reads the order back off the U.
WinoOrder wino_pick_order(DataType dt, int64_t ic_per_group, int64_t oc_per_group,
                          int64_t out_h = 0, int64_t out_w = 0,
                          int num_threads = 1) noexcept;

/// Single source of truth for the Winograd block sizes, the parallel
/// decomposition and the scratch layout.
struct Conv2DWinogradPlan {
    WinoOrder order = WinoOrder::F23; ///< transform order

    // ---- transform-domain geometry ----
    int64_t num_tile_rows = 0; ///< TH  = ceil(OH / tile)
    int64_t num_tile_cols = 0; ///< TW  = ceil(OW / tile)
    int64_t num_tiles = 0;     ///< TH · TW — the GEMM's M

    // ---- blocking ----
    int64_t icn_block = 0;  ///< the U's k-block (kc)
    int64_t ocn_block = 0;  ///< oc block (GEMM N), a multiple of nr
    int64_t tile_block = 0; ///< tiles per task, flattened, up to 24

    int64_t num_tile_blocks = 1; ///< ceil(tiles / tile_block)
    int64_t oc_chunks = 1;       ///< parallel filter-row chunks per plane; 1
                                 ///< unless the plane is too small for the pool
    int64_t num_tasks = 0;       ///< N · G · num_tile_blocks · oc_chunks

    WinoU u;      ///< the prepacked U's panel geometry
    int64_t v_lda = 0;       ///< elements per tile row of one V position =
                             ///< IC/G rounded up to the vector (the GEMM's lda)
    int64_t m_oc_stride = 0; ///< M's row (tile) stride in oc

    // ---- scratch (elements, not bytes), all per slot ----
    int64_t num_slots = 0;  ///< worker threads, else tasks
    int64_t v_p_stride = 0; ///< elements between transform positions
    int64_t v_size = 0;     ///< pos · v_p_stride
    int64_t m_size = 0;     ///< pos · tile_block · ocn_block
    int64_t slot_size = 0;  ///< V + M, rounded to 16
    size_t workspace_size = 0;
};

/// Resolve the plan (tile geometry, block sizes, task grid, per-slot scratch
/// bytes) from the descriptors. Returns a zeroed plan (workspace_size 0) — which
/// routes the kernel to its assert — for unsupported cases: non-f32/f16 dtypes,
/// non-k3s1, or groups the U layout cannot express.
Conv2DWinogradPlan get_conv2d_winograd_plan(const Conv2DAttributes& attrs,
                                            std::span<const TensorDesc> inputs,
                                            std::span<const TensorDesc> outputs, int num_threads,
                                            bool use_thread_slots = false, int64_t num_slots = 0);

/// Winograd Conv2D kernel (f32 / f16, k3s1, NHWC, F(2,3) or F(4,3)).
///
/// inputs[1] must be the prepacked U (PackedWinoWeightLast) produced by
/// Conv2D::prepackWeights. Scratch is pooled internally (sized by
/// get_conv2d_winograd_plan); @p workspace is vestigial.
Status conv2d_winograd_kernel(const Conv2DAttributes& attrs, TensorView& output,
                              std::span<const TensorView> inputs, const ComputeContext& ctx,
                              void* workspace);

} // namespace nnops::backend::cpu
