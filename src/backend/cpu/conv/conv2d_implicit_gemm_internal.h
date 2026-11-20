#pragma once
/// @file conv2d_implicit_gemm_internal.h
/// @brief Internals shared by the two implicit-GEMM Conv2D translation units.
///
/// The route is split one file per arithmetic family, the same variant-per-file
/// split the matmul routes use:
///
///   - conv2d_implicit_gemm.cpp    — the f32/f16/bf16 route, the plan, the
///                                   shared driver and the prepacked-filter pack;
///   - i8conv2d_implicit_gemm.cpp  — the quantized route (s8 weight against an
///                                   s8/u8/f32/f16 activation, fp or requantized
///                                   output) and the int8 panels' prepack.
///
/// What they share is here: the plan's block heuristics (the plan is one
/// decision for both families, so its constants and the ic-block ladder are
/// single-sourced), the staging loop the filter packers run on, and the two
/// per-dtype leaves the shared driver calls — each family owns its own leaf and
/// its own dtype ladder.

#include "conv2d_implicit_gemm.h"
#include "backend/cpu/common/memory_pool.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

namespace nnops::backend::cpu {

namespace impl {

// =========================================================================
//  Plan block heuristics — shared, because the plan serves both families
// =========================================================================

constexpr int64_t kOcnBlock = 128; ///< output-channel tile (the GEMM's N)
/// An oc_per_group at or below this stays one block: every extra oc block costs
/// a second pass over the pixels (A pack, accumulator fill, epilogue). Splitting
/// 144 into 80 + 64 cost the int8 expand layer a quarter of its time.
constexpr int64_t kOcnSingleCap = 192;
constexpr int64_t kMBlock = 24; ///< pixel tile (the GEMM's M): 4 × mr_max(6)
/// The widest pixel tile any route picks: the int8 SME route takes 16 * sme_mr
/// (256) rows. Stack buffers indexed by the tile are fixed at this width so the
/// file still compiles with the SME kernels switched off.
constexpr int64_t kMaxMBlock = 256;
/// Pixels a task needs before it is worth splitting the plane again: two M
/// blocks, i.e. one full tile plus a second to hide its prologue.
constexpr int64_t kTaskMinPixels = 2 * kMBlock;
/// Bytes one accumulator pass may cover (the plan's rows-per-pass knob).
constexpr int64_t kPassAccBytes = 128 * 1024;
constexpr int64_t kABSlack = 8; ///< elements past the A tile (vector over-read)

/// Cap on the shared packed-weight buffer: written once per call, read by every
/// task, so a small plane (few pixels per task) need not pay the whole pack per
/// task. Past 8 MiB it no longer fits an L3 alongside the working set.
constexpr int64_t kPrepackCap = int64_t{8} << 20;

/// Cap on the per-slot add_to residual / int32 accumulator tile set (elements).
/// Rows are split further until it fits: a residual path costs parallelism,
/// never memory.
constexpr int64_t kTaskTileCap = int64_t{64} << 10;

/// Input-channel block: kdim = ic_block × karea lands in ~144–256, and the
/// packed-B panel (nr_max × kdim) stays near L1/L2. A 1x1 filter has neither a
/// gather nor a karea multiplier, so K runs four times as wide.
inline int64_t get_ic_block(int64_t karea, int64_t ic_per_group) noexcept
{
    if (karea == 1) {
        return 256;
    }
    // A narrow channel count takes itself whole even when the ladder would split
    // it: with every channel of a pixel in the block, one filter row's taps are
    // adjacent, so the gather copies one run instead of `KW` runs of two. That
    // is the 3-channel stem's whole loss (IC=3, 7x7).
    if (ic_per_group * karea <= 256) {
        return std::max<int64_t>(ic_per_group, 1);
    }
    if (karea <= 9) {
        return 16;
    }
    if (karea <= 25) {
        return 8;
    }
    if (karea <= 49) {
        return 2;
    }
    return 1;
}

// =========================================================================
//  The two per-dtype leaves — one per family, called by the shared driver
// =========================================================================

/// The float family's ladder: f32/f16 everywhere, bf16 where its kernels exist.
Status conv2d_fp_impl(const Conv2DAttributes& attrs, TensorView& output,
                      std::span<const TensorView> inputs, const ComputeContext& ctx,
                      const Conv2DGemmPlan& plan, void* workspace);

/// The quantized family's ladder: (activation, output) dtype pairs.
Status conv2d_quant_impl(const Conv2DAttributes& attrs, TensorView& output,
                         std::span<const TensorView> inputs, const ComputeContext& ctx,
                         const Conv2DGemmPlan& plan, void* workspace);

// =========================================================================
//  The filter prepack's staging loop — shared by the fp and int8 packers
// =========================================================================

/// Run @p body over [0, n) with one staging tile per worker — the NHWC
/// pack's only scratch. Falls back to one tile and a serial loop when the
/// backend reports no thread ids: a shared tile would race.
template <class Body>
void with_stage(const ComputeContext& ctx, int64_t n, int64_t stage_len, Body body)
{
    const int nslots = std::max(ctx.cpu.thread_slot_count(), 1);
    const bool par = nslots > 1 && n > 1;
    // One char array keeps this helper type-agnostic.
    PoolArray<char> stages(static_cast<size_t>(nslots * stage_len));
    auto run = [&](int64_t s) {
        char* const stage =
            stages.data() +
            (par ? static_cast<int64_t>(ctx.cpu.current_thread_id()) : 0) * stage_len;
        body(s, stage);
    };
    if (par) {
        ctx.cpu.run(0, n, run);
    } else {
        for (int64_t s = 0; s < n; ++s) {
            run(s);
        }
    }
}

} // namespace impl

// =========================================================================
//  The quantized filter prepack (i8conv2d_implicit_gemm.cpp)
// =========================================================================

/// Fill an s8 blob's panels from the raw HWIO filter and write the R_b table
/// that trails them (Σ_k weight[oc,k] per output channel).
void conv_packed_weight_pack_i8(const TensorView& weight, TensorView& packed,
                                const ConvPackedWeight& g, int64_t ic_per_group,
                                int64_t oc_per_group, int64_t groups, const ComputeContext& ctx);

} // namespace nnops::backend::cpu
