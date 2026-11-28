/// @file conv2d_winograd.cpp
/// @brief Winograd Conv2D kernel (k3s1, f32/f16, NHWC, F(2,3) and
/// F(4,3)).
///
/// One plan, one dispatch, and the numeric order and scratch conventions
/// described in conv2d_winograd.h: the algorithm, the structure of the task
/// grid, the scratch layout and the numeric-ordering rules.

#include "conv2d_winograd.h"
#include "backend/cpu/matmul/matmul_helper.h"        // tile_mma_direct / tile_pack_rhs
#if defined(NNOPS_ARCH_SME)
#include "backend/cpu/aarch64/sme/mma_sme.hpp"       // available_sme, the ZA panel width
#endif
#include "backend/cpu/simd_kernel/simd_winograd.hpp" // the F(2,3) and F(4,3) transforms
#include "backend/cpu/common/memory_pool.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/assert.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace nnops::backend::cpu {

namespace k = nnops::kernel;

// =========================================================================
//  Winograd route (NHWC)
// =========================================================================

WinoOrder wino_pick_order(DataType dt, int64_t ic_per_group, int64_t oc_per_group, int64_t out_h,
                          int64_t out_w, int num_threads) noexcept
{
    // f16 is offered F(4,3) on the same terms as f32; what differs is the
    // accuracy it buys. Bᵀ scales a window by up to 5 and Aᵀ the products by 8,
    // so the GEMM's f16 accumulate rounds an intermediate ~8x the output, and
    // the error grows as √IC on top of that.
    //
    // Measured on an M4 against an f32 implicit GEMM (five seeds, 16x16/56x56),
    // by channels per group:
    //   IC/G   8: 1.5% of full scale, 0.15% rms, worst |err|/(1+|ref|) 0.04-0.22
    //   IC/G  16: 2.4%,              0.20%,                        0.18-0.30
    //   IC/G  32: 4.0%,              0.28%,                        0.25-0.72
    //   IC/G  64: 5.5%,              0.28%,                        0.65-1.16
    //   IC/G 128: 8.0%,              0.37%,                        -
    // F(2,3) holds ~0.2% of full scale and 0.01 element-wise on the same cases.
    // The element-wise tail is cancellation on near-zero outputs, not a bias:
    // at 64 channels and up an output near zero can be off by more than its own
    // magnitude. That is the caller's trade to make — pin winograd_order = 2 on
    // a layer that needs the accuracy.

    // What the wider order costs in traffic is the U *per tile it amortizes
    // over* — esz / tiles₄₃ — because the blob is streamed once per tile-block
    // pass; what it saves is a quarter of the multiplies per useful output.
    // The blob's absolute size is therefore the wrong test: it is blind to the
    // plane, and at 512x512 f32 (a 37.7 MB U, 2.4x the L3) F(4,3) measured
    // faster than F(2,3)'s 16.8 MB blob at 12x12 (1.3-1.5x), 14x14 (1.0-1.3x)
    // and 16x16 up (1.3-2.1x, 1/4/8/10 workers). An earlier 16 MB cap demoted
    // exactly those to F(2,3) and cost 14x14 C512->512 1.27-1.33x. What the cap
    // does catch is the 7-10x7-10 corner, where the padded last tile wastes
    // part of the plane and there are too few tiles to amortize the transform —
    // and those planes are under the floor below anyway. (A 20-worker datum once
    // put F(4,3) 1.38x behind at 28x28 on that filter; it does not reproduce at
    // ten workers, and if it ever does the fence to move is the floor's team
    // term, not a blob-size cap.)
    (void)dt;
    (void)ic_per_group;
    (void)oc_per_group;

    // The windows are read out of the packed panel in place, so the dearer
    // transform is the only cost the order adds: 0.61-0.92x of F(2,3) at 1
    // thread over every plane from 28x28 up, 0.60-0.86x at 20 threads except the
    // two narrowest filters (32 and 64 channels on a 28x28 plane, 1.27x and
    // 1.28x, against 21 and 55 us of work). A 14x14 plane is 16 F(4,3) tiles
    // and one whole-plane block: 1.23x F(2,3) at 1 thread (0.655 vs 0.804 ms on
    // 256->256). Past four workers that block no longer fills the pool and the
    // per-task transform shows, so the floor climbs with the team.
    const int64_t floor = num_threads <= 4 ? 14 : 28;
    return (out_h >= floor && out_w >= floor) ? WinoOrder::F43 : WinoOrder::F23;
}

WinoU wino_u(int64_t ic_per_group, int64_t oc_per_group, int64_t groups,
                                     DataType dt, WinoOrder order) noexcept
{
    WinoU u;
    if (ic_per_group <= 0 || oc_per_group <= 0 || groups <= 0) {
        return u;
    }
    if (dt != DataType::f32 && dt != DataType::f16) {
        return u;
    }
    u.order = order;
    u.pos = wino_pos(order);

    const int64_t esz = static_cast<int64_t>(data_type_size(dt));
    // The MMA's panel width, and the k-block length it will be given. kc divides
    // IC/G exactly, so the panels need no zero-padded k tail; the ceiling is the
    // same f16 accumulate budget MatMul uses (KC_CAP_F16), with f32 allowed to
    // run twice as wide because its accumulator is exact enough for it.
    // On an SME machine the panels are the ZA B blocks: 32 columns, K-major,
    // with the f16 K pairs interleaved. The transform GEMMs then run through
    // tile_mma_pack_sme like every other product, instead of the NEON direct
    // kernel that was 87% of this route's time.
#if defined(NNOPS_ARCH_SME)
    const bool za = aarch64::available_sme();
#else
    const bool za = false;
#endif
    // 32: the ZA B block's width (the f16 form interleaves its K pairs).
    constexpr int64_t kZaPanelN = 32;
    const int64_t nr = za ? kZaPanelN : (dt == DataType::f16 ? NR_MAX_F16 : NR_MAX_F32);
    const int64_t cap = (dt == DataType::f16) ? 128 : 256;
    int64_t kc = std::min(ic_per_group, cap);
    while (ic_per_group % kc != 0) {
        --kc;
    }

    u.za = za;
    u.nr = nr;
    u.kc = kc;
    u.nb = ic_per_group / kc;
    u.jb = (oc_per_group + nr - 1) / nr;
    u.jbg = groups * u.jb;
    u.ldd = static_cast<int64_t>(
        mma_pack_align(static_cast<size_t>(nr * kc) * static_cast<size_t>(esz)) / esz);
    u.plane = u.nb * u.jbg * u.ldd;
    u.total = u.pos * u.plane;
    return u;
}

namespace impl {
namespace {

using namespace nnops::simd;

/// Blocks stop shrinking here: below it a task re-reads its whole U slice
/// (pos·kc·oc) for too few tiles of MACs, and the per-task transform cost shows.
constexpr int64_t kTileBlockMin(WinoOrder o) noexcept { return o == WinoOrder::F43 ? 6 : 8; }
/// Output channels per GEMM (the N side), snapped down to whole panels — each
/// extra oc block is another pass over the task's tiles (accumulator traffic
/// plus one more B read).
constexpr int64_t kOcnBlock = 64;

/// Winograd kernel: per-slot scratch is V then M, and U is read in place.
/// The transform order is a template parameter because it fixes the position
/// count, the tile size and the two transforms' shapes — everything else about
/// the task is order-independent.
template <class T, WinoOrder O>
void conv2d_winograd_impl(const Conv2DAttributes& attrs, TensorView& output,
                                       std::span<const TensorView> inputs,
                                       const ComputeContext& ctx, const Conv2DWinogradPlan& plan,
                                       void* workspace)
{
    constexpr int64_t kpos = wino_pos(O);
    constexpr int64_t ktile = wino_tile(O);
    const auto& input = inputs[0];
    const auto& u_tensor = inputs[1]; // prepacked transform-domain weight
    const bool has_bias = inputs.size() > 2;

    // Shape is channel-last [N, H, W, C] — the physical order too — so the
    // memory steps come from the view's strides, not from the shape product
    // (a row may be pitch-padded).
    const int64_t logical_h = input.shape(1);
    const int64_t logical_w = input.shape(2);
    const int64_t logical_c = input.shape(3);
    const int64_t in_w_stride = input.stride_elems(2);
    const int64_t in_h_stride = input.stride_elems(1);
    const int64_t in_n_stride = input.stride_elems(0);

    const int64_t logical_oc = output.shape(3);
    const int64_t logical_oh = output.shape(1);
    const int64_t logical_ow = output.shape(2);
    const int64_t out_w_stride = output.stride_elems(2);
    const int64_t out_h_stride = output.stride_elems(1);
    const int64_t out_n_stride = output.stride_elems(0);

    const int64_t G = attrs.groups;
    const int64_t ic_per_group = logical_c / G;
    const int64_t oc_per_group = logical_oc / G;
    const int64_t PH = attrs.padding[0];
    const int64_t PW = attrs.padding[1];

    // U: [pos][nb][G·jb][ldd] — the GEMM's B panels, one set per transform
    // position, and the dims must match the geometry prepack derived them under.
    const WinoU& ug = plan.u;
    NNOPS_ASSERT(u_tensor.shape(0) == kpos && u_tensor.shape(1) == ug.nb &&
                 u_tensor.shape(2) == ug.jbg && u_tensor.shape(3) == ug.ldd);

    const int64_t TW = plan.num_tile_cols;
    const int64_t tiles = plan.num_tiles;

    const int64_t tile_block = plan.tile_block;
    const int64_t num_tile_blocks = plan.num_tile_blocks;
    const int64_t oc_chunks = plan.oc_chunks;
    const int64_t ocn_block = plan.ocn_block;
    const int64_t v_lda = plan.v_lda;
    const int64_t v_p_stride = plan.v_p_stride;
    const int64_t m_oc_stride = plan.m_oc_stride;
    const int64_t m_p_stride = tile_block * m_oc_stride;
    const int64_t oc_blocks = (oc_per_group + ocn_block - 1) / ocn_block;

    T* out_ptr = output.ptr<T>();
    const T* in_ptr = input.ptr<T>();
    const T* u_ptr = u_tensor.ptr<T>();
    const T* b_ptr = has_bias ? inputs[2].ptr<T>() : nullptr;

    // One channel block is one vector — 8 lanes for both dtypes (NEON f32 is
    // two float32x4_t behind the same abstraction).
    constexpr int64_t vec = simd_lane_for<T>;

    const int nslots = ctx.cpu.thread_slot_count();
    const bool per_thread = nslots > 0 && plan.num_slots < plan.num_tasks;

    // One task = one (batch, group, tile block, oc chunk): transform its tiles
    // into V once, run the pos GEMMs for its oc blocks into M, write them back.
    const auto run_task = [&](int64_t t) {
        int64_t idx = t;
        // The oc chunk varies fastest: the chunks of one plane are adjacent in
        // task order and re-read the same input lines, so whichever arrives
        // second finds them in L2.
        const int64_t chunk = idx % oc_chunks;
        idx /= oc_chunks;
        const int64_t tb = idx % num_tile_blocks;
        idx /= num_tile_blocks;
        const int64_t g = idx % G;
        idx /= G;
        const int64_t n = idx;

        const int64_t tile0 = tb * tile_block;
        const int64_t ntl = std::min(tile_block, tiles - tile0);

        const int64_t slot = per_thread ? static_cast<int64_t>(ctx.cpu.current_thread_id()) :
                                          std::min(t, plan.num_slots - 1);
        T* scratch = static_cast<T*>(workspace) + slot * plan.slot_size;
        T* v_all = scratch;
        T* m_all = v_all + plan.v_size;
        T* const a_packed = m_all + plan.m_size;
        (void)a_packed;

        const T* task_in = in_ptr + n * in_n_stride + g * ic_per_group;
        T* task_out = out_ptr + n * out_n_stride + g * oc_per_group;

        // Input transform, tile by tile: the tile's window is swept across the
        // channel blocks while its input lines are still hot.
        for (int64_t tl = 0; tl < ntl; ++tl) {
            const int64_t tt = tile0 + tl;
            const int64_t th = tt / TW;
            const int64_t tw = tt - th * TW;
            const int64_t ih0 = ktile * th - PH;
            const int64_t iw0 = ktile * tw - PW;
            for (int64_t ic = 0; ic < ic_per_group; ic += vec) {
                const int lanes = static_cast<int>(std::min<int64_t>(vec, ic_per_group - ic));
                // The block's channel offset goes into both pointers: V's row
                // channel and the activation's channel lane.
                T* v_tile = v_all + tl * v_lda + ic;
                if constexpr (O == WinoOrder::F23) {
                    k::winograd_input<T>(
                        task_in + ic, in_h_stride, in_w_stride, static_cast<int>(ih0),
                        static_cast<int>(iw0), static_cast<int>(logical_h),
                        static_cast<int>(logical_w), v_tile, v_p_stride, lanes);
                } else {
                    k::winograd43_input<T>(
                        task_in + ic, in_h_stride, in_w_stride, static_cast<int>(ih0),
                        static_cast<int>(iw0), static_cast<int>(logical_h),
                        static_cast<int>(logical_w), v_tile, v_p_stride, lanes);
                }
            }
        }

        // This chunk's oc blocks, disjoint; splitting on block boundaries rather
        // than column ranges keeps every start a whole panel into the prepacked U.
        const int64_t b0 = chunk * oc_blocks / oc_chunks;
        const int64_t b1 = (chunk + 1) * oc_blocks / oc_chunks;

        for (int64_t b = b0; b < b1; ++b) {
            const int64_t ocn0 = b * ocn_block;
            const int64_t ocnc = std::min(ocn_block, oc_per_group - ocn0);
            // The GEMM runs whole panels: the U's oc axis is padded to them (with
            // zeros, which is what keeps the pad columns' M at zero), and the
            // panel tail inside the last block would otherwise be a partial panel
            // the packed-B kernel cannot read. The write-back below still stops at
            // the real channels.
            const int64_t ocn_gemm = ((ocnc + ug.nr - 1) / ug.nr) * ug.nr;

            // M_p zeroes on the first k-block and accumulates after; bias is
            // added at write-back, never here (Aᵀ(·)A would distort it).
            for (int64_t kb = 0; kb < ug.nb; ++kb) {
                const int64_t kc = ug.kc;
                const T* b_base = u_ptr + (kb * ug.jbg + g * ug.jb + ocn0 / ug.nr) * ug.ldd;
                const T* a_base = v_all + kb * kc;
                for (int p = 0; p < kpos; ++p) {
                    T* const m_p = m_all + p * m_p_stride;
                    const T* const v_p = a_base + p * v_p_stride;
                    const T* const u_p = b_base + p * ug.plane;
#if defined(NNOPS_ARCH_SME)
                    if (ug.za) {
                        // The same product on the ZA tiles: pack this
                        // position's V slab into the kernel's A layout, then
                        // one tiled pass over (tiles, oc block, ic).
                        tile_pack_lhs_sme(false, static_cast<int>(ntl), static_cast<int>(kc),
                                          a_packed, v_p, static_cast<int>(v_lda));
                        tile_mma_pack_sme(
                            static_cast<int>(ntl), static_cast<int>(ocn_gemm),
                            static_cast<int>(kc), m_p, static_cast<int>(m_oc_stride), a_packed,
                            u_p, ug.ldd, -std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::infinity(), /*zero_mode=*/kb == 0,
                            /*col_bias=*/nullptr);
                        continue;
                    }
#endif
                    tile_mma_direct(static_cast<int>(ntl), static_cast<int>(ocn_gemm),
                                    static_cast<int>(kc), m_p, static_cast<int>(m_oc_stride),
                                    v_p, static_cast<int>(v_lda), u_p, /*ldb=*/-1,
                                    /*stage=*/nullptr, -std::numeric_limits<float>::infinity(),
                                    std::numeric_limits<float>::infinity(),
                                    /*zero_mode=*/kb == 0);
                }
            }

            // Output transform + bias + epilogue + write-back, per tile: its
            // tile×tile pixels are adjacent in memory, and a short output extent
            // leaves the pixels past it with a null destination.
            for (int64_t tl = 0; tl < ntl; ++tl) {
                const int64_t tt = tile0 + tl;
                const int64_t th = tt / TW;
                const int64_t tw = tt - th * TW;
                const int64_t oh = ktile * th;
                const int64_t ow = ktile * tw;
                for (int64_t oc = 0; oc < ocnc; oc += vec) {
                    const int lanes = static_cast<int>(std::min<int64_t>(vec, ocnc - oc));
                    T* base = task_out + oh * out_h_stride + ow * out_w_stride + ocn0 + oc;
                    T* dst[ktile][ktile];
                    for (int r = 0; r < ktile; ++r) {
                        for (int c = 0; c < ktile; ++c) {
                            dst[r][c] = (oh + r < logical_oh && ow + c < logical_ow) ?
                                            base + r * out_h_stride + c * out_w_stride :
                                            nullptr;
                        }
                    }
                    const T* bias_oc = b_ptr ? b_ptr + g * oc_per_group + ocn0 + oc : nullptr;
                    const T* m_tile = m_all + tl * m_oc_stride + oc;
                    if constexpr (O == WinoOrder::F23) {
                        k::winograd_output<T>(m_tile, m_p_stride, dst, attrs.epilogue,
                                                           bias_oc, attrs.add_to, lanes,
                                                           attrs.residual_pre_activation);
                    } else {
                        k::winograd43_output<T>(m_tile, m_p_stride, dst,
                                                             attrs.epilogue, bias_oc, attrs.add_to,
                                                             lanes, attrs.residual_pre_activation);
                    }
                }
            }
        }
    };

    ctx.cpu.run(0, plan.num_tasks, run_task);
}

/// Resolve the route: the activation is NHWC and U is prepacked as
/// B panels (PackedWinoWeightLast).
Conv2DWinogradPlan make_plan(const Conv2DAttributes& attrs, std::span<const TensorDesc> inputs,
                             std::span<const TensorDesc> outputs, int num_threads,
                             bool use_thread_slots, int64_t num_slots)
{
    Conv2DWinogradPlan plan;
    if (inputs.size() < 2 || outputs.empty()) {
        return plan;
    }

    const auto& in = inputs[0];
    const auto& u = inputs[1];
    const auto& out = outputs[0];

    const DataType dt = in.dtype;
    if (dt != DataType::f32 && dt != DataType::f16) {
        return plan;
    }
    if (u.dtype != dt || out.dtype != dt) {
        return plan;
    }
    if (inputs.size() > 2 && inputs[2].dtype != dt) {
        return plan;
    }

    // k3s1 only — the F(2,3) derivation assumes a 3×3, unit-stride, undilated
    // kernel.
    if (attrs.kernel_size[0] != 3 || attrs.kernel_size[1] != 3) {
        return plan;
    }
    if (attrs.stride[0] != 1 || attrs.stride[1] != 1) {
        return plan;
    }
    if (attrs.dilation[0] != 1 || attrs.dilation[1] != 1) {
        return plan;
    }
    if (in.rank != 4 || out.rank != 4 || u.rank != 4) {
        return plan;
    }
    if (!is_channel_last(in.layout) || !is_channel_last(out.layout)) {
        return plan;
    }
    if (u.layout != TensorLayout::PackedWinoWeightLast) {
        return plan;
    }

    const int64_t IC = in.dims[3];
    const int64_t OC = out.dims[3];
    const int64_t G = attrs.groups;
    if (G < 1 || IC % G != 0 || OC % G != 0) {
        return plan;
    }
    const int64_t ic_per_group = IC / G;
    const int64_t oc_per_group = OC / G;

    // The transform order is the blob's leading dimension: 16 positions is
    // F(2,3), 36 is F(4,3). Nothing else can tell them apart, and reading it off
    // the U is what lets the same Conv2D compute both.
    const WinoOrder order =
        (u.dims[0] == wino_pos(WinoOrder::F43)) ? WinoOrder::F43 : WinoOrder::F23;
    // The panels' geometry is derived, not read: the same function ran at prepack
    // time, so matching it against the descriptor keeps the two ends from drifting.
    const WinoU ug = wino_u(ic_per_group, oc_per_group, G, dt, order);
    if (ug.total == 0 || u.dims[0] != ug.pos || u.dims[1] != ug.nb || u.dims[2] != ug.jbg ||
        u.dims[3] != ug.ldd) {
        return plan;
    }
    plan.u = ug;
    plan.order = order;

    const int64_t OH = out.dims[1];
    const int64_t OW = out.dims[2];
    if (OH <= 0 || OW <= 0) {
        return plan;
    }

    const int64_t tile = wino_tile(order);
    const int64_t TH = (OH + tile - 1) / tile;
    const int64_t TW = (OW + tile - 1) / tile;
    const int64_t tiles = TH * TW;

    plan.num_tile_rows = TH;
    plan.num_tile_cols = TW;
    plan.num_tiles = tiles;
    plan.icn_block = ug.kc;

    // Whole panels, and an even number of them per block: the U's oc axis is
    // padded to panels, so a block boundary that lands inside one either drops
    // columns or forces a second pass over them. Snapping the cap *down* to a
    // panel is what turned OC 64 into a 60 + 4 pair of passes — the 4-wide pass
    // re-reads the whole V and costs about as much as a full one, and on a
    // 4-worker grid it pairs a nearly idle worker with a busy one. Rounding the
    // cap up (64 -> 72 at nr 12) and splitting the panels evenly instead gives
    // one 6-panel block for OC 64 and two for OC 128.
    const int64_t nr = ug.nr;
    const int64_t panels = (oc_per_group + nr - 1) / nr;
    const int64_t max_panels = std::max<int64_t>(1, (kOcnBlock + nr - 1) / nr);
    const int64_t nblk = std::max<int64_t>(1, (panels + max_panels - 1) / max_panels);
    plan.ocn_block = nr * ((panels + nblk - 1) / nblk);

    // V: one [tile][ic] plane per position, tiles contiguous inside a position
    // (that is the GEMM's A, so lda is a tile row). The row is padded to the
    // vector width so a tail block can be written whole, and the +8 per position
    // keeps the 16 planes off one L1 set.
    plan.v_lda = ((ic_per_group + 7) / 8) * 8;
    // M: [tile][oc] per position, a whole panel per row so the MMA's widest row
    // read stays inside it.
    plan.m_oc_stride = plan.ocn_block;

    // Tile blocks first — flattening repeats no work — then the oc axis, which
    // repeats the input transform (its input lines stay hot) but nothing of U.
    // A tile block is a pass over the whole U (pos·IC·OC), and no other block
    // re-uses it; an oc chunk reads a disjoint slice of the same bytes. So the
    // block axis is the only one that costs weight traffic, and blocks exist
    // for the pool and the arena, not for the GEMM: take the fewest that fill
    // it. Measured at 1 thread on NHWC f32, whole-plane against the old
    // ~12-tile block: 28x28 512->512 1.10x, 28x28 256 1.15x, 56x56 256 flat.
    const int64_t NB = (in.dims[0] > 0) ? in.dims[0] : 1;
    const int64_t nt = std::max<int64_t>(num_threads, 1);
    const int64_t oc_blocks = (oc_per_group + plan.ocn_block - 1) / plan.ocn_block;
    const int64_t waves = NB * G * oc_blocks;
    int64_t ntb = std::max<int64_t>(1, (nt + waves - 1) / waves);

    // V and M per task are the route's working set. Past L2 (2 MiB) the GEMM's
    // A stream and the output transform's M reads fall to L3 and a wider block
    // loses more than the U passes it saves — 224x224 64ch measured 1.07x of
    // the 12-tile block. A U that no longer fits L2 is worth streaming less
    // often, so the cap follows it; past 12 MiB the arena stops being one
    // pooled allocation, which is dearer than any pass count.
    const int64_t esz = static_cast<int64_t>(data_type_size(dt));
    const int64_t slots = std::max<int64_t>(1, (num_slots > 0) ? num_slots : nt);
    constexpr int64_t kArena = 24 << 20;
    const int64_t workset = std::max<int64_t>(2 << 20, std::min<int64_t>(ug.total * esz, 12 << 20));
    const int64_t slot_cap = std::min<int64_t>(workset / esz, kArena / (esz * slots));
    const int64_t per_tile = std::max<int64_t>(1, ug.pos * (plan.v_lda + plan.m_oc_stride));
    const int64_t tile_cap = std::max<int64_t>(1, slot_cap / per_tile);
    ntb = std::max<int64_t>(ntb, (tiles + tile_cap - 1) / tile_cap);
    // ... but never past the point where a task's U read stops paying.
    ntb = std::min<int64_t>(ntb, std::max<int64_t>(1, tiles / kTileBlockMin(order)));

    plan.tile_block = (tiles + ntb - 1) / ntb;
    plan.num_tile_blocks = (tiles + plan.tile_block - 1) / plan.tile_block;

    const int64_t plane_tasks = NB * G * plan.num_tile_blocks;
    plan.oc_chunks = 1;
    if (plane_tasks < nt && oc_blocks > 1) {
        const int64_t want = (nt + plane_tasks / 2) / plane_tasks;
        plan.oc_chunks = std::clamp<int64_t>(want, 1, oc_blocks);
    }
    plan.num_tasks = plane_tasks * plan.oc_chunks;

    plan.v_p_stride = plan.tile_block * plan.v_lda + 8;
    plan.v_size = ug.pos * plan.v_p_stride;
    plan.m_size = ug.pos * plan.tile_block * plan.m_oc_stride;
    // The ZA route packs one position's A at a time, so one tile block of
    // kc-wide rows is all it ever holds.
    // The packed A is [ceil(tiles/32)][k][32], not dense: a whole 32-row block
    // exists even when the tile block is shorter.
    constexpr int64_t kZaRows = 32; // the kernel's A block: two 16-row faces
    plan.a_size = ug.za ? ((plan.tile_block + kZaRows - 1) / kZaRows) * kZaRows * ug.kc : 0;

    // Round the slot to 16 elements so every slot base keeps the transform's
    // 32-byte f16 stores aligned.
    plan.slot_size = ((plan.v_size + plan.m_size + plan.a_size) + 15) / 16 * 16;
    // The slot count follows the *workers*, not the reported thread count: an
    // over-subscribed grid would otherwise multiply the arena for nothing.
    const int64_t avail = (num_slots > 0) ? num_slots : nt;
    plan.num_slots = (use_thread_slots && avail < plan.num_tasks) ? avail : plan.num_tasks;
    plan.workspace_size =
        static_cast<size_t>(plan.num_slots * plan.slot_size) * static_cast<size_t>(esz);
    return plan;
}

} // namespace
} // namespace impl

Conv2DWinogradPlan get_conv2d_winograd_plan(const Conv2DAttributes& attrs,
                                            std::span<const TensorDesc> inputs,
                                            std::span<const TensorDesc> outputs, int num_threads,
                                            bool use_thread_slots, int64_t num_slots)
{
    return impl::make_plan(attrs, inputs, outputs, num_threads, use_thread_slots,
                                   num_slots);
}

// =========================================================================
//  Public API
// =========================================================================

Status conv2d_winograd_kernel(const Conv2DAttributes& attrs, TensorView& output,
                              std::span<const TensorView> inputs, const ComputeContext& ctx,
                              void* /*workspace*/)
{
    // Never fall back to the reference conv2d here: U is not an HWIO filter.
    NNOPS_ASSERT(inputs.size() >= 2);
    const DataType dt = inputs[0].data_type();
    const bool dtypes_ok = (dt == DataType::f32 || dt == DataType::f16) &&
                           inputs[1].data_type() == dt && output.data_type() == dt &&
                           (inputs.size() <= 2 || inputs[2].data_type() == dt);
    // bf16 is refused rather than asserted on: the U blob's panels are the
    // dense fp pack layout, and a BFMMLA panel would need its own transform —
    // on top of a transform whose fp16 form is already at the precision limit.
    // The implicit-GEMM route carries bf16.
    if (!dtypes_ok) {
        return Status::unsupported;
    }
    NNOPS_ASSERT(inputs[1].layout() == TensorLayout::PackedWinoWeightLast &&
                 "conv2d_winograd_kernel: inputs[1] must be prepacked U "
                 "(run prepackWeights first)");

    std::vector<TensorDesc> descs;
    descs.reserve(inputs.size());
    for (const auto& t : inputs) {
        descs.push_back(t.desc());
    }
    const TensorDesc outs[] = {output.desc()};
    // Slot sizing: per worker when the caller reports worker ids; a single slot
    // when there is no pool (a serial run reuses it task after task); else one per
    // task (see Conv2DWinogradPlan::num_slots).
    const bool serial = !ctx.cpu.parallel_for;
    const Conv2DWinogradPlan plan = get_conv2d_winograd_plan(
        attrs, descs, outs, ctx.cpu.thread_count(), serial || ctx.cpu.thread_slot_count() > 0,
        ctx.cpu.thread_slot_count());
    NNOPS_ASSERT(plan.workspace_size > 0 &&
                 "conv2d_winograd_kernel: unsupported conv geometry (k3s1 only)");

    PoolPtr scratch(plan.workspace_size);
    const bool f43 = plan.order == WinoOrder::F43;
    if (dt == DataType::f32) {
        if (f43) {
            impl::conv2d_winograd_impl<float, WinoOrder::F43>(
                attrs, output, inputs, ctx, plan, scratch.as<float>());
        } else {
            impl::conv2d_winograd_impl<float, WinoOrder::F23>(
                attrs, output, inputs, ctx, plan, scratch.as<float>());
        }
    } else if (f43) {
        impl::conv2d_winograd_impl<half, WinoOrder::F43>(
            attrs, output, inputs, ctx, plan, scratch.as<half>());
    } else {
        impl::conv2d_winograd_impl<half, WinoOrder::F23>(
            attrs, output, inputs, ctx, plan, scratch.as<half>());
    }
    return Status::ok;
}

} // namespace nnops::backend::cpu
