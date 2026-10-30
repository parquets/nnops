/// @file pooling.cpp
/// @brief SIMD-optimized CPU implementation of 2D/3D spatial pooling in the
///        planar (NCHW/NCDHW) and channel-last (NHWC/NDHWC) layouts.
///
/// Key design decisions:
///   1. The layout picks the vector lane. Channel-last: the lane is the
///      channel, and a plane's last block runs short. Planar: one channel plane
///      at a time, the lane runs along output width.
///   2. SIMD kernels handle exclude_pad via in-kernel per-position valid_count.
///   3. Height-4 blocking (4 output rows per call) on the channel lane; the row
///      driver skips rows past OH, so the channel-last route needs no h1 tail.
///      A planar row has no h4 form.
///   4. In-kernel bounds checking for all kernel positions — no pre-splitting.
///   5. All SW values supported — offset math is just integer arithmetic.
///   6. Task grid: (n, block) when global, and (n, od, row group) with the
///      channel blocks swept inside the task when channel-last — see
///      pooling_last_impl.
///   7. Params passed by const reference (unified across all kernels).
///
/// Quantized pooling runs on the channel lane, so it covers the channel-last
/// layout; on planar it stays unsupported.

#include "nnops/ops/pooling.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "backend/cpu/simd_kernel/simd_pooling.hpp"

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace nnops::backend::cpu {

using namespace nnops::simd;
namespace k = nnops::kernel;

namespace {

template <typename T>
void pooling_impl(const PoolingAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t rank  = input.rank();
    const int64_t srank = PoolingAttributes::spatial_rank(rank);

    // One "block" per channel: a planar tensor has no channel vector.
    const int64_t N      = input.shape(0);
    const int64_t blocks = input.num_channel_blocks();

    const int64_t ID = (srank == 3) ? input.shape(2) : 1;
    const int64_t IH = input.shape(srank);
    const int64_t IW = input.shape(srank + 1);

    const int64_t OD = (srank == 3) ? output.shape(2) : 1;
    const int64_t OH = output.shape(srank);
    const int64_t OW = output.shape(srank + 1);

    const int64_t KD = (srank == 3) ? attrs.kernel_shape[0] : 1;
    const int64_t KH = attrs.kernel_shape[1];
    const int64_t KW = attrs.kernel_shape[2];

    const int64_t SD = (srank == 3) ? attrs.stride[0] : 1;
    const int64_t SH = attrs.stride[1];
    const int64_t SW = attrs.stride[2];

    const int64_t DD_ = (srank == 3) ? attrs.dilation[0] : 1;
    const int64_t DH  = attrs.dilation[1];
    const int64_t DW  = attrs.dilation[2];

    const int64_t PD = (srank == 3) ? attrs.padding[0] : 0;
    const int64_t PH = attrs.padding[1];
    const int64_t PW = attrs.padding[2];

    auto* out_ptr = output.ptr<T>();
    const auto* in_ptr = input.ptr<T>();

    // Kernel params (shared across all calls, od/oh updated per call)
    k::PoolingKernelParams p{};
    p.ID = ID; p.IH = IH; p.IW = IW;
    p.OH = OH; p.OW = OW;
    p.KD = KD; p.KH = KH; p.KW = KW;
    p.SD = SD; p.SH = SH; p.SW = SW;
    p.DD = DD_; p.DH = DH; p.DW = DW;
    p.PD = PD; p.PH = PH; p.PW = PW;
    p.out_row_stride = output.row_stride_elems();
    p.in_d_stride    = IH * input.row_stride_elems();
    p.in_row_stride  = input.row_stride_elems();
    p.exclude_pad    = attrs.exclude_pad;

    const int64_t in_ch_stride  = input.channel_block_stride_elems();
    const int64_t out_ch_stride = output.channel_block_stride_elems();
    const float avg_scale = 1.0f / static_cast<float>(KD * KH * KW);

    // Planar rows are vectorized along W, so there is one kernel and no h4 form.
    using PoolingFn = void (*)(T*, const T*, const k::PoolingKernelParams&,
                                float, bool);
    const PoolingFn pool_fn = (attrs.type == PoolingType::Max)
        ? k::maxpool_planar<T> : k::avgpool_planar<T>;

    const auto compute_block = [&](int64_t n, int64_t blk) {
        const T* in_base  = in_ptr + n * blocks * in_ch_stride + blk * in_ch_stride;
        T* out_base = out_ptr + n * blocks * out_ch_stride + blk * out_ch_stride;

        // Per-task copy: `p` is shared by every worker, so od/oh cannot live in it.
        k::PoolingKernelParams pc = p;

        for (int64_t od = 0; od < OD; ++od) {
            T* out_d = out_base + od * pc.out_row_stride * OH;

            for (int64_t oh = 0; oh < OH; ++oh) {
                pc.od = od;
                pc.oh = oh;
                pool_fn(out_d + oh * pc.out_row_stride, in_base,
                        pc, avg_scale, attrs.add_to);
            }
        }
    };

    ctx.cpu.run(0, N * blocks,
            [&](int64_t tid) { compute_block(tid / blocks, tid % blocks); });
}

// ============================================================
// Channel-last (NHWC/NDHWC) pooling — the lane spans channels
// ============================================================
//
// The kernel is the channel-lane one (pool_last): the lane is the channel and
// the window walk is the same at any pixel distance. What the layout decides is
// the grid. On a channel-last plane a cache line holds a run of one pixel's
// channels, so a
// (n, block) task per block would make every line be fetched C/8 times over —
// the pixel is only whole across the channel blocks. A task therefore owns a
// group of four output rows *and* every channel block of them: the row group's
// window (at most (4*SH + KH - SH) input rows) stays in cache for one sweep,
// and the plane is read once.
//
// When the plane is too short to fill the pool (seven 7x7 rows against twenty
// workers) the channel axis picks up the rest, nearest-rounded rather than
// ceiled so the pool runs one wave.

template <bool IsMax, typename T>
void pooling_last_impl(const PoolingAttributes& attrs,
                       TensorView& output,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t srank = PoolingAttributes::spatial_rank(input.rank());

    const int64_t N = input.shape(0);
    const int64_t C = input.shape(1);
    // Sixteen channels a chunk, not eight: a pixel's 16 f32 are one 64-byte
    // line, and a chunk that spans a whole line touches it once instead of
    // leaving the other half for the next chunk to fetch again.
    const int64_t nblocks = (C + 15) / 16;

    const int64_t ID = (srank == 3) ? input.shape(2) : 1;
    const int64_t IH = input.shape(srank);
    const int64_t IW = input.shape(srank + 1);
    const int64_t OD = (srank == 3) ? output.shape(2) : 1;
    const int64_t OH = output.shape(srank);
    const int64_t OW = output.shape(srank + 1);

    k::PoolingKernelParams p{};
    p.ID = ID; p.IH = IH; p.IW = IW;
    p.OH = OH; p.OW = OW;
    p.KD = (srank == 3) ? attrs.kernel_shape[0] : 1;
    p.KH = attrs.kernel_shape[1];
    p.KW = attrs.kernel_shape[2];
    p.SD = (srank == 3) ? attrs.stride[0] : 1;
    p.SH = attrs.stride[1];
    p.SW = attrs.stride[2];
    p.DD = (srank == 3) ? attrs.dilation[0] : 1;
    p.DH = attrs.dilation[1];
    p.DW = attrs.dilation[2];
    p.PD = (srank == 3) ? attrs.padding[0] : 0;
    p.PH = attrs.padding[1];
    p.PW = attrs.padding[2];
    p.out_row_stride = output.row_stride_elems();
    p.in_row_stride  = input.row_stride_elems();
    p.in_d_stride    = IH * p.in_row_stride;
    p.pix_stride     = C;   // a pixel is C channels wide
    p.exclude_pad    = attrs.exclude_pad;

    const float avg_scale = 1.0f / static_cast<float>(p.KD * p.KH * p.KW);

    auto* out_ptr = output.ptr<T>();
    const auto* in_ptr = input.ptr<T>();

    // A task is a strip of four output rows: the row group's window is
    // (4*SH + KH - SH) input rows of IW*C elements, and every channel chunk
    // walks it again. Cutting the width so a strip fits L2 keeps it resident
    // across the chunk sweep, which is what the sweep is for.
    const int64_t hgroups = (OH + 3) / 4;   // four output rows per task
    const int64_t in_rows = 4 * p.SH + (p.KH - 1) * p.DH;
    constexpr int64_t kStripBytes = 128 * 1024;
    int64_t strip = OW;
    {
        const int64_t per_col = in_rows * C * static_cast<int64_t>(sizeof(T));
        const int64_t cols = (per_col > 0) ? kStripBytes / per_col : OW;
        if (cols < p.SW * OW) {
            const int64_t w = (cols - (p.KW - 1) * p.DW) / p.SW;
            strip = std::clamp<int64_t>(w & ~int64_t{15}, 16, OW);
        }
    }
    const int64_t strips = (OW + strip - 1) / strip;
    const int64_t groups  = N * OD * hgroups * strips;
    const int64_t nt = ctx.cpu.thread_count();
    // The channel axis only splits when the grid cannot fill the pool, and the
    // split is picked by makespan rather than by pool width: a task count the
    // pool cannot divide evenly costs a whole extra wave.
    int64_t cchunks = 1;
    if (groups < nt && nblocks > 1) {
        int64_t best_cost = 0;
        for (int64_t c = 1; c <= nblocks; ++c) {
            const int64_t blocks_per = (nblocks + c - 1) / c;
            const int64_t waves = (groups * c + nt - 1) / nt;
            const int64_t cost = waves * blocks_per;
            if (best_cost == 0 || cost < best_cost) {
                best_cost = cost;
                cchunks = c;
            }
        }
    }
    const int64_t blocks_per = (nblocks + cchunks - 1) / cchunks;

    ctx.cpu.run(0, groups * cchunks, [&](int64_t tid) {
        const int64_t cc = tid % cchunks;
        const int64_t g  = tid / cchunks;
        const int64_t st = g % strips;
        const int64_t hg = (g / strips) % hgroups;
        const int64_t od = (g / (strips * hgroups)) % OD;
        const int64_t n  = g / (strips * hgroups * OD);

        // Per-task copy: `p` is shared by every worker, so od/oh cannot live in it.
        k::PoolingKernelParams pc = p;
        pc.od = od;
        pc.oh = hg * 4;
        pc.ow0 = st * strip;
        pc.ow_end = std::min(pc.ow0 + strip, OW);

        const T* in_img = in_ptr + n * ID * p.in_d_stride;
        // The kernel indexes the store by the absolute column, so the base stays
        // at the row start even though the loop covers only this strip.
        T* out_row = out_ptr + n * OD * OH * p.out_row_stride
                   + od * OH * p.out_row_stride + pc.oh * p.out_row_stride;

        const int64_t blk0 = cc * blocks_per;
        const int64_t blk1 = std::min(blk0 + blocks_per, nblocks);
        for (int64_t blk = blk0; blk < blk1; ++blk) {
            const int64_t c0 = blk * 16;
            // A chunk of 16 channels is one call; a ragged tail of 9..15 has no
            // single policy shape, so it runs as an 8-lane call and its remainder.
            int64_t seg_off[2] = {0, 0};
            int64_t seg_lanes[2] = {C - c0, 0};
            int nseg = 1;
            if (C - c0 > 16) {
                seg_lanes[0] = 16;
            } else if (C - c0 > 8) {
                seg_lanes[0] = 8;
                seg_off[1] = 8;
                seg_lanes[1] = C - c0 - 8;
                nseg = 2;
            }
            for (int s = 0; s < nseg; ++s) {
                const int64_t co = c0 + seg_off[s];
                if constexpr (IsMax) {
                    k::pool_last<true>(out_row + co, in_img + co, pc, 0.0f,
                                       attrs.add_to, seg_lanes[s]);
                } else {
                    k::pool_last<false>(out_row + co, in_img + co, pc, avg_scale,
                                        attrs.add_to, seg_lanes[s]);
                }
            }
        }
    });
}

// ============================================================
// Global pooling — one output per channel plane, no window walk
// ============================================================

template <bool IsMax, typename T>
void pooling_global_impl(const PoolingAttributes& attrs,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t srank = PoolingAttributes::spatial_rank(input.rank());
    const int64_t N      = input.shape(0);
    const int64_t C      = input.shape(1);
    const bool last      = is_channel_last(input.layout());
    // Blocks: one per channel plane for planar, ceil(C/16) for a channel-last
    // one. A channel-last task reads kGlobalCh channels of one pixel per visit:
    // 16 f32 is a whole 64-byte line, so the plane is fetched once rather than
    // twice, and a task still has work small enough to split across the pool.
    constexpr int64_t kGlobalCh = 16;
    const int64_t blocks = last ? (C + kGlobalCh - 1) / kGlobalCh
                                : input.num_channel_blocks();

    k::PoolingKernelParams p{};
    p.ID = (srank == 3) ? input.shape(2) : 1;
    p.IH = input.shape(srank);
    p.IW = input.shape(srank + 1);
    p.in_d_stride   = p.IH * input.row_stride_elems();
    p.in_row_stride = input.row_stride_elems();
    p.pix_stride    = last ? C : 8;

    const double count = static_cast<double>(p.ID) * p.IH * p.IW;
    const float scale = (count > 0.0) ? static_cast<float>(1.0 / count) : 0.0f;

    // The global output is one pixel per (n, channel), so a channel-last
    // image's stride is its whole plane — which is what the block stride is.
    const int64_t in_ch_stride  = last ? static_cast<int64_t>(p.ID) * p.in_d_stride
                                       : input.channel_block_stride_elems();
    const int64_t out_ch_stride = output.channel_block_stride_elems();
    const auto* in_ptr = input.ptr<T>();
    auto* out_ptr = output.ptr<T>();

    ctx.cpu.run(0, N * blocks, [&](int64_t tid) {
        const int64_t n   = tid / blocks;
        const int64_t blk = tid % blocks;

        if (last) {
            // A task that reads a whole line per pixel takes the plane in one
            // pass instead of leaving the other half of every line to another
            // one. A ragged tail of 9..15 channels has no single shape, so it
            // runs as an 8-lane task and its remainder.
            const int64_t c0    = blk * kGlobalCh;
            const int64_t lanes = std::min<int64_t>(kGlobalCh, C - c0);
            const T* in_base  = in_ptr + n * in_ch_stride + c0;
            T* out_base = out_ptr + n * out_ch_stride + c0;
            if (lanes == 16) {
                k::global_channel_vec<IsMax, T, 2>(out_base, in_base, p, scale,
                                                   attrs.add_to, lanes);
            } else if (lanes > 8) {
                k::global_channel_vec<IsMax, T, 2, 4, false>(out_base, in_base, p,
                                                             scale, attrs.add_to,
                                                             lanes);
            } else {
                k::global_store_lanes(out_base,
                                      k::global_channel<IsMax>(in_base, p, scale,
                                                               lanes),
                                      attrs.add_to, lanes);
            }
            return;
        }

        const T* in_base = in_ptr + tid * in_ch_stride;
        T* out_base = out_ptr + tid * out_ch_stride;

        s_store_add(out_base, k::global_planar<IsMax>(in_base, p, scale),
                    attrs.add_to);
    });
}

/// The fp driver: a native plane scan when global, the window walk otherwise.
template <typename T>
void pooling_fp(const PoolingAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx)
{
    const bool last = is_channel_last(inputs[0].layout());
    if (!attrs.global) {
        if (!last) {
            pooling_impl<T>(attrs, output, inputs, ctx);
        } else if (attrs.type == PoolingType::Max) {
            pooling_last_impl<true, T>(attrs, output, inputs, ctx);
        } else {
            pooling_last_impl<false, T>(attrs, output, inputs, ctx);
        }
    } else if (attrs.type == PoolingType::Max) {
        pooling_global_impl<true, T>(attrs, output, inputs, ctx);
    } else {
        pooling_global_impl<false, T>(attrs, output, inputs, ctx);
    }
}

/// A global window spelled out as the full extent. The quantized kernels have
/// no whole-plane form, but reducing a window that covers the input is the
/// same reduction — the walk just costs its per-tap bounds checks.
inline PoolingAttributes full_extent_window(PoolingAttributes attrs,
                                            const TensorView& in)
{
    const int64_t srank = PoolingAttributes::spatial_rank(in.rank());
    const auto first = static_cast<size_t>(3 - srank);
    for (int64_t d = 0; d < srank; ++d) {
        attrs.kernel_shape[first + static_cast<size_t>(d)] = in.shape(2 + d);
        attrs.stride[first + static_cast<size_t>(d)]   = 1;
        attrs.padding[first + static_cast<size_t>(d)]  = 0;
        attrs.dilation[first + static_cast<size_t>(d)] = 1;
    }
    attrs.global = false;
    return attrs;
}

// ============================================================
// Quantized pooling (s8/u8 input/output)
// ============================================================
//
// Activation is PerTensor-quantized (broadcast scale/zero_point). Values are
// dequantized to float, pooled, and requantized on store (round-to-nearest-even).
// MaxPooling additionally takes a raw-integer fast path when input and output
// share the same dtype, scale and zero_point (and !add_to) — then `max` in the
// integer domain matches `max` in the float domain exactly.

/// Channel-last quantized pooling — the fp grid (see pooling_last_impl) and the
/// same dequantize/requantize kernels, with the raw-integer max kept for the
/// full blocks: an 8-lane integer max is exact when input and output share their
/// scale and zero point, and a short last block takes the general path instead
/// of a masked copy of it.
template <bool IsMax, typename InT, typename OutT>
void pooling_last_impl_quant(const PoolingAttributes& attrs,
                             TensorView& output,
                             std::span<const TensorView> inputs,
                             const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t srank = PoolingAttributes::spatial_rank(input.rank());

    const int64_t N = input.shape(0);
    const int64_t C = input.shape(1);
    const int64_t nblocks = (C + 7) / 8;

    const int64_t ID = (srank == 3) ? input.shape(2) : 1;
    const int64_t IH = input.shape(srank);
    const int64_t IW = input.shape(srank + 1);
    const int64_t OD = (srank == 3) ? output.shape(2) : 1;
    const int64_t OH = output.shape(srank);
    const int64_t OW = output.shape(srank + 1);

    const auto& in_qp  = input.quant_params();
    const auto& out_qp = output.quant_params();
    const float in_scale  = in_qp.scale;
    const float in_zero   = static_cast<float>(in_qp.zero_point);
    const float out_scale = out_qp.scale;
    const float out_zero  = static_cast<float>(out_qp.zero_point);

    k::PoolingKernelParams p{};
    p.ID = ID; p.IH = IH; p.IW = IW;
    p.OH = OH; p.OW = OW;
    p.KD = (srank == 3) ? attrs.kernel_shape[0] : 1;
    p.KH = attrs.kernel_shape[1];
    p.KW = attrs.kernel_shape[2];
    p.SD = (srank == 3) ? attrs.stride[0] : 1;
    p.SH = attrs.stride[1];
    p.SW = attrs.stride[2];
    p.DD = (srank == 3) ? attrs.dilation[0] : 1;
    p.DH = attrs.dilation[1];
    p.DW = attrs.dilation[2];
    p.PD = (srank == 3) ? attrs.padding[0] : 0;
    p.PH = attrs.padding[1];
    p.PW = attrs.padding[2];
    p.out_row_stride = output.row_stride_elems();
    p.in_row_stride  = input.row_stride_elems();
    p.in_d_stride    = IH * p.in_row_stride;
    p.pix_stride     = C;
    p.exclude_pad    = attrs.exclude_pad;

    const float avg_scale = 1.0f / static_cast<float>(p.KD * p.KH * p.KW);

    const bool use_fast = IsMax && std::is_same_v<InT, OutT>
                       && !attrs.add_to
                       && (in_qp.scale == out_qp.scale)
                       && (in_qp.zero_point == out_qp.zero_point);

    auto* out_ptr = output.ptr<OutT>();
    const auto* in_ptr = input.ptr<InT>();

    const int64_t hgroups = (OH + 3) / 4;
    const int64_t groups  = N * OD * hgroups;
    const int64_t nt = ctx.cpu.thread_count();
    int64_t cchunks = 1;
    if (groups < nt && nblocks > 1) {
        cchunks = std::min<int64_t>((nt + groups / 2) / groups, nblocks);
    }
    const int64_t blocks_per = (nblocks + cchunks - 1) / cchunks;

    ctx.cpu.run(0, groups * cchunks, [&](int64_t tid) {
        const int64_t cc = tid % cchunks;
        const int64_t g  = tid / cchunks;
        const int64_t hg = g % hgroups;
        const int64_t od = (g / hgroups) % OD;
        const int64_t n  = g / (hgroups * OD);

        k::PoolingKernelParams pc = p;
        pc.od = od;
        pc.oh = hg * 4;

        const InT* in_img = in_ptr + n * ID * p.in_d_stride;
        OutT* out_row = out_ptr + n * OD * OH * p.out_row_stride
                      + od * OH * p.out_row_stride + pc.oh * p.out_row_stride;

        const int64_t blk0 = cc * blocks_per;
        const int64_t blk1 = std::min(blk0 + blocks_per, nblocks);
        for (int64_t blk = blk0; blk < blk1; ++blk) {
            const int64_t c0 = blk * 8;
            const int64_t lanes = std::min<int64_t>(8, C - c0);
            if constexpr (IsMax) {
                if constexpr (std::is_same_v<InT, OutT>) {
                    if (use_fast && lanes == k::kChannelLanes) {
                        k::maxpool_fast<4>(out_row + c0, in_img + c0, pc, out_zero);
                        continue;
                    }
                }
                k::pool_last_quant<true, InT, OutT>(
                    out_row + c0, in_img + c0, pc, 0.0f, in_scale, in_zero,
                    out_scale, out_zero, attrs.add_to, lanes);
            } else {
                k::pool_last_quant<false, InT, OutT>(
                    out_row + c0, in_img + c0, pc, avg_scale, in_scale, in_zero,
                    out_scale, out_zero, attrs.add_to, lanes);
            }
        }
    });
}

/// Layout dispatch shared by the four quantized instantiations. Quantized
/// pooling is channel-lane only, so there is one route and a type dispatch.
template <typename InT, typename OutT>
void pooling_quant(const PoolingAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx)
{
    if (attrs.type == PoolingType::Max) {
        pooling_last_impl_quant<true, InT, OutT>(attrs, output, inputs, ctx);
    } else {
        pooling_last_impl_quant<false, InT, OutT>(attrs, output, inputs, ctx);
    }
}

}  // anonymous namespace

Status pooling_cpu(const PoolingAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx,
                  void* /*workspace*/)
{
    const auto dtype  = inputs[0].data_type();
    const auto odtype = output.data_type();
    const bool quant = (dtype == DataType::s8 || dtype == DataType::u8);

    // Quantized pooling lives on the channel lane, so it needs a layout that
    // has one: channel-last. A planar plane has no channel vector.
    if (quant && !is_channel_last(inputs[0].layout())) {
        return Status::unsupported;
    }
    // ...and it has no whole-plane form, so a global request takes the window
    // walk with the window spelled out as the full extent.
    const PoolingAttributes qattrs =
        (quant && attrs.global) ? full_extent_window(attrs, inputs[0]) : attrs;

    switch (dtype) {
    case DataType::f32:
        pooling_fp<float>(attrs, output, inputs, ctx);
        return Status::ok;
    case DataType::f16:
        pooling_fp<half>(attrs, output, inputs, ctx);
        return Status::ok;
    case DataType::s8:
        if (odtype == DataType::s8) {
            pooling_quant<int8_t, int8_t>(qattrs, output, inputs, ctx);
        } else if (odtype == DataType::u8) {
            pooling_quant<int8_t, uint8_t>(qattrs, output, inputs, ctx);
        } else {
            NNOPS_ASSERT(!"pooling_cpu: s8 input requires s8/u8 output");
        }
        return Status::ok;
    case DataType::u8:
        if (odtype == DataType::s8) {
            pooling_quant<uint8_t, int8_t>(qattrs, output, inputs, ctx);
        } else if (odtype == DataType::u8) {
            pooling_quant<uint8_t, uint8_t>(qattrs, output, inputs, ctx);
        } else {
            NNOPS_ASSERT(!"pooling_cpu: u8 input requires s8/u8 output");
        }
        return Status::ok;
    default:
        NNOPS_ASSERT(!"pooling_cpu: unsupported data type");
    }
    return Status::ok;
}

}  // namespace nnops::backend::cpu
