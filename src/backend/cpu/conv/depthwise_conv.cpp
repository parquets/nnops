/// @file depthwise_conv.cpp
/// @brief SIMD-optimized CPU implementation of 2D/3D depthwise convolution in
///        the planar (NCHW/NCDHW) and channel-last (NHWC/NDHWC) layouts.
///
/// Planar input takes the raw [C, 1, (KD,) KH, KW] filter, one channel plane at
/// a time, and vectorizes along output width. Channel-last input takes the raw
/// HWCM filter [KH][KW][C] and vectors across channels, sweeping the channel
/// blocks of a row group inside the task so a pixel's channels are consumed
/// while its cache lines are hot (see dwconv_last_impl). Quantized depthwise
/// conv does not exist on the planar route.
///
/// Spatial rank is auto-detected from input tensor rank (4→2D, 5→3D).
/// A single unified kernel handles both 2D and 3D: the KD loop is always present;
/// for 2D, KD=1 so it degenerates to a single no-op iteration.

#include "nnops/ops/depthwise_conv.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "backend/cpu/simd_kernel/simd_depthwise_conv.hpp"

#include <algorithm>
#include <cstdint>

namespace nnops::backend::cpu {

using namespace nnops::simd;
namespace k = nnops::kernel;

// ============================================================
// Planar depthwise conv (NCHW/NCDHW) — the lane spans output width
// ============================================================

template <typename T>
void dwconv_impl(
    const DepthwiseConvAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t rank  = input.rank();
    const int64_t srank = DepthwiseConvAttributes::spatial_rank(rank);

    // One "block" per channel: a planar tensor has no channel vector.
    const int64_t N      = input.shape(0);
    const int64_t blocks = input.num_channel_blocks();

    // Spatial dims (auto-detect 2D vs 3D)
    const int64_t ID  = (srank == 3) ? input.shape(2) : 1;
    const int64_t IH  = input.shape(srank);
    const int64_t IW  = input.shape(srank + 1);

    const int64_t KD  = attrs.kernel_size[0];
    const int64_t KH  = attrs.kernel_size[1];
    const int64_t KW  = attrs.kernel_size[2];

    const int64_t OD  = (srank == 3) ? output.shape(2) : 1;
    const int64_t OH  = output.shape(srank);
    const int64_t OW  = output.shape(srank + 1);

    const int64_t SD  = attrs.stride[0];
    const int64_t SH  = attrs.stride[1];
    const int64_t SW  = attrs.stride[2];
    const int64_t DD  = attrs.dilation[0];
    const int64_t DH  = attrs.dilation[1];
    const int64_t DW  = attrs.dilation[2];
    const int64_t PD  = attrs.padding[0];
    const int64_t PH  = attrs.padding[1];
    const int64_t PW  = attrs.padding[2];

    auto* out_ptr = output.ptr<T>();
    const auto* in_ptr  = input.ptr<T>();
    const auto* w_ptr   = weight.ptr<T>();
    const auto* b_ptr   = has_bias ? inputs[2].ptr<T>() : nullptr;

    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t in_d_stride    = IH * in_row_stride;
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_d_stride   = OH * out_row_stride;
    const int64_t in_ch_stride   = input.channel_block_stride_elems();
    const int64_t out_ch_stride  = output.channel_block_stride_elems();
    const int64_t taps           = KD * KH * KW;

    // Kernel params (od/oh updated per call)
    k::DwConvParams p{};
    p.ID = ID; p.IH = IH; p.IW = IW;
    p.OH = OH; p.OW = OW;
    p.KD = KD; p.KH = KH; p.KW = KW;
    p.SD = SD; p.SH = SH; p.SW = SW;
    p.DD = DD; p.DH = DH; p.DW = DW;
    p.PD = PD; p.PH = PH; p.PW = PW;
    p.out_row_stride = out_row_stride;
    p.in_d_stride    = in_d_stride;
    p.in_row_stride  = in_row_stride;

    // Planar edge masks. The first and last width vectors of a row are the only
    // ones that read past it, and on a narrow plane they are the only ones
    // there are — that is exactly where planar_load's per-lane scalar loop used
    // to eat the whole kernel. A mask zeroes the lanes that fall outside the
    // row, so the same vector load stays valid and a tap costs a load and an
    // FMA instead of eight scalar iterations. It depends only on (edge, tap),
    // never on the row, so it is built once here for the whole call.
    constexpr int64_t kMaskLanes = simd_lane_for<T>;
    constexpr int64_t kMaskMaxKw = 16;
    uint8_t edge_needed[2 * kMaskMaxKw] = {};
    alignas(32) T edge_masks[2 * kMaskMaxKw * kMaskLanes] = {};
    const k::DwPlanarEdges edges =
        k::dwconv_planar_edge_masks<kMaskMaxKw, T>(p, edge_masks, edge_needed);

    const auto compute_block = [&](int64_t n, int64_t blk) {
        const T* in_base  = in_ptr  + n * blocks * in_ch_stride  + blk * in_ch_stride;
        T*       out_base = out_ptr + n * blocks * out_ch_stride + blk * out_ch_stride;
        // The raw [C, 1, (KD,) KH, KW] filter, one tap block per channel.
        const T* w_base   = w_ptr + blk * taps;
        const float bias  = b_ptr ? s_load(&b_ptr[blk]) : 0.0f;

        // Per-task copy: `p` is shared by every worker, so od/oh cannot live in it.
        k::DwConvParams pc = p;

        // The masked load reads up to PW before the row start and up to
        // L-1+(KW-1)*DW past its end, which lands in the neighbouring row — and
        // is harmless there, since those lanes are masked to zero — for every
        // row but the tensor's first and last. Those two blocks keep the
        // bounds-checked path; that is one channel of a batch, once.
        k::DwPlanarEdges blk_edges = edges;
        if ((n == 0 && blk == 0) || (n == N - 1 && blk == blocks - 1)) {
            blk_edges.masks = nullptr;
        }

        for (int64_t od = 0; od < OD; ++od) {
            T* out_d = out_base + od * out_d_stride;

            // Four rows per call, for the ILP: a single row's taps form one
            // dependent FMA chain. Rows past OH are skipped in-kernel.
            int64_t oh = 0;
            for (; oh + 3 < OH; oh += 4) {
                pc.od = od;
                pc.oh = oh;
                k::dwconv_planar_h4(out_d + oh * pc.out_row_stride, in_base,
                                    w_base, bias, pc, attrs.epilogue, attrs.add_to,
                                    blk_edges);
            }
            for (; oh < OH; ++oh) {
                pc.od = od;
                pc.oh = oh;
                k::dwconv_planar(out_d + oh * pc.out_row_stride, in_base, w_base,
                                 bias, pc, attrs.epilogue, attrs.add_to, blk_edges);
            }
        }
    };

    ctx.cpu.run(0, N * blocks,
            [&](int64_t tid) { compute_block(tid / blocks, tid % blocks); });
}

// ============================================================
// Channel-last (NHWC/NDHWC) depthwise — the lane spans channels
// ============================================================
//
// One task per output row, and every channel block of that row is swept before
// the task moves on: a row of a channel-last plane is IW*C elements and stays
// in cache across the block sweep, so the plane is still read exactly once.
// Splitting the grid by channel block would re-read every cache line C/8
// times.
//
// The weight is the raw HWCM filter [KH][KW][C] (TensorFlow's depthwise order
// with the multiplier dropped), whose tap stride is the channel count: one
// vector per tap is 8 consecutive channels of that tap, which is what the
// channel-lane kernel loads. No prepack step is involved — the layout already
// is the access pattern.
//
// When the plane is too short to fill the pool (seven 7x7 rows against twenty
// workers) the filter axis picks up the rest: the rows are split into channel
// chunks, nearest-rounded rather than ceiled so the pool runs one wave.

template <typename T>
void dwconv_last_impl(const DepthwiseConvAttributes& attrs, TensorView& output,
                      std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t rank  = input.rank();
    const int64_t srank = DepthwiseConvAttributes::spatial_rank(rank);

    const int64_t N = input.shape(0);
    const int64_t C = input.shape(1);

    const int64_t ID = (srank == 3) ? input.shape(2) : 1;
    const int64_t IH = input.shape(srank);
    const int64_t IW = input.shape(srank + 1);
    const int64_t OD = (srank == 3) ? output.shape(2) : 1;
    const int64_t OH = output.shape(srank);
    const int64_t OW = output.shape(srank + 1);

    k::DwConvParams p{};
    p.ID = ID; p.IH = IH; p.IW = IW;
    p.OH = OH; p.OW = OW;
    p.KD = attrs.kernel_size[0];
    p.KH = attrs.kernel_size[1];
    p.KW = attrs.kernel_size[2];
    p.SD = attrs.stride[0];
    p.SH = attrs.stride[1];
    p.SW = attrs.stride[2];
    p.DD = attrs.dilation[0];
    p.DH = attrs.dilation[1];
    p.DW = attrs.dilation[2];
    p.PD = attrs.padding[0];
    p.PH = attrs.padding[1];
    p.PW = attrs.padding[2];
    p.out_row_stride = output.row_stride_elems();
    p.in_row_stride  = input.row_stride_elems();
    p.in_d_stride    = IH * p.in_row_stride;
    p.pix_stride     = C;   // a pixel holds all C channels, and so does a tap
    p.w_tap_stride   = C;   // the raw HWCM filter is [tap][C]

    // Sixteen channels a chunk, always. A pixel's sixteen f32 are one 64-byte
    // line, so a sweep of the block loop touches every line exactly once; an
    // eight-channel chunk would touch each line twice for the same bytes, and a
    // strided sweep costs what a full one costs (measured: half-line reads of a
    // plane run at the plane's full time). The 3x3 kernel answers that by taking
    // both halves of the pixel in one call.
    constexpr int64_t chunk = 16;
    const int64_t nblocks = (C + chunk - 1) / chunk;

    auto* out_ptr = output.ptr<T>();
    const auto* in_ptr = input.ptr<T>();
    const auto* w_ptr  = weight.ptr<T>();
    const auto* b_ptr  = has_bias ? inputs[2].ptr<T>() : nullptr;

    const int64_t out_d_stride = OH * p.out_row_stride;

    // Four output rows per task: one row's taps are a dependent FMA chain,
    // four rows are four independent ones. The last group of a plane shorter
    // than a multiple of four runs single rows.
    //
    // The task is a *strip* of those four rows, not the whole width. A channel-
    // last row group is (4*SH + KH) input rows of IW*C elements, and every one
    // of the C/8 block sweeps walks it again; at C=256 on a 300-wide plane that
    // is 1.8 MB per group and each sweep came back from DRAM, which measured
    // 3x slower. Cutting the width so a strip's working set fits L2 keeps the
    // group resident across the block loop, which is the whole point of
    // sweeping the blocks inside the row.
    const int64_t hgroups = (OH + 3) / 4;
    const int64_t in_rows = 4 * p.SH + (p.KH - 1) * p.DH;  // rows a group reads
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
    const int64_t strips  = (OW + strip - 1) / strip;
    const int64_t groups  = N * OD * hgroups * strips;
    const int64_t nt = ctx.cpu.thread_count();
    // The filter axis only splits a group when the grid cannot fill the pool;
    // one chunk means the block loop runs inside the task, which is what keeps
    // the strip in cache. The split is picked by makespan, not by pool width: a
    // task count the pool cannot divide evenly costs a whole extra wave (21
    // tasks on 20 workers run in two), which is worth more than the idle
    // threads a smaller split leaves behind.
    int64_t chunks = 1;
    if (groups < nt && nblocks > 1) {
        int64_t best_cost = 0;
        for (int64_t c = 1; c <= nblocks; ++c) {
            const int64_t blocks_per = (nblocks + c - 1) / c;
            const int64_t waves = (groups * c + nt - 1) / nt;
            const int64_t cost = waves * blocks_per;
            if (best_cost == 0 || cost < best_cost) {
                best_cost = cost;
                chunks = c;
            }
        }
    }
    const int64_t blocks_per = (nblocks + chunks - 1) / chunks;

    ctx.cpu.run(0, groups * chunks, [&](int64_t tid) {
        const int64_t cc = tid % chunks;
        const int64_t g  = tid / chunks;
        const int64_t st = g % strips;
        const int64_t hg = (g / strips) % hgroups;
        const int64_t od = (g / (strips * hgroups)) % OD;
        const int64_t n  = g / (strips * hgroups * OD);

        // Per-task copy: `p` is shared by every worker, so od/oh cannot live in it.
        k::DwConvParams pc = p;
        pc.od = od;
        pc.oh = hg * 4;
        pc.ow0 = st * strip;
        pc.ow_end = std::min(pc.ow0 + strip, OW);

        const T* in_img = in_ptr + n * ID * p.in_d_stride;
        // The kernel indexes the store by the absolute column, so the base stays at
    // the row start even though the loop covers only this strip.
    T* out_rows = out_ptr + n * OD * out_d_stride + od * out_d_stride
                + pc.oh * p.out_row_stride;

        const int64_t rows_here = std::min<int64_t>(4, OH - pc.oh);
        const int64_t blk0 = cc * blocks_per;
        const int64_t blk1 = std::min(blk0 + blocks_per, nblocks);
        for (int64_t blk = blk0; blk < blk1; ++blk) {
            const int64_t c0 = blk * chunk;
            // One call per block, at the block's own width — full blocks are
            // `chunk` wide, only the plane's last one is short. That is what the
            // two-vector 3x3 kernel is for; handing it the whole remaining
            // channel count instead (`C - c0`) silently sends every full block
            // down the generic two-vector path, which covers sixteen channels
            // but re-derives the row pointers and reloads the taps per column.
            // The exception is 9..15 channels of a sixteen-wide block, which has
            // no single policy shape (two 8-lane vectors would read past the last
            // channel), so it runs as an 8-lane call and its remainder.
            const int64_t rem = std::min(C - c0, chunk);
            int64_t seg_off[2] = {0, 0};
            int64_t seg_lanes[2] = {rem, 0};
            int nseg = 1;
            if (rem > 8 && rem < chunk) {
                seg_lanes[0] = 8;
                seg_off[1] = 8;
                seg_lanes[1] = rem - 8;
                nseg = 2;
            }
            for (int s = 0; s < nseg; ++s) {
                // The channel offset is folded into all three bases: the kernel's
                // row table is built from `in_img`, which serves every tap.
                const int64_t co = c0 + seg_off[s];
                const T* in_blk = in_img + co;
                const T* w_blk  = w_ptr + co;
                const T* b_blk  = b_ptr ? b_ptr + co : nullptr;
                if (rows_here == 4) {
                    k::dwconv_last<4>(out_rows + co, in_blk, w_blk, b_blk, pc,
                                      attrs.epilogue, attrs.add_to, seg_lanes[s]);
                    continue;
                }
                for (int64_t r = 0; r < rows_here; ++r) {
                    k::DwConvParams pr = pc;
                    pr.oh = pc.oh + r;
                    k::dwconv_last<1>(out_rows + r * p.out_row_stride + co, in_blk,
                                      w_blk, b_blk, pr, attrs.epilogue,
                                      attrs.add_to, seg_lanes[s]);
                }
            }
        }
    });
}

// ============================================================
// Channel-last depthwise, quantized (s8/u8 activation, s8/u8 weight)
// ============================================================
//
// Same task grid and the same kernel as the fp route; the weight is HWCM s8/u8
// and is dequantized per row per block into a staging vector (see
// dwconv_last_quant). The activation is PerTensor and the weight PerChannel.

template <typename InT, typename OutT>
Status dwconv_last_impl_quant(const DepthwiseConvAttributes& attrs, TensorView& output,
                              std::span<const TensorView> inputs,
                              const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;
    NNOPS_ASSERT(!has_bias || inputs[2].data_type() == DataType::f32);

    const int64_t rank  = input.rank();
    const int64_t srank = DepthwiseConvAttributes::spatial_rank(rank);

    const int64_t N = input.shape(0);
    const int64_t C = input.shape(1);
    const int64_t nblocks = (C + 7) / 8;

    const int64_t ID = (srank == 3) ? input.shape(2) : 1;
    const int64_t IH = input.shape(srank);
    const int64_t IW = input.shape(srank + 1);
    const int64_t OD = (srank == 3) ? output.shape(2) : 1;
    const int64_t OH = output.shape(srank);
    const int64_t OW = output.shape(srank + 1);

    if (attrs.kernel_size[0] * attrs.kernel_size[1] * attrs.kernel_size[2] >
        k::kMaxLastQuantTaps) {
        return Status::unsupported;   // the staging buffer is sized for 9x9x9
    }

    const auto& in_qp  = input.quant_params();
    const auto& out_qp = output.quant_params();
    const auto& w_qp   = weight.quant_params();
    if (w_qp.granularity != QuantGranularity::PerChannel ||
        w_qp.scale_data == nullptr) {
        return Status::invalid_argument;
    }
    const float in_scale  = in_qp.scale;
    const float in_zero   = static_cast<float>(in_qp.zero_point);
    const float out_scale = out_qp.scale;
    const float out_zero  = static_cast<float>(out_qp.zero_point);

    k::DwConvParams p{};
    p.ID = ID; p.IH = IH; p.IW = IW;
    p.OH = OH; p.OW = OW;
    p.KD = attrs.kernel_size[0];
    p.KH = attrs.kernel_size[1];
    p.KW = attrs.kernel_size[2];
    p.SD = attrs.stride[0];
    p.SH = attrs.stride[1];
    p.SW = attrs.stride[2];
    p.DD = attrs.dilation[0];
    p.DH = attrs.dilation[1];
    p.DW = attrs.dilation[2];
    p.PD = attrs.padding[0];
    p.PH = attrs.padding[1];
    p.PW = attrs.padding[2];
    p.out_row_stride = output.row_stride_elems();
    p.in_row_stride  = input.row_stride_elems();
    p.in_d_stride    = IH * p.in_row_stride;
    p.pix_stride     = C;
    p.w_tap_stride   = C;   // the raw HWCM filter is [tap][C]

    auto* out_ptr = output.ptr<OutT>();
    const auto* in_ptr = input.ptr<InT>();
    const auto* w_ptr  = weight.ptr<InT>();
    const auto* b_ptr  = has_bias ? inputs[2].ptr<float>() : nullptr;

    const int64_t out_d_stride = OH * p.out_row_stride;

    // Four output rows per task, as on the fp route and for the same reason.
    const int64_t hgroups = (OH + 3) / 4;
    const int64_t in_rows = 4 * p.SH + (p.KH - 1) * p.DH;  // rows a group reads
    constexpr int64_t kStripBytes = 128 * 1024;
    int64_t strip = OW;
    {
        const int64_t per_col = in_rows * C * static_cast<int64_t>(sizeof(OutT));
        const int64_t cols = (per_col > 0) ? kStripBytes / per_col : OW;
        if (cols < p.SW * OW) {
            const int64_t w = (cols - (p.KW - 1) * p.DW) / p.SW;
            strip = std::clamp<int64_t>(w & ~int64_t{15}, 16, OW);
        }
    }
    const int64_t strips  = (OW + strip - 1) / strip;
    const int64_t groups  = N * OD * hgroups * strips;
    const int64_t nt = ctx.cpu.thread_count();
    int64_t cchunks = 1;
    if (groups < nt && nblocks > 1) {
        cchunks = std::min<int64_t>((nt + groups / 2) / groups, nblocks);
    }
    const int64_t blocks_per = (nblocks + cchunks - 1) / cchunks;

    ctx.cpu.run(0, groups * cchunks, [&](int64_t tid) {
        const int64_t cc = tid % cchunks;
        const int64_t g  = tid / cchunks;
        const int64_t st = g % strips;
        const int64_t hg = (g / strips) % hgroups;
        const int64_t od = (g / (strips * hgroups)) % OD;
        const int64_t n  = g / (strips * hgroups * OD);

        k::DwConvParams pc = p;
        pc.od = od;
        pc.oh = hg * 4;
        pc.ow0 = st * strip;
        pc.ow_end = std::min(pc.ow0 + strip, OW);

        const InT* in_img = in_ptr + n * ID * p.in_d_stride;
        OutT* out_rows = out_ptr + n * OD * out_d_stride + od * out_d_stride
                       + pc.oh * p.out_row_stride;

        const int64_t rows_here = std::min<int64_t>(4, OH - pc.oh);
        const int64_t blk0 = cc * blocks_per;
        const int64_t blk1 = std::min(blk0 + blocks_per, nblocks);
        for (int64_t blk = blk0; blk < blk1; ++blk) {
            const int64_t c0 = blk * 8;
            const int64_t lanes = std::min<int64_t>(8, C - c0);
            const InT* in_blk = in_img + c0;
            const InT* w_blk  = w_ptr + c0;
            const float* b_blk = b_ptr ? b_ptr + c0 : nullptr;
            const float* ws = w_qp.scale_data + c0;
            const int32_t* wz = w_qp.zero_point_data ? w_qp.zero_point_data + c0
                                                     : nullptr;
            if (rows_here == 4) {
                k::dwconv_last_quant<4, InT, OutT>(
                    out_rows + c0, in_blk, w_blk, ws, wz, b_blk, pc, attrs.epilogue,
                    attrs.add_to, lanes, in_scale, in_zero, out_scale, out_zero);
                continue;
            }
            for (int64_t r = 0; r < rows_here; ++r) {
                k::DwConvParams pr = pc;
                pr.oh = pc.oh + r;
                k::dwconv_last_quant<1, InT, OutT>(
                    out_rows + r * p.out_row_stride + c0, in_blk, w_blk, ws, wz, b_blk,
                    pr, attrs.epilogue, attrs.add_to, lanes, in_scale, in_zero,
                    out_scale, out_zero);
            }
        }
    });
    return Status::ok;
}

// ============================================================
// Entry point with dtype dispatch
// ============================================================

Status depthwise_conv_cpu(const DepthwiseConvAttributes& attrs,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    const auto odtype = output.data_type();

    // Quantized depthwise conv lives on the channel lane: its dequantize-on-load
    // kernels read a channel vector, which a planar plane does not have.
    if (!is_channel_last(inputs[0].layout()) &&
        (dtype == DataType::s8 || dtype == DataType::u8)) {
        return Status::unsupported;
    }

    if (is_channel_last(inputs[0].layout())) {
        switch (dtype) {
        case DataType::f32:
            dwconv_last_impl<float>(attrs, output, inputs, ctx);
            return Status::ok;
        case DataType::f16:
            dwconv_last_impl<half>(attrs, output, inputs, ctx);
            return Status::ok;
        case DataType::s8:
            if (odtype == DataType::s8) {
                return dwconv_last_impl_quant<int8_t, int8_t>(attrs, output, inputs, ctx);
            }
            if (odtype == DataType::u8) {
                return dwconv_last_impl_quant<int8_t, uint8_t>(attrs, output, inputs, ctx);
            }
            return Status::unsupported;
        case DataType::u8:
            if (odtype == DataType::s8) {
                return dwconv_last_impl_quant<uint8_t, int8_t>(attrs, output, inputs, ctx);
            }
            if (odtype == DataType::u8) {
                return dwconv_last_impl_quant<uint8_t, uint8_t>(attrs, output, inputs, ctx);
            }
            return Status::unsupported;
        default:
            return Status::unsupported;
        }
    }

    switch (dtype) {
    case DataType::f32:
        dwconv_impl<float>(attrs, output, inputs, ctx);
        return Status::ok;
    case DataType::f16:
        dwconv_impl<half>(attrs, output, inputs, ctx);
        return Status::ok;
    default:
        return Status::unsupported;   // planar has no quantized route
    }
}

}  // namespace nnops::backend::cpu
