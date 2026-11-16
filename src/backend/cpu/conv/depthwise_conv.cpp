/// @file depthwise_conv.cpp
/// @brief SIMD CPU depthwise 2D/3D convolution, channel-last.
///
/// Keeps the raw HWCM filter [KH][KW][C] (TF depthwise order, multiplier
/// dropped) and puts the lane on channels; all channel blocks of a row group are
/// swept inside one task so the group's input rows stay in cache across the
/// sweep. No prepack step is involved — the storage order already is the access
/// pattern. A task owns a strip of four rows cut to fit L2, and when the grid
/// cannot fill the pool the filter axis splits a strip into channel chunks.
///
/// Spatial rank comes from the input rank (4 → 2D, 5 → 3D). One kernel covers
/// both: the KD loop is always present, and for 2D KD = 1 makes it a single
/// iteration.
///
/// Dispatch:
///   depthwise_conv_cpu
///     f32/f16/bf16       -> dwconv_channel_last_impl       (lane on C)
///     s8/u8              -> dwconv_channel_last_quant_impl

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
// Channel-last depthwise — the lane spans channels
// ============================================================

template <typename T>
void dwconv_channel_last_impl(const DepthwiseConvAttributes& attrs, TensorView& output,
                              std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;
    // At most one of the two: the pre-activation order folds the destination
    // into the accumulator, so nothing is left for the post-epilogue add.
    const bool pre_act = attrs.add_to && attrs.residual_pre_activation;
    const bool add_to = attrs.add_to && !pre_act;

    const int64_t rank = input.rank();
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
    p.ID = ID;
    p.IH = IH;
    p.IW = IW;
    p.OH = OH;
    p.OW = OW;
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
    p.in_row_stride = input.row_stride_elems();
    p.in_d_stride = IH * p.in_row_stride;
    p.pix_stride = C;   // a pixel holds all C channels, and so does a tap
    p.w_tap_stride = C; // the raw HWCM filter is [tap][C]

    // Sixteen channels a chunk: a pixel's sixteen f32 are one 64-byte line, so
    // the block sweep touches every line exactly once. An eight-channel chunk
    // touches each line twice, and a strided sweep costs what a full one does
    // (measured: half-line reads of a plane run at the plane's full time). The
    // 3x3 kernel takes both halves of the pixel in one call instead.
    constexpr int64_t chunk = 16;
    const int64_t nblocks = (C + chunk - 1) / chunk;

    auto* out_ptr = output.ptr<T>();
    const auto* in_ptr = input.ptr<T>();
    const auto* w_ptr = weight.ptr<T>();
    const auto* b_ptr = has_bias ? inputs[2].ptr<T>() : nullptr;

    const int64_t out_d_stride = OH * p.out_row_stride;

    // Four output rows per task: one row's taps are a dependent FMA chain, four
    // rows are four independent ones. The task is a *strip* of them, not the
    // whole width: a group's 4*SH + (KH-1)*DH input rows are IW*C each and every
    // block sweep walks them again — 1.8 MB at C=256 on a 300-wide plane, so
    // each sweep came back from DRAM and measured 3x slower. A strip that fits
    // L2 stays resident across the sweep.
    const int64_t hgroups = (OH + 3) / 4;
    const int64_t in_rows = 4 * p.SH + (p.KH - 1) * p.DH; // rows a group reads
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
    const int64_t groups = N * OD * hgroups * strips;
    const int64_t nt = ctx.cpu.thread_count();
    // The filter axis splits a group only when the grid cannot fill the pool;
    // one chunk keeps the block loop inside the task and the strip in cache. The
    // split is picked by makespan, not pool width: a count the pool cannot
    // divide evenly costs a whole extra wave (21 tasks on 20 workers run in two).
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
        const int64_t g = tid / chunks;
        const int64_t st = g % strips;
        const int64_t hg = (g / strips) % hgroups;
        const int64_t od = (g / (strips * hgroups)) % OD;
        const int64_t n = g / (strips * hgroups * OD);

        // Per-task copy: `p` is shared by every worker, so od/oh cannot live in it.
        k::DwConvParams pc = p;
        pc.od = od;
        pc.oh = hg * 4;
        pc.ow0 = st * strip;
        pc.ow_end = std::min(pc.ow0 + strip, OW);

        const T* in_img = in_ptr + n * ID * p.in_d_stride;
        // The kernel indexes the store by the absolute column, so the base stays
        // at the row start even though the loop covers only this strip.
        T* out_rows =
            out_ptr + n * OD * out_d_stride + od * out_d_stride + pc.oh * p.out_row_stride;

        const int64_t rows_here = std::min<int64_t>(4, OH - pc.oh);
        const int64_t blk0 = cc * blocks_per;
        const int64_t blk1 = std::min(blk0 + blocks_per, nblocks);
        for (int64_t blk = blk0; blk < blk1; ++blk) {
            const int64_t c0 = blk * chunk;
            // One call per block at its own width: full blocks are `chunk` wide,
            // only the last is short. Passing `C - c0` instead silently sends
            // every block but the last down the generic two-vector path. A 9..15
            // tail runs as an 8-lane call and its remainder, because two 8-lane
            // vectors would read past the last channel.
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
                const T* w_blk = w_ptr + co;
                const T* b_blk = b_ptr ? b_ptr + co : nullptr;
                // The short last group runs as one call at its own height
                // rather than one call per row: on a 7-row plane the tail is 3
                // of the 7 rows, and at R = 1 a row is a quarter of the ILP and
                // shares no input row with its neighbour.
                if (rows_here == 4) {
                    k::dwconv_channel_last<4>(out_rows + co, in_blk, w_blk, b_blk, pc,
                                              attrs.epilogue, add_to, pre_act, seg_lanes[s]);
                } else if (rows_here == 3) {
                    k::dwconv_channel_last<3>(out_rows + co, in_blk, w_blk, b_blk, pc,
                                              attrs.epilogue, add_to, pre_act, seg_lanes[s]);
                } else if (rows_here == 2) {
                    k::dwconv_channel_last<2>(out_rows + co, in_blk, w_blk, b_blk, pc,
                                              attrs.epilogue, add_to, pre_act, seg_lanes[s]);
                } else {
                    k::dwconv_channel_last<1>(out_rows + co, in_blk, w_blk, b_blk, pc,
                                              attrs.epilogue, add_to, pre_act, seg_lanes[s]);
                }
            }
        }
    });
}

// ============================================================
// Channel-last depthwise, quantized (s8/u8 activation, s8/u8 weight)
// ============================================================
//
// Same task grid as the fp route; the HWCM weight is dequantized per row per
// block into a staging vector (see dwconv_channel_last_quant). The activation is
// PerTensor and the weight PerChannel.

template <typename InT, typename OutT>
Status dwconv_channel_last_quant_impl(const DepthwiseConvAttributes& attrs, TensorView& output,
                                      std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;
    NNOPS_ASSERT(!has_bias || inputs[2].data_type() == DataType::f32);
    // At most one of the two: the pre-activation order folds the destination
    // into the accumulator, so nothing is left for the post-epilogue add.
    const bool pre_act = attrs.add_to && attrs.residual_pre_activation;
    const bool add_to = attrs.add_to && !pre_act;

    const int64_t rank = input.rank();
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

    if (attrs.kernel_size[0] * attrs.kernel_size[1] * attrs.kernel_size[2] > k::kMaxLastQuantTaps) {
        return Status::unsupported; // the staging buffer is sized for 9x9x9
    }

    const auto& in_qp = input.quant_params();
    const auto& out_qp = output.quant_params();
    const auto& w_qp = weight.quant_params();
    if (w_qp.granularity != QuantGranularity::PerChannel || w_qp.scale_data == nullptr) {
        return Status::invalid_argument;
    }
    const float in_scale = in_qp.scale;
    const float in_zero = static_cast<float>(in_qp.zero_point);
    const float out_scale = out_qp.scale;
    const float out_zero = static_cast<float>(out_qp.zero_point);

    k::DwConvParams p{};
    p.ID = ID;
    p.IH = IH;
    p.IW = IW;
    p.OH = OH;
    p.OW = OW;
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
    p.in_row_stride = input.row_stride_elems();
    p.in_d_stride = IH * p.in_row_stride;
    p.pix_stride = C;
    p.w_tap_stride = C; // the raw HWCM filter is [tap][C]

    auto* out_ptr = output.ptr<OutT>();
    const auto* in_ptr = input.ptr<InT>();
    const auto* w_ptr = weight.ptr<InT>();
    const auto* b_ptr = has_bias ? inputs[2].ptr<float>() : nullptr;

    const int64_t out_d_stride = OH * p.out_row_stride;

    const int64_t hgroups = (OH + 3) / 4;
    const int64_t in_rows = 4 * p.SH + (p.KH - 1) * p.DH; // rows a group reads
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
    const int64_t strips = (OW + strip - 1) / strip;
    const int64_t groups = N * OD * hgroups * strips;
    const int64_t nt = ctx.cpu.thread_count();
    int64_t cchunks = 1;
    if (groups < nt && nblocks > 1) {
        cchunks = std::min<int64_t>((nt + groups / 2) / groups, nblocks);
    }
    const int64_t blocks_per = (nblocks + cchunks - 1) / cchunks;

    ctx.cpu.run(0, groups * cchunks, [&](int64_t tid) {
        const int64_t cc = tid % cchunks;
        const int64_t g = tid / cchunks;
        const int64_t st = g % strips;
        const int64_t hg = (g / strips) % hgroups;
        const int64_t od = (g / (strips * hgroups)) % OD;
        const int64_t n = g / (strips * hgroups * OD);

        k::DwConvParams pc = p;
        pc.od = od;
        pc.oh = hg * 4;
        pc.ow0 = st * strip;
        pc.ow_end = std::min(pc.ow0 + strip, OW);

        const InT* in_img = in_ptr + n * ID * p.in_d_stride;
        OutT* out_rows =
            out_ptr + n * OD * out_d_stride + od * out_d_stride + pc.oh * p.out_row_stride;

        const int64_t rows_here = std::min<int64_t>(4, OH - pc.oh);
        const int64_t blk0 = cc * blocks_per;
        const int64_t blk1 = std::min(blk0 + blocks_per, nblocks);
        for (int64_t blk = blk0; blk < blk1; ++blk) {
            const int64_t c0 = blk * 8;
            const int64_t lanes = std::min<int64_t>(8, C - c0);
            const InT* in_blk = in_img + c0;
            const InT* w_blk = w_ptr + c0;
            const float* b_blk = b_ptr ? b_ptr + c0 : nullptr;
            const float* ws = w_qp.scale_data + c0;
            const int32_t* wz = w_qp.zero_point_data ? w_qp.zero_point_data + c0 : nullptr;
            // One call at the short group's own height, as in the f32 driver.
            if (rows_here == 4) {
                k::dwconv_channel_last_quant<4, InT, OutT>(
                    out_rows + c0, in_blk, w_blk, ws, wz, b_blk, pc, attrs.epilogue, add_to,
                    pre_act, lanes, in_scale, in_zero, out_scale, out_zero);
            } else if (rows_here == 3) {
                k::dwconv_channel_last_quant<3, InT, OutT>(
                    out_rows + c0, in_blk, w_blk, ws, wz, b_blk, pc, attrs.epilogue, add_to,
                    pre_act, lanes, in_scale, in_zero, out_scale, out_zero);
            } else if (rows_here == 2) {
                k::dwconv_channel_last_quant<2, InT, OutT>(
                    out_rows + c0, in_blk, w_blk, ws, wz, b_blk, pc, attrs.epilogue, add_to,
                    pre_act, lanes, in_scale, in_zero, out_scale, out_zero);
            } else {
                k::dwconv_channel_last_quant<1, InT, OutT>(
                    out_rows + c0, in_blk, w_blk, ws, wz, b_blk, pc, attrs.epilogue, add_to,
                    pre_act, lanes, in_scale, in_zero, out_scale, out_zero);
            }
        }
    });
    return Status::ok;
}

// ============================================================
// Entry point with dtype dispatch
// ============================================================

Status depthwise_conv_cpu(const DepthwiseConvAttributes& attrs, TensorView& output,
                          std::span<const TensorView> inputs, const ComputeContext& ctx,
                          void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    const auto odtype = output.data_type();

    if (!is_channel_last(inputs[0].layout())) {
        return Status::unsupported;
    }

    switch (dtype) {
    case DataType::f32:
        dwconv_channel_last_impl<float>(attrs, output, inputs, ctx);
        return Status::ok;
    case DataType::f16:
        dwconv_channel_last_impl<half>(attrs, output, inputs, ctx);
        return Status::ok;
#ifdef NNOPS_ARCH_AARCH64
    case DataType::bf16:
        dwconv_channel_last_impl<bfloat16>(attrs, output, inputs, ctx);
        return Status::ok;
#endif
    case DataType::s8:
        if (odtype == DataType::s8) {
            return dwconv_channel_last_quant_impl<int8_t, int8_t>(attrs, output, inputs, ctx);
        }
        if (odtype == DataType::u8) {
            return dwconv_channel_last_quant_impl<int8_t, uint8_t>(attrs, output, inputs, ctx);
        }
        return Status::unsupported;
    case DataType::u8:
        if (odtype == DataType::s8) {
            return dwconv_channel_last_quant_impl<uint8_t, int8_t>(attrs, output, inputs, ctx);
        }
        if (odtype == DataType::u8) {
            return dwconv_channel_last_quant_impl<uint8_t, uint8_t>(attrs, output, inputs, ctx);
        }
        return Status::unsupported;
    default:
        return Status::unsupported;
    }
}

} // namespace nnops::backend::cpu
