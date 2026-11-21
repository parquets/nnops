/// @file depthwise_conv.cpp
/// @brief SIMD CPU depthwise 2D/3D convolution on NHWC/NDHWC.
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
///     f32/f16/bf16       -> dwconv_fp_impl       (lane on C)
///     s8/u8              -> depthwise_conv_i8.cpp (dwconv_quant_run)

#include "depthwise_conv_internal.h"
#include "conv_util.h"
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
// Spatial depthwise — the lane spans channels
// ============================================================

template <typename T>
void dwconv_fp_impl(const DepthwiseConvAttributes& attrs, TensorView& output,
                              std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;
    // At most one of the two: the pre-activation order folds the destination
    // into the accumulator, so nothing is left for the post-epilogue add.
    const bool pre_act = attrs.add_to && attrs.residual_pre_activation;
    const bool add_to = attrs.add_to && !pre_act;

    const int64_t srank = DepthwiseConvAttributes::spatial_rank(input.rank());
    const int64_t N = input.shape(0);
    const int64_t C = input.shape(input.rank() - 1);
    const int64_t OD = (srank == 3) ? output.shape(1) : 1;
    const int64_t OH = output.shape(srank - 1);
    const int64_t OW = output.shape(srank);

    const k::DwConvParams p = dwconv_params(attrs, input, output);

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
    const int64_t strip = conv_strip_cols(in_rows, C, static_cast<int64_t>(sizeof(T)), OW,
                                          p.KW, p.DW, p.SW);
    const int64_t strips = (OW + strip - 1) / strip;
    const int64_t groups = N * OD * hgroups * strips;
    // The filter axis splits a group only when the grid cannot fill the pool;
    // one chunk keeps the block loop inside the task and the strip in cache.
    const int64_t chunks = conv_channel_chunks(groups, nblocks, ctx.cpu.thread_count());
    const int64_t blocks_per = (nblocks + chunks - 1) / chunks;

    ctx.cpu.run(0, groups * chunks, [&](int64_t tid) {
        const ConvGridPos gp = conv_grid_pos(tid, strips, hgroups, OD, chunks);

        // Per-task copy: `p` is shared by every worker, so od/oh cannot live in it.
        k::DwConvParams pc = p;
        pc.od = gp.od;
        pc.oh = gp.hg * 4;
        pc.ow0 = gp.st * strip;
        pc.ow_end = std::min(pc.ow0 + strip, OW);

        const T* in_img = in_ptr + gp.n * p.ID * p.in_d_stride;
        // The kernel indexes the store by the absolute column, so the base stays
        // at the row start even though the loop covers only this strip.
        T* out_rows = out_ptr + gp.n * OD * out_d_stride + gp.od * out_d_stride +
                      pc.oh * p.out_row_stride;

        const int64_t rows_here = std::min<int64_t>(4, OH - pc.oh);
        const int64_t blk0 = gp.cc * blocks_per;
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
                    k::dwconv_block<4>(out_rows + co, in_blk, w_blk, b_blk, pc,
                                              attrs.epilogue, add_to, pre_act, seg_lanes[s]);
                } else if (rows_here == 3) {
                    k::dwconv_block<3>(out_rows + co, in_blk, w_blk, b_blk, pc,
                                              attrs.epilogue, add_to, pre_act, seg_lanes[s]);
                } else if (rows_here == 2) {
                    k::dwconv_block<2>(out_rows + co, in_blk, w_blk, b_blk, pc,
                                              attrs.epilogue, add_to, pre_act, seg_lanes[s]);
                } else {
                    k::dwconv_block<1>(out_rows + co, in_blk, w_blk, b_blk, pc,
                                              attrs.epilogue, add_to, pre_act, seg_lanes[s]);
                }
            }
        }
    });
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
        dwconv_fp_impl<float>(attrs, output, inputs, ctx);
        return Status::ok;
    case DataType::f16:
        dwconv_fp_impl<half>(attrs, output, inputs, ctx);
        return Status::ok;
#ifdef NNOPS_ARCH_AARCH64
    case DataType::bf16:
        dwconv_fp_impl<bfloat16>(attrs, output, inputs, ctx);
        return Status::ok;
#endif
    case DataType::s8:
    case DataType::u8:
        if (odtype != DataType::s8 && odtype != DataType::u8) {
            return Status::unsupported;
        }
        return dwconv_quant_run(attrs, output, inputs, ctx);
    default:
        return Status::unsupported;
    }
}

} // namespace nnops::backend::cpu
