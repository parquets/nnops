/// @file conv3d_implicit_gemm.cpp
/// @brief CPU Conv3D (NDHWC, f32/f16) — im2col → GEMM, with the
///        scalar gather as the fallback. The quantized route is
///        conv3d_implicit_gemm_i8.cpp; what the two share is in
///        conv3d_implicit_gemm_internal.h.
///
/// Forward convolution is the easy direction for a col matrix: the gathered
/// activation is the GEMM's A and the filter, read per group as
/// [tap·ic + ic][oc], is B,
///
///   A = the activation, [pixels][IC/G·KD·KH·KW], tap-major,
///   B = the filter, [IC/G·KD·KH·KW][OC/G],
///   C = [pixels × OC/G],
///
/// and what is left is the epilogue pass that writes the tile out. A tap of a
/// NDHWC activation is a contiguous IC/G-channel run, so the gather is
/// one vector copy per tap; the GEMM's C rows are pixels and its columns are
/// output channels, which is exactly the destination's own order.
///
/// The gather is tiled_im2col_3d (simd_kernel/simd_im2col.hpp),
/// one call per block, zero-filling the six padded faces. A block is a
/// rectangle of output — whole depth slices while they fit, otherwise a run of
/// rows inside one slice — because the col's row stride is the block's own
/// volume, so the rectangle has to be the unit. The pixel axis is what gets
/// blocked, never the oc axis. A 1x1x1 stride-1 convolution skips the gather
/// entirely and reads the input plane as A.
///
/// The GEMM writes straight into the destination whenever its rows are back to
/// back and no residual has to survive it; otherwise it writes a pooled staging
/// tile and the epilogue pass copies it out. The pass is skipped outright when
/// the direct tile needs nothing — no bias, no live epilogue, no add_to.
///
/// Two parallel arms, picked per call. The pool usually belongs *inside* one
/// GEMM: a single (batch, group) with enough pixels fills it, and the GEMM then
/// splits on its own larger dimension. A plane with many (batch, group) pairs
/// and little per-group work cannot fill it that way, so those calls run one
/// task per (batch, group, pixel block) with a serial GEMM each. The switch is
/// the GEMM's own block grid against the worker count
/// (`NNOPS_CONV3D_GRID=0/1` pins it for A/B).
///
/// Anything the flat operand views cannot describe falls back to the plain
/// output-driven gather, which is correct for every shape.
///
/// The filter is DHWIO, [KD, KH, KW, IC/G, OC] — TensorFlow's conv3d order.
/// That is the GEMM's B already: one tap is one contiguous OC run per input
/// channel, and two taps are one IC/G·OC step apart, so the raw filter is read
/// in place (a group is a pitched column slice) and the prepacked blob is only
/// the pack MatMul would otherwise repeat per call. Both the input and the
/// output are channel-last NDHWC ([N, D, H, W, C] as both the logical dims and
/// the memory order).

#include "backend/cpu/conv/conv3d_implicit_gemm_internal.h"
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"
#include "backend/cpu/simd_kernel/simd_im2col.hpp"
#include "backend/cpu/common/dtype_dispatch.hpp"
#include "backend/cpu/common/memory_pool.hpp"
#include "backend/cpu/matmul/matmul.h"
#include "backend/cpu/conv/depthwise_conv_internal.h" // DepthwiseConvAttributes mapping
#include "backend/cpu/matmul/matmul_helper.h"
#include "nnops/core/parallel_for.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <span>

namespace nnops::backend::cpu {

/// The depthwise family's CPU entry (depthwise_conv.cpp), shared with Conv1D's
/// depthwise route; declared here rather than in a header the way conv1d.cpp
/// does, since only this file's depthwise branch calls it.
Status depthwise_conv_cpu(const DepthwiseConvAttributes&, TensorView&,
                          std::span<const TensorView>, const ComputeContext&, void*);

using namespace nnops::simd;

namespace {

/// im2col → GEMM. Status::unsupported (or out_of_memory) when the operands
/// cannot be read as flat matrices, leaving the caller its gather; ok, or the
/// GEMM's own error, otherwise.
template <typename T>
Status conv3d_im2col_gemm(const ConvChannels& gm, TensorView& output,
                          std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t kdim = gm.IC_per_G * gm.karea();
    const int64_t npix = gm.npix();
    if (kdim <= 0 || npix <= 0 || gm.OC_per_G <= 0) {
        return Status::unsupported;
    }
    // The filter must read as one dense [IC/G·KD·KH·KW][OC/G] matrix per group.
    if (!conv3d_filter_is_matrix(weight, gm)) {
        return Status::unsupported;
    }

    const int64_t in_n_stride = input.stride_elems(0);
    const int64_t in_d_stride = input.stride_elems(1);
    const int64_t in_row_stride = input.row_stride_elems();
    const int64_t in_pix = input.shape(4); // W step: one whole channel vector
    const int64_t out_n_stride = output.stride_elems(0);
    const int64_t out_d_stride = output.stride_elems(1);
    const int64_t out_row_stride = output.row_stride_elems();

    const bool in_dense =
        (in_row_stride == gm.IW * in_pix) && (in_d_stride == gm.IH * in_row_stride);

    // A 1x1x1 stride-1 convolution is a plain GEMM over the input plane: one
    // pixel step is one channel vector on both sides, so nothing is gathered.
    const bool pointwise = gm.KD == 1 && gm.KH == 1 && gm.KW == 1 && gm.SD == 1 && gm.SH == 1 &&
                           gm.SW == 1 && gm.PD == 0 && gm.PH == 0 && gm.PW == 0 && in_dense;

    // The GEMM writes into the destination unless a residual has to survive it
    // or the destination's rows are not back to back.
    const bool dense_row = (out_row_stride == gm.OW * gm.OC);
    const bool direct = dense_row && !gm.add_to;
    const bool write_pass = !direct || has_bias || epilogue_is_active(gm.epilogue);

    const auto* in_ptr = input.ptr<T>();
    const auto* b_ptr = has_bias ? inputs[2].ptr<T>() : nullptr;
    auto* out_ptr = output.ptr<T>();
    const DataType vdt = input.data_type();
    const MatMulAttributes mm{};
    std::atomic<int> fail{0};

    // One pixel block: gather A, GEMM into C, scatter the rows C cannot hold.
    // @p ictx is the context the GEMM and the scatter run on — the real one in
    // the serial arm (the pool lives inside them), an empty one in the grid arm
    // (the pool lives over the tasks). @p wg is the group's B, shared read-only.
    const auto run_block = [&](int64_t n, int64_t g, int64_t od0, int64_t oh0, int64_t od_chunk,
                               int64_t roi_h, T* col, T* stage, const TensorView& wg,
                               const ComputeContext& ictx) {
        const int64_t nd = std::min(od_chunk, gm.OD - od0);
        const int64_t nh = std::min(roi_h, gm.OH - oh0);
        const int64_t nblk = nd * nh * gm.OW;
        const int64_t p0 = (od0 * gm.OH + oh0) * gm.OW;

        const T* in_g = in_ptr + n * in_n_stride + g * gm.IC_per_G;
        T* out_g = out_ptr + n * out_n_stride + g * gm.OC_per_G;
        const T* b_g = b_ptr ? b_ptr + g * gm.OC_per_G : nullptr;

        const T* a_src = pointwise ? in_g + p0 * in_pix : col;
        if (!pointwise) {
            // One (od, oh) row at a time so the serial arm can put the gather on
            // the pool too: it is bandwidth-bound and, on a narrow channel axis,
            // scalar, so a single thread can be most of a small layer.
            const int64_t rows = nd * nh;
            const auto gather_row = [&](int64_t r) {
                nnops::kernel::tiled_im2col_3d<T>(
                    col + r * gm.OW * kdim, static_cast<int>(kdim), in_g, in_pix, in_d_stride,
                    in_row_stride, static_cast<int>(gm.IC_per_G),
                    static_cast<int>(od0 + r / nh), static_cast<int>(oh0 + r % nh), /*rd=*/1,
                    /*rh=*/1, static_cast<int>(gm.OW), static_cast<int>(gm.ID),
                    static_cast<int>(gm.IH), static_cast<int>(gm.IW), static_cast<int>(gm.KD),
                    static_cast<int>(gm.KH), static_cast<int>(gm.KW), static_cast<int>(gm.SD),
                    static_cast<int>(gm.SH), static_cast<int>(gm.SW), static_cast<int>(gm.DD),
                    static_cast<int>(gm.DH), static_cast<int>(gm.DW), static_cast<int>(gm.PD),
                    static_cast<int>(gm.PH), static_cast<int>(gm.PW));
            };
            ictx.cpu.run(0, rows, gather_row);
        }

        const int64_t ashape[] = {nblk, kdim};
        TensorView am(std::span<const int64_t>(ashape, 2), vdt, const_cast<T*>(a_src),
                      (pointwise ? in_pix : kdim) * static_cast<int64_t>(sizeof(T)),
                      TensorLayout::Dense);

        T* ctile = direct ? out_g + p0 * gm.OC : stage;
        const int64_t cshape[] = {nblk, gm.OC_per_G};
        TensorView cm(std::span<const int64_t>(cshape, 2), vdt, ctile,
                      (direct ? gm.OC : gm.OC_per_G) * static_cast<int64_t>(sizeof(T)),
                      TensorLayout::Dense);

        const TensorView mins[] = {am, wg};
        const Status st = matmul_cpu(mm, cm, mins, ictx, nullptr);
        if (!is_ok(st)) {
            fail.store(static_cast<int>(st), std::memory_order_relaxed);
            return;
        }
        if (direct && !write_pass) {
            return;
        }

        const T* src_base = direct ? ctile : stage;
        const int64_t src_pitch = direct ? gm.OC : gm.OC_per_G;
        const int64_t oc_base = g * gm.OC_per_G;
        ictx.cpu.run(0, nblk, [&](int64_t p) {
            const int64_t od = p / (nh * gm.OW);
            const int64_t rem = p % (nh * gm.OW);
            const int64_t oh = rem / gm.OW;
            const int64_t ow = rem % gm.OW;
            T* dst = out_g + (od0 + od) * out_d_stride + (oh0 + oh) * out_row_stride + ow * gm.OC;
            const T* src = src_base + p * src_pitch;
            apply_epilogue_row<T>(dst, src, gm.OC_per_G, b_g, /*bias_step=*/1, gm.epilogue,
                                  oc_base, /*quant_step=*/1, gm.add_to, /*pre_act=*/false);
        });
    };

    const int64_t workers = std::max<int64_t>(ctx.cpu.worker_count(), 1);
    const int64_t ng = gm.N * gm.G;
    const int grant = conv3d_grid_override();
    // The GEMM's own block grid, roughly: a per-group GEMM narrower than the
    // pool cannot fill it (a depthwise group is one output channel), so those
    // calls run the task grid instead. A wide GEMM keeps the pool inside it —
    // its gather is parallel there too, and one big GEMM beats many serial ones.
    const int64_t gemm_blocks =
        ((npix + 47) / 48) * std::max<int64_t>(1, (gm.OC_per_G + 31) / 32);
    const bool task_grid = (grant == 1) || (grant != 0 && gemm_blocks < workers);

    if (!task_grid) {
        const Conv3DBlock blk =
            conv3d_choose_block(gm, kdim, static_cast<int64_t>(sizeof(T)), kConv3dColCapBytes);
        const int64_t nblk_max = blk.od_chunk * blk.roi_h * gm.OW;
        PoolArray<T> col(pointwise ? 0 : static_cast<size_t>(nblk_max * kdim));
        PoolArray<T> stage(direct ? 0 : static_cast<size_t>(nblk_max * gm.OC_per_G));
        if ((!pointwise && !col) || (!direct && !stage)) {
            return Status::out_of_memory;
        }
        for (int64_t n = 0; n < gm.N; ++n) {
            for (int64_t g = 0; g < gm.G; ++g) {
                const TensorView wg = conv3d_weight_group(weight, gm, g);
                for (int64_t od0 = 0; od0 < gm.OD; od0 += blk.od_chunk) {
                    for (int64_t oh0 = 0; oh0 < gm.OH; oh0 += blk.roi_h) {
                        run_block(n, g, od0, oh0, blk.od_chunk, blk.roi_h, col.data(),
                                  stage.data(), wg, ctx);
                    }
                }
            }
        }
        return is_ok(Status(fail.load())) ? Status::ok : Status(fail.load());
    }

    // One task per (batch, group, pixel block): a plane with many groups but few
    // pixels per group still fills the pool. Each task's GEMM is serial, so the
    // per-task column budget is a slice of the serial arm's cap.
    const int64_t budget = std::max<int64_t>(kConv3dColCapBytes / workers, kConv3dMinTaskColBytes);
    const Conv3DBlock blk =
        conv3d_choose_block(gm, kdim, static_cast<int64_t>(sizeof(T)), budget);
    const int64_t blocks_d = (gm.OD + blk.od_chunk - 1) / blk.od_chunk;
    const int64_t blocks_h = (gm.OH + blk.roi_h - 1) / blk.roi_h;
    const int64_t blocks = blocks_d * blocks_h;
    const int64_t tasks = ng * blocks;
    const int64_t slots = std::min<int64_t>(tasks, workers);
    const int64_t nblk_max = blk.od_chunk * blk.roi_h * gm.OW;

    PoolArray<T> col(pointwise ? 0 : static_cast<size_t>(nblk_max * kdim * slots));
    PoolArray<T> stage(direct ? 0 : static_cast<size_t>(nblk_max * gm.OC_per_G * slots));
    if ((!pointwise && !col) || (!direct && !stage)) {
        return Status::out_of_memory;
    }

    // The group's B is read-only, so every task shares it; the raw filter is
    // addressed in place and a prepacked blob is fixed at load time.
    const ComputeContext serial_ctx{};
    ctx.cpu.run(0, tasks, [&](int64_t t) {
        const int64_t slot = (slots < tasks) ? ctx.cpu.current_thread_id() : t;
        const int64_t bi = t % blocks;
        const int64_t gi = t / blocks;
        const int64_t n = gi / gm.G;
        const int64_t g = gi % gm.G;
        const int64_t od0 = (bi / blocks_h) * blk.od_chunk;
        const int64_t oh0 = (bi % blocks_h) * blk.roi_h;

        run_block(n, g, od0, oh0, blk.od_chunk, blk.roi_h,
                  pointwise ? nullptr : col.data() + slot * nblk_max * kdim,
                  direct ? nullptr : stage.data() + slot * nblk_max * gm.OC_per_G,
                  conv3d_weight_group(weight, gm, g), serial_ctx);
    });
    return is_ok(Status(fail.load())) ? Status::ok : Status(fail.load());
}

/// The output-driven gather: every output voxel walks its own taps. Correct for
/// every geometry, and the route a shape too odd for the col takes.
template <typename T>
void conv3d_gather(const ConvChannels& gm, TensorView& output, std::span<const TensorView> inputs,
                   const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t in_n_stride = input.stride_elems(0);
    const int64_t in_d_stride = input.stride_elems(1);
    const int64_t in_row_stride = input.row_stride_elems();
    const int64_t in_pix = input.shape(4);
    const int64_t out_n_stride = output.stride_elems(0);
    const int64_t out_d_stride = output.stride_elems(1);
    const int64_t out_row_stride = output.row_stride_elems();

    // DHWIO [KD][KH][KW][IC/G][OC]: the filter is walked by its own strides
    // rather than as one k block, so a pitched one is still readable.
    const int64_t w_oc_stride = weight.stride_elems(4);
    const int64_t w_ic_stride = weight.stride_elems(3);
    const int64_t w_kw_stride = weight.stride_elems(2);
    const int64_t w_kh_stride = weight.stride_elems(1);
    const int64_t w_kd_stride = weight.stride_elems(0);

    const auto* in_ptr = input.ptr<T>();
    const auto* w_ptr = weight.ptr<T>();
    const auto* b_ptr = has_bias ? inputs[2].ptr<T>() : nullptr;
    auto* out_ptr = output.ptr<T>();

    ctx.cpu.run(0, gm.N, [&](int64_t n) {
        for (int64_t g = 0; g < gm.G; ++g) {
            for (int64_t oc_l = 0; oc_l < gm.OC_per_G; ++oc_l) {
                const int64_t oc = g * gm.OC_per_G + oc_l;
                const T* w_oc = w_ptr + oc * w_oc_stride;
                const float bv = b_ptr ? s_load(&b_ptr[oc]) : 0.0f;
                for (int64_t od = 0; od < gm.OD; ++od) {
                    for (int64_t oh = 0; oh < gm.OH; ++oh) {
                        T* orow = out_ptr + n * out_n_stride + od * out_d_stride +
                                  oh * out_row_stride + oc;
                        for (int64_t ow = 0; ow < gm.OW; ++ow) {
                            float sum = 0.0f;
                            for (int64_t ic = 0; ic < gm.IC_per_G; ++ic) {
                                const int64_t ic_global = g * gm.IC_per_G + ic;
                                const T* w_ic = w_oc + ic * w_ic_stride;
                                for (int64_t kd = 0; kd < gm.KD; ++kd) {
                                    const int64_t id = od * gm.SD + kd * gm.DD - gm.PD;
                                    if (id < 0 || id >= gm.ID) {
                                        continue;
                                    }
                                    for (int64_t kh = 0; kh < gm.KH; ++kh) {
                                        const int64_t ih = oh * gm.SH + kh * gm.DH - gm.PH;
                                        if (ih < 0 || ih >= gm.IH) {
                                            continue;
                                        }
                                        const T* in_row = in_ptr + n * in_n_stride +
                                                          id * in_d_stride + ih * in_row_stride +
                                                          ic_global;
                                        const T* w_row = w_ic + kd * w_kd_stride + kh * w_kh_stride;
                                        for (int64_t kw = 0; kw < gm.KW; ++kw) {
                                            const int64_t iw = ow * gm.SW + kw * gm.DW - gm.PW;
                                            if (iw < 0 || iw >= gm.IW) {
                                                continue;
                                            }
                                            sum += s_load(&in_row[iw * in_pix]) *
                                                   s_load(&w_row[kw * w_kw_stride]);
                                        }
                                    }
                                }
                            }
                            apply_epilogue_store<T>(&orow[ow * gm.OC], sum + bv, gm.epilogue, oc,
                                                    gm.add_to, /*pre_act=*/false);
                        }
                    }
                }
            }
        }
    });
}

/// True when a blob was handed in place of the raw filter — the filter is then
/// addressable only through the blob, and there is nothing to gather.
inline bool prepacked_filter(std::span<const TensorView> inputs) noexcept
{
    return inputs.size() > 1 && is_packed_conv3d_weight(inputs[1].layout());
}

template <typename T>
Status conv3d_impl(const ConvChannels& gm, TensorView& output, std::span<const TensorView> inputs,
                   const ComputeContext& ctx)
{
    if (gm.N == 0 || gm.OC == 0 || gm.OD == 0 || gm.OH == 0 || gm.OW == 0) {
        return Status::ok;
    }
    const Status st = conv3d_im2col_gemm<T>(gm, output, inputs, ctx);
    if (is_ok(st)) {
        return Status::ok;
    }
    // A prepacked filter is a B matrix and nothing else, so there is no raw
    // matrix left for the gather; anything else gets the scalar route, which is
    // correct for every geometry (an unreadable pitch and an allocation
    // failure alike).
    if (prepacked_filter(inputs)) {
        return st;
    }
    conv3d_gather<T>(gm, output, inputs, ctx);
    return Status::ok;
}

} // anonymous namespace

// =========================================================================
//  The prepacked filter (Conv3D::prepackWeights)
// =========================================================================
//
// DHWIO already stores each group's filter in the GEMM's k-major order, so the
// blob is not a transpose — it is the pack MatMul would otherwise rebuild on
// every call: the whole-K B panels for an fp filter, the group's compact
// matrix for an s8 one, and on an SME machine the ZA kernels' whole-K block
// layout instead of either. MatMul reads a ZA blob in place and its tile plan
// takes both GEMMs to the engine; without one the same GEMM packs B per call
// and, on a MatMul blob, the ZA route is out of reach. The geometry is a
// function of the attributes and the filter's own dims, never the activation,
// so prepackWeights can answer the query without one.

Conv3DPackedWeight conv3d_packed_weight_geom(int64_t karea, int64_t ic_per_group,
                                             int64_t oc_per_group, int64_t groups,
                                             DataType weight_dtype, bool sme) noexcept
{
    Conv3DPackedWeight g;
    if (karea <= 0 || ic_per_group <= 0 || oc_per_group <= 0 || groups <= 0) {
        return g;
    }
    const int64_t kdim = ic_per_group * karea;
    if (sme) {
        // Same 64-column whole-K block MatMul's own blob uses; empty for a
        // dtype the engine has no kernel for.
        const MatMulPackedB mb = matmul_packed_b_sme_geom(oc_per_group, kdim, weight_dtype);
        if (!mb.ok()) {
            return g;
        }
        g.group_elems = ((mb.total + oc_per_group - 1) / oc_per_group) * oc_per_group;
    } else if (weight_dtype == DataType::s8) {
        g.group_elems = kdim * oc_per_group;
    } else {
        const MatMulPackedB mb = matmul_packed_b_geom(oc_per_group, kdim, weight_dtype);
        if (!mb.ok()) {
            return g;
        }
        // Each group's sub-blob is a whole numel on its own, so the next group
        // starts at a descriptor boundary.
        g.group_elems = ((mb.total + oc_per_group - 1) / oc_per_group) * oc_per_group;
    }
    g.groups = groups;
    g.total = g.group_elems * groups;
    return g;
}

bool conv3d_packed_blob_matches(const TensorView& wgt, const ConvChannels& gm)
{
    const Conv3DPackedWeight pg =
        conv3d_packed_weight_geom(gm.karea(), gm.IC_per_G, gm.OC_per_G, gm.G, wgt.data_type(),
                                  is_sme_conv3d_weight(wgt.layout()));
    return pg.total > 0 && wgt.numel() == pg.total;
}

TensorDesc conv3d_packed_weight_desc(const Conv3DAttributes& attrs, const TensorDesc& weight)
{
    // DHWIO is [KD][KH][KW][IC/G][OC]: the channel counts are the last two.
    if (weight.rank != 5 || !is_channel_last_weight_3d(weight.layout)) {
        return TensorDesc{};
    }
    const int64_t OC = weight.dims[4];
    const int64_t ICG = weight.dims[3];
    const int64_t G = attrs.groups;
    if (G <= 0 || OC % G != 0 || ICG <= 0) {
        return TensorDesc{};
    }
    // A depthwise filter (one channel per group, one output per group) has no
    // GEMM blob worth having: the grouped route would run one 1-wide GEMM per
    // channel, and compute() reads the raw filter far faster through the
    // channel-lane depthwise kernel. Declining here leaves the caller the raw
    // filter, which is the route it wants.
    if (ICG == 1 && OC == G) {
        return TensorDesc{};
    }
    const int64_t karea = attrs.kernel_size[0] * attrs.kernel_size[1] * attrs.kernel_size[2];
    // The ZA layout when this machine has the engine and a kernel for the
    // dtype: decided once, here, and carried by the tag — compute() never asks
    // the CPU again, so a blob is read exactly as it was written. A K shorter
    // than one 64-value run stays on the panels: a depthwise group's GEMM is
    // then smaller than the pack that feeds it, and the ZA tiles lose to the
    // narrow kernels on measured shapes.
    const bool sme = ICG * karea >= 64 &&
                     matmul_packed_b_sme_geom(OC / G, ICG * karea, weight.dtype).ok();
    const Conv3DPackedWeight g =
        conv3d_packed_weight_geom(karea, ICG, OC / G, G, weight.dtype, sme);
    if (g.total <= 0) {
        return TensorDesc{};
    }

    TensorDesc d;
    d.rank = 2;
    d.dtype = weight.dtype;
    d.layout = sme ? TensorLayout::PackedDHWIOSme : TensorLayout::PackedDHWIO;
    d.dims.resize(2);
    // OC first, the axis every consumer of a padded conv blob reads for shape
    // inference; group_elems is a multiple of OC/G, so OC divides the total.
    d.dims[0] = OC;
    d.dims[1] = g.total / OC;
    d.row_stride_elems = d.dims[1];
    return d;
}

void conv3d_packed_weight_pack(const Conv3DAttributes& attrs, const TensorView& weight,
                               TensorView& packed, const ComputeContext& ctx)
{
    const int64_t OC = weight.shape(4);
    const int64_t ICG = weight.shape(3);
    const int64_t G = attrs.groups;
    const int64_t OCG = OC / G;
    const int64_t karea = attrs.kernel_size[0] * attrs.kernel_size[1] * attrs.kernel_size[2];
    const int64_t kdim = ICG * karea;
    NNOPS_ASSERT(weight.stride_elems(4) == 1 && weight.stride_elems(3) == OC &&
                 weight.stride_elems(2) == ICG * OC &&
                 "conv3d_packed_weight_pack: the filter has no matrix reading");

    if (weight.data_type() == DataType::s8) {
        conv3d_packed_weight_pack_i8(attrs, weight, packed, ctx);
        return;
    }

    const DataType dt = weight.data_type();
    const int64_t esz = static_cast<int64_t>(data_type_size(dt));
    const bool sme = is_sme_conv3d_weight(packed.layout());
    const Conv3DPackedWeight g = conv3d_packed_weight_geom(karea, ICG, OCG, G, dt, sme);
    NNOPS_ASSERT(g.total > 0 && packed.numel() >= g.total &&
                 "conv3d_packed_weight_pack: the buffer does not match the geometry");

    // Row k = tap·IC/G + ic of a group's B is the filter's (tap, ic) row, of
    // which this group owns one OC/G run. matmul's pack splits its panels over
    // the pool, so the groups go one after another rather than nested inside a
    // group-parallel run.
    char* const blob = static_cast<char*>(packed.ptr<void>());
    const char* const src = static_cast<const char*>(weight.ptr<void>());
    for (int64_t gi = 0; gi < G; ++gi) {
        const int64_t raw_shape[] = {kdim, OCG};
        const TensorView raw(std::span<const int64_t>(raw_shape, 2), dt,
                             const_cast<char*>(src) + gi * OCG * esz, OC * esz,
                             TensorLayout::Dense);
        const int64_t dst_shape[] = {OCG, g.group_elems / OCG};
        TensorView dst(std::span<const int64_t>(dst_shape, 2), dt,
                       blob + gi * g.group_elems * esz, (g.group_elems / OCG) * esz,
                       sme ? TensorLayout::PackedBSme : TensorLayout::PackedB);
        matmul_packed_b_pack(raw, dst, /*transpose_b=*/false, ctx);
    }
}

TensorView conv3d_weight_group(const TensorView& weight, const ConvChannels& gm, int64_t g)
{
    const DataType dt = weight.data_type();
    const int64_t esz = static_cast<int64_t>(data_type_size(dt));
    const int64_t kdim = gm.IC_per_G * gm.karea();
    const int64_t shape[] = {kdim, gm.OC_per_G};
    char* const base = const_cast<char*>(static_cast<const char*>(weight.ptr<void>()));

    if (is_packed_conv3d_weight(weight.layout())) {
        const bool sme = is_sme_conv3d_weight(weight.layout());
        const Conv3DPackedWeight pg =
            conv3d_packed_weight_geom(gm.karea(), gm.IC_per_G, gm.OC_per_G, gm.G, dt, sme);
        char* const at = base + g * pg.group_elems * esz;
        if (dt == DataType::s8 && !sme) {
            return TensorView(std::span<const int64_t>(shape, 2), dt, at,
                              gm.OC_per_G * esz, TensorLayout::Dense, weight.quant_params());
        }
        const int64_t pshape[] = {gm.OC_per_G, pg.group_elems / gm.OC_per_G};
        return TensorView(std::span<const int64_t>(pshape, 2), dt, at,
                          (pg.group_elems / gm.OC_per_G) * esz,
                          sme ? TensorLayout::PackedBSme : TensorLayout::PackedB,
                          weight.quant_params());
    }
    return TensorView(std::span<const int64_t>(shape, 2), dt, base + g * gm.OC_per_G * esz,
                      weight.stride_elems(3) * esz, TensorLayout::Dense);
}

/// A depthwise filter — one input channel per group, one output channel per
/// group — is the channel-lane kernel's shape, not a GEMM: the im2col route
/// gathers C one-channel columns and runs C one-column GEMMs, which on the
/// measured shapes is the slowest case in the family. The raw DHWIO filter
/// already stores a tap as a contiguous run of C channels (the kernel's
/// w_tap_stride), so nothing has to be repacked; a prepacked blob is a GEMM
/// panel set and stays on that route.
inline bool conv3d_depthwise_shape(const TensorView& input, const TensorView& weight,
                                   TensorView& output, const ConvChannels& gm) noexcept
{
    if (is_packed_conv3d_weight(weight.layout()) || weight.rank() != 5 ||
        !is_channel_last_weight_3d(weight.layout())) {
        return false;
    }
    // Multiplier 1 only: the depthwise kernel has no output-channel axis.
    if (gm.G <= 0 || gm.G != gm.IC || gm.OC != gm.IC || gm.IC_per_G != 1) {
        return false;
    }
    // The kernel reads the filter as flat taps of C channels, so every tap
    // boundary has to be one: [KD][KH][KW][1][C] dense, not a pitched view.
    if (weight.stride_elems(4) != 1 || weight.stride_elems(3) != gm.OC ||
        weight.stride_elems(2) != gm.OC || weight.stride_elems(1) != gm.KW * gm.OC ||
        weight.stride_elems(0) != gm.KH * gm.KW * gm.OC) {
        return false;
    }
    // It derives both D strides from the row stride, so the depth axis has to
    // be dense on the way in and on the way out.
    if (input.stride_elems(1) != gm.IH * input.row_stride_elems() ||
        output.stride_elems(1) != gm.OH * output.row_stride_elems()) {
        return false;
    }
    const DataType dt = input.data_type();
    return (dt == DataType::f32 || dt == DataType::f16
#ifdef NNOPS_ARCH_AARCH64
            || dt == DataType::bf16
#endif
           ) &&
           weight.data_type() == dt;
}

/// Map the 3-D attributes onto the depthwise operator's — the spatial arrays,
/// the epilogue and the residual are the same fields; the groups live in the
/// shapes, not in the attributes.
inline DepthwiseConvAttributes conv3d_depthwise_attrs(const Conv3DAttributes& attrs) noexcept
{
    DepthwiseConvAttributes d;
    d.kernel_size = attrs.kernel_size;
    d.stride = attrs.stride;
    d.dilation = attrs.dilation;
    d.padding = attrs.padding;
    d.epilogue = attrs.epilogue;
    d.add_to = attrs.add_to;
    return d;
}

Status conv3d_cpu(const Conv3DAttributes& attrs, TensorView& output,
                  std::span<const TensorView> inputs, const ComputeContext& ctx,
                  void* /*workspace*/)
{
    const auto& input = inputs[0];
    const auto& weight = inputs[1];

    ConvChannels gm{};
    fill_conv_axes<3>(gm, attrs, input, output);
    gm.G = attrs.groups;
    if (!conv3d_fill_channels(gm, input, weight)) {
        return Status::unsupported;
    }
    // The blob must describe this call's K, or MatMul would read a panel packed
    // for another depth.
    if (is_packed_conv3d_weight(weight.layout()) && !conv3d_packed_blob_matches(weight, gm)) {
        return Status::invalid_argument;
    }

    if (weight.data_type() == DataType::s8) {
        return conv3d_i8_cpu(attrs, output, inputs, ctx);
    }
    if (conv3d_depthwise_shape(input, weight, output, gm)) {
        const DepthwiseConvAttributes dwa = conv3d_depthwise_attrs(attrs);
        return depthwise_conv_cpu(dwa, output, inputs, ctx, nullptr);
    }
    Status st = Status::unsupported;
    dispatch_f32_f16(input.data_type(), "conv3d_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        st = conv3d_impl<T>(gm, output, inputs, ctx);
    });
    return st;
}

} // namespace nnops::backend::cpu
