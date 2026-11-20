/// @file conv3d_implicit_gemm.cpp
/// @brief CPU Conv3D (NDHWC, f32/f16) — im2col → GEMM, with the
///        scalar gather as the fallback. The quantized route is
///        conv3d_implicit_gemm_i8.cpp; what the two share is in
///        conv3d_implicit_gemm_internal.h.
///
/// Forward convolution is the easy direction for a col matrix: the gathered
/// activation is the GEMM's A and the filter, repacked once per group into
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
/// task per (batch, group, pixel block) with a serial GEMM each — the filter is
/// then repacked once per group instead of once per (batch, group). The switch
/// is N*G against the worker count (`NNOPS_CONV3D_GRID=0/1` pins it for A/B).
///
/// Anything the flat operand views cannot describe falls back to the plain
/// output-driven gather, which is correct for every shape.
///
/// The weight stays [OC, IC/G, KD, KH, KW]; both the input and the output are
/// channel-last NDHWC ([N, D, H, W, C] as both the logical dims and the memory
/// order).

#include "backend/cpu/conv/conv3d_implicit_gemm_internal.h"
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"
#include "backend/cpu/simd_kernel/simd_im2col.hpp"
#include "backend/cpu/common/dtype_dispatch.hpp"
#include "backend/cpu/common/memory_pool.hpp"
#include "backend/cpu/matmul/matmul.h"
#include "nnops/core/parallel_for.hpp"

#include <algorithm>
#include <cstdlib>
#include <span>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

/// Largest col one GEMM call may materialise in the serial arm; past it the
/// pixel block splits. The task-grid arm gives each concurrent task a slice.
constexpr int64_t kColCapBytes = 8 << 20;
/// Cap on the shared filter repack in the task-grid arm (all groups at once).
constexpr int64_t kWeightPackCap = 8 << 20;
/// Smallest per-task column budget the task grid will use.
constexpr int64_t kMinTaskColBytes = int64_t{64} << 10;
/// Largest staged output tile; only caps the oc chunk when the GEMM cannot
/// write into the destination (a padded pitch, or add_to's live residual).
constexpr int64_t kTileCapBytes = 4 << 20;

/// `NNOPS_CONV3D_GRID=0/1` pins the dispatch for A/B: 0 the serial arm (the
/// pool runs inside each GEMM), 1 the task grid (the pool runs over
/// batch·group·blocks and each GEMM is serial). Read once.
inline int conv3d_grid_override()
{
    static const int v = [] {
        const char* s = std::getenv("NNOPS_CONV3D_GRID");
        return s != nullptr ? std::atoi(s) : -1;
    }();
    return v;
}

/// im2col → GEMM. Returns false when the operands cannot be read as flat
/// matrices, leaving the caller its gather.
template <typename T>
bool conv3d_im2col_gemm(const Conv3DGeom& gm, TensorView& output,
                        std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t karea = gm.karea();
    const int64_t kdim = gm.IC_per_G * karea;
    const int64_t npix = gm.npix();
    if (kdim <= 0 || npix <= 0 || gm.OC_per_G <= 0) {
        return false;
    }
    // The filter must be one dense [OC/G][IC/G·KD·KH·KW] matrix.
    if (weight.stride_elems(4) != 1 || weight.stride_elems(1) != karea) {
        return false;
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

    const int64_t wper = kdim * gm.OC_per_G;
    // The GEMM writes into the destination unless a residual has to survive it
    // or the destination's rows are not back to back.
    const bool dense_row = (out_row_stride == gm.OW * gm.OC);
    const bool direct = dense_row && !gm.add_to;
    const bool write_pass = !direct || has_bias || epilogue_is_active(gm.epilogue);

    const auto* in_ptr = input.ptr<T>();
    const auto* w_ptr = weight.ptr<T>();
    const auto* b_ptr = has_bias ? inputs[2].ptr<T>() : nullptr;
    auto* out_ptr = output.ptr<T>();
    const DataType vdt = input.data_type();
    const MatMulAttributes mm{};

    // One pixel block: gather A, GEMM into C, scatter the rows C cannot hold.
    // @p ictx is the context the GEMM and the scatter run on — the real one in
    // the serial arm (the pool lives inside them), an empty one in the grid arm
    // (the pool lives over the tasks).
    const auto run_block = [&](int64_t n, int64_t g, int64_t od0, int64_t oh0, int64_t od_chunk,
                               int64_t roi_h, T* col, T* stage, const T* wpacked,
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
        const int64_t bshape[] = {kdim, gm.OC_per_G};
        TensorView bm(std::span<const int64_t>(bshape, 2), vdt, const_cast<T*>(wpacked),
                      gm.OC_per_G * static_cast<int64_t>(sizeof(T)), TensorLayout::Dense);

        T* ctile = direct ? out_g + p0 * gm.OC : stage;
        const int64_t cshape[] = {nblk, gm.OC_per_G};
        TensorView cm(std::span<const int64_t>(cshape, 2), vdt, ctile,
                      (direct ? gm.OC : gm.OC_per_G) * static_cast<int64_t>(sizeof(T)),
                      TensorLayout::Dense);

        const TensorView mins[] = {am, bm};
        matmul_cpu(mm, cm, mins, ictx, nullptr);

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
            conv3d_choose_block(gm, kdim, static_cast<int64_t>(sizeof(T)), kColCapBytes);
        const int64_t nblk_max = blk.od_chunk * blk.roi_h * gm.OW;
        PoolArray<T> col(pointwise ? 0 : static_cast<size_t>(nblk_max * kdim));
        PoolArray<T> wpack(static_cast<size_t>(wper));
        PoolArray<T> stage(direct ? 0 : static_cast<size_t>(nblk_max * gm.OC_per_G));
        if ((!pointwise && !col) || !wpack || (!direct && !stage)) {
            return false;
        }
        for (int64_t n = 0; n < gm.N; ++n) {
            for (int64_t g = 0; g < gm.G; ++g) {
                conv3d_pack_weight(w_ptr + g * gm.OC_per_G * kdim, gm.OC_per_G, gm.IC_per_G,
                                   karea, wpack.data(), ctx);
                for (int64_t od0 = 0; od0 < gm.OD; od0 += blk.od_chunk) {
                    for (int64_t oh0 = 0; oh0 < gm.OH; oh0 += blk.roi_h) {
                        run_block(n, g, od0, oh0, blk.od_chunk, blk.roi_h, col.data(),
                                  stage.data(), wpack.data(), ctx);
                    }
                }
            }
        }
        return true;
    }

    // One task per (batch, group, pixel block): a plane with many groups but few
    // pixels per group still fills the pool. Each task's GEMM is serial, so the
    // per-task column budget is a slice of the serial arm's cap.
    const int64_t budget = std::max<int64_t>(kColCapBytes / workers, kMinTaskColBytes);
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
        return false;
    }

    // The filter is the same for every batch item, so it is packed once per
    // group when all groups fit the cap; otherwise each worker packs its own
    // group into its slot.
    const bool share_w = gm.G * wper * static_cast<int64_t>(sizeof(T)) <= kWeightPackCap;
    PoolArray<T> wshare(share_w ? static_cast<size_t>(gm.G * wper) : 0);
    PoolArray<T> wslot(share_w ? 0 : static_cast<size_t>(wper * slots));
    if ((share_w && !wshare) || (!share_w && !wslot)) {
        return false;
    }
    const ComputeContext serial_ctx{};
    if (share_w) {
        ctx.cpu.run(0, gm.G, [&](int64_t g) {
            conv3d_pack_weight(w_ptr + g * gm.OC_per_G * kdim, gm.OC_per_G, gm.IC_per_G, karea,
                               wshare.data() + g * wper, serial_ctx);
        });
    }

    ctx.cpu.run(0, tasks, [&](int64_t t) {
        const int64_t slot = (slots < tasks) ? ctx.cpu.current_thread_id() : t;
        const int64_t bi = t % blocks;
        const int64_t gi = t / blocks;
        const int64_t n = gi / gm.G;
        const int64_t g = gi % gm.G;
        const int64_t od0 = (bi / blocks_h) * blk.od_chunk;
        const int64_t oh0 = (bi % blocks_h) * blk.roi_h;

        T* const wbuf = share_w ? wshare.data() + g * wper : wslot.data() + slot * wper;
        if (!share_w) {
            conv3d_pack_weight(w_ptr + g * gm.OC_per_G * kdim, gm.OC_per_G, gm.IC_per_G, karea,
                               wbuf, serial_ctx);
        }
        run_block(n, g, od0, oh0, blk.od_chunk, blk.roi_h,
                  pointwise ? nullptr : col.data() + slot * nblk_max * kdim,
                  direct ? nullptr : stage.data() + slot * nblk_max * gm.OC_per_G, wbuf,
                  serial_ctx);
    });
    return true;
}

/// The output-driven gather: every output voxel walks its own taps. Correct for
/// every geometry, and the route a shape too odd for the col takes.
template <typename T>
void conv3d_gather(const Conv3DGeom& gm, TensorView& output, std::span<const TensorView> inputs,
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

    // The filter is walked by its own strides rather than as one k block.
    const int64_t w_oc_stride = weight.stride_elems(0);
    const int64_t w_ic_stride = weight.stride_elems(1);
    const int64_t w_d_stride = weight.stride_elems(2);
    const int64_t w_h_stride = weight.stride_elems(3);

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
                                        const T* w_row = w_ic + kd * w_d_stride + kh * w_h_stride;
                                        for (int64_t kw = 0; kw < gm.KW; ++kw) {
                                            const int64_t iw = ow * gm.SW + kw * gm.DW - gm.PW;
                                            if (iw < 0 || iw >= gm.IW) {
                                                continue;
                                            }
                                            sum += s_load(&in_row[iw * in_pix]) *
                                                   s_load(&w_row[kw]);
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

template <typename T>
void conv3d_impl(const Conv3DGeom& gm, TensorView& output, std::span<const TensorView> inputs,
                 const ComputeContext& ctx)
{
    if (gm.N == 0 || gm.OC == 0 || gm.OD == 0 || gm.OH == 0 || gm.OW == 0) {
        return;
    }
    if (conv3d_im2col_gemm<T>(gm, output, inputs, ctx)) {
        return;
    }
    conv3d_gather<T>(gm, output, inputs, ctx);
}

} // anonymous namespace

Status conv3d_cpu(const Conv3DAttributes& attrs, TensorView& output,
                  std::span<const TensorView> inputs, const ComputeContext& ctx,
                  void* /*workspace*/)
{
    const auto& input = inputs[0];
    const auto& weight = inputs[1];

    Conv3DGeom gm{};
    fill_conv_axes<3>(gm, attrs, input, output);
    gm.N = input.shape(0);
    gm.IC = input.shape(4);
    gm.OC = weight.shape(0);
    gm.IC_per_G = weight.shape(1);
    gm.G = attrs.groups;
    gm.OC_per_G = gm.OC / gm.G;

    if (weight.data_type() == DataType::s8) {
        return conv3d_i8_cpu(attrs, output, inputs, ctx);
    }
    dispatch_f32_f16(input.data_type(), "conv3d_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        conv3d_impl<T>(gm, output, inputs, ctx);
    });
    return Status::ok;
}

} // namespace nnops::backend::cpu
