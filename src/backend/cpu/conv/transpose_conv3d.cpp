/// @file transpose_conv3d.cpp
/// @brief CPU TransposeConv3D kernel (NDHWC) — the 2-D col route with a third
///        tap axis.
///
/// Same gather formulation as TransposeConv2D, one dimension further out. A col
/// GEMM per (n, group) does the channel reduction over IC/G, then a scatter
/// folds the (oc, kd, kh, kw) rows back into the volume
/// (simd_kernel/simd_col2im.hpp).
///
/// The volume is [ID·IH·IW] × [IC/G] voxels (one voxel stride of IC) and the
/// filter is permuted to [IC/G] × [KD·KH·KW·nc], so the col comes out
/// pixel-major and the scatter adds whole oc vectors. Depthwise, whose channels
/// are the vector axis, takes the input volume as its own col and weights each
/// tap instead.
///
/// The col route chunks the M side on an 8 MiB budget, which 3-D reaches sooner:
/// the pixel axis is one dimension larger. The merge that ends a scatter is the
/// only writer of an output element, which is what keeps add_to's original value.
///
/// The fallback is the plain output-driven gather, for the shapes the flat
/// operand views cannot describe: a pitched volume or a kernel taller than the
/// per-row tap list.
///
/// The weight stays [IC, OC/G, KD, KH, KW].

#include "nnops/ops/transpose_conv3d.hpp"
#include "backend/cpu/conv/conv_util.h"
#include "backend/cpu/conv/transpose_conv_common.h"
#include "nnops/core/parallel_for.hpp"
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"
#include "backend/cpu/matmul/matmul.h"
#include "backend/cpu/common/dtype_dispatch.hpp"
#include "backend/cpu/common/memory_pool.hpp"

#include <algorithm>

namespace nnops::backend::cpu {

using namespace nnops::simd;
using namespace nnops::kernel;

namespace {

// =========================================================================
//  NDHWC
// =========================================================================
//
// The 2-D NHWC col GEMM one axis deeper:
// A is the volume as [ID·IH·IW] × [IC/G] voxels (a voxel's channels are
// contiguous), B is the filter as [IC/G] × [KD·KH·KW·nc] taps, and the scatter
// adds whole oc vectors out of the pixel-major col. It is the 2-D NHWC
// structure (transpose_conv2d.cpp) one axis deeper — see that file for why the
// col beats the pixel kernel once the filter stops staying resident.
//
// Depthwise has its own route, as in 2-D: one channel per group makes the
// channel the vector axis, so the "col" is the input volume itself and each tap
// is one weighted vector add per voxel.

/// NDHWC through a col matrix — the 2-D route one axis deeper, and what
/// MNN's NC4HW4 deconvolution does: one GEMM per (batch, group) does the channel
/// reduction for the whole volume, and a depth/height tap list plus a
/// stride-residue scatter fold the col back. The chunk skeleton is transpose_conv_common.h's;
/// only the walk is 3-D. Returns false when the operands cannot be read as flat
/// matrices.
template <typename T>
bool tconv3d_gemm(const ConvChannels& gm, TensorView& output,
                               std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const auto& weight = inputs[1];

    if (weight.row_stride_elems() != gm.KW) {
        return false;
    }
    if (input.row_stride_elems() != gm.IW * gm.IC) {
        return false;
    }
    // The volume view is [ID·IH·IW] × [IC/G] voxels, so a depth slice has to be
    // one contiguous plane whose stride the H rows already imply.
    if (input.stride_elems(1) != gm.IH * gm.IW * gm.IC) {
        return false;
    }
    if (gm.KD > kMaxRowTaps || gm.KH > kMaxRowTaps) {
        return false;
    }

    const int64_t npix = gm.ID * gm.IH * gm.IW;
    if (npix <= 0 || gm.KD * gm.KH * gm.KW <= 0 || gm.IC_per_G <= 0) {
        return false;
    }

    // Two col orientations, exactly as in 2-D: [tap][oc] needs the filter
    // permuted; [oc][tap] lets the raw filter slice be B and pays a transpose
    // per voxel instead. Walk the smaller of the filter and the col.
    const bool direct_w = npix * gm.N < gm.IC_per_G;

    const Tconv3dColGeom cg = gm.col3_geom();
    const int64_t OH = gm.OH;
    return tconv_route<T>(
        gm, output, inputs, ctx, npix, gm.OD * gm.OH, gm.KD * gm.KH * gm.KW, direct_w,
        [cg, OH](const T* col, T* row, int64_t task, int64_t m, int64_t nc) {
            tconv3d_col2im_pixel_major<T>(
                cg, col, m, nc, nc, row, task / OH, task % OH,
                [](T* dst, const T* src, int64_t len, int64_t) { row_add(dst, src, len); });
        });
}

/// Depthwise: one channel per group means group g's input channel
/// is output channel g, so the vector axis is the channel itself and an output
/// voxel accumulates the whole C-wide tap vector at once, the "col" being the
/// input volume. The skeleton — the repack to [KD·KH·KW][C], the row scratch,
/// the merge — is transpose_conv_common.h's; only the walk is 3-D.
template <typename T>
bool tconv3d_depthwise(const ConvChannels& gm, TensorView& output,
                                    std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    if (gm.IC_per_G != 1 || gm.OC_per_G != 1) {
        return false;
    }

    const auto& input = inputs[0];
    const auto& weight = inputs[1];

    if (weight.row_stride_elems() != gm.KW) {
        return false;
    }
    if (input.row_stride_elems() != gm.IW * gm.IC) {
        return false;
    }
    if (input.stride_elems(1) != gm.IH * gm.IW * gm.IC) {
        return false;
    }

    const Tconv3dColGeom cg = gm.col3_geom();
    const int64_t OH = gm.OH;
    return tconv_depthwise<T>(
        gm, output, inputs, ctx, gm.OD * gm.OH, gm.KD * gm.KH * gm.KW, gm.IC,
        [cg, OH](const T* src, T* row, int64_t task, const T* wd, int64_t m, int64_t nc) {
            // The tap indexes the repacked filter, not a source stride.
            tconv3d_col2im_pixel_major<T>(cg, src, m, nc, 0, row, task / OH, task % OH,
                                          [wd, nc](T* dst, const T* s, int64_t len, int64_t tap) {
                                              row_fma(dst, s, wd + tap * nc, len);
                                          });
        });
}

/// The output-driven gather: every output voxel walks its own taps. Correct for
/// every geometry and layout, and the route a shape too odd for the col takes.
template <typename T>
void tconv3d_gather(const ConvChannels& gm, TensorView& output, std::span<const TensorView> inputs,
                    const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    // Channel-last: C is the innermost axis, D the one below H.
    const int64_t in_c_stride = input.stride_elems(4);
    const int64_t in_d_stride = input.stride_elems(1);
    const int64_t in_row_stride = input.row_stride_elems();
    const int64_t in_n_stride = input.stride_elems(0);
    const int64_t out_c_stride = output.stride_elems(4);
    const int64_t out_d_stride = output.stride_elems(1);
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_n_stride = output.stride_elems(0);

    const auto* in_ptr = input.ptr<T>();
    const auto* w_ptr = weight.ptr<T>();
    const auto* b_ptr = has_bias ? inputs[2].ptr<T>() : nullptr;
    auto* out_ptr = output.ptr<T>();

    // One voxel step is one channel; the row and slice strides carry the rest.
    const int64_t in_px = gm.IC;
    const int64_t out_px = gm.OC;

    const int64_t karea = gm.KD * gm.KH * gm.KW;

    ctx.cpu.run(0, gm.N * gm.OC, [&](int64_t task) {
        const int64_t n = task / gm.OC;
        const int64_t oc = task % gm.OC;
        const int64_t g = oc / gm.OC_per_G;
        const int64_t oc_l = oc % gm.OC_per_G;

        const T* in_n = in_ptr + n * in_n_stride;
        T* out_c = out_ptr + n * out_n_stride + oc * out_c_stride;

        for (int64_t od = 0; od < gm.OD; ++od) {
            for (int64_t oh = 0; oh < gm.OH; ++oh) {
                T* orow = out_c + od * out_d_stride + oh * out_row_stride;

                for (int64_t ow = 0; ow < gm.OW; ++ow) {
                    T* op = orow + ow * out_px;
                    float acc = 0.0f;

                    for (int64_t kd = 0; kd < gm.KD; ++kd) {
                        const int64_t num_d = od + gm.PD - kd * gm.DD;
                        if (num_d < 0 || num_d % gm.SD != 0) {
                            continue;
                        }
                        const int64_t id = num_d / gm.SD;
                        if (id >= gm.ID) {
                            continue;
                        }

                        for (int64_t kh = 0; kh < gm.KH; ++kh) {
                            const int64_t num_h = oh + gm.PH - kh * gm.DH;
                            if (num_h < 0 || num_h % gm.SH != 0) {
                                continue;
                            }
                            const int64_t ih = num_h / gm.SH;
                            if (ih >= gm.IH) {
                                continue;
                            }

                            for (int64_t kw = 0; kw < gm.KW; ++kw) {
                                const int64_t num_w = ow + gm.PW - kw * gm.DW;
                                if (num_w < 0 || num_w % gm.SW != 0) {
                                    continue;
                                }
                                const int64_t iw = num_w / gm.SW;
                                if (iw >= gm.IW) {
                                    continue;
                                }

                                const T* xp = in_n + (g * gm.IC_per_G) * in_c_stride +
                                              id * in_d_stride + ih * in_row_stride + iw * in_px;
                                const T* wp = w_ptr +
                                              ((g * gm.IC_per_G) * gm.OC_per_G + oc_l) * karea +
                                              (kd * gm.KH + kh) * gm.KW + kw;
                                for (int64_t ic_l = 0; ic_l < gm.IC_per_G; ++ic_l) {
                                    acc += s_load(xp + ic_l * in_c_stride) *
                                           s_load(wp + ic_l * (gm.OC_per_G * karea));
                                }
                            }
                        }
                    }

                    if (b_ptr) {
                        acc += s_load(&b_ptr[oc]);
                    }
                    apply_epilogue_store<T>(op, acc, gm.epilogue, oc, gm.add_to, gm.pre_act);
                }
            }
        }
    });
}

template <typename T>
void tconv3d_impl(const ConvChannels& gm, TensorView& output, std::span<const TensorView> inputs,
                  const ComputeContext& ctx)
{
    if (gm.N == 0 || gm.OC == 0 || gm.OD == 0 || gm.OH == 0 || gm.OW == 0) {
        return;
    }
    if (tconv3d_depthwise<T>(gm, output, inputs, ctx)) {
        return;
    }
    if (tconv3d_gemm<T>(gm, output, inputs, ctx)) {
        return;
    }
    tconv3d_gather<T>(gm, output, inputs, ctx);
}

} // anonymous namespace

Status transpose_conv3d_cpu(const TransposeConv3DAttributes& attrs, TensorView& output,
                            std::span<const TensorView> inputs, const ComputeContext& ctx,
                            void* /*workspace*/)
{
    const auto& input = inputs[0];
    const auto& weight = inputs[1];

    ConvChannels gm{};
    fill_conv_axes<3>(gm, attrs, input, output);
    gm.N = input.shape(0);
    gm.IC = input.shape(4);
    gm.OC_per_G = weight.shape(1);
    gm.G = attrs.groups;
    gm.OC = gm.OC_per_G * gm.G;
    gm.IC_per_G = gm.IC / gm.G;

    dispatch_f32_f16(input.data_type(), "transpose_conv3d_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        tconv3d_impl<T>(gm, output, inputs, ctx);
    });
    return Status::ok;
}

} // namespace nnops::backend::cpu
