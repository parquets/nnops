/// @file conv3d.cpp
/// @brief CPU Conv3D (NDHWC, f32/f16) — channel-last im2col → GEMM, with the
///        scalar gather as the fallback.
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
/// channel-last activation is a contiguous IC/G-channel run, so the gather is
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
/// Anything the flat operand views cannot describe falls back to the plain
/// output-driven gather, which is correct for every shape.
///
/// The weight stays [OC, IC/G, KD, KH, KW]; both the input and the output are
/// stored channel-last ([N, D, H, W, C] in memory, logical dims
/// [N, C, D, H, W]).

#include "nnops/ops/conv3d.hpp"
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"
#include "backend/cpu/simd_kernel/simd_im2col.hpp"
#include "backend/cpu/common/dtype_dispatch.hpp"
#include "backend/cpu/common/memory_pool.hpp"
#include "backend/cpu/matmul/matmul.h"
#include "nnops/core/parallel_for.hpp"

#include <algorithm>
#include <span>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

/// One convolution's geometry, read off the tensors once.
struct Geom {
    int64_t N, IC, ID, IH, IW;
    int64_t OC_per_G, IC_per_G, OC, G;
    int64_t KD, KH, KW, OD, OH, OW;
    int64_t SD, SH, SW, DD, DH, DW, PD, PH, PW;
    Epilogue epilogue;
    bool add_to;

    int64_t karea() const { return KD * KH * KW; }
    int64_t npix() const { return OD * OH * OW; }
};

/// Largest col one GEMM call may materialise; past it the pixel block splits.
constexpr int64_t kColCapBytes = 8 << 20;
/// Largest staged output tile; only caps the oc chunk when the GEMM cannot
/// write into the destination (a padded pitch, or add_to's live residual).
constexpr int64_t kTileCapBytes = 4 << 20;

/// Repack one group's filter [OC/G][IC/G][KD][KH][KW] into the GEMM's B,
/// [tap·ic + ic][oc]. Disjoint writes per output channel, so the split is free.
template <typename T>
void pack_weight_cb(const T* w, int64_t oc_per_g, int64_t ic_per_g, int64_t karea, T* out,
                    const ComputeContext& ctx)
{
    ctx.cpu.run(0, oc_per_g, [&](int64_t oc) {
        const T* src = w + oc * ic_per_g * karea;
        for (int64_t ic = 0; ic < ic_per_g; ++ic) {
            for (int64_t tap = 0; tap < karea; ++tap) {
                out[(tap * ic_per_g + ic) * oc_per_g + oc] = src[ic * karea + tap];
            }
        }
    });
}

/// im2col → GEMM. Returns false when the operands cannot be read as flat
/// matrices, leaving the caller its gather.
template <typename T>
bool conv3d_im2col_gemm(const Geom& gm, TensorView& output, std::span<const TensorView> inputs,
                        const ComputeContext& ctx)
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
    const int64_t in_d_stride = input.stride_elems(2);
    const int64_t in_row_stride = input.row_stride_elems();
    const int64_t in_pix = input.shape(1); // W step: one whole channel vector
    const int64_t out_n_stride = output.stride_elems(0);
    const int64_t out_d_stride = output.stride_elems(2);
    const int64_t out_row_stride = output.row_stride_elems();

    const bool in_dense =
        (in_row_stride == gm.IW * in_pix) && (in_d_stride == gm.IH * in_row_stride);

    // A 1x1x1 stride-1 convolution is a plain GEMM over the input plane: one
    // pixel step is one channel vector on both sides, so nothing is gathered.
    const bool pointwise = gm.KD == 1 && gm.KH == 1 && gm.KW == 1 && gm.SD == 1 && gm.SH == 1 &&
                           gm.SW == 1 && gm.PD == 0 && gm.PH == 0 && gm.PW == 0 && in_dense;

    const int64_t cap = kColCapBytes / static_cast<int64_t>(sizeof(T));
    const int64_t slice = gm.OH * gm.OW;

    // One rectangle per block, whole depth slices while the col fits one.
    int64_t od_chunk = 1;
    int64_t roi_h = gm.OH;
    if (kdim * slice <= cap) {
        od_chunk = std::max<int64_t>(1, std::min(gm.OD, cap / (kdim * slice)));
    } else {
        roi_h = std::max<int64_t>(1, std::min(gm.OH, cap / (kdim * gm.OW)));
    }
    const int64_t nblk_max = od_chunk * roi_h * gm.OW;

    // The GEMM writes into the destination unless a residual has to survive it
    // or the destination's rows are not back to back.
    const bool dense_row = (out_row_stride == gm.OW * gm.OC);
    const bool direct = dense_row && !gm.add_to;
    const bool write_pass = !direct || has_bias || epilogue_is_active(gm.epilogue);

    PoolArray<T> col(pointwise ? 0 : static_cast<size_t>(kdim * nblk_max));
    PoolArray<T> wpack(static_cast<size_t>(kdim * gm.OC_per_G));
    PoolArray<T> stage(direct ? 0 : static_cast<size_t>(nblk_max * gm.OC_per_G));
    if ((!pointwise && !col) || !wpack || (!direct && !stage)) {
        return false;
    }

    const auto* in_ptr = input.ptr<T>();
    const auto* w_ptr = weight.ptr<T>();
    const auto* b_ptr = has_bias ? inputs[2].ptr<T>() : nullptr;
    auto* out_ptr = output.ptr<T>();
    const DataType vdt = input.data_type();

    for (int64_t n = 0; n < gm.N; ++n) {
        const T* in_n = in_ptr + n * in_n_stride;
        T* out_n = out_ptr + n * out_n_stride;

        for (int64_t g = 0; g < gm.G; ++g) {
            const T* in_g = in_n + g * gm.IC_per_G;
            const T* w_g = w_ptr + g * gm.OC_per_G * kdim;
            T* out_g = out_n + g * gm.OC_per_G;
            const T* b_g = b_ptr ? b_ptr + g * gm.OC_per_G : nullptr;
            pack_weight_cb(w_g, gm.OC_per_G, gm.IC_per_G, karea, wpack.data(), ctx);

            for (int64_t od0 = 0; od0 < gm.OD; od0 += od_chunk) {
                const int64_t nd = std::min(od_chunk, gm.OD - od0);
                for (int64_t oh0 = 0; oh0 < gm.OH; oh0 += roi_h) {
                    const int64_t nh = std::min(roi_h, gm.OH - oh0);
                    const int64_t nblk = nd * nh * gm.OW;
                    const int64_t p0 = (od0 * gm.OH + oh0) * gm.OW;

                    const auto* a_src = pointwise
                        ? const_cast<const T*>(in_g + p0 * in_pix)
                        : static_cast<const T*>(col.data());
                    if (!pointwise) {
                        nnops::kernel::tiled_im2col_3d<T>(
                            col.data(), static_cast<int>(kdim), in_g, in_pix, in_d_stride,
                            in_row_stride, static_cast<int>(gm.IC_per_G),
                            static_cast<int>(od0), static_cast<int>(oh0), static_cast<int>(nd),
                            static_cast<int>(nh), static_cast<int>(gm.OW),
                            static_cast<int>(gm.ID), static_cast<int>(gm.IH),
                            static_cast<int>(gm.IW), static_cast<int>(gm.KD),
                            static_cast<int>(gm.KH), static_cast<int>(gm.KW),
                            static_cast<int>(gm.SD), static_cast<int>(gm.SH),
                            static_cast<int>(gm.SW), static_cast<int>(gm.DD),
                            static_cast<int>(gm.DH), static_cast<int>(gm.DW),
                            static_cast<int>(gm.PD), static_cast<int>(gm.PH),
                            static_cast<int>(gm.PW));
                    }

                    const int64_t ashape[] = {nblk, kdim};
                    TensorView am(std::span<const int64_t>(ashape, 2), vdt,
                                  const_cast<T*>(a_src),
                                  (pointwise ? in_pix : kdim) * static_cast<int64_t>(sizeof(T)),
                                  TensorLayout::Dense);
                    const int64_t bshape[] = {kdim, gm.OC_per_G};
                    TensorView bm(std::span<const int64_t>(bshape, 2), vdt, wpack.data(),
                                  gm.OC_per_G * static_cast<int64_t>(sizeof(T)),
                                  TensorLayout::Dense);

                    T* ctile = direct ? out_g + p0 * gm.OC : stage.data();
                    const int64_t cshape[] = {nblk, gm.OC_per_G};
                    TensorView cm(std::span<const int64_t>(cshape, 2), vdt, ctile,
                                  (direct ? gm.OC : gm.OC_per_G) *
                                      static_cast<int64_t>(sizeof(T)),
                                  TensorLayout::Dense);

                    const TensorView mins[] = {am, bm};
                    const MatMulAttributes mm{};
                    matmul_cpu(mm, cm, mins, ctx, nullptr);

                    if (direct && !write_pass) {
                        continue;
                    }

                    const T* src_base = direct ? ctile : stage.data();
                    const int64_t src_pitch = direct ? gm.OC : gm.OC_per_G;
                    const int64_t oc_base = g * gm.OC_per_G;
                    ctx.cpu.run(0, nblk, [&](int64_t p) {
                        const int64_t od = p / (nh * gm.OW);
                        const int64_t rem = p % (nh * gm.OW);
                        const int64_t oh = rem / gm.OW;
                        const int64_t ow = rem % gm.OW;
                        T* dst = out_g + (od0 + od) * out_d_stride +
                                 (oh0 + oh) * out_row_stride + ow * gm.OC;
                        const T* src = src_base + p * src_pitch;
                        apply_epilogue_row<T>(dst, src, gm.OC_per_G, b_g, /*bias_step=*/1,
                                              gm.epilogue, oc_base, /*quant_step=*/1, gm.add_to,
                                              /*pre_act=*/false);
                    });
                }
            }
        }
    }
    return true;
}

/// The output-driven gather: every output voxel walks its own taps. Correct for
/// every geometry, and the route a shape too odd for the col takes.
template <typename T>
void conv3d_gather(const Geom& gm, TensorView& output, std::span<const TensorView> inputs,
                   const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t in_n_stride = input.stride_elems(0);
    const int64_t in_d_stride = input.stride_elems(2);
    const int64_t in_row_stride = input.row_stride_elems();
    const int64_t in_pix = input.shape(1);
    const int64_t out_n_stride = output.stride_elems(0);
    const int64_t out_d_stride = output.stride_elems(2);
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
void conv3d_impl(const Geom& gm, TensorView& output, std::span<const TensorView> inputs,
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

    Geom gm{};
    gm.N = input.shape(0);
    gm.IC = input.shape(1);
    gm.ID = input.shape(2);
    gm.IH = input.shape(3);
    gm.IW = input.shape(4);
    gm.OC = weight.shape(0);
    gm.IC_per_G = weight.shape(1);
    gm.KD = attrs.kernel_size[0];
    gm.KH = attrs.kernel_size[1];
    gm.KW = attrs.kernel_size[2];
    gm.G = attrs.groups;
    gm.OC_per_G = gm.OC / gm.G;
    gm.OD = output.shape(2);
    gm.OH = output.shape(3);
    gm.OW = output.shape(4);
    gm.SD = attrs.stride[0];
    gm.SH = attrs.stride[1];
    gm.SW = attrs.stride[2];
    gm.DD = attrs.dilation[0];
    gm.DH = attrs.dilation[1];
    gm.DW = attrs.dilation[2];
    gm.PD = attrs.padding[0];
    gm.PH = attrs.padding[1];
    gm.PW = attrs.padding[2];
    gm.epilogue = attrs.epilogue;
    gm.add_to = attrs.add_to;

    dispatch_f32_f16(input.data_type(), "conv3d_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        conv3d_impl<T>(gm, output, inputs, ctx);
    });
    return Status::ok;
}

} // namespace nnops::backend::cpu
