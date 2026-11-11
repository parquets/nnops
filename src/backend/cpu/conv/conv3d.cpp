/// @file conv3d.cpp
/// @brief CPU Conv3D (NCDHW, f32/f16) — im2col → GEMM, with the scalar gather
///        as the fallback.
///
/// Forward convolution is the easy direction for a col matrix: the filter's own
/// [OC/G][IC/G·KD·KH·KW] rows are already the GEMM's A, so one GEMM per
/// (batch, group, pixel block) does the whole channel-and-tap reduction,
///
///   A = the filter, read where it lies (no transpose, no repack),
///   B = the gathered window rows, [IC/G·KD·KH·KW][pixels],
///   C = [OC/G × pixels],
///
/// and what is left is the epilogue pass that writes the tile out. That is the
/// mirror image of the transposed convolutions, whose col is the *output* of
/// the GEMM and has to be folded back into the image.
///
/// The gather is tiled_im2col_3d (simd_kernel/simd_im2col.hpp), one call per
/// block, zero-filling the six padded faces. A block is a rectangle of output —
/// whole depth slices while they fit, otherwise a run of rows inside one slice
/// — because the col's row stride is the block's own volume, so the rectangle
/// has to be the unit. The pixel axis is what gets blocked, never the oc axis:
/// every oc chunk shares one gather. A 1x1x1 stride-1 convolution skips the
/// gather entirely and reads the input plane as B, which is what the video
/// pointwise layers spend their time in.
///
/// The GEMM writes straight into the destination whenever its rows are back to
/// back, so the epilogue pass is a read-modify-write of the tile that is already
/// there; add_to (whose residual would be destroyed) and a padded pitch take the
/// pooled staging tile instead. The pass is skipped outright when the tile needs
/// nothing — no bias, no live epilogue, no add_to.
///
/// Anything the flat operand views cannot describe — a filter whose rows are not
/// its logical k width — falls back to the plain output-driven gather, which is
/// correct for every shape and layout.
///
/// The weight stays channel-first [OC, IC/G, KD, KH, KW]; both the input and the
/// output are [N, C, D, H, W].

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

/// Write one block's GEMM result into the output row: bias, epilogue, add_to.
/// @p src and @p dst may alias — the residual then comes from the same tile,
/// which is why a destination that needs add_to is staged instead.
template <typename T>
void write_tile(T* dst, const T* src, int64_t len, float bias_val, int64_t oc,
                const Geom& gm, bool has_bias)
{
    const T* tag = nullptr;
    constexpr int L = simd_lane_for<T>;
    int64_t p = 0;
    if (gm.epilogue.quant == EpilogueQuantType::None) {
        const auto bv = v_set1(tag, bias_val);
        for (; p + L <= len; p += L) {
            auto acc = v_load(src + p);
            if (has_bias) { acc = v_add(acc, bv); }
            acc = apply_epilogue_vec<T>(gm.epilogue, tag, acc);
            if (gm.add_to) { acc = v_add(acc, v_load(dst + p)); }
            v_store(dst + p, acc);
        }
    }
    for (; p < len; ++p) {
        float v = s_load(src + p);
        if (has_bias) { v += bias_val; }
        v = nnops::apply_epilogue(gm.epilogue, v, oc);
        if (gm.add_to) { v += s_load(dst + p); }
        s_store(dst + p, v);
    }
}

/// im2col → GEMM. Returns false when the operands cannot be read as flat
/// matrices, leaving the caller its gather.
template <typename T>
bool conv3d_im2col_gemm(const Geom& gm,
                        TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t karea = gm.karea();
    const int64_t kdim  = gm.IC_per_G * karea;
    const int64_t npix  = gm.npix();
    if (kdim <= 0 || npix <= 0 || gm.OC_per_G <= 0) { return false; }
    // A is the filter itself, so [OC/G][IC/G·KD·KH·KW] must be one dense
    // matrix: each k row back to back, and each OC row a whole k row further.
    if (weight.stride_elems(3) != gm.KW || weight.stride_elems(1) != karea) {
        return false;
    }

    const int64_t in_n_stride = input.stride_elems(0);
    const int64_t in_c_stride = input.stride_elems(1);
    const int64_t in_d_stride = input.stride_elems(2);
    const int64_t in_h_stride = input.row_stride_elems();
    const int64_t out_n_stride   = output.stride_elems(0);
    const int64_t out_c_stride   = output.stride_elems(1);
    const int64_t out_row_stride = output.row_stride_elems();

    // A 1x1x1 stride-1 convolution is a plain GEMM over the input plane: one
    // pixel step is one element on both sides, so no column has to be gathered.
    const bool pointwise =
        gm.KD == 1 && gm.KH == 1 && gm.KW == 1 &&
        gm.SD == 1 && gm.SH == 1 && gm.SW == 1 &&
        gm.PD == 0 && gm.PH == 0 && gm.PW == 0 &&
        in_h_stride == gm.IW && in_d_stride == gm.IH * gm.IW;

    const int64_t cap = kColCapBytes / static_cast<int64_t>(sizeof(T));
    const int64_t slice = gm.OH * gm.OW;

    // One rectangle per block, whole depth slices while the col fits one.
    int64_t od_chunk = 1;
    int64_t roi_h    = gm.OH;
    if (kdim * slice <= cap) {
        od_chunk = std::max<int64_t>(1, std::min(gm.OD, cap / (kdim * slice)));
    } else {
        roi_h = std::max<int64_t>(1, std::min(gm.OH, cap / (kdim * gm.OW)));
    }
    const int64_t nblk_max = od_chunk * roi_h * gm.OW;

    // The GEMM writes into the destination unless a residual has to survive it
    // or the destination's rows are not back to back.
    const bool dense_out = (out_row_stride == gm.OW);
    const bool in_place  = dense_out && !gm.add_to;
    const bool write_pass = has_bias || gm.add_to || epilogue_is_active(gm.epilogue);

    const int64_t tile_cap = (kTileCapBytes / static_cast<int64_t>(sizeof(T))) /
                             std::max<int64_t>(1, nblk_max);
    const int64_t cc = in_place ? gm.OC_per_G
                                : std::min(gm.OC_per_G, std::max<int64_t>(1, tile_cap));

    PoolArray<T> col(pointwise ? 0 : static_cast<size_t>(kdim * nblk_max));
    PoolArray<T> stage(in_place ? 0 : static_cast<size_t>(cc * nblk_max));
    if ((!pointwise && !col) || (!in_place && !stage)) { return false; }

    const auto* in_ptr = input.ptr<T>();
    const auto* w_ptr  = weight.ptr<T>();
    const auto* b_ptr  = has_bias ? inputs[2].ptr<T>() : nullptr;
    auto* out_ptr = output.ptr<T>();
    const DataType vdt = input.data_type();

    const int pw = gm.PW;  // the roi starts at ow = 0, so only PW can cut it

    for (int64_t n = 0; n < gm.N; ++n) {
        const T* in_n  = in_ptr + n * in_n_stride;
        T* out_n       = out_ptr + n * out_n_stride;

        for (int64_t g = 0; g < gm.G; ++g) {
            const T* in_g  = in_n + g * gm.IC_per_G * in_c_stride;
            const T* w_g   = w_ptr + g * gm.OC_per_G * kdim;
            T* out_g       = out_n + g * gm.OC_per_G * out_c_stride;

            for (int64_t od0 = 0; od0 < gm.OD; od0 += od_chunk) {
                const int64_t nd = std::min(od_chunk, gm.OD - od0);
                for (int64_t oh0 = 0; oh0 < gm.OH; oh0 += roi_h) {
                    const int64_t nh   = std::min(roi_h, gm.OH - oh0);
                    const int64_t nblk = nd * nh * gm.OW;
                    const int64_t p0   = (od0 * gm.OH + oh0) * gm.OW;

                    if (pointwise) {
                        // Nothing to gather: the input plane is B.
                    } else {
                        // How far the block start falls inside the input, in
                        // output coordinates — the walk subtracts the block
                        // origin itself, so a start pad has to include it.
                        const int64_t id = od0 * gm.SD - gm.PD;
                        const int64_t ih = oh0 * gm.SH - gm.PH;
                        const int pd = id < 0 ? static_cast<int>(-id)
                                              : (id >= gm.ID ? static_cast<int>(id - gm.ID) : 0);
                        const int ph = ih < 0 ? static_cast<int>(-ih)
                                              : (ih >= gm.IH ? static_cast<int>(ih - gm.IH) : 0);
                        nnops::kernel::tiled_im2col_3d<T>(
                            col.data(), in_g,
                            static_cast<int>(od0), static_cast<int>(oh0), 0,
                            static_cast<int>(nd), static_cast<int>(nh), static_cast<int>(gm.OW),
                            pd + static_cast<int>(od0 * gm.SD),
                            ph + static_cast<int>(oh0 * gm.SH), pw,
                            static_cast<int>(gm.PD), static_cast<int>(gm.PH),
                            static_cast<int>(gm.PW),
                            static_cast<int>(gm.IC_per_G), static_cast<int>(gm.ID),
                            static_cast<int>(gm.IH), static_cast<int>(gm.IW),
                            static_cast<int>(in_c_stride), static_cast<int>(in_d_stride),
                            static_cast<int>(in_h_stride),
                            static_cast<int>(gm.KD), static_cast<int>(gm.KH),
                            static_cast<int>(gm.KW),
                            static_cast<int>(gm.SD), static_cast<int>(gm.SH),
                            static_cast<int>(gm.SW),
                            static_cast<int>(gm.DD), static_cast<int>(gm.DH),
                            static_cast<int>(gm.DW));
                    }

                    const int64_t bshape[] = {kdim, nblk};
                    TensorView bm(std::span<const int64_t>(bshape, 2), vdt,
                                  const_cast<T*>(pointwise ? in_g + p0 : col.data()),
                                  (pointwise ? in_c_stride : nblk) * static_cast<int64_t>(sizeof(T)),
                                  TensorLayout::NCDHW);

                    for (int64_t oc0 = 0; oc0 < gm.OC_per_G; oc0 += cc) {
                        const int64_t ncc = std::min(cc, gm.OC_per_G - oc0);
                        T* ctile = in_place
                            ? out_g + oc0 * out_c_stride + p0
                            : stage.data();

                        const int64_t ashape[] = {ncc, kdim};
                        TensorView am(std::span<const int64_t>(ashape, 2), vdt,
                                      const_cast<T*>(w_g + oc0 * kdim),
                                      static_cast<int64_t>(kdim * sizeof(T)),
                                      TensorLayout::NCDHW);
                        const int64_t cshape[] = {ncc, nblk};
                        TensorView cm(std::span<const int64_t>(cshape, 2), vdt, ctile,
                                      (in_place ? out_c_stride : nblk) *
                                          static_cast<int64_t>(sizeof(T)),
                                      TensorLayout::NCDHW);

                        const TensorView mins[] = {am, bm};
                        const MatMulAttributes mm{};
                        matmul_cpu(mm, cm, mins, ctx, nullptr);

                        if (in_place && !write_pass) { continue; }

                        const int64_t crow = in_place ? out_c_stride : nblk;
                        ctx.cpu.run(0, ncc, [&](int64_t oc_l) {
                            const int64_t oc = g * gm.OC_per_G + oc0 + oc_l;
                            const float bv = b_ptr ? s_load(&b_ptr[oc]) : 0.0f;
                            const T* src = ctile + oc_l * crow;
                            T* dst = out_g + (oc0 + oc_l) * out_c_stride + p0;
                            if (dense_out) {
                                write_tile(dst, src, nblk, bv, oc, gm, has_bias);
                            } else {
                                for (int64_t od = 0; od < nd; ++od) {
                                    for (int64_t oh = 0; oh < nh; ++oh) {
                                        const int64_t off = (od * nh + oh) * gm.OW;
                                        write_tile(dst + (od0 + od) * gm.OH * out_row_stride +
                                                       (oh0 + oh) * out_row_stride,
                                                   src + off, gm.OW, bv, oc, gm, has_bias);
                                    }
                                }
                            }
                        });
                    }
                }
            }
        }
    }
    return true;
}

/// The output-driven gather: every output voxel walks its own taps. Correct for
/// every geometry and layout, and the route a shape too odd for the col takes.
template <typename T>
void conv3d_gather(const Geom& gm,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t in_n_stride = input.stride_elems(0);
    const int64_t in_c_stride = input.stride_elems(1);
    const int64_t in_d_stride = input.stride_elems(2);
    const int64_t in_h_stride = input.row_stride_elems();
    const int64_t out_n_stride   = output.stride_elems(0);
    const int64_t out_c_stride   = output.stride_elems(1);
    const int64_t out_d_stride   = output.stride_elems(2);
    const int64_t out_row_stride = output.row_stride_elems();

    // This route exists for layouts the flat operand views cannot describe, so
    // the filter is walked by its own strides rather than as one k block.
    const int64_t w_oc_stride = weight.stride_elems(0);
    const int64_t w_ic_stride = weight.stride_elems(1);
    const int64_t w_d_stride  = weight.stride_elems(2);
    const int64_t w_h_stride  = weight.stride_elems(3);

    const auto* in_ptr = input.ptr<T>();
    const auto* w_ptr  = weight.ptr<T>();
    const auto* b_ptr  = has_bias ? inputs[2].ptr<T>() : nullptr;
    auto* out_ptr = output.ptr<T>();

    ctx.cpu.run(0, gm.N, [&](int64_t n) {
        for (int64_t g = 0; g < gm.G; ++g) {
            for (int64_t oc_l = 0; oc_l < gm.OC_per_G; ++oc_l) {
                const int64_t oc = g * gm.OC_per_G + oc_l;
                const T* w_oc = w_ptr + oc * w_oc_stride;
                const float bv = b_ptr ? s_load(&b_ptr[oc]) : 0.0f;
                for (int64_t od = 0; od < gm.OD; ++od) {
                    for (int64_t oh = 0; oh < gm.OH; ++oh) {
                        T* orow = out_ptr + n * out_n_stride + oc * out_c_stride +
                                  od * out_d_stride + oh * out_row_stride;
                        for (int64_t ow = 0; ow < gm.OW; ++ow) {
                            float sum = 0.0f;
                            for (int64_t ic = 0; ic < gm.IC_per_G; ++ic) {
                                const int64_t ic_global = g * gm.IC_per_G + ic;
                                const T* in_c = in_ptr + n * in_n_stride +
                                                ic_global * in_c_stride;
                                const T* w_ic = w_oc + ic * w_ic_stride;
                                for (int64_t kd = 0; kd < gm.KD; ++kd) {
                                    const int64_t id = od * gm.SD + kd * gm.DD - gm.PD;
                                    if (id < 0 || id >= gm.ID) { continue; }
                                    for (int64_t kh = 0; kh < gm.KH; ++kh) {
                                        const int64_t ih = oh * gm.SH + kh * gm.DH - gm.PH;
                                        if (ih < 0 || ih >= gm.IH) { continue; }
                                        const T* in_row = in_c + id * in_d_stride +
                                                          ih * in_h_stride;
                                        const T* w_row = w_ic + kd * w_d_stride +
                                                         kh * w_h_stride;
                                        for (int64_t kw = 0; kw < gm.KW; ++kw) {
                                            const int64_t iw = ow * gm.SW + kw * gm.DW - gm.PW;
                                            if (iw < 0 || iw >= gm.IW) { continue; }
                                            sum += s_load(&in_row[iw]) * s_load(&w_row[kw]);
                                        }
                                    }
                                }
                            }
                            float val = nnops::apply_epilogue(gm.epilogue, sum + bv, oc);
                            if (gm.add_to) { val += s_load(&orow[ow]); }
                            s_store(&orow[ow], val);
                        }
                    }
                }
            }
        }
    });
}

template <typename T>
void conv3d_impl(const Geom& gm,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx)
{
    if (gm.N == 0 || gm.OC == 0 || gm.OD == 0 || gm.OH == 0 || gm.OW == 0) { return; }
    if (conv3d_im2col_gemm<T>(gm, output, inputs, ctx)) { return; }
    conv3d_gather<T>(gm, output, inputs, ctx);
}

}  // anonymous namespace

Status conv3d_cpu(const Conv3DAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx,
                  void* /*workspace*/)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];

    Geom gm{};
    gm.N  = input.shape(0);
    gm.IC = input.shape(1);
    gm.ID = input.shape(2);
    gm.IH = input.shape(3);
    gm.IW = input.shape(4);
    gm.OC       = weight.shape(0);
    gm.IC_per_G = weight.shape(1);
    gm.KD = attrs.kernel_size[0];
    gm.KH = attrs.kernel_size[1];
    gm.KW = attrs.kernel_size[2];
    gm.G  = attrs.groups;
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

}  // namespace nnops::backend::cpu
