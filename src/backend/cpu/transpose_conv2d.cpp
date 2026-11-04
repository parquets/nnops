/// @file transpose_conv2d.cpp
/// @brief CPU implementation of 2D transposed convolution, planar and
///        channel-last.
///
/// Gather formulation (output-driven), the exact inverse of regular conv's
/// indexing: output (oh, ow) reads input (ih, iw) where
/// oh + PH - kh·DH == ih·SH (and likewise for w), so every output pixel is
/// accumulated independently and add_to needs no save/restore pass. Weights are
/// planar [IC, OC/G, KH, KW] either way; the image tensor is NCHW or NHWC.
///
/// Planar dispatches one task per (n, oc) plane and walks the group's input
/// channels scalar. Channel-last repacks the weight once to
/// [G][KH][KW][IC/G][OC/G] (so consecutive output channels are contiguous) and
/// dispatches per (n, oh row), accumulating the whole OC/G vector with SIMD
/// since an input pixel's channels are contiguous too.

#include "nnops/ops/transpose_conv2d.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "simd_kernel/simd_epilogue.hpp"
#include "common/dtype_dispatch.hpp"
#include "common/memory_pool.hpp"

namespace nnops::backend::cpu {

using namespace nnops::simd;

template <typename T>
void tconv2d_impl(const TransposeConv2DAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t N  = input.shape(0);
    const int64_t IH = input.shape(input.rank() - 2);
    const int64_t IW = input.shape(input.rank() - 1);

    const int64_t IC       = weight.shape(0);
    const int64_t OC_per_G = weight.shape(1);
    const int64_t KH = attrs.kernel_size[0];
    const int64_t KW = attrs.kernel_size[1];

    const int64_t G = attrs.groups;
    const int64_t OC = OC_per_G * G;
    const int64_t IC_per_G = IC / G;

    const int64_t OH = output.shape(2);
    const int64_t OW = output.shape(3);

    const int64_t SH = attrs.stride[0];
    const int64_t SW = attrs.stride[1];
    const int64_t DH = attrs.dilation[0];
    const int64_t DW = attrs.dilation[1];
    const int64_t PH = attrs.padding[0];
    const int64_t PW = attrs.padding[1];

    if (N == 0 || OC == 0 || OH == 0 || OW == 0) { return; }

    auto* out_ptr = output.ptr<T>();
    const auto* in_ptr = input.ptr<T>();
    const auto* w_ptr  = weight.ptr<T>();
    const auto* b_ptr  = has_bias ? inputs[2].ptr<T>() : nullptr;

    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t in_n_stride    = input.stride_elems(0);
    const int64_t out_n_stride   = output.stride_elems(0);

    const auto epilogue = attrs.epilogue;
    const bool add_to = attrs.add_to;
    const T* type_tag = nullptr;

    if (!is_channel_last(input.layout())) {
        const int64_t in_c_stride  = input.stride_elems(1);
        const int64_t out_c_stride = output.stride_elems(1);

        const auto process = [&](int64_t task) {
            const int64_t n  = task / OC;
            const int64_t oc = task % OC;
            const int64_t g = oc / OC_per_G;
            const int64_t oc_l = oc % OC_per_G;

            const T* in_n = in_ptr + n * in_n_stride;
            T* out_plane = out_ptr + n * out_n_stride + oc * out_c_stride;

            for (int64_t oh = 0; oh < OH; ++oh) {
                T* orow = out_plane + oh * out_row_stride;

                for (int64_t ow = 0; ow < OW; ++ow) {
                    float acc = 0.0f;

                    for (int64_t kh = 0; kh < KH; ++kh) {
                        const int64_t num_h = oh + PH - kh * DH;
                        if (num_h < 0 || num_h % SH != 0) { continue; }
                        const int64_t ih = num_h / SH;
                        if (ih >= IH) { continue; }

                        for (int64_t kw = 0; kw < KW; ++kw) {
                            const int64_t num_w = ow + PW - kw * DW;
                            if (num_w < 0 || num_w % SW != 0) { continue; }
                            const int64_t iw = num_w / SW;
                            if (iw >= IW) { continue; }

                            for (int64_t ic_l = 0; ic_l < IC_per_G; ++ic_l) {
                                const int64_t ic = g * IC_per_G + ic_l;
                                const T* xp = in_n + ic * in_c_stride
                                            + ih * in_row_stride + iw;
                                const int64_t wi =
                                    ((ic * OC_per_G + oc_l) * KH + kh) * KW + kw;
                                acc += s_load(xp) * s_load(&w_ptr[wi]);
                            }
                        }
                    }

                    if (b_ptr) { acc += s_load(&b_ptr[oc]); }
                    acc = nnops::apply_epilogue(epilogue, acc, oc);
                    if (add_to) { acc += s_load(&orow[ow]); }
                    s_store(&orow[ow], acc);
                }
            }
        };

        ctx.cpu.run(0, N * OC, process);
        return;
    }

    // ---- channel-last ----
    constexpr int L = simd_lane_for<T>;
    const int64_t OC_span = OC_per_G;

    // Weight repack: [G][KH][KW][IC/G][OC/G].
    const int64_t wt_g_stride = KH * KW * IC_per_G * OC_span;
    PoolArray<T> wt(static_cast<size_t>(G * wt_g_stride));

    // Group-relative index into one [KH][KW][IC/G][OC/G] block.
    const auto wt_index = [&](int64_t kh, int64_t kw, int64_t ic_l, int64_t oc_l) {
        return ((kh * KW + kw) * IC_per_G + ic_l) * OC_span + oc_l;
    };

    for (int64_t ic = 0; ic < IC; ++ic) {
        const int64_t g    = ic / IC_per_G;
        const int64_t ic_l = ic % IC_per_G;
        for (int64_t oc_l = 0; oc_l < OC_span; ++oc_l) {
            const int64_t w_base = (ic * OC_span + oc_l) * KH * KW;
            for (int64_t kh = 0; kh < KH; ++kh) {
                for (int64_t kw = 0; kw < KW; ++kw) {
                    wt[static_cast<size_t>(g * wt_g_stride +
                                           wt_index(kh, kw, ic_l, oc_l))] =
                        w_ptr[w_base + kh * KW + kw];
                }
            }
        }
    }

    const auto process = [&](int64_t task) {
        const int64_t n  = task / OH;
        const int64_t oh = task % OH;

        const T* in_n = in_ptr + n * in_n_stride;
        T* orow = out_ptr + n * out_n_stride + oh * out_row_stride;

        for (int64_t ow = 0; ow < OW; ++ow) {
            T* dst = orow + ow * OC;

            for (int64_t g = 0; g < G; ++g) {
                const T* wg = wt.data() + g * wt_g_stride;
                const T* bg = b_ptr ? b_ptr + g * OC_span : nullptr;
                T* dg = dst + g * OC_span;

                int64_t c0 = 0;
                for (; c0 + L <= OC_span; c0 += L) {
                    vec_for<T> acc = v_zero(type_tag);

                    for (int64_t kh = 0; kh < KH; ++kh) {
                        const int64_t num_h = oh + PH - kh * DH;
                        if (num_h < 0 || num_h % SH != 0) { continue; }
                        const int64_t ih = num_h / SH;
                        if (ih >= IH) { continue; }

                        for (int64_t kw = 0; kw < KW; ++kw) {
                            const int64_t num_w = ow + PW - kw * DW;
                            if (num_w < 0 || num_w % SW != 0) { continue; }
                            const int64_t iw = num_w / SW;
                            if (iw >= IW) { continue; }

                            const T* pix = in_n + ih * in_row_stride
                                         + iw * IC + g * IC_per_G;
                            const T* wrow = wg + ((kh * KW + kw) * IC_per_G * OC_span
                                                + c0);

                            for (int64_t ic_l = 0; ic_l < IC_per_G; ++ic_l) {
                                const auto x = v_set1(type_tag, s_load(&pix[ic_l]));
                                acc = v_fmadd(x, v_load(wrow + ic_l * OC_span), acc);
                            }
                        }
                    }

                    acc = v_add(acc, bg ? v_load(bg + c0) : v_zero(type_tag));
                    acc = apply_epilogue_vec<T>(epilogue, type_tag, acc);
                    if (add_to) { acc = v_add(acc, v_load(dg + c0)); }
                    v_store(dg + c0, acc);
                }

                for (; c0 < OC_span; ++c0) {
                    float acc = 0.0f;

                    for (int64_t kh = 0; kh < KH; ++kh) {
                        const int64_t num_h = oh + PH - kh * DH;
                        if (num_h < 0 || num_h % SH != 0) { continue; }
                        const int64_t ih = num_h / SH;
                        if (ih >= IH) { continue; }

                        for (int64_t kw = 0; kw < KW; ++kw) {
                            const int64_t num_w = ow + PW - kw * DW;
                            if (num_w < 0 || num_w % SW != 0) { continue; }
                            const int64_t iw = num_w / SW;
                            if (iw >= IW) { continue; }

                            const T* pix = in_n + ih * in_row_stride
                                         + iw * IC + g * IC_per_G;

                            for (int64_t ic_l = 0; ic_l < IC_per_G; ++ic_l) {
                                acc += s_load(&pix[ic_l]) *
                                       s_load(&wg[wt_index(kh, kw, ic_l, c0)]);
                            }
                        }
                    }

                    if (bg) { acc += s_load(&bg[c0]); }
                    acc = nnops::apply_epilogue(epilogue, acc, g * OC_span + c0);
                    if (add_to) { acc += s_load(&dg[c0]); }
                    s_store(&dg[c0], acc);
                }
            }
        }
    };

    ctx.cpu.run(0, N * OH, process);
}

Status transpose_conv2d_cpu(const TransposeConv2DAttributes& attrs,
                            TensorView& output,
                            std::span<const TensorView> inputs,
                            const ComputeContext& ctx,
                            void* /*workspace*/)
{
    dispatch_f32_f16(inputs[0].data_type(), "transpose_conv2d_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        tconv2d_impl<T>(attrs, output, inputs, ctx);
    });
    return Status::ok;
}

}  // namespace nnops::backend::cpu
