/// @file transpose_conv2d.cpp
/// @brief CPU implementation of 2D transposed convolution on NHWC.
///
/// Gather formulation (output-driven), the exact inverse of regular conv's
/// indexing: output (oh, ow) reads input (ih, iw) where
/// oh + PH - kh·DH == ih·SH (and likewise for w), so every output pixel is
/// accumulated independently and add_to needs no save/restore pass — in either
/// order: residual_pre_activation reads the destination into the sum before
/// the epilogue at every merge site. Weights are [IC, OC/G, KH, KW]; the image
/// tensor is NHWC.
///
/// Three routes. The pixel kernel repacks the weight once to
/// [G][KH][KW][IC/G][OC/G] (so consecutive output channels are contiguous) and
/// dispatches per (n, oh row), accumulating the whole OC/G vector with SIMD
/// since an input pixel's channels are contiguous too. Two per-class facts are
/// hoisted out of the pixel loop: which (kh, ih) rows land on this output row,
/// and which (kw, shift) taps land on a given output-column residue. That
/// leaves a valid window [t_lo, t_hi) in which every tap of the class is in
/// range, and there the row is walked four pixels at a time against two oc
/// vectors — one weight vector then serves four pixels. Outside the window a
/// single pixel goes through the general path.
///
/// That reuse only survives while the filter does: a large filter is re-read
/// from DRAM on every tile, and a stride-1 filter has many taps per output
/// pixel, so both go through the col matrix instead — one GEMM per (n, group),
/// with the image as [npix][IC/G] and the filter repacked to
/// [IC/G][KH·KW·OC/G], followed by a stride-residue scatter that adds whole oc
/// vectors. Depthwise (one channel per group) has its own route: group g's
/// input channel *is* output channel g, so the vector axis is the channel and
/// every tap is one contiguous vector add over C.
///
/// Both GEMM routes are guarded on dense operands (a flat matrix view needs
/// both rows to be their logical width); anything else — and any kernel too
/// tall for the per-row tap list — falls back to the plain scalar gather, which
/// is correct for every shape.

#include "transpose_conv2d_internal.h"
#include "backend/cpu/simd_kernel/simd_col2im.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"
#include "backend/cpu/common/dtype_dispatch.hpp"
#include "backend/cpu/common/memory_pool.hpp"

#include <algorithm>
#include <cstring>
#include <type_traits>

namespace nnops::backend::cpu {

using namespace nnops::simd;
using namespace nnops::kernel;

namespace {



/// Depthwise. One channel per group means group g's input channel
/// is output channel g, so the vector axis is the channel itself: an output
/// pixel accumulates the whole C-wide tap vector at once, and the "col" the
/// scatter reads is the input plane. The skeleton — the repack to [KH·KW][C],
/// the row scratch, the merge — is transpose_conv_common.h's; only the walk is 2-D.
template <typename T>
bool tconv2d_depthwise(const ConvChannels& gm, TensorView& output,
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

    const Tconv2dColGeom cg = gm.col_geom();
    return tconv_depthwise<T>(
        gm, output, inputs, ctx, gm.OH, gm.KH * gm.KW, gm.IC,
        [cg](const T* src, T* row, int64_t oh, const T* wd, int64_t m, int64_t nc) {
            // The tap indexes the repacked filter, not a source stride.
            tconv2d_col2im_pixel_major<T>(cg, src, m, nc, 0, row, oh,
                                          [wd, nc](T* dst, const T* s, int64_t len, int64_t tap) {
                                              row_fma(dst, s, wd + tap * nc, len);
                                          });
        });
}

/// NHWC through a col matrix — onnxruntime's NHWC structure,
/// and what MNN's NC4HW4 deconvolution does: one GEMM per (batch, group) does
/// the channel reduction for the whole input plane, so the filter is walked
/// once per group instead of once per pixel tile, and a stride-residue scatter
/// then adds whole oc vectors out of the col. That pass over the filter is the
/// entire point on a layer where the pixel kernel's re-read goes to DRAM.
/// The chunk skeleton is transpose_conv_common.h's; only the walk is 2-D.
/// Returns false when the operands cannot be read as flat matrices.
template <typename T>
bool tconv2d_gemm(const ConvChannels& gm, TensorView& output,
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

    const int64_t npix = gm.IH * gm.IW;
    if (npix <= 0 || gm.KH * gm.KW <= 0 || gm.IC_per_G <= 0) {
        return false;
    }

    // Two col orientations. Ordered [tap][oc] the GEMM needs the filter
    // permuted to [IC/G][KH·KW·OC/G], but the col feeds the scatter as it is.
    // Ordered [oc][tap] the raw filter slice *is* B — oc runs faster than tap
    // in the [IC][OC/G][taps] weight already — and the col pays a transpose instead, per
    // batch item. One of the two is walked twice either way, so walk the
    // smaller: the filter once, or the col per batch item.
    const bool direct_w = npix * gm.N < gm.IC_per_G;

    const Tconv2dColGeom cg = gm.col_geom();
    return tconv_route<T>(
        gm, output, inputs, ctx, npix, gm.OH, gm.KH * gm.KW, direct_w,
        [cg](const T* col, T* row, int64_t oh, int64_t m, int64_t nc) {
            tconv2d_col2im_pixel_major<T>(
                cg, col, m, nc, nc, row, oh,
                [](T* dst, const T* src, int64_t len, int64_t) { row_add(dst, src, len); });
        });
}

template <typename T>
void tconv2d_impl(const ConvChannels& gm, TensorView& output, std::span<const TensorView> inputs,
                  const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t N = gm.N;
    const int64_t IH = gm.IH;
    const int64_t IW = gm.IW;

    const int64_t IC = gm.IC;
    const int64_t OC_per_G = gm.OC_per_G;
    const int64_t KH = gm.KH;
    const int64_t KW = gm.KW;

    const int64_t G = gm.G;
    const int64_t OC = gm.OC;
    const int64_t IC_per_G = gm.IC_per_G;

    const int64_t OH = gm.OH;
    const int64_t OW = gm.OW;

    const int64_t SH = gm.SH;
    const int64_t SW = gm.SW;
    const int64_t DH = gm.DH;
    const int64_t DW = gm.DW;
    const int64_t PH = gm.PH;
    const int64_t PW = gm.PW;

    if (N == 0 || OC == 0 || OH == 0 || OW == 0) {
        return;
    }

    auto* out_ptr = output.ptr<T>();
    const auto* in_ptr = input.ptr<T>();
    const auto* w_ptr = weight.ptr<T>();
    const auto* b_ptr = has_bias ? inputs[2].ptr<T>() : nullptr;

    const int64_t in_row_stride = input.row_stride_elems();
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t in_n_stride = input.stride_elems(0);
    const int64_t out_n_stride = output.stride_elems(0);

    const auto epilogue = gm.epilogue;
    const T* type_tag = nullptr;

    // ---- NHWC ----
    constexpr int L = simd_lane_for<T>;
    constexpr int64_t kPix = 4;
    const int64_t OC_span = OC_per_G;

    // Route. The pixel kernel's one trick is reusing a weight vector across
    // four pixels, which pays only when the filter stays resident between
    // tiles and few taps land on each output pixel; past that it re-reads the
    // filter from DRAM. Both other cases go through a col matrix, which walks
    // the filter once per (n, group) whatever the geometry. Depthwise fits
    // neither — its channels, not its pixels, are the vector axis.
    if (tconv2d_depthwise<T>(gm, output, inputs, ctx)) {
        return;
    }

    // In f16 every vector FMA is a convert pair, so the pixel kernel's
    // broadcast-and-reuse does not pay for itself the way it does in f32; the
    // GEMM kernels convert whole panels and win on every shape.
    const int64_t wbytes = gm.IC * OC_span * gm.KH * gm.KW * static_cast<int64_t>(sizeof(T));
    const bool pixel_form =
        std::is_same_v<T, float> && gm.KH * gm.KW <= 3 * gm.SH * gm.SW && wbytes <= (512 << 10);
    if (!pixel_form && tconv2d_gemm<T>(gm, output, inputs, ctx)) {
        return;
    }

    // Weight repack: [G][KH][KW][IC/G][OC/G].
    const int64_t wt_g_stride = KH * KW * IC_per_G * OC_span;
    PoolArray<T> wt(static_cast<size_t>(G * wt_g_stride));

    // Per-worker tap lists: the output row's (kh, ih) pairs, then one output
    // column residue's (kw, iw shift) pairs. Both are per-*class* facts, so the
    // divisibility tests leave the pixel loop entirely.
    const int64_t slots = std::max(1, ctx.cpu.worker_count());
    const int64_t tap_cap = KH + SW * KW;
    PoolArray<int64_t> tap_keys(static_cast<size_t>(slots * tap_cap));
    PoolArray<int64_t> tap_vals(static_cast<size_t>(slots * tap_cap));
    if (!wt || !tap_keys || !tap_vals) {
        return;
    }

    // Group-relative index into one [KH][KW][IC/G][OC/G] block.
    const auto wt_index = [&](int64_t kh, int64_t kw, int64_t ic_l, int64_t oc_l) {
        return ((kh * KW + kw) * IC_per_G + ic_l) * OC_span + oc_l;
    };

    for (int64_t ic = 0; ic < IC; ++ic) {
        const int64_t g = ic / IC_per_G;
        const int64_t ic_l = ic % IC_per_G;
        for (int64_t oc_l = 0; oc_l < OC_span; ++oc_l) {
            const int64_t w_base = (ic * OC_span + oc_l) * KH * KW;
            for (int64_t kh = 0; kh < KH; ++kh) {
                for (int64_t kw = 0; kw < KW; ++kw) {
                    wt[static_cast<size_t>(g * wt_g_stride + wt_index(kh, kw, ic_l, oc_l))] =
                        w_ptr[w_base + kh * KW + kw];
                }
            }
        }
    }

    const auto process = [&](int64_t task) {
        const int64_t n = task / OH;
        const int64_t oh = task % OH;
        const int64_t slot = static_cast<int64_t>(ctx.cpu.current_thread_id());

        int64_t* kh_of = tap_keys.data() + slot * tap_cap;
        int64_t* kw_of = kh_of + KH;
        int64_t* ih_of = tap_vals.data() + slot * tap_cap;
        int64_t* sh_of = ih_of + KH;

        const T* in_n = in_ptr + n * in_n_stride;
        T* orow_base = out_ptr + n * out_n_stride + oh * out_row_stride;

        int64_t nt = 0;
        for (int64_t kh = 0; kh < KH; ++kh) {
            const int64_t num_h = oh + PH - kh * DH;
            if (num_h < 0 || num_h % SH != 0) {
                continue;
            }
            const int64_t ih = num_h / SH;
            if (ih >= IH) {
                continue;
            }
            kh_of[nt] = kh;
            ih_of[nt] = ih;
            ++nt;
        }

        for (int64_t g = 0; g < G; ++g) {
            const T* wg = wt.data() + g * wt_g_stride;
            const T* bg = b_ptr ? b_ptr + g * OC_span : nullptr;
            const T* in_g = in_n + g * IC_per_G;

            // An output column belongs to residue class ow % SW, and inside a
            // class the taps are the same for every column — only the validity
            // window moves, so it becomes one [t_lo, t_hi) interval.
            for (int64_t r = 0; r < SW && r < OW; ++r) {
                const int64_t mic_r = (OW - r + SW - 1) / SW;

                int64_t nk = 0;
                int64_t t_lo = 0;
                int64_t t_hi = mic_r;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    // Only divisibility is a class fact. A negative offset is
                    // not: pixel t of the class sees t·SW more, so the tap
                    // starts to apply at t = -sh, which is what t_lo is.
                    const int64_t num_w = r + PW - kw * DW;
                    if (num_w % SW != 0) {
                        continue;
                    }
                    const int64_t sh = num_w / SW;
                    kw_of[nk] = kw;
                    sh_of[nk] = sh;
                    ++nk;
                    t_lo = std::max<int64_t>(t_lo, -sh);
                    t_hi = std::min<int64_t>(t_hi, IW - sh);
                }

                // One output pixel, channels [c_begin, OC_span). The edge pixels
                // and the oc tail come through here, where nothing is reusable.
                const auto pixel = [&](int64_t t, int64_t c_begin) {
                    T* dst = orow_base + (r + t * SW) * OC + g * OC_span;
                    int64_t c0 = c_begin;
                    for (; c0 + L <= OC_span; c0 += L) {
                        vec_for<T> acc = v_zero(type_tag);

                        for (int64_t a = 0; a < nt; ++a) {
                            const T* row = in_g + ih_of[a] * in_row_stride;
                            for (int64_t k = 0; k < nk; ++k) {
                                const int64_t iw = t + sh_of[k];
                                if (iw < 0 || iw >= IW) {
                                    continue;
                                }
                                const T* pix = row + iw * IC;
                                const T* wrow =
                                    wg + ((kh_of[a] * KW + kw_of[k]) * IC_per_G * OC_span + c0);
                                for (int64_t ic = 0; ic < IC_per_G; ++ic) {
                                    acc = v_fmadd(v_set1(type_tag, s_load(&pix[ic])),
                                                  v_load(wrow + ic * OC_span), acc);
                                }
                            }
                        }

                        acc = v_add(acc, bg ? v_load(bg + c0) : v_zero(type_tag));
                        acc = apply_epilogue_vec_dst<T>(epilogue, type_tag, acc, dst + c0,
                                                        gm.add_to, gm.pre_act);
                        v_store(dst + c0, acc);
                    }

                    for (; c0 < OC_span; ++c0) {
                        float acc = 0.0f;

                        for (int64_t a = 0; a < nt; ++a) {
                            const T* row = in_g + ih_of[a] * in_row_stride;
                            for (int64_t k = 0; k < nk; ++k) {
                                const int64_t iw = t + sh_of[k];
                                if (iw < 0 || iw >= IW) {
                                    continue;
                                }
                                const T* pix = row + iw * IC;
                                for (int64_t ic = 0; ic < IC_per_G; ++ic) {
                                    acc += s_load(&pix[ic]) *
                                           s_load(&wg[wt_index(kh_of[a], kw_of[k], ic, c0)]);
                                }
                            }
                        }

                        if (bg) {
                            acc += s_load(&bg[c0]);
                        }
                        apply_epilogue_store<T>(&dst[c0], acc, epilogue, g * OC_span + c0,
                                                gm.add_to, gm.pre_act);
                    }
                };

                // Four pixels of one class against two oc vectors: the weight
                // vector is then loaded once per four pixels and the broadcast
                // once per pixel, where the pixel loop pays both per pixel.
                const auto tile = [&](int64_t t0) {
                    T* d0 = orow_base + (r + t0 * SW) * OC + g * OC_span;
                    int64_t c0 = 0;
                    for (; c0 + 2 * L <= OC_span; c0 += 2 * L) {
                        vec_for<T> a0[kPix];
                        vec_for<T> a1[kPix];
                        for (int64_t j = 0; j < kPix; ++j) {
                            a0[j] = v_zero(type_tag);
                            a1[j] = v_zero(type_tag);
                        }

                        for (int64_t a = 0; a < nt; ++a) {
                            const T* row = in_g + ih_of[a] * in_row_stride;
                            for (int64_t k = 0; k < nk; ++k) {
                                const T* pix = row + (t0 + sh_of[k]) * IC;
                                const T* wrow =
                                    wg + ((kh_of[a] * KW + kw_of[k]) * IC_per_G * OC_span + c0);
                                for (int64_t ic = 0; ic < IC_per_G; ++ic) {
                                    const auto w0 = v_load(wrow + ic * OC_span);
                                    const auto w1 = v_load(wrow + ic * OC_span + L);
                                    for (int64_t j = 0; j < kPix; ++j) {
                                        const auto x = v_set1(type_tag, s_load(pix + j * IC + ic));
                                        a0[j] = v_fmadd(x, w0, a0[j]);
                                        a1[j] = v_fmadd(x, w1, a1[j]);
                                    }
                                }
                            }
                        }

                        const auto b0 = bg ? v_load(bg + c0) : v_zero(type_tag);
                        const auto b1 = bg ? v_load(bg + c0 + L) : v_zero(type_tag);
                        for (int64_t j = 0; j < kPix; ++j) {
                            T* dst = d0 + j * SW * OC;
                            vec_for<T> s0 = v_add(a0[j], b0);
                            vec_for<T> s1 = v_add(a1[j], b1);
                            s0 = apply_epilogue_vec_dst<T>(epilogue, type_tag, s0, dst + c0,
                                                           gm.add_to, gm.pre_act);
                            s1 = apply_epilogue_vec_dst<T>(epilogue, type_tag, s1, dst + c0 + L,
                                                           gm.add_to, gm.pre_act);
                            v_store(dst + c0, s0);
                            v_store(dst + c0 + L, s1);
                        }
                    }
                    if (c0 < OC_span) {
                        for (int64_t j = 0; j < kPix; ++j) {
                            pixel(t0 + j, c0);
                        }
                    }
                };

                // The edge pixels first, then tiles aligned to the valid window
                // — not to zero. Aligning to zero would throw the whole tile
                // away whenever the window does not start on a multiple of
                // kPix, which on a short row is the difference between reusing
                // the weights and not.
                int64_t t0 = 0;
                for (; t0 < t_lo && t0 < mic_r; ++t0) {
                    pixel(t0, 0);
                }
                for (; t0 + kPix <= t_hi; t0 += kPix) {
                    tile(t0);
                }
                for (; t0 < mic_r; ++t0) {
                    pixel(t0, 0);
                }
            }
        }
    };

    ctx.cpu.run(0, N * OH, process);
}

} // anonymous namespace

Status transpose_conv2d_cpu(const TransposeConv2DAttributes& attrs, TensorView& output,
                            std::span<const TensorView> inputs, const ComputeContext& ctx,
                            void* /*workspace*/)
{
    const auto& input = inputs[0];
    const auto& weight = inputs[1];

    ConvChannels gm{};
    fill_conv_axes<2>(gm, attrs, input, output);
    gm.N = input.shape(0);
    // Channel-last [N, IH, IW, IC]; the filter is [IC, OC/G, KH, KW].
    gm.IC = weight.shape(0);
    gm.OC_per_G = weight.shape(1);
    gm.G = attrs.groups;
    gm.OC = gm.OC_per_G * gm.G;
    gm.IC_per_G = gm.IC / gm.G;

    if (is_quantized_transpose_conv2d(weight.data_type())) {
        return transpose_conv2d_int8_cpu(attrs, gm, output, inputs, ctx);
    }

    if (!dispatch_f32_f16(input.data_type(), "transpose_conv2d_cpu", [&](auto tag) {
            using T = typename decltype(tag)::type;
            tconv2d_impl<T>(gm, output, inputs, ctx);
        })) {
        return Status::unsupported;
    }
    return Status::ok;
}

} // namespace nnops::backend::cpu
