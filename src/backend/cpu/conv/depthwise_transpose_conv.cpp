/// @file depthwise_transpose_conv.cpp
/// @brief SIMD CPU depthwise transposed 2D convolution on NHWC.
///
/// Gather form, the exact inverse of depthwise conv's indexing: output (oh, ow)
/// reads input (ih, iw) where oh + PH - kh·DH == ih·SH, so a tap contributes
/// only to one residue of the output coordinate and every output pixel is
/// accumulated once — bias and epilogue included — without the zero-fill and
/// save/restore an output-side scatter would need for add_to.
///
/// Two per-class facts are hoisted out of the pixel loop: the (kh, ih) pairs
/// that land on an output row, and the (kw, shift) pairs that land on one
/// output-column residue, where a residue step t reads input column shift + t.
/// A row therefore splits into min(SW, OW) residue classes, and inside each one
/// the valid window [t_lo, t_hi) — every tap in range — runs free of bounds
/// tests, with the two edges left to the scalar tap loop.
///
/// The lane spans channels and the HWCM filter is read directly. Four residue
/// steps share one weight vector per tap, and 16 channels (two vectors) are
/// swept per pass so a pixel's 64-byte line is touched once.
///
/// A kernel taller than the tap-list cap falls back to the plain scalar
/// gather, which is correct for any shape.

#include "depthwise_transpose_conv_internal.h"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"
#include "backend/cpu/common/memory_pool.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {


// ============================================================
// Scalar gather — the fallback for kernels past the tap-list cap
// ============================================================

template <typename T>
void dwtconv_scalar(const DwtConvGeom& gm, TensorView& output,
                         std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const T* in_ptr = input.ptr<T>();
    const T* w_ptr = weight.ptr<T>();
    const T* b_ptr = has_bias ? inputs[2].ptr<T>() : nullptr;
    T* out_ptr = output.ptr<T>();

    const int64_t in_n = input.stride_elems(0);
    const int64_t in_row = input.row_stride_elems();
    const int64_t out_n = output.stride_elems(0);
    const int64_t out_row = output.row_stride_elems();
    const int64_t w_kh = weight.stride_elems(0);
    const int64_t w_kw = weight.stride_elems(1);
    const int64_t C = gm.C;

    ctx.cpu.run(0, gm.N * gm.OH, [&](int64_t task) {
        const int64_t n = task / gm.OH;
        const int64_t oh = task % gm.OH;
        const T* in_np = in_ptr + n * in_n;
        T* orow = out_ptr + n * out_n + oh * out_row;

        for (int64_t ow = 0; ow < gm.OW; ++ow) {
            T* dst = orow + ow * C;
            for (int64_t c = 0; c < C; ++c) {
                float acc = 0.0f;
                for (int64_t kh = 0; kh < gm.KH; ++kh) {
                    const int64_t num_h = oh + gm.PH - kh * gm.DH;
                    if (num_h < 0 || num_h % gm.SH != 0) {
                        continue;
                    }
                    const int64_t ih = num_h / gm.SH;
                    if (ih >= gm.IH) {
                        continue;
                    }
                    const T* xrow = in_np + ih * in_row;
                    for (int64_t kw = 0; kw < gm.KW; ++kw) {
                        const int64_t num_w = ow + gm.PW - kw * gm.DW;
                        if (num_w < 0 || num_w % gm.SW != 0) {
                            continue;
                        }
                        const int64_t iw = num_w / gm.SW;
                        if (iw >= gm.IW) {
                            continue;
                        }
                        acc +=
                            s_load(&xrow[iw * C + c]) * s_load(&w_ptr[kh * w_kh + kw * w_kw + c]);
                    }
                }
                if (has_bias) {
                    acc += s_load(&b_ptr[c]);
                }
                apply_epilogue_store<T>(&dst[c], acc, gm.epilogue, c, gm.add_to, gm.pre_act);
            }
        }
    });
}


template <typename T>
void dwtconv_impl(const DwtConvGeom& gm, TensorView& output,
                          std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const T* in_ptr = input.ptr<T>();
    const T* w_ptr = weight.ptr<T>();
    const T* b_ptr = has_bias ? inputs[2].ptr<T>() : nullptr;
    T* out_ptr = output.ptr<T>();

    const int64_t in_n = input.stride_elems(0);
    const int64_t in_row = input.row_stride_elems();
    const int64_t out_n = output.stride_elems(0);
    const int64_t out_row = output.row_stride_elems();
    // The raw HWCM [KH][KW][C] filter: one tap is C contiguous channels.
    const int64_t w_kh = weight.stride_elems(0);
    const int64_t w_kw = weight.stride_elems(1);
    const int64_t C = gm.C;

    const T* tag = nullptr;
    constexpr int L = simd_lane_for<T>;
    // Sixteen channels a chunk: a pixel's 16 f32 is one 64-byte line, so the
    // sweep touches every line once.
    constexpr int64_t kChunk = 16;

    // The task grid is N×OH rows, and a row holds only OW pixels: split each
    // row's residue steps into strips when the grid cannot fill the pool.
    const int64_t rows = gm.N * gm.OH;
    const int64_t nt = std::max(1, ctx.cpu.thread_count());
    int64_t strips = 1;
    if (rows < nt) {
        const int64_t max_strips = std::max<int64_t>(1, (gm.OW + 3) / 4);
        strips = std::min<int64_t>((nt + rows - 1) / rows, max_strips);
        strips = std::max<int64_t>(strips, 1);
    }

    const int64_t nc = std::min(gm.SW, gm.OW);

    ctx.cpu.run(0, rows * strips, [&](int64_t task) {
        const int64_t st = task % strips;
        const int64_t row = task / strips;
        const int64_t oh = row % gm.OH;
        const int64_t n = row / gm.OH;

        const T* in_np = in_ptr + n * in_n;
        T* orow = out_ptr + n * out_n + oh * out_row;
        const TapList rt = row_taps(gm, oh);

        for (int64_t r = 0; r < nc; ++r) {
            const int64_t mic = (gm.OW - r + gm.SW - 1) / gm.SW;
            const int64_t per = (mic + strips - 1) / strips;
            const int64_t ts = std::min(mic, st * per);
            const int64_t te = std::min(mic, ts + per);
            if (ts >= te) {
                continue;
            }

            const TapList ct = col_taps(gm, r, mic);
            int64_t t_lo = 0;
            int64_t t_hi = mic;
            for (int64_t k = 0; k < ct.n; ++k) {
                t_lo = std::max(t_lo, -ct.b[k]);
                t_hi = std::min(t_hi, gm.IW - ct.b[k]);
            }
            if (t_lo > t_hi) {
                t_lo = t_hi;
            }

            const auto scalar_one = [&](int64_t t, int64_t c) {
                float acc = 0.0f;
                for (int64_t a = 0; a < rt.n; ++a) {
                    const T* xrow = in_np + rt.b[a] * in_row;
                    for (int64_t k = 0; k < ct.n; ++k) {
                        const int64_t iw = ct.b[k] + t;
                        if (iw < 0 || iw >= gm.IW) {
                            continue;
                        }
                        acc += s_load(&xrow[iw * C + c]) *
                               s_load(&w_ptr[rt.a[a] * w_kh + ct.a[k] * w_kw + c]);
                    }
                }
                if (has_bias) {
                    acc += s_load(&b_ptr[c]);
                }
                T* dst = orow + (r + t * gm.SW) * C + c;
                apply_epilogue_store<T>(dst, acc, gm.epilogue, c, gm.add_to, gm.pre_act);
            };

            const auto pixel = [&](int64_t t) {
                for (int64_t c0 = 0; c0 < C; c0 += kChunk) {
                    const int64_t cend = std::min(C, c0 + kChunk);
                    int64_t c = c0;
                    for (; c + L <= cend; c += L) {
                        vec_for<T> acc = v_zero(tag);
                        for (int64_t a = 0; a < rt.n; ++a) {
                            const T* xrow = in_np + rt.b[a] * in_row;
                            for (int64_t k = 0; k < ct.n; ++k) {
                                const int64_t iw = ct.b[k] + t;
                                if (iw < 0 || iw >= gm.IW) {
                                    continue;
                                }
                                acc = v_fmadd(v_load(xrow + iw * C + c),
                                              v_load(w_ptr + rt.a[a] * w_kh + ct.a[k] * w_kw + c),
                                              acc);
                            }
                        }
                        if (has_bias) {
                            acc = v_add(acc, v_load(b_ptr + c));
                        }
                        T* dst = orow + (r + t * gm.SW) * C + c;
                        acc = apply_epilogue_vec_dst<T>(gm.epilogue, tag, acc, dst, gm.add_to,
                                                        gm.pre_act);
                        v_store(dst, acc);
                    }
                    for (; c < cend; ++c) {
                        scalar_one(t, c);
                    }
                }
            };

            // Four residue steps share each tap's weight vector; the window is
            // tap-complete, so the tile takes no bounds test.
            const auto tile4 = [&](int64_t t) {
                for (int64_t c0 = 0; c0 < C; c0 += kChunk) {
                    const int64_t cend = std::min(C, c0 + kChunk);
                    int64_t c = c0;
                    for (; c + L <= cend; c += L) {
                        vec_for<T> a0 = v_zero(tag);
                        vec_for<T> a1 = v_zero(tag);
                        vec_for<T> a2 = v_zero(tag);
                        vec_for<T> a3 = v_zero(tag);
                        for (int64_t a = 0; a < rt.n; ++a) {
                            const T* xrow = in_np + rt.b[a] * in_row + c;
                            for (int64_t k = 0; k < ct.n; ++k) {
                                const vec_for<T> wv =
                                    v_load(w_ptr + rt.a[a] * w_kh + ct.a[k] * w_kw + c);
                                const T* src = xrow + (ct.b[k] + t) * C;
                                a0 = v_fmadd(v_load(src), wv, a0);
                                a1 = v_fmadd(v_load(src + C), wv, a1);
                                a2 = v_fmadd(v_load(src + 2 * C), wv, a2);
                                a3 = v_fmadd(v_load(src + 3 * C), wv, a3);
                            }
                        }
                        if (has_bias) {
                            const vec_for<T> bv = v_load(b_ptr + c);
                            a0 = v_add(a0, bv);
                            a1 = v_add(a1, bv);
                            a2 = v_add(a2, bv);
                            a3 = v_add(a3, bv);
                        }
                        T* d0 = orow + (r + t * gm.SW) * C + c;
                        T* d1 = d0 + gm.SW * C;
                        T* d2 = d1 + gm.SW * C;
                        T* d3 = d2 + gm.SW * C;
                        a0 = apply_epilogue_vec_dst<T>(gm.epilogue, tag, a0, d0, gm.add_to,
                                                       gm.pre_act);
                        a1 = apply_epilogue_vec_dst<T>(gm.epilogue, tag, a1, d1, gm.add_to,
                                                       gm.pre_act);
                        a2 = apply_epilogue_vec_dst<T>(gm.epilogue, tag, a2, d2, gm.add_to,
                                                       gm.pre_act);
                        a3 = apply_epilogue_vec_dst<T>(gm.epilogue, tag, a3, d3, gm.add_to,
                                                       gm.pre_act);
                        v_store(d0, a0);
                        v_store(d1, a1);
                        v_store(d2, a2);
                        v_store(d3, a3);
                    }
                    for (; c < cend; ++c) {
                        for (int64_t j = 0; j < 4; ++j) {
                            scalar_one(t + j, c);
                        }
                    }
                }
            };

            const int64_t ilo = std::max(t_lo, ts);
            const int64_t ihi = std::min(t_hi, te);
            int64_t t = ts;
            for (; t < ilo; ++t) {
                pixel(t);
            }
            for (; t + 4 <= ihi; t += 4) {
                tile4(t);
            }
            for (; t < ihi; ++t) {
                pixel(t);
            }
            for (; t < te; ++t) {
                pixel(t);
            }
        }
    });
}

/// The fp kernels' choice: the per-tap form for the channel lane, the scalar
/// form when the filter outruns the unrolled tap count.
template <class T>
void dwtconv_fp(const DwtConvGeom& gm, TensorView& output, std::span<const TensorView> inputs,
                const ComputeContext& ctx)
{
    if (gm.KH > kMaxTaps || gm.KW > kMaxTaps) {
        dwtconv_scalar<T>(gm, output, inputs, ctx);
    } else {
        dwtconv_impl<T>(gm, output, inputs, ctx);
    }
}

} // anonymous namespace

// ============================================================
// Entry point with dtype and layout dispatch
// ============================================================

Status depthwise_transpose_conv_cpu(const DepthwiseTransposeConv2DAttributes& attrs,
                                    TensorView& output, std::span<const TensorView> inputs,
                                    const ComputeContext& ctx, void* /*workspace*/)
{
    const auto& input = inputs[0];
    const auto& weight = inputs[1];

    DwtConvGeom gm{};
    gm.N = input.shape(0);
    gm.C = input.shape(3);
    gm.IH = input.shape(1);
    gm.IW = input.shape(2);
    gm.OH = output.shape(1);
    gm.OW = output.shape(2);
    gm.KH = attrs.kernel_size[0];
    gm.KW = attrs.kernel_size[1];
    gm.SH = attrs.stride[0];
    gm.SW = attrs.stride[1];
    gm.DH = attrs.dilation[0];
    gm.DW = attrs.dilation[1];
    gm.PH = attrs.padding[0];
    gm.PW = attrs.padding[1];
    gm.epilogue = attrs.epilogue;
    gm.add_to = attrs.add_to;
    gm.pre_act = attrs.add_to && attrs.residual_pre_activation;

    if (!is_channel_last(input.layout())) {
        return Status::unsupported;
    }

    // The per-tap form is what the channel lane reads; a filter in the
    // [C, 1, KH, KW] order — what a grouped TransposeConv2D hands
    // over — has its channels a plane apart, so it is repacked once here rather
    // than read strided through every tap.
    PoolArray<std::byte> w_per_tap;
    TensorView w_hwcm;
    std::array<TensorView, 3> operands;
    if (weight.rank() == 4) {
        if (weight.shape(0) != gm.C || weight.shape(1) != 1) {
            return Status::unsupported; // not one filter per channel
        }
        const int64_t taps = gm.KH * gm.KW;
        const size_t es = data_type_size(weight.data_type());
        w_per_tap.reset(static_cast<size_t>(taps * gm.C) * es);
        if (!w_per_tap) {
            return Status::unsupported;
        }

        const auto* src = weight.ptr<std::byte>();
        auto* dst = w_per_tap.data();
        const int64_t s_c = weight.stride_elems(0);
        const int64_t s_kh = weight.stride_elems(2);
        const int64_t s_kw = weight.stride_elems(3);
        for (int64_t c = 0; c < gm.C; ++c) {
            for (int64_t kh = 0; kh < gm.KH; ++kh) {
                for (int64_t kw = 0; kw < gm.KW; ++kw) {
                    const size_t from = static_cast<size_t>((c * s_c + kh * s_kh + kw * s_kw)) * es;
                    const size_t to = static_cast<size_t>(((kh * gm.KW + kw) * gm.C + c)) * es;
                    std::memcpy(dst + to, src + from, es);
                }
            }
        }
        const int64_t shape[] = {gm.KH, gm.KW, gm.C};
        w_hwcm = TensorView(shape, weight.data_type(), dst, TensorLayout::HWCM);

        operands[0] = inputs[0];
        operands[1] = w_hwcm;
        if (inputs.size() > 2) {
            operands[2] = inputs[2];
        }
        inputs = std::span<const TensorView>(operands.data(), inputs.size());
    }

    if (is_quantized_depthwise_transpose_conv2d(weight.data_type())) {
        return depthwise_transpose_conv_int8_cpu(attrs, gm, output, inputs, ctx);
    }
    if (input.data_type() != weight.data_type() || output.data_type() != input.data_type()) {
        return Status::unsupported;
    }

    switch (input.data_type()) {
    case DataType::f32:
        dwtconv_fp<float>(gm, output, inputs, ctx);
        return Status::ok;
    case DataType::f16:
        dwtconv_fp<half>(gm, output, inputs, ctx);
        return Status::ok;
#ifdef NNOPS_ARCH_AARCH64
    case DataType::bf16:
        dwtconv_fp<bfloat16>(gm, output, inputs, ctx);
        return Status::ok;
#endif
    default:
        return Status::unsupported;
    }
}

} // namespace nnops::backend::cpu
