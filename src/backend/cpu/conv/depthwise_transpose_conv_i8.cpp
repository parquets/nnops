/// @file depthwise_transpose_conv_i8.cpp
/// @brief CPU depthwise transposed 2D convolution on NHWC — the quantized route.
///
/// s8 weight against a u8/s8/f16/f32 activation and an int32 accumulate, with a
/// per-tensor activation and a per-channel weight. The tap lists drive the walk
/// unchanged: the reduction is still over taps, so the same per-row (kh, ih) and
/// per-residue (kw, shift) classes apply and only the arithmetic inside the
/// window differs. Every term is compensated for both zero points, then the
/// merge dequantizes into fp or requantizes to s8/u8 — bias, clip and the
/// epilogue included, in the same order the fp merge uses.
///
/// add_to is refused: a residual on a quantized destination would have to
/// requantize the existing value.
///
/// The float routes and the entry point are depthwise_transpose_conv.cpp; the
/// geometry and tap lists both families read are
/// depthwise_transpose_conv_internal.h.

#include "depthwise_transpose_conv_internal.h"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"
#include "backend/cpu/simd_kernel/simd_int8_lane.hpp"
#include "backend/cpu/simd_kernel/simd_quant.hpp"
#include "backend/cpu/common/memory_pool.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

// ============================================================
// Quantized — s8 weight against a u8/s8/f16/f32 activation
// ============================================================
//
// The tap lists drive both int8 layouts unchanged: the reduction is still over
// taps, so the same per-row (kh, ih) and per-residue (kw, shift) classes apply
// and only the arithmetic inside the window differs. Every term is
// (w − zp_w)·(x − zp_x) accumulated in int32, the reference's own expression;
// integer sums are exact and order-independent over these magnitudes, so the
// vector lanes and the scalar edges cannot disagree on a bit.
//
// The activation is normalised to one s8 plane per call (dwtconv_i8_plane), so
// every storage type meets one zero-point convention. The weight's zero point
// folds into the product — a [C, 1, KH, KW] filter broadcasts (w − zp_w) as a
// scalar, the HWCM filter is staged as int16 lanes once per call.
// The activation's cannot: (x − zp_x) does not fit a byte, so it becomes the
// accumulator's seed, −zp_x·Σ(w − zp_w) over the taps live across the whole
// window, and the boundary pixels, whose tap set is filtered, take the explicit
// per-tap form instead.

/// The s8 plane the int8 kernels read: one byte per element, the activation's
/// zero point shifted to the s8 convention.
struct DwtI8Plane {
    const int8_t* ptr = nullptr;
    int64_t n_stride = 0;   ///< elements between batches
    int64_t row_stride = 0; ///< elements between rows of one plane
    int64_t ch_stride = 0;  ///< 0: channels are contiguous in the NHWC plane
    float scale = 1.0f;
    int32_t zero = 0;
};

/// Normalise one activation to a dense s8 plane. A pre-quantized s8 activation
/// with dense rows is read in place; anything else is copied, u8 through a bit
/// shift and fp through the arch quantizer at `zp − 128`, so the byte difference
/// against the shifted zero point is still the reference's (x − zp_x) — the
/// clamp windows map onto each other under the same −128.
template <class TAct>
Status dwtconv_i8_plane(const TensorView& in, bool act_u8, DwtI8Plane& pl,
                        PoolArray<int8_t>& buf, const ComputeContext& ctx)
{
    constexpr bool kFpAct = !std::is_same_v<TAct, int8_t> && !std::is_same_v<TAct, uint8_t>;

    const int64_t N = in.shape(0);
    const int64_t C = in.shape(3);
    const int64_t IH = in.shape(1);
    const int64_t IW = in.shape(2);
    const int64_t row_len = IW * C;

    pl.scale = quant_scale_at(in.quant_params(), 0);
    pl.zero = quant_zp_at(in.quant_params(), 0) - (act_u8 ? 128 : 0);

    const int64_t in_ns = in.stride_elems(0);
    const int64_t in_rs = in.row_stride_elems();

    if (!kFpAct && !act_u8 && in_rs == row_len) {
        pl.ptr = in.ptr<int8_t>();
        pl.n_stride = in_ns;
        pl.row_stride = in_rs;
        pl.ch_stride = 0;
        return Status::ok;
    }

    const int64_t rows = N * IH;
    const int64_t total = rows * row_len;
    if (!buf.reset(static_cast<size_t>(total))) {
        return Status::out_of_memory;
    }
    if (total <= 0) {
        return Status::ok;
    }

    pl.ptr = buf.data();
    pl.n_stride = IH * IW * C;
    pl.row_stride = row_len;
    pl.ch_stride = 0;

    const TAct* src = in.ptr<TAct>();
    int8_t* dst = buf.data();
    const float s = pl.scale;
    const float z = static_cast<float>(pl.zero);

    const auto run = [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            const TAct* sp = src + (r / IH) * in_ns + (r % IH) * in_rs;
            int8_t* dp = dst + r * row_len;
            if constexpr (kFpAct) {
                quant_kernel::quantization<int8_t>(1, static_cast<int>(row_len), dp, 0, sp, 0, &s,
                                                   &z);
            } else if constexpr (std::is_same_v<TAct, uint8_t>) {
                for (int64_t k = 0; k < row_len; ++k) {
                    dp[k] = static_cast<int8_t>(static_cast<uint8_t>(sp[k] ^ 0x80u));
                }
            } else {
                std::memcpy(dp, sp, static_cast<size_t>(row_len));
            }
        }
    };

    // The pass is memory-bound and embarrassingly parallel over rows; below a
    // MiB the pool's own dispatch costs more than it saves.
    constexpr int64_t kInlineRows = int64_t{1} << 12;
    const int64_t nt = std::max<int64_t>(1, ctx.cpu.thread_count());
    if (nt == 1 || rows <= kInlineRows) {
        run(0, rows);
        return Status::ok;
    }
    const int64_t slab = (rows + nt - 1) / nt;
    const int64_t nslab = (rows + slab - 1) / slab;
    ctx.cpu.run(0, nslab, [&](int64_t i) { run(i * slab, std::min(i * slab + slab, rows)); });
    return Status::ok;
}

/// (w − zp_w) as int16 lanes: [KH·KW][stride], the channels padded with zeros so
/// a 16-lane load never reaches into the next tap's storage.
Status dwtconv_i8_weight(const TensorView& weight, int64_t KH, int64_t KW, int64_t C,
                         PoolArray<int16_t>& buf, int64_t& stride)
{
    stride = (C + kI8Lane - 1) / kI8Lane * kI8Lane;
    if (C <= 0) {
        return Status::ok;
    }
    if (!buf.reset(static_cast<size_t>(KH * KW * stride))) {
        return Status::out_of_memory;
    }

    const int8_t* w_ptr = weight.ptr<int8_t>();
    const int64_t w_kh = weight.stride_elems(0);
    const int64_t w_kw = weight.stride_elems(1);
    const QuantParams& qw = weight.quant_params();

    for (int64_t kh = 0; kh < KH; ++kh) {
        for (int64_t kw = 0; kw < KW; ++kw) {
            int16_t* d = buf.data() + (kh * KW + kw) * stride;
            const int8_t* s = w_ptr + kh * w_kh + kw * w_kw;
            for (int64_t c = 0; c < C; ++c) {
                d[c] = static_cast<int16_t>(static_cast<int32_t>(s[c]) - quant_zp_at(qw, c));
            }
            for (int64_t c = C; c < stride; ++c) {
                d[c] = 0;
            }
        }
    }
    return Status::ok;
}


/// Spatial int8: the lane spans channels, one 16-channel block of one pixel
/// per accumulate step, the filter read from the staged int16 copy.
template <typename TOut>
Status dwtconv_i8(const DwtConvGeom& gm, TensorView& output,
                       std::span<const TensorView> inputs, const DwtI8Plane& pl, const int16_t* wq,
                       int64_t wq_c, const ComputeContext& ctx)
{
    constexpr bool kIntOut = std::is_same_v<TOut, int8_t> || std::is_same_v<TOut, uint8_t>;

    const int64_t C = gm.C;
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int32_t* b_i32 = (has_bias && kIntOut) ? inputs[2].ptr<int32_t>() : nullptr;
    const TOut* b_out = (has_bias && !kIntOut) ? inputs[2].ptr<TOut>() : nullptr;
    const QuantParams& qw = weight.quant_params();

    TOut* out_ptr = output.ptr<TOut>();
    const int64_t out_n = output.stride_elems(0);
    const int64_t out_row = output.row_stride_elems();

    const QuantParams& qo = output.quant_params();
    const float out_scale = kIntOut ? quant_scale_at(qo, 0) : 0.0f;
    const int32_t zp_out = kIntOut ? quant_zp_at(qo, 0) : 0;

    const float s_x = pl.scale;
    const int32_t zp_x = pl.zero;
    const int64_t in_row = pl.row_stride;
    const bool relu = gm.epilogue.type == EpilogueActivateType::Relu;

    PoolArray<float> req_arr;
    const float* req = nullptr;
    if constexpr (kIntOut) {
        if (!req_arr.reset(static_cast<size_t>(C))) {
            return Status::out_of_memory;
        }
        for (int64_t c = 0; c < C; ++c) {
            req_arr[static_cast<size_t>(c)] =
                matmul_int8_requant_scale(quant_scale_at(qw, c), s_x, out_scale);
        }
        req = req_arr.data();
    }

    const int64_t rows = gm.N * gm.OH;
    const int64_t nt = std::max(1, ctx.cpu.thread_count());
    int64_t strips = 1;
    if (rows < nt) {
        const int64_t max_strips = std::max<int64_t>(1, (gm.OW + 3) / 4);
        strips = std::max<int64_t>(1, std::min<int64_t>((nt + rows - 1) / rows, max_strips));
    }
    const int64_t nc = std::min(gm.SW, gm.OW);

    ctx.cpu.run(0, rows * strips, [&](int64_t task) {
        const int64_t st = task % strips;
        const int64_t row = task / strips;
        const int64_t oh = row % gm.OH;
        const int64_t n = row / gm.OH;

        const int8_t* in_n = pl.ptr + n * pl.n_stride;
        TOut* orow = out_ptr + n * out_n + oh * out_row;
        const TapList rt = row_taps(gm, oh);
        const int8_t* xrow[kMaxTaps];
        for (int64_t a = 0; a < rt.n; ++a) {
            xrow[a] = in_n + rt.b[a] * in_row;
        }

        alignas(32) int32_t lanes[kI8Lane];
        alignas(32) float swsx[kI8Lane];

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

            // One channel at a time, for the fewer-than-16 tail.
            const auto scalar_one = [&](int64_t t, int64_t c) {
                int32_t acc = 0;
                for (int64_t a = 0; a < rt.n; ++a) {
                    for (int64_t k = 0; k < ct.n; ++k) {
                        const int64_t iw = ct.b[k] + t;
                        if (iw < 0 || iw >= gm.IW) {
                            continue;
                        }
                        acc += static_cast<int32_t>(wq[(rt.a[a] * gm.KW + ct.a[k]) * wq_c + c]) *
                               (static_cast<int32_t>(xrow[a][iw * C + c]) - zp_x);
                    }
                }
                TOut* dst = orow + (r + t * gm.SW) * C + c;
                if constexpr (kIntOut) {
                    int32_t v = acc + (has_bias ? b_i32[c] : 0);
                    if (relu) {
                        v = std::max(v, 0);
                    }
                    *dst = matmul_int8_requant<TOut>(v, req[c], zp_out);
                } else {
                    float f = static_cast<float>(acc) * (quant_scale_at(qw, c) * s_x);
                    if (has_bias) {
                        f += s_load(&b_out[c]);
                    }
                    apply_epilogue_store<TOut>(dst, f, gm.epilogue, c,
                                               /*add_to=*/false, /*pre_act=*/false);
                }
            };

            int64_t c0 = 0;
            if (C >= kI8Lane) {
                c0 = (C / kI8Lane) * kI8Lane;
            }

            for (int64_t cb = 0; cb < c0; cb += kI8Lane) {
                for (int64_t j = 0; j < kI8Lane; ++j) {
                    swsx[j] = quant_scale_at(qw, cb + j) * s_x;
                }

                // The tap-sum seed, built once per block: Σ(w − zp_w) over the
                // taps the whole window keeps.
                I8LaneAcc base;
                if (zp_x != 0) {
                    I8LaneAcc sum;
                    sum.set1(0);
                    for (int64_t a = 0; a < rt.n; ++a) {
                        for (int64_t k = 0; k < ct.n; ++k) {
                            sum.add_v(wq + (rt.a[a] * gm.KW + ct.a[k]) * wq_c + cb);
                        }
                    }
                    base.set1(0);
                    base.sub_scaled(sum, zp_x);
                } else {
                    base.set1(0);
                }

                const auto emit_block = [&](int64_t t) {
                    TOut* dst = orow + (r + t * gm.SW) * C + cb;
                    if constexpr (kIntOut) {
                        Int8EpilogueRow p{};
                        p.s_a = 1.0;
                        p.out_scale = 1.0;
                        p.s_b = req + cb;
                        p.zp_out = zp_out;
                        p.relu = relu;
                        p.col_bias = has_bias ? b_i32 + cb : nullptr;
                        int8_epilogue_row_quant<TOut>(lanes, dst, 0, kI8Lane, p);
                    } else {
                        for (int64_t j = 0; j < kI8Lane; j += 8) {
                            v_f32x8 f = v_cvt_i32_to_f32(lanes + j);
                            f = v_mul(f, v_load(swsx + j));
                            if (has_bias) {
                                if constexpr (std::is_same_v<TOut, half>) {
                                    f = v_add(f, v_cvt_f16_to_f32(b_out + cb + j));
                                } else {
                                    f = v_add(f, v_load(b_out + cb + j));
                                }
                            }
                            f = apply_epilogue_vec<float>(gm.epilogue,
                                                          static_cast<const float*>(nullptr), f);
                            if constexpr (std::is_same_v<TOut, half>) {
                                v_store(dst + j, v_cvt_f32_to_f16(f));
                            } else {
                                v_store(dst + j, f);
                            }
                        }
                    }
                };

                const auto edge_pixel = [&](int64_t t) {
                    I8LaneAcc acc;
                    acc.set1(0);
                    I8LaneAcc sum;
                    if (zp_x != 0) {
                        sum.set1(0);
                    }
                    for (int64_t a = 0; a < rt.n; ++a) {
                        for (int64_t k = 0; k < ct.n; ++k) {
                            const int64_t iw = ct.b[k] + t;
                            if (iw < 0 || iw >= gm.IW) {
                                continue;
                            }
                            const int16_t* wv = wq + (rt.a[a] * gm.KW + ct.a[k]) * wq_c + cb;
                            acc.madd_v(xrow[a] + iw * C + cb, wv);
                            if (zp_x != 0) {
                                sum.add_v(wv);
                            }
                        }
                    }
                    if (zp_x != 0) {
                        acc.sub_scaled(sum, zp_x);
                    }
                    acc.store(lanes);
                    emit_block(t);
                };

                const int64_t ilo = std::min(std::max(t_lo, ts), te);
                const int64_t ihi = std::min(t_hi, te);
                int64_t t = ts;
                for (; t < ilo; ++t) {
                    edge_pixel(t);
                }
                for (; t < ihi; ++t) {
                    I8LaneAcc acc = base;
                    for (int64_t a = 0; a < rt.n; ++a) {
                        for (int64_t k = 0; k < ct.n; ++k) {
                            acc.madd_v(xrow[a] + (ct.b[k] + t) * C + cb,
                                       wq + (rt.a[a] * gm.KW + ct.a[k]) * wq_c + cb);
                        }
                    }
                    acc.store(lanes);
                    emit_block(t);
                }
                for (; t < te; ++t) {
                    edge_pixel(t);
                }
            }

            for (int64_t c = c0; c < C; ++c) {
                for (int64_t t = ts; t < te; ++t) {
                    scalar_one(t, c);
                }
            }
        }
    });
    return Status::ok;
}

template <typename TOut>
Status dwtconv_i8_layout(const DwtConvGeom& gm, TensorView& output,
                         std::span<const TensorView> inputs, const DwtI8Plane& pl,
                         const ComputeContext& ctx)
{
    PoolArray<int16_t> wq;
    int64_t wq_c = 0;
    const Status st = dwtconv_i8_weight(inputs[1], gm.KH, gm.KW, gm.C, wq, wq_c);
    if (!is_ok(st)) {
        return st;
    }
    return dwtconv_i8<TOut>(gm, output, inputs, pl, wq.data(), wq_c, ctx);
}

template <class TAct>
Status dwtconv_i8_act(const DwtConvGeom& gm, TensorView& output, std::span<const TensorView> inputs,
                      bool act_u8, const ComputeContext& ctx)
{
    DwtI8Plane pl;
    PoolArray<int8_t> plane;
    const Status st = dwtconv_i8_plane<TAct>(inputs[0], act_u8, pl, plane, ctx);
    if (!is_ok(st)) {
        return st;
    }

    switch (output.data_type()) {
    case DataType::f32:
        return dwtconv_i8_layout<float>(gm, output, inputs, pl, ctx);
    case DataType::f16:
        return dwtconv_i8_layout<half>(gm, output, inputs, pl, ctx);
    case DataType::s8:
        return dwtconv_i8_layout<int8_t>(gm, output, inputs, pl, ctx);
    default:
        return dwtconv_i8_layout<uint8_t>(gm, output, inputs, pl, ctx);
    }
}

} // anonymous namespace

/// The quantized gate: which activation/weight/output/bias combinations have an
/// arithmetic here. add_to is refused — a residual on a quantized destination
/// would have to requantize the existing value.
Status depthwise_transpose_conv_int8_cpu(const DepthwiseTransposeConv2DAttributes& attrs,
                                         const DwtConvGeom& gm, TensorView& output,
                                         std::span<const TensorView> inputs,
                                         const ComputeContext& ctx)
{
    const DataType adt = inputs[0].data_type();
    const DataType odt = output.data_type();
    const bool fp_act = adt == DataType::f32 || adt == DataType::f16;
    const bool int_out = odt == DataType::s8 || odt == DataType::u8;

    if (!fp_act && adt != DataType::s8 && adt != DataType::u8) {
        return Status::unsupported;
    }
    if (!int_out && odt != DataType::f32 && odt != DataType::f16) {
        return Status::unsupported;
    }
    if (!int_out && fp_act && odt != adt) {
        return Status::unsupported;
    }
    if (inputs.size() > 2 && inputs[2].data_type() != (int_out ? DataType::s32 : odt)) {
        return Status::unsupported;
    }
    if (gm.add_to) {
        return Status::unsupported;
    }

    if (gm.N == 0 || gm.C == 0 || gm.OH == 0 || gm.OW == 0) {
        return Status::ok;
    }

    // The plane's storage follows the reference: an fp activation is quantized
    // to act_quant_dtype, a pre-quantized one keeps its own.
    const bool act_u8 = fp_act ? (attrs.act_quant_dtype == DataType::u8) : (adt == DataType::u8);

    switch (adt) {
    case DataType::f32:
        return dwtconv_i8_act<float>(gm, output, inputs, act_u8, ctx);
    case DataType::f16:
        return dwtconv_i8_act<half>(gm, output, inputs, act_u8, ctx);
    case DataType::u8:
        return dwtconv_i8_act<uint8_t>(gm, output, inputs, act_u8, ctx);
    default:
        return dwtconv_i8_act<int8_t>(gm, output, inputs, act_u8, ctx);
    }
}

} // namespace nnops::backend::cpu
