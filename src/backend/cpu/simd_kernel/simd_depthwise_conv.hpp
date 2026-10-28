#pragma once
/// @file simd_depthwise_conv.hpp
/// @brief SIMD kernels for 2D/3D depthwise convolution, packed and planar.
///
/// Packed (NCHWC8/NCDHWC8) weights are prepacked into dense [C8, (KD,) KH, KW, 8]
/// form and the lane is a channel; planar (NCHW/NCDHW) input reads the raw
/// [C, 1, (KD,) KH, KW] filter and vectorizes along output width instead.
/// Rank is
/// auto-detected: 2D is KD=OD=ID=1, which degenerates the depth loop.
/// One driver covers h4/h1 and f32/f16/quant; padding is bounds-checked
/// in-kernel, and all C8 blocks run SIMD since pad channels are zero.

#include "nnops/detail/simd/simd.hpp"
#include "nnops/core/epilogue.hpp"
#include "simd_epilogue.hpp"
#include "simd_access.hpp"

#include <cstdint>

namespace nnops::kernel {

struct DwConvParams {
    int64_t ID, IH, IW;           // input spatial dims
    int64_t OH, OW;               // output spatial dims (for store bounds)
    int64_t od, oh;               // current output position (set before call)
    int64_t KD, KH, KW;           // kernel shape
    int64_t SD, SH, SW;           // stride
    int64_t DD, DH, DW;           // dilation
    int64_t PD, PH, PW;           // padding
    int64_t out_row_stride;       // output row stride (elems)
    int64_t in_d_stride;          // input depth stride (elems)
    int64_t in_row_stride;        // input row stride (elems)
};

/// R output rows per call. W is the packed-weight element type; Tag only picks
/// the epilogue's f32/f16 overload. Both are deduced by the wrappers below.
template <int R, typename W, typename Tag, typename Src, typename Dst, typename V>
inline void dwconv_rows(const DwConvParams& p, const W* w, const Tag* tag,
                        const Epilogue& epilogue, const Src& src, const Dst& dst,
                        const V& vbias)
{
    for (int64_t ow = 0; ow < p.OW; ++ow) {
        V acc[R];
        for (int r = 0; r < R; ++r) {
            acc[r] = vbias;
        }

        for (int64_t kd = 0; kd < p.KD; ++kd) {
            const int64_t id = p.od * p.SD + kd * p.DD - p.PD;
            if (id < 0 || id >= p.ID) {
                continue;
            }
            const int64_t d_off = id * p.in_d_stride;

            for (int64_t kh = 0; kh < p.KH; ++kh) {
                int64_t ih[R];
                bool ok[R];
                for (int r = 0; r < R; ++r) {
                    ih[r] = (p.oh + r) * p.SH + kh * p.DH - p.PH;
                    ok[r] = (p.oh + r < p.OH) && ih[r] >= 0 && ih[r] < p.IH;
                }

                for (int64_t kw = 0; kw < p.KW; ++kw) {
                    const int64_t iw = ow * p.SW + kw * p.DW - p.PW;
                    if (iw < 0 || iw >= p.IW) {
                        continue;
                    }
                    const V vk = v_load(w + (kd * p.KH * p.KW + kh * p.KW + kw) * 8);

                    for (int r = 0; r < R; ++r) {
                        if (!ok[r]) {
                            continue;
                        }
                        acc[r] = v_fmadd(src.load_at(d_off + ih[r] * p.in_row_stride + iw * 8),
                                         vk, acc[r]);
                    }
                }
            }
        }

        for (int r = 0; r < R; ++r) {
            if (p.oh + r >= p.OH) {
                continue;
            }
            dst.store(r, ow, nnops::backend::cpu::apply_epilogue_vec(epilogue, tag, acc[r]));
        }
    }
}

// ============================================================
// Planar (NCHW/NCDHW) driver — the SIMD lane spans output width
// ============================================================
//
// The weight is the raw per-channel filter [C, 1, (KD,) KH, KW]: one scalar
// per tap, broadcast into the accumulate, instead of the packed 8-lane form.
// Padding is zero-filled per lane (planar_load), and a row's last block stores
// through a scalar tail.
//
// R output rows per call, for the same reason the packed kernel takes four: a
// ready-made row-modulo-R loop with ONE accumulator is a chain of KD·KH·KW
// dependent FMAs — 9 of them for a 3x3, four cycles each — and the taps have
// nothing to interleave. Four rows give four independent chains, so the FMA
// latency hides behind itself. The rows do not share loads (each reads its own
// input row), so unlike the packed kernel this buys no bandwidth — only ILP.

/// R output rows per call, one width-vector accumulate per row. T is the
/// element type; the accumulate and the epilogue ride it.
template <int R, typename T>
inline void dwconv_planar_rows(T* out, const T* in, const T* w, float bias,
                               const DwConvParams& p, const Epilogue& epilogue,
                               bool add_to)
{
    constexpr int64_t L = simd_lane_for<T>;
    const T zero = static_cast<T>(0);

    for (int64_t ow = 0; ow < p.OW; ow += L) {
        decltype(v_set1(out, bias)) acc[R];
        for (int r = 0; r < R; ++r) { acc[r] = v_set1(out, bias); }

        // Does this whole width vector — every lane, every row, every tap —
        // land inside the input? Then the taps need no bounds test and no
        // scalar fill: one branch per vector instead of one per tap. That is
        // the interior nn_compute reaches by peeling the padded border off at
        // the driver; here the same test is cheap enough to do per vector, so
        // a caller pays nothing and the border keeps the general path.
        // v_load_even deinterleaves 2L elements to produce one vector, so a
        // stride-2 vector touches twice the width a stride-1 one does — the
        // same guard planar_load uses on its fast path.
        const int64_t iw_first = ow * p.SW - p.PW;
        const int64_t span     = (p.SW == 2) ? 2 * L : L;
        const int64_t iw_end   = iw_first + (p.KW - 1) * p.DW + span;
        bool interior = (p.SW == 1 || p.SW == 2) && iw_first >= 0 && iw_end <= p.IW;
        for (int r = 0; interior && r < R; ++r) {
            const int64_t ih0 = (p.oh + r) * p.SH - p.PH;
            interior = (p.oh + r < p.OH) && ih0 >= 0 &&
                       ih0 + (p.KH - 1) * p.DH < p.IH;
        }

        for (int64_t kd = 0; kd < p.KD; ++kd) {
            const int64_t id = p.od * p.SD + kd * p.DD - p.PD;
            if (id < 0 || id >= p.ID) { continue; }
            const T* in_d = in + id * p.in_d_stride;

            for (int64_t kh = 0; kh < p.KH; ++kh) {
                const T* row[R];
                bool ok[R];
                for (int r = 0; r < R; ++r) {
                    const int64_t ih = (p.oh + r) * p.SH + kh * p.DH - p.PH;
                    ok[r] = (p.oh + r < p.OH) && ih >= 0 && ih < p.IH;
                    row[r] = in_d + ih * p.in_row_stride;
                }
                const T* w_row = w + (kd * p.KH + kh) * p.KW;

                if (interior) {
                    for (int64_t kw = 0; kw < p.KW; ++kw) {
                        const int64_t iw0 = ow * p.SW + kw * p.DW - p.PW;
                        const auto vk = v_set1(out, s_load(&w_row[kw]));
                        for (int r = 0; r < R; ++r) {
                            const T* src = row[r] + iw0;
                            acc[r] = v_fmadd(
                                (p.SW == 1) ? v_load(src) : v_load_even(src), vk, acc[r]);
                        }
                    }
                    continue;
                }

                for (int64_t kw = 0; kw < p.KW; ++kw) {
                    const int64_t iw0 = ow * p.SW + kw * p.DW - p.PW;
                    const auto vk = v_set1(out, s_load(&w_row[kw]));
                    for (int r = 0; r < R; ++r) {
                        if (!ok[r]) { continue; }
                        acc[r] = v_fmadd(planar_load(row[r], iw0, p.SW, p.IW, zero),
                                         vk, acc[r]);
                    }
                }
            }
        }

        const int64_t lanes = (p.OW - ow < L) ? p.OW - ow : L;
        for (int r = 0; r < R; ++r) {
            if (p.oh + r >= p.OH) { continue; }
            const auto v = nnops::backend::cpu::apply_epilogue_vec(epilogue, out, acc[r]);
            T* dst = out + r * p.out_row_stride + ow;
            if (lanes == L) {
                v_store_add(dst, v, add_to);
            } else {
                alignas(32) T tmp[L];
                v_store(tmp, v);
                for (int64_t l = 0; l < lanes; ++l) {
                    s_store_add(dst + l, s_load(&tmp[l]), add_to);
                }
            }
        }
    }
}

// ============================================================
// Public kernels — unchanged signatures
// ============================================================

template <typename T>
inline void dwconv_planar_h4(T* out, const T* in, const T* w, float bias,
                             const DwConvParams& p, const Epilogue& epilogue,
                             bool add_to)
{
    dwconv_planar_rows<4>(out, in, w, bias, p, epilogue, add_to);
}

/// One row — the tail of an h4 loop, and the whole kernel for a single-row call.
template <typename T>
inline void dwconv_planar(T* out, const T* in, const T* w, float bias,
                          const DwConvParams& p, const Epilogue& epilogue,
                          bool add_to)
{
    dwconv_planar_rows<1>(out, in, w, bias, p, epilogue, add_to);
}

template <typename T>
inline void dwconv_h4(T* output, const T* input, const T* w_base,
                      const T* bias_vec, const DwConvParams& p,
                      const Epilogue& epilogue, bool add_to)
{
    const auto vbias = bias_vec ? v_load(bias_vec) : v_zero(output);
    dwconv_rows<4>(p, w_base, output, epilogue,
                   FpSrc<T>{input}, FpDst<T>{output, p.out_row_stride, add_to}, vbias);
}

template <typename T>
inline void dwconv_h1(T* output, const T* input, const T* w_base,
                      const T* bias_vec, const DwConvParams& p,
                      const Epilogue& epilogue, bool add_to)
{
    const auto vbias = bias_vec ? v_load(bias_vec) : v_zero(output);
    dwconv_rows<1>(p, w_base, output, epilogue,
                   FpSrc<T>{input}, FpDst<T>{output, p.out_row_stride, add_to}, vbias);
}

// Quantized: the weights arrive already dequantized (f32, per-channel, with pad
// lanes neutralized by scale 0), so the accumulate is the same f32 FMA.

template <typename InT, typename OutT>
inline void dwconv_h4_quant(OutT* output, const InT* input, const float* wq,
                            const float* bias_vec, const DwConvParams& p,
                            const Epilogue& epilogue, bool add_to,
                            float in_scale, float in_zero,
                            float out_scale, float out_zero)
{
    const auto vbias = bias_vec ? v_load(bias_vec) : v_zero_f32x8();
    dwconv_rows<4>(p, wq, &out_scale, epilogue,
                   QuantSrc<InT>{input, in_scale, in_zero},
                   QuantDst<OutT>{output, p.out_row_stride, out_scale, out_zero, add_to},
                   vbias);
}

template <typename InT, typename OutT>
inline void dwconv_h1_quant(OutT* output, const InT* input, const float* wq,
                            const float* bias_vec, const DwConvParams& p,
                            const Epilogue& epilogue, bool add_to,
                            float in_scale, float in_zero,
                            float out_scale, float out_zero)
{
    const auto vbias = bias_vec ? v_load(bias_vec) : v_zero_f32x8();
    dwconv_rows<1>(p, wq, &out_scale, epilogue,
                   QuantSrc<InT>{input, in_scale, in_zero},
                   QuantDst<OutT>{output, p.out_row_stride, out_scale, out_zero, add_to},
                   vbias);
}

}  // namespace nnops::kernel
