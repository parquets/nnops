#pragma once
/// @file simd_pooling.hpp
/// @brief SIMD kernels for 2D/3D spatial pooling.
///
/// Two driver families, one per layout family:
///   packed  — the lane is a channel inside a C8 block, so the loop is over ow.
///             Every C8 block (partial ones included) runs the SIMD path, since
///             NCHWC8 pads channels with zeros (max ignores them, avg adds 0).
///   planar  — NCHW/NCDHW: the lane is an output column, so padding is filled
///             per lane instead of skipped, and one row is produced per call.
/// One driver covers every variant: the row count (h4/h1), the accumulate
/// (max/avg), and the element type (f32/f16 vs dequantize/requantize) are all
/// template parameters. Bounds are checked in-kernel, so any stride/padding
/// combination works. Quantized pooling is packed-only.

#include "nnops/detail/simd/simd.hpp"
#include "simd_access.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace nnops::kernel {

using namespace simd;

// ============================================================
// Kernel parameters — set once, od/oh updated per call
// ============================================================

struct PoolingKernelParams {
    int64_t ID, IH, IW;          // input spatial dims
    int64_t OH, OW;              // output spatial dims (for store bounds)
    int64_t od, oh;              // current output position (set before call)
    int64_t KD, KH, KW;          // kernel shape
    int64_t SD, SH, SW;          // stride
    int64_t DD, DH, DW;          // dilation
    int64_t PD, PH, PW;          // padding
    int64_t out_row_stride;      // output row stride (elems)
    int64_t in_d_stride;         // input depth stride (elems)
    int64_t in_row_stride;       // input row stride (elems)
    bool exclude_pad = false;    // AvgPool: divide by valid count instead of K_total
};

// ============================================================
// The driver: R output rows at a time, IsMax picks the accumulate
// ============================================================

template <int R, bool IsMax, typename Src, typename Dst>
inline void pool_rows(const PoolingKernelParams& p, float avg_scale,
                      const Src& src, const Dst& dst)
{
    using V = decltype(src.zero());
    const V vzero = src.zero();

    V vinit;
    if constexpr (IsMax) {
        vinit = src.set1(-std::numeric_limits<float>::infinity());
    } else {
        vinit = vzero;
    }

    for (int64_t ow = 0; ow < p.OW; ++ow) {
        V acc[R];
        int64_t cnt[R] = {};
        for (int r = 0; r < R; ++r) {
            acc[r] = vinit;
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
                    for (int r = 0; r < R; ++r) {
                        if (!ok[r]) {
                            continue;
                        }
                        const V v = src.load_at(d_off + ih[r] * p.in_row_stride + iw * 8);
                        acc[r] = IsMax ? v_max(acc[r], v) : v_add(acc[r], v);
                        ++cnt[r];
                    }
                }
            }
        }

        for (int r = 0; r < R; ++r) {
            if (p.oh + r >= p.OH) {
                continue;
            }
            if constexpr (IsMax) {
                if (cnt[r] == 0) {
                    acc[r] = vzero;
                }
            } else if (p.exclude_pad) {
                if (cnt[r] > 0) {
                    acc[r] = v_mul(acc[r], src.set1(1.0f / static_cast<float>(cnt[r])));
                }
            } else {
                acc[r] = v_mul(acc[r], src.set1(avg_scale));
            }
            dst.store(r, ow, acc[r]);
        }
    }
}

// ============================================================
// Planar (NCHW/NCDHW) driver — the SIMD lane spans output width
// ============================================================
//
// Same window walk as pool_rows, with two differences. The lane is an output
// column, so a tap outside the input cannot be left out — a padded column
// would read whatever the pitch holds — and is filled with the accumulate's
// identity instead (planar_load). And a row's last block carries fewer than L
// live lanes, so it stores through a scalar tail. One output row per call: the
// packed path's h4 blocking shares input rows across output rows, which is not
// where a width-vectorized kernel spends its time.

template <bool IsMax, typename T>
inline void pool_planar(T* out, const T* in, const PoolingKernelParams& p,
                        float avg_scale, bool add_to)
{
    constexpr int64_t L = simd_lane_for<T>;
    const T fill = IsMax ? std::numeric_limits<T>::lowest() : static_cast<T>(0);

    // Row-constant half of the tap count: the in-range (kd, kh) positions, each
    // spanning the full W window.
    int64_t row_taps = 0;
    for (int64_t kd = 0; kd < p.KD; ++kd) {
        const int64_t id = p.od * p.SD + kd * p.DD - p.PD;
        if (id < 0 || id >= p.ID) { continue; }
        for (int64_t kh = 0; kh < p.KH; ++kh) {
            const int64_t ih = p.oh * p.SH + kh * p.DH - p.PH;
            if (ih >= 0 && ih < p.IH) { ++row_taps; }
        }
    }

    // In-range taps of one output column. W padding (the only dim that can trim
    // a window per column) forces a per-lane count; otherwise it is a constant.
    const auto taps_of = [&](int64_t col) {
        if (p.PW == 0) { return row_taps * p.KW; }
        const int64_t iw_first = col * p.SW - p.PW;
        int64_t n = 0;
        for (int64_t kw = 0; kw < p.KW; ++kw) {
            const int64_t iw = iw_first + kw * p.DW;
            n += (iw >= 0 && iw < p.IW) ? 1 : 0;
        }
        return row_taps * n;
    };

    // Max: an all-padding window is 0 (the packed kernel and the reference both
    // say so). Avg: exclude_pad divides by the tap count. With no padding the
    // tap count is uniform and this is a single scale.
    const bool count_taps = IsMax ? (p.PD != 0 || p.PH != 0 || p.PW != 0)
                                  : p.exclude_pad;

    for (int64_t ow = 0; ow < p.OW; ow += L) {
        auto acc = IsMax ? v_set1(out, -std::numeric_limits<float>::infinity())
                         : v_zero(out);

        for (int64_t kd = 0; kd < p.KD; ++kd) {
            const int64_t id = p.od * p.SD + kd * p.DD - p.PD;
            if (id < 0 || id >= p.ID) { continue; }

            for (int64_t kh = 0; kh < p.KH; ++kh) {
                const int64_t ih = p.oh * p.SH + kh * p.DH - p.PH;
                if (ih < 0 || ih >= p.IH) { continue; }
                const T* row = in + id * p.in_d_stride + ih * p.in_row_stride;

                for (int64_t kw = 0; kw < p.KW; ++kw) {
                    const int64_t iw0 = ow * p.SW + kw * p.DW - p.PW;
                    const auto v = planar_load(row, iw0, p.SW, p.IW, fill);
                    acc = IsMax ? v_max(acc, v) : v_add(acc, v);
                }
            }
        }

        const int64_t lanes = (p.OW - ow < L) ? p.OW - ow : L;
        if (count_taps) {
            alignas(32) T scale[L];
            for (int64_t l = 0; l < L; ++l) {
                const int64_t cnt = (l < lanes) ? taps_of(ow + l) : 1;
                const float s = IsMax ? (cnt > 0 ? 1.0f : 0.0f)
                                      : (cnt > 0 ? 1.0f / static_cast<float>(cnt) : 0.0f);
                s_store(&scale[l], s);
            }
            acc = v_mul(acc, v_load(scale));
        } else if constexpr (!IsMax) {
            acc = v_mul(acc, v_set1(out, avg_scale));
        }

        if (lanes == L) {
            v_store_add(out + ow, acc, add_to);
        } else {
            alignas(32) T tmp[L];
            v_store(tmp, acc);
            for (int64_t l = 0; l < lanes; ++l) {
                s_store_add(out + ow + l, s_load(&tmp[l]), add_to);
            }
        }
    }
}

// ============================================================
// Global pooling — the window is the whole extent, so the reduction is a scan
// of contiguous memory instead of a window walk
// ============================================================
//
// One output element (planar) or one C8 block (packed) per channel plane, and
// no bounds math at all. The plane is walked row by row (the pitch can exceed
// the row width) with four accumulators, so the sum is off the add's latency
// chain, and a scalar tail, since neither a row width nor a plane size is a
// multiple of 8. f16 widens to f32 on load: a global sum spans thousands of
// values, far past what an f16 accumulator holds.

template <bool IsMax>
inline v_f32x8 global_op(const v_f32x8& a, const v_f32x8& b)
{
    if constexpr (IsMax) { return v_max(a, b); }
    return v_add(a, b);
}

template <bool IsMax>
inline float global_op(float a, float b)
{
    return IsMax ? std::max(a, b) : a + b;
}

template <typename T>
inline v_f32x8 global_load(const T* p)
{
    if constexpr (std::is_same_v<T, half>) {
        return v_cvt_f16_to_f32(p);
    } else {
        return v_load(p);
    }
}

/// One reduced vector → one scalar (max or sum, matching IsMax).
inline float global_fold(const v_f32x8& v, bool is_max)
{
    alignas(32) float lanes[8];
    v_store(lanes, v);
    float r = lanes[0];
    for (int i = 1; i < 8; ++i) {
        r = is_max ? std::max(r, lanes[i]) : r + lanes[i];
    }
    return r;
}

/// Fold @p n contiguous elements into the four accumulators (and the scalar
/// tail) — the body both layouts share, called once per plane or once per row.
template <bool IsMax, typename T>
inline void global_accumulate(v_f32x8& acc0, v_f32x8& acc1, v_f32x8& acc2,
                              v_f32x8& acc3, float& tail, const T* p, int64_t n)
{
    int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        acc0 = global_op<IsMax>(acc0, global_load(p + i));
        acc1 = global_op<IsMax>(acc1, global_load(p + i + 8));
        acc2 = global_op<IsMax>(acc2, global_load(p + i + 16));
        acc3 = global_op<IsMax>(acc3, global_load(p + i + 24));
    }
    for (; i + 8 <= n; i += 8) {
        acc0 = global_op<IsMax>(acc0, global_load(p + i));
    }
    for (; i < n; ++i) {
        tail = global_op<IsMax>(tail, s_load(p + i));
    }
}

/// Max/avg of one planar plane — one channel, the whole spatial extent.
///
/// A contiguous plane (pitch == width, which is what an allocated tensor has)
/// is one flat run, so a 7x7 avgpool — the shape the model runners actually
/// pool — vectorizes instead of falling entirely into the scalar tail. Only a
/// pitched plane walks row by row.
template <bool IsMax, typename T>
inline float global_planar(const T* in, const PoolingKernelParams& p, float scale)
{
    const v_f32x8 init = IsMax ? v_set1_f32x8(-std::numeric_limits<float>::infinity())
                               : v_zero_f32x8();
    v_f32x8 acc0 = init, acc1 = init, acc2 = init, acc3 = init;
    float tail = IsMax ? -std::numeric_limits<float>::infinity() : 0.0f;

    const bool flat = (p.in_row_stride == p.IW) &&
                      (p.in_d_stride == p.IH * p.in_row_stride);
    if (flat) {
        global_accumulate<IsMax>(acc0, acc1, acc2, acc3, tail, in,
                                 p.ID * p.IH * p.IW);
    } else {
        for (int64_t d = 0; d < p.ID; ++d) {
            const T* plane = in + d * p.in_d_stride;
            for (int64_t h = 0; h < p.IH; ++h) {
                global_accumulate<IsMax>(acc0, acc1, acc2, acc3, tail,
                                         plane + h * p.in_row_stride, p.IW);
            }
        }
    }

    acc0 = global_op<IsMax>(global_op<IsMax>(acc0, acc1),
                            global_op<IsMax>(acc2, acc3));
    const float v = global_op<IsMax>(global_fold(acc0, IsMax), tail);
    return IsMax ? v : v * scale;
}

/// Max/avg of one packed C8 plane — the lanes are the eight channels.
template <bool IsMax, typename T>
inline v_f32x8 global_packed(const T* in, const PoolingKernelParams& p, float scale)
{
    const v_f32x8 init = IsMax ? v_set1_f32x8(-std::numeric_limits<float>::infinity())
                               : v_zero_f32x8();
    v_f32x8 acc0 = init, acc1 = init, acc2 = init, acc3 = init;

    for (int64_t d = 0; d < p.ID; ++d) {
        const T* plane = in + d * p.in_d_stride;
        for (int64_t h = 0; h < p.IH; ++h) {
            const T* row = plane + h * p.in_row_stride;
            int64_t w = 0;
            for (; w + 4 <= p.IW; w += 4) {
                acc0 = global_op<IsMax>(acc0, global_load(row + (w + 0) * 8));
                acc1 = global_op<IsMax>(acc1, global_load(row + (w + 1) * 8));
                acc2 = global_op<IsMax>(acc2, global_load(row + (w + 2) * 8));
                acc3 = global_op<IsMax>(acc3, global_load(row + (w + 3) * 8));
            }
            for (; w < p.IW; ++w) {
                acc0 = global_op<IsMax>(acc0, global_load(row + w * 8));
            }
        }
    }

    const v_f32x8 v = global_op<IsMax>(global_op<IsMax>(acc0, acc1),
                                       global_op<IsMax>(acc2, acc3));
    return IsMax ? v : v_mul(v, v_set1_f32x8(scale));
}

/// Store a packed plane's reduced vector — narrowing for f16.
template <typename T>
inline void global_store(T* out, const v_f32x8& v, bool add_to)
{
    if constexpr (std::is_same_v<T, half>) {
        v_store_add(out, v_cvt_f32_to_f16(v), add_to);
    } else {
        v_store_add(out, v, add_to);
    }
}

// ============================================================
// MaxPool raw-integer fast path (input and output share scale + zero_point)
// ============================================================

template <int R, typename T>
inline void maxpool_fast(T* output, const T* input,
                         const PoolingKernelParams& p, float out_zero)
{
    constexpr T min_val = std::numeric_limits<T>::lowest();
    const T z = static_cast<T>(out_zero);

    for (int64_t ow = 0; ow < p.OW; ++ow) {
        T acc[R][8];
        bool any[R] = {};
        for (int r = 0; r < R; ++r) {
            for (int i = 0; i < 8; ++i) {
                acc[r][i] = min_val;
            }
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
                    for (int r = 0; r < R; ++r) {
                        if (!ok[r]) {
                            continue;
                        }
                        quant_kernel::max_8<T>(acc[r], acc[r],
                                input + d_off + ih[r] * p.in_row_stride + iw * 8);
                        any[r] = true;
                    }
                }
            }
        }

        for (int r = 0; r < R; ++r) {
            if (p.oh + r >= p.OH) {
                continue;
            }
            if (!any[r]) {
                for (int i = 0; i < 8; ++i) {
                    acc[r][i] = z;
                }
            }
            std::memcpy(output + r * p.out_row_stride + ow * 8, acc[r], 8);
        }
    }
}

// ============================================================
// Public kernels — unchanged signatures, one line each
// ============================================================

template <typename T>
inline void maxpool_h4(T* output, const T* input, const PoolingKernelParams& p,
                       float, bool add_to)
{
    pool_rows<4, true>(p, 0.0f, FpSrc<T>{input}, FpDst<T>{output, p.out_row_stride, add_to});
}

template <typename T>
inline void maxpool_h1(T* output, const T* input, const PoolingKernelParams& p,
                       float, bool add_to)
{
    pool_rows<1, true>(p, 0.0f, FpSrc<T>{input}, FpDst<T>{output, p.out_row_stride, add_to});
}

template <typename T>
inline void avgpool_h4(T* output, const T* input, const PoolingKernelParams& p,
                       float scale, bool add_to)
{
    pool_rows<4, false>(p, scale, FpSrc<T>{input}, FpDst<T>{output, p.out_row_stride, add_to});
}

template <typename T>
inline void avgpool_h1(T* output, const T* input, const PoolingKernelParams& p,
                       float scale, bool add_to)
{
    pool_rows<1, false>(p, scale, FpSrc<T>{input}, FpDst<T>{output, p.out_row_stride, add_to});
}

template <typename InT, typename OutT>
inline void maxpool_h4_quant(OutT* output, const InT* input, const PoolingKernelParams& p,
                             float in_scale, float in_zero,
                             float out_scale, float out_zero, bool add_to)
{
    pool_rows<4, true>(p, 0.0f, QuantSrc<InT>{input, in_scale, in_zero},
                       QuantDst<OutT>{output, p.out_row_stride, out_scale, out_zero, add_to});
}

template <typename InT, typename OutT>
inline void maxpool_h1_quant(OutT* output, const InT* input, const PoolingKernelParams& p,
                             float in_scale, float in_zero,
                             float out_scale, float out_zero, bool add_to)
{
    pool_rows<1, true>(p, 0.0f, QuantSrc<InT>{input, in_scale, in_zero},
                       QuantDst<OutT>{output, p.out_row_stride, out_scale, out_zero, add_to});
}

template <typename InT, typename OutT>
inline void avgpool_h4_quant(OutT* output, const InT* input, const PoolingKernelParams& p,
                             float scale, float in_scale, float in_zero,
                             float out_scale, float out_zero, bool add_to)
{
    pool_rows<4, false>(p, scale, QuantSrc<InT>{input, in_scale, in_zero},
                        QuantDst<OutT>{output, p.out_row_stride, out_scale, out_zero, add_to});
}

template <typename InT, typename OutT>
inline void avgpool_h1_quant(OutT* output, const InT* input, const PoolingKernelParams& p,
                             float scale, float in_scale, float in_zero,
                             float out_scale, float out_zero, bool add_to)
{
    pool_rows<1, false>(p, scale, QuantSrc<InT>{input, in_scale, in_zero},
                        QuantDst<OutT>{output, p.out_row_stride, out_scale, out_zero, add_to});
}

template <typename T>
inline void maxpool_planar(T* output, const T* input, const PoolingKernelParams& p,
                           float, bool add_to)
{
    pool_planar<true>(output, input, p, 0.0f, add_to);
}

template <typename T>
inline void avgpool_planar(T* output, const T* input, const PoolingKernelParams& p,
                           float scale, bool add_to)
{
    pool_planar<false>(output, input, p, scale, add_to);
}

template <typename T>
inline void maxpool_h4_quant_fast(T* output, const T* input,
                                  const PoolingKernelParams& p, float out_zero)
{
    maxpool_fast<4>(output, input, p, out_zero);
}

template <typename T>
inline void maxpool_h1_quant_fast(T* output, const T* input,
                                  const PoolingKernelParams& p, float out_zero)
{
    maxpool_fast<1>(output, input, p, out_zero);
}

}  // namespace nnops::kernel
