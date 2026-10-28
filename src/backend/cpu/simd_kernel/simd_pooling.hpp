#pragma once
/// @file simd_pooling.hpp
/// @brief SIMD kernels for 2D/3D spatial pooling (NCHWC8 only).
///
/// One driver covers every variant: the row count (h4/h1), the accumulate
/// (max/avg), and the element type (f32/f16 vs dequantize/requantize) are all
/// template parameters. Every C8 block — partial ones included — runs the SIMD
/// path, since NCHWC8 pads channels with zeros (max ignores them, avg adds 0).
/// Bounds are checked in-kernel, so any stride/padding combination works.

#include "nnops/detail/simd/simd.hpp"
#include "simd_access.hpp"

#include <cstdint>
#include <cstring>
#include <limits>

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
