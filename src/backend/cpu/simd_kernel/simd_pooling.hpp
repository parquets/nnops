#pragma once
/// @file simd_pooling.hpp
/// @brief SIMD kernels for 2D/3D spatial pooling.
///
/// Two drivers, one per lane choice. The channel driver lanes a channel vector
/// over output columns (channel-last); the channel-first driver, 2D and 3D,
/// lanes the output column. One channel driver covers every variant:
/// row count R, IsMax, and Src/Dst (f32/f16, dequantize/requantize) are template
/// parameters, and bounds are checked in-kernel, so any stride/padding works.
/// Quantized pooling runs on the channel lane only, not channel-first.
///
/// The channel walk mirrors the depthwise kernel: one row pointer per (kd, kh, r),
/// biased by the left pad, an interior span whose every tap is in the input row
/// that takes no bounds or null test, and NV adjacent channel vectors per call —
/// 2 where the channels allow, since two f32 vectors fill a cache line and a
/// one-vector sweep pays twice the line traffic for the same arithmetic.
///
/// A channel-first lane cannot skip an out-of-range tap (a padded lane would read the
/// pitch), so channel_first_load fills the accumulate's identity and a short last block
/// stores through a scalar tail. Channel-first runs one output row per call — the
/// channel path's R-row blocking shares input rows, which a width-vectorized
/// kernel does not need. Global pooling scans contiguous memory with four
/// accumulators and widens f16 to f32 on load so the sum never runs in f16. A
/// contiguous plane is one flat run, so even a small plane — a 7x7 global pool,
/// the shape the model runners see — vectorizes instead of falling into the tail.

#include "nnops/detail/simd/simd.hpp"
#include "simd_access.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>

namespace nnops::kernel {

using namespace simd;

// --- Kernel parameters: set once, od/oh updated per call ---

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
    int64_t pix_stride = 8;      // elems between adjacent pixels: the channel
                                 // count on a channel-last plane
    int64_t ow0 = 0;             // first output column this call covers
    int64_t ow_end = -1;         // one past the last; negative means OW
};

// --- The channel driver: R output rows per call, IsMax picks the accumulate ---

/// Whether a window can hold no tap at all. Max needs a per-window count only
/// then; a padding that reaches past the kernel on an axis, or a trailing
/// window that starts beyond the plane, is the only way it happens.
inline bool pooling_counts_taps(const PoolingKernelParams& p, bool is_max)
{
    if (!is_max) { return p.exclude_pad; }
    return p.PH > (p.KH - 1) * p.DH || p.PW > (p.KW - 1) * p.DW ||
           (p.OH > 0 && (p.OH - 1) * p.SH - p.PH >= p.IH) ||
           (p.OW > 0 && (p.OW - 1) * p.SW - p.PW >= p.IW);
}

template <int R, int NV, bool IsMax, bool COUNT, typename Src, typename Dst>
inline void pool_rows(const PoolingKernelParams& p, float avg_scale,
                      const Src (&srcs)[NV], const Dst (&dsts)[NV])
{
    using V = decltype(srcs[0].zero());
    using InPtr = const typename Src::elem*;
    const V vzero = srcs[0].zero();
    const int64_t pix = p.pix_stride;
    constexpr int64_t kVec = kChannelLanes;
    constexpr int64_t kMaxTapRows = 256;

    V vinit;
    if constexpr (IsMax) {
        vinit = srcs[0].set1(-std::numeric_limits<float>::infinity());
    } else {
        vinit = vzero;
    }

    constexpr bool count = COUNT;

    const int64_t nrow = p.KD * p.KH;
    InPtr row_base[kMaxTapRows];
    const bool have_tab = (nrow * R <= kMaxTapRows);
    bool all_rows = have_tab;
    if (have_tab) {
        int64_t t = 0;
        for (int64_t kd = 0; kd < p.KD; ++kd) {
            const int64_t id = p.od * p.SD + kd * p.DD - p.PD;
            const bool d_ok = (id >= 0 && id < p.ID);
            const InPtr in_d = srcs[0].ptr_at(id * p.in_d_stride - p.PW * pix);
            for (int64_t kh = 0; kh < p.KH; ++kh) {
                for (int r = 0; r < R; ++r) {
                    const int64_t ih = (p.oh + r) * p.SH + kh * p.DH - p.PH;
                    const bool ok = d_ok && (p.oh + r < p.OH) &&
                                    ih >= 0 && ih < p.IH;
                    row_base[t++] = ok ? in_d + ih * p.in_row_stride : nullptr;
                    all_rows = all_rows && ok;
                }
            }
        }
    }

    // Output columns whose whole window is inside the input row.
    const int64_t ow_lo = (p.PW + p.SW - 1) / p.SW;
    const int64_t last_hi = p.IW - 1 + p.PW - (p.KW - 1) * p.DW;
    const int64_t ow_hi = (last_hi >= 0) ? last_hi / p.SW : -1;
    const int64_t off_ok_lo = p.PW;             // tap offsets are measured from
    const int64_t off_ok_hi = p.IW + p.PW;      // the padded row start

    const int64_t ow_end = (p.ow_end >= 0) ? p.ow_end : p.OW;
    for (int64_t ow = p.ow0; ow < ow_end; ++ow) {
        V acc[R][NV];
        int64_t cnt[R] = {};
        for (int r = 0; r < R; ++r) {
            for (int k = 0; k < NV; ++k) { acc[r][k] = vinit; }
        }

        const bool interior = all_rows && ow >= ow_lo && ow <= ow_hi;
        const int64_t ow_off = ow * p.SW;

        for (int64_t kd = 0; kd < p.KD; ++kd) {
            for (int64_t kh = 0; kh < p.KH; ++kh) {
                InPtr row[R];
                if (have_tab) {
                    const int64_t t = (kd * p.KH + kh) * R;
                    for (int64_t r = 0; r < R; ++r) { row[r] = row_base[t + r]; }
                } else {
                    const int64_t id = p.od * p.SD + kd * p.DD - p.PD;
                    if (id < 0 || id >= p.ID) { continue; }
                    const InPtr in_d = srcs[0].ptr_at(id * p.in_d_stride - p.PW * pix);
                    for (int64_t r = 0; r < R; ++r) {
                        const int64_t ih = (p.oh + r) * p.SH + kh * p.DH - p.PH;
                        const bool ok = (p.oh + r < p.OH) && ih >= 0 && ih < p.IH;
                        row[r] = ok ? in_d + ih * p.in_row_stride : nullptr;
                    }
                }

                if (interior) {
                    for (int64_t kw = 0; kw < p.KW; ++kw) {
                        const int64_t at = (ow_off + kw * p.DW) * pix;
                        for (int k = 0; k < NV; ++k) {
                            for (int r = 0; r < R; ++r) {
                                const V v = srcs[k].load_ptr(row[r] + at + k * kVec);
                                acc[r][k] = IsMax ? v_max(acc[r][k], v)
                                                  : v_add(acc[r][k], v);
                            }
                        }
                    }
                    // Counted even for max: an all-pad window stores zero, not -inf.
                    if constexpr (COUNT) {
                        for (int r = 0; r < R; ++r) { cnt[r] += p.KW; }
                    }
                    continue;
                }

                int64_t valid_kw = 0;
                for (int64_t kw = 0; kw < p.KW; ++kw) {
                    const int64_t off = ow_off + kw * p.DW;
                    if (off < off_ok_lo || off >= off_ok_hi) {
                        continue;
                    }
                    ++valid_kw;
                    const int64_t at = off * pix;
                    for (int k = 0; k < NV; ++k) {
                        for (int r = 0; r < R; ++r) {
                            if (row[r] == nullptr) {
                                continue;
                            }
                            const V v = srcs[k].load_ptr(row[r] + at + k * kVec);
                            acc[r][k] = IsMax ? v_max(acc[r][k], v)
                                              : v_add(acc[r][k], v);
                        }
                    }
                }
                for (int r = 0; r < R; ++r) {
                    if constexpr (COUNT) {
                        if (row[r] != nullptr) { cnt[r] += valid_kw; }
                    }
                }
            }
        }

        for (int r = 0; r < R; ++r) {
            if (p.oh + r >= p.OH) {
                continue;
            }
            for (int k = 0; k < NV; ++k) {
                V v = acc[r][k];
                if constexpr (IsMax) {
                    if constexpr (COUNT) {
                        if (cnt[r] == 0) { v = vzero; }
                    }
                } else if (p.exclude_pad) {
                    if constexpr (COUNT) {
                        if (cnt[r] > 0) {
                            v = v_mul(v, srcs[k].set1(1.0f / static_cast<float>(cnt[r])));
                        }
                    }
                } else {
                    v = v_mul(v, srcs[k].set1(avg_scale));
                }
                dsts[k].store(r, ow, v);
            }
        }
    }
}

// --- channel-first driver: the SIMD lane spans output width ---

/// The value a padded lane contributes: the max's identity, or zero for the sum.
/// `half` is a struct with no numeric_limits specialization on x86, so its
/// lowest() is 0 — which would clamp any window that touches padding at zero.
template <typename T>
inline T pooling_fill(bool is_max) noexcept
{
    if (!is_max) { return T{}; }
    if constexpr (std::is_same_v<T, half>) {
        return ::nnops::backend::cpu::float_to_half(-65504.0f);
#ifdef NNOPS_ARCH_AARCH64
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        // bf16 also has no numeric_limits specialization; 0xFF7F is its most
        // negative finite value (rounding FLT_MAX to bf16 would give -inf).
        return ::nnops::backend::cpu::bfloat16_from_bits(0xFF7Fu);
#endif
    } else {
        return std::numeric_limits<T>::lowest();
    }
}

template <bool IsMax, typename T>
inline void pool_channel_first(T* out, const T* in, const PoolingKernelParams& p,
                        float avg_scale, bool add_to)
{
    constexpr int64_t L = simd_lane_for<T>;
    const T fill = pooling_fill<T>(IsMax);

    // Row-constant half of the tap count: in-range (kd, kh) positions.
    int64_t row_taps = 0;
    for (int64_t kd = 0; kd < p.KD; ++kd) {
        const int64_t id = p.od * p.SD + kd * p.DD - p.PD;
        if (id < 0 || id >= p.ID) { continue; }
        for (int64_t kh = 0; kh < p.KH; ++kh) {
            const int64_t ih = p.oh * p.SH + kh * p.DH - p.PH;
            if (ih >= 0 && ih < p.IH) { ++row_taps; }
        }
    }

    // W padding is the only dim that trims a window per column, so it forces a
    // per-lane count; otherwise the count is a constant — unless a window runs
    // off the right edge, which PW cannot see: a SAME-style pad is leading-only
    // and its trailing half is implicit, so PW == 0 with the last window past
    // IW would divide by taps that were never read.
    const bool cols_inside = p.PW == 0 &&
        (p.OW - 1) * p.SW + (p.KW - 1) * p.DW < p.IW;
    const auto taps_of = [&](int64_t col) {
        if (cols_inside) { return row_taps * p.KW; }
        const int64_t iw_first = col * p.SW - p.PW;
        int64_t n = 0;
        for (int64_t kw = 0; kw < p.KW; ++kw) {
            const int64_t iw = iw_first + kw * p.DW;
            n += (iw >= 0 && iw < p.IW) ? 1 : 0;
        }
        return row_taps * n;
    };

    // Max: an all-pad window is 0 (the channel kernel zeroes it too).
    // Avg: exclude_pad divides by the tap count; with no padding the count is
    // uniform and this is a single scale.
    const bool count_taps = IsMax
        ? (p.PH > (p.KH - 1) * p.DH || p.PW > (p.KW - 1) * p.DW ||
           (p.OH > 0 && (p.OH - 1) * p.SH - p.PH >= p.IH) ||
           (p.OW > 0 && (p.OW - 1) * p.SW - p.PW >= p.IW))
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
                    const auto v = channel_first_load(row, iw0, p.SW, p.IW, fill);
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

// --- Global pooling: the window is the whole extent, so a contiguous plane is one scan ---

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

/// Widened load of a short channel vector; zero-filled past @p lanes, and only
/// the live lanes are ever stored.
template <typename T>
inline v_f32x8 global_load_lanes(const T* p, int64_t lanes)
{
    if (lanes == kChannelLanes) { return global_load(p); }
    T tmp[kChannelLanes] = {};
    std::memcpy(tmp, p, static_cast<size_t>(lanes) * sizeof(T));
    return global_load(tmp);
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

/// Fold @p n contiguous elements into the four accumulators and the scalar tail.
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

/// Max/avg of one channel-first plane. A contiguous plane (pitch == width, as an
/// allocated tensor has) is one flat run; only a pitched plane walks row by row.
template <bool IsMax, typename T>
inline float global_channel_first(const T* in, const PoolingKernelParams& p, float scale)
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

template <typename T>
inline void global_store_lanes(T* out, const v_f32x8& v, bool add_to, int64_t lanes);

/// Max/avg of one channel vector's plane — eight channels of a pixel, every
/// @c pix_stride-th vector on a channel-last tensor. NV vectors per pass; a
/// short last block takes masked loads and stores only its live lanes.
template <bool IsMax, typename T, int NV, int U = 4, bool Full = true>
inline void global_channel_vec(T* out, const T* in, const PoolingKernelParams& p,
                               float scale, bool add_to, int64_t lanes)
{
    constexpr int64_t kVec = kChannelLanes;
    const v_f32x8 init = IsMax ? v_set1_f32x8(-std::numeric_limits<float>::infinity())
                               : v_zero_f32x8();
    v_f32x8 acc[NV][U];
    for (int k = 0; k < NV; ++k) {
        for (int u = 0; u < U; ++u) { acc[k][u] = init; }
    }

    // Live lanes of vector k: a ragged tail leaves the later vectors short.
    const auto live = [&](int k) {
        const int64_t l = lanes - k * kVec;
        return (l < 0) ? int64_t{0} : (l > kVec ? kVec : l);
    };
    // Full vectors are a separate instantiation, so the common case pays no
    // per-load lane test.
    const auto load = [&](const T* px, int k) {
        if constexpr (Full) {
            return global_load(px);
        } else {
            const int64_t l = live(k);
            return (l >= kVec) ? global_load(px) : global_load_lanes(px, l);
        }
    };

    for (int64_t d = 0; d < p.ID; ++d) {
        const T* plane = in + d * p.in_d_stride;
        for (int64_t h = 0; h < p.IH; ++h) {
            const T* row = plane + h * p.in_row_stride;
            int64_t w = 0;
            for (; w + U <= p.IW; w += U) {
                for (int k = 0; k < NV; ++k) {
                    const int64_t ko = k * kVec;
                    for (int u = 0; u < U; ++u) {
                        acc[k][u] = global_op<IsMax>(
                            acc[k][u], load(row + (w + u) * p.pix_stride + ko, k));
                    }
                }
            }
            for (; w < p.IW; ++w) {
                for (int k = 0; k < NV; ++k) {
                    acc[k][0] = global_op<IsMax>(
                        acc[k][0], load(row + w * p.pix_stride + k * kVec, k));
                }
            }
        }
    }

    for (int k = 0; k < NV; ++k) {
        v_f32x8 v = acc[k][0];
        for (int u = 1; u < U; ++u) { v = global_op<IsMax>(v, acc[k][u]); }
        if constexpr (!IsMax) { v = v_mul(v, v_set1_f32x8(scale)); }
        global_store_lanes(out + k * kVec, v, add_to, live(k));
    }
}

/// Single-vector form for the short last block of a channel-last task.
template <bool IsMax, typename T>
inline v_f32x8 global_channel(const T* in, const PoolingKernelParams& p, float scale,
                              int64_t lanes)
{
    constexpr int64_t kVec = kChannelLanes;
    const v_f32x8 init = IsMax ? v_set1_f32x8(-std::numeric_limits<float>::infinity())
                               : v_zero_f32x8();
    v_f32x8 acc0 = init, acc1 = init, acc2 = init, acc3 = init;

    const auto load = [&](const T* px) {
        return (lanes >= kVec) ? global_load(px) : global_load_lanes(px, lanes);
    };

    for (int64_t d = 0; d < p.ID; ++d) {
        const T* plane = in + d * p.in_d_stride;
        for (int64_t h = 0; h < p.IH; ++h) {
            const T* row = plane + h * p.in_row_stride;
            int64_t w = 0;
            for (; w + 4 <= p.IW; w += 4) {
                acc0 = global_op<IsMax>(acc0, load(row + (w + 0) * p.pix_stride));
                acc1 = global_op<IsMax>(acc1, load(row + (w + 1) * p.pix_stride));
                acc2 = global_op<IsMax>(acc2, load(row + (w + 2) * p.pix_stride));
                acc3 = global_op<IsMax>(acc3, load(row + (w + 3) * p.pix_stride));
            }
            for (; w < p.IW; ++w) {
                acc0 = global_op<IsMax>(acc0, load(row + w * p.pix_stride));
            }
        }
    }

    const v_f32x8 v = global_op<IsMax>(global_op<IsMax>(acc0, acc1),
                                       global_op<IsMax>(acc2, acc3));
    return IsMax ? v : v_mul(v, v_set1_f32x8(scale));
}

/// Store a reduced channel vector, narrowing for f16.
template <typename T>
inline void global_store(T* out, const v_f32x8& v, bool add_to)
{
    if constexpr (std::is_same_v<T, half>) {
        v_store_add(out, v_cvt_f32_to_f16(v), add_to);
    } else {
        v_store_add(out, v, add_to);
    }
}

// --- MaxPool raw-integer fast path (input and output share scale + zero_point) ---

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
                                input + d_off + ih[r] * p.in_row_stride + iw * p.pix_stride);
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
            std::memcpy(output + r * p.out_row_stride + ow * p.pix_stride, acc[r], 8);
        }
    }
}

// --- Channel-last wrappers ---

template <bool IsMax, typename T>
inline void pool_last(T* output, const T* input, const PoolingKernelParams& p,
                      float scale, bool add_to, int64_t lanes)
{
    if (lanes > kChannelLanes) {
        const FpSrc<T, true> src[2] = {
            FpSrc<T, true>{input, kChannelLanes},
            FpSrc<T, true>{input + kChannelLanes, kChannelLanes}};
        const FpDst<T, true> dst[2] = {
            FpDst<T, true>{output, p.out_row_stride, add_to, p.pix_stride,
                           kChannelLanes},
            FpDst<T, true>{output + kChannelLanes, p.out_row_stride, add_to,
                           p.pix_stride, kChannelLanes}};
        pool_rows_counted<2, IsMax>(p, scale, src, dst);
        return;
    }
    if (lanes == kChannelLanes) {
        const FpSrc<T, true> src[1] = {FpSrc<T, true>{input, lanes}};
        const FpDst<T, true> dst[1] = {
            FpDst<T, true>{output, p.out_row_stride, add_to, p.pix_stride, lanes}};
        pool_rows_counted<1, IsMax>(p, scale, src, dst);
    } else {
        const FpSrc<T, false> src[1] = {FpSrc<T, false>{input, lanes}};
        const FpDst<T, false> dst[1] = {
            FpDst<T, false>{output, p.out_row_stride, add_to, p.pix_stride, lanes}};
        pool_rows_counted<1, IsMax>(p, scale, src, dst);
    }
}

/// Straight-line 3x3 window on the channel lane: KD=1, DH=DW=1 and square
/// strides, folds the nine taps into literal offsets and walks the column
/// offsets, so the body carries no tap counter and no per-tap address
/// multiply. Only the count-free case; a window that can be all padding keeps
/// the rolled kernel that counts taps.
///
/// One output row at a time: its accumulators are one or two vectors, which the
/// allocator keeps in registers. A group's `V acc[R][NV]` array went to the
/// stack on x86-64 (the array's address is taken by the row loop), so every tap
/// was a load-store round trip on both the straight-line and the masked path.
/// Splitting per row also lets a group that touches the padding still run its
/// interior rows straight-line.
template <int R, int NV, bool IsMax, int S, typename Src, typename Dst>
inline void pool_rows_3x3(const PoolingKernelParams& p, float avg_scale,
                          const Src (&srcs)[NV], const Dst (&dsts)[NV])
{
    using V = decltype(srcs[0].zero());
    using InPtr = const typename Src::elem*;
    constexpr int64_t kL = kChannelLanes;
    const V vinit = IsMax ? srcs[0].set1(-std::numeric_limits<float>::infinity())
                          : srcs[0].zero();
    const int64_t pix = p.pix_stride;
    const int64_t rs = p.in_row_stride;

    const InPtr img = srcs[0].ptr_at(0);
    const int64_t ow_lo = (p.PW + S - 1) / S;
    const int64_t last_hi = p.IW - 1 + p.PW - 2;
    const int64_t ow_hi = (last_hi >= 0) ? last_hi / S : -1;
    const int64_t ow_end = (p.ow_end >= 0) ? p.ow_end : p.OW;

    // One input row's three horizontally adjacent taps into `acc`.
    const auto tap3 = [&](const Src& src, const InPtr rp, int64_t o0,
                          int64_t o1, int64_t o2, V& acc) {
        const V a = src.load_ptr(rp + o0);
        const V b = src.load_ptr(rp + o1);
        const V c = src.load_ptr(rp + o2);
        if constexpr (IsMax) { acc = v_max(v_max(v_max(acc, a), b), c); }
        else { acc = v_add(v_add(v_add(acc, a), b), c); }
    };

    // Row outer: its valid tap range and base pointer are fixed for the whole
    // width sweep, which is the whole cost of the boundary path on a small
    // plane (a 7x7 output is four columns wide, so the old column-outer order
    // rebuilt that setup once per output pixel).
    for (int r = 0; r < R; ++r) {
        const int64_t oh_r = p.oh + r;
        if (oh_r >= p.OH) { continue; }
        const int64_t ihb = oh_r * S - p.PH;
        const int64_t kh_lo = ihb >= 0 ? 0 : -ihb;
        const int64_t kh_hi = std::min<int64_t>(3, p.IH - ihb);
        const InPtr rp0 = img + (ihb + kh_lo) * rs;
        const V vs = IsMax ? vinit : srcs[0].set1(avg_scale);

        for (int64_t ow = p.ow0; ow < ow_end; ++ow) {
            const bool col_ok = (ow >= ow_lo && ow <= ow_hi);
            const int64_t col0 = ow * S - p.PW;
            const int64_t o0 = col0 * pix;
            const int64_t o1 = o0 + pix;
            const int64_t o2 = o1 + pix;

            V a0 = vinit;
            V a1 = vinit;

            if (col_ok && kh_lo == 0 && kh_hi == 3) {
                const InPtr rp = img + ihb * rs;
                tap3(srcs[0], rp, o0, o1, o2, a0);
                tap3(srcs[0], rp + rs, o0, o1, o2, a0);
                tap3(srcs[0], rp + 2 * rs, o0, o1, o2, a0);
                if constexpr (NV == 2) {
                    tap3(srcs[1], rp + kL, o0, o1, o2, a1);
                    tap3(srcs[1], rp + rs + kL, o0, o1, o2, a1);
                    tap3(srcs[1], rp + 2 * rs + kL, o0, o1, o2, a1);
                }
            } else {
                // Only the in-range taps of the window are visited, so the
                // boundary is a shorter loop rather than a bounds test per tap.
                const int64_t kw_lo = col0 >= 0 ? 0 : -col0;
                const int64_t kw_hi = std::min<int64_t>(3, p.IW - col0);
                for (int64_t kh = kh_lo; kh < kh_hi; ++kh) {
                    const InPtr rp = rp0 + (kh - kh_lo) * rs;
                    for (int64_t kw = kw_lo; kw < kw_hi; ++kw) {
                        const int64_t at = (col0 + kw) * pix;
                        const V x0 = srcs[0].load_ptr(rp + at);
                        a0 = IsMax ? v_max(a0, x0) : v_add(a0, x0);
                        if constexpr (NV == 2) {
                            const V x1 = srcs[1].load_ptr(rp + at + kL);
                            a1 = IsMax ? v_max(a1, x1) : v_add(a1, x1);
                        }
                    }
                }
            }

            if constexpr (!IsMax) {
                a0 = v_mul(a0, vs);
                if constexpr (NV == 2) { a1 = v_mul(a1, vs); }
            }
            dsts[0].store(r, ow, a0);
            if constexpr (NV == 2) { dsts[1].store(r, ow, a1); }
        }
    }
}

template <int NV, bool IsMax, typename Src, typename Dst>
inline bool pool_try_3x3(const PoolingKernelParams& p, float scale,
                         const Src (&srcs)[NV], const Dst (&dsts)[NV])
{
    if constexpr (!Src::full) { return false; }
    if (p.KD != 1 || p.KH != 3 || p.KW != 3 || p.DH != 1 || p.DW != 1 ||
        p.SH != p.SW || pooling_counts_taps(p, IsMax)) {
        return false;
    }
    if (p.SH == 1) {
        pool_rows_3x3<4, NV, IsMax, 1>(p, scale, srcs, dsts);
        return true;
    }
    if (p.SH == 2) {
        pool_rows_3x3<4, NV, IsMax, 2>(p, scale, srcs, dsts);
        return true;
    }
    return false;
}

/// Route override for A/B measurement: a nonzero store forces the rolled walk
/// and skips the straight-line 3x3 path. The environment seeds it once at
/// start-up (NNOPS_POOLING_ROUTE=roll); a bench flips the atomic in-process, so
/// both routes are timed over the same buffers. The hot check is one relaxed
/// load — a per-call getenv costs microseconds on Windows and would swamp a
/// small plane.
inline std::atomic<int> pool_route_roll{[] {
    const char* e = std::getenv("NNOPS_POOLING_ROUTE");
    return (e != nullptr && std::strcmp(e, "roll") == 0) ? 1 : 0;
}()};

inline bool pool_3x3_route_enabled()
{
    return pool_route_roll.load(std::memory_order_relaxed) == 0;
}

template <int NV, bool IsMax, typename Src, typename Dst>
inline void pool_rows_counted(const PoolingKernelParams& p, float scale,
                              const Src (&srcs)[NV], const Dst (&dsts)[NV])
{
    if (pool_3x3_route_enabled() &&
        pool_try_3x3<NV, IsMax>(p, scale, srcs, dsts)) {
        return;
    }
    if (pooling_counts_taps(p, IsMax)) {
        pool_rows<4, NV, IsMax, true>(p, scale, srcs, dsts);
    } else {
        pool_rows<4, NV, IsMax, false>(p, scale, srcs, dsts);
    }
}

template <bool IsMax, typename InT, typename OutT>
inline void pool_last_quant(OutT* output, const InT* input,
                            const PoolingKernelParams& p, float scale,
                            float in_scale, float in_zero,
                            float out_scale, float out_zero,
                            bool add_to, int64_t lanes)
{
    if (lanes > kChannelLanes) {
        const QuantSrc<InT, true> src[2] = {
            QuantSrc<InT, true>{input, in_scale, in_zero, kChannelLanes},
            QuantSrc<InT, true>{input + kChannelLanes, in_scale, in_zero,
                                kChannelLanes}};
        const QuantDst<OutT, true> dst[2] = {
            QuantDst<OutT, true>{output, p.out_row_stride, out_scale, out_zero,
                                 add_to, p.pix_stride, kChannelLanes},
            QuantDst<OutT, true>{output + kChannelLanes, p.out_row_stride,
                                 out_scale, out_zero, add_to, p.pix_stride,
                                 kChannelLanes}};
        pool_rows_counted<2, IsMax>(p, scale, src, dst);
        return;
    }
    if (lanes == kChannelLanes) {
        const QuantSrc<InT, true> src[1] = {
            QuantSrc<InT, true>{input, in_scale, in_zero, lanes}};
        const QuantDst<OutT, true> dst[1] = {
            QuantDst<OutT, true>{output, p.out_row_stride, out_scale, out_zero,
                                 add_to, p.pix_stride, lanes}};
        pool_rows_counted<1, IsMax>(p, scale, src, dst);
    } else {
        const QuantSrc<InT, false> src[1] = {
            QuantSrc<InT, false>{input, in_scale, in_zero, lanes}};
        const QuantDst<OutT, false> dst[1] = {
            QuantDst<OutT, false>{output, p.out_row_stride, out_scale, out_zero,
                                  add_to, p.pix_stride, lanes}};
        pool_rows_counted<1, IsMax>(p, scale, src, dst);
    }
}

/// Store a reduced channel vector, narrowing for f16, @p lanes elements only.
template <typename T>
inline void global_store_lanes(T* out, const v_f32x8& v, bool add_to, int64_t lanes)
{
    if (lanes == kChannelLanes) {
        global_store(out, v, add_to);
        return;
    }
    T tmp[kChannelLanes];
    global_store(tmp, v, false);
    for (int64_t l = 0; l < lanes; ++l) {
        s_store_add(out + l, s_load(tmp + l), add_to);
    }
}

// --- channel-first entry points ---

template <typename T>
inline void maxpool_channel_first(T* output, const T* input, const PoolingKernelParams& p,
                           float, bool add_to)
{
    pool_channel_first<true>(output, input, p, 0.0f, add_to);
}

template <typename T>
inline void avgpool_channel_first(T* output, const T* input, const PoolingKernelParams& p,
                           float scale, bool add_to)
{
    pool_channel_first<false>(output, input, p, scale, add_to);
}

}  // namespace nnops::kernel
