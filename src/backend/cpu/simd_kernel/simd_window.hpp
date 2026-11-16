#pragma once
/// @file simd_window.hpp
/// @brief Window geometry shared by the depthwise and pooling channel drivers.
///
/// Both drivers walk an output row the same way: drop the per-tap bounds tests
/// inside a span of columns whose whole horizontal window is inside the row,
/// and keep a checked path for the two padded edges. The span and the row table
/// are a property of the call's geometry, not of the kernel body, so they are
/// built here once and read by every variant.
///
/// The row table is the same indirection step XNNPACK performs for its pooling
/// and depthwise micro-kernels: one pointer per (output row, filter row), null
/// where the window reaches past the plane. A kernel that can drop the null
/// test entirely (every row valid) says so through `all_valid`.

#include <cstdint>

namespace nnops::kernel {

/// One call's column walk over an output row.
///
/// `end` is one past the last output column. The interior `[lo, hi]` is the
/// range whose every horizontal tap is inside the input row, so a tap loop
/// there needs no range test; `hi < lo` means there is no interior. A tap's
/// input offset, measured from the padded row start, is valid in
/// `[tap_lo, tap_hi)`.
struct WindowSpan {
    int64_t end;
    int64_t lo;
    int64_t hi;
    int64_t tap_lo;
    int64_t tap_hi;
};

template <class Params>
inline WindowSpan window_span(const Params& p, int64_t kernel_w, int64_t dilation_w) noexcept
{
    const int64_t end = (p.ow_end >= 0) ? p.ow_end : p.OW;
    const int64_t lo = (p.PW + p.SW - 1) / p.SW;
    const int64_t last = p.IW - 1 + p.PW - (kernel_w - 1) * dilation_w;
    const int64_t hi = (last >= 0) ? last / p.SW : -1;
    return {end, lo, hi, p.PW, p.IW + p.PW};
}

/// Result of a row-table build. `built` is false when the geometry needs more
/// rows than the caller's fixed-capacity table holds; the caller then walks the
/// rows on the fly. `all_valid` is true when every row is inside the plane.
struct RowTable {
    bool built;
    bool all_valid;
};

/// Fill @p table with one entry per (kd, kh, r): the input row biased by the
/// left pad so a tap is `row + (ow*SW + kw)*pix`, or null when the window
/// reaches past the plane. Indexed `[(kd*KH + kh)*R + r]`.
template <class Params, class T>
inline RowTable build_row_table(const Params& p, int64_t R, const T* image, const T** table,
                                int64_t capacity) noexcept
{
    const int64_t nrow = p.KD * p.KH;
    if (nrow * R > capacity) {
        return {false, false};
    }
    const int64_t pix = p.pix_stride;
    int64_t t = 0;
    bool all = true;
    for (int64_t kd = 0; kd < p.KD; ++kd) {
        const int64_t id = p.od * p.SD + kd * p.DD - p.PD;
        const bool d_ok = (id >= 0 && id < p.ID);
        const T* in_d = d_ok ? image + id * p.in_d_stride - p.PW * pix : nullptr;
        for (int64_t kh = 0; kh < p.KH; ++kh) {
            for (int64_t r = 0; r < R; ++r) {
                const int64_t ih = (p.oh + r) * p.SH + kh * p.DH - p.PH;
                const bool ok = d_ok && (p.oh + r < p.OH) && ih >= 0 && ih < p.IH;
                table[t++] = ok ? in_d + ih * p.in_row_stride : nullptr;
                all = all && ok;
            }
        }
    }
    return {true, all};
}

} // namespace nnops::kernel
