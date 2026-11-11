#pragma once
/// @file transpose_conv2d_col2im.h
/// @brief The col → output pass of transposed convolution: the output-driven
///        window walk, and the two scatters that fold a col matrix back into
///        image pixels.
///
/// Transposed convolution's gather form is output-driven: pixel (oh, ow) reads
/// input (ih, iw) wherever oh + PH - kh·DH == ih·SH. Run as a GEMM over the
/// input plane — which is what makes its arithmetic intensity right — the col
/// comes out indexed by *input* pixel, so the result has to be folded back into
/// the image under that inverse map. The fold is the whole difficulty of the
/// operator, and in both col orientations it is the same two facts:
///
///   * rows. Kernel row kh lands on output row oh only when oh + PH - kh·DH is
///     a non-negative multiple of SH, so one output row sees between 0 and KH
///     input rows (`tconv2d_row_taps`).
///   * columns. A pixel of residue class r sits at ow = r + t·SW, so a tap
///     whose column offset is a multiple of SW contributes to every SW-th
///     column of that class, over a contiguous run of t (`tconv2d_col_span`).
///     A tap that misses the class — the stride does not divide its offset —
///     contributes nothing to it.
///
/// The two scatters are the two col orientations:
///
///   * channel-major — one col row per (oc, tap) running over the input plane,
///     so a tap adds one contiguous run per residue class, and the run's
///     length is the only bound to get right.
///   * pixel-major — one col row per input pixel holding [tap][oc], so a tap
///     adds one oc vector per pixel, and the pixels of a class are SW apart in
///     the scratch row.
///
/// Both accumulate into a caller-owned scratch row and stop there. The caller
/// merges that row into the output with its own epilogue, which keeps one
/// writer per output element — and that is what lets add_to read the old value
/// without a save/restore pass. The accumulate is a callback because the same
/// walk serves an add (the fp and int8 cols) and a weighted add (depthwise,
/// whose "col" is the input itself, one channel per group).

#include <algorithm>
#include <cstdint>

namespace nnops::backend::cpu {

/// Tallest per-row tap list the scatter keeps on the stack, so also the largest
/// KH the col routes accept.
inline constexpr int64_t kMaxRowTaps = 16;

/// The window geometry a col2im pass reads. `mic` and `row_len` are derived
/// from the rest and exist because every route uses them.
struct Tconv2dColGeom {
    int64_t IH = 0, IW = 0;
    int64_t OW = 0;
    int64_t KH = 0, KW = 0;
    int64_t SH = 1, SW = 1;
    int64_t DH = 1, DW = 1;
    int64_t PH = 0, PW = 0;
    int64_t mic = 0;      ///< ⌈OW/SW⌉ — output pixels in a residue class
    int64_t row_len = 0;  ///< SW·mic — the channel-major scratch row width
};

inline Tconv2dColGeom make_col_geom(int64_t IH, int64_t IW, int64_t OW, int64_t KH,
                                    int64_t KW, int64_t SH, int64_t SW, int64_t DH,
                                    int64_t DW, int64_t PH, int64_t PW)
{
    Tconv2dColGeom g;
    g.IH = IH;
    g.IW = IW;
    g.OW = OW;
    g.KH = KH;
    g.KW = KW;
    g.SH = SH;
    g.SW = SW;
    g.DH = DH;
    g.DW = DW;
    g.PH = PH;
    g.PW = PW;
    g.mic = (OW + SW - 1) / SW;
    g.row_len = SW * g.mic;
    return g;
}

/// One axis: the kernel taps that land on output index `o`, and the input index
/// each reads — kernel k contributes where o + P - k·D is a non-negative multiple
/// of S. Returns the count (at most K, hence at most kMaxRowTaps). The 3-D
/// scatter runs this per axis, so the walk is not written twice.
inline int64_t tconv2d_axis_taps(int64_t I, int64_t K, int64_t S, int64_t D,
                                 int64_t P, int64_t o, int64_t* k_of, int64_t* i_of)
{
    int64_t taps = 0;
    for (int64_t k = 0; k < K; ++k) {
        const int64_t n = o + P - k * D;
        if (n < 0 || n % S != 0) { continue; }
        const int64_t i = n / S;
        if (i >= I) { continue; }
        k_of[taps] = k;
        i_of[taps] = i;
        ++taps;
    }
    return taps;
}

/// The kernel rows that land on output row `oh`, and the input row each reads.
/// Returns the count (at most KH, hence at most kMaxRowTaps).
inline int64_t tconv2d_row_taps(const Tconv2dColGeom& g, int64_t oh,
                                int64_t* kh_of, int64_t* ih_of)
{
    return tconv2d_axis_taps(g.IH, g.KH, g.SH, g.DH, g.PH, oh, kh_of, ih_of);
}

/// The input span one (kw, residue r) tap contributes to: input columns
/// [shift + tlo, shift + thi) land on output columns r + t·SW, t ∈ [tlo, thi).
struct Tconv2dColSpan {
    int64_t shift = 0;
    int64_t tlo = 0;
    int64_t thi = 0;
};

/// False when the tap misses this residue class entirely. A negative shift is
/// not a miss: pixel t of the class adds t·SW, so the tap starts applying at
/// t = -shift, which is exactly what tlo clamps to.
inline bool tconv2d_col_span(const Tconv2dColGeom& g, int64_t kw, int64_t r,
                             Tconv2dColSpan& s)
{
    const int64_t nd = r + g.PW - kw * g.DW;
    if (nd % g.SW != 0) { return false; }
    s.shift = nd / g.SW;
    const int64_t mic = (g.OW - r + g.SW - 1) / g.SW;
    s.tlo = std::max<int64_t>(0, -s.shift);
    s.thi = std::min<int64_t>(mic, g.IW - s.shift);
    return s.thi > s.tlo;
}

/// Channel-major col: rows are [oc·karea + kh·KW + kw] over the input plane, so
/// `col` points at this oc's first tap row. `acc(dst, src, n, tap)` adds one
/// tap's contiguous run to residue class r of `row`. `tap_stride` is where one
/// tap sits inside the col — the plane size for a real col, zero when the tap
/// only indexes `acc`'s weight (depthwise, whose source is the input itself).
template <typename T, typename Acc>
void tconv2d_col2im_channel_major(const Tconv2dColGeom& g, const T* col, T* row,
                                  int64_t oh, Acc&& acc, int64_t tap_stride)
{
    int64_t kh_of[kMaxRowTaps];
    int64_t ih_of[kMaxRowTaps];
    const int64_t taps = tconv2d_row_taps(g, oh, kh_of, ih_of);

    for (int64_t ti = 0; ti < taps; ++ti) {
        const int64_t tap_row = kh_of[ti] * g.KW;
        const T* base = col + tap_row * tap_stride + ih_of[ti] * g.IW;
        for (int64_t kw = 0; kw < g.KW; ++kw) {
            for (int64_t r = 0; r < g.SW && r < g.OW; ++r) {
                Tconv2dColSpan s;
                if (!tconv2d_col_span(g, kw, r, s)) { continue; }
                acc(row + r * g.mic + s.tlo,
                    base + kw * tap_stride + s.shift + s.tlo, s.thi - s.tlo,
                    tap_row + kw);
            }
        }
    }
}

/// Pixel-major col: rows are [ih·IW + iw] holding [tap][oc]. `m` is a pixel's
/// stride in the col buffer and `nc` the oc vector length. `tap_stride` is
/// where one tap sits inside a pixel — zero when the tap indexes only `acc`'s
/// weight (depthwise, whose source is the input itself and has no oc axis of
/// taps).
template <typename T, typename Acc>
void tconv2d_col2im_pixel_major(const Tconv2dColGeom& g, const T* col, int64_t m,
                                int64_t nc, int64_t tap_stride, T* row,
                                int64_t oh, Acc&& acc)
{
    int64_t kh_of[kMaxRowTaps];
    int64_t ih_of[kMaxRowTaps];
    const int64_t taps = tconv2d_row_taps(g, oh, kh_of, ih_of);

    for (int64_t ti = 0; ti < taps; ++ti) {
        const T* ihrow = col + ih_of[ti] * g.IW * m;
        for (int64_t kw = 0; kw < g.KW; ++kw) {
            const int64_t tap = kh_of[ti] * g.KW + kw;
            const T* colw = ihrow + tap * tap_stride;
            for (int64_t r = 0; r < g.SW && r < g.OW; ++r) {
                Tconv2dColSpan s;
                if (!tconv2d_col_span(g, kw, r, s)) { continue; }
                for (int64_t t = s.tlo; t < s.thi; ++t) {
                    acc(row + (r + t * g.SW) * nc, colw + (t + s.shift) * m, nc, tap);
                }
            }
        }
    }
}

}  // namespace nnops::backend::cpu
