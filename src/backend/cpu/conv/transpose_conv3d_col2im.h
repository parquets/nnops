#pragma once
/// @file transpose_conv3d_col2im.h
/// @brief The col → output pass of transposed 3-D convolution.
///
/// The 2-D fold (transpose_conv2d_col2im.h) with one more axis. Depth and height
/// are two independent tap lists — output (od, oh) reads input (id, ih) wherever
/// od + PD - kd·DD == id·SD and oh + PH - kh·DH == ih·SH — so their product is
/// the set of input slice rows that feed one output row, and every one of them
/// contributes under the *same* W-axis walk the 2-D scatter already does. That
/// is why the column side is shared rather than re-derived: the residue classes
/// and their spans depend on nothing but the W geometry.
///
/// The col row for one (oc, kd, kh, kw) is [npix] wide over the whole input
/// volume, so the slice row a tap reads sits at ((kd·KH + kh)·KW + kw)·npix +
/// id·(IH·IW) + ih·IW — the tap's row, then the depth slice, then the row.

#include "backend/cpu/conv/transpose_conv2d_col2im.h"

#include <cstdint>

namespace nnops::backend::cpu {

/// The window geometry one 3-D scatter reads. The W axis is the 2-D geometry
/// verbatim, so `w.mic`, `w.row_len` and `tconv2d_col_span` apply unchanged.
struct Tconv3dColGeom {
    int64_t ID = 0, IH = 0, IW = 0;
    int64_t OW = 0;
    int64_t KD = 0, KH = 0, KW = 0;
    int64_t SD = 1, SH = 1, SW = 1;
    int64_t DD = 1, DH = 1, DW = 1;
    int64_t PD = 0, PH = 0, PW = 0;
    int64_t plane = 0;  ///< IH·IW — elements per input depth slice
    Tconv2dColGeom w;   ///< the H/W pair, as the 2-D scatter sees it
};

inline Tconv3dColGeom make_col3_geom(int64_t ID, int64_t IH, int64_t IW, int64_t OW,
                                     int64_t KD, int64_t KH, int64_t KW, int64_t SD,
                                     int64_t SH, int64_t SW, int64_t DD, int64_t DH,
                                     int64_t DW, int64_t PD, int64_t PH, int64_t PW)
{
    Tconv3dColGeom g;
    g.ID = ID;
    g.IH = IH;
    g.IW = IW;
    g.OW = OW;
    g.KD = KD;
    g.KH = KH;
    g.KW = KW;
    g.SD = SD;
    g.SH = SH;
    g.SW = SW;
    g.DD = DD;
    g.DH = DH;
    g.DW = DW;
    g.PD = PD;
    g.PH = PH;
    g.PW = PW;
    g.plane = IH * IW;
    g.w = make_col_geom(IH, IW, OW, KH, KW, SH, SW, DH, DW, PH, PW);
    return g;
}

/// Channel-major col: rows are [oc·karea + (kd·KH + kh)·KW + kw] over the input
/// volume, so `col` points at this oc's first tap row. `acc(dst, src, n, tap)`
/// adds one tap's contiguous run to residue class r of `row`. The caller guards
/// KD and KH against kMaxRowTaps.
template <typename T, typename Acc>
void tconv3d_col2im_channel_major(const Tconv3dColGeom& g, const T* col, T* row,
                                  int64_t od, int64_t oh, Acc&& acc)
{
    int64_t kd_of[kMaxRowTaps];
    int64_t id_of[kMaxRowTaps];
    int64_t kh_of[kMaxRowTaps];
    int64_t ih_of[kMaxRowTaps];
    const int64_t dtaps = tconv2d_axis_taps(g.ID, g.KD, g.SD, g.DD, g.PD, od,
                                            kd_of, id_of);
    const int64_t htaps = tconv2d_row_taps(g.w, oh, kh_of, ih_of);
    const int64_t npix = g.ID * g.plane;

    for (int64_t di = 0; di < dtaps; ++di) {
        for (int64_t hi = 0; hi < htaps; ++hi) {
            const int64_t tap_row = (kd_of[di] * g.KH + kh_of[hi]) * g.KW;
            const T* base = col + tap_row * npix + id_of[di] * g.plane + ih_of[hi] * g.IW;
            for (int64_t kw = 0; kw < g.KW; ++kw) {
                for (int64_t r = 0; r < g.SW && r < g.OW; ++r) {
                    Tconv2dColSpan s;
                    if (!tconv2d_col_span(g.w, kw, r, s)) { continue; }
                    acc(row + r * g.w.mic + s.tlo,
                        base + kw * npix + s.shift + s.tlo, s.thi - s.tlo,
                        tap_row + kw);
                }
            }
        }
    }
}

/// Pixel-major col: voxel rows are [id·(IH·IW) + ih·IW + iw] holding [tap][oc],
/// which is the orientation the channel-last GEMM leaves the col in. `m` is a
/// voxel's stride in `col` and `nc` the oc vector length; `tap_stride` is where
/// one tap sits inside a voxel — zero for depthwise, whose source is the input
/// volume itself and has no oc axis of taps, only a weight per tap.
template <typename T, typename Acc>
void tconv3d_col2im_pixel_major(const Tconv3dColGeom& g, const T* col, int64_t m,
                                int64_t nc, int64_t tap_stride, T* row,
                                int64_t od, int64_t oh, Acc&& acc)
{
    int64_t kd_of[kMaxRowTaps];
    int64_t id_of[kMaxRowTaps];
    int64_t kh_of[kMaxRowTaps];
    int64_t ih_of[kMaxRowTaps];
    const int64_t dtaps = tconv2d_axis_taps(g.ID, g.KD, g.SD, g.DD, g.PD, od,
                                            kd_of, id_of);
    const int64_t htaps = tconv2d_row_taps(g.w, oh, kh_of, ih_of);

    for (int64_t di = 0; di < dtaps; ++di) {
        for (int64_t hi = 0; hi < htaps; ++hi) {
            const int64_t tap_row = (kd_of[di] * g.KH + kh_of[hi]) * g.KW;
            const T* slice = col + (id_of[di] * g.plane + ih_of[hi] * g.IW) * m;
            for (int64_t kw = 0; kw < g.KW; ++kw) {
                const int64_t tap = tap_row + kw;
                const T* colw = slice + tap * tap_stride;
                for (int64_t r = 0; r < g.SW && r < g.OW; ++r) {
                    Tconv2dColSpan s;
                    if (!tconv2d_col_span(g.w, kw, r, s)) { continue; }
                    for (int64_t t = s.tlo; t < s.thi; ++t) {
                        acc(row + (r + t * g.SW) * nc, colw + (t + s.shift) * m, nc,
                            tap);
                    }
                }
            }
        }
    }
}

}  // namespace nnops::backend::cpu
