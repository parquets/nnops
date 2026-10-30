#pragma once
/// @file simd_depthwise_conv.hpp
/// @brief SIMD kernels for 2D/3D depthwise convolution, planar and channel-last.
///
/// Two drivers, split by the axis the lane takes.
///
/// Channel-last (NHWC/NDHWC) reads the raw HWCM filter [KD][KH][KW][C] — TF's
/// depthwise order with the multiplier dropped — and puts the lane on the channel.
/// The pixel stride and the tap stride are both the channel count, set by the
/// driver; no prepack is involved, because the layout already is the access
/// pattern. A call covers NV adjacent channel vectors, 2 where the channels allow,
/// since a 64-byte line holds two f32 vectors and a one-vector call leaves the
/// other half of every line to a later pass — 2x the line traffic for the same
/// arithmetic.
///
/// Planar (NCHW/NCDHW) reads the raw [C, 1, (KD,) KH, KW] filter — one scalar per
/// tap, broadcast into the accumulate — and puts the lane on the output width.
///
/// Both run R = 4 output rows per call, R = 1 for a tail. Planar needs it for ILP
/// alone: one row's taps are a chain of KD*KH*KW dependent FMAs with nothing to
/// interleave, and four rows are four independent chains. A KD of 1 — the 2D
/// case — degenerates the depth loop, and padding is bounds-checked in-kernel.
///
/// Everything that depends only on the tap is hoisted out of the tap loops: a row
/// pointer per (kd, kh, r), left-pad-biased on the channel-last path, and an
/// interior span of output positions whose every tap is inside the row, where the
/// tap loop takes no bounds test and no null test — one branch per position instead
/// of one per tap.
/// The null test (does this input row exist) can only be skipped when every row of
/// the group does, so a group whose window reaches past the plane keeps the
/// checked path.
///
/// Three shapes were measured against this and lost: peeling the padded rows out
/// of the groups costs a quarter of the ILP per row and ran 14-22% slower; two
/// output positions per iteration share the weight vectors but need eight live
/// accumulators and reached 24 GFLOPS against 35; a channel-last column sweep ran
/// 1.4-1.9x slower at every shape, 1.4x even at C = 8 where only the kernel
/// differs.
///
/// The planar row table is gated on a row spanning at least two width vectors —
/// at one it costs more to fill than the arithmetic it saves (1T GFLOPS, 3x3: two
/// vectors 14.0 vs 14.3, four 18.5 vs 13.1, a 112-wide plane 28.3 vs 13.8). Its
/// edge vectors take a lane mask instead, folded into the tap's broadcast weight;
/// those lanes are real tensor neighbours, not zeros, so a non-finite value there
/// would reach the FMA as 0 * NaN, and the driver nulls the mask for the tensor's
/// first and last channel block.
///
/// The channel-last 3x3 kernels hold the nine taps in registers —
/// `dwconv_last3x3` for one channel vector, `dwconv_last3x3_16` for two, which
/// cannot hoist eighteen tap vectors and reloads one per column to feed four rows
/// and both halves. Sixteen channels is what the driver asks for: a pixel's 16
/// f32 is one 64-byte line, so a sweep touches every line once, where an
/// 8-channel block pays two sweeps for the same bytes.
///
/// Quantized channel-last dequantizes this block's taps once per call into a dense
/// staging block (9 vectors for a 3x3 against the group's thousands of FMAs), then
/// runs `dwconv_rows` on the f32 weights; a pad lane gets scale 0.

#include "nnops/detail/simd/simd.hpp"
#include "nnops/core/epilogue.hpp"
#include "simd_epilogue.hpp"
#include "simd_access.hpp"

#include <cstdint>
#include <cstring>

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
    int64_t ow0 = 0;              // first output column this call covers
    int64_t ow_end = -1;          // one past the last; negative means OW
    int64_t in_d_stride;          // input depth stride (elems)
    int64_t in_row_stride;        // input row stride (elems)
    int64_t pix_stride = 8;       // elems between pixels (= C, set by the driver)
    int64_t w_tap_stride = 8;     // elems between taps (= C of a raw HWCM filter,
                                  // 8 in the dequantized staging block)
};

/// R output rows per call; W is the weight type and Tag picks the epilogue overload.
template <int R, int NV, typename W, typename Tag, typename Src, typename Dst, typename V>
inline void dwconv_rows(const DwConvParams& p, const W* w, const Tag* tag,
                        const Epilogue& epilogue, const Src (&srcs)[NV],
                        const Dst (&dsts)[NV], const V (&vbias)[NV])
{
    using InPtr = const typename Src::elem*;
    const int64_t pix = p.pix_stride;
    const int64_t wtap = p.w_tap_stride;
    constexpr int64_t kVec = kChannelLanes;
    constexpr int64_t kMaxTapRows = 256;  // (kd, kh, r): 8x8 with R = 4 fits

    // The row table is built from vector 0's base, so vector k adds k*kVec to it.
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

    // Interior columns: every tap offset ow*SW + kw*DW - PW lands in [0, IW).
    const int64_t ow_lo = (p.PW + p.SW - 1) / p.SW;
    const int64_t last_hi = p.IW - 1 + p.PW - (p.KW - 1) * p.DW;
    const int64_t ow_hi = (last_hi >= 0) ? last_hi / p.SW : -1;

    // Row pointers have already absorbed -PW*pix, so a valid tap offset is
    // [PW, IW + PW).
    const int64_t off_ok_lo = p.PW;
    const int64_t off_ok_hi = p.IW + p.PW;

    const int64_t ow_end = (p.ow_end >= 0) ? p.ow_end : p.OW;
    for (int64_t ow = p.ow0; ow < ow_end; ++ow) {
        V acc[R][NV];
        for (int r = 0; r < R; ++r) {
            for (int k = 0; k < NV; ++k) { acc[r][k] = vbias[k]; }
        }

        const bool interior = all_rows && ow >= ow_lo && ow <= ow_hi;
        const int64_t ow_off = ow * p.SW;

        for (int64_t kd = 0; kd < p.KD; ++kd) {
            for (int64_t kh = 0; kh < p.KH; ++kh) {
                const W* w_row = w + (kd * p.KH + kh) * p.KW * wtap;
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
                            const V vk = v_load(w_row + kw * wtap + k * kVec);
                            for (int r = 0; r < R; ++r) {
                                acc[r][k] = v_fmadd(srcs[k].load_ptr(row[r] + at + k * kVec), vk,
                                                    acc[r][k]);
                            }
                        }
                    }
                    continue;
                }

                for (int64_t kw = 0; kw < p.KW; ++kw) {
                    const int64_t off = ow_off + kw * p.DW;
                    if (off < off_ok_lo || off >= off_ok_hi) {
                        continue;
                    }
                    const int64_t at = off * pix;
                    for (int k = 0; k < NV; ++k) {
                        const V vk = v_load(w_row + kw * wtap + k * kVec);
                        for (int r = 0; r < R; ++r) {
                            if (row[r] == nullptr) {
                                continue;
                            }
                            acc[r][k] = v_fmadd(srcs[k].load_ptr(row[r] + at + k * kVec), vk,
                                                acc[r][k]);
                        }
                    }
                }
            }
        }

        for (int r = 0; r < R; ++r) {
            if (p.oh + r >= p.OH) {
                continue;
            }
            for (int k = 0; k < NV; ++k) {
                dsts[k].store(r, ow,
                              nnops::backend::cpu::apply_epilogue_vec(epilogue, tag,
                                                                      acc[r][k]));
            }
        }
    }
}

// ============================================================
// Planar (NCHW/NCDHW) driver — the SIMD lane spans output width
// ============================================================

/// Lane masks for the plane's two edge width vectors.
struct DwPlanarEdges {
    const void* masks = nullptr;      ///< [2][KW][L] elements of T
    const uint8_t* needed = nullptr;  ///< [2][KW]
    int64_t kw = 0;
    int64_t last_ow = 0;              ///< ow of the final width vector
};

/// Fill `masks` ([2*MaxKw*L] elements of T) and `needed` ([2*MaxKw]) from `p`.
/// `masks == nullptr` is the general path; the buffers stay the caller's.
template <int64_t MaxKw, typename T>
inline DwPlanarEdges dwconv_planar_edge_masks(const DwConvParams& p, T* masks, uint8_t* needed)
{
    constexpr int64_t L = simd_lane_for<T>;
    DwPlanarEdges edges;
    if ((p.SW != 1 && p.SW != 2) || p.KW > MaxKw) {
        return edges;
    }

    const int64_t last_ow = ((p.OW + L - 1) / L - 1) * L;
    // One vector is both edges; e resolves to 0 for it and the two masks below
    // come out identical.
    const int64_t ow_edge[2] = {0, last_ow};
    for (int64_t e = 0; e < 2; ++e) {
        for (int64_t kw = 0; kw < p.KW; ++kw) {
            T* mk = masks + (e * p.KW + kw) * L;
            bool all = true;
            for (int64_t l = 0; l < L; ++l) {
                const int64_t col = (ow_edge[e] + l) * p.SW + kw * p.DW - p.PW;
                const bool ok = col >= 0 && col < p.IW;
                mk[l] = ok ? static_cast<T>(1) : static_cast<T>(0);
                all = all && ok;
            }
            needed[e * p.KW + kw] = all ? 0 : 1;
        }
    }

    edges.masks   = masks;
    edges.needed  = needed;
    edges.kw      = p.KW;
    edges.last_ow = last_ow;
    return edges;
}

/// Copy `n` elements, 1 <= n < L, as one switch-selected block move.
template <typename T>
inline void dwconv_copy_prefix(T* dst, const T* src, int64_t n)
{
    switch (n) {
        case 1: std::memcpy(dst, src, 1 * sizeof(T)); break;
        case 2: std::memcpy(dst, src, 2 * sizeof(T)); break;
        case 3: std::memcpy(dst, src, 3 * sizeof(T)); break;
        case 4: std::memcpy(dst, src, 4 * sizeof(T)); break;
        case 5: std::memcpy(dst, src, 5 * sizeof(T)); break;
        case 6: std::memcpy(dst, src, 6 * sizeof(T)); break;
        case 7: std::memcpy(dst, src, 7 * sizeof(T)); break;
        case 8: std::memcpy(dst, src, 8 * sizeof(T)); break;
        default: break;
    }
}

/// R output rows per call, one width-vector accumulate per row. @p SW is the width
/// stride when the caller knows it (1 or 2) and 0 when it does not, so the choice
/// stays out of the tap loop.
template <int R, int SW, typename T>
inline void dwconv_planar_rows(T* out, const T* in, const T* w, float bias,
                               const DwConvParams& p, const Epilogue& epilogue,
                               bool add_to, const DwPlanarEdges& edges)
{
    constexpr int64_t L = simd_lane_for<T>;
    const T zero = static_cast<T>(0);
    const auto v_bias = v_set1(out, bias);

    const int64_t sw = (SW != 0) ? SW : p.SW;
    const auto load_row = [&](const T* src) {
        if constexpr (SW == 1) {
            return v_load(src);
        } else if constexpr (SW == 2) {
            return v_load_even(src);
        } else {
            return (p.SW == 1) ? v_load(src) : v_load_even(src);
        }
    };

    // Every input row this call touches, one entry per (kd, kh, r), null where
    // the row does not exist; a table read replaces address arithmetic that
    // would otherwise be paid per output vector.
    constexpr int64_t kMaxTapRows = 256;   // (kd, kh, r): 8x8 with R=4 fits
    const T* row_tab[kMaxTapRows];
    // Worth building only when a row spans more than one width vector
    // (measured 1T GFLOPS, 3x3: two vectors 14.0 vs 14.3, four 18.5 vs 13.1,
    // a 112-wide plane 28.3 vs 13.8).
    const int64_t nvec = (p.OW + L - 1) / L;
    const bool have_tab = (p.KD * p.KH * R <= kMaxTapRows) && (nvec >= 2);
    bool all_rows = have_tab;
    if (have_tab) {
        int64_t t = 0;
        for (int64_t kd = 0; kd < p.KD; ++kd) {
            const int64_t id = p.od * p.SD + kd * p.DD - p.PD;
            const T* in_d = in + id * p.in_d_stride;
            const bool d_ok = (id >= 0 && id < p.ID);
            for (int64_t kh = 0; kh < p.KH; ++kh) {
                for (int r = 0; r < R; ++r) {
                    const int64_t ih = (p.oh + r) * p.SH + kh * p.DH - p.PH;
                    const bool ok = d_ok && (p.oh + r < p.OH) &&
                                    ih >= 0 && ih < p.IH;
                    row_tab[t++] = ok ? in_d + ih * p.in_row_stride : nullptr;
                    all_rows = all_rows && ok;
                }
            }
        }
    }

    // A stride-2 vector reads 2L inputs to produce L outputs, so the interior
    // guard widens with it — the same bound planar_load uses.
    const int64_t span  = (sw == 2) ? 2 * L : L;
    const int64_t need  = (p.KW - 1) * p.DW + span;    // past the first column
    const int64_t ow_lo = (p.PW + sw - 1) / sw;        // first ow with iw >= 0
    const int64_t ow_hi = (p.IW + p.PW - need >= 0)
                              ? (p.IW + p.PW - need) / sw : -1;

    for (int64_t ow = 0; ow < p.OW; ow += L) {
        decltype(v_set1(out, bias)) acc[R];
        for (int r = 0; r < R; ++r) { acc[r] = v_bias; }

        // Every lane of every tap inside the row: no bounds test, no scalar fill.
        const bool interior = all_rows && ow >= ow_lo && ow <= ow_hi;

        // Which edge vector this is: 0 first, 1 last, -1 neither.
        const int64_t e = (ow == 0) ? 0 : ((ow == edges.last_ow) ? 1 : -1);

        for (int64_t kd = 0; kd < p.KD; ++kd) {
            for (int64_t kh = 0; kh < p.KH; ++kh) {
                const T* row[R];
                if (have_tab) {
                    const int64_t t = (kd * p.KH + kh) * R;
                    for (int r = 0; r < R; ++r) { row[r] = row_tab[t + r]; }
                } else {
                    const int64_t id = p.od * p.SD + kd * p.DD - p.PD;
                    if (id < 0 || id >= p.ID) { continue; }
                    const T* in_d = in + id * p.in_d_stride;
                    for (int r = 0; r < R; ++r) {
                        const int64_t ih = (p.oh + r) * p.SH + kh * p.DH - p.PH;
                        const bool ok = (p.oh + r < p.OH) && ih >= 0 && ih < p.IH;
                        row[r] = ok ? in_d + ih * p.in_row_stride : nullptr;
                    }
                }
                const T* w_row = w + (kd * p.KH + kh) * p.KW;

                if (interior) {
                    for (int64_t kw = 0; kw < p.KW; ++kw) {
                        const int64_t iw0 = ow * sw + kw * p.DW - p.PW;
                        const auto vk = v_set1(out, s_load(&w_row[kw]));
                        for (int r = 0; r < R; ++r) {
                            acc[r] = v_fmadd(load_row(row[r] + iw0), vk, acc[r]);
                        }
                    }
                    continue;
                }

                // The precomputed mask is what keeps a narrow plane — where
                // every vector is an edge — out of the scalar fallback below.
                if (edges.masks != nullptr && e >= 0) {
                    for (int64_t kw = 0; kw < p.KW; ++kw) {
                        const int64_t iw0 = ow * sw + kw * p.DW - p.PW;
                        // The mask zeroes exactly the lanes this tap reads from
                        // outside the row and is the same for every row, so it
                        // folds into the broadcast weight once per tap.
                        auto vk = v_set1(out, s_load(&w_row[kw]));
                        if (edges.needed[e * edges.kw + kw]) {
                            vk = v_mul(vk, v_load(static_cast<const T*>(edges.masks) +
                                                  (e * edges.kw + kw) * L));
                        }
                        for (int r = 0; r < R; ++r) {
                            if (row[r] == nullptr) { continue; }
                            acc[r] = v_fmadd(load_row(row[r] + iw0), vk, acc[r]);
                        }
                    }
                    continue;
                }

                for (int64_t kw = 0; kw < p.KW; ++kw) {
                    const int64_t iw0 = ow * sw + kw * p.DW - p.PW;
                    const auto vk = v_set1(out, s_load(&w_row[kw]));
                    for (int r = 0; r < R; ++r) {
                        if (row[r] == nullptr) { continue; }
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
                // The short last vector is a block copy of `lanes` elements, not
                // a per-lane loop: on a 7-wide plane it is the whole row's
                // store, and the loop cost more than the nine taps above it.
                alignas(32) T tmp[L];
                if (add_to) {
                    alignas(32) T cur[L] = {};
                    dwconv_copy_prefix<T>(cur, dst, lanes);
                    v_store(tmp, v_add(v, v_load(cur)));
                } else {
                    v_store(tmp, v);
                }
                dwconv_copy_prefix<T>(dst, tmp, lanes);
            }
        }
    }
}

// ============================================================
// Public kernels
// ============================================================

template <typename T>
inline void dwconv_planar_h4(T* out, const T* in, const T* w, float bias,
                             const DwConvParams& p, const Epilogue& epilogue,
                             bool add_to, const DwPlanarEdges& edges)
{
    switch (p.SW) {
        case 1:  dwconv_planar_rows<4, 1>(out, in, w, bias, p, epilogue, add_to, edges); break;
        case 2:  dwconv_planar_rows<4, 2>(out, in, w, bias, p, epilogue, add_to, edges); break;
        default: dwconv_planar_rows<4, 0>(out, in, w, bias, p, epilogue, add_to, edges); break;
    }
}

/// One row — the tail of an h4 loop, and the whole kernel for a single-row call.
template <typename T>
inline void dwconv_planar(T* out, const T* in, const T* w, float bias,
                          const DwConvParams& p, const Epilogue& epilogue,
                          bool add_to, const DwPlanarEdges& edges)
{
    switch (p.SW) {
        case 1:  dwconv_planar_rows<1, 1>(out, in, w, bias, p, epilogue, add_to, edges); break;
        case 2:  dwconv_planar_rows<1, 2>(out, in, w, bias, p, epilogue, add_to, edges); break;
        default: dwconv_planar_rows<1, 0>(out, in, w, bias, p, epilogue, add_to, edges); break;
    }
}

// ============================================================
// Channel-last (NHWC/NDHWC) entry points
// ============================================================
//
// out_rows and in_img are the bases advanced by the channel block, w_tap0 the
// block's first channel of the tap-major weight, and lanes how many channels the
// call covers — up to 16, a whole 64-byte line of f32. A 3x3 takes
// dwconv_last3x3 at one vector and dwconv_last3x3_16 at two.

/// One depth plane, 3x3, no dilation, vertical stride 1 or 2 — a larger one would
/// still fit one row table but is not common enough to instantiate.
inline bool dwconv_last3x3_ok(const DwConvParams& p)
{
    return p.KD == 1 && p.KH == 3 && p.KW == 3 && p.DH == 1 && p.DW == 1 &&
           (p.SH == 1 || p.SH == 2);
}

/// One channel vector of a 3x3 filter, R output rows per call, with the nine tap
/// vectors held in registers for the whole width sweep — the generic kernel
/// reloads them every column, and the tap plus accumulator count is 13 of the 16
/// ymm registers, which is why the two-vector variant cannot do the same.
///
/// @p SH decides the table: output row r with filter row kh is input row
/// `oh*SH + r*SH + kh - PH`, so there are R*SH + 2 entries and a tap's row is
/// `row[r*SH + kh]` — compile-time indices. The horizontal stride stays runtime.
template <int R, int SH, typename T>
inline void dwconv_last3x3(T* out_rows, const T* in_img, const T* w_tap0,
                           const T* bias_vec, const DwConvParams& p,
                           const Epilogue& epilogue, bool add_to)
{
    using V = decltype(v_load(in_img));
    const int64_t pix = p.pix_stride;
    const int64_t wtap = p.w_tap_stride;
    const int64_t ow_end = (p.ow_end >= 0) ? p.ow_end : p.OW;

    const V vbias = bias_vec ? v_load(bias_vec) : v_zero(in_img);
    V wt[9];
    for (int t = 0; t < 9; ++t) { wt[t] = v_load(w_tap0 + t * wtap); }

    // One row pointer per distinct input row of the group, biased by the left
    // pad so a tap's address is `row + (ow*SW + kw)*pix`.
    constexpr int64_t kRows = R * SH + 2;
    const T* row[kRows];
    bool all_rows = true;
    for (int64_t i = 0; i < kRows; ++i) {
        const int64_t ih = p.oh * SH + i - p.PH;
        const bool ok = ih >= 0 && ih < p.IH;
        row[i] = ok ? in_img + ih * p.in_row_stride - p.PW * pix : nullptr;
        all_rows = all_rows && ok;
    }

    const int64_t ow_lo = (p.PW + p.SW - 1) / p.SW;
    const int64_t last_hi = p.IW - 1 + p.PW - 2 * p.DW;
    const int64_t ow_hi = (last_hi >= 0) ? last_hi / p.SW : -1;
    const int64_t off_ok_lo = p.PW;
    const int64_t off_ok_hi = p.IW + p.PW;

    for (int64_t ow = p.ow0; ow < ow_end; ++ow) {
        V acc[R];
        for (int r = 0; r < R; ++r) { acc[r] = vbias; }

        if (all_rows && ow >= ow_lo && ow <= ow_hi) {
            const int64_t o = ow * p.SW * pix;
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    const int64_t at = o + kw * pix;
                    for (int r = 0; r < R; ++r) {
                        acc[r] = v_fmadd(v_load(row[r * SH + kh] + at),
                                         wt[kh * 3 + kw], acc[r]);
                    }
                }
            }
        } else {
            const int64_t ow_off = ow * p.SW;
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    const int64_t off = ow_off + kw;
                    if (off < off_ok_lo || off >= off_ok_hi) { continue; }
                    const int64_t at = off * pix;
                    for (int r = 0; r < R; ++r) {
                        if (row[r * SH + kh] == nullptr) { continue; }
                        acc[r] = v_fmadd(v_load(row[r * SH + kh] + at),
                                         wt[kh * 3 + kw], acc[r]);
                    }
                }
            }
        }

        for (int r = 0; r < R; ++r) {
            if (p.oh + r >= p.OH) { continue; }
            T* dst = out_rows + r * p.out_row_stride + ow * pix;
            v_store_add(dst, nnops::backend::cpu::apply_epilogue_vec(epilogue,
                                                                     out_rows, acc[r]),
                        add_to);
        }
    }
}

/// The two-vector 3x3: sixteen channels, the same row table and eight
/// accumulators, with the tap vectors loaded per column inside the width loop —
/// eighteen of them plus eight accumulators is twenty-six registers — but one
/// weight load then feeds four rows and both halves, so a tap is ten loads for
/// eight FMAs.
template <int R, int SH, typename T>
inline void dwconv_last3x3_16(T* out_rows, const T* in_img, const T* w_tap0,
                              const T* bias_vec, const DwConvParams& p,
                              const Epilogue& epilogue, bool add_to)
{
    using V = decltype(v_load(in_img));
    constexpr int64_t kL = kChannelLanes;
    const int64_t pix = p.pix_stride;
    const int64_t wtap = p.w_tap_stride;
    const int64_t ow_end = (p.ow_end >= 0) ? p.ow_end : p.OW;

    const V vbias[2] = {bias_vec ? v_load(bias_vec) : v_zero(in_img),
                        bias_vec ? v_load(bias_vec + kL) : v_zero(in_img)};

    constexpr int64_t kRows = R * SH + 2;
    const T* row[kRows];
    bool all_rows = true;
    for (int64_t i = 0; i < kRows; ++i) {
        const int64_t ih = p.oh * SH + i - p.PH;
        const bool ok = ih >= 0 && ih < p.IH;
        row[i] = ok ? in_img + ih * p.in_row_stride - p.PW * pix : nullptr;
        all_rows = all_rows && ok;
    }

    const int64_t ow_lo = (p.PW + p.SW - 1) / p.SW;
    const int64_t last_hi = p.IW - 1 + p.PW - 2 * p.DW;
    const int64_t ow_hi = (last_hi >= 0) ? last_hi / p.SW : -1;
    const int64_t off_ok_lo = p.PW;
    const int64_t off_ok_hi = p.IW + p.PW;

    for (int64_t ow = p.ow0; ow < ow_end; ++ow) {
        V acc[R][2];
        for (int r = 0; r < R; ++r) { acc[r][0] = vbias[0]; acc[r][1] = vbias[1]; }

        if (all_rows && ow >= ow_lo && ow <= ow_hi) {
            const int64_t o = ow * p.SW * pix;
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    const int64_t at = o + kw * pix;
                    const T* wp = w_tap0 + (kh * 3 + kw) * wtap;
                    const V w0 = v_load(wp);
                    const V w1 = v_load(wp + kL);
                    for (int r = 0; r < R; ++r) {
                        const T* ip = row[r * SH + kh] + at;
                        acc[r][0] = v_fmadd(v_load(ip), w0, acc[r][0]);
                        acc[r][1] = v_fmadd(v_load(ip + kL), w1, acc[r][1]);
                    }
                }
            }
        } else {
            const int64_t ow_off = ow * p.SW;
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    const int64_t off = ow_off + kw;
                    if (off < off_ok_lo || off >= off_ok_hi) { continue; }
                    const int64_t at = off * pix;
                    const T* wp = w_tap0 + (kh * 3 + kw) * wtap;
                    const V w0 = v_load(wp);
                    const V w1 = v_load(wp + kL);
                    for (int r = 0; r < R; ++r) {
                        if (row[r * SH + kh] == nullptr) { continue; }
                        const T* ip = row[r * SH + kh] + at;
                        acc[r][0] = v_fmadd(v_load(ip), w0, acc[r][0]);
                        acc[r][1] = v_fmadd(v_load(ip + kL), w1, acc[r][1]);
                    }
                }
            }
        }

        for (int r = 0; r < R; ++r) {
            if (p.oh + r >= p.OH) { continue; }
            T* dst = out_rows + r * p.out_row_stride + ow * pix;
            v_store_add(dst, nnops::backend::cpu::apply_epilogue_vec(
                                 epilogue, out_rows, acc[r][0]), add_to);
            v_store_add(dst + kL, nnops::backend::cpu::apply_epilogue_vec(
                                      epilogue, out_rows, acc[r][1]), add_to);
        }
    }
}

/// The 8-channel call: pick the kernel for the shape, then the vertical stride.
template <int R, typename T>
inline void dwconv_last8(T* out_rows, const T* in_img, const T* w_tap0,
                         const T* bias_vec, const DwConvParams& p,
                         const Epilogue& epilogue, bool add_to)
{
    if (dwconv_last3x3_ok(p)) {
        if (p.SH == 1) {
            dwconv_last3x3<R, 1>(out_rows, in_img, w_tap0, bias_vec, p, epilogue,
                                 add_to);
        } else {
            dwconv_last3x3<R, 2>(out_rows, in_img, w_tap0, bias_vec, p, epilogue,
                                 add_to);
        }
        return;
    }
    const FpSrc<T, true> src[1] = {FpSrc<T, true>{in_img, kChannelLanes}};
    const FpDst<T, true> dst[1] = {FpDst<T, true>{out_rows, p.out_row_stride,
                                                  add_to, p.pix_stride,
                                                  kChannelLanes}};
    const auto vb = bias_vec ? v_load(bias_vec) : v_zero(out_rows);
    const decltype(vb) vbias[1] = {vb};
    dwconv_rows<R, 1>(p, w_tap0, out_rows, epilogue, src, dst, vbias);
}

template <int R, typename T>
inline void dwconv_last(T* out_rows, const T* in_img, const T* w_tap0,
                        const T* bias_vec, const DwConvParams& p,
                        const Epilogue& epilogue, bool add_to, int64_t lanes)
{
    if (lanes > kChannelLanes) {
        if (lanes == 2 * kChannelLanes && dwconv_last3x3_ok(p)) {
            if (p.SH == 1) {
                dwconv_last3x3_16<R, 1>(out_rows, in_img, w_tap0, bias_vec, p,
                                        epilogue, add_to);
            } else {
                dwconv_last3x3_16<R, 2>(out_rows, in_img, w_tap0, bias_vec, p,
                                        epilogue, add_to);
            }
            return;
        }
        const auto vb0 = bias_vec ? v_load(bias_vec) : v_zero(out_rows);
        const auto vb1 = bias_vec ? v_load(bias_vec + kChannelLanes) : v_zero(out_rows);
        const decltype(vb0) vbias[2] = {vb0, vb1};
        const FpSrc<T, true> src[2] = {
            FpSrc<T, true>{in_img, kChannelLanes},
            FpSrc<T, true>{in_img + kChannelLanes, kChannelLanes}};
        const FpDst<T, true> dst[2] = {
            FpDst<T, true>{out_rows, p.out_row_stride, add_to, p.pix_stride,
                           kChannelLanes},
            FpDst<T, true>{out_rows + kChannelLanes, p.out_row_stride, add_to,
                           p.pix_stride, kChannelLanes}};
        dwconv_rows<R, 2>(p, w_tap0, out_rows, epilogue, src, dst, vbias);
        return;
    }
    const auto vb = bias_vec ? load_lanes(bias_vec, lanes) : v_zero(out_rows);
    const decltype(vb) vbias[1] = {vb};
    if (lanes == kChannelLanes) {
        dwconv_last8<R>(out_rows, in_img, w_tap0, bias_vec, p, epilogue, add_to);
    } else {
        const FpSrc<T, false> src[1] = {FpSrc<T, false>{in_img, lanes}};
        const FpDst<T, false> dst[1] = {
            FpDst<T, false>{out_rows, p.out_row_stride, add_to, p.pix_stride, lanes}};
        dwconv_rows<R, 1>(p, w_tap0, out_rows, epilogue, src, dst, vbias);
    }
}

constexpr int64_t kMaxLastQuantTaps = 9 * 9 * 9;  // 729

/// Dequantize this block's taps once per call into a staging block, then run the
/// f32 kernel. Pad lanes get scale 0.
template <int R, typename InT, typename OutT>
inline void dwconv_last_quant(OutT* out_rows, const InT* in_img, const InT* w_tap0,
                              const float* w_scale, const int32_t* w_zero,
                              const float* bias_vec, const DwConvParams& p,
                              const Epilogue& epilogue, bool add_to, int64_t lanes,
                              float in_scale, float in_zero,
                              float out_scale, float out_zero)
{
    const int64_t taps = p.KD * p.KH * p.KW;
    if (taps > kMaxLastQuantTaps) { return; }   // the driver rejects this shape
    const int64_t C = p.pix_stride;

    alignas(32) float w_scale8[kChannelLanes];
    alignas(32) float w_zero8[kChannelLanes];
    for (int64_t l = 0; l < kChannelLanes; ++l) {
        const bool live = (l < lanes) && (w_scale != nullptr);
        w_scale8[l] = live ? w_scale[l] : 0.0f;
        w_zero8[l]  = (live && w_zero != nullptr)
                          ? static_cast<float>(w_zero[l]) : 0.0f;
    }

    alignas(32) float wq[kMaxLastQuantTaps * kChannelLanes];
    for (int64_t t = 0; t < taps; ++t) {
        InT wt[kChannelLanes] = {};
        std::memcpy(wt, w_tap0 + t * C, static_cast<size_t>(lanes) * sizeof(InT));
        v_store(&wq[t * kChannelLanes], dequantize_channel(wt, w_scale8, w_zero8));
    }

    // The staging block is dense 8-lane taps, so the tap stride is the vector
    // width even though the pixel stride is the channel count.
    DwConvParams pq = p;
    pq.w_tap_stride = kChannelLanes;

    const v_f32x8 vbias[1] = {bias_vec ? load_lanes(bias_vec, lanes)
                                       : v_zero_f32x8()};
    if (lanes == kChannelLanes) {
        const QuantSrc<InT, true> src[1] = {QuantSrc<InT, true>{in_img, in_scale,
                                                                in_zero, lanes}};
        const QuantDst<OutT, true> dst[1] = {
            QuantDst<OutT, true>{out_rows, p.out_row_stride, out_scale, out_zero,
                                 add_to, p.pix_stride, lanes}};
        dwconv_rows<R, 1>(pq, wq, &out_scale, epilogue, src, dst, vbias);
    } else {
        const QuantSrc<InT, false> src[1] = {QuantSrc<InT, false>{in_img, in_scale,
                                                                  in_zero, lanes}};
        const QuantDst<OutT, false> dst[1] = {
            QuantDst<OutT, false>{out_rows, p.out_row_stride, out_scale, out_zero,
                                  add_to, p.pix_stride, lanes}};
        dwconv_rows<R, 1>(pq, wq, &out_scale, epilogue, src, dst, vbias);
    }
}

}  // namespace nnops::kernel
