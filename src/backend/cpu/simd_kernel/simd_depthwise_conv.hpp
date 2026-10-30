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
    int64_t pix_stride = 8;       // elems between adjacent pixels: 8 for a
                                  // channel-packed block, the channel count for
                                  // a channel-last plane
    int64_t w_tap_stride = 8;     // elems between adjacent filter taps: 8 for a
                                  // packed [C8, KH, KW, 8] block, the channel
                                  // count for a raw [KH][KW][C] filter (and 8
                                  // for the dequantized staging block built
                                  // from one)
};

/// R output rows per call. W is the packed-weight element type; Tag only picks
/// the epilogue's f32/f16 overload. Both are deduced by the wrappers below.
///
/// The width loop is the innermost one, so everything that depends only on the
/// tap — the input row it reads, the output position it lands on — is hoisted
/// out of it, the way the planar sibling hoists its row table and the MLAS
/// NCHWc kernel peels its padding:
///
///   * one row pointer per (kd, kh, r), built once per call and already biased
///     by the left pad, so a tap's address is `row + ow*SW + kw*DW` with no
///     index arithmetic and no 64-bit product left in the loop;
///   * an interior span of output positions whose every tap is inside the
///     input row, where the tap loop takes no bounds test and no null test at
///     all — one branch per position instead of one per tap.
///
/// The null test is what the border costs: it can only be skipped when every
/// row of the group exists, so the first and last row group of a plane (and the
/// short group at its bottom) keep the checked path.
///
/// Two things were measured against this shape and lost. Peeling the padded
/// rows out of the groups (a one-row schedule above and below the interior
/// span) costs a quarter of the ILP per row and made every stage 14-22%
/// slower. Running two output positions per iteration shares the weight
/// vectors but needs eight live accumulators, and either the register file or
/// this compiler's allocation gives it back — 24 GFLOPS against 35.
template <int R, int NV, typename W, typename Tag, typename Src, typename Dst, typename V>
inline void dwconv_rows(const DwConvParams& p, const W* w, const Tag* tag,
                        const Epilogue& epilogue, const Src (&srcs)[NV],
                        const Dst (&dsts)[NV], const V (&vbias)[NV])
{
    using InPtr = const typename Src::elem*;
    // The pixel and the filter tap are both "one vector of channels" apart in
    // the packed layout, and both "the channel count" apart in a channel-last
    // plane, so the two strides are all that separate the two routes.
    //
    // NV is how many adjacent channel vectors of one pixel a call covers. The
    // packed route is always 1 (a block is one vector, and its planes are
    // dense); a channel-last plane wants 2, because a 64-byte cache line holds
    // two vectors and a single-vector call leaves the other half of every line
    // to a later pass — 2x the line traffic for the same arithmetic. Because
    // the row table is built from vector 0 and the access policies load the
    // address they are handed, vector k's channels are added here.
    const int64_t pix = p.pix_stride;
    const int64_t wtap = p.w_tap_stride;
    constexpr int64_t kVec = kChannelLanes;
    constexpr int64_t kMaxTapRows = 256;  // (kd, kh, r): 8x8 with R = 4 fits

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

    // The output positions whose whole kernel footprint is inside the input
    // row: the smallest tap offset is ow*SW - PW, the largest ow*SW +
    // (KW-1)*DW - PW, so the span is [ceil(PW/SW), floor((IW-1+PW-(KW-1)*DW)/SW)].
    const int64_t ow_lo = (p.PW + p.SW - 1) / p.SW;
    const int64_t last_hi = p.IW - 1 + p.PW - (p.KW - 1) * p.DW;
    const int64_t ow_hi = (last_hi >= 0) ? last_hi / p.SW : -1;

    // Tap offsets are measured from the padded row start, so the valid range is
    // [PW, IW + PW) — the row pointers already absorbed the bias.
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

/// Lane masks for the plane's two edge width vectors. A vector at the first or
/// last output position reads past the row — the left pad or the right tail —
/// and the general path pays for that with planar_load's per-lane scalar loop.
/// Zeroing the invalid lanes turns the same vector load into a valid one, and
/// because the mask is the same for every output row it folds into the tap's
/// broadcast weight: the R-row loop is then the interior loop, a load and an
/// FMA, rather than eight scalar iterations per tap.
///
/// Only the edge vectors can use it. A middle vector that is not interior (a
/// kernel wide enough to span a whole vector) still takes the scalar path,
/// which is what `e == -1` in the kernel means. Built once per operator call;
/// `needed[e*KW+kw] == 0` says the mask is all ones, and the multiply is
/// skipped. The masked load reads up to PW before the row start and up to
/// L-1+(KW-1)*DW past its end, which is inside the tensor for every row but the
/// first and last — the driver clears `masks` for those two row blocks. Those
/// lanes are real tensor neighbours, not zeros, so a non-finite value there
/// would reach the FMA (0 × NaN); the reference path never reads them.
struct DwPlanarEdges {
    const void* masks = nullptr;      ///< [2][KW][L] elements of T
    const uint8_t* needed = nullptr;  ///< [2][KW]
    int64_t kw = 0;
    int64_t last_ow = 0;              ///< ow of the final width vector
};

/// Fill `masks` ([2*MaxKw*L] elements of T) and `needed` ([2*MaxKw]) from `p` and
/// return the descriptor. The buffers stay the caller's; the result carries no
/// pointer of its own lifetime, so the caller may copy it per task and null
/// `masks` out. Returns a descriptor with `masks == nullptr` for the shapes the
/// masks do not serve — a stride the even-load cannot express, or a kernel past
/// the caller's cap — which is the same thing as asking for the general path.
template <int64_t MaxKw, typename T>
inline DwPlanarEdges dwconv_planar_edge_masks(const DwConvParams& p, T* masks, uint8_t* needed)
{
    constexpr int64_t L = simd_lane_for<T>;
    DwPlanarEdges edges;
    if ((p.SW != 1 && p.SW != 2) || p.KW > MaxKw) {
        return edges;
    }

    const int64_t last_ow = ((p.OW + L - 1) / L - 1) * L;
    // One vector is both edges; `e` in the kernel resolves to 0 for it and the
    // two masks below come out identical, so nothing special is needed.
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
            // All lanes valid: the multiply would be by ones.
            needed[e * p.KW + kw] = all ? 0 : 1;
        }
    }

    edges.masks   = masks;
    edges.needed  = needed;
    edges.kw      = p.KW;
    edges.last_ow = last_ow;
    return edges;
}

/// Copy `n` elements, 1 <= n < L. The count is a runtime value but has only
/// L-1 possibilities, all known to the caller, so the switch collapses each one
/// to a single block move — the per-lane loop it replaces was a load and a
/// store per element.
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

/// R output rows per call, one width-vector accumulate per row. T is the
/// element type; the accumulate and the epilogue ride it.
///
/// SW is the width stride when the caller knows it (1 or 2), 0 when it does
/// not. It is a template argument because everything it decides — which load
/// form the taps take, how wide a vector is in the input, hence where the
/// interior begins — would otherwise be a branch inside the tap loop, and the
/// tap loop is a load and an FMA.
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
    // the row does not exist. The width loop is the inner one, so the four
    // int64 multiplies that build these addresses used to be paid once per
    // output vector — more scalar work than the nine taps above them. A table
    // read replaces them. Kernels past the cap keep the arithmetic inline.
    constexpr int64_t kMaxTapRows = 256;   // (kd, kh, r): 8x8 with R=4 fits
    const T* row_tab[kMaxTapRows];
    // It is worth building only when a row spans more than one width vector:
    // at one vector the table costs more to fill than the arithmetic it saves.
    // Measured either side of the gate (1T GFLOPS, 3x3): a two-vector row is a
    // wash (14.0 vs 14.3), four vectors pay 18.5 vs 13.1, and a 112-wide plane
    // 28.3 vs 13.8.
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

    // Horizontal interior span, also per call: the vector at ow is interior
    // when everything it touches is inside the row. v_load_even deinterleaves
    // 2L elements to produce L outputs, so a stride-2 vector reads twice the
    // width a stride-1 one does — the same guard planar_load uses.
    const int64_t span  = (sw == 2) ? 2 * L : L;
    const int64_t need  = (p.KW - 1) * p.DW + span;    // past the first column
    const int64_t ow_lo = (p.PW + sw - 1) / sw;        // first ow with iw >= 0
    const int64_t ow_hi = (p.IW + p.PW - need >= 0)
                              ? (p.IW + p.PW - need) / sw : -1;

    for (int64_t ow = 0; ow < p.OW; ow += L) {
        decltype(v_set1(out, bias)) acc[R];
        for (int r = 0; r < R; ++r) { acc[r] = v_bias; }

        // Does this whole width vector — every lane, every row, every tap —
        // land inside the input? Then the taps need no bounds test and no
        // scalar fill: one branch per vector instead of one per tap. That is
        // the interior nn_compute reaches by peeling the padded border off at
        // the driver; here the same test is cheap enough to do per vector.
        const bool interior = all_rows && ow >= ow_lo && ow <= ow_hi;

        // Which edge vector this is: 0 first, 1 last, -1 neither. Only those
        // two carry a mask; a single-vector plane has one mask for both.
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

                // Edge vector with a precomputed mask. This is what keeps a
                // narrow plane — where EVERY width vector is an edge — out of
                // the per-lane scalar fallback below.
                if (edges.masks != nullptr && e >= 0) {
                    for (int64_t kw = 0; kw < p.KW; ++kw) {
                        const int64_t iw0 = ow * sw + kw * p.DW - p.PW;
                        // The mask zeroes exactly the lanes this tap would read
                        // from outside the row, and it is the same for every
                        // row — so it folds into the broadcast weight, once per
                        // tap, and the R-row loop stays the interior one:
                        // a load and an FMA, no extra multiply.
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
                // The last vector of a row is short by a fixed amount, so the
                // store is a block copy of that many elements, not a per-lane
                // loop: on a 7-wide plane this is the whole row's store, and
                // the loop was costing more than the nine taps above it.
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
// Public kernels — unchanged signatures
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

template <typename T>
inline void dwconv_h4(T* output, const T* input, const T* w_base,
                      const T* bias_vec, const DwConvParams& p,
                      const Epilogue& epilogue, bool add_to)
{
    const auto vb = bias_vec ? v_load(bias_vec) : v_zero(output);
    const decltype(vb) vbias[1] = {vb};
    const FpSrc<T> src[1] = {FpSrc<T>{input}};
    const FpDst<T> dst[1] = {FpDst<T>{output, p.out_row_stride, add_to}};
    dwconv_rows<4, 1>(p, w_base, output, epilogue, src, dst, vbias);
}

template <typename T>
inline void dwconv_h1(T* output, const T* input, const T* w_base,
                      const T* bias_vec, const DwConvParams& p,
                      const Epilogue& epilogue, bool add_to)
{
    const auto vb = bias_vec ? v_load(bias_vec) : v_zero(output);
    const decltype(vb) vbias[1] = {vb};
    const FpSrc<T> src[1] = {FpSrc<T>{input}};
    const FpDst<T> dst[1] = {FpDst<T>{output, p.out_row_stride, add_to}};
    dwconv_rows<1, 1>(p, w_base, output, epilogue, src, dst, vbias);
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
    const v_f32x8 vbias[1] = {bias_vec ? v_load(bias_vec) : v_zero_f32x8()};
    const QuantSrc<InT> src[1] = {QuantSrc<InT>{input, in_scale, in_zero}};
    const QuantDst<OutT> dst[1] = {
        QuantDst<OutT>{output, p.out_row_stride, out_scale, out_zero, add_to}};
    dwconv_rows<4, 1>(p, wq, &out_scale, epilogue, src, dst, vbias);
}

template <typename InT, typename OutT>
inline void dwconv_h1_quant(OutT* output, const InT* input, const float* wq,
                            const float* bias_vec, const DwConvParams& p,
                            const Epilogue& epilogue, bool add_to,
                            float in_scale, float in_zero,
                            float out_scale, float out_zero)
{
    const v_f32x8 vbias[1] = {bias_vec ? v_load(bias_vec) : v_zero_f32x8()};
    const QuantSrc<InT> src[1] = {QuantSrc<InT>{input, in_scale, in_zero}};
    const QuantDst<OutT> dst[1] = {
        QuantDst<OutT>{output, p.out_row_stride, out_scale, out_zero, add_to}};
    dwconv_rows<1, 1>(p, wq, &out_scale, epilogue, src, dst, vbias);
}

// ============================================================
// Channel-last (NHWC/NDHWC) entry points
// ============================================================
//
// These are the packed kernel (dwconv_rows) with two runtime strides instead of
// the C8 constants: a pixel is C elements away, because a pixel holds all of
// the plane's channels, and a filter tap is C elements away too, because the
// weight is the raw HWCM filter [tap][C] — TensorFlow's depthwise order with
// the multiplier dropped. No prepack step is involved: the layout already is
// the access pattern.
//
// The driver also keeps the packed route's shape — four output rows per call
// for the ILP, one channel block per call — and that is not incidental. A
// first cut gave the channel-last route its own kernel with NC pixels in flight
// down the contiguous axis, one output row per task, one task per (row, block).
// It measured 1.4-1.9x slower than the packed kernel on every shape, and at
// C = 8 — where the two layouts are byte-identical, so only the kernel differs
// — it was still 1.4x slower: the column sweep needs eight live accumulators to
// cover the same four pixels the row sweep gets from four, and the per-block
// call rebuilt the row table and re-walked the row for each of the C/8 blocks.

/// @p out_rows is the output row base advanced by the channel block, @p in_img
/// the image base advanced by the same, @p w_tap0 the block's first channel of
/// the tap-major weight, and @p lanes how many channels the call covers — up to
/// 16, which is a whole 64-byte line of f32.
///
/// Sixteen channels run as two vectors. One vector per call is what the packed
/// route does, and it is free there because a block's plane is dense; on a
/// channel-last plane the second half of every line belongs to the next block,
/// so a one-vector call leaves it for a later pass — two touches per line where
/// the packed route has one.
///
/// **A 3x3 takes a specialised kernel at both widths** (2026-10-30):
/// `dwconv_last3x3` below for one vector of channels and `dwconv_last3x3_16`
/// for two. Both hoist the row table and drop the per-tap pointer arithmetic.
/// The sixteen-channel one is what the driver asks for: an eight-channel block
/// touches every 64-byte line twice, and a strided half-line sweep of a plane
/// costs exactly what a full sweep of it costs (measured — a half-used-line read
/// runs at the plane's full time), so the narrow route pays twice the memory for
/// the same FMAs and no kernel-level win buys that back.

/// True when the shape takes the register-hoisted 3x3 kernels below: one depth
/// plane, a 3x3 filter, no dilation. The vertical stride is not a condition — a
/// 2x2 stride reads `R·SH + KH - 1` distinct input rows instead of `R + 2` and
/// is still one row table — but a vertical stride of 3 or more is not common
/// enough to instantiate.
inline bool dwconv_last3x3_ok(const DwConvParams& p)
{
    return p.KD == 1 && p.KH == 3 && p.KW == 3 && p.DH == 1 && p.DW == 1 &&
           (p.SH == 1 || p.SH == 2);
}

/// One 8-channel vector of a 3x3 depthwise filter, R output rows per call, with
/// the nine tap vectors held in registers for the whole width sweep.
///
/// This is what the generic kernel above cannot do. It walks the taps inside
/// the column loop, so every column reloads all nine weight vectors, and
/// re-derives the row pointer for each (kh, r) — nine weight loads and
/// twelve pointer fetches per column on top of the thirty-six input loads. The
/// taps are loop-invariant in `ow` and belong in registers; the reason they are
/// not is the register file: nine taps plus four accumulators is thirteen of
/// the sixteen ymm registers, and the sixteen-channel call wants eight
/// accumulators, which would be twenty-six — `dwconv_last3x3_16` below is how
/// that constraint is met, by reloading a tap per column and letting one load
/// feed four rows and both halves.
///
/// So the width-one trade is channels for taps: one vector per call, taps
/// hoisted. It costs a second walk over every cache line's other half, which is
/// why the driver never asks for it on a 3x3 — the sixteen-channel kernel below
/// is that route's answer, and this one is what a plane narrower than sixteen
/// channels falls back to.
///
/// @p SH is the vertical stride, needed because it decides how many distinct
/// input rows the group reads: output row r with filter row kh is input row
/// `oh·SH + r·SH + kh - PH`, so the table is `R·SH + 2` entries and a tap's row
/// is `row[r·SH + kh]` — compile-time indices, no arithmetic in the loop. The
/// horizontal stride stays a runtime value: it costs one scalar product per
/// column, which the loop below has slack for.
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
    // pad so a tap's address is `row + (ow·SW + kw)·pix`.
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

/// The two-vector 3x3: sixteen channels, the same nine taps and the same row
/// table, eight accumulators for a four-row group, and the weight vectors loaded
/// per tap inside the column loop — there is no room to hoist them (eight
/// accumulators plus eighteen tap vectors is twenty-six of sixteen registers),
/// but one weight load then feeds four rows and both halves, so a tap costs ten
/// loads for eight FMAs and the loop stays off the two-load-per-cycle floor.
///
/// This is the chunk width the memory system wants: a pixel's sixteen f32 are
/// one 64-byte line, so one sweep of the block loop touches every line once.
/// The eight-channel kernel above pays two sweeps for the same bytes.
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

/// Quantized channel-last. The HWCM filter is s8/u8, so this block's taps are
/// dequantized once per call into a staging block — nine vectors for a 3x3
/// against the row group's thousands of FMAs — and the kernel is then the f32
/// one above. Pad lanes are neutralized with scale 0, the convention the packed
/// route used: the byte they read is the neighbouring channel, and 0 * it is 0.
constexpr int64_t kMaxLastQuantTaps = 9 * 9 * 9;  // 729

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
    // width even though the plane's pixel stride is the channel count.
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
