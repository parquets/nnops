#pragma once
/// @file simd_depthwise_conv.hpp
/// @brief SIMD kernels for 2D/3D depthwise convolution.
///
/// The driver reads the raw HWCM filter [KD][KH][KW][C] — TF's depthwise order
/// with the multiplier dropped — and puts the lane on the channel. The pixel
/// stride and the tap stride are both the channel count, set by the driver; no
/// prepack is involved, because the layout already is the access pattern. A call
/// covers NV adjacent channel vectors, 2 where the channels allow, since a
/// 64-byte line holds two f32 vectors and a one-vector call leaves the other half
/// of every line to a later pass — 2x the line traffic for the same arithmetic.
///
/// It runs R = 4 output rows per call, R = 1 for a tail: one row's taps are a
/// chain of KD*KH*KW dependent FMAs with nothing to interleave, and four rows are
/// four independent chains. A KD of 1 — the 2D case — degenerates the depth loop,
/// and padding is bounds-checked in-kernel.
///
/// Route map for one channel block (`dwconv_block` / `dwconv_block_quant`):
///
///   lanes > 16 / not a 3x3 -> dwconv_rows<2>    (generic rolled, two vectors)
///   lanes == 16, 3x3 s1/s2 -> dwconv_3x3_2v     (rolled) or _2v_4row (R = 4)
///   lanes == 8,  3x3 s1/s2 -> dwconv_3x3_1v     (one vector)
///   lanes < 8              -> dwconv_rows<1>    (masked lane source)
///
/// The suffix counts channel *vectors*, not channels: `_1v` is 8 channels,
/// `_2v` is 16, and `_2v_4row` is the straight-line four-row body.
///
/// Everything that depends only on the tap is hoisted out of the tap loops: a row
/// pointer per (kd, kh, r), left-pad-biased, and an interior span of output
/// positions whose every tap is inside the row, where the tap loop takes no
/// bounds test and no null test — one branch per position instead of one per tap.
/// The null test (does this input row exist) can only be skipped when every row of
/// the group does, so a group whose window reaches past the plane keeps the
/// checked path.
///
/// Three shapes were measured against this and lost: peeling the padded rows out
/// of the groups costs a quarter of the ILP per row and ran 14-22% slower; two
/// output positions per iteration share the weight vectors but need eight live
/// accumulators and reached 24 GFLOPS against 35; a column sweep ran
/// 1.4-1.9x slower at every shape, 1.4x even at C = 8 where only the kernel
/// differs.
///
/// The 3x3 kernels hold the nine taps in registers —
/// `dwconv_3x3_1v` for one channel vector, `dwconv_3x3_2v` for two, which
/// cannot hoist eighteen tap vectors and reloads one per column to feed four rows
/// and both halves. Sixteen channels is what the driver asks for: a pixel's 16
/// f32 is one 64-byte line, so a sweep touches every line once, where an
/// 8-channel block pays two sweeps for the same bytes.
///
/// Quantized depthwise dequantizes this block's taps once per call into a dense
/// staging block (9 vectors for a 3x3 against the group's thousands of FMAs), then
/// runs `dwconv_rows` on the f32 weights; a pad lane gets scale 0.

#include "detail/bench_knobs.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "nnops/core/epilogue.hpp"
#include "simd_epilogue.hpp"
#include "simd_access.hpp"
#include "simd_window.hpp"

#include <cstdint>
#include <cstring>

namespace nnops::kernel {

struct DwConvParams {
    // ---- geometry (fixed for the whole call) ----
    int64_t ID, IH, IW;       // input spatial dims
    int64_t OH, OW;           // output spatial dims (for store bounds)
    int64_t KD, KH, KW;       // kernel shape
    int64_t SD, SH, SW;       // stride
    int64_t DD, DH, DW;       // dilation
    int64_t PD, PH, PW;       // padding
    int64_t out_row_stride;   // output row stride (elems)
    int64_t in_d_stride;      // input depth stride (elems)
    int64_t in_row_stride;    // input row stride (elems)
    int64_t pix_stride = 8;   // elems between pixels (= C, set by the driver)
    int64_t w_tap_stride = 8; // elems between taps (= C of a raw HWCM filter,
                              // 8 in the dequantized staging block)
    // ---- cursor (set before each call) ----
    int64_t od, oh;           // current output position
    int64_t ow0 = 0;          // first output column this call covers
    int64_t ow_end = -1;      // one past the last; negative means OW
};

/// R output rows per call; W is the weight type and Tag picks the epilogue overload.
///
/// @p add_to adds the destination after the epilogue, @p pre_act folds it into
/// the accumulator before it; the driver sets at most one of the two.
template <int R, int NV, typename W, typename Tag, typename Src, typename Dst, typename V>
inline void dwconv_rows(const DwConvParams& p, const W* w, const Tag* tag, const Epilogue& epilogue,
                        const Src (&srcs)[NV], const Dst (&dsts)[NV], const V (&vbias)[NV],
                        bool pre_act)
{
    using InPtr = const typename Src::elem*;
    const int64_t pix = p.pix_stride;
    const int64_t wtap = p.w_tap_stride;
    constexpr int64_t kVec = kChannelLanes;
    constexpr int64_t kMaxTapRows = 256; // (kd, kh, r): 8x8 with R = 4 fits

    // A short channel block's filter is short too: the last tap of the raw HWCM
    // weight ends at C, so a whole-vector read there would run past the filter.
    // The activation's live lanes are zero past `lanes`, so the lanes the weight
    // read must not touch contribute nothing.
    const auto load_w = [&](const W* wp) -> V {
        if constexpr (Src::full) {
            return v_load(wp);
        } else {
            return load_lanes(wp, srcs[0].lanes);
        }
    };

    // The row table is built from vector 0's base, so vector k adds k*kVec to it.
    InPtr row_base[kMaxTapRows];
    const RowTable rows = build_row_table(p, R, srcs[0].ptr_at(0), row_base, kMaxTapRows);
    const bool have_tab = rows.built;
    const bool all_rows = rows.all_valid;

    // Interior columns: every tap offset ow*SW + kw*DW - PW lands in [0, IW).
    // Row pointers have already absorbed -PW*pix, so a valid tap offset is
    // [PW, IW + PW).
    const WindowSpan span = window_span(p, p.KW, p.DW);
    const int64_t ow_lo = span.lo;
    const int64_t ow_hi = span.hi;
    const int64_t off_ok_lo = span.tap_lo;
    const int64_t off_ok_hi = span.tap_hi;
    const int64_t ow_end = span.end;
    for (int64_t ow = p.ow0; ow < ow_end; ++ow) {
        V acc[R][NV];
        for (int r = 0; r < R; ++r) {
            for (int k = 0; k < NV; ++k) {
                acc[r][k] = vbias[k];
            }
        }

        const bool interior = all_rows && ow >= ow_lo && ow <= ow_hi;
        const int64_t ow_off = ow * p.SW;

        for (int64_t kd = 0; kd < p.KD; ++kd) {
            for (int64_t kh = 0; kh < p.KH; ++kh) {
                const W* w_row = w + (kd * p.KH + kh) * p.KW * wtap;
                InPtr row[R];
                if (have_tab) {
                    const int64_t t = (kd * p.KH + kh) * R;
                    for (int64_t r = 0; r < R; ++r) {
                        row[r] = row_base[t + r];
                    }
                } else {
                    const int64_t id = p.od * p.SD + kd * p.DD - p.PD;
                    if (id < 0 || id >= p.ID) {
                        continue;
                    }
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
                            const V vk = load_w(w_row + kw * wtap + k * kVec);
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
                        const V vk = load_w(w_row + kw * wtap + k * kVec);
                        for (int r = 0; r < R; ++r) {
                            if (row[r] == nullptr) {
                                continue;
                            }
                            acc[r][k] =
                                v_fmadd(srcs[k].load_ptr(row[r] + at + k * kVec), vk, acc[r][k]);
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
                V a = acc[r][k];
                if (pre_act) {
                    a = v_add(a, dsts[k].load(r, ow));
                }
                dsts[k].store(r, ow, nnops::backend::cpu::apply_epilogue_vec(epilogue, tag, a));
            }
        }
    }
}

// ============================================================
// Entry points
// ============================================================
//
// out_rows and in_img are the bases advanced by the channel block, w_tap0 the
// block's first channel of the tap-major weight, and lanes how many channels the
// call covers — up to 16, a whole 64-byte line of f32. A 3x3 takes
// dwconv_3x3_1v at one vector and dwconv_3x3_2v at two.

/// One depth plane, 3x3, no dilation, vertical stride 1 or 2 — a larger one would
/// still fit one row table but is not common enough to instantiate.
inline bool dwconv_3x3_ok(const DwConvParams& p)
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
inline void dwconv_3x3_1v(T* out_rows, const T* in_img, const T* w_tap0,
                                    const T* bias_vec, const DwConvParams& p,
                                    const Epilogue& epilogue, bool add_to, bool pre_act)
{
    using V = decltype(v_load(in_img));
    const int64_t pix = p.pix_stride;
    const int64_t wtap = p.w_tap_stride;
    const int64_t ow_end = (p.ow_end >= 0) ? p.ow_end : p.OW;

    const V vbias = bias_vec ? v_load(bias_vec) : v_zero(in_img);
    V wt[9];
    for (int t = 0; t < 9; ++t) {
        wt[t] = v_load(w_tap0 + t * wtap);
    }
    // An inactive epilogue writes the accumulator out as it stands; the runtime
    // switch in apply_epilogue_vec would otherwise cost a load and an indirect
    // branch per store.
    const bool simple = !add_to && !pre_act && !nnops::epilogue_is_active(epilogue);

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

    const WindowSpan span = window_span(p, 3, p.DW);
    const int64_t ow_lo = span.lo;
    const int64_t ow_hi = span.hi;
    const int64_t off_ok_lo = span.tap_lo;
    const int64_t off_ok_hi = span.tap_hi;

    for (int64_t ow = p.ow0; ow < ow_end; ++ow) {
        V acc[R];
        for (int r = 0; r < R; ++r) {
            acc[r] = vbias;
        }

        if (all_rows && ow >= ow_lo && ow <= ow_hi) {
            const int64_t o = ow * p.SW * pix;
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    const int64_t at = o + kw * pix;
                    for (int r = 0; r < R; ++r) {
                        acc[r] = v_fmadd(v_load(row[r * SH + kh] + at), wt[kh * 3 + kw], acc[r]);
                    }
                }
            }
        } else {
            const int64_t ow_off = ow * p.SW;
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    const int64_t off = ow_off + kw;
                    if (off < off_ok_lo || off >= off_ok_hi) {
                        continue;
                    }
                    const int64_t at = off * pix;
                    for (int r = 0; r < R; ++r) {
                        if (row[r * SH + kh] == nullptr) {
                            continue;
                        }
                        acc[r] = v_fmadd(v_load(row[r * SH + kh] + at), wt[kh * 3 + kw], acc[r]);
                    }
                }
            }
        }

        for (int r = 0; r < R; ++r) {
            if (p.oh + r >= p.OH) {
                continue;
            }
            T* dst = out_rows + r * p.out_row_stride + ow * pix;
            if (simple) {
                v_store(dst, acc[r]);
                continue;
            }
            V a = acc[r];
            if (pre_act) {
                a = v_add(a, v_load(dst));
            }
            v_store_add(dst, nnops::backend::cpu::apply_epilogue_vec(epilogue, out_rows, a),
                        add_to);
        }
    }
}

/// The two-vector 3x3 with the two rows a group can reach past the plane — the
/// top one and the bottom one — known at compile time. A group whose window
/// leaves the plane falls back to the checked path for *every* column, and at
/// both ends of every plane that is exactly one row: skipping its tap at
/// compile time is what puts those groups back on the unchecked loop.
template <int R, int SH, bool TOP_OK, bool BOT_OK, typename T>
inline void dwconv_3x3_2v_masked(T* out_rows, const T* in_img, const T* w_tap0,
                                              const T* bias_vec, const DwConvParams& p,
                                              const Epilogue& epilogue, bool add_to, bool pre_act)
{
    using V = decltype(v_load(in_img));
    constexpr int64_t kL = kChannelLanes;
    constexpr int64_t kRows = (R - 1) * SH + 3;
    const int64_t pix = p.pix_stride;
    const int64_t wtap = p.w_tap_stride;
    const int64_t ow_end = (p.ow_end >= 0) ? p.ow_end : p.OW;

    const V vbias[2] = {bias_vec ? v_load(bias_vec) : v_zero(in_img),
                        bias_vec ? v_load(bias_vec + kL) : v_zero(in_img)};

    // A masked-out row never reaches the FMA, so it holds a pointer that is
    // valid to form — the plane's own first pixel — and is never dereferenced.
    const T* row[kRows];
    for (int64_t i = 0; i < kRows; ++i) {
        const int64_t ih = p.oh * SH + i - p.PH;
        row[i] = (!TOP_OK && i == 0) ? in_img : in_img + ih * p.in_row_stride - p.PW * pix;
    }

    const WindowSpan span = window_span(p, 3, p.DW);
    const int64_t ow_lo = span.lo;
    const int64_t ow_hi = span.hi;
    const int64_t off_ok_lo = span.tap_lo;
    const int64_t off_ok_hi = span.tap_hi;

    for (int64_t ow = p.ow0; ow < ow_end; ++ow) {
        V acc[R][2];
        for (int r = 0; r < R; ++r) {
            acc[r][0] = vbias[0];
            acc[r][1] = vbias[1];
        }

        if (ow >= ow_lo && ow <= ow_hi) {
            const int64_t o = ow * p.SW * pix;
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    const int64_t at = o + kw * pix;
                    const T* wp = w_tap0 + (kh * 3 + kw) * wtap;
                    const V w0 = v_load(wp);
                    const V w1 = v_load(wp + kL);
                    for (int r = 0; r < R; ++r) {
                        if ((!TOP_OK && r == 0 && kh == 0) || (!BOT_OK && r == R - 1 && kh == 2)) {
                            continue;
                        }
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
                    if (off < off_ok_lo || off >= off_ok_hi) {
                        continue;
                    }
                    const int64_t at = off * pix;
                    const T* wp = w_tap0 + (kh * 3 + kw) * wtap;
                    const V w0 = v_load(wp);
                    const V w1 = v_load(wp + kL);
                    for (int r = 0; r < R; ++r) {
                        if ((!TOP_OK && r == 0 && kh == 0) || (!BOT_OK && r == R - 1 && kh == 2)) {
                            continue;
                        }
                        const T* ip = row[r * SH + kh] + at;
                        acc[r][0] = v_fmadd(v_load(ip), w0, acc[r][0]);
                        acc[r][1] = v_fmadd(v_load(ip + kL), w1, acc[r][1]);
                    }
                }
            }
        }

        for (int r = 0; r < R; ++r) {
            if (p.oh + r >= p.OH) {
                continue;
            }
            T* dst = out_rows + r * p.out_row_stride + ow * pix;
            V a0 = acc[r][0];
            V a1 = acc[r][1];
            if (pre_act) {
                a0 = v_add(a0, v_load(dst));
                a1 = v_add(a1, v_load(dst + kL));
            }
            v_store_add(dst, nnops::backend::cpu::apply_epilogue_vec(epilogue, out_rows, a0),
                        add_to);
            v_store_add(dst + kL, nnops::backend::cpu::apply_epilogue_vec(epilogue, out_rows, a1),
                        add_to);
        }
    }
}

// A tap of the unrolled four-row body: two weight vectors, then eight FMAs whose
// source addresses are literals. The two rows a group can reach past the plane are
// compile-time guards, so a masked tap costs no branch. Token-pasted row names keep
// the eight accumulators in registers; an indexed array is what the rolled kernel
// spills.
//
// `wsrc` points at the packed copy of the taps (see the kernel), so a tap's two weight
// vectors sit at a literal displacement and cost no address arithmetic. `wsrc` is
// volatile on purpose: it keeps the eighteen loads inside the width loop, because
// hoisted they are eighteen live vectors on top of the eight accumulators and the
// allocator answers that by spilling the accumulators. A read-write barrier or a store
// into `wpack` blocks the hoist just as well, and both were measured worse: with the
// taps free to become FMA memory operands, the body re-loads every input per output row
// — seventy-two loads a column against thirty-six — which costs more than the eighteen
// pointer reloads it saves.
//
// `OFF` is the running byte-free column offset for this tap's `kw`.
#define NNOPS_DW_3X3_2V_4ROW_TAP(KH, KW, OFF)                                        \
    {                                                                        \
        const V w0 = v_load(wsrc + ((KH) * 3 + (KW)) * 2 * kL);              \
        const V w1 = v_load(wsrc + ((KH) * 3 + (KW)) * 2 * kL + kL);         \
        if (TOP_OK || (KH) != 0) {                                           \
            a00 = v_fmadd(v_load(row[(KH)] + (OFF)), w0, a00);               \
            a01 = v_fmadd(v_load(row[(KH)] + (OFF) + kL), w1, a01);          \
        }                                                                    \
        a10 = v_fmadd(v_load(row[SH + (KH)] + (OFF)), w0, a10);              \
        a11 = v_fmadd(v_load(row[SH + (KH)] + (OFF) + kL), w1, a11);         \
        a20 = v_fmadd(v_load(row[2 * SH + (KH)] + (OFF)), w0, a20);          \
        a21 = v_fmadd(v_load(row[2 * SH + (KH)] + (OFF) + kL), w1, a21);     \
        if (BOT_OK || (KH) != 2) {                                           \
            a30 = v_fmadd(v_load(row[3 * SH + (KH)] + (OFF)), w0, a30);      \
            a31 = v_fmadd(v_load(row[3 * SH + (KH)] + (OFF) + kL), w1, a31); \
        }                                                                    \
    }

// `rows_left` is p.OH - p.oh, hoisted: read from the params, the four store predicates
// cost four loads, four adds and four compares per column.
#define NNOPS_DW_3X3_2V_4ROW_STORE_AT(R, DST)                                                              \
    if ((R) < rows_left) {                                                                         \
        T* dst = (DST);                                                                            \
        if constexpr (ACT == kStorePlain) {                                                        \
            v_store(dst, a##R##0);                                                                 \
            v_store(dst + kL, a##R##1);                                                            \
        } else if constexpr (ACT == kStoreRelu) {                                                  \
            v_store(dst, v_relu(a##R##0, v_zero(in_img)));                                         \
            v_store(dst + kL, v_relu(a##R##1, v_zero(in_img)));                                    \
        } else if constexpr (ACT == kStoreRelu6) {                                                 \
            const V v0 = v_zero(in_img);                                                           \
            const V v6 = v_set1(in_img, 6.0f);                                                     \
            v_store(dst, v_min(v_max(a##R##0, v0), v6));                                           \
            v_store(dst + kL, v_min(v_max(a##R##1, v0), v6));                                      \
        } else {                                                                                   \
            V x0 = a##R##0;                                                                        \
            V x1 = a##R##1;                                                                        \
            if (pre_act) {                                                                         \
                x0 = v_add(x0, v_load(dst));                                                       \
                x1 = v_add(x1, v_load(dst + kL));                                                  \
            }                                                                                      \
            v_store_add(dst, nnops::backend::cpu::apply_epilogue_vec(epilogue, out_rows, x0),      \
                        add_to);                                                                   \
            v_store_add(dst + kL, nnops::backend::cpu::apply_epilogue_vec(epilogue, out_rows, x1), \
                        add_to);                                                                   \
        }                                                                                          \
    }

#define NNOPS_DW_3X3_2V_4ROW_STORE(R) NNOPS_DW_3X3_2V_4ROW_STORE_AT(R, out_rows + (R) * p.out_row_stride + ow * pix)

/// Store variants of the straight-line four-row 3x3 body. Folding a plain
/// activation into the store keeps the eight accumulators in registers; the
/// generic epilogue needs enough registers of its own that the allocator spills
/// them, which is why it is a separate instantiation.
enum DwStore : int {
    kStorePlain = 0,    ///< store the accumulator as it stands
    kStoreRelu = 1,     ///< fold a plain Relu into the store
    kStoreRelu6 = 2,    ///< fold a plain Relu6 into the store
    kStoreEpilogue = 3, ///< generic epilogue (and add_to / pre_act)
};

/// The two-vector 3x3 at four rows, straight-line: the nine taps of a column are
/// a compile-time sequence with literal source offsets, so the body carries no tap
/// counter, no row-table index and no accumulator array. The rolled variant below
/// rebuilds four row pointers per filter row and drives both loops off a counter.
///
/// @p ACT is a template argument rather than a runtime flag because the epilogue
/// still has to be *absent* from the instantiation: the four activation arms of a
/// live epilogue need enough registers of their own that the allocator spills the
/// eight accumulators to stack. A store-only body leaves eleven live vectors and the
/// interior loop compiles clean. The two plain arms keep the store straight-line,
/// so the register file the store-only path had survives.
template <int SH, bool TOP_OK, bool BOT_OK, DwStore ACT, typename T>
inline void dwconv_3x3_2v_4row(T* out_rows, const T* in_img, const T* w_tap0,
                                         const T* bias_vec, const DwConvParams& p,
                                         const Epilogue& epilogue, bool add_to, bool pre_act)
{
    if constexpr (ACT != kStoreEpilogue) {
        (void)epilogue;
        (void)add_to;
        (void)pre_act;
    }
    using V = decltype(v_load(in_img));
    constexpr int64_t kL = kChannelLanes;
    constexpr int64_t kRows = 3 * SH + 3;
    const int64_t pix = p.pix_stride;
    const int64_t wtap = p.w_tap_stride;
    const int64_t ow_end = (p.ow_end >= 0) ? p.ow_end : p.OW;

    const V vb0 = bias_vec ? v_load(bias_vec) : v_zero(in_img);
    const V vb1 = bias_vec ? v_load(bias_vec + kL) : v_zero(in_img);

    const T* row[kRows];
    for (int64_t i = 0; i < kRows; ++i) {
        const int64_t ih = p.oh * SH + i - p.PH;
        row[i] = in_img + ih * p.in_row_stride - p.PW * pix;
    }
    // A masked-out row never reaches an FMA, so it can hold any valid pointer.
    if (!TOP_OK) {
        row[0] = in_img;
    }

    // The nine taps copied into tap order. Addressing them through `wtap` costs an
    // imul per load, and the width loop has no general register to spare; copied, every
    // weight access is a literal displacement. One copy per four-row block, so the
    // amortised cost is a fraction of a load per column.
    alignas(32) T wpack[18 * kL];
    for (int64_t k = 0; k < 9; ++k) {
        v_store(wpack + k * 2 * kL, v_load(w_tap0 + k * wtap));
        v_store(wpack + k * 2 * kL + kL, v_load(w_tap0 + k * wtap + kL));
    }
#if defined(NNOPS_ARCH_AARCH64)
    // AArch64 has 32 vector registers: the eighteen taps and the eight
    // accumulators coexist, so the taps stay resident across the width sweep.
    // The volatile bar below is a 16-ymm remedy, and it costs this target a
    // third of the load ports on data that is already in registers.
    const T* const wsrc = wpack;
#else
    const T* const volatile wsrc = wpack;
#endif

    const WindowSpan span = window_span(p, 3, p.DW);
    const int64_t ow_lo = span.lo;
    const int64_t ow_hi = span.hi;
    const int64_t off_ok_lo = span.tap_lo;
    const int64_t off_ok_hi = span.tap_hi;
    const int64_t rows_left = p.OH - p.oh;

    // A column whose 3x3 window straddles the padded edge takes the rolled masked
    // pass. It rebuilds its own row table: the rolled loops index by a variable `kh`,
    // and one variable index is enough to put the table in memory, after which the
    // interior loop reloads `row[k]` from the stack on the load ports — measured at 89
    // stack accesses per column against 90 real loads. Constant indices let the
    // allocator keep the interior table in registers.
    const auto masked_col = [&](int64_t ow) {
        const T* mrow[kRows];
        for (int64_t i = 0; i < kRows; ++i) {
            const int64_t ih = p.oh * SH + i - p.PH;
            mrow[i] = in_img + ih * p.in_row_stride - p.PW * pix;
        }
        if (!TOP_OK) {
            mrow[0] = in_img;
        }
        V a00 = vb0, a01 = vb1, a10 = vb0, a11 = vb1;
        V a20 = vb0, a21 = vb1, a30 = vb0, a31 = vb1;
        const int64_t ow_off = ow * p.SW;
        for (int64_t kh = 0; kh < 3; ++kh) {
            for (int64_t kw = 0; kw < 3; ++kw) {
                const int64_t off = ow_off + kw;
                if (off < off_ok_lo || off >= off_ok_hi) {
                    continue;
                }
                const int64_t at = off * pix;
                const V w0 = v_load(w_tap0 + (kh * 3 + kw) * wtap);
                const V w1 = v_load(w_tap0 + (kh * 3 + kw) * wtap + kL);
                if (TOP_OK || kh != 0) {
                    a00 = v_fmadd(v_load(mrow[kh] + at), w0, a00);
                    a01 = v_fmadd(v_load(mrow[kh] + at + kL), w1, a01);
                }
                a10 = v_fmadd(v_load(mrow[SH + kh] + at), w0, a10);
                a11 = v_fmadd(v_load(mrow[SH + kh] + at + kL), w1, a11);
                a20 = v_fmadd(v_load(mrow[2 * SH + kh] + at), w0, a20);
                a21 = v_fmadd(v_load(mrow[2 * SH + kh] + at + kL), w1, a21);
                if (BOT_OK || kh != 2) {
                    a30 = v_fmadd(v_load(mrow[3 * SH + kh] + at), w0, a30);
                    a31 = v_fmadd(v_load(mrow[3 * SH + kh] + at + kL), w1, a31);
                }
            }
        }
        NNOPS_DW_3X3_2V_4ROW_STORE(0) NNOPS_DW_3X3_2V_4ROW_STORE(1) NNOPS_DW_3X3_2V_4ROW_STORE(2) NNOPS_DW_3X3_2V_4ROW_STORE(3)
    };

    const int64_t iw_lo = (ow_lo > p.ow0) ? ow_lo : p.ow0;
    const int64_t iw_hi = (ow_hi + 1 < ow_end) ? ow_hi + 1 : ow_end;
    const int64_t iw_tail = (iw_hi > iw_lo) ? iw_hi : iw_lo;

    for (int64_t ow = p.ow0; ow < iw_lo; ++ow) {
        masked_col(ow);
    }

    // Three walking source offsets, one per `kw`, and a walking destination offset: the
    // column index disappears from the loop, leaving literal displacements in the tap
    // addresses and no imul or lea per tap.
    const int64_t swpix = p.SW * pix;
    const int64_t stride = p.out_row_stride;
    int64_t o0 = iw_lo * swpix;
    int64_t o1 = o0 + pix;
    int64_t o2 = o1 + pix;
    int64_t dpos = iw_lo * pix;
    for (int64_t ow = iw_lo; ow < iw_hi; ++ow, o0 += swpix, o1 += swpix, o2 += swpix, dpos += pix) {
        T* dst0 = out_rows + dpos;
        V a00 = vb0, a01 = vb1, a10 = vb0, a11 = vb1;
        V a20 = vb0, a21 = vb1, a30 = vb0, a31 = vb1;
        NNOPS_DW_3X3_2V_4ROW_TAP(0, 0, o0)
        NNOPS_DW_3X3_2V_4ROW_TAP(0, 1, o1)
        NNOPS_DW_3X3_2V_4ROW_TAP(0, 2, o2)
        NNOPS_DW_3X3_2V_4ROW_TAP(1, 0, o0)
        NNOPS_DW_3X3_2V_4ROW_TAP(1, 1, o1)
        NNOPS_DW_3X3_2V_4ROW_TAP(1, 2, o2)
        NNOPS_DW_3X3_2V_4ROW_TAP(2, 0, o0)
        NNOPS_DW_3X3_2V_4ROW_TAP(2, 1, o1)
        NNOPS_DW_3X3_2V_4ROW_TAP(2, 2, o2)
        NNOPS_DW_3X3_2V_4ROW_STORE_AT(0, dst0)
        NNOPS_DW_3X3_2V_4ROW_STORE_AT(1, dst0 + stride)
        NNOPS_DW_3X3_2V_4ROW_STORE_AT(2, dst0 + 2 * stride)
        NNOPS_DW_3X3_2V_4ROW_STORE_AT(3, dst0 + 3 * stride)
    }

    for (int64_t ow = iw_tail; ow < ow_end; ++ow) {
        masked_col(ow);
    }
}

#undef NNOPS_DW_3X3_2V_4ROW_TAP
#undef NNOPS_DW_3X3_2V_4ROW_STORE
#undef NNOPS_DW_3X3_2V_4ROW_STORE_AT

/// The (top, bottom) matrix for the straight-line four-row body. Written once
/// here so every caller reads as a single kernel choice instead of four nested
/// template branches.
template <int SH, DwStore ACT, typename T>
inline void dwconv_3x3_2v_4row_topbot(bool top_ok, bool bot_ok, T* out_rows, const T* in_img,
                                      const T* w_tap0, const T* bias_vec, const DwConvParams& p,
                                      const Epilogue& epilogue, bool add_to, bool pre_act)
{
    if (top_ok) {
        if (bot_ok) {
            dwconv_3x3_2v_4row<SH, true, true, ACT>(out_rows, in_img, w_tap0, bias_vec, p,
                                                    epilogue, add_to, pre_act);
        } else {
            dwconv_3x3_2v_4row<SH, true, false, ACT>(out_rows, in_img, w_tap0, bias_vec, p,
                                                     epilogue, add_to, pre_act);
        }
    } else if (bot_ok) {
        dwconv_3x3_2v_4row<SH, false, true, ACT>(out_rows, in_img, w_tap0, bias_vec, p, epilogue,
                                                 add_to, pre_act);
    } else {
        dwconv_3x3_2v_4row<SH, false, false, ACT>(out_rows, in_img, w_tap0, bias_vec, p, epilogue,
                                                  add_to, pre_act);
    }
}

/// The same matrix for the masked multi-row body.
template <int R, int SH, typename T>
inline void dwconv_3x3_2v_masked_topbot(bool top_ok, bool bot_ok, T* out_rows, const T* in_img,
                                        const T* w_tap0, const T* bias_vec, const DwConvParams& p,
                                        const Epilogue& epilogue, bool add_to, bool pre_act)
{
    if (top_ok) {
        if (bot_ok) {
            dwconv_3x3_2v_masked<R, SH, true, true>(out_rows, in_img, w_tap0, bias_vec, p,
                                                    epilogue, add_to, pre_act);
        } else {
            dwconv_3x3_2v_masked<R, SH, true, false>(out_rows, in_img, w_tap0, bias_vec, p,
                                                     epilogue, add_to, pre_act);
        }
    } else if (bot_ok) {
        dwconv_3x3_2v_masked<R, SH, false, true>(out_rows, in_img, w_tap0, bias_vec, p, epilogue,
                                                 add_to, pre_act);
    } else {
        dwconv_3x3_2v_masked<R, SH, false, false>(out_rows, in_img, w_tap0, bias_vec, p, epilogue,
                                                  add_to, pre_act);
    }
}

/// The two-vector 3x3: sixteen channels, the same row table and eight
/// accumulators, with the tap vectors loaded per column inside the width loop —
/// eighteen of them plus eight accumulators is twenty-six registers — but one
/// weight load then feeds four rows and both halves, so a tap is ten loads for
/// eight FMAs.
template <int R, int SH, typename T>
inline void dwconv_3x3_2v(T* out_rows, const T* in_img, const T* w_tap0,
                                       const T* bias_vec, const DwConvParams& p,
                                       const Epilogue& epilogue, bool add_to, bool pre_act)
{
    // A group misses at most its top row when `oh*SH - PH` is -1 and at most its
    // bottom row when the last one is exactly `IH`; a wider reach keeps the
    // checked table below.
    const int64_t ih0 = p.oh * SH - p.PH;
    const int64_t ih_max = ih0 + (R - 1) * SH + 2;
    if (ih0 >= -1 && ih_max <= p.IH) {
        const bool top_ok = ih0 >= 0;
        const bool bot_ok = ih_max < p.IH;
        if (R == 4) {
            const bool plain = !add_to && !pre_act && epilogue.quant == EpilogueQuantType::None &&
                               epilogue.min_clip == -std::numeric_limits<float>::infinity() &&
                               epilogue.max_clip == std::numeric_limits<float>::infinity();
            const DwStore act =
                !nnops::epilogue_is_active(epilogue)                    ? kStorePlain :
                (plain && epilogue.type == EpilogueActivateType::Relu)  ? kStoreRelu :
                (plain && epilogue.type == EpilogueActivateType::Relu6) ? kStoreRelu6 :
                                                                          kStoreEpilogue;
            switch (act) {
            case kStorePlain:
                dwconv_3x3_2v_4row_topbot<SH, kStorePlain>(top_ok, bot_ok, out_rows, in_img,
                                                           w_tap0, bias_vec, p, epilogue, add_to,
                                                           pre_act);
                break;
            case kStoreRelu:
                dwconv_3x3_2v_4row_topbot<SH, kStoreRelu>(top_ok, bot_ok, out_rows, in_img,
                                                          w_tap0, bias_vec, p, epilogue, add_to,
                                                          pre_act);
                break;
            case kStoreRelu6:
                dwconv_3x3_2v_4row_topbot<SH, kStoreRelu6>(top_ok, bot_ok, out_rows, in_img,
                                                           w_tap0, bias_vec, p, epilogue, add_to,
                                                           pre_act);
                break;
            default:
                dwconv_3x3_2v_4row_topbot<SH, kStoreEpilogue>(top_ok, bot_ok, out_rows, in_img,
                                                              w_tap0, bias_vec, p, epilogue,
                                                              add_to, pre_act);
                break;
            }
            return;
        }
        dwconv_3x3_2v_masked_topbot<R, SH>(top_ok, bot_ok, out_rows, in_img, w_tap0, bias_vec, p,
                                           epilogue, add_to, pre_act);
        return;
    }

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

    const WindowSpan span = window_span(p, 3, p.DW);
    const int64_t ow_lo = span.lo;
    const int64_t ow_hi = span.hi;
    const int64_t off_ok_lo = span.tap_lo;
    const int64_t off_ok_hi = span.tap_hi;

    for (int64_t ow = p.ow0; ow < ow_end; ++ow) {
        V acc[R][2];
        for (int r = 0; r < R; ++r) {
            acc[r][0] = vbias[0];
            acc[r][1] = vbias[1];
        }

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
                    if (off < off_ok_lo || off >= off_ok_hi) {
                        continue;
                    }
                    const int64_t at = off * pix;
                    const T* wp = w_tap0 + (kh * 3 + kw) * wtap;
                    const V w0 = v_load(wp);
                    const V w1 = v_load(wp + kL);
                    for (int r = 0; r < R; ++r) {
                        if (row[r * SH + kh] == nullptr) {
                            continue;
                        }
                        const T* ip = row[r * SH + kh] + at;
                        acc[r][0] = v_fmadd(v_load(ip), w0, acc[r][0]);
                        acc[r][1] = v_fmadd(v_load(ip + kL), w1, acc[r][1]);
                    }
                }
            }
        }

        for (int r = 0; r < R; ++r) {
            if (p.oh + r >= p.OH) {
                continue;
            }
            T* dst = out_rows + r * p.out_row_stride + ow * pix;
            V a0 = acc[r][0];
            V a1 = acc[r][1];
            if (pre_act) {
                a0 = v_add(a0, v_load(dst));
                a1 = v_add(a1, v_load(dst + kL));
            }
            v_store_add(dst, nnops::backend::cpu::apply_epilogue_vec(epilogue, out_rows, a0),
                        add_to);
            v_store_add(dst + kL, nnops::backend::cpu::apply_epilogue_vec(epilogue, out_rows, a1),
                        add_to);
        }
    }
}

/// The 8-channel call: pick the kernel for the shape, then the vertical stride.
template <int R, typename T>
inline void dwconv_block_1v(T* out_rows, const T* in_img, const T* w_tap0, const T* bias_vec,
                                  const DwConvParams& p, const Epilogue& epilogue, bool add_to,
                                  bool pre_act)
{
    if (dwconv_3x3_ok(p)) {
        if (p.SH == 1) {
            dwconv_3x3_1v<R, 1>(out_rows, in_img, w_tap0, bias_vec, p, epilogue, add_to,
                                          pre_act);
        } else {
            dwconv_3x3_1v<R, 2>(out_rows, in_img, w_tap0, bias_vec, p, epilogue, add_to,
                                          pre_act);
        }
        return;
    }
    const FpSrc<T, true> src[1] = {FpSrc<T, true>{in_img, kChannelLanes}};
    const FpDst<T, true> dst[1] = {
        FpDst<T, true>{out_rows, p.out_row_stride, add_to, p.pix_stride, kChannelLanes}};
    const auto vb = bias_vec ? v_load(bias_vec) : v_zero(out_rows);
    const decltype(vb) vbias[1] = {vb};
    dwconv_rows<R, 1>(p, w_tap0, out_rows, epilogue, src, dst, vbias, pre_act);
}

template <int R, typename T>
inline void dwconv_block(T* out_rows, const T* in_img, const T* w_tap0, const T* bias_vec,
                                const DwConvParams& p, const Epilogue& epilogue, bool add_to,
                                bool pre_act, int64_t lanes)
{
    if (lanes > kChannelLanes) {
        if (lanes == 2 * kChannelLanes && dwconv_3x3_ok(p)) {
            if (p.SH == 1) {
                dwconv_3x3_2v<R, 1>(out_rows, in_img, w_tap0, bias_vec, p, epilogue,
                                                 add_to, pre_act);
            } else {
                dwconv_3x3_2v<R, 2>(out_rows, in_img, w_tap0, bias_vec, p, epilogue,
                                                 add_to, pre_act);
            }
            return;
        }
        const auto vb0 = bias_vec ? v_load(bias_vec) : v_zero(out_rows);
        const auto vb1 = bias_vec ? v_load(bias_vec + kChannelLanes) : v_zero(out_rows);
        const decltype(vb0) vbias[2] = {vb0, vb1};
        const FpSrc<T, true> src[2] = {FpSrc<T, true>{in_img, kChannelLanes},
                                       FpSrc<T, true>{in_img + kChannelLanes, kChannelLanes}};
        const FpDst<T, true> dst[2] = {
            FpDst<T, true>{out_rows, p.out_row_stride, add_to, p.pix_stride, kChannelLanes},
            FpDst<T, true>{out_rows + kChannelLanes, p.out_row_stride, add_to, p.pix_stride,
                           kChannelLanes}};
        dwconv_rows<R, 2>(p, w_tap0, out_rows, epilogue, src, dst, vbias, pre_act);
        return;
    }
    const auto vb = bias_vec ? load_lanes(bias_vec, lanes) : v_zero(out_rows);
    const decltype(vb) vbias[1] = {vb};
    if (lanes == kChannelLanes) {
        dwconv_block_1v<R>(out_rows, in_img, w_tap0, bias_vec, p, epilogue, add_to, pre_act);
    } else {
        const FpSrc<T, false> src[1] = {FpSrc<T, false>{in_img, lanes}};
        const FpDst<T, false> dst[1] = {
            FpDst<T, false>{out_rows, p.out_row_stride, add_to, p.pix_stride, lanes}};
        dwconv_rows<R, 1>(p, w_tap0, out_rows, epilogue, src, dst, vbias, pre_act);
    }
}

constexpr int64_t kMaxQuantTaps = 9 * 9 * 9; // 729

/// Dequantize this block's taps once per call into a staging block, then run the
/// f32 kernel. Pad lanes get scale 0.
template <int R, typename InT, typename OutT>
inline void dwconv_block_quant(OutT* out_rows, const InT* in_img, const InT* w_tap0,
                                      const float* w_scale, const int32_t* w_zero,
                                      const float* bias_vec, const DwConvParams& p,
                                      const Epilogue& epilogue, bool add_to, bool pre_act,
                                      int64_t lanes, float in_scale, float in_zero, float out_scale,
                                      float out_zero)
{
    const int64_t taps = p.KD * p.KH * p.KW;
    if (taps > kMaxQuantTaps) {
        return;
    } // the driver rejects this shape
    const int64_t C = p.pix_stride;

    alignas(32) float w_scale8[kChannelLanes];
    alignas(32) float w_zero8[kChannelLanes];
    for (int64_t l = 0; l < kChannelLanes; ++l) {
        const bool live = (l < lanes) && (w_scale != nullptr);
        w_scale8[l] = live ? w_scale[l] : 0.0f;
        w_zero8[l] = (live && w_zero != nullptr) ? static_cast<float>(w_zero[l]) : 0.0f;
    }

    alignas(32) float wq[kMaxQuantTaps * kChannelLanes];
    for (int64_t t = 0; t < taps; ++t) {
        InT wt[kChannelLanes] = {};
        std::memcpy(wt, w_tap0 + t * C, static_cast<size_t>(lanes) * sizeof(InT));
        v_store(&wq[t * kChannelLanes], dequantize_channel(wt, w_scale8, w_zero8));
    }

    // The staging block is dense 8-lane taps, so the tap stride is the vector
    // width even though the pixel stride is the channel count.
    DwConvParams pq = p;
    pq.w_tap_stride = kChannelLanes;

    const v_f32x8 vbias[1] = {bias_vec ? load_lanes(bias_vec, lanes) : v_zero_f32x8()};
    if (lanes == kChannelLanes) {
        const QuantSrc<InT, true> src[1] = {QuantSrc<InT, true>{in_img, in_scale, in_zero, lanes}};
        const QuantDst<OutT, true> dst[1] = {QuantDst<OutT, true>{
            out_rows, p.out_row_stride, out_scale, out_zero, add_to, p.pix_stride, lanes}};
        dwconv_rows<R, 1>(pq, wq, &out_scale, epilogue, src, dst, vbias, pre_act);
    } else {
        const QuantSrc<InT, false> src[1] = {
            QuantSrc<InT, false>{in_img, in_scale, in_zero, lanes}};
        const QuantDst<OutT, false> dst[1] = {QuantDst<OutT, false>{
            out_rows, p.out_row_stride, out_scale, out_zero, add_to, p.pix_stride, lanes}};
        dwconv_rows<R, 1>(pq, wq, &out_scale, epilogue, src, dst, vbias, pre_act);
    }
}

} // namespace nnops::kernel
