#pragma once
/// @file mma_i8mm.hpp
/// @brief AArch64 FEAT_I8MM (SMMLA) int8 MMA kernels: s8×s8 → int32.
///
/// One SMMLA multiplies a 2×8 A fragment by an 8×2 B fragment and accumulates
/// the 2×2 int32 result — four times the work of one SDOT lane group, at
/// exactly twice the instruction cost on the cores measured, so the two routes
/// have the same MAC rate. The operands are therefore k-grouped by eight, and a
/// kernel here covers a whole M panel times the N side of one 12-column B
/// panel — the same shape, tile geometry and MR ladder as mma_i8.hpp. The panel
/// width is 12 on purpose: every plan, panel count and blob above this file is
/// shared with the SDOT route, so only the panel *stride* (k rounded up to
/// eight) and the operand layout inside it differ.
///
/// Two A presentations share the same B (the pack_i8mm.hpp layouts):
///   - packed A (mma_pack_mr{8,4,1}_nr12_i8mm): [ceil(kc/8)][rows][8] bytes, so
///     a 2-row fragment — rows r, r+1, eight k values each — is one 16-byte
///     load of the panel.
///   - row-major A (mma_direct_mr{8,4,1}_nr12_i8mm): the same fragment is two
///     8-byte loads through `lda` and a combine. A short final k group is staged
///     through a zeroed buffer, because a row-major row holds exactly kc bytes.
///
/// The accumulators hold each 2×2 block row-major: [r0c0, r0c1, r1c0, r1c1], so
/// a row's two columns are the low (or high) 64 bits and vtrn1q/vtrn2q pair two
/// accumulators into one four-column vector.
///
/// Clamping and zero_mode are the SDOT kernels' contract; `mr` is the live row
/// count, rows past it alias the last live one in A and in C, and only the live
/// rows are stored. `ldb` (0 = the kernel's own slot) overrides the
/// panel-to-panel step for a prepacked B pitched for another k length.
///
/// The bodies compile only when the build enabled i8mm (`NNOPS_ENABLE_I8MM`
/// defines `NNOPS_ARCH_I8MM`); without the macro i8mm_enabled() is a false
/// constant and the whole file is empty, which is the pure-SDOT build.

#if defined(NNOPS_ARCH_I8MM)

#include <algorithm>
#include <arm_neon.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>
#include "backend/cpu/aarch64/i8_requant_store.hpp"
#include "backend/cpu/common/common_utils.hpp"

namespace nnops::backend::cpu::aarch64 {

/// B panel width; shared with the SDOT route, so no plan forks on it.
inline constexpr int mma_i8mm_nr_max = 12;
/// Bytes of one 8-k group of a full B panel (12 columns x 8 k).
inline constexpr int mma_i8mm_group_bytes = mma_i8mm_nr_max * 8;

/// A panel slot's byte pitch: the caller's `ldb` when it gave one, else the
/// uniform slot `ceil(kbytes/8)` groups of 96 bytes take.
inline constexpr int mma_i8mm_panel_stride(int ldb, int kbytes) noexcept
{
    return (ldb > 0) ? ldb :
                       static_cast<int>(mma_pack_align(mma_i8mm_group_bytes * ((kbytes + 7) >> 3)));
}

/// One row pair's twelve columns: pair the six 2×2 accumulators into the even
/// row's three vectors and the odd row's three, add C (unless zero_mode),
/// clamp, store. `co` may alias `ce`; it is only written when store_odd.
template <bool zero_mode>
__attribute__((target("i8mm"), always_inline)) inline void
i8mm_store12(int32_t* NNOPS_RESTRICT ce, int32_t* NNOPS_RESTRICT co, bool store_odd, int32x4_t c0,
             int32x4_t c1, int32x4_t c2, int32x4_t c3, int32x4_t c4, int32x4_t c5, int32x4_t vmin,
             int32x4_t vmax) noexcept
{
    int32x4_t v_e0 =
        vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(c0), vreinterpretq_u64_s32(c1)));
    int32x4_t v_e1 =
        vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(c2), vreinterpretq_u64_s32(c3)));
    int32x4_t v_e2 =
        vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(c4), vreinterpretq_u64_s32(c5)));
    int32x4_t v_o0 =
        vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(c0), vreinterpretq_u64_s32(c1)));
    int32x4_t v_o1 =
        vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(c2), vreinterpretq_u64_s32(c3)));
    int32x4_t v_o2 =
        vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(c4), vreinterpretq_u64_s32(c5)));

    if constexpr (!zero_mode) {
        v_e0 = vaddq_s32(v_e0, vld1q_s32(ce + 0));
        v_e1 = vaddq_s32(v_e1, vld1q_s32(ce + 4));
        v_e2 = vaddq_s32(v_e2, vld1q_s32(ce + 8));
    }
    vst1q_s32(ce + 0, vminq_s32(vmaxq_s32(v_e0, vmin), vmax));
    vst1q_s32(ce + 4, vminq_s32(vmaxq_s32(v_e1, vmin), vmax));
    vst1q_s32(ce + 8, vminq_s32(vmaxq_s32(v_e2, vmin), vmax));

    if (store_odd) {
        if constexpr (!zero_mode) {
            v_o0 = vaddq_s32(v_o0, vld1q_s32(co + 0));
            v_o1 = vaddq_s32(v_o1, vld1q_s32(co + 4));
            v_o2 = vaddq_s32(v_o2, vld1q_s32(co + 8));
        }
        vst1q_s32(co + 0, vminq_s32(vmaxq_s32(v_o0, vmin), vmax));
        vst1q_s32(co + 4, vminq_s32(vmaxq_s32(v_o1, vmin), vmax));
        vst1q_s32(co + 8, vminq_s32(vmaxq_s32(v_o2, vmin), vmax));
    }
}

/// The live width of a panel that is not a whole 12 columns: lane stores only.
/// Kept for the row-major-A kernels, which have no caller today; the packed
/// route narrows the panel itself (see the narrow tail below).
template <bool zero_mode>
__attribute__((target("i8mm"), always_inline)) inline void
i8mm_store12_tail(int32_t* NNOPS_RESTRICT ce, int32_t* NNOPS_RESTRICT co, bool store_odd, int rem,
                  int32x4_t c0, int32x4_t c1, int32x4_t c2, int32x4_t c3, int32x4_t c4,
                  int32x4_t c5, int32_t clamp_min, int32_t clamp_max) noexcept
{
    const auto put = [&](int32_t* dst, int, int32_t x) {
        dst[0] = std::min(std::max(x, clamp_min), clamp_max);
    };
    if (rem > 0) {
        int32_t x = vgetq_lane_s32(c0, 0);
        if constexpr (!zero_mode) {
            x += ce[0];
        }
        put(ce, 0, x);
    }
    if (rem > 1) {
        int32_t x = vgetq_lane_s32(c0, 1);
        if constexpr (!zero_mode) {
            x += ce[1];
        }
        put(ce + 1, 1, x);
    }
    if (rem > 2) {
        int32_t x = vgetq_lane_s32(c1, 0);
        if constexpr (!zero_mode) {
            x += ce[2];
        }
        put(ce + 2, 2, x);
    }
    if (rem > 3) {
        int32_t x = vgetq_lane_s32(c1, 1);
        if constexpr (!zero_mode) {
            x += ce[3];
        }
        put(ce + 3, 3, x);
    }
    if (rem > 4) {
        int32_t x = vgetq_lane_s32(c2, 0);
        if constexpr (!zero_mode) {
            x += ce[4];
        }
        put(ce + 4, 4, x);
    }
    if (rem > 5) {
        int32_t x = vgetq_lane_s32(c2, 1);
        if constexpr (!zero_mode) {
            x += ce[5];
        }
        put(ce + 5, 5, x);
    }
    if (rem > 6) {
        int32_t x = vgetq_lane_s32(c3, 0);
        if constexpr (!zero_mode) {
            x += ce[6];
        }
        put(ce + 6, 6, x);
    }
    if (rem > 7) {
        int32_t x = vgetq_lane_s32(c3, 1);
        if constexpr (!zero_mode) {
            x += ce[7];
        }
        put(ce + 7, 7, x);
    }
    if (rem > 8) {
        int32_t x = vgetq_lane_s32(c4, 0);
        if constexpr (!zero_mode) {
            x += ce[8];
        }
        put(ce + 8, 8, x);
    }
    if (rem > 9) {
        int32_t x = vgetq_lane_s32(c4, 1);
        if constexpr (!zero_mode) {
            x += ce[9];
        }
        put(ce + 9, 9, x);
    }
    if (rem > 10) {
        int32_t x = vgetq_lane_s32(c5, 0);
        if constexpr (!zero_mode) {
            x += ce[10];
        }
        put(ce + 10, 10, x);
    }
    if (rem > 11) {
        int32_t x = vgetq_lane_s32(c5, 1);
        if constexpr (!zero_mode) {
            x += ce[11];
        }
        put(ce + 11, 11, x);
    }
    if (store_odd) {
        if (rem > 0) {
            int32_t x = vgetq_lane_s32(c0, 2);
            if constexpr (!zero_mode) {
                x += co[0];
            }
            put(co, 0, x);
        }
        if (rem > 1) {
            int32_t x = vgetq_lane_s32(c0, 3);
            if constexpr (!zero_mode) {
                x += co[1];
            }
            put(co + 1, 1, x);
        }
        if (rem > 2) {
            int32_t x = vgetq_lane_s32(c1, 2);
            if constexpr (!zero_mode) {
                x += co[2];
            }
            put(co + 2, 2, x);
        }
        if (rem > 3) {
            int32_t x = vgetq_lane_s32(c1, 3);
            if constexpr (!zero_mode) {
                x += co[3];
            }
            put(co + 3, 3, x);
        }
        if (rem > 4) {
            int32_t x = vgetq_lane_s32(c2, 2);
            if constexpr (!zero_mode) {
                x += co[4];
            }
            put(co + 4, 4, x);
        }
        if (rem > 5) {
            int32_t x = vgetq_lane_s32(c2, 3);
            if constexpr (!zero_mode) {
                x += co[5];
            }
            put(co + 5, 5, x);
        }
        if (rem > 6) {
            int32_t x = vgetq_lane_s32(c3, 2);
            if constexpr (!zero_mode) {
                x += co[6];
            }
            put(co + 6, 6, x);
        }
        if (rem > 7) {
            int32_t x = vgetq_lane_s32(c3, 3);
            if constexpr (!zero_mode) {
                x += co[7];
            }
            put(co + 7, 7, x);
        }
        if (rem > 8) {
            int32_t x = vgetq_lane_s32(c4, 2);
            if constexpr (!zero_mode) {
                x += co[8];
            }
            put(co + 8, 8, x);
        }
        if (rem > 9) {
            int32_t x = vgetq_lane_s32(c4, 3);
            if constexpr (!zero_mode) {
                x += co[9];
            }
            put(co + 9, 9, x);
        }
        if (rem > 10) {
            int32_t x = vgetq_lane_s32(c5, 2);
            if constexpr (!zero_mode) {
                x += co[10];
            }
            put(co + 10, 10, x);
        }
        if (rem > 11) {
            int32_t x = vgetq_lane_s32(c5, 3);
            if constexpr (!zero_mode) {
                x += co[11];
            }
            put(co + 11, 11, x);
        }
    }
}

// =========================================================================
//  narrow tail
// =========================================================================
//
// A panel is 12 columns wide and N is rarely a multiple of it. The last panel
// narrows to the live remainder instead of paying for twelve columns of dead
// lanes: 1-4 live columns cost 4, 5-8 cost 8, 9-11 cost 12. `NB` is the column
// block count (one block is two columns), `MRT` the rung's live row count and
// `SLIT` the k split that keeps enough SMMLA chains in flight when NB is small.
//
// A rung is always emitted at its own height (walk_levels_exact), so a tail
// carries MRT live rows and the packed-A walk is MRT*8 bytes per k group.

/// Row pairs one k group of a tail panel covers.
template <int MRT> inline constexpr int i8mm_tail_pairs = (MRT + 1) / 2;

/// One k group of a tail panel: MRT rows of packed A against NB column blocks.
/// `a_ptr` advances past this rung's MRT*8 bytes; the caller owns the B stride.
template <int MRT, int NB>
__attribute__((target("i8mm"), always_inline)) inline void
i8mm_tail_group(const int8_t*& a_ptr, const int8_t* b_ptr,
                int32x4_t (&acc)[i8mm_tail_pairs<MRT>][NB]) noexcept
{
    constexpr int kPairs = i8mm_tail_pairs<MRT>;
    int8x16_t va[kPairs];
    if constexpr (MRT == 1) {
        // The fragment's second row is zero, so the odd lanes stay dead.
        va[0] = vcombine_s8(vld1_s8(a_ptr), vdup_n_s8(0));
    } else {
        for (int p = 0; p < kPairs; ++p) {
            va[p] = vld1q_s8(a_ptr + p * 16);
        }
    }
    a_ptr += MRT * 8;
    for (int j = 0; j < NB; ++j) {
        const int8x16_t vb = vld1q_s8(b_ptr + j * 16);
        for (int p = 0; p < kPairs; ++p) {
            acc[p][j] = vmmlaq_s32(acc[p][j], va[p], vb);
        }
    }
}

/// The tail panel's whole K. `SLIT == 2` walks the even and odd k groups in
/// separate passes so each half's accumulators stay distinct registers (a
/// run-time half index spills them); the fold afterwards is exact integer
/// addition, so the result is bit-identical to the unsplit form.
template <int MRT, int NB, int SLIT>
__attribute__((target("i8mm"), always_inline)) inline void
i8mm_tail_mma(const int8_t* A, const int8_t* B, int kg,
              int32x4_t (&acc)[SLIT][i8mm_tail_pairs<MRT>][NB]) noexcept
{
    constexpr int kPairs = i8mm_tail_pairs<MRT>;
    constexpr int kBB = mma_i8mm_group_bytes;
    if constexpr (SLIT == 1) {
        const int8_t* a = A;
        const int8_t* b = B;
        for (int k = 0; k < kg; ++k, b += kBB) {
            i8mm_tail_group<MRT, NB>(a, b, acc[0]);
        }
    } else {
        // Each pass skips every other k group, so the A block is addressed from
        // the group index rather than advanced by the group walk.
        {
            const int8_t* b = B;
            for (int k = 0; k < kg; k += 2, b += 2 * kBB) {
                const int8_t* a = A + static_cast<size_t>(k) * MRT * 8;
                i8mm_tail_group<MRT, NB>(a, b, acc[0]);
            }
        }
        {
            const int8_t* b = B + kBB;
            for (int k = 1; k < kg; k += 2, b += 2 * kBB) {
                const int8_t* a = A + static_cast<size_t>(k) * MRT * 8;
                i8mm_tail_group<MRT, NB>(a, b, acc[1]);
            }
        }
        for (int p = 0; p < kPairs; ++p) {
            for (int j = 0; j < NB; ++j) {
                acc[0][p][j] = vaddq_s32(acc[0][p][j], acc[1][p][j]);
            }
        }
    }
}

/// Two column blocks as two four-column vectors: the even row's, then the odd
/// row's. One row per fragment (MRT == 1) has only the low 64 bits live.
template <int MRT>
__attribute__((target("i8mm"), always_inline)) inline void
i8mm_tail_pair4(int32x4_t ca, int32x4_t cb, int32x4_t& ve, int32x4_t& vo) noexcept
{
    if constexpr (MRT == 1) {
        ve = vcombine_s32(vget_low_s32(ca), vget_low_s32(cb));
        vo = ve;
    } else {
        ve = vreinterpretq_s32_u64(
            vtrn1q_u64(vreinterpretq_u64_s32(ca), vreinterpretq_u64_s32(cb)));
        vo = vreinterpretq_s32_u64(
            vtrn2q_u64(vreinterpretq_u64_s32(ca), vreinterpretq_u64_s32(cb)));
    }
}

/// Store the narrow panel's accumulators. A group of four columns is stored as
/// a vector; only the last group can be ragged, and only it goes lane by lane —
/// and reads C, when accumulating, only where a lane is live.
template <bool zero_mode, int MRT, int NB>
__attribute__((target("i8mm"), always_inline)) inline void
i8mm_tail_store(int32_t* C, int ldc, int mr, int live,
                int32x4_t (&acc)[i8mm_tail_pairs<MRT>][NB], int32_t clamp_min,
                int32_t clamp_max) noexcept
{
    constexpr int kPairs = i8mm_tail_pairs<MRT>;
    const int32x4_t vmin = vdupq_n_s32(clamp_min);
    const int32x4_t vmax = vdupq_n_s32(clamp_max);
    // Rows past the live count alias the last live one, as the full panel does.
    int32_t* row[MRT];
    for (int r = 0; r < MRT; ++r) {
        row[r] = C + std::min(r, mr - 1) * ldc;
    }
    const auto put = [&](int32_t* dst, bool full, int col, int32x4_t v) {
        if (full) {
            if constexpr (!zero_mode) {
                v = vaddq_s32(v, vld1q_s32(dst));
            }
            vst1q_s32(dst, vminq_s32(vmaxq_s32(v, vmin), vmax));
        } else {
            int32_t x[4];
            vst1q_s32(x, v);
            for (int c = 0; c < 4 && col + c < live; ++c) {
                int32_t y = x[c];
                if constexpr (!zero_mode) {
                    y += dst[c];
                }
                dst[c] = std::min(std::max(y, clamp_min), clamp_max);
            }
        }
    };
    for (int g = 0; g < NB / 2; ++g) {
        const int col = 4 * g;
        const bool full = col + 4 <= live;
        for (int p = 0; p < kPairs; ++p) {
            int32x4_t ve, vo;
            i8mm_tail_pair4<MRT>(acc[p][2 * g], acc[p][2 * g + 1], ve, vo);
            put(row[2 * p] + col, full, col, ve);
            if constexpr (MRT > 1) {
                put(row[2 * p + 1] + col, full, col, vo);
            }
        }
    }
}

/// One (NB, SLIT) tail shape: zero the accumulators, run the MMA, store.
template <bool zero_mode, int MRT, int NB, int SLIT>
__attribute__((target("i8mm"), always_inline)) inline void
i8mm_tail_nb(int32_t* C, int ldc, int mr, int live, int kg, const int8_t* A, const int8_t* B,
             int32_t clamp_min, int32_t clamp_max) noexcept
{
    int32x4_t acc[SLIT][i8mm_tail_pairs<MRT>][NB];
    for (int h = 0; h < SLIT; ++h) {
        for (int p = 0; p < i8mm_tail_pairs<MRT>; ++p) {
            for (int j = 0; j < NB; ++j) {
                acc[h][p][j] = vdupq_n_s32(0);
            }
        }
    }
    i8mm_tail_mma<MRT, NB, SLIT>(A, B, kg, acc);
    i8mm_tail_store<zero_mode, MRT, NB>(C, ldc, mr, live, acc[0], clamp_min, clamp_max);
}

/// The narrow panel through one rung: NB=2 for 1-4 live columns, 4 for 5-8,
/// 6 for 9-11. NB=2 splits K in two to keep sixteen SMMLA chains alive.
template <bool zero_mode, int MRT>
__attribute__((target("i8mm"), always_inline)) inline void
i8mm_tail(int32_t* C, int ldc, int mr, int live, int kg, const int8_t* A, const int8_t* B,
          int32_t clamp_min, int32_t clamp_max) noexcept
{
    if (live <= 0) {
        return;
    }
    if (live <= 4) {
        i8mm_tail_nb<zero_mode, MRT, 2, 2>(C, ldc, mr, live, kg, A, B, clamp_min, clamp_max);
    } else if (live <= 8) {
        i8mm_tail_nb<zero_mode, MRT, 4, 1>(C, ldc, mr, live, kg, A, B, clamp_min, clamp_max);
    } else {
        i8mm_tail_nb<zero_mode, MRT, 6, 1>(C, ldc, mr, live, kg, A, B, clamp_min, clamp_max);
    }
}

// =========================================================================
//  the requantizing store
// =========================================================================
//
// The same accumulators written as the destination's bytes instead of int32:
//
//   out = sat(round_away((acc + bias[n]) * (s_a/s_out * s_b[n])) + zp_out)
//
// Relu is a floor of zero before the scale — the compensated accumulate is the
// true product, so its sign is the activation's. `vqmovn`/`vqmovun` saturate at
// the destination's range, which is exactly the clamp the row pass applies, so
// no separate min/max is needed, and the zero point is added with a saturating
// add so a product past int32 still lands on an end byte.
//
// The kernels take the per-column bias already folded with the A-side shift
// ((a_bias_offset + zp_a) * r_b[n], see the caller), so one vector add is the
// whole per-element correction.

/// One 4-column group of a row pair: transpose, scale, store the live columns.
template <int MRT, class TOut>
__attribute__((target("i8mm"), always_inline)) inline void
i8mm_store_pair_epi(TOut* ce, TOut* co, int col, int live, int32x4_t ca, int32x4_t cb,
                    float32x4_t vscale, int32x4_t vbias, int32x4_t vlo,
                    int32x4_t vzp) noexcept
{
    int32x4_t ve, vo;
    i8mm_tail_pair4<MRT>(ca, cb, ve, vo);
    const float32x4_t sc = vscale;
    const int32x4_t bi = vbias;
    const bool full = col + 4 <= live;
    if (full) {
        i8_store4q<TOut>(ce + col, ve, sc, bi, vlo, vzp);
    } else {
        i8_store4q_lanes<TOut>(ce + col, live - col, ve, sc, bi, vlo, vzp);
    }
    if constexpr (MRT > 1) {
        if (full) {
            i8_store4q<TOut>(co + col, vo, sc, bi, vlo, vzp);
        } else {
            i8_store4q_lanes<TOut>(co + col, live - col, vo, sc, bi, vlo, vzp);
        }
    }
}

/// A full twelve-column panel: three 4-column groups per row pair.
template <int MRT, class TOut>
__attribute__((target("i8mm"), always_inline)) inline void
i8mm_store12_epi(TOut* NNOPS_RESTRICT ce, TOut* NNOPS_RESTRICT co, int32x4_t c0, int32x4_t c1,
                 int32x4_t c2, int32x4_t c3, int32x4_t c4, int32x4_t c5,
                 const float32x4_t (&sc)[3], const int32x4_t (&bi)[3], int32x4_t vlo,
                 int32x4_t vzp) noexcept
{
    i8mm_store_pair_epi<MRT, TOut>(ce, co, 0, 12, c0, c1, sc[0], bi[0], vlo, vzp);
    i8mm_store_pair_epi<MRT, TOut>(ce, co, 4, 12, c2, c3, sc[1], bi[1], vlo, vzp);
    i8mm_store_pair_epi<MRT, TOut>(ce, co, 8, 12, c4, c5, sc[2], bi[2], vlo, vzp);
}

/// The narrow panel through one rung, requantized.
template <int MRT, int NB, class TOut>
__attribute__((target("i8mm"), always_inline)) inline void
i8mm_tail_store_epi(TOut* C, int64_t ldm, int mr, int live,
                    int32x4_t (&acc)[i8mm_tail_pairs<MRT>][NB], float32x4_t vk,
                    float32x4_t vreq, const float* s_b, const int32_t* col_bias, int32x4_t vlo,
                    int32x4_t vzp) noexcept
{
    constexpr int kPairs = i8mm_tail_pairs<MRT>;
    TOut* row[MRT];
    for (int r = 0; r < MRT; ++r) {
        row[r] = C + std::min(r, mr - 1) * ldm;
    }
    float32x4_t sc[NB / 2];
    int32x4_t bi[NB / 2];
    i8_col_q_live<NB / 2>(sc, bi, vk, vreq, s_b, col_bias, live);
    for (int g = 0; g < NB / 2; ++g) {
        const int col = 4 * g;
        for (int p = 0; p < kPairs; ++p) {
            i8mm_store_pair_epi<MRT, TOut>(row[2 * p], row[(MRT > 1) ? 2 * p + 1 : 2 * p], col,
                                           live, acc[p][2 * g], acc[p][2 * g + 1], sc[g], bi[g],
                                           vlo, vzp);
        }
    }
}

/// One (NB, SLIT) requantizing tail, mirroring i8mm_tail_nb.
template <int MRT, int NB, int SLIT, class TOut>
__attribute__((target("i8mm"), always_inline)) inline void
i8mm_tail_epi_nb(TOut* C, int64_t ldm, int mr, int live, int kg, const int8_t* A,
                 const int8_t* B, float32x4_t vk, float32x4_t vreq, const float* s_b,
                 const int32_t* col_bias, int32x4_t vlo, int32x4_t vzp) noexcept
{
    int32x4_t acc[SLIT][i8mm_tail_pairs<MRT>][NB];
    for (int h = 0; h < SLIT; ++h) {
        for (int p = 0; p < i8mm_tail_pairs<MRT>; ++p) {
            for (int j = 0; j < NB; ++j) {
                acc[h][p][j] = vdupq_n_s32(0);
            }
        }
    }
    i8mm_tail_mma<MRT, NB, SLIT>(A, B, kg, acc);
    i8mm_tail_store_epi<MRT, NB, TOut>(C, ldm, mr, live, acc[0], vk, vreq, s_b, col_bias, vlo,
                                       vzp);
}

/// The narrow panel's requantizing store, dispatched on the live width.
template <int MRT, class TOut>
__attribute__((target("i8mm"), always_inline)) inline void
i8mm_tail_epi(TOut* C, int64_t ldm, int mr, int live, int kg, const int8_t* A, const int8_t* B,
              float32x4_t vk, float32x4_t vreq, const float* s_b, const int32_t* col_bias,
              int32x4_t vlo, int32x4_t vzp) noexcept
{
    if (live <= 0) {
        return;
    }
    if (live <= 4) {
        i8mm_tail_epi_nb<MRT, 2, 2, TOut>(C, ldm, mr, live, kg, A, B, vk, vreq, s_b, col_bias,
                                          vlo, vzp);
    } else if (live <= 8) {
        i8mm_tail_epi_nb<MRT, 4, 1, TOut>(C, ldm, mr, live, kg, A, B, vk, vreq, s_b, col_bias,
                                          vlo, vzp);
    } else {
        i8mm_tail_epi_nb<MRT, 6, 1, TOut>(C, ldm, mr, live, kg, A, B, vk, vreq, s_b, col_bias,
                                          vlo, vzp);
    }
}

// =========================================================================
//  one walk: packed A (int32 and requantized) and row-major A
// =========================================================================
//
// One template per route and height. A panel is six two-column blocks per row
// pair, so the accumulators are [pairs][6] and every index into them is a
// literal: a loop to a template bound costs nothing where the compiler unrolls
// it and everything where it does not — the array is then indexed at run time
// and the whole block moves to the stack.

/// The C rows, aliasing the last live one past `mr`.
template <int MR, class T>
__attribute__((always_inline)) inline void
i8mm_rows(T* C, int64_t ldm, int mr, T* (&row)[MR]) noexcept
{
    row[0] = C;
    if constexpr (MR > 1) {
        row[1] = C + std::min<int64_t>(1, mr - 1) * ldm;
    }
    if constexpr (MR > 2) {
        row[2] = C + std::min<int64_t>(2, mr - 1) * ldm;
    }
    if constexpr (MR > 3) {
        row[3] = C + std::min<int64_t>(3, mr - 1) * ldm;
    }
    if constexpr (MR > 4) {
        row[4] = C + std::min<int64_t>(4, mr - 1) * ldm;
    }
    if constexpr (MR > 5) {
        row[5] = C + std::min<int64_t>(5, mr - 1) * ldm;
    }
    if constexpr (MR > 6) {
        row[6] = C + std::min<int64_t>(6, mr - 1) * ldm;
    }
    if constexpr (MR > 7) {
        row[7] = C + std::min<int64_t>(7, mr - 1) * ldm;
    }
}

/// The row-major A rows, aliasing the last live one.
template <int MR>
__attribute__((always_inline)) inline void
i8mm_a_rows(const int8_t* A, int lda, int mr, const int8_t* (&a)[MR]) noexcept
{
    a[0] = A;
    if constexpr (MR > 1) {
        a[1] = A + std::min(1, mr - 1) * lda;
    }
    if constexpr (MR > 2) {
        a[2] = A + std::min(2, mr - 1) * lda;
    }
    if constexpr (MR > 3) {
        a[3] = A + std::min(3, mr - 1) * lda;
    }
    if constexpr (MR > 4) {
        a[4] = A + std::min(4, mr - 1) * lda;
    }
    if constexpr (MR > 5) {
        a[5] = A + std::min(5, mr - 1) * lda;
    }
    if constexpr (MR > 6) {
        a[6] = A + std::min(6, mr - 1) * lda;
    }
    if constexpr (MR > 7) {
        a[7] = A + std::min(7, mr - 1) * lda;
    }
}

/// The odd row of pair `P`: a pair whose second row is past `MR` folds onto
/// the last live row (the one-row rung's fragment has no second row at all).
template <int MR, int P>
inline constexpr int i8mm_odd_row = (2 * P + 1 < MR) ? 2 * P + 1 : MR - 1;

/// One two-column block against every row pair's fragment.
template <int P, int J>
__attribute__((target("i8mm"), always_inline)) inline void
i8mm_fma_block(const int8x16_t (&va)[P], const int8_t* b_ptr,
               int32x4_t (&acc)[P][6]) noexcept
{
    const int8x16_t vb = vld1q_s8(b_ptr + J * 16);
    acc[0][J] = vmmlaq_s32(acc[0][J], va[0], vb);
    if constexpr (P > 1) {
        acc[1][J] = vmmlaq_s32(acc[1][J], va[1], vb);
    }
    if constexpr (P > 2) {
        acc[2][J] = vmmlaq_s32(acc[2][J], va[2], vb);
        acc[3][J] = vmmlaq_s32(acc[3][J], va[3], vb);
    }
}

/// One k group of a packed-A panel.
template <int MR>
__attribute__((target("i8mm"), always_inline)) inline void
i8mm_group(const int8_t*& a_ptr, const int8_t* b_ptr,
           int32x4_t (&acc)[i8mm_tail_pairs<MR>][6]) noexcept
{
    constexpr int P = i8mm_tail_pairs<MR>;
    int8x16_t va[P];
    if constexpr (MR == 1) {
        // The fragment's second row is zero, so the odd lanes stay dead.
        va[0] = vcombine_s8(vld1_s8(a_ptr), vdup_n_s8(0));
        a_ptr += 8;
    } else {
        va[0] = vld1q_s8(a_ptr + 0 * 16);
        if constexpr (MR > 2) {
            va[1] = vld1q_s8(a_ptr + 1 * 16);
        }
        if constexpr (MR > 4) {
            va[2] = vld1q_s8(a_ptr + 2 * 16);
            va[3] = vld1q_s8(a_ptr + 3 * 16);
        }
        a_ptr += MR * 8;
    }
    i8mm_fma_block<P, 0>(va, b_ptr, acc);
    i8mm_fma_block<P, 1>(va, b_ptr, acc);
    i8mm_fma_block<P, 2>(va, b_ptr, acc);
    i8mm_fma_block<P, 3>(va, b_ptr, acc);
    i8mm_fma_block<P, 4>(va, b_ptr, acc);
    i8mm_fma_block<P, 5>(va, b_ptr, acc);
}

/// Zero every accumulator of a panel.
template <int MR, std::size_t... I>
__attribute__((target("i8mm"), always_inline)) inline void
i8mm_acc_zero(int32x4_t (&acc)[i8mm_tail_pairs<MR>][6], std::index_sequence<I...>) noexcept
{
    const int32x4_t vz = vdupq_n_s32(0);
    ((acc[I / 6][I % 6] = vz), ...);
}

/// The panel's whole K from packed A into zeroed `acc`.
template <int MR>
__attribute__((target("i8mm"), always_inline)) inline void
i8mm_mma(const int8_t* A, const int8_t* B, int kg,
         int32x4_t (&acc)[i8mm_tail_pairs<MR>][6]) noexcept
{
    i8mm_acc_zero<MR>(acc, std::make_index_sequence<i8mm_tail_pairs<MR> * 6>{});
    const int8_t* a = A;
    const int8_t* b = B;
    for (int k = 0; k < kg; ++k, b += mma_i8mm_group_bytes) {
        i8mm_group<MR>(a, b, acc);
    }
}

/// The row pairs' fragments out of a caller's matrix. A short final k group is
/// staged through a zeroed row per A row — a row holds exactly `kc` bytes — and
/// the pointers do not advance on it.
template <int MR>
__attribute__((target("i8mm"), always_inline)) inline void
i8mm_frags(const int8_t* (&a)[MR], int k, int kg, int kc,
           int8x16_t (&va)[i8mm_tail_pairs<MR>]) noexcept
{
    if (k + 1 < kg || (kc & 7) == 0) {
        if constexpr (MR == 1) {
            va[0] = vcombine_s8(vld1_s8(a[0]), vdup_n_s8(0));
        } else {
            va[0] = vcombine_s8(vld1_s8(a[0]), vld1_s8(a[1]));
            if constexpr (MR > 2) {
                va[1] = vcombine_s8(vld1_s8(a[2]), vld1_s8(a[3]));
            }
            if constexpr (MR > 4) {
                va[2] = vcombine_s8(vld1_s8(a[4]), vld1_s8(a[5]));
                va[3] = vcombine_s8(vld1_s8(a[6]), vld1_s8(a[7]));
            }
        }
        if constexpr (MR == 1) {
            a[0] += 8;
        } else {
            for (int r = 0; r < MR; ++r) {
                a[r] += 8;
            }
        }
    } else {
        const int live = kc - 8 * k;
        int8_t t[MR][8] = {};
        for (int r = 0; r < MR; ++r) {
            std::memcpy(t[r], a[r], static_cast<size_t>(live));
        }
        if constexpr (MR == 1) {
            va[0] = vcombine_s8(vld1_s8(t[0]), vdup_n_s8(0));
        } else {
            va[0] = vcombine_s8(vld1_s8(t[0]), vld1_s8(t[1]));
            if constexpr (MR > 2) {
                va[1] = vcombine_s8(vld1_s8(t[2]), vld1_s8(t[3]));
            }
            if constexpr (MR > 4) {
                va[2] = vcombine_s8(vld1_s8(t[4]), vld1_s8(t[5]));
                va[3] = vcombine_s8(vld1_s8(t[6]), vld1_s8(t[7]));
            }
        }
    }
}

/// The panel's whole K from row-major rows into zeroed `acc`.
template <int MR>
__attribute__((target("i8mm"), always_inline)) inline void
i8mm_mma_rows(const int8_t* const (&A)[MR], const int8_t* B, int kg, int kc,
              int32x4_t (&acc)[i8mm_tail_pairs<MR>][6]) noexcept
{
    constexpr int P = i8mm_tail_pairs<MR>;
    i8mm_acc_zero<MR>(acc, std::make_index_sequence<P * 6>{});
    const int8_t* a[MR];
    for (int r = 0; r < MR; ++r) {
        a[r] = A[r];
    }
    const int8_t* b = B;
    for (int k = 0; k < kg; ++k, b += mma_i8mm_group_bytes) {
        int8x16_t va[P];
        i8mm_frags<MR>(a, k, kg, kc, va);
        i8mm_fma_block<P, 0>(va, b, acc);
        i8mm_fma_block<P, 1>(va, b, acc);
        i8mm_fma_block<P, 2>(va, b, acc);
        i8mm_fma_block<P, 3>(va, b, acc);
        i8mm_fma_block<P, 4>(va, b, acc);
        i8mm_fma_block<P, 5>(va, b, acc);
    }
}

/// Store one panel of int32 accumulators: a row pair at a time, every row's C
/// read before that pair is written.
template <int MR, bool zero_mode>
__attribute__((target("i8mm"), always_inline)) inline void
i8mm_store_all(int32_t* const (&row)[MR], int col0, int mr,
               const int32x4_t (&acc)[i8mm_tail_pairs<MR>][6], int32x4_t vmin,
               int32x4_t vmax) noexcept
{
    i8mm_store12<zero_mode>(row[0] + col0, row[i8mm_odd_row<MR, 0>] + col0, MR > 1 && mr > 1,
                            acc[0][0], acc[0][1], acc[0][2], acc[0][3], acc[0][4], acc[0][5],
                            vmin, vmax);
    if constexpr (MR > 2) {
        i8mm_store12<zero_mode>(row[2] + col0, row[i8mm_odd_row<MR, 1>] + col0, mr > 3,
                                acc[1][0], acc[1][1], acc[1][2], acc[1][3], acc[1][4], acc[1][5],
                                vmin, vmax);
    }
    if constexpr (MR > 4) {
        i8mm_store12<zero_mode>(row[4] + col0, row[i8mm_odd_row<MR, 2>] + col0, mr > 5,
                                acc[2][0], acc[2][1], acc[2][2], acc[2][3], acc[2][4], acc[2][5],
                                vmin, vmax);
    }
    if constexpr (MR > 6) {
        i8mm_store12<zero_mode>(row[6] + col0, row[i8mm_odd_row<MR, 3>] + col0, mr > 7,
                                acc[3][0], acc[3][1], acc[3][2], acc[3][3], acc[3][4], acc[3][5],
                                vmin, vmax);
    }
}

/// The same for a panel whose live width is short of twelve: lane stores only.
template <int MR, bool zero_mode>
__attribute__((target("i8mm"), always_inline)) inline void
i8mm_store_all_tail(int32_t* const (&row)[MR], int col0, int rem, int mr,
                    const int32x4_t (&acc)[i8mm_tail_pairs<MR>][6], int32_t clamp_min,
                    int32_t clamp_max) noexcept
{
    i8mm_store12_tail<zero_mode>(row[0] + col0, row[i8mm_odd_row<MR, 0>] + col0,
                                 MR > 1 && mr > 1, rem, acc[0][0], acc[0][1], acc[0][2],
                                 acc[0][3], acc[0][4], acc[0][5], clamp_min, clamp_max);
    if constexpr (MR > 2) {
        i8mm_store12_tail<zero_mode>(row[2] + col0, row[i8mm_odd_row<MR, 1>] + col0, mr > 3, rem,
                                     acc[1][0], acc[1][1], acc[1][2], acc[1][3], acc[1][4],
                                     acc[1][5], clamp_min, clamp_max);
    }
    if constexpr (MR > 4) {
        i8mm_store12_tail<zero_mode>(row[4] + col0, row[i8mm_odd_row<MR, 2>] + col0, mr > 5, rem,
                                     acc[2][0], acc[2][1], acc[2][2], acc[2][3], acc[2][4],
                                     acc[2][5], clamp_min, clamp_max);
    }
    if constexpr (MR > 6) {
        i8mm_store12_tail<zero_mode>(row[6] + col0, row[i8mm_odd_row<MR, 3>] + col0, mr > 7, rem,
                                     acc[3][0], acc[3][1], acc[3][2], acc[3][3], acc[3][4],
                                     acc[3][5], clamp_min, clamp_max);
    }
}

/// Store one panel of accumulators, requantized.
template <int MR, class TOut>
__attribute__((target("i8mm"), always_inline)) inline void
i8mm_store_all_epi(TOut* const (&row)[MR], int col0,
                   const int32x4_t (&acc)[i8mm_tail_pairs<MR>][6], const float32x4_t (&sc)[3],
                   const int32x4_t (&bi)[3], int32x4_t vlo, int32x4_t vzp) noexcept
{
    i8mm_store12_epi<MR, TOut>(row[0] + col0, row[i8mm_odd_row<MR, 0>] + col0, acc[0][0],
                               acc[0][1], acc[0][2], acc[0][3], acc[0][4], acc[0][5], sc, bi,
                               vlo, vzp);
    if constexpr (MR > 2) {
        i8mm_store12_epi<MR, TOut>(row[2] + col0, row[i8mm_odd_row<MR, 1>] + col0, acc[1][0],
                                   acc[1][1], acc[1][2], acc[1][3], acc[1][4], acc[1][5], sc, bi,
                                   vlo, vzp);
    }
    if constexpr (MR > 4) {
        i8mm_store12_epi<MR, TOut>(row[4] + col0, row[i8mm_odd_row<MR, 2>] + col0, acc[2][0],
                                   acc[2][1], acc[2][2], acc[2][3], acc[2][4], acc[2][5], sc, bi,
                                   vlo, vzp);
    }
    if constexpr (MR > 6) {
        i8mm_store12_epi<MR, TOut>(row[6] + col0, row[i8mm_odd_row<MR, 3>] + col0, acc[3][0],
                                   acc[3][1], acc[3][2], acc[3][3], acc[3][4], acc[3][5], sc, bi,
                                   vlo, vzp);
    }
}

// =========================================================================
//  packed A + packed B  (MR = 8, 4, 1)
// =========================================================================

template <int MR, bool zero_mode = false>
__attribute__((target("i8mm"))) inline void
mma_pack_mr_nr12_i8mm(int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
                      const int8_t* NNOPS_RESTRICT A, const int8_t* NNOPS_RESTRICT B, int ldb,
                      int32_t clamp_min, int32_t clamp_max) noexcept
{
    int32_t* row[MR];
    i8mm_rows<MR>(C, ldc, mr, row);

    const int kbytes = (kc + 7) & ~7;
    const int kg = kbytes >> 3;
    const int ldb_panel = mma_i8mm_panel_stride(ldb, kbytes);
    const int32x4_t vmin = vdupq_n_s32(clamp_min);
    const int32x4_t vmax = vdupq_n_s32(clamp_max);

    int n = 0;
    for (; n + mma_i8mm_nr_max <= nc; n += mma_i8mm_nr_max) {
        int32x4_t acc[i8mm_tail_pairs<MR>][6];
        i8mm_mma<MR>(A, B, kg, acc);
        i8mm_store_all<MR, zero_mode>(row, n, mr, acc, vmin, vmax);
        B += ldb_panel;
    }
    i8mm_tail<zero_mode, MR>(row[0] + n, ldc, std::min(mr, MR), nc - n, kg, A, B, clamp_min,
                             clamp_max);
}

// =========================================================================
//  the requantizing variant (packed A + packed B)
// =========================================================================
//
// One MMA call covers the whole K, so the accumulators go straight to the
// destination — no int32 tile reaches memory and no row pass follows. `row_k`
// is s_a / s_out, `s_b` the per-column weight scale (or `s_b_scalar`), and
// `col_bias` the per-column int32 bias with the A-side shift already folded in.
// Relu is a floor of zero before the scale; nothing else is fused.

template <int MR, class TOut>
__attribute__((target("i8mm"))) inline void
mma_pack_mr_nr12_i8mm_epi(int mr, int nc, int kc, TOut* NNOPS_RESTRICT C, int64_t ldc,
                          const int8_t* NNOPS_RESTRICT A, const int8_t* NNOPS_RESTRICT B, int ldb,
                          const float* s_b, float s_b_scalar, float row_k,
                          const int32_t* col_bias, int32_t zp_out, bool relu) noexcept
{
    TOut* row[MR];
    i8mm_rows<MR>(C, ldc, mr, row);

    const int kbytes = (kc + 7) & ~7;
    const int kg = kbytes >> 3;
    const int ldb_panel = mma_i8mm_panel_stride(ldb, kbytes);
    const float32x4_t vk = vdupq_n_f32(row_k);
    const float32x4_t vreq = vdupq_n_f32(row_k * s_b_scalar);
    const int32x4_t vlo = vdupq_n_s32(relu ? 0 : INT32_MIN);
    const int32x4_t vzp = vdupq_n_s32(zp_out);

    int n = 0;
    for (; n + mma_i8mm_nr_max <= nc; n += mma_i8mm_nr_max) {
        int32x4_t acc[i8mm_tail_pairs<MR>][6];
        i8mm_mma<MR>(A, B, kg, acc);
        float32x4_t sc[3];
        int32x4_t bi[3];
        i8_col_q<3>(sc, bi, vk, vreq, s_b ? s_b + n : nullptr,
                    col_bias ? col_bias + n : nullptr);
        i8mm_store_all_epi<MR, TOut>(row, n, acc, sc, bi, vlo, vzp);
        B += ldb_panel;
    }
    i8mm_tail_epi<MR, TOut>(row[0] + n, ldc, std::min(mr, MR), nc - n, kg, A, B, vk, vreq,
                            s_b ? s_b + n : nullptr, col_bias ? col_bias + n : nullptr, vlo, vzp);
}

/// The requantizing rungs, consumed by matmul_helper.cpp; rows are {8,4,1}.
template <class TOut>
inline constexpr std::array<MmaI8EpiFn<TOut>, 3> mma_pack_epi_fns_i8mm = {
    &mma_pack_mr_nr12_i8mm_epi<8, TOut>,
    &mma_pack_mr_nr12_i8mm_epi<4, TOut>,
    &mma_pack_mr_nr12_i8mm_epi<1, TOut>,
};

// =========================================================================
//  row-major A + packed B  (MR = 8, 4, 1)
// =========================================================================
//
// The fragment is two 8-byte loads and a combine; a short final k group is
// staged through a zeroed 8-byte row per A row, since a row holds kc bytes.

template <int MR, bool zero_mode = false>
__attribute__((target("i8mm"))) inline void
mma_direct_mr_nr12_i8mm(int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
                        const int8_t* NNOPS_RESTRICT A, int lda, const int8_t* NNOPS_RESTRICT B,
                        int ldb, int32_t clamp_min, int32_t clamp_max) noexcept
{
    int32_t* row[MR];
    i8mm_rows<MR>(C, ldc, mr, row);
    const int8_t* ar[MR];
    i8mm_a_rows<MR>(A, lda, mr, ar);

    const int kbytes = (kc + 7) & ~7;
    const int kg = kbytes >> 3;
    const int ldb_panel = mma_i8mm_panel_stride(ldb, kbytes);
    const int32x4_t vmin = vdupq_n_s32(clamp_min);
    const int32x4_t vmax = vdupq_n_s32(clamp_max);

    for (int n = 0; n < nc; n += mma_i8mm_nr_max) {
        int32x4_t acc[i8mm_tail_pairs<MR>][6];
        i8mm_mma_rows<MR>(ar, B, kg, kc, acc);
        const int rem = nc - n;
        if (rem >= mma_i8mm_nr_max) {
            i8mm_store_all<MR, zero_mode>(row, n, mr, acc, vmin, vmax);
        } else {
            i8mm_store_all_tail<MR, zero_mode>(row, n, rem, mr, acc, clamp_min, clamp_max);
        }
        B += ldb_panel;
    }
}

// Dispatch tables, consumed by matmul_helper.cpp; rows are the MR levels {8,4,1}.
template <bool Z>
inline constexpr auto mma_pack_fns_i8mm =
    std::array{&mma_pack_mr_nr12_i8mm<8, Z>, &mma_pack_mr_nr12_i8mm<4, Z>,
               &mma_pack_mr_nr12_i8mm<1, Z>};

template <bool Z>
inline constexpr auto mma_direct_fns_i8mm = std::array{
    &mma_direct_mr_nr12_i8mm<8, Z>, &mma_direct_mr_nr12_i8mm<4, Z>,
    &mma_direct_mr_nr12_i8mm<1, Z>};

} // namespace nnops::backend::cpu::aarch64

#endif // NNOPS_ARCH_I8MM
