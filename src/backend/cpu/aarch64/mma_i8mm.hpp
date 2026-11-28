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
//  packed A + packed B  (mr = 8, 4, 1)
// =========================================================================

template <bool zero_mode = false>
__attribute__((target("i8mm"))) inline void
mma_pack_mr8_nr12_i8mm(int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
                       const int8_t* NNOPS_RESTRICT A, const int8_t* NNOPS_RESTRICT B, int ldb,
                       int32_t clamp_min, int32_t clamp_max) noexcept
{
    int32_t* NNOPS_RESTRICT c0_ptr = C + 0 * ldc;
    int32_t* NNOPS_RESTRICT c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    int32_t* NNOPS_RESTRICT c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    int32_t* NNOPS_RESTRICT c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;
    int32_t* NNOPS_RESTRICT c4_ptr = (mr > 4) ? C + 4 * ldc : c3_ptr;
    int32_t* NNOPS_RESTRICT c5_ptr = (mr > 5) ? C + 5 * ldc : c4_ptr;
    int32_t* NNOPS_RESTRICT c6_ptr = (mr > 6) ? C + 6 * ldc : c5_ptr;
    int32_t* NNOPS_RESTRICT c7_ptr = (mr > 7) ? C + 7 * ldc : c6_ptr;

    const int kbytes = (kc + 7) & ~7;
    const int kg = kbytes >> 3;
    const int ldb_panel = mma_i8mm_panel_stride(ldb, kbytes);
    const int32x4_t vmin = vdupq_n_s32(clamp_min);
    const int32x4_t vmax = vdupq_n_s32(clamp_max);

    int n = 0;
    for (; n + mma_i8mm_nr_max <= nc; n += mma_i8mm_nr_max) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;
        int32x4_t v_c03 = v_c00, v_c04 = v_c00, v_c05 = v_c00;
        int32x4_t v_c10 = vdupq_n_s32(0), v_c11 = v_c10, v_c12 = v_c10;
        int32x4_t v_c13 = v_c10, v_c14 = v_c10, v_c15 = v_c10;
        int32x4_t v_c20 = vdupq_n_s32(0), v_c21 = v_c20, v_c22 = v_c20;
        int32x4_t v_c23 = v_c20, v_c24 = v_c20, v_c25 = v_c20;
        int32x4_t v_c30 = vdupq_n_s32(0), v_c31 = v_c30, v_c32 = v_c30;
        int32x4_t v_c33 = v_c30, v_c34 = v_c30, v_c35 = v_c30;

        const int8_t* NNOPS_RESTRICT a_ptr = A;
        const int8_t* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kg; ++k) {
            const int8x16_t v_a0 = vld1q_s8(a_ptr + 0 * 16); // rows 0,1
            const int8x16_t v_a1 = vld1q_s8(a_ptr + 1 * 16); // rows 2,3
            const int8x16_t v_a2 = vld1q_s8(a_ptr + 2 * 16); // rows 4,5
            const int8x16_t v_a3 = vld1q_s8(a_ptr + 3 * 16); // rows 6,7
            a_ptr += 4 * 16;

            const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16); // columns 0,1
            v_c00 = vmmlaq_s32(v_c00, v_a0, v_b0);
            v_c10 = vmmlaq_s32(v_c10, v_a1, v_b0);
            v_c20 = vmmlaq_s32(v_c20, v_a2, v_b0);
            v_c30 = vmmlaq_s32(v_c30, v_a3, v_b0);

            const int8x16_t v_b1 = vld1q_s8(b_ptr + 1 * 16); // columns 2,3
            v_c01 = vmmlaq_s32(v_c01, v_a0, v_b1);
            v_c11 = vmmlaq_s32(v_c11, v_a1, v_b1);
            v_c21 = vmmlaq_s32(v_c21, v_a2, v_b1);
            v_c31 = vmmlaq_s32(v_c31, v_a3, v_b1);

            const int8x16_t v_b2 = vld1q_s8(b_ptr + 2 * 16); // columns 4,5
            v_c02 = vmmlaq_s32(v_c02, v_a0, v_b2);
            v_c12 = vmmlaq_s32(v_c12, v_a1, v_b2);
            v_c22 = vmmlaq_s32(v_c22, v_a2, v_b2);
            v_c32 = vmmlaq_s32(v_c32, v_a3, v_b2);

            const int8x16_t v_b3 = vld1q_s8(b_ptr + 3 * 16); // columns 6,7
            v_c03 = vmmlaq_s32(v_c03, v_a0, v_b3);
            v_c13 = vmmlaq_s32(v_c13, v_a1, v_b3);
            v_c23 = vmmlaq_s32(v_c23, v_a2, v_b3);
            v_c33 = vmmlaq_s32(v_c33, v_a3, v_b3);

            const int8x16_t v_b4 = vld1q_s8(b_ptr + 4 * 16); // columns 8,9
            v_c04 = vmmlaq_s32(v_c04, v_a0, v_b4);
            v_c14 = vmmlaq_s32(v_c14, v_a1, v_b4);
            v_c24 = vmmlaq_s32(v_c24, v_a2, v_b4);
            v_c34 = vmmlaq_s32(v_c34, v_a3, v_b4);

            const int8x16_t v_b5 = vld1q_s8(b_ptr + 5 * 16); // columns 10,11
            v_c05 = vmmlaq_s32(v_c05, v_a0, v_b5);
            v_c15 = vmmlaq_s32(v_c15, v_a1, v_b5);
            v_c25 = vmmlaq_s32(v_c25, v_a2, v_b5);
            v_c35 = vmmlaq_s32(v_c35, v_a3, v_b5);

            b_ptr += mma_i8mm_group_bytes;
        }

        i8mm_store12<zero_mode>(c0_ptr, c1_ptr, mr > 1, v_c00, v_c01, v_c02, v_c03, v_c04, v_c05,
                               vmin, vmax);
        i8mm_store12<zero_mode>(c2_ptr, c3_ptr, mr > 3, v_c10, v_c11, v_c12, v_c13, v_c14, v_c15,
                               vmin, vmax);
        i8mm_store12<zero_mode>(c4_ptr, c5_ptr, mr > 5, v_c20, v_c21, v_c22, v_c23, v_c24, v_c25,
                               vmin, vmax);
        i8mm_store12<zero_mode>(c6_ptr, c7_ptr, mr > 7, v_c30, v_c31, v_c32, v_c33, v_c34, v_c35,
                               vmin, vmax);

        c0_ptr += 12;
        c1_ptr += 12;
        c2_ptr += 12;
        c3_ptr += 12;
        c4_ptr += 12;
        c5_ptr += 12;
        c6_ptr += 12;
        c7_ptr += 12;
        B += ldb_panel;
    }
    i8mm_tail<zero_mode, 8>(c0_ptr, ldc, mr, nc - n, kg, A, B, clamp_min, clamp_max);
}

template <bool zero_mode = false>
__attribute__((target("i8mm"))) inline void
mma_pack_mr4_nr12_i8mm(int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
                       const int8_t* NNOPS_RESTRICT A, const int8_t* NNOPS_RESTRICT B, int ldb,
                       int32_t clamp_min, int32_t clamp_max) noexcept
{
    int32_t* NNOPS_RESTRICT c0_ptr = C + 0 * ldc;
    int32_t* NNOPS_RESTRICT c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    int32_t* NNOPS_RESTRICT c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    int32_t* NNOPS_RESTRICT c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;

    const int kbytes = (kc + 7) & ~7;
    const int kg = kbytes >> 3;
    const int ldb_panel = mma_i8mm_panel_stride(ldb, kbytes);
    const int32x4_t vmin = vdupq_n_s32(clamp_min);
    const int32x4_t vmax = vdupq_n_s32(clamp_max);

    int n = 0;
    for (; n + mma_i8mm_nr_max <= nc; n += mma_i8mm_nr_max) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;
        int32x4_t v_c03 = v_c00, v_c04 = v_c00, v_c05 = v_c00;
        int32x4_t v_c10 = vdupq_n_s32(0), v_c11 = v_c10, v_c12 = v_c10;
        int32x4_t v_c13 = v_c10, v_c14 = v_c10, v_c15 = v_c10;

        const int8_t* NNOPS_RESTRICT a_ptr = A;
        const int8_t* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kg; ++k) {
            const int8x16_t v_a0 = vld1q_s8(a_ptr + 0 * 16); // rows 0,1
            const int8x16_t v_a1 = vld1q_s8(a_ptr + 1 * 16); // rows 2,3
            a_ptr += 2 * 16;

            const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);
            v_c00 = vmmlaq_s32(v_c00, v_a0, v_b0);
            v_c10 = vmmlaq_s32(v_c10, v_a1, v_b0);

            const int8x16_t v_b1 = vld1q_s8(b_ptr + 1 * 16);
            v_c01 = vmmlaq_s32(v_c01, v_a0, v_b1);
            v_c11 = vmmlaq_s32(v_c11, v_a1, v_b1);

            const int8x16_t v_b2 = vld1q_s8(b_ptr + 2 * 16);
            v_c02 = vmmlaq_s32(v_c02, v_a0, v_b2);
            v_c12 = vmmlaq_s32(v_c12, v_a1, v_b2);

            const int8x16_t v_b3 = vld1q_s8(b_ptr + 3 * 16);
            v_c03 = vmmlaq_s32(v_c03, v_a0, v_b3);
            v_c13 = vmmlaq_s32(v_c13, v_a1, v_b3);

            const int8x16_t v_b4 = vld1q_s8(b_ptr + 4 * 16);
            v_c04 = vmmlaq_s32(v_c04, v_a0, v_b4);
            v_c14 = vmmlaq_s32(v_c14, v_a1, v_b4);

            const int8x16_t v_b5 = vld1q_s8(b_ptr + 5 * 16);
            v_c05 = vmmlaq_s32(v_c05, v_a0, v_b5);
            v_c15 = vmmlaq_s32(v_c15, v_a1, v_b5);

            b_ptr += mma_i8mm_group_bytes;
        }

        i8mm_store12<zero_mode>(c0_ptr, c1_ptr, mr > 1, v_c00, v_c01, v_c02, v_c03, v_c04, v_c05,
                               vmin, vmax);
        i8mm_store12<zero_mode>(c2_ptr, c3_ptr, mr > 3, v_c10, v_c11, v_c12, v_c13, v_c14, v_c15,
                               vmin, vmax);

        c0_ptr += 12;
        c1_ptr += 12;
        c2_ptr += 12;
        c3_ptr += 12;
        B += ldb_panel;
    }
    i8mm_tail<zero_mode, 4>(c0_ptr, ldc, mr, nc - n, kg, A, B, clamp_min, clamp_max);
}

template <bool zero_mode = false>
__attribute__((target("i8mm"))) inline void
mma_pack_mr1_nr12_i8mm(int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
                       const int8_t* NNOPS_RESTRICT A, const int8_t* NNOPS_RESTRICT B, int ldb,
                       int32_t clamp_min, int32_t clamp_max) noexcept
{
    (void)mr;  // one row, so the row count never varies
    (void)ldc; // one row, so the C row stride never applies

    int32_t* NNOPS_RESTRICT c0_ptr = C;

    const int kbytes = (kc + 7) & ~7;
    const int kg = kbytes >> 3;
    const int ldb_panel = mma_i8mm_panel_stride(ldb, kbytes);
    const int32x4_t vmin = vdupq_n_s32(clamp_min);
    const int32x4_t vmax = vdupq_n_s32(clamp_max);
    const int8x8_t vz = vdup_n_s8(0);

    int n = 0;
    for (; n + mma_i8mm_nr_max <= nc; n += mma_i8mm_nr_max) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;
        int32x4_t v_c03 = v_c00, v_c04 = v_c00, v_c05 = v_c00;

        const int8_t* NNOPS_RESTRICT a_ptr = A;
        const int8_t* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kg; ++k) {
            // The second row of the fragment is zero: the odd lanes of every
            // accumulator are dead and only the low 64 bits are stored.
            const int8x16_t v_a = vcombine_s8(vld1_s8(a_ptr), vz);
            a_ptr += 8;

            v_c00 = vmmlaq_s32(v_c00, v_a, vld1q_s8(b_ptr + 0 * 16));
            v_c01 = vmmlaq_s32(v_c01, v_a, vld1q_s8(b_ptr + 1 * 16));
            v_c02 = vmmlaq_s32(v_c02, v_a, vld1q_s8(b_ptr + 2 * 16));
            v_c03 = vmmlaq_s32(v_c03, v_a, vld1q_s8(b_ptr + 3 * 16));
            v_c04 = vmmlaq_s32(v_c04, v_a, vld1q_s8(b_ptr + 4 * 16));
            v_c05 = vmmlaq_s32(v_c05, v_a, vld1q_s8(b_ptr + 5 * 16));

            b_ptr += mma_i8mm_group_bytes;
        }

        int32x4_t v = vcombine_s32(vget_low_s32(v_c00), vget_low_s32(v_c01));
        if constexpr (!zero_mode) {
            v = vaddq_s32(v, vld1q_s32(c0_ptr + 0));
        }
        vst1q_s32(c0_ptr + 0, vminq_s32(vmaxq_s32(v, vmin), vmax));
        v = vcombine_s32(vget_low_s32(v_c02), vget_low_s32(v_c03));
        if constexpr (!zero_mode) {
            v = vaddq_s32(v, vld1q_s32(c0_ptr + 4));
        }
        vst1q_s32(c0_ptr + 4, vminq_s32(vmaxq_s32(v, vmin), vmax));
        v = vcombine_s32(vget_low_s32(v_c04), vget_low_s32(v_c05));
        if constexpr (!zero_mode) {
            v = vaddq_s32(v, vld1q_s32(c0_ptr + 8));
        }
        vst1q_s32(c0_ptr + 8, vminq_s32(vmaxq_s32(v, vmin), vmax));

        c0_ptr += 12;
        B += ldb_panel;
    }
    i8mm_tail<zero_mode, 1>(c0_ptr, 0, 1, nc - n, kg, A, B, clamp_min, clamp_max);
}

// =========================================================================
//  the requantizing variants (packed A + packed B)
// =========================================================================
//
// One MMA call covers the whole K, so the accumulators go straight to the
// destination — no int32 tile reaches memory and no row pass follows. `row_k`
// is s_a / s_out, `s_b` the per-column weight scale (or `s_b_scalar`), and
// `col_bias` the per-column int32 bias with the A-side shift already folded in.
// Relu is a floor of zero before the scale; nothing else is fused.

template <class TOut>
__attribute__((target("i8mm"))) inline void
mma_pack_mr8_nr12_i8mm_epi(int mr, int nc, int kc, TOut* NNOPS_RESTRICT C, int64_t ldc,
                           const int8_t* NNOPS_RESTRICT A, const int8_t* NNOPS_RESTRICT B,
                           int ldb, const float* s_b, float s_b_scalar, float row_k,
                           const int32_t* col_bias, int32_t zp_out, bool relu) noexcept
{
    TOut* NNOPS_RESTRICT c0_ptr = C + 0 * ldc;
    TOut* NNOPS_RESTRICT c1_ptr = C + 1 * ldc;
    TOut* NNOPS_RESTRICT c2_ptr = C + 2 * ldc;
    TOut* NNOPS_RESTRICT c3_ptr = C + 3 * ldc;
    TOut* NNOPS_RESTRICT c4_ptr = C + 4 * ldc;
    TOut* NNOPS_RESTRICT c5_ptr = C + 5 * ldc;
    TOut* NNOPS_RESTRICT c6_ptr = C + 6 * ldc;
    TOut* NNOPS_RESTRICT c7_ptr = C + 7 * ldc;

    const int kbytes = (kc + 7) & ~7;
    const int kg = kbytes >> 3;
    const int ldb_panel = mma_i8mm_panel_stride(ldb, kbytes);
    const float32x4_t vk = vdupq_n_f32(row_k);
    const float32x4_t vreq = vdupq_n_f32(row_k * s_b_scalar);
    const int32x4_t vlo = vdupq_n_s32(relu ? 0 : INT32_MIN);
    const int32x4_t vzp = vdupq_n_s32(zp_out);

    int n = 0;
    for (; n + mma_i8mm_nr_max <= nc; n += mma_i8mm_nr_max) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;
        int32x4_t v_c03 = v_c00, v_c04 = v_c00, v_c05 = v_c00;
        int32x4_t v_c10 = vdupq_n_s32(0), v_c11 = v_c10, v_c12 = v_c10;
        int32x4_t v_c13 = v_c10, v_c14 = v_c10, v_c15 = v_c10;
        int32x4_t v_c20 = vdupq_n_s32(0), v_c21 = v_c20, v_c22 = v_c20;
        int32x4_t v_c23 = v_c20, v_c24 = v_c20, v_c25 = v_c20;
        int32x4_t v_c30 = vdupq_n_s32(0), v_c31 = v_c30, v_c32 = v_c30;
        int32x4_t v_c33 = v_c30, v_c34 = v_c30, v_c35 = v_c30;

        const int8_t* NNOPS_RESTRICT a_ptr = A;
        const int8_t* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kg; ++k) {
            const int8x16_t v_a0 = vld1q_s8(a_ptr + 0 * 16); // rows 0,1
            const int8x16_t v_a1 = vld1q_s8(a_ptr + 1 * 16); // rows 2,3
            const int8x16_t v_a2 = vld1q_s8(a_ptr + 2 * 16); // rows 4,5
            const int8x16_t v_a3 = vld1q_s8(a_ptr + 3 * 16); // rows 6,7
            a_ptr += 4 * 16;

            const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16); // columns 0,1
            v_c00 = vmmlaq_s32(v_c00, v_a0, v_b0);
            v_c10 = vmmlaq_s32(v_c10, v_a1, v_b0);
            v_c20 = vmmlaq_s32(v_c20, v_a2, v_b0);
            v_c30 = vmmlaq_s32(v_c30, v_a3, v_b0);

            const int8x16_t v_b1 = vld1q_s8(b_ptr + 1 * 16); // columns 2,3
            v_c01 = vmmlaq_s32(v_c01, v_a0, v_b1);
            v_c11 = vmmlaq_s32(v_c11, v_a1, v_b1);
            v_c21 = vmmlaq_s32(v_c21, v_a2, v_b1);
            v_c31 = vmmlaq_s32(v_c31, v_a3, v_b1);

            const int8x16_t v_b2 = vld1q_s8(b_ptr + 2 * 16); // columns 4,5
            v_c02 = vmmlaq_s32(v_c02, v_a0, v_b2);
            v_c12 = vmmlaq_s32(v_c12, v_a1, v_b2);
            v_c22 = vmmlaq_s32(v_c22, v_a2, v_b2);
            v_c32 = vmmlaq_s32(v_c32, v_a3, v_b2);

            const int8x16_t v_b3 = vld1q_s8(b_ptr + 3 * 16); // columns 6,7
            v_c03 = vmmlaq_s32(v_c03, v_a0, v_b3);
            v_c13 = vmmlaq_s32(v_c13, v_a1, v_b3);
            v_c23 = vmmlaq_s32(v_c23, v_a2, v_b3);
            v_c33 = vmmlaq_s32(v_c33, v_a3, v_b3);

            const int8x16_t v_b4 = vld1q_s8(b_ptr + 4 * 16); // columns 8,9
            v_c04 = vmmlaq_s32(v_c04, v_a0, v_b4);
            v_c14 = vmmlaq_s32(v_c14, v_a1, v_b4);
            v_c24 = vmmlaq_s32(v_c24, v_a2, v_b4);
            v_c34 = vmmlaq_s32(v_c34, v_a3, v_b4);

            const int8x16_t v_b5 = vld1q_s8(b_ptr + 5 * 16); // columns 10,11
            v_c05 = vmmlaq_s32(v_c05, v_a0, v_b5);
            v_c15 = vmmlaq_s32(v_c15, v_a1, v_b5);
            v_c25 = vmmlaq_s32(v_c25, v_a2, v_b5);
            v_c35 = vmmlaq_s32(v_c35, v_a3, v_b5);

            b_ptr += mma_i8mm_group_bytes;
        }

        float32x4_t sc[3];
        int32x4_t bi[3];
        i8_col_q<3>(sc, bi, vk, vreq, s_b ? s_b + n : nullptr,
                      col_bias ? col_bias + n : nullptr);
        i8mm_store12_epi<8, TOut>(c0_ptr, c1_ptr, v_c00, v_c01, v_c02, v_c03, v_c04, v_c05, sc,
                                  bi, vlo, vzp);
        i8mm_store12_epi<8, TOut>(c2_ptr, c3_ptr, v_c10, v_c11, v_c12, v_c13, v_c14, v_c15, sc,
                                  bi, vlo, vzp);
        i8mm_store12_epi<8, TOut>(c4_ptr, c5_ptr, v_c20, v_c21, v_c22, v_c23, v_c24, v_c25, sc,
                                  bi, vlo, vzp);
        i8mm_store12_epi<8, TOut>(c6_ptr, c7_ptr, v_c30, v_c31, v_c32, v_c33, v_c34, v_c35, sc,
                                  bi, vlo, vzp);

        c0_ptr += 12;
        c1_ptr += 12;
        c2_ptr += 12;
        c3_ptr += 12;
        c4_ptr += 12;
        c5_ptr += 12;
        c6_ptr += 12;
        c7_ptr += 12;
        B += ldb_panel;
    }
    i8mm_tail_epi<8, TOut>(c0_ptr, ldc, mr, nc - n, kg, A, B, vk, vreq,
                           s_b ? s_b + n : nullptr, col_bias ? col_bias + n : nullptr, vlo, vzp);
}

template <class TOut>
__attribute__((target("i8mm"))) inline void
mma_pack_mr4_nr12_i8mm_epi(int mr, int nc, int kc, TOut* NNOPS_RESTRICT C, int64_t ldc,
                           const int8_t* NNOPS_RESTRICT A, const int8_t* NNOPS_RESTRICT B,
                           int ldb, const float* s_b, float s_b_scalar, float row_k,
                           const int32_t* col_bias, int32_t zp_out, bool relu) noexcept
{
    TOut* NNOPS_RESTRICT c0_ptr = C + 0 * ldc;
    TOut* NNOPS_RESTRICT c1_ptr = C + 1 * ldc;
    TOut* NNOPS_RESTRICT c2_ptr = C + 2 * ldc;
    TOut* NNOPS_RESTRICT c3_ptr = C + 3 * ldc;

    const int kbytes = (kc + 7) & ~7;
    const int kg = kbytes >> 3;
    const int ldb_panel = mma_i8mm_panel_stride(ldb, kbytes);
    const float32x4_t vk = vdupq_n_f32(row_k);
    const float32x4_t vreq = vdupq_n_f32(row_k * s_b_scalar);
    const int32x4_t vlo = vdupq_n_s32(relu ? 0 : INT32_MIN);
    const int32x4_t vzp = vdupq_n_s32(zp_out);

    int n = 0;
    for (; n + mma_i8mm_nr_max <= nc; n += mma_i8mm_nr_max) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;
        int32x4_t v_c03 = v_c00, v_c04 = v_c00, v_c05 = v_c00;
        int32x4_t v_c10 = vdupq_n_s32(0), v_c11 = v_c10, v_c12 = v_c10;
        int32x4_t v_c13 = v_c10, v_c14 = v_c10, v_c15 = v_c10;

        const int8_t* NNOPS_RESTRICT a_ptr = A;
        const int8_t* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kg; ++k) {
            const int8x16_t v_a0 = vld1q_s8(a_ptr + 0 * 16); // rows 0,1
            const int8x16_t v_a1 = vld1q_s8(a_ptr + 1 * 16); // rows 2,3
            a_ptr += 2 * 16;

            const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);
            v_c00 = vmmlaq_s32(v_c00, v_a0, v_b0);
            v_c10 = vmmlaq_s32(v_c10, v_a1, v_b0);

            const int8x16_t v_b1 = vld1q_s8(b_ptr + 1 * 16);
            v_c01 = vmmlaq_s32(v_c01, v_a0, v_b1);
            v_c11 = vmmlaq_s32(v_c11, v_a1, v_b1);

            const int8x16_t v_b2 = vld1q_s8(b_ptr + 2 * 16);
            v_c02 = vmmlaq_s32(v_c02, v_a0, v_b2);
            v_c12 = vmmlaq_s32(v_c12, v_a1, v_b2);

            const int8x16_t v_b3 = vld1q_s8(b_ptr + 3 * 16);
            v_c03 = vmmlaq_s32(v_c03, v_a0, v_b3);
            v_c13 = vmmlaq_s32(v_c13, v_a1, v_b3);

            const int8x16_t v_b4 = vld1q_s8(b_ptr + 4 * 16);
            v_c04 = vmmlaq_s32(v_c04, v_a0, v_b4);
            v_c14 = vmmlaq_s32(v_c14, v_a1, v_b4);

            const int8x16_t v_b5 = vld1q_s8(b_ptr + 5 * 16);
            v_c05 = vmmlaq_s32(v_c05, v_a0, v_b5);
            v_c15 = vmmlaq_s32(v_c15, v_a1, v_b5);

            b_ptr += mma_i8mm_group_bytes;
        }

        float32x4_t sc[3];
        int32x4_t bi[3];
        i8_col_q<3>(sc, bi, vk, vreq, s_b ? s_b + n : nullptr,
                      col_bias ? col_bias + n : nullptr);
        i8mm_store12_epi<4, TOut>(c0_ptr, c1_ptr, v_c00, v_c01, v_c02, v_c03, v_c04, v_c05, sc,
                                  bi, vlo, vzp);
        i8mm_store12_epi<4, TOut>(c2_ptr, c3_ptr, v_c10, v_c11, v_c12, v_c13, v_c14, v_c15, sc,
                                  bi, vlo, vzp);

        c0_ptr += 12;
        c1_ptr += 12;
        c2_ptr += 12;
        c3_ptr += 12;
        B += ldb_panel;
    }
    i8mm_tail_epi<4, TOut>(c0_ptr, ldc, mr, nc - n, kg, A, B, vk, vreq,
                           s_b ? s_b + n : nullptr, col_bias ? col_bias + n : nullptr, vlo, vzp);
}

template <class TOut>
__attribute__((target("i8mm"))) inline void
mma_pack_mr1_nr12_i8mm_epi(int mr, int nc, int kc, TOut* NNOPS_RESTRICT C, int64_t ldc,
                           const int8_t* NNOPS_RESTRICT A, const int8_t* NNOPS_RESTRICT B,
                           int ldb, const float* s_b, float s_b_scalar, float row_k,
                           const int32_t* col_bias, int32_t zp_out, bool relu) noexcept
{
    (void)mr;  // one row, so the row count never varies
    (void)ldc; // one row, so the output pitch never applies
    TOut* NNOPS_RESTRICT c0_ptr = C;

    const int kbytes = (kc + 7) & ~7;
    const int kg = kbytes >> 3;
    const int ldb_panel = mma_i8mm_panel_stride(ldb, kbytes);
    const float32x4_t vk = vdupq_n_f32(row_k);
    const float32x4_t vreq = vdupq_n_f32(row_k * s_b_scalar);
    const int32x4_t vlo = vdupq_n_s32(relu ? 0 : INT32_MIN);
    const int32x4_t vzp = vdupq_n_s32(zp_out);
    const int8x8_t vz = vdup_n_s8(0);

    int n = 0;
    for (; n + mma_i8mm_nr_max <= nc; n += mma_i8mm_nr_max) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;
        int32x4_t v_c03 = v_c00, v_c04 = v_c00, v_c05 = v_c00;

        const int8_t* NNOPS_RESTRICT a_ptr = A;
        const int8_t* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kg; ++k) {
            const int8x16_t v_a = vcombine_s8(vld1_s8(a_ptr), vz);
            a_ptr += 8;

            v_c00 = vmmlaq_s32(v_c00, v_a, vld1q_s8(b_ptr + 0 * 16));
            v_c01 = vmmlaq_s32(v_c01, v_a, vld1q_s8(b_ptr + 1 * 16));
            v_c02 = vmmlaq_s32(v_c02, v_a, vld1q_s8(b_ptr + 2 * 16));
            v_c03 = vmmlaq_s32(v_c03, v_a, vld1q_s8(b_ptr + 3 * 16));
            v_c04 = vmmlaq_s32(v_c04, v_a, vld1q_s8(b_ptr + 4 * 16));
            v_c05 = vmmlaq_s32(v_c05, v_a, vld1q_s8(b_ptr + 5 * 16));

            b_ptr += mma_i8mm_group_bytes;
        }

        float32x4_t sc[3];
        int32x4_t bi[3];
        i8_col_q<3>(sc, bi, vk, vreq, s_b ? s_b + n : nullptr,
                      col_bias ? col_bias + n : nullptr);
        i8mm_store12_epi<1, TOut>(c0_ptr, c0_ptr, v_c00, v_c01, v_c02, v_c03, v_c04, v_c05, sc,
                                  bi, vlo, vzp);
        c0_ptr += 12;
        B += ldb_panel;
    }
    i8mm_tail_epi<1, TOut>(c0_ptr, 1, 1, nc - n, kg, A, B, vk, vreq,
                           s_b ? s_b + n : nullptr, col_bias ? col_bias + n : nullptr, vlo, vzp);
}

/// Dispatch tables for the requantizing kernels, rows are the MR levels {8,4,1}.
template <class TOut>
inline constexpr std::array<MmaI8EpiFn<TOut>, 3> mma_pack_epi_fns_i8mm = {
    &mma_pack_mr8_nr12_i8mm_epi<TOut>,
    &mma_pack_mr4_nr12_i8mm_epi<TOut>,
    &mma_pack_mr1_nr12_i8mm_epi<TOut>,
};

// =========================================================================
//  row-major A + packed B  (mr = 8, 4, 1)
// =========================================================================
//
// The fragment is two 8-byte loads and a combine; a short final k group is
// staged through a zeroed 8-byte row per A row, since a row holds kc bytes.

template <bool zero_mode = false>
__attribute__((target("i8mm"))) inline void
mma_direct_mr8_nr12_i8mm(int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
                         const int8_t* NNOPS_RESTRICT A, int lda, const int8_t* NNOPS_RESTRICT B,
                         int ldb, int32_t clamp_min, int32_t clamp_max) noexcept
{
    int32_t* NNOPS_RESTRICT c0_ptr = C + 0 * ldc;
    int32_t* NNOPS_RESTRICT c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    int32_t* NNOPS_RESTRICT c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    int32_t* NNOPS_RESTRICT c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;
    int32_t* NNOPS_RESTRICT c4_ptr = (mr > 4) ? C + 4 * ldc : c3_ptr;
    int32_t* NNOPS_RESTRICT c5_ptr = (mr > 5) ? C + 5 * ldc : c4_ptr;
    int32_t* NNOPS_RESTRICT c6_ptr = (mr > 6) ? C + 6 * ldc : c5_ptr;
    int32_t* NNOPS_RESTRICT c7_ptr = (mr > 7) ? C + 7 * ldc : c6_ptr;

    // Rows past the live count read the last live row's bytes.
    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;
    const int r4 = (mr > 4) ? 4 : r3;
    const int r5 = (mr > 5) ? 5 : r4;
    const int r6 = (mr > 6) ? 6 : r5;
    const int r7 = (mr > 7) ? 7 : r6;

    const int kbytes = (kc + 7) & ~7;
    const int kg = kbytes >> 3;
    const int ldb_panel = mma_i8mm_panel_stride(ldb, kbytes);
    const int32x4_t vmin = vdupq_n_s32(clamp_min);
    const int32x4_t vmax = vdupq_n_s32(clamp_max);

    for (int n = 0; n < nc; n += mma_i8mm_nr_max) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;
        int32x4_t v_c03 = v_c00, v_c04 = v_c00, v_c05 = v_c00;
        int32x4_t v_c10 = vdupq_n_s32(0), v_c11 = v_c10, v_c12 = v_c10;
        int32x4_t v_c13 = v_c10, v_c14 = v_c10, v_c15 = v_c10;
        int32x4_t v_c20 = vdupq_n_s32(0), v_c21 = v_c20, v_c22 = v_c20;
        int32x4_t v_c23 = v_c20, v_c24 = v_c20, v_c25 = v_c20;
        int32x4_t v_c30 = vdupq_n_s32(0), v_c31 = v_c30, v_c32 = v_c30;
        int32x4_t v_c33 = v_c30, v_c34 = v_c30, v_c35 = v_c30;

        const int8_t* NNOPS_RESTRICT a0 = A + 0 * lda;
        const int8_t* NNOPS_RESTRICT a1 = A + r1 * lda;
        const int8_t* NNOPS_RESTRICT a2 = A + r2 * lda;
        const int8_t* NNOPS_RESTRICT a3 = A + r3 * lda;
        const int8_t* NNOPS_RESTRICT a4 = A + r4 * lda;
        const int8_t* NNOPS_RESTRICT a5 = A + r5 * lda;
        const int8_t* NNOPS_RESTRICT a6 = A + r6 * lda;
        const int8_t* NNOPS_RESTRICT a7 = A + r7 * lda;
        const int8_t* NNOPS_RESTRICT b_ptr = B;

        for (int k = 0; k < kg; ++k) {
            int8x16_t v_a0, v_a1, v_a2, v_a3;
            if (k + 1 < kg || (kc & 7) == 0) {
                v_a0 = vcombine_s8(vld1_s8(a0), vld1_s8(a1));
                v_a1 = vcombine_s8(vld1_s8(a2), vld1_s8(a3));
                v_a2 = vcombine_s8(vld1_s8(a4), vld1_s8(a5));
                v_a3 = vcombine_s8(vld1_s8(a6), vld1_s8(a7));
                a0 += 8;
                a1 += 8;
                a2 += 8;
                a3 += 8;
                a4 += 8;
                a5 += 8;
                a6 += 8;
                a7 += 8;
            } else {
                const int live = kc - 8 * k;
                int8_t t[8][8] = {};
                std::memcpy(t[0], a0, static_cast<size_t>(live));
                std::memcpy(t[1], a1, static_cast<size_t>(live));
                std::memcpy(t[2], a2, static_cast<size_t>(live));
                std::memcpy(t[3], a3, static_cast<size_t>(live));
                std::memcpy(t[4], a4, static_cast<size_t>(live));
                std::memcpy(t[5], a5, static_cast<size_t>(live));
                std::memcpy(t[6], a6, static_cast<size_t>(live));
                std::memcpy(t[7], a7, static_cast<size_t>(live));
                v_a0 = vcombine_s8(vld1_s8(t[0]), vld1_s8(t[1]));
                v_a1 = vcombine_s8(vld1_s8(t[2]), vld1_s8(t[3]));
                v_a2 = vcombine_s8(vld1_s8(t[4]), vld1_s8(t[5]));
                v_a3 = vcombine_s8(vld1_s8(t[6]), vld1_s8(t[7]));
            }

            const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);
            v_c00 = vmmlaq_s32(v_c00, v_a0, v_b0);
            v_c10 = vmmlaq_s32(v_c10, v_a1, v_b0);
            v_c20 = vmmlaq_s32(v_c20, v_a2, v_b0);
            v_c30 = vmmlaq_s32(v_c30, v_a3, v_b0);

            const int8x16_t v_b1 = vld1q_s8(b_ptr + 1 * 16);
            v_c01 = vmmlaq_s32(v_c01, v_a0, v_b1);
            v_c11 = vmmlaq_s32(v_c11, v_a1, v_b1);
            v_c21 = vmmlaq_s32(v_c21, v_a2, v_b1);
            v_c31 = vmmlaq_s32(v_c31, v_a3, v_b1);

            const int8x16_t v_b2 = vld1q_s8(b_ptr + 2 * 16);
            v_c02 = vmmlaq_s32(v_c02, v_a0, v_b2);
            v_c12 = vmmlaq_s32(v_c12, v_a1, v_b2);
            v_c22 = vmmlaq_s32(v_c22, v_a2, v_b2);
            v_c32 = vmmlaq_s32(v_c32, v_a3, v_b2);

            const int8x16_t v_b3 = vld1q_s8(b_ptr + 3 * 16);
            v_c03 = vmmlaq_s32(v_c03, v_a0, v_b3);
            v_c13 = vmmlaq_s32(v_c13, v_a1, v_b3);
            v_c23 = vmmlaq_s32(v_c23, v_a2, v_b3);
            v_c33 = vmmlaq_s32(v_c33, v_a3, v_b3);

            const int8x16_t v_b4 = vld1q_s8(b_ptr + 4 * 16);
            v_c04 = vmmlaq_s32(v_c04, v_a0, v_b4);
            v_c14 = vmmlaq_s32(v_c14, v_a1, v_b4);
            v_c24 = vmmlaq_s32(v_c24, v_a2, v_b4);
            v_c34 = vmmlaq_s32(v_c34, v_a3, v_b4);

            const int8x16_t v_b5 = vld1q_s8(b_ptr + 5 * 16);
            v_c05 = vmmlaq_s32(v_c05, v_a0, v_b5);
            v_c15 = vmmlaq_s32(v_c15, v_a1, v_b5);
            v_c25 = vmmlaq_s32(v_c25, v_a2, v_b5);
            v_c35 = vmmlaq_s32(v_c35, v_a3, v_b5);

            b_ptr += mma_i8mm_group_bytes;
        }

        if (n + mma_i8mm_nr_max <= nc) {
            i8mm_store12<zero_mode>(c0_ptr, c1_ptr, mr > 1, v_c00, v_c01, v_c02, v_c03, v_c04,
                                    v_c05, vmin, vmax);
            i8mm_store12<zero_mode>(c2_ptr, c3_ptr, mr > 3, v_c10, v_c11, v_c12, v_c13, v_c14,
                                    v_c15, vmin, vmax);
            i8mm_store12<zero_mode>(c4_ptr, c5_ptr, mr > 5, v_c20, v_c21, v_c22, v_c23, v_c24,
                                    v_c25, vmin, vmax);
            i8mm_store12<zero_mode>(c6_ptr, c7_ptr, mr > 7, v_c30, v_c31, v_c32, v_c33, v_c34,
                                    v_c35, vmin, vmax);
        } else {
            const int rem = nc - n;
            i8mm_store12_tail<zero_mode>(c0_ptr, c1_ptr, mr > 1, rem, v_c00, v_c01, v_c02, v_c03,
                                         v_c04, v_c05, clamp_min, clamp_max);
            i8mm_store12_tail<zero_mode>(c2_ptr, c3_ptr, mr > 3, rem, v_c10, v_c11, v_c12, v_c13,
                                         v_c14, v_c15, clamp_min, clamp_max);
            i8mm_store12_tail<zero_mode>(c4_ptr, c5_ptr, mr > 5, rem, v_c20, v_c21, v_c22, v_c23,
                                         v_c24, v_c25, clamp_min, clamp_max);
            i8mm_store12_tail<zero_mode>(c6_ptr, c7_ptr, mr > 7, rem, v_c30, v_c31, v_c32, v_c33,
                                         v_c34, v_c35, clamp_min, clamp_max);
        }

        c0_ptr += 12;
        c1_ptr += 12;
        c2_ptr += 12;
        c3_ptr += 12;
        c4_ptr += 12;
        c5_ptr += 12;
        c6_ptr += 12;
        c7_ptr += 12;
        B += ldb_panel;
    }
}

template <bool zero_mode = false>
__attribute__((target("i8mm"))) inline void
mma_direct_mr4_nr12_i8mm(int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
                         const int8_t* NNOPS_RESTRICT A, int lda, const int8_t* NNOPS_RESTRICT B,
                         int ldb, int32_t clamp_min, int32_t clamp_max) noexcept
{
    int32_t* NNOPS_RESTRICT c0_ptr = C + 0 * ldc;
    int32_t* NNOPS_RESTRICT c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    int32_t* NNOPS_RESTRICT c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    int32_t* NNOPS_RESTRICT c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;

    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;

    const int kbytes = (kc + 7) & ~7;
    const int kg = kbytes >> 3;
    const int ldb_panel = mma_i8mm_panel_stride(ldb, kbytes);
    const int32x4_t vmin = vdupq_n_s32(clamp_min);
    const int32x4_t vmax = vdupq_n_s32(clamp_max);

    for (int n = 0; n < nc; n += mma_i8mm_nr_max) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;
        int32x4_t v_c03 = v_c00, v_c04 = v_c00, v_c05 = v_c00;
        int32x4_t v_c10 = vdupq_n_s32(0), v_c11 = v_c10, v_c12 = v_c10;
        int32x4_t v_c13 = v_c10, v_c14 = v_c10, v_c15 = v_c10;

        const int8_t* NNOPS_RESTRICT a0 = A + 0 * lda;
        const int8_t* NNOPS_RESTRICT a1 = A + r1 * lda;
        const int8_t* NNOPS_RESTRICT a2 = A + r2 * lda;
        const int8_t* NNOPS_RESTRICT a3 = A + r3 * lda;
        const int8_t* NNOPS_RESTRICT b_ptr = B;

        for (int k = 0; k < kg; ++k) {
            int8x16_t v_a0, v_a1;
            if (k + 1 < kg || (kc & 7) == 0) {
                v_a0 = vcombine_s8(vld1_s8(a0), vld1_s8(a1));
                v_a1 = vcombine_s8(vld1_s8(a2), vld1_s8(a3));
                a0 += 8;
                a1 += 8;
                a2 += 8;
                a3 += 8;
            } else {
                const int live = kc - 8 * k;
                int8_t t[4][8] = {};
                std::memcpy(t[0], a0, static_cast<size_t>(live));
                std::memcpy(t[1], a1, static_cast<size_t>(live));
                std::memcpy(t[2], a2, static_cast<size_t>(live));
                std::memcpy(t[3], a3, static_cast<size_t>(live));
                v_a0 = vcombine_s8(vld1_s8(t[0]), vld1_s8(t[1]));
                v_a1 = vcombine_s8(vld1_s8(t[2]), vld1_s8(t[3]));
            }

            const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);
            v_c00 = vmmlaq_s32(v_c00, v_a0, v_b0);
            v_c10 = vmmlaq_s32(v_c10, v_a1, v_b0);

            const int8x16_t v_b1 = vld1q_s8(b_ptr + 1 * 16);
            v_c01 = vmmlaq_s32(v_c01, v_a0, v_b1);
            v_c11 = vmmlaq_s32(v_c11, v_a1, v_b1);

            const int8x16_t v_b2 = vld1q_s8(b_ptr + 2 * 16);
            v_c02 = vmmlaq_s32(v_c02, v_a0, v_b2);
            v_c12 = vmmlaq_s32(v_c12, v_a1, v_b2);

            const int8x16_t v_b3 = vld1q_s8(b_ptr + 3 * 16);
            v_c03 = vmmlaq_s32(v_c03, v_a0, v_b3);
            v_c13 = vmmlaq_s32(v_c13, v_a1, v_b3);

            const int8x16_t v_b4 = vld1q_s8(b_ptr + 4 * 16);
            v_c04 = vmmlaq_s32(v_c04, v_a0, v_b4);
            v_c14 = vmmlaq_s32(v_c14, v_a1, v_b4);

            const int8x16_t v_b5 = vld1q_s8(b_ptr + 5 * 16);
            v_c05 = vmmlaq_s32(v_c05, v_a0, v_b5);
            v_c15 = vmmlaq_s32(v_c15, v_a1, v_b5);

            b_ptr += mma_i8mm_group_bytes;
        }

        if (n + mma_i8mm_nr_max <= nc) {
            i8mm_store12<zero_mode>(c0_ptr, c1_ptr, mr > 1, v_c00, v_c01, v_c02, v_c03, v_c04,
                                    v_c05, vmin, vmax);
            i8mm_store12<zero_mode>(c2_ptr, c3_ptr, mr > 3, v_c10, v_c11, v_c12, v_c13, v_c14,
                                    v_c15, vmin, vmax);
        } else {
            const int rem = nc - n;
            i8mm_store12_tail<zero_mode>(c0_ptr, c1_ptr, mr > 1, rem, v_c00, v_c01, v_c02, v_c03,
                                         v_c04, v_c05, clamp_min, clamp_max);
            i8mm_store12_tail<zero_mode>(c2_ptr, c3_ptr, mr > 3, rem, v_c10, v_c11, v_c12, v_c13,
                                         v_c14, v_c15, clamp_min, clamp_max);
        }

        c0_ptr += 12;
        c1_ptr += 12;
        c2_ptr += 12;
        c3_ptr += 12;
        B += ldb_panel;
    }
}

template <bool zero_mode = false>
__attribute__((target("i8mm"))) inline void
mma_direct_mr1_nr12_i8mm(int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
                         const int8_t* NNOPS_RESTRICT A, int lda, const int8_t* NNOPS_RESTRICT B,
                         int ldb, int32_t clamp_min, int32_t clamp_max) noexcept
{
    (void)mr;
    (void)ldc;
    (void)lda; // one row: A is read consecutively

    int32_t* NNOPS_RESTRICT c0_ptr = C;

    const int kbytes = (kc + 7) & ~7;
    const int kg = kbytes >> 3;
    const int ldb_panel = mma_i8mm_panel_stride(ldb, kbytes);
    const int32x4_t vmin = vdupq_n_s32(clamp_min);
    const int32x4_t vmax = vdupq_n_s32(clamp_max);
    const int8x8_t vz = vdup_n_s8(0);

    for (int n = 0; n < nc; n += mma_i8mm_nr_max) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;
        int32x4_t v_c03 = v_c00, v_c04 = v_c00, v_c05 = v_c00;

        const int8_t* NNOPS_RESTRICT a_ptr = A;
        const int8_t* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kg; ++k) {
            int8x8_t v_row;
            if (k + 1 < kg || (kc & 7) == 0) {
                v_row = vld1_s8(a_ptr);
                a_ptr += 8;
            } else {
                int8_t t[8] = {};
                std::memcpy(t, a_ptr, static_cast<size_t>(kc - 8 * k));
                v_row = vld1_s8(t);
            }
            const int8x16_t v_a = vcombine_s8(v_row, vz);

            v_c00 = vmmlaq_s32(v_c00, v_a, vld1q_s8(b_ptr + 0 * 16));
            v_c01 = vmmlaq_s32(v_c01, v_a, vld1q_s8(b_ptr + 1 * 16));
            v_c02 = vmmlaq_s32(v_c02, v_a, vld1q_s8(b_ptr + 2 * 16));
            v_c03 = vmmlaq_s32(v_c03, v_a, vld1q_s8(b_ptr + 3 * 16));
            v_c04 = vmmlaq_s32(v_c04, v_a, vld1q_s8(b_ptr + 4 * 16));
            v_c05 = vmmlaq_s32(v_c05, v_a, vld1q_s8(b_ptr + 5 * 16));

            b_ptr += mma_i8mm_group_bytes;
        }

        const int rem = nc - n;
        if (rem >= mma_i8mm_nr_max) {
            int32x4_t v = vcombine_s32(vget_low_s32(v_c00), vget_low_s32(v_c01));
            if constexpr (!zero_mode) {
                v = vaddq_s32(v, vld1q_s32(c0_ptr + 0));
            }
            vst1q_s32(c0_ptr + 0, vminq_s32(vmaxq_s32(v, vmin), vmax));
            v = vcombine_s32(vget_low_s32(v_c02), vget_low_s32(v_c03));
            if constexpr (!zero_mode) {
                v = vaddq_s32(v, vld1q_s32(c0_ptr + 4));
            }
            vst1q_s32(c0_ptr + 4, vminq_s32(vmaxq_s32(v, vmin), vmax));
            v = vcombine_s32(vget_low_s32(v_c04), vget_low_s32(v_c05));
            if constexpr (!zero_mode) {
                v = vaddq_s32(v, vld1q_s32(c0_ptr + 8));
            }
            vst1q_s32(c0_ptr + 8, vminq_s32(vmaxq_s32(v, vmin), vmax));
        } else {
            if (rem > 0) {
                int32_t x = vgetq_lane_s32(v_c00, 0);
                if constexpr (!zero_mode) {
                    x += c0_ptr[0];
                }
                c0_ptr[0] = std::min(std::max(x, clamp_min), clamp_max);
            }
            if (rem > 1) {
                int32_t x = vgetq_lane_s32(v_c00, 1);
                if constexpr (!zero_mode) {
                    x += c0_ptr[1];
                }
                c0_ptr[1] = std::min(std::max(x, clamp_min), clamp_max);
            }
            if (rem > 2) {
                int32_t x = vgetq_lane_s32(v_c01, 0);
                if constexpr (!zero_mode) {
                    x += c0_ptr[2];
                }
                c0_ptr[2] = std::min(std::max(x, clamp_min), clamp_max);
            }
            if (rem > 3) {
                int32_t x = vgetq_lane_s32(v_c01, 1);
                if constexpr (!zero_mode) {
                    x += c0_ptr[3];
                }
                c0_ptr[3] = std::min(std::max(x, clamp_min), clamp_max);
            }
            if (rem > 4) {
                int32_t x = vgetq_lane_s32(v_c02, 0);
                if constexpr (!zero_mode) {
                    x += c0_ptr[4];
                }
                c0_ptr[4] = std::min(std::max(x, clamp_min), clamp_max);
            }
            if (rem > 5) {
                int32_t x = vgetq_lane_s32(v_c02, 1);
                if constexpr (!zero_mode) {
                    x += c0_ptr[5];
                }
                c0_ptr[5] = std::min(std::max(x, clamp_min), clamp_max);
            }
            if (rem > 6) {
                int32_t x = vgetq_lane_s32(v_c03, 0);
                if constexpr (!zero_mode) {
                    x += c0_ptr[6];
                }
                c0_ptr[6] = std::min(std::max(x, clamp_min), clamp_max);
            }
            if (rem > 7) {
                int32_t x = vgetq_lane_s32(v_c03, 1);
                if constexpr (!zero_mode) {
                    x += c0_ptr[7];
                }
                c0_ptr[7] = std::min(std::max(x, clamp_min), clamp_max);
            }
            if (rem > 8) {
                int32_t x = vgetq_lane_s32(v_c04, 0);
                if constexpr (!zero_mode) {
                    x += c0_ptr[8];
                }
                c0_ptr[8] = std::min(std::max(x, clamp_min), clamp_max);
            }
            if (rem > 9) {
                int32_t x = vgetq_lane_s32(v_c04, 1);
                if constexpr (!zero_mode) {
                    x += c0_ptr[9];
                }
                c0_ptr[9] = std::min(std::max(x, clamp_min), clamp_max);
            }
            if (rem > 10) {
                int32_t x = vgetq_lane_s32(v_c05, 0);
                if constexpr (!zero_mode) {
                    x += c0_ptr[10];
                }
                c0_ptr[10] = std::min(std::max(x, clamp_min), clamp_max);
            }
            if (rem > 11) {
                int32_t x = vgetq_lane_s32(v_c05, 1);
                if constexpr (!zero_mode) {
                    x += c0_ptr[11];
                }
                c0_ptr[11] = std::min(std::max(x, clamp_min), clamp_max);
            }
        }

        c0_ptr += 12;
        B += ldb_panel;
    }
}

// Dispatch tables, consumed by matmul_helper.cpp; rows are the MR levels {8,4,1}.
template <bool Z>
inline constexpr auto mma_pack_fns_i8mm =
    std::array{&mma_pack_mr8_nr12_i8mm<Z>, &mma_pack_mr4_nr12_i8mm<Z>, &mma_pack_mr1_nr12_i8mm<Z>};

template <bool Z>
inline constexpr auto mma_direct_fns_i8mm = std::array{
    &mma_direct_mr8_nr12_i8mm<Z>, &mma_direct_mr4_nr12_i8mm<Z>, &mma_direct_mr1_nr12_i8mm<Z>};

} // namespace nnops::backend::cpu::aarch64

#endif // NNOPS_ARCH_I8MM
