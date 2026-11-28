#pragma once
/// @file mma_f16.hpp
/// @brief AArch64 NEON float16 mrxncxkc MMA kernels over uniform 16-wide B.
///
/// One templated rung per A presentation, with the height a template parameter:
/// `MR` ∈ {8, 4, 1} is the panel's row count, while the calls' `mr` (1..MR) is
/// the live height the M decomposition emits, so each level serves every panel
/// the largest-first walk produces instead of needing a kernel per height.
/// A and B may both be packed (`mma_pack_*`) or only B packed while A is read
/// row-major through `lda` (`mma_direct_*`), whose `ldb` is B's k-row step:
/// negative selects the packed layout (rows 16 apart), any other value is a
/// caller's matrix read in place. `zero_mode` drops the C read-add.
/// Accumulation, clamping and the C add are all in f16, like the rest of the
/// aarch64 fp16 kernels.
///
/// Packed B is uniform: ceil(nc / 16) panels, every one a full 16 columns wide
/// with the columns past `nc` zero-filled, each in a
/// mma_pack_align(16*kc*sizeof(float16_t))-byte slot. That lets the kernel read
/// 16 halfs per k at every column, so a tail wider than 8 costs one k-loop
/// instead of two; a tail of 8 or less takes the single-vector loop, and only
/// the lanes it stores are written.
///
/// Rows past `mr` alias the last live row — the convention pack_trans_* fills on
/// the A side and the C row pointers repeat here — so a partial-M tile needs no
/// per-row guards.
///
/// Bias seed: 1 is per-row (a dense row is an output channel), 2 is per-column
/// (NHWC, rows are pixels, so the bias lands on the panel's 16 lanes). Seed 2
/// reads all 16 lanes, so the caller's bias buffer must be padded to the panel
/// width. Seeding costs two loads per row per panel; priming C first would cost
/// a write and a read-back. The row seed goes through the same alias index as A
/// and C, so a partial M stays in range. The MR=1 rung reads one row: `mr` and
/// `ldc` are not applied (`lda` only exists on the packed-B route).
///
/// Every index into the accumulator block is a literal: the group walk and the
/// store are written out at the panel's two widths instead of looping to a
/// template bound. A loop costs nothing where the compiler unrolls it and
/// everything where it does not -- the array is then indexed at run time and
/// the whole block moves to the stack.

#include <array>
#include <arm_neon.h>
#include <algorithm>
#include "backend/cpu/common/common_utils.hpp"

namespace nnops::backend::cpu::aarch64 {

// =========================================================================
//  one walk: the group, the K loop, the stores
// =========================================================================
//
// Every rung -- packed A or row-major, any height -- is the same walk: one k
// step of MR rows against NG eight-column groups, then a store. The
// accumulators are laid out [NG][MR] so a column is contiguous, and every index
// into them is a literal (see the header).

/// The alias index of each row: rows past `mr` reuse the last live one.
template <int MR>
__attribute__((always_inline)) inline void
f16_row_alias(int mr, int (&rr)[MR]) noexcept
{
    rr[0] = 0;
    if constexpr (MR > 1) {
        rr[1] = std::min(1, mr - 1);
    }
    if constexpr (MR > 2) {
        rr[2] = std::min(2, mr - 1);
    }
    if constexpr (MR > 3) {
        rr[3] = std::min(3, mr - 1);
    }
    if constexpr (MR > 4) {
        rr[4] = std::min(4, mr - 1);
    }
    if constexpr (MR > 5) {
        rr[5] = std::min(5, mr - 1);
    }
    if constexpr (MR > 6) {
        rr[6] = std::min(6, mr - 1);
    }
    if constexpr (MR > 7) {
        rr[7] = std::min(7, mr - 1);
    }
}

/// The C rows, aliasing the last live one.
template <int MR>
__attribute__((always_inline)) inline void
f16_row_ptrs(float16_t* C, int64_t ldm, const int (&rr)[MR], float16_t* (&row)[MR]) noexcept
{
    row[0] = C;
    if constexpr (MR > 1) {
        row[1] = C + rr[1] * ldm;
    }
    if constexpr (MR > 2) {
        row[2] = C + rr[2] * ldm;
    }
    if constexpr (MR > 3) {
        row[3] = C + rr[3] * ldm;
    }
    if constexpr (MR > 4) {
        row[4] = C + rr[4] * ldm;
    }
    if constexpr (MR > 5) {
        row[5] = C + rr[5] * ldm;
    }
    if constexpr (MR > 6) {
        row[6] = C + rr[6] * ldm;
    }
    if constexpr (MR > 7) {
        row[7] = C + rr[7] * ldm;
    }
}

/// The row-major A rows, aliasing the last live one.
template <int MR>
__attribute__((always_inline)) inline void
f16_a_ptrs(const float16_t* A, int lda, const int (&rr)[MR], const float16_t* (&a)[MR]) noexcept
{
    a[0] = A;
    if constexpr (MR > 1) {
        a[1] = A + rr[1] * lda;
    }
    if constexpr (MR > 2) {
        a[2] = A + rr[2] * lda;
    }
    if constexpr (MR > 3) {
        a[3] = A + rr[3] * lda;
    }
    if constexpr (MR > 4) {
        a[4] = A + rr[4] * lda;
    }
    if constexpr (MR > 5) {
        a[5] = A + rr[5] * lda;
    }
    if constexpr (MR > 6) {
        a[6] = A + rr[6] * lda;
    }
    if constexpr (MR > 7) {
        a[7] = A + rr[7] * lda;
    }
}

/// One seed value: the row's bias (`r`), the panel column's (`col`) or zero.
template <int seed>
__attribute__((always_inline)) inline float16x8_t
f16_seed([[maybe_unused]] const float16_t* bias, [[maybe_unused]] int r,
         [[maybe_unused]] int col) noexcept
{
    static_assert(seed >= 0 && seed <= 2, "seed is 0 (zero), 1 (row) or 2 (column)");
    if constexpr (seed == 1) {
        return vdupq_n_f16(bias[r]);
    } else if constexpr (seed == 2) {
        return vld1q_f16(bias + col);
    } else {
        return vdupq_n_f16(0.0f16);
    }
}

/// The bias a panel starting at column `n` seeds from (seed 2 steps with it).
template <int seed>
__attribute__((always_inline)) inline const float16_t*
f16_panel_bias(const float16_t* bias, int n) noexcept
{
    if constexpr (seed == 2) {
        return bias + n;
    } else {
        return bias;
    }
}

/// Seed one column group's accumulators from the bias (or zero).
template <int MR, int seed>
__attribute__((always_inline)) inline void
f16_col_init(float16x8_t (&col)[MR], const float16_t* bias, const int (&rr)[MR],
             int col0) noexcept
{
    col[0] = f16_seed<seed>(bias, rr[0], col0);
    if constexpr (MR > 1) {
        col[1] = f16_seed<seed>(bias, rr[1], col0);
    }
    if constexpr (MR > 2) {
        col[2] = f16_seed<seed>(bias, rr[2], col0);
    }
    if constexpr (MR > 3) {
        col[3] = f16_seed<seed>(bias, rr[3], col0);
    }
    if constexpr (MR > 4) {
        col[4] = f16_seed<seed>(bias, rr[4], col0);
    }
    if constexpr (MR > 5) {
        col[5] = f16_seed<seed>(bias, rr[5], col0);
    }
    if constexpr (MR > 6) {
        col[6] = f16_seed<seed>(bias, rr[6], col0);
    }
    if constexpr (MR > 7) {
        col[7] = f16_seed<seed>(bias, rr[7], col0);
    }
}

/// One row against one column group, from a broadcast A value.
template <int MR>
__attribute__((always_inline)) inline void
f16_col_n(float16_t a, float16x8_t vb, float16x8_t (&col)[MR]) noexcept
{
    static_assert(MR == 1, "a broadcast is the one-row rung's form");
    col[0] = vfmaq_n_f16(col[0], vb, a);
}

/// One column group of the four-row rung: eight halfs hold four rows' A.
template <int MR>
__attribute__((always_inline)) inline void
f16_col4(float16x4_t va, float16x8_t vb, float16x8_t (&col)[MR]) noexcept
{
    static_assert(MR == 4, "a four-row A fragment is 4 halfs");
    col[0] = vfmaq_lane_f16(col[0], vb, va, 0);
    col[1] = vfmaq_lane_f16(col[1], vb, va, 1);
    col[2] = vfmaq_lane_f16(col[2], vb, va, 2);
    col[3] = vfmaq_lane_f16(col[3], vb, va, 3);
}

/// One column group of the eight-row rung: one 16-byte A register holds them.
template <int MR>
__attribute__((always_inline)) inline void
f16_col8(float16x8_t va, float16x8_t vb, float16x8_t (&col)[MR]) noexcept
{
    static_assert(MR == 8, "an eight-row A fragment is 8 halfs");
    col[0] = vfmaq_laneq_f16(col[0], vb, va, 0);
    col[1] = vfmaq_laneq_f16(col[1], vb, va, 1);
    col[2] = vfmaq_laneq_f16(col[2], vb, va, 2);
    col[3] = vfmaq_laneq_f16(col[3], vb, va, 3);
    col[4] = vfmaq_laneq_f16(col[4], vb, va, 4);
    col[5] = vfmaq_laneq_f16(col[5], vb, va, 5);
    col[6] = vfmaq_laneq_f16(col[6], vb, va, 6);
    col[7] = vfmaq_laneq_f16(col[7], vb, va, 7);
}

/// One k step of a packed-A panel: MR rows against NG eight-column groups.
template <int MR, int NG>
__attribute__((always_inline)) inline void
f16_group(const float16_t*& a_ptr, const float16_t* b_ptr, float16x8_t (&acc)[NG][MR]) noexcept
{
    static_assert(NG >= 1 && NG <= 2, "a panel is one or two column groups");
    if constexpr (MR == 8) {
        const float16x8_t va = vld1q_f16(a_ptr);
        a_ptr += 8;
        f16_col8<MR>(va, vld1q_f16(b_ptr + 0 * 8), acc[0]);
        if constexpr (NG > 1) {
            f16_col8<MR>(va, vld1q_f16(b_ptr + 1 * 8), acc[1]);
        }
    } else if constexpr (MR == 4) {
        const float16x4_t va = vld1_f16(a_ptr);
        a_ptr += 4;
        f16_col4<MR>(va, vld1q_f16(b_ptr + 0 * 8), acc[0]);
        if constexpr (NG > 1) {
            f16_col4<MR>(va, vld1q_f16(b_ptr + 1 * 8), acc[1]);
        }
    } else {
        // One half per k: a lane of a 16-byte load would read past a panel
        // packed one row per k, so the value arrives as a broadcast.
        const float16_t a = *a_ptr++;
        f16_col_n<MR>(a, vld1q_f16(b_ptr + 0 * 8), acc[0]);
        if constexpr (NG > 1) {
            f16_col_n<MR>(a, vld1q_f16(b_ptr + 1 * 8), acc[1]);
        }
    }
}

/// The panel's whole K from packed A into seeded `acc`.
template <int MR, int NG, int seed>
__attribute__((always_inline)) inline void
f16_mma(const float16_t* A, const float16_t* B, int kc, int bstep, const float16_t* bias,
        const int (&rr)[MR], float16x8_t (&acc)[NG][MR]) noexcept
{
    f16_col_init<MR, seed>(acc[0], bias, rr, 0);
    if constexpr (NG > 1) {
        f16_col_init<MR, seed>(acc[1], bias, rr, 8);
    }
    const float16_t* a_ptr = A;
    const float16_t* b_ptr = B;
    for (int k = 0; k < kc; ++k, b_ptr += bstep) {
        f16_group<MR, NG>(a_ptr, b_ptr, acc);
    }
}

/// One k step of a row-major panel: MR rows against NG column groups. Each
/// row's scalar is its own load -- there is no shared A vector to amortise
/// them from -- and nothing but the accumulators is live across the K loop.
template <int MR, int NG>
__attribute__((always_inline)) inline void
f16_step_rows(const float16_t* const (&a)[MR], const float16_t* b_ptr,
              float16x8_t (&acc)[NG][MR]) noexcept
{
    static_assert(NG >= 1 && NG <= 2, "a panel is one or two column groups");
    const float16x8_t vb0 = vld1q_f16(b_ptr + 0 * 8);
    float16x8_t vb1 = vb0;
    if constexpr (NG > 1) {
        vb1 = vld1q_f16(b_ptr + 1 * 8);
    }
    const float16_t v0 = *a[0];
    acc[0][0] = vfmaq_n_f16(acc[0][0], vb0, v0);
    if constexpr (NG > 1) {
        acc[1][0] = vfmaq_n_f16(acc[1][0], vb1, v0);
    }
    if constexpr (MR > 1) {
        const float16_t v1 = *a[1];
        acc[0][1] = vfmaq_n_f16(acc[0][1], vb0, v1);
        if constexpr (NG > 1) {
            acc[1][1] = vfmaq_n_f16(acc[1][1], vb1, v1);
        }
    }
    if constexpr (MR > 2) {
        const float16_t v2 = *a[2];
        acc[0][2] = vfmaq_n_f16(acc[0][2], vb0, v2);
        if constexpr (NG > 1) {
            acc[1][2] = vfmaq_n_f16(acc[1][2], vb1, v2);
        }
    }
    if constexpr (MR > 3) {
        const float16_t v3 = *a[3];
        acc[0][3] = vfmaq_n_f16(acc[0][3], vb0, v3);
        if constexpr (NG > 1) {
            acc[1][3] = vfmaq_n_f16(acc[1][3], vb1, v3);
        }
    }
    if constexpr (MR > 4) {
        const float16_t v4 = *a[4];
        acc[0][4] = vfmaq_n_f16(acc[0][4], vb0, v4);
        if constexpr (NG > 1) {
            acc[1][4] = vfmaq_n_f16(acc[1][4], vb1, v4);
        }
    }
    if constexpr (MR > 5) {
        const float16_t v5 = *a[5];
        acc[0][5] = vfmaq_n_f16(acc[0][5], vb0, v5);
        if constexpr (NG > 1) {
            acc[1][5] = vfmaq_n_f16(acc[1][5], vb1, v5);
        }
    }
    if constexpr (MR > 6) {
        const float16_t v6 = *a[6];
        acc[0][6] = vfmaq_n_f16(acc[0][6], vb0, v6);
        if constexpr (NG > 1) {
            acc[1][6] = vfmaq_n_f16(acc[1][6], vb1, v6);
        }
    }
    if constexpr (MR > 7) {
        const float16_t v7 = *a[7];
        acc[0][7] = vfmaq_n_f16(acc[0][7], vb0, v7);
        if constexpr (NG > 1) {
            acc[1][7] = vfmaq_n_f16(acc[1][7], vb1, v7);
        }
    }
}

/// Advance every A row pointer by `step` elements.
template <int MR>
__attribute__((always_inline)) inline void
f16_advance(const float16_t* (&a)[MR], int step) noexcept
{
    a[0] += step;
    if constexpr (MR > 1) {
        a[1] += step;
    }
    if constexpr (MR > 2) {
        a[2] += step;
    }
    if constexpr (MR > 3) {
        a[3] += step;
    }
    if constexpr (MR > 4) {
        a[4] += step;
    }
    if constexpr (MR > 5) {
        a[5] += step;
    }
    if constexpr (MR > 6) {
        a[6] += step;
    }
    if constexpr (MR > 7) {
        a[7] += step;
    }
}

/// The panel's whole K from row-major rows into seeded `acc`.
template <int MR, int NG, int seed>
__attribute__((always_inline)) inline void
f16_mma_rows(const float16_t* const (&A)[MR], const float16_t* B, int kc, int bstep,
             const float16_t* bias, const int (&rr)[MR], float16x8_t (&acc)[NG][MR]) noexcept
{
    f16_col_init<MR, seed>(acc[0], bias, rr, 0);
    if constexpr (NG > 1) {
        f16_col_init<MR, seed>(acc[1], bias, rr, 8);
    }
    const float16_t* a[MR];
    a[0] = A[0];
    if constexpr (MR > 1) {
        a[1] = A[1];
    }
    if constexpr (MR > 2) {
        a[2] = A[2];
    }
    if constexpr (MR > 3) {
        a[3] = A[3];
    }
    if constexpr (MR > 4) {
        a[4] = A[4];
    }
    if constexpr (MR > 5) {
        a[5] = A[5];
    }
    if constexpr (MR > 6) {
        a[6] = A[6];
    }
    if constexpr (MR > 7) {
        a[7] = A[7];
    }
    const float16_t* b_ptr = B;
    for (int k = 0; k < kc; ++k, b_ptr += bstep) {
        f16_step_rows<MR, NG>(a, b_ptr, acc);
        f16_advance<MR>(a, 1);
    }
}

// ---- the store ------------------------------------------------------------
//
// A group of eight columns is written in two passes: every row's C is folded
// into its accumulator first, then the rows are stored. Aliased rows share an
// address, so a row-at-a-time add would have the second duplicate reading C
// with the first one's result already in it. The narrower chunks of a ragged
// group keep the row-at-a-time form the older kernels used.

/// Fold C into one column group's accumulators, every row before any store.
template <int MR>
__attribute__((always_inline)) inline void
f16_add_c(float16_t* const (&row)[MR], int off, float16x8_t (&col)[MR]) noexcept
{
    col[0] = vaddq_f16(col[0], vld1q_f16(row[0] + off));
    if constexpr (MR > 1) {
        col[1] = vaddq_f16(col[1], vld1q_f16(row[1] + off));
    }
    if constexpr (MR > 2) {
        col[2] = vaddq_f16(col[2], vld1q_f16(row[2] + off));
    }
    if constexpr (MR > 3) {
        col[3] = vaddq_f16(col[3], vld1q_f16(row[3] + off));
    }
    if constexpr (MR > 4) {
        col[4] = vaddq_f16(col[4], vld1q_f16(row[4] + off));
    }
    if constexpr (MR > 5) {
        col[5] = vaddq_f16(col[5], vld1q_f16(row[5] + off));
    }
    if constexpr (MR > 6) {
        col[6] = vaddq_f16(col[6], vld1q_f16(row[6] + off));
    }
    if constexpr (MR > 7) {
        col[7] = vaddq_f16(col[7], vld1q_f16(row[7] + off));
    }
}

/// Clamp and store one column group, no C add.
template <int MR>
__attribute__((always_inline)) inline void
f16_store_c(float16_t* const (&row)[MR], int off, const float16x8_t (&col)[MR],
            float16x8_t vmin, float16x8_t vmax) noexcept
{
    vst1q_f16(row[0] + off, vminq_f16(vmaxq_f16(col[0], vmin), vmax));
    if constexpr (MR > 1) {
        vst1q_f16(row[1] + off, vminq_f16(vmaxq_f16(col[1], vmin), vmax));
    }
    if constexpr (MR > 2) {
        vst1q_f16(row[2] + off, vminq_f16(vmaxq_f16(col[2], vmin), vmax));
    }
    if constexpr (MR > 3) {
        vst1q_f16(row[3] + off, vminq_f16(vmaxq_f16(col[3], vmin), vmax));
    }
    if constexpr (MR > 4) {
        vst1q_f16(row[4] + off, vminq_f16(vmaxq_f16(col[4], vmin), vmax));
    }
    if constexpr (MR > 5) {
        vst1q_f16(row[5] + off, vminq_f16(vmaxq_f16(col[5], vmin), vmax));
    }
    if constexpr (MR > 6) {
        vst1q_f16(row[6] + off, vminq_f16(vmaxq_f16(col[6], vmin), vmax));
    }
    if constexpr (MR > 7) {
        vst1q_f16(row[7] + off, vminq_f16(vmaxq_f16(col[7], vmin), vmax));
    }
}

/// The four-lane chunk of a row: read C, add, clamp.
template <bool zero_mode>
__attribute__((always_inline)) inline float16x4_t
f16_lane4(const float16_t* p, float16x8_t v, float16x4_t vmin4, float16x4_t vmax4) noexcept
{
    float16x4_t v0 = vget_low_f16(v);
    if constexpr (!zero_mode) {
        v0 = vadd_f16(v0, vld1_f16(p));
    }
    return vmin_f16(vmax_f16(v0, vmin4), vmax4);
}

/// The two-lane chunk of a row.
template <bool zero_mode>
__attribute__((always_inline)) inline float16x4_t
f16_lane2(const float16_t* p, float16x8_t v, bool hi, float16x4_t vmin4,
          float16x4_t vmax4) noexcept
{
    float16x4_t v0 = hi ? vget_high_f16(v) : vget_low_f16(v);
    if constexpr (!zero_mode) {
        v0 = vset_lane_f16(vget_lane_f16(v0, 0) + p[0], v0, 0);
        v0 = vset_lane_f16(vget_lane_f16(v0, 1) + p[1], v0, 1);
    }
    return vmin_f16(vmax_f16(v0, vmin4), vmax4);
}

/// The single lane of a row.
template <bool zero_mode>
__attribute__((always_inline)) inline float16_t
f16_lane1(const float16_t* p, float16x8_t v, int lanes, float16_t cmin, float16_t cmax) noexcept
{
    float16_t x = (lanes & 6) == 6 ? vgetq_lane_f16(v, 6) :
                  (lanes & 4)      ? vgetq_lane_f16(v, 4) :
                  (lanes & 2)      ? vgetq_lane_f16(v, 2) :
                                     vgetq_lane_f16(v, 0);
    if constexpr (!zero_mode) {
        x = x + p[0];
    }
    return std::min(std::max(x, cmin), cmax);
}

/// The live lanes of a group: one stage (four halfs, then two, then one) for
/// every row before that stage's stores, so duplicate (aliased) rows all read
/// the same C.
template <int MR, bool zero_mode>
__attribute__((always_inline)) inline void
f16_store_lanes(float16_t* const (&row)[MR], int off, int lanes, const float16x8_t (&col)[MR],
                float16x8_t vmin, float16x8_t vmax, float16_t cmin, float16_t cmax) noexcept
{
    const float16x4_t vmin4 = vget_low_f16(vmin);
    const float16x4_t vmax4 = vget_low_f16(vmax);
    if (lanes & 4) {
        float16x4_t o[MR];
        o[0] = f16_lane4<zero_mode>(row[0] + off, col[0], vmin4, vmax4);
        if constexpr (MR > 1) {
            o[1] = f16_lane4<zero_mode>(row[1] + off, col[1], vmin4, vmax4);
        }
        if constexpr (MR > 2) {
            o[2] = f16_lane4<zero_mode>(row[2] + off, col[2], vmin4, vmax4);
        }
        if constexpr (MR > 3) {
            o[3] = f16_lane4<zero_mode>(row[3] + off, col[3], vmin4, vmax4);
        }
        if constexpr (MR > 4) {
            o[4] = f16_lane4<zero_mode>(row[4] + off, col[4], vmin4, vmax4);
        }
        if constexpr (MR > 5) {
            o[5] = f16_lane4<zero_mode>(row[5] + off, col[5], vmin4, vmax4);
        }
        if constexpr (MR > 6) {
            o[6] = f16_lane4<zero_mode>(row[6] + off, col[6], vmin4, vmax4);
        }
        if constexpr (MR > 7) {
            o[7] = f16_lane4<zero_mode>(row[7] + off, col[7], vmin4, vmax4);
        }
        vst1_f16(row[0] + off, o[0]);
        if constexpr (MR > 1) {
            vst1_f16(row[1] + off, o[1]);
        }
        if constexpr (MR > 2) {
            vst1_f16(row[2] + off, o[2]);
        }
        if constexpr (MR > 3) {
            vst1_f16(row[3] + off, o[3]);
        }
        if constexpr (MR > 4) {
            vst1_f16(row[4] + off, o[4]);
        }
        if constexpr (MR > 5) {
            vst1_f16(row[5] + off, o[5]);
        }
        if constexpr (MR > 6) {
            vst1_f16(row[6] + off, o[6]);
        }
        if constexpr (MR > 7) {
            vst1_f16(row[7] + off, o[7]);
        }
        off += 4;
    }
    if (lanes & 2) {
        const bool hi = (lanes & 4) != 0;
        float16x4_t o[MR];
        o[0] = f16_lane2<zero_mode>(row[0] + off, col[0], hi, vmin4, vmax4);
        if constexpr (MR > 1) {
            o[1] = f16_lane2<zero_mode>(row[1] + off, col[1], hi, vmin4, vmax4);
        }
        if constexpr (MR > 2) {
            o[2] = f16_lane2<zero_mode>(row[2] + off, col[2], hi, vmin4, vmax4);
        }
        if constexpr (MR > 3) {
            o[3] = f16_lane2<zero_mode>(row[3] + off, col[3], hi, vmin4, vmax4);
        }
        if constexpr (MR > 4) {
            o[4] = f16_lane2<zero_mode>(row[4] + off, col[4], hi, vmin4, vmax4);
        }
        if constexpr (MR > 5) {
            o[5] = f16_lane2<zero_mode>(row[5] + off, col[5], hi, vmin4, vmax4);
        }
        if constexpr (MR > 6) {
            o[6] = f16_lane2<zero_mode>(row[6] + off, col[6], hi, vmin4, vmax4);
        }
        if constexpr (MR > 7) {
            o[7] = f16_lane2<zero_mode>(row[7] + off, col[7], hi, vmin4, vmax4);
        }
        vst1_lane_f16(row[0] + off + 0, o[0], 0);
        vst1_lane_f16(row[0] + off + 1, o[0], 1);
        if constexpr (MR > 1) {
            vst1_lane_f16(row[1] + off + 0, o[1], 0);
            vst1_lane_f16(row[1] + off + 1, o[1], 1);
        }
        if constexpr (MR > 2) {
            vst1_lane_f16(row[2] + off + 0, o[2], 0);
            vst1_lane_f16(row[2] + off + 1, o[2], 1);
        }
        if constexpr (MR > 3) {
            vst1_lane_f16(row[3] + off + 0, o[3], 0);
            vst1_lane_f16(row[3] + off + 1, o[3], 1);
        }
        if constexpr (MR > 4) {
            vst1_lane_f16(row[4] + off + 0, o[4], 0);
            vst1_lane_f16(row[4] + off + 1, o[4], 1);
        }
        if constexpr (MR > 5) {
            vst1_lane_f16(row[5] + off + 0, o[5], 0);
            vst1_lane_f16(row[5] + off + 1, o[5], 1);
        }
        if constexpr (MR > 6) {
            vst1_lane_f16(row[6] + off + 0, o[6], 0);
            vst1_lane_f16(row[6] + off + 1, o[6], 1);
        }
        if constexpr (MR > 7) {
            vst1_lane_f16(row[7] + off + 0, o[7], 0);
            vst1_lane_f16(row[7] + off + 1, o[7], 1);
        }
        off += 2;
    }
    if (lanes & 1) {
        float16_t o[MR];
        o[0] = f16_lane1<zero_mode>(row[0] + off, col[0], lanes, cmin, cmax);
        if constexpr (MR > 1) {
            o[1] = f16_lane1<zero_mode>(row[1] + off, col[1], lanes, cmin, cmax);
        }
        if constexpr (MR > 2) {
            o[2] = f16_lane1<zero_mode>(row[2] + off, col[2], lanes, cmin, cmax);
        }
        if constexpr (MR > 3) {
            o[3] = f16_lane1<zero_mode>(row[3] + off, col[3], lanes, cmin, cmax);
        }
        if constexpr (MR > 4) {
            o[4] = f16_lane1<zero_mode>(row[4] + off, col[4], lanes, cmin, cmax);
        }
        if constexpr (MR > 5) {
            o[5] = f16_lane1<zero_mode>(row[5] + off, col[5], lanes, cmin, cmax);
        }
        if constexpr (MR > 6) {
            o[6] = f16_lane1<zero_mode>(row[6] + off, col[6], lanes, cmin, cmax);
        }
        if constexpr (MR > 7) {
            o[7] = f16_lane1<zero_mode>(row[7] + off, col[7], lanes, cmin, cmax);
        }
        row[0][off] = o[0];
        if constexpr (MR > 1) {
            row[1][off] = o[1];
        }
        if constexpr (MR > 2) {
            row[2][off] = o[2];
        }
        if constexpr (MR > 3) {
            row[3][off] = o[3];
        }
        if constexpr (MR > 4) {
            row[4][off] = o[4];
        }
        if constexpr (MR > 5) {
            row[5][off] = o[5];
        }
        if constexpr (MR > 6) {
            row[6][off] = o[6];
        }
        if constexpr (MR > 7) {
            row[7][off] = o[7];
        }
    }
}

/// Write one column group of a panel: MR rows, `live` columns left in the panel
/// from this group's first column.
template <int MR, bool zero_mode>
__attribute__((always_inline)) inline void
f16_store_col(float16_t* const (&row)[MR], int col0, int col, int live,
              float16x8_t (&col_acc)[MR], float16x8_t vmin, float16x8_t vmax, float16_t cmin,
              float16_t cmax) noexcept
{
    const int l = live - col;
    if (l >= 8) {
        if constexpr (!zero_mode) {
            f16_add_c<MR>(row, col0 + col, col_acc);
        }
        f16_store_c<MR>(row, col0 + col, col_acc, vmin, vmax);
    } else if (l > 0) {
        f16_store_lanes<MR, zero_mode>(row, col0 + col, l, col_acc, vmin, vmax, cmin, cmax);
    }
}

/// Write the live columns of NG groups, starting at `col0`.
template <int MR, int NG, bool zero_mode>
__attribute__((always_inline)) inline void
f16_store(float16_t* const (&row)[MR], int col0, int live, float16x8_t (&acc)[NG][MR],
          float16x8_t vmin, float16x8_t vmax, float16_t cmin, float16_t cmax) noexcept
{
    static_assert(NG >= 1 && NG <= 2, "a panel is one or two column groups");
    f16_store_col<MR, zero_mode>(row, col0, 0, live, acc[0], vmin, vmax, cmin, cmax);
    if constexpr (NG > 1) {
        f16_store_col<MR, zero_mode>(row, col0, 8, live, acc[1], vmin, vmax, cmin, cmax);
    }
}

/// One narrow remainder panel through NG groups, packed A.
template <int MR, int NG, bool zero_mode, int seed>
__attribute__((always_inline)) inline void
f16_tail(float16_t* const (&row)[MR], int n, int live, int kc, int bstep, const float16_t* A,
         const float16_t* B, const float16_t* bias, const int (&rr)[MR], float16x8_t vmin,
         float16x8_t vmax, float16_t cmin, float16_t cmax) noexcept
{
    float16x8_t acc[NG][MR];
    f16_mma<MR, NG, seed>(A, B, kc, bstep, bias, rr, acc);
    f16_store<MR, NG, zero_mode>(row, n, live, acc, vmin, vmax, cmin, cmax);
}

/// One narrow remainder panel through NG groups, row-major A.
template <int MR, int NG, bool zero_mode, int seed>
__attribute__((always_inline)) inline void
f16_tail_rows(float16_t* const (&row)[MR], int n, int live, int kc, int bstep,
              const float16_t* const (&a)[MR], const float16_t* B, const float16_t* bias,
              const int (&rr)[MR], float16x8_t vmin, float16x8_t vmax, float16_t cmin,
              float16_t cmax) noexcept
{
    float16x8_t acc[NG][MR];
    f16_mma_rows<MR, NG, seed>(a, B, kc, bstep, bias, rr, acc);
    f16_store<MR, NG, zero_mode>(row, n, live, acc, vmin, vmax, cmin, cmax);
}

// =========================================================================
//  packed A + packed B  (MR = 8, 4, 1)
// =========================================================================

/// One packed-A rung: the whole N side into C.
template <int MR, bool zero_mode = false, int seed = 0>
inline void mma_pack_mr_nr16_mrxncxkc_f16(int mr, int nc, int kc, float16_t* NNOPS_RESTRICT C,
                                          int ldc, const float16_t* NNOPS_RESTRICT A,
                                          const float16_t* NNOPS_RESTRICT B, float clamp_min,
                                          float clamp_max,
                                          const float16_t* NNOPS_RESTRICT bias = nullptr) noexcept
{
    int rr[MR];
    f16_row_alias<MR>(mr, rr);
    float16_t* row[MR];
    f16_row_ptrs<MR>(C, ldc, rr, row);
    const int ldb_panel =
        static_cast<int>(mma_pack_align(16 * kc * sizeof(float16_t)) / sizeof(float16_t));
    const float16_t cmin = static_cast<float16_t>(clamp_min);
    const float16_t cmax = static_cast<float16_t>(clamp_max);
    const float16x8_t vmin = vdupq_n_f16(cmin);
    const float16x8_t vmax = vdupq_n_f16(cmax);

    int n = 0;
    for (; n + 16 <= nc; n += 16) {
        float16x8_t acc[2][MR];
        f16_mma<MR, 2, seed>(A, B, kc, 16, f16_panel_bias<seed>(bias, n), rr, acc);
        f16_store<MR, 2, zero_mode>(row, n, 16, acc, vmin, vmax, cmin, cmax);
        B += ldb_panel;
    }

    // The live remainder narrows to the groups it needs: 8 or 16 columns.
    const int live = nc - n;
    if (live > 8) {
        f16_tail<MR, 2, zero_mode, seed>(row, n, live, kc, 16, A, B,
                                         f16_panel_bias<seed>(bias, n), rr, vmin, vmax, cmin,
                                         cmax);
    } else if (live > 0) {
        f16_tail<MR, 1, zero_mode, seed>(row, n, live, kc, 16, A, B,
                                         f16_panel_bias<seed>(bias, n), rr, vmin, vmax, cmin,
                                         cmax);
    }
}

// =========================================================================
//  raw A + packed B  (MR = 8, 4, 1)
// =========================================================================
//
// The same walk with each row's scalar loaded per k instead of a shared 16-byte
// panel load, so the rung reads A through `lda`.

/// One row-major-A rung: the whole N side into C.
template <int MR, bool zero_mode = false, int seed = 0>
inline void mma_direct_mr_nr16_mrxncxkc_f16(
    int mr, int nc, int kc, float16_t* NNOPS_RESTRICT C, int ldc,
    const float16_t* NNOPS_RESTRICT A, int lda, const float16_t* NNOPS_RESTRICT B, int ldb,
    float clamp_min, float clamp_max, const float16_t* NNOPS_RESTRICT bias = nullptr) noexcept
{
    int rr[MR];
    f16_row_alias<MR>(mr, rr);
    float16_t* row[MR];
    f16_row_ptrs<MR>(C, ldc, rr, row);
    const float16_t* a0[MR];
    f16_a_ptrs<MR>(A, lda, rr, a0);
    const int ldb_panel =
        static_cast<int>(mma_pack_align(16 * kc * sizeof(float16_t)) / sizeof(float16_t));
    const int bstep = (ldb < 0) ? 16 : ldb;
    int panel_step = ldb_panel;
    if constexpr (MR == 1) {
        // Below -1 a negative ldb is how far apart the caller's packed panels
        // sit; without one the panels are this call's own and the slot is right.
        if (ldb < -1) {
            panel_step = -ldb;
        }
    }
    const float16_t cmin = static_cast<float16_t>(clamp_min);
    const float16_t cmax = static_cast<float16_t>(clamp_max);
    const float16x8_t vmin = vdupq_n_f16(cmin);
    const float16x8_t vmax = vdupq_n_f16(cmax);

    int n = 0;
    if constexpr (MR == 1) {
        // Two packed panels per pass: at one row the loop is nothing but weight
        // stream, and its scalar A load costs a whole load slot of its own, so
        // one scalar over 32 columns is the margin.
        for (; ldb < 0 && n <= nc - 32; n += 32) {
            const float16_t* const bp = f16_panel_bias<seed>(bias, n);
            float16x8_t acc[4][1];
            f16_col_init<1, seed>(acc[0], bp, rr, 0);
            f16_col_init<1, seed>(acc[1], bp, rr, 8);
            f16_col_init<1, seed>(acc[2], bp, rr, 16);
            f16_col_init<1, seed>(acc[3], bp, rr, 24);
            const float16_t* ap = A;
            const float16_t* b0_ptr = B;
            const float16_t* b1_ptr = B + panel_step;
            for (int k = 0; k < kc; ++k) {
                const float16x8_t d0 = vld1q_f16(b0_ptr + 0 * 8);
                const float16x8_t d1 = vld1q_f16(b0_ptr + 1 * 8);
                const float16x8_t d2 = vld1q_f16(b1_ptr + 0 * 8);
                const float16x8_t d3 = vld1q_f16(b1_ptr + 1 * 8);
                b0_ptr += bstep;
                b1_ptr += bstep;
                const float16_t a = *ap++;
                f16_col_n<1>(a, d0, acc[0]);
                f16_col_n<1>(a, d1, acc[1]);
                f16_col_n<1>(a, d2, acc[2]);
                f16_col_n<1>(a, d3, acc[3]);
            }
            f16_store_col<1, zero_mode>(row, n, 0, 32, acc[0], vmin, vmax, cmin, cmax);
            f16_store_col<1, zero_mode>(row, n, 8, 32, acc[1], vmin, vmax, cmin, cmax);
            f16_store_col<1, zero_mode>(row, n, 16, 32, acc[2], vmin, vmax, cmin, cmax);
            f16_store_col<1, zero_mode>(row, n, 24, 32, acc[3], vmin, vmax, cmin, cmax);
            B += 2 * panel_step;
        }
    }

    for (; n + 16 <= nc; n += 16) {
        float16x8_t acc[2][MR];
        f16_mma_rows<MR, 2, seed>(a0, B, kc, bstep, f16_panel_bias<seed>(bias, n), rr, acc);
        f16_store<MR, 2, zero_mode>(row, n, 16, acc, vmin, vmax, cmin, cmax);
        B += panel_step;
    }

    // The live remainder narrows to the groups it needs, as the packed rung's
    // does.
    const int live = nc - n;
    if (live > 8) {
        f16_tail_rows<MR, 2, zero_mode, seed>(row, n, live, kc, bstep, a0, B,
                                              f16_panel_bias<seed>(bias, n), rr, vmin, vmax,
                                              cmin, cmax);
    } else if (live > 0) {
        f16_tail_rows<MR, 1, zero_mode, seed>(row, n, live, kc, bstep, a0, B,
                                              f16_panel_bias<seed>(bias, n), rr, vmin, vmax,
                                              cmin, cmax);
    }
}

// ---- dispatch tables, consumed by matmul_helper.cpp ---------------------

template <bool Z, int Seed = 0>
inline constexpr auto mma_pack_fns_f16 =
    std::array{&mma_pack_mr_nr16_mrxncxkc_f16<8, Z, Seed>,
               &mma_pack_mr_nr16_mrxncxkc_f16<4, Z, Seed>,
               &mma_pack_mr_nr16_mrxncxkc_f16<1, Z, Seed>};

template <bool Z, int Seed = 0>
inline constexpr auto mma_direct_fns_f16 =
    std::array{&mma_direct_mr_nr16_mrxncxkc_f16<8, Z, Seed>,
               &mma_direct_mr_nr16_mrxncxkc_f16<4, Z, Seed>,
               &mma_direct_mr_nr16_mrxncxkc_f16<1, Z, Seed>};

} // namespace nnops::backend::cpu::aarch64
