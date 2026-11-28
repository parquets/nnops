#pragma once
/// @file mma_f32.hpp
/// @brief AArch64 NEON float32 mrxncxkc MMA kernels over uniform 12-wide B.
///
/// One templated rung per A presentation, with the height a template parameter:
/// `MR` ∈ {8, 4, 1} is the panel's row count, while the calls' `mr` (1..MR) is
/// the live height the M decomposition emits, so each level serves every panel
/// the largest-first walk produces instead of needing a kernel per height.
///
/// A and B may both be packed (`mma_pack_*`) or only B packed while A is
/// read row-major (`mma_direct_*`), whose `ldb` is B's k-row step: negative
/// selects the packed layout (rows 12 apart), any other value is a caller's
/// matrix read in place. `zero_mode` drops the C read-add.
///
/// Packed B is uniform: ceil(nc / 12) panels, every one a full 12 columns wide,
/// each in a mma_pack_align(12*kc*sizeof(float))-byte slot (12*kc floats). The
/// columns past `nc` are padding the tail's store discards, so the tail works
/// one panel wide and runs only the groups it stores: three for a 9-12 column
/// remainder, two for 5-8, one for 1-4. The dead lanes past `nc` are never
/// accumulated at all rather than computed and thrown away.
///
/// Rows past `mr` alias the last live row in A and C -- the convention the
/// packers fill on the A side and the C row pointers repeat -- so a partial-M
/// tile needs no per-row guards: the tail stores the duplicate rows back to the
/// same address.
///
/// Bias seed: 1 is per-row (`bias[r]`) -- a dense row is an output channel;
/// 2 is per-column (`bias + n + off`) -- the NHWC shape, where rows are pixels
/// and the bias lands on the panel's lanes. Seeding each row's three panel
/// vectors costs a few loads, where priming the tile first would cost a full
/// output write and read-back. Seed 2 loads a whole 12-lane panel even on a
/// short tail, since `n` is its start. The MR=1 rung ignores `mr` and `ldc`,
/// and the packed-B one `lda` too.
///
/// Every index into the accumulator block is a literal: the group walk and the
/// store are written out at the panel's three widths instead of looping to a
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
// step of MR rows against NG four-column groups, then a store. The
// accumulators are laid out [NG][MR] so a column is contiguous, and every index
// into them is a literal (see the header).

/// The alias index of each row: rows past `mr` reuse the last live one.
template <int MR>
__attribute__((always_inline)) inline void
f32_row_alias(int mr, int (&rr)[MR]) noexcept
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
f32_row_ptrs(float* C, int64_t ldm, const int (&rr)[MR], float* (&row)[MR]) noexcept
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
f32_a_ptrs(const float* A, int lda, const int (&rr)[MR], const float* (&a)[MR]) noexcept
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
__attribute__((always_inline)) inline float32x4_t
f32_seed([[maybe_unused]] const float* bias, [[maybe_unused]] int r,
         [[maybe_unused]] int col) noexcept
{
    static_assert(seed >= 0 && seed <= 2, "seed is 0 (zero), 1 (row) or 2 (column)");
    if constexpr (seed == 1) {
        return vdupq_n_f32(bias[r]);
    } else if constexpr (seed == 2) {
        return vld1q_f32(bias + col);
    } else {
        return vdupq_n_f32(0.0f);
    }
}

/// The bias a panel starting at column `n` seeds from (seed 2 steps with it).
template <int seed>
__attribute__((always_inline)) inline const float*
f32_panel_bias(const float* bias, int n) noexcept
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
f32_col_init(float32x4_t (&col)[MR], const float* bias, const int (&rr)[MR],
             int col0) noexcept
{
    col[0] = f32_seed<seed>(bias, rr[0], col0);
    if constexpr (MR > 1) {
        col[1] = f32_seed<seed>(bias, rr[1], col0);
    }
    if constexpr (MR > 2) {
        col[2] = f32_seed<seed>(bias, rr[2], col0);
    }
    if constexpr (MR > 3) {
        col[3] = f32_seed<seed>(bias, rr[3], col0);
    }
    if constexpr (MR > 4) {
        col[4] = f32_seed<seed>(bias, rr[4], col0);
    }
    if constexpr (MR > 5) {
        col[5] = f32_seed<seed>(bias, rr[5], col0);
    }
    if constexpr (MR > 6) {
        col[6] = f32_seed<seed>(bias, rr[6], col0);
    }
    if constexpr (MR > 7) {
        col[7] = f32_seed<seed>(bias, rr[7], col0);
    }
}

/// One row against one column group, from a broadcast A value.
template <int MR>
__attribute__((always_inline)) inline void
f32_col_n(float a, float32x4_t vb, float32x4_t (&col)[MR]) noexcept
{
    static_assert(MR == 1, "a broadcast is the one-row rung's form");
    col[0] = vfmaq_n_f32(col[0], vb, a);
}

/// One column group: MR rows (as one or two 16-byte A registers) against one B
/// vector, into that column's MR accumulators.
template <int MR>
__attribute__((always_inline)) inline void
f32_col(float32x4_t va0, float32x4_t va1, float32x4_t vb, float32x4_t (&col)[MR]) noexcept
{
    if constexpr (MR == 4) {
        col[0] = vfmaq_laneq_f32(col[0], vb, va0, 0);
        col[1] = vfmaq_laneq_f32(col[1], vb, va0, 1);
        col[2] = vfmaq_laneq_f32(col[2], vb, va0, 2);
        col[3] = vfmaq_laneq_f32(col[3], vb, va0, 3);
    } else {
        static_assert(MR == 8, "a float A panel is 4 or 8 rows");
        col[0] = vfmaq_laneq_f32(col[0], vb, va0, 0);
        col[1] = vfmaq_laneq_f32(col[1], vb, va0, 1);
        col[2] = vfmaq_laneq_f32(col[2], vb, va0, 2);
        col[3] = vfmaq_laneq_f32(col[3], vb, va0, 3);
        col[4] = vfmaq_laneq_f32(col[4], vb, va1, 0);
        col[5] = vfmaq_laneq_f32(col[5], vb, va1, 1);
        col[6] = vfmaq_laneq_f32(col[6], vb, va1, 2);
        col[7] = vfmaq_laneq_f32(col[7], vb, va1, 3);
    }
}

/// One k step of a packed-A panel: MR rows against NG four-column groups.
template <int MR, int NG>
__attribute__((always_inline)) inline void
f32_group(const float*& a_ptr, const float* b_ptr, float32x4_t (&acc)[NG][MR]) noexcept
{
    static_assert(NG >= 1 && NG <= 3, "a panel is one to three column groups");
    if constexpr (MR == 1) {
        // One float per k: a lane of a 16-byte load would read past a panel
        // packed one row per k, so the value arrives as a broadcast.
        const float a = *a_ptr++;
        f32_col_n<MR>(a, vld1q_f32(b_ptr + 0 * 4), acc[0]);
        if constexpr (NG > 1) {
            f32_col_n<MR>(a, vld1q_f32(b_ptr + 1 * 4), acc[1]);
        }
        if constexpr (NG > 2) {
            f32_col_n<MR>(a, vld1q_f32(b_ptr + 2 * 4), acc[2]);
        }
    } else {
        const float32x4_t va0 = vld1q_f32(a_ptr);
        const float32x4_t va1 = (MR == 8) ? vld1q_f32(a_ptr + 4) : va0;
        a_ptr += (MR == 8) ? 8 : 4;
        f32_col<MR>(va0, va1, vld1q_f32(b_ptr + 0 * 4), acc[0]);
        if constexpr (NG > 1) {
            f32_col<MR>(va0, va1, vld1q_f32(b_ptr + 1 * 4), acc[1]);
        }
        if constexpr (NG > 2) {
            f32_col<MR>(va0, va1, vld1q_f32(b_ptr + 2 * 4), acc[2]);
        }
    }
}

/// The panel's whole K from packed A into seeded `acc`.
template <int MR, int NG, int seed>
__attribute__((always_inline)) inline void
f32_mma(const float* A, const float* B, int kc, int bstep, const float* bias,
        const int (&rr)[MR], float32x4_t (&acc)[NG][MR]) noexcept
{
    f32_col_init<MR, seed>(acc[0], bias, rr, 0);
    if constexpr (NG > 1) {
        f32_col_init<MR, seed>(acc[1], bias, rr, 4);
    }
    if constexpr (NG > 2) {
        f32_col_init<MR, seed>(acc[2], bias, rr, 8);
    }
    const float* a_ptr = A;
    const float* b_ptr = B;
    for (int k = 0; k < kc; ++k, b_ptr += bstep) {
        f32_group<MR, NG>(a_ptr, b_ptr, acc);
    }
}

/// One k step of a row-major panel: MR rows against NG column groups. Each
/// row's scalar is its own load -- there is no shared A vector to amortise
/// them from -- and nothing but the accumulators is live across the K loop.
template <int MR, int NG>
__attribute__((always_inline)) inline void
f32_step_rows(const float* const (&a)[MR], const float* b_ptr,
              float32x4_t (&acc)[NG][MR]) noexcept
{
    static_assert(NG >= 1 && NG <= 3, "a panel is one to three column groups");
    const float32x4_t vb0 = vld1q_f32(b_ptr + 0 * 4);
    float32x4_t vb1 = vb0;
    float32x4_t vb2 = vb0;
    if constexpr (NG > 1) {
        vb1 = vld1q_f32(b_ptr + 1 * 4);
    }
    if constexpr (NG > 2) {
        vb2 = vld1q_f32(b_ptr + 2 * 4);
    }
    const float v0 = *a[0];
    acc[0][0] = vfmaq_n_f32(acc[0][0], vb0, v0);
    if constexpr (NG > 1) {
        acc[1][0] = vfmaq_n_f32(acc[1][0], vb1, v0);
    }
    if constexpr (NG > 2) {
        acc[2][0] = vfmaq_n_f32(acc[2][0], vb2, v0);
    }
    if constexpr (MR > 1) {
        const float v1 = *a[1];
        acc[0][1] = vfmaq_n_f32(acc[0][1], vb0, v1);
        if constexpr (NG > 1) {
            acc[1][1] = vfmaq_n_f32(acc[1][1], vb1, v1);
        }
        if constexpr (NG > 2) {
            acc[2][1] = vfmaq_n_f32(acc[2][1], vb2, v1);
        }
    }
    if constexpr (MR > 2) {
        const float v2 = *a[2];
        acc[0][2] = vfmaq_n_f32(acc[0][2], vb0, v2);
        if constexpr (NG > 1) {
            acc[1][2] = vfmaq_n_f32(acc[1][2], vb1, v2);
        }
        if constexpr (NG > 2) {
            acc[2][2] = vfmaq_n_f32(acc[2][2], vb2, v2);
        }
    }
    if constexpr (MR > 3) {
        const float v3 = *a[3];
        acc[0][3] = vfmaq_n_f32(acc[0][3], vb0, v3);
        if constexpr (NG > 1) {
            acc[1][3] = vfmaq_n_f32(acc[1][3], vb1, v3);
        }
        if constexpr (NG > 2) {
            acc[2][3] = vfmaq_n_f32(acc[2][3], vb2, v3);
        }
    }
    if constexpr (MR > 4) {
        const float v4 = *a[4];
        acc[0][4] = vfmaq_n_f32(acc[0][4], vb0, v4);
        if constexpr (NG > 1) {
            acc[1][4] = vfmaq_n_f32(acc[1][4], vb1, v4);
        }
        if constexpr (NG > 2) {
            acc[2][4] = vfmaq_n_f32(acc[2][4], vb2, v4);
        }
    }
    if constexpr (MR > 5) {
        const float v5 = *a[5];
        acc[0][5] = vfmaq_n_f32(acc[0][5], vb0, v5);
        if constexpr (NG > 1) {
            acc[1][5] = vfmaq_n_f32(acc[1][5], vb1, v5);
        }
        if constexpr (NG > 2) {
            acc[2][5] = vfmaq_n_f32(acc[2][5], vb2, v5);
        }
    }
    if constexpr (MR > 6) {
        const float v6 = *a[6];
        acc[0][6] = vfmaq_n_f32(acc[0][6], vb0, v6);
        if constexpr (NG > 1) {
            acc[1][6] = vfmaq_n_f32(acc[1][6], vb1, v6);
        }
        if constexpr (NG > 2) {
            acc[2][6] = vfmaq_n_f32(acc[2][6], vb2, v6);
        }
    }
    if constexpr (MR > 7) {
        const float v7 = *a[7];
        acc[0][7] = vfmaq_n_f32(acc[0][7], vb0, v7);
        if constexpr (NG > 1) {
            acc[1][7] = vfmaq_n_f32(acc[1][7], vb1, v7);
        }
        if constexpr (NG > 2) {
            acc[2][7] = vfmaq_n_f32(acc[2][7], vb2, v7);
        }
    }
}

/// Advance every A row pointer by `step` elements.
template <int MR>
__attribute__((always_inline)) inline void
f32_advance(const float* (&a)[MR], int step) noexcept
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
f32_mma_rows(const float* const (&A)[MR], const float* B, int kc, int bstep,
             const float* bias, const int (&rr)[MR], float32x4_t (&acc)[NG][MR]) noexcept
{
    f32_col_init<MR, seed>(acc[0], bias, rr, 0);
    if constexpr (NG > 1) {
        f32_col_init<MR, seed>(acc[1], bias, rr, 4);
    }
    if constexpr (NG > 2) {
        f32_col_init<MR, seed>(acc[2], bias, rr, 8);
    }
    const float* a[MR];
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
    const float* b_ptr = B;
    for (int k = 0; k < kc; ++k, b_ptr += bstep) {
        f32_step_rows<MR, NG>(a, b_ptr, acc);
        f32_advance<MR>(a, 1);
    }
}

// ---- the store ------------------------------------------------------------
//
// One column group is written in two passes: every row's C is folded into its
// accumulator first, then the rows are stored. Aliased rows share an address,
// so a row-at-a-time add would have the second duplicate reading C with the
// first one's result already in it.

/// Fold C into one column group's accumulators, every row before any store.
template <int MR>
__attribute__((always_inline)) inline void
f32_add_c(float* const (&row)[MR], int off, float32x4_t (&col)[MR]) noexcept
{
    col[0] = vaddq_f32(col[0], vld1q_f32(row[0] + off));
    if constexpr (MR > 1) {
        col[1] = vaddq_f32(col[1], vld1q_f32(row[1] + off));
    }
    if constexpr (MR > 2) {
        col[2] = vaddq_f32(col[2], vld1q_f32(row[2] + off));
    }
    if constexpr (MR > 3) {
        col[3] = vaddq_f32(col[3], vld1q_f32(row[3] + off));
    }
    if constexpr (MR > 4) {
        col[4] = vaddq_f32(col[4], vld1q_f32(row[4] + off));
    }
    if constexpr (MR > 5) {
        col[5] = vaddq_f32(col[5], vld1q_f32(row[5] + off));
    }
    if constexpr (MR > 6) {
        col[6] = vaddq_f32(col[6], vld1q_f32(row[6] + off));
    }
    if constexpr (MR > 7) {
        col[7] = vaddq_f32(col[7], vld1q_f32(row[7] + off));
    }
}

/// Clamp and store one column group, no C add.
template <int MR>
__attribute__((always_inline)) inline void
f32_store_c(float* const (&row)[MR], int off, const float32x4_t (&col)[MR],
            float32x4_t vmin, float32x4_t vmax) noexcept
{
    vst1q_f32(row[0] + off, vminq_f32(vmaxq_f32(col[0], vmin), vmax));
    if constexpr (MR > 1) {
        vst1q_f32(row[1] + off, vminq_f32(vmaxq_f32(col[1], vmin), vmax));
    }
    if constexpr (MR > 2) {
        vst1q_f32(row[2] + off, vminq_f32(vmaxq_f32(col[2], vmin), vmax));
    }
    if constexpr (MR > 3) {
        vst1q_f32(row[3] + off, vminq_f32(vmaxq_f32(col[3], vmin), vmax));
    }
    if constexpr (MR > 4) {
        vst1q_f32(row[4] + off, vminq_f32(vmaxq_f32(col[4], vmin), vmax));
    }
    if constexpr (MR > 5) {
        vst1q_f32(row[5] + off, vminq_f32(vmaxq_f32(col[5], vmin), vmax));
    }
    if constexpr (MR > 6) {
        vst1q_f32(row[6] + off, vminq_f32(vmaxq_f32(col[6], vmin), vmax));
    }
    if constexpr (MR > 7) {
        vst1q_f32(row[7] + off, vminq_f32(vmaxq_f32(col[7], vmin), vmax));
    }
}

/// 1-3 columns of one row: the C correction is read scalarly (a 16-byte load
/// would read past the tile), added and clamped; nothing is written yet.
template <bool zero_mode>
__attribute__((always_inline)) inline float32x4_t
f32_lane_out(const float* p, int lanes, float32x4_t v, float32x4_t vmin,
             float32x4_t vmax) noexcept
{
    if constexpr (!zero_mode) {
        float32x4_t c = vdupq_n_f32(p[0]);
        if (lanes > 1) {
            c = vsetq_lane_f32(p[1], c, 1);
        }
        if (lanes > 2) {
            c = vsetq_lane_f32(p[2], c, 2);
        }
        v = vaddq_f32(v, c);
    }
    return vminq_f32(vmaxq_f32(v, vmin), vmax);
}

/// Write the live lanes of one row.
__attribute__((always_inline)) inline void
f32_lane_store(float* p, int lanes, float32x4_t v) noexcept
{
    vst1q_lane_f32(p + 0, v, 0);
    if (lanes > 1) {
        vst1q_lane_f32(p + 1, v, 1);
    }
    if (lanes > 2) {
        vst1q_lane_f32(p + 2, v, 2);
    }
}

/// The live lanes of a group: every row's C read happens before any of them is
/// written, so duplicate (aliased) rows all see the same C.
template <int MR, bool zero_mode>
__attribute__((always_inline)) inline void
f32_store_lanes(float* const (&row)[MR], int off, int lanes, const float32x4_t (&col)[MR],
                float32x4_t vmin, float32x4_t vmax) noexcept
{
    float32x4_t v[MR];
    v[0] = f32_lane_out<zero_mode>(row[0] + off, lanes, col[0], vmin, vmax);
    if constexpr (MR > 1) {
        v[1] = f32_lane_out<zero_mode>(row[1] + off, lanes, col[1], vmin, vmax);
    }
    if constexpr (MR > 2) {
        v[2] = f32_lane_out<zero_mode>(row[2] + off, lanes, col[2], vmin, vmax);
    }
    if constexpr (MR > 3) {
        v[3] = f32_lane_out<zero_mode>(row[3] + off, lanes, col[3], vmin, vmax);
    }
    if constexpr (MR > 4) {
        v[4] = f32_lane_out<zero_mode>(row[4] + off, lanes, col[4], vmin, vmax);
    }
    if constexpr (MR > 5) {
        v[5] = f32_lane_out<zero_mode>(row[5] + off, lanes, col[5], vmin, vmax);
    }
    if constexpr (MR > 6) {
        v[6] = f32_lane_out<zero_mode>(row[6] + off, lanes, col[6], vmin, vmax);
    }
    if constexpr (MR > 7) {
        v[7] = f32_lane_out<zero_mode>(row[7] + off, lanes, col[7], vmin, vmax);
    }
    f32_lane_store(row[0] + off, lanes, v[0]);
    if constexpr (MR > 1) {
        f32_lane_store(row[1] + off, lanes, v[1]);
    }
    if constexpr (MR > 2) {
        f32_lane_store(row[2] + off, lanes, v[2]);
    }
    if constexpr (MR > 3) {
        f32_lane_store(row[3] + off, lanes, v[3]);
    }
    if constexpr (MR > 4) {
        f32_lane_store(row[4] + off, lanes, v[4]);
    }
    if constexpr (MR > 5) {
        f32_lane_store(row[5] + off, lanes, v[5]);
    }
    if constexpr (MR > 6) {
        f32_lane_store(row[6] + off, lanes, v[6]);
    }
    if constexpr (MR > 7) {
        f32_lane_store(row[7] + off, lanes, v[7]);
    }
}

/// Write one column group of a panel: MR rows, `live` columns left in the panel
/// from this group's first column.
template <int MR, bool zero_mode>
__attribute__((always_inline)) inline void
f32_store_col(float* const (&row)[MR], int col0, int col, int live,
              float32x4_t (&col_acc)[MR], float32x4_t vmin, float32x4_t vmax) noexcept
{
    const int l = live - col;
    if (l >= 4) {
        if constexpr (!zero_mode) {
            f32_add_c<MR>(row, col0 + col, col_acc);
        }
        f32_store_c<MR>(row, col0 + col, col_acc, vmin, vmax);
    } else if (l > 0) {
        f32_store_lanes<MR, zero_mode>(row, col0 + col, l, col_acc, vmin, vmax);
    }
}

/// Write the live columns of NG groups, starting at `col0`.
template <int MR, int NG, bool zero_mode>
__attribute__((always_inline)) inline void
f32_store(float* const (&row)[MR], int col0, int live, float32x4_t (&acc)[NG][MR],
          float32x4_t vmin, float32x4_t vmax) noexcept
{
    static_assert(NG >= 1 && NG <= 3, "a panel is one to three column groups");
    f32_store_col<MR, zero_mode>(row, col0, 0, live, acc[0], vmin, vmax);
    if constexpr (NG > 1) {
        f32_store_col<MR, zero_mode>(row, col0, 4, live, acc[1], vmin, vmax);
    }
    if constexpr (NG > 2) {
        f32_store_col<MR, zero_mode>(row, col0, 8, live, acc[2], vmin, vmax);
    }
}

/// One narrow remainder panel through NG groups, packed A.
template <int MR, int NG, bool zero_mode, int seed>
__attribute__((always_inline)) inline void
f32_tail(float* const (&row)[MR], int n, int live, int kc, int bstep, const float* A,
         const float* B, const float* bias, const int (&rr)[MR], float32x4_t vmin,
         float32x4_t vmax) noexcept
{
    float32x4_t acc[NG][MR];
    f32_mma<MR, NG, seed>(A, B, kc, bstep, bias, rr, acc);
    f32_store<MR, NG, zero_mode>(row, n, live, acc, vmin, vmax);
}

/// One narrow remainder panel through NG groups, row-major A.
template <int MR, int NG, bool zero_mode, int seed>
__attribute__((always_inline)) inline void
f32_tail_rows(float* const (&row)[MR], int n, int live, int kc, int bstep,
              const float* const (&a)[MR], const float* B, const float* bias,
              const int (&rr)[MR], float32x4_t vmin, float32x4_t vmax) noexcept
{
    float32x4_t acc[NG][MR];
    f32_mma_rows<MR, NG, seed>(a, B, kc, bstep, bias, rr, acc);
    f32_store<MR, NG, zero_mode>(row, n, live, acc, vmin, vmax);
}

// =========================================================================
//  packed A + packed B  (MR = 8, 4, 1)
// =========================================================================

/// One packed-A rung: the whole N side into C.
template <int MR, bool zero_mode = false, int seed = 0>
inline void mma_pack_mr_nr12_mrxncxkc_f32(int mr, int nc, int kc, float* NNOPS_RESTRICT C,
                                          int ldc, const float* NNOPS_RESTRICT A,
                                          const float* NNOPS_RESTRICT B, float clamp_min,
                                          float clamp_max,
                                          const float* NNOPS_RESTRICT bias = nullptr) noexcept
{
    int rr[MR];
    f32_row_alias<MR>(mr, rr);
    float* row[MR];
    f32_row_ptrs<MR>(C, ldc, rr, row);
    const int ldb_panel = static_cast<int>(mma_pack_align(12 * kc * sizeof(float)) / sizeof(float));
    const float32x4_t vmin = vdupq_n_f32(clamp_min);
    const float32x4_t vmax = vdupq_n_f32(clamp_max);

    int n = 0;
    for (; n + 12 <= nc; n += 12) {
        float32x4_t acc[3][MR];
        f32_mma<MR, 3, seed>(A, B, kc, 12, f32_panel_bias<seed>(bias, n), rr, acc);
        f32_store<MR, 3, zero_mode>(row, n, 12, acc, vmin, vmax);
        B += ldb_panel;
    }

    // The live remainder narrows to the groups it needs: 4, 8 or 12 columns.
    const int live = nc - n;
    if (live > 8) {
        f32_tail<MR, 3, zero_mode, seed>(row, n, live, kc, 12, A, B,
                                         f32_panel_bias<seed>(bias, n), rr, vmin, vmax);
    } else if (live > 4) {
        f32_tail<MR, 2, zero_mode, seed>(row, n, live, kc, 12, A, B,
                                         f32_panel_bias<seed>(bias, n), rr, vmin, vmax);
    } else if (live > 0) {
        f32_tail<MR, 1, zero_mode, seed>(row, n, live, kc, 12, A, B,
                                         f32_panel_bias<seed>(bias, n), rr, vmin, vmax);
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
inline void mma_direct_mr_nr12_mrxncxkc_f32(int mr, int nc, int kc, float* NNOPS_RESTRICT C,
                                            int ldc, const float* NNOPS_RESTRICT A, int lda,
                                            const float* NNOPS_RESTRICT B, int ldb,
                                            float clamp_min, float clamp_max,
                                            const float* NNOPS_RESTRICT bias = nullptr) noexcept
{
    int rr[MR];
    f32_row_alias<MR>(mr, rr);
    float* row[MR];
    f32_row_ptrs<MR>(C, ldc, rr, row);
    const float* a0[MR];
    f32_a_ptrs<MR>(A, lda, rr, a0);
    const int ldb_panel = static_cast<int>(mma_pack_align(12 * kc * sizeof(float)) / sizeof(float));
    const int bstep = (ldb < 0) ? 12 : ldb;
    int panel_step = ldb_panel;
    if constexpr (MR == 1) {
        // Below -1 a negative ldb is how far apart the caller's packed panels
        // sit; without one the panels are this call's own and the slot is right.
        if (ldb < -1) {
            panel_step = -ldb;
        }
    }
    const float32x4_t vmin = vdupq_n_f32(clamp_min);
    const float32x4_t vmax = vdupq_n_f32(clamp_max);

    int n = 0;
    if constexpr (MR == 1) {
        // Two packed panels per pass: at one row the loop is nothing but weight
        // stream, and its scalar A load costs a whole load slot of its own, so
        // one scalar over 24 columns is the margin.
        for (; ldb < 0 && n <= nc - 24; n += 24) {
            const float* const bp = f32_panel_bias<seed>(bias, n);
            float32x4_t acc[6][1];
            f32_col_init<1, seed>(acc[0], bp, rr, 0);
            f32_col_init<1, seed>(acc[1], bp, rr, 4);
            f32_col_init<1, seed>(acc[2], bp, rr, 8);
            f32_col_init<1, seed>(acc[3], bp, rr, 12);
            f32_col_init<1, seed>(acc[4], bp, rr, 16);
            f32_col_init<1, seed>(acc[5], bp, rr, 20);
            const float* ap = A;
            const float* b0_ptr = B;
            const float* b1_ptr = B + panel_step;
            for (int k = 0; k < kc; ++k) {
                const float32x4_t d0 = vld1q_f32(b0_ptr + 0 * 4);
                const float32x4_t d1 = vld1q_f32(b0_ptr + 1 * 4);
                const float32x4_t d2 = vld1q_f32(b0_ptr + 2 * 4);
                const float32x4_t d3 = vld1q_f32(b1_ptr + 0 * 4);
                const float32x4_t d4 = vld1q_f32(b1_ptr + 1 * 4);
                const float32x4_t d5 = vld1q_f32(b1_ptr + 2 * 4);
                b0_ptr += bstep;
                b1_ptr += bstep;
                const float a = *ap++;
                f32_col_n<1>(a, d0, acc[0]);
                f32_col_n<1>(a, d1, acc[1]);
                f32_col_n<1>(a, d2, acc[2]);
                f32_col_n<1>(a, d3, acc[3]);
                f32_col_n<1>(a, d4, acc[4]);
                f32_col_n<1>(a, d5, acc[5]);
            }
            f32_store_col<1, zero_mode>(row, n, 0, 24, acc[0], vmin, vmax);
            f32_store_col<1, zero_mode>(row, n, 4, 24, acc[1], vmin, vmax);
            f32_store_col<1, zero_mode>(row, n, 8, 24, acc[2], vmin, vmax);
            f32_store_col<1, zero_mode>(row, n, 12, 24, acc[3], vmin, vmax);
            f32_store_col<1, zero_mode>(row, n, 16, 24, acc[4], vmin, vmax);
            f32_store_col<1, zero_mode>(row, n, 20, 24, acc[5], vmin, vmax);
            B += 2 * panel_step;
        }
    }

    for (; n + 12 <= nc; n += 12) {
        float32x4_t acc[3][MR];
        f32_mma_rows<MR, 3, seed>(a0, B, kc, bstep, f32_panel_bias<seed>(bias, n), rr, acc);
        f32_store<MR, 3, zero_mode>(row, n, 12, acc, vmin, vmax);
        B += panel_step;
    }

    // The live remainder narrows to the groups it needs, as the packed rung's
    // does; only a full panel pays for all three.
    const int live = nc - n;
    if (live > 8) {
        f32_tail_rows<MR, 3, zero_mode, seed>(row, n, live, kc, bstep, a0, B,
                                              f32_panel_bias<seed>(bias, n), rr, vmin, vmax);
    } else if (live > 4) {
        f32_tail_rows<MR, 2, zero_mode, seed>(row, n, live, kc, bstep, a0, B,
                                              f32_panel_bias<seed>(bias, n), rr, vmin, vmax);
    } else if (live > 0) {
        f32_tail_rows<MR, 1, zero_mode, seed>(row, n, live, kc, bstep, a0, B,
                                              f32_panel_bias<seed>(bias, n), rr, vmin, vmax);
    }
}

// ---- dispatch tables, consumed by matmul_helper.cpp ---------------------

template <bool Z, int Seed = 0>
inline constexpr auto mma_pack_fns_f32 =
    std::array{&mma_pack_mr_nr12_mrxncxkc_f32<8, Z, Seed>,
               &mma_pack_mr_nr12_mrxncxkc_f32<4, Z, Seed>,
               &mma_pack_mr_nr12_mrxncxkc_f32<1, Z, Seed>};

template <bool Z, int Seed = 0>
inline constexpr auto mma_direct_fns_f32 =
    std::array{&mma_direct_mr_nr12_mrxncxkc_f32<8, Z, Seed>,
               &mma_direct_mr_nr12_mrxncxkc_f32<4, Z, Seed>,
               &mma_direct_mr_nr12_mrxncxkc_f32<1, Z, Seed>};

} // namespace nnops::backend::cpu::aarch64
