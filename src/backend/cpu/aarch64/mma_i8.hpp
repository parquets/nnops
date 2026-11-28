#pragma once
/// @file mma_i8.hpp
/// @brief AArch64 NEON int8 (SDOT) mrxncxkc MMA kernels: s8×s8 → int32.
///
/// One call covers the whole N side of one M panel: C[mr][nc] += A[mr][kc] ×
/// B[kc][nc] panel by panel, and every int32 result is clamped to
/// [clamp_min, clamp_max]. The int8 sibling of mma_f32.hpp / mma_f16.hpp, with
/// the same runtime panel height `mr` and the same partial-M convention: rows
/// past `mr` reuse the last live row's index in A and C, so a short panel needs
/// no per-row guards: the tail stores the duplicate rows back to the same
/// address.
///
/// Two A presentations share the same packed B:
///   - packed A (mma_pack_mr{8,4,1}_nr12_*): four-byte k groups, one group
///     per row, [ceil(kc/4)][MR][4] bytes (32 per group at MR=8) — exactly what
///     pack_trans_nN_dp4a_i8 writes. A 16-byte load holds four rows' groups, so
///     one vector feeds four vdotq_laneq lanes.
///   - row-major A (mma_direct_mr{8,4,1}_nr12_*): read through `lda` (int8
///     elements) and each row's group broadcast with vdupq_n_s32, so they
///     require kc % 4 == 0 — a row-major row has no zero-filled final group.
///
/// B is always in the dot-product packers' layout (pack_trans_n12_dp4a_i8 /
/// pack_copy_n12_dp4a_i8), uniform across panels: ceil(nc/12) panels of 12
/// columns each, [ceil(kc/4)][12][4] bytes — 48 bytes per k group — and a
/// panel of 12*kbytes bytes sits in a mma_pack_align(12*kbytes)-byte slot.
/// Twelve is the widest panel an 8-row block keeps in 32 vector registers (24
/// accumulators + 3 B + one A broadcast stay live, 28 in all, nothing spills)
/// and is mma_i8_nr_max on this arch.
///
/// `ldb` (0 = that slot), the B-side argument that follows the operand it
/// strides, overrides the panel-to-panel step in bytes, for a prepacked B whose
/// panels are pitched for another layout — e.g. one cut into k-block segments.
/// Nothing inside a call moves, so a segment reads exactly like the panel a
/// per-call pack would have written.
///
/// Only a panel's live columns are stored, so the columns past `nc` never reach
/// C — the packers zero-fill them. The same goes for the packed-A panel's row
/// slots past `mr`: the kernels take their SDOT lanes straight from the panel,
/// so a short panel's dead slots must repeat the last live row (the pack_trans_*
/// convention that mma_f32.hpp relies on) — they are not masked off in here.
///
/// Arithmetic is signed × signed (vdotq_s32 / vdotq_laneq_s32), the contract
/// x86_64/mma_i8.hpp also presents: both take int8 A and int8 B and return the
/// exact s8×s8 product, so the MatMul driver is arch-blind. x86 reaches the
/// same answer through u8×s8 plus a bias correction; SDOT has no unsigned
/// restriction, so there is nothing to correct here and mma_i8_b_panel_sums
/// below is a no-op.
///
/// A uint8 A is shifted into signed range by the A pack (XOR 0x80, i.e. a-128)
/// rather than by anything in here: SDOT then accumulates Σ(a-128)·b, and the
/// 128·Σb the shift introduced is removed once in the epilogue, which already
/// has the full-K column sums (`matmul_int8_compensate`'s u8_offset). That is
/// why this arch has no panel tail — see mma_i8_folds_a_bias.
///
/// `kc` is the number of int8 elements. On the packed route a final partial
/// group (1..3) is consumed as a whole zero-padded group, matching the packers.
///
/// Tile sizes (AArch64-optimised): M ∈ {8, 4, 1}    N: 12 plus an in-kernel
/// tail. The dispatch tables at the bottom are indexed by the MR levels
/// {8,4,1}; because every kernel takes the runtime panel height, a remainder
/// panel below its own MR runs through the level above it.

#include <array>
#include <algorithm>
#include <arm_neon.h>
#include <cstdint>
#include "backend/cpu/aarch64/i8_requant_store.hpp"
#include "backend/cpu/common/common_utils.hpp"

namespace nnops::backend::cpu::aarch64 {

/// Packed-B panel width: the kernels' uniform nr.
inline constexpr int mma_i8_nr_max = 12;

/// Bytes a packed-B panel carries past its nr*kbytes payload; SDOT needs none.
inline constexpr int mma_i8_b_panel_tail = 0;

/// A packed-B panel's byte pitch: the caller's `ldb` when it gave one (a
/// prepacked blob's own), else the uniform slot `k` int8 k-values take.
inline constexpr int mma_i8_panel_stride(int ldb, int k) noexcept
{
    return (ldb > 0) ? ldb : static_cast<int>(mma_pack_align(12 * k));
}

/// False: a shifted A's correction stays with the epilogue, not the kernel.
inline constexpr bool mma_i8_folds_a_bias = false;

/// No-op: SDOT needs no column sums. Same signature as x86 so matmul_helper is arch-blind.
inline void mma_i8_b_panel_sums(int8_t* /*panel*/, int /*kbytes*/, bool /*a_biased*/) noexcept {}

// =========================================================================
//  one walk: the group, the K loop, the stores
// =========================================================================
//
// Every int8 rung — int32 accumulators or requantized, packed A or row-major —
// is the same walk: one k group of MR rows against NG four-column groups, then
// a store. The accumulators are laid out [NG][MR] so a column is contiguous,
// and every index into them (and into the per-column scale and bias) is a
// literal. A loop to a template bound costs nothing where the compiler unrolls
// it and everything where it does not: the array is then indexed at run time
// and the whole block moves to the stack.

/// One four-column group: MR rows (as one or two 16-byte A registers) against
/// one B vector, into that column's MR accumulators.
template <int MR>
__attribute__((always_inline)) inline void
i8_sdot_col(int8x16_t va0, int8x16_t va1, int8x16_t vb, int32x4_t (&col)[MR]) noexcept
{
    if constexpr (MR == 1) {
        col[0] = vdotq_s32(col[0], vb, va0);
    } else if constexpr (MR == 4) {
        col[0] = vdotq_laneq_s32(col[0], vb, va0, 0);
        col[1] = vdotq_laneq_s32(col[1], vb, va0, 1);
        col[2] = vdotq_laneq_s32(col[2], vb, va0, 2);
        col[3] = vdotq_laneq_s32(col[3], vb, va0, 3);
    } else {
        static_assert(MR == 8, "an int8 A panel is 1, 4 or 8 rows");
        col[0] = vdotq_laneq_s32(col[0], vb, va0, 0);
        col[1] = vdotq_laneq_s32(col[1], vb, va0, 1);
        col[2] = vdotq_laneq_s32(col[2], vb, va0, 2);
        col[3] = vdotq_laneq_s32(col[3], vb, va0, 3);
        col[4] = vdotq_laneq_s32(col[4], vb, va1, 0);
        col[5] = vdotq_laneq_s32(col[5], vb, va1, 1);
        col[6] = vdotq_laneq_s32(col[6], vb, va1, 2);
        col[7] = vdotq_laneq_s32(col[7], vb, va1, 3);
    }
}

/// Zero one column's accumulators.
template <int MR>
__attribute__((always_inline)) inline void
i8_sdot_col_zero(int32x4_t (&col)[MR]) noexcept
{
    const int32x4_t vz = vdupq_n_s32(0);
    col[0] = vz;
    if constexpr (MR > 1) {
        col[1] = vz;
    }
    if constexpr (MR > 2) {
        col[2] = vz;
    }
    if constexpr (MR > 3) {
        col[3] = vz;
    }
    if constexpr (MR > 4) {
        col[4] = vz;
    }
    if constexpr (MR > 5) {
        col[5] = vz;
    }
    if constexpr (MR > 6) {
        col[6] = vz;
    }
    if constexpr (MR > 7) {
        col[7] = vz;
    }
}

/// One k group of a packed-A panel: MR rows against NG four-column groups.
template <int MR, int NG>
__attribute__((always_inline)) inline void
i8_sdot_group(const int8_t*& a_ptr, const int8_t* b_ptr, int32x4_t (&acc)[NG][MR]) noexcept
{
    static_assert(NG >= 1 && NG <= 3, "a panel is one to three column groups");
    int8x16_t va0;
    int8x16_t va1;
    if constexpr (MR == 1) {
        // The row's four bytes are splatted into every lane; each B lane is a
        // column, so one plain dot covers the group.
        va0 = vreinterpretq_s8_s32(vdupq_n_s32(*reinterpret_cast<const int32_t*>(a_ptr)));
        va1 = va0;
        a_ptr += 4;
    } else if constexpr (MR == 4) {
        va0 = vld1q_s8(a_ptr);
        va1 = va0;
        a_ptr += 16;
    } else {
        static_assert(MR == 8, "an int8 A panel is 1, 4 or 8 rows");
        va0 = vld1q_s8(a_ptr + 0 * 16);
        va1 = vld1q_s8(a_ptr + 1 * 16);
        a_ptr += 32;
    }
    i8_sdot_col<MR>(va0, va1, vld1q_s8(b_ptr + 0 * 16), acc[0]);
    if constexpr (NG > 1) {
        i8_sdot_col<MR>(va0, va1, vld1q_s8(b_ptr + 1 * 16), acc[1]);
    }
    if constexpr (NG > 2) {
        i8_sdot_col<MR>(va0, va1, vld1q_s8(b_ptr + 2 * 16), acc[2]);
    }
}

/// The panel's whole K into a zeroed `acc`.
template <int MR, int NG>
__attribute__((always_inline)) inline void
i8_sdot_mma(const int8_t* A, const int8_t* B, int kg, int32x4_t (&acc)[NG][MR]) noexcept
{
    i8_sdot_col_zero<MR>(acc[0]);
    if constexpr (NG > 1) {
        i8_sdot_col_zero<MR>(acc[1]);
    }
    if constexpr (NG > 2) {
        i8_sdot_col_zero<MR>(acc[2]);
    }
    const int8_t* a_ptr = A;
    const int8_t* b_ptr = B;
    for (int k = 0; k < kg; ++k, b_ptr += 48) {
        i8_sdot_group<MR, NG>(a_ptr, b_ptr, acc);
    }
}

/// The output rows, aliasing the last live one past `mr` (the ladder emits a
/// rung at its own height, so in practice `mr == MR`).
template <int MR, class T>
__attribute__((always_inline)) inline void
i8_sdot_row_ptrs(T* C, int64_t ldm, int mr, T* (&row)[MR]) noexcept
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

/// Advance every row pointer by `step` elements.
template <int MR, class T>
__attribute__((always_inline)) inline void
i8_sdot_advance(T* (&row)[MR], int64_t step) noexcept
{
    row[0] += step;
    if constexpr (MR > 1) {
        row[1] += step;
    }
    if constexpr (MR > 2) {
        row[2] += step;
    }
    if constexpr (MR > 3) {
        row[3] += step;
    }
    if constexpr (MR > 4) {
        row[4] += step;
    }
    if constexpr (MR > 5) {
        row[5] += step;
    }
    if constexpr (MR > 6) {
        row[6] += step;
    }
    if constexpr (MR > 7) {
        row[7] += step;
    }
}

// ---- the int32 store ------------------------------------------------------

/// Four int32 columns of one row: add C unless zero-mode, clamp, store.
template <bool zero_mode>
__attribute__((always_inline)) inline void
i8_store4q_i32(int32_t* p, int32x4_t v, int32x4_t vmin, int32x4_t vmax) noexcept
{
    if constexpr (!zero_mode) {
        v = vaddq_s32(v, vld1q_s32(p));
    }
    vst1q_s32(p, vminq_s32(vmaxq_s32(v, vmin), vmax));
}

/// 1-3 int32 columns: the C correction is read scalarly (a 16-byte load would
/// read past the tile) and written back with lane stores.
template <bool zero_mode>
__attribute__((always_inline)) inline void
i8_store4q_i32_lanes(int32_t* p, int lanes, int32x4_t v, int32x4_t vmin,
                     int32x4_t vmax) noexcept
{
    if constexpr (!zero_mode) {
        int32x4_t c = vdupq_n_s32(p[0]);
        if (lanes > 1) {
            c = vsetq_lane_s32(p[1], c, 1);
        }
        if (lanes > 2) {
            c = vsetq_lane_s32(p[2], c, 2);
        }
        v = vaddq_s32(v, c);
    }
    v = vminq_s32(vmaxq_s32(v, vmin), vmax);
    vst1q_lane_s32(p + 0, v, 0);
    if (lanes > 1) {
        vst1q_lane_s32(p + 1, v, 1);
    }
    if (lanes > 2) {
        vst1q_lane_s32(p + 2, v, 2);
    }
}

/// Write one column group of an int32 panel: MR rows, `live` columns left in
/// the panel from this group's first column.
template <int MR, bool zero_mode>
__attribute__((always_inline)) inline void
i8_sdot_store_col_i32(int32_t* const (&row)[MR], int col0, int col, int live,
                      const int32x4_t (&col_acc)[MR], int32x4_t vmin,
                      int32x4_t vmax) noexcept
{
    const int l = live - col;
    if (l >= 4) {
        i8_store4q_i32<zero_mode>(row[0] + col0 + col, col_acc[0], vmin, vmax);
        if constexpr (MR > 1) {
            i8_store4q_i32<zero_mode>(row[1] + col0 + col, col_acc[1], vmin, vmax);
        }
        if constexpr (MR > 2) {
            i8_store4q_i32<zero_mode>(row[2] + col0 + col, col_acc[2], vmin, vmax);
        }
        if constexpr (MR > 3) {
            i8_store4q_i32<zero_mode>(row[3] + col0 + col, col_acc[3], vmin, vmax);
        }
        if constexpr (MR > 4) {
            i8_store4q_i32<zero_mode>(row[4] + col0 + col, col_acc[4], vmin, vmax);
        }
        if constexpr (MR > 5) {
            i8_store4q_i32<zero_mode>(row[5] + col0 + col, col_acc[5], vmin, vmax);
        }
        if constexpr (MR > 6) {
            i8_store4q_i32<zero_mode>(row[6] + col0 + col, col_acc[6], vmin, vmax);
        }
        if constexpr (MR > 7) {
            i8_store4q_i32<zero_mode>(row[7] + col0 + col, col_acc[7], vmin, vmax);
        }
    } else if (l > 0) {
        i8_store4q_i32_lanes<zero_mode>(row[0] + col0 + col, l, col_acc[0], vmin, vmax);
        if constexpr (MR > 1) {
            i8_store4q_i32_lanes<zero_mode>(row[1] + col0 + col, l, col_acc[1], vmin, vmax);
        }
        if constexpr (MR > 2) {
            i8_store4q_i32_lanes<zero_mode>(row[2] + col0 + col, l, col_acc[2], vmin, vmax);
        }
        if constexpr (MR > 3) {
            i8_store4q_i32_lanes<zero_mode>(row[3] + col0 + col, l, col_acc[3], vmin, vmax);
        }
        if constexpr (MR > 4) {
            i8_store4q_i32_lanes<zero_mode>(row[4] + col0 + col, l, col_acc[4], vmin, vmax);
        }
        if constexpr (MR > 5) {
            i8_store4q_i32_lanes<zero_mode>(row[5] + col0 + col, l, col_acc[5], vmin, vmax);
        }
        if constexpr (MR > 6) {
            i8_store4q_i32_lanes<zero_mode>(row[6] + col0 + col, l, col_acc[6], vmin, vmax);
        }
        if constexpr (MR > 7) {
            i8_store4q_i32_lanes<zero_mode>(row[7] + col0 + col, l, col_acc[7], vmin, vmax);
        }
    }
}

/// Write the live columns of NG groups, starting at `col0`.
template <int MR, int NG, bool zero_mode>
__attribute__((always_inline)) inline void
i8_sdot_store_i32(int32_t* const (&row)[MR], int col0, int live, int32x4_t (&acc)[NG][MR],
                  int32x4_t vmin, int32x4_t vmax) noexcept
{
    static_assert(NG >= 1 && NG <= 3, "a panel is one to three column groups");
    i8_sdot_store_col_i32<MR, zero_mode>(row, col0, 0, live, acc[0], vmin, vmax);
    if constexpr (NG > 1) {
        i8_sdot_store_col_i32<MR, zero_mode>(row, col0, 4, live, acc[1], vmin, vmax);
    }
    if constexpr (NG > 2) {
        i8_sdot_store_col_i32<MR, zero_mode>(row, col0, 8, live, acc[2], vmin, vmax);
    }
}

/// One narrow remainder panel through NG groups, int32 destination.
template <int MR, int NG, bool zero_mode>
__attribute__((always_inline)) inline void
i8_sdot_tail_i32(int32_t* const (&row)[MR], int n, int live, int kg, const int8_t* A,
                 const int8_t* B, int32x4_t vmin, int32x4_t vmax) noexcept
{
    int32x4_t acc[NG][MR];
    i8_sdot_mma<MR, NG>(A, B, kg, acc);
    i8_sdot_store_i32<MR, NG, zero_mode>(row, n, live, acc, vmin, vmax);
}

// =========================================================================
//  packed A + packed B  (mr = 8, 4, 1)
// =========================================================================

/// One packed-A rung: the whole N side into int32 C.
template <int MR, bool zero_mode>
inline void mma_pack_mr_nr12_mrxncxkc_i8(int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C,
                                         int ldc, const int8_t* NNOPS_RESTRICT A,
                                         const int8_t* NNOPS_RESTRICT B, int ldb,
                                         int32_t clamp_min, int32_t clamp_max) noexcept
{
    int32_t* row[MR];
    i8_sdot_row_ptrs<MR, int32_t>(C, ldc, mr, row);
    const int kbytes = (kc + 3) & ~3;
    const int kg = kbytes >> 2;
    const int ldb_panel = mma_i8_panel_stride(ldb, kbytes);
    const int32x4_t vmin = vdupq_n_s32(clamp_min);
    const int32x4_t vmax = vdupq_n_s32(clamp_max);

    int n = 0;
    for (; n + mma_i8_nr_max <= nc; n += mma_i8_nr_max) {
        int32x4_t acc[3][MR];
        i8_sdot_mma<MR, 3>(A, B, kg, acc);
        i8_sdot_store_i32<MR, 3, zero_mode>(row, n, 12, acc, vmin, vmax);
        B += ldb_panel;
    }

    // The live remainder narrows to the groups it needs: 4, 8 or 12 columns.
    const int live = nc - n;
    if (live > 8) {
        i8_sdot_tail_i32<MR, 3, zero_mode>(row, n, live, kg, A, B, vmin, vmax);
    } else if (live > 4) {
        i8_sdot_tail_i32<MR, 2, zero_mode>(row, n, live, kg, A, B, vmin, vmax);
    } else if (live > 0) {
        i8_sdot_tail_i32<MR, 1, zero_mode>(row, n, live, kg, A, B, vmin, vmax);
    }
}

// =========================================================================
//  raw A + packed B  (mr = 8, 4, 1) — kc % 4 == 0
// =========================================================================
//
// The same walk with each row's four k bytes broadcast instead of a shared
// 16-byte panel load, so the rung reads A through `lda`. The A fragment of the
// next group is loaded at the end of the k loop body, which keeps the MMA at
// the top of an iteration off the load's latency.

/// The rung's A rows, aliasing the last live one past `mr`.
template <int MR>
__attribute__((always_inline)) inline void
i8_sdot_a_ptrs(const int8_t* A, int lda, int mr, const int8_t* (&a)[MR]) noexcept
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

/// One k group of a row-major panel: MR rows against NG column groups. The
/// row's four bytes are splatted per group — this route has no shared 16-byte A
/// load to amortise them from — and nothing but the accumulators is live across
/// the K loop.
template <int MR, int NG>
__attribute__((always_inline)) inline void
i8_sdot_step_rows(const int8_t* const (&a)[MR], const int8_t* b_ptr,
                  int32x4_t (&acc)[NG][MR]) noexcept
{
    static_assert(NG >= 1 && NG <= 3, "a panel is one to three column groups");
    const int8x16_t vb0 = vld1q_s8(b_ptr + 0 * 16);
    int8x16_t vb1 = vb0;
    int8x16_t vb2 = vb0;
    if constexpr (NG > 1) {
        vb1 = vld1q_s8(b_ptr + 1 * 16);
    }
    if constexpr (NG > 2) {
        vb2 = vld1q_s8(b_ptr + 2 * 16);
    }
    const auto bcast = [](const int8_t* p) {
        return vreinterpretq_s8_s32(vdupq_n_s32(*reinterpret_cast<const int32_t*>(p)));
    };
    const int8x16_t va0 = bcast(a[0]);
    acc[0][0] = vdotq_s32(acc[0][0], vb0, va0);
    if constexpr (NG > 1) {
        acc[1][0] = vdotq_s32(acc[1][0], vb1, va0);
    }
    if constexpr (NG > 2) {
        acc[2][0] = vdotq_s32(acc[2][0], vb2, va0);
    }
    if constexpr (MR > 1) {
        const int8x16_t va = bcast(a[1]);
        acc[0][1] = vdotq_s32(acc[0][1], vb0, va);
        if constexpr (NG > 1) {
            acc[1][1] = vdotq_s32(acc[1][1], vb1, va);
        }
        if constexpr (NG > 2) {
            acc[2][1] = vdotq_s32(acc[2][1], vb2, va);
        }
    }
    if constexpr (MR > 2) {
        const int8x16_t va = bcast(a[2]);
        acc[0][2] = vdotq_s32(acc[0][2], vb0, va);
        if constexpr (NG > 1) {
            acc[1][2] = vdotq_s32(acc[1][2], vb1, va);
        }
        if constexpr (NG > 2) {
            acc[2][2] = vdotq_s32(acc[2][2], vb2, va);
        }
    }
    if constexpr (MR > 3) {
        const int8x16_t va = bcast(a[3]);
        acc[0][3] = vdotq_s32(acc[0][3], vb0, va);
        if constexpr (NG > 1) {
            acc[1][3] = vdotq_s32(acc[1][3], vb1, va);
        }
        if constexpr (NG > 2) {
            acc[2][3] = vdotq_s32(acc[2][3], vb2, va);
        }
    }
    if constexpr (MR > 4) {
        const int8x16_t va = bcast(a[4]);
        acc[0][4] = vdotq_s32(acc[0][4], vb0, va);
        if constexpr (NG > 1) {
            acc[1][4] = vdotq_s32(acc[1][4], vb1, va);
        }
        if constexpr (NG > 2) {
            acc[2][4] = vdotq_s32(acc[2][4], vb2, va);
        }
    }
    if constexpr (MR > 5) {
        const int8x16_t va = bcast(a[5]);
        acc[0][5] = vdotq_s32(acc[0][5], vb0, va);
        if constexpr (NG > 1) {
            acc[1][5] = vdotq_s32(acc[1][5], vb1, va);
        }
        if constexpr (NG > 2) {
            acc[2][5] = vdotq_s32(acc[2][5], vb2, va);
        }
    }
    if constexpr (MR > 6) {
        const int8x16_t va = bcast(a[6]);
        acc[0][6] = vdotq_s32(acc[0][6], vb0, va);
        if constexpr (NG > 1) {
            acc[1][6] = vdotq_s32(acc[1][6], vb1, va);
        }
        if constexpr (NG > 2) {
            acc[2][6] = vdotq_s32(acc[2][6], vb2, va);
        }
    }
    if constexpr (MR > 7) {
        const int8x16_t va = bcast(a[7]);
        acc[0][7] = vdotq_s32(acc[0][7], vb0, va);
        if constexpr (NG > 1) {
            acc[1][7] = vdotq_s32(acc[1][7], vb1, va);
        }
        if constexpr (NG > 2) {
            acc[2][7] = vdotq_s32(acc[2][7], vb2, va);
        }
    }
}

/// The panel's whole K from row-major rows into zeroed `acc`.
template <int MR, int NG>
__attribute__((always_inline)) inline void
i8_sdot_mma_rows(const int8_t* const (&A)[MR], const int8_t* B, int kg,
                 int32x4_t (&acc)[NG][MR]) noexcept
{
    i8_sdot_col_zero<MR>(acc[0]);
    if constexpr (NG > 1) {
        i8_sdot_col_zero<MR>(acc[1]);
    }
    if constexpr (NG > 2) {
        i8_sdot_col_zero<MR>(acc[2]);
    }
    const int8_t* a[MR];
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
    const int8_t* b_ptr = B;
    for (int k = 0; k < kg; ++k, b_ptr += 48) {
        i8_sdot_step_rows<MR, NG>(a, b_ptr, acc);
        i8_sdot_advance<MR>(a, 4);
    }
}

/// One row-major-A rung: the whole N side into int32 C.
template <int MR, bool zero_mode>
inline void mma_direct_mr_nr12_mrxncxkc_i8(int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C,
                                           int ldc, const int8_t* NNOPS_RESTRICT A, int lda,
                                           const int8_t* NNOPS_RESTRICT B, int ldb,
                                           int32_t clamp_min, int32_t clamp_max) noexcept
{
    int32_t* row[MR];
    i8_sdot_row_ptrs<MR, int32_t>(C, ldc, mr, row);
    const int8_t* a0[MR];
    i8_sdot_a_ptrs<MR>(A, lda, mr, a0);
    const int kg = kc >> 2;
    const int ldb_panel = mma_i8_panel_stride(ldb, kc);
    const int32x4_t vmin = vdupq_n_s32(clamp_min);
    const int32x4_t vmax = vdupq_n_s32(clamp_max);

    int n = 0;
    for (; n + mma_i8_nr_max <= nc; n += mma_i8_nr_max) {
        int32x4_t acc[3][MR];
        i8_sdot_mma_rows<MR, 3>(a0, B, kg, acc);
        i8_sdot_store_i32<MR, 3, zero_mode>(row, n, 12, acc, vmin, vmax);
        B += ldb_panel;
    }

    // The live remainder narrows to the groups it needs, as the packed rung's
    // does; only a full panel pays for all three.
    const int live = nc - n;
    if (live > 8) {
        int32x4_t acc[3][MR];
        i8_sdot_mma_rows<MR, 3>(a0, B, kg, acc);
        i8_sdot_store_i32<MR, 3, zero_mode>(row, n, live, acc, vmin, vmax);
    } else if (live > 4) {
        int32x4_t acc[2][MR];
        i8_sdot_mma_rows<MR, 2>(a0, B, kg, acc);
        i8_sdot_store_i32<MR, 2, zero_mode>(row, n, live, acc, vmin, vmax);
    } else if (live > 0) {
        int32x4_t acc[1][MR];
        i8_sdot_mma_rows<MR, 1>(a0, B, kg, acc);
        i8_sdot_store_i32<MR, 1, zero_mode>(row, n, live, acc, vmin, vmax);
    }
}

// =========================================================================
//  the requantizing variants (packed A + packed B)
// =========================================================================
//
// SDOT's accumulators are already one row's four columns, so the store is a
// straight requantize — no transpose step. One call covers the whole K and the
// accumulators go to the 8-bit destination; no int32 tile reaches memory.
//
// Every index into the accumulator, scale and bias blocks is a literal: the
// group walk and the store are written out at the panel's three widths instead
// of looping to a template bound. A loop costs nothing where the compiler
// unrolls it and everything where it does not — the array is then indexed at
// run time and the whole block moves to the stack (GCC 14 does exactly that on
// the narrow path, and turns the ragged byte copy into a memcpy call).

/// Requantize and write one group of a panel: MR rows, `live` columns left in
/// the panel from this group's first column.
template <int MR, class TOut>
__attribute__((always_inline)) inline void
i8_sdot_store_col(TOut* const (&row)[MR], int col0, int col, int live,
                  const int32x4_t (&col_acc)[MR], float32x4_t sc, int32x4_t bi, int32x4_t vlo,
                  int32x4_t vzp) noexcept
{
    const int l = live - col;
    if (l >= 4) {
        i8_store4q<TOut>(row[0] + col0 + col, col_acc[0], sc, bi, vlo, vzp);
        if constexpr (MR > 1) {
            i8_store4q<TOut>(row[1] + col0 + col, col_acc[1], sc, bi, vlo, vzp);
        }
        if constexpr (MR > 2) {
            i8_store4q<TOut>(row[2] + col0 + col, col_acc[2], sc, bi, vlo, vzp);
        }
        if constexpr (MR > 3) {
            i8_store4q<TOut>(row[3] + col0 + col, col_acc[3], sc, bi, vlo, vzp);
        }
        if constexpr (MR > 4) {
            i8_store4q<TOut>(row[4] + col0 + col, col_acc[4], sc, bi, vlo, vzp);
        }
        if constexpr (MR > 5) {
            i8_store4q<TOut>(row[5] + col0 + col, col_acc[5], sc, bi, vlo, vzp);
        }
        if constexpr (MR > 6) {
            i8_store4q<TOut>(row[6] + col0 + col, col_acc[6], sc, bi, vlo, vzp);
        }
        if constexpr (MR > 7) {
            i8_store4q<TOut>(row[7] + col0 + col, col_acc[7], sc, bi, vlo, vzp);
        }
    } else if (l > 0) {
        i8_store4q_lanes<TOut>(row[0] + col0 + col, l, col_acc[0], sc, bi, vlo, vzp);
        if constexpr (MR > 1) {
            i8_store4q_lanes<TOut>(row[1] + col0 + col, l, col_acc[1], sc, bi, vlo, vzp);
        }
        if constexpr (MR > 2) {
            i8_store4q_lanes<TOut>(row[2] + col0 + col, l, col_acc[2], sc, bi, vlo, vzp);
        }
        if constexpr (MR > 3) {
            i8_store4q_lanes<TOut>(row[3] + col0 + col, l, col_acc[3], sc, bi, vlo, vzp);
        }
        if constexpr (MR > 4) {
            i8_store4q_lanes<TOut>(row[4] + col0 + col, l, col_acc[4], sc, bi, vlo, vzp);
        }
        if constexpr (MR > 5) {
            i8_store4q_lanes<TOut>(row[5] + col0 + col, l, col_acc[5], sc, bi, vlo, vzp);
        }
        if constexpr (MR > 6) {
            i8_store4q_lanes<TOut>(row[6] + col0 + col, l, col_acc[6], sc, bi, vlo, vzp);
        }
        if constexpr (MR > 7) {
            i8_store4q_lanes<TOut>(row[7] + col0 + col, l, col_acc[7], sc, bi, vlo, vzp);
        }
    }
}

/// Requantize and write the live columns of NG groups, starting at `col0`.
template <int MR, int NG, class TOut>
__attribute__((always_inline)) inline void
i8_sdot_store(TOut* const (&row)[MR], int col0, int live, int32x4_t (&acc)[NG][MR],
              const float32x4_t (&sc)[NG], const int32x4_t (&bi)[NG], int32x4_t vlo,
              int32x4_t vzp) noexcept
{
    static_assert(NG >= 1 && NG <= 3, "a panel is one to three column groups");
    i8_sdot_store_col<MR, TOut>(row, col0, 0, live, acc[0], sc[0], bi[0], vlo, vzp);
    if constexpr (NG > 1) {
        i8_sdot_store_col<MR, TOut>(row, col0, 4, live, acc[1], sc[1], bi[1], vlo, vzp);
    }
    if constexpr (NG > 2) {
        i8_sdot_store_col<MR, TOut>(row, col0, 8, live, acc[2], sc[2], bi[2], vlo, vzp);
    }
}

/// One narrow remainder panel through NG groups, requantized.
template <int MR, int NG, class TOut>
__attribute__((always_inline)) inline void
i8_sdot_tail(TOut* const (&row)[MR], int n, int live, int kg, const int8_t* A, const int8_t* B,
             float32x4_t vk, float32x4_t vreq, const float* s_b, const int32_t* col_bias,
             int32x4_t vlo, int32x4_t vzp) noexcept
{
    int32x4_t acc[NG][MR];
    i8_sdot_mma<MR, NG>(A, B, kg, acc);
    float32x4_t sc[NG];
    int32x4_t bi[NG];
    i8_col_q_live<NG>(sc, bi, vk, vreq, s_b ? s_b + n : nullptr,
                      col_bias ? col_bias + n : nullptr, live);
    i8_sdot_store<MR, NG, TOut>(row, n, live, acc, sc, bi, vlo, vzp);
}

template <int MR, class TOut>
__attribute__((always_inline)) inline void
mma_pack_mr_nr12_epi_i8(int mr, int nc, int kc, TOut* NNOPS_RESTRICT C, int64_t ldm,
                        const int8_t* NNOPS_RESTRICT A, const int8_t* NNOPS_RESTRICT B, int ldb,
                        const float* s_b, float s_b_scalar, float row_k,
                        const int32_t* col_bias, int32_t zp_out, bool relu) noexcept
{
    TOut* row[MR];
    i8_sdot_row_ptrs<MR, TOut>(C, ldm, mr, row);
    const int kbytes = (kc + 3) & ~3;
    const int kg = kbytes >> 2;
    const int ldb_panel = mma_i8_panel_stride(ldb, kbytes);
    const float32x4_t vk = vdupq_n_f32(row_k);
    const float32x4_t vreq = vdupq_n_f32(row_k * s_b_scalar);
    const int32x4_t vlo = vdupq_n_s32(relu ? 0 : INT32_MIN);
    const int32x4_t vzp = vdupq_n_s32(zp_out);

    int n = 0;
    for (; n + mma_i8_nr_max <= nc; n += mma_i8_nr_max) {
        int32x4_t acc[3][MR];
        i8_sdot_mma<MR, 3>(A, B, kg, acc);
        float32x4_t sc[3];
        int32x4_t bi[3];
        i8_col_q<3>(sc, bi, vk, vreq, s_b ? s_b + n : nullptr, col_bias ? col_bias + n : nullptr);
        i8_sdot_store<MR, 3, TOut>(row, n, 12, acc, sc, bi, vlo, vzp);
        B += ldb_panel;
    }

    // The live remainder narrows to the groups it needs: 4, 8 or 12 columns.
    const int live = nc - n;
    if (live > 8) {
        i8_sdot_tail<MR, 3, TOut>(row, n, live, kg, A, B, vk, vreq, s_b, col_bias, vlo, vzp);
    } else if (live > 4) {
        i8_sdot_tail<MR, 2, TOut>(row, n, live, kg, A, B, vk, vreq, s_b, col_bias, vlo, vzp);
    } else if (live > 0) {
        i8_sdot_tail<MR, 1, TOut>(row, n, live, kg, A, B, vk, vreq, s_b, col_bias, vlo, vzp);
    }
}

/// The requantizing rungs, consumed by matmul_helper.cpp; rows are {8,4,1}.
template <class TOut>
inline constexpr std::array<MmaI8EpiFn<TOut>, 3> mma_pack_epi_fns_i8 = {
    &mma_pack_mr_nr12_epi_i8<8, TOut>,
    &mma_pack_mr_nr12_epi_i8<4, TOut>,
    &mma_pack_mr_nr12_epi_i8<1, TOut>,
};

// Dispatch tables, consumed by matmul_helper.cpp; rows are the MR levels {8,4,1}.
template <bool Z>
inline constexpr auto mma_pack_fns_i8 =
    std::array{&mma_pack_mr_nr12_mrxncxkc_i8<8, Z>, &mma_pack_mr_nr12_mrxncxkc_i8<4, Z>,
               &mma_pack_mr_nr12_mrxncxkc_i8<1, Z>};

// Row-major A (`lda` in int8 elements, kc % 4 == 0), same MR levels.
template <bool Z>
inline constexpr auto mma_direct_fns_i8 =
    std::array{&mma_direct_mr_nr12_mrxncxkc_i8<8, Z>, &mma_direct_mr_nr12_mrxncxkc_i8<4, Z>,
               &mma_direct_mr_nr12_mrxncxkc_i8<1, Z>};

} // namespace nnops::backend::cpu::aarch64
