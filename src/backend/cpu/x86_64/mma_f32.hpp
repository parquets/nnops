#pragma once
/// @file mma_f32.hpp
/// @brief x86_64 AVX2/FMA float32 MMA (matrix micro-accumulate) kernels.
///
/// One templated rung per A presentation, with the height a template parameter:
/// `MR` ∈ {6, 4, 2, 1} is the panel's row count, while the calls' `mr` (1..MR)
/// is the live height the M decomposition emits, so each level serves every
/// panel the largest-first walk produces instead of needing a kernel per
/// height. A and B may both be packed (`mma_pack_*`) or only B packed while A
/// is read row-major through `lda` (`mma_direct_*`), whose `ldb` is B's k-row
/// step: negative selects the packed layout (rows 16 apart), any other value is
/// a caller's matrix read in place. `zero_mode` drops the C read-add.
///
/// Packed B is uniform: ceil(nc / 16) panels, every one a full 16 columns wide
/// with the columns past `nc` zero-filled, each in an
/// mma_pack_align(16*kc*sizeof(float))-byte slot. That lets the kernel read 16
/// floats per k at every column, so a tail wider than 8 costs one k-loop
/// instead of two; a tail of 8 or less takes the single-vector loop, and only
/// the lanes it stores are written.
///
/// Rows past `mr` alias the last live row — the convention pack_trans_* fills on
/// the A side and the C row pointers repeat here — so a partial-M tile needs no
/// per-row guards. Aliased rows share an address, so every stage adds C for all
/// rows before it stores any of them.
///
/// Bias seed: 1 is per-row (a dense row is an output channel), 2 is per-column
/// (NHWC, rows are pixels, so the bias lands on the panel's 16 lanes). Seed 2
/// reads all 16 lanes, so the caller's bias buffer must be padded to the panel
/// width. The row seed goes through the same alias index as A and C, so a
/// partial M stays in range.
///
/// Every index into the accumulator block is a literal: the group walk and the
/// store are written out at the panel's two widths instead of looping to a
/// template bound. A loop costs nothing where the compiler unrolls it and
/// everything where it does not -- the array is then indexed at run time and
/// the whole block moves to the stack.

#include <array>
#include <immintrin.h>
#include <algorithm>
#include <cstdint>
#include "backend/cpu/common/common_utils.hpp"

namespace nnops::backend::cpu::x86_64 {

// =========================================================================
//  one walk: the seed, the K loop, the stores
// =========================================================================
//
// Every rung -- packed A or row-major, any height -- is the same walk: one k
// step of MR rows against NG eight-column groups, then a store. The
// accumulator is laid out [NG][MR] so a group's rows are contiguous, and every
// index into it is a literal (see the header).

/// The alias index of each row: rows past `mr` reuse the last live one.
template <int MR>
NNOPS_FORCEINLINE void f32_row_alias(int mr, int (&rr)[MR]) noexcept
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
}

/// The C rows, aliasing the last live one.
template <int MR>
NNOPS_FORCEINLINE void f32_row_ptrs(float* C, int ldc, const int (&rr)[MR],
                                    float* (&row)[MR]) noexcept
{
    row[0] = C;
    if constexpr (MR > 1) {
        row[1] = C + rr[1] * ldc;
    }
    if constexpr (MR > 2) {
        row[2] = C + rr[2] * ldc;
    }
    if constexpr (MR > 3) {
        row[3] = C + rr[3] * ldc;
    }
    if constexpr (MR > 4) {
        row[4] = C + rr[4] * ldc;
    }
    if constexpr (MR > 5) {
        row[5] = C + rr[5] * ldc;
    }
}

/// The row-major A rows, aliasing the last live one.
template <int MR>
NNOPS_FORCEINLINE void f32_a_ptrs(const float* A, int lda, const int (&rr)[MR],
                                  const float* (&a)[MR]) noexcept
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
}

/// One seed accumulator: the row's bias (`r`), the panel column's (`col`) or
/// zero. The row index is already aliased, so `bias[r]` stays in range.
template <int seed>
NNOPS_FORCEINLINE __m256 f32_seed([[maybe_unused]] const float* bias,
                                  [[maybe_unused]] int r,
                                  [[maybe_unused]] int col) noexcept
{
    if constexpr (seed == 1) {
        return _mm256_set1_ps(bias[r]);
    } else if constexpr (seed == 2) {
        return _mm256_loadu_ps(bias + col);
    } else {
        return _mm256_setzero_ps();
    }
}

/// The bias a panel starting at column `n` seeds from (seed 2 steps with it).
template <int seed>
NNOPS_FORCEINLINE const float* f32_panel_bias(const float* bias, int n) noexcept
{
    if constexpr (seed == 2) {
        return bias + n;
    } else {
        return bias;
    }
}

/// Seed one group's accumulators from the bias (or zero).
template <int MR, int seed>
NNOPS_FORCEINLINE void f32_col_init(__m256 (&col)[MR], const float* bias, const int (&rr)[MR],
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
}

NNOPS_FORCEINLINE __m256 f32_b_load(const float* b) noexcept
{
    return _mm256_loadu_ps(b);
}

/// One k step of a packed-A panel: MR rows against NG eight-column groups.
template <int MR, int NG>
NNOPS_FORCEINLINE void f32_group(const float*& a_ptr, const float* b_ptr,
                                 const int (&rr)[MR], __m256 (&acc)[NG][MR]) noexcept
{
    // The advance is hoisted above the reads on purpose: MSVC then emits the
    // pointer add at the top of the k loop, which measured ~10% faster than
    // letting it fold the loads and add after them.
    const float* const a_cur = a_ptr;
    a_ptr += MR;
    const __m256 vb0 = f32_b_load(b_ptr + 0 * 8);
    __m256 vb1 = vb0;
    if constexpr (NG > 1) {
        vb1 = f32_b_load(b_ptr + 1 * 8);
    }

    const __m256 va0 = _mm256_broadcast_ss(a_cur + rr[0]);
    acc[0][0] = _mm256_fmadd_ps(va0, vb0, acc[0][0]);
    if constexpr (NG > 1) {
        acc[1][0] = _mm256_fmadd_ps(va0, vb1, acc[1][0]);
    }
    if constexpr (MR > 1) {
        const __m256 va1 = _mm256_broadcast_ss(a_cur + rr[1]);
        acc[0][1] = _mm256_fmadd_ps(va1, vb0, acc[0][1]);
        if constexpr (NG > 1) {
            acc[1][1] = _mm256_fmadd_ps(va1, vb1, acc[1][1]);
        }
    }
    if constexpr (MR > 2) {
        const __m256 va2 = _mm256_broadcast_ss(a_cur + rr[2]);
        acc[0][2] = _mm256_fmadd_ps(va2, vb0, acc[0][2]);
        if constexpr (NG > 1) {
            acc[1][2] = _mm256_fmadd_ps(va2, vb1, acc[1][2]);
        }
    }
    if constexpr (MR > 3) {
        const __m256 va3 = _mm256_broadcast_ss(a_cur + rr[3]);
        acc[0][3] = _mm256_fmadd_ps(va3, vb0, acc[0][3]);
        if constexpr (NG > 1) {
            acc[1][3] = _mm256_fmadd_ps(va3, vb1, acc[1][3]);
        }
    }
    if constexpr (MR > 4) {
        const __m256 va4 = _mm256_broadcast_ss(a_cur + rr[4]);
        acc[0][4] = _mm256_fmadd_ps(va4, vb0, acc[0][4]);
        if constexpr (NG > 1) {
            acc[1][4] = _mm256_fmadd_ps(va4, vb1, acc[1][4]);
        }
    }
    if constexpr (MR > 5) {
        const __m256 va5 = _mm256_broadcast_ss(a_cur + rr[5]);
        acc[0][5] = _mm256_fmadd_ps(va5, vb0, acc[0][5]);
        if constexpr (NG > 1) {
            acc[1][5] = _mm256_fmadd_ps(va5, vb1, acc[1][5]);
        }
    }
}

/// One k step of a row-major panel: MR rows against NG eight-column groups.
/// Each row's scalar is its own load, and the row pointers walk with k.
template <int MR, int NG>
NNOPS_FORCEINLINE void f32_step_rows(const float* (&a)[MR], const float* b_ptr,
                                     __m256 (&acc)[NG][MR]) noexcept
{
    const __m256 vb0 = f32_b_load(b_ptr + 0 * 8);
    __m256 vb1 = vb0;
    if constexpr (NG > 1) {
        vb1 = f32_b_load(b_ptr + 1 * 8);
    }

    const float* const p0 = a[0];


    const __m256 va0 = _mm256_broadcast_ss(p0);


    a[0] = p0 + 1;
    acc[0][0] = _mm256_fmadd_ps(va0, vb0, acc[0][0]);
    if constexpr (NG > 1) {
        acc[1][0] = _mm256_fmadd_ps(va0, vb1, acc[1][0]);
    }
    if constexpr (MR > 1) {
        const float* const p1 = a[1];

        const __m256 va1 = _mm256_broadcast_ss(p1);

        a[1] = p1 + 1;
        acc[0][1] = _mm256_fmadd_ps(va1, vb0, acc[0][1]);
        if constexpr (NG > 1) {
            acc[1][1] = _mm256_fmadd_ps(va1, vb1, acc[1][1]);
        }
    }
    if constexpr (MR > 2) {
        const float* const p2 = a[2];

        const __m256 va2 = _mm256_broadcast_ss(p2);

        a[2] = p2 + 1;
        acc[0][2] = _mm256_fmadd_ps(va2, vb0, acc[0][2]);
        if constexpr (NG > 1) {
            acc[1][2] = _mm256_fmadd_ps(va2, vb1, acc[1][2]);
        }
    }
    if constexpr (MR > 3) {
        const float* const p3 = a[3];

        const __m256 va3 = _mm256_broadcast_ss(p3);

        a[3] = p3 + 1;
        acc[0][3] = _mm256_fmadd_ps(va3, vb0, acc[0][3]);
        if constexpr (NG > 1) {
            acc[1][3] = _mm256_fmadd_ps(va3, vb1, acc[1][3]);
        }
    }
    if constexpr (MR > 4) {
        const float* const p4 = a[4];

        const __m256 va4 = _mm256_broadcast_ss(p4);

        a[4] = p4 + 1;
        acc[0][4] = _mm256_fmadd_ps(va4, vb0, acc[0][4]);
        if constexpr (NG > 1) {
            acc[1][4] = _mm256_fmadd_ps(va4, vb1, acc[1][4]);
        }
    }
    if constexpr (MR > 5) {
        const float* const p5 = a[5];

        const __m256 va5 = _mm256_broadcast_ss(p5);

        a[5] = p5 + 1;
        acc[0][5] = _mm256_fmadd_ps(va5, vb0, acc[0][5]);
        if constexpr (NG > 1) {
            acc[1][5] = _mm256_fmadd_ps(va5, vb1, acc[1][5]);
        }
    }
}

/// The panel's whole K from packed A into seeded `acc`.
template <int MR, int NG, int seed>
NNOPS_FORCEINLINE void f32_mma(const float* A, const float* B, int kc, int bstep,
                               const float* bias, const int (&rr)[MR],
                               __m256 (&acc)[NG][MR]) noexcept
{
    f32_col_init<MR, seed>(acc[0], bias, rr, 0);
    if constexpr (NG > 1) {
        f32_col_init<MR, seed>(acc[1], bias, rr, 8);
    }
    const float* a_ptr = A;
    const float* b_ptr = B;
    for (int k = 0; k < kc; ++k) {
        f32_group<MR, NG>(a_ptr, b_ptr, rr, acc);
        b_ptr += bstep;
    }
}

/// The panel's whole K from row-major rows into seeded `acc`.
template <int MR, int NG, int seed>
NNOPS_FORCEINLINE void f32_mma_rows(const float* (&a)[MR], const float* B, int kc,
                                    int bstep, const float* bias, const int (&rr)[MR],
                                    __m256 (&acc)[NG][MR]) noexcept
{
    f32_col_init<MR, seed>(acc[0], bias, rr, 0);
    if constexpr (NG > 1) {
        f32_col_init<MR, seed>(acc[1], bias, rr, 8);
    }
    const float* b_ptr = B;
    for (int k = 0; k < kc; ++k, b_ptr += bstep) {
        f32_step_rows<MR, NG>(a, b_ptr, acc);
    }
}

// ---- the store ------------------------------------------------------------
//
// A group of eight columns is written in two passes: every row's C is folded
// into its accumulator first, then the rows are stored. Aliased rows share an
// address, so a row-at-a-time add would have the second duplicate reading C
// with the first one's result already in it.

NNOPS_FORCEINLINE __m128 f32_clamp4(__m128 v, __m128 vmin, __m128 vmax) noexcept
{
    return _mm_max_ps(_mm_min_ps(v, vmax), vmin);
}

/// Fold C into one group's accumulators, every row before any store.
template <int MR>
NNOPS_FORCEINLINE void f32_add_c(float* const (&row)[MR], int off, __m256 (&col)[MR]) noexcept
{
    col[0] = _mm256_add_ps(col[0], f32_b_load(row[0] + off));
    if constexpr (MR > 1) {
        col[1] = _mm256_add_ps(col[1], f32_b_load(row[1] + off));
    }
    if constexpr (MR > 2) {
        col[2] = _mm256_add_ps(col[2], f32_b_load(row[2] + off));
    }
    if constexpr (MR > 3) {
        col[3] = _mm256_add_ps(col[3], f32_b_load(row[3] + off));
    }
    if constexpr (MR > 4) {
        col[4] = _mm256_add_ps(col[4], f32_b_load(row[4] + off));
    }
    if constexpr (MR > 5) {
        col[5] = _mm256_add_ps(col[5], f32_b_load(row[5] + off));
    }
}

/// Clamp and store one full group, no C add.
template <int MR>
NNOPS_FORCEINLINE void f32_store_c(float* const (&row)[MR], int off, __m256 (&col)[MR],
                                   __m256 vmin, __m256 vmax) noexcept
{
    col[0] = _mm256_max_ps(_mm256_min_ps(col[0], vmax), vmin);
    _mm256_storeu_ps(row[0] + off, col[0]);
    if constexpr (MR > 1) {
        col[1] = _mm256_max_ps(_mm256_min_ps(col[1], vmax), vmin);
        _mm256_storeu_ps(row[1] + off, col[1]);
    }
    if constexpr (MR > 2) {
        col[2] = _mm256_max_ps(_mm256_min_ps(col[2], vmax), vmin);
        _mm256_storeu_ps(row[2] + off, col[2]);
    }
    if constexpr (MR > 3) {
        col[3] = _mm256_max_ps(_mm256_min_ps(col[3], vmax), vmin);
        _mm256_storeu_ps(row[3] + off, col[3]);
    }
    if constexpr (MR > 4) {
        col[4] = _mm256_max_ps(_mm256_min_ps(col[4], vmax), vmin);
        _mm256_storeu_ps(row[4] + off, col[4]);
    }
    if constexpr (MR > 5) {
        col[5] = _mm256_max_ps(_mm256_min_ps(col[5], vmax), vmin);
        _mm256_storeu_ps(row[5] + off, col[5]);
    }
}

/// The live lanes of a group: one stage (four floats, then two, then one) for
/// every row before that stage's stores, so duplicate (aliased) rows all read
/// the same C.
template <int MR, bool zero_mode>
NNOPS_FORCEINLINE void f32_store_lanes(float* const (&row)[MR], int off, int lanes,
                                       __m256 (&col)[MR], __m256 vmin, __m256 vmax) noexcept
{
    const __m128 vmin4 = _mm256_castps256_ps128(vmin);
    const __m128 vmax4 = _mm256_castps256_ps128(vmax);
    __m128 lo[MR];
    lo[0] = _mm256_castps256_ps128(col[0]);
    if constexpr (MR > 1) {
        lo[1] = _mm256_castps256_ps128(col[1]);
    }
    if constexpr (MR > 2) {
        lo[2] = _mm256_castps256_ps128(col[2]);
    }
    if constexpr (MR > 3) {
        lo[3] = _mm256_castps256_ps128(col[3]);
    }
    if constexpr (MR > 4) {
        lo[4] = _mm256_castps256_ps128(col[4]);
    }
    if constexpr (MR > 5) {
        lo[5] = _mm256_castps256_ps128(col[5]);
    }

    if (lanes & 4) {
        if constexpr (!zero_mode) {
            lo[0] = _mm_add_ps(lo[0], _mm_loadu_ps(row[0] + off));
            if constexpr (MR > 1) {
                lo[1] = _mm_add_ps(lo[1], _mm_loadu_ps(row[1] + off));
            }
            if constexpr (MR > 2) {
                lo[2] = _mm_add_ps(lo[2], _mm_loadu_ps(row[2] + off));
            }
            if constexpr (MR > 3) {
                lo[3] = _mm_add_ps(lo[3], _mm_loadu_ps(row[3] + off));
            }
            if constexpr (MR > 4) {
                lo[4] = _mm_add_ps(lo[4], _mm_loadu_ps(row[4] + off));
            }
            if constexpr (MR > 5) {
                lo[5] = _mm_add_ps(lo[5], _mm_loadu_ps(row[5] + off));
            }
        }
        lo[0] = f32_clamp4(lo[0], vmin4, vmax4);
        _mm_storeu_ps(row[0] + off, lo[0]);
        if constexpr (MR > 1) {
            lo[1] = f32_clamp4(lo[1], vmin4, vmax4);
            _mm_storeu_ps(row[1] + off, lo[1]);
        }
        if constexpr (MR > 2) {
            lo[2] = f32_clamp4(lo[2], vmin4, vmax4);
            _mm_storeu_ps(row[2] + off, lo[2]);
        }
        if constexpr (MR > 3) {
            lo[3] = f32_clamp4(lo[3], vmin4, vmax4);
            _mm_storeu_ps(row[3] + off, lo[3]);
        }
        if constexpr (MR > 4) {
            lo[4] = f32_clamp4(lo[4], vmin4, vmax4);
            _mm_storeu_ps(row[4] + off, lo[4]);
        }
        if constexpr (MR > 5) {
            lo[5] = f32_clamp4(lo[5], vmin4, vmax4);
            _mm_storeu_ps(row[5] + off, lo[5]);
        }
        off += 4;
        lo[0] = _mm256_extractf128_ps(col[0], 1);
        if constexpr (MR > 1) {
            lo[1] = _mm256_extractf128_ps(col[1], 1);
        }
        if constexpr (MR > 2) {
            lo[2] = _mm256_extractf128_ps(col[2], 1);
        }
        if constexpr (MR > 3) {
            lo[3] = _mm256_extractf128_ps(col[3], 1);
        }
        if constexpr (MR > 4) {
            lo[4] = _mm256_extractf128_ps(col[4], 1);
        }
        if constexpr (MR > 5) {
            lo[5] = _mm256_extractf128_ps(col[5], 1);
        }
    }

    if (lanes & 2) {
        __m128 rest[MR];
        rest[0] = lo[0];
        if constexpr (MR > 1) {
            rest[1] = lo[1];
        }
        if constexpr (MR > 2) {
            rest[2] = lo[2];
        }
        if constexpr (MR > 3) {
            rest[3] = lo[3];
        }
        if constexpr (MR > 4) {
            rest[4] = lo[4];
        }
        if constexpr (MR > 5) {
            rest[5] = lo[5];
        }
        if constexpr (!zero_mode) {
            lo[0] = _mm_add_ps(
                lo[0], _mm_castsi128_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(row[0] + off))));
            if constexpr (MR > 1) {
                lo[1] = _mm_add_ps(
                    lo[1], _mm_castsi128_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(row[1] + off))));
            }
            if constexpr (MR > 2) {
                lo[2] = _mm_add_ps(
                    lo[2], _mm_castsi128_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(row[2] + off))));
            }
            if constexpr (MR > 3) {
                lo[3] = _mm_add_ps(
                    lo[3], _mm_castsi128_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(row[3] + off))));
            }
            if constexpr (MR > 4) {
                lo[4] = _mm_add_ps(
                    lo[4], _mm_castsi128_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(row[4] + off))));
            }
            if constexpr (MR > 5) {
                lo[5] = _mm_add_ps(
                    lo[5], _mm_castsi128_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(row[5] + off))));
            }
        }
        lo[0] = f32_clamp4(lo[0], vmin4, vmax4);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(row[0] + off), _mm_castps_si128(lo[0]));
        if constexpr (MR > 1) {
            lo[1] = f32_clamp4(lo[1], vmin4, vmax4);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(row[1] + off), _mm_castps_si128(lo[1]));
        }
        if constexpr (MR > 2) {
            lo[2] = f32_clamp4(lo[2], vmin4, vmax4);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(row[2] + off), _mm_castps_si128(lo[2]));
        }
        if constexpr (MR > 3) {
            lo[3] = f32_clamp4(lo[3], vmin4, vmax4);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(row[3] + off), _mm_castps_si128(lo[3]));
        }
        if constexpr (MR > 4) {
            lo[4] = f32_clamp4(lo[4], vmin4, vmax4);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(row[4] + off), _mm_castps_si128(lo[4]));
        }
        if constexpr (MR > 5) {
            lo[5] = f32_clamp4(lo[5], vmin4, vmax4);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(row[5] + off), _mm_castps_si128(lo[5]));
        }
        off += 2;
        lo[0] = _mm_movehl_ps(rest[0], rest[0]);
        if constexpr (MR > 1) {
            lo[1] = _mm_movehl_ps(rest[1], rest[1]);
        }
        if constexpr (MR > 2) {
            lo[2] = _mm_movehl_ps(rest[2], rest[2]);
        }
        if constexpr (MR > 3) {
            lo[3] = _mm_movehl_ps(rest[3], rest[3]);
        }
        if constexpr (MR > 4) {
            lo[4] = _mm_movehl_ps(rest[4], rest[4]);
        }
        if constexpr (MR > 5) {
            lo[5] = _mm_movehl_ps(rest[5], rest[5]);
        }
    }

    if (lanes & 1) {
        if constexpr (!zero_mode) {
            lo[0] = _mm_add_ss(lo[0], _mm_load_ss(row[0] + off));
            if constexpr (MR > 1) {
                lo[1] = _mm_add_ss(lo[1], _mm_load_ss(row[1] + off));
            }
            if constexpr (MR > 2) {
                lo[2] = _mm_add_ss(lo[2], _mm_load_ss(row[2] + off));
            }
            if constexpr (MR > 3) {
                lo[3] = _mm_add_ss(lo[3], _mm_load_ss(row[3] + off));
            }
            if constexpr (MR > 4) {
                lo[4] = _mm_add_ss(lo[4], _mm_load_ss(row[4] + off));
            }
            if constexpr (MR > 5) {
                lo[5] = _mm_add_ss(lo[5], _mm_load_ss(row[5] + off));
            }
        }
        _mm_store_ss(row[0] + off, _mm_max_ss(_mm_min_ss(lo[0], vmax4), vmin4));
        if constexpr (MR > 1) {
            _mm_store_ss(row[1] + off, _mm_max_ss(_mm_min_ss(lo[1], vmax4), vmin4));
        }
        if constexpr (MR > 2) {
            _mm_store_ss(row[2] + off, _mm_max_ss(_mm_min_ss(lo[2], vmax4), vmin4));
        }
        if constexpr (MR > 3) {
            _mm_store_ss(row[3] + off, _mm_max_ss(_mm_min_ss(lo[3], vmax4), vmin4));
        }
        if constexpr (MR > 4) {
            _mm_store_ss(row[4] + off, _mm_max_ss(_mm_min_ss(lo[4], vmax4), vmin4));
        }
        if constexpr (MR > 5) {
            _mm_store_ss(row[5] + off, _mm_max_ss(_mm_min_ss(lo[5], vmax4), vmin4));
        }
    }
}

/// Write one group of a panel: MR rows, `live` columns left in the panel from
/// this group's first column.
template <int MR, bool zero_mode>
NNOPS_FORCEINLINE void f32_store_col(float* const (&row)[MR], int col0, int col, int live,
                                     __m256 (&col_acc)[MR], __m256 vmin, __m256 vmax) noexcept
{
    const int l = live - col;
    if (l >= 8) {
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
NNOPS_FORCEINLINE void f32_store(float* const (&row)[MR], int col0, int live,
                                 __m256 (&acc)[NG][MR], __m256 vmin, __m256 vmax) noexcept
{
    f32_store_col<MR, zero_mode>(row, col0, 0, live, acc[0], vmin, vmax);
    if constexpr (NG > 1) {
        f32_store_col<MR, zero_mode>(row, col0, 8, live, acc[1], vmin, vmax);
    }
    if constexpr (NG > 2) {
        f32_store_col<MR, zero_mode>(row, col0, 16, live, acc[2], vmin, vmax);
    }
    if constexpr (NG > 3) {
        f32_store_col<MR, zero_mode>(row, col0, 24, live, acc[3], vmin, vmax);
    }
}

// =========================================================================
//  packed A + packed B  (MR = 6, 4, 2, 1)
// =========================================================================

/// One packed-A rung: the whole N side into C.
template <int MR, bool zero_mode = false, int seed = 0>
inline void mma_pack_mr_nr16_mrxncxkc_f32(int mr, int nc, int kc, float* NNOPS_RESTRICT C,
                                          int ldc, const float* NNOPS_RESTRICT A,
                                          const float* NNOPS_RESTRICT B, float clamp_min,
                                          float clamp_max,
                                          const float* NNOPS_RESTRICT bias = nullptr) noexcept
{
    int rr[MR];
    f32_row_alias<MR>(mr, rr);
    float* row[MR];
    f32_row_ptrs<MR>(C, ldc, rr, row);
    const int ldb_panel =
        static_cast<int>(mma_pack_align(16 * kc * sizeof(float)) / sizeof(float));
    const __m256 vmin = _mm256_set1_ps(clamp_min);
    const __m256 vmax = _mm256_set1_ps(clamp_max);

    int n = 0;
    for (; n < nc - 15; n += 16) {
        __m256 acc[2][MR];
        f32_mma<MR, 2, seed>(A, B, kc, 16, f32_panel_bias<seed>(bias, n), rr, acc);
        f32_store<MR, 2, zero_mode>(row, n, 16, acc, vmin, vmax);
        B += ldb_panel;
    }

    const int live = nc - n;
    if (live > 8) {
        __m256 acc[2][MR];
        f32_mma<MR, 2, seed>(A, B, kc, 16, f32_panel_bias<seed>(bias, n), rr, acc);
        f32_store<MR, 2, zero_mode>(row, n, live, acc, vmin, vmax);
    } else if (live > 0) {
        __m256 acc[1][MR];
        f32_mma<MR, 1, seed>(A, B, kc, 16, f32_panel_bias<seed>(bias, n), rr, acc);
        f32_store<MR, 1, zero_mode>(row, n, live, acc, vmin, vmax);
    }
}

// =========================================================================
//  raw A + packed B  (MR = 6, 4, 2, 1)
// =========================================================================
//
// The same walk with each row's scalar loaded per k instead of a shared panel
// element, so the rung reads A through `lda`.

/// One row-major-A rung: the whole N side into C.
template <int MR, bool zero_mode = false, int seed = 0>
inline void mma_direct_mr_nr16_mrxncxkc_f32(
    int mr, int nc, int kc, float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A, int lda, const float* NNOPS_RESTRICT B, int ldb,
    float clamp_min, float clamp_max,
    const float* NNOPS_RESTRICT bias = nullptr) noexcept
{
    int rr[MR];
    f32_row_alias<MR>(mr, rr);
    float* row[MR];
    f32_row_ptrs<MR>(C, ldc, rr, row);
    const int ldb_panel =
        static_cast<int>(mma_pack_align(16 * kc * sizeof(float)) / sizeof(float));
    const __m256 vmin = _mm256_set1_ps(clamp_min);
    const __m256 vmax = _mm256_set1_ps(clamp_max);
    const int row_step = (ldb < 0) ? 16 : ldb;
    int panel_step = ldb_panel;
    if constexpr (MR == 1) {
        // Below -1 a negative ldb is how far apart the caller's packed panels
        // sit; without one the panels are this call's own and the slot is right.
        if (ldb < -1) {
            panel_step = -ldb;
        }
    }

    int n = 0;
    if constexpr (MR == 1) {
        // Two packed panels per pass: at one row the loop is nothing but weight
        // stream, and its scalar A load costs a whole load slot of its own, so
        // one scalar over 32 columns is the margin.
        for (; ldb < 0 && n <= nc - 32; n += 32) {
            const float* const bp = f32_panel_bias<seed>(bias, n);
            __m256 acc[4][1];
            f32_col_init<1, seed>(acc[0], bp, rr, 0);
            f32_col_init<1, seed>(acc[1], bp, rr, 8);
            f32_col_init<1, seed>(acc[2], bp, rr, 16);
            f32_col_init<1, seed>(acc[3], bp, rr, 24);
            const float* ap = A;
            const float* b0_ptr = B;
            const float* b1_ptr = B + panel_step;
            for (int k = 0; k < kc; ++k) {
                const __m256 d0 = f32_b_load(b0_ptr + 0 * 8);
                const __m256 d1 = f32_b_load(b0_ptr + 1 * 8);
                const __m256 d2 = f32_b_load(b1_ptr + 0 * 8);
                const __m256 d3 = f32_b_load(b1_ptr + 1 * 8);
                b0_ptr += row_step;
                b1_ptr += row_step;
                const __m256 va = _mm256_broadcast_ss(ap++);
                acc[0][0] = _mm256_fmadd_ps(va, d0, acc[0][0]);
                acc[1][0] = _mm256_fmadd_ps(va, d1, acc[1][0]);
                acc[2][0] = _mm256_fmadd_ps(va, d2, acc[2][0]);
                acc[3][0] = _mm256_fmadd_ps(va, d3, acc[3][0]);
            }
            f32_store<1, 4, zero_mode>(row, n, 32, acc, vmin, vmax);
            B += 2 * panel_step;
        }
    }

    for (; n < nc - 15; n += 16) {
        const float* a0[MR];
        f32_a_ptrs<MR>(A, lda, rr, a0);
        __m256 acc[2][MR];
        f32_mma_rows<MR, 2, seed>(a0, B, kc, row_step, f32_panel_bias<seed>(bias, n), rr, acc);
        f32_store<MR, 2, zero_mode>(row, n, 16, acc, vmin, vmax);
        B += panel_step;
    }

    const int live = nc - n;
    if (live > 8) {
        const float* a0[MR];
        f32_a_ptrs<MR>(A, lda, rr, a0);
        __m256 acc[2][MR];
        f32_mma_rows<MR, 2, seed>(a0, B, kc, row_step, f32_panel_bias<seed>(bias, n), rr, acc);
        f32_store<MR, 2, zero_mode>(row, n, live, acc, vmin, vmax);
    } else if (live > 0) {
        const float* a0[MR];
        f32_a_ptrs<MR>(A, lda, rr, a0);
        __m256 acc[1][MR];
        f32_mma_rows<MR, 1, seed>(a0, B, kc, row_step, f32_panel_bias<seed>(bias, n), rr, acc);
        f32_store<MR, 1, zero_mode>(row, n, live, acc, vmin, vmax);
    }
}

// ---- dispatch tables, consumed by matmul_helper.cpp ---------------------

template <bool Z, int Seed = 0>
inline constexpr auto mma_pack_fns_f32 =
    std::array{&mma_pack_mr_nr16_mrxncxkc_f32<6, Z, Seed>,
               &mma_pack_mr_nr16_mrxncxkc_f32<4, Z, Seed>,
               &mma_pack_mr_nr16_mrxncxkc_f32<2, Z, Seed>,
               &mma_pack_mr_nr16_mrxncxkc_f32<1, Z, Seed>};

template <bool Z, int Seed = 0>
inline constexpr auto mma_direct_fns_f32 =
    std::array{&mma_direct_mr_nr16_mrxncxkc_f32<6, Z, Seed>,
               &mma_direct_mr_nr16_mrxncxkc_f32<4, Z, Seed>,
               &mma_direct_mr_nr16_mrxncxkc_f32<2, Z, Seed>,
               &mma_direct_mr_nr16_mrxncxkc_f32<1, Z, Seed>};

} // namespace nnops::backend::cpu::x86_64
