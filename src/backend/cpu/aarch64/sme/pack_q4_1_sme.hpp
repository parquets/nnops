#pragma once
/// @file pack_q4_1_sme.hpp
/// @brief AArch64 SME Q4_1 decode into the ZA kernels' RHS layout.
///
/// pack_q4_1.hpp decodes a Q4_1 block stream into the NEON panels — the
/// uniform [k][nr] rectangles, or BFMMLA's operand layout. The ZA kernels read
/// a different B, the [n/32][k][32] float block pack_f32_sme.hpp writes (or
/// [n/32][pair][64] half pairs for f16), so the decode is re-targeted here
/// rather than run into a dense matrix first, which would write and re-read
/// the whole weight.
///
/// Two directions, because B arrives either way:
///  - [K][N] (MatMul's default): every k row carries its own block stream, so
///    a panel row is exactly one 32-value block when the panel starts on a
///    block boundary — one decode, one 128-byte store.
///  - [N][K] (Linear's weight, MatMul transposed): a decode yields 32
///    consecutive k of ONE column, and the destination wants one k row's 32
///    columns. That is a 32x32 transpose; four columns at a time is the unit,
///    a 4x4 32-bit transpose with 16-byte row stores. (Building it through ZA
///    was measured 3-5x slower than this on an M4: the ZA row-load/column-store
///    stream, not the transpose, was the limit.)
///
/// The transposed path needs a 32-aligned k origin (a Q4_1 block boundary).
/// The SME plan's k-block starts always are; a misaligned one falls back to a
/// scalar column sweep, which is correct if slow.
///
/// Columns past @p nc are zeroed, and so are the k lanes a short k-block
/// leaves unread, so the workspace is deterministic for any kc (an odd f16 kc
/// zero-pads its last K pair).

#if defined(NNOPS_ARCH_SME)

#include <algorithm>
#include <cstdint>
#include <cstring>

#include <arm_neon.h>

#include "backend/cpu/aarch64/pack_q4_1.hpp"
#include "backend/cpu/aarch64/sme/mma_sme.hpp"

namespace nnops::backend::cpu::aarch64 {

namespace q4_sme_detail {

/// Values and bytes of one Q4_1 block, for the index math below.
inline constexpr int kQ4Elems = static_cast<int>(q4_1_block_elems);
inline constexpr int kQ4Bytes = static_cast<int>(q4_1_block_bytes);

/// The 4x4 transpose of four columns' 32-bit runs (an f32 value or an f16 K
/// pair) into four destination rows of 4 words. @p o is the first row.
inline void store_rows4(uint32_t* o, const uint32_t* c0, const uint32_t* c1, const uint32_t* c2,
                        const uint32_t* c3, int64_t row_stride) noexcept
{
    const uint32x4_t a = vld1q_u32(c0);
    const uint32x4_t b = vld1q_u32(c1);
    const uint32x4_t c = vld1q_u32(c2);
    const uint32x4_t d = vld1q_u32(c3);
    const uint32x4x2_t t01 = vtrnq_u32(a, b);
    const uint32x4x2_t t23 = vtrnq_u32(c, d);
    vst1q_u32(o, vcombine_u32(vget_low_u32(t01.val[0]), vget_low_u32(t23.val[0])));
    vst1q_u32(o + row_stride, vcombine_u32(vget_low_u32(t01.val[1]), vget_low_u32(t23.val[1])));
    vst1q_u32(o + 2 * row_stride,
              vcombine_u32(vget_high_u32(t01.val[0]), vget_high_u32(t23.val[0])));
    vst1q_u32(o + 3 * row_stride,
              vcombine_u32(vget_high_u32(t01.val[1]), vget_high_u32(t23.val[1])));
}

/// Decode @p count consecutive values of one row's block stream starting at
/// element @p at into @p out. @p count <= 32 and the window must stay inside
/// the row; blocks past the row's end decode as zeros.
template <class T>
inline void decode_window(const uint8_t* row, int64_t row_blocks, int at, int count, T* out,
                          float scale) noexcept
{
    T tmp[kQ4Elems];
    const int blk = at >> 5;
    const int off = at & 31;
    int done = 0;
    for (int b = blk; done < count; ++b) {
        if (b < row_blocks) {
            decode_q4_1_block<T>(row + static_cast<int64_t>(b) * kQ4Bytes, tmp, scale);
        } else {
            std::fill(tmp, tmp + kQ4Elems, T{});
        }
        const int start = (b == blk) ? off : 0;
        const int n = std::min(kQ4Elems - start, count - done);
        std::memcpy(out + done, tmp + start, static_cast<size_t>(n) * sizeof(T));
        done += n;
    }
}

/// One destination row of a [K][N] panel: the @p ncols columns at @p at, zero
/// past them. A block-aligned full row is a single decode.
template <class T>
inline void copy_row(T* out, const uint8_t* row, int64_t row_blocks, int at, int ncols,
                     float scale) noexcept
{
    if ((at & 31) == 0 && ncols == sme_fp_nr) {
        decode_q4_1_block<T>(row + static_cast<int64_t>(at >> 5) * kQ4Bytes, out, scale);
        return;
    }
    decode_window<T>(row, row_blocks, at, ncols, out, scale);
    std::fill(out + ncols, out + sme_fp_nr, T{});
}

// ---- [K][N]: one block per destination row -------------------------------

inline void pack_b_copy_f32(int n_start, int nc, int k_start, int kc, float* dst,
                            const uint8_t* src, int64_t row_blocks, float scale) noexcept
{
    const int64_t row_bytes = row_blocks * static_cast<int64_t>(kQ4Bytes);
    const int panels = (nc + sme_fp_nr - 1) / sme_fp_nr;
    for (int nb = 0; nb < panels; ++nb) {
        const int n0 = nb * sme_fp_nr;
        const int cols = std::min(sme_fp_nr, nc - n0);
        const int at = n_start + n0;
        float* const panel = dst + static_cast<int64_t>(nb) * kc * sme_fp_nr;
        for (int k = 0; k < kc; ++k) {
            const uint8_t* row = src + static_cast<int64_t>(k_start + k) * row_bytes;
            copy_row(panel + static_cast<int64_t>(k) * sme_fp_nr, row, row_blocks, at, cols,
                     scale);
        }
    }
}

inline void pack_b_copy_f16(int n_start, int nc, int k_start, int kc, half* dst,
                            const uint8_t* src, int64_t row_blocks, float scale) noexcept
{
    static_assert(sme_fp_nr % 8 == 0, "the pair zip walks 8 columns at a time");
    const int64_t row_bytes = row_blocks * static_cast<int64_t>(kQ4Bytes);
    const int panels = (nc + sme_fp_nr - 1) / sme_fp_nr;
    const int k_pairs = (kc + 1) / 2;
    alignas(16) half v0[sme_fp_nr];
    alignas(16) half v1[sme_fp_nr];
    for (int nb = 0; nb < panels; ++nb) {
        const int n0 = nb * sme_fp_nr;
        const int cols = std::min(sme_fp_nr, nc - n0);
        const int at = n_start + n0;
        half* const panel = dst + static_cast<int64_t>(nb) * k_pairs * (2 * sme_fp_nr);
        for (int p = 0; p < k_pairs; ++p) {
            const int k = 2 * p;
            const uint8_t* const row0 = src + static_cast<int64_t>(k_start + k) * row_bytes;
            copy_row(v0, row0, row_blocks, at, cols, scale);
            if (k + 1 < kc) {
                copy_row(v1, row0 + row_bytes, row_blocks, at, cols, scale);
            } else {
                std::fill(v1, v1 + sme_fp_nr, half{});
            }
            // Rows k and k+1 interleave one column pair per destination half,
            // so the pair row is two zips of 8 columns.
            half* const out = panel + static_cast<int64_t>(p) * (2 * sme_fp_nr);
            for (int j = 0; j < sme_fp_nr; j += 8) {
                const float16x8_t a = vld1q_f16(reinterpret_cast<const float16_t*>(v0 + j));
                const float16x8_t b = vld1q_f16(reinterpret_cast<const float16_t*>(v1 + j));
                vst1q_f16(reinterpret_cast<float16_t*>(out + 2 * j), vzip1q_f16(a, b));
                vst1q_f16(reinterpret_cast<float16_t*>(out + 2 * j + 8), vzip2q_f16(a, b));
            }
        }
    }
}

// ---- [N][K]: four columns at a time through the 4x4 transpose ------------

/// f32: one word per (column, k). The 32-k group is eight 4x4 blocks.
inline void pack_b_trans_f32(int n_start, int nc, int k_start, int kc, float* dst,
                             const uint8_t* src, int64_t row_blocks, float scale) noexcept
{
    const int64_t row_bytes = row_blocks * static_cast<int64_t>(kQ4Bytes);
    const int panels = (nc + sme_fp_nr - 1) / sme_fp_nr;
    const int blk0 = k_start >> 5;
    alignas(16) float col[4][kQ4Elems];

    for (int nb = 0; nb < panels; ++nb) {
        const int n0 = nb * sme_fp_nr;
        const int cols = std::min(sme_fp_nr, nc - n0);
        const uint8_t* const base = src + static_cast<int64_t>(n_start + n0) * row_bytes;
        auto* const panel = reinterpret_cast<uint32_t*>(dst + static_cast<int64_t>(nb) * kc *
                                                                   sme_fp_nr);
        for (int kb = 0; kb * kQ4Elems < kc; ++kb) {
            const int kbase = kb * kQ4Elems;
            const int rows = std::min(kQ4Elems, kc - kbase);
            const int64_t blk = blk0 + kb;
            for (int j0 = 0; j0 < cols; j0 += 4) {
                // Four columns; the ones past nc decode as zeros, which is
                // exactly the padding the panel wants.
                for (int jj = 0; jj < 4; ++jj) {
                    const int j = j0 + jj;
                    if (j < cols && blk < row_blocks) {
                        decode_q4_1_block<float>(
                            base + static_cast<int64_t>(j) * row_bytes +
                                blk * static_cast<int64_t>(kQ4Bytes),
                            col[jj], scale);
                        for (int k = rows; k < kQ4Elems; ++k) {
                            col[jj][k] = 0.0f;
                        }
                    } else {
                        std::fill(col[jj], col[jj] + kQ4Elems, 0.0f);
                    }
                }
                const auto* const c0 = reinterpret_cast<const uint32_t*>(col[0]);
                const auto* const c1 = reinterpret_cast<const uint32_t*>(col[1]);
                const auto* const c2 = reinterpret_cast<const uint32_t*>(col[2]);
                const auto* const c3 = reinterpret_cast<const uint32_t*>(col[3]);
                uint32_t* const o = panel + static_cast<int64_t>(kbase) * sme_fp_nr + j0;
                int k4 = 0;
                for (; k4 + 4 <= rows; k4 += 4) {
                    store_rows4(o + static_cast<int64_t>(k4) * sme_fp_nr, c0 + k4, c1 + k4,
                                c2 + k4, c3 + k4, sme_fp_nr);
                }
                for (; k4 < rows; ++k4) {
                    uint32_t* const row = o + static_cast<int64_t>(k4) * sme_fp_nr;
                    row[0] = c0[k4];
                    row[1] = c1[k4];
                    row[2] = c2[k4];
                    row[3] = c3[k4];
                }
            }
        }
    }
}

/// f16: one word per (column, K pair). The 32-k group is 16 pairs, i.e. four
/// 4x4 blocks — the same walk on half the words.
inline void pack_b_trans_f16(int n_start, int nc, int k_start, int kc, half* dst,
                             const uint8_t* src, int64_t row_blocks, float scale) noexcept
{
    const int64_t row_bytes = row_blocks * static_cast<int64_t>(kQ4Bytes);
    const int panels = (nc + sme_fp_nr - 1) / sme_fp_nr;
    const int k_pairs = (kc + 1) / 2;
    const int blk0 = k_start >> 5;
    alignas(16) half col[4][kQ4Elems];

    for (int nb = 0; nb < panels; ++nb) {
        const int n0 = nb * sme_fp_nr;
        const int cols = std::min(sme_fp_nr, nc - n0);
        const uint8_t* const base = src + static_cast<int64_t>(n_start + n0) * row_bytes;
        auto* const panel = reinterpret_cast<uint32_t*>(dst + static_cast<int64_t>(nb) * k_pairs *
                                                                   (2 * sme_fp_nr));
        for (int kb = 0; kb * kQ4Elems < kc; ++kb) {
            const int kbase = kb * kQ4Elems;
            const int rows = std::min(kQ4Elems, kc - kbase);
            const int pairs = (rows + 1) / 2;
            const int64_t blk = blk0 + kb;
            for (int j0 = 0; j0 < cols; j0 += 4) {
                for (int jj = 0; jj < 4; ++jj) {
                    const int j = j0 + jj;
                    if (j < cols && blk < row_blocks) {
                        decode_q4_1_block<half>(
                            base + static_cast<int64_t>(j) * row_bytes +
                                blk * static_cast<int64_t>(kQ4Bytes),
                            col[jj], scale);
                        // An odd k-block tails its last pair, which must read
                        // as zero.
                        for (int k = rows; k < kQ4Elems; ++k) {
                            col[jj][k] = half{};
                        }
                    } else {
                        std::fill(col[jj], col[jj] + kQ4Elems, half{});
                    }
                }
                const auto* const c0 = reinterpret_cast<const uint32_t*>(col[0]);
                const auto* const c1 = reinterpret_cast<const uint32_t*>(col[1]);
                const auto* const c2 = reinterpret_cast<const uint32_t*>(col[2]);
                const auto* const c3 = reinterpret_cast<const uint32_t*>(col[3]);
                uint32_t* const o =
                    panel + static_cast<int64_t>(kbase / 2) * sme_fp_nr + j0;
                int p4 = 0;
                for (; p4 + 4 <= pairs; p4 += 4) {
                    store_rows4(o + static_cast<int64_t>(p4) * sme_fp_nr, c0 + p4, c1 + p4,
                                c2 + p4, c3 + p4, sme_fp_nr);
                }
                for (; p4 < pairs; ++p4) {
                    uint32_t* const row = o + static_cast<int64_t>(p4) * sme_fp_nr;
                    row[0] = c0[p4];
                    row[1] = c1[p4];
                    row[2] = c2[p4];
                    row[3] = c3[p4];
                }
            }
        }
    }
}

// ---- scalar fallbacks: an unaligned k origin -----------------------------

inline void pack_b_trans_f32_scalar(int n_start, int nc, int k_start, int kc, float* dst,
                                    const uint8_t* src, int64_t row_blocks,
                                    float scale) noexcept
{
    const int64_t row_bytes = row_blocks * static_cast<int64_t>(kQ4Bytes);
    const int panels = (nc + sme_fp_nr - 1) / sme_fp_nr;
    const int blk0 = k_start >> 5;
    const int off0 = k_start & 31;
    alignas(16) float v[kQ4Elems];
    for (int nb = 0; nb < panels; ++nb) {
        const int n0 = nb * sme_fp_nr;
        const int cols = std::min(sme_fp_nr, nc - n0);
        const uint8_t* const base = src + static_cast<int64_t>(n_start + n0) * row_bytes;
        float* const panel = dst + static_cast<int64_t>(nb) * kc * sme_fp_nr;
        for (int j = 0; j < cols; ++j) {
            const uint8_t* const row = base + static_cast<int64_t>(j) * row_bytes;
            int done = 0;
            for (int b = blk0; done < kc; ++b) {
                if (b < row_blocks) {
                    decode_q4_1_block<float>(row + static_cast<int64_t>(b) * kQ4Bytes, v, scale);
                } else {
                    std::fill(v, v + kQ4Elems, 0.0f);
                }
                const int start = (b == blk0) ? off0 : 0;
                const int n = std::min(kQ4Elems - start, kc - done);
                for (int i = 0; i < n; ++i) {
                    panel[static_cast<int64_t>(done + i) * sme_fp_nr + j] = v[start + i];
                }
                done += n;
            }
        }
        // Columns past nc are never written by the sweep above.
        for (int k = 0; k < kc; ++k) {
            float* const row = panel + static_cast<int64_t>(k) * sme_fp_nr;
            std::fill(row + cols, row + sme_fp_nr, 0.0f);
        }
    }
}

inline void pack_b_trans_f16_scalar(int n_start, int nc, int k_start, int kc, half* dst,
                                    const uint8_t* src, int64_t row_blocks,
                                    float scale) noexcept
{
    const int64_t row_bytes = row_blocks * static_cast<int64_t>(kQ4Bytes);
    const int panels = (nc + sme_fp_nr - 1) / sme_fp_nr;
    const int k_pairs = (kc + 1) / 2;
    const int blk0 = k_start >> 5;
    const int off0 = k_start & 31;
    alignas(16) half v[kQ4Elems];
    for (int nb = 0; nb < panels; ++nb) {
        const int n0 = nb * sme_fp_nr;
        const int cols = std::min(sme_fp_nr, nc - n0);
        const uint8_t* const base = src + static_cast<int64_t>(n_start + n0) * row_bytes;
        half* const panel = dst + static_cast<int64_t>(nb) * k_pairs * (2 * sme_fp_nr);
        for (int j = 0; j < cols; ++j) {
            const uint8_t* const row = base + static_cast<int64_t>(j) * row_bytes;
            int done = 0;
            for (int b = blk0; done < kc; ++b) {
                if (b < row_blocks) {
                    decode_q4_1_block<half>(row + static_cast<int64_t>(b) * kQ4Bytes, v, scale);
                } else {
                    std::fill(v, v + kQ4Elems, half{});
                }
                const int start = (b == blk0) ? off0 : 0;
                const int n = std::min(kQ4Elems - start, kc - done);
                for (int i = 0; i < n; ++i) {
                    const int k = done + i;
                    panel[static_cast<int64_t>(k >> 1) * (2 * sme_fp_nr) + 2 * j + (k & 1)] =
                        v[start + i];
                }
                done += n;
            }
        }
        for (int p = 0; p < k_pairs; ++p) {
            half* const row = panel + static_cast<int64_t>(p) * (2 * sme_fp_nr);
            std::fill(row + 2 * cols, row + 2 * sme_fp_nr, half{});
        }
    }
}

} // namespace q4_sme_detail

/// Decode one k-block of a Q4_1 B into the ZA kernels' panel layout: f32
/// [n/32][k][32], f16 [n/32][ceil(kc/2)][64]. @p trans means B is physically
/// [N][K]; @p src_row_blocks is its row pitch in 20-byte blocks.
inline void pack_q4_1_b_sme(bool trans, int n_start, int nc, int k_start, int kc, float* dst,
                            const void* src, int64_t src_row_blocks, float scale) noexcept
{
    const auto* const base = static_cast<const uint8_t*>(src);
    if (!trans) {
        q4_sme_detail::pack_b_copy_f32(n_start, nc, k_start, kc, dst, base, src_row_blocks, scale);
    } else if ((k_start & 31) == 0) {
        q4_sme_detail::pack_b_trans_f32(n_start, nc, k_start, kc, dst, base, src_row_blocks,
                                        scale);
    } else {
        q4_sme_detail::pack_b_trans_f32_scalar(n_start, nc, k_start, kc, dst, base,
                                               src_row_blocks, scale);
    }
}

inline void pack_q4_1_b_sme(bool trans, int n_start, int nc, int k_start, int kc, half* dst,
                            const void* src, int64_t src_row_blocks, float scale) noexcept
{
    const auto* const base = static_cast<const uint8_t*>(src);
    if (!trans) {
        q4_sme_detail::pack_b_copy_f16(n_start, nc, k_start, kc, dst, base, src_row_blocks, scale);
    } else if ((k_start & 31) == 0) {
        q4_sme_detail::pack_b_trans_f16(n_start, nc, k_start, kc, dst, base, src_row_blocks,
                                        scale);
    } else {
        q4_sme_detail::pack_b_trans_f16_scalar(n_start, nc, k_start, kc, dst, base,
                                               src_row_blocks, scale);
    }
}

} // namespace nnops::backend::cpu::aarch64

#endif // NNOPS_ARCH_SME
