#pragma once
/// @file pack_i8_sme.hpp
/// @brief AArch64 SME int8 pack kernels for MMA LHS/RHS.
///
/// One `SMOPA ZA.S` is a 4-way dot: 16 rows x 16 columns x 4 K. The pack
/// interleaves K in groups of four, which is how the instruction reads its
/// lanes: operand lane 4i..4i+3 are row (column) i's four K values.
///   - A: [m/16][ceil(kc/4)][64]
///   - B: [n/64][ceil(kc/4)][256], four 16-column panels of 16 columns x 4 K.
/// Padding K slots and columns past @p nc are zeroed.
///
/// `xor_bias` applies the u8-activation shift (XOR 0x80) the NEON pack uses;
/// padded K slots stay 0, which is the identity in the shifted domain.
///
/// `trans` reads the physical K×M (A) / N×K (B) layout, which is what MatMul's
/// transpose_a / transpose_b describe.

#if defined(NNOPS_ARCH_SME)

#include <algorithm>
#include <cstdint>

#include <arm_neon.h>

#include "backend/cpu/aarch64/sme/mma_sme.hpp"

namespace nnops::backend::cpu::aarch64 {

/// Elements one packed A slab occupies: [ceil(mc/16)][ceil(kc/4)][64].
inline int64_t packed_a_elems_i8_sme(int mc, int kc) noexcept
{
    return static_cast<int64_t>((mc + sme_mr - 1) / sme_mr) * ((kc + 3) / 4) * 64;
}

/// Elements one packed B slab occupies:
/// [ceil(nc/64)][ceil(kc/4)][256].
inline int64_t packed_b_elems_i8_sme(int nc, int kc) noexcept
{
    return static_cast<int64_t>((nc + sme_nr_max - 1) / sme_nr_max) * ((kc + 3) / 4) *
           (4 * sme_nr_max);
}

#if defined(NNOPS_ARCH_SME)
/// The A pack through ZA: one K group (four K bytes) is a 32-bit word and the
/// destination is [k group][row] words, so this is the same 16x16 word
/// transpose the float packers use — rows in horizontally, groups out
/// vertically. Needs a whole number of K groups in a word-aligned row.
__arm_locally_streaming __arm_new("za") __attribute__((target("+sme2")))
inline void pack_a_i8_sme_za(int mc, int kc, int8_t* dst, const int8_t* src,
                             int64_t lds) noexcept
{
    const int k_groups = kc / 4;
    const auto* const sw = reinterpret_cast<const uint32_t*>(src);
    const int64_t lds_w = lds / 4;
    for (int m0 = 0; m0 < mc; m0 += sme_mr) {
        const int rows = std::min(sme_mr, mc - m0);
        auto* const blk = reinterpret_cast<uint32_t*>(
            dst + static_cast<int64_t>(m0 / sme_mr) * k_groups * 64);
        for (int g0 = 0; g0 < k_groups; g0 += sme_mr) {
            const int glive = std::min(sme_mr, k_groups - g0);
            const svbool_t pk = svwhilelt_b32(0, glive);
            for (int i = 0; i < rows; ++i) {
                svld1_hor_za32(0, static_cast<uint32_t>(i), pk,
                               sw + static_cast<int64_t>(m0 + i) * lds_w + g0);
            }
            const svbool_t pr = svwhilelt_b32(0, rows);
            for (int j = 0; j < glive; ++j) {
                svst1_ver_za32(0, static_cast<uint32_t>(j), pr,
                               blk + static_cast<int64_t>(g0 + j) * sme_mr);
            }
        }
    }
}
#endif

/// Pack @p mc rows of A (row stride @p lds elements) into
/// [m/16][ceil(kc/4)][64].
inline void pack_a_i8_sme(int mc, int kc, int8_t* dst, const int8_t* src, int64_t lds,
                          bool trans, bool xor_bias) noexcept
{
    const int k_groups = (kc + 3) / 4;
#if defined(NNOPS_ARCH_SME)
    if (!trans && !xor_bias && (kc & 3) == 0 && (lds & 3) == 0 &&
        (reinterpret_cast<uintptr_t>(src) & 3) == 0 && available_sme()) {
        pack_a_i8_sme_za(mc, kc, dst, src, lds);
        return;
    }
#endif
    for (int m = 0; m < mc; ++m) {
        int8_t* out = dst + static_cast<int64_t>(m / sme_mr) * k_groups * 64 + (m % sme_mr) * 4;
        for (int g = 0; g < k_groups; ++g) {
            for (int p = 0; p < 4; ++p) {
                const int k = 4 * g + p;
                int8_t v = 0;
                if (k < kc) {
                    v = trans ? src[static_cast<int64_t>(k) * lds + m] :
                                src[static_cast<int64_t>(m) * lds + k];
                    if (xor_bias) { v = static_cast<int8_t>(static_cast<uint8_t>(v) ^ 0x80u); }
                }
                out[static_cast<int64_t>(g) * 64 + p] = v;
            }
        }
    }
}

/// Pack @p nc columns of B (row-major, stride @p ldb, @p kc rows) into
/// [n/64][ceil(kc/4)][256].
inline void pack_b_i8_sme(int nc, int kc, int8_t* dst, const int8_t* src, int64_t ldb,
                          bool trans) noexcept
{
    const int n_blocks = (nc + sme_nr_max - 1) / sme_nr_max;
    const int k_groups = (kc + 3) / 4;
#if defined(__ARM_NEON)
    // A k group is four source rows; 16 columns at a time transpose to the 64
    // bytes the kernel reads (four K values per column, 16 columns a panel).
    // Eight zips replace 64 byte copies.
    if (!trans) {
        const int full_g = kc / 4;                 // groups with all four K
        const int full_n = nc / sme_za_n * sme_za_n;
        for (int nb = 0; nb < n_blocks; ++nb) {
            const int n0 = nb * sme_nr_max;
            const int cols = std::min(sme_nr_max, nc - n0);
            int8_t* const blk = dst + static_cast<int64_t>(nb) * k_groups * (4 * sme_nr_max);
            for (int g = 0; g < k_groups; ++g) {
                int8_t* const out = blk + static_cast<int64_t>(g) * (4 * sme_nr_max);
                const int k = 4 * g;
                int j = 0;
                if (g < full_g) {
                    for (; j + sme_za_n <= cols; j += sme_za_n) {
                        const int8_t* const p = src + static_cast<int64_t>(k) * ldb + n0 + j;
                        const uint8x16_t r0 = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
                        const uint8x16_t r1 =
                            vld1q_u8(reinterpret_cast<const uint8_t*>(p + ldb));
                        const uint8x16_t r2 =
                            vld1q_u8(reinterpret_cast<const uint8_t*>(p + 2 * ldb));
                        const uint8x16_t r3 =
                            vld1q_u8(reinterpret_cast<const uint8_t*>(p + 3 * ldb));
                        const uint8x16_t a_lo = vzip1q_u8(r0, r1);
                        const uint8x16_t a_hi = vzip2q_u8(r0, r1);
                        const uint8x16_t b_lo = vzip1q_u8(r2, r3);
                        const uint8x16_t b_hi = vzip2q_u8(r2, r3);
                        uint8_t* const o = reinterpret_cast<uint8_t*>(out) +
                                           static_cast<int64_t>(j / sme_za_n) * 64;
                        vst1q_u8(o, vzip1q_u16(a_lo, b_lo));
                        vst1q_u8(o + 16, vzip2q_u16(a_lo, b_lo));
                        vst1q_u8(o + 32, vzip1q_u16(a_hi, b_hi));
                        vst1q_u8(o + 48, vzip2q_u16(a_hi, b_hi));
                    }
                }
                for (; j < sme_nr_max; ++j) {
                    uint8_t* const cell = reinterpret_cast<uint8_t*>(out) +
                                          static_cast<int64_t>(j / sme_za_n) * 64 +
                                          (j % sme_za_n) * 4;
                    for (int p = 0; p < 4; ++p) {
                        const int kk = k + p;
                        cell[p] = (j < cols && kk < kc) ?
                                      src[static_cast<int64_t>(kk) * ldb + n0 + j] :
                                      0;
                    }
                }
            }
        }
        return;
    }
#endif
    if (trans) {
        // The transposed source's four K bytes of one column are already one
        // contiguous word, so a k group of 16 columns is simply those 16 words
        // in order — no byte transpose. The scalar walk below does 64 byte
        // loads and 64 byte stores for the same 64 bytes.
        const int full_g = kc / 4;
        for (int nb = 0; nb < n_blocks; ++nb) {
            const int n0 = nb * sme_nr_max;
            const int cols = std::min(sme_nr_max, nc - n0);
            int8_t* const blk = dst + static_cast<int64_t>(nb) * k_groups * (4 * sme_nr_max);
            for (int g = 0; g < k_groups; ++g) {
                int8_t* const out = blk + static_cast<int64_t>(g) * (4 * sme_nr_max);
                const int k = 4 * g;
                int j = 0;
                if (g < full_g) {
                    for (; j + 16 <= cols; j += 16) {
                        const int8_t* const srow = src + static_cast<int64_t>(n0 + j) * ldb + k;
                        int8_t* const o = out + static_cast<int64_t>(j / 16) * 64;
                        for (int jj = 0; jj < 16; ++jj) {
                            uint32_t w;
                            __builtin_memcpy(&w, srow + static_cast<int64_t>(jj) * ldb, 4);
                            __builtin_memcpy(o + jj * 4, &w, 4);
                        }
                    }
                }
                for (; j < sme_nr_max; ++j) {
                    int8_t* const cell = out + static_cast<int64_t>(j / 16) * 64 + (j % 16) * 4;
                    for (int p = 0; p < 4; ++p) {
                        const int kk = k + p;
                        cell[p] = (j < cols && kk < kc) ?
                                      src[static_cast<int64_t>(n0 + j) * ldb + kk] :
                                      0;
                    }
                }
            }
        }
        return;
    }
    for (int nb = 0; nb < n_blocks; ++nb) {
        const int n0 = nb * sme_nr_max;
        const int cols = std::min(sme_nr_max, nc - n0);
        int8_t* blk = dst + static_cast<int64_t>(nb) * k_groups * (4 * sme_nr_max);
        for (int g = 0; g < k_groups; ++g) {
            int8_t* out = blk + static_cast<int64_t>(g) * (4 * sme_nr_max);
            for (int j = 0; j < sme_nr_max; ++j) {
                int8_t* cell = out + static_cast<int64_t>(j / sme_za_n) * (4 * sme_za_n) +
                               (j % sme_za_n) * 4;
                for (int p = 0; p < 4; ++p) {
                    const int k = 4 * g + p;
                    int8_t v = 0;
                    if (j < cols && k < kc) {
                        v = trans ? src[static_cast<int64_t>(n0 + j) * ldb + k] :
                                    src[static_cast<int64_t>(k) * ldb + n0 + j];
                    }
                    cell[p] = v;
                }
            }
        }
    }
}

} // namespace nnops::backend::cpu::aarch64

#endif // NNOPS_ARCH_SME
