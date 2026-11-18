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

#include "backend/cpu/aarch64/mma_sme.hpp"

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

/// Pack @p mc rows of A (row stride @p lds elements) into
/// [m/16][ceil(kc/4)][64].
inline void pack_a_i8_sme(int mc, int kc, int8_t* dst, const int8_t* src, int64_t lds,
                          bool trans, bool xor_bias) noexcept
{
    const int k_groups = (kc + 3) / 4;
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
