#pragma once
/// @file pack_f32_sme.hpp
/// @brief AArch64 SME float32 pack kernels for MMA LHS/RHS.
///
/// Repacks matrix rows/columns into the layouts mma_f32_sme.hpp reads. Both
/// transforms write k-major blocks whose 16 (A) / 64 (B) values per k are
/// contiguous, not the NEON pack's 12/16-column panels:
///   - A: [m/16][kc][16], the 16 rows of an M tile contiguous per k. Padding
///     rows past @p mc are left undefined: the kernel masks them.
///   - B: [n/64][kc][64], 64 contiguous columns per k. Padding columns past
///     @p nc are zeroed by the pack; the kernel's predicates keep them out of
///     the accumulator anyway.
///
/// `trans` reads the physical K×M (A) / N×K (B) layout, which is what MatMul's
/// transpose_a / transpose_b describe.

#if defined(NNOPS_ARCH_SME)

#include <algorithm>
#include <cstdint>

#include "backend/cpu/aarch64/mma_sme.hpp"

namespace nnops::backend::cpu::aarch64 {

/// Elements one packed A slab occupies: [ceil(mc/16)][kc][16].
inline int64_t packed_a_elems_f32_sme(int mc, int kc) noexcept
{
    return static_cast<int64_t>((mc + sme_mr - 1) / sme_mr) * kc * sme_mr;
}

/// Elements one packed B slab occupies: [ceil(nc/64)][kc][64].
inline int64_t packed_b_elems_f32_sme(int nc, int kc) noexcept
{
    return static_cast<int64_t>((nc + sme_nr_max - 1) / sme_nr_max) * kc * sme_nr_max;
}

/// Pack @p mc rows of A (row stride @p lds elements) into [m/16][kc][16].
inline void pack_a_f32_sme(int mc, int kc, float* dst, const float* src, int64_t lds,
                           bool trans = false) noexcept
{
    for (int m = 0; m < mc; ++m) {
        float* out = dst + (static_cast<int64_t>(m / sme_mr) * kc) * sme_mr + (m % sme_mr);
        if (trans) {
            // A's physical layout is K×M: row k, column m.
            for (int k = 0; k < kc; ++k) {
                out[static_cast<int64_t>(k) * sme_mr] = src[static_cast<int64_t>(k) * lds + m];
            }
        } else {
            const float* in = src + static_cast<int64_t>(m) * lds;
            for (int k = 0; k < kc; ++k) {
                out[static_cast<int64_t>(k) * sme_mr] = in[k];
            }
        }
    }
}

/// Pack @p nc columns of B (row-major, stride @p ldb, @p kc rows) into
/// [n/64][kc][64]; padding columns are zeroed.
inline void pack_b_f32_sme(int nc, int kc, float* dst, const float* src, int64_t ldb,
                           bool trans = false) noexcept
{
    const int n_blocks = (nc + sme_nr_max - 1) / sme_nr_max;
    for (int nb = 0; nb < n_blocks; ++nb) {
        const int n0 = nb * sme_nr_max;
        const int cols = std::min(sme_nr_max, nc - n0);
        float* blk = dst + static_cast<int64_t>(nb) * kc * sme_nr_max;
        for (int k = 0; k < kc; ++k) {
            float* out = blk + static_cast<int64_t>(k) * sme_nr_max;
            if (trans) {
                // B's physical layout is N×K: row n, column k.
                for (int j = 0; j < cols; ++j) {
                    out[j] = src[static_cast<int64_t>(n0 + j) * ldb + k];
                }
            } else {
                const float* in = src + static_cast<int64_t>(k) * ldb + n0;
                for (int j = 0; j < cols; ++j) {
                    out[j] = in[j];
                }
            }
            for (int j = cols; j < sme_nr_max; ++j) {
                out[j] = 0.0f;
            }
        }
    }
}

} // namespace nnops::backend::cpu::aarch64

#endif // NNOPS_ARCH_SME
