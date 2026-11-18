#pragma once
/// @file pack_f16_sme.hpp
/// @brief AArch64 SME float16 pack kernels for MMA LHS/RHS.
///
/// Mirrors pack_f32_sme.hpp on half, with the K-pair interleave the hardware
/// two-way dot needs: the MOPA reads lane pairs, so A stores [row][k,k+1] and B
/// [col][k,k+1] already zipped, and the kernel's K loop is a straight load.
/// Building the interleave in the loop instead costs five permutes per K pair
/// that contend with the MOPA issue slot.
///   - A: [m/16][ceil(kc/2)][32], lane 2i and 2i+1 holding row i's K pair.
///   - B: [n/64][ceil(kc/2)][128], column j's K pair at j*2 and j*2+1.
/// An odd kc pads the last K pair with zeros at pack time; padding columns past
/// @p nc are zeroed too.
///
/// `trans` reads the physical K×M (A) / N×K (B) layout, which is what MatMul's
/// transpose_a / transpose_b describe.

#if defined(NNOPS_ARCH_SME)

#include <algorithm>
#include <cstdint>

#include <arm_neon.h>

#include "nnops/detail/half.hpp"

#include "backend/cpu/aarch64/mma_sme.hpp"

namespace nnops::backend::cpu::aarch64 {

/// Elements one packed A slab occupies: [ceil(mc/16)][ceil(kc/2)][32].
inline int64_t packed_a_elems_f16_sme(int mc, int kc) noexcept
{
    return static_cast<int64_t>((mc + sme_mr - 1) / sme_mr) * ((kc + 1) / 2) * (2 * sme_mr);
}

/// Elements one packed B slab occupies: [ceil(nc/64)][ceil(kc/2)][128].
inline int64_t packed_b_elems_f16_sme(int nc, int kc) noexcept
{
    return static_cast<int64_t>((nc + sme_nr_max - 1) / sme_nr_max) * ((kc + 1) / 2) *
           (2 * sme_nr_max);
}

/// Pack @p mc rows of A (row stride @p lds elements) into
/// [m/16][ceil(kc/2)][32]; an odd K tail is zeroed.
inline void pack_a_f16_sme(int mc, int kc, half* dst, const half* src, int64_t lds,
                           bool trans = false) noexcept
{
    const int k_pairs = (kc + 1) / 2;
    for (int m = 0; m < mc; ++m) {
        half* out = dst + static_cast<int64_t>(m / sme_mr) * k_pairs * (2 * sme_mr) +
                    (m % sme_mr) * 2;
        const half* in = src + static_cast<int64_t>(m) * lds;
        for (int kp = 0; kp < k_pairs; ++kp) {
            const int k = 2 * kp;
            half* lane = out + static_cast<int64_t>(kp) * 2 * sme_mr;
            lane[0] = trans ? src[static_cast<int64_t>(k) * lds + m] : in[k];
            lane[1] = k + 1 < kc ? (trans ? src[static_cast<int64_t>(k + 1) * lds + m]
                                          : in[k + 1])
                                 : half{};
        }
    }
}

/// Pack @p nc columns of B (row-major, stride @p ldb, @p kc rows) into
/// [n/64][ceil(kc/2)][128]; an odd K tail and padding columns are zeroed.
inline void pack_b_f16_sme(int nc, int kc, half* dst, const half* src, int64_t ldb,
                           bool trans = false) noexcept
{
    const int n_blocks = (nc + sme_nr_max - 1) / sme_nr_max;
    const int k_pairs = (kc + 1) / 2;
    for (int nb = 0; nb < n_blocks; ++nb) {
        const int n0 = nb * sme_nr_max;
        const int cols = std::min(sme_nr_max, nc - n0);
        half* blk = dst + static_cast<int64_t>(nb) * k_pairs * (2 * sme_nr_max);
        for (int kp = 0; kp < k_pairs; ++kp) {
            const int k = 2 * kp;
            half* out = blk + static_cast<int64_t>(kp) * 2 * sme_nr_max;
            if (trans) {
                for (int j = 0; j < cols; ++j) {
                    out[2 * j] = src[static_cast<int64_t>(n0 + j) * ldb + k];
                    out[2 * j + 1] =
                        k + 1 < kc ? src[static_cast<int64_t>(n0 + j) * ldb + k + 1] : half{};
                }
            } else {
                // Rows k and k+1 are contiguous, so the pair interleave is one
                // zip of 8 columns at a time. This pack is on the critical path
                // of a prefill (small M), where B's traffic is not amortised.
                const half* in0 = src + static_cast<int64_t>(k) * ldb + n0;
                const half* in1 = k + 1 < kc ? in0 + ldb : nullptr;
                int j = 0;
                for (; j + 8 <= cols; j += 8) {
                    const float16x8_t a =
                        vld1q_f16(reinterpret_cast<const float16_t*>(in0 + j));
                    const float16x8_t b =
                        in1 ? vld1q_f16(reinterpret_cast<const float16_t*>(in1 + j)) :
                              vdupq_n_f16(0.0f);
                    vst1q_f16(reinterpret_cast<float16_t*>(out + 2 * j), vzip1q_f16(a, b));
                    vst1q_f16(reinterpret_cast<float16_t*>(out + 2 * j + 8), vzip2q_f16(a, b));
                }
                for (; j < cols; ++j) {
                    out[2 * j] = in0[j];
                    out[2 * j + 1] = in1 ? in1[j] : half{};
                }
            }
            for (int j = cols; j < sme_nr_max; ++j) {
                out[2 * j] = half{};
                out[2 * j + 1] = half{};
            }
        }
    }
}

} // namespace nnops::backend::cpu::aarch64

#endif // NNOPS_ARCH_SME
