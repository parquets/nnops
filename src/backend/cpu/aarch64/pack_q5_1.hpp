#pragma once
/// @file pack_q5_1.hpp
/// @brief AArch64 NEON Q5_1 decode and packed-B kernels.
///
/// Q5_1 is llama.cpp's block format: 32 values in 24 bytes — an fp16 delta `d`,
/// an fp16 minimum `m`, a 4-byte mask holding the 5th bit of every quant, and
/// 16 bytes of nibbles (low nibble of byte j is value j, high nibble is value
/// j + 16). Value j is `d * q + m` with q in [0, 31]. The block layout and the
/// portable scalar reference live in <nnops/detail/q5_1.hpp>.
///
/// `pack_q5_1_rhs` is the GEMM-side entry point: it fuses unpack + dequantize
/// + f16 narrowing into the RHS panel pack, so a Q5_1 weight never passes
/// through an intermediate dense buffer. It writes the same uniform
/// `nr_max`-wide, `[k][nr]`-major panels the f32/f16 packers write, which is
/// what lets MatMul treat a Q5_1 B as just another packed-B source.
///
/// Unlike x86_64, whose f32 and f16 RHS panels are both 16 wide, AArch64's are
/// nr_f32[0] = 12 and nr_f16[0] = 16, so the panel width here is a per element
/// type constant rather than one number.
///
/// The pad lanes of a tail panel (past its live width) are zero-filled here,
/// unlike the transposing f32/f16 packers, which alias the last live row: a
/// Q5_1 decode has no row to alias, and the MMA kernels only store live
/// columns anyway.

#include <arm_neon.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "backend/cpu/aarch64/pack_f16.hpp"
#include "backend/cpu/aarch64/pack_f32.hpp"
#include "backend/cpu/common/common_utils.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/q5_1.hpp"

namespace nnops::backend::cpu::aarch64 {

using nnops::backend::cpu::half;
using nnops::backend::cpu::half_from_bits;
using nnops::backend::cpu::half_to_float;

/// The eight bits of one qh byte as 0/1 lanes: lane i takes bit i.
///
/// Lane i keeps only its own bit with a per-lane power-of-two mask, and `tst`
/// turns the survivors into 0 or 1 — the counterpart of x86_64's `srlv`, but
/// without a variable shift, whose negative-count behaviour is the sort of
/// thing that silently reads the wrong bit on a target that cannot be tested
/// here.
inline uint16x8_t q5_1_spread_bits(uint32_t byte) noexcept {
    static const uint8_t pow2_data[8] = {1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x8_t b    = vdup_n_u8(static_cast<uint8_t>(byte));
    const uint8x8_t pow2 = vld1_u8(pow2_data);
    const uint8x8_t held = vand_u8(b, pow2);              // 0 or 1<<lane
    const uint8x8_t one  = vand_u8(vtst_u8(held, held), vdup_n_u8(1)); // 0/1
    return vmovl_u8(one);
}

/// Decode one block into 32 f32 or f16 values (f16 rounds once per element).
///
/// The nibble bytes are widened once into four 8-lane halves of 16-bit quants;
/// the 5th-bit mask is spread in 16-bit lanes, so the widening to 32 bits —
/// which the float conversion needs — happens once per group, after the mask
/// has been merged in.
///
/// @param scale multiplies the whole block: the decode is `q·d + m`, so scaling
///              `d` and `m` once per block scales all 32 values — MatMul's alpha
///              without a per-element pass.
template <class T>
inline void decode_q5_1_block(const uint8_t* p, T* out,
                              float scale = 1.0f) noexcept {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>,
                  "decode_q5_1_block: T must be float or half");

    uint16_t d_bits, m_bits;
    uint32_t qh;
    std::memcpy(&d_bits, p + 0, sizeof(d_bits));
    std::memcpy(&m_bits, p + 2, sizeof(m_bits));
    std::memcpy(&qh,     p + 4, sizeof(qh));

    float d = half_to_float(half_from_bits(d_bits));
    float m = half_to_float(half_from_bits(m_bits));
    if (scale != 1.0f) {
        d *= scale;
        m *= scale;
    }
    const float32x4_t vd = vdupq_n_f32(d);
    const float32x4_t vm = vdupq_n_f32(m);

    const uint8x16_t qs = vld1q_u8(p + 8);
    const uint8x16_t lo = vandq_u8(qs, vdupq_n_u8(0x0F));
    const uint8x16_t hi = vshrq_n_u8(qs, 4);

    // One group is eight consecutive values: q·d + m for the merged quant, then
    // the f32 or f16 store. Value j takes bit j of qh, so the four groups walk
    // qh's four bytes in order.
    auto emit = [&](int g, uint16x8_t nib, uint32_t bits) {
        const uint16x8_t q16 =
            vorrq_u16(nib, vshlq_n_u16(q5_1_spread_bits(bits), 4));
        const float32x4_t v0 =
            vfmaq_f32(vm, vcvtq_f32_u32(vmovl_u16(vget_low_u16(q16))), vd);
        const float32x4_t v1 =
            vfmaq_f32(vm, vcvtq_f32_u32(vmovl_u16(vget_high_u16(q16))), vd);
        if constexpr (std::is_same_v<T, float>) {
            vst1q_f32(out + 8 * g,     v0);
            vst1q_f32(out + 8 * g + 4, v1);
        } else {
            vst1q_f16(reinterpret_cast<float16_t*>(out) + 8 * g,
                      vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1)));
        }
    };

    emit(0, vmovl_u8(vget_low_u8(lo)),  qh         & 0xFFu);  // values  0..7
    emit(1, vmovl_u8(vget_high_u8(lo)), (qh >> 8)  & 0xFFu);  // values  8..15
    emit(2, vmovl_u8(vget_low_u8(hi)),  (qh >> 16) & 0xFFu);  // values 16..23
    emit(3, vmovl_u8(vget_high_u8(hi)),  qh >> 24);           // values 24..31
}

// =========================================================================
//  Packed-B panels
// =========================================================================
//
// Panels are the uniform `nr_max`-wide kind the mrxncxkc MMA kernels read:
// ceil(nc/nr_max) panels of `kc` rows, each row `nr_max` live lanes, all at
// the caller's aligned panel stride `ldd` (elements). A tail panel's lanes
// past its live width are zero-filled.

/// Panel width the MatMul pack route uses on aarch64, per element type:
/// nr_f32[0] = 12 for f32, nr_f16[0] = 16 for f16.
template <class T>
inline constexpr int q5_1_nr_max = std::is_same_v<T, float> ? nr_f32[0]
                                                             : nr_f16[0];

/// Row-major B (phys [K][N]): panel columns [n_start, n_start + width) of rows
/// [k_start, k_start + kc). A panel can straddle two 32-value blocks, so at
/// most two block decodes feed one row. @p width is ignored when NR > 0.
template <class T, int NR>
void pack_q5_1_copy_panel(T* out, const uint8_t* src, int64_t row_blocks,
                          int k_start, int kc, int n_start, int width,
                          float scale = 1.0f) noexcept {
    constexpr int kPanel = q5_1_nr_max<T>;
    const int nb    = (NR > 0) ? NR : width;
    const int blk0  = n_start >> 5;
    const int off0  = n_start & 31;
    const int first = std::min(nb, 32 - off0);
    const bool two  = (nb > first);
    const int64_t row_bytes = row_blocks * static_cast<int64_t>(q5_1_block_bytes);

    T v0[q5_1_block_elems];
    T v1[q5_1_block_elems];
    for (int k = 0; k < kc; ++k) {
        const uint8_t* row = src + static_cast<int64_t>(k_start + k) * row_bytes;
        T* o = out + static_cast<int64_t>(k) * kPanel;

        decode_q5_1_block<T>(row + static_cast<int64_t>(blk0) * q5_1_block_bytes, v0,
                             scale);
        for (int j = 0; j < first; ++j) {
            o[j] = v0[off0 + j];
        }
        if (two) {
            decode_q5_1_block<T>(
                row + static_cast<int64_t>(blk0 + 1) * q5_1_block_bytes, v1, scale);
            for (int j = first; j < nb; ++j) {
                o[j] = v1[j - first];
            }
        }
        for (int j = nb; j < kPanel; ++j) {
            o[j] = T{};
        }
    }
}

/// Transposed B (phys [N][K]): panel rows [n_start, n_start + width) of columns
/// [k_start, k_start + kc). Each source row is swept linearly, so every block
/// decode yields up to 32 consecutive k values — the efficient direction for
/// Linear's [N][K] weights.
template <class T, int NR>
void pack_q5_1_trans_panel(T* out, const uint8_t* src, int64_t row_blocks,
                           int k_start, int kc, int n_start, int width,
                           float scale = 1.0f) noexcept {
    constexpr int kPanel = q5_1_nr_max<T>;
    const int nb     = (NR > 0) ? NR : width;
    const int blk0   = k_start >> 5;
    const int off0   = k_start & 31;
    const int nblk   = (off0 + kc + 31) >> 5;
    const int64_t row_bytes = row_blocks * static_cast<int64_t>(q5_1_block_bytes);

    T vals[q5_1_block_elems];
    for (int j = 0; j < nb; ++j) {
        const uint8_t* row = src + static_cast<int64_t>(n_start + j) * row_bytes
                           + static_cast<int64_t>(blk0) * q5_1_block_bytes;

        int k = 0;
        for (int b = 0; b < nblk; ++b) {
            decode_q5_1_block<T>(row + static_cast<int64_t>(b) * q5_1_block_bytes,
                                 vals, scale);
            const int start = (b == 0) ? off0 : 0;
            const int count = std::min(32 - start, kc - k);
            T* o = out + static_cast<int64_t>(k) * kPanel + j;
            for (int i = 0; i < count; ++i) {
                o[static_cast<int64_t>(i) * kPanel] = vals[start + i];
            }
            k += count;
        }
    }

    // Lanes past the live width: never read, never stored to C, but keep the
    // panel fully defined so the workspace is deterministic.
    for (int k = 0; k < kc; ++k) {
        T* o = out + static_cast<int64_t>(k) * kPanel;
        for (int j = nb; j < kPanel; ++j) {
            o[j] = T{};
        }
    }
}

/// Pack the Q5_1 RHS into uniform panels for one k-block.
///
/// @param trans      true when B is physically [N][K], false for [K][N].
/// @param n_start    first logical column of this tile.
/// @param nc         live columns in the tile.
/// @param k_start    first logical k of the k-block.
/// @param kc         k length of the block.
/// @param dst        first panel; panels advance by @p ldd elements.
/// @param ldd        aligned panel stride in elements (nr_max * kc rounded).
/// @param src        first block of the batch element's B.
/// @param row_blocks 24-byte blocks per physical row (the B row pitch / 24).
/// @param scale      MatMul alpha. It folds into each block's `d` and `m`
///                   (two multiplies per 32 weights), so the pack itself does
///                   no extra work per element.
template <class T>
void pack_q5_1_rhs(bool trans, int n_start, int nc, int k_start, int kc,
                   T* dst, int ldd, const void* src, int64_t row_blocks,
                   float scale) noexcept {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>,
                  "pack_q5_1_rhs: T must be float or half");

    constexpr int kPanel = q5_1_nr_max<T>;
    const uint8_t* base = static_cast<const uint8_t*>(src);

    int n = 0;
    for (; n + kPanel <= nc; n += kPanel) {
        if (trans) {
            pack_q5_1_trans_panel<T, kPanel>(
                dst, base, row_blocks, k_start, kc, n_start + n, 0, scale);
        } else {
            pack_q5_1_copy_panel<T, kPanel>(
                dst, base, row_blocks, k_start, kc, n_start + n, 0, scale);
        }
        dst += ldd;
    }
    if (n < nc) {
        if (trans) {
            pack_q5_1_trans_panel<T, 0>(
                dst, base, row_blocks, k_start, kc, n_start + n, nc - n, scale);
        } else {
            pack_q5_1_copy_panel<T, 0>(
                dst, base, row_blocks, k_start, kc, n_start + n, nc - n, scale);
        }
    }
}

}  // namespace nnops::backend::cpu::aarch64
