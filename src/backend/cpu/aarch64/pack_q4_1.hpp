#pragma once
/// @file pack_q4_1.hpp
/// @brief AArch64 NEON Q4_1 decode and packed-B kernels.
///
/// Q4_1 is llama.cpp's affine 4-bit block format: 32 values in 20 bytes — an
/// fp16 delta `d`, an fp16 minimum `m`, then 16 nibble bytes (byte j's low
/// nibble is value j, its high nibble value j + 16). Value j is `d * q + m`
/// with q in [0, 15]. The layout and the portable scalar reference live in
/// <nnops/detail/q4_1.hpp>.
///
/// `pack_q4_1_rhs` is the GEMM-side entry point: it fuses decode + the
/// narrowing into the RHS panel pack, so a Q4_1 weight never passes through an
/// intermediate dense buffer. For f32/f16 it writes the same uniform
/// `nr_max`-wide, `[k][nr]`-major panels the f32/f16 packers write, which is
/// what lets MatMul treat a Q4_1 B as just another packed-B source — the MMA
/// route knows nothing about the format. `scale` (MatMul's alpha) folds into
/// each block's `d` and `m`: two multiplies per 32 values, not one per element.
///
/// bf16 has its own destination order: its kernel reads pack_bf16.hpp's BFMMLA
/// operand layout (⌈kc/4⌉ k-groups of 64 elements, a colpair holding
/// {col 2c k0..3, col 2c+1 k0..3}), so the bf16 overload writes that directly
/// rather than the `[k][nr]` rectangle. The decode is the same fused one.
///
/// The panel width is a per element type constant here: aarch64's nr_f32[0] is
/// 12, nr_f16[0] is 16 and the bf16 panel is 16 columns.
///
/// A tail panel's lanes past its live width are zero-filled, unlike the
/// transposing f32/f16 packers, which alias the last live row: a Q4_1 decode has
/// no row to alias, and the MMA kernels only store live columns anyway. The
/// bf16 layout additionally pads a short final k-group to four lanes.

#include <arm_neon.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "backend/cpu/aarch64/pack_bf16.hpp"
#include "backend/cpu/aarch64/pack_f16.hpp"
#include "backend/cpu/aarch64/pack_f32.hpp"
#include "backend/cpu/common/common_utils.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/q4_1.hpp"

namespace nnops::backend::cpu::aarch64 {

using nnops::backend::cpu::half;
using nnops::backend::cpu::half_from_bits;
using nnops::backend::cpu::half_to_float;

/// Decode one block into 32 f32, f16 or bf16 values; an f16/bf16 destination
/// rounds once per element.
template <class T>
inline void decode_q4_1_block(const uint8_t* p, T* out,
                              float scale = 1.0f) noexcept {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, half> ||
                      std::is_same_v<T, bfloat16>,
                  "decode_q4_1_block: T must be float, half or bfloat16");

    uint16_t d_bits, m_bits;
    std::memcpy(&d_bits, p + 0, sizeof(d_bits));
    std::memcpy(&m_bits, p + 2, sizeof(m_bits));

    float d = half_to_float(half_from_bits(d_bits));
    float m = half_to_float(half_from_bits(m_bits));
    if (scale != 1.0f) {
        d *= scale;
        m *= scale;
    }
    const float32x4_t vd = vdupq_n_f32(d);
    const float32x4_t vm = vdupq_n_f32(m);

    const uint8x16_t qs = vld1q_u8(p + 4);
    const uint8x16_t lo = vandq_u8(qs, vdupq_n_u8(0x0F));
    const uint8x16_t hi = vshrq_n_u8(qs, 4);

    auto emit = [&](int g, uint16x8_t nib) {
        const float32x4_t v0 =
            vfmaq_f32(vm, vcvtq_f32_u32(vmovl_u16(vget_low_u16(nib))), vd);
        const float32x4_t v1 =
            vfmaq_f32(vm, vcvtq_f32_u32(vmovl_u16(vget_high_u16(nib))), vd);
        if constexpr (std::is_same_v<T, float>) {
            vst1q_f32(out + 8 * g,     v0);
            vst1q_f32(out + 8 * g + 4, v1);
        } else if constexpr (std::is_same_v<T, half>) {
            vst1q_f16(reinterpret_cast<float16_t*>(out) + 8 * g,
                      vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1)));
        } else {
            vst1q_u16(reinterpret_cast<uint16_t*>(out + 8 * g),
                      bf16_narrow(::nnops::simd::v_f32x8(v0, v1)));
        }
    };

    emit(0, vmovl_u8(vget_low_u8(lo)));   // values  0..7
    emit(1, vmovl_u8(vget_high_u8(lo)));  // values  8..15
    emit(2, vmovl_u8(vget_low_u8(hi)));   // values 16..23
    emit(3, vmovl_u8(vget_high_u8(hi)));  // values 24..31
}

// =========================================================================
//  Packed-B panels — the uniform `nr_max`-wide panels the mrxncxkc kernels read
// =========================================================================

/// Panel width per element type: nr_f32[0] (12) for f32, nr_f16[0] (16) for
/// f16, and the 16-column bf16 panel.
template <class T>
inline constexpr int q4_1_nr_max =
    std::is_same_v<T, float> ? nr_f32[0]
    : std::is_same_v<T, half> ? nr_f16[0]
                              : bf16_nr_max;

/// Row-major B (phys [K][N]): a panel is columns [n_start, +width) of rows
/// [k_start, +kc); it straddles at most two 32-value blocks. Width is ignored
/// when NR > 0.
template <class T, int NR>
void pack_q4_1_copy_panel(T* out, const uint8_t* src, int64_t row_blocks,
                          int k_start, int kc, int n_start, int width,
                          float scale = 1.0f) noexcept {
    constexpr int kPanel = q4_1_nr_max<T>;
    const int nb    = (NR > 0) ? NR : width;
    const int blk0  = n_start >> 5;
    const int off0  = n_start & 31;
    const int first = std::min(nb, 32 - off0);
    const bool two  = (nb > first);
    const int64_t row_bytes = row_blocks * static_cast<int64_t>(q4_1_block_bytes);

    T v0[q4_1_block_elems];
    T v1[q4_1_block_elems];
    for (int k = 0; k < kc; ++k) {
        const uint8_t* row = src + static_cast<int64_t>(k_start + k) * row_bytes;
        T* o = out + static_cast<int64_t>(k) * kPanel;

        decode_q4_1_block<T>(row + static_cast<int64_t>(blk0) * q4_1_block_bytes, v0,
                             scale);
        for (int j = 0; j < first; ++j) {
            o[j] = v0[off0 + j];
        }
        if (two) {
            decode_q4_1_block<T>(
                row + static_cast<int64_t>(blk0 + 1) * q4_1_block_bytes, v1, scale);
            for (int j = first; j < nb; ++j) {
                o[j] = v1[j - first];
            }
        }
        for (int j = nb; j < kPanel; ++j) {
            o[j] = T{};
        }
    }
}

/// Transposed B (phys [N][K]): panel rows [n_start, +width) of columns
/// [k_start, +kc). Each source row is swept linearly, so one decode yields up
/// to 32 consecutive k values — the efficient direction for Linear's weights.
template <class T, int NR>
void pack_q4_1_trans_panel(T* out, const uint8_t* src, int64_t row_blocks,
                           int k_start, int kc, int n_start, int width,
                           float scale = 1.0f) noexcept {
    constexpr int kPanel = q4_1_nr_max<T>;
    const int nb     = (NR > 0) ? NR : width;
    const int blk0   = k_start >> 5;
    const int off0   = k_start & 31;
    const int nblk   = (off0 + kc + 31) >> 5;
    const int64_t row_bytes = row_blocks * static_cast<int64_t>(q4_1_block_bytes);

    T vals[q4_1_block_elems];
    for (int j = 0; j < nb; ++j) {
        const uint8_t* row = src + static_cast<int64_t>(n_start + j) * row_bytes
                           + static_cast<int64_t>(blk0) * q4_1_block_bytes;

        int k = 0;
        for (int b = 0; b < nblk; ++b) {
            decode_q4_1_block<T>(row + static_cast<int64_t>(b) * q4_1_block_bytes,
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

    // Lanes past the live width never reach C, but zeroing them keeps the workspace deterministic.
    for (int k = 0; k < kc; ++k) {
        T* o = out + static_cast<int64_t>(k) * kPanel;
        for (int j = nb; j < kPanel; ++j) {
            o[j] = T{};
        }
    }
}

/// Pack one k-block of the Q4_1 RHS into uniform panels. @p trans means B is
/// physically [N][K]; @p dst steps by @p ldd (64-byte-aligned nr_max * kc) per
/// panel, and @p row_blocks is the B row pitch in 20-byte blocks.
template <class T>
void pack_q4_1_rhs(bool trans, int n_start, int nc, int k_start, int kc,
                   T* dst, int ldd, const void* src, int64_t row_blocks,
                   float scale) noexcept {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>,
                  "pack_q4_1_rhs: T must be float or half");

    constexpr int kPanel = q4_1_nr_max<T>;
    const uint8_t* base = static_cast<const uint8_t*>(src);

    int n = 0;
    for (; n + kPanel <= nc; n += kPanel) {
        if (trans) {
            pack_q4_1_trans_panel<T, kPanel>(
                dst, base, row_blocks, k_start, kc, n_start + n, 0, scale);
        } else {
            pack_q4_1_copy_panel<T, kPanel>(
                dst, base, row_blocks, k_start, kc, n_start + n, 0, scale);
        }
        dst += ldd;
    }
    if (n < nc) {
        if (trans) {
            pack_q4_1_trans_panel<T, 0>(
                dst, base, row_blocks, k_start, kc, n_start + n, nc - n, scale);
        } else {
            pack_q4_1_copy_panel<T, 0>(
                dst, base, row_blocks, k_start, kc, n_start + n, nc - n, scale);
        }
    }
}

// =========================================================================
//  bf16 — the BFMMLA operand layout
// =========================================================================
//
// The kernels above write the uniform `[k][nr]` panels the FMA/MMA kernels
// read. bf16's kernel reads pack_bf16.hpp's layout instead: ⌈kc/4⌉ k-groups of
// 64 elements, where colpair c holds {col 2c k0..3, col 2c+1 k0..3}. The same
// fused Q4_1 decode feeds it directly — only the destination order changes.

/// Zero the lanes a short panel or a short final k-group leaves unread:
/// columns past `width`, and k past `kc`.
inline void q4_1_zero_bf16_tail(bfloat16* out, int kc, int width) noexcept {
    const int groups = (kc + 3) >> 2;
    const bool short_k = (kc & 3) != 0;
    if (width >= bf16_nr_max && !short_k) { return; }

    for (int g = 0; g < groups; ++g) {
        bfloat16* grp = out + static_cast<int64_t>(g) * bf16_b_k4_elems;
        for (int cp = 0; cp < bf16_nr_max / 2; ++cp) {
            const bool c0 = 2 * cp < width;
            const bool c1 = 2 * cp + 1 < width;
            for (int i = 0; i < 4; ++i) {
                const bool live_k = g * 4 + i < kc;
                if (!(c0 && live_k)) { grp[cp * 8 + i] = bfloat16{}; }
                if (!(c1 && live_k)) { grp[cp * 8 + 4 + i] = bfloat16{}; }
            }
        }
    }
}

/// One bf16 panel: columns [n_start, +width) of rows [k_start, +kc), written in
/// the BFMMLA operand layout. @p Trans means B is physically [N][K].
template <bool Trans>
void pack_q4_1_bf16_panel(bfloat16* out, const uint8_t* src, int64_t row_blocks,
                          int k_start, int kc, int n_start, int width,
                          float scale) noexcept {
    q4_1_zero_bf16_tail(out, kc, width);
    const int64_t row_bytes = row_blocks * static_cast<int64_t>(q4_1_block_bytes);

    if constexpr (Trans) {
        // One source row per column; the layout pairs adjacent columns. The
        // block stream runs along k, so the tile origin is k_start's.
        const int kblk = k_start >> 5;
        const int koff = k_start & 31;
        const int nblk = (koff + kc + 31) >> 5;
        for (int cp = 0; cp < bf16_nr_max / 2 && 2 * cp < width; ++cp) {
            const bool pair = 2 * cp + 1 < width;
            const uint8_t* row0 =
                src + static_cast<int64_t>(n_start + 2 * cp) * row_bytes;
            const uint8_t* row1 =
                pair ? src + static_cast<int64_t>(n_start + 2 * cp + 1) * row_bytes
                     : nullptr;

            int done = 0;
            for (int b = 0; b < nblk; ++b) {
                bfloat16 v0[q4_1_block_elems];
                decode_q4_1_block<bfloat16>(
                    row0 + static_cast<int64_t>(kblk + b) * q4_1_block_bytes, v0,
                    scale);
                bfloat16 v1[q4_1_block_elems];
                if (pair) {
                    decode_q4_1_block<bfloat16>(
                        row1 + static_cast<int64_t>(kblk + b) * q4_1_block_bytes,
                        v1, scale);
                }
                const int start = (b == 0) ? koff : 0;
                const int count = std::min(32 - start, kc - done);
                for (int i = 0; i < count; ++i) {
                    const int k = done + i;
                    bfloat16* grp =
                        out + static_cast<int64_t>(k >> 2) * bf16_b_k4_elems;
                    grp[cp * 8 + (k & 3)] = v0[start + i];
                    if (pair) { grp[cp * 8 + 4 + (k & 3)] = v1[start + i]; }
                }
                done += count;
            }
        }
    } else {
        // Source is [K][N]: each k row carries its own block stream, and the
        // tile origin is n_start's — this axis, not k, is the block direction.
        const int nblk = n_start >> 5;
        const int noff = n_start & 31;
        const bool two = noff + width > 32;
        for (int k = 0; k < kc; ++k) {
            const uint8_t* row =
                src + static_cast<int64_t>(k_start + k) * row_bytes;
            bfloat16 v0[q4_1_block_elems];
            decode_q4_1_block<bfloat16>(
                row + static_cast<int64_t>(nblk) * q4_1_block_bytes, v0, scale);
            bfloat16 v1[q4_1_block_elems];
            if (two) {
                decode_q4_1_block<bfloat16>(
                    row + static_cast<int64_t>(nblk + 1) * q4_1_block_bytes, v1,
                    scale);
            }
            bfloat16* grp = out + static_cast<int64_t>(k >> 2) * bf16_b_k4_elems;
            for (int c = 0; c < width; ++c) {
                const int col = noff + c;
                grp[(c >> 1) * 8 + (c & 1) * 4 + (k & 3)] =
                    (col < 32) ? v0[col] : v1[col - 32];
            }
        }
    }
}

/// Pack one k-block of the Q4_1 RHS into bf16 BFMMLA panels. @p trans means B
/// is physically [N][K]; @p dst steps by @p ldd (bf16_b_panel_stride(kc)) per
/// panel, and @p row_blocks is the B row pitch in 20-byte blocks.
inline void pack_q4_1_rhs(bool trans, int n_start, int nc, int k_start, int kc,
                          bfloat16* dst, int ldd, const void* src,
                          int64_t row_blocks, float scale) noexcept {
    const uint8_t* base = static_cast<const uint8_t*>(src);

    int n = 0;
    for (; n + bf16_nr_max <= nc; n += bf16_nr_max) {
        if (trans) {
            pack_q4_1_bf16_panel<true>(dst, base, row_blocks, k_start, kc,
                                       n_start + n, bf16_nr_max, scale);
        } else {
            pack_q4_1_bf16_panel<false>(dst, base, row_blocks, k_start, kc,
                                        n_start + n, bf16_nr_max, scale);
        }
        dst += ldd;
    }
    if (n < nc) {
        if (trans) {
            pack_q4_1_bf16_panel<true>(dst, base, row_blocks, k_start, kc,
                                       n_start + n, nc - n, scale);
        } else {
            pack_q4_1_bf16_panel<false>(dst, base, row_blocks, k_start, kc,
                                        n_start + n, nc - n, scale);
        }
    }
}

}  // namespace nnops::backend::cpu::aarch64
