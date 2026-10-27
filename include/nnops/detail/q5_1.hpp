#pragma once
/// @file q5_1.hpp
/// @brief llama.cpp Q5_1 block decoder — fused unpack + dequantize.
///
/// A Q5_1 tensor is a stream of 24-byte blocks covering 32 values each:
///
///   offset  size  field
///   0       2     d   fp16 delta (scale)
///   2       2     m   fp16 minimum (offset)
///   4       4     qh  the 5th bit of every quant: bit j is the high bit of
///                     value j (little-endian)
///   8       16    qs  4-bit quants: low nibble of byte j is value j,
///                     high nibble is value j + 16
///
/// and value j is
///
///   y[j] = d * ( low4(qs[j % 16], j < 16 ? low : high) | (bit j of qh) << 4 ) + m
///
/// This is byte-for-byte ggml's `block_q5_1` / `dequantize_row_q5_1`; the block
/// itself is self-describing (d and m live in the data), so no QuantParams are
/// involved.
///
/// `pack_q5` fuses the nibble/bit unpacking with the dequantization and the
/// f32 → f16 narrowing: each block is read once and the values are written
/// straight to the output, with no intermediate quantized buffer. The output
/// is a dense f32 or f16 array in logical order, i.e. element i corresponds to
/// value i of the block stream. This is the portable scalar reference; the CPU
/// backend's GEMM route uses the AVX2 implementation in
/// src/backend/cpu/x86_64/pack_q5_1.hpp and the NEON one in
/// src/backend/cpu/aarch64/pack_q5_1.hpp, both of which decode each block once
/// per row and pack the RHS panels straight from the block stream.
///
/// The count need not be a multiple of 32: a trailing partial block is decoded
/// only up to `count`, which lets a tensor whose innermost dimension is not
/// block-aligned be unpacked without padding the destination. The source is
/// still ceil(count / 32) whole blocks, matching data_type_row_bytes().

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "nnops/core/data_type.hpp"
#include "nnops/detail/half.hpp"

namespace nnops {

using backend::cpu::float_to_half;
using backend::cpu::half;
using backend::cpu::half_from_bits;
using backend::cpu::half_to_float;

/// One llama.cpp Q5_1 block: `d`, `m`, the 5th-bit mask and 16 nibbles.
///
/// Provided for callers that want to inspect a block; pack_q5() itself reads the
/// bytes with memcpy so an unaligned source (a slice of a GGUF file, say) is
/// fine.
struct q5_1_block {
    uint16_t d;       ///< fp16 delta
    uint16_t m;       ///< fp16 minimum
    uint8_t  qh[4];   ///< 5th bits, bit j ↔ value j (little-endian)
    uint8_t  qs[16];  ///< low nibble → value j, high nibble → value j + 16
};
static_assert(sizeof(q5_1_block) == q5_1_block_bytes,
              "q5_1_block must be exactly one llama.cpp Q5_1 block");

/// Decode a block: writes its 32 values as f32 to `out`.
inline void q5_1_unpack_block(const void* block, float* out) noexcept {
    const uint8_t* p = static_cast<const uint8_t*>(block);

    uint16_t d_bits, m_bits;
    uint32_t qh;
    std::memcpy(&d_bits, p + 0, sizeof(d_bits));
    std::memcpy(&m_bits, p + 2, sizeof(m_bits));
    std::memcpy(&qh,     p + 4, sizeof(qh));

    const float d = half_to_float(half_from_bits(d_bits));
    const float m = half_to_float(half_from_bits(m_bits));
    const uint8_t* qs = p + 8;

    for (int j = 0; j < 16; ++j) {
        // Bit j of qh is the high bit of value j; bit j + 16 the high bit of
        // value j + 16. The nibble order matches llama.cpp: the low nibble of
        // byte j belongs to the first half, the high nibble to the second.
        const uint32_t lo = static_cast<uint32_t>(qs[j] & 0x0F) | (((qh >> j)        & 1u) << 4);
        const uint32_t hi = static_cast<uint32_t>(qs[j] >> 4)   | (((qh >> (j + 16)) & 1u) << 4);
        out[j]      = d * static_cast<float>(lo) + m;
        out[j + 16] = d * static_cast<float>(hi) + m;
    }
}

/// Fused unpack + dequantize: decode @p count values from a stream of 24-byte
/// Q5_1 blocks into a dense f32 or f16 array.
///
/// @param blocks  first block; blocks are tightly packed, 24 bytes apart.
/// @param count   number of logical values to produce (may be 0). The source
///                must hold ceil(count / 32) blocks.
/// @param out     destination array of @p count values; T is float or half.
///
/// f16 output rounds once per element at the end. The arithmetic is done in
/// f32, matching llama.cpp's dequantize path.
template <class T>
void pack_q5(const void* blocks, int64_t count, T* out) noexcept {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>,
                  "pack_q5: T must be float or half");

    if (count <= 0) {
        return;
    }

    const uint8_t* src = static_cast<const uint8_t*>(blocks);
    const int64_t nblocks = (count + q5_1_block_elems - 1) / q5_1_block_elems;

    for (int64_t b = 0; b < nblocks; ++b) {
        float vals[q5_1_block_elems];
        q5_1_unpack_block(src + b * static_cast<int64_t>(q5_1_block_bytes), vals);

        const int64_t base = b * q5_1_block_elems;
        const int n = static_cast<int>(
            std::min<int64_t>(q5_1_block_elems, count - base));

        if constexpr (std::is_same_v<T, float>) {
            for (int j = 0; j < n; ++j) {
                out[base + j] = vals[j];
            }
        } else {
            for (int j = 0; j < n; ++j) {
                out[base + j] = float_to_half(vals[j]);
            }
        }
    }
}

}  // namespace nnops
