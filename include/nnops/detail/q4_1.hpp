#pragma once
/// @file q4_1.hpp
/// @brief llama.cpp Q4_1 block decoder — fused unpack + dequantize.
///
/// A Q4_1 tensor is a stream of 20-byte blocks covering 32 values each:
///
///   offset  size  field
///   0       2     d   fp16 delta (scale)
///   2       2     m   fp16 minimum (offset)
///   4       16    qs  4-bit quants: low nibble of byte j is value j,
///                     high nibble is value j + 16
///
///   y[j] = d * q + m,  q in [0, 15]
///
/// which is byte-for-byte ggml's `block_q4_1` / `dequantize_row_q4_1`. The block
/// is self-describing (d and m live in the data), so no QuantParams are involved.
///
/// `pack_q4` fuses the nibble unpacking, the dequantization and the f32 → f16
/// narrowing: each block is read once into a 32-float stack buffer and the
/// values are written straight to the output, with no intermediate quantized
/// buffer. This is the portable scalar reference; the CPU backend's GEMM route
/// uses `pack_q4_1_rhs` in src/backend/cpu/x86_64/pack_q4_1.hpp and the NEON
/// one in src/backend/cpu/aarch64/pack_q4_1.hpp, which fuse the same decode into
/// the RHS panel pack so a Q4_1 weight never passes through a dense buffer.
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

/// One llama.cpp Q4_1 block: `d`, `m` and 16 nibble bytes.
/// Provided for callers that want to inspect a block; pack_q4() itself reads the
/// bytes with memcpy so an unaligned source (a slice of a GGUF file, say) is
/// fine.
struct q4_1_block {
    uint16_t d;       ///< fp16 delta
    uint16_t m;       ///< fp16 minimum
    uint8_t  qs[16];  ///< low nibble → value j, high nibble → value j + 16
};
static_assert(sizeof(q4_1_block) == q4_1_block_bytes,
              "q4_1_block must be exactly one llama.cpp Q4_1 block");

/// Decode a block: writes its 32 values as f32 to `out`.
inline void q4_1_unpack_block(const void* block, float* out) noexcept {
    const uint8_t* p = static_cast<const uint8_t*>(block);

    uint16_t d_bits, m_bits;
    std::memcpy(&d_bits, p + 0, sizeof(d_bits));
    std::memcpy(&m_bits, p + 2, sizeof(m_bits));

    const float d = half_to_float(half_from_bits(d_bits));
    const float m = half_to_float(half_from_bits(m_bits));
    const uint8_t* qs = p + 4;

    for (int j = 0; j < 16; ++j) {
        const float lo = static_cast<float>(qs[j] & 0x0F);
        const float hi = static_cast<float>(qs[j] >> 4);
        out[j]      = d * lo + m;
        out[j + 16] = d * hi + m;
    }
}

/// Fused unpack + dequantize: decode `count` values from tightly packed 20-byte
/// Q4_1 blocks into a dense f32 or f16 array. Arithmetic runs in f32, matching
/// llama.cpp's dequantize path, and f16 output rounds once at the store.
template <class T>
void pack_q4(const void* blocks, int64_t count, T* out) noexcept {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>,
                  "pack_q4: T must be float or half");

    if (count <= 0) {
        return;
    }

    const uint8_t* src = static_cast<const uint8_t*>(blocks);
    const int64_t nblocks = (count + q4_1_block_elems - 1) / q4_1_block_elems;

    for (int64_t b = 0; b < nblocks; ++b) {
        float vals[q4_1_block_elems];
        q4_1_unpack_block(src + b * static_cast<int64_t>(q4_1_block_bytes), vals);

        const int64_t base = b * q4_1_block_elems;
        const int n = static_cast<int>(
            std::min<int64_t>(q4_1_block_elems, count - base));

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
