#pragma once
/// @file q4_1.hpp
/// @brief llama.cpp Q4_1 block codec — quantize, and fused unpack + dequantize.
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
/// `quantize_q4` is the encoder — llama.cpp's `quantize_row_q4_1_ref`, bit for
/// bit — so a weight quantized here and one quantized by ggml are the same bytes.
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
using backend::cpu::half_to_bits;
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

/// Quantize `count` f32 or f16 values into Q4_1 blocks — llama.cpp's
/// `quantize_row_q4_1_ref`, bit for bit: the block's own min and max set
/// d = (max - min) / 15 and m = min, and every value rounds half up into
/// 0..15 with the low nibble of byte j holding value j and the high nibble
/// value j + 16. A block whose values are all equal gets d = 0, so it decodes
/// back to m exactly.
///
/// `count` need not be a multiple of 32, mirroring pack_q4(): a trailing
/// partial block takes its d and m from the values that exist and leaves the
/// missing nibbles zero, which the decoder never reads.
template <class T>
void quantize_q4(const T* src, int64_t count, void* blocks) noexcept {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>,
                  "quantize_q4: T must be float or half");
    if (count <= 0) {
        return;
    }

    const auto load = [](const T* p) {
        if constexpr (std::is_same_v<T, float>) {
            return *p;
        } else {
            return half_to_float(*p);
        }
    };

    constexpr int kQMax = 15;
    uint8_t* dst = static_cast<uint8_t*>(blocks);
    const int64_t nblocks = (count + q4_1_block_elems - 1) / q4_1_block_elems;

    for (int64_t b = 0; b < nblocks; ++b) {
        const int64_t base = b * q4_1_block_elems;
        const int n = static_cast<int>(
            std::min<int64_t>(q4_1_block_elems, count - base));
        const T* x = src + base;

        float vmin = load(x);
        float vmax = vmin;
        for (int j = 1; j < n; ++j) {
            const float v = load(x + j);
            vmin = v < vmin ? v : vmin;
            vmax = v > vmax ? v : vmax;
        }

        const float d = (vmax - vmin) / static_cast<float>(kQMax);
        const float id = (d != 0.0f) ? 1.0f / d : 0.0f;

        uint8_t* blk = dst + b * static_cast<int64_t>(q4_1_block_bytes);
        const uint16_t d_bits = half_to_bits(float_to_half(d));
        const uint16_t m_bits = half_to_bits(float_to_half(vmin));
        std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
        std::memcpy(blk + 2, &m_bits, sizeof(m_bits));

        uint8_t qs[16] = {};
        for (int j = 0; j < n; ++j) {
            // Truncating the +0.5 gives llama.cpp's round-half-up; the clamp
            // only guards a value that slid past max through rounding.
            const float qf = (load(x + j) - vmin) * id + 0.5f;
            const uint8_t q = static_cast<uint8_t>(
                qf < static_cast<float>(kQMax) ? static_cast<int>(qf) : kQMax);
            if (j < 16) {
                qs[j] = q;
            } else {
                qs[j - 16] |= static_cast<uint8_t>(q << 4);
            }
        }
        std::memcpy(blk + 4, qs, sizeof(qs));
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
