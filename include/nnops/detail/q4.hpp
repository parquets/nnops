#pragma once
/// @file q4.hpp
/// @brief llama.cpp Q4_0 / Q4_1 / Q8_0 block codec — quantize, and fused unpack
///        + dequantize.
///
/// All three formats cover 32 values per block. Q4_0 and Q4_1 share the same 16
/// nibble bytes (byte j's low nibble is value j, its high nibble value j + 16);
/// Q8_0 stores the 32 values as signed bytes:
///
///   Q4_0  18 bytes:  fp16 d, then 16 nibble bytes
///                   y[j] = d · (q − 8),  q in [0, 15]
///
///   Q4_1  20 bytes:  fp16 d, fp16 m, then 16 nibble bytes
///                   y[j] = d · q + m,    q in [0, 15]
///
///   Q8_0  34 bytes:  fp16 d, then 32 int8
///                   y[j] = d · q,        q in [-127, 127]
///
/// That is byte-for-byte ggml's `block_q4_0` / `block_q4_1` / `block_q8_0`, and
/// the encoders are `quantize_row_q4_0_ref` / `quantize_row_q4_1_ref` /
/// `quantize_row_q8_0_ref`, bit for bit. The blocks are self-describing, so no
/// QuantParams are involved.
///
/// `pack_q4` fuses the unpack, the dequantization and the f32 → f16 narrowing:
/// each block is read once into a 32-float stack buffer and the values are
/// written straight to the output. This is the portable scalar
/// reference; the CPU backends' GEMM routes use the vectorized pack kernels in
/// src/backend/cpu/{aarch64,x86_64}/pack_q4.hpp, which fuse the same decode into
/// the RHS panel pack so a quantized weight never passes through a dense buffer.
///
/// Q4_0's scale is signed (`d = peak / −8`, where `peak` is the signed value of
/// largest magnitude), which is what makes the 0..15 nibble range asymmetric
/// around 8; Q8_0's is `d = amax / 127`. The encoders follow ggml's exact
/// rounding, including the trailing partial block: only the values that exist
/// set `d`/`m`, and the missing quants stay zero and are never read.

#include <algorithm>
#include <cmath>
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
struct q4_1_block {
    uint16_t d;     ///< fp16 delta
    uint16_t m;     ///< fp16 minimum
    uint8_t qs[16]; ///< low nibble → value j, high nibble → value j + 16
};
static_assert(sizeof(q4_1_block) == q4_1_block_bytes,
              "q4_1_block must be exactly one llama.cpp Q4_1 block");

/// One llama.cpp Q4_0 block: `d` and 16 nibble bytes.
struct q4_0_block {
    uint16_t d;     ///< fp16 delta
    uint8_t qs[16]; ///< low nibble → value j, high nibble → value j + 16
};
static_assert(sizeof(q4_0_block) == q4_0_block_bytes,
              "q4_0_block must be exactly one llama.cpp Q4_0 block");

/// One llama.cpp Q8_0 block: `d` and 32 signed bytes.
struct q8_0_block {
    uint16_t d;     ///< fp16 delta
    int8_t qs[32];  ///< signed quants, value j is qs[j]
};
static_assert(sizeof(q8_0_block) == q8_0_block_bytes,
              "q8_0_block must be exactly one llama.cpp Q8_0 block");

/// Q4_1 carries a per-block minimum; Q4_0 and Q8_0 do not (Q4_0's nibbles are
/// biased by 8, Q8_0's bytes are signed).
constexpr bool q4_has_min(DataType fmt) noexcept { return fmt == DataType::q4_1; }

/// Bytes of one block of @p fmt.
constexpr size_t q4_block_bytes(DataType fmt) noexcept { return data_type_block_bytes(fmt); }

/// Decode a block into 32 f32 values: `d·(q−8)` for Q4_0, `d·q + m` for Q4_1,
/// `d·q` for Q8_0.
inline void q4_unpack_block(const void* block, float* out, DataType fmt) noexcept
{
    const uint8_t* p = static_cast<const uint8_t*>(block);
    uint16_t d_bits = 0, m_bits = 0;
    std::memcpy(&d_bits, p + 0, sizeof(d_bits));

    if (fmt == DataType::q8_0) {
        const float d = half_to_float(half_from_bits(d_bits));
        const auto* qs = reinterpret_cast<const int8_t*>(p + 2);
        for (int j = 0; j < 32; ++j) {
            out[j] = d * static_cast<float>(qs[j]);
        }
        return;
    }

    const bool with_min = q4_has_min(fmt);
    if (with_min) {
        std::memcpy(&m_bits, p + 2, sizeof(m_bits));
    }
    const float d = half_to_float(half_from_bits(d_bits));
    const float m = with_min ? half_to_float(half_from_bits(m_bits)) : 0.0f;
    const uint8_t* qs = p + (with_min ? 4 : 2);
    const int bias = with_min ? 0 : 8;

    for (int j = 0; j < 16; ++j) {
        const float lo = static_cast<float>(static_cast<int>(qs[j] & 0x0F) - bias);
        const float hi = static_cast<float>(static_cast<int>(qs[j] >> 4) - bias);
        out[j] = d * lo + m;
        out[j + 16] = d * hi + m;
    }
}

namespace q4_detail {

/// Q4_1: d = (max − min)/15, m = min, every value rounds half up into 0..15.
template <class T, class Load> void quantize_1(const T* x, int n, uint8_t* blk, Load load) noexcept
{
    float vmin = load(x);
    float vmax = vmin;
    for (int j = 1; j < n; ++j) {
        const float v = load(x + j);
        vmin = v < vmin ? v : vmin;
        vmax = v > vmax ? v : vmax;
    }
    const float d = (vmax - vmin) / 15.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    const uint16_t d_bits = half_to_bits(float_to_half(d));
    const uint16_t m_bits = half_to_bits(float_to_half(vmin));
    std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
    std::memcpy(blk + 2, &m_bits, sizeof(m_bits));

    uint8_t qs[16] = {};
    for (int j = 0; j < n; ++j) {
        const float qf = (load(x + j) - vmin) * id + 0.5f;
        const int q = qf < 15.0f ? static_cast<int>(qf) : 15;
        if (j < 16) {
            qs[j] = static_cast<uint8_t>(q);
        } else {
            qs[j - 16] |= static_cast<uint8_t>(q << 4);
        }
    }
    std::memcpy(blk + 4, qs, sizeof(qs));
}

/// Q8_0: d = amax / 127 (amax is the absolute maximum), each value rounds
/// half away from zero into [-127, 127] — ggml's `quantize_row_q8_0_ref`.
template <class T, class Load> void quantize_8(const T* x, int n, uint8_t* blk, Load load) noexcept
{
    float amax = 0.0f;
    for (int j = 0; j < n; ++j) {
        amax = std::max(amax, std::fabs(load(x + j)));
    }
    const float d = amax / 127.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    const uint16_t d_bits = half_to_bits(float_to_half(d));
    std::memcpy(blk + 0, &d_bits, sizeof(d_bits));

    int8_t qs[q4_block_elems] = {};
    for (int j = 0; j < n; ++j) {
        qs[j] = static_cast<int8_t>(std::round(load(x + j) * id));
    }
    std::memcpy(blk + 2, qs, sizeof(qs));
}

/// Q4_0: d = peak / −8 (peak is the signed value of largest magnitude), and
/// ggml's `(int8_t)(x·id + 8.5)`, clamped high to 15.
template <class T, class Load> void quantize_0(const T* x, int n, uint8_t* blk, Load load) noexcept
{
    float amax = 0.0f;
    float peak = 0.0f;
    for (int j = 0; j < n; ++j) {
        const float v = load(x + j);
        if (amax < std::fabs(v)) {
            amax = std::fabs(v);
            peak = v;
        }
    }
    const float d = peak / -8.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    const uint16_t d_bits = half_to_bits(float_to_half(d));
    std::memcpy(blk + 0, &d_bits, sizeof(d_bits));

    uint8_t qs[16] = {};
    for (int j = 0; j < n; ++j) {
        const int q = std::min(15, static_cast<int>(static_cast<int8_t>(load(x + j) * id + 8.5f)));
        if (j < 16) {
            qs[j] = static_cast<uint8_t>(q);
        } else {
            qs[j - 16] |= static_cast<uint8_t>(q << 4);
        }
    }
    std::memcpy(blk + 2, qs, sizeof(qs));
}

} // namespace q4_detail

/// Quantize `count` f32 or f16 values into Q4_0, Q4_1 or Q8_0 blocks. `fmt`
/// defaults to Q4_1 for source compatibility.
template <class T>
void quantize_q4(const T* src, int64_t count, void* blocks, DataType fmt = DataType::q4_1) noexcept
{
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

    uint8_t* dst = static_cast<uint8_t*>(blocks);
    const int64_t nblocks = (count + q4_block_elems - 1) / q4_block_elems;
    const int64_t block_bytes = static_cast<int64_t>(q4_block_bytes(fmt));

    for (int64_t b = 0; b < nblocks; ++b) {
        const int64_t base = b * q4_block_elems;
        const int n = static_cast<int>(std::min<int64_t>(q4_block_elems, count - base));
        uint8_t* blk = dst + b * block_bytes;
        if (fmt == DataType::q8_0) {
            q4_detail::quantize_8<T>(src + base, n, blk, load);
        } else if (q4_has_min(fmt)) {
            q4_detail::quantize_1<T>(src + base, n, blk, load);
        } else {
            q4_detail::quantize_0<T>(src + base, n, blk, load);
        }
    }
}

/// Fused unpack + dequantize: decode `count` values from tightly packed blocks
/// into a dense f32 or f16 array. Arithmetic runs in f32; f16 output rounds
/// once at the store.
template <class T>
void pack_q4(const void* blocks, int64_t count, T* out, DataType fmt = DataType::q4_1) noexcept
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>,
                  "pack_q4: T must be float or half");
    if (count <= 0) {
        return;
    }

    const uint8_t* src = static_cast<const uint8_t*>(blocks);
    const int64_t nblocks = (count + q4_block_elems - 1) / q4_block_elems;
    const int64_t block_bytes = static_cast<int64_t>(q4_block_bytes(fmt));

    for (int64_t b = 0; b < nblocks; ++b) {
        float vals[q4_block_elems];
        q4_unpack_block(src + b * block_bytes, vals, fmt);

        const int64_t base = b * q4_block_elems;
        const int n = static_cast<int>(std::min<int64_t>(q4_block_elems, count - base));
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

} // namespace nnops
