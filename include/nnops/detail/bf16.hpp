#pragma once
/// @file bf16.hpp
/// @brief bfloat16 ↔ float32 conversion utilities.
///
/// bfloat16 (Brain Floating Point) uses the same 8-bit exponent as float32,
/// but only 7 bits of mantissa (vs 23 for f32). This makes conversion trivial:
///   bf16 → f32: left-shift 16 bits (pad mantissa with zeros)
///   f32 → bf16: truncate lower 16 bits (optionally round-to-nearest-even)
///
/// Reference: IEEE 754-2008, Google Brain bfloat16 specification.

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace nnops {

/// Convert bfloat16 (stored as uint16_t) to float32.
inline float bf16_to_float(uint16_t v) noexcept {
    static_assert(sizeof(float) == 4);
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

/// Convert float32 to bfloat16 (stored as uint16_t) with round-to-nearest-even.
inline uint16_t float_to_bf16(float v) noexcept {
    static_assert(sizeof(float) == 4);
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(float));
    // Round to nearest even: add 0x7FFF (rounding constant) + LSB of the
    // truncated mantissa for tie-breaking.
    bits += 0x7FFF + ((bits >> 16) & 1);
    return static_cast<uint16_t>(bits >> 16);
}

}  // namespace nnops
