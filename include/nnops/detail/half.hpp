#pragma once
/// @file half.hpp
/// @brief IEEE 754 binary16 (half-precision) type and conversion utilities.
///
/// Portable implementation using bit manipulation. For x86_64 with F16C
/// support, conversion can use _mm_cvtss_sh / _mm_cvtph_ps intrinsics
/// directly in the micro-kernel files; this header provides the scalar
/// fallback and the type definition.

#include <cstdint>

namespace nnops::backend::cpu {

/// @brief Half-precision (binary16) value stored in a uint16_t.
///
/// This is a trivial wrapper so that overload resolution and type safety
/// work for MMA/Pack dispatch on fp16. The raw bits follow IEEE 754.
struct half {
    uint16_t bits;

    constexpr half() noexcept : bits(0) {}
    explicit constexpr half(uint16_t b) noexcept : bits(b) {}
};

/// @brief Convert an IEEE 754 binary16 value to float32.
///
/// Handles normals, subnormals, zero, infinity, and NaN.
inline float half_to_float(half h) noexcept {
    constexpr uint32_t magic_val = 113U << 23;
    constexpr uint32_t shifted_exp = 0x7c00U << 13;

    union { uint32_t u; float f; } o;
    o.u = (static_cast<uint32_t>(h.bits) & 0x7fffU) << 13;
    const uint32_t exp = shifted_exp & o.u;
    o.u += (127U - 15U) << 23;  // exponent bias adjust

    if (exp == shifted_exp) {
        o.u += (128U - 16U) << 23;  // Inf / NaN
    } else if (exp == 0) {
        o.u += 1U << 23;
        union { uint32_t u; float f; } magic;
        magic.u = magic_val;
        o.f -= magic.f;  // renormalize subnormal
    }

    o.u |= (static_cast<uint32_t>(h.bits) & 0x8000U) << 16;  // sign
    return o.f;
}

/// @brief Convert a float32 value to IEEE 754 binary16.
///
/// Rounds to nearest even. Handles overflow to infinity, subnormals,
/// and NaN payload truncation.
inline half float_to_half(float f) noexcept {
    constexpr uint32_t f32infty_u = 255U << 23;
    constexpr uint32_t f16max_u   = (127U + 16U) << 23;
    constexpr uint32_t denorm_magic_u = ((127U - 15U) + (23U - 10U) + 1U) << 23;
    constexpr uint32_t sign_mask = 0x80000000U;

    union { uint32_t u; float f; } x;
    x.f = f;

    const uint32_t sign = x.u & sign_mask;
    x.u ^= sign;

    uint16_t bits;
    if (x.u >= f16max_u) {
        // Inf or NaN: all exponent bits set
        bits = (x.u > f32infty_u) ? static_cast<uint16_t>(0x7e00)   // NaN → qNaN
                                  : static_cast<uint16_t>(0x7c00);  // Inf
    } else {
        if (x.u < (113U << 23)) {
            // Subnormal or zero
            union { uint32_t u; float f; } magic;
            magic.u = denorm_magic_u;
            x.f += magic.f;
            bits = static_cast<uint16_t>(x.u - magic.u);
        } else {
            const uint32_t mant_odd = (x.u >> 13) & 1U;
            x.u += (static_cast<uint32_t>(15 - 127) << 23) + 0xfffU;
            x.u += mant_odd;
            bits = static_cast<uint16_t>(x.u >> 13);
        }
    }

    bits = static_cast<uint16_t>(bits | static_cast<uint16_t>(sign >> 16));
    return half{bits};
}

/// @brief Convert array of fp16 to fp32 in-place (scalar loop).
/// @{
inline void convert_half_to_float(float* dst, const half* src, int n) noexcept {
    for (int i = 0; i < n; ++i) {
        dst[i] = half_to_float(src[i]);
    }
}

inline void convert_float_to_half(half* dst, const float* src, int n) noexcept {
    for (int i = 0; i < n; ++i) {
        dst[i] = float_to_half(src[i]);
    }
}
/// @}

}  // namespace nnops::backend::cpu
