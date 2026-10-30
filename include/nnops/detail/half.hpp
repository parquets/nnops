#pragma once
/// @file half.hpp
/// @brief IEEE 754 binary16 (half-precision) type and conversion utilities.
///
/// On x86_64/generic, half is a uint16_t wrapper with software conversion
/// routines (kernels pair it with F16C intrinsics). On AArch64 it is the native
/// __fp16 and the compiler does the conversions.
///
/// Because `half(bits)` is a *numeric* conversion of the integer where half is
/// __fp16, raw binary16 storage goes through half_from_bits() / half_to_bits(),
/// which reinterpret the bits (memcpy on AArch64).

#include <cstdint>

#if defined(__aarch64__) || defined(_M_ARM64) || defined(NNOPS_ARCH_AARCH64)

namespace nnops::backend::cpu {

using half = __fp16;

inline float half_to_float(half h) noexcept {
    return static_cast<float>(h);
}

inline half float_to_half(float f) noexcept {
    return static_cast<__fp16>(f);
}

inline uint16_t half_to_bits(half h) noexcept {
    uint16_t bits;
    __builtin_memcpy(&bits, &h, sizeof(bits));
    return bits;
}

inline half half_from_bits(uint16_t bits) noexcept {
    half h;
    __builtin_memcpy(&h, &bits, sizeof(bits));
    return h;
}

inline void convert_half_to_float(float* dst, const half* src, int n) noexcept {
    for (int i = 0; i < n; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
}

inline void convert_float_to_half(half* dst, const float* src, int n) noexcept {
    for (int i = 0; i < n; ++i) {
        dst[i] = static_cast<__fp16>(src[i]);
    }
}

}  // namespace nnops::backend::cpu

#else  // x86_64 / generic — uint16_t wrapper with software conversion

namespace nnops::backend::cpu {

/// binary16 stored in a uint16_t: a distinct type so fp16 overload resolution
/// and type safety work for MMA/Pack dispatch. The raw bits are IEEE 754.
struct half {
    uint16_t bits;

    constexpr half() noexcept : bits(0) {}
    explicit constexpr half(uint16_t b) noexcept : bits(b) {}
};

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

    o.u |= (static_cast<uint32_t>(h.bits) & 0x8000U) << 16;
    return o.f;
}

/// Rounds to nearest even; overflow goes to infinity and NaN is canonicalized
/// to qNaN with its payload dropped.
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

inline uint16_t half_to_bits(half h) noexcept {
    return h.bits;
}

inline half half_from_bits(uint16_t bits) noexcept {
    return half(bits);
}

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

}  // namespace nnops::backend::cpu

#endif  // AArch64 vs x86_64
