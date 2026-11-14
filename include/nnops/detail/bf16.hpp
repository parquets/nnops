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

#include "nnops/detail/simd/cpu_features.hpp"  // NNOPS_ARCH_*

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace nnops {

/// Which backend carries bf16 kernels: the CPU on AArch64 (its bf16 SIMD path),
/// the CUDA backend on any build that has it. The two flags stay apart because
/// the admission is per backend — a CUDA build on x86-64 admits bf16 for
/// Backend::CUDA and still refuses it for Backend::CPU, where no kernel exists
/// and validation returns Status::unsupported instead.
namespace detail {
#if defined(NNOPS_ARCH_AARCH64)
inline constexpr bool bf16_cpu_enabled = true;
#else
inline constexpr bool bf16_cpu_enabled = false;
#endif

#if defined(NNOPS_HAS_CUDA)
inline constexpr bool bf16_cuda_enabled = true;
#else
inline constexpr bool bf16_cuda_enabled = false;
#endif

/// Some backend in this build carries bf16, so the dtype exists at all.
inline constexpr bool bf16_enabled = bf16_cpu_enabled || bf16_cuda_enabled;
}  // namespace detail

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

namespace backend::cpu {

/// bfloat16 held as its raw IEEE bits — the counterpart of `half`, and the
/// element type the CPU kernels are instantiated on. Every SIMD operation on it
/// widens to f32 (see simd.hpp), so a kernel that templates on the element type
/// gets f32 arithmetic and narrows once per store. The float conversion is
/// implicit both ways so the scalar tails can treat a value as a number, as they
/// do for the native `__fp16`.
struct bfloat16 {
    uint16_t bits;

    constexpr bfloat16() noexcept : bits(0) {}
    explicit constexpr bfloat16(uint16_t b) noexcept : bits(b) {}
    inline bfloat16(float f) noexcept : bits(::nnops::float_to_bf16(f)) {}

    inline operator float() const noexcept {
        return ::nnops::bf16_to_float(bits);
    }
};

inline float bfloat16_to_float(bfloat16 h) noexcept { return h; }
inline bfloat16 float_to_bfloat16(float f) noexcept { return bfloat16(f); }

inline uint16_t bfloat16_to_bits(bfloat16 h) noexcept { return h.bits; }
inline bfloat16 bfloat16_from_bits(uint16_t bits) noexcept {
    return bfloat16(bits);
}

inline void convert_bfloat16_to_float(float* dst, const bfloat16* src, int n) noexcept {
    for (int i = 0; i < n; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
}

inline void convert_float_to_bfloat16(bfloat16* dst, const float* src, int n) noexcept {
    for (int i = 0; i < n; ++i) {
        dst[i] = bfloat16(src[i]);
    }
}

}  // namespace backend::cpu

}  // namespace nnops
