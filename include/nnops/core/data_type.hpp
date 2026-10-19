#pragma once
/// @file data_type.hpp
/// @brief DataType enum and compile-time traits for the nnops library.

#include <cstdint>
#include <cstddef>

namespace nnops {

/// Supported data types for tensor elements.
enum class DataType : uint8_t {
    f32  = 0,  ///< 32-bit IEEE float
    f16  = 1,  ///< 16-bit IEEE half-precision (stored as uint16_t)
    bf16 = 2,  ///< bfloat16 (stored as uint16_t)
    s8      = 3,  ///< signed 8-bit integer
    u8      = 4,  ///< unsigned 8-bit integer
    s32     = 5,  ///< signed 32-bit integer
    s64     = 6,  ///< signed 64-bit integer
    f8_e4m3 = 7,  ///< 8-bit float E4M3 (4 exponent, 3 mantissa) — precision
    f8_e5m2 = 8,  ///< 8-bit float E5M2 (5 exponent, 2 mantissa) — range
    s4      = 9,  ///< signed 4-bit integer (2 values packed per byte, low nibble first)
    u4      = 10, ///< unsigned 4-bit integer (2 values packed per byte, low nibble first)
    q5_1    = 11, ///< llama.cpp Q5_1 block format: 32 values in a 24-byte block
                  ///< (fp16 d, fp16 m, 4-byte 5th-bit mask, 16 nibble bytes).
                  ///< Self-describing: the per-block scale/offset live in the
                  ///< data, so QuantParams is not used. Decode with pack_q5()
                  ///< in <nnops/detail/q5_1.hpp> (portable scalar); the CPU
                  ///< backend's vectorized pack kernels live in
                  ///< src/backend/cpu/x86_64/pack_q5_1.hpp. See q5_1_block_*
                  ///< for sizing.
};

/// Compile-time traits for each DataType.
template <DataType D>
struct DataTypeTraits;

template <>
struct DataTypeTraits<DataType::f32> {
    using ctype = float;
    static constexpr size_t size = 4;
    static constexpr const char* name = "f32";
};

template <>
struct DataTypeTraits<DataType::f16> {
    using ctype = uint16_t;
    static constexpr size_t size = 2;
    static constexpr const char* name = "f16";
};

template <>
struct DataTypeTraits<DataType::bf16> {
    using ctype = uint16_t;
    static constexpr size_t size = 2;
    static constexpr const char* name = "bf16";
};

template <>
struct DataTypeTraits<DataType::s8> {
    using ctype = int8_t;
    static constexpr size_t size = 1;
    static constexpr const char* name = "s8";
};

template <>
struct DataTypeTraits<DataType::u8> {
    using ctype = uint8_t;
    static constexpr size_t size = 1;
    static constexpr const char* name = "u8";
};

template <>
struct DataTypeTraits<DataType::s32> {
    using ctype = int32_t;
    static constexpr size_t size = 4;
    static constexpr const char* name = "s32";
};

template <>
struct DataTypeTraits<DataType::s64> {
    using ctype = int64_t;
    static constexpr size_t size = 8;
    static constexpr const char* name = "s64";
};

template <>
struct DataTypeTraits<DataType::f8_e4m3> {
    using ctype = uint8_t;
    static constexpr size_t size = 1;
    static constexpr const char* name = "f8_e4m3";
};

template <>
struct DataTypeTraits<DataType::f8_e5m2> {
    using ctype = uint8_t;
    static constexpr size_t size = 1;
    static constexpr const char* name = "f8_e5m2";
};

template <>
struct DataTypeTraits<DataType::s4> {
    using ctype = uint8_t;   ///< packed byte storage; 2 int4 values per byte (low nibble first)
    static constexpr size_t size = 0;  ///< sub-byte — use packed helpers, not sizeof(ctype)
    static constexpr const char* name = "s4";
};

template <>
struct DataTypeTraits<DataType::u4> {
    using ctype = uint8_t;   ///< packed byte storage; 2 uint4 values per byte (low nibble first)
    static constexpr size_t size = 0;  ///< sub-byte — use packed helpers, not sizeof(ctype)
    static constexpr const char* name = "u4";
};

template <>
struct DataTypeTraits<DataType::q5_1> {
    using ctype = uint8_t;   ///< block storage; 32 values per 24-byte block
    static constexpr size_t size = 0;  ///< block-packed — use q5_1_block_bytes
    static constexpr const char* name = "q5_1";
};

// ---- Block-packed formats ------------------------------------------------
//
// q5_1 has no per-element storage unit: values live in 32-element blocks of
// 24 bytes, so data_type_size() returns 0 (like the 4-bit types) and every
// row/stride computation must go through data_type_row_bytes() /
// data_type_stride_unit() instead of multiplying by the element size.

/// Elements covered by one Q5_1 block.
inline constexpr int64_t q5_1_block_elems = 32;

/// Bytes of one Q5_1 block: fp16 d, fp16 m, 4-byte 5th-bit mask, 16 nibbles.
inline constexpr size_t q5_1_block_bytes = 24;

/// Whether the type stores values in fixed-size blocks rather than as a
/// whole number of bytes per element.
constexpr bool is_block_packed_dtype(DataType dt) noexcept {
    return dt == DataType::q5_1;
}

/// Returns the size in bytes of a single element of the given DataType.
constexpr size_t data_type_size(DataType dt) noexcept {
    switch (dt) {
    case DataType::f32:  return 4;
    case DataType::f16:  return 2;
    case DataType::bf16: return 2;
    case DataType::s8:      return 1;
    case DataType::u8:      return 1;
    case DataType::s32:     return 4;
    case DataType::s64:     return 8;
    case DataType::f8_e4m3: return 1;
    case DataType::f8_e5m2: return 1;
    case DataType::s4:   return 0;  // sub-byte: 2 elems/byte (use packed helpers)
    case DataType::u4:   return 0;  // sub-byte: 2 elems/byte (use packed helpers)
    case DataType::q5_1: return 0;  // block-packed: 32 elems / 24-byte block
    }
    return 0;
}

/// Storage unit of one row step: the element size for ordinary types, or the
/// block size for block-packed types (q5_1 → 24 bytes). Zero means the type
/// has no single-unit storage (s4/u4 nibbles).
constexpr size_t data_type_stride_unit(DataType dt) noexcept {
    return is_block_packed_dtype(dt) ? q5_1_block_bytes : data_type_size(dt);
}

/// Bytes occupied by one row of @p cols logical elements.
///
/// Ordinary types: cols × elem_size. Sub-byte types: ceil(cols / 2).
/// Q5_1: ceil(cols / 32) × 24.
constexpr size_t data_type_row_bytes(DataType dt, int64_t cols) noexcept {
    if (cols <= 0) {
        return 0;
    }
    if (dt == DataType::q5_1) {
        return static_cast<size_t>((cols + q5_1_block_elems - 1) / q5_1_block_elems)
             * q5_1_block_bytes;
    }
    if (dt == DataType::s4 || dt == DataType::u4) {
        return static_cast<size_t>((cols + 1) / 2);
    }
    return static_cast<size_t>(cols) * data_type_size(dt);
}

/// Whether a DataType represents quantized integer storage that uses
/// scale/zero_point dequantization (s8, u8, s4, u4).
///
/// FP8 types (f8_e4m3, f8_e5m2) are NOT included here — they are
/// floating-point formats that don't use integer zero_point semantics.
/// Neither is q5_1: its per-block (d, m) are embedded in the data itself, so
/// it needs no QuantParams and is not part of this scale/zero_point family.
constexpr bool is_quantized_dtype(DataType dt) noexcept {
    switch (dt) {
    case DataType::s8:
    case DataType::u8:
    case DataType::s4:
    case DataType::u4:
        return true;
    default:
        return false;
    }
}

/// Returns the bit width of a single element of the given DataType.
/// Essential for packed (sub-byte) types: s4/u4 → 4 bits, s8/u8 → 8 bits.
/// Returns 0 for unknown types.
constexpr int data_type_bits(DataType dt) noexcept {
    switch (dt) {
    case DataType::s4:  return 4;
    case DataType::u4:  return 4;
    case DataType::q5_1: return 5;
    case DataType::s8:  return 8;
    case DataType::u8:  return 8;
    case DataType::f8_e4m3: return 8;
    case DataType::f8_e5m2: return 8;
    case DataType::f16: return 16;
    case DataType::bf16: return 16;
    case DataType::f32: return 32;
    case DataType::s32: return 32;
    case DataType::s64: return 64;
    }
    return 0;
}

}  // namespace nnops
