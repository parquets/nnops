#pragma once
/// @file data_type.hpp
/// @brief DataType enum and compile-time traits for the nnops library.
///
/// q4_0, q4_1 and q8_0 are llama.cpp block formats covering 32 values per block.
/// q4_0/q4_1 share 16 nibble bytes (value j in the low nibble of byte j, value
/// j + 16 in the high); q8_0 stores 32 signed bytes.
/// q4_0 is 18 bytes (fp16 d only) and decodes to d·(q − 8); q4_1 is 20 bytes
/// (fp16 d and m) and decodes to d·q + m; q8_0 is 34 bytes (fp16 d, 32 int8)
/// and decodes to d·q. None has a per-element storage
/// unit, so data_type_size() returns 0 — like the 4-bit types — and every
/// row/stride computation must go through data_type_row_bytes() /
/// data_type_stride_unit() rather than multiplying by the element size. All are
/// self-describing, so QuantParams is not used for them; decode with pack_q4()
/// in <nnops/detail/q4.hpp> (vectorized pack kernels in
/// src/backend/cpu/x86_64/pack_q4.hpp and aarch64/pack_q4.hpp).
///
/// is_quantized_dtype() covers s8/u8/s4/u4 only: fp8 (f8_e4m3/f8_e5m2) are float
/// formats with no integer zero_point, and the block formats carry their own
/// per-block scale (and offset).

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
    q4_1    = 11, ///< llama.cpp Q4_1 block format (see file header)
    q4_0    = 12, ///< llama.cpp Q4_0 block format (see file header)
    q8_0    = 13, ///< llama.cpp Q8_0 block format (see file header)
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
struct DataTypeTraits<DataType::q4_1> {
    using ctype = uint8_t;   ///< block storage; 32 values per 20-byte block
    static constexpr size_t size = 0;  ///< block-packed — use data_type_block_bytes
    static constexpr const char* name = "q4_1";
};

template <>
struct DataTypeTraits<DataType::q4_0> {
    using ctype = uint8_t;   ///< block storage; 32 values per 18-byte block
    static constexpr size_t size = 0;  ///< block-packed — use data_type_block_bytes
    static constexpr const char* name = "q4_0";
};

template <>
struct DataTypeTraits<DataType::q8_0> {
    using ctype = uint8_t;   ///< block storage; 32 values per 34-byte block
    static constexpr size_t size = 0;  ///< block-packed — use data_type_block_bytes
    static constexpr const char* name = "q8_0";
};

// ---- Block-packed formats ----
// No per-element storage unit: use data_type_row_bytes()/stride_unit(), not size.

/// Elements covered by one Q4_0/Q4_1/Q8_0 block.
inline constexpr int64_t q4_block_elems = 32;

/// Bytes of one Q4_1 block: fp16 d, fp16 m, 16 nibbles.
inline constexpr size_t q4_1_block_bytes = 20;

/// Bytes of one Q4_0 block: fp16 d, 16 nibbles.
inline constexpr size_t q4_0_block_bytes = 18;

/// Bytes of one Q8_0 block: fp16 d, 32 signed bytes.
inline constexpr size_t q8_0_block_bytes = 34;

/// Back-compat aliases; every block format shares the 32-value block.
inline constexpr int64_t q4_1_block_elems = q4_block_elems;
inline constexpr int64_t q4_0_block_elems = q4_block_elems;
inline constexpr int64_t q8_0_block_elems = q4_block_elems;

/// True for types stored in fixed-size blocks rather than whole bytes per element.
constexpr bool is_block_packed_dtype(DataType dt) noexcept {
    return dt == DataType::q4_0 || dt == DataType::q4_1 || dt == DataType::q8_0;
}

/// Bytes of one block of a block-packed dtype; 0 for every other type.
constexpr size_t data_type_block_bytes(DataType dt) noexcept {
    switch (dt) {
    case DataType::q4_0: return q4_0_block_bytes;
    case DataType::q4_1: return q4_1_block_bytes;
    case DataType::q8_0: return q8_0_block_bytes;
    default:             return 0;
    }
}

/// Values covered by one block of a block-packed dtype; 0 for every other type.
constexpr int64_t data_type_block_elems(DataType dt) noexcept {
    switch (dt) {
    case DataType::q4_0:
    case DataType::q4_1:
    case DataType::q8_0: return q4_block_elems;
    default:             return 0;
    }
}

/// Size in bytes of one element of @p dt.
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
    case DataType::q4_0: return 0;  // block-packed: 32 elems / 18-byte block
    case DataType::q4_1: return 0;  // block-packed: 32 elems / 20-byte block
    case DataType::q8_0: return 0;  // block-packed: 32 elems / 34-byte block
    }
    return 0;
}

/// Storage unit of one row step: element size, or block size for block-packed
/// (q4_0 → 18, q4_1 → 20, q8_0 → 34 bytes); 0 for nibble types, which have no
/// single-unit storage.
constexpr size_t data_type_stride_unit(DataType dt) noexcept {
    return is_block_packed_dtype(dt) ? data_type_block_bytes(dt) : data_type_size(dt);
}

/// Bytes in one row of @p cols elements; sub-byte and block-packed round up.
constexpr size_t data_type_row_bytes(DataType dt, int64_t cols) noexcept {
    if (cols <= 0) {
        return 0;
    }
    if (const size_t bb = data_type_block_bytes(dt); bb != 0) {
        const int64_t be = data_type_block_elems(dt);
        return static_cast<size_t>((cols + be - 1) / be) * bb;
    }
    if (dt == DataType::s4 || dt == DataType::u4) {
        return static_cast<size_t>((cols + 1) / 2);
    }
    return static_cast<size_t>(cols) * data_type_size(dt);
}

/// True for integer quantized storage that uses scale/zero_point dequantization
/// (s8, u8, s4, u4).
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

/// Bit width of one element; essential for packed types (s4/u4 → 4). 0 if unknown.
constexpr int data_type_bits(DataType dt) noexcept {
    switch (dt) {
    case DataType::s4:  return 4;
    case DataType::u4:  return 4;
    case DataType::q4_0: return 4;
    case DataType::q4_1: return 4;
    case DataType::q8_0: return 8;
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
