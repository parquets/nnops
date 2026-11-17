#pragma once
/// @file sme_gemm.hpp
/// @brief AArch64 SME GEMM micro-kernels (f32, and f16 with f32 accumulation).
///
/// SME replaces the vector FMA with outer products into the ZA matrix: one
/// `FMOPA ZA.S` adds a 16x16 rank-1 update, so a k-step costs one instruction
/// per 16-column output tile instead of one per 8-lane vector. SME2's four ZA
/// tiles hold a 16x64 block, and the K loop issues four independent FMOPAs per
/// k so the pipeline never waits on one tile's dependency chain.
///
/// The layouts are the ones the instruction wants, and they are *not* the NEON
/// pack's, so SME owns its packing path rather than reusing the 12/16-column
/// panels:
///   A: [m/16][k][16]   — the 16 rows of an M tile are contiguous per k
///   B: [n/64][k][64]   — 64 contiguous columns per k
/// C: row-major, any ldc.
///
/// B is blocked so the four tiles read one 256-byte (f32) / 128-byte (f16)
/// stream per k instead of four streams `kc*16` apart: on an M4 that is the
/// difference between ~700 and ~1125 GFLOPS f32. Padding past @p nc is zeroed
/// by the pack; the kernel's predicates keep it out of the accumulator anyway.
///
/// f16 uses the hardware two-way dot (`svmopa_za32_f16_m`), which consumes two
/// K per instruction: 512 MACs per MOPA against 256 for the f32 form. The
/// instruction reads lane pairs, so its operands are *pre-interleaved at pack
/// time* — A at [row][k,k+1], B at [col][k,k+1] — and the K loop is then one
/// 64-byte load per operand and one MOPA, exactly the f32 shape. Zipping in
/// the loop instead costs five permutes per K pair that contend with the MOPA
/// issue slot, which is what made the loop-built version slower than f32.
///
/// The MOPA predicate must be b16 — a b32 predicate activates only the even
/// lanes and silently drops half the products. An odd kc pads the last K pair
/// with zeros at pack time.
///
/// Only SVL 64 bytes (16 f32 lanes) is supported; available() says so.
///
/// The whole file is compiled only when the build enabled SME
/// (`NNOPS_ENABLE_SME`), and every entry point is called behind available().

#include "nnops/detail/half.hpp"

#include <cstdint>

namespace nnops::backend::cpu::sme {

#if defined(NNOPS_ARCH_SME)

/// Columns of B in one blocked pass; also the pack's block width.
constexpr int kNB = 64;

/// Elements one packed B slab occupies: f32 `[ceil(nc/64)][kc][64]`, f16 the
/// same with K rounded up to a pair `[ceil(nc/64)][ceil(kc/2)][128]`.
inline int64_t packed_b_elems_f32(int nc, int kc) noexcept
{
    return static_cast<int64_t>((nc + kNB - 1) / kNB) * kc * kNB;
}
inline int64_t packed_b_elems_f16(int nc, int kc) noexcept
{
    return static_cast<int64_t>((nc + kNB - 1) / kNB) * ((kc + 1) / 2) * (2 * kNB);
}

/// Elements a packed A slab occupies: f32 `[m/16][kc][16]`, f16
/// `[m/16][ceil(kc/2)][32]`.
inline int64_t packed_a_elems_f32(int mc, int kc) noexcept
{
    return static_cast<int64_t>((mc + 15) / 16) * kc * 16;
}
inline int64_t packed_a_elems_f16(int mc, int kc) noexcept
{
    return static_cast<int64_t>((mc + 15) / 16) * ((kc + 1) / 2) * 32;
}

/// True when the binary carries the kernels, the CPU has SME, and the streaming
/// vector length is the 64 bytes the kernels are written for.
bool available() noexcept;

/// Pack @p mc rows of A (row stride @p lds elements) into [m/16][kc][16]; the
/// f16 form stores K pairs as [m/16][ceil(kc/2)][32] and zeroes an odd tail.
/// Padding rows past @p mc are left undefined: the kernel masks them. @p trans
/// reads A as the physical K×M layout (MatMul's transpose_a).
void pack_a_f32(int mc, int kc, float* dst, const float* src, int64_t lds,
                bool trans = false) noexcept;
void pack_a_f16(int mc, int kc, half* dst, const half* src, int64_t lds,
                bool trans = false) noexcept;

/// Pack @p nc columns of B (row-major, stride @p ldb, @p kc rows) into
/// [n/64][kc][64] (f16: [n/64][ceil(kc/2)][128], K pairs interleaved per
/// column); padding columns and an odd K tail are zeroed. @p trans reads B as
/// the physical N×K layout (MatMul's transpose_b).
void pack_b_f32(int nc, int kc, float* dst, const float* src, int64_t ldb,
                bool trans = false) noexcept;
void pack_b_f16(int nc, int kc, half* dst, const half* src, int64_t ldb,
                bool trans = false) noexcept;

/// C[m][n] += A[m][k] * B[k][n] for one k-block, mc x nc x kc.
///
/// @p a and @p b are the packed layouts above. @p zero_mode overwrites C
/// instead of accumulating; @p col_bias (per output column) seeds the
/// accumulator and overwrites C. Every stored element is clamped to
/// [@p lo, @p hi].
/// @p b_block_stride is the element distance between 64-column blocks: a
/// per-call pack lays them kc*64 apart, a whole-K prepacked blob K*64. 0 means
/// the per-call form.
void gemm_f32(int mc, int nc, int kc, float* c, int ldc, const float* a, const float* b,
              bool zero_mode, float lo, float hi, const float* col_bias,
              int64_t b_block_stride = 0) noexcept;
/// @p b_block_stride is the element distance between 64-column blocks: a
/// per-call pack lays them ceil(kc/2)*128 apart, a whole-K prepacked blob
/// ceil(K/2)*128. 0 means the per-call form.
void gemm_f16(int mc, int nc, int kc, half* c, int ldc, const half* a, const half* b,
              bool zero_mode, float lo, float hi, const half* col_bias,
              int64_t b_block_stride = 0) noexcept;

/// int8 (s8 x s8 -> int32). One `SMOPA ZA.S` is a 4-way dot: 16 rows x 16
/// columns x 4 K = 1024 MACs, four times the f32 form. The pack interleaves K
/// in groups of four, which is how the instruction reads its lanes: operand
/// lane 4i..4i+3 are row (column) i's four K values.
///   A: [m/16][ceil(kc/4)][64]    B: [n/64][ceil(kc/4)][256]
inline int64_t packed_a_elems_i8(int mc, int kc) noexcept
{
    return static_cast<int64_t>((mc + 15) / 16) * ((kc + 3) / 4) * 64;
}
inline int64_t packed_b_elems_i8(int nc, int kc) noexcept
{
    return static_cast<int64_t>((nc + kNB - 1) / kNB) * ((kc + 3) / 4) * (4 * kNB);
}

/// @p xor_bias applies the u8-activation shift (XOR 0x80) the NEON pack uses;
/// padded K slots stay 0, which is the identity in the shifted domain.
void pack_a_i8(int mc, int kc, int8_t* dst, const int8_t* src, int64_t lds, bool trans,
               bool xor_bias) noexcept;
void pack_b_i8(int nc, int kc, int8_t* dst, const int8_t* src, int64_t ldb,
               bool trans) noexcept;

/// C[m][n] += A[m][k] * B[k][n] as int32, clamped to [@p lo, @p hi].
void gemm_i8(int mc, int nc, int kc, int32_t* c, int ldc, const int8_t* a, const int8_t* b,
             bool zero_mode, int32_t lo, int32_t hi) noexcept;

#else

inline bool available() noexcept { return false; }

#endif

} // namespace nnops::backend::cpu::sme
