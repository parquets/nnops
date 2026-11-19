#pragma once
/// @file mma_sme.hpp
/// @brief AArch64 SME micro-kernels: the shared tile geometry and runtime gate.
///
/// The SME sibling of mma_f32.hpp / pack_f32.hpp. The kernels live in
/// mma_{f32,f16,i8}_sme.hpp, their layouts in pack_{f32,f16,i8}_sme.hpp, and
/// the `_sme` suffix marks the SME engine. Like the NEON kernels, the bodies are
/// header-only inline functions; each one carries a function-level
/// `target("+sme2")` attribute, so the surrounding library stays baseline
/// AArch64 and only the emitted kernel bodies use SME.
///
/// SME replaces the vector FMA with outer products into the ZA matrix: one
/// `FMOPA ZA.S` adds a 16x16 rank-1 update, so a k-step costs one instruction
/// per 16-column output tile instead of one per 8-lane vector. SME2's four ZA
/// tiles hold a 16x64 block, and the K loop issues four independent FMOPAs per
/// k so the pipeline never waits on one tile's dependency chain.
///
/// The layouts are the ones the instruction wants, and they are *not* the NEON
/// pack's, so the SME family owns its packing path rather than reusing the
/// 12/16-column panels:
///   A: [m/16][k][16]   — the 16 rows of an M tile are contiguous per k
///   B: [n/64][k][64]   — 64 contiguous columns per k
/// C: row-major, any ldc.
///
/// B is blocked so the four tiles read one 256-byte (f32) / 128-byte (f16)
/// stream per k instead of four streams `kc*16` apart: on an M4 that is the
/// difference between ~700 and ~1125 GFLOPS f32. Padding past @p nc is zeroed
/// by the pack; the kernel's predicates keep it out of the accumulator anyway.
///
/// Only SVL 64 bytes (16 f32 lanes) is supported; available_sme() says so.
///
/// The kernels compile only when the build enabled SME (`NNOPS_ENABLE_SME`
/// defines `NNOPS_ARCH_SME`), and every entry point is called behind
/// available_sme(). Without the macro available_sme() is a false constant, so
/// an arch-neutral caller still compiles.

#if defined(NNOPS_ARCH_SME)

#include <atomic>
#include <cstddef>

#include <arm_sme.h>

#include "nnops/detail/simd/cpu_features.hpp"

#include "detail/bench_knobs.hpp"

#endif

namespace nnops::backend::cpu::aarch64 {

/// Rows in one ZA tile (SVL 64 B / sizeof(f32)).
inline constexpr int sme_mr = 16;

/// Columns in one ZA tile, i.e. the N extent one MOPA updates.
inline constexpr int sme_za_n = 16;

/// Columns of B in one blocked pass — the four ZA tiles; also the int8 pack's
/// block width (the SMOPA kernel is 16x64).
inline constexpr int sme_nr_max = 4 * sme_za_n;

/// Columns of B in one float pass: the 2x2 ZA tile's width. The float pack uses
/// it as the block width too, so a tile's B stream is contiguous; at 64 the
/// kernel read half of every 256-byte K row and lost about half the throughput.
inline constexpr int sme_fp_nr = 2 * sme_za_n;

#if defined(NNOPS_ARCH_SME)

namespace sme_detail {

/// The streaming vector length, asked of the hardware in streaming mode. Safe
/// only after cpu_has_sme().
__arm_locally_streaming __attribute__((target("+sme2")))
inline size_t probe_svl() noexcept
{
    return static_cast<size_t>(svcntsb());
}

/// The f32 output clamp shared by the f32 and f16 kernels.
__arm_locally_streaming __attribute__((target("+sme2"), always_inline))
inline svfloat32_t clamp_v(svbool_t pg, svfloat32_t v, svfloat32_t lo, svfloat32_t hi) noexcept
{
    return svmin_f32_x(pg, svmax_f32_x(pg, v, lo), hi);
}

} // namespace sme_detail

/// True when the binary carries the kernels, the CPU has SME, and the streaming
/// vector length is the 64 bytes the kernels are written for.
inline bool available_sme() noexcept
{
    // Device-wide verdict, cached; the experiment knob below is read per call.
    static const bool on = [] {
        if (!simd::cpu_has_sme()) {
            return false;
        }
        // The kernels assume 16 f32 lanes; a wider SVL needs a retiled kernel.
        if (simd::cpu_sme_svl() != 0 && simd::cpu_sme_svl() != 64) {
            return false;
        }
        return sme_detail::probe_svl() == 64;
    }();
    return on && ::nnops::detail::disable_sme.load(std::memory_order_relaxed) == 0;
}

#else

inline bool available_sme() noexcept { return false; }

#endif

} // namespace nnops::backend::cpu::aarch64
