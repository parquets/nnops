#pragma once

/// @file common_utils.hpp
/// @brief Small cross-platform helpers shared by the micro-kernel headers:
///        inlining controls, the restrict qualifier and align_up.
///
/// One header rather than one per knob, because every kernel needs
/// NNOPS_RESTRICT and most want the inlining controls too.
///
/// NNOPS_FORCEINLINE makes an always-inline function; NNOPS_NOINLINE refuses
/// inlining. On MSVC only __forceinline/__declspec(noinline) carry weight, on
/// GCC/Clang the attributes do.

#ifndef NNOPS_FORCEINLINE
#if defined(_MSC_VER)
#define NNOPS_FORCEINLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define NNOPS_FORCEINLINE inline __attribute__((always_inline))
#else
#define NNOPS_FORCEINLINE inline
#endif
#endif

/// Refuses inlining. Used to keep a rarely-taken variant of a micro-kernel out
/// of the hot function: MSVC allocates registers per function, and a second
/// unrolled k-loop inlined next to the first one makes it spill accumulators
/// that then round-trip through the stack on the FMA critical path.
#ifndef NNOPS_NOINLINE
#if defined(_MSC_VER)
#define NNOPS_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define NNOPS_NOINLINE __attribute__((noinline))
#else
#define NNOPS_NOINLINE
#endif
#endif

/// NNOPS_RESTRICT is the restrict qualifier for kernel pointer parameters.

#ifndef NNOPS_RESTRICT
#if defined(_MSC_VER)
#define NNOPS_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define NNOPS_RESTRICT __restrict__
#else
#define NNOPS_RESTRICT
#endif
#endif

namespace nnops::backend::cpu {

/// Round n up to a multiple of AlignBytes (a power of two).
template <int AlignBytes>
constexpr int align_up(int n) {
    return (n + AlignBytes - 1) & ~(AlignBytes - 1);
}

}  // namespace nnops::backend::cpu

