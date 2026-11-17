#pragma once
/// @file bench_knobs.hpp
/// @brief Experiment-only switches for the test and benchmark binaries.
///
/// A bench stores to these atomics directly and the decision point in the
/// backend does one relaxed load. Nothing here is an application-facing tuning
/// API: the values are process-wide, they default to the automatic route, and
/// the arms they select are only the ones a bench compares. The header lives
/// under src/ so it is not part of the installed interface.

#include <atomic>

namespace nnops::detail {

/// LinearAttention route: 0 automatic, -1 force the per-token loop, 1 force the
/// chunked (WY) route.
inline std::atomic<int> linear_attention_route{0};

/// Spatial pooling: 0 automatic (the straight-line 3x3 route when it is
/// eligible), 1 force the rolled window walk.
inline std::atomic<int> pooling_3x3_roll{0};

/// CUDA GEMM split-K: -1 automatic, 0 disable, N > 1 force N splits (clamped by
/// the same rules as the auto choice).
inline std::atomic<int> cuda_splitk{-1};

/// CUDA f16 GEMM: 0 automatic (the hand-written PTX kernel once its pose holds),
/// -1 force the CuTe engine, 1 force the PTX kernel. The pose gate still applies
/// to the forced value: alpha, beta, bias, epilogue and dtype are checked first.
inline std::atomic<int> cuda_ptx_h{0};

/// CUDA f32 GEMM: 0 automatic (the hand-written SIMT kernel once its grid fills
/// the machine), -1 force the CuTe engine, 1 force the SIMT kernel.
inline std::atomic<int> cuda_ptx_f{0};

/// CUDA int8 GEMM: 0 automatic (the hand-written PTX tile once its pose holds),
/// -1 force the CuTe engine, 1 force the PTX tile, N >= 2 force it with menu row
/// N - 2. The pose gate still applies: destination kind, alignment and the
/// transposed-side fills are checked first.
inline std::atomic<int> cuda_ptx_i8{0};

/// CPU int8: 1 hides NEON i8mm (SMMLA) so the SDOT route is taken. The feature
/// scan runs once, so set this before the first CpuFeatures::get().
inline std::atomic<int> disable_i8mm{0};

/// CPU SME: 1 hides the SME GEMM route so the NEON routes are taken. Unlike the
/// feature scan this is read on every availability query, so a bench or a test
/// can flip it between calls to compare the two engines on one binary.
inline std::atomic<int> disable_sme{0};

}  // namespace nnops::detail
