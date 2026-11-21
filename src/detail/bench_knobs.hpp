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

/// CUDA Conv2D's 1x1 route: -1 automatic, 0 keep the implicit-GEMM conv kernel,
/// 1 hand the unit-stride 1x1 to the dense GEMM.
inline std::atomic<int> cuda_conv_pw{-1};

/// CPU int8: 1 hides NEON i8mm (SMMLA) so the SDOT route is taken. The feature
/// scan runs once, so set this before the first CpuFeatures::get().
inline std::atomic<int> disable_i8mm{0};

/// CPU SME: 1 hides the SME GEMM route so the NEON routes are taken. Unlike the
/// feature scan this is read on every availability query, so a bench or a test
/// can flip it between calls to compare the two engines on one binary.
inline std::atomic<int> disable_sme{0};

}  // namespace nnops::detail
