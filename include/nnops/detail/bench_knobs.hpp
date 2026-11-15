#pragma once
/// @file bench_knobs.hpp
/// @brief Experiment-only switches. Production code never reads the environment:
///        a test or benchmark stores to these atomics directly, and the decision
///        point does one relaxed load.
///
/// This is not an application-facing tuning API — the values are process-wide and
/// the arms they select are the ones a bench compares. Keeping the knobs off the
/// environment matters on Windows in particular, where a per-call getenv costs
/// microseconds and can swamp a small operator.

#include <atomic>

namespace nnops::detail {

/// LinearAttention route: 0 automatic, -1 force the per-token loop, 1 force the
/// chunked (WY) route.
inline std::atomic<int> linear_attention_route{0};

/// Channel-last pooling: 0 automatic (the straight-line 3x3 route when it is
/// eligible), 1 force the rolled window walk.
inline std::atomic<int> pooling_3x3_roll{0};

/// Channel-first depthwise: 0 automatic (the straight-line 3x3 interior when it
/// is eligible), 1 force the rolled tap loop.
inline std::atomic<int> depthwise_3x3_roll{0};

/// CUDA GEMM split-K: -1 automatic, 0 disable, N > 1 force N splits (clamped by
/// the same rules as the auto choice).
inline std::atomic<int> cuda_splitk{-1};

/// CPU int8: 1 hides NEON i8mm (SMMLA) so the SDOT route is taken. The feature
/// scan runs once, so set this before the first CpuFeatures::get().
inline std::atomic<int> disable_i8mm{0};

}  // namespace nnops::detail
