#pragma once
/// @file parallel_for.hpp
/// @brief Type aliases for the external parallel_for interface.

#include <cstdint>
#include <functional>

namespace nnops {

/// Body function called for each work item: body(index).
using ParallelForBody = std::function<void(int64_t)>;

/// Parallel-for hook signature:
///   pf(start, end, body) — process range [begin, end) with body(index).
/// If nullptr or empty, the implementation falls back to sequential execution.
using ParallelForFn = std::function<void(int64_t, int64_t, const ParallelForBody&)>;

}  // namespace nnops
