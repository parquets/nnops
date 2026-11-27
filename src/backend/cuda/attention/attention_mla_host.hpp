#pragma once
/// @file attention_mla_host.hpp
/// @brief The one thing the absorbed-MLA tile tells the host about itself: how
///        many calls it has actually carried, as opposed to declined.
///
/// The tile is chosen by shape (absorbed, f16, 576 x 512, a head group of at
/// least sixteen) and may still decline a call the device cannot hold. A
/// decline falls back to the general route, which answers *correctly*, so a
/// test that only compares outputs cannot tell the two apart — and neither can
/// a benchmark whose row silently moved. Tests and benches read this counter
/// around a call instead (the same contract as la_wy_route_runs, see
/// linear_attention_wy_host.hpp).

#include <cstdint>

namespace nnops::backend::cuda {

/// Monotone count of absorbed-MLA calls this process has run on the dedicated
/// tile (launch and merge both succeeded).
int64_t attention_mla_runs();

} // namespace nnops::backend::cuda
