#pragma once
/// @file openmp_backend.hpp
/// @brief A CpuBackend driven by OpenMP — a provider for the hooks, not a
/// dependency of the library.
///
/// CpuBackend (parallel_for.hpp) owns no thread pool: callers install the hooks
/// of whatever scheduler they already have. This provider is header-only and
/// compile-time optional — without -fopenmp / /openmp (_OPENMP undefined) every
/// entry point degrades to the sequential backend, so it installs unconditionally.
/// It targets a persistent team, created on the first region and reused, because
/// ctx.cpu.run is called per kernel and per-call spawning measures its own
/// start-up on short calls. thread_id is omp_get_thread_num(); in a serialized
/// nested region (a host app already inside an OpenMP loop) every worker reports
/// 0 — install num_threads only there, which makes thread_slot_count() return 0
/// so operators fall back to per-task scratch.
///
/// Up to the fast-core count the split is schedule(static), one contiguous chunk
/// per worker; past it the grid is over-subscribed and claimed. An equal share
/// over more threads than fast cores ends on the slowest worker: on the
/// i5-13600KF (6P+8E) a conv layer runs 3.2x at 8 threads (the E-cores hold a
/// slow eighth) and 5.3x at 20, versus 9.2x with four items per worker. Below the
/// fast-core count extra tasks only shrink the kernels' blocks, so auto_items()
/// keeps it static there. The item count is reported through num_threads() so
/// operators cut a finer grid; thread_slots() stays the real worker count so
/// scratch is not sized by it.
///
/// Wait policy is the embedder's: vcomp's default spin measured ~2x slower at 8T
/// than OMP_WAIT_POLICY=PASSIVE on the i5-13600KF (1T / 4T / 20T unaffected).

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <algorithm>
#include <cstdint>

#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/cpu_features.hpp"

namespace nnops {

/// True when this translation unit was compiled with OpenMP enabled.
inline constexpr bool openmp_available() noexcept {
#if defined(_OPENMP)
    return true;
#else
    return false;
#endif
}

/// How the team is fed: one equal share per worker, or an over-subscribed item
/// grid workers claim (see the file header for the measurement).
struct OpenMPSplit {
    /// Tasks per worker (>= 1), or 0 for auto_items(team).
    int items_per_thread = 0;
    /// Workers claim the next item instead of taking a fixed share. Forced on
    /// when auto_items over-subscribes the grid.
    bool dynamic = false;
};

/// Items per worker for a team of @p team: `ceil(team / fast cores)`, and 1
/// when the team fits the fast cores or the platform reports no classes.
inline int auto_items(int team)
{
    const int fast = simd::CpuFeatures::get().fast_core_count();
    if (fast <= 0 || team <= fast) { return 1; }
    return (team + fast - 1) / fast;
}

/// Build a CpuBackend that runs `ctx.cpu.run` as an OpenMP parallel for.
///
/// @param nthreads team size; <= 0 uses `omp_get_max_threads()` (and follows
///                 `OMP_NUM_THREADS`).
inline CpuBackend openmp_backend(int nthreads, OpenMPSplit split) {
    CpuBackend b;
#if defined(_OPENMP)
    const int team = nthreads > 0 ? nthreads : omp_get_max_threads();
    const bool items_auto = split.items_per_thread <= 0;
    const int items = items_auto ? auto_items(team)
                                  : std::max(split.items_per_thread, 1);
    const bool dynamic = split.dynamic || (items_auto && items > 1);
    b.parallel_for = [team, dynamic](int64_t begin, int64_t end,
                                     const ParallelForBody& body) {
        if (end <= begin) { return; }
        // Team size is pinned deliberately — the *item* count follows the pool,
        // never the other way round. Sizing the team from the work
        // (min(nthreads, end - begin)) measures far worse on vcomp: a region
        // whose num_threads differs from the last tears down and rebuilds the
        // team, ~0.7 ms per transition at 20 threads, costing 6 ms of an 18 ms
        // ResNet-18 forward. So a caller with few items gives the pool more
        // tasks (the conv plan splits its oc-blocks for exactly this,
        // conv2d_implicit_gemm.cpp) rather than shrink the team.
        if (dynamic) {
#pragma omp parallel for schedule(dynamic, 1) num_threads(team)
            for (int64_t i = begin; i < end; ++i) {
                body(i);
            }
        } else {
#pragma omp parallel for schedule(static) num_threads(team)
            for (int64_t i = begin; i < end; ++i) {
                body(i);
            }
        }
    };
    b.num_threads = [team, items] { return team * items; };
    b.thread_id = [] { return omp_get_thread_num(); };
    b.thread_slots = [team] { return team; };
#else
    (void)nthreads;
    (void)split;
#endif
    return b;
}

/// One equal share per worker on a persistent team (no split).
inline CpuBackend openmp_backend(int nthreads = 0) {
    return openmp_backend(nthreads, OpenMPSplit{});
}

}  // namespace nnops
