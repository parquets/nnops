#pragma once
/// @file openmp_backend.hpp
/// @brief A CpuBackend driven by OpenMP — a provider for the hooks, not a
/// dependency of the library.
///
/// `CpuBackend` (parallel_for.hpp) deliberately owns no thread pool: callers
/// install the hooks of whatever scheduler they already have. This header is
/// one concrete provider for the common case — an OpenMP runtime — and it is
/// header-only and compile-time optional. Without `-fopenmp` / `/openmp`
/// (`_OPENMP` undefined) every entry point here degrades to the sequential
/// backend, so callers can install it unconditionally.
///
/// The adapter points at a persistent thread team, which is what the operators
/// want: `ctx.cpu.run` is called once per kernel (and several times per call in
/// the conv and norm families), so a backend that spawns workers per call
/// measures its own start-up cost on short calls. The team is created on the
/// first parallel region and reused afterwards.
///
/// Thread ids: `thread_id` returns `omp_get_thread_num()`, so scratch indexed by
/// `current_thread_id()` works exactly as it does with any other pool, provided
/// the region actually runs with the requested team size. Inside a *serialized*
/// nested region (the default when a host app calls nnops from its own OpenMP
/// parallel loop) every worker reports 0 — correct but aliased. Callers that run
/// nested should install `num_threads` only: `thread_slot_count()` then returns
/// 0 and the operators fall back to per-task scratch.

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

/// Build a CpuBackend that runs `ctx.cpu.run` as an OpenMP parallel for.
///
/// @param nthreads team size; <= 0 uses `omp_get_max_threads()` (and follows
///                 `OMP_NUM_THREADS`). The count is pinned per region, so a
///                 caller asking for 4 threads gets 4 even if the process
///                 default is larger.
///
/// Up to the machine's fast-core count the split is `schedule(static)`:
/// operators already cut their work into roughly equal blocks sized from
/// `thread_count()`, so one contiguous chunk per worker keeps each worker's
/// scratch and its slice of the operands local. Past that count the provider
/// over-subscribes the grid and lets workers claim it (`auto_items`).
///
/// Wait policy: this only supplies the hooks — the OpenMP runtime's own knobs
/// still apply. On MSVC (vcomp) the default is an active spin, which on a hybrid
/// CPU costs a great deal at some team sizes: 8T measured ~2x slower than
/// `OMP_WAIT_POLICY=PASSIVE` on the i5-13600KF this was written on, while 1T /
/// 4T / 20T were unaffected. That is an embedder-side setting; an application
/// that calls in continuously may prefer the spin, a benchmark ladder does not.
///
/// How the team is fed: one equal share per worker (the historical behavior), or
/// an over-subscribed task grid that workers claim as they finish.
///
/// The second arm exists for hybrid parts. A `static` split hands every worker
/// the same share regardless of what that worker can do, so on a 6P+8E such as
/// the i5-13600KF the region ends when the slowest E-core finishes its share and
/// the P-cores idle through the tail: conv measured 6.9x over 6 P-cores, 3.2x
/// at 8 threads (two E-cores holding an eighth of the work at a third of the
/// speed) and 5.3x at 20 — less than the 6-thread peak. Four items per worker
/// plus a dynamic claim lets the fast cores take the tail and reaches 9.2x on
/// the same layer, and the runners are 1.37x faster at 8 threads and 1.16x at
/// 20. Below the fast-core count the extra tasks only shrink the kernels'
/// blocks, so the provider stays static there — `auto_items` is where that line
/// is drawn, and `items_per_thread = 1` turns it off entirely. The item count is
/// reported through `num_threads()` so operators cut their grid finer without
/// knowing about the topology — which is also why `thread_slots()` exists:
/// per-thread scratch must stay at the real worker count while `num_threads()`
/// reports the inflated grid.
struct OpenMPSplit {
    /// Tasks per worker the operators should cut their grid into (>= 1), or 0
    /// for `auto_items(team)` — 1 unless the team is wider than the machine's
    /// fast cores, which is the shape that wants a split at all.
    int items_per_thread = 0;
    /// Workers claim the next task instead of taking a fixed share. Forced on
    /// when the item count is left to `auto_items` and it over-subscribes the
    /// grid: a finer grid only pays when the items are claimed, not sliced.
    bool dynamic = false;
};

/// Tasks per worker for a team of @p team: `ceil(team / fast cores)`, and 1
/// when the team fits on the fast cores or the platform reports no classes.
///
/// An equal static share over more workers than fast cores ends on the slowest
/// worker — on the i5-13600KF (6 P + 8 E) a conv layer runs 6.9x over its
/// 1-thread time at 6 threads and 3.2x at 8, because two E-cores hold an eighth
/// of the work at a third of the speed. Four items per worker on 20 threads
/// takes the same layer to 9.2x. Below the fast-core count there is nothing to
/// rebalance and the extra tasks only shrink the kernels' blocks.
inline int auto_items(int team)
{
    const int fast = simd::CpuFeatures::get().fast_core_count();
    if (fast <= 0 || team <= fast) { return 1; }
    return (team + fast - 1) / fast;
}

/// Without OpenMP this returns a default (sequential) CpuBackend.
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
        // The team size is pinned, deliberately: it is the *item* count that
        // should follow the pool, never the other way round. Sizing the team
        // from the work (min(nthreads, end - begin)) looks right for an idle
        // worker and measures far worse on vcomp: a region whose num_threads
        // differs from the one before it tears the team down and rebuilds it,
        // and at 20 threads that is ~0.7 ms per transition. On the ResNet-18
        // runner it cost 6 ms of an 18 ms forward — every op that followed a
        // narrow one paid for the rebuild. So a caller with few items should
        // give the pool more tasks (the conv plan splits its oc-blocks for
        // exactly this, conv2d_implicit_gemm.cpp) rather than shrink the team.
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

/// One equal share per worker on a persistent team — the behavior every caller
/// had before OpenMPSplit existed.
inline CpuBackend openmp_backend(int nthreads = 0) {
    return openmp_backend(nthreads, OpenMPSplit{});
}

}  // namespace nnops
