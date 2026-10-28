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
/// The split is `schedule(static)`: operators already cut their work into
/// roughly equal blocks sized from `thread_count()`, so one contiguous chunk
/// per worker keeps each worker's scratch and its slice of the operands local.
/// `dynamic` was measured against it on a hybrid 6P+8E part and came out 10-20%
/// behind on conv (the E-cores finish their single block early either way, and
/// dynamic also loses the locality), so the balanced split stays.
///
/// Wait policy: this only supplies the hooks — the OpenMP runtime's own knobs
/// still apply. On MSVC (vcomp) the default is an active spin, which on a hybrid
/// CPU costs a great deal at some team sizes: 8T measured ~2x slower than
/// `OMP_WAIT_POLICY=PASSIVE` on the i5-13600KF this was written on, while 1T /
/// 4T / 20T were unaffected. That is an embedder-side setting; an application
/// that calls in continuously may prefer the spin, a benchmark ladder does not.
///
/// Without OpenMP this returns a default (sequential) CpuBackend.
inline CpuBackend openmp_backend(int nthreads = 0) {
    CpuBackend b;
#if defined(_OPENMP)
    b.parallel_for = [nthreads](int64_t begin, int64_t end, const ParallelForBody& body) {
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
        // exactly this, conv2d_im2col.cpp) rather than shrink the team.
        const int nt = nthreads > 0 ? nthreads : omp_get_max_threads();
#pragma omp parallel for schedule(static) num_threads(nt)
        for (int64_t i = begin; i < end; ++i) {
            body(i);
        }
    };
    b.num_threads = [nthreads] { return nthreads > 0 ? nthreads : omp_get_max_threads(); };
    b.thread_id = [] { return omp_get_thread_num(); };
#else
    (void)nthreads;
#endif
    return b;
}

}  // namespace nnops
