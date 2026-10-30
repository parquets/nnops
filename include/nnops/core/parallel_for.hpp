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

/// Worker-count hook: returns the number of worker threads (>= 1).
using ThreadCountFn = std::function<int()>;

/// Thread-id hook: returns the current worker's id in [0, thread_count()).
using ThreadIdFn = std::function<int()>;

/// Scratch-slot hook: returns how many distinct ids `thread_id` can report.
/// Defaults to `thread_count()`, which is only right when the two are the same
/// number — a backend that hands its workers *more items than workers* (an
/// over-subscribed task grid) still has `thread_count()` distinct ids, and
/// scratch indexed by `current_thread_id()` must stay sized to that.
using ThreadSlotsFn = std::function<int()>;

/// CPU execution backend: the parallelism hooks an operator needs to schedule
/// work and to size/index per-thread scratch. All members are optional; a null
/// member means "sequential / unknown". nnops owns no thread pool — callers
/// install the hooks they already have.
struct CpuBackend {
    /// Process range [begin, end) with body(index).
    ParallelForFn parallel_for = nullptr;
    /// Number of worker threads (>= 1).
    ThreadCountFn num_threads = nullptr;
    /// Current worker thread id (0-based).
    ThreadIdFn thread_id = nullptr;
    /// Distinct worker ids `thread_id` can report. Unset means `num_threads`,
    /// which holds for every pool that runs one item per worker; a backend that
    /// over-subscribes its task grid sets this to the real worker count so
    /// per-thread scratch is not sized by the item count.
    ThreadSlotsFn thread_slots = nullptr;

    /// Run body over [begin, end): parallel when a hook is installed,
    /// sequential otherwise.
    void run(int64_t begin, int64_t end, const ParallelForBody& body) const {
        if (parallel_for) {
            parallel_for(begin, end, body);
        } else {
            for (int64_t i = begin; i < end; ++i) body(i);
        }
    }

    /// Worker thread count, or 1 when no hook reports it.
    int thread_count() const { return num_threads ? num_threads() : 1; }

    /// Current worker thread id (0-based), or 0 when no hook reports it.
    int current_thread_id() const { return thread_id ? thread_id() : 0; }

    /// Number of per-thread scratch slots available: `thread_slots()` (else
    /// thread_count()) when both num_threads and thread_id are installed, so
    /// operators can size their scratch by worker count and index it with
    /// current_thread_id(); else 0, meaning operators must fall back to
    /// per-task scratch indexing.
    int thread_slot_count() const {
        if (!(num_threads && thread_id)) { return 0; }
        return thread_slots ? thread_slots() : thread_count();
    }
};

}  // namespace nnops
