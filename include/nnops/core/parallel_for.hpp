#pragma once
/// @file parallel_for.hpp
/// @brief Type aliases for the external parallel_for interface.
///
/// nnops owns no thread pool: CpuBackend holds optional hooks that callers fill
/// with whatever scheduler they already have, and a null hook means "sequential
/// / unknown". thread_slots() differs from num_threads() for a backend that
/// hands its workers more items than workers (an over-subscribed task grid):
/// thread_id still reports only thread_slots() distinct ids, so scratch indexed
/// by current_thread_id() must be sized to that, not to the item count.

#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace nnops {

/// A non-owning callable view: the object pointer and one invoker, no
/// allocation and no per-callable class. The callable must outlive the view.
///
/// This is the body type rather than std::function for a measured reason: every
/// `ctx.cpu.run(..., [&]{})` call site used to instantiate std::function's
/// _Func_impl_no_alloc machinery — a ctor, a copy, a move, a destructor, a
/// call operator and a target-type accessor, each with a mangled name that
/// embeds the lambda's, which in turn embeds the enclosing function's. That one
/// family was 38% of the archive's symbol-name bytes. The view needs the single
/// invoker below, so a call site costs one symbol instead of eight.
template <typename Sig>
class FunctionRef;

template <typename R, typename... A>
class FunctionRef<R(A...)> {
public:
    template <typename F,
              typename = std::enable_if_t<
                  !std::is_same_v<std::remove_cvref_t<F>, FunctionRef>>>
    FunctionRef(F&& f) noexcept
        : obj_(const_cast<void*>(static_cast<const void*>(std::addressof(f)))),
          fn_([](void* o, A... a) -> R {
              return (*static_cast<std::remove_reference_t<F>*>(o))(
                  static_cast<A&&>(a)...);
          })
    {}

    R operator()(A... a) const { return fn_(obj_, static_cast<A&&>(a)...); }

private:
    void* obj_;
    R (*fn_)(void*, A...);
};

/// Body function: body(index).
using ParallelForBody = FunctionRef<void(int64_t)>;

/// pf(begin, end, body) — process [begin, end); empty means sequential.
using ParallelForFn = std::function<void(int64_t, int64_t, const ParallelForBody&)>;

/// Grid size (>= 1) — the worker count unless the backend over-subscribes.
using ThreadCountFn = std::function<int()>;

/// Current worker id in [0, thread_slots()).
using ThreadIdFn = std::function<int()>;

/// Distinct ids thread_id can report; defaults to thread_count().
using ThreadSlotsFn = std::function<int()>;

/// CPU execution hooks: scheduling and per-thread scratch sizing/indexing.
struct CpuBackend {
    /// Process range [begin, end) with body(index).
    ParallelForFn parallel_for = nullptr;
    /// Grid size (>= 1); the real worker count unless over-subscribed.
    ThreadCountFn num_threads = nullptr;
    /// Current worker thread id (0-based).
    ThreadIdFn thread_id = nullptr;
    /// Real worker count for scratch sizing; unset means num_threads.
    ThreadSlotsFn thread_slots = nullptr;

    /// Run [begin, end): parallel with a hook, sequential otherwise.
    void run(int64_t begin, int64_t end, const ParallelForBody& body) const {
        if (parallel_for) {
            parallel_for(begin, end, body);
        } else {
            for (int64_t i = begin; i < end; ++i) body(i);
        }
    }

    /// Grid size, or 1 with no hook.
    int thread_count() const { return num_threads ? num_threads() : 1; }

    /// Concurrent workers: thread_slots() when installed, else the grid. This is
    /// the count that bounds scratch, cache residency and tile sizes — an
    /// over-subscribed grid hands one worker several items and says nothing
    /// about how many blocks can be live at once.
    int worker_count() const {
        const int slots = thread_slot_count();
        return slots > 0 ? slots : thread_count();
    }

    /// Current worker id, or 0 with no hook.
    int current_thread_id() const { return thread_id ? thread_id() : 0; }

    /// Scratch slots: thread_slots()/thread_count() when both hooks are
    /// installed, else 0 (callers fall back to per-task scratch).
    int thread_slot_count() const {
        if (!(num_threads && thread_id)) { return 0; }
        return thread_slots ? thread_slots() : thread_count();
    }
};

}  // namespace nnops
