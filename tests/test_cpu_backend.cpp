/// @file test_cpu_backend.cpp
/// @brief Tests for the CpuBackend execution hooks (parallel_for / num_threads /
///        thread_id) and their default sequential fallback.

#include "nnops/core/compute_context.hpp"
#include "common/test_harness.hpp"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

using namespace nnops;

namespace {

// A minimal fixed-size pool that records, per worker, the thread id it ran
// under, and exposes itself as a CpuBackend for injection into ComputeContext.
struct RecordingPool {
    explicit RecordingPool(int nthreads) : nthreads_(nthreads) {
        cpu.parallel_for = [this](int64_t begin, int64_t end, const ParallelForBody& body) {
            std::atomic<int64_t> next{begin};
            std::vector<std::thread> workers;
            workers.reserve(static_cast<size_t>(nthreads_));
            for (int t = 0; t < nthreads_; ++t) {
                workers.emplace_back([&, t]() {
                    current_thread_id_ = t;
                    for (;;) {
                        int64_t i = next.fetch_add(1, std::memory_order_relaxed);
                        if (i >= end) { break; }
                        body(i);
                    }
                });
            }
            for (auto& w : workers) { w.join(); }
        };
        cpu.num_threads = [this]() { return nthreads_; };
        cpu.thread_id = []() { return current_thread_id_; };
    }

    int nthreads_;
    CpuBackend cpu;
    inline static thread_local int current_thread_id_ = -1;
};

}  // anonymous namespace

NNOPS_TEST(cpu_backend_sequential_fallback) {
    // A default ComputeContext has no hooks installed: run() is sequential and
    // the worker-count / thread-id accessors fall back to 1 / 0.
    ComputeContext ctx;
    NNOPS_EXPECT_EQ(ctx.cpu.thread_count(), 1);
    NNOPS_EXPECT_EQ(ctx.cpu.current_thread_id(), 0);

    int64_t sum = 0;
    ctx.cpu.run(0, 10, [&](int64_t i) { sum += i; });
    NNOPS_EXPECT_EQ(sum, 45);
}

NNOPS_TEST(cpu_backend_thread_count_and_id) {
    RecordingPool pool(4);
    NNOPS_EXPECT_EQ(pool.cpu.thread_count(), 4);

    std::atomic<int> bad{0};
    std::atomic<int64_t> calls{0};

    pool.cpu.run(0, 1000, [&](int64_t) {
        // The thread_id hook must report a valid, in-range worker id.
        const int id = pool.cpu.current_thread_id();
        if (id < 0 || id >= 4) {
            bad.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        calls.fetch_add(1, std::memory_order_relaxed);
    });

    NNOPS_EXPECT_EQ(bad.load(), 0);
    NNOPS_EXPECT_EQ(calls.load(), 1000);
}
