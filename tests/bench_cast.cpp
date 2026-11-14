/// @file bench_cast.cpp
/// @brief Cast throughput over the dtype pairs a mixed-precision graph uses.
///
/// Cast is a pure data-movement op, so the rows report GB/s (source bytes plus
/// destination bytes per call) rather than elements. The interesting pairs are
/// the float islands (f32↔f16, the boundary a lowering inserts), the index
/// direction (f32→s32 and s32→f32), and the quantize boundary (s8→f32 and the
/// saturating f32→s8). The f32→f32 row is the copy floor the rest can be held
/// against.
///
/// Every row is a 1-D tensor: the element tile is the parallel unit, so this is
/// the shape that used to leave the whole call on one thread.
///
/// Measurement follows the other benches: per-round min, median across rounds.
/// `NNOPS_BENCH_THREADS=1,4` pins the ladder.

#include "nnops/ops/cast.hpp"
#include "nnops/detail/half.hpp"
#include "common/bench_harness.hpp"
#include "common/cpu_pool.hpp"
#include "common/random_tensor.hpp"
#include "common/test_helpers.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>
#include <thread>
#include <vector>

using namespace nnops;
using half = nnops::backend::cpu::half;

namespace {

struct BenchPool {
    explicit BenchPool(int nthreads) : nthreads_(nthreads) {
        if (nnops::test::bench_use_openmp()) {
            cpu = nnops::openmp_backend(nthreads);
            return;
        }
        cpu.parallel_for = [this](int64_t begin, int64_t end, const ParallelForBody& body) {
            this->parallel_for(begin, end, body);
        };
        cpu.num_threads = [this]() { return nthreads_; };
        cpu.thread_id = []() { return current_thread_id_; };
    }

    void parallel_for(int64_t begin, int64_t end, const ParallelForBody& body) {
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
    }

    int nthreads_;
    CpuBackend cpu;
    inline static thread_local int current_thread_id_ = 0;
};

template <class Fn>
double time_ms(Fn&& fn, int iters) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) { fn(); }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() /
           std::max(iters, 1);
}

double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

std::vector<int> bench_threads() {
    std::vector<int> threads;
    if (const char* env = std::getenv("NNOPS_BENCH_THREADS")) {
        for (const char* p = env; *p;) {
            char* end = nullptr;
            const long v = std::strtol(p, &end, 10);
            if (end != p && v > 0) { threads.push_back(static_cast<int>(v)); }
            p = (end != p) ? end : p + 1;
            if (*p == ',') { ++p; }
        }
        if (!threads.empty()) { return threads; }
    }
    threads = {1};
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw > 1) { threads.push_back(hw); }
    return threads;
}

struct Case {
    DataType from;
    DataType to;
    const char* name;
};

size_t dtype_size(DataType d) {
    switch (d) {
    case DataType::f32: return 4;
    case DataType::f16: return 2;
    case DataType::s32: return 4;
    case DataType::s8:  return 1;
    default:            return 4;
    }
}

const char* dtype_name(DataType d) {
    switch (d) {
    case DataType::f32: return "f32";
    case DataType::f16: return "f16";
    case DataType::s32: return "s32";
    case DataType::s8:  return "s8";
    default:            return "?";
    }
}

void run_case(const Case& c, int64_t n, const std::vector<int>& threads) {
    auto [src_f, src_view] = test::make_random_tensor({n}, -1.0f, 1.0f, 7);
    (void)src_view;

    std::vector<float>    src32, dst32;
    std::vector<half>     src16, dst16;
    std::vector<int32_t>  src_i32, dst_i32;
    std::vector<int8_t>   src_i8, dst_i8;

    void* sp = nullptr;
    switch (c.from) {
    case DataType::f16: src16 = test::f32_to_f16(src_f); sp = src16.data(); break;
    case DataType::s32:
        src_i32.resize(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            src_i32[static_cast<size_t>(i)] =
                static_cast<int32_t>(src_f[static_cast<size_t>(i)] * 1000.0f);
        }
        sp = src_i32.data();
        break;
    case DataType::s8:
        src_i8.resize(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            src_i8[static_cast<size_t>(i)] = static_cast<int8_t>(i % 251 - 125);
        }
        sp = src_i8.data();
        break;
    default:
        src32 = src_f;
        sp = src32.data();
        break;
    }

    void* op_ptr = nullptr;
    switch (c.to) {
    case DataType::f16:   dst16.resize(static_cast<size_t>(n));   op_ptr = dst16.data(); break;
    case DataType::s32:   dst_i32.resize(static_cast<size_t>(n)); op_ptr = dst_i32.data(); break;
    case DataType::s8:    dst_i8.resize(static_cast<size_t>(n));  op_ptr = dst_i8.data(); break;
    default:              dst32.resize(static_cast<size_t>(n));   op_ptr = dst32.data(); break;
    }

    const int64_t shape[] = {n};
    TensorView in(std::span<const int64_t>(shape, 1), c.from, sp);
    TensorView out(std::span<const int64_t>(shape, 1), c.to, op_ptr);

    CastAttributes attrs;
    attrs.to = c.to;
    auto op = Cast::create(attrs, Backend::CPU);

    const TensorView ins[] = {in};
    const double bytes = static_cast<double>(n) *
        (static_cast<double>(dtype_size(c.from)) +
         static_cast<double>(dtype_size(c.to)));

    const int iters = n <= (int64_t{1} << 16) ? 50
                    : n <= (int64_t{1} << 20) ? 10 : 3;

    std::printf("  %s->%-4s %8.2f MB |", dtype_name(c.from), dtype_name(c.to),
                bytes / (1024.0 * 1024.0));
    for (int nt : threads) {
        BenchPool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

        std::vector<double> times;
        for (int round = 0; round < 3; ++round) {
            times.push_back(time_ms([&] { op->compute(out, ins, use_ctx, nullptr); }, iters));
        }
        const double ms = median_of(times);
        const double gbs = bytes / (ms * 1.0e6);
        std::printf("  %2dT %8.3f ms %7.1f GB/s |", nt, ms, gbs);
    }
    std::printf("\n");
}

}  // namespace

NNOPS_BENCH(cast_throughput) {
    std::printf("--- cast_throughput (1-D, element-tile parallel) ---\n");
    const std::vector<int> threads = bench_threads();
    const int64_t n = int64_t{1} << 22;   // 4M elements: 16 MB f32 source

    const Case cases[] = {
        {DataType::f32, DataType::f32, "copy"},
        {DataType::f32, DataType::f16, "down"},
        {DataType::f16, DataType::f32, "up"},
        {DataType::f32, DataType::s32, "index-down"},
        {DataType::s32, DataType::f32, "index-up"},
        {DataType::s8,  DataType::f32, "quant-up"},
        {DataType::f32, DataType::s8,  "quant-down"},
    };
    for (const Case& c : cases) { run_case(c, n, threads); }
}
