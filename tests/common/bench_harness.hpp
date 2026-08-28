#pragma once
/// @file bench_harness.hpp
/// @brief Minimal benchmarking utilities for nnops.
///
/// No external dependencies. Uses std::chrono for timing.

#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace nnops::test {

struct BenchResult {
    std::string name;
    double elapsed_ms;
    int64_t iterations;
};

/// Run a benchmark function `iterations` times and report timing.
template <typename F>
BenchResult benchmark(const std::string& name, int64_t iterations, F&& fn) {
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    return {name, ms, iterations};
}

inline void report_bench(const BenchResult& r) {
    std::cout << "  BENCH " << r.name
              << ": " << r.elapsed_ms << " ms"
              << " (" << r.iterations << " iters)" << std::endl;
}

// ---- benchmark registration (mirrors the test framework) ----
using BenchFn = std::function<void()>;

inline std::vector<BenchFn>& bench_registry() {
    static std::vector<BenchFn> reg;
    return reg;
}

struct BenchRegistrar {
    explicit BenchRegistrar(BenchFn fn) { bench_registry().push_back(std::move(fn)); }
};

}  // namespace nnops::test

#define NNOPS_BENCH(name) \
    static void _nnops_bench_##name(); \
    static ::nnops::test::BenchRegistrar _nnops_bench_reg_##name(_nnops_bench_##name); \
    static void _nnops_bench_##name()
