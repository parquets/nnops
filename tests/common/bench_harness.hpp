#pragma once
/// @file bench_harness.hpp
/// @brief Minimal benchmarking utilities for nnops.
///
/// No external dependencies. Uses std::chrono for timing.

#include <chrono>
#include <functional>
#include <iostream>
#include <string>

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

}  // namespace nnops::test
