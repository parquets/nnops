/// @file bench_harness.cpp
/// @brief Benchmark runner — bench_main() entry point.

#include "bench_harness.hpp"
#include "cpu_pool.hpp"

#include <cstring>
#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    // Optional substring filter on the benchmark name; no argument runs all.
    const std::string filter = (argc > 1) ? argv[1] : "";

    // Name the scheduler: every multi-thread row below is measured on it, and
    // the two backends are one env var apart (NNOPS_BENCH_OMP).
    std::cout << "=== nnops benchmarks (cpu pool: " << nnops::test::bench_pool_name()
              << ") ===" << std::endl;
    for (auto& e : nnops::test::bench_registry()) {
        if (!filter.empty() && e.name.find(filter) == std::string::npos) {
            continue;
        }
        std::cout << "--- " << e.name << " ---" << std::endl;
        // An arm that checks its timing against a reference throws on a
        // mismatch; that must not take the rest of the run down with it.
        try {
            e.fn();
        } catch (const std::exception& err) {
            std::cout << "!! " << e.name << ": " << err.what() << std::endl;
        }
    }
    return 0;
}
