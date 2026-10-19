/// @file bench_harness.cpp
/// @brief Benchmark runner — bench_main() entry point.

#include "bench_harness.hpp"

#include <cstring>

int main(int argc, char** argv) {
    // Optional substring filter on the benchmark name; no argument runs all.
    const std::string filter = (argc > 1) ? argv[1] : "";

    std::cout << "=== nnops benchmarks ===" << std::endl;
    for (auto& e : nnops::test::bench_registry()) {
        if (!filter.empty() && e.name.find(filter) == std::string::npos) {
            continue;
        }
        std::cout << "--- " << e.name << " ---" << std::endl;
        e.fn();
    }
    return 0;
}
