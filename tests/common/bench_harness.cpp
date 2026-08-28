/// @file bench_harness.cpp
/// @brief Benchmark runner — bench_main() entry point.

#include "bench_harness.hpp"

int main() {
    std::cout << "=== nnops benchmarks ===" << std::endl;
    for (auto& fn : nnops::test::bench_registry()) {
        fn();
    }
    return 0;
}
