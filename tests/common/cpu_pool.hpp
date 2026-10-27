#pragma once
/// @file bench_pool.hpp
/// @brief Which CpuBackend the multithreaded bench rows run on.
///
/// The benchmarks carry two interchangeable schedulers for the same hooks:
///
///   OpenMP        — a persistent team (nnops/core/openmp_backend.hpp), the
///                   backend an embedding application would normally install.
///   SimplePool    — one thread per call, spawned and joined inside
///                   parallel_for; measures the caller-side start-up too.
///
/// `NNOPS_BENCH_OMP` picks: unset or "1" uses OpenMP when the build has it,
/// "0"/"false" forces SimplePool. Both live in one binary, so the comparison is
/// same-instrument: the only variable is the backend. When the build has no
/// OpenMP the row falls back to SimplePool (and says so if it was asked for).

#include "nnops/core/openmp_backend.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace nnops::test {

/// True when the benches should drive `ctx.cpu` with OpenMP.
inline bool bench_use_openmp() {
    static const bool use = [] {
        const char* s = std::getenv("NNOPS_BENCH_OMP");
        if (s != nullptr && (std::strcmp(s, "0") == 0 || std::strcmp(s, "false") == 0)) {
            return false;
        }
        if (!nnops::openmp_available()) {
            if (s != nullptr) {
                std::fprintf(stderr,
                             "NNOPS_BENCH_OMP is set but this build has no OpenMP "
                             "(configure with -DNNOPS_OPENMP=ON); using SimplePool\n");
            }
            return false;
        }
        return true;
    }();
    return use;
}

/// One line naming the backend, for the header of a threaded table.
inline const char* bench_pool_name() {
    return bench_use_openmp() ? "openmp" : "simple-pool";
}

}  // namespace nnops::test
