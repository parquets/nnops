/// @file cuda_bench_helper.hpp
/// @brief CUDA-event timing and the report lines the CUDA benchmarks share.
///
/// A CUDA call is asynchronous, so wall-clock timing around compute() would
/// measure the launch. Each sample therefore brackets the call with two events
/// on the default stream and waits for the closing one. Samples are collected
/// one per call and the report uses their median, which drops the occasional
/// hitch from the host or the clock; a warm-up run of each case pays the
/// module load and the first-touch of the kernel's constant bank.
///
/// The table reports milliseconds per call and GFLOPS, computed from the
/// operator's own FLOP count for the geometry it was handed — so a Conv2D row
/// and a MatMul row with the same MACs are directly comparable.
///
/// Only compiled under NNOPS_HAS_CUDA.

#pragma once

#ifdef NNOPS_HAS_CUDA

#include "cuda_test_helper.hpp"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace nnops::test {

/// Submit one call without waiting: the caller's events bracket it. Only the
/// status is checked here, so a failing launch surfaces at the next synchronise.
template <class Op>
void cuda_launch(Op& op, TensorView& out, std::span<const TensorView> ins)
{
    const Status st = op.compute({&out, 1}, ins, cuda_context(), nullptr);
    if (st != Status::ok) {
        throw std::runtime_error(std::string("cuda compute: ") + to_string(st));
    }
}

/// Median GPU milliseconds of `launch`, after `warmup` untimed calls.
template <class F>
double cuda_median_ms(F&& launch, int warmup = 3, int samples = 11)
{
    CudaBuffer::sync();
    for (int i = 0; i < warmup; ++i) { launch(); }
    CudaBuffer::sync();

    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    NNOPS_CUDA_CHECK(cudaEventCreate(&begin));
    NNOPS_CUDA_CHECK(cudaEventCreate(&end));

    std::vector<double> ms;
    ms.reserve(static_cast<size_t>(samples));
    for (int i = 0; i < samples; ++i) {
        NNOPS_CUDA_CHECK(cudaEventRecord(begin));
        launch();
        NNOPS_CUDA_CHECK(cudaEventRecord(end));
        NNOPS_CUDA_CHECK(cudaEventSynchronize(end));
        float elapsed = 0.0f;
        NNOPS_CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin, end));
        ms.push_back(static_cast<double>(elapsed));
    }
    NNOPS_CUDA_CHECK(cudaEventDestroy(begin));
    NNOPS_CUDA_CHECK(cudaEventDestroy(end));

    const size_t mid = ms.size() / 2;
    std::nth_element(ms.begin(), ms.begin() + static_cast<long>(mid), ms.end());
    return ms[mid];
}

/// One report row: geometry, milliseconds, and the throughput it implies.
inline void print_cuda_row(const char* tag, const std::string& geometry,
                           double ms, double gflops)
{
    std::printf("  %-18s %-34s %9.4f ms %9.2f GFLOPS\n", tag, geometry.c_str(),
                ms, gflops);
}

/// 2·M·N·K over the elapsed time, in GFLOPS.
inline double cuda_gflops(double ms, double flops)
{
    return ms > 0.0 ? flops / (ms * 1e6) : 0.0;
}

/// "# 12.3 GFLOPS" style header for a section.
inline void print_cuda_header(const char* title)
{
    std::printf("\n=== %s ===\n", title);
}

}  // namespace nnops::test

#endif  // NNOPS_HAS_CUDA
