/// @file bench_cuda_linear.cpp
/// @brief CUDA Linear throughput — the dense GEMM with its [N, K] weight.
///
/// Linear is MatMul with transpose_b fixed, so on the device it runs the same
/// 64x64x64 tile as bench_cuda_matmul; the rows here are the shapes a graph
/// actually feeds it (a hidden-size projection, an FFN expansion, a classifier
/// head) plus the batched form [B, S, K] whose leading dims flatten into M.
///
/// The weight is [N, K] and the product is X·Wᵀ, which is the transposed-B copy
/// atom — the route that reads the filter as rows instead of columns. GFLOPS is
/// again 2·M·N·K over the flattened M, so a row is directly comparable with the
/// MatMul table's `tb` column.
///
/// The last section prices the output write-back: bias alone, and bias with
/// add_to, against the plain product. add_to on this operator reads the
/// destination *after* the activation, so it is a genuine extra global load.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "detail/bench_knobs.hpp"
#include "nnops/ops/linear.hpp"
#include "common/bench_harness.hpp"
#include "common/cuda_test_helper.hpp"
#include "common/cuda_bench_helper.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <string>
#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

std::string linear_geometry(const std::vector<int64_t>& x_shape, int64_t n)
{
    std::string s;
    for (size_t i = 0; i < x_shape.size(); ++i) {
        s += (i ? "x" : "") + std::to_string(x_shape[i]);
    }
    s += " -> " + std::to_string(n);
    return s;
}

/// One event-timed launch, for the interleaved A/B below (the shared
/// `cuda_median_ms` runs one arm to completion, which is what the drift between
/// two processes makes incomparable).
template <typename F> double timed_launch(F&& launch, cudaEvent_t begin, cudaEvent_t end)
{
    NNOPS_CUDA_CHECK(cudaEventRecord(begin));
    launch();
    NNOPS_CUDA_CHECK(cudaEventRecord(end));
    NNOPS_CUDA_CHECK(cudaEventSynchronize(end));
    float elapsed = 0.0f;
    NNOPS_CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin, end));
    return double(elapsed);
}

double median_of(std::vector<double> ms)
{
    const size_t mid = ms.size() / 2;
    std::nth_element(ms.begin(), ms.begin() + std::ptrdiff_t(mid), ms.end());
    return ms[mid];
}

/// One Linear call, timed with events. Returns GFLOPS.
double run_linear(const std::vector<int64_t>& x_shape, int64_t n, DataType dt,
                  bool bias, bool add_to, Epilogue ep, const char* tag)
{
    LinearAttributes attrs;
    attrs.epilogue = ep;
    attrs.add_to = add_to;

    const int64_t k = x_shape.back();
    int64_t m = 1;
    for (size_t i = 0; i + 1 < x_shape.size(); ++i) { m *= x_shape[i]; }

    HostTensor hx = make_host_random(x_shape, dt, false, 11, -0.5f, 0.5f);
    HostTensor hw = make_host_random(std::vector<int64_t>{n, k}, dt, false, 13, -0.25f, 0.25f);

    std::vector<int64_t> out_shape = x_shape;
    out_shape.back() = n;
    HostTensor hc = add_to
                        ? make_host_random(out_shape, dt, false, 37, 0.0f, 0.5f)
                        : make_host_zeroed(out_shape, dt, false);

    DeviceTensor dx = to_device(hx);
    DeviceTensor dw = to_device(hw);
    DeviceTensor dc = to_device(hc);
    DeviceTensor dbias;
    if (bias) {
        HostTensor hbias = make_host_random(std::vector<int64_t>{n}, dt, false, 19, -0.25f, 0.25f);
        dbias = to_device(hbias);
    }

    auto op = Linear::create(attrs, Backend::CUDA);
    std::vector<TensorView> ins = {dx.view, dw.view};
    if (bias) { ins.push_back(dbias.view); }

    TensorView out = dc.view;
    auto launch = [&] { cuda_launch(*op, out, ins); };
    const double ms = cuda_median_ms(launch);

    const double flops = 2.0 * double(m) * double(n) * double(k);
    print_cuda_row(tag, linear_geometry(x_shape, n), ms, cuda_gflops(ms, flops));
    return cuda_gflops(ms, flops);
}

}  // namespace

NNOPS_BENCH(cuda_linear_f32) {
    print_cuda_header("CUDA Linear f32");

    const int64_t rows[] = {1, 16, 128, 512, 1024, 2048};
    for (int64_t m : rows) {
        run_linear({m, 1024}, 1024, DataType::f32, true, false, Epilogue{}, "f32");
    }
    // An FFN expansion and a classifier head at a token-batch size.
    run_linear({512, 4096}, 4096, DataType::f32, true, false, Epilogue{}, "f32");
    run_linear({512, 4096}, 11008, DataType::f32, true, false, Epilogue{}, "f32");
    run_linear({128, 768}, 1000, DataType::f32, true, false, Epilogue{}, "f32");
}

NNOPS_BENCH(cuda_linear_f16) {
    print_cuda_header("CUDA Linear f16 (tensor core)");

    const int64_t rows[] = {1, 16, 128, 512, 1024, 2048};
    for (int64_t m : rows) {
        run_linear({m, 1024}, 1024, DataType::f16, true, false, Epilogue{}, "f16");
    }
    run_linear({512, 4096}, 4096, DataType::f16, true, false, Epilogue{}, "f16");
    run_linear({512, 4096}, 11008, DataType::f16, true, false, Epilogue{}, "f16");
}

NNOPS_BENCH(cuda_linear_batched) {
    print_cuda_header("CUDA Linear batched [B, S, K]");
    for (auto shape : {std::vector<int64_t>{8, 128, 1024},
                       std::vector<int64_t>{32, 128, 1024},
                       std::vector<int64_t>{2, 512, 4096}}) {
        run_linear(shape, 1024, DataType::f32, true, false, Epilogue{}, "f32");
        run_linear(shape, 1024, DataType::f16, true, false, Epilogue{}, "f16");
    }
}

NNOPS_BENCH(cuda_linear_writeback) {
    print_cuda_header("CUDA Linear output write-back (2048x1024x1024)");

    Epilogue relu;
    relu.type = EpilogueActivateType::Relu;

    for (DataType dt : {DataType::f32, DataType::f16}) {
        const char* tag = dt == DataType::f32 ? "f32" : "f16";
        const double plain = run_linear({2048, 1024}, 1024, dt, false, false,
                                        Epilogue{}, tag);
        const double with_bias = run_linear({2048, 1024}, 1024, dt, true, false,
                                            Epilogue{}, tag);
        const double residual = run_linear({2048, 1024}, 1024, dt, true, true, relu,
                                           tag);
        std::printf("  %s: bias %.1f%%, bias+relu+add_to %.1f%% of the plain product\n",
                    tag, 100.0 * (plain - with_bias) / plain,
                    100.0 * (plain - residual) / plain);
    }
}

/// The small Linear shapes whose split choice changed: their tiled grid is
/// under twenty blocks, so the starved rule now cuts the reduction. The cut and
/// the uncut product are interleaved, because the planes can cost more than the
/// extra blocks buy at these sizes.
NNOPS_BENCH(cuda_linear_split_ab) {
    print_cuda_header("CUDA Linear f16 small shapes: split-K A/B");
    struct Shape {
        std::vector<int64_t> x;
        int64_t n;
    };
    const Shape shapes[] = {{{1, 1024}, 1024},  {{16, 1024}, 1024}, {{128, 1024}, 1024},
                            {{16, 4096}, 4096}, {{2048, 1024}, 1024}};
    auto& knob = nnops::detail::cuda_splitk;

    for (const Shape& s : shapes) {
        const int64_t k = s.x.back();
        int64_t m = 1;
        for (size_t i = 0; i + 1 < s.x.size(); ++i) { m *= s.x[i]; }
        HostTensor hx = make_host_random(s.x, DataType::f16, false, 11, -0.5f, 0.5f);
        HostTensor hw = make_host_random(std::vector<int64_t>{s.n, k}, DataType::f16, false, 13,
                                         -0.25f, 0.25f);
        std::vector<int64_t> out_shape = s.x;
        out_shape.back() = s.n;
        HostTensor hc = make_host_zeroed(out_shape, DataType::f16, false);
        DeviceTensor dx = to_device(hx);
        DeviceTensor dw = to_device(hw);
        DeviceTensor dc = to_device(hc);
        LinearAttributes attrs;
        auto op = Linear::create(attrs, Backend::CUDA);
        std::vector<TensorView> ins = {dx.view, dw.view};
        TensorView out = dc.view;
        auto launch = [&] { cuda_launch(*op, out, ins); };

        cudaEvent_t begin = nullptr, end = nullptr;
        NNOPS_CUDA_CHECK(cudaEventCreate(&begin));
        NNOPS_CUDA_CHECK(cudaEventCreate(&end));
        for (int v : {-1, 0}) {
            knob.store(v, std::memory_order_relaxed);
            launch();
        }
        CudaBuffer::sync();

        std::vector<double> ms_auto;
        std::vector<double> ms_off;
        for (int round = 0; round < 12; ++round) {
            if (round % 2 == 0) {
                knob.store(-1, std::memory_order_relaxed);
                ms_auto.push_back(timed_launch(launch, begin, end));
                knob.store(0, std::memory_order_relaxed);
                ms_off.push_back(timed_launch(launch, begin, end));
            } else {
                knob.store(0, std::memory_order_relaxed);
                ms_off.push_back(timed_launch(launch, begin, end));
                knob.store(-1, std::memory_order_relaxed);
                ms_auto.push_back(timed_launch(launch, begin, end));
            }
        }
        knob.store(-1, std::memory_order_relaxed);
        NNOPS_CUDA_CHECK(cudaEventDestroy(begin));
        NNOPS_CUDA_CHECK(cudaEventDestroy(end));

        const double flops = 2.0 * double(m) * double(s.n) * double(k);
        const std::string geo = linear_geometry(s.x, s.n);
        const double am = median_of(ms_auto);
        const double om = median_of(ms_off);
        print_cuda_row("f16 auto", geo, am, cuda_gflops(am, flops));
        print_cuda_row("f16 split off", geo, om, cuda_gflops(om, flops));
        std::printf("    ratio auto/off %.3f\n", am / om);
    }
}

#endif  // NNOPS_HAS_CUDA
