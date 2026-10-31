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

#include "nnops/ops/linear.hpp"
#include "common/bench_harness.hpp"
#include "common/cuda_test_helper.hpp"
#include "common/cuda_bench_helper.hpp"

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

#endif  // NNOPS_HAS_CUDA
