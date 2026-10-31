/// @file bench_cuda_matmul.cpp
/// @brief CUDA MatMul throughput — the CuTe GEMM across sizes, dtypes and layouts.
///
/// One row per geometry, reporting GPU milliseconds per call and GFLOPS over
/// 2·M·N·K. The kernel behind it is one CTA tile (64x64x64, 3-stage cp.async,
/// 128 threads) with an m16n8k16 tensor-core MMA on f16 and a 32x32x16 FMA twin
/// on f32, so the tables answer two questions: how close f32 gets to the FMA
/// peak, and how much the tensor-core path buys over it at the same shape.
///
/// Sections:
///  - f32 and f16 square-and-oblong sweeps, from one wave (128³) to a graph
///    that fills the GPU (~2048³). The oblong rows are the LLM shapes: a long
///    K against a small M is the decode GEMM, a large M against equal N and K
///    is the prefill one.
///  - the four transpose combinations on a size that is large enough for the
///    copy atom to matter and small enough to run red-hot.
///  - batched GEMMs, where the batch axis multiplies the grid and each matrix
///    is small: the row shows how much the per-matrix tail costs when there are
///    not enough CTAs to fill 38 SMs from one matrix alone.
///  - the epilogue, as the delta between a plain product and bias+Relu: the
///    accumulator never leaves registers, so what a row shows is the extra
///    registers and the activation's arithmetic, not memory traffic.
///
/// Reading the numbers: an RTX 3060 Ti (38 SMs, 1.67 GHz) peaks at 16.2 TFLOPS
/// f32 and 32.4 TFLOPS f16 through the tensor cores, and a 64x64x64 tile with a
/// single MMA shape is a first-generation pipeline — the point of these rows is
/// the shape of the curve, not a CUTLASS-grade roofline. Timing is CUDA-event
/// based with a warm-up, median of 11 samples.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/matmul.hpp"
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

std::string matmul_geometry(int64_t m, int64_t n, int64_t k, bool ta, bool tb)
{
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%lldx%lldx%lld%s%s", (long long)m, (long long)n,
                  (long long)k, ta ? " ta" : "", tb ? " tb" : "");
    return buf;
}

/// One dense product on the device, timed with events. Returns GFLOPS.
double run_dense(int64_t m, int64_t n, int64_t k, DataType dt, bool ta, bool tb,
                 Epilogue ep, bool bias, const char* tag)
{
    MatMulAttributes attrs;
    attrs.transpose_a = ta;
    attrs.transpose_b = tb;
    attrs.epilogue = ep;

    const std::vector<int64_t> a_shape =
        ta ? std::vector<int64_t>{k, m} : std::vector<int64_t>{m, k};
    const std::vector<int64_t> b_shape =
        tb ? std::vector<int64_t>{n, k} : std::vector<int64_t>{k, n};

    HostTensor ha = make_host_random(a_shape, dt, false, 17, -0.5f, 0.5f);
    HostTensor hb = make_host_random(b_shape, dt, false, 29, -0.5f, 0.5f);
    HostTensor hc = make_host_zeroed(std::vector<int64_t>{m, n}, dt, false);

    DeviceTensor da = to_device(ha);
    DeviceTensor db = to_device(hb);
    DeviceTensor dc = to_device(hc);
    DeviceTensor dbias;
    if (bias) {
        HostTensor hbias = make_host_random(std::vector<int64_t>{n}, dt, false, 31, -0.25f, 0.25f);
        dbias = to_device(hbias);
    }

    auto op = MatMul::create(attrs, Backend::CUDA);
    std::vector<TensorView> ins = {da.view, db.view};
    if (bias) { ins.push_back(dbias.view); }

    TensorView out = dc.view;
    auto launch = [&] { cuda_launch(*op, out, ins); };
    const double ms = cuda_median_ms(launch);

    const double flops = 2.0 * double(m) * double(n) * double(k);
    print_cuda_row(tag, matmul_geometry(m, n, k, ta, tb), ms, cuda_gflops(ms, flops));
    return cuda_gflops(ms, flops);
}

/// A batched product: A [b, m, k] against B [b, k, n].
double run_batched(int64_t b, int64_t m, int64_t n, int64_t k, DataType dt,
                   const char* tag)
{
    HostTensor ha = make_host_random(std::vector<int64_t>{b, m, k}, dt, false, 19, -0.5f, 0.5f);
    HostTensor hb = make_host_random(std::vector<int64_t>{b, k, n}, dt, false, 23, -0.5f, 0.5f);
    HostTensor hc = make_host_zeroed(std::vector<int64_t>{b, m, n}, dt, false);

    DeviceTensor da = to_device(ha);
    DeviceTensor db = to_device(hb);
    DeviceTensor dc = to_device(hc);

    auto op = MatMul::create(MatMulAttributes{}, Backend::CUDA);
    const TensorView ins[] = {da.view, db.view};
    TensorView out = dc.view;
    auto launch = [&] { cuda_launch(*op, out, ins); };
    const double ms = cuda_median_ms(launch);

    const double flops = 2.0 * double(b) * double(m) * double(n) * double(k);
    char geo[96];
    std::snprintf(geo, sizeof(geo), "b%lld %lldx%lldx%lld", (long long)b,
                  (long long)m, (long long)n, (long long)k);
    print_cuda_row(tag, geo, ms, cuda_gflops(ms, flops));
    return cuda_gflops(ms, flops);
}

void sweep(DataType dt, const char* tag)
{
    const int64_t sizes[] = {128, 256, 512, 1024, 2048};
    for (int64_t s : sizes) {
        run_dense(s, s, s, dt, false, false, Epilogue{}, false, tag);
    }
    // A decode-shaped product and a prefill-shaped one.
    run_dense(16, 4096, 4096, dt, false, false, Epilogue{}, false, tag);
    run_dense(512, 1024, 4096, dt, false, false, Epilogue{}, false, tag);
}

}  // namespace

NNOPS_BENCH(cuda_matmul_f32) {
    print_cuda_header("CUDA MatMul f32");
    sweep(DataType::f32, "f32");
}

NNOPS_BENCH(cuda_matmul_f16) {
    print_cuda_header("CUDA MatMul f16 (tensor core)");
    sweep(DataType::f16, "f16");
}

NNOPS_BENCH(cuda_matmul_transposes) {
    print_cuda_header("CUDA MatMul transpose combinations (1024^3)");
    for (int flags = 0; flags < 4; ++flags) {
        const bool ta = (flags & 1) != 0;
        const bool tb = (flags & 2) != 0;
        run_dense(1024, 1024, 1024, DataType::f32, ta, tb, Epilogue{}, false,
                  "f32");
        run_dense(1024, 1024, 1024, DataType::f16, ta, tb, Epilogue{}, false,
                  "f16");
    }
}

NNOPS_BENCH(cuda_matmul_batched) {
    print_cuda_header("CUDA MatMul batched");
    for (int64_t b : {2, 8, 32}) {
        run_batched(b, 128, 128, 128, DataType::f32, "f32");
        run_batched(b, 128, 128, 128, DataType::f16, "f16");
    }
    // One big matrix per batch: the batch axis is nearly free here.
    run_batched(4, 1024, 1024, 1024, DataType::f16, "f16");
}

NNOPS_BENCH(cuda_matmul_epilogue) {
    print_cuda_header("CUDA MatMul epilogue (1024^3)");

    const double plain = run_dense(1024, 1024, 1024, DataType::f32, false, false,
                                   Epilogue{}, false, "f32 plain");

    Epilogue relu;
    relu.type = EpilogueActivateType::Relu;
    const double fused = run_dense(1024, 1024, 1024, DataType::f32, false, false,
                                   relu, true, "f32 bias+relu");
    std::printf("  bias+relu costs %.1f%% of the plain product\n",
                100.0 * (plain - fused) / plain);

    const double plain16 = run_dense(1024, 1024, 1024, DataType::f16, false, false,
                                     Epilogue{}, false, "f16 plain");
    const double fused16 = run_dense(1024, 1024, 1024, DataType::f16, false, false,
                                     relu, true, "f16 bias+relu");
    std::printf("  bias+relu costs %.1f%% of the plain product\n",
                100.0 * (plain16 - fused16) / plain16);
}

#endif  // NNOPS_HAS_CUDA
