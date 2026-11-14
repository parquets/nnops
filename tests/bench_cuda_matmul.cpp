/// @file bench_cuda_matmul.cpp
/// @brief CUDA MatMul throughput — the CuTe GEMM across sizes, dtypes and layouts.
///
/// One row per geometry, reporting GPU milliseconds per call and GFLOPS over
/// 2·M·N·K. The kernel behind it is one CTA tile per dtype, 128 threads and a
/// cp.async pipeline: 64x64x64 3-stage with an m16n8k16 tensor-core MMA on f16,
/// and on f32 a 16-wide-k FMA tile that goes 64x64x16 3-stage at small grids and
/// 64x128x16 2-stage once the grid can refill the machine. The tables therefore
/// answer two questions: how close f32 gets to the FMA peak, and how much the
/// tensor-core path buys over it at the same shape.
///
/// Sections:
///  - f32 and f16 square-and-oblong sweeps, from one wave (128³) to a graph
///    that fills the GPU (4096³). The oblong rows are the LLM shapes: a long
///    K against a small M is the decode GEMM, a large M against equal N and K
///    is the prefill one.
///  - the same ladder against cuBLAS, which is the only external reference this
///    backend has: a ratio under 1.0 is ours losing, and it prices what is left
///    of the distance to a tuned library rather than to the roofline.
///  - the four transpose combinations on a size that is large enough for the
///    copy atom to matter and small enough to run red-hot.
///  - batched GEMMs, where the batch axis multiplies the grid and each matrix
///    is small: the row shows how much the per-matrix tail costs when there are
///    not enough CTAs to fill 38 SMs from one matrix alone.
///  - the epilogue, as the delta between a plain product and bias+Relu: the
///    accumulator never leaves registers, so what a row shows is the extra
///    registers and the activation's arithmetic, not memory traffic.
///
/// Reading the numbers: an RTX 3060 Ti (38 SMs, 1.67 GHz) nominally peaks at
/// 16.2 TFLOPS f32, and a pure-FMA loop sustains 15.3 of it. cuBLAS tops out
/// around 10.8 TFLOPS at 4096³. Timing is CUDA-event based with a warm-up,
/// median of 11 samples.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/matmul.hpp"
#include "common/bench_harness.hpp"
#include "common/cuda_test_helper.hpp"
#include "common/cuda_bench_helper.hpp"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef NNOPS_HAS_CUDA

#include <cublas_v2.h>

#define NNOPS_CUBLAS_CHECK(call)                                              \
    do {                                                                      \
        const cublasStatus_t st_ = (call);                                    \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                   \
            throw std::runtime_error("cublas: " + std::to_string(int(st_)));  \
        }                                                                     \
    } while (0)

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

/// The same row-major product through cuBLAS. cuBLAS is column-major, so it is
/// handed the transposes — C^T = B^T·A^T — which costs nothing because the
/// leading dimensions come out as the operand's own row length (n for B, k for
/// A, n for C) and only the argument order flips.
void cublas_dense(cublasHandle_t h, int64_t m, int64_t n, int64_t k, DataType dt,
                  const void* a, const void* b, void* c)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int ni = int(n), mi = int(m), ki = int(k);
    if (dt == DataType::f16) {
        NNOPS_CUBLAS_CHECK(cublasGemmEx(
            h, CUBLAS_OP_N, CUBLAS_OP_N, ni, mi, ki, &alpha, b, CUDA_R_16F, ni, a,
            CUDA_R_16F, ki, &beta, c, CUDA_R_16F, ni, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        NNOPS_CUBLAS_CHECK(cublasSgemm(
            h, CUBLAS_OP_N, CUBLAS_OP_N, ni, mi, ki, &alpha,
            static_cast<const float*>(b), ni, static_cast<const float*>(a), ki,
            &beta, static_cast<float*>(c), ni));
    }
}

/// The transposed-B twin: `C = A·Bᵀ` with B stored `[n][k]`. cuBLAS reaches it
/// as the column-major product `Cᵀ = B·Aᵀ`, which needs no copy — the operand
/// row lengths and `n` for C are the leading dimensions either way.
void cublas_dense_nt(cublasHandle_t h, int64_t m, int64_t n, int64_t k, DataType dt,
                     const void* a, const void* b, void* c)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int ni = int(n), mi = int(m), ki = int(k);
    if (dt == DataType::f16) {
        NNOPS_CUBLAS_CHECK(cublasGemmEx(
            h, CUBLAS_OP_T, CUBLAS_OP_N, ni, mi, ki, &alpha, b, CUDA_R_16F, ki, a,
            CUDA_R_16F, ki, &beta, c, CUDA_R_16F, ni, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        NNOPS_CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, ni, mi, ki,
                                       &alpha, static_cast<const float*>(b), ki,
                                       static_cast<const float*>(a), ki, &beta,
                                       static_cast<float*>(c), ni));
    }
}

/// Both arms of one geometry, prepared once and then run alternately: a clock
/// excursion moves the two together, and the ratio survives it. Running one arm
/// to completion and then the other does not — the same kernel measured 10-13%
/// apart across two processes, and back-to-back arms drift by as much.
struct DenseArms {
    HostTensor ha, hb, hc;      ///< nnops operands (host side, uploads once)
    DeviceTensor da, db, dc;
    std::unique_ptr<MatMul> op;
    TensorView out;
    std::vector<TensorView> ins;

    HostTensor ga, gb, gc;      ///< the same geometry for cuBLAS
    DeviceTensor gda, gdb, gdc;
    cublasHandle_t handle = nullptr;

    int64_t m = 0, n = 0, k = 0;
    DataType dt = DataType::f32;

    ~DenseArms()
    {
        if (handle != nullptr) { cublasDestroy(handle); }
    }
};

/// One launch, bracketed by a pair of events. The caller reuses the events so
/// the loop does not pay a create/destroy a sample.
template <class F>
double timed_launch(F&& launch, cudaEvent_t begin, cudaEvent_t end)
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

void bench_vs_cublas(int64_t m, int64_t n, int64_t k, DataType dt, bool tb = false)
{
    DenseArms a;
    a.m = m;
    a.n = n;
    a.k = k;
    a.dt = dt;
    const char* tag = dt == DataType::f32 ? "f32 nnops" : "f16 nnops";
    const char* ctag = dt == DataType::f32 ? "f32 cublas" : "f16 cublas";

    const std::vector<int64_t> b_shape = tb ? std::vector<int64_t>{n, k}
                                            : std::vector<int64_t>{k, n};
    a.ha = make_host_random(std::vector<int64_t>{m, k}, dt, false, 17, -0.5f, 0.5f);
    a.hb = make_host_random(b_shape, dt, false, 29, -0.5f, 0.5f);
    a.hc = make_host_zeroed(std::vector<int64_t>{m, n}, dt, false);
    a.da = to_device(a.ha);
    a.db = to_device(a.hb);
    a.dc = to_device(a.hc);
    MatMulAttributes attrs;
    attrs.transpose_b = tb;
    a.op = MatMul::create(attrs, Backend::CUDA);
    a.ins = {a.da.view, a.db.view};
    a.out = a.dc.view;

    a.ga = make_host_random(std::vector<int64_t>{m, k}, dt, false, 17, -0.5f, 0.5f);
    a.gb = make_host_random(b_shape, dt, false, 29, -0.5f, 0.5f);
    a.gc = make_host_zeroed(std::vector<int64_t>{m, n}, dt, false);
    a.gda = to_device(a.ga);
    a.gdb = to_device(a.gb);
    a.gdc = to_device(a.gc);
    NNOPS_CUBLAS_CHECK(cublasCreate(&a.handle));

    auto ours = [&] { cuda_launch(*a.op, a.out, a.ins); };
    auto theirs = [&] {
        if (tb) {
            cublas_dense_nt(a.handle, m, n, k, dt, a.gda.buf.data(),
                            a.gdb.buf.data(), a.gdc.buf.data());
        } else {
            cublas_dense(a.handle, m, n, k, dt, a.gda.buf.data(), a.gdb.buf.data(),
                         a.gdc.buf.data());
        }
    };

    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    NNOPS_CUDA_CHECK(cudaEventCreate(&begin));
    NNOPS_CUDA_CHECK(cudaEventCreate(&end));

    CudaBuffer::sync();
    ours();
    theirs();
    CudaBuffer::sync();

    // Twelve samples each, alternating which arm leads, so neither side always
    // runs into the same cache and clock state.
    std::vector<double> ms_ours;
    std::vector<double> ms_theirs;
    for (int i = 0; i < 12; ++i) {
        if (i % 2 == 0) {
            ms_ours.push_back(timed_launch(ours, begin, end));
            ms_theirs.push_back(timed_launch(theirs, begin, end));
        } else {
            ms_theirs.push_back(timed_launch(theirs, begin, end));
            ms_ours.push_back(timed_launch(ours, begin, end));
        }
    }
    NNOPS_CUDA_CHECK(cudaEventDestroy(begin));
    NNOPS_CUDA_CHECK(cudaEventDestroy(end));

    const double om = median_of(ms_ours);
    const double tm = median_of(ms_theirs);
    const double flops = 2.0 * double(m) * double(n) * double(k);
    const std::string geo = matmul_geometry(m, n, k, false, tb);
    print_cuda_row(tag, geo, om, cuda_gflops(om, flops));
    print_cuda_row(ctag, geo, tm, cuda_gflops(tm, flops));
    std::printf("  %-18s %-34s %9.4f x   nnops / cublas\n", "ratio", geo.c_str(),
                om / tm);
}

/// cuBLAS on freshly allocated bytes — the reference arm of `run_dense`. Same
/// timing helper, same warm-up, so the two rows are directly comparable.
double run_cublas(int64_t m, int64_t n, int64_t k, DataType dt, const char* tag)
{
    HostTensor ha = make_host_random(std::vector<int64_t>{m, k}, dt, false, 17, -0.5f, 0.5f);
    HostTensor hb = make_host_random(std::vector<int64_t>{k, n}, dt, false, 29, -0.5f, 0.5f);
    HostTensor hc = make_host_zeroed(std::vector<int64_t>{m, n}, dt, false);

    const DeviceTensor da = to_device(ha);
    const DeviceTensor db = to_device(hb);
    const DeviceTensor dc = to_device(hc);

    cublasHandle_t handle = nullptr;
    NNOPS_CUBLAS_CHECK(cublasCreate(&handle));
    auto launch = [&] {
        cublas_dense(handle, m, n, k, dt, da.buf.data(), db.buf.data(), dc.buf.data());
    };
    const double ms = cuda_median_ms(launch);
    NNOPS_CUBLAS_CHECK(cublasDestroy(handle));

    const double flops = 2.0 * double(m) * double(n) * double(k);
    print_cuda_row(tag, matmul_geometry(m, n, k, false, false), ms,
                   cuda_gflops(ms, flops));
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
    const int64_t sizes[] = {128, 256, 512, 1024, 2048, 4096};
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

NNOPS_BENCH(cuda_matmul_cublas) {
    print_cuda_header("CUDA MatMul vs cuBLAS");
    // The squares, then the two oblongs a graph actually asks for: a prefill
    // product (a few hundred rows against a deep k) and the decode one.
    const int64_t shapes[][3] = {{512, 512, 512},   {1024, 1024, 1024},
                                 {2048, 2048, 2048}, {4096, 4096, 4096},
                                 {512, 4096, 4096}, {16, 4096, 4096}};
    for (const auto& s : shapes) {
        for (DataType dt : {DataType::f32, DataType::f16}) {
            bench_vs_cublas(s[0], s[1], s[2], dt);
        }
    }
}

/// The decode product alone, so `NNOPS_CUDA_SPLITK` can sweep its narrow-m
/// split count without the squares in the same run also seeing the knob. The
/// `tb` rows are the same products as `Linear` asks for them, with the weight
/// stored `[n][k]`.
NNOPS_BENCH(cuda_matmul_decode) {
    print_cuda_header("CUDA MatMul decode shapes vs cuBLAS");
    for (DataType dt : {DataType::f32, DataType::f16}) {
        bench_vs_cublas(16, 4096, 4096, dt);
        bench_vs_cublas(16, 8192, 2048, dt);
        bench_vs_cublas(16, 2048, 8192, dt);
        bench_vs_cublas(16, 4096, 4096, dt, true);
        bench_vs_cublas(16, 8192, 2048, dt, true);
        bench_vs_cublas(16, 2048, 8192, dt, true);
        bench_vs_cublas(1, 4096, 4096, dt, true);
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
