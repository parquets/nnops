/// @file bench_cuda_matmul.cpp
/// @brief CUDA MatMul throughput — the hand-written PTX GEMM across sizes,
///        dtypes and layouts.
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
#include "detail/bench_knobs.hpp"
#include "backend/cuda/matmul/matmul_f16_host.hpp"
#include "backend/cuda/matmul/matmul_f32_host.hpp"

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

/// The int8 twins. Same column-major transpose trick as above; the only int8
/// specificity is that the scalars are int32 and the compute type is 32I, which
/// is cuBLAS's name for the same IMMA instructions the hand-written tile emits.
/// cuBLAS requires the leading dimensions of an 8-bit operand to be multiples of
/// four, which the square sizes here satisfy.
void cublas_i8_nn(cublasHandle_t h, int64_t m, int64_t n, int64_t k,
                  const void* a, const void* b, void* c)
{
    const int32_t alpha = 1;
    const int32_t beta = 0;
    const int ni = int(n), mi = int(m), ki = int(k);
    NNOPS_CUBLAS_CHECK(cublasGemmEx(
        h, CUBLAS_OP_N, CUBLAS_OP_N, ni, mi, ki, &alpha, b, CUDA_R_8I, ni, a,
        CUDA_R_8I, ki, &beta, c, CUDA_R_32I, ni, CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void cublas_i8_nt(cublasHandle_t h, int64_t m, int64_t n, int64_t k,
                  const void* a, const void* b, void* c)
{
    const int32_t alpha = 1;
    const int32_t beta = 0;
    const int ni = int(n), mi = int(m), ki = int(k);
    NNOPS_CUBLAS_CHECK(cublasGemmEx(
        h, CUBLAS_OP_T, CUBLAS_OP_N, ni, mi, ki, &alpha, b, CUDA_R_8I, ki, a,
        CUDA_R_8I, ki, &beta, c, CUDA_R_32I, ni, CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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

void bench_vs_cublas(int64_t m, int64_t n, int64_t k, DataType dt, bool tb = false,
                     const char* tag_override = nullptr)
{
    DenseArms a;
    a.m = m;
    a.n = n;
    a.k = k;
    a.dt = dt;
    const char* tag = dt == DataType::f32 ? "f32 nnops" : "f16 nnops";
    if (tag_override != nullptr) { tag = tag_override; }
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

/// The hand-written PTX kernel against cuBLAS, every 128 along the diagonal,
/// both B layouts.
NNOPS_BENCH(cuda_matmul_ptx_cublas) {
    print_cuda_header("CUDA MatMul f16: hand-written PTX vs cuBLAS, 128..4096");
    for (int64_t side = 128; side <= 4096; side += 128) {
        bench_vs_cublas(side, side, side, DataType::f16, true);
    }
    for (int64_t side = 128; side <= 4096; side += 512) {
        bench_vs_cublas(side, side, side, DataType::f16, false);
    }
}

/// The decode product alone, so `detail::cuda_splitk` can sweep its narrow-m
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

/// u8 × s8 → s32: the MatMulInteger shape. The QuantParams are per-tensor, so
/// the measured overhead is the two reduction sweeps plus the compensation.
double run_i8(int64_t m, int64_t n, int64_t k, const char* tag,
              bool tb = false)
{
    MatMulAttributes attrs;
    attrs.output_dtype = DataType::s32;
    attrs.transpose_b = tb;

    test::XorShift128 rng(17);
    std::vector<uint8_t> a(static_cast<size_t>(m * k));
    for (auto& e : a) { e = static_cast<uint8_t>(rng.next_u64() & 0xFFu); }
    std::vector<int8_t> b(static_cast<size_t>(k * n));
    for (auto& e : b) {
        e = static_cast<int8_t>(int32_t(rng.next_u64() % 255) - 127);
    }

    QuantParams qa;
    qa.scale = 0.01f;
    qa.zero_point = 128;
    QuantParams qb;
    qb.scale = 0.004f;
    qb.zero_point = 3;

    CudaBuffer da(a.size()), db(b.size());
    CudaBuffer dc(static_cast<size_t>(m * n) * sizeof(int32_t));
    da.upload(a.data(), a.size());
    db.upload(b.data(), b.size());
    dc.zero();

    const int64_t as[] = {m, k};
    const int64_t bs[] = {tb ? n : k, tb ? k : n};
    const int64_t cs[] = {m, n};
    TensorView av(std::span<const int64_t>(as, 2), DataType::u8, da.data(),
                  TensorLayout::Dense, qa);
    TensorView bv(std::span<const int64_t>(bs, 2), DataType::s8, db.data(),
                  TensorLayout::Dense, qb);
    TensorView cv(std::span<const int64_t>(cs, 2), DataType::s32, dc.data());

    auto op = MatMul::create(attrs, Backend::CUDA);
    const TensorView ins[] = {av, bv};
    const double ms = cuda_median_ms([&] { cuda_launch(*op, cv, ins); });

    const double flops = 2.0 * double(m) * double(n) * double(k);
    print_cuda_row(tag, matmul_geometry(m, n, k, false, tb), ms,
                   cuda_gflops(ms, flops));
    return cuda_gflops(ms, flops);
}

/// The hand-written PTX f16 GEMM (64x32 per warp) through the operator, against
/// cuBLAS, on the NN pose.
NNOPS_BENCH(cuda_matmul_f16_nn) {
    print_cuda_header("CUDA MatMul f16 NN: hand-written PTX vs cuBLAS");
    for (int64_t side : {256, 512, 1024, 2048, 4096}) {
        MatMulAttributes attrs;   // NN: B is [K, N]
        HostTensor ha = make_host_random(std::vector<int64_t>{side, side},
                                         DataType::f16, false, 17, -0.5f, 0.5f);
        HostTensor hb = make_host_random(std::vector<int64_t>{side, side},
                                         DataType::f16, false, 29, -0.5f, 0.5f);
        HostTensor hc = make_host_zeroed(std::vector<int64_t>{side, side},
                                         DataType::f16, false);
        DeviceTensor da = to_device(ha);
        DeviceTensor db = to_device(hb);
        DeviceTensor dc_man = to_device(hc);

        // The oracle is the CPU route, so the kernel is checked against the
        // operator contract rather than against another device engine.
        auto op_cpu = MatMul::create(attrs, Backend::CPU);
        const std::vector<TensorDesc> descs = {ha.view.desc(), hb.view.desc()};
        const TensorDesc od = op_cpu->getOutputTensorDesc(descs).front();
        HostTensor href = make_host_zeroed(od.dims, od.dtype, false);
        const TensorView cpu_ins[] = {ha.view, hb.view};
        cpu_run(*op_cpu, href.view, cpu_ins);

        auto op = MatMul::create(attrs, Backend::CUDA);
        const TensorView ins[] = {da.view, db.view};
        TensorView o_ptx = dc_man.view;
        const double ms_man = cuda_median_ms([&] { cuda_launch(*op, o_ptx, ins); });

        from_device(dc_man, hc);
        expect_same(hc, href, kF16AccumTol, 2e-2f, "cuda_matmul_f16_nn");

        DeviceTensor gda = to_device(ha);
        DeviceTensor gdb = to_device(hb);
        DeviceTensor gdc = to_device(hc);
        cublasHandle_t handle = nullptr;
        NNOPS_CUBLAS_CHECK(cublasCreate(&handle));
        const double ms_cub = cuda_median_ms([&] {
            cublas_dense(handle, side, side, side, DataType::f16, gda.buf.data(),
                         gdb.buf.data(), gdc.buf.data());
        });
        cublasDestroy(handle);

        const double flops = 2.0 * double(side) * double(side) * double(side);
        print_cuda_row("ptx", matmul_geometry(side, side, side, false, false),
                       ms_man, cuda_gflops(ms_man, flops));
        print_cuda_row("cublas", matmul_geometry(side, side, side, false, false),
                       ms_cub, cuda_gflops(ms_cub, flops));
    }
}

NNOPS_BENCH(cuda_matmul_f16_tn) {
    print_cuda_header("CUDA MatMul f16 TN: hand-written PTX vs cuBLAS");
    for (int64_t side : {256, 512, 1024, 2048, 4096}) {
        MatMulAttributes attrs;
        attrs.transpose_b = true;
        HostTensor ha = make_host_random(std::vector<int64_t>{side, side},
                                         DataType::f16, false, 17, -0.5f, 0.5f);
        HostTensor hb = make_host_random(std::vector<int64_t>{side, side},
                                         DataType::f16, false, 29, -0.5f, 0.5f);
        HostTensor hc = make_host_zeroed(std::vector<int64_t>{side, side},
                                         DataType::f16, false);
        DeviceTensor da = to_device(ha);
        DeviceTensor db = to_device(hb);
        DeviceTensor dc_man = to_device(hc);

        auto op_cpu = MatMul::create(attrs, Backend::CPU);
        const std::vector<TensorDesc> descs = {ha.view.desc(), hb.view.desc()};
        const TensorDesc od = op_cpu->getOutputTensorDesc(descs).front();
        HostTensor href = make_host_zeroed(od.dims, od.dtype, false);
        const TensorView cpu_ins[] = {ha.view, hb.view};
        cpu_run(*op_cpu, href.view, cpu_ins);

        auto op = MatMul::create(attrs, Backend::CUDA);
        const TensorView ins[] = {da.view, db.view};
        TensorView o_ptx = dc_man.view;
        const double ms_man =
            cuda_median_ms([&] { cuda_launch(*op, o_ptx, ins); });

        from_device(dc_man, hc);
        expect_same(hc, href, kF16AccumTol, 2e-2f, "cuda_matmul_f16_tn");

        // cuBLAS on its own buffers, in the same warm state, so the two rows of
        // a size are one measurement rather than two runs.
        DeviceTensor gda = to_device(ha);
        DeviceTensor gdb = to_device(hb);
        DeviceTensor gdc = to_device(hc);
        cublasHandle_t handle = nullptr;
        NNOPS_CUBLAS_CHECK(cublasCreate(&handle));
        const double ms_cub = cuda_median_ms([&] {
            cublas_dense_nt(handle, side, side, side, DataType::f16,
                            gda.buf.data(), gdb.buf.data(), gdc.buf.data());
        });
        cublasDestroy(handle);

        const double flops = 2.0 * double(side) * double(side) * double(side);
        print_cuda_row("ptx", matmul_geometry(side, side, side, false, true),
                       ms_man, cuda_gflops(ms_man, flops));
        print_cuda_row("cublas", matmul_geometry(side, side, side, false, true),
                       ms_cub, cuda_gflops(ms_cub, flops));
    }
}

/// Fixed 1024x1024 output, k grown: if the efficiency climbs with k at a
/// constant block count, the shallow sizes are paying a per-CTA startup cost
/// rather than missing parallelism.
NNOPS_BENCH(cuda_matmul_k_ramp) {
    print_cuda_header("CUDA MatMul f32 1024x1024xK: k ramp at a fixed grid");
    for (int64_t k : {1024, 2048, 4096, 8192}) {
        nnops::backend::cuda::F32Call call_ours;
        HostTensor ha = make_host_random(std::vector<int64_t>{1024, k},
                                         DataType::f32, false, 17, -0.5f, 0.5f);
        HostTensor hb = make_host_random(std::vector<int64_t>{1024, k},
                                         DataType::f32, false, 29, -0.5f, 0.5f);
        HostTensor hc = make_host_zeroed(std::vector<int64_t>{1024, 1024},
                                         DataType::f32, false);
        DeviceTensor da = to_device(ha);
        DeviceTensor db = to_device(hb);
        DeviceTensor dc = to_device(hc);
        call_ours.a = da.view.ptr<void>();
        call_ours.b = db.view.ptr<void>();
        call_ours.c = dc.view.ptr<void>();
        call_ours.m = 1024;
        call_ours.n = 1024;
        call_ours.k = int(k);
        call_ours.lda = k;
        call_ours.ldb = k;
        call_ours.ldc = 1024;
        call_ours.trans_b = true;
        call_ours.tile = 0;
        const double ms = cuda_median_ms([&] {
            nnops::backend::cuda::f32_dense_call(call_ours, nullptr);
        });

        cublasHandle_t handle = nullptr;
        NNOPS_CUBLAS_CHECK(cublasCreate(&handle));
        HostTensor ga = make_host_random(std::vector<int64_t>{1024, k},
                                         DataType::f32, false, 17, -0.5f, 0.5f);
        HostTensor gb = make_host_random(std::vector<int64_t>{1024, k},
                                         DataType::f32, false, 29, -0.5f, 0.5f);
        HostTensor gc = make_host_zeroed(std::vector<int64_t>{1024, 1024},
                                         DataType::f32, false);
        DeviceTensor gda = to_device(ga);
        DeviceTensor gdb = to_device(gb);
        DeviceTensor gdc = to_device(gc);
        const double ms_cub = cuda_median_ms([&] {
            cublas_dense_nt(handle, 1024, 1024, k, DataType::f32, gda.buf.data(),
                            gdb.buf.data(), gdc.buf.data());
        });
        cublasDestroy(handle);

        const double flops = 2.0 * 1024.0 * 1024.0 * double(k);
        print_cuda_row("simt", matmul_geometry(1024, 1024, k, false, true), ms,
                       cuda_gflops(ms, flops));
        print_cuda_row("cublas", matmul_geometry(1024, 1024, k, false, true), ms_cub,
                       cuda_gflops(ms_cub, flops));
    }
}

/// Tile times split-K at 1024^3, which is where cuBLAS runs a 128x64 tile with
/// gridDim.z = 2 (`ampere_sgemm_128x64_tn`): the question is whether the same
/// shape plus a cut beats our 128x128 tile with none.
NNOPS_BENCH(cuda_matmul_shallow_f) {
    print_cuda_header("CUDA MatMul f32 1024^3 TN: tile x split-K");
    auto& knob = nnops::detail::cuda_splitk;
    for (int64_t side : {1024, 1536}) {
        MatMulAttributes attrs;
        attrs.transpose_b = true;
        HostTensor ha = make_host_random(std::vector<int64_t>{side, side},
                                         DataType::f32, false, 17, -0.5f, 0.5f);
        HostTensor hb = make_host_random(std::vector<int64_t>{side, side},
                                         DataType::f32, false, 29, -0.5f, 0.5f);
        HostTensor hc = make_host_zeroed(std::vector<int64_t>{side, side},
                                         DataType::f32, false);
        DeviceTensor da = to_device(ha);
        DeviceTensor db = to_device(hb);
        DeviceTensor dc = to_device(hc);
        const double flops = 2.0 * double(side) * double(side) * double(side);
        for (int tile = 0; tile < 4; ++tile) {
            for (int splits : {0, 2, 4}) {
                knob.store(splits);
                nnops::backend::cuda::F32Call call;
                call.a = da.view.ptr<void>();
                call.b = db.view.ptr<void>();
                call.c = dc.view.ptr<void>();
                call.m = int(side);
                call.n = int(side);
                call.k = int(side);
                call.lda = side;
                call.ldb = side;
                call.ldc = side;
                call.trans_b = true;
                call.tile = tile;
                const double ms = cuda_median_ms([&] {
                    nnops::backend::cuda::f32_dense_call(call, nullptr);
                });
                char tag[48];
                std::snprintf(tag, sizeof(tag), "tile%d/split%d", tile, splits);
                print_cuda_row(tag, matmul_geometry(side, side, side, false, true),
                               ms, cuda_gflops(ms, flops));
            }
        }
        knob.store(-1);
    }
}

/// The two n-major poses at one size, one launch each, so a profiler can put
/// the A-arrival-order pair side by side: NN reads A `[m][k]`, NT reads it
/// `[k][m]`, and that difference is the whole 10% between them.
NNOPS_BENCH(cuda_matmul_one_nn_nt) {
    print_cuda_header("CUDA MatMul f32 2048^3 NN then NT (single launches)");
    for (int pose = 0; pose < 2; ++pose) {
        MatMulAttributes attrs;
        attrs.transpose_a = pose == 1;
        HostTensor ha = make_host_random(std::vector<int64_t>{2048, 2048},
                                         DataType::f32, false, 17, -0.5f, 0.5f);
        HostTensor hb = make_host_random(std::vector<int64_t>{2048, 2048},
                                         DataType::f32, false, 29, -0.5f, 0.5f);
        HostTensor hc = make_host_zeroed(std::vector<int64_t>{2048, 2048},
                                         DataType::f32, false);
        DeviceTensor da = to_device(ha);
        DeviceTensor db = to_device(hb);
        DeviceTensor dc = to_device(hc);
        auto op = MatMul::create(attrs, Backend::CUDA);
        const TensorView ins[] = {da.view, db.view};
        TensorView out = dc.view;
        cuda_launch(*op, out, ins);
        CudaBuffer::sync();
    }
}

/// 4096^3 f32 NN, one launch: the hand-written SIMT kernel, so a profiler lands
/// on it rather than on a ladder of launches.
NNOPS_BENCH(cuda_matmul_profile_nn) {
    print_cuda_header("CUDA MatMul f32 4096^3 NN (single launch)");
    MatMulAttributes attrs;   // NN: B is [K, N]
    HostTensor ha = make_host_random(std::vector<int64_t>{4096, 4096},
                                     DataType::f32, false, 17, -0.5f, 0.5f);
    HostTensor hb = make_host_random(std::vector<int64_t>{4096, 4096},
                                     DataType::f32, false, 29, -0.5f, 0.5f);
    HostTensor hc = make_host_zeroed(std::vector<int64_t>{4096, 4096},
                                     DataType::f32, false);
    DeviceTensor da = to_device(ha);
    DeviceTensor db = to_device(hb);
    DeviceTensor dc = to_device(hc);
    auto op = MatMul::create(attrs, Backend::CUDA);
    const TensorView ins[] = {da.view, db.view};
    TensorView out = dc.view;
    cuda_launch(*op, out, ins);
    CudaBuffer::sync();
}

/// One 1024^3 f32 launch each -- ours and cuBLAS -- so a profiler lands on a
/// single kernel per run and the two launch configurations can be compared
/// without guessing which of a ladder's launches it caught.
NNOPS_BENCH(cuda_matmul_one_f) {
    print_cuda_header("CUDA MatMul f32 1024^3 (single launch, ours)");
    MatMulAttributes attrs;
    attrs.transpose_b = true;
    HostTensor ha = make_host_random(std::vector<int64_t>{1024, 1024},
                                     DataType::f32, false, 17, -0.5f, 0.5f);
    HostTensor hb = make_host_random(std::vector<int64_t>{1024, 1024},
                                     DataType::f32, false, 29, -0.5f, 0.5f);
    HostTensor hc = make_host_zeroed(std::vector<int64_t>{1024, 1024},
                                     DataType::f32, false);
    DeviceTensor da = to_device(ha);
    DeviceTensor db = to_device(hb);
    DeviceTensor dc = to_device(hc);
    auto op = MatMul::create(attrs, Backend::CUDA);
    const TensorView ins[] = {da.view, db.view};
    TensorView out = dc.view;
    cuda_launch(*op, out, ins);
    CudaBuffer::sync();
}

NNOPS_BENCH(cuda_matmul_one_f_cublas) {
    print_cuda_header("CUDA MatMul f32 1024^3 (single launch, cuBLAS)");
    HostTensor ha = make_host_random(std::vector<int64_t>{1024, 1024},
                                     DataType::f32, false, 17, -0.5f, 0.5f);
    HostTensor hb = make_host_random(std::vector<int64_t>{1024, 1024},
                                     DataType::f32, false, 29, -0.5f, 0.5f);
    HostTensor hc = make_host_zeroed(std::vector<int64_t>{1024, 1024},
                                     DataType::f32, false);
    DeviceTensor da = to_device(ha);
    DeviceTensor db = to_device(hb);
    DeviceTensor dc = to_device(hc);
    cublasHandle_t handle = nullptr;
    NNOPS_CUBLAS_CHECK(cublasCreate(&handle));
    cublas_dense_nt(handle, 1024, 1024, 1024, DataType::f32, da.buf.data(),
                    db.buf.data(), dc.buf.data());
    CudaBuffer::sync();
    cublasDestroy(handle);
}

/// The f32 GEMM against cuBLAS on the interleaved harness -- twelve samples with
/// the arms alternating, which is the only comparison this host's +-10% drift
/// leaves trustworthy.
NNOPS_BENCH(cuda_matmul_interleaved_f) {
    print_cuda_header("CUDA MatMul f32 TN: interleaved vs cuBLAS");
    // Both B layouts, since the k-major and n-major poses take different tiles.
    for (bool tb : {true, false}) {
        for (int64_t side : {2048, 4096}) {
            bench_vs_cublas(side, side, side, DataType::f32, tb,
                            tb ? "f32 simt TN" : "f32 simt NN");
        }
    }
}

/// The f32 split-K cut, swept by the knob: which shapes want the reduction cut
/// and how far. Collapsing the tail of a shallow grid is the whole question.
NNOPS_BENCH(cuda_matmul_splitk_f) {
    print_cuda_header("CUDA MatMul f32 TN: split-K sweep (0 = one pass)");
    auto& knob = nnops::detail::cuda_splitk;
    for (int64_t side : {512, 1024, 2048}) {
        MatMulAttributes attrs;
        attrs.transpose_b = true;
        HostTensor ha = make_host_random(std::vector<int64_t>{side, side},
                                         DataType::f32, false, 17, -0.5f, 0.5f);
        HostTensor hb = make_host_random(std::vector<int64_t>{side, side},
                                         DataType::f32, false, 29, -0.5f, 0.5f);
        HostTensor hc = make_host_zeroed(std::vector<int64_t>{side, side},
                                         DataType::f32, false);
        DeviceTensor da = to_device(ha);
        DeviceTensor db = to_device(hb);
        DeviceTensor dc = to_device(hc);
        auto op = MatMul::create(attrs, Backend::CUDA);
        const TensorView ins[] = {da.view, db.view};
        const double flops = 2.0 * double(side) * double(side) * double(side);
        for (int splits : {0, 2, 4, 8}) {
            knob.store(splits);
            TensorView out = dc.view;
            const double ms = cuda_median_ms([&] { cuda_launch(*op, out, ins); });
            char tag[32];
            std::snprintf(tag, sizeof(tag), "split%d", splits);
            print_cuda_row(tag, matmul_geometry(side, side, side, false, true), ms,
                           cuda_gflops(ms, flops));
        }
        knob.store(-1);
    }
}

/// The f32 tile menu, at the same geometry, so the register block and the ring
/// depth can be compared without rebuilding.
NNOPS_BENCH(cuda_matmul_simt_tiles) {
    print_cuda_header("CUDA MatMul f32 TN: hand-written SIMT tile sweep");
    static const char* const kNames[] = {"8x8/2x4 8warp", "8x8/2x2 4warp",
                                         "8x8/2x2 4warp no-raster",
                                         "8x8/1x4 64x128", "8x8/2x4 s3",
                                         "8x4/2x4 128x64 8warp", "8x4 64x64 4blk",
                                         "8x8 128x64 pad0 4blk",
                                         "8x4 64x64 padded 4blk"};
    for (int64_t side : {1024, 2048, 4096}) {
        MatMulAttributes attrs;
        attrs.transpose_b = true;
        HostTensor ha = make_host_random(std::vector<int64_t>{side, side},
                                         DataType::f32, false, 17, -0.5f, 0.5f);
        HostTensor hb = make_host_random(std::vector<int64_t>{side, side},
                                         DataType::f32, false, 29, -0.5f, 0.5f);
        HostTensor hc = make_host_zeroed(std::vector<int64_t>{side, side},
                                         DataType::f32, false);
        DeviceTensor da = to_device(ha);
        DeviceTensor db = to_device(hb);
        DeviceTensor dc = to_device(hc);
        HostTensor href;
        if (side == 1024) {
            auto op_cpu = MatMul::create(attrs, Backend::CPU);
            const std::vector<TensorDesc> descs = {ha.view.desc(), hb.view.desc()};
            const TensorDesc od = op_cpu->getOutputTensorDesc(descs).front();
            href = make_host_zeroed(od.dims, od.dtype, false);
            const TensorView cpu_ins[] = {ha.view, hb.view};
            cpu_run(*op_cpu, href.view, cpu_ins);
        }

        const double flops = 2.0 * double(side) * double(side) * double(side);
        for (int t = 0; t < 9; ++t) {
            nnops::backend::cuda::F32Call call;
            call.a = da.view.ptr<void>();
            call.b = db.view.ptr<void>();
            call.c = dc.view.ptr<void>();
            call.m = int(side);
            call.n = int(side);
            call.k = int(side);
            call.lda = side;
            call.ldb = side;
            call.ldc = side;
            call.trans_b = true;
            call.tile = t;
            const double ms = cuda_median_ms([&] {
                nnops::backend::cuda::f32_dense_call(call, nullptr);
            });
            if (side == 1024) {
                from_device(dc, hc);
                char what[96];
                std::snprintf(what, sizeof(what), "cuda_matmul_simt_tiles %s @%lld",
                              kNames[t], static_cast<long long>(side));
                expect_same(hc, href, 1e-4f, 1e-3f, what);
            }
            print_cuda_row(kNames[t], matmul_geometry(side, side, side, false, true),
                           ms, cuda_gflops(ms, flops));
        }
    }

    // The same menu on the n-major pose, in the same warm state: its tile
    // ranking is not the k-major one's.
    for (int64_t side : {1024, 2048, 4096}) {
        HostTensor ha = make_host_random(std::vector<int64_t>{side, side},
                                         DataType::f32, false, 17, -0.5f, 0.5f);
        HostTensor hb = make_host_random(std::vector<int64_t>{side, side},
                                         DataType::f32, false, 29, -0.5f, 0.5f);
        HostTensor hc = make_host_zeroed(std::vector<int64_t>{side, side},
                                         DataType::f32, false);
        DeviceTensor da = to_device(ha);
        DeviceTensor db = to_device(hb);
        DeviceTensor dc = to_device(hc);
        const double flops = 2.0 * double(side) * double(side) * double(side);
        for (int t = 0; t < 9; ++t) {
            nnops::backend::cuda::F32Call call;
            call.a = da.view.ptr<void>();
            call.b = db.view.ptr<void>();
            call.c = dc.view.ptr<void>();
            call.m = int(side);
            call.n = int(side);
            call.k = int(side);
            call.lda = side;
            call.ldb = side;
            call.ldc = side;
            call.trans_b = false;
            call.tile = t;
            const double ms = cuda_median_ms([&] {
                nnops::backend::cuda::f32_dense_call(call, nullptr);
            });
            print_cuda_row(kNames[t],
                           matmul_geometry(side, side, side, false, false), ms,
                           cuda_gflops(ms, flops));
        }
    }
}

/// The same tile reached two ways at 4096^3 f16 TN: through the operator (which
/// picks it by pose) and through a direct call with the tile named, on one pair
/// of buffers. The sweep's direct call measured 4.41 ms where the operator
/// measures 5.07, and this is what tells the two apart.
NNOPS_BENCH(cuda_matmul_h_path_cmp) {
    print_cuda_header("CUDA MatMul f16 4096^3 TN: operator route vs direct call");
    const int64_t side = 4096;
    MatMulAttributes attrs;
    attrs.transpose_b = true;
    HostTensor ha = make_host_random(std::vector<int64_t>{side, side},
                                     DataType::f16, false, 17, -0.5f, 0.5f);
    HostTensor hb = make_host_random(std::vector<int64_t>{side, side},
                                     DataType::f16, false, 29, -0.5f, 0.5f);
    HostTensor hc = make_host_zeroed(std::vector<int64_t>{side, side},
                                     DataType::f16, false);
    DeviceTensor da = to_device(ha);
    DeviceTensor db = to_device(hb);
    DeviceTensor dc = to_device(hc);
    const double flops = 2.0 * double(side) * double(side) * double(side);

    auto op = MatMul::create(attrs, Backend::CUDA);
    const TensorView ins[] = {da.view, db.view};
    TensorView out = dc.view;
    const double ms_op = cuda_median_ms([&] { cuda_launch(*op, out, ins); });

    nnops::backend::cuda::F16Call call;
    call.a = da.view.ptr<void>();
    call.b = db.view.ptr<void>();
    call.c = dc.view.ptr<void>();
    call.m = int(side);
    call.n = int(side);
    call.k = int(side);
    call.lda = side;
    call.ldb = side;
    call.ldc = side;
    call.trans_b = true;
    // tile 0 again at the end: if the last measurement always wins, the sweep's
    // 15% for the direct epilogue was order, not the epilogue.
    for (int t : {0, 0, 0, 0, 0}) {
        call.tile = t;
        const double ms = cuda_median_ms([&] {
            nnops::backend::cuda::f16_dense_call(call, nullptr);
        });
        char tag[32];
        std::snprintf(tag, sizeof(tag), "direct tile%d", t);
        print_cuda_row(tag, matmul_geometry(side, side, side, false, true), ms,
                       cuda_gflops(ms, flops));
    }
    print_cuda_row("operator", matmul_geometry(side, side, side, false, true), ms_op,
                   cuda_gflops(ms_op, flops));
}

/// f16 against cuBLAS on the interleaved harness, both B layouts.
NNOPS_BENCH(cuda_matmul_interleaved_h) {
    print_cuda_header("CUDA MatMul f16: interleaved vs cuBLAS");
    for (bool tb : {true, false}) {
        for (int64_t side : {1024, 2048, 4096}) {
            bench_vs_cublas(side, side, side, DataType::f16, tb,
                            tb ? "f16 ptx TN" : "f16 ptx NN");
        }
    }
}

/// f32 across all four transpose poses in one warm state: the hand-written SIMT
/// tile and cuBLAS, on its own buffers.
NNOPS_BENCH(cuda_matmul_f32_poses) {
    print_cuda_header("CUDA MatMul f32: hand-written SIMT vs cuBLAS");
    for (int pose = 0; pose < 4; ++pose)
    for (int64_t side : {256, 512, 1024, 2048, 4096}) {
        const bool ta = (pose & 1) != 0;
        const bool tb = (pose & 2) != 0;
        MatMulAttributes attrs;
        attrs.transpose_a = ta;
        attrs.transpose_b = tb;
        HostTensor ha = make_host_random(std::vector<int64_t>{side, side},
                                         DataType::f32, false, 17, -0.5f, 0.5f);
        HostTensor hb = make_host_random(std::vector<int64_t>{side, side},
                                         DataType::f32, false, 29, -0.5f, 0.5f);
        HostTensor hc = make_host_zeroed(std::vector<int64_t>{side, side},
                                         DataType::f32, false);
        DeviceTensor da = to_device(ha);
        DeviceTensor db = to_device(hb);
        DeviceTensor dc_simt = to_device(hc);

        auto op_cpu = MatMul::create(attrs, Backend::CPU);
        const std::vector<TensorDesc> descs = {ha.view.desc(), hb.view.desc()};
        const TensorDesc od = op_cpu->getOutputTensorDesc(descs).front();
        HostTensor href = make_host_zeroed(od.dims, od.dtype, false);
        const TensorView cpu_ins[] = {ha.view, hb.view};
        cpu_run(*op_cpu, href.view, cpu_ins);

        auto op = MatMul::create(attrs, Backend::CUDA);
        const TensorView ins[] = {da.view, db.view};
        TensorView o_simt = dc_simt.view;
        const double ms_simt = cuda_median_ms([&] { cuda_launch(*op, o_simt, ins); });

        from_device(dc_simt, hc);
        expect_same(hc, href, 1e-4f, 1e-3f, "cuda_matmul_f32_poses");

        DeviceTensor gda = to_device(ha);
        DeviceTensor gdb = to_device(hb);
        DeviceTensor gdc = to_device(hc);
        cublasHandle_t handle = nullptr;
        NNOPS_CUBLAS_CHECK(cublasCreate(&handle));
        const double ms_cub = cuda_median_ms([&] {
            if (ta && tb) {
                cublas_dense_nt(handle, side, side, side, DataType::f32,
                                gdb.buf.data(), gda.buf.data(), gdc.buf.data());
            } else if (ta) {
                cublas_dense(handle, side, side, side, DataType::f32,
                             gdb.buf.data(), gda.buf.data(), gdc.buf.data());
            } else if (tb) {
                cublas_dense_nt(handle, side, side, side, DataType::f32,
                                gda.buf.data(), gdb.buf.data(), gdc.buf.data());
            } else {
                cublas_dense(handle, side, side, side, DataType::f32,
                             gda.buf.data(), gdb.buf.data(), gdc.buf.data());
            }
        });
        cublasDestroy(handle);

        const double flops = 2.0 * double(side) * double(side) * double(side);
        print_cuda_row("simt", matmul_geometry(side, side, side, ta, tb),
                       ms_simt, cuda_gflops(ms_simt, flops));
        print_cuda_row("cublas", matmul_geometry(side, side, side, ta, tb),
                       ms_cub, cuda_gflops(ms_cub, flops));
    }
}

/// The tile menu: every compiled `F16TileShape`, at the same geometry, so the
/// per-warp value tile and the cp.async ring can be compared without rebuilding.
/// Correctness rides on the smallest size only; the large ones are timings.
NNOPS_BENCH(cuda_matmul_ptx_tiles) {
    print_cuda_header("CUDA MatMul f16 TN: hand-written PTX tile sweep");
    static const char* const kNames[] = {"64x64/2x2/bk32/s3", "64x32/2x4/bk32/s3",
                                         "64x32/1x4/bk32/s3", "64x64/skew-panel",
                                         "64x64/direct-epi", "32x32/2x2 4blk"};
    for (int64_t side : {256, 1024, 4096}) {
        MatMulAttributes attrs;
        attrs.transpose_b = true;
        HostTensor ha = make_host_random(std::vector<int64_t>{side, side},
                                         DataType::f16, false, 17, -0.5f, 0.5f);
        HostTensor hb = make_host_random(std::vector<int64_t>{side, side},
                                         DataType::f16, false, 29, -0.5f, 0.5f);
        HostTensor hc = make_host_zeroed(std::vector<int64_t>{side, side},
                                         DataType::f16, false);
        DeviceTensor da = to_device(ha);
        DeviceTensor db = to_device(hb);
        DeviceTensor dc = to_device(hc);

        HostTensor href;
        if (side <= 1024) {
            auto op_cpu = MatMul::create(attrs, Backend::CPU);
            const std::vector<TensorDesc> descs = {ha.view.desc(), hb.view.desc()};
            const TensorDesc od = op_cpu->getOutputTensorDesc(descs).front();
            href = make_host_zeroed(od.dims, od.dtype, false);
            const TensorView cpu_ins[] = {ha.view, hb.view};
            cpu_run(*op_cpu, href.view, cpu_ins);
        }

        const double flops = 2.0 * double(side) * double(side) * double(side);
        for (int t = 0; t < 6; ++t) {
            nnops::backend::cuda::F16Call call;
            call.a = da.view.ptr<void>();
            call.b = db.view.ptr<void>();
            call.c = dc.view.ptr<void>();
            call.m = int(side);
            call.n = int(side);
            call.k = int(side);
            call.lda = side;
            call.ldb = side;
            call.ldc = side;
            call.trans_b = true;
            call.tile = t;
            const bool ok = nnops::backend::cuda::f16_dense_call(call, nullptr);
            if (cudaDeviceSynchronize() != cudaSuccess) {
                std::fprintf(stderr, "tile %d @%lld: %s\n", t,
                             static_cast<long long>(side),
                             cudaGetErrorString(cudaGetLastError()));
                std::fflush(stderr);
                continue;
            }
            const double ms = cuda_median_ms([&] {
                nnops::backend::cuda::f16_dense_call(call, nullptr);
            });
            from_device(dc, hc);
            char what[96];
            std::snprintf(what, sizeof(what), "cuda_matmul_ptx_tiles %s @%lld",
                          kNames[t], static_cast<long long>(side));
            std::fprintf(stderr, "tile %d @%lld launch=%d\n", t,
                         static_cast<long long>(side), int(ok));
            std::fflush(stderr);
            if (side <= 1024) { expect_same(hc, href, kF16AccumTol, 2e-2f, what); }
            print_cuda_row(kNames[t], matmul_geometry(side, side, side, false, true),
                           ms, cuda_gflops(ms, flops));
        }
    }

    // The n-major pose takes its own panel layout, so the menu has to be swept
    // there too: the two largest sizes are enough to rank it.
    for (int64_t side : {1024, 4096}) {
        MatMulAttributes attrs;   // NN: B is [K, N]
        HostTensor ha = make_host_random(std::vector<int64_t>{side, side},
                                         DataType::f16, false, 17, -0.5f, 0.5f);
        HostTensor hb = make_host_random(std::vector<int64_t>{side, side},
                                         DataType::f16, false, 29, -0.5f, 0.5f);
        HostTensor hc = make_host_zeroed(std::vector<int64_t>{side, side},
                                         DataType::f16, false);
        DeviceTensor da = to_device(ha);
        DeviceTensor db = to_device(hb);
        DeviceTensor dc = to_device(hc);
        const double flops = 2.0 * double(side) * double(side) * double(side);
        for (int t = 0; t < 6; ++t) {
            nnops::backend::cuda::F16Call call;
            call.a = da.view.ptr<void>();
            call.b = db.view.ptr<void>();
            call.c = dc.view.ptr<void>();
            call.m = int(side);
            call.n = int(side);
            call.k = int(side);
            call.lda = side;
            call.ldb = side;
            call.ldc = side;
            call.trans_b = false;
            call.tile = t;
            const double ms = cuda_median_ms([&] {
                nnops::backend::cuda::f16_dense_call(call, nullptr);
            });
            print_cuda_row(kNames[t], matmul_geometry(side, side, side, false, false),
                           ms, cuda_gflops(ms, flops));
        }
    }
}

/// One 4096^3 f16 / f32 launch each, for a profiler that lands on one kernel.
NNOPS_BENCH(cuda_matmul_one_h) {
    print_cuda_header("CUDA MatMul f16 4096^3 (single launch)");
    run_dense(4096, 4096, 4096, DataType::f16, false, false, Epilogue{}, false,
              "f16");
}

NNOPS_BENCH(cuda_matmul_one_s) {
    print_cuda_header("CUDA MatMul f32 4096^3 (single launch)");
    run_dense(4096, 4096, 4096, DataType::f32, false, false, Epilogue{}, false,
              "f32");
}

/// The square ladder the tuning sweeps run: 128 to 4096 in steps of 128.
NNOPS_BENCH(cuda_matmul_ladder) {
    print_cuda_header("CUDA MatMul ladder 128..4096 step 128 (square, f32/f16)");
    for (int64_t side = 128; side <= 4096; side += 128) {
        run_dense(side, side, side, DataType::f32, false, false, Epilogue{},
                  false, "f32");
        run_dense(side, side, side, DataType::f16, false, false, Epilogue{},
                  false, "f16");
    }
}

/// The same ladder against cuBLAS, for the ratio rather than the absolute.
NNOPS_BENCH(cuda_matmul_ladder_cublas) {
    print_cuda_header("CUDA MatMul ladder vs cuBLAS (square, f32/f16)");
    for (int64_t side = 128; side <= 4096; side += 256) {
        for (DataType dt : {DataType::f32, DataType::f16}) {
            bench_vs_cublas(side, side, side, dt);
        }
    }
}

/// One 4096^3 launch, for a profiler that has to land on exactly this kernel.
NNOPS_BENCH(cuda_matmul_i8_one) {
    print_cuda_header("CUDA MatMul int8 4096^3 (single launch)");
    run_i8(4096, 4096, 4096, "i8 tb", true);
}

NNOPS_BENCH(cuda_matmul_i8) {
    print_cuda_header("CUDA MatMul int8 (u8 x s8 -> s32)");
    for (int64_t s : {128, 256, 512, 1024, 2048, 4096}) {
        run_i8(s, s, s, "i8");
    }
    run_i8(16, 4096, 4096, "i8");
    run_i8(512, 1024, 4096, "i8");
    // Decode: the narrow-m kernel, both operand orientations.
    run_i8(1, 4096, 4096, "i8");
    run_i8(4, 4096, 4096, "i8");
    run_i8(16, 4096, 4096, "i8 tb", true);
    run_i8(16, 8192, 2048, "i8 tb", true);
    run_i8(1, 4096, 4096, "i8 tb", true);
    // The k-contiguous B (Linear's [N, K] weight) takes the cp.async fill.
    for (int64_t s : {512, 1024, 2048, 4096}) {
        run_i8(s, s, s, "i8 tb", true);
    }
    run_i8(512, 1024, 4096, "i8 tb", true);
    // K scaling at a fixed output: a constant cost (epilogue, output write)
    // shows up as a non-linear share.
    run_i8(4096, 4096, 512, "i8 tb", true);
    run_i8(4096, 4096, 1024, "i8 tb", true);
    run_i8(4096, 4096, 2048, "i8 tb", true);
}

/// The int8 operator route across its output dtypes and both B layouts, on the
/// interleaved harness: twelve samples, so the number is not one lucky launch.
/// Correctness has already been settled by the CPU oracle in test_cuda_matmul.
NNOPS_BENCH(cuda_matmul_i8_ptx) {
    print_cuda_header("CUDA MatMul int8 u8xs8->s32: output dtypes, both B layouts");

    // The raw integer destination and the dequantizing ones: the epilogue
    // families the hand-written kernel answers for.
    for (DataType odt : {DataType::s32, DataType::s8, DataType::u8,
                        DataType::f32, DataType::f16}) {
    for (bool tb : {true, false}) {
        for (int64_t side : {512, 1024, 2048, 4096}) {
            const int64_t m = side, n = side, k = side;
            MatMulAttributes attrs;
            attrs.output_dtype = odt;
            attrs.transpose_b = tb;

            test::XorShift128 rng(17);
            std::vector<uint8_t> a(static_cast<size_t>(m * k));
            for (auto& e : a) { e = static_cast<uint8_t>(rng.next_u64() & 0xFFu); }
            std::vector<int8_t> b(static_cast<size_t>(k * n));
            for (auto& e : b) {
                e = static_cast<int8_t>(int32_t(rng.next_u64() % 255) - 127);
            }
            QuantParams qa;
            qa.scale = 0.01f;
            qa.zero_point = 128;
            QuantParams qb;
            qb.scale = 0.004f;
            qb.zero_point = 3;

            CudaBuffer da(a.size()), db(b.size());
            CudaBuffer dc(static_cast<size_t>(m * n) * sizeof(int32_t));
            da.upload(a.data(), a.size());
            db.upload(b.data(), b.size());
            dc.zero();

            const int64_t as[] = {m, k};
            const int64_t bs[] = {tb ? n : k, tb ? k : n};
            const int64_t cs[] = {m, n};
            TensorView av(std::span<const int64_t>(as, 2), DataType::u8,
                          da.data(), TensorLayout::Dense, qa);
            TensorView bv(std::span<const int64_t>(bs, 2), DataType::s8,
                          db.data(), TensorLayout::Dense, qb);
            TensorView cv(std::span<const int64_t>(cs, 2), odt, dc.data());

            auto op = MatMul::create(attrs, Backend::CUDA);
            const TensorView ins[] = {av, bv};
            auto ptx = [&] { cuda_launch(*op, cv, ins); };

            cudaEvent_t begin = nullptr;
            cudaEvent_t end = nullptr;
            NNOPS_CUDA_CHECK(cudaEventCreate(&begin));
            NNOPS_CUDA_CHECK(cudaEventCreate(&end));
            CudaBuffer::sync();
            ptx();
            CudaBuffer::sync();

            std::vector<double> ms_ptx;
            for (int i = 0; i < 12; ++i) {
                ms_ptx.push_back(timed_launch(ptx, begin, end));
            }
            NNOPS_CUDA_CHECK(cudaEventDestroy(begin));
            NNOPS_CUDA_CHECK(cudaEventDestroy(end));

            std::sort(ms_ptx.begin(), ms_ptx.end());
            const double p = ms_ptx[ms_ptx.size() / 2];
            const double flops = 2.0 * double(m) * double(n) * double(k);
            const auto geo = matmul_geometry(m, n, k, false, tb);
            print_cuda_row(tb ? "i8 ptx NT" : "i8 ptx NN", geo, p,
                           cuda_gflops(p, flops));
        }
    }
    }
}

/// The hand-written int8 PTX tile against cuBLAS, which reaches the same IMMA
/// instructions through `CUBLAS_COMPUTE_32I`. One caveat travels with the
/// numbers: cuBLAS computes the bare integer product, while the operator's
/// MatMulInteger contract also sweeps the two reduction planes and applies the
/// zero-point compensation per output element -- that extra work is ours and is
/// included.
NNOPS_BENCH(cuda_matmul_i8_cublas) {
    print_cuda_header("CUDA MatMul int8 s8xs8->s32: hand-written PTX vs cuBLAS");

    struct Shape {
        int64_t m, n, k;
        bool tb;
    };
    const Shape shapes[] = {
        // Squares, both B layouts.
        {128, 128, 128, false},    {256, 256, 256, false},
        {512, 512, 512, false},    {1024, 1024, 1024, false},
        {2048, 2048, 2048, false}, {4096, 4096, 4096, false},
        {128, 128, 128, true},     {256, 256, 256, true},
        {512, 512, 512, true},     {1024, 1024, 1024, true},
        {2048, 2048, 2048, true},  {4096, 4096, 4096, true},
        // A fixed output with k growing: the constant cost (epilogue, output
        // write) shows up as a non-linear share.
        {4096, 4096, 512, true},
        {4096, 4096, 1024, true},
        {4096, 4096, 2048, true},
        // Off-square, then the decode shapes, which take the narrow dp4a kernel
        // rather than the tile.
        {512, 1024, 4096, true},
        {16, 4096, 4096, false},
        {16, 4096, 4096, true},
        {1, 4096, 4096, true},
    };

    for (const Shape& sh : shapes) {
        const int64_t m = sh.m, n = sh.n, k = sh.k;
        const bool tb = sh.tb;
        MatMulAttributes attrs;
        attrs.output_dtype = DataType::s32;
        attrs.transpose_b = tb;

        test::XorShift128 rng(17);
        std::vector<int8_t> a(static_cast<size_t>(m * k));
        for (auto& e : a) {
            e = static_cast<int8_t>(int32_t(rng.next_u64() % 255) - 127);
        }
        std::vector<int8_t> b(static_cast<size_t>(k * n));
        for (auto& e : b) {
            e = static_cast<int8_t>(int32_t(rng.next_u64() % 255) - 127);
        }
        QuantParams qa;
        qa.scale = 0.01f;
        qa.zero_point = -5;
        QuantParams qb;
        qb.scale = 0.004f;
        qb.zero_point = 3;

        CudaBuffer da(a.size()), db(b.size());
        CudaBuffer dc(static_cast<size_t>(m * n) * sizeof(int32_t));
        da.upload(a.data(), a.size());
        db.upload(b.data(), b.size());
        dc.zero();

        const int64_t as[] = {m, k};
        const int64_t bs[] = {tb ? n : k, tb ? k : n};
        const int64_t cs[] = {m, n};
        TensorView av(std::span<const int64_t>(as, 2), DataType::s8, da.data(),
                      TensorLayout::Dense, qa);
        TensorView bv(std::span<const int64_t>(bs, 2), DataType::s8, db.data(),
                      TensorLayout::Dense, qb);
        TensorView cv(std::span<const int64_t>(cs, 2), DataType::s32, dc.data());

        auto op = MatMul::create(attrs, Backend::CUDA);
        const TensorView ins[] = {av, bv};

        cublasHandle_t handle = nullptr;
        NNOPS_CUBLAS_CHECK(cublasCreate(&handle));

        // The layout mapping is what the whole comparison rests on, so it is
        // checked rather than assumed: with both zero points at zero the
        // operator contract collapses to `raw`, which is exactly the bare
        // product cuBLAS is asked for, and the two must agree byte for byte.
        if (m * n <= (int64_t(1) << 20)) {
            QuantParams qa0 = qa;
            qa0.zero_point = 0;
            QuantParams qb0 = qb;
            qb0.zero_point = 0;
            TensorView a0(std::span<const int64_t>(as, 2), DataType::s8,
                          da.data(), TensorLayout::Dense, qa0);
            TensorView b0(std::span<const int64_t>(bs, 2), DataType::s8,
                          db.data(), TensorLayout::Dense, qb0);
            TensorView c0(std::span<const int64_t>(cs, 2), DataType::s32,
                          dc.data());
            CudaBuffer dref(static_cast<size_t>(m * n) * sizeof(int32_t));
            dref.zero();
            const TensorView in0[] = {a0, b0};
            cuda_launch(*op, c0, in0);
            if (tb) {
                cublas_i8_nt(handle, m, n, k, da.data(), db.data(),
                             dref.data());
            } else {
                cublas_i8_nn(handle, m, n, k, da.data(), db.data(),
                             dref.data());
            }
            std::vector<int32_t> h0(static_cast<size_t>(m * n));
            std::vector<int32_t> h1(static_cast<size_t>(m * n));
            dc.download(h0.data(), h0.size() * sizeof(int32_t));
            dref.download(h1.data(), h1.size() * sizeof(int32_t));
            size_t bad = 0;
            for (size_t i = 0; i < h0.size(); ++i) {
                if (h0[i] != h1[i]) { ++bad; }
            }
            std::printf("    check %s %lldx%lldx%lld: %zu of %zu differ\n",
                        tb ? "NT" : "NN", static_cast<long long>(m),
                        static_cast<long long>(n), static_cast<long long>(k),
                        bad, h0.size());
            dc.zero();
        }

        auto ours = [&] { cuda_launch(*op, cv, ins); };
        auto theirs = [&] {
            if (tb) {
                cublas_i8_nt(handle, m, n, k, da.data(), db.data(), dc.data());
            } else {
                cublas_i8_nn(handle, m, n, k, da.data(), db.data(), dc.data());
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

        std::vector<double> ms_ours, ms_theirs;
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
        cublasDestroy(handle);

        std::sort(ms_ours.begin(), ms_ours.end());
        std::sort(ms_theirs.begin(), ms_theirs.end());
        const double o = ms_ours[ms_ours.size() / 2];
        const double t = ms_theirs[ms_theirs.size() / 2];
        const double flops = 2.0 * double(m) * double(n) * double(k);
        const auto geo = matmul_geometry(m, n, k, false, tb);
        // The narrow-m gate sends at most sixteen rows to the dp4a kernel, so the
        // label says which of the two hand-written routes answered.
        const bool narrow = m <= 16 && k >= 256;
        char tag[48];
        std::snprintf(tag, sizeof(tag), "i8 %s %s", narrow ? "dp4a" : "ptx",
                      tb ? "NT" : "NN");
        print_cuda_row(tag, geo, o, cuda_gflops(o, flops));
        char ctag[48];
        std::snprintf(ctag, sizeof(ctag), "i8 cublas %s", tb ? "NT" : "NN");
        print_cuda_row(ctag, geo, t, cuda_gflops(t, flops));
        std::printf("    ratio %s/cublas %.3f\n", narrow ? "dp4a" : "ptx",
                    o / t);
    }
}

/// The same 4096^3 at the automatic tile against cuBLAS, both B layouts, so "is
/// cuBLAS's lead a tiling difference or an instruction-path one" is a
/// measurement, not an argument.
NNOPS_BENCH(cuda_matmul_i8_cublas_tiles) {
    print_cuda_header("CUDA MatMul int8 4096^3: auto tile vs cuBLAS");

    for (bool tb : {true, false}) {
        const int64_t m = 4096, n = 4096, k = 4096;
        MatMulAttributes attrs;
        attrs.output_dtype = DataType::s32;
        attrs.transpose_b = tb;

        test::XorShift128 rng(17);
        std::vector<int8_t> a(static_cast<size_t>(m * k));
        for (auto& e : a) {
            e = static_cast<int8_t>(int32_t(rng.next_u64() % 255) - 127);
        }
        std::vector<int8_t> b(static_cast<size_t>(k * n));
        for (auto& e : b) {
            e = static_cast<int8_t>(int32_t(rng.next_u64() % 255) - 127);
        }
        QuantParams qa;
        qa.scale = 0.01f;
        qa.zero_point = -5;
        QuantParams qb;
        qb.scale = 0.004f;
        qb.zero_point = 3;

        CudaBuffer da(a.size()), db(b.size());
        CudaBuffer dc(static_cast<size_t>(m * n) * sizeof(int32_t));
        da.upload(a.data(), a.size());
        db.upload(b.data(), b.size());
        dc.zero();
        const int64_t as[] = {m, k};
        const int64_t bs[] = {tb ? n : k, tb ? k : n};
        const int64_t cs[] = {m, n};
        TensorView av(std::span<const int64_t>(as, 2), DataType::s8, da.data(),
                      TensorLayout::Dense, qa);
        TensorView bv(std::span<const int64_t>(bs, 2), DataType::s8, db.data(),
                      TensorLayout::Dense, qb);
        TensorView cv(std::span<const int64_t>(cs, 2), DataType::s32, dc.data());
        auto op = MatMul::create(attrs, Backend::CUDA);
        const TensorView ins[] = {av, bv};
        const double flops = 2.0 * double(m) * double(n) * double(k);
        const auto geo = matmul_geometry(m, n, k, false, tb);

        cublasHandle_t handle = nullptr;
        NNOPS_CUBLAS_CHECK(cublasCreate(&handle));
        const double cm = cuda_median_ms([&] {
            if (tb) {
                cublas_i8_nt(handle, m, n, k, da.data(), db.data(), dc.data());
            } else {
                cublas_i8_nn(handle, m, n, k, da.data(), db.data(), dc.data());
            }
        });
        cublasDestroy(handle);
        print_cuda_row(tb ? "i8 cublas NT" : "i8 cublas NN", geo, cm,
                       cuda_gflops(cm, flops));

        const double ms = cuda_median_ms([&] { cuda_launch(*op, cv, ins); });
        print_cuda_row(tb ? "i8 auto NT" : "i8 auto NN", geo, ms,
                       cuda_gflops(ms, flops));
        std::printf("    ratio auto/cublas %.3f\n", ms / cm);
    }
}

#endif  // NNOPS_HAS_CUDA
