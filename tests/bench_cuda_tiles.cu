/// @file bench_cuda_tiles.cu
/// @brief CUDA f32 GEMM tile search — one geometry under every candidate CTA tile.
///
/// A tile is a template parameter of the kernel, so each candidate is its own
/// kernel and the list has to be a compile-time one; the operators pick between
/// exactly two of them (`dense_gemm_launch`), and this bench is what prices that
/// heuristic. It is CUDA source rather than a C++ demo, unlike the other benches,
/// because the tile is spelled inside the kernel rather than handed to a launch.
/// For the same reason it repeats the timing and report lines of
/// `cuda_bench_helper.hpp` instead of including it: that header pulls in the
/// host test helpers, whose SIMD layer does not survive nvcc.
///
/// The candidates beyond the shipped pair ask three questions. Does the block
/// want 256 threads: on 64x128 that halves the thread tile (Tn 8 -> 4, so 32
/// accumulators instead of 64) and drops the register cap from 170 to 128; the
/// third-stage variant asks the same with the extra shared-memory headroom. Does
/// the A panel want the swizzle the narrow tile uses rather than the wide tile's
/// 4-float pad: the pad buys a loop-invariant row address for 2 KB a stage, the
/// swizzle pays a runtime row XOR per load instead. And does a squarer 128x128
/// CTA (the shape the CuTe tutorial uses) or a 64x256 N panel beat 64x128.
///
/// bK is 16 throughout: one stage of a 128x128 tile at bK=64 is 64 KB, more than
/// the whole per-block budget, and the tutorial's bK=8 belongs to a tensor-core
/// copy atom — an f32 row of 8 floats is half a 16 B load.
///
/// The f16 half of the bench asks a narrower question, since the LDSM copy
/// helpers pin that kernel to four warps with a 32x32 per-warp tile: the CTA tile
/// still moves (64x128 compiles, because `gemm()` replicates the accumulator
/// fragment), but only the block count it costs is at stake. There the wide tile
/// halves the residents, three to two, so it is a question about k depth and grid
/// depth rather than about traffic alone — hence the extra short-k geometry.
///
/// Reading the rows: tiles are only comparable within one geometry, and the two
/// the operators ship are named as such. This bench only times; a candidate is
/// adopted only after the CPU-oracle tests have run it.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "common/bench_harness.hpp"

#include "backend/cuda/gemm/cute_gemm.cuh"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef NNOPS_HAS_CUDA

namespace {

void check(cudaError_t err)
{
    if (err != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(err)); }
}

/// Median GPU milliseconds of one launch, after a warm-up: one sample per call,
/// so a host hitch moves a sample rather than the result.
template <class F>
double median_ms(F&& launch, int warmup = 3, int samples = 11)
{
    check(cudaDeviceSynchronize());
    for (int i = 0; i < warmup; ++i) { launch(); }
    check(cudaDeviceSynchronize());

    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    check(cudaEventCreate(&begin));
    check(cudaEventCreate(&end));

    std::vector<float> ms;
    ms.reserve(size_t(samples));
    for (int i = 0; i < samples; ++i) {
        check(cudaEventRecord(begin));
        launch();
        check(cudaEventRecord(end));
        check(cudaEventSynchronize(end));
        float elapsed = 0.0f;
        check(cudaEventElapsedTime(&elapsed, begin, end));
        ms.push_back(elapsed);
    }

    check(cudaEventDestroy(begin));
    check(cudaEventDestroy(end));

    const size_t mid = ms.size() / 2;
    std::nth_element(ms.begin(), ms.begin() + std::ptrdiff_t(mid), ms.end());
    return double(ms[mid]);
}

std::string geometry(int64_t m, int64_t n, int64_t k)
{
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%lldx%lldx%lld", (long long)m, (long long)n,
                  (long long)k);
    return buf;
}

double gflops(double ms, int64_t m, int64_t n, int64_t k)
{
    return ms > 0.0 ? 2.0 * double(m) * double(n) * double(k) / (ms * 1e6) : 0.0;
}

/// One tile on one geometry. The buffers are the caller's, so a whole list runs
/// on one set of operands: at 4096³ the operands are 512 MB and an upload per
/// tile would cost more than the measurement.
template <class Tile>
double run_tile(const char* tag, const void* a, const void* b, void* c, int64_t m,
                int64_t n, int64_t k)
{
    nnops::backend::cuda::GemmParams p;
    p.a = a;
    p.b = b;
    p.c = c;
    p.m = int(m);
    p.n = int(n);
    p.k = int(k);
    p.lda = k;
    p.ldb = n;
    p.ldc = n;

    auto launch = [&] {
        nnops::backend::cuda::dense_gemm_f32_tile<Tile, false, false>(p, nullptr);
    };
    const double ms = median_ms(launch);
    const double rate = gflops(ms, m, n, k);

    std::printf("  %-18s %-34s %9.4f ms %9.2f GFLOPS\n", tag, geometry(m, n, k).c_str(),
                ms, rate);
    return rate;
}

void run_tile_shape(int64_t m, int64_t n, int64_t k)
{
    namespace cuda_gemm = nnops::backend::cuda;

    void* a = nullptr;
    void* b = nullptr;
    void* c = nullptr;
    const size_t sa = size_t(m) * size_t(k) * sizeof(float);
    const size_t sb = size_t(k) * size_t(n) * sizeof(float);
    const size_t sc = size_t(m) * size_t(n) * sizeof(float);
    check(cudaMalloc(&a, sa));
    check(cudaMalloc(&b, sb));
    check(cudaMalloc(&c, sc));
    check(cudaMemset(a, 0, sa));
    check(cudaMemset(b, 0, sb));

    run_tile<cuda_gemm::kF32Narrow>("64x64 t128 s3 b4", a, b, c, m, n, k);
    run_tile<cuda_gemm::kF32Wide>("64x128 t128 s2 b3", a, b, c, m, n, k);
    // The same two shipped tiles with the A panel swizzled instead of padded: the
    // pad keeps the row address loop invariant, the swizzle costs a runtime row
    // XOR per load but 2 KB less shared memory a stage.
    run_tile<cuda_gemm::GemmTileF32<128, 8, 64, 128, 16, 2, 3, 0>>("64x128 t128 swz",
                                                                    a, b, c, m, n, k);
    run_tile<cuda_gemm::GemmTileF32<256, 8, 64, 128, 16, 2, 2, 0>>("64x128 t256 swz",
                                                                   a, b, c, m, n, k);
    run_tile<cuda_gemm::GemmTileF32<256, 8, 64, 128, 16, 2, 2, 4>>("64x128 t256 s2 b2",
                                                                   a, b, c, m, n, k);
    run_tile<cuda_gemm::GemmTileF32<256, 8, 64, 128, 16, 3, 2, 4>>("64x128 t256 s3 b2",
                                                                   a, b, c, m, n, k);
    run_tile<cuda_gemm::GemmTileF32<256, 8, 128, 128, 16, 2, 2, 4>>("128x128 t256 s2 b2",
                                                                    a, b, c, m, n, k);
    // 128x128 at the one other k-tile that fits, 256 threads. bK has to be a
    // multiple of 16 (the A copy moves sixteen floats per pass) and be a power of
    // two for the swizzle; k=32 is the only candidate — k=64 is 64 KB a stage,
    // over the whole per-block budget. One block of shared memory is all that is
    // left at k=32, so the register cap is 256 and nothing here spills.
    run_tile<cuda_gemm::GemmTileF32<256, 8, 128, 128, 16, 2, 2, 0>>("128x128 k16 swz",
                                                                    a, b, c, m, n, k);
    run_tile<cuda_gemm::GemmTileF32<256, 8, 128, 128, 32, 2, 1, 4>>("128x128 k32 pad",
                                                                    a, b, c, m, n, k);
    run_tile<cuda_gemm::GemmTileF32<256, 8, 128, 128, 32, 2, 1, 0>>("128x128 k32 swz",
                                                                    a, b, c, m, n, k);
    run_tile<cuda_gemm::GemmTileF32<256, 8, 64, 256, 16, 2, 2, 4>>("64x256 t256 s2 b2", a,
                                                                   b, c, m, n, k);

    check(cudaFree(a));
    check(cudaFree(b));
    check(cudaFree(c));
}

/// The f16 twin of `run_tile`: same layout, tensor-core kernel.
template <class Tile>
double run_tile_h(const char* tag, const void* a, const void* b, void* c, int64_t m,
                  int64_t n, int64_t k)
{
    nnops::backend::cuda::GemmParams p;
    p.a = a;
    p.b = b;
    p.c = c;
    p.m = int(m);
    p.n = int(n);
    p.k = int(k);
    p.lda = k;
    p.ldb = n;
    p.ldc = n;

    auto launch = [&] {
        nnops::backend::cuda::dense_gemm_f16_tile<Tile, false, false>(p, nullptr);
    };
    const double ms = median_ms(launch);
    const double rate = gflops(ms, m, n, k);

    std::printf("  %-18s %-34s %9.4f ms %9.2f GFLOPS\n", tag, geometry(m, n, k).c_str(),
                ms, rate);
    return rate;
}

void run_tile_shape_h(int64_t m, int64_t n, int64_t k)
{
    namespace cuda_gemm = nnops::backend::cuda;

    void* a = nullptr;
    void* b = nullptr;
    void* c = nullptr;
    const size_t sa = size_t(m) * size_t(k) * 2;
    const size_t sb = size_t(k) * size_t(n) * 2;
    const size_t sc = size_t(m) * size_t(n) * 2;
    check(cudaMalloc(&a, sa));
    check(cudaMalloc(&b, sb));
    check(cudaMalloc(&c, sc));
    check(cudaMemset(a, 0, sa));
    check(cudaMemset(b, 0, sb));

    // The shape this kernel shipped with before the search: 48 KB, two blocks.
    run_tile_h<cuda_gemm::GemmTileH<128, 64, 64, 64, 3, 2, 2, 32, 32, 8>>("64x64 s3 b2", a,
                                                                         b, c, m, n, k);
    run_tile_h<cuda_gemm::kF16Narrow>("64x64 s2 b3", a, b, c, m, n, k);
    run_tile_h<cuda_gemm::kF16Wide>("64x128 s2 b2", a, b, c, m, n, k);

    check(cudaFree(a));
    check(cudaFree(b));
    check(cudaFree(c));
}

}  // namespace

NNOPS_BENCH(cuda_matmul_tiles) {
    // Rows are unbuffered so a failing tile is visible at the row that broke
    // rather than lost with the buffer.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("\n=== CUDA MatMul f32 tile search (explicit tiles) ===\n");
    for (int64_t s : {1024, 2048, 4096}) {
        run_tile_shape(s, s, s);
    }
    // The oblong pair the matmul benches sweep: a prefill GEMM and a decode GEMM,
    // where the grid is small enough that the tile is a question about the tail.
    run_tile_shape(512, 1024, 4096);
    run_tile_shape(16, 4096, 4096);

    std::printf("\n=== CUDA MatMul f16 tile search (explicit tiles) ===\n");
    for (int64_t s : {1024, 2048, 4096}) {
        run_tile_shape_h(s, s, s);
    }
    run_tile_shape_h(512, 1024, 4096);
    // Short k with a full grid: the wide tile has no traffic left to save here and
    // pays for its missing third block instead.
    run_tile_shape_h(4096, 4096, 1024);
    run_tile_shape_h(16, 4096, 4096);
}

#endif  // NNOPS_HAS_CUDA
