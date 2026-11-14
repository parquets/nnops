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
/// cuBLAS runs as one more arm of that search, not on a separate row: the same
/// kernel measured 13% apart in two processes here, so only arms interleaved in
/// one process are comparable, and the ratio column is relative to the shipped
/// tile.
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
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef NNOPS_HAS_CUDA

#include <cublas_v2.h>

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

/// One arm of the f16 search: a tag, a launch closure over the shared operands,
/// and the samples it collected.
struct Cand {
    const char* tag;
    std::function<void()> run;
    std::vector<double> ms;
};

/// Every candidate once per round, rotating which arm leads, then each arm's
/// median. This is what makes the numbers worth reading: the same kernel
/// measured 13% apart in two processes, and two arms measured back to back drift
/// together, which is several times the difference a tile choice makes.
/// Interleaved, the arms see the same clock and the same cache state, so the
/// ratio between them survives.
void round_robin(std::vector<Cand>& cands, int64_t m, int64_t n, int64_t k,
                 int rounds = 11)
{
    for (auto& c : cands) { c.run(); }
    check(cudaDeviceSynchronize());

    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    check(cudaEventCreate(&begin));
    check(cudaEventCreate(&end));

    for (int r = 0; r < rounds; ++r) {
        const size_t lead = size_t(r) % cands.size();
        for (size_t i = 0; i < cands.size(); ++i) {
            Cand& c = cands[(lead + i) % cands.size()];
            check(cudaEventRecord(begin));
            c.run();
            check(cudaEventRecord(end));
            check(cudaEventSynchronize(end));
            float elapsed = 0.0f;
            check(cudaEventElapsedTime(&elapsed, begin, end));
            c.ms.push_back(double(elapsed));
        }
    }

    check(cudaEventDestroy(begin));
    check(cudaEventDestroy(end));

    auto median = [](std::vector<double>& v) {
        const size_t mid = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + std::ptrdiff_t(mid), v.end());
        return v[mid];
    };
    // The first candidate is the shipped one, so the ratio column reads as
    // "what a switch away from today's kernel would buy".
    const double base = median(cands[0].ms);
    for (auto& c : cands) {
        const double ms = median(c.ms);
        std::printf("  %-18s %-34s %9.4f ms %9.2f GFLOPS  %6.3f x\n", c.tag,
                    geometry(m, n, k).c_str(), ms, gflops(ms, m, n, k), base / ms);
    }
}

/// cuBLAS for a row-major f16 product: column-major C^T = B^T·A^T, as in the
/// matmul bench.
void cublas_h(cublasHandle_t h, int64_t m, int64_t n, int64_t k, const void* a,
              const void* b, void* c)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int ni = int(n), mi = int(m), ki = int(k);
    const cublasStatus_t st = cublasGemmEx(
        h, CUBLAS_OP_N, CUBLAS_OP_N, ni, mi, ki, &alpha, b, CUDA_R_16F, ni, a,
        CUDA_R_16F, ki, &beta, c, CUDA_R_16F, ni, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) { throw std::runtime_error("cublasGemmEx"); }
}

/// The transposed-B twin: `C = A·Bᵀ` with B stored `[n][k]`, reached as the
/// column-major product with B's rows as the k dimension.
void cublas_h_nt(cublasHandle_t h, int64_t m, int64_t n, int64_t k, const void* a,
                 const void* b, void* c)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int ni = int(n), mi = int(m), ki = int(k);
    const cublasStatus_t st = cublasGemmEx(
        h, CUBLAS_OP_T, CUBLAS_OP_N, ni, mi, ki, &alpha, b, CUDA_R_16F, ki, a,
        CUDA_R_16F, ki, &beta, c, CUDA_R_16F, ni, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) { throw std::runtime_error("cublasGemmEx"); }
}

/// `TB` stores B as [n][k], which is how `Linear` hands a weight over: the same
/// tiles, but the B panel is a k-contiguous strip per column and the product is
/// the one a decode GEMM asks for.
template <bool TB>
void run_tile_shape_h_t(int64_t m, int64_t n, int64_t k)
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

    nnops::backend::cuda::GemmParams p;
    p.a = a;
    p.b = b;
    p.c = c;
    p.m = int(m);
    p.n = int(n);
    p.k = int(k);
    p.lda = k;
    p.ldb = TB ? k : n;
    p.ldc = n;

    std::vector<Cand> cands;
    auto add = [&](const char* tag, auto* tile) {
        using Tile = std::remove_pointer_t<decltype(tile)>;
        cands.push_back({tag,
                         [p] {
                             nnops::backend::cuda::dense_gemm_f16_tile<Tile, false, TB>(
                                 p, nullptr);
                         },
                         {}});
    };

    add("64x64 s2 b3", static_cast<cuda_gemm::kF16Narrow*>(nullptr));   // shipped
    add("64x128 s2 b2", static_cast<cuda_gemm::kF16Wide*>(nullptr));    // shipped
    // Two of the shipped tile side by side in one CTA: same 32x32 warp tiles and
    // the same four warps per ldmatrix group, but eight warps and two tiles a
    // block, 48 KB and two residents. It is the only way to raise warps per SM
    // without a hand-written ldmatrix.
    {
        const nnops::backend::cuda::GemmParams p4 = p;
        cands.push_back({"64x64 G2",
                         [p4] {
                             cuda_gemm::dense_gemm_f16_groups<
                                 cuda_gemm::kF16Narrow, 2, false, TB>(p4, nullptr);
                         },
                         {}});
        // Four tiles, 80 KB and one resident, used to be the winning arm at one m
        // tile (1.61x at 16x4096x4096, 1.89x at 32x2048x4096). It no longer
        // compiles: 512 threads against a 64x32 B panel asks `gemm_copy_half` for
        // 128 rows per pass, and its static_assert refuses to walk off the panel.
        // Two groups is the ceiling on that tile, and two is what ships.
    }
    // One more stage on the same tile: 48 KB still leaves both residents, so this
    // is a third k chunk in flight at no occupancy cost -- the depth a 0.84-wave
    // grid is short of.
    {
        const nnops::backend::cuda::GemmParams p5 = p;
        using S4 = cuda_gemm::GemmTileH<128, 64, 64, 32, 4, 2, 2, 32, 32, 4>;
        add("64x64 s4", static_cast<S4*>(nullptr));
        cands.push_back({"64x64 s4 G2",
                         [p5] {
                             cuda_gemm::dense_gemm_f16_groups<S4, 2, false, TB>(p5,
                                                                               nullptr);
                         },
                         {}});
    }
    // The L2 raster width: how many m tiles one group walks before it advances
    // n. Four is shipped; eight and sixteen are the wider bands it replaced, and
    // they are what the width was chosen against.
    add("64x64 g8",
        static_cast<cuda_gemm::GemmTileH<128, 64, 64, 64, 2, 2, 2, 32, 32, 8>*>(nullptr));
    add("64x64 g16",
        static_cast<cuda_gemm::GemmTileH<128, 64, 64, 64, 2, 2, 2, 32, 32, 16>*>(nullptr));
    // The shape this kernel shipped with before the 3-block variants: 48 KB and
    // two residents, kept as the measure of what the split bought.
    add("64x64 s3 b2",
        static_cast<cuda_gemm::GemmTileH<128, 64, 64, 64, 3, 2, 2, 32, 32, 8>*>(nullptr));
    // Eight warps per CTA is not expressible through CuTe's LDSM copy helpers:
    // `make_tiled_copy_A/B` tiles the atom over the MMA's thread layout, and any
    // warp grid beyond 2x2 leaves it too few values per thread
    // ("TiledCopy uses too few vals for selected CopyAtom", copy_atom.hpp:206)
    // or hands `copy()` mismatched slices. Four warps is the ceiling this kernel
    // has without hand-written ldmatrix loads.

    // cuBLAS on its own buffers: neither arm warms the other's pages.
    void* ga = nullptr;
    void* gb = nullptr;
    void* gc = nullptr;
    check(cudaMalloc(&ga, sa));
    check(cudaMalloc(&gb, sb));
    check(cudaMalloc(&gc, sc));
    check(cudaMemset(ga, 0, sa));
    check(cudaMemset(gb, 0, sb));
    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasCreate");
    }
    if constexpr (TB) {
        cands.push_back(
            {"cublas", [=] { cublas_h_nt(handle, m, n, k, ga, gb, gc); }, {}});
    } else {
        cands.push_back({"cublas", [=] { cublas_h(handle, m, n, k, ga, gb, gc); }, {}});
    }

    round_robin(cands, m, n, k);

    cublasDestroy(handle);
    check(cudaFree(ga));
    check(cudaFree(gb));
    check(cudaFree(gc));
    check(cudaFree(a));
    check(cudaFree(b));
    check(cudaFree(c));
}

void run_tile_shape_h(int64_t m, int64_t n, int64_t k)
{
    run_tile_shape_h_t<false>(m, n, k);
}

/// The transposed-B tile question, asked alone: the shipped tile at one and two
/// groups, the same tile with a fourth stage, and cuBLAS on its own buffers.
/// Everything else in the f16 search is a question about the m axis, which this
/// layout does not have — one m tile is all a decode GEMM has.
///
/// These arms launch the kernel directly, so they run with no k split, while the
/// operator cuts this shape in two (0.15 ms against the 0.24 a row here shows).
/// The ratio column is what to read; the absolute number is the unsplit arm.
void run_tile_shape_h_tb(int64_t m, int64_t n, int64_t k)
{
    namespace cuda_gemm = nnops::backend::cuda;

    void* a = nullptr;
    void* b = nullptr;
    void* c = nullptr;
    const size_t sa = size_t(m) * size_t(k) * 2;
    const size_t sb = size_t(n) * size_t(k) * 2;
    const size_t sc = size_t(m) * size_t(n) * 2;
    check(cudaMalloc(&a, sa));
    check(cudaMalloc(&b, sb));
    check(cudaMalloc(&c, sc));
    check(cudaMemset(a, 0, sa));
    check(cudaMemset(b, 0, sb));

    nnops::backend::cuda::GemmParams p;
    p.a = a;
    p.b = b;
    p.c = c;
    p.m = int(m);
    p.n = int(n);
    p.k = int(k);
    p.lda = k;
    p.ldb = k;
    p.ldc = n;

    using S4 = cuda_gemm::GemmTileH<128, 64, 64, 32, 4, 2, 2, 32, 32, 4>;

    std::vector<Cand> cands;
    cands.push_back({"64x64 s3 G1", [p] { cuda_gemm::dense_gemm_f16_tile<
                                             cuda_gemm::kF16Narrow, false, true>(
                                             p, nullptr); }, {}});
    cands.push_back({"64x64 s3 G2", [p] { cuda_gemm::dense_gemm_f16_groups<
                                             cuda_gemm::kF16Narrow, 2, false, true>(
                                             p, nullptr); }, {}});
    cands.push_back({"64x64 s4 G1",
                     [p] { cuda_gemm::dense_gemm_f16_tile<S4, false, true>(p, nullptr); },
                     {}});
    cands.push_back({"64x64 s4 G2",
                     [p] {
                         cuda_gemm::dense_gemm_f16_groups<S4, 2, false, true>(p, nullptr);
                     },
                     {}});
    // A narrower m tile is what a decode GEMM wants and what this kernel cannot
    // give it: the ldmatrix copy helper takes only a 32x32 warp tile
    // (`size<1>(src_c) == size<1>(dst_c)`), so bM=32 needs wm*mmaM=64 and fails
    // the tile's own assert, while mmaM=16 and bN=32 both fail inside CuTe. 64
    // rows is the floor, and three quarters of them are waste at m=16.

    void* ga = nullptr;
    void* gb = nullptr;
    void* gc = nullptr;
    check(cudaMalloc(&ga, sa));
    check(cudaMalloc(&gb, sb));
    check(cudaMalloc(&gc, sc));
    check(cudaMemset(ga, 0, sa));
    check(cudaMemset(gb, 0, sb));
    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasCreate");
    }
    cands.push_back({"cublas", [=] { cublas_h_nt(handle, m, n, k, ga, gb, gc); }, {}});

    round_robin(cands, m, n, k);

    cublasDestroy(handle);
    check(cudaFree(ga));
    check(cudaFree(gb));
    check(cudaFree(gc));
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
    for (int64_t s : {512, 1024, 2048, 4096}) {
        run_tile_shape_h(s, s, s);
    }
    run_tile_shape_h(512, 4096, 4096);
    run_tile_shape_h(512, 1024, 4096);
    // Short k with a full grid: the wide tile has no traffic left to save here and
    // pays for its missing third block instead.
    run_tile_shape_h(4096, 4096, 1024);
    run_tile_shape_h(16, 4096, 4096);
}

/// The transposed-B decode product on its own, so the tile question can be asked
/// about the layout `Linear` actually hands over without the squares in the same
/// run paying for it.
NNOPS_BENCH(cuda_matmul_tiles_tb) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("\n=== CUDA MatMul f16 tile search, B stored [n][k] ===\n");
    run_tile_shape_h_tb(16, 4096, 4096);
    run_tile_shape_h_tb(64, 4096, 4096);
    run_tile_shape_h_tb(16, 8192, 2048);
}

#endif  // NNOPS_HAS_CUDA
