/// @file bench_pack_mma.cpp
/// @brief Micro-benchmarks for the pack/MMA micro-kernels, in isolation.
///
/// bench_matmul.cpp measures end-to-end GEMM, where a gap between the
/// pack_a=1 route (transposed A) and the pack_a=0 route (raw strided A) is
/// visible but not attributable. This harness calls the micro-kernels directly
/// so the cost of the A access pattern can be separated from tiling, packing,
/// threading and the epilogue.
///
/// Three geometries are timed per K, all against the same packed-B panel:
///   pack      — mma_pack with interleaved A   (the pack_a=1 inner loop)
///   direct/lda— mma_direct on row-major A with lda == K, which is exactly the
///               layout the pack_a=0 dispatch feeds it for a compact input
///   direct/pad— mma_direct on the same A with a padded row pitch (lda = K + 64),
///               the wide-stride geometry PACK_A_STRIDE_THRESHOLD exists to catch;
///               isolating it from direct/lda separates any stride/cache-line
///               effect from the kernel's own instruction mix
///
/// GFLOPS are computed from the kernel's real tile (mr x nr x K x 2), not from
/// a nominal size, so the three rows are directly comparable.

#include "nnops/detail/simd.hpp"           // NNOPS_ARCH_* selection
#include "common/bench_harness.hpp"
#include "common/random_tensor.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#if defined(NNOPS_ARCH_X86_64)
  #include "backend/cpu/x86_64/mma_pack_f32.hpp"
  #include "backend/cpu/x86_64/mma_direct_f32.hpp"
  using namespace nnops::backend::cpu::x86_64;
  #define NNOPS_BENCH_MMA_MR 6
  #define NNOPS_BENCH_MMA_NR 16
  #define NNOPS_BENCH_MMA_PACK   mma_pack_6x16_f32
  #define NNOPS_BENCH_MMA_DIRECT mma_direct_6x16_f32
#elif defined(NNOPS_ARCH_AARCH64)
  #include "backend/cpu/aarch64/mma_pack_f32.hpp"
  #include "backend/cpu/aarch64/mma_direct_f32.hpp"
  using namespace nnops::backend::cpu::aarch64;
  #define NNOPS_BENCH_MMA_MR 8
  #define NNOPS_BENCH_MMA_NR 12
  #define NNOPS_BENCH_MMA_PACK   mma_pack_8x12_f32
  #define NNOPS_BENCH_MMA_DIRECT mma_direct_8x12_f32
#else
  #error "Unsupported architecture"
#endif

namespace {

constexpr int MR = NNOPS_BENCH_MMA_MR;
constexpr int NR = NNOPS_BENCH_MMA_NR;

constexpr int REPS = 200000;

/// Extra row pitch for the wide-stride A case (the geometry
/// PACK_A_STRIDE_THRESHOLD exists to catch). Kept distinct from K at every
/// bench point so the two direct rows never collapse into the same layout.
constexpr int STRIDE_PAD = 64;

/// Distinct C tiles cycled through so consecutive kernel calls do not
/// serialize through a store->load on the same accumulator block. Without this
/// the per-call epilogue latency dominates at small K and the numbers are
/// meaningless.
constexpr int C_TILES = 64;

/// Fill with a runtime-derived PRNG. Constant data lets the compiler fold the
/// whole FMA chain (a constant A and B fold to a constant C), which reports
/// impossible throughput.
void fill_random(float* p, size_t n, uint64_t seed) {
    nnops::test::XorShift128 rng(seed);
    for (size_t i = 0; i < n; ++i) {
        p[i] = static_cast<float>(rng.next_u64() % 1000) * 1e-3f;
    }
}

/// Sum of every kernel result, written through a volatile so the timings
/// below cannot be optimised away. Never read.
volatile double g_sink = 0.0;

/// Time `mma` over REPS calls; returns GFLOPS for the mr x nr x K tile.
template <typename Fn>
double time_kernel(Fn&& mma, int K) {
    // C_TILES separate MR x NR accumulator blocks, kept live across the loop.
    std::vector<float> C(static_cast<size_t>(C_TILES) * MR * NR);
    for (auto& v : C) { v = 0.0f; }

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < REPS; ++i) {
        mma(C.data() + (static_cast<size_t>(i % C_TILES) * MR * NR));
    }
    const auto t1 = std::chrono::steady_clock::now();

    // Sink every tile so nothing is dead-code-eliminated.
    for (float v : C) { g_sink += static_cast<double>(v); }

    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return 2.0 * MR * NR * static_cast<double>(K) * REPS / ns;
}

struct Case {
    const char* name;
    double gflops;
};

void report(int K, const std::vector<Case>& cases) {
    std::printf("  pack_mma f32  mr=%d nr=%d  K=%d  reps=%d\n", MR, NR, K, REPS);
    for (const auto& c : cases) {
        std::printf("      %-14s %8.1f GFLOPS\n", c.name, c.gflops);
    }
    std::printf("      %-14s %8.3f\n", "direct/pack",
                cases[1].gflops / cases[0].gflops);
}

void bench_geometry(int K) {
    const int lda_pad = K + STRIDE_PAD;

    std::vector<float> A_packed(static_cast<size_t>(MR) * K);
    std::vector<float> A_padded(static_cast<size_t>(MR) * lda_pad);
    std::vector<float> A_compact(static_cast<size_t>(MR) * K);
    std::vector<float> B(static_cast<size_t>(NR) * K);

    // Seed from the address, so the data is runtime-opaque but reproducible.
    const uint64_t seed = 0x9E3779B97F4A7C15ull
                        ^ static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&K));
    fill_random(A_packed.data(), A_packed.size(), seed ^ 1);
    fill_random(A_padded.data(), A_padded.size(), seed ^ 2);
    fill_random(A_compact.data(), A_compact.size(), seed ^ 3);
    fill_random(B.data(), B.size(), seed ^ 4);

    const double g_pack = time_kernel(
        [&](float* c) {
            NNOPS_BENCH_MMA_PACK(c, NR, A_packed.data(), B.data(), NR, K, -1e30f, 1e30f);
        }, K);
    const double g_padded = time_kernel(
        [&](float* c) {
            NNOPS_BENCH_MMA_DIRECT(c, NR, A_padded.data(), lda_pad,
                                   B.data(), NR, K, -1e30f, 1e30f);
        }, K);
    const double g_compact = time_kernel(
        [&](float* c) {
            NNOPS_BENCH_MMA_DIRECT(c, NR, A_compact.data(), K,
                                   B.data(), NR, K, -1e30f, 1e30f);
        }, K);

    // cases[0] is pack and cases[1] is a direct row, so report()'s ratio holds.
    report(K, {{"mma_pack", g_pack},
               {"mma_direct/lda", g_compact},
               {"mma_direct/pad", g_padded}});
}

}  // namespace

NNOPS_BENCH(pack_mma_f32_k128) { bench_geometry(128); }
NNOPS_BENCH(pack_mma_f32_k512) { bench_geometry(512); }
