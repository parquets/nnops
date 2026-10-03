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
/// GFLOPS are computed from each kernel's real tile (mr x nr x K x 2), not from
/// a nominal size, so the three rows are directly comparable even where the
/// pack and direct paths use different tile heights (see NNOPS_BENCH_MMA_MR*).
/// Both f32 and f16 are timed; the f16 pair is the interesting one because its
/// pack and direct kernels share a tile height (mr=8 on aarch64, mr=6 on
/// x86_64), so any gap is the kernel itself, not a panel-width difference.

#include "nnops/detail/simd.hpp"           // NNOPS_ARCH_* selection
#include "nnops/detail/half.hpp"           // half, float_to_half, half_to_float
#include "common/bench_harness.hpp"
#include "common/random_tensor.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#if defined(NNOPS_ARCH_X86_64)
  #include "backend/cpu/x86_64/mma_pack_f32.hpp"
  #include "backend/cpu/x86_64/mma_direct_f32.hpp"
  #include "backend/cpu/x86_64/mma_pack_f16.hpp"
  #include "backend/cpu/x86_64/mma_direct_f16.hpp"
  using namespace nnops::backend::cpu::x86_64;
  #define NNOPS_BENCH_MMA_MR 6
  #define NNOPS_BENCH_MMA_MR_DIRECT 6
  #define NNOPS_BENCH_MMA_NR 16
  #define NNOPS_BENCH_MMA_PACK   mma_pack_6x16_f32<false>
  #define NNOPS_BENCH_MMA_DIRECT mma_direct_6x16_f32<false>
  // f16 packs and reads A at the same height (mr=6, nr=16) on x86_64.
  #define NNOPS_BENCH_MMA_F16_MR 6
  #define NNOPS_BENCH_MMA_F16_MR_DIRECT 6
  #define NNOPS_BENCH_MMA_F16_NR 16
  #define NNOPS_BENCH_MMA_F16_PACK   mma_pack_6x16_f16<false>
  #define NNOPS_BENCH_MMA_F16_DIRECT mma_direct_6x16_f16<false>
#elif defined(NNOPS_ARCH_AARCH64)
  #include "backend/cpu/aarch64/mma_pack_f32.hpp"
  #include "backend/cpu/aarch64/mma_direct_f32.hpp"
  #include "backend/cpu/aarch64/mma_pack_f16.hpp"
  #include "backend/cpu/aarch64/mma_direct_f16.hpp"
  using namespace nnops::backend::cpu::aarch64;
  // The f32 pack and direct paths tile M differently: the packed layout keeps
  // four rows per A vector and runs at mr=8, while row-major A pins one vector
  // per row and has to drop to mr=6 to stay inside the register file. These
  // macros must track arch::mr_f32 / arch::mr_f32_direct.
  #define NNOPS_BENCH_MMA_MR 8
  #define NNOPS_BENCH_MMA_MR_DIRECT 6
  #define NNOPS_BENCH_MMA_NR 12
  #define NNOPS_BENCH_MMA_PACK   mma_pack_8x12_f32<false>
  #define NNOPS_BENCH_MMA_DIRECT mma_direct_6x12_f32<false>
  // f16 has no such split: both routes use mr=8, nr=16 (there is no
  // mr_f16_direct), so pack-vs-direct here is purely the kernel + copy.
  #define NNOPS_BENCH_MMA_F16_MR 8
  #define NNOPS_BENCH_MMA_F16_MR_DIRECT 8
  #define NNOPS_BENCH_MMA_F16_NR 16
  #define NNOPS_BENCH_MMA_F16_PACK   mma_pack_8x16_f16<false>
  #define NNOPS_BENCH_MMA_F16_DIRECT mma_direct_8x16_f16<false>
#else
  #error "Unsupported architecture"
#endif

namespace {

using nnops::backend::cpu::half;

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

// ---- dtype bridges (float vs half) ----------------------------------------
// The f32 and f16 kernels take the same shape of arguments; these let the
// harness be written once and instantiated per dtype.

template <typename T> T as_type(float v) noexcept;
template <> float as_type<float>(float v) noexcept { return v; }
template <> half  as_type<half>(float v) noexcept { return nnops::backend::cpu::float_to_half(v); }

template <typename T> double as_double(T v) noexcept;
template <> double as_double<float>(float v) noexcept { return static_cast<double>(v); }
template <> double as_double<half>(half v) noexcept {
    return static_cast<double>(nnops::backend::cpu::half_to_float(v));
}

/// Fill with a runtime-derived PRNG. Constant data lets the compiler fold the
/// whole FMA chain (a constant A and B fold to a constant C), which reports
/// impossible throughput.
template <typename T>
void fill_random(T* p, size_t n, uint64_t seed) {
    nnops::test::XorShift128 rng(seed);
    for (size_t i = 0; i < n; ++i) {
        p[i] = as_type<T>(static_cast<float>(rng.next_u64() % 1000) * 1e-3f);
    }
}

/// Kernel signatures, per dtype. Instantiated with the concrete micro-kernels.
template <typename T>
using PackKernel   = void (*)(T* NNOPS_RESTRICT, int, const T* NNOPS_RESTRICT,
                              const T* NNOPS_RESTRICT, int, int, float, float);

template <typename T>
using DirectKernel = void (*)(T* NNOPS_RESTRICT, int, const T* NNOPS_RESTRICT, int,
                              const T* NNOPS_RESTRICT, int, int, float, float);

/// Sum of every kernel result, written through a volatile so the timings
/// below cannot be optimised away. Never read.
volatile double g_sink = 0.0;

/// Time `mma` over REPS calls; returns GFLOPS for the mr x nr x K tile.
/// `mr` is the tile height the kernel was instantiated at, which is not the
/// same for the pack and direct rows on aarch64 f32.
template <typename T, typename Fn>
double time_kernel(Fn&& mma, int mr, int nr, int K) {
    // C_TILES separate mr x nr accumulator blocks, kept live across the loop.
    const size_t tile = static_cast<size_t>(mr) * nr;
    std::vector<T> C(static_cast<size_t>(C_TILES) * tile);
    for (auto& v : C) { v = as_type<T>(0.0f); }

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < REPS; ++i) {
        mma(C.data() + (static_cast<size_t>(i % C_TILES) * tile));
    }
    const auto t1 = std::chrono::steady_clock::now();

    // Sink every tile so nothing is dead-code-eliminated.
    for (T v : C) { g_sink += as_double<T>(v); }

    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return 2.0 * mr * nr * static_cast<double>(K) * REPS / ns;
}

struct Case {
    const char* name;
    double gflops;
};

void report(const char* dtype, int mr, int nr, int K, const std::vector<Case>& cases) {
    std::printf("  pack_mma %s  mr=%d nr=%d  K=%d  reps=%d\n", dtype, mr, nr, K, REPS);
    for (const auto& c : cases) {
        std::printf("      %-14s %8.1f GFLOPS\n", c.name, c.gflops);
    }
    std::printf("      %-14s %8.3f\n", "direct/pack",
                cases[1].gflops / cases[0].gflops);
}

template <typename T>
void bench_geometry(const char* dtype, int mr, int mr_direct, int nr, int K,
                    PackKernel<T> pack_fn, DirectKernel<T> direct_fn) {
    const int lda_pad = K + STRIDE_PAD;

    std::vector<T> A_packed(static_cast<size_t>(mr) * K);
    std::vector<T> A_padded(static_cast<size_t>(mr_direct) * lda_pad);
    std::vector<T> A_compact(static_cast<size_t>(mr_direct) * K);
    std::vector<T> B(static_cast<size_t>(nr) * K);

    // Seed from the address, so the data is runtime-opaque but reproducible.
    const uint64_t seed = 0x9E3779B97F4A7C15ull
                        ^ static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&K));
    fill_random<T>(A_packed.data(), A_packed.size(), seed ^ 1);
    fill_random<T>(A_padded.data(), A_padded.size(), seed ^ 2);
    fill_random<T>(A_compact.data(), A_compact.size(), seed ^ 3);
    fill_random<T>(B.data(), B.size(), seed ^ 4);

    const double g_pack = time_kernel<T>(
        [&](T* c) { pack_fn(c, nr, A_packed.data(), B.data(), nr, K, -1e30f, 1e30f); },
        mr, nr, K);
    const double g_padded = time_kernel<T>(
        [&](T* c) { direct_fn(c, nr, A_padded.data(), lda_pad, B.data(), nr, K, -1e30f, 1e30f); },
        mr_direct, nr, K);
    const double g_compact = time_kernel<T>(
        [&](T* c) { direct_fn(c, nr, A_compact.data(), K, B.data(), nr, K, -1e30f, 1e30f); },
        mr_direct, nr, K);

    // cases[0] is pack and cases[1] is a direct row, so report()'s ratio holds.
    report(dtype, mr, nr, K, {{"mma_pack", g_pack},
                              {"mma_direct/lda", g_compact},
                              {"mma_direct/pad", g_padded}});
}

void bench_f32(int K) {
    bench_geometry<float>("f32", NNOPS_BENCH_MMA_MR, NNOPS_BENCH_MMA_MR_DIRECT,
                          NNOPS_BENCH_MMA_NR, K,
                          NNOPS_BENCH_MMA_PACK, NNOPS_BENCH_MMA_DIRECT);
}

void bench_f16(int K) {
    bench_geometry<half>("f16", NNOPS_BENCH_MMA_F16_MR, NNOPS_BENCH_MMA_F16_MR_DIRECT,
                         NNOPS_BENCH_MMA_F16_NR, K,
                         NNOPS_BENCH_MMA_F16_PACK, NNOPS_BENCH_MMA_F16_DIRECT);
}

}  // namespace

NNOPS_BENCH(pack_mma_f32_k128) { bench_f32(128); }
NNOPS_BENCH(pack_mma_f32_k512) { bench_f32(512); }
NNOPS_BENCH(pack_mma_f16_k128) { bench_f16(128); }
NNOPS_BENCH(pack_mma_f16_k512) { bench_f16(512); }
