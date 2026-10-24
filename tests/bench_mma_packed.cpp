/// @file bench_mma_packed.cpp
/// @brief Micro-benchmarks for the x86_64 / aarch64 packed-AB / packed-B MMA
///        kernels at every tile height in the arch's mr menu.
///
/// Both kernel families take a runtime `mr`, so one 6x16 kernel serves any
/// partial-M tile by aliasing rows past `mr` onto row mr-1. This harness shows
/// what that costs: for each MR level the kernel is timed at every
/// `mr = 1..MR` and the throughput is reported against the *live* tile
/// (mr x 16 x K x 2 flops), so a row is directly comparable to the full-height
/// one and to the shorter levels.
///
/// Two B layouts are timed per dtype and K:
///   packed-AB — A as [kc][MR], the interleaved layout mma_pack writes
///   packed-B  — A row-major with row pitch lda == kc, the direct layout
///
/// The driver picks the widest level that fits the remaining rows; if a
/// partial tile ran at a fraction of the full-height rate, that choice would
/// cost throughput, and the per-mr rows are what shows it.
///
/// A second sweep varies `nc` at level 6 / mr 6 and reports each width against
/// nc=16. B is packed as `ceil(nc/16)` full 16-wide panels, so a run of whole
/// panels is one k-loop each and the tail panel is one more (8-lane when the
/// remainder is at most 8, the full 16-wide body otherwise).

#include "common/bench_harness.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/simd.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <vector>

#if defined(NNOPS_ARCH_X86_64) || defined(NNOPS_ARCH_AARCH64)

#if defined(NNOPS_ARCH_X86_64)
  #include "backend/cpu/x86_64/pack_f32.hpp"   // mr_f32 / nr_f32 menus
  #include "backend/cpu/x86_64/pack_f16.hpp"
  #include "backend/cpu/x86_64/mma_f32.hpp"
  #include "backend/cpu/x86_64/mma_f16.hpp"
  namespace bench_arch = nnops::backend::cpu::x86_64;
#else
  #include "backend/cpu/aarch64/pack_f32.hpp"  // mr_f32 / nr_f32 menus
  #include "backend/cpu/aarch64/pack_f16.hpp"
  #include "backend/cpu/aarch64/mma_f32.hpp"
  #include "backend/cpu/aarch64/mma_f16.hpp"
  namespace bench_arch = nnops::backend::cpu::aarch64;
#endif

namespace {

using namespace bench_arch;
using nnops::backend::cpu::half;
using nnops::backend::cpu::float_to_half;
using nnops::backend::cpu::half_to_float;
using nnops::backend::cpu::mma_pack_align;

constexpr int REPS = 100000;

/// RHS panel width of the active arch. Both x86 dtypes are 16 wide; aarch64
/// f32 is 12 wide (aarch64 f16 is 16).
template <typename T>
constexpr int nr_of() noexcept {
    if constexpr (std::is_same_v<T, float>) { return bench_arch::nr_f32[0]; }
    else { return bench_arch::nr_f16[0]; }
}

/// Widest and middle tile heights in the arch's mr menu: {6,4,1} on x86,
/// {8,4,1} on aarch64. The x86 menu also times the extra mr=2 level.
template <typename T>
constexpr int mr_head() noexcept {
    if constexpr (std::is_same_v<T, float>) { return bench_arch::mr_f32[0]; }
    else { return bench_arch::mr_f16[0]; }
}

template <typename T>
constexpr int mr_mid() noexcept {
    if constexpr (std::is_same_v<T, float>) { return bench_arch::mr_f32[1]; }
    else { return bench_arch::mr_f16[1]; }
}

/// Distinct C tiles cycled through so consecutive calls do not serialize
/// through a store->load on the same accumulator block.
constexpr int C_TILES = 64;

// ---- dtype bridges --------------------------------------------------------

template <typename T> T as_type(float v) noexcept;
template <> float as_type<float>(float v) noexcept { return v; }
template <> half  as_type<half>(float v) noexcept { return float_to_half(v); }

template <typename T> double as_double(T v) noexcept;
template <> double as_double<float>(float v) noexcept { return static_cast<double>(v); }
template <> double as_double<half>(half v) noexcept {
    return static_cast<double>(half_to_float(v));
}

/// Constant data would let the compiler fold the FMA chain; seed from the
/// address so the data is runtime-opaque but reproducible. (A local xorshift
/// keeps this file free of the <span>-dependent tensor helpers.)
template <typename T>
void fill_random(T* p, size_t n, uint64_t seed) {
    uint64_t s = seed | 1;
    for (size_t i = 0; i < n; ++i) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        p[i] = as_type<T>(static_cast<float>(s % 1000) * 1e-3f);
    }
}

/// Kernel signatures, per dtype (mr first, as in XNNPACK).
template <typename T>
using AbKernel = void (*)(int, int, int, T*, int, const T*, const T*, float, float);
template <typename T>
using BKernel  = void (*)(int, int, int, T*, int, const T*, int, const T*,
                          float, float);

template <typename T>
AbKernel<T> ab_fn(int level) {
#ifdef NNOPS_ARCH_X86_64
    if constexpr (std::is_same_v<T, float>) {
        switch (level) {
        case 6: return mma_packed_ab_mr6_nr16_mrxncxkc_f32<false>;
        case 4: return mma_packed_ab_mr4_nr16_mrxncxkc_f32<false>;
        case 2: return mma_packed_ab_mr2_nr16_mrxncxkc_f32<false>;
        default: return mma_packed_ab_mr1_nr16_mrxncxkc_f32<false>;
        }
    } else {
        switch (level) {
        case 6: return mma_packed_ab_mr6_nr16_mrxncxkc_f16<false>;
        case 4: return mma_packed_ab_mr4_nr16_mrxncxkc_f16<false>;
        case 2: return mma_packed_ab_mr2_nr16_mrxncxkc_f16<false>;
        default: return mma_packed_ab_mr1_nr16_mrxncxkc_f16<false>;
        }
    }
#else
    if constexpr (std::is_same_v<T, float>) {
        switch (level) {
        case 8: return mma_packed_ab_mr8_nr12_mrxncxkc_f32<false>;
        case 4: return mma_packed_ab_mr4_nr12_mrxncxkc_f32<false>;
        default: return mma_packed_ab_mr1_nr12_mrxncxkc_f32<false>;
        }
    } else {
        switch (level) {
        case 8: return mma_packed_ab_mr8_nr16_mrxncxkc_f16<false>;
        case 4: return mma_packed_ab_mr4_nr16_mrxncxkc_f16<false>;
        default: return mma_packed_ab_mr1_nr16_mrxncxkc_f16<false>;
        }
    }
#endif
}

template <typename T>
BKernel<T> b_fn(int level) {
#ifdef NNOPS_ARCH_X86_64
    if constexpr (std::is_same_v<T, float>) {
        switch (level) {
        case 6: return mma_packed_b_mr6_nr16_mrxncxkc_f32<false>;
        case 4: return mma_packed_b_mr4_nr16_mrxncxkc_f32<false>;
        case 2: return mma_packed_b_mr2_nr16_mrxncxkc_f32<false>;
        default: return mma_packed_b_mr1_nr16_mrxncxkc_f32<false>;
        }
    } else {
        switch (level) {
        case 6: return mma_packed_b_mr6_nr16_mrxncxkc_f16<false>;
        case 4: return mma_packed_b_mr4_nr16_mrxncxkc_f16<false>;
        case 2: return mma_packed_b_mr2_nr16_mrxncxkc_f16<false>;
        default: return mma_packed_b_mr1_nr16_mrxncxkc_f16<false>;
        }
    }
#else
    if constexpr (std::is_same_v<T, float>) {
        switch (level) {
        case 8: return mma_packed_b_mr8_nr12_mrxncxkc_f32<false>;
        case 4: return mma_packed_b_mr4_nr12_mrxncxkc_f32<false>;
        default: return mma_packed_b_mr1_nr12_mrxncxkc_f32<false>;
        }
    } else {
        switch (level) {
        case 8: return mma_packed_b_mr8_nr16_mrxncxkc_f16<false>;
        case 4: return mma_packed_b_mr4_nr16_mrxncxkc_f16<false>;
        default: return mma_packed_b_mr1_nr16_mrxncxkc_f16<false>;
        }
    }
#endif
}

/// Sum of every kernel result, written through a volatile so the timings
/// cannot be optimised away. Never read.
volatile double g_sink = 0.0;

/// Time `reps` calls of one kernel tile; returns GFLOPS for the live
/// mr x nc x kc tile. `MR` is the tile height C is sized at.
template <typename T, typename Fn>
double time_kernel(Fn&& mma, int mr, int MR, int nc, int ldc, int kc, int reps) {
    const size_t tile = static_cast<size_t>(MR) * ldc;
    std::vector<T> C(static_cast<size_t>(C_TILES) * tile, as_type<T>(0.0f));

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        mma(C.data() + (static_cast<size_t>(i % C_TILES) * tile));
    }
    const auto t1 = std::chrono::steady_clock::now();

    for (T v : C) { g_sink += as_double<T>(v); }

    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return 2.0 * mr * nc * static_cast<double>(kc) * reps / ns;
}

template <typename T, int MR>
void bench_ab(const char* dtype, int kc) {
    constexpr int NR = nr_of<T>();
    const int ldb = static_cast<int>(mma_pack_align(NR * kc * sizeof(T)) / sizeof(T));
    const uint64_t seed = 0x9E3779B97F4A7C15ull
                        ^ static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&kc));

    // A as [kc][MR]; the kernel reads row rMr for every live row.
    std::vector<T> A(static_cast<size_t>(kc) * MR);
    std::vector<T> B(static_cast<size_t>(ldb));
    fill_random<T>(A.data(), A.size(), seed ^ 1);
    fill_random<T>(B.data(), B.size(), seed ^ 2);

    const AbKernel<T> fn = ab_fn<T>(MR);

    std::printf("  mma_packed_ab %s  level=%d nr=%d  kc=%d  reps=%d\n",
                dtype, MR, NR, kc, REPS);
    for (int mr = 1; mr <= MR; ++mr) {
        const double g = time_kernel<T>(
            [&](T* c) { fn(mr, NR, kc, c, NR, A.data(), B.data(), -1e30f, 1e30f); },
            mr, MR, NR, NR, kc, REPS);
        std::printf("      mr=%d/%d  %8.1f GFLOPS\n", mr, MR, g);
    }
}

template <typename T, int MR>
void bench_b(const char* dtype, int kc) {
    constexpr int NR = nr_of<T>();
    const int lda = kc;
    const int ldb = static_cast<int>(mma_pack_align(NR * kc * sizeof(T)) / sizeof(T));
    const uint64_t seed = 0x9E3779B97F4A7C15ull
                        ^ static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&kc));

    // A row-major with row pitch lda == kc (the compact-input geometry).
    std::vector<T> A(static_cast<size_t>(lda) * MR);
    std::vector<T> B(static_cast<size_t>(ldb));
    fill_random<T>(A.data(), A.size(), seed ^ 3);
    fill_random<T>(B.data(), B.size(), seed ^ 4);

    const BKernel<T> fn = b_fn<T>(MR);

    std::printf("  mma_packed_b %s  level=%d nr=%d  kc=%d  reps=%d\n",
                dtype, MR, NR, kc, REPS);
    for (int mr = 1; mr <= MR; ++mr) {
        const double g = time_kernel<T>(
            [&](T* c) {
                fn(mr, NR, kc, c, NR, A.data(), lda, B.data(), -1e30f, 1e30f);
            },
            mr, MR, NR, NR, kc, REPS);
        std::printf("      mr=%d/%d  %8.1f GFLOPS\n", mr, MR, g);
    }
}

// ---- nc sweep -------------------------------------------------------------
// The kernel takes the full output width: it consumes `ceil(nc/16)` full
// 16-wide panels, so a wide tile is a run of full-panel k-loops plus one tail
// k-loop, while a narrow tile is only the tail. The sweep below fixes the tile
// at level 6 / mr 6 / kc and varies nc, reporting each width against nc=16.

constexpr int NC_LDC = 64;   // C row pitch, wider than any nc in the sweep
constexpr int NC_KC = 256;

/// Panels the packer lays down for `nc` columns at width `nr`: every panel is
/// a full `nr`-wide slot, the last one zero-padded past `nc`.
int panel_slots(int nc, int nr) { return (nc + nr - 1) / nr; }

/// Same work in every measurement, so a 1-column call is not timed for
/// nanoseconds. Clamped so the widest tiles still finish quickly.
int reps_for(int mr, int nc, int kc) {
    const double calls = 5e9 / (2.0 * mr * nc * kc);
    return static_cast<int>(std::min(4e6, std::max(2e4, calls)));
}

struct NcPoint { int nc; double gflops; };

void print_nc_rows(const std::vector<NcPoint>& rows) {
    double ref = 0.0;
    for (const auto& r : rows) { if (r.nc == 16) { ref = r.gflops; } }
    for (const auto& r : rows) {
        std::printf("      nc=%-3d %8.1f GFLOPS   %5.1f%% of nc=16\n",
                    r.nc, r.gflops, 100.0 * r.gflops / ref);
    }
}

template <typename T>
void bench_nc_ab(const char* dtype) {
    constexpr int MR = mr_head<T>();
    constexpr int NR = nr_of<T>();
    const int ldb = static_cast<int>(mma_pack_align(NR * NC_KC * sizeof(T)) / sizeof(T));
    std::vector<T> A(static_cast<size_t>(NC_KC) * MR);
    fill_random<T>(A.data(), A.size(), 0xA11CE);

    const AbKernel<T> fn = ab_fn<T>(MR);
    std::printf("  mma_packed_ab %s  level=%d mr=%d kc=%d ldc=%d  (nc sweep)\n",
                dtype, MR, MR, NC_KC, NC_LDC);

    std::vector<NcPoint> rows;
    for (int nc : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16,
                   17, 20, 24, 28, 31, 32, 33, 40, 48, 64}) {
        const int nslots = panel_slots(nc, NR);
        std::vector<T> B(static_cast<size_t>(nslots) * ldb);
        fill_random<T>(B.data(), B.size(), 0xB0B ^ static_cast<uint64_t>(nc));

        const int reps = reps_for(MR, nc, NC_KC);
        const double g = time_kernel<T>(
            [&](T* c) {
                fn(MR, nc, NC_KC, c, NC_LDC, A.data(), B.data(), -1e30f, 1e30f);
            },
            MR, MR, nc, NC_LDC, NC_KC, reps);
        rows.push_back({nc, g});
    }
    print_nc_rows(rows);
}

template <typename T>
void bench_nc_b(const char* dtype) {
    constexpr int MR = mr_head<T>();
    constexpr int NR = nr_of<T>();
    const int lda = NC_KC;
    const int ldb = static_cast<int>(mma_pack_align(NR * NC_KC * sizeof(T)) / sizeof(T));
    std::vector<T> A(static_cast<size_t>(lda) * MR);
    fill_random<T>(A.data(), A.size(), 0xC0FFEE);

    const BKernel<T> fn = b_fn<T>(MR);
    std::printf("  mma_packed_b %s  level=%d mr=%d kc=%d ldc=%d  (nc sweep)\n",
                dtype, MR, MR, NC_KC, NC_LDC);

    std::vector<NcPoint> rows;
    for (int nc : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16,
                   17, 20, 24, 28, 31, 32, 33, 40, 48, 64}) {
        const int nslots = panel_slots(nc, NR);
        std::vector<T> B(static_cast<size_t>(nslots) * ldb);
        fill_random<T>(B.data(), B.size(), 0xBEEF ^ static_cast<uint64_t>(nc));

        const int reps = reps_for(MR, nc, NC_KC);
        const double g = time_kernel<T>(
            [&](T* c) {
                fn(MR, nc, NC_KC, c, NC_LDC, A.data(), lda, B.data(), -1e30f, 1e30f);
            },
            MR, MR, nc, NC_LDC, NC_KC, reps);
        rows.push_back({nc, g});
    }
    print_nc_rows(rows);
}

template <typename T>
void bench_nc_dtype(const char* dtype) {
    bench_nc_ab<T>(dtype);
    bench_nc_b<T>(dtype);
}

// ---- large-nc tail sweep ---------------------------------------------------
// Above 128 columns a 1..15-column tail is a small fraction of the work, so
// the absolute throughput stays near peak; what it shows is the *shape*: one
// narrow block for t = 1..8, two for t = 9..15, and a step back to the full
// panel rate at t = 16. Each base is its own reference (t = 0).

template <typename Launch>
void print_large_nc(const char* name, const char* dtype, int mr, Launch&& launch) {
    const int bases[] = {128, 256, 512};
    constexpr int NBASE = 3;
    std::vector<double> grid[NBASE];

    for (int b = 0; b < NBASE; ++b) {
        for (int t = 0; t <= 15; ++t) {
            const int nc = bases[b] + t;
            const int reps = reps_for(mr, nc, NC_KC);
            grid[b].push_back(launch(nc, reps));
        }
    }

    std::printf("  %s %s  level=%d mr=%d kc=%d  (tail t = nc - base)\n",
                name, dtype, mr, mr, NC_KC);
    std::printf("      %-6s", "t");
    for (int b = 0; b < NBASE; ++b) { std::printf("   base=%-4d GFLOPS     %%", bases[b]); }
    std::printf("\n");
    for (int t = 0; t <= 15; ++t) {
        std::printf("      t=%-4d", t);
        for (int b = 0; b < NBASE; ++b) {
            const double ref = grid[b][0];
            std::printf("   %10.1f %6.1f", grid[b][t], 100.0 * grid[b][t] / ref);
        }
        std::printf("\n");
    }
}

template <typename T>
void bench_large_nc_ab(const char* dtype) {
    constexpr int MR = mr_head<T>();
    constexpr int NR = nr_of<T>();
    const int ldb = static_cast<int>(mma_pack_align(NR * NC_KC * sizeof(T)) / sizeof(T));
    std::vector<T> A(static_cast<size_t>(NC_KC) * MR);
    fill_random<T>(A.data(), A.size(), 0x1DEA);
    const AbKernel<T> fn = ab_fn<T>(MR);

    print_large_nc("mma_packed_ab", dtype, MR, [&](int nc, int reps) {
        const int nslots = panel_slots(nc, NR);
        std::vector<T> B(static_cast<size_t>(nslots) * ldb);
        fill_random<T>(B.data(), B.size(), 0xDEAD ^ static_cast<uint64_t>(nc));
        return time_kernel<T>(
            [&](T* c) {
                fn(MR, nc, NC_KC, c, nc, A.data(), B.data(), -1e30f, 1e30f);
            },
            MR, MR, nc, nc, NC_KC, reps);
    });
}

template <typename T>
void bench_large_nc_b(const char* dtype) {
    constexpr int MR = mr_head<T>();
    constexpr int NR = nr_of<T>();
    const int lda = NC_KC;
    const int ldb = static_cast<int>(mma_pack_align(NR * NC_KC * sizeof(T)) / sizeof(T));
    std::vector<T> A(static_cast<size_t>(lda) * MR);
    fill_random<T>(A.data(), A.size(), 0xBEE5);
    const BKernel<T> fn = b_fn<T>(MR);

    print_large_nc("mma_packed_b", dtype, MR, [&](int nc, int reps) {
        const int nslots = panel_slots(nc, NR);
        std::vector<T> B(static_cast<size_t>(nslots) * ldb);
        fill_random<T>(B.data(), B.size(), 0xF00D ^ static_cast<uint64_t>(nc));
        return time_kernel<T>(
            [&](T* c) {
                fn(MR, nc, NC_KC, c, nc, A.data(), lda, B.data(), -1e30f, 1e30f);
            },
            MR, MR, nc, nc, NC_KC, reps);
    });
}

template <typename T>
void bench_large_nc_dtype(const char* dtype) {
    bench_large_nc_ab<T>(dtype);
    bench_large_nc_b<T>(dtype);
}

template <typename T>
void bench_dtype(const char* dtype, int kc) {
    bench_ab<T, mr_head<T>()>(dtype, kc);
    bench_ab<T, mr_mid<T>()>(dtype, kc);
#ifdef NNOPS_ARCH_X86_64
    bench_ab<T, 2>(dtype, kc);   // extra level in the x86 menu only
#endif
    bench_ab<T, 1>(dtype, kc);
    bench_b<T, mr_head<T>()>(dtype, kc);
    bench_b<T, mr_mid<T>()>(dtype, kc);
#ifdef NNOPS_ARCH_X86_64
    bench_b<T, 2>(dtype, kc);
#endif
    bench_b<T, 1>(dtype, kc);
}

void bench_f32(int kc) { bench_dtype<float>("f32", kc); }
void bench_f16(int kc) { bench_dtype<half>("f16", kc); }

}  // namespace

NNOPS_BENCH(mma_packed_f32_k128) { bench_f32(128); }
NNOPS_BENCH(mma_packed_f32_k512) { bench_f32(512); }
NNOPS_BENCH(mma_packed_f16_k128) { bench_f16(128); }
NNOPS_BENCH(mma_packed_f16_k512) { bench_f16(512); }
NNOPS_BENCH(mma_packed_nc_f32) { bench_nc_dtype<float>("f32"); }
NNOPS_BENCH(mma_packed_nc_f16) { bench_nc_dtype<half>("f16"); }
NNOPS_BENCH(mma_packed_tail_f32) { bench_large_nc_dtype<float>("f32"); }
NNOPS_BENCH(mma_packed_tail_f16) { bench_large_nc_dtype<half>("f16"); }

#endif  // NNOPS_ARCH_X86_64 || NNOPS_ARCH_AARCH64
