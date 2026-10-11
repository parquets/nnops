/// @file test_cpu_features.cpp
/// @brief Tests for the runtime cache-size probes in CpuFeatures.
///
/// A cache size is not something a test can choose — it is a property of the
/// host — so there is no fixed value to compare against. What *can* be pinned
/// down is that the number was actually probed rather than left at the member's
/// default, and that it is self-consistent with the geometry a real cache has.
///
/// That distinction is the whole point here, because the defaults are plausible
/// cache sizes and a broken probe is invisible from the outside. This is not
/// hypothetical: the x86_64 path walked CPUID leaf 4, stopped at the first
/// level-2 entry, and returned without ever recording a level-1 one, so
/// l1_cache_size() stayed at the 32 KiB fallback on a machine whose L1D is
/// 48 KiB. Every consumer kept working and nothing failed.

#include "nnops/detail/simd/cpu_features.hpp"
#include "common/test_harness.hpp"

#include <cstddef>

using namespace nnops::simd;

// ============================================================
// Host-independent invariants
// ============================================================

NNOPS_TEST(cpu_features_cache_sizes_are_plausible) {
    const CpuFeatures& f = CpuFeatures::get();

    // Every x86_64/aarch64 part this library runs on has at least 16 KiB of L1D
    // (the baseline ISA rules out anything older), and an L2 no smaller than the
    // L1 — the hierarchy only ever grows outward.
    NNOPS_EXPECT_TRUE(f.l1_cache_size() >= size_t{16384});
    NNOPS_EXPECT_TRUE(f.l2_cache_size() > f.l1_cache_size());

    // Cache sizes are set-associative geometry, so both are a whole number of
    // KiB. A bogus decode (say, forgetting the "+1"s in the CPUID fields, which
    // would land on 32767) fails here.
    NNOPS_EXPECT_EQ(f.l1_cache_size() % size_t{1024}, size_t{0});
    NNOPS_EXPECT_EQ(f.l2_cache_size() % size_t{1024}, size_t{0});

    // The shared-L2 accessor falls back to the per-core value when no separate
    // figure is known; it must never report *less* than the per-core one.
    NNOPS_EXPECT_TRUE(f.l2_shared_cache_size() >= f.l2_cache_size());
}

// ============================================================
// x86_64: the probe must agree with CPUID
// ============================================================

#if defined(NNOPS_ARCH_X86_64)

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace {

/// An independent re-run of the leaf-4 walk, written from the CPUID
/// documentation rather than by calling into cpu_features.cpp. Duplication is
/// the point: it is the only way to tell a probed value from a defaulted one
/// without hard-coding the answer for one machine.
void expected_x86_caches(size_t& l1_data, size_t& l2) {
    for (int subleaf = 0; ; ++subleaf) {
        int info[4] = {0};
    #if defined(_MSC_VER)
        __cpuidex(info, 4, subleaf);
    #else
        unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
        __cpuid_count(4, subleaf, eax, ebx, ecx, edx);
        info[0] = static_cast<int>(eax);
        info[1] = static_cast<int>(ebx);
        info[2] = static_cast<int>(ecx);
        info[3] = static_cast<int>(edx);
    #endif
        const int cache_type = info[0] & 0x1F;
        if (cache_type == 0) { break; }
        const int cache_level = (info[0] >> 5) & 0x7;
        if (cache_level == 1 && cache_type == 1) {
            l1_data = static_cast<size_t>(((info[1] >> 22) & 0x3FF) + 1) *
                      (((info[1] >> 12) & 0x3FF) + 1) *
                      ((info[1] & 0xFFF) + 1) *
                      static_cast<size_t>(info[2] + 1);
        } else if (cache_level == 2 && (cache_type == 1 || cache_type == 3)) {
            l2 = static_cast<size_t>(((info[1] >> 22) & 0x3FF) + 1) *
                 (((info[1] >> 12) & 0x3FF) + 1) *
                 ((info[1] & 0xFFF) + 1) *
                 static_cast<size_t>(info[2] + 1);
        }
    }
}

}  // namespace

NNOPS_TEST(cpu_features_x86_cache_sizes_match_cpuid) {
    size_t exp_l1 = 0;
    size_t exp_l2 = 0;
    expected_x86_caches(exp_l1, exp_l2);

    if (exp_l1 == 0 && exp_l2 == 0) {
        return;  // no leaf 4 on this part; the defaults are all there is
    }
    NNOPS_EXPECT_EQ(CpuFeatures::get().l1_cache_size(), exp_l1);
    NNOPS_EXPECT_EQ(CpuFeatures::get().l2_cache_size(), exp_l2);
}

#else  // !NNOPS_ARCH_X86_64

NNOPS_TEST(cpu_features_x86_cache_sizes_match_cpuid) {
    // AArch64 reads caches through the OS (getauxval / sysfs / sysctl), not
    // CPUID, and this host may not even expose them. The invariants above still
    // apply.
}

#endif
