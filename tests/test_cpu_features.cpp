/// @file test_cpu_features.cpp
/// @brief Tests for the runtime cache-size probes in CpuFeatures.
///
/// A cache size is a property of the host, so there is no fixed value to
/// compare against. What can be pinned down is that the number was probed
/// rather than left at the member's default, and that it is self-consistent
/// with the geometry a real cache has.
///
/// That distinction is the whole point: the defaults (32 KiB L1D, 256 KiB L2)
/// are themselves plausible cache sizes, so a probe that silently never records
/// a level is invisible from the outside. The x86_64 test therefore re-runs
/// CPUID itself instead of hard-coding the answer for one machine.

#include "nnops/detail/simd/cpu_features.hpp"
#include "common/test_harness.hpp"

#include <cstddef>

using namespace nnops::simd;

// ============================================================
// Host-independent invariants
// ============================================================

NNOPS_TEST(cpu_features_cache_sizes_are_plausible) {
    const CpuFeatures& f = CpuFeatures::get();

    // The baseline ISA rules out a part with less than 16 KiB of L1D, and the
    // hierarchy only grows outward, so L2 is strictly larger.
    NNOPS_EXPECT_TRUE(f.l1_cache_size() >= size_t{16384});
    NNOPS_EXPECT_TRUE(f.l2_cache_size() > f.l1_cache_size());

    // Set-associative geometry makes both a whole number of KiB; a field-decode
    // slip in the probe fails here.
    NNOPS_EXPECT_EQ(f.l1_cache_size() % size_t{1024}, size_t{0});
    NNOPS_EXPECT_EQ(f.l2_cache_size() % size_t{1024}, size_t{0});

    // The shared-L2 accessor falls back to the per-core value when no separate
    // figure is known, so it must never report less than the per-core one.
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
/// documentation rather than by calling into cpu_features.cpp. The duplication
/// is the point: it is the only way to tell a probed value from a defaulted one
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
    // CPUID, and this host may not expose them. The invariants above still
    // apply.
}

#endif
