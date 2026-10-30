/// @file cpu_features.cpp
/// @brief Runtime CPU feature, cache-size and hybrid-topology detection.
///
/// x86_64 reads CPUID directly: leaf 1 / leaf 7 for the ISA bits, with the AVX
/// family gated on the OS-enabled XCR0 state, and leaf 4's deterministic-cache
/// walk for the L1D and L2 figures the GEMM tile solver derives kc (L1) and
/// mc/nc (L2) from. On AArch64 the ISA comes from getauxval (HWCAP/HWCAP2) or
/// IsProcessorFeaturePresent, and the cache sizes from sysfs, sysctl or
/// GetLogicalProcessorInformation.
///
/// Every detector leaves the caller's default in place when it finds nothing, so
/// the header's 32 KiB L1D / 256 KiB L2 stand in on a platform that reports no
/// geometry. detect_fast_cores() returns the fast-class core count on a hybrid
/// part and 0 when the platform reports a single class, where an equal split is
/// already even.

#include "nnops/detail/simd/cpu_features.hpp"

#include <vector>

#if defined(NNOPS_ARCH_X86_64)

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

namespace {

bool cpuid_bit(unsigned int leaf, unsigned int subleaf, unsigned int reg, unsigned int bit) {
#if defined(_MSC_VER)
    int cpu_info[4] = {0};
    __cpuidex(cpu_info, leaf, subleaf);
    return (cpu_info[reg] & (1u << bit)) != 0;
#else
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
    const unsigned int* regs[4] = {&eax, &ebx, &ecx, &edx};
    return (*regs[reg] & (1u << bit)) != 0;
#endif
}

bool os_avx_support() {
#if defined(_MSC_VER)
    return (_xgetbv(0) & 0x6) == 0x6;  // bits 1 (SSE) and 2 (AVX) set
#else
    unsigned int eax = 0, edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (eax & 0x6) == 0x6;
#endif
}

bool os_avx512_support() {
    // Check XCR0 bits: 1(SSE), 2(AVX), 5(OPMASK), 6(ZMM_HI256), 7(HI16_ZMM)
#if defined(_MSC_VER)
    uint64_t xcr0 = _xgetbv(0);
    return (xcr0 & 0xE6) == 0xE6;
#else
    unsigned int eax = 0, edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (eax & 0xE6) == 0xE6;
#endif
}

// The level-1 entry must be the *data* cache (type 1): type 2 is the instruction
// cache and never holds a packed panel. And the walk must not stop at the first
// cache it can use — the enumeration is not guaranteed to be level-ordered.
// Leaf 4 is absent before Nehalem/Bulldozer; cache_type 0 then ends the walk and
// both outputs keep their defaults.
inline void cpuid_cache_sizes(size_t& l1_data_bytes, size_t& l2_bytes) {
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
        if (cache_type == 0) { break; }  // no more caches

        const int cache_level = (info[0] >> 5) & 0x7;
        // type 1 = data cache, type 3 = unified cache
        const bool is_l1_data = (cache_level == 1 && cache_type == 1);
        const bool is_l2      = (cache_level == 2 && (cache_type == 1 || cache_type == 3));
        if (!is_l1_data && !is_l2) { continue; }

        const int line_size  = (info[1] & 0xFFF) + 1;
        const int partitions = ((info[1] >> 12) & 0x3FF) + 1;
        const int ways       = ((info[1] >> 22) & 0x3FF) + 1;
        const int sets       = info[2] + 1;
        const size_t bytes =
            static_cast<size_t>(ways) * partitions * line_size * sets;
        if (is_l1_data) { l1_data_bytes = bytes; }
        else            { l2_bytes = bytes; }
    }
}

} // anonymous namespace

#elif defined(NNOPS_ARCH_AARCH64)

#if defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#include <cstdio>
#include <cstdlib>

namespace {

// Read a Linux sysfs cache size file ("48K" etc.); 0 if absent or unparseable,
// leaving the caller's default. cpu0 rather than a sum: every core this targets
// has the same L1 and L2. index1 is the data cache, index2 the unified one
// (index0 is L1I).
size_t read_sysfs_cache_size(const char* path) {
    std::FILE* fp = std::fopen(path, "r");
    if (!fp) { return 0; }
    char buf[32] = {};
    size_t out = 0;
    if (std::fgets(buf, sizeof(buf), fp)) {
        char* end = nullptr;
        unsigned long val = std::strtoul(buf, &end, 10);
        if (end != buf && val > 0) {
            if (*end == 'K' || *end == 'k') {
                val *= 1024;
            } else if (*end == 'M' || *end == 'm') {
                val *= 1024 * 1024;
            } else if (*end == 'G' || *end == 'g') {
                val *= 1024 * 1024 * 1024;
            }
            out = static_cast<size_t>(val);
        }
    }
    std::fclose(fp);
    return out;
}

} // anonymous namespace
#elif defined(_WIN32)
#include <Windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#endif  // NNOPS_ARCH_AARCH64

// ---- Topology: is this a hybrid part, and how many cores run at full speed? ----
#if defined(_WIN32)
#include <Windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace {

/// Fast-class core count, or 0 on a single class. Windows publishes one
/// EfficiencyClass per core; macOS exposes the P cluster as
/// hw.perflevel0.physicalcpu.
int detect_fast_cores() {
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) { return 0; }
    std::vector<char> buf(static_cast<size_t>(len));
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()),
            &len)) {
        return 0;
    }
    int best = -1;
    int best_count = 0;
    int total = 0;
    for (DWORD off = 0; off < len;) {
        const auto* e = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
            buf.data() + off);
        if (e->Size == 0) { break; }
        const int cls = static_cast<int>(e->Processor.EfficiencyClass);
        ++total;
        if (cls > best) {
            best = cls;
            best_count = 1;
        } else if (cls == best) {
            ++best_count;
        }
        off += e->Size;
    }
    return (best_count > 0 && best_count < total) ? best_count : 0;
#elif defined(__APPLE__)
    int32_t perf = 0;
    size_t len = sizeof(perf);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &perf, &len, nullptr, 0) != 0 ||
        perf <= 0) {
        return 0;
    }
    // A single-perflevel Mac answers perflevel0 with the whole CPU, so only a
    // machine with a second level has a fast class.
    int32_t cores = 0;
    len = sizeof(cores);
    if (sysctlbyname("hw.physicalcpu", &cores, &len, nullptr, 0) == 0 &&
        cores <= perf) {
        return 0;
    }
    return perf;
#else
    return 0;
#endif
}

} // anonymous namespace

namespace nnops {
namespace simd {

void CpuFeatures::detect() {
    features_ = 0;

#if defined(NNOPS_ARCH_X86_64)
    // SSE 4.1: CPUID.1.ECX[19]
    if (cpuid_bit(1, 0, 2, 19)) {
        features_ |= static_cast<uint32_t>(CpuIsa::SSE4_1);
    }

    // AVX: CPUID.1.ECX[28] + OS XSAVE support
    if (cpuid_bit(1, 0, 2, 28) && os_avx_support()) {
        features_ |= static_cast<uint32_t>(CpuIsa::AVX);

        // FMA: CPUID.1.ECX[12]
        if (cpuid_bit(1, 0, 2, 12)) {
            features_ |= static_cast<uint32_t>(CpuIsa::FMA);
        }

        // AVX2: CPUID.7.EBX[5]
        if (cpuid_bit(7, 0, 1, 5)) {
            features_ |= static_cast<uint32_t>(CpuIsa::AVX2);
        }

        // AVX-512F: CPUID.7.EBX[16] + OS support
        if (cpuid_bit(7, 0, 1, 16) && os_avx512_support()) {
            features_ |= static_cast<uint32_t>(CpuIsa::AVX512F);
        }
    }

    cpuid_cache_sizes(l1_cache_size_, l2_cache_size_);

#elif defined(NNOPS_ARCH_AARCH64)
    // NEON (ASIMD) is mandatory on AArch64
    features_ |= static_cast<uint32_t>(CpuIsa::NEON);

#if defined(__linux__)
    long hwcap = getauxval(AT_HWCAP);
    long hwcap2 = getauxval(AT_HWCAP2);

    if (hwcap & HWCAP_ASIMDDP) {
        features_ |= static_cast<uint32_t>(CpuIsa::NEON_DOT);
    }
    #ifdef HWCAP_FPHP
    if (hwcap & HWCAP_FPHP) {
        features_ |= static_cast<uint32_t>(CpuIsa::NEON_FP16);
    }
    #endif

#elif defined(_WIN32)
    // NEON (ASIMD) is always available. 43 is
    // PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE, spelled numerically so this compiles
    // against SDKs that predate the enumerator.
    if (IsProcessorFeaturePresent(43)) {
        features_ |= static_cast<uint32_t>(CpuIsa::NEON_DOT);
    }
#endif

    // Apple's L1D is read in the sysctl block below.
#if defined(__linux__)
    {
        // sysfs only: there is no getauxval(AT_L1D_CACHESIZE), and HWCAP does
        // not describe cache geometry at all.
        const size_t l1 = read_sysfs_cache_size(
            "/sys/devices/system/cpu/cpu0/cache/index1/size");
        if (l1 > 0) { l1_cache_size_ = l1; }
    }
#endif

#if defined(__linux__)
    {
        // getauxval(AT_L2_CACHESIZE) exists since Linux 6.6; the header may not
        // define the macro yet.
        #ifndef AT_L2_CACHESIZE
        #define AT_L2_CACHESIZE 43
        #endif
        long aux_l2 = getauxval(AT_L2_CACHESIZE);
        if (aux_l2 > 0) {
            l2_cache_size_ = static_cast<size_t>(aux_l2);
        } else {
            const size_t l2 = read_sysfs_cache_size(
                "/sys/devices/system/cpu/cpu0/cache/index2/size");
            if (l2 > 0) { l2_cache_size_ = l2; }
        }
    }
#elif defined(_WIN32)
    {
        // One pass collects both levels, each taking the first matching entry:
        // the cache entries are not documented to be ordered by level, so
        // stopping at the first L2 can walk past the L1 data cache.
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION buf[256];
        DWORD len = sizeof(buf);
        if (GetLogicalProcessorInformation(buf, &len)) {
            DWORD count = len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
            bool have_l1 = false;
            bool have_l2 = false;
            for (DWORD i = 0; i < count; ++i) {
                if (buf[i].Relationship != RelationCache) { continue; }
                const CACHE_DESCRIPTOR& c = buf[i].Cache;
                if (c.Level == 1 && c.Type == CacheData && !have_l1) {
                    l1_cache_size_ = c.Size;
                    have_l1 = true;
                } else if (c.Level == 2 && !have_l2) {
                    l2_cache_size_ = c.Size;
                    have_l2 = true;
                }
            }
        }
    }
#elif defined(__APPLE__)
    {
        int64_t l1_size = 0;
        size_t len = sizeof(l1_size);
        if (sysctlbyname("hw.l1dcachesize", &l1_size, &len, nullptr, 0) == 0 &&
            l1_size > 0) {
            l1_cache_size_ = static_cast<size_t>(l1_size);
        }

        int64_t l2_size = 0;
        len = sizeof(l2_size);
        if (sysctlbyname("hw.l2cachesize", &l2_size, &len, nullptr, 0) == 0 &&
            l2_size > 0) {
            l2_cache_size_ = static_cast<size_t>(l2_size);
        }
    }
#endif

    // On a hybrid Mac hw.l2cachesize is the *efficiency* cluster's L2; ask for the
    // performance cluster's explicitly when the OS exposes it. Leaving the member
    // 0 makes l2_shared_cache_size() fall back to l2_cache_size_.
#if defined(__APPLE__)
    {
        int64_t l2_perf = 0;
        size_t len = sizeof(l2_perf);
        if (sysctlbyname("hw.perflevel0.l2cachesize", &l2_perf, &len, nullptr, 0) == 0 &&
            l2_perf > 0) {
            l2_shared_cache_size_ = static_cast<size_t>(l2_perf);
        }
    }
#endif

#endif  // NNOPS_ARCH_AARCH64

    fast_cores_ = detect_fast_cores();
}

} // namespace simd
} // namespace nnops
