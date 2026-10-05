/// @file cpu_features.cpp
/// @brief Runtime CPU feature detection implementation.
///
/// x86: uses CPUID instruction (eax=1 for SSE/AVX, eax=7 for AVX2/AVX512).
/// ARM: uses OS-provided APIs (getauxval on Linux, IsProcessorFeaturePresent on Windows).

#include "nnops/detail/simd/cpu_features.hpp"

#if defined(NNOPS_ARCH_X86_64)

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

namespace {

// Execute CPUID with eax=leaf, ecx=subleaf. Returns true if supported.
// We check the OS has enabled the feature via xgetbv where required.
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

// Check that the OS has enabled AVX state via XCR0.
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
#elif defined(_WIN32)
#include <Windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#endif  // NNOPS_ARCH_AARCH64

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

    // ---- L2 cache size via CPUID leaf 4 (deterministic cache parameters) ----
    for (int subleaf = 0; ; ++subleaf) {
    #if defined(_MSC_VER)
        int info[4] = {0};
        __cpuidex(info, 4, subleaf);
    #else
        unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
        __cpuid_count(4, subleaf, eax, ebx, ecx, edx);
        int info[4] = {static_cast<int>(eax), static_cast<int>(ebx),
                       static_cast<int>(ecx), static_cast<int>(edx)};
    #endif
        int cache_type = info[0] & 0x1F;
        if (cache_type == 0) { break; }  // no more caches

        int cache_level = (info[0] >> 5) & 0x7;
        // type 1 = data cache, type 3 = unified cache
        if (cache_level == 2 && (cache_type == 1 || cache_type == 3)) {
            int line_size  = (info[1] & 0xFFF) + 1;
            int partitions = ((info[1] >> 12) & 0x3FF) + 1;
            int ways       = ((info[1] >> 22) & 0x3FF) + 1;
            int sets       = info[2] + 1;
            l2_cache_size_ = static_cast<size_t>(ways) * partitions * line_size * sets;
            break;
        }
    }

#elif defined(NNOPS_ARCH_AARCH64)
    // NEON (ASIMD) is mandatory on AArch64
    features_ |= static_cast<uint32_t>(CpuIsa::NEON);

#if defined(__linux__)
    long hwcap = getauxval(AT_HWCAP);
    long hwcap2 = getauxval(AT_HWCAP2);

    if (hwcap & HWCAP_ASIMDDP) {
        features_ |= static_cast<uint32_t>(CpuIsa::NEON_DOT);
    }
    // FP16 vector arithmetic
    #ifdef HWCAP_FPHP
    if (hwcap & HWCAP_FPHP) {
        features_ |= static_cast<uint32_t>(CpuIsa::NEON_FP16);
    }
    #endif

#elif defined(_WIN32)
    // ARM64 Windows: NEON (ASIMD) is always available.
    // Dot-product: IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE),
    // which is the value 43 in winnt.h. Spelled numerically so this compiles
    // against SDKs that predate the enumerator.
    if (IsProcessorFeaturePresent(43)) {
        features_ |= static_cast<uint32_t>(CpuIsa::NEON_DOT);
    }
#endif

    // ---- L2 cache size detection ----
#if defined(__linux__)
    {
        // Try getauxval(AT_L2_CACHESIZE) first (available since Linux 6.6)
        #ifndef AT_L2_CACHESIZE
        #define AT_L2_CACHESIZE 43
        #endif
        long aux_l2 = getauxval(AT_L2_CACHESIZE);
        if (aux_l2 > 0) {
            l2_cache_size_ = static_cast<size_t>(aux_l2);
        } else {
            // Fallback: read Linux sysfs for per-CPU L2 cache
            std::FILE* fp = std::fopen(
                "/sys/devices/system/cpu/cpu0/cache/index2/size", "r");
            if (fp) {
                char buf[32] = {};
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
                        l2_cache_size_ = static_cast<size_t>(val);
                    }
                }
                std::fclose(fp);
            }
        }
    }
#elif defined(_WIN32)
    {
        // Windows ARM64: use GetLogicalProcessorInformation
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION buf[256];
        DWORD len = sizeof(buf);
        if (GetLogicalProcessorInformation(buf, &len)) {
            DWORD count = len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
            for (DWORD i = 0; i < count; ++i) {
                if (buf[i].Relationship == RelationCache &&
                    buf[i].Cache.Level == 2) {
                    l2_cache_size_ = buf[i].Cache.Size;
                    break;
                }
            }
        }
    }
#elif defined(__APPLE__)
    {
        // macOS Apple Silicon: use sysctl
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

    // hw.l2cachesize is the *efficiency* cluster's L2 on a hybrid Mac. Ask for
    // the performance cluster's explicitly when the OS exposes it; leaving the
    // member 0 makes l2_shared_cache_size() fall back to l2_cache_size_.
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
}

} // namespace simd
} // namespace nnops
