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
#elif defined(_WIN32)
#include <Windows.h>
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
    // ARM64 Windows: NEON is always available.
    // Dot-product: check PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE
    if (IsProcessorFeaturePresent(43)) {  // PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE
        features_ |= static_cast<uint32_t>(CpuIsa::NEON_DOT);
    }
#endif

#endif  // NNOPS_ARCH_AARCH64
}

} // namespace simd
} // namespace nnops
