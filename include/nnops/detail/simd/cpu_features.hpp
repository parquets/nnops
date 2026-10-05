#pragma once
/// @file cpu_features.hpp
/// @brief Runtime CPU feature detection via CPUID (x86) or OS queries (ARM).
///
/// Pattern adapted from:
///   - onnxruntime: cpuid_info.h / cpuid_info.cc (CPUIDInfo singleton)
///   - OpenCV: simd_intrinsics.hpp (compile-time ISA selection)
///
/// Provides a singleton CpuFeatures that detects available ISA levels once
/// at first access. Used by optimized kernels to select the best code path.

#include <cstddef>
#include <cstdint>

// NNOPS_ARCH_X86_64 / NNOPS_ARCH_AARCH64 are defined by simd.hpp.
// Re-detect here to avoid circular include (simd/simd.hpp → cpu_features.hpp).
#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64)
  #if !defined(NNOPS_ARCH_X86_64)
    #define NNOPS_ARCH_X86_64 1
  #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
  #if !defined(NNOPS_ARCH_AARCH64)
    #define NNOPS_ARCH_AARCH64 1
  #endif
#endif

// F16C (16-bit half-precision float conversions) is baseline on the target
// x86_64 platform. GCC/Clang define __F16C__ with -mf16c, but MSVC's /arch:AVX2
// does not, so define it explicitly to keep the guarded f16 SIMD path active.
#if defined(NNOPS_ARCH_X86_64) && !defined(__F16C__)
  #define __F16C__ 1
#endif

namespace nnops {
namespace simd {

/// Bitmask of supported ISA extensions (can be combined with |).
enum class CpuIsa : uint32_t {
    None       = 0,

    // x86
    SSE4_1     = 1 << 0,   ///< SSE 4.1 (128-bit)
    AVX        = 1 << 1,   ///< AVX (256-bit float ops, no FMA)
    AVX2       = 1 << 2,   ///< AVX2 (256-bit int ops, FMA via FMA3)
    FMA        = 1 << 3,   ///< FMA3 (fused multiply-add)
    AVX512F    = 1 << 4,   ///< AVX-512 Foundation (512-bit)

    // ARM
    NEON       = 1 << 16,  ///< ARM NEON (128-bit ASIMD)
    NEON_FP16  = 1 << 17,  ///< ARM NEON FP16 vector arithmetic
    NEON_DOT   = 1 << 18,  ///< ARMv8.2 dot-product
};

/// Runtime CPU feature detector.
///
/// Usage:
///   if (CpuFeatures::get().has(CpuIsa::AVX2)) { ... use AVX2 kernel ... }
class CpuFeatures {
public:
    /// Get the singleton instance (initialized on first call).
    static const CpuFeatures& get() {
        static CpuFeatures instance;
        return instance;
    }

    /// Check whether a specific ISA extension is supported.
    bool has(CpuIsa isa) const {
        return (features_ & static_cast<uint32_t>(isa)) != 0;
    }

    /// Raw feature bitmask.
    uint32_t raw() const { return features_; }

    /// L1 data cache size in bytes.
    ///
    /// On Apple Silicon `hw.l1dcachesize` reports the *efficiency* cluster's L1D
    /// (64 KiB on an M4) while the performance cluster has 128 KiB. The smaller
    /// figure is the useful one: worker threads land on either cluster, so a
    /// working set sized for the bigger one overflows on the E cores.
    /// Defaults to 32768 (a typical x86 L1D) when no detection path is available.
    size_t l1_cache_size() const { return l1_cache_size_; }

    /// L2 cache size in bytes (per-core). Detected via CPUID on x86_64 and
    /// OS cache queries on AArch64; defaults to 262144 (256KB) when no
    /// detection path is available.
    size_t l2_cache_size() const { return l2_cache_size_; }

    /// L2 in bytes shared by the cores the compute actually lands on.
    ///
    /// On Apple Silicon `hw.l2cachesize` reports the *efficiency* cluster's L2
    /// (4 MiB on an M4) while the performance cluster has 16 MiB, so tiling
    /// heuristics keyed off l2_cache_size() would size themselves against a
    /// cache four times smaller than the one the work runs in. Equals
    /// l2_cache_size() everywhere the two cannot differ.
    size_t l2_shared_cache_size() const {
        return l2_shared_cache_size_ ? l2_shared_cache_size_ : l2_cache_size_;
    }

private:
    CpuFeatures() { detect(); }
    void detect();

    uint32_t features_ = 0;
    size_t l1_cache_size_ = 32768;   // 32KB default (typical x86 L1D)
    size_t l2_cache_size_ = 262144;  // 256KB default (Haswell baseline)
    size_t l2_shared_cache_size_ = 0;  // 0 = no separate value known
};

// ============================================================
// Convenience free-standing check functions (inline, fast)
// ============================================================
inline bool cpu_has_sse4_1()  { return CpuFeatures::get().has(CpuIsa::SSE4_1); }
inline bool cpu_has_avx()     { return CpuFeatures::get().has(CpuIsa::AVX); }
inline bool cpu_has_avx2()    { return CpuFeatures::get().has(CpuIsa::AVX2); }
inline bool cpu_has_fma()     { return CpuFeatures::get().has(CpuIsa::FMA); }
inline bool cpu_has_avx512f() { return CpuFeatures::get().has(CpuIsa::AVX512F); }
inline bool cpu_has_neon()    { return CpuFeatures::get().has(CpuIsa::NEON); }

} // namespace simd
} // namespace nnops
