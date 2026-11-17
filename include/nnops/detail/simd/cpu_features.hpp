#pragma once
/// @file cpu_features.hpp
/// @brief Runtime CPU feature, cache-size and P-core detection.
///
/// One singleton, evaluated on first access: CPUID on x86_64, OS queries on
/// AArch64. Pattern adapted from onnxruntime's CPUIDInfo (cpuid_info.h/.cc)
/// and OpenCV's simd_intrinsics.hpp.
///
/// The cache sizes feed the kc/mc bounds, so they must describe the caches the
/// compute runs in. On Apple platforms the OS's system-wide query reports the
/// E-core L1D and L2 rather than the P-cores': l1_cache_size() keeps the smaller
/// E figure because a worker can land on either cluster, while
/// l2_shared_cache_size() surfaces the P-cluster figure that mc/nc tile against.
/// fast_core_count() is the P-core count (Windows EfficiencyClass, macOS
/// hw.perflevel0): the workers that run a task at the same speed. Past it the
/// slower cores make a static split end on the slowest finisher, so 0
/// (homogeneous or unknown) is the case that needs no over-subscription.

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
    NEON_BF16  = 1 << 19,  ///< Armv8.6 bfloat16 (BFMMLA, BFCVT), FEAT_BF16
    NEON_I8MM  = 1 << 20,  ///< Armv8.6 int8 matrix multiply (SMMLA), FEAT_I8MM
    SME        = 1 << 21,  ///< Armv9 Scalable Matrix Extension (MOPA/ZA)
    SME2       = 1 << 22,  ///< SME2 (four ZA tiles, multi-vector MOPA)
};

/// Runtime CPU feature detector (process-wide singleton).
class CpuFeatures {
public:
    /// Get the singleton instance (initialized on first call).
    static const CpuFeatures& get() {
        static CpuFeatures instance;
        return instance;
    }

    bool has(CpuIsa isa) const {
        return (features_ & static_cast<uint32_t>(isa)) != 0;
    }

    uint32_t raw() const { return features_; }

    /// L1 data cache in bytes — the data cache, since this sizes the kc bound.
    /// On Apple platforms this is the E-core figure (see the file header).
    /// Defaults to 32768 when no detection path is available.
    size_t l1_cache_size() const { return l1_cache_size_; }

    /// Per-core L2 in bytes; defaults to 262144 (256 KiB) when undetectable.
    size_t l2_cache_size() const { return l2_cache_size_; }

    /// L2 in bytes shared by the cores the compute lands on — on Apple
    /// platforms the P-cluster's figure, not the E-cluster's. Falls back to
    /// l2_cache_size() where the two cannot differ.
    size_t l2_shared_cache_size() const {
        return l2_shared_cache_size_ ? l2_shared_cache_size_ : l2_cache_size_;
    }

    /// P-core count — the workers that run a task at the same speed; 0 when no
    /// efficiency classes are reported.
    int fast_core_count() const { return fast_cores_; }

    /// SME streaming vector length in bytes, when the platform publishes it.
    /// 0 means unknown: the kernel has to ask the hardware in streaming mode.
    size_t sme_svl_bytes() const { return sme_svl_; }

private:
    CpuFeatures() { detect(); }
    void detect();

    uint32_t features_ = 0;
    size_t l1_cache_size_ = 32768;   // 32KB default (typical x86 L1D)
    size_t l2_cache_size_ = 262144;  // 256KB default (Haswell baseline)
    size_t l2_shared_cache_size_ = 0;  // 0 = no separate value known
    int fast_cores_ = 0;             // 0 = no efficiency classes reported
    size_t sme_svl_ = 0;             // 0 = unknown
};

inline bool cpu_has_sse4_1()  { return CpuFeatures::get().has(CpuIsa::SSE4_1); }
inline bool cpu_has_avx()     { return CpuFeatures::get().has(CpuIsa::AVX); }
inline bool cpu_has_avx2()    { return CpuFeatures::get().has(CpuIsa::AVX2); }
inline bool cpu_has_fma()     { return CpuFeatures::get().has(CpuIsa::FMA); }
inline bool cpu_has_avx512f() { return CpuFeatures::get().has(CpuIsa::AVX512F); }
inline bool cpu_has_neon()    { return CpuFeatures::get().has(CpuIsa::NEON); }
/// FEAT_BF16: the BFMMLA bf16 matrix-multiply instruction is available.
inline bool cpu_has_bf16()    { return CpuFeatures::get().has(CpuIsa::NEON_BF16); }
/// FEAT_I8MM: the SMMLA int8 matrix-multiply instruction is available.
inline bool cpu_has_i8mm()    { return CpuFeatures::get().has(CpuIsa::NEON_I8MM); }
/// FEAT_SME: the Scalable Matrix Extension (ZA state, MOPA) is available.
inline bool cpu_has_sme()     { return CpuFeatures::get().has(CpuIsa::SME); }
/// FEAT_SME2: four ZA tiles and the multi-vector SME2 forms are available.
inline bool cpu_has_sme2()    { return CpuFeatures::get().has(CpuIsa::SME2); }
/// SME streaming vector length in bytes, or 0 when the OS does not say.
inline size_t cpu_sme_svl()   { return CpuFeatures::get().sme_svl_bytes(); }

} // namespace simd
} // namespace nnops
