---
name: simd-infrastructure
description: "SIMD abstraction layer design — cross-platform vector wrappers for SSE, AVX2, NEON with scalar fallback"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2c7fb42c-b26c-4d7e-a6dd-2de30d713ad8
  modified: 2026-07-20T15:20:55.278Z
---

# SIMD Abstraction Layer

Cross-platform SIMD intrinsic wrappers for x86_64 and AArch64, with automatic scalar fallback. Design informed by OpenCV's Universal Intrinsics and onnxruntime's MLAS.

**Why:** Zero-dependency operator library needs a uniform SIMD API across platforms. Raw intrinsics are ISA-specific and unportable. A thin type-safe wrapper allows writing kernels once that compile to SSE, AVX2+FMA, NEON, or scalar C++ depending on the target.

**How to apply:** Include `"nnops/detail/simd/simd.hpp"` in optimized kernel files. Use `VecF32x4` for 128-bit operations and `VecF32x8` for 256-bit operations. Guard AVX2 code paths with `cpu_has_avx2()` at runtime and compile the file with `/arch:AVX2` (MSVC) or `-mavx2 -mfma` (GCC/Clang).

## File Layout

```
include/nnops/detail/simd/
├── simd.hpp              — main entry point (includes cpu_features + unified wrappers)
├── cpu_features.hpp      — runtime CPU detection (CpuFeatures singleton, CpuIsa enum)
├── vec_f32x4.hpp         — unified 128-bit float32 vector (selects arch at compile time)
├── vec_f32x8.hpp         — unified 256-bit float32 vector (selects arch at compile time)
├── arch/
│   ├── scalar.hpp        — pure C++ fallback (always works, correctness reference)
│   ├── x86/
│   │   ├── sse.hpp       — SSE4.1 backend (__m128), VecF32x8 emulated with two lanes
│   │   └── avx2.hpp      — AVX2 + FMA3 backend (__m256), includes VecF32x4 via low 128
│   └── arm/
│       └── neon.hpp      — ARM NEON backend (float32x4_t), VecF32x8 via two lanes

src/detail/
└── cpu_features.cpp      — CPU detection implementation (CPUID / getauxval / Win32 API)
```

## Compile-Time Selection Logic

```
vec_f32x4.hpp:
  x86_64     → arch/x86/sse.hpp    (SSE4.1 always available on x86_64)
  AArch64    → arch/arm/neon.hpp   (NEON always available on AArch64)
  otherwise  → arch/scalar.hpp     (portable C++)

vec_f32x8.hpp:
  x86_64 + __AVX__  → arch/x86/avx2.hpp  (native 256-bit, FMA3)
  x86_64 (no AVX)   → arch/x86/sse.hpp   (emulated with two __m128)
  AArch64            → arch/arm/neon.hpp  (emulated with two float32x4_t)
  otherwise          → arch/scalar.hpp    (portable C++)
```

## Runtime Detection

`CpuFeatures::get()` probes hardware once at first access:

| Method | x86_64 | AArch64 Linux | AArch64 Windows |
|---|---|---|---|
| SSE4.1 | CPUID.1.ECX[19] | N/A | N/A |
| AVX | CPUID.1.ECX[28] + XGETBV XCR0[2:1]==11 | N/A | N/A |
| AVX2 | CPUID.7.EBX[5] + XCR0 check | N/A | N/A |
| FMA | CPUID.1.ECX[12] | N/A | N/A |
| AVX-512F | CPUID.7.EBX[16] + XCR0[7:5] check | N/A | N/A |
| NEON | N/A | always on AArch64 | always on AArch64 |
| NEON DOT | N/A | getauxval(HWCAP_ASIMDDP) | IsProcessorFeaturePresent(43) |

## API Reference — VecF32x4 (128-bit)

| Operation | SSE4.1 | NEON | Description |
|---|---|---|---|
| `vec_load_f32x4(p)` | `_mm_loadu_ps` | `vld1q_f32` | Unaligned load |
| `vec_store_f32x4(p, a)` | `_mm_storeu_ps` | `vst1q_f32` | Unaligned store |
| `vec_set1_f32x4(s)` | `_mm_set1_ps` | `vdupq_n_f32` | Broadcast scalar |
| `vec_zero_f32x4()` | `_mm_setzero_ps` | zero | Zero vector |
| `vec_add_f32x4(a,b)` | `_mm_add_ps` | `vaddq_f32` | Addition |
| `vec_sub_f32x4(a,b)` | `_mm_sub_ps` | `vsubq_f32` | Subtraction |
| `vec_mul_f32x4(a,b)` | `_mm_mul_ps` | `vmulq_f32` | Multiplication |
| `vec_div_f32x4(a,b)` | `_mm_div_ps` | vrecpe NR | Division |
| `vec_fmadd_f32x4(a,b,c)` | `_mm_add_ps(_mm_mul_ps,a)` | `vfmaq_f32` | a*b+c (FMA) |
| `vec_min_f32x4(a,b)` | `_mm_min_ps` | `vminq_f32` | Minimum |
| `vec_max_f32x4(a,b)` | `_mm_max_ps` | `vmaxq_f32` | Maximum |
| `vec_abs_f32x4(a)` | ANDNOT sign bit | `vabsq_f32` | Absolute value |
| `vec_neg_f32x4(a)` | XOR sign bit | `vnegq_f32` | Negation |
| `vec_sqrt_f32x4(a)` | `_mm_sqrt_ps` | vrsqrte NR | Square root |
| `vec_rcp_f32x4(a)` | `_mm_rcp_ps` | vrecpe NR | Reciprocal |
| `vec_rsqrt_f32x4(a)` | `_mm_rsqrt_ps` | vrsqrte NR | Inverse sqrt |
| `vec_reduce_sum_f32x4(a)` | hadd+shuffle | vpadd | Horizontal sum |

## API Reference — VecF32x8 (256-bit)

Same operations as VecF32x4 but for 8 floats. When compiled with `__AVX__`:
- All ops use native `_mm256_*` intrinsics
- `vec_fmadd_f32x8` uses `_mm256_fmadd_ps` (true FMA)
- When AVX is not available, ops decompose to two VecF32x4 calls

## Usage Pattern

```cpp
#include "nnops/detail/simd/simd.hpp"
using namespace nnops::simd;

// Always-available 128-bit path (compiles to SSE or NEON)
void relu_f32(const float* in, float* out, int64_t n) {
    VecF32x4 zero = vec_zero_f32x4();
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        VecF32x4 v = vec_load_f32x4(in + i);
        vec_store_f32x4(out + i, vec_max_f32x4(v, zero));
    }
    for (; i < n; ++i) out[i] = (in[i] > 0.0f ? in[i] : 0.0f);  // tail
}

// Optional 256-bit AVX2 path (file must be compiled with /arch:AVX2)
void relu_f32_avx2(const float* in, float* out, int64_t n) {
    VecF32x8 zero = vec_zero_f32x8();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        VecF32x8 v = vec_load_f32x8(in + i);
        vec_store_f32x8(out + i, vec_max_f32x8(v, zero));
    }
    // tail: fall through to scalar or 128-bit
}
```

## Reference Patterns

- **OpenCV**: `modules/core/include/opencv2/core/hal/intrin.hpp` + `intrin_sse.hpp` / `intrin_avx.hpp` / `intrin_neon.hpp` / `intrin_cpp.hpp`
- **onnxruntime**: `mlas/lib/mlasi.h` (MLAS_FLOAT32X4 type alias + cross-platform wrappers), `cpuid_info.h` (runtime detection)

See [[reference-projects]] for more on these projects, [[project-architecture]] for overall design.
