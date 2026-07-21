---
name: simd-infrastructure
description: "SIMD abstraction layer design — cross-platform vector wrappers for SSE, AVX2, NEON with scalar fallback"
metadata:
  node_type: memory
  type: project
  originSessionId: 2c7fb42c-b26c-4d7e-a6dd-2de30d713ad8
  modified: 2026-07-21T14:21:00.717Z
---

# SIMD Abstraction Layer

Cross-platform SIMD intrinsic wrappers for x86_64 and AArch64, with automatic scalar fallback. Design informed by OpenCV's Universal Intrinsics and onnxruntime's MLAS.

**Why:** Zero-dependency operator library needs a uniform SIMD API across platforms. Raw intrinsics are ISA-specific and unportable. A thin type-safe wrapper allows writing kernels once that compile to SSE, AVX2+FMA, NEON, or scalar C++ depending on the target.

**How to apply:** Include `"nnops/detail/simd/simd.hpp"` in optimized kernel files. Use `v_f32x4` for 128-bit operations and `v_f32x8` for 256-bit operations. Guard AVX2 code paths with `cpu_has_avx2()` at runtime and compile the file with `/arch:AVX2` (MSVC) or `-mavx2 -mfma` (GCC/Clang).

## Relationship to GEMM Micro-Kernels

The SIMD abstraction layer (`include/nnops/detail/simd/`) serves platform-independent operator code (e.g., element-wise ops like Relu, BatchNorm). In contrast, the GEMM micro-kernels in `src/backend/cpu/{x86_64,aarch64}/` use **raw architecture-specific intrinsics directly** (NEON `float32x4_t`/`float16x8_t`, AVX `__m256`, SSE `__m128`) — not the SIMD wrappers. This is intentional:

- Micro-kernels are hand-tuned for a specific ISA and tile size; portability is not a goal
- The dispatch layer selects the right micro-kernel at compile time (`#ifdef __x86_64__` / `#ifdef __aarch64__`)
- The SIMD abstraction layer is for writing portable kernels that compile once per operator; the micro-kernels are per-ISA specializations of the GEMM primitive

The `NNOPS_RESTRICT` macro (`src/backend/cpu/common/restrict.hpp`) is the only shared utility used by both the SIMD abstraction and the raw micro-kernels.

## Naming Convention

- **Types**: `v_f32x4`, `v_f32x8`, `v_f16x8` (lowercase, `fp` prefix for float precision)
- **Overloaded ops** (return type deduced from arg types): `v_add(a,b)`, `v_mul(a,b)`, `v_fmadd(a,b,c)`, `v_min(a,b)`, `v_sqrt(a)`, `v_reduce_sum(a)`, `v_store(p,a)`, `v_cvt_f16_to_f32(a)`, etc.
- **Non-overloaded ops** (need suffix for return type): `v_load_f32x4(p)`, `v_set1_f32x4(s)`, `v_zero_f32x4()`
- **Constants**: `simd_len_f32x4`, `simd_len_f32x8`, `simd_len_f16x8`

## File Layout

```
include/nnops/detail/simd/
├── simd.hpp              — main entry point (includes cpu_features + unified wrappers)
├── cpu_features.hpp      — runtime CPU detection (CpuFeatures singleton, CpuIsa enum)
├── vec_f32x4.hpp         — unified 128-bit float32 vector (selects arch at compile time)
├── vec_f32x8.hpp         — unified 256-bit float32 vector (selects arch at compile time)
├── vec_f16x8.hpp         — unified 128-bit float16 vector (NEON FP16 or scalar)
├── arch/
│   ├── scalar.hpp        — pure C++ fallback (f32 + f16, correctness reference)
│   ├── x86/
│   │   ├── sse.hpp       — SSE4.1 backend (__m128), v_f32x8 emulated with two lanes
│   │   └── avx2.hpp      — AVX2 + FMA3 backend (__m256), includes v_f32x4 via low 128
│   └── arm/
│       └── neon.hpp      — ARM NEON backend (f32 + f16, native FP16 on ARMv8.2+)

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

## API Reference — v_f32x4 (128-bit)

| Operation | SSE4.1 | NEON | Description |
|---|---|---|---|
| `v_load_f32x4(p)` | `_mm_loadu_ps` | `vld1q_f32` | Unaligned load |
| `v_store(p, a)` | `_mm_storeu_ps` | `vst1q_f32` | Unaligned v_store |
| `v_set1_f32x4(s)` | `_mm_set1_ps` | `vdupq_n_f32` | Broadcast scalar |
| `v_zero_f32x4()` | `_mm_setzero_ps` | zero | Zero vector |
| `v_add(a,b)` | `_mm_add_ps` | `vaddq_f32` | Addition |
| `v_sub(a,b)` | `_mm_sub_ps` | `vsubq_f32` | Subtraction |
| `v_mul(a,b)` | `_mm_mul_ps` | `vmulq_f32` | Multiplication |
| `v_div(a,b)` | `_mm_div_ps` | vrecpe NR | Division |
| `v_fmadd(a,b,c)` | `_mm_add_ps(_mm_mul_ps,a)` | `vfmaq_f32` | a*b+c (FMA) |
| `v_min(a,b)` | `_mm_min_ps` | `vminq_f32` | Minimum |
| `v_max(a,b)` | `_mm_max_ps` | `vmaxq_f32` | Maximum |
| `v_abs(a)` | ANDNOT sign bit | `vabsq_f32` | Absolute value |
| `v_neg(a)` | XOR sign bit | `vnegq_f32` | Negation |
| `v_sqrt(a)` | `_mm_sqrt_ps` | vrsqrte NR | Square root |
| `v_rcp(a)` | `_mm_rcp_ps` | vrecpe NR | Reciprocal |
| `v_rsqrt(a)` | `_mm_rsqrt_ps` | vrsqrte NR | Inverse v_sqrt |
| `v_reduce_sum(a)` | hadd+shuffle | vpadd | Horizontal sum |

## API Reference — v_f16x8 (128-bit)

FP16 (half-precision) vectors for ARM NEON. On AArch64 with `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` (ARMv8.2+):
- `v_f16x8` wraps `float16x8_t`
- Native NEON FP16 intrinsics: vaddq_f16, vmulq_f16, vfmaq_f16, vsqrtq_f16, etc.
- On other platforms: scalar fallback using `uint16_t` storage
- `v_fp16x4` has been removed as impractical; use v_f16x8 for all fp16 SIMD work

FP16 conversion (from onnxruntime MLAS technique):
- **f32→f16**: Denorm magic (`((127-15)+(23-10)+1) << 23` float + subtract int) handles subnormals via FP addition hardware. Normal path: single v_add of `(-112<<23) + 0xFFF + mant_odd` for bias+rounding, then `>> 13`.
- **f16→f32**: Magic subtraction (`113 << 23`) renormalizes subnormals via FP hardware. Three-branch logic: normal (bias adjust), subnormal (+1<<23 then -magic), Inf/NaN (extra v_exp adjust).

| Operation | NEON FP16 | Scalar fallback | Description |
|---|---|---|---|
| `v_load_f16x8(p)` | `vld1q_u16` + reinterpret | uint16_t[8] copy | Load from uint16_t* |
| `v_store(p, a)` | `vst1_u16` + reinterpret | copy | Store to uint16_t* |
| `v_set1_f16x8(s)` | `vdupq_n_f16` | f32→f16 broadcast | Set all lanes to float value |
| `v_cvt_f16_to_f32(a)` | `vcvt_f32_f16` | f16→f32 per-element | Widen to f32 |
| `v_cvt_f32_to_f16(a)` | `vcvt_f16_f32` | f32→f16 per-element | Narrow from f32 |
| `v_add/v_sub/v_mul/v_div(a,b)` | vadd/vsub/vmul/vdiv_f16 | f32 arithmetic + convert | Element-wise arithmetic |
| `v_fmadd(a,b,c)` | `vfma_f16(c, a, b)` | a*b+c in f32 → f16 | Fused multiply-v_add |
| `v_min/v_max(a,b)` | vmin/vmax_f16 | f32 v_min/v_max + convert | Element-wise v_min/v_max |
| `v_abs/v_neg(a)` | vabs/vneg_f16 | bitwise sign clear/xor | Sign operations |
| `v_cmplt/v_cmpgt(a,b)` | vclt/vcgt + reinterpret | f32 compare → mask | Comparison (0xFFFF/0x0000) |
| `v_sqrt(a)` | `vsqrt_f16` | f32 v_sqrt + convert | Square root |
| `v_reduce_sum(a)` | `vaddv_f16` | f32 sum | Horizontal sum (returns float) |

## API Reference — v_f32x8 (256-bit)

Same operations as v_f32x4 but for 8 floats. When compiled with `__AVX__`:
- All ops use native `_mm256_*` intrinsics
- `v_fmadd` uses `_mm256_fmadd_ps` (true FMA)
- When AVX is not available, ops decompose to two v_f32x4 calls

## Usage Pattern

```cpp
#include "nnops/detail/simd/simd.hpp"
using namespace nnops::simd;

// Always-available 128-bit path (compiles to SSE or NEON)
void relu_f32(const float* in, float* out, int64_t n) {
    v_f32x4 zero = v_zero_f32x4();
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        v_f32x4 v = v_load_f32x4(in + i);
        v_store(out + i, v_max(v, zero));
    }
    for (; i < n; ++i) out[i] = (in[i] > 0.0f ? in[i] : 0.0f);  // tail
}

// Optional 256-bit AVX2 path (file must be compiled with /arch:AVX2)
void relu_f32_avx2(const float* in, float* out, int64_t n) {
    v_f32x8 zero = v_zero_f32x8();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        v_f32x8 v = v_load_f32x8(in + i);
        v_store(out + i, v_max(v, zero));
    }
}
```

## Reference Patterns

- **OpenCV**: `modules/core/include/opencv2/core/hal/intrin.hpp` + `intrin_sse.hpp` / `intrin_avx.hpp` / `intrin_neon.hpp` / `intrin_cpp.hpp`
- **onnxruntime**: `mlas/lib/mlasi.h` (MLAS_FLOAT32X4 type alias + cross-platform wrappers), `cpuid_info.h` (runtime detection)

See [[reference-projects]] for more on these projects, [[project-architecture]] for overall design.
