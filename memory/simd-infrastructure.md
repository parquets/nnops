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

**How to apply:** Include `"nnops/detail/simd/simd.hpp"` in optimized kernel files. Use `v_fp32x4` for 128-bit operations and `v_fp32x8` for 256-bit operations. Guard AVX2 code paths with `cpu_has_avx2()` at runtime and compile the file with `/arch:AVX2` (MSVC) or `-mavx2 -mfma` (GCC/Clang).

## Relationship to GEMM Micro-Kernels

The SIMD abstraction layer (`include/nnops/detail/simd/`) serves platform-independent operator code (e.g., element-wise ops like Relu, BatchNorm). In contrast, the GEMM micro-kernels in `src/backend/cpu/{x86_64,aarch64}/` use **raw architecture-specific intrinsics directly** (NEON `float32x4_t`/`float16x8_t`, AVX `__m256`, SSE `__m128`) — not the SIMD wrappers. This is intentional:

- Micro-kernels are hand-tuned for a specific ISA and tile size; portability is not a goal
- The dispatch layer selects the right micro-kernel at compile time (`#ifdef __x86_64__` / `#ifdef __aarch64__`)
- The SIMD abstraction layer is for writing portable kernels that compile once per operator; the micro-kernels are per-ISA specializations of the GEMM primitive

The `NNOPS_RESTRICT` macro (`src/backend/cpu/common/restrict.hpp`) is the only shared utility used by both the SIMD abstraction and the raw micro-kernels.

## Naming Convention

- **Types**: `v_fp32x4`, `v_fp32x8`, `v_fp16x4`, `v_fp16x8` (lowercase, `fp` prefix for float precision)
- **Overloaded ops** (return type deduced from arg types): `add(a,b)`, `mul(a,b)`, `fmadd(a,b,c)`, `min(a,b)`, `sqrt(a)`, `reduce_sum(a)`, `store(p,a)`, `cvt_f16_to_f32(a)`, etc.
- **Non-overloaded ops** (need suffix for return type): `load_fp32x4(p)`, `set1_fp32x4(s)`, `zero_fp32x4()`
- **Constants**: `simd_len_fp32x4`, `simd_len_fp32x8`, `simd_len_fp16x4`, `simd_len_fp16x8`

## File Layout

```
include/nnops/detail/simd/
├── simd.hpp              — main entry point (includes cpu_features + unified wrappers)
├── cpu_features.hpp      — runtime CPU detection (CpuFeatures singleton, CpuIsa enum)
├── vec_f32x4.hpp         — unified 128-bit float32 vector (selects arch at compile time)
├── vec_f32x8.hpp         — unified 256-bit float32 vector (selects arch at compile time)
├── vec_f16x4.hpp         — unified 64-bit float16 vector (NEON FP16 or scalar)
├── vec_f16x8.hpp         — unified 128-bit float16 vector (NEON FP16 or scalar)
├── arch/
│   ├── scalar.hpp        — pure C++ fallback (f32 + f16, correctness reference)
│   ├── x86/
│   │   ├── sse.hpp       — SSE4.1 backend (__m128), v_fp32x8 emulated with two lanes
│   │   └── avx2.hpp      — AVX2 + FMA3 backend (__m256), includes v_fp32x4 via low 128
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

## API Reference — v_fp32x4 (128-bit)

| Operation | SSE4.1 | NEON | Description |
|---|---|---|---|
| `load_fp32x4(p)` | `_mm_loadu_ps` | `vld1q_f32` | Unaligned load |
| `store(p, a)` | `_mm_storeu_ps` | `vst1q_f32` | Unaligned store |
| `set1_fp32x4(s)` | `_mm_set1_ps` | `vdupq_n_f32` | Broadcast scalar |
| `zero_fp32x4()` | `_mm_setzero_ps` | zero | Zero vector |
| `add(a,b)` | `_mm_add_ps` | `vaddq_f32` | Addition |
| `sub(a,b)` | `_mm_sub_ps` | `vsubq_f32` | Subtraction |
| `mul(a,b)` | `_mm_mul_ps` | `vmulq_f32` | Multiplication |
| `div(a,b)` | `_mm_div_ps` | vrecpe NR | Division |
| `fmadd(a,b,c)` | `_mm_add_ps(_mm_mul_ps,a)` | `vfmaq_f32` | a*b+c (FMA) |
| `min(a,b)` | `_mm_min_ps` | `vminq_f32` | Minimum |
| `max(a,b)` | `_mm_max_ps` | `vmaxq_f32` | Maximum |
| `abs(a)` | ANDNOT sign bit | `vabsq_f32` | Absolute value |
| `neg(a)` | XOR sign bit | `vnegq_f32` | Negation |
| `sqrt(a)` | `_mm_sqrt_ps` | vrsqrte NR | Square root |
| `rcp(a)` | `_mm_rcp_ps` | vrecpe NR | Reciprocal |
| `rsqrt(a)` | `_mm_rsqrt_ps` | vrsqrte NR | Inverse sqrt |
| `reduce_sum(a)` | hadd+shuffle | vpadd | Horizontal sum |

## API Reference — v_fp16x4 (64-bit) and v_fp16x8 (128-bit)

FP16 (half-precision) vectors for ARM NEON. On AArch64 with `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` (ARMv8.2+):
- `v_fp16x4` wraps `float16x4_t`, `v_fp16x8` wraps `float16x8_t`
- Native NEON FP16 intrinsics: vadd_f16, vmulq_f16, vfmaq_f16, vsqrtq_f16, etc.
- On other platforms: scalar fallback using `uint16_t` storage

FP16 conversion (from onnxruntime MLAS technique):
- **f32→f16**: Denorm magic (`((127-15)+(23-10)+1) << 23` float + subtract int) handles subnormals via FP addition hardware. Normal path: single add of `(-112<<23) + 0xFFF + mant_odd` for bias+rounding, then `>> 13`.
- **f16→f32**: Magic subtraction (`113 << 23`) renormalizes subnormals via FP hardware. Three-branch logic: normal (bias adjust), subnormal (+1<<23 then -magic), Inf/NaN (extra exp adjust).

| Operation | NEON FP16 | Scalar fallback | Description |
|---|---|---|---|
| `load_fp16x4(p)` | `vld1_u16` + reinterpret | uint16_t[4] copy | Load from uint16_t* |
| `store(p, a)` | `vst1_u16` + reinterpret | copy | Store to uint16_t* |
| `set1_fp16x4(s)` | `vdup_n_f16` | f32→f16 broadcast | Set all lanes to float value |
| `cvt_f16_to_f32(a)` | `vcvt_f32_f16` | f16→f32 per-element | Widen to f32 |
| `cvt_f32_to_f16(a)` | `vcvt_f16_f32` | f32→f16 per-element | Narrow from f32 |
| `add/sub/mul/div(a,b)` | vadd/vsub/vmul/vdiv_f16 | f32 arithmetic + convert | Element-wise arithmetic |
| `fmadd(a,b,c)` | `vfma_f16(c, a, b)` | a*b+c in f32 → f16 | Fused multiply-add |
| `min/max(a,b)` | vmin/vmax_f16 | f32 min/max + convert | Element-wise min/max |
| `abs/neg(a)` | vabs/vneg_f16 | bitwise sign clear/xor | Sign operations |
| `cmplt/cmpgt(a,b)` | vclt/vcgt + reinterpret | f32 compare → mask | Comparison (0xFFFF/0x0000) |
| `sqrt(a)` | `vsqrt_f16` | f32 sqrt + convert | Square root |
| `reduce_sum(a)` | `vaddv_f16` | f32 sum | Horizontal sum (returns float) |

## API Reference — v_fp32x8 (256-bit)

Same operations as v_fp32x4 but for 8 floats. When compiled with `__AVX__`:
- All ops use native `_mm256_*` intrinsics
- `fmadd` uses `_mm256_fmadd_ps` (true FMA)
- When AVX is not available, ops decompose to two v_fp32x4 calls

## Usage Pattern

```cpp
#include "nnops/detail/simd/simd.hpp"
using namespace nnops::simd;

// Always-available 128-bit path (compiles to SSE or NEON)
void relu_f32(const float* in, float* out, int64_t n) {
    v_fp32x4 zero = zero_fp32x4();
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        v_fp32x4 v = load_fp32x4(in + i);
        store(out + i, max(v, zero));
    }
    for (; i < n; ++i) out[i] = (in[i] > 0.0f ? in[i] : 0.0f);  // tail
}

// Optional 256-bit AVX2 path (file must be compiled with /arch:AVX2)
void relu_f32_avx2(const float* in, float* out, int64_t n) {
    v_fp32x8 zero = zero_fp32x8();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        v_fp32x8 v = load_fp32x8(in + i);
        store(out + i, max(v, zero));
    }
}
```

## Reference Patterns

- **OpenCV**: `modules/core/include/opencv2/core/hal/intrin.hpp` + `intrin_sse.hpp` / `intrin_avx.hpp` / `intrin_neon.hpp` / `intrin_cpp.hpp`
- **onnxruntime**: `mlas/lib/mlasi.h` (MLAS_FLOAT32X4 type alias + cross-platform wrappers), `cpuid_info.h` (runtime detection)

See [[reference-projects]] for more on these projects, [[project-architecture]] for overall design.
