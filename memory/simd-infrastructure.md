---
name: simd-infrastructure
description: "SIMD abstraction layer design — cross-platform vector wrappers for SSE, AVX2, NEON with scalar fallback"
metadata:
  node_type: memory
  type: project
  originSessionId: 2c7fb42c-b26c-4d7e-a6dd-2de30d713ad8
  modified: 2026-07-22T14:15:46.869Z
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

- **Types**: `v_f32x4`, `v_f32x8`, `v_f16x8` (lowercase; `v_` prefix for vector types)
- **Overloaded ops** (return type deduced from arg types): `v_add(a,b)`, `v_mul(a,b)`, `v_fmadd(a,b,c)`, `v_min(a,b)`, `v_sqrt(a)`, `v_reduce_sum(a)`, `v_store(p,a)`, `v_cvt_f16_to_f32(a)`, etc.
- **Generic ops** (overloaded on pointer type, 2026-07-22): `v_load(p)`, `v_set1(p, s)`, `v_zero(p)` dispatch on pointer type → `v_f32x8` or `v_f16x8`. `v_store(p, v)` was already overloaded. Scalar: `sload(p)` → float, `sstore(p, v)` — all in `nnops::simd`.
- **Constants**: `simd_len_f32x4`, `simd_len_f32x8`, `simd_len_f16x8`

## File Layout

```
include/nnops/detail/
├── half.hpp              — IEEE 754 binary16 type + conversion (used by SIMD and GEMM)
└── simd/
    ├── simd.hpp          — main entry point (includes cpu_features + vec types + generic ops)
    ├── cpu_features.hpp  — runtime CPU detection (CpuFeatures singleton, CpuIsa enum)
    ├── vec_f32x4.hpp     — unified 128-bit float32 vector (selects arch at compile time)
    ├── vec_f32x8.hpp     — unified 256-bit float32 vector (selects arch at compile time)
    ├── vec_f16x8.hpp     — unified 128-bit float16 vector (NEON FP16 / x86 F16C+AVX2 / scalar)
    ├── arch/
    │   ├── scalar.hpp    — pure C++ fallback (f32 + f16, correctness reference)
    │   ├── x86/
    │   │   ├── sse.hpp   — SSE4.1 backend (__m128) + x86 fp16x8 (F16C+AVX2, __m128i storage)
    │   │   ├── sse_mathfunc.hpp  — transcendental math on __m128
    │   │   ├── avx2.hpp  — AVX2 + FMA3 backend (__m256)
    │   │   └── avx2_mathfunc.hpp — transcendental math on __m256
    │   └── arm/
    │       └── neon.hpp  — ARM NEON backend (f32 + f16, native FP16 on ARMv8.2+)

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

vec_f16x8.hpp:
  AArch64 + __ARM_FEATURE_FP16_VECTOR_ARITHMETIC  → arch/arm/neon.hpp   (native float16x8_t)
  x86_64 + __F16C__                                → arch/x86/sse.hpp    (F16C+AVX2, __m128i → __m256 → compute → __m128i)
  otherwise                                        → arch/scalar.hpp     (portable C++)
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
| `v_store(p, a)` | `_mm_storeu_ps` | `vst1q_f32` | Unaligned store |
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
| `v_rsqrt(a)` | `_mm_rsqrt_ps` | vrsqrte NR | Inverse sqrt |
| `v_reduce_sum(a)` | hadd+shuffle | vpadd | Horizontal sum |

## API Reference — v_f16x8 (128-bit)

FP16 (half-precision) vectors. `v_fp16x4` has been removed as impractical — use `v_fp16x8` for all fp16 SIMD work.

Three backends:
- **ARM NEON** (AArch64 + `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC`): Native `float16x8_t`, full native FP16 arithmetic via NEON intrinsics.
- **x86 F16C+AVX2** (x86_64 + `__F16C__`): `__m128i` storage (8 × uint16 bits), convert→compute→convert pattern — widen to `__m256` fp32 via `_mm256_cvtph_ps`, compute with AVX/AVX2/FMA3 intrinsics, narrow back via `_mm256_cvtps_ph`. Assumes modern CPU (F16C, SSE4.1, AVX2, FMA3 all available). See [[x86-cpu-assumptions]].
- **Scalar** (everything else): `uint16_t[8]` storage, per-element f16→f32→compute→f16 loop.

FP16 conversion (from onnxruntime MLAS technique):
- **f32→f16**: Denorm magic (`((127-15)+(23-10)+1) << 23` float + subtract int) handles subnormals via FP addition hardware. Normal path: single add of `(-112<<23) + 0xFFF + mant_odd` for bias+rounding, then `>> 13`.
- **f16→f32**: Magic subtraction (`113 << 23`) renormalizes subnormals via FP hardware. Three-branch logic: normal (bias adjust), subnormal (+1<<23 then -magic), Inf/NaN (extra exp adjust).

| Operation | NEON FP16 | x86 (F16C+AVX2) | Scalar fallback | Description |
|---|---|---|---|---|
| `v_load_f16x8(p)` | `vld1q_u16` + reinterpret | `_mm_loadu_si128` | uint16_t[8] copy | Load from uint16_t* |
| `v_store(p, a)` | `vst1_u16` + reinterpret | `_mm_storeu_si128` | copy | Store to uint16_t* |
| `v_set1_f16x8(s)` | `vdupq_n_f16` | `_mm256_set1_ps` + `_mm256_cvtps_ph` | f32→f16 broadcast | Set all lanes to float value |
| `v_cvt_f16_to_f32(a)` | `vcvt_f32_f16` (low) + `vcvt_high_f32_f16` | `_mm256_cvtph_ps` → v_f32x8 | f16→f32 per-element | Widen to f32x8 |
| `v_cvt_f32_to_f16(a)` | `vcvt_f16_f32` + `vcvt_high_f16_f32` | `_mm256_insertf128_ps` + `_mm256_cvtps_ph` | f32→f16 per-element | Narrow from f32x8 |
| `v_add/v_sub/v_mul/v_div` | vadd/vsub/vmul/vdiv_f16 | cvt→`_mm256_add/sub/mul/div_ps`→cvt | f32 arithmetic + convert | Element-wise arithmetic |
| `v_fmadd(a,b,c)` | `vfmaq_f16(c, a, b)` | cvt→`_mm256_fmadd_ps`→cvt | a*b+c in f32 → f16 | Fused multiply-add |
| `v_min/v_max` | vmin/vmax_f16 | cvt→`_mm256_min/max_ps`→cvt | f32 min/max + convert | Element-wise min/max |
| `v_abs/v_neg` | vabs/vneg_f16 | `_mm_andnot/xor_si128` with 0x8000 | bitwise sign clear/xor | Sign operations |
| `v_cmplt/v_cmpgt` | vclt/vcgt + reinterpret | cvt→`_mm256_cmp_ps`→srli→pack | f32 compare → 0xFFFF/0x0000 | Comparison mask |
| `v_sqrt` | `vsqrtq_f16` | cvt→`_mm256_sqrt_ps`→cvt | f32 sqrt + convert | Square root |
| `v_exp/v_log/v_sin/v_cos/v_tan/v_tanh` | `exp_ps` on fp32 (via cvt) | `exp256_ps` etc. from avx2_mathfunc | f32 math + convert | Transcendental math |
| `v_reduce_sum(a)` | `vaddvq_f16` | `_mm256_cvtph_ps` + hadd+shuffle | f32 sum | Horizontal sum (returns float) |

## API Reference — v_f32x8 (256-bit)

Same operations as v_f32x4 but for 8 floats. All ops use `v_` prefix (`v_load_f32x8`, `v_store`, `v_set1_f32x8`, `v_add`, `v_fmadd`, etc.). When compiled with `__AVX__`:
- All ops use native `_mm256_*` intrinsics
- `v_fmadd` uses `_mm256_fmadd_ps` (true FMA)
- When AVX is not available, ops decompose to two v_f32x4 calls

## Generic SIMD + Scalar Helpers (2026-07-22)

All generic helpers live in `nnops::simd` (declared in `simd.hpp`). They dispatch on pointer
type so kernel code is type-generic across fp32 and fp16. The old `simd_utils.hpp` has been
removed — everything is now part of the SIMD layer.

### Vector operations (lane = 8)

| Call | float* | half* |
|---|---|---|
| `v_load(ptr)` | `v_load_f32x8(ptr)` → `v_f32x8` | `reinterpret_cast` → `v_load_f16x8` → `v_f16x8` |
| `v_store(ptr, v)` | `v_store(float*, v_f32x8)` | `v_store(half*, v_f16x8)` → `reinterpret_cast` → backend |
| `v_set1(ptr, s)` | `v_set1_f32x8(s)` → `v_f32x8` | `v_set1_f16x8(s)` → `v_f16x8` |
| `v_zero(ptr)` | `v_zero_f32x8()` → `v_f32x8` | `v_zero_f16x8()` → `v_f16x8` |

### Scalar operations (fp16↔fp32 conversion)

| Call | float* | half* |
|---|---|---|
| `s_load(ptr)` | `return *p` | `half_to_float(*p)` |
| `s_store(ptr, v)` | `*p = v` | `*p = float_to_half(v)` |

### Usage

```cpp
#include "nnops/detail/simd/simd.hpp"
using namespace nnops::simd;

template <typename T>  // float or uint16_t
void kernel(T* input, T* output, float scale) {
    auto vacc = v_zero(input);               // zero-init accumulator
    auto vk   = v_set1(input, scale);        // broadcast scalar
    auto vin  = v_load(input);               // vector load
    vacc = v_fmadd(vin, vk, vacc);
    v_store(output, vacc);

    // Scalar tail
    float s = sload(input + rem);            // always returns float
    sstore(output + rem, s * scale);         // converts back to T
}
```

The public `half.hpp` (fp16 conversion utilities) lives at `include/nnops/detail/half.hpp`
and is used by both the SIMD layer and GEMM micro-kernels.

## Reference Patterns

- **OpenCV**: `modules/core/include/opencv2/core/hal/intrin.hpp` + `intrin_sse.hpp` / `intrin_avx.hpp` / `intrin_neon.hpp` / `intrin_cpp.hpp`
- **onnxruntime**: `mlas/lib/mlasi.h` (MLAS_FLOAT32X4 type alias + cross-platform wrappers), `cpuid_info.h` (runtime detection)

See [[reference-projects]] for more on these projects, [[project-architecture]] for overall design.
