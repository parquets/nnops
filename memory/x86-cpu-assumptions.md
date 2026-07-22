---
name: x86-cpu-assumptions
description: "x86_64 target platform baseline ISA assumptions — F16C, SSE4.1, AVX2, FMA3 are guaranteed"
metadata:
  type: project
  originSessionId: f6988556-d913-4271-9503-4b8a944d85a6
  modified: 2026-07-21T16:22:27.327Z
---

# x86_64 Target Platform Assumptions

The x86_64 target platform is assumed to be modern. The following ISA features are guaranteed available and do NOT need runtime detection or optional compile-time guards:

| Feature | Intrinsics | Key Macros |
|---------|-----------|------------|
| SSE 4.1 | `_mm_*` (128-bit) | `__SSE4_1__` |
| AVX | `_mm256_*` (256-bit) | `__AVX__` |
| AVX2 | `_mm256_*` integer ops | `__AVX2__` |
| FMA3 | `_mm_fmadd_*`, `_mm256_fmadd_*` | `__FMA__` |
| F16C | `_mm256_cvtph_ps`, `_mm256_cvtps_ph` | `__F16C__` |

**Why:** Simplifies SIMD code — no need for runtime dispatch or optional code paths for fp16 operations. All target CPUs are Haswell or newer (Intel) / Excavator or newer (AMD), where these features are baseline.

**How to apply:** x86 fp16x8 SIMD code uses `_mm256_cvtph_ps`/`_mm256_cvtps_ph` for convert→compute→convert pattern. Math functions use avx2_mathfunc.hpp. Guard fp16x8 x86 code with `defined(NNOPS_ARCH_X86_64)` in dispatchers (vec_f16x8.hpp); within arch headers, guard with `__F16C__` for self-documentation (always defined on target).

**Compilation:** Requires `/arch:AVX2` (MSVC) or `-mavx2 -mfma -mf16c` (GCC/Clang).
