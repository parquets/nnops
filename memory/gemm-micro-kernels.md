---
name: gemm-micro-kernels
description: "GEMM micro-kernel infrastructure — transpose, pack, and MMA primitives for aarch64 and x86_64 with f32/f16 support"
metadata: 
  node_type: memory
  type: project
  modified: 2026-07-22T16:20:52.348Z
  originSessionId: 5c3ed336-1c64-4cbc-aea8-a232e6df9151
---

# GEMM Micro-Kernels

Low-level CPU GEMM building blocks ported from nn_compute. Four file types per ISA: transpose (register-level), pack (data layout), mma_pack (inner accumulation from packed data), mma_direct (inner accumulation from strided row-major data — zero-copy path). Two ISAs supported: AArch64 NEON and x86_64 AVX2+FMA/F16C.

**Why:** Optimized GEMM is the backbone of MatMul, Linear, and Conv (im2col+GEMM). These micro-kernels provide the primitive operations that a future tiling dispatch will call. Direct raw intrinsics are used (not the SIMD abstraction layer) because these kernels are inherently ISA-specific and hand-tuned per tile size.

**How to apply:** Include the appropriate architecture header in dispatch code guarded by `#ifdef __x86_64__` / `#ifdef __aarch64__`. Select panel sizes from `mr_<dtype>[m_idx]` / `nr_<dtype>[n_idx]` constants. Call pack to repack A into LHS panels and B into RHS panels, then call mma micro-kernels in a tiled loop.

## File Layout

```
src/backend/cpu/common/
└── restrict.hpp         — cross-platform NNOPS_RESTRICT macro

include/nnops/detail/
└── half.hpp             — half type (uint16_t) + f16↔f32 conversion (shared by SIMD and GEMM)

src/backend/cpu/x86_64/
├── transpose.hpp        — __m128 (4×f32), __m256 (8×f32), __m128i (8×i16) transposes
├── pack_f32.hpp         — LHS/RHS f32 pack kernels (K step=8 via __m256)
├── pack_f16.hpp         — LHS/RHS f16 pack kernels (F16C intrinsics)
├── mma_pack_f32.hpp     — 9 f32 MMA pack kernels (1/4/6 × 1/8/16), AVX2+FMA
├── mma_pack_f16.hpp     — 9 f16 MMA pack kernels (1/4/6 × 1/8/16), F16C+FMA, fp32 accum
├── mma_direct_f32.hpp   — 9 f32 MMA direct kernels (1/4/6 × 1/8/16), AVX2+FMA
└── mma_direct_f16.hpp   — 9 f16 MMA direct kernels (1/4/6 × 1/8/16), F16C+FMA, fp32 accum

src/backend/cpu/aarch64/
├── transpose.hpp        — float32x4_t + int32x4_t + float16x4_t/float16x8_t transposes
├── pack_f32.hpp         — LHS/RHS f32 pack kernels (K step=4 via float32x4_t)
├── pack_f16.hpp         — LHS/RHS f16 pack kernels (native NEON fp16, K step=8 via float16x8_t)
├── mma_pack_f32.hpp     — 9 f32 MMA pack kernels (1/4/8 × 1/4/12), NEON FMA
├── mma_pack_f16.hpp     — 9 f16 MMA pack kernels (1/4/8 × 1/8/24), native NEON fp16 FMA
├── mma_direct_f32.hpp   — 9 f32 MMA direct kernels (1/4/8 × 1/4/12), NEON FMA
└── mma_direct_f16.hpp   — 9 f16 MMA direct kernels (1/4/8 × 1/8/24), native NEON fp16 FMA
```

## Three-Layer Architecture

### 1. Transpose (register-level)

Register-to-register matrix transpose primitives. Called by pack to reorder loaded vectors.

**AArch64:** Uses `vzipq_f32` (interleave) + `vcombine_f32` (extract low/high halves). f16 transposes use `vreinterpretq_f32_f16` to reuse f32 zip-combine logic.

**x86_64:** Uses `_mm_unpacklo_ps`/`_mm_unpackhi_ps` + `_mm_shuffle_ps` for __m128; `_mm256_unpacklo_ps`/`_mm256_unpackhi_ps` + `_mm256_permute2f128_ps` for cross-lane shuffle in __m256; `_mm_unpacklo_epi16` + `_mm_castsi128_ps`/`_mm_castps_si128` reinterpret tricks for __m128i i16.

| Width (f32) | AArch64 | x86_64 __m128 | x86_64 __m256 |
|---|---|---|---|
| 2-way | — | `transpose_2x4` | `transpose_2x8` |
| 4-way | `transpose_4x4_f32` | `transpose_4x4` | `transpose_4x8` |
| 6-way | `transpose_6x4_f32` | `transpose_6x4` | `transpose_6x8` |
| 8-way | `transpose_8x4_f32` | `transpose_8x4` | `transpose_8x8` |
| 12-way | `transpose_12x4_f32` | `transpose_12x4` | `transpose_12x8` |
| 16-way | `transpose_16x4_f32` | `transpose_16x4` | `transpose_16x8` |

f16 (AArch64): 2x4, 2x8, 4x4, 4x8, 8x8, 12x8, 16x8, 24x8
f16 (x86_64 __m128i): 4x8, 6x8, 8x8, 12x8, 16x8

### 2. Pack (data layout)

Converts matrix data into micro-kernel-friendly layouts. Two transform types:

- **Transpose pack (LHS/A matrix):** `pack_trans_nN_<dtype>(output, input, ir_step, K, scale)` — reads N rows from input (stride=ir_step), transposes N×K_step blocks, scales, writes contiguously. Output layout: N rows each of K elements, interleaved.
- **Copy pack (RHS/B matrix):** `pack_copy_nN_<dtype>(output, input, ir_step, K, scale)` — reads N-element contiguous rows (stride=ir_step), scales, writes contiguously. Output layout: K blocks each of N elements.

K step differs by ISA: x86_64 processes 8 K elements per iteration (__m256 width), aarch64 processes 4 (f32, float32x4_t) or 8 (f16, float16x8_t). Scalar tail handles remaining K < K_step.

### 3. MMA (matrix micro-accumulate)

The innermost GEMM loop: `C[mr][nr] += A[mr][K] × B_packed[nr][K]` with post-accumulation clamp to [clamp_min, clamp_max].

**Tile sizes (panel dimensions):**

| Arch | dtype | mr values (M) | nr values (N) | Total kernels |
|---|---|---|---|---|
| AArch64 | f32 | 8, 4, 1 | 12, 4, 1 | 9 (3×3) |
| AArch64 | f16 | 8, 4, 1 | 24, 8, 1 | 9 (3×3) |
| x86_64 | f32 | 6, 4, 1 | 16, 8, 1 | 9 (3×3) |
| x86_64 | f16 | 6, 4, 1 | 16, 8, 1 | 9 (3×3) |

**Naming convention:** `mma_pack_<mr>x<nr>_<dtype>`

**Inner loop pattern (f32):** Broadcast A element to all lanes → load packed B row → FMA into accumulator → advance A by mr, B by ldb. After K loop: load existing C, add accumulator, clamp, store.

**FP16 divergence:**
- **x86_64**: `half` type (uint16_t storage). Load via `_mm256_cvtph_ps(_mm_loadu_si128(...))` → fp32; broadcast A via `_mm256_cvtph_ps(_mm_set1_epi16(...))`; accumulate in fp32 FMA; clamp in fp32; store via `_mm256_cvtps_ph(..., _MM_FROUND_TO_NEAREST_INT)` → `_mm_storeu_si128(...)`.
- **AArch64**: Native `float16_t` type. All arithmetic directly in fp16 via `vfmaq_lane_f16` / `vfmaq_laneq_f16` (broadcast from lane of `float16x4_t`/`float16x8_t`). Clamp uses `vminq_f16`/`vmaxq_f16`.

### 4. MMA Direct (unpacked, zero-copy path)

Variant of MMA that reads A and B directly from their original row-major layouts (A with stride `lda`, B with stride `ldb`) — no pre-packing step. This is useful for small matrices where pack overhead would dominate, or for edge panels that don't fill a full micro-tile.

**Signature:** `mma_direct_<mr>x<nr>_<dtype>(C, ldc, A, lda, B, ldb, K, clamp_min, clamp_max)`

Compare with mma_pack: `mma_pack_<mr>x<nr>_<dtype>(C, ldc, A_packed, B_packed, ldb, K, clamp_min, clamp_max)` — no `lda` parameter because A is already packed contiguously.

**K unrolling strategy differs by ISA:**

| Arch | K unroll | How |
|---|---|---|
| AArch64 f32 | 4 | Load 4 A values as `float32x4_t` → broadcast lanes via `vfmaq_laneq_f32` |
| AArch64 f16 | 4 | Load 4 A values as `float16x4_t` → broadcast lanes via `vfmaq_lane_f16` |
| x86_64 f32 | 1 | Broadcast single A via `_mm256_broadcast_ss` per K iteration |
| x86_64 f16 | 1 | Convert single A via `_mm256_cvtph_ps(_mm_set1_epi16(...))` per K iteration |

**Tile sizes:** Same panel dimensions as mma_pack — mr/nr constants are shared.

**Files:**

```
src/backend/cpu/{x86_64,aarch64}/
├── mma_direct_f32.hpp    — 9 f32 direct kernels per arch
└── mma_direct_f16.hpp    — 9 f16 direct kernels per arch
```

**Bugs fixed from reference during porting:**
- AArch64 f16 `mma_direct_4x8_f16`: copy-paste error — `vfmaq_lane_f16(v_c00, ...)` used as accumulator for all rows instead of correct `v_c10`/`v_c20`/`v_c30`
- AArch64 f16 `mma_direct_4x16_f16` and `4x32_f16`: same accumulator aliasing bug
- x86_64 f32 `mma_direct_4x8_f32`: dead `v_b1` load from copy-paste of 16-wide variant
- x86_64 f16 `mma_direct_1x24_f16`: dead `A_ptr1`–`A_ptr3` declarations in 1-row kernel
- x86_64 f16 `mma_direct_6x{1,8,16}_f16`: reference had empty function bodies — fully implemented

## Panel Size Dispatch

Each pack file exports `mr_<dtype>[3]` and `nr_<dtype>[3]` constants. A tiling dispatch iterates over M in mr panels and N in nr panels:

```cpp
// Pseudocode for future tiling dispatch
for (int m = 0; m < M; m += mr[m_idx]) {
    pack_trans_nN_f32(packed_A, A + m * lda, lda, K, scale); // LHS pack
    for (int n = 0; n < N; n += nr[n_idx]) {
        pack_copy_nN_f32(packed_B, B + n, ldb, K, scale); // RHS pack
        mma_pack_NxN_f32(C + m * ldc + n, ldc, packed_A, packed_B, nr, K, clamp_min, clamp_max);
    }
}
```

The constants are in `nnops::backend::cpu::aarch64` or `nnops::backend::cpu::x86_64` namespace, guarded by the same `#ifdef` that the dispatch layer uses.

## Modern C++ Improvements Over Reference

Compared to the nn_compute reference implementation:
- `noexcept` on all functions
- `constexpr` for panel size arrays (compile-time constants)
- `NNOPS_RESTRICT` for all pointer parameters (aliasing hint to compiler)
- Lambda helpers for clamp (`auto clamp8 = [&](__m256& v){ ... };`) and f16 convert-store, reducing code duplication
- Doxygen `@file` / `@brief` documentation blocks
- Bugs fixed from reference: unreachable scalar tails in pack_copy, pointer-offset typos (`output + 0 * 0` → `output + 0 * 4`), out-of-bounds A broadcasts in f16 MMA kernels with mr=4

## Namespaces

- `nnops::backend::cpu::x86_64` — all x86_64 micro-kernels
- `nnops::backend::cpu::aarch64` — all AArch64 micro-kernels
- `nnops::backend::cpu::half` — shared fp16 storage type (used by x86_64 kernels)

## Reference

All kernels ported from `D:\vscode\nn_compute\src\cpu\kernel\` — see [[reference-projects]] for broader context.

See [[project-architecture]] for overall layout, [[simd-infrastructure]] for distinction between raw micro-kernels and the portable SIMD abstraction layer.
