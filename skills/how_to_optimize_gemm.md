# How to Optimize GEMM — Design Notes for the Tiled MatMul Backend

A record of the optimization decisions, constraints, and trade-offs that shaped the
tiled GEMM implementation in [src/backend/cpu/matmul.cpp](../src/backend/cpu/matmul.cpp).

---

## 1. High-Level Architecture

### 1.1 Loop Order: NKM

```text
for n in [0..N, step Nc]:         # N outer
  for k in [0..K, step Kc]:       # K middle
    for m in [0..M, step Mc]:     # M inner
```

**Why NKM?** B is packed once per Kc×Nc block in the K-loop and reused across all
M panels. Putting N outermost means packed_B stays live in L2 across the entire M
sweep, maximizing reuse of the packed RHS.

Contrast with KMN (A-packed reuse) or MNK (many C reloads). NKM gives the best
L2 residency for the tiled data under the constraint that B is always packed.

### 1.2 Packing Strategy

- **B (RHS) — Always packed.** Packing transposes or copies B into a contiguous
  Kc×Nc layout consumed by the MMA micro-kernel.
- **A (LHS) — Conditional.** Skipped only when `sizeof(elem) × lda < 4096` **and**
  `!transpose_a`. When the row stride fits in one page, the TLB and L1 cache handle
  the strided access cheaply enough; packing adds unnecessary overhead.

When A is packed, the layout switches from row-major to **interleaved**:
`A[m + k × mr]` — mr elements from each k-step, so the MMA kernel loads contiguous
vectors along k.

### 1.3 Workspace Model (TensorRT Pattern)

Following TensorRT's `IPluginV2DynamicExt::getWorkspaceSize`:

- `TensorDesc` carries metadata only (dims, dtype, layout) — no data pointer.
- `getWorkspaceSize(inputs, outputs)` returns the byte count. The caller allocates.
- Workspace layout: `[packed_A (Mc × Kc × elem)] [packed_B (Nc × Kc × elem)]`.

---

## 2. Tiling Parameters

### 2.1 Kc — Fixed Constant

| dtype | Kc   | Rationale |
|-------|------|-----------|
| f32   | 128  | Fits 2× mr×Kc panels in L1 (6×128×4 = 3KB on x86); good ILP without blowing register pressure. |
| f16   | 256  | Double the f32 value — same byte count. Larger Kc amortizes packing overhead for f16. |

### 2.2 Mc — Free Parameter

**Mc = 192** — chosen to be divisible by the largest mr on both architectures:

- x86_64: 192 / 6 = 32 panels (also clean for mr=4: 48 panels)
- aarch64: 192 / 8 = 24 panels (also clean for mr=4: 48 panels)

Larger Mc amortizes B-packing cost over more M rows. But going too large eats L2
capacity and hurts cache residency of packed_B. 192 is a pragmatic balance —
reasonably large for throughput, clean on both arches, and keeps the A pack buffer
at a modest 96 KB (f32, Kc=128).

### 2.3 Nc — L2 Cache Constraint (the key derivation)

#### The Working Set During MMA

During a `MrNcKcMMa` call, the **active** working set in L2 is:

- **A panel:** `mr × Kc` — only the current mr rows (the micro-kernel panel, not all of Mc)
- **B panel:** `Nc × Kc` — the full packed B block
- **C panel:** `mr × Nc` — the mr×Nc slice of C being accumulated

With double buffering (pack next while compute current), multiply by 2:

```text
(mr × Kc + Nc × Kc + mr × Nc) × 2 × elem_size < L2_SIZE
```

**Critical insight:** Only **mr** (micro-kernel panel, e.g. 6 or 8) appears — not
**Mc** (tiled block). The rest of packed A is in L2 but not the hot working set
during a single MMA call.

#### Solving for Nc

```text
Nc < (L2_SIZE / (2 × elem_size) - mr × Kc) / (Kc + mr)
```

With L2 = 256 KB (Haswell baseline):

| Arch    | dtype | mr | Kc  | Nc (max) | Nc (rounded) | Divisor check   |
|---------|-------|----|-----|----------|-------------|-----------------|
| x86_64  | f32   | 6  | 128 | 238.8    | **224**     | 224/16 = 14 ✓   |
| x86_64  | f16   | 6  | 256 | 244.3    | **240**     | 240/16 = 15 ✓   |
| aarch64 | f32   | 8  | 128 | 233.4    | **228**     | 228/12 = 19 ✓   |
| aarch64 | f16   | 8  | 256 | 240.5    | **240**     | 240/16 = 15 ✓   |

Nc is rounded down to the nearest multiple of the largest nr for clean panel
decomposition. At runtime, both Mc and Nc are clamped to the actual problem
dimensions (no padding waste).

### 2.4 Why Mc ≠ Nc — the AM-GM Proof

For a fixed L2 budget (`L2_BUDGET = mr × Kc + Nc × Kc + mr × Nc`), the
computational intensity of an MMA call is:

```text
CI = mr × Nc × Kc / ((mr + Nc) × Kc + mr × Nc)
   ≈ mr × Nc / (mr + Nc)          [Kc dominates]
```

By AM-GM, the product `mr × Nc` is maximized when `mr = Nc`. But mr is fixed by
the micro-kernel (6 or 8), and Nc is much larger (~224–240). The formula does
**not** produce a balanced tile — it produces the tile that minimizes data movement
**given** the architecture's fixed mr. Mc is a separate free parameter entirely,
chosen for panel divisibility and pack amortization.

---

## 3. Runtime L2 Cache Detection

### 3.1 x86_64: CPUID Leaf 4

Intel/AMD CPUs report deterministic cache parameters via CPUID leaf 4.
We iterate sub-leaves looking for a Level-2 cache of type Data (1) or Unified (3):

```text
Size = Ways × Partitions × LineSize × Sets
```

Each field is decoded:
- Line size: `(EBX[11:0] + 1)`
- Partitions: `(EBX[21:12] + 1)`
- Ways: `(EBX[31:22] + 1)`
- Sets: `(ECX + 1)`

### 3.2 aarch64 / Fallback

Default to 256 KB — a safe baseline that covers all target CPUs (Cortex-A72 onwards,
Apple M1+ have ≥ 256 KB L2 per core, but larger real L2 values only help — the
formula safely under-provisions the workspace, never over-subscribes).

---

## 4. Cross-Platform Micro-Kernel Panel Sizes

### 4.1 x86_64 (AVX2 / FMA / F16C)

| dtype | mr (largest→1) | nr (largest→1) | Rationale |
|-------|---------------|---------------|-----------|
| f32   | {6, 4, 1}     | {16, 8, 1}    | 16 YMM regs; mr=6 leaves 10 regs for 5× nr/2 accumulators. nr=16 uses 8 regs for B + accums. |
| f16   | {6, 4, 1}     | {16, 8, 1}    | Same YMM budget; F16C converts on load. |

### 4.2 aarch64 (NEON)

| dtype | mr (largest→1) | nr (largest→1) | Rationale |
|-------|---------------|---------------|-----------|
| f32   | {8, 4, 1}     | {12, 4, 1}    | 32 NEON regs; mr=8 uses 8 for A, nr=12 uses 12 for accumulators. |
| f16   | {8, 4, 1}     | {16, 8, 1}    | More nr regs available with half-width accumulators. |

### 4.3 Largest-First Decomposition

Any Mc/Nc is decomposed into panels by iterating `{mr0, mr1, mr2}` (largest first):

```cpp
// Decompose Mc into mr panels — fill as many mr0 as possible,
// then mr1, then mr2 (always 1) for any remainder.
int m_off = 0;
for (int mi = 0; mi < 3 && m_off < mc_actual; ++mi) {
    int mr = MR[mi];
    while (m_off + mr <= mc_actual) {
        // issue MMA call for mr rows
        m_off += mr;
    }
}
```

This ensures Mc = 192 decomposes cleanly as 32 × mr0 on x86_64 (192/6 = 32)
and 24 × mr0 on aarch64 (192/8 = 24) — no remainder panels needed.

---

## 5. Micro-Kernel Dispatch

Two dispatch families per dtype:

### 5.1 mma_direct (A unpacked)

```text
C[mr][nr] += A[mr][Kc] × B[Kc][nr]
```

A is row-major with stride `lda`; B is packed contiguous. The kernel broadcasts
a single A element per K-step and multiplies it against a vector of B, accumulating
into registers. Used when A's stride is small enough to skip packing.

### 5.2 mma_pack (A packed)

```text
C[mr][nr] += A[mr][Kc] × B[Kc][nr]
```

A is interleaved (`A[m + k × mr]`); B is packed contiguous. Both inputs are
contiguous in the k-dimension, enabling fully vectorized inner loops with no
strided access.

### 5.3 Kernel Name Composition (Macro-Based)

Architecture-specific kernels use `#ifdef` blocks only at the top of
[imatmul.cpp](../src/backend/cpu/imatmul.cpp). The dispatch logic is shared:
macros paste `mr` and `nr` into kernel names (e.g., `mma_pack_6x16_f32`), and
the compiler resolves the correct specialization at compile time.

---

## 6. Epilogue Handling

| Epilogue Type | Implementation |
|---------------|---------------|
| None | Clamp to `(-inf, +inf)` — fused into the MMA kernel (no-op clamp). |
| Relu | Clamp to `[0, +inf)` — fused into the MMA kernel. |
| Gelu / Sigmoid / … | Post-processing scan over the full C matrix after all K-blocks complete. |

Fusing Relu into the MMA kernel avoids a separate pass over C. The clamp
is applied per-element as the accumulation completes, inside the register file.

---

## 7. Key Design Decisions

1. **2D only for v1.** Batched matmul (rank > 2) falls through to the reference
   kernel. Batched tiling adds significant complexity (broadcast semantics, stride
   management across batch dims) and is deferred.

2. **No multi-threading in v1.** The tiled kernel is single-threaded. Parallel
   decomposition across the N or M dimension is a natural next step.

3. **External workspace allocation.** The kernel never allocates internally.
   `getWorkspaceSize()` lets the runtime pre-allocate once and reuse the buffer
   across operator invocations.

4. **Architecture constants duplicated, not shared.** Panel sizes and half-precision
   pointer types appear in both [imatmul.cpp](../src/backend/cpu/imatmul.cpp) and
   [matmul.cpp](../src/backend/cpu/matmul.cpp). This is intentional — they are ISA
   facts, not implementation details. Each file is self-contained and can be read
   without cross-referencing.

5. **Runtime L2 detection, not compile-time.** Hardcoding L2 size to a lowest-common
   denominator wastes cache on larger CPUs. The CPUID path adds ~10 lines of code
   and pays for itself on any CPU with > 256 KB L2.

6. **Mc = 192 is arbitrary but portable.** It divides cleanly by both 6 (x86_64
   mr0) and 8 (aarch64 mr0), keeps workspace reasonable (~96 KB for A pack), and
   amortizes B-packing overhead. Tuning Mc per-architecture or per-problem-size
   could squeeze out more performance but adds complexity for diminishing returns.

---

## 8. References

- onnxruntime: `cpuid_info.h` / `cpuid_info.cc` — CPUID singleton pattern
- TensorRT: `IPluginV2DynamicExt::getWorkspaceSize` — workspace sizing API
- nn_compute: `mma_direct_f32.hpp`, `pack_f32.hpp` — micro-kernel and pack design
- ComputeLibrary: GEMM assembly kernels — mr/nr panel size selection
- GotoBLAS: the original tiled GEMM loop design and packing model
