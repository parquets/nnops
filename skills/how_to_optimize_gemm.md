# How to Optimize GEMM — Design Notes for the Tiled MatMul Backend

A record of the optimization decisions, constraints, and trade-offs that shaped the
tiled GEMM implementation in [src/backend/cpu/matmul.cpp](../src/backend/cpu/matmul.cpp)
and the shared pack/MMA helpers in [src/backend/cpu/matmul_helper.h](../src/backend/cpu/matmul_helper.h).

---

## 1. High-Level Architecture

### 1.1 Loop Order: NKM and MKN (plan-driven)

The loop order is chosen by the static pack decision, not hard-coded:

```text
NKM (default)                       MKN (A-only pack)
for n in [0..N, step Nc]:           for k in [0..K, step Kc]:
  for k in [0..K, step Kc]:           for m in [0..M, step Mc]:
    for m in [0..M, step Mc]:           for n in [0..N, step Nc]:
```

**Why NKM?** B is packed once per Kc×Nc block in the K-loop and reused across all
M panels. Putting N outermost means packed_B stays live in L2 across the entire M
sweep, maximizing reuse of the packed RHS.

**Why MKN?** When *only* A is packed (B stays raw), the A pack is the expensive
operation, so we hoist it: A is packed once per Kc×Mc block and reused across the
entire N sweep. Packing B in MKN order would re-pack B `M/mc` times, so MKN is
chosen *only* when `pack_a && !pack_b`.

**Routing table** (see `make_pack_plan<T>` in matmul_helper.h):

| pack_a | pack_b | loop order | mma call                          | split dim |
|--------|--------|------------|-----------------------------------|-----------|
| ✓      | ✓      | NKM (fused)| `tile_mma_pack(..., -1)`          | N |
| ✓      | ✗      | **MKN**    | `tile_mma_pack(..., b_raw, ldb)`  | M |
| ✗      | ✓      | NKM (fused)| `tile_mma_direct(..., pack_b, -1)`| N |
| ✗      | ✗      | NKM (direct)| `tile_mma_direct(..., b_raw, ldb)`| N |

### 1.2 Packing Strategy — static cost model (`PackPlan`)

The decision lives in one place: `make_pack_plan<T>(attrs, M_s, N_s, lda, ldb)`.

```text
pack_a = transpose_a || (N_s > nc && lda > thr)
pack_b = transpose_b || (M_s > mc && ldb > thr)
mkn_order = pack_a && !pack_b
thr = 4096 / sizeof(T)   // page-size heuristic: f32 → 1024, f16 → 2048
```

Rules, in order:

1. **Correctness.** `transpose_a ⇒ pack_a`, `transpose_b ⇒ pack_b` — the
   micro-kernel requires A as `[m][k]` and B as `[k][nr]` panels.
2. **Stride + reuse.** Pack only when the row stride is page-scattered (`lda/ldb > thr`)
   **and** the panel is actually re-read across tile blocks (A is re-read `N_s/nc`
   times, B `M_s/mc` times). A single-pass read is never worth the extra copy.
3. **Loop order.** MKN only when A is packed and B is raw.

The **global vs per-slab** split matters under threading: `pack_b` is decided
globally (workspace is a global resource; sizing can't know the thread count), while
`pack_a` can be refined per-slab (the stack buffer is per-thread): under an N-split,
each thread reads A exactly once, so stride-triggered `pack_a` collapses to
`transpose_a`.

### 1.3 Workspace Model (TensorRT Pattern)

Following TensorRT's `IPluginV2DynamicExt::getWorkspaceSize`:

- `TensorDesc` carries metadata only (dims, dtype, layout) — no data pointer.
- `getWorkspaceSize(inputs, outputs)` returns the byte count. The caller allocates.
- **Workspace holds packed B only.** Packed A lives in a 72 KB per-thread stack
  buffer (`MC_TARGET × Kc × elem`: f32 144×128×4, f16 144×256×2), so every
  `pack_a`-only / direct path needs **zero** workspace.
- `TensorDesc.row_stride_elems` carries the source view's physical row pitch (in
  elements) through the sizing path, so a wide non-transposed B is sized correctly.
  `0` = unknown → treat as compact (`last_dim` elems/row).
- **Thread-count invariant sizing.** Each n-block owns `num_panels(nc, nr)` packed-B
  panels at a uniform 64-byte-aligned full-Kc stride; the blocks tile the buffer
  exactly, so the total is `num_panels(N, nr) × ldd_b × elem` — independent of both
  `nc` and the thread count.

---

## 2. Tiling Parameters

### 2.1 Kc — Fixed Constant

| dtype | Kc   | Rationale |
|-------|------|-----------|
| f32   | 128  | Fits 2× mr×Kc panels in L1 (6×128×4 = 3KB on x86); good ILP without blowing register pressure. |
| f16   | 256  | Double the f32 value — same byte count. Larger Kc amortizes packing overhead for f16. |

### 2.2 Mc — Free Parameter

**Mc = 144 (`MC_TARGET`)** — chosen to be divisible by the largest mr on both
architectures:

- x86_64: 144 / 6 = 24 panels (also clean for mr=4: 36 panels)
- aarch64: 144 / 8 = 18 panels (also clean for mr=4: 36 panels)

Larger Mc amortizes B-packing cost over more M rows, but too large eats L2 capacity
and hurts packed_B residency. 144 keeps the A pack stack buffer at exactly 72 KB on
both arches (f32: 144×128×4; f16: 144×256×2 — same byte count).

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

This ensures Mc = 144 decomposes cleanly as 24 × mr0 on x86_64 (144/6 = 24)
and 18 × mr0 on aarch64 (144/8 = 18) — no remainder panels needed.

---

## 5. Micro-Kernel Dispatch

Two dispatch families per dtype:

### 5.1 mma_direct (A unpacked)

```text
C[mr][nr] += A[mr][Kc] × B[Kc][nr]
```

A is row-major with stride `lda`; B is packed contiguous (or raw, signalled by
`ldb >= 0`). The kernel broadcasts a single A element per K-step and multiplies it
against a vector of B, accumulating into registers. Used when A's stride is small
enough to skip packing.

### 5.2 mma_pack (A packed)

```text
C[mr][nr] += A[mr][Kc] × B[Kc][nr]
```

A is interleaved (`A[m + k × mr]`); B is packed contiguous (or raw via `ldb`). Both
inputs are contiguous in the k-dimension when B is packed, enabling fully vectorized
inner loops with no strided access.

**The `ldb < 0` convention** describes B for both entry points: `ldb < 0` means B
is packed ([K][nr], panels advance by the aligned stride `ldd_b`); `ldb >= 0` means
B is raw (row stride `ldb`). This single convention lets one fused kernel serve
`pack_b`/raw-B, and `transpose_b`/non-transposed B alike.

### 5.3 Kernel Name Composition (Macro-Based)

Architecture-specific kernels use `#ifdef` blocks only at the top of
[matmul_helper.cpp](../src/backend/cpu/matmul_helper.cpp). The dispatch logic is
shared: macros paste `mr` and `nr` into kernel names (e.g., `mma_pack_6x16_f32`),
and the compiler resolves the correct specialization at compile time.

---

## 6. Epilogue Handling

| Epilogue Type | Implementation |
|---------------|---------------|
| None | Clamp to `(-inf, +inf)` — fused into the MMA kernel (no-op clamp). |
| Relu | Clamp to `[0, +inf)` — fused into the MMA kernel. |
| Gelu / Sigmoid / … | Post-processing scan over the full C matrix after all K-blocks complete. |

Fusing Relu into the MMA kernel avoids a separate pass over C. The clamp
is applied per-element as the accumulation completes, inside the register file.
Beta scaling (`C = A×B + beta×C`) is fused into the first k-block (`tile_scale`).

---

## 7. Multithreading

The tiled kernel is parallelized through `ctx.cpu_parallel_for` at **tile-block**
granularity (not bare rows): N-split produces `ceil(N/nc)` blocks, M-split
`ceil(M/mc)` blocks.

- **Split dimension** follows the routing table: MKN → M-split (A-pack traffic
  stays MK), everything else → N-split (BLIS convention).
- **No barriers, no reductions.** Each block writes a disjoint C tile; the k-loop is
  independent per thread. The result is **bit-identical to serial** — the thread
  count does not change any output value (asserted by `matmul_threaded_matches_serial`).
- **Workspace slicing.** Each n-block's packed-B panels start at
  `workspace + blk × num_panels(nc,nr) × ldd_b × elem`; panel ranges are disjoint,
  so the total stays `num_panels(N,nr)` regardless of thread count.
- **Per-slab pack_a refinement.** Under N-split, each thread reads A once, so
  `pack_a` reduces to `transpose_a` (stride-triggered packing is correctly disabled).
- **Batched matmul** iterates batch elements serially, parallelizing *within* each
  element. Flattening `(batch × block)` is a possible future extension.
- **L2 contention.** With T threads, T hot working sets may exceed L2 and spill to
  L3 — consistent with BLIS/OpenBLAS. Shrinking `nc` per-thread would restore
  residency but break the thread-count-invariant workspace formula; noted as a future
  tuning point. (The large L2-derived `nc ≈ 224–240` also means an N-split over
  N ≈ 1024 yields only ~2–4 blocks — the M-split MKN path scales better on such
  shapes.)

---

## 8. Key Design Decisions

1. **2D + batched.** Rank-2 runs the fused tiled kernels; rank > 2 runs the same
   kernels per batch element with numpy-style broadcast (shared logic with the
   reference). Non-f32/f16 dtypes fall through to the reference kernel.

2. **Multi-threading is now in.** Parallel decomposition is block-granularity and
   bit-identical to serial; see §7.

3. **External workspace allocation.** The kernel never allocates internally.
   `getWorkspaceSize()` lets the runtime pre-allocate once and reuse the buffer
   across operator invocations.

4. **Architecture constants duplicated, not shared.** Panel sizes and half-precision
   pointer types appear in both the pack/MMA helpers and the matmul kernels. This is
   intentional — they are ISA facts, not implementation details. Each file is
   self-contained and can be read without cross-referencing.

5. **Runtime L2 detection, not compile-time.** Hardcoding L2 size to a lowest-common
   denominator wastes cache on larger CPUs. The CPUID path adds ~10 lines of code
   and pays for itself on any CPU with > 256 KB L2.

6. **Mc = 144 is arbitrary but portable.** It divides cleanly by both 6 (x86_64
   mr0) and 8 (aarch64 mr0), keeps the A pack stack buffer at exactly 72 KB on both
   arches, and amortizes B-packing overhead. Tuning Mc per-architecture or
   per-problem-size could squeeze out more performance but adds complexity for
   diminishing returns.

---

## 9. References

- onnxruntime: `cpuid_info.h` / `cpuid_info.cc` — CPUID singleton pattern
- TensorRT: `IPluginV2DynamicExt::getWorkspaceSize` — workspace sizing API
- nn_compute: `mma_direct_f32.hpp`, `pack_f32.hpp` — micro-kernel and pack design
- ComputeLibrary: GEMM assembly kernels — mr/nr panel size selection
- GotoBLAS: the original tiled GEMM loop design and packing model
