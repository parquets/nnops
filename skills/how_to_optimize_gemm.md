# How to Optimize GEMM — Design Notes for the Tiled MatMul Backend

A record of the optimization decisions, constraints, and trade-offs that shaped the
tiled GEMM implementation in [src/backend/cpu/matmul.cpp](../src/backend/cpu/matmul.cpp)
and the shared pack/MMA helpers in [src/backend/cpu/matmul_helper.h](../src/backend/cpu/matmul_helper.h).

---

## 1. High-Level Architecture

### 1.1 Loop Order: NKM or MKN, fixed by the split direction

There is no routing table. `get_matmul_plan` splits the GEMM on its **larger
dimension** (`split_n = (N > M)`) and the block-level loop order follows the
split, mirroring onnxruntime's MLAS `sgemm.cpp`. B is *always* packed either way.

```text
N-split (N > M):  NKM        M-split (M >= N):  MKN
for n-block:                 for m-block:
  for k in [0..K, step Kc]:    for k in [0..K, step Kc]:
    for m in [0..M, step Mc]:    for n in [0..N, step Nc]:
```

**Why NKM on the N-split?** B is packed once per Kc×Nc block in the K-loop and
reused across all M panels. Putting N outermost means packed_B stays live in L2
across the entire M sweep, maximizing reuse of the packed RHS.

**Why an M-split at all?** The split dimension is the parallel dimension
(`ctx.cpu.run` walks the blocks). Splitting on the larger dimension is what gives
a small-M, large-N GEMM enough blocks to fill the pool. The cost is that each
m-block needs its **own** full-N packed-B slice, since the blocks run
concurrently and each packs its B per k-block — see §7 for how the scratch is
sized against that.

### 1.2 Packing Strategy — always-pack-B, threshold-pack-A

The decision is two trivial rules (see `PACK_A_STRIDE_THRESHOLD` and
`gemm_is_small` in matmul_helper.h, applied in `get_matmul_plan`):

```text
pack_b = true                                             // rhs always packed
pack_a = is_i8 || transpose_a || (lda > PACK_A_STRIDE_THRESHOLD)   // 1024 elements
```

Rules, in order:

1. **B is always packed.** The packed RHS is what the micro-kernel consumes
   (`tile_mma_pack(..., -1)` / `tile_mma_direct(..., -1)`); packing it is the
   RHS-reuse win and is never skipped.
2. **A packs on correctness, dtype or wide stride.** int8 has no direct-A route
   at all (`tile_pack_lhs_i8` is unconditional); `transpose_a` forces packing (the
   micro-kernel wants A as `[m][k]` panels); otherwise A packs only when its row
   stride exceeds 1024 elements (page-scattered reads). Otherwise A is read
   directly via `tile_mma_direct`.
3. **Fast path (f32 only).** f32 GEMMs with `M*N*K < 1024` MACs skip the tiled
   path entirely and go to the reference. f16 has no reference kernel, so it
   always takes the tiled path.

### 1.3 Workspace Model (TensorRT Pattern)

Following TensorRT's `IPluginV2DynamicExt::getWorkspaceSize`:

- `TensorDesc` carries metadata only (dims, dtype, layout) — no data pointer.
- `getWorkspaceSize(inputs, outputs)` returns the byte count. The caller allocates.
- **Workspace holds packed B only.** Packed A lives in a per-thread stack buffer
  sized `num_panels_max(MC_TARGET, mr) × ldd_a` (see matmul_helper.h):
  84 KiB on aarch64 / 78 KiB on x86_64 for f32, exactly half for f16. Note this is
  *not* the naive `MC_TARGET × Kc × elem` (72 KiB) — the greedy largest-first
  decomposition of an `mc < MC_TARGET` can need more panels than MC_TARGET's does,
  so the buffer is sized for the worst `mc`, not for `MC_TARGET` itself. Either
  way, every `pack_a`-only / direct path needs **zero** workspace.
- `TensorDesc.row_stride_elems` feeds the **pack decision for A**: a stride wider
  than `PACK_A_STRIDE_THRESHOLD` makes A pack (`0` = unknown → treat as compact,
  `K` elems/row for a non-transposed A, `M` for a transposed one). It does not
  affect the workspace size, which depends only on N, Kc, dtype and the NR panel
  list.
- **Thread-count invariant sizing.** Each block owns `num_panels(nc, nr)` packed-B
  panels at a uniform 64-byte-aligned full-Kc stride, so the packed-B footprint is
  `num_panels(N, nr) × ldd_b × elem` — independent of the thread count. See §7 for
  what that one slice maps onto in each split.

---

## 2. Tiling Parameters

### 2.1 Kc — Fixed Constant

| dtype | Kc  | Rationale |
|-------|-----|-----------|
| f32   | 128 | Fits 2× mr×Kc panels in L1 (6×128×4 = 3KB on x86); good ILP without blowing register pressure. |
| f16   | 128 | Same value as f32 — it bounds the per-k-block fp16 accumulation error, and half the elements make the panels half the size. |
| s8    | 512 | 4× the float value: int8 elements are 1 byte, so the same byte budget buys 4× the K. |

`KC_F16I4 = 256` is declared as a placeholder for a future fp16×int4 path; nothing
reads it.

Kc is also clamped to the actual K (`plan.kc = min(KC_*, K)`), so short reductions
never pad.

### 2.2 Mc — Free Parameter

**Mc = 144 (`MC_TARGET`)** — chosen to be divisible by the largest mr on both
architectures:

- x86_64: 144 / 6 = 24 panels (also clean for mr=4: 36 panels)
- aarch64: 144 / 8 = 18 panels (also clean for mr=4: 36 panels)

Larger Mc amortizes B-packing cost over more M rows, but too large eats L2 capacity
and hurts packed_B residency.

`mc` is clamped to `min(MC_TARGET, M)`, and on the M-split with more than one
thread it becomes `M / num_threads` (still capped at `MC_TARGET`) so the m-blocks
are spread evenly. The A pack stack buffer is sized for the worst `mc <=
MC_TARGET` (see §1.3), not for `MC_TARGET` itself — the greedy panel
decomposition is not monotonic in `mc`, so 144 rows need 18 panels (aarch64) while
143 rows need 21.

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

`compute_nc` solves that inequality and subtracts one for a safety margin:

```text
Nc < (L2_SIZE / (2 × elem_size) - mr × Kc) / (Kc + mr)   - 1
```

With L2 = 256 KB (Haswell baseline) and Kc = 128:

| Arch    | dtype | mr | nr_max | Nc (raw) | Nc (rounded) | Divisor check |
|---------|-------|----|--------|----------|--------------|---------------|
| x86_64  | f32   | 6  | 16     | 237      | **224**      | 224/16 = 14 ✓ |
| x86_64  | f16   | 6  | 16     | 482      | **480**      | 480/16 = 30 ✓ |
| aarch64 | f32   | 8  | 12     | 232      | **228**      | 228/12 = 19 ✓ |
| aarch64 | f16   | 8  | 16     | 473      | **464**      | 464/16 = 29 ✓ |

Nc is rounded down to the nearest multiple of the largest nr for clean panel
decomposition, then clamped into `[nr_max, N]` so the loop always makes progress
and the sizing matches the kernel exactly. At runtime Mc is clamped to the problem
dimensions too (no padding waste). The actual L2 size is read at runtime (§3), so
these figures are the floor, not the values a real CPU will use.

### 2.4 Why Mc ≠ Nc — the AM-GM Proof

For a fixed L2 budget (`L2_BUDGET = mr × Kc + Nc × Kc + mr × Nc`), the
computational intensity of an MMA call is:

```text
CI = mr × Nc × Kc / ((mr + Nc) × Kc + mr × Nc)
   ≈ mr × Nc / (mr + Nc)          [Kc dominates]
```

By AM-GM, the product `mr × Nc` is maximized when `mr = Nc`. But mr is fixed by
the micro-kernel (6 or 8), and Nc is hundreds of columns (224–480 across the
dtype/arch table in §2.3). The formula does
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

### 3.2 aarch64 — OS Queries

AArch64 has no CPUID, so [cpu_features.cpp](../src/detail/cpu_features.cpp) asks
the OS, per platform:

- **Linux:** `getauxval(AT_L2_CACHESIZE)` (Linux 6.6+), else the sysfs file
  `/sys/devices/system/cpu/cpu0/cache/index2/size` (parsing the K/M/G suffix).
- **Windows ARM64:** `GetLogicalProcessorInformation`, first `RelationCache` with
  `Level == 2`.
- **macOS:** `sysctlbyname("hw.l2cachesize")`.

256 KB is the **fallback** when every query fails (or the header says nothing
detects a size), not the aarch64 default. Under-provisioning Nc only costs reuse;
it never over-subscribes the scratch.

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

Any Mc/Nc is decomposed into panels by three consecutive largest-first loops,
each step selecting the matching kernel from a dispatch table. In
[matmul_helper.cpp](../src/backend/cpu/matmul_helper.cpp) this is written out
rather than driven by an index (f32 RHS shown; LHS and f16/f8 are identical with
their own tables):

```cpp
int n = 0;
for (; n + NR_F32[0] <= nc; n += NR_F32[0]) {
    pack_fns[0](dst, src, lds, kc, scale);
    dst += ldd;
    src += NR_F32[0] * pack_lds;
}
// ... same body for NR_F32[1], then NR_F32[2] (== 1)
```

`NR_F32[2]` is always 1, so the last loop cleans up any remainder one panel at a
time. This ensures Mc = 144 decomposes cleanly as 24 × mr0 on x86_64 (144/6 = 24)
and 18 × mr0 on aarch64 (144/8 = 18) — no remainder panels needed.

The greedy walk is what makes `num_panels()` the correct sizing primitive, and
why it is not monotonic: see §1.3 and `num_panels_max` in matmul_helper.h.

---

## 5. Micro-Kernel Dispatch

Two dispatch families per dtype:

### 5.1 mma_direct (A unpacked)

```text
C[mr][nr] += A[mr][Kc] × B[Kc][nr]
```

A is row-major with stride `lda`; B follows the same `ldb` convention as
`tile_mma_pack` (`ldb < 0` = packed, `ldb >= 0` = raw row stride). The kernel
broadcasts a single A element per K-step and multiplies it against a vector of B,
accumulating into registers. Used when A does not need packing.

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

### 5.3 Kernel Selection — Dispatch Tables

`matmul_helper.cpp` holds no macro-generated kernel names. Each micro-kernel is a
plain named function template on `zero_mode` in its arch header
(`mma_pack_6x16_f32<zero_mode>`, `pack_trans_n6_f32`, …), and the file selects
between them with `constexpr std::array` **dispatch tables** of function pointers,
one per dtype × (LHS/RHS) × (trans/plain):

```cpp
constexpr std::array<std::array<PackF32Fn, 3>, 2> pack_trans_f32_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {pack_trans_n6_f32, pack_trans_n4_f32, pack_trans_n1_f32},
    ...
```

The tables are indexed by the panel level (`{mr0,…}`, `{mr1,…}`, `{mr2,…}`) in the
largest-first loop of §4.3. `zero_mode` is compile-time, so the runtime flag in
`tile_mma_pack` picks between two pre-instantiated tables rather than branching in
the inner loop.

Because the tables are per-arch, the `#ifdef NNOPS_ARCH_*` blocks appear
**throughout** matmul_helper.cpp (one per table initializer), not only at the top
of the file.

---

## 6. Epilogue Handling

| Epilogue Type | Implementation |
|---------------|---------------|
| None | Clamp to `(-inf, +inf)` — the `min_clip`/`max_clip` defaults, folded into the MMA kernel (no-op clamp). |
| Relu | `min_clip = 0` — folded into the MMA kernel. |
| Relu6 | `min_clip = 0`, `max_clip = 6` — folded into the MMA kernel. |
| Gelu / Sigmoid / Tanh / Silu / HardSwish / LeakyRelu / Elu | Applied in place per tile by `epilogue_inplace`, once that tile's last k-block has completed. |

`matmul_cpu` folds the two pure-clamp activations into `min_clip`/`max_clip`
before dispatching, so the kernel never branches per element for them.
`kblock_clamp` returns `±inf` for every k-block except the last — intermediate
partial sums must accumulate unclamped, and only the finished tile is clamped.

`has_inplace_epilogue` skips the in-place pass for `None` and `Relu` (both already
handled by the clamp). `Relu6` is folded *and* still passes through the in-place
pass, which clamps it a second time; the clamp is idempotent, so the result is
unchanged — the redundancy is only a minor wasted pass.

Everything else runs through `apply_epilogue` / `apply_epilogue_vec` in
[simd_epilogue.hpp](../src/backend/cpu/simd_kernel/simd_epilogue.hpp). This is not
a second pass over C: it happens inside the same tile loop, per `(m, n)` tile, at
`last_k`, so the values are still hot.

Beta scaling (`C = A×B + beta×C`) is fused too: `tile_scale` applies `beta` at
`k == 0` for `beta ∉ {0, 1}`, and `zero_mode` lets the first k-block skip the C
load entirely when `beta == 0`.

The int8 path is separate: it folds only `Relu` (as a clamp on the requantized
output) and computes the raw int8 reductions needed for the zero-point
compensation (`compute_int8_reductions`).

---

## 7. Multithreading

The tiled kernel is parallelized through `ctx.cpu.run` at **tile-block**
granularity (not bare rows). The split dimension is the larger one, so the block
count is `ceil(N/nc)` on the N-split and `split_block_count(M, mc) = ceil(M/mc)` on
the M-split.

- **Split dimension** is the larger dimension: N when `N > M`, else M. The M-split
  exists precisely so a small-M / large-N GEMM still fills the pool.
- **No barriers, no reductions.** Each block writes a disjoint C tile; the k-loop is
  independent per thread. The result is **bit-identical to serial** — the thread
  count does not change any output value (asserted by `matmul_threaded_matches_serial`).
- **Nc shrinks with the thread count.** On the N-split with more than one thread,
  `nc = min(N / num_threads, nc_cap)`. Without that, the L2-derived `nc`
  (224/228 for f32) would leave an N = 1024 GEMM with only `ceil(1024/224) = 5`
  blocks regardless of pool size; the division brings it to 8 blocks at 8 threads.
- **Workspace slicing.** The packed-B footprint is always
  `num_panels(N, nr) × ldd_b × elem` (see §1.3), but what that one slice maps onto
  differs by split:
  - *N-split:* the n-blocks tile the single full-N slice. Block `blk` writes at
    `workspace + blk × num_panels(nc,nr) × ldd_b × elem`; the panel ranges are
    disjoint and the last block takes the remainder.
  - *M-split:* every m-block needs its **own** full-N slice, because the blocks run
    concurrently and each packs its B per k-block. When the backend reports worker
    ids, the scratch is instead one slot per worker thread (`plan.num_slots <
    num_blocks`), reused across every block a thread claims — a thread packs one
    slice at a time, so this is safe.
  - *M-split, int8:* the exception — the dispatch hoists the B pack out of the
    parallel region (one pack per k-block, then M is partitioned), so every m-block
    reads the same slice and `num_slots` is 1.
- **Batched matmul** iterates batch elements serially, parallelizing *within* each
  element. Flattening `(batch × block)` is a possible future extension.
- **L2 contention.** With T threads, T hot working sets may exceed L2 and spill to
  L3 — consistent with BLIS/OpenBLAS. Shrinking `nc` per-thread would restore
  residency but break the thread-count-invariant workspace formula; noted as a future
  tuning point.

---

## 8. Key Design Decisions

1. **2D + batched.** Rank-2 runs the fused tiled kernels; rank > 2 runs the same
   kernels per batch element with numpy-style broadcast (shared logic with the
   reference). Fused tiled kernels exist for f32, f16, and s8×s8 → s32/s8
   (`matmul_dispatch_int8`); every other dtype combination falls through to the
   reference kernel.

2. **Multi-threading is now in.** Parallel decomposition is block-granularity and
   bit-identical to serial; see §7.

3. **External workspace allocation.** The kernel never allocates internally.
   `getWorkspaceSize()` lets the runtime pre-allocate once and reuse the buffer
   across operator invocations.

4. **Architecture constants: shared for float, restated by hand for int8.** The
   float panel sizes are single-sourced: `matmul_helper.h` builds `MR_F32`/`NR_F32`/
   `MR_F16`/`NR_F16` out of `arch::mr_f32` / `nr_f32` / `mr_f16` / `nr_f16`, which
   each arch header defines exactly once. int8 has no `arch::` array, so `MR_I8` /
   `NR_I8` are literals in an `#ifdef` block in `matmul_helper.h` that must stay in
   sync with the hand-written `mma_pack_<mr>x<nr>_s8s8_*` kernels (`{6,4,1}`×
   `{16,8,4,1}` on x86_64, `{8,4,1}`×`{12,8,4,1}` on aarch64). The panel lists are
   also restated in prose in each pack/MMA file's header comment — they are ISA
   facts, so each file stays readable without cross-referencing.

5. **Runtime L2 detection, not compile-time.** Hardcoding L2 size to a lowest-common
   denominator wastes cache on larger CPUs. The CPUID path adds ~10 lines of code
   and pays for itself on any CPU with > 256 KB L2.

6. **Mc = 144 is arbitrary but portable.** It divides cleanly by both 6 (x86_64
   mr0) and 8 (aarch64 mr0), and it is large enough to amortize B-packing overhead.
   It is also the knob that fixes the A-pack stack buffer: the buffer is sized
   `pack_a_stack_elems() = num_panels_max(MC_TARGET, mr) × ldd_a` — worst-case panel
   counts 26 (x86_64) / 21 (aarch64), not the 24 / 18 that MC_TARGET itself
   decomposes into — giving 78 KB / 84 KB for f32 and exactly half that for f16.
   Growing Mc grows that buffer linearly, so it is not free. Tuning Mc
   per-architecture or per-problem-size could squeeze out more performance but adds
   complexity for diminishing returns.

---

## 9. References

- onnxruntime: `cpuid_info.h` / `cpuid_info.cc` — CPUID singleton pattern
- TensorRT: `IPluginV2DynamicExt::getWorkspaceSize` — workspace sizing API
- nn_compute: `mma_direct_f32.hpp`, `pack_f32.hpp` — micro-kernel and pack design
- ComputeLibrary: GEMM assembly kernels — mr/nr panel size selection
- GotoBLAS: the original tiled GEMM loop design and packing model
