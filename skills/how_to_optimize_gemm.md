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
- **Workspace holds packed B and, on the packed-A route, packed A.** Packed B is
  hoisted out of the parallel region on an M-split (one slice serves every
  m-block); packed A is per-m-block and takes its own region of the same
  workspace. Both are sized from the plan's own `mc`/`nc`/`kc` (see §2), the
  packed-A region at `num_panels_max(mc, mr) × ldd_a` per block. Note this is
  *not* the naive `mc × Kc × elem` — the greedy largest-first decomposition of a
  short final block can need more panels than the tile's own `mc` does, so the
  region is sized for the worst height in `[1, mc]`, not for `mc` itself. Every
  direct-A path needs **zero** workspace.
- `TensorDesc.row_stride_elems` feeds the **pack decision for A**: a stride wider
  than `PACK_A_STRIDE_THRESHOLD` makes A pack (`0` = unknown → treat as compact,
  `K` elems/row for a non-transposed A, `M` for a transposed one). It does not
  affect the workspace size, which depends only on N, Kc, dtype and the NR panel
  list.
- **Sizing follows the split, and comes from the plan.** The packed-B region is
  `num_slots × np_slice × ldd_b × elem`, every panel at a uniform 64-byte-aligned
  full-Kc stride: on the M-split `np_slice = num_panels(N, nr)` with one slot per
  concurrent m-block, on the N-split `np_slice = num_panels_max(nc, nr)` with one
  slot per n-block (`ceil(N/nc)`). Neither the dispatch nor the workspace test
  recomputes these — `np_slice` is carried in the plan for exactly that reason.
  See §7 for what a slot maps onto in each split.

---

## 2. Tiling Parameters

### 2.1 One rule for Mc, Nc and Kc

`choose_matmul_tile` in
[matmul_helper.h](../src/backend/cpu/matmul_helper.h) picks all three together:
it maximises the tile's arithmetic intensity

```text
AI = 2·mc·nc·kc / (mc·kc + nc·kc + mc·nc)
```

subject to four constraints on the tile itself:

1. **Aggregate L2** — the `num_threads` concurrent blocks' packed A and B fit
   half of the shared L2:
   ```text
   C = l2_shared / (2 · nt · esz)          kc · (mc + nc) ≤ C
   ```
2. **L1 prefetch residency** — the running `mr_max` rows of A plus the current and
   next `nr_max` columns of B fit L1, so the next B panel can be prefetched while
   the current one is consumed:
   ```text
   kc_l1 = snap32( l1 / ((mr_max + 2·nr_max) · esz) )
   ```
3. **Ceiling** — `mc ≤ MC_MAX = 768`, `nc ≤ NC_MAX = 768`.
4. **Panel grids** — `mc` on `MC_ALIGN = 24`, `nc` on `NC_ALIGN = 48`,
   `kc` on `KC_ALIGN = 32`.

Spending the L2 budget fully, the unconstrained optimum is closed-form:

```text
mc = nc = √C ,  kc = √C / 2
```

so the rule is `s = 2·isqrt(C)` for the total `mc + nc`, split evenly, then
snapped to the grids and clamped by the shape and the split:

```text
split_n :  mc = snap24(min(mc_t, M))
           nc = balance_tile(N, nc_t, 48, nt)
else    :  mc = balance_tile(M, mc_t, 24, nt)
           nc = min(nc_t, N)
```

The split side is the one the pool divides, so it is the one balanced; the other
spans whole and is only bounded by the ceiling and the shape. Both branches then
finish `nc` the same way: `round_nc_target` (down to a 48 multiple when the tile
is at least that wide, else to an `nr_max` multiple — that alone is already exact
for the panel grid) and `clamp_nc` into `[nr_max, N]`. `s` is clamped to
`[MC_ALIGN + NC_ALIGN, 2·MC_MAX]`, so even a huge budget starts from a tile the
ceiling can hold.

`balance_tile` is the fifth constraint, and the one that only exists because
blocks are handed out to a pool:

```text
cap_a = snap_down(min(cap, dim), align)      // largest legal tile
n_min = ceil(dim / cap_a)                    // fewest blocks the cap allows
n_blk = ceil(n_min / nt) · nt                // round the count up to the pool
tile  = round_up(ceil(dim / n_blk), align)   // the even share, on the grid
tile  = min(tile, cap_a)
```

It keeps the block count a multiple of the worker count, so no worker ends up
with a double-length share while another idles. At `nt == 1` it short-circuits to
`cap_a` — with one worker there is no balance to strike.

Rounding the count *up* can pull the tile below what the budget allows — 8192 rows
in 24 blocks is 360, not the 504 the L2 budget would give — so balancing trades
some AI for even work distribution. Measured (Release, 7 interleaved paired
rounds, median over cases, win counts), balanced vs the same rule with
`balance_tile` neutered:

| T | ratio | wins |
|---|---|---|
| 1 | 0.975–1.015 | 1–3/7 — flat, the hard gate |
| 2 | 1.327 | 6–7/7 |
| 4 | 1.450 | 5–6/7 |
| 8 | 1.059 | 6/7 (one 256³ case 0.887) |
| 10 | 0.914 | 0–2/7 |

Without it, `mc` at the 768 ceiling gave M = 1024 two blocks for *any* thread
count — two workers idle, two running double — which the earlier unbalanced-vs-HEAD
run measured at 0.81 / 0.72 for 2T / 4T. Chained onto that, the whole change is
roughly 2T +8%, 4T +4%, 1T −1%, 10T −8%. The 10T loss is the same mechanism
pointing the other way: balanced coarser blocks fit `nt` exactly, the old finer
ones kept 11 blocks on 8–10 threads. ≥8T was previously written off as
unmeasurable on this fanless M4, so it is left as a known cost rather than tuned
to — but it is 0–2/7 wins with ratios clustered in 0.905–0.972 over seven shapes,
so it is not noise (same-binary 3v3 null control: median 0.98, IQR 0.88–1.03).

With `kc` pinned (attention's path) the same code runs with `s = C / kc` in place
of `2·isqrt(C)` — the objective is monotonic in `mc == nc` once `kc` is fixed, so
the tile simply takes the whole budget — and the re-derivation below is skipped.

`kc` is re-derived from the `mc + nc` the routing actually produced — so a shape
that shrinks `mc` gets the freed budget back into `kc` — and clamped to the L1
bound, the dtype cap, and `K`:

```text
kc = snap32(clamp(min(C/(mc+nc), kc_l1, kc_cap, K), KC_MIN, K))
```

On the M4 (f32, `l2_shared` = 16 MiB, `l1` = 64 KiB), for M = N = 8192, K = 4096:

| nt | C | mc | nc | kc | blocks | AI |
|---|---|---|---|---|---|---|
| 1 | 2 097 152 | 768 | 768 | 128 | 11 | 192 |
| 2 | 1 048 576 | 696 | 768 | 128 | 12 | 190 |
| 4 | 524 288 | 696 | 720 | 128 | 12 | 188 |
| 8 | 262 144 | 360 | 480 | 128 | 23 | 158 |

The 768 ceiling binds at `nt = 1`; above that the L2 budget does, and from 8T the
block balancing starts to. **`kc` never leaves `KC_CAP_F32 = 128`** — `C/(mc+nc)`
is 1365 at 1T and still 266 at 8T, so the unconstrained optimum is unreachable and
the cap, not the budget, decides `kc`. That is the deliberate cost of the shorter
k-block (§2.2): the three-heuristic scheme this replaced sat at a flat **AI 169**
on every f32 shape (`nc` pinned below its L2 target by a 1024 cap, `mc` a fixed
144), and the new rule's AI beats that comfortably at 1T–4T but not at 8T, where
rounding the block count up to 24 pulls `mc` from 504 to 360. What the rule is
*for* is the block count tracking the pool — 11/12/12/23 blocks for 1/2/4/8 workers
instead of the old 57.

**Alignment.** `NC_ALIGN = 48 = lcm(12, 16)` is the `nr_max` grid across both
arches, so a 48-aligned `nc` decomposes into whole B panels; `MC_ALIGN = 24 =
lcm(6, 8)` is the `mr_max` grid, so a 24-aligned `mc` leaves no partial A panel
either way. The two differ because the panel lists do: 48 is the widest `nr_max`
the fp pack can be handed, while `mc` decomposes into `mr_max` panels — aligning
`mc` to 48 would only coarsen the even share `balance_tile` is trying to hit.

### 2.2 Per-dtype Kc cap

The formula above asks for `kc` in the high hundreds to thousands on every dtype,
because `esz` makes the model see more room than the kernel can actually use:

| dtype | `kc_cap` | Why the cap |
|-------|----------|-------------|
| f32   | 128 | Not a budget limit but a length limit — kept equal to f16's for one rule. The budget would allow 1365 here (§2.1). |
| f16   | 128 | Bounds the per-k-block fp16 accumulation error — **the aarch64 f16 kernel accumulates in f16**, so a longer k-block means a longer rounding chain before the f32 C read-modify-write. At K = 4096, kc 128 → 256 moved the worst deviation 0.23% → 0.29% of the output magnitude. |
| s8    | 512 | The formula predicts 1344–2048 here (`esz = 1` makes the model see 8× the room); 512 is the value the int8 kernel is tuned and tested at. |

`KC_MIN = 128` is the floor: a shape too small to fill the budget still gets a
full 128-wide k-block. `kc = min(kc, K)` always holds, so short reductions never
pad.

`KC_F16I4 = 256` is declared as a placeholder for a future fp16×int4 path; nothing
reads it.

**On Apple the L1 bound (§2.1 constraint 2) is slack for every dtype** once
`kc ≤ 128` (f32 would need 512, f16 800). It only binds on x86_64, where a 32 KiB
L1D gives f32 `kc_l1 = 192`. It is kept as the prefetch-residency guard it was
specified as, not as the term that drives the choice.

**Attention shares the rule.** `resolve_tile_sizes` — which attention's
non-flash path calls for both GEMM1 and GEMM2 — is a thin wrapper over the same
`choose_matmul_tile`, passing attention's fixed kc (`KC_F32`/`KC_F16` = 128) as
`kc_fixed`. Attention has no M/N split of its own, so both call sites take the
same `split_n = (N > M)` default as matmul. Only the non-flash path is affected:
the flash branch sizes its `Br`/`Bc` from `resolve_flash_tile_sizes`, which does
not read `NC_MAX`. Attention's `nc1` moved 1008 → 768 with the ceiling.

### 2.3 Mc, Nc and Kc are all derived

There is no fixed `MC_TARGET` any more. Every one of the three comes out of the
rule in §2.1; the only per-dtype inputs are `mr_max`, `nr_max`, `esz` and
`kc_cap`, and the only machine inputs are `l1_cache_size()` and
`l2_shared_cache_size()` (§3).

---

## 3. Runtime Cache Detection

The tile rule (§2.1) has exactly two machine inputs, both read here:
`l1_cache_size()` for the prefetch bound and `l2_shared_cache_size()` for the
residency budget.

### 3.1 x86_64: CPUID Leaf 4

Intel/AMD CPUs report deterministic cache parameters via CPUID leaf 4.
We iterate sub-leaves looking for the L1D (Level 1, type Data) and the L2
(Level 2, type Data or Unified):

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
- **macOS:** `sysctlbyname("hw.l2cachesize")`, plus `hw.l1dcachesize` for the L1
  and `hw.perflevel0.l2cachesize` for the shared L2 (below).

**Apple's `hw.l2cachesize` is the efficiency cluster's L2** — 4 MiB on an M4,
against 16 MiB on the performance cluster — so the residency budget would size
itself against a cache four times smaller than the one the work runs in.
`l2_shared_cache_size()` is the accessor that asks for the P cluster explicitly
(`hw.perflevel0.l2cachesize`) and falls back to `l2_cache_size()` where the two
cannot differ. `hw.l1dcachesize` has the mirror-image property: it reports the
*E* cluster's L1D (64 KiB vs the P cluster's 128 KiB), which is the useful one
because workers land on either cluster and a working set sized for the bigger
one overflows on the E cores.

Two fallbacks, both deliberately conservative: 32 KB for the L1 (typical x86 L1D)
and 256 KB for the L2 when every query fails (not the aarch64 default). Under-
provisioning only costs reuse; it never over-subscribes the scratch.

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
time. In practice it never runs: `nc` is a multiple of `NC_ALIGN = 48 = lcm(12, 16)`,
the `nr_max` grid on both arches, so the widest loop consumes the whole tile.

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
- **The block count is a multiple of the worker count.** The split side's tile is
  not `larger / num_threads` any more — it is `balance_tile`'s even share of a
  block count rounded *up* to `nt` (§2.1), so every worker gets the same number of
  blocks instead of some idling while others run double. This is the constraint
  that made the 768 ceiling affordable: without it, M = 1024 resolved to `mc = 768`
  and 2 blocks at *any* pool size. The alignment round-up can still hand back one
  block fewer than targeted (8192 rows over 8 workers targets 24 blocks, gets 23),
  so the count is a target, not a guarantee.
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

5. **Runtime cache detection, not compile-time.** Hardcoding a cache size to a
   lowest-common denominator wastes cache on larger CPUs, and on a hybrid part the
   *wrong cluster's* size is worse than a conservative one: Apple's
   `hw.l2cachesize` is the efficiency cluster's 4 MiB, a quarter of the 16 MiB the
   work runs in. The CPUID path adds ~10 lines of code and pays for itself on any
   CPU with > 256 KB L2.

6. **Mc/Nc/Kc are one derivation, not three heuristics.** Earlier the three came
   from separate rules that shared an L2 budget without knowing about each other:
   `nc` from a single-block working set capped at 1024, `kc` from a
   `num_threads`-blocks-in-half-of-L2 rule capped at 256, and `mc` a flat 144.
   They are now picked together by `choose_matmul_tile` (§2.1) to maximise the
   tile's arithmetic intensity subject to the aggregate-L2, L1-prefetch, ceiling
   and alignment constraints — plus `balance_tile`, which keeps the block count a
   multiple of the worker count and is what makes a large tile affordable. The
   packed-A region is sized `num_panels_max(mc, mr) × ldd_a` from the plan's own
   `mc` — worst-case panel counts exceed `num_panels(mc, mr)` because the greedy
   decomposition is not monotonic in `mc` (24k − 1 needs 3k + 3 panels, 24k only
   3k).

---

## 9. References

- onnxruntime: `cpuid_info.h` / `cpuid_info.cc` — CPUID singleton pattern
- TensorRT: `IPluginV2DynamicExt::getWorkspaceSize` — workspace sizing API
- nn_compute: `mma_direct_f32.hpp`, `pack_f32.hpp` — micro-kernel and pack design
- ComputeLibrary: GEMM assembly kernels — mr/nr panel size selection
- GotoBLAS: the original tiled GEMM loop design and packing model
