# How to Optimize GEMM — Design Notes for the Tiled MatMul Backend

A record of the optimization decisions, constraints, and trade-offs that shaped the
tiled GEMM implementation in [src/backend/cpu/matmul.cpp](../src/backend/cpu/matmul.cpp)
and the shared pack/MMA helpers in [src/backend/cpu/matmul_helper.h](../src/backend/cpu/matmul_helper.h).
The two routes that bypass the plan — the small-GEMM / GEMV paths (rules 3 and 4
below) and the fused Q5_1 decode — are self-contained and live beside it in
[matmul_small.cpp](../src/backend/cpu/matmul_small.cpp) and
[matmul_q5_fused.cpp](../src/backend/cpu/matmul_q5_fused.cpp); what they share with
the driver (dims, batch walk, epilogue predicates) is in
[matmul_internal.h](../src/backend/cpu/matmul_internal.h).

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

The decision is two trivial rules (see `PACK_A_K_THRESHOLD` and
`gemm_is_small` in matmul_helper.h, applied in `get_matmul_plan`):

```text
pack_b = true                                        // rhs always packed
pack_a = is_i8 || transpose_a || (K > PACK_A_K_THRESHOLD)   // 1024 elements
```

Rules, in order:

1. **B is always packed.** The packed RHS is what the micro-kernel consumes
   (`tile_mma_pack(..., -1)` / `tile_mma_direct(..., -1)`); packing it is the
   RHS-reuse win and is never skipped. The one exception is below the plan
   entirely: at M ≤ 4 the driver takes the GEMV route (rule 4), which never
   touches the pack at all.
2. **A packs on correctness, dtype or depth.** int8 has no direct-A route at
   all (`tile_pack_lhs_i8` is unconditional); `transpose_a` forces packing (the
   micro-kernel wants A as `[m][k]` panels); otherwise A packs only when the
   reduction depth exceeds 1024 elements, where a panel's direct row reads no
   longer stay inside a page and packing turns them sequential. Otherwise A is
   read directly via `tile_mma_direct`. The row *pitch* never enters the
   decision.
3. **Small-GEMM path (f32 and f16).** GEMMs with `M*N*K < 1024` MACs never reach
   the plan: each batch element runs the arch's naive SIMD kernel
   (`x86_64/matmul_naive.hpp`, `aarch64/matmul_naive.hpp`) plus one epilogue
   pass. It has no pack, no tiles and no scratch, and the pool is only used once
   `batch × M×N×K ≥ 32768`. Two loop shapes keep it vectorized at every N:
   wide N sweeps eight (four on NEON) columns per row of A; narrow N switches to
   one vectorized-over-K dot product per output. Operands stored the other way
   round are transposed into a 1024-element stack scratch. Unsupported dtype
   pairs and "no workspace" calls still go to the reference. Both this rule and
   the next live in `matmul_small.cpp`; `matmul_small_fp` is the whole driver
   interface to them.
4. **GEMV path (f32 and f16, M ≤ 4 at any size).** Above the small-GEMM budget
   the packed route is the wrong trade for a handful of output rows: it sweeps
   all of B twice (write the packed copy, read it back) and reads it a third
   time in the micro-kernel, while a direct kernel reads B once. So at
   `M <= GEMV_M_MAX` (4) the driver calls the arch's `matmul_gemv` and skips the
   plan. The kernel walks **k outer, n-block inner** with `gemv_block = 1024`
   (4 KB contiguous of B per k, accumulators L1-resident); a `[N][K]` B gets the
   dot-product twin that keeps one k-vector accumulator per output row.
   `transpose_a && transpose_b` is the one layout pair the direct kernel
   declines, and the packed route answers it. Blocks give the pool N-parallel
   work; a batch > 1 parallelizes over the batch instead.

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
- `TensorDesc.row_stride_elems` only describes the input's actual row pitch;
  the pack decision for A is `K`-based (§1.2), and the packed-A kernel reads
  rows by that pitch either way. It does not affect the workspace size, which
  depends only on N, Kc, dtype and the NR panel list.
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
| s8    | 512 | Not a precision limit — the accumulator is int32, so `kc` cannot round. The formula predicts 1344–2048 here (`esz = 1` makes the model see 8× the room), and below this cap the budget term is what decides `kc`, so on a host where it never binds (a large shared L2) the plan reports the host's L2 rather than the rule's answer — 16 MiB asks for 512 where 2 MiB asks for 256. Capping at 256 would make `kc` host-invariant; measured, it cost 3–13% on the s8 GEMM (worst at 10 threads on a deep K, where the shorter k-block buys a second C read-modify-write), so the cap stays at the tuned value. |

`KC_MIN = 128` is the floor: a shape too small to fill the budget still gets a
full 128-wide k-block. `kc = min(kc, K)` always holds, so short reductions never
pad.

`KC_F16I4 = 256` is declared as a placeholder for a future fp16×int4 path; nothing
reads it.

**On Apple the L1 bound (§2.1 constraint 2) is slack for every dtype**: the caps
(128 fp / 512 s8) all sit below their own bound (f32 would need 512, f16 800,
s8 2048). It only binds on x86_64, where a 32 KiB L1D gives f32 `kc_l1 = 192`. It
is kept as the prefetch-residency guard it was specified as, not as the term that
drives the choice.

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

The int8 path folds only `Relu` (as a clamp on the requantized output). Its
output pass is fused exactly like fp's: `int8_epilogue_tile` runs inside
`matmul_tile` at `last_k`, on the tile the last k-block just produced, and does
compensation plus (for an s8 output) requantization in one vectorized row walk
(`int8_epilogue_row_s32` / `_s8` in
[simd_epilogue.hpp](../src/backend/cpu/simd_kernel/simd_epilogue.hpp)). Only the
s8 destination moves per batch element, so the parameters
(`Int8EpilogueParams`) are built once per call.

The one thing that cannot be tile-local is the **reduction** input: the
compensation needs `r_a[m] = Σ_k A[m,k]` and `r_b[n] = Σ_k B[k,n]` over the
*full* K, so they must be complete before the first tile finishes. They stay a
pre-pass (`compute_int8_reductions`, kernels in
[simd_int8_reduce.hpp](../src/backend/cpu/simd_kernel/simd_int8_reduce.hpp)),
one sweep of A and B, 0.03 ms of a 2.6 ms call at 1024×512×1024. Folding them
into the packers is the only way to remove that sweep, and it would either
duplicate the A sum across every N-block task or need per-task scratch.

Both passes are **skipped whenever they are the identity** — no s8 output, no
activation, no A shift left to withdraw, and zero points of zero
(`matmul_int8_compensate` then returns the raw accumulator unchanged). Before
that gate existed they were a separate sweep of A and B plus a scalar pass over
all of C, and at shallow K they dominated the call (K=128 was 111 of the
kernel's ~570 GOPS). Worth checking the same recursion whenever an epilogue grows
a branch that is a no-op for the default attributes.

The fusion's measured value is modest: on s8→s8 with per-tensor scales, 256³
went 0.120 → 0.108 ms/call at 1T and 1024×512×1024 was flat (2.59 → 2.60 ms).
The pass is compute-bound on the f64 requant, not on the C re-read, so removing
the second sweep of C does not buy much; the win is structural — one dispatch
instead of two, no full-M×N pass.

### 6.1 fp activations → int8 GEMM → fp output (W8A8, offline parameters)

The deployment shape: the activation quantization parameters are calibrated
offline, A arrives as f32/f16 carrying them in its `QuantParams` (PerTensor or
PerToken), B is s8 with a per-output-channel scale, and C must come back in A's
dtype. `is_fp_quant_matmul(a, b, a_qp)` is the predicate; `matmul_cpu` routes it
to `matmul_dq<T>`.

```text
A_q[m,k] = clamp(rne(A[m,k] · (1/s_a[m]) + zp_a[m]), qmin, qmax)   // s8 or u8
acc      = Σ_k (A_q[m,k] + u8_shift) · B[k,n]                     // the int8 MMA
C[m,n]   = ep(clamp(acc · s_a[m] · s_b[n], min_clip, max_clip))   // f32 math
```

Three pieces, and only the first is new machinery:

1. **The quantize pass** (`quantize_activation<T>`) reuses
   `quant_kernel::quantization` — per-row scale, per-row-or-null zero, FMA +
   RNE + saturating clamp, f32 and f16 overloads on both arches. It slab-splits
   rows over `ctx.cpu.run`, so it costs 0.031 ms of a 2.3 ms call at
   1024×512×1024 (1.3%) and ~17 Gvalues/s at 1T.
2. **The GEMM is the existing int8 route.** The trick that makes this free:
   `matmul_dq` describes A_q as a compact s8/u8 `TensorView` with *A's shape and
   A's QuantParams* — the same view shape it hands to the plan, and the same one
   it materializes when the pack has to read it. `get_matmul_plan`, the batch
   walk, `lda`, `compute_int8_reductions` and — critically —
   `a_biased`/`a_bias_offset`, which come from the *target* dtype, are then all
   correct with **zero plan changes**. Never derive the pack bias from the
   original fp A: on x86 that loses an XOR and on aarch64 adds one, and every
   output is wrong. That is what `R::a_eff_type(attrs, a)` exists for: on the
   fused route it answers `attrs.act_quant_dtype`, everywhere else
   `a.data_type()`.
3. **The epilogue** is `dq_epilogue_tile<TOut>` (in
   [simd_epilogue.hpp](../src/backend/cpu/simd_kernel/simd_epilogue.hpp)),
   modelled on `int8_epilogue_tile`: same compensation, then
   `float(acc) · (s_a·s_b)` in f32, then clip, then `apply_epilogue`. All
   indices are absolute columns, so it works unchanged for the N-split.

The order matters: **clip cannot be folded into the MMA clamp**, because the MMA
clamps the int32 accumulate while the clip applies after dequantization — even
`Relu` has to run in the epilogue. `DqRoute<TOut> : I8Route` therefore sets
`always_epilogue = true`, `c_base()` always returns the int32 accumulator (the
output is fp, so it cannot accumulate in place), and `extra_workspace` is
`M·N·4` unconditionally.

Per-token arrays are indexed by A's **global** row, so the per-batch refresh uses
`row_base = a_offset / lda` (batch strides are whole rows). A broadcast A lands
at row 0 of the array, which is what the reference does too.

The same route serves the "A is already quantized, give me fp out" case
(`output_dtype = f32/f16` on an s8/u8 A): no quantize pass, everything else
identical.

Measured (Release, 20 logical cores, `bench_matmul matmul_dq_*`). The point
cases pin one shape each; `matmul_dq_size_sweep` / `_tb_size_sweep` /
`_f16_size_sweep` / `_rect_sweep` replay the same square ladder as
`matmul_i8_size_sweep` (plus the N-split, deep-K and 1-row corners) so the two
tables read side by side, and `matmul_dq_vs_fp_sweep` prints dq/dense per size:

At 1024×512×1024, 1 thread:

- dq, per-token, f32 activations: **~460 GFLOPS** (457.9 median of the A/B rows)
- same-shape `matmul_i8_nn_large`: 540 GOPS — dq is **85%** of int8
- dense f32 B over the same A: 142 GFLOPS — **dq/dense = 3.2**
- dense f16 B at 4T: 293 GFLOPS against dq's 957 — **dq/dense = 3.3**; a B that
  is only half as wide saves less on paper, but the f16 dense route scales
  worse across threads, so the measured ratio holds
- the quantize pass on its own: 1.3% of the call

**The reduction sweep is conditional.** `compensate()` is the identity when
`a_bias_offset + zp_a == 0` for every row and every `zp_b == 0`, which is the
ordinary symmetric s8 W8A8 setup; the dispatch then skips
`compute_int8_reductions` entirely (`int8_passes == false`, while `int8_ep`
keeps the dequant output pass running) and the row pass drops both loads.
Measured on the same instrument by flipping that one condition: **+3%** at
1024×512×1024 and **+13%** at 256³ on one thread, and one fewer `ctx.cpu.run`
per call, which in the bench harness is worth another ~20-30% at 4T because
`SimplePool` re-spawns its workers every call. u8 activations (`zp_a = 128`)
keep the sweep: `coef_rb = 128` is not zero.

The m-ladder (K=N=2048, dq/dense) goes 1.9, 1.5, 3.1, 3.2, 3.6, 3.5 at M = 1, 4,
16, 64, 256, 1024. At M ≤ 4 both routes are weight-streaming bound and the dq
side wins only on B's width — 4 MB against 16 MB — because it has no GEMV
kernel and runs the tiled GEMM for a handful of rows; that is also where
dropping the reduction sweep helps most. From M=16 up the int8 FLOP rate takes
over and the ratio settles near 3.5. High-thread columns sag for *every* route
in that harness (SimplePool spawns workers per call); read the 1T column.

One more harness trap, measured while writing this: under sustained 4T load this
host enters throttled episodes lasting seconds, during which **both** routes in
an A/B read 4-5x slow, so whichever row happens to be in flight takes the hit.
With all four A/B rows in one process the f16 row (last) read dense 55 / dq 145
in three consecutive runs; moved to the front it read 241 / 646, and the row
that ended up last sagged instead. That is why `matmul_dq_vs_fp_ab_nn`, `_tb`
and `_f16` are separate registrations — run each in its own process when the
multi-threaded numbers matter. 1T rows are stable in any layout.

**The quantize pass is fused into the A pack** when nothing else needs A_q: no
compensation (`int8_passes == false`), an M-split, and an untransposed A. Then
`DqRoute<TOut, AIn>` is instantiated with `AIn = float/half`, `matmul_dq` passes
the *fp* `a` to the dispatcher, and `pack_a` calls
`tile_pack_quant_lhs_i8`: each panel row is quantized into a 4 KB stack buffer
(`MR_MAX_I8 × kc ≤ 8 × 512`) and then handed to the ordinary
`pack_trans_i8_lhs[level]` kernel, so the panel bytes and the rounding are
identical to quantize-then-pack — every dq test compares bit for bit against
`reference::matmul_dq_ref`, and `matmul_dq_shape_sweep_symmetric` runs the whole
tiling-boundary table through this route (the non-symmetric sweeps pin the
materialized one). The stack copy stays in L1; what
disappears is the full M×K write plus its read, the `PoolPtr` for A_q, and the
`ctx.cpu.run` that used to carry the pass.

Three gates, each with a reason:

- **no compensation** — the sweep is the only other reader of A_q, and the pack
  cannot accumulate `r_a` per row without per-task scratch. A non-zero `zp_a`
  or `zp_b` keeps the materialized route.
- **M-split** — under `split_n` the same A row is packed by every N-block task,
  so the fusion would quantize it `nt` times instead of once.
- **untransposed A** — the transposing pack is the one that maps M×K to panels;
  `transpose_a` already routes to `matmul_dq_ref`.

Measured with the same instrument (an interleaved A/B, 8 rounds per config,
1024×512×1024 f32): **1T 455→461 (+1%)**, 2T 710→740 (+4%), 4T 978→1115
(+14%), 8T 936→1119 (+20%). At 256³ on one thread it is a wash (±1%). The
shape of that curve is the point: the arithmetic saved is small (1.3% of the
call), but the extra `ctx.cpu.run` is not, and `SimplePool` re-spawns workers
per call — so the fusion is worth about one run's worth of overhead, which only
multi-threaded measurement shows.

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
   kernels per batch element with numpy-style broadcast. Fused tiled kernels
   exist for f32, f16, and s8×s8 → s32/s8 (`matmul_dispatch_int8`); small f32/f16
   products go to the naive arch kernel (§1.2 rule 3), M ≤ 4 f32/f16 products to
   the GEMV kernel (§1.2 rule 4), and every other dtype combination falls through
   to the reference kernel.

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
