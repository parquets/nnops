# How to Tune a CUDA GEMM on This Host

A record of the method used to take the hand-written f32 SIMT GEMM
(`src/backend/cuda/matmul/matmul_manual_f.cuh`) from 1.27x cuBLAS to 1.11x, and
f16/bf16's PTX kernel to 0.98-1.10x. The method matters more than the numbers:
most of the knobs turned out to be noise, and two changes carried everything.

## 1. Measurement discipline first

The host drifts **+-10% run to run**. Anything smaller than that is invisible
unless the measurement is built to survive it:

- **Interleave the arms.** `bench_vs_cublas` runs twelve samples alternating
  which side leads, and prints the ratio of medians. Two arms timed to
  completion separately disagree by 10-13% on identical code.
- **Compare inside one process, one warm state.** A tile sweep that runs every
  candidate back to back ranks them reliably; the same candidates across two
  processes do not.
- **Change one thing per build.** Several "wins" in this session evaporated when
  two knobs moved at once (the 4-block tile first looked bad because it was
  bundled with a register cap and a padding removal).
- **Distrust absolute GFLOPS across sessions.** cuBLAS's own 4096^3 f32 number
  moved between 9.8k and 10.5k across hours on the same card; the ratio did not.

## 2. The tool ladder

Work down it only as far as the question needs:

| Question | Tool | Cost |
|---|---|---|
| Is A faster than B? | interleaved event timing (`bench_vs_cublas`) | free |
| What launch shape does the reference use? | `ncu --set launchstats` | no elevation |
| Which pipe is saturated / how many CTAs live? | `ncu --set detailed` (SpeedOfLight, Occupancy, LaunchStats) | **one elevated run** |
| Why is it slow at equal occupancy? | `ncu --metrics smsp__inst_executed*` (dynamic counts per pipe) | elevated |
| Where do the instructions come from? | `cuobjdump -sass` + opcode histogram | free |
| What does the kernel actually store? | `cuobjdump --dump-resource-usage` (REG/STACK) | free |

Notes that cost time to learn:

- On Windows the binary is at
  `C:/Program Files/NVIDIA Corporation/Nsight Compute <ver>/ncu.bat`; the
  CUDA toolkit ships only compute-sanitizer. Launch stats need no elevation,
  hardware counters fail with `ERR_NVGPUCTRPERM` until one elevated run.
- `STACK:256` in `--dump-resource-usage` means an accumulator array fell into
  local memory. It happens from a *single* run-time index anywhere in the
  kernel (here: the epilogue's `for (; c < LC; ++c)` tail). Unroll it away and
  the frame drops to 8 bytes and the registers come back.
- Static SASS histograms are only comparable between kernels with the same
  epilogue. Ours has no bias/activation path, CuTe's carries the general one, so
  its static count is dominated by `MUFU.EX2`/`RCP`. Use **dynamic** counts from
  ncu for cross-kernel comparison, static ones only to find *where* in the code.

## 3. The diagnostic ladder that worked

Each step is a symptom, the tool that named it, and the knob that fixed it.

1. **Same occupancy, higher IPC, longer duration** (ncu detailed) → the kernel
   executes more instructions rather than stalling more. `smsp__inst_executed`
   per pipe: FMA equal, **ALU 2.26x** the reference, LSU +12%. That is the whole
   story: instruction count, not latency.
2. **SASS: FFMA 768 / FMUL 384 / FADD 256** where the reference had pure FFMA →
   the accumulators were written as `acc += a*x + b*y + c*z + d*w`, which ptxas
   compiles as a sum tree. Writing four chained `acc = fmaf(a, x, acc)` fixed
   it (FFMA 1024 / FMUL 128 / FADD 0) and bought 7-9%.
3. **SASS body histogram: FFMA density 72% against the reference's 86%**, with
   41 ISETP per k tile → the fill tested every vector against the bounds. Split
   at **block level** (`kFull`, `aFull`, `bFull` decided once per k tile; the
   interior runs a check-free cp.async loop) and the excess disappears: +13% on
   top of step 2.
4. **Waves per SM 1.12 on a 128-block grid** (ncu launch stats) → a 64-block grid
   is under one wave and the second wave runs 14 blocks. Use a smaller CTA (more
   blocks) or fewer CTAs per SM: a 64x64 tile with four resident blocks turned
   1024^3 from 6.45k to 7.41k TFLOPS.
5. **A transposed pose 25 registers heavier than its twin** (resource usage) →
   two fragment arrays were declared, one per arrival order, and the unused one
   stayed live. One unified array plus `__launch_bounds__(threads, 3)` restored
   the third resident block (+8% on NT, +21% on TT).

## 4. What did *not* work (measured, with numbers)

Recording these matters as much as the wins:

- **Fragment shadow / software pipelining** (the +64 register CuTe-style double
  buffer): not built, because its cheap stand-in — deeper cp.async rings —
  lost 5-20% at every size. The ring is not the limiter.
- **Stage-cycle unrolling** (compile-time stage indices, CuTe's `k_body<RD,WR>`):
  neutral on NN, 5% worse on TN.
- **Manual address hoisting**: byte-identical SASS; ptxas already does it.
- **32-bit fill offsets**: SASS 3960 -> 3904 instructions (1.4%), i.e. noise.
- **Matching the reference's tile orientation** (64x128 for the n-major pose):
  2% worse than 128x64.
- **Forcing four blocks a SM** on the wide tiles (register cap + no panel skew):
  3.9k and 6.9k TFLOPS against 8.85k.
- **Split-K at 1024^3** even after profiling cuBLAS and copying its exact
  geometry (`ampere_sgemm_128x64_tn`, grid (8,16,2)): 0.4055 against 0.3951 ms.
  Its split pays because its tile is smaller; ours pays the plane traffic.
- **The f32 four-block tile does not transfer to f16**: a 64x64 CTA with a 32x32
  per-warp tile and four resident blocks measured 28.5k GFLOPS against 34.7k for
  the 128x128 CTA at 4096^3 NN -- an 18% loss. For the MMA kernel the per-warp
  tile (MMA work per shared-memory load) dominates; for the SIMT kernel
  occupancy did. Only TN at 1024^3 gained (22.3k against 20.6k).
- **Rectangular CTA variants** (128x256, 64x256, 32-wide columns): all one block
  a SM or 2 B/FMA, all worse.

## 5. Results of this round

f32, interleaved against cuBLAS, ratio = ours/cuBLAS (lower is better):

| pose | 2048^3 before | 2048^3 after | 4096^3 before | 4096^3 after |
|---|---|---|---|---|
| NN | 1.19 | **1.034** | 1.17 | **1.040** |
| TN | 1.27 | **1.134** | 1.27 | **1.115** |

f16 (PTX MMA kernel), after the same block-level fill split:

| pose | 1024^3 | 2048^3 | 4096^3 |
|---|---|---|---|
| TN | 1.123 | 1.041 | 1.114 |
| NN | **0.864** | 1.089 | **1.005** |

The fill fix transferred unchanged: the MMA does the math, so the fill's
per-vector bounds tests were a larger share of the remaining instructions than
they are in the f32 kernel.

Two changes carried it: **fmaf chains** (+7-9%) and **block-level fill
fullness** (+13%). Everything else in this session was noise or negative.

## 6. Checklist for the next round

- Measure with `bench_vs_cublas`, never with two sequential timings.
- `ncu --set detailed` once, elevated, on one launch of each arm; read
  Occupancy, Waves, SM%, L1/TEX and DRAM before touching anything.
- If IPC is high and duration long, count instructions per pipe; do not tune
  tiles.
- If the FFMA density of the loop body is under ~85%, find what else is in it.
- Check `STACK` — a non-zero frame is an array that lost its registers.
- Write FMA chains explicitly. Never write a sum and hope for contraction.
- Everything else: change one knob, keep the number, write the loss down.
