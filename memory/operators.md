---
name: operators
description: "List of implemented operators with their attributes, input/output specs, and implementation patterns"
metadata:
  node_type: memory
  type: project
  originSessionId: 2c7fb42c-b26c-4d7e-a6dd-2de30d713ad8
  modified: 2026-07-21T14:32:59.172Z
---

# Implemented Operators

All operators support: class API (`Op::create()` → `op->compute()`) and functional API (free function).
Reference project patterns from onnxruntime (OpKernel + attributes struct), ComputeLibrary (IOperator + kernel dispatch), and TensorRT (plugin capability interfaces).

## Epilogue (Core Type)

- **File:** `include/nnops/core/epilogue.hpp`
- **Purpose:** Post-processing applied during output write-back, enabling operator fusion without a separate kernel launch. Intentionally separate from [[#Activation]] — keeps the standalone Activation operator clean (no None sentinel) and allows future non-activation types (quantize/dequantize).
- **Types:** `EpilogueActivateType` enum — None, Relu, Gelu, Sigmoid, Tanh, LeakyRelu, Silu, HardSwish, Elu
- **Struct:** `Epilogue` — `type` (default None=identity), `alpha` (LeakyRelu/Elu), `beta` (HardSwish), plus per-channel quantization fields (`quant_scales`, `quant_zero_points`, `quant_param_count`, `quant_axis` — reserved for future Dequantize/Requantize)
- **API:** Two `apply_epilogue()` overloads — scalar `apply_epilogue(ep, x)` for activation types, and channel-aware `apply_epilogue(ep, x, channel)` for future per-channel quantization (activation types delegate to scalar overload)
- **Users:** Conv2D, Conv3D, MatMul, Linear — each holds `Epilogue epilogue{};` in their attributes and applies it at the output write-back site
- **Added:** 2026-07-21

## Conv2D

- **Files:** `include/nnops/ops/conv2d.hpp`, `src/ops/conv2d.cpp`, `src/backend/cpu/reference/conv2d_ref.cpp`
- **Attributes:** `Conv2DAttributes` — kernel_size[2], stride[2], dilation[2], padding[2], groups, auto_pad, epilogue
- **Input:** `[N, IC, IH, IW]`, weight `[OC, IC/G, KH, KW]`, optional bias `[OC]`
- **Output:** `[N, OC, OH, OW]`
- **CPU Reference:** 7-level nested loop, per-sample parallel via `parallel_for`; output write: `apply_epilogue(attrs.epilogue, sum, oc_global)`

## Conv3D

- **Files:** `include/nnops/ops/conv3d.hpp`, `src/ops/conv3d.cpp`, `src/backend/cpu/reference/conv3d_ref.cpp`
- **Attributes:** `Conv3DAttributes` — kernel_size[3], stride[3], dilation[3], padding[3], groups, auto_pad, epilogue
- **Input:** `[N, IC, ID, IH, IW]`, weight `[OC, IC/G, KD, KH, KW]`, optional bias `[OC]`
- **Output:** `[N, OC, OD, OH, OW]` (NCDHW layout)
- **CPU Reference:** 9-level nested loop, per-sample parallel via `parallel_for`; output write: `apply_epilogue(attrs.epilogue, sum, oc_global)`
- **Added:** 2026-07-21
- **Note:** Same Impl/Pimpl pattern as Conv2D. Follows the 2D→3D upgrade pattern established by [[#Pooling]].

## Activation

- **Files:** `include/nnops/ops/activation.hpp`, `src/ops/activation.cpp`, `src/backend/cpu/activation.cpp`, `src/backend/cpu/reference/activation_ref.cpp`
- **Types:** Relu, LeakyRelu, Sigmoid, Tanh, Gelu, Silu, HardSwish, Elu
- **Attributes:** `ActivationAttributes` — type, alpha, beta
- **SIMD Kernel** (`src/backend/cpu/activation.cpp`): 8-wide SIMD (v_f32x8) + scalar tail，v_f32x4 已删除以简化代码。8 种激活函数全部向量化。
- **Design Decisions:**
  1. 无需 blend/select — 所有条件分支通过 v_max/v_min 分解消除：LeakyRelu = `max(x,0) + α*min(x,0)`，ELU = `max(x,0) + α*(exp(min(x,0))-1)`，Relu = `max(x,0)`
  2. 8-wide SIMD → scalar tail，无需 4-wide 中间阶段（AVX2 原生 256-bit，SSE 模拟为两个 __m128）
  3. add_to 支持：load + accumulate + store 替代 plain store
- **Dispatch:** `Backend::CPU` → `activation_cpu`（SIMD 优化），reference 保留用于正确性基线
- **Note:** Standalone operator — distinct from [[#Epilogue (Core Type)]] which fuses activation into Conv/MatMul output write-back.

## Pooling

- **Files:** `include/nnops/ops/pooling.hpp`, `src/ops/pooling.cpp`, `src/backend/cpu/reference/pooling_ref.cpp`, `src/backend/cpu/pooling.cpp`
- **Types:** Max, Average, AverageExcludePad, Lp
- **Attributes:** `PoolingAttributes` — `std::array<int64_t, 3>` for kernel/stride/padding/dilation (layout: `[KD, KH, KW]`), `p_norm`, `auto_pad`, `add_to`
- **Input:** 2D `[N, C, IH, IW]` or 3D `[N, C, ID, IH, IW]`
- **CPU Reference:** Triple-nested (2D) / quadruple-nested (3D) loop per output position, with full bounds checking per kernel element. All 4 pool types supported.
- **CPU SIMD:** Height-4 blocking (process 4 output rows simultaneously) + 8-wide v_f32x8 SIMD over output width dimension. Design combines best practices from nn_compute (pre-positioned pointers, trust boundary — no per-element bounds checks in SIMD kernels, region splitting) and onnxruntime MLAS (two-phase reduction noted for future).
  - **Pre-positioned pointer passing** (nn_compute pattern): Caller pre-positions input+output pointers to (id, ih, iw_start), SIMD kernels use only relative offsets — eliminates positional computation bugs (depth-off-by-oh_start, ow_start-vs-ow confusion).
  - **Trust boundary**: SIMD kernels called only for interior valid region (od/oh/ow within `[beg, end)` where every kernel element maps to valid input). No per-element `vd0`/`vh0` validity checks inside SIMD kernels.
  - **Region splitting**: pad-depth front/back → scalar, pad-top/bottom → scalar, pad-left/right → scalar, interior → SIMD (h4 for aligned blocks, h1 for remainder).
  - **SIMD gating**: Only active when `SW == 1` && type is Max or Average. AverageExcludePad and Lp use scalar path (per-element counting required). stride > 1 uses all-scalar fallback (nnops SIMD has no gather operations).
  - **add_to**: Branch-hoisted inside SIMD kernels; one extra `load + v_add` before `v_store` when enabled.
  - **Unified 2D/3D**: Single code path; 2D is KD=1/SD=1/DD=1/PD=0/ID=1/OD=1 degenerate case.
- **Note:** Upgraded from 2D-only to 2D/3D on 2026-07-20. SIMD kernel added 2026-07-21. Reference sources: `d:\vscode\nn_compute\src\cpu\kernel\pooling\` and `d:\git\onnxruntime\onnxruntime\core\mlas\lib\pooling.cpp`.

## Linear

- **Files:** `include/nnops/ops/linear.hpp`, `src/ops/linear.cpp`, `src/backend/cpu/reference/linear_ref.cpp`
- **Attributes:** `LinearAttributes` — epilogue (added 2026-07-21; previously Linear had no attributes struct)
- **Formula:** `output = input × weight^T + bias`
- **Shapes:** input `[M, K]`, weight `[N, K]`, bias `[N]`, output `[M, N]`
- **CPU Reference:** Triple-nested GEMM, per-row parallel; output write: `apply_epilogue(attrs.epilogue, sum, n)`
- **Convenience:** `Linear::create(Backend::CPU)` still works (defaults to empty `LinearAttributes{}`)

## MatMul

- **Files:** `include/nnops/ops/matmul.hpp`, `src/ops/matmul.cpp`, `src/backend/cpu/reference/matmul_ref.cpp`
- **Attributes:** `MatMulAttributes` — transpose_a, transpose_b, epilogue
- **CPU Reference:** Triple-nested GEMM with transpose handling, per-row parallel; output write: `apply_epilogue(attrs.epilogue, sum, n)`

## Attention

- **Files:** `include/nnops/ops/attention.hpp`, `src/ops/attention.cpp`, `src/backend/cpu/reference/attention_ref.cpp`
- **Attributes:** `AttentionAttributes` — num_heads, scale (0=auto `1/v_sqrt(d)`), use_causal_mask
- **Input:** Q/K/V — merged `[B, S, H*D]` or explicit `[B, H, S, D]`, optional mask
- **CPU Reference:** Per-head parallel → QK^T → scale → mask → softmax → weighted V sum
- **Added:** 2026-07-20

## Operator Implementation Pattern

Every operator follows this recipe:

1. **Header** (`include/nnops/ops/<name>.hpp`): Attributes struct + Op class (inherits OpBase) + functional free function declarations. Class includes `using OpBase::compute;` to expose single-output convenience overload alongside the span-based override.
2. **Dispatch** (`src/ops/<name>.cpp`): `create()` factory, `compute()` with `switch(backend_)`, functional wrappers. Dispatch layer asserts `outputs.size() == 1` and extracts `outputs[0]` before calling backend kernels (which keep `const TensorView& output` parameter).
3. **CPU Reference** (`src/backend/cpu/reference/<name>_ref.cpp`): Naive implementation in `nnops::backend::cpu::reference` namespace
4. **Tests** (`tests/test_<name>.cpp`): Hand-verified small tests + random data tests + class/functional parity tests

### Multi-Output Support (2026-07-22)

`OpBase::compute()` primary virtual takes `std::span<const TensorView> outputs` (matching the span-based inputs). A non-virtual convenience overload `compute(const TensorView& output, ...)` delegates to the span version for single-output callers. Kernel functions are unchanged — they receive `outputs[0]` extracted by the dispatch layer. Future multi-output operators (Split, TopK, etc.) can iterate `outputs[i]` directly.

## GEMM Micro-Kernels (Backend Infrastructure)

Not an operator per se, but the low-level building block for future optimized MatMul/Linear/Conv GEMM dispatch. Three layers organized per-ISA:

| Layer | Files (per arch) | Purpose |
|---|---|---|
| Transpose | `src/backend/cpu/{x86_64,aarch64}/transpose.hpp` | Register-level matrix transpose (2/4/6/8/12/16/24-way) |
| Pack | `src/backend/cpu/{x86_64,aarch64}/pack_f32.hpp`, `pack_f16.hpp` | LHS transpose pack + RHS copy pack for GEMM data layout |
| MMA Pack | `src/backend/cpu/{x86_64,aarch64}/mma_pack_f32.hpp`, `mma_pack_f16.hpp` | C[mr][nr] += A_packed[mr][K] × B_packed[nr][K] with clamp |
| MMA Direct | `src/backend/cpu/{x86_64,aarch64}/mma_direct_f32.hpp`, `mma_direct_f16.hpp` | C[mr][nr] += A[mr][K] × B[K][nr] from strided row-major (no pack) |

**Panel sizes (tuned per ISA):**

| Arch | dtype | mr[] (LHS heights) | nr[] (RHS widths) |
|---|---|---|---|
| AArch64 | f32 | {8, 4, 1} | {12, 4, 1} |
| AArch64 | f16 | {8, 4, 1} | {24, 8, 1} |
| x86_64 | f32 | {6, 4, 1} | {16, 8, 1} |
| x86_64 | f16 | {6, 4, 1} | {16, 8, 1} |

**K step:** x86_64 = 8 (__m256 width), AArch64 = 4 (float32x4_t width for f32, 8 for f16 using float16x8_t)

**FP16 approach divergence:**
- **x86_64**: Data stored as `half` (uint16_t). F16C `_mm256_cvtph_ps`/`_mm256_cvtps_ph` for convert, fp32 FMA accumulators, clamp in fp32 then round-to-nearest to fp16 on v_store
- **AArch64**: Native NEON fp16 arithmetic (`float16_t` type, `vfmaq_lane_f16` intrinsics) — requires ARMv8.2-A+

**Shared utilities:**
- `src/backend/cpu/common/restrict.hpp` — cross-platform `NNOPS_RESTRICT` macro (MSVC `__restrict`, GCC/Clang `__restrict__`)
- `src/backend/cpu/common/half.hpp` — `half` struct (uint16_t storage) + `half_to_float`/`float_to_half` converters for x86_64 fp16 kernels

**Porting reference:** `D:\vscode\nn_compute` — all 10 kernel files ported with modern C++ improvements (noexcept, constexpr, lambda helpers, Doxygen). See [[reference-projects]].

**Added:** 2026-07-21 (aarch64) and 2026-07-21 (x86_64)

## DepthwiseConv2D

- **Files:** `include/nnops/ops/depthwise_conv2d.hpp`, `src/ops/depthwise_conv2d.cpp`, `src/backend/cpu/reference/depthwise_conv2d_ref.cpp`, `src/backend/cpu/depthwise_conv2d.cpp`
- **Attributes:** `DepthwiseConv2DAttributes` — kernel_size[2], stride[2], dilation[2], padding[2], epilogue, add_to
- **Input:** `[N, C, IH, IW]`, weight `[C, 1, KH, KW]`, optional bias `[C]`
- **Output:** `[N, C, OH, OW]` (same channels as input — no channel change in depthwise)
- **Formula:** Each channel is convolved independently with its own KH×KW filter — no cross-channel mixing
- **CPU Reference:** 6-level nested loop (N, C, OH, OW, KH, KW), per-sample parallel via `parallel_for`; pad handling via bounds check
- **CPU SIMD:** Height-4 blocking (process 4 output rows simultaneously) + 8-wide v_f32x8 SIMD over output W dimension. Region splitting: pad-top/bottom → scalar, pad-left/right → scalar, interior → SIMD (h4 for aligned blocks, h1 for remainder). Design informed by nn_compute (height blocking + region dispatch) and onnxruntime MLAS (3×3 depthwise kernel specialization, edge handling).
- **SIMD gating:** Only active when stride_w == 1 (contiguous input reads); stride > 1 falls back to scalar path automatically.
- **Added:** 2026-07-21

## add_to (Output Accumulation)

All operators support `bool add_to{false}` in their Attributes. When true, the kernel adds its result to the existing output buffer instead of overwriting:

```
add_to=false (default):  output[i]  = result
add_to=true:             output[i] += result
```

**Purpose:** Enables residual connections and skip connections without a separate v_add kernel. For example, `Conv(input, output, {.add_to=true})` performs `output += Conv(input)` in a single kernel launch.

**Implementation pattern:**
- Scalar: `out[i] = attrs.add_to ? out[i] + val : val;`
- SIMD: branch-hoisted outside the loop; add_to path does `v_store(y, v_add(load(y), computed))` (one extra load per vector)

**Added:** 2026-07-21
