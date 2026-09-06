---
name: operators
description: "List of implemented operators with their attributes, input/output specs, and implementation patterns"
metadata:
  node_type: memory
  type: project
  originSessionId: 2c7fb42c-b26c-4d7e-a6dd-2de30d713ad8
  modified: 2026-07-25T02:45:03.094Z
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

## DepthwiseConv2D

- **Files:** `include/nnops/ops/depthwise_conv2d.hpp`, `src/ops/depthwise_conv2d.cpp`, `src/backend/cpu/depthwise_conv2d.cpp`, `src/backend/cpu/reference/depthwise_conv2d_ref.cpp`, `src/backend/cuda/depthwise_conv2d.cu`
- **Attributes:** `DepthwiseConv2DAttributes` — kernel_size[2], stride[2], dilation[2], padding[2], epilogue, add_to
- **Input:** `[N, C, IH, IW]`, weight `[C, 1, KH, KW]`, optional bias `[C]`
- **Output:** `[N, C, OH, OW]` (NCHW layout)
- **Formula:** Per-channel convolution — each input channel is convolved with its own independent KH×KW filter. No cross-channel mixing.
- **CPU SIMD Kernel** (`src/backend/cpu/depthwise_conv2d.cpp`): Height-4 blocking (process 4 output rows simultaneously to reuse kernel weights), width SIMD via `simd_lane_for<T>` (8 lanes), region splitting (pad regions → scalar, interior → SIMD), kernel weight pre-load into registers, N×C parallel via `ctx.cpu.run`. Falls back to reference when SIMD unavailable.
- **Data types:** f32 and f16 (templated `depthwise_conv2d_impl<T>`, dtype dispatch at entry)
- **CUDA Kernel** ([depthwise_conv2d.cu](src/backend/cuda/depthwise_conv2d.cu)): Grid = N×C blocks, one per (sample, channel). Epilogue fusion via `device_apply_epilogue()` supporting all 8 activation types. Handles stride, dilation, padding. Optional bias. Supports add_to.
- **Design reference:** nn_compute depthwise_conv (h4 blocking + w8 SIMD), onnxruntime MLAS sconv_nchw_depthwise (3×3 kernel specialization), ARM ComputeLibrary NEDepthwiseConvolutionLayer (stride/dilation handling)
- **Added:** 2026-07-20 (CPU reference), 2026-07-23 (CPU SIMD + CUDA)

## Eltwise

- **Files:** `include/nnops/ops/eltwise.hpp`, `src/ops/eltwise.cpp`, `src/backend/cpu/eltwise.cpp`, `src/backend/cpu/reference/eltwise_ref.cpp`, `src/backend/cuda/eltwise.cu`
- **Types:** Add, Sub, Mul, Div
- **Attributes:** `EltwiseAttributes` — type, add_to
- **Input (2):** A `[*]`, B `[*]` (same shape and dtype)
- **Output:** C `[*]` (same shape and dtype)
- **Formula:** `C[i] = A[i] op B[i]` for op ∈ {+, −, ×, ÷}
- **SIMD Kernel** (`src/backend/cpu/eltwise.cpp`): Row-by-row pitch-aware processing with `simd_lane_for<T>` (8) wide SIMD + scalar tail. Uses generic API (`v_load`/`v_store`/`s_load`/`s_store`) to support both float and half in one code path. All 4 operations vectorized. Branch on op type hoisted outside row loop; add_to checked inside loop body.
- **Data types:** f32 and f16 (templated `eltwise_impl<T>`, dtype dispatch at entry)
- **CUDA Kernel** ([eltwise.cu](src/backend/cuda/eltwise.cu)): Flat grid-stride loop, single kernel with `switch` on `EltwiseType` containing the inner loop for each operation. Supports add_to.
- **Added:** 2026-07-24

## Unary

- **Files:** `include/nnops/ops/unary.hpp`, `src/ops/unary.cpp`, `src/backend/cpu/unary.cpp`, `src/backend/cpu/reference/unary_ref.cpp`, `src/backend/cuda/unary.cu`
- **Types:** Exp, Log, Sin, Cos, Tan, Tanh, Abs, Neg, Sqrt
- **Attributes:** `UnaryAttributes` — type (default: Exp), add_to
- **Input (1):** X `[*]` (any shape)
- **Output:** Y `[*]` (same shape and dtype)
- **Formula:** `Y[i] = op(X[i])` for op ∈ {exp, ln, sin, cos, tan, tanh, abs, −, sqrt}
- **SIMD Kernel** (`src/backend/cpu/unary.cpp`): Row-by-row pitch-aware processing with `simd_lane_for<T>` (8) wide SIMD + scalar tail. Uses generic API (`v_load`/`v_store`/`s_load`/`s_store`) to support both float and half. All 9 operations vectorized — transcendental ops (exp/log/sin/cos/tan/tanh) use SIMD intrinsics; algebraic ops (abs/neg/sqrt) use direct SIMD instructions. Branch on op type hoisted outside row loop; add_to checked inside loop body.
- **Data types:** f32 and f16 (templated `unary_impl<T>`, dtype dispatch at entry)
- **CUDA Kernel** ([unary.cu](src/backend/cuda/unary.cu)): Device function pointer dispatch via `get_unary_fn(UnaryType)` — returns `float (*)(float)` for each operation. Two launch strategies: `unary_flat_kernel` (grid-stride loop for rank ≤ 1) and `unary_kernel` (row-by-row pitch-aware for rank ≥ 2). Supports add_to.
- **Added:** 2026-07-24 (CPU SIMD + CUDA); 2026-07-20 (CPU reference)

## Activation
- **Types:** Relu, LeakyRelu, Sigmoid, Tanh, Gelu, Silu, HardSwish, Elu
- **Attributes:** `ActivationAttributes` — type, alpha, beta
- **Data types:** f32 (v_f32x8) and f16 (v_f16x8) via single templated implementation `activation_impl<T>`
- **SIMD Kernel** (`src/backend/cpu/activation.cpp`): row-by-row pitch-aware processing with `simd_lane_for<T>` (8) wide SIMD + scalar tail. Uses generic API (`v_load`/`v_store`/`v_set1`/`s_load`/`s_store`) to support both float and half in one code path. All 8 activation types vectorized.
- **Design Decisions:**
  1. 无需 blend/select — 所有条件分支通过 v_max/v_min 分解消除：LeakyRelu = `max(x,0) + α*min(x,0)`，ELU = `max(x,0) + α*(exp(min(x,0))-1)`，Relu = `max(x,0)`
  2. `simd_lane_for<T>` (f32→8, f16→8) — SIMD width selected at compile time by data type, all SIMD backends support lane=8
  3. Row-by-row processing via `row_stride_elems()` respects pitch padding — no contiguous-tensor assumption
  4. add_to 支持：load + accumulate + store 替代 plain store
- **Dispatch:** `Backend::CPU` → `activation_cpu` → `activation_impl<float>` / `activation_impl<half>`（SIMD 优化），reference 保留用于正确性基线
- **Note:** Standalone operator — distinct from [[#Epilogue (Core Type)]] which fuses activation into Conv/MatMul output write-back.
- **Updated:** 2026-07-22 — templated for f16 support, pitch-aware rows, `simd_lane_for<T>`

## Pooling

- **Files:** `include/nnops/ops/pooling.hpp`, `src/ops/pooling.cpp`, `src/backend/cpu/reference/pooling_ref.cpp`, `src/backend/cuda/pooling.cu`
- **Types:** Max, Average, AverageExcludePad, Lp
- **Attributes:** `PoolingAttributes` — `std::array<int64_t, 3>` for kernel/stride/padding/dilation (layout: `[KD, KH, KW]`)
- **Input:** 2D `[N, C, IH, IW]` or 3D `[N, C, ID, IH, IW]`
- **Note:** Upgraded from 2D-only to 2D/3D on 2026-07-20. Spatial rank auto-detected from input tensor rank (4→2D, 5→3D).

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
- **Attributes:** `AttentionAttributes` — num_heads, scale (0=auto `1/sqrt(d)`), use_causal_mask
- **Input:** Q/K/V — merged `[B, S, H*D]` or explicit `[B, H, S, D]`, optional mask
- **CPU Reference:** Per-head parallel → QK^T → scale → mask → softmax → weighted V sum
- **Added:** 2026-07-20

## BatchNorm

- **Files:** `include/nnops/ops/batch_norm.hpp`, `src/ops/batch_norm.cpp`, `src/backend/cpu/batch_norm.cpp`, `src/backend/cpu/reference/batch_norm_ref.cpp`, `src/backend/cuda/batch_norm.cu`
- **Attributes:** `BatchNormAttributes` — epsilon (1e-5), spatial (true=per-channel shape [C], false=per-element), add_to
- **Input (5):** X `[N, C, D1...]`, scale `[C]`, bias `[C]`, mean `[C]`, var `[C]`
- **Output:** Y `[N, C, D1...]`
- **Formula (fused):** `y = x * (inv_std * scale) + (bias - mean * inv_std * scale)`
- **SIMD Kernel** (`src/backend/cpu/batch_norm.cpp`): Templated `batch_norm_impl<T>` using generic SIMD API (`v_load`/`v_store`/`v_set1`/`s_load`/`s_store`). Two-code paths: spatial mode (row-by-row SIMD over spatial dims) and non-spatial mode (flat vectorized). Pitch-aware via `row_stride_elems()` / `stride_elems()`.
- **Data types:** f32 and f16 via dtype dispatch at entry point
- **Updated:** 2026-07-22 — templated for fp16 support with generic SIMD API, pitch-aware strides

## LayerNorm

- **Files:** `include/nnops/ops/layer_norm.hpp`, `src/ops/layer_norm.cpp`, `src/backend/cpu/layer_norm.cpp`, `src/backend/cpu/reference/layer_norm_ref.cpp`, `src/backend/cuda/layer_norm.cu`
- **Attributes:** `LayerNormAttributes` — axis (-1), epsilon (1e-5), add_to
- **Input (2-3):** X `[*]`, scale `[norm_shape]`, optional bias `[norm_shape]`
- **Output:** Y `[*]`
- **Formula:** `y = (x - mean) / sqrt(var + epsilon) * scale + bias`
- **Algorithm:** Two-pass SIMD: Pass 1 — SIMD reduction (sum, sum_sq) using typed SIMD API, then scalar tail uses Welford's online algorithm for numerical stability, merged via parallel Welford formula (Chan et al.); Pass 2 — normalize with scale/bias using typed SIMD + scalar tail. Fast path for contiguous tail (axis == rank-1, the 99% LLM case); general axis falls back to full Welford scalar pass with pre-computed inner offsets.
- **SIMD Kernel** (`src/backend/cpu/layer_norm.cpp`): Templated `layer_norm_impl<T>`, dtype dispatch at entry. Both passes use the generic typed API. Parallel dispatch over num_rows via `ctx.cpu.run`. Pitch-aware via `row_stride_elems()` / `stride_elems()`.
- **Data types:** f32 and f16
- **Reference:** onnxruntime `MlasLayerNormF32` / `ComputeJobGenericShared`
- **Added:** 2026-07-22

## RMSNorm

- **Files:** `include/nnops/ops/rms_norm.hpp`, `src/ops/rms_norm.cpp`, `src/backend/cpu/rms_norm.cpp`, `src/backend/cpu/reference/rms_norm_ref.cpp`, `src/backend/cuda/rms_norm.cu`
- **Attributes:** `RMSNormAttributes` — axis (-1), epsilon (1e-5), add_to
- **Input (2):** X `[*]`, scale `[norm_shape]`
- **Output:** Y `[*]`
- **Formula:** `rms = sqrt(mean(x^2) + epsilon)`, `y = x / rms * scale`
- **Note:** Unlike LayerNorm, no mean subtraction and no bias.
- **Algorithm:** Two-pass SIMD (same pattern as LayerNorm): Pass 1 — SIMD reduction of sum_sq; Pass 2 — apply `x * inv_rms * scale`. Fast path for axis == rank-1, general scalar fallback for other axes.
- **SIMD Kernel** (`src/backend/cpu/rms_norm.cpp`): Templated `rms_norm_impl<T>`, dtype dispatch. Same SIMD patterns as LayerNorm.
- **Data types:** f32 and f16
- **Reference:** onnxruntime `ComputeJob` with `simplified=true`
- **Added:** 2026-07-22

## Softmax

- **Files:** `include/nnops/ops/softmax.hpp`, `src/ops/softmax.cpp`, `src/backend/cpu/softmax.cpp`, `src/backend/cpu/reference/softmax_ref.cpp`, `src/backend/cuda/softmax.cu`
- **Attributes:** `SoftmaxAttributes` — axis (-1), log_softmax (false)
- **Input (1):** X `[*]` (any rank >= 1)
- **Output:** Y `[*]` (same shape as input)
- **Formula:** `softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))`, `log_softmax(x_i) = (x_i - max) - log(sum(exp(x_j - max)))`
- **Algorithm (per row, mirrors onnxruntime):** Three-stage pipeline — (1) ReduceMax: SIMD `v_max` reduction + scalar tail → max_val; (2) ComputeSumExp: SIMD `v_exp` on `(x - max)` + reduce sum, with exp values stored to output for reuse (avoids recomputing exp in normalization pass); (3) Normalize: softmax = exp/sum via SIMD multiply, or log_softmax = `(x - max) - log(sum)`. Fast path for axis == rank-1 (contiguous last dim, the 99% LLM attention case); general axis falls back to scalar path with pre-computed inner offsets.
- **SIMD Kernel** (`src/backend/cpu/softmax.cpp`): Templated `softmax_impl<T>`, dtype dispatch at entry. All three passes use the generic typed API (`v_load`/`v_store`/`v_max`/`v_exp`/`v_add`/`v_mul`/`v_set1`/`v_reduce_sum`/`s_load`/`s_store`). `reduce_max_vec` helper reduces SIMD max vector to scalar via temp store+scan (the SIMD layer has `v_reduce_sum` but no `v_reduce_max`). Parallel dispatch over num_rows via `ctx.cpu.run`. Pitch-aware via `row_stride_elems()` / `stride_elems()`.
- **Data types:** f32 and f16
- **Reference:** onnxruntime `MlasComputeSoftmax` / `MlasReduceMaximumF32Kernel` / `MlasComputeSumExpF32Kernel` / `MlasComputeSoftmaxOutputF32Kernel`
- **Added:** 2026-07-22

## CumSum

- **Files:** `include/nnops/ops/cumsum.hpp`, `src/ops/cumsum.cpp`, `src/backend/cpu/reference/cumsum_ref.cpp`, `src/backend/cuda/cumsum.cu`
- **Attributes:** `CumSumAttributes` — exclusive (false), reverse (false), axis (0)
- **Input (1):** X `[*]` (any rank >= 1)
- **Output:** Y `[*]` (same shape as input)
- **Formula:** Inclusive: `output[i] = sum(input[0..i])`, Exclusive: `output[0] = 0, output[i] = sum(input[0..i-1])`. Reverse: sum from last element backward.
- **Algorithm:** Tensor decomposed along axis into upper/lower dim groups. Each independent scan vector runs sequentially along axis. CPU reference uses `ctx.cpu.run` over upper slices. CUDA kernel uses one block per upper slice, one thread per lower-dim position, sequential scan per thread.
- **Data types:** f32 and f16 (CPU reference f32 only; CUDA supports both)
- **add_to:** Removed 2026-07-25 — this operator no longer supports output accumulation.
- **Added:** 2026-07-20 (CPU reference), 2026-07-23 (CUDA)

## Reduce

- **Files:** `include/nnops/ops/reduce.hpp`, `src/ops/reduce.cpp`, `src/backend/cpu/reduce.cpp`, `src/backend/cpu/reference/reduce_ref.cpp`, `src/backend/cuda/reduce.cu`
- **Types:** Sum, Min, Max, Mean
- **Attributes:** `ReduceAttributes` — type (default: Sum), axis (int64_t, default: 0, negative wraps from end), keepdims (false)
- **Input (1):** X `[*]` (any rank >= 1)
- **Output:** Y `[*]` (axis collapsed, or size-1 if keepdims)
- **Formula:** Sum=Σx, Mean=Σx/N, Max=max(x), Min=min(x) over specified axis
- **Single-axis only.** No multi-axis or reduce-all support (2026-07-25 simplification).
- **CPU SIMD** (`src/backend/cpu/reduce.cpp`): Fast path for contiguous tail (axis == rank-1). Sum/Mean via `v_reduce_sum`; Max/Min via `v_max`/`v_min` + store-scan. Non-contiguous axis → reference. Templated for f32/f16.
- **CPU Reference** (`src/backend/cpu/reference/reduce_ref.cpp`): Outer/inner loop decomposition with axis stride. Handles any single axis including non-contiguous.
- **CUDA** ([reduce.cu](src/backend/cuda/reduce.cu)): Single-axis, two-kernel pattern (fast contiguous + general with pre-computed offsets). Reuses `block_reduce_sum`/`block_reduce_max` plus `block_reduce_min` in [cuda_common.cuh](src/backend/cuda/cuda_common.cuh). Shared memory: block_size × sizeof(float). Mean = sum/N computed once per block.
- **Data types:** f32 and f16 (SIMD + CUDA); f32 only (reference)
- **Added:** 2026-07-25

## Operator Implementation Pattern

Every operator follows this recipe:

1. **Header** (`include/nnops/ops/<name>.hpp`): Attributes struct + Op class (inherits OpBase) + functional free function declarations. Class includes `using OpBase::compute;` to expose single-output convenience overload alongside the span-based override.
2. **Dispatch** (`src/ops/<name>.cpp`): `create()` factory, `compute()` with `switch(backend_)`, functional wrappers. Dispatch layer asserts `outputs.size() == 1` and extracts `outputs[0]` before calling backend kernels (which keep `TensorView& output` parameter).
3. **CPU Reference** (`src/backend/cpu/reference/<name>_ref.cpp`): Naive implementation in `nnops::backend::cpu::reference` namespace
4. **Tests** (`tests/test_<name>.cpp`): Hand-verified small tests + random data tests + class/functional parity tests

### Multi-Output Support (2026-07-22)

`OpBase::compute()` primary virtual takes `std::span<TensorView> outputs` (mutable, matching the span-based inputs). A non-virtual convenience overload `compute(TensorView& output, ...)` delegates to the span version for single-output callers. Output tensors are writable — only input tensors are `const`. Kernel functions receive `outputs[0]` extracted by the dispatch layer. Future multi-output operators (Split, TopK, etc.) can iterate `outputs[i]` directly.

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
- **x86_64**: Data stored as `half` (uint16_t). F16C `_mm256_cvtph_ps`/`_mm256_cvtps_ph` for convert, fp32 FMA accumulators, clamp in fp32 then round-to-nearest to fp16 on store
- **AArch64**: Native NEON fp16 arithmetic (`float16_t` type, `vfmaq_lane_f16` intrinsics) — requires ARMv8.2-A+

**Shared utilities:**
- `src/backend/cpu/common/restrict.hpp` — cross-platform `NNOPS_RESTRICT` macro (MSVC `__restrict`, GCC/Clang `__restrict__`)
- `include/nnops/detail/simd/half.hpp` — `half` struct (uint16_t storage) + `half_to_float`/`float_to_half` converters for x86_64 fp16 kernels

**Porting reference:** `D:\vscode\nn_compute` — all 10 kernel files ported with modern C++ improvements (noexcept, constexpr, lambda helpers, Doxygen). See [[reference-projects]].

**Added:** 2026-07-21 (aarch64) and 2026-07-21 (x86_64)

## add_to (Output Accumulation)

Most operators support `bool add_to{false}` in their Attributes. When true, the kernel adds its result to the existing output buffer instead of overwriting.
**Exceptions (removed 2026-07-25):** Softmax and CumSum no longer support add_to — their semantics don't benefit from output accumulation.

```
add_to=false (default):  output[i]  = result
add_to=true:             output[i] += result
```

**Purpose:** Enables residual connections and skip connections without a separate add kernel. For example, `Conv(input, output, {.add_to=true})` performs `output += Conv(input)` in a single kernel launch.

**Implementation pattern:**
- Scalar: `out[i] = attrs.add_to ? out[i] + val : val;`
- SIMD: branch-hoisted outside the loop; add_to path does `store(y, add(load(y), computed))` (one extra load per vector)

**Added:** 2026-07-21
