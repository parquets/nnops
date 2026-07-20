---
name: operators
description: "List of implemented operators with their attributes, input/output specs, and implementation patterns"
metadata:
  node_type: memory
  type: project
  originSessionId: 2c7fb42c-b26c-4d7e-a6dd-2de30d713ad8
  modified: 2026-07-20T17:20:21.477Z
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

- **Files:** `include/nnops/ops/activation.hpp`, `src/ops/activation.cpp`, `src/backend/cpu/reference/activation_ref.cpp`
- **Types:** Relu, LeakyRelu, Sigmoid, Tanh, Gelu, Silu, HardSwish, Elu
- **Attributes:** `ActivationAttributes` — type, alpha, beta
- **Pattern:** Element-wise functor dispatched via `switch(ActivationType)`, parallel_for over flat range
- **Note:** Standalone operator — distinct from [[#Epilogue (Core Type)]] which fuses activation into Conv/MatMul output write-back.

## Pooling

- **Files:** `include/nnops/ops/pooling.hpp`, `src/ops/pooling.cpp`, `src/backend/cpu/reference/pooling_ref.cpp`
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

## Operator Implementation Pattern

Every operator follows this recipe:

1. **Header** (`include/nnops/ops/<name>.hpp`): Attributes struct + Op class (inherits OpBase) + functional free function declarations
2. **Dispatch** (`src/ops/<name>.cpp`): `create()` factory, `compute()` with `switch(backend_)`, functional wrappers
3. **CPU Reference** (`src/backend/cpu/reference/<name>_ref.cpp`): Naive implementation in `nnops::backend::cpu::reference` namespace
4. **Tests** (`tests/test_<name>.cpp`): Hand-verified small tests + random data tests + class/functional parity tests

## add_to (Output Accumulation)

All operators support `bool add_to{false}` in their Attributes. When true, the kernel adds its result to the existing output buffer instead of overwriting:

```
add_to=false (default):  output[i]  = result
add_to=true:             output[i] += result
```

**Purpose:** Enables residual connections and skip connections without a separate add kernel. For example, `Conv(input, output, {.add_to=true})` performs `output += Conv(input)` in a single kernel launch.

**Implementation pattern:**
- Scalar: `out[i] = attrs.add_to ? out[i] + val : val;`
- SIMD: branch-hoisted outside the loop; add_to path does `store(y, add(load(y), computed))` (one extra load per vector)

**Added:** 2026-07-21
