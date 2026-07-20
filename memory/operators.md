---
name: operators
description: "List of implemented operators with their attributes, input/output specs, and implementation patterns"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2c7fb42c-b26c-4d7e-a6dd-2de30d713ad8
  modified: 2026-07-20T15:07:03.002Z
---

# Implemented Operators

All operators support: class API (`Op::create()` → `op->compute()`) and functional API (free function).
Reference project patterns from onnxruntime (OpKernel + attributes struct), ComputeLibrary (IOperator + kernel dispatch), and TensorRT (plugin capability interfaces).

## Conv2D

- **Files:** `include/nnops/ops/conv2d.hpp`, `src/ops/conv2d.cpp`, `src/backend/cpu/reference/conv2d_ref.cpp`
- **Attributes:** `Conv2DAttributes` — stride[2], dilation[2], padding[2], groups, auto_pad
- **Input:** `[N, IC, IH, IW]`, weight `[OC, IC/G, KH, KW]`, optional bias `[OC]`
- **Output:** `[N, OC, OH, OW]`
- **CPU Reference:** 7-level nested loop, per-sample parallel via `parallel_for`

See also [[pooling-upgrade]] for the 2D→3D pattern that could apply to Conv3D in the future.

## Activation

- **Files:** `include/nnops/ops/activation.hpp`, `src/ops/activation.cpp`, `src/backend/cpu/reference/activation_ref.cpp`
- **Types:** Relu, LeakyRelu, Sigmoid, Tanh, Gelu, Silu, HardSwish, Elu
- **Attributes:** `ActivationAttributes` — type, alpha, beta
- **Pattern:** Element-wise functor dispatched via `switch(ActivationType)`, parallel_for over flat range

## Pooling

- **Files:** `include/nnops/ops/pooling.hpp`, `src/ops/pooling.cpp`, `src/backend/cpu/reference/pooling_ref.cpp`
- **Types:** Max, Average, AverageExcludePad, Lp
- **Attributes:** `PoolingAttributes` — `std::array<int64_t, 3>` for kernel/stride/padding/dilation (layout: `[KD, KH, KW]`)
- **Input:** 2D `[N, C, IH, IW]` or 3D `[N, C, ID, IH, IW]`
- **Note:** Upgraded from 2D-only to 2D/3D on 2026-07-20. Spatial rank auto-detected from input tensor rank (4→2D, 5→3D).

## Linear

- **Files:** `include/nnops/ops/linear.hpp`, `src/ops/linear.cpp`, `src/backend/cpu/reference/linear_ref.cpp`
- **Formula:** `output = input × weight^T + bias`
- **Shapes:** input `[M, K]`, weight `[N, K]`, bias `[N]`, output `[M, N]`
- **CPU Reference:** Triple-nested GEMM, per-row parallel

## MatMul

- **Files:** `include/nnops/ops/matmul.hpp`, `src/ops/matmul.cpp`, `src/backend/cpu/reference/matmul_ref.cpp`
- **Attributes:** `MatMulAttributes` — transpose_a, transpose_b
- **CPU Reference:** Triple-nested GEMM with transpose handling, per-row parallel

## Attention

- **Files:** `include/nnops/ops/attention.hpp`, `src/ops/attention.cpp`, `src/backend/cpu/reference/attention_ref.cpp`
- **Attributes:** `AttentionAttributes` — num_heads, scale (0=auto `1/sqrt(d)`), use_causal_mask
- **Input:** Q/K/V — merged `[B, S, H*D]` or explicit `[B, H, S, D]`, optional mask
- **CPU Reference:** Per-head parallel → QK^T → scale → mask → softmax → weighted V sum
- **Added:** 2026-07-20

## Softmax

- **Files:** `include/nnops/ops/softmax.hpp`, `src/ops/softmax.cpp`, `src/backend/cpu/reference/softmax_ref.cpp`
- **Attributes:** `SoftmaxAttributes` — axis (int64_t, default -1), log_softmax (bool)
- **Input:** `[*]` (any rank >= 1)
- **Output:** `[*]` same shape as input
- **Numerical:** max-subtraction before exp for stability; supports log-softmax mode
- **Added:** 2026-07-21

## CumSum

- **Files:** `include/nnops/ops/cumsum.hpp`, `src/ops/cumsum.cpp`, `src/backend/cpu/reference/cumsum_ref.cpp`
- **Attributes:** `CumSumAttributes` — exclusive (bool), reverse (bool), axis (int64_t, default 0)
- **Input:** `[*]` (rank >= 1)
- **Output:** `[*]` same shape as input
- **Algorithm:** Decomposes tensor into upper/lower dims along axis; recurrence `out[i] = in[i-1] + out[i-1]` (exclusive) or `out[i] = in[i] + out[i-1]` (inclusive)
- **Added:** 2026-07-21

## BatchNorm

- **Files:** `include/nnops/ops/batch_norm.hpp`, `src/ops/batch_norm.cpp`, `src/backend/cpu/reference/batch_norm_ref.cpp`
- **Attributes:** `BatchNormAttributes` — epsilon (float, 1e-5), spatial (bool, true)
- **Input:** X `[N,C,*]`, scale `[C]`, bias `[C]`, mean `[C]`, var `[C]`
- **Output:** Y `[N,C,*]`
- **Inference only:** No training mode. Fused formula: `y = x * (inv_std * scale) + (bias - mean * inv_std * scale)`
- **Added:** 2026-07-21

## LayerNorm

- **Files:** `include/nnops/ops/layer_norm.hpp`, `src/ops/layer_norm.cpp`, `src/backend/cpu/reference/layer_norm_ref.cpp`
- **Attributes:** `LayerNormAttributes` — axis (int64_t, default -1), epsilon (float, 1e-5)
- **Input:** X `[*]`, scale broadcastable to `X.shape[axis:]`, optional bias
- **Output:** Y `[*]`
- **Algorithm:** Welford's online algorithm for numerically stable mean/variance; normalizes over `X.shape[axis:]`
- **Added:** 2026-07-21

## RMSNorm

- **Files:** `include/nnops/ops/rms_norm.hpp`, `src/ops/rms_norm.cpp`, `src/backend/cpu/reference/rms_norm_ref.cpp`
- **Attributes:** `RMSNormAttributes` — axis (int64_t, default -1), epsilon (float, 1e-5)
- **Input:** X `[*]`, scale broadcastable to `X.shape[axis:]`
- **Output:** Y `[*]`
- **Formula:** `y = x / sqrt(mean(x^2) + eps) * scale` — no mean subtraction, no bias
- **Added:** 2026-07-21

## Operator Implementation Pattern

Every operator follows this recipe:

1. **Header** (`include/nnops/ops/<name>.hpp`): Attributes struct + Op class (inherits OpBase) + functional free function declarations
2. **Dispatch** (`src/ops/<name>.cpp`): `create()` factory, `compute()` with `switch(backend_)`, functional wrappers
3. **CPU Reference** (`src/backend/cpu/reference/<name>_ref.cpp`): Naive implementation in `nnops::backend::cpu::reference` namespace
4. **Tests** (`tests/test_<name>.cpp`): Hand-verified small tests + random data tests + class/functional parity tests
