---
name: reference-projects
description: "Key patterns and insights from onnxruntime, TensorRT, ComputeLibrary, and OpenCV used to inform nnops design"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 2c7fb42c-b26c-4d7e-a6dd-2de30d713ad8
  modified: 2026-07-20T15:20:24.720Z
---

# Reference Project Insights

Four reference projects were studied before designing nnops. Their key patterns informed the architecture.

## onnxruntime

**Location:** `D:\git\onnxruntime`

Key patterns adopted:
- **OpKernel base class** → our `OpBase` (flat virtual base)
- **Attributes struct** per operator (e.g. `ConvAttributes`, `PoolAttributes`) → our `Conv2DAttributes`, `PoolingAttributes`
- **Registration macros** at bottom of .cc files — we simplified this to `create()` factory methods
- **Compute method** pattern: get inputs, validate, allocate output, run kernel
- **Element-wise ops** use functors with `operator()(first, last)` for parallelization

## TensorRT

**Location:** `D:\git\TensorRT`

Key patterns noted:
- **Plugin capability architecture** (IPluginV3: kCORE/kBUILD/kRUNTIME) — interesting but too complex for our needs; we use a simpler single-class approach
- **PluginTensorDesc** (dims, type, format) → inspired our `TensorView` with shape + data_type + layout
- **REGISTER_TENSORRT_PLUGIN** macro + static `PluginRegistrar` — similar intent to our operator registry (future feature)
- **Integration points** (`addPluginV2`/`addPluginV3`) — our operators are designed to be wrappable as TensorRT plugins

## ARM ComputeLibrary

**Location:** `D:\git\ComputeLibrary`

Key patterns adopted:
- **ITensorInfo / ITensor separation** (metadata vs data) → our `TensorView` is non-owning (like ITensorInfo + buffer pointer)
- **IOperator → INEOperator → ICpuOperator** hierarchy → simplified to our flat `OpBase`
- **Strategy pattern** for Conv2d (Winograd/GEMM/Direct dispatch) → we have `switch(backend_)` with future optimized-CPU dispatch
- **ISA dispatch** (SVE2 > SVE > NEON) via heuristics → our planned optimized CPU kernels will follow this pattern
- **Workspace memory** returned via `workspace()` method → our `getWorkspace()`
- **ITensorPack** mapping slot IDs to tensors → we use `std::span<const TensorView>` for simpler interface

## OpenCV

**Location:** `D:\git\opencv` (modules/core/include/opencv2/core/hal/)

Key patterns adopted for SIMD abstraction:
- **Universal Intrinsics** (`intrin.hpp` + arch-specific backends) → our `include/nnops/detail/simd/` layer
- **Type-safe vector types**: `v_float32x4` (128-bit), `v_float32x8` (256-bit) → our `VecF32x4`, `VecF32x8`
- **Per-ISA backend files**: `intrin_sse.hpp`, `intrin_avx.hpp`, `intrin_neon.hpp` → our `arch/x86/sse.hpp`, `arch/x86/avx2.hpp`, `arch/arm/neon.hpp`
- **Scalar C++ fallback**: `intrin_cpp.hpp` → our `arch/scalar.hpp` — always available, correctness baseline
- **Compile-time dispatch**: `#if CV_SSE2 ... #elif CV_NEON ... #else #include "intrin_cpp.hpp"` → our `vec_f32x4.hpp` / `vec_f32x8.hpp` use `#if NNOPS_ARCH_X86_64 ... #elif NNOPS_ARCH_AARCH64`
- **128-bit operations**: `v_load`, `v_store`, `v_add`, `v_mul`, `v_fma`, `v_min`, `v_max`, `v_sqrt`, `v_abs`, `v_neg` → our `vec_load_f32x4`, `vec_store_f32x4`, etc.
- **256-bit operations**: `v256_load`, `v256_fma`, etc. → our `vec_fmadd_f32x8`, etc.
- **FMA priority**: SSE does mul+add, AVX2 uses `_mm256_fmadd_ps`, NEON uses `vfmaq_f32` — same pattern in our code
- **NEON reciprocal/sqrt via Newton-Raphson**: `vrecpeq_f32` + `vrecpsq_f32` refinement → same in our `vec_rcp_f32x4` / `vec_sqrt_f32x4`

Also referenced onnxruntime's approach for runtime CPU detection:
- **CPUIDInfo singleton** (cpu_features.h/cc) → our `CpuFeatures::get()` singleton
- **Cascade dispatch**: SSE4.1 → AVX → AVX2+FMA → AVX512F → our `CpuIsa` enum bits

## Key Differences from Reference Projects

| Aspect | onnxruntime | ComputeLibrary | nnops (our choice) |
|---|---|---|---|
| Memory ownership | Tensor owns data | Tensor owns data | User owns data; TensorView is non-owning |
| Thread pool | Internal | Internal | External `parallel_for` hook |
| Dependencies | Eigen, MLAS, etc. | None (self-contained) | None (self-contained) |
| Type dispatch | Runtime macros | Templates + heuristics | `switch` + functors (simple) |
| Backend model | ExecutionProvider plugins | CPU/GPU split in src | `switch(backend_)` in compute() |
