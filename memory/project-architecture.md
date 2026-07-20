---
name: project-architecture
description: Overall architecture and design decisions for the nnops multi-backend operator library
metadata: 
  node_type: memory
  type: project
  originSessionId: 2c7fb42c-b26c-4d7e-a6dd-2de30d713ad8
  modified: 2026-07-20T17:02:55.382Z
---

# nnops Project Architecture

nnops is a C++23 multi-backend neural network operator library with zero third-party dependencies.

**Why:** The project aims to provide high-performance, self-contained compute operators that can be easily integrated into onnxruntime and TensorRT as custom ops, while supporting CPU (x86_64/aarch64), CUDA, and Vulkan backends.

**How to apply:**
- Namespace: `nnops`
- All operator headers go in `include/nnops/ops/`
- All backend kernels go in `src/backend/<backend>/`
- Operator dispatch is in `src/ops/` — one .cpp per operator
- No internal memory allocation — user owns all buffers
- No internal thread pool — uses external `ParallelForFn` via `ComputeContext`
- SIMD abstraction layer (SSE/AVX2/NEON/scalar) for cross-platform vectorized kernels

## Key Design Decisions

| Decision | Choice |
|---|---|
| OpBase hierarchy | Flat virtual base (pure virtual) |
| TensorView storage | `SmallVector<int64_t, 8>` for shape/stride (zero heap for <= 8D) |
| Backend dispatch | `switch(backend_)` in compute() |
| CPU parallelism | `std::function` passed via `ComputeContext::cpu_parallel_for` |
| Memory | User owns all buffers; workspace passed as `void*` |
| Functional API | Free function wrapping create+compute |
| Build | Modern CMake, C++23, explicit source files, no `file(GLOB)` |
| Tests | Self-contained harness, XorShift128 RNG, `allclose()` comparison |

## Directory Structure

```
include/nnops/core/     — DataType, TensorView, OpBase, Backend, ComputeContext, Epilogue
include/nnops/ops/      — Operator headers (Conv2D, Conv3D, Activation, Pooling, Linear, MatMul, Attention)
include/nnops/detail/   — SmallVector, SIMD abstraction layer, assertions
src/ops/                — Operator dispatch (one .cpp per op)
src/detail/             — CPU feature detection implementation
src/backend/cpu/reference/ — Naive CPU kernels (correctness baseline)
src/backend/cuda/       — CUDA kernels (optional)
src/backend/vulkan/     — Vulkan kernels (optional)
tests/                  — Unit tests + common test utilities
```
