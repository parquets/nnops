---
name: project-architecture
description: Overall architecture and design decisions for the nnops multi-backend operator library
metadata: 
  node_type: memory
  type: project
  originSessionId: 2c7fb42c-b26c-4d7e-a6dd-2de30d713ad8
  modified: 2026-07-22T16:20:48.367Z
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
| TensorView storage | `SmallVector<int64_t, 8>` for shape (zero heap for <= 8D); `int64_t pitch_` for row pitch in bytes |
| TensorView data access | `ptr<T>()` / `ptr<T>(row_id)` (typed, OpenCV-style) — replaces old `data()` and `data_as<T>()`; `is_empty()` for null checks |
| Backend dispatch | `switch(backend_)` in compute() |
| CPU parallelism | `CpuBackend` struct (parallel_for / num_threads / thread_id `std::function`s) embedded as `ComputeContext::cpu` |
| Memory | User owns all buffers; workspace passed as `void*` |
| Functional API | Free function wrapping create+compute |
| Build | Modern CMake, C++23, explicit source files, no `file(GLOB)` |
| Tests | Self-contained harness, XorShift128 RNG, `allclose()` comparison |

## Directory Structure

```
include/nnops/core/     — DataType, TensorView, OpBase, Backend, ComputeContext, Epilogue
include/nnops/ops/      — Operator headers (Conv2D, Conv3D, DepthwiseConv2D, Eltwise, Unary, Activation, Pooling, Linear, MatMul, Attention, BatchNorm, LayerNorm, RMSNorm, Softmax, CumSum)
include/nnops/detail/   — SmallVector, SIMD abstraction layer, assertions
src/ops/                — Operator dispatch (one .cpp per op)
src/detail/             — CPU feature detection implementation
src/backend/cpu/
  ├── common/           — Shared CPU utilities (restrict.hpp, half.hpp)
  ├── reference/        — Naive CPU kernels (correctness baseline, all operators)
  ├── x86_64/           — x86_64 GEMM micro-kernels (transpose.hpp, pack_f32/f16, mma_pack_f32/f16)
  │                       using AVX2+FMA for f32, F16C+FMA for fp16
  └── aarch64/          — AArch64 GEMM micro-kernels (transpose.hpp, pack_f32/f16, mma_pack_f32/f16)
                          using NEON for f32, NEON FP16 for fp16 (ARMv8.2+)
src/backend/cuda/       — CUDA kernels (optional)
src/backend/vulkan/     — Vulkan kernels (optional)
tests/                  — Unit tests (14 files, 135 tests) + common test utilities
```
