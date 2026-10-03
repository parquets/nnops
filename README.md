# nnops

A self-contained, **zero-dependency** C++23 operator library for neural-network inference, with optimized backends for CPU (x86_64 / aarch64), CUDA, and Vulkan.

`nnops` is designed to be dropped into inference runtimes (onnxruntime custom ops, TensorRT plugins, lightweight embedded engines) as the compute layer. It gives you a small, explicit API over `TensorView` buffers with **no internal memory allocation and no internal thread pool** — the caller owns all buffers and injects its own `parallel_for` through `ComputeContext`.

## Highlights

- **Zero third-party dependencies** on the CPU path (only the C++ standard library).
- **Multi-backend dispatch** — `Backend::CPU` (x86_64 / aarch64), `Backend::CUDA`, `Backend::Vulkan` — selected per operator at creation.
- **SIMD kernels** for the hot path: AVX2 + FMA + F16C + AVX-VNNI on x86_64, NEON (with native fp16) on aarch64, with a scalar fallback and naive *reference* kernels kept as a correctness baseline.
- **Fusion-friendly** — an `Epilogue` (activation / quantization) fused into Conv/MatMul/Linear write-back, and an `add_to` flag that turns every elementwise-ish op into a residual accumulate.
- **Explicit memory model** — a non-owning `TensorView` with a pitch (row-stride) layout, so padded/aligned buffers are first-class.
- **~30 operators** covering convolution, GEMM, attention, normalization, element-wise math, reductions, and tensor layout ops.

## Backends and data types

| Backend | Status | Notes |
|---|---|---|
| CPU x86_64 | ✅ default | Requires SSE4.1 / AVX2 / FMA3 / F16C / AVX-VNNI |
| CPU aarch64 | ✅ | NEON; ARMv8.2+ for native fp16 kernels |
| CUDA | ⚙️ optional | `-DNNOPS_BUILD_CUDA=ON` |
| Vulkan | ⚙️ optional | `-DNNOPS_BUILD_VULKAN=ON` (eltwise + unary) |

`DataType` covers `f32`, `f16`, `bf16`, `s8`, `u8`, `s32`, `s64`, `f8_e4m3`, `f8_e5m2`, `s4`, `u4`. The optimized SIMD kernels target `f32` and `f16`; `s8×s8 → s32/s8` integer GEMM is available for `MatMul` / `QuantLinear`.

## Requirements

- CMake ≥ 3.21
- A C++23 compiler (MSVC 2022, GCC ≥ 12, Clang ≥ 16)
- Optional: CUDA Toolkit (for the CUDA backend), Vulkan SDK (for the Vulkan backend)

## Building

```bash
# CPU-only (default)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# With CUDA
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNNOPS_BUILD_CUDA=ON

# With Vulkan (requires the Vulkan SDK's glslc)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNNOPS_BUILD_VULKAN=ON
```

| Option | Default | Meaning |
|---|---|---|
| `NNOPS_BUILD_TESTS` | `ON` | Build the unit-test demos |
| `NNOPS_BUILD_BENCHMARKS` | `ON` | Build the micro-benchmark binary |
| `NNOPS_BUILD_CUDA` | `OFF` | Build the CUDA backend |
| `NNOPS_BUILD_VULKAN` | `OFF` | Build the Vulkan backend |

Install as a CMake package:

```bash
cmake --install build --config Release --prefix <prefix>
# targets: nnops::nnops (interface), nnops::cpu, nnops::cuda, nnops::vulkan
```

## Quick start

Operators use a class API: `Op::create(attrs, backend)` → `getOutputTensorDesc(...)` → `compute(...)`.

```cpp
#include "nnops/nnops.hpp"

using namespace nnops;

int main() {
    // C = A × B   (A: [2,3], B: [3,2] → C: [2,2])
    float a_data[6] = {1, 2, 3, 4, 5, 6};
    float b_data[6] = {1, 0, 0, 1, 1, 1};
    float c_data[4] = {0, 0, 0, 0};

    const int64_t ashape[] = {2, 3};
    const int64_t bshape[] = {3, 2};
    const int64_t cshape[] = {2, 2};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);
    TensorView c(cshape, DataType::f32, c_data);

    auto op = MatMul::create(/*attrs=*/{}, Backend::CPU);
    const TensorView inputs[] = {a, b};
    op->compute(c, inputs);

    // c_data = [[4, 5], [10, 11]]
}
```

Two patterns worth knowing:

- **Injected parallelism** — pass your own `parallel_for` (and optional `num_threads`/`thread_id` for per-worker scratch) via `ComputeContext::cpu`. If omitted, operators run single-threaded.
- **Workspace** — operators that need scratch report their requirement through `getWorkspaceSize(inputs, outputs)`; you allocate the buffer and pass it as `compute`'s `workspace` argument. Allocation never happens inside the library.

## Operators

| Category | Operators |
|---|---|
| Convolution | `Conv2D`, `Conv3D`, `DepthwiseConv2D`, `TransposeConv2D` |
| Matrix / linear | `MatMul` (f32 / f16 / s8×s8), `Linear`, `QuantLinear` |
| Attention | `Attention`, `CausalAttention`, `LinearAttention` |
| Normalization | `LayerNorm`, `RMSNorm`, `BatchNorm`, `Softmax` (softmax & log-softmax) |
| Element-wise | `Eltwise` (add/sub/mul/div), `Unary` (exp/log/sin/cos/tan/tanh/abs/neg/sqrt), `Activation` (relu/leaky_relu/sigmoid/tanh/gelu/silu/hard_swish/elu), `Clamp` |
| Reduction | `Reduce` (sum/mean/max/min), `ArgMinMax`, `TopK`, `CumSum` |
| Shape / layout | `Concat`, `Slice`, `Permute`, `Flatten`, `Embed`, `Resize`, `GridSample`, `LayoutConvert` |
| Positional | `RoPE` (rotary position embeddings) |

### Fusion and accumulation

- **Epilogue** — Conv/MatMul/Linear accept an `Epilogue` (ReLU, GELU, SiLU, …) applied during output write-back, fusing the activation into the kernel with no extra launch.
- **`add_to`** — most operators take `add_to{false}`; when `true`, the result is *added* to the existing output buffer instead of overwriting it, so residual/skip connections need no separate add kernel.

## Design overview

```
include/nnops/core/     DataType, TensorView, OpBase, Backend, ComputeContext, Epilogue
include/nnops/ops/      Operator headers (one per op)
src/ops/                Dispatch layer — create() factory + backend switch
src/backend/cpu/        SIMD kernels, reference kernels, tiled GEMM/Conv/Attention
  ├── x86_64/           AVX2/FMA/F16C/VNNI micro-kernels (pack, transpose, MMA)
  ├── aarch64/          NEON micro-kernels
  └── simd_kernel/      cross-platform SIMD primitives
src/backend/cuda/       CUDA kernels (optional)
src/backend/vulkan/     Vulkan kernels (optional)
tests/                  Standalone test demos (one executable per file)
```

Key design choices:

- `TensorView` is a **non-owning view** — shape stored inline (zero heap for rank ≤ 8), with a byte `pitch` between rows (OpenCV `cv::Mat::step` style). Packed layouts (`NCHWC8`, …) and padded rows are supported.
- Backend kernels keep a `TensorView&` and are dispatched through `switch (backend_)`.
- The CPU SIMD abstraction (`v_load` / `v_store` / `v_fmadd` / `v_reduce_sum` …) lets a single templated kernel target f32 and f16 across SSE/AVX2/NEON/scalar.

## Testing

Every `tests/test_*.cpp` builds into its own runnable demo (a `main()` returning its failed-test count) and is registered with CTest.

```bash
# Run all tests
ctest --test-dir build -C Release --output-on-failure

# or, via the aggregation script
scripts/run_tests.sh Release
```

Tests use a self-contained harness (XorShift128 RNG, `allclose` comparison) — no external test framework.

## Benchmarks

```bash
# Windows (multi-config)
./build/tests/Release/nnops_bench.exe
# POSIX
./build/tests/nnops_bench
```

Micro-benchmarks for `MatMul`, `Conv2D`, and `Attention` (including depthwise and fused attention kernels).
