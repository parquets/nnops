---
name: cuda-infrastructure
description: "CUDA backend architecture, kernel design patterns, build integration, and operator coverage"
metadata:
  node_type: memory
  type: project
  modified: 2026-07-25
---

# CUDA Backend Infrastructure

The CUDA backend provides GPU-accelerated kernels for 10 operators, all hand-written from scratch with no cuBLAS or cuDNN dependency. Only `cuda_runtime.h` and `cuda_fp16.h` are required.

**Why:** Zero-dependency CUDA kernels keep the library self-contained and avoid version lock-in with NVIDIA's libraries. Hand-written kernels also allow consistent patterns across operators (pitch-aware access, add_to support, f16/f32 dispatch).

**How to apply:**
- All CUDA sources live in `src/backend/cuda/` — one `.cu` file per operator + `cuda_common.cuh`
- Dispatch is wired via `#ifdef NNOPS_HAS_CUDA` blocks in `src/ops/<name>.cpp`
- Every operator follows a two-layer entry pattern: `void <op>_cuda(...)` dispatches on dtype → `void <op>_cuda_impl<T>(...)` launches the kernel
- All arithmetic is done in `float` — f16 inputs are loaded via `s_load()` (convert to float), computed in float, stored via `s_store()` (convert back to half)
- Build is optional: `cmake -DNNOPS_BUILD_CUDA=ON`

## Architecture Overview

### Common Header (`cuda_common.cuh`)

Located at [src/backend/cuda/cuda_common.cuh](src/backend/cuda/cuda_common.cuh) (147 lines).

| Symbol | Value / Purpose |
|---|---|
| `kCudaBlockSize` | 256 (default threads per block) |
| `kCudaWarpSize` | 32 |
| `ceil_div(a, b)` | Integer ceiling division template |
| `half_to_float(h)` | Wrapper around `__half2float` |
| `float_to_half(f)` | Wrapper around `__float2half` |
| `s_load(ptr)` | Scalar load with auto f16→f32 conversion (mirrors CPU SIMD `s_load`) |
| `s_store(ptr, val)` | Scalar store with auto f32→f16 conversion (mirrors CPU SIMD `s_store`) |
| `warp_reduce_sum(val)` | Warp-level sum via `__shfl_down_sync` |
| `warp_reduce_max(val)` | Warp-level max via `__shfl_down_sync` |
| `block_reduce_sum(val)` | 2-phase: warp reduce → shared memory merge of warp leaders |
| `block_reduce_max(val)` | Same 2-phase pattern for max |
| `CUDA_CHECK(call)` | Error-checking macro — currently a no-op (placeholder) |

### Kernel Entry Point Pattern

Every `.cu` file follows this exact two-layer pattern:

```cpp
// Layer 1 — dtype dispatch (f32/f16 only)
void <op>_cuda(
    const <Op>Attributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx,
    void* /*workspace*/)   // <— workspace unused for all 10 operators
{
    switch (inputs[0].data_type()) {
    case DataType::F32: return <op>_cuda_impl<float>(...);
    case DataType::F16: return <op>_cuda_impl<__half>(...);
    default: NNOPS_ASSERT(false);
    }
}

// Layer 2 — extract dims/strides, configure grid/block, launch kernel
template <typename T>
void <op>_cuda_impl(...) {
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    dim3 grid = ...;
    dim3 block = min(actual_dim, kCudaBlockSize);
    <op>_kernel<<<grid, block, shared_bytes, stream>>>(...);
}
```

- Block size: `kCudaBlockSize = 256`, clamped to `min(actual_dim, 256)` when the dimension is smaller
- Grid size: capped at 65535 (pre-Volta max grid dimension)
- Stream: obtained from `ctx.cuda_stream` (passed as `void*`, cast to `cudaStream_t`)

### Device Function Pointer Dispatch

Two operators use device-side function pointer tables for compile-time variant selection:

- **Activation** ([activation.cu](src/backend/cuda/activation.cu)): `device_fn<ActivationType> get_activation_fn(TYPE)` returns function pointer to one of 8 math functions
- **Unary** ([unary.cu](src/backend/cuda/unary.cu)): `device_fn<UnaryType> get_unary_fn(TYPE)` returns function pointer to one of 9 math functions

The other operators use a `switch` statement inside the kernel (Eltwise, Pooling) or dedicated single-purpose kernels (Softmax, LayerNorm, RMSNorm, BatchNorm, CumSum, DepthwiseConv2D).

### Shared Memory Usage

Only three operators use shared memory for block-level reductions:

| Operator | Shared Memory per Block | Purpose |
|---|---|---|
| Softmax | `block_size * sizeof(float)` | ReduceMax + ComputeSum via warp→shared reduction |
| LayerNorm | `block_size * sizeof(WelfordStats)` | Welford partial merge (mean + M2) |
| RMSNorm | `block_size * sizeof(float)` | Sum-of-squares reduction |

Welford stats struct for LayerNorm: `{float mean, float m2, int count}` with Chan et al.'s parallel merge formula.

The other 7 operators (Activation, BatchNorm, Pooling, CumSum, DepthwiseConv2D, Eltwise, Unary) launch with 0 shared memory bytes.

### Temporary Device Allocations

Some operators allocate temporary device memory internally via `cudaMalloc`/`cudaFree`:

- **BatchNorm** (spatial mode): `d_new_scale` + `d_new_bias` (C floats each) — precomputed fused params
- **Softmax** (general axis): `d_offsets` (norm_size int64_t) — pre-computed inner offsets
- **LayerNorm** (general axis): Same offset pattern as Softmax
- **RMSNorm** (general axis): Same offset pattern

All use `cudaMalloc` + `cudaMemcpyAsync` on the operator's stream, then `cudaFree` after kernel completion.

### Fast Path vs General Path

Three normalization operators (Softmax, LayerNorm, RMSNorm) use a dual-kernel strategy:

- **Fast path kernel** (`<op>_kernel`): For axis == rank-1 (contiguous tail) — the 99% LLM case. No offset indirection needed.
- **General path kernel** (`<op>_general_kernel`): For arbitrary axis. Pre-computes inner offsets on host, copies to device, then uses them for indirect access.

Activation and Unary also use dual kernels:
- `<op>_flat_kernel`: Grid-stride loop for rank <= 1
- `<op>_kernel`: Row-by-row pitch-aware processing for rank >= 2

### Data Type Support

| Type | Native CUDA type | Compute precision |
|---|---|---|
| f32 | `float` | Native float |
| f16 | `__half` | Load as float, compute in float, store as half |

No native `__half` math is used — all arithmetic is in `float`, maintaining numerical precision at a performance cost. The `s_load`/`s_store` helpers encapsulate the conversion.

### Pitch-Aware Access

Activation and Unary kernels use `row_stride_elems()` for pitch-aware memory access when tensors have rank >= 2. This mirrors the CPU SIMD approach (see [[simd-infrastructure]] and [[pitch-design]]). Other operators use explicit stride arithmetic matching the tensor layout.

## Operator Inventory (10 operators)

| # | Operator | File | Lines | Key Features |
|---|---|---|---|---|
| 1 | Softmax | [softmax.cu](src/backend/cuda/softmax.cu) | 315 | 3-pass (max→exp+sum→norm), shared memory, fast/general paths |
| 2 | LayerNorm | [layer_norm.cu](src/backend/cuda/layer_norm.cu) | 309 | 2-pass Welford, shared memory, fast/general paths |
| 3 | RMSNorm | [rms_norm.cu](src/backend/cuda/rms_norm.cu) | 252 | 2-pass (sum_sq→norm), shared memory, fast/general paths |
| 4 | Activation | [activation.cu](src/backend/cuda/activation.cu) | 209 | Device fn pointer dispatch, flat/row kernels, 8 types |
| 5 | BatchNorm | [batch_norm.cu](src/backend/cuda/batch_norm.cu) | 211 | Inference-only, spatial/non-spatial modes, fused params |
| 6 | Pooling | [pooling.cu](src/backend/cuda/pooling.cu) | 257 | Unified 2D/3D, switch-based dispatch, 4 pool types |
| 7 | CumSum | [cumsum.cu](src/backend/cuda/cumsum.cu) | 161 | Sequential per-thread scan, upper/lower dimension decomposition |
| 8 | DepthwiseConv2D | [depthwise_conv2d.cu](src/backend/cuda/depthwise_conv2d.cu) | 225 | Per-channel NCHW, epilogue fusion (8 activation types), stride/dilation/padding |
| 9 | Eltwise | [eltwise.cu](src/backend/cuda/eltwise.cu) | 134 | Flat grid-stride, switch-based, 4 ops (Add/Sub/Mul/Div) |
| 10 | Unary | [unary.cu](src/backend/cuda/unary.cu) | 183 | Device fn pointer dispatch, flat/row kernels, 9 ops |

## Operators NOT Yet Implemented on CUDA

These operators return `nullptr` for `Backend::CUDA`:

| Operator | Status |
|---|---|
| Conv2D | `return nullptr; // backend::cuda::conv2d_cuda` — scaffolded but not implemented |
| Conv3D | `return nullptr; // backend::cuda::conv3d_cuda` — scaffolded but not implemented |
| MatMul | `return nullptr;` — no scaffolding |
| Linear | `return nullptr;` — no scaffolding |
| Attention | No `NNOPS_HAS_CUDA` block in dispatch file — not scaffolded |

## Build Integration

**Top-level CMakeLists.txt:**
- `option(NNOPS_BUILD_CUDA "Build CUDA backend" OFF)` — disabled by default
- AVX2 flags guarded to CXX only (`$<$<COMPILE_LANGUAGE:CXX>:...>`) to avoid nvcc errors

**src/CMakeLists.txt:**
- When enabled: `enable_language(CUDA)`, `CMAKE_CUDA_STANDARD 20`
- All 10 `.cu` files listed in `NNOPS_CUDA_SOURCES`
- Creates `nnops_cuda` STATIC library, links publicly to `nnops_cpu`
- `target_compile_definitions(nnops_cpu PUBLIC NNOPS_HAS_CUDA)` — makes the define available to all consumers
- `target_link_libraries(nnops_cpu PUBLIC nnops_cuda)` — CUDA kernels linked into CPU library

**tests/CMakeLists.txt:**
- `find_package(CUDAToolkit REQUIRED)` when CUDA enabled
- Each per-file test demo links `CUDA::cudart` (via `nnops_add_test_demo` in `tests/CMakeLists.txt`)

### Dispatch Wiring Pattern

Each operator's dispatch file (`src/ops/<name>.cpp`) uses:

```cpp
#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void <op>_cuda(
        const <Op>Attributes& attrs,
        TensorView& output,
        std::span<const TensorView> inputs,
        const ComputeContext& ctx,
        void* workspace);
}
#endif

// In resolve_<op>_kernel():
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::<op>_cuda;
#endif
```

Downstream consumers only link against `nnops_cpu` — the CUDA backend is transparent when `NNOPS_BUILD_CUDA=ON`.

## Key Design Decisions

1. **No external CUDA libraries** — no cuBLAS, no cuDNN. All kernels are hand-written to keep the library self-contained and avoid version lock-in.

2. **Float-only compute** — f16 inputs are promoted to float for all arithmetic, then truncated on store. This sacrifices some throughput for numerical consistency with the CPU path.

3. **C++20 for CUDA** (vs C++23 for CPU) — avoids nvcc compatibility issues with bleeding-edge C++ features.

4. **CUDA_CHECK is a no-op** — error checking is scaffolded but not implemented. Production use would need `cudaGetLastError()` checks after each kernel launch.

5. **No workspace required** — all 10 operators work with zero workspace bytes. Temporary allocations are handled internally with `cudaMalloc`/`cudaFree`.

6. **`nnops_cpu` owns the CUDA linkage** — the CUDA library is compiled separately but linked into `nnops_cpu`, so consumers see a single unified library.

7. **add_to support** — 8 of 10 operators support output accumulation (all except Softmax and CumSum, which had it removed 2026-07-25).

8. **C++20 `std::span`** used for input tensor lists — consistent with the CPU backend.

## Commit History

- `039384d` — `feat(cuda): add CUDA backend for 8 operators` (Softmax, LayerNorm, RMSNorm, Activation, BatchNorm, Pooling, CumSum, DepthwiseConv2D) — 17 files, 2209 insertions
- `fc49e15` — `feat(unary): add element-wise unary math operator with CPU and CUDA backends` — 5 files, 869 insertions (added Unary CUDA kernel)
- Eltwise CUDA kernel was added separately alongside the eltwise operator

## Related Memory

- [[operators]] — full operator catalog with shapes, attributes, and all backend implementations
- [[project-architecture]] — directory structure, backend dispatch model
- [[simd-infrastructure]] — CPU SIMD layer; CUDA `s_load`/`s_store` mirrors the CPU API
- [[pitch-design]] — pitch-aware row access shared between CPU and CUDA kernels
