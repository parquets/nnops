# CUDA Runtime API 接口参考

## 概述

CUDA Runtime API 是 NVIDIA CUDA 平台的核心宿主端（host-side）接口，提供设备管理、内存管理、流控制、图捕获、事件同步等功能。本文档整理所有常用接口及其在 nnops 项目中的实际使用方式。

nnops 的 CUDA 后端仅依赖两个头文件：
- `<cuda_runtime.h>` — 运行时 API（stream, malloc, free, memcpy, 等）
- `<cuda_fp16.h>` — 半精度浮点支持（`__half`, `__half2float`, `__float2half`）

**不依赖** cuBLAS、cuDNN 或任何其他 NVIDIA 库。

---

## 1. 错误处理

### API

```cpp
typedef enum {
    cudaSuccess                    =  0,
    cudaErrorInvalidValue          =  1,
    cudaErrorMemoryAllocation      =  2,
    cudaErrorInitializationError   =  3,
    cudaErrorLaunchFailure         =  4,
    cudaErrorLaunchTimeout         =  6,
    cudaErrorLaunchOutOfResources  =  7,
    cudaErrorInvalidDeviceFunction =  8,
    cudaErrorInvalidConfiguration  =  9,
    cudaErrorInvalidDevice         = 10,
    cudaErrorInvalidMemcpyDirection= 12,
    cudaErrorUnknown               = 30,
    cudaErrorNotReady              = 34,  // event query: not yet complete
    // ... 共 100+ 错误码
} cudaError_t;

const char* cudaGetErrorString(cudaError_t error);   // 错误码 → 可读字符串
const char* cudaGetErrorName(cudaError_t error);      // 错误码 → 枚举名
```

### nnops 实现

```cpp
// cuda_common.cuh:166
inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        // 生产环境应在此处 log + abort
        (void)file; (void)line;   // 当前为 no-op（占位符）
    }
}
#define CUDA_CHECK(err) cuda_check(err, __FILE__, __LINE__)
```

**设计决策：** 当前 `CUDA_CHECK` 为 no-op 占位符。生产环境应调用 `cudaGetErrorString(err)` 输出可读错误信息后 abort。

**典型用法：**
```cpp
CUDA_CHECK(cudaMalloc(&d_ptr, size));                     // 检查内存分配
CUDA_CHECK(cudaMemcpyAsync(dst, src, size, kind, stream));// 检查异步拷贝
CUDA_CHECK(cudaFree(d_ptr));                              // 检查内存释放
```

**注意：** 内核启动（`kernel<<<...>>>`）不返回 `cudaError_t`，需要用 `cudaGetLastError()` 检查：
```cpp
kernel<<<grid, block, shared, stream>>>(...);
CUDA_CHECK(cudaGetLastError());  // 捕获启动错误
// 内核执行错误必须用 cudaDeviceSynchronize() 或 cudaStreamSynchronize() 检查
```

---

## 2. 流管理 (Stream)

### API

```cpp
typedef struct CUstream_st* cudaStream_t;  // 不透明指针

// 创建/销毁
cudaError_t cudaStreamCreate(cudaStream_t* pStream);              // 默认流（同步）
cudaError_t cudaStreamCreateWithFlags(
    cudaStream_t* pStream, unsigned int flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);

// flags:
#define cudaStreamDefault    0x0   // 默认流
#define cudaStreamNonBlocking 0x1  // 非阻塞流

// 同步
cudaError_t cudaStreamSynchronize(cudaStream_t stream);  // 阻塞直到 stream 完成
cudaError_t cudaStreamQuery(cudaStream_t stream);        // 非阻塞查询（cudaSuccess/cudaErrorNotReady）

// 等待
cudaError_t cudaStreamWaitEvent(
    cudaStream_t stream, cudaEvent_t event, unsigned int flags);

// 优先级
cudaError_t cudaStreamCreateWithPriority(
    cudaStream_t* pStream, unsigned int flags, int priority);
```

### nnops 使用方式

**外部传入，不自行创建：** nnops 的所有 CUDA 算子从 `ComputeContext::cuda_stream` 获取 stream，由调用者管理生命周期：

```cpp
struct ComputeContext {
    void* cuda_stream = nullptr;  // cudaStream_t, 外部传入，不透明指针
};
```

**所有算子的统一模式：**
```cpp
cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);
kernel<<<grid, block, shared, stream>>>(...);
// 如果需要临时设备内存拷贝，也用同一条 stream
cudaMemcpyAsync(d_buf, h_buf.data(), size, cudaMemcpyHostToDevice, stream);
```

**设计理由：** nnops 不管理 stream 生命周期。调用者负责创建/销毁 stream，保证所有算子在同一 stream 上按序执行。这是典型的推理引擎模式（如 onnxruntime、TensorRT）。

**非阻塞流场景：** nnops 不做任何关于 `cudaStreamNonBlocking` 的假设。只要调用者传入正确的 stream，算子本身无状态、无隐式同步。

---

## 3. 内存管理 (Memory)

### API

```cpp
// 设备内存分配/释放
cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaMemset(void* devPtr, int value, size_t count);              // 同步
cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count,
                             cudaStream_t stream);                           // 异步

// 页锁定主机内存（DMA 加速）
cudaError_t cudaMallocHost(void** ptr, size_t size);            // 页锁定
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags);
cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags);

// 统一内存（CUDA 6.0+）
cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags);

// 3D 分配（纹理/3D 数组）
cudaError_t cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent);
cudaError_t cudaMallocArray(cudaArray_t* array,
                             const cudaChannelFormatDesc* desc,
                             size_t width, size_t height, unsigned int flags);

// 内存拷贝
cudaError_t cudaMemcpy(void* dst, const void* src,
                        size_t count, cudaMemcpyKind kind);      // 同步
cudaError_t cudaMemcpyAsync(void* dst, const void* src,
                             size_t count, cudaMemcpyKind kind,
                             cudaStream_t stream);               // 异步
cudaError_t cudaMemcpy2D(void* dst, size_t dpitch,
                          const void* src, size_t spitch,
                          size_t width, size_t height,
                          cudaMemcpyKind kind);                  // 同步 2D
cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch,
                                const void* src, size_t spitch,
                                size_t width, size_t height,
                                cudaMemcpyKind kind,
                                cudaStream_t stream);            // 异步 2D

// cudaMemcpyKind:
enum cudaMemcpyKind {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault         = 4   // CUDA 6.0+: 从指针自动推断方向
};

// Pinned memory 的异步拷贝可以不指定 stream
```

### nnops 使用方式

**原则：算子本身不申请持久设备内存。** 只有少数算子需要**临时**设备缓冲区，在 `compute()` 调用期间申请、使用后立即释放：

| 算子 | 临时分配 | 用途 |
|---|---|---|
| Softmax | `cudaMalloc(d_offsets)` | general path 的 inner offsets（`int64_t[*]`） |
| LayerNorm | 同上 | 同上 |
| RMSNorm | 同上 | 同上 |
| Reduce | `cudaMalloc(d_offsets)` | general path 的 inner offsets（`int64_t[*]`） |
| BatchNorm | `cudaMalloc(d_new_scale)` + `cudaMalloc(d_new_bias)` | spatial mode 的预计算融合参数 |

**生命周期模式：**
```cpp
int64_t* d_offsets = nullptr;
cudaMalloc(&d_offsets, norm_size * sizeof(int64_t));
cudaMemcpyAsync(d_offsets, h_offsets.data(),
                norm_size * sizeof(int64_t),
                cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, shared, stream>>>(..., d_offsets, ...);
cudaFree(d_offsets);   // 注意：这是同步的！在非空 stream 上可能先于 kernel 执行
```

**关键隐患：** `cudaFree` 和 `cudaMalloc` 在某些早期 CUDA 版本中会做隐式全局同步。如果需要严格异步，应使用 memory pool（`cudaMemPool`）或 RingBuffer 复用策略。nnops 当前接受这一限制。

**`getWorkspaceSize()` 设计：** 所有算子返回 0。临时内存由算子内部管理，不走 workspace 通道。

---

## 4. 内核启动 (Kernel Launch)

### API

```cpp
// 三重尖括号语法 (host-side)
kernel_name<<<gridDim, blockDim, sharedMemBytes, stream>>>(args...);

// gridDim:  dim3,  {x, y=1, z=1} — 1D/2D/3D 网格
// blockDim: dim3,  {x, y=1, z=1} — 1D/2D/3D 线程块
// sharedMemBytes: size_t, 动态共享内存字节数
// stream: cudaStream_t, 关联的流（0 = 默认流/legacy）

// 配合使用的 API
cudaError_t cudaFuncSetAttribute(const void* func,
                                  cudaFuncAttribute attr, int value);
cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr,
                                   const void* func);

// cudaFuncAttributes:
struct cudaFuncAttributes {
    size_t sharedSizeBytes;    // 静态共享内存
    size_t constSizeBytes;
    size_t localSizeBytes;     // 每线程局部内存
    int maxThreadsPerBlock;
    int numRegs;               // 每线程寄存器数
    int ptxVersion;
    int binaryVersion;
};

// For kernel attributes:
enum cudaFuncAttribute {
    cudaFuncAttributeMaxDynamicSharedMemorySize,  // 最大动态共享内存
    cudaFuncAttributePreferredSharedMemoryCarveout, // 首选共享/L1 比例
};
```

### nnops 使用方式

```cpp
// 标准启动配置
const int block_size = std::min(static_cast<int>(dim_size), kCudaBlockSize);  // ≤ 256
const int shared_bytes = block_size * sizeof(float);   // 仅 reduction 算子需要
const int grid = std::min(static_cast<int>(outer_dim_count), 65535);  // pre-Volta 限制

// 所有算子使用相同的 stream 获取方式
cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

// 启动
reduce_fast_kernel<T><<<grid, block_size, shared_bytes, stream>>>(...);
```

**常量定义：**
| 符号 | 值 | 出处 |
|---|---|---|
| `kCudaBlockSize` | 256 | `cuda_common.cuh` |
| `kCudaWarpSize` | 32 | `cuda_common.cuh` |
| 最大 grid (x) | 65535 | pre-Volta (sm_30~sm_60) 限制 |

**Grid/Block 选择策略：**
- **Flat kernel**（Activation, Eltwise, Unary）：`grid = ceil_div(N, 256)`，线程跨步处理
- **Row kernel**（Softmax, LayerNorm, RMSNorm, BatchNorm, Pooling, CumSum, Reduce）：`grid = num_rows`（每个 block 处理一行），上限 65535
- **DWConv**：`grid = {out_w, N * C}`，`block = {min(out_h, 16), min(kH * kW, 16)}`（2D thread block）

> ⚠️ **注意：** `reduce.cu` 是唯一一个 `cudaMalloc`/`cudaMemcpyAsync`/`cudaFree` **未**使用 `CUDA_CHECK` 包裹的 `.cu` 文件（行 218-227）。这是已知的不一致，需要在生产环境中修复。

---

## 5. 设备管理 (Device)

### API

```cpp
// 设备计数与选择
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDevice(int* device);                        // 当前设备
cudaError_t cudaSetDevice(int device);                         // 切换设备
cudaError_t cudaDeviceGetAttribute(int* value,
                                    cudaDeviceAttr attr,
                                    int device);               // 查询单个属性

// 设备属性
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);

struct cudaDeviceProp {
    char   name[256];               // GPU 名称（如 "NVIDIA GeForce RTX 4080"）
    size_t totalGlobalMem;          // 总全局内存（bytes）
    size_t sharedMemPerBlock;       // 每 block 共享内存（bytes）
    size_t sharedMemPerBlockOptin;  // opt-in 最大共享内存
    int    regsPerBlock;            // 每 block 32-bit 寄存器数
    int    warpSize;                // 线程束大小（当前全部 = 32）
    int    maxThreadsPerBlock;      // 每 block 最大线程数（当前全部 ≥ 1024）
    int    maxThreadsDim[3];        // 每维最大线程数
    int    maxGridSize[3];          // 每维最大网格数
    int    maxThreadsPerMultiProcessor;
    int    multiProcessorCount;     // SM 数量
    int    clockRate;               // KHz
    int    memoryClockRate;         // KHz
    int    memoryBusWidth;          // bits
    int    totalConstMem;           // 常量内存
    int    major, minor;            // 计算能力 (major.minor)
    int    pciBusID, pciDeviceID;
    int    maxTexture1D;
    int    unifiedAddressing;       // 是否支持统一寻址
    int    computeMode;             // cudaComputeModeDefault/Exclusive/Prohibited/etc
    int    concurrentKernels;       // 是否支持多 kernel 并发
    int    canMapHostMemory;        // 是否支持 host memory mapping
    int    asyncEngineCount;        // DMA engine 数量
    int    deviceOverlap;           // 是否支持 device 和 memory 拷贝重叠
    // ... 更多字段
};

// cudaDeviceAttr 枚举（部分常用）：
cudaDevAttrMaxThreadsPerBlock      // = 1
cudaDevAttrMaxBlockDimX            // = 2
cudaDevAttrMaxGridDimX             // = 5
cudaDevAttrMaxSharedMemoryPerBlock // = 8
cudaDevAttrWarpSize                // = 10
cudaDevAttrMaxRegistersPerBlock    // = 12
cudaDevAttrMultiProcessorCount     // = 16
cudaDevAttrComputeCapabilityMajor  // = 75
cudaDevAttrMaxSharedMemoryPerBlockOptin  // = 97

// 设备间通信（multi-GPU）
cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer,
                                     int device, int peerDevice);
```

### nnops 使用方式

**当前 nnops 不调用任何设备管理 API。** 设备选择和属性查询由调用者（如模型推理框架）在外部完成。这意味着：
- 没有 `cudaSetDevice` 调用（假设当前设备已正确设置）
- 没有运行时 SM/共享内存大小查询（使用硬编码的 `kCudaBlockSize = 256` 和 pre-Volta grid 限制）

**隐含约束：**
- Grid size ≤ 65535 — 假设最低支持 sm_30（Kepler），这排除了 Tesla K40 以后的 GPU
- Block size ≤ 256 — 与设备无关的安全选择，最大 occupancy 兼顾
- 不查询 `sharedMemPerBlock` — 动态共享内存使用 `block_size * sizeof(float)` ≤ 1024 bytes，远低于所有 GPU 的限制（≥ 48KB）
- CMake 编译目标：`compute_86,sm_86` — 当前硬编码为 Ampere（RTX 30 系列）

**需要运行时查询的场景（未来考虑）：**
- 按 SM 数自适应 grid 大小（`multiProcessorCount * maxBlocksPerSM`）
- 大 shared memory buffer（如 GEMM tile）需确认设备上限
- 异构 GPU 集群需要知道 `totalGlobalMem` 以决定张量分割

---

## 6. CUDA Graph（图捕获与重放）

### API

```cpp
// Graph 对象
cudaError_t cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags);
cudaError_t cudaGraphDestroy(cudaGraph_t graph);

// 实例化
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec,
                                  cudaGraph_t graph,
                                  unsigned int flags);         // launch 后自动释放
cudaError_t cudaGraphInstantiateWithFlags(
    cudaGraphExec_t* pGraphExec, cudaGraph_t graph,
    unsigned long long flags);                                 // 更多控制

// 图执行
cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec,
                             cudaStream_t stream);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec);

// 更新参数（无需重新实例化）
cudaError_t cudaGraphExecUpdate(cudaGraphExec_t graphExec,
                                 cudaGraph_t graph,
                                 cudaGraphNode_t* errorNode,
                                 cudaGraphExecUpdateResult* updateResult_out);

// 捕获
cudaError_t cudaStreamBeginCapture(cudaStream_t stream,
                                    cudaStreamCaptureMode mode);
cudaError_t cudaStreamEndCapture(cudaStream_t stream,
                                  cudaGraph_t* pGraph);
cudaError_t cudaStreamIsCapturing(cudaStream_t stream,
                                   cudaStreamCaptureStatus* pCaptureStatus);

// 调试/导出
cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph,
                                    const char* path, unsigned int flags);

// 图节点（手动构建，不用 capture）
cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t* pGraphNode,
                                    cudaGraph_t graph,
                                    const cudaGraphNode_t* pDependencies,
                                    size_t numDependencies,
                                    const cudaKernelNodeParams* pNodeParams);
cudaError_t cudaGraphAddMemcpyNode(...);
cudaError_t cudaGraphAddMemsetNode(...);
cudaError_t cudaGraphAddHostNode(...);
cudaError_t cudaGraphAddEmptyNode(...);
cudaError_t cudaGraphAddEventRecordNode(...);
cudaError_t cudaGraphAddEventWaitNode(...);

// 上传/下载（跨进程传输）
cudaError_t cudaGraphUpload(cudaGraph_t graph, cudaStream_t stream);
```

### nnops 图捕获兼容性

nnops 的 CUDA 算子**可以**被捕获到 CUDA Graph 中，但有以下约束：

1. **涉及 `cudaMalloc`/`cudaFree` 的算子不能放入 static graph**（即 general path 的 Softmax、LayerNorm、RMSNorm、Reduce 和 spatial BatchNorm）。只有 fast path（axis == rank-1）是 graph-safe 的。
2. **空 workspace**：所有算子 `getWorkspaceSize() == 0`，无需在 graph 外部预分配 workspace。
3. **无 host 同步**：内核不调用 `cudaDeviceSynchronize` 或 `cudaStreamSynchronize`，对 graph 友好。
4. **捕获建议**：在 `cudaStreamBeginCapture` 前先 warmup 一次 fast path（触发 lazy 分支选择），避免 capture 期间触发 general path。

**Graph-safe 条件：**
| 算子 | Graph-safe | 条件 |
|---|---|---|
| Softmax | 仅 fast path | axis == rank-1 |
| LayerNorm | 仅 fast path | axis == rank-1 |
| RMSNorm | 仅 fast path | axis == rank-1 |
| Reduce | 仅 fast path | axis == rank-1 |
| BatchNorm | 仅 non-spatial | spatial mode 有 `cudaMalloc` |
| Activation | ✓ | 全部 graph-safe |
| Eltwise | ✓ | 全部 graph-safe |
| Unary | ✓ | 全部 graph-safe |
| Pooling | ✓ | 全部 graph-safe |
| CumSum | ✓ | 全部 graph-safe |
| DepthwiseConv2D | ✓ | 全部 graph-safe |

---

## 7. 事件 (Event)

### API

```cpp
typedef struct CUevent_st* cudaEvent_t;  // 不透明指针

// 创建/销毁
cudaError_t cudaEventCreate(cudaEvent_t* event);                // 默认
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);

// flags:
#define cudaEventDefault      0x0
#define cudaEventBlockingSync 0x1  // event 同步使用阻塞而非轮询（减少 CPU 占用）
#define cudaEventDisableTiming 0x2 // 禁止计时（减少开销）
#define cudaEventInterprocess  0x4 // 跨进程共享 event

// 操作
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);        // 阻塞等待
cudaError_t cudaEventQuery(cudaEvent_t event);              // cudaSuccess/cudaErrorNotReady
cudaError_t cudaEventElapsedTime(float* ms,
                                  cudaEvent_t start, cudaEvent_t stop);

// 流间依赖
cudaError_t cudaStreamWaitEvent(cudaStream_t stream,
                                 cudaEvent_t event, unsigned int flags);
```

### nnops 使用方式

**当前 nnops 不创建或不使用 Event。** 所有算子间依赖通过共享同一 `cudaStream_t` 实现（FIFO 语义）。Event 是调用者的工具。

**典型调用者场景（不在 nnops 内部）：**
```cpp
// 推理引擎中：测量延迟
cudaEventRecord(start_event, stream);
op->compute(outputs, inputs, ctx);
cudaEventRecord(stop_event, stream);
cudaEventSynchronize(stop_event);
cudaEventElapsedTime(&ms, start_event, stop_event);

// 流间同步：stream2 等待 stream1 完成
cudaEventRecord(cross_event, stream1);
cudaStreamWaitEvent(stream2, cross_event, 0);
```

---

## 8. 同步 (Synchronization)

### API

```cpp
// 设备级同步（全局屏障）
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaDeviceReset(void);                            // 销毁所有分配 + 重置设备

// 流级同步
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);             // 非阻塞

// Event 级同步
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventQuery(cudaEvent_t event);
```

### nnops 使用方式

**nnops 的 `compute()` 是异步的。** 不调用任何同步 API。调用者负责在需要结果时同步：

```cpp
// 典型推理流程
op1->compute(o1, i1, ctx);  // 异步
op2->compute(o2, o1, ctx);  // 同一 stream，自动在 op1 之后执行
op3->compute(o3, o2, ctx);  // 同上

// 需要结果时
cudaStreamSynchronize(static_cast<cudaStream_t>(ctx.cuda_stream));
// 或: cudaDeviceSynchronize();
```

---

## 9. 设备属性查询 (cudaDeviceAttr)

### 常用属性清单

调用模式：
```cpp
int value;
cudaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerBlock, device);
```

| 属性 | 类型 | 说明 | 典型值 (A100) |
|---|---|---|---|
| `cudaDevAttrMaxThreadsPerBlock` | int | 每 block 最大线程数 | 1024 |
| `cudaDevAttrMaxBlockDimX/Y/Z` | int | 各维最大 block 大小 | 1024 |
| `cudaDevAttrMaxGridDimX/Y/Z` | int | 各维最大 grid 大小 | 2147483647 (2^31-1) |
| `cudaDevAttrMaxSharedMemoryPerBlock` | int | 每 block 最大共享内存 (bytes) | 49152 (48KB) |
| `cudaDevAttrMaxSharedMemoryPerBlockOptin` | int | opt-in 最大共享内存 | 163840 (160KB) |
| `cudaDevAttrWarpSize` | int | 线程束大小 | 32 |
| `cudaDevAttrMaxRegistersPerBlock` | int | 每 block 最大 32-bit 寄存器数 | 65536 |
| `cudaDevAttrMultiProcessorCount` | int | SM 数量 | 108 |
| `cudaDevAttrMaxRegistersPerMultiprocessor` | int | 每 SM 寄存器数 | 65536 |
| `cudaDevAttrComputeCapabilityMajor/Minor` | int | 计算能力 | 8.0 |
| `cudaDevAttrUnifiedAddressing` | int | 是否统一寻址 | 1 |
| `cudaDevAttrConcurrentKernels` | int | 是否支持 kernel 并发 | 1 |
| `cudaDevAttrMaxThreadsPerMultiProcessor` | int | 每 SM 最大线程数 | 2048 |
| `cudaDevAttrMemoryClockRate` | int | 显存频率 (KHz) | 1215000 |
| `cudaDevAttrGlobalMemoryBusWidth` | int | 显存总线位宽 (bits) | 5120 (HBM2e) |
| `cudaDevAttrAsyncEngineCount` | int | 异步引擎数 | 2 |
| `cudaDevAttrDeviceOverlap` | int | 是否支持计算与拷贝重叠 | 1 |
| `cudaDevAttrCooperativeLaunch` | int | 是否支持协作式启动 | 1 |

---

## 10. nnops 中的 CUDA API 使用矩阵

| API | 使用位置 | 频率 |
|---|---|---|
| `cudaStream_t` (来自 ctx) | 所有 11 个 `.cu` 文件 | 每算子 1 次 |
| `cudaMalloc` | batchnorm, softmax, layer_norm, rms_norm, reduce | general path 中 |
| `cudaFree` | 同上 | general path 中 |
| `cudaMemcpyAsync` (`cudaMemcpyHostToDevice`) | 同上 | general path 中 |
| `cudaGetLastError` | 当前未使用 | — |
| `cudaDeviceSynchronize` | 当前未使用 | — |
| `cudaStreamCreate/Destroy` | 当前未使用（外部管理） | — |
| `cudaEventCreate/Destroy/Record` | 当前未使用 | — |
| `cudaSetDevice/GetDeviceProperties` | 当前未使用 | — |
| `cudaFuncSetAttribute` | 当前未使用 | — |
| CUDA Graphs | 当前未使用（但大部分算子 graph-compatible） | — |

---

## 11. 关键设计决策

1. **Stream 外部管理** — nnops 不创建/销毁 stream，不假设 stream 类型（default/non-blocking）。由推理引擎统一调度。
2. **临时内存内部分配** — 不使用 workspace 通道。临时 buffer 在 compute() 内 `cudaMalloc`/`cudaFree`，代价是可能的隐式同步。
3. **统一内核入口模式** — 所有算子遵循 `void <op>_cuda()` → `void <op>_cuda_impl<T>()` 两层结构，stream 从 `ctx.cuda_stream` 提取。
4. **CUDA_CHECK no-op** — 错误检查框架已搭建但未实现，生产环境需补充 `cudaGetErrorString` + abort。
5. **无运行时设备查询** — 使用安全的硬编码常量（block=256, max grid=65535），不依赖设备属性运行时查询。
6. **Graph 兼容性分层** — 7 个算子 fully graph-safe，4 个算子仅 fast path 可放入 static graph。
7. **Float-only compute** — f16 通过 `s_load`/`s_store` 隐式转换为 float 运算，不依赖原生 half 指令（除了 `__half2float`/`__float2half`）。

---

## 12. 参考资源

- [CUDA Runtime API 官方文档](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- nnops CUDA 基础设施文档：`memory/cuda-infrastructure.md`
- nnops 公共头文件：`src/backend/cuda/cuda_common.cuh`
