# Vulkan 线程模型与内存模型

> 基于 Vulkan 1.4 + Roadmap 2026 标准，全面覆盖 Host 端多线程、GPU 端执行层次、SPIR-V 内存模型、Synchronization2 同步原语及实用模式。

---

## 目录

1. [Host 端线程模型](#1-host-端线程模型)
2. [GPU 端执行模型](#2-gpu-端执行模型)
3. [队列模型与异步计算](#3-队列模型与异步计算)
4. [同步原语全览](#4-同步原语全览)
5. [SPIR-V 内存模型](#5-spir-v-内存模型)
6. [Pipeline Barrier 与 Execution/Memory Dependency](#6-pipeline-barrier-与-executionmemory-dependency)
7. [Timeline Semaphore 深度解析](#7-timeline-semaphore-深度解析)
8. [Host-Device 同步模式](#8-host-device-同步模式)
9. [多线程实用模式](#9-多线程实用模式)
10. [nnops 相关考量](#10-nnops-相关考量)
11. [快速参考卡片](#11-快速参考卡片)

---

## 1. Host 端线程模型

### 1.1 核心理念

Vulkan 的 Host 端线程模型与 OpenGL 截然不同：

```
OpenGL:  单线程状态机，驱动内部隐式同步
Vulkan:  应用完全掌控多线程，驱动不做任何内部同步
```

Vulkan 规范明确：**驱动不引入任何内部多线程**。如果应用想要多核 CPU 性能，必须自己管理同步。多线程在 Vulkan 中只提供 **Host 端扩展**——任何与 Device 交互的操作仍然需要正确的同步。

### 1.2 External Synchronization（外部同步规则）

Vulkan 规范中标注为 **"externally synchronized"** 的参数表示：

> 如果两条命令访问同一个 externally-synchronized 对象，且至少一条声明它是 externally-synchronized，则调用者必须确保它们不会同时执行。

**关键理解**：

- "外部"意味着同步责任在 API 之外（即你的代码）
- 你可以用任何方式实现：mutex、限制到单线程、每线程独立对象
- Vulkan 不关心**如何**同步，只要求**必须**同步

#### 对象的外部同步分类

```
┌─────────────────────────────────────────────────────────┐
│               Externally Synchronized                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ CommandPool  │  │ DescriptorPool│  │ Queue        │  │
│  │ (alloc/free/ │  │ (alloc/free/ │  │ (submit/     │  │
│  │  reset/      │  │  update/     │  │  present)    │  │
│  │  recording)  │  │  reset)      │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ CmdBuffer    │  │ Device       │  │ PipelineCache│  │
│  │ (recording)  │  │ (most calls) │  │ (merge)      │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
├─────────────────────────────────────────────────────────┤
│            NOT Externally Synchronized                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ VkBuffer     │  │ VkImage      │  │ VkPipeline   │  │
│  │ VkDeviceMem  │  │ VkSampler    │  │ VkRenderPass │  │
│  │ (immutable)  │  │ (immutable)  │  │ (immutable)  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 1.3 Command Pool 线程规则

**Command Pool 是 Host 端线程模型的核心约束对象**：

```
一个 Command Pool 必须被外部同步
├─ 不能从多线程同时分配 command buffer
├─ 不能从多线程同时释放 command buffer
├─ 不能从多线程同时 reset pool 或 command buffer
└─ ── 最关键 ──
    不能从多线程同时录制从同一 Pool 分配的不同 Command Buffer！
    vkCmd* 函数隐式使用其父 Pool 作为 externally-synchronized 参数
```

**推荐模式**：

```cpp
// ✅ 每线程一个独立的 Command Pool
struct ThreadContext {
    VkCommandPool cmd_pool;     // 线程独占
    VkDescriptorPool desc_pool;  // 线程独占（如果需要）
};

// Thread 1: 录制 compute dispatch
VkCommandBuffer cmd1 = allocate_from(thread1.cmd_pool);
vkCmdDispatch(cmd1, ...);  // 安全 — 独占 pool

// Thread 2: 同时录制不同的 compute dispatch
VkCommandBuffer cmd2 = allocate_from(thread2.cmd_pool);
vkCmdDispatch(cmd2, ...);  // 安全 — 不同的 pool

// 主线程：提交
vkQueueSubmit(queue, {cmd1, cmd2});  // 需要外部同步 Queue
```

### 1.4 Descriptor Pool 线程规则

Descriptor Pool 同样是 externally-synchronized。两种策略：

```cpp
// 策略 A：每线程独立 Pool（推荐 — 零锁开销）
VkDescriptorPool thread_pools[NUM_THREADS];
// 每个线程管理自己的 descriptor set 分配/更新

// 策略 B：共享 Pool + Mutex（适合低频分配场景）
std::mutex desc_pool_mutex;
{
    std::lock_guard lock(desc_pool_mutex);
    vkAllocateDescriptorSets(device, &alloc_info, &set);
}
```

### 1.5 Command Buffer 生命周期与线程

```cpp
// 状态转换与线程所有权

// Initial → Recording：在分配线程中
vkBeginCommandBuffer(cmd, ...);  // 必须在分配线程

// Recording 状态：
//   - cmd 不能离开录制线程
//   - 父 Pool 被隐式锁定
vkCmdDispatch(cmd, ...);         // 线程安全（Pool 独占）
vkCmdPipelineBarrier2(cmd, ...); // 同上

// Recording → Executable：
vkEndCommandBuffer(cmd);         // 必须在录制线程

// Executable 状态：
//   - cmd 可以自由传递到任意线程
//   - 可以被提交到 Queue
std::thread([cmd] {
    vkQueueSubmit2(queue, &submit_info, VK_NULL_HANDLE);
}).detach();

// Executable → Invalid（ONE_TIME_SUBMIT）：
//   自动转换，cmd 此后不可重用

// Executable → Invalid（reset）：
vkResetCommandBuffer(cmd, ...);  // 必须独占 Pool
// 或整个 Pool reset：
vkResetCommandPool(device, pool, ...);  // 必须独占 Pool
```

### 1.6 Device 级别的线程安全

`VkDevice` 本身对大多数操作是 externally-synchronized。但有两个重要例外：

| 操作 | 线程安全性 |
|------|-----------|
| `vkQueueSubmit` / `vkQueueSubmit2` | **可以在不同 Queue 上并发调用**，无需外部同步 |
| `vkAllocateMemory` / `vkFreeMemory` | **可以并发调用**（Vulkan 1.1+） |
| `vkCreate*` / `vkDestroy*` | 需要外部同步 |
| `vkWaitForFences` / `vkWaitSemaphores` | 可以并发调用 |

### 1.7 Vulkan 1.4 中支持 Free-Threaded 的扩展

| 扩展 | 对多线程的影响 |
|------|---------------|
| `VK_EXT_descriptor_heap` (2026) | 消除 descriptor pool 碎片问题，简化多线程 descriptor 管理 |
| `VK_KHR_push_descriptor` (1.4 core) | 直接在 command buffer 中写 descriptor，无需 shared pool |
| Synchronization2 (1.3 core) | 64-bit stage/access flags，消除 pNext 链竞争 |
| `VK_EXT_shader_object` | 消除 pipeline 创建时的内部同步（compute 受益有限） |

---

## 2. GPU 端执行模型

### 2.1 执行层次结构

```
vkCmdDispatch(cmd, Gx, Gy, Gz)
│
├─ Global Grid (NDRange) ───────────────── 总问题空间
│  └─ Workgroup (0,0,0)  ───────────────── 调度基本单位
│     ├─ Subgroup 0 (Warp/Wavefront) ───── SIMD 执行单位
│     │  ├─ Invocation 0  ──────────────── 单线程
│     │  ├─ Invocation 1
│     │  ├─ ...
│     │  └─ Invocation 31 (subgroupSize)
│     ├─ Subgroup 1
│     └─ ...
│  ├─ Workgroup (1,0,0)
│  └─ ...
└─
```

### 2.2 各级别详细说明

#### Workgroup（调度基本单位）

```
Workgroup 是 GPU 调度器分配的最小单位。

关键属性：
├─ 原子性：一旦被调度到某个 Compute Unit (CU/SM)，整个 workgroup
│          的所有 invocation 始终驻留在该 CU 上直到完成
├─ 共享内存：同一个 workgroup 内的所有 invocation 共享 LDS (Local
│           Data Store)，这是 shared memory 的硬件基础
└─ 上限：Vulkan 保证值 ≥ 128 invocations/workgroup（1.3）
         Roadmap 2026 提升至 ≥ 1024
```

#### Subgroup（SIMD 执行单位）

```
Subgroup = Warp (NVIDIA) = Wavefront (AMD) = SIMD Group (Intel/Metal)

关键属性：
├─ 固定大小：subgroupSize = 32 (NVIDIA) / 64 (AMD) / 4-128 (Mobile)
│            必须通过 VkPhysicalDeviceSubgroupProperties 查询
├─ 锁步执行：同一 subgroup 内的所有 invocation 同时执行同一指令
├─ 隐式同步：subgroup 内不需要显式 barrier
├─ Shuffle：通过 subgroupShuffle 直接交换寄存器（最快通信路径）
└─ Vote/Ballot：硬件级条件判断和位掩码操作
```

#### Invocation（单线程）

```
Invocation = 最小执行单位

每个 invocation 拥有：
├─ 独立的程序计数器（概念上）
├─ 独立的寄存器文件
└─ 独立的 gl_GlobalInvocationID / gl_LocalInvocationID
```

### 2.3 SIMT 执行与 Warp Divergence

```
核心概念：GPU 是 SIMD 机器，不独立执行每个 invocation

正常执行（无 divergence）：
指令: result = a[gid] + b[gid]
  线程 0-31 同时执行，100% 利用率

分支 divergence：
if (gid % 2 == 0) {
    result = a[gid] * 2.0;    // 偶数线程执行
} else {
    result = a[gid] * 3.0;    // 奇数线程执行
}
  执行过程：
  ① 所有 32 线程执行 *2.0，奇数被 mask  → 50% 利用率
  ② 所有 32 线程执行 *3.0，偶数被 mask  → 50% 利用率
  总体：2× 指令数，50% 利用率
```

### 2.4 延迟隐藏与 Occupancy

```
GPU 通过 时间复用 来隐藏内存延迟：

时间线：
  Warp 0: [计算] ────→ [等待内存 400 cycles] ────→ [计算]
  Warp 1:           [计算] ────→ [等待内存] ────→ [计算]
  Warp 2:                     [计算] ────→ [等待]
  Warp 3:                               [计算] ────→
  ...

Occupancy = 同时驻留在 CU 上的 warp 数量
高 Occupancy → 更多 warp 可供切换 → 更好的延迟隐藏

Occupancy 受限于：
├─ VGPR (向量寄存器) 使用量 / workgroup
├─ LDS (Shared Memory) 分配量 / workgroup
└─ 硬件 warp 槽位总数
```

### 2.5 跨厂商术语映射

| Vulkan/OpenCL | CUDA | NVIDIA 硬件 | AMD 硬件 | Metal/D3D12 | Mobile (ARM/QC) |
|:---|:---|:---|:---|:---|:---|
| Workgroup | Thread Block | Thread Block | Workgroup | Thread Group | Workgroup |
| Invocation | Thread | Thread | Work-item | Thread | Thread |
| Subgroup | Warp (32) | Warp (32) | Wavefront (32/64) | SIMD Group | Warp (4–128) |
| Compute Unit | SM | SM | CU | — | Shader Core |
| Shared Memory | Shared Memory | L1/SMEM (48KB) | LDS (64KB) | Tile Memory | Tile Memory |

---

## 3. 队列模型与异步计算

### 3.1 Queue Family 概念

```
VkPhysicalDevice
├─ Queue Family 0: Graphics | Compute | Transfer  (通用队列)
├─ Queue Family 1: Compute | Transfer            (纯计算队列 — 异步计算)
└─ Queue Family 2: Transfer                      (DMA 传输队列)

从 Queue Family 创建多个 VkQueue：
  Queue Family 0 → VkQueue[0], VkQueue[1]  (两个图形队列)
  Queue Family 1 → VkQueue[2], VkQueue[3]  (两个异步计算队列)
```

### 3.2 异构队列调度

```
时间线：Graphics Queue 和 Compute Queue 并行执行

Graphics Queue:  [Frame N Render]         [Frame N+1 Render]
Compute Queue:     [Post-process N-1] [Physics N]  [AI N]

关键：
├─ 不同 Queue 之间没有隐式同步
├─ 必须显式使用 Semaphore 来同步跨队列依赖
└─ Timeline Semaphore (Vulkan 1.2+) 是跨队列同步的首选
```

### 3.3 异步 Compute 编排

```cpp
// 典型模式：Graphics 渲染 + 异步 Compute 后处理

// Queue 0 (Graphics): Render frame N
// Queue 1 (Async Compute): Post-process frame N-1

VkSemaphoreSubmitInfo wait_infos[] = {
    // Compute queue 等待上一帧的渲染完成
    { .semaphore = render_done_sem, .value = frame_n,
      .stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT },
};

VkSemaphoreSubmitInfo signal_infos[] = {
    // Compute 完成后发信号
    { .semaphore = compute_done_sem, .value = frame_n,
      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT },
};

VkCommandBufferSubmitInfo cmd_infos[] = {
    { .commandBuffer = post_process_cmd },
};

VkSubmitInfo2 submit = {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
    .waitSemaphoreInfoCount = 1,
    .pWaitSemaphoreInfos = wait_infos,
    .commandBufferInfoCount = 1,
    .pCommandBufferInfos = cmd_infos,
    .signalSemaphoreInfoCount = 1,
    .pSignalSemaphoreInfos = signal_infos,
};

vkQueueSubmit2(compute_queue, 1, &submit, VK_NULL_HANDLE);
```

### 3.4 Queue Ownership Transfer

当 Buffer/Image 需要在不同 Queue Family 之间共享时：

```cpp
// Queue Family 0 → Queue Family 1

// ① Release from Queue Family 0
VkBufferMemoryBarrier2 release = {
    .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
    .dstStageMask = VK_PIPELINE_STAGE_2_NONE,
    .dstAccessMask = VK_ACCESS_2_NONE,
    .srcQueueFamilyIndex = QUEUE_FAMILY_0,
    .dstQueueFamilyIndex = QUEUE_FAMILY_1,
    .buffer = shared_buffer,
};
vkCmdPipelineBarrier2(cmd_q0, &dep_info);

// ② Acquire on Queue Family 1
VkBufferMemoryBarrier2 acquire = {
    .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
    .srcAccessMask = VK_ACCESS_2_NONE,
    .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
    .srcQueueFamilyIndex = QUEUE_FAMILY_0,
    .dstQueueFamilyIndex = QUEUE_FAMILY_1,
    .buffer = shared_buffer,
};
vkCmdPipelineBarrier2(cmd_q1, &dep_info);
```

### 3.5 Vulkan 1.4 中 Queue 相关增强

| 特性 | 说明 |
|------|------|
| Synchronization2 (1.3 core, 1.4 mandatory) | `vkQueueSubmit2` 统一 Queue 提交接口 |
| Timeline Semaphore (1.2 core) | 64-bit 单调计数器替代 Binary Semaphore/Fence |
| `VK_EXT_descriptor_heap` (2026) | 跨 Queue 共享 descriptor heap |
| `maxBoundDescriptorSets` 提升 | 1.3: ≥4, Roadmap 2026: ≥8 |

---

## 4. 同步原语全览

### 4.1 四种同步原语对比

```
                    Host 可见     Device 可见     状态模型       跨 Queue
Fence               ✅ (wait)    ❌ (仅 signal)   Binary          ❌
Binary Semaphore    ❌           ✅ (signal/wait)  Binary          ✅
Timeline Semaphore  ✅ (wait/    ✅ (signal/wait)  Monotonic       ✅
                       query/                      uint64_t
                       signal)
Event               ✅ (get)     ✅ (set/wait)     Binary          ❌ (单 Queue)
```

### 4.2 详细能力矩阵

| 操作 | Fence | Binary Sem | Timeline Sem | Event |
|------|:-----:|:----------:|:------------:|:-----:|
| GPU Signal | ✅ | ✅ | ✅ | ✅ (set) |
| GPU Wait | ❌ | ✅ | ✅ | ✅ (wait) |
| CPU Signal | ❌ | ❌ | ✅ (`vkSignalSemaphore`) | ✅ (`vkSetEvent`) |
| CPU Wait | ✅ (`vkWaitForFences`) | ❌ | ✅ (`vkWaitSemaphores`) | ✅ (`vkGetEventStatus`) |
| CPU Query (non-blocking) | ✅ (`vkGetFenceStatus`) | ❌ | ✅ (`vkGetSemaphoreCounterValue`) | ✅ |
| 需 Reset | ✅ (手动) | ❌ (自动 unsignal) | ❌ (单调递增) | ✅ (手动) |
| 多消费者 | ❌ | ❌ (1:1) | ✅ (任意数量) | ❌ |
| Out-of-Order 等待 | ❌ | ❌ | ✅ (wait-before-signal) | ❌ |
| Swapchain 兼容 | ✅ | ✅ | ❌ | ❌ |
| Vulkan 版本 | 1.0 | 1.0 | 1.2 core | 1.0 |

### 4.3 选择指南

```
场景                                    推荐原语
──────────────────────────────────────────────────────
CPU 等待 GPU 完成一批工作              Timeline Semaphore 或 Fence
GPU-GPU 单次同步（同 Queue）           Pipeline Barrier（最快）
GPU-GPU 单次同步（跨 Queue）           Binary Semaphore 或 Timeline Sem
多消费者等待同一生产者                 Timeline Semaphore（唯一方案）
CPU→GPU 触发                          Timeline Semaphore（vkSignalSemaphore）
Swapchain acquire/present             Binary Semaphore（唯一方案）
细粒度资源状态同步（单 Queue 内）     Pipeline Barrier + Event
多线程提前提交（wait-before-signal）   Timeline Semaphore（唯一方案）
```

---

## 5. SPIR-V 内存模型

### 5.1 概述

Vulkan 内存模型由 **SPV_KHR_vulkan_memory_model** 扩展定义，在 SPIR-V 层面提供精细的内存排序控制。GPU 的内存模型比 CPU 的 C++11 内存模型**更宽松**——这既是性能优势也是正确性挑战。

```
关键区别 vs C++11 内存模型：
┌──────────────────────────────────────────────────────────┐
│ C++11                 │ Vulkan/SPIR-V                    │
├────────────────────────┼──────────────────────────────────┤
│ 默认 SeqCst           │ 不支持 SeqCst（必须不用）         │
│ 所有访问默认 atomic   │ 访问默认 Private（不共享）        │
│ data-race = UB        │ data-race = UB                   │
│ 编译器可重排          │ 编译器严格按照 SPIR-V 指令顺序    │
│ Mutex/Lock 保护       │ 无 Mutex — 全部显式 barrier/atomic│
└──────────────────────────────────────────────────────────┘
```

### 5.2 内存访问的可用性与可见性

SPIR-V 内存模型的核心理念是将内存操作的传播分为两个阶段：

```
Availability（可用性）：
  一个写入操作"渗透"到内存系统的过程
  per (位置, 内存域) 跟踪
  目标：让写入值到达特定内存域

Visibility（可见性）：
  一个读取操作"看到"已可用写入的过程  
  per (agent, reference) 跟踪
  目标：让特定 agent 能观察到写入值

过程示意：

  │ 写入者                    │ 读取者                    │
  │ W = 42                    │                           │
  │ MakeAvailable(Device) ────┼──→ (42 available in      │
  │                           │      Device domain)       │
  │                           │ MakeVisible(Device) ──────┼→ 读取到 42
```

### 5.3 内存域 (Memory Domain)

从窄到宽的内存域层次：

```
Subgroup Instance
  └─ Workgroup Instance     ← shared memory 通常在此域
      └─ Shader Call Instance  (Ray Tracing)
          └─ Fragment Interlock Instance
              └─ Queue Family Instance
                  ├─ Shader Domain
                  ├─ Device Domain     ← SSBO 通常需要此域
                  └─ Host Domain       ← CPU 可见需要此域
```

### 5.4 内存语义 (Memory Semantics)

```glsl
// SPIR-V Memory Semantics bits（对应 GLSL memoryBarrier 的语义参数）

// 排序约束
// Relaxed (0x0)         — 无排序约束
// Acquire (0x2)         — 后续内存操作不被重排到此操作之前
// Release (0x4)         — 之前的内存操作不被重排到此操作之后
// AcquireRelease (0x8)  — 两者皆有（用于 atomic RMW）

// 存储类约束
// UniformMemory (0x40)  — 限制到 Uniform/StorageBuffer 类内存
// WorkgroupMemory (0x100)— 限制到 Workgroup 类内存（shared memory）
// ImageMemory (0x800)   — 限制到 Image 类内存

// 域操作
// MakeAvailable (0x2000)  — 执行 Availability 操作
// MakeVisible  (0x4000)   — 执行 Visibility 操作
```

### 5.5 每指令内存访问标志

Vulkan 1.1+ / SPV_KHR_vulkan_memory_model 引入了**每指令**的内存访问标志来替代装饰器：

```glsl
// ---- 旧方式（已废弃）：装饰器 ----
// layout(coherent) buffer Buf { ... };   ← deprecated!

// ---- 新方式（推荐）：每指令标志 ----
// 通过 SPIR-V MemoryAccess bits，无需 GLSL 语法变更
// 编译器从访问模式自动推导

// 下面的 C++ 伪代码展示对应的 SPIR-V 语义：

// MakePointerAvailable — 对指针目标的写入执行 Availability 操作
//   OpStore %ptr %value MakePointerAvailable  ← SPIR-V 指令级
//   使写入对 scope 参数指定的其他 invocation 可见

// MakePointerVisible — 对指针目标的读取执行 Visibility 操作
//   %val = OpLoad %ptr MakePointerVisible     ← SPIR-V 指令级
//   确保读取到其他 invocation 已 MakeAvailable 的写入

// NonPrivatePointer — 声明指针指向非私有（共享）内存
//   没有此标志 = 私有访问（默认假设，允许激进优化）
//   有此标志 = 共享访问（遵守 inter-thread 排序规则）
```

### 5.6 Coherent 和 Volatile 的废弃

```glsl
// Vulkan 1.0 方式（DEPRECATED in Vulkan 1.1+）：
layout(coherent) buffer Buf { float data[]; };    // ❌ 废弃
volatile float v = buf.data[0];                    // ❌ 废弃

// Vulkan 1.1+ 方式（VulkanKHR 内存模型）：
// 使用 MakePointerAvailable/MakePointerVisible 替代 coherent
// 使用 Volatile Memory Access bit 替代 volatile 装饰器
// 优势：
//   ① 每指令粒度（不是每变量）
//   ② 与 Variable Pointers 兼容
//   ③ 更精确地匹配硬件实现
//   ④ 更好的性能（仅在需要同步的指令上使用）
```

### 5.7 Synchronizes-With 关系

```
Synchronizes-With = 基本的跨 invocation 排序关系

四种子关系：
┌────────────────────┬──────────────────────────────────────┐
│ 1. Atomic → Atomic │ Release atomic → Acquire atomic      │
│                    │ 读取到写入值或其 release sequence 中   │
│                    │ 的值                                  │
├────────────────────┼──────────────────────────────────────┤
│ 2. Rel Barrier →   │ Release barrier 之后的 atomic write  │
│    Acq Atomic      │ → Acquire atomic 读取                │
├────────────────────┼──────────────────────────────────────┤
│ 3. Rel Atomic →    │ Release atomic → Acquire barrier     │
│    Acq Barrier     │ 之前的 atomic read                   │
├────────────────────┼──────────────────────────────────────┤
│ 4. Rel Barrier →   │ Release barrier → atomic write →     │
│    Acq Barrier     │ atomic read → Acquire barrier        │
│                    │ 需要中间的 atomic write/read pair     │
└────────────────────┴──────────────────────────────────────┘

重点：纯 non-atomic 的普通 load/store 不参与 synchronizes-with！
必须使用 atomic 操作（或 barrier + atomic pair）来建立 happens-before。
```

### 5.8 Scope（作用域）

Scope 决定了内存操作的同步范围：

```glsl
// SPIR-V Scope 枚举
// Subgroup              — 仅同一 subgroup 内的 invocation
// Workgroup             — 同一 workgroup 内的所有 invocation
// QueueFamilyKHR        — 同一 Queue Family 的所有 invocation
// Device                — 所有 Device invocation
//                       (需要 VulkanMemoryModelDeviceScopeKHR 能力)

// GLSL 中对应的使用：
// gl_ScopeSubgroup
// gl_ScopeWorkgroup
// gl_ScopeQueueFamilyKHR
// gl_ScopeDevice

// 示例：Subgroup-level release/acquire
// 仅在 subgroup 内可见 — 最快
subgroupMemoryBarrier();  // scope = Subgroup

// 示例：Device-level barrier
// 对所有 shader invocation 可见 — 最慢但最广
memoryBarrier();  // scope = Device（配合 atomic）
```

### 5.9 Release Sequence

```
Release Sequence = 一个 Release 操作 A 之后接续的修改序列

定义：
  从 A 开始，在 scoped modification order 上的最长连续子序列
  只包含被任意 agent 执行的 atomic RMW 操作
  （注意：不含 plain atomic write，即使同一 agent 也不含）

作用：
  Acquire 操作读取 Release Sequence 中任意一个值，
  都能与原始 Release 操作 synchronize-with。

为什么排除 plain atomic write？
  Vulkan 有意与 C++ 在此分道扬镳 — 避免编译器需要追踪
  所有 atomic write 的连续性，降低硬件实现复杂度。
```

### 5.10 GLSL 中的内存模型实践

```glsl
#version 450
// 使用 Vulkan 内存模型 — 通过 SPIR-V 编译标志启用
// glslc 不需要特殊 #extension — 由 --target-env 决定

layout(local_size_x = 256) in;

// SSBO 声明（不加 coherent — 由每指令决定同步）
layout(set=0, binding=0) buffer DataBuf {
    uint flags[];  // 用于同步的 atomic flag
    float data[];
};

void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int tid = int(gl_LocalInvocationID.x);

    // ── 生产者 (偶数线程) ──
    if (tid % 2 == 0) {
        data[gid] = compute_result();

        // Release: 确保 data 写入对 Workgroup scope 可见
        // 对应 SPIR-V: OpStore MakePointerAvailable
        // GLSL 中没有直接语法 — 通过 memoryBarrier 或 atomicStore
        atomicStore(flags[0], 1u,
                    gl_ScopeWorkgroup,
                    gl_StorageSemanticsBuffer,
                    gl_SemanticsRelease);
    }

    // ── 消费者 (奇数线程) ──
    if (tid % 2 == 1) {
        // Acquire: 确保读取到生产者的写入
        uint flag = atomicLoad(flags[0],
                               gl_ScopeWorkgroup,
                               gl_StorageSemanticsBuffer,
                               gl_SemanticsAcquire);
        if (flag == 1u) {
            float val = data[gid - 1];  // 读到的值保证是最新的
        }
    }
}
```

---

## 6. Pipeline Barrier 与 Execution/Memory Dependency

### 6.1 两种依赖类型

```
Pipeline Barrier 产生两种独立的依赖：

① Execution Dependency（执行依赖）
   ┌─────────────┐         ┌─────────────┐
   │ Stage A     │ ──→wait │ Stage B     │
   │ (src stage) │         │ (dst stage) │
   └─────────────┘         └─────────────┘
   含义：Stage A 完成 → Stage B 才能开始
   控制：仅 srcStageMask + dstStageMask

② Memory Dependency（内存依赖）
   含义：Stage A 的写入 → 对 Stage B 的读取可见
   控制：srcAccessMask + dstAccessMask

关键洞察：
  - Execution Dependency 保证执行顺序
  - Memory Dependency 保证数据可见性
  - 两者独立：可以有执行依赖但无内存依赖（如 WAR hazard），
    也可以两者都有（如 RAW hazard）
```

### 6.2 Synchronization2 Pipeline Barrier（Vulkan 1.3+）

```cpp
// ── 传统方式（Vulkan 1.0，已不推荐） ──
VkMemoryBarrier old_barrier = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
};
vkCmdPipelineBarrier(cmd,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    0, 1, &old_barrier, 0, nullptr, 0, nullptr);

// ── Synchronization2 方式（Vulkan 1.3+，推荐） ──
VkMemoryBarrier2 barrier = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
    .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
    .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
};

VkDependencyInfo dep_info = {
    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
    .memoryBarrierCount = 1,
    .pMemoryBarriers = &barrier,
};

vkCmdPipelineBarrier2(cmd, &dep_info);
```

### 6.3 Compute Shader 常用 Pipeline Stage / Access Flag

```cpp
// Pipeline Stages (VK_PIPELINE_STAGE_2_*)
VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT          // compute shader 执行
VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT            // 所有命令
VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT             // 最早期（无等待）
VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT           // 最晚期
VK_PIPELINE_STAGE_2_TRANSFER_BIT                // transfer 操作

// Access Flags (VK_ACCESS_2_*)
VK_ACCESS_2_SHADER_STORAGE_READ_BIT             // SSBO 读取
VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT            // SSBO 写入
VK_ACCESS_2_SHADER_SAMPLED_READ_BIT             // texture 读取
VK_ACCESS_2_MEMORY_READ_BIT                     // 所有内存读取
VK_ACCESS_2_MEMORY_WRITE_BIT                    // 所有内存写入
VK_ACCESS_2_SHADER_READ_BIT                     // 所有 shader 读取
VK_ACCESS_2_SHADER_WRITE_BIT                    // 所有 shader 写入
VK_ACCESS_2_TRANSFER_WRITE_BIT                  // transfer 写入

// 对于 compute-to-compute 的 buffer barrier（nnops 最常见模式）：
VkBufferMemoryBarrier2 buf_barrier = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
    .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
    .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    .buffer = my_buffer,
    .offset = 0,
    .size = VK_WHOLE_SIZE,
};
```

### 6.4 只须执行依赖的场景

```cpp
// WAR (Write-After-Read) hazard：只需执行依赖，不需要内存依赖
// 场景：Buffer A 先被 shader 读取，然后被第二个 dispatch 写入

VkMemoryBarrier2 war_barrier = {
    .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .srcAccessMask = VK_ACCESS_2_NONE,              // ← 不需要 access mask！
    .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .dstAccessMask = VK_ACCESS_2_NONE,              // ← 也不需要！
};
// 作用：仅确保第一个 dispatch 完成 → 第二个 dispatch 开始
// 性能：比完整 memory barrier 更快
```

### 6.5 Barrier 性能提示

| Barrier 类型 | 相对开销 | 使用建议 |
|:---|:---|:---|
| 仅 Execution Dependency | 最低 | WAR hazard |
| Buffer Memory Barrier (specific buffer) | 中 | 单 buffer 的 RAW/WAW |
| Image Memory Barrier (specific image) | 中 | 单 image 的状态转换 |
| Global Memory Barrier | 最高 | 多 buffer 的一次性同步 |
| Per-resource barrier (多个) | 中 | 知道具体资源的场景 |

**最佳实践**：优先使用 `VkBufferMemoryBarrier2` / `VkImageMemoryBarrier2` 指定具体资源，而非全局 `VkMemoryBarrier2`。全局 barrier 会刷新所有缓存，成本更高。

---

## 7. Timeline Semaphore 深度解析

### 7.1 设计动机

Timeline Semaphore（Vulkan 1.2 core）用一个**单调递增的 64 位计数器**替换了 Binary Semaphore 和 Fence 的大部分使用场景。

```
Binary Semaphore 的痛点：
├─ 1:1 信号-等待匹配 → N 消费者需 N 个 semaphore
├─ 自动 unsignal → 不能重复等待
├─ Host 不可见 → 无法查询 GPU 进度
└─ 不能 wait-before-signal → 多线程提交必须排序

Timeline Semaphore 的解决方案：
├─ 计数器模型 → 任意数量消费者等同一个值
├─ 单调递增 → 一旦到达值 N，永远 ≥ N
├─ Host 可查询 → vkGetSemaphoreCounterValue（非阻塞）
└─ wait-before-signal → 线程无需同步即可提交
```

### 7.2 完整 C++ API

```cpp
// ── 创建 ──
VkSemaphoreTypeCreateInfo type_info = {
    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
    .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
    .initialValue = 0,  // 起始计数器值
};

VkSemaphoreCreateInfo sem_ci = {
    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    .pNext = &type_info,
};
VkSemaphore timeline_sem;
vkCreateSemaphore(device, &sem_ci, nullptr, &timeline_sem);

// ── Host 端 Signal ──
VkSemaphoreSignalInfo signal_info = {
    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
    .semaphore = timeline_sem,
    .value = 5,  // signal 到值 5
};
vkSignalSemaphore(device, &signal_info);

// ── Host 端 Wait ──
VkSemaphoreWaitInfo wait_info = {
    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
    .semaphoreCount = 1,
    .pSemaphores = &timeline_sem,
    .pValues = &(uint64_t){5},  // 等待值 5
};
vkWaitSemaphores(device, &wait_info, UINT64_MAX);

// ── Host 端非阻塞查询 ──
uint64_t current_value;
vkGetSemaphoreCounterValue(device, timeline_sem, &current_value);
// if (current_value >= 5) { /* 工作已完成 */ }

// ── GPU 端 Signal + Wait (via vkQueueSubmit2) ──
VkSemaphoreSubmitInfo wait_sem_info = {
    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
    .semaphore = timeline_sem,
    .value = 3,   // GPU 等待值 ≥ 3
    .stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
};

VkSemaphoreSubmitInfo signal_sem_info = {
    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
    .semaphore = timeline_sem,
    .value = 4,   // GPU 完成后 signal 到值 4
    .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
};

VkSubmitInfo2 submit = {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
    .waitSemaphoreInfoCount = 1,
    .pWaitSemaphoreInfos = &wait_sem_info,
    .commandBufferInfoCount = 1,
    .pCommandBufferInfos = &cmd_info,
    .signalSemaphoreInfoCount = 1,
    .pSignalSemaphoreInfos = &signal_sem_info,
};
vkQueueSubmit2(queue, 1, &submit, VK_NULL_HANDLE);
```

### 7.3 Wait-Before-Signal（前向等待）

Timeline Semaphore 的独特能力：可以在 signal 之前提交 wait。

```cpp
// 线程 A：提前提交一个 wait（此时 signal 可能还未发生）
// 线程 B：后续提交对应的 signal

// Thread A:
VkSemaphoreSubmitInfo wait_early = {
    .semaphore = timeline_sem, .value = 10,  // 等待值 10
    .stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
};
vkQueueSubmit2(queue, 1, &submit_with_wait, VK_NULL_HANDLE);
// 此时值 10 可能还没被 signal — GPU 会休眠等待

// Thread B (稍后):
VkSemaphoreSubmitInfo signal_late = {
    .semaphore = timeline_sem, .value = 10,  // signal 到 10
    .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
};
vkQueueSubmit2(queue, 1, &submit_with_signal, VK_NULL_HANDLE);
// GPU 现在可以继续执行线程 A 等待的工作
```

**优势**：多线程可以独立提交工作，无需跨线程同步。

**注意事项**：
- 如果 signal 永远不来 → GPU 永久挂起（类似死锁）
- `vkDeviceWaitIdle` 无法解决这种死锁 → 使用 `vkWaitSemaphores` drain GPU

### 7.4 替代 Fence 管理 Flight Frames

```cpp
// ── 传统方式：MAX_FRAMES_IN_FLIGHT 个 Fence + Ring Buffer ──
const int MAX_FRAMES = 2;
VkFence fences[MAX_FRAMES];
int frame_index = 0;

vkQueueSubmit(queue, ..., fences[frame_index]);  // signal fence
frame_index = (frame_index + 1) % MAX_FRAMES;
vkWaitForFences(device, 1, &fences[frame_index], VK_TRUE, UINT64_MAX);
vkResetFences(device, 1, &fences[frame_index]);
// 复杂：管理 ring buffer、确保 fence 在正确状态

// ── 现代方式：单个 Timeline Semaphore ──
uint64_t timeline_value = 0;

VkSemaphoreSubmitInfo signal_info = {
    .semaphore = timeline_sem,
    .value = ++timeline_value,  // 每次提交递增
    .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
};

vkQueueSubmit2(queue, 1, &submit, VK_NULL_HANDLE);

// 等待最旧的待处理工作（确保不超出 MAX_FRAMES）
if (timeline_value > MAX_FRAMES) {
    uint64_t wait_value = timeline_value - MAX_FRAMES;
    vkWaitSemaphores(device, &wait_info_with_value(wait_value), UINT64_MAX);
    // 无需 reset！
}

// 查询 GPU 进度（非阻塞）
uint64_t completed;
vkGetSemaphoreCounterValue(device, timeline_sem, &completed);
float progress = (float)completed / timeline_value;
```

### 7.5 多消费者模式

```cpp
// 一次 signal，多个 queue 等待 — 不需要多个 semaphore！

// Producer queue: signal value 5
vkQueueSubmit2(producer_queue, 1, &submit_signal_5, ...);

// Consumer queue 1: wait value 5
vkQueueSubmit2(consumer_queue_1, 1, &submit_wait_5, ...);

// Consumer queue 2: wait value 5 (同一个 semaphore!)
vkQueueSubmit2(consumer_queue_2, 1, &submit_wait_5, ...);

// Consumer queue 3: wait value 5
vkQueueSubmit2(consumer_queue_3, 1, &submit_wait_5, ...);
// 全部合法 — 不需要 3 个 binary semaphore
```

### 7.6 局限：WSI 不兼容

```
Timeline Semaphore 不能用于：
├─ vkAcquireNextImageKHR — 必须用 Binary Semaphore（或 Fence）
├─ vkQueuePresentKHR — 必须用 Binary Semaphore
└─ 原因：WSI 扩展规范未更新以支持 Timeline Semaphore

变通方案：
  1. 内部同步全部用 Timeline Semaphore
  2. 仅在 WSI 边界使用 Binary Semaphore
  3. 通过 vkQueueSubmit2 的 wait/signal 完成两者之间的桥接
```

---

## 8. Host-Device 同步模式

### 8.1 CPU 等待 GPU 完成

```cpp
// ── 模式 1：Fence（传统，Vulkan 1.0） ──
VkFenceCreateInfo fence_ci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
VkFence fence;
vkCreateFence(device, &fence_ci, nullptr, &fence);

vkQueueSubmit(queue, 1, &submit, fence);

// 阻塞等待
vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

// 非阻塞查询
VkResult result = vkGetFenceStatus(device, fence);
// VK_SUCCESS = signaled, VK_NOT_READY = 还在执行

vkDestroyFence(device, fence, nullptr);

// ── 模式 2：Timeline Semaphore（推荐，Vulkan 1.2+） ──
uint64_t signal_value = ++timeline_counter;

VkSemaphoreSubmitInfo signal_info = {
    .semaphore = timeline_sem,
    .value = signal_value,
    .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
};
vkQueueSubmit2(queue, 1, &submit, VK_NULL_HANDLE);

// 阻塞等待
VkSemaphoreWaitInfo wait_info = {
    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
    .semaphoreCount = 1,
    .pSemaphores = &timeline_sem,
    .pValues = &signal_value,
};
vkWaitSemaphores(device, &wait_info, UINT64_MAX);

// 非阻塞查询
uint64_t current;
vkGetSemaphoreCounterValue(device, timeline_sem, &current);
// if (current >= signal_value) { /* 完成 */ }
```

### 8.2 CPU 触发 GPU 工作

```cpp
// 只有 Timeline Semaphore 支持 Host → GPU signal

// CPU 端 "踢" GPU
VkSemaphoreSignalInfo signal_info = {
    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
    .semaphore = timeline_sem,
    .value = 1,  // CPU 设置值
};
vkSignalSemaphore(device, &signal_info);

// GPU 端等待 CPU 的 signal
VkSemaphoreSubmitInfo wait_info = {
    .semaphore = timeline_sem,
    .value = 1,  // GPU 等待 CPU 设置的值
    .stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
};
vkQueueSubmit2(queue, 1, &submit, VK_NULL_HANDLE);
```

### 8.3 GPU 完成后的 Readback

```cpp
// ── 完整的 GPU → CPU readback 流程 ──

// ① 创建 host-visible staging buffer
VkBuffer staging;
VkDeviceMemory staging_mem;
allocate_staging_buffer(&staging, &staging_mem, data_size);

// ② Record GPU → staging copy
vkCmdCopyBuffer(cmd, gpu_buffer, staging_buffer, 1, &copy_region);

// ③ Submit + signal timeline semaphore
uint64_t readback_value = ++timeline_counter;
vkQueueSubmit2(queue, 1, &submit_with_signal(readback_value), VK_NULL_HANDLE);

// ④ CPU wait
vkWaitSemaphores(device, &wait_info_for(readback_value), UINT64_MAX);

// ⑤ Map + read
void* mapped;
vkMapMemory(device, staging_mem, 0, data_size, 0, &mapped);
memcpy(cpu_data, mapped, data_size);
vkUnmapMemory(device, staging_mem);

// ⑥ 可选：invalidate cache（non-coherent memory）
VkMappedMemoryRange range = {
    .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
    .memory = staging_mem,
    .offset = 0,
    .size = data_size,
};
vkInvalidateMappedMemoryRanges(device, 1, &range);
```

### 8.4 Non-Coherent Memory 的 Cache 管理

```cpp
// 当 memory type 不是 HOST_COHERENT 时，需要显式 cache 管理：

// GPU 写入 → CPU 读取：
vkInvalidateMappedMemoryRanges(device, 1, &range);

// CPU 写入 → GPU 读取：
vkFlushMappedMemoryRanges(device, 1, &range);

// 查询 memory type 是否 coherent：
VkPhysicalDeviceMemoryProperties mem_props;
vkGetPhysicalDeviceMemoryProperties(phys_dev, &mem_props);
bool is_coherent = mem_props.memoryTypes[type_index].propertyFlags
                 & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
```

---

## 9. 多线程实用模式

### 9.1 每线程 Command Pool（基本模式）

```cpp
class WorkerThread {
    VkCommandPool cmd_pool_;
    std::vector<VkCommandBuffer> cmds_;

    void init(VkDevice device, uint32_t queue_family_index) {
        VkCommandPoolCreateInfo pool_ci = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = queue_family_index,
        };
        vkCreateCommandPool(device, &pool_ci, nullptr, &cmd_pool_);
    }

    VkCommandBuffer record_work() {
        VkCommandBuffer cmd;
        VkCommandBufferAllocateInfo alloc_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = cmd_pool_,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        vkAllocateCommandBuffers(device, &alloc_info, &cmd);

        vkBeginCommandBuffer(cmd, &begin_info);
        // ... 录制 compute dispatches ...
        vkEndCommandBuffer(cmd);

        return cmd;  // 传递给提交线程
    }
};
```

### 9.2 生产者-消费者提交模式

```cpp
// 多个录制线程 → 一个提交线程

std::mutex submission_mutex;
std::vector<VkCommandBuffer> pending_submissions;
uint64_t next_timeline_value = 0;

// ── 录制线程 ──
void record_and_submit(int thread_id) {
    VkCommandBuffer cmd = worker_threads[thread_id].record_work();

    std::lock_guard lock(submission_mutex);
    pending_submissions.push_back(cmd);
}

// ── 提交线程 ──
void drain_submissions() {
    std::vector<VkCommandBuffer> batch;
    {
        std::lock_guard lock(submission_mutex);
        batch.swap(pending_submissions);
    }

    if (batch.empty()) return;

    // 批量提交
    std::vector<VkCommandBufferSubmitInfo> cmd_infos;
    for (auto cmd : batch) {
        cmd_infos.push_back({ .commandBuffer = cmd });
    }

    uint64_t signal_value = ++next_timeline_value;
    VkSemaphoreSubmitInfo signal_info = {
        .semaphore = timeline_sem,
        .value = signal_value,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
    };

    VkSubmitInfo2 submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .commandBufferInfoCount = (uint32_t)cmd_infos.size(),
        .pCommandBufferInfos = cmd_infos.data(),
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &signal_info,
    };

    vkQueueSubmit2(queue, 1, &submit, VK_NULL_HANDLE);
}
```

### 9.3 无锁多线程提交（Wait-Before-Signal）

```cpp
// 利用 Timeline Semaphore 的 wait-before-signal 特性
// 多个线程可以完全独立提交，无需任何跨线程同步！

std::atomic<uint64_t> g_timeline{0};

void thread_work(int thread_id, VkQueue queue) {
    // 每个线程有自己的 command pool
    VkCommandBuffer cmd = allocate_and_record(thread_id);

    uint64_t my_value = g_timeline.fetch_add(1) + 1;

    // 等待前一个提交（无论哪个线程）
    VkSemaphoreSubmitInfo wait_info = {
        .semaphore = timeline_sem,
        .value = my_value - 1,  // 等前一个
        .stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    };

    // Signal 自己的完成
    VkSemaphoreSubmitInfo signal_info = {
        .semaphore = timeline_sem,
        .value = my_value,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
    };

    VkSubmitInfo2 submit = { /* wait_info + cmd + signal_info */ };
    vkQueueSubmit2(queue, 1, &submit, VK_NULL_HANDLE);
    // 无需 mutex！
}
```

### 9.4 Double-Buffered Readback

```cpp
// 对 nnops 测试最有用的模式：读出 GPU 结果无 stall

struct ReadbackSlot {
    VkBuffer staging;
    VkDeviceMemory staging_mem;
    uint64_t fence_value;   // 这个 slot 何时完成
    bool in_flight;
};

ReadbackSlot slots[2];
int current_slot = 0;

void async_readback(VkBuffer gpu_buffer, size_t size, void* cpu_dst) {
    ReadbackSlot& slot = slots[current_slot];

    // 等待当前 slot 的上一轮 readback 完成
    if (slot.in_flight) {
        vkWaitSemaphores(device, &wait_info_for(slot.fence_value), UINT64_MAX);
        slot.in_flight = false;
    }

    // Copy GPU → staging
    vkCmdCopyBuffer(cmd, gpu_buffer, slot.staging, 1, &copy_region);

    // Submit + signal
    uint64_t signal_value = ++timeline_counter;
    vkQueueSubmit2(queue, 1, &submit_with_signal(signal_value), VK_NULL_HANDLE);

    // Wait
    vkWaitSemaphores(device, &wait_info_for(signal_value), UINT64_MAX);

    // Invalidate + read
    vkInvalidateMappedMemoryRanges(device, 1, &range);
    void* mapped;
    vkMapMemory(device, slot.staging_mem, 0, size, 0, &mapped);
    memcpy(cpu_dst, mapped, size);
    vkUnmapMemory(device, slot.staging_mem);

    // 记录状态
    slot.fence_value = signal_value;
    slot.in_flight = true;

    // 切换 slot
    current_slot = (current_slot + 1) % 2;
}
```

### 9.5 Persistent Mapping（Ring Buffer 模式）

```cpp
// 对于重复 readback 的优化：
// 使用 persistently mapped ring buffer 而非每次 map/unmap

struct PersistentRingBuffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void* mapped_ptr;     // 一次 map，永远不 unmap
    size_t total_size;
    std::atomic<size_t> write_offset{0};
};

void init_ring_buffer(PersistentRingBuffer* rb, size_t size) {
    allocate_staging_buffer(&rb->buffer, &rb->memory, size);
    vkMapMemory(device, rb->memory, 0, size, 0, &rb->mapped_ptr);
    rb->total_size = size;
}

// GPU 写入 ring buffer 的某段
void gpu_write_to_ring(VkCommandBuffer cmd, VkBuffer src,
                       PersistentRingBuffer* rb, size_t size) {
    size_t offset = rb->write_offset.fetch_add(size);
    // wrap-around 处理省略...
    vkCmdCopyBuffer(cmd, src, rb->buffer, 1, &copy_region_at(offset));
}

// CPU 直接读取 mapped_ptr — 无需 unmap！
void cpu_read_from_ring(PersistentRingBuffer* rb, size_t offset,
                        void* dst, size_t size) {
    memcpy(dst, (uint8_t*)rb->mapped_ptr + offset, size);
}
```

---

## 10. nnops 相关考量

### 10.1 当前 nnops 的同步模式

```
当前 nnops Vulkan 后端的同步模型：

每个 operator 调用：
  vkBeginCommandBuffer(cmd)
  ├─ vkCmdPipelineBarrier  (buffer ownership transfer, if needed)
  ├─ vkCmdBindDescriptorSets
  ├─ vkCmdPushConstants
  ├─ vkCmdBindPipeline
  ├─ vkCmdDispatch
  └─ vkEndCommandBuffer(cmd)

  vkQueueSubmit(queue, 1, &submit, fence)
  vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX)
  vkResetFences(...)

特点：
  ✅ 简单、正确
  ⚠️ 每次 dispatch 后全 stall（wait-before-next-op）
  ⚠️ 无法利用 GPU 流水线并发（dispatch N+1 要等 N 的 readback）
```

### 10.2 可以改进的方向

| 改进 | 收益 | 难度 |
|------|------|:---:|
| **Timeline Semaphore 替代 Fence** | 更灵活的非阻塞查询，无需 fence reset | 低 |
| **Synchronization2 替代旧 barrier API** | 更清晰的 64-bit stage/access mask | 低 |
| **Persistent mapped staging buffer** | 去除 map/unmap 开销（readback 热路径） | 中 |
| **异步 readback 流水线** | 读回帧 N 的同时执行帧 N+1 | 中 |
| **多线程 command recording** | 多个 operator 并行录制 command buffer | 中 |
| **VK_KHR_push_descriptor** | 每 dispatch push descriptor，消除 pool 管理 | 低 |

### 10.3 nnops 中内存模型的简化视图

```
nnops 当前全部 operator 为 "单 dispatch、无跨 workgroup 通信" 模式：

Shader 内：
  无 shared memory（当前 eltwise/unary）
  无 atomic 操作
  无跨 workgroup 依赖
  → 不需要 barrier() 或 memoryBarrier()

Shader 间（Host 端同步）：
  所有 operator 顺序执行
  fence-stall 保证前一个 dispatch 的内存写入全部可见
  → 隐式满足内存模型要求

对 nnops 的意义：
  当前模式在 Vulkan 内存模型中是安全的
  — 但未来引入 shared memory / atomic 时需要显式 barrier
```

---

## 11. 快速参考卡片

### 线程模型速查

| 问题 | 答案 |
|------|------|
| Command Pool 可以多线程共享吗？ | ❌ 必须 externally-synchronized；推荐每线程一个 |
| 同一 Pool 的多个 CmdBuffer 可以并发录制吗？ | ❌ 不可以 — 父 Pool 被隐式锁定 |
| 录制完成的 CmdBuffer 可以跨线程传递吗？ | ✅ 可以 — Executable 状态后自由传递 |
| Descriptor Pool 可以多线程共享吗？ | ❌ 需要外部同步；推荐每线程一个 |
| 多个线程可以同时 vkQueueSubmit 吗？ | ✅ 可以（不同 Queue 或同一 Queue 外部同步） |
| VkDevice 操作是线程安全的吗？ | 部分 — create/destroy 需同步，allocate/free 可并发 (1.1+) |

### GPU 层次速查

| 层次 | 硬件单位 | 同步方式 | 内存共享 |
|------|----------|----------|----------|
| Invocation | 单线程 | 隐式（同一 wave 内） | 寄存器 |
| Subgroup (32/64) | Warp/Wavefront | subgroupBarrier() | subgroup shuffle |
| Workgroup (≤1024) | SM/CU | barrier() | shared memory (LDS) |
| Global Grid | 全部 CU | atomic + memoryBarrier | SSBO (global memory) |

### 同步原语速查

| 原语 | 何时用 |
|------|--------|
| `vkCmdPipelineBarrier2` | 同一 Queue 内 stage→stage 转换 |
| Binary Semaphore | Swapchain acquire/present |
| Timeline Semaphore | 所有其他场景（GPU-GPU、CPU-GPU、多消费者） |
| Fence | 仅当 Vulkan < 1.2 或不需 swapchain 时用 |
| Event | 细粒度 GPU 内条件同步 |
| `vkQueueSubmit2` | 现代 Queue 提交（配合 Timeline Semaphore） |

### 内存模型速查

| 操作 | GLSL / SPIR-V | 作用域 |
|------|---------------|--------|
| 读取 shared memory 的写入 | `barrier() + memoryBarrierShared()` | Workgroup |
| 读取 SSBO 的写入（同 workgroup） | `memoryBarrierBuffer()` | Workgroup + |
| 读取 SSBO 的写入（跨 workgroup） | `atomicStore(Release) + atomicLoad(Acquire)` | Device |
| 非私有指针声明 | `NonPrivatePointer` (SPIR-V flag) | Device |
| 使写入对 Device 可见 | `MakePointerAvailable + Device scope` | Device |

### Synchronization2 Pipeline Stage 速查

```
VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT            ← 最常用
VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT              ← signal (最迟)
VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT               ← wait (最早)
VK_PIPELINE_STAGE_2_TRANSFER_BIT                  ← buffer copy
VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT
VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT
VK_PIPELINE_STAGE_2_HOST_BIT
```

### Synchronization2 Access Flag 速查

```
VK_ACCESS_2_SHADER_STORAGE_READ_BIT       ← SSBO 读取
VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT      ← SSBO 写入
VK_ACCESS_2_SHADER_READ_BIT               ← 所有 shader 读取
VK_ACCESS_2_SHADER_WRITE_BIT              ← 所有 shader 写入
VK_ACCESS_2_MEMORY_READ_BIT               ← 所有内存读取
VK_ACCESS_2_MEMORY_WRITE_BIT              ← 所有内存写入
VK_ACCESS_2_TRANSFER_READ_BIT             ← transfer 读取
VK_ACCESS_2_TRANSFER_WRITE_BIT            ← transfer 写入
VK_ACCESS_2_NONE                           ← 仅执行依赖
```
