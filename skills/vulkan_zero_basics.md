# Vulkan 零基础入门：从 C++ 到第一次 Compute Dispatch

> **读者假设**：你会写 C++，知道指针、结构体、类、线程、`malloc/free`、编译与链接。
>
> **你不需要知道**：Vulkan、OpenGL、CUDA、GPU 架构、图形学。
>
> **读完本文你能做到**：
>
> - 看懂 nnops 里任意一个 `src/backend/vulkan/<op>_vulkan.cpp`；
> - 说清楚 shader、SPIR-V、pipeline、layout、descriptor set、push constant 分别是什么、为什么缺一不可；
> - 自己写出一个最小的 Vulkan compute 程序；
> - 判断一段 Vulkan 代码在生命周期、绑定、同步上有没有坑。
>
> **写法约定**：每出现一个新名词，先给一个最小例子或一段代码，再解释它的定义。**不集中列举术语。** 如果你读到某段发现有个词没解释，说明我漏了，请告诉我补上。

---

## 目录

0. [怎么读这篇文档](#0-怎么读这篇文档)
1. [基础词汇：GPU、驱动、API、句柄、对象](#1-基础词汇gpu驱动api句柄对象)
2. [为什么 Vulkan 看起来这么复杂](#2-为什么-vulkan-看起来这么复杂)
3. [核心类比：一次 GPU 计算 = 一次"远程函数调用"](#3-核心类比一次-gpu-计算--一次远程函数调用)
4. [对象地图：Vulkan 程序里到底有哪些东西](#4-对象地图vulkan-程序里到底有哪些东西)
5. [第一次创建对象：搭建 Vulkan 环境](#5-第一次创建对象搭建-vulkan-环境)
6. [Buffer 与 DeviceMemory：数据放在哪](#6-buffer-与-devicememory数据放在哪)
7. [CommandBuffer：先录制，后执行](#7-commandbuffer先录制后执行)
8. [Shader、GLSL、SPIR-V、ShaderModule](#8-shaderglslspir-vshadermodule)
9. [Descriptor：GPU 怎么拿到数据](#9-descriptorgpu-怎么拿到数据)
10. [三个 "layout" 千万别搞混](#10-三个-layout-千万别搞混)
11. [Pipeline：编译好的 kernel](#11-pipeline编译好的-kernel)
12. [特化常量与推送常量](#12-特化常量与推送常量)
13. [一次 dispatch 的完整生命周期](#13-一次-dispatch-的完整生命周期)
14. [逐个对象详解](#14-逐个对象详解)
15. [Shader 与 C++ 的逐行对应](#15-shader-与-c-的逐行对应)
16. [Workgroup 与 dispatch 的映射](#16-workgroup-与-dispatch-的映射)
17. [nnops 对象映射总表](#17-nnops-对象映射总表)
18. [常见误解 FAQ](#18-常见误解-faq)
19. [术语表](#19-术语表)
20. [下一步读什么](#20-下一步读什么)

---

## 0. 怎么读这篇文档

### 0.1 这篇文档的路线

```text
第 1 章   用具体例子把 GPU、驱动、API、句柄、对象这些词逐个带出来
第 2 章   解释"为什么 Vulkan 这么啰嗦"
第 3 章   给出全文的核心类比（函数调用）
第 4 章   鸟瞰所有对象，知道它们谁包含谁
第 5 章   从零创建第一批对象（Instance / Device / Queue / Pool）
第 6 章   解释数据怎么放（Buffer + Memory）
第 7 章   解释命令怎么录、怎么提交（CommandBuffer + Queue + Fence）
第 8 章   解释 shader 是什么、怎么变成 SPIR-V
第 9 章   解释 GPU 怎么拿到数据（Descriptor）
第 10 章  把三种 "layout" 一次讲透
第 11 章  解释 pipeline 是什么、为什么贵
第 12 章  解释两种常量
第 13 章  把所有东西串成一次完整调用
第 14 章  逐对象速查
第 15 章  把 shader 和 C++ 逐行对上
第 16 章  解释 GPU 的线程模型
第 17 章  映射回 nnops 的代码
第 18 章  FAQ
第 19 章  术语表
```

### 0.2 如果你只想尽快看懂代码

读第 3、4、10、11、13、17 章。这六章覆盖了 nnops Vulkan 后端的全部核心概念。

### 0.3 如果你要自己写一个最小程序

从第 5 章开始按顺序读，每章都有可运行的代码骨架。

---

## 1. 基础词汇：GPU、驱动、API、句柄、对象

### 1.1 GPU 和 CPU 到底差在哪

你熟悉 CPU：几个到几十个很强的核心，支持乱序执行、分支预测、多级缓存、虚拟内存。CPU 擅长"很多不同的判断和跳转"。

GPU 是另一种东西：

| 维度 | CPU | GPU |
| --- | --- | --- |
| 核心数 | 4–64 个强核心 | 几百到几千个简单核心 |
| 执行方式 | 每个核心独立跑不同代码 | 大量核心**同时跑同一段代码**，只是数据不同（SIMT） |
| 内存 | 和 CPU 共享系统内存，缓存一致 | 通常有自己的显存；访问系统内存要过 PCIe，慢很多 |
| 擅长 | 分支多、逻辑复杂的串行任务 | 同样的算术做几百万次（逐元素、矩阵乘） |
| 内存保护 | 越界访问会段错误 | 很多情况下**没有页保护**，越界读写是未定义行为 |

最后一行非常重要：在 CPU 上，数组越界通常立刻 `segfault`；在 GPU 的 shader 里，越界读写一个 buffer 可能悄无声息地读到垃圾数据，或者把别的数据写坏。所以 Vulkan shader 里**必须自己做边界检查**。

### 1.2 Vulkan 不是"一个库"

这是新手最容易误解的一点。你 `#include <vulkan/vulkan.h>` 拿到的不是实现，而是：

1. **函数声明**（`vkCreateBuffer` 等）；
2. **枚举、结构体、常量**；
3. **函数指针的加载机制**。

真正的实现分散在：

```text
你的程序
   │  调用 vkCreateBuffer(...)
   ▼
Vulkan Loader（vulkan-1.dll / libvulkan.so）   ← 系统自带或 SDK 安装
   │  按 GPU 厂商转发
   ▼
ICD（厂商驱动，例如 Qualcomm 的 libvulkan_adreno.so）  ← 真正干活
   │
   ▼
GPU 硬件
```

**对你的影响**：

- 不同厂商、不同驱动版本对同一个函数的实现可能不同；
- 某些功能是"可选"的，必须在运行时查询；
- 驱动有 bug 是常态，尤其是新扩展；
- 你写的代码必须在"规范允许"的范围内，不能依赖某个驱动的巧合行为。

### 1.3 Vulkan 函数的三层

Vulkan 的函数按"作用于谁"分成三类，这是读懂函数名的关键：

| 层级 | 前缀/参数 | 例子 | 说明 |
| --- | --- | --- | --- |
| 全局 | 没有对象参数 | `vkCreateInstance`、`vkEnumerateInstanceExtensionProperties` | 不需要任何已有对象 |
| Instance 级 | 第一个参数是 `VkInstance` 或 `VkPhysicalDevice` | `vkEnumeratePhysicalDevices`、`vkGetPhysicalDeviceProperties` | 查询"这台机器有什么" |
| Device 级 | 第一个参数是 `VkDevice` | `vkCreateBuffer`、`vkCreatePipeline`、`vkQueueSubmit` | 真正创建/使用资源 |

记忆方法：**参数列表第一个是 `VkDevice` 的，就是设备级函数**。设备级函数通常可以通过函数指针表直接调用，开销更低。

### 1.4 句柄（handle）是什么

Vulkan 的所有对象都用"句柄"表示。句柄的定义大致是：

```cpp
// 可分派句柄（dispatchable）：本质是一个指针
#define VK_DEFINE_HANDLE(object) typedef struct object##_T* object;

// 不可分派句柄（non-dispatchable）：
//   32 位平台 → uint64_t；64 位平台（VK_USE_64_BIT_PTR_DEFINES=1）→ 指针
#if VK_USE_64_BIT_PTR_DEFINES
#define VK_DEFINE_NON_DISPATCHABLE_HANDLE(object) typedef struct object##_T* object;
#else
#define VK_DEFINE_NON_DISPATCHABLE_HANDLE(object) typedef uint64_t object;
#endif
```

例如：

```cpp
VkDevice      device;      // 本质是 struct VkDevice_T*
VkBuffer      buffer;      // 本质是一个 64 位 ID（或指针，取决于平台）
VkPipeline    pipeline;    // 同上
```

**关键点**：

- 你不能 `new VkBuffer`，也不能 `*buffer` 解引用；
- 它只是驱动内部对象的一个编号/门票；
- 创建用 `vkCreate*`，销毁用 `vkDestroy*`，必须成对；
- 无效句柄用 `VK_NULL_HANDLE` 表示（就是 0）。

"可分派"和"不可分派"的区别，简单说：**可分派句柄可以作为函数第一个参数，用来查函数表；不可分派句柄不能**。你写业务代码时基本不用关心这个区别，只要知道它们都是不透明 ID。

### 1.5 对象（object）与 create/destroy 配对

Vulkan 的每个对象都遵循同一个模式：

```cpp
VkXxxCreateInfo ci = {};
ci.sType = VK_STRUCTURE_TYPE_XXX_CREATE_INFO;
// ... 填参数 ...
VkXxx object = VK_NULL_HANDLE;
VkResult res = vkCreateXxx(device, &ci, nullptr, &object);

// 用完
vkDestroyXxx(device, object, nullptr);
```

这个模式有五个固定部分，逐个解释：

| 部分 | 是什么 | 为什么需要 |
| --- | --- | --- |
| `VkXxxCreateInfo` | 一个装满创建参数的结构体 | Vulkan 不想写几十个参数的函数，统一用结构体传参 |
| `ci.sType` | 一个枚举，标明"我是哪种 CreateInfo" | 驱动靠它做类型检查/分发 |
| `&ci` | 指向参数结构体的指针 | 传参 |
| `nullptr`（第三个参数） | 自定义内存分配器，通常传 `nullptr` | 高级用法，入门不用管 |
| `&object` | 输出参数，驱动把新对象的句柄写进来 | C 风格返回 |

**如果 `sType` 写错会怎样**：校验层会报错；没有校验层时，驱动可能读错结构体布局，产生未定义行为。所以每个结构体创建后第一件事就是设 `sType`。

### 1.6 `sType` 和 `pNext`：Vulkan 的通用扩展机制

大多数 Vulkan 结构体都有这两个字段：

```cpp
typedef struct VkBufferCreateInfo {
    VkStructureType    sType;   // 我是谁
    const void*        pNext;   // 我后面还链着谁
    VkBufferCreateFlags flags;
    VkDeviceSize       size;
    VkBufferUsageFlags usage;
    VkSharingMode      sharingMode;
    // ...
} VkBufferCreateInfo;
```

- `sType`：必须填对应的 `VK_STRUCTURE_TYPE_*`。
- `pNext`：指向另一个结构体，形成一条链表。用于"给已有结构体追加可选参数"，比如：

```cpp
VkPhysicalDevice16BitStorageFeatures storage = {};
storage.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
storage.storageBuffer16BitAccess = VK_TRUE;

VkDeviceCreateInfo dci = {};
dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
dci.pNext = &storage;   // 把 16-bit storage 特性链进去
```

**记忆**：`sType` 是"我是谁"，`pNext` 是"我还带了谁"。

### 1.7 `VkResult`：错误怎么返回

Vulkan 不用异常，所有可能失败的函数返回 `VkResult`：

```cpp
typedef enum VkResult {
    VK_SUCCESS = 0,
    VK_NOT_READY = 1,
    VK_TIMEOUT = 2,
    // ...
    VK_ERROR_INITIALIZATION_FAILED = -3,
    VK_ERROR_DEVICE_LOST = -4,
    VK_ERROR_OUT_OF_HOST_MEMORY = -1,
    VK_ERROR_OUT_OF_DEVICE_MEMORY = -2,
    // ...
} VkResult;
```

规则：

- `VK_SUCCESS`（0）表示成功；
- 正数通常是"成功但有情况"（比如 `VK_NOT_READY`）；
- 负数表示失败。

nnops 的 `vulkan_common.hpp` 里定义了一个宏：

```cpp
#define VK_CHECK(expr) nnops::backend::vulkan::vk_check(expr, __FILE__, __LINE__, #expr)
```

它检查返回值，失败就打印文件名、行号、表达式并 `abort()`。**新代码都应该用它**，不要忽略 `VkResult`。

### 1.8 Version、Extension、Feature 三者的区别

这三个词经常混，必须分清：

| 概念 | 例子 | 怎么查询 | 怎么启用 |
| --- | --- | --- | --- |
| **版本 (version)** | Vulkan 1.0 / 1.1 / 1.2 / 1.3 / 1.4 | `vkEnumerateInstanceVersion`、`VkPhysicalDeviceProperties::apiVersion` | 创建 Instance 时传 `apiVersion` |
| **扩展 (extension)** | `VK_KHR_16bit_storage`、`VK_KHR_cooperative_matrix` | `vkEnumerateDeviceExtensionProperties` | 创建 Device 时放进 `ppEnabledExtensionNames` |
| **特性 (feature)** | `storageBuffer16BitAccess`、`shaderFloat16` | `vkGetPhysicalDeviceFeatures2` | 创建 Device 时通过 `pNext` 链入对应的 Features 结构体 |

关系：

```text
版本      = 核心功能集合（1.0 有 A，1.1 增加 B，……）
扩展      = 核心之外追加的一组函数/枚举/结构体
特性      = 某个扩展或核心能力里的一个布尔开关
```

**典型错误**：只启用了扩展，忘了启用特性；或者反过来。两者都要满足，功能才能用。

### 1.9 校验层（Validation Layer）

Vulkan 的默认行为是"不检查"：你绑错 layout、用已销毁的对象、越界写 buffer，驱动可能什么都不说，结果就是错的，或者程序崩溃在别处。

**校验层**是开发期插入的一层"检查器"：

```cpp
const char* layers[] = { "VK_LAYER_KHRONOS_validation" };
VkInstanceCreateInfo ci = {};
ci.enabledLayerCount = 1;
ci.ppEnabledLayerNames = layers;
```

它会在你每次 API 调用时检查参数合法性，把问题打印到 stderr。**开发和测试时一定要开**，发布时关掉（有性能开销，而且不是所有设备都安装了校验层）。

nnops 的测试环境目前没有启用校验层。如果你在改 Vulkan 后端，建议本地加上，能省很多调试时间。

### 1.10 "未定义行为"在 Vulkan 里意味着什么

C++ 里的 UB 你已经熟悉了。Vulkan 里的 UB 更隐蔽：

| 操作 | 可能的结果 |
| --- | --- |
| shader 越界读 SSBO | 读到垃圾数据、读到别的 buffer、GPU 挂起、驱动崩溃 |
| 绑定不匹配的 descriptor set | 读到错误数据，或校验层报错 |
| 使用已销毁的 buffer | 驱动崩溃、画面花屏、结果随机 |
| 在命令执行期间修改 descriptor set | 结果不确定 |
| `sType` 写错 | 驱动按错误布局解析，崩溃 |

**关键认知**：Vulkan 的很多错误**不会**在出错的那一行暴露，而是在几帧之后以完全无关的形式出现。所以生命周期和绑定规则必须一开始就学对。

### 1.11 外部同步（externally synchronized）

Vulkan 规范里大量对象标注为 "externally synchronized"，意思是：

> 如果两个线程同时访问这个对象，**你自己**必须保证它们不会同时发生。驱动不做任何加锁。

典型需要外部同步的对象：

- `VkCommandPool`：同一 pool 不能并发分配/重置/录制；
- `VkDescriptorPool`：同一 pool 不能并发分配/重置；
- `VkQueue`：同一队列的提交需要同步；
- `VkDevice`：部分操作（创建/销毁）需要同步。

这和多线程 CPU 编程的规则一样，只是 Vulkan 把责任明确写进了规范。第 7 章会讲 nnops 的做法。

### 1.12 本节小结

到这里的名词，后面会反复出现：

```text
GPU 通过驱动执行命令；Vulkan 是一套规范，不是库；
所有东西都是句柄；创建用 CreateInfo + vkCreate，销毁用 vkDestroy；
sType 标类型，pNext 链扩展；VkResult 返回错误；
版本/扩展/特性三件套都要查；开发时开校验层；
Vulkan 的 UB 很隐蔽；多线程访问共享对象要自己加锁。
```

---

## 2. 为什么 Vulkan 看起来这么复杂

### 2.1 三种 API 风格对比

同样是"在 GPU 上跑一个逐元素加法"，三种 API 的写法完全不同。

#### CUDA：最接近普通 C++

```cpp
// 一个函数调用，传指针、传参数、启动
my_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, n);
```

CUDA 的运行时帮你做了：内存拷贝、参数打包、kernel 启动、错误检查。你只需关心"函数 + 指针 + 网格大小"。

#### OpenGL：全局状态机

```cpp
glUseProgram(prog);
glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf_a);
glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf_a);
glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf_b);
glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buf_b);
// ...再绑一堆...
glDispatchCompute(groups, 1, 1);
```

OpenGL 有一个全局的"当前状态"：当前 program、当前 buffer、当前纹理……你每次都要先把它设成想要的值。驱动在背后做大量检查和同步。

#### Vulkan：显式对象 + 命令录制

```cpp
VkDescriptorSet ds = ...;                    // 先准备"参数包"
vkUpdateDescriptorSets(...);                 // 把 buffer 塞进参数包
vkCmdBindPipeline(cmd, COMPUTE, pipeline);   // 绑定"编译好的函数"
vkCmdBindDescriptorSets(cmd, ..., ds);       // 绑定参数包
vkCmdPushConstants(cmd, ..., &pc);           // 传小标量
vkCmdDispatch(cmd, groups, 1, 1);            // 记录"启动"这条命令
// 此时还没跑！要 vkQueueSubmit 才真正开始
```

没有全局状态，没有隐式同步，没有运行时编译。所有东西都是你显式创建、显式绑定、显式提交的。

### 2.2 复杂度的来源：把隐藏状态摊开

| 维度 | CUDA | OpenGL | Vulkan |
| --- | --- | --- | --- |
| 资源绑定 | 直接传设备指针 | 驱动维护全局绑定状态 | 你显式写 `VkDescriptorSet` |
| 状态管理 | 驱动隐式 | 全局状态机，驱动兜底 | 全部由你创建 / 销毁 / 缓存 |
| 命令执行 | 调用即入队 | 调用即入队 | 先**录制**，`vkQueueSubmit` 才执行 |
| 同步 | 默认流隐式同步 | 驱动隐式 | 显式 fence / semaphore / barrier |
| 错误检查 | 返回错误码 | 驱动兜底 | 大量情况是**未定义行为**，你负责 |
| 着色器 | 运行时 JIT | 运行时编译 | 离线编译成 SPIR-V，管线预创建 |

一句话：**Vulkan 不是把简单的事做复杂了，而是把驱动以前偷偷替你做的事，全部变成你必须显式写出来的对象。**

### 2.3 为什么推理引擎需要这种"啰嗦"

对 nnops 这种推理库，Vulkan 的设计其实是优点：

1. **没有运行时编译抖动**：模型第一次跑完之后，所有 pipeline 都在缓存里，后续请求延迟稳定。
2. **没有隐藏同步**：CPU 不会被驱动偷偷阻塞，可以自己决定什么时候等。
3. **可多线程**：命令录制可以并行，不会抢全局锁。
4. **可预测**：你能精确知道每一步做了什么，方便定位性能问题。

代价就是入门曲线陡。本文的目标就是把这根曲线削平。

---

## 3. 核心类比：一次 GPU 计算 = 一次"远程函数调用"

### 3.1 你熟悉的写法

假设你要实现 eltwise 加法，CPU 版本大概是这样：

```cpp
struct Params {
    int total;    // 元素个数
    int op;       // 操作类型：0=Add, 1=Sub, ...
    int add_to;   // 0=覆盖, 1=累加
};

void eltwise_kernel(const float* a, const float* b, float* c, Params p);

// 调用
eltwise_kernel(a, b, c, {1000, /*Add*/0, /*overwrite*/0});
```

这里有**六类**信息，注意它们的区别：

| 信息 | 例子 | 什么时候确定 |
| --- | --- | --- |
| 1. 函数体（代码逻辑） | `c[i] = a[i] + b[i]` | 写代码时 |
| 2. 编译期就固定的选项 | `op = Add`, `add_to = 0` | 编译时（每种组合一个版本） |
| 3. 编译后的机器码 | 编译器输出 | 编译/链接时 |
| 4. 函数签名（参数类型） | `(const float*, const float*, float*, Params)` | 编译时 |
| 5. 运行期的大块数据（指针） | `a`, `b`, `c` | 每次调用 |
| 6. 运行期的小标量 | `total` | 每次调用 |

### 3.2 Vulkan 把这六类信息拆给六个对象

Vulkan 没有"一次函数调用"这种原子操作。它要求你把上面六类信息分别交给不同的对象管理：

| C++ 里的东西 | Vulkan 对象 | 什么时候确定 | 为什么必须单独存在 |
| --- | --- | --- | --- |
| 函数体 | `.comp` 源码 → SPIR-V 二进制 | 离线编译 | GPU 不跑 C++，跑 SPIR-V |
| 编译期选项 | Specialization Constant（特化常量） | 创建 pipeline 时 | 让驱动在编译时删掉死分支 |
| 编译后的机器码 | `VkPipeline` | 创建一次，缓存复用 | 创建很贵，执行很便宜 |
| 函数签名 | `VkDescriptorSetLayout` + `VkPipelineLayout` + push constant range | 创建 pipeline 时 | 驱动建管线时要校验参数形状 |
| 实参指针 | `VkDescriptorSet` 里的 `VkBuffer` | 每次调用 | 数据地址每次不同 |
| 实参标量 | Push Constant（推送常量） | 每次调用 | 小标量走最快通道 |
| "调用"动作 | `vkCmdBind*` + `vkCmdDispatch` | 每次调用 | Vulkan 要求每类参数用对应命令绑定 |
| 一串调用 | `VkCommandBuffer` | 录制后提交 | 先录制、后执行，可复用、可多线程 |

### 3.3 并排看一遍

```cpp
// ───────────── C++ ─────────────
void eltwise_kernel(const float* a,
                    const float* b,
                    float* c,
                    int total,
                    int op);           // op 是模板参数或编译期常量

eltwise_kernel(a, b, c, 1000, 0);

// ───────────── Vulkan ─────────────
// ① 函数体：eltwise_f32.comp  →  SPIR-V  →  VkShaderModule
// ② 编译期选项：constant_id 0 = OP = 0
// ③ 机器码：VkPipeline（缓存）
// ④ 签名：VkDescriptorSetLayout(3 个 SSBO) + VkPipelineLayout(+4 字节 push constant)
// ⑤ 实参指针：VkDescriptorSet { buffer_a, buffer_b, buffer_c }
// ⑥ 实参标量：ComputePushConstants { total = 1000 }
//
// 调用：
vkCmdBindPipeline(cmd, COMPUTE, pipeline);
vkCmdBindDescriptorSets(cmd, COMPUTE, pipeline.layout, 0, 1, &desc_set, 0, nullptr);
vkCmdPushConstants(cmd, pipeline.layout, COMPUTE, 0, sizeof(pc), &pc);
vkCmdDispatch(cmd, ceil(1000 / 256), 1, 1);
```

### 3.4 为什么非要拆开

你可能会问：为什么不能像 CUDA 一样，一个函数调用把所有东西都传进去？

因为 Vulkan 要支持**提前编译**和**缓存**：

- 如果 `op` 是运行时参数，shader 里就要写 `if (OP == 0) ... else if (OP == 1) ...`，每次执行都判断；
- 如果参数形状不固定，驱动就无法在编译时生成最优的寄存器分配和指令序列；
- 如果每次调用都重新编译，就会有毫秒级抖动，推理引擎无法接受。

拆开的代价是代码变长；收益是运行时可预测、可缓存、可优化。

### 3.5 一句话记忆

> **着色器是函数体，管线是编译好的函数，布局是函数签名，描述符集是实参指针，推送常量是实参标量，命令缓冲区是调用记录。**

这句话会贯穿全文。每次看到一个 Vulkan 对象，先问自己：它对应 C++ 里的哪一部分？

---

## 4. 对象地图：Vulkan 程序里到底有哪些东西

### 4.1 先理解五个层次

假设你要跑一次 eltwise，实际用到的对象是：先 `vkCreateInstance` 连上 Vulkan（Instance）；枚举机器上的 GPU（PhysicalDevice）；选一块并 `vkCreateDevice` 打开它（Device）；从 Device 取出一个提交队列（Queue）；再建命令池和描述符池（Pool），往里放 buffer、pipeline……

把这些对象按"谁包含谁"画出来，就是下面的五层。Vulkan 的对象不是平铺的，而是分层的，从外到内：

```text
第 1 层：Instance
         "我和 Vulkan 运行时建立连接"

第 2 层：PhysicalDevice
         "这台机器上有哪些 GPU，各支持什么"

第 3 层：Device
         "我选择在这块 GPU 上打开一个会话"

第 4 层：资源与命令对象
         Buffer、Memory、Descriptor、Pipeline、CommandBuffer、Queue……

第 5 层：GPU 上的执行
         Workgroup → Subgroup → Invocation
```

**关键规则**：内层对象的销毁顺序必须和外层相反。先销毁 Device 下的所有资源，再销毁 Device，再销毁 Instance。顺序错了就是未定义行为。

### 4.2 全图

```text
┌─ 你的进程 ───────────────────────────────────────────────────────────┐
│                                                                      │
│  VkInstance              应用与 Vulkan 运行时的连接（最先创建）       │
│    │                                                                 │
│    └─ VkPhysicalDevice   一块物理 GPU（枚举出来，只用来查询能力）      │
│         │                                                            │
│         └─ VkDevice      逻辑设备：你和 GPU 之间的"会话"              │
│              │                                                       │
│              ├─ VkQueue                向 GPU 提交命令的通道          │
│              │                                                       │
│              ├─ VkCommandPool          命令缓冲区的分配器             │
│              │    └─ VkCommandBuffer   录制下来的一串命令             │
│              │                                                       │
│              ├─ VkDescriptorPool       描述符集的分配器               │
│              │    └─ VkDescriptorSet   "参数包"：里面装着 buffer 句柄  │
│              │                                                       │
│              ├─ VkDescriptorSetLayout  "参数包"的类型签名             │
│              ├─ VkPipelineLayout       完整的调用约定                 │
│              ├─ VkShaderModule         SPIR-V 的包装                  │
│              ├─ VkPipeline             编译好的 kernel                │
│              │                                                       │
│              ├─ VkBuffer               数据的"视图/句柄"              │
│              │    └─ VkDeviceMemory    真正分配出来的显存              │
│              │                                                       │
│              └─ VkFence / VkSemaphore  同步：谁等谁                   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.3 哪些是"类型"，哪些是"实例"

这是另一个容易混的点：

| 对象 | 是"类型/模板"吗 | 是"实例"吗 | 类比 |
| --- | --- | --- | --- |
| `VkDescriptorSetLayout` | ✅ 描述形状 | ❌ | `struct` 定义 |
| `VkDescriptorSet` | ❌ | ✅ 装着实参 | `struct` 变量 |
| `VkPipelineLayout` | ✅ 描述调用约定 | ❌ | 函数原型 |
| `VkPipeline` | ❌ | ✅ 编译产物 | 函数指针 |
| `VkCommandPool` | ❌ | ✅ 分配器 | 内存池 |
| `VkCommandBuffer` | ❌ | ✅ 命令序列 | 一段脚本 |

理解这一点，第 10 章讲三个 layout 时就不会晕。

### 4.4 谁创建、谁销毁

Vulkan 里最常见的 bug 不是算错，而是**生命周期搞错**。nnops 的分工是：

| 对象 | 创建者 | 生命周期 | nnops 里的位置 |
| --- | --- | --- | --- |
| `VkInstance` | 应用 / 测试环境 | 进程级 | `VulkanTestEnv` |
| `VkPhysicalDevice` | 驱动（枚举得到） | 随 Instance | `VulkanTestEnv::pick_physical_device` |
| `VkDevice` | 应用 / 测试环境 | 会话级 | `VulkanTestEnv::create_device` |
| `VkQueue` | 驱动（从 Device 取出） | 随 Device | `VulkanTestEnv::create_device` |
| `VkCommandPool` | 应用 / 测试环境 | 会话级 | `VulkanTestEnv::create_command_pool` |
| `VkCommandBuffer` | 应用 | 每次提交 | `VulkanTestEnv::begin_cmd` |
| `VkDescriptorPool` | 应用 / 测试环境 | 会话级 | `VulkanTestEnv::create_descriptor_pool` |
| `VkDescriptorSet` | 算子（每次 dispatch） | 到 pool reset | `vulkan_record_dispatch` |
| `VkBuffer` + `VkDeviceMemory` | **调用方** | 由调用方决定 | 算子不碰 |
| `VkPipeline` / Layout | **nnops 后端**（缓存） | 到 `destroy_device_pipelines` | `VulkanPipelineCache` |

> 关键边界：**nnops 的算子永远不分配显存**。它只负责"把这几个 `VkBuffer` 绑上去、录制 dispatch"。谁来创建 buffer、放多少数据、什么时候销毁，全在 `ComputeContext` 之外。

### 4.5 句柄不是指针（再强调一次）

CUDA 里你会直接把设备指针传进 kernel；Vulkan 里你传的是"句柄的句柄"：

```text
float*  →  VkBuffer  →  VkDescriptorSet  →  shader 里的 buffer
```

- `float*` 是 CPU 地址；
- `VkBuffer` 是 GPU 资源句柄；
- `VkDescriptorSet` 是"把这几个句柄打包"的容器；
- shader 里通过 `binding = N` 访问。

每多一层，都是因为 GPU 只能通过驱动的间接表访问内存，不能直接解引用 CPU 指针。

---

## 5. 第一次创建对象：搭建 Vulkan 环境

这一章按顺序创建第一批对象。每一步都说明：**为什么需要它、参数是什么意思、不这样做会怎样**。

### 5.1 创建 `VkInstance`

**它是什么**：你的进程和 Vulkan loader 之间的连接。所有全局函数和物理设备枚举都挂在它下面。

**为什么需要**：Vulkan 没有全局状态，必须有一个对象承载"我和 Vulkan 运行时已经初始化"这件事。

```cpp
VkApplicationInfo app = {};
app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
app.pApplicationName = "nnops";
app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
app.pEngineName = "nnops";
app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
app.apiVersion = VK_API_VERSION_1_3;   // 我需要的 Vulkan 核心版本

VkInstanceCreateInfo ci = {};
ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
ci.pApplicationInfo = &app;
// ci.enabledLayerCount / ppEnabledLayerNames：校验层
// ci.enabledExtensionCount / ppEnabledExtensionNames：实例扩展

VkInstance instance = VK_NULL_HANDLE;
VK_CHECK(vkCreateInstance(&ci, nullptr, &instance));
```

**参数解释**：

| 字段 | 作用 | 建议 |
| --- | --- | --- |
| `pApplicationName` | 驱动/调试工具里显示的名字 | 随便填 |
| `apiVersion` | 你希望使用的 Vulkan 版本 | 填你实际需要的，不要盲目填最高 |
| `enabledLayerCount` | 启用哪些校验层 | 开发期填 `VK_LAYER_KHRONOS_validation` |
| `enabledExtensionCount` | 启用哪些实例扩展 | compute-only 程序通常不需要 |

**不这样做会怎样**：没有 Instance，后面所有函数都无法调用。

### 5.2 枚举 `VkPhysicalDevice`

**它是什么**：一块真实的 GPU。它不是"你创建的对象"，而是系统里已经存在的硬件，你只能枚举和查询。

**为什么需要**：一台机器可能有集成 GPU、独立 GPU、软件渲染器。你要选一个。

```cpp
uint32_t count = 0;
VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, nullptr));
std::vector<VkPhysicalDevice> devices(count);
VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, devices.data()));

for (VkPhysicalDevice dev : devices) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(dev, &props);
    // props.deviceName      例如 "Adreno (TM) 840"
    // props.deviceType      DISCRETE_GPU / INTEGRATED_GPU / CPU / ...
    // props.apiVersion      该设备支持的 Vulkan 版本
    // props.limits          各种上限，比如 maxComputeWorkGroupInvocations
}
```

**`vkEnumerate*` 的两段式调用**：先传 `nullptr` 拿数量，再分配数组拿内容。这是 Vulkan 的固定模式，后面会反复见到。

选设备的常见策略：

1. 优先 `VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`（独显）；
2. 否则选第一个；
3. 软件渲染器（`CPU` 类型）通常只用于测试。

nnops 的测试环境就是这么做的。

### 5.3 查询队列族（Queue Family）

**队列族是什么**：GPU 的队列按"能力"分组。一个族里的队列都支持同一组操作。常见的能力位：

| 能力位 | 含义 |
| --- | --- |
| `VK_QUEUE_GRAPHICS_BIT` | 能画图 |
| `VK_QUEUE_COMPUTE_BIT` | 能跑 compute dispatch |
| `VK_QUEUE_TRANSFER_BIT` | 能做拷贝 |
| `VK_QUEUE_SPARSE_BINDING_BIT` | 支持稀疏内存 |

**为什么需要**：创建队列和命令池时都要指定"用哪个族"。compute-only 程序通常选第一个带 `COMPUTE` 的族。

```cpp
uint32_t family_count = 0;
vkGetPhysicalDeviceQueueFamilyProperties(phys, &family_count, nullptr);
std::vector<VkQueueFamilyProperties> families(family_count);
vkGetPhysicalDeviceQueueFamilyProperties(phys, &family_count, families.data());

uint32_t compute_family = 0;
for (uint32_t i = 0; i < family_count; ++i) {
    if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        compute_family = i;
        break;
    }
}
```

### 5.4 创建 `VkDevice`

**它是什么**：逻辑设备，你与 GPU 的"会话"。所有资源（buffer、pipeline、command pool）都挂在它下面。

**为什么需要**：物理设备是只读的，不能直接创建资源。你必须"打开"一个逻辑设备，并在打开时声明启用哪些扩展和特性。

```cpp
float priority = 1.0f;

VkDeviceQueueCreateInfo qci = {};
qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
qci.queueFamilyIndex = compute_family;   // 用哪个族
qci.queueCount = 1;                      // 要几个队列
qci.pQueuePriorities = &priority;        // 每个队列的优先级

VkDeviceCreateInfo dci = {};
dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
dci.queueCreateInfoCount = 1;
dci.pQueueCreateInfos = &qci;
// dci.enabledExtensionCount / ppEnabledExtensionNames：设备扩展
// dci.pNext：链入特性结构体

VkDevice device = VK_NULL_HANDLE;
VK_CHECK(vkCreateDevice(phys, &dci, nullptr, &device));
```

**关键点**：

- 创建 Device 时就要决定启用哪些扩展/特性，之后不能改；
- 启用了不支持的扩展会失败；
- 需要的特性没启用，使用时会失败或产生 UB。

### 5.5 取出 `VkQueue`

**队列是什么**：向 GPU 提交工作的通道。`vkQueueSubmit` 把 command buffer 送进去。

**为什么需要**：Device 创建时声明了队列族和数量，但队列对象由驱动持有，你只是"取出来"用。

```cpp
VkQueue queue = VK_NULL_HANDLE;
vkGetDeviceQueue(device, compute_family, 0, &queue);
```

注意：**队列不是你创建的**，所以也**不需要你销毁**。它随 Device 一起销毁。

### 5.6 创建 `VkCommandPool`

**命令池是什么**：命令缓冲区的分配器。

**为什么需要**：命令缓冲区不能直接 `new`，必须从池里分配。池还负责"批量回收"。

```cpp
VkCommandPoolCreateInfo ci = {};
ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
ci.queueFamilyIndex = compute_family;   // 命令池与队列族绑定
ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

VkCommandPool cmd_pool = VK_NULL_HANDLE;
VK_CHECK(vkCreateCommandPool(device, &ci, nullptr, &cmd_pool));
```

**`queueFamilyIndex` 为什么必须是队列族**：命令缓冲区最终要提交到某个队列，池必须和队列的族一致，否则提交会失败。

### 5.7 创建 `VkDescriptorPool`

**描述符池是什么**：描述符集的分配器。

**为什么需要**：描述符集也不能直接 `new`，必须从池里分配。池的容量在创建时就要预估。

```cpp
VkDescriptorPoolSize sizes[1] = {};
sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
sizes[0].descriptorCount = 256;   // 最多 256 个 storage buffer 描述符

VkDescriptorPoolCreateInfo ci = {};
ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
ci.maxSets = 256;                  // 最多 256 个描述符集
ci.poolSizeCount = 1;
ci.pPoolSizes = sizes;

VkDescriptorPool desc_pool = VK_NULL_HANDLE;
VK_CHECK(vkCreateDescriptorPool(device, &ci, nullptr, &desc_pool));
```

**`maxSets` 和 `descriptorCount` 的区别**：

- `maxSets`：能分配多少个"参数包"；
- `descriptorCount`：所有参数包里最多有多少个"槽位"；
- 一个 3-binding 的 set 会消耗 3 个 storage buffer 槽位。

如果池不够大，`vkAllocateDescriptorSets` 会失败（`VK_ERROR_OUT_OF_POOL_MEMORY`）。

### 5.8 完整初始化代码

把上面几步串起来：

```cpp
struct VulkanEnv {
    VkInstance       instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys     = VK_NULL_HANDLE;
    VkDevice         device   = VK_NULL_HANDLE;
    VkQueue          queue    = VK_NULL_HANDLE;
    VkCommandPool    cmd_pool = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    uint32_t         compute_family = 0;

    void init() {
        // ① Instance
        VkApplicationInfo app = {};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ici = {};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &app;
        VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));

        // ② PhysicalDevice
        uint32_t n = 0;
        vkEnumeratePhysicalDevices(instance, &n, nullptr);
        std::vector<VkPhysicalDevice> devs(n);
        vkEnumeratePhysicalDevices(instance, &n, devs.data());
        phys = devs[0];

        // ③ Queue family
        uint32_t fc = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &fc, nullptr);
        std::vector<VkQueueFamilyProperties> fams(fc);
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &fc, fams.data());
        for (uint32_t i = 0; i < fc; ++i) {
            if (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                compute_family = i;
                break;
            }
        }

        // ④ Device
        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci = {};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = compute_family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci = {};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        VK_CHECK(vkCreateDevice(phys, &dci, nullptr, &device));

        // ⑤ Queue
        vkGetDeviceQueue(device, compute_family, 0, &queue);

        // ⑥ CommandPool
        VkCommandPoolCreateInfo pci = {};
        pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.queueFamilyIndex = compute_family;
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK(vkCreateCommandPool(device, &pci, nullptr, &cmd_pool));

        // ⑦ DescriptorPool
        VkDescriptorPoolSize ps = {};
        ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = 256;
        VkDescriptorPoolCreateInfo dpci = {};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 256;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &ps;
        VK_CHECK(vkCreateDescriptorPool(device, &dpci, nullptr, &desc_pool));
    }

    void shutdown() {
        // 顺序：先等 GPU 空闲，再按创建的反序销毁
        vkDeviceWaitIdle(device);
        vkDestroyDescriptorPool(device, desc_pool, nullptr);
        vkDestroyCommandPool(device, cmd_pool, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
    }
};
```

**销毁顺序为什么重要**：Device 还活着的时候销毁它下面的资源；Device 销毁后再销毁 Instance。反过来会访问已释放的对象。

---

## 6. Buffer 与 DeviceMemory：数据放在哪

### 6.1 为什么 `VkBuffer` 和 `VkDeviceMemory` 是两步

CUDA 里：

```cpp
float* p;
cudaMalloc(&p, size);   // 一步：分配显存并返回指针
```

Vulkan 里拆成四步：

```cpp
VkBufferCreateInfo bci = {};
bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
bci.size = size_bytes;                                  // 多大
bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;         // 用来干什么
bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;            // 是否跨队列共享
VkBuffer buffer = VK_NULL_HANDLE;
VK_CHECK(vkCreateBuffer(device, &bci, nullptr, &buffer));   // ① 创建"描述"

VkMemoryRequirements req;
vkGetBufferMemoryRequirements(device, buffer, &req);         // ② 查询内存需求

VkMemoryAllocateInfo mai = {};
mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
mai.allocationSize = req.size;                              // ③ 分配真实内存
mai.memoryTypeIndex = find_memory_type(phys, req.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
VkDeviceMemory memory = VK_NULL_HANDLE;
VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &memory));

VK_CHECK(vkBindBufferMemory(device, buffer, memory, 0));    // ④ 绑定
```

**为什么拆**：

1. **子分配**：一个大的 `VkDeviceMemory` 可以绑定多个 `VkBuffer`，每个 offset 不同。这是 Vulkan 内存池的基础。
2. **对齐**：不同 GPU 对 buffer 的开始地址有对齐要求，先查询再分配才不会浪费。
3. **内存类型选择**：同一块 buffer 可以放在显存、系统内存、或 CPU 可见的共享内存里，由你决定。

**类比**：

```cpp
// VkBuffer 像 std::vector 的"容量声明"：告诉系统我要多大、干什么用
// VkDeviceMemory 像 malloc 返回的真实内存块
// vkBindBufferMemory 像把 vector 的数据指针指向这块内存
```

### 6.2 内存类型：`HOST_VISIBLE`、`COHERENT`、`DEVICE_LOCAL`

GPU 的内存不是"一块"，而是分类型的。每种类型是一组属性的组合：

| 属性 | 含义 | 典型用途 |
| --- | --- | --- |
| `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` | 位于显存，GPU 访问最快 | 大块权重、中间结果 |
| `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` | CPU 能 `vkMapMemory` 访问 | 上传输入、读回输出 |
| `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT` | CPU 写完不需要 flush | 简单测试、小数据 |
| `VK_MEMORY_PROPERTY_HOST_CACHED_BIT` | CPU 读取有缓存，快 | 频繁读回的结果 |
| `VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT` | 延迟分配，用于 tile 内存 | 移动端高级用法 |

**关键事实**：不是所有组合都存在。你要枚举 `VkPhysicalDeviceMemoryProperties`，找到满足要求的类型。

```cpp
uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t type_bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(phys, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    throw std::runtime_error("no suitable memory type");
}
```

**两种典型策略**：

| 场景 | 推荐内存类型 | 原因 |
| --- | --- | --- |
| 测试 / 小数据 | `HOST_VISIBLE \| HOST_COHERENT` | 实现简单，`memcpy` 即可 |
| 生产 / 大数据 | `DEVICE_LOCAL` + staging buffer | GPU 访问快，但需要额外拷贝 |

nnops 的测试环境选第一种：buffer 直接是可映射的系统内存，GPU 通过 PCIe 访问。简单，但带宽不如显存。

### 6.3 映射与读写

创建好 buffer + memory 之后，CPU 怎么读写？

**写入（CPU → GPU）**：

```cpp
void* mapped = nullptr;
VK_CHECK(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped));
memcpy(mapped, host_data, size_bytes);

// 如果内存不是 HOST_COHERENT，需要 flush
VkMappedMemoryRange range = {};
range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
range.memory = memory;
range.offset = 0;
range.size = VK_WHOLE_SIZE;
vkFlushMappedMemoryRanges(device, 1, &range);

vkUnmapMemory(device, memory);
```

**读取（GPU → CPU）**：

```cpp
VK_CHECK(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped));
// 如果内存不是 HOST_COHERENT，需要 invalidate
vkInvalidateMappedMemoryRanges(device, 1, &range);
memcpy(host_out, mapped, size_bytes);
vkUnmapMemory(device, memory);
```

**`HOST_COHERENT` 的意义**：

- 有它：CPU 写完 GPU 立刻可见，不需要 flush/invalidate；
- 没有它：必须手动 flush（写后）和 invalidate（读前）；
- 测试环境用有它的内存，所以代码里可以省掉这些步骤（但 nnops 仍然防御性地调用了）。

**常见错误**：

1. 映射了 `DEVICE_LOCAL` 的内存（通常不允许）；
2. 忘记 flush，GPU 读到旧数据；
3. 在 GPU 还在使用这块内存时 map/unmap；
4. map 的范围和 buffer 大小不一致。

### 6.4 一个完整的 buffer 创建 + 上传例子

```cpp
struct GpuBuffer {
    VkBuffer       buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void*          mapped = nullptr;
    size_t         size   = 0;
};

GpuBuffer create_and_upload(VkDevice device, VkPhysicalDevice phys,
                            const void* data, size_t bytes) {
    GpuBuffer out;
    out.size = bytes;

    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(device, &bci, nullptr, &out.buffer));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, out.buffer, &req);

    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory_type(
        phys, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &out.memory));
    VK_CHECK(vkBindBufferMemory(device, out.buffer, out.memory, 0));

    VK_CHECK(vkMapMemory(device, out.memory, 0, VK_WHOLE_SIZE, 0, &out.mapped));
    memcpy(out.mapped, data, bytes);
    return out;
}
```

这个结构就是 nnops 测试环境里 `VulkanTestEnv::Buffer` 的简化版。区别是测试环境会持久映射，并在读写时做 flush/invalidate。

---

## 7. CommandBuffer：先录制，后执行

### 7.1 为什么"录制"和"执行"要分开

如果每条 `vkCmd*` 都立即执行，CPU 和 GPU 就完全串行了：

```text
CPU: 发命令1 → 等GPU执行完 → 发命令2 → 等GPU执行完 → ...
```

Vulkan 的做法是：

```text
CPU: 把命令1、2、3 都录进 command buffer → vkQueueSubmit → 继续干别的
GPU: 按顺序执行 command buffer 里的命令
```

这带来三个能力：

1. **异步**：CPU 提交后可以立刻准备下一批数据；
2. **多线程录制**：每个线程录自己的 command buffer，互不干扰；
3. **复用**：同一段命令可以重复提交（只要不修改资源）。

### 7.2 `VkCommandPool` 和 `VkCommandBuffer` 的关系

```text
VkCommandPool
  ├─ VkCommandBuffer A
  ├─ VkCommandBuffer B
  └─ VkCommandBuffer C
```

- Pool 是分配器，与队列族绑定；
- Buffer 从 Pool 分配，可以一次分配多个；
- Pool 可以整体 reset，一次性回收所有 buffer；
- 如果创建 Pool 时带了 `RESET_COMMAND_BUFFER_BIT`，也可以单独 reset 某个 buffer。

**线程规则**：同一个 Pool 不能被多个线程同时分配/重置/录制。推荐每个线程一个 Pool。

### 7.3 CommandBuffer 的状态机

```text
Initial
  │ vkBeginCommandBuffer
  ▼
Recording ──── vkCmd* ────→ 只能录制，不能提交
  │ vkEndCommandBuffer
  ▼
Executable ─── vkQueueSubmit ───→ 可提交，不能录制
  │
  ▼
Pending → 执行中 → 执行完
  │
  ▼
Invalid / 可以重新 begin（取决于 pool 的 reset 标志）
```

对 nnops 算子的意义：**算子被调用时，command buffer 已经处于 Recording 状态**。所以算子只调用 `vkCmd*`，绝不调用 `vkBeginCommandBuffer` / `vkEndCommandBuffer`——那是调用方的职责。

### 7.4 常用命令

| 命令 | 作用 |
| --- | --- |
| `vkCmdBindPipeline` | 绑定"编译好的函数" |
| `vkCmdBindDescriptorSets` | 绑定"参数包" |
| `vkCmdPushConstants` | 传小标量 |
| `vkCmdDispatch` | 启动 compute |
| `vkCmdCopyBuffer` | buffer 之间拷贝 |
| `vkCmdPipelineBarrier` | 插入执行/内存依赖 |
| `vkCmdFillBuffer` | 用固定值填充 buffer |

所有 `vkCmd*` 都只录制，不执行。

### 7.5 提交：`vkQueueSubmit`

**队列提交是什么**：把 command buffer 交给 GPU 队列。

```cpp
VkSubmitInfo si = {};
si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
si.commandBufferCount = 1;
si.pCommandBuffers = &cmd;

VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
```

第三个参数 `VK_NULL_HANDLE` 是 fence。如果传一个 fence，GPU 执行完后会 signal 它，CPU 可以等待。

**fence 是什么**：一个同步对象，有"未触发/已触发"两种状态。CPU 用 `vkWaitForFences` 等它触发。

```cpp
VkFenceCreateInfo fci = {};
fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
VkFence fence = VK_NULL_HANDLE;
vkCreateFence(device, &fci, nullptr, &fence);

vkQueueSubmit(queue, 1, &si, fence);
vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);  // 阻塞直到完成
vkResetFences(device, 1, &fence);                         // 复位，下次再用
```

**`vkQueueWaitIdle` 是什么**：等整个队列空闲的粗暴方法，等价于"等这个队列上所有提交都完成"。

```cpp
vkQueueWaitIdle(queue);
```

测试环境用它，因为简单。生产环境应该用 fence 或 timeline semaphore，避免阻塞整个队列。

### 7.6 一个完整的最小提交例子

```cpp
// 1. 分配 command buffer
VkCommandBufferAllocateInfo ai = {};
ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
ai.commandPool = cmd_pool;
ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
ai.commandBufferCount = 1;
VkCommandBuffer cmd = VK_NULL_HANDLE;
VK_CHECK(vkAllocateCommandBuffers(device, &ai, &cmd));

// 2. begin
VkCommandBufferBeginInfo bi = {};
bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

// 3. 录制（这里由算子完成）
vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                        layout, 0, 1, &desc_set, 0, nullptr);
vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                   0, sizeof(pc), &pc);
vkCmdDispatch(cmd, groups, 1, 1);

// 4. end
VK_CHECK(vkEndCommandBuffer(cmd));

// 5. 提交并等待
VkSubmitInfo si = {};
si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
si.commandBufferCount = 1;
si.pCommandBuffers = &cmd;
VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
VK_CHECK(vkQueueWaitIdle(queue));

// 6. 回收
vkFreeCommandBuffers(device, cmd_pool, 1, &cmd);
```

---

## 8. Shader、GLSL、SPIR-V、ShaderModule

### 8.1 着色器是什么

着色器就是"在 GPU 上执行的函数"。对 compute 来说，每个 shader 线程执行一次 `main()`，处理一个数据元素。

```glsl
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer BufA { float a[]; };
layout(set = 0, binding = 2) buffer BufC { float c[]; };

void main() {
    uint gid = gl_GlobalInvocationID.x;
    c[gid] = a[gid] * 2.0;
}
```

### 8.2 GLSL 是什么

GLSL（OpenGL Shading Language）是一种类 C 语言，是 Vulkan 最常用的 shader 前端。它的语法你大部分能看懂：

- 变量、数组、`if/else`、`for`；
- 向量类型 `vec2/vec3/vec4`，矩阵 `mat4`；
- 大量内置数学函数：`exp`、`sin`、`sqrt`、`dot`、`mix`；
- 用 `layout(...)` 声明资源绑定、工作组大小、常量。

Vulkan 也支持其他前端（HLSL 通过 DXC 编译到 SPIR-V），但 nnops 只用 GLSL。

### 8.3 SPIR-V 是什么

SPIR-V 是一种二进制中间表示。它不是 GLSL，也不是 GPU 机器码，而是"驱动可以理解的标准字节码"。

```text
eltwise_f32.comp   ──glslc──→   eltwise_f32.spv   ──vkCreateComputePipelines──→   VkPipeline
   GLSL 源码                      SPIR-V 字节码                驱动编译成 GPU 机器码
   （可读）                        （二进制）                   （最终可执行）
```

**为什么需要 SPIR-V**：

- 不同 GPU 的机器码不同，不能直接分发源码；
- 驱动需要在创建管线时做最终编译和优化；
- SPIR-V 比 GLSL 更接近机器，编译更快，也更容易校验。

### 8.4 编译链

```bash
# 1. GLSL → SPIR-V
glslc --target-env=vulkan1.2 -o eltwise_f32.spv eltwise_f32.comp

# 2. SPIR-V → C++ 头文件（把字节数组嵌进程序）
python spv_to_header.py --name g_eltwise_f32 --spv eltwise_f32.spv --out eltwise_f32_spv.h
```

nnops 的构建系统会自动做这两步，生成 `constexpr uint32_t g_eltwise_f32_spv[]`。

**`--target-env=vulkan1.2` 是什么**：告诉 `glslc` 按 Vulkan 1.2 的规则生成 SPIR-V（影响可用的能力，比如 16-bit storage）。

### 8.5 `VkShaderModule`

**它是什么**：SPIR-V 字节的 Vulkan 包装。

**为什么需要**：`vkCreateComputePipelines` 不接受裸字节，必须传一个 `VkShaderModule`。

```cpp
VkShaderModuleCreateInfo ci = {};
ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
ci.codeSize = spirv_size;          // 字节数，必须是 4 的倍数
ci.pCode = spirv_data;             // const uint32_t*

VkShaderModule module = VK_NULL_HANDLE;
VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &module));
```

**重要**：`VkShaderModule` 只在创建管线时有用。管线创建完成后，驱动已经复制了需要的信息，可以立刻销毁：

```cpp
vkCreateComputePipelines(...);
vkDestroyShaderModule(device, module, nullptr);   // 安全
```

### 8.6 入口点

`VkPipelineShaderStageCreateInfo::pName` 指定 shader 的入口函数名。GLSL 里固定是 `"main"`。

```cpp
stage_ci.pName = "main";
```

如果写错，管线创建会失败（`VK_ERROR_INVALID_SHADER_NV` 或类似错误）。

---

## 9. Descriptor：GPU 怎么拿到数据

### 9.1 为什么 GPU 不能直接使用 CPU 指针

CPU 的指针是虚拟地址，只在当前进程的页表里有效。GPU 有独立的地址空间，而且不同厂商、不同驱动的映射方式不同。所以你不能把 `float*` 直接传给 shader。

Vulkan 的解决方案是**描述符**：一个由驱动管理的间接引用。

```text
CPU 侧：float* → VkBuffer → VkDescriptorSet → shader 里的 binding
```

- `VkBuffer`：告诉驱动"这块资源在哪、多大"；
- `VkDescriptorSet`：把若干 `VkBuffer` 打包；
- shader 通过 `binding = N` 访问对应的 buffer。

### 9.2 描述符、binding、set 三者的编号关系

这是最容易晕的地方。先把三个层次画出来：

```text
set = 0                          ← 第 0 个"参数包"
  ├─ binding = 0  → VkBuffer A   ← 包里的第 0 个槽
  ├─ binding = 1  → VkBuffer B   ← 第 1 个槽
  └─ binding = 2  → VkBuffer C   ← 第 2 个槽

set = 1                          ← 第 1 个"参数包"（nnops 不用）
  └─ binding = 0  → ...
```

对应关系：

| GLSL | C++ | 绑定命令 |
| --- | --- | --- |
| `layout(set = 0, binding = 0)` | `bindings[0].binding = 0` | `vkCmdBindDescriptorSets(..., firstSet=0, ...)` |
| `layout(set = 0, binding = 1)` | `bindings[1].binding = 1` | 同上 |
| `layout(set = 1, binding = 0)` | 另一个 set layout | `vkCmdBindDescriptorSets(..., firstSet=1, ...)` |

**必须一致**：shader 里写了几行 `layout(set=0, binding=N)`，C++ 里就必须创建几个 `VkDescriptorSetLayoutBinding`，且编号对应。少一个或错一位，shader 就会读到别的 buffer，或者直接 UB。

nnops 的约定：**全部使用 `set = 0`，binding 从 0 连续编号，输入在前、输出在后**。

### 9.3 `VkDescriptorSetLayout` 是什么

**它是什么**：描述"一个参数包长什么样"。它不包含任何真实数据，只描述槽位。

```cpp
VkDescriptorSetLayoutBinding bindings[3] = {};
for (uint32_t i = 0; i < 3; ++i) {
    bindings[i].binding         = i;                                  // 槽位编号
    bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;  // 槽位类型
    bindings[i].descriptorCount = 1;                                  // 这个槽里有几个资源
    bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;        // 哪个阶段能访问
}

VkDescriptorSetLayoutCreateInfo ci = {};
ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
ci.bindingCount = 3;
ci.pBindings = bindings;

VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
VK_CHECK(vkCreateDescriptorSetLayout(device, &ci, nullptr, &set_layout));
```

**每个字段的含义**：

| 字段 | 含义 | nnops 取值 |
| --- | --- | --- |
| `binding` | 槽位编号，对应 shader 里的 `binding = N` | 0, 1, 2 |
| `descriptorType` | 槽里放什么类型的资源 | `STORAGE_BUFFER` |
| `descriptorCount` | 槽里放几个资源（数组） | 1 |
| `stageFlags` | 哪个 shader 阶段可见 | `COMPUTE_BIT` |
| `pImmutableSamplers` | 不可变采样器（图像用） | `nullptr` |

**类比**：

```cpp
// 这就像声明了一个函数类型
using EltwiseFn = void(const float*, const float*, float*);
// 但还没有任何实参
```

### 9.4 `VkDescriptorPool` 是什么

**它是什么**：描述符集的分配器。

**为什么需要**：描述符集不能直接 `new`，必须从池里分配。池在创建时就要声明容量。

```cpp
VkDescriptorPoolSize size = {};
size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
size.descriptorCount = 256;

VkDescriptorPoolCreateInfo ci = {};
ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
ci.maxSets = 256;
ci.poolSizeCount = 1;
ci.pPoolSizes = &size;

VkDescriptorPool pool = VK_NULL_HANDLE;
VK_CHECK(vkCreateDescriptorPool(device, &ci, nullptr, &pool));
```

**回收策略**：

| 方式 | 做法 | 适用 |
| --- | --- | --- |
| 整池 reset | `vkResetDescriptorPool(pool, 0)` | 批量回收，nnops 测试环境 |
| 单独 free | 池创建时带 `FREE_DESCRIPTOR_SET_BIT`，然后 `vkFreeDescriptorSets` | 精细控制 |
| 不做回收 | 每次用新池 | 简单但浪费 |

**注意**：reset 之后，池里所有描述符集都失效，不能再使用。

### 9.5 `VkDescriptorSet` 是什么

**它是什么**：一个具体的"参数包实例"，里面装着真实的资源句柄。

**怎么得到**：

```cpp
VkDescriptorSetAllocateInfo ai = {};
ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
ai.descriptorPool = pool;
ai.descriptorSetCount = 1;
ai.pSetLayouts = &set_layout;   // 按这个布局分配

VkDescriptorSet set = VK_NULL_HANDLE;
VK_CHECK(vkAllocateDescriptorSets(device, &ai, &set));
```

**怎么填**：

```cpp
VkDescriptorBufferInfo info_a = {};
info_a.buffer = buffer_a;
info_a.offset = 0;
info_a.range  = VK_WHOLE_SIZE;

VkWriteDescriptorSet write = {};
write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
write.dstSet = set;
write.dstBinding = 0;                              // 写到 binding 0
write.dstArrayElement = 0;
write.descriptorCount = 1;
write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
write.pBufferInfo = &info_a;

vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
```

**`VkDescriptorBufferInfo` 的三个字段**：

| 字段 | 含义 |
| --- | --- |
| `buffer` | 哪个 `VkBuffer` |
| `offset` | 从 buffer 的第几个字节开始 |
| `range` | 用多少字节；`VK_WHOLE_SIZE` 表示全部 |

### 9.6 为什么每次 dispatch 都要重新分配

描述符集里装的是"当前这批 buffer 的句柄"。下一批数据的 `VkBuffer` 可能不同，所以内容需要更新。

三种做法：

| 方式 | 做法 | 优缺点 |
| --- | --- | --- |
| 重新分配 | `vkAllocateDescriptorSets` → `vkUpdateDescriptorSets` | 简单；每次消耗一个 set |
| 复用 + 更新 | 保留 set，只 `vkUpdateDescriptorSets` | 省分配；但要保证 GPU 不在使用 |
| Push Descriptor | `vkCmdPushDescriptorSet` | 无需池；需要 `VK_KHR_push_descriptor` |

nnops 选第一种：每次分配一个，pool reset 时批量回收。测试环境 `maxSets=256` 足够。

**重要限制**：`vkUpdateDescriptorSets` 之后，**不能**在命令执行期间修改同一个 set。否则是 UB。这就是为什么每个 dispatch 用新 set。

### 9.7 descriptor 的类型：SSBO 和 UBO

| 类型 | Vulkan 枚举 | 特点 | nnops 使用 |
| --- | --- | --- | --- |
| SSBO | `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER` | 可读写，大小几乎无上限，支持运行时长度数组 | ✅ 全部 |
| UBO | `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER` | 只读，保证 64KB，走常量缓存 | ❌ |
| 存储图像 | `VK_DESCRIPTOR_TYPE_STORAGE_IMAGE` | 可读写图像 | ❌ |
| 采样图像 | `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER` | 纹理采样 | ❌ |

nnops 全用 SSBO，因为：

- 没有 64KB 上限；
- 支持 `float a[]` 这种运行时长度数组；
- 可写；
- 和 `float*` 数组的映射最直接。

### 9.8 生命周期总览

```text
创建 Device
  │
  ├─ 创建 DescriptorPool
  │
  ├─ 创建 DescriptorSetLayout（每个 pipeline 一套）
  │
  └─ 每次 dispatch:
       ├─ vkAllocateDescriptorSets(pool, set_layout) → set
       ├─ vkUpdateDescriptorSets(set, buffers)
       ├─ vkCmdBindDescriptorSets(set)
       └─ 提交后: vkResetDescriptorPool(pool) 回收全部 set
```

---

## 10. 三个 "layout" 千万别搞混

先用一个具体例子区分。同一个 eltwise 里：

- `layout(std430, set = 0, binding = 0)` 里的 `std430` 是**内存布局**；
- 三个 `layout(set=0, binding=N)` 合起来描述的"参数包"是**描述符集布局**；
- 把参数包和 `int total` 打包成"这个管线接受什么"，是**管线布局**。

三者都叫 layout，但回答的问题完全不同：

| 中文 | Vulkan 类型 | 一句话解释 | 类比 |
| --- | --- | --- | --- |
| **内存布局** | 无类型，写在 shader 里（`std430` / `std140`） | 一个 buffer 内部的字节怎么排 | `struct` 的 `sizeof` / 对齐 |
| **描述符集布局** | `VkDescriptorSetLayout` | 一个"参数包"里有哪些槽位、每个槽位是什么类型 | 函数参数表的**类型签名** |
| **管线布局** | `VkPipelineLayout` | 一个管线接受哪些描述符集 + 哪些推送常量 | 完整的**调用约定 / ABI** |

### 10.1 内存布局（memory layout）

`std430` / `std140` 描述的是**数据在 buffer 里怎么排**。它不是 Vulkan 对象，不需要创建，只影响 shader 怎么读。

```glsl
// SSBO 默认 std430：数组元素紧密排列
layout(std430, set = 0, binding = 0) buffer BufA { float a[]; };
```

**为什么会有这个规则**：GPU 的 load/store 指令要求数据按特定边界对齐，编译器要按 `std430`/`std140` 生成正确的偏移。

**`std140` vs `std430` 的关键区别**：

| 类型 | `std140`（UBO 默认） | `std430`（SSBO 默认） |
| --- | --- | --- |
| `float` | 4 字节，4 对齐 | 4 字节，4 对齐 |
| `vec2` | 8 字节，8 对齐 | 8 字节，8 对齐 |
| `vec3` | 12 字节，**16 对齐** | 12 字节，**16 对齐** |
| `vec4` | 16 字节，16 对齐 | 16 字节，16 对齐 |
| 数组元素 | 每个元素按 16 字节对齐（有填充） | 紧密排列，无尾填充 |

因为 nnops 所有 buffer 都是同构 `float` / `half` 数组，所以内存布局这件事在项目里几乎不用操心。**先记住它存在即可。**

### 10.2 描述符集布局（`VkDescriptorSetLayout`）

它定义"一个参数包长什么样"：

```cpp
// nnops 为 eltwise 创建的 set layout 等价于：
//   binding 0: STORAGE_BUFFER (只读 A)
//   binding 1: STORAGE_BUFFER (只读 B)
//   binding 2: STORAGE_BUFFER (可写 C)
```

它**不包含任何真实数据**，只是"这里有三个 storage buffer 槽位"的声明。

### 10.3 管线布局（`VkPipelineLayout`）

管线布局把"描述符集布局"和"推送常量范围"组合起来，回答"这个管线接受什么"：

```cpp
VkPushConstantRange pc_range = {};
pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
pc_range.offset     = 0;
pc_range.size       = sizeof(ComputePushConstants);  // 4 字节

VkPipelineLayoutCreateInfo pl_ci = {};
pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
pl_ci.setLayoutCount         = 1;
pl_ci.pSetLayouts            = &set_layout;          // 1 个描述符集
pl_ci.pushConstantRangeCount = 1;
pl_ci.pPushConstantRanges    = &pc_range;            // 1 段 4 字节推送常量
VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &layout));
```

**为什么需要两层**：一个管线理论上可以接受多个描述符集（`set = 0`、`set = 1`……），每个集有自己的布局。所以必须有"集布局"和"管线布局"两层。

### 10.4 三者关系图

```text
        shader 源码                      C++ 侧创建的对象
┌─────────────────────────┐
│ layout(set=0,binding=0) │──┐
│ layout(set=0,binding=1) │──┼──→ VkDescriptorSetLayout ──┐
│ layout(set=0,binding=2) │──┘    ("参数包的类型签名")      │
└─────────────────────────┘                                │
                                                           ├──→ VkPipelineLayout
┌─────────────────────────┐                                │      ("完整调用约定")
│ layout(push_constant)   │──────→ push constant range ────┘
│   uniform { int total; }│
└─────────────────────────┘

                                             │ 建管线时传入
                                             ▼
                                    VkComputePipelineCreateInfo
                                             │
                                             ▼
                                        VkPipeline
                                    ("编译好的函数")
```

### 10.5 兼容性规则（一句话）

> **用某个 `VkDescriptorSetLayout` 分配出来的描述符集，只能绑定到 `VkPipelineLayout` 里引用了"兼容布局"的管线。**

这就是为什么 nnops 把 `set_layout` 和 `layout` 都存进 `VulkanPipeline` 结构体——它们必须成对使用，且和 pipeline 一起缓存：

```cpp
struct VulkanPipeline {
    VkPipeline            pipeline   = VK_NULL_HANDLE;  // 编译好的函数
    VkPipelineLayout      layout     = VK_NULL_HANDLE;  // 调用约定
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;  // 参数包签名
};
```

### 10.6 绑定时的完整对应

```cpp
// 建管线时：layout 已经写进 VkPipeline
VkComputePipelineCreateInfo pipeline_ci = {};
pipeline_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
pipeline_ci.stage  = stage_ci;
pipeline_ci.layout = layout;
vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_ci, nullptr, &pipeline);

// 录制时：
vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);      // ① 选函数
vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                        layout,          // ② 必须和建管线时同一个 layout
                        0, 1, &desc_set, //    第 0 个参数包
                        0, nullptr);     //    无动态偏移
vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT,           // ③ 同一个 layout
                   0, sizeof(pc), &pc);
```

**硬规则**：`vkCmdBindDescriptorSets` 用的 `layout` 必须和创建 `pipeline` 时用的 `layout` 是同一条（或兼容）。`desc_set` 又必须按 `set_layout` 分配。三者版本错配 = 未定义行为。

---

## 11. Pipeline：编译好的 kernel

### 11.1 Pipeline 是什么

**它是什么**：一个"编译好的、状态固定的可执行单元"。

在 C++ 里，函数是"代码 + 入口"。在 Vulkan 里，pipeline 是"代码 + 布局 + 固定状态 + 特化常量（创建管线时固化的编译期常量，第 12 章详述）"的打包结果。

```cpp
// C++ 里：
void kernel(const float* a, const float* b, float* c, int total);
//            ~~~~~~~~~~ 参数类型 ~~~~~~~~~~  ~~~~~ 小标量

// Vulkan 里：
VkPipeline  =  已编译的机器码（op 已固化）
VkPipelineLayout = 参数类型 + 小标量范围
```

### 11.2 为什么必须预创建

CUDA / OpenGL 可以在运行时编译着色器，Vulkan 反过来：**着色器离线编译成 SPIR-V，管线在运行时创建一次，之后永远复用。**

原因有三：

1. **可预测性**：推理引擎不能接受某个请求突然触发几毫秒的着色器编译抖动。
2. **可优化**：驱动在创建管线时看到**全部**固定状态（特化常量、布局、workgroup 大小），可以生成高度特化的机器码，甚至把 `OP == 0` 的分支整段删掉。
3. **可缓存**：`VkPipeline` 建好之后 `vkCmdBindPipeline` 只是写一个句柄，几乎免费。

代价是 `vkCreateComputePipelines` 很贵（毫秒级），所以**必须缓存**。

### 11.3 一个 compute pipeline 由什么组成

```text
VkComputePipelineCreateInfo
├── VkPipelineShaderStageCreateInfo
│     ├── stage         = VK_SHADER_STAGE_COMPUTE_BIT
│     ├── module        = VkShaderModule（SPIR-V）
│     ├── pName         = "main"
│     └── pSpecializationInfo
│           ├── constant_id 0 = OP      (例如 Add = 0)
│           └── constant_id 1 = ADD_TO  (例如 0)
└── layout              = VkPipelineLayout
```

没有别的了。compute pipeline 不像 graphics pipeline 那样有几十个阶段（顶点、光栅化、混合……），它只有一个计算阶段。所以 Vulkan 的计算管线其实**很简单**，复杂的是外围的布局和绑定。

### 11.4 创建流程逐行

nnops 的 `VulkanPipelineCache::create_pipeline()`（[vulkan_common.cpp](../src/backend/vulkan/vulkan_common.cpp)）：

```text
① vkCreateShaderModule(SPIR-V 字节)          // 把 SPIR-V 包成 Vulkan 对象
② vkCreateDescriptorSetLayout(num_bindings)  // 声明"参数包"有几个槽
③ vkCreatePipelineLayout(set_layout, pc)     // 声明"调用约定"
④ vkCreateComputePipelines(layout, spec)     // 真正编译
⑤ vkDestroyShaderModule()                    // SPIR-V 用完即可销毁
```

**为什么必须是这个顺序**：建管线时驱动要同时看到"代码"和"参数形状"，才能生成正确的机器码并做校验。所以 ① 提供代码，②③ 提供形状，④ 才能编译。

### 11.5 管线里到底固化了什么

`vkCreateComputePipelines` 之后，下面这些东西**再也不能改**：

| 固化项 | 在 nnops 里的来源 | 改了的后果 |
| --- | --- | --- |
| 着色器代码 | `spirv_data` / `spirv_size` | 必须新建管线 |
| 特化常量 | `spec_op`, `spec_add_to` | 必须新建管线 |
| 参数形状 | `num_bindings` → `set_layout` | 必须新建管线 |
| 推送常量范围 | `sizeof(ComputePushConstants)` | 必须新建管线 |
| workgroup 大小 | shader 里的 `local_size_x` | 必须改 shader 并重建 |

这解释了两件常被问到的事：

1. **为什么 `local_size_x` 不能像 CUDA 那样运行时指定？** 因为它编译进了 SPIR-V，是管线的一部分。CUDA 的 `<<<grid, block>>>` 在 Vulkan 里对应的是 `vkCmdDispatch` 的 grid 部分；block 部分已经固化。
2. **为什么换个 op 就要新管线？** `OP` 是特化常量，驱动在编译时把其他分支整段删掉了。同一份 `.comp` 源码编译出的两条管线，机器码是不同的。

### 11.6 两种 "pipeline cache" 不要混

| 名字 | 是什么 | nnops 里的对应 |
| --- | --- | --- |
| `VkPipelineCache` | Vulkan 的驱动级缓存对象，可以把编译结果序列化到磁盘 | **没有使用** |
| `VulkanPipelineCache` | nnops 自己写的 C++ 单例，内部是 `unordered_map<PipelineKey, VulkanPipeline>` | `vulkan_common.hpp` |

`VkPipelineCache` 是"驱动帮我记住编译结果"；`VulkanPipelineCache` 是"我自己记住已经创建过的 VkPipeline 句柄"。两者名字像，作用不同。

### 11.7 为什么 nnops 用 `PipelineKey` 做缓存键

一条 pipeline 对应的是一组**完全固定**的编译参数。任何一项变了，就是另一条管线：

```cpp
struct PipelineKey {
    VkDevice        device;         // 不同 GPU 的机器码不同
    const uint32_t* spirv_data;     // 不同 shader（f32/f16、eltwise/unary）
    size_t          spirv_size;
    uint32_t        spec_op;        // Add vs Sub vs Mul ... 是不同管线
    uint32_t        spec_add_to;    // 覆盖 vs 累加 是不同管线
    uint32_t        num_bindings;   // 2 个 buffer vs 3 个 buffer
};
```

`spec_op` 是 key 的一部分，这正是"特化常量"的含义：**它不是运行时参数，而是编译期常量**。所以你每多支持一种 op，就多一条管线——但运行时分文不花。

### 11.8 创建的开销与缓存策略

| 操作 | 典型耗时 | 频率 |
| --- | --- | --- |
| `vkCreateComputePipelines` | 毫秒级 | 首次 |
| `VulkanPipelineCache::get_or_create` 命中 | 微秒级 | 每次调用 |
| `vkCmdBindPipeline` | 纳秒级 | 每次 dispatch |

所以策略很明确：**首次创建，之后永远命中缓存**。绝不能把 `vkCreateComputePipelines` 放在热路径里。

---

## 12. 特化常量与推送常量

### 12.1 特化常量

**它是什么**：shader 里声明成 `layout(constant_id = N) const` 的常量。创建管线时，你通过 `VkSpecializationInfo` 给它一个具体值。

**为什么需要**：让驱动在编译时把值代入，删掉不可能执行的分支。

```glsl
layout(constant_id = 0) const int OP = 0;       // 默认值 0
layout(constant_id = 1) const int ADD_TO = 0;

if (OP == 0) {
    result = va + vb;
} else if (OP == 1) {
    result = va - vb;
}
// 如果创建管线时 OP = 0，编译器只会留下第一段
```

C++ 侧：

```cpp
VkSpecializationMapEntry entries[2] = {};
entries[0].constantID = 0;                 // 对应 shader 里的 constant_id 0
entries[0].offset     = 0;                 // 在 data 里的字节偏移
entries[0].size       = sizeof(uint32_t);

entries[1].constantID = 1;
entries[1].offset     = sizeof(uint32_t);
entries[1].size       = sizeof(uint32_t);

uint32_t data[2] = { spec_op, spec_add_to };

VkSpecializationInfo info = {};
info.mapEntryCount = 2;
info.pMapEntries   = entries;
info.dataSize      = sizeof(data);
info.pData         = data;
```

**关键点**：

- 特化值是**创建管线时**提供的，之后不能改；
- 不同值 = 不同管线；
- `offset` 是相对于 `pData` 的字节偏移，不是相对 shader 的。

### 12.2 推送常量

**它是什么**：每次 dispatch 时按值传入的一小块数据，直接内联在 command buffer 里，不占描述符。

**为什么需要**：像 `total` 这种每次调用都可能变的小标量，用描述符太重，用特化常量又太死。

```glsl
layout(push_constant) uniform PushConstants {
    int total;
} pc;
```

C++ 侧：

```cpp
struct ComputePushConstants {
    int32_t total;
};

ComputePushConstants pc = { static_cast<int32_t>(total) };
vkCmdPushConstants(cmd, pipeline.layout,
                   VK_SHADER_STAGE_COMPUTE_BIT,
                   0,              // offset
                   sizeof(pc),     // size
                   &pc);
```

**约束**：

- Vulkan 保证至少 128 字节；
- 内存布局必须和 C++ struct 完全一致；
- 范围要在 `VkPipelineLayout` 里声明。

### 12.3 两者对比

| | 特化常量 (Specialization Constant) | 推送常量 (Push Constant) |
| --- | --- | --- |
| 何时确定 | **创建 pipeline 时** | 每次 dispatch 时 |
| 能否运行时改 | ❌ 改了就换管线 | ✅ 每条命令都能不同 |
| 存在哪 | 编译进机器码 | 内联在 command buffer 里 |
| 典型用途 | op 类型、算法分支、tile 大小 | 元素总数、stride、偏移 |
| nnops 例子 | `OP`, `ADD_TO` | `total` |
| 数量 / 大小限制 | 由实现决定 | 保证 ≥ 128 字节 |

判断标准很简单：**这个值变了，是否需要重新编译？** 是→特化常量；否→推送常量。

---

## 13. 一次 dispatch 的完整生命周期

这一章把所有东西串起来。以 eltwise 为例，从用户代码角度看完整流程。

### 13.1 四个阶段

```text
═══════════════════════════════════════════════════════════════════
阶段 A：一次性初始化（进程级，只做一次）
═══════════════════════════════════════════════════════════════════
  vkCreateInstance
  vkEnumeratePhysicalDevices → pick GPU
  vkCreateDevice
  vkGetDeviceQueue
  vkCreateCommandPool
  vkCreateDescriptorPool

═══════════════════════════════════════════════════════════════════
阶段 B：每次调用（CPU 侧准备）
═══════════════════════════════════════════════════════════════════
  vkCreateBuffer + vkAllocateMemory + vkBindBufferMemory   ← 调用方
  memcpy(mapped, host_data) + vkFlushMappedMemoryRanges    ← 调用方

  vkAllocateCommandBuffers
  vkBeginCommandBuffer

  ┌─ 算子 compute() 内部 ────────────────────────────────────┐
  │  选择 SPIR-V (f32 / f16)                                 │
  │  构建 PipelineKey                                        │
  │  VulkanPipelineCache::get_or_create(key)                 │
  │     → 命中：直接返回（微秒级）                            │
  │     → 未命中：vkCreateComputePipelines（毫秒级，仅首次）  │
  │                                                          │
  │  vulkan_record_dispatch():                               │
  │    vkAllocateDescriptorSets(pool)   ← 从 pool 拿一个参数包 │
  │    vkUpdateDescriptorSets(A,B,C)    ← 把 buffer 塞进去    │
  │    vkCmdBindPipeline(compute, p)    ← 选函数               │
  │    vkCmdBindDescriptorSets(set=0)   ← 绑参数包            │
  │    vkCmdPushConstants(total)        ← 传标量              │
  │    vkCmdDispatch(ceil(total/256))   ← 记录"启动"          │
  └──────────────────────────────────────────────────────────┘

  vkEndCommandBuffer

═══════════════════════════════════════════════════════════════════
阶段 C：提交（这一刻 GPU 才真正开始工作）
═══════════════════════════════════════════════════════════════════
  vkQueueSubmit(queue, cmd_buffer)
  vkQueueWaitIdle(queue)          ← 测试环境里简单粗暴地等完

═══════════════════════════════════════════════════════════════════
阶段 D：回收
═══════════════════════════════════════════════════════════════════
  vkInvalidateMappedMemoryRanges
  memcpy(host_out, mapped)
  vkFreeCommandBuffers
  vkResetDescriptorPool(pool, 0)   ← 一次性回收本批所有描述符集
```

### 13.2 每个阶段在干什么

| 阶段 | 谁负责 | 目的 | 出错会怎样 |
| --- | --- | --- | --- |
| A 初始化 | 应用 / 测试环境 | 建立与 GPU 的会话 | 后续全部失败 |
| B 准备 | 调用方 + 算子 | 准备数据、录制命令 | 结果错误或崩溃 |
| C 提交 | 调用方 | 让 GPU 开始执行 | GPU 不工作 |
| D 回收 | 调用方 | 释放临时资源 | 内存泄漏或 use-after-free |

### 13.3 为什么"录制"和"执行"要分开

如果每条 `vkCmd*` 都立即执行，CPU 和 GPU 就完全串行了。Vulkan 的做法是：

- `vkCmd*` 只往 command buffer 里**写指令**（纯 CPU 操作，很快）；
- `vkQueueSubmit` 把整个指令缓冲一次性交给 GPU；
- GPU 可以异步执行，CPU 可以继续准备下一批。

这带来两个能力：**多线程并行录制**（每个线程录自己的 command buffer）和**命令复用**（同一段命令可以重复提交）。nnops 目前是"每算子一个 command buffer、提交后等待"，属于最简单正确的用法；进阶用法见[线程与内存模型](vulkan_threading_memory_model.md)。

### 13.4 nnops 的边界

```text
┌─────────────── 调用方负责 ───────────────┐
│ Instance / Device / Queue                │
│ CommandPool / CommandBuffer              │
│ DescriptorPool                           │
│ VkBuffer + VkDeviceMemory                │
│ 提交与同步                                │
└──────────────────────────────────────────┘
                    │  ComputeContext
                    ▼
┌─────────────── nnops 负责 ───────────────┐
│ 构建 PipelineKey                          │
│ 创建 / 缓存 VkPipeline                    │
│ 每次调用分配 + 填充 DescriptorSet         │
│ 录制 bind + push + dispatch               │
└──────────────────────────────────────────┘
```

这就是为什么 `ComputeContext` 里装的都是 `void*` 的 Vulkan 句柄——nnops 核心层不想让 `vulkan.h` 泄漏到公共头文件里。

---

## 14. 逐个对象详解

这些对象在第 4 章都已介绍过，这里只做速查，并展开最容易困惑的几个。每个对象的"一句话"都对应一个具体用途，比如 `VkBuffer` 是数据句柄、`VkDeviceMemory` 是真实显存，两者通过 `vkBindBufferMemory` 绑定。

| 对象 | 一句话 | 容易踩的坑 |
| --- | --- | --- |
| `VkInstance` | Vulkan 运行时入口 | 极少出错，创建时传 `apiVersion` |
| `VkPhysicalDevice` | 物理 GPU | 只用于查询，不执行任何操作 |
| `VkDevice` | 逻辑设备 | 创建时要声明启用哪些扩展/特性 |
| `VkQueue` | 提交通道 | 从 Device 取出，不是自己创建 |
| `VkBuffer` | 数据句柄（不是内存） | 必须和 `VkDeviceMemory` 绑定 |
| `VkDeviceMemory` | 真实显存 | 一个 buffer 一块内存，也可以子分配 |
| `VkCommandPool` | 命令缓冲分配器 | 与队列族绑定 |
| `VkCommandBuffer` | 录制的命令序列 | 录制完才能提交 |
| `VkDescriptorPool` | 描述符集分配器 | 容量要预估够 |
| `VkDescriptorSetLayout` | 参数包签名 | 和 pipeline 的 layout 必须兼容 |
| `VkDescriptorSet` | 参数包实例 | 一般每次 dispatch 重新分配/更新 |
| `VkPipelineLayout` | 完整调用约定 | 建管线时传入 |
| `VkShaderModule` | SPIR-V 包装 | 建完管线即可销毁 |
| `VkPipeline` | 编译好的 kernel | 贵，必须缓存 |
| `VkFence` / `VkSemaphore` | 同步原语 | fence 给 CPU 等，semaphore 给 GPU 等 |

### 14.1 `VkBuffer` 和 `VkDeviceMemory` 为什么是两步

CUDA 里 `cudaMalloc(&p, size)` 一步完成。Vulkan 拆成两步：

```cpp
// 第一步：描述"我要一个 buffer"（只是元数据，没有分配内存）
VkBufferCreateInfo bci = {};
bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
bci.size  = size_bytes;
bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | ...;
vkCreateBuffer(device, &bci, nullptr, &buffer);

// 第二步：查询驱动需要什么样的内存
VkMemoryRequirements req;
vkGetBufferMemoryRequirements(device, buffer, &req);
// req.size            实际需要的字节数（可能比 size_bytes 大）
// req.alignment       对齐要求
// req.memoryTypeBits  哪些内存类型可用

// 第三步：按需求分配真实显存
VkMemoryAllocateInfo mai = {};
mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
mai.allocationSize  = req.size;
mai.memoryTypeIndex = find_memory_type(phys, req.memoryTypeBits, ...);
vkAllocateMemory(device, &mai, nullptr, &memory);

// 第四步：绑定
vkBindBufferMemory(device, buffer, memory, 0);
```

为什么要拆？因为这让**子分配**成为可能：一个大的 `VkDeviceMemory` 可以绑定多个 `VkBuffer`（各自偏移不同），这是 Vulkan 内存池的基础。nnops 的测试环境图省事，每个 buffer 单独的 memory。

测试环境还要求内存是 `HOST_VISIBLE | HOST_COHERENT`：

- `HOST_VISIBLE`：CPU 能 `vkMapMemory` 读写；
- `HOST_COHERENT`：CPU 写完不需要手动 flush，GPU 就能看见；
- 代价：这类内存通常就是系统内存，GPU 通过 PCIe 访问，带宽不如显存。

### 14.2 CommandBuffer 的状态机

CommandBuffer 不是"想写就写"的普通数组，它有明确状态：

```text
Initial
  │ vkBeginCommandBuffer
  ▼
Recording ──── vkCmd* ────→ 只能录制，不能提交
  │ vkEndCommandBuffer
  ▼
Executable ─── vkQueueSubmit ───→ 可提交，不能录制
  │
  ▼
Pending → Invalid / 重新 begin（取决于 pool 的 reset 标志）
```

对 nnops 算子的意义：**算子被调用时，command buffer 已经处于 Recording 状态**。所以算子只调用 `vkCmd*`，绝不调用 `vkBeginCommandBuffer` / `vkEndCommandBuffer`——那是调用方的职责。

### 14.3 DescriptorSet 为什么每次调用都重新分配

`VkDescriptorSet` 里装的是"指向当前这批 buffer 的句柄"。下一批数据的 `VkBuffer` 可能不同，所以内容需要更新。

Vulkan 提供三种方式：

| 方式 | 做法 | 适用 |
| --- | --- | --- |
| 重新分配 | `vkAllocateDescriptorSets` → `vkUpdateDescriptorSets` | nnops 当前做法 |
| 复用 + 更新 | 保留 set，只 `vkUpdateDescriptorSets` | 更省，但要自己管理生命周期 |
| Push Descriptor | `vkCmdPushDescriptorSet` | 无需 pool（`VK_KHR_push_descriptor` 扩展） |

nnops 选择"每次分配一个，pool reset 时批量回收"，因为实现最简单，且测试里 pool 是 `maxSets=256`，绰绰有余。

> 注意：`vkUpdateDescriptorSets` 之后，**不能**在命令执行期间修改同一个 set，否则是未定义行为。这就是为什么每个 dispatch 用新 set。

### 14.4 Fence 和 Semaphore 的区别

```text
Fence:     CPU 等 GPU        vkWaitForFences(device, ...)
Semaphore: GPU 等 GPU        vkQueueSubmit(..., wait_sem, signal_sem)
```

nnops 测试用的是 `vkQueueWaitIdle`（最粗暴的"等整个队列空"），连 fence 都省了。生产环境应该用 fence 或 timeline semaphore，详见[线程与内存模型](vulkan_threading_memory_model.md)。

### 14.5 `VkPhysicalDeviceLimits`：那些上限在哪

创建管线、dispatch 时可能撞到的上限：

| 上限 | 含义 | 典型值 |
| --- | --- | --- |
| `maxComputeWorkGroupInvocations` | 一个 workgroup 最多多少线程 | 保证 ≥ 128，常见 1024 |
| `maxComputeWorkGroupSize[0..2]` | 每个维度最大值 | 常见 1024/1024/64 |
| `maxComputeWorkGroupCount[0..2]` | dispatch 每个维度最多多少组 | 保证 ≥ 65535 |
| `maxPushConstantsSize` | 推送常量最大字节数 | 保证 ≥ 128 |
| `maxDescriptorSetStorageBuffers` | 一个 set 最多多少 SSBO | 常见 8–16 |

这些值从 `VkPhysicalDeviceProperties::limits` 里读。nnops 的 `local_size_x = 256` 在所有支持 Vulkan 的 GPU 上都没问题。

---

## 15. Shader 与 C++ 的逐行对应

以 `eltwise_f32.comp` 和 `eltwise_vulkan.cpp` 为例，把两侧一一对上：

| GLSL（shader 侧） | C++（后端侧） | 产生了什么 Vulkan 对象 |
| --- | --- | --- |
| `layout(local_size_x = 256) in;` | `kVulkanWorkgroupSize = 256` | 编译进 SPIR-V；dispatch 时用它算 group 数 |
| `layout(constant_id = 0) const int OP;` | `key.spec_op = eltwise_op_to_spec(attrs.type);` | `VkSpecializationMapEntry`，影响 `VkPipeline` |
| `layout(set=0, binding=0) readonly buffer BufA;` | `bindings[0].binding = 0;` | `VkDescriptorSetLayout` 的一个 binding |
| `layout(push_constant) uniform { int total; } pc;` | `ComputePushConstants pc = { total };` | `VkPushConstantRange` |
| `gl_GlobalInvocationID.x` | `gid` | 无对象，GPU 运行时提供 |

**关键映射**：shader 里写了几行 `layout(set=0, binding=N)`，C++ 里就必须创建几个 `VkDescriptorSetLayoutBinding`，并且 `num_bindings` 要对得上。少一个，管线创建失败或绑定错位。

---

## 16. Workgroup 与 dispatch 的映射

### 16.1 三层并行

假设 `total = 1000`，shader 里写 `local_size_x = 256`，C++ 调 `vkCmdDispatch(cmd, 4, 1, 1)`。GPU 实际会启动 4 × 256 = 1024 个线程，按三层组织：

```text
一次 vkCmdDispatch(Gx, Gy, Gz)
│
├─ Workgroup (工作组) × Gx*Gy*Gz 个        ← 由 CPU 指定数量
│    │  每个 workgroup 有 local_size_x*y*z 个线程
│    │  一个 workgroup 调度到 GPU 的一个 CU（计算单元）
│    │  workgroup 内部可以共享 shared memory、barrier
│    │
│    ├─ Subgroup / Wave (32 或 64 线程)     ← 硬件调度单位
│    │   同一 subgroup 的线程锁步执行，可以 shuffle
│    │
│    └─ Invocation (单个线程)                ← shader 的 main() 一次执行
│        读取自己的 gl_GlobalInvocationID
```

**为什么叫"工作组"**：同一个 workgroup 内的线程可以共享 `shared` 内存，并用 `barrier()` 同步。不同 workgroup 之间不能直接共享，除非通过全局内存 + 原子/屏障。

### 16.2 一维逐元素的映射

```cpp
// C++ 侧（vulkan_common.cpp）
uint32_t group_count = ceil_div(total_elements, 256);
vkCmdDispatch(cmd, group_count, 1, 1);
```

```glsl
// Shader 侧
int gid = int(gl_GlobalInvocationID.x);   // = group_id * 256 + local_id
if (gid >= pc.total) return;              // 必须的边界守卫
```

例如 `total = 1000`：

```text
group 0: gid 0..255
group 1: gid 256..511
group 2: gid 512..767
group 3: gid 768..1023   ← 只有 768..999 有效，后 24 个线程越界
```

`ceil(1000/256) = 4`，所以最后一组必然有线程越界。**没有边界守卫 = 读写越界 = 未定义行为**。

### 16.3 内置变量

| 变量 | 含义 | 例子（Gx=4, local=256） |
| --- | --- | --- |
| `gl_GlobalInvocationID` | 整个 dispatch 内的全局编号 | 0..1023 |
| `gl_LocalInvocationID` | workgroup 内编号 | 0..255 |
| `gl_WorkGroupID` | 当前 workgroup 编号 | 0..3 |
| `gl_NumWorkGroups` | 总 workgroup 数 | 4 |
| `gl_WorkGroupSize` | 每个 workgroup 的线程数 | 256 |

关系：

```text
gl_GlobalInvocationID = gl_WorkGroupID * gl_WorkGroupSize + gl_LocalInvocationID
```

### 16.4 和 CUDA 的对应

| CUDA | Vulkan | 说明 |
| --- | --- | --- |
| `blockDim` | `local_size_*` | 写死在 shader 里 |
| `gridDim` | `vkCmdDispatch` 参数 | 运行时指定 |
| `threadIdx.x` | `gl_LocalInvocationID.x` | workgroup 内编号 |
| `blockIdx.x` | `gl_WorkGroupID.x` | workgroup 编号 |
| `blockIdx.x * blockDim.x + threadIdx.x` | `gl_GlobalInvocationID.x` | 全局编号 |
| `__shared__` | `shared` | 需手动 `barrier()` |
| `__syncthreads()` | `barrier()` | |
| `__shfl_sync` | `subgroupShuffle` | |

---

## 17. nnops 对象映射总表

| Vulkan 对象 | nnops 里的承载者 | 文件 |
| --- | --- | --- |
| `VkDevice` | `ComputeContext::vulkan_device` | compute_context.hpp |
| `VkCommandBuffer` | `ComputeContext::vulkan_cmd_buffer` | compute_context.hpp |
| `VkDescriptorPool` | `ComputeContext::vulkan_descriptor_pool` | compute_context.hpp |
| `VkBuffer[]` | `ComputeContext::vulkan_buffers` | compute_context.hpp |
| `VkPipeline` | `VulkanPipeline::pipeline` | vulkan_common.hpp |
| `VkPipelineLayout` | `VulkanPipeline::layout` | vulkan_common.hpp |
| `VkDescriptorSetLayout` | `VulkanPipeline::set_layout` | vulkan_common.hpp |
| `VkShaderModule` | 临时对象，建完管线即销毁 | vulkan_common.cpp |
| SPIR-V 字节 | `g_<op>_<dtype>_spv[]`（`constexpr` 数组） | `<op>_f32_spv.h` |
| 特化常量 | `PipelineKey::spec_op` / `spec_add_to` | vulkan_common.hpp |
| 推送常量 | `ComputePushConstants{ int32_t total; }` | vulkan_common.hpp |
| Workgroup 大小 | `kVulkanWorkgroupSize = 256` | vulkan_common.hpp |
| 管线缓存 | `VulkanPipelineCache::instance()` | vulkan_common.cpp |
| 一次性录制 | `vulkan_record_dispatch()` | vulkan_common.cpp |

---

## 18. 常见误解 FAQ

### Q1：为什么不能像 CUDA 那样直接传指针？

CUDA 的指针只在同一进程的 CUDA runtime 里有效，驱动帮你做地址翻译。Vulkan 是跨厂商、跨进程的显式 API，GPU 只能通过**描述符**访问内存。所以指针要包进 `VkBuffer`，再包进 `VkDescriptorSet`。

### Q2：为什么每次改一个操作就要新管线？不能传个 `op` 参数吗？

可以传（用推送常量），但那样 shader 里就要写 `if (OP == 0) ... else if (OP == 1) ...`，每次执行都做分支判断。用特化常量让**驱动在编译时就把其他分支删掉**，运行时零开销。nnops 选择后者，代价是管线数量多几条。

### Q3：管线创建到底多贵？

`vkCreateComputePipelines` 通常在毫秒级，取决于驱动和特化程度。一个模型几十个算子、每算子几种 dtype/op 组合，首次运行可能多花几十毫秒。缓存之后 `get_or_create` 只是一次哈希查找，微秒级。所以**绝不能在热路径里反复创建管线**。

### Q4：`set = 0, binding = 0` 到底对应什么？

- `set = 0`：第 0 个描述符集 → `vkCmdBindDescriptorSets(..., firstSet = 0, ...)`
- `binding = 0`：该 set 内的第 0 个槽位 → `VkDescriptorSetLayoutBinding{ .binding = 0 }`
- 槽位里放的是 `VkBuffer` → `VkDescriptorBufferInfo{ .buffer = ... }`

三者编号必须一致，否则 shader 读到的就是别的 buffer（或者直接未定义行为）。

### Q5：可以一条管线同时处理 f32 和 f16 吗？

不行。`float` 和 `half` 是不同的 SPIR-V 类型，需要不同的 shader blob，因此是两条管线。nnops 每个算子有 `<op>_f32` 和 `<op>_f16` 两套 SPIR-V，`PipelineKey::spirv_data` 区分它们。

### Q6：`VkBuffer` 是"一块显存"吗？

不是。`VkBuffer` 是"一段有大小和用途的区域"的描述，**内存由 `VkDeviceMemory` 提供**，两者通过 `vkBindBufferMemory` 绑定。一个 `VkDeviceMemory` 可以绑定多个 buffer。

### Q7：为什么必须 `vkDeviceWaitIdle` / `vkQueueWaitIdle` 才能销毁对象？

GPU 可能还在异步执行命令。销毁一个正在被 GPU 使用的对象是未定义行为。所以销毁前必须确保队列空闲（或至少用 fence/semaphore 确认相关命令已完成）。nnops 测试环境的 `submit_and_wait` 就是"提交 → 等空闲 → 回收资源"。

### Q8：`VkDescriptorSetLayout` 和 `VkPipelineLayout` 能不能只建一个？

不能。前者只描述"一个参数包"，后者描述"一个管线要几个参数包 + 推送常量范围"。一个管线理论上可以接受多个描述符集（`set = 0`、`set = 1`……），所以必须是两层。

### Q9：为什么算子代码里到处是 `static_cast<VkXxx>(ctx.vulkan_xxx)`？

因为 `ComputeContext` 在公共头文件里，不能 `#include <vulkan/vulkan.h>`（否则所有用户都被迫装 Vulkan SDK）。所以字段用 `void*`，算子内部再转回真实类型。

### Q10：SSBO 和 UBO 该用哪个？

nnops 全用 SSBO（`buffer`），因为：

- SSBO 没有 64 KB 上限
- SSBO 支持运行时长度数组 `float a[]`
- SSBO 可写

UBO（`uniform`）适合小、只读、频繁访问的参数块，走常量缓存，但 nnops 用推送常量代替了它。

### Q11：为什么每个结构体都要设 `sType`？

因为 Vulkan 用 `void*` 和 `pNext` 把结构体串起来，驱动只能靠 `sType` 判断"这个指针实际指向哪种结构体"。不设或设错 = 驱动按错误布局解析 = UB。

### Q12：为什么我的结果错了，但没有任何错误码？

Vulkan 的很多错误是 UB，不返回错误码。常见原因：

- shader 越界读写；
- binding 编号错位；
- descriptor set 与 pipeline layout 不匹配；
- 使用了已销毁或未初始化的资源；
- 缺少内存屏障（跨 dispatch）。

开校验层、用 `VK_CHECK`、逐层核对绑定关系是唯一的办法。

### Q13：compute 程序需要 swapchain 吗？

不需要。swapchain（交换链）是图形程序用来把图像显示到屏幕的。compute-only 程序只需要 buffer + queue，不需要窗口系统集成。

### Q14：为什么 `local_size_x` 是 256？

256 是 32 和 64 的公倍数，能填满常见的 subgroup/wavefront；同时不超过所有 Vulkan 设备保证的 `maxComputeWorkGroupInvocations`（≥128）。nnops 全项目统一用它，便于计算 group 数。

### Q15：`VkPipelineCache` 和 nnops 的 `VulkanPipelineCache` 是同一个东西吗？

不是。`VkPipelineCache` 是 Vulkan 的驱动级缓存对象，可以把编译结果序列化到磁盘；nnops 的 `VulkanPipelineCache` 是一个 C++ 单例，内部用 `unordered_map` 记住"这个 key 对应哪个 `VkPipeline` 句柄"。名字像，作用不同。

### Q16：一个 command buffer 里能录多个算子吗？

可以。command buffer 本质上是一串命令，你可以依次录制 dispatch A、dispatch B、dispatch C，然后一次提交。nnops 的测试环境通常是"一算子一提交"，但 API 本身支持合批。注意：跨 dispatch 的数据依赖需要正确的 barrier。

### Q17：为什么 `VkShaderModule` 建完管线就能销毁？

因为 `vkCreateComputePipelines` 会把 SPIR-V 编译成 GPU 机器码，并把需要的信息复制进管线对象。`VkShaderModule` 只是编译输入，之后不再被引用。

### Q18：`VK_NULL_HANDLE` 是什么？

就是 0，表示"空句柄"。大多数创建函数的输出参数在失败时会保持 `VK_NULL_HANDLE`。销毁函数传 `VK_NULL_HANDLE` 通常是安全的（no-op），但不要依赖这一点。

---

## 19. 术语表

| 英文 | 常见中文 | 一句话 |
| --- | --- | --- |
| Instance | 实例 | 应用与 Vulkan 运行时的连接 |
| Physical Device | 物理设备 | 一块真实 GPU |
| Logical Device | 逻辑设备 | 你与 GPU 的会话 |
| Queue | 队列 | 提交命令的通道 |
| Queue Family | 队列族 | 具有相同能力的队列集合 |
| Command Buffer | 命令缓冲区 | 录制下来的一串命令 |
| Command Pool | 命令池 | 命令缓冲区的分配器 |
| Pipeline | 管线 | 编译好的、状态固定的可执行单元 |
| Pipeline Layout | 管线布局 | 管线接受的描述符集 + 推送常量 |
| Descriptor | 描述符 | 指向资源（buffer/image）的一个引用 |
| Descriptor Set | 描述符集 | 一组描述符，即"参数包" |
| Descriptor Set Layout | 描述符集布局 | "参数包"的类型签名 |
| Descriptor Pool | 描述符池 | 描述符集的分配器 |
| Binding | 绑定点 | 描述符集里的槽位编号 |
| Push Constant | 推送常量 | 内联在命令里的按值小参数 |
| Specialization Constant | 特化常量 | 创建管线时固化的编译期常量 |
| Shader Module | 着色器模块 | SPIR-V 二进制的包装 |
| SPIR-V | — | Vulkan 的着色器字节码格式 |
| GLSL | — | 类 C 的着色器语言 |
| Workgroup | 工作组 | 一次 dispatch 中可共享内存的线程组 |
| Invocation | 调用/线程 | 着色器 `main()` 的一次执行 |
| Subgroup / Wave | 子组/波前 | 硬件锁步执行的线程组（32/64） |
| SSBO | 着色器存储缓冲对象 | 可读写的大块数据缓冲区 |
| UBO | 统一缓冲对象 | 只读的小参数缓冲区 |
| Fence | 栅栏 | CPU 等待 GPU 的同步对象 |
| Semaphore | 信号量 | GPU 等待 GPU 的同步对象 |
| Barrier | 屏障 | 管线阶段之间的执行/内存依赖 |
| Memory Type | 内存类型 | 显存/系统内存等属性组合 |
| Host Visible | 主机可见 | CPU 能映射访问 |
| Host Coherent | 主机一致 | CPU 写完无需手动 flush |
| Device Local | 设备本地 | 位于显存 |
| Validation Layer | 校验层 | 开发期检查 API 误用的层 |
| ICD | 可安装客户端驱动 | 厂商提供的 Vulkan 实现 |
| sType | 结构体类型 | 标明"我是哪种结构体" |
| pNext | 扩展链 | 指向下一个可选结构体 |
| VkResult | 返回码 | Vulkan 的错误返回类型 |
| Extension | 扩展 | 核心之外追加的功能 |
| Feature | 特性 | 可查询/开启的能力开关 |

---

## 20. 下一步读什么

```text
你在这里
   │
   ▼
vulkan_zero_basics.md          ← 建立整体心智模型（本文）
   │
   ├─→ vulkan_runtime_interfaces.md
   │     nnops 的结构体、接口、生命周期逐条对照
   │
   ├─→ vulkan_compute_shader_grammar.md
   │     GLSL 写法的完整参考（workgroup、SSBO、shared、subgroup、atomic）
   │
   ├─→ how_to_add_vulkan_op.md
   │     从零加一个算子：shader → 后端 → 调度 → 测试
   │
   └─→ vulkan_threading_memory_model.md
         多线程录制、同步原语、SPIR-V 内存模型、Timeline Semaphore
```

如果只想尽快读懂现有代码，读本文第 3、4、10、11、13、17 章，然后直接看 [运行时结构体与接口](vulkan_runtime_interfaces.md)。
