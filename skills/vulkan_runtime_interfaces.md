# Vulkan 运行时结构体与接口说明

> 基于 nnops 项目 Vulkan 后端源码分析，详细说明 Vulkan 计算管线在运行时所涉及的全部结构体、接口及生命周期。

---

## 目录

1. [架构总览](#1-架构总览)
2. [ComputeContext — 上下文传递](#2-computecontext--上下文传递)
3. [SpirvBlob — SPIR-V 着色器载体](#3-spirvblob--spir-v-着色器载体)
4. [ComputePushConstants — 推送常量](#4-computepushconstants--推送常量)
5. [PipelineKey — 管线缓存键](#5-pipelinekey--管线缓存键)
6. [VulkanPipeline — 管线对象](#6-vulkanpipeline--管线对象)
7. [VulkanPipelineCache — 管线缓存单例](#7-vulkanpipelinecache--管线缓存单例)
8. [vulkan_record_dispatch — 录制调度命令](#8-vulkan_record_dispatch--录制调度命令)
9. [着色器编译管线](#9-着色器编译管线)
10. [VulkanTestEnv — 测试基础设施](#10-vulkantestenv--测试基础设施)
11. [VulkanBuffer — 测试缓冲区](#11-vulkanbuffer--测试缓冲区)
12. [算子实现标准模板](#12-算子实现标准模板)
13. [数据流全景图](#13-数据流全景图)
14. [常量与约定](#14-常量与约定)

---

## 1. 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                      nnops 调度层                                │
│  src/ops/<op>.cpp                                                │
│  resolve_kernel() → case Backend::Vulkan: return vulkan_xxx     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ 函数指针
┌──────────────────────────▼──────────────────────────────────────┐
│                   Vulkan 算子入口 (C++)                           │
│  src/backend/vulkan/<op>_vulkan.cpp                              │
│                                                                  │
│  ① 选择 SPIR-V blob (f32 / f16)                                  │
│  ② 从 ComputeContext 提取 VkDevice / VkCommandBuffer / ...       │
│  ③ 构建 PipelineKey → VulkanPipelineCache::get_or_create()       │
│  ④ 构建 ComputePushConstants                                     │
│  ⑤ vulkan_record_dispatch() → 录制到 Command Buffer              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                 Vulkan 公共基础设施                               │
│  src/backend/vulkan/vulkan_common.{hpp,cpp}                      │
│                                                                  │
│  VulkanPipelineCache (单例)      vulkan_record_dispatch()        │
│  ├─ get_or_create(key)           ├─ vkAllocateDescriptorSets    │
│  ├─ create_pipeline(key)         ├─ vkUpdateDescriptorSets      │
│  ├─ destroy_device_pipelines()   ├─ vkCmdBindPipeline           │
│  └─ clear()                      ├─ vkCmdBindDescriptorSets     │
│                                  ├─ vkCmdPushConstants          │
│                                  └─ vkCmdDispatch               │
└─────────────────────────────────────────────────────────────────┘
```

**关键设计决策：**

| 决策 | 说明 |
|------|------|
| Pipeline 缓存 | 全局单例，按 `(device, SPIR-V, spec_op, spec_add_to, num_bindings)` 缓存 |
| Descriptor Set | 每次 `compute()` 从用户提供的 Pool 分配，submit 后通过 Pool Reset 回收 |
| 内存管理 | 算子不做内存分配——用户负责 `VkBuffer` + `VkDeviceMemory` 的完整生命周期 |
| 着色器嵌入 | SPIR-V 以 `constexpr uint32_t[]` 编译期内嵌，零运行时文件 I/O |
| 特化常量 | 操作类型(OP)和 add_to 标志在管线创建时固化，编译器消除死分支 |

---

## 2. ComputeContext — 上下文传递

**文件**: [include/nnops/core/compute_context.hpp](include/nnops/core/compute_context.hpp)

```cpp
struct ComputeContext {
    Backend expected_backend = Backend::CPU;  // 目标后端
    ParallelForFn cpu_parallel_for = nullptr; // CPU 并行钩子
    void* cuda_stream = nullptr;              // CUDA 流

    // ── Vulkan 专用字段 ──
    void* vulkan_cmd_buffer = nullptr;        // VkCommandBuffer — 录制目标
    void* vulkan_device = nullptr;            // VkDevice — 管线创建/描述符分配
    void* vulkan_descriptor_pool = nullptr;   // VkDescriptorPool — 每调用分配描述符集
    const void* vulkan_buffers = nullptr;     // VkBuffer 句柄数组 (输入先, 输出后)
    int vulkan_buffers_count = 0;             // 数组条目数
};
```

### 字段详解

| 字段 | Vulkan 类型 | 用途 | 生命周期要求 |
|------|-------------|------|-------------|
| `vulkan_cmd_buffer` | `VkCommandBuffer` | 算子将 `vkCmd*` 命令录制到此 buffer | Recording 状态，算子不调用 `vkBeginCommandBuffer`/`vkEndCommandBuffer` |
| `vulkan_device` | `VkDevice` | 管线创建、描述符集分配 | 必须在算子调用期间保持有效 |
| `vulkan_descriptor_pool` | `VkDescriptorPool` | 每次 dispatch 从此池分配 1 个描述符集 | 需要 `FREE_DESCRIPTOR_SET_BIT` 或整个池 reset |
| `vulkan_buffers` | `const VkBuffer*` | 指向 VkBuffer 句柄数组的指针 | 数组在算子调用期间保持有效 |
| `vulkan_buffers_count` | `int` | 缓冲区数量 | 必须 ≥ 着色器中声明的 binding 数量 |

### 缓冲区数组布局约定

```
Eltwise (3 buffers):   [0]=input A,  [1]=input B,  [2]=output
Unary   (2 buffers):   [0]=input,    [1]=output
```

---

## 3. SpirvBlob — SPIR-V 着色器载体

**文件**: [src/backend/vulkan/vulkan_common.hpp](src/backend/vulkan/vulkan_common.hpp)

```cpp
struct SpirvBlob {
    const uint32_t* data;       // SPIR-V 二进制数据指针
    size_t          size_bytes; // 字节数
};
```

### 实际使用方式

项目中 SPIR-V 数据以全局 `constexpr` 数组嵌入（由构建系统自动生成）：

```cpp
// 声明于 eltwise_f32_spv.h（构建时由 spv_to_header.py 生成）
namespace nnops::backend::vulkan {
    constexpr uint32_t g_eltwise_f32_spv[] = { 0x07230203, 0x00010000, ... };
    constexpr size_t   g_eltwise_f32_spv_len = sizeof(g_eltwise_f32_spv);
}
```

在算子代码中直接使用这些全局变量填充 `PipelineKey`：

```cpp
key.spirv_data = g_eltwise_f32_spv;
key.spirv_size = g_eltwise_f32_spv_len;
```

**注意**：`SpirvBlob` 结构体本身在代码中声明但当前未被直接使用——实际数据通过 `PipelineKey` 的 `spirv_data`/`spirv_size` 字段传递。

---

## 4. ComputePushConstants — 推送常量

**文件**: [src/backend/vulkan/vulkan_common.hpp](src/backend/vulkan/vulkan_common.hpp)

```cpp
struct ComputePushConstants {
    int32_t total;  // 总元素数量
};
```

### 详解

| 项目 | 值 |
|------|-----|
| 大小 | 4 bytes (单个 int32) |
| 传递方式 | `vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc)` |
| Shader 侧 | `layout(push_constant) uniform PushConstants { int total; } pc;` |
| 用途 | 告知着色器本次 dispatch 需要处理的元素总数 |
| Vulkan 限制 | 保证 ≥ 128 bytes（当前仅用 4 bytes，远未触及上限） |

---

## 5. PipelineKey — 管线缓存键

**文件**: [src/backend/vulkan/vulkan_common.hpp](src/backend/vulkan/vulkan_common.hpp)

```cpp
struct PipelineKey {
    VkDevice        device;         // 不同设备的管线独立缓存
    const uint32_t* spirv_data;     // SPIR-V 二进制数据指针 (标识着色器)
    size_t          spirv_size;     // SPIR-V 数据大小 (用于 hash)
    uint32_t        spec_op;        // 特化常量: 操作类型 (0=Add, 1=Sub, ...)
    uint32_t        spec_add_to;    // 特化常量: 是否累加到输出 (0=overwrite, 1=accumulate)
    uint32_t        num_bindings;   // 描述符绑定数量 (eltwise=3, unary=2)

    bool operator==(const PipelineKey& other) const noexcept;
};

struct PipelineKeyHash {
    size_t operator()(const PipelineKey& k) const noexcept;
};
```

### 缓存键的语义

每个唯一的 `PipelineKey` 对应一个唯一的 `VkPipeline`。改变任何字段都会导致创建新管线：

| 字段变化 | 触发原因 | 是否会创建新管线 |
|----------|----------|:---:|
| `device` | 不同 VkDevice | ✅ |
| `spirv_data` | f32 vs f16 shader, 不同算子 | ✅ |
| `spirv_size` | 着色器代码修改 | ✅ |
| `spec_op` | 不同操作类型 (Add/Sub/Mul/...) | ✅ |
| `spec_add_to` | overwrite vs accumulate | ✅ |
| `num_bindings` | 不同数量的 I/O buffer | ✅ |

`PipelineKeyHash` 将所有字段混合为单个 `size_t`，用于 `std::unordered_map` 的桶查找。

---

## 6. VulkanPipeline — 管线对象

**文件**: [src/backend/vulkan/vulkan_common.hpp](src/backend/vulkan/vulkan_common.hpp)

```cpp
struct VulkanPipeline {
    VkPipeline            pipeline   = VK_NULL_HANDLE;  // 计算管线句柄
    VkPipelineLayout      layout     = VK_NULL_HANDLE;  // 管线布局 (descriptor set + push constant)
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;  // 描述符集布局 (binding 定义)
};
```

### 三个 Vulkan 对象的关系

```
VkDescriptorSetLayout (set_layout)
  │ 定义: binding 0 = STORAGE_BUFFER, binding 1 = STORAGE_BUFFER, ...
  │ 创建时传入: num_bindings → 创建对应数量的 STORAGE_BUFFER binding
  │
  └─→ VkPipelineLayout (layout)
        │ 组合: 1× VkDescriptorSetLayout + 1× VkPushConstantRange
        │
        └─→ VkPipeline (pipeline)
              │ 组合: VkShaderModule (SPIR-V) + VkPipelineLayout
              │       + VkSpecializationInfo (spec_op, spec_add_to)
              │
              └─→ vkCmdBindPipeline(cmd, COMPUTE, pipeline)
```

### 创建流程 (VulkanPipelineCache::create_pipeline)

```
vkCreateShaderModule(device, SPIR-V)
  │
  ├─→ vkCreateDescriptorSetLayout(device, bindings[0..N-1])
  │     └─ 每个 binding: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage=COMPUTE
  │
  ├─→ vkCreatePipelineLayout(device, {set_layout}, {push_constant_range})
  │     └─ push_constant_range: stage=COMPUTE, size=sizeof(ComputePushConstants)
  │
  ├─→ vkCreateComputePipelines(device, {stage_ci})
  │     └─ stage_ci: shader_module + "main" + specialization_info
  │
  └─→ vkDestroyShaderModule(device, shader_module)  // 管线创建后可销毁
```

### 销毁

```
vkDestroyPipeline(device, pipeline)
vkDestroyPipelineLayout(device, layout)
vkDestroyDescriptorSetLayout(device, set_layout)
```

---

## 7. VulkanPipelineCache — 管线缓存单例

**文件**: [src/backend/vulkan/vulkan_common.hpp](src/backend/vulkan/vulkan_common.hpp), 实现在 [vulkan_common.cpp](src/backend/vulkan/vulkan_common.cpp)

```cpp
class VulkanPipelineCache {
public:
    static VulkanPipelineCache& instance();  // 全局单例

    VulkanPipeline get_or_create(const PipelineKey& key);  // 获取或创建管线
    void destroy_device_pipelines(VkDevice device);        // 清理指定设备的所有管线
    void clear();                                          // 清理全部管线

private:
    VulkanPipeline create_pipeline(const PipelineKey& key);

    std::mutex mutex_;
    std::unordered_map<PipelineKey, VulkanPipeline, PipelineKeyHash> cache_;
};
```

### 线程安全策略

采用 **双检查锁定 (Double-Checked Locking)** 模式：

```cpp
VulkanPipeline get_or_create(const PipelineKey& key) {
    // ① 读锁：快速路径 — 命中缓存直接返回
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) return it->second;
    }

    // ② 无锁：创建管线（昂贵操作，不阻塞其他线程的缓存读取）
    VulkanPipeline pipeline = create_pipeline(key);

    // ③ 写锁：Double-check — 另一个线程可能抢先创建了相同管线
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            // 放弃冗余管线，返回已有结果
            destroy_pipeline_objects(key.device, pipeline);
            return it->second;
        }
        cache_[key] = pipeline;
        return pipeline;
    }
}
```

### 生命周期

```
应用启动
  │
  ├─ 首次 compute() → VulkanPipelineCache::get_or_create()
  │   └─ 缓存未命中 → create_pipeline() → 存入 cache_
  │
  ├─ 后续 compute() (相同 key) → get_or_create()
  │   └─ 缓存命中 → 直接返回 (加锁 + 查找，微秒级)
  │
  └─ 应用退出前
      └─ VulkanPipelineCache::destroy_device_pipelines(device)
          └─ 遍历 cache_，销毁该 device 的所有管线
          └─ 必须在 vkDestroyDevice 之前调用！
```

---

## 8. vulkan_record_dispatch — 录制调度命令

**文件**: [src/backend/vulkan/vulkan_common.cpp](src/backend/vulkan/vulkan_common.cpp)

```cpp
void vulkan_record_dispatch(
    VkCommandBuffer        cmd,              // 命令缓冲区
    VkDevice               device,           // 设备句柄
    VkDescriptorPool       pool,             // 描述符池 (从此分配描述符集)
    const VulkanPipeline&  pipeline,         // 管线 (layout + set_layout + pipeline)
    const VkBuffer*        buffers,          // VkBuffer 数组 [input..., output]
    uint32_t               num_buffers,      // 缓冲区数量
    const void*            push_constants,   // 推送常量数据指针
    uint32_t               pc_size,          // 推送常量字节数
    uint32_t               total_elements);  // 总元素数 (用于计算 workgroup 数量)
```

### 内部执行步骤

```
Step 1: vkAllocateDescriptorSets
  ├─ 从 pool 分配 1 个描述符集
  └─ 使用 pipeline.set_layout 匹配绑定布局

Step 2: vkUpdateDescriptorSets
  ├─ 为每个 binding i: 填充 VkDescriptorBufferInfo{buffer=buffers[i], offset=0, range=VK_WHOLE_SIZE}
  ├─ 所有 binding 类型: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
  └─ 写入描述符集

Step 3: vkCmdBindPipeline
  └─ 绑定计算管线: VK_PIPELINE_BIND_POINT_COMPUTE

Step 4: vkCmdBindDescriptorSets
  └─ 绑定描述符集到 set=0

Step 5: vkCmdPushConstants
  └─ 推送常量: stage=COMPUTE, offset=0, size=pc_size

Step 6: vkCmdDispatch
  ├─ group_count = ceil(total_elements / 256)
  └─ dispatch: (group_count, 1, 1)
```

### 内存管理

```
每次调用 vulkan_record_dispatch:
  ├─ new VkDescriptorBufferInfo[num_buffers]   (堆分配)
  ├─ new VkWriteDescriptorSet[num_buffers]     (堆分配)
  ├─ vkAllocateDescriptorSets(...)             (从 pool 分配)
  └─ delete[] writes; delete[] buf_infos       (函数内释放)

Pool 回收:
  ├─ 测试环境: submit_and_wait() 后 vkResetDescriptorPool(pool, 0)
  └─ 生产环境: 用户负责 Pool 管理
```

---

## 9. 着色器编译管线

### 9.1 整体流程

```
shaders/<op>_f32.comp  (GLSL 源码)
  │
  │  glslc --target-env=vulkan1.2 -o <op>_f32.spv <op>_f32.comp
  │
  ├─→ <op>_f32.spv  (SPIR-V 二进制, 每 4 字节 little-endian word)
  │
  │  python spv_to_header.py --name g_<op>_f32 --spv <op>_f32.spv --out <op>_f32_spv.h
  │
  └─→ <op>_f32_spv.h  (C++ constexpr uint32_t 数组, 编译期内嵌)
        │
        │  #include "<op>_f32_spv.h"
        │
        └─→ C++ 编译器 → 嵌入 .text 段 → nnops_vulkan.lib
```

### 9.2 spv_to_header.py

**文件**: [src/backend/vulkan/spv_to_header.py](src/backend/vulkan/spv_to_header.py)

```python
def spv_to_c(name, spv_path, h_path):
    with open(spv_path, 'rb') as f:
        data = f.read()
    words = []
    for i in range(0, len(data), 4):
        w = int.from_bytes(data[i:i+4], 'little')
        words.append(str(w))

    # 输出:
    #   namespace nnops::backend::vulkan {
    #       constexpr uint32_t g_<name>_spv[] = { 0x07230203, ... };
    #       constexpr size_t   g_<name>_spv_len = sizeof(g_<name>_spv);
    #   }
```

**命名约定**：

| 格式 | 示例 |
|------|------|
| GLSL 源文件 | `eltwise_f32.comp` |
| SPIR-V 二进制 | `eltwise_f32.spv` |
| C++ 头文件 | `eltwise_f32_spv.h` |
| C++ 数组名 | `g_eltwise_f32_spv` / `g_eltwise_f32_spv_len` |

### 9.3 CMake 集成

```cmake
set(VULKAN_SHADERS eltwise_f32 unary_f32 eltwise_f16 unary_f16)

foreach(shader ${VULKAN_SHADERS})
    # Step 1: .comp → .spv
    add_custom_command(
        OUTPUT ${SHADER_SPV}
        COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.2 -o ${SHADER_SPV} ${SHADER_SRC}
        DEPENDS ${SHADER_SRC}
    )
    # Step 2: .spv → _spv.h
    add_custom_command(
        OUTPUT ${SHADER_HDR}
        COMMAND Python3::Interpreter spv_to_header.py --name g_${shader} ...
        DEPENDS ${SHADER_SPV}
    )
endforeach()
```

---

## 10. VulkanTestEnv — 测试基础设施

**文件**: [tests/common/vulkan_test_helper.hpp](tests/common/vulkan_test_helper.hpp)

```cpp
class VulkanTestEnv {
public:
    VulkanTestEnv();   // 创建 Instance → Device → CommandPool → DescriptorPool
    ~VulkanTestEnv();  // 逆序销毁 + 清理 PipelineCache

    // 访问器
    VkDevice         device()          const;
    VkDescriptorPool descriptor_pool() const;
    VkQueue          queue()           const;
    bool             supports_fp16()   const;

    // 缓冲区操作
    Buffer create_buffer(size_t size_bytes, VkBufferUsageFlags usage);
    void   destroy_buffer(Buffer& buf);
    void   copy_to_device(Buffer& buf, const void* data, size_t size);
    void   copy_from_device(Buffer& buf, void* data, size_t size);

    // 命令录制
    VkCommandBuffer begin_cmd();       // 分配 + BeginCommandBuffer
    void            submit_and_wait(); // EndCommandBuffer → QueueSubmit → WaitIdle → Free → ResetPool
};
```

### 资源创建顺序

```
VulkanTestEnv()
  ├─ vkCreateInstance()
  ├─ vkEnumeratePhysicalDevices() → 优先选 DISCRETE_GPU
  │   └─ 查询 VK_KHR_16bit_storage 扩展支持
  ├─ vkCreateDevice()
  │   ├─ 条件启用 VK_KHR_16bit_storage
  │   └─ 获取 Compute Queue
  ├─ vkCreateCommandPool(RESET_COMMAND_BUFFER_BIT)
  └─ vkCreateDescriptorPool(STORAGE_BUFFER × 256, maxSets=256)
```

### 资源销毁顺序

```
~VulkanTestEnv()
  ├─ vkDeviceWaitIdle()
  ├─ VulkanPipelineCache::destroy_device_pipelines(device)  ← 必须在 DestroyDevice 之前
  ├─ vkDestroyDescriptorPool()
  ├─ vkDestroyCommandPool()
  ├─ vkDestroyDevice()
  └─ vkDestroyInstance()
```

### 每次测试的标准模式

```
begin_cmd()
  └─→ 构建 ComputeContext (设置 vulkan_* 字段)
      └─→ 调用算子 compute()
          └─→ submit_and_wait()
              └─→ copy_from_device() 验证结果
```

---

## 11. VulkanBuffer — 测试缓冲区

**文件**: [tests/common/vulkan_test_helper.hpp](tests/common/vulkan_test_helper.hpp) (VulkanTestEnv::Buffer)

```cpp
struct Buffer {
    VkBuffer       buffer = VK_NULL_HANDLE;  // Vulkan 缓冲区句柄
    VkDeviceMemory memory = VK_NULL_HANDLE;  // 设备内存
    size_t         size   = 0;              // 分配大小 (字节)
    void*          mapped = nullptr;         // 持久映射指针 (HOST_VISIBLE + HOST_COHERENT)
};
```

### 创建流程

```
create_buffer(size, usage)
  ├─ vkCreateBuffer(STORAGE_BUFFER | TRANSFER_SRC | TRANSFER_DST)
  ├─ vkGetBufferMemoryRequirements()
  ├─ 查找 HOST_VISIBLE | HOST_COHERENT 内存类型
  ├─ vkAllocateMemory()
  ├─ vkBindBufferMemory()
  └─ vkMapMemory() → buf.mapped (持久映射)
```

### 数据搬运

```
Host → Device:
  copy_to_device(buf, data, size)
    ├─ memcpy(buf.mapped, data, size)     // 写入映射内存
    └─ vkFlushMappedMemoryRanges()        // 确保 GPU 可见

Device → Host:
  copy_from_device(buf, data, size)
    ├─ vkInvalidateMappedMemoryRanges()   // 使 CPU 缓存失效
    └─ memcpy(data, buf.mapped, size)     // 读取
```

---

## 12. 算子实现标准模板

以 `eltwise_vulkan` 为例展示完整的算子入口实现：

```cpp
// 文件: src/backend/vulkan/<op>_vulkan.cpp
#include "vulkan_common.hpp"
#include "<op>_f32_spv.h"
#include "<op>_f16_spv.h"
#include "nnops/ops/<op>.hpp"

namespace nnops::backend::vulkan {

// 枚举 → 特化常量映射
static constexpr uint32_t eltwise_op_to_spec(EltwiseType type) {
    switch (type) {
    case EltwiseType::Add: return 0;
    case EltwiseType::Sub: return 1;
    // ...
    }
    return 0;
}

void eltwise_vulkan(
    const EltwiseAttributes& attrs,
    TensorView& /*output*/,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx,
    void* /*workspace*/)
{
    // ① 获取总元素数并空检查
    const int64_t total = inputs[0].numel();
    if (total == 0) return;

    // ② 根据 dtype 选择 SPIR-V blob
    const uint32_t* spirv_data = nullptr;
    size_t spirv_size = 0;
    switch (inputs[0].data_type()) {
    case DataType::f32: spirv_data = g_eltwise_f32_spv; spirv_size = g_eltwise_f32_spv_len; break;
    case DataType::f16: spirv_data = g_eltwise_f16_spv; spirv_size = g_eltwise_f16_spv_len; break;
    default: NNOPS_ASSERT(!"unsupported dtype"); return;
    }

    // ③ 从 ComputeContext 提取 Vulkan 资源
    VkCommandBuffer  cmd    = static_cast<VkCommandBuffer>(ctx.vulkan_cmd_buffer);
    VkDevice         device = static_cast<VkDevice>(ctx.vulkan_device);
    VkDescriptorPool pool   = static_cast<VkDescriptorPool>(ctx.vulkan_descriptor_pool);
    const VkBuffer*  bufs   = static_cast<const VkBuffer*>(ctx.vulkan_buffers);

    // ④ 构建 PipelineKey 并获取管线
    PipelineKey key = {};
    key.device       = device;
    key.spirv_data   = spirv_data;
    key.spirv_size   = spirv_size;
    key.spec_op      = eltwise_op_to_spec(attrs.type);
    key.spec_add_to  = attrs.add_to ? 1u : 0u;
    key.num_bindings = 3;  // A, B, output

    VulkanPipeline pipeline = VulkanPipelineCache::instance().get_or_create(key);

    // ⑤ 录制调度命令
    ComputePushConstants pc = { static_cast<int32_t>(total) };
    VkBuffer dispatch_bufs[3] = { bufs[0], bufs[1], bufs[2] };

    vulkan_record_dispatch(cmd, device, pool, pipeline,
                           dispatch_bufs, 3,
                           &pc, sizeof(pc),
                           static_cast<uint32_t>(total));
}

}  // namespace nnops::backend::vulkan
```

### 各步骤中涉及的接口汇总

| 步骤 | 使用的接口/结构体 | 来源 |
|------|-------------------|------|
| ① 空检查 | `TensorView::numel()` | 核心 API |
| ② SPIR-V 选择 | `g_*_spv`, `g_*_spv_len` (constexpr 数组) | 构建生成的 `*_spv.h` |
| ③ 资源提取 | `ComputeContext::vulkan_*` 字段 | compute_context.hpp |
| ④ 管线获取 | `PipelineKey`, `VulkanPipelineCache::get_or_create()` | vulkan_common.hpp |
| ⑤ 录制 | `ComputePushConstants`, `vulkan_record_dispatch()` | vulkan_common.hpp |

---

## 13. 数据流全景图

```
用户代码                       nnops 调度层                  Vulkan 后端
────────                      ────────────                  ──────────

① 创建 VkBuffer + 填充数据
② 创建 VkCommandBuffer
   vkBeginCommandBuffer()
                              ┌──────────────────────┐
③ 构建 ComputeContext:        │                      │
   .vulkan_cmd_buffer = cmd   │                      │
   .vulkan_device     = dev   │                      │
   .vulkan_descriptor_pool    │                      │
   .vulkan_buffers    = bufs  │                      │
                              │                      │
④ op->compute(output, inputs, │  ctx)                 │
                              │  ┌──────────────────┐ │
                              │  │ resolve_kernel() │ │
                              │  │ → vulkan_xxx()   │ │
                              │  └────────┬─────────┘ │
                              │           │           │
                              │  ┌────────▼─────────┐ │
                              │  │ 选择 SPIR-V blob │ │
                              │  │ 提取 Vulkan 资源 │ │
                              │  │ 构建 PipelineKey │ │
                              │  │ get_or_create()  │ │ ← VulkanPipelineCache
                              │  │ 构建 PushConst   │ │
                              │  │ record_dispatch  │ │
                              │  └────────┬─────────┘ │
                              │           │           │
                              │   ┌───────▼─────────┐ │
                              │   │ vkAllocateDescSets  │
                              │   │ vkUpdateDescSets    │
                              │   │ vkCmdBindPipeline   │ ← VkCommandBuffer
                              │   │ vkCmdBindDescSets   │
                              │   │ vkCmdPushConstants  │
                              │   │ vkCmdDispatch       │
                              │   └──────────────────┘ │
                              └─────────────────────────┘

⑤ vkEndCommandBuffer(cmd)
⑥ vkQueueSubmit() + vkQueueWaitIdle()
⑦ vkInvalidateMappedMemoryRanges()
⑧ memcpy() 读回结果验证
⑨ vkResetDescriptorPool()
```

---

## 14. 常量与约定

| 名称 | 值 | 定义位置 | 说明 |
|------|-----|----------|------|
| `kVulkanWorkgroupSize` | `256` | vulkan_common.hpp | 所有着色器的 `local_size_x` |
| `ComputePushConstants` 大小 | `4` bytes | vulkan_common.hpp | 仅包含 `int32_t total` |
| Push Constant 范围 | `offset=0, size=4` | vulkan_common.cpp | `VK_SHADER_STAGE_COMPUTE_BIT` |
| Descriptor Set | `set=0` | 所有 .comp shader | 全部使用同一 set |
| Binding 起始 | `0` | 所有 .comp shader | 连续编号 |
| Binding 类型 | `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER` | vulkan_common.cpp | 全部 SSBO |
| 特化常量 constant_id 0 | `OP` (操作类型) | 所有 .comp shader | Eltwise: Add=0~Pow=6; Unary: Exp=0~Sqrt=8 |
| 特化常量 constant_id 1 | `ADD_TO` (累加标志) | 所有 .comp shader | 0=overwrite, 1=accumulate |
| SPIR-V 嵌入格式 | `constexpr uint32_t[]` | spv_to_header.py | 编译期内嵌 |
| glslc 目标环境 | `vulkan1.2` | CMakeLists.txt | 支持 16-bit storage SPIR-V 能力 |
| PipelineCache 线程安全 | `std::mutex` + 双检查锁定 | vulkan_common.cpp | 读写均安全 |
| 描述符池容量 (测试) | `maxSets=256`, `STORAGE_BUFFER×256` | vulkan_test_helper.hpp | 绰绰有余 |
| Vulkan API 版本 (测试) | `VK_API_VERSION_1_3` | vulkan_test_helper.hpp | Instance 级别 |
