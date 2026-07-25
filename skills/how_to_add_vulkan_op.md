# Vulkan 算子开发指南

## 概述

本文档详细说明如何在 nnops 项目中为算子添加 Vulkan GPU 后端支持。基于 `eltwise` 和 `unary` 算子的开发实践总结而成。

---

## 1. 架构概览

Vulkan 后端遵循与 CUDA 后端相同的三层架构：

```
include/nnops/ops/<op>.hpp          # 算子公共头文件（attributes, class）
src/ops/<op>.cpp                    # 算子调度层（resolve kernel, create, compute）
src/backend/vulkan/<op>_vulkan.cpp  # Vulkan 后端实现
src/backend/vulkan/shaders/<op>_f32.comp  # GLSL 计算着色器
```

关键组件：
- **Pipeline Cache** — 全局单例 `VulkanPipelineCache`，按 `(VkDevice, SPIR-V, 特化常量)` 缓存计算管线
- **Descriptor Set** — 每次 `compute()` 调用从用户提供的 `VkDescriptorPool` 分配，绑定 I/O 缓冲区
- **Push Constants** — 传递 `total` 元素数量等运行时参数
- **Specialization Constants** — 在管线创建时固化操作类型（Add/Sub/...），允许编译器优化分支

---

## 2. 计算着色器 (GLSL)

> 💡 **详细语法参考**：本文仅给出 nnops 项目中的标准模板。完整的 GLSL compute shader 语法（扩展声明、数据类型、shared memory、barrier、subgroup、原子操作等）请见 [Vulkan Compute Shader 语法参考](vulkan_compute_shader_grammar.md)。

### 2.1 基础模板

```glsl
#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// 特化常量 — 在管线创建时确定，编译器会消除死分支
layout(constant_id = 0) const int OP = 0;       // 操作类型
layout(constant_id = 1) const int ADD_TO = 0;   // 是否累加到输出

// 存储缓冲区绑定 — 输入/输出通过 SSBO
layout(set = 0, binding = 0) readonly buffer BufA { float a[]; };
layout(set = 0, binding = 1) readonly buffer BufB { float b[]; };
layout(set = 0, binding = 2) buffer BufC { float c[]; };

// Push Constants — 每次调用可变
layout(push_constant) uniform PushConstants {
    int total;  // 总元素数
} pc;

void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= pc.total) return;

    float result;
    // 分支在管线创建时被特化常量消除
    if (OP == 0) { result = a[gid] + b[gid]; }
    else if (OP == 1) { result = a[gid] - b[gid]; }
    // ...

    if (ADD_TO == 1) { c[gid] += result; }
    else { c[gid] = result; }
}
```

### 2.2 关键规范

| 项目 | 约定 |
|------|------|
| workgroup 大小 | `local_size_x = 256`，保持与 CUDA block size 一致 |
| 绑定点 | 所有缓冲使用 `set=0`，从 binding 0 开始连续编号 |
| 缓冲类型 | 只读输入用 `readonly buffer`，输出用 `buffer` |
| 存储类 | 所有缓冲声明为 SSBO (`std430` 布局，由编译器推导) |
| 特化常量 | constant_id 0 = 操作类型, constant_id 1 = add_to 标志 |
| Push Constants | 用于每调用变化的参数 (total elements) |
| 溢出保护 | 必须在 shader 开头检查 `gid >= pc.total` |

### 2.3 内置数学函数

GLSL 450 支持以下函数，直接映射到 Vulkan GPU 实现：

- `exp(x)`, `log(x)`, `sqrt(x)`
- `sin(x)`, `cos(x)`, `tan(x)`
- `tanh(x)`, `abs(x)`
- 精度：GPU `exp()` 等函数的精度与 CPU 标准库存在微小差异（~1e-4 相对误差），测试时需适当放宽容差

---

## 3. Vulkan 后端实现 (C++)

### 3.1 文件结构

```
src/backend/vulkan/
├── vulkan_common.hpp          # 公共基础设施（PipelineCache, 辅助函数）
├── vulkan_common.cpp          # PipelineCache 实现
├── eltwise_f32_spv.h          # 自动生成的 SPIR-V 嵌入头文件
├── unary_f32_spv.h            # 自动生成的 SPIR-V 嵌入头文件
├── eltwise_vulkan.cpp         # Eltwise 算子 Vulkan 实现
├── unary_vulkan.cpp           # Unary 算子 Vulkan 实现
├── shaders/
│   ├── eltwise_f32.comp       # GLSL 着色器源码
│   └── unary_f32.comp         # GLSL 着色器源码
├── spv_to_header.py           # SPIR-V → C 头文件转换脚本
├── eltwise_f32.spv            # 编译产物 (构建时生成)
└── unary_f32.spv              # 编译产物 (构建时生成)
```

### 3.2 算子实现模板

以 eltwise 为例：

```cpp
#include "vulkan_common.hpp"
#include "eltwise_f32_spv.h"       // 嵌入的 SPIR-V 数组
#include "nnops/ops/eltwise.hpp"

namespace nnops::backend::vulkan {

// 枚举 → 特化常量映射
static constexpr uint32_t eltwise_op_to_spec(EltwiseType type) {
    switch (type) {
    case EltwiseType::Add: return 0;
    case EltwiseType::Sub: return 1;
    case EltwiseType::Mul: return 2;
    case EltwiseType::Div: return 3;
    }
    return 0;
}

void eltwise_vulkan(
    const EltwiseAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx,
    void* /*workspace*/)
{
    const int64_t total = inputs[0].numel();
    if (total == 0) return;

    // 1. 从 ComputeContext 提取 Vulkan 资源
    VkCommandBuffer  cmd    = static_cast<VkCommandBuffer>(ctx.vulkan_cmd_buffer);
    VkDevice         device = static_cast<VkDevice>(ctx.vulkan_device);
    VkDescriptorPool pool   = static_cast<VkDescriptorPool>(ctx.vulkan_descriptor_pool);
    const VkBuffer*  bufs   = static_cast<const VkBuffer*>(ctx.vulkan_buffers);

    // 2. 获取或创建计算管线（缓存自动处理）
    auto& cache = VulkanPipelineCache::instance();
    PipelineKey key = {};
    key.device       = device;
    key.spirv_data   = g_eltwise_f32_spv;
    key.spirv_size   = g_eltwise_f32_spv_len;
    key.spec_op      = eltwise_op_to_spec(attrs.type);
    key.spec_add_to  = attrs.add_to ? 1u : 0u;
    key.num_bindings = 3;  // A, B, output 共 3 个缓冲区

    VulkanPipeline pipeline = cache.get_or_create(key);

    // 3. 记录调度命令到 Command Buffer
    ComputePushConstants pc = {};
    pc.total = static_cast<int32_t>(total);

    VkBuffer dispatch_bufs[3] = { bufs[0], bufs[1], bufs[2] };

    vulkan_record_dispatch(cmd, device, pool, pipeline,
                           dispatch_bufs, 3,
                           &pc, sizeof(pc),
                           static_cast<uint32_t>(total));
}

}  // namespace nnops::backend::vulkan
```

### 3.3 vulkan_record_dispatch 内部流程

```cpp
void vulkan_record_dispatch(...) {
    // Step 1: 从用户提供的 pool 分配描述符集
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    // Step 2: 更新描述符集，指向输入/输出 VkBuffer
    vkUpdateDescriptorSets(device, num_buffers, writes, 0, nullptr);

    // Step 3: 绑定管线和描述符集
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    vkCmdBindDescriptorSets(cmd, ..., 1, &desc_set, ...);

    // Step 4: 推送常量
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    // Step 5: 调度计算
    uint32_t groups = ceil_div(total, 256);
    vkCmdDispatch(cmd, groups, 1, 1);
}
```

---

## 4. VulkanPipelineCache — 管线缓存

### 4.1 缓存键

```cpp
struct PipelineKey {
    VkDevice        device;         // 不同的 device 有独立的缓存
    const uint32_t* spirv_data;     // SPIR-V 二进制数据
    size_t          spirv_size;     // SPIR-V 数据大小
    uint32_t        spec_op;        // 特化常量：操作类型
    uint32_t        spec_add_to;    // 特化常量：add_to 标志
    uint32_t        num_bindings;   // 描述符绑定数量
};
```

### 4.2 生命周期

1. **创建**：首次调用 `get_or_create(key)` 时创建管线（线程安全，双检查锁定）
2. **复用**：相同 key 的后续调用直接返回缓存结果
3. **销毁**：调用 `destroy_device_pipelines(device)` 在设备销毁前清理该设备的所有管线

### 4.3 管线创建流程

```
vkCreateShaderModule
  → vkCreateDescriptorSetLayout (根据 num_bindings)
    → vkCreatePipelineLayout (绑定 set layout + push constant range)
      → vkCreateComputePipelines (应用特化常量)
        → vkDestroyShaderModule (管线创建后可销毁)
```

### 4.4 特化常量应用

```cpp
VkSpecializationMapEntry spec_entries[2] = {
    { 0, 0, sizeof(uint32_t) },  // constant_id=0, offset=0
    { 1, sizeof(uint32_t), sizeof(uint32_t) },  // constant_id=1
};
uint32_t spec_data[2] = { key.spec_op, key.spec_add_to };

VkSpecializationInfo spec_info = {
    .mapEntryCount = 2,
    .pMapEntries   = spec_entries,
    .dataSize      = sizeof(spec_data),
    .pData         = spec_data,
};
```

---

## 5. 内存管理

### 5.1 缓冲区生命周期（用户管理）

算子不负责内存分配。用户负责：
1. 创建 `VkBuffer` + `VkDeviceMemory`（主机可见 + 主机一致）
2. 通过 `ComputeContext::vulkan_buffers` 传递 VkBuffer 句柄
3. 同步：提交 command buffer → 等待完成 → 验证结果
4. 销毁缓冲区

### 5.2 描述符集管理

```
每次 compute() 调用:
  vkAllocateDescriptorSets(pool, 1, &set)   // 从用户 pool 分配
  vkUpdateDescriptorSets(...)               // 指向当前缓冲区
  vkCmdBindDescriptorSets(...)              // 绑定到 command buffer
  
pool 回收策略:
  - 测试中: 每次 submit_and_wait 后 vkResetDescriptorPool
  - 生产中: 用户负责 pool 管理（reset/individual-free）
```

### 5.3 ComputeContext Vulkan 字段

```cpp
struct ComputeContext {
    // ... other fields ...

    void* vulkan_cmd_buffer = nullptr;       // VkCommandBuffer
    void* vulkan_device = nullptr;           // VkDevice
    void* vulkan_descriptor_pool = nullptr;  // VkDescriptorPool
    const void* vulkan_buffers = nullptr;    // VkBuffer 句柄数组
    int vulkan_buffers_count = 0;            // 缓冲区数量
};
```

缓冲区数组布局（按绑定顺序）：
- **Eltwise**: `[A, B, output]` — 3 个缓冲区
- **Unary**: `[input, output]` — 2 个缓冲区

---

## 6. 着色器编译与嵌入

### 6.1 CMake 构建流程

```cmake
# 1. 编译 GLSL → SPIR-V
add_custom_command(
    OUTPUT ${SHADER_SPV}
    COMMAND glslc -o ${SHADER_SPV} ${SHADER_SRC}
    DEPENDS ${SHADER_SRC}
)

# 2. SPIR-V → C 头文件（嵌入为 constexpr uint32_t 数组）
add_custom_command(
    OUTPUT ${SHADER_HDR}
    COMMAND Python3::Interpreter
        spv_to_header.py --name g_eltwise_f32 --spv ${SPV} --out ${HDR}
    DEPENDS ${SHADER_SPV}
)
```

### 6.2 spv_to_header.py

将 SPIR-V 二进制（每 4 字节 little-endian word）转换为 C++ constexpr 数组：

```python
def spv_to_c(name, spv_path, h_path):
    with open(spv_path, 'rb') as f:
        data = f.read()
    words = []
    for i in range(0, len(data), 4):
        w = int.from_bytes(data[i:i+4], 'little')
        words.append(str(w))
    # 输出: constexpr uint32_t g_eltwise_f32_spv[] = { ... };
    #       constexpr size_t g_eltwise_f32_spv_len = sizeof(...);
```

### 6.3 手动编译

```bash
glslc -o eltwise_f32.spv eltwise_f32.comp
python spv_to_header.py --name g_eltwise_f32 --spv eltwise_f32.spv --out eltwise_f32_spv.h
spirv-val eltwise_f32.spv  # 验证 SPIR-V 有效性
```

---

## 7. 调度层集成

### 7.1 声明并接线 Vulkan 内核

在 `src/ops/<op>.cpp` 中：

```cpp
// 条件编译，声明 Vulkan 内核函数
#ifdef NNOPS_HAS_VULKAN
namespace backend::vulkan {
    void eltwise_vulkan(const EltwiseAttributes& attrs,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         void* workspace);
}
#endif

// 在 kernel resolver 中接线
auto resolve_eltwise_kernel(Backend backend) -> Eltwise::Impl::KernelFn {
    switch (backend) {
    case Backend::CPU:    return backend::cpu::eltwise_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:   return backend::cuda::eltwise_cuda;
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan: return backend::vulkan::eltwise_vulkan;
#endif
    }
    return nullptr;
}
```

---

## 8. 测试框架

### 8.1 测试辅助类

`tests/common/vulkan_test_helper.hpp` 提供 `VulkanTestEnv`：

```cpp
class VulkanTestEnv {
    // 自动创建: Instance, Device, Compute Queue, Command Pool, Descriptor Pool
    // 自动销毁（含管线缓存清理）：析构时按正确顺序销毁所有资源
    
    Buffer create_buffer(size_t, VkBufferUsageFlags);  // 创建+映射设备缓冲
    void copy_to_device(Buffer&, const void*, size_t); // 主机→设备
    void copy_from_device(Buffer&, void*, size_t);     // 设备→主机
    VkCommandBuffer begin_cmd();                       // 开始录制
    void submit_and_wait();                            // 提交+等待+释放+重置pool
};
```

### 8.2 测试模式

```cpp
NNOPS_TEST(eltwise_vulkan_add_1d) {
    const int N = 256;
    float a[N], b[N];
    // ... 填充输入数据 ...

    // 1. 创建 Vulkan 环境
    test::VulkanTestEnv env;

    // 2. 创建设备缓冲区
    auto buf_a = env.create_buffer(N * sizeof(float), ...);
    auto buf_b = env.create_buffer(N * sizeof(float), ...);
    auto buf_o = env.create_buffer(N * sizeof(float), ...);

    // 3. 复制数据到设备
    env.copy_to_device(buf_a, a, N * sizeof(float));
    env.copy_to_device(buf_b, b, N * sizeof(float));

    // 4. 构建 ComputeContext
    ComputeContext ctx;
    ctx.expected_backend = Backend::Vulkan;
    ctx.vulkan_device = env.device();
    ctx.vulkan_descriptor_pool = env.descriptor_pool();
    ctx.vulkan_buffers = /* VkBuffer 句柄数组 */;
    ctx.vulkan_buffers_count = 3;

    // 5. 开始录制
    VkCommandBuffer cmd = env.begin_cmd();
    ctx.vulkan_cmd_buffer = cmd;

    // 6. 执行算子
    auto op = Eltwise::create(attrs, Backend::Vulkan);
    op->compute(output_view, input_views, ctx);

    // 7. 提交并等待
    env.submit_and_wait();

    // 8. 读回结果并验证
    std::vector<float> result(N);
    env.copy_from_device(buf_o, result.data(), N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], expected[i], 1e-4f);
    }

    // 9. 清理缓冲区（VulkanTestEnv 析构自动清理设备/实例）
    env.destroy_buffer(buf_a);
    env.destroy_buffer(buf_b);
    env.destroy_buffer(buf_o);
}
```

---

## 9. 添加新算子的完整检查清单

1. **编写 GLSL 计算着色器**
   - [ ] `src/backend/vulkan/shaders/<op>_f32.comp`
   - [ ] 使用特化常量处理操作类型变体
   - [ ] 使用 push constants 传递运行时参数
   - [ ] workgroup 大小 = 256
   - [ ] 检查 `gid >= total` 防止越界

2. **编译着色器并生成嵌入头文件**
   - [ ] `glslc -o <op>_f32.spv <op>_f32.comp`
   - [ ] `python spv_to_header.py --name g_<op>_f32 ...`
   - [ ] `spirv-val <op>_f32.spv` 验证

3. **实现 Vulkan 后端 C++ 代码**
   - [ ] `src/backend/vulkan/<op>_vulkan.cpp`
   - [ ] 包含 `vulkan_common.hpp` 和 `*_f32_spv.h`
   - [ ] 枚举 → 特化常量映射函数
   - [ ] 从 `ComputeContext` 提取 Vulkan 资源
   - [ ] 调用 `VulkanPipelineCache::get_or_create()` 获取管线
   - [ ] 调用 `vulkan_record_dispatch()` 记录命令

4. **更新调度层**
   - [ ] 在 `src/ops/<op>.cpp` 中添加 Vulkan 前向声明（`#ifdef NNOPS_HAS_VULKAN`）
   - [ ] 在 `resolve_*_kernel()` 中添加 `case Backend::Vulkan`

5. **更新 CMake 构建**
   - [ ] 在 `src/CMakeLists.txt` 的 `VULKAN_SHADERS` 列表中添加着色器名
   - [ ] 在 `NNOPS_VULKAN_SOURCES` 列表中添加新的 `.cpp` 文件

6. **编写测试**
   - [ ] CPU 测试: `tests/test_<op>.cpp`（基础正确性）
   - [ ] Vulkan 测试: `tests/test_<op>_vulkan.cpp`
   - [ ] Vulkan 测试中使用 `VulkanTestEnv` 管理设备
   - [ ] GPU 精度容差使用 `1e-4f` 相对误差（`exp` 等函数可能需要 `2e-4f`）
   - [ ] 更新 `tests/CMakeLists.txt` 添加新测试源文件

7. **构建和测试**
   - [ ] `cmake .. -DNNOPS_BUILD_VULKAN=ON -DNNOPS_BUILD_TESTS=ON`
   - [ ] `cmake --build . --config Release`
   - [ ] `./tests/Release/nnops_test.exe` 确保所有测试通过

---

## 10. 常见问题

### Q: 算子调用失败，VkResult = -3 (VK_ERROR_INITIALIZATION_FAILED)
- 检查 SPIR-V 有效性：`spirv-val <shader>.spv`
- 检查特化常量 constant_id 与代码一致
- 检查描述符绑定数量与着色器中声明一致

### Q: 算子调用失败，VkResult = -4 (VK_ERROR_DEVICE_LOST)
- 通常是 GPU 资源耗尽或 pipeline cache 未清理导致
- 确保在销毁 VkDevice 前调用 `VulkanPipelineCache::instance().destroy_device_pipelines(device)`
- 确保 VkDescriptorPool 有足够容量（`maxSets` 和 descriptor count）

### Q: GPU 结果与 CPU 结果有微小差异
- GPU 数学函数使用不同的逼近算法，相对误差在 1e-4 级别是正常的
- 测试中使用相对容差：`tol = 1e-5f + 2e-4f * abs(expected)`

### Q: 程序退出时崩溃 (Segfault)
- 确保 `VulkanTestEnv` 析构函数在 `vkDestroyDevice` 之前调用了 pipeline cache cleanup
- 确保 VkDescriptorSet 不在 command buffer 还在录制时被释放

---

## 11. fp16 (半精度浮点) 支持

### 11.1 设计原则

遵循与 CPU/CUDA 后端一致的 **"f16 加载 → f32 计算 → f16 存储"** 模式：

```text
GPU Shader 内部流程:
  float16_t v = buf[gid];       // 1. 从 SSBO 以 fp16 加载
  float v32  = float(v);        // 2. 隐式扩展为 fp32（零成本）
  float r    = v32 + ...;       // 3. fp32 计算（exp, log, +, -, *, /...）
  buf[gid]   = float16_t(r);    // 4. 窄化为 fp16 存储
```

只需要 **VK_KHR_16bit_storage**（`storageBuffer16BitAccess`），不需要 `shaderFloat16`。因为所有计算都在 fp32 执行，GPU 不需要支持 fp16 原生运算单元。绝大多数 Vulkan 1.1+ GPU 都满足此条件。

### 11.2 fp16 Shader 模板

```glsl
#version 450
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_16bit_storage : require    // ← 启用 float16_t 在 SSBO 中

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const int OP = 0;
layout(constant_id = 1) const int ADD_TO = 0;

// SSBO 类型声明为 float16_t 而非 float
layout(set = 0, binding = 0) readonly buffer BufA { float16_t a[]; };
layout(set = 0, binding = 1) readonly buffer BufB { float16_t b[]; };
layout(set = 0, binding = 2) buffer BufC { float16_t c[]; };

layout(push_constant) uniform PushConstants {
    int total;
} pc;

void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= pc.total) return;

    // 加载为 fp16，隐式扩展为 fp32
    float va = float(a[gid]);
    float vb = float(b[gid]);
    float result;

    if (OP == 0)      result = va + vb;
    else if (OP == 1) result = va - vb;
    else if (OP == 2) result = va * vb;
    else              result = va / vb;

    // 写入时窄化回 fp16
    if (ADD_TO == 1) {
        c[gid] = float16_t(float(c[gid]) + result);
    } else {
        c[gid] = float16_t(result);
    }
}
```

**关键 GLSL 语法**：

| 元素 | f32 shader | f16 shader |
|------|------------|------------|
| 扩展声明 | 无 | `#extension GL_EXT_shader_16bit_storage : require` |
| SSBO 类型 | `float a[]` | `float16_t a[]` |
| 加载 | `float va = a[gid]` | `float va = float(a[gid])` (隐式) |
| 计算 | `float` | `float` (同 f32) |
| 存储 | `c[gid] = result` | `c[gid] = float16_t(result)` |
| `add_to` 读取 | `c[gid] += result` | `float(c[gid]) + result` → `float16_t(...)` |

`float16_t` ↔ `float` 转换在 GLSL 中是隐式的，无需显式 `float()` / `float16_t()` 转换——但显式写出有助于代码可读性。

### 11.3 C++ 后端 dtype 分发

在算子入口点添加 dtype 判断，选择对应的 SPIR-V blob：

```cpp
#include "my_op_f32_spv.h"
#include "my_op_f16_spv.h"

void my_op_vulkan(...) {
    const auto dtype = inputs[0].data_type();

    // 根据 dtype 选择 SPIR-V
    const uint32_t* spirv_data = nullptr;
    size_t spirv_size = 0;
    switch (dtype) {
    case DataType::f32:
        spirv_data = g_my_op_f32_spv;
        spirv_size = g_my_op_f32_spv_len;
        break;
    case DataType::f16:
        spirv_data = g_my_op_f16_spv;
        spirv_size = g_my_op_f16_spv_len;
        break;
    default:
        NNOPS_ASSERT(!"unsupported data type");
        return;
    }

    // PipelineKey 无需改动——spirv_data 指针天然区分 f32/f16
    PipelineKey key = {};
    key.spirv_data = spirv_data;
    key.spirv_size = spirv_size;
    // ...
}
```

**不需要修改**：
- `PipelineKey` — `spirv_data` 指针已能区分 f32 和 f16
- `VulkanPipelineCache` — 缓存逻辑对 dtype 无感知
- `vulkan_record_dispatch` — descriptor set 分配与 buffer 类型无关
- `ComputeContext` — `VkBuffer` 句柄与元素类型无关

### 11.4 Device 侧要求

测试环境的 device 创建需启用 `VK_KHR_16bit_storage` 和 `storageBuffer16BitAccess` 特性：

```cpp
// 1. 查询物理设备是否支持该扩展
uint32_t ext_count = 0;
vkEnumerateDeviceExtensionProperties(phys_device, nullptr, &ext_count, nullptr);
std::vector<VkExtensionProperties> exts(ext_count);
vkEnumerateDeviceExtensionProperties(phys_device, nullptr, &ext_count, exts.data());
bool supports_16bit = false;
for (auto& ext : exts) {
    if (strcmp(ext.extensionName, VK_KHR_16BIT_STORAGE_EXTENSION_NAME) == 0) {
        supports_16bit = true;
        break;
    }
}

// 2. 条件启用扩展和特性
VkPhysicalDevice16BitStorageFeatures storage_features = {};
storage_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
storage_features.storageBuffer16BitAccess = VK_TRUE;

VkDeviceCreateInfo dci = {};
// ... queue info ...
dci.pNext = supports_16bit ? &storage_features : nullptr;

const char* enabled_exts[] = { VK_KHR_16BIT_STORAGE_EXTENSION_NAME };
dci.enabledExtensionCount = supports_16bit ? 1 : 0;
dci.ppEnabledExtensionNames = supports_16bit ? enabled_exts : nullptr;
```

### 11.5 CMake 配置

在 `src/CMakeLists.txt` 中：

```cmake
# glslc 需要 --target-env=vulkan1.2 以支持 16bit storage SPIR-V 特性
COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.2 -o ${SHADER_SPV} ${SHADER_SRC}

# 将 fp16 shader 名称加入列表
set(VULKAN_SHADERS eltwise_f32 unary_f32 eltwise_f16 unary_f16)
```

### 11.6 fp16 测试要点

1. **缓冲区大小**：`numel * sizeof(uint16_t)` = `numel * 2`（不是 `sizeof(float)`）

2. **TensorView dtype**：`DataType::f16`

3. **参考值计算**：CPU 参考值也必须走 fp16 往返（float→f16→float），确保对比双方都在 fp16 精度下

4. **精度容差**：
   - 直接算术 (Add/Sub/Mul/Div/Abs/Neg): `1e-3f`
   - 超越函数 (Exp/Log/Sin/Cos/Tan/Sqrt): `1e-3f + 5e-3f * |expected|`
   - fp16 仅有 ~3.3 位十进制有效数字，比 f32 宽松约 100 倍

5. **输入范围约束**：fp16 最大值为 65504，最小正规数为 ~6.1e-5。尤其是 `exp` 测试要控制输入在 [-10, 10] 范围内

6. **fp16 转换工具**：复用 `nnops/detail/half.hpp` 中的 `float_to_half()` / `half_to_float()`

```cpp
#include "nnops/detail/half.hpp"
using nnops::backend::cpu::half;
using nnops::backend::cpu::half_to_float;
using nnops::backend::cpu::float_to_half;

// float 数组 → fp16 数组
std::vector<uint16_t> f16_bits(n);
for (size_t i = 0; i < n; ++i)
    f16_bits[i] = float_to_half(float_data[i]).bits;

// 创建 DataType::f16 的 TensorView
TensorView view(shape, DataType::f16, f16_bits.data());
```

### 11.7 检查清单：添加 fp16 支持

1. **编写 `_f16.comp` shader**
   - [ ] 添加 `#extension GL_EXT_shader_16bit_storage : require`
   - [ ] SSBO 类型改为 `float16_t[]`
   - [ ] 加载时隐式转换 `float va = float(a[gid])`
   - [ ] 存储时显式转换 `c[gid] = float16_t(result)`
   - [ ] `add_to` 路径先将 `float16_t` 转为 `float` 再运算

2. **更新 CMake**
   - [ ] 将 shader 名称加入 `VULKAN_SHADERS` 列表
   - [ ] 确保 glslc 使用 `--target-env=vulkan1.2`

3. **更新 C++ 后端**
   - [ ] `#include "xxx_f16_spv.h"`
   - [ ] 在入口点添加 `switch (dtype)` 选择 SPIR-V

4. **启用设备特性**
   - [ ] 查询 `VK_KHR_16bit_storage` 扩展
   - [ ] 链入 `VkPhysicalDevice16BitStorageFeatures`

5. **编写测试**
   - [ ] 缓冲区大小使用 `sizeof(uint16_t) * numel`
   - [ ] TensorView 使用 `DataType::f16`
   - [ ] 参考值走 fp16 往返
   - [ ] 使用 fp16 级别容差
