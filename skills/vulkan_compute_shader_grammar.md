# Vulkan Compute Shader (GLSL) 语法参考

> 基于 nnops 项目 Vulkan 后端开发实践，覆盖从基础到高级的完整 GLSL 计算着色器语法。

---

## 目录

1. [版本与扩展声明](#1-版本与扩展声明)
2. [执行模型：Workgroup 与 Invocation](#2-执行模型workgroup-与-invocation)
3. [数据绑定：SSBO / UBO / Push Constant](#3-数据绑定ssbo--ubo--push-constant)
4. [Specialization Constant（特化常量）](#4-specialization-constant特化常量)
5. [数据类型与精度](#5-数据类型与精度)
6. [内存修饰符与 Barrier 同步](#6-内存修饰符与-barrier-同步)
7. [共享内存 (Shared Memory)](#7-共享内存-shared-memory)
8. [Subgroup 操作](#8-subgroup-操作)
9. [原子操作 (Atomic Operations)](#9-原子操作-atomic-operations)
10. [内置数学函数参考](#10-内置数学函数参考)
11. [纹理与图像操作](#11-纹理与图像操作)
12. [Warp Divergence 与性能考量](#12-warp-divergence-与性能考量)
13. [编译工具链：glslc → SPIR-V](#13-编译工具链glslc--spir-v)
14. [Vulkan 1.4 / Roadmap 2026 新特性](#14-vulkan-14--roadmap-2026-新特性)

---

## 1. 版本与扩展声明

### 1.1 版本

```glsl
#version 450
```

Vulkan 1.0 使用 `#version 450`（对应 GLSL 4.50 / OpenGL 4.5 核心子集）。Vulkan 不支持 `#version 460`+ 的所有特性，但可通过扩展获得等效功能。

| `#version` | 对应 SPIR-V | Vulkan 版本 | 关键新增 |
|------------|-------------|-------------|----------|
| `450` | 1.0 | Vulkan 1.0 | 基础功能 |
| `450` + ext | 1.3 | Vulkan 1.1 | 16-bit storage, subgroups |
| `460` | 1.5 | Vulkan 1.2 | 完整 16-bit, GL_KHR_memory_scope_semantics |

### 1.2 扩展声明

GLSL 扩展在 Vulkan 中通过 `#extension` 启用，对应底层的 SPIR-V 能力和 Vulkan Device Extension：

```glsl
// ---- include 指令（支持 #include 跨文件共享代码） ----
#extension GL_GOOGLE_include_directive : enable

// ---- 16-bit 存储（fp16 / int16 在 SSBO/UBO/Push Constant 中） ----
// 需要 VK_KHR_16bit_storage + storageBuffer16BitAccess
#extension GL_EXT_shader_16bit_storage : require

// ---- fp16 原生算术（float16_t 全功能支持） ----
// 需要 VK_KHR_shader_float16_int8 + shaderFloat16
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

// ---- 8-bit / 16-bit 整数算术 ----
// 需要 VK_KHR_shader_float16_int8 + shaderInt8 / shaderInt16
#extension GL_EXT_shader_explicit_arithmetic_types_int8  : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

// ---- Subgroup 操作（warp 级别 shuffle/vote/ballot） ----
// Vulkan 1.1 core — 多数 GPU 原生支持
#extension GL_KHR_shader_subgroup_basic          : require
#extension GL_KHR_shader_subgroup_vote           : require
#extension GL_KHR_shader_subgroup_arithmetic     : require
#extension GL_KHR_shader_subgroup_ballot         : require
#extension GL_KHR_shader_subgroup_shuffle        : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require
#extension GL_KHR_shader_subgroup_clustered      : require

// ---- 显式算术类型（int64/float64） ----
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : enable

// ---- 非均匀控制流内的资源访问（如 divergent branch 中 read texture） ----
#extension GL_EXT_nonuniform_qualifier : require

// ---- 8-bit / 16-bit 整数存储（在 SSBO 中） ----
#extension GL_EXT_shader_8bit_storage  : require  // 需要 VK_KHR_8bit_storage
#extension GL_EXT_shader_16bit_storage : require  // 需要 VK_KHR_16bit_storage
```

**require vs enable**：
- `require`：如果扩展不可用，编译失败（推荐，尽早发现问题）
- `enable`：如果扩展不可用，静默忽略（运行时可能出错，更隐蔽）
- `warn`：如果扩展不可用，编译警告

### 1.3 扩展与 Vulkan Device Feature 的对应关系

```text
GLSL Extension                             → Vulkan Device Capability
───────────────────────────────────────────────────────────────────────
GL_EXT_shader_16bit_storage                → VK_KHR_16bit_storage.storageBuffer16BitAccess
GL_EXT_shader_16bit_storage                → SPIR-V: StorageBuffer16BitAccess
GL_EXT_shader_8bit_storage                 → VK_KHR_8bit_storage.storageBuffer8BitAccess
GL_EXT_shader_explicit_arithmetic_types_float16 → VK_KHR_shader_float16_int8.shaderFloat16
GL_EXT_shader_explicit_arithmetic_types_int8   → VK_KHR_shader_float16_int8.shaderInt8
GL_KHR_shader_subgroup_basic               → Vulkan 1.1 core (subgroupSize 查询)
GL_EXT_nonuniform_qualifier                → VkPhysicalDeviceDescriptorIndexingFeatures
```

---

## 2. 执行模型：Workgroup 与 Invocation

### 2.1 Workgroup 声明

```glsl
// 1D 逐元素 — nnops 项目标准
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// 2D 图像处理（如卷积 / 池化）
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// 3D 体素
layout(local_size_x = 8, local_size_y = 8, local_size_z = 4) in;
```

**约束**：
- `local_size_x * local_size_y * local_size_z ≤ 128`（Vulkan 保证值；多数 GPU 支持到 1024）
- `local_size_x` 个别维度上限因 GPU 而异，一般为 1024
- **最佳实践**：选择 32 和 64 的公倍数（256, 512, 1024），以填满 warp/wavefront 调度单元

### 2.2 内置变量

dispatch: `vkCmdDispatch(cmd, Gx, Gy, Gz)`

| 变量 | 类型 | 含义 | 示例 (Gx=4, local_x=256) |
|------|------|------|---------------------------|
| `gl_GlobalInvocationID` | `uvec3` | 全局线程索引（跨所有 workgroup） | `x ∈ [0, 1023]` |
| `gl_LocalInvocationID` | `uvec3` | 当前 workgroup 内线程索引 | `x ∈ [0, 255]` |
| `gl_WorkGroupID` | `uvec3` | 当前 workgroup 在 dispatch 中的位置 | `x ∈ [0, 3]` |
| `gl_NumWorkGroups` | `uvec3` | `vkCmdDispatch` 传入的总 workgroup 数 | `(4, 1, 1)` |
| `gl_WorkGroupSize` | `uvec3` | `local_size_*` 常量的运行时副本 | `(256, 1, 1)` |

**关系**：`gl_GlobalInvocationID = gl_WorkGroupID * gl_WorkGroupSize + gl_LocalInvocationID`

### 2.3 1D 逐元素标准模式（nnops 项目全部使用）

```glsl
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    // ① 全局索引
    int gid = int(gl_GlobalInvocationID.x);

    // ② 边界守卫 — 绝对不可省略！
    if (gid >= pc.total) return;

    // ③ 处理 a[gid]
    float x = a[gid];
    c[gid] = x * 2.0f;
}
```

**为什么边界守卫必须存在**：
- `total` 不一定是 256 的倍数，最后一组 workgroup 有部分线程越界
- GPU 无 MMU 页保护，越界访问 SSBO = 未定义行为（可能读到垃圾数据、可能触发 GPU page fault/TDR）

---

## 3. 数据绑定：SSBO / UBO / Push Constant

### 3.1 SSBO (Shader Storage Buffer Object)

**项目首选**——全部 buffer 绑定使用 SSBO：

```glsl
// ---- 语法格式 ----
layout(set = 0, binding = 0) readonly  buffer Name { float arr[]; };
layout(set = 0, binding = 1) writeonly buffer Name { float arr[]; };
layout(set = 0, binding = 2) buffer Name { float arr[]; };  // 可读写

// ---- 隐式 std430 布局（SSBO 默认） ----
// 可以声明 struct 成员，编译器自动按 std430 对齐：
//   float          → 4 bytes, 4-byte aligned
//   vec2           → 8 bytes, 8-byte aligned
//   vec3           → 12 bytes, 16-byte aligned  (尾对齐到 16!)
//   vec4           → 16 bytes, 16-byte aligned
//   struct { ... } → 按最大成员对齐
//   数组元素       → 紧密排列，无尾填充（与 UBO std140 不同！）

// ---- 可选：显式 std430 声明 ----
layout(std430, set = 0, binding = 0) buffer MyBuf {
    int   count;
    vec4  positions[];
};
```

**`set` 与 `binding` 的关系**：

```
set=0, binding=0 → VkDescriptorSetLayoutBinding{ binding=0, descriptorType=STORAGE_BUFFER }
set=0, binding=1 → VkDescriptorSetLayoutBinding{ binding=1, descriptorType=STORAGE_BUFFER }
set=1, binding=0 → 另一个 descriptor set layout 的 binding 0
```

nnops 项目全部使用 `set=0`，binding 从 0 开始连续编号（最多支持 3 个 binding：A, B, output）。

### 3.2 UBO (Uniform Buffer Object)

适合小型（≤64KB）、频繁读取的参数块：

```glsl
// std140 布局 — 严格对齐规则
layout(std140, set = 0, binding = 0) uniform Params {
    mat4  mvp;        // offset 0, 64 bytes (4×vec4)
    vec4  color;      // offset 64, 16 bytes
    float alpha;      // offset 80, 4 bytes
    int   flags;      // offset 84, 4 bytes
    // 总大小：88 bytes（自动填充到 16 字节边界 = 96 bytes）
} params;
```

**UBO vs SSBO**：

| 特性 | UBO (Uniform Buffer) | SSBO (Storage Buffer) |
|------|---------------------|----------------------|
| 最大大小 | 保证 64KB | 无限制 (GPU 内存上限) |
| 写入 | 不支持 | 支持 (`readonly`/`writeonly`/可读写) |
| 布局 | std140（严格对齐，有填充） | std430（紧凑对齐） |
| 访问速度 | 通常更快（常量缓存路径） | 稍慢（L2 cache 路径） |
| 运行时大小数组 | 不支持 | 支持 `float arr[]` |
| nnops 使用 | 否 | 是（所有 buffer） |

### 3.3 Push Constants

**最快路径**——数据内联在 Command Buffer 中，无需 descriptor：

```glsl
// ---- Shader 侧 ----
layout(push_constant) uniform PushConstants {
    int    total;         // offset 0
    float  alpha;         // offset 4
    int    flags;         // offset 8
} pc;

// ---- C++ 侧 (vulkan_common.cpp) ----
ComputePushConstants pc = { total };
vkCmdPushConstants(cmd, pipeline_layout,
                   VK_SHADER_STAGE_COMPUTE_BIT,
                   0,                 // offset
                   sizeof(pc),        // size
                   &pc);              // data
```

**约束**：
- 最大大小：Vulkan 保证 ≥ **128 bytes**（多数 GPU 支持 256 bytes）
- 布局：与 C struct 相同的自然对齐
- nnops 项目当前只用 4 bytes（`int total`），远未触及上限

---

## 4. Specialization Constant（特化常量）

### 4.1 语法

```glsl
layout(constant_id = 0) const int  OP      = 0;   // 默认值 = 0
layout(constant_id = 1) const int  ADD_TO  = 0;
layout(constant_id = 2) const bool USE_BIAS = false;
layout(constant_id = 3) const float ALPHA   = 1.0f;
```

- `constant_id`：0-based 整数，在一个 shader 内唯一
- 类型：仅标量——`bool`、`int`、`uint`、`float`、`double`
- 默认值：C++ 侧不提供值时使用
- 支持数组（`const float coeffs[4]`）

### 4.2 C++ 侧注入

```cpp
// 声明映射条目
VkSpecializationMapEntry entries[2] = {
    { 0, 0,                  sizeof(uint32_t) },  // constant_id=0, offset=0
    { 1, sizeof(uint32_t),   sizeof(uint32_t) },  // constant_id=1, offset=4
};

// 打包数据
uint32_t data[2] = { 0, 0 };  // OP=Add, ADD_TO=overwrite

VkSpecializationInfo spec_info = {};
spec_info.mapEntryCount = 2;
spec_info.pMapEntries   = entries;
spec_info.dataSize      = sizeof(data);
spec_info.pData         = data;

// 挂到 VkPipelineShaderStageCreateInfo
VkPipelineShaderStageCreateInfo stage_ci = {};
stage_ci.pSpecializationInfo = &spec_info;
```

### 4.3 编译器行为

Pipeline 创建时，驱动编译器将特化常量替换为实际值，然后执行：
1. **常量传播**：`if (OP == 0)` → `if (true)` → 消除死分支
2. **死代码消除**：移除不可达分支的指令
3. **循环展开启发式**：已知常量边界的循环可能被展开

效果：**零运行时分支开销**。nnops 项目中一个 shader 覆盖所有算子变体（如 eltwise 的 Add/Sub/Mul/Div + add_to 标志 = 8 个 pipeline）。

---

## 5. 数据类型与精度

### 5.1 标量类型

| GLSL 类型 | 位数 | nnops 对应 | 说明 |
|-----------|------|------------|------|
| `float` | 32 | `DataType::f32` | 标准单精度 |
| `double` | 64 | — | 需要 `GL_EXT_shader_explicit_arithmetic_types_float64` |
| `float16_t` | 16 | `DataType::f16` | 需要 `GL_EXT_shader_16bit_storage`（存储）或 `GL_EXT_shader_explicit_arithmetic_types_float16`（算术） |
| `int` | 32 | `DataType::s32` | 有符号 32-bit |
| `uint` | 32 | — | 无符号 32-bit |
| `int16_t` | 16 | — | 有符号 16-bit，需要 `GL_EXT_shader_explicit_arithmetic_types_int16` |
| `uint16_t` | 16 | `DataType::f16` (bits) | 无符号 16-bit（fp16 的比特表示） |
| `int8_t` | 8 | `DataType::s8` | 需要 `GL_EXT_shader_explicit_arithmetic_types_int8` |
| `uint8_t` | 8 | `DataType::u8` | 同上 |
| `int64_t` | 64 | `DataType::s64` | 需要 `GL_EXT_shader_explicit_arithmetic_types_int64` |
| `bool` | ~ | — | 条件值 |

### 5.2 向量类型

```glsl
// ---- f32 向量（nnops 常用） ----
vec2  v2;   // 2 分量
vec3  v3;   // 3 分量
vec4  v4;   // 4 分量 — SIMD 友好，一次处理 4 个元素

// ---- 其他精度 ----
f16vec4  hv4;    // float16_t 向量 (4×16=64 bits)
i8vec4   i8v4;   // int8 向量
u16vec2  u16v2;  // uint16 向量
i64vec2  i64v2;  // int64 向量

// ---- 矩阵 ----
mat4  m4;   // 4×4 float 矩阵
f16mat4x4 hm;  // 4×4 float16 矩阵

// ---- 构造函数 ----
vec4 v = vec4(1.0, 2.0, 3.0, 4.0);
vec4 s = vec4(1.0);              // 所有分量 = 1.0
vec3 t = vec3(v);                // 截取前 3 个分量
```

### 5.3 Swizzle（分量访问）

```glsl
vec4 v = vec4(1.0, 2.0, 3.0, 4.0);
float x = v.x;      // 1.0
vec2  xy = v.xy;    // (1.0, 2.0)
vec3  rgb = v.rgb;  // (1.0, 2.0, 3.0)
vec4  rearr = v.wzyx; // (4.0, 3.0, 2.0, 1.0)  — 任意重排
vec3  repeat = v.xxx; // (1.0, 1.0, 1.0)       — 重复

// 命名集合：rgba, xyzw, stpq
```

### 5.4 精度修饰符（Fragment Shader 中常用，Compute 中少见）

```glsl
highp   float hf;    // 高精度（默认，Compute Shader 中无影响）
mediump float mf;    // 中精度
lowp    float lf;    // 低精度 — 移动 GPU 可能用 fp16 实现
```

Compute Shader 中精度修饰符通常被忽略——GLSL 会发出警告但按 `highp` 处理。

### 5.5 fp16 使用场景对比

```glsl
// ---- 场景 A：只需存储（nnops 当前方案） ----
// 要求：VK_KHR_16bit_storage
// 扩展：GL_EXT_shader_16bit_storage
layout(set=0, binding=0) buffer Buf { float16_t arr[]; };
float x = float(arr[gid]);  // f16→f32 加载
// ... f32 计算 ...
arr[gid] = float16_t(result); // f32→f16 存储

// ---- 场景 B：原生 fp16 算术（少用） ----
// 要求：VK_KHR_shader_float16_int8.shaderFloat16
// 扩展：GL_EXT_shader_explicit_arithmetic_types_float16
float16_t a = arr[gid];       // f16 加载
float16_t b = float16_t(2.0hf);
float16_t c = a * b + sin(a); // f16 原生运算（低功耗/加倍吞吐）
arr[gid] = c;                 // f16 存储
```

nnops 选择场景 A，因为：
1. 最大化 GPU 兼容性（大多数 Vulkan 1.1+ GPU 支持 16-bit storage）
2. f32 乘积累加器避免 fp16 精度损失
3. 与 CPU/CUDA 后端行为一致

---

## 6. 内存修饰符与 Barrier 同步

### 6.1 Memory Access Qualifiers（内存访问修饰符）

```glsl
// ---- 只读访问（GPU 可做 cache 优化） ----
layout(set=0, binding=0) readonly buffer InputBuf { float arr[]; };

// ---- 只写访问（写合并优化） ----
layout(set=0, binding=1) writeonly buffer OutputBuf { float arr[]; };

// ---- 默认：可读写 ----
layout(set=0, binding=2) buffer RWBuf { float arr[]; };

// ---- coherent: 跨 workgroup 写入对其他 workgroup 可见 ----
// 必须配合 barrier 使用
layout(set=0, binding=0) coherent buffer FlagsBuf { uint flags[]; };

// ---- volatile: 每次访问都读写内存（禁止寄存器缓存） ----
// 用于 memory-mapped I/O 或跨 workgroup 轮询
volatile float x = buf[0];

// ---- restrict: 承诺该指针不与其他指针别名 ----
// 允许编译器做更激进的优化
restrict float* ptr = ...;
```

### 6.2 Barrier 类型

```glsl
// ---- 控制屏障（Execution Barrier） ----
// 等待 workgroup 内所有线程到达此点
barrier();

// ---- 内存屏障（Memory Barrier） ----
// 确保 shared memory 写入对 workgroup 内所有线程可见
memoryBarrierShared();
barrier();                         // 通常配合使用

// 确保 buffer 写入对后续访问可见
memoryBarrierBuffer();

// 确保 image 写入对后续访问可见
memoryBarrierImage();

// ---- 分组屏障（Vulkan 1.1+） ----
groupMemoryBarrier();   // shared memory 写入 → workgroup 可见
```

### 6.3 标准 Reduce 模式（带 Shared Memory + Barrier）

```glsl
shared float cache[256];  // workgroup 内共享

void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int tid = int(gl_LocalInvocationID.x);

    // Step 1: 并行加载 → shared memory
    cache[tid] = (gid < pc.total) ? a[gid] : 0.0f;
    barrier();  // ← 确保所有线程写入完成

    // Step 2: 树状归约
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        barrier();  // ← 每轮归约后同步
    }

    // Step 3: 线程 0 写回全局结果
    if (tid == 0) {
        output[gl_WorkGroupID.x] = cache[0];
    }
}
```

**性能提示**：256 个线程的 workgroup，树状归约 `log2(256) = 8` 轮 barrier。Barrier 开销 ≈2-5 GPU 周期/次，8 轮 ≈20-40 周期，远小于 global memory 延迟（~200-800 周期）。

---

## 7. 共享内存 (Shared Memory)

### 7.1 声明与使用

```glsl
// ---- 静态大小 ----
shared float tile[16][16];    // 16×16 float tile = 1024 bytes
shared vec4  vectors[64];     // 64 个 vec4 = 1024 bytes

// ---- 动态大小（Vulkan 未知，OpenGL 有） ----
// Vulkan 中不支持动态 shared memory 声明
```

**硬件限制**（Vulkan 保证/常见 GPU）：

| GPU 类型 | shared memory / workgroup | 典型值 |
|----------|--------------------------|--------|
| Vulkan 保证 | ≥ 16 KB | 16 KB |
| Intel iGPU | 64 KB | |
| NVIDIA GTX/RTX | 48 KB (可配置到 96 KB) | |
| AMD RDNA2+ | 64 KB | |
| Apple Silicon | 32 KB | |
| Mali/Adreno | 16-32 KB | |

### 7.2 Bank Conflict 与 Padding

```glsl
// ---- 有 bank conflict（32-way） ----
shared float data[256];   // stride=32 访问时每 32 个地址冲突

// ---- 通过 padding 消除 ----
shared float data[256 + 16];  // 多余的元素打散 bank 映射
```

适用场景：reduce、矩阵乘 tile、卷积 im2col、softmax。

---

## 8. Subgroup 操作

Subgroup = warp (NVIDIA) / wavefront (AMD) / subgroup (Intel)，是 GPU 的**最小调度单元**（通常 32 或 64 线程）。

### 8.1 基础查询

```glsl
#extension GL_KHR_shader_subgroup_basic : require

uint size   = gl_SubgroupSize;              // 当前 subgroup 大小 (32/64)
uint id     = gl_SubgroupInvocationID;      // subgroup 内线程 ID [0, size-1]
uint num    = gl_NumSubgroups;              // workgroup 内 subgroup 数量
uint group  = gl_SubgroupID;                // 当前 subgroup 在 workgroup 中的索引
```

### 8.2 Shuffle（线程间数据交换）

```glsl
#extension GL_KHR_shader_subgroup_shuffle : require

// 从 subgroup 内指定线程读取值（无需 shared memory！）
float v = a[gid];
float neighbor = subgroupShuffle(v, 0);        // 从线程 0 读取

// 相对偏移
float next = subgroupShuffleUp(v, 1);           // 从 tid+1 读取
float prev = subgroupShuffleDown(v, 1);         // 从 tid-1 读取

// XOR 交换（蝴蝶模式）
float partner = subgroupShuffleXor(v, 1);       // 从 tid^1 读取
```

### 8.3 Vote（表决/判断）

```glsl
#extension GL_KHR_shader_subgroup_vote : require

bool cond = (a[gid] > 0.0);
bool any_true   = subgroupAny(cond);            // 任意线程为 true?
bool all_true   = subgroupAll(cond);            // 全部线程为 true?
bool elect      = subgroupElect();              // 仅一个线程返回 true
```

### 8.4 Ballot（位掩码）

```glsl
#extension GL_KHR_shader_subgroup_ballot : require

uvec4 mask = subgroupBallot(cond);             // 128-bit 位掩码
bool first = subgroupInverseBallot(mask);       // 最低位
bool excl  = subgroupBallotExclusiveBitCount(mask); // 排他前缀和
```

### 8.5 算术 Reduce/Scan

```glsl
#extension GL_KHR_shader_subgroup_arithmetic : require

float sum = subgroupAdd(v);          // subgroup 内求和
float min = subgroupMin(v);          // subgroup 内最小值
float max = subgroupMax(v);          // subgroup 内最大值

// 前缀/后缀扫描
float excl = subgroupExclusiveAdd(v);  // 排他前缀和
float incl = subgroupInclusiveAdd(v);  // 包含前缀和
```

### 8.6 Subgroup Barrier

```glsl
// Subgroup 级别屏障（比 barrier() 更快）
subgroupBarrier();
subgroupMemoryBarrier();
subgroupMemoryBarrierShared();
```

**Subgroup 操作的优势**：无需 shared memory、无需 barrier，2-5 个时钟周期完成（比 shared memory reduce 快 10-100 倍）。

---

## 9. 原子操作 (Atomic Operations)

```glsl
// ---- 类型限制 ----
// 原子操作仅支持 int / uint 类型（浮点需用 floatBitsToInt/intBitsToFloat 转换）

// ---- 基础操作 ----
uint old = atomicAdd(counter, 1u);           // 原子加
int  old = atomicMin(ptr, val);              // 原子最小值
uint old = atomicMax(ptr, val);              // 原子最大值
uint old = atomicAnd(ptr, mask);             // 位与
uint old = atomicOr(ptr, mask);              // 位或
uint old = atomicXor(ptr, mask);             // 位异或
uint old = atomicExchange(ptr, val);         // 原子交换
uint old = atomicCompSwap(ptr, cmp, val);    // 比较并交换 (CAS)

// ---- 浮点原子（变通方案） ----
float atomicAddFloat(uint* ptr, float val) {
    uint expected = *ptr;  // 注意：这行需要 coherent 修饰符
    while (true) {
        float newval = intBitsToFloat(expected) + val;
        uint old = atomicCompSwap(ptr, expected, floatBitsToInt(newval));
        if (old == expected) break;
        expected = old;
    }
}
```

**性能提示**：原子操作需要全局内存锁定和跨 workgroup 串行化。在需要大量原子操作的场景（如 histogram），优先使用 shared memory 内的原子操作 + 最终 workgroup 归约。

---

## 10. 内置数学函数参考

### 10.1 基础算术

```glsl
abs(x), sign(x)         // 绝对值、符号
floor(x), ceil(x), trunc(x), round(x), roundEven(x)
fract(x)                // 小数部分 = x - floor(x)
mod(x, y)               // x - y * floor(x/y)
min(x, y), max(x, y), clamp(x, lo, hi)
mix(a, b, t)            // lerp: a*(1-t) + b*t
step(edge, x)           // x < edge ? 0 : 1
smoothstep(e0, e1, x)   // Hermite 平滑阶跃
```

### 10.2 幂和指数

```glsl
pow(x, y), exp(x), exp2(x)
log(x), log2(x)
sqrt(x), inversesqrt(x)  // 1/sqrt(x) — 多数 GPU 有硬件加速
```

### 10.3 三角函数

```glsl
sin(x), cos(x), tan(x)
asin(x), acos(x), atan(x), atan2(y, x)
sinh(x), cosh(x), tanh(x)
radians(deg), degrees(rad)
```

### 10.4 位操作 (int/uint)

```glsl
bitfieldExtract(val, offset, bits), bitfieldInsert(base, insert, offset, bits)
bitCount(val), findLSB(val), findMSB(val)
bitfieldReverse(val)
```

### 10.5 类型重解释

```glsl
floatBitsToInt(f), floatBitsToUint(f)
intBitsToFloat(i), uintBitsToFloat(u)

// fp16 手动打包/解包（不使用扩展时）
uint packHalf2x16(vec2 v), vec2 unpackHalf2x16(uint u)
```

### 10.6 几何函数

```glsl
length(v), distance(v1, v2)
dot(v1, v2), cross(v1, v2)
normalize(v)
reflect(i, n), refract(i, n, eta)  // 需要光追或渲染
faceforward(n, i, nref)
```

---

## 11. 纹理与图像操作

虽然 nnops 当前全部使用 SSBO，但纹理适用于 2D 卷积、池化、上采样等图像类算子。

### 11.1 Image Load/Store（存储图像）

```glsl
// ---- 声明 ----
layout(set=0, binding=0, rgba32f) uniform readonly  image2D  input_img;
layout(set=0, binding=1, rgba32f) uniform writeonly image2D  output_img;
layout(set=0, binding=2, rgba16f) uniform            image2D  rw_img;  // fp16 格式

// ---- 读写（integer 坐标，不受过滤/wrap 影响） ----
vec4 val   = imageLoad(input_img, ivec2(x, y));
vec4 neigh = imageLoad(input_img, ivec2(x+1, y));  // 手动 fetch 邻居
imageStore(output_img, ivec2(x, y), result);
```

### 11.2 Sampled Image（采样图像）

```glsl
// ---- 声明 ----
layout(set=0, binding=0) uniform sampler2D input_tex;

// ---- 采样（float 坐标，受过滤/wrap 影响） ----
vec4 val = texture(input_tex, vec2(u, v));     // 双线性插值
vec4 lod = textureLod(input_tex, vec2(u,v), 0.0);  // 明确 mip level

// ---- texel fetch（integer 坐标，无过滤） ----
vec4 raw = texelFetch(input_tex, ivec2(x, y), 0);
```

### 11.3 格式限定符

```glsl
r32f, rg32f, rgba32f     // 32-bit float per channel
r16f, rgba16f             // 16-bit float per channel
r8, rg8, rgba8            // 8-bit normalized unsigned
r8ui, rgba8ui             // 8-bit unsigned integer
r8i, rgba8i               // 8-bit signed integer
```

---

## 12. Warp Divergence 与性能考量

### 12.1 什么是 Divergence

GPU 以 warp/subgroup（32-64 线程）为单位执行。同一 warp 内的线程共享一个 PC（程序计数器）。如果线程走不同分支，两个分支**都执行**，部分线程在每条路径上被 mask 掉。

```glsl
// ---- 有 divergence ----
int gid = int(gl_GlobalInvocationID.x);
if (gid % 2 == 0) {
    c[gid] = a[gid] * 2.0;    // 偶数线程执行
} else {
    c[gid] = a[gid] * 3.0;    // 奇数线程执行
}
// 50% 利用率：每条分支只有一半线程活跃

// ---- 无 divergence（特化常量消除） ----
if (OP == 0) { result = va + vb; }   // OP 是 specialization constant
// 编译器在 pipeline 创建时消除这个分支
// 运行时所有 32 个线程走同一条路径 — 100% 利用率
```

### 12.2 控制 Divergence 的策略

```glsl
// ---- 不好：workgroup 内部按线程 ID 分支 ----
if (gl_LocalInvocationID.x < 128) { /* path A */ }
else                                { /* path B */ }
// 同一个 warp 内的线程 (0..31) 都在 path A → 无 divergence ✓
// 但 warp 4 (128..159) 会分裂 → divergence ✗

// ---- 好：按 warp 边界分支 ----
if (gl_LocalInvocationID.x / 32 == 0) { /* path A - warp 0 */ }
// warp 0 全部走 path A → 无 divergence ✓
```

### 12.3 Nonuniform Resource Access

当线程在 divergent 分支中访问 descriptor：

```glsl
#extension GL_EXT_nonuniform_qualifier : require

// nonuniformEXT 告诉驱动：这个索引可能在同一 warp 内不同
texture(nonuniformEXT(sampler_array[gid % num_samplers]), uv);
```

---

## 13. 编译工具链：glslc → SPIR-V

### 13.1 基本编译

```bash
# 基础编译（Vulkan 1.0 目标）
glslc -o output.spv input.comp

# 指定目标环境（推荐）
glslc --target-env=vulkan1.2 -o output.spv input.comp
glslc --target-env=vulkan1.3 -o output.spv input.comp

# 包含路径
glslc -I./shaders/include -o output.spv input.comp

# 调试信息
glslc -g -o output.spv input.comp  # 完整调试信息（源码映射）

# 优化
glslc -O -o output.spv input.comp  # 性能优化（-Os=大小优化）

# 预处理输出（调试宏展开）
glslc -E input.comp               # 只预处理，不编译

# 警告
glslc -Wall -Werror -o output.spv input.comp
```

### 13.2 验证 SPIR-V

```bash
# SPIR-V 合法性验证
spirv-val output.spv

# 反汇编（查看生成的 SPIR-V 指令）
spirv-dis output.spv

# 交叉编译回 GLSL（检查编译器优化）
spirv-cross output.spv

# 检查使用了哪些 Capability
spirv-dis output.spv | grep -E "Capability|OpCapability"
# 输出示例：
#   OpCapability Shader
#   OpCapability StorageBuffer16BitAccess  ← fp16 storage
#   OpCapability Float16                    ← fp16 arithmetic (应有/无)
```

### 13.3 nnops 项目的编译管线

```
.comp 源文件
    │
    ├─ [glslc --target-env=vulkan1.2]
    │    └─→ .spv (SPIR-V 二进制, 32-bit word 小端)
    │
    └─ [spv_to_header.py --name g_<op>]
         └─→ _spv.h (C++ constexpr uint32_t 数组, 编译期内嵌)
              └─→ C++ 编译器 → nnops_vulkan.lib
```

整个管线在 `src/CMakeLists.txt` 中通过 CMake `add_custom_command` 实现。最终 SPIR-V 以 `constexpr uint32_t[]` 形式嵌入 `.text` 段，**无运行时文件 I/O**。

---

## 14. Vulkan 1.4 / Roadmap 2026 新特性

> **关于 "Vulkan 2.0"**：截至 2026 年 7 月，Khronos 官方未发布 "Vulkan 2.0"。Vulkan 生态通过**扩展驱动演进 + Roadmap 里程碑**来提升基线能力。Vulkan 1.4（2024 年 12 月）是当前最新主版本，Roadmap 2026（2026 年 1 月发布）针对高端 GPU 制定了更高的强制特性集。

### 14.1 版本演进时间线

```
Vulkan 1.0 (2016)  → Vulkan 1.1 (2018) → Vulkan 1.2 (2020)
  ↓                      ↓                    ↓
 基础计算+图形        Subgroup + 16-bit    完整的 16-bit 存储
                                           Descriptor Indexing

Vulkan 1.3 (2022)  → Vulkan 1.4 (2024.12) → Roadmap 2026 (2026.01)
  ↓                      ↓                    ↓
 Dynamic Rendering     Push Descriptor      VRS, Compute Derivatives,
                       Dynamic Rendering    Descriptor Heap,
                       Local Read,          Shader Clock Queries
                       Maintenance 5/6
```

**设计哲学**：Vulkan 不使用大版本号（如 D3D 的 9→10→11→12），而是通过以下机制演进：

| 机制 | 说明 | 示例 |
|------|------|------|
| **Core Promotion** | 广泛支持的扩展提升为核心特性 | `VK_KHR_push_descriptor` → Vulkan 1.4 core |
| **Roadmap Milestone** | 对高端 GPU 的额外强制特性集 | Roadmap 2026 要求 VRS、Compute Derivatives |
| **Device Extension** | 硬件厂商选择支持的附加能力 | `VK_EXT_descriptor_heap`、`VK_NV_cooperative_matrix2` |
| **SPIR-V Capability** | Shader 层面的能力声明 | `SPV_KHR_untyped_pointers`、`SPV_KHR_bfloat16` |

---

### 14.2 Vulkan 1.4 核心强制特性（2024 年 12 月）

以下扩展从 Vulkan 1.3 的可选特性提升为 1.4 的**必须支持**特性。所有 Vulkan 1.4 设备的 shader 均可用。

#### 14.2.1 VK_KHR_push_descriptor — 命令缓冲区内描述符更新

**取代传统的 descriptor set 分配流程**，直接在 command buffer 中写入 descriptor：

```glsl
// Shader 侧保持不变 — 仍用标准 set/binding 布局
layout(set = 0, binding = 0) readonly buffer BufA { float a[]; };
layout(set = 0, binding = 1) readonly buffer BufB { float b[]; };
layout(set = 0, binding = 2) buffer BufC { float c[]; };
```

```cpp
// C++ 侧：放弃 vkAllocateDescriptorSets，直接用 push descriptor
VkWriteDescriptorSet writes[] = { /* buffer info for binding 0, 1, 2 */ };
vkCmdPushDescriptorSetKHR(cmd,
    VK_PIPELINE_BIND_POINT_COMPUTE,
    pipeline_layout, 0, 3, writes);
// 不再需要 descriptor pool / descriptor set！
```

**对 nnops 的影响**：

- 可大幅简化 [vulkan_common.cpp](src/backend/vulkan/vulkan_common.cpp) 中的 descriptor 分配逻辑
- 消除 `VkDescriptorPool` 大小预估和碎片问题
- 每次 dispatch 前直接 push descriptor 即可

#### 14.2.2 VK_EXT_scalar_block_layout — C 风格结构体布局

允许 SSBO/UBO 结构体使用类似 C 的对齐规则（按成员自身大小对齐），而非 std140/std430 的 vec4 对齐：

```glsl
// Vulkan 1.3: std430 布局 — vec3 尾对齐到 16 bytes
// offset: | x(0) | y(4) | z(8) | --pad(12) | w(16) |
struct Vertex { vec3 position; float w; };  // 占用 20 bytes!

// Vulkan 1.4: scalar layout — 与 C struct 一致
// offset: | x(0) | y(4) | z(8) | w(12) |
// 需要在 VkPipelineShaderStageCreateInfo 中设置
// VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO
// 或声明时加 scalar 修饰符
layout(scalar, set=0, binding=0) buffer Buf { vec3 position; float w; };
// 现在占用 16 bytes，无尾填充
```

**优势**：直接映射 C++ struct 到 GLSL，消除手动 padding 和维护负担。

#### 14.2.3 VK_KHR_shader_subgroup_rotate — Subgroup 旋转操作

在 subgroup 内做环形移位，是许多 reduce/scan 算法的核心原语：

```glsl
// 新内置函数（Vulkan 1.4 core）
uint subgroupRotate(uint value, int delta);
float subgroupRotate(float value, int delta);

// 聚簇旋转（仅 subgroup 的子集间旋转）
uint subgroupClusteredRotate(uint value, int delta, uint clusterSize);

// 使用案例：高效的 warp 级前缀和
uint val = gl_SubgroupInvocationID;
// 旋转 1 位：每个线程获得邻居的值
uint neighbor = subgroupRotate(val, 1);
// tid=0 得到 tid=1 的值，tid=31 得到 tid=0 的值（环形）
```

**性能**：单条硬件指令，比 shared memory + barrier 快 10-50 倍。

#### 14.2.4 VK_KHR_shader_float_controls2 — 浮点控制增强

对浮点运算的精细控制：

```glsl
// 设置浮点 rounding mode 和 denorm 行为
// 通过 SPIR-V FPFastMathDefaultMode 或 FPEncoding 控制
// 无需 GLSL 语法变更 — 编译器通过 specialization constant 驱动
```

实际效果：可指定是否允许 flush-to-zero、选择 rounding mode（RTE/RTZ/RTP/RTN）。

#### 14.2.5 VK_KHR_shader_expect_assume — 编译器优化提示

给 SPIR-V 编译器提供分支预测和数据范围提示：

```glsl
// assume: 告诉编译器某个条件一定为真/假 → 消除边界检查
// expect: 告诉编译器某个条件大概率/小概率成立 → 优化分支预测

// 使用案例：workgroup 边界通常不触发
if (gid >= pc.total) return;  // 编译器无法推断 total 的分布

// 带 hint 的边界检查（通过 SPIR-V OpExpectKHR/OpAssumeKHR）
// 编译器可将 else 路径移出热路径
```

当前 glslc 的 GLSL 前端对此支持有限，主要通过 SPIR-V 汇编或 C++ inline SPIR-V 使用。

#### 14.2.6 VK_KHR_dynamic_rendering_local_read — 动态渲染本地读取

允许在 dynamic rendering pass 内读取之前写入的 attachment：

```glsl
// Vulkan 1.3: dynamic rendering pass 内只能写，不能读 attachment
// Vulkan 1.4: 可以读（至少支持 storage 资源和单采样 color attachment）

layout(set=0, binding=0, rgba8) uniform readonly image2D input_img;
layout(set=0, binding=1, rgba8) uniform writeonly image2D output_img;

// 在同一个 dynamic rendering pass 内：
vec4 prev = imageLoad(input_img, ivec2(x, y));   // Vulkan 1.4: OK
vec4 result = process(prev);
imageStore(output_img, ivec2(x, y), result);
```

#### 14.2.7 VK_KHR_maintenance5 和 VK_KHR_maintenance6

大量小的可用性提升，与 compute shader 相关的主要有：

| 特性 | 来源 | 说明 |
|------|------|------|
| `vkCmdBindIndexBuffer2` | Maint5 | 绑定 index buffer 的子范围 |
| `VkShaderModuleCreateInfo` 内联 | Maint5 | `VkPipelineShaderStageCreateInfo` 可直接嵌入 shader 源码，跳过 `vkCreateShaderModule` |
| `vkGetRenderingAreaGranularity` | Maint5 | 查询最优 render area 对齐 |
| `VK_NULL_HANDLE` 解绑 | Maint6 | 可传入 `VK_NULL_HANDLE` 解绑 index buffer |
| `maxCombinedImageSamplerDescriptorCount` | Maint6 | 查询 sampler + image descriptor 上限 |

#### 14.2.8 其他 Vulkan 1.4 强制特性

| 特性 | Compute 相关度 | 说明 |
|------|:---:|------|
| `hostImageCopy` | ⭐⭐ | 直接从 host 内存拷贝到 image，减少 staging buffer |
| `pipelineRobustness` | ⭐⭐⭐ | 越界访问不会导致 GPU 崩溃（返回 0 或 clamped 值） |
| `pushDescriptor` | ⭐⭐⭐ | 见 14.2.1 |
| `scalarBlockLayout` | ⭐⭐⭐ | 见 14.2.2 |
| `shaderSubgroupRotate` | ⭐⭐⭐ | 见 14.2.3 |
| 8K 渲染 + 8 render targets | ⭐ | 图形侧重 |
| `vertexAttributeInstanceRateDivisor` | — | 图形专用 |
| `indexTypeUint8` | — | 图形专用 |

---

### 14.3 Roadmap 2026 里程碑

2026 年 1 月发布，为**高端 GPU**（桌面独显、高端手机）制定比 Vulkan 1.4 更高的门槛。符合 Roadmap 2026 的设备预计 2026 年底上市。

#### 14.3.1 Compute Shader Derivatives — 计算着色器导数运算

**最重要的 compute shader 新特性**。允许在 compute shader 中使用图形 shader 中的导数函数：

```glsl
#extension GL_KHR_compute_shader_derivatives : require
// 需要 VK_KHR_compute_shader_derivatives 设备扩展

void main() {
    // 在 compute shader 中使用 dFdx / dFdy！
    // 要求同一 subgroup 内的相邻线程处理相邻像素（2×2 quad）
    float dx = dFdx(value);   // 线程 (x,y) 和 (x+1,y) 之间的差值
    float dy = dFdy(value);   // 线程 (x,y) 和 (x,y+1) 之间的差值
    float grad = length(vec2(dx, dy));

    // 有硬件加速的导数！不需要手动计算
    vec3 normal = normalize(cross(dFdx(pos), dFdy(pos)));
}
```

**应用场景**：

- Compute-based 程序化纹理的 mipmap 级别选择
- Compute shader 中的法线计算（如粒子系统、procedural mesh）
- 自适应采样率决策

**限制**：

- 仅在线程按 2D grid 排列且同一 quad（2×2 线程块）内所有线程活跃时有效
- NVIDIA warp 内 2×2 quad 天然满足；AMD wave64 需注意 dispatch 维度
- 不是所有 GPU 都支持——Roadmap 2026 才将其强制

#### 14.3.2 Variable Rate Shading (VRS) — 可变速率着色

允许在 framebuffer 不同区域使用不同的着色速率：

```
标准渲染：  1 fragment shader invocation / pixel
VRS 1×2：   1 invocation / 每 2 个水平像素
VRS 2×2：   1 invocation / 每 4 个像素
VRS 4×4：   1 invocation / 每 16 个像素
```

compute shader 可通过 fragment density map 控制着色速率。主要用于图形优化，compute 侧影响有限。

#### 14.3.3 Shader Clock Queries — 着色器内时钟查询

允许在 shader 内部查询 GPU 时钟周期，支持纳秒级 profiling：

```glsl
#extension GL_EXT_shader_clock : require

void main() {
    uint64_t start = clockRealtimeEXT();  // 实时时钟（纳秒）
    // ... 被测代码 ...
    uint64_t elapsed = clockRealtimeEXT() - start;
    // 将结果写入 debug buffer
}
```

**注意事项**：

- 时钟在不同 SM/CU 之间可能不同步
- 仅用于 profiling，不可用于跨 workgroup 同步
- Mobile GPU 可能有额外限制

#### 14.3.4 Host Image Copies — 主机端图像拷贝

直接从 CPU 内存拷贝到 GPU image（无需 staging buffer）：

```cpp
// 不再需要：vkMapMemory(staging_buffer) → memcpy → vkCmdCopyBufferToImage
// 直接：host memory → VkImage
VkHostImageCopyDevicePerformanceQueryEXT perf_query = {};
vkGetPhysicalDeviceHostImageCopyProperties(phys_dev, ..., &perf_query);
vkCopyMemoryToImageEXT(device, &copy_info);  // one-step copy
```

#### 14.3.5 更高描述符和 Shader 接口上限

| 参数 | Vulkan 1.3 保证值 | Roadmap 2026 要求 |
|------|-------------------|-------------------|
| `maxBoundDescriptorSets` | ≥ 4 | ≥ 8 |
| `maxDescriptorSetStorageBuffers` | ≥ 8 | ≥ 32 |
| `maxPerStageDescriptorStorageBuffers` | ≥ 4 | ≥ 16 |
| `maxStorageBufferRange` | ≥ 128 MB | ≥ 1 GB |
| `maxComputeWorkGroupInvocations` | ≥ 128 | ≥ 1024 |
| `maxComputeSharedMemorySize` | ≥ 16 KB | ≥ 64 KB |

---

### 14.4 VK_EXT_descriptor_heap — 全新描述符系统（2026 年 1 月）

**Vulkan 历史上最重要的架构变革之一**。用 D3D12 风格的单一描述符堆模型完全替换传统的 descriptor set 机制。

#### 14.4.1 设计动机

传统描述符系统的痛点：

```cpp
// 传统方式：Vulkan 1.0-1.4
VkDescriptorPool pool;            // ① 预估池大小 → 容易耗尽或浪费
VkDescriptorSetLayout layouts[];  // ② 为每种 shader 组合创建 layout
VkDescriptorSet sets[];           // ③ 从池中分配 set
vkUpdateDescriptorSets(...);      // ④ 逐个更新 binding
vkCmdBindDescriptorSets(...);     // ⑤ 绑定到 command buffer
// 问题：池碎片、layout 组合爆炸、频繁分配/释放
```

**描述符堆方式**：

```cpp
// VK_EXT_descriptor_heap：单一用户管理的堆
VkBuffer descriptor_heap;  // 标志：VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT
// 应用直接在 buffer 中写入 descriptor 数据
// 不再需要 VkDescriptorPool、VkDescriptorSetLayout、VkDescriptorSet！
```

#### 14.4.2 两种访问模式

**模式 A：映射访问（向后兼容）**

分配连续区域，将 heap 区域映射到现有 set/binding 布局：

```cpp
// 在 descriptor heap 中分配 slot
VkDescriptorHeapSlotEXT slot;
vkAllocateDescriptorHeapSlotsEXT(device, &alloc_info, &slot);

// 将 heap 映射到 shader 的 set/binding
VkBindDescriptorHeapInfoEXT bind_info = {
    .heap = heap,
    .firstSlot = slot,
    .pipelineLayout = layout,  // 兼容传统 layout
};
vkCmdBindDescriptorHeapsEXT(cmd, 1, &bind_info);
```

Shader 代码不变——仍用 `layout(set=0, binding=0)` 声明。

**模式 B：直接访问（需要 VK_KHR_shader_untyped_pointers）**

Shader 直接通过指针索引 heap 中的 descriptor——完全绕过 set/binding 映射：

```glsl
#extension GL_EXT_shader_untyped_pointers : require
// 需要 VK_KHR_shader_untyped_pointers + VK_EXT_descriptor_heap

void main() {
    // 通过 push constant 获得 heap 中的 offset
    // 直接读取 descriptor 数据（实现相关的结构）
    // 极高灵活度，但需要 VK_KHR_shader_untyped_pointers
}
```

#### 14.4.3 硬件与工具支持

| 厂商 | 支持状态 |
|------|----------|
| **NVIDIA** | Driver 610+ 完整支持；Nsight Graphics 2026.2 支持调试/捕获/回放 |
| **AMD** | 2026 年初宣布支持，具体驱动版本待定 |
| **Imagination** | Driver 26.1 支持 `VK_KHR_shader_untyped_pointers` |
| **Intel / Arm / Qualcomm** | 规划中 |

#### 14.4.4 对 nnops 的潜在影响

当前 nnops 使用传统 descriptor set 方式（[vulkan_common.cpp](src/backend/vulkan/vulkan_common.cpp) 中的 `VulkanPipelineCache`）。描述符堆可以：

1. **消除预先描述的符池大小限制**——动态分配
2. **简化多 shader 场景**——单一堆替代多套 layout
3. **减少 CPU 开销**——批量写入 vs 逐个 vkUpdateDescriptorSets
4. **迁移策略**：保持现有管线不变，仅在 dispatch 层替换 descriptor 分配路径

---

### 14.5 VK_KHR_shader_untyped_pointers — 无类型着色器指针（2025 年 8 月）

允许 shader 中使用无类型指针直接索引内存，绕过 Vulkan 的类型安全约束。

```glsl
#extension GL_EXT_shader_untyped_pointers : require

// 底层 SPIR-V 能力：SPV_KHR_untyped_pointers
// 允许在 shader 中将指针强制转换、按字节偏移、type-punning

// 用例 1：直接内存重解释
layout(set=0, binding=0) buffer Buf {
    // 无类型访问 — 用 uint 数组 view 一段内存
    // 配合指针运算实现 int/float 交错读取
} buf;

// 用例 2：Descriptor Heap 直接访问
// 通过无类型指针直接索引 heap 中的 descriptor slot
// 无需 set/binding layout 映射
```

**安全警告**：

- 无类型指针绕过 Vulkan 的类型安全保护
- 错误的偏移或 alignment 导致未定义行为（GPU 崩溃或静默数据损坏）
- 需要应用层严格的内存布局纪律

**驱动支持**：Mesa NVK/ANV/RADV 已支持，NVIDIA Vulkan Beta Driver 596.10+ 支持。

---

### 14.6 VK_EXT_shader_object — 无管线着色器

消除传统 `VkPipeline` 对象，改用独立的 `VkShaderEXT` 对象。

#### 14.6.1 基本用法

```cpp
// 传统方式：
// VkShaderModule → VkPipelineShaderStageCreateInfo
// → VkComputePipelineCreateInfo → vkCreateComputePipelines → VkPipeline
// → vkCmdBindPipeline → vkCmdDispatch

// Shader Object 方式：
VkShaderCreateInfoEXT shader_ci = {
    .sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
    .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
    .pCode = spirv_data,
    .codeSize = spirv_size,
    .pName = "main",
    .setLayoutCount = 1,
    .pSetLayouts = &descriptor_set_layout,
    .pushConstantRangeCount = 1,
    .pPushConstantRanges = &push_const_range,
    // 不需要 specialization constants — 特化在 shader object 创建时注入
    .pSpecializationInfo = &spec_info,
};

VkShaderEXT shader;
vkCreateShadersEXT(device, 1, &shader_ci, NULL, &shader);

// 绑定和 dispatch
vkCmdBindShadersEXT(cmd, 1, &stage, &shader);  // stage = VK_SHADER_STAGE_COMPUTE_BIT
vkCmdDispatch(cmd, groups_x, 1, 1);
```

#### 14.6.2 与 Compute Shader 的关系

| 方面 | VkPipeline（传统） | VkShaderEXT（新） |
|------|-------------------|-------------------|
| 编译时机 | `vkCreateComputePipelines`（可能卡顿） | `vkCreateShadersEXT` |
| 特化常量 | pipeline 创建时注入 | shader object 创建时注入 |
| 性能 | 基线 | 规范保证不慢于传统方式 |
| 切换开销 | 绑定 pipeline | 绑定 shader object |
| nnops 收益 | — | **边际收益**（compute 无图形状态需消除） |

**重要结论**：`VK_EXT_shader_object` 对 compute shader 的收益远小于图形 shader。compute pipeline 本身几乎无状态（无 rasterization、无 blending、无 depth/stencil），因此"消除静态状态"的优势不存在。NVIDIA 官方示例 [vk_minimal_latest](https://github.com/nvpro-samples/vk_minimal_latest) **有意对 compute 保留传统 VkPipeline 路径**。

---

### 14.7 Cooperative Matrix — Tensor Core / Matrix Core 加速

Vulkan 1.4 核心特性，提供厂商中立的矩阵加速硬件抽象，覆盖 NVIDIA Tensor Cores、AMD Matrix Cores、Imagination AI Accelerators、Arm/Qualcomm NPU。

#### 14.7.1 核心 API (GLSL)

```glsl
#extension GL_KHR_cooperative_matrix : enable
// Vulkan 1.4 中已是 core，无需 require

// 声明合作矩阵类型
coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> matA;
coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matB;
coopmat<float,      gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> matC;

// 从 buffer 加载 tile
coopMatLoad(matA, buf_a, offset_a, stride_a, gl_CooperativeMatrixLayoutRowMajor);
coopMatLoad(matB, buf_b, offset_b, stride_b, gl_CooperativeMatrixLayoutRowMajor);
coopMatLoad(matC, buf_c, offset_c, stride_c, gl_CooperativeMatrixLayoutRowMajor);

// 矩阵乘加：D = A × B + C — 在 Tensor Core 上执行！
matC = coopMatMulAdd(matA, matB, matC);

// 存储结果
coopMatStore(matC, buf_d, offset_d, stride_d, gl_CooperativeMatrixLayoutRowMajor);
```

**关键概念**：

- `coopmat` 将矩阵分片（tile）分布在 subgroup（通常 32 线程 NVIDIA / 64 线程 AMD）**所有** invocation 之间
- 每个线程只持有矩阵的一个**片段**（fragment），而非完整元素
- `coopMatLoad` 和 `coopMatStore` 负责内存 ↔ tile 的转置和解包
- `coopMatMulAdd` 在专用硬件（tensor core）上执行，不占用 ALU

#### 14.7.2 查询支持矩阵尺寸

并非所有 M×N×K 组合都合法——必须在运行时查询：

```cpp
VkPhysicalDeviceCooperativeMatrixPropertiesKHR props;
vkGetPhysicalDeviceCooperativeMatrixProperties(phys_dev, &count, &props);

// 典型的合法组合（NVIDIA Volta/Turing/Ampere/Ada）：
//   A: f16, M×K = 16×16, ScopeSubgroup
//   B: f16, K×N = 16×16, ScopeSubgroup
//   C: f32, M×N = 16×16, ScopeSubgroup
//   D: f32, M×N = 16×16, ScopeSubgroup
```

#### 14.7.3 Cooperative Matrix 2 (VK_NV_cooperative_matrix2, 2025 年 2 月)

NVIDIA 在 Vulkanised 2025 发布了 7 项增强（当前为 NV 厂商扩展，待 KHR 推广）：

| 特性 | 说明 | 代码示例 |
|------|------|----------|
| **Flexible Dimensions** | 矩阵尺寸通过特化常量动态指定 | `coopmat<f16, Subgroup, M, N, UseA>` |
| **Workgroup Scope** | tile 跨整个 workgroup（不限于 subgroup） | `gl_ScopeWorkgroup` |
| **Tensor Layouts** | 声明式多维内存寻址，自动 bounds check | `tensorLayoutNV<2> layout;` |
| **Block Loads** | 优化的批量内存加载 | `coopMatLoadBlockNV(...)` |
| **Reductions** | 内置归约操作（sum/min/max） | `coopMatReduceNV(...)` |
| **Conversions** | 无中间变量的类型转换 | `coopMatConvertNV(...)` |
| **Per-element Operations** | 单个元素级别的操作 | `coopMatElementOpNV(...)` |

**Tensor Layout 示例**（最强大的新特性）：

```glsl
// 声明 5D tensor layout：维度、stride、offset、span 每个维度独立
tensorLayoutNV<2> tensorA = createTensorLayoutNV(2);
tensorA = setTensorLayoutDimensionNV(tensorA, M, K);

// 用 slice 操作切出子矩阵（零拷贝）
coopMatLoadTensorNV(matA, inputA.x, slice(tensorA, row, M, k, K));
// 支持 transpose、space-to-depth、depth-to-space 等 tensorView 操作
```

**性能数据**（NVIDIA RTX 4070）：

| 实现方案 | FP16 GEMM 吞吐 | 代码复杂度 |
|----------|---------------|-----------|
| 简单 CoopMat1 | ~8 TFLOPS | 低（naïve 加载） |
| 手优 CoopMat1 | ~98 TFLOPS | 高（手动 shared memory 分级 + pipelining） |
| **CoopMat2** | **~97 TFLOPS** | 中（tensor layout API，无需手动 staging） |
| 理论峰值 Tensor Core | ~116 TFLOPS | — |

**结论**：Cooperative Matrix 2 在保持接近手优性能的同时大幅降低代码复杂度。

#### 14.7.4 对 nnops 的影响

对于 GEMM/MatMul 算子的未来优化路径：

1. **短期**：继续使用当前 SIMD + tiled GEMM（CPU 后端）
2. **中期**：Vulkan GEMM 可使用 `VK_KHR_cooperative_matrix` 获得 tensor core 加速
3. **长期**：引入 tensor layout 进一步简化代码，支持混合精度（f16→f32→f16）

---

### 14.8 新数值类型扩展（2025 年）

#### 14.8.1 VK_KHR_shader_bfloat16 — Brain Float 16

ML/AI 推理的首选格式——与 fp32 相同指数范围（8-bit exponent），仅截断尾数：

```glsl
#extension GL_EXT_shader_explicit_arithmetic_types_bfloat16 : require
// 需要 VK_KHR_shader_bfloat16

// bfloat16 在 SSBO 中的使用
layout(set=0, binding=0) readonly buffer BufA { bfloat16_t a[]; };
layout(set=0, binding=1) buffer BufOut { bfloat16_t out[]; };

void main() {
    // bfloat16 → float 转换是简单的左移 16 位（零损耗）
    float val = float(a[gid]);        // bf16→f32 精确映射
    float result = val * 2.0f + 1.0f;
    out[gid] = bfloat16_t(result);    // f32→bf16：直接截断低 16 位
}
```

| 格式 | 总位 | 符号 | 指数 | 尾数 | 范围 | 精度 |
|------|------|------|------|------|------|------|
| fp32 | 32 | 1 | 8 | 23 | ±3.4×10³⁸ | ~7 位十进制 |
| **bfloat16** | **16** | **1** | **8** | **7** | **±3.4×10³⁸** | **~2 位十进制** |
| fp16 | 16 | 1 | 5 | 10 | ±65504 | ~3 位十进制 |

**关键洞察**：

- bf16↔f32 转换只需截断/补零——零硬件开销
- bf16 保留了 f32 的完整动态范围（相同指数位）——梯度在训练中不易溢出/下溢
- ML 推理对尾数精度不敏感——bf16 是最优格式
- fp16 更适合精确计算但范围有限——两者互补

#### 14.8.2 VK_EXT_shader_float8 — 8 位浮点（2025 年 7 月）

超低精度浮点，用于量化推理和数据流压缩：

```glsl
#extension GL_EXT_shader_float8 : require
// 需要 VK_EXT_shader_float8

// E4M3 (4-bit exponent, 3-bit mantissa) — 正向范围大
// E5M2 (5-bit exponent, 2-bit mantissa) — 动态范围更大
float8_e4m3_t  small_val;  // 用于激活值 / 特征数据
float8_e5m2_t  wide_val;   // 用于梯度 / 权重
```

**应用场景**：

- LLM 推理（8-bit 量化权重）
- 数据传输压缩（减少带宽 4× vs fp32）
- 配合 cooperative matrix 做量化矩阵乘

#### 14.8.3 VK_EXT_shader_long_vector — 长向量（2025 年 12 月）

突破 GLSL 传统的 4 分量向量上限：

```glsl
#extension GL_EXT_shader_long_vector : require
// SPIR-V 扩展：SPV_EXT_long_vector

// 支持 8、16、32 分量向量！
float8   v8;    // 8×32 = 256 bits — 适配 AVX
float16  v16;   // 16×32 = 512 bits — 适配 AVX-512
float32  v32;   // 32×32 = 1024 bits

// 对大块逐元素操作非常有用
v16 = a[gid:gid+15] * b[gid:gid+15] + c[gid:gid+15];  // 伪代码
```

**当前状态**：扩展规范已定义，glslc/驱动支持仍在开发中。主要用于未来宽 SIMD 的代码生成（CPU/GPU 上有 AVX-512 / wide warp 场景）。

#### 14.8.4 VK_KHR_shader_fma — 保证正确舍入的 FMA（2025 年 10 月）

保证 `fma(a, b, c)` 执行**单次舍入**的乘加操作（而非先乘后加的两步舍入）：

```glsl
#extension GL_KHR_shader_fma : require
// SPIR-V: SPV_KHR_fma

// fma: a*b + c 在一步内完成，只舍入一次 → 更高精度
float precise = fma(a, b, c);

// 与普通 a*b + c 的区别（两步舍入）：
// a = 1.0000001, b = 1.0000001, c = -1.0
// 普通: (1.0000001*1.0000001) - 1.0 = 1.0000002 - 1.0 = 0.0000002 (舍入两次)
// fma:   1.0000001*1.0000001 - 1.0   = 0.00000020000001     (舍入一次)
```

**应用场景**：高精度线性代数、Kahan summation、财务计算、迭代求精。

---

### 14.9 新 SPIR-V 能力与编译器增强

#### 14.9.1 SPV_KHR_compute_shader_derivatives

允许 SPIR-V compute shader 中包含 `OpDPdx`/`OpDPdy` 等导数指令（对应 GLSL `dFdx`/`dFdy`）：

```
OpCapability ComputeDerivativeGroupQuadsKHR
OpDPdx %float %value   ← compute shader 中的导数指令（以前仅 fragment shader 允许）
```

Roadmap 2026 强制后，将在所有高端 GPU 上可用。

#### 14.9.2 SPV_KHR_untyped_pointers

允许 SPIR-V 中声明不透明指针类型和指针强制转换操作：

```
OpCapability UntypedPointersKHR
OpPtrCast %uint_ptr %untyped_ptr   ← 无类型指针 → 有类型指针转换
OpUntypedAccessChainKHR ...         ← 无类型指针的偏移运算
```

#### 14.9.3 SPV_KHR_bfloat16

允许 SPIR-V 中使用 bfloat16 类型和算术运算：

```
OpCapability BFloat16KHR
%bf16 = OpTypeFloat 16
OpFAdd %bf16 %a %b   ← bf16 原生加法
```

#### 14.9.4 SPV_EXT_long_vector

允许 SPIR-V 中声明超过 4 分量的向量类型：

```
OpCapability LongVectorEXT
%v8float = OpTypeVector %float 8    ← 8 分量向量
%v16float = OpTypeVector %float 16  ← 16 分量向量
```

#### 14.9.5 SPV_KHR_fma

保证 SPIR-V 中的 FMA 指令执行单次舍入：

```
OpCapability FMAKHR
%result = OpFma %float %a %b %c    ← 保证单次舍入的 fma
```

与之前未标记 `NoContraction` 时的编译器行为不同——原先允许但不保证 fma contraction。

---

### 14.10 Vulkan 1.4 下 nnops 的迁移路径

#### 14.10.1 立即可用（零修改，所有 Vulkan 1.4 设备）

- ✅ `VK_KHR_push_descriptor`（简化 descriptor 管理）
- ✅ `VK_KHR_shader_subgroup_rotate`（高效 reduce/scan 原语）
- ✅ `VK_EXT_scalar_block_layout`（C 结构体直接映射到 SSBO）
- ✅ `pipelineRobustness`（越界访问安全性提升）

#### 14.10.2 条件可用（需要 query + fallback）

- 🔶 `VK_KHR_cooperative_matrix`（GEMM tensor core 加速——需运行时查询支持的 M×N×K）
- 🔶 `VK_EXT_descriptor_heap`（取决于厂商支持——NVIDIA 已就绪，AMD/Intel 待定）
- 🔶 `VK_EXT_shader_object`（compute 收益有限，但可简化 pipeline 管理）

#### 14.10.3 未来规划

- 🔜 `VK_KHR_compute_shader_derivatives`（Roadmap 2026 后广泛可用时，用于 compute-based 自适应采样）
- 🔜 `VK_KHR_shader_bfloat16`（ML 推理场景的 GEMM 加速）
- 🔜 Cooperative Matrix 2（待 KHR 推广后，简化 tensor core 代码）

#### 14.10.4 推荐的最低目标

| 目标用户 | 推荐 Vulkan 版本 | 关键特性 |
|----------|-----------------|----------|
| 通用计算 | **Vulkan 1.3** | 当前 nnops 基线 |
| 新项目 | **Vulkan 1.4** | push descriptor + scalar layout + robust access |
| 高端优化 | **Roadmap 2026** | compute derivatives + descriptor heap + 更高上限 |
| ML 推理 | **1.4 + bf16 + coopmat** | bfloat16 + tensor core GEMM |

---

## 附录 A：nnops 项目 Shader 实例

### 完整 f32 eltwise shader ([eltwise_f32.comp](../src/backend/vulkan/shaders/eltwise_f32.comp))

```glsl
#version 450
#extension GL_GOOGLE_include_directive : enable
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(constant_id = 0) const int OP = 0;
layout(constant_id = 1) const int ADD_TO = 0;
layout(set = 0, binding = 0) readonly buffer BufA { float a[]; };
layout(set = 0, binding = 1) readonly buffer BufB { float b[]; };
layout(set = 0, binding = 2) buffer BufC { float c[]; };
layout(push_constant) uniform PushConstants { int total; } pc;

void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= pc.total) return;
    float va = a[gid], vb = b[gid], result;
    if (OP == 0)      result = va + vb;
    else if (OP == 1) result = va - vb;
    else if (OP == 2) result = va * vb;
    else              result = va / vb;
    if (ADD_TO == 1) c[gid] += result;
    else             c[gid] = result;
}
```

### 完整 f16 eltwise shader ([eltwise_f16.comp](../src/backend/vulkan/shaders/eltwise_f16.comp))

```glsl
#version 450
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_16bit_storage : require
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(constant_id = 0) const int OP = 0;
layout(constant_id = 1) const int ADD_TO = 0;
layout(set = 0, binding = 0) readonly buffer BufA { float16_t a[]; };
layout(set = 0, binding = 1) readonly buffer BufB { float16_t b[]; };
layout(set = 0, binding = 2) buffer BufC { float16_t c[]; };
layout(push_constant) uniform PushConstants { int total; } pc;

void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= pc.total) return;
    float va = float(a[gid]), vb = float(b[gid]), result;
    if (OP == 0)      result = va + vb;
    else if (OP == 1) result = va - vb;
    else if (OP == 2) result = va * vb;
    else              result = va / vb;
    if (ADD_TO == 1) c[gid] = float16_t(float(c[gid]) + result);
    else             c[gid] = float16_t(result);
}
```

## 附录 B：快速参考卡片

| 需求 | 语法 | nnops 使用 |
|------|------|-----------|
| Workgroup 大小 | `layout(local_size_x = 256) in;` | ✅ 全部用 256 |
| 全局线程 ID | `gl_GlobalInvocationID.x` | ✅ 1D 索引 |
| 边界守卫 | `if (gid >= pc.total) return;` | ✅ 必须 |
| SSBO 只读 | `readonly buffer Buf { float a[]; }` | ✅ |
| SSBO 可读写 | `buffer Buf { float c[]; }` | ✅ |
| 特化常量 | `layout(constant_id=0) const int OP=0;` | ✅ OP, ADD_TO |
| Push Constant | `layout(push_constant) uniform PC {...};` | ✅ total |
| fp16 存储 | `#extension GL_EXT_shader_16bit_storage` | ✅ f16 shaders |
| Shared Memory | `shared float tile[256];` | 🔜 reduce/softmax |
| Subgroup | `subgroupAdd(v)` | 🔜 高效 reduce |
| Barrier | `barrier();` | 🔜 shared mem 使用 |
| 原子操作 | `atomicAdd(counter, 1u)` | 🔜 histogram |
