# 项目描述

这是一个多后端的算子库，基于现代 C++23 标准，旨在提供高性能、可扩展的计算算子和工具，支持多种硬件后端（如 CPU、CUDA、Vulkan 等），以满足不同的模型推理应用场景的需求。

# 项目目录

代码的组织结构如下

```
docs/
skills/
include/
  └── nnops/
      ├── conv2d.hpp
      ├── activation.hpp
      ├── pooling.hpp
      └── linear.hpp
      └── matmul.hpp
src/
  ├── backend/
      ├── cpu/
      ├── cuda/
      └── vulkan/
  |__ ops/
      ├── conv2d.cpp
      ├── activation.cpp
      ├── pooling.cpp
      └── linear.cpp
tests/
  ├── test_conv2d.cpp
  ├── test_activation.cpp
  └── test_pooling.cpp
```

# 项目特性

+ 项目的 namespace 为 nnops
+ 算子内部使用模板元编程和现代 C++ 特性（如 concepts、constexpr、ranges 等）来实现高性能和类型安全。
+ 支持多种数据类型（如 float、half、int8_t 等）和不同的张量布局（如 NCHW、NCHWC8）。
+ 支持多种硬件后端，包括 CPU、CUDA 和 Vulkan，并提供统一的接口来调用不同后端的实现，并且 CPU 支持 x86_64/aarch64。
+ 提供丰富的算子集合，包括 Conv、Activation、Pool、Linear、MatMul 等常用操作。
+ 算子包含类和函数式两种调用方式。
+ 提供单元测试和基准测试，确保算子的正确性和性能。
+ 提供详细的文档和示例代码，帮助用户快速上手和集成到自己的项目中。
+ 算子内部不申请额外的内存，所有的内存管理由用户负责，确保高性能和低延迟。
+ 算子不自己定义线程池，通过外部提供的 parallel_for 接口实现 CPU 侧的多线程
+ 不依赖第三方库，所有的实现都是自包含的，便于移植和集成。
+ 所有算子提供朴素的CPU 参考实现。

# 相关接口

列出部分关键接口
 
+ getWorkspace: 获取算子运行时所需的临时缓存，对于 CPU，该缓存可能和算子内并行数量有关
+ compute: 对算子进行前向计算，cuda 支持 stream, CPU 支持多线程
+ create: 构建算子类
+ getOpType: 对应 onnx 的 OpType
+ getBackend: 获取当前算子的计算后端

# 相关数据结构

+ TensorView 类，不持有数据的所有权，但是包含张量数据的描述、数据指针，描述包含 shape 、stride、layout、data_type 等基本信息
+ OpBase 类：算子的基类

# 参考项目 

在实现项目前需要对下面的项目相关算子的接口与实现有大致的了解：

+ onnxruntime: 如果没有，请询问 onnxruntime 项目的目录
+ TensorRT: 如果没有，请询问 TensorRT 项目的目录
+ ComputeLibrary: 如果没有，请询问 ComputeLibrary 项目的目录
+ OpenCV: 如果没有，请询问 OpenCV 项目的目录。主要参考 OpenCV 的 Universal Intrinsics (modules/core/include/opencv2/core/hal/intrin*.hpp) — 一套跨平台的 SIMD 指令封装，将 SSE/AVX/NEON 等不同 ISA 统一为类型安全的 vector 类型 (v_float32x4, v_float32x8 等)，提供 load/store/arithmetic/comparison/FMA 等通用操作。nnops 的 SIMD 抽象层（include/nnops/detail/simd/）借鉴了 OpenCV 的这一设计模式。

如果不能准确的获取上面的目录，不要开始写代码或者重构项目。额外的，需要确保算子能够很方便的接入 onnxruntime 和 TensorRT 的自定义算子。

# 构建方式

+ 项目使用现代 CMake 构建
+ CPU 相关算子必须构建
+ 可选构建 CUDA/Vulkan 算子
+ CMake 需要明确指定 .cpp 文件，不使用 glob 等搜索方式
+ CMake 的写法尽量现代，规范

# 语言风格

+ 项目使用 C++23 标准，尽量使用现代 C++ 特性

# 额外的

如果你有更好的想法，可以直接询问并确认后实现