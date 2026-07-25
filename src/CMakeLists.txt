# ---- CPU implementation library (always built) ----
add_library(nnops_cpu STATIC)
target_link_libraries(nnops_cpu PUBLIC nnops)

# Gather CPU sources explicitly (no file(GLOB))
set(NNOPS_CPU_SOURCES
    # Operator dispatch layer
    ops/conv2d.cpp
    ops/activation.cpp
    ops/pooling.cpp
    ops/linear.cpp
    ops/matmul.cpp
    ops/attention.cpp
    ops/softmax.cpp
    ops/cumsum.cpp
    ops/batch_norm.cpp
    ops/layer_norm.cpp
    ops/rms_norm.cpp
    ops/conv3d.cpp
    ops/depthwise_conv2d.cpp
    ops/eltwise.cpp
    ops/unary.cpp
    ops/reduce.cpp

    # CPU feature detection (runtime CPUID / NEON probing)
    detail/cpu_features.cpp

    # CPU optimized kernels (SIMD-accelerated)
    backend/cpu/activation.cpp
    backend/cpu/batch_norm.cpp
    backend/cpu/depthwise_conv2d.cpp
    backend/cpu/layer_norm.cpp
    backend/cpu/pooling.cpp
    backend/cpu/rms_norm.cpp
    backend/cpu/softmax.cpp
    backend/cpu/eltwise.cpp
    backend/cpu/unary.cpp
    backend/cpu/reduce.cpp

    # CPU tiled GEMM dispatch (platform-abstracted over x86_64 / aarch64 micro-kernels)
    backend/cpu/imatmul.cpp

    # CPU tiled matmul kernel (NKM loop with pack + MMA)
    backend/cpu/matmul.cpp

    # CPU reference kernels (always compiled — correctness baseline)
    backend/cpu/reference/conv2d_ref.cpp
    backend/cpu/reference/activation_ref.cpp
    backend/cpu/reference/pooling_ref.cpp
    backend/cpu/reference/linear_ref.cpp
    backend/cpu/reference/matmul_ref.cpp
    backend/cpu/reference/attention_ref.cpp
    backend/cpu/reference/softmax_ref.cpp
    backend/cpu/reference/cumsum_ref.cpp
    backend/cpu/reference/batch_norm_ref.cpp
    backend/cpu/reference/layer_norm_ref.cpp
    backend/cpu/reference/rms_norm_ref.cpp
    backend/cpu/reference/conv3d_ref.cpp
    backend/cpu/reference/depthwise_conv2d_ref.cpp
    backend/cpu/reference/eltwise_ref.cpp
    backend/cpu/reference/unary_ref.cpp
    backend/cpu/reference/reduce_ref.cpp
)

target_sources(nnops_cpu PRIVATE ${NNOPS_CPU_SOURCES})
target_include_directories(nnops_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# ---- CUDA backend (optional) ----
if(NNOPS_BUILD_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    set(NNOPS_CUDA_SOURCES
        backend/cuda/softmax.cu
        backend/cuda/layer_norm.cu
        backend/cuda/rms_norm.cu
        backend/cuda/activation.cu
        backend/cuda/batch_norm.cu
        backend/cuda/pooling.cu
        backend/cuda/cumsum.cu
        backend/cuda/depthwise_conv2d.cu
        backend/cuda/eltwise.cu
        backend/cuda/unary.cu
        backend/cuda/reduce.cu
    )

    add_library(nnops_cuda STATIC ${NNOPS_CUDA_SOURCES})
    target_link_libraries(nnops_cuda PUBLIC nnops)
    target_include_directories(nnops_cuda PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(nnops_cpu PUBLIC NNOPS_HAS_CUDA)
    target_link_libraries(nnops_cpu PUBLIC nnops_cuda)
endif()

# ---- Vulkan backend (optional) ----
if(NNOPS_BUILD_VULKAN)
    find_package(Vulkan REQUIRED)

    set(NNOPS_VULKAN_SOURCES
        backend/vulkan/conv2d_vulkan.cpp
        backend/vulkan/activation_vulkan.cpp
        backend/vulkan/pooling_vulkan.cpp
    )

    add_library(nnops_vulkan STATIC ${NNOPS_VULKAN_SOURCES})
    target_link_libraries(nnops_vulkan PUBLIC nnops Vulkan::Vulkan)
    target_include_directories(nnops_vulkan PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(nnops_cpu PUBLIC NNOPS_HAS_VULKAN)
    target_link_libraries(nnops_cpu PUBLIC nnops_vulkan)
endif()

# ---- Alias targets for unified consumption ----
add_library(nnops::nnops ALIAS nnops)
add_library(nnops::cpu   ALIAS nnops_cpu)
