# ---- CPU implementation library (always built) ----
add_library(nnops_cpu STATIC)
target_link_libraries(nnops_cpu PUBLIC nnops)

# Gather CPU sources explicitly (no file(GLOB))
set(NNOPS_CPU_SOURCES
    # Operator dispatch layer
    ops/clamp.cpp
    ops/conv2d.cpp
    ops/activation.cpp
    ops/pooling.cpp
    ops/linear.cpp
    ops/matmul.cpp
    ops/attention.cpp
    ops/softmax.cpp
    ops/cumsum.cpp
    ops/norm.cpp
    ops/conv3d.cpp
    ops/depthwise_conv.cpp
    ops/eltwise.cpp
    ops/embed.cpp
    ops/permute.cpp
    ops/slice.cpp
    ops/unary.cpp
    ops/reduce.cpp
    ops/resize.cpp
    ops/grid_sample.cpp
    ops/layout_convert.cpp
    ops/linear_attention.cpp
    ops/quant_linear.cpp
    ops/concat.cpp
    ops/argminmax.cpp
    ops/topk.cpp
    ops/rope.cpp
    ops/flatten.cpp
    ops/transpose_conv2d.cpp
    ops/moe.cpp

    # CPU feature detection (runtime CPUID / NEON probing)
    detail/cpu_features.cpp

    # Operator scratch-memory pool (shared, thread-safe singleton)
    backend/cpu/common/memory_pool.cpp

    # Layout conversion utilities (pack/unpack between NCHW and NCHWC8)
    backend/cpu/layout_convert.cpp

    # CPU optimized kernels (SIMD-accelerated)
    backend/cpu/activation.cpp
    backend/cpu/clamp.cpp
    backend/cpu/norm.cpp
    backend/cpu/softmax.cpp
    backend/cpu/eltwise.cpp
    backend/cpu/embed.cpp
    backend/cpu/permute.cpp
    backend/cpu/slice.cpp
    backend/cpu/unary.cpp
    backend/cpu/reduce.cpp
    backend/cpu/resize.cpp
    backend/cpu/grid_sample.cpp
    backend/cpu/quant_linear.cpp
    backend/cpu/concat.cpp
    backend/cpu/argminmax.cpp
    backend/cpu/topk.cpp
    backend/cpu/rope.cpp
    backend/cpu/flatten.cpp
    backend/cpu/transpose_conv2d.cpp
    backend/cpu/moe.cpp

    # CPU tiled matmul kernel (NKM loop with pack + MMA), plus the two routes
    # that bypass the plan (small naive / GEMV, fused Q4_1). matmul_helper is the
    # shared pack/MMA layer that conv2d and attention also drive.
    backend/cpu/matmul/matmul.cpp
    backend/cpu/matmul/matmul_helper.cpp
    backend/cpu/matmul/matmul_small.cpp
    backend/cpu/matmul/matmul_qblock_fused.cpp

    # Linear is the MatMul GEMM with a transposed weight, a per-column bias and
    # an fp output for quantized operands; it drives matmul.cpp above.
    backend/cpu/matmul/linear.cpp

    # CPU convolution kernels: the implicit-GEMM route (both layouts — planar
    # NCHW and channel-last NHWC, fp and quantized, one file since they share a
    # plan, a scratch layout and a driver), the Winograd F(2x2,3x3) route, the
    # depthwise kernel and pooling (which sweeps the same windows the two above
    # do, and shares their channel-lane / width-lane split).
    backend/cpu/conv/conv2d_implicit_gemm.cpp
    backend/cpu/conv/conv2d_winograd.cpp
    backend/cpu/conv/depthwise_conv.cpp
    backend/cpu/conv/pooling.cpp
    backend/cpu/conv/conv3d.cpp

    # CPU tiled attention kernels (fused QK^T + softmax + AV; linear attention
    # lives alongside it)
    backend/cpu/attention/attention.cpp
    backend/cpu/attention/linear_attention.cpp

    backend/cpu/cumsum.cpp
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
        backend/cuda/depthwise_conv.cu
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

    # Find Python for the SPIR-V → header conversion script (needed early)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    # ---- Find glslc for shader compilation ----
    find_program(GLSLC_EXECUTABLE glslc
        HINTS "${Vulkan_GLSLC_EXECUTABLE}"
              "$ENV{VULKAN_SDK}/Bin"
              "$ENV{VK_SDK_PATH}/Bin")
    if(NOT GLSLC_EXECUTABLE)
        message(FATAL_ERROR "glslc not found (part of Vulkan SDK shaderc tools). "
                            "Set VULKAN_SDK or VK_SDK_PATH environment variable.")
    endif()

    # ---- Shader compilation: .comp → .spv ----
    set(VULKAN_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/backend/vulkan/shaders")
    set(VULKAN_SHADER_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/vulkan_shaders")

    # List of shaders to compile
    set(VULKAN_SHADERS eltwise_f32 unary_f32 eltwise_f16 unary_f16)
    set(VULKAN_SPV_FILES "")
    set(VULKAN_SPV_HEADERS "")

    foreach(shader ${VULKAN_SHADERS})
        set(SHADER_SRC  "${VULKAN_SHADER_DIR}/${shader}.comp")
        set(SHADER_SPV  "${VULKAN_SHADER_OUT_DIR}/${shader}.spv")
        set(SHADER_HDR  "${CMAKE_CURRENT_SOURCE_DIR}/backend/vulkan/${shader}_spv.h")

        add_custom_command(
            OUTPUT ${SHADER_SPV}
            COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.2 -o ${SHADER_SPV} ${SHADER_SRC}
            DEPENDS ${SHADER_SRC}
            COMMENT "Compiling Vulkan shader: ${shader}.comp → ${shader}.spv"
        )

        # Generate C header from SPIR-V (runs after .spv is produced)
        add_custom_command(
            OUTPUT ${SHADER_HDR}
            COMMAND Python3::Interpreter
                "${CMAKE_CURRENT_SOURCE_DIR}/backend/vulkan/spv_to_header.py"
                --name "g_${shader}" --spv ${SHADER_SPV} --out ${SHADER_HDR}
            DEPENDS ${SHADER_SPV} "${CMAKE_CURRENT_SOURCE_DIR}/backend/vulkan/spv_to_header.py"
            COMMENT "Generating SPIR-V header: ${shader}_spv.h"
        )

        list(APPEND VULKAN_SPV_FILES ${SHADER_SPV})
        list(APPEND VULKAN_SPV_HEADERS ${SHADER_HDR})
    endforeach()

    # Custom target to ensure shaders are built before the library
    add_custom_target(vulkan_shaders DEPENDS ${VULKAN_SPV_HEADERS})

    # ---- Vulkan backend sources ----
    set(NNOPS_VULKAN_SOURCES
        backend/vulkan/vulkan_common.cpp
        backend/vulkan/eltwise_vulkan.cpp
        backend/vulkan/unary_vulkan.cpp
    )

    add_library(nnops_vulkan STATIC
        ${NNOPS_VULKAN_SOURCES}
        ${VULKAN_SPV_FILES}
    )
    add_dependencies(nnops_vulkan vulkan_shaders)

    target_link_libraries(nnops_vulkan PUBLIC nnops Vulkan::Vulkan)
    target_include_directories(nnops_vulkan PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
    )
    target_compile_definitions(nnops_cpu PUBLIC NNOPS_HAS_VULKAN)
    target_link_libraries(nnops_cpu PUBLIC nnops_vulkan)
endif()

# ---- Alias targets for unified consumption ----
add_library(nnops::nnops ALIAS nnops)
add_library(nnops::cpu   ALIAS nnops_cpu)
