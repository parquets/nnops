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
    ops/depthwise_conv.cpp
    ops/eltwise.cpp
    ops/unary.cpp
    ops/reduce.cpp
    ops/resize.cpp
    ops/grid_sample.cpp
    ops/layout_convert.cpp
    ops/quant_linear.cpp
    ops/concat.cpp
    ops/argminmax.cpp
    ops/topk.cpp

    # CPU feature detection (runtime CPUID / NEON probing)
    detail/cpu_features.cpp

    # Layout conversion utilities (pack/unpack between NCHW and NCHWC8)
    backend/cpu/layout_convert.cpp

    # CPU optimized kernels (SIMD-accelerated)
    backend/cpu/activation.cpp
    backend/cpu/batch_norm.cpp
    backend/cpu/depthwise_conv.cpp
    backend/cpu/layer_norm.cpp
    backend/cpu/pooling.cpp
    backend/cpu/rms_norm.cpp
    backend/cpu/softmax.cpp
    backend/cpu/eltwise.cpp
    backend/cpu/unary.cpp
    backend/cpu/reduce.cpp
    backend/cpu/resize.cpp
    backend/cpu/grid_sample.cpp
    backend/cpu/quant_linear.cpp
    backend/cpu/concat.cpp
    backend/cpu/argminmax.cpp
    backend/cpu/topk.cpp

    # CPU tiled matmul kernel (NKM loop with pack + MMA)
    backend/cpu/matmul.cpp
    backend/cpu/matmul_impl.cpp

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
    backend/cpu/reference/depthwise_conv_ref.cpp
    backend/cpu/reference/eltwise_ref.cpp
    backend/cpu/reference/unary_ref.cpp
    backend/cpu/reference/reduce_ref.cpp
    backend/cpu/reference/resize_ref.cpp
    backend/cpu/reference/quant_linear_ref.cpp
    backend/cpu/reference/concat_ref.cpp
    backend/cpu/reference/argminmax_ref.cpp
    backend/cpu/reference/topk_ref.cpp
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
