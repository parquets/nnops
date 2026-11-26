# ---- CPU implementation library (always built) ----
add_library(nnops_cpu STATIC)
target_link_libraries(nnops_cpu PUBLIC nnops)

# Gather CPU sources explicitly (no file(GLOB))
set(NNOPS_CPU_SOURCES
    # Operator dispatch layer
    ops/clamp.cpp
    ops/cast.cpp
    ops/conv2d.cpp
    ops/activation.cpp
    ops/pooling.cpp
    ops/linear.cpp
    ops/matmul.cpp
    ops/softmax.cpp
    ops/cumsum.cpp
    ops/norm.cpp
    ops/conv3d.cpp
    ops/depthwise_conv.cpp
    ops/depthwise_transpose_conv.cpp
    ops/eltwise.cpp
    ops/embed.cpp
    ops/permute.cpp
    ops/slice.cpp
    ops/unary.cpp
    ops/reduce.cpp
    ops/resize.cpp
    ops/grid_sample.cpp
    ops/linear_attention.cpp
    ops/group_query_attention.cpp
    ops/multi_latent_attention.cpp
    ops/quant_linear.cpp
    ops/concat.cpp
    ops/argminmax.cpp
    ops/topk.cpp
    ops/rope.cpp
    ops/flatten.cpp
    ops/transpose_conv2d.cpp
    ops/transpose_conv3d.cpp
    ops/conv1d.cpp
    ops/pool1d.cpp
    ops/moe.cpp
    ops/rnn.cpp
    ops/gru.cpp
    ops/lstm.cpp
    ops/pad.cpp
    ops/gather.cpp
    ops/scatter.cpp
    ops/split.cpp
    ops/tile.cpp
    ops/where.cpp
    ops/compare.cpp
    ops/roi_align.cpp
    ops/non_max_suppression.cpp
    ops/pixel_shuffle.cpp

    # CPU feature detection (runtime CPUID / NEON probing)
    detail/cpu_features.cpp

    # Operator scratch-memory pool (shared, thread-safe singleton)
    backend/cpu/common/memory_pool.cpp

    # CPU optimized kernels (SIMD-accelerated)
    backend/cpu/activation.cpp
    backend/cpu/clamp.cpp
    backend/cpu/cast.cpp
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
    backend/cpu/pad.cpp
    backend/cpu/gather.cpp
    backend/cpu/scatter.cpp
    backend/cpu/split.cpp
    backend/cpu/tile.cpp
    backend/cpu/where.cpp
    backend/cpu/compare.cpp
    backend/cpu/roi_align.cpp
    backend/cpu/non_max_suppression.cpp
    backend/cpu/pixel_shuffle.cpp

    # CPU tiled matmul kernel (NKM loop with pack + MMA), split one file per
    # arithmetic family — the float routes with the classifier, the quantized
    # routes — plus the routes that bypass the plan (small naive / GEMV, fused
    # Q4_1). matmul_helper is the shared pack/MMA layer that conv2d and
    # attention also drive.
    backend/cpu/matmul/matmul.cpp
    backend/cpu/matmul/matmul_i8.cpp
    backend/cpu/matmul/matmul_helper_i8.cpp
    backend/cpu/matmul/matmul_helper.cpp
    backend/cpu/matmul/matmul_small.cpp
    backend/cpu/matmul/matmul_qblock_fused.cpp

    # Linear is the MatMul GEMM with a transposed weight, a per-column bias and
    # an fp output for quantized operands; it drives matmul.cpp above. MoE is
    # the routed FFN on top of the same GEMM — per-expert MatMul calls, one
    # output-channel block per task — so it lives with them.
    backend/cpu/matmul/linear.cpp
    backend/cpu/matmul/moe.cpp

    # CPU convolution kernels: the implicit-GEMM route (NHWC) split one file per
    # arithmetic family — the float route with the plan, scratch layout and
    # driver both families share, and the quantized route with the s8 panels'
    # prepack — then the Winograd F(2x2,3x3) route, the depthwise kernel and
    # pooling (which sweeps the same windows the two above do, and shares their
    # channel-lane / width-lane split).
    backend/cpu/conv/conv2d_implicit_gemm.cpp
    backend/cpu/conv/conv2d_implicit_gemm_i8.cpp
    backend/cpu/conv/conv2d_winograd.cpp
    backend/cpu/conv/depthwise_conv.cpp
    backend/cpu/conv/depthwise_conv_i8.cpp
    backend/cpu/conv/depthwise_transpose_conv.cpp
    backend/cpu/conv/depthwise_transpose_conv_i8.cpp
    backend/cpu/conv/transpose_conv2d.cpp
    backend/cpu/conv/transpose_conv2d_i8.cpp
    backend/cpu/conv/transpose_conv3d.cpp
    backend/cpu/conv/pooling.cpp
    backend/cpu/conv/pooling_i8.cpp
    backend/cpu/conv/conv3d_implicit_gemm.cpp
    backend/cpu/conv/conv3d_implicit_gemm_i8.cpp

    # Conv1D is the 2D GEMM kernel on a rank-4 lift (conv_util.h), so this
    # file only translates attributes; Pool1D lives with pooling.cpp.
    backend/cpu/conv/conv1d.cpp

    # CPU tiled attention kernels (fused QK^T + softmax + AV) — one core shared
    # by GroupQueryAttention (which covers ordinary MHA at H_kv == H) and
    # MultiLatentAttention, split one file per route (standard / flash /
    # decode), with the plan that picks between them in attention_plan.cpp;
    # linear attention lives alongside it.
    backend/cpu/attention/attention.cpp
    backend/cpu/attention/attention_plan.cpp
    backend/cpu/attention/attention_standard.cpp
    backend/cpu/attention/attention_flash.cpp
    backend/cpu/attention/attention_decode.cpp
    backend/cpu/attention/linear_attention.cpp

    # The recurrent family: one driver for RNN, GRU and LSTM — split into the
    # plan (tile grid, chunk, q8 scratch), the packed-weight blob, the fp step
    # kernels and their int8 twin; recurrent.cpp walks the timeline.
    backend/cpu/rnn/recurrent.cpp
    backend/cpu/rnn/recurrent_plan.cpp
    backend/cpu/rnn/recurrent_pack.cpp
    backend/cpu/rnn/recurrent_step.cpp
    backend/cpu/rnn/recurrent_q8.cpp

    backend/cpu/cumsum.cpp
)

target_sources(nnops_cpu PRIVATE ${NNOPS_CPU_SOURCES})
target_include_directories(nnops_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(NNOPS_ENABLE_SME)
    # The kernels are header-only and carry their own target("+sme2") attribute,
    # so the library stays baseline AArch64 and only the emitted bodies use SME.
    target_compile_definitions(nnops_cpu PUBLIC NNOPS_ARCH_SME=1)
endif()

# ---- CUDA backend (optional) ----
if(NNOPS_BUILD_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80)
    endif()

    set(NNOPS_CUDA_SOURCES
        backend/cuda/softmax.cu
        backend/cuda/activation.cu
        backend/cuda/cumsum.cu
        backend/cuda/eltwise.cu
        backend/cuda/unary.cu
        backend/cuda/reduce.cu
        backend/cuda/argminmax.cu
        backend/cuda/resize.cu
        backend/cuda/norm.cu
        backend/cuda/topk.cu
        # The pure-movement family: an index walk with no arithmetic in it, one
        # file per operator over the layout copy engine in move_common.cuh.
        backend/cuda/permute.cu
        backend/cuda/slice.cu
        backend/cuda/tile.cu
        backend/cuda/concat.cu
        backend/cuda/split.cu
        backend/cuda/pad.cu
        # Index-driven movement, where a layout cannot describe the mapping:
        # one index lookup per row or per element.
        backend/cuda/gather.cu
        backend/cuda/scatter.cu
        # Convolutions and the pooling that goes with them, one file an
        # operator family: Conv2D (the PTX implicit GEMM, split one file per
        # arithmetic family with the shared geometry in conv2d_common.cuh),
        # depthwise, and spatial pooling.
        backend/cuda/matmul/matmul.cu
        backend/cuda/matmul/linear.cu
        backend/cuda/matmul/matmul_f16.cu
        backend/cuda/matmul/matmul_f16_tail.cu
        backend/cuda/matmul/matmul_bf16.cu
        # w4a16: the same 16-bit tile with the weight panel decoded from a
        # llama.cpp Q4_0/Q4_1/Q8_0 block stream.
        backend/cuda/matmul/matmul_q4.cu
        backend/cuda/matmul/matmul_f32.cu
        backend/cuda/matmul/matmul_i8_ptx.cu
        backend/cuda/conv/conv2d.cu
        backend/cuda/conv/conv2d_i8.cu
        backend/cuda/conv/transpose_conv2d.cu
        backend/cuda/conv/depthwise_transpose_conv.cu
        backend/cuda/conv/depthwise_conv.cu
        backend/cuda/conv/pooling.cu
        # The recurrent family: one timeline driver shared by RNN, GRU and
        # LSTM — the chunked input projection and the per-step product both run
        # on the dense GEMM engine, and one elementwise kernel does the gates.
        backend/cuda/rnn/recurrent.cu
        # Attention: one tiled online-softmax kernel behind GroupQueryAttention
        # (MHA at H_kv == H) and MultiLatentAttention, both forms — see
        # backend/common/attention_config.hpp for the shared description.
        backend/cuda/attention/attention.cu
        # The general elementwise and movement family: conversions, clamp,
        # compare/logical, where, flatten, the pixel-shuffle pair, embedding
        # lookup and the quantize/dequantize pair.
        backend/cuda/cast.cu
        backend/cuda/clamp.cu
        backend/cuda/compare.cu
        backend/cuda/where.cu
        backend/cuda/flatten.cu
        backend/cuda/pixel_shuffle.cu
        backend/cuda/embed.cu
        backend/cuda/quant_linear.cu
        backend/cuda/grid_sample.cu
        backend/cuda/roi_align.cu
        backend/cuda/rope.cu
        backend/cuda/non_max_suppression.cu
        # Unsupported operators' entry points, so the CUDA library resolves every
        # dispatch site the CPU backend references.
        backend/cuda/not_implemented.cu
    )

    # The elementwise/movement translation units are small and cicc fails to
    # read its own intermediate bitcode for them under the target-wide
    # --split-compile=0 (the flag that keeps the fat GEMM TU tractable). They
    # take split-compile=1 instead; it is a compile-time knob only. Every TU
    # that inlines the movement odometer (move_common.cuh) needs it too.
    set_source_files_properties(
        backend/cuda/cast.cu
        backend/cuda/compare.cu
        backend/cuda/where.cu
        backend/cuda/quant_linear.cu
        backend/cuda/grid_sample.cu
        backend/cuda/roi_align.cu
        backend/cuda/rope.cu
        backend/cuda/non_max_suppression.cu
        backend/cuda/permute.cu
        backend/cuda/slice.cu
        backend/cuda/tile.cu
        backend/cuda/concat.cu
        backend/cuda/split.cu
        backend/cuda/flatten.cu
        backend/cuda/gather.cu
        backend/cuda/pad.cu
        backend/cuda/scatter.cu
        PROPERTIES COMPILE_OPTIONS "--split-compile=1")

    add_library(nnops_cuda STATIC ${NNOPS_CUDA_SOURCES})
    target_link_libraries(nnops_cuda PUBLIC nnops)
    target_include_directories(nnops_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include)
    target_compile_options(nnops_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>)
    # A fat translation unit -- matmul_i8_ptx.cu carries three tiles x four poses
    # x two element types x two destination families -- spends nearly all of its
    # wall clock in cicc's optimizer, on one core. This lets those phases spread
    # over the machine; nvcc documents it as having no effect on the emitted code
    # (`--threads` does nothing here: it is ignored for a single architecture,
    # which is all we generate).
    target_compile_options(nnops_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--split-compile=0>)
    # The sources are UTF-8 (the comments use ×, −, ² …); without this the host
    # pass reads them as the ANSI code page and prints a C4819 per file per
    # translation unit — thousands of warnings, and a real hazard for a byte
    # that ends in a backslash.
    if(MSVC)
        target_compile_options(nnops_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
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

# ---- Metal backend (Apple only) ----
if(NNOPS_BUILD_METAL)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

    set(NNOPS_METAL_SOURCES
        backend/metal/metal_common.mm
        backend/metal/unary.mm
        backend/metal/eltwise.mm
    )
    add_library(nnops_metal STATIC ${NNOPS_METAL_SOURCES})
    target_link_libraries(nnops_metal PUBLIC nnops
        ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    target_include_directories(nnops_metal PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
    )
    target_compile_options(nnops_metal PRIVATE
        $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)

    # The op dispatch layer lives in nnops_cpu and resolves the Metal entry
    # points; the definition is PUBLIC so the dispatch table is compiled with
    # them and consumers inherit the link.
    target_compile_definitions(nnops_cpu PUBLIC NNOPS_HAS_METAL)
    target_link_libraries(nnops_cpu PUBLIC nnops_metal)
endif()

# ---- Alias targets for unified consumption ----
add_library(nnops::nnops ALIAS nnops)
add_library(nnops::cpu   ALIAS nnops_cpu)
