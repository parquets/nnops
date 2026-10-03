# ---- Test sources (one standalone demo executable per file) ----
set(TEST_SOURCES
    test_tensor_view.cpp
    test_matmul.cpp
    test_matmul_int8.cpp
    test_conv2d.cpp
    test_activation.cpp
    test_pooling.cpp
    test_linear.cpp
    test_pack_mma.cpp
    test_attention.cpp
    test_causal_attention.cpp
    test_clamp.cpp
    test_softmax.cpp
    test_cumsum.cpp
    test_norm.cpp
    test_simd.cpp
    test_depthwise_conv.cpp
    test_eltwise.cpp
    test_embed.cpp
    test_permute.cpp
    test_slice.cpp
    test_unary.cpp
    test_reduce.cpp
    test_shape_inference.cpp
    test_resize.cpp
    test_grid_sample.cpp
    test_layout_convert.cpp
    test_quant_linear.cpp
    test_concat.cpp
    test_argminmax.cpp
    test_topk.cpp
    test_rope.cpp
    test_simd_unary.cpp
    test_simd_eltwise.cpp
    test_simd_add_fuse_norm.cpp
    test_flatten.cpp
    test_linear_attention.cpp
    test_transpose_conv2d.cpp
    test_cpu_backend.cpp
    test_memory_pool.cpp
)

# Each test file links against `common/test_harness.cpp` (which supplies main)
# and becomes its own runnable demo. Every demo is also registered with CTest.
function(nnops_add_test_demo src)
    get_filename_component(name "${src}" NAME_WE)   # strip .cpp → e.g. test_matmul
    add_executable(${name} "${src}" common/test_harness.cpp)
    target_link_libraries(${name} PRIVATE nnops_cpu)
    if(NNOPS_BUILD_CUDA)
        target_link_libraries(${name} PRIVATE CUDA::cudart)
    endif()
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

if(NNOPS_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

foreach(src ${TEST_SOURCES})
    nnops_add_test_demo(${src})
endforeach()

# ---- Vulkan tests (only when Vulkan backend is enabled) ----
if(NNOPS_BUILD_VULKAN)
    find_package(Vulkan REQUIRED)
    foreach(src test_eltwise_vulkan.cpp test_unary_vulkan.cpp)
        get_filename_component(name "${src}" NAME_WE)
        add_executable(${name} "${src}" common/test_harness.cpp common/vulkan_test_helper.cpp)
        target_link_libraries(${name} PRIVATE nnops_cpu Vulkan::Vulkan)
        target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)
        add_test(NAME ${name} COMMAND ${name})
    endforeach()
endif()

# ---- Benchmarks (optional) ----
if(NNOPS_BUILD_BENCHMARKS)
    set(BENCH_SOURCES common/bench_harness.cpp bench_matmul.cpp bench_conv2d.cpp bench_attention.cpp)
    add_executable(nnops_bench ${BENCH_SOURCES})
    target_link_libraries(nnops_bench PRIVATE nnops_cpu)
    target_include_directories(nnops_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)
endif()
