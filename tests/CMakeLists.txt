# ---- Test sources ----
set(TEST_SOURCES
    common/test_harness.cpp
    test_tensor_view.cpp
    test_matmul.cpp
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
    test_flatten.cpp
    test_linear_attention.cpp
    test_transpose_conv2d.cpp
)

# Vulkan tests (only when Vulkan backend is enabled)
if(NNOPS_BUILD_VULKAN)
    find_package(Vulkan REQUIRED)
    list(APPEND TEST_SOURCES
        common/vulkan_test_helper.cpp
        test_eltwise_vulkan.cpp
        test_unary_vulkan.cpp
    )
endif()

add_executable(nnops_test ${TEST_SOURCES})
target_link_libraries(nnops_test PRIVATE nnops_cpu)
if(NNOPS_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(nnops_test PRIVATE CUDA::cudart)
endif()
if(NNOPS_BUILD_VULKAN)
    target_link_libraries(nnops_test PRIVATE Vulkan::Vulkan)
endif()
target_include_directories(nnops_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)

# Register with CTest
add_test(NAME nnops_test COMMAND nnops_test)

# ---- Benchmarks (optional) ----
if(NNOPS_BUILD_BENCHMARKS)
    # Benchmarks are optional; add benchmark .cpp files here when ready.
    # set(BENCH_SOURCES common/bench_harness.cpp bench_conv2d.cpp)
    # add_executable(nnops_bench ${BENCH_SOURCES})
    # target_link_libraries(nnops_bench PRIVATE nnops_cpu)
endif()
