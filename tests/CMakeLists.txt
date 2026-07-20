# ---- Test sources ----
set(TEST_SOURCES
    common/test_harness.cpp
    test_tensor_view.cpp
    test_conv2d.cpp
    test_activation.cpp
    test_pooling.cpp
    test_linear.cpp
    test_matmul.cpp
    test_attention.cpp
    test_simd.cpp
)

add_executable(nnops_test ${TEST_SOURCES})
target_link_libraries(nnops_test PRIVATE nnops_cpu)
target_include_directories(nnops_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Register with CTest
add_test(NAME nnops_test COMMAND nnops_test)

# ---- Benchmarks (optional) ----
if(NNOPS_BUILD_BENCHMARKS)
    # Benchmarks are optional; add benchmark .cpp files here when ready.
    # set(BENCH_SOURCES common/bench_harness.cpp bench_conv2d.cpp)
    # add_executable(nnops_bench ${BENCH_SOURCES})
    # target_link_libraries(nnops_bench PRIVATE nnops_cpu)
endif()
