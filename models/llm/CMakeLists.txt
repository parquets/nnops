# Fake-weight Qwen2/Qwen2.5 inference demo. The forward pass is composed from
# the public operator headers; nothing here is part of the library.
add_executable(nnops_qwen2_llm
    main.cpp
    qwen2_model.cpp
)
target_link_libraries(nnops_qwen2_llm PRIVATE nnops_cpu)
# nnops_cpu pulls in nnops_cuda and NNOPS_HAS_CUDA when the backend is on; the
# demo's own staging copies call the runtime directly.
if(NNOPS_BUILD_CUDA)
    # tests/ is the only directory that finds the toolkit, and an imported
    # target is not visible outside it.
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(nnops_qwen2_llm PRIVATE CUDA::cudart)
endif()
target_include_directories(nnops_qwen2_llm PRIVATE ${CMAKE_SOURCE_DIR})

# The demo carries its own prefill-vs-stepwise check, so a tiny run is the
# LLM-side regression test: q8 cache, RoPE positions and both attention routes.
if(NNOPS_BUILD_TESTS)
    add_test(NAME llm_qwen2_selftest
             COMMAND nnops_qwen2_llm --preset tiny --dtype f16 --prompt 8
                     --steps 2 --selftest --threads 4)
endif()
