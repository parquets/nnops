# ---- End-to-end model tests ----
#
# Each runner rebuilds a torchvision network out of nnops operators and checks
# it layer by layer against the activations PyTorch recorded. The checkpoints
# live in models/data/<model>/, which is generated (and not committed) — see
# models/README.md. A runner whose artifact is missing exits 77, which CTest
# reports as skipped rather than failed.

set(MODEL_SOURCES
    resnet18.cpp
    mobilenetv2.cpp
)

foreach(src ${MODEL_SOURCES})
    get_filename_component(name "${src}" NAME_WE)   # e.g. resnet18
    add_executable(model_${name} "${src}")
    target_link_libraries(model_${name} PRIVATE nnops_cpu)
    target_include_directories(model_${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(model_${name} PRIVATE
        NNOPS_MODEL_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
    add_test(NAME model_${name} COMMAND model_${name})
    set_tests_properties(model_${name} PROPERTIES SKIP_RETURN_CODE 77)
    # The same graph with f16 activations and weights. The artifact stays f32
    # (its ref.* tensors are the PyTorch checkpoints); the runner converts the
    # tensors the graph consumes and uses the f16 tolerance it defaults to.
    add_test(NAME model_${name}_f16 COMMAND model_${name} --f16)
    set_tests_properties(model_${name}_f16 PROPERTIES SKIP_RETURN_CODE 77)
endforeach()

# The LLM demo: a fake-weight Qwen2 decoder built from the same operators.
# Unlike the torchvision runners it needs no exported artifact — it checks
# itself with --selftest, which CTest runs.
add_subdirectory(llm)
