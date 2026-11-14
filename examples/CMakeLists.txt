# Metal backend examples (built only with NNOPS_BUILD_METAL).
add_executable(nnops_metal_demo metal_demo.mm)
target_link_libraries(nnops_metal_demo PRIVATE nnops_cpu)
target_compile_options(nnops_metal_demo PRIVATE
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
