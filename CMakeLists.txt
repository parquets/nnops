cmake_minimum_required(VERSION 3.21)
project(nnops
    VERSION 0.1.0
    DESCRIPTION "Multi-backend operator library for neural networks"
    LANGUAGES CXX
)

# ---- Options ----
option(NNOPS_BUILD_TESTS      "Build unit tests"           ON)
option(NNOPS_BUILD_BENCHMARKS "Build benchmarks"           ON)
option(NNOPS_BUILD_CUDA       "Build CUDA backend"         OFF)
option(NNOPS_BUILD_VULKAN     "Build Vulkan backend"       OFF)

# ---- C++ Standard ----
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# MSVC: use UTF-8 source encoding, no deprecation warnings for std::span use
if(MSVC)
    add_compile_options(/utf-8 /bigobj)
endif()

# ---- Export compile commands (for IDE support) ----
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Main library target (header-only interface) ----
add_library(nnops INTERFACE)
target_include_directories(nnops INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(nnops INTERFACE cxx_std_23)

# ---- CPU implementation library (always built) ----
add_library(nnops_cpu STATIC)
target_link_libraries(nnops_cpu PUBLIC nnops)

# Gather CPU sources explicitly (no file(GLOB))
set(NNOPS_CPU_SOURCES
    # Operator dispatch layer
    src/ops/conv2d.cpp
    src/ops/activation.cpp
    src/ops/pooling.cpp
    src/ops/linear.cpp
    src/ops/matmul.cpp
    src/ops/attention.cpp

    # CPU feature detection (runtime CPUID / NEON probing)
    src/detail/cpu_features.cpp

    # CPU reference kernels (always compiled — correctness baseline)
    src/backend/cpu/reference/conv2d_ref.cpp
    src/backend/cpu/reference/activation_ref.cpp
    src/backend/cpu/reference/pooling_ref.cpp
    src/backend/cpu/reference/linear_ref.cpp
    src/backend/cpu/reference/matmul_ref.cpp
    src/backend/cpu/reference/attention_ref.cpp
)

target_sources(nnops_cpu PRIVATE ${NNOPS_CPU_SOURCES})

# ---- CUDA backend (optional) ----
if(NNOPS_BUILD_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    set(NNOPS_CUDA_SOURCES
        src/backend/cuda/conv2d_cuda.cu
        src/backend/cuda/activation_cuda.cu
        src/backend/cuda/pooling_cuda.cu
    )

    add_library(nnops_cuda STATIC ${NNOPS_CUDA_SOURCES})
    target_link_libraries(nnops_cuda PUBLIC nnops)
    target_include_directories(nnops_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(nnops_cpu PUBLIC NNOPS_HAS_CUDA)
    target_link_libraries(nnops_cpu PUBLIC nnops_cuda)
endif()

# ---- Vulkan backend (optional) ----
if(NNOPS_BUILD_VULKAN)
    find_package(Vulkan REQUIRED)

    set(NNOPS_VULKAN_SOURCES
        src/backend/vulkan/conv2d_vulkan.cpp
        src/backend/vulkan/activation_vulkan.cpp
        src/backend/vulkan/pooling_vulkan.cpp
    )

    add_library(nnops_vulkan STATIC ${NNOPS_VULKAN_SOURCES})
    target_link_libraries(nnops_vulkan PUBLIC nnops Vulkan::Vulkan)
    target_include_directories(nnops_vulkan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(nnops_cpu PUBLIC NNOPS_HAS_VULKAN)
    target_link_libraries(nnops_cpu PUBLIC nnops_vulkan)
endif()

# ---- Alias targets for unified consumption ----
add_library(nnops::nnops ALIAS nnops)
add_library(nnops::cpu   ALIAS nnops_cpu)

# ---- Installation ----
include(GNUInstallDirs)
install(TARGETS nnops nnops_cpu EXPORT nnopsTargets)
install(EXPORT nnopsTargets
    NAMESPACE nnops::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nnops
)
install(DIRECTORY include/nnops DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# ---- Tests ----
if(NNOPS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
