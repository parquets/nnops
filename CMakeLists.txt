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
option(NNOPS_OPENMP           "Build the OpenMP CpuBackend adapter" OFF)

# ---- C++ Standard ----
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# MSVC: use UTF-8 source encoding, no deprecation warnings for std::span use.
# int8 VNNI: MSVC has no /arch:AVXVNNI switch — the AVX-VNNI intrinsics are
# declared unconditionally in immintrin.h, so we keep /arch:AVX2 (for the AVX2
# SIMD layer) and manually define __AVXVNNI__ to select our VNNI micro-kernels.
# The target always has VNNI, so no runtime check is needed.
# NOTE: scoped to CXX only — CUDA/nvcc does not understand MSVC flags
if(MSVC)
    # /Zc:inline drops COMDATs a TU never references — a static lib carries every
    # one of them, so this alone is ~3x smaller objects.
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/bigobj> $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2> $<$<COMPILE_LANGUAGE:CXX>:/D__AVXVNNI__> $<$<COMPILE_LANGUAGE:CXX>:/Zc:inline>)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)|(amd64)|(i686)")
    add_compile_options(-mavx2 -mfma -mf16c -mavxvnni)
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

# ---- OpenMP (optional) ----
# The library owns no thread pool (CpuBackend is three hooks); this option only
# decides whether nnops/core/openmp_backend.hpp has an OpenMP runtime to bind
# its parallel_for to. Linking it INTERFACE is what gives consumers the
# -fopenmp / /openmp flag that defines _OPENMP in their translation units, so an
# installed build without the option still compiles the header as a sequential
# no-op.
if(NNOPS_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries(nnops INTERFACE OpenMP::OpenMP_CXX)
endif()

# ---- Source subdirectories ----
add_subdirectory(src)

# ---- Installation ----
include(GNUInstallDirs)
set(NNOPS_INSTALL_TARGETS nnops nnops_cpu)
if(NNOPS_BUILD_CUDA)
    list(APPEND NNOPS_INSTALL_TARGETS nnops_cuda)
endif()
if(NNOPS_BUILD_VULKAN)
    list(APPEND NNOPS_INSTALL_TARGETS nnops_vulkan)
endif()
install(TARGETS ${NNOPS_INSTALL_TARGETS} EXPORT nnopsTargets)
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
