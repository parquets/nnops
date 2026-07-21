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
# Also enable AVX2+F16C (assumed baseline on x86_64 target platforms)
if(MSVC)
    add_compile_options(/utf-8 /bigobj /arch:AVX2)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)|(amd64)|(i686)")
    add_compile_options(-mavx2 -mfma -mf16c)
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

# ---- Source subdirectories ----
add_subdirectory(src)

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
