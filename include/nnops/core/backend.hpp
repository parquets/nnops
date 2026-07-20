#pragma once
/// @file backend.hpp
/// @brief Backend enum for selecting the compute backend.

#include <cstdint>

namespace nnops {

/// Available compute backends.
enum class Backend : uint8_t {
    CPU    = 0,  ///< CPU reference or optimized implementation
    CUDA   = 1,  ///< NVIDIA CUDA GPU backend
    Vulkan = 2,  ///< Vulkan GPU backend
};

}  // namespace nnops
