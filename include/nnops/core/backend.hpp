#pragma once
/// @file backend.hpp
/// @brief Backend enum for selecting the compute backend.
///
/// The CPU backend pools operator scratch memory and reuses it across
/// invocations; release_scratch_memory() returns that pool to the OS. Live
/// blocks are never freed; the pool assumes no concurrent allocate/deallocate,
/// so call it between invocations — later allocations simply acquire fresh blocks
/// until the pool warms up again.

#include <cstdint>

namespace nnops {

/// Available compute backends.
enum class Backend : uint8_t {
    CPU    = 0,  ///< CPU reference or optimized implementation
    CUDA   = 1,  ///< NVIDIA CUDA GPU backend
    Vulkan = 2,  ///< Vulkan GPU backend
};

/// Force-release pooled operator scratch back to the OS.
void release_scratch_memory();

}  // namespace nnops
