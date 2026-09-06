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

/// Force-release all operator scratch memory currently pooled by the CPU
/// backend back to the OS. The CPU backend keeps a small internal pool of
/// operator scratch memory that is reused across invocations; this reclaims it
/// (e.g. at a checkpoint, or to return memory to the system after a burst of
/// inference). Safe to call at any time — memory handed out to an in-flight
/// operator is untouched. Subsequent allocations simply acquire fresh blocks
/// from the OS until the pool warms up again.
void release_scratch_memory();

}  // namespace nnops
