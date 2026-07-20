#pragma once
/// @file compute_context.hpp
/// @brief ComputeContext — backend-specific execution parameters.

#include "nnops/core/parallel_for.hpp"
#include <cstdint>

namespace nnops {

/// Execution context passed to operator compute() calls.
/// Each backend reads only the fields it cares about.
struct ComputeContext {
    /// CPU parallelism hook. nullptr means sequential execution.
    ParallelForFn cpu_parallel_for = nullptr;

    /// CUDA stream (cudaStream_t), opaque pointer.
    void* cuda_stream = nullptr;

    /// Vulkan command buffer (VkCommandBuffer), opaque pointer.
    void* vulkan_cmd_buffer = nullptr;
};

}  // namespace nnops
