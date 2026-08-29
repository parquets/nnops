#pragma once
/// @file compute_context.hpp
/// @brief ComputeContext — backend-specific execution parameters.

#include "nnops/core/parallel_for.hpp"
#include "nnops/core/backend.hpp"
#include <cstdint>

namespace nnops {

/// Execution context passed to operator compute() calls.
/// Each backend reads only the fields it cares about.
struct ComputeContext {
    /// Target backend for execution. Callers read this to select the backend
    /// when creating operators. Default: CPU.
    Backend expected_backend = Backend::CPU;

    /// CPU parallelism hook. nullptr means sequential execution.
    ParallelForFn cpu_parallel_for = nullptr;

    /// CUDA stream (cudaStream_t), opaque pointer.
    void* cuda_stream = nullptr;

    // ---- Vulkan backend fields ----
    /// VkCommandBuffer to record dispatch commands into.
    void* vulkan_cmd_buffer = nullptr;

    /// VkDevice handle, used for pipeline creation and descriptor set allocation.
    void* vulkan_device = nullptr;

    /// VkDescriptorPool for allocating per-call descriptor sets.
    /// Must be created with VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT.
    void* vulkan_descriptor_pool = nullptr;

    /// Array of VkBuffer handles, one per tensor (inputs first, then outputs).
    /// For eltwise: [A, B, output]; for unary: [input, output].
    const void* vulkan_buffers = nullptr;

    /// Number of entries in vulkan_buffers array.
    int vulkan_buffers_count = 0;
};

}  // namespace nnops
