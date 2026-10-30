#pragma once
/// @file compute_context.hpp
/// @brief ComputeContext — backend-specific execution parameters.
///
/// Each backend reads only the fields it cares about. Nothing in the library
/// dispatches on expected_backend — an operator is bound to its backend at
/// creation — so it is informational for callers and tests.
///
/// Vulkan: vulkan_descriptor_pool must be created with
/// VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, since per-call descriptor
/// sets are allocated from it and freed. vulkan_buffers holds one VkBuffer per
/// tensor, inputs first — [A, B, output] for eltwise, [input, output] for unary.

#include "nnops/core/parallel_for.hpp"
#include "nnops/core/backend.hpp"
#include <cstdint>

namespace nnops {

/// Execution context passed to operator compute() calls.
struct ComputeContext {
    /// Backend this context was prepared for (informational).
    Backend expected_backend = Backend::CPU;

    /// CPU hooks: parallelism, worker count, thread id.
    CpuBackend cpu;

    void* cuda_stream = nullptr;  ///< cudaStream_t, opaque.

    // ---- Vulkan backend fields ----
    void* vulkan_cmd_buffer = nullptr;  ///< VkCommandBuffer.

    void* vulkan_device = nullptr;  ///< VkDevice.

    void* vulkan_descriptor_pool = nullptr;  ///< VkDescriptorPool; see header.

    const void* vulkan_buffers = nullptr;  ///< VkBuffer handles, inputs first; see header.

    int vulkan_buffers_count = 0;  ///< Entries in vulkan_buffers.
};

}  // namespace nnops
