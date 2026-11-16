#pragma once
/// @file compute_context.hpp
/// @brief ComputeContext — backend-specific execution parameters.
///
/// Each backend reads only the fields it cares about. Nothing in the library
/// dispatches on expected_backend — an operator is bound to its backend at
/// creation — so it is informational for callers and tests.
///
/// Vulkan: descriptor sets are one-shot. vulkan_record_dispatch allocates one
/// per call from vulkan_descriptor_pool and never frees it individually; the
/// caller reclaims them by resetting the pool once the submitted work has
/// completed. The pool must therefore outlive the command buffer. (A pool
/// created with VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT may instead
/// free sets individually.) vulkan_buffers holds one VkBuffer per tensor,
/// inputs first — [A, B, output] for eltwise, [input, output] for unary.

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

    /// Metal buffers (id<MTLBuffer> bridged to void*), inputs first,
    /// output last; the backend owns the device, queue and pipelines.
    const void* metal_buffers = nullptr;
    int metal_buffers_count = 0;
};

}  // namespace nnops
