/// @file vulkan_common.hpp
/// @brief Common utilities for Vulkan compute kernels.
///
/// Provides:
///   - VulkanPipelineCache: lazy, thread-safe pipeline creation per VkDevice
///   - VulkanDescriptorSet: one-shot descriptor set allocation and binding
///   - Embedded SPIR-V shader data (generated at build time from .comp files)
///   - vk_check helper macro
///
/// Design:
///   - Pipelines are created once per (device, op_type, specialization) triple
///     and cached indefinitely (destroyed only when the cache is explicitly cleared).
///   - Descriptor sets are allocated per compute() call from a user-provided pool
///     and freed after command recording (VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT).
///   - Push constants carry runtime parameters (total element count).
///   - The user provides: VkDevice, VkDescriptorPool, VkCommandBuffer, VkBuffer handles
///     for each tensor via ComputeContext.

#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace nnops::backend::vulkan {

// ============================================================
// Constants
// ============================================================

constexpr int kVulkanWorkgroupSize = 256;

// ============================================================
// ceil_div
// ============================================================

template <typename T, typename U>
constexpr auto ceil_div(T a, U b) noexcept {
    return (a + b - 1) / b;
}

// ============================================================
// Embedded SPIR-V shader data
// ============================================================

/// Holds pre-compiled SPIR-V binary for a compute shader.
struct SpirvBlob {
    const uint32_t* data;
    size_t          size_bytes;
};

/// External declarations for embedded SPIR-V shaders.
/// These are generated at build time by compiling .comp → .spv and
/// converting to C arrays via CMake's file(READ ... HEX) mechanism.
/// The arrays are declared in the operator .cpp files.
///   - g_eltwise_f32_spv / g_eltwise_f32_spv_len
///   - g_unary_f32_spv  / g_unary_f32_spv_len

// ============================================================
// Push constant struct for eltwise and unary kernels
// ============================================================

struct ComputePushConstants {
    int32_t total;  // total number of elements to process
};

// ============================================================
// VkCheck helper macro
// ============================================================

inline void vk_check(VkResult res, const char* file, int line, const char* expr) {
    if (res != VK_SUCCESS) {
        fprintf(stderr, "[%s:%d] VK_CHECK(%s) failed: %d\n", file, line, expr, res);
        std::abort();
    }
}

#define VK_CHECK(expr) nnops::backend::vulkan::vk_check(expr, __FILE__, __LINE__, #expr)

// ============================================================
// VulkanPipeline — all state needed to dispatch a compute shader
// ============================================================

struct VulkanPipeline {
    VkPipeline            pipeline = VK_NULL_HANDLE;
    VkPipelineLayout      layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
};

// ============================================================
// Pipeline key — identifies a unique compute pipeline
// ============================================================

struct PipelineKey {
    VkDevice        device;
    const uint32_t* spirv_data;
    size_t          spirv_size;
    uint32_t        spec_op;      // specialization constant: OP type
    uint32_t        spec_add_to;  // specialization constant: ADD_TO flag
    uint32_t        num_bindings; // number of descriptor set bindings

    bool operator==(const PipelineKey& other) const noexcept {
        return device == other.device
            && spirv_data == other.spirv_data
            && spirv_size == other.spirv_size
            && spec_op == other.spec_op
            && spec_add_to == other.spec_add_to
            && num_bindings == other.num_bindings;
    }
};

struct PipelineKeyHash {
    size_t operator()(const PipelineKey& k) const noexcept {
        size_t h = reinterpret_cast<size_t>(k.device);
        h ^= reinterpret_cast<size_t>(k.spirv_data) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= k.spirv_size + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= (static_cast<size_t>(k.spec_op) << 16) | static_cast<size_t>(k.spec_add_to);
        h ^= static_cast<size_t>(k.num_bindings) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// ============================================================
// VulkanPipelineCache — global, thread-safe pipeline cache
// ============================================================

class VulkanPipelineCache {
public:
    static VulkanPipelineCache& instance() {
        static VulkanPipelineCache cache;
        return cache;
    }

    /// Get or create a compute pipeline.
    /// Returns VK_NULL_HANDLE on failure.
    VulkanPipeline get_or_create(const PipelineKey& key);

    /// Destroy all cached pipelines for a given device (call before vkDestroyDevice).
    void destroy_device_pipelines(VkDevice device);

    /// Destroy everything.
    void clear();

private:
    VulkanPipelineCache() = default;
    ~VulkanPipelineCache();
    VulkanPipelineCache(const VulkanPipelineCache&) = delete;
    VulkanPipelineCache& operator=(const VulkanPipelineCache&) = delete;

    VulkanPipeline create_pipeline(const PipelineKey& key);

    std::mutex mutex_;
    std::unordered_map<PipelineKey, VulkanPipeline, PipelineKeyHash> cache_;
};

// ============================================================
// Descriptor set helper — allocate, update, bind
// ============================================================

/// Allocate a descriptor set, update it with buffer handles, and bind it to
/// the command buffer along with push constants.
///
/// @param cmd              VkCommandBuffer to record into.
/// @param device           VkDevice handle.
/// @param pool             VkDescriptorPool (must have FREE_DESCRIPTOR_SET_BIT).
/// @param pipeline         The VulkanPipeline (layout + set_layout) to use.
/// @param buffers          Array of VkBuffer handles for each binding.
/// @param num_buffers      Number of buffer bindings.
/// @param push_constants   Push constant data.
/// @param pc_size          Size of push constant data in bytes.
void vulkan_record_dispatch(
    VkCommandBuffer        cmd,
    VkDevice               device,
    VkDescriptorPool       pool,
    const VulkanPipeline&  pipeline,
    const VkBuffer*        buffers,
    uint32_t               num_buffers,
    const void*            push_constants,
    uint32_t               pc_size,
    uint32_t               total_elements);

}  // namespace nnops::backend::vulkan
