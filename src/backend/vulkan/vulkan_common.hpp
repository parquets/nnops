#pragma once
/// @file vulkan_common.hpp
/// @brief Vulkan compute plumbing shared by the backend's operators.
///
/// A pipeline is cached for the process lifetime under (device, SPIR-V blob,
/// specialization constants, binding count), since building one is expensive.
/// Descriptor sets are one-shot: vulkan_record_dispatch allocates one
/// per call from a caller-owned VkDescriptorPool, so ComputeContext requires
/// that pool to carry VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT. Every
/// binding is a whole storage buffer; the only runtime parameter is the element
/// count, pushed as a ComputePushConstants. The caller supplies the device, pool,
/// command buffer and per-tensor VkBuffers through ComputeContext.
///
/// The SPIR-V blobs are generated at build time: glslc compiles each .comp and
/// spv_to_header.py emits a `<name>_spv.h` with `g_<name>_spv[]` and
/// `g_<name>_spv_len`, which the operator .cpp files include.

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace nnops::backend::vulkan {

constexpr int kVulkanWorkgroupSize = 256;

template <typename T, typename U>
constexpr auto ceil_div(T a, U b) noexcept {
    return (a + b - 1) / b;
}

/// Holds pre-compiled SPIR-V binary for a compute shader.
struct SpirvBlob {
    const uint32_t* data;
    size_t          size_bytes;
};

struct ComputePushConstants {
    int32_t total;  // total number of elements to process
};

inline void vk_check(VkResult res, const char* file, int line, const char* expr) {
    if (res != VK_SUCCESS) {
        fprintf(stderr, "[%s:%d] VK_CHECK(%s) failed: %d\n", file, line, expr, res);
        std::abort();
    }
}

#define VK_CHECK(expr) nnops::backend::vulkan::vk_check(expr, __FILE__, __LINE__, #expr)

struct VulkanPipeline {
    VkPipeline            pipeline = VK_NULL_HANDLE;
    VkPipelineLayout      layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
};

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

class VulkanPipelineCache {
public:
    static VulkanPipelineCache& instance() {
        static VulkanPipelineCache cache;
        return cache;
    }

    /// Get or create a compute pipeline; VK_NULL_HANDLE on failure.
    VulkanPipeline get_or_create(const PipelineKey& key);

    /// Destroy all cached pipelines for a given device (call before vkDestroyDevice).
    void destroy_device_pipelines(VkDevice device);

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

/// Allocate a descriptor set from @p pool, bind @p buffers with @p
/// push_constants, and dispatch over @p total_elements.
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
