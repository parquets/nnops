/// @file vulkan_test_helper.hpp
/// @brief Minimal Vulkan test infrastructure for operator testing.
///
/// Creates a self-contained Vulkan compute environment:
///   - Instance, device, compute queue
///   - Command pool and descriptor pool
///   - Buffer creation / host↔device transfer helpers
///
/// Designed for correctness tests: create device buffers, run Vulkan operators,
/// read back results, and compare against CPU reference.
///
/// All resources are destroyed on destruction. Use only when NNOPS_HAS_VULKAN
/// is defined.

#pragma once

#ifdef NNOPS_HAS_VULKAN

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace nnops::test {

/// Opaque helper to clean up Vulkan pipeline cache for a device.
/// Implemented in vulkan_test_helper.cpp to avoid including
/// backend/vulkan/vulkan_common.hpp from this header.
void vulkan_pipeline_cache_cleanup(VkDevice device);

// ============================================================
// RAII Vulkan test environment
// ============================================================

class VulkanTestEnv {
public:
    VulkanTestEnv() {
        create_instance();
        pick_physical_device();
        create_device();
        create_command_pool();
        create_descriptor_pool();
    }

    ~VulkanTestEnv() {
        if (device_ != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device_);
            // Pipeline cache cleanup is handled by vulkan_common.cpp.
            // Forward-call through an opaque helper to avoid including
            // vulkan_common.hpp from this public test header.
            vulkan_pipeline_cache_cleanup(device_);
        }
        if (desc_pool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device_, desc_pool_, nullptr);
        }
        if (cmd_pool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, cmd_pool_, nullptr);
        }
        if (device_ != VK_NULL_HANDLE) {
            vkDestroyDevice(device_, nullptr);
        }
        if (instance_ != VK_NULL_HANDLE) {
            vkDestroyInstance(instance_, nullptr);
        }
    }

    VkDevice         device()          const { return device_; }
    VkCommandBuffer  cmd_buffer()      const { return cmd_buf_; }
    VkDescriptorPool descriptor_pool() const { return desc_pool_; }
    VkQueue          queue()           const { return queue_; }
    VkCommandPool    cmd_pool()        const { return cmd_pool_; }
    bool             supports_fp16()   const { return supports_16bit_storage_; }

    // ---- Buffer helpers ----

    /// Create a storage buffer and allocate+map device memory.
    /// Returns VkBuffer + VkDeviceMemory pair.
    struct Buffer {
        VkBuffer       buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        size_t         size   = 0;
        void*          mapped = nullptr;
    };

    Buffer create_buffer(size_t size_bytes, VkBufferUsageFlags usage) {
        Buffer buf;
        buf.size = size_bytes;

        VkBufferCreateInfo bci = {};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size_bytes;
        bci.usage = usage | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                         | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                         | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VK_CHECK(vkCreateBuffer(device_, &bci, nullptr, &buf.buffer));

        // Allocate memory
        VkMemoryRequirements mem_req;
        vkGetBufferMemoryRequirements(device_, buf.buffer, &mem_req);

        uint32_t mem_type = find_memory_type(
            mem_req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        VkMemoryAllocateInfo mai = {};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = mem_req.size;
        mai.memoryTypeIndex = mem_type;

        VK_CHECK(vkAllocateMemory(device_, &mai, nullptr, &buf.memory));
        VK_CHECK(vkBindBufferMemory(device_, buf.buffer, buf.memory, 0));
        VK_CHECK(vkMapMemory(device_, buf.memory, 0, size_bytes, 0, &buf.mapped));

        return buf;
    }

    void destroy_buffer(Buffer& buf) {
        if (buf.mapped) vkUnmapMemory(device_, buf.memory);
        if (buf.buffer) vkDestroyBuffer(device_, buf.buffer, nullptr);
        if (buf.memory) vkFreeMemory(device_, buf.memory, nullptr);
        buf = {};
    }

    /// Copy host data to device buffer (blocking).
    void copy_to_device(Buffer& buf, const void* data, size_t size) {
        memcpy(buf.mapped, data, size);
        // Host-coherent memory: no explicit flush needed for reads by GPU
        // But for correctness, add a memory barrier
        VkMappedMemoryRange range = {};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = buf.memory;
        range.offset = 0;
        range.size = VK_WHOLE_SIZE;
        vkFlushMappedMemoryRanges(device_, 1, &range);
    }

    /// Copy device buffer data to host (blocking).
    void copy_from_device(Buffer& buf, void* data, size_t size) {
        VkMappedMemoryRange range = {};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = buf.memory;
        range.offset = 0;
        range.size = VK_WHOLE_SIZE;
        vkInvalidateMappedMemoryRanges(device_, 1, &range);
        memcpy(data, buf.mapped, size);
    }

    // ---- Command submission ----

    /// Begin recording into a fresh command buffer.
    VkCommandBuffer begin_cmd() {
        VkCommandBufferAllocateInfo ai = {};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = cmd_pool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(device_, &ai, &cmd_buf_));

        VkCommandBufferBeginInfo bi = {};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd_buf_, &bi));

        return cmd_buf_;
    }

    /// End recording, submit, wait, free the command buffer, and reset the
    /// descriptor pool so that subsequent dispatches can reuse it.
    void submit_and_wait() {
        VK_CHECK(vkEndCommandBuffer(cmd_buf_));

        VkSubmitInfo si = {};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_buf_;

        VK_CHECK(vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(queue_));

        vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd_buf_);
        cmd_buf_ = VK_NULL_HANDLE;

        // Reset descriptor pool to reclaim all allocated descriptor sets.
        // Safe because the queue is idle — no command buffers are in flight.
        vkResetDescriptorPool(device_, desc_pool_, 0);
    }

private:
    void create_instance() {
        VkApplicationInfo app_info = {};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &app_info;

        VK_CHECK(vkCreateInstance(&ci, nullptr, &instance_));
    }

    void pick_physical_device() {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance_, &count, nullptr);
        if (count == 0) throw std::runtime_error("No Vulkan physical devices");
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance_, &count, devices.data());

        // Prefer discrete GPU
        for (auto& d : devices) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(d, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                physical_device_ = d;
                return;
            }
        }
        physical_device_ = devices[0]; // fallback

        // Query device extension support for fp16 storage
        uint32_t ext_count = 0;
        vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &ext_count, nullptr);
        std::vector<VkExtensionProperties> exts(ext_count);
        vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &ext_count, exts.data());
        for (const auto& ext : exts) {
            if (strcmp(ext.extensionName, VK_KHR_16BIT_STORAGE_EXTENSION_NAME) == 0) {
                supports_16bit_storage_ = true;
                break;
            }
        }
    }

    void create_device() {
        // Find compute queue family
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &count, nullptr);
        std::vector<VkQueueFamilyProperties> qfps(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &count, qfps.data());

        queue_family_ = 0;
        for (uint32_t i = 0; i < count; ++i) {
            if (qfps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queue_family_ = i;
                break;
            }
        }

        float priority = 1.0f;
        VkDeviceQueueCreateInfo qci = {};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = queue_family_;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;

        // Conditionally enable 16-bit storage for fp16 support
        VkPhysicalDevice16BitStorageFeatures storage_features = {};
        storage_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
        storage_features.storageBuffer16BitAccess = VK_TRUE;

        const char* device_exts[16];
        uint32_t dev_ext_count = 0;
        if (supports_16bit_storage_) {
            device_exts[dev_ext_count++] = VK_KHR_16BIT_STORAGE_EXTENSION_NAME;
        }

        VkDeviceCreateInfo dci = {};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = dev_ext_count;
        dci.ppEnabledExtensionNames = dev_ext_count ? device_exts : nullptr;
        dci.pNext = supports_16bit_storage_ ? &storage_features : nullptr;

        VK_CHECK(vkCreateDevice(physical_device_, &dci, nullptr, &device_));
        vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
    }

    void create_command_pool() {
        VkCommandPoolCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.queueFamilyIndex = queue_family_;
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &cmd_pool_));
    }

    void create_descriptor_pool() {
        VkDescriptorPoolSize pool_sizes[] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256 },
        };

        VkDescriptorPoolCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        // Use FREE_DESCRIPTOR_SET_BIT so individual sets can be freed if needed,
        // but we also reset the whole pool after each submit_and_wait to reclaim
        // all sets simply. The reset is safe because the queue is idle at that point.
        ci.maxSets = 256;
        ci.poolSizeCount = 1;
        ci.pPoolSizes = pool_sizes;

        VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &desc_pool_));
    }

    uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags props) {
        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props);
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
            if ((type_bits & (1u << i)) &&
                (mem_props.memoryTypes[i].propertyFlags & props) == props) {
                return i;
            }
        }
        throw std::runtime_error("Failed to find suitable memory type");
    }

    static void VK_CHECK(VkResult res) {
        if (res != VK_SUCCESS) {
            char msg[256];
            snprintf(msg, sizeof(msg), "Vulkan error: VkResult = %d", res);
            throw std::runtime_error(msg);
        }
    }

    VkInstance       instance_               = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_         = VK_NULL_HANDLE;
    VkDevice         device_                 = VK_NULL_HANDLE;
    VkQueue          queue_                  = VK_NULL_HANDLE;
    VkCommandPool    cmd_pool_               = VK_NULL_HANDLE;
    VkCommandBuffer  cmd_buf_                = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_              = VK_NULL_HANDLE;
    uint32_t         queue_family_           = 0;
    bool             supports_16bit_storage_ = false;
};

}  // namespace nnops::test

#endif  // NNOPS_HAS_VULKAN
