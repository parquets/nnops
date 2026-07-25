/// @file vulkan_test_helper.cpp
/// @brief Opaque helpers for Vulkan test infrastructure.

#ifdef NNOPS_HAS_VULKAN

#include "backend/vulkan/vulkan_common.hpp"
#include <vulkan/vulkan.h>

namespace nnops::test {

void vulkan_pipeline_cache_cleanup(VkDevice device) {
    nnops::backend::vulkan::VulkanPipelineCache::instance().destroy_device_pipelines(device);
}

}  // namespace nnops::test

#endif  // NNOPS_HAS_VULKAN
