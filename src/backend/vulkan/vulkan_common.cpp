/// @file vulkan_common.cpp
/// @brief Implementation of Vulkan pipeline cache and dispatch helpers.

#include "vulkan_common.hpp"
#include <cstring>

namespace nnops::backend::vulkan {

// ============================================================
// VulkanPipelineCache implementation
// ============================================================

VulkanPipeline VulkanPipelineCache::get_or_create(const PipelineKey& key) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;
        }
    }

    // Create pipeline outside lock (expensive, don't want to block others)
    VulkanPipeline pipeline = create_pipeline(key);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        // Double-check: another thread may have created it while we were working
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            // Destroy the one we just created (wasted work but correct)
            if (pipeline.pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(key.device, pipeline.pipeline, nullptr);
                vkDestroyPipelineLayout(key.device, pipeline.layout, nullptr);
                vkDestroyDescriptorSetLayout(key.device, pipeline.set_layout, nullptr);
            }
            return it->second;
        }
        cache_[key] = pipeline;
        return pipeline;
    }
}

VulkanPipeline VulkanPipelineCache::create_pipeline(const PipelineKey& key) {
    VulkanPipeline result = {};

    VkDevice device = key.device;

    // ---- Step 1: Create shader module ----
    VkShaderModuleCreateInfo shader_ci = {};
    shader_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_ci.codeSize = key.spirv_size;
    shader_ci.pCode = key.spirv_data;

    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkResult res = vkCreateShaderModule(device, &shader_ci, nullptr, &shader_module);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "vkCreateShaderModule failed: %d\n", res);
        return result;
    }

    // ---- Step 2: Create descriptor set layout ----
    // All bindings are storage buffers (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
    VkDescriptorSetLayoutBinding* bindings =
        new VkDescriptorSetLayoutBinding[key.num_bindings]();

    for (uint32_t i = 0; i < key.num_bindings; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dsl_ci = {};
    dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_ci.bindingCount = key.num_bindings;
    dsl_ci.pBindings = bindings;

    res = vkCreateDescriptorSetLayout(device, &dsl_ci, nullptr, &result.set_layout);
    delete[] bindings;
    if (res != VK_SUCCESS) {
        fprintf(stderr, "vkCreateDescriptorSetLayout failed: %d\n", res);
        vkDestroyShaderModule(device, shader_module, nullptr);
        return result;
    }

    // ---- Step 3: Create pipeline layout ----
    VkPushConstantRange pc_range = {};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = sizeof(ComputePushConstants);

    VkPipelineLayoutCreateInfo pl_ci = {};
    pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount = 1;
    pl_ci.pSetLayouts = &result.set_layout;
    pl_ci.pushConstantRangeCount = 1;
    pl_ci.pPushConstantRanges = &pc_range;

    res = vkCreatePipelineLayout(device, &pl_ci, nullptr, &result.layout);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "vkCreatePipelineLayout failed: %d\n", res);
        vkDestroyDescriptorSetLayout(device, result.set_layout, nullptr);
        vkDestroyShaderModule(device, shader_module, nullptr);
        return result;
    }

    // ---- Step 4: Create compute pipeline with specialization constants ----
    VkSpecializationMapEntry spec_entries[2] = {};
    spec_entries[0].constantID = 0;
    spec_entries[0].offset = 0;
    spec_entries[0].size = sizeof(uint32_t);
    spec_entries[1].constantID = 1;
    spec_entries[1].offset = sizeof(uint32_t);
    spec_entries[1].size = sizeof(uint32_t);

    uint32_t spec_data[2] = { key.spec_op, key.spec_add_to };

    VkSpecializationInfo spec_info = {};
    spec_info.mapEntryCount = 2;
    spec_info.pMapEntries = spec_entries;
    spec_info.dataSize = sizeof(spec_data);
    spec_info.pData = spec_data;

    VkPipelineShaderStageCreateInfo stage_ci = {};
    stage_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_ci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_ci.module = shader_module;
    stage_ci.pName = "main";
    stage_ci.pSpecializationInfo = &spec_info;

    VkComputePipelineCreateInfo pipeline_ci = {};
    pipeline_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_ci.stage = stage_ci;
    pipeline_ci.layout = result.layout;

    res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_ci,
                                   nullptr, &result.pipeline);

    // Shader module can be destroyed after pipeline creation
    vkDestroyShaderModule(device, shader_module, nullptr);

    if (res != VK_SUCCESS) {
        fprintf(stderr, "vkCreateComputePipelines failed: %d\n", res);
        vkDestroyPipelineLayout(device, result.layout, nullptr);
        vkDestroyDescriptorSetLayout(device, result.set_layout, nullptr);
        result.pipeline = VK_NULL_HANDLE;
        result.layout = VK_NULL_HANDLE;
        result.set_layout = VK_NULL_HANDLE;
    }

    return result;
}

void VulkanPipelineCache::destroy_device_pipelines(VkDevice device) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.begin();
    while (it != cache_.end()) {
        if (it->first.device == device) {
            const auto& p = it->second;
            vkDestroyPipeline(device, p.pipeline, nullptr);
            vkDestroyPipelineLayout(device, p.layout, nullptr);
            vkDestroyDescriptorSetLayout(device, p.set_layout, nullptr);
            it = cache_.erase(it);
        } else {
            ++it;
        }
    }
}

void VulkanPipelineCache::clear() {
    // Must have collected all devices before clearing
    std::lock_guard<std::mutex> lock(mutex_);
    // We don't track all devices, so we can only destroy by iterating
    for (auto& [key, p] : cache_) {
        vkDestroyPipeline(key.device, p.pipeline, nullptr);
        vkDestroyPipelineLayout(key.device, p.layout, nullptr);
        vkDestroyDescriptorSetLayout(key.device, p.set_layout, nullptr);
    }
    cache_.clear();
}

VulkanPipelineCache::~VulkanPipelineCache() {
    clear();
}

// ============================================================
// vulkan_record_dispatch
// ============================================================

void vulkan_record_dispatch(
    VkCommandBuffer        cmd,
    VkDevice               device,
    VkDescriptorPool       pool,
    const VulkanPipeline&  pipeline,
    const VkBuffer*        buffers,
    uint32_t               num_buffers,
    const void*            push_constants,
    uint32_t               pc_size,
    uint32_t               total_elements)
{
    // ---- Step 1: Allocate descriptor set ----
    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &pipeline.set_layout;

    VkDescriptorSet desc_set = VK_NULL_HANDLE;
    VkResult res = vkAllocateDescriptorSets(device, &alloc_info, &desc_set);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "vkAllocateDescriptorSets failed: %d\n", res);
        return;
    }

    // ---- Step 2: Update descriptor set with buffer info ----
    VkDescriptorBufferInfo* buf_infos = new VkDescriptorBufferInfo[num_buffers]();
    VkWriteDescriptorSet* writes = new VkWriteDescriptorSet[num_buffers]();

    for (uint32_t i = 0; i < num_buffers; ++i) {
        buf_infos[i].buffer = buffers[i];
        buf_infos[i].offset = 0;
        buf_infos[i].range = VK_WHOLE_SIZE;

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = desc_set;
        writes[i].dstBinding = i;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buf_infos[i];
    }

    vkUpdateDescriptorSets(device, num_buffers, writes, 0, nullptr);
    delete[] writes;
    delete[] buf_infos;

    // ---- Step 3: Bind pipeline and descriptor set ----
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline.layout, 0, 1, &desc_set, 0, nullptr);

    // ---- Step 4: Push constants ----
    vkCmdPushConstants(cmd, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, pc_size, push_constants);

    // ---- Step 5: Dispatch ----
    uint32_t group_count = ceil_div(total_elements, static_cast<uint32_t>(kVulkanWorkgroupSize));
    vkCmdDispatch(cmd, group_count, 1, 1);
}

}  // namespace nnops::backend::vulkan
