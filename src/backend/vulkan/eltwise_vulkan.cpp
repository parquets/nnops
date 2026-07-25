/// @file eltwise_vulkan.cpp
/// @brief Vulkan compute implementation of element-wise binary operations.
///
/// Uses pre-compiled SPIR-V compute shaders and specialization constants
/// for the operation type (Add/Sub/Mul/Div) and add_to flag.
///
/// Design:
///   - One compute pipeline per (op_type, add_to) combination.
///   - Pipelines are lazily created and cached via VulkanPipelineCache.
///   - Per-call descriptor set allocated from user-provided pool.
///   - Supports f32 only (f16 support requires VK_KHR_16bit_storage).
///
/// Requires ComputeContext fields:
///   - vulkan_cmd_buffer:      VkCommandBuffer to record into
///   - vulkan_device:          VkDevice handle
///   - vulkan_descriptor_pool: VkDescriptorPool (FREE_DESCRIPTOR_SET_BIT)
///   - vulkan_buffers:         array of VkBuffer handles [input A, input B, output]

#include "vulkan_common.hpp"
#include "eltwise_f32_spv.h"
#include "nnops/ops/eltwise.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops::backend::vulkan {

// ============================================================
// Helpers to map enums to specialization constants
// ============================================================

static constexpr uint32_t eltwise_op_to_spec(EltwiseType type) noexcept {
    switch (type) {
    case EltwiseType::Add: return 0;
    case EltwiseType::Sub: return 1;
    case EltwiseType::Mul: return 2;
    case EltwiseType::Div: return 3;
    }
    return 0;
}

// ============================================================
// Entry point
// ============================================================

void eltwise_vulkan(
    const EltwiseAttributes& attrs,
    TensorView& /*output*/,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx,
    void* /*workspace*/)
{
    const auto& A = inputs[0];
    const auto& B = inputs[1];
    const int64_t total = A.numel();
    if (total == 0) return;
    NNOPS_ASSERT(A.numel() == B.numel());

    // ---- Extract Vulkan resources from context ----
    VkCommandBuffer  cmd    = static_cast<VkCommandBuffer>(ctx.vulkan_cmd_buffer);
    VkDevice         device = static_cast<VkDevice>(ctx.vulkan_device);
    VkDescriptorPool pool   = static_cast<VkDescriptorPool>(ctx.vulkan_descriptor_pool);

    // vulkan_buffers layout: [input A, input B, output]
    // Passed via vulkan_buffers/vulkan_buffers_count in ComputeContext
    // For eltwise: 3 buffers minimum
    NNOPS_ASSERT(ctx.vulkan_buffers_count >= 3);
    const VkBuffer* bufs = static_cast<const VkBuffer*>(ctx.vulkan_buffers);

    // ---- Get or create compute pipeline ----
    auto& cache = VulkanPipelineCache::instance();
    PipelineKey key = {};
    key.device       = device;
    key.spirv_data   = g_eltwise_f32_spv;
    key.spirv_size   = g_eltwise_f32_spv_len;
    key.spec_op      = eltwise_op_to_spec(attrs.type);
    key.spec_add_to  = attrs.add_to ? 1u : 0u;
    key.num_bindings = 3;  // A, B, C

    VulkanPipeline pipeline = cache.get_or_create(key);
    if (pipeline.pipeline == VK_NULL_HANDLE) {
        fprintf(stderr, "eltwise_vulkan: failed to create pipeline\n");
        return;
    }

    // ---- Record dispatch commands ----
    ComputePushConstants pc = {};
    pc.total = static_cast<int32_t>(total);

    // Buffer bindings: [0]=A, [1]=B, [2]=output
    VkBuffer dispatch_buffers[3] = { bufs[0], bufs[1], bufs[2] };

    vulkan_record_dispatch(cmd, device, pool, pipeline,
                           dispatch_buffers, 3,
                           &pc, sizeof(pc),
                           static_cast<uint32_t>(total));
}

}  // namespace nnops::backend::vulkan
