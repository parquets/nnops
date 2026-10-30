/// @file unary_vulkan.cpp
/// @brief Vulkan compute implementation of element-wise unary math operations.
///
/// Pre-compiled SPIR-V shaders with specialization constants for the op
/// (Exp/Log/Sin/Cos/Tan/Tanh/Abs/Neg/Sqrt) and the add_to flag. Supports f32 and
/// f16; the f16 shader needs 16-bit storage (GL_EXT_shader_16bit_storage) and
/// computes in fp32, widening loads and narrowing stores.
///
/// Design:
///   - One compute pipeline per (device, dtype, op, add_to), created lazily and
///     cached in VulkanPipelineCache.
///   - The descriptor set is allocated per call from the caller's pool.
///
/// Requires ComputeContext: vulkan_cmd_buffer, vulkan_device,
/// vulkan_descriptor_pool (FREE_DESCRIPTOR_SET_BIT), and vulkan_buffers holding
/// [input, output].

#include "vulkan_common.hpp"
#include "unary_f32_spv.h"
#include "unary_f16_spv.h"
#include "nnops/ops/unary.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops::backend::vulkan {

static constexpr uint32_t unary_op_to_spec(UnaryType type) noexcept {
    switch (type) {
    case UnaryType::Exp:  return 0;
    case UnaryType::Log:  return 1;
    case UnaryType::Sin:  return 2;
    case UnaryType::Cos:  return 3;
    case UnaryType::Tan:  return 4;
    case UnaryType::Tanh: return 5;
    case UnaryType::Abs:  return 6;
    case UnaryType::Neg:  return 7;
    case UnaryType::Sqrt: return 8;
    }
    return 0;
}

void unary_vulkan(
    const UnaryAttributes& attrs,
    TensorView& /*output*/,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx,
    void* /*workspace*/)
{
    const auto& input = inputs[0];
    const int64_t total = input.numel();
    if (total == 0) { return; }

    const auto dtype = input.data_type();
    const uint32_t* spirv_data = nullptr;
    size_t spirv_size = 0;

    switch (dtype) {
    case DataType::f32:
        spirv_data = g_unary_f32_spv;
        spirv_size = g_unary_f32_spv_len;
        break;
    case DataType::f16:
        spirv_data = g_unary_f16_spv;
        spirv_size = g_unary_f16_spv_len;
        break;
    default:
        NNOPS_ASSERT(!"unsupported data type (only f32 and f16)");
        return;
    }

    VkCommandBuffer  cmd    = static_cast<VkCommandBuffer>(ctx.vulkan_cmd_buffer);
    VkDevice         device = static_cast<VkDevice>(ctx.vulkan_device);
    VkDescriptorPool pool   = static_cast<VkDescriptorPool>(ctx.vulkan_descriptor_pool);

    NNOPS_ASSERT(ctx.vulkan_buffers_count >= 2);
    const VkBuffer* bufs = static_cast<const VkBuffer*>(ctx.vulkan_buffers);

    auto& cache = VulkanPipelineCache::instance();
    PipelineKey key = {};
    key.device       = device;
    key.spirv_data   = spirv_data;
    key.spirv_size   = spirv_size;
    key.spec_op      = unary_op_to_spec(attrs.type);
    key.spec_add_to  = attrs.add_to ? 1u : 0u;
    key.num_bindings = 2;  // input, output

    VulkanPipeline pipeline = cache.get_or_create(key);
    if (pipeline.pipeline == VK_NULL_HANDLE) {
        fprintf(stderr, "unary_vulkan: failed to create pipeline\n");
        return;
    }

    ComputePushConstants pc = {};
    pc.total = static_cast<int32_t>(total);

    VkBuffer dispatch_buffers[2] = { bufs[0], bufs[1] };

    vulkan_record_dispatch(cmd, device, pool, pipeline,
                           dispatch_buffers, 2,
                           &pc, sizeof(pc),
                           static_cast<uint32_t>(total));
}

}  // namespace nnops::backend::vulkan
