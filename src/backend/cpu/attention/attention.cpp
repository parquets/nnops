/// @file attention.cpp
/// @brief The shared CPU attention core's public entry, plus the three
///        operators' kernel entries.
///
/// See attention.h for the routes, the head mapping and the scratch contract.
/// The kernels themselves live per route — attention_standard.cpp,
/// attention_flash.cpp, attention_decode.cpp — and the plan that picks between
/// them is attention_plan.cpp; this file validates the call, allocates the
/// scratch once, and dispatches.
///
/// The only per-operator code here is the attribute → AttentionConfig mapping
/// at the bottom: MultiHeadAttention, GroupQueryAttention and
/// MultiLatentAttention all run the kernels above.

#include "attention_internal.h"
#include "backend/cpu/common/memory_pool.hpp"           // internal scratch-memory pool
#include "nnops/ops/group_query_attention.hpp"
#include "nnops/ops/multi_head_attention.hpp"
#include "nnops/ops/multi_latent_attention.hpp"

namespace nnops::backend::cpu {

// =========================================================================
//  Public API
// =========================================================================

Status attention_cpu(const AttentionConfig& cfg,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* /*workspace*/)
{
    const size_t required = cfg.v_is_k ? 2u : 3u;
    if (inputs.size() < required) { return Status::invalid_argument; }

    const DataType dt = inputs[0].data_type();
    const size_t seqlens = seqlens_slot(cfg);

    bool dtypes_ok = (dt == DataType::f32 || dt == DataType::f16) &&
                     output.data_type() == dt;
    for (size_t i = 1; i < inputs.size() && dtypes_ok; ++i) {
        if (i == seqlens || inputs[i].is_empty()) { continue; }  // optional slots
        dtypes_ok = inputs[i].data_type() == dt;
    }
    if (!dtypes_ok) { return Status::unsupported; }

    // The optional per-batch run lengths are int32 [B]; the kernels index them
    // by batch, so a short tensor must not reach them.
    if (has_seqlens_input(cfg, inputs)) {
        const auto& s = inputs[seqlens];
        if (s.data_type() != DataType::s32 || s.rank() != 1 ||
            s.shape(0) != inputs[0].shape(0)) {
            return Status::invalid_argument;
        }
    }

    // One plan decides tiles and scratch, and the scratch is allocated once
    // here, outside the parallel dispatch — no per-task allocation. Slots are
    // indexed by worker id, so the count is the workers, not the grid: an
    // over-subscribed provider would otherwise allocate 4x the scratch.
    detail::SmallVector<TensorDesc, 16> descs;
    for (size_t i = 0; i < inputs.size(); ++i) { descs.push_back(inputs[i].desc()); }
    const TensorDesc outs[] = {output.desc()};
    const AttentionPlan plan = get_attention_plan(cfg, descs, outs,
                                                  ctx.cpu.worker_count(),
                                                  ctx.cpu.thread_slot_count() > 0);
    PoolPtr scratch(plan.workspace_size);

    if (plan.use_decode) {
        attention_decode_cpu(cfg, output, inputs, ctx, plan, scratch.get());
    } else if (plan.use_flash) {
        attention_flash_cpu(cfg, output, inputs, ctx, plan, scratch.get());
    } else {
        attention_standard_cpu(cfg, output, inputs, ctx, plan, scratch.get());
    }
    return Status::ok;
}

// =========================================================================
//  Operator entry points
// =========================================================================

Status multi_head_attention_cpu(const MultiHeadAttentionAttributes& attrs,
                                TensorView& output,
                                std::span<const TensorView> inputs,
                                const ComputeContext& ctx,
                                void* workspace)
{
    AttentionConfig cfg;
    cfg.num_heads = attrs.num_heads;
    cfg.scale = attrs.scale;
    cfg.causal = attrs.causal;
    cfg.position_offset = attrs.position_offset;
    return attention_cpu(cfg, output, inputs, ctx, workspace);
}

Status group_query_attention_cpu(const GroupQueryAttentionAttributes& attrs,
                                 TensorView& output,
                                 std::span<const TensorView> inputs,
                                 const ComputeContext& ctx,
                                 void* workspace)
{
    AttentionConfig cfg;
    cfg.num_heads = attrs.num_heads;
    cfg.num_kv_heads = attrs.kv_num_heads;
    cfg.scale = attrs.scale;
    cfg.causal = attrs.causal;
    cfg.position_offset = attrs.position_offset;
    return attention_cpu(cfg, output, inputs, ctx, workspace);
}

Status multi_latent_attention_cpu(const MultiLatentAttentionAttributes& attrs,
                                  TensorView& output,
                                  std::span<const TensorView> inputs,
                                  const ComputeContext& ctx,
                                  void* workspace)
{
    AttentionConfig cfg;
    cfg.num_heads = attrs.num_heads;
    cfg.num_kv_heads = attrs.kv_num_heads;
    cfg.v_head_dim = attrs.v_head_dim > 0 ? attrs.v_head_dim : attrs.kv_lora_rank;
    cfg.scale = attrs.scale != 0.0f
        ? attrs.scale
        : 1.0f / std::sqrt(static_cast<float>(attrs.qk_nope_head_dim + attrs.qk_rope_head_dim));
    cfg.causal = attrs.causal;
    cfg.position_offset = attrs.position_offset;
    cfg.v_is_k = true;
    return attention_cpu(cfg, output, inputs, ctx, workspace);
}

}  // namespace nnops::backend::cpu
