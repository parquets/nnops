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
#include "backend/cpu/common/dtype_dispatch.hpp"
#include "backend/cpu/common/memory_pool.hpp"           // internal scratch-memory pool
#include "nnops/ops/group_query_attention.hpp"
#include "nnops/ops/multi_head_attention.hpp"
#include "nnops/ops/multi_latent_attention.hpp"

namespace nnops::backend::cpu {

namespace {

/// One attention call over a single key run: the route dispatch and its one
/// scratch allocation. A varlen call runs this once per packed segment.
Status attention_single(const AttentionConfig& cfg,
                        TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx)
{
    const size_t required = cfg.v_is_k ? 2u : 3u;
    if (inputs.size() < required) { return Status::invalid_argument; }

    const DataType dt = inputs[0].data_type();
    const size_t seqlens = seqlens_slot(cfg);

    bool dtypes_ok = is_float_dtype(dt) && output.data_type() == dt;
    for (size_t i = 1; i < inputs.size() && dtypes_ok; ++i) {
        // Everything from seqlens_k on is int32 metadata (the run lengths, the
        // two varlen offset vectors); the slots before it are activations.
        if (i >= seqlens || inputs[i].is_empty()) { continue; }
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

/// A packed row range of a merged [1, total, W] tensor: the rows are a
/// contiguous span of the parent's storage and keep its pitch, so no data is
/// copied — the routes see one segment as if it were the whole [1, len, W]
/// problem.
inline TensorView packed_row_view(const TensorView& parent, int64_t row_off,
                                  int64_t rows, int64_t width)
{
    const int64_t dims[3] = {1, rows, width};
    char* base = const_cast<char*>(parent.ptr<char>()) + row_off * parent.pitch();
    return TensorView(std::span<const int64_t>(dims, 3), parent.data_type(), base,
                      parent.pitch(), parent.layout());
}

/// Varlen: run each packed segment as its own [1, len, W] problem. The segments
/// share no key, so the routes' per-(batch, head) work stays inside one segment
/// and the causal diagonal restarts at each cu offset.
Status attention_varlen(const AttentionConfig& cfg,
                        TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx)
{
    if (cfg.position_offset >= 0) { return Status::invalid_argument; }
    if (inputs.size() <= cu_k_slot(cfg) || inputs[cu_k_slot(cfg)].is_empty() ||
        has_mask_input(cfg, inputs) || has_past_view(cfg, inputs) ||
        has_seqlens_input(cfg, inputs)) {
        return Status::invalid_argument;
    }
    const auto& Q = inputs[0];
    const auto& K = inputs[1];
    const auto& V = cfg.v_is_k ? inputs[1] : inputs[2];
    if (Q.rank() != 3 || K.rank() != 3 || V.rank() != 3 || Q.shape(0) != 1) {
        return Status::invalid_argument;
    }

    const auto& cq = inputs[cu_q_slot(cfg)];
    const auto& ck = inputs[cu_k_slot(cfg)];
    if (cq.data_type() != DataType::s32 || ck.data_type() != DataType::s32 ||
        cq.rank() != 1 || ck.rank() != 1 || cq.shape(0) != ck.shape(0) ||
        cq.shape(0) < 2) {
        return Status::invalid_argument;
    }
    const int32_t* qo = cq.ptr<int32_t>();
    const int32_t* ko = ck.ptr<int32_t>();
    const int64_t nseg = cq.shape(0) - 1;
    const int64_t total_q = Q.shape(1);
    const int64_t total_k = K.shape(1);

    for (int64_t s = 0; s < nseg; ++s) {
        const int64_t q0 = qo[s], q1 = qo[s + 1];
        const int64_t k0 = ko[s], k1 = ko[s + 1];
        if (q0 < 0 || q1 < q0 || q1 > total_q ||
            k0 < 0 || k1 < k0 || k1 > total_k) {
            return Status::invalid_argument;
        }
        const int64_t len_q = q1 - q0;
        const int64_t len_k = k1 - k0;
        // Bottom-right alignment needs the segment's query rows to sit inside
        // its key rows; a shorter key run would leave queries past the end.
        if (len_k < len_q) { return Status::invalid_argument; }
        if (len_q == 0) { continue; }

        const TensorView qs = packed_row_view(Q, q0, len_q, Q.shape(2));
        const TensorView ks = packed_row_view(K, k0, len_k, K.shape(2));
        const TensorView vs = cfg.v_is_k
            ? ks : packed_row_view(V, k0, len_k, V.shape(2));
        const TensorView in3[3] = {qs, ks, vs};

        TensorView os = packed_row_view(output, q0, len_q, output.shape(2));

        AttentionConfig seg = cfg;
        seg.position_offset = -1;
        const Status st = attention_single(seg, os, in3, ctx);
        if (st != Status::ok) { return st; }
    }
    return Status::ok;
}

/// out *= act(gate), one pass over the finished output.
///
/// The gate rides the output rather than any route's inner loop: every route
/// has written the whole output by the time it returns, so this is the single
/// place the multiply is applied — decode, standard, flash and the varlen
/// recursion all get it without knowing it exists. Under varlen the gate is the
/// parent's tensor (no segment slicing), which the flat row walk handles.
template <class T>
void output_gate_impl(const AttentionConfig& cfg, TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx)
{
    const auto& g = inputs[gate_slot(cfg)];
    const int64_t n = output.shape(output.rank() - 1);
    if (n <= 0) { return; }
    const int64_t rows = output.numel() / n;
    const int64_t y_rs = output.row_stride_elems();
    const int64_t g_rs = g.row_stride_elems();
    T* y = output.ptr<T>();
    const T* gp = g.ptr<T>();
    const EpilogueActivateType act = cfg.output_gate;
    ctx.cpu.run(0, rows, [&](int64_t r) {
        gate_scale_row_dispatch(act, y + r * y_rs, gp + r * g_rs, n);
    });
}

Status apply_output_gate(const AttentionConfig& cfg, TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx)
{
    if (cfg.output_gate == EpilogueActivateType::None) { return Status::ok; }
    const size_t s = gate_slot(cfg);
    // The gate rides the output, so it must be exactly its geometry; the
    // operators check this too, but the route's dtype loop skips slot 9 (it
    // sits past seqlens_k, where every other slot is int32 metadata).
    if (inputs.size() <= s || inputs[s].is_empty()) {
        return Status::invalid_argument;
    }
    const auto& g = inputs[s];
    if (g.data_type() != output.data_type() || g.rank() != output.rank()) {
        return Status::invalid_argument;
    }
    for (int64_t i = 0; i < output.rank(); ++i) {
        if (g.shape(i) != output.shape(i)) { return Status::invalid_argument; }
    }
    dispatch_f32_f16(output.data_type(), "attention_output_gate", [&](auto tag) {
        using T = typename decltype(tag)::type;
        output_gate_impl<T>(cfg, output, inputs, ctx);
    });
    return Status::ok;
}

}  // anonymous namespace

// =========================================================================
//  Public API
// =========================================================================

Status attention_cpu(const AttentionConfig& cfg,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* /*workspace*/)
{
    const Status st = has_varlen(cfg, inputs)
        ? attention_varlen(cfg, output, inputs, ctx)
        : attention_single(cfg, output, inputs, ctx);
    if (st != Status::ok) { return st; }
    // The output gate is the last thing done to the output, after every route
    // (and every varlen segment) has written it.
    return apply_output_gate(cfg, output, inputs, ctx);
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
    cfg.softcap = attrs.softcap;
    cfg.window = attrs.window;
    cfg.output_gate = attrs.output_gate;
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
    cfg.softcap = attrs.softcap;
    cfg.window = attrs.window;
    cfg.output_gate = attrs.output_gate;
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
    // Absorbed: one latent row per token and the value is its prefix. The
    // non-absorbed form takes a separate V tensor, as GQA does.
    cfg.v_is_k = attrs.absorbed;
    return attention_cpu(cfg, output, inputs, ctx, workspace);
}

}  // namespace nnops::backend::cpu
