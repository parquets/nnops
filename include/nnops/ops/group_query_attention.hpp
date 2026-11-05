#pragma once
/// @file group_query_attention.hpp
/// @brief Grouped-query scaled dot-product attention.
///
/// Multi-head attention with fewer key/value heads than query heads: query head
/// h reads kv head h / (H / H_kv), so consecutive query heads share one K/V
/// pair. H_kv == H is ordinary MHA; H_kv == 1 is multi-query attention.
///
///   scores = Q·Kᵀ·scale (+ mask);  output = softmax(scores, -1)·V
///
/// Inputs, planar only:
///   [0] Q [B, Sq, H·D]    or [B, H,   Sq, D]
///   [1] K [B, Sk, H_kv·D] or [B, H_kv, Sk, D]
///   [2] V [B, Sk, H_kv·D] or [B, H_kv, Sk, D]
///   [3] optional additive mask, flat [Sq, Sk], shared by every batch and head.
/// The output follows Q's shape. scale = 0 means 1/sqrt(D).
///
/// `causal` (default true, as in onnxruntime's GroupQueryAttention) masks query
/// row i to key positions j <= q0 + i, with q0 = position_offset or, by
/// default, Sk - Sq: the bottom-right alignment makes a shorter query run the
/// KV-cache decode case, where the Sq new tokens are the last Sq positions of
/// the cached key run.
///
/// Chunked input: inputs[4] and inputs[5] are an optional past K and past V —
/// the same layout as K/V, with Sk_past rows. The operator reads the run
/// [past | new] by streaming each segment in place, so the caller never
/// concatenates the cache for a step, and passes position_offset = Sk_past for
/// a chunked prefill. The mask stays [Sq, Sk_past + Sk_new]. The past is
/// read-only: no present/cache output is written, growing the cache stays the
/// caller's job. Optional slots are positional, so a step call without a mask
/// passes an empty view in inputs[3].
///
/// `position_offset` is what makes a non-tail chunk decodable: it is the
/// absolute position of query row 0, and each row sees j <= position_offset + i
/// regardless of where the chunk sits in the run.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

struct GroupQueryAttentionAttributes {
    /// Query heads (H).
    int64_t num_heads = 8;

    /// Key/value heads (H_kv). 0 means num_heads; otherwise H % H_kv == 0.
    int64_t kv_num_heads = 0;

    /// Attention scale; 0 means 1/sqrt(head_dim).
    float scale = 0.0f;

    /// Causal mask: j <= q0 + i, with q0 = position_offset or Sk - Sq.
    bool causal = true;

    /// Absolute position of query row 0; negative means bottom-right. Only
    /// consulted when causal. With a past segment, the chunked-prefill value is
    /// the past length.
    int64_t position_offset = -1;
};

class GroupQueryAttention : public OpBase {
public:
    static std::unique_ptr<GroupQueryAttention> create(
        const GroupQueryAttentionAttributes& attrs,
        Backend backend = Backend::CPU);

    static std::unique_ptr<GroupQueryAttention> create(Backend backend = Backend::CPU) {
        return create(GroupQueryAttentionAttributes{}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    Status validate(std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs) const override;

    size_t getWorkspaceSize(std::span<const TensorDesc> inputs,
                            std::span<const TensorDesc> outputs) const override;

    using OpBase::compute;

    Status compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::GroupQueryAttention; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::PlanarOnly; }

    const GroupQueryAttentionAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~GroupQueryAttention();

private:
    GroupQueryAttention(const GroupQueryAttentionAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    GroupQueryAttentionAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
