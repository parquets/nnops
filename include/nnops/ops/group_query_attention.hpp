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
/// row i to key positions j <= i + (Sk - Sq): the bottom-right alignment makes
/// a shorter query run the KV-cache decode case, where the Sq new tokens are
/// the last Sq positions of the cached key run. No past/present cache is taken
/// or written — the caller owns the cache and passes the whole key run.

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

    /// Causal mask, bottom-right aligned: j <= i + (Sk - Sq).
    bool causal = true;
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
