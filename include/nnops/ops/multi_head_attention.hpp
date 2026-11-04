#pragma once
/// @file multi_head_attention.hpp
/// @brief Multi-head scaled dot-product Attention (every query head has its own
///        key/value head).
///
/// scores = Q·Kᵀ·scale (+ mask, if given); output = softmax(scores, last dim)·V.
/// Q/K/V arrive as [B, S, H·D] (reshaped internally to [B, H, S, D]) or already
/// as [B, H, S, D]; inputs[0..2] = Q, K, V and inputs[3] = optional additive
/// mask, flat [Sq, Sk] and shared by every batch and head. scale = 0 means
/// 1/sqrt(head_dim). Planar layouts only.
///
/// Grouped-query attention — several query heads per key/value head — is
/// GroupQueryAttention, and the absorbed DeepSeek-style latent form is
/// MultiLatentAttention; this operator is the H == Hkv case and does not
/// implement either.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes for the multi-head attention operator.
struct MultiHeadAttentionAttributes {
    int64_t num_heads = 8;

    /// Attention scale; 0 means 1/sqrt(head_dim).
    float scale = 0.0f;

    /// Causal masking: query row i attends to key positions j <= i + (Sk - Sq).
    /// The bottom-right alignment makes the Sq < Sk case the KV-cache decode
    /// convention — the new queries are the last Sq positions of the key run.
    bool causal = false;
};

class MultiHeadAttention : public OpBase {
public:
    static std::unique_ptr<MultiHeadAttention> create(
        const MultiHeadAttentionAttributes& attrs,
        Backend backend = Backend::CPU);

    static std::unique_ptr<MultiHeadAttention> create(Backend backend = Backend::CPU) {
        return create(MultiHeadAttentionAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::MultiHeadAttention; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::PlanarOnly; }

    const MultiHeadAttentionAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in multi_head_attention.cpp (Pimpl pattern)
    ~MultiHeadAttention();

private:
    MultiHeadAttention(const MultiHeadAttentionAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    MultiHeadAttentionAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
