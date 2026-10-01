#pragma once
/// @file attention.hpp
/// @brief Multi-head scaled dot-product Attention operator.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes for the multi-head attention operator.
struct AttentionAttributes {
    /// Number of attention heads.
    int64_t num_heads = 8;

    /// Scaling factor applied to QK^T before softmax.
    /// If 0, auto-computed as 1/sqrt(head_dim).
    float scale = 0.0f;

    /// Number of key-value head groups for GQA (Grouped Query Attention).
    /// When 0 (default), equals num_heads (standard MHA).
    /// When < num_heads, KV heads are shared across query head groups.
    int64_t num_group = 0;
};

/// Multi-head scaled dot-product Attention operator (class-based API).
///
/// Computes:
///   scores = Q @ K^T * scale
///   if mask provided: scores += mask
///   attn_weights = softmax(scores, dim=-1)
///   output = attn_weights @ V
///
/// Input layout (both supported):
///   [B, S, H*D]  — merged heads, operator reshapes to [B, H, S, D] internally
///   [B, H, S, D] — explicit per-head layout
///
/// inputs[0] = Q (query)
/// inputs[1] = K (key)
/// inputs[2] = V (value)
/// inputs[3] = mask (optional, broadcastable to [B, 1, Sq, Sk] or [B, H, Sq, Sk])
class Attention : public OpBase {
public:
    /// Create an Attention operator.
    static std::unique_ptr<Attention> create(const AttentionAttributes& attrs,
                                             Backend backend = Backend::CPU);

    /// Create with defaults.
    static std::unique_ptr<Attention> create(Backend backend = Backend::CPU) {
        return create(AttentionAttributes{}, backend);
    }

    // ---- OpBase interface ----
    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    size_t getWorkspaceSize(std::span<const TensorDesc> inputs,
                            std::span<const TensorDesc> outputs) const override;

    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Attention; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::PlanarOnly; }

    const AttentionAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in attention.cpp (Pimpl pattern)
    ~Attention();

private:
    Attention(const AttentionAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    AttentionAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
