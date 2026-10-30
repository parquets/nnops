#pragma once
/// @file attention.hpp
/// @brief Multi-head scaled dot-product Attention.
///
/// scores = Q·Kᵀ·scale (+ mask, if given); output = softmax(scores, last dim)·V.
/// Q/K/V arrive as [B, S, H·D] (reshaped internally to [B, H, S, D]) or already
/// as [B, H, S, D]; inputs[0..2] = Q, K, V and inputs[3] = optional additive
/// mask, flat [Sq, Sk] and shared by every batch and head. scale = 0 means
/// 1/sqrt(head_dim). num_group is accepted but not implemented — K/V use
/// num_heads (standard MHA). Planar layouts only.

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
    int64_t num_heads = 8;
    float scale = 0.0f;
    int64_t num_group = 0;
};

class Attention : public OpBase {
public:
    static std::unique_ptr<Attention> create(const AttentionAttributes& attrs,
                                             Backend backend = Backend::CPU);

    static std::unique_ptr<Attention> create(Backend backend = Backend::CPU) {
        return create(AttentionAttributes{}, backend);
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
