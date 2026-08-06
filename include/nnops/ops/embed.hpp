#pragma once
/// @file embed.hpp
/// @brief Embed operator — embedding table lookup (gather).
///
/// Looks up embedding vectors from a weight table by integer indices.
/// This is the first step in every LLM/VLM/VLA: token IDs → dense vectors.
///
/// Supports Planar layout only. For packed layouts, use LayoutConvert first.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes for the Embed operator.
struct EmbedAttributes {
    // Reserved for future: float scale, int64_t padding_idx, etc.
};

/// Embedding table lookup operator.
///
/// Computes: output[...] = weight[clamp(indices[...], 0, V-1)]
/// where V = weight.shape(0).
///
/// Inputs (2):
///   inputs[0] = weight   [vocab_size, dim]  (f32 or f16, planar)
///   inputs[1] = indices  [*]                (int64, planar)
///
/// Outputs (1):
///   outputs[0] = output  [*indices_shape..., dim]  (same dtype as weight)
class Embed : public OpBase {
public:
    /// Create an Embed operator for the specified backend.
    static std::unique_ptr<Embed> create(const EmbedAttributes& attrs = {},
                                          Backend backend = Backend::CPU);

    /// Create with defaults.
    static std::unique_ptr<Embed> create(Backend backend) {
        return create(EmbedAttributes{}, backend);
    }

    // ---- OpBase interface ----
    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Embed; }
    Backend getBackend() const override { return backend_; }

    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::PlanarOnly;
    }

    const EmbedAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in embed.cpp (Pimpl pattern)

private:
    Embed(const EmbedAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    EmbedAttributes attrs_;
    Backend backend_;
};

// ============================================================
// Functional API
// ============================================================

/// Functional embedding lookup.
void embed(const TensorView& weight,
           const TensorView& indices,
           TensorView& output,
           const EmbedAttributes& attrs = {},
           const ComputeContext& ctx = {});

}  // namespace nnops
