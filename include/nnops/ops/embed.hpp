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
/// Supports direct lookup (f32/f16 weight) and int8/uint8 quantized weight
/// with per-row dequantization via weight.quant_params():
///   output[j] = (weight[idx, j] - zero_point[idx]) * scale[idx]
///
/// Quantization parameters (scale, zero_point, granularity) are stored in
/// the weight tensor via TensorView::QuantParams. Supports PerTensor
/// (single scale/zp for all rows), PerToken (per-row scale/zp), and
/// PerChannel — for the 2D weight [vocab_size, dim], PerChannel quantizes
/// along axis 0 (the vocab entry), i.e. per-row, identical to PerToken.
///
/// Inputs (2):
///   inputs[0] = weight   [vocab_size, dim]  (f32, f16, s8, u8)
///   inputs[1] = indices  [*]                (s64 or s32)
///
/// Outputs (1):
///   outputs[0] = output  [*indices_shape..., dim]
///                dtype = weight dtype (f32/f16), or f32/f16 when weight is int8
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

    /// Workspace (bytes) required only for int8/uint8 weight with f16 output:
    /// a f32 staging buffer for the dequantized gathered rows. Returns 0 for
    /// f32 output (dequantized directly into the output) and for float weight.
    size_t getWorkspaceSize(std::span<const TensorDesc> inputs,
                            std::span<const TensorDesc> outputs) const override;

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
/// For int8 weight, quantization params (scale/zero_point) are read from
/// weight.quant_params() via TensorView.
void embed(const TensorView& weight,
           const TensorView& indices,
           TensorView& output,
           const EmbedAttributes& attrs = {},
           const ComputeContext& ctx = {});

}  // namespace nnops
