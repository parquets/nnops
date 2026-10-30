#pragma once
/// @file embed.hpp
/// @brief Embed operator — embedding table lookup (gather).
///
/// output[...] = weight[clamp(indices[...], 0, V-1)], V = weight.shape(0), with
/// D following the index shape.
///
/// An f32/f16 weight is copied row by row. An s8/u8 weight is dequantized per
/// row straight into the f32 or f16 output in one pass, so the op needs no
/// workspace. Only per-row quantization applies: PerToken, or PerChannel on a
/// 2D [V, D] weight (axis 0 is the vocab entry, so it is the same thing);
/// PerTensor is rejected.
///
/// Planar only — pack/unpack with LayoutConvert first.
///
///   inputs[0] weight   [V, D]       f32 | f16 | s8 | u8
///   inputs[1] indices  [...]        s64 | s32
///   outputs[0] output  [<indices dims>, D]
///                      the weight's dtype, f32 or f16 when the weight is int

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

struct EmbedAttributes {
    // Reserved for future: float scale, int64_t padding_idx, etc.
};

class Embed : public OpBase {
public:
    static std::unique_ptr<Embed> create(const EmbedAttributes& attrs = {},
                                          Backend backend = Backend::CPU);

    static std::unique_ptr<Embed> create(Backend backend) {
        return create(EmbedAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Embed; }
    Backend getBackend() const override { return backend_; }

    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::PlanarOnly;
    }

    const EmbedAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~Embed();

private:
    Embed(const EmbedAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    EmbedAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
