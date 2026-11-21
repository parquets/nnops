#pragma once
/// @file pixel_shuffle.hpp
/// @brief SpaceToDepth and PixelShuffle — the inverse pair that trades spatial
///        resolution for channels.
///
/// Both read a channel-last (NHWC) activation and write one; neither changes
/// the batch axis:
///
///   SpaceToDepth  [N, H, W, C]           -> [N, H/b, W/b, C·b²]
///   PixelShuffle  [N, H, W, C_out·b²]    -> [N, H·b, W·b, C_out]
///
/// SpaceToDepth (ONNX's op, channel-last) with block b:
///   Y[n, h, w, c·b² + bh·b + bw] = X[n, h·b + bh, w·b + bw, c]
/// H and W must be divisible by b. This is the patch-merging downsample of a
/// ViT / VLM: fold every b×b patch into one token's channels.
///
/// PixelShuffle is ONNX's DepthToSpace and torch.pixel_shuffle, with its two
/// channel orderings:
///   DCR (default): Y[n, h·b + bh, w·b + bw, c] = X[n, h, w, c·b² + bh·b + bw]
///   CRD:           Y[n, h·b + bh, w·b + bw, c] = X[n, h, w, (bh·b + bw)·C_out + c]
/// DCR is PyTorch's `pixel_shuffle` and ONNX's default; CRD is ONNX's
/// alternative. C must be divisible by b². SpaceToDepth(b) followed by
/// PixelShuffle(b, DCR) is the identity.
///
/// Both are pure element permutations: any dtype the movement family carries
/// (f32 / f16 / s8 / u8 / s32 / s64, plus bf16 on aarch64) works, and a pitched
/// NHWC view is walked by its strides. Rank 4 only — the 3-D (NDHWC) forms a
/// video backbone would want are not implemented.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

struct SpaceToDepthAttributes {
    /// Side of the b×b pixel block folded into channels; >= 1.
    int64_t block_size = 2;
};

class SpaceToDepth : public OpBase {
public:
    static std::unique_ptr<SpaceToDepth> create(const SpaceToDepthAttributes& attrs,
                                                Backend backend = Backend::CPU);

    static std::unique_ptr<SpaceToDepth> create(Backend backend = Backend::CPU) {
        return create(SpaceToDepthAttributes{}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    Status validate(std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs) const override;

    using OpBase::compute;

    Status compute(std::span<TensorView> outputs,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx = {},
                   void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::SpaceToDepth; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::ChannelLastOnly;
    }

    const SpaceToDepthAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~SpaceToDepth();

private:
    SpaceToDepth(const SpaceToDepthAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    SpaceToDepthAttributes attrs_;
    Backend backend_;
};

/// Channel ordering of PixelShuffle / DepthToSpace.
enum class PixelShuffleMode : uint8_t {
    Dcr,  ///< channel = c_out·b² + bh·b + bw (PyTorch, ONNX default)
    Crd,  ///< channel = (bh·b + bw)·C_out + c_out (ONNX's alternative)
};

struct PixelShuffleAttributes {
    /// Upsampling factor; >= 1. The input channels must divide by b².
    int64_t block_size = 2;
    PixelShuffleMode mode = PixelShuffleMode::Dcr;
};

class PixelShuffle : public OpBase {
public:
    static std::unique_ptr<PixelShuffle> create(const PixelShuffleAttributes& attrs,
                                                Backend backend = Backend::CPU);

    static std::unique_ptr<PixelShuffle> create(Backend backend = Backend::CPU) {
        return create(PixelShuffleAttributes{}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    Status validate(std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs) const override;

    using OpBase::compute;

    Status compute(std::span<TensorView> outputs,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx = {},
                   void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::PixelShuffle; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::ChannelLastOnly;
    }

    const PixelShuffleAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~PixelShuffle();

private:
    PixelShuffle(const PixelShuffleAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    PixelShuffleAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
