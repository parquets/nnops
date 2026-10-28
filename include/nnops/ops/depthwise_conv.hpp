#pragma once
/// @file depthwise_conv.hpp
/// @brief DepthwiseConv operator — depthwise separable 2D/3D convolution, on
///        either a planar (NCHW/NCDHW) or a packed (NCHWC8/NCDHWC8) tensor.
///
/// Depthwise convolution applies a separate filter to each input channel independently.
/// There is no cross-channel mixing. This is the first step of a depthwise-separable
/// convolution (the second step being a 1x1 pointwise conv).
///
/// Spatial rank is auto-detected from input tensor rank:
///   2D (rank 4): Input [N, C, IH, IW], Weight [C, 1, KH, KW]
///   3D (rank 5): Input [N, C, ID, IH, IW], Weight [C, 1, KD, KH, KW]
/// Output channels equal input channels. Optional bias: [C].
///
/// A packed input takes the weights prepacked by prepackWeights(); a planar
/// input takes the raw weight and bias, and supports f32/f16 only.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/epilogue.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes describing a depthwise convolution (2D or 3D).
///
/// All spatial attribute arrays have size 3: [KD, KH, KW].
/// For 2D input, KD/SD/DD/PD are ignored.
struct DepthwiseConvAttributes {
    /// Kernel size along depth, height, and width.
    std::array<int64_t, 3> kernel_size = {1, 1, 1};

    /// Stride along depth, height, and width.
    std::array<int64_t, 3> stride = {1, 1, 1};

    /// Dilation along depth, height, and width.
    std::array<int64_t, 3> dilation = {1, 1, 1};

    /// Symmetric zero-padding along depth, height, and width.
    std::array<int64_t, 3> padding = {0, 0, 0};

    /// Post-processing applied during output write-back (default: identity).
    Epilogue epilogue{};

    /// If true, add result to existing output buffer instead of overwriting.
    /// Enables residual connections without a separate add kernel:
    ///   output += DepthwiseConv(input)  rather than  output = DepthwiseConv(input)
    bool add_to = false;

    /// Return the spatial rank (2 or 3) given the input tensor rank.
    static constexpr int64_t spatial_rank(int64_t input_rank) noexcept {
        return input_rank - 2;  // 4→2, 5→3
    }
};

/// DepthwiseConv operator (class-based API).
///
/// Each input channel is convolved with its own independent filter, with no
/// cross-channel mixing; the input's layout (planar or packed) is the output's.
///
/// 2D: Input [N, C, IH, IW], Weight [C, 1, KH, KW], Output [N, C, OH, OW].
/// 3D: Input [N, C, ID, IH, IW], Weight [C, 1, KD, KH, KW], Output [N, C, OD, OH, OW].
/// Optional bias: [C].
class DepthwiseConv : public OpBase {
public:
    static std::unique_ptr<DepthwiseConv> create(
        const DepthwiseConvAttributes& attrs,
        Backend backend = Backend::CPU);

    static std::unique_ptr<DepthwiseConv> create(Backend backend = Backend::CPU) {
        return create(DepthwiseConvAttributes{}, backend);
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

    void prepackWeights(std::span<const TensorView> inputs,
                        std::span<TensorView> outputs,
                        const ComputeContext& ctx = {}) override;

    OpType  getOpType()  const override { return OpType::DepthwiseConv; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::Any; }

    const DepthwiseConvAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in depthwise_conv.cpp (Pimpl pattern)
    ~DepthwiseConv();

private:
    DepthwiseConv(const DepthwiseConvAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    DepthwiseConvAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
