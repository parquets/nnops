#pragma once
/// @file depthwise_conv.hpp
/// @brief DepthwiseConv operator — per-channel 2D/3D convolution, channel-last.
///
/// One filter per input channel and no cross-channel mixing — the first half of
/// a depthwise-separable convolution. The output has the input's channel count
/// and layout; the optional bias is [C].
///
/// The spatial rank comes from the input rank: rank 4 is 2D
/// (Input [N, C, IH, IW] stored NHWC, Weight HWCM [KH][KW][C]), rank 5 is 3D
/// (Input [N, C, ID, IH, IW] stored NDHWC, Weight [KD][KH][KW][C]).
///
/// The route reads the raw HWCM filter: every tap is a contiguous run of C
/// channels, the pattern the channel-lane kernel loads as one vector. It covers
/// f32/f16 and the quantized s8/u8 pairs (PerTensor activation, PerChannel
/// weight).
///
/// The bias, the epilogue and the add_to residual all run in the output's domain,
/// and `residual_pre_activation` folds the destination's old contents into the
/// sum before the clamp and the activation on every route.

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

/// The spatial arrays are size 3, [KD, KH, KW]; for a 2D input KD/SD/DD/PD are
/// ignored.
struct DepthwiseConvAttributes {
    std::array<int64_t, 3> kernel_size = {1, 1, 1};

    std::array<int64_t, 3> stride = {1, 1, 1};

    std::array<int64_t, 3> dilation = {1, 1, 1};

    std::array<int64_t, 3> padding = {0, 0, 0};

    Epilogue epilogue{};

    /// Accumulate into the existing output instead of overwriting (residual).
    bool add_to = false;

    /// Order of the add_to term relative to the epilogue:
    ///   false: y = epilogue(conv) + residual
    ///   true:  y = epilogue(conv + residual)
    /// The pre-activation form is the ResNet block's `relu(dwconv + shortcut)`,
    /// which otherwise needs a separate add and activation. Only meaningful with
    /// add_to. The CPU routes implement it — f32/f16 and the quantized s8/u8
    /// pair — and the CUDA kernel refuses it rather than adding the residual on
    /// the wrong side.
    bool residual_pre_activation = false;

    static constexpr int64_t spatial_rank(int64_t input_rank) noexcept {
        return input_rank - 2;  // 4→2, 5→3
    }
};

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

    OpType  getOpType()  const override { return OpType::DepthwiseConv; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override
    {
        return LayoutSupport::ChannelLastOnly;
    }

    const DepthwiseConvAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~DepthwiseConv();

private:
    DepthwiseConv(const DepthwiseConvAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    DepthwiseConvAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
