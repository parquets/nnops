#pragma once
/// @file depthwise_conv.hpp
/// @brief DepthwiseConv operator — per-channel 2D/3D convolution, on a planar
///        (NCHW/NCDHW) or a channel-last (NHWC/NDHWC) tensor.
///
/// One filter per input channel and no cross-channel mixing — the first half of
/// a depthwise-separable convolution. The output has the input's channel count
/// and layout; the optional bias is [C].
///
/// The spatial rank comes from the input rank: rank 4 is 2D
/// (Input [N, C, IH, IW], Weight [C, 1, KH, KW]), rank 5 is 3D
/// (Input [N, C, ID, IH, IW], Weight [C, 1, KD, KH, KW]).
///
/// Both routes read the raw filter: a planar input the [C, 1, (KD,) KH, KW]
/// form as it is, a channel-last one the HWCM [KH][KW][C] form. The planar
/// route has f32/f16 kernels only; the channel lane also covers the quantized
/// s8/u8 pairs (PerTensor activation, PerChannel weight).

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
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::Any; }

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
