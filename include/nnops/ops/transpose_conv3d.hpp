#pragma once
/// @file transpose_conv3d.hpp
/// @brief TransposeConv3D — transposed 3D convolution (deconvolution).
///
/// The 3-D form of TransposeConv2D, in ONNX's ConvTranspose convention: the
/// image is [N, IC, ID, IH, IW], logical channels first and stored channel-first
/// (NCDHW) or channel-last (NDHWC); the weight is channel-first
/// [IC, OC/G, KD, KH, KW] — input channel outermost, the opposite of Conv3D's
/// weight — in both layouts; and the output is [N, OC, OD, OH, OW] in the input's
/// layout, bias [OC] optional. Grouped, dilated and strided, with output_padding
/// resolving the output size when the stride does not:
///   OD = (ID - 1)·SD - 2·PD + DD·(KD - 1) + output_padding[0] + 1.
///
/// Data types: f32/f16. The floating-point route only, like Conv3D — there is no
/// quantized 3-D route, and the operator is refused on any other layout or dtype.

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

/// Attributes describing a 3D transposed convolution.
struct TransposeConv3DAttributes {
    /// Kernel size [KD, KH, KW].
    std::array<int64_t, 3> kernel_size = {1, 1, 1};

    /// Stride [SD, SH, SW].
    std::array<int64_t, 3> stride = {1, 1, 1};

    /// Dilation [DD, DH, DW].
    std::array<int64_t, 3> dilation = {1, 1, 1};

    /// Symmetric zero-padding [PD, PH, PW]; subtracted from the output (the
    /// opposite of regular conv).
    std::array<int64_t, 3> padding = {0, 0, 0};

    /// Extra size on each spatial dimension's trailing edge, to disambiguate
    /// the output size when stride > 1.
    std::array<int64_t, 3> output_padding = {0, 0, 0};

    /// Groups for grouped / depthwise transpose convolution.
    int64_t groups = 1;

    /// Post-processing applied during output write-back (default: identity).
    Epilogue epilogue{};

    /// If true, add into the existing output buffer instead of overwriting.
    bool add_to = false;

    /// Order of the add_to term relative to the epilogue:
    ///   false: y = epilogue(conv) + residual
    ///   true:  y = epilogue(conv + residual)
    /// Only meaningful with add_to.
    bool residual_pre_activation = false;
};

class TransposeConv3D : public OpBase {
public:
    static std::unique_ptr<TransposeConv3D> create(
        const TransposeConv3DAttributes& attrs,
        Backend backend = Backend::CPU);

    static std::unique_ptr<TransposeConv3D> create(Backend backend = Backend::CPU) {
        return create(TransposeConv3DAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::TransposeConv3D; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::ChannelFirstOrChannelLast;
    }

    const TransposeConv3DAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in transpose_conv3d.cpp
    ~TransposeConv3D();

private:
    TransposeConv3D(const TransposeConv3DAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    TransposeConv3DAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
