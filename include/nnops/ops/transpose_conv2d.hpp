#pragma once
/// @file transpose_conv2d.hpp
/// @brief TransposeConv2D — transposed 2D convolution (deconvolution).
///
/// The backward of a regular convolution: each input pixel is scaled by the
/// kernel and scattered/accumulated over a region of the output (one-to-many).
/// NCHWC8 packed input/output; weight is planar [IC, OC/G, KH, KW]; bias [OC]
/// optional. Reference: ncnn Deconvolution (scatter-add approach).

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

/// Attributes describing a 2D transposed convolution.
struct TransposeConv2DAttributes {
    /// Kernel size [KH, KW].
    std::array<int64_t, 2> kernel_size = {1, 1};

    /// Stride [SH, SW].
    std::array<int64_t, 2> stride = {1, 1};

    /// Dilation [DH, DW].
    std::array<int64_t, 2> dilation = {1, 1};

    /// Symmetric zero-padding [PH, PW]; subtracted from the output (the
    /// opposite of regular conv).
    std::array<int64_t, 2> padding = {0, 0};

    /// Extra size on each spatial dimension's trailing edge, to disambiguate
    /// the output size when stride > 1.
    std::array<int64_t, 2> output_padding = {0, 0};

    /// Groups for grouped / depthwise transpose convolution.
    int64_t groups = 1;

    /// Post-processing applied during output write-back (default: identity).
    Epilogue epilogue{};

    /// If true, add into the existing output buffer instead of overwriting.
    bool add_to = false;
};

class TransposeConv2D : public OpBase {
public:
    static std::unique_ptr<TransposeConv2D> create(
        const TransposeConv2DAttributes& attrs,
        Backend backend = Backend::CPU);

    static std::unique_ptr<TransposeConv2D> create(Backend backend = Backend::CPU) {
        return create(TransposeConv2DAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::TransposeConv2D; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::PackedOnly; }

    const TransposeConv2DAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in transpose_conv2d.cpp
    ~TransposeConv2D();

private:
    TransposeConv2D(const TransposeConv2DAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    TransposeConv2DAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
