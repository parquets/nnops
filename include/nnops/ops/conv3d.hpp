#pragma once
/// @file conv3d.hpp
/// @brief 3D convolution with NCDHW layout.
///
/// Input [N, IC, ID, IH, IW], weight [OC, IC/G, KD, KH, KW] (ONNX-style, also
/// when grouped), output [N, OC, OD, OH, OW], optional bias [OC]. Grouped and
/// dilated; planar layouts only.

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

/// Attributes describing a 3D convolution.
struct Conv3DAttributes {
    std::array<int64_t, 3> kernel_size = {1, 1, 1};
    std::array<int64_t, 3> stride = {1, 1, 1};
    std::array<int64_t, 3> dilation = {1, 1, 1};
    std::array<int64_t, 3> padding = {0, 0, 0};
    int64_t groups = 1;

    enum class AutoPad : uint8_t {
        NOTSET = 0,
        SAME_UPPER,
        SAME_LOWER,
        VALID,
    };
    AutoPad auto_pad = AutoPad::NOTSET;

    Epilogue epilogue{};

    /// Accumulate into the existing output instead of overwriting.
    bool add_to = false;
};

class Conv3D : public OpBase {
public:
    static std::unique_ptr<Conv3D> create(const Conv3DAttributes& attrs,
                                          Backend backend = Backend::CPU);

    static std::unique_ptr<Conv3D> create(Backend backend = Backend::CPU) {
        return create(Conv3DAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Conv3D; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::PlanarOnly; }

    const Conv3DAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in conv3d.cpp (Pimpl pattern)
    ~Conv3D();

private:
    Conv3D(const Conv3DAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    Conv3DAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
