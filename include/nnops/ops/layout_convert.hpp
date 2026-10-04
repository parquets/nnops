#pragma once
/// @file layout_convert.hpp
/// @brief LayoutConvert operator — convert tensor data between memory layouts.
///
/// Supports conversions between NCHW ↔ NCHWC8 (2D) and NCDHW ↔ NCDHWC8 (3D)
/// for f32 and f16 data types. The logical shape is preserved; only the
/// physical memory layout changes.
///
/// Supported layout pairs:
///   NCHW   ↔ NCHWC8     (2D pack/unpack)
///   NCDHW  ↔ NCDHWC8    (3D pack/unpack)

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/tensor_layout.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <memory>
#include <span>

namespace nnops {

/// Attributes for the LayoutConvert operator.
struct LayoutConvertAttributes {
    /// Target memory layout for the output tensor.
    TensorLayout target_layout = TensorLayout::NCHW;
};

/// LayoutConvert operator (class-based API).
///
/// Converts tensor data from the input layout to the target layout.
/// The logical shape [N, C, (D,) H, W] is preserved; only the physical
/// memory layout changes. For pack operations (NCHW → NCHWC8), the
/// last C8 block is zero-padded if C is not a multiple of pack_size.
class LayoutConvert : public OpBase {
public:
    static std::unique_ptr<LayoutConvert> create(
        const LayoutConvertAttributes& attrs,
        Backend backend = Backend::CPU);

    static std::unique_ptr<LayoutConvert> create(
        TensorLayout target_layout,
        Backend backend = Backend::CPU) {
        return create(LayoutConvertAttributes{target_layout}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::LayoutConvert; }
    Backend getBackend() const override { return backend_; }

    const LayoutConvertAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~LayoutConvert();

private:
    LayoutConvert(const LayoutConvertAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    LayoutConvertAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
