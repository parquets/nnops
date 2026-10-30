#pragma once
/// @file layout_convert.hpp
/// @brief LayoutConvert operator — pack/unpack between planar and C8-blocked.
///
/// NCHW ↔ NCHWC8 (2D) and NCDHW ↔ NCDHWC8 (3D). The logical shape is
/// preserved; only the physical layout changes. On pack the trailing C8 block
/// is zero-padded when C is not a multiple of 8; unpack drops the pad lanes.
/// f32, f16, s8 and u8 — on the s8/u8 path the pad lanes take the activation
/// zero point, so a padded lane dequantizes to zero contribution.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/tensor_layout.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <memory>
#include <span>

namespace nnops {

struct LayoutConvertAttributes {
    TensorLayout target_layout = TensorLayout::NCHW;
};

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

    Status validate(std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs) const override;

    using OpBase::compute;

    Status compute(std::span<TensorView> outputs,
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
