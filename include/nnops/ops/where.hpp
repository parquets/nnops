#pragma once
/// @file where.hpp
/// @brief Where — select from x or y per element by a condition.
///
/// The condition broadcasts against x and y under the NumPy rules (all three
/// share one output shape). A condition element is true when its byte is
/// nonzero; the condition is u8/s8, the branches and the output share one dtype
/// (f32/f16/s8/u8/s32/s64). Planar.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

struct WhereAttributes {
};

class Where : public OpBase {
public:
    static std::unique_ptr<Where> create(const WhereAttributes& attrs,
                                         Backend backend = Backend::CPU);
    static std::unique_ptr<Where> create(Backend backend = Backend::CPU) {
        return create(WhereAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Where; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::PlanarOnly;
    }

    struct Impl;
    ~Where();

private:
    Where(const WhereAttributes& attrs, Backend backend);
    std::unique_ptr<Impl> impl_;
    WhereAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
