#pragma once
/// @file cumsum.hpp
/// @brief Cumulative sum along one axis.
///
/// inclusive: out[i] = sum(in[0..i]); exclusive: out[0] = 0, out[i] =
/// sum(in[0..i-1]). reverse walks from the last element backward. Input rank
/// >= 1; output has the input's shape. Planar layouts only.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes for the CumSum operator.
struct CumSumAttributes {
    bool exclusive = false;
    bool reverse = false;
    int64_t axis = 0;
};

class CumSum : public OpBase {
public:
    static std::unique_ptr<CumSum> create(const CumSumAttributes& attrs = {},
                                          Backend backend = Backend::CPU);

    static std::unique_ptr<CumSum> create(Backend backend) {
        return create(CumSumAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::CumSum; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::PlanarOnly; }

    const CumSumAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in cumsum.cpp (Pimpl pattern)
    ~CumSum();

private:
    CumSum(const CumSumAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    CumSumAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
