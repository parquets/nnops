#pragma once
/// @file clamp.hpp
/// @brief Element-wise clamp to [min, max].
///
/// output[i] = max(min_val, min(max_val, input[i])). Any shape, rank or layout
/// (Planar and Packed), processed a row at a time by row stride; output matches
/// input.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes for the Clamp operator.
struct ClampAttributes {
    float min_val = -std::numeric_limits<float>::infinity();
    float max_val = std::numeric_limits<float>::infinity();

    /// Accumulate into the existing output instead of overwriting.
    bool add_to = false;
};

class Clamp : public OpBase {
public:
    static std::unique_ptr<Clamp> create(const ClampAttributes& attrs = {},
                                          Backend backend = Backend::CPU);

    static std::unique_ptr<Clamp> create(Backend backend) {
        return create(ClampAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Clamp; }
    Backend getBackend() const override { return backend_; }

    const ClampAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in clamp.cpp (Pimpl pattern)
    ~Clamp();

private:
    Clamp(const ClampAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    ClampAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
