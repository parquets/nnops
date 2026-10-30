#pragma once
/// @file flatten.hpp
/// @brief Flatten operator — any layout to dense contiguous planar data.
///
/// The counterpart of `np.ascontiguousarray()` / `.contiguous()`: channel
/// packing is unrolled and pitch padding removed, so the output carries the
/// input's logical shape and dtype. Inputs may be NCHW, NCHWC8, NCDHW or
/// NCDHWC8, padded or not; f32 and f16 only.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <memory>
#include <span>

namespace nnops {

struct FlattenAttributes {
    // Reserved for future options (e.g. flatten_to_1d, target_dtype).
};

class Flatten : public OpBase {
public:
    static std::unique_ptr<Flatten> create(const FlattenAttributes& attrs = {},
                                            Backend backend = Backend::CPU);

    static std::unique_ptr<Flatten> create(Backend backend) {
        return create(FlattenAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Flatten; }
    Backend getBackend() const override { return backend_; }

    const FlattenAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~Flatten();

private:
    Flatten(const FlattenAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    FlattenAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
