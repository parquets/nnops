#pragma once
/// @file permute.hpp
/// @brief Permute operator — dimension reordering (transpose).
///
/// Equivalent to `np.transpose()` / `torch.permute()`, with
/// output.shape[i] = input.shape[perm[i]]. Planar layout only; packed layouts
/// must go through LayoutConvert first.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/detail/small_vector.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

struct PermuteAttributes {
    /// perm[i] is the input dim mapped to output dim i; one entry per rank.
    detail::SmallVector<int64_t, TensorDesc::kMaxRank> perm;
};

class Permute : public OpBase {
public:
    static std::unique_ptr<Permute> create(const PermuteAttributes& attrs,
                                            Backend backend = Backend::CPU);

    static std::unique_ptr<Permute> create(Backend backend) {
        return create(PermuteAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Permute; }
    Backend getBackend() const override { return backend_; }

    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::PlanarOnly;
    }

    const PermuteAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in permute.cpp
    ~Permute();

private:
    Permute(const PermuteAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    PermuteAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
