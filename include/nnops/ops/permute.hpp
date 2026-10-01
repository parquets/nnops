#pragma once
/// @file permute.hpp
/// @brief Permute operator — dimension reordering (transpose).
///
/// Rearranges tensor dimensions according to a given permutation order.
/// Equivalent to `np.transpose()` or `torch.permute()`.
///
/// Supports Planar layout only. For packed layouts, use LayoutConvert first.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/detail/small_vector.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes for the Permute operator.
struct PermuteAttributes {
    /// Permutation order. perm[i] specifies which input dimension
    /// maps to output dimension i.
    /// Default: empty (caller must provide valid permutation).
    detail::SmallVector<int64_t, TensorDesc::kMaxRank> perm;
};

/// Dimension reordering (transpose) operator.
///
/// Computes: output[d0, d1, ...] = input[perm[d0], perm[d1], ...]
///   where output.shape[i] = input.shape[perm[i]].
///
/// Input:  X [*]  (any rank, planar only)
/// Output: Y [*]  (same dtype, permuted shape, planar)
class Permute : public OpBase {
public:
    /// Create a Permute operator for the specified backend.
    static std::unique_ptr<Permute> create(const PermuteAttributes& attrs,
                                            Backend backend = Backend::CPU);

    /// Create with defaults.
    static std::unique_ptr<Permute> create(Backend backend) {
        return create(PermuteAttributes{}, backend);
    }

    // ---- OpBase interface ----
    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Permute; }
    Backend getBackend() const override { return backend_; }

    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::PlanarOnly;
    }

    const PermuteAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in permute.cpp (Pimpl pattern)
    ~Permute();

private:
    Permute(const PermuteAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    PermuteAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
