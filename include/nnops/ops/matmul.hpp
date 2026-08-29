#pragma once
/// @file matmul.hpp
/// @brief MatMul operator: C = A × B  (matrix multiplication).

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/epilogue.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes for matrix multiplication.
///
/// Semantics: C = alpha × (A × B) + beta × bias
/// where bias is an optional third input broadcast over M rows.
struct MatMulAttributes {
    bool transpose_a = false;
    bool transpose_b = false;

    /// Scale factor for the A×B matrix product (fused into pack — zero overhead).
    float alpha = 1.0f;

    /// Scale factor for old C values: C_out = alpha×A×B + bias + beta×C_old.
    /// beta=0 (default) overwrites C; beta=1 adds A×B to existing C.
    float beta = 0.0f;

    /// Post-processing applied during output write-back (default: identity).
    Epilogue epilogue{};
};

/// Matrix multiplication operator (class-based API).
///
/// Supports N-D batch matmul with numpy-style broadcasting:
///   A: [..., M, K]  (or [..., K, M] if transpose_a)
///   B: [..., K, N]  (or [..., N, K] if transpose_b)
///   C: [..., M, N]
///
/// Leading batch dimensions are broadcast-compatible: each dim must be equal,
/// or one of them must be 1 (broadcast to the other), or one tensor may have
/// fewer dimensions (missing dims are treated as 1).
///
/// Examples:
///   - 2D × 2D:    [M,K] × [K,N] → [M,N]
///   - 3D × 3D:    [B,M,K] × [B,K,N] → [B,M,N]
///   - 2D × 3D:    [M,K] × [B,K,N] → [B,M,N]   (broadcast A across batch)
///   - 3D × 2D:    [B,M,K] × [K,N] → [B,M,N]   (broadcast B across batch)
///   - 4D × 3D:    [2,1,M,K] × [3,K,N] → [2,3,M,N]
class MatMul : public OpBase {
public:
    /// Create a MatMul operator for the specified backend.
    static std::unique_ptr<MatMul> create(const MatMulAttributes& attrs = {},
                                          Backend backend = Backend::CPU);

    // ---- OpBase interface ----
    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    size_t getWorkspaceSize(std::span<const TensorDesc> inputs,
                            std::span<const TensorDesc> outputs) const override;

    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::MatMul; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::PlanarOnly; }

    const MatMulAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in matmul.cpp (Pimpl pattern)

private:
    MatMul(const MatMulAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    MatMulAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
