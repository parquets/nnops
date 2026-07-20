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
struct MatMulAttributes {
    bool transpose_a = false;
    bool transpose_b = false;

    /// Post-processing applied during output write-back (default: identity).
    Epilogue epilogue{};

    /// If true, add result to existing output buffer instead of overwriting.
    bool add_to = false;
};

/// Matrix multiplication operator (class-based API).
///
/// Computes: C = A × B
///   A: [M, K] (or [K, M] if transpose_a)
///   B: [K, N] (or [N, K] if transpose_b)
///   C: [M, N]
class MatMul : public OpBase {
public:
    /// Create a MatMul operator for the specified backend.
    static std::unique_ptr<MatMul> create(const MatMulAttributes& attrs = {},
                                          Backend backend = Backend::CPU);

    // ---- OpBase interface ----
    size_t getWorkspace() const override { return 0; }

    void compute(const TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::MatMul; }
    Backend getBackend() const override { return backend_; }

    const MatMulAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in matmul.cpp (Pimpl pattern)

private:
    MatMul(const MatMulAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    MatMulAttributes attrs_;
    Backend backend_;
};

/// Functional matmul.
void matmul(const TensorView& a,
            const TensorView& b,
            const TensorView& c,
            const MatMulAttributes& attrs = {},
            const ComputeContext& ctx = {});

}  // namespace nnops
