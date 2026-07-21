#pragma once
/// @file op_base.hpp
/// @brief OpBase — abstract base class for all operator implementations.

#include "nnops/core/tensor_view.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/op_type.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace nnops {

/// Abstract base class for all operators.
///
/// Each operator:
///   - Is created via a static create() factory method.
///   - Reports its workspace requirements via getWorkspace().
///   - Executes via compute(), which takes pre-allocated output tensors,
///     input tensors, a compute context, and a user-provided workspace buffer.
///   - Does NOT internally allocate memory — the user manages all buffers.
class OpBase {
public:
    virtual ~OpBase() = default;

    /// Returns the workspace size in bytes required by this operator.
    /// For CPU backends, this may depend on the parallel_for grain size.
    /// Call after create() to allocate workspace before calling compute().
    virtual size_t getWorkspace() const = 0;

    /// Execute the operator (primary virtual — multi-output).
    ///
    /// @param outputs   Pre-allocated output tensors. User manages memory.
    ///                  Single-output operators use outputs[0].
    /// @param inputs    Input tensors (immutable during compute). Number and
    ///                  meaning depend on the specific operator.
    /// @param ctx       Backend-specific execution context (stream, thread pool, etc.)
    /// @param workspace Optional pre-allocated scratch buffer of at least
    ///                  getWorkspace() bytes. May be nullptr if getWorkspace() == 0.
    virtual void compute(std::span<const TensorView> outputs,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx = {},
                         void* workspace = nullptr) = 0;

    /// Convenience overload for single-output operators.
    /// Calls the span-based virtual above — no override needed in subclasses.
    void compute(const TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr)
    {
        const TensorView outputs[] = {output};
        compute(std::span<const TensorView>(outputs), inputs, ctx, workspace);
    }

    /// Returns the ONNX-style operation type identifier.
    virtual OpType getOpType() const = 0;

    /// Returns the backend this operator was created for.
    virtual Backend getBackend() const = 0;
};

}  // namespace nnops
