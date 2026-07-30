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
#include <vector>

namespace nnops {

/// Abstract base class for all operators.
///
/// Each operator:
///   - Is created via a static create() factory method.
///   - Reports its workspace requirements via getWorkspaceSize().
///   - Executes via compute(), which takes pre-allocated output tensors,
///     input tensors, a compute context, and a user-provided workspace buffer.
///   - Does NOT internally allocate memory — the user manages all buffers.
///   - Provides shape inference via getOutputShapes() — given input
///     TensorDescs (shapes + dtype, no data), returns the expected output
///     TensorDescs. This enables graph-level shape propagation and memory
///     planning before execution.
class OpBase {
public:
    virtual ~OpBase() = default;

    /// Compute the output tensor descriptors from input tensor descriptors.
    ///
    /// This is pure shape inference — no data buffers are accessed. The
    /// returned TensorDesc values describe the expected shape, dtype, and
    /// layout of each output tensor. Used by graph engines and runtime
    /// memory planners to pre-allocate output buffers.
    ///
    /// Most operators produce exactly one output. A few (e.g. training-mode
    /// BatchNorm) may produce multiple outputs.
    ///
    /// @param inputs  Input tensor descriptors (shapes + dtype + layout).
    /// @return        Output tensor descriptors. Size is typically 1.
    virtual std::vector<TensorDesc> getOutputShapes(
        std::span<const TensorDesc> inputs) const = 0;

    /// Returns the workspace size in bytes required by this operator.
    /// Receives input/output tensor descriptors (shapes + dtype, no data) to
    /// compute workspace from actual dimensions. Analogous to TensorRT's
    /// IPluginV2DynamicExt::getWorkspaceSize.
    /// Default returns 0 — override only if the operator needs scratch memory.
    virtual size_t getWorkspaceSize(std::span<const TensorDesc> inputs,
                                    std::span<const TensorDesc> outputs) const
    {
        return 0;
    }

    /// Execute the operator (primary virtual — multi-output).
    ///
    /// @param outputs   Pre-allocated output tensors. User manages memory.
    ///                  Single-output operators use outputs[0].
    /// @param inputs    Input tensors (immutable during compute). Number and
    ///                  meaning depend on the specific operator.
    /// @param ctx       Backend-specific execution context (stream, thread pool, etc.)
    /// @param workspace Optional pre-allocated scratch buffer of at least
    ///                  getWorkspaceSize() bytes. May be nullptr if size == 0.
    virtual void compute(std::span<TensorView> outputs,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx = {},
                         void* workspace = nullptr) = 0;

    /// Convenience overload for single-output operators.
    /// Calls the span-based virtual above — no override needed in subclasses.
    void compute(TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr)
    {
        TensorView outputs[] = {output};
        compute(std::span<TensorView>(outputs), inputs, ctx, workspace);
    }

    /// Prepack weights into an optimized format for the target backend.
    ///
    /// Dual-behavior API — the caller discovers the required buffer, then
    /// fills it, in two steps:
    ///
    ///   Step 1 — query shape/size: pass an empty outputs[0] (is_empty()
    ///   returns true). The function fills outputs[0] with shape, dtype,
    ///   layout, and pitch so the caller can allocate the exact buffer.
    ///
    ///   Step 2 — perform prepack: pass an outputs[0] whose data pointer
    ///   points to a buffer of at least the size described in step 1.
    ///   The function writes the prepacked weight data into that buffer.
    ///
    /// @param inputs   Raw weight tensor(s) in standard layout (e.g. NCHW).
    /// @param outputs  outputs[0] — empty (query) or pre-allocated (prepack).
    /// @param ctx      Compute context (may be unused).
    virtual void prepackWeights(std::span<const TensorView> inputs,
                                std::span<TensorView> outputs,
                                const ComputeContext& ctx = {})
    {
        (void)inputs;
        (void)outputs;
        (void)ctx;
    }

    /// Returns the ONNX-style operation type identifier.
    virtual OpType getOpType() const = 0;

    /// Returns the backend this operator was created for.
    virtual Backend getBackend() const = 0;
};

}  // namespace nnops
