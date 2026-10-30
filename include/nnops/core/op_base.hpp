#pragma once
/// @file op_base.hpp
/// @brief OpBase — abstract base class for all operator implementations.
///
/// An operator never allocates a caller buffer: compute() takes pre-allocated outputs,
/// inputs, a ComputeContext, and a user workspace of getWorkspaceSize() bytes.
/// getOutputTensorDesc() infers each output's shape, dtype and layout from the
/// input descriptors (no data is read), which is what a graph engine needs to
/// plan buffers. validate() is the descriptor-only twin of compute()'s contract
/// check, so a malformed graph fails at plan time and the rules live in one
/// place. prepackWeights() is dual-behavior: an empty outputs[0] (is_empty() is
/// the query) is filled with shape/dtype/layout/pitch to allocate against, while
/// a populated one receives the packed weight data.

#include "nnops/core/tensor_view.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/op_type.hpp"
#include "nnops/core/status.hpp"
#include "nnops/detail/small_vector.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace nnops {

/// Abstract base class for all operators.
class OpBase {
public:
    virtual ~OpBase() = default;

    /// Infer output descriptors from input descriptors; no data is read.
    virtual std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const = 0;

    /// Workspace bytes required; receives descriptors (shapes + dtype, no data).
    /// Default 0 — override only if the operator needs scratch memory.
    virtual size_t getWorkspaceSize(std::span<const TensorDesc> inputs,
                                    std::span<const TensorDesc> outputs) const
    {
        return 0;
    }

    /// Check the call from descriptors only — no data read, nothing written.
    /// Default: ok.
    virtual Status validate(std::span<const TensorDesc> inputs,
                            std::span<const TensorDesc> outputs) const
    {
        (void)inputs;
        (void)outputs;
        return Status::ok;
    }

    /// Execute the operator; on error nothing is written to the outputs.
    virtual Status compute(std::span<TensorView> outputs,
                           std::span<const TensorView> inputs,
                           const ComputeContext& ctx = {},
                           void* workspace = nullptr) = 0;

    /// Convenience overload for single-output operators.
    Status compute(TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx = {},
                   void* workspace = nullptr)
    {
        TensorView outputs[] = {output};
        return compute(std::span<TensorView>(outputs), inputs, ctx, workspace);
    }

    /// Query (empty outputs[0]) or perform (populated) the weight prepack.
    virtual void prepackWeights(std::span<const TensorView> inputs,
                                std::span<TensorView> outputs,
                                const ComputeContext& ctx = {})
    {
        (void)inputs;
        (void)outputs;
        (void)ctx;
    }

    /// Layout capabilities this instance accepts; backend-specific. Default Any.
    virtual LayoutSupport getLayoutSupport() const noexcept { return LayoutSupport::Any; }

    /// ONNX-style operation type identifier.
    virtual OpType getOpType() const = 0;

    /// Backend this operator was created for.
    virtual Backend getBackend() const = 0;

protected:
    /// Build descriptors from the call's views (inline, no heap) and run validate().
    Status validate_views(std::span<TensorView> outputs,
                          std::span<const TensorView> inputs) const
    {
        // Inline covers every operator's fixed arity (MoE, at nine, is the widest).
        constexpr size_t kInline = 16;
        if (inputs.size() <= kInline && outputs.size() <= kInline) {
            detail::SmallVector<TensorDesc, kInline> in_d;
            detail::SmallVector<TensorDesc, kInline> out_d;
            for (const auto& v : inputs) { in_d.push_back(v.desc()); }
            for (const auto& v : outputs) { out_d.push_back(v.desc()); }
            return validate(std::span<const TensorDesc>(in_d.data(), in_d.size()),
                            std::span<const TensorDesc>(out_d.data(), out_d.size()));
        }
        // Wider than inline (a very wide Concat): one allocation, not a rejection.
        std::vector<TensorDesc> in_d;
        std::vector<TensorDesc> out_d;
        in_d.reserve(inputs.size());
        out_d.reserve(outputs.size());
        for (const auto& v : inputs) { in_d.push_back(v.desc()); }
        for (const auto& v : outputs) { out_d.push_back(v.desc()); }
        return validate(in_d, out_d);
    }
};

}  // namespace nnops
