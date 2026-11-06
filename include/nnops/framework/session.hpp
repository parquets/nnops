#pragma once
/// @file session.hpp
/// @brief Session — binds a Model to buffers and executes it.
///
/// A Session lowers each graph node to an nnops operator once, at construction,
/// and then plans and runs per call:
///
///   1. shapes    — the graph's dynamic extents (-1) are resolved from the input
///                  tensors, and every value's descriptor is inferred in
///                  dependency order by the operators themselves
///                  (OpBase::getOutputTensorDesc), so the framework never
///                  re-implements an operator's shape rule;
///   2. placement — liveness is taken over the whole call and buffers are
///                  recycled greedily, so a residual connection's input and the
///                  next layer's scratch share one block. Graph inputs are
///                  never recycled: their storage is the caller's tensor, and
///                  the run reads it in place;
///   3. execute   — nodes run in dependency order through
///                  OpBase::compute(), the two metadata kinds (Shape / Cast)
///                  and constant values costing no kernel at all.
///
/// Shapes are resolved per call rather than cached, which is what lets one
/// session take any batch size without a re-plan path; the buffers themselves
/// are pooled, so a steady shape does not allocate after the first run.
///
/// A session reads its graph from the Model rather than copying it — the
/// constants in a graph are the weights, and duplicating them would double a
/// model's memory for nothing. The Model must therefore outlive the Session(s)
/// built from it.

#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/status.hpp"
#include "nnops/framework/graph.hpp"
#include "nnops/framework/model.hpp"
#include "nnops/framework/tensor.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace nnops::framework {

struct SessionOptions {
    /// Backend the operators are created for.
    Backend backend = Backend::CPU;

    /// Passed to every compute() call. Its cpu hooks carry the parallelism
    /// (see nnops/core/openmp_backend.hpp); the default is a single thread.
    ComputeContext ctx{};
};

class Session {
public:
    explicit Session(const Model& model,
                     const SessionOptions& opts = SessionOptions{});
    ~Session();

    Session(Session&&) noexcept;
    Session& operator=(Session&&) noexcept;
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    /// Execute with caller-owned outputs: `outputs` must match the graph's
    /// output descriptors in count, dtype and rank.
    Status run(std::span<const Tensor> inputs, std::span<Tensor> outputs);

    /// Execute and allocate the outputs; sized from the resolved graph shapes.
    Status run(std::span<const Tensor> inputs, std::vector<Tensor>& outputs);

    /// Graph inputs, in model order (dynamic extents are -1).
    const std::vector<ValueInfo>& input_info() const noexcept;
    const std::vector<ValueInfo>& output_info() const noexcept;

    /// Message for the last failure, construction included.
    const std::string& last_error() const noexcept;

    /// True when every node lowered to an operator.
    bool ok() const noexcept;

    /// High-water mark of the buffer plan over the calls so far, in bytes.
    size_t peak_buffer_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nnops::framework
