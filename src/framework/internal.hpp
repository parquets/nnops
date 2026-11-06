#pragma once
/// @file internal.hpp
/// @brief The framework's implementation-side declarations.
///
/// The four stages of a load — parse, import, optimize, lower — live in
/// separate translation units but share one vocabulary: a graph of Node and
/// ValueInfo, a handful of helpers that infer and execute a single node, and a
/// Status plus a message for anything that goes wrong. Keeping the node-level
/// helpers here is what lets constant folding evaluate a node exactly the way
/// the session will, instead of growing a second, subtly different evaluator.

#include "nnops/core/compute_context.hpp"
#include "nnops/core/op_base.hpp"
#include "nnops/core/status.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/framework/graph.hpp"
#include "nnops/framework/model.hpp"
#include "nnops/framework/tensor.hpp"
#include "nnops/ops/conv2d.hpp"

#include "onnx_proto.hpp"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace nnops::framework::detail {

/// ONNX GraphProto → the framework graph: values, constants and nodes, with
/// every node already assigned its kind and nnops OpType. No lowering and no
/// passes, so the result is the model as written. `opset` is the ai.onnx import
/// version, which decides between an operator's attribute and input spellings.
Status import_graph(const GraphProto& proto, const std::string& base_dir,
                    int64_t opset, Graph& out, std::string& error);

/// Resolve one node's output descriptors from its input descriptors — the
/// operators answer for themselves through OpBase::getOutputTensorDesc(); the
/// two framework kinds answer here. `impl` is the node's lowered operator, or
/// null for Shape and Cast.
Status infer_node(const Node& node, const OpBase* impl,
                  std::span<const TensorDesc> inputs,
                  std::vector<TensorDesc>& outputs, std::string& error);

/// Execute one node. Shape nodes are a no-op (their buffer is an alias the
/// planner established); Cast and Op dispatch to the framework conversion or
/// the lowered operator.
Status run_node(const Node& node, OpBase* impl, std::span<TensorView> outputs,
                std::span<const TensorView> inputs, const ComputeContext& ctx,
                void* workspace = nullptr);

/// A Conv node's attributes, without the epilogue (which is the node's own).
/// The session reads them back to weigh the route a constant filter could take,
/// so the policy has one spelling rather than two.
Conv2DAttributes conv2d_attributes(const Node& node);

/// Lower one node to the operator that runs it. `impl` is left null for the
/// framework kinds, which need no kernel.
Status lower_node(const Node& node, Backend backend,
                  std::unique_ptr<OpBase>& impl, std::string& error);

/// Element-wise dtype conversion (f32/f16/s8/u8/s32/s64).
Status cast_tensor(const TensorView& src, TensorView& dst);

/// Run the passes the load options select.
void optimize_graph(Graph& graph, const LoadOptions& opts);

}  // namespace nnops::framework::detail
