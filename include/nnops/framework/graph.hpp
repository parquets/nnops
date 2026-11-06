#pragma once
/// @file graph.hpp
/// @brief The framework's graph IR — values, nodes and attributes.
///
/// The IR sits between the ONNX file and nnops operators, and is the unit the
/// optimizer works on. A node keeps the ONNX op type (so passes can reason in
/// the vocabulary the model was written in), the nnops OpType it lowers to, and
/// its attributes as a small name→value list. Three node kinds exist:
///
///   Op    — runs an nnops operator; getOutputTensorDesc() infers its outputs.
///   Shape — a metadata-only node (Reshape, Flatten, Squeeze, Identity, …): the
///           output *is* the input's bytes under a different descriptor, so the
///           planner aliases the two onto one buffer and execution is a no-op.
///   Cast  — element-wise dtype conversion; the framework's own kernel, since
///           no nnops operator covers it.
///
/// Values carry a TensorDesc whose dims may be negative: -1 marks a dynamic
/// extent that the session resolves from the shapes it is handed at run time.
/// Constant values (initializers and anything folding produced) keep their bytes
/// in the graph, side by side with their descriptor.

#include "nnops/core/data_type.hpp"
#include "nnops/core/epilogue.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/op_type.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nnops::framework {

/// One attribute of a graph node.
struct Attribute {
    enum class Type : uint8_t { Int, Float, String, Ints, Floats };

    Type type = Type::Int;
    int64_t i = 0;
    float f = 0.0f;
    std::string s;
    std::vector<int64_t> ints;
    std::vector<float> floats;
};

/// A node's attribute list — insertion-ordered, linear lookup (a node has a
/// handful, and a map would cost more than it saves).
class Attributes {
public:
    bool has(std::string_view name) const noexcept;

    int64_t get_int(std::string_view name, int64_t fallback = 0) const noexcept;
    float get_float(std::string_view name, float fallback = 0.0f) const noexcept;
    std::string_view get_string(std::string_view name) const noexcept;
    std::span<const int64_t> get_ints(std::string_view name) const noexcept;
    std::span<const float> get_floats(std::string_view name) const noexcept;

    void set_int(std::string_view name, int64_t value);
    void set_float(std::string_view name, float value);
    void set_string(std::string_view name, std::string value);
    void set_ints(std::string_view name, std::vector<int64_t> value);
    void set_floats(std::string_view name, std::vector<float> value);

    void erase(std::string_view name);

    const std::vector<std::pair<std::string, Attribute>>& items() const noexcept
    {
        return items_;
    }

private:
    Attribute* find(std::string_view name) noexcept;
    const Attribute* find(std::string_view name) const noexcept;
    Attribute& obtain(std::string_view name);

    std::vector<std::pair<std::string, Attribute>> items_;
};

/// What a node resolves to at execution time.
enum class NodeKind : uint8_t {
    Op = 0,    ///< An nnops operator (getOutputTensorDesc + compute).
    Shape = 1, ///< Metadata only: output aliases input 0 under a new descriptor.
    Cast = 2,  ///< Element-wise dtype conversion.
};

/// A graph node.
struct Node {
    std::string name;    ///< ONNX node name, may be empty.
    std::string op_type; ///< ONNX op type ("Conv", "Relu", …).
    NodeKind kind = NodeKind::Op;
    OpType op = OpType::Conv2D; ///< nnops operator; meaningful for NodeKind::Op.

    std::vector<int> inputs;  ///< Value ids; -1 for an omitted optional input.
    std::vector<int> outputs; ///< Value ids, in the operator's output order.

    Attributes attrs;

    /// NodeKind::Shape: value whose bytes this node's output reinterprets.
    int alias_of = -1;

    /// NodeKind::Cast destination dtype.
    DataType target_dtype = DataType::f32;

    /// Activation fused into this node's epilogue by the optimizer; the node
    /// that carried it is removed, and -1 means nothing was fused.
    EpilogueActivateType fused_activation = EpilogueActivateType::None;
    float fused_alpha = 0.0f;
    float fused_beta = 1.0f;
};

/// A tensor slot: name, descriptor and whether the bytes live in the graph.
struct ValueInfo {
    std::string name;
    TensorDesc desc;        ///< dims may contain -1 for a dynamic extent.
    bool constant = false;  ///< true when data() is populated.
};

/// The graph: values, nodes, the input/output value lists, and constant bytes.
class Graph {
public:
    std::vector<ValueInfo>& values() noexcept { return values_; }
    const std::vector<ValueInfo>& values() const noexcept { return values_; }

    std::vector<Node>& nodes() noexcept { return nodes_; }
    const std::vector<Node>& nodes() const noexcept { return nodes_; }

    /// Graph inputs (data supplied at run time), in model order.
    std::vector<int>& inputs() noexcept { return inputs_; }
    const std::vector<int>& inputs() const noexcept { return inputs_; }

    /// Graph outputs, in model order.
    std::vector<int>& outputs() noexcept { return outputs_; }
    const std::vector<int>& outputs() const noexcept { return outputs_; }

    /// Constant bytes for value `id`; empty for a non-constant value.
    std::vector<std::byte>& constant_data(int id) noexcept
    {
        return constant_data_[static_cast<size_t>(id)];
    }
    const std::vector<std::byte>& constant_data(int id) const noexcept
    {
        return constant_data_[static_cast<size_t>(id)];
    }

    /// Append a value; returns its id.
    int add_value(ValueInfo info);

    /// Look a value up by name; -1 when absent.
    int find_value(std::string_view name) const noexcept;

    /// Node ids in dependency order. Empty when the graph has a cycle.
    std::vector<int> topological_order() const;

    /// Drop nodes whose outputs nothing consumes; values are left in place so
    /// ids stay stable.
    void remove_dead_nodes();

private:
    std::vector<ValueInfo> values_;
    std::vector<std::vector<std::byte>> constant_data_;
    std::vector<Node> nodes_;
    std::vector<int> inputs_;
    std::vector<int> outputs_;
};

}  // namespace nnops::framework
