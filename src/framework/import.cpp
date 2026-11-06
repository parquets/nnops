/// @file import.cpp
/// @brief ONNX GraphProto → the framework graph.
///
/// This is the only place that speaks ONNX. Every node is translated into the
/// library's own vocabulary here: an OpType plus a normalised attribute set
/// (defaults filled in, 2D geometry folded into the 3-slot window arrays, a
/// multi-axis reduce expanded into a chain), or one of the framework's
/// metadata kinds for the work that is not a kernel. Nothing downstream has to
/// know what "axis" meant in which opset — that whole question is answered once,
/// here, against `opset_`.
///
/// The typed attribute structs are built in lowering.cpp, not here: import
/// decides *what* a node means, lowering decides which operator and which
/// fields carry it. The split keeps this file free of nnops operator headers
/// beyond the operator-type enums.
///
/// Not modelled, and reported by name rather than approximated: control-flow
/// subgraphs (If/Loop), sparse tensors, external weight files (read, but only
/// when the model names them), and the pre-opset-13 Softmax that flattens its
/// input to 2-D at `axis`.

#include "internal.hpp"

#include "nnops/ops/activation.hpp"
#include "nnops/ops/eltwise.hpp"
#include "nnops/ops/pooling.hpp"
#include "nnops/ops/reduce.hpp"
#include "nnops/ops/unary.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <unordered_map>

namespace nnops::framework::detail {

namespace {

// ---- Data types ------------------------------------------------------------

bool map_elem_type(int32_t onnx_type, DataType& out) noexcept
{
    switch (onnx_type) {
    case 1:  out = DataType::f32;  return true;  // FLOAT
    case 2:  out = DataType::u8;   return true;  // UINT8
    case 3:  out = DataType::s8;   return true;  // INT8
    case 6:  out = DataType::s32;  return true;  // INT32
    case 7:  out = DataType::s64;  return true;  // INT64
    case 9:  out = DataType::u8;   return true;  // BOOL
    case 10: out = DataType::f16;  return true;  // FLOAT16
    case 16: out = DataType::bf16; return true;  // BFLOAT16
    default: return false;
    }
}

const char* onnx_type_name(int32_t onnx_type) noexcept
{
    switch (onnx_type) {
    case 1: return "FLOAT";
    case 2: return "UINT8";
    case 3: return "INT8";
    case 4: return "UINT16";
    case 5: return "INT16";
    case 6: return "INT32";
    case 7: return "INT64";
    case 9: return "BOOL";
    case 10: return "FLOAT16";
    case 11: return "DOUBLE";
    case 12: return "UINT32";
    case 13: return "UINT64";
    case 16: return "BFLOAT16";
    default: return "unknown";
    }
}

int64_t tensor_numel(const TensorProto& t) noexcept
{
    int64_t n = 1;
    for (int64_t d : t.dims) { n *= d; }
    return n;
}

// ---- Payload conversion ----------------------------------------------------

bool read_external_blob(const TensorProto& t, const std::string& base_dir,
                        std::string& payload, std::string& error)
{
    if (t.external_location.empty()) {
        error = "tensor '" + t.name + "' is external but names no location";
        return false;
    }
    std::string path = t.external_location;
    const bool absolute = !path.empty() &&
        (path.front() == '/' || path.front() == '\\' ||
         (path.size() > 1 && path[1] == ':'));
    if (!absolute) { path = base_dir + "/" + path; }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error = "cannot open external tensor data '" + path + "'";
        return false;
    }
    file.seekg(0, std::ios::end);
    const auto size = static_cast<int64_t>(file.tellg());
    const int64_t offset = t.external_offset;
    int64_t length = t.external_length;
    if (length < 0) { length = size - offset; }
    if (offset < 0 || length < 0 || offset + length > size) {
        error = "external tensor data for '" + t.name + "' is out of range";
        return false;
    }
    payload.resize(static_cast<size_t>(length));
    file.seekg(offset, std::ios::beg);
    file.read(payload.data(), length);
    return true;
}

Status tensor_to_bytes(const TensorProto& t, const std::string& base_dir,
                       DataType& dtype, std::vector<std::byte>& bytes,
                       std::string& error)
{
    if (!map_elem_type(t.data_type, dtype)) {
        error = std::string("tensor '") + t.name + "' has unsupported data type " +
                onnx_type_name(t.data_type);
        return Status::unsupported;
    }
    const int64_t count = tensor_numel(t);
    const size_t elem = data_type_size(dtype);
    const size_t total = static_cast<size_t>(count) * elem;

    std::string raw = t.raw_data;
    if (t.external && !read_external_blob(t, base_dir, raw, error)) {
        return Status::invalid_argument;
    }

    bytes.assign(total, std::byte{0});
    if (!raw.empty()) {
        if (raw.size() != total) {
            error = "tensor '" + t.name + "' holds " + std::to_string(raw.size()) +
                    " bytes, its shape needs " + std::to_string(total);
            return Status::invalid_argument;
        }
        std::memcpy(bytes.data(), raw.data(), total);
        return Status::ok;
    }

    auto store = [&](size_t i, const void* value) {
        std::memcpy(bytes.data() + i * elem, value, elem);
    };
    const size_t n = static_cast<size_t>(count);

    switch (dtype) {
    case DataType::f32:
        if (t.float_data.size() != n) { break; }
        for (size_t i = 0; i < n; ++i) { store(i, &t.float_data[i]); }
        return Status::ok;
    case DataType::f16:
    case DataType::bf16:
        if (t.int32_data.size() != n) { break; }
        for (size_t i = 0; i < n; ++i) {
            const uint16_t bits = static_cast<uint16_t>(t.int32_data[i] & 0xFFFF);
            store(i, &bits);
        }
        return Status::ok;
    case DataType::s8:
        if (t.int32_data.size() != n) { break; }
        for (size_t i = 0; i < n; ++i) {
            const int8_t v = static_cast<int8_t>(t.int32_data[i]);
            store(i, &v);
        }
        return Status::ok;
    case DataType::u8:
        if (t.int32_data.size() != n) { break; }
        for (size_t i = 0; i < n; ++i) {
            const uint8_t v = static_cast<uint8_t>(t.int32_data[i]);
            store(i, &v);
        }
        return Status::ok;
    case DataType::s32:
        if (t.int32_data.size() != n) { break; }
        for (size_t i = 0; i < n; ++i) { store(i, &t.int32_data[i]); }
        return Status::ok;
    case DataType::s64:
        if (t.int64_data.size() != n) { break; }
        for (size_t i = 0; i < n; ++i) { store(i, &t.int64_data[i]); }
        return Status::ok;
    default:
        break;
    }

    error = "tensor '" + t.name + "' carries no payload for its declared type";
    return Status::invalid_argument;
}

// ---- Operator classification -----------------------------------------------

bool activation_for(const std::string& op, ActivationType& out)
{
    if (op == "Relu")        { out = ActivationType::Relu; return true; }
    if (op == "LeakyRelu")   { out = ActivationType::LeakyRelu; return true; }
    if (op == "Sigmoid")     { out = ActivationType::Sigmoid; return true; }
    if (op == "Tanh")        { out = ActivationType::Tanh; return true; }
    if (op == "Gelu")        { out = ActivationType::Gelu; return true; }
    if (op == "Silu" || op == "Swish") { out = ActivationType::Silu; return true; }
    if (op == "HardSwish")   { out = ActivationType::HardSwish; return true; }
    if (op == "Elu")         { out = ActivationType::Elu; return true; }
    if (op == "PRelu")       { out = ActivationType::PRelu; return true; }
    if (op == "HardSigmoid") { out = ActivationType::HardSigmoid; return true; }
    if (op == "Softplus")    { out = ActivationType::Softplus; return true; }
    if (op == "Selu")        { out = ActivationType::Selu; return true; }
    if (op == "Mish")        { out = ActivationType::Mish; return true; }
    if (op == "QuickGelu")   { out = ActivationType::QuickGelu; return true; }
    return false;
}

bool unary_for(const std::string& op, UnaryType& out)
{
    if (op == "Exp")   { out = UnaryType::Exp; return true; }
    if (op == "Log")   { out = UnaryType::Log; return true; }
    if (op == "Sin")   { out = UnaryType::Sin; return true; }
    if (op == "Cos")   { out = UnaryType::Cos; return true; }
    if (op == "Tan")   { out = UnaryType::Tan; return true; }
    if (op == "Abs")   { out = UnaryType::Abs; return true; }
    if (op == "Neg")   { out = UnaryType::Neg; return true; }
    if (op == "Sqrt")  { out = UnaryType::Sqrt; return true; }
    if (op == "Erf")   { out = UnaryType::Erf; return true; }
    if (op == "Round") { out = UnaryType::Round; return true; }
    if (op == "Ceil")  { out = UnaryType::Ceil; return true; }
    if (op == "Floor") { out = UnaryType::Floor; return true; }
    if (op == "Reciprocal") { out = UnaryType::Recip; return true; }
    if (op == "Sign")  { out = UnaryType::Sign; return true; }
    if (op == "Exp2")  { out = UnaryType::Exp2; return true; }
    if (op == "Log2")  { out = UnaryType::Log2; return true; }
    if (op == "Log10") { out = UnaryType::Log10; return true; }
    if (op == "Asin")  { out = UnaryType::Asin; return true; }
    if (op == "Acos")  { out = UnaryType::Acos; return true; }
    if (op == "Atan")  { out = UnaryType::Atan; return true; }
    if (op == "Sinh")  { out = UnaryType::Sinh; return true; }
    if (op == "Cosh")  { out = UnaryType::Cosh; return true; }
    if (op == "Asinh") { out = UnaryType::Asinh; return true; }
    if (op == "Acosh") { out = UnaryType::Acosh; return true; }
    if (op == "Atanh") { out = UnaryType::Atanh; return true; }
    if (op == "Erfc")  { out = UnaryType::Erfc; return true; }
    return false;
}

bool eltwise_for(const std::string& op, EltwiseType& out)
{
    if (op == "Add") { out = EltwiseType::Add; return true; }
    if (op == "Sub") { out = EltwiseType::Sub; return true; }
    if (op == "Mul") { out = EltwiseType::Mul; return true; }
    if (op == "Div") { out = EltwiseType::Div; return true; }
    if (op == "Min") { out = EltwiseType::Min; return true; }
    if (op == "Max") { out = EltwiseType::Max; return true; }
    if (op == "Pow") { out = EltwiseType::Pow; return true; }
    return false;
}

bool reduce_for(const std::string& op, ReduceType& out)
{
    if (op == "ReduceSum")  { out = ReduceType::Sum; return true; }
    if (op == "ReduceMean") { out = ReduceType::Mean; return true; }
    if (op == "ReduceMax")  { out = ReduceType::Max; return true; }
    if (op == "ReduceMin")  { out = ReduceType::Min; return true; }
    return false;
}

/// ONNX's published default for the operators whose alpha/beta is not the
/// nnops neutral (0 / 1).
float default_alpha(ActivationType type) noexcept
{
    switch (type) {
    case ActivationType::LeakyRelu: return 0.01f;
    case ActivationType::Elu:       return 1.0f;
    case ActivationType::HardSigmoid: return 0.2f;
    case ActivationType::Selu:      return 1.6732632f;
    case ActivationType::QuickGelu: return 1.702f;
    default: return 0.0f;
    }
}

float default_beta(ActivationType type) noexcept
{
    switch (type) {
    case ActivationType::HardSigmoid: return 0.5f;
    case ActivationType::Selu:        return 1.0507010f;
    default: return 1.0f;
    }
}

int64_t auto_pad_code(const std::string& text) noexcept
{
    if (text == "SAME_UPPER") { return 1; }
    if (text == "SAME_LOWER") { return 2; }
    if (text == "VALID")      { return 3; }
    return 0;  // NOTSET
}

int64_t pick(const std::vector<int64_t>& values, size_t index, int64_t fallback) noexcept
{
    return index < values.size() ? values[index] : fallback;
}

// ---- Attribute helpers -----------------------------------------------------

const AttributeProto* find_attr(const NodeProto& node, const char* name)
{
    for (const auto& a : node.attributes) {
        if (a.name == name) { return &a; }
    }
    return nullptr;
}

int64_t attr_int(const NodeProto& node, const char* name, int64_t fallback)
{
    const AttributeProto* a = find_attr(node, name);
    return a != nullptr ? a->i : fallback;
}

float attr_float(const NodeProto& node, const char* name, float fallback)
{
    const AttributeProto* a = find_attr(node, name);
    return a != nullptr ? a->f : fallback;
}

std::string attr_string(const NodeProto& node, const char* name, const char* fallback = "")
{
    const AttributeProto* a = find_attr(node, name);
    return a != nullptr ? a->s : std::string(fallback);
}

std::vector<int64_t> attr_ints(const NodeProto& node, const char* name)
{
    const AttributeProto* a = find_attr(node, name);
    return a != nullptr ? a->ints : std::vector<int64_t>{};
}

// ---- Importer --------------------------------------------------------------

class Importer {
public:
    Importer(Graph& graph, std::string base_dir, int64_t opset)
        : graph_(graph), base_dir_(std::move(base_dir)), opset_(opset) {}

    Status run(const GraphProto& proto);
    const std::string& error() const noexcept { return error_; }

private:
    int ensure_value(const std::string& name);
    int ensure_input(const std::string& name)
    {
        return name.empty() ? -1 : ensure_value(name);
    }

    void apply_type(const ValueInfoProto& info);
    Status add_initializer(const TensorProto& t);
    Status add_node(const NodeProto& proto, size_t index);
    Status add_shape_node(const NodeProto& proto, Node& node);
    Status add_constant_node(const NodeProto& proto);

    bool is_constant(int id) const;
    bool constant_ints(int id, std::vector<int64_t>& out) const;
    bool constant_floats(int id, std::vector<float>& out) const;
    const TensorProto* initializer_input(const NodeProto& proto, size_t index) const;

    /// Emit the value at `proto.outputs[0]` as a constant of `dtype`.
    int emit_constant(const std::string& name, DataType dtype,
                      std::vector<int64_t> dims, std::vector<std::byte> bytes);

    Status fail_node(const NodeProto& proto, size_t index, const std::string& what)
    {
        const std::string label = !proto.name.empty() ? proto.name
            : (proto.outputs.empty() ? std::string() : proto.outputs[0]);
        error_ = "node " + std::to_string(index) + " (" + proto.op_type +
                 (label.empty() ? "" : " '" + label + "'") + "): " + what;
        return Status::unsupported;
    }

    Graph& graph_;
    std::string base_dir_;
    int64_t opset_ = 0;
    std::string error_;
    std::unordered_map<std::string, int> ids_;
    std::unordered_map<std::string, const TensorProto*> initializer_by_name_;
};

int Importer::ensure_value(const std::string& name)
{
    auto it = ids_.find(name);
    if (it != ids_.end()) { return it->second; }
    ValueInfo info;
    info.name = name;
    const int id = graph_.add_value(std::move(info));
    ids_.emplace(name, id);
    return id;
}

void Importer::apply_type(const ValueInfoProto& info)
{
    if (info.name.empty()) { return; }
    if (info.elem_type == 0 && info.dims.empty()) { return; }
    if (info.dims.size() > static_cast<size_t>(TensorDesc::kMaxRank)) { return; }

    const int id = ensure_value(info.name);
    ValueInfo& value = graph_.values()[static_cast<size_t>(id)];
    DataType dtype = DataType::f32;
    if (map_elem_type(info.elem_type, dtype)) { value.desc.dtype = dtype; }
    if (!info.dims.empty()) {
        value.desc.rank = static_cast<int64_t>(info.dims.size());
        for (size_t i = 0; i < info.dims.size(); ++i) {
            value.desc.dims[i] = info.dims[i];
        }
    }
}

bool Importer::is_constant(int id) const
{
    return id >= 0 && graph_.values()[static_cast<size_t>(id)].constant;
}

bool Importer::constant_ints(int id, std::vector<int64_t>& out) const
{
    if (!is_constant(id)) { return false; }
    const TensorDesc& desc = graph_.values()[static_cast<size_t>(id)].desc;
    const auto& bytes = graph_.constant_data(id);
    const int64_t count = desc.numel();
    out.resize(static_cast<size_t>(count));
    if (desc.dtype == DataType::s64) {
        std::memcpy(out.data(), bytes.data(), bytes.size());
        return true;
    }
    if (desc.dtype == DataType::s32) {
        std::vector<int32_t> tmp(static_cast<size_t>(count));
        std::memcpy(tmp.data(), bytes.data(), bytes.size());
        for (int64_t i = 0; i < count; ++i) {
            out[static_cast<size_t>(i)] = tmp[static_cast<size_t>(i)];
        }
        return true;
    }
    return false;
}

bool Importer::constant_floats(int id, std::vector<float>& out) const
{
    if (!is_constant(id)) { return false; }
    const TensorDesc& desc = graph_.values()[static_cast<size_t>(id)].desc;
    if (desc.dtype != DataType::f32) { return false; }
    const auto& bytes = graph_.constant_data(id);
    out.resize(static_cast<size_t>(desc.numel()));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return true;
}

const TensorProto* Importer::initializer_input(const NodeProto& proto, size_t index) const
{
    if (index >= proto.inputs.size()) { return nullptr; }
    auto it = initializer_by_name_.find(proto.inputs[index]);
    return it == initializer_by_name_.end() ? nullptr : it->second;
}

int Importer::emit_constant(const std::string& name, DataType dtype,
                            std::vector<int64_t> dims, std::vector<std::byte> bytes)
{
    const int id = ensure_value(name);
    ValueInfo& value = graph_.values()[static_cast<size_t>(id)];
    value.desc.dtype = dtype;
    value.desc.rank = static_cast<int64_t>(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) { value.desc.dims[i] = dims[i]; }
    value.constant = true;
    graph_.constant_data(id) = std::move(bytes);
    return id;
}

Status Importer::add_initializer(const TensorProto& t)
{
    if (t.name.empty()) {
        error_ = "an initializer has no name";
        return Status::invalid_argument;
    }
    DataType dtype = DataType::f32;
    std::vector<std::byte> bytes;
    const Status st = tensor_to_bytes(t, base_dir_, dtype, bytes, error_);
    if (is_error(st)) { return st; }
    emit_constant(t.name, dtype, t.dims, std::move(bytes));
    return Status::ok;
}

// Reshape / Flatten / Squeeze / Unsqueeze / Identity / Dropout: no kernel —
// the output is the input's bytes under another descriptor.
Status Importer::add_shape_node(const NodeProto& proto, Node& node)
{
    node.kind = NodeKind::Shape;
    node.alias_of = ensure_input(proto.inputs[0]);
    if (node.alias_of < 0) {
        return fail_node(proto, 0, "a shape node needs its input");
    }
    node.inputs.push_back(node.alias_of);

    const std::string& op = proto.op_type;
    if (op == "Flatten") {
        node.attrs.set_int("axis", attr_int(proto, "axis", 1));
    } else if (op == "Reshape") {
        node.attrs.set_int("allowzero", attr_int(proto, "allowzero", 0));
        std::vector<int64_t> shape;
        if (proto.inputs.size() > 1) {
            const int shape_id = ensure_input(proto.inputs[1]);
            node.inputs.push_back(shape_id);
            if (constant_ints(shape_id, shape)) {
                node.attrs.set_ints("shape", std::move(shape));
            }
        }
        if (!node.attrs.has("shape")) {
            return fail_node(proto, 0, "Reshape needs a constant target shape");
        }
    } else if (op == "Squeeze" || op == "Unsqueeze") {
        std::vector<int64_t> axes = attr_ints(proto, "axes");
        if (axes.empty() && proto.inputs.size() > 1) {
            constant_ints(ensure_input(proto.inputs[1]), axes);
        }
        node.attrs.set_ints("axes", std::move(axes));
    } else if (op != "Identity" && op != "Dropout") {
        return fail_node(proto, 0, "unhandled shape operator");
    }
    return Status::ok;
}

Status Importer::add_constant_node(const NodeProto& proto)
{
    if (proto.outputs.empty()) {
        error_ = "a Constant node has no output";
        return Status::invalid_argument;
    }
    const std::string& name = proto.outputs[0];

    if (const AttributeProto* a = find_attr(proto, "value")) {
        if (!a->has_tensor) {
            error_ = "Constant '" + name + "' has no tensor payload";
            return Status::unsupported;
        }
        DataType dtype = DataType::f32;
        std::vector<std::byte> bytes;
        const Status st = tensor_to_bytes(a->t, base_dir_, dtype, bytes, error_);
        if (is_error(st)) { return st; }
        emit_constant(name, dtype, a->t.dims, std::move(bytes));
        return Status::ok;
    }
    if (const AttributeProto* a = find_attr(proto, "value_float")) {
        std::vector<std::byte> bytes(sizeof(float));
        std::memcpy(bytes.data(), &a->f, sizeof(float));
        emit_constant(name, DataType::f32, {}, std::move(bytes));
        return Status::ok;
    }
    if (const AttributeProto* a = find_attr(proto, "value_int")) {
        std::vector<std::byte> bytes(sizeof(int64_t));
        std::memcpy(bytes.data(), &a->i, sizeof(int64_t));
        emit_constant(name, DataType::s64, {}, std::move(bytes));
        return Status::ok;
    }
    if (const AttributeProto* a = find_attr(proto, "value_ints")) {
        std::vector<std::byte> bytes(a->ints.size() * sizeof(int64_t));
        std::memcpy(bytes.data(), a->ints.data(), bytes.size());
        emit_constant(name, DataType::s64,
                      {static_cast<int64_t>(a->ints.size())}, std::move(bytes));
        return Status::ok;
    }
    if (const AttributeProto* a = find_attr(proto, "value_floats")) {
        std::vector<std::byte> bytes(a->floats.size() * sizeof(float));
        std::memcpy(bytes.data(), a->floats.data(), bytes.size());
        emit_constant(name, DataType::f32,
                      {static_cast<int64_t>(a->floats.size())}, std::move(bytes));
        return Status::ok;
    }
    error_ = "Constant '" + name + "' has no supported value attribute";
    return Status::unsupported;
}

Status Importer::add_node(const NodeProto& proto, size_t index)
{
    for (const auto& a : proto.attributes) {
        if (a.has_subgraph) {
            return fail_node(proto, index, "control-flow subgraphs are not supported");
        }
    }
    const std::string& op = proto.op_type;
    if (!proto.domain.empty() && proto.domain != "ai.onnx" && proto.domain != "ai.onnx.ml") {
        return fail_node(proto, index, "unsupported operator domain '" + proto.domain + "'");
    }
    if (op == "Constant") { return add_constant_node(proto); }

    Node node;
    node.name = proto.name;
    node.op_type = op;
    for (const auto& name : proto.outputs) {
        node.outputs.push_back(name.empty() ? -1 : ensure_value(name));
    }
    if (node.outputs.empty()) {
        return fail_node(proto, index, "a node with no output");
    }
    auto bind_inputs = [&]() {
        for (const auto& name : proto.inputs) { node.inputs.push_back(ensure_input(name)); }
    };

    ActivationType act_type;
    if (activation_for(op, act_type)) {
        if (op == "Gelu" && attr_string(proto, "approximate", "none") != "tanh") {
            act_type = ActivationType::GeluExact;
        }
        node.op = OpType::Activation;
        bind_inputs();
        node.attrs.set_int("type", static_cast<int64_t>(act_type));
        node.attrs.set_float("alpha", attr_float(proto, "alpha", default_alpha(act_type)));
        node.attrs.set_float("beta", attr_float(proto, "beta", default_beta(act_type)));
        graph_.nodes().push_back(std::move(node));
        return Status::ok;
    }

    UnaryType unary_type;
    if (unary_for(op, unary_type)) {
        node.op = OpType::Unary;
        bind_inputs();
        node.attrs.set_int("type", static_cast<int64_t>(unary_type));
        graph_.nodes().push_back(std::move(node));
        return Status::ok;
    }

    EltwiseType eltwise_type;
    if (eltwise_for(op, eltwise_type)) {
        if (proto.inputs.size() != 2) {
            return fail_node(proto, index, "a binary element-wise operator needs two inputs");
        }
        node.op = OpType::Eltwise;
        bind_inputs();
        node.attrs.set_int("type", static_cast<int64_t>(eltwise_type));
        graph_.nodes().push_back(std::move(node));
        return Status::ok;
    }

    if (op == "Conv" || op == "ConvTranspose") {
        std::vector<int64_t> kernel = attr_ints(proto, "kernel_shape");
        if (kernel.empty()) {
            const TensorProto* w = initializer_input(proto, 1);
            if (w == nullptr || w->dims.size() != 4) {
                return fail_node(proto, index, "missing kernel_shape and a 4-D initializer weight");
            }
            kernel = {w->dims[2], w->dims[3]};
        }
        const std::vector<int64_t> strides = attr_ints(proto, "strides");
        const std::vector<int64_t> dilations = attr_ints(proto, "dilations");
        const std::vector<int64_t> pads = attr_ints(proto, "pads");
        if (!pads.empty() && (pads.size() < 4 || pads[0] != pads[2] || pads[1] != pads[3])) {
            return fail_node(proto, index,
                             "asymmetric padding is not supported (the kernels take a symmetric pad)");
        }
        node.op = (op == "Conv") ? OpType::Conv2D : OpType::TransposeConv2D;
        // One filter per input channel is the depthwise operator, which has its
        // own kernels — the grouped-conv route runs a GEMM per group, and a
        // group of one channel is the worst geometry it can be handed. The
        // filter has to be an initializer to know the split without a shape.
        if (op == "Conv" && attr_int(proto, "group", 1) > 1 &&
            auto_pad_code(attr_string(proto, "auto_pad")) == 0) {
            const int64_t group = attr_int(proto, "group", 1);
            const TensorProto* w = initializer_input(proto, 1);
            if (w != nullptr && w->dims.size() == 4 && w->dims[0] == group &&
                w->dims[1] == 1) {
                node.op = OpType::DepthwiseConv;
            }
        }
        bind_inputs();
        node.attrs.set_ints("kernel", {kernel[0], kernel[1]});
        node.attrs.set_ints("stride", {pick(strides, 0, 1), pick(strides, 1, 1)});
        node.attrs.set_ints("dilation", {pick(dilations, 0, 1), pick(dilations, 1, 1)});
        node.attrs.set_ints("padding", {pick(pads, 0, 0), pick(pads, 1, 0)});
        node.attrs.set_int("groups", attr_int(proto, "group", 1));
        node.attrs.set_int("auto_pad", auto_pad_code(attr_string(proto, "auto_pad")));
        if (op == "ConvTranspose") {
            const std::vector<int64_t> output_padding = attr_ints(proto, "output_padding");
            node.attrs.set_ints("output_padding",
                                {pick(output_padding, 0, 0), pick(output_padding, 1, 0)});
        }
        graph_.nodes().push_back(std::move(node));
        return Status::ok;
    }

    if (op == "MatMul") {
        node.op = OpType::MatMul;
        bind_inputs();
        node.attrs.set_int("transpose_a", 0);
        node.attrs.set_int("transpose_b", 0);
        graph_.nodes().push_back(std::move(node));
        return Status::ok;
    }

    if (op == "Gemm") {
        const int64_t trans_a = attr_int(proto, "transA", 0);
        const float alpha = attr_float(proto, "alpha", 1.0f);
        const float beta = attr_float(proto, "beta", 1.0f);
        const bool trans_b = attr_int(proto, "transB", 0) != 0;
        bind_inputs();
        if (trans_a != 0) {
            return fail_node(proto, index, "Gemm with transA is not supported");
        }
        if (alpha != 1.0f) {
            return fail_node(proto, index, "Gemm with alpha != 1 is not supported");
        }

        // ONNX's C is scaled by beta: 0 discards it, 1 adds it, anything else
        // is a scaled accumulate no operator here expresses.
        const bool has_c = node.inputs.size() > 2 && node.inputs[2] >= 0;
        if (has_c && beta != 0.0f && beta != 1.0f) {
            return fail_node(proto, index, "Gemm with a scaled C (beta neither 0 nor 1)");
        }
        const bool use_c = has_c && beta == 1.0f;

        // Linear is the fused X·Wᵀ + bias kernel, so it takes the Gemm already
        // in that shape: transposed weight and a bias that is a vector.
        bool linear_form = trans_b;
        if (linear_form && use_c) {
            const TensorDesc& bias = graph_.values()[static_cast<size_t>(node.inputs[2])].desc;
            linear_form = bias.rank <= 1 || (bias.rank == 2 && bias.dims[0] == 1);
        }
        if (linear_form) {
            node.op = OpType::Linear;
            if (!use_c) { node.inputs.resize(2); }
            graph_.nodes().push_back(std::move(node));
            return Status::ok;
        }

        node.op = OpType::MatMul;
        node.attrs.set_int("transpose_a", 0);
        node.attrs.set_int("transpose_b", trans_b ? 1 : 0);
        if (!use_c) {
            node.inputs.resize(2);
            graph_.nodes().push_back(std::move(node));
            return Status::ok;
        }

        // MatMul into a private product, then a broadcast add into the output.
        Node matmul = std::move(node);
        const int product = graph_.add_value(ValueInfo{});
        matmul.outputs = {product};
        const int bias_value = matmul.inputs[2];
        matmul.inputs.resize(2);

        Node add;
        add.name = matmul.name.empty() ? std::string("gemm_bias") : matmul.name + "_bias";
        add.op_type = "Add";
        add.op = OpType::Eltwise;
        add.attrs.set_int("type", static_cast<int64_t>(EltwiseType::Add));
        add.inputs = {product, bias_value};
        add.outputs = {ensure_value(proto.outputs[0])};
        graph_.nodes().push_back(std::move(matmul));
        graph_.nodes().push_back(std::move(add));
        return Status::ok;
    }

    if (op == "MaxPool" || op == "AveragePool" || op == "GlobalAveragePool" ||
        op == "GlobalMaxPool") {
        const bool global = op.rfind("Global", 0) == 0;
        const bool is_max = (op == "MaxPool" || op == "GlobalMaxPool");
        const std::vector<int64_t> kernel = attr_ints(proto, "kernel_shape");
        const std::vector<int64_t> strides = attr_ints(proto, "strides");
        const std::vector<int64_t> pads = attr_ints(proto, "pads");
        const std::vector<int64_t> dilations = attr_ints(proto, "dilations");
        if (!global && kernel.size() < 2) {
            return fail_node(proto, index, "pooling without kernel_shape");
        }
        if (attr_int(proto, "ceil_mode", 0) != 0) {
            return fail_node(proto, index, "ceil_mode is not supported");
        }
        if (!pads.empty() && (pads.size() < 4 || pads[0] != pads[2] || pads[1] != pads[3])) {
            return fail_node(proto, index, "asymmetric padding is not supported");
        }
        node.op = OpType::Pooling;
        bind_inputs();
        node.attrs.set_int("type", is_max ? 0 : 1);  // PoolingType::Max / Average
        node.attrs.set_int("global", global ? 1 : 0);
        node.attrs.set_ints("kernel_shape",
                            {1, global ? 1 : kernel[0], global ? 1 : kernel[1]});
        node.attrs.set_ints("stride", {1, pick(strides, 0, 1), pick(strides, 1, 1)});
        node.attrs.set_ints("padding", {0, pick(pads, 0, 0), pick(pads, 1, 0)});
        node.attrs.set_ints("dilation", {1, pick(dilations, 0, 1), pick(dilations, 1, 1)});
        // count_include_pad=0 excludes the padded taps from the divisor, which
        // is nnops's exclude_pad.
        node.attrs.set_int("exclude_pad", attr_int(proto, "count_include_pad", 0) == 0 ? 1 : 0);
        node.attrs.set_int("auto_pad", auto_pad_code(attr_string(proto, "auto_pad")));
        graph_.nodes().push_back(std::move(node));
        return Status::ok;
    }

    if (op == "BatchNormalization" || op == "LayerNormalization") {
        const bool batch = op == "BatchNormalization";
        node.op = OpType::Norm;
        bind_inputs();
        node.attrs.set_int("type", batch ? 0 : 1);  // NormType::BatchNorm / LayerNorm
        node.attrs.set_float("epsilon", attr_float(proto, "epsilon", 1e-5f));
        if (!batch) { node.attrs.set_int("axis", attr_int(proto, "axis", -1)); }
        graph_.nodes().push_back(std::move(node));
        return Status::ok;
    }

    if (op == "Softmax" || op == "LogSoftmax") {
        node.op = OpType::Softmax;
        bind_inputs();
        // Opset 13 moved the default axis to the last dimension; older models
        // say 1, which is the same axis for the 2-D tensors they were written for.
        node.attrs.set_int("axis", attr_int(proto, "axis", opset_ >= 13 ? -1 : 1));
        node.attrs.set_int("log_softmax", op == "LogSoftmax" ? 1 : 0);
        graph_.nodes().push_back(std::move(node));
        return Status::ok;
    }

    if (op == "Concat") {
        node.op = OpType::Concat;
        bind_inputs();
        node.attrs.set_int("axis", attr_int(proto, "axis", -1));
        graph_.nodes().push_back(std::move(node));
        return Status::ok;
    }

    if (op == "Transpose") {
        node.op = OpType::Permute;
        bind_inputs();
        node.attrs.set_ints("perm", attr_ints(proto, "perm"));
        graph_.nodes().push_back(std::move(node));
        return Status::ok;
    }

    if (op == "Reshape" || op == "Flatten" || op == "Squeeze" || op == "Unsqueeze" ||
        op == "Identity" || op == "Dropout") {
        const Status st = add_shape_node(proto, node);
        if (is_error(st)) { return st; }
        graph_.nodes().push_back(std::move(node));
        return Status::ok;
    }

    if (op == "Cast") {
        DataType target = DataType::f32;
        if (!map_elem_type(static_cast<int32_t>(attr_int(proto, "to", 0)), target)) {
            return fail_node(proto, index, "Cast to an unsupported data type");
        }
        node.kind = NodeKind::Cast;
        node.target_dtype = target;
        bind_inputs();
        graph_.nodes().push_back(std::move(node));
        return Status::ok;
    }

    if (op == "Shape") {
        const int src = ensure_input(proto.inputs[0]);
        const TensorDesc& desc = graph_.values()[static_cast<size_t>(src)].desc;
        bool statically_known = desc.rank > 0;
        for (int64_t i = 0; i < desc.rank; ++i) {
            if (desc.dims[static_cast<size_t>(i)] < 0) { statically_known = false; }
        }
        if (!statically_known) {
            return fail_node(proto, index, "Shape of a value with a dynamic extent");
        }
        int64_t start = attr_int(proto, "start", 0);
        int64_t end = attr_int(proto, "end", desc.rank);
        if (start < 0) { start += desc.rank; }
        if (end < 0) { end += desc.rank; }
        start = std::clamp<int64_t>(start, 0, desc.rank);
        end = std::clamp<int64_t>(end, start, desc.rank);

        std::vector<int64_t> shape;
        for (int64_t i = start; i < end; ++i) {
            shape.push_back(desc.dims[static_cast<size_t>(i)]);
        }
        std::vector<std::byte> bytes(shape.size() * sizeof(int64_t));
        std::memcpy(bytes.data(), shape.data(), bytes.size());
        emit_constant(proto.outputs[0], DataType::s64,
                      {static_cast<int64_t>(shape.size())}, std::move(bytes));
        return Status::ok;
    }

    if (op == "ConstantOfShape") {
        std::vector<int64_t> shape;
        if (proto.inputs.empty() || !constant_ints(ensure_input(proto.inputs[0]), shape)) {
            return fail_node(proto, index, "ConstantOfShape needs a constant shape");
        }
        DataType dtype = DataType::f32;
        float fill = 0.0f;
        if (const AttributeProto* a = find_attr(proto, "value");
            a != nullptr && a->has_tensor) {
            std::vector<std::byte> unit;
            std::string ignored;
            if (is_error(tensor_to_bytes(a->t, base_dir_, dtype, unit, ignored)) ||
                unit.size() < data_type_size(dtype)) {
                return fail_node(proto, index, "ConstantOfShape fill value is unsupported");
            }
            if (dtype == DataType::f32) { std::memcpy(&fill, unit.data(), sizeof(float)); }
        }
        int64_t count = 1;
        for (int64_t d : shape) { count *= d; }
        std::vector<std::byte> bytes(static_cast<size_t>(count) * data_type_size(dtype));
        if (dtype == DataType::f32) {
            auto* values = reinterpret_cast<float*>(bytes.data());
            for (int64_t i = 0; i < count; ++i) { values[i] = fill; }
        }
        emit_constant(proto.outputs[0], dtype, shape, std::move(bytes));
        return Status::ok;
    }

    if (op == "Clip") {
        node.op = OpType::Clamp;
        bind_inputs();
        // Opset 11 moved the bounds from attributes to optional inputs. Clamp
        // carries them inside its attributes, so a bound given as an input has
        // to be a constant to survive the translation — and once it is read,
        // the input goes away: Clamp takes exactly one.
        float lo = attr_float(proto, "min", -INFINITY);
        float hi = attr_float(proto, "max", INFINITY);
        std::vector<float> bound;
        const auto read_bound = [&](size_t index, float& out) {
            if (node.inputs.size() <= index || node.inputs[index] < 0) { return true; }
            if (!constant_floats(node.inputs[index], bound) || bound.empty()) { return false; }
            out = bound[0];
            return true;
        };
        if (!read_bound(1, lo)) {
            return fail_node(proto, index, "Clip with a non-constant min");
        }
        if (!read_bound(2, hi)) {
            return fail_node(proto, index, "Clip with a non-constant max");
        }
        node.inputs.resize(1);
        node.attrs.set_float("min", lo);
        node.attrs.set_float("max", hi);
        graph_.nodes().push_back(std::move(node));
        return Status::ok;
    }

    if (op == "Gather") {
        node.op = OpType::Gather;
        bind_inputs();
        node.attrs.set_int("axis", attr_int(proto, "axis", 0));
        graph_.nodes().push_back(std::move(node));
        return Status::ok;
    }

    if (op == "Slice") {
        node.op = OpType::Slice;
        bind_inputs();
        std::vector<int64_t> starts, ends, axes, steps;
        if (opset_ >= 10) {
            if (node.inputs.size() > 1) { constant_ints(node.inputs[1], starts); }
            if (node.inputs.size() > 2) { constant_ints(node.inputs[2], ends); }
            if (node.inputs.size() > 3 && node.inputs[3] >= 0) {
                constant_ints(node.inputs[3], axes);
            }
            if (node.inputs.size() > 4 && node.inputs[4] >= 0) {
                constant_ints(node.inputs[4], steps);
            }
        } else {
            starts = attr_ints(proto, "starts");
            ends = attr_ints(proto, "ends");
            axes = attr_ints(proto, "axes");
        }
        if (starts.empty() || ends.size() != starts.size()) {
            return fail_node(proto, index, "Slice needs constant starts and ends");
        }
        node.attrs.set_ints("starts", std::move(starts));
        node.attrs.set_ints("ends", std::move(ends));
        node.attrs.set_ints("axes", std::move(axes));
        node.attrs.set_ints("steps", std::move(steps));
        graph_.nodes().push_back(std::move(node));
        return Status::ok;
    }

    if (op == "Pad") {
        node.op = OpType::Pad;
        bind_inputs();
        std::vector<int64_t> pads = attr_ints(proto, "pads");
        float value = attr_float(proto, "value", 0.0f);
        if (opset_ >= 11) {
            if (node.inputs.size() > 1 && node.inputs[1] >= 0) {
                constant_ints(node.inputs[1], pads);
            }
            std::vector<float> fill;
            if (node.inputs.size() > 2 && node.inputs[2] >= 0 &&
                constant_floats(node.inputs[2], fill) && !fill.empty()) {
                value = fill[0];
            }
        }
        if (pads.empty()) { return fail_node(proto, index, "Pad needs a constant pad list"); }
        const std::string mode = attr_string(proto, "mode", "constant");
        int64_t mode_code = 0;
        if (mode == "edge") {
            mode_code = 1;
        } else if (mode == "reflect") {
            mode_code = 2;
        } else if (mode == "wrap") {
            mode_code = 3;
        } else if (mode != "constant") {
            return fail_node(proto, index, "unsupported Pad mode '" + mode + "'");
        }
        node.attrs.set_ints("pads", std::move(pads));
        node.attrs.set_int("mode", mode_code);
        node.attrs.set_float("constant_value", value);
        graph_.nodes().push_back(std::move(node));
        return Status::ok;
    }

    if (op == "Resize" || op == "Upsample") {
        node.op = OpType::Resize;
        bind_inputs();
        const std::string mode = attr_string(proto, "mode", "nearest");
        node.attrs.set_int("mode", mode == "linear" ? 1 : 0);
        const std::string coord = attr_string(
            proto, "coordinate_transformation_mode",
            op == "Upsample" ? "asymmetric" : "half_pixel");
        // 0 = HalfPixel, 1 = AlignCorners, 2 = Asymmetric
        int64_t coord_code = 0;
        if (coord == "align_corners") {
            coord_code = 1;
        } else if (coord == "asymmetric") {
            coord_code = 2;
        }
        node.attrs.set_int("coord_mode", coord_code);

        std::vector<int64_t> sizes;
        std::vector<float> scales;
        if (opset_ >= 11) {
            if (node.inputs.size() > 3 && node.inputs[3] >= 0) {
                constant_ints(node.inputs[3], sizes);
            }
            if (sizes.empty() && node.inputs.size() > 2 && node.inputs[2] >= 0) {
                constant_floats(node.inputs[2], scales);
            }
        } else if (node.inputs.size() > 1 && node.inputs[1] >= 0) {
            constant_floats(node.inputs[1], scales);
        }

        std::vector<int64_t> output_size;
        if (!sizes.empty()) {
            output_size = std::move(sizes);
        } else if (!scales.empty()) {
            const TensorDesc& desc = graph_.values()[static_cast<size_t>(node.inputs[0])].desc;
            if (static_cast<int64_t>(scales.size()) != desc.rank || desc.rank <= 0) {
                return fail_node(proto, index, "Resize scales must cover the input rank");
            }
            for (int64_t i = 0; i < desc.rank; ++i) {
                if (desc.dims[static_cast<size_t>(i)] < 0) {
                    return fail_node(proto, index, "Resize by scale needs a static input shape");
                }
                output_size.push_back(static_cast<int64_t>(std::floor(
                    static_cast<double>(desc.dims[static_cast<size_t>(i)]) *
                    static_cast<double>(scales[static_cast<size_t>(i)]))));
            }
        } else {
            return fail_node(proto, index, "Resize needs constant sizes or scales");
        }

        // nnops takes the spatial slots only; a 2-D input reads H/W.
        const size_t first_spatial = output_size.size() > 2 ? output_size.size() - 2 : 0;
        std::vector<int64_t> spatial(output_size.begin() + static_cast<long>(first_spatial),
                                     output_size.end());
        node.attrs.set_ints("output_size", std::move(spatial));
        graph_.nodes().push_back(std::move(node));
        return Status::ok;
    }

    ReduceType reduce_type;
    if (reduce_for(op, reduce_type)) {
        std::vector<int64_t> axes = attr_ints(proto, "axes");
        if (axes.empty() && opset_ >= 13 && proto.inputs.size() > 1 && !proto.inputs[1].empty()) {
            constant_ints(ensure_input(proto.inputs[1]), axes);
        }
        const int64_t keepdims = attr_int(proto, "keepdims", 1);
        const int src = ensure_input(proto.inputs[0]);
        if (axes.empty()) {
            const TensorDesc& desc = graph_.values()[static_cast<size_t>(src)].desc;
            if (desc.rank <= 0) {
                return fail_node(proto, index, "a reduce with no axes needs a known rank");
            }
            for (int64_t d = 0; d < desc.rank; ++d) { axes.push_back(d); }
        }
        std::sort(axes.begin(), axes.end());
        // nnops reduces one axis at a time. With keepdims the axes survive each
        // step, so ascending order is fine; without it the dimensions below are
        // removed, so the largest axis has to go first.
        std::vector<int64_t> order = axes;
        if (keepdims == 0) { std::reverse(order.begin(), order.end()); }

        int current = src;
        for (size_t step = 0; step < order.size(); ++step) {
            const bool last = step + 1 == order.size();
            Node reduce;
            reduce.name = proto.name;
            reduce.op_type = op;
            reduce.op = OpType::Reduce;
            reduce.attrs.set_int("type", static_cast<int64_t>(reduce_type));
            reduce.attrs.set_int("axis", order[step]);
            reduce.attrs.set_int("keepdims", last ? keepdims : 1);
            reduce.inputs = {current};
            reduce.outputs = {last ? node.outputs[0] : graph_.add_value(ValueInfo{})};
            current = reduce.outputs[0];
            graph_.nodes().push_back(std::move(reduce));
        }
        return Status::ok;
    }

    if (op == "Where") {
        node.op = OpType::Where;
        bind_inputs();
        graph_.nodes().push_back(std::move(node));
        return Status::ok;
    }

    if (op == "Tile") {
        node.op = OpType::Tile;
        bind_inputs();
        std::vector<int64_t> repeats;
        if (node.inputs.size() > 1 && !constant_ints(node.inputs[1], repeats)) {
            return fail_node(proto, index, "Tile needs constant repeats");
        }
        node.attrs.set_ints("repeats", std::move(repeats));
        graph_.nodes().push_back(std::move(node));
        return Status::ok;
    }

    if (op == "Expand") {
        node.op = OpType::Expand;
        bind_inputs();
        std::vector<int64_t> shape;
        if (node.inputs.size() > 1 && !constant_ints(node.inputs[1], shape)) {
            return fail_node(proto, index, "Expand needs a constant target shape");
        }
        node.attrs.set_ints("shape", std::move(shape));
        graph_.nodes().push_back(std::move(node));
        return Status::ok;
    }

    if (op == "Split") {
        node.op = OpType::Split;
        bind_inputs();
        std::vector<int64_t> split = attr_ints(proto, "split");
        if (split.empty() && opset_ >= 13 && node.inputs.size() > 1 && node.inputs[1] >= 0) {
            constant_ints(node.inputs[1], split);
        }
        node.attrs.set_int("axis", attr_int(proto, "axis", 0));
        node.attrs.set_ints("split", std::move(split));
        node.attrs.set_int("num_outputs",
                           attr_int(proto, "num_outputs",
                                    static_cast<int64_t>(node.outputs.size())));
        graph_.nodes().push_back(std::move(node));
        return Status::ok;
    }

    if (op == "TopK") {
        node.op = OpType::TopK;
        bind_inputs();
        int64_t k = attr_int(proto, "k", 0);
        if (opset_ >= 10 && node.inputs.size() > 1 && node.inputs[1] >= 0) {
            std::vector<int64_t> kv;
            if (constant_ints(node.inputs[1], kv) && !kv.empty()) { k = kv[0]; }
        }
        if (k <= 0) { return fail_node(proto, index, "TopK needs a constant k"); }
        node.attrs.set_int("k", k);
        node.attrs.set_int("axis", attr_int(proto, "axis", -1));
        node.attrs.set_int("type", 0);  // TopKType::Max
        graph_.nodes().push_back(std::move(node));
        return Status::ok;
    }

    if (op == "ArgMax" || op == "ArgMin") {
        node.op = op == "ArgMax" ? OpType::ArgMax : OpType::ArgMin;
        bind_inputs();
        node.attrs.set_int("type", op == "ArgMax" ? 0 : 1);
        node.attrs.set_int("axis", attr_int(proto, "axis", 0));
        node.attrs.set_int("keepdims", attr_int(proto, "keepdims", 1));
        graph_.nodes().push_back(std::move(node));
        return Status::ok;
    }

    return fail_node(proto, index, "operator is not supported");
}

Status Importer::run(const GraphProto& proto)
{
    // Types first: a node's translation may need a value's shape (Shape,
    // Resize-by-scale, a Reduce with no axes).
    for (const auto& info : proto.value_infos) { apply_type(info); }
    for (const auto& info : proto.inputs) { apply_type(info); }
    for (const auto& info : proto.outputs) { apply_type(info); }

    for (const auto& t : proto.initializers) {
        initializer_by_name_[t.name] = &t;
        const Status st = add_initializer(t);
        if (is_error(st)) { return st; }
    }

    // A declared input that is also an initializer is a weight, not an input.
    for (const auto& info : proto.inputs) {
        if (info.name.empty() || initializer_by_name_.count(info.name) != 0) { continue; }
        graph_.inputs().push_back(ensure_value(info.name));
    }

    for (size_t i = 0; i < proto.nodes.size(); ++i) {
        const Status st = add_node(proto.nodes[i], i);
        if (is_error(st)) { return st; }
    }

    for (const auto& info : proto.outputs) {
        if (info.name.empty()) { continue; }
        const int id = ensure_value(info.name);
        if (is_constant(id)) {
            error_ = "graph output '" + info.name + "' is a constant";
            return Status::invalid_argument;
        }
        graph_.outputs().push_back(id);
    }
    if (graph_.outputs().empty()) {
        error_ = "the graph declares no output";
        return Status::invalid_argument;
    }
    return Status::ok;
}

}  // namespace

Status import_graph(const GraphProto& proto, const std::string& base_dir,
                    int64_t opset, Graph& out, std::string& error)
{
    Importer importer(out, base_dir, opset);
    const Status st = importer.run(proto);
    if (is_error(st)) {
        error = importer.error();
        return st;
    }
    return Status::ok;
}

}  // namespace nnops::framework::detail
