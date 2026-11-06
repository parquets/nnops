#pragma once
/// @file onnx_writer.hpp
/// @brief A minimal ONNX ModelProto writer, for tests that need a real file.
///
/// No python, no onnx package, no protobuf runtime on this host, so the tests
/// emit the wire format themselves. This is the mirror of src/framework/
/// onnx_proto.cpp: the same field numbers, written instead of read. It covers
/// exactly what the framework's tests need — a graph of nodes, initializers and
/// typed inputs/outputs — and nothing else; a field it does not know is a field
/// the framework does not read either.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace nnops::test {

inline void put_varint(std::string& out, uint64_t value)
{
    while (value >= 0x80) {
        out.push_back(static_cast<char>((value & 0x7F) | 0x80));
        value >>= 7;
    }
    out.push_back(static_cast<char>(value));
}

inline void put_tag(std::string& out, uint32_t field, uint32_t wire)
{
    put_varint(out, (static_cast<uint64_t>(field) << 3) | wire);
}

inline void put_int64(std::string& out, uint32_t field, int64_t value)
{
    put_tag(out, field, 0);
    put_varint(out, static_cast<uint64_t>(value));
}

inline void put_fixed32(std::string& out, uint32_t field, uint32_t bits)
{
    put_tag(out, field, 5);
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<char>((bits >> (8 * i)) & 0xFF));
    }
}

inline void put_bytes(std::string& out, uint32_t field,
                      const void* data, size_t size)
{
    put_tag(out, field, 2);
    put_varint(out, size);
    out.append(static_cast<const char*>(data), size);
}

inline void put_string(std::string& out, uint32_t field, const std::string& text)
{
    put_bytes(out, field, text.data(), text.size());
}

inline void put_message(std::string& out, uint32_t field, const std::string& body)
{
    put_bytes(out, field, body.data(), body.size());
}

inline uint32_t float_bits(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// ---- Messages --------------------------------------------------------------

/// onnx.TensorProto, built from either float or int64 data.
struct OnnxTensor {
    std::string name;
    std::vector<int64_t> dims;
    int32_t data_type = 1;  // FLOAT
    std::vector<float> floats;
    std::vector<int64_t> ints;

    std::string encode() const
    {
        std::string body;
        for (int64_t d : dims) { put_int64(body, 1, d); }
        put_int64(body, 2, data_type);
        for (float f : floats) { put_fixed32(body, 4, float_bits(f)); }
        for (int64_t i : ints) { put_int64(body, 7, i); }
        put_string(body, 8, name);
        return body;
    }
};

inline OnnxTensor tensor_f32(std::string name, std::vector<int64_t> dims,
                            std::vector<float> values)
{
    OnnxTensor t;
    t.name = std::move(name);
    t.dims = std::move(dims);
    t.data_type = 1;
    t.floats = std::move(values);
    return t;
}

inline OnnxTensor tensor_i64(std::string name, std::vector<int64_t> dims,
                            std::vector<int64_t> values)
{
    OnnxTensor t;
    t.name = std::move(name);
    t.dims = std::move(dims);
    t.data_type = 7;  // INT64
    t.ints = std::move(values);
    return t;
}

/// onnx.ValueInfoProto with a tensor type; a dim of -1 becomes a dim_param.
struct OnnxValue {
    std::string name;
    int32_t elem_type = 1;
    std::vector<int64_t> dims;

    std::string encode() const
    {
        std::string shape;
        for (int64_t d : dims) {
            std::string dim;
            if (d < 0) {
                put_string(dim, 2, "N");
            } else {
                put_int64(dim, 1, d);
            }
            put_message(shape, 1, dim);
        }
        std::string tensor;
        put_int64(tensor, 1, elem_type);
        put_message(tensor, 2, shape);

        std::string type;
        put_message(type, 1, tensor);

        std::string body;
        put_string(body, 1, name);
        put_message(body, 2, type);
        return body;
    }
};

/// onnx.AttributeProto; `kind` picks both the value field and the type enum.
struct OnnxAttr {
    enum class Kind { Int, Float, String, Ints, Floats };

    std::string name;
    Kind kind = Kind::Int;
    int64_t i = 0;
    float f = 0.0f;
    std::string s;
    std::vector<int64_t> ints;
    std::vector<float> floats;

    std::string encode() const
    {
        std::string body;
        put_string(body, 1, name);
        int64_t type = 0;
        switch (kind) {
        case Kind::Int:    put_int64(body, 3, i); type = 2; break;
        case Kind::Float:  put_fixed32(body, 2, float_bits(f)); type = 1; break;
        case Kind::String: put_string(body, 4, s); type = 3; break;
        case Kind::Ints:
            for (int64_t v : ints) { put_int64(body, 8, v); }
            type = 7;
            break;
        case Kind::Floats:
            for (float v : floats) { put_fixed32(body, 7, float_bits(v)); }
            type = 6;
            break;
        }
        put_int64(body, 20, type);
        return body;
    }
};

inline OnnxAttr attr_int(std::string name, int64_t value)
{
    OnnxAttr a;
    a.name = std::move(name);
    a.kind = OnnxAttr::Kind::Int;
    a.i = value;
    return a;
}

inline OnnxAttr attr_float(std::string name, float value)
{
    OnnxAttr a;
    a.name = std::move(name);
    a.kind = OnnxAttr::Kind::Float;
    a.f = value;
    return a;
}

inline OnnxAttr attr_string(std::string name, std::string value)
{
    OnnxAttr a;
    a.name = std::move(name);
    a.kind = OnnxAttr::Kind::String;
    a.s = std::move(value);
    return a;
}

inline OnnxAttr attr_ints(std::string name, std::vector<int64_t> values)
{
    OnnxAttr a;
    a.name = std::move(name);
    a.kind = OnnxAttr::Kind::Ints;
    a.ints = std::move(values);
    return a;
}

struct OnnxNode {
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<OnnxAttr> attrs;

    std::string encode() const
    {
        std::string body;
        for (const auto& v : inputs) { put_string(body, 1, v); }
        for (const auto& v : outputs) { put_string(body, 2, v); }
        put_string(body, 3, name);
        put_string(body, 4, op_type);
        for (const auto& a : attrs) { put_message(body, 5, a.encode()); }
        return body;
    }
};

struct OnnxModel {
    int64_t ir_version = 7;
    int64_t opset = 13;
    std::string graph_name = "g";
    std::vector<OnnxTensor> initializers;
    std::vector<OnnxValue> inputs;
    std::vector<OnnxValue> outputs;
    std::vector<OnnxNode> nodes;

    std::string serialize() const
    {
        std::string graph;
        for (const auto& n : nodes) { put_message(graph, 1, n.encode()); }
        put_string(graph, 2, graph_name);
        for (const auto& t : initializers) { put_message(graph, 5, t.encode()); }
        for (const auto& v : inputs) { put_message(graph, 11, v.encode()); }
        for (const auto& v : outputs) { put_message(graph, 12, v.encode()); }

        std::string opset_body;
        put_string(opset_body, 1, "");
        put_int64(opset_body, 2, opset);

        std::string body;
        put_int64(body, 1, ir_version);
        put_message(body, 8, opset_body);
        put_message(body, 7, graph);
        return body;
    }
};

}  // namespace nnops::test
