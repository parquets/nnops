/// @file onnx_proto.cpp
/// @brief The protobuf wire reader and the ONNX message schemas it fills.
///
/// Every parse function follows the same shape: a `while (reader.next(field,
/// wire))` loop whose cases are the field numbers from onnx.proto, and a
/// default that skips the payload. A message is rejected only when a field this
/// code reads is malformed — a length that overruns the buffer, a varint that
/// runs off the end, a key whose field number is zero. Skipping an unknown
/// field is not an error: that is how a reader stays compatible with a schema
/// that grew after it was written.

#include "onnx_proto.hpp"

#include <cstdlib>
#include <cstring>

namespace nnops::framework::detail {

namespace {

/// Wire types, from the protobuf encoding.
enum : uint32_t {
    kWireVarint = 0,
    kWireFixed64 = 1,
    kWireLength = 2,
    kWireFixed32 = 5,
};

std::string from_bytes(const uint8_t* data, size_t len)
{
    return std::string(reinterpret_cast<const char*>(data), len);
}

float bits_to_float(uint32_t bits)
{
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

double bits_to_double(uint64_t bits)
{
    double d = 0.0;
    std::memcpy(&d, &bits, sizeof(d));
    return d;
}

}  // namespace

// ---- Reader ----------------------------------------------------------------

bool Reader::next(uint32_t& field, uint32_t& wire)
{
    if (p_ >= end_) { return false; }
    uint64_t key = 0;
    if (!varint(key)) { return false; }
    field = static_cast<uint32_t>(key >> 3);
    wire = static_cast<uint32_t>(key & 7u);
    if (field == 0) {
        failed_ = true;
        return false;
    }
    return true;
}

bool Reader::varint(uint64_t& value)
{
    uint64_t result = 0;
    uint32_t shift = 0;
    while (p_ < end_) {
        const uint8_t byte = *p_++;
        result |= static_cast<uint64_t>(byte & 0x7Fu) << shift;
        if ((byte & 0x80u) == 0) {
            value = result;
            return true;
        }
        shift += 7;
        if (shift >= 64) { break; }
    }
    failed_ = true;
    return false;
}

bool Reader::fixed32(uint32_t& value)
{
    if (static_cast<size_t>(end_ - p_) < 4) {
        failed_ = true;
        return false;
    }
    value = static_cast<uint32_t>(p_[0]) | (static_cast<uint32_t>(p_[1]) << 8) |
            (static_cast<uint32_t>(p_[2]) << 16) | (static_cast<uint32_t>(p_[3]) << 24);
    p_ += 4;
    return true;
}

bool Reader::fixed64(uint64_t& value)
{
    if (static_cast<size_t>(end_ - p_) < 8) {
        failed_ = true;
        return false;
    }
    value = 0;
    for (int i = 7; i >= 0; --i) {
        value = (value << 8) | p_[static_cast<size_t>(i)];
    }
    p_ += 8;
    return true;
}

bool Reader::bytes(const uint8_t*& data, size_t& len)
{
    uint64_t n = 0;
    if (!varint(n)) { return false; }
    if (n > static_cast<uint64_t>(end_ - p_)) {
        failed_ = true;
        return false;
    }
    data = p_;
    len = static_cast<size_t>(n);
    p_ += len;
    return true;
}

bool Reader::sub(Reader& out)
{
    const uint8_t* data = nullptr;
    size_t len = 0;
    if (!bytes(data, len)) { return false; }
    out = Reader(data, data + len);
    return true;
}

bool Reader::skip(uint32_t wire)
{
    switch (wire) {
    case kWireVarint: {
        uint64_t ignored = 0;
        return varint(ignored);
    }
    case kWireFixed64: {
        uint64_t ignored = 0;
        return fixed64(ignored);
    }
    case kWireLength: {
        const uint8_t* data = nullptr;
        size_t len = 0;
        return bytes(data, len);
    }
    case kWireFixed32: {
        uint32_t ignored = 0;
        return fixed32(ignored);
    }
    default:
        failed_ = true;
        return false;
    }
}

bool Reader::repeated_int64(uint32_t wire, std::vector<int64_t>& out)
{
    if (wire == kWireLength) {
        Reader packed;
        if (!sub(packed)) { return false; }
        while (!packed.done()) {
            uint64_t v = 0;
            if (!packed.varint(v)) { return false; }
            out.push_back(static_cast<int64_t>(v));
        }
        return true;
    }
    if (wire != kWireVarint) { failed_ = true; return false; }
    uint64_t v = 0;
    if (!varint(v)) { return false; }
    out.push_back(static_cast<int64_t>(v));
    return true;
}

bool Reader::repeated_int32(uint32_t wire, std::vector<int32_t>& out)
{
    if (wire == kWireLength) {
        Reader packed;
        if (!sub(packed)) { return false; }
        while (!packed.done()) {
            uint64_t v = 0;
            if (!packed.varint(v)) { return false; }
            out.push_back(static_cast<int32_t>(v));
        }
        return true;
    }
    if (wire != kWireVarint) { failed_ = true; return false; }
    uint64_t v = 0;
    if (!varint(v)) { return false; }
    out.push_back(static_cast<int32_t>(v));
    return true;
}

bool Reader::repeated_uint64(uint32_t wire, std::vector<uint64_t>& out)
{
    if (wire == kWireLength) {
        Reader packed;
        if (!sub(packed)) { return false; }
        while (!packed.done()) {
            uint64_t v = 0;
            if (!packed.varint(v)) { return false; }
            out.push_back(v);
        }
        return true;
    }
    if (wire != kWireVarint) { failed_ = true; return false; }
    uint64_t v = 0;
    if (!varint(v)) { return false; }
    out.push_back(v);
    return true;
}

bool Reader::repeated_float(uint32_t wire, std::vector<float>& out)
{
    if (wire == kWireLength) {
        Reader packed;
        if (!sub(packed)) { return false; }
        while (!packed.done()) {
            uint32_t bits = 0;
            if (!packed.fixed32(bits)) { return false; }
            out.push_back(bits_to_float(bits));
        }
        return true;
    }
    if (wire != kWireFixed32) { failed_ = true; return false; }
    uint32_t bits = 0;
    if (!fixed32(bits)) { return false; }
    out.push_back(bits_to_float(bits));
    return true;
}

bool Reader::repeated_double(uint32_t wire, std::vector<double>& out)
{
    if (wire == kWireLength) {
        Reader packed;
        if (!sub(packed)) { return false; }
        while (!packed.done()) {
            uint64_t bits = 0;
            if (!packed.fixed64(bits)) { return false; }
            out.push_back(bits_to_double(bits));
        }
        return true;
    }
    if (wire != kWireFixed64) { failed_ = true; return false; }
    uint64_t bits = 0;
    if (!fixed64(bits)) { return false; }
    out.push_back(bits_to_double(bits));
    return true;
}

// ---- Messages --------------------------------------------------------------

namespace {

constexpr int32_t kDataLocationExternal = 1;

bool parse_tensor(Reader& in, TensorProto& out, std::string& error);

bool parse_string_entry(Reader& in, std::string& key, std::string& value)
{
    while (!in.done()) {
        uint32_t field = 0, wire = 0;
        if (!in.next(field, wire)) { return !in.failed(); }
        const uint8_t* data = nullptr;
        size_t len = 0;
        switch (field) {
        case 1:
            if (wire != kWireLength || !in.bytes(data, len)) { return false; }
            key = from_bytes(data, len);
            break;
        case 2:
            if (wire != kWireLength || !in.bytes(data, len)) { return false; }
            value = from_bytes(data, len);
            break;
        default:
            if (!in.skip(wire)) { return false; }
        }
    }
    return !in.failed();
}

bool parse_tensor(Reader& in, TensorProto& out, std::string& error)
{
    while (!in.done()) {
        uint32_t field = 0, wire = 0;
        if (!in.next(field, wire)) { break; }
        const uint8_t* data = nullptr;
        size_t len = 0;
        uint64_t v = 0;
        switch (field) {
        case 1:  // dims
            if (!in.repeated_int64(wire, out.dims)) { break; }
            continue;
        case 2:  // data_type
            if (wire != kWireVarint || !in.varint(v)) { break; }
            out.data_type = static_cast<int32_t>(v);
            continue;
        case 4:  // float_data
            if (!in.repeated_float(wire, out.float_data)) { break; }
            continue;
        case 5:  // int32_data
            if (!in.repeated_int32(wire, out.int32_data)) { break; }
            continue;
        case 7:  // int64_data
            if (!in.repeated_int64(wire, out.int64_data)) { break; }
            continue;
        case 8:  // name
            if (wire != kWireLength || !in.bytes(data, len)) { break; }
            out.name = from_bytes(data, len);
            continue;
        case 9:  // raw_data
            if (wire != kWireLength || !in.bytes(data, len)) { break; }
            out.raw_data.assign(reinterpret_cast<const char*>(data), len);
            continue;
        case 10:  // double_data
            if (!in.repeated_double(wire, out.double_data)) { break; }
            continue;
        case 11:  // uint64_data
            if (!in.repeated_uint64(wire, out.uint64_data)) { break; }
            continue;
        case 13: {  // external_data: key/value entries
            Reader entry;
            if (wire != kWireLength || !in.sub(entry)) { break; }
            std::string key, value;
            if (!parse_string_entry(entry, key, value)) { break; }
            if (key == "location") {
                out.external_location = value;
            } else if (key == "offset") {
                out.external_offset = std::strtoll(value.c_str(), nullptr, 10);
            } else if (key == "length") {
                out.external_length = std::strtoll(value.c_str(), nullptr, 10);
            }
            continue;
        }
        case 14:  // data_location
            if (wire != kWireVarint || !in.varint(v)) { break; }
            out.external = (static_cast<int32_t>(v) == kDataLocationExternal);
            continue;
        default:
            if (!in.skip(wire)) { break; }
            continue;
        }
        error = "malformed TensorProto field " + std::to_string(field);
        return false;
    }
    if (in.failed()) {
        error = "truncated TensorProto";
        return false;
    }
    return true;
}

bool parse_attribute(Reader& in, AttributeProto& out, std::string& error)
{
    while (!in.done()) {
        uint32_t field = 0, wire = 0;
        if (!in.next(field, wire)) { break; }
        const uint8_t* data = nullptr;
        size_t len = 0;
        uint64_t v = 0;
        uint32_t bits = 0;
        switch (field) {
        case 1:  // name
            if (wire != kWireLength || !in.bytes(data, len)) { break; }
            out.name = from_bytes(data, len);
            continue;
        case 2:  // f
            if (wire != kWireFixed32 || !in.fixed32(bits)) { break; }
            out.f = bits_to_float(bits);
            continue;
        case 3:  // i
            if (wire != kWireVarint || !in.varint(v)) { break; }
            out.i = static_cast<int64_t>(v);
            continue;
        case 4:  // s
            if (wire != kWireLength || !in.bytes(data, len)) { break; }
            out.s = from_bytes(data, len);
            continue;
        case 5: {  // t
            Reader nested;
            if (wire != kWireLength || !in.sub(nested)) { break; }
            if (!parse_tensor(nested, out.t, error)) { return false; }
            out.has_tensor = true;
            continue;
        }
        case 6:    // g — a subgraph (If/Loop body)
        case 11: {  // graphs
            if (wire != kWireLength || !in.skip(wire)) { break; }
            out.has_subgraph = true;
            continue;
        }
        case 7:  // floats
            if (!in.repeated_float(wire, out.floats)) { break; }
            continue;
        case 8:  // ints
            if (!in.repeated_int64(wire, out.ints)) { break; }
            continue;
        case 20:  // type
            if (wire != kWireVarint || !in.varint(v)) { break; }
            out.type = static_cast<int32_t>(v);
            continue;
        default:
            if (!in.skip(wire)) { break; }
            continue;
        }
        error = "malformed AttributeProto field " + std::to_string(field);
        return false;
    }
    if (in.failed()) {
        error = "truncated AttributeProto";
        return false;
    }
    return true;
}

bool parse_node(Reader& in, NodeProto& out, std::string& error)
{
    while (!in.done()) {
        uint32_t field = 0, wire = 0;
        if (!in.next(field, wire)) { break; }
        const uint8_t* data = nullptr;
        size_t len = 0;
        switch (field) {
        case 1:  // input
            if (wire != kWireLength || !in.bytes(data, len)) { break; }
            out.inputs.push_back(from_bytes(data, len));
            continue;
        case 2:  // output
            if (wire != kWireLength || !in.bytes(data, len)) { break; }
            out.outputs.push_back(from_bytes(data, len));
            continue;
        case 3:  // name
            if (wire != kWireLength || !in.bytes(data, len)) { break; }
            out.name = from_bytes(data, len);
            continue;
        case 4:  // op_type
            if (wire != kWireLength || !in.bytes(data, len)) { break; }
            out.op_type = from_bytes(data, len);
            continue;
        case 5: {  // attribute
            Reader nested;
            if (wire != kWireLength || !in.sub(nested)) { break; }
            AttributeProto attr;
            if (!parse_attribute(nested, attr, error)) { return false; }
            out.attributes.push_back(std::move(attr));
            continue;
        }
        case 7:  // domain
            if (wire != kWireLength || !in.bytes(data, len)) { break; }
            out.domain = from_bytes(data, len);
            continue;
        default:
            if (!in.skip(wire)) { break; }
            continue;
        }
        error = "malformed NodeProto field " + std::to_string(field);
        return false;
    }
    if (in.failed()) {
        error = "truncated NodeProto";
        return false;
    }
    return true;
}

bool parse_shape(Reader& in, std::vector<int64_t>& dims, std::string& error)
{
    while (!in.done()) {
        uint32_t field = 0, wire = 0;
        if (!in.next(field, wire)) { break; }
        if (field == 1 && wire == kWireLength) {
            Reader dim;
            if (!in.sub(dim)) { break; }
            int64_t extent = -1;  // a dim_param (or an absent value) is dynamic
            while (!dim.done()) {
                uint32_t df = 0, dw = 0;
                if (!dim.next(df, dw)) { break; }
                uint64_t v = 0;
                if (df == 1 && dw == kWireVarint) {
                    if (!dim.varint(v)) { break; }
                    extent = static_cast<int64_t>(v);
                } else if (!dim.skip(dw)) {
                    break;
                }
            }
            if (dim.failed()) {
                error = "truncated TensorShapeProto::Dimension";
                return false;
            }
            dims.push_back(extent);
            continue;
        }
        if (!in.skip(wire)) { break; }
    }
    if (in.failed()) {
        error = "truncated TensorShapeProto";
        return false;
    }
    return true;
}

bool parse_type(Reader& in, ValueInfoProto& out, std::string& error)
{
    while (!in.done()) {
        uint32_t field = 0, wire = 0;
        if (!in.next(field, wire)) { break; }
        if (field == 1 && wire == kWireLength) {  // tensor_type
            Reader tensor;
            if (!in.sub(tensor)) { break; }
            while (!tensor.done()) {
                uint32_t tf = 0, tw = 0;
                if (!tensor.next(tf, tw)) { break; }
                uint64_t v = 0;
                if (tf == 1 && tw == kWireVarint) {
                    if (!tensor.varint(v)) { break; }
                    out.elem_type = static_cast<int32_t>(v);
                    continue;
                }
                if (tf == 2 && tw == kWireLength) {
                    Reader shape;
                    if (!tensor.sub(shape)) { break; }
                    if (!parse_shape(shape, out.dims, error)) { return false; }
                    continue;
                }
                if (!tensor.skip(tw)) { break; }
            }
            if (tensor.failed()) {
                error = "truncated TypeProto.Tensor";
                return false;
            }
            continue;
        }
        // sequence/map/optional/sparse/opaque types are not tensors and are
        // left empty; the importer rejects an input or output that has none.
        if (!in.skip(wire)) { break; }
    }
    if (in.failed()) {
        error = "truncated TypeProto";
        return false;
    }
    return true;
}

bool parse_value_info(Reader& in, ValueInfoProto& out, std::string& error)
{
    while (!in.done()) {
        uint32_t field = 0, wire = 0;
        if (!in.next(field, wire)) { break; }
        const uint8_t* data = nullptr;
        size_t len = 0;
        if (field == 1 && wire == kWireLength) {
            if (!in.bytes(data, len)) { break; }
            out.name = from_bytes(data, len);
            continue;
        }
        if (field == 2 && wire == kWireLength) {
            Reader type;
            if (!in.sub(type)) { break; }
            if (!parse_type(type, out, error)) { return false; }
            continue;
        }
        if (!in.skip(wire)) { break; }
    }
    if (in.failed()) {
        error = "truncated ValueInfoProto";
        return false;
    }
    return true;
}

bool parse_graph(Reader& in, GraphProto& out, std::string& error)
{
    while (!in.done()) {
        uint32_t field = 0, wire = 0;
        if (!in.next(field, wire)) { break; }
        const uint8_t* data = nullptr;
        size_t len = 0;
        switch (field) {
        case 1: {  // node
            Reader nested;
            if (wire != kWireLength || !in.sub(nested)) { break; }
            NodeProto node;
            if (!parse_node(nested, node, error)) { return false; }
            out.nodes.push_back(std::move(node));
            continue;
        }
        case 2:  // name
            if (wire != kWireLength || !in.bytes(data, len)) { break; }
            out.name = from_bytes(data, len);
            continue;
        case 5: {  // initializer
            Reader nested;
            if (wire != kWireLength || !in.sub(nested)) { break; }
            TensorProto tensor;
            if (!parse_tensor(nested, tensor, error)) { return false; }
            out.initializers.push_back(std::move(tensor));
            continue;
        }
        case 11:
        case 12:
        case 13: {  // input / output / value_info
            Reader nested;
            if (wire != kWireLength || !in.sub(nested)) { break; }
            ValueInfoProto info;
            if (!parse_value_info(nested, info, error)) { return false; }
            if (field == 11) {
                out.inputs.push_back(std::move(info));
            } else if (field == 12) {
                out.outputs.push_back(std::move(info));
            } else {
                out.value_infos.push_back(std::move(info));
            }
            continue;
        }
        default:
            if (!in.skip(wire)) { break; }
            continue;
        }
        error = "malformed GraphProto field " + std::to_string(field);
        return false;
    }
    if (in.failed()) {
        error = "truncated GraphProto";
        return false;
    }
    return true;
}

bool parse_opset(Reader& in, std::pair<std::string, int64_t>& out)
{
    while (!in.done()) {
        uint32_t field = 0, wire = 0;
        if (!in.next(field, wire)) { break; }
        const uint8_t* data = nullptr;
        size_t len = 0;
        uint64_t v = 0;
        if (field == 1 && wire == kWireLength) {
            if (!in.bytes(data, len)) { break; }
            out.first = from_bytes(data, len);
            continue;
        }
        if (field == 2 && wire == kWireVarint) {
            if (!in.varint(v)) { break; }
            out.second = static_cast<int64_t>(v);
            continue;
        }
        if (!in.skip(wire)) { break; }
    }
    return !in.failed();
}

}  // namespace

bool parse_model(const void* bytes, size_t size, ModelProto& out, std::string& error)
{
    const auto* begin = static_cast<const uint8_t*>(bytes);
    Reader in(begin, begin + size);

    while (!in.done()) {
        uint32_t field = 0, wire = 0;
        if (!in.next(field, wire)) { break; }
        const uint8_t* data = nullptr;
        size_t len = 0;
        uint64_t v = 0;
        switch (field) {
        case 1:  // ir_version
            if (wire != kWireVarint || !in.varint(v)) { break; }
            out.ir_version = static_cast<int64_t>(v);
            continue;
        case 2:  // producer_name
            if (wire != kWireLength || !in.bytes(data, len)) { break; }
            out.producer_name = from_bytes(data, len);
            continue;
        case 7: {  // graph
            Reader nested;
            if (wire != kWireLength || !in.sub(nested)) { break; }
            if (!parse_graph(nested, out.graph, error)) { return false; }
            continue;
        }
        case 8: {  // opset_import
            Reader nested;
            if (wire != kWireLength || !in.sub(nested)) { break; }
            std::pair<std::string, int64_t> entry{"", 0};
            if (!parse_opset(nested, entry)) { break; }
            out.opset_imports.push_back(std::move(entry));
            continue;
        }
        default:
            if (!in.skip(wire)) { break; }
            continue;
        }
        error = "malformed ModelProto field " + std::to_string(field);
        return false;
    }

    if (in.failed()) {
        error = "truncated ModelProto";
        return false;
    }
    if (out.graph.nodes.empty() && out.graph.inputs.empty()) {
        error = "the model carries no graph";
        return false;
    }
    return true;
}

}  // namespace nnops::framework::detail
