#pragma once
/// @file onnx_proto.hpp
/// @brief A self-contained ONNX protobuf decoder (internal to src/framework).
///
/// The upstream ONNX schema ships as protobuf-generated C++, which would put a
/// protobuf runtime in front of every consumer of this library and a second
/// build system in front of the NDK cross-build. What ONNX actually writes is a
/// narrow slice of the protobuf wire format — varints, length-delimited fields
/// and fixed 32/64-bit values, with no groups and no extensions — so a reader
/// for it is a few hundred lines and no dependency at all.
///
/// Field numbers are from onnx/onnx.proto (checked against the checkout in
/// D:\git\onnx). Each message here carries only the fields this framework reads;
/// every other field is skipped by wire type, which is what lets an older reader
/// stay compatible with a newer producer. onnx.proto is proto2, so a repeated
/// numeric field arrives unpacked unless the schema marks it [packed = true] —
/// the readers below accept either form for both repeated shapes.
///
/// Nothing here validates semantics: an unknown data type, a truncated message
/// or a bad wire type comes back as false with `error` filled, and it is the
/// importer that turns those into a Status.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nnops::framework::detail {

// ---- Wire-format reader ----------------------------------------------------

/// A cursor over one length-delimited protobuf message.
class Reader {
public:
    Reader() = default;
    Reader(const uint8_t* begin, const uint8_t* end) : p_(begin), end_(end) {}

    /// True when the cursor has run past the message.
    bool done() const noexcept { return p_ >= end_; }

    /// Read the next field key; false at the end of the message or on a
    /// malformed key (field 0 does not exist).
    bool next(uint32_t& field, uint32_t& wire);

    bool varint(uint64_t& value);
    bool fixed32(uint32_t& value);
    bool fixed64(uint64_t& value);

    /// A length-delimited payload (string, bytes, or an embedded message).
    bool bytes(const uint8_t*& data, size_t& len);

    /// Consume a payload this reader has no use for.
    bool skip(uint32_t wire);

    /// Sub-reader over an embedded message; false when the length overruns.
    bool sub(Reader& out);

    /// Repeated numeric payloads, in either their packed (wire 2) or unpacked
    /// (wire 0 / 5) form.
    bool repeated_int64(uint32_t wire, std::vector<int64_t>& out);
    bool repeated_int32(uint32_t wire, std::vector<int32_t>& out);
    bool repeated_float(uint32_t wire, std::vector<float>& out);
    bool repeated_double(uint32_t wire, std::vector<double>& out);
    bool repeated_uint64(uint32_t wire, std::vector<uint64_t>& out);

    bool failed() const noexcept { return failed_; }
    void fail() noexcept { failed_ = true; }

private:
    const uint8_t* p_ = nullptr;
    const uint8_t* end_ = nullptr;
    bool failed_ = false;
};

// ---- Messages --------------------------------------------------------------

/// onnx.TensorProto, as far as an inference-only reader needs it.
struct TensorProto {
    std::vector<int64_t> dims;
    int32_t data_type = 0;  ///< onnx.TensorProto.DataType
    std::string name;

    /// Payload in exactly one of these, per how the producer wrote it.
    std::string raw_data;
    std::vector<float> float_data;
    std::vector<int32_t> int32_data;
    std::vector<int64_t> int64_data;
    std::vector<double> double_data;
    std::vector<uint64_t> uint64_data;

    /// data_location == EXTERNAL: the payload lives beside the model file.
    bool external = false;
    std::string external_location;
    int64_t external_offset = 0;
    int64_t external_length = -1;
};

/// onnx.AttributeProto, flattened: exactly one of the value fields is set,
/// selected by `type`.
struct AttributeProto {
    std::string name;
    int32_t type = 0;
    float f = 0.0f;
    int64_t i = 0;
    std::string s;
    std::vector<float> floats;
    std::vector<int64_t> ints;
    TensorProto t;
    bool has_tensor = false;
    /// A control-flow attribute (If/Loop body): kept as a flag so the importer
    /// can name the operator instead of silently dropping the branch.
    bool has_subgraph = false;
};

struct NodeProto {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::string name;
    std::string op_type;
    std::string domain;
    std::vector<AttributeProto> attributes;
};

/// onnx.ValueInfoProto: the tensor type only. A symbolically-dimensioned tensor
/// keeps -1 in that slot.
struct ValueInfoProto {
    std::string name;
    int32_t elem_type = 0;
    std::vector<int64_t> dims;
};

struct GraphProto {
    std::vector<NodeProto> nodes;
    std::string name;
    std::vector<TensorProto> initializers;
    std::vector<ValueInfoProto> inputs;
    std::vector<ValueInfoProto> outputs;
    std::vector<ValueInfoProto> value_infos;
};

struct ModelProto {
    int64_t ir_version = 0;
    std::string producer_name;
    GraphProto graph;
    /// (domain, version) pairs; "ai.onnx" (or empty) is the standard opset.
    std::vector<std::pair<std::string, int64_t>> opset_imports;
};

// ---- Entry point -----------------------------------------------------------

/// Parse a whole ModelProto. Returns false with `error` describing the field
/// that could not be read.
bool parse_model(const void* bytes, size_t size, ModelProto& out, std::string& error);

}  // namespace nnops::framework::detail
