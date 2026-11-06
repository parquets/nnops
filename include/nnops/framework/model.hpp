#pragma once
/// @file model.hpp
/// @brief Model — an ONNX file parsed into a Graph and ready to be executed.
///
/// Loading is the whole front half of the pipeline and runs in three steps:
///
///   1. parse    — the ONNX protobuf is read by a self-contained wire-format
///                 decoder (no protobuf runtime, no code generation, so the
///                 same source builds for MSVC and for the NDK cross-build);
///   2. import   — every ONNX node is mapped onto the nnops vocabulary: an
///                 OpType, an OpBase, or one of the framework's metadata nodes
///                 (Reshape / Flatten / Squeeze / Identity / Cast). Initializers
///                 and Constant nodes become graph constants;
///   3. optimize — constant folding, dead-node elimination, and the
///                 producer+activation fusions the operators' epilogues accept.
///
/// A Model is immutable once loaded; Session binds one to buffers and runs it.
/// Parsing never throws and never reads past the end of the file: a malformed
/// model is a Status, and last_error() names the field or node that failed.

#include "nnops/core/status.hpp"
#include "nnops/framework/graph.hpp"

#include <cstddef>
#include <string>

namespace nnops::framework {

/// Load-time switches.
struct LoadOptions {
    /// Run the graph passes (constant folding, dead-node elimination, epilogue
    /// fusion). Off keeps the graph exactly as imported, which is what the
    /// pass tests compare against.
    bool optimize = true;

    /// Largest tensor (in bytes) constant folding will evaluate. A folded node
    /// costs load time and is only worth it when the result is small; the
    /// default covers the shape/weight arithmetic real models carry.
    size_t fold_limit_bytes = 1u << 22;
};

class Model {
public:
    Model() = default;

    Status load_from_file(const char* path, const LoadOptions& opts = {});
    Status load_from_memory(const void* bytes, size_t size,
                            const LoadOptions& opts = {});

    const Graph& graph() const noexcept { return graph_; }

    /// Message for the last failure; empty when the model loaded.
    const std::string& last_error() const noexcept { return error_; }

    /// ai.onnx opset version the model was exported against; 0 when the file
    /// names no ai.onnx import.
    int64_t opset() const noexcept { return opset_; }

    bool valid() const noexcept { return loaded_; }

private:
    Graph graph_;
    std::string error_;
    int64_t opset_ = 0;
    bool loaded_ = false;
};

}  // namespace nnops::framework
