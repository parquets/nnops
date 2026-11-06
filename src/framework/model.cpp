/// @file model.cpp
/// @brief Load orchestration — parse, import, optimize, in that order.
///
/// The three stages live in their own translation units; this file is the only
/// place that knows they compose. What it adds around them is the file read and
/// the base directory the importer needs to resolve external weight blobs —
/// ONNX stores anything past 2 GB beside the model rather than inside it, and a
/// path in the proto is relative to the model file, not to the process.
///
/// A failed load clears the graph. Half a model is worse than none: a caller
/// that ignores the Status would otherwise run a graph missing the nodes whose
/// import failed.

#include "nnops/framework/model.hpp"

#include "internal.hpp"

#include <fstream>
#include <string>
#include <vector>

namespace nnops::framework {

namespace {

std::string directory_of(const char* path)
{
    const std::string text(path);
    const size_t slash = text.find_last_of("/\\");
    return slash == std::string::npos ? std::string() : text.substr(0, slash);
}

int64_t standard_opset(const detail::ModelProto& proto)
{
    int64_t version = 0;
    for (const auto& [domain, value] : proto.opset_imports) {
        if (domain.empty() || domain == "ai.onnx") { version = value; }
    }
    return version;
}

/// The shared tail of both entry points: decode, import, optimize.
Status build_graph(const void* bytes, size_t size, const std::string& base_dir,
                   const LoadOptions& opts, Graph& graph, int64_t& opset,
                   bool& loaded, std::string& error)
{
    detail::ModelProto proto;
    if (!detail::parse_model(bytes, size, proto, error)) {
        return Status::invalid_argument;
    }

    opset = standard_opset(proto);
    if (opset == 0) {
        error = "the model imports no ai.onnx opset";
        return Status::invalid_argument;
    }

    const Status st = detail::import_graph(proto.graph, base_dir, opset, graph, error);
    if (is_error(st)) { return st; }

    if (opts.optimize) { detail::optimize_graph(graph, opts); }
    loaded = true;
    return Status::ok;
}

}  // namespace

Status Model::load_from_memory(const void* bytes, size_t size,
                               const LoadOptions& opts)
{
    graph_ = Graph{};
    error_.clear();
    opset_ = 0;
    loaded_ = false;

    if (bytes == nullptr || size == 0) {
        error_ = "the model buffer is empty";
        return Status::invalid_argument;
    }

    const Status st = build_graph(bytes, size, std::string(), opts, graph_,
                                  opset_, loaded_, error_);
    if (is_error(st)) { graph_ = Graph{}; }
    return st;
}

Status Model::load_from_file(const char* path, const LoadOptions& opts)
{
    graph_ = Graph{};
    error_.clear();
    opset_ = 0;
    loaded_ = false;

    if (path == nullptr || *path == '\0') {
        error_ = "no model path given";
        return Status::invalid_argument;
    }

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        error_ = std::string("cannot open model '") + path + "'";
        return Status::invalid_argument;
    }
    const std::streamoff length = file.tellg();
    if (length <= 0) {
        error_ = std::string("model '") + path + "' is empty";
        return Status::invalid_argument;
    }
    std::vector<char> bytes(static_cast<size_t>(length));
    file.seekg(0, std::ios::beg);
    if (!file.read(bytes.data(), length)) {
        error_ = std::string("cannot read model '") + path + "'";
        return Status::invalid_argument;
    }

    const Status st = build_graph(bytes.data(), bytes.size(), directory_of(path),
                                  opts, graph_, opset_, loaded_, error_);
    if (is_error(st)) { graph_ = Graph{}; }
    return st;
}

}  // namespace nnops::framework
