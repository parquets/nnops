/// @file optimizer.cpp
/// @brief The graph passes a load runs: constant folding, BatchNorm folding,
///        activation fusion, dead-node elimination.
///
/// Folding is the pass that pays for itself on the models this framework was
/// built for: a Reshape whose target shape is the output of a Shape/Concat
/// chain becomes a constant at load time instead of an allocation and a kernel
/// per call, and a ConstantOfShape becomes its fill. It is bounded by
/// `fold_limit_bytes` because a folded tensor costs load time and, more to the
/// point, graph memory that a per-call buffer would have recycled.
///
/// BatchNorm folding is what a torchvision export needs. Every conv in those
/// networks is followed by a BatchNormalization, which is a second kernel over
/// the whole activation to apply a per-channel affine map. In inference the
/// statistics are frozen, so the map is a constant and it commutes into the
/// convolution's weights and bias — one kernel instead of two, and the
/// activation behind the BN then has a conv to fuse into.
///
/// Fusion targets one pattern — a producer whose output feeds exactly one
/// Activation — and only where the producer's operator carries an Epilogue
/// that can express the activation. The consumer must not be a graph output
/// (the activated value would then replace an un-activated one a caller
/// expects) and the producer's output must not be one either, for the same
/// reason in the other direction.
///
/// Dead-node elimination runs last and for free: every pass above may strand a
/// node, and none of them needs to know it.

#include "internal.hpp"

#include "nnops/core/backend.hpp"
#include "nnops/ops/activation.hpp"
#include "nnops/ops/norm.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace nnops::framework::detail {

namespace {

bool carries_epilogue(OpType op) noexcept
{
    switch (op) {
    case OpType::Conv2D:
    case OpType::DepthwiseConv:
    case OpType::TransposeConv2D:
    case OpType::MatMul:
    case OpType::Linear:
        return true;
    default:
        return false;
    }
}

/// The fusion's activation, or None when the epilogue cannot express it.
EpilogueActivateType epilogue_for(ActivationType type) noexcept
{
    switch (type) {
    case ActivationType::Relu:        return EpilogueActivateType::Relu;
    case ActivationType::LeakyRelu:   return EpilogueActivateType::LeakyRelu;
    case ActivationType::Sigmoid:     return EpilogueActivateType::Sigmoid;
    case ActivationType::Tanh:        return EpilogueActivateType::Tanh;
    case ActivationType::Gelu:        return EpilogueActivateType::Gelu;
    case ActivationType::Silu:        return EpilogueActivateType::Silu;
    case ActivationType::HardSwish:   return EpilogueActivateType::HardSwish;
    case ActivationType::Elu:         return EpilogueActivateType::Elu;
    default:                          return EpilogueActivateType::None;
    }
}

bool is_float_dtype(DataType dtype) noexcept
{
    return dtype == DataType::f32 || dtype == DataType::f16;
}

bool is_graph_output(const Graph& graph, int value)
{
    const auto& outputs = graph.outputs();
    return std::find(outputs.begin(), outputs.end(), value) != outputs.end();
}

// ---- Constant folding ------------------------------------------------------

void fold_constants(Graph& graph, size_t limit)
{
    const std::vector<int> order = graph.topological_order();
    for (int node_id : order) {
        const Node& node = graph.nodes()[static_cast<size_t>(node_id)];
        if (node.inputs.empty()) { continue; }

        bool all_constant = true;
        for (int v : node.inputs) {
            if (v < 0 || !graph.values()[static_cast<size_t>(v)].constant) {
                all_constant = false;
                break;
            }
        }
        if (!all_constant) { continue; }

        std::vector<TensorDesc> in_descs;
        in_descs.reserve(node.inputs.size());
        for (int v : node.inputs) {
            if (v >= 0) { in_descs.push_back(graph.values()[static_cast<size_t>(v)].desc); }
        }

        std::unique_ptr<OpBase> impl;
        std::string error;
        if (is_error(lower_node(node, Backend::CPU, impl, error))) { continue; }

        std::vector<TensorDesc> out_descs;
        if (is_error(infer_node(node, impl.get(), in_descs, out_descs, error))) { continue; }
        if (out_descs.empty()) { continue; }

        size_t bytes = 0;
        for (const TensorDesc& desc : out_descs) { bytes += desc.nbytes(); }
        if (bytes > limit) { continue; }

        std::vector<Tensor> buffers(out_descs.size());
        std::vector<TensorView> out_views;
        std::vector<TensorView> in_views;
        bool allocated = true;
        for (size_t i = 0; i < out_descs.size(); ++i) {
            if (is_error(buffers[i].reset(out_descs[i]))) {
                allocated = false;
                break;
            }
            out_views.push_back(buffers[i].view());
        }
        if (!allocated) { continue; }
        for (int v : node.inputs) {
            if (v < 0) { continue; }
            const ValueInfo& info = graph.values()[static_cast<size_t>(v)];
            in_views.emplace_back(std::span<const int64_t>(info.desc.dims.data(),
                                                           static_cast<size_t>(info.desc.rank)),
                                  info.desc.dtype,
                                  graph.constant_data(v).data());
        }

        const size_t workspace = impl != nullptr
            ? impl->getWorkspaceSize(in_descs, out_descs)
            : 0;
        std::vector<std::byte> scratch(workspace);
        const Status st = run_node(node, impl.get(), out_views, in_views,
                                   ComputeContext{},
                                   workspace > 0 ? scratch.data() : nullptr);
        if (is_error(st)) { continue; }

        for (size_t i = 0; i < node.outputs.size(); ++i) {
            const int v = node.outputs[i];
            if (v < 0) { continue; }
            ValueInfo& info = graph.values()[static_cast<size_t>(v)];
            info.desc = out_descs[i];
            info.constant = true;
            std::vector<std::byte>& bytes_out = graph.constant_data(v);
            bytes_out.resize(buffers[i].nbytes());
            std::memcpy(bytes_out.data(), buffers[i].data(), buffers[i].nbytes());
        }
    }
}

// ---- BatchNorm folding -----------------------------------------------------

/// The f32 contents of a constant value, or false when `id` is not one.
bool constant_floats(const Graph& graph, int id, std::vector<float>& out)
{
    if (id < 0) { return false; }
    const ValueInfo& info = graph.values()[static_cast<size_t>(id)];
    if (!info.constant || info.desc.dtype != DataType::f32) { return false; }
    const std::vector<std::byte>& bytes = graph.constant_data(id);
    if (bytes.size() != info.desc.nbytes()) { return false; }
    out.resize(static_cast<size_t>(info.desc.numel()));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return true;
}

int add_constant(Graph& graph, std::string name, const TensorDesc& desc,
                 const void* data, size_t bytes)
{
    ValueInfo info;
    info.name = std::move(name);
    info.desc = desc;
    info.constant = true;
    const int id = graph.add_value(std::move(info));
    std::vector<std::byte>& dst = graph.constant_data(id);
    dst.resize(bytes);
    std::memcpy(dst.data(), data, bytes);
    return id;
}

std::string unique_name(const Graph& graph, const std::string& base)
{
    std::string name = base;
    for (int suffix = 1; graph.find_value(name) >= 0; ++suffix) {
        name = base + "_" + std::to_string(suffix);
    }
    return name;
}

/// The one node reading `value` other than the producer, or -1 when it has no
/// reader or several.
int single_consumer(const Graph& graph, int producer, int value)
{
    int consumer = -1;
    const auto& nodes = graph.nodes();
    for (size_t j = 0; j < nodes.size(); ++j) {
        if (static_cast<int>(j) == producer) { continue; }
        for (int v : nodes[j].inputs) {
            if (v != value) { continue; }
            if (consumer >= 0 && consumer != static_cast<int>(j)) { return -1; }
            consumer = static_cast<int>(j);
        }
    }
    return consumer;
}

/// Point every reader of `from` at `to`.
void redirect_readers(Graph& graph, int from, int to)
{
    for (auto& node : graph.nodes()) {
        for (int& v : node.inputs) {
            if (v == from) { v = to; }
        }
    }
    for (int& v : graph.outputs()) {
        if (v == from) { v = to; }
    }
}

/// conv -> BatchNormalization becomes one conv, which is the other half of why
/// an exporter's depthwise blocks need this pass too. The normalization is an
/// affine per-channel map with frozen statistics, so it scales the filter (a
/// whole row per output channel, whether that row is [IC/G, KH, KW] or a
/// depthwise [1, KH, KW]) and shifts the bias:
///
///   s_c = gamma_c / sqrt(var_c + eps)
///   w'  = w · s        (per output channel)
///   b'  = (b - mean_c) · s_c + beta_c
///
/// It needs the filter and the four statistics to be constants, the conv to
/// feed nobody but this BN, and neither result to be a graph output — an
/// intermediate someone asked for must keep the values it means.
void fuse_batch_norms(Graph& graph)
{
    auto& nodes = graph.nodes();
    for (size_t i = 0; i < nodes.size(); ++i) {
        Node& conv = nodes[i];
        if (conv.kind != NodeKind::Op ||
            (conv.op != OpType::Conv2D && conv.op != OpType::DepthwiseConv)) {
            continue;
        }
        if (conv.outputs.size() != 1 || conv.outputs[0] < 0) { continue; }
        if (conv.inputs.size() < 2 || conv.inputs[0] < 0 || conv.inputs[1] < 0) { continue; }

        const int produced = conv.outputs[0];
        if (is_graph_output(graph, produced)) { continue; }

        const int consumer = single_consumer(graph, static_cast<int>(i), produced);
        if (consumer < 0) { continue; }
        Node& bn = nodes[static_cast<size_t>(consumer)];
        if (bn.kind != NodeKind::Op || bn.op != OpType::Norm) { continue; }
        if (bn.attrs.get_int("type", -1) != static_cast<int64_t>(NormType::BatchNorm)) {
            continue;
        }
        if (bn.inputs.size() != 5 || bn.outputs.size() != 1 || bn.outputs[0] < 0) { continue; }
        if (is_graph_output(graph, bn.outputs[0])) { continue; }

        const ValueInfo weight_info = graph.values()[static_cast<size_t>(conv.inputs[1])];
        if (!weight_info.constant || weight_info.desc.dtype != DataType::f32 ||
            weight_info.desc.rank != 4) {
            continue;
        }

        std::vector<float> weight, gamma, beta, mean, var, old_bias;
        if (!constant_floats(graph, conv.inputs[1], weight)) { continue; }
        if (!constant_floats(graph, bn.inputs[1], gamma)) { continue; }
        if (!constant_floats(graph, bn.inputs[2], beta)) { continue; }
        if (!constant_floats(graph, bn.inputs[3], mean)) { continue; }
        if (!constant_floats(graph, bn.inputs[4], var)) { continue; }

        const int64_t channels = weight_info.desc.dims[0];
        if (channels <= 0 || weight.size() != static_cast<size_t>(weight_info.desc.numel())) {
            continue;
        }
        if (static_cast<int64_t>(gamma.size()) != channels ||
            static_cast<int64_t>(beta.size()) != channels ||
            static_cast<int64_t>(mean.size()) != channels ||
            static_cast<int64_t>(var.size()) != channels) {
            continue;
        }

        const bool has_bias = conv.inputs.size() > 2 && conv.inputs[2] >= 0;
        if (has_bias && (!constant_floats(graph, conv.inputs[2], old_bias) ||
                         static_cast<int64_t>(old_bias.size()) != channels)) {
            continue;
        }

        const float eps = bn.attrs.get_float("epsilon", 1e-5f);
        const int64_t per_channel = static_cast<int64_t>(weight.size()) / channels;
        std::vector<float> bias(static_cast<size_t>(channels));
        for (int64_t c = 0; c < channels; ++c) {
            const size_t uc = static_cast<size_t>(c);
            const float scale = gamma[uc] / std::sqrt(var[uc] + eps);
            const float shift = (has_bias ? old_bias[uc] : 0.0f) - mean[uc];
            float* row = weight.data() + c * per_channel;
            for (int64_t k = 0; k < per_channel; ++k) { row[k] *= scale; }
            bias[uc] = shift * scale + beta[uc];
        }

        const int folded_value = bn.outputs[0];
        const std::string base = graph.values()[static_cast<size_t>(folded_value)].name;

        TensorDesc bias_desc;
        bias_desc.dtype = DataType::f32;
        bias_desc.layout = weight_info.desc.layout;
        bias_desc.rank = 1;
        bias_desc.dims[0] = channels;

        const int new_weight = add_constant(graph, unique_name(graph, base + "/weight"),
                                            weight_info.desc, weight.data(),
                                            weight.size() * sizeof(float));
        const int new_bias = add_constant(graph, unique_name(graph, base + "/bias"), bias_desc,
                                          bias.data(), bias.size() * sizeof(float));

        conv.inputs.resize(3);
        conv.inputs[1] = new_weight;
        conv.inputs[2] = new_bias;
        redirect_readers(graph, folded_value, produced);
    }
}

// ---- Activation fusion -----------------------------------------------------

void fuse_activations(Graph& graph)
{
    auto& nodes = graph.nodes();
    const size_t n = nodes.size();

    for (size_t i = 0; i < n; ++i) {
        Node& producer = nodes[i];
        if (producer.kind != NodeKind::Op || !carries_epilogue(producer.op)) { continue; }
        if (producer.outputs.size() != 1 || producer.outputs[0] < 0) { continue; }
        if (producer.fused_activation != EpilogueActivateType::None) { continue; }

        const int produced = producer.outputs[0];
        if (is_graph_output(graph, produced)) { continue; }
        if (!is_float_dtype(graph.values()[static_cast<size_t>(produced)].desc.dtype)) {
            continue;
        }

        const int consumer = single_consumer(graph, static_cast<int>(i), produced);
        if (consumer < 0) { continue; }

        Node& act = nodes[static_cast<size_t>(consumer)];
        if (act.kind != NodeKind::Op) { continue; }
        if (act.inputs.size() != 1 || act.outputs.size() != 1 || act.outputs[0] < 0) {
            continue;
        }
        if (is_graph_output(graph, act.outputs[0])) { continue; }

        EpilogueActivateType fused = EpilogueActivateType::None;
        float alpha = 0.0f;
        float beta = 1.0f;
        if (act.op == OpType::Activation) {
            const auto type = static_cast<ActivationType>(act.attrs.get_int("type"));
            fused = epilogue_for(type);
            alpha = act.attrs.get_float("alpha", 0.0f);
            beta = act.attrs.get_float("beta", 1.0f);
        } else if (act.op == OpType::Clamp) {
            // A Clamp to [0, 6] is exactly the epilogue's Relu6, and it is what
            // an exporter writes for torchvision's ReLU6.
            if (act.attrs.get_float("min", 0.0f) == 0.0f &&
                act.attrs.get_float("max", 0.0f) == 6.0f) {
                fused = EpilogueActivateType::Relu6;
            }
        }
        if (fused == EpilogueActivateType::None) { continue; }

        producer.fused_activation = fused;
        producer.fused_alpha = alpha;
        producer.fused_beta = beta;

        // Every reader of the activation's result now reads the producer's.
        // The activation's result then has no reader, and remove_dead_nodes()
        // takes the node with it.
        redirect_readers(graph, act.outputs[0], produced);
    }
}

}  // namespace

void optimize_graph(Graph& graph, const LoadOptions& opts)
{
    fold_constants(graph, opts.fold_limit_bytes);

    // The sweep between the two fusions is not decorative: a folded BN is still
    // a node reading the conv's output, so the activation fusion would see two
    // consumers and leave the activation alone. Only once the BN is gone does
    // the conv have the one it can absorb.
    fuse_batch_norms(graph);
    graph.remove_dead_nodes();
    fuse_activations(graph);
    graph.remove_dead_nodes();
}

}  // namespace nnops::framework::detail
