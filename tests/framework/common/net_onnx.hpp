#pragma once
/// @file net_onnx.hpp
/// @brief torchvision networks written as ONNX ModelProto bytes, built from the
///        exported artifacts under models/data/.
///
/// The artifacts hold what PyTorch produced for these exact networks: the
/// weights under torchvision's module names, and one recorded activation per
/// checkpoint ("ref.conv1", "ref.layer2.1", ...). This header turns the first
/// into an ONNX graph and leaves the second to the runner, so the check is
/// PyTorch's own numbers against whatever the framework computed — not one
/// nnops operator against another.
///
/// The graphs are the torchvision definitions, node for node:
///
///   resnet18       conv1 7x7 s2 -> BN -> relu -> maxpool 3x3 s2
///                  4 stages x 2 BasicBlocks (3x3 s?, 3x3 s1; a 1x1 s2 shortcut
///                  when the stage changes channels or resolution)
///                  -> global average pool -> flatten -> FC
///   mobilenetv2    stem 3x3 s2 -> 17 inverted-residual blocks (1x1 expand ->
///                  3x3 depthwise s? -> 1x1 project, plus the input when the
///                  block keeps both stride and width) -> head 1x1
///                  -> global average pool -> flatten -> FC
///
/// `record` makes every recorded activation a graph output as well. The session
/// then hands the intermediate values back, which is what lets the runner price
/// a failure to the layer that drifted; the cost is that the optimizer keeps its
/// hands off the activation that would otherwise fuse into a conv.

#include "common/onnx_writer.hpp"
#include "models/common/model_io.hpp"

#include <string>
#include <vector>

namespace framework_test {

using nnops::test::OnnxNode;
using nnops::test::OnnxValue;
using nnops::test::attr_float;
using nnops::test::attr_int;
using nnops::test::attr_ints;
using nnops::test::tensor_f32;

/// A network as ModelProto bytes, plus the names its graph outputs carry in
/// order. `output_names[0]` is the network's result; the rest are checkpoints.
struct NetModel {
    std::string bytes;
    std::string input_name = "input";
    std::vector<std::string> output_names;
};

namespace detail {

inline std::vector<int64_t> dims_of(const nnops::TensorView& t)
{
    return std::vector<int64_t>(t.shape(), t.shape() + t.rank());
}

}  // namespace detail

/// Assembles one network: initializers lifted from the artifact, nodes written
/// with ONNX's own attributes (the importer reads those attributes, so this is
/// also the ONNX the framework must accept, not a private dialect).
class NetBuilder {
public:
    NetBuilder(const model::ModelArtifact& art, bool record)
        : art_(art), record_(record)
    {
        model_.opset = 13;
        model_.graph_name = "nnops_framework_model";
    }

    /// The artifact's tensor @p name, verbatim, as an initializer of the same
    /// name — the graph then needs no renaming between PyTorch and ONNX.
    void weight(const std::string& name)
    {
        const nnops::TensorView t = art_.at(name);
        model_.initializers.push_back(
            tensor_f32(name, detail::dims_of(t),
                       std::vector<float>(t.ptr<float>(), t.ptr<float>() + t.numel())));
    }

    /// The four BatchNorm statistics under @p prefix, in BatchNormalization's
    /// input order.
    void batch_norm_weights(const std::string& prefix)
    {
        weight(prefix + ".weight");
        weight(prefix + ".bias");
        weight(prefix + ".running_mean");
        weight(prefix + ".running_var");
    }

    /// A scalar f32 initializer. Opset 11 moved Clip's bounds from attributes
    /// to optional inputs, and an exporter writes them as constants like this.
    void scalar(const std::string& name, float value)
    {
        model_.initializers.push_back(tensor_f32(name, {}, {value}));
    }

    /// The graph input, declared with the artifact's shape. `dynamic_batch`
    /// leaves dim 0 open so one session can take any batch.
    void input(const std::string& name, bool dynamic_batch)
    {
        const nnops::TensorView t = art_.at(name);
        OnnxValue v;
        v.name = name;
        v.dims = detail::dims_of(t);
        if (dynamic_batch && !v.dims.empty()) { v.dims[0] = -1; }
        model_.inputs.push_back(v);
        input_name_ = name;
    }

    // ---- node helpers; each returns its output name ----

    std::string conv(const std::string& x, const std::string& w, const std::string& out,
                     int64_t kernel, int64_t stride, int64_t pad, int64_t groups = 1)
    {
        OnnxNode n;
        n.name = out;
        n.op_type = "Conv";
        n.inputs = {x, w};
        n.outputs = {out};
        n.attrs = {attr_ints("kernel_shape", {kernel, kernel}),
                   attr_ints("strides", {stride, stride}),
                   attr_ints("pads", {pad, pad, pad, pad}),
                   attr_int("group", groups)};
        push(std::move(n));
        return out;
    }

    std::string batch_norm(const std::string& x, const std::string& prefix,
                           const std::string& out)
    {
        OnnxNode n;
        n.name = out;
        n.op_type = "BatchNormalization";
        n.inputs = {x, prefix + ".weight", prefix + ".bias",
                    prefix + ".running_mean", prefix + ".running_var"};
        n.outputs = {out};
        n.attrs = {attr_float("epsilon", 1e-5f)};
        push(std::move(n));
        return out;
    }

    std::string relu(const std::string& x, const std::string& out)
    {
        OnnxNode n;
        n.name = out;
        n.op_type = "Relu";
        n.inputs = {x};
        n.outputs = {out};
        push(std::move(n));
        return out;
    }

    std::string clip(const std::string& x, const std::string& lo, const std::string& hi,
                     const std::string& out)
    {
        OnnxNode n;
        n.name = out;
        n.op_type = "Clip";
        n.inputs = {x, lo, hi};
        n.outputs = {out};
        push(std::move(n));
        return out;
    }

    std::string add(const std::string& a, const std::string& b, const std::string& out)
    {
        OnnxNode n;
        n.name = out;
        n.op_type = "Add";
        n.inputs = {a, b};
        n.outputs = {out};
        push(std::move(n));
        return out;
    }

    std::string max_pool(const std::string& x, const std::string& out,
                         int64_t kernel, int64_t stride, int64_t pad)
    {
        OnnxNode n;
        n.name = out;
        n.op_type = "MaxPool";
        n.inputs = {x};
        n.outputs = {out};
        n.attrs = {attr_ints("kernel_shape", {kernel, kernel}),
                   attr_ints("strides", {stride, stride}),
                   attr_ints("pads", {pad, pad, pad, pad})};
        push(std::move(n));
        return out;
    }

    std::string global_avg_pool(const std::string& x, const std::string& out)
    {
        OnnxNode n;
        n.name = out;
        n.op_type = "GlobalAveragePool";
        n.inputs = {x};
        n.outputs = {out};
        push(std::move(n));
        return out;
    }

    std::string flatten(const std::string& x, const std::string& out)
    {
        OnnxNode n;
        n.name = out;
        n.op_type = "Flatten";
        n.inputs = {x};
        n.outputs = {out};
        n.attrs = {attr_int("axis", 1)};
        push(std::move(n));
        return out;
    }

    /// Y = X·Wᵀ + b, which is what torchvision's Linear exports as.
    std::string gemm(const std::string& x, const std::string& w, const std::string& b,
                     const std::string& out)
    {
        OnnxNode n;
        n.name = out;
        n.op_type = "Gemm";
        n.inputs = {x, w, b};
        n.outputs = {out};
        n.attrs = {attr_int("transB", 1)};
        push(std::move(n));
        return out;
    }

    // ---- bookkeeping ----

    /// Record @p name as a checkpoint the session must hand back.
    void checkpoint(std::string name)
    {
        if (record_) { checkpoints_.push_back(std::move(name)); }
    }

    /// Name the network's result. Called once, before finish().
    void result(std::string name) { result_ = std::move(name); }

    NetModel finish()
    {
        NetModel net;
        net.input_name = input_name_;
        net.output_names.push_back(result_);
        for (const auto& c : checkpoints_) { net.output_names.push_back(c); }

        for (const auto& name : net.output_names) {
            model_.outputs.push_back(value_info(name));
        }
        net.bytes = model_.serialize();
        return net;
    }

private:
    void push(OnnxNode n) { model_.nodes.push_back(std::move(n)); }

    /// A declared output. The shape is only advisory — the session infers it
    /// from the operators — but the artifact knows it, so it is written down.
    OnnxValue value_info(const std::string& name) const
    {
        OnnxValue v;
        v.name = name;
        const std::string ref = "ref." + name;
        if (art_.has(ref)) { v.dims = detail::dims_of(art_.at(ref)); }
        return v;
    }

    const model::ModelArtifact& art_;
    bool record_;
    nnops::test::OnnxModel model_;
    std::string input_name_;
    std::string result_;
    std::vector<std::string> checkpoints_;
};

/// One torchvision BasicBlock. @p downsample selects the 1x1-conv shortcut,
/// which is what PyTorch builds when the block changes the channel count or the
/// resolution. Returns the block's output value.
inline std::string resnet_basic_block(NetBuilder& b, const std::string& p,
                                      const std::string& x, int64_t stride,
                                      bool downsample)
{
    b.conv(x, p + ".conv1.weight", p + ".conv1", 3, stride, 1);
    b.batch_norm(p + ".conv1", p + ".bn1", p + ".bn1.act");
    b.relu(p + ".bn1.act", p + ".relu1");

    b.conv(p + ".relu1", p + ".conv2.weight", p + ".conv2", 3, 1, 1);
    b.batch_norm(p + ".conv2", p + ".bn2", p + ".bn2.act");

    std::string residual = x;
    if (downsample) {
        b.conv(x, p + ".downsample.0.weight", p + ".downsample.conv", 1, stride, 0);
        b.batch_norm(p + ".downsample.conv", p + ".downsample.1", p + ".downsample");
        residual = p + ".downsample";
    }

    b.add(p + ".bn2.act", residual, p + ".sum");
    return b.relu(p + ".sum", p);
}

inline NetModel build_resnet18(const model::ModelArtifact& art, bool record,
                               bool dynamic_batch = false)
{
    NetBuilder b(art, record);
    b.input("input", dynamic_batch);

    b.weight("conv1.weight");
    b.weight("bn1.weight");
    b.weight("bn1.bias");
    b.weight("bn1.running_mean");
    b.weight("bn1.running_var");
    b.weight("fc.weight");
    b.weight("fc.bias");

    b.conv("input", "conv1.weight", "conv1", 7, 2, 3);
    b.checkpoint("conv1");

    b.batch_norm("conv1", "bn1", "bn1");
    b.checkpoint("bn1");

    b.relu("bn1", "relu");
    b.checkpoint("relu");

    b.max_pool("relu", "maxpool", 3, 2, 1);
    b.checkpoint("maxpool");

    // The first block of a stage strides; the second keeps the resolution.
    const int64_t stage_stride[4] = {1, 2, 2, 2};
    std::string act = "maxpool";
    for (int64_t stage = 1; stage <= 4; ++stage) {
        for (int64_t blk = 0; blk < 2; ++blk) {
            const std::string p =
                "layer" + std::to_string(stage) + "." + std::to_string(blk);
            const int64_t stride = (blk == 0) ? stage_stride[stage - 1] : 1;
            for (int64_t k : {1, 2}) {
                b.weight(p + ".conv" + std::to_string(k) + ".weight");
                b.weight(p + ".bn" + std::to_string(k) + ".weight");
                b.weight(p + ".bn" + std::to_string(k) + ".bias");
                b.weight(p + ".bn" + std::to_string(k) + ".running_mean");
                b.weight(p + ".bn" + std::to_string(k) + ".running_var");
            }
            if (stride != 1) {
                b.weight(p + ".downsample.0.weight");
                b.weight(p + ".downsample.1.weight");
                b.weight(p + ".downsample.1.bias");
                b.weight(p + ".downsample.1.running_mean");
                b.weight(p + ".downsample.1.running_var");
            }
            act = resnet_basic_block(b, p, act, stride, stride != 1);
            b.checkpoint(act);
        }
    }

    b.global_avg_pool(act, "avgpool");
    b.checkpoint("avgpool");
    b.flatten("avgpool", "flat");
    b.gemm("flat", "fc.weight", "fc.bias", "logits");
    b.result("logits");
    return b.finish();
}

/// One MobileNetV2 InvertedResidual. @p t is the expansion ratio (no expand
/// stage when it is 1), @p in_c the input width the depthwise filter groups by,
/// and @p residual says whether the block adds its input back.
inline std::string mobilenet_block(NetBuilder& b, const std::string& p,
                                   const std::string& x, int64_t in_c, int64_t t,
                                   int64_t stride, bool residual,
                                   const std::string& lo, const std::string& hi)
{
    const std::string c = p + ".conv.";
    std::string stage_in = x;

    if (t != 1) {
        b.weight(c + "0.0.weight");
        b.batch_norm_weights(c + "0.1");
        b.conv(x, c + "0.0.weight", p + ".expand", 1, 1, 0);
        b.batch_norm(p + ".expand", c + "0.1", p + ".expand.bn");
        b.clip(p + ".expand.bn", lo, hi, p + ".expand.act");
        stage_in = p + ".expand.act";
    }

    // The depthwise stage is a nested conv+bn pair; the projection is a flat
    // one, which is how the exporter laid the module tree out.
    const std::string dw = (t != 1) ? c + "1.0.weight" : c + "0.0.weight";
    const std::string dw_bn = (t != 1) ? c + "1.1" : c + "0.1";
    const std::string proj_w = (t != 1) ? c + "2.weight" : c + "1.weight";
    const std::string proj_bn = (t != 1) ? c + "3" : c + "2";

    b.weight(dw);
    b.batch_norm_weights(dw_bn);
    b.weight(proj_w);
    b.batch_norm_weights(proj_bn);

    // A depthwise stage is a grouped conv with one channel per group.
    b.conv(stage_in, dw, p + ".dw", 3, stride, 1, in_c * t);
    b.batch_norm(p + ".dw", dw_bn, p + ".dw.bn");
    b.clip(p + ".dw.bn", lo, hi, p + ".dw.act");

    b.conv(p + ".dw.act", proj_w, p + ".proj", 1, 1, 0);
    if (residual) {
        b.batch_norm(p + ".proj", proj_bn, p + ".proj.bn");
        return b.add(p + ".proj.bn", x, p);
    }
    return b.batch_norm(p + ".proj", proj_bn, p);
}

inline NetModel build_mobilenetv2(const model::ModelArtifact& art, bool record,
                                  bool dynamic_batch = false)
{
    NetBuilder b(art, record);
    b.input("input", dynamic_batch);
    b.scalar("clip_min", 0.0f);
    b.scalar("clip_max", 6.0f);
    const std::string lo = "clip_min";
    const std::string hi = "clip_max";

    // Stem, and the head further down: a single ConvBNReLU each, so the weight
    // is features.N.0 and the BN's four are features.N.1.*.
    b.weight("features.0.0.weight");
    b.batch_norm_weights("features.0.1");
    b.conv("input", "features.0.0.weight", "features.0.conv", 3, 2, 1);
    b.batch_norm("features.0.conv", "features.0.1", "features.0.bn");
    b.clip("features.0.bn", lo, hi, "features.0");
    b.checkpoint("features.0");

    // (t, c, n, s) — torchvision's inverted_residual_setting.
    const int64_t setting[][4] = {
        {1, 16, 1, 1}, {6, 24, 2, 2}, {6, 32, 3, 2}, {6, 64, 4, 2},
        {6, 96, 3, 1}, {6, 160, 3, 2}, {6, 320, 1, 1},
    };

    std::string act = "features.0";
    int64_t in_c = 32;
    int64_t index = 1;
    for (const auto& spec : setting) {
        const int64_t t = spec[0], c = spec[1], repeats = spec[2], s = spec[3];
        for (int64_t r = 0; r < repeats; ++r) {
            const int64_t stride = (r == 0) ? s : 1;
            const std::string p = "features." + std::to_string(index);
            act = mobilenet_block(b, p, act, in_c, t, stride,
                                  stride == 1 && in_c == c, lo, hi);
            b.checkpoint(act);
            in_c = c;
            ++index;
        }
    }

    // Head.
    b.weight("features.18.0.weight");
    b.batch_norm_weights("features.18.1");
    b.weight("classifier.1.weight");
    b.weight("classifier.1.bias");
    b.conv(act, "features.18.0.weight", "features.18.conv", 1, 1, 0);
    b.batch_norm("features.18.conv", "features.18.1", "features.18.bn");
    b.clip("features.18.bn", lo, hi, "features.18");
    b.checkpoint("features.18");

    b.global_avg_pool("features.18", "avgpool");
    b.flatten("avgpool", "flat");
    b.gemm("flat", "classifier.1.weight", "classifier.1.bias", "logits");
    b.result("logits");
    return b.finish();
}

}  // namespace framework_test
