/// @file mobilenetv2.cpp
/// @brief End-to-end MobileNetV2 forward pass with nnops operators, checked
///        layer by layer against PyTorch.
///
/// The graph is the torchvision definition, rebuilt op by op:
///
///   stem 3x3 s2 (32) -> BN -> ReLU6
///   17 inverted-residual blocks: 1x1 expand -> BN -> ReLU6 ->
///                                3x3 depthwise s? -> BN -> ReLU6 ->
///                                1x1 project -> BN, plus the input when the
///                                block is stride-1 and keeps its channels
///   head 1x1 (1280) -> BN -> ReLU6 -> global avg pool -> fc 1280->1000
///
/// The depthwise stage is DepthwiseConv on the exported weight as it is
/// ([C, 1, 3, 3]): planar input takes the raw filter, no prepack.
///
/// NNOPS_MODEL_LAYOUT=nhwc runs the same graph with every activation
/// channel-last (see net_ops.hpp): the 1x1s and the stem read HWIO filters, the
/// depthwise stages HWCM [KH, KW, C], the global pool its channel-last route,
/// and each checkpoint is reordered back to NCHW before it is compared. Only
/// the BN fold is mandatory there — the Norm operator is planar — which the
/// fused path does anyway.
///
/// Usage:  model_mobilenetv2 [artifact-dir] [--atol X] [--rtol Y]
/// Exit:   0 all checkpoints pass, 1 a checkpoint failed, 77 no artifact
///         (the test is skipped — export it first, see models/README.md).

#include "common/model_check.hpp"
#include "common/model_io.hpp"
#include "common/net_ops.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

using model::Buf;
using model::ModelArtifact;
using model::ModelCheck;
using nnops::TensorView;

Buf norm(const ModelArtifact& m, const std::string& prefix, const TensorView& x)
{
    return model::batch_norm(x, model::batch_norm_params(m, prefix));
}

/// conv -> BN -> relu6, as one operator call when fusion is on: the BN folds
/// into the weight and a bias, and the relu6 becomes the conv's epilogue. With
/// None the same path is conv -> BN only (the projection stages).
Buf conv_bn(const ModelArtifact& m, const TensorView& x, const std::string& conv,
            const std::string& bn, nnops::EpilogueActivateType act,
            int64_t kernel, int64_t stride, int64_t pad)
{
    if (model::fuse_enabled()) {
        return model::conv2d_bn_act(x, m.at(conv + ".weight"),
                                    model::batch_norm_params(m, bn), act,
                                    kernel, stride, pad);
    }
    Buf y = model::conv2d(x, m.at(conv + ".weight"), kernel, stride, pad);
    y = norm(m, bn, y.view);
    switch (act) {
    case nnops::EpilogueActivateType::None:  return y;
    case nnops::EpilogueActivateType::Relu6: return model::relu6(y.view);
    default: std::abort();  // not used by this network
    }
}

/// depthwise -> BN -> relu6, fused the same way. The filter is [C, 1, KH, KW]
/// and the BN is per channel, which is the filter's own axis.
Buf dw_bn_relu6(const ModelArtifact& m, const TensorView& x, const std::string& dw,
                const std::string& bn, int64_t kernel, int64_t stride, int64_t pad)
{
    if (model::fuse_enabled()) {
        return model::depthwise_bn_act(x, m.at(dw),
                                       model::batch_norm_params(m, bn),
                                       nnops::EpilogueActivateType::Relu6,
                                       kernel, stride, pad);
    }
    Buf y = model::depthwise_conv(x, m.at(dw), kernel, stride, pad);
    y = norm(m, bn, y.view);
    return model::relu6(y.view);
}

/// One InvertedResidual. @p t is the expansion ratio (the expand stage is
/// omitted when it is 1) and @p out_c the projection width.
Buf inverted_residual(const ModelArtifact& m, const std::string& prefix,
                      const TensorView& x, int64_t out_c, int64_t t,
                      int64_t stride, bool residual)
{
    const std::string c = prefix + ".conv.";
    // With no expansion the depthwise stage is conv[0] instead of conv[1].
    const std::string dw = (t != 1) ? c + "1.0.weight" : c + "0.0.weight";
    const std::string dw_bn = (t != 1) ? c + "1.1" : c + "0.1";
    const std::string proj_conv = (t != 1) ? c + "2" : c + "1";
    const std::string proj_bn = (t != 1) ? c + "3" : c + "2";

    Buf expand;
    const TensorView* stage_in = &x;
    if (t != 1) {
        expand = conv_bn(m, x, c + "0.0", c + "0.1",
                         nnops::EpilogueActivateType::Relu6, 1, 1, 0);
        stage_in = &expand.view;
    }

    Buf y = dw_bn_relu6(m, *stage_in, dw, dw_bn, 3, stride, 1);
    // The projection has no activation after it, and that is still a fusion:
    // the BN merges into the conv, so one pass instead of two.
    y = conv_bn(m, y.view, proj_conv, proj_bn,
                nnops::EpilogueActivateType::None, 1, 1, 0);

    if (residual) { y = model::add(y.view, x); }
    return y;
}

/// (t, c, n, s) — torchvision's inverted_residual_setting: expansion ratio,
/// output channels, repeat count, stride of the first repeat.
struct BlockSpec {
    int64_t t, c, n, s;
};
constexpr BlockSpec kSetting[] = {
    {1, 16, 1, 1},  {6, 24, 2, 2},  {6, 32, 3, 2}, {6, 64, 4, 2},
    {6, 96, 3, 1},  {6, 160, 3, 2}, {6, 320, 1, 1},
};

/// Each row of the profile table is one checkpoint (a whole inverted-residual
/// block for features.1..17), timed without the comparison itself.
int forward(const ModelArtifact& m, ModelCheck& check, const TensorView& input)
{
    Buf act = model::layer("features.0", [&] {
        return conv_bn(m, input, "features.0.0", "features.0.1",
                       nnops::EpilogueActivateType::Relu6, 3, 2, 1);
    });
    model::compare_checkpoint(check, "features.0", act.view, m.at("ref.features.0"));

    int64_t in_c = 32;
    int64_t index = 1;
    for (const auto& spec : kSetting) {
        for (int64_t r = 0; r < spec.n; ++r) {
            const int64_t stride = (r == 0) ? spec.s : 1;
            const std::string p = "features." + std::to_string(index);
            Buf out = model::layer(p.c_str(), [&] {
                return inverted_residual(m, p, act.view, spec.c, spec.t, stride,
                                         stride == 1 && in_c == spec.c);
            });
            model::compare_checkpoint(check, p, out.view, m.at("ref." + p));
            act = std::move(out);
            in_c = spec.c;
            ++index;
        }
    }

    act = model::layer("features.18", [&] {
        return conv_bn(m, act.view, "features.18.0", "features.18.1",
                       nnops::EpilogueActivateType::Relu6, 1, 1, 0);
    });
    model::compare_checkpoint(check, "features.18", act.view,
                              m.at("ref.features.18"));

    Buf pooled = model::layer("avgpool", [&] {
        return model::global_avg_pool(act.view);
    });
    Buf logits = model::layer("logits", [&] {
        return model::linear(model::flatten_to_2d(pooled.view),
                             m.at("classifier.1.weight"),
                             m.at("classifier.1.bias"));
    });
    model::compare_checkpoint(check, "logits", logits.view, m.at("ref.logits"));
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    std::string dir = NNOPS_MODEL_DATA_DIR "/mobilenetv2";
    double atol = 1e-3;
    double rtol = 1e-3;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strcmp(a, "--atol") == 0 && i + 1 < argc) {
            atol = std::atof(argv[++i]);
        } else if (std::strcmp(a, "--rtol") == 0 && i + 1 < argc) {
            rtol = std::atof(argv[++i]);
        } else if (a[0] != '-') {
            dir = a;
        }
    }

    ModelArtifact artifact;
    std::string error;
    if (!artifact.load(dir, error)) {
        std::fprintf(stderr, "model_mobilenetv2: %s\n", error.c_str());
        return 77;
    }

    const bool channel_last = model::channel_last();
    if (channel_last && !model::fuse_enabled()) {
        // The Norm operator reads planar tensors, so a channel-last graph has
        // to fold the BN into the conv — which is what fusion already does.
        std::fprintf(stderr,
                     "mobilenetv2: NNOPS_MODEL_LAYOUT=nhwc needs the BN fold "
                     "(unset NNOPS_MODEL_FUSE)\n");
        return 2;
    }
    Buf cl_input;
    TensorView input = artifact.at("input");
    if (channel_last) {
        cl_input = model::to_channel_last(input);
        input = cl_input.view;
    }

    std::printf("mobilenetv2: %zu tensors from %s (conv+bn+relu6 %s, %s)\n",
                artifact.count(), dir.c_str(),
                model::fuse_enabled() ? "folded" : "unfused",
                channel_last ? "nhwc" : "nchw");
    // The untimed pass is the model-load step: it prepacks any Winograd weights
    // (and first-touches every buffer) and it is the pass that checks every
    // checkpoint, so the number below is the steady state with no comparison in
    // it — which is what torch and ORT report.
    ModelCheck verify(atol, rtol);
    forward(artifact, verify, input);
    ModelCheck timed(atol, rtol, /*verbose=*/false);
    timed.set_enabled(false);
    model::prof_reset();
    const auto t0 = std::chrono::steady_clock::now();
    forward(artifact, timed, input);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("forward: %.1f ms\n", ms);
    model::prof_report("mobilenetv2", ms);
    return verify.report("mobilenetv2") == 0 ? 0 : 1;
}
