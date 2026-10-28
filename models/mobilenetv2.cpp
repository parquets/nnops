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
    return model::batch_norm(x, m.at(prefix + ".weight"),
                             m.at(prefix + ".bias"),
                             m.at(prefix + ".running_mean"),
                             m.at(prefix + ".running_var"));
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
    const std::string proj = (t != 1) ? c + "2.weight" : c + "1.weight";
    const std::string proj_bn = (t != 1) ? c + "3" : c + "2";

    Buf expand;
    const TensorView* stage_in = &x;
    if (t != 1) {
        expand = model::conv2d(x, m.at(c + "0.0.weight"), 1, 1, 0);
        expand = norm(m, c + "0.1", expand.view);
        expand = model::relu6(expand.view);
        stage_in = &expand.view;
    }

    Buf y = model::depthwise_conv(*stage_in, m.at(dw), 3, stride, 1);
    y = norm(m, dw_bn, y.view);
    y = model::relu6(y.view);
    y = model::conv2d(y.view, m.at(proj), 1, 1, 0);
    y = norm(m, proj_bn, y.view);

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
int forward(const ModelArtifact& m, ModelCheck& check)
{
    Buf act = model::layer("features.0", [&] {
        Buf y = model::conv2d(m.at("input"), m.at("features.0.0.weight"), 3, 2, 1);
        y = norm(m, "features.0.1", y.view);
        return model::relu6(y.view);
    });
    check.compare("features.0", act.view, m.at("ref.features.0"));

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
            check.compare(p, out.view, m.at("ref." + p));
            act = std::move(out);
            in_c = spec.c;
            ++index;
        }
    }

    act = model::layer("features.18", [&] {
        Buf y = model::conv2d(act.view, m.at("features.18.0.weight"), 1, 1, 0);
        y = norm(m, "features.18.1", y.view);
        return model::relu6(y.view);
    });
    check.compare("features.18", act.view, m.at("ref.features.18"));

    Buf pooled = model::layer("avgpool", [&] {
        return model::global_avg_pool(act.view);
    });
    Buf logits = model::layer("logits", [&] {
        return model::linear(model::flatten_to_2d(pooled.view),
                             m.at("classifier.1.weight"),
                             m.at("classifier.1.bias"));
    });
    check.compare("logits", logits.view, m.at("ref.logits"));
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

    std::printf("mobilenetv2: %zu tensors from %s\n", artifact.count(),
                dir.c_str());
    // The untimed pass is the model-load step: it prepacks any Winograd weights
    // (and first-touches every buffer), so the number below is the steady state.
    { ModelCheck warm(atol, rtol, /*verbose=*/false); forward(artifact, warm); }
    ModelCheck check(atol, rtol);
    model::prof_reset();
    const auto t0 = std::chrono::steady_clock::now();
    forward(artifact, check);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("forward: %.1f ms\n", ms);
    model::prof_report("mobilenetv2", ms);
    return check.report("mobilenetv2") == 0 ? 0 : 1;
}
