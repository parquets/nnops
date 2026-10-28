/// @file resnet18.cpp
/// @brief End-to-end ResNet-18 forward pass with nnops operators, checked
///        layer by layer against PyTorch.
///
/// The graph is the torchvision definition, rebuilt op by op:
///
///   conv1 7x7 s2 p3 -> bn1 -> relu -> maxpool 3x3 s2 p1
///   4 stages x 2 BasicBlocks (3x3 s?, 3x3 s1; a 1x1 s2 shortcut when the
///   stage changes channels / resolution) -> global avg pool -> fc 512->1000
///
/// Every checkpoint the exporter recorded ("ref.conv1", "ref.layer2.1", ...)
/// is compared here, so a failure names the layer that drifted rather than
/// just the logits.
///
/// Usage:  model_resnet18 [artifact-dir] [--atol X] [--rtol Y]
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

/// BatchNorm at @p prefix: scale, shift and the two running statistics.
Buf norm(const ModelArtifact& m, const std::string& prefix, const TensorView& x)
{
    return model::batch_norm(x, m.at(prefix + ".weight"),
                             m.at(prefix + ".bias"),
                             m.at(prefix + ".running_mean"),
                             m.at(prefix + ".running_var"));
}

/// One BasicBlock. @p downsample selects the 1x1-conv shortcut, which is what
/// PyTorch builds when the block changes the channel count or resolution.
Buf basic_block(const ModelArtifact& m, const std::string& prefix,
                const TensorView& x, int64_t stride, bool downsample)
{
    Buf y = model::conv2d(x, m.at(prefix + ".conv1.weight"), 3, stride, 1);
    y = norm(m, prefix + ".bn1", y.view);
    y = model::relu(y.view);
    y = model::conv2d(y.view, m.at(prefix + ".conv2.weight"), 3, 1, 1);
    y = norm(m, prefix + ".bn2", y.view);

    // main = main + shortcut, then the block's own relu.
    if (downsample) {
        Buf skip = model::conv2d(x, m.at(prefix + ".downsample.0.weight"),
                                 1, stride, 0);
        skip = norm(m, prefix + ".downsample.1", skip.view);
        y = model::add(y.view, skip.view);
    } else {
        y = model::add(y.view, x);
    }
    return model::relu(y.view);
}

/// Run the network and compare every recorded checkpoint. Each row of the
/// profile table is one checkpoint, timed without the comparison itself.
int forward(const ModelArtifact& m, ModelCheck& check)
{
    Buf act = model::layer("conv1", [&] {
        return model::conv2d(m.at("input"), m.at("conv1.weight"), 7, 2, 3);
    });
    check.compare("conv1", act.view, m.at("ref.conv1"));

    act = model::layer("bn1", [&] { return norm(m, "bn1", act.view); });
    check.compare("bn1", act.view, m.at("ref.bn1"));

    act = model::layer("relu", [&] { return model::relu(act.view); });
    check.compare("relu", act.view, m.at("ref.relu"));

    act = model::layer("maxpool", [&] { return model::max_pool(act.view, 3, 2, 1); });
    check.compare("maxpool", act.view, m.at("ref.maxpool"));

    // layer1 keeps the resolution; the other stages start with stride 2.
    const int64_t stage_stride[4] = {1, 2, 2, 2};
    for (int64_t stage = 1; stage <= 4; ++stage) {
        const std::string layer = "layer" + std::to_string(stage);
        for (int64_t blk = 0; blk < 2; ++blk) {
            const std::string p = layer + "." + std::to_string(blk);
            const int64_t stride = (blk == 0) ? stage_stride[stage - 1] : 1;
            Buf out = model::layer(p.c_str(), [&] {
                return basic_block(m, p, act.view, stride, stride != 1);
            });
            check.compare(p, out.view, m.at("ref." + p));
            act = std::move(out);
        }
    }

    Buf pooled = model::layer("avgpool", [&] {
        return model::global_avg_pool(act.view);
    });
    check.compare("avgpool", pooled.view, m.at("ref.avgpool"));

    Buf logits = model::layer("logits", [&] {
        return model::linear(model::flatten_to_2d(pooled.view),
                             m.at("fc.weight"), m.at("fc.bias"));
    });
    check.compare("logits", logits.view, m.at("ref.logits"));
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    std::string dir = NNOPS_MODEL_DATA_DIR "/resnet18";
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
        std::fprintf(stderr, "model_resnet18: %s\n", error.c_str());
        return 77;
    }

    std::printf("resnet18: %zu tensors from %s\n", artifact.count(),
                dir.c_str());
    // The untimed pass is the model-load step: it prepacks the Winograd weights
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
    model::prof_report("resnet18", ms);
    return check.report("resnet18") == 0 ? 0 : 1;
}
