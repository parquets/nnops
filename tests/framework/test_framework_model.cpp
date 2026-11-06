/// @file test_framework_model.cpp
/// @brief Whole networks through the ONNX path, checked layer by layer against
///        the activations PyTorch recorded.
///
/// The model-level counterpart of test_framework_onnx.cpp: instead of a
/// hand-written model checked against hand-written arithmetic, this one takes a
/// torchvision network — ResNet-18 or MobileNetV2 — through the whole chain,
///
///   artifact weights -> ONNX ModelProto -> Model -> Session -> outputs
///
/// and compares every checkpoint the exporter recorded. A failure names the
/// layer that drifted, so a wrong attribute, a transposed weight or a
/// mishandled residual shows up at the node that caused it rather than as "the
/// logits are off".
///
/// Two passes per network, and they check different things:
///
///   1. unfused  — the load optimizer is off and every recorded activation is a
///                 graph output, so the session hands each one back. This is the
///                 ONNX -> nnops mapping under test, layer by layer.
///   2. optimized — the network's result only, so the passes may rewrite the
///                 graph (they may not touch a value a caller asked for). The
///                 node count must drop and the result must still match.
///
/// The artifact is the same one models/resnet18.cpp and models/mobilenetv2.cpp
/// read; see models/README.md. Without it the network is skipped and, when
/// nothing is left to run, the exit code is 77 with CTest reporting a skip.
///
/// Usage:  test_framework_model [artifact-dir] [--atol X] [--rtol Y]
/// Exit:   0 every checkpoint passed, 1 a checkpoint failed, 77 no artifact.

#include "framework/common/net_onnx.hpp"
#include "models/common/model_check.hpp"
#include "nnops/framework/framework.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using framework_test::NetModel;
using model::ModelArtifact;
using model::ModelCheck;
using nnops::framework::LoadOptions;
using nnops::framework::Model;
using nnops::framework::Session;
using nnops::framework::Tensor;
using nnops::Status;

using BuildFn = NetModel (*)(const ModelArtifact&, bool, bool);

struct NetSpec {
    const char* name;
    BuildFn build;
};

/// The network's graph outputs, computed. `error` is empty unless the run was
/// rejected — a load failure, a node that would not lower, or a shape the
/// session could not resolve.
struct Run {
    bool ok = false;
    std::string error;
    std::vector<Tensor> outputs;
    size_t peak_bytes = 0;
};

Run forward(const Model& model, const ModelArtifact& art, const std::string& input_name,
            bool optimize)
{
    Run run;
    const char* label = optimize ? "optimized" : "unfused";

    Session session(model);
    if (!session.ok()) {
        run.error = std::string(label) + ": session: " + session.last_error();
        return run;
    }

    const nnops::TensorView in = art.at(input_name);
    Tensor input(std::span<const int64_t>(in.shape(), static_cast<size_t>(in.rank())),
                 nnops::DataType::f32);
    const Status copied = input.copy_from(in.ptr<float>(), in.nbytes());
    if (nnops::is_error(copied)) {
        run.error = std::string(label) + ": input: " + nnops::to_string(copied);
        return run;
    }

    const Status st = session.run({&input, 1}, run.outputs);
    if (nnops::is_error(st)) {
        run.error = std::string(label) + ": run: " + session.last_error();
        return run;
    }
    run.peak_bytes = session.peak_buffer_bytes();
    run.ok = true;
    return run;
}

/// Pass 1: every graph output is compared against `ref.<name>`.
int check_layers(const ModelArtifact& art, const NetSpec& spec, const LoadOptions& opts,
                 double atol, double rtol)
{
    const NetModel traced = spec.build(art, /*record=*/true, /*dynamic_batch=*/false);

    Model model;
    const Status st = model.load_from_memory(traced.bytes.data(), traced.bytes.size(), opts);
    if (nnops::is_error(st)) {
        std::printf("  %-24s LOAD FAILED: %s\n", "load", model.last_error().c_str());
        return 1;
    }

    Run run = forward(model, art, traced.input_name, opts.optimize);
    if (!run.ok) {
        std::printf("  %-24s FAILED: %s\n", "run", run.error.c_str());
        return 1;
    }

    std::printf("%s: %zu nodes, %zu checkpoints, plan peak %.1f MB\n", spec.name,
                model.graph().nodes().size(), run.outputs.size(),
                static_cast<double>(run.peak_bytes) / (1024.0 * 1024.0));

    ModelCheck check(atol, rtol);
    for (size_t i = 0; i < run.outputs.size(); ++i) {
        const std::string& name = traced.output_names[i];
        check.compare(name, run.outputs[i].view(), art.at("ref." + name));
    }
    return check.report(spec.name);
}

/// Pass 2: the passes on against the passes off, on the network's own result.
int check_optimized(const ModelArtifact& art, const NetSpec& spec, double atol, double rtol)
{
    const NetModel net = spec.build(art, /*record=*/false, /*dynamic_batch=*/false);

    LoadOptions plain;
    plain.optimize = false;
    LoadOptions folded;
    folded.optimize = true;

    Model raw;
    Model opt;
    if (nnops::is_error(raw.load_from_memory(net.bytes.data(), net.bytes.size(), plain)) ||
        nnops::is_error(opt.load_from_memory(net.bytes.data(), net.bytes.size(), folded))) {
        std::printf("  %-24s LOAD FAILED\n", "optimize");
        return 1;
    }

    const size_t raw_nodes = raw.graph().nodes().size();
    const size_t opt_nodes = opt.graph().nodes().size();
    std::printf("%s: passes %zu -> %zu nodes\n", spec.name, raw_nodes, opt_nodes);
    if (opt_nodes >= raw_nodes) {
        std::printf("  the passes rewrote nothing\n");
        return 1;
    }

    Run a = forward(raw, art, net.input_name, false);
    Run b = forward(opt, art, net.input_name, true);
    if (!a.ok || !b.ok) {
        std::printf("  %-24s FAILED: %s\n", "run", (a.ok ? b.error : a.error).c_str());
        return 1;
    }

    ModelCheck check(atol, rtol);
    check.compare("logits", b.outputs[0].view(), art.at("ref.logits"));
    // The fold is exact arithmetic on the weights, not an approximation, so the
    // two graphs may differ by rounding and nothing else.
    check.compare("passes_equivalence", b.outputs[0].view(), a.outputs[0].view());
    return check.report(spec.name);
}

}  // namespace

int main(int argc, char** argv)
{
    std::string dir = NNOPS_MODEL_DATA_DIR;
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

    const NetSpec nets[] = {
        {"resnet18", framework_test::build_resnet18},
        {"mobilenetv2", framework_test::build_mobilenetv2},
    };

    LoadOptions unfused;
    unfused.optimize = false;

    int failed = 0;
    int ran = 0;
    for (const NetSpec& spec : nets) {
        ModelArtifact art;
        std::string error;
        if (!art.load(dir + "/" + spec.name, error)) {
            std::printf("%s: skipped (%s)\n", spec.name, error.c_str());
            continue;
        }
        ++ran;
        std::printf("--- %s (%zu tensors) ---\n", spec.name, art.count());
        failed += check_layers(art, spec, unfused, atol, rtol);
        failed += check_optimized(art, spec, atol, rtol);
    }

    if (ran == 0) {
        std::printf("no model artifact under %s; nothing to check\n", dir.c_str());
        return 77;
    }
    std::printf("\n%s\n", failed == 0 ? "ALL PASSED" : "FAILURES");
    return failed == 0 ? 0 : 1;
}
