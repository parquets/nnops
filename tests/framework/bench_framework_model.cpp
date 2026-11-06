/// @file bench_framework_model.cpp
/// @brief Model-level benchmarks for the ONNX framework: what the pipeline
///        costs to set up, and what each call costs on top of the kernels.
///
/// The operator benches price a kernel. This one prices the layer above it —
/// the part a caller pays for turning a file into a model and a model into
/// calls — on the two torchvision networks the accuracy test covers:
///
///   build     the artifact's f32 weights into an ONNX ModelProto (bytes)
///   load      Model::load_from_memory: wire decode + import, passes off
///   optimize  the same load with the passes on (fold, BN fold, fusion)
///   lower     Session construction: one operator per node
///   first     the first run: shape inference, buffer placement, allocation
///   steady    the same session afterwards — the number to compare with
///             models/resnet18.cpp, which runs the same layer shapes through
///             the operators directly
///
/// and two shapes of overhead:
///
///   passes    the optimized graph against the raw one, interleaved
///   batch     the same session at batch 1, 2 and 4 — the plan is rebuilt per
///             call, so this is where a per-call cost would show up as a
///             departure from linear
///   dispatch  a chain of trivial nodes, quoted in ns/node: the framework's
///             own cost per node with the arithmetic made negligible
///
/// Measurement rules: an untimed call runs first (page faults, weight prepack),
/// every row is the median of several rounds, and the A/B interleaves both arms
/// round by round so host drift lands on both.
///
/// No artifact means no model: the network is skipped, and 77 is returned when
/// nothing ran.

#include "framework/common/net_onnx.hpp"
#include "models/common/model_io.hpp"
#include "nnops/framework/framework.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace {

using framework_test::NetModel;
using model::ModelArtifact;
using nnops::framework::LoadOptions;
using nnops::framework::Model;
using nnops::framework::Session;
using nnops::framework::Tensor;

using BuildFn = NetModel (*)(const ModelArtifact&, bool, bool);

struct NetSpec {
    const char* name;
    BuildFn build;
};

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

/// One timed call, repeated as often as it takes to fill a batch.
double median_ms(int rounds, int iters, const std::function<void()>& fn)
{
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(rounds));
    for (int r = 0; r < rounds; ++r) {
        const Clock::time_point start = Clock::now();
        for (int i = 0; i < iters; ++i) { fn(); }
        samples.push_back(ms_since(start) / iters);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

/// The artifact's one-image input, repeated @p batch times. The artifact only
/// records batch 1, and every batch item is the same image — the point of the
/// row is the plan, not the data.
Tensor make_input(const ModelArtifact& art, const std::string& name, int64_t batch)
{
    const nnops::TensorView in = art.at(name);
    std::vector<int64_t> shape(in.shape(), in.shape() + in.rank());
    if (!shape.empty()) { shape[0] = batch; }
    Tensor t(std::span<const int64_t>(shape.data(), shape.size()), nnops::DataType::f32);

    const size_t per_image = in.nbytes();
    const size_t words = per_image / sizeof(float);
    float* dst = t.ptr<float>();
    for (int64_t b = 0; b < batch; ++b) {
        std::memcpy(dst + static_cast<size_t>(b) * words, in.ptr<float>(), per_image);
    }
    return t;
}

void print_row(const char* label, double ms)
{
    std::printf("  %-12s %9.3f ms\n", label, ms);
}

// ---- Pipeline --------------------------------------------------------------

void bench_pipeline(const ModelArtifact& art, const NetSpec& spec)
{
    const NetModel net = spec.build(art, /*record=*/false, /*dynamic_batch=*/false);
    const double mb = static_cast<double>(net.bytes.size()) / (1024.0 * 1024.0);

    const Clock::time_point build_start = Clock::now();
    spec.build(art, false, false);
    const double build_ms = ms_since(build_start);

    LoadOptions plain;
    plain.optimize = false;
    LoadOptions folded;
    folded.optimize = true;

    // One Model reused round by round: the second load replaces the first
    // graph, so the row measures the work and not the allocator's warm-up.
    Model scratch;
    const double load_ms = median_ms(3, 1, [&] {
        scratch.load_from_memory(net.bytes.data(), net.bytes.size(), plain);
    });
    const double opt_ms = median_ms(3, 1, [&] {
        scratch.load_from_memory(net.bytes.data(), net.bytes.size(), folded);
    });

    Model raw_model;
    raw_model.load_from_memory(net.bytes.data(), net.bytes.size(), plain);
    Model opt_model;
    opt_model.load_from_memory(net.bytes.data(), net.bytes.size(), folded);

    const double lower_ms = median_ms(3, 1, [&] { Session s(opt_model); });

    std::printf("  %-12s %9.1f MB\n", "graph", mb);
    std::printf("  %-12s %9.3f ms  (weights -> ModelProto bytes)\n", "build", build_ms);
    std::printf("  %-12s %9.3f ms  (ModelProto -> graph, %zu nodes)\n", "load", load_ms,
                raw_model.graph().nodes().size());
    std::printf("  %-12s %9.3f ms  (%zu nodes)\n", "optimize", opt_ms,
                opt_model.graph().nodes().size());
    std::printf("  %-12s %9.3f ms\n", "lower", lower_ms);

    Tensor input = make_input(art, net.input_name, 1);
    Session session(opt_model);
    std::vector<Tensor> outputs;

    const Clock::time_point first_start = Clock::now();
    session.run({&input, 1}, outputs);
    const double first_ms = ms_since(first_start);

    const double steady = median_ms(7, 3, [&] { session.run({&input, 1}, outputs); });
    std::printf("  %-12s %9.3f ms  (plan + allocation)\n", "first run", first_ms);
    std::printf("  %-12s %9.3f ms  (median of 21)\n", "steady run", steady);
    std::printf("  %-12s %9.1f MB  (plan peak)\n", "buffers",
                static_cast<double>(session.peak_buffer_bytes()) / (1024.0 * 1024.0));
}

// ---- What the passes buy ---------------------------------------------------

void bench_passes(const ModelArtifact& art, const NetSpec& spec)
{
    const NetModel net = spec.build(art, /*record=*/false, /*dynamic_batch=*/false);

    LoadOptions plain;
    plain.optimize = false;
    LoadOptions folded;
    folded.optimize = true;

    Model raw_model;
    Model opt_model;
    raw_model.load_from_memory(net.bytes.data(), net.bytes.size(), plain);
    opt_model.load_from_memory(net.bytes.data(), net.bytes.size(), folded);

    Tensor input = make_input(art, net.input_name, 1);
    Session raw(raw_model);
    Session opt(opt_model);

    std::vector<Tensor> outputs;
    if (!raw.ok() || !opt.ok()) {
        std::printf("  session failed: %s\n",
                    raw.ok() ? opt.last_error().c_str() : raw.last_error().c_str());
        return;
    }
    raw.run({&input, 1}, outputs);  // warm both: page faults, weight prepack
    opt.run({&input, 1}, outputs);

    // Interleaved, so a drifting host lands on both arms.
    const int rounds = 5;
    std::vector<double> a;
    std::vector<double> b;
    for (int r = 0; r < rounds; ++r) {
        const Clock::time_point start = Clock::now();
        for (int i = 0; i < 3; ++i) { raw.run({&input, 1}, outputs); }
        a.push_back(ms_since(start) / 3);

        const Clock::time_point start2 = Clock::now();
        for (int i = 0; i < 3; ++i) { opt.run({&input, 1}, outputs); }
        b.push_back(ms_since(start2) / 3);
    }
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    const double unfused = a[a.size() / 2];
    const double optimized = b[b.size() / 2];

    std::printf("  %-12s %9.3f ms  (%zu nodes)\n", "unfused", unfused,
                raw_model.graph().nodes().size());
    std::printf("  %-12s %9.3f ms  (%zu nodes, %.2fx)\n", "optimized", optimized,
                opt_model.graph().nodes().size(), unfused / optimized);
}

// ---- Batch scaling ---------------------------------------------------------

void bench_batch(const ModelArtifact& art, const NetSpec& spec)
{
    const NetModel net = spec.build(art, /*record=*/false, /*dynamic_batch=*/true);

    LoadOptions opts;
    Model model;
    model.load_from_memory(net.bytes.data(), net.bytes.size(), opts);

    Session session(model);
    if (!session.ok()) {
        std::printf("  session failed: %s\n", session.last_error().c_str());
        return;
    }

    std::vector<Tensor> outputs;
    double reference = 0.0;
    for (int64_t batch : {int64_t{1}, int64_t{2}, int64_t{4}}) {
        Tensor input = make_input(art, net.input_name, batch);
        session.run({&input, 1}, outputs);  // warm, and let the pool grow

        const double ms = median_ms(5, 2, [&] { session.run({&input, 1}, outputs); });
        if (batch == 1) { reference = ms; }
        std::printf("  batch %-6lld %9.3f ms  (%.2f ms/image)\n",
                    static_cast<long long>(batch), ms, ms / static_cast<double>(batch));
        if (reference > 0.0 && batch != 1) {
            std::printf("  %-12s %9.2fx  vs batch 1\n", "", ms / reference);
        }
    }
}

// ---- The framework's own per-node cost -------------------------------------

void bench_dispatch()
{
    // A chain of trivial nodes: every Add reads the previous one and a distinct
    // constant, so nothing folds, nothing fuses, and the arithmetic is a 1 K
    // element vector add — small enough that what is left is the framework.
    constexpr int64_t kNodes = 64;
    constexpr int64_t kWidth = 256;

    nnops::test::OnnxModel model;
    model.opset = 13;

    nnops::test::OnnxValue in;
    in.name = "x";
    in.dims = {1, kWidth};
    model.inputs.push_back(in);

    nnops::test::OnnxValue out;
    out.name = "y";
    out.dims = {1, kWidth};
    model.outputs.push_back(out);

    std::vector<float> ones(static_cast<size_t>(kWidth), 1.0f);
    for (int64_t i = 0; i < kNodes; ++i) {
        const std::string name = "c" + std::to_string(i);
        model.initializers.push_back(nnops::test::tensor_f32(name, {1, kWidth}, ones));

        nnops::test::OnnxNode add;
        add.name = "add" + std::to_string(i);
        add.op_type = "Add";
        add.inputs = {i == 0 ? "x" : "h" + std::to_string(i - 1), name};
        add.outputs = {i + 1 == kNodes ? "y" : "h" + std::to_string(i)};
        model.nodes.push_back(add);
    }

    const std::string bytes = model.serialize();
    LoadOptions opts;
    Model m;
    m.load_from_memory(bytes.data(), bytes.size(), opts);

    Session session(m);
    if (!session.ok()) {
        std::printf("  session failed: %s\n", session.last_error().c_str());
        return;
    }

    const std::vector<int64_t> shape{1, kWidth};
    Tensor input(std::span<const int64_t>(shape.data(), shape.size()), nnops::DataType::f32);
    std::vector<Tensor> outputs;
    session.run({&input, 1}, outputs);

    const double ms = median_ms(7, 20, [&] { session.run({&input, 1}, outputs); });
    const double ns = ms * 1e6 / static_cast<double>(kNodes);
    std::printf("  %-12s %9.3f ms  (%lld nodes + 1 plan, %.2f us/node)\n", "chain", ms,
                static_cast<long long>(kNodes), ns / 1000.0);
}

}  // namespace

int main(int argc, char** argv)
{
    std::string dir = NNOPS_MODEL_DATA_DIR;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] != '-') { dir = argv[i]; }
    }

    const NetSpec nets[] = {
        {"resnet18", framework_test::build_resnet18},
        {"mobilenetv2", framework_test::build_mobilenetv2},
    };

    std::printf("=== nnops framework model benchmarks ===\n");

    int ran = 0;
    for (const NetSpec& spec : nets) {
        ModelArtifact art;
        std::string error;
        if (!art.load(dir + "/" + spec.name, error)) {
            std::printf("--- %s ---\n  skipped (%s)\n", spec.name, error.c_str());
            continue;
        }
        ++ran;
        std::printf("--- %s pipeline ---\n", spec.name);
        bench_pipeline(art, spec);
        std::printf("--- %s passes ---\n", spec.name);
        bench_passes(art, spec);
        std::printf("--- %s batch ---\n", spec.name);
        bench_batch(art, spec);
    }

    std::printf("--- framework dispatch ---\n");
    bench_dispatch();

    if (ran == 0) {
        std::printf("no model artifact under %s; only the dispatch row ran\n", dir.c_str());
        return 77;
    }
    return 0;
}
