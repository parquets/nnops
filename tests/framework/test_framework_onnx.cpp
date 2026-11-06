/// @file test_framework_onnx.cpp
/// @brief End-to-end tests for the ONNX framework: a model written byte by
///        byte, loaded, planned and run, checked against a hand-rolled
///        reference.
///
/// The reference here is deliberately hand-written rather than built out of the
/// operators under test: comparing Conv2D to Conv2D proves only that the
/// framework calls it. Each stage below is the arithmetic the model means —
/// the conv is a sum over a padded 3x3 window, the pooling a max over 2x2, the
/// Gemm a dot product plus a bias — so a wrong attribute (a stride, a padding,
/// a transposed weight) shows up as a wrong number.
///
/// The model exercises both halves of the runtime: a real operator chain, and
/// the metadata nodes (Flatten and Identity) whose outputs must end up aliased
/// onto their inputs' buffers.

#include "common/onnx_writer.hpp"
#include "common/test_harness.hpp"

#include "nnops/framework/framework.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using nnops::framework::LoadOptions;
using nnops::framework::Model;
using nnops::framework::Session;
using nnops::framework::SessionOptions;
using nnops::framework::Tensor;
using nnops::test::OnnxModel;
using nnops::test::OnnxNode;
using nnops::test::OnnxValue;
using nnops::test::attr_int;
using nnops::test::attr_ints;
using nnops::test::tensor_f32;

constexpr int64_t kInH = 4;
constexpr int64_t kInW = 4;
constexpr int64_t kOutChannels = 2;
constexpr int64_t kGemmN = 3;

/// Weights, held in the test so the reference reads the same numbers.
struct Weights {
    std::vector<float> conv;   // [2][1][3][3]
    std::vector<float> conv_b; // [2]
    std::vector<float> gemm;   // [8][3]
    std::vector<float> gemm_b; // [3]

    Weights()
    {
        conv.assign(2 * 1 * 3 * 3, 1.0f);
        conv[9] = 0.5f;  // oc 1 gets a different filter
        conv_b = {0.5f, -0.5f};
        gemm.resize(8 * kGemmN);
        for (int k = 0; k < 8; ++k) {
            for (int n = 0; n < kGemmN; ++n) {
                gemm[static_cast<size_t>(k * kGemmN + n)] =
                    0.1f * static_cast<float>(k + n + 1);
            }
        }
        gemm_b = {0.25f, -0.25f, 0.0f};
    }
};

/// Conv(k3 p1) → Relu → MaxPool(2,2) → Flatten → Identity → Gemm, batch
/// dynamic, written as the bytes an exporter would produce.
std::string build_model(const Weights& w)
{
    OnnxModel model;
    model.opset = 13;

    OnnxValue x;
    x.name = "x";
    x.dims = {-1, 1, kInH, kInW};
    model.inputs.push_back(x);

    OnnxValue y;
    y.name = "y";
    y.dims = {-1, 1, kGemmN};
    model.outputs.push_back(y);

    model.initializers.push_back(tensor_f32("w_conv", {2, 1, 3, 3}, w.conv));
    model.initializers.push_back(tensor_f32("b_conv", {2}, w.conv_b));
    model.initializers.push_back(tensor_f32("w_gemm", {8, kGemmN}, w.gemm));
    model.initializers.push_back(tensor_f32("b_gemm", {kGemmN}, w.gemm_b));

    OnnxNode conv;
    conv.name = "conv";
    conv.op_type = "Conv";
    conv.inputs = {"x", "w_conv", "b_conv"};
    conv.outputs = {"conv_out"};
    conv.attrs = {attr_ints("kernel_shape", {3, 3}),
                  attr_ints("strides", {1, 1}),
                  attr_ints("pads", {1, 1, 1, 1}),
                  attr_int("group", 1)};
    model.nodes.push_back(conv);

    OnnxNode relu;
    relu.name = "relu";
    relu.op_type = "Relu";
    relu.inputs = {"conv_out"};
    relu.outputs = {"relu_out"};
    model.nodes.push_back(relu);

    OnnxNode pool;
    pool.name = "pool";
    pool.op_type = "MaxPool";
    pool.inputs = {"relu_out"};
    pool.outputs = {"pool_out"};
    pool.attrs = {attr_ints("kernel_shape", {2, 2}), attr_ints("strides", {2, 2})};
    model.nodes.push_back(pool);

    OnnxNode flatten;
    flatten.name = "flatten";
    flatten.op_type = "Flatten";
    flatten.inputs = {"pool_out"};
    flatten.outputs = {"flat_out"};
    flatten.attrs = {attr_int("axis", 1)};
    model.nodes.push_back(flatten);

    OnnxNode identity;
    identity.name = "identity";
    identity.op_type = "Identity";
    identity.inputs = {"flat_out"};
    identity.outputs = {"ident_out"};
    model.nodes.push_back(identity);

    OnnxNode gemm;
    gemm.name = "gemm";
    gemm.op_type = "Gemm";
    gemm.inputs = {"ident_out", "w_gemm", "b_gemm"};
    gemm.outputs = {"y"};
    gemm.attrs = {attr_int("transB", 0)};
    model.nodes.push_back(gemm);

    return model.serialize();
}

/// The same arithmetic, written out by hand.
std::vector<float> reference(const Weights& w, const float* x, int64_t batch)
{
    std::vector<float> out(static_cast<size_t>(batch * kGemmN), 0.0f);
    const int64_t pooled = kInH / 2;
    constexpr int64_t kFlat = kOutChannels * 2 * 2;

    for (int64_t n = 0; n < batch; ++n) {
        const float* image = x + n * (kInH * kInW);
        std::vector<float> flat(static_cast<size_t>(kFlat), 0.0f);

        for (int64_t oc = 0; oc < kOutChannels; ++oc) {
            std::vector<float> plane(static_cast<size_t>(kInH * kInW), 0.0f);
            for (int64_t h = 0; h < kInH; ++h) {
                for (int64_t iw = 0; iw < kInW; ++iw) {
                    float sum = w.conv_b[static_cast<size_t>(oc)];
                    for (int64_t kh = 0; kh < 3; ++kh) {
                        for (int64_t kw = 0; kw < 3; ++kw) {
                            const int64_t ih = h + kh - 1;
                            const int64_t iwc = iw + kw - 1;
                            if (ih < 0 || ih >= kInH || iwc < 0 || iwc >= kInW) { continue; }
                            sum += image[ih * kInW + iwc] *
                                   w.conv[static_cast<size_t>((oc * 9) + kh * 3 + kw)];
                        }
                    }
                    plane[static_cast<size_t>(h * kInW + iw)] = sum > 0.0f ? sum : 0.0f;
                }
            }
            for (int64_t oh = 0; oh < pooled; ++oh) {
                for (int64_t ow = 0; ow < pooled; ++ow) {
                    float best = plane[static_cast<size_t>((oh * 2) * kInW + ow * 2)];
                    for (int64_t dh = 0; dh < 2; ++dh) {
                        for (int64_t dw = 0; dw < 2; ++dw) {
                            const float v = plane[static_cast<size_t>((oh * 2 + dh) * kInW + ow * 2 + dw)];
                            best = v > best ? v : best;
                        }
                    }
                    flat[static_cast<size_t>((oc * 4) + oh * 2 + ow)] = best;
                }
            }
        }

        for (int64_t k = 0; k < kGemmN; ++k) {
            float sum = w.gemm_b[static_cast<size_t>(k)];
            for (int64_t i = 0; i < kFlat; ++i) {
                sum += flat[static_cast<size_t>(i)] * w.gemm[static_cast<size_t>(i * kGemmN + k)];
            }
            out[static_cast<size_t>(n * kGemmN + k)] = sum;
        }
    }
    return out;
}

Tensor make_tensor(std::initializer_list<int64_t> shape)
{
    return Tensor(std::span<const int64_t>(shape.begin(), shape.size()),
                  nnops::DataType::f32);
}

std::vector<float> ramp(int64_t count)
{
    std::vector<float> values(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        values[static_cast<size_t>(i)] = static_cast<float>(i + 1);
    }
    return values;
}

void check(const Session& session, const Tensor& input, const Tensor& output,
           const Weights& w, int64_t batch, float tolerance)
{
    if (!session.ok()) { throw std::runtime_error(session.last_error()); }
    const auto want = reference(w, input.ptr<float>(), batch);
    NNOPS_EXPECT_EQ(static_cast<int64_t>(want.size()), output.numel());
    const float* got = output.ptr<float>();
    for (size_t i = 0; i < want.size(); ++i) {
        NNOPS_EXPECT_NEAR(got[i], want[i], tolerance);
    }
}

}  // namespace

// ---- A model round-trips through load, plan and run ----

NNOPS_TEST(framework_onnx_conv_chain)
{
    const Weights w;
    const std::string bytes = build_model(w);

    Model model;
    NNOPS_EXPECT_EQ(model.load_from_memory(bytes.data(), bytes.size()),
                    nnops::Status::ok);
    NNOPS_EXPECT_TRUE(model.valid());
    NNOPS_EXPECT_EQ(model.opset(), 13);
    NNOPS_EXPECT_EQ(static_cast<int64_t>(model.graph().inputs().size()), 1);
    NNOPS_EXPECT_EQ(static_cast<int64_t>(model.graph().outputs().size()), 1);

    Session session(model);
    NNOPS_EXPECT_TRUE(session.ok());

    Tensor input = make_tensor({1, 1, kInH, kInW});
    const auto values = ramp(kInH * kInW);
    NNOPS_EXPECT_EQ(input.copy_from(values.data(), values.size() * sizeof(float)),
                    nnops::Status::ok);

    std::vector<Tensor> outputs;
    NNOPS_EXPECT_EQ(session.run({&input, 1}, outputs), nnops::Status::ok);
    NNOPS_EXPECT_EQ(static_cast<int64_t>(outputs.size()), 1);
    NNOPS_EXPECT_EQ(outputs[0].rank(), 2);
    NNOPS_EXPECT_EQ(outputs[0].shape()[0], 1);
    NNOPS_EXPECT_EQ(outputs[0].shape()[1], kGemmN);

    check(session, input, outputs[0], w, 1, 1e-3f);
}

// ---- One session, several batches: shapes are resolved per call ----

NNOPS_TEST(framework_onnx_dynamic_batch)
{
    const Weights w;
    const std::string bytes = build_model(w);

    Model model;
    NNOPS_EXPECT_EQ(model.load_from_memory(bytes.data(), bytes.size()),
                    nnops::Status::ok);
    Session session(model);
    NNOPS_EXPECT_TRUE(session.ok());

    for (int64_t batch : {int64_t{2}, int64_t{3}}) {
        Tensor input = make_tensor({batch, 1, kInH, kInW});
        const auto values = ramp(batch * kInH * kInW);
        NNOPS_EXPECT_EQ(input.copy_from(values.data(), values.size() * sizeof(float)),
                        nnops::Status::ok);

        std::vector<Tensor> outputs;
        NNOPS_EXPECT_EQ(session.run({&input, 1}, outputs), nnops::Status::ok);
        NNOPS_EXPECT_EQ(outputs[0].shape()[0], batch);
        check(session, input, outputs[0], w, batch, 1e-3f);
    }

    // The pool grew to the largest batch and stayed there.
    NNOPS_EXPECT_TRUE(session.peak_buffer_bytes() > 0);
}

// ---- The passes must not change what the model computes ----

NNOPS_TEST(framework_onnx_optimize_equivalence)
{
    const Weights w;
    const std::string bytes = build_model(w);

    LoadOptions plain;
    plain.optimize = false;
    LoadOptions folded;
    folded.optimize = true;

    Model raw;
    Model opt;
    NNOPS_EXPECT_EQ(raw.load_from_memory(bytes.data(), bytes.size(), plain),
                    nnops::Status::ok);
    NNOPS_EXPECT_EQ(opt.load_from_memory(bytes.data(), bytes.size(), folded),
                    nnops::Status::ok);

    // The chain folds nothing and fuses Relu into the conv's epilogue, so the
    // optimized graph is strictly smaller.
    NNOPS_EXPECT_TRUE(opt.graph().nodes().size() < raw.graph().nodes().size());

    Tensor input = make_tensor({2, 1, kInH, kInW});
    const auto values = ramp(2 * kInH * kInW);
    NNOPS_EXPECT_EQ(input.copy_from(values.data(), values.size() * sizeof(float)),
                    nnops::Status::ok);

    Session session_raw(raw);
    Session session_opt(opt);
    NNOPS_EXPECT_TRUE(session_raw.ok());
    NNOPS_EXPECT_TRUE(session_opt.ok());

    std::vector<Tensor> a;
    std::vector<Tensor> b;
    NNOPS_EXPECT_EQ(session_raw.run({&input, 1}, a), nnops::Status::ok);
    NNOPS_EXPECT_EQ(session_opt.run({&input, 1}, b), nnops::Status::ok);
    NNOPS_EXPECT_EQ(a[0].numel(), b[0].numel());

    const float* pa = a[0].ptr<float>();
    const float* pb = b[0].ptr<float>();
    for (int64_t i = 0; i < a[0].numel(); ++i) {
        NNOPS_EXPECT_NEAR(pa[i], pb[i], 1e-5f);
    }
}

// ---- Metadata nodes: Reshape reads its target from an initializer ----

NNOPS_TEST(framework_onnx_reshape)
{
    OnnxModel model;
    model.opset = 13;

    OnnxValue x;
    x.name = "x";
    x.dims = {1, 2, 3, 4};
    model.inputs.push_back(x);

    OnnxValue y;
    y.name = "y";
    y.dims = {4, 6};
    model.outputs.push_back(y);

    model.initializers.push_back(nnops::test::tensor_i64("shape", {2}, {4, 6}));

    OnnxNode reshape;
    reshape.op_type = "Reshape";
    reshape.inputs = {"x", "shape"};
    reshape.outputs = {"y"};
    model.nodes.push_back(reshape);

    const std::string bytes = model.serialize();
    Model loaded;
    NNOPS_EXPECT_EQ(loaded.load_from_memory(bytes.data(), bytes.size()),
                    nnops::Status::ok);

    Session session(loaded);
    NNOPS_EXPECT_TRUE(session.ok());

    Tensor input = make_tensor({1, 2, 3, 4});
    const auto values = ramp(24);
    NNOPS_EXPECT_EQ(input.copy_from(values.data(), values.size() * sizeof(float)),
                    nnops::Status::ok);

    std::vector<Tensor> outputs;
    NNOPS_EXPECT_EQ(session.run({&input, 1}, outputs), nnops::Status::ok);
    NNOPS_EXPECT_EQ(outputs[0].rank(), 2);
    NNOPS_EXPECT_EQ(outputs[0].shape()[0], 4);
    NNOPS_EXPECT_EQ(outputs[0].shape()[1], 6);

    const float* got = outputs[0].ptr<float>();
    for (int64_t i = 0; i < 24; ++i) {
        NNOPS_EXPECT_NEAR(got[i], values[static_cast<size_t>(i)], 0.0f);
    }
}

// ---- A malformed model is a Status with the node named ----

NNOPS_TEST(framework_onnx_reports_unsupported)
{
    OnnxModel model;
    OnnxValue x;
    x.name = "x";
    x.dims = {1, 4};
    model.inputs.push_back(x);
    OnnxValue y;
    y.name = "y";
    y.dims = {1, 4};
    model.outputs.push_back(y);

    OnnxNode weird;
    weird.name = "gate";
    weird.op_type = "If";
    weird.inputs = {"x"};
    weird.outputs = {"y"};
    model.nodes.push_back(weird);

    const std::string bytes = model.serialize();
    Model loaded;
    NNOPS_EXPECT_TRUE(nnops::is_error(loaded.load_from_memory(bytes.data(), bytes.size())));
    NNOPS_EXPECT_FALSE(loaded.valid());
    NNOPS_EXPECT_TRUE(loaded.last_error().find("gate") != std::string::npos);
}

// ---- A truncated file is rejected, not read past ----

NNOPS_TEST(framework_onnx_truncated)
{
    const Weights w;
    const std::string bytes = build_model(w);
    const std::string cut = bytes.substr(0, bytes.size() / 2);

    Model model;
    NNOPS_EXPECT_TRUE(nnops::is_error(model.load_from_memory(cut.data(), cut.size())));
    NNOPS_EXPECT_FALSE(model.last_error().empty());
}
