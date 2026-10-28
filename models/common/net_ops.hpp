#pragma once
/// @file net_ops.hpp
/// @brief One-line wrappers around the nnops operators a CNN forward pass
///        needs: shape inference, buffer allocation and execution in one call.
///
/// The model runners read like the PyTorch definition of the network — that is
/// the point. Every wrapper returns a freshly allocated activation; nothing is
/// reused, so a runner can hold intermediates for the per-layer comparison.

#include "nnops/ops/activation.hpp"
#include "nnops/ops/clamp.hpp"
#include "nnops/ops/conv2d.hpp"
#include "nnops/ops/depthwise_conv.hpp"
#include "nnops/ops/eltwise.hpp"
#include "nnops/ops/layout_convert.hpp"
#include "nnops/ops/linear.hpp"
#include "nnops/ops/norm.hpp"
#include "nnops/ops/pooling.hpp"

#include "model_io.hpp"
#include "model_prof.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace model {

using nnops::Backend;
using nnops::DataType;
using nnops::TensorDesc;
using nnops::TensorView;

/// The ComputeContext every wrapper hands to compute(). Default: no hooks, i.e.
/// the sequential backend — which is what the model runners measure. A
/// model-level bench installs a thread pool here rather than plumbing a context
/// through every wrapper.
inline nnops::ComputeContext& context()
{
    static nnops::ComputeContext ctx;
    return ctx;
}

/// Run one operator and hand back its (single) output buffer.
template <class Op>
Buf run(const char* what, Op& op, std::span<const TensorView> inputs)
{
    ProfScope prof(what);
    std::vector<TensorDesc> in_desc;
    in_desc.reserve(inputs.size());
    for (const auto& v : inputs) { in_desc.push_back(v.desc()); }

    const auto out_desc = op.getOutputTensorDesc(in_desc);
    Buf out = alloc(out_desc[0]);

    TensorView outputs[] = {out.view};
    require_ok(op.compute(outputs, inputs, context()), what);
    return out;
}

/// The Winograd-transformed copy of a 3×3 weight (U = G·g·Gᵀ), made once per
/// weight buffer — the artifact hands out one view per tensor, so the pointer
/// identifies it, and a runner that calls conv2d per layer pays for the
/// transform once.
inline const TensorView& winograd_weight(const TensorView& w,
                                         const nnops::Conv2DAttributes& attrs)
{
    static std::map<const void*, Buf> cache;
    auto it = cache.find(w.ptr<float>());
    if (it != cache.end()) { return it->second.view; }

    // Two-step prepack: query the U descriptor, allocate it, fill it.
    auto op = nnops::Conv2D::create(attrs, Backend::CPU);
    TensorView query;
    const TensorView in[] = {w};
    TensorView q[] = {query};
    op->prepackWeights(in, q);
    Buf u = alloc(q[0].desc());
    TensorView pack[] = {u.view};
    op->prepackWeights(in, pack);

    return cache.emplace(w.ptr<float>(), std::move(u)).first->second.view;
}

/// Run a conv on the weight it should take: Winograd reads the transformed U
/// when the geometry pays for it, the raw filter otherwise. Shape inference
/// always reads the raw weight, so it is queried first — only compute() sees U.
inline Buf conv2d_run(const TensorView& x, const TensorView& w,
                      const TensorView* b, const nnops::Conv2DAttributes& attrs,
                      const char* wino_row = "conv2d/wino",
                      const char* gemm_row = "conv2d/im2col")
{
    const bool wino = nnops::winograd_applicable(
        attrs, x.shape(1) / attrs.groups, x.shape(2), x.shape(3));
    // The route is in the row name, so the profile can price the two against
    // each other just by reading the table. A fused caller renames the rows, so
    // a folded conv and a plain one do not share a line.
    ProfScope prof(wino ? wino_row : gemm_row);
    auto op = nnops::Conv2D::create(attrs, Backend::CPU);
    const TensorView u = wino ? winograd_weight(w, attrs) : w;

    TensorDesc in_desc[3] = {x.desc(), w.desc()};
    TensorView in[3] = {x, u};
    size_t n = 2;
    if (b) {
        in_desc[n] = b->desc();
        in[n++] = *b;
    }
    Buf out = alloc(op->getOutputTensorDesc({in_desc, n})[0]);
    TensorView outputs[] = {out.view};
    require_ok(op->compute(outputs, {in, n}, context()), "conv2d");
    return out;
}

inline Buf conv2d(const TensorView& x, const TensorView& w, const TensorView& b,
                  int64_t kernel, int64_t stride, int64_t pad,
                  int64_t groups = 1)
{
    nnops::Conv2DAttributes attrs;
    attrs.kernel_size = {kernel, kernel};
    attrs.stride = {stride, stride};
    attrs.padding = {pad, pad};
    attrs.groups = groups;
    return conv2d_run(x, w, &b, attrs);
}

/// Convolution with no bias (every conv in these networks is followed by a
/// BatchNorm, which carries the shift).
inline Buf conv2d(const TensorView& x, const TensorView& w,
                  int64_t kernel, int64_t stride, int64_t pad,
                  int64_t groups = 1)
{
    nnops::Conv2DAttributes attrs;
    attrs.kernel_size = {kernel, kernel};
    attrs.stride = {stride, stride};
    attrs.padding = {pad, pad};
    attrs.groups = groups;
    return conv2d_run(x, w, nullptr, attrs);
}

/// Depthwise 2D convolution. The torch weight [C, 1, KH, KW] and bias [C] go in
/// as they are: a planar (NCHW) input takes the raw filter, with no prepack.
inline Buf depthwise_conv(const TensorView& x, const TensorView& w, const TensorView& b,
                          int64_t kernel, int64_t stride, int64_t pad)
{
    nnops::DepthwiseConvAttributes attrs;
    attrs.kernel_size = {1, kernel, kernel};
    attrs.stride = {1, stride, stride};
    attrs.padding = {0, pad, pad};
    auto op = nnops::DepthwiseConv::create(attrs, Backend::CPU);
    const TensorView in[] = {x, w, b};
    return run("depthwise_conv", *op, in);
}

inline Buf depthwise_conv(const TensorView& x, const TensorView& w,
                          int64_t kernel, int64_t stride, int64_t pad)
{
    nnops::DepthwiseConvAttributes attrs;
    attrs.kernel_size = {1, kernel, kernel};
    attrs.stride = {1, stride, stride};
    attrs.padding = {0, pad, pad};
    auto op = nnops::DepthwiseConv::create(attrs, Backend::CPU);
    const TensorView in[] = {x, w};
    return run("depthwise_conv", *op, in);
}

inline Buf batch_norm(const TensorView& x, const TensorView& scale,
                      const TensorView& bias, const TensorView& mean,
                      const TensorView& var, float eps = 1e-5f)
{
    nnops::NormAttributes attrs;
    attrs.type = nnops::NormType::BatchNorm;
    attrs.epsilon = eps;
    auto op = nnops::Norm::create(attrs, Backend::CPU);
    const TensorView in[] = {x, scale, bias, mean, var};
    return run("batch_norm", *op, in);
}

inline Buf relu(const TensorView& x)
{
    nnops::ActivationAttributes attrs;
    attrs.type = nnops::ActivationType::Relu;
    auto op = nnops::Activation::create(attrs, Backend::CPU);
    const TensorView in[] = {x};
    return run("relu", *op, in);
}

/// ReLU6 — as PyTorch defines it, a clamp to [0, 6].
inline Buf relu6(const TensorView& x)
{
    nnops::ClampAttributes attrs;
    attrs.min_val = 0.0f;
    attrs.max_val = 6.0f;
    auto op = nnops::Clamp::create(attrs, Backend::CPU);
    const TensorView in[] = {x};
    return run("relu6", *op, in);
}

// ============================================================
// conv -> BatchNorm -> activation as one operator call
// ============================================================
//
// A torchvision BatchNorm is a per-channel affine on the conv's output:
//
//   y = gamma * (conv(x, w) - mean) / sqrt(var + eps) + beta
//     = conv(x, w * s) + t,   s = gamma / sqrt(var + eps),   t = beta - mean * s
//
// so for inference it folds into the weight and a bias — exactly, up to the
// rounding order. What that buys is not arithmetic but the two extra passes:
// the activation then rides the conv's write-back as its epilogue, and the BN
// never makes a round trip through memory at all.
//
// The fold is a load-time transform, and a runner calls the fused wrapper once
// per layer per forward, so it is cached per weight buffer — the same reasoning
// as winograd_weight()'s U cache.

/// The four tensors of a BatchNorm, plus epsilon. By value: the artifact hands
/// out one view per tensor and they must outlive the fold.
struct BatchNormParams {
    TensorView gamma;
    TensorView beta;
    TensorView mean;
    TensorView var;
    float eps = 1e-5f;
};

/// torchvision's BatchNorm tensor names, under a module prefix.
inline BatchNormParams batch_norm_params(const ModelArtifact& m,
                                         const std::string& prefix)
{
    return {m.at(prefix + ".weight"), m.at(prefix + ".bias"),
            m.at(prefix + ".running_mean"), m.at(prefix + ".running_var"), 1e-5f};
}

inline Buf batch_norm(const TensorView& x, const BatchNormParams& bn)
{
    return batch_norm(x, bn.gamma, bn.beta, bn.mean, bn.var, bn.eps);
}

/// True when the runners should fold (NNOPS_MODEL_FUSE=0 is the unfused
/// baseline: conv -> batch_norm -> activation as three separate calls).
inline bool fuse_enabled()
{
    static const bool on = [] {
        const char* s = std::getenv("NNOPS_MODEL_FUSE");
        return !(s != nullptr &&
                 (std::strcmp(s, "0") == 0 || std::strcmp(s, "false") == 0));
    }();
    return on;
}

namespace detail {

/// A 1-D f32 buffer over @p values — the folded bias, and nothing else so far.
inline Buf make_vector(std::vector<float> values)
{
    Buf b;
    b.data = std::move(values);
    const int64_t shape[] = {static_cast<int64_t>(b.data.size())};
    b.view = TensorView(shape, DataType::f32, b.data.data());
    return b;
}

/// A conv+BN pair folded into the weight it should be called with: the scaled
/// filter and the shift as a bias vector.
struct Folded {
    Buf weight;
    Buf bias;
    // What the fold was made from. The cache is keyed on the weight's address,
    // and an address is only unique while the tensor is alive: a caller who
    // frees one model and loads another can see the same pointer. Rebuild
    // rather than hand back a fold of somebody else's filter.
    std::vector<int64_t> shape;
    const void* bn_key[4] = {nullptr, nullptr, nullptr, nullptr};
};

inline const Folded& folded(const TensorView& w, const BatchNormParams& bn)
{
    static std::map<const void*, Folded> cache;
    const void* bn_key[4] = {bn.gamma.ptr<float>(), bn.beta.ptr<float>(),
                             bn.mean.ptr<float>(), bn.var.ptr<float>()};
    auto it = cache.find(w.ptr<float>());
    if (it != cache.end()) {
        bool same = it->second.shape.size() == static_cast<size_t>(w.rank());
        for (int64_t d = 0; same && d < w.rank(); ++d) {
            same = it->second.shape[static_cast<size_t>(d)] == w.shape(d);
        }
        for (int i = 0; same && i < 4; ++i) { same = it->second.bn_key[i] == bn_key[i]; }
        if (same) { return it->second; }
    }

    // Output channel block: OC filters for a conv, one filter per channel for
    // depthwise (OC == C there), so the loop covers both.
    const int64_t oc   = w.shape(0);
    const float* gamma = bn.gamma.ptr<float>();
    const float* beta  = bn.beta.ptr<float>();
    const float* mean  = bn.mean.ptr<float>();
    const float* var   = bn.var.ptr<float>();

    Folded f;
    f.weight = alloc(w.desc());
    const int64_t per = static_cast<int64_t>(f.weight.data.size()) / oc;
    std::vector<float> bias(static_cast<size_t>(oc));
    const float* src = w.ptr<float>();
    float* dst = f.weight.ptr();
    for (int64_t o = 0; o < oc; ++o) {
        const float s = gamma[o] / std::sqrt(var[o] + bn.eps);
        for (int64_t i = 0; i < per; ++i) {
            dst[o * per + i] = src[o * per + i] * s;
        }
        bias[static_cast<size_t>(o)] = beta[o] - mean[o] * s;
    }
    f.bias = make_vector(std::move(bias));
    f.shape.assign(w.rank(), 0);
    for (int64_t d = 0; d < w.rank(); ++d) { f.shape[static_cast<size_t>(d)] = w.shape(d); }
    for (int i = 0; i < 4; ++i) { f.bn_key[i] = bn_key[i]; }
    cache[w.ptr<float>()] = std::move(f);
    return cache[w.ptr<float>()];
}

}  // namespace detail

/// conv -> BatchNorm -> @p act in one call: BN folded into the weight and bias,
/// @p act (Relu / Relu6 / None) riding the conv's epilogue. None is the
/// projection case — conv+BN only, which still merges two passes into one.
inline Buf conv2d_bn_act(const TensorView& x, const TensorView& w,
                         const BatchNormParams& bn,
                         nnops::EpilogueActivateType act,
                         int64_t kernel, int64_t stride, int64_t pad,
                         int64_t groups = 1)
{
    nnops::Conv2DAttributes attrs;
    attrs.kernel_size = {kernel, kernel};
    attrs.stride = {stride, stride};
    attrs.padding = {pad, pad};
    attrs.groups = groups;
    attrs.epilogue.type = act;
    const detail::Folded& f = detail::folded(w, bn);
    return conv2d_run(x, f.weight.view, &f.bias.view, attrs,
                      "conv2d_fused/wino", "conv2d_fused/im2col");
}

/// Depthwise conv -> BatchNorm -> @p act in one call. The filter is [C, 1, KH,
/// KW] and the BN is per channel, which is exactly the filter's own axis.
inline Buf depthwise_bn_act(const TensorView& x, const TensorView& w,
                            const BatchNormParams& bn,
                            nnops::EpilogueActivateType act,
                            int64_t kernel, int64_t stride, int64_t pad)
{
    nnops::DepthwiseConvAttributes attrs;
    attrs.kernel_size = {1, kernel, kernel};
    attrs.stride = {1, stride, stride};
    attrs.padding = {0, pad, pad};
    attrs.epilogue.type = act;
    const detail::Folded& f = detail::folded(w, bn);
    auto op = nnops::DepthwiseConv::create(attrs, Backend::CPU);
    const TensorView in[] = {x, f.weight.view, f.bias.view};
    return run("depthwise_fused", *op, in);
}

inline Buf add(const TensorView& a, const TensorView& b)
{
    nnops::EltwiseAttributes attrs;
    attrs.type = nnops::EltwiseType::Add;
    auto op = nnops::Eltwise::create(attrs, Backend::CPU);
    const TensorView in[] = {a, b};
    return run("add", *op, in);
}

inline Buf to_layout(const TensorView& x, nnops::TensorLayout target)
{
    auto op = nnops::LayoutConvert::create(target, Backend::CPU);
    const TensorView in[] = {x};
    return run("layout_convert", *op, in);
}

/// MaxPool2d, straight on the NCHW tensor — the pooling kernels take planar
/// input, so the runner stays in the layout the PyTorch references use.
inline Buf max_pool(const TensorView& x, int64_t kernel, int64_t stride,
                    int64_t pad)
{
    nnops::PoolingAttributes attrs;
    attrs.type = nnops::PoolingType::Max;
    attrs.kernel_shape = {1, kernel, kernel};
    attrs.stride = {1, stride, stride};
    attrs.padding = {0, pad, pad};
    auto op = nnops::Pooling::create(attrs, Backend::CPU);
    const TensorView in[] = {x};
    return run("max_pool", *op, in);
}

/// Average over the whole spatial extent (PyTorch's AdaptiveAvgPool2d(1)) —
/// the pooling operator's global mode, one call instead of a mean per axis.
inline Buf global_avg_pool(const TensorView& x)
{
    nnops::PoolingAttributes attrs;
    attrs.type = nnops::PoolingType::Average;
    attrs.global = true;
    auto op = nnops::Pooling::create(attrs, Backend::CPU);
    const TensorView in[] = {x};
    return run("global_avg_pool", *op, in);
}

inline Buf linear(const TensorView& x, const TensorView& w, const TensorView& b)
{
    auto op = nnops::Linear::create(nnops::LinearAttributes{}, Backend::CPU);
    const TensorView in[] = {x, w, b};
    return run("linear", *op, in);
}

/// A [N, C, 1, 1] pooled tensor seen as [N, C] — the reshape is a no-op on
/// contiguous data, and Linear takes its reduction over the last axis.
inline TensorView flatten_to_2d(TensorView& pooled)
{
    const int64_t shape[] = {pooled.shape(0), pooled.shape(1)};
    return TensorView(shape, pooled.data_type(), pooled.ptr<uint8_t>());
}

}  // namespace model
