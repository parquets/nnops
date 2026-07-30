/// @file depthwise_conv2d.cpp
/// @brief DepthwiseConv2D operator dispatch — connects class API to backend implementations.
///
/// Input/output are always NCHWC8. Weights are prepacked via prepackWeights()
/// before compute(). The prepack step converts raw NCHW weight [C, 1, KH, KW]
/// to dense [C8, KH, KW, 8] format for efficient SIMD access.

#include "nnops/ops/depthwise_conv2d.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <cstring>

namespace nnops {

using nnops::simd::s_load;
using nnops::simd::s_store;

// Forward declarations of backend kernel entry points
namespace backend::cpu::reference {
    void depthwise_conv2d_ref(const DepthwiseConv2DAttributes& attrs,
                               TensorView& output,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* workspace);
}

namespace backend::cpu {
    void depthwise_conv2d_cpu(const DepthwiseConv2DAttributes& attrs,
                               TensorView& output,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void depthwise_conv2d_cuda(const DepthwiseConv2DAttributes& attrs,
                                TensorView& output,
                                std::span<const TensorView> inputs,
                                const ComputeContext& ctx,
                                void* workspace);
}
#endif

// ============================================================
// Impl — holds the backend-bound kernel function pointer
// ============================================================
struct DepthwiseConv2D::Impl {
    using KernelFn = void (*)(const DepthwiseConv2DAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_depthwise_conv2d_kernel(Backend backend) -> DepthwiseConv2D::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::depthwise_conv2d_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::depthwise_conv2d_cuda;
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return nullptr;
#endif
    }
    return nullptr;
}

// ============================================================
// Prepack helpers — dtype-dispatched weight & bias packing
// ============================================================

template <typename T>
void prepack_dwconv_weight_impl(const TensorView& weight_nchw,
                                 TensorView& packed_out)
{
    const int64_t C  = weight_nchw.shape(0);
    const int64_t KH = weight_nchw.shape(2);
    const int64_t KW = weight_nchw.shape(3);
    const int64_t C8 = (C + 7) / 8;

    const auto* src = weight_nchw.ptr<T>();
    auto* dst = packed_out.ptr<T>();

    for (int64_t c8 = 0; c8 < C8; ++c8) {
        for (int64_t kh = 0; kh < KH; ++kh) {
            for (int64_t kw = 0; kw < KW; ++kw) {
                int64_t dst_base = (c8 * KH * KW + kh * KW + kw) * 8;
                int64_t c_base = c8 * 8;
                for (int64_t lane = 0; lane < 8; ++lane) {
                    int64_t c = c_base + lane;
                    float val = (c < C)
                        ? s_load(&src[c * KH * KW + kh * KW + kw])
                        : 0.0f;
                    s_store(&dst[dst_base + lane], val);
                }
            }
        }
    }
}

template <typename T>
void prepack_dwconv_bias_impl(const TensorView& bias_nchw,
                               TensorView& packed_out)
{
    const int64_t C  = bias_nchw.shape(0);
    const int64_t C8 = (C + 7) / 8;

    const auto* src = bias_nchw.ptr<T>();
    auto* dst = packed_out.ptr<T>();

    for (int64_t c8 = 0; c8 < C8; ++c8) {
        int64_t c_base = c8 * 8;
        for (int64_t lane = 0; lane < 8; ++lane) {
            int64_t c = c_base + lane;
            float val = (c < C) ? s_load(&src[c]) : 0.0f;
            s_store(&dst[c8 * 8 + lane], val);
        }
    }
}

void prepack_dwconv_weight(const TensorView& weight_nchw,
                            TensorView& packed_out)
{
    switch (weight_nchw.data_type()) {
    case DataType::f32:
        prepack_dwconv_weight_impl<float>(weight_nchw, packed_out);
        return;
    case DataType::f16:
        prepack_dwconv_weight_impl<backend::cpu::half>(weight_nchw, packed_out);
        return;
    default:
        NNOPS_ASSERT(!"prepack_dwconv_weight: unsupported dtype");
    }
}

void prepack_dwconv_bias(const TensorView& bias_nchw,
                          TensorView& packed_out)
{
    switch (bias_nchw.data_type()) {
    case DataType::f32:
        prepack_dwconv_bias_impl<float>(bias_nchw, packed_out);
        return;
    case DataType::f16:
        prepack_dwconv_bias_impl<backend::cpu::half>(bias_nchw, packed_out);
        return;
    default:
        NNOPS_ASSERT(!"prepack_dwconv_bias: unsupported dtype");
    }
}

}  // anonymous namespace

// ============================================================
// Factory
// ============================================================
std::unique_ptr<DepthwiseConv2D> DepthwiseConv2D::create(
    const DepthwiseConv2DAttributes& attrs, Backend backend)
{
    return std::unique_ptr<DepthwiseConv2D>(new DepthwiseConv2D(attrs, backend));
}

// ============================================================
// Constructor
// ============================================================
DepthwiseConv2D::DepthwiseConv2D(const DepthwiseConv2DAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_depthwise_conv2d_kernel(backend);
}

// ============================================================
// getOutputShapes
// ============================================================
std::vector<TensorDesc> DepthwiseConv2D::getOutputShapes(
    std::span<const TensorDesc> inputs) const
{
    return {depthwise_conv2d_output_shape(
        attrs_.kernel_size, attrs_.stride, attrs_.dilation, attrs_.padding,
        0 /* NOTSET */, inputs)};
}

// ============================================================
// prepackWeights — dual-behavior: empty → metadata, filled → pack
// ============================================================
void DepthwiseConv2D::prepackWeights(std::span<const TensorView> inputs,
                                      std::span<TensorView> outputs,
                                      const ComputeContext& ctx)
{
    (void)ctx;
    NNOPS_ASSERT(inputs.size() >= 1 && inputs.size() <= 2);
    const bool has_bias = inputs.size() > 1;
    const int64_t num_outputs = has_bias ? 2 : 1;
    NNOPS_ASSERT(outputs.size() == static_cast<size_t>(num_outputs));

    const auto& weight = inputs[0];
    const int64_t C  = weight.shape(0);
    const int64_t KH = weight.shape(2);
    const int64_t KW = weight.shape(3);
    const int64_t C8 = (C + 7) / 8;
    const DataType dtype = weight.data_type();
    const int64_t elem_size = static_cast<int64_t>(data_type_size(dtype));

    if (outputs[0].is_empty()) {
        // ---- Query mode: fill metadata so caller can allocate ----

        // Packed weight: [C8, KH, KW, 8]
        {
            const int64_t w_shape[] = {C8, KH, KW, 8};
            const int64_t w_pitch = KW * 8 * elem_size;
            outputs[0] = TensorView(
                std::span<const int64_t>(w_shape, 4), dtype,
                nullptr, w_pitch, TensorLayout::NCHW);
        }

        // Packed bias: [C8, 8]
        if (has_bias) {
            const int64_t b_shape[] = {C8, 8};
            const int64_t b_pitch = 8 * elem_size;
            outputs[1] = TensorView(
                std::span<const int64_t>(b_shape, 2), dtype,
                nullptr, b_pitch, TensorLayout::NCHW);
        }
    } else {
        // ---- Pack mode: fill the buffer ----
        prepack_dwconv_weight(weight, outputs[0]);

        if (has_bias) {
            prepack_dwconv_bias(inputs[1], outputs[1]);
        }
    }
}

// ============================================================
// compute — direct dispatch to backend (input/output already NCHWC8)
// ============================================================
void DepthwiseConv2D::compute(std::span<TensorView> outputs,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* workspace)
{
    NNOPS_ASSERT(inputs.size() >= 2);
    NNOPS_ASSERT(inputs.size() <= 3);  // input, prepacked_weight [, prepacked_bias]
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());
    NNOPS_ASSERT(!inputs[1].is_empty());

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

// ============================================================
// Functional API — convenience wrappers (use prepack flow)
// ============================================================
void depthwise_conv2d(const TensorView& input,
                       const TensorView& weight,
                       TensorView& output,
                       const DepthwiseConv2DAttributes& attrs,
                       const ComputeContext& ctx,
                       void* workspace)
{
    auto op = DepthwiseConv2D::create(attrs, ctx.expected_backend);

    // Prepack weight (query → allocate → pack)
    TensorView packed_w;
    op->prepackWeights({&weight, 1}, {&packed_w, 1});

    std::vector<char> wbuf(packed_w.nbytes());
    packed_w = TensorView(packed_w.shape_span(), packed_w.data_type(),
                           wbuf.data(), packed_w.pitch(), packed_w.layout());
    op->prepackWeights({&weight, 1}, {&packed_w, 1});

    const TensorView ins[] = {input, packed_w};
    op->compute(output, ins, ctx, workspace);
}

void depthwise_conv2d(const TensorView& input,
                       const TensorView& weight,
                       const TensorView& bias,
                       TensorView& output,
                       const DepthwiseConv2DAttributes& attrs,
                       const ComputeContext& ctx,
                       void* workspace)
{
    auto op = DepthwiseConv2D::create(attrs, ctx.expected_backend);

    // Prepack weight + bias (query → allocate → pack)
    TensorView packed_w, packed_b;
    op->prepackWeights({&weight, &bias}, {&packed_w, &packed_b});

    std::vector<char> wbuf(packed_w.nbytes());
    packed_w = TensorView(packed_w.shape_span(), packed_w.data_type(),
                           wbuf.data(), packed_w.pitch(), packed_w.layout());
    std::vector<char> bbuf(packed_b.nbytes());
    packed_b = TensorView(packed_b.shape_span(), packed_b.data_type(),
                           bbuf.data(), packed_b.pitch(), packed_b.layout());
    op->prepackWeights({&weight, &bias}, {&packed_w, &packed_b});

    const TensorView ins[] = {input, packed_w, packed_b};
    op->compute(output, ins, ctx, workspace);
}

}  // namespace nnops
