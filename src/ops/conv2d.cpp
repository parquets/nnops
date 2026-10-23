/// @file conv2d.cpp
/// @brief Conv2D operator dispatch — connects class API to backend implementations.

#include "nnops/ops/conv2d.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "backend/cpu/conv2d_im2col.h"
#include "backend/cpu/conv2d_winograd.h"
#include "backend/cpu/simd_kernel/simd_winograd.hpp"

namespace nnops {

struct Conv2D::Impl {
    using KernelFn = void (*)(const Conv2DAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};
Conv2D::~Conv2D() = default;


namespace {
auto resolve_conv2d_kernel(Backend backend, ConvAlgorithm algorithm)
    -> Conv2D::Impl::KernelFn
{
    if (algorithm == ConvAlgorithm::Winograd && backend == Backend::CPU) {
        return backend::cpu::conv2d_winograd_kernel;
    }
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::conv2d_im2col_kernel;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return nullptr;
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return nullptr;
#endif
    default:
        return nullptr;
    }
}

/// Pack the raw [OC, IC/G, 3, 3] weight into U[16][OC][IC/G] = G g Gᵀ.
/// @p ld_u / @p ld_c are the row / column strides of one 4×4 transform block
/// inside the U buffer (p = i·4 + j maps to p·OC·ICG).
template <class T>
void wino_pack_weight(const TensorView& weight, TensorView& u_out,
                      int64_t OC, int64_t ICG, int64_t ld_u, int64_t ld_c)
{
    const T* g = weight.ptr<T>();
    T* u = u_out.ptr<T>();
    for (int64_t oc = 0; oc < OC; ++oc) {
        for (int64_t ic = 0; ic < ICG; ++ic) {
            nnops::kernel::winograd_weight_transform(
                g + (oc * ICG + ic) * 9, 3, u + oc * ICG + ic,
                static_cast<int>(ld_u), static_cast<int>(ld_c));
        }
    }
}
}  // anonymous namespace

std::unique_ptr<Conv2D> Conv2D::create(const Conv2DAttributes& attrs,
                                        Backend backend)
{
    return std::unique_ptr<Conv2D>(new Conv2D(attrs, backend));
}

Conv2D::Conv2D(const Conv2DAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_conv2d_kernel(backend, attrs_.algorithm);
}

std::vector<TensorDesc> Conv2D::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    std::vector<TensorDesc> outs = conv2d_output_shape(
        attrs_.kernel_size, attrs_.stride, attrs_.dilation, attrs_.padding,
        static_cast<int>(attrs_.auto_pad), attrs_.groups, inputs);

    // The quantized route writes fp output. A fp activation already carries the
    // output dtype (shape inference copies the input dtype); a pre-quantized
    // activation (u8/s8) leaves it to attributes.output_dtype.
    if (inputs.size() > 1 && is_quantized_conv2d(inputs[1].dtype)) {
        const DataType act = inputs[0].dtype;
        const bool fp_act = (act == DataType::f32 || act == DataType::f16);
        outs[0].dtype = fp_act ? act : attrs_.output_dtype;
    }
    return outs;
}

size_t Conv2D::getWorkspaceSize(std::span<const TensorDesc> inputs,
                                std::span<const TensorDesc> outputs) const
{
    (void)inputs;
    (void)outputs;
    // im2col scratch is pooled internally by the CPU kernel.
    return 0;
}

void Conv2D::compute(std::span<TensorView> outputs,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace)
{
    NNOPS_ASSERT(inputs.size() >= 2);
    NNOPS_ASSERT(inputs.size() <= 3);  // input, weight [, bias]
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());
    NNOPS_ASSERT(!inputs[1].is_empty());
    NNOPS_ASSERT(is_layout_supported(inputs[0].layout(), LayoutSupport::PlanarOnly));

    // Winograd has no integer counterpart, so the quantized route always runs
    // the im2col kernel — including for the k3s1 geometry Winograd would take.
    if (backend_ == Backend::CPU && is_quantized_conv2d(inputs[1].data_type())) {
        backend::cpu::conv2d_im2col_kernel(attrs_, output, inputs, ctx, workspace);
        return;
    }

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

void Conv2D::prepackWeights(std::span<const TensorView> inputs,
                            std::span<TensorView> outputs,
                            const ComputeContext& ctx)
{
    (void)ctx;

    // compute() receives U and cannot fall back to the raw-weight im2col path,
    // so every unsupported configuration has to fail here, at prepack time.
    NNOPS_ASSERT(attrs_.algorithm == ConvAlgorithm::Winograd &&
                 "Conv2D::prepackWeights: attrs.algorithm must be Winograd");
    NNOPS_ASSERT(attrs_.kernel_size[0] == 3 && attrs_.kernel_size[1] == 3 &&
                 attrs_.stride[0] == 1 && attrs_.stride[1] == 1 &&
                 attrs_.dilation[0] == 1 && attrs_.dilation[1] == 1 &&
                 "Conv2D::prepackWeights: Winograd requires k3s1");

    NNOPS_ASSERT(inputs.size() >= 1);
    NNOPS_ASSERT(outputs.size() >= 1);
    const auto& weight = inputs[0];
    NNOPS_ASSERT(weight.rank() == 4);
    NNOPS_ASSERT(is_layout_supported(weight.layout(), LayoutSupport::PlanarOnly));

    // Weight: [OC, IC/G, 3, 3]
    const int64_t OC  = weight.shape(0);
    const int64_t ICG = weight.shape(1);
    const DataType dtype = weight.data_type();
    const int64_t elem_size = static_cast<int64_t>(data_type_size(dtype));

    if (outputs[0].is_empty()) {
        // ---- Query: U is [16, OC, IC/G] — one 4×4 transform-domain kernel per
        // position. The pitch is explicit: PackedWinoWeight is not
        // channel-packed, so TensorDesc::row_pitch() would align the row to
        // 32 B, which is not ICG × elem_size in general.
        const int64_t u_shape[] = {16, OC, ICG};
        const int64_t u_pitch = ICG * elem_size;
        outputs[0] = TensorView(std::span<const int64_t>(u_shape, 3), dtype,
                                nullptr, u_pitch, TensorLayout::PackedWinoWeight);
        return;
    }

    // ---- Pack: U[p][oc][ic] = (G g Gᵀ)[p], row stride between p planes.
    NNOPS_ASSERT(outputs[0].shape(0) == 16 && outputs[0].shape(1) == OC &&
                 outputs[0].shape(2) == ICG);
    const int64_t plane = OC * ICG;  // elements between transform positions
    const int64_t ld_c  = plane;     // column (j) step of the 4×4 block
    const int64_t ld_u  = 4 * plane; // row (i) step: p = i·4 + j

    if (dtype == DataType::f32) {
        wino_pack_weight<float>(weight, outputs[0], OC, ICG, ld_u, ld_c);
    } else {
        NNOPS_ASSERT(dtype == DataType::f16);
        wino_pack_weight<backend::cpu::half>(weight, outputs[0], OC, ICG, ld_u, ld_c);
    }
}

}  // namespace nnops
