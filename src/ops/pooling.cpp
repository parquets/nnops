/// @file pooling.cpp
/// @brief Pooling operator dispatch with NCHW→NCHWC8 auto-conversion.

#include "nnops/ops/pooling.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "nnops/core/tensor_layout.hpp"
#include "backend/cpu/layout_convert.hpp"

#include <vector>

namespace nnops {

namespace backend::cpu::reference {
    void pooling_ref(const PoolingAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);
}

namespace backend::cpu {
    void pooling_cpu(const PoolingAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void pooling_cuda(const PoolingAttributes& attrs,
                       TensorView& output,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace);
}
#endif

// ============================================================
// Impl
// ============================================================
struct Pooling::Impl {
    using KernelFn = void (*)(const PoolingAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_pooling_kernel(Backend backend) -> Pooling::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::pooling_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::pooling_cuda;
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return nullptr;
#endif
    }
    return nullptr;
}
}  // anonymous namespace

std::unique_ptr<Pooling> Pooling::create(const PoolingAttributes& attrs,
                                          Backend backend)
{
    return std::unique_ptr<Pooling>(new Pooling(attrs, backend));
}

Pooling::Pooling(const PoolingAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_pooling_kernel(backend);
}

std::vector<TensorDesc> Pooling::getOutputShapes(
    std::span<const TensorDesc> inputs) const
{
    return {pooling_output_shape(
        attrs_.kernel_shape, attrs_.stride, attrs_.dilation, attrs_.padding,
        static_cast<int>(attrs_.auto_pad), inputs)};
}

// ============================================================
// Helper: compute aligned pitch for NCHWC8 tensor
// ============================================================
namespace {

inline int64_t nchwc8_pitch_bytes(DataType dtype, int64_t W) {
    const int64_t row_bytes = W * 8 * static_cast<int64_t>(data_type_size(dtype));
    return ((row_bytes + 31) / 32) * 32;
}

/// Pack an NCHW/NCDHW input to NCHWC8/NCDHWC8 in temp buffer.
/// Returns a span holding the temp TensorView (stored in the caller's temp_in).
void auto_pack_input(const TensorView& input,
                     std::vector<char>& temp_in,
                     TensorView& packed_in,
                     const ComputeContext& ctx)
{
    const int64_t W = input.shape(input.rank() - 1);
    const int64_t pitch = nchwc8_pitch_bytes(input.data_type(), W);
    TensorDesc desc = input.desc();
    size_t storage = nchwc8_storage_bytes(desc);
    temp_in.resize(storage);

    TensorLayout packed_layout = (input.rank() == 5)
        ? TensorLayout::NCDHWC8 : TensorLayout::NCHWC8;

    packed_in = TensorView(input.shape_span(), input.data_type(),
                           temp_in.data(), pitch, packed_layout);

    if (input.rank() == 5) {
        pack_ncdhw_to_ncdhwc8(input, packed_in, ctx);
    } else {
        pack_nchw_to_nchwc8(input, packed_in, ctx);
    }
}

/// Unpack NCHWC8/NCDHWC8 output to NCHW/NCDHW.
void auto_unpack_output(const TensorView& packed_out,
                        TensorView& output,
                        const ComputeContext& ctx)
{
    if (output.rank() == 5) {
        unpack_ncdhwc8_to_ncdhw(packed_out, output, ctx);
    } else {
        unpack_nchwc8_to_nchw(packed_out, output, ctx);
    }
}

}  // anonymous namespace

void Pooling::compute(std::span<TensorView> outputs,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace)
{
    (void) workspace;
    NNOPS_ASSERT(inputs.size() == 1);
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    auto& input = inputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!input.is_empty());

    const bool in_nchw  = !is_channel_packed(input.layout());
    const bool out_nchw = !is_channel_packed(output.layout());

    if (!in_nchw && !out_nchw) {
        // Fast path: NCHWC8 → NCHWC8, zero overhead
        impl_->kernel_fn(attrs_, output, inputs, ctx, nullptr);
        return;
    }

    // Auto-conversion path: pack input and/or unpack output
    std::vector<char> temp_in, temp_out;
    TensorView packed_in, packed_out;

    // Effective input: pack if NCHW
    std::span<const TensorView> eff_inputs = inputs;
    if (in_nchw) {
        auto_pack_input(input, temp_in, packed_in, ctx);
        eff_inputs = {&packed_in, 1};
    }

    // Effective output: allocate NCHWC8 temp if output is NCHW
    TensorView* eff_output = &output;
    if (out_nchw) {
        const int64_t W = output.shape(output.rank() - 1);
        const int64_t pitch = nchwc8_pitch_bytes(output.data_type(), W);
        TensorDesc desc = output.desc();
        temp_out.resize(nchwc8_storage_bytes(desc));

        TensorLayout packed_layout = (output.rank() == 5)
            ? TensorLayout::NCDHWC8 : TensorLayout::NCHWC8;

        packed_out = TensorView(output.shape_span(), output.data_type(),
                                temp_out.data(), pitch, packed_layout);

        // For add_to: preserve existing output values by packing them into the temp
        if (attrs_.add_to) {
            if (output.rank() == 5) {
                pack_ncdhw_to_ncdhwc8(output, packed_out, ctx);
            } else {
                pack_nchw_to_nchwc8(output, packed_out, ctx);
            }
        }

        eff_output = &packed_out;
    }

    impl_->kernel_fn(attrs_, *eff_output, eff_inputs, ctx, nullptr);

    // Unpack output if needed
    if (out_nchw) {
        auto_unpack_output(packed_out, output, ctx);
    }
}

// Functional API
void pooling(const TensorView& input,
             TensorView& output,
             const PoolingAttributes& attrs,
             const ComputeContext& ctx)
{
    auto op = Pooling::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {input};
    op->compute(output, ins, ctx, nullptr);
}

}  // namespace nnops
