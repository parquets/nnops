/// @file split.cpp
/// @brief Split operator dispatch.

#include "nnops/ops/split.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

namespace backend::cpu {
Status split_cpu(const SplitAttributes& attrs,
                 std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx,
                 void* workspace);
}

struct Split::Impl {
    using KernelFn = Status (*)(const SplitAttributes&,
                                std::span<TensorView>,
                                std::span<const TensorView>,
                                const ComputeContext&,
                                void*);
    KernelFn kernel_fn = nullptr;
};
Split::~Split() = default;

NNOPS_OP_CREATE(Split, SplitAttributes)

Split::Split(const SplitAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = (backend == Backend::CPU) ? backend::cpu::split_cpu
                                                 : nullptr;
}

std::vector<TensorDesc> Split::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    const auto& in = inputs[0];
    const int64_t n = attrs_.output_count();
    if (n <= 0 || in.rank < 1) { return {}; }
    const int64_t axis = resolve_axis(attrs_.axis, in.rank);
    if (axis < 0 || axis >= in.rank) { return {}; }

    std::vector<TensorDesc> outs;
    outs.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        TensorDesc d = in;
        const int64_t len = attrs_.split.empty()
            ? in.dims[static_cast<size_t>(axis)] / n
            : attrs_.split[static_cast<size_t>(i)];
        d.dims[static_cast<size_t>(axis)] = len;
        outs.push_back(d);
    }
    return outs;
}

Status Split::validate(std::span<const TensorDesc> inputs,
                       std::span<const TensorDesc> outputs) const
{
    const int64_t n = attrs_.output_count();
    if (n <= 0) { return Status::invalid_argument; }
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1, 1, n));

    const auto& in = inputs[0];
    if (!is_layout_supported(in.layout, LayoutSupport::ChannelFirstOnly)) {
        return Status::unsupported;
    }
    if (!detail::dtype_in<DataType::f32, DataType::f16, DataType::s8,
                          DataType::u8, DataType::s32, DataType::s64>(in.dtype)) {
        return Status::unsupported;
    }
    const int64_t axis = resolve_axis(attrs_.axis, in.rank);
    if (axis < 0 || axis >= in.rank) { return Status::invalid_argument; }
    const int64_t dim = in.dims[static_cast<size_t>(axis)];

    if (attrs_.split.empty()) {
        if (dim % n != 0) { return Status::invalid_argument; }
    } else {
        int64_t sum = 0;
        for (int64_t s : attrs_.split) {
            if (s < 0) { return Status::invalid_argument; }
            sum += s;
        }
        if (sum != dim) { return Status::invalid_argument; }
    }

    for (int64_t i = 0; i < n; ++i) {
        const auto& o = outputs[static_cast<size_t>(i)];
        if (o.dtype != in.dtype || o.rank != in.rank) {
            return Status::invalid_argument;
        }
        const int64_t want = attrs_.split.empty()
            ? dim / n : attrs_.split[static_cast<size_t>(i)];
        if (o.dims[static_cast<size_t>(axis)] != want) {
            return Status::invalid_argument;
        }
    }
    return Status::ok;
}

Status Split::compute(std::span<TensorView> outputs,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace)
{
    const int64_t n = attrs_.output_count();
    NNOPS_TRY(detail::check_views(outputs, inputs, 1, 1, static_cast<size_t>(n > 0 ? n : 1)));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs, inputs, ctx, workspace);
}

}  // namespace nnops
