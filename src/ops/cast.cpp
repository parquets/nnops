/// @file cast.cpp
/// @brief Cast operator dispatch.

#include "nnops/ops/cast.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(CastAttributes, cast_cpu)

NNOPS_OP_BOILERPLATE(Cast, CastAttributes,
        backend::cpu::cast_cpu, nullptr, nullptr)

namespace {

/// Every dtype the CPU cast kernels cover. bf16 rides the f16 slot through
/// detail::dtype_in's per-backend gate.
constexpr Status cast_dtype_ok(Backend be, DataType d)
{
    return detail::dtype_in<DataType::f32, DataType::f16, DataType::s32,
                            DataType::s64, DataType::s8, DataType::u8>(be, d)
        ? Status::ok : Status::unsupported;
}

}  // anonymous namespace

std::vector<TensorDesc> Cast::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    auto out = identity_output_shape(inputs);
    out[0].dtype = attrs_.to;
    return out;
}

Status Cast::validate(std::span<const TensorDesc> inputs,
                      std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1));
    if (outputs[0].dtype != attrs_.to) { return Status::invalid_argument; }
    NNOPS_TRY(cast_dtype_ok(backend_, inputs[0].dtype));
    NNOPS_TRY(cast_dtype_ok(backend_, attrs_.to));

    // Shape is preserved element for element; a differing numel is the caller's
    // mistake, not something to truncate.
    if (inputs[0].numel() != outputs[0].numel()) {
        return Status::invalid_argument;
    }
    return Status::ok;
}

Status Cast::compute(std::span<TensorView> outputs,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace)
{
    NNOPS_TRY(detail::check_views(backend_, outputs, inputs, 1, 1, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (outputs[0].data_type() != attrs_.to) { return Status::invalid_argument; }
    NNOPS_TRY(cast_dtype_ok(backend_, inputs[0].data_type()));
    NNOPS_TRY(cast_dtype_ok(backend_, outputs[0].data_type()));
    if (inputs[0].numel() != outputs[0].numel()) { return Status::invalid_argument; }
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
