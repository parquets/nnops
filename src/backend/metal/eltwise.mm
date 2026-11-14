/// @file eltwise_metal.mm
/// @brief Metal binary elementwise kernels (f32/f16).
///
/// Same-shape operands only: the Metal path declines broadcasting and leaves it
/// to the CPU route, which is where the example backend draws the line.
/// Requires ComputeContext::metal_buffers with [A, B, output].

#include "metal_common.hpp"

#include "nnops/ops/eltwise.hpp"

namespace nnops::backend::metal {

namespace {

int32_t eltwise_code(EltwiseType type) {
    switch (type) {
    case EltwiseType::Add: return 0;
    case EltwiseType::Sub: return 1;
    case EltwiseType::Mul: return 2;
    case EltwiseType::Div: return 3;
    case EltwiseType::Min: return 4;
    case EltwiseType::Max: return 5;
    case EltwiseType::Pow: return 6;
    }
    return -1;
}

bool same_shape(const TensorView& a, const TensorView& b) {
    if (a.rank() != b.rank()) { return false; }
    for (int64_t d = 0; d < a.rank(); ++d) {
        if (a.shape(d) != b.shape(d)) { return false; }
    }
    return true;
}

}  // namespace

Status eltwise_metal(const EltwiseAttributes& attrs, TensorView& /*output*/,
                     std::span<const TensorView> inputs, const ComputeContext& ctx,
                     void* /*workspace*/) {
    const auto& a = inputs[0];
    const auto& b = inputs[1];
    if (!same_shape(a, b)) { return Status::unsupported; }

    const int64_t total = a.numel();
    if (total == 0) { return Status::ok; }

    const int32_t code = eltwise_code(attrs.type);
    if (code < 0) { return Status::unsupported; }

    switch (a.data_type()) {
    case DataType::f32:
        return dispatch(KernelFn::EltwiseF32, code, attrs.add_to, ctx, total, 3);
    case DataType::f16:
        return dispatch(KernelFn::EltwiseF16, code, attrs.add_to, ctx, total, 3);
    default:
        return Status::unsupported;
    }
}

}  // namespace nnops::backend::metal
