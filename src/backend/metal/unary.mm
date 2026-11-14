/// @file unary_metal.mm
/// @brief Metal unary kernels (f32/f16).
///
/// The op code handed to MSL mirrors UnaryType's declaration order, which is
/// kept explicit here rather than cast so a reordering of the enum is a compile
/// error instead of a quiet mis-dispatch. Requires ComputeContext::metal_buffers
/// with [input, output].

#include "metal_common.hpp"

#include "nnops/detail/assert.hpp"
#include "nnops/ops/unary.hpp"

namespace nnops::backend::metal {

namespace {

int32_t unary_code(UnaryType type) {
    switch (type) {
    case UnaryType::Exp:   return 0;
    case UnaryType::Log:   return 1;
    case UnaryType::Sin:   return 2;
    case UnaryType::Cos:   return 3;
    case UnaryType::Tan:   return 4;
    case UnaryType::Tanh:  return 5;
    case UnaryType::Abs:   return 6;
    case UnaryType::Neg:   return 7;
    case UnaryType::Sqrt:  return 8;
    case UnaryType::Erf:   return -1;  // MSL has no erf/erfc builtin
    case UnaryType::Round: return 10;
    case UnaryType::Ceil:  return 11;
    case UnaryType::Floor: return 12;
    case UnaryType::Recip: return 13;
    case UnaryType::Sign:  return 14;
    case UnaryType::Exp2:  return 15;
    case UnaryType::Log2:  return 16;
    case UnaryType::Log10: return 17;
    case UnaryType::Asin:  return 18;
    case UnaryType::Acos:  return 19;
    case UnaryType::Atan:  return 20;
    case UnaryType::Sinh:  return 21;
    case UnaryType::Cosh:  return 22;
    case UnaryType::Asinh: return 23;
    case UnaryType::Acosh: return 24;
    case UnaryType::Atanh: return 25;
    case UnaryType::Erfc:  return -1;
    }
    return -1;
}

}  // namespace

Status unary_metal(const UnaryAttributes& attrs, TensorView& /*output*/,
                   std::span<const TensorView> inputs, const ComputeContext& ctx,
                   void* /*workspace*/) {
    const auto& input = inputs[0];
    const int64_t total = input.numel();
    if (total == 0) { return Status::ok; }

    const int32_t code = unary_code(attrs.type);
    if (code < 0) { return Status::unsupported; }

    switch (input.data_type()) {
    case DataType::f32:
        return dispatch(KernelFn::UnaryF32, code, attrs.add_to, ctx, total, 2);
    case DataType::f16:
        return dispatch(KernelFn::UnaryF16, code, attrs.add_to, ctx, total, 2);
    default:
        return Status::unsupported;
    }
}

}  // namespace nnops::backend::metal
