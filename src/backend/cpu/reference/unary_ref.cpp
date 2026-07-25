/// @file unary_ref.cpp
/// @brief Naive CPU reference implementation of element-wise unary operations.
///
/// Correctness baseline. Uses a straightforward single loop over all elements.
/// Matches the SIMD kernel's output exactly (same dtype dispatch, same switch).

#include "nnops/ops/unary.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <cmath>

namespace nnops::backend::cpu::reference {

using nnops::simd::s_load;
using nnops::simd::s_store;

template <typename T>
void unary_impl_ref(const UnaryAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs)
{
    const auto& input = inputs[0];
    const int64_t total = input.numel();
    if (total == 0) return;
    NNOPS_ASSERT(output.numel() == total);

    const auto* in_ptr  = input.ptr<T>();
    auto*       out_ptr = output.ptr<T>();
    const bool add_to = attrs.add_to;

    switch (attrs.type) {

    case UnaryType::Exp: {
        for (int64_t i = 0; i < total; ++i) {
            float rv = std::exp(s_load(&in_ptr[i]));
            if (add_to) {
                s_store(&out_ptr[i], s_load(&out_ptr[i]) + rv);
            } else {
                s_store(&out_ptr[i], rv);
            }
        }
        break;
    }

    case UnaryType::Log: {
        for (int64_t i = 0; i < total; ++i) {
            float rv = std::log(s_load(&in_ptr[i]));
            if (add_to) {
                s_store(&out_ptr[i], s_load(&out_ptr[i]) + rv);
            } else {
                s_store(&out_ptr[i], rv);
            }
        }
        break;
    }

    case UnaryType::Sin: {
        for (int64_t i = 0; i < total; ++i) {
            float rv = std::sin(s_load(&in_ptr[i]));
            if (add_to) {
                s_store(&out_ptr[i], s_load(&out_ptr[i]) + rv);
            } else {
                s_store(&out_ptr[i], rv);
            }
        }
        break;
    }

    case UnaryType::Cos: {
        for (int64_t i = 0; i < total; ++i) {
            float rv = std::cos(s_load(&in_ptr[i]));
            if (add_to) {
                s_store(&out_ptr[i], s_load(&out_ptr[i]) + rv);
            } else {
                s_store(&out_ptr[i], rv);
            }
        }
        break;
    }

    case UnaryType::Tan: {
        for (int64_t i = 0; i < total; ++i) {
            float rv = std::tan(s_load(&in_ptr[i]));
            if (add_to) {
                s_store(&out_ptr[i], s_load(&out_ptr[i]) + rv);
            } else {
                s_store(&out_ptr[i], rv);
            }
        }
        break;
    }

    case UnaryType::Tanh: {
        for (int64_t i = 0; i < total; ++i) {
            float rv = std::tanh(s_load(&in_ptr[i]));
            if (add_to) {
                s_store(&out_ptr[i], s_load(&out_ptr[i]) + rv);
            } else {
                s_store(&out_ptr[i], rv);
            }
        }
        break;
    }

    case UnaryType::Abs: {
        for (int64_t i = 0; i < total; ++i) {
            float v = s_load(&in_ptr[i]);
            float rv = (v < 0.0f ? -v : v);
            if (add_to) {
                s_store(&out_ptr[i], s_load(&out_ptr[i]) + rv);
            } else {
                s_store(&out_ptr[i], rv);
            }
        }
        break;
    }

    case UnaryType::Neg: {
        for (int64_t i = 0; i < total; ++i) {
            float rv = -s_load(&in_ptr[i]);
            if (add_to) {
                s_store(&out_ptr[i], s_load(&out_ptr[i]) + rv);
            } else {
                s_store(&out_ptr[i], rv);
            }
        }
        break;
    }

    case UnaryType::Sqrt: {
        for (int64_t i = 0; i < total; ++i) {
            float rv = std::sqrt(s_load(&in_ptr[i]));
            if (add_to) {
                s_store(&out_ptr[i], s_load(&out_ptr[i]) + rv);
            } else {
                s_store(&out_ptr[i], rv);
            }
        }
        break;
    }

    }  // switch
}

void unary_ref(const UnaryAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& /*ctx*/,
                 void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        unary_impl_ref<float>(attrs, output, inputs);
        return;
    case DataType::f16:
        unary_impl_ref<half>(attrs, output, inputs);
        return;
    default:
        NNOPS_ASSERT(!"unary_ref: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu::reference
