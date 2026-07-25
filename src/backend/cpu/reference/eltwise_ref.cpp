/// @file eltwise_ref.cpp
/// @brief Naive CPU reference implementation of element-wise binary operations.
///
/// Correctness baseline. Uses a straightforward single loop over all elements.
/// Matches the SIMD kernel's output exactly (same dtype dispatch, same switch).

#include "nnops/ops/eltwise.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/simd/simd.hpp"

namespace nnops::backend::cpu::reference {

using nnops::simd::s_load;
using nnops::simd::s_store;

template <typename T>
void eltwise_impl_ref(const EltwiseAttributes& attrs,
                       TensorView& output,
                       std::span<const TensorView> inputs)
{
    const auto& A = inputs[0];
    const auto& B = inputs[1];
    const int64_t total = A.numel();
    if (total == 0) return;
    NNOPS_ASSERT(A.numel() == B.numel());
    NNOPS_ASSERT(output.numel() == total);

    const auto* a_ptr = A.ptr<T>();
    const auto* b_ptr = B.ptr<T>();
    auto*       o_ptr = output.ptr<T>();
    const bool add_to = attrs.add_to;

    switch (attrs.type) {
    case EltwiseType::Add: {
        for (int64_t i = 0; i < total; ++i) {
            float rv = s_load(&a_ptr[i]) + s_load(&b_ptr[i]);
            if (add_to) {
                s_store(&o_ptr[i], s_load(&o_ptr[i]) + rv);
            } else {
                s_store(&o_ptr[i], rv);
            }
        }
        break;
    }
    case EltwiseType::Sub: {
        for (int64_t i = 0; i < total; ++i) {
            float rv = s_load(&a_ptr[i]) - s_load(&b_ptr[i]);
            if (add_to) {
                s_store(&o_ptr[i], s_load(&o_ptr[i]) + rv);
            } else {
                s_store(&o_ptr[i], rv);
            }
        }
        break;
    }
    case EltwiseType::Mul: {
        for (int64_t i = 0; i < total; ++i) {
            float rv = s_load(&a_ptr[i]) * s_load(&b_ptr[i]);
            if (add_to) {
                s_store(&o_ptr[i], s_load(&o_ptr[i]) + rv);
            } else {
                s_store(&o_ptr[i], rv);
            }
        }
        break;
    }
    case EltwiseType::Div: {
        for (int64_t i = 0; i < total; ++i) {
            float rv = s_load(&a_ptr[i]) / s_load(&b_ptr[i]);
            if (add_to) {
                s_store(&o_ptr[i], s_load(&o_ptr[i]) + rv);
            } else {
                s_store(&o_ptr[i], rv);
            }
        }
        break;
    }
    }
}

void eltwise_ref(const EltwiseAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& /*ctx*/,
                   void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        eltwise_impl_ref<float>(attrs, output, inputs);
        return;
    case DataType::f16:
        eltwise_impl_ref<half>(attrs, output, inputs);
        return;
    default:
        NNOPS_ASSERT(!"eltwise_ref: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu::reference
