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
    if (total == 0) { return; }
    NNOPS_ASSERT(A.numel() == B.numel());
    NNOPS_ASSERT(output.numel() == total);

    const auto* a_ptr = A.ptr<T>();
    const auto* b_ptr = B.ptr<T>();
    auto*       o_ptr = output.ptr<T>();
    const bool add_to = attrs.add_to;

    // Row-by-row dimensions (for rank >= 2, respects pitch / packed layouts)
    const int64_t rank = A.rank();
    const bool use_flat = (rank < 2);
    const int64_t num_rows = use_flat ? total : A.total_rows();
    const int64_t last_dim = use_flat ? total : A.shape(rank - 1) * A.channel_pack_size();
    const int64_t a_rs = use_flat ? 1 : A.row_stride_elems();
    const int64_t b_rs = use_flat ? 1 : B.row_stride_elems();
    const int64_t o_rs = use_flat ? 1 : output.row_stride_elems();

    switch (attrs.type) {
    case EltwiseType::Add: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* a_row = a_ptr + r * a_rs;
            const T* b_row = b_ptr + r * b_rs;
            T* o_row = o_ptr + r * o_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float rv = s_load(&a_row[i]) + s_load(&b_row[i]);
                if (add_to) {
                    s_store(&o_row[i], s_load(&o_row[i]) + rv);
                } else {
                    s_store(&o_row[i], rv);
                }
            }
        }
        break;
    }
    case EltwiseType::Sub: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* a_row = a_ptr + r * a_rs;
            const T* b_row = b_ptr + r * b_rs;
            T* o_row = o_ptr + r * o_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float rv = s_load(&a_row[i]) - s_load(&b_row[i]);
                if (add_to) {
                    s_store(&o_row[i], s_load(&o_row[i]) + rv);
                } else {
                    s_store(&o_row[i], rv);
                }
            }
        }
        break;
    }
    case EltwiseType::Mul: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* a_row = a_ptr + r * a_rs;
            const T* b_row = b_ptr + r * b_rs;
            T* o_row = o_ptr + r * o_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float rv = s_load(&a_row[i]) * s_load(&b_row[i]);
                if (add_to) {
                    s_store(&o_row[i], s_load(&o_row[i]) + rv);
                } else {
                    s_store(&o_row[i], rv);
                }
            }
        }
        break;
    }
    case EltwiseType::Div: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* a_row = a_ptr + r * a_rs;
            const T* b_row = b_ptr + r * b_rs;
            T* o_row = o_ptr + r * o_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float rv = s_load(&a_row[i]) / s_load(&b_row[i]);
                if (add_to) {
                    s_store(&o_row[i], s_load(&o_row[i]) + rv);
                } else {
                    s_store(&o_row[i], rv);
                }
            }
        }
        break;
    }
    case EltwiseType::Min: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* a_row = a_ptr + r * a_rs;
            const T* b_row = b_ptr + r * b_rs;
            T* o_row = o_ptr + r * o_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float av = s_load(&a_row[i]);
                float bv = s_load(&b_row[i]);
                float rv = av < bv ? av : bv;
                if (add_to) {
                    s_store(&o_row[i], s_load(&o_row[i]) + rv);
                } else {
                    s_store(&o_row[i], rv);
                }
            }
        }
        break;
    }
    case EltwiseType::Max: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* a_row = a_ptr + r * a_rs;
            const T* b_row = b_ptr + r * b_rs;
            T* o_row = o_ptr + r * o_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float av = s_load(&a_row[i]);
                float bv = s_load(&b_row[i]);
                float rv = av > bv ? av : bv;
                if (add_to) {
                    s_store(&o_row[i], s_load(&o_row[i]) + rv);
                } else {
                    s_store(&o_row[i], rv);
                }
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
