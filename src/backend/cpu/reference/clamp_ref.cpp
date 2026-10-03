/// @file clamp_ref.cpp
/// @brief Scalar CPU reference implementation of element-wise Clamp.
///
/// Correctness baseline. Processes element-by-element using s_load/s_store
/// for transparent half-to-float conversion.

#include "nnops/ops/clamp.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <algorithm>

namespace nnops::backend::cpu::reference {

using nnops::simd::s_load;
using nnops::simd::s_store;

template <typename T>
void clamp_impl_ref(const ClampAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs)
{
    const auto& input = inputs[0];
    const int64_t total = input.numel();
    if (total == 0) { return; }
    NNOPS_ASSERT(output.numel() == total);

    const auto* in_ptr  = input.ptr<T>();
    auto*       out_ptr = output.ptr<T>();
    const bool add_to = attrs.add_to;
    const float min_val = attrs.min_val;
    const float max_val = attrs.max_val;

    // Row-by-row dimensions (pitch-aware, supports packed layouts)
    const int64_t rank = input.rank();
    const bool use_flat = (rank < 2);
    const int64_t num_rows = use_flat ? total : input.total_rows();
    const int64_t last_dim = use_flat ? total : input.shape(rank - 1) * input.channel_pack_size();
    const int64_t in_rs = use_flat ? 1 : input.row_stride_elems();
    const int64_t out_rs = use_flat ? 1 : output.row_stride_elems();

    for (int64_t r = 0; r < num_rows; ++r) {
        const T* in_row = in_ptr + r * in_rs;
        T* out_row = out_ptr + r * out_rs;
        for (int64_t i = 0; i < last_dim; ++i) {
            float v = s_load(&in_row[i]);
            float rv = std::max(min_val, std::min(max_val, v));
            if (add_to) {
                s_store(&out_row[i], s_load(&out_row[i]) + rv);
            } else {
                s_store(&out_row[i], rv);
            }
        }
    }
}

void clamp_ref(const ClampAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& /*ctx*/,
                 void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        clamp_impl_ref<float>(attrs, output, inputs);
        return;
    case DataType::f16:
        clamp_impl_ref<half>(attrs, output, inputs);
        return;
    default:
        NNOPS_ASSERT(!"clamp_ref: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu::reference
