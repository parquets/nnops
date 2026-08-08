/// @file slice.cpp
/// @brief SIMD-optimized CPU implementation of Slice.
///
/// For planar data, slice is a strided sub-tensor extraction.
///
/// Optimization tiers:
///   1. Inner-dim contiguous (innermost output dim maps to step-1 input):
///      memcpy per output row — optimal.
///   2. General n-D: scalar strided copy, parallel over outer dims.
///
/// Both paths use cpu_parallel_for over outer-row index.

#include "nnops/ops/slice.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"

#include <algorithm>
#include <cstring>

namespace nnops::backend::cpu {

namespace {

template <typename T>
void slice_impl(const SliceAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();
    const int64_t total = output.numel();
    if (total == 0) {
        return;
    }

    const size_t n_axes = attrs.axes.size();

    // Resolve effective start/step per axis
    int64_t eff_start[TensorDesc::kMaxRank];
    int64_t eff_step[TensorDesc::kMaxRank];
    for (int64_t d = 0; d < rank; ++d) {
        eff_start[d] = 0;
        eff_step[d]  = 1;
    }
    for (size_t a = 0; a < n_axes; ++a) {
        int64_t ax = attrs.axes[a];
        if (ax < 0) {
            ax += rank;
        }
        eff_start[static_cast<size_t>(ax)] = attrs.starts[a];
        eff_step[static_cast<size_t>(ax)]  =
            (a < attrs.steps.size()) ? attrs.steps[a] : int64_t(1);
    }

    // Input strides in elements (planar → contiguous innermost)
    int64_t in_strides[TensorDesc::kMaxRank];
    in_strides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * input.shape(i + 1);
    }

    // Outer-only strides (for decomposing outer_idx over dims 0..rank-2)
    int64_t outer_strides[TensorDesc::kMaxRank];
    if (rank >= 2) {
        outer_strides[rank - 2] = 1;
        for (int64_t i = rank - 3; i >= 0; --i) {
            outer_strides[i] = outer_strides[i + 1] * output.shape(i + 1);
        }
    }

    const T* in_ptr  = input.ptr<T>();
    T*       out_ptr = output.ptr<T>();

    const int64_t inner_dim = output.shape(rank - 1);
    const int64_t outer_total = total / inner_dim;

    // Input stride delta when innermost output dim increments by 1
    const int64_t in_inner_step = eff_step[static_cast<size_t>(rank - 1)];

    auto body = [&](int64_t outer_idx) {
        // Decompose outer_idx into multi-index for dims 0..rank-2
        int64_t tmp = outer_idx;
        int64_t in_base = 0;
        for (int64_t d = 0; d < rank - 1; ++d) {
            int64_t coord = tmp / outer_strides[d];
            tmp %= outer_strides[d];
            int64_t in_coord = eff_start[static_cast<size_t>(d)]
                             + coord * eff_step[static_cast<size_t>(d)];
            in_base += in_coord * in_strides[d];
        }

        // Add innermost start offset
        in_base += eff_start[static_cast<size_t>(rank - 1)];

        T* out_row = out_ptr + outer_idx * inner_dim;

        if (in_inner_step == 1) {
            // Contiguous in both input and output: memcpy
            std::memcpy(out_row, in_ptr + in_base,
                        static_cast<size_t>(inner_dim) * sizeof(T));
        } else {
            // Strided access in input
            for (int64_t i = 0; i < inner_dim; ++i) {
                out_row[i] = in_ptr[in_base + i * in_inner_step];
            }
        }
    };

    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, outer_total, body);
    }
    else {
        for (int64_t o = 0; o < outer_total; ++o) {
            body(o);
        }
    }
}

}  // anonymous namespace

void slice_cpu(const SliceAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        slice_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        slice_impl<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"slice_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
