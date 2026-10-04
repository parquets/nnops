/// @file permute.cpp
/// @brief SIMD-optimized CPU implementation of dimension permute (transpose).
///
/// For planar data, permute is a memory rearrangement where each output
/// element comes from a permuted position in the input.
///
/// Optimization tiers:
///   1. Last-2-dims swap (rank>=2, only last 2 dims transposed):
///      tiled 8×8 SIMD transpose via v_transpose_8x8, batched over
///      all leading dimensions.
///      Covers: 2D transpose, batched matrix transpose (B,M,K)→(B,K,M),
///      NCHW↔NHCW, multi-head dim reordering, etc.
///   2. Inner dim contiguous (in_step==1): memcpy per output row.
///   3. General n-D: scalar strided copy, parallel over outer dim.

#include "nnops/ops/permute.hpp"
#include "simd_kernel/simd_permute.hpp"
#include "common/dtype_dispatch.hpp"

#include <cstring>

namespace nnops::backend::cpu {

template <typename T>
void permute_last_two_swap_impl(TensorView& output,
                                std::span<const TensorView> inputs,
                                const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();

    // Batch = product of all leading dims
    int64_t batch = 1;
    for (int64_t i = 0; i < rank - 2; ++i) {
        batch *= input.shape(i);
    }

    const int64_t M = input.shape(rank - 2);  // rows per slice
    const int64_t K = input.shape(rank - 1);  // cols per slice
    const int64_t slice_elems = M * K;        // elements per 2D slice

    const T* in_ptr  = input.ptr<T>();
    T*       out_ptr = output.ptr<T>();

    // Leading stride = stride to advance by 1 in the outermost batch dim.
    // For planar data, this equals slice_elems (since dims rank-2..rank-1
    // are the innermost).
    // But we use shape multiplication for correctness: the stride from
    // batch element b to b+1 is exactly M * K for planar data.
    const int64_t in_batch_stride = slice_elems;
    const int64_t out_batch_stride = slice_elems;  // same element count

    // Within each slice: input [M, K] row-major → output [K, M] row-major
    // in_ld = K (row stride in elements), out_ld = M

    auto body = [&](int64_t b) {
        kernel::tiled_transpose_2d<T>(
            in_ptr  + b * in_batch_stride,
            out_ptr + b * out_batch_stride,
            M, K,
            K,   // in_ld: elements between input rows
            M);  // out_ld: elements between output rows
    };

    ctx.cpu.run(0, batch, body);
}

template <typename T>
void permute_impl(const PermuteAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();
    const auto& perm = attrs.perm;

    if (rank == 1) {
        // 1D permute is identity
        std::memcpy(output.ptr<void>(), input.ptr<void>(),
                    static_cast<size_t>(input.numel()) * sizeof(T));
        return;
    }

    const int64_t total = output.numel();
    if (total == 0) { return; }

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
    const int64_t last_perm_dim = perm[static_cast<size_t>(rank - 1)];
    const int64_t in_inner_step = in_strides[static_cast<size_t>(last_perm_dim)];

    auto body = [&](int64_t outer_idx) {
        // Decompose outer_idx into multi-index for dims 0..rank-2
        // Uses outer_strides (which exclude the inner dim factor)
        int64_t tmp = outer_idx;
        int64_t in_base = 0;
        for (int64_t d = 0; d < rank - 1; ++d) {
            int64_t coord = tmp / outer_strides[d];
            tmp %= outer_strides[d];
            in_base += coord * in_strides[perm[static_cast<size_t>(d)]];
        }

        T* out_row = out_ptr + outer_idx * inner_dim;

        if (in_inner_step == 1) {
            // Contiguous in both input and output: memcpy
            std::memcpy(out_row, in_ptr + in_base,
                        static_cast<size_t>(inner_dim) * sizeof(T));
        } else {
            for (int64_t i = 0; i < inner_dim; ++i) {
                out_row[i] = in_ptr[in_base + i * in_inner_step];
            }
        }
    };

    ctx.cpu.run(0, outer_total, body);
}

void permute_cpu(const PermuteAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx,
                 void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    const int64_t rank = inputs[0].rank();

    // Detect "last two dims swap": perm[i]==i for i < rank-2,
    // perm[rank-2]==rank-1, perm[rank-1]==rank-2.
    // This covers pure 2D transpose (rank==2) and batched 2D transpose
    // (rank>2, e.g. [B,M,K]→[B,K,M], NCHW→NHCW).
    bool is_last_two_swap = (rank >= 2);
    if (is_last_two_swap) {
        for (int64_t i = 0; i < rank - 2; ++i) {
            if (attrs.perm[static_cast<size_t>(i)] != i) {
                is_last_two_swap = false;
                break;
            }
        }
        if (is_last_two_swap) {
            is_last_two_swap =
                (attrs.perm[static_cast<size_t>(rank - 2)] == rank - 1 &&
                 attrs.perm[static_cast<size_t>(rank - 1)] == rank - 2);
        }
    }

    if (is_last_two_swap) {
        dispatch_f32_f16(dtype, "permute_cpu", [&](auto tag) {
            using T = typename decltype(tag)::type;
            permute_last_two_swap_impl<T>(output, inputs, ctx);
        });
        return;
    }

    // General path
    dispatch_f32_f16(dtype, "permute_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        permute_impl<T>(attrs, output, inputs, ctx);
    });
}

}  // namespace nnops::backend::cpu
