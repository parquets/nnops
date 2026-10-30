/// @file permute.cpp
/// @brief SIMD-optimized CPU implementation of dimension permute (transpose).
///
/// Planar data: each output element comes from a permuted input position. Two
/// routes, picked from `perm`:
///   1. Last-two-dims swap (perm[i]==i for i<rank-2, and the last two
///      exchanged): tiled 8x8 SIMD transpose via kernel::tiled_transpose_2d,
///      batched over all leading dims. Covers 2D transpose and
///      batched (B,M,K)->(B,K,M).
///   2. General n-D: scalar strided copy, parallel over the outer index; a row
///      whose innermost output dim maps to input stride 1 is a memcpy.
///
/// The batch stride between slices is M*K: planar data makes dims rank-2 and
/// rank-1 the innermost and contiguous.

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

    int64_t batch = 1;
    for (int64_t i = 0; i < rank - 2; ++i) {
        batch *= input.shape(i);
    }

    const int64_t M = input.shape(rank - 2);
    const int64_t K = input.shape(rank - 1);
    const int64_t slice_elems = M * K;

    const T* in_ptr  = input.ptr<T>();
    T*       out_ptr = output.ptr<T>();

    const int64_t in_batch_stride = slice_elems;
    const int64_t out_batch_stride = slice_elems;

    auto body = [&](int64_t b) {
        kernel::tiled_transpose_2d<T>(
            in_ptr  + b * in_batch_stride,
            out_ptr + b * out_batch_stride,
            M, K,
            K,
            M);
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
        std::memcpy(output.ptr<void>(), input.ptr<void>(),
                    static_cast<size_t>(input.numel()) * sizeof(T));
        return;
    }

    const int64_t total = output.numel();
    if (total == 0) { return; }

    int64_t in_strides[TensorDesc::kMaxRank];
    in_strides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * input.shape(i + 1);
    }

    // Strides over dims 0..rank-2 only: outer_idx has the inner dim divided out.
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

    const int64_t last_perm_dim = perm[static_cast<size_t>(rank - 1)];
    const int64_t in_inner_step = in_strides[static_cast<size_t>(last_perm_dim)];

    auto body = [&](int64_t outer_idx) {
        int64_t tmp = outer_idx;
        int64_t in_base = 0;
        for (int64_t d = 0; d < rank - 1; ++d) {
            int64_t coord = tmp / outer_strides[d];
            tmp %= outer_strides[d];
            in_base += coord * in_strides[perm[static_cast<size_t>(d)]];
        }

        T* out_row = out_ptr + outer_idx * inner_dim;

        if (in_inner_step == 1) {
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

Status permute_cpu(const PermuteAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx,
                 void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    const int64_t rank = inputs[0].rank();

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
        return Status::ok;
    }

    dispatch_f32_f16(dtype, "permute_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        permute_impl<T>(attrs, output, inputs, ctx);
    });
    return Status::ok;
}

}  // namespace nnops::backend::cpu
