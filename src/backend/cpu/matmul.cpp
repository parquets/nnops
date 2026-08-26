/// @file matmul.cpp
/// @brief Tiled matrix multiplication kernel — NKM loop with matmul_helper pack + MMA.
///
/// Tiling (Mc, Nc), L2 cache constraint, workspace sizing, and the pack/MMA
/// entry points all live in matmul_helper.{h,cpp}; this file owns the NKM
/// loops and the operator-level dispatch:
///
///   - direct path:  no transpose, A/B read raw via row strides
///   - packed path:  transpose_a or transpose_b, panels packed into the
///                   caller-provided workspace (see matmul_get_workspace_size);
///                   without a workspace the reference kernel is used
///   - non-f32/f16 dtypes fall back to the reference kernel
///
/// Beta scaling is fused into the first k-block; the epilogue (and the Relu
/// clamp) applies after the last k-block, matching matmul_ref semantics.

#include "matmul.h"
#include "matmul_helper.h"
#include "simd_kernel/simd_epilogue.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/assert.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

// Forward-declare reference kernel for the fallback paths.
namespace nnops::backend::cpu::reference {
extern void matmul_ref(const MatMulAttributes& attrs,
                       TensorView& output,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace);
}

namespace nnops::backend::cpu {
namespace {

// =========================================================================
//  Generic 2D GEMM kernels — NKM tiled loops
// =========================================================================

/// Clamp bounds for the current k-block. Relu clamps only on the final
/// k-block; intermediate partial sums must accumulate unclamped.
inline std::pair<float, float> kblock_clamp(const MatMulAttributes& attrs, bool last_k) noexcept {
    const float inf = std::numeric_limits<float>::infinity();
    if (!last_k) {
        return {-inf, inf};
    }
    if (attrs.epilogue.type == EpilogueActivateType::Relu) {
        return {0.0f, inf};
    }
    return {-inf, inf};
}

/// Non-clamp epilogues (Relu is folded into the MMA clamp) are applied
/// in-place after the last k-block of each tile.
inline bool has_inplace_epilogue(const MatMulAttributes& attrs) noexcept {
    return attrs.epilogue.type != EpilogueActivateType::None &&
           attrs.epilogue.type != EpilogueActivateType::Relu;
}

// =========================================================================
//  Direct path — A: M×K raw, B: K×N raw (no transpose)
// =========================================================================
template <typename T>
void matmul_kernel_2d_direct_flt(const MatMulAttributes& attrs,
                                 T* c_ptr, int ldc,
                                 const T* a_ptr, int lda,
                                 const T* b_ptr, int ldb,
                                 int M, int N, int K,
                                 int kc)
{
    constexpr int mr_max = mr_max_flt<T>();
    constexpr int nr_max = nr_max_flt<T>();

    int mc, nc;
    resolve_tile_sizes(M, N, mr_max, nr_max, kc, static_cast<int>(sizeof(T)), mc, nc);

    const bool epilogue = has_inplace_epilogue(attrs);

    for (int n = 0; n < N; n += nc) {
        int actual_nc = std::min(nc, N - n);
        for (int k = 0; k < K; k += kc) {
            int actual_kc = std::min(kc, K - k);
            const bool last_k = (k + kc >= K);
            const T* b_sub = b_ptr + k * ldb + n;  // B is K×N, row k, col n

            for (int m = 0; m < M; m += mc) {
                int actual_mc = std::min(mc, M - m);
                const T* a_sub = a_ptr + m * lda + k;  // A is M×K, row m, col k
                T* c_tile = c_ptr + m * ldc + n;

                // Fuse beta scaling into first k-block (subsequent blocks accumulate).
                if (k == 0 && attrs.beta != 1.0f) {
                    tile_scale(c_tile, ldc, attrs.beta, actual_mc, actual_nc);
                }

                const auto [cmin, cmax] = kblock_clamp(attrs, last_k);
                tile_mma_direct(actual_mc, actual_nc, actual_kc,
                                c_tile, ldc,
                                a_sub, lda, b_sub, ldb,
                                cmin, cmax);

                if (epilogue && last_k) {
                    epilogue_inplace(actual_mc, actual_nc, c_tile, ldc,
                                     static_cast<const T*>(nullptr), attrs.epilogue);
                }
            }
        }
    }
}

// =========================================================================
//  Packed path — A and B panels packed into the workspace
// =========================================================================
template <typename T>
void matmul_kernel_2d_packed_flt(const MatMulAttributes& attrs,
                                 T* c_ptr, int ldc,
                                 const T* a_ptr, int lda,
                                 const T* b_ptr, int ldb,
                                 int M, int N, int K, int kc,
                                 void* workspace)
{
    constexpr int mr_max = mr_max_flt<T>();
    constexpr int nr_max = nr_max_flt<T>();

    int mc, nc;
    resolve_tile_sizes(M, N, mr_max, nr_max, kc, static_cast<int>(sizeof(T)), mc, nc);

    // Workspace layout: packed A panels then packed B panels, each at a uniform
    // 64-byte-aligned stride. pack_b is offset by the full-Kc A stride (upper
    // bound of the per-k-block stride), matching matmul_get_workspace_size.
    const int ldd_a_full = align_up<PANEL_ALIGN_BYTES>(mr_max * kc * static_cast<int>(sizeof(T)))
                           / static_cast<int>(sizeof(T));
    T* pack_a = static_cast<T*>(workspace);
    const int* mr_arr = std::is_same_v<T, float> ? MR_F32 : MR_F16;
    const int num_panels_a = num_panels(mc, mr_arr);
    T* pack_b = pack_a + num_panels_a * ldd_a_full;

    const bool epilogue = has_inplace_epilogue(attrs);

    for (int n = 0; n < N; n += nc) {
        int actual_nc = std::min(nc, N - n);
        for (int k = 0; k < K; k += kc) {
            int actual_kc = std::min(kc, K - k);
            const bool last_k = (k + kc >= K);

            // Uniform 64-byte-aligned panel strides for this k-block (based on
            // the actual k-block length, so pack and MMA agree on placement).
            const int ldd_a = align_up<PANEL_ALIGN_BYTES>(mr_max * actual_kc * static_cast<int>(sizeof(T)))
                              / static_cast<int>(sizeof(T));
            const int ldd_b = align_up<PANEL_ALIGN_BYTES>(nr_max * actual_kc * static_cast<int>(sizeof(T)))
                              / static_cast<int>(sizeof(T));

            // Pack B panel.
            const T* b_src = attrs.transpose_b
                ? b_ptr + n * ldb + k   // B phys is N×K, row n, col k
                : b_ptr + k * ldb + n;  // B phys is K×N, row k, col n
            tile_pack_rhs(attrs.transpose_b, actual_nc, actual_kc,
                          pack_b, ldd_b, b_src, ldb, 1.0f);

            for (int m = 0; m < M; m += mc) {
                int actual_mc = std::min(mc, M - m);
                T* c_tile = c_ptr + m * ldc + n;

                // Pack A panel.
                const T* a_src = attrs.transpose_a
                    ? a_ptr + k * lda + m   // A phys is K×M, row k, col m
                    : a_ptr + m * lda + k;  // A phys is M×K, row m, col k
                tile_pack_lhs(attrs.transpose_a, actual_mc, actual_kc,
                              pack_a, ldd_a, a_src, lda, 1.0f);

                // Fuse beta scaling into first k-block (subsequent blocks accumulate).
                if (k == 0 && attrs.beta != 1.0f) {
                    tile_scale(c_tile, ldc, attrs.beta, actual_mc, actual_nc);
                }

                const auto [cmin, cmax] = kblock_clamp(attrs, last_k);
                // ldb < 0 signals packed B.
                tile_mma_pack(actual_mc, actual_nc, actual_kc,
                              c_tile, ldc,
                              pack_a, pack_b, -1,
                              cmin, cmax);

                if (epilogue && last_k) {
                    epilogue_inplace(actual_mc, actual_nc, c_tile, ldc,
                                     static_cast<const T*>(nullptr), attrs.epilogue);
                }
            }
        }
    }
}

// =========================================================================
//  Batched dispatch — per-batch-element 2D GEMM (numpy-style broadcasting)
// =========================================================================

template <typename T>
void matmul_dispatch_2d(const MatMulAttributes& attrs,
                        TensorView& output,
                        std::span<const TensorView> inputs,
                        int kc, void* workspace)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    const int64_t a_rank = a.rank();
    const int64_t b_rank = b.rank();
    const int64_t M  = attrs.transpose_a ? a.shape(a_rank - 1) : a.shape(a_rank - 2);
    const int64_t N  = attrs.transpose_b ? b.shape(b_rank - 2) : b.shape(b_rank - 1);
    const int64_t K  = attrs.transpose_b ? b.shape(b_rank - 1) : b.shape(b_rank - 2);

    const int ldc = static_cast<int>(output.row_stride_elems());
    const int lda = static_cast<int>(a.row_stride_elems());
    const int ldb = static_cast<int>(b.row_stride_elems());

    T* c_base = output.ptr<T>();
    const T* a_base = a.ptr<T>();
    const T* b_base = b.ptr<T>();

    // Broadcast batch dimensions (same logic as matmul_ref).
    const int64_t batch_a_dims = a_rank - 2;
    const int64_t batch_b_dims = b_rank - 2;
    const int64_t batch_ndim   = std::max(batch_a_dims, batch_b_dims);

    std::vector<int64_t> batch_a_shape(batch_ndim, 1);
    std::vector<int64_t> batch_b_shape(batch_ndim, 1);
    std::vector<int64_t> batch_out_shape(batch_ndim, 1);

    for (int64_t i = 0; i < batch_a_dims; ++i) {
        batch_a_shape[batch_ndim - batch_a_dims + i] = a.shape(i);
    }
    for (int64_t i = 0; i < batch_b_dims; ++i) {
        batch_b_shape[batch_ndim - batch_b_dims + i] = b.shape(i);
    }

    int64_t total_batch = 1;
    for (int64_t i = 0; i < batch_ndim; ++i) {
        const int64_t da = batch_a_shape[i];
        const int64_t db = batch_b_shape[i];
        if (da == db) {
            batch_out_shape[i] = da;
        } else if (da == 1) {
            batch_out_shape[i] = db;
        } else if (db == 1) {
            batch_out_shape[i] = da;
        } else {
            NNOPS_ASSERT(!"MatMul: incompatible batch dimensions for broadcast");
        }
        total_batch *= batch_out_shape[i];
    }

    const bool use_packed = attrs.transpose_a || attrs.transpose_b;

    for (int64_t bi = 0; bi < total_batch; ++bi) {
        // Unflatten batch index → multi-dimensional coords.
        int64_t rem = bi;
        int64_t a_offset = 0;
        int64_t b_offset = 0;
        int64_t c_offset = 0;

        for (int64_t d = batch_ndim - 1; d >= 0; --d) {
            const int64_t coord = rem % batch_out_shape[d];
            rem /= batch_out_shape[d];

            const int64_t a_dim = d - (batch_ndim - batch_a_dims);
            if (a_dim >= 0) {
                const int64_t a_coord = (batch_a_shape[d] == 1) ? 0 : coord;
                a_offset += a_coord * a.stride_elems(a_dim);
            }

            const int64_t b_dim = d - (batch_ndim - batch_b_dims);
            if (b_dim >= 0) {
                const int64_t b_coord = (batch_b_shape[d] == 1) ? 0 : coord;
                b_offset += b_coord * b.stride_elems(b_dim);
            }

            c_offset += coord * output.stride_elems(d);
        }

        if (use_packed) {
            matmul_kernel_2d_packed_flt<T>(attrs,
                c_base + c_offset, ldc,
                a_base + a_offset, lda,
                b_base + b_offset, ldb,
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                kc, workspace);
        } else {
            matmul_kernel_2d_direct_flt<T>(attrs,
                c_base + c_offset, ldc,
                a_base + a_offset, lda,
                b_base + b_offset, ldb,
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                kc);
        }
    }
}

}  // anonymous namespace

// =========================================================================
//  Public API
// =========================================================================

void matmul_kernel(const MatMulAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx,
                   void* workspace)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    NNOPS_ASSERT(a.rank() >= 2);
    NNOPS_ASSERT(b.rank() >= 2);
    NNOPS_ASSERT(inputs.size() >= 2 && inputs.size() <= 3);

    const auto dt_a = a.data_type();
    const auto dt_b = b.data_type();

    // Only f32 and f16 have fused kernels; everything else uses the reference.
    const bool supported = (dt_a == DataType::f32 && dt_b == DataType::f32) ||
                           (dt_a == DataType::f16 && dt_b == DataType::f16);
    if (!supported) {
        reference::matmul_ref(attrs, output, inputs, ctx, workspace);
        return;
    }

    // The packed path needs the caller-provided workspace; without it the
    // reference kernel keeps correctness (users opt into the fast path by
    // allocating matmul_get_workspace_size() bytes).
    const bool use_packed = attrs.transpose_a || attrs.transpose_b;
    if (use_packed && workspace == nullptr) {
        reference::matmul_ref(attrs, output, inputs, ctx, workspace);
        return;
    }

    const int kc = (dt_a == DataType::f32) ? KC_F32 : KC_F16;

    if (dt_a == DataType::f32) {
        matmul_dispatch_2d<float>(attrs, output, inputs, kc, workspace);
    } else {
        matmul_dispatch_2d<half>(attrs, output, inputs, kc, workspace);
    }
}

}  // namespace nnops::backend::cpu
