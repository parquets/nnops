/// @file matmul.cpp
/// @brief Tiled matrix multiplication kernel — plan-driven pack + MMA dispatch.
///
/// Packing, tiling, and the static pack decision all live in
/// matmul_helper.{h,cpp}; this file owns the tiled loops and the operator-level
/// dispatch. The pack decision (make_pack_plan<T>) picks, for each GEMM:
///
///   global pack_a | pack_b | loop order | split dim
///   --------------+--------+------------+----------
///   T             | T      | NKM (fused)| N  (B-pack traffic stays NK)
///   T             | F      | MKN (fused)| M  (A-pack traffic stays MK)
///   F             | T      | NKM (fused)| N
///   F             | F      | NKM (direct)| N  (BLIS convention)
///
/// A is packed into a per-thread stack buffer only when the global plan packs it
/// AND the split keeps A reused (MKN), or when transpose_a forces it (N-split
/// NKM). B is packed into the caller-provided workspace when the global plan
/// packs it; the per-block panel slices tile that buffer exactly, so the size is
/// thread-count invariant. Work is split at tile-block granularity through
/// ctx.cpu_parallel_for (serial fallback otherwise); C tiles are disjoint and
/// there are no reductions, so the result is bit-identical to serial.
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
//  Shared helpers
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
//  Tile-block kernels — one call processes one n-block (NKM) or m-block (MKN)
// =========================================================================

/// Direct NKM: raw A (M×K) × raw B (K×N), one n-block [n, n+actual_nc).
template <typename T>
void matmul_block_direct_nkm(const MatMulAttributes& attrs,
                             T* c_ptr, int ldc,
                             const T* a_ptr, int lda,
                             const T* b_ptr, int ldb,
                             int M, int K, int kc, int mc,
                             int n, int actual_nc)
{
    const bool epilogue = has_inplace_epilogue(attrs);

    for (int k = 0; k < K; k += kc) {
        int actual_kc = std::min(kc, K - k);
        const bool last_k = (k + kc >= K);
        const T* b_sub = b_ptr + k * ldb + n;  // B is K×N, row k, col n

        for (int m = 0; m < M; m += mc) {
            int actual_mc = std::min(mc, M - m);
            const T* a_sub = a_ptr + m * lda + k;  // A is M×K, row m, col k
            T* c_tile = c_ptr + m * ldc + n;

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

/// Fused NKM: B always packed into pack_b_slice; A packed on the stack only
/// when @p pack_a (transpose_a on this path), else read raw. One n-block
/// [n, n+actual_nc).
template <typename T>
void matmul_block_fused_nkm(const MatMulAttributes& attrs,
                            T* c_ptr, int ldc,
                            const T* a_ptr, int lda,
                            const T* b_ptr, int ldb,
                            int M, int K, int kc, int mc,
                            int n, int actual_nc,
                            bool pack_a, T* pack_b_slice)
{
    constexpr int mr_max = mr_max_flt<T>();
    constexpr int nr_max = nr_max_flt<T>();

    // Packed A lives on the kernel stack (72 KB, 64-byte aligned).
    alignas(PANEL_ALIGN_BYTES) T pack_a_buf[pack_a_stack_elems<T>()];

    const bool epilogue = has_inplace_epilogue(attrs);

    for (int k = 0; k < K; k += kc) {
        int actual_kc = std::min(kc, K - k);
        const bool last_k = (k + kc >= K);

        // Uniform 64-byte-aligned panel strides for this k-block.
        const int ldd_a = align_up<PANEL_ALIGN_BYTES>(mr_max * actual_kc * static_cast<int>(sizeof(T)))
                          / static_cast<int>(sizeof(T));
        const int ldd_b = align_up<PANEL_ALIGN_BYTES>(nr_max * actual_kc * static_cast<int>(sizeof(T)))
                          / static_cast<int>(sizeof(T));

        // B panel: packed ([K][nr]) regardless of transpose_b.
        const T* b_src = attrs.transpose_b
            ? b_ptr + n * ldb + k     // B phys is N×K, row n, col k
            : b_ptr + k * ldb + n;    // B phys is K×N, row k, col n
        tile_pack_rhs(attrs.transpose_b, actual_nc, actual_kc,
                      pack_b_slice, ldd_b, b_src, ldb, 1.0f);

        for (int m = 0; m < M; m += mc) {
            int actual_mc = std::min(mc, M - m);
            T* c_tile = c_ptr + m * ldc + n;

            if (k == 0 && attrs.beta != 1.0f) {
                tile_scale(c_tile, ldc, attrs.beta, actual_mc, actual_nc);
            }

            const auto [cmin, cmax] = kblock_clamp(attrs, last_k);

            if (pack_a) {
                const T* a_src = attrs.transpose_a
                    ? a_ptr + k * lda + m    // A phys is K×M, row k, col m
                    : a_ptr + m * lda + k;   // A phys is M×K, row m, col k
                tile_pack_lhs(attrs.transpose_a, actual_mc, actual_kc,
                              pack_a_buf, ldd_a, a_src, lda, 1.0f);
                tile_mma_pack(actual_mc, actual_nc, actual_kc,
                              c_tile, ldc,
                              pack_a_buf, pack_b_slice, -1,
                              cmin, cmax);
            } else {
                const T* a_sub = a_ptr + m * lda + k;  // A phys is M×K, row m, col k
                tile_mma_direct(actual_mc, actual_nc, actual_kc,
                                c_tile, ldc,
                                a_sub, lda, pack_b_slice, -1,
                                cmin, cmax);
            }

            if (epilogue && last_k) {
                epilogue_inplace(actual_mc, actual_nc, c_tile, ldc,
                                 static_cast<const T*>(nullptr), attrs.epilogue);
            }
        }
    }
}

/// Fused MKN: A packed once per k-block and reused across all n-blocks; B is
/// raw K×N. One m-block [m, m+actual_mc), full N width.
template <typename T>
void matmul_block_fused_mkn(const MatMulAttributes& attrs,
                            T* c_ptr, int ldc,
                            const T* a_ptr, int lda,
                            const T* b_ptr, int ldb,
                            int N, int K, int kc, int nc,
                            int m, int actual_mc)
{
    constexpr int mr_max = mr_max_flt<T>();

    // Packed A lives on the kernel stack (72 KB, 64-byte aligned).
    alignas(PANEL_ALIGN_BYTES) T pack_a_buf[pack_a_stack_elems<T>()];

    const bool epilogue = has_inplace_epilogue(attrs);

    for (int k = 0; k < K; k += kc) {
        int actual_kc = std::min(kc, K - k);
        const bool last_k = (k + kc >= K);

        const int ldd_a = align_up<PANEL_ALIGN_BYTES>(mr_max * actual_kc * static_cast<int>(sizeof(T)))
                          / static_cast<int>(sizeof(T));

        // Pack A once for this m-block, reuse across every n-block.
        const T* a_src = attrs.transpose_a
            ? a_ptr + k * lda + m    // A phys is K×M, row k, col m
            : a_ptr + m * lda + k;   // A phys is M×K, row m, col k
        tile_pack_lhs(attrs.transpose_a, actual_mc, actual_kc,
                      pack_a_buf, ldd_a, a_src, lda, 1.0f);

        T* c_tile = c_ptr + m * ldc;  // full row range [0, N)
        if (k == 0 && attrs.beta != 1.0f) {
            tile_scale(c_tile, ldc, attrs.beta, actual_mc, N);
        }

        const auto [cmin, cmax] = kblock_clamp(attrs, last_k);

        for (int n = 0; n < N; n += nc) {
            int actual_nc = std::min(nc, N - n);
            const T* b_mma = b_ptr + k * ldb + n;  // B raw K×N, row k, col n
            tile_mma_pack(actual_mc, actual_nc, actual_kc,
                          c_tile + n, ldc,
                          pack_a_buf, b_mma, ldb,
                          cmin, cmax);
        }

        if (epilogue && last_k) {
            epilogue_inplace(actual_mc, N, c_tile, ldc,
                             static_cast<const T*>(nullptr), attrs.epilogue);
        }
    }
}

// =========================================================================
//  Batched dispatch — plan-driven routing + block-granularity parallelism
// =========================================================================

template <typename T>
void matmul_dispatch_2d(const MatMulAttributes& attrs,
                        TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx,
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

    // Global plan (full M×N) → routing + split dimension.
    const PackPlan plan = make_pack_plan<T>(attrs, M, N, lda, ldb);

    // A workspace is needed iff pack_b; without one, fall back to reference.
    if (plan.pack_b && workspace == nullptr) {
        reference::matmul_ref(attrs, output, inputs, ctx, nullptr);
        return;
    }

    // Resolve tile sizes once (M/N/K are shared across all batch elements).
    constexpr int mr_max = mr_max_flt<T>();
    constexpr int nr_max = nr_max_flt<T>();
    int mc, nc;
    resolve_tile_sizes(static_cast<int>(M), static_cast<int>(N),
                       mr_max, nr_max, kc, static_cast<int>(sizeof(T)), mc, nc);

    // Workspace slice constants for the N-split packed-B path: each full
    // n-block owns np_full panels at the full-Kc aligned stride.
    const int ldd_b_full = align_up<PANEL_ALIGN_BYTES>(nr_max * kc * static_cast<int>(sizeof(T)))
                           / static_cast<int>(sizeof(T));
    const int np_full = num_panels(nc, std::is_same_v<T, float> ? NR_F32 : NR_F16);

    // Split dimension: MKN → M-split, otherwise N-split (BLIS convention).
    const bool split_m = plan.mkn_order;
    const int64_t num_blocks = split_m ? (M + mc - 1) / mc : (N + nc - 1) / nc;

    // Per-slab pack_a for N-split: stride-triggered packing is refined away —
    // each thread reads A exactly once, so pack_a reduces to transpose_a.
    const bool pack_a_slab = attrs.transpose_a;

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

    const int Mi = static_cast<int>(M);
    const int Ni = static_cast<int>(N);
    const int Ki = static_cast<int>(K);

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

        T* c_p = c_base + c_offset;
        const T* a_p = a_base + a_offset;
        const T* b_p = b_base + b_offset;

        auto run_block = [&](int64_t blk) {
            if (split_m) {
                const int m = static_cast<int>(blk) * mc;
                const int actual_mc = std::min(mc, Mi - m);
                matmul_block_fused_mkn<T>(attrs, c_p, ldc, a_p, lda, b_p, ldb,
                                          Ni, Ki, kc, nc, m, actual_mc);
            } else if (plan.pack_b) {
                const int n = static_cast<int>(blk) * nc;
                const int actual_nc = std::min(nc, Ni - n);
                T* pack_b_slice = static_cast<T*>(workspace) + blk * np_full * ldd_b_full;
                matmul_block_fused_nkm<T>(attrs, c_p, ldc, a_p, lda, b_p, ldb,
                                          Mi, Ki, kc, mc, n, actual_nc,
                                          pack_a_slab, pack_b_slice);
            } else {
                const int n = static_cast<int>(blk) * nc;
                const int actual_nc = std::min(nc, Ni - n);
                matmul_block_direct_nkm<T>(attrs, c_p, ldc, a_p, lda, b_p, ldb,
                                           Mi, Ki, kc, mc, n, actual_nc);
            }
        };

        if (ctx.cpu_parallel_for) {
            ctx.cpu_parallel_for(0, num_blocks, run_block);
        } else {
            for (int64_t blk = 0; blk < num_blocks; ++blk) {
                run_block(blk);
            }
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

    const int kc = (dt_a == DataType::f32) ? KC_F32 : KC_F16;

    if (dt_a == DataType::f32) {
        matmul_dispatch_2d<float>(attrs, output, inputs, ctx, kc, workspace);
    } else {
        matmul_dispatch_2d<half>(attrs, output, inputs, ctx, kc, workspace);
    }
}

}  // namespace nnops::backend::cpu
