#pragma once
/// @file matmul_tiled_f32.hpp
/// @brief x86_64 AVX2/FMA tiled GEMM dispatch for float32.
///
/// Glues the existing pack + MMA micro-kernels into a complete tiled GEMM
/// with three-level M/N micro-panel iteration, transpose handling, epilogue
/// fusion, and optional batch loop.
///
/// Reference pattern: nn_compute shgemm_impl + TiledMatMulInterface

#include <immintrin.h>
#include <vector>
#include <cfloat>
#include <cstring>

#include "nnops/ops/matmul.hpp"
#include "nnops/core/parallel_for.hpp"
#include "pack_f32.hpp"
#include "mma_pack_f32.hpp"

namespace nnops::backend::cpu::x86_64 {

// =========================================================================
//  Micro-kernel function pointer tables (3×3: mr × nr tiers)
// =========================================================================

/// Signature for MMA micro-kernels.
using MmaPackFn = void (*)(float* NNOPS_RESTRICT C, int ldc,
                           const float* NNOPS_RESTRICT A,
                           const float* NNOPS_RESTRICT B,
                           int ldb, int K,
                           float clamp_min, float clamp_max) noexcept;

/// Signature for LHS pack (transpose) routines.
using PackTransFn = void (*)(float* NNOPS_RESTRICT output,
                             const float* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale) noexcept;

/// Signature for RHS pack (copy) routines.
using PackCopyFn = void (*)(float* NNOPS_RESTRICT output,
                            const float* NNOPS_RESTRICT input,
                            int ir_step, int K, float scale) noexcept;

/// 3×3 MMA function tables indexed by [m_idx][n_idx].
/// mr tiers: 0→6, 1→4, 2→1.  nr tiers: 0→16, 1→8, 2→1.
inline constexpr MmaPackFn mma_tbl_f32[3][3] = {
    {mma_pack_6x16_f32, mma_pack_6x8_f32,  mma_pack_6x1_f32},
    {mma_pack_4x16_f32, mma_pack_4x8_f32,  mma_pack_4x1_f32},
    {mma_pack_1x16_f32, mma_pack_1x8_f32,  mma_pack_1x1_f32},
};

/// 3×3 LHS pack (transpose) tables indexed by [m_idx][n/a].
/// mr tiers: 0→6, 1→4, 2→1.
inline constexpr PackTransFn pack_trans_tbl_f32[3] = {
    pack_trans_n6_f32,
    pack_trans_n4_f32,
    pack_trans_n1_f32,
};

/// 3×3 RHS pack (copy) tables indexed by [n_idx][n/a].
/// nr tiers: 0→16, 1→8, 2→1.
inline constexpr PackCopyFn pack_copy_tbl_f32[3] = {
    pack_copy_n16_f32,
    pack_copy_n8_f32,
    pack_copy_n1_f32,
};

/// RHS pack (transpose) tables for transpose_b path.
/// nr tiers: 0→16, 1→8, 2→1.
inline constexpr PackTransFn pack_trans_rhs_tbl_f32[3] = {
    pack_trans_n16_f32,
    pack_trans_n8_f32,
    pack_trans_n1_f32,
};

// =========================================================================
//  Index helpers — map an actual mr/nr size to its tier index
// =========================================================================

inline constexpr int m_idx_for(int mr) noexcept {
    return (mr == 6) ? 0 : (mr == 4) ? 1 : 2;
}

inline constexpr int n_idx_for(int nr) noexcept {
    return (nr == 16) ? 0 : (nr == 8) ? 1 : 2;
}

/// @returns the largest mr that fits in `rem`.
inline int choose_mr(int64_t rem) noexcept {
    if (rem >= 6) return 6;
    if (rem >= 4) return 4;
    return 1;
}

/// @returns the largest nr that fits in `rem`.
inline int choose_nr(int64_t rem) noexcept {
    if (rem >= 16) return 16;
    if (rem >= 8)  return 8;
    return 1;
}

// =========================================================================
//  Post-MMA epilogue — applied for non-clamp activation types
// =========================================================================

/// Apply epilogue to an mr×nr output block starting at C.
inline void apply_epilogue_block(float* C, int64_t ldc, int mr, int nr,
                                  const Epilogue& ep) noexcept {
    for (int m = 0; m < mr; ++m) {
        for (int n = 0; n < nr; ++n) {
            C[m * ldc + n] = apply_epilogue(ep, C[m * ldc + n], n);
        }
    }
}

// =========================================================================
//  Core tiled 2D GEMM  (no batch — processes a single [M,K]×[K,N]→[M,N])
// =========================================================================

/// Tiled float32 GEMM.
///
/// C[M][N] = A[M][K] × B[K][N]  (optionally transposed)
///
/// @param C            Output matrix, M × N row-major, ldc = N.
/// @param ldc          Leading dimension (row stride) of C.
/// @param A            Input A matrix.
/// @param lda          Leading dimension (row stride) of A in memory.
/// @param B            Input B matrix.
/// @param ldb          Leading dimension (row stride) of B in memory.
/// @param M            Number of rows of A / C.
/// @param N            Number of columns of B / C.
/// @param K            Inner dimension (columns of A, rows of B).
/// @param transpose_a  If true, A is stored as K×M and A^T is used.
/// @param transpose_b  If true, B is stored as N×K and B^T is used.
/// @param epilogue     Post-processing activation.
/// @param add_to       If true, add result to C; otherwise overwrite.
/// @param ctx          Compute context (for parallel_for).
inline void gemm_tiled_f32(
    float* C, int64_t ldc,
    const float* A, int64_t lda,
    const float* B, int64_t ldb,
    int64_t M, int64_t N, int64_t K,
    bool transpose_a, bool transpose_b,
    const Epilogue& epilogue,
    bool add_to,
    const ComputeContext& ctx)
{
    // ---- zero output if not adding to existing C ----
    if (!add_to && M > 0 && N > 0) {
        if (ldc == N) {
            std::memset(C, 0, static_cast<size_t>(M * N) * sizeof(float));
        } else {
            for (int64_t m = 0; m < M; ++m) {
                std::memset(C + m * ldc, 0, static_cast<size_t>(N) * sizeof(float));
            }
        }
    }

    // ---- clamp bounds from epilogue ----
    float clamp_min = -FLT_MAX;
    float clamp_max =  FLT_MAX;
    bool epilogue_is_clamp = false;
    if (epilogue.type == EpilogueActivateType::Relu) {
        clamp_min = 0.0f;
        epilogue_is_clamp = true;
    } else if (epilogue.type == EpilogueActivateType::None) {
        epilogue_is_clamp = true;  // identity = no extra pass needed
    }

    // ---- workspace (pack buffers) ----
    // Maximum panel sizes: mr_max=6, nr_max=16 → 22*K elements
    constexpr int mr_max = 6;
    constexpr int nr_max = 16;
    std::vector<float> workspace(static_cast<size_t>(mr_max * K + nr_max * K));
    float* pack_A = workspace.data();
    float* pack_B = pack_A + mr_max * K;

    // ---- M loop (row panels) ----
    for (int64_t m = 0; m < M; ) {
        const int mr = choose_mr(M - m);
        const int m_i = m_idx_for(mr);

        // Pack A panel [m : m+mr, :]
        const float* a_panel = transpose_a
            ? A + m              // A is K×M; column m = row m of A^T
            : A + m * lda;       // A is M×K; start of row m
        const int64_t a_step = lda;  // stride between consecutive elements in a column/row
        pack_trans_tbl_f32[m_i](pack_A, a_panel, static_cast<int>(a_step),
                                static_cast<int>(K), 1.0f);

        // ---- N loop (column panels) ----
        for (int64_t n = 0; n < N; ) {
            const int nr = choose_nr(N - n);
            const int n_i = n_idx_for(nr);

            // Pack B panel [:, n : n+nr]
            if (transpose_b) {
                // B is N×K; B^T[:, n:n+nr] = B[n:n+nr, :]
                // → pack_trans reads nr rows of K, transposes to K×nr
                pack_trans_rhs_tbl_f32[n_i](
                    pack_B, B + n * ldb, static_cast<int>(ldb),
                    static_cast<int>(K), 1.0f);
            } else {
                // B is K×N; contiguous columns n..n+nr-1 per row
                pack_copy_tbl_f32[n_i](
                    pack_B, B + n, static_cast<int>(ldb),
                    static_cast<int>(K), 1.0f);
            }

            // Call MMA micro-kernel
            mma_tbl_f32[m_i][n_i](
                C + m * ldc + n, static_cast<int>(ldc),
                pack_A, pack_B, nr,
                static_cast<int>(K), clamp_min, clamp_max);

            // Apply non-clamp epilogue to the just-written block
            if (!epilogue_is_clamp) {
                apply_epilogue_block(C + m * ldc + n, ldc, mr, nr, epilogue);
            }

            n += nr;
        }
        m += mr;
    }
}

// =========================================================================
//  Batch-aware tiled MatMul  —  top-level kernel conforming to KernelFn
// =========================================================================

/// Tiled float32 MatMul kernel with N-D batch support.
///
/// For each batch element, extracts the 2D sub-matrix and calls gemm_tiled_f32.
inline void matmul_tiled_f32(
    const MatMulAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx,
    void* /*workspace*/)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    const int64_t a_rank = a.rank();
    const int64_t b_rank = b.rank();

    // ---- extract matrix dimensions ----
    const int64_t M  = attrs.transpose_a ? a.shape(a_rank - 1) : a.shape(a_rank - 2);
    const int64_t Ka = attrs.transpose_a ? a.shape(a_rank - 2) : a.shape(a_rank - 1);
    const int64_t Kb = attrs.transpose_b ? b.shape(b_rank - 1) : b.shape(b_rank - 2);
    const int64_t N  = attrs.transpose_b ? b.shape(b_rank - 2) : b.shape(b_rank - 1);
    const int64_t K  = Ka;  // validated equal in compute()

    const auto* a_base = a.ptr<float>();
    const auto* b_base = b.ptr<float>();
    auto*       c_base = output.ptr<float>();

    const int64_t a_row_stride = a.row_stride_elems();
    const int64_t b_row_stride = b.row_stride_elems();
    const int64_t c_row_stride = output.row_stride_elems();

    // ---- compute broadcast batch shape ----
    const int64_t batch_a_dims = a_rank - 2;
    const int64_t batch_b_dims = b_rank - 2;
    const int64_t batch_ndim   = std::max(batch_a_dims, batch_b_dims);

    std::vector<int64_t> batch_a_shape(batch_ndim, 1);
    std::vector<int64_t> batch_b_shape(batch_ndim, 1);
    std::vector<int64_t> batch_out_shape(batch_ndim, 1);

    for (int64_t i = 0; i < batch_a_dims; ++i)
        batch_a_shape[batch_ndim - batch_a_dims + i] = a.shape(i);
    for (int64_t i = 0; i < batch_b_dims; ++i)
        batch_b_shape[batch_ndim - batch_b_dims + i] = b.shape(i);

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

    // ---- batch iteration ----
    for (int64_t bi = 0; bi < total_batch; ++bi) {
        int64_t rem = bi;
        int64_t a_offset = 0;
        int64_t b_offset = 0;
        int64_t c_offset = 0;

        for (int64_t d = batch_ndim - 1; d >= 0; --d) {
            const int64_t coord = rem % batch_out_shape[d];
            rem /= batch_out_shape[d];

            {
                const int64_t a_dim = d - (batch_ndim - batch_a_dims);
                if (a_dim >= 0) {
                    const int64_t a_coord = (batch_a_shape[d] == 1) ? 0 : coord;
                    a_offset += a_coord * a.stride_elems(a_dim);
                }
            }
            {
                const int64_t b_dim = d - (batch_ndim - batch_b_dims);
                if (b_dim >= 0) {
                    const int64_t b_coord = (batch_b_shape[d] == 1) ? 0 : coord;
                    b_offset += b_coord * b.stride_elems(b_dim);
                }
            }
            c_offset += coord * output.stride_elems(d);
        }

        gemm_tiled_f32(
            c_base + c_offset, c_row_stride,
            a_base + a_offset, a_row_stride,
            b_base + b_offset, b_row_stride,
            M, N, K,
            attrs.transpose_a, attrs.transpose_b,
            attrs.epilogue, attrs.add_to,
            ctx);
    }
}

}  // namespace nnops::backend::cpu::x86_64
