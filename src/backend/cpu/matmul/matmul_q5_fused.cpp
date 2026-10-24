/// @file matmul_q5_fused.cpp
/// @brief Fused Q5_1 small-M MatMul — decode straight into the dot product.
///
/// The packed route re-decodes every Q5_1 block into uniform panels, writes
/// them to scratch and reads them back. At small M there are too few output
/// rows to amortise that round trip, so this route skips it: no packed B, no
/// workspace, no k-blocking — each kernel sweeps K once and decodes B in
/// registers. x86-only (the arch's mma_q5_f32_{trans,copy}); everywhere else
/// the packed route handles Q5_1.
///
/// The M limits are per layout and measured; the driver asks this file to run
/// the product and gets false back when the shape is out of range.

#include "matmul_internal.h"
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"   // tile_scale / epilogue_inplace

#include <algorithm>

namespace nnops::backend::cpu {

#if defined(NNOPS_ARCH_X86_64)

/// Benchmark-only cap for the fused Q5_1 route: negative = the limits below,
/// 0 = disabled, else caps M (so one process can A/B both routes).
int q5_fused_max_m_override = -1;

namespace {

/// Largest M routed to the fused Q5_1 kernels, measured at K=N=4096: the dot
/// kernel stays 2x..3.8x ahead to M=16 and loses at 32, where the per-panel
/// re-decode outweighs the packed panel round trip; the outer kernel (2 rows
/// per panel, four accumulators each) wins to M=4.
constexpr int Q5_FUSED_MAX_M_TRANS = 16;
constexpr int Q5_FUSED_MAX_M_COPY  = 4;

/// Two Q5_1 blocks per worker task, which is also the copy kernel's minimum.
constexpr int Q5_FUSED_NTILE = 2 * static_cast<int>(q5_1_block_elems);

/// Call the kernel instantiated for @p MR rows, recursing down to the live count.
template <bool ZERO, bool TRANS, int MR>
inline void q5_fused_rows(int mr, int nc, int K, float* c, int ldc,
                          const float* a, int lda, const uint8_t* B,
                          int64_t row_blocks, int n_start,
                          float cmin, float cmax) {
    if (mr >= MR) {
        if constexpr (TRANS) {
            arch::mma_q5_f32_trans<MR, ZERO>(nc, K, c, ldc, a, lda, B, row_blocks,
                                             n_start, cmin, cmax);
        } else {
            arch::mma_q5_f32_copy<MR, ZERO>(nc, K, c, ldc, a, lda, B, row_blocks,
                                            n_start, cmin, cmax);
        }
    } else if constexpr (MR > 1) {
        q5_fused_rows<ZERO, TRANS, MR - 1>(mr, nc, K, c, ldc, a, lda, B,
                                           row_blocks, n_start, cmin, cmax);
    }
}

/// One m-panel at a time: 8 rows for the dot kernel, 2 for the outer product.
template <bool ZERO, bool TRANS>
inline void q5_fused_dispatch(int M, int nc, int K, float* C, int ldc,
                              const float* A, int lda, const uint8_t* B,
                              int64_t row_blocks, int n_start,
                              float cmin, float cmax) {
    constexpr int kPanel = TRANS ? 8 : 2;
    for (int m = 0; m < M; m += kPanel) {
        q5_fused_rows<ZERO, TRANS, kPanel>(
            std::min(kPanel, M - m), nc, K, C + static_cast<int64_t>(m) * ldc,
            ldc, A + static_cast<int64_t>(m) * lda, lda, B, row_blocks,
            n_start, cmin, cmax);
    }
}

/// Select the kernel family for one n tile; ZERO is the beta == 0 case.
template <bool ZERO>
inline void q5_fused_run(bool trans, int M, int nc, int K, float* C, int ldc,
                         const float* A, int lda, const uint8_t* B,
                         int64_t row_blocks, int n_start,
                         float cmin, float cmax) {
    if (trans) {
        q5_fused_dispatch<ZERO, true>(M, nc, K, C, ldc, A, lda, B, row_blocks,
                                      n_start, cmin, cmax);
    } else {
        q5_fused_dispatch<ZERO, false>(M, nc, K, C, ldc, A, lda, B, row_blocks,
                                       n_start, cmin, cmax);
    }
}

}  // namespace

bool matmul_q5_fused(const MatMulAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     int M, int N, int K)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    const bool trans = attrs.transpose_b;
    if (K <= 0 || N <= 0) {
        return false;
    }
    int max_m = trans ? Q5_FUSED_MAX_M_TRANS : Q5_FUSED_MAX_M_COPY;
    if (q5_fused_max_m_override >= 0) {
        max_m = q5_fused_max_m_override;
    }
    if (M > max_m) {
        return false;
    }
    const int lda = static_cast<int>(a.row_stride_elems());
    const int ldc = static_cast<int>(output.row_stride_elems());
    const int64_t row_blocks = b.row_stride_elems();

    const float* a_base = a.ptr<float>();
    const uint8_t* b_base = static_cast<const uint8_t*>(b.ptr<void>());
    float* c_base = output.ptr<float>();

    const auto [cmin, cmax] = kblock_clamp(attrs, /*last_k=*/true);
    const bool epilogue = has_inplace_epilogue(attrs);
    const bool scale_c = attrs.beta != 0.0f && attrs.beta != 1.0f;

    const BatchWalk batch(a, b);
    const int nt = Q5_FUSED_NTILE;
    const int64_t n_tiles = split_block_count(N, nt);

    // One flat task list over (batch, n tile): batch elements need no fork/join.
    ctx.cpu.run(0, batch.total * n_tiles, [&](int64_t task) {
        const int64_t bi = task / n_tiles;
        const int64_t t = task % n_tiles;
        const int n_start = static_cast<int>(t) * nt;
        const int nc = std::min(nt, static_cast<int>(N) - n_start);

        int64_t a_off, b_off, c_off;
        batch.offsets(bi, output, a_off, b_off, c_off);
        const float* a_p = a_base + a_off;
        const uint8_t* b_p = q5_rhs_at(b_base, b_off);
        float* c_p = c_base + c_off + n_start;

        if (scale_c) {
            tile_scale(c_p, ldc, attrs.beta, M, nc);
        }

        if (attrs.beta == 0.0f) {
            q5_fused_run<true>(trans, M, nc, K, c_p, ldc, a_p, lda, b_p,
                               row_blocks, n_start, cmin, cmax);
        } else {
            q5_fused_run<false>(trans, M, nc, K, c_p, ldc, a_p, lda, b_p,
                                row_blocks, n_start, cmin, cmax);
        }

        if (epilogue) {
            epilogue_inplace(M, nc, c_p, ldc,
                             static_cast<const float*>(nullptr), attrs.epilogue);
        }
    });
    return true;
}

#endif  // NNOPS_ARCH_X86_64

}  // namespace nnops::backend::cpu
