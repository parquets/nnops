/// @file matmul_qblock_fused.cpp
/// @brief Fused block-quant small-M MatMul — decode straight into the product.
///
/// The packed route re-decodes every Q5_1/Q4_1 block into uniform panels,
/// writes them to scratch and reads them back. At small M there are too few
/// output rows to amortise that round trip, so this route skips it: no packed
/// B, no workspace, no k-blocking — each kernel sweeps K once and decodes B in
/// registers. x86-only (the arch's mma_qblock_f32_{trans,copy}); everywhere
/// else the packed route handles the block formats.
///
/// The M limits are per layout and per format, measured; they live on the block
/// policies in mma_qblock.hpp. The driver asks this file to run the product and
/// gets false back when the shape is out of range.

#include "matmul_internal.h"
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"   // tile_scale / epilogue_inplace

#include <algorithm>

namespace nnops::backend::cpu {

#if defined(NNOPS_ARCH_X86_64)

/// Benchmark-only cap for the fused block-quant route: negative = the policy's
/// limits, 0 = disabled, else caps M (so one process can A/B both routes). It
/// covers Q5_1 and Q4_1 alike, which is what keeps the format ladders
/// comparing two packed routes unless they ask for the fused one.
int qblock_fused_max_m_override = -1;

namespace {

/// Two quant blocks per worker task, which is also the copy kernel's minimum
/// tile.
constexpr int QBLOCK_FUSED_NTILE = 2 * arch::Q5Block::block_elems;

/// Call the kernel instantiated for @p MR rows, recursing down to the live count.
template <class BQ, bool ZERO, bool TRANS, int MR>
inline void qblock_fused_rows(int mr, int nc, int K, float* c, int ldc,
                              const float* a, int lda, const uint8_t* B,
                              int64_t row_blocks, int n_start, float alpha,
                              float cmin, float cmax) {
    if (mr >= MR) {
        if constexpr (TRANS) {
            arch::mma_qblock_f32_trans<BQ, MR, ZERO>(
                nc, K, c, ldc, a, lda, B, row_blocks, n_start, alpha, cmin, cmax);
        } else {
            arch::mma_qblock_f32_copy<BQ, MR, ZERO>(
                nc, K, c, ldc, a, lda, B, row_blocks, n_start, alpha, cmin, cmax);
        }
    } else if constexpr (MR > 1) {
        qblock_fused_rows<BQ, ZERO, TRANS, MR - 1>(
            mr, nc, K, c, ldc, a, lda, B, row_blocks, n_start, alpha, cmin, cmax);
    }
}

/// One m-panel at a time: 8 rows for the dot kernel, 2 for the outer product.
template <class BQ, bool ZERO, bool TRANS>
inline void qblock_fused_dispatch(int M, int nc, int K, float* C, int ldc,
                                  const float* A, int lda, const uint8_t* B,
                                  int64_t row_blocks, int n_start, float alpha,
                                  float cmin, float cmax) {
    constexpr int kPanel = TRANS ? 8 : 2;
    for (int m = 0; m < M; m += kPanel) {
        qblock_fused_rows<BQ, ZERO, TRANS, kPanel>(
            std::min(kPanel, M - m), nc, K, C + static_cast<int64_t>(m) * ldc,
            ldc, A + static_cast<int64_t>(m) * lda, lda, B, row_blocks,
            n_start, alpha, cmin, cmax);
    }
}

/// Select the kernel family for one n tile; ZERO is the beta == 0 case.
template <class BQ, bool ZERO>
inline void qblock_fused_run(bool trans, int M, int nc, int K, float* C, int ldc,
                             const float* A, int lda, const uint8_t* B,
                             int64_t row_blocks, int n_start, float alpha,
                             float cmin, float cmax) {
    if (trans) {
        qblock_fused_dispatch<BQ, ZERO, true>(M, nc, K, C, ldc, A, lda, B,
                                              row_blocks, n_start, alpha, cmin,
                                              cmax);
    } else {
        qblock_fused_dispatch<BQ, ZERO, false>(M, nc, K, C, ldc, A, lda, B,
                                               row_blocks, n_start, alpha, cmin,
                                               cmax);
    }
}

/// The format-independent body: one flat task list over (batch, n tile).
template <class BQ>
inline void qblock_fused_all(const MatMulAttributes& attrs, TensorView& output,
                             std::span<const TensorView> inputs,
                             const ComputeContext& ctx, bool trans, int M,
                             int N, int K) {
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    const int lda = static_cast<int>(a.row_stride_elems());
    const int ldc = static_cast<int>(output.row_stride_elems());
    const int64_t row_blocks = b.row_stride_elems();

    const float* a_base = a.ptr<float>();
    const uint8_t* b_base = static_cast<const uint8_t*>(b.ptr<void>());
    float* c_base = output.ptr<float>();

    const auto [cmin, cmax] = kblock_clamp(attrs, /*last_k=*/true);
    const bool epilogue = has_inplace_epilogue(attrs);
    // A bias primes each tile with beta·C_old + bias before the k loop, so this
    // route always accumulates (the overwriting mode would drop it) and the
    // last-k clamp + epilogue run on the biased sum, as on the tiled route.
    const float* bias = inputs.size() > 2 ? inputs[2].ptr<float>() : nullptr;
    const bool prime_c = (bias != nullptr);
    const bool scale_c = !prime_c && attrs.beta != 0.0f && attrs.beta != 1.0f;

    const BatchWalk batch(a, b);
    const int nt = QBLOCK_FUSED_NTILE;
    const int64_t n_tiles = split_block_count(N, nt);

    ctx.cpu.run(0, batch.total * n_tiles, [&](int64_t task) {
        const int64_t bi = task / n_tiles;
        const int64_t t = task % n_tiles;
        const int n_start = static_cast<int>(t) * nt;
        const int nc = std::min(nt, static_cast<int>(N) - n_start);

        int64_t a_off, b_off, c_off;
        batch.offsets(bi, output, a_off, b_off, c_off);
        const float* a_p = a_base + a_off;
        const uint8_t* b_p = block_rhs_at(b_base, b_off, BQ::dtype);
        float* c_p = c_base + c_off + n_start;

        if (prime_c) {
            tile_scale_add_bias(c_p, ldc, attrs.beta, bias + n_start, M, nc);
        } else if (scale_c) {
            tile_scale(c_p, ldc, attrs.beta, M, nc);
        }

        // The clamp of a k-block route runs once at the end; accumulating into
        // C instead means the clamp has to run on the sum, not on the partial.
        if (attrs.beta == 0.0f && !prime_c) {
            qblock_fused_run<BQ, true>(trans, M, nc, K, c_p, ldc, a_p, lda, b_p,
                                       row_blocks, n_start, attrs.alpha, cmin,
                                       cmax);
        } else {
            qblock_fused_run<BQ, false>(trans, M, nc, K, c_p, ldc, a_p, lda, b_p,
                                        row_blocks, n_start, attrs.alpha, cmin,
                                        cmax);
        }

        if (epilogue) {
            epilogue_inplace(M, nc, c_p, ldc,
                             static_cast<const float*>(nullptr), attrs.epilogue);
        }
    });
}

/// One format's route: the measured per-layout M limit, then the kernels.
template <class BQ>
inline bool qblock_fused_try(const MatMulAttributes& attrs, TensorView& output,
                             std::span<const TensorView> inputs,
                             const ComputeContext& ctx, int M, int N, int K) {
    const bool trans = attrs.transpose_b;
    int max_m = trans ? BQ::fused_max_m_trans : BQ::fused_max_m_copy;
    if (qblock_fused_max_m_override >= 0) {
        max_m = qblock_fused_max_m_override;
    }
    if (M > max_m) {
        return false;
    }
    qblock_fused_all<BQ>(attrs, output, inputs, ctx, trans, M, N, K);
    return true;
}

}  // namespace

bool matmul_qblock_fused(const MatMulAttributes& attrs,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         int M, int N, int K)
{
    if (K <= 0 || N <= 0) {
        return false;
    }
    switch (inputs[1].data_type()) {
    case DataType::q5_1: return qblock_fused_try<arch::Q5Block>(attrs, output, inputs, ctx, M, N, K);
    case DataType::q4_1: return qblock_fused_try<arch::Q4Block>(attrs, output, inputs, ctx, M, N, K);
    default:             return false;
    }
}

#endif  // NNOPS_ARCH_X86_64

}  // namespace nnops::backend::cpu
