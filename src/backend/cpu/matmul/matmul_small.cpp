/// @file matmul_small.cpp
/// @brief Small f32/f16 MatMul — the naive SIMD kernel and the direct GEMV.
///
/// Both routes skip the plan entirely: no pack, no tiles, no scratch, one pass
/// over the product. Which one runs is decided here, so the driver only asks
/// "was it small?" and the two thresholds stay with the code they gate.
///
///   - M*N*K < GEMM_FAST_PATH_THRESHOLD: packing and tiling cost more than the
///     arithmetic; run the arch's naive triple loop, then one epilogue pass over
///     the (tiny) output.
///   - M <= GEMV_M_MAX: packing B writes and then re-reads a whole N×K operand
///     for a handful of output rows; stream B once through the arch's GEMV.
///
/// The batch walk shared with the driver lives in matmul_internal.h.

#include "matmul_internal.h"
#include "backend/cpu/simd_kernel/simd_epilogue.hpp" // apply_epilogue
#include "nnops/detail/half.hpp"

#include <algorithm>
#include <type_traits>

#if defined(NNOPS_ARCH_AARCH64)
#include <arm_neon.h>
#endif

namespace nnops::backend::cpu {
namespace {

/// Total MACs below which the small path stays on the calling thread: a pool
/// spawn costs more than the work it would spread.
constexpr int64_t NAIVE_PARALLEL_MACS = 32768;

/// M at or below this takes the direct GEMV.
constexpr int64_t GEMV_M_MAX = 4;

/// C = beta·C_old + bias over every batch element, before anything accumulates:
/// the accumulator-init the tiled route does in its first k-block. The kernel
/// then adds the product with beta = 1, and the bias — unscaled by alpha, added
/// before clamp and activation — is already in place.
template <typename T>
void prime_bias_pass(const MatMulAttributes& attrs, TensorView& output, const BatchWalk& batch,
                     T* c_base, int ldc, int M, int N, const T* bias, const ComputeContext& ctx)
{
    const auto prime = [&](int64_t bi) {
        int64_t a_off, b_off, c_off;
        batch.offsets(bi, output, a_off, b_off, c_off);
        tile_scale_add_bias(c_base + c_off, ldc, attrs.residual_pre_activation ? 1.0f : attrs.beta,
                            bias, M, N);
    };
    if (batch.total > 1) {
        ctx.cpu.run(0, batch.total, prime);
    } else {
        prime(0);
    }
}

/// Apply the clamp and activation epilogue in place over each batch element's
/// M×N output, the same order the tiled route folds into its last k-block.
template <typename T>
void apply_epilogue_pass(const MatMulAttributes& attrs, TensorView& output, const BatchWalk& batch,
                         T* c_base, int ldc, int M, int N)
{
    const Epilogue& ep = attrs.epilogue;
    const float inf = std::numeric_limits<float>::infinity();
    const bool clamp = ep.min_clip > -inf || ep.max_clip < inf;
    if (ep.type == EpilogueActivateType::None && !clamp) {
        return;
    }
    for (int64_t bi = 0; bi < batch.total; ++bi) {
        int64_t a_off, b_off, c_off;
        batch.offsets(bi, output, a_off, b_off, c_off);
        T* c_p = c_base + c_off;
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                T& slot = c_p[m * ldc + n];
                float f = 0.0f;
                if constexpr (std::is_same_v<T, float>) {
                    f = slot;
                } else {
                    f = half_to_float(slot);
                }
                f = std::min(std::max(f, ep.min_clip), ep.max_clip);
                if constexpr (std::is_same_v<T, float>) {
                    slot = apply_epilogue(ep, f);
                } else {
                    slot = float_to_half(apply_epilogue(ep, f));
                }
            }
        }
    }
}

#if defined(NNOPS_ARCH_AARCH64)
/// One 64-column block of an f16 SME blob: [ceil(K/2)][128], column j's K pair
/// at 2j and 2j+1. `vld2q_f16` de-interleaves 8 columns' two K values in one
/// instruction, so a paired K loop costs no extra traffic. C = alpha·A·B +
/// beta·C, M <= 4 by construction.
template <int M>
void gemv_blocked_f16(int K, const half* a, int lda, const half* b_blk, half* c, int ldc,
                      float alpha, float beta, int width)
{
    const int pairs = (K + 1) / 2;
    for (int j0 = 0; j0 < width; j0 += 16) {
        const int w = std::min(16, width - j0);
        float32x4_t acc[M][4];
        for (int m = 0; m < M; ++m) {
            for (int v = 0; v < 4; ++v) { acc[m][v] = vdupq_n_f32(0.0f); }
        }
        for (int p = 0; p < pairs; ++p) {
            const float16_t* row =
                reinterpret_cast<const float16_t*>(b_blk + static_cast<int64_t>(p) * 128 + 2 * j0);
            float av0[M], av1[M];
            for (int m = 0; m < M; ++m) {
                av0[m] = static_cast<float>(a[static_cast<int64_t>(m) * lda + 2 * p]);
                av1[m] = (2 * p + 1 < K) ?
                             static_cast<float>(a[static_cast<int64_t>(m) * lda + 2 * p + 1]) :
                             0.0f;
            }
            for (int q = 0; q < 2; ++q) { // 8 columns each
                const float16x8x2_t v = vld2q_f16(row + q * 16);
                const float32x4_t lo0 = vcvt_f32_f16(vget_low_f16(v.val[0]));
                const float32x4_t hi0 = vcvt_f32_f16(vget_high_f16(v.val[0]));
                const float32x4_t lo1 = vcvt_f32_f16(vget_low_f16(v.val[1]));
                const float32x4_t hi1 = vcvt_f32_f16(vget_high_f16(v.val[1]));
                for (int m = 0; m < M; ++m) {
                    acc[m][2 * q] = vfmaq_n_f32(acc[m][2 * q], lo0, av0[m]);
                    acc[m][2 * q] = vfmaq_n_f32(acc[m][2 * q], lo1, av1[m]);
                    acc[m][2 * q + 1] = vfmaq_n_f32(acc[m][2 * q + 1], hi0, av0[m]);
                    acc[m][2 * q + 1] = vfmaq_n_f32(acc[m][2 * q + 1], hi1, av1[m]);
                }
            }
        }
        for (int m = 0; m < M; ++m) {
            half* dst = c + static_cast<int64_t>(m) * ldc + j0;
            float16_t* out = reinterpret_cast<float16_t*>(dst);
            float32x4_t res[4];
            for (int v = 0; v < 4; ++v) {
                res[v] = (alpha != 1.0f) ? vmulq_n_f32(acc[m][v], alpha) : acc[m][v];
            }
            if (beta != 0.0f && w == 16) {
                for (int v = 0; v < 4; ++v) {
                    res[v] = vmlaq_n_f32(res[v], vcvt_f32_f16(vld1_f16(out + v * 4)), beta);
                }
            }
            if (w == 16) {
                vst1q_f16(out, vcombine_f16(vcvt_f16_f32(res[0]), vcvt_f16_f32(res[1])));
                vst1q_f16(out + 8, vcombine_f16(vcvt_f16_f32(res[2]), vcvt_f16_f32(res[3])));
            } else {
                float tmp[16];
                for (int v = 0; v < 4; ++v) { vst1q_f32(tmp + v * 4, res[v]); }
                for (int i = 0; i < w; ++i) {
                    const float v = (beta != 0.0f) ? tmp[i] + beta * static_cast<float>(dst[i])
                                                   : tmp[i];
                    dst[i] = static_cast<half>(v);
                }
            }
        }
    }
}
#endif // NNOPS_ARCH_AARCH64

/// M <= GEMV_M_MAX: stream B once through the arch's GEMV kernel. Blocks are
/// split over N on a single batch element, else over the batch.
template <typename T>
void matmul_gemv(const MatMulAttributes& attrs, TensorView& output,
                 std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];
    const auto [M, N, K] = matmul_dims(attrs, a, b);

    const int lda = static_cast<int>(a.row_stride_elems());
    const int ldb = static_cast<int>(b.row_stride_elems());
    const int ldc = static_cast<int>(output.row_stride_elems());
    const int Mi = static_cast<int>(M);
    const int Ni = static_cast<int>(N);
    const int Ki = static_cast<int>(K);
    // B's size picks the traversal, which decides how wide a slice is worth one
    // call (see arch::gemv_slice).
    const int span = arch::gemv_slice<T>(Mi, N, K);
    // The SME blob: one [K][64] block per 64 output columns, blocks K*64 apart.
    const bool blocked_b = b.layout() == TensorLayout::PackedBSme;
    const int block_w = 64;
    const int blocks = blocked_b ? (Ni + block_w - 1) / block_w : (Ni + span - 1) / span;

    const T* a_base = a.ptr<T>();
    T* c_base = output.ptr<T>();
    const T* b_base = b.ptr<T>();

    const BatchWalk batch(a, b);
    // A bias primes C with beta·C_old + bias; the kernel then adds the product.
    const T* bias = inputs.size() > 2 ? inputs[2].ptr<T>() : nullptr;
    if (bias != nullptr) {
        prime_bias_pass<T>(attrs, output, batch, c_base, ldc, Mi, Ni, bias, ctx);
    }
    // With a bias primed, or a residual in C, the kernel must accumulate.
    const float beta = (bias != nullptr || attrs.residual_pre_activation) ? 1.0f : attrs.beta;
    const auto block = [&](int64_t bi, int nb) {
        int64_t a_off, b_off, c_off;
        batch.offsets(bi, output, a_off, b_off, c_off);
        const int n0 = static_cast<int>(nb) * (blocked_b ? block_w : span);
        const int n1 = std::min(n0 + (blocked_b ? block_w : span), Ni);
        if (blocked_b) {
            const int64_t stride = std::is_same_v<T, float> ?
                                       static_cast<int64_t>(Ki) * block_w :
                                       static_cast<int64_t>((Ki + 1) / 2) * (2 * block_w);
            const T* b_blk = b_base + b_off + static_cast<int64_t>(nb) * stride;
            const int width = n1 - n0;
            T* c_blk = c_base + c_off + n0;
            if constexpr (std::is_same_v<T, float>) {
                // f32: one block is a row-major [Ki][64]; the column range is
                // relative to it, and C moves with the block.
                arch::matmul_gemv(Mi, Ni, Ki, a_base + a_off, lda, attrs.transpose_a, b_blk,
                                  block_w, /*trans_b=*/false, c_blk, ldc, attrs.alpha, beta, 0,
                                  width);
            }
#if defined(NNOPS_ARCH_AARCH64)
            else {
                const half* a_h = a_base + a_off;
                half* c_h = c_blk;
                const half* b_h = b_blk;
                switch (Mi) {
                case 1: gemv_blocked_f16<1>(Ki, a_h, lda, b_h, c_h, ldc, attrs.alpha, beta, width); break;
                case 2: gemv_blocked_f16<2>(Ki, a_h, lda, b_h, c_h, ldc, attrs.alpha, beta, width); break;
                case 3: gemv_blocked_f16<3>(Ki, a_h, lda, b_h, c_h, ldc, attrs.alpha, beta, width); break;
                default: gemv_blocked_f16<4>(Ki, a_h, lda, b_h, c_h, ldc, attrs.alpha, beta, width); break;
                }
            }
#endif
            return;
        }
        arch::matmul_gemv(Mi, Ni, Ki, a_base + a_off, lda, attrs.transpose_a, b_base + b_off, ldb,
                          attrs.transpose_b, c_base + c_off, ldc, attrs.alpha, beta, n0, n1);
    };

    if (batch.total > 1) {
        const auto element = [&](int64_t bi) {
            for (int nb = 0; nb < blocks; ++nb) {
                block(bi, nb);
            }
        };
        if (batch.total * M * N * K >= NAIVE_PARALLEL_MACS) {
            ctx.cpu.run(0, batch.total, element);
        } else {
            for (int64_t bi = 0; bi < batch.total; ++bi) {
                element(bi);
            }
        }
    } else if (M * N * K >= NAIVE_PARALLEL_MACS) {
        ctx.cpu.run(0, blocks, [&](int64_t nb) { block(0, nb); });
    } else {
        for (int nb = 0; nb < blocks; ++nb) {
            block(0, nb);
        }
    }

    apply_epilogue_pass<T>(attrs, output, batch, c_base, ldc, Mi, Ni);
}

template <typename T>
void matmul_naive_small(const MatMulAttributes& attrs, TensorView& output,
                        std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];
    const auto [M, N, K] = matmul_dims(attrs, a, b);

    const int lda = static_cast<int>(a.row_stride_elems());
    const int ldb = static_cast<int>(b.row_stride_elems());
    const int ldc = static_cast<int>(output.row_stride_elems());
    const int Mi = static_cast<int>(M);
    const int Ni = static_cast<int>(N);
    const int Ki = static_cast<int>(K);

    const T* a_base = a.ptr<T>();
    T* c_base = output.ptr<T>();
    // The SME blob: one [K][64] block per 64 output columns, blocks K*64 apart.
    const bool blocked_b = b.layout() == TensorLayout::PackedBSme;
    const int block_w = 64;
    const T* b_base = b.ptr<T>();

    const BatchWalk batch(a, b);
    // Same accumulator-init as the GEMV route: prime, then accumulate beta = 1.
    const T* bias = inputs.size() > 2 ? inputs[2].ptr<T>() : nullptr;
    if (bias != nullptr) {
        prime_bias_pass<T>(attrs, output, batch, c_base, ldc, Mi, Ni, bias, ctx);
    }
    // With a bias primed, or a residual in C, the kernel must accumulate.
    const float beta = (bias != nullptr || attrs.residual_pre_activation) ? 1.0f : attrs.beta;
    const auto gemm = [&](int64_t bi) {
        int64_t a_off, b_off, c_off;
        batch.offsets(bi, output, a_off, b_off, c_off);
        arch::matmul_naive(Mi, Ni, Ki, a_base + a_off, lda, attrs.transpose_a, b_base + b_off, ldb,
                           attrs.transpose_b, c_base + c_off, ldc, attrs.alpha, beta);
    };

    if (batch.total > 1 && batch.total * M * N * K >= NAIVE_PARALLEL_MACS) {
        ctx.cpu.run(0, batch.total, gemm);
    } else {
        for (int64_t bi = 0; bi < batch.total; ++bi) {
            gemm(bi);
        }
    }

    apply_epilogue_pass<T>(attrs, output, batch, c_base, ldc, Mi, Ni);
}

} // namespace

bool matmul_small_fp(const MatMulAttributes& attrs, TensorView& output,
                     std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];
    const DataType dt_a = a.data_type();
    const bool float_a = dt_a == DataType::f32 || dt_a == DataType::f16;
    if (!float_a || b.data_type() != dt_a) {
        return false;
    }
    // A NEON panel blob is not a row-major matrix, so it goes to the tiled
    // route. The SME blob is: inside one 64-column block it is plain [K][64]
    // row-major, which is exactly what the GEMV reads — see matmul_gemv.
    const bool sme_blob = b.layout() == TensorLayout::PackedBSme;
    if (is_packed_matmul_b(b.layout()) && !sme_blob) {
        return false;
    }

    const auto [M, N, K] = matmul_dims(attrs, a, b);

    // The naive sweep reads B row by row and has no blocked form; a blob goes
    // to the GEMV below (M <= 4) or to the SME route.
    if (gemm_is_small(M, N, K) && !sme_blob) {
        if (dt_a == DataType::f32) {
            matmul_naive_small<float>(attrs, output, inputs, ctx);
        } else {
            matmul_naive_small<half>(attrs, output, inputs, ctx);
        }
        return true;
    }

    // A transposed A against a transposed B stays on the packed route — that
    // pair is the one layout the direct kernel cannot read.
    if (M <= GEMV_M_MAX && !(attrs.transpose_a && attrs.transpose_b)) {
        if (dt_a == DataType::f32) {
            matmul_gemv<float>(attrs, output, inputs, ctx);
        } else {
            matmul_gemv<half>(attrs, output, inputs, ctx);
        }
        return true;
    }

    return false;
}

} // namespace nnops::backend::cpu
