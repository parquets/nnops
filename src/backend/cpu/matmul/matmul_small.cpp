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
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"   // apply_epilogue
#include "nnops/detail/half.hpp"

#include <algorithm>
#include <type_traits>

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
void prime_bias_pass(const MatMulAttributes& attrs, TensorView& output,
                     const BatchWalk& batch, T* c_base, int ldc,
                     int M, int N, const T* bias, const ComputeContext& ctx)
{
    const auto prime = [&](int64_t bi) {
        int64_t a_off, b_off, c_off;
        batch.offsets(bi, output, a_off, b_off, c_off);
        tile_scale_add_bias(c_base + c_off, ldc, attrs.beta, bias, M, N);
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
void apply_epilogue_pass(const MatMulAttributes& attrs, TensorView& output,
                         const BatchWalk& batch, T* c_base, int ldc,
                         int M, int N)
{
    const Epilogue& ep = attrs.epilogue;
    const float inf = std::numeric_limits<float>::infinity();
    const bool clamp = ep.min_clip > -inf || ep.max_clip < inf;
    if (ep.type == EpilogueActivateType::None && !clamp) { return; }
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

/// M <= GEMV_M_MAX: stream B once through the arch's GEMV kernel. Blocks are
/// split over N on a single batch element, else over the batch.
template <typename T>
void matmul_gemv(const MatMulAttributes& attrs, TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx)
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
    const int blocks = (Ni + span - 1) / span;

    const T* a_base = a.ptr<T>();
    const T* b_base = b.ptr<T>();
    T* c_base = output.ptr<T>();

    const BatchWalk batch(a, b);
    // A bias primes C with beta·C_old + bias; the kernel then adds the product.
    const T* bias = inputs.size() > 2 ? inputs[2].ptr<T>() : nullptr;
    if (bias != nullptr) {
        prime_bias_pass<T>(attrs, output, batch, c_base, ldc, Mi, Ni, bias, ctx);
    }
    const float beta = (bias != nullptr) ? 1.0f : attrs.beta;
    const auto block = [&](int64_t bi, int nb) {
        int64_t a_off, b_off, c_off;
        batch.offsets(bi, output, a_off, b_off, c_off);
        const int n0 = static_cast<int>(nb) * span;
        const int n1 = std::min(n0 + span, Ni);
        arch::matmul_gemv(Mi, Ni, Ki,
                          a_base + a_off, lda, attrs.transpose_a,
                          b_base + b_off, ldb, attrs.transpose_b,
                          c_base + c_off, ldc, attrs.alpha, beta, n0, n1);
    };

    if (batch.total > 1) {
        const auto element = [&](int64_t bi) {
            for (int nb = 0; nb < blocks; ++nb) { block(bi, nb); }
        };
        if (batch.total * M * N * K >= NAIVE_PARALLEL_MACS) {
            ctx.cpu.run(0, batch.total, element);
        } else {
            for (int64_t bi = 0; bi < batch.total; ++bi) { element(bi); }
        }
    } else if (M * N * K >= NAIVE_PARALLEL_MACS) {
        ctx.cpu.run(0, blocks, [&](int64_t nb) { block(0, nb); });
    } else {
        for (int nb = 0; nb < blocks; ++nb) { block(0, nb); }
    }

    apply_epilogue_pass<T>(attrs, output, batch, c_base, ldc, Mi, Ni);
}

template <typename T>
void matmul_naive_small(const MatMulAttributes& attrs, TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx)
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
    const T* b_base = b.ptr<T>();
    T* c_base = output.ptr<T>();

    const BatchWalk batch(a, b);
    // Same accumulator-init as the GEMV route: prime, then accumulate beta = 1.
    const T* bias = inputs.size() > 2 ? inputs[2].ptr<T>() : nullptr;
    if (bias != nullptr) {
        prime_bias_pass<T>(attrs, output, batch, c_base, ldc, Mi, Ni, bias, ctx);
    }
    const float beta = (bias != nullptr) ? 1.0f : attrs.beta;
    const auto gemm = [&](int64_t bi) {
        int64_t a_off, b_off, c_off;
        batch.offsets(bi, output, a_off, b_off, c_off);
        arch::matmul_naive(Mi, Ni, Ki,
                           a_base + a_off, lda, attrs.transpose_a,
                           b_base + b_off, ldb, attrs.transpose_b,
                           c_base + c_off, ldc, attrs.alpha, beta);
    };

    if (batch.total > 1 && batch.total * M * N * K >= NAIVE_PARALLEL_MACS) {
        ctx.cpu.run(0, batch.total, gemm);
    } else {
        for (int64_t bi = 0; bi < batch.total; ++bi) { gemm(bi); }
    }

    apply_epilogue_pass<T>(attrs, output, batch, c_base, ldc, Mi, Ni);
}

}  // namespace

bool matmul_small_fp(const MatMulAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];
    const DataType dt_a = a.data_type();
    const bool float_a = dt_a == DataType::f32 || dt_a == DataType::f16;
    if (!float_a || b.data_type() != dt_a) { return false; }
    // A prepacked RHS is panel-formatted, not a row-major matrix, and both small
    // routes here read B as rows of N: it goes to the tiled route instead.
    if (is_packed_matmul_b(b.layout())) { return false; }

    const auto [M, N, K] = matmul_dims(attrs, a, b);

    if (gemm_is_small(M, N, K)) {
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

}  // namespace nnops::backend::cpu
