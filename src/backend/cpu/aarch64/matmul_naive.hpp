#pragma once
/// @file matmul_naive.hpp
/// @brief AArch64 NEON GEMM for shapes below the small-GEMM threshold.
///
/// A plain triple loop with the n axis vectorized 4-wide: per row of A one
/// a[m,k] is broadcast and FMA'd across a 4-lane row of B. No packing, tiling,
/// blocking or threading — below GEMM_FAST_PATH_THRESHOLD MACs those setups cost
/// more than the arithmetic they save.
///
/// The hot loop wants B as [K][N]; a transposed B ([N][K]) is copied into a
/// stack scratch first. A transposed A only changes the stride the scalar
/// broadcast walks.
///
/// fp16 is widened to f32 for the accumulate and narrowed again on write-back.

#include <arm_neon.h>
#include "nnops/detail/half.hpp"

namespace nnops::backend::cpu::aarch64 {

namespace naive_detail {

/// Elements an operand may transpose into the stack scratch: the small-GEMM
/// MAC budget, so both a [N][K] B and a [M][K] A fit.
constexpr int64_t scratch_elems = 1024;

/// Vector width: below this N the kernel switches to per-output dot products.
constexpr int lanes = 4;

// ---- operand access: one element / four lanes / one lane write-back ----

inline float to_f32(const float* p) noexcept { return *p; }
inline float to_f32(const half* p) noexcept { return static_cast<float>(*p); }

inline float32x4_t load4(const float* p) noexcept { return vld1q_f32(p); }
inline float32x4_t load4(const half* p) noexcept { return vcvt_f32_f16(vld1_f16(p)); }

inline void store4(float* p, float32x4_t v) noexcept { vst1q_f32(p, v); }
inline void store4(half* p, float32x4_t v) noexcept { vst1_f16(p, vcvt_f16_f32(v)); }

inline void store1(float* p, float v) noexcept { *p = v; }
inline void store1(half* p, float v) noexcept { *p = static_cast<half>(v); }

inline float hsum(float32x4_t v) noexcept { return vaddvq_f32(v); }

/// Fully scalar fallback: any operand size and layout.
template <typename T>
inline void scalar_gemm(int M, int N, int K,
                        const T* a, int lda, bool ta,
                        const T* b, int ldb, bool tb,
                        T* c, int ldc, float beta) noexcept {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += to_f32(a + (ta ? k * lda + m : m * lda + k)) *
                       to_f32(b + (tb ? n * ldb + k : k * ldb + n));
            }
            T* dst = c + m * ldc + n;
            store1(dst, (beta != 0.0f) ? acc + beta * to_f32(dst) : acc);
        }
    }
}

/// The n-vectorized rows loop; B is [K][N] with row stride @p bstride.
template <typename T>
inline void rows(int M, int N, int K,
                 const T* a, int lda, bool ta,
                 const T* b, int bstride,
                 T* c, int ldc, float beta) noexcept {
    for (int m = 0; m < M; ++m) {
        // A row: m*lda when row-major, plain m when A is stored [K][M].
        const T* a_row = a + (ta ? m : m * lda);
        const int a_step = ta ? lda : 1;
        T* c_row = c + m * ldc;

        int n = 0;
        for (; n + 4 <= N; n += 4) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int k = 0; k < K; ++k) {
                acc = vfmaq_n_f32(acc, load4(b + k * bstride + n),
                                  to_f32(a_row + k * a_step));
            }
            if (beta != 0.0f) {
                acc = vfmaq_n_f32(acc, load4(c_row + n), beta);
            }
            store4(c_row + n, acc);
        }
        // Columns left below the vector width: same loop, one lane at a time.
        for (; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += to_f32(a_row + k * a_step) * to_f32(b + k * bstride + n);
            }
            store1(c_row + n, (beta != 0.0f) ? acc + beta * to_f32(c_row + n) : acc);
        }
    }
}

/// Narrow-N companion to rows(): one dot product per output, vectorized over K.
/// A is [M][K] and B is [N][K], both unit-stride in k.
template <typename T>
inline void dots(int M, int N, int K,
                 const T* a, int astride,
                 const T* b, int bstride,
                 T* c, int ldc, float beta) noexcept {
    for (int m = 0; m < M; ++m) {
        const T* a_row = a + m * astride;
        T* c_row = c + m * ldc;
        for (int n = 0; n < N; ++n) {
            const T* b_row = b + n * bstride;
            float32x4_t acc = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k + lanes <= K; k += lanes) {
                acc = vfmaq_f32(acc, load4(a_row + k), load4(b_row + k));
            }
            float sum = hsum(acc);
            for (; k < K; ++k) { sum += to_f32(a_row + k) * to_f32(b_row + k); }
            store1(c_row + n, (beta != 0.0f) ? sum + beta * to_f32(c_row + n) : sum);
        }
    }
}

template <typename T>
inline void gemm(int M, int N, int K,
                 const T* a, int lda, bool ta,
                 const T* b, int ldb, bool tb,
                 T* c, int ldc, float beta) noexcept {
    if (N >= lanes) {
        // Wide N: sweep the n axis; a [N][K] B is turned into [K][N] first.
        if (!tb) {
            rows<T>(M, N, K, a, lda, ta, b, ldb, c, ldc, beta);
            return;
        }
        if (static_cast<int64_t>(N) * K > scratch_elems) {
            scalar_gemm<T>(M, N, K, a, lda, ta, b, ldb, tb, c, ldc, beta);
            return;
        }
        alignas(16) T bt[scratch_elems];
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) { bt[k * N + n] = b[n * ldb + k]; }
        }
        rows<T>(M, N, K, a, lda, ta, bt, N, c, ldc, beta);
        return;
    }

    // Narrow N: dot products need both operands contiguous in k, which is the
    // opposite layout, so an operand stored the k-minor way is copied first.
    if (static_cast<int64_t>(M) * K > scratch_elems ||
        static_cast<int64_t>(N) * K > scratch_elems) {
        scalar_gemm<T>(M, N, K, a, lda, ta, b, ldb, tb, c, ldc, beta);
        return;
    }
    alignas(16) T at[scratch_elems];
    alignas(16) T bt[scratch_elems];
    const T* ak = a;
    const T* bk = b;
    int akstride = lda;
    int bkstride = ldb;
    if (ta) {
        for (int m = 0; m < M; ++m) {
            for (int k = 0; k < K; ++k) { at[m * K + k] = a[k * lda + m]; }
        }
        ak = at;
        akstride = K;
    }
    if (!tb) {
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) { bt[n * K + k] = b[k * ldb + n]; }
        }
        bk = bt;
        bkstride = K;
    }
    dots<T>(M, N, K, ak, akstride, bk, bkstride, c, ldc, beta);
}

}  // namespace naive_detail

/// C = A·B + beta·C for a small M×N×K, operands in either transpose layout.
/// The epilogue is the caller's.
inline void matmul_naive(int M, int N, int K,
                         const float* a, int lda, bool trans_a,
                         const float* b, int ldb, bool trans_b,
                         float* c, int ldc, float beta) noexcept {
    naive_detail::gemm<float>(M, N, K, a, lda, trans_a, b, ldb, trans_b, c, ldc, beta);
}

inline void matmul_naive(int M, int N, int K,
                         const half* a, int lda, bool trans_a,
                         const half* b, int ldb, bool trans_b,
                         half* c, int ldc, float beta) noexcept {
    naive_detail::gemm<half>(M, N, K, a, lda, trans_a, b, ldb, trans_b, c, ldc, beta);
}

}  // namespace nnops::backend::cpu::aarch64
