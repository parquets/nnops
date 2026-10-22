#pragma once
/// @file matmul_naive.hpp
/// @brief x86_64 AVX2/FMA direct kernels: small GEMMs and GEMV, no packing.
///
/// matmul_naive is a plain triple loop with the n axis vectorized 8-wide: per
/// row of A one a[m,k] is broadcast and FMA'd across an 8-lane row of B. No
/// packing, tiling, blocking or threading — below GEMM_FAST_PATH_THRESHOLD MACs
/// those setups cost more than the arithmetic they save.
///
/// The hot loop wants B as [K][N]; a transposed B ([N][K]) is copied into a
/// stack scratch first. A transposed A only changes the stride the scalar
/// broadcast walks.
///
/// matmul_gemv covers the other small-M extreme (M <= 4 at any size): it reads
/// B once, so no pack is ever justified, but it blocks over n so that each k
/// contributes a contiguous run of B (see the comment at gemv_block).
///
/// fp16 loads/stores go through F16C, but accumulation stays f32.

#include <immintrin.h>
#include "nnops/detail/half.hpp"

namespace nnops::backend::cpu::x86_64 {

/// GEMV n-block: the slice of N one call owns. M rows of f32 accumulators stay
/// in L1 (16 KB at most) and each k contributes a contiguous 4 KB run of B, so
/// the prefetcher still sees sequential data.
constexpr int gemv_block = 1024;

/// Output rows one GEMV call handles; the driver's M cutoff must not exceed it.
constexpr int gemv_max_m = 4;

namespace naive_detail {

/// Elements an operand may transpose into the stack scratch: the small-GEMM
/// MAC budget, so both a [N][K] B and a [M][K] A fit.
constexpr int64_t scratch_elems = 1024;

/// Vector width: below this N the kernel switches to per-output dot products.
constexpr int lanes = 8;

// ---- operand access: one element / eight lanes / one lane write-back ----

inline float to_f32(const float* p) noexcept { return *p; }
inline float to_f32(const half* p) noexcept { return half_to_float(*p); }

inline __m256 load8(const float* p) noexcept { return _mm256_loadu_ps(p); }
inline __m256 load8(const half* p) noexcept {
    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

inline void store8(float* p, __m256 v) noexcept { _mm256_storeu_ps(p, v); }
inline void store8(half* p, __m256 v) noexcept {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p),
                     _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
}

inline void store1(float* p, float v) noexcept { *p = v; }
inline void store1(half* p, float v) noexcept { *p = float_to_half(v); }

inline float hsum(__m256 v) noexcept {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

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
        for (; n + 8 <= N; n += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int k = 0; k < K; ++k) {
                acc = _mm256_fmadd_ps(_mm256_set1_ps(to_f32(a_row + k * a_step)),
                                      load8(b + k * bstride + n), acc);
            }
            if (beta != 0.0f) {
                acc = _mm256_fmadd_ps(_mm256_set1_ps(beta), load8(c_row + n), acc);
            }
            store8(c_row + n, acc);
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
            __m256 acc = _mm256_setzero_ps();
            int k = 0;
            for (; k + lanes <= K; k += lanes) {
                acc = _mm256_fmadd_ps(load8(a_row + k), load8(b_row + k), acc);
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
        alignas(32) T bt[scratch_elems];
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
    alignas(32) T at[scratch_elems];
    alignas(32) T bt[scratch_elems];
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

// ---- GEMV (M small) ----------------------------------------------------
//
// The packed route sweeps all of B twice (write the packed copy, then read it
// back) for a handful of output rows. These kernels read B once instead.

/// C[m][n0:n1] += A[m][:] · B[:][n] for a row-major [K][N] B. k outer, n-block
/// inner: B is read once and in order, and the accumulators ping-pong through
/// L1 — the traversal a GEMV wants (the reverse order walks B with an ldb
/// stride, one cache line at a time, which no prefetcher follows).
/// A transposed A only changes the stride the scalar broadcast walks.
template <typename T>
inline void gemv_nn(int M, int N, int K,
                    const T* a, int lda, bool ta,
                    const T* b, int ldb,
                    T* c, int ldc, float beta,
                    int n0, int n1) noexcept {
    const int nb = n1 - n0;
    alignas(64) float acc[gemv_max_m][gemv_block];
    for (int m = 0; m < M; ++m) {
        for (int j = 0; j < nb; ++j) { acc[m][j] = 0.0f; }
    }

    for (int k = 0; k < K; ++k) {
        const T* b_row = b + static_cast<int64_t>(k) * ldb + n0;
        float av[gemv_max_m];
        for (int m = 0; m < M; ++m) {
            av[m] = to_f32(a + (ta ? static_cast<int64_t>(k) * lda + m
                                   : static_cast<int64_t>(m) * lda + k));
        }
        int j = 0;
        for (; j + lanes <= nb; j += lanes) {
            const __m256 bv = load8(b_row + j);
            for (int m = 0; m < M; ++m) {
                __m256 v = _mm256_loadu_ps(acc[m] + j);
                v = _mm256_fmadd_ps(_mm256_set1_ps(av[m]), bv, v);
                _mm256_storeu_ps(acc[m] + j, v);
            }
        }
        for (; j < nb; ++j) {
            const float bj = to_f32(b_row + j);
            for (int m = 0; m < M; ++m) { acc[m][j] += av[m] * bj; }
        }
    }

    for (int m = 0; m < M; ++m) {
        for (int j = 0; j < nb; ++j) {
            T* dst = c + m * ldc + n0 + j;
            store1(dst, (beta != 0.0f) ? acc[m][j] + beta * to_f32(dst) : acc[m][j]);
        }
    }
}

/// C[m][n0:n1] += A[m][:] · B[n][:] for a [N][K] B: one k-vector accumulator
/// per output row, B read once and in order. Holds row-major A only.
template <typename T>
inline void gemv_tn(int M, int N, int K,
                    const T* a, int lda,
                    const T* b, int ldb,
                    T* c, int ldc, float beta,
                    int n0, int n1) noexcept {
    for (int n = n0; n < n1; ++n) {
        const T* b_row = b + static_cast<int64_t>(n) * ldb;
        __m256 vacc[gemv_max_m];
        for (int m = 0; m < M; ++m) { vacc[m] = _mm256_setzero_ps(); }

        int k = 0;
        for (; k + lanes <= K; k += lanes) {
            const __m256 bv = load8(b_row + k);
            for (int m = 0; m < M; ++m) {
                vacc[m] = _mm256_fmadd_ps(load8(a + static_cast<int64_t>(m) * lda + k),
                                          bv, vacc[m]);
            }
        }
        for (int m = 0; m < M; ++m) {
            float sum = hsum(vacc[m]);
            for (int t = k; t < K; ++t) {
                sum += to_f32(a + static_cast<int64_t>(m) * lda + t) * to_f32(b_row + t);
            }
            T* dst = c + m * ldc + n;
            store1(dst, (beta != 0.0f) ? sum + beta * to_f32(dst) : sum);
        }
    }
}

}  // namespace naive_detail

/// C[m][n_begin:n_end] = A·B + beta·C for M <= gemv_max_m, no packing. Handles
/// every layout except a transposed A against a transposed B (the caller keeps
/// that on the packed route).
inline void matmul_gemv(int M, int N, int K,
                        const float* a, int lda, bool trans_a,
                        const float* b, int ldb, bool trans_b,
                        float* c, int ldc, float beta,
                        int n_begin, int n_end) noexcept {
    if (trans_b) {
        naive_detail::gemv_tn<float>(M, N, K, a, lda, b, ldb, c, ldc, beta, n_begin, n_end);
    } else {
        naive_detail::gemv_nn<float>(M, N, K, a, lda, trans_a, b, ldb, c, ldc, beta,
                                     n_begin, n_end);
    }
}

inline void matmul_gemv(int M, int N, int K,
                        const half* a, int lda, bool trans_a,
                        const half* b, int ldb, bool trans_b,
                        half* c, int ldc, float beta,
                        int n_begin, int n_end) noexcept {
    if (trans_b) {
        naive_detail::gemv_tn<half>(M, N, K, a, lda, b, ldb, c, ldc, beta, n_begin, n_end);
    } else {
        naive_detail::gemv_nn<half>(M, N, K, a, lda, trans_a, b, ldb, c, ldc, beta,
                                    n_begin, n_end);
    }
}

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

}  // namespace nnops::backend::cpu::x86_64
