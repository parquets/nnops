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
/// B once, so no pack is ever justified, and it trades the two traversals of a
/// given B (see the section comment below) on the operand's size.
///
/// fp16 loads/stores go through F16C, but accumulation stays f32.

#include <immintrin.h>
#include "nnops/detail/half.hpp"

namespace nnops::backend::cpu::x86_64 {

/// GEMV n-block: the slice of N one call owns. This is the driver's unit of
/// work — the granularity it hands to a thread — not the kernel's tile.
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
                        T* c, int ldc, float alpha, float beta) noexcept {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += to_f32(a + (ta ? k * lda + m : m * lda + k)) *
                       to_f32(b + (tb ? n * ldb + k : k * ldb + n));
            }
            if (alpha != 1.0f) { acc *= alpha; }
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
                 T* c, int ldc, float alpha, float beta) noexcept {
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
            if (alpha != 1.0f) { acc = _mm256_mul_ps(acc, _mm256_set1_ps(alpha)); }
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
            if (alpha != 1.0f) { acc *= alpha; }
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
                 T* c, int ldc, float alpha, float beta) noexcept {
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
            if (alpha != 1.0f) { sum *= alpha; }
            store1(c_row + n, (beta != 0.0f) ? sum + beta * to_f32(c_row + n) : sum);
        }
    }
}

template <typename T>
inline void gemm(int M, int N, int K,
                 const T* a, int lda, bool ta,
                 const T* b, int ldb, bool tb,
                 T* c, int ldc, float alpha, float beta) noexcept {
    if (N >= lanes) {
        // Wide N: sweep the n axis; a [N][K] B is turned into [K][N] first.
        if (!tb) {
            rows<T>(M, N, K, a, lda, ta, b, ldb, c, ldc, alpha, beta);
            return;
        }
        if (static_cast<int64_t>(N) * K > scratch_elems) {
            scalar_gemm<T>(M, N, K, a, lda, ta, b, ldb, tb, c, ldc, alpha, beta);
            return;
        }
        alignas(32) T bt[scratch_elems];
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) { bt[k * N + n] = b[n * ldb + k]; }
        }
        rows<T>(M, N, K, a, lda, ta, bt, N, c, ldc, alpha, beta);
        return;
    }

    // Narrow N: dot products need both operands contiguous in k, which is the
    // opposite layout, so an operand stored the k-minor way is copied first.
    if (static_cast<int64_t>(M) * K > scratch_elems ||
        static_cast<int64_t>(N) * K > scratch_elems) {
        scalar_gemm<T>(M, N, K, a, lda, ta, b, ldb, tb, c, ldc, alpha, beta);
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
    dots<T>(M, N, K, ak, akstride, bk, bkstride, c, ldc, alpha, beta);
}

// ---- GEMV (M small) ----------------------------------------------------
//
// The packed route sweeps all of B twice (write the packed copy, then read it
// back) for a handful of output rows. These kernels read B once instead.
//
// Two traversals, because the answer depends on where B lives. Cached B (up to
// L2) is reread per n-tile for free, so the register tile keeps the whole k
// sweep of one 16-64 column tile in registers and pays nothing per FMA beyond
// one B load — and that load is shared by all M rows. B out of cache cannot be
// reread, and a tile-wide walk would then visit it in ldb-sized strides, which
// reaches a fraction of the memory bandwidth; there the accumulator goes back
// into L1 and k becomes the outer loop, so B is read once in address order.
// The L1 reload/store pair that costs is only paid per unrolled group of k.

/// Accumulator vectors per n-tile, by row count: M * vecs stays inside the 16
/// YMM registers with one B vector in flight. A tile is vecs * lanes columns
/// wide, so each k contributes a 24-96 element run of B.
template <int M>
constexpr int gemv_vecs = (M <= 1) ? 12 : (M == 2 ? 6 : (M == 3 ? 4 : 3));

/// One n-tile of the [K][N] B kernel: C[m][j0:j0+V*lanes] += A[m][:] · B[:][j].
/// The whole k sweep lives inside the tile, so the M*V accumulators never leave
/// registers and B is loaded once per element.
template <typename T, int M, int V>
inline void gemv_nn_tile(int K,
                         const T* a, int lda, bool ta,
                         const T* b, int ldb,
                         T* c, int ldc, float alpha, float beta, int j0) noexcept {
    __m256 acc[M][V];
    for (int m = 0; m < M; ++m) {
        for (int v = 0; v < V; ++v) { acc[m][v] = _mm256_setzero_ps(); }
    }

    for (int k = 0; k < K; ++k) {
        const T* b_row = b + static_cast<int64_t>(k) * ldb + j0;
        float av[M];
        for (int m = 0; m < M; ++m) {
            av[m] = to_f32(a + (ta ? static_cast<int64_t>(k) * lda + m
                                   : static_cast<int64_t>(m) * lda + k));
        }
        for (int v = 0; v < V; ++v) {
            const __m256 bv = load8(b_row + v * lanes);
            for (int m = 0; m < M; ++m) {
                acc[m][v] = _mm256_fmadd_ps(_mm256_set1_ps(av[m]), bv, acc[m][v]);
            }
        }
    }

    for (int m = 0; m < M; ++m) {
        for (int v = 0; v < V; ++v) {
            T* dst = c + static_cast<int64_t>(m) * ldc + j0 + v * lanes;
            __m256 out = (alpha != 1.0f)
                ? _mm256_mul_ps(acc[m][v], _mm256_set1_ps(alpha))
                : acc[m][v];
            store8(dst, (beta != 0.0f) ? _mm256_fmadd_ps(_mm256_set1_ps(beta), load8(dst),
                                                         out)
                                       : out);
        }
    }
}

/// C[m][n0:n1] += A[m][:] · B[:][n] for a row-major [K][N] B. n is cut into
/// register-sized tiles; within a tile k is the inner loop, so the accumulator
/// never leaves registers and B is loaded once per element. Built for a B that
/// is reread per tile out of cache.
/// A transposed A only changes the stride the scalar broadcast walks.
template <typename T, int M>
inline void gemv_nn_rows(int N, int K,
                         const T* a, int lda, bool ta,
                         const T* b, int ldb,
                         T* c, int ldc, float alpha, float beta,
                         int n0, int n1) noexcept {
    (void)N;
    constexpr int V = gemv_vecs<M>;

    int j = n0;
    for (; j + V * lanes <= n1; j += V * lanes) {
        gemv_nn_tile<T, M, V>(K, a, lda, ta, b, ldb, c, ldc, alpha, beta, j);
    }
    // Column tail: single-vector tiles, then the scalars below one vector.
    for (; j + lanes <= n1; j += lanes) {
        gemv_nn_tile<T, M, 1>(K, a, lda, ta, b, ldb, c, ldc, alpha, beta, j);
    }
    for (; j < n1; ++j) {
        float sum[gemv_max_m] = {};
        for (int k = 0; k < K; ++k) {
            const float bj = to_f32(b + static_cast<int64_t>(k) * ldb + j);
            for (int m = 0; m < M; ++m) {
                sum[m] += to_f32(a + (ta ? static_cast<int64_t>(k) * lda + m
                                         : static_cast<int64_t>(m) * lda + k)) * bj;
            }
        }
        for (int m = 0; m < M; ++m) {
            T* dst = c + static_cast<int64_t>(m) * ldc + j;
            const float v = (alpha != 1.0f) ? sum[m] * alpha : sum[m];
            store1(dst, (beta != 0.0f) ? v + beta * to_f32(dst) : v);
        }
    }
}

/// Columns the streaming kernel accumulates over in one pass, by row count,
/// sized so the accumulator is an L1-resident 32-64 KB and each k contributes a
/// long contiguous run of B.
template <int M>
constexpr int gemv_stream_cap = (M <= 1) ? 8192 : 4096;

/// k rows one accumulator load/store pair covers. Both extra factors cost
/// nothing while B is the wall (one row of f32 at M=1); they pay as soon as the
/// accumulator traffic or the widening of a narrow B load becomes the limit.
template <typename T, int M>
constexpr int gemv_stream_unroll = (M <= 1) ? (sizeof(T) == 2 ? 2 : 1) : 4;

/// B bytes above which the register tile is given up for the streaming kernel.
/// Below this the operand is cache-resident, so reusing each B element across
/// rows from a register tile beats streaming; above it B is streaming from
/// DRAM and only an in-order read reaches the bandwidth. The cutoff is 1.5x the
/// P-core L2: the two kernels cross over near the cache size, and an E-core
/// (smaller L2) only makes the streaming side more attractive.
constexpr int64_t gemv_stream_bytes = 24 * 1024 * 1024;

inline bool gemv_streams(int64_t b_bytes) noexcept { return b_bytes > gemv_stream_bytes; }

/// C[m][n0:n1] += A[m][:] · B[:][n] for a row-major [K][N] B, k outer: the
/// accumulator spans whole cap-wide slices of N in L1, so each k reads a long
/// contiguous run of B and the operand is walked once in address order. The
/// accumulator is reloaded and stored per k — an extra L1 access per FMA, which
/// is free while B itself is coming from memory.
template <typename T, int M>
inline void gemv_nn_stream(int K,
                           const T* a, int lda, bool ta,
                           const T* b, int ldb,
                           T* c, int ldc, float alpha, float beta,
                           int n0, int n1) noexcept {
    constexpr int CAP = gemv_stream_cap<M>;
    constexpr int U = gemv_stream_unroll<T, M>;
    alignas(64) float acc[gemv_max_m][CAP];

    for (int j0 = n0; j0 < n1; j0 += CAP) {
        const int nb = (n1 - j0 < CAP) ? (n1 - j0) : CAP;
        for (int m = 0; m < M; ++m) {
            for (int j = 0; j < nb; ++j) { acc[m][j] = 0.0f; }
        }

        int k = 0;
        for (; k + U <= K; k += U) {
            const T* b_row[U];
            float av[M][U];
            for (int u = 0; u < U; ++u) {
                b_row[u] = b + static_cast<int64_t>(k + u) * ldb + j0;
                for (int m = 0; m < M; ++m) {
                    av[m][u] = to_f32(a + (ta ? static_cast<int64_t>(k + u) * lda + m
                                              : static_cast<int64_t>(m) * lda + k + u));
                }
            }
            int j = 0;
            for (; j + lanes <= nb; j += lanes) {
                __m256 bv[U];
                for (int u = 0; u < U; ++u) { bv[u] = load8(b_row[u] + j); }
                for (int m = 0; m < M; ++m) {
                    float* p = acc[m] + j;
                    __m256 s = _mm256_loadu_ps(p);
                    for (int u = 0; u < U; ++u) { s = _mm256_fmadd_ps(_mm256_set1_ps(av[m][u]), bv[u], s); }
                    _mm256_storeu_ps(p, s);
                }
            }
            for (; j < nb; ++j) {
                for (int m = 0; m < M; ++m) {
                    for (int u = 0; u < U; ++u) {
                        acc[m][j] += av[m][u] * to_f32(b_row[u] + j);
                    }
                }
            }
        }
        // Rows left below the unroll depth.
        for (; k < K; ++k) {
            const T* b_row = b + static_cast<int64_t>(k) * ldb + j0;
            float av[M];
            for (int m = 0; m < M; ++m) {
                av[m] = to_f32(a + (ta ? static_cast<int64_t>(k) * lda + m
                                       : static_cast<int64_t>(m) * lda + k));
            }
            int j = 0;
            for (; j + lanes <= nb; j += lanes) {
                const __m256 bv = load8(b_row + j);
                for (int m = 0; m < M; ++m) {
                    float* p = acc[m] + j;
                    _mm256_storeu_ps(p, _mm256_fmadd_ps(_mm256_set1_ps(av[m]), bv,
                                                       _mm256_loadu_ps(p)));
                }
            }
            for (; j < nb; ++j) {
                const float bj = to_f32(b_row + j);
                for (int m = 0; m < M; ++m) { acc[m][j] += av[m] * bj; }
            }
        }

        for (int m = 0; m < M; ++m) {
            T* c_row = c + static_cast<int64_t>(m) * ldc;
            for (int j = 0; j < nb; ++j) {
                T* dst = c_row + j0 + j;
                const float v = (alpha != 1.0f) ? acc[m][j] * alpha : acc[m][j];
                store1(dst, (beta != 0.0f) ? v + beta * to_f32(dst) : v);
            }
        }
    }
}

/// Independent accumulators per output row for the [N][K] B kernel. Each k step
/// carries one FMA into one of them, so the FMA latency hides behind its
/// siblings instead of stalling the row's single dependency chain.
template <int M>
constexpr int gemv_tn_unroll = (M <= 1) ? 4 : 2;

/// C[m][n0:n1] += A[m][:] · B[n][:] for a [N][K] B: one k-vector accumulator
/// per output row, B read once and in order. Holds row-major A only.
template <typename T, int M>
inline void gemv_tn_rows(int N, int K,
                         const T* a, int lda,
                         const T* b, int ldb,
                         T* c, int ldc, float alpha, float beta,
                         int n0, int n1) noexcept {
    (void)N;
    constexpr int U = gemv_tn_unroll<M>;

    for (int n = n0; n < n1; ++n) {
        const T* b_row = b + static_cast<int64_t>(n) * ldb;
        __m256 acc[M][U];
        for (int m = 0; m < M; ++m) {
            for (int u = 0; u < U; ++u) { acc[m][u] = _mm256_setzero_ps(); }
        }

        int k = 0;
        for (; k + U * lanes <= K; k += U * lanes) {
            __m256 bv[U];
            for (int u = 0; u < U; ++u) { bv[u] = load8(b_row + k + u * lanes); }
            for (int m = 0; m < M; ++m) {
                const T* a_row = a + static_cast<int64_t>(m) * lda;
                for (int u = 0; u < U; ++u) {
                    acc[m][u] = _mm256_fmadd_ps(load8(a_row + k + u * lanes), bv[u],
                                                acc[m][u]);
                }
            }
        }

        for (int m = 0; m < M; ++m) {
            __m256 v = acc[m][0];
            for (int u = 1; u < U; ++u) { v = _mm256_add_ps(v, acc[m][u]); }
            float sum = hsum(v);
            for (int t = k; t < K; ++t) {
                sum += to_f32(a + static_cast<int64_t>(m) * lda + t) * to_f32(b_row + t);
            }
            T* dst = c + m * ldc + n;
            const float f = (alpha != 1.0f) ? sum * alpha : sum;
            store1(dst, (beta != 0.0f) ? f + beta * to_f32(dst) : f);
        }
    }
}

/// Dispatch a row count of 1..gemv_max_m onto the templated kernels, choosing
/// the traversal by operand size: cache-resident B favours the register tile,
/// DRAM-resident B the streaming kernel.
template <typename T>
inline void gemv_nn(int M, int N, int K,
                    const T* a, int lda, bool ta,
                    const T* b, int ldb,
                    T* c, int ldc, float alpha, float beta,
                    int n0, int n1) noexcept {
    if (gemv_streams(static_cast<int64_t>(N) * K * static_cast<int64_t>(sizeof(T)))) {
        switch (M) {
            case 1: gemv_nn_stream<T, 1>(K, a, lda, ta, b, ldb, c, ldc, alpha, beta, n0, n1); break;
            case 2: gemv_nn_stream<T, 2>(K, a, lda, ta, b, ldb, c, ldc, alpha, beta, n0, n1); break;
            case 3: gemv_nn_stream<T, 3>(K, a, lda, ta, b, ldb, c, ldc, alpha, beta, n0, n1); break;
            default: gemv_nn_stream<T, gemv_max_m>(K, a, lda, ta, b, ldb, c, ldc, alpha, beta,
                                                   n0, n1); break;
        }
        return;
    }
    switch (M) {
        case 1: gemv_nn_rows<T, 1>(N, K, a, lda, ta, b, ldb, c, ldc, alpha, beta, n0, n1); break;
        case 2: gemv_nn_rows<T, 2>(N, K, a, lda, ta, b, ldb, c, ldc, alpha, beta, n0, n1); break;
        case 3: gemv_nn_rows<T, 3>(N, K, a, lda, ta, b, ldb, c, ldc, alpha, beta, n0, n1); break;
        default: gemv_nn_rows<T, gemv_max_m>(N, K, a, lda, ta, b, ldb, c, ldc, alpha, beta,
                                             n0, n1); break;
    }
}

template <typename T>
inline void gemv_tn(int M, int N, int K,
                    const T* a, int lda,
                    const T* b, int ldb,
                    T* c, int ldc, float alpha, float beta,
                    int n0, int n1) noexcept {
    switch (M) {
        case 1: gemv_tn_rows<T, 1>(N, K, a, lda, b, ldb, c, ldc, alpha, beta, n0, n1); break;
        case 2: gemv_tn_rows<T, 2>(N, K, a, lda, b, ldb, c, ldc, alpha, beta, n0, n1); break;
        case 3: gemv_tn_rows<T, 3>(N, K, a, lda, b, ldb, c, ldc, alpha, beta, n0, n1); break;
        default: gemv_tn_rows<T, gemv_max_m>(N, K, a, lda, b, ldb, c, ldc, alpha, beta,
                                             n0, n1); break;
    }
}

}  // namespace naive_detail

/// Width of the N slice one GEMV call owns — the driver's thread granularity.
/// The streaming kernel accumulates over exactly one slice, so it wants a wide
/// one; the register tile retiles internally, so it keeps the narrow slice and
/// leaves the pool more blocks to spread.
template <typename T>
inline int gemv_slice(int M, int64_t N, int64_t K) noexcept {
    if (!naive_detail::gemv_streams(N * K * static_cast<int64_t>(sizeof(T)))) {
        return gemv_block;
    }
    switch (M) {
        case 1: return naive_detail::gemv_stream_cap<1>;
        case 2: return naive_detail::gemv_stream_cap<2>;
        default: return naive_detail::gemv_stream_cap<gemv_max_m>;
    }
}

/// C[m][n_begin:n_end] = A·B + beta·C for M <= gemv_max_m, no packing. Handles
/// every layout except a transposed A against a transposed B (the caller keeps
/// that on the packed route).
inline void matmul_gemv(int M, int N, int K,
                        const float* a, int lda, bool trans_a,
                        const float* b, int ldb, bool trans_b,
                        float* c, int ldc, float alpha, float beta,
                        int n_begin, int n_end) noexcept {
    if (trans_b) {
        naive_detail::gemv_tn<float>(M, N, K, a, lda, b, ldb, c, ldc, alpha, beta,
                                     n_begin, n_end);
    } else {
        naive_detail::gemv_nn<float>(M, N, K, a, lda, trans_a, b, ldb, c, ldc, alpha, beta,
                                     n_begin, n_end);
    }
}

inline void matmul_gemv(int M, int N, int K,
                        const half* a, int lda, bool trans_a,
                        const half* b, int ldb, bool trans_b,
                        half* c, int ldc, float alpha, float beta,
                        int n_begin, int n_end) noexcept {
    if (trans_b) {
        naive_detail::gemv_tn<half>(M, N, K, a, lda, b, ldb, c, ldc, alpha, beta,
                                    n_begin, n_end);
    } else {
        naive_detail::gemv_nn<half>(M, N, K, a, lda, trans_a, b, ldb, c, ldc, alpha, beta,
                                    n_begin, n_end);
    }
}

/// C = alpha·(A·B) + beta·C for a small M×N×K, operands in either transpose
/// layout. The epilogue is the caller's.
inline void matmul_naive(int M, int N, int K,
                         const float* a, int lda, bool trans_a,
                         const float* b, int ldb, bool trans_b,
                         float* c, int ldc, float alpha, float beta) noexcept {
    naive_detail::gemm<float>(M, N, K, a, lda, trans_a, b, ldb, trans_b, c, ldc,
                              alpha, beta);
}

inline void matmul_naive(int M, int N, int K,
                         const half* a, int lda, bool trans_a,
                         const half* b, int ldb, bool trans_b,
                         half* c, int ldc, float alpha, float beta) noexcept {
    naive_detail::gemm<half>(M, N, K, a, lda, trans_a, b, ldb, trans_b, c, ldc,
                             alpha, beta);
}

}  // namespace nnops::backend::cpu::x86_64
