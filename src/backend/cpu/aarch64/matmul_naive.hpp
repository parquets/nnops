#pragma once
/// @file matmul_naive.hpp
/// @brief AArch64 NEON direct kernels: small GEMMs and GEMV, no packing.
///
/// matmul_naive is a plain triple loop with the n axis vectorized 4-wide: per
/// row of A one a[m,k] is broadcast and FMA'd across a 4-lane row of B. No
/// packing, tiling, blocking or threading — below GEMM_FAST_PATH_THRESHOLD MACs
/// those setups cost more than the arithmetic they save. fp16 is widened to f32
/// to accumulate and narrowed on write-back.
///
/// The rows loop wants B as [K][N]; a [N][K] B is transposed into a stack
/// scratch first, and a transposed A only changes the stride the scalar
/// broadcast walks. N below `lanes` takes dots() instead: one dot product per
/// output, which needs both operands unit-stride in k, so an operand stored
/// [K][*] is copied first. An operand past scratch_elems drops the call to the
/// scalar fallback.
///
/// matmul_gemv covers the small-M extreme (M <= 4 at any size), reading B once
/// — a pack would write and re-read the whole operand for a handful of rows.
/// Two traversals, picked by B's byte size against gemv_stream_bytes: cached B
/// is reread per n-tile for free, so the register tile keeps a whole k sweep in
/// registers and its B load is shared by all M rows; B out of cache cannot be
/// reread, and a tile-wide walk would visit it in ldb-sized strides, so the
/// streaming kernel puts k outermost and reads B once in address order — one
/// extra L1 load/store per unrolled group of k, free while B itself is the wall.
///
/// A [N][K] B takes gemv_tn_rows, which reads B once in order and holds
/// row-major A only: the driver keeps transposed A against transposed B off
/// this route.

#include <arm_neon.h>
#include "nnops/detail/half.hpp"

namespace nnops::backend::cpu::aarch64 {

/// GEMV n-block: the narrow slice of N one call owns — the driver's thread
/// granularity, not the kernel's tile.
constexpr int gemv_block = 1024;

/// Output rows one GEMV call handles; the driver's M cutoff must not exceed it.
constexpr int gemv_max_m = 4;

namespace naive_detail {

/// Elements one operand may transpose into the stack scratch: the small-GEMM
/// MAC budget, so either a [N][K] B or a [M][K] A fits.
constexpr int64_t scratch_elems = 1024;

/// NEON vector width in floats; gemm() takes dots() below this N.
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
        // ta: A is [K][M], so row m is column m with stride lda.
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
            if (alpha != 1.0f) { acc = vmulq_n_f32(acc, alpha); }
            if (beta != 0.0f) {
                acc = vfmaq_n_f32(acc, load4(c_row + n), beta);
            }
            store4(c_row + n, acc);
        }
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
            float32x4_t acc = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k + lanes <= K; k += lanes) {
                acc = vfmaq_f32(acc, load4(a_row + k), load4(b_row + k));
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
        // tb: transpose the [N][K] B into [K][N] scratch first.
        if (!tb) {
            rows<T>(M, N, K, a, lda, ta, b, ldb, c, ldc, alpha, beta);
            return;
        }
        if (static_cast<int64_t>(N) * K > scratch_elems) {
            scalar_gemm<T>(M, N, K, a, lda, ta, b, ldb, tb, c, ldc, alpha, beta);
            return;
        }
        alignas(16) T bt[scratch_elems];
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) { bt[k * N + n] = b[n * ldb + k]; }
        }
        rows<T>(M, N, K, a, lda, ta, bt, N, c, ldc, alpha, beta);
        return;
    }

    // Narrow N: dots need both operands unit-stride in k, so a [K][*] operand
    // is copied first.
    if (static_cast<int64_t>(M) * K > scratch_elems ||
        static_cast<int64_t>(N) * K > scratch_elems) {
        scalar_gemm<T>(M, N, K, a, lda, ta, b, ldb, tb, c, ldc, alpha, beta);
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
    dots<T>(M, N, K, ak, akstride, bk, bkstride, c, ldc, alpha, beta);
}

// ---- GEMV (M small) ----------------------------------------------------

/// Accumulator vectors per n-tile, by row count: M * vecs plus one B vector and
/// the A broadcasts stay inside the 32 NEON registers. A tile is vecs * lanes
/// columns wide, so each k contributes a 24-64 element run of B.
template <int M>
constexpr int gemv_vecs = (M <= 1) ? 16 : (M == 2 ? 12 : (M == 3 ? 8 : 6));

/// One n-tile of the [K][N] B kernel, exactly @p V vectors wide: the whole k
/// sweep lives inside it, so the M*V accumulators never leave registers.
template <typename T, int M, int V>
inline void gemv_nn_tile(int K,
                         const T* a, int lda, bool ta,
                         const T* b, int ldb,
                         T* c, int ldc, float alpha, float beta, int j0) noexcept {
    float32x4_t acc[M][V];
    for (int m = 0; m < M; ++m) {
        for (int v = 0; v < V; ++v) { acc[m][v] = vdupq_n_f32(0.0f); }
    }

    for (int k = 0; k < K; ++k) {
        const T* b_row = b + static_cast<int64_t>(k) * ldb + j0;
        float av[M];
        for (int m = 0; m < M; ++m) {
            av[m] = to_f32(a + (ta ? static_cast<int64_t>(k) * lda + m
                                   : static_cast<int64_t>(m) * lda + k));
        }
        for (int v = 0; v < V; ++v) {
            const float32x4_t bv = load4(b_row + v * lanes);
            for (int m = 0; m < M; ++m) {
                acc[m][v] = vfmaq_n_f32(acc[m][v], bv, av[m]);
            }
        }
    }

    for (int m = 0; m < M; ++m) {
        for (int v = 0; v < V; ++v) {
            T* dst = c + static_cast<int64_t>(m) * ldc + j0 + v * lanes;
            float32x4_t out = (alpha != 1.0f) ? vmulq_n_f32(acc[m][v], alpha)
                                              : acc[m][v];
            store4(dst, (beta != 0.0f) ? vfmaq_n_f32(out, load4(dst), beta) : out);
        }
    }
}

/// C[m][n0:n1] += A[m][:] · B[:][n] for a row-major [K][N] B, k inner within
/// each register-sized n-tile. For a B that is reread per tile out of cache.
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

/// Columns the streaming kernel accumulates over in one pass, by row count:
/// sized so the accumulator is an L1-resident 32-64 KB.
template <int M>
constexpr int gemv_stream_cap = (M <= 1) ? 8192 : 4096;

/// k rows one accumulator load/store pair covers. At M = 1 one row of f32 is
/// the wall, so the depth stays 1; f16 takes 2 there because the widening is
/// what costs, and M >= 2 takes 4 as accumulator traffic starts to matter.
template <typename T, int M>
constexpr int gemv_stream_unroll = (M <= 1) ? (sizeof(T) == 2 ? 2 : 1) : 4;

/// B bytes above which the register tile is given up for the streaming kernel:
/// 24 MB, 1.5x the P-core L2. The kernels cross over near the cache size, and
/// an E-core (smaller L2) only makes the streaming side more attractive.
constexpr int64_t gemv_stream_bytes = 24 * 1024 * 1024;

inline bool gemv_streams(int64_t b_bytes) noexcept { return b_bytes > gemv_stream_bytes; }

/// C[m][n0:n1] += A[m][:] · B[:][n] for a row-major [K][N] B, k outer: the
/// accumulator is an L1 slab spanning cap-wide slices of N, so each k reads a
/// long contiguous run of B and the operand is walked once in address order.
template <typename T, int M>
inline void gemv_nn_stream(int K,
                           const T* a, int lda, bool ta,
                           const T* b, int ldb,
                           T* c, int ldc, float alpha, float beta,
                           int n0, int n1) noexcept {
    constexpr int CAP = gemv_stream_cap<M>;
    constexpr int U = gemv_stream_unroll<T, M>;
    alignas(16) float acc[gemv_max_m][CAP];

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
                float32x4_t bv[U];
                for (int u = 0; u < U; ++u) { bv[u] = load4(b_row[u] + j); }
                for (int m = 0; m < M; ++m) {
                    float* p = acc[m] + j;
                    float32x4_t s = vld1q_f32(p);
                    for (int u = 0; u < U; ++u) { s = vfmaq_n_f32(s, bv[u], av[m][u]); }
                    vst1q_f32(p, s);
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
        for (; k < K; ++k) {
            const T* b_row = b + static_cast<int64_t>(k) * ldb + j0;
            float av[M];
            for (int m = 0; m < M; ++m) {
                av[m] = to_f32(a + (ta ? static_cast<int64_t>(k) * lda + m
                                       : static_cast<int64_t>(m) * lda + k));
            }
            int j = 0;
            for (; j + lanes <= nb; j += lanes) {
                const float32x4_t bv = load4(b_row + j);
                for (int m = 0; m < M; ++m) {
                    float* p = acc[m] + j;
                    vst1q_f32(p, vfmaq_n_f32(vld1q_f32(p), bv, av[m]));
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

/// Independent accumulators per output row for the [N][K] B kernel: one FMA per
/// k step, so its latency hides behind its siblings.
template <int M>
constexpr int gemv_tn_unroll = (M <= 1) ? 8 : (M == 2 ? 4 : 2);

/// C[m][n0:n1] += A[m][:] · B[n][:] for a [N][K] B: U k-vector accumulators
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
        float32x4_t acc[M][U];
        for (int m = 0; m < M; ++m) {
            for (int u = 0; u < U; ++u) { acc[m][u] = vdupq_n_f32(0.0f); }
        }

        int k = 0;
        for (; k + U * lanes <= K; k += U * lanes) {
            float32x4_t bv[U];
            for (int u = 0; u < U; ++u) { bv[u] = load4(b_row + k + u * lanes); }
            for (int m = 0; m < M; ++m) {
                const T* a_row = a + static_cast<int64_t>(m) * lda;
                for (int u = 0; u < U; ++u) {
                    acc[m][u] = vfmaq_f32(acc[m][u], load4(a_row + k + u * lanes), bv[u]);
                }
            }
        }

        for (int m = 0; m < M; ++m) {
            float32x4_t v = acc[m][0];
            for (int u = 1; u < U; ++u) { v = vaddq_f32(v, acc[m][u]); }
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

/// Width of the N slice one GEMV call owns. Streaming accumulates over exactly
/// one slice, so it wants a wide one; the register tile retiles internally and
/// keeps the narrow gemv_block, leaving the pool more blocks to spread.
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

/// C[m][n_begin:n_end] = alpha·(A·B) + beta·C for M <= gemv_max_m, no packing.
/// Every layout except a transposed A against a transposed B.
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

}  // namespace nnops::backend::cpu::aarch64
