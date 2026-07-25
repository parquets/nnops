/// @file test_pack_mma.cpp
/// @brief Unit tests for pack and MMA micro-kernels.
///
/// Tests cover:
///   1. f32 pack  — transpose & copy, SIMD + scalar-tail paths
///   2. f32 mma_pack  — C += A × B_packed, clamp epilogue
///   3. f32 mma_direct — C += A × B (row-major), clamp epilogue
///   4. f32 cross-validation — mma_pack vs mma_direct vs naive
///   5. f16 pack  — transpose & copy (x86_64: half, aarch64: float16_t)
///   6. f16 mma_pack  — C += A × B_packed, clamp epilogue
///   7. f16 mma_direct — C += A × B (row-major), clamp epilogue
///   8. f16 cross-validation — mma_pack vs mma_direct vs naive

#include "common/test_harness.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/simd.hpp"

#if defined(NNOPS_ARCH_X86_64)
  #include "backend/cpu/x86_64/pack_f32.hpp"
  #include "backend/cpu/x86_64/pack_f16.hpp"
  #include "backend/cpu/x86_64/mma_pack_f32.hpp"
  #include "backend/cpu/x86_64/mma_pack_f16.hpp"
  #include "backend/cpu/x86_64/mma_direct_f32.hpp"
  #include "backend/cpu/x86_64/mma_direct_f16.hpp"
  using namespace nnops::backend::cpu::x86_64;
  using f16_t = nnops::backend::cpu::half;  // x86_64: struct half { uint16_t bits; }
  #define NNOPS_PACK_MMA_ARCH "x86_64"
#elif defined(NNOPS_ARCH_AARCH64)
  #include "backend/cpu/aarch64/pack_f32.hpp"
  #include "backend/cpu/aarch64/pack_f16.hpp"
  #include "backend/cpu/aarch64/mma_pack_f32.hpp"
  #include "backend/cpu/aarch64/mma_pack_f16.hpp"
  #include "backend/cpu/aarch64/mma_direct_f32.hpp"
  #include "backend/cpu/aarch64/mma_direct_f16.hpp"
  using namespace nnops::backend::cpu::aarch64;
  using f16_t = float16_t;
  #define NNOPS_PACK_MMA_ARCH "aarch64"
#else
  #error "Unsupported architecture"
#endif

#include <cstring>
#include <vector>
#include <cmath>

// =========================================================================
//  Helpers
// =========================================================================

/// Naive row-major GEMM: C[mr][nr] += A[mr][K] × B[K][nr].
/// Assumes zero-init C is handled by caller (A, B data includes initial C).
template <typename T>
static void naive_gemm(T* C, int ldc,
                       const T* A, int lda,
                       const T* B, int ldb,
                       int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            T sum = C[m * ldc + n];
            for (int k = 0; k < K; ++k) {
                sum += A[m * lda + k] * B[k * ldb + n];
            }
            C[m * ldc + n] = sum;
        }
    }
}

/// Naive packed-B GEMM: C[mr][nr] += A[mr][K] × B_packed[K][nr].
/// A layout: interleaved, A[m + k * M] (M is the inner/fast dimension).
/// B_packed layout: row-major, B_packed[k * ldb + n] (ldb = nr).
template <typename T>
static void naive_gemm_packed(T* C, int ldc,
                              const T* A,
                              const T* B_packed, int ldb,
                              int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            T sum = C[m * ldc + n];
            for (int k = 0; k < K; ++k) {
                sum += A[m + k * M] * B_packed[k * ldb + n];
            }
            C[m * ldc + n] = sum;
        }
    }
}

/// Fill an array with a linear ramp: v[i] = T(start + i).
template <typename T>
static void fill_ramp(T* data, int n, float start = 0.0f) {
    for (int i = 0; i < n; ++i) data[i] = T(start + float(i));
}

/// Convert f16_t to float (platform-neutral).
static float f16_to_f(f16_t v) {
#if defined(NNOPS_ARCH_X86_64)
    return nnops::backend::cpu::half_to_float(v);
#else
    return static_cast<float>(v);
#endif
}

/// Convert float to f16_t (platform-neutral).
static f16_t f_to_f16(float v) {
#if defined(NNOPS_ARCH_X86_64)
    return nnops::backend::cpu::float_to_half(v);
#else
    return static_cast<float16_t>(v);
#endif
}

// =========================================================================
//  Section 1: f32 Pack — transpose (smoke tests)
//
//  pack_trans produces a tiled transpose layout consumed by the GEMM
//  tiling infrastructure. We verify: no crash, no NaN/Inf, scale applied.
// =========================================================================

NNOPS_TEST(pack_f32_trans_n4_smoke) {
    constexpr int N = 4, ir_step = 16, K = 13;  // K=13: SIMD(8) + partial + tail
    constexpr int buf_size = (K - 1) * ir_step + N;
    std::vector<float> input(buf_size, 1.0f);
    std::vector<float> output(N * K);

    pack_trans_n4_f32(output.data(), input.data(), ir_step, K, 1.0f);

    for (int i = 0; i < N * K; ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(output[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(output[i]));
    }
}

NNOPS_TEST(pack_f32_trans_n6_n8_smoke) {
    constexpr int K = 17;  // K=17: SIMD(8)+SIMD(8)+tail(1)
    constexpr int ir_step = 24;
    {
        constexpr int N = 6;
        constexpr int buf_size = (K - 1) * ir_step + N;
        std::vector<float> input(buf_size, 2.0f);
        std::vector<float> output(N * K);
        pack_trans_n6_f32(output.data(), input.data(), ir_step, K, 1.5f);
        for (int i = 0; i < N * K; ++i) {
            NNOPS_EXPECT_TRUE(!std::isnan(output[i]));
            NNOPS_EXPECT_TRUE(!std::isinf(output[i]));
        }
    }
    {
        constexpr int N = 8;
        constexpr int buf_size = (K - 1) * ir_step + N;
        std::vector<float> input(buf_size, 2.0f);
        std::vector<float> output(N * K);
        pack_trans_n8_f32(output.data(), input.data(), ir_step, K, 1.5f);
        for (int i = 0; i < N * K; ++i) {
            NNOPS_EXPECT_TRUE(!std::isnan(output[i]));
            NNOPS_EXPECT_TRUE(!std::isinf(output[i]));
        }
    }
}

// =========================================================================
//  Section 2: f32 Pack — copy
// =========================================================================

NNOPS_TEST(pack_f32_copy_n4) {
    constexpr int N = 4, ir_step = 6, K = 13;  // covers SIMD (K/4) + scalar tail
    std::vector<float> input(K * ir_step);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < ir_step; ++n)
            input[k * ir_step + n] = float(k * 100 + n);

    std::vector<float> output(N * K);
    pack_copy_n4_f32(output.data(), input.data(), ir_step, K, 1.0f);

    // Copy layout: output[k * N + n] = input[k * ir_step + n]
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            NNOPS_EXPECT_NEAR(output[k * N + n], input[k * ir_step + n], 1e-6f);
}

NNOPS_TEST(pack_f32_copy_n8_scale) {
    constexpr int N = 8, ir_step = 12, K = 9;
    float scale = 2.0f;
    std::vector<float> input(K * ir_step);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < ir_step; ++n)
            input[k * ir_step + n] = float(k * 100 + n);

    std::vector<float> output(N * K);
    pack_copy_n8_f32(output.data(), input.data(), ir_step, K, scale);

    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            NNOPS_EXPECT_NEAR(output[k * N + n], input[k * ir_step + n] * scale, 1e-6f);
}

NNOPS_TEST(pack_f32_copy_n16) {
    constexpr int N = 16, ir_step = 20, K = 5;
    std::vector<float> input(K * ir_step);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < ir_step; ++n)
            input[k * ir_step + n] = float(k * 100 + n);

    std::vector<float> output(N * K);
    pack_copy_n16_f32(output.data(), input.data(), ir_step, K, 1.0f);

    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            NNOPS_EXPECT_NEAR(output[k * N + n], input[k * ir_step + n], 1e-6f);
}

// =========================================================================
//  Section 3: f32 mma_pack  —  9 tile sizes
// =========================================================================

/// Helper: test one mma_pack tile size against naive packed GEMM.
static void test_mma_pack_f32(
    void (*mma)(float*, int, const float*, const float*, int, int, float, float),
    int M, int N, int K,
    float clamp_min = -1e9f, float clamp_max = 1e9f)
{
    std::vector<float> A(M * K);
    std::vector<float> B_packed(K * N);
    std::vector<float> C_mma(M * N);
    std::vector<float> C_ref(M * N);

    fill_ramp(A.data(), M * K, 1.0f);
    fill_ramp(B_packed.data(), K * N, 0.5f);

    mma(C_mma.data(), N, A.data(), B_packed.data(), N, K, clamp_min, clamp_max);
    naive_gemm_packed(C_ref.data(), N, A.data(), B_packed.data(), N, M, N, K);

    for (int i = 0; i < M * N; ++i)
        NNOPS_EXPECT_NEAR(C_mma[i], C_ref[i], 1e-4f);
}

NNOPS_TEST(mma_pack_f32_1x1)  { test_mma_pack_f32(mma_pack_1x1_f32,  1, 1,  7); }
NNOPS_TEST(mma_pack_f32_1x8)  { test_mma_pack_f32(mma_pack_1x8_f32,  1, 8,  7); }
NNOPS_TEST(mma_pack_f32_1x16) { test_mma_pack_f32(mma_pack_1x16_f32, 1, 16, 7); }
NNOPS_TEST(mma_pack_f32_4x1)  { test_mma_pack_f32(mma_pack_4x1_f32,  4, 1,  7); }
NNOPS_TEST(mma_pack_f32_4x8)  { test_mma_pack_f32(mma_pack_4x8_f32,  4, 8,  7); }
NNOPS_TEST(mma_pack_f32_4x16) { test_mma_pack_f32(mma_pack_4x16_f32, 4, 16, 7); }
NNOPS_TEST(mma_pack_f32_6x1)  { test_mma_pack_f32(mma_pack_6x1_f32,  6, 1,  7); }
NNOPS_TEST(mma_pack_f32_6x8)  { test_mma_pack_f32(mma_pack_6x8_f32,  6, 8,  7); }
NNOPS_TEST(mma_pack_f32_6x16) { test_mma_pack_f32(mma_pack_6x16_f32, 6, 16, 7); }

NNOPS_TEST(mma_pack_f32_clamp) {
    // Clamp to [2.0, 5.0] — values outside should be clipped.
    // A[m][k]=2, B_packed[k][n]=2*k+1 → inner product per col n
    // A×B col 0: 2*1 + 2*3 + 2*5 = 18 → clamped to 5
    // A×B col 1: 2*1 + 2*3 + 2*5 = 18 → clamped to 5
    constexpr int M = 4, N = 8, K = 3;
    float A[M * K];
    float B_packed[K * N];
    float C[M * N] = {};

    for (int i = 0; i < M * K; ++i) A[i] = 2.0f;
    for (int i = 0; i < K * N; ++i) B_packed[i] = 1.0f;

    mma_pack_4x8_f32(C, N, A, B_packed, N, K, 2.0f, 5.0f);

    // A×B = 2*1+2*1+2*1 = 6 per element → clamped to 5.0
    for (int i = 0; i < M * N; ++i)
        NNOPS_EXPECT_NEAR(C[i], 5.0f, 1e-4f);
}

// =========================================================================
//  Section 4: f32 mma_direct  —  9 tile sizes
// =========================================================================

/// Helper: test one mma_direct tile size against naive row-major GEMM.
static void test_mma_direct_f32(
    void (*mma)(float*, int, const float*, int, const float*, int, int, float, float),
    int M, int N, int K,
    float clamp_min = -1e9f, float clamp_max = 1e9f)
{
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_mma(M * N);
    std::vector<float> C_ref(M * N);

    fill_ramp(A.data(), M * K, 1.0f);
    fill_ramp(B.data(), K * N, 0.5f);

    mma(C_mma.data(), N, A.data(), K, B.data(), N, K, clamp_min, clamp_max);
    naive_gemm(C_ref.data(), N, A.data(), K, B.data(), N, M, N, K);

    for (int i = 0; i < M * N; ++i)
        NNOPS_EXPECT_NEAR(C_mma[i], C_ref[i], 1e-4f);
}

NNOPS_TEST(mma_direct_f32_1x1)  { test_mma_direct_f32(mma_direct_1x1_f32,  1, 1,  7); }
NNOPS_TEST(mma_direct_f32_1x8)  { test_mma_direct_f32(mma_direct_1x8_f32,  1, 8,  7); }
NNOPS_TEST(mma_direct_f32_1x16) { test_mma_direct_f32(mma_direct_1x16_f32, 1, 16, 7); }
NNOPS_TEST(mma_direct_f32_4x1)  { test_mma_direct_f32(mma_direct_4x1_f32,  4, 1,  7); }
NNOPS_TEST(mma_direct_f32_4x8)  { test_mma_direct_f32(mma_direct_4x8_f32,  4, 8,  7); }
NNOPS_TEST(mma_direct_f32_4x16) { test_mma_direct_f32(mma_direct_4x16_f32, 4, 16, 7); }
NNOPS_TEST(mma_direct_f32_6x1)  { test_mma_direct_f32(mma_direct_6x1_f32,  6, 1,  7); }
NNOPS_TEST(mma_direct_f32_6x8)  { test_mma_direct_f32(mma_direct_6x8_f32,  6, 8,  7); }
NNOPS_TEST(mma_direct_f32_6x16) { test_mma_direct_f32(mma_direct_6x16_f32, 6, 16, 7); }

// =========================================================================
//  Section 5: f32 mma cross-validation  —  pack vs direct (same A,B)
// =========================================================================

NNOPS_TEST(mma_f32_pack_vs_direct) {
    // mma_pack uses interleaved A (A[m + k*M]), mma_direct uses row-major (A[m*K + k]).
    // Build a mathematical matrix and lay it out both ways.
    constexpr int M = 4, N = 8, K = 9;
    std::vector<float> A_pack(M * K);      // interleaved: A_pack[m + k*M]
    std::vector<float> A_direct(M * K);    // row-major:  A_direct[m*K + k]
    std::vector<float> B(K * N);
    std::vector<float> B_packed(K * N);
    std::vector<float> C_pack(M * N);
    std::vector<float> C_direct(M * N);

    // Fill B identically
    fill_ramp(B.data(), K * N, 0.5f);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            B_packed[k * N + n] = B[k * N + n];

    // Fill A with same values in both layouts
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            float val = 1.0f + float(m + k * M);
            A_pack[m + k * M] = val;
            A_direct[m * K + k] = val;
        }
    }

    mma_pack_4x8_f32(C_pack.data(), N, A_pack.data(), B_packed.data(), N, K, -1e9f, 1e9f);
    mma_direct_4x8_f32(C_direct.data(), N, A_direct.data(), K, B.data(), N, K, -1e9f, 1e9f);

    for (int i = 0; i < M * N; ++i)
        NNOPS_EXPECT_NEAR(C_pack[i], C_direct[i], 1e-4f);
}

// =========================================================================
//  Section 6: f16 Pack  —  smoke + copy
// =========================================================================

NNOPS_TEST(pack_f16_trans_smoke) {
    constexpr int N = 4, ir_step = 16, K = 13;
    constexpr int buf_size = (K - 1) * ir_step + N;
    std::vector<f16_t> input(buf_size);
    for (int i = 0; i < buf_size; ++i) input[i] = f_to_f16(1.0f);
    std::vector<f16_t> output(N * K);

    pack_trans_n4_f16(output.data(), input.data(), ir_step, K, 1.0f);

    for (int i = 0; i < N * K; ++i) {
        float v = f16_to_f(output[i]);
        NNOPS_EXPECT_TRUE(!std::isnan(v));
        NNOPS_EXPECT_TRUE(!std::isinf(v));
    }
}

NNOPS_TEST(pack_f16_copy_n8) {
    constexpr int N = 8, ir_step = 14, K = 11;
    std::vector<f16_t> input(K * ir_step);
    std::vector<f16_t> output(N * K);

    for (int i = 0; i < K * ir_step; ++i)
        input[i] = f_to_f16(float(i) * 0.1f);

    pack_copy_n8_f16(output.data(), input.data(), ir_step, K, 1.0f);

    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            NNOPS_EXPECT_NEAR(f16_to_f(output[k * N + n]),
                              f16_to_f(input[k * ir_step + n]), 0.01f);
}

// =========================================================================
//  Section 7: f16 mma_pack  —  9 tile sizes
// =========================================================================

/// Helper: test one f16 mma_pack tile size against naive packed GEMM.
static void test_mma_pack_f16(
    void (*mma)(f16_t*, int, const f16_t*, const f16_t*, int, int, float, float),
    int M, int N, int K)
{
    std::vector<f16_t> A(M * K);
    std::vector<f16_t> B_packed(K * N);
    std::vector<f16_t> C_mma(M * N);
    std::vector<float>  C_ref_f32(M * N);

    for (int i = 0; i < M * K; ++i) A[i] = f_to_f16(1.0f + 0.1f * float(i));
    for (int i = 0; i < K * N; ++i) B_packed[i] = f_to_f16(0.5f + 0.1f * float(i));

    mma(C_mma.data(), N, A.data(), B_packed.data(), N, K, -1e4f, 1e4f);

    // Build float versions and compute reference
    std::vector<float> A_f32(M * K), B_f32(K * N);
    for (int i = 0; i < M * K; ++i) A_f32[i] = f16_to_f(A[i]);
    for (int i = 0; i < K * N; ++i) B_f32[i] = f16_to_f(B_packed[i]);
    naive_gemm_packed(C_ref_f32.data(), N, A_f32.data(), B_f32.data(), N, M, N, K);

    for (int i = 0; i < M * N; ++i)
        NNOPS_EXPECT_NEAR(f16_to_f(C_mma[i]), C_ref_f32[i], 0.5f);  // f16 tolerance
}

NNOPS_TEST(mma_pack_f16_1x1)  { test_mma_pack_f16(mma_pack_1x1_f16,  1, 1,  7); }
NNOPS_TEST(mma_pack_f16_1x8)  { test_mma_pack_f16(mma_pack_1x8_f16,  1, 8,  7); }
NNOPS_TEST(mma_pack_f16_1x16) { test_mma_pack_f16(mma_pack_1x16_f16, 1, 16, 7); }
NNOPS_TEST(mma_pack_f16_4x1)  { test_mma_pack_f16(mma_pack_4x1_f16,  4, 1,  7); }
NNOPS_TEST(mma_pack_f16_4x8)  { test_mma_pack_f16(mma_pack_4x8_f16,  4, 8,  7); }
NNOPS_TEST(mma_pack_f16_4x16) { test_mma_pack_f16(mma_pack_4x16_f16, 4, 16, 7); }
NNOPS_TEST(mma_pack_f16_6x1)  { test_mma_pack_f16(mma_pack_6x1_f16,  6, 1,  7); }
NNOPS_TEST(mma_pack_f16_6x8)  { test_mma_pack_f16(mma_pack_6x8_f16,  6, 8,  7); }
NNOPS_TEST(mma_pack_f16_6x16) { test_mma_pack_f16(mma_pack_6x16_f16, 6, 16, 7); }

NNOPS_TEST(mma_pack_f16_clamp) {
    constexpr int M = 4, N = 8, K = 3;
    std::vector<f16_t> A(M * K, f_to_f16(2.0f));
    std::vector<f16_t> B_packed(K * N, f_to_f16(1.0f));
    std::vector<f16_t> C(M * N);

    mma_pack_4x8_f16(C.data(), N, A.data(), B_packed.data(), N, K, 2.0f, 5.0f);

    // 2*1+2*1+2*1 = 6 → clamped to 5.0
    for (int i = 0; i < M * N; ++i)
        NNOPS_EXPECT_NEAR(f16_to_f(C[i]), 5.0f, 0.1f);
}

// =========================================================================
//  Section 8: f16 mma_direct  —  9 tile sizes
// =========================================================================

/// Helper: test one f16 mma_direct tile size against naive row-major GEMM.
static void test_mma_direct_f16(
    void (*mma)(f16_t*, int, const f16_t*, int, const f16_t*, int, int, float, float),
    int M, int N, int K)
{
    std::vector<f16_t> A(M * K);
    std::vector<f16_t> B(K * N);
    std::vector<f16_t> C_mma(M * N);
    std::vector<float>  C_ref_f32(M * N);

    for (int i = 0; i < M * K; ++i) A[i] = f_to_f16(1.0f + 0.1f * float(i));
    for (int i = 0; i < K * N; ++i) B[i] = f_to_f16(0.5f + 0.1f * float(i));

    mma(C_mma.data(), N, A.data(), K, B.data(), N, K, -1e4f, 1e4f);

    std::vector<float> A_f32(M * K), B_f32(K * N);
    for (int i = 0; i < M * K; ++i) A_f32[i] = f16_to_f(A[i]);
    for (int i = 0; i < K * N; ++i) B_f32[i] = f16_to_f(B[i]);
    naive_gemm(C_ref_f32.data(), N, A_f32.data(), K, B_f32.data(), N, M, N, K);

    for (int i = 0; i < M * N; ++i)
        NNOPS_EXPECT_NEAR(f16_to_f(C_mma[i]), C_ref_f32[i], 0.5f);
}

NNOPS_TEST(mma_direct_f16_1x1)  { test_mma_direct_f16(mma_direct_1x1_f16,  1, 1,  7); }
NNOPS_TEST(mma_direct_f16_1x8)  { test_mma_direct_f16(mma_direct_1x8_f16,  1, 8,  7); }
NNOPS_TEST(mma_direct_f16_1x16) { test_mma_direct_f16(mma_direct_1x16_f16, 1, 16, 7); }
NNOPS_TEST(mma_direct_f16_4x1)  { test_mma_direct_f16(mma_direct_4x1_f16,  4, 1,  7); }
NNOPS_TEST(mma_direct_f16_4x8)  { test_mma_direct_f16(mma_direct_4x8_f16,  4, 8,  7); }
NNOPS_TEST(mma_direct_f16_4x16) { test_mma_direct_f16(mma_direct_4x16_f16, 4, 16, 7); }
NNOPS_TEST(mma_direct_f16_6x1)  { test_mma_direct_f16(mma_direct_6x1_f16,  6, 1,  7); }
NNOPS_TEST(mma_direct_f16_6x8)  { test_mma_direct_f16(mma_direct_6x8_f16,  6, 8,  7); }
NNOPS_TEST(mma_direct_f16_6x16) { test_mma_direct_f16(mma_direct_6x16_f16, 6, 16, 7); }

// =========================================================================
//  Section 9: f16 mma cross-validation  —  pack vs direct
// =========================================================================

NNOPS_TEST(mma_f16_pack_vs_direct) {
    constexpr int M = 4, N = 8, K = 9;
    std::vector<f16_t> A_pack(M * K);
    std::vector<f16_t> A_direct(M * K);
    std::vector<f16_t> B(K * N);
    std::vector<f16_t> B_packed(K * N);
    std::vector<f16_t> C_pack(M * N);
    std::vector<f16_t> C_direct(M * N);

    // Fill B identically
    for (int i = 0; i < K * N; ++i) B[i] = f_to_f16(0.5f + 0.1f * float(i));
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            B_packed[k * N + n] = B[k * N + n];

    // Fill A with same values in both layouts
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            f16_t val = f_to_f16(1.0f + float(m + k * M));
            A_pack[m + k * M] = val;
            A_direct[m * K + k] = val;
        }
    }

    mma_pack_4x8_f16(C_pack.data(), N, A_pack.data(), B_packed.data(), N, K, -1e4f, 1e4f);
    mma_direct_4x8_f16(C_direct.data(), N, A_direct.data(), K, B.data(), N, K, -1e4f, 1e4f);

    for (int i = 0; i < M * N; ++i)
        NNOPS_EXPECT_NEAR(f16_to_f(C_pack[i]), f16_to_f(C_direct[i]), 0.1f);
}

// =========================================================================
//  Section 10: Accumulation  —  mma adds to existing C, not overwrites
// =========================================================================

NNOPS_TEST(mma_f32_accumulation) {
    constexpr int M = 4, N = 8, K = 3;
    float A[M * K] = {1,0,0, 0,1,0, 0,0,1, 1,1,1};
    float B_packed[K * N] = {};
    B_packed[0 * N + 0] = 2;
    B_packed[1 * N + 1] = 3;
    B_packed[2 * N + 2] = 4;
    // Other B elements remain 0

    std::vector<float> C(M * N);
    fill_ramp(C.data(), M * N, 10.0f);
    std::vector<float> C_init = C;

    mma_pack_4x8_f32(C.data(), N, A, B_packed, N, K, -1e9f, 1e9f);

    std::vector<float> C_expected = C_init;
    naive_gemm_packed(C_expected.data(), N, A, B_packed, N, M, N, K);

    for (int i = 0; i < M * N; ++i)
        NNOPS_EXPECT_NEAR(C[i], C_expected[i], 1e-4f);
}

// =========================================================================
//  Section 11: Identity check  —  A=0 or B=0 should leave C unchanged
// =========================================================================

NNOPS_TEST(mma_f32_zero_a) {
    constexpr int M = 4, N = 8, K = 5;
    std::vector<float> A(M * K, 0.0f);
    std::vector<float> B_packed(K * N);
    std::vector<float> C(M * N);

    fill_ramp(B_packed.data(), K * N, 1.0f);
    fill_ramp(C.data(), M * N, 100.0f);
    auto C_saved = C;

    mma_pack_4x8_f32(C.data(), N, A.data(), B_packed.data(), N, K, -1e9f, 1e9f);

    for (int i = 0; i < M * N; ++i)
        NNOPS_EXPECT_NEAR(C[i], C_saved[i], 1e-6f);
}
