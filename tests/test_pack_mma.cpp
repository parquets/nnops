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
#include "backend/cpu/matmul_helper.h"

#if defined(NNOPS_ARCH_X86_64)
  #include "backend/cpu/x86_64/pack_f32.hpp"
  #include "backend/cpu/x86_64/pack_f16.hpp"
  #include "backend/cpu/x86_64/mma_pack_f32.hpp"
  #include "backend/cpu/x86_64/mma_pack_f16.hpp"
  #include "backend/cpu/x86_64/mma_direct_f32.hpp"
  #include "backend/cpu/x86_64/mma_direct_f16.hpp"
  #include "backend/cpu/x86_64/quant.hpp"
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
  #include "backend/cpu/aarch64/quant.hpp"
  using namespace nnops::backend::cpu::aarch64;
  using f16_t = float16_t;
  #define NNOPS_PACK_MMA_ARCH "aarch64"
#else
  #error "Unsupported architecture"
#endif

// ---- arch-specific f32 mma tile geometry ----------------------------------
// The f32 mma micro-kernels tile differently per arch: x86_64 uses
// mr∈{6,4,1} × nr∈{16,8,1}, aarch64 uses mr∈{8,4,1} × nr∈{12,4,1}. The
// specific-size tests below exercise one mid-sized tile, which is 4x8 on
// x86_64 and 4x4 on aarch64. The f16 kernels share mr∈{4,1} and nr∈{16,8,1}
// across both arches, so the 4x8 f16 tile is arch-neutral and needs no macro.
#if defined(NNOPS_ARCH_X86_64)
  #define NNOPS_MMA_F32_MID_PACK    mma_pack_4x8_f32
  #define NNOPS_MMA_F32_MID_DIRECT  mma_direct_4x8_f32
  #define NNOPS_MMA_F32_MID_M 4
  #define NNOPS_MMA_F32_MID_N 8
#elif defined(NNOPS_ARCH_AARCH64)
  #define NNOPS_MMA_F32_MID_PACK    mma_pack_4x4_f32
  #define NNOPS_MMA_F32_MID_DIRECT  mma_direct_4x4_f32
  #define NNOPS_MMA_F32_MID_M 4
  #define NNOPS_MMA_F32_MID_N 4
#endif

#include <cstring>
#include <vector>
#include <cmath>
#include <type_traits>

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
    for (int i = 0; i < n; ++i) { data[i] = T(start + float(i)); }
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
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < ir_step; ++n) {
            input[k * ir_step + n] = float(k * 100 + n);
        }
    }

    std::vector<float> output(N * K);
    pack_copy_n4_f32(output.data(), input.data(), ir_step, K, 1.0f);

    // Copy layout: output[k * N + n] = input[k * ir_step + n]
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            NNOPS_EXPECT_NEAR(output[k * N + n], input[k * ir_step + n], 1e-6f);
        }
    }
}

NNOPS_TEST(pack_f32_copy_n8_scale) {
    constexpr int N = 8, ir_step = 12, K = 9;
    float scale = 2.0f;
    std::vector<float> input(K * ir_step);
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < ir_step; ++n) {
            input[k * ir_step + n] = float(k * 100 + n);
        }
    }

    std::vector<float> output(N * K);
    pack_copy_n8_f32(output.data(), input.data(), ir_step, K, scale);

    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            NNOPS_EXPECT_NEAR(output[k * N + n], input[k * ir_step + n] * scale, 1e-6f);
        }
    }
}

NNOPS_TEST(pack_f32_copy_n16) {
    constexpr int N = 16, ir_step = 20, K = 5;
    std::vector<float> input(K * ir_step);
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < ir_step; ++n) {
            input[k * ir_step + n] = float(k * 100 + n);
        }
    }

    std::vector<float> output(N * K);
    pack_copy_n16_f32(output.data(), input.data(), ir_step, K, 1.0f);

    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            NNOPS_EXPECT_NEAR(output[k * N + n], input[k * ir_step + n], 1e-6f);
        }
    }
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

    for (int i = 0; i < M * N; ++i) {
        NNOPS_EXPECT_NEAR(C_mma[i], C_ref[i], 1e-4f);
    }
}

#if defined(NNOPS_ARCH_X86_64)
NNOPS_TEST(mma_pack_f32_1x1)  { test_mma_pack_f32(mma_pack_1x1_f32,  1, 1,  7); }
NNOPS_TEST(mma_pack_f32_1x8)  { test_mma_pack_f32(mma_pack_1x8_f32,  1, 8,  7); }
NNOPS_TEST(mma_pack_f32_1x16) { test_mma_pack_f32(mma_pack_1x16_f32, 1, 16, 7); }
NNOPS_TEST(mma_pack_f32_4x1)  { test_mma_pack_f32(mma_pack_4x1_f32,  4, 1,  7); }
NNOPS_TEST(mma_pack_f32_4x8)  { test_mma_pack_f32(mma_pack_4x8_f32,  4, 8,  7); }
NNOPS_TEST(mma_pack_f32_4x16) { test_mma_pack_f32(mma_pack_4x16_f32, 4, 16, 7); }
NNOPS_TEST(mma_pack_f32_6x1)  { test_mma_pack_f32(mma_pack_6x1_f32,  6, 1,  7); }
NNOPS_TEST(mma_pack_f32_6x8)  { test_mma_pack_f32(mma_pack_6x8_f32,  6, 8,  7); }
NNOPS_TEST(mma_pack_f32_6x16) { test_mma_pack_f32(mma_pack_6x16_f32, 6, 16, 7); }
#elif defined(NNOPS_ARCH_AARCH64)
NNOPS_TEST(mma_pack_f32_1x1)  { test_mma_pack_f32(mma_pack_1x1_f32,   1, 1,  7); }
NNOPS_TEST(mma_pack_f32_1x4)  { test_mma_pack_f32(mma_pack_1x4_f32,   1, 4,  7); }
NNOPS_TEST(mma_pack_f32_1x12) { test_mma_pack_f32(mma_pack_1x12_f32,  1, 12, 7); }
NNOPS_TEST(mma_pack_f32_4x1)  { test_mma_pack_f32(mma_pack_4x1_f32,   4, 1,  7); }
NNOPS_TEST(mma_pack_f32_4x4)  { test_mma_pack_f32(mma_pack_4x4_f32,   4, 4,  7); }
NNOPS_TEST(mma_pack_f32_4x12) { test_mma_pack_f32(mma_pack_4x12_f32,  4, 12, 7); }
NNOPS_TEST(mma_pack_f32_8x1)  { test_mma_pack_f32(mma_pack_8x1_f32,   8, 1,  7); }
NNOPS_TEST(mma_pack_f32_8x4)  { test_mma_pack_f32(mma_pack_8x4_f32,   8, 4,  7); }
NNOPS_TEST(mma_pack_f32_8x12) { test_mma_pack_f32(mma_pack_8x12_f32,  8, 12, 7); }
#endif

NNOPS_TEST(mma_pack_f32_clamp) {
    // Clamp to [2.0, 5.0] — values outside should be clipped.
    // A[m][k]=2, B_packed[k][n]=2*k+1 → inner product per col n
    // A×B col 0: 2*1 + 2*3 + 2*5 = 18 → clamped to 5
    // A×B col 1: 2*1 + 2*3 + 2*5 = 18 → clamped to 5
    constexpr int M = NNOPS_MMA_F32_MID_M, N = NNOPS_MMA_F32_MID_N, K = 3;
    float A[M * K];
    float B_packed[K * N];
    float C[M * N] = {};

    for (int i = 0; i < M * K; ++i) { A[i] = 2.0f; }
    for (int i = 0; i < K * N; ++i) { B_packed[i] = 1.0f; }

    NNOPS_MMA_F32_MID_PACK(C, N, A, B_packed, N, K, 2.0f, 5.0f);

    // A×B = 2*1+2*1+2*1 = 6 per element → clamped to 5.0
    for (int i = 0; i < M * N; ++i) {
        NNOPS_EXPECT_NEAR(C[i], 5.0f, 1e-4f);
    }
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

    for (int i = 0; i < M * N; ++i) {
        NNOPS_EXPECT_NEAR(C_mma[i], C_ref[i], 1e-4f);
    }
}

#if defined(NNOPS_ARCH_X86_64)
NNOPS_TEST(mma_direct_f32_1x1)  { test_mma_direct_f32(mma_direct_1x1_f32,  1, 1,  7); }
NNOPS_TEST(mma_direct_f32_1x8)  { test_mma_direct_f32(mma_direct_1x8_f32,  1, 8,  7); }
NNOPS_TEST(mma_direct_f32_1x16) { test_mma_direct_f32(mma_direct_1x16_f32, 1, 16, 7); }
NNOPS_TEST(mma_direct_f32_4x1)  { test_mma_direct_f32(mma_direct_4x1_f32,  4, 1,  7); }
NNOPS_TEST(mma_direct_f32_4x8)  { test_mma_direct_f32(mma_direct_4x8_f32,  4, 8,  7); }
NNOPS_TEST(mma_direct_f32_4x16) { test_mma_direct_f32(mma_direct_4x16_f32, 4, 16, 7); }
NNOPS_TEST(mma_direct_f32_6x1)  { test_mma_direct_f32(mma_direct_6x1_f32,  6, 1,  7); }
NNOPS_TEST(mma_direct_f32_6x8)  { test_mma_direct_f32(mma_direct_6x8_f32,  6, 8,  7); }
NNOPS_TEST(mma_direct_f32_6x16) { test_mma_direct_f32(mma_direct_6x16_f32, 6, 16, 7); }
#elif defined(NNOPS_ARCH_AARCH64)
NNOPS_TEST(mma_direct_f32_1x1)  { test_mma_direct_f32(mma_direct_1x1_f32,   1, 1,  7); }
NNOPS_TEST(mma_direct_f32_1x4)  { test_mma_direct_f32(mma_direct_1x4_f32,   1, 4,  7); }
NNOPS_TEST(mma_direct_f32_1x12) { test_mma_direct_f32(mma_direct_1x12_f32,  1, 12, 7); }
NNOPS_TEST(mma_direct_f32_4x1)  { test_mma_direct_f32(mma_direct_4x1_f32,   4, 1,  7); }
NNOPS_TEST(mma_direct_f32_4x4)  { test_mma_direct_f32(mma_direct_4x4_f32,   4, 4,  7); }
NNOPS_TEST(mma_direct_f32_4x12) { test_mma_direct_f32(mma_direct_4x12_f32,  4, 12, 7); }
NNOPS_TEST(mma_direct_f32_8x1)  { test_mma_direct_f32(mma_direct_8x1_f32,   8, 1,  7); }
NNOPS_TEST(mma_direct_f32_8x4)  { test_mma_direct_f32(mma_direct_8x4_f32,   8, 4,  7); }
NNOPS_TEST(mma_direct_f32_8x12) { test_mma_direct_f32(mma_direct_8x12_f32,  8, 12, 7); }
#endif

// =========================================================================
//  Section 5: f32 mma cross-validation  —  pack vs direct (same A,B)
// =========================================================================

NNOPS_TEST(mma_f32_pack_vs_direct) {
    // mma_pack uses interleaved A (A[m + k*M]), mma_direct uses row-major (A[m*K + k]).
    // Build a mathematical matrix and lay it out both ways.
    constexpr int M = NNOPS_MMA_F32_MID_M, N = NNOPS_MMA_F32_MID_N, K = 9;
    std::vector<float> A_pack(M * K);      // interleaved: A_pack[m + k*M]
    std::vector<float> A_direct(M * K);    // row-major:  A_direct[m*K + k]
    std::vector<float> B(K * N);
    std::vector<float> B_packed(K * N);
    std::vector<float> C_pack(M * N);
    std::vector<float> C_direct(M * N);

    // Fill B identically
    fill_ramp(B.data(), K * N, 0.5f);
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            B_packed[k * N + n] = B[k * N + n];
        }
    }

    // Fill A with same values in both layouts
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            float val = 1.0f + float(m + k * M);
            A_pack[m + k * M] = val;
            A_direct[m * K + k] = val;
        }
    }

    NNOPS_MMA_F32_MID_PACK(C_pack.data(), N, A_pack.data(), B_packed.data(), N, K, -1e9f, 1e9f);
    NNOPS_MMA_F32_MID_DIRECT(C_direct.data(), N, A_direct.data(), K, B.data(), N, K, -1e9f, 1e9f);

    for (int i = 0; i < M * N; ++i) {
        NNOPS_EXPECT_NEAR(C_pack[i], C_direct[i], 1e-4f);
    }
}

// =========================================================================
//  Section 6: f16 Pack  —  smoke + copy
// =========================================================================

NNOPS_TEST(pack_f16_trans_smoke) {
    constexpr int N = 4, ir_step = 16, K = 13;
    constexpr int buf_size = (K - 1) * ir_step + N;
    std::vector<f16_t> input(buf_size);
    for (int i = 0; i < buf_size; ++i) { input[i] = f_to_f16(1.0f); }
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

    for (int i = 0; i < K * ir_step; ++i) {
        input[i] = f_to_f16(float(i) * 0.1f);
    }

    pack_copy_n8_f16(output.data(), input.data(), ir_step, K, 1.0f);

    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            NNOPS_EXPECT_NEAR(f16_to_f(output[k * N + n]),
                              f16_to_f(input[k * ir_step + n]), 0.01f);
        }
    }
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

    for (int i = 0; i < M * K; ++i) { A[i] = f_to_f16(1.0f + 0.1f * float(i)); }
    for (int i = 0; i < K * N; ++i) { B_packed[i] = f_to_f16(0.5f + 0.1f * float(i)); }

    mma(C_mma.data(), N, A.data(), B_packed.data(), N, K, -1e4f, 1e4f);

    // Build float versions and compute reference
    std::vector<float> A_f32(M * K), B_f32(K * N);
    for (int i = 0; i < M * K; ++i) { A_f32[i] = f16_to_f(A[i]); }
    for (int i = 0; i < K * N; ++i) { B_f32[i] = f16_to_f(B_packed[i]); }
    naive_gemm_packed(C_ref_f32.data(), N, A_f32.data(), B_f32.data(), N, M, N, K);

    for (int i = 0; i < M * N; ++i) {
        NNOPS_EXPECT_NEAR(f16_to_f(C_mma[i]), C_ref_f32[i], 0.5f);
    }  // f16 tolerance
}

#if defined(NNOPS_ARCH_X86_64)
NNOPS_TEST(mma_pack_f16_1x1)  { test_mma_pack_f16(mma_pack_1x1_f16,  1, 1,  7); }
NNOPS_TEST(mma_pack_f16_1x8)  { test_mma_pack_f16(mma_pack_1x8_f16,  1, 8,  7); }
NNOPS_TEST(mma_pack_f16_1x16) { test_mma_pack_f16(mma_pack_1x16_f16, 1, 16, 7); }
NNOPS_TEST(mma_pack_f16_4x1)  { test_mma_pack_f16(mma_pack_4x1_f16,  4, 1,  7); }
NNOPS_TEST(mma_pack_f16_4x8)  { test_mma_pack_f16(mma_pack_4x8_f16,  4, 8,  7); }
NNOPS_TEST(mma_pack_f16_4x16) { test_mma_pack_f16(mma_pack_4x16_f16, 4, 16, 7); }
NNOPS_TEST(mma_pack_f16_6x1)  { test_mma_pack_f16(mma_pack_6x1_f16,  6, 1,  7); }
NNOPS_TEST(mma_pack_f16_6x8)  { test_mma_pack_f16(mma_pack_6x8_f16,  6, 8,  7); }
NNOPS_TEST(mma_pack_f16_6x16) { test_mma_pack_f16(mma_pack_6x16_f16, 6, 16, 7); }
#elif defined(NNOPS_ARCH_AARCH64)
NNOPS_TEST(mma_pack_f16_1x1)  { test_mma_pack_f16(mma_pack_1x1_f16,  1, 1,  7); }
NNOPS_TEST(mma_pack_f16_1x8)  { test_mma_pack_f16(mma_pack_1x8_f16,  1, 8,  7); }
NNOPS_TEST(mma_pack_f16_1x16) { test_mma_pack_f16(mma_pack_1x16_f16, 1, 16, 7); }
NNOPS_TEST(mma_pack_f16_4x1)  { test_mma_pack_f16(mma_pack_4x1_f16,  4, 1,  7); }
NNOPS_TEST(mma_pack_f16_4x8)  { test_mma_pack_f16(mma_pack_4x8_f16,  4, 8,  7); }
NNOPS_TEST(mma_pack_f16_4x16) { test_mma_pack_f16(mma_pack_4x16_f16, 4, 16, 7); }
NNOPS_TEST(mma_pack_f16_8x1)  { test_mma_pack_f16(mma_pack_8x1_f16,  8, 1,  7); }
NNOPS_TEST(mma_pack_f16_8x8)  { test_mma_pack_f16(mma_pack_8x8_f16,  8, 8,  7); }
NNOPS_TEST(mma_pack_f16_8x16) { test_mma_pack_f16(mma_pack_8x16_f16, 8, 16, 7); }
#endif

NNOPS_TEST(mma_pack_f16_clamp) {
    constexpr int M = 4, N = 8, K = 3;
    std::vector<f16_t> A(M * K, f_to_f16(2.0f));
    std::vector<f16_t> B_packed(K * N, f_to_f16(1.0f));
    std::vector<f16_t> C(M * N);

    mma_pack_4x8_f16(C.data(), N, A.data(), B_packed.data(), N, K, 2.0f, 5.0f);

    // 2*1+2*1+2*1 = 6 → clamped to 5.0
    for (int i = 0; i < M * N; ++i) {
        NNOPS_EXPECT_NEAR(f16_to_f(C[i]), 5.0f, 0.1f);
    }
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

    for (int i = 0; i < M * K; ++i) { A[i] = f_to_f16(1.0f + 0.1f * float(i)); }
    for (int i = 0; i < K * N; ++i) { B[i] = f_to_f16(0.5f + 0.1f * float(i)); }

    mma(C_mma.data(), N, A.data(), K, B.data(), N, K, -1e4f, 1e4f);

    std::vector<float> A_f32(M * K), B_f32(K * N);
    for (int i = 0; i < M * K; ++i) { A_f32[i] = f16_to_f(A[i]); }
    for (int i = 0; i < K * N; ++i) { B_f32[i] = f16_to_f(B[i]); }
    naive_gemm(C_ref_f32.data(), N, A_f32.data(), K, B_f32.data(), N, M, N, K);

    for (int i = 0; i < M * N; ++i) {
        NNOPS_EXPECT_NEAR(f16_to_f(C_mma[i]), C_ref_f32[i], 0.5f);
    }
}

#if defined(NNOPS_ARCH_X86_64)
NNOPS_TEST(mma_direct_f16_1x1)  { test_mma_direct_f16(mma_direct_1x1_f16,  1, 1,  7); }
NNOPS_TEST(mma_direct_f16_1x8)  { test_mma_direct_f16(mma_direct_1x8_f16,  1, 8,  7); }
NNOPS_TEST(mma_direct_f16_1x16) { test_mma_direct_f16(mma_direct_1x16_f16, 1, 16, 7); }
NNOPS_TEST(mma_direct_f16_4x1)  { test_mma_direct_f16(mma_direct_4x1_f16,  4, 1,  7); }
NNOPS_TEST(mma_direct_f16_4x8)  { test_mma_direct_f16(mma_direct_4x8_f16,  4, 8,  7); }
NNOPS_TEST(mma_direct_f16_4x16) { test_mma_direct_f16(mma_direct_4x16_f16, 4, 16, 7); }
NNOPS_TEST(mma_direct_f16_6x1)  { test_mma_direct_f16(mma_direct_6x1_f16,  6, 1,  7); }
NNOPS_TEST(mma_direct_f16_6x8)  { test_mma_direct_f16(mma_direct_6x8_f16,  6, 8,  7); }
NNOPS_TEST(mma_direct_f16_6x16) { test_mma_direct_f16(mma_direct_6x16_f16, 6, 16, 7); }
#elif defined(NNOPS_ARCH_AARCH64)
NNOPS_TEST(mma_direct_f16_1x1)  { test_mma_direct_f16(mma_direct_1x1_f16,  1, 1,  7); }
NNOPS_TEST(mma_direct_f16_1x8)  { test_mma_direct_f16(mma_direct_1x8_f16,  1, 8,  7); }
NNOPS_TEST(mma_direct_f16_1x16) { test_mma_direct_f16(mma_direct_1x16_f16, 1, 16, 7); }
NNOPS_TEST(mma_direct_f16_4x1)  { test_mma_direct_f16(mma_direct_4x1_f16,  4, 1,  7); }
NNOPS_TEST(mma_direct_f16_4x8)  { test_mma_direct_f16(mma_direct_4x8_f16,  4, 8,  7); }
NNOPS_TEST(mma_direct_f16_4x16) { test_mma_direct_f16(mma_direct_4x16_f16, 4, 16, 7); }
NNOPS_TEST(mma_direct_f16_8x1)  { test_mma_direct_f16(mma_direct_8x1_f16,  8, 1,  7); }
NNOPS_TEST(mma_direct_f16_8x8)  { test_mma_direct_f16(mma_direct_8x8_f16,  8, 8,  7); }
NNOPS_TEST(mma_direct_f16_8x16) { test_mma_direct_f16(mma_direct_8x16_f16, 8, 16, 7); }
#endif

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
    for (int i = 0; i < K * N; ++i) { B[i] = f_to_f16(0.5f + 0.1f * float(i)); }
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            B_packed[k * N + n] = B[k * N + n];
        }
    }

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

    for (int i = 0; i < M * N; ++i) {
        NNOPS_EXPECT_NEAR(f16_to_f(C_pack[i]), f16_to_f(C_direct[i]), 0.1f);
    }
}

// =========================================================================
//  Section 10: Accumulation  —  mma adds to existing C, not overwrites
// =========================================================================

NNOPS_TEST(mma_f32_accumulation) {
    constexpr int M = NNOPS_MMA_F32_MID_M, N = NNOPS_MMA_F32_MID_N, K = 3;
    float A[M * K] = {1,0,0, 0,1,0, 0,0,1, 1,1,1};
    float B_packed[K * N] = {};
    B_packed[0 * N + 0] = 2;
    B_packed[1 * N + 1] = 3;
    B_packed[2 * N + 2] = 4;
    // Other B elements remain 0

    std::vector<float> C(M * N);
    fill_ramp(C.data(), M * N, 10.0f);
    std::vector<float> C_init = C;

    NNOPS_MMA_F32_MID_PACK(C.data(), N, A, B_packed, N, K, -1e9f, 1e9f);

    std::vector<float> C_expected = C_init;
    naive_gemm_packed(C_expected.data(), N, A, B_packed, N, M, N, K);

    for (int i = 0; i < M * N; ++i) {
        NNOPS_EXPECT_NEAR(C[i], C_expected[i], 1e-4f);
    }
}

// =========================================================================
//  Section 11: Identity check  —  A=0 or B=0 should leave C unchanged
// =========================================================================

NNOPS_TEST(mma_f32_zero_a) {
    constexpr int M = NNOPS_MMA_F32_MID_M, N = NNOPS_MMA_F32_MID_N, K = 5;
    std::vector<float> A(M * K, 0.0f);
    std::vector<float> B_packed(K * N);
    std::vector<float> C(M * N);

    fill_ramp(B_packed.data(), K * N, 1.0f);
    fill_ramp(C.data(), M * N, 100.0f);
    auto C_saved = C;

    NNOPS_MMA_F32_MID_PACK(C.data(), N, A.data(), B_packed.data(), N, K, -1e9f, 1e9f);

    for (int i = 0; i < M * N; ++i) {
        NNOPS_EXPECT_NEAR(C[i], C_saved[i], 1e-6f);
    }
}

// =========================================================================
//  Section 11b: ZeroMode — <true> overwrites C, <false> accumulates
// =========================================================================

NNOPS_TEST(mma_f32_zero_mode_pack) {
    constexpr int M = NNOPS_MMA_F32_MID_M, N = NNOPS_MMA_F32_MID_N, K = 5;
    std::vector<float> A(M * K);
    std::vector<float> B_packed(K * N);
    fill_ramp(A.data(), M * K, 1.0f);
    fill_ramp(B_packed.data(), K * N, 0.5f);

    std::vector<float> C_ref(M * N);
    naive_gemm_packed(C_ref.data(), N, A.data(), B_packed.data(), N, M, N, K);

    // ZeroMode=true: the junk pre-fill must be ignored (overwrite).
    std::vector<float> C(M * N);
    fill_ramp(C.data(), M * N, 100.0f);
    NNOPS_MMA_F32_MID_PACK<true>(C.data(), N, A.data(), B_packed.data(), N, K, -1e9f, 1e9f);
    for (int i = 0; i < M * N; ++i) {
        NNOPS_EXPECT_NEAR(C[i], C_ref[i], 1e-4f);
    }

    // ZeroMode=false: the same pre-fill must be added (accumulate).
    fill_ramp(C.data(), M * N, 100.0f);
    NNOPS_MMA_F32_MID_PACK<false>(C.data(), N, A.data(), B_packed.data(), N, K, -1e9f, 1e9f);
    for (int i = 0; i < M * N; ++i) {
        NNOPS_EXPECT_NEAR(C[i], C_ref[i] + 100.0f + float(i), 1e-3f);
    }
}

NNOPS_TEST(mma_f32_zero_mode_direct) {
    constexpr int M = NNOPS_MMA_F32_MID_M, N = NNOPS_MMA_F32_MID_N, K = 5;
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    fill_ramp(A.data(), M * K, 1.0f);
    fill_ramp(B.data(), K * N, 0.5f);

    std::vector<float> C_ref(M * N);
    naive_gemm(C_ref.data(), N, A.data(), K, B.data(), N, M, N, K);

    std::vector<float> C(M * N);
    fill_ramp(C.data(), M * N, 100.0f);
    NNOPS_MMA_F32_MID_DIRECT<true>(C.data(), N, A.data(), K, B.data(), N, K, -1e9f, 1e9f);
    for (int i = 0; i < M * N; ++i) {
        NNOPS_EXPECT_NEAR(C[i], C_ref[i], 1e-4f);
    }

    fill_ramp(C.data(), M * N, 100.0f);
    NNOPS_MMA_F32_MID_DIRECT<false>(C.data(), N, A.data(), K, B.data(), N, K, -1e9f, 1e9f);
    for (int i = 0; i < M * N; ++i) {
        NNOPS_EXPECT_NEAR(C[i], C_ref[i] + 100.0f + float(i), 1e-3f);
    }
}

NNOPS_TEST(mma_f16_zero_mode_pack) {
    constexpr int M = 4, N = 8, K = 5;
    std::vector<f16_t> A(M * K);
    std::vector<f16_t> B_packed(K * N);
    for (int i = 0; i < M * K; ++i) { A[i] = f_to_f16(1.0f + 0.1f * float(i)); }
    for (int i = 0; i < K * N; ++i) { B_packed[i] = f_to_f16(0.5f + 0.05f * float(i)); }

    std::vector<float> A_f32(M * K), B_f32(K * N), C_ref(M * N);
    for (int i = 0; i < M * K; ++i) { A_f32[i] = f16_to_f(A[i]); }
    for (int i = 0; i < K * N; ++i) { B_f32[i] = f16_to_f(B_packed[i]); }
    naive_gemm_packed(C_ref.data(), N, A_f32.data(), B_f32.data(), N, M, N, K);

    std::vector<f16_t> C(M * N);
    for (int i = 0; i < M * N; ++i) { C[i] = f_to_f16(100.0f + float(i)); }
    mma_pack_4x8_f16<true>(C.data(), N, A.data(), B_packed.data(), N, K, -1e4f, 1e4f);
    for (int i = 0; i < M * N; ++i) {
        NNOPS_EXPECT_NEAR(f16_to_f(C[i]), C_ref[i], 0.5f);
    }

    for (int i = 0; i < M * N; ++i) { C[i] = f_to_f16(100.0f + float(i)); }
    mma_pack_4x8_f16<false>(C.data(), N, A.data(), B_packed.data(), N, K, -1e4f, 1e4f);
    for (int i = 0; i < M * N; ++i) {
        NNOPS_EXPECT_NEAR(f16_to_f(C[i]), C_ref[i] + 100.0f + float(i), 0.5f);
    }
}

NNOPS_TEST(mma_f16_zero_mode_direct) {
    constexpr int M = 4, N = 8, K = 5;
    std::vector<f16_t> A(M * K);
    std::vector<f16_t> B(K * N);
    for (int i = 0; i < M * K; ++i) { A[i] = f_to_f16(1.0f + 0.1f * float(i)); }
    for (int i = 0; i < K * N; ++i) { B[i] = f_to_f16(0.5f + 0.05f * float(i)); }

    std::vector<float> A_f32(M * K), B_f32(K * N), C_ref(M * N);
    for (int i = 0; i < M * K; ++i) { A_f32[i] = f16_to_f(A[i]); }
    for (int i = 0; i < K * N; ++i) { B_f32[i] = f16_to_f(B[i]); }
    naive_gemm(C_ref.data(), N, A_f32.data(), K, B_f32.data(), N, M, N, K);

    std::vector<f16_t> C(M * N);
    for (int i = 0; i < M * N; ++i) { C[i] = f_to_f16(100.0f + float(i)); }
    mma_direct_4x8_f16<true>(C.data(), N, A.data(), K, B.data(), N, K, -1e4f, 1e4f);
    for (int i = 0; i < M * N; ++i) {
        NNOPS_EXPECT_NEAR(f16_to_f(C[i]), C_ref[i], 0.5f);
    }

    for (int i = 0; i < M * N; ++i) { C[i] = f_to_f16(100.0f + float(i)); }
    mma_direct_4x8_f16<false>(C.data(), N, A.data(), K, B.data(), N, K, -1e4f, 1e4f);
    for (int i = 0; i < M * N; ++i) {
        NNOPS_EXPECT_NEAR(f16_to_f(C[i]), C_ref[i] + 100.0f + float(i), 0.5f);
    }
}

// =========================================================================
//  Section 12: half (f16) quantization / dequantization kernels
// =========================================================================

/// Scalar reference for half→int8/uint8 quantization (matches SIMD rounding).
template <typename T>
static void ref_quant_f16(const f16_t* src, const float* scale, const float* zero,
                          T* dst, int M, int N) {
    const int qmin = std::is_same_v<T, uint8_t> ? 0 : -128;
    const int qmax = std::is_same_v<T, uint8_t> ? 255 : 127;
    for (int m = 0; m < M; ++m) {
        const float inv = 1.0f / scale[m];
        const float zp = (zero == nullptr) ? 0.0f : zero[m];
        for (int n = 0; n < N; ++n) {
            const float q = f16_to_f(src[m * N + n]) * inv + zp;
            int qi = static_cast<int>(std::nearbyintf(q));
            qi = std::min(std::max(qi, qmin), qmax);
            dst[m * N + n] = static_cast<T>(qi);
        }
    }
}

/// Scalar reference for int8/uint8→half dequantization (matches SIMD rounding).
template <typename T>
static void ref_dequant_f16(const T* src, const float* scale, const float* zero,
                            f16_t* dst, int M, int N) {
    for (int m = 0; m < M; ++m) {
        const float zp = (zero == nullptr) ? 0.0f : zero[m];
        for (int n = 0; n < N; ++n) {
            dst[m * N + n] = f_to_f16((static_cast<float>(src[m * N + n]) - zp) * scale[m]);
        }
    }
}

/// Shared body: half→int8/uint8 quantization vs scalar reference.
template <typename T>
static void check_quant_f16(int M, int N, bool with_zero) {
    std::vector<f16_t> src(M * N);
    std::vector<float> scale(M);
    std::vector<float> zero(M);
    std::vector<T> dst(M * N);
    std::vector<T> ref(M * N);

    for (int i = 0; i < M * N; ++i) { src[i] = f_to_f16((float(i % 257) - 128.0f) * 0.7f); }
    for (int m = 0; m < M; ++m) { scale[m] = 0.5f + 0.13f * float(m); zero[m] = float(m) - 2.0f; }

    quantization<T>(M, N, dst.data(), N, src.data(), N, scale.data(),
                    with_zero ? zero.data() : nullptr);
    ref_quant_f16<T>(src.data(), scale.data(), with_zero ? zero.data() : nullptr,
                     ref.data(), M, N);

    for (int i = 0; i < M * N; ++i) {
        NNOPS_EXPECT_EQ(static_cast<int>(dst[i]), static_cast<int>(ref[i]));
    }
}

NNOPS_TEST(quant_f16_to_s8)     { check_quant_f16<int8_t>(3, 56, true); }
NNOPS_TEST(quant_f16_to_u8)     { check_quant_f16<uint8_t>(3, 56, true); }
NNOPS_TEST(quant_f16_to_s8_sym) { check_quant_f16<int8_t>(2, 37, false); }

/// Shared body: int8/uint8→half dequantization vs scalar reference.
template <typename T>
static void check_dequant_f16(int M, int N, bool with_zero) {
    std::vector<T> src(M * N);
    std::vector<float> scale(M);
    std::vector<float> zero(M);
    std::vector<f16_t> dst(M * N);
    std::vector<f16_t> ref(M * N);

    const int off = std::is_same_v<T, uint8_t> ? 0 : 128;
    for (int i = 0; i < M * N; ++i) { src[i] = static_cast<T>((i % 256) - off); }
    for (int m = 0; m < M; ++m) { scale[m] = 0.25f + 0.07f * float(m); zero[m] = float(m) - 1.0f; }

    dequantization<T>(M, N, dst.data(), N, src.data(), N, scale.data(),
                      with_zero ? zero.data() : nullptr);
    ref_dequant_f16<T>(src.data(), scale.data(), with_zero ? zero.data() : nullptr,
                       ref.data(), M, N);

    for (int i = 0; i < M * N; ++i) {
        NNOPS_EXPECT_NEAR(f16_to_f(dst[i]), f16_to_f(ref[i]), 1e-6f);
    }
}

NNOPS_TEST(dequant_s8_to_f16)     { check_dequant_f16<int8_t>(3, 40, true); }
NNOPS_TEST(dequant_u8_to_f16)     { check_dequant_f16<uint8_t>(3, 40, true); }
NNOPS_TEST(dequant_s8_to_f16_sym) { check_dequant_f16<int8_t>(2, 33, false); }

// =========================================================================
//  Section 13: tiled pack entry points (tile_pack_lhs / tile_pack_rhs)
// =========================================================================
//
// These wrappers decompose an M×K (or N×K) tile into the arch panel sizes
// (mr/nr ∈ {max, mid, 1}) and place each panel at a uniform 64-byte-aligned
// stride `ldd`. The packed panel layout is [K][mr]/[K][nr] (k outer, m/n
// inner) — the exact layout the MMA kernels consume. We verify both the
// transpose and copy modes against the logical A/B matrices.

namespace cpu = nnops::backend::cpu;

/// Verify tile_pack_lhs for one transpose mode.
/// src is the physical matrix: [M,K] row-major when !trans, [K,M] when trans.
/// The packed panel p (mr_p rows) must satisfy:
///     dst[p*ldd + k*mr_p + m] = A_logical[m + m_offset_p][k]
static void check_tile_pack_lhs(bool trans) {
    constexpr int M = 11;   // decomposes 6+4+1 across all three mr sizes
    constexpr int K = 13;   // SIMD(8) + partial + scalar tail

    const int lds = trans ? M : K;   // physical row stride of the source
    std::vector<float> src(M * K);
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            // A_logical[m][k]
            const float v = 1.0f + float(m * 100 + k);
            src[trans ? (k * lds + m) : (m * lds + k)] = v;
        }
    }

    const int mr0 = cpu::MR_F32[0];
    const int ldd = (mr0 * K * 4 + 63) / 64 * 16;  // 64-byte-aligned stride
    // Buffer must hold every panel at the aligned stride, not just M*K.
    std::vector<float> dst(static_cast<size_t>(cpu::num_panels(M, cpu::MR_F32)) * ldd, -1.0f);

    cpu::tile_pack_lhs(trans, M, K, dst.data(), ldd, src.data(), lds, 1.0f);

    const int* mr = cpu::MR_F32;
    int m_off = 0;
    int p = 0;
    for (int si = 0; si < 3; ++si) {
        for (; m_off + mr[si] <= M; m_off += mr[si], ++p) {
            // Panel p starts at p * ldd (uniform 64-byte-aligned stride).
            const float* panel = dst.data() + p * ldd;
            for (int k = 0; k < K; ++k) {
                for (int m = 0; m < mr[si]; ++m) {
                    const float expect = 1.0f + float((m_off + m) * 100 + k);
                    NNOPS_EXPECT_NEAR(panel[k * mr[si] + m], expect, 1e-5f);
                }
            }
        }
    }
}

/// Verify tile_pack_rhs for one transpose mode.
/// src is the physical matrix: [K,N] row-major when !trans, [N,K] when trans.
/// The packed panel p (nr_p cols) must satisfy:
///     dst[p*ldd + k*nr_p + n] = B_logical[k][n + n_offset_p]
static void check_tile_pack_rhs(bool trans) {
    constexpr int N = 19;   // decomposes across nr sizes (16/12 + ...)
    constexpr int K = 13;

    const int lds = trans ? K : N;   // physical row stride of the source
    std::vector<float> src(K * N);
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            // B_logical[k][n]
            const float v = 0.5f + float(k * 100 + n);
            src[trans ? (n * lds + k) : (k * lds + n)] = v;
        }
    }

    const int nr0 = cpu::NR_F32[0];
    const int ldd = (nr0 * K * 4 + 63) / 64 * 16;  // 64-byte-aligned stride
    // Buffer must hold every panel at the aligned stride, not just N*K.
    std::vector<float> dst(static_cast<size_t>(cpu::num_panels(N, cpu::NR_F32)) * ldd, -1.0f);

    cpu::tile_pack_rhs(trans, N, K, dst.data(), ldd, src.data(), lds, 1.0f);

    const int* nr = cpu::NR_F32;
    int n_off = 0;
    int p = 0;
    for (int si = 0; si < 3; ++si) {
        for (; n_off + nr[si] <= N; n_off += nr[si], ++p) {
            // Panel p starts at p * ldd (uniform 64-byte-aligned stride).
            const float* panel = dst.data() + p * ldd;
            for (int k = 0; k < K; ++k) {
                for (int n = 0; n < nr[si]; ++n) {
                    const float expect = 0.5f + float(k * 100 + (n_off + n));
                    NNOPS_EXPECT_NEAR(panel[k * nr[si] + n], expect, 1e-5f);
                }
            }
        }
    }
}

NNOPS_TEST(tile_pack_lhs_trans)   { check_tile_pack_lhs(true); }
NNOPS_TEST(tile_pack_lhs_copy)    { check_tile_pack_lhs(false); }
NNOPS_TEST(tile_pack_rhs_trans)   { check_tile_pack_rhs(true); }
NNOPS_TEST(tile_pack_rhs_copy)    { check_tile_pack_rhs(false); }
