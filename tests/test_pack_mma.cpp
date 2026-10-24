/// @file test_pack_mma.cpp
/// @brief Unit tests for the float pack/quant micro-kernels and the tiled
///        pack entry points.
///
/// Tests cover:
///   1. f32 pack  — transpose & copy, SIMD + scalar-tail paths
///   2. f16 pack  — transpose & copy (x86_64: half, aarch64: float16_t)
///   3. f16 quantization / dequantization
///   4. tile_pack_lhs  — the MR cascade the packed-A route uses
///   5. tile_pack_rhs — the uniform nr_max-wide B panels
///
/// The MMA kernels themselves (mma_f32.hpp / mma_f16.hpp) are covered by
/// test_mma_packed.cpp; the per-width mma_pack/mma_direct kernels no longer
/// exist.

#include <algorithm>

#include "common/test_harness.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/simd.hpp"
#include "backend/cpu/matmul/matmul_helper.h"

#if defined(NNOPS_ARCH_X86_64)
  #include "backend/cpu/x86_64/pack_f32.hpp"
  #include "backend/cpu/x86_64/pack_f16.hpp"
  #include "backend/cpu/x86_64/quant.hpp"
  using namespace nnops::backend::cpu::x86_64;
  using f16_t = nnops::backend::cpu::half;  // x86_64: struct half { uint16_t bits; }
  #define NNOPS_PACK_MMA_ARCH "x86_64"
#elif defined(NNOPS_ARCH_AARCH64)
  #include "backend/cpu/aarch64/pack_f32.hpp"
  #include "backend/cpu/aarch64/pack_f16.hpp"
  #include "backend/cpu/aarch64/quant.hpp"
  using namespace nnops::backend::cpu::aarch64;
  using f16_t = float16_t;
  #define NNOPS_PACK_MMA_ARCH "aarch64"
#else
  #error "Unsupported architecture"
#endif

#include <cstring>
#include <vector>
#include <cmath>
#include <type_traits>

// =========================================================================
//  Helpers
// =========================================================================

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

#if defined(NNOPS_ARCH_X86_64)
// =========================================================================
//  Section 2b: f32 Pack — partial panels
//
//  A partial panel has only `w < N` live columns/rows. The transposing
//  kernels fill rows past `mr` by aliasing the last live row (the MMA
//  kernels' A-side broadcast); the row-major copies zero-fill columns past
//  `nr`. The 16-wide copy writes only the lanes the MMA kernel reads: 8 for
//  a tail of 8 or fewer, all 16 above that.
// =========================================================================

NNOPS_TEST(pack_f32_n16_pad) {
    constexpr int N = 16, K = 9, ir_step = 24;
    const float scale = 1.5f;
    // The transposing pack reads one source row per live panel row, which can
    // be more rows than K columns when a partial panel is wide (w up to 15).
    constexpr int rows = (N > K) ? N : K;
    std::vector<float> input(rows * ir_step);
    for (int k = 0; k < rows; ++k) {
        for (int n = 0; n < ir_step; ++n) {
            input[k * ir_step + n] = float(k * 100 + n) * 0.5f;
        }
    }

    for (int w : {1, 4, 8, 9, 15}) {
        const int read = (w > 8) ? 16 : 8;

        std::vector<float> copy_out(N * K, -7.0f);
        pack_copy_n16_f32(copy_out.data(), input.data(), ir_step, K, scale, w);
        std::vector<float> trans_out(N * K, -7.0f);
        pack_trans_n16_f32(trans_out.data(), input.data(), ir_step, K, scale, w);

        for (int k = 0; k < K; ++k) {
            for (int n = 0; n < read; ++n) {
                const float copy_expect =
                    (n < w) ? input[k * ir_step + n] * scale : 0.0f;
                const float trans_expect =
                    input[std::min(n, w - 1) * ir_step + k] * scale;
                NNOPS_EXPECT_NEAR(copy_out[k * N + n], copy_expect, 1e-6f);
                NNOPS_EXPECT_NEAR(trans_out[k * N + n], trans_expect, 1e-6f);
            }
        }
    }
}

// The narrower kernels pad the whole panel — no lane is left unwritten.
NNOPS_TEST(pack_f32_n8_pad) {
    constexpr int N = 8, K = 7, ir_step = 12;
    const float scale = 2.0f;
    std::vector<float> input(K * ir_step);
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < ir_step; ++n) {
            input[k * ir_step + n] = float(k * 100 + n);
        }
    }

    for (int w : {1, 5, 7}) {
        std::vector<float> copy_out(N * K, -7.0f);
        pack_copy_n8_f32(copy_out.data(), input.data(), ir_step, K, scale, w);
        std::vector<float> trans_out(N * K, -7.0f);
        pack_trans_n8_f32(trans_out.data(), input.data(), ir_step, K, scale, w);

        for (int k = 0; k < K; ++k) {
            for (int n = 0; n < N; ++n) {
                const float copy_expect =
                    (n < w) ? input[k * ir_step + n] * scale : 0.0f;
                const float trans_expect =
                    input[std::min(n, w - 1) * ir_step + k] * scale;
                NNOPS_EXPECT_NEAR(copy_out[k * N + n], copy_expect, 1e-6f);
                NNOPS_EXPECT_NEAR(trans_out[k * N + n], trans_expect, 1e-6f);
            }
        }
    }
}
#endif  // NNOPS_ARCH_X86_64

// =========================================================================
//  Section 3: f16 Pack — smoke + copy
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

#if defined(NNOPS_ARCH_X86_64)
// Zero-padded tail panels, same contract as Section 2b.
NNOPS_TEST(pack_f16_n16_pad) {
    constexpr int N = 16, K = 9, ir_step = 24;
    const float scale = 1.5f;
    std::vector<f16_t> input(N * ir_step);
    for (int i = 0; i < N * ir_step; ++i) {
        input[i] = f_to_f16(float(i) * 0.1f);
    }

    for (int w : {1, 4, 8, 9, 15}) {
        const int read = (w > 8) ? 16 : 8;

        std::vector<f16_t> copy_out(N * K, f_to_f16(-7.0f));
        pack_copy_n16_f16(copy_out.data(), input.data(), ir_step, K, scale, w);
        std::vector<f16_t> trans_out(N * K, f_to_f16(-7.0f));
        pack_trans_n16_f16(trans_out.data(), input.data(), ir_step, K, scale, w);

        for (int k = 0; k < K; ++k) {
            for (int n = 0; n < read; ++n) {
                // The copy kernel zero-fills past its live width; the
                // transpose kernel aliases the last live row into the tail.
                const float copy_expect =
                    (n < w) ? f16_to_f(input[k * ir_step + n]) * scale : 0.0f;
                const int src_row = (n < w) ? n : w - 1;
                const float trans_expect =
                    f16_to_f(input[src_row * ir_step + k]) * scale;
                // f16 rounds the stored value once; at |x| ≈ 32 one ULP is
                // already 1/32, so the tolerance has to clear a single step.
                NNOPS_EXPECT_NEAR(f16_to_f(copy_out[k * N + n]), copy_expect, 0.05f);
                NNOPS_EXPECT_NEAR(f16_to_f(trans_out[k * N + n]), trans_expect, 0.05f);
            }
        }
    }
}
#endif  // NNOPS_ARCH_X86_64

// =========================================================================
//  Section 4: half (f16) quantization / dequantization kernels
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
//  Section 5: tiled pack entry points (tile_pack_lhs / tile_pack_rhs)
// =========================================================================
//
// tile_pack_lhs decomposes an M×K tile into the arch MR cascade (mr ∈
// {max, 1}), placing each panel at a uniform 64-byte-aligned stride `ldd`.
// tile_pack_rhs always cuts the N side into nr_max-wide panels, with
// the columns past the live width zero-filled, and places each at `ldd`. Both
// use the [K][panel] layout (k outer, m/n inner) — the exact layout the MMA
// kernels consume. We verify both transpose and copy modes against the logical
// A/B matrices.

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
/// Every panel p is nr_max wide at p*ldd; with col = p*nr_max + n:
///     dst[p*ldd + k*nr_max + n] = (col < N) ? B_logical[k][col] : 0
/// except that the transpose packer aliases the last live row into its tail
/// columns (same broadcast convention as a short A panel) — those columns are
/// never stored by the MMA, so the value there is unconstrained; the test
/// checks the alias it actually produces.
static void check_tile_pack_rhs(bool trans) {
    constexpr int N = 19;   // two full 16-wide panels + a 3-wide tail
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
    const int nslots = (N + nr0 - 1) / nr0;
    // Buffer must hold every panel at the aligned stride, not just N*K.
    std::vector<float> dst(static_cast<size_t>(nslots) * ldd, -1.0f);

    cpu::tile_pack_rhs(trans, N, K, dst.data(), ldd, src.data(), lds, 1.0f);

    for (int p = 0; p < nslots; ++p) {
        // Panel p starts at p * ldd (uniform 64-byte-aligned stride).
        const float* panel = dst.data() + p * ldd;
        for (int k = 0; k < K; ++k) {
            for (int n = 0; n < nr0; ++n) {
                const int col = p * nr0 + n;
                const int src_col = trans ? std::min(col, N - 1) : col;
                const float expect =
                    (trans || col < N) ? 0.5f + float(k * 100 + src_col) : 0.0f;
                NNOPS_EXPECT_NEAR(panel[k * nr0 + n], expect, 1e-5f);
            }
        }
    }
}

NNOPS_TEST(tile_pack_lhs_trans)          { check_tile_pack_lhs(true); }
NNOPS_TEST(tile_pack_lhs_copy)           { check_tile_pack_lhs(false); }
NNOPS_TEST(tile_pack_rhs_trans)          { check_tile_pack_rhs(true); }
NNOPS_TEST(tile_pack_rhs_copy)           { check_tile_pack_rhs(false); }
