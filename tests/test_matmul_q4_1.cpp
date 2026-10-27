/// MatMul with a Q4_1 (llama.cpp block format) right-hand side.
///
/// The RHS is stored as a stream of 20-byte blocks; MatMul decodes it straight
/// into its uniform packed-B panels, so the comparison here is against a dense
/// reference built from the same block stream:
///
///   1. encode known 4-bit quants into blocks (d/m exact in f16),
///   2. run MatMul with the Q4_1 B,
///   3. compute the expected C from the *decoded* B, honouring transpose_b.
///
/// Covered: both physical B layouts, M/N/K tails (K not block-aligned, N not a
/// multiple of the 16-wide panel), both split directions (N > M and M >= N),
/// f32 and f16 A/output, and a batched B where the batch stride is counted in
/// blocks rather than elements.
///
/// Q4_1 rides the same two routes as Q5_1: the packed plan (every M) and, on
/// x86_64 at M ≤ 16 ([N][K]) / M ≤ 4 ([K][N]), the fused small-M kernels that
/// decode straight into the reduction. The last two tests sweep the fused
/// route's edge cases.

#include "nnops/ops/matmul.hpp"
#include "nnops/detail/q4_1.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace nnops;

namespace {

/// Row-major Q4_1 matrix (whole-block row padding) plus its logical values.
struct Q4Matrix {
    std::vector<uint8_t> bytes;
    std::vector<float>   decoded;  ///< rows × cols, row-major
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t blocks_per_row = 0;
    size_t  block_offset = 0;      ///< position of this matrix in @c bytes
};

/// Encode 32 quants into one block exactly as llama.cpp does.
void encode_block(const int q[q4_1_block_elems], float d, float m,
                  uint8_t out[q4_1_block_bytes]) {
    const half dh = float_to_half(d);
    const half mh = float_to_half(m);
    const uint16_t d_bits = backend::cpu::half_to_bits(dh);
    const uint16_t m_bits = backend::cpu::half_to_bits(mh);

    uint8_t qs[16];
    for (int j = 0; j < 16; ++j) {
        const int q0 = q[j];
        const int q1 = q[j + 16];
        qs[j] = static_cast<uint8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));
    }

    std::memcpy(out + 0, &d_bits, 2);
    std::memcpy(out + 2, &m_bits, 2);
    std::memcpy(out + 4, qs, 16);
}

/// Build one [rows, cols] Q4_1 matrix. d/m and the resulting products are
/// exact in f16, so the decoded values are exact in f32 and f16 alike.
Q4Matrix build_q4_matrix(int64_t rows, int64_t cols, uint32_t seed) {
    Q4Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.blocks_per_row = (cols + q4_1_block_elems - 1) / q4_1_block_elems;
    m.bytes.assign(static_cast<size_t>(rows * m.blocks_per_row
                                       * static_cast<int64_t>(q4_1_block_bytes)), 0);
    m.decoded.assign(static_cast<size_t>(rows * cols), 0.0f);

    const float d_choices[4] = {0.25f, 0.5f, 0.75f, 1.0f};
    const float m_choices[5] = {-2.0f, -1.5f, -1.0f, -0.5f, 0.0f};

    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t b = 0; b < m.blocks_per_row; ++b) {
            int q[q4_1_block_elems];
            for (int j = 0; j < q4_1_block_elems; ++j) {
                q[j] = static_cast<int>((seed + r * 13 + b * 7 + j * 5) & 0x0F);
            }
            const float d = d_choices[(seed + r + b) % 4];
            const float mn = m_choices[(seed + r + 2 * b) % 5];

            encode_block(q, d, mn,
                         m.bytes.data() + static_cast<size_t>(
                             (r * m.blocks_per_row + b)
                             * static_cast<int64_t>(q4_1_block_bytes)));

            const half dh = float_to_half(d);
            const half mh = float_to_half(mn);
            const int64_t base = r * cols + b * q4_1_block_elems;
            const int n = static_cast<int>(
                std::min<int64_t>(q4_1_block_elems, cols - b * q4_1_block_elems));
            for (int j = 0; j < n; ++j) {
                m.decoded[static_cast<size_t>(base + j)] =
                    half_to_float(dh) * static_cast<float>(q[j]) + half_to_float(mh);
            }
        }
    }
    return m;
}

/// Append @p src to @p dst, recording where it landed (the batched-B case).
void append_q4_matrix(std::vector<uint8_t>& dst, Q4Matrix& src) {
    src.block_offset = dst.size();
    dst.insert(dst.end(), src.bytes.begin(), src.bytes.end());
}

std::vector<float> gen_floats(int64_t n, uint32_t seed) {
    std::vector<float> v(static_cast<size_t>(n));
    uint32_t s = seed;
    for (int64_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        v[static_cast<size_t>(i)] =
            -1.0f + 2.0f * static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
    }
    return v;
}

/// Expected C = A × B, with A in logical [M][K] order and B given by its
/// decoded values and physical layout.
template <class T>
std::vector<float> reference_matmul(const T* a, const Q4Matrix& b,
                                    int64_t M, int64_t N, int64_t K,
                                    bool transpose_b) {
    std::vector<float> c(static_cast<size_t>(M * N), 0.0f);
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                const float bv = b.decoded[static_cast<size_t>(
                    transpose_b ? (n * b.cols + k) : (k * b.cols + n))];
                if constexpr (std::is_same_v<T, float>) {
                    acc += a[m * K + k] * bv;
                } else {
                    acc += half_to_float(a[m * K + k]) *
                           half_to_float(float_to_half(bv));
                }
            }
            if constexpr (std::is_same_v<T, half>) {
                c[static_cast<size_t>(m * N + n)] = half_to_float(float_to_half(acc));
            } else {
                c[static_cast<size_t>(m * N + n)] = acc;
            }
        }
    }
    return c;
}

/// Physical [K][M] copy of a logical [M][K] buffer (transpose_a storage).
template <class T>
std::vector<T> transpose_a_storage(const std::vector<T>& logical, int64_t M, int64_t K) {
    std::vector<T> phys(static_cast<size_t>(M * K));
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            phys[static_cast<size_t>(k * M + m)] = logical[static_cast<size_t>(m * K + k)];
        }
    }
    return phys;
}

struct Case {
    int64_t M, N, K;
    bool transpose_b;
    bool f16;
    bool transpose_a;
    float beta = 0.0f;                         ///< 0/1/other: overwrite / accumulate / scale
    float alpha = 1.0f;                        ///< scales the product only
    EpilogueActivateType ep = EpilogueActivateType::None;
    bool bias = false;                         ///< adds a per-column [N] input
};

/// Deterministic prior C contents, so beta != 0 has something to act on.
float ramp_value(int64_t i) {
    return 0.25f * static_cast<float>(i % 11) - 1.0f;
}

float apply_expected_epilogue(float v, EpilogueActivateType ep) {
    switch (ep) {
    case EpilogueActivateType::Relu:  return std::max(v, 0.0f);
    case EpilogueActivateType::Relu6: return std::min(std::max(v, 0.0f), 6.0f);
    default:                          return v;
    }
}

void run_case(const Case& c) {
    const int64_t M = c.M, N = c.N, K = c.K;

    // A in logical [M][K] order, then the physical copy the flag asks for.
    const std::vector<float> a_f32 = gen_floats(M * K, 0x51u + static_cast<uint32_t>(M + N + K));
    std::vector<half> a_f16_logical;
    if (c.f16) {
        a_f16_logical.resize(a_f32.size());
        for (size_t i = 0; i < a_f32.size(); ++i) {
            a_f16_logical[i] = float_to_half(a_f32[i]);
        }
    }

    std::vector<float> a_f32_phys = c.transpose_a ? transpose_a_storage(a_f32, M, K)
                                                  : a_f32;
    std::vector<half> a_f16_phys;
    if (c.f16) {
        a_f16_phys = c.transpose_a ? transpose_a_storage(a_f16_logical, M, K)
                                   : a_f16_logical;
    }

    // Physical B: [N][K] when transposed, [K][N] otherwise.
    const Q4Matrix b = c.transpose_b ? build_q4_matrix(N, K, 0x1234u)
                                     : build_q4_matrix(K, N, 0x1234u);
    const int64_t brows = c.transpose_b ? N : K;
    const int64_t bcols = c.transpose_b ? K : N;

    const int64_t ashape_mk[] = {M, K};
    const int64_t ashape_km[] = {K, M};
    const std::span<const int64_t> ashape = c.transpose_a
        ? std::span<const int64_t>(ashape_km, 2)
        : std::span<const int64_t>(ashape_mk, 2);
    const int64_t bshape[] = {brows, bcols};

    const bool a16 = c.f16;
    TensorView a(ashape, a16 ? DataType::f16 : DataType::f32,
                 a16 ? static_cast<void*>(a_f16_phys.data())
                     : static_cast<void*>(a_f32_phys.data()));
    TensorView bv(bshape, DataType::q4_1, const_cast<uint8_t*>(b.bytes.data()));

    MatMulAttributes attrs{};
    attrs.transpose_b = c.transpose_b;
    attrs.transpose_a = c.transpose_a;
    attrs.beta = c.beta;
    attrs.alpha = c.alpha;
    attrs.epilogue.type = c.ep;
    auto op = MatMul::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {a.desc(), bv.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].dtype, a16 ? DataType::f16 : DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].dims[0], M);
    NNOPS_EXPECT_EQ(descs[0].dims[1], N);

    std::vector<float> out_f32;
    std::vector<half>  out_f16;
    TensorView output = [&]() {
        if (a16) {
            out_f16.resize(static_cast<size_t>(M * N));
            for (int64_t i = 0; i < M * N; ++i) {
                out_f16[static_cast<size_t>(i)] = float_to_half(ramp_value(i));
            }
            return nnops::test::make_planar(descs[0], out_f16.data());
        }
        out_f32.resize(static_cast<size_t>(M * N));
        for (int64_t i = 0; i < M * N; ++i) {
            out_f32[static_cast<size_t>(i)] = ramp_value(i);
        }
        return nnops::test::make_planar(descs[0], out_f32.data());
    }();

    // Optional [N] bias in the output dtype, deterministic and exactly
    // representable in f16.
    std::vector<float> bias_f32;
    std::vector<half>  bias_f16;
    const int64_t bias_shape[] = {N};
    if (c.bias) {
        bias_f32.resize(static_cast<size_t>(N));
        bias_f16.resize(static_cast<size_t>(N));
        for (int64_t n = 0; n < N; ++n) {
            const float v = 0.125f * static_cast<float>(n % 9) - 0.5f;
            bias_f32[static_cast<size_t>(n)] = v;
            bias_f16[static_cast<size_t>(n)] = float_to_half(v);
        }
    }
    const TensorView bias_view(
        bias_shape, a16 ? DataType::f16 : DataType::f32,
        a16 ? static_cast<void*>(bias_f16.data())
            : static_cast<void*>(bias_f32.data()));

    const TensorView ins2[] = {a, bv};
    const TensorView ins3[] = {a, bv, bias_view};
    op->compute(output, c.bias ? std::span<const TensorView>(ins3, 3)
                               : std::span<const TensorView>(ins2, 2));

    std::vector<float> want = a16
        ? reference_matmul<half>(a_f16_logical.data(), b, M, N, K, c.transpose_b)
        : reference_matmul<float>(a_f32.data(), b, M, N, K, c.transpose_b);

    // Post-accumulation semantics shared by every route: C = alpha·(A×B)
    // + beta·C_old, then the clamp/epilogue. alpha scales the product only.
    for (int64_t i = 0; i < M * N; ++i) {
        float v = want[static_cast<size_t>(i)];
        if (c.alpha != 1.0f) {
            v *= c.alpha;
        }
        if (c.beta != 0.0f) {
            const float old = a16
                ? half_to_float(float_to_half(ramp_value(i))) : ramp_value(i);
            v += c.beta * old;
        }
        if (c.bias) {
            // Unscaled by alpha, added before the clamp/epilogue.
            const int64_t n = i % N;
            v += a16 ? half_to_float(float_to_half(bias_f32[static_cast<size_t>(n)]))
                     : bias_f32[static_cast<size_t>(n)];
        }
        want[static_cast<size_t>(i)] = apply_expected_epilogue(v, c.ep);
    }

    // The f16 route accumulates in the arch's native precision (fp32 on x86,
    // native fp16 on aarch64), so take the shared arch-aware tolerance instead
    // of a fixed value: near-zero results from a long cancelling reduction are
    // exactly where the wider aarch64 error shows up.
    const float rtol = a16 ? nnops::test::kF16AccumTol : 1e-4f;
    const float atol = a16 ? nnops::test::kF16AccumTol : 1e-3f;
    float worst = 0.0f;
    for (int64_t i = 0; i < M * N; ++i) {
        const float got = a16
            ? half_to_float(out_f16[static_cast<size_t>(i)])
            : out_f32[static_cast<size_t>(i)];
        const float diff = std::abs(got - want[static_cast<size_t>(i)]);
        worst = std::max(worst, diff);
        NNOPS_EXPECT_TRUE(diff <= atol + rtol * std::abs(want[static_cast<size_t>(i)]));
    }
    (void)worst;
}

/// Batched: A [2, M, K] × B [2, N, K] → C [2, M, N]. The B batch stride is
/// N·ceil(K/32) *blocks*, which exercises the block-unit batch offset.
NNOPS_TEST(matmul_q4_1_batched) {
    constexpr int64_t B = 2, M = 4, N = 6, K = 70;
    constexpr bool transpose_b = true;

    std::vector<uint8_t> bbytes;
    Q4Matrix b0 = build_q4_matrix(N, K, 0x77u);
    Q4Matrix b1 = build_q4_matrix(N, K, 0x99u);
    append_q4_matrix(bbytes, b0);
    append_q4_matrix(bbytes, b1);

    const std::vector<float> a0 = gen_floats(M * K, 0x100u);
    const std::vector<float> a1 = gen_floats(M * K, 0x200u);
    std::vector<float> adata = a0;
    adata.insert(adata.end(), a1.begin(), a1.end());

    const int64_t ashape[] = {B, M, K};
    const int64_t bshape[] = {B, N, K};
    TensorView a(ashape, DataType::f32, adata.data());
    TensorView bv(bshape, DataType::q4_1, bbytes.data());

    MatMulAttributes attrs{};
    attrs.transpose_b = transpose_b;
    auto op = MatMul::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {a.desc(), bv.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(3));
    NNOPS_EXPECT_EQ(descs[0].dims[0], B);
    NNOPS_EXPECT_EQ(descs[0].dims[1], M);
    NNOPS_EXPECT_EQ(descs[0].dims[2], N);

    std::vector<float> out(static_cast<size_t>(B * M * N));
    auto output = nnops::test::make_planar(descs[0], out.data());

    const TensorView ins[] = {a, bv};
    op->compute(output, ins);

    for (int64_t bi = 0; bi < B; ++bi) {
        const Q4Matrix& bm = bi == 0 ? b0 : b1;
        const std::vector<float> want = reference_matmul<float>(
            adata.data() + bi * M * K, bm, M, N, K, transpose_b);
        for (int64_t i = 0; i < M * N; ++i) {
            const float got = out[static_cast<size_t>(bi * M * N + i)];
            NNOPS_EXPECT_TRUE(std::abs(got - want[static_cast<size_t>(i)]) <=
                              1e-3f + 1e-4f * std::abs(want[static_cast<size_t>(i)]));
        }
    }
}

}  // namespace

NNOPS_TEST(matmul_q4_1_f32) {
    // N > M (split on N), N a non-multiple of the 16-wide panel, K off-block.
    run_case({5, 37, 70, /*transpose_b=*/false, /*f16=*/false, /*transpose_a=*/false});
    // M >= N (split on M), Linear-style [N][K] weight.
    run_case({37, 5, 70, /*transpose_b=*/true,  /*f16=*/false, /*transpose_a=*/false});
    // Block-aligned N and K, wider than one panel.
    run_case({24, 71, 160, /*transpose_b=*/true, /*f16=*/false, /*transpose_a=*/false});
    // Small square with K below one block.
    run_case({9, 9, 33, /*transpose_b=*/false, /*f16=*/false, /*transpose_a=*/false});
    // Transposed A: physically [K][M], so the pack-A route runs as well.
    run_case({17, 23, 70, /*transpose_b=*/false, /*f16=*/false, /*transpose_a=*/true});
    run_case({17, 23, 70, /*transpose_b=*/true,  /*f16=*/false, /*transpose_a=*/true});
    // alpha rides the Q4_1 RHS pack (every decoded weight scaled), so the
    // packed route needs no post-pass; both physical layouts covered.
    run_case({5, 37, 70, /*transpose_b=*/false, /*f16=*/false, /*transpose_a=*/false,
              0.0f, 2.5f});
    run_case({37, 5, 70, /*transpose_b=*/true,  /*f16=*/false, /*transpose_a=*/false,
              0.0f, -1.5f});
    run_case({24, 71, 160, /*transpose_b=*/true, /*f16=*/false, /*transpose_a=*/false,
              0.5f, 0.5f});
    // A bias enters the accumulator as beta·C_old + bias before the first
    // k-block, so the overwriting (beta = 0) case must still keep it, and the
    // f16 panel scaling of alpha stays separate.
    run_case({5, 37, 70, /*transpose_b=*/false, /*f16=*/false, /*transpose_a=*/false,
              0.0f, 1.0f, EpilogueActivateType::None, /*bias=*/true});
    run_case({37, 5, 70, /*transpose_b=*/true,  /*f16=*/false, /*transpose_a=*/false,
              0.5f, 2.0f, EpilogueActivateType::None, /*bias=*/true});
}

/// Decode sizes: M at or below the fused route's limits, so these are the
/// shapes a decode step actually has.
NNOPS_TEST(matmul_q4_1_small_m) {
    run_case({1, 37, 70, /*transpose_b=*/false, false, false});
    run_case({2, 37, 70, /*transpose_b=*/false, false, false});
    run_case({2, 64, 33, /*transpose_b=*/false, false, false});   // K tail
    run_case({1, 71, 160, /*transpose_b=*/true, false, false});
    run_case({4, 71, 160, /*transpose_b=*/true, false, false});
    run_case({3, 41, 33, /*transpose_b=*/true, false, false});    // partial m rows
    run_case({8, 130, 70, /*transpose_b=*/true, false, false});   // 16-wide panel tails
    run_case({4, 37, 70, /*transpose_b=*/true, false, false, 1.0f});
    run_case({2, 37, 70, /*transpose_b=*/false, false, false, 0.5f});
    run_case({2, 37, 70, /*transpose_b=*/false, false, false, 0.0f, 1.0f,
              EpilogueActivateType::Relu6});
    run_case({7, 37, 70, /*transpose_b=*/true, false, false, 0.0f, 2.0f,
              EpilogueActivateType::Relu});
    run_case({2, 37, 70, /*transpose_b=*/false, false, false, 0.0f, 1.0f,
              EpilogueActivateType::None, /*bias=*/true});
    run_case({8, 71, 160, /*transpose_b=*/true, false, false, 0.5f, 1.5f,
              EpilogueActivateType::Relu6, /*bias=*/true});
}

/// The small-M fused route: Q4_1 blocks decoded straight into the reduction
/// (no packed-B panel, and no 5th-bit mask to spread). [K][N] B runs the
/// outer-product kernel (MR ≤ 2), [N][K] B the dot-product one (M ≤ 16, two
/// 8-row panels); the cases cover row aliasing, K tails, N tiles, beta and the
/// clamped epilogue.
NNOPS_TEST(matmul_q4_1_fused_small_m) {
    run_case({1, 37, 70, /*transpose_b=*/false, false, false});
    run_case({2, 37, 70, /*transpose_b=*/false, false, false});
    run_case({2, 64, 33, /*transpose_b=*/false, false, false});   // K tail
    run_case({1, 71, 160, /*transpose_b=*/true, false, false});
    run_case({2, 71, 160, /*transpose_b=*/true, false, false});
    run_case({3, 41, 33, /*transpose_b=*/true, false, false});    // MR=4, partial rows
    run_case({5, 37, 95, /*transpose_b=*/true, false, false});    // MR=8, K tail
    run_case({8, 130, 70, /*transpose_b=*/true, false, false});   // 64-wide tiles + N tail
    run_case({16, 71, 95, /*transpose_b=*/true, false, false});   // two 8-row panels
    run_case({16, 37, 33, /*transpose_b=*/true, false, false, 0.5f});  // panels + beta

    // beta ∉ {0}: inherited C and the in-place scale.
    run_case({4, 37, 70, /*transpose_b=*/true, false, false, 1.0f});
    run_case({2, 37, 70, /*transpose_b=*/false, false, false, 1.0f});
    run_case({5, 37, 70, /*transpose_b=*/true, false, false, 0.5f});

    // Clamp folds into the final write.
    run_case({2, 37, 70, /*transpose_b=*/false, false, false, 0.0f, 1.0f,
              EpilogueActivateType::Relu6});
    run_case({7, 37, 70, /*transpose_b=*/true, false, false, 0.0f, 1.0f,
              EpilogueActivateType::Relu});

    // alpha in the fused kernels, which scale the finished dot (full blocks
    // and the scalar k tail alike), with and without beta.
    run_case({16, 37, 33, /*transpose_b=*/true, false, false, 0.0f, 2.0f});
    run_case({2, 37, 70, /*transpose_b=*/false, false, false, 0.0f, -0.5f});
    run_case({5, 37, 70, /*transpose_b=*/true, false, false, 0.5f, 3.0f});

    // The bias primes the tile before the fused kernel runs, so the accumulate
    // mode has to be used even at beta = 0.
    run_case({2, 37, 70, /*transpose_b=*/false, false, false, 0.0f, 1.0f,
              EpilogueActivateType::None, /*bias=*/true});
    run_case({8, 71, 160, /*transpose_b=*/true, false, false, 0.0f, 2.0f,
              EpilogueActivateType::None, /*bias=*/true});
    run_case({16, 37, 33, /*transpose_b=*/true, false, false, 0.5f, 1.5f,
              EpilogueActivateType::Relu6, /*bias=*/true});
}

NNOPS_TEST(matmul_q4_1_f16) {
    run_case({5, 37, 70, /*transpose_b=*/false, /*f16=*/true, /*transpose_a=*/false});
    run_case({37, 5, 70, /*transpose_b=*/true,  /*f16=*/true, /*transpose_a=*/false});
    run_case({24, 71, 160, /*transpose_b=*/true, /*f16=*/true, /*transpose_a=*/false});
    run_case({17, 23, 70, /*transpose_b=*/true, /*f16=*/true, /*transpose_a=*/true});
    // The f16 panel is scaled in f16 after the decode, so the rounding is one
    // more step than the f32 route's — same arch-aware tolerance.
    run_case({5, 37, 70, /*transpose_b=*/false, /*f16=*/true, /*transpose_a=*/false,
              0.0f, 2.0f});
    run_case({37, 5, 70, /*transpose_b=*/true,  /*f16=*/true, /*transpose_a=*/false,
              0.0f, 0.5f});
    run_case({5, 37, 70, /*transpose_b=*/false, /*f16=*/true, /*transpose_a=*/false,
              0.0f, 1.0f, EpilogueActivateType::None, /*bias=*/true});
    run_case({37, 5, 70, /*transpose_b=*/true,  /*f16=*/true, /*transpose_a=*/false,
              0.5f, 1.0f, EpilogueActivateType::None, /*bias=*/true});
}
