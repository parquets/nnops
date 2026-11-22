/// MatMul with a Q4_1 (llama.cpp block format) right-hand side: 32 values in a
/// 20-byte block (fp16 d, fp16 m, 16 nibble bytes), decoded inside the RHS pack
/// so the weight never becomes a dense buffer.
///
/// The reference comes from the same block stream, not the operator: encode
/// exact-in-f16 d/m and quants, run MatMul, then recompute C from the decoded B
/// honouring transpose_b. Those decoded values are exact in f32 and f16 alike.
///
/// Covered: both physical B layouts, M/N/K tails (K off the 32-value block, N
/// off the RHS panel), both split directions (N > M, M >= N), f32 and f16
/// A/output, alpha/beta, the clamp epilogue, a [N] bias, and a batched B whose
/// per-batch offset is counted in blocks rather than elements.
///
/// Two routes share the file: the packed plan serves every M; on x86_64 the
/// fused small-M kernels (fused_max_m_trans = 16 for [N][K], _copy = 8 for
/// [K][N]) decode straight into the reduction. matmul_q4_1_small_m and
/// matmul_q4_1_fused_small_m sweep that route's edges.

#include "nnops/ops/matmul.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/detail/q4.hpp"
#include "backend/cpu/matmul/matmul_helper.h"
#if defined(NNOPS_ARCH_SME)
#include "backend/cpu/aarch64/sme/mma_sme.hpp" // available_sme — the ZA route's gate
#include "detail/bench_knobs.hpp"              // disable_sme — the ZA / NEON A/B pin
#endif
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

using namespace nnops;
using nnops::backend::cpu::bfloat16;
using nnops::test::bf16_to_f32;
using nnops::test::f32_to_bf16;

namespace {

#if defined(NNOPS_ARCH_SME)
/// A fixed-size claim-per-item pool, so the N-split runs on more than one
/// worker (the default context is sequential).
struct SimplePool {
    explicit SimplePool(int nthreads) : nthreads_(nthreads)
    {
        cpu.parallel_for = [this](int64_t begin, int64_t end, const ParallelForBody& body) {
            this->parallel_for(begin, end, body);
        };
        cpu.num_threads = [this] { return nthreads_; };
        cpu.thread_id = [] { return current_thread_id_; };
    }

    void parallel_for(int64_t begin, int64_t end, const ParallelForBody& body)
    {
        std::atomic<int64_t> next{begin};
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(nthreads_));
        for (int t = 0; t < nthreads_; ++t) {
            workers.emplace_back([&, t]() {
                current_thread_id_ = t;
                for (;;) {
                    const int64_t i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= end) {
                        break;
                    }
                    body(i);
                }
            });
        }
        for (auto& w : workers) {
            w.join();
        }
    }

    int nthreads_;
    CpuBackend cpu;
    inline static thread_local int current_thread_id_ = 0;
};

/// Pin the ZA route on or off for one test body, restoring the previous knob.
struct SmePin {
    explicit SmePin(bool on)
        : previous(detail::disable_sme.load(std::memory_order_relaxed))
    {
        detail::disable_sme.store(on ? 0 : 1, std::memory_order_relaxed);
    }
    ~SmePin() { detail::disable_sme.store(previous, std::memory_order_relaxed); }
    SmePin(const SmePin&) = delete;
    SmePin& operator=(const SmePin&) = delete;
    int previous;
};
#endif

/// Row-major Q4_1 matrix (whole-block row padding) plus its logical values.
struct Q4Matrix {
    std::vector<uint8_t> bytes;
    std::vector<float> decoded; ///< rows × cols, row-major
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t blocks_per_row = 0;
    size_t block_offset = 0; ///< position of this matrix in @c bytes
};

/// Encode 32 quants into one block exactly as llama.cpp does.
void encode_block(const int q[q4_1_block_elems], float d, float m, uint8_t* out, DataType fmt)
{
    const uint16_t d_bits = backend::cpu::half_to_bits(float_to_half(d));
    const uint16_t m_bits = backend::cpu::half_to_bits(float_to_half(m));

    if (fmt == DataType::q8_0) {
        std::memcpy(out + 0, &d_bits, 2);
        for (int j = 0; j < q8_0_block_elems; ++j) {
            out[2 + j] = static_cast<uint8_t>(static_cast<int8_t>(q[j]));
        }
        return;
    }

    uint8_t qs[16];
    for (int j = 0; j < 16; ++j) {
        const int q0 = q[j];
        const int q1 = q[j + 16];
        qs[j] = static_cast<uint8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));
    }

    std::memcpy(out + 0, &d_bits, 2);
    if (q4_has_min(fmt)) {
        std::memcpy(out + 2, &m_bits, 2);
        std::memcpy(out + 4, qs, 16);
    } else {
        std::memcpy(out + 2, qs, 16);
    }
}

/// Build one [rows, cols] Q4 matrix with f16-exact d/m, so the decoded values
/// are exact too. `fmt` selects Q4_0 (18-byte, nibbles − 8) or Q4_1.
Q4Matrix build_q4_matrix(int64_t rows, int64_t cols, uint32_t seed, DataType fmt = DataType::q4_1)
{
    Q4Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.blocks_per_row = (cols + q4_1_block_elems - 1) / q4_1_block_elems;
    const int64_t bb = static_cast<int64_t>(data_type_block_bytes(fmt));
    m.bytes.assign(static_cast<size_t>(rows * m.blocks_per_row * bb), 0);
    m.decoded.assign(static_cast<size_t>(rows * cols), 0.0f);

    const float d_choices[4] = {0.25f, 0.5f, 0.75f, 1.0f};
    const float m_choices[5] = {-2.0f, -1.5f, -1.0f, -0.5f, 0.0f};

    const bool wide = (fmt == DataType::q8_0); // int8 quants, not 4-bit

    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t b = 0; b < m.blocks_per_row; ++b) {
            int q[q4_1_block_elems];
            for (int j = 0; j < q4_1_block_elems; ++j) {
                q[j] = wide ? static_cast<int>((seed + r * 13 + b * 7 + j * 5) % 255) - 127
                            : static_cast<int>((seed + r * 13 + b * 7 + j * 5) & 0x0F);
            }
            const float d = d_choices[(seed + r + b) % 4];
            const float mn = m_choices[(seed + r + 2 * b) % 5];

            encode_block(q, d, mn,
                         m.bytes.data() + static_cast<size_t>((r * m.blocks_per_row + b) * bb),
                         fmt);

            const half dh = float_to_half(d);
            const half mh = float_to_half(mn);
            const int64_t base = r * cols + b * q4_1_block_elems;
            const int n =
                static_cast<int>(std::min<int64_t>(q4_1_block_elems, cols - b * q4_1_block_elems));
            for (int j = 0; j < n; ++j) {
                const float qv = wide ? static_cast<float>(q[j]) :
                                 q4_has_min(fmt) ? static_cast<float>(q[j]) :
                                                   static_cast<float>(q[j] - 8);
                m.decoded[static_cast<size_t>(base + j)] =
                    half_to_float(dh) * qv + (q4_has_min(fmt) ? half_to_float(mh) : 0.0f);
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

/// Expected C = A × B from A in logical [M][K] order and B's decoded values.
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
                } else if constexpr (std::is_same_v<T, half>) {
                    acc += half_to_float(a[m * K + k]) *
                           half_to_float(float_to_half(bv));
                } else {
                    // bf16: the decode rounds each weight once, then the MMA
                    // accumulates those widths in f32. static_cast goes through
                    // bfloat16's float conversion; bf16_to_float takes raw bits.
                    acc += static_cast<float>(a[m * K + k]) *
                           bf16_to_float(float_to_bf16(bv));
                }
            }
            if constexpr (std::is_same_v<T, half>) {
                c[static_cast<size_t>(m * N + n)] = half_to_float(float_to_half(acc));
            } else if constexpr (std::is_same_v<T, bfloat16>) {
                c[static_cast<size_t>(m * N + n)] = bf16_to_float(float_to_bf16(acc));
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
    float beta = 0.0f;  ///< 0/1/other: overwrite / accumulate / scale
    float alpha = 1.0f; ///< scales the product only
    EpilogueActivateType ep = EpilogueActivateType::None;
    bool bias = false;             ///< adds a per-column [N] input
    bool bf16 = false;             ///< aarch64 only; excludes f16
    DataType fmt = DataType::q4_1; ///< B block format
};

/// Deterministic prior C contents, so beta != 0 has something to act on.
float ramp_value(int64_t i) { return 0.25f * static_cast<float>(i % 11) - 1.0f; }

float apply_expected_epilogue(float v, EpilogueActivateType ep)
{
    switch (ep) {
    case EpilogueActivateType::Relu:
        return std::max(v, 0.0f);
    case EpilogueActivateType::Relu6:
        return std::min(std::max(v, 0.0f), 6.0f);
    default:
        return v;
    }
}

void run_case(const Case& c)
{
    const int64_t M = c.M, N = c.N, K = c.K;

    const std::vector<float> a_f32 = gen_floats(M * K, 0x51u + static_cast<uint32_t>(M + N + K));
    std::vector<half> a_f16_logical;
    if (c.f16) {
        a_f16_logical.resize(a_f32.size());
        for (size_t i = 0; i < a_f32.size(); ++i) {
            a_f16_logical[i] = float_to_half(a_f32[i]);
        }
    }
    std::vector<bfloat16> a_bf16_logical;
    if (c.bf16) {
        a_bf16_logical = f32_to_bf16(a_f32);
    }

    std::vector<float> a_f32_phys = c.transpose_a ? transpose_a_storage(a_f32, M, K) : a_f32;
    std::vector<half> a_f16_phys;
    if (c.f16) {
        a_f16_phys = c.transpose_a ? transpose_a_storage(a_f16_logical, M, K) : a_f16_logical;
    }
    std::vector<bfloat16> a_bf16_phys;
    if (c.bf16) {
        a_bf16_phys = c.transpose_a ? transpose_a_storage(a_bf16_logical, M, K) : a_bf16_logical;
    }

    // Physical B: [N][K] when transposed, [K][N] otherwise.
    const Q4Matrix b = c.transpose_b ? build_q4_matrix(N, K, 0x1234u, c.fmt) :
                                       build_q4_matrix(K, N, 0x1234u, c.fmt);
    const int64_t brows = c.transpose_b ? N : K;
    const int64_t bcols = c.transpose_b ? K : N;

    const int64_t ashape_mk[] = {M, K};
    const int64_t ashape_km[] = {K, M};
    const std::span<const int64_t> ashape = c.transpose_a ? std::span<const int64_t>(ashape_km, 2) :
                                                            std::span<const int64_t>(ashape_mk, 2);
    const int64_t bshape[] = {brows, bcols};

    const bool a16 = c.f16 || c.bf16;
    const DataType a_dt = c.bf16 ? DataType::bf16 : (c.f16 ? DataType::f16 : DataType::f32);
    void* const a_ptr = c.bf16 ? static_cast<void*>(a_bf16_phys.data()) :
                        c.f16  ? static_cast<void*>(a_f16_phys.data()) :
                                 static_cast<void*>(a_f32_phys.data());
    TensorView a(ashape, a_dt, a_ptr);
    TensorView bv(bshape, c.fmt, const_cast<uint8_t*>(b.bytes.data()));

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
    NNOPS_EXPECT_EQ(descs[0].dtype, a_dt);
    NNOPS_EXPECT_EQ(descs[0].dims[0], M);
    NNOPS_EXPECT_EQ(descs[0].dims[1], N);

    std::vector<float>    out_f32;
    std::vector<half>     out_f16;
    std::vector<bfloat16> out_bf16;
    TensorView output = [&]() {
        if (c.bf16) {
            out_bf16.resize(static_cast<size_t>(M * N));
            for (int64_t i = 0; i < M * N; ++i) {
                out_bf16[static_cast<size_t>(i)] =
                    backend::cpu::float_to_bfloat16(ramp_value(i));
            }
            return nnops::test::make_channel_first(descs[0], out_bf16.data());
        }
        if (c.f16) {
            out_f16.resize(static_cast<size_t>(M * N));
            for (int64_t i = 0; i < M * N; ++i) {
                out_f16[static_cast<size_t>(i)] = float_to_half(ramp_value(i));
            }
            return nnops::test::make_channel_first(descs[0], out_f16.data());
        }
        out_f32.resize(static_cast<size_t>(M * N));
        for (int64_t i = 0; i < M * N; ++i) {
            out_f32[static_cast<size_t>(i)] = ramp_value(i);
        }
        return nnops::test::make_channel_first(descs[0], out_f32.data());
    }();

    // Optional [N] bias in the output dtype, exactly representable in each fp
    // type at these values.
    std::vector<float>    bias_f32;
    std::vector<half>     bias_f16;
    std::vector<bfloat16> bias_bf16;
    const int64_t bias_shape[] = {N};
    if (c.bias) {
        bias_f32.resize(static_cast<size_t>(N));
        bias_f16.resize(static_cast<size_t>(N));
        bias_bf16.resize(static_cast<size_t>(N));
        for (int64_t n = 0; n < N; ++n) {
            const float v = 0.125f * static_cast<float>(n % 9) - 0.5f;
            bias_f32[static_cast<size_t>(n)] = v;
            bias_f16[static_cast<size_t>(n)] = float_to_half(v);
            bias_bf16[static_cast<size_t>(n)] =
                backend::cpu::float_to_bfloat16(v);
        }
    }
    void* const bias_ptr = c.bf16 ? static_cast<void*>(bias_bf16.data())
                         : c.f16 ? static_cast<void*>(bias_f16.data())
                                 : static_cast<void*>(bias_f32.data());
    const TensorView bias_view(bias_shape, a_dt, bias_ptr);

    const TensorView ins2[] = {a, bv};
    const TensorView ins3[] = {a, bv, bias_view};
    op->compute(output, c.bias ? std::span<const TensorView>(ins3, 3)
                               : std::span<const TensorView>(ins2, 2));

    std::vector<float> want = c.bf16
        ? reference_matmul<bfloat16>(a_bf16_logical.data(), b, M, N, K, c.transpose_b)
        : c.f16
            ? reference_matmul<half>(a_f16_logical.data(), b, M, N, K, c.transpose_b)
            : reference_matmul<float>(a_f32.data(), b, M, N, K, c.transpose_b);

    // Every route shares these post-accumulation semantics: C = alpha·(A×B)
    // + beta·C_old, then the bias and the clamp/epilogue.
    for (int64_t i = 0; i < M * N; ++i) {
        float v = want[static_cast<size_t>(i)];
        if (c.alpha != 1.0f) {
            v *= c.alpha;
        }
        if (c.beta != 0.0f) {
            const float old = c.bf16
                ? backend::cpu::bfloat16_to_float(backend::cpu::float_to_bfloat16(ramp_value(i)))
                : c.f16 ? half_to_float(float_to_half(ramp_value(i)))
                        : ramp_value(i);
            v += c.beta * old;
        }
        if (c.bias) {
            // Unscaled by alpha, added before the clamp/epilogue.
            const int64_t n = i % N;
            const float bvn = bias_f32[static_cast<size_t>(n)];
            v += c.bf16
                ? backend::cpu::bfloat16_to_float(backend::cpu::float_to_bfloat16(bvn))
                : c.f16 ? half_to_float(float_to_half(bvn)) : bvn;
        }
        want[static_cast<size_t>(i)] = apply_expected_epilogue(v, c.ep);
    }

    // The f16 route accumulates in the arch's native precision (fp32 on x86,
    // fp16 on aarch64), so use the shared arch-aware tolerance, not a fixed
    // one: near-zero cancelling sums are where the wider aarch64 error shows.
    // bf16 carries its partial sums through a bf16 C between k-blocks, so its
    // tolerance is wider still (test_helpers.hpp).
    const float accum_tol = c.bf16 ? nnops::test::kBF16AccumTol
                          : a16     ? nnops::test::kF16AccumTol
                                    : 0.0f;
    const float rtol = a16 ? accum_tol : 1e-4f;
    const float atol = a16 ? accum_tol : 1e-3f;
    float worst = 0.0f;
    for (int64_t i = 0; i < M * N; ++i) {
        const float got = c.bf16
            ? backend::cpu::bfloat16_to_float(out_bf16[static_cast<size_t>(i)])
            : c.f16 ? half_to_float(out_f16[static_cast<size_t>(i)])
                    : out_f32[static_cast<size_t>(i)];
        const float diff = std::abs(got - want[static_cast<size_t>(i)]);
        worst = std::max(worst, diff);
        if (!(diff <= atol + rtol * std::abs(want[static_cast<size_t>(i)]))) {
            std::printf("  mismatch i=%lld (m=%lld n=%lld) got=%g want=%g diff=%g tol=%g\n",
                        static_cast<long long>(i),
                        static_cast<long long>(i / N),
                        static_cast<long long>(i % N),
                        static_cast<double>(got),
                        static_cast<double>(want[static_cast<size_t>(i)]),
                        static_cast<double>(diff),
                        static_cast<double>(atol + rtol * std::abs(want[static_cast<size_t>(i)])));
        }
        NNOPS_EXPECT_TRUE(diff <= atol + rtol * std::abs(want[static_cast<size_t>(i)]));
    }
    (void)worst;
}

/// Batched A [2, M, K] × B [2, N, K]: the per-batch B offset is N·ceil(K/32)
/// blocks, exercising the block-unit batch stride.
NNOPS_TEST(matmul_q4_1_batched) {
    // N is wide enough for the SME route, so the larger M rides its own
    // per-batch block offset; the smaller M stays on the fused decode.
    constexpr int64_t B = 2, N = 40, K = 70;
    constexpr int64_t M_values[] = {4, 40};
    constexpr bool transpose_b = true;

    std::vector<uint8_t> bbytes;
    Q4Matrix b0 = build_q4_matrix(N, K, 0x77u);
    Q4Matrix b1 = build_q4_matrix(N, K, 0x99u);
    append_q4_matrix(bbytes, b0);
    append_q4_matrix(bbytes, b1);

    for (const int64_t M : M_values) {
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
        auto output = nnops::test::make_channel_first(descs[0], out.data());

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
}

} // namespace

NNOPS_TEST(matmul_q4_1_f32)
{
    // N > M (split on N), N off the RHS panel, K off-block.
    run_case({5, 37, 70, /*transpose_b=*/false, /*f16=*/false, /*transpose_a=*/false});
    // M >= N (split on M), Linear-style [N][K] weight.
    run_case({37, 5, 70, /*transpose_b=*/true, /*f16=*/false, /*transpose_a=*/false});
    // Block-aligned N and K, wider than one panel.
    run_case({24, 71, 160, /*transpose_b=*/true, /*f16=*/false, /*transpose_a=*/false});
    // Small square with K below one block.
    run_case({9, 9, 33, /*transpose_b=*/false, /*f16=*/false, /*transpose_a=*/false});
    // Transposed A: physically [K][M], so the pack-A route runs as well.
    run_case({17, 23, 70, /*transpose_b=*/false, /*f16=*/false, /*transpose_a=*/true});
    run_case({17, 23, 70, /*transpose_b=*/true, /*f16=*/false, /*transpose_a=*/true});
    // alpha rides the Q4_1 RHS pack (every decoded weight scaled), so the
    // packed route needs no post-pass.
    run_case({5, 37, 70, /*transpose_b=*/false, /*f16=*/false, /*transpose_a=*/false, 0.0f, 2.5f});
    run_case({37, 5, 70, /*transpose_b=*/true, /*f16=*/false, /*transpose_a=*/false, 0.0f, -1.5f});
    run_case({24, 71, 160, /*transpose_b=*/true, /*f16=*/false, /*transpose_a=*/false, 0.5f, 0.5f});
    // A bias enters the accumulator as beta·C_old + bias before the first
    // k-block, so the overwriting (beta = 0) case must still keep it; the bias
    // is added unscaled by alpha.
    run_case({5, 37, 70, /*transpose_b=*/false, /*f16=*/false, /*transpose_a=*/false, 0.0f, 1.0f,
              EpilogueActivateType::None, /*bias=*/true});
    run_case({37, 5, 70, /*transpose_b=*/true, /*f16=*/false, /*transpose_a=*/false, 0.5f, 2.0f,
              EpilogueActivateType::None, /*bias=*/true});
}

/// Q4_0 shares the block layout and every route; only the decode differs
/// (d·(q − 8), no min). Cover the packed route, the fused small-M route and the
/// SME tile (M > 4, N >= 16), with both B layouts and the N/K tails.
NNOPS_TEST(matmul_q4_0)
{
    auto q0 = [](int64_t M, int64_t N, int64_t K, bool tb, bool f16 = false) {
        Case c{M, N, K, tb, f16, false};
        c.fmt = DataType::q4_0;
        return c;
    };

    run_case(q0(5, 37, 70, /*tb=*/false));   // copy, N tail
    run_case(q0(37, 5, 70, /*tb=*/true));    // trans, non-panel N
    run_case(q0(2, 64, 33, /*tb=*/false));   // K tail, fused route
    run_case(q0(17, 23, 70, /*tb=*/false));  // packed route
    run_case(q0(20, 130, 160, /*tb=*/true)); // SME tile, N/K tails
    run_case(q0(8, 130, 70, /*tb=*/true));   // fused limit, panel tails
    run_case(q0(5, 37, 70, /*tb=*/false, /*f16=*/true));
    Case scaled = q0(24, 71, 160, /*tb=*/true);
    scaled.alpha = 0.5f;
    scaled.beta = 0.5f;
    scaled.ep = EpilogueActivateType::Relu;
    run_case(scaled);
}

/// M split with a Q4 B: N is narrow, so the plan splits M and the full-N B
/// pack is hoisted — both m-blocks share one decode. On an SME build this runs
/// the ZA path; the NEON twin below pins ZA off.
NNOPS_TEST(matmul_q4_hoisted_b)
{
    run_case({1000, 64, 128, /*transpose_b=*/false, false, false});
    Case c{1000, 64, 128, /*transpose_b=*/false, false, false};
    c.fmt = DataType::q4_0;
    run_case(c);
}

/// Decode-sized M, at or below the fused route's limits.
NNOPS_TEST(matmul_q4_1_small_m)
{
    run_case({1, 37, 70, /*transpose_b=*/false, false, false});
    run_case({2, 37, 70, /*transpose_b=*/false, false, false});
    run_case({2, 64, 33, /*transpose_b=*/false, false, false}); // K tail
    run_case({1, 71, 160, /*transpose_b=*/true, false, false});
    run_case({4, 71, 160, /*transpose_b=*/true, false, false});
    run_case({3, 41, 33, /*transpose_b=*/true, false, false});  // partial m rows
    run_case({8, 130, 70, /*transpose_b=*/true, false, false}); // panel tails
    run_case({4, 37, 70, /*transpose_b=*/true, false, false, 1.0f});
    run_case({2, 37, 70, /*transpose_b=*/false, false, false, 0.5f});
    run_case(
        {2, 37, 70, /*transpose_b=*/false, false, false, 0.0f, 1.0f, EpilogueActivateType::Relu6});
    run_case(
        {7, 37, 70, /*transpose_b=*/true, false, false, 0.0f, 2.0f, EpilogueActivateType::Relu});
    run_case({2, 37, 70, /*transpose_b=*/false, false, false, 0.0f, 1.0f,
              EpilogueActivateType::None, /*bias=*/true});
    run_case({8, 71, 160, /*transpose_b=*/true, false, false, 0.5f, 1.5f,
              EpilogueActivateType::Relu6, /*bias=*/true});
}

/// The fused small-M route decodes Q4_1 blocks straight into the reduction (no
/// packed-B panel): [K][N] B runs the outer-product kernel (MR ≤ 2), [N][K] B
/// the dot-product one (MR ≤ 8 per call). Cases cover partial-M panels, K
/// tails, N tiles, beta and the clamped epilogue.
NNOPS_TEST(matmul_q4_1_fused_small_m) {
    run_case({1, 37, 70, /*transpose_b=*/false, false, false});
    run_case({2, 37, 70, /*transpose_b=*/false, false, false});
    run_case({2, 64, 33, /*transpose_b=*/false, false, false});   // K tail
    run_case({5, 37, 70, /*transpose_b=*/false, false, false});   // copy-limit band
    run_case({8, 37, 70, /*transpose_b=*/false, false, false});   // at the copy limit
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

/// The same route with f16 activations — the w4a16 decode case. Each M limit is
/// visited exactly: at it the fused kernel runs, one row past it the packed
/// plan takes the call, and both must land on the same decoded product.
NNOPS_TEST(matmul_q4_1_fused_small_m_f16) {
    run_case({1, 71, 160, /*transpose_b=*/true, true, false});
    run_case({3, 41, 33, /*transpose_b=*/true, true, false});    // MR=4, K tail
    run_case({16, 71, 95, /*transpose_b=*/true, true, false});   // two 8-row panels
    run_case({32, 71, 95, /*transpose_b=*/true, true, false});   // at the trans limit
    run_case({33, 71, 95, /*transpose_b=*/true, true, false});   // past it
    run_case({1, 37, 70, /*transpose_b=*/false, true, false});
    run_case({5, 37, 70, /*transpose_b=*/false, true, false});   // copy-limit band
    run_case({8, 37, 70, /*transpose_b=*/false, true, false});   // at the copy limit
    run_case({9, 37, 70, /*transpose_b=*/false, true, false});   // past it
    run_case({16, 37, 33, /*transpose_b=*/true, true, false, 0.5f, 2.0f});
    run_case({2, 37, 70, /*transpose_b=*/false, true, false, 0.0f, -0.5f});
    run_case({2, 37, 70, /*transpose_b=*/false, true, false, 0.0f, 1.0f,
              EpilogueActivateType::Relu6});
    run_case({8, 71, 160, /*transpose_b=*/true, true, false, 0.0f, 2.0f,
              EpilogueActivateType::None, /*bias=*/true});
    run_case({16, 37, 33, /*transpose_b=*/true, true, false, 0.5f, 1.5f,
              EpilogueActivateType::Relu, /*bias=*/true});
}

NNOPS_TEST(matmul_q4_1_f16) {
    run_case({5, 37, 70, /*transpose_b=*/false, /*f16=*/true, /*transpose_a=*/false});
    run_case({37, 5, 70, /*transpose_b=*/true,  /*f16=*/true, /*transpose_a=*/false});
    run_case({24, 71, 160, /*transpose_b=*/true, /*f16=*/true, /*transpose_a=*/false});
    run_case({17, 23, 70, /*transpose_b=*/true, /*f16=*/true, /*transpose_a=*/true});
    // The f16 route folds alpha into each block's d/m in f32, then rounds once
    // on store — the same arch-aware tolerance covers it.
    run_case({5, 37, 70, /*transpose_b=*/false, /*f16=*/true, /*transpose_a=*/false,
              0.0f, 2.0f});
    run_case({37, 5, 70, /*transpose_b=*/true,  /*f16=*/true, /*transpose_a=*/false,
              0.0f, 0.5f});
    run_case({5, 37, 70, /*transpose_b=*/false, /*f16=*/true, /*transpose_a=*/false,
              0.0f, 1.0f, EpilogueActivateType::None, /*bias=*/true});
    run_case({37, 5, 70, /*transpose_b=*/true,  /*f16=*/true, /*transpose_a=*/false,
              0.5f, 1.0f, EpilogueActivateType::None, /*bias=*/true});
}

#if defined(NNOPS_ARCH_AARCH64)
/// The bf16 Q4_1 pack must be byte-identical to decoding to a dense bf16 matrix
/// and running the ordinary bf16 RHS pack: the GEMM only knows the panel layout,
/// so a difference here is a layout bug no tolerance should hide.
NNOPS_TEST(pack_q4_1_bf16_matches_dense_pack) {
    constexpr int64_t N = 37;   // panels 16 + 16 + 5
    constexpr int kc = 70;      // kc off a multiple of 4
    const int ldd = static_cast<int>(backend::cpu::aarch64::bf16_b_panel_stride(kc));
    const int panels = static_cast<int>((N + 15) / 16);

    // Both physical B layouts, and a k window that starts mid-block (koff = 6)
    // as well as at the row origin.
    for (bool trans : {false, true}) {
        for (int k_start : {0, 70}) {
            const int64_t k_total = k_start + kc;
            const Q4Matrix b =
                trans ? build_q4_matrix(N, k_total, 0x77u) : build_q4_matrix(k_total, N, 0x77u);
            const int64_t rows = trans ? N : k_total;
            const int64_t cols = trans ? k_total : N;

            std::vector<bfloat16> dense(static_cast<size_t>(rows * cols));
            for (int64_t r = 0; r < rows; ++r) {
                for (int64_t c = 0; c < cols; ++c) {
                    dense[static_cast<size_t>(r * cols + c)] = backend::cpu::float_to_bfloat16(
                        b.decoded[static_cast<size_t>(r * cols + c)]);
                }
            }

            // The dense pack has no k_start: shift its source to the window.
            const bfloat16* dense_base = trans ?
                                             dense.data() + k_start :
                                             dense.data() + static_cast<int64_t>(k_start) * cols;

            std::vector<bfloat16> want(static_cast<size_t>(panels * ldd), bfloat16{});
            std::vector<bfloat16> got(static_cast<size_t>(panels * ldd), bfloat16{});
            backend::cpu::tile_pack_rhs(trans, static_cast<int>(N), kc, want.data(), ldd,
                                        dense_base, static_cast<int>(cols), 1.0f);
            backend::cpu::tile_pack_rhs_q4(trans, 0, static_cast<int>(N), k_start, kc, got.data(),
                                           ldd, b.bytes.data(), b.blocks_per_row, 1.0f,
                                           DataType::q4_1);

            for (size_t i = 0; i < want.size(); ++i) {
                if (want[i].bits != got[i].bits) {
                    std::printf("  pack mismatch trans=%d k_start=%d i=%zu "
                                "want=%04x got=%04x\n",
                                static_cast<int>(trans), k_start, i,
                                static_cast<unsigned>(want[i].bits),
                                static_cast<unsigned>(got[i].bits));
                }
                NNOPS_EXPECT_EQ(want[i].bits, got[i].bits);
            }
        }
    }
}

/// bf16 activations × a Q4_1 weight. The bf16 B pack writes the BFMMLA operand
/// layout rather than the [k][nr] panels, so both physical B layouts, the
/// 16-column panel tail, the 4-value k-group tail and the k-block tail are all
/// exercised against the f32 decode.
void run_bf16(int64_t M, int64_t N, int64_t K, bool transpose_b,
              bool transpose_a = false, float beta = 0.0f, float alpha = 1.0f,
              EpilogueActivateType ep = EpilogueActivateType::None,
              bool bias = false) {
    Case c{};
    c.M = M;
    c.N = N;
    c.K = K;
    c.transpose_b = transpose_b;
    c.transpose_a = transpose_a;
    c.beta = beta;
    c.alpha = alpha;
    c.ep = ep;
    c.bias = bias;
    c.bf16 = true;
    run_case(c);
}

NNOPS_TEST(matmul_q4_1_bf16) {
    // Both B layouts; N off the 16-column panel; K off both 32 and 4.
    run_bf16(5, 37, 70, /*transpose_b=*/false);
    run_bf16(37, 5, 70, /*transpose_b=*/true);
    run_bf16(24, 71, 160, /*transpose_b=*/true);
    run_bf16(17, 23, 70, /*transpose_b=*/true, /*transpose_a=*/true);
    run_bf16(3, 33, 95, /*transpose_b=*/false);   // short final k-group
    run_bf16(1, 16, 32, /*transpose_b=*/true);    // a full panel, no padding
    run_bf16(9, 31, 129, /*transpose_b=*/true);   // past one k-block

    // The post-accumulation semantics the f32/f16 cases cover.
    run_bf16(8, 71, 160, /*transpose_b=*/true, false, /*beta=*/0.0f,
             /*alpha=*/0.5f, EpilogueActivateType::Relu6, /*bias=*/true);
    run_bf16(16, 37, 33, /*transpose_b=*/true, false, /*beta=*/1.0f);
}

NNOPS_TEST(matmul_q4_1_fused_small_m_bf16) {
    // The f32/f16 boundary sweep, now that the fused small-M kernels take bf16.
    // At each per-layout M limit the fused kernel runs; one row past it the
    // packed plan takes the call, and both must land on the same decoded product.
    run_bf16(1, 71, 160, /*transpose_b=*/true);
    run_bf16(3, 41, 33, /*transpose_b=*/true);
    run_bf16(16, 71, 95, /*transpose_b=*/true);   // at the trans limit
    run_bf16(17, 71, 95, /*transpose_b=*/true);   // past it
    run_bf16(1, 37, 70, /*transpose_b=*/false);
    run_bf16(5, 37, 70, /*transpose_b=*/false);   // copy-limit band
    run_bf16(8, 37, 70, /*transpose_b=*/false);   // at the copy limit
    run_bf16(9, 37, 70, /*transpose_b=*/false);   // past it
    run_bf16(16, 37, 33, /*transpose_b=*/true, false, 0.0f, 0.5f);
    run_bf16(2, 37, 70, /*transpose_b=*/false, false, 0.0f, -0.5f);
    run_bf16(2, 37, 70, /*transpose_b=*/false, false, 0.0f, 1.0f,
             EpilogueActivateType::Relu6);
    run_bf16(8, 71, 160, /*transpose_b=*/true, false, 0.0f, 2.0f,
             EpilogueActivateType::None, /*bias=*/true);
    run_bf16(16, 37, 33, /*transpose_b=*/true, false, 0.5f, 1.5f,
             EpilogueActivateType::Relu, /*bias=*/true);
}
#endif

#if defined(NNOPS_ARCH_SME)
/// The ZA route over a Q4_1 B: M past the fused small-M limits (trans 16, copy
/// 8), N wide enough for the n-block grid, and — the point — K past the plan's
/// kc, so the second k-block decodes from a nonzero origin. Both physical B
/// layouts, f32 and f16.
NNOPS_TEST(matmul_q4_1_sme_route) {
    const SmePin sme_on(true);
    // f32: kc 1024, so K = 1050 takes a full block and a 26-value tail.
    run_case({40, 40, 1050, /*transpose_b=*/true, false, false});
    run_case({40, 72, 1100, /*transpose_b=*/false, false, false});
    // Shapes just past the fused limits; the copy case also packs a transposed A.
    run_case({17, 40, 70, /*transpose_b=*/true, false, false});
    run_case({17, 40, 70, /*transpose_b=*/false, false, /*transpose_a=*/true});
    // The planned route's post-accumulation semantics through the ZA tiles.
    run_case({40, 40, 160, true, false, false, 0.5f, 1.0f, EpilogueActivateType::Relu6,
              /*bias=*/true});
    // alpha folds into the decode, which is what lets this route keep a scaled
    // call rather than falling back to the NEON panels.
    run_case({40, 40, 160, true, false, false, 0.0f, -1.5f});
    run_case({40, 40, 160, false, true, false, 0.0f, 2.5f});
    // f16: kc 2048, so K = 2050 takes a full block and a 2-value tail.
    run_case({40, 40, 2050, /*transpose_b=*/true, true, false});
    run_case({40, 72, 2100, /*transpose_b=*/false, true, false});
    run_case({33, 40, 95, true, true, false, 0.0f, 1.0f, EpilogueActivateType::None,
              /*bias=*/true});
}

/// The ZA route splits N across the pool and each block decodes its own Q4_1
/// slice into its own scratch slot: the threaded product must equal the
/// single-threaded one. Both physical B layouts, a column tail included.
NNOPS_TEST(matmul_q4_1_sme_threaded) {
    const SmePin sme_on(true);
    for (bool tb : {false, true}) {
        constexpr int64_t M = 40, K = 1100, N = 300;
        std::vector<float> a = gen_floats(M * K, 0x31u);
        Q4Matrix b = tb ? build_q4_matrix(N, K, 0x51u) : build_q4_matrix(K, N, 0x51u);
        const int64_t ashape[] = {M, K};
        const int64_t bshape[] = {tb ? N : K, tb ? K : N};
        TensorView av(ashape, DataType::f32, a.data());
        TensorView bv(bshape, DataType::q4_1, b.bytes.data());

        MatMulAttributes attrs{};
        attrs.transpose_b = tb;
        auto op = MatMul::create(attrs, Backend::CPU);
        const TensorDesc arr[] = {av.desc(), bv.desc()};
        auto descs = op->getOutputTensorDesc(arr);
        const TensorView ins[] = {av, bv};

        std::vector<float> serial(static_cast<size_t>(M * N));
        std::vector<float> threaded(static_cast<size_t>(M * N));
        auto out_serial = nnops::test::make_channel_first(descs[0], serial.data());
        auto out_threaded = nnops::test::make_channel_first(descs[0], threaded.data());
        op->compute(out_serial, ins);

        SimplePool pool(4);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        op->compute(out_threaded, ins, ctx, nullptr);

        for (int64_t i = 0; i < M * N; ++i) {
            const float s = serial[static_cast<size_t>(i)];
            const float t = threaded[static_cast<size_t>(i)];
            NNOPS_EXPECT_TRUE(std::abs(s - t) <= 1e-6f + 1e-5f * std::abs(s));
        }
    }
}

/// The same calls with the ZA route pinned off: the NEON packed plan owns them,
/// so both engines answer to one reference at the same shapes. f16 stays short
/// here — its NEON kernel accumulates in f16 through every 128-wide k-block, so
/// a 2k-deep K is outside the shared tolerance; the ZA tile holds f32 across its
/// (much longer) k-block and covers that depth above.
NNOPS_TEST(matmul_q4_1_neon_shapes_match_sme) {
    const SmePin sme_off(false);
    run_case({40, 40, 1050, true, false, false});
    run_case({40, 72, 1100, false, false, false});
    run_case({40, 40, 95, true, true, false});
    run_case({40, 72, 160, false, true, false});
}

namespace {

/// The Q4_1 matrix as a dense panel-typed buffer. Exact for the generator's
/// d/m choices (every product and sum is representable), so the comparison
/// below can be bitwise.
template <class T>
std::vector<T> dense_from_q4(const Q4Matrix& b)
{
    std::vector<T> d(b.decoded.size());
    for (size_t i = 0; i < d.size(); ++i) {
        if constexpr (std::is_same_v<T, float>) {
            d[i] = b.decoded[i];
        } else {
            d[i] = float_to_half(b.decoded[i]);
        }
    }
    return d;
}

template <class T> uint32_t pack_bits(T v)
{
    if constexpr (std::is_same_v<T, float>) {
        uint32_t u;
        std::memcpy(&u, &v, sizeof(u));
        return u;
    } else {
        return static_cast<uint32_t>(backend::cpu::half_to_bits(v));
    }
}

/// The ZA Q4 pack against decode-then-dense-pack, bit for bit.
template <class T> void check_sme_pack(DataType dt, DataType fmt = DataType::q4_1)
{
    constexpr int64_t N = 37; // a full 32-column panel plus a 5-column tail
    constexpr int kc = 70;    // off a 32-value block, and an odd f16 pair count
    const int ldd = static_cast<int>(backend::cpu::sme_b_panel_stride(dt, kc));
    constexpr int panels = 2;

    for (bool trans : {false, true}) {
        // 70 is off a block boundary: the transposed pack takes its scalar
        // fallback there, so both paths are held to the same answer.
        for (int k_start : {0, 32, 70}) {
            const int64_t k_total = k_start + kc;
            const Q4Matrix b = trans ? build_q4_matrix(N, k_total, 0x77u, fmt) :
                                       build_q4_matrix(k_total, N, 0x77u, fmt);
            const std::vector<T> dense = dense_from_q4<T>(b);
            const T* const dense_base =
                trans ? dense.data() + k_start : dense.data() + static_cast<int64_t>(k_start) * N;
            const int lds = static_cast<int>(trans ? k_total : N);

            std::vector<T> want(static_cast<size_t>(panels) * ldd, T{});
            std::vector<T> got(static_cast<size_t>(panels) * ldd, T{});
            backend::cpu::tile_pack_rhs_sme(trans, static_cast<int>(N), kc, want.data(), dense_base,
                                            lds);
            backend::cpu::tile_pack_rhs_q4_sme(trans, 0, static_cast<int>(N), k_start, kc,
                                               got.data(), b.bytes.data(), b.blocks_per_row, 1.0f,
                                               fmt);

            for (size_t i = 0; i < want.size(); ++i) {
                const uint32_t w = pack_bits(want[i]);
                const uint32_t g = pack_bits(got[i]);
                if (w != g) {
                    std::printf("  q4 sme pack mismatch dt=%d trans=%d k_start=%d i=%zu "
                                "want=%08x got=%08x\n",
                                static_cast<int>(dt), static_cast<int>(trans), k_start, i,
                                static_cast<unsigned>(w), static_cast<unsigned>(g));
                }
                NNOPS_EXPECT_EQ(w, g);
            }
        }
    }
}

} // namespace

/// The ZA panel pack must be byte-identical to decoding into a dense matrix and
/// running the ordinary SME RHS pack: the kernels only know the layout, so a
/// difference is a layout bug no tolerance should hide.
NNOPS_TEST(pack_q4_sme_matches_dense_pack)
{
    check_sme_pack<float>(DataType::f32, DataType::q4_1);
    check_sme_pack<half>(DataType::f16, DataType::q4_1);
    check_sme_pack<float>(DataType::f32, DataType::q4_0);
    check_sme_pack<half>(DataType::f16, DataType::q4_0);
}

/// The ZA route over a Q4_0 B — same shapes as the Q4_1 route test, so the
/// format's only difference is the decode.
NNOPS_TEST(matmul_q4_0_sme_route)
{
    const SmePin sme_on(true);
    auto q0 = [](int64_t M, int64_t N, int64_t K, bool tb, bool f16 = false) {
        Case c{M, N, K, tb, f16, false};
        c.fmt = DataType::q4_0;
        return c;
    };
    run_case(q0(40, 40, 1050, /*tb=*/true));
    run_case(q0(40, 72, 1100, /*tb=*/false));
    run_case(q0(17, 40, 70, /*tb=*/false));
    Case scaled = q0(40, 40, 160, /*tb=*/true);
    scaled.alpha = -1.5f;
    run_case(scaled);
    run_case(q0(40, 72, 2100, /*tb=*/false, /*f16=*/true));
}

/// The same Q4_0 calls with the ZA route pinned off: the NEON packed plan owns
/// them, so both engines answer to one reference at the same shapes.
NNOPS_TEST(matmul_q4_0_neon_route)
{
    const SmePin sme_off(false);
    auto q0 = [](int64_t M, int64_t N, int64_t K, bool tb, bool f16 = false) {
        Case c{M, N, K, tb, f16, false};
        c.fmt = DataType::q4_0;
        return c;
    };
    run_case(q0(40, 40, 1050, /*tb=*/true));
    run_case(q0(40, 72, 1100, /*tb=*/false));
    run_case(q0(17, 40, 70, /*tb=*/false));
    Case scaled = q0(40, 40, 160, /*tb=*/true);
    scaled.alpha = -1.5f;
    run_case(scaled);
    run_case(q0(40, 72, 160, /*tb=*/false, /*f16=*/true));
}

/// The M-split hoist on the NEON plan: ZA pinned off, N narrow so the plan
/// splits M and one full-N B pack serves both m-blocks.
NNOPS_TEST(matmul_q4_hoisted_b_neon)
{
    const SmePin sme_off(false);
    run_case({1000, 64, 128, /*transpose_b=*/false, false, false});
    Case c{1000, 64, 128, /*transpose_b=*/false, false, false};
    c.fmt = DataType::q4_0;
    run_case(c);
}
#endif // NNOPS_ARCH_SME

// ============================================================
// Q8_0 — 32 signed bytes per block, d·q
// ============================================================
//
// Q8_0 rides every route the Q4 formats do (same 32-value block, same packers,
// same fused kernels); only the decode differs and the quants are signed.

NNOPS_TEST(matmul_q8_0)
{
    auto q8 = [](int64_t M, int64_t N, int64_t K, bool tb, bool f16 = false) {
        Case c{M, N, K, tb, f16, false};
        c.fmt = DataType::q8_0;
        return c;
    };

    run_case(q8(5, 37, 70, /*tb=*/false));   // copy, N tail
    run_case(q8(37, 5, 70, /*tb=*/true));    // trans, non-panel N
    run_case(q8(2, 64, 33, /*tb=*/false));   // K tail, fused route
    run_case(q8(17, 23, 70, /*tb=*/false));  // packed route
    run_case(q8(20, 130, 160, /*tb=*/true)); // N/K tails
    run_case(q8(8, 130, 70, /*tb=*/true));   // panel tails
    run_case(q8(5, 37, 70, /*tb=*/false, /*f16=*/true));
    Case scaled = q8(24, 71, 160, /*tb=*/true);
    scaled.alpha = 0.5f;
    scaled.beta = 0.5f;
    scaled.ep = EpilogueActivateType::Relu;
    run_case(scaled);
}

/// The fused small-M route (M above it falls through to the packed plan).
NNOPS_TEST(matmul_q8_0_small_m)
{
    run_case({1, 37, 70, /*transpose_b=*/false, false, false, 0.0f, 1.0f,
              EpilogueActivateType::None, false, false, DataType::q8_0});
    run_case({8, 37, 70, /*transpose_b=*/false, false, false, 0.0f, 1.0f,
              EpilogueActivateType::None, false, false, DataType::q8_0});
    run_case({16, 71, 95, /*transpose_b=*/true, false, false, 0.0f, 1.0f,
              EpilogueActivateType::None, false, false, DataType::q8_0});
    run_case({17, 71, 95, /*transpose_b=*/true, false, false, 0.0f, 1.0f,
              EpilogueActivateType::None, false, false, DataType::q8_0}); // past it
    run_case({2, 37, 70, /*transpose_b=*/false, false, false, 0.5f, 1.0f,
              EpilogueActivateType::Relu6, true, false, DataType::q8_0});
}

/// Batched A [2, M, K] × B [2, N, K]: the per-batch offset is counted in Q8_0
/// blocks (34 bytes each), not elements.
NNOPS_TEST(matmul_q8_0_batched)
{
    constexpr int64_t B = 2, N = 40, K = 70;
    constexpr int64_t M_values[] = {4, 40};
    constexpr bool transpose_b = true;

    std::vector<uint8_t> bbytes;
    Q4Matrix b0 = build_q4_matrix(N, K, 0x77u, DataType::q8_0);
    Q4Matrix b1 = build_q4_matrix(N, K, 0x99u, DataType::q8_0);
    append_q4_matrix(bbytes, b0);
    append_q4_matrix(bbytes, b1);

    for (const int64_t M : M_values) {
        const std::vector<float> a0 = gen_floats(M * K, 0x100u);
        const std::vector<float> a1 = gen_floats(M * K, 0x200u);
        std::vector<float> adata = a0;
        adata.insert(adata.end(), a1.begin(), a1.end());

        const int64_t ashape[] = {B, M, K};
        const int64_t bshape[] = {B, N, K};
        TensorView a(ashape, DataType::f32, adata.data());
        TensorView bv(bshape, DataType::q8_0, bbytes.data());

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
        auto output = nnops::test::make_channel_first(descs[0], out.data());

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
}
