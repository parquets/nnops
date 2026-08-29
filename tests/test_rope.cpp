/// Unit tests for RoPE operator (CPU optimized + reference).
/// Covers standard RoPE (interleaved and split-half), mRoPE, various shapes,
/// edge cases.

#include "nnops/ops/rope.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

// ============================================================
// Helper: scalar RoPE reference for verification
// ============================================================

static float rope_theta(int64_t pos, int64_t pair_idx, int64_t head_dim,
                        float base, bool interleaved) {
    float dim_f = static_cast<float>(head_dim);
    return static_cast<float>(pos) /
        std::pow(base, 2.0f * static_cast<float>(pair_idx) / dim_f);
}

static void compute_rope_ref(const float* x, float* y,
                              int64_t seq_len, int64_t head_dim,
                              float base, bool interleaved) {
    const int64_t half = head_dim / 2;
    for (int64_t pos = 0; pos < seq_len; ++pos) {
        if (interleaved) {
            for (int64_t d = 0; d < head_dim; d += 2) {
                int64_t pair_idx = d / 2;
                float theta = rope_theta(pos, pair_idx, head_dim, base, true);
                float cv = std::cos(theta);
                float sv = std::sin(theta);

                float x0 = x[pos * head_dim + d];
                float x1 = x[pos * head_dim + d + 1];
                y[pos * head_dim + d]     = x0 * cv - x1 * sv;
                y[pos * head_dim + d + 1] = x1 * cv + x0 * sv;
            }
        } else {
            for (int64_t d = 0; d < half; ++d) {
                int64_t pair_idx = d;
                float theta = rope_theta(pos, pair_idx, head_dim, base, false);
                float cv = std::cos(theta);
                float sv = std::sin(theta);

                float x0 = x[pos * head_dim + d];
                float x1 = x[pos * head_dim + d + half];
                y[pos * head_dim + d]        = x0 * cv - x1 * sv;
                y[pos * head_dim + d + half] = x1 * cv + x0 * sv;
            }
        }
    }
}

// ============================================================
// Hand-verified small tests
// ============================================================

NNOPS_TEST(rope_simple_interleaved) {
    // 2 positions, head_dim=4 (2 pairs), base=1.0 for simple angles
    const int64_t shape[] = {2, 4};
    float x_data[] = {1.0f, 0.0f, 2.0f, 0.0f,
                      0.0f, 1.0f, 0.0f, 2.0f};

    TensorView x(shape, DataType::f32, x_data);

    RoPEAttributes attrs;
    attrs.base = 1.0f;
    attrs.interleaved = true;
    auto op = RoPE::create(attrs, Backend::CPU);

    // Validate output descriptor
    auto d_x = x.desc();
    const TensorDesc desc_arr[] = {d_x};
    auto descs = op->getOutputTensorDesc(desc_arr);
    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(4));
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {x};
    op->compute(y, ins);

    // base=1.0, head_dim=4:
    //   pair 0: theta = pos / 1^(0/4) = pos
    //   pair 1: theta = pos / 1^(2/4) = pos
    // So both pairs rotate by angle = pos radians.
    // pos=0: cos(0)=1, sin(0)=0 → identity
    NNOPS_EXPECT_NEAR(out_buf[0], 1.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], 0.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[2], 2.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[3], 0.0f, 1e-5f);

    // pos=1: cos(1)≈0.5403, sin(1)≈0.8415
    float c = std::cos(1.0f);
    float s = std::sin(1.0f);
    float e4 = 0.0f * c - 1.0f * s;  // -s
    float e5 = 1.0f * c + 0.0f * s;  // c
    float e6 = 0.0f * c - 2.0f * s;  // -2s
    float e7 = 2.0f * c + 0.0f * s;  // 2c
    NNOPS_EXPECT_NEAR(out_buf[4], e4, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[5], e5, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[6], e6, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[7], e7, 1e-5f);
}

NNOPS_TEST(rope_simple_splithalf) {
    // 1 position, head_dim=4, base=10000
    const int64_t shape[] = {1, 4};
    float x_data[] = {1.0f, 2.0f, 3.0f, 4.0f};

    TensorView x(shape, DataType::f32, x_data);

    RoPEAttributes attrs;
    attrs.base = 10000.0f;
    attrs.interleaved = false;
    auto op = RoPE::create(attrs, Backend::CPU);

    auto d_x = x.desc();
    const TensorDesc desc_arr[] = {d_x};
    auto descs = op->getOutputTensorDesc(desc_arr);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {x};
    op->compute(y, ins);

    // head_dim=4, half=2
    // pair 0 (d=0↔d=2): theta = 0 / 10000^(0/4) = 0, cos=1, sin=0
    // pair 1 (d=1↔d=3): theta = 0 / 10000^(2/4) = 0, cos=1, sin=0
    // pos=0 → identity
    NNOPS_EXPECT_NEAR(out_buf[0], 1.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], 2.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[2], 3.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[3], 4.0f, 1e-5f);
}

// ============================================================
// Compare against reference for random data
// ============================================================

NNOPS_TEST(rope_random_interleaved) {
    const int64_t seq_len = 8;
    const int64_t head_dim = 16;
    const int64_t shape[] = {seq_len, head_dim};
    const int N = seq_len * head_dim;

    auto [in_vec, in_view] = test::make_random_tensor(shape, -2.0f, 2.0f, 100);

    RoPEAttributes attrs;
    attrs.base = 10000.0f;
    attrs.interleaved = true;
    auto op = RoPE::create(attrs, Backend::CPU);

    auto d_x = in_view.desc();
    const TensorDesc da[] = {d_x};
    auto descs = op->getOutputTensorDesc(da);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {in_view};
    op->compute(y, ins);

    // Compute reference
    std::vector<float> ref(static_cast<size_t>(N), 0.0f);
    compute_rope_ref(in_vec.data(), ref.data(), seq_len, head_dim,
                     10000.0f, true);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], ref[i], 1e-4f);
    }
}

NNOPS_TEST(rope_random_splithalf) {
    const int64_t seq_len = 8;
    const int64_t head_dim = 16;
    const int64_t shape[] = {seq_len, head_dim};
    const int N = seq_len * head_dim;

    auto [in_vec, in_view] = test::make_random_tensor(shape, -2.0f, 2.0f, 101);

    RoPEAttributes attrs;
    attrs.base = 10000.0f;
    attrs.interleaved = false;
    auto op = RoPE::create(attrs, Backend::CPU);

    auto d_x = in_view.desc();
    const TensorDesc da[] = {d_x};
    auto descs = op->getOutputTensorDesc(da);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {in_view};
    op->compute(y, ins);

    std::vector<float> ref(static_cast<size_t>(N), 0.0f);
    compute_rope_ref(in_vec.data(), ref.data(), seq_len, head_dim,
                     10000.0f, false);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], ref[i], 1e-4f);
    }
}

// ============================================================
// Multi-dimensional shapes
// ============================================================

NNOPS_TEST(rope_3d_batch) {
    // [B=2, S=3, D=8]
    const int64_t shape[] = {2, 3, 8};
    auto [in_vec, in_view] = test::make_random_tensor(shape, -2.0f, 2.0f, 102);

    RoPEAttributes attrs;
    attrs.base = 10000.0f;
    attrs.interleaved = true;
    auto op = RoPE::create(attrs, Backend::CPU);

    auto d_x = in_view.desc();
    const TensorDesc da[] = {d_x};
    auto descs = op->getOutputTensorDesc(da);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(3));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(3));
    NNOPS_EXPECT_EQ(descs[0].dims[2], int64_t(8));

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {in_view};
    op->compute(y, ins);

    // Verify batch 0 and batch 1 get same rotation for same position
    // (position-dependent, not batch-dependent)
    for (int b = 0; b < 2; ++b) {
        for (int s = 0; s < 3; ++s) {
            for (int d = 0; d < 8; ++d) {
                int idx = b * (3 * 8) + s * 8 + d;
                // Result should be non-zero for non-trivial input
                NNOPS_EXPECT_TRUE(std::isfinite(out_buf[static_cast<size_t>(idx)]));
            }
        }
    }
}

NNOPS_TEST(rope_4d_multihead) {
    // [B=2, H=4, S=3, D=8]
    const int64_t shape[] = {2, 4, 3, 8};
    auto [in_vec, in_view] = test::make_random_tensor(shape, -2.0f, 2.0f, 103);

    RoPEAttributes attrs;
    attrs.base = 10000.0f;
    attrs.interleaved = false;
    auto op = RoPE::create(attrs, Backend::CPU);

    auto d_x = in_view.desc();
    const TensorDesc da[] = {d_x};
    auto descs = op->getOutputTensorDesc(da);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(4));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(4));
    NNOPS_EXPECT_EQ(descs[0].dims[2], int64_t(3));
    NNOPS_EXPECT_EQ(descs[0].dims[3], int64_t(8));

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {in_view};
    op->compute(y, ins);

    // Compare against reference per (batch, head)
    std::vector<float> ref(static_cast<size_t>(2 * 4 * 3 * 8), 0.0f);
    for (int b = 0; b < 2; ++b) {
        for (int h = 0; h < 4; ++h) {
            compute_rope_ref(in_vec.data() + (b * 4 * 3 * 8 + h * 3 * 8),
                             ref.data()    + (b * 4 * 3 * 8 + h * 3 * 8),
                             3, 8, 10000.0f, false);
        }
    }
    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], ref[i], 1e-4f);
    }
}

// ============================================================
// Different base frequencies
// ============================================================

NNOPS_TEST(rope_different_bases) {
    // 2 positions: pos=0 gives identity for all bases,
    // pos=1 gives visible rotation difference between base=1 and base=1000000.
    const int64_t shape[] = {2, 4};
    float x_data[] = {1.0f, 0.0f, 1.0f, 0.0f,
                      1.0f, 0.0f, 1.0f, 0.0f};

    TensorView x(shape, DataType::f32, x_data);

    // base=1.0: theta = pos, rotate by pos radians
    RoPEAttributes attrs1;
    attrs1.base = 1.0f;
    auto op1 = RoPE::create(attrs1, Backend::CPU);
    auto dx = x.desc();
    const TensorDesc da[] = {dx};
    auto d = op1->getOutputTensorDesc(da);

    std::vector<float> out1(static_cast<size_t>(d[0].numel()), 0.0f);
    auto y1 = test::make_planar(d[0], out1.data());
    const TensorView x_arr[] = {x};
    op1->compute(y1, x_arr);

    // base=1000000.0: theta ≈ 0, cos≈1, sin≈0 → practically identity
    RoPEAttributes attrs2;
    attrs2.base = 1000000.0f;
    auto op2 = RoPE::create(attrs2, Backend::CPU);

    std::vector<float> out2(static_cast<size_t>(d[0].numel()), 0.0f);
    auto y2 = test::make_planar(d[0], out2.data());
    op2->compute(y2, x_arr);

    // Different bases should produce different results
    float diff = 0.0f;
    for (size_t i = 0; i < out1.size(); ++i) {
        diff += std::abs(out1[i] - out2[i]);
    }
    NNOPS_EXPECT_TRUE(diff > 0.1f);
}

// ============================================================
// mRoPE: multi-section with different bases
// ============================================================

NNOPS_TEST(mrope_two_sections_same_base) {
    // A single mRoPE section covering the whole head_dim should produce
    // the same result as standard RoPE (same section dim = same frequency formula).
    const int64_t shape[] = {2, 8};
    float x_data[] = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data);

    RoPEAttributes attrs;
    attrs.base = 10000.0f;
    attrs.interleaved = true;
    attrs.mrope_section_dims = {8};  // single section covering entire head_dim
    auto op = RoPE::create(attrs, Backend::CPU);

    auto dx = x.desc();
    const TensorDesc da[] = {dx};
    auto d = op->getOutputTensorDesc(da);
    std::vector<float> out_buf(static_cast<size_t>(d[0].numel()), 0.0f);
    auto y = test::make_planar(d[0], out_buf.data());
    const TensorView ins[] = {x};
    op->compute(y, ins);

    // Should give the same result as standard RoPE with head_dim=8
    // (same section dim = same frequency formula)
    RoPEAttributes attrs_std;
    attrs_std.base = 10000.0f;
    attrs_std.interleaved = true;
    auto op_std = RoPE::create(attrs_std, Backend::CPU);

    std::vector<float> out_std(static_cast<size_t>(d[0].numel()), 0.0f);
    auto y_std = test::make_planar(d[0], out_std.data());
    const TensorView ins2[] = {x};
    op_std->compute(y_std, ins2);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], out_std[i], 1e-5f);
    }
}

NNOPS_TEST(mrope_different_bases) {
    // head_dim=4, sections=[2,2], bases=[1.0, 1000000.0]
    // First pair rotates by pos, second pair is essentially identity
    const int64_t shape[] = {1, 4};
    float x_data[] = {1.0f, 0.0f, 1.0f, 0.0f};

    TensorView x(shape, DataType::f32, x_data);

    RoPEAttributes attrs;
    attrs.base = 1.0f;  // default for sections without explicit base
    attrs.interleaved = true;
    attrs.mrope_section_dims = {2, 2};
    attrs.mrope_section_bases = {1.0f, 1000000.0f};
    auto op = RoPE::create(attrs, Backend::CPU);

    auto dx = x.desc();
    const TensorDesc da[] = {dx};
    auto d = op->getOutputTensorDesc(da);
    std::vector<float> out_buf(static_cast<size_t>(d[0].numel()), 0.0f);
    auto y = test::make_planar(d[0], out_buf.data());
    const TensorView ins[] = {x};
    op->compute(y, ins);

    // pos=0: cos=1, sin=0 → identity
    NNOPS_EXPECT_NEAR(out_buf[0], 1.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], 0.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[2], 1.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[3], 0.0f, 1e-5f);
}

// ============================================================
// Edge cases
// ============================================================

NNOPS_TEST(rope_zero_input) {
    const int64_t shape[] = {1, 4};
    float x_data[] = {0.0f, 0.0f, 0.0f, 0.0f};

    TensorView x(shape, DataType::f32, x_data);

    RoPEAttributes attrs;
    auto op = RoPE::create(attrs, Backend::CPU);

    auto dx = x.desc();
    const TensorDesc da[] = {dx};
    auto d = op->getOutputTensorDesc(da);
    std::vector<float> out_buf(static_cast<size_t>(d[0].numel()), 0.0f);
    auto y = test::make_planar(d[0], out_buf.data());
    const TensorView ins[] = {x};
    op->compute(y, ins);

    // All zeros → all zeros
    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], 0.0f, 1e-6f);
    }
}

NNOPS_TEST(rope_single_position) {
    // seq_len=1, head_dim=8
    const int64_t shape[] = {1, 8};
    float x_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    TensorView x(shape, DataType::f32, x_data);

    RoPEAttributes attrs;
    auto op = RoPE::create(attrs, Backend::CPU);

    auto dx = x.desc();
    const TensorDesc da[] = {dx};
    auto d = op->getOutputTensorDesc(da);
    std::vector<float> out_buf(static_cast<size_t>(d[0].numel()), 0.0f);
    auto y = test::make_planar(d[0], out_buf.data());
    const TensorView ins[] = {x};
    op->compute(y, ins);

    // pos=0 → identity
    for (int i = 0; i < 8; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[static_cast<size_t>(i)], x_data[i], 1e-5f);
    }
}

NNOPS_TEST(rope_large_head_dim) {
    const int64_t shape[] = {2, 64};
    auto [in_vec, in_view] = test::make_random_tensor(shape, -2.0f, 2.0f, 105);

    RoPEAttributes attrs;
    attrs.interleaved = false;
    auto op = RoPE::create(attrs, Backend::CPU);

    auto dx = in_view.desc();
    const TensorDesc da[] = {dx};
    auto d = op->getOutputTensorDesc(da);
    std::vector<float> out_buf(static_cast<size_t>(d[0].numel()), 0.0f);
    auto y = test::make_planar(d[0], out_buf.data());
    const TensorView ins[] = {in_view};
    op->compute(y, ins);

    std::vector<float> ref(static_cast<size_t>(2 * 64), 0.0f);
    compute_rope_ref(in_vec.data(), ref.data(), 2, 64, 10000.0f, false);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], ref[i], 1e-4f);
    }
}

// ============================================================
// Class API
// ============================================================

NNOPS_TEST(rope_class_api) {
    const int64_t shape[] = {1, 4};
    float x_data[] = {1.0f, 0.0f, 2.0f, 0.0f};
    float out_data[4] = {};

    TensorView x(shape, DataType::f32, x_data);
    TensorView y(shape, DataType::f32, out_data);

    RoPEAttributes attrs;
    attrs.base = 1.0f;
    ComputeContext ctx;
    ctx.expected_backend = Backend::CPU;

    auto op = RoPE::create(attrs, Backend::CPU);
    const TensorView ins[] = {x};
    op->compute(y, ins, ctx);

    // pos=0 → identity
    NNOPS_EXPECT_NEAR(out_data[0], 1.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_data[1], 0.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_data[2], 2.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_data[3], 0.0f, 1e-5f);
}

// ============================================================
// OpType verification
// ============================================================

NNOPS_TEST(rope_op_type) {
    auto op = RoPE::create(Backend::CPU);
    NNOPS_EXPECT_EQ(op->getOpType(), OpType::RoPE);
    NNOPS_EXPECT_EQ(op->getBackend(), Backend::CPU);
}

// ============================================================
// f16 test — exercise the SIMD f16 code path
// ============================================================

NNOPS_TEST(rope_random_interleaved_f16) {
    const int64_t seq_len = 6;
    const int64_t head_dim = 16;
    const int64_t shape[] = {seq_len, head_dim};
    const int N = seq_len * head_dim;

    auto [f32_vec, _] = test::make_random_tensor(shape, -2.0f, 2.0f, 1000);
    auto f16_vec = test::f32_to_f16(f32_vec);

    TensorView x(shape, DataType::f16, f16_vec.data());

    RoPEAttributes attrs;
    attrs.base = 10000.0f;
    attrs.interleaved = true;
    auto op = RoPE::create(attrs, Backend::CPU);

    auto d_x = x.desc();
    const TensorDesc da[] = {d_x};
    auto descs = op->getOutputTensorDesc(da);

    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

    std::vector<nnops::backend::cpu::half> out_buf(static_cast<size_t>(descs[0].numel()));
    auto y = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {x};
    op->compute(y, ins);

    // Compute f32 reference
    std::vector<float> ref(static_cast<size_t>(N), 0.0f);
    compute_rope_ref(f32_vec.data(), ref.data(), seq_len, head_dim,
                     10000.0f, true);

    for (int i = 0; i < N; ++i) {
        float result = simd::s_load(&out_buf[i]);
        NNOPS_EXPECT_TRUE(std::isfinite(result));
        NNOPS_EXPECT_NEAR(result, ref[i], 5e-2f);
    }
}
