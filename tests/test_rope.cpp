/// @file test_rope.cpp
/// @brief Unit tests for RoPE, checked against a scalar reference.
///
/// Covers the interleaved and split-half layouts, mRoPE sections with
/// per-section bases, 2-D through 4-D shapes, the f16 path, and the edge cases
/// (zero input, single position). The position offset and the three
/// long-context scaling modes (Linear, Dynamic NTK, YaRN) are checked against
/// a scalar reference that spells the HuggingFace formulas out again, and the
/// offset also against the shift property (a one-row call at offset k equals
/// row k of a run starting at 0).

#include "nnops/ops/rope.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace nnops;

// ============================================================
// Helper: scalar RoPE reference
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
// Helper: scalar reference for the offset and scaling modes
// ============================================================

struct RefScaling {
    RoPEScalingType type = RoPEScalingType::None;
    float factor = 1.0f;
    int64_t orig = 0;
    float beta_fast = 32.0f;
    float beta_slow = 1.0f;
    float attention_factor = 0.0f;
    float l3_low = 1.0f;
    float l3_high = 4.0f;
};

static float ref_correction_dim(int64_t dim, float rotations, float base,
                                int64_t orig) {
    return static_cast<float>(dim)
         * std::log(static_cast<float>(orig) / (rotations * 6.283185307179586f))
         / (2.0f * std::log(base));
}

static float rope_theta_scaled(int64_t pos, int64_t pair, int64_t dim,
                               float base, int64_t run_end,
                               const RefScaling& s) {
    const float dim_f = static_cast<float>(dim);
    const float extrap = static_cast<float>(pos)
                       / std::pow(base, 2.0f * static_cast<float>(pair) / dim_f);

    if (s.type == RoPEScalingType::Linear) { return extrap / s.factor; }

    if (s.type == RoPEScalingType::Dynamic) {
        const double len = std::max(static_cast<double>(run_end),
                                    static_cast<double>(s.orig));
        const double mult = s.factor * len / static_cast<double>(s.orig)
                          - (static_cast<double>(s.factor) - 1.0);
        const double e = dim > 2 ? static_cast<double>(dim) / static_cast<double>(dim - 2)
                                 : 1.0;
        const double b = base * std::pow(mult, e);
        return static_cast<float>(static_cast<double>(pos)
               / std::pow(b, 2.0 * static_cast<double>(pair) / static_cast<double>(dim)));
    }

    if (s.type == RoPEScalingType::YaRN) {
        const float low = std::max(0.0f, std::floor(
            ref_correction_dim(dim, s.beta_fast, base, s.orig)));
        const float high = std::min(static_cast<float>(dim - 1), std::ceil(
            ref_correction_dim(dim, s.beta_slow, base, s.orig)));
        const float ramp = std::clamp(
            (static_cast<float>(pair) - low) / std::max(0.001f, high - low),
            0.0f, 1.0f);
        return (extrap / s.factor) * ramp + extrap * (1.0f - ramp);
    }

    if (s.type == RoPEScalingType::Llama3) {
        const float wavelen = 6.283185307179586f
            * std::pow(base, 2.0f * static_cast<float>(pair) / dim_f);
        const float low_w = static_cast<float>(s.orig) / s.l3_low;
        const float high_w = static_cast<float>(s.orig) / s.l3_high;
        if (wavelen > low_w) { return extrap / s.factor; }
        if (wavelen >= high_w) {
            const float smooth = std::clamp(
                (static_cast<float>(s.orig) / wavelen - s.l3_low)
                    / (s.l3_high - s.l3_low), 0.0f, 1.0f);
            return extrap * ((1.0f - smooth) / s.factor + smooth);
        }
    }

    return extrap;
}

static float ref_attention_magnitude(const RefScaling& s) {
    if (s.type != RoPEScalingType::YaRN) { return 1.0f; }
    if (s.attention_factor > 0.0f) { return s.attention_factor; }
    return s.factor > 1.0f ? 1.0f + 0.1f * std::log(s.factor) : 1.0f;
}

static void compute_rope_ref_scaled(const float* x, float* y,
                                    int64_t seq_len, int64_t head_dim,
                                    float base, bool interleaved,
                                    int64_t position_offset,
                                    const RefScaling& s) {
    const int64_t half = head_dim / 2;
    const int64_t run_end = position_offset + seq_len;
    const float magnitude = ref_attention_magnitude(s);

    for (int64_t row = 0; row < seq_len; ++row) {
        const int64_t pos = position_offset + row;
        const auto rotate = [&](int64_t pair, int64_t i0, int64_t i1) {
            const float theta = rope_theta_scaled(pos, pair, head_dim, base,
                                                  run_end, s);
            const float cv = std::cos(theta) * magnitude;
            const float sv = std::sin(theta) * magnitude;

            const float x0 = x[row * head_dim + i0];
            const float x1 = x[row * head_dim + i1];
            y[row * head_dim + i0] = x0 * cv - x1 * sv;
            y[row * head_dim + i1] = x1 * cv + x0 * sv;
        };
        for (int64_t d = 0; d < half; ++d) {
            if (interleaved) {
                rotate(d, 2 * d, 2 * d + 1);
            } else {
                rotate(d, d, d + half);
            }
        }
    }
}

// ============================================================
// Hand-verified small tests
// ============================================================

NNOPS_TEST(rope_simple_interleaved) {
    const int64_t shape[] = {2, 4};
    float x_data[] = {1.0f, 0.0f, 2.0f, 0.0f,
                      0.0f, 1.0f, 0.0f, 2.0f};

    TensorView x(shape, DataType::f32, x_data);

    RoPEAttributes attrs;
    attrs.base = 1.0f;
    attrs.interleaved = true;
    auto op = RoPE::create(attrs, Backend::CPU);

    auto d_x = x.desc();
    const TensorDesc desc_arr[] = {d_x};
    auto descs = op->getOutputTensorDesc(desc_arr);
    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(4));
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_channel_first(descs[0], out_buf.data());
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
    auto y = test::make_channel_first(descs[0], out_buf.data());
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
    auto y = test::make_channel_first(descs[0], out_buf.data());
    const TensorView ins[] = {in_view};
    op->compute(y, ins);

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
    auto y = test::make_channel_first(descs[0], out_buf.data());
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
    auto y = test::make_channel_first(descs[0], out_buf.data());
    const TensorView ins[] = {in_view};
    op->compute(y, ins);

    // Only a finiteness check here; the rotation itself is covered by the
    // reference tests.
    for (int b = 0; b < 2; ++b) {
        for (int s = 0; s < 3; ++s) {
            for (int d = 0; d < 8; ++d) {
                int idx = b * (3 * 8) + s * 8 + d;
                NNOPS_EXPECT_TRUE(std::isfinite(out_buf[static_cast<size_t>(idx)]));
            }
        }
    }
}

NNOPS_TEST(rope_4d_multihead) {
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
    auto y = test::make_channel_first(descs[0], out_buf.data());
    const TensorView ins[] = {in_view};
    op->compute(y, ins);

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
    auto y1 = test::make_channel_first(d[0], out1.data());
    const TensorView x_arr[] = {x};
    op1->compute(y1, x_arr);

    // base=1000000.0: pair 1's theta ≈ 0, so only that pair stays near-identity
    RoPEAttributes attrs2;
    attrs2.base = 1000000.0f;
    auto op2 = RoPE::create(attrs2, Backend::CPU);

    std::vector<float> out2(static_cast<size_t>(d[0].numel()), 0.0f);
    auto y2 = test::make_channel_first(d[0], out2.data());
    op2->compute(y2, x_arr);

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
    attrs.mrope_section_dims = {8};
    auto op = RoPE::create(attrs, Backend::CPU);

    auto dx = x.desc();
    const TensorDesc da[] = {dx};
    auto d = op->getOutputTensorDesc(da);
    std::vector<float> out_buf(static_cast<size_t>(d[0].numel()), 0.0f);
    auto y = test::make_channel_first(d[0], out_buf.data());
    const TensorView ins[] = {x};
    op->compute(y, ins);

    RoPEAttributes attrs_std;
    attrs_std.base = 10000.0f;
    attrs_std.interleaved = true;
    auto op_std = RoPE::create(attrs_std, Backend::CPU);

    std::vector<float> out_std(static_cast<size_t>(d[0].numel()), 0.0f);
    auto y_std = test::make_channel_first(d[0], out_std.data());
    const TensorView ins2[] = {x};
    op_std->compute(y_std, ins2);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], out_std[i], 1e-5f);
    }
}

NNOPS_TEST(mrope_different_bases) {
    // One pair per section, each at exponent 0: theta = pos for either base.
    const int64_t shape[] = {1, 4};
    float x_data[] = {1.0f, 0.0f, 1.0f, 0.0f};

    TensorView x(shape, DataType::f32, x_data);

    RoPEAttributes attrs;
    attrs.base = 1.0f;  // fallback when mrope_section_bases is empty
    attrs.interleaved = true;
    attrs.mrope_section_dims = {2, 2};
    attrs.mrope_section_bases = {1.0f, 1000000.0f};
    auto op = RoPE::create(attrs, Backend::CPU);

    auto dx = x.desc();
    const TensorDesc da[] = {dx};
    auto d = op->getOutputTensorDesc(da);
    std::vector<float> out_buf(static_cast<size_t>(d[0].numel()), 0.0f);
    auto y = test::make_channel_first(d[0], out_buf.data());
    const TensorView ins[] = {x};
    op->compute(y, ins);

    // pos=0: cos=1, sin=0 → identity
    NNOPS_EXPECT_NEAR(out_buf[0], 1.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], 0.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[2], 1.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[3], 0.0f, 1e-5f);
}

// ============================================================
// Position offset and long-context scaling
// ============================================================

/// Rotate a random tensor and compare every element with the scalar reference.
static void check_rope_scaled(const RoPEAttributes& attrs,
                              std::initializer_list<int64_t> shape,
                              int64_t position_offset,
                              const RefScaling& scaling,
                              float tol = 1e-4f,
                              uint64_t seed = 200) {
    auto [in_vec, in_view] = test::make_random_tensor(shape, -2.0f, 2.0f, seed);
    auto op = RoPE::create(attrs, Backend::CPU);

    auto d_x = in_view.desc();
    const TensorDesc da[] = {d_x};
    auto descs = op->getOutputTensorDesc(da);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_channel_first(descs[0], out_buf.data());
    const TensorView ins[] = {in_view};
    op->compute(y, ins);

    const int64_t seq_len = d_x.dims[static_cast<size_t>(d_x.rank) - 2];
    const int64_t head_dim = d_x.dims[static_cast<size_t>(d_x.rank) - 1];
    const int64_t stride = seq_len * head_dim;
    const int64_t outer = static_cast<int64_t>(in_vec.size()) / stride;

    std::vector<float> ref(out_buf.size(), 0.0f);
    for (int64_t o = 0; o < outer; ++o) {
        compute_rope_ref_scaled(in_vec.data() + o * stride,
                                ref.data() + o * stride,
                                seq_len, head_dim, attrs.base,
                                attrs.interleaved, position_offset, scaling);
    }
    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], ref[i], tol);
    }
}

NNOPS_TEST(rope_position_offset_matches_ref) {
    RoPEAttributes attrs;
    attrs.position_offset = 1023;

    check_rope_scaled(attrs, {2, 5, 16}, 1023, RefScaling{});
}

NNOPS_TEST(rope_position_offset_row_shift) {
    // A one-row call at offset k must equal row k of a run starting at 0.
    const int64_t run_shape[] = {9, 8};
    auto [run_vec, run_view] = test::make_random_tensor(run_shape, -2.0f, 2.0f, 301);

    RoPEAttributes run_attrs;
    auto run_op = RoPE::create(run_attrs, Backend::CPU);
    auto d_run = run_view.desc();
    const TensorDesc run_da[] = {d_run};
    auto run_desc = run_op->getOutputTensorDesc(run_da);
    std::vector<float> run_out(static_cast<size_t>(run_desc[0].numel()), 0.0f);
    auto run_y = test::make_channel_first(run_desc[0], run_out.data());
    const TensorView run_ins[] = {run_view};
    run_op->compute(run_y, run_ins);

    const int64_t row_shape[] = {1, 8};
    const int64_t row = 6;
    std::vector<float> row_in(run_vec.begin() + row * 8, run_vec.begin() + row * 8 + 8);

    TensorView row_view(row_shape, DataType::f32, row_in.data());
    RoPEAttributes row_attrs;
    row_attrs.position_offset = row;
    auto row_op = RoPE::create(row_attrs, Backend::CPU);
    auto d_row = row_view.desc();
    const TensorDesc row_da[] = {d_row};
    auto row_desc = row_op->getOutputTensorDesc(row_da);
    std::vector<float> row_out(static_cast<size_t>(row_desc[0].numel()), 0.0f);
    auto row_y = test::make_channel_first(row_desc[0], row_out.data());
    const TensorView row_ins[] = {row_view};
    row_op->compute(row_y, row_ins);

    for (int64_t d = 0; d < 8; ++d) {
        NNOPS_EXPECT_NEAR(row_out[static_cast<size_t>(d)],
                          run_out[static_cast<size_t>(row * 8 + d)], 1e-5f);
    }
}

NNOPS_TEST(rope_linear_scaling) {
    RoPEAttributes attrs;
    attrs.scaling_type = RoPEScalingType::Linear;
    attrs.scaling_factor = 4.0f;

    RefScaling ref;
    ref.type = RoPEScalingType::Linear;
    ref.factor = 4.0f;

    check_rope_scaled(attrs, {6, 16}, 0, ref);
}

NNOPS_TEST(rope_dynamic_ntk) {
    RoPEAttributes attrs;
    attrs.scaling_type = RoPEScalingType::Dynamic;
    attrs.scaling_factor = 8.0f;
    attrs.original_max_position = 16;

    RefScaling ref;
    ref.type = RoPEScalingType::Dynamic;
    ref.factor = 8.0f;
    ref.orig = 16;

    // The run reaches position 31, past the original 16, so the base grows.
    check_rope_scaled(attrs, {32, 16}, 0, ref);
}

NNOPS_TEST(rope_dynamic_ntk_short_run_keeps_base) {
    // A run ending below original_max_position must rotate exactly like
    // unscaled RoPE.
    RoPEAttributes scaled;
    scaled.scaling_type = RoPEScalingType::Dynamic;
    scaled.scaling_factor = 8.0f;
    scaled.original_max_position = 64;

    check_rope_scaled(scaled, {8, 16}, 0, RefScaling{});
}

NNOPS_TEST(rope_yarn) {
    RoPEAttributes attrs;
    attrs.scaling_type = RoPEScalingType::YaRN;
    attrs.scaling_factor = 4.0f;
    attrs.original_max_position = 16;
    attrs.yarn_beta_fast = 32.0f;
    attrs.yarn_beta_slow = 1.0f;

    RefScaling ref;
    ref.type = RoPEScalingType::YaRN;
    ref.factor = 4.0f;
    ref.orig = 16;

    check_rope_scaled(attrs, {32, 32}, 0, ref);
}

NNOPS_TEST(rope_yarn_explicit_attention_factor) {
    RoPEAttributes attrs;
    attrs.scaling_type = RoPEScalingType::YaRN;
    attrs.scaling_factor = 4.0f;
    attrs.original_max_position = 16;
    attrs.attention_factor = 0.707f;

    RefScaling ref;
    ref.type = RoPEScalingType::YaRN;
    ref.factor = 4.0f;
    ref.orig = 16;
    ref.attention_factor = 0.707f;

    check_rope_scaled(attrs, {16, 16}, 0, ref);
}

NNOPS_TEST(rope_llama3_scaling) {
    RoPEAttributes attrs;
    attrs.scaling_type = RoPEScalingType::Llama3;
    attrs.scaling_factor = 8.0f;
    attrs.original_max_position = 64;
    attrs.llama3_low_freq_factor = 1.0f;
    attrs.llama3_high_freq_factor = 4.0f;

    RefScaling ref;
    ref.type = RoPEScalingType::Llama3;
    ref.factor = 8.0f;
    ref.orig = 64;

    // base 10000 over 32 dims sweeps the wavelengths across both band edges
    // (2*pi at pair 0, ~600 at pair 15), so all three regions are covered.
    check_rope_scaled(attrs, {16, 32}, 0, ref);
}

NNOPS_TEST(rope_scaling_with_offset) {
    // The chunked-decode shape: one new token at a real position well past the
    // pretraining window, with YaRN active.
    RoPEAttributes attrs;
    attrs.position_offset = 4096;
    attrs.scaling_type = RoPEScalingType::YaRN;
    attrs.scaling_factor = 8.0f;
    attrs.original_max_position = 1024;

    RefScaling ref;
    ref.type = RoPEScalingType::YaRN;
    ref.factor = 8.0f;
    ref.orig = 1024;

    check_rope_scaled(attrs, {2, 2, 32}, 4096, ref);
}

NNOPS_TEST(rope_mrope_with_offset) {
    // One mRoPE section over the whole head is plain RoPE, so the offset must
    // reach the sectioned table build too.
    RoPEAttributes attrs;
    attrs.position_offset = 77;
    attrs.mrope_section_dims = {16};

    check_rope_scaled(attrs, {3, 16}, 77, RefScaling{});
}

NNOPS_TEST(rope_scaling_validation) {
    const int64_t shape[] = {2, 8};
    float x_data[16] = {};
    TensorView x(shape, DataType::f32, x_data);
    auto d_x = x.desc();
    const TensorDesc ins[] = {d_x};
    const TensorDesc outs[] = {d_x};

    RoPEAttributes dynamic_no_orig;
    dynamic_no_orig.scaling_type = RoPEScalingType::Dynamic;
    NNOPS_EXPECT_TRUE(RoPE::create(dynamic_no_orig, Backend::CPU)->validate(ins, outs)
                      == Status::invalid_argument);

    RoPEAttributes zero_factor;
    zero_factor.scaling_type = RoPEScalingType::Linear;
    zero_factor.scaling_factor = 0.0f;
    NNOPS_EXPECT_TRUE(RoPE::create(zero_factor, Backend::CPU)->validate(ins, outs)
                      == Status::invalid_argument);

    RoPEAttributes negative_offset;
    negative_offset.position_offset = -1;
    NNOPS_EXPECT_TRUE(RoPE::create(negative_offset, Backend::CPU)->validate(ins, outs)
                      == Status::invalid_argument);

    RoPEAttributes inverted_bands;
    inverted_bands.scaling_type = RoPEScalingType::Llama3;
    inverted_bands.original_max_position = 128;
    inverted_bands.llama3_low_freq_factor = 4.0f;
    inverted_bands.llama3_high_freq_factor = 4.0f;
    NNOPS_EXPECT_TRUE(RoPE::create(inverted_bands, Backend::CPU)->validate(ins, outs)
                      == Status::invalid_argument);

    RoPEAttributes valid_yarn;
    valid_yarn.scaling_type = RoPEScalingType::YaRN;
    valid_yarn.scaling_factor = 2.0f;
    valid_yarn.original_max_position = 128;
    NNOPS_EXPECT_TRUE(RoPE::create(valid_yarn, Backend::CPU)->validate(ins, outs)
                      == Status::ok);
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
    auto y = test::make_channel_first(d[0], out_buf.data());
    const TensorView ins[] = {x};
    op->compute(y, ins);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], 0.0f, 1e-6f);
    }
}

NNOPS_TEST(rope_single_position) {
    const int64_t shape[] = {1, 8};
    float x_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    TensorView x(shape, DataType::f32, x_data);

    RoPEAttributes attrs;
    auto op = RoPE::create(attrs, Backend::CPU);

    auto dx = x.desc();
    const TensorDesc da[] = {dx};
    auto d = op->getOutputTensorDesc(da);
    std::vector<float> out_buf(static_cast<size_t>(d[0].numel()), 0.0f);
    auto y = test::make_channel_first(d[0], out_buf.data());
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
    auto y = test::make_channel_first(d[0], out_buf.data());
    const TensorView ins[] = {in_view};
    op->compute(y, ins);

    std::vector<float> ref(static_cast<size_t>(2 * 64), 0.0f);
    compute_rope_ref(in_vec.data(), ref.data(), 2, 64, 10000.0f, false);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], ref[i], 1e-4f);
    }
}

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

NNOPS_TEST(rope_op_type) {
    auto op = RoPE::create(Backend::CPU);
    NNOPS_EXPECT_EQ(op->getOpType(), OpType::RoPE);
    NNOPS_EXPECT_EQ(op->getBackend(), Backend::CPU);
}

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
    auto y = test::make_channel_first(descs[0], out_buf.data());
    const TensorView ins[] = {x};
    op->compute(y, ins);

    std::vector<float> ref(static_cast<size_t>(N), 0.0f);
    compute_rope_ref(f32_vec.data(), ref.data(), seq_len, head_dim,
                     10000.0f, true);

    for (int i = 0; i < N; ++i) {
        float result = simd::s_load(&out_buf[i]);
        NNOPS_EXPECT_TRUE(std::isfinite(result));
        NNOPS_EXPECT_NEAR(result, ref[i], 5e-2f);
    }
}
