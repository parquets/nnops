/// @file test_embed.cpp
/// @brief Unit tests for Embed operator (CPU backend).

#include "nnops/ops/embed.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/test_helpers.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <vector>
#include <algorithm>
#include <cstring>

using namespace nnops;

// ============================================================
// Hand-verified tests
// ============================================================

NNOPS_TEST(embed_basic) {
    // Weight: [4, 3]
    float weight_data[] = {
        0.1f, 0.2f, 0.3f,   // token 0
        1.1f, 1.2f, 1.3f,   // token 1
        2.1f, 2.2f, 2.3f,   // token 2
        3.1f, 3.2f, 3.3f,   // token 3
    };
    const int64_t w_shape[] = {4, 3};
    TensorView weight(w_shape, DataType::f32, weight_data);

    // Indices: [2] → lookup tokens {1, 3}
    int64_t idx_data[] = {1, 3};
    const int64_t idx_shape[] = {2};
    TensorView indices(idx_shape, DataType::i64, idx_data);

    auto op = Embed::create(EmbedAttributes{}, Backend::CPU);

    const TensorDesc in_arr[] = {weight.desc(), indices.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 3);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(6);
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {weight, indices};
    op->compute(output, ins);

    // [1, 3] → row 1: {1.1, 1.2, 1.3}
    NNOPS_EXPECT_NEAR(out_buf[0], 1.1f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], 1.2f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[2], 1.3f, 1e-5f);
    // [3] → row 3: {3.1, 3.2, 3.3}
    NNOPS_EXPECT_NEAR(out_buf[3], 3.1f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[4], 3.2f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[5], 3.3f, 1e-5f);
}

NNOPS_TEST(embed_out_of_bounds) {
    float weight_data[] = {
        0.1f, 0.2f,   // token 0
        1.1f, 1.2f,   // token 1
        2.1f, 2.2f,   // token 2
    };
    const int64_t w_shape[] = {3, 2};
    TensorView weight(w_shape, DataType::f32, weight_data);

    // Out-of-range indices: -1 → 0, 5 → 2
    int64_t idx_data[] = {-1, 5};
    const int64_t idx_shape[] = {2};
    TensorView indices(idx_shape, DataType::i64, idx_data);

    auto op = Embed::create(EmbedAttributes{}, Backend::CPU);
    const TensorDesc in_arr[] = {weight.desc(), indices.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    std::vector<float> out_buf(4);
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {weight, indices};
    op->compute(output, ins);

    // -1 clamped to 0 → {0.1, 0.2}
    NNOPS_EXPECT_NEAR(out_buf[0], 0.1f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], 0.2f, 1e-5f);
    // 5 clamped to 2 → {2.1, 2.2}
    NNOPS_EXPECT_NEAR(out_buf[2], 2.1f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[3], 2.2f, 1e-5f);
}

// ============================================================
// Random data tests
// ============================================================

NNOPS_TEST(embed_random_f32) {
    const int64_t V = 50, D = 8, N = 200;
    auto [w_vec, weight] = test::make_random_tensor({V, D}, -1.0f, 1.0f, 111);

    // Random indices in [0, V-1]
    std::vector<int64_t> idx_vec(N);
    for (int64_t i = 0; i < N; ++i)
        idx_vec[i] = static_cast<int64_t>(i * 7 % V);  // deterministic spread
    const int64_t idx_shape[] = {N};
    TensorView indices(idx_shape, DataType::i64, idx_vec.data());

    auto op = Embed::create(EmbedAttributes{}, Backend::CPU);
    const TensorDesc in_arr[] = {weight.desc(), indices.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], N);
    NNOPS_EXPECT_EQ(descs[0].dims[1], D);

    std::vector<float> out_buf(static_cast<size_t>(N * D));
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {weight, indices};
    op->compute(output, ins);

    // Compare each output row against weight[idx]
    for (int64_t n = 0; n < N; ++n) {
        int64_t idx = idx_vec[n];
        for (int64_t d = 0; d < D; ++d) {
            float expected = w_vec[static_cast<size_t>(idx * D + d)];
            float result = out_buf[static_cast<size_t>(n * D + d)];
            NNOPS_EXPECT_NEAR(result, expected, 1e-5f);
        }
    }
}

NNOPS_TEST(embed_random_f16) {
    const int64_t V = 20, D = 4, N = 30;
    auto [w_vec, _] = test::make_random_tensor({V, D}, -1.0f, 1.0f, 222);

    // Convert weight to f16
    std::vector<nnops::backend::cpu::half> w_half(static_cast<size_t>(V * D));
    for (size_t i = 0; i < w_half.size(); ++i)
        simd::s_store(&w_half[i], w_vec[i]);

    const int64_t w_shape[] = {V, D};
    TensorView weight(w_shape, DataType::f16, w_half.data());

    std::vector<int64_t> idx_vec(N);
    for (int64_t i = 0; i < N; ++i) idx_vec[i] = static_cast<int64_t>(i % V);
    const int64_t idx_shape[] = {N};
    TensorView indices(idx_shape, DataType::i64, idx_vec.data());

    auto op = Embed::create(EmbedAttributes{}, Backend::CPU);
    const TensorDesc in_arr[] = {weight.desc(), indices.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

    std::vector<nnops::backend::cpu::half> out_buf(static_cast<size_t>(N * D));
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {weight, indices};
    op->compute(output, ins);

    for (int64_t n = 0; n < N; ++n) {
        int64_t idx = idx_vec[n];
        for (int64_t d = 0; d < D; ++d) {
            float expected = w_vec[static_cast<size_t>(idx * D + d)];
            float result = simd::s_load(&out_buf[static_cast<size_t>(n * D + d)]);
            NNOPS_EXPECT_NEAR(result, expected, 5e-3f);  // f16 tolerance
        }
    }
}

// ============================================================
// 2D indices (batch, sequence)
// ============================================================

NNOPS_TEST(embed_batch_indices) {
    const int64_t V = 10, D = 3, B = 2, S = 4;
    auto [w_vec, weight] = test::make_random_tensor({V, D}, -1.0f, 1.0f, 333);

    // 2D indices: [B, S]
    int64_t idx_data[] = {0, 1, 2, 3,  4, 5, 6, 7};
    const int64_t idx_shape[] = {B, S};
    TensorView indices(idx_shape, DataType::i64, idx_data);

    auto op = Embed::create(EmbedAttributes{}, Backend::CPU);
    const TensorDesc in_arr[] = {weight.desc(), indices.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 3);
    NNOPS_EXPECT_EQ(descs[0].dims[0], B);
    NNOPS_EXPECT_EQ(descs[0].dims[1], S);
    NNOPS_EXPECT_EQ(descs[0].dims[2], D);

    std::vector<float> out_buf(static_cast<size_t>(B * S * D));
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {weight, indices};
    op->compute(output, ins);

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t s = 0; s < S; ++s) {
            int64_t idx = idx_data[b * S + s];
            for (int64_t d = 0; d < D; ++d) {
                float expected = w_vec[static_cast<size_t>(idx * D + d)];
                float result = out_buf[static_cast<size_t>((b * S + s) * D + d)];
                NNOPS_EXPECT_NEAR(result, expected, 1e-5f);
            }
        }
    }
}

// ============================================================
// Degenerate cases
// ============================================================

NNOPS_TEST(embed_zero_dim) {
    // Weight: [V, 1] — single-dim embeddings
    float weight_data[] = {10.0f, 20.0f, 30.0f};
    const int64_t w_shape[] = {3, 1};
    TensorView weight(w_shape, DataType::f32, weight_data);

    int64_t idx_data[] = {0, 2};
    const int64_t idx_shape[] = {2};
    TensorView indices(idx_shape, DataType::i64, idx_data);

    auto op = Embed::create(EmbedAttributes{}, Backend::CPU);
    const TensorDesc in_arr[] = {weight.desc(), indices.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 1);

    float out_buf[2] = {};
    TensorView output = test::make_planar(descs[0], out_buf);
    const TensorView ins[] = {weight, indices};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], 10.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], 30.0f, 1e-5f);
}

NNOPS_TEST(embed_shape_inference) {
    const int64_t w_shape[] = {100, 64};
    const int64_t i_shape[] = {2, 5};
    float w_dummy[1] = {};
    int64_t i_dummy[1] = {};
    TensorView weight(w_shape, DataType::f32, w_dummy);
    TensorView indices(i_shape, DataType::i64, i_dummy);

    auto op = Embed::create(EmbedAttributes{}, Backend::CPU);
    const TensorDesc in_arr[] = {weight.desc(), indices.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, 3);  // indices.rank(2) + weight.rank-1(1) = 3
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 5);
    NNOPS_EXPECT_EQ(descs[0].dims[2], 64);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
}

// ============================================================
// Functional API test
// ============================================================

NNOPS_TEST(embed_functional_api) {
    float weight_data[] = {
        0.1f, 0.2f, 0.3f,
        1.1f, 1.2f, 1.3f,
    };
    const int64_t w_shape[] = {2, 3};
    TensorView weight(w_shape, DataType::f32, weight_data);

    int64_t idx_data[] = {1, 0};
    const int64_t idx_shape[] = {2};
    TensorView indices(idx_shape, DataType::i64, idx_data);

    // Class API
    std::vector<float> out_class(6);
    {
        auto op = Embed::create(EmbedAttributes{}, Backend::CPU);
        const TensorDesc in_arr[] = {weight.desc(), indices.desc()};
        auto descs = op->getOutputTensorDesc(in_arr);
        TensorView output = test::make_planar(descs[0], out_class.data());
        const TensorView ins[] = {weight, indices};
        op->compute(output, ins);
    }

    // Functional API
    std::vector<float> out_func(6);
    {
        auto op = Embed::create(EmbedAttributes{}, Backend::CPU);
        const TensorDesc in_arr[] = {weight.desc(), indices.desc()};
        auto descs = op->getOutputTensorDesc(in_arr);
        TensorView output = test::make_planar(descs[0], out_func.data());
        embed(weight, indices, output);
    }

    for (size_t i = 0; i < 6; ++i) {
        NNOPS_EXPECT_NEAR(out_class[i], out_func[i], 1e-5f);
    }
}
