/// @file test_argminmax.cpp
/// @brief ArgMax / ArgMin (CPU): hand-verified indices, brute-force cross-checks,
/// and the descriptor / OpType surface.

#include "nnops/ops/argminmax.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"

#include <vector>
#include <cstring>
#include <limits>

using namespace nnops;

// ============================================================
// Hand-verified tests — ArgMax
// ============================================================

NNOPS_TEST(argmax_1d_basic) {
    const int64_t shape[] = {5};
    float data[] = {1.0f, 5.0f, 3.0f, 9.0f, 2.0f};
    int64_t out_data = -1;

    TensorView in(shape, DataType::f32, data);

    ArgMinMaxAttributes attrs;
    attrs.type = ArgMinMaxType::Max;
    attrs.axis = 0;
    auto op = ArgMax::create(attrs, Backend::CPU);

    auto d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 0);  // scalar
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::s64);

    auto output = test::make_channel_first(descs[0], &out_data);
    const TensorView ins[] = {in};
    op->compute(output, ins);

    NNOPS_EXPECT_EQ(out_data, 3);  // index of max (9.0)
}

NNOPS_TEST(argmax_1d_with_keepdims) {
    const int64_t shape[] = {5};
    float data[] = {10.0f, 3.0f, 7.0f, 1.0f, 8.0f};
    int64_t out_data[1] = {-1};

    TensorView in(shape, DataType::f32, data);

    ArgMinMaxAttributes attrs;
    attrs.type = ArgMinMaxType::Max;
    attrs.axis = 0;
    attrs.keepdims = true;
    auto op = ArgMax::create(attrs, Backend::CPU);

    auto d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 1);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::s64);

    auto output = test::make_channel_first(descs[0], out_data);
    const TensorView ins[] = {in};
    op->compute(output, ins);

    NNOPS_EXPECT_EQ(out_data[0], 0);  // max at index 0 (10.0)
}

NNOPS_TEST(argmax_2d_axis1) {
    const int64_t shape[] = {2, 4};  // 4 columns per row, axis=1
    float data[] = {1.0f, 9.0f, 3.0f, 2.0f,
                    5.0f, 1.0f, 8.0f, 4.0f};
    int64_t out_data[2] = {-1, -1};

    TensorView in(shape, DataType::f32, data);

    ArgMinMaxAttributes attrs;
    attrs.type = ArgMinMaxType::Max;
    attrs.axis = 1;
    auto op = ArgMax::create(attrs, Backend::CPU);

    auto d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);

    auto output = test::make_channel_first(descs[0], out_data);
    const TensorView ins[] = {in};
    op->compute(output, ins);

    NNOPS_EXPECT_EQ(out_data[0], 1);  // row 0: max 9.0 at idx 1
    NNOPS_EXPECT_EQ(out_data[1], 2);  // row 1: max 8.0 at idx 2
}

NNOPS_TEST(argmax_3d_axis0) {
    const int64_t shape[] = {3, 2, 2};  // axis=0: per (row,col) depth argmax
    // Depth 0: [1,2; 3,4]  Depth 1: [5,6; 7,8]  Depth 2: [2,1; 4,3]
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f,
                    2.0f, 1.0f, 4.0f, 3.0f};
    int64_t out_data[4] = {-1};

    TensorView in(shape, DataType::f32, data);

    ArgMinMaxAttributes attrs;
    attrs.type = ArgMinMaxType::Max;
    attrs.axis = 0;
    auto op = ArgMax::create(attrs, Backend::CPU);

    auto d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 2);

    auto output = test::make_channel_first(descs[0], out_data);
    const TensorView ins[] = {in};
    op->compute(output, ins);

    // Position (0,0): depths [1, 5, 2] → max 5 at depth 1
    NNOPS_EXPECT_EQ(out_data[0], 1);
    // Position (0,1): depths [2, 6, 1] → max 6 at depth 1
    NNOPS_EXPECT_EQ(out_data[1], 1);
    // Position (1,0): depths [3, 7, 4] → max 7 at depth 1
    NNOPS_EXPECT_EQ(out_data[2], 1);
    // Position (1,1): depths [4, 8, 3] → max 8 at depth 1
    NNOPS_EXPECT_EQ(out_data[3], 1);
}

// ============================================================
// Hand-verified tests — ArgMin
// ============================================================

NNOPS_TEST(argmin_1d_basic) {
    const int64_t shape[] = {5};
    float data[] = {8.0f, 5.0f, 1.0f, 9.0f, 2.0f};
    int64_t out_data = -1;

    TensorView in(shape, DataType::f32, data);

    ArgMinMaxAttributes attrs;
    attrs.type = ArgMinMaxType::Min;
    attrs.axis = 0;
    auto op = ArgMin::create(attrs, Backend::CPU);

    auto d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 0);  // scalar
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::s64);

    auto output = test::make_channel_first(descs[0], &out_data);
    const TensorView ins[] = {in};
    op->compute(output, ins);

    NNOPS_EXPECT_EQ(out_data, 2);  // min 1.0 at idx 2
}

NNOPS_TEST(argmin_2d_axis1) {
    const int64_t shape[] = {2, 3};
    float data[] = {5.0f, 1.0f, 8.0f,
                    4.0f, 7.0f, 2.0f};
    int64_t out_data[2] = {-1, -1};

    TensorView in(shape, DataType::f32, data);

    ArgMinMaxAttributes attrs;
    attrs.type = ArgMinMaxType::Min;
    attrs.axis = 1;
    auto op = ArgMin::create(attrs, Backend::CPU);

    auto d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    auto output = test::make_channel_first(descs[0], out_data);
    const TensorView ins[] = {in};
    op->compute(output, ins);

    NNOPS_EXPECT_EQ(out_data[0], 1);  // row 0: min 1.0 at idx 1
    NNOPS_EXPECT_EQ(out_data[1], 2);  // row 1: min 2.0 at idx 2
}

// ============================================================
// Random tests
// ============================================================

NNOPS_TEST(argmax_random_vs_ref) {
    auto [data, in] = test::make_random_tensor({4, 10}, -1.0f, 1.0f, 123);

    ArgMinMaxAttributes attrs;
    attrs.type = ArgMinMaxType::Max;
    attrs.axis = 1;

    auto op = ArgMax::create(attrs, Backend::CPU);
    auto d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    std::vector<int64_t> out_data(4, -1);
    auto output = test::make_channel_first(descs[0], out_data.data());
    const TensorView ins[] = {in};
    op->compute(output, ins);

    for (int64_t r = 0; r < 4; ++r) {
        int64_t best_idx = 0;
        float best_val = data[r * 10];
        for (int64_t c = 1; c < 10; ++c) {
            if (data[r * 10 + c] > best_val) {
                best_val = data[r * 10 + c];
                best_idx = c;
            }
        }
        NNOPS_EXPECT_EQ(out_data[r], best_idx);
    }
}

NNOPS_TEST(argmin_random_vs_ref) {
    auto [data, in] = test::make_random_tensor({3, 8}, -1.0f, 1.0f, 456);

    ArgMinMaxAttributes attrs;
    attrs.type = ArgMinMaxType::Min;
    attrs.axis = 1;

    auto op = ArgMin::create(attrs, Backend::CPU);
    auto d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    std::vector<int64_t> out_data(3, -1);
    auto output = test::make_channel_first(descs[0], out_data.data());
    const TensorView ins[] = {in};
    op->compute(output, ins);

    for (int64_t r = 0; r < 3; ++r) {
        int64_t best_idx = 0;
        float best_val = data[r * 8];
        for (int64_t c = 1; c < 8; ++c) {
            if (data[r * 8 + c] < best_val) {
                best_val = data[r * 8 + c];
                best_idx = c;
            }
        }
        NNOPS_EXPECT_EQ(out_data[r], best_idx);
    }
}

// ============================================================
// The walk follows the strides, not the shape
// ============================================================

/// A 5-element row stored in an 8-element pitch, reduced along the channel axis
/// so the dims after it (H, W) are the ones the pitch breaks. The padding cells
/// hold a value that would win, so a walk that reads them answers with the wrong
/// index rather than settling for a tie.
NNOPS_TEST(argmax_pitched_row) {
    constexpr int64_t N = 2, C = 4, H = 3, W = 5;
    constexpr int64_t pitch = 8;
    constexpr float pad_val = 100.0f;

    std::vector<float> logical(N * C * H * W);
    for (size_t i = 0; i < logical.size(); ++i) {
        logical[i] = static_cast<float>((i * 37) % 23) * 0.5f;
    }
    auto at = [&](int64_t n, int64_t c, int64_t h, int64_t w) -> float {
        return logical[((n * C + c) * H + h) * W + w];
    };

    std::vector<float> storage(N * C * H * pitch, pad_val);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    storage[((n * C + c) * H + h) * pitch + w] = at(n, c, h, w);
                }
            }
        }
    }

    const int64_t shape[] = {N, C, H, W};
    TensorView in(shape, DataType::f32, storage.data(), pitch * 4);

    ArgMinMaxAttributes attrs;
    attrs.type = ArgMinMaxType::Max;
    attrs.axis = 1;
    auto op = ArgMax::create(attrs, Backend::CPU);
    const TensorDesc d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    std::vector<int64_t> out_data(N * H * W, -1);
    auto output = test::make_channel_first(descs[0], out_data.data());
    const TensorView ins[] = {in};
    op->compute(output, ins);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t w = 0; w < W; ++w) {
                int64_t best_idx = 0;
                float best_val = at(n, 0, h, w);
                for (int64_t c = 1; c < C; ++c) {
                    if (at(n, c, h, w) > best_val) {
                        best_val = at(n, c, h, w);
                        best_idx = c;
                    }
                }
                NNOPS_EXPECT_EQ(out_data[(n * H + h) * W + w], best_idx);
            }
        }
    }
}

/// Channel-last memory: (n, h, w, c) with the channel innermost, reduced over
/// the channel — the dims after the axis are interleaved, so the flat inner
/// index has to be decoded, not added.
NNOPS_TEST(argmax_channel_last) {
    constexpr int64_t N = 2, C = 4, H = 3, W = 5;

    // Memory order (n, h, w, c).
    std::vector<float> storage(N * H * W * C);
    for (size_t i = 0; i < storage.size(); ++i) {
        storage[i] = static_cast<float>((i * 29) % 17) * 0.25f;
    }
    auto at = [&](int64_t n, int64_t c, int64_t h, int64_t w) -> float {
        return storage[((n * H + h) * W + w) * C + c];
    };

    const int64_t shape[] = {N, C, H, W};
    TensorView in(shape, DataType::f32, storage.data(), TensorLayout::NHWC);

    ArgMinMaxAttributes attrs;
    attrs.type = ArgMinMaxType::Max;
    attrs.axis = 1;
    auto op = ArgMax::create(attrs, Backend::CPU);
    const TensorDesc d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    // The output inherits the input layout, so it is channel-last too: the
    // values are read back through its own strides.
    std::vector<int64_t> out_data(N * H * W, -1);
    auto output = test::make_channel_first(descs[0], out_data.data());
    const TensorView ins[] = {in};
    op->compute(output, ins);

    const int64_t o_n = output.stride_elems(0);
    const int64_t o_h = output.stride_elems(1);
    const int64_t o_w = output.stride_elems(2);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t w = 0; w < W; ++w) {
                int64_t best_idx = 0;
                float best_val = at(n, 0, h, w);
                for (int64_t c = 1; c < C; ++c) {
                    if (at(n, c, h, w) > best_val) {
                        best_val = at(n, c, h, w);
                        best_idx = c;
                    }
                }
                NNOPS_EXPECT_EQ(out_data[n * o_n + h * o_h + w * o_w], best_idx);
            }
        }
    }
}

NNOPS_TEST(argmax_class_1d) {
    const int64_t shape[] = {4};
    float data[] = {1.0f, 8.0f, 3.0f, 6.0f};
    int64_t out_data = -1;

    TensorView in(shape, DataType::f32, data);

    ArgMinMaxAttributes attrs;
    attrs.type = ArgMinMaxType::Max;
    attrs.axis = 0;
    auto op = ArgMax::create(attrs, Backend::CPU);
    auto d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    auto output = test::make_channel_first(descs[0], &out_data);
    const TensorView ins[] = {in};
    op->compute(output, ins);

    NNOPS_EXPECT_EQ(out_data, 1);  // max 8.0 at idx 1
}

NNOPS_TEST(argmin_class_1d) {
    const int64_t shape[] = {4};
    float data[] = {7.0f, 3.0f, 9.0f, 1.0f};
    int64_t out_data = -1;

    TensorView in(shape, DataType::f32, data);

    ArgMinMaxAttributes attrs;
    attrs.type = ArgMinMaxType::Min;
    attrs.axis = 0;
    auto op = ArgMin::create(attrs, Backend::CPU);
    auto d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    auto output = test::make_channel_first(descs[0], &out_data);
    const TensorView ins[] = {in};
    op->compute(output, ins);

    NNOPS_EXPECT_EQ(out_data, 3);  // min 1.0 at idx 3
}

NNOPS_TEST(argmax_op_type) {
    auto op = ArgMax::create(Backend::CPU);
    NNOPS_EXPECT_EQ(static_cast<int>(op->getOpType()), static_cast<int>(OpType::ArgMax));
}

NNOPS_TEST(argmin_op_type) {
    auto op = ArgMin::create(Backend::CPU);
    NNOPS_EXPECT_EQ(static_cast<int>(op->getOpType()), static_cast<int>(OpType::ArgMin));
}
