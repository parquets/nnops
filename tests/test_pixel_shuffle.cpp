/// @file test_pixel_shuffle.cpp
/// @brief SpaceToDepth / PixelShuffle on the CPU backend.
///
/// Hand cases pin the channel order of both directions and of the CRD / DCR
/// PixelShuffle modes; the random round-trip checks that SpaceToDepth(b)
/// followed by PixelShuffle(b, DCR) is the identity for b = 2 and 3. A row-padded
/// NHWC case covers the stride walk, and u8 / f16 cover the movement dtypes.

#include "nnops/ops/pixel_shuffle.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/test_helpers.hpp"

#include <cstring>
#include <vector>

using namespace nnops;

namespace {

TensorView make_out(const TensorDesc& d, std::vector<float>& buf)
{
    buf.assign(static_cast<size_t>(d.numel()), 0.0f);
    return TensorView(std::span<const int64_t>(d.dims.data(), static_cast<size_t>(d.rank)),
                      DataType::f32, buf.data(), d.layout);
}

} // anonymous namespace

NNOPS_TEST(space_to_depth_hand) {
    // X = [[1, 2], [3, 4]] (one batch, one channel) folds to channels
    // [x(0,0), x(0,1), x(1,0), x(1,1)].
    const int64_t xs[] = {1, 2, 2, 1};
    float x_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    TensorView x(xs, DataType::f32, x_data, TensorLayout::NHWC);

    SpaceToDepthAttributes attrs;
    attrs.block_size = 2;
    auto op = SpaceToDepth::create(attrs, Backend::CPU);

    const TensorDesc in_desc[] = {x.desc()};
    const auto out_desc = op->getOutputTensorDesc(in_desc);
    NNOPS_EXPECT_EQ(out_desc[0].rank, 4);
    NNOPS_EXPECT_EQ(out_desc[0].dims[0], int64_t{1});
    NNOPS_EXPECT_EQ(out_desc[0].dims[1], int64_t{1});
    NNOPS_EXPECT_EQ(out_desc[0].dims[2], int64_t{1});
    NNOPS_EXPECT_EQ(out_desc[0].dims[3], int64_t{4});

    std::vector<float> out_buf;
    auto out = make_out(out_desc[0], out_buf);
    const TensorView ins[] = {x};
    NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::ok);

    NNOPS_EXPECT_NEAR(out_buf[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[1], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[2], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[3], 4.0f, 1e-6f);
}

NNOPS_TEST(pixel_shuffle_hand_dcr) {
    // The inverse of the case above: four channels unfold into a 2×2 image.
    const int64_t xs[] = {1, 1, 1, 4};
    float x_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    TensorView x(xs, DataType::f32, x_data, TensorLayout::NHWC);

    PixelShuffleAttributes attrs;
    attrs.block_size = 2;
    attrs.mode = PixelShuffleMode::Dcr;
    auto op = PixelShuffle::create(attrs, Backend::CPU);

    const TensorDesc in_desc[] = {x.desc()};
    const auto out_desc = op->getOutputTensorDesc(in_desc);
    NNOPS_EXPECT_EQ(out_desc[0].dims[0], int64_t{1});
    NNOPS_EXPECT_EQ(out_desc[0].dims[1], int64_t{2});
    NNOPS_EXPECT_EQ(out_desc[0].dims[2], int64_t{2});
    NNOPS_EXPECT_EQ(out_desc[0].dims[3], int64_t{1});

    std::vector<float> out_buf;
    auto out = make_out(out_desc[0], out_buf);
    const TensorView ins[] = {x};
    NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::ok);

    NNOPS_EXPECT_NEAR(out_buf[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[1], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[2], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[3], 4.0f, 1e-6f);
}

NNOPS_TEST(pixel_shuffle_dcr_vs_crd) {
    // Channels 0..7, two output channels, 2×2 unfold.
    const int64_t xs[] = {1, 1, 1, 8};
    std::vector<float> x_data(8);
    for (int i = 0; i < 8; ++i) { x_data[static_cast<size_t>(i)] = static_cast<float>(i); }
    TensorView x(xs, DataType::f32, x_data.data(), TensorLayout::NHWC);

    const float dcr_want[] = {0, 4, 1, 5, 2, 6, 3, 7};
    const float crd_want[] = {0, 1, 2, 3, 4, 5, 6, 7};

    for (int mode = 0; mode < 2; ++mode) {
        PixelShuffleAttributes attrs;
        attrs.block_size = 2;
        attrs.mode = mode == 0 ? PixelShuffleMode::Dcr : PixelShuffleMode::Crd;
        auto op = PixelShuffle::create(attrs, Backend::CPU);

        const TensorDesc in_desc[] = {x.desc()};
        const auto out_desc = op->getOutputTensorDesc(in_desc);
        NNOPS_EXPECT_EQ(out_desc[0].dims[3], int64_t{2});

        std::vector<float> out_buf;
        auto out = make_out(out_desc[0], out_buf);
        const TensorView ins[] = {x};
        NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::ok);

        const float* want = mode == 0 ? dcr_want : crd_want;
        for (int i = 0; i < 8; ++i) {
            NNOPS_EXPECT_NEAR(out_buf[static_cast<size_t>(i)], want[i], 1e-6f);
        }
    }
}

NNOPS_TEST(pixel_shuffle_round_trip) {
    const int64_t blocks[] = {2, 3};
    const int64_t widths[] = {8, 9};
    for (int bi = 0; bi < 2; ++bi) {
        const int64_t b = blocks[bi];
        const int64_t N = 2, H = 6, W = widths[bi], C = 5;
        auto [x_vec, _] = test::make_random_tensor({N, H, W, C}, -2.0f, 2.0f, 500 + b);

        const int64_t xs[] = {N, H, W, C};
        TensorView x(xs, DataType::f32, x_vec.data(), TensorLayout::NHWC);

        SpaceToDepthAttributes sattrs;
        sattrs.block_size = b;
        auto s2d = SpaceToDepth::create(sattrs, Backend::CPU);
        const TensorDesc x_desc[] = {x.desc()};
        const auto down_desc = s2d->getOutputTensorDesc(x_desc);

        std::vector<float> down_buf;
        auto down = make_out(down_desc[0], down_buf);
        const TensorView down_ins[] = {x};
        NNOPS_EXPECT_TRUE(s2d->compute(down, down_ins) == Status::ok);

        PixelShuffleAttributes pattrs;
        pattrs.block_size = b;
        pattrs.mode = PixelShuffleMode::Dcr;
        auto ps = PixelShuffle::create(pattrs, Backend::CPU);
        const TensorDesc down_desc_arr[] = {down_desc[0]};
        const auto back_desc = ps->getOutputTensorDesc(down_desc_arr);
        NNOPS_EXPECT_EQ(back_desc[0].dims[1], H);
        NNOPS_EXPECT_EQ(back_desc[0].dims[2], W);
        NNOPS_EXPECT_EQ(back_desc[0].dims[3], C);

        std::vector<float> back_buf;
        auto back = make_out(back_desc[0], back_buf);
        const TensorView back_ins[] = {down};
        NNOPS_EXPECT_TRUE(ps->compute(back, back_ins) == Status::ok);

        for (size_t i = 0; i < back_buf.size(); ++i) {
            NNOPS_EXPECT_NEAR(back_buf[i], x_vec[i], 1e-6f);
        }
    }
}

NNOPS_TEST(space_to_depth_pitched_rows) {
    // 6×3 pixels → a 72-byte row, padded to 96; the output row is 144 → 160.
    const int64_t N = 1, H = 4, W = 6, C = 3;
    const int64_t shape[] = {N, H, W, C};
    std::vector<float> dense(static_cast<size_t>(N * H * W * C));
    for (size_t i = 0; i < dense.size(); ++i) { dense[i] = static_cast<float>(i); }

    TensorView x_dense(shape, DataType::f32, dense.data(), TensorLayout::NHWC);
    const int64_t row_elems = W * C;
    const int64_t pitch = x_dense.desc().row_pitch(32);
    NNOPS_EXPECT_TRUE(pitch > row_elems * 4);

    std::vector<uint8_t> packed(static_cast<size_t>(N * H) * static_cast<size_t>(pitch), 0);
    for (int64_t r = 0; r < N * H; ++r) {
        std::memcpy(packed.data() + r * pitch, dense.data() + r * row_elems,
                    static_cast<size_t>(row_elems) * sizeof(float));
    }
    TensorView x_packed(shape, DataType::f32, packed.data(), pitch, TensorLayout::NHWC);

    SpaceToDepthAttributes attrs;
    attrs.block_size = 2;
    auto op = SpaceToDepth::create(attrs, Backend::CPU);

    const TensorDesc dense_desc[] = {x_dense.desc()};
    const auto want_desc = op->getOutputTensorDesc(dense_desc);
    std::vector<float> want_buf;
    auto want = make_out(want_desc[0], want_buf);
    const TensorView dense_ins[] = {x_dense};
    NNOPS_EXPECT_TRUE(op->compute(want, dense_ins) == Status::ok);

    const TensorDesc packed_desc[] = {x_packed.desc()};
    const auto got_desc = op->getOutputTensorDesc(packed_desc);
    const int64_t out_pitch = got_desc[0].row_pitch(32);
    std::vector<uint8_t> out_packed(got_desc[0].storage_bytes(32), 0);
    TensorView got_view(std::span<const int64_t>(got_desc[0].dims.data(), 4), DataType::f32,
                        out_packed.data(), out_pitch, TensorLayout::NHWC);
    const TensorView packed_ins[] = {x_packed};
    NNOPS_EXPECT_TRUE(op->compute(got_view, packed_ins) == Status::ok);

    const int64_t out_row_elems = (W / 2) * (C * 4);
    NNOPS_EXPECT_TRUE(out_pitch > out_row_elems * 4);
    std::vector<float> got(want_buf.size(), 0.0f);
    for (int64_t r = 0; r < N * (H / 2); ++r) {
        std::memcpy(got.data() + r * out_row_elems, out_packed.data() + r * out_pitch,
                    static_cast<size_t>(out_row_elems) * sizeof(float));
    }
    for (size_t i = 0; i < got.size(); ++i) {
        NNOPS_EXPECT_NEAR(got[i], want_buf[i], 1e-6f);
    }
}

NNOPS_TEST(pixel_shuffle_u8) {
    const int64_t xs[] = {1, 2, 2, 1};
    uint8_t x_data[] = {1, 2, 3, 4};
    TensorView x(xs, DataType::u8, x_data, TensorLayout::NHWC);

    SpaceToDepthAttributes attrs;
    attrs.block_size = 2;
    auto op = SpaceToDepth::create(attrs, Backend::CPU);
    const TensorDesc in_desc[] = {x.desc()};
    const auto out_desc = op->getOutputTensorDesc(in_desc);
    NNOPS_EXPECT_EQ(out_desc[0].dtype, DataType::u8);

    uint8_t out_data[4] = {};
    TensorView out(std::span<const int64_t>(out_desc[0].dims.data(), 4), DataType::u8, out_data,
                   TensorLayout::NHWC);
    const TensorView ins[] = {x};
    NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::ok);
    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_EQ(out_data[i], static_cast<uint8_t>(i + 1));
    }
}

NNOPS_TEST(pixel_shuffle_f16) {
    auto [x_f32, _] = test::make_random_tensor({1, 3, 4, 8}, -1.0f, 1.0f, 510);
    auto x_h = test::f32_to_f16(x_f32);
    const int64_t xs[] = {1, 3, 4, 8};
    TensorView x(xs, DataType::f16, x_h.data(), TensorLayout::NHWC);

    PixelShuffleAttributes attrs;
    attrs.block_size = 2;
    auto op = PixelShuffle::create(attrs, Backend::CPU);
    const TensorDesc in_desc[] = {x.desc()};
    const auto out_desc = op->getOutputTensorDesc(in_desc);
    NNOPS_EXPECT_EQ(out_desc[0].dims[1], int64_t{6});
    NNOPS_EXPECT_EQ(out_desc[0].dims[2], int64_t{8});
    NNOPS_EXPECT_EQ(out_desc[0].dims[3], int64_t{2});

    std::vector<nnops::backend::cpu::half> out_h(static_cast<size_t>(out_desc[0].numel()));
    TensorView out(std::span<const int64_t>(out_desc[0].dims.data(), 4), DataType::f16,
                   out_h.data(), TensorLayout::NHWC);
    const TensorView ins[] = {x};
    NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::ok);

    // DCR: out[n, h·2+bh, w·2+bw, c] = x[n, h, w, c·4 + bh·2 + bw].
    const auto* xp = x_h.data();
    const auto* yp = out_h.data();
    for (int64_t n = 0; n < 1; ++n) {
        for (int64_t h = 0; h < 3; ++h) {
            for (int64_t w = 0; w < 4; ++w) {
                for (int64_t c = 0; c < 2; ++c) {
                    for (int64_t bh = 0; bh < 2; ++bh) {
                        for (int64_t bw = 0; bw < 2; ++bw) {
                            const int64_t src =
                                ((n * 3 + h) * 4 + w) * 8 + c * 4 + bh * 2 + bw;
                            const int64_t dst =
                                ((n * 6 + h * 2 + bh) * 8 + (w * 2 + bw)) * 2 + c;
                            NNOPS_EXPECT_NEAR(simd::s_load(&yp[dst]), simd::s_load(&xp[src]),
                                              0.0f);
                        }
                    }
                }
            }
        }
    }
}

NNOPS_TEST(pixel_shuffle_validation) {
    const int64_t xs[] = {1, 2, 2, 4};
    std::vector<float> buf(16, 0.0f);
    TensorView x(xs, DataType::f32, buf.data(), TensorLayout::NHWC);

    auto expect = [&](SpaceToDepthAttributes attrs, TensorDesc out, Status want) {
        auto op = SpaceToDepth::create(attrs, Backend::CPU);
        const TensorDesc in_desc[] = {x.desc()};
        const TensorDesc out_desc[] = {out};
        NNOPS_EXPECT_TRUE(op->validate(in_desc, out_desc) == want);
    };

    TensorDesc ok_out;
    ok_out.rank = 4;
    ok_out.dtype = DataType::f32;
    ok_out.layout = TensorLayout::NHWC;
    ok_out.dims.resize(4);
    ok_out.dims[0] = 1;
    ok_out.dims[1] = 1;
    ok_out.dims[2] = 1;
    ok_out.dims[3] = 16;

    SpaceToDepthAttributes good;
    good.block_size = 2;
    expect(good, ok_out, Status::ok);

    SpaceToDepthAttributes zero;
    zero.block_size = 0;
    expect(zero, ok_out, Status::invalid_argument);

    // H = 2 divides only by 1 and 2; block 3 does not fit.
    SpaceToDepthAttributes big;
    big.block_size = 3;
    expect(big, ok_out, Status::invalid_argument);

    // Wrong output width.
    TensorDesc bad = ok_out;
    bad.dims[3] = 8;
    expect(good, bad, Status::invalid_argument);

    // Dense activations have no spatial route.
    TensorView xd(xs, DataType::f32, buf.data(), TensorLayout::Dense);
    {
        auto op = SpaceToDepth::create(good, Backend::CPU);
        const TensorDesc in_desc[] = {xd.desc()};
        const TensorDesc out_desc[] = {ok_out};
        NNOPS_EXPECT_TRUE(op->validate(in_desc, out_desc) == Status::unsupported);
    }
    // Rank 3 is not a spatial activation here.
    {
        const int64_t r3[] = {2, 2, 4};
        TensorView x3(r3, DataType::f32, buf.data(), TensorLayout::NHWC);
        auto op = SpaceToDepth::create(good, Backend::CPU);
        const TensorDesc in_desc[] = {x3.desc()};
        const TensorDesc out_desc[] = {ok_out};
        NNOPS_EXPECT_TRUE(op->validate(in_desc, out_desc) == Status::invalid_argument);
    }

    // PixelShuffle: the input channels must divide by b².
    const int64_t ps_shape[] = {1, 2, 2, 3};
    std::vector<float> ps_buf(12, 0.0f);
    TensorView ps_x(ps_shape, DataType::f32, ps_buf.data(), TensorLayout::NHWC);
    {
        PixelShuffleAttributes a;
        a.block_size = 2;
        auto op = PixelShuffle::create(a, Backend::CPU);
        const TensorDesc in_desc[] = {ps_x.desc()};
        TensorDesc y;
        y.rank = 4;
        y.dtype = DataType::f32;
        y.layout = TensorLayout::NHWC;
        y.dims.resize(4);
        y.dims[0] = 1;
        y.dims[1] = 4;
        y.dims[2] = 4;
        y.dims[3] = 0;
        const TensorDesc out_desc[] = {y};
        NNOPS_EXPECT_TRUE(op->validate(in_desc, out_desc) == Status::invalid_argument);
    }
}
