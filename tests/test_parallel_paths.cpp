/// @file test_parallel_paths.cpp
/// @brief Every generic CPU operator that splits its work across the pool must
///        produce exactly what the single-threaded route produces.
///
/// The unit tests run with no CpuBackend hook, so they never execute a task-grid
/// body more than once. This file drives the same operators twice — serial, then
/// on four threads — and compares the results bit for bit. That is the only
/// thing standing between a tiling bug (a run copied twice, a row never copied,
/// a stencil's table slice off by one) and a passing suite.
///
/// The ops here are the ones whose dispatch changed to a block copy or a pool:
/// concat, split, tile/expand, pad, gather, scatter, flatten, the broadcast
/// element-wise family, resize, grid sampling, BatchNorm, ArgMax and the
/// per-token DequantizeLinear.

#include "nnops/ops/argminmax.hpp"
#include "nnops/ops/compare.hpp"
#include "nnops/ops/concat.hpp"
#include "nnops/ops/eltwise.hpp"
#include "nnops/ops/flatten.hpp"
#include "nnops/ops/gather.hpp"
#include "nnops/ops/grid_sample.hpp"
#include "nnops/ops/norm.hpp"
#include "nnops/ops/pad.hpp"
#include "nnops/ops/quant_linear.hpp"
#include "nnops/ops/resize.hpp"
#include "nnops/ops/rope.hpp"
#include "nnops/ops/scatter.hpp"
#include "nnops/ops/split.hpp"
#include "nnops/ops/tile.hpp"
#include "nnops/ops/where.hpp"
#include "common/test_harness.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <thread>
#include <vector>

using namespace nnops;

namespace {

/// A four-worker pool over the operator's own hooks.
ComputeContext par_ctx(int nthreads = 4) {
    ComputeContext ctx;
    auto slots = std::make_shared<int>(nthreads);
    ctx.cpu.num_threads = [slots]() { return *slots; };
    ctx.cpu.parallel_for = [slots](int64_t begin, int64_t end, const ParallelForBody& body) {
        const int nt = *slots;
        const int64_t n = end - begin;
        if (n <= 1 || nt <= 1) {
            for (int64_t i = begin; i < end; ++i) {
                body(i);
            }
            return;
        }
        const int64_t per = (n + nt - 1) / nt;
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(nt - 1));
        for (int t = 1; t < nt; ++t) {
            const int64_t lo = begin + t * per;
            const int64_t hi = std::min(lo + per, end);
            if (lo >= hi) {
                break;
            }
            workers.emplace_back([&body, lo, hi] {
                for (int64_t i = lo; i < hi; ++i) {
                    body(i);
                }
            });
        }
        const int64_t hi0 = std::min(begin + per, end);
        for (int64_t i = begin; i < hi0; ++i) {
            body(i);
        }
        for (auto& w : workers) {
            w.join();
        }
    };
    return ctx;
}

std::vector<float> make_data(int64_t n, uint32_t seed) {
    std::vector<float> v(static_cast<size_t>(n));
    uint32_t s = seed;
    for (int64_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        v[static_cast<size_t>(i)] = (static_cast<float>(s >> 9) * (1.0f / 8388608.0f)) - 1.0f;
    }
    return v;
}

bool same_f32(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

TensorView nhwc(std::vector<float>& buf, int64_t N, int64_t H, int64_t W, int64_t C) {
    const int64_t shape[] = {N, H, W, C};
    return TensorView(std::span<const int64_t>(shape, 4), DataType::f32, buf.data(),
                      TensorLayout::NHWC);
}

} // namespace

NNOPS_TEST(parallel_concat_matches_serial) {
    constexpr int64_t N = 1, H = 20, W = 24, C = 40;
    std::vector<float> a = make_data(N * H * W * C, 1);
    std::vector<float> b = make_data(N * H * W * C, 2);
    std::vector<float> s(static_cast<size_t>(2 * N * H * W * C), -7.0f);
    std::vector<float> p(s.size(), -7.0f);

    const int64_t oshape[] = {N, H, W, 2 * C};
    ConcatAttributes attrs;
    attrs.axis = -1;
    auto op = Concat::create(attrs, Backend::CPU);

    {
        TensorView va = nhwc(a, N, H, W, C);
        TensorView vb = nhwc(b, N, H, W, C);
        TensorView vo(std::span<const int64_t>(oshape, 4), DataType::f32, s.data(),
                      TensorLayout::NHWC);
        const TensorView ins[] = {va, vb};
        op->compute(vo, ins);
    }
    {
        TensorView va = nhwc(a, N, H, W, C);
        TensorView vb = nhwc(b, N, H, W, C);
        TensorView vo(std::span<const int64_t>(oshape, 4), DataType::f32, p.data(),
                      TensorLayout::NHWC);
        const TensorView ins[] = {va, vb};
        op->compute(vo, ins, par_ctx());
    }
    NNOPS_EXPECT_TRUE(same_f32(s, p));
}

NNOPS_TEST(parallel_split_matches_serial) {
    constexpr int64_t N = 1, H = 20, W = 24, C = 48;
    std::vector<float> a = make_data(N * H * W * C, 3);

    std::vector<float> s(static_cast<size_t>(N * H * W * C), -7.0f);
    std::vector<float> p(s.size(), -7.0f);

    const int64_t sshape[] = {N, H, W, C / 3};
    SplitAttributes attrs;
    attrs.axis = -1;
    attrs.split = {C / 3, C / 3, C / 3};
    auto op = Split::create(attrs, Backend::CPU);

    const auto run = [&](std::vector<float>& dst, const ComputeContext& ctx) {
        TensorView va = nhwc(a, N, H, W, C);
        const int64_t part = N * H * W * (C / 3);
        TensorView v0(std::span<const int64_t>(sshape, 4), DataType::f32, dst.data(),
                      TensorLayout::NHWC);
        TensorView v1(std::span<const int64_t>(sshape, 4), DataType::f32, dst.data() + part,
                      TensorLayout::NHWC);
        TensorView v2(std::span<const int64_t>(sshape, 4), DataType::f32, dst.data() + 2 * part,
                      TensorLayout::NHWC);
        const TensorView ins[] = {va};
        TensorView outs[] = {v0, v1, v2};
        op->compute(outs, ins, ctx);
    };
    run(s, ComputeContext{});
    run(p, par_ctx());
    NNOPS_EXPECT_TRUE(same_f32(s, p));
}

NNOPS_TEST(parallel_expand_and_pad_match_serial) {
    // expand: [1,H,1,C] -> [1,H,S,C], the gqa repeat
    constexpr int64_t N = 1, Hh = 3, S = 40, C = 24;
    std::vector<float> a = make_data(N * Hh * C, 4);
    const int64_t ishape[] = {N, Hh, 1, C};
    const int64_t oshape[] = {N, Hh, S, C};
    {
        std::vector<float> s(static_cast<size_t>(N * Hh * S * C), -7.0f);
        std::vector<float> p(s.size(), -7.0f);
        ExpandAttributes attrs;
        auto op = Expand::create(attrs, Backend::CPU);
        const auto run = [&](std::vector<float>& dst, const ComputeContext& ctx) {
            TensorView va(std::span<const int64_t>(ishape, 4), DataType::f32, a.data());
            TensorView vo(std::span<const int64_t>(oshape, 4), DataType::f32, dst.data());
            const TensorView ins[] = {va};
            op->compute(vo, ins, ctx);
        };
        run(s, ComputeContext{});
        run(p, par_ctx());
        NNOPS_EXPECT_TRUE(same_f32(s, p));
    }

    // pad: constant 1 pixel on H and W of an NHWC map
    constexpr int64_t H = 11, W = 13, Cc = 24;
    std::vector<float> x = make_data(N * H * W * Cc, 5);
    const int64_t pshape[] = {N, H + 2, W + 2, Cc};
    std::vector<float> s2(static_cast<size_t>(N * (H + 2) * (W + 2) * Cc), -7.0f);
    std::vector<float> p2(s2.size(), -7.0f);
    PadAttributes attrs;
    attrs.pads = {0, 1, 1, 0, 0, 1, 1, 0};
    attrs.mode = PadAttributes::Mode::Constant;
    attrs.constant_value = 0.25f;
    auto op = Pad::create(attrs, Backend::CPU);
    const auto run = [&](std::vector<float>& dst, const ComputeContext& ctx) {
        TensorView vx = nhwc(x, N, H, W, Cc);
        TensorView vo(std::span<const int64_t>(pshape, 4), DataType::f32, dst.data(),
                      TensorLayout::NHWC);
        const TensorView ins[] = {vx};
        op->compute(vo, ins, ctx);
    };
    run(s2, ComputeContext{});
    run(p2, par_ctx());
    NNOPS_EXPECT_TRUE(same_f32(s2, p2));
}

NNOPS_TEST(parallel_gather_scatter_flatten_match_serial) {
    // gather: an embedding table
    constexpr int64_t V = 512, D = 96, T = 40;
    std::vector<float> w = make_data(V * D, 6);
    std::vector<int32_t> idx(static_cast<size_t>(T));
    for (int64_t i = 0; i < T; ++i) {
        idx[static_cast<size_t>(i)] = static_cast<int32_t>((i * 977) % V) - 0;
    }
    idx[1] = -3; // a wrapped negative index
    {
        const int64_t wshape[] = {V, D};
        const int64_t ishape[] = {T};
        const int64_t oshape[] = {T, D};
        std::vector<float> s(static_cast<size_t>(T * D), -7.0f);
        std::vector<float> p(s.size(), -7.0f);
        GatherAttributes attrs;
        attrs.axis = 0;
        auto op = Gather::create(attrs, Backend::CPU);
        const auto run = [&](std::vector<float>& dst, const ComputeContext& ctx) {
            TensorView vw(std::span<const int64_t>(wshape, 2), DataType::f32, w.data());
            TensorView vi(std::span<const int64_t>(ishape, 1), DataType::s32, idx.data());
            TensorView vo(std::span<const int64_t>(oshape, 2), DataType::f32, dst.data());
            const TensorView ins[] = {vw, vi};
            op->compute(vo, ins, ctx);
        };
        run(s, ComputeContext{});
        run(p, par_ctx());
        NNOPS_EXPECT_TRUE(same_f32(s, p));
    }

    // scatter_nd: rows of a table
    constexpr int64_t R = 300, Dr = 64, TT = 48;
    std::vector<float> data = make_data(R * Dr, 7);
    std::vector<float> upd = make_data(TT * Dr, 8);
    std::vector<int64_t> sidx(static_cast<size_t>(TT));
    for (int64_t i = 0; i < TT; ++i) {
        sidx[static_cast<size_t>(i)] = (i * 37) % R;
    }
    {
        const int64_t dshape[] = {R, Dr};
        const int64_t ishape[] = {TT, 1};
        const int64_t ushape[] = {TT, Dr};
        std::vector<float> s(static_cast<size_t>(R * Dr), -7.0f);
        std::vector<float> p(s.size(), -7.0f);
        ScatterNDAttributes attrs;
        auto op = ScatterND::create(attrs, Backend::CPU);
        const auto run = [&](std::vector<float>& dst, const ComputeContext& ctx) {
            TensorView vd(std::span<const int64_t>(dshape, 2), DataType::f32, data.data());
            TensorView vi(std::span<const int64_t>(ishape, 2), DataType::s64, sidx.data());
            TensorView vu(std::span<const int64_t>(ushape, 2), DataType::f32, upd.data());
            TensorView vo(std::span<const int64_t>(dshape, 2), DataType::f32, dst.data());
            const TensorView ins[] = {vd, vi, vu};
            op->compute(vo, ins, ctx);
        };
        run(s, ComputeContext{});
        run(p, par_ctx());
        NNOPS_EXPECT_TRUE(same_f32(s, p));
    }

    // flatten: NHWC read as dense
    constexpr int64_t N = 1, H = 17, W = 19, Cc = 24;
    std::vector<float> x = make_data(N * H * W * Cc, 9);
    {
        const int64_t oshape[] = {N, H, W, Cc};
        std::vector<float> s(static_cast<size_t>(N * H * W * Cc), -7.0f);
        std::vector<float> p(s.size(), -7.0f);
        FlattenAttributes attrs;
        auto op = Flatten::create(attrs, Backend::CPU);
        const auto run = [&](std::vector<float>& dst, const ComputeContext& ctx) {
            TensorView vx = nhwc(x, N, H, W, Cc);
            TensorView vo(std::span<const int64_t>(oshape, 4), DataType::f32, dst.data());
            const TensorView ins[] = {vx};
            op->compute(vo, ins, ctx);
        };
        run(s, ComputeContext{});
        run(p, par_ctx());
        NNOPS_EXPECT_TRUE(same_f32(s, p));
    }
}

NNOPS_TEST(parallel_broadcast_ops_match_serial) {
    constexpr int64_t M = 300, Nn = 40;

    std::vector<float> a = make_data(M * Nn, 10);
    std::vector<float> b = make_data(Nn, 11);
    const int64_t ashape[] = {M, Nn};
    const int64_t bshape[] = {Nn};

    {
        std::vector<float> s(static_cast<size_t>(M * Nn), -7.0f);
        std::vector<float> p(s.size(), -7.0f);
        EltwiseAttributes attrs;
        attrs.type = EltwiseType::Add;
        auto op = Eltwise::create(attrs, Backend::CPU);
        const auto run = [&](std::vector<float>& dst, const ComputeContext& ctx) {
            TensorView va(std::span<const int64_t>(ashape, 2), DataType::f32, a.data());
            TensorView vb(std::span<const int64_t>(bshape, 1), DataType::f32, b.data());
            TensorView vo(std::span<const int64_t>(ashape, 2), DataType::f32, dst.data());
            const TensorView ins[] = {va, vb};
            op->compute(vo, ins, ctx);
        };
        run(s, ComputeContext{});
        run(p, par_ctx());
        NNOPS_EXPECT_TRUE(same_f32(s, p));
    }

    // where with a broadcast mask
    {
        std::vector<float> y = make_data(M * Nn, 12);
        std::vector<uint8_t> mask(static_cast<size_t>(Nn));
        for (int64_t i = 0; i < Nn; ++i) {
            mask[static_cast<size_t>(i)] = static_cast<uint8_t>((i % 3) == 0 ? 0 : 1);
        }
        const int64_t mshape[] = {1, Nn};
        std::vector<float> s(static_cast<size_t>(M * Nn), -7.0f);
        std::vector<float> p(s.size(), -7.0f);
        WhereAttributes attrs;
        auto op = Where::create(attrs, Backend::CPU);
        const auto run = [&](std::vector<float>& dst, const ComputeContext& ctx) {
            TensorView vm(std::span<const int64_t>(mshape, 2), DataType::u8, mask.data());
            TensorView vx(std::span<const int64_t>(ashape, 2), DataType::f32, a.data());
            TensorView vy(std::span<const int64_t>(ashape, 2), DataType::f32, y.data());
            TensorView vo(std::span<const int64_t>(ashape, 2), DataType::f32, dst.data());
            const TensorView ins[] = {vm, vx, vy};
            op->compute(vo, ins, ctx);
        };
        run(s, ComputeContext{});
        run(p, par_ctx());
        NNOPS_EXPECT_TRUE(same_f32(s, p));
    }

    // compare → u8
    {
        const int64_t bshape2[] = {1, Nn};
        std::vector<uint8_t> s(static_cast<size_t>(M * Nn), 9);
        std::vector<uint8_t> p(s.size(), 9);
        CompareAttributes attrs;
        attrs.type = CompareType::Greater;
        auto op = Compare::create(attrs, Backend::CPU);
        const auto run = [&](std::vector<uint8_t>& dst, const ComputeContext& ctx) {
            TensorView va(std::span<const int64_t>(ashape, 2), DataType::f32, a.data());
            TensorView vb(std::span<const int64_t>(bshape2, 2), DataType::f32, b.data());
            TensorView vo(std::span<const int64_t>(ashape, 2), DataType::u8, dst.data());
            const TensorView ins[] = {va, vb};
            op->compute(vo, ins, ctx);
        };
        run(s, ComputeContext{});
        run(p, par_ctx());
        NNOPS_EXPECT_TRUE(std::memcmp(s.data(), p.data(), s.size()) == 0);
    }
}

NNOPS_TEST(parallel_spatial_ops_match_serial) {
    // bilinear resize x2 on a small map
    constexpr int64_t N = 1, H = 21, W = 23, C = 24;
    std::vector<float> a = make_data(N * H * W * C, 13);
    {
        const int64_t oshape[] = {N, 2 * H, 2 * W, C};
        std::vector<float> s(static_cast<size_t>(N * 4 * H * W * C), -7.0f);
        std::vector<float> p(s.size(), -7.0f);
        ResizeAttributes attrs;
        attrs.mode = ResizeMode::Linear;
        attrs.coord_mode = CoordinateTransformMode::HalfPixel;
        auto op = Resize::create(attrs, Backend::CPU);
        const auto run = [&](std::vector<float>& dst, const ComputeContext& ctx) {
            TensorView va = nhwc(a, N, H, W, C);
            TensorView vo(std::span<const int64_t>(oshape, 4), DataType::f32, dst.data(),
                          TensorLayout::NHWC);
            const TensorView ins[] = {va};
            op->compute(vo, ins, ctx);
        };
        run(s, ComputeContext{});
        run(p, par_ctx());
        NNOPS_EXPECT_TRUE(same_f32(s, p));
    }

    // grid sample, bilinear, zeros padding
    {
        constexpr int64_t OH = 15, OW = 16;
        std::vector<float> grid = make_data(static_cast<int64_t>(N) * OH * OW * 2, 14);
        for (float& g : grid) {
            g *= 1.3f; // push some samples out of bounds
        }
        const int64_t gshape[] = {N, OH, OW, 2};
        const int64_t oshape[] = {N, OH, OW, C};
        std::vector<float> s(static_cast<size_t>(N * OH * OW * C), -7.0f);
        std::vector<float> p(s.size(), -7.0f);
        GridSampleAttributes attrs;
        attrs.mode = GridSampleMode::Bilinear;
        attrs.padding_mode = GridSamplePaddingMode::Zeros;
        attrs.align_corners = false;
        auto op = GridSample::create(attrs, Backend::CPU);
        const auto run = [&](std::vector<float>& dst, const ComputeContext& ctx) {
            TensorView va = nhwc(a, N, H, W, C);
            TensorView vg(std::span<const int64_t>(gshape, 4), DataType::f32, grid.data());
            TensorView vo(std::span<const int64_t>(oshape, 4), DataType::f32, dst.data(),
                          TensorLayout::NHWC);
            const TensorView ins[] = {va, vg};
            op->compute(vo, ins, ctx);
        };
        run(s, ComputeContext{});
        run(p, par_ctx());
        NNOPS_EXPECT_TRUE(same_f32(s, p));
    }

    // spatial BatchNorm
    {
        std::vector<float> sc(C), bi(C), me(C), va(C);
        for (int64_t c = 0; c < C; ++c) {
            const auto i = static_cast<size_t>(c);
            sc[i] = 0.5f + 0.01f * static_cast<float>(c);
            bi[i] = 0.05f * static_cast<float>(c);
            me[i] = -0.3f + 0.01f * static_cast<float>(c);
            va[i] = 1.0f + 0.02f * static_cast<float>(c);
        }
        const int64_t cshape[] = {C};
        std::vector<float> s(static_cast<size_t>(N * H * W * C), -7.0f);
        std::vector<float> p(s.size(), -7.0f);
        NormAttributes attrs;
        attrs.type = NormType::BatchNorm;
        attrs.spatial = true;
        auto op = Norm::create(attrs, Backend::CPU);
        const auto run = [&](std::vector<float>& dst, const ComputeContext& ctx) {
            TensorView vx = nhwc(a, N, H, W, C);
            TensorView vo(std::span<const int64_t>(vx.shape_span().data(), 4), DataType::f32,
                          dst.data(), TensorLayout::NHWC);
            TensorView vs(std::span<const int64_t>(cshape, 1), DataType::f32, sc.data());
            TensorView vb(std::span<const int64_t>(cshape, 1), DataType::f32, bi.data());
            TensorView vm(std::span<const int64_t>(cshape, 1), DataType::f32, me.data());
            TensorView vv(std::span<const int64_t>(cshape, 1), DataType::f32, va.data());
            const TensorView ins[] = {vx, vs, vb, vm, vv};
            op->compute(vo, ins, ctx);
        };
        run(s, ComputeContext{});
        run(p, par_ctx());
        NNOPS_EXPECT_TRUE(same_f32(s, p));
    }
}

NNOPS_TEST(parallel_rope_matches_serial) {
    // The table build and the rotation are both row-parallel; a table row or a
    // row tile landing on the wrong position would still be "a valid rotation".
    constexpr int64_t Bb = 2, Hh = 3, S = 200, D = 32;
    std::vector<float> a = make_data(Bb * Hh * S * D, 16);
    const int64_t shape[] = {Bb, Hh, S, D};
    std::vector<float> s(a.size(), -7.0f);
    std::vector<float> p(a.size(), -7.0f);

    for (bool interleaved : {true, false}) {
        RoPEAttributes attrs;
        attrs.rotary_dim = D;
        attrs.interleaved = interleaved;
        attrs.base = 10000.0f;
        auto op = RoPE::create(attrs, Backend::CPU);
        const auto run = [&](std::vector<float>& dst, const ComputeContext& ctx) {
            TensorView va(std::span<const int64_t>(shape, 4), DataType::f32, a.data());
            TensorView vo(std::span<const int64_t>(shape, 4), DataType::f32, dst.data());
            const TensorView ins[] = {va};
            op->compute(vo, ins, ctx);
        };
        run(s, ComputeContext{});
        run(p, par_ctx());
        NNOPS_EXPECT_TRUE(same_f32(s, p));
    }
}

NNOPS_TEST(parallel_reductions_match_serial) {
    // argmax over many inner positions (a channel-last feature map)
    constexpr int64_t N = 2, Hh = 9, Ww = 10, C = 64;
    std::vector<float> a = make_data(N * Hh * Ww * C, 15);
    {
        const int64_t ashape[] = {N, Hh, Ww, C};
        const int64_t oshape[] = {N, Hh, Ww};
        std::vector<int64_t> s(static_cast<size_t>(N * Hh * Ww), -7);
        std::vector<int64_t> p(s.size(), -7);
        ArgMinMaxAttributes attrs;
        attrs.type = ArgMinMaxType::Max;
        attrs.axis = -1;
        auto op = ArgMax::create(attrs, Backend::CPU);
        const auto run = [&](std::vector<int64_t>& dst, const ComputeContext& ctx) {
            TensorView va(std::span<const int64_t>(ashape, 4), DataType::f32, a.data());
            TensorView vo(std::span<const int64_t>(oshape, 3), DataType::s64, dst.data());
            const TensorView ins[] = {va};
            op->compute(vo, ins, ctx);
        };
        run(s, ComputeContext{});
        run(p, par_ctx());
        NNOPS_EXPECT_TRUE(std::memcmp(s.data(), p.data(), s.size() * sizeof(int64_t)) == 0);
    }

    // per-token dequantize
    constexpr int64_t M = 200, Nn = 48;
    {
        std::vector<int8_t> q(static_cast<size_t>(M * Nn));
        for (int64_t i = 0; i < M * Nn; ++i) {
            q[static_cast<size_t>(i)] = static_cast<int8_t>((i * 31) % 251 - 125);
        }
        std::vector<float> sc(static_cast<size_t>(M), 0.01f);
        std::vector<int8_t> zp(static_cast<size_t>(M), 3);
        const int64_t qshape[] = {M, Nn};
        const int64_t sshape[] = {M};
        std::vector<float> s(static_cast<size_t>(M * Nn), -7.0f);
        std::vector<float> p(s.size(), -7.0f);
        QuantLinearAttributes attrs;
        attrs.axis = -1;
        auto op = DequantizeLinear::create(attrs, Backend::CPU);
        const auto run = [&](std::vector<float>& dst, const ComputeContext& ctx) {
            TensorView vq(std::span<const int64_t>(qshape, 2), DataType::s8, q.data());
            TensorView vs(std::span<const int64_t>(sshape, 1), DataType::f32, sc.data());
            TensorView vz(std::span<const int64_t>(sshape, 1), DataType::s8, zp.data());
            TensorView vo(std::span<const int64_t>(qshape, 2), DataType::f32, dst.data());
            const TensorView ins[] = {vq, vs, vz};
            op->compute(vo, ins, ctx);
        };
        run(s, ComputeContext{});
        run(p, par_ctx());
        NNOPS_EXPECT_TRUE(same_f32(s, p));
    }
}
