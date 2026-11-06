/// @file test_depthwise_transpose_conv.cpp
/// @brief DepthwiseTransposeConv2D tests — channel-first and channel-last.
///
/// Channel-first: the raw [C, 1, KH, KW] filter goes straight to the operator,
/// and the output is compared with the scalar reference. Two cases are
/// hand-computed so the suite is not only reference against reference.
///
/// Channel-last: the same values in both orders — channel-first plus the raw
/// channel-first filter for the reference, channel-last plus TensorFlow's
/// [KH][KW][C] filter for the kernel. The route reads the filter as the graph
/// holds it, so the only plumbing is the transpose between the orders.
///
/// The cases sweep the branch structure rather than only the shapes: stride 1
/// takes the contiguous-store lane in both layouts, stride 2 the residue-class
/// split (channel-first stages the strided store through a lane buffer); a
/// 1x1 kernel leaves the interior window empty, dilation moves the taps; the
/// channel counts pick out sub-vector, partial, full and multi-block sweeps.
///
/// The threaded cases run the same call on a real pool and compare bytes:
/// every output element is accumulated by exactly one task in a fixed tap
/// order, so the thread count may not change the result.

#include "nnops/ops/transpose_conv2d.hpp"
#include "nnops/ops/depthwise_transpose_conv.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/detail/half.hpp"
#include "common/test_harness.hpp"
#include "common/cpu_pool.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/channel_last.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace nnops;
using nnops::backend::cpu::half;

// The reference kernel is defined in tests/reference/.
namespace nnops::backend::cpu::reference {
    void depthwise_transpose_conv_ref(const DepthwiseTransposeConv2DAttributes& attrs,
                                      TensorView& output,
                                      std::span<const TensorView> inputs,
                                      const ComputeContext& ctx,
                                      void* workspace);
}

// ============================================================
// Channel-first
// ============================================================

static void test_channel_first_vs_ref(const std::vector<int64_t>& in_shape,
                                      const std::vector<int64_t>& w_shape,
                                      const DepthwiseTransposeConv2DAttributes& attrs,
                                      bool has_bias = false,
                                      float tol = 1e-4f)
{
    auto [in_vec, in] = test::make_random_tensor(in_shape);
    auto [w_vec,  w]  = test::make_random_tensor(w_shape);
    std::vector<float> b_vec;
    TensorView b;
    if (has_bias) {
        auto p = test::make_random_tensor({w_shape[0]});
        b_vec = std::move(p.first);
        b = p.second;
    }

    auto op = DepthwiseTransposeConv2D::create(attrs, Backend::CPU);

    std::vector<TensorDesc> descs = {in.desc(), w.desc()};
    if (has_bias) { descs.push_back(b.desc()); }
    auto out_descs = op->getOutputTensorDesc(descs);

    std::vector<float> out_buf(static_cast<size_t>(out_descs[0].numel()),
                               attrs.add_to ? 1.0f : 0.0f);
    std::vector<float> ref_buf(out_buf.size(), attrs.add_to ? 1.0f : 0.0f);
    auto out = test::make_channel_first(out_descs[0], out_buf.data());
    auto ref = test::make_channel_first(out_descs[0], ref_buf.data());

    {
        ComputeContext ctx;
        std::vector<TensorView> ref_ins = {in, w};
        if (has_bias) { ref_ins.push_back(b); }
        backend::cpu::reference::depthwise_transpose_conv_ref(
            attrs, ref, ref_ins, ctx, nullptr);
    }

    std::vector<TensorView> ins = {in, w};
    if (has_bias) { ins.push_back(b); }
    NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));
    NNOPS_EXPECT_TRUE(test::allclose(out, ref, tol, tol));
}

/// The channel-first route in f16. The f32 reference runs on the same values
/// widened back from f16, so the only error left is the kernel's own f16
/// arithmetic.
static void test_channel_first_f16_vs_ref(const std::vector<int64_t>& in_shape,
                                          const std::vector<int64_t>& w_shape,
                                          const DepthwiseTransposeConv2DAttributes& attrs,
                                          bool has_bias = false)
{
    auto [in_vec, in_f32] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_f32]  = test::make_random_tensor(w_shape);
    std::vector<float> b_vec;
    TensorView b_f32;
    if (has_bias) {
        auto p = test::make_random_tensor({w_shape[0]});
        b_vec = std::move(p.first);
        b_f32 = p.second;
    }

    // Round in place so the reference and the kernel see the same values.
    std::vector<half> in_f16(in_vec.size()), w_f16(w_vec.size()), b_f16(b_vec.size());
    for (size_t i = 0; i < in_f16.size(); ++i) {
        simd::s_store(&in_f16[i], in_vec[i]);
        in_vec[i] = simd::s_load(&in_f16[i]);
    }
    for (size_t i = 0; i < w_f16.size(); ++i) {
        simd::s_store(&w_f16[i], w_vec[i]);
        w_vec[i] = simd::s_load(&w_f16[i]);
    }
    for (size_t i = 0; i < b_f16.size(); ++i) {
        simd::s_store(&b_f16[i], b_vec[i]);
        b_vec[i] = simd::s_load(&b_f16[i]);
    }

    TensorView in(std::span<const int64_t>(in_shape), DataType::f16, in_f16.data());
    TensorView w(std::span<const int64_t>(w_shape), DataType::f16, w_f16.data());
    const int64_t b_shape[] = {w_shape[0]};
    TensorView b(std::span<const int64_t>(b_shape, 1), DataType::f16, b_f16.data());

    auto op = DepthwiseTransposeConv2D::create(attrs, Backend::CPU);
    const TensorDesc in_d[] = {in.desc(), w.desc(), b.desc()};
    auto out_descs = op->getOutputTensorDesc(
        std::span<const TensorDesc>(in_d, has_bias ? 3 : 2));

    std::vector<float> ref_buf(static_cast<size_t>(out_descs[0].numel()),
                               attrs.add_to ? 1.0f : 0.0f);
    auto ref = test::make_channel_first(out_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        const TensorView r[] = {in_f32, w_f32, b_f32};
        backend::cpu::reference::depthwise_transpose_conv_ref(
            attrs, ref, std::span<const TensorView>(r, has_bias ? 3 : 2), ctx, nullptr);
    }

    std::vector<half> out_buf(static_cast<size_t>(out_descs[0].numel()));
    auto out = test::make_channel_first(out_descs[0], out_buf.data());
    for (auto& v : out_buf) { simd::s_store(&v, attrs.add_to ? 1.0f : 0.0f); }

    const TensorView ins[] = {in, w, b};
    NNOPS_EXPECT_TRUE(
        is_ok(op->compute(out, std::span<const TensorView>(ins, has_bias ? 3 : 2))));

    const auto got = test::f16_to_f32(out_buf);
    for (size_t i = 0; i < got.size(); ++i) {
        NNOPS_EXPECT_NEAR(got[i], ref_buf[i], test::kF16AccumTol);
    }
}

// ============================================================
// Channel-last
// ============================================================

template <class Shape>
static int64_t dwt_spatial_elems(const Shape& shape)
{
    int64_t s = 1;
    for (size_t i = 2; i < shape.size(); ++i) { s *= shape[i]; }
    return s;
}

template <typename T>
static void test_channel_last_vs_ref(const std::vector<int64_t>& in_shape,
                                     const std::vector<int64_t>& w_shape,
                                     const DepthwiseTransposeConv2DAttributes& attrs,
                                     bool has_bias = false,
                                     float tol = 1e-4f)
{
    constexpr DataType dt = std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
    const auto to_T = [](float v) -> T {
        if constexpr (std::is_same_v<T, float>) {
            return v;
        } else {
            T h;
            simd::s_store(&h, v);
            return h;
        }
    };
    const auto as_f = [](T v) -> float {
        if constexpr (std::is_same_v<T, float>) { return v; }
        else { return simd::s_load(&v); }
    };

    const int64_t N   = in_shape[0];
    const int64_t C   = in_shape[1];
    const int64_t inS = dwt_spatial_elems(in_shape);
    const int64_t taps = w_shape[2] * w_shape[3];

    auto [in_vec, in_channel_first] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_channel_first]  = test::make_random_tensor(w_shape);
    std::vector<float> b_vec;
    TensorView b_channel_first;
    if (has_bias) {
        auto p = test::make_random_tensor({C});
        b_vec = std::move(p.first);
        b_channel_first = p.second;
    }

    auto op = DepthwiseTransposeConv2D::create(attrs, Backend::CPU);

    // ---- Reference, in the channel-first order ---------------------------------
    const TensorDesc ref_in[] = {
        in_channel_first.desc(), w_channel_first.desc(), b_channel_first.desc()};
    auto ref_descs = op->getOutputTensorDesc(
        std::span<const TensorDesc>(ref_in, has_bias ? 3 : 2));
    const int64_t outS  = dwt_spatial_elems(ref_descs[0].dims);
    const int64_t out_n = N * C * outS;

    std::vector<float> ref_buf(static_cast<size_t>(out_n),
                               attrs.add_to ? 1.0f : 0.0f);
    auto ref = test::make_channel_first(ref_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        const TensorView r[] = {in_channel_first, w_channel_first, b_channel_first};
        backend::cpu::reference::depthwise_transpose_conv_ref(
            attrs, ref, std::span<const TensorView>(r, has_bias ? 3 : 2), ctx, nullptr);
    }

    // ---- The same values, in the channel-last order ----------------------------
    const auto in_cl_f = test::to_channel_last<float>(in_vec, N, C, inS);
    const auto w_cl_f  = test::depthwise_to_hwcm<float>(w_vec, C, taps);
    std::vector<T> in_cl(in_cl_f.size());
    std::vector<T> w_cl(w_cl_f.size());
    for (size_t i = 0; i < in_cl.size(); ++i) { in_cl[i] = to_T(in_cl_f[i]); }
    for (size_t i = 0; i < w_cl.size(); ++i)  { w_cl[i]  = to_T(w_cl_f[i]); }

    TensorView in_v(std::span<const int64_t>(in_shape), dt, in_cl.data(),
                    TensorLayout::NHWC);
    const int64_t wsh[] = {w_shape[2], w_shape[3], C};
    TensorView w_v(std::span<const int64_t>(wsh, 3), dt, w_cl.data(),
                   TensorLayout::HWCM);

    std::vector<T> b_cl;
    TensorView b_v;
    if (has_bias) {
        b_cl.resize(static_cast<size_t>(C));
        for (int64_t c = 0; c < C; ++c) {
            b_cl[static_cast<size_t>(c)] = to_T(b_vec[static_cast<size_t>(c)]);
        }
        const int64_t bsh[] = {C};
        b_v = TensorView(std::span<const int64_t>(bsh, 1), dt, b_cl.data());
    }

    const TensorDesc cl_in[] = {in_v.desc(), w_v.desc(), b_v.desc()};
    auto out_descs = op->getOutputTensorDesc(
        std::span<const TensorDesc>(cl_in, has_bias ? 3 : 2));
    NNOPS_EXPECT_TRUE(out_descs[0].layout == TensorLayout::NHWC);
    NNOPS_EXPECT_EQ(out_descs[0].numel(), out_n);

    std::vector<T> out_buf(static_cast<size_t>(out_n),
                           to_T(attrs.add_to ? 1.0f : 0.0f));
    TensorView out_v(std::span<const int64_t>(out_descs[0].dims.data(),
                                              static_cast<size_t>(out_descs[0].rank)),
                     dt, out_buf.data(), TensorLayout::NHWC);

    std::vector<TensorView> ins = {in_v, w_v};
    if (has_bias) { ins.push_back(b_v); }
    NNOPS_EXPECT_TRUE(is_ok(op->compute(out_v, ins)));

    // ---- Back to the channel-first order for the comparison --------------------
    const auto got = test::to_channel_first<T>(out_buf, N, C, outS);
    for (int64_t i = 0; i < out_n; ++i) {
        NNOPS_EXPECT_NEAR(as_f(got[static_cast<size_t>(i)]),
                          ref_buf[static_cast<size_t>(i)], tol);
    }
}

// ============================================================
// Hand-computed cases
// ============================================================

NNOPS_TEST(dwtconv_hand_exact) {
    // 1x1 plane, 2x2 filter, stride 1: each output pixel is the input pixel
    // times the matching tap, so the output *is* the filter.
    const int64_t ishape[] = {1, 2, 1, 1};
    const int64_t wshape[] = {2, 1, 2, 2};
    std::vector<float> in_buf(2, 1.0f);
    std::vector<float> w_buf = {1, 2, 3, 4, 5, 6, 7, 8};

    TensorView in(ishape, DataType::f32, in_buf.data());
    TensorView w(wshape, DataType::f32, w_buf.data());

    DepthwiseTransposeConv2DAttributes attrs{{2, 2}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};
    auto op = DepthwiseTransposeConv2D::create(attrs, Backend::CPU);

    const TensorDesc in_desc[] = {in.desc(), w.desc()};
    auto out_descs = op->getOutputTensorDesc(in_desc);
    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 2);
    NNOPS_EXPECT_EQ(out_descs[0].dims[3], 2);

    std::vector<float> out_buf(static_cast<size_t>(out_descs[0].numel()));
    auto out = test::make_channel_first(out_descs[0], out_buf.data());
    const TensorView ins[] = {in, w};
    NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));

    const float expect[] = {1, 2, 3, 4, 5, 6, 7, 8};
    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], expect[i], 1e-6f);
    }
}

NNOPS_TEST(dwtconv_hand_exact_stride2) {
    // Two input rows, a 2x1 filter, stride 2: row ih lands at output rows
    // 2·ih + kh, so the output is [a·w0, a·w1, b·w0, b·w1].
    const int64_t ishape[] = {1, 1, 2, 1};
    const int64_t wshape[] = {1, 1, 2, 1};
    std::vector<float> in_buf = {1.0f, 2.0f};
    std::vector<float> w_buf  = {3.0f, 5.0f};

    TensorView in(ishape, DataType::f32, in_buf.data());
    TensorView w(wshape, DataType::f32, w_buf.data());

    DepthwiseTransposeConv2DAttributes attrs{{2, 1}, {2, 1}, {1, 1}, {0, 0}, {0, 0}};
    auto op = DepthwiseTransposeConv2D::create(attrs, Backend::CPU);

    const TensorDesc in_desc[] = {in.desc(), w.desc()};
    auto out_descs = op->getOutputTensorDesc(in_desc);
    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 4);

    std::vector<float> out_buf(static_cast<size_t>(out_descs[0].numel()));
    auto out = test::make_channel_first(out_descs[0], out_buf.data());
    const TensorView ins[] = {in, w};
    NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));

    const float expect[] = {3.0f, 5.0f, 6.0f, 10.0f};
    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], expect[i], 1e-6f);
    }
}

// ============================================================
// Channel-first table
// ============================================================

NNOPS_TEST(dwtconv_channel_first_vs_ref_basic) {
    // Odd output width 15 against an 8-lane vector: the class loop ends on a
    // partial tile and the row edge runs the scalar path.
    test_channel_first_vs_ref({1, 4, 12, 13}, {4, 1, 3, 3},
        DepthwiseTransposeConv2DAttributes{{3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
}

NNOPS_TEST(dwtconv_channel_first_vs_ref_stride2_padding) {
    // Both coordinates split into two residue classes; the store is strided.
    test_channel_first_vs_ref({1, 6, 9, 11}, {6, 1, 3, 3},
        DepthwiseTransposeConv2DAttributes{{3, 3}, {2, 2}, {1, 1}, {1, 1}, {0, 0}});
}

NNOPS_TEST(dwtconv_channel_first_vs_ref_output_padding) {
    test_channel_first_vs_ref({2, 5, 7, 9}, {5, 1, 3, 3},
        DepthwiseTransposeConv2DAttributes{{3, 3}, {2, 2}, {1, 1}, {0, 0}, {1, 1}});
}

NNOPS_TEST(dwtconv_channel_first_vs_ref_dilation) {
    test_channel_first_vs_ref({1, 4, 8, 10}, {4, 1, 3, 3},
        DepthwiseTransposeConv2DAttributes{{3, 3}, {1, 1}, {2, 2}, {0, 0}, {0, 0}});
}

NNOPS_TEST(dwtconv_channel_first_vs_ref_bias) {
    test_channel_first_vs_ref({2, 8, 8, 8}, {8, 1, 3, 3},
        DepthwiseTransposeConv2DAttributes{{3, 3}, {2, 2}, {1, 1}, {1, 1}, {0, 0}}, true);
}

NNOPS_TEST(dwtconv_channel_first_vs_ref_add_to) {
    DepthwiseTransposeConv2DAttributes attrs{{3, 3}, {1, 1}, {1, 1}, {1, 1}, {0, 0}};
    attrs.add_to = true;
    test_channel_first_vs_ref({1, 3, 8, 8}, {3, 1, 3, 3}, attrs);
}

NNOPS_TEST(dwtconv_channel_first_vs_ref_relu_epilogue) {
    DepthwiseTransposeConv2DAttributes attrs{{3, 3}, {2, 2}, {1, 1}, {1, 1}, {0, 0}};
    attrs.epilogue.type = EpilogueActivateType::Relu;
    test_channel_first_vs_ref({1, 5, 6, 11}, {5, 1, 3, 3}, attrs);
}

NNOPS_TEST(dwtconv_channel_first_vs_ref_1x1_stride2) {
    // A 1x1 kernel leaves the interior window empty: every class step is an
    // edge step, so the scalar tap loop carries the whole row.
    test_channel_first_vs_ref({1, 8, 9, 9}, {8, 1, 1, 1},
        DepthwiseTransposeConv2DAttributes{{1, 1}, {2, 2}, {1, 1}, {0, 0}, {0, 0}});
}

NNOPS_TEST(dwtconv_channel_first_f16_vs_ref) {
    test_channel_first_f16_vs_ref({1, 8, 10, 11}, {8, 1, 3, 3},
        DepthwiseTransposeConv2DAttributes{{3, 3}, {1, 1}, {1, 1}, {1, 1}, {0, 0}}, true);
    // Stride 2 and dilation on both axes; the f16 lanes stage through the
    // strided store path.
    test_channel_first_f16_vs_ref({2, 6, 7, 9}, {6, 1, 3, 3},
        DepthwiseTransposeConv2DAttributes{{3, 3}, {2, 2}, {2, 2}, {1, 1}, {0, 0}});
}

// ============================================================
// Channel-last table
// ============================================================

NNOPS_TEST(dwtconv_channel_last_vs_ref_basic) {
    test_channel_last_vs_ref<float>({2, 16, 10, 10}, {16, 1, 3, 3},
        DepthwiseTransposeConv2DAttributes{{3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
        false, 1e-4f);
}

NNOPS_TEST(dwtconv_channel_last_vs_ref_partial_block) {
    // Five channels: one vector chunk, a scalar tail.
    test_channel_last_vs_ref<float>({1, 5, 8, 9}, {5, 1, 3, 3},
        DepthwiseTransposeConv2DAttributes{{3, 3}, {2, 2}, {1, 1}, {1, 1}, {0, 0}},
        false, 1e-4f);
}

NNOPS_TEST(dwtconv_channel_last_vs_ref_multi_block) {
    // Seventeen channels cross the 16-wide chunk boundary and end on a
    // one-channel tail.
    test_channel_last_vs_ref<float>({1, 17, 7, 9}, {17, 1, 3, 3},
        DepthwiseTransposeConv2DAttributes{{3, 3}, {1, 1}, {1, 1}, {1, 1}, {0, 0}},
        true, 1e-4f);
}

NNOPS_TEST(dwtconv_channel_last_vs_ref_stride2_dilation) {
    test_channel_last_vs_ref<float>({1, 8, 9, 10}, {8, 1, 3, 3},
        DepthwiseTransposeConv2DAttributes{{3, 3}, {2, 2}, {2, 2}, {0, 0}, {0, 0}},
        false, 1e-4f);
}

NNOPS_TEST(dwtconv_channel_last_vs_ref_add_to_relu) {
    DepthwiseTransposeConv2DAttributes attrs{{3, 3}, {2, 2}, {1, 1}, {1, 1}, {0, 0}};
    attrs.add_to = true;
    attrs.epilogue.type = EpilogueActivateType::Relu;
    test_channel_last_vs_ref<float>({1, 6, 6, 11}, {6, 1, 3, 3}, attrs, false, 1e-4f);
}

NNOPS_TEST(dwtconv_channel_last_vs_ref_output_padding) {
    test_channel_last_vs_ref<float>({2, 4, 6, 8}, {4, 1, 3, 3},
        DepthwiseTransposeConv2DAttributes{{3, 3}, {2, 2}, {1, 1}, {0, 0}, {1, 1}},
        true, 1e-4f);
}

NNOPS_TEST(dwtconv_channel_last_vs_ref_1x1_stride2) {
    test_channel_last_vs_ref<float>({1, 8, 9, 9}, {8, 1, 1, 1},
        DepthwiseTransposeConv2DAttributes{{1, 1}, {2, 2}, {1, 1}, {0, 0}, {0, 0}},
        false, 1e-4f);
}

NNOPS_TEST(dwtconv_channel_last_vs_ref_f16) {
    test_channel_last_vs_ref<half>({1, 8, 8, 10}, {8, 1, 3, 3},
        DepthwiseTransposeConv2DAttributes{{3, 3}, {1, 1}, {1, 1}, {1, 1}, {0, 0}},
        true, test::kF16AccumTol);
    test_channel_last_vs_ref<half>({1, 24, 9, 9}, {24, 1, 3, 3},
        DepthwiseTransposeConv2DAttributes{{3, 3}, {2, 2}, {1, 1}, {1, 1}, {0, 0}},
        false, test::kF16AccumTol);
}

// ============================================================
// Threading
// ============================================================

struct SimplePool {
    explicit SimplePool(int nthreads) : nthreads_(nthreads) {
        if (nnops::test::bench_use_openmp()) {
            cpu = nnops::openmp_backend(nthreads);
            return;
        }
        cpu.parallel_for = [this](int64_t begin, int64_t end, const ParallelForBody& body) {
            this->parallel_for(begin, end, body);
        };
        cpu.num_threads = [this]() { return nthreads_; };
        cpu.thread_id = []() { return current_thread_id_; };
    }

    void parallel_for(int64_t begin, int64_t end, const ParallelForBody& body) {
        std::atomic<int64_t> next{begin};
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(nthreads_));
        for (int t = 0; t < nthreads_; ++t) {
            workers.emplace_back([&, t]() {
                current_thread_id_ = t;
                for (;;) {
                    int64_t i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= end) { break; }
                    body(i);
                }
            });
        }
        for (auto& w : workers) { w.join(); }
    }

    int nthreads_;
    CpuBackend cpu;
    inline static thread_local int current_thread_id_ = 0;
};

namespace {

struct ThreadCase {
    int64_t in[4];
    int64_t kh, kw, sh, sw, ph, pw;
    bool last;
};

}  // namespace

NNOPS_TEST(dwtconv_threaded_matches_serial) {
    const ThreadCase cases[] = {
        {{1, 32, 24, 24}, 3, 3, 1, 1, 1, 1, false},
        {{2, 16, 20, 20}, 3, 3, 2, 2, 1, 1, false},
        {{1, 48, 22, 22}, 3, 3, 2, 2, 1, 1, true},
        {{2, 17, 18, 19}, 3, 3, 1, 1, 1, 1, true},
        {{1, 8, 16, 16},   1, 1, 2, 2, 0, 0, true},
        // Row grids too small for the pool: the channel-first route splits each
        // row's residue steps into strips here, so the serial run must agree.
        {{1, 3, 5, 7},     3, 3, 1, 1, 1, 1, false},
        {{2, 2, 4, 4},     3, 3, 2, 2, 1, 1, false},
    };

    for (const auto& c : cases) {
        DepthwiseTransposeConv2DAttributes attrs{
            {c.kh, c.kw}, {c.sh, c.sw}, {1, 1}, {c.ph, c.pw}, {0, 0}};
        auto op = DepthwiseTransposeConv2D::create(attrs, Backend::CPU);

        auto [in_vec, in_cf] = test::make_random_tensor(
            std::span<const int64_t>(c.in, 4));
        auto [w_vec, w_cf] = test::make_random_tensor({c.in[1], 1, c.kh, c.kw});

        std::vector<float> in_cl, w_cl;
        TensorView in = in_cf;
        TensorView w = w_cf;
        if (c.last) {
            in_cl = test::to_channel_last<float>(in_vec, c.in[0], c.in[1],
                                                 c.in[2] * c.in[3]);
            w_cl = test::depthwise_to_hwcm<float>(w_vec, c.in[1], c.kh * c.kw);
            in = TensorView(std::span<const int64_t>(c.in, 4), DataType::f32,
                            in_cl.data(), TensorLayout::NHWC);
            const int64_t wsh[] = {c.kh, c.kw, c.in[1]};
            w = TensorView(std::span<const int64_t>(wsh, 3), DataType::f32,
                           w_cl.data(), TensorLayout::HWCM);
        }

        const TensorDesc in_desc[] = {in.desc(), w.desc()};
        auto out_descs = op->getOutputTensorDesc(in_desc);
        const size_t out_numel = static_cast<size_t>(out_descs[0].numel());
        const TensorLayout out_layout =
            c.last ? TensorLayout::NHWC : TensorLayout::NCHW;

        std::vector<float> serial(out_numel);
        TensorView serial_v(std::span<const int64_t>(out_descs[0].dims.data(),
                                                     static_cast<size_t>(out_descs[0].rank)),
                            DataType::f32, serial.data(), out_layout);
        const TensorView ins[] = {in, w};
        NNOPS_EXPECT_TRUE(is_ok(op->compute(serial_v, ins)));

        for (int nthreads : {2, 4, 8}) {
            SimplePool pool(nthreads);
            ComputeContext ctx;
            ctx.cpu = pool.cpu;

            std::vector<float> threaded(out_numel);
            TensorView out_v(std::span<const int64_t>(out_descs[0].dims.data(),
                                                      static_cast<size_t>(out_descs[0].rank)),
                             DataType::f32, threaded.data(), out_layout);
            NNOPS_EXPECT_TRUE(is_ok(op->compute(out_v, ins, ctx)));

            const bool same = std::memcmp(serial.data(), threaded.data(),
                                          out_numel * sizeof(float)) == 0;
            NNOPS_EXPECT_TRUE(same);
            if (!same) {
                throw std::runtime_error(
                    "dwtconv threaded mismatch (" + std::to_string(nthreads) +
                    " threads, last=" + std::to_string(c.last) + ")");
            }
        }
    }
}

// ============================================================
// Descriptor and refusals
// ============================================================

NNOPS_TEST(dwtconv_output_desc) {
    auto [in_vec, in] = test::make_random_tensor({2, 4, 7, 9});
    auto [w_vec, w] = test::make_random_tensor({4, 1, 3, 3});

    DepthwiseTransposeConv2DAttributes attrs{{3, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1}};
    auto op = DepthwiseTransposeConv2D::create(attrs, Backend::CPU);

    const TensorDesc in_desc[] = {in.desc(), w.desc()};
    auto out_descs = op->getOutputTensorDesc(in_desc);
    NNOPS_EXPECT_EQ(out_descs[0].rank, 4);
    NNOPS_EXPECT_EQ(out_descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(out_descs[0].dims[1], 4);
    // (7-1)*2 - 2 + 1*(3-1) + 1 + 1 = 14; (9-1)*2 - 2 + 2 + 1 + 1 = 18
    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 14);
    NNOPS_EXPECT_EQ(out_descs[0].dims[3], 18);
    NNOPS_EXPECT_TRUE(out_descs[0].layout == TensorLayout::NCHW);
    NNOPS_EXPECT_TRUE(out_descs[0].dtype == DataType::f32);

    // Channel-last inherits the layout.
    std::vector<float> in_cl(in_vec.size());
    TensorView in_last(std::span<const int64_t>(in.shape(),
                                                static_cast<size_t>(in.rank())),
                       DataType::f32, in_cl.data(), TensorLayout::NHWC);
    const int64_t wsh[] = {3, 3, 4};
    std::vector<float> w_cl(w_vec.size());
    TensorView w_last(std::span<const int64_t>(wsh, 3), DataType::f32, w_cl.data(),
                      TensorLayout::HWCM);
    const TensorDesc last_desc[] = {in_last.desc(), w_last.desc()};
    auto last_out = op->getOutputTensorDesc(last_desc);
    NNOPS_EXPECT_TRUE(last_out[0].layout == TensorLayout::NHWC);
    NNOPS_EXPECT_EQ(last_out[0].dims[2], 14);
    NNOPS_EXPECT_EQ(last_out[0].dims[3], 18);
}

NNOPS_TEST(dwtconv_refusals) {
    DepthwiseTransposeConv2DAttributes attrs{{3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};
    auto op = DepthwiseTransposeConv2D::create(attrs, Backend::CPU);

    auto [in_vec, in] = test::make_random_tensor({1, 4, 6, 6});
    auto [w_vec, w] = test::make_random_tensor({4, 1, 3, 3});
    const TensorDesc in_desc[] = {in.desc(), w.desc()};
    auto out_descs = op->getOutputTensorDesc(in_desc);
    std::vector<float> out_buf(static_cast<size_t>(out_descs[0].numel()));
    auto out = test::make_channel_first(out_descs[0], out_buf.data());

    // A channel-last weight is not accepted by a channel-first activation.
    {
        const int64_t wsh[] = {3, 3, 4};
        std::vector<float> w_cl(w_vec.size());
        TensorView w_last(std::span<const int64_t>(wsh, 3), DataType::f32,
                          w_cl.data(), TensorLayout::HWCM);
        const TensorView ins[] = {in, w_last};
        NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::unsupported);
    }

    // A channel-last activation takes the transpose-conv [C, 1, KH, KW] filter
    // as well as HWCM, but one filter per channel is the geometry here: a
    // multiplier is refused.
    {
        std::vector<float> in_cl(in_vec.size());
        TensorView in_last(std::span<const int64_t>(in.shape(),
                                                    static_cast<size_t>(in.rank())),
                           DataType::f32, in_cl.data(), TensorLayout::NHWC);
        auto [wm_vec, wm] = test::make_random_tensor({4, 2, 3, 3});
        const TensorView ins[] = {in_last, wm};
        NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::invalid_argument);
    }

    // Channel counts must agree.
    {
        auto [w3_vec, w3] = test::make_random_tensor({3, 1, 3, 3});
        const TensorView ins[] = {in, w3};
        NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::invalid_argument);
    }

    // Only f32/f16 here: an s8 weight selects the quantized route (whose own
    // suite covers it), and an s8 activation against an fp weight is a mixed
    // pair no route takes.
    {
        std::vector<int8_t> in_i8(in_vec.size(), 1);
        TensorView in_s8(std::span<const int64_t>(in.shape(),
                                                  static_cast<size_t>(in.rank())),
                         DataType::s8, in_i8.data());
        const TensorView ins[] = {in_s8, w};
        NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::unsupported);
    }
}

// ============================================================
// TransposeConv2D's depthwise geometry — group == in_channels runs on these
// kernels
// ============================================================

/// A TransposeConv2D whose group count equals its input channels is the
/// depthwise geometry, and compute() hands it to these kernels instead of the
/// per-group col GEMM. Its filter is [IC, OC/G, KH, KW] = [C, 1, KH, KW], which
/// a channel-first activation reads in place and a channel-last one has
/// repacked. Compare against depthwise_transpose_conv_ref on the same values.
static void transpose_conv2d_depthwise_vs_ref(
    const std::vector<int64_t>& in_shape,
    const DepthwiseTransposeConv2DAttributes& dw,
    bool has_bias,
    bool channel_last)
{
    const int64_t N  = in_shape[0];
    const int64_t C  = in_shape[1];
    const int64_t H  = in_shape[2];
    const int64_t W  = in_shape[3];
    const int64_t S  = H * W;
    const int64_t KH = dw.kernel_size[0];
    const int64_t KW = dw.kernel_size[1];

    auto [in_vec, in_cf] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_cf]  = test::make_random_tensor({C, 1, KH, KW});
    std::vector<float> b_vec;
    TensorView b_cf;
    if (has_bias) {
        auto p = test::make_random_tensor({C});
        b_vec = std::move(p.first);
        b_cf  = p.second;
    }

    // ---- The reference: the depthwise operator's own shape rule and kernel ---
    auto dw_op = DepthwiseTransposeConv2D::create(dw, Backend::CPU);
    const TensorDesc ref_in[] = {in_cf.desc(), w_cf.desc(), b_cf.desc()};
    auto ref_descs = dw_op->getOutputTensorDesc(
        std::span<const TensorDesc>(ref_in, has_bias ? 3 : 2));
    const int64_t out_n = ref_descs[0].numel();
    const int64_t out_S = out_n / (N * C);

    std::vector<float> ref_buf(static_cast<size_t>(out_n), dw.add_to ? 1.0f : 0.0f);
    auto ref = test::make_channel_first(ref_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        const TensorView r[] = {in_cf, w_cf, b_cf};
        backend::cpu::reference::depthwise_transpose_conv_ref(
            dw, ref, std::span<const TensorView>(r, has_bias ? 3 : 2), ctx, nullptr);
    }

    // ---- TransposeConv2D at the same geometry --------------------------------
    TransposeConv2DAttributes attrs;
    attrs.kernel_size    = dw.kernel_size;
    attrs.stride         = dw.stride;
    attrs.dilation       = dw.dilation;
    attrs.padding        = dw.padding;
    attrs.output_padding = dw.output_padding;
    attrs.groups         = C;
    attrs.epilogue       = dw.epilogue;
    attrs.add_to         = dw.add_to;

    auto conv = TransposeConv2D::create(attrs, Backend::CPU);

    std::vector<float> out_buf(static_cast<size_t>(out_n), dw.add_to ? 1.0f : 0.0f);
    if (channel_last) {
        auto in_cl = test::to_channel_last<float>(in_vec, N, C, S);
        const int64_t in_cl_shape[] = {N, C, H, W};
        TensorView in_v(in_cl_shape, DataType::f32, in_cl.data(), TensorLayout::NHWC);

        const TensorDesc cl_in[] = {in_v.desc(), w_cf.desc(), b_cf.desc()};
        auto out_descs = conv->getOutputTensorDesc(
            std::span<const TensorDesc>(cl_in, has_bias ? 3 : 2));
        NNOPS_EXPECT_EQ(out_descs[0].numel(), out_n);
        NNOPS_EXPECT_TRUE(out_descs[0].layout == TensorLayout::NHWC);

        auto out = test::make_channel_first(out_descs[0], out_buf.data());
        const TensorView ins[] = {in_v, w_cf, b_cf};
        NNOPS_EXPECT_TRUE(is_ok(conv->compute(
            out, std::span<const TensorView>(ins, has_bias ? 3 : 2))));
        out_buf = test::to_channel_first<float>(out_buf, N, C, out_S);
    } else {
        const TensorDesc cf_in[] = {in_cf.desc(), w_cf.desc(), b_cf.desc()};
        auto out_descs = conv->getOutputTensorDesc(
            std::span<const TensorDesc>(cf_in, has_bias ? 3 : 2));
        NNOPS_EXPECT_EQ(out_descs[0].numel(), out_n);

        auto out = test::make_channel_first(out_descs[0], out_buf.data());
        const TensorView ins[] = {in_cf, w_cf, b_cf};
        NNOPS_EXPECT_TRUE(is_ok(conv->compute(
            out, std::span<const TensorView>(ins, has_bias ? 3 : 2))));
    }

    for (int64_t i = 0; i < out_n; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[static_cast<size_t>(i)],
                          ref_buf[static_cast<size_t>(i)], 1e-4f);
    }
}

NNOPS_TEST(tconv2d_group_eq_channels_depthwise_channel_first) {
    DepthwiseTransposeConv2DAttributes dw{{3, 3}, {1, 1}, {1, 1}, {1, 1}, {0, 0}};
    dw.epilogue.type = EpilogueActivateType::Relu;
    transpose_conv2d_depthwise_vs_ref({1, 6, 8, 9}, dw, true, false);
}

NNOPS_TEST(tconv2d_group_eq_channels_depthwise_stride2_output_padding) {
    // Stride 2 with output_padding: the geometry the operator reads back from
    // the output size, not from the input's.
    DepthwiseTransposeConv2DAttributes dw{{3, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1}};
    dw.add_to = true;
    transpose_conv2d_depthwise_vs_ref({2, 5, 7, 6}, dw, false, false);
}

NNOPS_TEST(tconv2d_group_eq_channels_depthwise_channel_last) {
    DepthwiseTransposeConv2DAttributes dw{{3, 3}, {1, 1}, {1, 1}, {1, 1}, {0, 0}};
    dw.epilogue.type = EpilogueActivateType::Relu;
    transpose_conv2d_depthwise_vs_ref({1, 8, 8, 9}, dw, true, true);
}

NNOPS_TEST(tconv2d_group_eq_channels_depthwise_channel_last_add_to) {
    // C=17 leaves a one-lane tail past two full vectors; stride 2 makes the
    // output plane a different size from the input's.
    DepthwiseTransposeConv2DAttributes dw{{2, 2}, {2, 2}, {1, 1}, {0, 0}, {0, 0}};
    dw.add_to = true;
    transpose_conv2d_depthwise_vs_ref({1, 17, 7, 6}, dw, false, true);
}
