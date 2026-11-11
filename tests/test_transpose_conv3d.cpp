/// @file test_transpose_conv3d.cpp
/// @brief TransposeConv3D correctness tests — channel-first (NCDHW) and
///        channel-last (NDHWC).
///
/// Two kinds of case: hand-computed small shapes that pin exact output values,
/// and random cases against reference::transpose_conv3d_channel_first_ref
/// (tests/reference), which scatters input-driven and so shares no code with
/// any route the kernel takes. The channel-last cases run the operator on NDHWC
/// tensors and transpose the result back element for element before comparing,
/// so one golden serves both layouts — including the depthwise route, whose
/// channel is the vector axis, and the layout-aware gather fallback.
///
/// Input:   [N, IC, ID, IH, IW]
/// Weight:  [IC, OC/G, KD, KH, KW], channel-first in both layouts
/// Output:  [N, OC, OD, OH, OW], in the input's layout

#include "nnops/ops/transpose_conv3d.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/channel_last.hpp"
#include "common/cpu_pool.hpp"

#include <atomic>
#include <span>
#include <thread>
#include <vector>
#include <cstdint>

using namespace nnops;

// Defined in tests/reference/; the channel-first scalar golden.
namespace nnops::backend::cpu::reference {
    void transpose_conv3d_channel_first_ref(const TransposeConv3DAttributes& attrs,
                                            TensorView& output,
                                            std::span<const TensorView> inputs,
                                            const ComputeContext& ctx,
                                            void* workspace);
}

/// One channel-first case against the scalar golden. `residual` fills the
/// destination with a mixed-sign tensor first, so add_to's two orders cannot
/// agree by accident; the golden gets its own copy of the same values.
static void test_vs_ref(const std::vector<int64_t>& in_shape,
                        const std::vector<int64_t>& w_shape,
                        const TransposeConv3DAttributes& attrs,
                        bool has_bias = false,
                        bool residual = false)
{
    const int64_t OC = w_shape[1] * attrs.groups;

    auto [in_vec, in_view] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_view]  = test::make_random_tensor(w_shape);
    const bool has_bias_ = has_bias;
    std::vector<float> b_vec;
    TensorView b_view;
    if (has_bias_) {
        auto p = test::make_random_tensor({OC});
        b_vec = std::move(p.first);
        b_view = p.second;
    }

    auto op = TransposeConv3D::create(attrs, Backend::CPU);

    std::vector<TensorDesc> out_descs;
    if (has_bias_) {
        const TensorDesc d[] = {in_view.desc(), w_view.desc(), b_view.desc()};
        out_descs = op->getOutputTensorDesc(d);
    } else {
        const TensorDesc d[] = {in_view.desc(), w_view.desc()};
        out_descs = op->getOutputTensorDesc(d);
    }

    const auto numel = static_cast<size_t>(out_descs[0].numel());
    std::vector<float> seed;
    if (residual) {
        auto p = test::make_random_tensor(
            std::span<const int64_t>(out_descs[0].dims.data(),
                                     static_cast<size_t>(out_descs[0].rank)));
        seed = std::move(p.first);
    }

    std::vector<float> out_buf = seed.empty() ? std::vector<float>(numel, 0.0f) : seed;
    auto out = test::make_channel_first(out_descs[0], out_buf.data());
    {
        TensorView outs[] = {out};
        ComputeContext ctx;
        if (has_bias_) {
            const TensorView ins[] = {in_view, w_view, b_view};
            op->compute(outs, ins, ctx);
        } else {
            const TensorView ins[] = {in_view, w_view};
            op->compute(outs, ins, ctx);
        }
    }

    std::vector<float> ref_buf = seed.empty() ? std::vector<float>(numel, 0.0f) : seed;
    auto out_ref = test::make_channel_first(out_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        if (has_bias_) {
            const TensorView ins[] = {in_view, w_view, b_view};
            backend::cpu::reference::transpose_conv3d_channel_first_ref(
                attrs, out_ref, ins, ctx, nullptr);
        } else {
            const TensorView ins[] = {in_view, w_view};
            backend::cpu::reference::transpose_conv3d_channel_first_ref(
                attrs, out_ref, ins, ctx, nullptr);
        }
    }

    NNOPS_EXPECT_TRUE(test::allclose(out, out_ref, 1e-4f, 1e-4f));
}

NNOPS_TEST(tconv3d_kernel1x1_exact) {
    // Input [1, 2, 2, 2, 2] all 3.0; weight [IC=2, OC/G=1, 1, 1, 1] = {2.0, 3.0},
    // so every output voxel is 3·2 + 3·3 = 15.
    const int64_t ishape[] = {1, 2, 2, 2, 2};
    const int64_t wshape[] = {2, 1, 1, 1, 1};

    std::vector<float> in_data(16, 3.0f);
    std::vector<float> w_data = {2.0f, 3.0f};
    TensorView in_view(ishape, DataType::f32, in_data.data(), TensorLayout::NCDHW);
    TensorView w_view(wshape, DataType::f32, w_data.data());

    TransposeConv3DAttributes attrs;
    auto op = TransposeConv3D::create(attrs, Backend::CPU);

    const TensorDesc d[] = {in_view.desc(), w_view.desc()};
    const auto descs = op->getOutputTensorDesc(d);
    NNOPS_EXPECT_EQ(descs[0].rank, 5);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 1);
    NNOPS_EXPECT_EQ(descs[0].dims[2], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[3], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[4], 2);

    std::vector<float> out_buf(8, 0.0f);
    auto out = test::make_channel_first(descs[0], out_buf.data());
    {
        TensorView outs[] = {out};
        const TensorView ins[] = {in_view, w_view};
        NNOPS_EXPECT_EQ(op->compute(outs, ins), Status::ok);
    }
    for (float v : out_buf) { NNOPS_EXPECT_NEAR(v, 15.0f, 1e-5f); }
}

NNOPS_TEST(tconv3d_output_shape_formula) {
    // OD = (ID-1)·SD - 2·PD + DD·(KD-1) + OP + 1, one term per axis.
    TransposeConv3DAttributes attrs;
    attrs.kernel_size    = {3, 4, 2};
    attrs.stride         = {2, 3, 1};
    attrs.dilation       = {1, 2, 1};
    attrs.padding        = {1, 0, 2};
    attrs.output_padding = {1, 0, 0};
    attrs.groups         = 1;

    auto op = TransposeConv3D::create(attrs, Backend::CPU);
    auto [in_vec, in_view] = test::make_random_tensor({1, 4, 5, 6, 7});
    auto [w_vec,  w_view]  = test::make_random_tensor({4, 3, 3, 4, 2});
    const TensorDesc d[] = {in_view.desc(), w_view.desc()};
    const auto descs = op->getOutputTensorDesc(d);

    // OD = 4·2 - 2 + 1·2 + 1 + 1 = 10
    // OH = 5·3 - 0 + 2·3 + 0 + 1 = 22
    // OW = 6·1 - 4 + 1·1 + 0 + 1 = 4
    NNOPS_EXPECT_EQ(descs[0].dims[1], 3);
    NNOPS_EXPECT_EQ(descs[0].dims[2], 10);
    NNOPS_EXPECT_EQ(descs[0].dims[3], 22);
    NNOPS_EXPECT_EQ(descs[0].dims[4], 4);
}

NNOPS_TEST(tconv3d_stride1) {
    TransposeConv3DAttributes attrs;
    attrs.kernel_size = {3, 3, 3};
    attrs.stride      = {1, 1, 1};
    attrs.padding     = {1, 1, 1};

    test_vs_ref({1, 8, 4, 5, 6}, {8, 16, 3, 3, 3}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(tconv3d_stride2) {
    // SW = 2 splits every output row into two residue classes, and SD/SH do the
    // same for the two tap axes above it.
    TransposeConv3DAttributes attrs;
    attrs.kernel_size = {3, 3, 3};
    attrs.stride      = {2, 2, 2};
    attrs.padding     = {1, 1, 1};

    test_vs_ref({2, 8, 3, 4, 5}, {8, 8, 3, 3, 3}, attrs);
}

NNOPS_TEST(tconv3d_grouped) {
    TransposeConv3DAttributes attrs;
    attrs.kernel_size = {2, 3, 2};
    attrs.stride      = {1, 2, 1};
    attrs.padding     = {0, 1, 0};
    attrs.groups      = 4;

    test_vs_ref({1, 16, 3, 4, 3}, {16, 3, 2, 3, 2}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(tconv3d_dilated) {
    TransposeConv3DAttributes attrs;
    attrs.kernel_size = {2, 3, 2};
    attrs.dilation    = {2, 2, 2};
    attrs.padding     = {1, 1, 1};

    test_vs_ref({1, 4, 4, 4, 4}, {4, 6, 2, 3, 2}, attrs);
}

NNOPS_TEST(tconv3d_output_padding) {
    // output_padding only ever extends the trailing edge of the output.
    TransposeConv3DAttributes attrs;
    attrs.kernel_size    = {2, 2, 2};
    attrs.stride         = {2, 2, 2};
    attrs.output_padding = {1, 1, 1};

    test_vs_ref({1, 4, 3, 3, 3}, {4, 5, 2, 2, 2}, attrs);
}

NNOPS_TEST(tconv3d_residual_pre_act) {
    // y = relu6(conv + residual) — the pre-activation order, against the same
    // shape in the other order further down.
    TransposeConv3DAttributes attrs;
    attrs.kernel_size = {3, 3, 3};
    attrs.stride      = {2, 2, 2};
    attrs.padding     = {1, 1, 1};
    attrs.epilogue    = {EpilogueActivateType::Relu6};
    attrs.add_to      = true;
    attrs.residual_pre_activation = true;

    test_vs_ref({1, 8, 3, 4, 4}, {8, 8, 3, 3, 3}, attrs, /*has_bias=*/true,
                /*residual=*/true);
}

NNOPS_TEST(tconv3d_residual_post_act) {
    // y = relu6(conv) + residual.
    TransposeConv3DAttributes attrs;
    attrs.kernel_size = {3, 3, 3};
    attrs.stride      = {2, 2, 2};
    attrs.padding     = {1, 1, 1};
    attrs.epilogue    = {EpilogueActivateType::Relu6};
    attrs.add_to      = true;
    attrs.residual_pre_activation = false;

    test_vs_ref({1, 8, 3, 4, 4}, {8, 8, 3, 3, 3}, attrs, /*has_bias=*/true,
                /*residual=*/true);
}

NNOPS_TEST(tconv3d_kernel_taller_than_tap_list) {
    // KD = 17 is past the per-row tap list the col route keeps on the stack, so
    // this is the gather.
    TransposeConv3DAttributes attrs;
    attrs.kernel_size = {17, 1, 1};

    test_vs_ref({1, 2, 1, 1, 3}, {2, 3, 17, 1, 1}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(tconv3d_wide_oc) {
    // OC = 64 with a one-voxel volume: the col is one batch item wide and the
    // GEMM chunks the M rows.
    TransposeConv3DAttributes attrs;
    attrs.kernel_size = {2, 2, 2};
    attrs.padding     = {0, 0, 0};

    test_vs_ref({1, 8, 2, 2, 2}, {8, 64, 2, 2, 2}, attrs);
}

NNOPS_TEST(tconv3d_f16) {
    using half = nnops::backend::cpu::half;

    TransposeConv3DAttributes attrs;
    attrs.kernel_size = {3, 3, 3};
    attrs.stride      = {2, 2, 2};
    attrs.padding     = {1, 1, 1};

    const std::vector<int64_t> in_shape = {1, 8, 3, 4, 4};
    const std::vector<int64_t> w_shape  = {8, 16, 3, 3, 3};

    auto [in_vec, in_f32] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_f32]  = test::make_random_tensor(w_shape);

    const auto in16 = test::f32_to_f16(in_vec);
    const auto w16  = test::f32_to_f16(w_vec);
    auto in_r = test::f16_to_f32(in16);
    auto w_r  = test::f16_to_f32(w16);

    auto op = TransposeConv3D::create(attrs, Backend::CPU);
    const TensorDesc d[] = {in_f32.desc(), w_f32.desc()};
    const auto descs = op->getOutputTensorDesc(d);
    const auto numel = static_cast<size_t>(descs[0].numel());

    TensorView in_ref(in_shape, DataType::f32, in_r.data(), TensorLayout::NCDHW);
    TensorView w_ref(w_shape, DataType::f32, w_r.data());
    std::vector<float> ref_buf(numel, 0.0f);
    auto out_ref = test::make_channel_first(descs[0], ref_buf.data());
    {
        const TensorView ins[] = {in_ref, w_ref};
        ComputeContext ctx;
        backend::cpu::reference::transpose_conv3d_channel_first_ref(
            attrs, out_ref, ins, ctx, nullptr);
    }

    TensorView in16_view(in_shape, DataType::f16,
                         const_cast<half*>(in16.data()), TensorLayout::NCDHW);
    TensorView w16_view(w_shape, DataType::f16, const_cast<half*>(w16.data()));
    std::vector<half> out16(numel);
    TensorView out16_view(descs[0].dims, DataType::f16, out16.data(),
                          TensorLayout::NCDHW);
    {
        TensorView outs[] = {out16_view};
        const TensorView ins[] = {in16_view, w16_view};
        ComputeContext ctx;
        NNOPS_EXPECT_EQ(op->compute(outs, ins, ctx), Status::ok);
    }

    const auto got = test::f16_to_f32(out16);
    TensorView got_view(descs[0].dims, DataType::f32, const_cast<float*>(got.data()),
                        TensorLayout::NCDHW);
    NNOPS_EXPECT_TRUE(test::allclose(got_view, out_ref, test::kF16AccumTol,
                                     test::kF16AccumTol));
}

// =========================================================================
//  Channel-last (NDHWC)
// =========================================================================

/// One channel-last case: the operator runs on NDHWC tensors, and the result —
/// transposed back element for element — is compared against the channel-first
/// scalar golden on the same values. The weight stays channel-first in both.
static void test_cl_vs_ref(const std::vector<int64_t>& in_shape,
                           const std::vector<int64_t>& w_shape,
                           const TransposeConv3DAttributes& attrs,
                           bool has_bias = false,
                           bool residual = false)
{
    const int64_t N    = in_shape[0];
    const int64_t IC   = in_shape[1];
    const int64_t S_in = in_shape[2] * in_shape[3] * in_shape[4];
    const int64_t OC   = w_shape[1] * attrs.groups;

    auto [in_vec, in_view] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_view]  = test::make_random_tensor(w_shape);

    std::vector<float> b_vec;
    TensorView b_view;
    if (has_bias) {
        auto p = test::make_random_tensor({OC});
        b_vec = std::move(p.first);
        b_view = p.second;
    }

    // The same values in NDHWC, and the descriptors asked of a channel-last call
    // — so the output layout the operator reports is part of what is checked.
    const auto in_cl = test::to_channel_last<float>(in_vec, N, IC, S_in);
    TensorView in_cl_view(in_shape, DataType::f32,
                          const_cast<float*>(in_cl.data()), TensorLayout::NDHWC);

    auto op = TransposeConv3D::create(attrs, Backend::CPU);

    std::vector<TensorDesc> out_descs;
    if (has_bias) {
        const TensorDesc d[] = {in_cl_view.desc(), w_view.desc(), b_view.desc()};
        out_descs = op->getOutputTensorDesc(d);
    } else {
        const TensorDesc d[] = {in_cl_view.desc(), w_view.desc()};
        out_descs = op->getOutputTensorDesc(d);
    }
    NNOPS_EXPECT_EQ(out_descs[0].layout, TensorLayout::NDHWC);

    const auto numel = static_cast<size_t>(out_descs[0].numel());
    std::vector<float> seed;
    if (residual) {
        seed = test::make_random_tensor(
            std::span<const int64_t>(out_descs[0].dims.data(),
                                     static_cast<size_t>(out_descs[0].rank)))
                   .first;
    }

    // The golden: channel-first in, channel-first out — the descriptor says
    // NDHWC, so the golden's view is built by hand in the layout it reads.
    const auto out_dims = std::span<const int64_t>(
        out_descs[0].dims.data(), static_cast<size_t>(out_descs[0].rank));
    std::vector<float> ref_buf = seed.empty() ? std::vector<float>(numel, 0.0f) : seed;
    TensorView out_ref(out_dims, DataType::f32, ref_buf.data(), TensorLayout::NCDHW);
    {
        ComputeContext ctx;
        if (has_bias) {
            const TensorView ins[] = {in_view, w_view, b_view};
            backend::cpu::reference::transpose_conv3d_channel_first_ref(
                attrs, out_ref, ins, ctx, nullptr);
        } else {
            const TensorView ins[] = {in_view, w_view};
            backend::cpu::reference::transpose_conv3d_channel_first_ref(
                attrs, out_ref, ins, ctx, nullptr);
        }
    }

    std::vector<float> out_cl(numel, 0.0f);
    if (residual) {
        out_cl = test::to_channel_last<float>(
            seed, N, OC, numel / static_cast<size_t>(N * OC));
    }
    TensorView out_cl_view(out_dims, DataType::f32, out_cl.data(),
                           TensorLayout::NDHWC);
    {
        TensorView outs[] = {out_cl_view};
        ComputeContext ctx;
        if (has_bias) {
            const TensorView ins[] = {in_cl_view, w_view, b_view};
            NNOPS_EXPECT_EQ(op->compute(outs, ins, ctx), Status::ok);
        } else {
            const TensorView ins[] = {in_cl_view, w_view};
            NNOPS_EXPECT_EQ(op->compute(outs, ins, ctx), Status::ok);
        }
    }

    const auto got = test::to_channel_first<float>(
        out_cl, N, OC, numel / static_cast<size_t>(N * OC));
    TensorView got_view(out_dims, DataType::f32,
                        const_cast<float*>(got.data()), TensorLayout::NCDHW);
    NNOPS_EXPECT_TRUE(test::allclose(got_view, out_ref, 1e-4f, 1e-4f));
}

NNOPS_TEST(tconv3d_channel_last_stride1) {
    TransposeConv3DAttributes attrs;
    attrs.kernel_size = {3, 3, 3};
    attrs.padding     = {1, 1, 1};

    test_cl_vs_ref({1, 8, 4, 5, 6}, {8, 16, 3, 3, 3}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(tconv3d_channel_last_stride2) {
    // SW = 2 splits each output row into residue classes, SD and SH do the same
    // above it — the pixel-major scatter's whole geometry.
    TransposeConv3DAttributes attrs;
    attrs.kernel_size = {3, 3, 3};
    attrs.stride      = {2, 2, 2};
    attrs.padding     = {1, 1, 1};

    test_cl_vs_ref({2, 8, 3, 4, 5}, {8, 8, 3, 3, 3}, attrs);
}

NNOPS_TEST(tconv3d_channel_last_residual) {
    // The merge is the only writer: bias, pre-activation residual, epilogue and
    // add_to all happen there, on an output that already holds values.
    TransposeConv3DAttributes attrs;
    attrs.kernel_size = {3, 3, 3};
    attrs.stride      = {2, 2, 2};
    attrs.padding     = {1, 1, 1};
    attrs.epilogue    = {EpilogueActivateType::Relu6};
    attrs.add_to      = true;
    attrs.residual_pre_activation = true;

    test_cl_vs_ref({1, 8, 3, 4, 4}, {8, 8, 3, 3, 3}, attrs, /*has_bias=*/true,
                   /*residual=*/true);
}

NNOPS_TEST(tconv3d_channel_last_depthwise) {
    // One channel per group: the depthwise route, whose col is the input itself.
    TransposeConv3DAttributes attrs;
    attrs.kernel_size = {3, 3, 3};
    attrs.stride      = {2, 2, 2};
    attrs.padding     = {1, 1, 1};
    attrs.groups      = 8;

    test_cl_vs_ref({1, 8, 3, 4, 4}, {8, 1, 3, 3, 3}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(tconv3d_channel_last_grouped) {
    // IC/G = 2 with G = 4: neither depthwise nor the plain one-group shape.
    TransposeConv3DAttributes attrs;
    attrs.kernel_size = {2, 3, 2};
    attrs.stride      = {1, 2, 1};
    attrs.padding     = {0, 1, 0};
    attrs.groups      = 4;

    test_cl_vs_ref({1, 8, 3, 4, 3}, {8, 3, 2, 3, 2}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(tconv3d_channel_last_gather_fallback) {
    // KD = 17 is past the per-row tap list, so this is the layout-aware gather.
    TransposeConv3DAttributes attrs;
    attrs.kernel_size = {17, 1, 1};

    test_cl_vs_ref({1, 2, 1, 2, 3}, {2, 3, 17, 1, 1}, attrs, /*has_bias=*/true);
}

// ============================================================
// Thread identity
// ============================================================

/// Fixed-size pool over a CpuBackend (the test_conv2d.cpp pattern).
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

NNOPS_TEST(tconv3d_threaded_matches_serial) {
    // The channel-first col route gives each task one output row and the merge
    // is the only writer of a row, so a thread count may only change the GEMM's
    // accumulation order. add_to with the pre-activation order puts both the
    // scratch merge and the residual read under the comparison.
    const std::vector<int64_t> in_shape = {2, 16, 4, 8, 8};
    const std::vector<int64_t> w_shape  = {16, 16, 3, 3, 3};

    auto [in_vec, in_view] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_view]  = test::make_random_tensor(w_shape);

    TransposeConv3DAttributes attrs;
    attrs.kernel_size = {3, 3, 3};
    attrs.stride      = {2, 2, 2};
    attrs.padding     = {1, 1, 1};
    attrs.epilogue    = {EpilogueActivateType::Relu6};
    attrs.add_to      = true;
    attrs.residual_pre_activation = true;

    auto op = TransposeConv3D::create(attrs, Backend::CPU);
    const TensorDesc d[] = {in_view.desc(), w_view.desc()};
    const auto out_descs = op->getOutputTensorDesc(d);
    const auto numel = static_cast<size_t>(out_descs[0].numel());
    const auto seed = test::make_random_tensor(
                          std::span<const int64_t>(out_descs[0].dims.data(),
                                                   static_cast<size_t>(out_descs[0].rank)))
                          .first;

    std::vector<float> ref_buf = seed;
    auto ref = test::make_channel_first(out_descs[0], ref_buf.data());
    {
        TensorView outs[] = {ref};
        const TensorView ins[] = {in_view, w_view};
        NNOPS_EXPECT_EQ(op->compute(outs, ins, ComputeContext{}), Status::ok);
    }

    for (int nt : {2, 4, 20}) {
        std::vector<float> out_buf = seed;
        auto out = test::make_channel_first(out_descs[0], out_buf.data());
        SimplePool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        TensorView outs[] = {out};
        const TensorView ins[] = {in_view, w_view};
        NNOPS_EXPECT_EQ(op->compute(outs, ins, ctx), Status::ok);
        NNOPS_EXPECT_TRUE(test::allclose(out, ref, 1e-5f, 1e-5f));
    }
}
