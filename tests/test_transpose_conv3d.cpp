/// @file test_transpose_conv3d.cpp
/// @brief TransposeConv3D correctness tests on NDHWC activations.
///
/// Hand-computed small shapes pin exact output values, and random cases compare
/// against reference::transpose_conv3d_channel_first_ref (tests/reference),
/// which scatters input-driven and so shares no code with any route the kernel
/// takes. The cases run the operator on NDHWC tensors and transpose
/// the result back element for element before comparing, so one golden serves
/// both orders — including the depthwise route, whose channel is the vector
/// axis, and the layout-aware gather fallback.
///
/// Input:   [N, ID, IH, IW, IC] (channel-last)
/// Weight:  [IC, OC/G, KD, KH, KW] (dense)
/// Output:  [N, OD, OH, OW, OC]

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

// Defined in tests/reference/; the scalar golden, in dense order.
namespace nnops::backend::cpu::reference {
    void transpose_conv3d_channel_first_ref(const TransposeConv3DAttributes& attrs,
                                            TensorView& output,
                                            std::span<const TensorView> inputs,
                                            const ComputeContext& ctx,
                                            void* workspace);
}

// =========================================================================
//  NDHWC
// =========================================================================

/// One case: the operator runs on NDHWC tensors, and the result — transposed
/// back element for element — is compared against the scalar golden on the same
/// values. The weight order stays dense in both.
static void test_vs_ref(const std::vector<int64_t>& in_shape,
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

    // The same values in NDHWC, and the descriptors asked of an NDHWC call
    // — so the output layout the operator reports is part of what is checked.
    const auto in_cl = test::to_channel_last<float>(in_vec, N, IC, S_in);
    // Channel-last dims: [N, ID, IH, IW, IC].
    const int64_t in_cl_shape[] = {in_shape[0], in_shape[2], in_shape[3], in_shape[4],
                                   in_shape[1]};
    TensorView in_cl_view(std::span<const int64_t>(in_cl_shape, 5), DataType::f32,
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
    // The reported desc is channel-last [N, OD, OH, OW, OC]; the golden reads
    // dense [N, OC, OD, OH, OW].
    const int64_t OD = out_descs[0].dims[1];
    const int64_t OH = out_descs[0].dims[2];
    const int64_t OW = out_descs[0].dims[3];
    const int64_t ref_dims[] = {N, OC, OD, OH, OW};
    std::vector<float> seed;
    if (residual) {
        seed = test::make_random_tensor(std::span<const int64_t>(ref_dims, 5)).first;
    }

    // The golden: dense in, dense out.
    const auto out_dims = std::span<const int64_t>(
        out_descs[0].dims.data(), static_cast<size_t>(out_descs[0].rank));
    std::vector<float> ref_buf = seed.empty() ? std::vector<float>(numel, 0.0f) : seed;
    TensorView out_ref(std::span<const int64_t>(ref_dims, 5), DataType::f32,
                       ref_buf.data(), TensorLayout::Dense);
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
    TensorView got_view(std::span<const int64_t>(ref_dims, 5), DataType::f32,
                        const_cast<float*>(got.data()), TensorLayout::Dense);
    NNOPS_EXPECT_TRUE(test::allclose(got_view, out_ref, 1e-4f, 1e-4f));
}

NNOPS_TEST(tconv3d_stride1) {
    TransposeConv3DAttributes attrs;
    attrs.kernel_size = {3, 3, 3};
    attrs.padding     = {1, 1, 1};

    test_vs_ref({1, 8, 4, 5, 6}, {8, 16, 3, 3, 3}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(tconv3d_stride2) {
    // SW = 2 splits each output row into residue classes, SD and SH do the same
    // above it — the pixel-major scatter's whole geometry.
    TransposeConv3DAttributes attrs;
    attrs.kernel_size = {3, 3, 3};
    attrs.stride      = {2, 2, 2};
    attrs.padding     = {1, 1, 1};

    test_vs_ref({2, 8, 3, 4, 5}, {8, 8, 3, 3, 3}, attrs);
}

NNOPS_TEST(tconv3d_residual) {
    // The merge is the only writer: bias, pre-activation residual, epilogue and
    // add_to all happen there, on an output that already holds values.
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

NNOPS_TEST(tconv3d_depthwise) {
    // One channel per group: the depthwise route, whose col is the input itself.
    TransposeConv3DAttributes attrs;
    attrs.kernel_size = {3, 3, 3};
    attrs.stride      = {2, 2, 2};
    attrs.padding     = {1, 1, 1};
    attrs.groups      = 8;

    test_vs_ref({1, 8, 3, 4, 4}, {8, 1, 3, 3, 3}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(tconv3d_grouped) {
    // IC/G = 2 with G = 4: neither depthwise nor the plain one-group shape.
    TransposeConv3DAttributes attrs;
    attrs.kernel_size = {2, 3, 2};
    attrs.stride      = {1, 2, 1};
    attrs.padding     = {0, 1, 0};
    attrs.groups      = 4;

    test_vs_ref({1, 8, 3, 4, 3}, {8, 3, 2, 3, 2}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(tconv3d_gather_fallback) {
    // KD = 17 is past the per-row tap list, so this is the layout-aware gather.
    TransposeConv3DAttributes attrs;
    attrs.kernel_size = {17, 1, 1};

    test_vs_ref({1, 2, 1, 2, 3}, {2, 3, 17, 1, 1}, attrs, /*has_bias=*/true);
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
    // The merge is the only writer of an output element, so a thread count may
    // only change the GEMM's accumulation order. add_to with the pre-activation
    // order puts both the scratch merge and the residual read under the
    // comparison.
    const std::vector<int64_t> in_shape = {2, 16, 4, 8, 8};
    const std::vector<int64_t> w_shape  = {16, 16, 3, 3, 3};
    const int64_t N  = in_shape[0];
    const int64_t IC = in_shape[1];
    const int64_t S  = in_shape[2] * in_shape[3] * in_shape[4];

    auto [in_vec, in_cf] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_view] = test::make_random_tensor(w_shape);

    const auto in_cl = test::to_channel_last<float>(in_vec, N, IC, S);
    TensorView in_view(in_shape, DataType::f32, const_cast<float*>(in_cl.data()),
                       TensorLayout::NDHWC);

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
    const auto out_dims = std::span<const int64_t>(
        out_descs[0].dims.data(), static_cast<size_t>(out_descs[0].rank));
    const auto numel = static_cast<size_t>(out_descs[0].numel());
    const int64_t OC = out_descs[0].dims[1];
    const int64_t outS = static_cast<int64_t>(numel) / (N * OC);

    // A logical residual built in dense order, permuted into each destination's order.
    const auto seed_cf = test::make_random_tensor(out_dims).first;
    const auto seed_cl = test::to_channel_last<float>(seed_cf, N, OC, outS);

    std::vector<float> ref_buf = seed_cl;
    TensorView ref(out_dims, DataType::f32, ref_buf.data(), TensorLayout::NDHWC);
    {
        TensorView outs[] = {ref};
        const TensorView ins[] = {in_view, w_view};
        NNOPS_EXPECT_EQ(op->compute(outs, ins, ComputeContext{}), Status::ok);
    }

    for (int nt : {2, 4, 20}) {
        std::vector<float> out_buf = seed_cl;
        TensorView out(out_dims, DataType::f32, out_buf.data(), TensorLayout::NDHWC);
        SimplePool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        TensorView outs[] = {out};
        const TensorView ins[] = {in_view, w_view};
        NNOPS_EXPECT_EQ(op->compute(outs, ins, ctx), Status::ok);
        NNOPS_EXPECT_TRUE(test::allclose(out, ref, 1e-5f, 1e-5f));
    }
}
