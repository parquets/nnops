/// @file test_conv3d.cpp
/// @brief Conv3D correctness tests — channel-first (NCDHW), f32 and f16.
///
/// Two kinds of case: hand-computed small shapes that pin exact output values,
/// and random cases against reference::conv3d_channel_first_ref
/// (tests/reference), which accumulates input-driven and so shares no index
/// arithmetic with the im2col gather the kernel takes. The pitched cases give
/// the output or the filter a row stride wider than its row, which is what
/// selects the staged write pass and the gather fallback respectively.
///
/// Input:   [N, IC, ID, IH, IW]
/// Weight:  [OC, IC/G, KD, KH, KW], channel-first
/// Output:  [N, OC, OD, OH, OW]

#include "nnops/ops/conv3d.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/cpu_pool.hpp"

#include <algorithm>
#include <atomic>
#include <span>
#include <thread>
#include <vector>
#include <cstdint>

using namespace nnops;

// Defined in tests/reference/; the channel-first scalar golden.
namespace nnops::backend::cpu::reference {
    void conv3d_channel_first_ref(const Conv3DAttributes& attrs,
                                  TensorView& output,
                                  std::span<const TensorView> inputs,
                                  const ComputeContext& ctx,
                                  void* workspace);
}

/// Space @p v's logical rows back to back — what the reference writes and what
/// the comparisons read, so a pitched view can be checked against it.
static std::vector<float> depitch(const TensorView& v, int64_t row_len)
{
    std::vector<float> out(static_cast<size_t>(v.numel()));
    const float* p = v.ptr<float>();
    const int64_t stride = v.row_stride_elems();
    const int64_t rows = v.numel() / row_len;
    for (int64_t r = 0; r < rows; ++r) {
        std::copy_n(p + r * stride, row_len, out.data() + r * row_len);
    }
    return out;
}

/// Zero the logical rows of @p v, or seed them row by row from a dense buffer.
static void init_rows(TensorView& v, const std::vector<float>& seed, int64_t row_len)
{
    float* p = v.ptr<float>();
    const int64_t stride = v.row_stride_elems();
    const int64_t rows = v.numel() / row_len;
    for (int64_t r = 0; r < rows; ++r) {
        if (seed.empty()) {
            std::fill_n(p + r * stride, row_len, 0.0f);
        } else {
            std::copy_n(seed.data() + r * row_len, row_len, p + r * stride);
        }
    }
}

/// One case against the scalar golden. `residual` fills the destination with a
/// mixed-sign tensor first, so add_to's two orders cannot agree by accident.
/// `pitch_w` gives the filter rows a wider stride (the gather fallback) and
/// `pitch_out` does the same to the output rows (the staged write pass).
static void test_vs_ref(const std::vector<int64_t>& in_shape,
                        const std::vector<int64_t>& w_shape,
                        const Conv3DAttributes& attrs,
                        bool has_bias = false,
                        bool residual = false,
                        bool pitch_w = false,
                        bool pitch_out = false)
{
    const int64_t OC = w_shape[0];

    auto [in_vec, in_view] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_view]  = test::make_random_tensor(w_shape);
    std::vector<float> b_vec;
    TensorView b_view;
    if (has_bias) {
        auto p = test::make_random_tensor({OC});
        b_vec = std::move(p.first);
        b_view = p.second;
    }

    // A pitched filter: same values, rows pushed apart. The operator has to notice
    // it cannot read the filter as one flat matrix.
    std::vector<float> w_pad;
    TensorView w_op = w_view;
    if (pitch_w) {
        const int64_t row_len = w_view.row_stride_elems();
        const int64_t rows = w_view.numel() / row_len;
        const int64_t pitch =
            ((row_len * static_cast<int64_t>(sizeof(float)) + 31) / 32) * 32 + 32;
        w_pad.assign(static_cast<size_t>(rows * (pitch / 4)), 0.0f);
        for (int64_t r = 0; r < rows; ++r) {
            std::copy_n(w_vec.data() + r * row_len, row_len,
                        w_pad.data() + r * (pitch / 4));
        }
        w_op = TensorView(std::span<const int64_t>(w_shape), DataType::f32,
                          w_pad.data(), pitch);
    }

    auto op = Conv3D::create(attrs, Backend::CPU);

    std::vector<TensorDesc> out_descs;
    if (has_bias) {
        const TensorDesc d[] = {in_view.desc(), w_view.desc(), b_view.desc()};
        out_descs = op->getOutputTensorDesc(d);
    } else {
        const TensorDesc d[] = {in_view.desc(), w_view.desc()};
        out_descs = op->getOutputTensorDesc(d);
    }

    const auto numel = static_cast<size_t>(out_descs[0].numel());
    const int64_t OW = out_descs[0].dims[4];

    std::vector<float> seed;
    if (residual) {
        seed = test::make_random_tensor(
            std::span<const int64_t>(out_descs[0].dims.data(),
                                     static_cast<size_t>(out_descs[0].rank)))
                   .first;
    }

    // The operator's output: dense, or one row pitch wide.
    std::vector<float> out_buf(numel);
    TensorView out = test::make_channel_first(out_descs[0], out_buf.data());
    if (pitch_out) {
        const int64_t pitch = out_descs[0].row_pitch(32) + 32;
        out_buf.assign(static_cast<size_t>((numel / OW) * (pitch / 4)), 0.0f);
        out = TensorView(std::span<const int64_t>(out_descs[0].dims.data(),
                                                  static_cast<size_t>(out_descs[0].rank)),
                         DataType::f32, out_buf.data(), pitch);
    }
    init_rows(out, seed, OW);
    {
        TensorView outs[] = {out};
        ComputeContext ctx;
        if (has_bias) {
            const TensorView ins[] = {in_view, w_op, b_view};
            NNOPS_EXPECT_EQ(op->compute(outs, ins, ctx), Status::ok);
        } else {
            const TensorView ins[] = {in_view, w_op};
            NNOPS_EXPECT_EQ(op->compute(outs, ins, ctx), Status::ok);
        }
    }

    std::vector<float> ref_buf = seed.empty() ? std::vector<float>(numel, 0.0f) : seed;
    auto out_ref = test::make_channel_first(out_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        if (has_bias) {
            const TensorView ins[] = {in_view, w_view, b_view};
            backend::cpu::reference::conv3d_channel_first_ref(attrs, out_ref, ins, ctx,
                                                              nullptr);
        } else {
            const TensorView ins[] = {in_view, w_view};
            backend::cpu::reference::conv3d_channel_first_ref(attrs, out_ref, ins, ctx,
                                                              nullptr);
        }
    }

    if (pitch_out) {
        const auto got = depitch(out, OW);
        TensorView got_view(out_descs[0].dims, DataType::f32,
                            const_cast<float*>(got.data()), TensorLayout::NCDHW);
        NNOPS_EXPECT_TRUE(test::allclose(got_view, out_ref, 1e-4f, 1e-4f));
    } else {
        NNOPS_EXPECT_TRUE(test::allclose(out, out_ref, 1e-4f, 1e-4f));
    }
}

NNOPS_TEST(conv3d_kernel1x1_exact) {
    // Input [1, 2, 2, 2, 2] all 3.0; weight [OC=1, IC=2, 1, 1, 1] = {2.0, 3.0},
    // bias 1.0, so every output voxel is 3·2 + 3·3 + 1 = 16.
    const int64_t ishape[] = {1, 2, 2, 2, 2};
    const int64_t wshape[] = {1, 2, 1, 1, 1};
    const int64_t bshape[] = {1};

    std::vector<float> in_data(16, 3.0f);
    std::vector<float> w_data = {2.0f, 3.0f};
    std::vector<float> b_data = {1.0f};
    TensorView in_view(ishape, DataType::f32, in_data.data(), TensorLayout::NCDHW);
    TensorView w_view(wshape, DataType::f32, w_data.data());
    TensorView b_view(bshape, DataType::f32, b_data.data());

    Conv3DAttributes attrs;
    auto op = Conv3D::create(attrs, Backend::CPU);

    const TensorDesc d[] = {in_view.desc(), w_view.desc(), b_view.desc()};
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
        const TensorView ins[] = {in_view, w_view, b_view};
        NNOPS_EXPECT_EQ(op->compute(outs, ins), Status::ok);
    }
    for (float v : out_buf) { NNOPS_EXPECT_NEAR(v, 16.0f, 1e-5f); }
}

NNOPS_TEST(conv3d_output_shape_formula) {
    // O = floor((I + 2P - D·(K-1) - 1) / S) + 1, one term per axis.
    Conv3DAttributes attrs;
    attrs.kernel_size = {3, 4, 2};
    attrs.stride      = {2, 3, 1};
    attrs.dilation    = {1, 1, 1};
    attrs.padding     = {1, 1, 2};

    auto op = Conv3D::create(attrs, Backend::CPU);
    auto [in_vec, in_view] = test::make_random_tensor({1, 4, 5, 6, 7});
    auto [w_vec,  w_view]  = test::make_random_tensor({3, 4, 3, 4, 2});
    const TensorDesc d[] = {in_view.desc(), w_view.desc()};
    const auto descs = op->getOutputTensorDesc(d);

    // OD = floor((5 + 2 - 2 - 1)/2) + 1 = 3
    // OH = floor((6 + 2 - 3 - 1)/3) + 1 = 2
    // OW = floor((7 + 4 - 1 - 1)/1) + 1 = 10
    NNOPS_EXPECT_EQ(descs[0].dims[1], 3);
    NNOPS_EXPECT_EQ(descs[0].dims[2], 3);
    NNOPS_EXPECT_EQ(descs[0].dims[3], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[4], 10);
}

NNOPS_TEST(conv3d_k3_pad1) {
    Conv3DAttributes attrs;
    attrs.kernel_size = {3, 3, 3};
    attrs.padding     = {1, 1, 1};

    test_vs_ref({1, 8, 4, 5, 6}, {16, 8, 3, 3, 3}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(conv3d_k2_stride2) {
    Conv3DAttributes attrs;
    attrs.kernel_size = {2, 2, 2};
    attrs.stride      = {2, 2, 2};
    attrs.padding     = {0, 0, 0};

    test_vs_ref({2, 8, 4, 6, 6}, {8, 8, 2, 2, 2}, attrs);
}

NNOPS_TEST(conv3d_grouped) {
    Conv3DAttributes attrs;
    attrs.kernel_size = {2, 3, 2};
    attrs.stride      = {1, 2, 1};
    attrs.padding     = {0, 1, 0};
    attrs.groups      = 4;

    test_vs_ref({1, 12, 3, 4, 3}, {20, 3, 2, 3, 2}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(conv3d_depthwise) {
    // IC/G = OC/G = 1: the GEMM degenerates to kdim = KD·KH·KW rows.
    Conv3DAttributes attrs;
    attrs.kernel_size = {3, 3, 3};
    attrs.padding     = {1, 1, 1};
    attrs.groups      = 8;

    test_vs_ref({1, 8, 3, 4, 4}, {8, 1, 3, 3, 3}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(conv3d_dilated) {
    Conv3DAttributes attrs;
    attrs.kernel_size = {2, 3, 2};
    attrs.dilation    = {2, 2, 2};
    attrs.padding     = {1, 1, 1};

    test_vs_ref({1, 4, 4, 4, 4}, {6, 4, 2, 3, 2}, attrs);
}

NNOPS_TEST(conv3d_pointwise) {
    // 1x1x1 stride 1: the input plane is the col, with no gather at all.
    Conv3DAttributes attrs;

    test_vs_ref({1, 16, 2, 3, 5}, {8, 16, 1, 1, 1}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(conv3d_pointwise_blocked) {
    // The same shortcut with the pixel blocks split across the depth axis.
    Conv3DAttributes attrs;

    test_vs_ref({2, 32, 5, 3, 3}, {16, 32, 1, 1, 1}, attrs);
}

NNOPS_TEST(conv3d_residual_post_act) {
    // y = relu6(conv + bias) + residual — the order Conv3D's add_to takes.
    Conv3DAttributes attrs;
    attrs.kernel_size = {3, 3, 3};
    attrs.padding     = {1, 1, 1};
    attrs.epilogue    = {EpilogueActivateType::Relu6};
    attrs.add_to      = true;

    test_vs_ref({1, 8, 3, 4, 4}, {8, 8, 3, 3, 3}, attrs, /*has_bias=*/true,
                /*residual=*/true);
}

NNOPS_TEST(conv3d_multiblock) {
    // A col far past one block on three axes at once (kdim = 16·27 = 432 rows).
    Conv3DAttributes attrs;
    attrs.kernel_size = {3, 3, 3};
    attrs.padding     = {1, 1, 1};

    test_vs_ref({1, 16, 4, 10, 10}, {16, 16, 3, 3, 3}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(conv3d_row_split) {
    // kdim·OH·OW past the col cap, so a block is a run of rows inside one depth
    // slice — the roi_h branch rather than whole slices.
    Conv3DAttributes attrs;
    attrs.kernel_size = {5, 5, 5};
    attrs.padding     = {2, 2, 2};

    test_vs_ref({1, 256, 2, 2, 40}, {8, 256, 5, 5, 5}, attrs);
}

NNOPS_TEST(conv3d_pitched_output) {
    // A row stride wider than OW: the GEMM cannot write the tile where it lands.
    Conv3DAttributes attrs;
    attrs.kernel_size = {3, 3, 3};
    attrs.padding     = {1, 1, 1};

    test_vs_ref({1, 8, 3, 4, 5}, {8, 8, 3, 3, 3}, attrs, /*has_bias=*/true,
                /*residual=*/true, /*pitch_w=*/false, /*pitch_out=*/true);
}

NNOPS_TEST(conv3d_pitched_weight) {
    // Filter rows that are not its logical k width: the gather fallback.
    Conv3DAttributes attrs;
    attrs.kernel_size = {3, 3, 3};
    attrs.padding     = {1, 1, 1};

    test_vs_ref({1, 8, 3, 4, 5}, {8, 8, 3, 3, 3}, attrs, /*has_bias=*/true,
                /*residual=*/false, /*pitch_w=*/true);
}

NNOPS_TEST(conv3d_f16) {
    using half = nnops::backend::cpu::half;

    Conv3DAttributes attrs;
    attrs.kernel_size = {3, 3, 3};
    attrs.padding     = {1, 1, 1};

    const std::vector<int64_t> in_shape = {1, 8, 3, 4, 4};
    const std::vector<int64_t> w_shape  = {16, 8, 3, 3, 3};

    auto [in_vec, in_f32] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_f32]  = test::make_random_tensor(w_shape);

    const auto in16 = test::f32_to_f16(in_vec);
    const auto w16  = test::f32_to_f16(w_vec);
    auto in_r = test::f16_to_f32(in16);
    auto w_r  = test::f16_to_f32(w16);

    auto op = Conv3D::create(attrs, Backend::CPU);
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
        backend::cpu::reference::conv3d_channel_first_ref(attrs, out_ref, ins, ctx,
                                                          nullptr);
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

NNOPS_TEST(conv3d_threaded_matches_serial) {
    // The grid splits over batches, oc chunks and output rows, and every task
    // writes rows of its own, so a thread count may only change accumulation
    // order. The residual puts the staged write pass under the comparison.
    const std::vector<int64_t> in_shape = {2, 16, 4, 8, 8};
    const std::vector<int64_t> w_shape  = {32, 16, 3, 3, 3};

    auto [in_vec, in_view] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_view]  = test::make_random_tensor(w_shape);

    Conv3DAttributes attrs;
    attrs.kernel_size = {3, 3, 3};
    attrs.padding     = {1, 1, 1};
    attrs.add_to      = true;

    auto op = Conv3D::create(attrs, Backend::CPU);
    const TensorDesc d[] = {in_view.desc(), w_view.desc()};
    const auto out_descs = op->getOutputTensorDesc(d);
    const auto numel = static_cast<size_t>(out_descs[0].numel());
    const int64_t OW = out_descs[0].dims[4];
    const auto seed = test::make_random_tensor(
                          std::span<const int64_t>(out_descs[0].dims.data(),
                                                   static_cast<size_t>(out_descs[0].rank)))
                          .first;

    std::vector<float> ref_buf = seed;
    TensorView ref = test::make_channel_first(out_descs[0], ref_buf.data());
    init_rows(ref, seed, OW);
    {
        TensorView outs[] = {ref};
        const TensorView ins[] = {in_view, w_view};
        NNOPS_EXPECT_EQ(op->compute(outs, ins, ComputeContext{}), Status::ok);
    }

    for (int nt : {2, 4, 20}) {
        std::vector<float> out_buf = seed;
        TensorView out = test::make_channel_first(out_descs[0], out_buf.data());
        init_rows(out, seed, OW);
        SimplePool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        TensorView outs[] = {out};
        const TensorView ins[] = {in_view, w_view};
        NNOPS_EXPECT_EQ(op->compute(outs, ins, ctx), Status::ok);
        NNOPS_EXPECT_TRUE(test::allclose(out, ref, 1e-5f, 1e-5f));
    }
}
