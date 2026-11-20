/// @file test_conv3d.cpp
/// @brief Conv3D correctness tests — NDHWC, f32 and f16.
///
/// Two kinds of case: hand-computed small shapes that pin exact output values,
/// and random cases against reference::conv3d_channel_first_ref
/// (tests/reference). The golden is dense, so each case converts
/// its activation to NDHWC for the operator and converts the operator's output
/// back for the comparison. The pitched cases give the output or the filter a
/// row stride wider than its row, which is what selects the staged write pass
/// and the gather fallback respectively.
///
/// Input:   [N, ID, IH, IW, IC] (channel-last)
/// Weight:  [OC, IC/G, KD, KH, KW]
/// Output:  [N, OD, OH, OW, OC]

#include "nnops/ops/conv3d.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/channel_last.hpp"
#include "common/compare.hpp"
#include "common/cpu_pool.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <span>
#include <thread>
#include <vector>
#include <cstdint>

using namespace nnops;

// Defined in tests/reference/; the dense-order scalar golden.
namespace nnops::backend::cpu::reference {
    void conv3d_channel_first_ref(const Conv3DAttributes& attrs,
                                  TensorView& output,
                                  std::span<const TensorView> inputs,
                                  const ComputeContext& ctx,
                                  void* workspace);
    void conv3d_int8_ref(const Conv3DAttributes& attrs,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         void* workspace);
}

/// Space @p v's logical rows back to back — what the comparison reads, so a
/// pitched view can be checked against a dense one.
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
    const int64_t N = in_shape[0];
    const int64_t IC = in_shape[1];
    const int64_t in_spatial = in_shape[2] * in_shape[3] * in_shape[4];
    const int64_t OC = w_shape[0];

    auto [in_vec, in_cf] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_view]  = test::make_random_tensor(w_shape);
    std::vector<float> b_vec;
    TensorView b_view;
    if (has_bias) {
        auto p = test::make_random_tensor({OC});
        b_vec = std::move(p.first);
        b_view = p.second;
    }

    auto in_cl = test::to_channel_last<float>(in_vec, N, IC, in_spatial);
    // Channel-last dims: [N, ID, IH, IW, IC].
    const int64_t in_dims[] = {in_shape[0], in_shape[2], in_shape[3], in_shape[4], in_shape[1]};
    TensorView in_view(in_dims, DataType::f32, in_cl.data(), TensorLayout::NDHWC);

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
    // The output desc is channel-last: [N, OD, OH, OW, OC].
    const int64_t OD = out_descs[0].dims[1];
    const int64_t OH = out_descs[0].dims[2];
    const int64_t OW = out_descs[0].dims[3];
    const int64_t out_spatial = OD * OH * OW;
    const int64_t out_row = OW * OC;
    // The reference reads and writes dense [N, OC, OD, OH, OW].
    const int64_t ref_dims[] = {N, OC, OD, OH, OW};

    std::vector<float> seed_cf;
    std::vector<float> seed_cl;
    if (residual) {
        seed_cl = test::make_random_tensor(
                      std::span<const int64_t>(out_descs[0].dims.data(),
                                               static_cast<size_t>(out_descs[0].rank)))
                      .first;
        seed_cf = test::to_channel_first<float>(seed_cl, N, OC, out_spatial);
    }

    // The operator's output: dense, or one row pitch wide.
    std::vector<float> out_buf(numel);
    TensorView out = test::make_channel_first(out_descs[0], out_buf.data());
    if (pitch_out) {
        const int64_t pitch = out_descs[0].row_pitch(32) + 32;
        out_buf.assign(static_cast<size_t>((numel / out_row) * (pitch / 4)), 0.0f);
        out = TensorView(std::span<const int64_t>(out_descs[0].dims.data(),
                                                  static_cast<size_t>(out_descs[0].rank)),
                         DataType::f32, out_buf.data(), pitch, TensorLayout::NDHWC);
    }
    init_rows(out, seed_cl, out_row);
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

    std::vector<float> ref_buf = seed_cf.empty() ? std::vector<float>(numel, 0.0f) : seed_cf;
    TensorView out_ref(std::span<const int64_t>(ref_dims, 5), DataType::f32, ref_buf.data());
    {
        ComputeContext ctx;
        if (has_bias) {
            const TensorView ins[] = {in_cf, w_view, b_view};
            backend::cpu::reference::conv3d_channel_first_ref(attrs, out_ref, ins, ctx,
                                                              nullptr);
        } else {
            const TensorView ins[] = {in_cf, w_view};
            backend::cpu::reference::conv3d_channel_first_ref(attrs, out_ref, ins, ctx,
                                                              nullptr);
        }
    }

    const std::vector<float> got_cl = pitch_out ? depitch(out, out_row) : out_buf;
    const auto got_cf = test::to_channel_first<float>(got_cl, N, OC, out_spatial);
    TensorView got_view(std::span<const int64_t>(ref_dims, 5), DataType::f32,
                        const_cast<float*>(got_cf.data()));
    NNOPS_EXPECT_TRUE(test::allclose(got_view, out_ref, 1e-4f, 1e-4f));
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
    TensorView in_view(ishape, DataType::f32, in_data.data(), TensorLayout::NDHWC);
    TensorView w_view(wshape, DataType::f32, w_data.data());
    TensorView b_view(bshape, DataType::f32, b_data.data());

    Conv3DAttributes attrs;
    auto op = Conv3D::create(attrs, Backend::CPU);

    const TensorDesc d[] = {in_view.desc(), w_view.desc(), b_view.desc()};
    const auto descs = op->getOutputTensorDesc(d);
    NNOPS_EXPECT_EQ(descs[0].rank, 5);
    // [N, OD, OH, OW, OC]
    NNOPS_EXPECT_EQ(descs[0].dims[1], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[2], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[3], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[4], 1);

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
    auto [in_vec, in_dense] = test::make_random_tensor({1, 4, 5, 6, 7});
    auto [w_vec,  w_view]   = test::make_random_tensor({3, 4, 3, 4, 2});
    // Channel-last: [N, ID, IH, IW, IC].
    const int64_t ishape[] = {1, 5, 6, 7, 4};
    TensorView in_view(ishape, DataType::f32, in_vec.data(), TensorLayout::NDHWC);
    (void)in_dense;
    const TensorDesc d[] = {in_view.desc(), w_view.desc()};
    const auto descs = op->getOutputTensorDesc(d);

    // OD = floor((5 + 2 - 2 - 1)/2) + 1 = 3
    // OH = floor((6 + 2 - 3 - 1)/3) + 1 = 2
    // OW = floor((7 + 4 - 1 - 1)/1) + 1 = 10
    // [N, OD, OH, OW, OC]
    NNOPS_EXPECT_EQ(descs[0].dims[1], 3);
    NNOPS_EXPECT_EQ(descs[0].dims[2], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[3], 10);
    NNOPS_EXPECT_EQ(descs[0].dims[4], 3);
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
    // 1x1x1 stride 1: the input plane is A, with no gather at all.
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
    // A row stride wider than OW·OC: the GEMM cannot write the tile where it lands.
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
    const int64_t in_spatial = in_shape[2] * in_shape[3] * in_shape[4];

    auto [in_vec, in_f32] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_f32]  = test::make_random_tensor(w_shape);

    const auto in16 = test::f32_to_f16(in_vec);
    const auto w16  = test::f32_to_f16(w_vec);
    auto in_r = test::f16_to_f32(in16);
    auto w_r  = test::f16_to_f32(w16);

    // Channel-last activation dims [N, ID, IH, IW, IC].
    const int64_t in_dims[] = {in_shape[0], in_shape[2], in_shape[3], in_shape[4], in_shape[1]};
    // Dense dims for the reference [N, IC, ID, IH, IW].
    const int64_t in_dims_cf[] = {in_shape[0], in_shape[1], in_shape[2], in_shape[3], in_shape[4]};
    auto op = Conv3D::create(attrs, Backend::CPU);
    const TensorDesc d[] = {in_f32.desc(), w_f32.desc()};
    const auto descs = op->getOutputTensorDesc(d);
    const auto numel = static_cast<size_t>(descs[0].numel());
    const int64_t N = in_shape[0];
    const int64_t IC = in_shape[1];
    const int64_t OC = w_shape[0];
    // [N, OD, OH, OW, OC]
    const int64_t out_spatial = descs[0].dims[1] * descs[0].dims[2] * descs[0].dims[3];
    const int64_t ref_dims5[] = {N, OC, descs[0].dims[1], descs[0].dims[2], descs[0].dims[3]};

    TensorView in_ref(std::span<const int64_t>(in_dims_cf, 5), DataType::f32, in_r.data());
    TensorView w_ref(w_shape, DataType::f32, w_r.data());
    std::vector<float> ref_buf(numel, 0.0f);
    TensorView out_ref(std::span<const int64_t>(ref_dims5, 5), DataType::f32, ref_buf.data());
    {
        const TensorView ins[] = {in_ref, w_ref};
        ComputeContext ctx;
        backend::cpu::reference::conv3d_channel_first_ref(attrs, out_ref, ins, ctx,
                                                          nullptr);
    }

    auto in16_cl = test::to_channel_last<half>(in16, N, IC, in_spatial);
    TensorView in16_view(in_dims, DataType::f16, in16_cl.data(), TensorLayout::NDHWC);
    TensorView w16_view(w_shape, DataType::f16, const_cast<half*>(w16.data()));
    std::vector<half> out16(numel);
    TensorView out16_view(descs[0].dims, DataType::f16, out16.data(),
                          TensorLayout::NDHWC);
    {
        TensorView outs[] = {out16_view};
        const TensorView ins[] = {in16_view, w16_view};
        ComputeContext ctx;
        NNOPS_EXPECT_EQ(op->compute(outs, ins, ctx), Status::ok);
    }

    const auto got_cl = test::f16_to_f32(out16);
    const auto got_cf = test::to_channel_first<float>(got_cl, N, OC, out_spatial);
    TensorView got_view(std::span<const int64_t>(ref_dims5, 5), DataType::f32,
                        const_cast<float*>(got_cf.data()));
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
    const int64_t N = in_shape[0];
    const int64_t IC = in_shape[1];
    const int64_t in_spatial = in_shape[2] * in_shape[3] * in_shape[4];

    auto [in_vec, in_dense] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_view]   = test::make_random_tensor(w_shape);
    (void)in_dense;

    auto in_cl = test::to_channel_last<float>(in_vec, N, IC, in_spatial);
    // Channel-last dims: [N, ID, IH, IW, IC].
    const int64_t in_dims[] = {in_shape[0], in_shape[2], in_shape[3], in_shape[4], in_shape[1]};
    TensorView in_view(in_dims, DataType::f32, in_cl.data(), TensorLayout::NDHWC);

    Conv3DAttributes attrs;
    attrs.kernel_size = {3, 3, 3};
    attrs.padding     = {1, 1, 1};
    attrs.add_to      = true;

    auto op = Conv3D::create(attrs, Backend::CPU);
    const TensorDesc d[] = {in_view.desc(), w_view.desc()};
    const auto out_descs = op->getOutputTensorDesc(d);
    const auto numel = static_cast<size_t>(out_descs[0].numel());
    // [N, OD, OH, OW, OC]
    const int64_t OC = out_descs[0].dims[4];
    const int64_t OW = out_descs[0].dims[3];
    const int64_t out_spatial = out_descs[0].dims[1] * out_descs[0].dims[2] * OW;
    const int64_t out_row = OW * OC;
    // The seed is already channel-last, the order the destination is written in.
    const auto seed = test::make_random_tensor(
                          std::span<const int64_t>(out_descs[0].dims.data(),
                                                   static_cast<size_t>(out_descs[0].rank)))
                          .first;

    std::vector<float> ref_buf = seed;
    TensorView ref = test::make_channel_first(out_descs[0], ref_buf.data());
    init_rows(ref, seed, out_row);
    {
        TensorView outs[] = {ref};
        const TensorView ins[] = {in_view, w_view};
        NNOPS_EXPECT_EQ(op->compute(outs, ins, ComputeContext{}), Status::ok);
    }

    for (int nt : {2, 4, 20}) {
        std::vector<float> out_buf = seed;
        TensorView out = test::make_channel_first(out_descs[0], out_buf.data());
        init_rows(out, seed, out_row);
        SimplePool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        TensorView outs[] = {out};
        const TensorView ins[] = {in_view, w_view};
        NNOPS_EXPECT_EQ(op->compute(outs, ins, ctx), Status::ok);
        NNOPS_EXPECT_TRUE(test::allclose(out, ref, 1e-5f, 1e-5f));
    }
}

// ============================================================
// Quantized route — s8 filter, u8/s8/f32/f16 activation
// ============================================================

namespace {

using qhalf = nnops::backend::cpu::half;
using nnops::backend::cpu::half_to_float;
using nnops::backend::cpu::float_to_half;

inline float q_as_float(float v) { return v; }
inline float q_as_float(qhalf v) { return half_to_float(v); }
inline float q_as_float(int8_t v) { return static_cast<float>(v); }
inline float q_as_float(uint8_t v) { return static_cast<float>(v); }

inline size_t q_elem_size(DataType dt) { return static_cast<size_t>(data_type_size(dt)); }

inline QuantParams q_per_tensor(float scale, int32_t zp)
{
    QuantParams qp;
    qp.granularity = QuantGranularity::PerTensor;
    qp.scale = scale;
    qp.zero_point = zp;
    return qp;
}

inline QuantParams q_per_channel(const float* scale, const int32_t* zp, int64_t n)
{
    QuantParams qp;
    qp.granularity = QuantGranularity::PerChannel;
    qp.scale_data = scale;
    qp.zero_point_data = zp;
    qp.num_scales = n;
    return qp;
}

/// float → s8/u8 storage, with the arithmetic the quantizers use.
inline int32_t q_quant(float x, float inv_scale, int32_t zp, bool u8)
{
    const int32_t qmin = u8 ? 0 : -128;
    const int32_t qmax = u8 ? 255 : 127;
    const float v = std::fmaf(x, inv_scale, static_cast<float>(zp));
    int32_t q = static_cast<int32_t>(std::nearbyintf(v));
    return std::min(std::max(q, qmin), qmax);
}

struct QCase3 {
    int64_t n = 1, id = 4, ih = 6, iw = 6, ic = 8;
    int64_t oc = 8, icg = 8;
    std::array<int64_t, 3> k = {3, 3, 3};
    std::array<int64_t, 3> s = {1, 1, 1};
    std::array<int64_t, 3> d = {1, 1, 1};
    std::array<int64_t, 3> p = {1, 1, 1};
    int64_t groups = 1;
    DataType act = DataType::u8;
    DataType out = DataType::u8;
    DataType act_q = DataType::s8;
    bool bias = false;
    bool asym_w = false;
    float s_x = 0.0078125f; // 1/128 — exact in binary
    int32_t zp_x = 0;
    float s_out = 0.03125f;
    int32_t zp_out = 0;
    EpilogueActivateType ep = EpilogueActivateType::None;
    uint64_t seed = 1;
};

/// Run one quantized case through the kernel and the NDHWC reference on the
/// same stored values; an integer destination must agree bit for bit, an fp one
/// within a small relative tolerance.
std::string run_q_case(const QCase3& c, const ComputeContext& ctx = {})
{
    const int64_t in_n = c.n * c.id * c.ih * c.iw * c.ic;
    const int64_t icg = c.icg;
    const int64_t OC = c.oc;
    const bool is_u8 = c.act == DataType::u8;
    const bool int_out = c.out == DataType::s8 || c.out == DataType::u8;

    // Activation, NDHWC storage directly — both arms read the same order.
    const size_t aesz = q_elem_size(c.act);
    std::vector<uint8_t> act_buf(static_cast<size_t>(in_n) * aesz);
    {
        test::XorShift128 rng(c.seed);
        const float inv = 1.0f / c.s_x;
        for (int64_t i = 0; i < in_n; ++i) {
            const float v = rng.next_float(-1.0f, 1.0f);
            if (c.act == DataType::f32) {
                reinterpret_cast<float*>(act_buf.data())[i] = v;
            } else if (c.act == DataType::f16) {
                reinterpret_cast<qhalf*>(act_buf.data())[i] = float_to_half(v);
            } else if (is_u8) {
                act_buf[static_cast<size_t>(i)] =
                    static_cast<uint8_t>(q_quant(v, inv, c.zp_x, true));
            } else {
                reinterpret_cast<int8_t*>(act_buf.data())[i] =
                    static_cast<int8_t>(q_quant(v, inv, c.zp_x, false));
            }
        }
    }

    // Filter [OC][IC/G][KD][KH][KW].
    const int64_t wn = OC * icg * c.k[0] * c.k[1] * c.k[2];
    std::vector<int8_t> w_buf(static_cast<size_t>(wn));
    {
        test::XorShift128 rng(c.seed + 1);
        for (auto& v : w_buf) {
            v = static_cast<int8_t>(q_quant(rng.next_float(-1.0f, 1.0f), 127.0f, 0, false));
        }
    }

    std::vector<float> w_s(static_cast<size_t>(OC));
    std::vector<int32_t> w_zp(static_cast<size_t>(OC));
    for (int64_t oc = 0; oc < OC; ++oc) {
        w_s[static_cast<size_t>(oc)] = 0.01f * static_cast<float>(1 + oc % 5);
        w_zp[static_cast<size_t>(oc)] =
            c.asym_w ? static_cast<int32_t>(-40 + (oc * 7) % 60) : 0;
    }

    std::vector<uint8_t> bias_buf;
    if (c.bias) {
        bias_buf.resize(static_cast<size_t>(OC) * (int_out ? sizeof(int32_t) : q_elem_size(c.out)));
        test::XorShift128 rng(c.seed + 2);
        for (int64_t oc = 0; oc < OC; ++oc) {
            if (int_out) {
                reinterpret_cast<int32_t*>(bias_buf.data())[oc] =
                    static_cast<int32_t>(rng.next_float(-1500.0f, 1500.0f));
            } else if (c.out == DataType::f16) {
                reinterpret_cast<qhalf*>(bias_buf.data())[oc] =
                    float_to_half(rng.next_float(-0.5f, 0.5f));
            } else {
                reinterpret_cast<float*>(bias_buf.data())[oc] = rng.next_float(-0.5f, 0.5f);
            }
        }
    }

    const QuantParams qa = q_per_tensor(c.s_x, c.zp_x);
    const QuantParams qw = q_per_channel(w_s.data(), w_zp.data(), OC);
    const QuantParams qo = int_out ? q_per_tensor(c.s_out, c.zp_out) : QuantParams{};

    Conv3DAttributes attrs;
    attrs.kernel_size = c.k;
    attrs.stride = c.s;
    attrs.dilation = c.d;
    attrs.padding = c.p;
    attrs.groups = c.groups;
    attrs.epilogue.type = c.ep;
    attrs.output_dtype = c.out;
    attrs.act_quant_dtype = c.act_q;

    auto op = Conv3D::create(attrs, Backend::CPU);

    const int64_t ish[] = {c.n, c.id, c.ih, c.iw, c.ic};
    const int64_t wsh[] = {OC, icg, c.k[0], c.k[1], c.k[2]};
    const int64_t bsh[] = {OC};
    TensorView in(ish, c.act, act_buf.data(), TensorLayout::NDHWC, qa);
    TensorView wt(wsh, DataType::s8, w_buf.data(), TensorLayout::Dense, qw);
    const DataType bdt = int_out ? DataType::s32 : c.out;
    TensorView bv(bsh, bdt, bias_buf.empty() ? nullptr : bias_buf.data());

    std::vector<TensorView> ins = {in, wt};
    if (c.bias) {
        ins.push_back(bv);
    }
    std::vector<TensorDesc> descs;
    for (const auto& v : ins) {
        descs.push_back(v.desc());
    }
    const auto od = op->getOutputTensorDesc(descs);
    const DataType odt = od[0].dtype;
    const int64_t out_n = od[0].numel();

    std::vector<uint8_t> ref_buf(od[0].nbytes(), 0);
    {
        TensorView ref_out(std::span<const int64_t>(od[0].dims.data(), 5), odt, ref_buf.data(),
                           TensorLayout::NDHWC, qo);
        nnops::backend::cpu::reference::conv3d_int8_ref(attrs, ref_out, ins, ctx, nullptr);
    }

    std::vector<uint8_t> op_buf(od[0].nbytes(), 0);
    {
        TensorView op_out(std::span<const int64_t>(od[0].dims.data(), 5), odt, op_buf.data(),
                          TensorLayout::NDHWC, qo);
        TensorView outs[] = {op_out};
        const Status st = op->compute(outs, std::span<const TensorView>(ins), ctx, nullptr);
        if (!is_ok(st)) {
            return std::string("kernel returned ") + to_string(st);
        }
    }

    auto compare = [&](auto tag) -> std::string {
        using T = decltype(tag);
        const T* got = reinterpret_cast<const T*>(op_buf.data());
        const T* want = reinterpret_cast<const T*>(ref_buf.data());
        int64_t nbad = 0, first = -1;
        for (int64_t i = 0; i < out_n; ++i) {
            const float a = q_as_float(got[i]);
            const float b = q_as_float(want[i]);
            const bool same = int_out ? (a == b)
                                      : (std::abs(a - b) <= 1e-5f * std::max(1.0f, std::abs(b)));
            if (!same) {
                if (first < 0) {
                    first = i;
                }
                ++nbad;
            }
        }
        if (nbad == 0) {
            return {};
        }
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "%lld of %lld elements differ (first at %lld: kernel %.9g, ref %.9g)",
                      static_cast<long long>(nbad), static_cast<long long>(out_n),
                      static_cast<long long>(first), double(q_as_float(got[first])),
                      double(q_as_float(want[first])));
        return msg;
    };

    if (odt == DataType::f32) {
        return compare(float{});
    }
    if (odt == DataType::f16) {
        return compare(qhalf{});
    }
    if (odt == DataType::s8) {
        return compare(int8_t{});
    }
    return compare(uint8_t{});
}

} // namespace


NNOPS_TEST(conv3d_int8_threaded_matches_serial) {
    QCase3 c;
    c.n = 2;
    c.id = 4;
    c.ih = 8;
    c.iw = 8;
    c.ic = 16;
    c.icg = 16;
    c.oc = 16;
    c.groups = 1;
    c.bias = true;
    c.ep = EpilogueActivateType::Relu;

    const std::string serial = run_q_case(c, ComputeContext{});
    if (!serial.empty()) {
        throw std::runtime_error("conv3d int8 serial: " + serial);
    }
    for (int nt : {2, 4}) {
        SimplePool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const std::string diff = run_q_case(c, ctx);
        if (!diff.empty()) {
            throw std::runtime_error("conv3d int8 threaded (" + std::to_string(nt) + "T): " +
                                     diff);
        }
    }
}

NNOPS_TEST(conv3d_int8_refusals) {
    std::vector<uint8_t> a(1 * 2 * 2 * 2 * 4, 0);
    std::vector<int8_t> w(4 * 4 * 1 * 1 * 1, 1);
    std::vector<uint8_t> o(1 * 2 * 2 * 2 * 4, 0);

    const int64_t ish[] = {1, 2, 2, 2, 4};
    const int64_t wsh[] = {4, 4, 1, 1, 1};
    TensorView in(ish, DataType::u8, a.data(), TensorLayout::NDHWC);
    TensorView wt(wsh, DataType::s8, w.data(), TensorLayout::Dense);

    // add_to has no residual stage on the integer path.
    {
        Conv3DAttributes attrs;
        attrs.kernel_size = {1, 1, 1};
        attrs.add_to = true;
        auto op = Conv3D::create(attrs, Backend::CPU);
        TensorView out(ish, DataType::u8, o.data(), TensorLayout::NDHWC);
        TensorView outs[] = {out};
        const TensorView ins[] = {in, wt};
        NNOPS_EXPECT_EQ(op->compute(outs, ins, ComputeContext{}, nullptr), Status::unsupported);
    }

    // An fp activation needs its quantization parameters.
    {
        std::vector<float> af(static_cast<size_t>(1 * 2 * 2 * 2 * 4), 0.5f);
        std::vector<float> of(static_cast<size_t>(1 * 2 * 2 * 2 * 4), 0.0f);
        TensorView in_fp(ish, DataType::f32, af.data(), TensorLayout::NDHWC);
        Conv3DAttributes attrs;
        attrs.kernel_size = {1, 1, 1};
        attrs.output_dtype = DataType::f32;
        auto op = Conv3D::create(attrs, Backend::CPU);
        TensorView out(ish, DataType::f32, of.data(), TensorLayout::NDHWC);
        TensorView outs[] = {out};
        const TensorView ins[] = {in_fp, wt};
        NNOPS_EXPECT_EQ(op->compute(outs, ins, ComputeContext{}, nullptr),
                        Status::invalid_argument);
    }
}
