/// Unit tests for Conv2D operator (CPU reference).
/// Uses Conv2D class API exclusively.

#include "nnops/ops/conv2d.hpp"
#include "backend/cpu/simd_kernel/simd_im2col.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <cstring>
#include <cstdint>
#include <cmath>

using namespace nnops;

// ============================================================
// Hand-verified small tests
// ============================================================

NNOPS_TEST(conv2d_basic_no_pad) {
    // 1x1x4x4 input all-ones, 1x1x3x3 kernel all-0.5, stride=1, pad=0
    // Expected: 2x2 output, each = 9 * (1.0 * 0.5) = 4.5
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t wshape[] = {1, 1, 3, 3};

    std::vector<float> in_buf(16, 1.0f);
    std::vector<float> w_buf(9, 0.5f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};

    auto op = Conv2D::create(attrs, Backend::CPU);

    const TensorDesc desc_arr[] = {input.desc(), weight.desc()};
    auto outs = op->getOutputTensorDesc(desc_arr);

    int64_t out_numel = 1;
    for (int64_t i = 0; i < outs[0].rank; ++i) {
        out_numel *= outs[0].dims[i];
    }
    std::vector<float> out_buf(static_cast<size_t>(out_numel), 0.0f);
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {input, weight};
    op->compute(out, ins);

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], 4.5f, 1e-4f);
    }
}

NNOPS_TEST(conv2d_with_bias) {
    // Same as above but with bias = 0.5
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t wshape[] = {1, 1, 3, 3};
    const int64_t bshape[] = {1};

    std::vector<float> in_buf(16, 1.0f);
    std::vector<float> w_buf(9, 0.5f);
    std::vector<float> b_buf = {0.5f};

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());
    TensorView bias(bshape, DataType::f32, b_buf.data());

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};

    auto op = Conv2D::create(attrs, Backend::CPU);

    const TensorDesc desc_arr[] = {input.desc(), weight.desc()};
    auto outs = op->getOutputTensorDesc(desc_arr);

    int64_t out_numel = 1;
    for (int64_t i = 0; i < outs[0].rank; ++i) {
        out_numel *= outs[0].dims[i];
    }
    std::vector<float> out_buf(static_cast<size_t>(out_numel), 0.0f);
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {input, weight, bias};
    op->compute(out, ins);

    // Each output = 4.5 + 0.5 = 5.0
    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], 5.0f, 1e-4f);
    }
}

NNOPS_TEST(conv2d_stride_2) {
    // 1x1x6x6 input, 1x1x3x3 kernel, stride=2, pad=0
    // Expected: 2x2 output
    const int64_t ishape[] = {1, 1, 6, 6};
    const int64_t wshape[] = {1, 1, 3, 3};

    std::vector<float> in_buf(36, 1.0f);
    std::vector<float> w_buf(9, 0.5f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {2, 2};
    attrs.padding = {0, 0};

    auto op = Conv2D::create(attrs, Backend::CPU);

    const TensorDesc desc_arr[] = {input.desc(), weight.desc()};
    auto outs = op->getOutputTensorDesc(desc_arr);

    int64_t out_numel = 1;
    for (int64_t i = 0; i < outs[0].rank; ++i) {
        out_numel *= outs[0].dims[i];
    }
    std::vector<float> out_buf(static_cast<size_t>(out_numel), 0.0f);
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {input, weight};
    op->compute(out, ins);

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], 4.5f, 1e-4f);
    }
}

NNOPS_TEST(conv2d_padding_1) {
    // 1x1x2x2 input, 1x1x3x3 kernel, stride=1, pad=1
    // Output should be 2x2 (same spatial size)
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t wshape[] = {1, 1, 3, 3};

    std::vector<float> in_buf(4, 1.0f);
    std::vector<float> w_buf(9, 0.5f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {1, 1};

    auto op = Conv2D::create(attrs, Backend::CPU);

    const TensorDesc desc_arr[] = {input.desc(), weight.desc()};
    auto outs = op->getOutputTensorDesc(desc_arr);

    int64_t out_numel = 1;
    for (int64_t i = 0; i < outs[0].rank; ++i) {
        out_numel *= outs[0].dims[i];
    }
    std::vector<float> out_buf(static_cast<size_t>(out_numel), 0.0f);
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {input, weight};
    op->compute(out, ins);

    // Each output element sees partial kernel region (padded zeros)
    // Top-left: only bottom-right 2x2 of kernel overlaps => 1.0 * 0.5 * 4 = 2.0
    NNOPS_EXPECT_NEAR(out_buf[0], 2.0f, 1e-4f);
    // Top-right: bottom-left 2x2 of kernel overlaps => 2.0
    NNOPS_EXPECT_NEAR(out_buf[1], 2.0f, 1e-4f);
}

NNOPS_TEST(conv2d_grouped) {
    // Grouped conv with groups=2, IC=2, OC=2
    // Each group: 1 input channel -> 1 output channel
    const int64_t ishape[] = {1, 2, 3, 3};
    const int64_t wshape[] = {2, 1, 2, 2};  // [OC, IC/G, KH, KW] = [2, 1, 2, 2]

    std::vector<float> in_buf(18, 1.0f);
    std::vector<float> w_buf(8, 1.0f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());

    Conv2DAttributes attrs;
    attrs.kernel_size = {2, 2};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};
    attrs.groups  = 2;

    auto op = Conv2D::create(attrs, Backend::CPU);

    const TensorDesc desc_arr[] = {input.desc(), weight.desc()};
    auto outs = op->getOutputTensorDesc(desc_arr);

    int64_t out_numel = 1;
    for (int64_t i = 0; i < outs[0].rank; ++i) {
        out_numel *= outs[0].dims[i];
    }
    std::vector<float> out_buf(static_cast<size_t>(out_numel), 0.0f);
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {input, weight};
    op->compute(out, ins);

    // Each group: 1 input channel convolved with 1 filter => 4 products * 1.0 = 4.0
    for (int i = 0; i < 8; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], 4.0f, 1e-4f);
    }
}

// ============================================================
// Random comparison test
// ============================================================

NNOPS_TEST(conv2d_random_small) {
    auto [in_vec, input]   = test::make_random_tensor({1, 3, 8, 8});
    auto [w_vec, weight]   = test::make_random_tensor({4, 3, 3, 3});

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};

    auto op = Conv2D::create(attrs, Backend::CPU);

    const TensorDesc desc_arr[] = {input.desc(), weight.desc()};
    auto outs = op->getOutputTensorDesc(desc_arr);

    int64_t out_numel = 1;
    for (int64_t i = 0; i < outs[0].rank; ++i) {
        out_numel *= outs[0].dims[i];
    }
    std::vector<float> out_buf(static_cast<size_t>(out_numel));
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {input, weight};
    op->compute(out, ins);

    // Should not crash or produce NaN
    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

// ============================================================
// Class API standalone test
// ============================================================

NNOPS_TEST(conv2d_class_api) {
    // Validate Conv2D class API with getOutputTensorDesc + make_planar pipeline.
    // 1x1x4x4 input all-ones, 1x1x3x3 kernel all-0.5, stride=1, pad=0
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t wshape[] = {1, 1, 3, 3};

    std::vector<float> in_buf(16, 1.0f);
    std::vector<float> w_buf(9, 0.5f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};

    auto op = Conv2D::create(attrs, Backend::CPU);

    const TensorDesc desc_arr[] = {input.desc(), weight.desc()};
    auto outs = op->getOutputTensorDesc(desc_arr);

    int64_t out_numel = 1;
    for (int64_t i = 0; i < outs[0].rank; ++i) {
        out_numel *= outs[0].dims[i];
    }
    std::vector<float> out_buf(static_cast<size_t>(out_numel), 0.0f);
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {input, weight};
    op->compute(out, ins);

    // Expected: 2x2 output, each = 9 * (1.0 * 0.5) = 4.5
    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], 4.5f, 1e-4f);
    }
}

// ============================================================
// Tiled im2col + GEMM fast path (vs reference)
// ============================================================

// Reference kernel for comparison (test-local declaration, following the
// test_matmul.cpp pattern).
namespace nnops::backend::cpu::reference {
void conv2d_ref(const Conv2DAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* workspace);
}

namespace {

/// Run the fast path (workspace provided) and the reference, and report
/// whether every output element agrees within (rtol, atol).
bool conv2d_fast_vs_ref(const Conv2DAttributes& attrs,
                        const TensorView& input, const TensorView& weight,
                        const TensorView* bias,
                        float rtol, float atol)
{
    std::vector<TensorDesc> descs = {input.desc(), weight.desc()};
    std::vector<TensorView> ins   = {input, weight};
    if (bias) {
        descs.push_back(bias->desc());
        ins.push_back(*bias);
    }

    auto op = Conv2D::create(attrs, Backend::CPU);
    auto out_descs = op->getOutputTensorDesc(descs);
    const int64_t out_numel = out_descs[0].numel();

    // Fast path (allocated workspace).
    std::vector<float> fast_buf(static_cast<size_t>(out_numel));
    auto fast_out = test::make_planar(out_descs[0], fast_buf.data());
    std::vector<char> workspace(op->getWorkspaceSize(descs, out_descs));
    NNOPS_EXPECT_TRUE(workspace.size() > 0);
    op->compute(fast_out, ins, {}, workspace.data());

    // Reference baseline.
    std::vector<float> ref_buf(static_cast<size_t>(out_numel));
    auto ref_out = test::make_planar(out_descs[0], ref_buf.data());
    nnops::backend::cpu::reference::conv2d_ref(attrs, ref_out, ins, {}, nullptr);

    for (int64_t i = 0; i < out_numel; ++i) {
        const float diff = std::abs(fast_buf[i] - ref_buf[i]);
        const float thr  = atol + rtol * std::max(std::abs(fast_buf[i]), std::abs(ref_buf[i]));
        if (diff > thr) {
            return false;
        }
    }
    return true;
}

}  // anonymous namespace

NNOPS_TEST(conv2d_im2col_matches_ref) {
    // A spread of configs exercising stride, padding, dilation, groups, bias,
    // and multi-block paths (ic_per_group > icn_block, oc_per_group > ocn_block)
    // — each must match the reference to fp32 rounding.
    struct Case { std::array<int64_t,4> ishape; std::array<int64_t,4> wshape;
                  int64_t kh, kw, sh, sw, ph, pw, dh, dw, groups; bool bias; };
    const Case cases[] = {
        {{1, 3, 16, 16}, {8, 3, 3, 3},    3, 3, 1, 1, 1, 1, 1, 1, 1, true},
        {{2, 4, 13, 15}, {6, 4, 3, 3},    3, 3, 2, 2, 1, 1, 1, 1, 1, false},
        {{1, 2, 12, 12}, {4, 2, 3, 3},    3, 3, 1, 1, 2, 2, 1, 1, 1, true},
        {{1, 4, 20, 20}, {8, 2, 2, 2},    2, 2, 2, 2, 1, 1, 1, 1, 2, false},
        {{1, 3, 16, 16}, {6, 3, 1, 1},    1, 1, 1, 1, 0, 0, 1, 1, 1, true},
        {{1, 2, 14, 14}, {4, 2, 3, 3},    3, 3, 1, 1, 1, 1, 1, 1, 1, true},
        {{1, 16, 8, 8},  {32, 16, 3, 3},  3, 3, 1, 1, 1, 1, 1, 1, 1, true},
        {{1, 4, 8, 8},   {144, 4, 3, 3},  3, 3, 1, 1, 1, 1, 1, 1, 1, false},
        {{1, 3, 18, 18}, {6, 3, 3, 3},    3, 3, 1, 1, 1, 1, 2, 2, 1, false},
    };

    for (const auto& c : cases) {
        auto [in_vec, input]   = test::make_random_tensor(c.ishape, -1.0f, 1.0f, 77);
        auto [w_vec, weight]   = test::make_random_tensor(c.wshape, -1.0f, 1.0f, 78);
        std::vector<float> b_vec;
        TensorView bias;
        if (c.bias) {
            b_vec = std::vector<float>(static_cast<size_t>(c.wshape[0]), 0.0f);
            test::XorShift128 rng(79);
            rng.fill_float(b_vec.data(), static_cast<int64_t>(b_vec.size()), -1.0f, 1.0f);
            const int64_t bshape[] = {c.wshape[0]};
            bias = TensorView(bshape, DataType::f32, b_vec.data());
        }

        Conv2DAttributes attrs;
        attrs.kernel_size = {c.kh, c.kw};
        attrs.stride      = {c.sh, c.sw};
        attrs.dilation    = {c.dh, c.dw};
        attrs.padding     = {c.ph, c.pw};
        attrs.groups      = c.groups;

        const TensorView* bp = c.bias ? &bias : nullptr;
        NNOPS_EXPECT_TRUE(conv2d_fast_vs_ref(attrs, input, weight, bp, 1e-3f, 1e-4f));
    }
}

NNOPS_TEST(conv2d_im2col_epilogue_add_to) {
    // Residual + Relu epilogue: the epilogue must apply to bias + Σ, and only
    // then be added to the pre-existing output (matching conv2d_ref exactly).
    const int64_t ishape[] = {1, 2, 10, 10};
    const int64_t wshape[] = {4, 2, 3, 3};
    const int64_t bshape[] = {4};

    auto [in_vec, input]  = test::make_random_tensor(ishape, -1.0f, 1.0f, 91);
    auto [w_vec, weight]  = test::make_random_tensor(wshape, -1.0f, 1.0f, 92);
    auto [b_vec, bias]    = test::make_random_tensor(bshape, -0.5f, 0.5f, 93);

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {1, 1};
    attrs.epilogue.type = EpilogueActivateType::Relu;
    attrs.add_to = true;

    std::vector<TensorDesc> descs = {input.desc(), weight.desc(), bias.desc()};
    std::vector<TensorView> ins   = {input, weight, bias};

    auto op = Conv2D::create(attrs, Backend::CPU);
    auto out_descs = op->getOutputTensorDesc(descs);
    const int64_t out_numel = out_descs[0].numel();

    // Pre-existing residual output (same for fast path and reference).
    std::vector<int64_t> out_shape(out_descs[0].dims.data(),
                                   out_descs[0].dims.data() + out_descs[0].rank);
    auto [res_vec, _] = test::make_random_tensor(out_shape, -0.5f, 0.5f, 94);

    std::vector<float> fast_buf(static_cast<size_t>(out_numel));
    auto fast_out = test::make_planar(out_descs[0], fast_buf.data());
    std::vector<float> ref_buf(static_cast<size_t>(out_numel));
    auto ref_out = test::make_planar(out_descs[0], ref_buf.data());
    for (int64_t i = 0; i < out_numel; ++i) {
        fast_buf[i] = res_vec[i];
        ref_buf[i]  = res_vec[i];
    }

    std::vector<char> workspace(op->getWorkspaceSize(descs, out_descs));
    op->compute(fast_out, ins, {}, workspace.data());
    nnops::backend::cpu::reference::conv2d_ref(attrs, ref_out, ins, {}, nullptr);

    for (int64_t i = 0; i < out_numel; ++i) {
        NNOPS_EXPECT_NEAR(fast_buf[i], ref_buf[i], 1e-4f);
    }
}

NNOPS_TEST(conv2d_im2col_f16) {
    // f16 fast path must match the f32 reference within f16 precision.
    const int64_t ishape[] = {1, 4, 14, 14};
    const int64_t wshape[] = {8, 4, 3, 3};

    auto [in_f32, input_f32]   = test::make_random_tensor(ishape, -1.0f, 1.0f, 101);
    auto [w_f32, weight_f32]   = test::make_random_tensor(wshape, -1.0f, 1.0f, 102);

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {1, 1};

    // f32 reference.
    std::vector<TensorDesc> f32_descs = {input_f32.desc(), weight_f32.desc()};
    std::vector<TensorView> f32_ins   = {input_f32, weight_f32};
    auto op = Conv2D::create(attrs, Backend::CPU);
    auto f32_out_descs = op->getOutputTensorDesc(f32_descs);
    const int64_t out_numel = f32_out_descs[0].numel();
    std::vector<float> ref_buf(static_cast<size_t>(out_numel));
    auto ref_out = test::make_planar(f32_out_descs[0], ref_buf.data());
    nnops::backend::cpu::reference::conv2d_ref(attrs, ref_out, f32_ins, {}, nullptr);

    // f16 fast path.
    auto in_f16  = test::f32_to_f16(in_f32);
    auto w_f16   = test::f32_to_f16(w_f32);
    TensorView input_f16(ishape, DataType::f16, in_f16.data());
    TensorView weight_f16(wshape, DataType::f16, w_f16.data());

    std::vector<TensorDesc> f16_descs = {input_f16.desc(), weight_f16.desc()};
    std::vector<TensorView> f16_ins   = {input_f16, weight_f16};
    auto f16_out_descs = op->getOutputTensorDesc(f16_descs);
    std::vector<nnops::backend::cpu::half> out_f16(static_cast<size_t>(out_numel));
    auto fast_out = test::make_planar(f16_out_descs[0], out_f16.data());
    std::vector<char> workspace(op->getWorkspaceSize(f16_descs, f16_out_descs));
    NNOPS_EXPECT_TRUE(workspace.size() > 0);
    op->compute(fast_out, f16_ins, {}, workspace.data());

    auto fast_f32 = test::f16_to_f32(out_f16);
    for (int64_t i = 0; i < out_numel; ++i) {
        NNOPS_EXPECT_NEAR(fast_f32[i], ref_buf[i], 2e-2f);
    }
}

// ============================================================
// tiled_im2col_3d kernel correctness (vs brute-force)
// ============================================================

NNOPS_TEST(conv3d_im2col_kernel_matches_ref) {
    // tiled_im2col_3d must match a naive per-position gather for a spread of
    // stride / padding / dilation / block-origin configs.
    struct Cfg { int ID, IH, IW, KD, KH, KW, SD, SH, SW, DD, DH, DW, PD, PH, PW;
                 int rod, roh, row, rd, rh, rw; };
    const Cfg cfgs[] = {
        // full output, pad 1, stride 1, dilation 1, block at origin
        {5, 5, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1,  0, 0, 0, 5, 5, 5},
        // no pad, stride 2
        {6, 6, 6, 2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0,  0, 0, 0, 3, 3, 3},
        // dilation 2, pad 2
        {7, 7, 7, 3, 3, 3, 1, 1, 1, 2, 2, 2, 2, 2, 2,  0, 0, 0, 7, 7, 7},
        // sub-block at a non-zero origin (local vs global padding differ)
        {6, 6, 6, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 0, 3, 4, 6},
    };

    for (const auto& c : cfgs) {
        const int64_t ishape[] = {1, 1, c.ID, c.IH, c.IW};
        auto [in_vec, input] = test::make_random_tensor(ishape, -1.0f, 1.0f, 55);
        const float* in = input.ptr<float>();

        const int id_step = c.IH * c.IW;
        const int ih_step = c.IW;
        const int icn_step = c.ID * id_step;

        const int karea = c.KD * c.KH * c.KW;
        const int roi_volume = c.rd * c.rh * c.rw;

        std::vector<float> col(static_cast<size_t>(karea * roi_volume), -12345.0f);
        nnops::kernel::tiled_im2col_3d<float>(
            col.data(), in,
            c.rod, c.roh, c.row, c.rd, c.rh, c.rw,
            c.PD, c.PH, c.PW,   // local padding == global (block covers from origin in first 3 cfg)
            c.PD, c.PH, c.PW,
            1, c.ID, c.IH, c.IW,
            icn_step, id_step, ih_step,
            c.KD, c.KH, c.KW, c.SD, c.SH, c.SW, c.DD, c.DH, c.DW);

        // Brute-force reference.
        bool ok = true;
        for (int kd = 0; kd < c.KD && ok; ++kd) {
            for (int kh = 0; kh < c.KH && ok; ++kh) {
                for (int kw = 0; kw < c.KW && ok; ++kw) {
                    const int row0 = (kd * c.KH + kh) * c.KW + kw;
                    for (int od = 0; od < c.rd && ok; ++od) {
                        for (int oh = 0; oh < c.rh && ok; ++oh) {
                            for (int ow = 0; ow < c.rw; ++ow) {
                                const int id = (c.rod + od) * c.SD + kd * c.DD - c.PD;
                                const int ih = (c.roh + oh) * c.SH + kh * c.DH - c.PH;
                                const int iw = (c.row + ow) * c.SW + kw * c.DW - c.PW;
                                float expect = 0.0f;
                                if (id >= 0 && id < c.ID && ih >= 0 && ih < c.IH && iw >= 0 && iw < c.IW) {
                                    expect = in[id * id_step + ih * ih_step + iw];
                                }
                                const float got = col[row0 * roi_volume + (od * c.rh + oh) * c.rw + ow];
                                if (got != expect) { ok = false; break; }
                            }
                        }
                    }
                }
            }
        }
        NNOPS_EXPECT_TRUE(ok);
    }
}
