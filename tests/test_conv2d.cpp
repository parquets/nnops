/// Unit tests for Conv2D operator (CPU reference).
/// Uses Conv2D class API exclusively.

#include "nnops/ops/conv2d.hpp"
#include "backend/cpu/conv2d_im2col.h"   // get_conv2d_plan, Conv2DPlan
#include "backend/cpu/simd_kernel/simd_im2col.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <atomic>
#include <thread>
#include <cstdio>
#include <stdexcept>

using namespace nnops;
using nnops::backend::cpu::half;

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

// Minimal fixed-size thread pool exposing a CpuBackend (test_matmul.cpp
// pattern). Items in [begin, end) are claimed via an atomic counter; each
// worker runs body(i) until the range is exhausted.
struct SimplePool {
    explicit SimplePool(int nthreads) : nthreads_(nthreads) {
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

/// Run the fast path and the reference, and report whether every output element
/// agrees within (rtol, atol).
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

    // Fast path (scratch pooled internally by the kernel).
    std::vector<float> fast_buf(static_cast<size_t>(out_numel));
    auto fast_out = test::make_planar(out_descs[0], fast_buf.data());
    op->compute(fast_out, ins, {}, nullptr);

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

/// Run one conv config on a single set of values: the f32 reference is the
/// ground truth, and the f16 fast path runs on the same values converted to
/// f16. Returns the max absolute deviation over the output.
float conv2d_f16_case(const Conv2DAttributes& attrs,
                      const std::vector<int64_t>& ishape,
                      const std::vector<int64_t>& wshape,
                      bool bias, uint64_t seed)
{
    auto [in32v, in32] = test::make_random_tensor(ishape, -1.0f, 1.0f, seed);
    auto [w32v, w32]   = test::make_random_tensor(wshape, -1.0f, 1.0f, seed + 1);

    std::vector<float> b32v;
    TensorView b32;
    if (bias) {
        b32v = std::vector<float>(static_cast<size_t>(wshape[0]), 0.0f);
        test::XorShift128 rng(seed + 2);
        rng.fill_float(b32v.data(), static_cast<int64_t>(b32v.size()), -1.0f, 1.0f);
        const int64_t bshape[] = {wshape[0]};
        b32 = TensorView(bshape, DataType::f32, b32v.data());
    }

    auto op = Conv2D::create(attrs, Backend::CPU);

    // f32 reference (ground truth).
    std::vector<TensorDesc> d32 = {in32.desc(), w32.desc()};
    std::vector<TensorView> i32 = {in32, w32};
    if (bias) { d32.push_back(b32.desc()); i32.push_back(b32); }
    auto od32 = op->getOutputTensorDesc(d32);
    const int64_t n = od32[0].numel();
    std::vector<float> ref_buf(static_cast<size_t>(n));
    auto ref_out = test::make_planar(od32[0], ref_buf.data());
    nnops::backend::cpu::reference::conv2d_ref(attrs, ref_out, i32, {}, nullptr);

    // f16 fast path on the same values.
    auto in16v = test::f32_to_f16(in32v);
    auto w16v  = test::f32_to_f16(w32v);
    TensorView in16(std::span<const int64_t>(ishape), DataType::f16, in16v.data());
    TensorView w16(std::span<const int64_t>(wshape), DataType::f16, w16v.data());

    std::vector<TensorDesc> d16 = {in16.desc(), w16.desc()};
    std::vector<TensorView> i16 = {in16, w16};
    std::vector<half> b16v;
    TensorView b16;
    if (bias) {
        b16v = test::f32_to_f16(b32v);
        const int64_t bshape[] = {wshape[0]};
        b16 = TensorView(bshape, DataType::f16, b16v.data());
        d16.push_back(b16.desc());
        i16.push_back(b16);
    }
    auto od16 = op->getOutputTensorDesc(d16);
    std::vector<half> out16(static_cast<size_t>(n));
    auto fast_out = test::make_planar(od16[0], out16.data());
    op->compute(fast_out, i16, {}, nullptr);

    auto fast32 = test::f16_to_f32(out16);
    float dev = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        dev = std::max(dev,
            std::abs(fast32[static_cast<size_t>(i)] - ref_buf[static_cast<size_t>(i)]));
    }
    return dev;
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

        // ---- kernel area past 3x3 ------------------------------------------
        // get_ic_block() picks icn_block from karea alone: 8 through 3x3, 4
        // through 5x5, 2 through 7x7, 1 past that. Every case above sits on the
        // top rung, so nothing here exercised the ic-block ladder, the narrow
        // im2col panel it produces, or the deeper ic_per_group / icn_block walk
        // that comes with it.

        // 5x5 "same" + bias — karea 25, the first rung down.
        {{1, 3, 32, 32}, {8, 3, 5, 5},     5, 5, 1, 1, 2, 2, 1, 1, 1, true},
        // 5x5 stride 2 (MobileNet-style downsample): strided gather + big kernel.
        {{1, 4, 28, 28}, {16, 4, 5, 5},    5, 5, 2, 2, 1, 1, 1, 1, 1, false},
        // 5x5 dilated 2: karea is still 25, but the gathered window is 9 wide.
        {{1, 3, 32, 32}, {8, 3, 5, 5},     5, 5, 1, 1, 2, 2, 2, 2, 1, false},
        // 5x5 with groups: NG > 1 with the same narrow panel.
        {{1, 8, 20, 20}, {16, 4, 5, 5},    5, 5, 1, 1, 2, 2, 1, 1, 2, false},
        // wide OC (144 > ocn_block 128) over a 5x5 reduction.
        {{1, 16, 14, 14}, {144, 16, 5, 5}, 5, 5, 1, 1, 2, 2, 1, 1, 1, true},
        // non-square 5x7 — karea 35, still on the icn_block-2 rung.
        {{1, 3, 24, 28}, {6, 3, 5, 7},     5, 7, 1, 1, 2, 3, 1, 1, 1, false},

        // 7x7 ResNet stem (stride 2, pad 3) + bias — karea 49, icn_block 2.
        {{1, 3, 32, 32}, {16, 3, 7, 7},    7, 7, 2, 2, 3, 3, 1, 1, 1, true},
        // 7x7 with IC (8) well past icn_block (2): four ic-block iterations.
        {{1, 8, 16, 16}, {8, 8, 7, 7},     7, 7, 1, 1, 3, 3, 1, 1, 1, false},
        // 7x7 with groups: NG > 1 on the narrow panel.
        {{1, 4, 32, 32}, {8, 2, 7, 7},     7, 7, 1, 1, 3, 3, 1, 1, 2, true},

        // karea 81 > 49: the last rung, icn_block 1 (a rank-1 panel).
        {{1, 2, 24, 24}, {4, 2, 9, 9},     9, 9, 1, 1, 4, 4, 1, 1, 1, false},

        // ---- filter-dominated planes ---------------------------------------
        // 256 filters over a 14x14 plane: the expansion loop grows the oh-block
        // to cover the whole plane, so the block is 14 rows — deliberately not a
        // multiple of kAlign. The case above with 144 filters over 8x8 is the
        // only other one that reaches the loop, and its OH is aligned, so this is
        // what exercises the unaligned block end to end.
        {{1, 16, 28, 28}, {256, 16, 3, 3}, 3, 3, 2, 2, 1, 1, 1, 1, 1, false},

        // ---- 1x1 pointwise (the GEMM-direct route) --------------------------
        // At karea 1 with unit stride and zero padding the gather degenerates to
        // a dense row copy, so the plan drops the col buffer and reads B straight
        // from the input. The single 1x1 row above is IC 3 / OC 6, which covers
        // none of the variants that route actually has: IC past the 64-channel
        // ic-block cap (one K pass instead of several), OC past ocn_block,
        // groups (the group-relative ldb), and a batch dimension.
        // IC 128 > 64: K spans the whole group in a single block.
        {{1, 128, 8, 8},   {192, 128, 1, 1}, 1, 1, 1, 1, 0, 0, 1, 1, 1, true},
        // groups 2 with icg 128: pins the group offset and the in-group ldb.
        {{1, 256, 8, 8},   {64, 128, 1, 1},  1, 1, 1, 1, 0, 0, 1, 1, 2, false},
        // batch 2 with IC 128: the batch step must land on the right plane.
        {{2, 128, 10, 10}, {64, 128, 1, 1},  1, 1, 1, 1, 0, 0, 1, 1, 1, false},
        // dilation on a 1x1 is a no-op (there is only one tap), so it must not
        // push the case off the pointwise route.
        {{1, 128, 12, 12}, {32, 128, 1, 1},  1, 1, 1, 1, 0, 0, 3, 3, 1, false},
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

NNOPS_TEST(conv2d_im2col_threaded_matches_serial) {
    // Threaded vs serial bit-identical: the oh-blocks are partitioned into
    // disjoint per-chunk output tiles, so parallel execution (SimplePool) must
    // match the single-threaded result exactly — and both match the reference.
    struct Case { std::array<int64_t,4> ishape; std::array<int64_t,4> wshape;
                  int64_t kh, kw, sh, sw, ph, pw, dh, dw, groups; bool bias; };
    const Case cases[] = {
        // batch=1, groups=1 with a tall OH → the oh-chunks split kicks in.
        {{1, 3, 32, 32}, {8, 3, 3, 3},   3, 3, 1, 1, 1, 1, 1, 1, 1, true},
        {{1, 4, 24, 24}, {12, 4, 3, 3},  3, 3, 2, 2, 1, 1, 1, 1, 1, false},
        // grouped + dilated, small NG so chunks still split.
        {{1, 4, 20, 20}, {8, 2, 3, 3},   3, 3, 1, 1, 1, 1, 2, 2, 2, false},
        // batch=2 exercises both the (batch) and the oh-chunk dimensions.
        {{2, 3, 16, 16}, {8, 3, 3, 3},   3, 3, 1, 1, 1, 1, 1, 1, 1, true},
        // 7x7: karea 49 drops icn_block to 2, so each oh-chunk re-reads the
        // weight panel far more often — still disjoint output tiles, so the
        // per-chunk results must stay bit-identical.
        {{1, 4, 32, 32}, {8, 4, 7, 7},   7, 7, 1, 1, 3, 3, 1, 1, 1, true},
        // 5x5 stride 2: large karea and a strided im2col gather.
        {{1, 8, 28, 28}, {16, 8, 5, 5},  5, 5, 2, 2, 2, 2, 1, 1, 1, false},
        // filter-dominated plane: the expansion loop meets the chunk split here.
        // At 2 threads it collapses to one whole-plane block, at 4 it stops
        // halfway (oh_block 8, 2 blocks, 2 chunks) — both must stay bit-identical
        // to the serial run.
        {{1, 16, 28, 28}, {256, 16, 3, 3}, 3, 3, 2, 2, 1, 1, 1, 1, 1, false},
        // 1x1 pointwise with IC past the old ic-block cap: one K pass over the
        // whole group with B read off the input plane. The chunk split has no col
        // buffer to absorb the row offset here, so a per-chunk B base that was
        // wrong by even a row would break the bit-identity below.
        {{1, 128, 16, 16}, {192, 128, 1, 1}, 1, 1, 1, 1, 0, 0, 1, 1, 1, true},
    };

    for (const auto& c : cases) {
        for (int nthreads : {2, 4}) {
            auto [in_vec, input] = test::make_random_tensor(c.ishape, -1.0f, 1.0f, 711);
            auto [w_vec, weight] = test::make_random_tensor(c.wshape, -1.0f, 1.0f, 712);
            std::vector<float> b_vec;
            TensorView bias;
            if (c.bias) {
                b_vec = std::vector<float>(static_cast<size_t>(c.wshape[0]), 0.0f);
                test::XorShift128 rng(713);
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

            std::vector<TensorDesc> descs = {input.desc(), weight.desc()};
            std::vector<TensorView> ins   = {input, weight};
            if (c.bias) {
                descs.push_back(bias.desc());
                ins.push_back(bias);
            }
            auto op = Conv2D::create(attrs, Backend::CPU);
            auto out_descs = op->getOutputTensorDesc(descs);
            const int64_t out_numel = out_descs[0].numel();

            // Serial (default context, no parallelism hook → oh_chunks == 1).
            std::vector<float> serial_buf(static_cast<size_t>(out_numel));
            auto serial_out = test::make_planar(out_descs[0], serial_buf.data());
            op->compute(serial_out, ins, {}, nullptr);

            // Threaded via SimplePool.
            SimplePool pool(nthreads);
            ComputeContext ctx;
            ctx.cpu = pool.cpu;
            std::vector<float> threaded_buf(static_cast<size_t>(out_numel));
            auto threaded_out = test::make_planar(out_descs[0], threaded_buf.data());
            op->compute(threaded_out, ins, ctx, nullptr);

            for (int64_t i = 0; i < out_numel; ++i) {
                if (serial_buf[static_cast<size_t>(i)] != threaded_buf[static_cast<size_t>(i)]) {
                    throw std::runtime_error(
                        std::string("conv2d_im2col_threaded_matches_serial: bit mismatch at ") +
                        std::to_string(i) + " (" + std::to_string(nthreads) + " threads)");
                }
            }

            // Both match the reference.
            std::vector<float> ref_buf(static_cast<size_t>(out_numel));
            auto ref_out = test::make_planar(out_descs[0], ref_buf.data());
            nnops::backend::cpu::reference::conv2d_ref(attrs, ref_out, ins, {}, nullptr);
            NNOPS_EXPECT_TRUE(test::allclose(serial_out, ref_out, 1e-3f, 1e-4f));
        }
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

    op->compute(fast_out, ins, {}, nullptr);
    nnops::backend::cpu::reference::conv2d_ref(attrs, ref_out, ins, {}, nullptr);

    for (int64_t i = 0; i < out_numel; ++i) {
        NNOPS_EXPECT_NEAR(fast_buf[i], ref_buf[i], 1e-4f);
    }
}

NNOPS_TEST(conv2d_im2col_identity_epilogue_add_to) {
    // add_to with no activation: the residual is folded into the tile at
    // icn == 0 and no epilogue runs at all — an identity epilogue would only
    // copy the tile back over itself. The no-bias form additionally takes the
    // zero_mode seeding path, so the accumulator must come out of the
    // micro-kernel as bias-free Σ + residual rather than being pre-filled.
    // Geometry: IC 16 over icn_block 8 → two icn blocks (only the first is
    // unseeded), OC 40 under ocn_block 128 with a 4-row MR tail, and an 18x18
    // plane split into 4-row blocks → N 72 then a 36 tail under nr 16.
    const int64_t oshape[] = {1, 40, 18, 18};

    for (bool with_bias : {true, false}) {
        const int64_t ishape[] = {1, 16, 18, 18};
        const int64_t wshape[] = {40, 16, 3, 3};
        const int64_t bshape[] = {40};

        auto [in_vec, input] = test::make_random_tensor(ishape, -1.0f, 1.0f, 101);
        auto [w_vec, weight] = test::make_random_tensor(wshape, -1.0f, 1.0f, 102);
        auto [b_vec, bias]   = test::make_random_tensor(bshape, -0.5f, 0.5f, 103);

        Conv2DAttributes attrs;
        attrs.kernel_size = {3, 3};
        attrs.stride      = {1, 1};
        attrs.padding     = {1, 1};
        attrs.add_to      = true;   // epilogue stays identity (None)

        std::vector<TensorDesc> descs = {input.desc(), weight.desc()};
        std::vector<TensorView> ins   = {input, weight};
        if (with_bias) {
            descs.push_back(bias.desc());
            ins.push_back(bias);
        }

        auto op = Conv2D::create(attrs, Backend::CPU);
        auto out_descs = op->getOutputTensorDesc(descs);
        const int64_t out_numel = out_descs[0].numel();

        // Pre-existing residual output, identical for the fast path and the ref.
        auto [res_vec, _] = test::make_random_tensor(oshape, -0.5f, 0.5f, 104);
        std::vector<float> fast_buf(static_cast<size_t>(out_numel));
        auto fast_out = test::make_planar(out_descs[0], fast_buf.data());
        std::vector<float> ref_buf(static_cast<size_t>(out_numel));
        auto ref_out = test::make_planar(out_descs[0], ref_buf.data());
        for (int64_t i = 0; i < out_numel; ++i) {
            fast_buf[i] = res_vec[i];
            ref_buf[i]  = res_vec[i];
        }

        op->compute(fast_out, ins, {}, nullptr);
        nnops::backend::cpu::reference::conv2d_ref(attrs, ref_out, ins, {}, nullptr);

        for (int64_t i = 0; i < out_numel; ++i) {
            NNOPS_EXPECT_NEAR(fast_buf[i], ref_buf[i], 1e-4f);
        }
    }
}

NNOPS_TEST(conv2d_im2col_epilogue_relu6) {
    // Relu6 is a pure clamp of bias + Σ. The fast path and conv2d_ref share
    // apply_epilogue, so comparing them proves nothing here — compare against
    // the same convolution run with no epilogue, clamped by hand instead.
    const int64_t ishape[] = {1, 3, 12, 12};
    const int64_t wshape[] = {6, 3, 3, 3};
    const int64_t bshape[] = {6};

    auto [in_vec, input]  = test::make_random_tensor(ishape, -1.0f, 1.0f, 95);
    auto [w_vec, weight]  = test::make_random_tensor(wshape, -2.0f, 2.0f, 96);
    auto [b_vec, bias]    = test::make_random_tensor(bshape, -1.0f, 1.0f, 97);

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {1, 1};

    std::vector<TensorDesc> descs = {input.desc(), weight.desc(), bias.desc()};
    std::vector<TensorView> ins   = {input, weight, bias};

    auto out_descs = Conv2D::create(attrs, Backend::CPU)->getOutputTensorDesc(descs);
    const int64_t out_numel = out_descs[0].numel();

    std::vector<float> plain_buf(static_cast<size_t>(out_numel));
    auto plain_out = test::make_planar(out_descs[0], plain_buf.data());
    auto plain_op = Conv2D::create(attrs, Backend::CPU);
    plain_op->compute(plain_out, ins, {}, nullptr);

    attrs.epilogue.type = EpilogueActivateType::Relu6;
    std::vector<float> relu6_buf(static_cast<size_t>(out_numel));
    auto relu6_out = test::make_planar(out_descs[0], relu6_buf.data());
    auto relu6_op = Conv2D::create(attrs, Backend::CPU);
    relu6_op->compute(relu6_out, ins, {}, nullptr);

    int64_t below_lo = 0, above_hi = 0;
    for (int64_t i = 0; i < out_numel; ++i) {
        const float want = std::min(std::max(plain_buf[i], 0.0f), 6.0f);
        NNOPS_EXPECT_NEAR(relu6_buf[i], want, 1e-4f);
        below_lo += (plain_buf[i] < 0.0f);
        above_hi += (plain_buf[i] > 6.0f);
    }
    // Both clamp bounds must actually bite, otherwise the case is vacuous.
    NNOPS_EXPECT_TRUE(below_lo > 0 && above_hi > 0);
}

NNOPS_TEST(conv2d_im2col_f16_matches_ref) {
    // f16 fast path against the f32 reference on the same values, across the
    // configurations the f32 table above covers: bias, groups > 1, stride and
    // dilation > 1, a non-square kernel, and K = C*KH*KW past KC_F16 (128) so
    // the reduction spans multiple k-blocks.
    struct Case { std::vector<int64_t> ishape, wshape;
                  int64_t kh, kw, sh, sw, ph, pw, dh, dw, groups; bool bias; };
    const Case cases[] = {
        // baseline 3x3 + bias.
        {{1, 3, 16, 16}, {8, 3, 3, 3},   3, 3, 1, 1, 1, 1, 1, 1, 1, true},
        // stride 2.
        {{2, 4, 13, 15}, {6, 4, 3, 3},   3, 3, 2, 2, 1, 1, 1, 1, 1, false},
        // padding 2 + bias.
        {{1, 2, 12, 12}, {4, 2, 3, 3},   3, 3, 1, 1, 2, 2, 1, 1, 1, true},
        // groups 2.
        {{1, 4, 20, 20}, {8, 2, 2, 2},   2, 2, 2, 2, 1, 1, 1, 1, 2, false},
        // 1x1 pointwise + bias.
        {{1, 3, 16, 16}, {6, 3, 1, 1},   1, 1, 1, 1, 0, 0, 1, 1, 1, true},
        // 1x1 pointwise with K = 128 = KC_F16 past the ic-block cap: the f16 side
        // has to reduce over the whole group in one block, too.
        {{1, 128, 8, 8}, {32, 128, 1, 1}, 1, 1, 1, 1, 0, 0, 1, 1, 1, true},
        // non-square kernel 3x5.
        {{1, 3, 18, 22}, {6, 3, 3, 5},   3, 5, 1, 1, 1, 2, 1, 1, 1, false},
        // dilation 2.
        {{1, 3, 18, 18}, {6, 3, 3, 3},   3, 3, 1, 1, 1, 1, 2, 2, 1, false},
        // K = 32*3*3 = 288 > KC_F16 (128): multi-k-block reduction.
        {{1, 32, 10, 10}, {16, 32, 3, 3}, 3, 3, 1, 1, 1, 1, 1, 1, 1, true},
        // wide N (32) with groups, deeper K, no bias.
        {{1, 16, 12, 12}, {32, 8, 3, 3}, 3, 3, 1, 1, 1, 1, 1, 1, 2, false},

        // ---- kernel area past 3x3 (the f32 table's new rows, in f16) -------
        // 5x5, K = 8*25 = 200 > KC_F16 (128): multi-k-block with a large karea.
        {{1, 8, 16, 16}, {16, 8, 5, 5},  5, 5, 1, 1, 2, 2, 1, 1, 1, true},
        // 5x5 stride 2, groups 2.
        {{1, 8, 20, 20}, {16, 4, 5, 5},  5, 5, 2, 2, 2, 2, 1, 1, 2, false},
        // 7x7 ResNet stem (stride 2, pad 3) + bias — karea 49, icn_block 2.
        {{1, 3, 32, 32}, {16, 3, 7, 7},  7, 7, 2, 2, 3, 3, 1, 1, 1, true},
        // 7x7 with IC (8) past icn_block (2).
        {{1, 8, 16, 16}, {8, 8, 7, 7},   7, 7, 1, 1, 3, 3, 1, 1, 1, false},
        // non-square 5x7 — karea 35.
        {{1, 3, 20, 24}, {6, 3, 5, 7},   5, 7, 1, 1, 2, 3, 1, 1, 1, false},
        // karea 81 > 49: icn_block 1.
        {{1, 2, 20, 20}, {4, 2, 9, 9},   9, 9, 1, 1, 4, 4, 1, 1, 1, false},
    };

    const float tol = nnops::test::kF16AccumTol;
    float worst = 0.0f;
    for (const auto& c : cases) {
        Conv2DAttributes attrs;
        attrs.kernel_size = {c.kh, c.kw};
        attrs.stride      = {c.sh, c.sw};
        attrs.dilation    = {c.dh, c.dw};
        attrs.padding     = {c.ph, c.pw};
        attrs.groups      = c.groups;
        worst = std::max(worst, conv2d_f16_case(attrs, c.ishape, c.wshape, c.bias, 300));
    }
    std::printf("      f16 conv2d (im2col): max deviation %.4f (tol %.3f)\n", worst, tol);
    if (worst > tol) {
        throw std::runtime_error("f16 conv2d deviation " + std::to_string(worst) +
                                 " > tol " + std::to_string(tol));
    }
}

NNOPS_TEST(conv2d_im2col_f16_pitched_input) {
    // The im2col gather must walk the input through the view's row pitch: the
    // f16 input here is a padded buffer (W + 3 slack columns per row, zero
    // filled) whose logical region holds the same values as the dense f32
    // reference input. Any pitch-ignoring read picks up the zero slack and
    // diverges.
    const int64_t C = 4, H = 9, W = 11, Wpad = W + 3;
    const int64_t wshape[] = {8, C, 3, 3};

    const int64_t ishape[] = {1, C, H, W};
    auto [in32v, in32] = test::make_random_tensor(ishape, -1.0f, 1.0f, 401);
    auto [w32v, w32]   = test::make_random_tensor(wshape, -1.0f, 1.0f, 402);

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.padding     = {1, 1};

    auto op = Conv2D::create(attrs, Backend::CPU);

    const std::vector<TensorDesc> d32 = {in32.desc(), w32.desc()};
    const std::vector<TensorView> i32 = {in32, w32};
    auto od = op->getOutputTensorDesc(d32);
    const int64_t n = od[0].numel();
    std::vector<float> ref_buf(static_cast<size_t>(n));
    auto ref_out = test::make_planar(od[0], ref_buf.data());
    nnops::backend::cpu::reference::conv2d_ref(attrs, ref_out, i32, {}, nullptr);

    // Padded f16 input: rows of Wpad elements, only the first W logical.
    auto in16v = test::f32_to_f16(in32v);
    std::vector<half> padded(static_cast<size_t>(C * H * Wpad), static_cast<half>(0));
    for (int64_t c = 0; c < C; ++c) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t w = 0; w < W; ++w) {
                padded[static_cast<size_t>((c * H + h) * Wpad + w)] =
                    in16v[static_cast<size_t>((c * H + h) * W + w)];
            }
        }
    }
    TensorView in16(ishape, DataType::f16, padded.data(),
                    Wpad * static_cast<int64_t>(sizeof(half)));
    auto w16v = test::f32_to_f16(w32v);
    TensorView w16(wshape, DataType::f16, w16v.data());

    const std::vector<TensorDesc> d16 = {in16.desc(), w16.desc()};
    const std::vector<TensorView> i16 = {in16, w16};
    auto od16 = op->getOutputTensorDesc(d16);
    std::vector<half> out16(static_cast<size_t>(n));
    auto fast_out = test::make_planar(od16[0], out16.data());
    op->compute(fast_out, i16, {}, nullptr);

    auto fast32 = test::f16_to_f32(out16);
    float worst = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        worst = std::max(worst,
            std::abs(fast32[static_cast<size_t>(i)] - ref_buf[static_cast<size_t>(i)]));
    }
    std::printf("      f16 conv2d (pitched input): max deviation %.4f\n", worst);
    NNOPS_EXPECT_NEAR(worst, 0.0f, nnops::test::kF16AccumTol);
}

// ============================================================
// Large-kernel planning (5x5 / 7x7 / 9x9)
// ============================================================

NNOPS_TEST(conv2d_large_kernel_plan_ic_block_ladder) {
    // get_ic_block() picks icn_block from karea alone — 64 at 1x1, 8 through
    // 3x3, 4 through 5x5, 2 through 7x7, and 1 past that. One GEMM call reduces
    // over icn_block * karea, so the ladder exists to hold that K roughly flat
    // as the kernel area grows: the im2col panel is written karea times per
    // input channel either way, and a constant K is what keeps the weight
    // panel's re-read cost per output proportional rather than exploding.
    //
    // The correctness tables above now run every rung, but they only observe the
    // result. Pin the ladder itself, so retuning the thresholds is a visible,
    // deliberate change rather than a silent shift in how much traffic every
    // 5x5/7x7 in the tree generates.
    struct Ladder { int64_t kh, kw, want_icn, stride; };
    const Ladder ladder[] = {
        // karea 1 carries stride 2 on purpose: at unit stride with zero padding a
        // 1x1 kernel takes the plan's pointwise path, which bypasses get_ic_block
        // altogether, so this rung needs a strided case to stay on the ladder.
        {1, 1, 64, 2},   // karea  1
        {1, 5,  8, 1},   // karea  5  — inside the band the 3x3 cases use
        {3, 3,  8, 1},   // karea  9  — that band's edge
        {3, 5,  4, 1},   // karea 15  — first rung down
        {5, 5,  4, 1},   // karea 25  — that band's edge
        {5, 7,  2, 1},   // karea 35
        {7, 7,  2, 1},   // karea 49  — that band's edge
        {9, 9,  1, 1},   // karea 81  — past the ladder: a rank-1 panel
    };

    for (const Ladder& l : ladder) {
        const int64_t IC = 64, OC = 64, HW = 24;
        const int64_t ishape[] = {1, IC, HW, HW};
        const int64_t wshape[] = {OC, IC, l.kh, l.kw};

        std::vector<float> in_buf(static_cast<size_t>(IC * HW * HW), 0.0f);
        std::vector<float> w_buf(static_cast<size_t>(OC * IC * l.kh * l.kw), 0.0f);
        TensorView input(ishape, DataType::f32, in_buf.data());
        TensorView weight(wshape, DataType::f32, w_buf.data());

        Conv2DAttributes attrs;
        attrs.kernel_size = {l.kh, l.kw};
        attrs.stride      = {l.stride, l.stride};
        attrs.padding     = {l.kh / 2, l.kw / 2};

        auto op = Conv2D::create(attrs, Backend::CPU);
        const TensorDesc descs[] = {input.desc(), weight.desc()};
        const auto out_descs = op->getOutputTensorDesc(descs);

        const auto plan = nnops::backend::cpu::get_conv2d_plan(
            attrs, descs, out_descs, /*num_threads=*/1, /*use_thread_slots=*/false);

        // Every rung must stay on the im2col path — that is what makes the
        // icn_block expectation below a statement about get_ic_block rather than
        // about the pointwise shortcut.
        NNOPS_EXPECT_FALSE(plan.pointwise);
        NNOPS_EXPECT_TRUE(plan.workspace_size > 0);
        NNOPS_EXPECT_EQ(plan.icn_block, l.want_icn);
        // The plan's im2col panel must be exactly the block the kernel is handed:
        // icn_block channels x karea taps x oh_block output rows x OW columns.
        NNOPS_EXPECT_EQ(plan.col_size,
                        plan.icn_block * l.kh * l.kw * plan.oh_block * out_descs[0].dims[3]);
    }
}

NNOPS_TEST(conv2d_im2col_plan_expands_the_oh_block_when_the_filter_panel_dominates) {
    // The initial oh-block is ceil((PH+1)/kAlign)*kAlign, so a 3x3 stride-2 pad-1
    // layer starts at 4 rows however short the plane is. A small plane with many
    // filters then issued wide-M / narrow-N GEMMs and re-read the weight panel
    // once per block. The expansion loop mirrors the halving loop above: when one
    // block per (batch, group) already keeps half the pool busy — so the split
    // buys no parallelism — and the per-group filter count exceeds the plane
    // area (the proxy MLAS uses to pick its whole-plane ExpandThenGemm), grow the
    // block to cover the plane. These cases pin both sides of that gate.
    struct Case {
        std::array<int64_t,4> ishape; std::array<int64_t,4> wshape;
        int64_t kh, kw, sh, sw, ph, pw, groups; int nthreads;
        int64_t want_oh_block, want_blocks; bool want_pointwise;
    };
    const Case cases[] = {
        // 1x128x28x28 -> 512 3x3 s2 p1: 512 filters > the 14x14 plane's 196
        // positions. 14 rows is not a multiple of kAlign — the block lands on OH
        // itself, since the kernel takes its row count from min(oh_block, OH-oh).
        {{1, 128, 28, 28}, {512, 128, 3, 3}, 3, 3, 2, 2, 1, 1, 1, 1, 14, 1, false},
        // Same shape at 4 threads: the mirrored half-pool gate stops it halfway,
        // at the first block that would leave more than half the pool idle.
        {{1, 128, 28, 28}, {512, 128, 3, 3}, 3, 3, 2, 2, 1, 1, 1, 4,  8, 2, false},
        // ... and at 8 threads it does not fire at all.
        {{1, 128, 28, 28}, {512, 128, 3, 3}, 3, 3, 2, 2, 1, 1, 1, 8,  4, 4, false},
        // 1x1 pointwise over a short plane: 2048 filters > 49 positions. This one
        // also happens to be the pointwise case, so it pins that the expansion
        // loop still lands on the whole plane when the col buffer is gone.
        {{1, 512, 7, 7}, {2048, 512, 1, 1}, 1, 1, 1, 1, 0, 0, 1, 1, 7, 1, true},
        // Counter-case: 64 filters <= 3136 positions, so the 14-block split the
        // halving measurements tuned is left alone.
        {{1, 64, 56, 56}, {64, 64, 3, 3}, 3, 3, 1, 1, 1, 1, 1, 1, 4, 14, false},
    };

    for (const Case& c : cases) {
        std::vector<float> in_buf(
            static_cast<size_t>(c.ishape[0] * c.ishape[1] * c.ishape[2] * c.ishape[3]), 0.0f);
        std::vector<float> w_buf(
            static_cast<size_t>(c.wshape[0] * c.wshape[1] * c.wshape[2] * c.wshape[3]), 0.0f);
        TensorView input(c.ishape, DataType::f32, in_buf.data());
        TensorView weight(c.wshape, DataType::f32, w_buf.data());

        Conv2DAttributes attrs;
        attrs.kernel_size = {c.kh, c.kw};
        attrs.stride      = {c.sh, c.sw};
        attrs.padding     = {c.ph, c.pw};
        attrs.groups      = c.groups;

        auto op = Conv2D::create(attrs, Backend::CPU);
        const TensorDesc descs[] = {input.desc(), weight.desc()};
        const auto out_descs = op->getOutputTensorDesc(descs);

        const auto plan = nnops::backend::cpu::get_conv2d_plan(
            attrs, descs, out_descs, c.nthreads, /*use_thread_slots=*/false);

        NNOPS_EXPECT_EQ(plan.oh_block, c.want_oh_block);
        NNOPS_EXPECT_EQ(plan.num_oh_blocks, c.want_blocks);
        // nthreads=1 with NG=1 leaves exactly one parallel task.
        if (c.nthreads == 1) {
            NNOPS_EXPECT_EQ(plan.oh_chunks, 1);
        }
        NNOPS_EXPECT_EQ(plan.pointwise, c.want_pointwise);
        // Pointwise carries no col buffer at all; otherwise the plan's im2col
        // panel must be exactly the block the kernel is handed.
        NNOPS_EXPECT_EQ(plan.col_size,
                        c.want_pointwise
                            ? 0
                            : plan.icn_block * c.kh * c.kw * plan.oh_block * out_descs[0].dims[3]);
    }
}

NNOPS_TEST(conv2d_pointwise_routing) {
    // A 1x1 kernel at unit stride with zero padding makes the im2col gather a
    // dense row copy, so the plan drops the col buffer (col_size 0) and lifts
    // icn_block off the get_ic_block(1) == 64 cap to the whole group. Pin the
    // accept side and every reject side — each reject is a case where reading B
    // off the input plane would be wrong.
    struct Case {
        std::array<int64_t,4> ishape, wshape;
        int64_t kh, kw, sh, sw, ph, pw, groups;
        bool want_pointwise;
    };
    const Case cases[] = {
        // k1 s1 p0 — the route itself.
        {{1, 64, 8, 8},   {16, 64, 1, 1},  1, 1, 1, 1, 0, 0, 1, true},
        // icg 128 past the 64 cap: the single K block is the whole group.
        {{1, 128, 8, 8},  {16, 128, 1, 1}, 1, 1, 1, 1, 0, 0, 1, true},
        // groups 2 at icg 128 — no extra guard is needed for groups; the
        // group-relative channel stride is what makes that true.
        {{1, 256, 8, 8},  {32, 128, 1, 1}, 1, 1, 1, 1, 0, 0, 2, true},
        // stride 2: an output row is no longer a contiguous input row.
        {{1, 64, 16, 16}, {16, 64, 1, 1},  1, 1, 2, 2, 0, 0, 1, false},
        // pad 1: the first output row's window starts outside the input plane.
        {{1, 64, 16, 16}, {16, 64, 1, 1},  1, 1, 1, 1, 1, 1, 1, false},
        // k3 s1 p0: a real gather, whatever the stride and padding.
        {{1, 64, 16, 16}, {16, 64, 3, 3},  3, 3, 1, 1, 0, 0, 1, false},
    };

    for (const Case& c : cases) {
        std::vector<float> in_buf(
            static_cast<size_t>(c.ishape[0] * c.ishape[1] * c.ishape[2] * c.ishape[3]), 0.0f);
        std::vector<float> w_buf(
            static_cast<size_t>(c.wshape[0] * c.wshape[1] * c.wshape[2] * c.wshape[3]), 0.0f);
        TensorView input(c.ishape, DataType::f32, in_buf.data());
        TensorView weight(c.wshape, DataType::f32, w_buf.data());

        Conv2DAttributes attrs;
        attrs.kernel_size = {c.kh, c.kw};
        attrs.stride      = {c.sh, c.sw};
        attrs.padding     = {c.ph, c.pw};
        attrs.groups      = c.groups;

        auto op = Conv2D::create(attrs, Backend::CPU);
        const TensorDesc descs[] = {input.desc(), weight.desc()};
        const auto out_descs = op->getOutputTensorDesc(descs);
        const auto plan = nnops::backend::cpu::get_conv2d_plan(
            attrs, descs, out_descs, /*num_threads=*/1, /*use_thread_slots=*/false);

        NNOPS_EXPECT_EQ(plan.pointwise, c.want_pointwise);
        if (c.want_pointwise) {
            // One K pass over the whole group and no col buffer at all.
            NNOPS_EXPECT_EQ(plan.icn_block, c.ishape[1] / c.groups);
            NNOPS_EXPECT_EQ(plan.col_size, 0);
        } else {
            NNOPS_EXPECT_EQ(plan.col_size,
                            plan.icn_block * c.kh * c.kw * plan.oh_block * out_descs[0].dims[3]);
        }
    }

    // A pitched input has to stay on the im2col path, and the guard is the only
    // thing that keeps it there: the pointwise B read takes roi_area contiguous
    // elements per channel row, which holds only when the row pitch is IW itself.
    // So build a padded buffer — W + 3 slack columns per row, zero filled, the
    // same logical values as the dense reference input — and check both that the
    // plan rejects it and that the values still come out right: any
    // pitch-ignoring read would pick up the zero slack instead.
    const int64_t C = 128, H = 9, W = 11, Wpad = W + 3;
    const int64_t ishape[] = {1, C, H, W};
    const int64_t wshape[] = {16, C, 1, 1};

    auto [in_vec, in32] = test::make_random_tensor(ishape, -1.0f, 1.0f, 501);
    auto [w_vec, w32]   = test::make_random_tensor(wshape, -1.0f, 1.0f, 502);

    Conv2DAttributes attrs;   // 1x1, stride 1, no padding — otherwise eligible
    auto op = Conv2D::create(attrs, Backend::CPU);

    const std::vector<TensorDesc> d32 = {in32.desc(), w32.desc()};
    const std::vector<TensorView> i32 = {in32, w32};
    auto od = op->getOutputTensorDesc(d32);
    const int64_t n = od[0].numel();
    std::vector<float> ref_buf(static_cast<size_t>(n));
    auto ref_out = test::make_planar(od[0], ref_buf.data());
    nnops::backend::cpu::reference::conv2d_ref(attrs, ref_out, i32, {}, nullptr);

    std::vector<float> padded(static_cast<size_t>(C * H * Wpad), 0.0f);
    for (int64_t c = 0; c < C; ++c) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t w = 0; w < W; ++w) {
                padded[static_cast<size_t>((c * H + h) * Wpad + w)] =
                    in_vec[static_cast<size_t>((c * H + h) * W + w)];
            }
        }
    }
    TensorView pitched(ishape, DataType::f32, padded.data(),
                       Wpad * static_cast<int64_t>(sizeof(float)));

    const TensorDesc pdescs[] = {pitched.desc(), w32.desc()};
    const auto pplan = nnops::backend::cpu::get_conv2d_plan(
        attrs, pdescs, od, /*num_threads=*/1, /*use_thread_slots=*/false);
    NNOPS_EXPECT_FALSE(pplan.pointwise);
    // Still the 1x1 rung of the ic-block ladder, with the col buffer back.
    NNOPS_EXPECT_EQ(pplan.icn_block, int64_t{64});
    NNOPS_EXPECT_EQ(pplan.col_size, int64_t{64} * pplan.oh_block * W);

    const std::vector<TensorView> pins = {pitched, w32};
    std::vector<float> out_buf(static_cast<size_t>(n));
    auto out = test::make_planar(od[0], out_buf.data());
    op->compute(out, pins, {}, nullptr);
    NNOPS_EXPECT_TRUE(test::allclose(out, ref_out, 1e-3f, 1e-4f));
}

NNOPS_TEST(conv2d_pointwise_epilogue_add_to) {
    // The pointwise route skips the im2col gather but must still run the whole
    // epilogue chain: bias seeded into the accumulator, the activation applied to
    // bias + Σ, and the residual add on top of that. IC 128 also makes this the
    // single-K-pass case, where no earlier ic-block seeds the accumulator, so the
    // zero_mode / add_to interaction is pinned too.
    const int64_t ishape[] = {1, 128, 9, 9};
    const int64_t wshape[] = {64, 128, 1, 1};
    const int64_t bshape[] = {64};

    auto [in_vec, input] = test::make_random_tensor(ishape, -1.0f, 1.0f, 511);
    auto [w_vec, weight] = test::make_random_tensor(wshape, -2.0f, 2.0f, 512);
    auto [b_vec, bias]   = test::make_random_tensor(bshape, -1.0f, 1.0f, 513);

    Conv2DAttributes attrs;   // 1x1, stride 1, no padding — the pointwise route
    attrs.epilogue.type = EpilogueActivateType::Relu6;
    attrs.add_to        = true;

    std::vector<TensorDesc> descs = {input.desc(), weight.desc(), bias.desc()};
    std::vector<TensorView> ins   = {input, weight, bias};

    auto op = Conv2D::create(attrs, Backend::CPU);
    auto out_descs = op->getOutputTensorDesc(descs);
    const int64_t out_numel = out_descs[0].numel();

    // Pre-existing residual output, identical for the fast path and the ref.
    std::vector<int64_t> out_shape(out_descs[0].dims.data(),
                                   out_descs[0].dims.data() + out_descs[0].rank);
    auto [res_vec, _] = test::make_random_tensor(out_shape, -0.5f, 0.5f, 514);

    std::vector<float> fast_buf(static_cast<size_t>(out_numel));
    auto fast_out = test::make_planar(out_descs[0], fast_buf.data());
    std::vector<float> ref_buf(static_cast<size_t>(out_numel));
    auto ref_out = test::make_planar(out_descs[0], ref_buf.data());
    for (int64_t i = 0; i < out_numel; ++i) {
        fast_buf[static_cast<size_t>(i)] = res_vec[static_cast<size_t>(i)];
        ref_buf[static_cast<size_t>(i)]  = res_vec[static_cast<size_t>(i)];
    }

    op->compute(fast_out, ins, {}, nullptr);
    nnops::backend::cpu::reference::conv2d_ref(attrs, ref_out, ins, {}, nullptr);

    for (int64_t i = 0; i < out_numel; ++i) {
        NNOPS_EXPECT_NEAR(fast_buf[static_cast<size_t>(i)],
                          ref_buf[static_cast<size_t>(i)], 1e-4f);
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
