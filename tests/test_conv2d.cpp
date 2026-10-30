/// Unit tests for Conv2D operator (CPU reference).
/// Uses Conv2D class API exclusively.

#include "nnops/ops/conv2d.hpp"
#include "backend/cpu/conv/conv2d_implicit_gemm.h"   // get_conv2d_implicit_gemm_plan, Conv2DGemmPlan
#include "backend/cpu/matmul/matmul_helper.h" // nr_max_flt / mma_pack_align / tile_pack_rhs
#include "backend/cpu/simd_kernel/simd_im2col.hpp"
#include "common/test_harness.hpp"
#include "common/cpu_pool.hpp"
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
#include <type_traits>

using namespace nnops;
using nnops::backend::cpu::half;

/// Packed-B scratch (elements per slot) a float conv plan must reserve: the
/// plan's im2col block — icn_block channels x karea taps x oh_block rows x OW
/// columns — re-laid out in nr_max-wide panels. Mirrors get_conv2d_implicit_gemm_plan;
/// nr_max is arch-dependent (12 on aarch64, 16 on x86_64).
static int64_t expected_pack_size(const nnops::backend::cpu::Conv2DGemmPlan& plan,
                                  int64_t karea, int64_t out_w) {
    constexpr int nr = nnops::backend::cpu::nr_max_flt<float>();
    const int64_t kdim = plan.icn_block * karea;
    return ((plan.oh_block * out_w + nr - 1) / nr)
         * static_cast<int64_t>(nnops::backend::cpu::mma_pack_align(
               static_cast<size_t>(nr * kdim * sizeof(float))) / sizeof(float));
}

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
        // OpenMP when the build has it (common/cpu_pool.hpp): the adapter then
        // runs under the same bit-exact comparisons as the kernels.
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
        // get_ic_block() picks icn_block from karea alone: 16 through 3x3, 8
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
    // Geometry: IC 32 over icn_block 16 → two icn blocks (only the first is
    // unseeded), OC 40 under ocn_block 128 with a 4-row MR tail, and an 18x18
    // plane split into 4-row blocks → N 72 then a 36 tail under nr 16.
    const int64_t oshape[] = {1, 40, 18, 18};

    for (bool with_bias : {true, false}) {
        const int64_t ishape[] = {1, 32, 18, 18};
        const int64_t wshape[] = {40, 32, 3, 3};
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
    // get_ic_block() picks icn_block from karea alone — 64 at 1x1, 16 through
    // 3x3, 8 through 5x5, 2 through 7x7, and 1 past that. One GEMM call reduces
    // over icn_block * karea, so the ladder exists to hold that K roughly flat
    // (in the 144-200 range) as the kernel area grows: a short K makes every
    // ic-block re-read and rewrite the whole output tile and restart the
    // packer, while a long one only costs col buffer.
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
        {1, 5, 16, 1},   // karea  5  — inside the band the 3x3 cases use
        {3, 3, 16, 1},   // karea  9  — that band's edge
        {3, 5,  8, 1},   // karea 15  — first rung down
        {5, 5,  8, 1},   // karea 25  — that band's edge
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

        const auto plan = nnops::backend::cpu::get_conv2d_implicit_gemm_plan(
            attrs, descs, out_descs, /*num_threads=*/1, /*use_thread_slots=*/false);

        // Every rung must stay on the im2col path — that is what makes the
        // icn_block expectation below a statement about get_ic_block rather than
        // about the pointwise shortcut.
        NNOPS_EXPECT_FALSE(plan.pointwise);
        NNOPS_EXPECT_TRUE(plan.workspace_size > 0);
        NNOPS_EXPECT_EQ(plan.icn_block, l.want_icn);
        // The gather writes the packed panels directly, so the fp route carries
        // no col matrix; what the kernel is handed is the packed-B block —
        // icn_block channels x karea taps x oh_block output rows x OW columns.
        NNOPS_EXPECT_EQ(plan.col_size, 0);
        NNOPS_EXPECT_EQ(plan.pack_size,
                        expected_pack_size(plan, l.kh * l.kw, out_descs[0].dims[3]));
    }
}

NNOPS_TEST(conv2d_fused_im2col_pack_matches_two_step) {
    // tiled_im2col_2d_packed must land every value on the same panel slot the
    // two-step route does (tiled_im2col_2d into [K][roi_area], then
    // tile_pack_rhs into nr_max-wide panels): same gather, same padding, same
    // zero-filled tail of the last panel. Both are pure re-orderings, so the
    // region the MMA reads must match bit for bit — on f32 and f16, and across
    // stride, dilation, padding and a roi that is not a whole number of panels.
    struct Case {
        int64_t ic, ih, iw, kh, kw, sh, sw, ph, pw, dh, dw;
        int64_t oh, ohc, ow;   // roi origin (output row), roi rows, roi width
    };
    const Case cases[] = {
        { 3,  9, 11, 3, 3, 1, 1, 1, 1, 1, 1, 0, 4,  9},  // padding on all sides
        { 2, 12, 10, 3, 3, 2, 2, 1, 1, 1, 1, 2, 4,  5},  // stride 2, interior roi
        { 2,  9, 13, 3, 3, 1, 1, 2, 2, 2, 2, 0, 4,  7},  // dilation 2, pad 2
        { 1,  7,  8, 5, 1, 1, 1, 0, 2, 1, 1, 3, 4,  8},  // 5x1 taps, top/left pad
        { 4,  6,  6, 1, 1, 1, 1, 0, 0, 1, 1, 0, 6,  6},  // 1x1 kernel
        { 5,  9,  9, 3, 3, 1, 1, 1, 1, 1, 1, 4, 5,  5},  // roi_area 25 → tail
        { 7,  8,  9, 3, 3, 1, 1, 1, 1, 1, 1, 0, 8,  9},  // 72 columns, 8 rows
    };

    for (const Case& c : cases) {
        const int64_t karea    = c.kh * c.kw;
        const int64_t kdim     = c.ic * karea;
        const int64_t roi_area = c.ohc * c.ow;
        const int64_t ih0 = c.oh * c.sh - c.ph;
        const int64_t iw0 = -c.pw;
        const int roi_pad_h = static_cast<int>(ih0 < 0 ? -ih0 : (ih0 >= c.ih ? ih0 - c.ih : 0));
        const int roi_pad_w = static_cast<int>(iw0 < 0 ? -iw0 : (iw0 >= c.iw ? iw0 - c.iw : 0));

        auto check = [&](auto tag) {
            using T = decltype(tag);
            constexpr int nr = nnops::backend::cpu::nr_max_flt<T>();
            const int ldd = static_cast<int>(
                nnops::backend::cpu::mma_pack_align(
                    static_cast<size_t>(nr) * kdim * sizeof(T)) / sizeof(T));
            const int64_t npanel = (roi_area + nr - 1) / nr;

            auto to_T = [](float v) -> T {
                if constexpr (std::is_same_v<T, float>) {
                    return v;
                } else {
                    return nnops::backend::cpu::float_to_half(v);
                }
            };

            // Small integers are exact in f16, so both dtypes hold the same
            // values and the comparison stays bit-exact.
            const int64_t in_n = c.ic * c.ih * c.iw;
            std::vector<T> in(static_cast<size_t>(in_n));
            for (int64_t i = 0; i < in_n; ++i) {
                in[static_cast<size_t>(i)] =
                    to_T(static_cast<float>((i * 37) % 61 - 30));
            }

            std::vector<T> col(static_cast<size_t>(kdim * roi_area));
            std::vector<T> want(static_cast<size_t>(npanel * ldd), to_T(-7.0f));
            std::vector<T> got(static_cast<size_t>(npanel * ldd), to_T(-7.0f));

            nnops::kernel::tiled_im2col_2d<T>(
                col.data(), in.data(), 0, static_cast<int>(c.oh),
                static_cast<int>(c.ow), static_cast<int>(c.ohc),
                roi_pad_h, roi_pad_w, static_cast<int>(c.ph), static_cast<int>(c.pw),
                static_cast<int>(c.ic), static_cast<int>(c.ih), static_cast<int>(c.iw),
                static_cast<int>(c.ih * c.iw), static_cast<int>(c.iw),
                static_cast<int>(c.kh), static_cast<int>(c.kw),
                static_cast<int>(c.sh), static_cast<int>(c.sw),
                static_cast<int>(c.dh), static_cast<int>(c.dw));
            nnops::backend::cpu::tile_pack_rhs(false, static_cast<int>(roi_area),
                                               static_cast<int>(kdim), want.data(), ldd,
                                               col.data(), static_cast<int>(roi_area), 1.0f);

            nnops::kernel::tiled_im2col_2d_packed<T, nr>(
                got.data(), ldd, in.data(), 0, static_cast<int>(c.oh),
                static_cast<int>(c.ow), static_cast<int>(c.ohc),
                roi_pad_h, roi_pad_w, static_cast<int>(c.ph), static_cast<int>(c.pw),
                static_cast<int>(c.ic), static_cast<int>(c.ih), static_cast<int>(c.iw),
                static_cast<int>(c.ih * c.iw), static_cast<int>(c.iw),
                static_cast<int>(c.kh), static_cast<int>(c.kw),
                static_cast<int>(c.sh), static_cast<int>(c.sw),
                static_cast<int>(c.dh), static_cast<int>(c.dw));

            // Only [0, nr*kdim) of each panel is ever read by the MMA; the
            // ldd tail is padding neither route has to initialize.
            for (int64_t p = 0; p < npanel; ++p) {
                NNOPS_EXPECT_TRUE(std::memcmp(want.data() + p * ldd,
                                              got.data() + p * ldd,
                                              sizeof(T) * nr * kdim) == 0);
            }
        };

        check(float{});
        check(half{});
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
        // Counter-case: 64 filters <= 3136 positions, so the expansion gate never
        // fires. The row count still moves off the filter-derived 4, but only to
        // what the col-budget growth loop allows (28 rows at 16 channels x 9 taps
        // x 56 columns -> 32 rows of col cap), not to the whole plane.
        {{1, 64, 56, 56}, {64, 64, 3, 3}, 3, 3, 1, 1, 1, 1, 1, 1, 28, 2, false},
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

        const auto plan = nnops::backend::cpu::get_conv2d_implicit_gemm_plan(
            attrs, descs, out_descs, c.nthreads, /*use_thread_slots=*/false);

        NNOPS_EXPECT_EQ(plan.oh_block, c.want_oh_block);
        NNOPS_EXPECT_EQ(plan.num_oh_blocks, c.want_blocks);
        // nthreads=1 with NG=1 leaves exactly one parallel task.
        if (c.nthreads == 1) {
            NNOPS_EXPECT_EQ(plan.oh_chunks, 1);
        }
        NNOPS_EXPECT_EQ(plan.pointwise, c.want_pointwise);
        // The fp route has no col matrix at all — the gather is fused with the
        // pack — so the scratch the kernel is handed is the packed-B block.
        NNOPS_EXPECT_EQ(plan.col_size, 0);
        NNOPS_EXPECT_EQ(plan.pack_size,
                        expected_pack_size(plan, c.kh * c.kw, out_descs[0].dims[3]));
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
        const auto plan = nnops::backend::cpu::get_conv2d_implicit_gemm_plan(
            attrs, descs, out_descs, /*num_threads=*/1, /*use_thread_slots=*/false);

        NNOPS_EXPECT_EQ(plan.pointwise, c.want_pointwise);
        // Neither route keeps a col matrix on fp: pointwise reads B off the
        // input plane, everything else gathers straight into the packed panels.
        NNOPS_EXPECT_EQ(plan.col_size, 0);
        NNOPS_EXPECT_EQ(plan.pack_size,
                        expected_pack_size(plan, c.kh * c.kw, out_descs[0].dims[3]));
        if (c.want_pointwise) {
            // One K pass over the whole group.
            NNOPS_EXPECT_EQ(plan.icn_block, c.ishape[1] / c.groups);
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
    const auto pplan = nnops::backend::cpu::get_conv2d_implicit_gemm_plan(
        attrs, pdescs, od, /*num_threads=*/1, /*use_thread_slots=*/false);
    NNOPS_EXPECT_FALSE(pplan.pointwise);
    // Still the 1x1 rung of the ic-block ladder, with the packed-B scratch back
    // (the gather handles the pitch; it writes the panels directly).
    NNOPS_EXPECT_EQ(pplan.icn_block, int64_t{64});
    NNOPS_EXPECT_EQ(pplan.col_size, 0);
    NNOPS_EXPECT_EQ(pplan.pack_size, expected_pack_size(pplan, 1, W));

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

// ============================================================
// Wide frames — 1280 x 720 / 960 (ResNet / VGG front layers)
// ============================================================

NNOPS_TEST(conv2d_wide_frame_plan) {
    // A 1280-wide frame is the first plane in the suite that is not square.
    // The plan's block heuristics are the same ones the 224 tests pin, but the
    // width makes every OW-dependent quantity (the col panel, the N side of the
    // GEMM, the nr panels per row) different from OH, and the row count turns
    // the oh walk into hundreds of blocks. These are the front layers of
    // ResNet50 / VGG16 at that frame size: the stem (7x7 s2, icn 2), VGG block1
    // (3x3 at full resolution, icn 16), and the ResNet 1x1 expand at a quarter
    // resolution, which takes the pointwise route with no col buffer at all.
    struct Case {
        int64_t N, IC, IH, IW, OC, K, S, PH, PW;
        int64_t want_oh, want_ow, want_icn, want_ocn, want_ohb, want_blocks;
        bool want_pointwise; int nt;
    };
    const Case cases[] = {
        // ResNet50 conv1 on 720p -> 360x640, 90 four-row blocks.
        {1,  3, 720, 1280,  64, 7, 2, 3, 3,  360,  640,  2,  64, 4,  90, false, 1},
        {1,  3, 720, 1280,  64, 7, 2, 3, 3,  360,  640,  2,  64, 4,  90, false, 4},
        // VGG16 block1 on 720p: the plane is unchanged, 180 blocks deep.
        {1, 64, 720, 1280,  64, 3, 1, 1, 1,  720, 1280, 16,  64, 4, 180, false, 1},
        // ... and on 960p, 240 blocks.
        {1, 64, 960, 1280,  64, 3, 1, 1, 1,  960, 1280, 16,  64, 4, 240, false, 4},
        // ResNet50 conv2_x expand on 720p: 1x1 p0 at a quarter resolution.
        {1, 64, 180,  320, 256, 1, 1, 0, 0,  180,  320, 64, 128, 4,  45, true,  4},
    };

    // Shape inference and the plan read descriptors only, so one placeholder
    // element stands in for the data pointer: a 960p VGG plane is 314 MB and
    // nothing here would ever touch it.
    float placeholder = 0.0f;
    for (const Case& c : cases) {
        const int64_t ishape[] = {c.N, c.IC, c.IH, c.IW};
        const int64_t wshape[] = {c.OC, c.IC, c.K, c.K};
        TensorView input(ishape, DataType::f32, &placeholder);
        TensorView weight(wshape, DataType::f32, &placeholder);

        Conv2DAttributes attrs;
        attrs.kernel_size = {c.K, c.K};
        attrs.stride      = {c.S, c.S};
        attrs.padding     = {c.PH, c.PW};

        auto op = Conv2D::create(attrs, Backend::CPU);
        const TensorDesc descs[] = {input.desc(), weight.desc()};
        const auto out_descs = op->getOutputTensorDesc(descs);
        const auto plan = nnops::backend::cpu::get_conv2d_implicit_gemm_plan(
            attrs, descs, out_descs, c.nt, /*use_thread_slots=*/false);

        NNOPS_EXPECT_EQ(out_descs[0].dims[2], c.want_oh);
        NNOPS_EXPECT_EQ(out_descs[0].dims[3], c.want_ow);
        NNOPS_EXPECT_EQ(plan.icn_block, c.want_icn);
        NNOPS_EXPECT_EQ(plan.ocn_block, c.want_ocn);
        NNOPS_EXPECT_EQ(plan.oh_block, c.want_ohb);
        NNOPS_EXPECT_EQ(plan.num_oh_blocks, c.want_blocks);
        NNOPS_EXPECT_EQ(plan.oh_chunks, std::min<int64_t>(c.nt, c.want_blocks));
        NNOPS_EXPECT_EQ(plan.pointwise, c.want_pointwise);
        // No col matrix on the fp route; the packed-B block the kernel is
        // handed is icn x karea taps x oh_block rows x the frame's output width.
        NNOPS_EXPECT_EQ(plan.col_size, 0);
        NNOPS_EXPECT_EQ(plan.pack_size,
                        expected_pack_size(plan, c.K * c.K, c.want_ow));
    }
}

NNOPS_TEST(conv2d_small_plane_oc_split_plan) {
    // A small output plane cannot fill a wide pool from its oh-blocks alone: a
    // 7x7 plane is two four-row blocks and that is the whole supply. The plan
    // therefore splits the filter rows as well, so the task grid is
    // (batch, group) x oh-chunks x oc-chunks. These are the tails of a 224
    // ResNet-18, where that supply is the difference between two working threads
    // and a full pool, plus the cases that must stay on the plane axis alone.
    struct Case {
        int64_t IC, IH, OC, K, S, nt;
        int64_t want_ocn, want_ohb, want_oc_chunks;
    };
    const Case cases[] = {
        // 14x14 plane, 512 filters: two oh-blocks, so the pool is starved and
        // the filter rows split until it is fed (20 threads / 2 tasks = 10).
        {256, 14, 512, 3, 2, 20, 128, 4, 10},
        // Same plane at 4 threads: two tasks already keep half the pool busy,
        // so the plane axis stays on its own.
        {256, 14, 512, 3, 2,  4, 128, 4, 1},
        // 7x7 plane, 512 filters, 20 threads: still two oh-blocks (the halving
        // floor is two N panels wide), so the filter axis takes the whole load.
        {512,  7, 512, 3, 1, 20, 128, 4, 10},
        // 56x56 plane: fourteen oh-chunks fill the pool on their own and a
        // filter split there would re-gather B for nothing.
        { 64, 56, 128, 3, 2, 20, 128, 2, 1},
        // One filter block cannot be split at all — 20 threads must make do.
        { 64, 56,  64, 3, 1, 20,  64, 4, 1},
        // A plane whose block count is ODD: 17 rows over two-row blocks is 9,
        // so 20 threads want 2 chunks (18 tasks, one wave) and the ceiling
        // would pick 3 (27 tasks, a second wave of seven that the barrier waits
        // on — 152 against 226 GFLOPS in bench_conv2d's odd-plane rows).
        { 64, 17,  64, 3, 1, 20,  64, 2, 2},
    };

    float placeholder = 0.0f;
    for (const Case& c : cases) {
        const int64_t ishape[] = {1, c.IC, c.IH, c.IH};
        const int64_t wshape[] = {c.OC, c.IC, c.K, c.K};
        TensorView input(ishape, DataType::f32, &placeholder);
        TensorView weight(wshape, DataType::f32, &placeholder);

        Conv2DAttributes attrs;
        attrs.kernel_size = {c.K, c.K};
        attrs.stride      = {c.S, c.S};
        attrs.padding     = {c.K / 2, c.K / 2};

        auto op = Conv2D::create(attrs, Backend::CPU);
        const TensorDesc descs[] = {input.desc(), weight.desc()};
        const auto out_descs = op->getOutputTensorDesc(descs);
        const auto plan = nnops::backend::cpu::get_conv2d_implicit_gemm_plan(
            attrs, descs, out_descs, c.nt, /*use_thread_slots=*/true);

        NNOPS_EXPECT_EQ(plan.ocn_block, c.want_ocn);
        NNOPS_EXPECT_EQ(plan.oh_block, c.want_ohb);
        NNOPS_EXPECT_EQ(plan.oc_chunks, c.want_oc_chunks);
        // Slots follow the task grid, which now counts the filter axis.
        const int64_t tasks = plan.oh_chunks * plan.oc_chunks;
        NNOPS_EXPECT_EQ(plan.num_slots, std::min<int64_t>(c.nt, tasks));
        // Every task must own at least one filter row: the even split floors.
        NNOPS_EXPECT_TRUE(c.OC / plan.oc_chunks >= 2 * 6);
    }
}

NNOPS_TEST(conv2d_plan_slots_follow_workers_not_items) {
    // A backend may hand its operators an over-subscribed task grid: the same
    // workers, told there are `items` times as many tasks as there are of them,
    // so a hybrid part's fast cores can claim the tail instead of idling
    // through it. Per-slot scratch must follow the *worker* count, not that
    // inflated figure — four times the arena for four times the items would
    // push the pack/accumulator scratch out of cache for nothing.
    const int64_t ishape[] = {1, 256, 14, 14};
    const int64_t wshape[] = {256, 256, 3, 3};
    float placeholder = 0.0f;
    TensorView input(ishape, DataType::f32, &placeholder);
    TensorView weight(wshape, DataType::f32, &placeholder);

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride      = {2, 2};
    attrs.padding     = {1, 1};

    auto op = Conv2D::create(attrs, Backend::CPU);
    const TensorDesc descs[] = {input.desc(), weight.desc()};
    const auto out_descs = op->getOutputTensorDesc(descs);
    const auto get = [&](int nt, int64_t slots) {
        return nnops::backend::cpu::get_conv2d_implicit_gemm_plan(
            attrs, descs, out_descs, nt, /*use_thread_slots=*/true, slots);
    };

    // Same reported thread count, so the same block sizes and the same grid:
    // only the slot count moved, and only the scratch follows it.
    const auto one_per_worker = get(20, 0);
    const auto five_workers   = get(20, 5);
    NNOPS_EXPECT_EQ(five_workers.oh_chunks, one_per_worker.oh_chunks);
    NNOPS_EXPECT_EQ(five_workers.oc_chunks, one_per_worker.oc_chunks);
    NNOPS_EXPECT_EQ(five_workers.num_slots, 5);
    NNOPS_EXPECT_TRUE(five_workers.workspace_size < one_per_worker.workspace_size);

    // A 4x grid over 20 workers: 80 reported threads, 20 slots — never 80.
    const auto over = get(80, 20);
    NNOPS_EXPECT_EQ(over.num_slots,
                    std::min<int64_t>(20, over.oh_chunks * over.oc_chunks));

    // No slot count given: the historical min(threads, tasks) rule.
    const auto per_task = get(20, 0);
    NNOPS_EXPECT_EQ(per_task.num_slots,
                    std::min<int64_t>(20, per_task.oh_chunks * per_task.oc_chunks));
}

NNOPS_TEST(conv2d_oversubscribed_grid_matches_serial) {
    // The over-subscribed grid is a scheduling choice, not a numeric one: a
    // worker claiming task 57 must produce exactly what the equal split did,
    // including on the per-slot scratch it reuses across tasks.
    const int64_t ishape[] = {1, 64, 28, 28};
    const int64_t wshape[] = {64, 64, 3, 3};
    auto [in_vec, input]   = test::make_random_tensor(ishape, -1.0f, 1.0f, 911);
    auto [w_vec, weight]   = test::make_random_tensor(wshape, -1.0f, 1.0f, 912);

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride      = {1, 1};
    attrs.padding     = {1, 1};

    std::vector<TensorDesc> descs = {input.desc(), weight.desc()};
    std::vector<TensorView> ins   = {input, weight};
    auto op = Conv2D::create(attrs, Backend::CPU);
    const auto out_descs = op->getOutputTensorDesc(descs);

    std::vector<float> ref_buf(static_cast<size_t>(out_descs[0].numel()), 0.0f);
    TensorView ref = test::make_planar(out_descs[0], ref_buf.data());
    NNOPS_EXPECT_EQ(op->compute(ref, ins, ComputeContext{}, nullptr), Status::ok);

    for (int nt : {2, 4, 8}) {
        std::vector<float> out_buf(static_cast<size_t>(out_descs[0].numel()), 0.0f);
        TensorView out = test::make_planar(out_descs[0], out_buf.data());
        ComputeContext ctx;
        ctx.cpu = nnops::openmp_backend(nt, nnops::OpenMPSplit{/*items_per_thread=*/4,
                                                              /*dynamic=*/true});
        NNOPS_EXPECT_EQ(op->compute(out, ins, ctx, nullptr), Status::ok);
        for (size_t i = 0; i < out_buf.size(); ++i) {
            NNOPS_EXPECT_NEAR(out_buf[i], ref_buf[i], 1e-5f);
        }
    }
}

NNOPS_TEST(conv2d_small_plane_oc_split_matches_ref) {
    // The filter split is only a partition of the oc loop, so every thread
    // configuration must reproduce the serial result — and the 20-thread plan
    // here really does split (ten filter chunks of a 14x14 plane).
    const int64_t ishape[] = {1, 32, 14, 14};
    const int64_t wshape[] = {64, 32, 3, 3};
    auto [in_vec, input]   = test::make_random_tensor(ishape, -1.0f, 1.0f, 901);
    auto [w_vec, weight]   = test::make_random_tensor(wshape, -1.0f, 1.0f, 902);

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride      = {2, 2};
    attrs.padding     = {1, 1};

    std::vector<TensorDesc> descs = {input.desc(), weight.desc()};
    std::vector<TensorView> ins   = {input, weight};
    auto op = Conv2D::create(attrs, Backend::CPU);
    const auto out_descs = op->getOutputTensorDesc(descs);
    const auto plan = nnops::backend::cpu::get_conv2d_implicit_gemm_plan(
        attrs, descs, out_descs, 20, /*use_thread_slots=*/true);
    NNOPS_EXPECT_TRUE(plan.oc_chunks > 1);

    std::vector<float> ref_buf(static_cast<size_t>(out_descs[0].numel()), 0.0f);
    TensorView ref = test::make_planar(out_descs[0], ref_buf.data());
    {
        NNOPS_EXPECT_EQ(op->compute(ref, ins, ComputeContext{}, nullptr), Status::ok);
    }

    for (int nt : {2, 4, 20}) {
        std::vector<float> out_buf(static_cast<size_t>(out_descs[0].numel()), 0.0f);
        TensorView out = test::make_planar(out_descs[0], out_buf.data());
        SimplePool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        NNOPS_EXPECT_EQ(op->compute(out, ins, ctx, nullptr), Status::ok);
        for (size_t i = 0; i < out_buf.size(); ++i) {
            NNOPS_EXPECT_NEAR(out_buf[i], ref_buf[i], 1e-5f);
        }
    }
}

NNOPS_TEST(conv2d_wide_frame_matches_ref) {
    // 1280x720 through a 3x3 layer and a 1x1 pointwise layer — VGG16's first
    // conv and ResNet50's conv2_x 1x1 at that frame size. The tiled path runs
    // the full plane (103 oh-blocks, 80 nr panels across a row); the reference
    // is run on a crop of the top rows and only those rows are compared, since
    // output row r of a 3x3 reads at most input row r+1 (a 66-row crop is exact
    // for the first 64) and a full-frame reference is not affordable. The full
    // output is checked for thread-identity: the chunks own disjoint row
    // ranges, so 4 threads must reproduce the serial result bit for bit — that
    // is where a per-chunk row offset or a missed halo would show.
    const int64_t ishape[] = {1, 3, 720, 1280};
    const int64_t wshape[] = {8, 3, 3, 3};
    const int64_t IH = ishape[2], IW = ishape[3];

    auto [in_vec, input] = test::make_random_tensor(ishape, -1.0f, 1.0f, 881);
    auto [w_vec, weight] = test::make_random_tensor(wshape, -1.0f, 1.0f, 882);

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride      = {1, 1};
    attrs.padding     = {1, 1};

    std::vector<TensorDesc> descs = {input.desc(), weight.desc()};
    std::vector<TensorView> ins   = {input, weight};
    auto op = Conv2D::create(attrs, Backend::CPU);
    auto out_descs = op->getOutputTensorDesc(descs);
    const int64_t OH = out_descs[0].dims[2], OW = out_descs[0].dims[3];
    const int64_t out_numel = out_descs[0].numel();

    // The wide walk must actually be the one under test. With IC 3 the single
    // ic-block is 3x9x1280 = 34560 col elements per row, so the 256K-element
    // col budget still allows 7 rows and the growth loop takes them (the same
    // plane at IC 64 is pinned to 4 — see conv2d_wide_frame_plan). Either way
    // the plane is walked in a hundred-odd blocks of a few rows.
    const auto plan = nnops::backend::cpu::get_conv2d_implicit_gemm_plan(attrs, descs, out_descs, 1, false);
    NNOPS_EXPECT_EQ(plan.oh_block, int64_t{7});
    NNOPS_EXPECT_EQ(plan.num_oh_blocks, int64_t{103});

    std::vector<float> serial_buf(static_cast<size_t>(out_numel));
    auto serial_out = test::make_planar(out_descs[0], serial_buf.data());
    op->compute(serial_out, ins, {}, nullptr);

    SimplePool pool(4);
    ComputeContext ctx;
    ctx.cpu = pool.cpu;
    std::vector<float> threaded_buf(static_cast<size_t>(out_numel));
    auto threaded_out = test::make_planar(out_descs[0], threaded_buf.data());
    op->compute(threaded_out, ins, ctx, nullptr);
    int64_t mismatches = 0;
    for (int64_t i = 0; i < out_numel; ++i) {
        if (serial_buf[static_cast<size_t>(i)] != threaded_buf[static_cast<size_t>(i)]) {
            ++mismatches;
        }
    }
    NNOPS_EXPECT_EQ(mismatches, int64_t{0});

    const int64_t OC = wshape[0];
    const int64_t crop_rows = 66, check_rows = 64;
    const int64_t cshape[] = {1, ishape[1], crop_rows, IW};
    std::vector<float> crop_buf(static_cast<size_t>(ishape[1] * crop_rows * IW));
    for (int64_t c = 0; c < ishape[1]; ++c) {
        std::memcpy(crop_buf.data() + static_cast<size_t>(c * crop_rows * IW),
                    in_vec.data() + static_cast<size_t>(c * IH * IW),
                    sizeof(float) * static_cast<size_t>(crop_rows * IW));
    }
    TensorView crop(cshape, DataType::f32, crop_buf.data());
    const int64_t rshape[] = {1, OC, crop_rows, IW};
    std::vector<float> ref_buf(static_cast<size_t>(OC * crop_rows * IW));
    TensorView ref_out(rshape, DataType::f32, ref_buf.data());
    const std::vector<TensorView> crop_ins = {crop, weight};
    nnops::backend::cpu::reference::conv2d_ref(attrs, ref_out, crop_ins, {}, nullptr);

    float worst = 0.0f;
    for (int64_t oc = 0; oc < OC; ++oc) {
        for (int64_t r = 0; r < check_rows; ++r) {
            const float* a = serial_buf.data() + static_cast<size_t>((oc * OH + r) * OW);
            const float* b = ref_buf.data() + static_cast<size_t>((oc * crop_rows + r) * IW);
            for (int64_t w = 0; w < OW; ++w) {
                worst = std::max(worst, std::abs(a[w] - b[w]));
            }
        }
    }
    std::printf("      wide frame 3x3 (1280x720): max deviation %.6f\n", worst);
    NNOPS_EXPECT_NEAR(worst, 0.0f, 1e-4f);

    // 1x1 pointwise on a 1280-wide plane: B comes straight off the input with
    // no col buffer, so the per-chunk row base is the only thing locating B.
    // There is no halo, so a 32-row crop is exact.
    {
        const int64_t pshape[] = {1, 8, 180, 1280};
        const int64_t pwshape[] = {16, 8, 1, 1};
        auto [p_in_vec, p_in] = test::make_random_tensor(pshape, -1.0f, 1.0f, 883);
        auto [p_w_vec, p_w]   = test::make_random_tensor(pwshape, -1.0f, 1.0f, 884);

        Conv2DAttributes pattrs;
        pattrs.kernel_size = {1, 1};
        std::vector<TensorDesc> pdescs = {p_in.desc(), p_w.desc()};
        std::vector<TensorView> pins   = {p_in, p_w};
        auto pop = Conv2D::create(pattrs, Backend::CPU);
        auto pouts = pop->getOutputTensorDesc(pdescs);
        NNOPS_EXPECT_TRUE(nnops::backend::cpu::get_conv2d_implicit_gemm_plan(
            pattrs, pdescs, pouts, 4, /*use_thread_slots=*/false).pointwise);

        const int64_t pn = pouts[0].numel();
        const int64_t POW = pouts[0].dims[3];
        std::vector<float> p_buf(static_cast<size_t>(pn));
        auto p_out = test::make_planar(pouts[0], p_buf.data());
        pop->compute(p_out, pins, {}, nullptr);

        std::vector<float> p_threaded(static_cast<size_t>(pn));
        auto p_tout = test::make_planar(pouts[0], p_threaded.data());
        pop->compute(p_tout, pins, ctx, nullptr);
        NNOPS_EXPECT_TRUE(p_buf == p_threaded);

        const int64_t p_crop = 32;
        const int64_t pc_shape[] = {1, 8, p_crop, 1280};
        std::vector<float> pc_buf(static_cast<size_t>(8 * p_crop * 1280));
        for (int64_t c = 0; c < 8; ++c) {
            std::memcpy(pc_buf.data() + static_cast<size_t>(c * p_crop * 1280),
                        p_in_vec.data() + static_cast<size_t>(c * 180 * 1280),
                        sizeof(float) * static_cast<size_t>(p_crop * 1280));
        }
        TensorView p_crop_in(pc_shape, DataType::f32, pc_buf.data());
        const int64_t pr_shape[] = {1, 16, p_crop, 1280};
        std::vector<float> pr_buf(static_cast<size_t>(16 * p_crop * 1280));
        TensorView p_ref(pr_shape, DataType::f32, pr_buf.data());
        const std::vector<TensorView> pc_ins = {p_crop_in, p_w};
        nnops::backend::cpu::reference::conv2d_ref(pattrs, p_ref, pc_ins, {}, nullptr);

        float p_worst = 0.0f;
        for (int64_t oc = 0; oc < 16; ++oc) {
            for (int64_t r = 0; r < p_crop; ++r) {
                const float* a = p_buf.data() + static_cast<size_t>((oc * 180 + r) * POW);
                const float* b = pr_buf.data() + static_cast<size_t>((oc * p_crop + r) * 1280);
                for (int64_t w = 0; w < POW; ++w) {
                    p_worst = std::max(p_worst, std::abs(a[w] - b[w]));
                }
            }
        }
        std::printf("      wide frame 1x1 (1280x180): max deviation %.6f\n", p_worst);
        NNOPS_EXPECT_NEAR(p_worst, 0.0f, 1e-4f);
    }
}

// ============================================================
// The prepacked filter — Conv2D::prepackWeights on the planar route
// ============================================================
//
// A planar float filter already *is* the GEMM's A operand, so its prepacked
// form is a densified [OC][IC/G·KH·KW] copy: what the two-step contract buys
// here is a total one — every supported filter has exactly one prepacked form,
// so a graph engine can prepack unconditionally and hand the blob to compute()
// on any route. What it must not change is a single output byte.

NNOPS_TEST(conv2d_prepacked_filter) {
    struct Case {
        int64_t N, IC, IH, IW, OC, K, S, PH, G;
        bool bias;
    };
    const Case cases[] = {
        // the basic gather, with a bias
        {1,  3, 16, 16,   8, 3, 1, 1, 1, true},
        // stride 2 on an odd plane
        {2,  4, 13, 15,   6, 3, 2, 1, 1, false},
        // 5x5 with groups
        {1,  8, 20, 20,   8, 5, 1, 2, 2, true},
        // 7x7 stem
        {1,  3, 32, 32,  16, 7, 2, 3, 1, false},
        // 1x1 pointwise: K is the whole channel count in one pass
        {1, 128, 8, 8,  192, 1, 1, 0, 1, true},
        // a ResNet stage plane
        {1, 64, 56, 56,  64, 3, 1, 1, 1, false},
    };

    for (size_t ci = 0; ci < std::size(cases); ++ci) {
        const Case& c = cases[ci];
        const int64_t ICG = c.IC / c.G;
        const int64_t ishape[] = {c.N, c.IC, c.IH, c.IW};
        const int64_t wshape[] = {c.OC, ICG, c.K, c.K};
        const int64_t bshape[] = {c.OC};
        const uint64_t seed = 900 + static_cast<uint64_t>(ci);

        auto [in_vec, input] = test::make_random_tensor(ishape, -1.0f, 1.0f, seed);
        auto [w_vec, weight] = test::make_random_tensor(wshape, -1.0f, 1.0f, seed + 1);
        std::vector<float> b_vec;
        TensorView bias;
        if (c.bias) {
            b_vec.resize(static_cast<size_t>(c.OC));
            test::XorShift128 rng(seed + 2);
            rng.fill_float(b_vec.data(), c.OC, -1.0f, 1.0f);
            bias = TensorView(bshape, DataType::f32, b_vec.data());
        }

        Conv2DAttributes attrs;
        attrs.kernel_size = {c.K, c.K};
        attrs.stride      = {c.S, c.S};
        attrs.padding     = {c.PH, c.PH};
        attrs.groups      = c.G;
        // Auto builds the Winograd U for a k3s1 fp filter; the im2col panels
        // are what this route's prepack produces, so say which one is meant.
        attrs.algorithm   = ConvAlgorithm::ImplicitGemm;

        auto op = Conv2D::create(attrs, Backend::CPU);

        std::vector<TensorView> ins = {input, weight};
        if (c.bias) { ins.push_back(bias); }
        std::vector<TensorDesc> descs;
        for (const auto& v : ins) { descs.push_back(v.desc()); }
        const auto od = op->getOutputTensorDesc(descs);
        const int64_t out_n = od[0].numel();

        // The raw filter's output is the reference.
        std::vector<float> raw_buf(static_cast<size_t>(out_n));
        auto raw_out = test::make_planar(od[0], raw_buf.data());
        NNOPS_EXPECT_EQ(op->compute(raw_out, ins, {}, nullptr), Status::ok);

        // Query the packed descriptor, allocate what it asks for, fill it.
        TensorView q;
        const TensorView wq[] = {weight};
        TensorView oq[] = {q};
        op->prepackWeights(wq, oq);
        q = oq[0];
        NNOPS_EXPECT_TRUE(q.rank() > 0);
        NNOPS_EXPECT_EQ(q.layout(), TensorLayout::PackedOIHW);
        NNOPS_EXPECT_EQ(q.shape(0), c.OC);
        // Dense [OC][IC/G·KH·KW]: the copy is the whole pack, and the buffer
        // the caller allocates is exactly that matrix.
        NNOPS_EXPECT_EQ(q.shape(1), ICG * c.K * c.K);
        // numel() is the buffer size a caller sizes its allocation from.
        NNOPS_EXPECT_EQ(q.desc().numel(), q.shape(0) * q.shape(1));

        std::vector<float> blob(static_cast<size_t>(q.storage_bytes() / sizeof(float)));
        TensorView packed(q.shape_span(), q.data_type(), blob.data(), q.pitch(),
                          q.layout());
        const TensorView wf[] = {weight};
        TensorView of[] = {packed};
        op->prepackWeights(wf, of);

        std::vector<TensorView> packed_ins = ins;
        packed_ins[1] = packed;
        std::vector<float> packed_buf(static_cast<size_t>(out_n));
        auto packed_out = test::make_planar(od[0], packed_buf.data());
        NNOPS_EXPECT_EQ(op->compute(packed_out, packed_ins, {}, nullptr), Status::ok);

        NNOPS_EXPECT_TRUE(std::memcmp(raw_buf.data(), packed_buf.data(),
                                      static_cast<size_t>(out_n) * sizeof(float)) == 0);
    }
}
