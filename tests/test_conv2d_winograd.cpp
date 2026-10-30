/// @file test_conv2d_winograd.cpp
/// @brief Unit tests for the Winograd F(2×2, 3×3) Conv2D path (k3s1).
///
/// The three transforms are checked against naive scalar references written
/// longhand from the F(2,3) definition; then the prepack contract (query
/// metadata, explicit pitch) and the plan's admission and filter-row split;
/// then end-to-end Winograd vs conv2d_ref on the *raw* weight, f32 and f16,
/// plus winograd_applicable's policy and a serial-vs-pooled run.
///
/// Winograd and direct convolution are algebraically equal but not bit-identical
/// (the transform reorders the sums and, at f16, adds a rounding step), so every
/// comparison against conv2d_ref is tolerance-based. The serial-vs-pooled check
/// is exact instead: same per-element operation order, different task split.

#include "nnops/ops/conv2d.hpp"
#include "backend/cpu/conv/conv2d_winograd.h"
#include "backend/cpu/simd_kernel/simd_winograd.hpp"
#include "common/test_harness.hpp"
#include "common/cpu_pool.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

using namespace nnops;
using nnops::backend::cpu::half;

// Reference kernel, test-local as in test_conv2d.cpp / test_matmul.cpp (it is
// not exported through a header).
namespace nnops::backend::cpu::reference {
void conv2d_ref(const Conv2DAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* workspace);
}

// ============================================================
// Transform matrices, restated independently of the kernel
// ============================================================
//
// Bᵀ is applied as (d·B) then (Bᵀ·e); Aᵀ as (Aᵀ·m) then (·A); G as written.
// Longhand, so a typo in simd_winograd.hpp cannot hide behind a shared helper.

static void naive_btdb(const float d[4][4], float v[4][4]) {
    static const float Bt[4][4] = {
        {1, 0, -1, 0}, {0, 1, 1, 0}, {0, -1, 1, 0}, {0, 1, 0, -1}};
    float e[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) { s += d[i][k] * Bt[j][k]; }  // d·B
            e[i][j] = s;
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) { s += Bt[i][k] * e[k][j]; }
            v[i][j] = s;
        }
    }
}

static void naive_atma(const float m[4][4], float y[2][2]) {
    static const float At[2][4] = {{1, 1, 1, 0}, {0, 1, -1, -1}};
    float t[2][4];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) { s += At[i][k] * m[k][j]; }
            t[i][j] = s;
        }
    }
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) { s += t[i][k] * At[j][k]; }
            y[i][j] = s;
        }
    }
}

static void naive_gggt(const float g[3][3], float u[4][4]) {
    static const float G[4][3] = {
        {1.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f},
        {0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}};
    float t[4][3];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 3; ++k) { s += G[i][k] * g[k][j]; }
            t[i][j] = s;
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 3; ++k) { s += t[i][k] * G[j][k]; }  // ·Gᵀ
            u[i][j] = s;
        }
    }
}

// ============================================================
// Transform unit tests
// ============================================================

NNOPS_TEST(wino_weight_transform_matches_naive) {
    test::XorShift128 rng(0xC0FFEE);
    for (int trial = 0; trial < 20; ++trial) {
        float g[3][3], ubuf[4][4];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) { g[i][j] = rng.next_float(-2.0f, 2.0f); }
        }
        nnops::kernel::winograd_weight_transform(&g[0][0], 3, &ubuf[0][0], 4);
        float want[4][4];
        naive_gggt(g, want);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                NNOPS_EXPECT_NEAR(ubuf[i][j], want[i][j], 1e-5f);
            }
        }
    }
}

NNOPS_TEST(wino_weight_transform_plane_strides) {
    // Same transform, but with the p = i·4 + j position scattered into 16
    // planes the way prepack stores U (ld_u = 4·plane, ld_c = plane).
    const int plane = 3;  // pretend OC·ICG == 3
    const int ld_u = 4 * plane;
    const int ld_c = plane;
    test::XorShift128 rng(42);
    float g[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) { g[i][j] = rng.next_float(-1.0f, 1.0f); }
    }
    std::vector<float> u(16 * plane, -99.0f);
    nnops::kernel::winograd_weight_transform(&g[0][0], 3, u.data(), ld_u, ld_c);

    float want[4][4];
    naive_gggt(g, want);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const int p = i * 4 + j;
            NNOPS_EXPECT_NEAR(u[p * plane + 0], want[i][j], 1e-5f);
        }
    }
}

NNOPS_TEST(wino_input_transform_matches_naive) {
    // 34 columns = the minimum halo width for TW_pad = 16 (the offset-2 loads
    // touch 2·(TW_pad−8) + 17 = 33).
    constexpr int kHaloW = 34;
    constexpr int kHaloH = 8;
    constexpr int kTiles = 8;
    std::vector<float> halo(kHaloW * kHaloH);
    test::XorShift128 rng(7);
    for (auto& x : halo) { x = rng.next_float(-3.0f, 3.0f); }

    // Two column groups, tw0 = 0 and 8 (pw0 = 0 and 16); the second one's loads
    // reach the end of the padded row.
    for (int tw0 : {0, 8}) {
        std::vector<float> v(16 * kTiles, -99.0f);
        nnops::kernel::winograd_input_tile8<float>(halo.data(), kHaloW,
                                                   2, 2 * tw0, v.data(), kTiles);
        for (int t = 0; t < kTiles; ++t) {
            float d[4][4];
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    d[r][c] = halo[(2 + r) * kHaloW + (2 * tw0 + 2 * t + c)];
                }
            }
            float want[4][4];
            naive_btdb(d, want);
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    NNOPS_EXPECT_NEAR(v[(i * 4 + j) * kTiles + t], want[i][j], 1e-4f);
                }
            }
        }
    }
}

NNOPS_TEST(wino_output_transform_matches_naive) {
    constexpr int kOcn = 2;
    constexpr int kTb = 8;
    constexpr int kOutW = 32;

    std::vector<float> m(16 * kOcn * kTb);
    test::XorShift128 rng(11);
    for (auto& x : m) { x = rng.next_float(-2.0f, 2.0f); }

    std::vector<float> out0(kOutW, -1.0f), out1(kOutW, -1.0f);
    const int oc = 1;
    const Epilogue ep{};
    nnops::kernel::winograd_output_tile8<float>(m.data(), kOcn * kTb, kTb, oc, 0,
                                                out0.data(), out1.data(),
                                                0, kOutW, true, ep, 0.0f, false);

    for (int t = 0; t < kTb; ++t) {
        float mm[4][4];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                mm[i][j] = m[(i * 4 + j) * kOcn * kTb + oc * kTb + t];
            }
        }
        float want[2][2];
        naive_atma(mm, want);
        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < 2; ++c) {
                const float got = (r == 0 ? out0 : out1)[2 * t + c];
                NNOPS_EXPECT_NEAR(got, want[r][c], 1e-4f);
            }
        }
    }
}

NNOPS_TEST(wino_output_transform_applies_bias_and_masks) {
    constexpr int kOcn = 1;
    constexpr int kTb = 8;
    constexpr int kOutW = 10;  // < 2·kTb: the tail of the last group must be masked

    std::vector<float> m(16 * kOcn * kTb, 0.0f);
    std::vector<float> out0(kOutW, 5.0f), out1(kOutW, 5.0f);
    const Epilogue ep{};
    nnops::kernel::winograd_output_tile8<float>(m.data(), kOcn * kTb, kTb, 0, 0,
                                                out0.data(), out1.data(),
                                                0, kOutW, /*row1_ok=*/false,
                                                ep, 2.0f, /*add_to=*/false);

    // row1_ok=false leaves out1 untouched (5.0); out0 takes the bias on every
    // column below OW and nothing past it.
    for (int i = 0; i < kOutW; ++i) { NNOPS_EXPECT_NEAR(out0[i], 2.0f, 1e-6f); }
    for (int i = 0; i < kOutW; ++i) { NNOPS_EXPECT_NEAR(out1[i], 5.0f, 1e-6f); }
}

// ============================================================
// Prepack contract
// ============================================================

NNOPS_TEST(wino_prepack_query) {
    const int64_t wshape[] = {5, 3, 3, 3};  // OC=5, IC/G=3, 3×3
    std::vector<float> w_buf(5 * 3 * 9, 0.25f);
    TensorView weight(wshape, DataType::f32, w_buf.data());

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride = {1, 1};
    attrs.algorithm = ConvAlgorithm::Winograd;
    auto op = Conv2D::create(attrs, Backend::CPU);

    TensorView empty;
    TensorView packed;
    {
        const TensorView w_arr[] = {weight};
        TensorView out_arr[] = {empty};
        op->prepackWeights(w_arr, out_arr);
        packed = out_arr[0];
    }

    NNOPS_EXPECT_EQ(packed.rank(), 3);
    NNOPS_EXPECT_EQ(packed.shape(0), 16);
    NNOPS_EXPECT_EQ(packed.shape(1), 5);
    NNOPS_EXPECT_EQ(packed.shape(2), 3);
    NNOPS_EXPECT_EQ(packed.data_type(), DataType::f32);
    NNOPS_EXPECT_EQ(static_cast<int64_t>(packed.layout()),
                    static_cast<int64_t>(TensorLayout::PackedWinoWeight));
    // Explicit pitch — the packed weight is *not* channel-packed, so the
    // default 32 B row alignment would not be IC/G × elem_size.
    NNOPS_EXPECT_EQ(packed.pitch(), 3 * static_cast<int64_t>(sizeof(float)));
}

NNOPS_TEST(wino_plan_gates_on_geometry) {
    // The plan is the single gate for "is this a Winograd configuration": a
    // zeroed plan (workspace_size 0) means there is nothing to run. That is the
    // Release-build observable — the NNOPS_ASSERTs compile out under NDEBUG.
    const int64_t ishape[] = {1, 4, 8, 8};
    const int64_t oshape[] = {1, 6, 6, 6};
    const int64_t ushape[] = {16, 6, 4};
    std::vector<float> in_buf(1 * 4 * 8 * 8, 0.0f);
    std::vector<float> out_buf(1 * 6 * 6 * 6, 0.0f);
    std::vector<float> u_buf(16 * 6 * 4, 0.0f);
    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView output(oshape, DataType::f32, out_buf.data());
    TensorView u(ushape, DataType::f32, u_buf.data(), 4 * static_cast<int64_t>(sizeof(float)),
                 TensorLayout::PackedWinoWeight);

    const TensorDesc ins[] = {input.desc(), u.desc()};
    const TensorDesc outs[] = {output.desc()};

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride = {1, 1};
    attrs.padding = {1, 1};

    NNOPS_EXPECT_TRUE(nnops::backend::cpu::get_conv2d_winograd_plan(
                          attrs, ins, outs, 1, false).workspace_size > 0);

    Conv2DAttributes k5 = attrs;
    k5.kernel_size = {5, 5};
    NNOPS_EXPECT_EQ(nnops::backend::cpu::get_conv2d_winograd_plan(
                        k5, ins, outs, 1, false).workspace_size, 0u);

    Conv2DAttributes s2 = attrs;
    s2.stride = {2, 2};
    NNOPS_EXPECT_EQ(nnops::backend::cpu::get_conv2d_winograd_plan(
                        s2, ins, outs, 1, false).workspace_size, 0u);

    Conv2DAttributes d2 = attrs;
    d2.dilation = {2, 2};
    NNOPS_EXPECT_EQ(nnops::backend::cpu::get_conv2d_winograd_plan(
                        d2, ins, outs, 1, false).workspace_size, 0u);
}

NNOPS_TEST(wino_plan_splits_the_filter_rows) {
    // A 14x14 plane is 7 tile rows and one 8-wide tile block — 7 tasks, and the
    // tile axis has nothing left to give. The plan grows a second axis on the
    // filter rows: the nearest whole number of chunks to nt/plane_tasks, floored
    // at two M-panels of work each so the repeated halo + input transform stays
    // amortized.
    std::vector<float> in_buf(256 * 14 * 14, 0.0f);
    std::vector<float> out_buf(256 * 14 * 14, 0.0f);
    std::vector<float> u_buf(16 * 256 * 256, 0.0f);
    const int64_t ishape[] = {1, 256, 14, 14};
    const int64_t oshape[] = {1, 256, 14, 14};
    const int64_t ushape[] = {16, 256, 256};
    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView output(oshape, DataType::f32, out_buf.data());
    TensorView u(ushape, DataType::f32, u_buf.data(),
                 256 * static_cast<int64_t>(sizeof(float)),
                 TensorLayout::PackedWinoWeight);
    const TensorDesc ins[] = {input.desc(), u.desc()};
    const TensorDesc outs[] = {output.desc()};

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride = {1, 1};
    attrs.padding = {1, 1};
    const auto plan = [&](int nt) {
        return nnops::backend::cpu::get_conv2d_winograd_plan(attrs, ins, outs, nt, true);
    };

    // TH 7 · one tile block = 7 plane tasks. Rounding, not ceiling: 16 threads
    // want 2 chunks (14 tasks, one wave), and 3 would run a straggler.
    struct Row { int nt; int64_t chunks; };
    for (const Row& r : {Row{1, 1}, Row{4, 1}, Row{8, 1}, Row{12, 2},
                         Row{14, 2}, Row{16, 2}, Row{20, 3}, Row{28, 4}}) {
        const auto p = plan(r.nt);
        NNOPS_EXPECT_EQ(p.oc_chunks, r.chunks);
        NNOPS_EXPECT_EQ(p.num_tasks, 7 * r.chunks);
        NNOPS_EXPECT_EQ(p.num_slots, std::min<int64_t>(r.nt, p.num_tasks));
        // Every task keeps at least two M-panels (6 rows each on x86_64 f32).
        NNOPS_EXPECT_TRUE(256 / p.oc_chunks >= 2 * 6);
    }

    // A plane that already fills the pool keeps the single dispatch: 56 tasks.
    std::vector<float> in56(64 * 56 * 56, 0.0f);
    std::vector<float> out56(64 * 56 * 56, 0.0f);
    std::vector<float> u56(16 * 64 * 64, 0.0f);
    const int64_t i56[] = {1, 64, 56, 56};
    const int64_t o56[] = {1, 64, 56, 56};
    const int64_t u56s[] = {16, 64, 64};
    TensorView in_v(i56, DataType::f32, in56.data());
    TensorView out_v(o56, DataType::f32, out56.data());
    TensorView u_v(u56s, DataType::f32, u56.data(),
                   64 * static_cast<int64_t>(sizeof(float)),
                   TensorLayout::PackedWinoWeight);
    const TensorDesc ins56[] = {in_v.desc(), u_v.desc()};
    const TensorDesc outs56[] = {out_v.desc()};
    const auto wide = nnops::backend::cpu::get_conv2d_winograd_plan(
        attrs, ins56, outs56, 20, true);
    NNOPS_EXPECT_EQ(wide.oc_chunks, int64_t{1});
    NNOPS_EXPECT_EQ(wide.num_tasks, int64_t{56});

    // And the floor stops it: 16 filter rows cannot be cut into two 12-row
    // tasks, so the starved 7-task grid stays as it is.
    std::vector<float> in16(16 * 14 * 14, 0.0f);
    std::vector<float> out16(16 * 14 * 14, 0.0f);
    std::vector<float> u16(16 * 16 * 16, 0.0f);
    const int64_t i16[] = {1, 16, 14, 14};
    const int64_t o16[] = {1, 16, 14, 14};
    const int64_t u16s[] = {16, 16, 16};
    TensorView i16v(i16, DataType::f32, in16.data());
    TensorView o16v(o16, DataType::f32, out16.data());
    TensorView u16v(u16s, DataType::f32, u16.data(),
                    16 * static_cast<int64_t>(sizeof(float)),
                    TensorLayout::PackedWinoWeight);
    const TensorDesc ins16[] = {i16v.desc(), u16v.desc()};
    const TensorDesc outs16[] = {o16v.desc()};
    const auto narrow = nnops::backend::cpu::get_conv2d_winograd_plan(
        attrs, ins16, outs16, 20, true);
    NNOPS_EXPECT_EQ(narrow.oc_chunks, int64_t{1});
    NNOPS_EXPECT_EQ(narrow.num_tasks, int64_t{7});
}

// ============================================================
// End-to-end: Winograd vs conv2d_ref
// ============================================================

namespace {

// Minimal fixed-size thread pool exposing a CpuBackend (as in test_conv2d.cpp).
struct SimplePool {
    explicit SimplePool(int nthreads) : nthreads_(nthreads) {
        // OpenMP when the build has it (common/cpu_pool.hpp).
        if (nnops::test::bench_use_openmp()) {
            cpu = nnops::openmp_backend(nthreads);
            return;
        }
        cpu.parallel_for = [this](int64_t begin, int64_t end, const ParallelForBody& body) {
            this->run(begin, end, body);
        };
        cpu.num_threads = [this]() { return nthreads_; };
        cpu.thread_id = []() { return current_thread_id_; };
    }

    void run(int64_t begin, int64_t end, const ParallelForBody& body) {
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

int64_t desc_numel(const TensorDesc& d) {
    int64_t n = 1;
    for (int64_t i = 0; i < d.rank; ++i) { n *= d.dims[static_cast<size_t>(i)]; }
    return n;
}

/// Run the Winograd path (prepacking U from the raw weight) and conv2d_ref on
/// that same raw weight; require agreement within (rtol, atol).
///
/// f32 only — f16 has its own check below.
void wino_vs_ref_f32(std::initializer_list<int64_t> in_shape,
                     std::initializer_list<int64_t> w_shape,
                     Conv2DAttributes attrs, bool has_bias,
                     float fill_out = 0.0f,
                     const ComputeContext* ctx = nullptr,
                     float rtol = 1e-4f, float atol = 1e-4f,
                     ConvAlgorithm algorithm = ConvAlgorithm::Winograd)
{
    auto [in_buf, in_view] = test::make_random_tensor(
        std::span<const int64_t>(in_shape.begin(), in_shape.size()), -1.0f, 1.0f, 101);
    auto [w_buf, w_view] = test::make_random_tensor(
        std::span<const int64_t>(w_shape.begin(), w_shape.size()), -0.5f, 0.5f, 202);

    std::vector<float> b_buf;
    TensorView bias_view;
    if (has_bias) {
        b_buf.resize(static_cast<size_t>(w_shape.begin()[0]));
        test::XorShift128 rng(303);
        for (auto& x : b_buf) { x = rng.next_float(-0.5f, 0.5f); }
        const int64_t bshape[] = {static_cast<int64_t>(b_buf.size())};
        bias_view = TensorView(bshape, DataType::f32, b_buf.data());
    }

    std::vector<TensorDesc> descs = {in_view.desc(), w_view.desc()};
    if (has_bias) { descs.push_back(bias_view.desc()); }

    attrs.algorithm = algorithm;
    auto op = Conv2D::create(attrs, Backend::CPU);
    const auto outs = op->getOutputTensorDesc(descs);
    const int64_t on = desc_numel(outs[0]);

    std::vector<float> out_wino(static_cast<size_t>(on), fill_out);
    auto out_view = test::make_planar(outs[0], out_wino.data());

    // Two-step prepack: query the shape with an empty view, then fill it.
    std::vector<float> u_buf;
    TensorView u_view;
    {
        TensorView empty;
        const TensorView w_in[] = {w_view};
        TensorView o[] = {empty};
        op->prepackWeights(w_in, o);
        const TensorView q = o[0];
        u_buf.assign(static_cast<size_t>(q.storage_bytes() / sizeof(float)), 0.0f);
        u_view = TensorView(q.shape_span(), DataType::f32, u_buf.data(),
                            q.pitch(), TensorLayout::PackedWinoWeight);
        const TensorView w_in2[] = {w_view};
        TensorView o2[] = {u_view};
        op->prepackWeights(w_in2, o2);
    }

    std::vector<TensorView> ins = {in_view, u_view};
    if (has_bias) { ins.push_back(bias_view); }
    const ComputeContext default_ctx{};
    op->compute(out_view, ins, ctx ? *ctx : default_ctx);

    // Reference: conv2d_ref on the raw weight.
    std::vector<float> out_ref(static_cast<size_t>(on), fill_out);
    auto ref_view = test::make_planar(outs[0], out_ref.data());
    std::vector<TensorView> rins = {in_view, w_view};
    if (has_bias) { rins.push_back(bias_view); }
    nnops::backend::cpu::reference::conv2d_ref(attrs, ref_view, rins,
                                               ctx ? *ctx : default_ctx, nullptr);

    for (int64_t i = 0; i < on; ++i) {
        const float a = out_wino[static_cast<size_t>(i)];
        const float b = out_ref[static_cast<size_t>(i)];
        const float tol = atol + rtol * std::max(std::abs(a), std::abs(b));
        if (std::abs(a - b) > tol) {
            NNOPS_EXPECT_NEAR(a, b, tol);
        }
    }
}

}  // anonymous namespace

// ---- selection policy and the Auto algorithm ------------------------------

NNOPS_TEST(wino_applicable_thresholds) {
    Conv2DAttributes a;
    a.kernel_size = {3, 3};
    a.padding = {1, 1};

    // Every condition met: k3s1, 32 channels, a 14x14 input.
    NNOPS_EXPECT_TRUE(winograd_applicable(a, 32, 14, 14));
    NNOPS_EXPECT_TRUE(winograd_applicable(a, 512, 56, 56));

    NNOPS_EXPECT_FALSE(winograd_applicable(a, 31, 14, 14));  // too few channels
    NNOPS_EXPECT_FALSE(winograd_applicable(a, 64, 13, 14));  // too short
    NNOPS_EXPECT_FALSE(winograd_applicable(a, 64, 14, 13));  // too narrow

    Conv2DAttributes k1 = a;
    k1.kernel_size = {1, 1};
    NNOPS_EXPECT_FALSE(winograd_applicable(k1, 64, 28, 28));

    Conv2DAttributes s2 = a;
    s2.stride = {2, 2};
    NNOPS_EXPECT_FALSE(winograd_applicable(s2, 64, 28, 28));

    Conv2DAttributes d2 = a;
    d2.dilation = {2, 2};
    NNOPS_EXPECT_FALSE(winograd_applicable(d2, 64, 28, 28));

    // The caller's off switch: ImplicitGemm keeps the prepack from happening.
    Conv2DAttributes off = a;
    off.algorithm = ConvAlgorithm::ImplicitGemm;
    NNOPS_EXPECT_FALSE(winograd_applicable(off, 64, 28, 28));
}

NNOPS_TEST(wino_auto_follows_the_weight) {
    // Auto is the default: handing it the prepacked U is what selects Winograd.
    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride = {1, 1};
    attrs.padding = {1, 1};
    NNOPS_EXPECT_TRUE(attrs.algorithm == ConvAlgorithm::Auto);
    wino_vs_ref_f32({1, 4, 8, 8}, {6, 4, 3, 3}, attrs, /*has_bias=*/false,
                    0.0f, nullptr, 1e-4f, 1e-4f, ConvAlgorithm::Auto);
}

NNOPS_TEST(wino_matches_ref_plain) {
    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride = {1, 1};
    attrs.padding = {0, 0};
    wino_vs_ref_f32({1, 4, 8, 8}, {6, 4, 3, 3}, attrs, /*has_bias=*/false);
}

NNOPS_TEST(wino_matches_ref_pad1) {
    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride = {1, 1};
    attrs.padding = {1, 1};
    wino_vs_ref_f32({1, 3, 9, 7}, {5, 3, 3, 3}, attrs, /*has_bias=*/false);
}

NNOPS_TEST(wino_matches_ref_wide) {
    // 14x18 output: 7 tile rows over a 16-wide tile block (two column groups),
    // with bias.
    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride = {1, 1};
    attrs.padding = {1, 1};
    wino_vs_ref_f32({1, 16, 14, 18}, {20, 16, 3, 3}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(wino_matches_ref_groups) {
    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride = {1, 1};
    attrs.padding = {1, 1};
    attrs.groups = 2;
    wino_vs_ref_f32({1, 6, 7, 7}, {8, 3, 3, 3}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(wino_matches_ref_odd_output) {
    // OH and OW both odd: the last tile row's second output row and the last
    // tile's second column are out of range.
    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride = {1, 1};
    attrs.padding = {0, 0};
    wino_vs_ref_f32({1, 4, 9, 9}, {4, 4, 3, 3}, attrs, /*has_bias=*/false);
}

NNOPS_TEST(wino_matches_ref_epilogue) {
    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride = {1, 1};
    attrs.padding = {1, 1};
    attrs.epilogue.type = EpilogueActivateType::Relu6;
    wino_vs_ref_f32({1, 5, 8, 8}, {7, 5, 3, 3}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(wino_matches_ref_add_to) {
    // add_to with a non-identity epilogue: the residual is the pre-filled
    // output, added *after* the activation, so fill_out is part of the answer.
    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride = {1, 1};
    attrs.padding = {1, 1};
    attrs.add_to = true;
    attrs.epilogue.type = EpilogueActivateType::Relu;
    wino_vs_ref_f32({1, 4, 6, 6}, {6, 4, 3, 3}, attrs, /*has_bias=*/false,
                    /*fill_out=*/0.75f);
}

NNOPS_TEST(wino_matches_ref_add_to_identity) {
    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride = {1, 1};
    attrs.padding = {1, 1};
    attrs.add_to = true;
    wino_vs_ref_f32({1, 4, 6, 6}, {6, 4, 3, 3}, attrs, /*has_bias=*/true,
                    /*fill_out=*/-0.3f);
}

NNOPS_TEST(wino_matches_ref_batch) {
    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride = {1, 1};
    attrs.padding = {1, 1};
    wino_vs_ref_f32({3, 4, 6, 6}, {6, 4, 3, 3}, attrs, /*has_bias=*/false);
}

NNOPS_TEST(wino_matches_ref_threaded) {
    // Through a 4-thread pool — exercises the per-thread slot indexing
    // (thread_slot_count() > 0) and the disjointness of tile rows and blocks.
    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride = {1, 1};
    attrs.padding = {1, 1};
    SimplePool pool(4);
    ComputeContext ctx;
    ctx.cpu = pool.cpu;
    wino_vs_ref_f32({1, 8, 12, 12}, {8, 8, 3, 3}, attrs, /*has_bias=*/true,
                    /*fill_out=*/0.0f, &ctx);
}

NNOPS_TEST(wino_oc_split_matches_ref) {
    // A 6x6 output plane is 3 tile rows and one tile block, so a 6-thread pool
    // splits the filter rows in two and the tasks write disjoint oc ranges —
    // the shape that catches a mis-sliced range (an `ocnc` clipped to one block
    // shows up as the back half of the filter never being applied).
    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride = {1, 1};
    attrs.padding = {1, 1};
    SimplePool pool(6);
    ComputeContext ctx;
    ctx.cpu = pool.cpu;
    wino_vs_ref_f32({1, 64, 6, 6}, {64, 64, 3, 3}, attrs, /*has_bias=*/true,
                    /*fill_out=*/0.0f, &ctx);

    // Same shape through 12 threads: the chunk count moves (2 -> 4), the answer
    // does not.
    SimplePool pool12(12);
    ComputeContext ctx12;
    ctx12.cpu = pool12.cpu;
    wino_vs_ref_f32({1, 64, 6, 6}, {64, 64, 3, 3}, attrs, /*has_bias=*/true,
                    /*fill_out=*/0.0f, &ctx12);
}

NNOPS_TEST(wino_threads_bit_identical_batch_groups) {
    // Batch and group are both part of the task grid, and the pooled cases above
    // keep them at 1 — here tasks of *different* (batch, group) run at the same
    // time, so a shared scratch buffer would show up. Each element keeps its
    // operation order, so the two results must be bit-identical.
    auto [in_buf, in_view] = test::make_random_tensor({2, 8, 14, 16}, -1.0f, 1.0f, 71);
    auto [w_buf, w_view] = test::make_random_tensor({8, 4, 3, 3}, -0.5f, 0.5f, 72);
    auto [b_buf, b_view] = test::make_random_tensor({8}, -0.5f, 0.5f, 73);

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride = {1, 1};
    attrs.padding = {1, 1};
    attrs.groups = 2;
    attrs.algorithm = ConvAlgorithm::Winograd;

    auto op = Conv2D::create(attrs, Backend::CPU);
    const TensorDesc descs[] = {in_view.desc(), w_view.desc(), b_view.desc()};
    const auto outs = op->getOutputTensorDesc(descs);
    const int64_t on = desc_numel(outs[0]);

    // Prepack U once; both runs read the same transformed weight.
    std::vector<float> u_buf;
    TensorView u_view;
    {
        TensorView empty;
        const TensorView w_in[] = {w_view};
        TensorView o[] = {empty};
        op->prepackWeights(w_in, o);
        const TensorView q = o[0];
        u_buf.assign(static_cast<size_t>(q.storage_bytes() / sizeof(float)), 0.0f);
        u_view = TensorView(q.shape_span(), DataType::f32, u_buf.data(),
                            q.pitch(), TensorLayout::PackedWinoWeight);
        const TensorView w_in2[] = {w_view};
        TensorView o2[] = {u_view};
        op->prepackWeights(w_in2, o2);
    }

    const std::vector<TensorView> ins = {in_view, u_view, b_view};

    std::vector<float> out_serial(static_cast<size_t>(on), 0.0f);
    auto serial_view = test::make_planar(outs[0], out_serial.data());
    op->compute(serial_view, ins, ComputeContext{});

    SimplePool pool(4);
    ComputeContext ctx;
    ctx.cpu = pool.cpu;
    std::vector<float> out_pool(static_cast<size_t>(on), 0.0f);
    auto pool_view = test::make_planar(outs[0], out_pool.data());
    op->compute(pool_view, ins, ctx);

    // 2 batches · 2 groups · 7 tile rows = 28 tasks over 4 workers, so the pool
    // path really does run (and shares slots between) several (batch, group)
    // pairs.
    int64_t mismatches = 0;
    for (int64_t i = 0; i < on; ++i) {
        const float a = out_serial[static_cast<size_t>(i)];
        const float b = out_pool[static_cast<size_t>(i)];
        if (std::memcmp(&a, &b, sizeof(float)) != 0) { ++mismatches; }
    }
    NNOPS_EXPECT_EQ(mismatches, int64_t{0});
}

// ============================================================
// f16
// ============================================================

NNOPS_TEST(wino_f16_matches_ref) {
    auto [in_f16, in_view] = test::make_random_f16_tensor({1, 8, 8, 8}, -1.0f, 1.0f, 11);
    auto [w_f16, w_view] = test::make_random_f16_tensor({8, 8, 3, 3}, -0.5f, 0.5f, 12);

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride = {1, 1};
    attrs.padding = {1, 1};
    attrs.algorithm = ConvAlgorithm::Winograd;

    auto op = Conv2D::create(attrs, Backend::CPU);

    const TensorDesc descs[] = {in_view.desc(), w_view.desc()};
    auto outs = op->getOutputTensorDesc(descs);
    const int64_t on = desc_numel(outs[0]);

    std::vector<half> out_wino(static_cast<size_t>(on), half(0.0f));
    auto out_view = test::make_planar(outs[0], out_wino.data());

    std::vector<half> u_buf;
    TensorView u_view;
    {
        TensorView empty;
        const TensorView w_in[] = {w_view};
        TensorView o[] = {empty};
        op->prepackWeights(w_in, o);
        const TensorView q = o[0];
        u_buf.assign(static_cast<size_t>(q.storage_bytes() / sizeof(half)), half(0.0f));
        u_view = TensorView(q.shape_span(), DataType::f16, u_buf.data(),
                            q.pitch(), TensorLayout::PackedWinoWeight);
        const TensorView w_in2[] = {w_view};
        TensorView o2[] = {u_view};
        op->prepackWeights(w_in2, o2);
    }

    const TensorView ins[] = {in_view, u_view};
    op->compute(out_view, ins);

    // conv2d_ref is f32-only, so widen the (already f16-quantized) operands and
    // run the reference convolution in f32 on exactly the values Winograd saw.
    auto in_f32 = test::f16_to_f32(in_f16);
    auto w_f32 = test::f16_to_f32(w_f16);
    const int64_t ishape[] = {1, 8, 8, 8};
    const int64_t wshape[] = {8, 8, 3, 3};
    TensorView in_ref(ishape, DataType::f32, in_f32.data());
    TensorView w_ref(wshape, DataType::f32, w_f32.data());
    std::vector<float> ref2(on, 0.0f);
    auto ref2_view = test::make_planar(outs[0], ref2.data());
    const TensorView rins[] = {in_ref, w_ref};
    nnops::backend::cpu::reference::conv2d_ref(attrs, ref2_view, rins,
                                               ComputeContext{}, nullptr);

    const auto got = test::f16_to_f32(out_wino);
    float max_diff = 0.0f;
    for (int64_t i = 0; i < on; ++i) {
        max_diff = std::max(max_diff,
                            std::abs(got[static_cast<size_t>(i)] - ref2[static_cast<size_t>(i)]));
    }
    NNOPS_EXPECT_TRUE(max_diff < test::kF16AccumTol);
}
