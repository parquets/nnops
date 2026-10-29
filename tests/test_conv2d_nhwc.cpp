/// Unit tests for the channel-last CPU Conv2D route: NHWC activations with an
/// HWIO weight (TensorFlow's [KH][KW][IC/G][OC] filter), floating-point
/// (f32/f16) and quantized, against the planar reference implementations.
///
/// Every case keeps the same values in both orders — NCHW/OIHW for the
/// reference, NHWC/HWIO for the kernel — and compares the kernel's output,
/// converted back to NCHW, against the reference. The conversion itself is
/// covered by test_layout_convert; here it is only plumbing, so a case that
/// fails is a kernel bug.

#include "nnops/ops/conv2d.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/detail/half.hpp"
#include "backend/cpu/conv/conv2d_implicit_gemm.h"
#include "backend/cpu/conv/conv2d_winograd.h"
#include "common/cpu_pool.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/channel_last.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

using namespace nnops;
using nnops::backend::cpu::half;
using nnops::backend::cpu::float_to_half;
using nnops::backend::cpu::half_to_float;

namespace nnops::backend::cpu::reference {
void conv2d_ref(const Conv2DAttributes& attrs, TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx, void* workspace);
void conv2d_int8_ref(const Conv2DAttributes& attrs, TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx, void* workspace);
}

namespace {

// ============================================================
// Small type / layout plumbing
// ============================================================

template <class T> struct dtype_tag;
template <> struct dtype_tag<float>   { static constexpr DataType value = DataType::f32; };
template <> struct dtype_tag<half>    { static constexpr DataType value = DataType::f16; };
template <> struct dtype_tag<int8_t>  { static constexpr DataType value = DataType::s8; };
template <> struct dtype_tag<uint8_t> { static constexpr DataType value = DataType::u8; };

inline float as_float(float v)   { return v; }
inline float as_float(half v)    { return half_to_float(v); }
inline float as_float(int8_t v)  { return static_cast<float>(v); }
inline float as_float(uint8_t v) { return static_cast<float>(v); }

inline size_t elem_size(DataType dt) { return static_cast<size_t>(data_type_size(dt)); }

inline QuantParams per_tensor(float scale, int32_t zp) {
    QuantParams qp;
    qp.granularity = QuantGranularity::PerTensor;
    qp.scale = scale;
    qp.zero_point = zp;
    return qp;
}

inline QuantParams per_channel(const float* scale, const int32_t* zp, int64_t n) {
    QuantParams qp;
    qp.granularity = QuantGranularity::PerChannel;
    qp.scale_data = scale;
    qp.zero_point_data = zp;
    qp.num_scales = n;
    return qp;
}

/// float → s8/u8 storage, with the arithmetic the quantizers use.
int32_t quant_int(float x, float inv_scale, int32_t zp, bool u8) {
    const int32_t qmin = u8 ? 0 : -128;
    const int32_t qmax = u8 ? 255 : 127;
    const float v = std::fmaf(x, inv_scale, static_cast<float>(zp));
    int32_t q = static_cast<int32_t>(std::nearbyintf(v));
    return std::min(std::max(q, qmin), qmax);
}

/// Minimal fixed-size thread pool exposing a CpuBackend (the test_conv2d
/// pattern): the channel-last kernel must be bit-identical under it.
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

// ============================================================
// Floating-point cases
// ============================================================

struct FpCase {
    std::array<int64_t, 4> ishape;  ///< N, IC, IH, IW
    int64_t oc, kh, kw, sh, sw, ph, pw, dh, dw, groups;
    bool bias;
};

struct FpResult {
    float worst = 0.0f;   ///< max |got − ref| / (1 + |ref|)
    int64_t idx = -1;     ///< where it happened
    float got = 0.0f, want = 0.0f;
};

/// Run one case on the channel-last route and on the planar reference, on the
/// same values, and report the worst scaled deviation of the output.
///
/// @p ep / @p add_to extend the case to the epilogue paths; with add_to both
/// sides start from the same residual, permuted into their own order.
FpResult run_fp_case(const FpCase& c, DataType dt,
                     EpilogueActivateType ep = EpilogueActivateType::None,
                     bool add_to = false, uint64_t seed = 900,
                     ConvAlgorithm alg = ConvAlgorithm::Auto)
{
    const int64_t N = c.ishape[0], IC = c.ishape[1];
    const int64_t IH = c.ishape[2], IW = c.ishape[3];
    const int64_t ICG = IC / c.groups;
    const int64_t ish[] = {N, IC, IH, IW};
    const int64_t wsh[] = {c.oc, ICG, c.kh, c.kw};          // OIHW: the reference
    const int64_t wsh_cl[] = {c.kh, c.kw, ICG, c.oc};       // HWIO: the kernel
    const int64_t bsh[] = {c.oc};

    auto [in32v, in32] = test::make_random_tensor(std::span<const int64_t>(ish, 4),
                                                  -1.0f, 1.0f, seed);
    auto [w32v, w32]   = test::make_random_tensor(std::span<const int64_t>(wsh, 4),
                                                  -1.0f, 1.0f, seed + 1);
    std::vector<float> b32v;
    if (c.bias) {
        b32v.resize(static_cast<size_t>(c.oc));
        test::XorShift128 rng(seed + 2);
        rng.fill_float(b32v.data(), c.oc, -1.0f, 1.0f);
    }

    Conv2DAttributes attrs;
    attrs.kernel_size = {c.kh, c.kw};
    attrs.stride      = {c.sh, c.sw};
    attrs.dilation    = {c.dh, c.dw};
    attrs.padding     = {c.ph, c.pw};
    attrs.groups      = c.groups;
    attrs.epilogue.type = ep;
    attrs.add_to      = add_to;
    attrs.algorithm   = alg;

    auto op = Conv2D::create(attrs, Backend::CPU);

    // ---- the planar reference, on the NCHW / OIHW values -------------------
    std::vector<TensorView> ref_ins = {in32, w32};
    {
        std::vector<TensorDesc> d = {in32.desc(), w32.desc()};
        if (c.bias) {
            TensorView bv(bsh, DataType::f32, b32v.data());
            ref_ins.push_back(bv);
            d.push_back(bv.desc());
        }
        auto od = op->getOutputTensorDesc(d);
        const int64_t OC = od[0].dims[1], OH = od[0].dims[2], OW = od[0].dims[3];
        const int64_t out_n = N * OC * OH * OW;

        // The residual both sides start from (add_to), NCHW order.
        std::vector<float> res_nchw;
        if (add_to) {
            res_nchw.resize(static_cast<size_t>(out_n));
            test::XorShift128 rng(seed + 3);
            rng.fill_float(res_nchw.data(), out_n, -0.5f, 0.5f);
        }

        std::vector<float> ref_buf(static_cast<size_t>(out_n));
        for (int64_t i = 0; i < out_n; ++i) {
            ref_buf[static_cast<size_t>(i)] = add_to ? res_nchw[static_cast<size_t>(i)] : 0.0f;
        }
        auto ref_out = test::make_planar(od[0], ref_buf.data());
        nnops::backend::cpu::reference::conv2d_ref(attrs, ref_out, ref_ins, {}, nullptr);

        // ---- the channel-last values ---------------------------------------
        auto in_cl = test::nchw_to_nhwc<float>(in32v, N, IC, IH, IW);
        auto w_cl  = test::oihw_to_hwio<float>(w32v, c.oc, ICG, c.kh, c.kw);
        const std::vector<float> res_cl = add_to
            ? test::nchw_to_nhwc<float>(res_nchw, N, OC, OH, OW) : std::vector<float>{};

        auto check = [&](auto tag) -> FpResult {
            using T = decltype(tag);
            constexpr DataType dtt = dtype_tag<T>::value;
            auto to_T = [](float v) -> T {
                if constexpr (std::is_same_v<T, float>) { return v; }
                else { return float_to_half(v); }
            };

            std::vector<T> in_buf(in_cl.size()), w_buf(w_cl.size());
            for (size_t i = 0; i < in_buf.size(); ++i) { in_buf[i] = to_T(in_cl[i]); }
            for (size_t i = 0; i < w_buf.size(); ++i) { w_buf[i] = to_T(w_cl[i]); }

            TensorView in_v(std::span<const int64_t>(ish, 4), dtt, in_buf.data(),
                            TensorLayout::NHWC);
            TensorView w_v(std::span<const int64_t>(wsh_cl, 4), dtt, w_buf.data(),
                           TensorLayout::HWIO);

            // Winograd consumes a prepacked U, not the raw HWIO filter: query
            // its panels from the op, lay them out, then let the op fill them.
            // The channel-last U is the GEMM's B panels, so its rank and shape
            // are the kernel's, not U[16][OC][IC/G].
            std::vector<T> u_buf;
            TensorView u_v;
            if (alg == ConvAlgorithm::Winograd) {
                TensorView empty;
                const TensorView w_q[] = {w_v};
                TensorView o_q[] = {empty};
                op->prepackWeights(w_q, o_q);
                const TensorView q = o_q[0];
                u_buf.assign(static_cast<size_t>(q.storage_bytes() / sizeof(T)), T{});
                u_v = TensorView(q.shape_span(), dtt, u_buf.data(), q.pitch(),
                                 q.layout());
                const TensorView w_f[] = {w_v};
                TensorView o_f[] = {u_v};
                op->prepackWeights(w_f, o_f);
            }

            std::vector<TensorView> ins = {in_v};
            ins.push_back(alg == ConvAlgorithm::Winograd ? u_v : w_v);
            std::vector<T> b_buf;
            if (c.bias) {
                b_buf.resize(static_cast<size_t>(c.oc));
                for (int64_t i = 0; i < c.oc; ++i) { b_buf[i] = to_T(b32v[i]); }
                TensorView b_v(bsh, dtt, b_buf.data());
                ins.push_back(b_v);
            }

            std::vector<T> out_buf(static_cast<size_t>(out_n));
            for (int64_t i = 0; i < out_n; ++i) {
                out_buf[static_cast<size_t>(i)] = to_T(
                    add_to ? res_cl[static_cast<size_t>(i)] : 0.0f);
            }
            const int64_t osh[] = {N, OC, OH, OW};
            TensorView out_v(std::span<const int64_t>(osh, 4), dtt, out_buf.data(),
                             TensorLayout::NHWC);
            op->compute(out_v, ins, {}, nullptr);

            // Back to planar order for the comparison.
            const auto got = test::nhwc_to_nchw<T>(out_buf, N, OC, OH, OW);
            FpResult r;
            for (int64_t i = 0; i < out_n; ++i) {
                const float a = as_float(got[static_cast<size_t>(i)]);
                const float b = ref_buf[static_cast<size_t>(i)];
                const float rel = std::abs(a - b) / (1.0f + std::abs(b));
                if (rel > r.worst) { r.worst = rel; r.idx = i; r.got = a; r.want = b; }
            }
            return r;
        };

        return (dt == DataType::f32) ? check(float{}) : check(half{});
    }
}

/// Throw when the worst scaled deviation exceeds @p tol.
void expect_fp_close(const FpCase& c, DataType dt, float tol,
                     EpilogueActivateType ep = EpilogueActivateType::None,
                     bool add_to = false, uint64_t seed = 900, int idx = -1,
                     ConvAlgorithm alg = ConvAlgorithm::Auto)
{
    const FpResult r = run_fp_case(c, dt, ep, add_to, seed, alg);
    if (r.worst > tol) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "conv2d_nhwc vs reference (case %d): |got %.6g - ref %.6g| / (1+|ref|) "
                      "= %.3g at elem %lld (tol %.3g)",
                      idx, double(r.got), double(r.want), double(r.worst),
                      static_cast<long long>(r.idx), double(tol));
        throw std::runtime_error(msg);
    }
}

}  // namespace

// ============================================================
// Hand-verified geometry
// ============================================================

NNOPS_TEST(conv2d_nhwc_fp_hand_exact) {
    // 1x1x4x4 all-ones, 1x1x3x3 kernel all-0.5, stride 1, pad 0, bias 0.5:
    // each of the 2x2 outputs is 9 · 1 · 0.5 + 0.5 = 5.0. The same values in
    // NHWC / HWIO orders must give the same numbers.
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t wshape[] = {3, 3, 1, 1};
    const int64_t bshape[] = {1};

    std::vector<float> in_buf(16, 1.0f);   // NHWC [1][4][4][1]
    std::vector<float> w_buf(9, 0.5f);     // HWIO [3][3][1][1]
    std::vector<float> b_buf = {0.5f};

    TensorView input(ishape, DataType::f32, in_buf.data(), TensorLayout::NHWC);
    TensorView weight(wshape, DataType::f32, w_buf.data(), TensorLayout::HWIO);
    TensorView bias(bshape, DataType::f32, b_buf.data());

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride      = {1, 1};
    attrs.padding     = {0, 0};

    auto op = Conv2D::create(attrs, Backend::CPU);
    const TensorDesc descs[] = {input.desc(), weight.desc(), bias.desc()};
    auto outs = op->getOutputTensorDesc(descs);
    NNOPS_EXPECT_EQ(outs[0].layout, TensorLayout::NHWC);
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t{2});
    NNOPS_EXPECT_EQ(outs[0].dims[3], int64_t{2});

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()), 0.0f);
    auto out = test::make_planar(outs[0], out_buf.data());
    const TensorView ins[] = {input, weight, bias};
    NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));

    for (int i = 0; i < 4; ++i) { NNOPS_EXPECT_NEAR(out_buf[static_cast<size_t>(i)], 5.0f, 1e-5f); }
}

/// The activations are NHWC, so the input is [N][H][W][C] — a transposed or
/// channel-planar read shows up immediately here: channel c of pixel (h, w)
/// holds a value only that channel's filter can reproduce.
NNOPS_TEST(conv2d_nhwc_fp_channel_order) {
    // 1x1x2x2 NHWC input, 2 output channels, per-channel 1x1 filters that pick
    // one input channel each: out[oc=0] = 2·c0, out[oc=1] = 3·c1.
    const int64_t ishape[] = {1, 2, 2, 2};
    const int64_t wshape[] = {1, 1, 2, 2};   // HWIO [1][1][IC=2][OC=2]

    // pixel (h, w) = [c0, c1];  stored [h][w][c]
    std::vector<float> in_buf = {1.0f, 5.0f,    2.0f, 6.0f,
                                 3.0f, 7.0f,    4.0f, 8.0f};
    std::vector<float> w_buf  = {2.0f, 0.0f,    0.0f, 3.0f};

    TensorView input(ishape, DataType::f32, in_buf.data(), TensorLayout::NHWC);
    TensorView weight(wshape, DataType::f32, w_buf.data(), TensorLayout::HWIO);

    Conv2DAttributes attrs;   // 1x1 — the pointwise route
    auto op = Conv2D::create(attrs, Backend::CPU);
    const TensorDesc descs[] = {input.desc(), weight.desc()};
    auto outs = op->getOutputTensorDesc(descs);

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()), 0.0f);
    auto out = test::make_planar(outs[0], out_buf.data());
    const TensorView ins[] = {input, weight};
    NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));

    // out is [1][2][2][2] = [n][h][w][oc]
    const float want[] = {2.0f, 15.0f,   4.0f, 18.0f,
                          6.0f, 21.0f,   8.0f, 24.0f};
    for (int i = 0; i < 8; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[static_cast<size_t>(i)], want[i], 1e-5f);
    }
}

// ============================================================
// Floating point vs the planar reference
// ============================================================

NNOPS_TEST(conv2d_nhwc_fp_matches_ref) {
    struct Case { FpCase c; uint64_t seed; };
    const Case cases[] = {
        // ---- 3x3, the dominant shape --------------------------------------
        {{{1, 3, 16, 16},   8, 3, 3, 1, 1, 1, 1, 1, 1, 1, true},  1},
        // stride 2 with an odd plane: the pixel tiles end on a partial m_block.
        {{{2, 4, 13, 15},   6, 3, 3, 2, 2, 1, 1, 1, 1, 1, false}, 2},
        // pad 2 on both axes.
        {{{1, 2, 12, 12},   4, 3, 3, 1, 1, 2, 2, 1, 1, 1, true},  3},
        // dilation 2.
        {{{1, 3, 18, 18},   6, 3, 3, 1, 1, 1, 1, 2, 2, 1, false}, 4},
        // groups 2 (oc_per_group 4, ic_per_group 2).
        {{{1, 4, 20, 20},   8, 3, 3, 1, 1, 1, 1, 1, 1, 2, true},  5},
        // groups == IC: ic_per_group 1, a rank-1 panel (kdim = 1 · 9).
        {{{1, 8, 14, 14},   8, 3, 3, 1, 1, 1, 1, 1, 1, 8, false}, 6},
        // IC 40 past icn_block 16 with a tail of 8: three ic blocks, the last on
        // the shorter panel stride — and icn_block < ic_per_group, so the B pack
        // goes through the tap-major staging tile.
        {{{1, 40, 12, 12},  24, 3, 3, 1, 1, 1, 1, 1, 1, 1, true}, 7},
        // IC 48 = 3 × 16 exactly: no tail, three full blocks.
        {{{1, 48, 10, 10},  16, 3, 3, 1, 1, 1, 1, 1, 1, 1, false}, 8},
        // wide OC (256 > ocn_block 128): two filter blocks, two panel sets.
        {{{1, 16, 14, 14},  256, 3, 3, 2, 2, 1, 1, 1, 1, 1, false}, 9},
        // OW 28 with m_block 24 — a 24 + 4 pixel split per row.
        {{{1, 8, 15, 29},   12, 3, 3, 1, 1, 1, 1, 1, 1, 1, true}, 10},

        // ---- other kernel areas -------------------------------------------
        // 5x5 (kdim 8 · 25 = 200).
        {{{1, 8, 18, 18},   16, 5, 5, 1, 1, 2, 2, 1, 1, 1, true},  11},
        // 5x5 stride 2 (a MobileNet downsample).
        {{{1, 4, 28, 28},   16, 5, 5, 2, 2, 1, 1, 1, 1, 1, false}, 12},
        // 7x7 stride 2 pad 3 (a ResNet stem), kdim 2 · 49 = 98.
        {{{1, 3, 32, 32},   16, 7, 7, 2, 2, 3, 3, 1, 1, 1, true},  13},
        // 9x9 — the rank-1 panel rung (icn_block 1), kdim 81.
        {{{1, 2, 24, 24},   4, 9, 9, 1, 1, 4, 4, 1, 1, 1, false},  14},
        // non-square 3x5 with an asymmetric pad.
        {{{1, 3, 18, 22},   6, 3, 5, 1, 1, 1, 2, 1, 1, 1, false},  15},
        // 2x2 stride 2 with groups.
        {{{1, 4, 20, 20},   8, 2, 2, 2, 2, 1, 1, 1, 1, 2, false},  16},

        // ---- 1x1 pointwise (A read from the input row) ---------------------
        // IC 128 = the whole group in one K pass, direct-A read.
        {{{1, 128, 8, 8},   192, 1, 1, 1, 1, 0, 0, 1, 1, 1, true}, 17},
        // IC 512 > the 256 ic-block cap: two K passes over one tile.
        {{{1, 512, 5, 5},   64, 1, 1, 1, 1, 0, 0, 1, 1, 1, false}, 18},
        // groups 2 at icg 64 with a batch dimension.
        {{{2, 128, 7, 7},   64, 1, 1, 1, 1, 0, 0, 1, 1, 2, false}, 19},
        // OW 5 < m_block 24: a single partial pixel tile.
        {{{1, 32, 4, 5},    12, 1, 1, 1, 1, 0, 0, 1, 1, 1, true},  20},
        // A 1x1 with a kernel-size-1 attribute but a pitched (non-pointwise)
        // geometry is not reachable here — stride 2 keeps it on the gather.
        {{{1, 32, 12, 12},  16, 1, 1, 2, 2, 0, 0, 1, 1, 1, false}, 21},

        // ---- full-size planes: a ResNet stage -----------------------------
        // Every case above fits a handful of tiles; these are the shapes the
        // model runners actually call, where a plane is thousands of pixels
        // and the pixel blocks, not the filter, are what the grid runs out of.
        {{{1, 64, 56, 56},  64, 3, 3, 1, 1, 1, 1, 1, 1, 1, true},  22},
        {{{1, 64, 112, 112}, 64, 3, 3, 2, 2, 1, 1, 1, 1, 1, false}, 23},
        {{{1, 128, 28, 28}, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, true},   24},
        {{{1, 64, 56, 56}, 256, 1, 1, 1, 1, 0, 0, 1, 1, 1, false},  25},
    };

    for (size_t i = 0; i < std::size(cases); ++i) {
        expect_fp_close(cases[i].c, DataType::f32, 2e-4f,
                        EpilogueActivateType::None, false, cases[i].seed,
                        static_cast<int>(i));
    }
}

NNOPS_TEST(conv2d_nhwc_f16_matches_ref) {
    struct Case { FpCase c; uint64_t seed; };
    const Case cases[] = {
        {{{1, 3, 16, 16},  8, 3, 3, 1, 1, 1, 1, 1, 1, 1, true},  31},
        {{{2, 4, 13, 15},  6, 3, 3, 2, 2, 1, 1, 1, 1, 1, false}, 32},
        // the staged-pack path (icn_block 16 < ic_per_group 40, with a tail).
        {{{1, 40, 12, 12}, 24, 3, 3, 1, 1, 1, 1, 1, 1, 1, true}, 33},
        // groups, and a 5x5.
        {{{1, 8, 20, 20},  16, 5, 5, 1, 1, 2, 2, 1, 1, 2, false}, 34},
        // 7x7 stem.
        {{{1, 3, 32, 32},  16, 7, 7, 2, 2, 3, 3, 1, 1, 1, true},  35},
        // 1x1 pointwise, K = 128 in one pass.
        {{{1, 128, 8, 8},  32, 1, 1, 1, 1, 0, 0, 1, 1, 1, true},  36},
        // 1x1 with two K passes (IC 512).
        {{{1, 512, 5, 5},  64, 1, 1, 1, 1, 0, 0, 1, 1, 1, false}, 37},
        // a partial pixel tile.
        {{{1, 32, 4, 5},   12, 1, 1, 1, 1, 0, 0, 1, 1, 1, true},  38},
    };

    for (size_t i = 0; i < std::size(cases); ++i) {
        expect_fp_close(cases[i].c, DataType::f16, nnops::test::kF16AccumTol,
                        EpilogueActivateType::None, false, cases[i].seed,
                        static_cast<int>(i));
    }
}

NNOPS_TEST(conv2d_nhwc_fp_epilogues) {
    // The finish pass is per *column* here (a tile's rows are pixels), so every
    // activation and the residual add have their own path through it.
    const FpCase c = {{1, 6, 11, 11}, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, true};

    const EpilogueActivateType eps[] = {
        EpilogueActivateType::None,      EpilogueActivateType::Relu,
        EpilogueActivateType::Relu6,     EpilogueActivateType::Gelu,
        EpilogueActivateType::Sigmoid,   EpilogueActivateType::Tanh,
        EpilogueActivateType::LeakyRelu, EpilogueActivateType::Silu,
        EpilogueActivateType::HardSwish, EpilogueActivateType::Elu,
    };
    for (size_t i = 0; i < std::size(eps); ++i) {
        expect_fp_close(c, DataType::f32, 2e-4f, eps[i], false, 400 + i,
                        static_cast<int>(i));
        expect_fp_close(c, DataType::f32, 2e-4f, eps[i], true, 500 + i,
                        static_cast<int>(i) + 100);
    }

    // The clamp folds into the MMA store only when there is neither a bias nor
    // a residual; that path has to agree with the plain-convolution + clamp
    // reference too, so cover it without a bias.
    FpCase no_bias = c;
    no_bias.bias = false;
    expect_fp_close(no_bias, DataType::f32, 2e-4f, EpilogueActivateType::Relu, false, 600);
    expect_fp_close(no_bias, DataType::f32, 2e-4f, EpilogueActivateType::Relu6, false, 601);

    // add_to with neither a bias nor an activation is the one case where the
    // accumulator starts at the residual alone — no seed at all, so the tile
    // must still be left alone (not zeroed) before the MMA.
    expect_fp_close(no_bias, DataType::f32, 2e-4f, EpilogueActivateType::None, true, 602);
    // ... and with an activation but no bias the residual is saved and added
    // after it, so the accumulator has to start empty even though the tile
    // still holds the residual.
    expect_fp_close(no_bias, DataType::f32, 2e-4f, EpilogueActivateType::Relu, true, 603);
    expect_fp_close(no_bias, DataType::f32, 2e-4f, EpilogueActivateType::Relu6, true, 604);
    expect_fp_close(no_bias, DataType::f16, nnops::test::kF16AccumTol,
                    EpilogueActivateType::Relu, true, 605);
}

// ============================================================
// Winograd F(2×2, 3×3) on the channel-last route
// ============================================================
//
// The winograd kernel here is the same f2,3 transform as the planar one with
// the GEMM's operands swapped: the pixels are M and the output channels N, so
// U is prepacked into the MMA's B panels instead of the [16][OC][IC/G] planes.
// Everything else — the epilogue, the bias, the residual — must land on exactly
// the values the reference computes, which is what these cases check.

NNOPS_TEST(conv2d_nhwc_wino_matches_ref) {
    struct Case { FpCase c; uint64_t seed; };
    const Case f32_cases[] = {
        // The shapes Winograd is selected for: >= 32 channels per group and a
        // plane of at least 14x14.
        {{{1, 32, 16, 16},  32, 3, 3, 1, 1, 1, 1, 1, 1, 1, true},   71},
        {{{1, 64, 14, 14},  64, 3, 3, 1, 1, 1, 1, 1, 1, 1, false},  72},
        // Several oc blocks (256 > kOcnBlock 64) over a small plane.
        {{{1, 32, 14, 14}, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, true},   73},
        // Odd OC: the last panel is mostly the zero padding prepack wrote.
        {{{1, 32, 15, 15},   5, 3, 3, 1, 1, 1, 1, 1, 1, 1, true},   74},
        // IC/G 40 is not a multiple of the vector, and 13x15 is odd on both
        // axes: the last tile of every row and of the last row are partial.
        {{{2, 40, 13, 15},  24, 3, 3, 1, 1, 1, 1, 1, 1, 1, false},  75},
        // groups 2 (IC/G 8, OC/G 4 — a single narrow panel per group).
        {{{1, 16, 20, 20},   8, 3, 3, 1, 1, 1, 1, 1, 1, 2, true},   76},
        // A single channel per group (IC/G 1) and a filter over the whole plane.
        {{{1, 2, 10, 10},    4, 3, 3, 1, 1, 1, 1, 1, 1, 2, false},  77},
        // IC 256 = two k-blocks of 128 at f32's 256 cap → exactly one; 512 → two.
        {{{1, 512, 8, 8},   64, 3, 3, 1, 1, 1, 1, 1, 1, 1, true},   78},
        // pad 0 (OH = IH − 2) and pad 2.
        {{{1, 32, 18, 18},  16, 3, 3, 1, 1, 0, 0, 1, 1, 1, false},  79},
        {{{1, 32, 12, 12},  16, 3, 3, 1, 1, 2, 2, 1, 1, 1, true},   80},
    };
    for (size_t i = 0; i < std::size(f32_cases); ++i) {
        expect_fp_close(f32_cases[i].c, DataType::f32, 2e-4f,
                        EpilogueActivateType::None, false, f32_cases[i].seed,
                        static_cast<int>(i), ConvAlgorithm::Winograd);
    }

    // f16: the transform runs in f32 and only the accumulator is narrowed, so
    // the tolerance is the shared f16 one.
    const Case f16_cases[] = {
        {{{1, 32, 16, 16},  32, 3, 3, 1, 1, 1, 1, 1, 1, 1, true},   81},
        {{{1, 64, 14, 14},  32, 3, 3, 1, 1, 1, 1, 1, 1, 1, false},  82},
        {{{1, 40, 13, 15},  24, 3, 3, 1, 1, 1, 1, 1, 1, 1, true},   83},
        {{{2, 16, 20, 20},   8, 3, 3, 1, 1, 1, 1, 1, 1, 2, false},  84},
    };
    for (size_t i = 0; i < std::size(f16_cases); ++i) {
        expect_fp_close(f16_cases[i].c, DataType::f16, nnops::test::kF16AccumTol,
                        EpilogueActivateType::None, false, f16_cases[i].seed,
                        static_cast<int>(i), ConvAlgorithm::Winograd);
    }
}

NNOPS_TEST(conv2d_nhwc_wino_epilogues) {
    // Bias and epilogue are applied to the real 2×2 outputs of the transform, in
    // a single write-back — the same contract the planar route keeps.
    const FpCase c = {{1, 32, 15, 14}, 24, 3, 3, 1, 1, 1, 1, 1, 1, 1, true};
    const EpilogueActivateType eps[] = {
        EpilogueActivateType::None,      EpilogueActivateType::Relu,
        EpilogueActivateType::Relu6,     EpilogueActivateType::Gelu,
        EpilogueActivateType::Sigmoid,   EpilogueActivateType::Tanh,
        EpilogueActivateType::LeakyRelu, EpilogueActivateType::Silu,
        EpilogueActivateType::HardSwish, EpilogueActivateType::Elu,
    };
    for (size_t i = 0; i < std::size(eps); ++i) {
        expect_fp_close(c, DataType::f32, 2e-4f, eps[i], false, 700 + i,
                        static_cast<int>(i), ConvAlgorithm::Winograd);
        expect_fp_close(c, DataType::f32, 2e-4f, eps[i], true, 800 + i,
                        static_cast<int>(i) + 100, ConvAlgorithm::Winograd);
    }
    FpCase no_bias = c;
    no_bias.bias = false;
    expect_fp_close(no_bias, DataType::f32, 2e-4f, EpilogueActivateType::Relu6,
                    false, 900, -1, ConvAlgorithm::Winograd);
    expect_fp_close(no_bias, DataType::f32, 2e-4f, EpilogueActivateType::Relu,
                    true, 901, -1, ConvAlgorithm::Winograd);
}

NNOPS_TEST(conv2d_nhwc_wino_hand_exact) {
    // A 3x3 filter whose only nonzero tap is the center, and per-channel deltas:
    // out[oc] == in[oc] pixel for pixel. That pins the pixel mapping (the tile
    // flattening and the 2x2 write-back), the channel mapping (the GEMM's N and
    // the panels' columns) and the transform round trip in one case.
    constexpr int64_t IC = 8, OC = 8, H = 4, W = 4;
    const int64_t ish[] = {1, IC, H, W};
    const int64_t wsh[] = {3, 3, IC, OC};
    std::vector<float> in_buf(static_cast<size_t>(IC * H * W)), w_buf(9 * IC * OC, 0.0f);
    for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
            for (int64_t c = 0; c < IC; ++c) {
                in_buf[static_cast<size_t>((h * W + w) * IC + c)] =
                    0.25f * static_cast<float>(h * 100 + w * 10 + c + 1);
            }
        }
    }
    for (int64_t c = 0; c < OC; ++c) { w_buf[static_cast<size_t>((4 * IC + c) * OC + c)] = 1.0f; }

    TensorView in_v(std::span<const int64_t>(ish, 4), DataType::f32, in_buf.data(),
                    TensorLayout::NHWC);
    TensorView w_v(std::span<const int64_t>(wsh, 4), DataType::f32, w_buf.data(),
                   TensorLayout::HWIO);

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.padding = {1, 1};
    attrs.algorithm = ConvAlgorithm::Winograd;
    auto op = Conv2D::create(attrs, Backend::CPU);

    TensorView q;
    const TensorView wq[] = {w_v};
    TensorView oq[] = {q};
    op->prepackWeights(wq, oq);
    std::vector<float> u_buf(static_cast<size_t>(oq[0].storage_bytes() / sizeof(float)), 0.0f);
    TensorView u_v(oq[0].shape_span(), DataType::f32, u_buf.data(), oq[0].pitch(),
                   oq[0].layout());
    const TensorView wf[] = {w_v};
    TensorView of[] = {u_v};
    op->prepackWeights(wf, of);

    const int64_t osh[] = {1, OC, H, W};
    std::vector<float> out_buf(static_cast<size_t>(OC * H * W), 0.0f);
    TensorView out_v(std::span<const int64_t>(osh, 4), DataType::f32, out_buf.data(),
                     TensorLayout::NHWC);
    const TensorView ins[] = {in_v, u_v};
    NNOPS_EXPECT_TRUE(is_ok(op->compute(out_v, ins)));

    for (size_t i = 0; i < in_buf.size(); ++i) {
        if (std::abs(out_buf[i] - in_buf[i]) > 1e-5f) {
            std::printf("elem %zu (pixel %zu, ch %zu): got %g want %g\n", i,
                        i / static_cast<size_t>(IC), i % static_cast<size_t>(IC),
                        double(out_buf[i]), double(in_buf[i]));
            NNOPS_EXPECT_NEAR(out_buf[i], in_buf[i], 1e-5f);
            break;
        }
    }
}

NNOPS_TEST(conv2d_nhwc_wino_prepack_matches_planar) {
    // Both U forms hold the same G g Gᵀ — the channel-last one is just the GEMM's
    // B panels, so entry (p, kb, g·jb + j, k, col) must equal the planar U's
    // (p, oc, ic) for oc = (g·jb + j)·nr + col, ic = kb·kc + k. That equality is
    // the whole prepack contract, and the kernel cannot be right without it.
    const int64_t IC = 40, OC = 24, G = 1;   // no vector multiple either way
    const int64_t ICG = IC / G, OCG = OC / G;
    const int64_t wsh[] = {OC, ICG, 3, 3};           // OIHW
    const int64_t wsh_cl[] = {3, 3, ICG, OC};        // HWIO
    auto [w32v, w32] = test::make_random_tensor(
        std::span<const int64_t>(wsh, 4), -1.0f, 1.0f, 909);
    const auto w_cl = test::oihw_to_hwio<float>(w32v, OC, ICG, 3, 3);

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.padding = {1, 1};
    attrs.groups = G;
    attrs.algorithm = ConvAlgorithm::Winograd;
    auto op = Conv2D::create(attrs, Backend::CPU);

    // Prepack both forms from the same values.
    TensorView w_pl(wsh, DataType::f32, w32v.data()),
               w_cl_v(std::span<const int64_t>(wsh_cl, 4), DataType::f32,
                      const_cast<float*>(w_cl.data()), TensorLayout::HWIO);
    std::vector<float> u_pl_buf, u_cl_buf;
    TensorView u_pl, u_cl;
    {
        TensorView q;
        const TensorView in1[] = {w_pl};
        TensorView o1[] = {q};
        op->prepackWeights(in1, o1);
        const TensorView qp = o1[0];
        u_pl_buf.assign(static_cast<size_t>(qp.storage_bytes() / sizeof(float)), 0.0f);
        u_pl = TensorView(qp.shape_span(), DataType::f32, u_pl_buf.data(), qp.pitch(),
                          qp.layout());
        const TensorView in2[] = {w_pl};
        TensorView o2[] = {u_pl};
        op->prepackWeights(in2, o2);

        TensorView q2;
        const TensorView in3[] = {w_cl_v};
        TensorView o3[] = {q2};
        op->prepackWeights(in3, o3);
        const TensorView qc = o3[0];
        u_cl_buf.assign(static_cast<size_t>(qc.storage_bytes() / sizeof(float)), 0.0f);
        u_cl = TensorView(qc.shape_span(), DataType::f32, u_cl_buf.data(), qc.pitch(),
                          qc.layout());
        const TensorView in4[] = {w_cl_v};
        TensorView o4[] = {u_cl};
        op->prepackWeights(in4, o4);
    }
    NNOPS_EXPECT_EQ(static_cast<int64_t>(u_pl.layout()),
                    static_cast<int64_t>(TensorLayout::PackedWinoWeight));
    NNOPS_EXPECT_EQ(static_cast<int64_t>(u_cl.layout()),
                    static_cast<int64_t>(TensorLayout::PackedWinoWeightLast));

    const auto ug = backend::cpu::wino_channel_last_u(ICG, OCG, G, DataType::f32);
    const float* pl = u_pl_buf.data();   // [p][oc][ic], row stride ICG
    const float* cl = u_cl_buf.data();
    for (int64_t p = 0; p < 16; ++p) {
        for (int64_t oc = 0; oc < OCG; ++oc) {
            for (int64_t ic = 0; ic < ICG; ++ic) {
                const int64_t kb = ic / ug.kc, k = ic - kb * ug.kc;
                const int64_t j = oc / ug.nr, col = oc - j * ug.nr;
                const float want = pl[(p * OC + oc) * ICG + ic];
                const float got = cl[p * ug.plane + (kb * ug.jbg + j) * ug.ldd
                                     + k * ug.nr + col];
                if (got != want) {
                    NNOPS_EXPECT_NEAR(got, want, 1e-6f);
                    return;
                }
            }
        }
    }
}

NNOPS_TEST(conv2d_nhwc_wino_plan_and_auto) {
    // Auto follows the prepacked U it is handed, whichever form it is: the
    // planar U pairs with a planar activation and the panels with a channel-last
    // one, and a mismatched pair must not resolve to a plan at all.
    const int64_t ishape[] = {1, 32, 16, 16};
    const int64_t oshape[] = {1, 32, 14, 14};
    std::vector<float> in_buf(32 * 16 * 16, 0.5f);
    std::vector<float> out_buf(32 * 14 * 14, 0.0f);
    TensorView input(std::span<const int64_t>(ishape, 4), DataType::f32,
                     in_buf.data(), TensorLayout::NHWC);
    TensorView output(std::span<const int64_t>(oshape, 4), DataType::f32,
                      out_buf.data(), TensorLayout::NHWC);

    const backend::cpu::WinoChannelLastU ug =
        backend::cpu::wino_channel_last_u(32, 32, 1, DataType::f32);
    NNOPS_EXPECT_EQ(ug.nr, int64_t{16});
    NNOPS_EXPECT_EQ(ug.kc, int64_t{32});   // one k-block: 32 divides 32
    NNOPS_EXPECT_EQ(ug.nb, int64_t{1});
    NNOPS_EXPECT_EQ(ug.jb, int64_t{2});
    NNOPS_EXPECT_EQ(ug.jbg, int64_t{2});

    const int64_t u_shape[] = {16, ug.nb, ug.jbg, ug.ldd};
    std::vector<float> u_buf(static_cast<size_t>(ug.total), 0.0f);
    TensorView u(std::span<const int64_t>(u_shape, 4), DataType::f32, u_buf.data(),
                 ug.ldd * static_cast<int64_t>(sizeof(float)),
                 TensorLayout::PackedWinoWeightLast);
    // Fill the panels with the identity-free garbage a memset leaves: all zero
    // is a valid U, so the plan is all this case is about.
    const TensorDesc ins[] = {input.desc(), u.desc()};
    const TensorDesc outs[] = {output.desc()};

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.padding = {1, 1};

    const auto plan = backend::cpu::get_conv2d_winograd_plan(attrs, ins, outs, 1, false);
    NNOPS_EXPECT_TRUE(plan.route == backend::cpu::Conv2DWinogradRoute::ChannelLast);
    NNOPS_EXPECT_TRUE(plan.workspace_size > 0);
    NNOPS_EXPECT_EQ(plan.icn_block, ug.kc);
    NNOPS_EXPECT_EQ(plan.num_tiles, int64_t{7 * 7});

    // A planar U with a channel-last activation has no route.
    const int64_t pu_shape[] = {16, 32, 32};
    std::vector<float> pu_buf(16 * 32 * 32, 0.0f);
    TensorView pu(std::span<const int64_t>(pu_shape, 3), DataType::f32, pu_buf.data(),
                  32 * static_cast<int64_t>(sizeof(float)),
                  TensorLayout::PackedWinoWeight);
    const TensorDesc bad[] = {input.desc(), pu.desc()};
    NNOPS_EXPECT_EQ(backend::cpu::get_conv2d_winograd_plan(attrs, bad, outs, 1, false)
                        .workspace_size, 0u);
}

NNOPS_TEST(conv2d_nhwc_threaded_matches_serial) {
    // The row / filter chunks partition the output into disjoint tiles, so the
    // threaded run must be bit-identical to the serial one.
    struct Case { FpCase c; uint64_t seed; };
    const Case cases[] = {
        {{{1, 3, 32, 32},   8, 3, 3, 1, 1, 1, 1, 1, 1, 1, true},  61},
        {{{1, 16, 28, 28}, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, false}, 62},
        {{{1, 128, 16, 16}, 192, 1, 1, 1, 1, 0, 0, 1, 1, 1, true}, 63},
        {{{2, 40, 14, 14},  64, 3, 3, 1, 1, 1, 1, 1, 1, 1, false}, 64},
        // The model runners' hard cases: a plane small enough that the task
        // grid runs out of rows and splits on the filter axis as well.
        {{{1, 64, 56, 56},  64, 3, 3, 1, 1, 1, 1, 1, 1, 1, true},  65},
        {{{1, 128, 28, 28}, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, false}, 66},
        {{{1, 256, 14, 14}, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, true},  67},
        {{{1, 192, 14, 14},  64, 1, 1, 1, 1, 0, 0, 1, 1, 1, false}, 68},
        {{{1, 512, 7, 7},   512, 1, 1, 1, 1, 0, 0, 1, 1, 1, true},   69},
    };

    int case_i = 0;
    for (const auto& cs : cases) {
        const FpCase& c = cs.c;
        const int64_t N = c.ishape[0], IC = c.ishape[1];
        const int64_t IH = c.ishape[2], IW = c.ishape[3];
        const int64_t ICG = IC / c.groups;
        const int64_t ish[] = {N, IC, IH, IW};
        const int64_t wsh[] = {c.oc, ICG, c.kh, c.kw};        // OIHW
        const int64_t wsh_cl[] = {c.kh, c.kw, ICG, c.oc};     // HWIO
        const int64_t bsh[] = {c.oc};

        auto [in32v, _] = test::make_random_tensor(std::span<const int64_t>(ish, 4),
                                                   -1.0f, 1.0f, cs.seed);
        auto [w32v, __] = test::make_random_tensor(std::span<const int64_t>(wsh, 4),
                                                   -1.0f, 1.0f, cs.seed + 1);
        std::vector<float> b32v;
        if (c.bias) {
            b32v.resize(static_cast<size_t>(c.oc));
            test::XorShift128 rng(cs.seed + 2);
            rng.fill_float(b32v.data(), c.oc, -1.0f, 1.0f);
        }

        auto in_cl = test::nchw_to_nhwc<float>(in32v, N, IC, IH, IW);
        auto w_cl  = test::oihw_to_hwio<float>(w32v, c.oc, ICG, c.kh, c.kw);

        Conv2DAttributes attrs;
        attrs.kernel_size = {c.kh, c.kw};
        attrs.stride = {c.sh, c.sw};
        attrs.dilation = {c.dh, c.dw};
        attrs.padding = {c.ph, c.pw};
        attrs.groups = c.groups;

        auto op = Conv2D::create(attrs, Backend::CPU);
        TensorView in_v(std::span<const int64_t>(ish, 4), DataType::f32, in_cl.data(),
                        TensorLayout::NHWC);
        TensorView w_v(std::span<const int64_t>(wsh_cl, 4), DataType::f32, w_cl.data(),
                       TensorLayout::HWIO);
        std::vector<TensorView> ins = {in_v, w_v};
        if (c.bias) { ins.push_back(TensorView(bsh, DataType::f32, b32v.data())); }
        std::vector<TensorDesc> descs;
        for (const auto& v : ins) { descs.push_back(v.desc()); }
        auto od = op->getOutputTensorDesc(descs);
        const int64_t out_n = od[0].numel();
        const int64_t OC = od[0].dims[1], OH = od[0].dims[2], OW = od[0].dims[3];
        const int64_t osh[] = {N, OC, OH, OW};

        auto make_out = [&](std::vector<float>& buf) {
            TensorView v(std::span<const int64_t>(osh, 4), DataType::f32, buf.data(),
                         TensorLayout::NHWC);
            return v;
        };

        std::vector<float> serial_buf(static_cast<size_t>(out_n));
        std::vector<float> threaded_buf(static_cast<size_t>(out_n));
        auto serial_out = make_out(serial_buf);
        auto threaded_out = make_out(threaded_buf);
        op->compute(serial_out, ins, {}, nullptr);

        for (int nthreads : {2, 4, 8, 20}) {
            SimplePool pool(nthreads);
            ComputeContext ctx;
            ctx.cpu = pool.cpu;
            op->compute(threaded_out, ins, ctx, nullptr);
            if (std::memcmp(serial_buf.data(), threaded_buf.data(),
                            static_cast<size_t>(out_n) * sizeof(float)) != 0) {
                throw std::runtime_error(
                    "conv2d_nhwc threaded run differs from serial at " +
                    std::to_string(nthreads) + " threads, case " +
                    std::to_string(case_i) + " (" + std::to_string(IC) + "x" +
                    std::to_string(IH) + "x" + std::to_string(IW) + " -> " +
                    std::to_string(c.oc) + "ch, " + std::to_string(c.kh) + "x" +
                    std::to_string(c.kw) + " s" + std::to_string(c.sh) + ")");
            }
        }
        ++case_i;
    }
}

// ============================================================
// Plan and layout contract
// ============================================================

namespace {

/// A dims-only descriptor for plan queries (no data).
TensorDesc shape_desc(std::array<int64_t, 4> s, DataType dt, TensorLayout lay) {
    TensorDesc d;
    for (int64_t v : s) { d.dims.push_back(v); }
    d.rank = 4;
    d.dtype = dt;
    d.layout = lay;
    return d;
}

}  // namespace

NNOPS_TEST(conv2d_nhwc_plan_blocks) {
    using namespace nnops::backend::cpu;

    struct Case {
        std::array<int64_t, 4> ishape;   // N, IC, IH, IW
        std::array<int64_t, 4> wshape;   // OC, IC/G, KH, KW (OIHW, as authored)
    };
    const Case cases[] = {
        {{1, 128, 8, 8},  {64, 128, 1, 1}},   // 1x1: pointwise, whole group
        {{1, 40, 12, 12}, {24, 40, 3, 3}},    // 3x3: icn 16, tail 8, staged pack
        {{1, 16, 12, 12}, {24, 16, 3, 3}},    // 3x3: icn == icg, contiguous slice
        {{1, 64, 20, 20}, {64, 64, 9, 9}},    // 9x9: the rank-1 rung
    };

    for (const auto& c : cases) {
        Conv2DAttributes attrs;
        attrs.kernel_size = {c.wshape[2], c.wshape[3]};

        const std::array<int64_t, 4> w_hwio = {
            c.wshape[2], c.wshape[3], c.wshape[1], c.wshape[0]};
        const TensorDesc ins[] = {shape_desc(c.ishape, DataType::f32, TensorLayout::NHWC),
                                  shape_desc(w_hwio, DataType::f32, TensorLayout::HWIO)};
        const int64_t OH = c.ishape[2] - c.wshape[2] + 1;
        const int64_t OW = c.ishape[3] - c.wshape[3] + 1;
        const TensorDesc outs[] = {
            shape_desc({c.ishape[0], c.wshape[0], OH, OW}, DataType::f32, TensorLayout::NHWC)};

        const auto plan = get_conv2d_implicit_gemm_plan(attrs, ins, outs, /*num_threads=*/1, false);

        NNOPS_EXPECT_TRUE(plan.workspace_size > 0);
        NNOPS_EXPECT_EQ(plan.m_block, int64_t{24});
        NNOPS_EXPECT_TRUE(plan.icn_block >= 1 && plan.icn_block <= c.wshape[1]);
        NNOPS_EXPECT_EQ(plan.ocn_block, std::min<int64_t>(128, c.wshape[0]));
        // A 1x1 filter is the whole point: A is the input row, no gather buffer
        // and no tap-major staging (its k slice is one HWIO rectangle).
        if (c.wshape[2] == 1) {
            NNOPS_EXPECT_TRUE(plan.pointwise);
            NNOPS_EXPECT_EQ(plan.a_size, int64_t{0});
            NNOPS_EXPECT_EQ(plan.stage_size, int64_t{0});
        } else {
            NNOPS_EXPECT_FALSE(plan.pointwise);
            NNOPS_EXPECT_TRUE(plan.a_size > 0);
            // Any multi-tap filter stages: the taps of one block are separate
            // HWIO rectangles whatever the ic block covers.
            NNOPS_EXPECT_TRUE(plan.stage_size > 0);
        }
    }

    // The shared prepack is worth it while the whole packed weight fits the
    // 8 MiB cap; a 512x512x3x3 f32 filter (~9 MB packed) is past it and falls
    // back to a per-task pack.
    {
        Conv2DAttributes attrs;
        attrs.kernel_size = {3, 3};
        const TensorDesc small_ins[] = {
            shape_desc({1, 128, 16, 16}, DataType::f32, TensorLayout::NHWC),
            shape_desc({3, 3, 128, 128}, DataType::f32, TensorLayout::HWIO)};
        const TensorDesc small_outs[] = {
            shape_desc({1, 128, 16, 16}, DataType::f32, TensorLayout::NHWC)};
        const auto small = get_conv2d_implicit_gemm_plan(attrs, small_ins, small_outs, 1, false);
        NNOPS_EXPECT_TRUE(small.shared_elems > 0);
        NNOPS_EXPECT_EQ(small.pack_size, int64_t{0});

        const TensorDesc big_ins[] = {
            shape_desc({1, 512, 8, 8}, DataType::f32, TensorLayout::NHWC),
            shape_desc({3, 3, 512, 512}, DataType::f32, TensorLayout::HWIO)};
        const TensorDesc big_outs[] = {
            shape_desc({1, 512, 8, 8}, DataType::f32, TensorLayout::NHWC)};
        const auto big = get_conv2d_implicit_gemm_plan(attrs, big_ins, big_outs, 1, false);
        NNOPS_EXPECT_EQ(big.shared_elems, int64_t{0});
        NNOPS_EXPECT_TRUE(big.pack_size > 0);
    }
}

NNOPS_TEST(conv2d_nhwc_layout_pairing) {
    // The activation's order and the weight's order must agree: the channel-last
    // kernel reads HWIO, the planar one OIHW. A mismatched pair is a caller
    // error, not something to guess at with a per-call weight reshape.
    const int64_t ishape[] = {1, 4, 8, 8};
    const int64_t wshape[] = {3, 3, 4, 8};   // HWIO [KH][KW][IC/G][OC]
    std::vector<float> in_buf(4 * 8 * 8, 0.0f);
    std::vector<float> w_buf(8 * 4 * 9, 0.0f);

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};

    auto op = Conv2D::create(attrs, Backend::CPU);

    auto compute_with = [&](TensorLayout in_lay, TensorLayout w_lay) {
        TensorView in(ishape, DataType::f32, in_buf.data(), in_lay);
        TensorView wt(wshape, DataType::f32, w_buf.data(), w_lay);
        const TensorDesc descs[] = {in.desc(), wt.desc()};
        const auto outs = op->getOutputTensorDesc(descs);
        std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()), 0.0f);
        auto out = test::make_planar(outs[0], out_buf.data());
        const TensorView ins[] = {in, wt};
        return op->compute(out, ins);
    };

    NNOPS_EXPECT_TRUE(is_ok(compute_with(TensorLayout::NHWC, TensorLayout::HWIO)));
    NNOPS_EXPECT_TRUE(is_ok(compute_with(TensorLayout::NCHW, TensorLayout::NCHW)));
    NNOPS_EXPECT_EQ(compute_with(TensorLayout::NHWC, TensorLayout::NCHW),
                    Status::invalid_argument);
    NNOPS_EXPECT_EQ(compute_with(TensorLayout::NCHW, TensorLayout::HWIO),
                    Status::invalid_argument);
    NNOPS_EXPECT_EQ(op->getLayoutSupport(), LayoutSupport::PlanarOrChannelLast);
}

// ============================================================
// Quantized route
// ============================================================

namespace {

struct QCase {
    std::array<int64_t, 4> ishape = {1, 1, 3, 3};  // N, IC, IH, IW
    std::array<int64_t, 4> wshape = {1, 1, 2, 2};  // OC, IC/G, KH, KW (OIHW)
    int64_t kh = 2, kw = 2, sh = 1, sw = 1, ph = 0, pw = 0, dh = 1, dw = 1;
    int64_t groups = 1;
    DataType act = DataType::s8;      // activation storage / dtype
    DataType out = DataType::f32;     // output dtype
    DataType act_q = DataType::s8;    // act_quant_dtype (fp activations only)
    bool bias = false;
    bool asym_w = false;              // per-channel weight zero point != 0
    float s_x = 0.0078125f;           // 1/128 — exact in binary
    int32_t zp_x = 0;
    float s_out = 0.03125f;
    int32_t zp_out = 0;
    EpilogueActivateType ep = EpilogueActivateType::None;
    uint64_t seed = 1;
};

/// Run one quantized case through the channel-last kernel and the planar
/// reference on the same stored values, and require agreement.
///
/// An integer destination must match bit for bit (the accumulator is int32 and
/// both epilogues run the same requant). An fp destination is compared with a
/// small relative tolerance: the dequant path is also identical arithmetic, but
/// a last-bit difference in it would not be a bug worth failing on.
std::string run_q_case(const QCase& c, const ComputeContext& ctx = {}, bool exact = false)
{
    const int64_t N = c.ishape[0], IC = c.ishape[1];
    const int64_t IH = c.ishape[2], IW = c.ishape[3];
    const int64_t OC = c.wshape[0], ICG = c.wshape[1];
    const bool is_u8 = c.act == DataType::u8;
    const bool int_out = c.out == DataType::s8 || c.out == DataType::u8;
    const size_t esz = elem_size(c.act);
    const int64_t ish[] = {N, IC, IH, IW};
    const int64_t wsh[] = {OC, ICG, c.kh, c.kw};          // OIHW: the reference
    const int64_t wsh_cl[] = {c.kh, c.kw, ICG, OC};       // HWIO: the kernel
    const int64_t bsh[] = {OC};

    // ---- the activation values, then both storage orders ------------------
    std::vector<float> act_nchw(static_cast<size_t>(N * IC * IH * IW));
    {
        test::XorShift128 rng(c.seed);
        rng.fill_float(act_nchw.data(), static_cast<int64_t>(act_nchw.size()), -1.0f, 1.0f);
    }
    const auto act_nhwc = test::nchw_to_nhwc<float>(act_nchw, N, IC, IH, IW);

    auto store_act = [&](const std::vector<float>& v) {
        std::vector<uint8_t> dst(v.size() * esz);
        const float inv = 1.0f / c.s_x;
        for (size_t i = 0; i < v.size(); ++i) {
            if (c.act == DataType::f32) {
                reinterpret_cast<float*>(dst.data())[i] = v[i];
            } else if (c.act == DataType::f16) {
                reinterpret_cast<half*>(dst.data())[i] = float_to_half(v[i]);
            } else {
                dst[i] = static_cast<uint8_t>(quant_int(v[i], inv, c.zp_x, is_u8));
            }
        }
        return dst;
    };
    auto act_ref = store_act(act_nchw);
    auto act_op  = store_act(act_nhwc);

    // ---- the weight, in OIHW (reference) and HWIO (kernel) -----------------
    std::vector<int8_t> w_oihw(static_cast<size_t>(OC * ICG * c.kh * c.kw));
    {
        test::XorShift128 rng(c.seed + 1);
        for (auto& v : w_oihw) {
            v = static_cast<int8_t>(quant_int(rng.next_float(-1.0f, 1.0f), 127.0f, 0, false));
        }
    }
    auto w_hwio = test::oihw_to_hwio<int8_t>(w_oihw, OC, ICG, c.kh, c.kw);

    // ---- bias: the output dtype on an fp destination, int32 on an integer one
    std::vector<uint8_t> bias_buf;
    if (c.bias) {
        bias_buf.resize(static_cast<size_t>(OC) * (int_out ? sizeof(int32_t) : elem_size(c.out)));
        test::XorShift128 rng(c.seed + 2);
        for (int64_t oc = 0; oc < OC; ++oc) {
            if (int_out) {
                reinterpret_cast<int32_t*>(bias_buf.data())[oc] =
                    static_cast<int32_t>(rng.next_float(-1500.0f, 1500.0f));
                continue;
            }
            const float b = rng.next_float(-0.5f, 0.5f);
            if (c.out == DataType::f16) {
                reinterpret_cast<half*>(bias_buf.data())[oc] = float_to_half(b);
            } else {
                reinterpret_cast<float*>(bias_buf.data())[oc] = b;
            }
        }
    }

    // ---- quantization parameters ------------------------------------------
    const QuantParams qa = per_tensor(c.s_x, c.zp_x);
    std::vector<float>   w_s(static_cast<size_t>(OC));
    std::vector<int32_t> w_zp(static_cast<size_t>(OC));
    for (int64_t oc = 0; oc < OC; ++oc) {
        w_s[static_cast<size_t>(oc)] = 0.01f * static_cast<float>(1 + oc % 5);
        w_zp[static_cast<size_t>(oc)] =
            c.asym_w ? static_cast<int32_t>(-40 + (oc * 7) % 60) : 0;
    }
    const QuantParams qw = per_channel(w_s.data(), w_zp.data(), OC);
    const QuantParams qo = int_out ? per_tensor(c.s_out, c.zp_out) : QuantParams{};

    Conv2DAttributes attrs;
    attrs.kernel_size = {c.kh, c.kw};
    attrs.stride = {c.sh, c.sw};
    attrs.dilation = {c.dh, c.dw};
    attrs.padding = {c.ph, c.pw};
    attrs.groups = c.groups;
    attrs.epilogue.type = c.ep;
    attrs.output_dtype = c.out;
    attrs.act_quant_dtype = c.act_q;

    auto op = Conv2D::create(attrs, Backend::CPU);

    // ---- the reference, on the planar views -------------------------------
    TensorView ref_in(std::span<const int64_t>(ish, 4), c.act,
                      act_ref.data(), TensorLayout::NCHW, qa);
    TensorView ref_w(std::span<const int64_t>(wsh, 4), DataType::s8,
                     w_oihw.data(), TensorLayout::NCHW, qw);
    const DataType bdt = int_out ? DataType::s32 : c.out;
    TensorView ref_b(bsh, bdt, bias_buf.empty() ? nullptr : bias_buf.data());

    std::vector<TensorView> ref_ins = {ref_in, ref_w};
    if (c.bias) { ref_ins.push_back(ref_b); }
    std::vector<TensorDesc> ref_descs;
    for (const auto& v : ref_ins) { ref_descs.push_back(v.desc()); }
    const auto od = op->getOutputTensorDesc(ref_descs);

    const int64_t OH = od[0].dims[2], OW = od[0].dims[3];
    const int64_t out_n = od[0].numel();
    const DataType odt = od[0].dtype;

    std::vector<uint8_t> ref_buf(od[0].nbytes(), 0);
    {
        auto ref_out = TensorView(std::span<const int64_t>(od[0].dims.data(), 4), odt,
                                  ref_buf.data(), TensorLayout::NCHW, qo);
        nnops::backend::cpu::reference::conv2d_int8_ref(attrs, ref_out, ref_ins, ctx, nullptr);
    }

    // ---- the kernel, on the channel-last views ----------------------------
    TensorView op_in(std::span<const int64_t>(ish, 4), c.act,
                     act_op.data(), TensorLayout::NHWC, qa);
    TensorView op_w(std::span<const int64_t>(wsh_cl, 4), DataType::s8,
                    w_hwio.data(), TensorLayout::HWIO, qw);
    TensorView op_b(bsh, bdt, bias_buf.empty() ? nullptr : bias_buf.data());
    std::vector<TensorView> op_ins = {op_in, op_w};
    if (c.bias) { op_ins.push_back(op_b); }
    std::vector<TensorDesc> op_descs;
    for (const auto& v : op_ins) { op_descs.push_back(v.desc()); }
    const auto od2 = op->getOutputTensorDesc(op_descs);
    if (od2[0].layout != TensorLayout::NHWC) { return "output layout is not NHWC"; }
    if (od2[0].dtype != odt || od2[0].numel() != out_n) { return "output desc mismatch"; }

    std::vector<uint8_t> op_buf(od2[0].nbytes(), 0);
    {
        auto op_out = TensorView(std::span<const int64_t>(od2[0].dims.data(), 4), odt,
                                 op_buf.data(), TensorLayout::NHWC, qo);
        const Status st = op->compute(op_out, op_ins, ctx, nullptr);
        if (!is_ok(st)) { return std::string("kernel returned ") + to_string(st); }
    }

    // ---- compare in the logical (NCHW) order ------------------------------
    auto compare = [&](auto tag) -> std::string {
        using T = decltype(tag);
        const T* src = reinterpret_cast<const T*>(op_buf.data());
        std::vector<T> nhwc(src, src + out_n);
        const auto got = test::nhwc_to_nchw<T>(nhwc, N, OC, OH, OW);
        const T* want = reinterpret_cast<const T*>(ref_buf.data());
        int64_t nbad = 0;
        int64_t first = -1;
        for (int64_t i = 0; i < out_n; ++i) {
            const float a = as_float(got[static_cast<size_t>(i)]);
            const float b = as_float(want[i]);
            const bool same = (exact || int_out)
                ? (a == b)
                : (std::abs(a - b) <= 1e-5f * std::max(1.0f, std::abs(b)));
            if (!same) {
                if (first < 0) { first = i; }
                ++nbad;
            }
        }
        if (nbad == 0) { return {}; }
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "%lld of %lld elements differ (first at %lld: kernel %.9g, ref %.9g)",
                      static_cast<long long>(nbad), static_cast<long long>(out_n),
                      static_cast<long long>(first),
                      double(as_float(got[static_cast<size_t>(first)])),
                      double(as_float(want[first])));
        return msg;
    };

    if (odt == DataType::f32)   { return compare(float{}); }
    if (odt == DataType::f16)   { return compare(half{}); }
    if (odt == DataType::s8)    { return compare(int8_t{}); }
    return compare(uint8_t{});
}

/// Run a case and throw with the diagnostic when the routes disagree.
void expect_q_match(const QCase& c, int idx, const ComputeContext& ctx = {},
                    bool exact = false)
{
    const std::string diff = run_q_case(c, ctx, exact);
    if (!diff.empty()) {
        throw std::runtime_error("conv2d_nhwc int8 vs reference (case " +
                                 std::to_string(idx) + "): " + diff);
    }
}

}  // namespace

NNOPS_TEST(conv2d_nhwc_i8_hand_exact) {
    // The planar hand case, in channel-last orders:
    //   input  [1,1,3,3] = 1 2 3 / 4 5 6 / 7 8 9     s8, zp_x = 2,  s_x = 0.25
    //   weight [KH,KW,IC,OC] = [2,2,1,1] = 1 -1 / 2 0    s8, zp_w = 1,  s_w = 0.5
    //   accumulators  -1, -3, -7, -9   →  -0.125, -0.375, -0.875, -1.125
    const int64_t ishape[] = {1, 1, 3, 3};
    const int64_t wshape[] = {2, 2, 1, 1};

    std::vector<int8_t> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};   // [h][w][c], c == 1
    std::vector<int8_t> wt = {1, -1, 2, 0};                 // [kh][kw][ic][oc]
    std::vector<float>  ws = {0.5f};
    std::vector<int32_t> wz = {1};

    Conv2DAttributes attrs;
    attrs.kernel_size = {2, 2};
    attrs.stride = {1, 1};
    attrs.padding = {0, 0};

    TensorView input(ishape, DataType::s8, in.data(), TensorLayout::NHWC,
                     per_tensor(0.25f, 2));
    TensorView weight(wshape, DataType::s8, wt.data(), TensorLayout::HWIO,
                      per_channel(ws.data(), wz.data(), 1));

    auto op = Conv2D::create(attrs, Backend::CPU);
    const TensorDesc descs[] = {input.desc(), weight.desc()};
    auto outs = op->getOutputTensorDesc(descs);
    NNOPS_EXPECT_EQ(outs[0].layout, TensorLayout::NHWC);
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t{2});
    NNOPS_EXPECT_EQ(outs[0].dims[3], int64_t{2});

    std::vector<float> out_buf(4, 0.0f);
    auto out = test::make_planar(outs[0], out_buf.data());
    const TensorView ins[] = {input, weight};
    NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));

    const float expected[4] = {-0.125f, -0.375f, -0.875f, -1.125f};
    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[static_cast<size_t>(i)], expected[i], 1e-6f);
    }
}

NNOPS_TEST(conv2d_nhwc_i8_matches_ref_all_routes) {
    std::vector<QCase> cases;
    auto add = [&](QCase c) { cases.push_back(c); };

    // ---- pre-quantized s8 activation, symmetric weight -------------------
    { QCase c; c.ishape = {1, 4, 8, 8}; c.wshape = {8, 4, 3, 3}; c.kh = c.kw = 3;
      add(c); }
    // ---- s8 with a nonzero zero point + zero padding ----------------------
    { QCase c; c.ishape = {2, 3, 7, 7}; c.wshape = {5, 3, 3, 3}; c.kh = c.kw = 3;
      c.ph = c.pw = 1; c.zp_x = -7; add(c); }
    // ---- u8 activation, the classic symmetric form -----------------------
    { QCase c; c.ishape = {1, 4, 8, 8}; c.wshape = {8, 4, 3, 3}; c.kh = c.kw = 3;
      c.act = DataType::u8; c.zp_x = 128; add(c); }
    // ---- u8 with an asymmetric zero point (XOR + r_b correction) ----------
    { QCase c; c.ishape = {1, 4, 9, 9}; c.wshape = {6, 4, 3, 3}; c.kh = c.kw = 3;
      c.ph = c.pw = 1; c.act = DataType::u8; c.zp_x = 173; add(c); }
    // ---- asymmetric weight zero point → the per-pixel r_a sweep runs -----
    { QCase c; c.ishape = {1, 6, 8, 8}; c.wshape = {7, 6, 3, 3}; c.kh = c.kw = 3;
      c.asym_w = true; add(c); }
    // ---- both asymmetric: every compensation term is live ----------------
    { QCase c; c.ishape = {1, 5, 7, 7}; c.wshape = {6, 5, 3, 3}; c.kh = c.kw = 3;
      c.ph = c.pw = 1; c.act = DataType::u8; c.zp_x = 130; c.asym_w = true; add(c); }
    // ---- u8 whose zero point does not cancel the shift (col_term) --------
    { QCase c; c.ishape = {1, 8, 6, 6}; c.wshape = {4, 8, 3, 3}; c.kh = c.kw = 3;
      c.act = DataType::u8; c.zp_x = 120; add(c); }
    // ---- fp32 activation quantized to s8 inside the plane ----------------
    { QCase c; c.ishape = {1, 4, 8, 8}; c.wshape = {8, 4, 3, 3}; c.kh = c.kw = 3;
      c.act = DataType::f32; c.act_q = DataType::s8; add(c); }
    // ---- fp32 activation quantized to u8 ---------------------------------
    { QCase c; c.ishape = {1, 4, 8, 8}; c.wshape = {8, 4, 3, 3}; c.kh = c.kw = 3;
      c.act = DataType::f32; c.act_q = DataType::u8; c.zp_x = 128; c.asym_w = true;
      add(c); }
    // ---- f16 activation with u8 storage, f16 output ----------------------
    { QCase c; c.ishape = {1, 4, 8, 8}; c.wshape = {8, 4, 3, 3}; c.kh = c.kw = 3;
      c.act = DataType::f16; c.out = DataType::f16; c.act_q = DataType::u8;
      c.zp_x = 117; add(c); }
    // ---- f16 activation → f32 output -------------------------------------
    { QCase c; c.ishape = {1, 3, 6, 6}; c.wshape = {4, 3, 3, 3}; c.kh = c.kw = 3;
      c.act = DataType::f16; c.out = DataType::f32; c.act_q = DataType::s8; add(c); }
    // ---- 1x1 pointwise, pre-quantized u8 (the materialized plane) --------
    { QCase c; c.ishape = {1, 32, 6, 6}; c.wshape = {16, 32, 1, 1}; c.kh = c.kw = 1;
      c.act = DataType::u8; c.zp_x = 128; add(c); }
    // ---- 1x1 pointwise, pre-quantized s8 (read in place) ----------------
    { QCase c; c.ishape = {1, 32, 6, 6}; c.wshape = {16, 32, 1, 1}; c.kh = c.kw = 1;
      c.act = DataType::s8; c.zp_x = 0; add(c); }
    // ---- 1x1 whose channel group is not a multiple of 4 (gather, not A
    //      straight out of the row) ----------------------------------------
    { QCase c; c.ishape = {1, 30, 5, 5}; c.wshape = {12, 30, 1, 1}; c.kh = c.kw = 1;
      c.act = DataType::s8; c.asym_w = true; add(c); }
    // ---- 1x1 with an fp activation (plane + pointwise) -------------------
    { QCase c; c.ishape = {1, 16, 5, 5}; c.wshape = {12, 16, 1, 1}; c.kh = c.kw = 1;
      c.act = DataType::f32; add(c); }
    // ---- groups = 2 ------------------------------------------------------
    { QCase c; c.ishape = {1, 4, 8, 8}; c.wshape = {8, 2, 3, 3}; c.kh = c.kw = 3;
      c.groups = 2; add(c); }
    // ---- groups = 4, u8 --------------------------------------------------
    { QCase c; c.ishape = {1, 8, 8, 8}; c.wshape = {16, 2, 3, 3}; c.kh = c.kw = 3;
      c.groups = 4; c.act = DataType::u8; c.zp_x = 120; add(c); }
    // ---- depthwise (ic_per_group 1) --------------------------------------
    { QCase c; c.ishape = {1, 6, 7, 7}; c.wshape = {6, 1, 3, 3}; c.kh = c.kw = 3;
      c.ph = c.pw = 1; c.groups = 6; add(c); }
    // ---- stride 2 + dilation 2 -------------------------------------------
    { QCase c; c.ishape = {1, 4, 16, 16}; c.wshape = {8, 4, 3, 3}; c.kh = c.kw = 3;
      c.sh = c.sw = 2; c.dh = c.dw = 2; c.ph = c.pw = 1; c.zp_x = 3; c.asym_w = true;
      add(c); }
    // ---- several ic blocks with a tail (40 = 2 · 16 + 8), several oc blocks
    { QCase c; c.ishape = {1, 40, 12, 12}; c.wshape = {160, 40, 3, 3}; c.kh = c.kw = 3;
      c.act = DataType::u8; c.zp_x = 128; c.asym_w = true; c.bias = true;
      c.ep = EpilogueActivateType::Relu6; add(c); }
    // ---- batch 2 with bias + sigmoid, fp activation ----------------------
    { QCase c; c.ishape = {2, 8, 9, 9}; c.wshape = {12, 8, 3, 3}; c.kh = c.kw = 3;
      c.act = DataType::f32; c.bias = true; c.ep = EpilogueActivateType::Sigmoid;
      add(c); }
    // ---- a 5x5 and a 7x7 (the wider-kernel rungs) ------------------------
    { QCase c; c.ishape = {1, 8, 16, 16}; c.wshape = {16, 8, 5, 5}; c.kh = c.kw = 5;
      c.ph = c.pw = 2; c.act = DataType::u8; c.zp_x = 128; c.asym_w = true; add(c); }
    { QCase c; c.ishape = {1, 4, 14, 14}; c.wshape = {8, 4, 7, 7}; c.kh = c.kw = 7;
      c.ph = c.pw = 3; add(c); }
    // ---- two K passes on a 1x1 (ic 40 > icn_block 16) -------------------
    { QCase c; c.ishape = {1, 40, 6, 6}; c.wshape = {64, 40, 1, 1}; c.kh = c.kw = 1;
      c.act = DataType::u8; c.zp_x = 128; c.asym_w = true; add(c); }
    // ---- f16 output with bias -------------------------------------------
    { QCase c; c.ishape = {1, 8, 8, 8}; c.wshape = {8, 8, 3, 3}; c.kh = c.kw = 3;
      c.act = DataType::f16; c.out = DataType::f16; c.bias = true; add(c); }

    for (size_t i = 0; i < cases.size(); ++i) {
        cases[i].seed = 1000 + i;
        expect_q_match(cases[i], static_cast<int>(i));
    }
}

NNOPS_TEST(conv2d_nhwc_i8_epilogues) {
    const EpilogueActivateType eps[] = {
        EpilogueActivateType::None,      EpilogueActivateType::Relu,
        EpilogueActivateType::Relu6,     EpilogueActivateType::Gelu,
        EpilogueActivateType::Sigmoid,   EpilogueActivateType::Tanh,
        EpilogueActivateType::LeakyRelu, EpilogueActivateType::Silu,
        EpilogueActivateType::HardSwish, EpilogueActivateType::Elu,
    };
    int idx = 0;
    for (auto ep : eps) {
        for (int with_bias = 0; with_bias < 2; ++with_bias) {
            QCase c;
            c.ishape = {1, 6, 7, 7};
            c.wshape = {8, 6, 3, 3};
            c.kh = c.kw = 3;
            c.ph = c.pw = 1;
            c.act = DataType::s8;
            c.zp_x = -5;
            c.asym_w = true;
            c.bias = with_bias != 0;
            c.ep = ep;
            c.seed = static_cast<uint64_t>(2000 + idx);
            expect_q_match(c, idx++, {}, /*exact=*/true);
        }
    }
}

NNOPS_TEST(conv2d_nhwc_i8_int_out) {
    // Integer destination: int32 bias, requant to s8/u8. Bit-exact, including
    // the per-column bias riding in the epilogue.
    for (DataType odt : {DataType::s8, DataType::u8}) {
        QCase c;
        c.ishape = {1, 8, 9, 9};
        c.wshape = {16, 8, 3, 3};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.out = odt;
        c.act = DataType::u8;
        c.zp_x = 128;
        c.asym_w = true;
        c.bias = true;
        c.s_out = 0.0625f;
        c.zp_out = (odt == DataType::u8) ? 90 : -20;
        c.seed = 3000;
        expect_q_match(c, 0, {}, /*exact=*/true);
    }
}

NNOPS_TEST(conv2d_nhwc_i8_threaded_matches_serial) {
    // Integer accumulation is associative, so the threaded run must be
    // bit-identical to the serial one — the chunk split only partitions tiles.
    QCase c;
    c.ishape = {1, 16, 20, 20};
    c.wshape = {160, 16, 3, 3};
    c.kh = c.kw = 3;
    c.ph = c.pw = 1;
    c.act = DataType::u8;
    c.zp_x = 128;
    c.asym_w = true;
    c.bias = true;
    c.seed = 4000;

    const std::string serial = run_q_case(c, {}, /*exact=*/true);
    if (!serial.empty()) { throw std::runtime_error("serial: " + serial); }

    for (int nthreads : {2, 4}) {
        SimplePool pool(nthreads);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const std::string threaded = run_q_case(c, ctx, /*exact=*/true);
        if (!threaded.empty()) {
            throw std::runtime_error(threaded + " (" + std::to_string(nthreads) +
                                     " threads)");
        }
    }
}

