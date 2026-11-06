/// @file test_depthwise_transpose_conv_int8.cpp
/// @brief Bit-exactness tests for the quantized CPU DepthwiseTransposeConv2D
///        route, in both layouts.
///
/// Route matrix: activation u8 / s8 / f16 / f32 (per-tensor QuantParams) ×
/// weight s8 (per-channel QuantParams) → f32 / f16 / s8 / u8 output, across
/// bias, epilogues, stride / dilation / padding / output_padding.
///
/// Every table-driven case is compared bit for bit against
/// `depthwise_transpose_conv_int8_ref`. The dot product is integer, so the
/// kernel's order and the reference's per-tap order agree exactly, and both
/// sides share the dequant/requant expression. The fp-activation cases use a
/// power-of-two activation scale, which makes x·(1/scale) exact.
///
/// Channel-last cases run the kernel on channel-last storage (the activation as
/// [N][H][W][C], the filter as HWCM [KH][KW][C]) and the reference on the same
/// values in channel-first order, converting the result back before the
/// comparison: the layouts are a permutation, so the bits must survive it.
///
/// Threading is part of the contract: the same case must give identical bits at
/// 2/4/8 threads, since an output row's pixels belong to exactly one task.

#include "nnops/ops/depthwise_transpose_conv.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/detail/half.hpp"
#include "backend/cpu/matmul/matmul.h"
#include "common/test_harness.hpp"
#include "common/cpu_pool.hpp"
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
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace nnops;
using nnops::backend::cpu::half;
using nnops::backend::cpu::float_to_half;
using nnops::backend::cpu::half_to_bits;
using nnops::backend::cpu::half_to_float;

namespace nnops::backend::cpu::reference {
void depthwise_transpose_conv_int8_ref(const DepthwiseTransposeConv2DAttributes& attrs,
                                       TensorView& output,
                                       std::span<const TensorView> inputs,
                                       const ComputeContext& ctx,
                                       void* workspace);
}

namespace {

QuantParams per_tensor(float scale, int32_t zp) {
    QuantParams qp;
    qp.granularity = QuantGranularity::PerTensor;
    qp.scale = scale;
    qp.zero_point = zp;
    return qp;
}

QuantParams per_channel(const float* scale, const int32_t* zp, int64_t n) {
    QuantParams qp;
    qp.granularity = QuantGranularity::PerChannel;
    qp.scale_data = scale;
    qp.zero_point_data = zp;
    qp.num_scales = n;
    return qp;
}

/// float → s8/u8 storage, with the arithmetic the kernels use.
int32_t quant_int(float x, float inv_scale, int32_t zp, bool u8) {
    const int32_t qmin = u8 ? 0 : -128;
    const int32_t qmax = u8 ? 255 : 127;
    const float v = std::fmaf(x, inv_scale, static_cast<float>(zp));
    int32_t q = static_cast<int32_t>(std::nearbyintf(v));
    return std::min(std::max(q, qmin), qmax);
}

/// Spawn-per-call pool, or OpenMP when the build has it — the same helper the
/// fp test uses, so both suites exercise the same team plumbing.
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

inline size_t elem_size(DataType dt) {
    return static_cast<size_t>(data_type_size(dt));
}

struct Case {
    std::array<int64_t, 4> ishape = {1, 1, 3, 3};  // N, C, IH, IW
    int64_t kh = 2, kw = 2, sh = 1, sw = 1, ph = 0, pw = 0, dh = 1, dw = 1;
    int64_t out_pad_h = 0, out_pad_w = 0;
    DataType act = DataType::s8;    // activation storage / dtype
    DataType out = DataType::f32;   // output dtype
    DataType act_q = DataType::s8;  // act_quant_dtype (fp activations only)
    bool bias = false;
    bool channel_last = false;      // NHWC activation (and therefore output)
    float s_x = 0.0078125f;         // 1/128 — exact in binary
    int32_t zp_x = 0;
    float s_out = 0.03125f;         // requant scale, integer output only
    int32_t zp_out = 0;
    bool asym_w = false;            // per-channel weight zero point != 0
    EpilogueActivateType ep = EpilogueActivateType::None;
    uint64_t seed = 1;
};

/// Owns a case's channel-first buffers and the views over them — views last, so
/// a resize dangles. A channel-last case permutes these on the way in and out.
struct Fixture {
    Case c;
    std::vector<uint8_t> act;         // activation storage [N][C][IH][IW]
    std::vector<int8_t>  w;           // weight storage [C][1][KH][KW]
    std::vector<uint8_t> bias;        // bias storage (empty when unused)
    std::vector<float>   w_s;         // per-channel weight scales
    std::vector<int32_t> w_zp;        // per-channel weight zero points
    QuantParams qa, qw, out_q;
    DepthwiseTransposeConv2DAttributes attrs;
    std::vector<TensorView> ins;

    explicit Fixture(const Case& cs) : c(cs) {
        const int64_t N = c.ishape[0], C = c.ishape[1];
        const int64_t IH = c.ishape[2], IW = c.ishape[3];
        const bool is_u8 = (c.act == DataType::u8);
        const bool int_out = (c.out == DataType::s8 || c.out == DataType::u8);

        // ---- activation ----
        act.resize(static_cast<size_t>(N * C * IH * IW) * elem_size(c.act));
        {
            test::XorShift128 rng(c.seed);
            const float inv = 1.0f / c.s_x;
            for (int64_t i = 0; i < N * C * IH * IW; ++i) {
                const float x = rng.next_float(-1.0f, 1.0f);
                if (c.act == DataType::f32) {
                    reinterpret_cast<float*>(act.data())[i] = x;
                } else if (c.act == DataType::f16) {
                    reinterpret_cast<half*>(act.data())[i] = float_to_half(x);
                } else {
                    act[static_cast<size_t>(i)] = static_cast<uint8_t>(
                        quant_int(x, inv, c.zp_x, is_u8));
                }
            }
        }

        // ---- weight + its per-channel parameters ----
        const int64_t wnumel = C * c.kh * c.kw;
        w.resize(static_cast<size_t>(wnumel));
        w_s.resize(static_cast<size_t>(C));
        w_zp.resize(static_cast<size_t>(C));
        {
            test::XorShift128 rng(c.seed + 1);
            for (int64_t i = 0; i < wnumel; ++i) {
                w[static_cast<size_t>(i)] =
                    static_cast<int8_t>(quant_int(rng.next_float(-1.0f, 1.0f), 127.0f, 0, false));
            }
            for (int64_t ch = 0; ch < C; ++ch) {
                w_s[static_cast<size_t>(ch)] = 0.01f * static_cast<float>(1 + ch % 5);
                w_zp[static_cast<size_t>(ch)] =
                    c.asym_w ? static_cast<int32_t>(-40 + (ch * 7) % 60) : 0;
            }
        }

        // ---- bias: the output dtype on an fp destination, int32 on s8/u8 ----
        if (c.bias) {
            bias.resize(static_cast<size_t>(C) *
                        (int_out ? sizeof(int32_t) : elem_size(c.out)));
            test::XorShift128 rng(c.seed + 2);
            for (int64_t ch = 0; ch < C; ++ch) {
                if (int_out) {
                    reinterpret_cast<int32_t*>(bias.data())[ch] =
                        static_cast<int32_t>(rng.next_float(-1500.0f, 1500.0f));
                    continue;
                }
                const float b = rng.next_float(-0.5f, 0.5f);
                if (c.out == DataType::f16) {
                    reinterpret_cast<half*>(bias.data())[ch] = float_to_half(b);
                } else {
                    reinterpret_cast<float*>(bias.data())[ch] = b;
                }
            }
        }

        // ---- views ----
        qa = per_tensor(c.s_x, c.zp_x);
        qw = per_channel(w_s.data(), w_zp.data(), C);
        if (int_out) { out_q = per_tensor(c.s_out, c.zp_out); }

        attrs.kernel_size = {c.kh, c.kw};
        attrs.stride = {c.sh, c.sw};
        attrs.dilation = {c.dh, c.dw};
        attrs.padding = {c.ph, c.pw};
        attrs.output_padding = {c.out_pad_h, c.out_pad_w};
        attrs.epilogue.type = c.ep;
        attrs.output_dtype = c.out;
        attrs.act_quant_dtype = c.act_q;

        const std::span<const int64_t> ishape(c.ishape.data(), 4);
        ins.push_back(TensorView(ishape, c.act, act.data(), TensorLayout::NCHW, qa));
        const int64_t wshape[] = {C, 1, c.kh, c.kw};
        ins.push_back(TensorView(std::span<const int64_t>(wshape, 4), DataType::s8,
                                 w.data(), TensorLayout::NCHW, qw));
        if (c.bias) {
            const int64_t bshape[] = {C};
            const DataType bdt = int_out ? DataType::s32 : c.out;
            ins.push_back(TensorView(std::span<const int64_t>(bshape, 1), bdt,
                                     bias.data(), TensorLayout::NCHW));
        }
    }

    std::vector<TensorDesc> descs() const {
        std::vector<TensorDesc> d;
        d.reserve(ins.size());
        for (const auto& v : ins) { d.push_back(v.desc()); }
        return d;
    }
};

/// Deterministic initial output contents (no add_to on this route: overwritten).
void seed_output(std::vector<uint8_t>& buf, DataType dt, uint64_t seed) {
    test::XorShift128 rng(seed);
    if (dt == DataType::s8 || dt == DataType::u8) {
        for (auto& b : buf) { b = static_cast<uint8_t>(rng.next_u64() & 0xFFu); }
        return;
    }
    const int64_t n = static_cast<int64_t>(buf.size() / elem_size(dt));
    for (int64_t i = 0; i < n; ++i) {
        const float v = rng.next_float(-1.0f, 1.0f);
        if (dt == DataType::f16) {
            reinterpret_cast<half*>(buf.data())[i] = float_to_half(v);
        } else {
            reinterpret_cast<float*>(buf.data())[i] = v;
        }
    }
}

/// A view over a descriptor's geometry, with the QuantParams an integer
/// destination needs.
TensorView make_view(const TensorDesc& desc, void* data, const QuantParams& qp = {}) {
    auto shape = std::span<const int64_t>(desc.dims.data(), static_cast<size_t>(desc.rank));
    return TensorView(shape, desc.dtype, data, desc.layout, qp);
}

/// [N][C][H][W] → dense [N][H][W][C], element-wise so it works for every
/// storage width.
std::vector<uint8_t> to_cl(std::span<const uint8_t> src, size_t esz, int64_t N,
                           int64_t C, int64_t H, int64_t W) {
    std::vector<uint8_t> dst(src.size());
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h = 0; h < H; ++h) {
                const uint8_t* s = src.data() + (((n * C + c) * H + h) * W) *
                                                static_cast<int64_t>(esz);
                for (int64_t w = 0; w < W; ++w) {
                    std::memcpy(dst.data() + ((((n * H + h) * W + w) * C + c) *
                                              static_cast<int64_t>(esz)),
                                s + w * static_cast<int64_t>(esz), esz);
                }
            }
        }
    }
    return dst;
}

/// The inverse: dense [N][H][W][C] → dense [N][C][H][W].
std::vector<uint8_t> to_cf(std::span<const uint8_t> src, size_t esz, int64_t N,
                           int64_t C, int64_t H, int64_t W) {
    std::vector<uint8_t> dst(src.size());
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    std::memcpy(dst.data() + (((n * C + c) * H + h) * W + w) *
                                              static_cast<int64_t>(esz),
                                src.data() + ((((n * H + h) * W + w) * C + c) *
                                              static_cast<int64_t>(esz)), esz);
                }
            }
        }
    }
    return dst;
}

struct Result {
    std::vector<uint8_t> buf;
    TensorDesc desc;
    Status status = Status::ok;
};

Result run_route(const Case& c, const ComputeContext& ctx, bool reference) {
    Fixture f(c);
    auto op = DepthwiseTransposeConv2D::create(f.attrs, Backend::CPU);

    // The channel-first descriptor carries the logical output geometry either
    // way; a channel-last call only changes the storage order.
    const std::vector<TensorDesc> cf_ins = f.descs();
    const auto cf_outs = op->getOutputTensorDesc(cf_ins);
    const int64_t N  = cf_outs[0].dims[0];
    const int64_t C  = cf_outs[0].dims[1];
    const int64_t OH = cf_outs[0].dims[2];
    const int64_t OW = cf_outs[0].dims[3];
    const size_t  esz = elem_size(cf_outs[0].dtype);

    Result r;
    r.desc = cf_outs[0];

    if (reference || !c.channel_last) {
        r.buf.assign(static_cast<size_t>(cf_outs[0].numel()) * esz, 0);
        seed_output(r.buf, cf_outs[0].dtype, c.seed + 3);
        auto out = make_view(cf_outs[0], r.buf.data(), f.out_q);
        if (reference) {
            nnops::backend::cpu::reference::depthwise_transpose_conv_int8_ref(
                f.attrs, out, f.ins, ctx, nullptr);
        } else {
            r.status = op->compute(out, f.ins, ctx, nullptr);
        }
        return r;
    }

    // ---- channel-last: permute the activation in, the output back ----
    const int64_t IH = c.ishape[2], IW = c.ishape[3];
    std::vector<uint8_t> act_cl =
        to_cl(f.act, elem_size(c.act), N, C, IH, IW);

    const int64_t ishape_cl[] = {N, C, IH, IW};
    TensorView in_cl(std::span<const int64_t>(ishape_cl, 4), c.act, act_cl.data(),
                     TensorLayout::NHWC, f.qa);

    const std::vector<int8_t> w_cl = test::depthwise_to_hwcm<int8_t>(
        std::span<const int8_t>(f.w.data(), f.w.size()), C, c.kh * c.kw);
    const int64_t wshape_cl[] = {c.kh, c.kw, C};
    TensorView w_cl_v(std::span<const int64_t>(wshape_cl, 3), DataType::s8,
                      const_cast<int8_t*>(w_cl.data()), TensorLayout::HWCM, f.qw);

    std::vector<TensorDesc> cl_ins;
    cl_ins.push_back(in_cl.desc());
    cl_ins.push_back(w_cl_v.desc());
    if (c.bias) { cl_ins.push_back(f.ins.back().desc()); }
    const auto cl_outs = op->getOutputTensorDesc(cl_ins);
    NNOPS_EXPECT_EQ(cl_outs[0].dtype, cf_outs[0].dtype);
    NNOPS_EXPECT_EQ(cl_outs[0].layout, TensorLayout::NHWC);

    std::vector<uint8_t> out_cl(static_cast<size_t>(cl_outs[0].numel()) * esz, 0);
    seed_output(out_cl, cl_outs[0].dtype, c.seed + 3);
    auto cl_view = make_view(cl_outs[0], out_cl.data(), f.out_q);

    std::vector<TensorView> cl_views;
    cl_views.push_back(in_cl);
    cl_views.push_back(w_cl_v);
    if (c.bias) { cl_views.push_back(f.ins.back()); }
    r.status = op->compute(cl_view, cl_views, ctx, nullptr);

    r.buf = to_cf(out_cl, esz, N, C, OH, OW);
    return r;
}

/// First differing output element, as a description (empty when bit-identical).
std::string first_difference(const Result& a, const Result& b) {
    if (a.buf.size() != b.buf.size()) { return "size mismatch"; }
    if (std::memcmp(a.buf.data(), b.buf.data(), a.buf.size()) == 0) { return {}; }

    const size_t esz = elem_size(a.desc.dtype);
    const size_t n = a.buf.size() / esz;
    for (size_t i = 0; i < n; ++i) {
        if (std::memcmp(a.buf.data() + i * esz, b.buf.data() + i * esz, esz) != 0) {
            char msg[160];
            if (a.desc.dtype == DataType::f16) {
                const half* x = reinterpret_cast<const half*>(a.buf.data()) + i;
                const half* y = reinterpret_cast<const half*>(b.buf.data()) + i;
                std::snprintf(msg, sizeof(msg),
                              "elem %zu: kernel 0x%04x (%.6f) vs ref 0x%04x (%.6f)",
                              i, unsigned(half_to_bits(*x)), double(half_to_float(*x)),
                              unsigned(half_to_bits(*y)), double(half_to_float(*y)));
            } else if (a.desc.dtype == DataType::s8) {
                std::snprintf(msg, sizeof(msg), "elem %zu: kernel %d vs ref %d", i,
                              int(static_cast<int8_t>(a.buf[i])),
                              int(static_cast<int8_t>(b.buf[i])));
            } else if (a.desc.dtype == DataType::u8) {
                std::snprintf(msg, sizeof(msg), "elem %zu: kernel %d vs ref %d", i,
                              int(a.buf[i]), int(b.buf[i]));
            } else {
                const float* x = reinterpret_cast<const float*>(a.buf.data()) + i;
                const float* y = reinterpret_cast<const float*>(b.buf.data()) + i;
                std::snprintf(msg, sizeof(msg), "elem %zu: kernel %.9g vs ref %.9g", i,
                              double(*x), double(*y));
            }
            return msg;
        }
    }
    return "byte mismatch with equal elements";
}

/// Run a case through both routes and require bit-identical output.
void expect_matches_ref(const Case& c, const ComputeContext& ctx = {}, int idx = -1) {
    const Result fast = run_route(c, ctx, /*reference=*/false);
    if (fast.status != Status::ok) {
        throw std::runtime_error(std::string("depthwise_transpose_conv_int8 (case ") +
                                 std::to_string(idx) + (c.channel_last ? ", NHWC" : ", NCHW") +
                                 "): compute returned " +
                                 std::to_string(static_cast<int>(fast.status)));
    }
    const Result ref = run_route(c, ctx, /*reference=*/true);
    const std::string diff = first_difference(fast, ref);
    if (!diff.empty()) {
        throw std::runtime_error(std::string("depthwise_transpose_conv_int8 vs reference (case ") +
                                 std::to_string(idx) + (c.channel_last ? ", NHWC" : ", NCHW") +
                                 "): " + diff);
    }
}

/// Every case runs in both layouts.
void expect_matches_ref_both(Case c, int idx) {
    c.channel_last = false;
    expect_matches_ref(c, {}, idx);
    c.channel_last = true;
    expect_matches_ref(c, {}, idx);
}

// ============================================================
// Hand-computed cases — pin the whole formula
// ============================================================

NNOPS_TEST(dwt_i8_hand_exact) {
    // in [1,1,2,2] = 1..4, s8 zp_x 2 s_x 0.25; w [1,1,2,2] = {1,-1,2,0}, s8
    // zp_w 1 s_w 0.5. The deconv of those two 2x2s is the 3x3 below; acc is
    // (w−1)·(x−2) summed per output pixel, scaled by s_w·s_x = 0.125.
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t wshape[] = {1, 1, 2, 2};

    std::vector<int8_t> in = {1, 2, 3, 4};
    std::vector<int8_t> wt = {1, -1, 2, 0};
    std::vector<float>  ws = {0.5f};
    std::vector<int32_t> wz = {1};

    DepthwiseTransposeConv2DAttributes attrs;
    attrs.kernel_size = {2, 2};
    attrs.stride = {1, 1};
    attrs.padding = {0, 0};

    TensorView input(ishape, DataType::s8, in.data(), TensorLayout::NCHW,
                     per_tensor(0.25f, 2));
    TensorView weight(wshape, DataType::s8, wt.data(), TensorLayout::NCHW,
                      per_channel(ws.data(), wz.data(), 1));

    auto op = DepthwiseTransposeConv2D::create(attrs, Backend::CPU);
    const TensorDesc descs[] = {input.desc(), weight.desc()};
    auto outs = op->getOutputTensorDesc(descs);
    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(outs[0].dims[2], 3);
    NNOPS_EXPECT_EQ(outs[0].dims[3], 3);

    std::vector<float> out_buf(9, 0.0f);
    auto out = make_view(outs[0], out_buf.data());
    const TensorView ins[] = {input, weight};
    NNOPS_EXPECT_EQ(op->compute(out, ins), Status::ok);

    const float expected[9] = {0.0f, 0.25f, 0.0f,
                               -0.125f, -0.125f, -0.5f,
                               0.125f, 0.125f, -0.25f};
    for (int i = 0; i < 9; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[static_cast<size_t>(i)], expected[i], 1e-6f);
    }
}

NNOPS_TEST(dwt_i8_int_out_hand_exact) {
    // The same operands on an s8 destination: req = s_w·s_x / s_out = 0.125 /
    // 0.125 = 1, so the output is acc + bias + zp_out.
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t wshape[] = {1, 1, 2, 2};
    const int64_t bshape[] = {1};

    std::vector<int8_t> in = {1, 2, 3, 4};
    std::vector<int8_t> wt = {1, -1, 2, 0};
    std::vector<float>  ws = {0.5f};
    std::vector<int32_t> wz = {1};
    int32_t bias = 5;

    DepthwiseTransposeConv2DAttributes attrs;
    attrs.kernel_size = {2, 2};
    attrs.stride = {1, 1};
    attrs.padding = {0, 0};
    attrs.output_dtype = DataType::s8;

    TensorView input(ishape, DataType::s8, in.data(), TensorLayout::NCHW,
                     per_tensor(0.25f, 2));
    TensorView weight(wshape, DataType::s8, wt.data(), TensorLayout::NCHW,
                      per_channel(ws.data(), wz.data(), 1));
    TensorView bias_view(bshape, DataType::s32, &bias, TensorLayout::NCHW);

    auto op = DepthwiseTransposeConv2D::create(attrs, Backend::CPU);
    const TensorDesc descs[] = {input.desc(), weight.desc(), bias_view.desc()};
    auto outs = op->getOutputTensorDesc(descs);
    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::s8);

    std::vector<uint8_t> buf(9, 0);
    auto out = make_view(outs[0], buf.data(), per_tensor(0.125f, 10));
    const TensorView ins[] = {input, weight, bias_view};
    NNOPS_EXPECT_EQ(op->compute(out, ins), Status::ok);

    // acc = {0, 2, 0, -1, -1, -4, 1, 1, -2}; + bias + zp_out.
    const int32_t expected[9] = {15, 17, 15, 14, 14, 11, 16, 16, 13};
    for (int i = 0; i < 9; ++i) {
        NNOPS_EXPECT_EQ(int(static_cast<int8_t>(buf[static_cast<size_t>(i)])),
                        int(expected[i]));
    }
}

// ============================================================
// Table-driven: every route against the reference, bit for bit
// ============================================================

NNOPS_TEST(dwt_i8_matches_ref_fp_out) {
    std::vector<Case> cases;

    // ---- pre-quantized s8 activation, symmetric weight ----
    {
        Case c;
        c.ishape = {1, 4, 8, 8};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        cases.push_back(c);
    }
    // ---- stride 2 (the residue classes) + a zero point ----
    {
        Case c;
        c.ishape = {1, 6, 9, 9};
        c.kh = c.kw = 3;
        c.sh = c.sw = 2;
        c.ph = c.pw = 1;
        c.zp_x = -7;
        cases.push_back(c);
    }
    // ---- u8 activation: zp == 128 (the classic symmetric u8 form) ----
    {
        Case c;
        c.ishape = {2, 8, 7, 7};
        c.kh = c.kw = 3;
        c.act = DataType::u8;
        c.zp_x = 128;
        cases.push_back(c);
    }
    // ---- u8 asymmetric zero point: the shift must cancel exactly ----
    {
        Case c;
        c.ishape = {1, 5, 8, 9};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.act = DataType::u8;
        c.zp_x = 173;
        cases.push_back(c);
    }
    // ---- asymmetric weight zero point ----
    {
        Case c;
        c.ishape = {1, 17, 8, 8};
        c.kh = c.kw = 3;
        c.asym_w = true;
        cases.push_back(c);
    }
    // ---- dilation 2 ----
    {
        Case c;
        c.ishape = {1, 8, 7, 7};
        c.kh = c.kw = 3;
        c.dh = c.dw = 2;
        cases.push_back(c);
    }
    // ---- stride 2 with output_padding: the odd trailing edge ----
    {
        Case c;
        c.ishape = {2, 3, 6, 7};
        c.kh = 5; c.kw = 5;
        c.sh = c.sw = 2;
        c.ph = c.pw = 2;
        c.out_pad_h = c.out_pad_w = 1;
        cases.push_back(c);
    }
    // ---- bias + relu6 on an s8 activation ----
    {
        Case c;
        c.ishape = {1, 16, 6, 6};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.bias = true;
        c.ep = EpilogueActivateType::Relu6;
        cases.push_back(c);
    }
    // ---- f32 activation, quantized to u8 inside the kernel ----
    {
        Case c;
        c.ishape = {1, 8, 8, 8};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.act = DataType::f32;
        c.act_q = DataType::u8;
        c.bias = true;
        cases.push_back(c);
    }
    // ---- f16 activation and destination ----
    {
        Case c;
        c.ishape = {1, 24, 9, 9};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.act = DataType::f16;
        c.out = DataType::f16;
        c.bias = true;
        c.ep = EpilogueActivateType::Relu;
        cases.push_back(c);
    }
    // ---- 1x1: one tap, no residue split ----
    {
        Case c;
        c.ishape = {2, 5, 9, 9};
        c.kh = c.kw = 1;
        c.sh = c.sw = 2;
        cases.push_back(c);
    }

    for (size_t i = 0; i < cases.size(); ++i) {
        expect_matches_ref_both(cases[i], static_cast<int>(i));
    }
}

NNOPS_TEST(dwt_i8_matches_ref_int_out) {
    std::vector<Case> cases;

    // ---- s8 destination, int32 bias, Relu before the requant ----
    {
        Case c;
        c.ishape = {1, 8, 8, 8};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.out = DataType::s8;
        c.s_out = 0.125f;
        c.zp_out = -3;
        c.bias = true;
        c.ep = EpilogueActivateType::Relu;
        cases.push_back(c);
    }
    // ---- u8 destination with a non-zero zero point ----
    {
        Case c;
        c.ishape = {2, 5, 9, 10};
        c.kh = c.kw = 3;
        c.sh = c.sw = 2;
        c.ph = c.pw = 1;
        c.out = DataType::u8;
        c.s_out = 0.0625f;
        c.zp_out = 100;
        c.bias = true;
        cases.push_back(c);
    }
    // ---- u8 activation into a u8 destination ----
    {
        Case c;
        c.ishape = {1, 16, 7, 7};
        c.kh = c.kw = 3;
        c.act = DataType::u8;
        c.zp_x = 120;
        c.out = DataType::u8;
        c.zp_out = 128;
        c.asym_w = true;
        cases.push_back(c);
    }
    // ---- f16 activation into an s8 destination ----
    {
        Case c;
        c.ishape = {1, 8, 8, 8};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.act = DataType::f16;
        c.out = DataType::s8;
        c.bias = true;
        c.ep = EpilogueActivateType::Relu;
        cases.push_back(c);
    }

    for (size_t i = 0; i < cases.size(); ++i) {
        expect_matches_ref_both(cases[i], static_cast<int>(i));
    }
}

// ============================================================
// Threading — the same bits at every team size
// ============================================================

NNOPS_TEST(dwt_i8_threaded_matches_serial) {
    // The channel-last route splits a row into strips when the row grid cannot
    // fill the pool; the channel-first one gives each (n, c, oh) to one task.
    // Both must land on the same bytes.
    struct ThreadCase { Case c; };
    std::vector<ThreadCase> cases;
    {
        Case c;
        c.ishape = {1, 17, 9, 9};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.ep = EpilogueActivateType::Relu6;
        cases.push_back({c});
    }
    {
        Case c;
        c.ishape = {2, 8, 12, 13};
        c.kh = c.kw = 3;
        c.sh = c.sw = 2;
        c.ph = c.pw = 1;
        c.act = DataType::u8;
        c.zp_x = 128;
        c.out = DataType::u8;
        cases.push_back({c});
    }

    for (size_t ci = 0; ci < cases.size(); ++ci) {
        for (bool channel_last : {false, true}) {
            Case c = cases[ci].c;
            c.channel_last = channel_last;
            const Result serial = run_route(c, {}, static_cast<int>(ci));
            NNOPS_EXPECT_EQ(serial.status, Status::ok);

            for (int nt : {2, 4, 8}) {
                SimplePool pool(nt);
                ComputeContext ctx;
                ctx.cpu = pool.cpu;
                const Result threaded = run_route(c, ctx, static_cast<int>(ci));
                NNOPS_EXPECT_EQ(threaded.status, Status::ok);
                NNOPS_EXPECT_TRUE(std::memcmp(serial.buf.data(), threaded.buf.data(),
                                              serial.buf.size()) == 0);
            }
        }
    }
}

// ============================================================
// Descriptors and refusals
// ============================================================

NNOPS_TEST(dwt_i8_output_desc) {
    // f32 activation: the output follows the activation unless output_dtype is
    // an integer storage.
    const int64_t ishape[] = {1, 8, 6, 6};
    const int64_t wshape[] = {8, 1, 3, 3};
    std::vector<float> in(1 * 8 * 6 * 6, 0.0f);
    std::vector<int8_t> w(8 * 9, 1);
    std::vector<float> ws(8, 0.01f);
    std::vector<int32_t> wz(8, 0);

    TensorView input(ishape, DataType::f32, in.data(), TensorLayout::NCHW,
                     per_tensor(0.125f, 0));
    TensorView weight(wshape, DataType::s8, w.data(), TensorLayout::NCHW,
                      per_channel(ws.data(), wz.data(), 8));
    const TensorDesc descs[] = {input.desc(), weight.desc()};

    DepthwiseTransposeConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride = {2, 2};
    attrs.padding = {1, 1};
    attrs.output_padding = {1, 1};

    auto op = DepthwiseTransposeConv2D::create(attrs, Backend::CPU);
    auto outs = op->getOutputTensorDesc(descs);
    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(outs[0].dims[1], 8);
    // (6-1)*2 - 2 + 1*(3-1) + 1 + 1 = 12
    NNOPS_EXPECT_EQ(outs[0].dims[2], 12);
    NNOPS_EXPECT_EQ(outs[0].dims[3], 12);

    attrs.output_dtype = DataType::u8;
    op = DepthwiseTransposeConv2D::create(attrs, Backend::CPU);
    outs = op->getOutputTensorDesc(descs);
    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::u8);

    // A channel-last call reports the matching layout.
    const int64_t ishape_cl[] = {1, 8, 6, 6};
    TensorView in_cl(ishape_cl, DataType::f32, in.data(), TensorLayout::NHWC,
                     per_tensor(0.125f, 0));
    std::vector<int8_t> w_cl(w.size());
    const int64_t wshape_cl[] = {3, 3, 8};
    TensorView w_cl_v(wshape_cl, DataType::s8, w_cl.data(), TensorLayout::HWCM,
                      per_channel(ws.data(), wz.data(), 8));
    const TensorDesc cl_descs[] = {in_cl.desc(), w_cl_v.desc()};
    op = DepthwiseTransposeConv2D::create(attrs, Backend::CPU);
    const auto cl_outs = op->getOutputTensorDesc(cl_descs);
    NNOPS_EXPECT_EQ(cl_outs[0].layout, TensorLayout::NHWC);
    NNOPS_EXPECT_EQ(cl_outs[0].dtype, DataType::u8);
}

NNOPS_TEST(dwt_i8_refusals) {
    const int64_t ishape[] = {1, 4, 6, 6};
    const int64_t wshape[] = {4, 1, 3, 3};
    std::vector<int8_t> in(4 * 36, 1);
    std::vector<int8_t> w(4 * 9, 1);
    std::vector<float> ws(4, 0.01f);
    std::vector<int32_t> wz(4, 0);

    TensorView input(ishape, DataType::s8, in.data(), TensorLayout::NCHW,
                     per_tensor(0.125f, 0));
    TensorView weight(wshape, DataType::s8, w.data(), TensorLayout::NCHW,
                      per_channel(ws.data(), wz.data(), 4));

    // A weight layout from the other family is refused, not reinterpreted.
    const int64_t wshape_bad[] = {3, 3, 4};
    TensorView w_bad(wshape_bad, DataType::s8, w.data(), TensorLayout::HWCM,
                     per_channel(ws.data(), wz.data(), 4));
    {
        DepthwiseTransposeConv2DAttributes attrs;
        attrs.kernel_size = {3, 3};
        auto op = DepthwiseTransposeConv2D::create(attrs, Backend::CPU);
        const TensorDesc d[] = {input.desc(), w_bad.desc()};
        auto outs = op->getOutputTensorDesc(d);
        std::vector<float> out(static_cast<size_t>(outs[0].numel()), 0.0f);
        TensorView out_v = make_view(outs[0], out.data());
        const TensorView ins[] = {input, w_bad};
        NNOPS_EXPECT_EQ(op->compute(out_v, ins), Status::unsupported);
    }

    // A channel count that disagrees with the activation's is a descriptor
    // error, not an unsupported one.
    {
        const int64_t wshape_c[] = {5, 1, 3, 3};
        std::vector<int8_t> w5(5 * 9, 1);
        std::vector<float> ws5(5, 0.01f);
        std::vector<int32_t> wz5(5, 0);
        TensorView w5_v(wshape_c, DataType::s8, w5.data(), TensorLayout::NCHW,
                        per_channel(ws5.data(), wz5.data(), 5));
        DepthwiseTransposeConv2DAttributes attrs;
        attrs.kernel_size = {3, 3};
        auto op = DepthwiseTransposeConv2D::create(attrs, Backend::CPU);
        const TensorDesc d[] = {input.desc(), w5_v.desc()};
        auto outs = op->getOutputTensorDesc(d);
        std::vector<float> out(static_cast<size_t>(outs[0].numel()), 0.0f);
        TensorView out_v = make_view(outs[0], out.data());
        const TensorView ins[] = {input, w5_v};
        NNOPS_EXPECT_EQ(op->compute(out_v, ins), Status::invalid_argument);
    }

    // An output dtype with no arithmetic is refused.
    {
        DepthwiseTransposeConv2DAttributes attrs;
        attrs.kernel_size = {3, 3};
        attrs.output_dtype = DataType::f8_e4m3;
        auto op = DepthwiseTransposeConv2D::create(attrs, Backend::CPU);
        const TensorDesc d[] = {input.desc(), weight.desc()};
        auto outs = op->getOutputTensorDesc(d);
        std::vector<float> out(static_cast<size_t>(outs[0].numel()), 0.0f);
        TensorView out_v = make_view(outs[0], out.data());
        const TensorView ins[] = {input, weight};
        NNOPS_EXPECT_EQ(op->compute(out_v, ins), Status::unsupported);
    }
}

}  // namespace
