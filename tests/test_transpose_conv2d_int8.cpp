/// @file test_transpose_conv2d_int8.cpp
/// @brief Bit-exactness tests for the quantized CPU TransposeConv2D route, in
///        both layouts.
///
/// Route matrix: activation u8 / s8 / f16 / f32 (per-tensor QuantParams) ×
/// weight s8 (per-channel QuantParams) → f32 / f16 / s8 / u8 output, across
/// bias, epilogues, groups and depthwise, stride / dilation / padding /
/// output_padding, pitched input and output storage.
///
/// Every table-driven case is compared bit for bit against
/// `transpose_conv2d_int8_ref`. The dot product is integer, so the kernel's
/// GEMM order and the reference's ic-outer order agree exactly, and both sides
/// share the dequant/requant expression. The fp-activation cases use a
/// power-of-two activation scale, which makes x·(1/scale) exact.
///
/// Channel-last cases run the kernel on channel-last storage and the reference
/// on the same values in channel-first order, converting the result back before
/// the comparison: the layouts are a permutation, so the bits must survive it.
///
/// Integer destinations run the epilogue activation on the int32 accumulator
/// before the requant; the asymmetric-activation / symmetric-weight cases give
/// the compensation both live terms.
///
/// Threading is part of the contract: the same case must give identical bits at
/// 2/4/8 threads, since an output row's merge has to stay inside its own task.

#include "nnops/ops/transpose_conv2d.hpp"
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
void transpose_conv2d_int8_ref(const TransposeConv2DAttributes& attrs,
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

inline size_t elem_size(DataType dt) {
    return static_cast<size_t>(data_type_size(dt));
}

struct Case {
    std::array<int64_t, 4> ishape = {1, 1, 3, 3};  // N, IC, IH, IW
    std::array<int64_t, 4> wshape = {1, 1, 2, 2};  // IC, OC/G, KH, KW
    int64_t kh = 2, kw = 2, sh = 1, sw = 1, ph = 0, pw = 0, dh = 1, dw = 1;
    int64_t out_pad_h = 0, out_pad_w = 0;
    int64_t groups = 1;
    DataType act = DataType::s8;      // activation storage / dtype
    DataType out = DataType::f32;     // output dtype
    DataType act_q = DataType::s8;    // act_quant_dtype (fp activations only)
    bool bias = false;
    bool channel_last = false;        // NHWC activation (and therefore output)
    float s_x = 0.0078125f;           // 1/128 — exact in binary
    int32_t zp_x = 0;
    float s_out = 0.03125f;           // requant scale, integer output only
    int32_t zp_out = 0;
    bool asym_w = false;              // per-channel weight zero point != 0
    EpilogueActivateType ep = EpilogueActivateType::None;
    int64_t in_pitch = 0;             // > IW → pitched input storage
    int64_t out_pitch = 0;            // > OW → pitched output storage
    uint64_t seed = 1;
};

/// Owns a case's channel-first buffers and the views over them — views last, so
/// a resize dangles. A channel-last case permutes these on the way in and out.
struct Fixture {
    Case c;
    std::vector<uint8_t> act;         // activation storage
    std::vector<int8_t>  w;           // weight storage [IC][OC/G][KH][KW]
    std::vector<uint8_t> bias;        // bias storage (empty when unused)
    std::vector<float>   w_s;         // per-channel weight scales
    std::vector<int32_t> w_zp;        // per-channel weight zero points
    QuantParams qa, qw;
    QuantParams out_q;                // the requant's parameters (integer output)
    TransposeConv2DAttributes attrs;
    std::vector<TensorView> ins;

    explicit Fixture(const Case& cs) : c(cs) {
        const int64_t N = c.ishape[0], IC = c.ishape[1];
        const int64_t IH = c.ishape[2], IW = c.ishape[3];
        const int64_t OC = c.wshape[1] * c.groups;
        const int64_t pitch = c.in_pitch ? c.in_pitch : IW;
        const bool is_u8 = (c.act == DataType::u8);
        const bool int_out = (c.out == DataType::s8 || c.out == DataType::u8);

        // ---- activation ----
        act.resize(static_cast<size_t>(N * IC * IH * pitch) * elem_size(c.act));
        {
            test::XorShift128 rng(c.seed);
            const float inv = 1.0f / c.s_x;
            for (int64_t i = 0; i < N * IC * IH; ++i) {
                for (int64_t w = 0; w < pitch; ++w) {
                    const int64_t idx = i * pitch + w;
                    // Pitch tail sentinel: reading past the row shows up here.
                    const float x = (w < IW) ? rng.next_float(-1.0f, 1.0f) : 1000.0f;
                    if (c.act == DataType::f32) {
                        reinterpret_cast<float*>(act.data())[idx] = x;
                    } else if (c.act == DataType::f16) {
                        reinterpret_cast<half*>(act.data())[idx] = float_to_half(x);
                    } else {
                        act[static_cast<size_t>(idx)] = static_cast<uint8_t>(
                            quant_int(x, inv, c.zp_x, is_u8));
                    }
                }
            }
        }

        // ---- weight + its per-channel parameters ----
        NNOPS_EXPECT_EQ(c.wshape[0], IC);
        const int64_t wnumel = IC * c.wshape[1] * c.kh * c.kw;
        w.resize(static_cast<size_t>(wnumel));
        w_s.resize(static_cast<size_t>(OC));
        w_zp.resize(static_cast<size_t>(OC));
        {
            test::XorShift128 rng(c.seed + 1);
            for (int64_t i = 0; i < wnumel; ++i) {
                w[static_cast<size_t>(i)] =
                    static_cast<int8_t>(quant_int(rng.next_float(-1.0f, 1.0f), 127.0f, 0, false));
            }
            for (int64_t oc = 0; oc < OC; ++oc) {
                w_s[static_cast<size_t>(oc)] = 0.01f * static_cast<float>(1 + oc % 5);
                w_zp[static_cast<size_t>(oc)] =
                    c.asym_w ? static_cast<int32_t>(-40 + (oc * 7) % 60) : 0;
            }
        }

        // ---- bias: the output dtype on an fp destination, int32 on s8/u8 ----
        if (c.bias) {
            bias.resize(static_cast<size_t>(OC) *
                        (int_out ? sizeof(int32_t) : elem_size(c.out)));
            test::XorShift128 rng(c.seed + 2);
            for (int64_t oc = 0; oc < OC; ++oc) {
                if (int_out) {
                    reinterpret_cast<int32_t*>(bias.data())[oc] =
                        static_cast<int32_t>(rng.next_float(-1500.0f, 1500.0f));
                    continue;
                }
                const float b = rng.next_float(-0.5f, 0.5f);
                if (c.out == DataType::f16) {
                    reinterpret_cast<half*>(bias.data())[oc] = float_to_half(b);
                } else {
                    reinterpret_cast<float*>(bias.data())[oc] = b;
                }
            }
        }

        // ---- views ----
        qa = per_tensor(c.s_x, c.zp_x);
        qw = per_channel(w_s.data(), w_zp.data(), OC);
        if (int_out) { out_q = per_tensor(c.s_out, c.zp_out); }

        attrs.kernel_size = {c.kh, c.kw};
        attrs.stride = {c.sh, c.sw};
        attrs.dilation = {c.dh, c.dw};
        attrs.padding = {c.ph, c.pw};
        attrs.output_padding = {c.out_pad_h, c.out_pad_w};
        attrs.groups = c.groups;
        attrs.epilogue.type = c.ep;
        attrs.output_dtype = c.out;
        attrs.act_quant_dtype = c.act_q;

        const std::span<const int64_t> ishape(c.ishape.data(), 4);
        if (c.in_pitch) {
            ins.push_back(TensorView(ishape, c.act, act.data(),
                                     c.in_pitch * static_cast<int64_t>(elem_size(c.act)),
                                     TensorLayout::NCHW, qa));
        } else {
            ins.push_back(TensorView(ishape, c.act, act.data(), TensorLayout::NCHW, qa));
        }
        const std::span<const int64_t> wshape(c.wshape.data(), 4);
        ins.push_back(TensorView(wshape, DataType::s8, w.data(), TensorLayout::NCHW, qw));
        if (c.bias) {
            const int64_t bshape[] = {OC};
            const DataType bdt = int_out ? DataType::s32 : c.out;
            ins.push_back(TensorView(bshape, bdt, bias.data(), TensorLayout::NCHW));
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

/// A view with the QuantParams an integer destination needs.
TensorView make_out_view(const TensorDesc& desc, void* data, const QuantParams& qp,
                         int64_t pitch_elems = 0) {
    auto shape = std::span<const int64_t>(desc.dims.data(), static_cast<size_t>(desc.rank));
    if (pitch_elems > 0) {
        return TensorView(shape, desc.dtype, data,
                          pitch_elems * static_cast<int64_t>(data_type_size(desc.dtype)),
                          desc.layout, qp);
    }
    return TensorView(shape, desc.dtype, data, desc.layout, qp);
}

struct OutPlan { size_t bytes = 0; int64_t pitch = 0; };

OutPlan out_plan(const TensorDesc& od, const Case& c) {
    const int64_t OW = od.dims[static_cast<size_t>(od.rank) - 1];
    OutPlan p;
    p.pitch = c.out_pitch ? c.out_pitch : OW;
    p.bytes = static_cast<size_t>(od.numel() / OW * p.pitch) * elem_size(od.dtype);
    return p;
}

struct Result {
    std::vector<uint8_t> buf;
    TensorDesc desc;
};

/// [N][C][H][W] (pitched rows allowed) → dense [N][H][W][C], element-wise so it
/// works for every storage width.
std::vector<uint8_t> to_cl(std::span<const uint8_t> src, size_t esz, int64_t N,
                           int64_t C, int64_t H, int64_t W, int64_t pitch) {
    std::vector<uint8_t> dst(src.size());
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h = 0; h < H; ++h) {
                const uint8_t* s = src.data() +
                    (((n * C + c) * H + h) * pitch) * static_cast<int64_t>(esz);
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

Result run_route(const Case& c, const ComputeContext& ctx, bool reference) {
    Fixture f(c);
    auto op = TransposeConv2D::create(f.attrs, Backend::CPU);

    // The channel-first descriptor carries the logical output geometry either
    // way; a channel-last call only changes the storage order.
    std::vector<TensorDesc> cf_ins = f.descs();
    const auto cf_outs = op->getOutputTensorDesc(cf_ins);
    const int64_t N  = cf_outs[0].dims[0];
    const int64_t OC = cf_outs[0].dims[1];
    const int64_t OH = cf_outs[0].dims[2];
    const int64_t OW = cf_outs[0].dims[3];
    const size_t  esz = elem_size(cf_outs[0].dtype);

    if (reference) {
        Result r;
        r.desc = cf_outs[0];
        const OutPlan p = out_plan(cf_outs[0], c);
        r.buf.assign(p.bytes, 0);
        // The same pre-existing contents as the kernel's buffer: a pitched row's
        // tail is nobody's to write, and seeding both makes that show up.
        seed_output(r.buf, cf_outs[0].dtype, c.seed + 3);
        auto out = make_out_view(cf_outs[0], r.buf.data(), f.out_q, p.pitch);
        nnops::backend::cpu::reference::transpose_conv2d_int8_ref(
            f.attrs, out, f.ins, ctx, nullptr);
        return r;
    }

    if (!c.channel_last) {
        Result r;
        r.desc = cf_outs[0];
        const OutPlan p = out_plan(cf_outs[0], c);
        r.buf.assign(p.bytes, 0);
        seed_output(r.buf, cf_outs[0].dtype, c.seed + 3);
        auto out = make_out_view(cf_outs[0], r.buf.data(), f.out_q, p.pitch);
        NNOPS_EXPECT_EQ(op->compute(out, f.ins, ctx, nullptr), Status::ok);
        return r;
    }

    // ---- channel-last: permute the activation in, the output back ----
    const int64_t IC = c.ishape[1], IH = c.ishape[2], IW = c.ishape[3];
    const int64_t in_pitch = c.in_pitch ? c.in_pitch : IW;
    std::vector<uint8_t> act_cl =
        to_cl(f.act, elem_size(c.act), N, IC, IH, IW, in_pitch);

    const int64_t ishape_cl[] = {N, IC, IH, IW};
    TensorView in_cl(std::span<const int64_t>(ishape_cl, 4), c.act, act_cl.data(),
                     TensorLayout::NHWC, f.qa);

    std::vector<TensorDesc> cl_ins;
    cl_ins.push_back(in_cl.desc());
    for (size_t i = 1; i < f.ins.size(); ++i) { cl_ins.push_back(f.ins[i].desc()); }
    const auto cl_outs = op->getOutputTensorDesc(cl_ins);
    NNOPS_EXPECT_EQ(cl_outs[0].dtype, cf_outs[0].dtype);
    NNOPS_EXPECT_EQ(cl_outs[0].layout, TensorLayout::NHWC);

    std::vector<uint8_t> out_cl(static_cast<size_t>(cl_outs[0].numel()) * esz, 0);
    auto cl_view = make_out_view(cl_outs[0], out_cl.data(), f.out_q);

    std::vector<TensorView> cl_views;
    cl_views.push_back(in_cl);
    for (size_t i = 1; i < f.ins.size(); ++i) { cl_views.push_back(f.ins[i]); }
    NNOPS_EXPECT_EQ(op->compute(cl_view, cl_views, ctx, nullptr), Status::ok);

    Result r;
    r.desc = cf_outs[0];
    r.buf = to_cf(out_cl, esz, N, OC, OH, OW);
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
    const Result ref  = run_route(c, ctx, /*reference=*/true);
    const std::string diff = first_difference(fast, ref);
    if (!diff.empty()) {
        throw std::runtime_error(std::string("transpose_conv2d_int8 vs reference (case ") +
                                 std::to_string(idx) + (c.channel_last ? ", NHWC" : ", NCHW") +
                                 "): " + diff);
    }
}

/// Every case runs in both layouts.
void expect_matches_ref_both(Case c, int idx) {
    c.channel_last = false;
    c.in_pitch = c.out_pitch = 0;   // pitched storage is a channel-first case
    expect_matches_ref(c, {}, idx);
    c.channel_last = true;
    expect_matches_ref(c, {}, idx);
}

// ============================================================
// Hand-computed case — pins the whole formula
// ============================================================

NNOPS_TEST(tconv2d_i8_hand_exact) {
    // in [1,1,2,2] = 1..4, s8 zp_x 2 s_x 0.25; w [1,1,2,2] = {1,-1,2,0}, s8
    // zp_w 1 s_w 0.5. The deconv of those two 2x2s is the 3x3 below; acc is
    // (w−1)·(x−2) summed per output pixel, scaled by s_w·s_x = 0.125.
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t wshape[] = {1, 1, 2, 2};

    std::vector<int8_t> in = {1, 2, 3, 4};
    std::vector<int8_t> wt = {1, -1, 2, 0};
    std::vector<float>  ws = {0.5f};
    std::vector<int32_t> wz = {1};

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {2, 2};
    attrs.stride = {1, 1};
    attrs.padding = {0, 0};

    TensorView input(ishape, DataType::s8, in.data(), TensorLayout::NCHW,
                     per_tensor(0.25f, 2));
    TensorView weight(wshape, DataType::s8, wt.data(), TensorLayout::NCHW,
                      per_channel(ws.data(), wz.data(), 1));

    auto op = TransposeConv2D::create(attrs, Backend::CPU);
    const TensorDesc descs[] = {input.desc(), weight.desc()};
    auto outs = op->getOutputTensorDesc(descs);
    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(outs[0].dims[2], 3);
    NNOPS_EXPECT_EQ(outs[0].dims[3], 3);

    std::vector<float> out_buf(9, 0.0f);
    auto out = test::make_channel_first(outs[0], out_buf.data());
    const TensorView ins[] = {input, weight};
    op->compute(out, ins);

    const float expected[9] = {0.0f, 0.25f, 0.0f,
                               -0.125f, -0.125f, -0.5f,
                               0.125f, 0.125f, -0.25f};
    for (int i = 0; i < 9; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[static_cast<size_t>(i)], expected[i], 1e-6f);
    }
}

NNOPS_TEST(tconv2d_i8_int_out_hand_exact) {
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

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {2, 2};
    attrs.stride = {1, 1};
    attrs.padding = {0, 0};
    attrs.output_dtype = DataType::s8;

    TensorView input(ishape, DataType::s8, in.data(), TensorLayout::NCHW,
                     per_tensor(0.25f, 2));
    TensorView weight(wshape, DataType::s8, wt.data(), TensorLayout::NCHW,
                      per_channel(ws.data(), wz.data(), 1));
    TensorView bias_view(bshape, DataType::s32, &bias, TensorLayout::NCHW);

    auto op = TransposeConv2D::create(attrs, Backend::CPU);
    const TensorDesc descs[] = {input.desc(), weight.desc(), bias_view.desc()};
    auto outs = op->getOutputTensorDesc(descs);
    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::s8);

    std::vector<uint8_t> buf(9, 0);
    auto out = make_out_view(outs[0], buf.data(), per_tensor(0.125f, 10));
    const TensorView ins[] = {input, weight, bias_view};
    op->compute(out, ins);

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

NNOPS_TEST(tconv2d_i8_matches_ref_fp_out) {
    std::vector<Case> cases;

    // ---- pre-quantized s8 activation, symmetric weight (no compensation) ----
    {
        Case c;
        c.ishape = {1, 4, 8, 8};
        c.wshape = {4, 8, 3, 3};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        cases.push_back(c);
    }
    // ---- stride 2 (the residue classes) + zero point ----
    {
        Case c;
        c.ishape = {1, 4, 8, 8};
        c.wshape = {4, 8, 3, 3};
        c.kh = c.kw = 3;
        c.sh = c.sw = 2;
        c.ph = c.pw = 1;
        c.zp_x = -7;
        cases.push_back(c);
    }
    // ---- u8 activation: zp == 128 (the classic symmetric u8 form) ----
    {
        Case c;
        c.ishape = {2, 4, 7, 7};
        c.wshape = {4, 6, 3, 3};
        c.kh = c.kw = 3;
        c.act = DataType::u8;
        c.zp_x = 128;
        cases.push_back(c);
    }
    // ---- u8 asymmetric zero point: the plane shift must cancel exactly ----
    {
        Case c;
        c.ishape = {1, 4, 9, 9};
        c.wshape = {4, 6, 3, 3};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.act = DataType::u8;
        c.zp_x = 173;
        cases.push_back(c);
    }
    // ---- asymmetric weight zero point ----
    {
        Case c;
        c.ishape = {1, 6, 8, 8};
        c.wshape = {6, 7, 3, 3};
        c.kh = c.kw = 3;
        c.asym_w = true;
        cases.push_back(c);
    }
    // ---- dilation 2 ----
    {
        Case c;
        c.ishape = {1, 4, 6, 6};
        c.wshape = {4, 8, 3, 3};
        c.kh = c.kw = 3;
        c.dh = c.dw = 2;
        c.asym_w = true;
        cases.push_back(c);
    }
    // ---- output_padding ----
    {
        Case c;
        c.ishape = {1, 4, 5, 5};
        c.wshape = {4, 8, 1, 1};
        c.kh = c.kw = 1;
        c.sh = c.sw = 2;
        c.out_pad_h = c.out_pad_w = 1;
        cases.push_back(c);
    }
    // ---- groups = 2 ----
    {
        Case c;
        c.ishape = {1, 4, 8, 8};
        c.wshape = {4, 4, 3, 3};
        c.kh = c.kw = 3;
        c.groups = 2;
        c.ph = c.pw = 1;
        cases.push_back(c);
    }
    // ---- depthwise (groups == IC), the K = 1 GEMM ----
    {
        Case c;
        c.ishape = {1, 8, 9, 9};
        c.wshape = {8, 1, 3, 3};
        c.kh = c.kw = 3;
        c.groups = 8;
        c.sh = c.sw = 2;
        c.ph = c.pw = 1;
        c.bias = true;
        cases.push_back(c);
    }
    // ---- fp32 activation quantized into the plane (s8 storage) ----
    {
        Case c;
        c.ishape = {1, 4, 8, 8};
        c.wshape = {4, 8, 3, 3};
        c.kh = c.kw = 3;
        c.act = DataType::f32;
        c.act_q = DataType::s8;
        cases.push_back(c);
    }
    // ---- fp32 activation quantized to u8 ----
    {
        Case c;
        c.ishape = {1, 4, 8, 8};
        c.wshape = {4, 8, 3, 3};
        c.kh = c.kw = 3;
        c.act = DataType::f32;
        c.act_q = DataType::u8;
        c.zp_x = 128;
        c.asym_w = true;
        c.bias = true;
        cases.push_back(c);
    }
    // ---- f16 activation and f16 output ----
    {
        Case c;
        c.ishape = {1, 4, 8, 8};
        c.wshape = {4, 8, 3, 3};
        c.kh = c.kw = 3;
        c.act = DataType::f16;
        c.out = DataType::f16;
        c.act_q = DataType::u8;
        c.zp_x = 117;
        c.bias = true;
        c.ep = EpilogueActivateType::Sigmoid;
        cases.push_back(c);
    }
    // ---- wide OC over several chunks, bias + Relu6 ----
    {
        Case c;
        c.ishape = {1, 16, 12, 12};
        c.wshape = {16, 160, 3, 3};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.act = DataType::u8;
        c.zp_x = 128;
        c.asym_w = true;
        c.bias = true;
        c.ep = EpilogueActivateType::Relu6;
        cases.push_back(c);
    }
    // ---- pitched input storage ----
    {
        Case c;
        c.ishape = {1, 4, 8, 7};
        c.wshape = {4, 8, 3, 3};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.in_pitch = 12;
        c.zp_x = -3;
        cases.push_back(c);
    }
    // ---- pitched output storage ----
    {
        Case c;
        c.ishape = {1, 4, 7, 7};
        c.wshape = {4, 8, 3, 3};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.out_pitch = 11;
        c.ep = EpilogueActivateType::Relu;
        cases.push_back(c);
    }

    for (size_t i = 0; i < cases.size(); ++i) {
        cases[i].seed = 1000 + i;
        const bool pitched = cases[i].in_pitch || cases[i].out_pitch;
        if (pitched) {
            expect_matches_ref(cases[i], {}, static_cast<int>(i));
        } else {
            expect_matches_ref_both(cases[i], static_cast<int>(i));
        }
    }
}

// ============================================================
// Epilogues: the clip runs before the activation, matching fp's order
// ============================================================

NNOPS_TEST(tconv2d_i8_epilogues_match_ref) {
    const EpilogueActivateType eps[] = {
        EpilogueActivateType::None,      EpilogueActivateType::Relu,
        EpilogueActivateType::Relu6,     EpilogueActivateType::Gelu,
        EpilogueActivateType::Sigmoid,   EpilogueActivateType::Tanh,
        EpilogueActivateType::LeakyRelu, EpilogueActivateType::Silu,
        EpilogueActivateType::HardSwish, EpilogueActivateType::Elu,
    };
    int seed = 2000;
    for (auto ep : eps) {
        for (int with_bias = 0; with_bias < 2; ++with_bias) {
            Case c;
            c.ishape = {1, 6, 7, 7};
            c.wshape = {6, 8, 3, 3};
            c.kh = c.kw = 3;
            c.ph = c.pw = 1;
            c.zp_x = -5;
            c.asym_w = true;
            c.bias = with_bias != 0;
            c.ep = ep;
            c.seed = static_cast<uint64_t>(seed++);
            expect_matches_ref_both(c, static_cast<int>(seed));
        }
    }
}

// ============================================================
// Integer output (s8 / u8) — int32 bias, requant
// ============================================================

NNOPS_TEST(tconv2d_i8_int_out_matches_ref) {
    std::vector<Case> cases;

    // ---- s8 out / s8 act, asymmetric weight, int32 bias ----
    {
        Case c;
        c.ishape = {1, 4, 8, 8};
        c.wshape = {4, 8, 3, 3};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.zp_x = -9;
        c.out = DataType::s8;
        c.asym_w = true;
        c.bias = true;
        c.zp_out = -7;
        cases.push_back(c);
    }
    // ---- u8 out / u8 act with the classic zp 128 ----
    {
        Case c;
        c.ishape = {1, 4, 8, 8};
        c.wshape = {4, 8, 3, 3};
        c.kh = c.kw = 3;
        c.act = DataType::u8;
        c.zp_x = 128;
        c.out = DataType::u8;
        c.bias = true;
        c.zp_out = 128;
        cases.push_back(c);
    }
    // ---- stride 2 + output_padding, u8 out ----
    {
        Case c;
        c.ishape = {2, 5, 7, 7};
        c.wshape = {5, 6, 3, 3};
        c.kh = c.kw = 3;
        c.sh = c.sw = 2;
        c.ph = c.pw = 1;
        c.out_pad_h = c.out_pad_w = 1;
        c.act = DataType::u8;
        c.zp_x = 173;
        c.out = DataType::u8;
        c.asym_w = true;
        c.zp_out = 3;
        cases.push_back(c);
    }
    // ---- Relu before the requant ----
    {
        Case c;
        c.ishape = {1, 6, 7, 7};
        c.wshape = {6, 8, 3, 3};
        c.kh = c.kw = 3;
        c.out = DataType::s8;
        c.ep = EpilogueActivateType::Relu;
        c.bias = true;
        cases.push_back(c);
    }
    // ---- fp32 activation quantized to s8, s8 out ----
    {
        Case c;
        c.ishape = {1, 4, 8, 8};
        c.wshape = {4, 8, 3, 3};
        c.kh = c.kw = 3;
        c.act = DataType::f32;
        c.act_q = DataType::s8;
        c.out = DataType::s8;
        c.bias = true;
        cases.push_back(c);
    }
    // ---- f16 activation quantized to u8, u8 out ----
    {
        Case c;
        c.ishape = {1, 4, 8, 8};
        c.wshape = {4, 8, 3, 3};
        c.kh = c.kw = 3;
        c.act = DataType::f16;
        c.act_q = DataType::u8;
        c.zp_x = 117;
        c.out = DataType::u8;
        c.zp_out = 120;
        cases.push_back(c);
    }
    // ---- groups = 2 ----
    {
        Case c;
        c.ishape = {1, 4, 8, 8};
        c.wshape = {4, 4, 3, 3};
        c.kh = c.kw = 3;
        c.groups = 2;
        c.out = DataType::s8;
        c.bias = true;
        c.asym_w = true;
        cases.push_back(c);
    }
    // ---- depthwise, u8 out ----
    {
        Case c;
        c.ishape = {1, 8, 9, 9};
        c.wshape = {8, 1, 3, 3};
        c.kh = c.kw = 3;
        c.groups = 8;
        c.ph = c.pw = 1;
        c.out = DataType::u8;
        c.zp_out = 128;
        c.bias = true;
        cases.push_back(c);
    }
    // ---- saturation-heavy requant scale (most outputs clamp) ----
    {
        Case c;
        c.ishape = {1, 16, 10, 10};
        c.wshape = {16, 160, 3, 3};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.act = DataType::u8;
        c.zp_x = 128;
        c.out = DataType::s8;
        c.asym_w = true;
        c.bias = true;
        c.s_out = 0.001953125f;
        cases.push_back(c);
    }
    // ---- pitched input, integer output ----
    {
        Case c;
        c.ishape = {1, 4, 8, 7};
        c.wshape = {4, 8, 3, 3};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.in_pitch = 12;
        c.out = DataType::s8;
        c.bias = true;
        cases.push_back(c);
    }

    for (size_t i = 0; i < cases.size(); ++i) {
        cases[i].seed = 5000 + i;
        const bool pitched = cases[i].in_pitch || cases[i].out_pitch;
        if (pitched) {
            expect_matches_ref(cases[i], {}, static_cast<int>(i));
        } else {
            expect_matches_ref_both(cases[i], static_cast<int>(i));
        }
    }
}

// ============================================================
// Threading: identical bits across thread counts
// ============================================================

/// Minimal fixed-size pool exposing a CpuBackend (test_conv2d.cpp pattern).
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

NNOPS_TEST(tconv2d_i8_threaded_matches_serial) {
    struct TCase {
        std::array<int64_t, 4> ishape, wshape;
        int64_t kh, kw, sh, sw, ph, pw, groups;
        DataType act, out;
        bool channel_last;
    };
    const TCase cases[] = {
        {{1, 8, 24, 24}, {8, 16, 3, 3}, 3, 3, 1, 1, 1, 1, 1, DataType::u8, DataType::f32, false},
        {{1, 8, 24, 24}, {8, 16, 3, 3}, 3, 3, 1, 1, 1, 1, 1, DataType::s8,  DataType::f32, true},
        {{2, 4, 16, 16}, {4, 4, 3, 3},  3, 3, 1, 1, 0, 0, 2, DataType::f32, DataType::s8,  false},
        {{2, 8, 16, 16}, {8, 12, 3, 3}, 3, 3, 2, 2, 1, 1, 1, DataType::u8, DataType::u8,  true},
        {{1, 16, 12, 12}, {16, 32, 3, 3}, 3, 3, 1, 1, 1, 1, 1, DataType::f16, DataType::f16, true},
    };

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
        const auto& tc = cases[ci];
        Case c;
        c.ishape = tc.ishape;
        c.wshape = tc.wshape;
        c.kh = tc.kh; c.kw = tc.kw;
        c.sh = tc.sh; c.sw = tc.sw;
        c.ph = tc.ph; c.pw = tc.pw;
        c.groups = tc.groups;
        c.act = tc.act;
        c.out = tc.out;
        c.channel_last = tc.channel_last;
        if (tc.act == DataType::f32 || tc.act == DataType::f16) {
            c.act_q = DataType::s8;
        }
        c.zp_x = (tc.act == DataType::u8) ? 128 : -4;
        c.asym_w = true;
        c.bias = (ci % 2) == 0;
        c.seed = 4000 + ci;

        const Result serial = run_route(c, {}, false);
        const Result ref    = run_route(c, {}, true);
        NNOPS_EXPECT_TRUE(first_difference(serial, ref).empty());

        for (int nthreads : {2, 4, 8}) {
            SimplePool pool(nthreads);
            ComputeContext ctx;
            ctx.cpu = pool.cpu;
            const Result threaded = run_route(c, ctx, false);
            const std::string diff = first_difference(serial, threaded);
            NNOPS_EXPECT_TRUE(diff.empty());
            if (!diff.empty()) {
                throw std::runtime_error("tconv2d_i8_threaded (" +
                                         std::to_string(nthreads) + " threads): " + diff);
            }
        }
    }
}

// ============================================================
// Output descriptor
// ============================================================

NNOPS_TEST(tconv2d_i8_output_desc) {
    auto desc_of = [](const Case& c) {
        Fixture f(c);
        std::vector<TensorDesc> ins = f.descs();
        auto op = TransposeConv2D::create(f.attrs, Backend::CPU);
        return op->getOutputTensorDesc(ins)[0];
    };

    Case c;
    c.ishape = {1, 4, 6, 6};
    c.wshape = {4, 8, 3, 3};
    c.kh = c.kw = 3;

    // An fp activation keeps its own dtype on an fp destination.
    c.act = DataType::f16;
    NNOPS_EXPECT_EQ(desc_of(c).dtype, DataType::f16);
    c.act = DataType::f32;
    NNOPS_EXPECT_EQ(desc_of(c).dtype, DataType::f32);

    // An integer output comes from output_dtype whatever the activation is.
    c.out = DataType::s8;
    NNOPS_EXPECT_EQ(desc_of(c).dtype, DataType::s8);
    c.out = DataType::u8;
    NNOPS_EXPECT_EQ(desc_of(c).dtype, DataType::u8);
    c.act = DataType::s8;
    NNOPS_EXPECT_EQ(desc_of(c).dtype, DataType::u8);

    // A pre-quantized activation is dequantized to output_dtype.
    c.out = DataType::f32;
    NNOPS_EXPECT_EQ(desc_of(c).dtype, DataType::f32);
    c.out = DataType::f16;
    NNOPS_EXPECT_EQ(desc_of(c).dtype, DataType::f16);
    c.act = DataType::u8;
    NNOPS_EXPECT_EQ(desc_of(c).dtype, DataType::f16);

    // The quantized route also keeps the output geometry.
    c.out = DataType::f32;
    c.ishape = {1, 4, 5, 6};
    c.sh = c.sw = 2;
    c.ph = c.pw = 1;
    const auto d = desc_of(c);
    NNOPS_EXPECT_EQ(d.dims[2], 9);
    NNOPS_EXPECT_EQ(d.dims[3], 11);
}

// ============================================================
// The route refuses what it cannot represent
// ============================================================

NNOPS_TEST(tconv2d_i8_refusals) {
    auto ok_with = [](const Case& c, bool add_to) {
        Fixture f(c);
        f.attrs.add_to = add_to;
        auto op = TransposeConv2D::create(f.attrs, Backend::CPU);
        const auto outs = op->getOutputTensorDesc(f.descs());
        std::vector<uint8_t> buf(
            static_cast<size_t>(outs[0].numel()) * elem_size(outs[0].dtype), 0);
        auto out = make_out_view(outs[0], buf.data(), f.out_q);
        return op->compute(out, f.ins, {}, nullptr) == Status::ok;
    };

    Case base;
    base.ishape = {1, 4, 6, 6};
    base.wshape = {4, 8, 3, 3};
    base.kh = base.kw = 3;

    NNOPS_EXPECT_TRUE(ok_with(base, /*add_to=*/false));
    // add_to would have to keep the pre-existing destination, and the integer
    // accumulate has no fold for it.
    NNOPS_EXPECT_FALSE(ok_with(base, /*add_to=*/true));

    // An fp activation with an fp destination of its own dtype.
    {
        Case c = base;
        c.act = DataType::f32;
        NNOPS_EXPECT_TRUE(ok_with(c, false));
    }

    // s32 is not an output this route writes.
    {
        Case c = base;
        c.out = DataType::s32;
        NNOPS_EXPECT_FALSE(ok_with(c, false));
    }

    // Neither is an fp activation whose destination disagrees with it — a
    // descriptor rule, so it has to be refused by a hand-built view.
    {
        Case c = base;
        c.act = DataType::f32;
        Fixture f(c);
        auto op = TransposeConv2D::create(f.attrs, Backend::CPU);
        const auto outs = op->getOutputTensorDesc(f.descs());
        std::vector<uint8_t> buf(
            static_cast<size_t>(outs[0].numel()) * sizeof(uint16_t), 0);
        const auto shape = std::span<const int64_t>(outs[0].dims.data(),
                                                    static_cast<size_t>(outs[0].rank));
        TensorView out(shape, DataType::f16, buf.data(), outs[0].layout);
        NNOPS_EXPECT_EQ(op->compute(out, f.ins, {}, nullptr), Status::unsupported);
    }
}

}  // namespace
