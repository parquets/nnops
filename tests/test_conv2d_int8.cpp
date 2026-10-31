/// @file test_conv2d_int8.cpp
/// @brief Bit-exactness tests for the quantized CPU Conv2D route.
///
/// Route matrix: activation u8 / s8 / f16 / f32 (per-tensor QuantParams) ×
/// weight s8 (per-channel QuantParams) → f32 / f16 / s8 / u8 output, across
/// bias, every epilogue activation, groups, depthwise, stride/dilation, pitched
/// input and the prepacked filter.
///
/// Every table-driven case is compared bit for bit against `conv2d_int8_ref`:
/// the GEMM is integer, and both sides share the dequant expression, so
/// agreement is exact (f16 outputs are compared by their bit patterns). The
/// fp-activation cases use power-of-two activation scales, which makes
/// `x · (1/scale)` exact — the reference's fma and the vector quant kernel's
/// mul+add then agree bitwise as well.
///
/// Threading is part of the contract: the same case must give identical bits at
/// 2/4/8 threads, since an oc-block's epilogue has to stay inside its own task.
///
/// An integer destination runs the epilogue activation on the int32 accumulator
/// before the requant, and a per-channel compensation folds into the bias — the
/// asymmetric-activation / symmetric-weight case uses a non-zero zp_eff so that
/// fold is not the identity and the reference still has to match exactly.
///
/// The fp-activation scratch layout is pinned by conv2d_i8_fp_act_large_layout
/// at a size where a bad offset is not absorbed by the arena's tail: the
/// quantized plane (plan.quant_size) is quantized once per call into the front
/// of the arena and shared across slots, every per-slot region starts past it,
/// and acc / pack_a are absolute offsets that already account for both, the
/// plan's workspace_size being the layout's total. Addressing acc/pack_a from
/// the plane rather than from the arena start once put that write past the end
/// of the allocation — silent in Release, heap corruption in a debug CRT.
///
/// The prepacked filter must not change the answer. Here the prepack is not
/// cosmetic: the A operand is a set of MMA panels, so a raw filter pays a pack
/// per task on every call, and the blob also carries the per-channel sums the
/// epilogue compensates with — which is what keeps the packed call cheaper than
/// the raw one rather than merely equal. The panels hold the same bytes in the
/// layout the MMA wanted, so raw and packed must be bit-identical.

#include "nnops/ops/conv2d.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/detail/half.hpp"
#include "backend/cpu/conv/conv2d_implicit_gemm.h"   // get_conv2d_implicit_gemm_plan, Conv2DGemmPlan
#include "common/test_harness.hpp"
#include "common/cpu_pool.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"

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
void conv2d_int8_ref(const Conv2DAttributes& attrs,
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
    std::array<int64_t, 4> wshape = {1, 1, 2, 2};  // OC, IC/G, KH, KW
    int64_t kh = 2, kw = 2, sh = 1, sw = 1, ph = 0, pw = 0, dh = 1, dw = 1;
    int64_t groups = 1;
    DataType act = DataType::s8;      // activation storage / dtype
    DataType out = DataType::f32;     // output dtype
    DataType act_q = DataType::s8;    // act_quant_dtype (fp activations only)
    bool bias = false;
    float s_x = 0.0078125f;           // 1/128 — exact in binary
    int32_t zp_x = 0;
    float s_out = 0.03125f;           // requant scale, integer output only
    int32_t zp_out = 0;
    bool asym_w = false;              // per-channel weight zero point != 0
    bool add_to = false;
    EpilogueActivateType ep = EpilogueActivateType::None;
    int64_t in_pitch = 0;             // > IW → pitched input storage
    int64_t out_pitch = 0;            // > OW → pitched output storage
    uint64_t seed = 1;
};

/// Owns a case's buffers and the views over them — views last, so a resize dangles.
struct Fixture {
    Case c;
    std::vector<uint8_t> act;         // activation storage
    std::vector<int8_t>  w;           // weight storage [OC][IC/G][KH][KW]
    std::vector<uint8_t> bias;        // bias storage (empty when unused)
    std::vector<float>   w_s;         // per-channel weight scales
    std::vector<int32_t> w_zp;        // per-channel weight zero points
    QuantParams qa, qw;
    QuantParams out_q;                // the requant's parameters (integer output)
    Conv2DAttributes attrs;
    std::vector<TensorView> ins;

    explicit Fixture(const Case& cs) : c(cs) {
        const int64_t N = c.ishape[0], IC = c.ishape[1];
        const int64_t IH = c.ishape[2], IW = c.ishape[3];
        const int64_t OC = c.wshape[0];
        const int64_t pitch = c.in_pitch ? c.in_pitch : IW;
        const bool is_u8 = (c.act == DataType::u8);
        const bool fp_act = (c.act == DataType::f32 || c.act == DataType::f16);
        const bool quant_target_u8 = (c.act_q == DataType::u8);
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
            (void)quant_target_u8;
        }

        // ---- weight + its per-channel parameters ----
        const int64_t wnumel = OC * c.wshape[1] * c.kh * c.kw;
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
        attrs.groups = c.groups;
        attrs.add_to = c.add_to;
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

/// Deterministic initial output contents (only `add_to` reads them back).
void seed_output(std::vector<uint8_t>& buf, DataType dt, uint64_t seed) {
    test::XorShift128 rng(seed);
    if (dt == DataType::s8 || dt == DataType::u8) {
        // No add_to on an integer destination: the contents are overwritten.
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

/// make_planar, keeping the quant params an integer destination needs.
/// @p pitch_elems is 0 for a dense run, else the padded row stride.
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

/// Storage and the pitched row stride a case's output needs; the tail keeps the
/// seed (nothing may write it).
struct OutPlan { size_t bytes = 0; int64_t pitch = 0; };

OutPlan out_plan(const TensorDesc& od, const Case& c) {
    const int64_t OW = od.dims[od.rank - 1];
    OutPlan p;
    p.pitch = c.out_pitch ? c.out_pitch : OW;
    p.bytes = static_cast<size_t>(od.numel() / OW * p.pitch) *
              elem_size(od.dtype);
    return p;
}

struct Result {
    std::vector<uint8_t> buf;
    TensorDesc desc;
};

Result run_route(const Case& c, const ComputeContext& ctx, bool reference) {
    Fixture f(c);
    std::vector<TensorDesc> ins = f.descs();
    auto op = Conv2D::create(f.attrs, Backend::CPU);
    auto outs = op->getOutputTensorDesc(ins);

    Result r;
    r.desc = outs[0];
    const OutPlan op_layout = out_plan(outs[0], c);
    r.buf.assign(op_layout.bytes, 0);
    seed_output(r.buf, outs[0].dtype, c.seed + 3);
    // make_planar would drop the QuantParams an integer requant needs.
    auto out = make_out_view(outs[0], r.buf.data(), f.out_q, op_layout.pitch);

    if (reference) {
        nnops::backend::cpu::reference::conv2d_int8_ref(f.attrs, out, f.ins, ctx, nullptr);
    } else {
        op->compute(out, f.ins, ctx, nullptr);
    }
    return r;
}

/// The same call with the filter prepacked through the two-step contract.
Result run_route_prepacked(const Case& c, const ComputeContext& ctx) {
    Fixture f(c);
    std::vector<TensorDesc> ins = f.descs();
    auto op = Conv2D::create(f.attrs, Backend::CPU);
    auto outs = op->getOutputTensorDesc(ins);

    TensorView q;
    const TensorView wq[] = {f.ins[1]};
    TensorView oq[] = {q};
    op->prepackWeights(wq, oq);
    q = oq[0];
    NNOPS_EXPECT_TRUE(q.rank() > 0);
    NNOPS_EXPECT_EQ(q.layout(), TensorLayout::PackedOIHW);
    NNOPS_EXPECT_EQ(q.shape(0), c.wshape[0]);

    std::vector<int8_t> blob(q.storage_bytes());
    TensorView packed(q.shape_span(), q.data_type(), blob.data(), q.pitch(),
                      q.layout(), q.quant_params());
    const TensorView wf[] = {f.ins[1]};
    TensorView of[] = {packed};
    op->prepackWeights(wf, of);

    std::vector<TensorView> pins = f.ins;
    pins[1] = packed;

    Result r;
    r.desc = outs[0];
    const OutPlan op_layout = out_plan(outs[0], c);
    r.buf.assign(op_layout.bytes, 0);
    seed_output(r.buf, outs[0].dtype, c.seed + 3);
    auto out = make_out_view(outs[0], r.buf.data(), f.out_q, op_layout.pitch);
    NNOPS_EXPECT_EQ(op->compute(out, pins, ctx, nullptr), Status::ok);
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
        throw std::runtime_error(std::string("conv2d_int8 vs reference (case ") +
                                 std::to_string(idx) + "): " + diff);
    }
}

Case base_case() { return Case{}; }

// ============================================================
// Hand-computed case — pins the whole formula
// ============================================================

NNOPS_TEST(conv2d_i8_hand_exact) {
    // in [1,1,3,3] = 1..9, s8 zp_x 2 s_x 0.25; w [1,1,2,2] = {1,-1,2,0}, s8
    // zp_w 1 s_w 0.5. acc = -1, -3, -7, -9 → × (s_w·s_x) = the four values below.
    const int64_t ishape[] = {1, 1, 3, 3};
    const int64_t wshape[] = {1, 1, 2, 2};
    const int64_t bshape[] = {1};

    std::vector<int8_t> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int8_t> wt = {1, -1, 2, 0};
    std::vector<float>  ws = {0.5f};
    std::vector<int32_t> wz = {1};

    Conv2DAttributes attrs;
    attrs.kernel_size = {2, 2};
    attrs.stride = {1, 1};
    attrs.padding = {0, 0};

    TensorView input(ishape, DataType::s8, in.data(), TensorLayout::NCHW,
                     per_tensor(0.25f, 2));
    TensorView weight(wshape, DataType::s8, wt.data(), TensorLayout::NCHW,
                      per_channel(ws.data(), wz.data(), 1));

    auto op = Conv2D::create(attrs, Backend::CPU);
    const TensorDesc descs[] = {input.desc(), weight.desc()};
    auto outs = op->getOutputTensorDesc(descs);
    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(outs[0].dims[2], 2);
    NNOPS_EXPECT_EQ(outs[0].dims[3], 2);

    std::vector<float> out_buf(4, 0.0f);
    auto out = test::make_planar(outs[0], out_buf.data());
    const TensorView ins[] = {input, weight};
    op->compute(out, ins);

    const float expected[4] = {-0.125f, -0.375f, -0.875f, -1.125f};
    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[static_cast<size_t>(i)], expected[i], 1e-6f);
    }
    (void)bshape;
}

// ============================================================
// Table-driven: every route against the reference, bit for bit
// ============================================================

NNOPS_TEST(conv2d_i8_matches_ref_all_routes) {
    std::vector<Case> cases;

    // ---- pre-quantized s8 activation, symmetric weight (no compensation) ----
    {
        Case c;
        c.ishape = {1, 4, 8, 8};
        c.wshape = {8, 4, 3, 3};
        c.kh = c.kw = 3;
        c.act = DataType::s8;
        c.zp_x = 0;
        cases.push_back(c);
    }
    // ---- s8 activation with a nonzero zero point + zero padding ----
    {
        Case c;
        c.ishape = {2, 3, 7, 7};
        c.wshape = {5, 3, 3, 3};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.act = DataType::s8;
        c.zp_x = -7;
        cases.push_back(c);
    }
    // ---- u8 activation: zp == 128 (the classic symmetric u8 form) ----
    {
        Case c;
        c.ishape = {1, 4, 8, 8};
        c.wshape = {8, 4, 3, 3};
        c.kh = c.kw = 3;
        c.act = DataType::u8;
        c.zp_x = 128;
        cases.push_back(c);
    }
    // ---- u8 activation: asymmetric zero point (XOR + r_b correction) ----
    {
        Case c;
        c.ishape = {1, 4, 9, 9};
        c.wshape = {6, 4, 3, 3};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.act = DataType::u8;
        c.zp_x = 173;
        cases.push_back(c);
    }
    // ---- asymmetric weight zero point → the r_a / r_b sweeps run ----
    {
        Case c;
        c.ishape = {1, 6, 8, 8};
        c.wshape = {7, 6, 3, 3};
        c.kh = c.kw = 3;
        c.act = DataType::s8;
        c.zp_x = 0;
        c.asym_w = true;
        cases.push_back(c);
    }
    // ---- both asymmetric: every compensation term is live ----
    {
        Case c;
        c.ishape = {1, 5, 7, 7};
        c.wshape = {6, 5, 3, 3};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.act = DataType::u8;
        c.zp_x = 130;
        c.asym_w = true;
        cases.push_back(c);
    }
    // ---- fp32 activation, quantized into the compact plane, then gathered ----
    {
        Case c;
        c.ishape = {1, 4, 8, 8};
        c.wshape = {8, 4, 3, 3};
        c.kh = c.kw = 3;
        c.act = DataType::f32;
        c.act_q = DataType::s8;
        cases.push_back(c);
    }
    // ---- fp32 activation quantized to u8 ----
    {
        Case c;
        c.ishape = {1, 4, 8, 8};
        c.wshape = {8, 4, 3, 3};
        c.kh = c.kw = 3;
        c.act = DataType::f32;
        c.act_q = DataType::u8;
        c.zp_x = 128;
        c.asym_w = true;
        cases.push_back(c);
    }
    // ---- fp16 activation with u8 storage, fp16 output ----
    {
        Case c;
        c.ishape = {1, 4, 8, 8};
        c.wshape = {8, 4, 3, 3};
        c.kh = c.kw = 3;
        c.act = DataType::f16;
        c.out = DataType::f16;
        c.act_q = DataType::u8;
        c.zp_x = 117;
        cases.push_back(c);
    }
    // ---- f16 activation with an f32 output_dtype: the output stays f16 ----
    {
        Case c;
        c.ishape = {1, 3, 6, 6};
        c.wshape = {4, 3, 3, 3};
        c.kh = c.kw = 3;
        c.act = DataType::f16;
        c.out = DataType::f32;
        c.act_q = DataType::s8;
        cases.push_back(c);
    }
    // ---- 1x1 pointwise, pre-quantized (plan.pointwise) ----
    {
        Case c;
        c.ishape = {1, 32, 6, 6};
        c.wshape = {16, 32, 1, 1};
        c.kh = c.kw = 1;
        c.act = DataType::u8;
        c.zp_x = 128;
        cases.push_back(c);
    }
    // ---- 1x1 with an fp activation: quantized into the compact plane, so it
    //      still reaches the pointwise route ----
    {
        Case c;
        c.ishape = {1, 16, 5, 5};
        c.wshape = {12, 16, 1, 1};
        c.kh = c.kw = 1;
        c.act = DataType::f32;
        cases.push_back(c);
    }
    // ---- groups = 2, s8 activation ----
    {
        Case c;
        c.ishape = {1, 4, 8, 8};
        c.wshape = {8, 2, 3, 3};
        c.kh = c.kw = 3;
        c.groups = 2;
        c.act = DataType::s8;
        cases.push_back(c);
    }
    // ---- groups = 4, u8 activation ----
    {
        Case c;
        c.ishape = {1, 8, 8, 8};
        c.wshape = {16, 2, 3, 3};
        c.kh = c.kw = 3;
        c.groups = 4;
        c.act = DataType::u8;
        c.zp_x = 120;
        cases.push_back(c);
    }
    // ---- depthwise (groups == IC) ----
    {
        Case c;
        c.ishape = {1, 6, 7, 7};
        c.wshape = {6, 1, 3, 3};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.groups = 6;
        c.act = DataType::s8;
        cases.push_back(c);
    }
    // ---- stride 2 + dilation 2 ----
    {
        Case c;
        c.ishape = {1, 4, 16, 16};
        c.wshape = {8, 4, 3, 3};
        c.kh = c.kw = 3;
        c.sh = c.sw = 2;
        c.dh = c.dw = 2;
        c.ph = c.pw = 1;
        c.act = DataType::s8;
        c.zp_x = 3;
        c.asym_w = true;
        cases.push_back(c);
    }
    // ---- large: 160 OC over two oc-blocks (128 + 32); bias and Relu6 ----
    {
        Case c;
        c.ishape = {1, 16, 20, 20};
        c.wshape = {160, 16, 3, 3};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.act = DataType::u8;
        c.zp_x = 128;
        c.asym_w = true;
        c.bias = true;
        c.ep = EpilogueActivateType::Relu6;
        cases.push_back(c);
    }
    // ---- batch 2 with bias + sigmoid ----
    {
        Case c;
        c.ishape = {2, 8, 9, 9};
        c.wshape = {12, 8, 3, 3};
        c.kh = c.kw = 3;
        c.act = DataType::f32;
        c.bias = true;
        c.ep = EpilogueActivateType::Sigmoid;
        cases.push_back(c);
    }
    // ---- pitched input (row stride > IW) ----
    {
        Case c;
        c.ishape = {1, 4, 8, 7};
        c.wshape = {8, 4, 3, 3};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.in_pitch = 12;
        c.act = DataType::s8;
        c.zp_x = -3;
        cases.push_back(c);
    }
    // ---- pitched input on the pointwise geometry (must leave pointwise) ----
    {
        Case c;
        c.ishape = {1, 16, 4, 6};
        c.wshape = {8, 16, 1, 1};
        c.kh = c.kw = 1;
        c.in_pitch = 10;
        c.act = DataType::u8;
        c.zp_x = 128;
        cases.push_back(c);
    }
    // ---- f16 output with bias ----
    {
        Case c;
        c.ishape = {1, 8, 8, 8};
        c.wshape = {8, 8, 3, 3};
        c.kh = c.kw = 3;
        c.act = DataType::f16;
        c.out = DataType::f16;
        c.bias = true;
        cases.push_back(c);
    }

    for (size_t i = 0; i < cases.size(); ++i) {
        cases[i].seed = 1000 + i;
        expect_matches_ref(cases[i], {}, static_cast<int>(i));
    }
}

// ============================================================
// Epilogues: the clip runs before the activation, matching fp's order
// ============================================================

NNOPS_TEST(conv2d_i8_epilogues_match_ref) {
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
            c.wshape = {8, 6, 3, 3};
            c.kh = c.kw = 3;
            c.ph = c.pw = 1;
            c.act = DataType::s8;
            c.zp_x = -5;
            c.asym_w = true;
            c.bias = with_bias != 0;
            c.ep = ep;
            c.seed = static_cast<uint64_t>(seed++);
            expect_matches_ref(c);
        }
    }
}

// ============================================================
// Integer output (s8 / u8) — int32 bias, requant
// ============================================================

/// Minimal fixed-size pool exposing a CpuBackend (test_conv2d.cpp pattern).
struct SimplePool {
    explicit SimplePool(int nthreads) : nthreads_(nthreads) {
        // OpenMP when the build has it (common/cpu_pool.hpp), so the adapter
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

NNOPS_TEST(conv2d_i8_int_out_hand_exact) {
    // conv2d_i8_hand_exact's operands: acc = -1, -3, -7, -9, s_w·s_x = 0.125,
    // so the requant scale is exactly 0.125 / s_out.
    const int64_t ishape[] = {1, 1, 3, 3};
    const int64_t wshape[] = {1, 1, 2, 2};
    const int64_t bshape[] = {1};

    std::vector<int8_t>  in = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int8_t>  wt = {1, -1, 2, 0};
    std::vector<float>   ws = {0.5f};
    std::vector<int32_t> wz = {1};

    auto run = [&](DataType out_dt, float s_out, int32_t zp_out,
                   const int32_t* bias, EpilogueActivateType ep) {
        Conv2DAttributes attrs;
        attrs.kernel_size = {2, 2};
        attrs.stride = {1, 1};
        attrs.padding = {0, 0};
        attrs.output_dtype = out_dt;
        attrs.epilogue.type = ep;

        TensorView input(ishape, DataType::s8, in.data(), TensorLayout::NCHW,
                         per_tensor(0.25f, 2));
        TensorView weight(wshape, DataType::s8, wt.data(), TensorLayout::NCHW,
                          per_channel(ws.data(), wz.data(), 1));
        TensorView bias_view(bshape, DataType::s32,
                             const_cast<int32_t*>(bias), TensorLayout::NCHW);

        auto op = Conv2D::create(attrs, Backend::CPU);
        const TensorDesc descs[] = {input.desc(), weight.desc(), bias_view.desc()};
        auto outs = op->getOutputTensorDesc(descs);
        NNOPS_EXPECT_EQ(outs[0].dtype, out_dt);

        std::vector<uint8_t> buf(static_cast<size_t>(outs[0].nbytes()), 0);
        auto out = make_out_view(outs[0], buf.data(), per_tensor(s_out, zp_out));
        const TensorView ins[] = {input, weight, bias_view};
        op->compute(out, ins);

        std::vector<int32_t> got(4);
        for (int i = 0; i < 4; ++i) {
            got[static_cast<size_t>(i)] = (out_dt == DataType::u8)
                ? int32_t(buf[static_cast<size_t>(i)])
                : int32_t(static_cast<int8_t>(buf[static_cast<size_t>(i)]));
        }
        return got;
    };

    {
        // bias 5 → 4, 2, -2, -4; req 1 → v + 10.
        const int32_t bias = 5;
        const auto got = run(DataType::s8, 0.125f, 10, &bias, EpilogueActivateType::None);
        const int32_t want[4] = {14, 12, 8, 6};
        for (int i = 0; i < 4; ++i) { NNOPS_EXPECT_EQ(got[static_cast<size_t>(i)], want[i]); }
    }
    {
        // req 64 → 256, 128, -128, -256: both clamps of the s8 range.
        const int32_t bias = 5;
        const auto got = run(DataType::s8, 1.0f / 512.0f, 0, &bias, EpilogueActivateType::None);
        const int32_t want[4] = {127, 127, -128, -128};
        for (int i = 0; i < 4; ++i) { NNOPS_EXPECT_EQ(got[static_cast<size_t>(i)], want[i]); }
    }
    {
        // Same values on u8: 256 saturates at 255, 128 fits, negatives floor at 0.
        const int32_t bias = 5;
        const auto got = run(DataType::u8, 1.0f / 512.0f, 0, &bias, EpilogueActivateType::None);
        const int32_t want[4] = {255, 128, 0, 0};
        for (int i = 0; i < 4; ++i) { NNOPS_EXPECT_EQ(got[static_cast<size_t>(i)], want[i]); }
    }
    {
        // Relu runs on the int32 value, before the requant: all-negative → 0.
        const int32_t bias = 0;
        const auto got = run(DataType::u8, 0.125f, 0, &bias, EpilogueActivateType::Relu);
        const int32_t want[4] = {0, 0, 0, 0};
        for (int i = 0; i < 4; ++i) { NNOPS_EXPECT_EQ(got[static_cast<size_t>(i)], want[i]); }
    }
}

/// Integer destinations over the same geometry spread as the fp route.
NNOPS_TEST(conv2d_i8_int_out_matches_ref) {
    std::vector<Case> cases;

    // ---- s8 out / s8 act, asymmetric weight, int32 bias ----
    {
        Case c;
        c.ishape = {1, 4, 8, 8};
        c.wshape = {8, 4, 3, 3};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.act = DataType::s8;
        c.zp_x = -9;
        c.out = DataType::s8;
        c.asym_w = true;
        c.bias = true;
        c.s_out = 0.03125f;
        c.zp_out = -7;
        cases.push_back(c);
    }
    // ---- u8 out / u8 act with the classic zp 128 ----
    {
        Case c;
        c.ishape = {1, 4, 8, 8};
        c.wshape = {8, 4, 3, 3};
        c.kh = c.kw = 3;
        c.act = DataType::u8;
        c.zp_x = 128;
        c.out = DataType::u8;
        c.bias = true;
        c.zp_out = 128;
        cases.push_back(c);
    }
    // ---- u8 out, no bias: the compensate-only epilogue ----
    {
        Case c;
        c.ishape = {2, 5, 7, 7};
        c.wshape = {6, 5, 3, 3};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.act = DataType::u8;
        c.zp_x = 173;
        c.out = DataType::u8;
        c.asym_w = true;
        c.zp_out = 3;
        cases.push_back(c);
    }
    // ---- asymmetric activation, symmetric weight: the compensation folds into
    //      the bias, and zp_eff = 173 − 128 = 45 keeps that fold non-identity ----
    {
        Case c;
        c.ishape = {1, 4, 9, 9};
        c.wshape = {8, 4, 3, 3};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.act = DataType::u8;
        c.zp_x = 173;
        c.out = DataType::s8;
        c.bias = true;
        c.zp_out = -7;
        cases.push_back(c);
    }
    // ---- the same fold with no bias at all: the vector is the correction only ----
    {
        Case c;
        c.ishape = {2, 3, 8, 8};
        c.wshape = {5, 3, 3, 3};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.act = DataType::s8;
        c.zp_x = -11;                     // zp_eff = −11
        c.out = DataType::u8;
        c.zp_out = 128;
        cases.push_back(c);
    }
    // ---- Relu before the requant ----
    {
        Case c;
        c.ishape = {1, 6, 7, 7};
        c.wshape = {8, 6, 3, 3};
        c.kh = c.kw = 3;
        c.act = DataType::s8;
        c.out = DataType::s8;
        c.ep = EpilogueActivateType::Relu;
        c.bias = true;
        c.s_out = 0.0625f;
        cases.push_back(c);
    }
    // ---- fp32 activation quantized to s8, s8 out ----
    {
        Case c;
        c.ishape = {1, 4, 8, 8};
        c.wshape = {8, 4, 3, 3};
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
        c.wshape = {8, 4, 3, 3};
        c.kh = c.kw = 3;
        c.act = DataType::f16;
        c.act_q = DataType::u8;
        c.zp_x = 117;
        c.out = DataType::u8;
        c.zp_out = 120;
        cases.push_back(c);
    }
    // ---- groups = 2 (bias indexed per output channel) ----
    {
        Case c;
        c.ishape = {1, 4, 8, 8};
        c.wshape = {8, 2, 3, 3};
        c.kh = c.kw = 3;
        c.groups = 2;
        c.act = DataType::s8;
        c.out = DataType::s8;
        c.bias = true;
        c.asym_w = true;
        cases.push_back(c);
    }
    // ---- 1x1 pointwise, u8 out ----
    {
        Case c;
        c.ishape = {1, 32, 6, 6};
        c.wshape = {16, 32, 1, 1};
        c.kh = c.kw = 1;
        c.act = DataType::u8;
        c.zp_x = 128;
        c.out = DataType::u8;
        c.zp_out = 128;
        c.bias = true;
        cases.push_back(c);
    }
    // ---- 160 OC over two oc-blocks, saturation-heavy scale ----
    {
        Case c;
        c.ishape = {1, 16, 12, 12};
        c.wshape = {160, 16, 3, 3};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.act = DataType::u8;
        c.zp_x = 128;
        c.out = DataType::s8;
        c.asym_w = true;
        c.bias = true;
        c.s_out = 0.001953125f;   // req ×4 — most outputs clamp
        cases.push_back(c);
    }
    // ---- pitched input, integer output ----
    {
        Case c;
        c.ishape = {1, 4, 8, 7};
        c.wshape = {8, 4, 3, 3};
        c.kh = c.kw = 3;
        c.ph = c.pw = 1;
        c.in_pitch = 12;
        c.act = DataType::s8;
        c.out = DataType::s8;
        c.bias = true;
        cases.push_back(c);
    }

    for (size_t i = 0; i < cases.size(); ++i) {
        cases[i].seed = 5000 + i;
        expect_matches_ref(cases[i], {}, static_cast<int>(i));
    }

    // Threading: an oc-block's epilogue must stay within its own task.
    for (int nthreads : {2, 4}) {
        SimplePool pool(nthreads);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        for (size_t i = 0; i < cases.size(); ++i) {
            expect_matches_ref(cases[i], ctx, static_cast<int>(i));
        }
    }
}

// ============================================================
// Threading: identical bits across thread counts
// ============================================================

NNOPS_TEST(conv2d_i8_threaded_matches_serial) {
    struct TCase { std::array<int64_t, 4> ishape; std::array<int64_t, 4> wshape;
                   int64_t kh, kw, sh, sw, ph, pw, groups; DataType act; bool fp_act; };
    const TCase cases[] = {
        // tall plane: the oh-chunk split kicks in (1 batch, 1 group)
        {{1, 8, 32, 32}, {16, 8, 3, 3},  3, 3, 1, 1, 1, 1, 1, DataType::u8,  false},
        // fp activation: quantized once into the shared plane, then gathered
        {{1, 8, 24, 24}, {16, 8, 3, 3},  3, 3, 1, 1, 1, 1, 1, DataType::f32, true},
        // batch + groups: tasks spread over every dimension
        {{2, 4, 20, 20}, {8, 2, 3, 3},   3, 3, 1, 1, 1, 1, 2, DataType::s8,  false},
        // pointwise with u8: per-chunk B base off the input plane
        {{1, 64, 16, 16}, {96, 64, 1, 1}, 1, 1, 1, 1, 0, 0, 1, DataType::u8, false},
        // stride 2 + asym weight zero point (both reductions live)
        {{1, 8, 28, 28}, {16, 8, 3, 3},  3, 3, 2, 2, 1, 1, 1, DataType::u8,  false},
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
        if (tc.fp_act) { c.act_q = DataType::s8; }
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
                throw std::runtime_error("conv2d_i8_threaded (" +
                                         std::to_string(nthreads) + " threads): " + diff);
            }
        }
    }
}

// The fp-activation scratch layout at a size (200704 B of quantized plane)
// where a bad offset is not absorbed by the arena's tail — see the header.
NNOPS_TEST(conv2d_i8_fp_act_large_layout) {
    for (DataType target : {DataType::s8, DataType::u8}) {
        Case c;
        c.ishape = {1, 64, 56, 56};
        c.wshape = {256, 64, 1, 1};     // 2 oc-blocks: the acc region is reused
        c.kh = c.kw = 1;
        c.act = DataType::f32;          // quantized into the plane, then pointwise
        c.act_q = target;
        c.bias = true;
        c.asym_w = true;                // r_a/r_b live, so the epilogue reads both
        c.zp_x = (target == DataType::u8) ? 140 : -6;
        c.seed = 7000 + static_cast<uint64_t>(target == DataType::u8);

        expect_matches_ref(c, {}, 0);

        SimplePool pool(4);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        expect_matches_ref(c, ctx, 1);
    }
}

// ============================================================
// Output descriptor + plan
// ============================================================

NNOPS_TEST(conv2d_i8_output_desc) {
    auto desc_of = [](const Case& c) {
        Fixture f(c);
        std::vector<TensorDesc> ins = f.descs();
        auto op = Conv2D::create(f.attrs, Backend::CPU);
        return op->getOutputTensorDesc(ins)[0];
    };

    // fp activation → the output follows the activation dtype.
    Case c;
    c.ishape = {1, 4, 6, 6};
    c.wshape = {8, 4, 3, 3};
    c.kh = c.kw = 3;
    c.act = DataType::f16;
    NNOPS_EXPECT_EQ(desc_of(c).dtype, DataType::f16);
    c.act = DataType::f32;
    NNOPS_EXPECT_EQ(desc_of(c).dtype, DataType::f32);

    // An integer output comes from attributes.output_dtype whatever the
    // activation storage is.
    c.out = DataType::s8;
    NNOPS_EXPECT_EQ(desc_of(c).dtype, DataType::s8);
    c.out = DataType::u8;
    NNOPS_EXPECT_EQ(desc_of(c).dtype, DataType::u8);
    c.act = DataType::f16;
    NNOPS_EXPECT_EQ(desc_of(c).dtype, DataType::u8);
    c.act = DataType::s8;
    NNOPS_EXPECT_EQ(desc_of(c).dtype, DataType::u8);
    c.out = DataType::f32;

    // Pre-quantized activation → attributes.output_dtype.
    c.act = DataType::u8;
    c.out = DataType::f32;
    NNOPS_EXPECT_EQ(desc_of(c).dtype, DataType::f32);
    c.out = DataType::f16;
    NNOPS_EXPECT_EQ(desc_of(c).dtype, DataType::f16);
    c.act = DataType::s8;
    NNOPS_EXPECT_EQ(desc_of(c).dtype, DataType::f16);

    // An f32 weight takes the fp route, which ignores output_dtype.
    Case fp;
    fp.ishape = {1, 4, 6, 6};
    fp.wshape = {8, 4, 3, 3};
    fp.kh = fp.kw = 3;
    fp.act = DataType::f32;
    fp.out = DataType::f16;   // ignored with an f32 *weight*
    {
        Fixture f(fp);
        // Swap the weight dtype to f32 so the fp route handles it.
        std::vector<float> wf(f.w.size());
        for (size_t i = 0; i < wf.size(); ++i) { wf[i] = static_cast<float>(f.w[i]) * 0.01f; }
        const std::span<const int64_t> wshape(fp.wshape.data(), 4);
        TensorView w32(wshape, DataType::f32, wf.data());
        const TensorDesc descs[] = {f.ins[0].desc(), w32.desc()};
        auto op = Conv2D::create(f.attrs, Backend::CPU);
        NNOPS_EXPECT_EQ(op->getOutputTensorDesc(descs)[0].dtype, DataType::f32);
    }
}

NNOPS_TEST(conv2d_i8_pitched_output) {
    // A destination whose rows are 32-byte aligned has a row pitch wider than
    // OW. The planar GEMM's N axis is an oh-block's rows flattened into one
    // contiguous run, so the plan pins oh_block to one row for a pitched
    // destination; the values must stay bit-identical to the reference, which
    // honors the pitch, and the row tail must keep its seed.
    Case c = base_case();
    c.ishape = {1, 8, 9, 7};          // OH 9, OW 7
    c.wshape = {12, 4, 3, 3};         // groups 2, icg 4 / ocg 6
    c.kh = c.kw = 3;
    c.ph = c.pw = 1;
    c.groups = 2;
    c.bias = true;

    // The geometry gives the dense plan a multi-row block; the pitch is what
    // pins it. (The dense assertion is the control: it must not already be 1.)
    {
        Fixture f(c);
        std::vector<TensorDesc> ins = f.descs();
        auto op = Conv2D::create(f.attrs, Backend::CPU);
        auto outs = op->getOutputTensorDesc(ins);
        const int64_t OW = outs[0].dims[outs[0].rank - 1];
        const int64_t pitch = 32 / static_cast<int64_t>(elem_size(outs[0].dtype));
        std::vector<uint8_t> pad(static_cast<size_t>(outs[0].numel() / OW * pitch) *
                                 elem_size(outs[0].dtype));
        const std::span<const int64_t> oshape(outs[0].dims.data(),
                                              static_cast<size_t>(outs[0].rank));
        TensorView pv(oshape, outs[0].dtype, pad.data(),
                      pitch * static_cast<int64_t>(elem_size(outs[0].dtype)));

        const TensorDesc dense[]   = {outs[0]};
        const TensorDesc pitched[] = {pv.desc()};
        const auto pd = backend::cpu::get_conv2d_implicit_gemm_plan(f.attrs, ins, dense, 1, false);
        const auto pp = backend::cpu::get_conv2d_implicit_gemm_plan(f.attrs, ins, pitched, 1, false);
        NNOPS_EXPECT_TRUE(pd.oh_block > 1);
        NNOPS_EXPECT_EQ(pp.oh_block, 1);
        NNOPS_EXPECT_EQ(pp.num_oh_blocks, outs[0].dims[2]);
    }

    // s8 activation, f32 destination, bias + Relu: 7 f32 rows → 32 B.
    c.out_pitch = 8;
    c.ep = EpilogueActivateType::Relu;
    expect_matches_ref(c);

    // u8 activation requantized to an integer destination: 7 bytes → 32 B.
    c.act = DataType::u8;
    c.zp_x = 128;
    c.out = DataType::s8;
    c.s_out = 0.05f;
    c.out_pitch = 32;
    c.ep = EpilogueActivateType::None;
    expect_matches_ref(c);

    // f16 destination, and an fp activation materialized into a compact plane.
    c.act = DataType::f16;
    c.zp_x = 0;
    c.out = DataType::f16;
    c.out_pitch = 16;
    expect_matches_ref(c);

    // The same contract through the prepacked-filter path.
    const Result dense = run_route(c, {}, /*reference=*/true);
    const Result prepacked = run_route_prepacked(c, {});
    const std::string diff = first_difference(prepacked, dense);
    if (!diff.empty()) {
        throw std::runtime_error("conv2d_int8 prepacked, pitched output: " + diff);
    }
}

NNOPS_TEST(conv2d_i8_plan) {
    auto plan_of = [](const Case& c) {
        Fixture f(c);
        std::vector<TensorDesc> ins = f.descs();
        auto op = Conv2D::create(f.attrs, Backend::CPU);
        auto outs = op->getOutputTensorDesc(ins);
        const TensorDesc o[] = {outs[0]};
        return backend::cpu::get_conv2d_implicit_gemm_plan(f.attrs, ins, o, 1, false);
    };

    // Pointwise: a dense 1x1 reads the plane directly — no im2col buffer, and K
    // spans the whole group.
    Case pw;
    pw.ishape = {1, 32, 5, 5};
    pw.wshape = {16, 32, 1, 1};
    pw.kh = pw.kw = 1;
    pw.act = DataType::s8;
    {
        const auto p = plan_of(pw);
        NNOPS_EXPECT_TRUE(p.pointwise);
        NNOPS_EXPECT_EQ(p.col_size, 0);
        NNOPS_EXPECT_EQ(p.icn_block, 32);
        NNOPS_EXPECT_TRUE(p.acc_size > 0);
        NNOPS_EXPECT_TRUE(p.pack_size > 0);
        NNOPS_EXPECT_TRUE(p.pack_a_size > 0);
        const auto sc = backend::cpu::gemm_scratch_layout(p, 1);
        NNOPS_EXPECT_EQ(sc.total, static_cast<int64_t>(p.workspace_size));
        NNOPS_EXPECT_TRUE(sc.off_acc + p.num_slots * p.acc_size * 4 <=
                          static_cast<int64_t>(p.workspace_size));
    }

    // An fp activation is materialized as a compact int8 plane first, so the
    // same geometry stays pointwise and the plan carries that plane's size.
    pw.act = DataType::f32;
    {
        const auto p = plan_of(pw);
        NNOPS_EXPECT_TRUE(p.pointwise);
        NNOPS_EXPECT_EQ(p.col_size, 0);
        NNOPS_EXPECT_EQ(p.quant_size, 1 * 32 * 5 * 5);
        NNOPS_EXPECT_TRUE(p.workspace_size >= static_cast<size_t>(p.quant_size));
        // The layout contract: plane at the arena's front, slots past it, and
        // acc / packed-A as absolute offsets that already account for both.
        const auto sc = backend::cpu::gemm_scratch_layout(p, 1);
        NNOPS_EXPECT_EQ(sc.off_acc, sc.head + p.num_slots * sc.slot_stride);
        NNOPS_EXPECT_TRUE(sc.head >= p.quant_size);
        NNOPS_EXPECT_EQ(sc.off_pack_a + p.num_slots * p.pack_a_size,
                        static_cast<int64_t>(p.workspace_size));
    }

    // A pitched input leaves pointwise (im2col walks the pitch) — unless the
    // activation is fp, whose materialized plane is compact.
    pw.act = DataType::s8;
    pw.in_pitch = 8;
    {
        const auto p = plan_of(pw);
        NNOPS_EXPECT_FALSE(p.pointwise);
        NNOPS_EXPECT_EQ(p.quant_size, 0);
    }
    pw.act = DataType::f32;
    {
        const auto p = plan_of(pw);
        NNOPS_EXPECT_TRUE(p.pointwise);
        NNOPS_EXPECT_EQ(p.quant_size, 1 * 32 * 5 * 5);
    }

    // One fp plane for the whole call, however many slots and tasks there are:
    // a per-task plane would have to duplicate the oh-chunk halos and hold more.
    Case batched;
    batched.ishape = {4, 16, 20, 20};
    batched.wshape = {32, 16, 3, 3};
    batched.kh = batched.kw = 3;
    batched.act = DataType::f32;
    {
        Fixture f(batched);
        std::vector<TensorDesc> ins = f.descs();
        auto op = Conv2D::create(f.attrs, Backend::CPU);
        auto outs = op->getOutputTensorDesc(ins);
        const TensorDesc o[] = {outs[0]};
        const int64_t whole = 4 * 16 * 20 * 20;
        for (bool slots : {false, true}) {
            for (int nt : {1, 4}) {
                const auto p = backend::cpu::get_conv2d_implicit_gemm_plan(f.attrs, ins, o, nt, slots);
                NNOPS_EXPECT_EQ(p.quant_size, whole);
            }
        }
    }

    // add_to is reference-only: a zeroed plan.
    Case at;
    at.ishape = {1, 4, 7, 7};
    at.wshape = {6, 4, 3, 3};
    at.kh = at.kw = 3;
    at.act = DataType::u8;
    at.zp_x = 128;
    at.add_to = true;
    NNOPS_EXPECT_EQ(plan_of(at).icn_block, 0);

    // An unsupported output dtype is refused too (s32 has no fp epilogue).
    Case bad;
    bad.ishape = {1, 4, 7, 7};
    bad.wshape = {6, 4, 3, 3};
    bad.kh = bad.kw = 3;
    bad.act = DataType::u8;
    bad.out = DataType::s32;
    NNOPS_EXPECT_EQ(plan_of(bad).icn_block, 0);

    // Multi-slot sizing: with thread slots the scratch scales with the pool.
    Case big;
    big.ishape = {1, 16, 20, 20};
    big.wshape = {64, 16, 3, 3};
    big.kh = big.kw = 3;
    big.act = DataType::u8;
    big.zp_x = 128;
    {
        Fixture f(big);
        std::vector<TensorDesc> ins = f.descs();
        auto op = Conv2D::create(f.attrs, Backend::CPU);
        auto outs = op->getOutputTensorDesc(ins);
        const TensorDesc o[] = {outs[0]};
        const auto p1 = backend::cpu::get_conv2d_implicit_gemm_plan(f.attrs, ins, o, 1, false);
        const auto p4 = backend::cpu::get_conv2d_implicit_gemm_plan(f.attrs, ins, o, 4, true);
        NNOPS_EXPECT_TRUE(p4.workspace_size >= p1.workspace_size);
        NNOPS_EXPECT_TRUE(p4.num_slots <= 4);
    }
}

// ============================================================
// End to end: the quantized route against the fp route
// ============================================================

NNOPS_TEST(conv2d_i8_vs_fp_path) {
    // int8 route on stored integers vs the f32 route on the exact dequantized
    // values, against the quantization-error bound
    //     K · ( (s_w/2)·max|a| + (max|w| + s_w/2)·(s_x/2) ).
    const int64_t ishape[] = {1, 8, 12, 12};
    const int64_t wshape[] = {16, 8, 3, 3};
    const int64_t K = 8 * 3 * 3;
    const int64_t OC = 16;
    const int64_t IW = 12;

    std::vector<float> a(static_cast<size_t>(1 * 8 * 12 * 12));
    std::vector<float> wf(static_cast<size_t>(OC * K));
    test::XorShift128 rng(777);
    rng.fill_float(a.data(), static_cast<int64_t>(a.size()), -1.0f, 1.0f);
    rng.fill_float(wf.data(), static_cast<int64_t>(wf.size()), -1.0f, 1.0f);

    const float s_x = 0.0078125f;   // 1/128
    const int32_t zp_x = 128;       // u8 storage
    const int64_t act_numel = static_cast<int64_t>(a.size());
    std::vector<uint8_t> aq(static_cast<size_t>(act_numel));
    std::vector<float>   adq(static_cast<size_t>(act_numel));
    float max_abs_a = 0.0f;
    for (int64_t i = 0; i < act_numel; ++i) {
        const int32_t q = quant_int(a[i], 1.0f / s_x, zp_x, true);
        aq[static_cast<size_t>(i)] = static_cast<uint8_t>(q);
        adq[static_cast<size_t>(i)] = static_cast<float>(q - zp_x) * s_x;
        max_abs_a = std::max(max_abs_a, std::fabs(a[i]));
    }

    std::vector<int8_t> wq(static_cast<size_t>(OC * K));
    std::vector<float>  wdq(static_cast<size_t>(OC * K));
    std::vector<float>  w_s(static_cast<size_t>(OC));
    std::vector<int32_t> w_zp(static_cast<size_t>(OC), 0);
    float max_abs_w = 0.0f;
    for (int64_t oc = 0; oc < OC; ++oc) {
        w_s[static_cast<size_t>(oc)] = 0.01f * static_cast<float>(1 + oc % 4);
    }
    for (int64_t oc = 0; oc < OC; ++oc) {
        const float s = w_s[static_cast<size_t>(oc)];
        for (int64_t k = 0; k < K; ++k) {
            const int64_t i = oc * K + k;
            const int32_t q = quant_int(wf[static_cast<size_t>(i)], 1.0f / s, 0, false);
            wq[static_cast<size_t>(i)] = static_cast<int8_t>(q);
            wdq[static_cast<size_t>(i)] = static_cast<float>(q) * s;
            max_abs_w = std::max(max_abs_w, std::fabs(wf[static_cast<size_t>(i)]));
        }
    }

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride = {1, 1};
    attrs.padding = {1, 1};

    // int8 route.
    TensorView in_q(ishape, DataType::u8, aq.data(), TensorLayout::NCHW,
                    per_tensor(s_x, zp_x));
    TensorView w_q(wshape, DataType::s8, wq.data(), TensorLayout::NCHW,
                   per_channel(w_s.data(), w_zp.data(), OC));
    auto op_q = Conv2D::create(attrs, Backend::CPU);
    const TensorDesc qdescs[] = {in_q.desc(), w_q.desc()};
    auto qouts = op_q->getOutputTensorDesc(qdescs);
    std::vector<float> qbuf(static_cast<size_t>(qouts[0].numel()), 0.0f);
    auto qout = test::make_planar(qouts[0], qbuf.data());
    const TensorView qins[] = {in_q, w_q};
    op_q->compute(qout, qins);

    // fp route on the exactly-dequantized operands.
    TensorView in_f(ishape, DataType::f32, adq.data());
    TensorView w_f(wshape, DataType::f32, wdq.data());
    std::vector<float> fbuf(static_cast<size_t>(qouts[0].numel()), 0.0f);
    auto fout = test::make_planar(qouts[0], fbuf.data());
    const TensorView fins[] = {in_f, w_f};
    auto op_f = Conv2D::create(attrs, Backend::CPU);
    op_f->compute(fout, fins);

    float worst = 0.0f;
    for (size_t i = 0; i < qbuf.size(); ++i) {
        worst = std::max(worst, std::fabs(qbuf[i] - fbuf[i]));
    }
    const float tol = static_cast<float>(K) *
                      (0.5f * 0.01f * 4.0f * max_abs_a +
                       (max_abs_w + 0.5f * 0.01f * 4.0f) * (0.5f * s_x));
    std::printf("      conv2d_i8_vs_fp: max deviation %.6f (bound %.6f)\n",
                double(worst), double(tol));
    NNOPS_EXPECT_TRUE(worst <= tol);
    (void)IW;
}

}  // namespace

// ============================================================
// The prepacked filter — Conv2D::prepackWeights on the planar int8 route
// ============================================================

NNOPS_TEST(conv2d_i8_prepacked_filter) {
    std::vector<Case> cases;
    auto add = [&](Case c) { cases.push_back(c); };

    // nothing to compensate: the sums are never read
    { Case c; c.ishape = {1, 4, 8, 8}; c.wshape = {8, 4, 3, 3}; c.kh = c.kw = 3;
      add(c); }
    // u8 whose zero point does not cancel the shift (the column term)
    { Case c; c.ishape = {1, 8, 6, 6}; c.wshape = {4, 8, 3, 3}; c.kh = c.kw = 3;
      c.act = DataType::u8; c.zp_x = 120; add(c); }
    // asymmetric weight zero point: the per-pixel sums are live
    { Case c; c.ishape = {1, 6, 8, 8}; c.wshape = {7, 6, 3, 3}; c.kh = c.kw = 3;
      c.asym_w = true; add(c); }
    // both terms, a bias and a nonlinear epilogue
    { Case c; c.ishape = {1, 5, 7, 7}; c.wshape = {6, 5, 3, 3}; c.kh = c.kw = 3;
      c.ph = c.pw = 1; c.act = DataType::u8; c.zp_x = 130; c.asym_w = true;
      c.bias = true; c.ep = EpilogueActivateType::Relu6; add(c); }
    // groups 4: the per-group panel sets and the sums table both have to move
    { Case c; c.ishape = {1, 8, 8, 8}; c.wshape = {16, 2, 3, 3}; c.kh = c.kw = 3;
      c.groups = 4; c.act = DataType::u8; c.zp_x = 120; add(c); }
    // several oc blocks (160 = 128 + 32) with an ic tail (40 = 2 · 16 + 8) —
    // the partial sums span three ic blocks
    { Case c; c.ishape = {1, 40, 12, 12}; c.wshape = {160, 40, 3, 3};
      c.kh = c.kw = 3; c.act = DataType::u8; c.zp_x = 128; c.asym_w = true;
      c.bias = true; add(c); }
    // 1x1 pointwise: K is the whole channel count in one pass
    { Case c; c.ishape = {1, 32, 6, 6}; c.wshape = {16, 32, 1, 1}; c.kh = c.kw = 1;
      add(c); }
    // an integer destination: the requant path reads the sums too
    { Case c; c.ishape = {1, 8, 8, 8}; c.wshape = {12, 8, 3, 3}; c.kh = c.kw = 3;
      c.out = DataType::s8; c.asym_w = true; c.bias = true; add(c); }
    // a 7x7 at the narrow panel rung
    { Case c; c.ishape = {1, 4, 14, 14}; c.wshape = {8, 4, 7, 7}; c.kh = c.kw = 7;
      c.ph = c.pw = 3; c.asym_w = true; add(c); }
    // a pitched activation: the im2col walk has to skip the row tail either way
    { Case c; c.ishape = {1, 4, 9, 9}; c.wshape = {8, 4, 3, 3}; c.kh = c.kw = 3;
      c.in_pitch = 12; c.act = DataType::u8; c.zp_x = 128; add(c); }
    // an fp activation, quantized into the plane
    { Case c; c.ishape = {1, 4, 8, 8}; c.wshape = {8, 4, 3, 3}; c.kh = c.kw = 3;
      c.act = DataType::f32; c.act_q = DataType::u8; c.zp_x = 128; add(c); }

    for (size_t i = 0; i < cases.size(); ++i) {
        cases[i].seed = 4000 + i;
        const Result raw = run_route(cases[i], {}, /*reference=*/false);
        const Result packed = run_route_prepacked(cases[i], {});
        const std::string diff = first_difference(raw, packed);
        if (!diff.empty()) {
            throw std::runtime_error(
                "conv2d_int8 prepacked filter (case " + std::to_string(i) +
                ") differs from the raw filter: " + diff);
        }
    }
}
