/// @file test_cuda_conv2d.cpp
/// @brief CUDA Conv2D against the CPU backend.
///
/// Each case builds one activation/filter pair, runs the operator on both
/// backends over the same bytes, and compares. The CPU route is the oracle, so
/// the plane sizes, the padding ring, the bias placement and the add_to
/// ordering are all pinned by the operator's own contract.
///
/// Geometry coverage follows what the implicit GEMM branches on: kernel size
/// 1x1/3x3 and a non-square 3x2, stride 1 and 2, dilation 2, groups 1/2 and a
/// depthwise-shaped group count, and plane sizes that leave a partial pixel
/// tile (a 7x7 output is 49 pixels against a 64-pixel tile).
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/conv2d.hpp"
#include "nnops/detail/half.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/cuda_test_helper.hpp"
#include "detail/bench_knobs.hpp"
#include "common/channel_last.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

struct ConvCase {
    int64_t n = 1, ic = 8, ih = 14, iw = 14;
    int64_t oc = 16;
    int64_t kh = 3, kw = 3;
    int64_t sh = 1, sw = 1;
    int64_t dh = 1, dw = 1;
    int64_t ph = 1, pw = 1;
    int64_t groups = 1;
    DataType dtype = DataType::f32;
    bool use_bias = false;
    bool add_to = false;
    Epilogue epilogue{};
};

struct Outputs {
    HostTensor tensor;
    int64_t oh = 0, ow = 0;
};

/// Run one case on both backends and compare. Returns the reference result so a
/// caller can cross-check two routes against each other.
Outputs run_case(const ConvCase& c, const char* what)
{
    Conv2DAttributes attrs;
    attrs.kernel_size = {c.kh, c.kw};
    attrs.stride = {c.sh, c.sw};
    attrs.dilation = {c.dh, c.dw};
    attrs.padding = {c.ph, c.pw};
    attrs.groups = c.groups;
    attrs.epilogue = c.epilogue;
    attrs.add_to = c.add_to;

    const TensorLayout act_layout = TensorLayout::NHWC;
    const TensorLayout wgt_layout = TensorLayout::HWIO;

    const int64_t icg = c.ic / c.groups;
    const int64_t ocg = c.oc / c.groups;

    // NHWC dims are the storage order: [N, H, W, C].
    HostTensor act = make_host_random(std::vector<int64_t>{c.n, c.ih, c.iw, c.ic}, c.dtype,
                                      false, 23, -0.5f, 0.5f, act_layout);
    HostTensor wgt = make_host_random(
        std::vector<int64_t>{c.kh, c.kw, icg, c.oc},
        c.dtype, false, 29, -0.25f, 0.25f, wgt_layout);
    HostTensor bias;
    if (c.use_bias) {
        bias = make_host_random(std::vector<int64_t>{c.oc}, c.dtype, false, 31, -0.25f, 0.25f);
    }
    (void)ocg;

    std::vector<TensorDesc> descs = {act.view.desc(), wgt.view.desc()};
    if (c.use_bias) { descs.push_back(bias.view.desc()); }

    auto op_cpu = Conv2D::create(attrs, Backend::CPU);
    const std::vector<TensorDesc> outs = op_cpu->getOutputTensorDesc(descs);
    NNOPS_EXPECT_TRUE(!outs.empty());
    const TensorDesc od = outs.front();

    Outputs res;
    // NHWC output dims are [N, H, W, C], so H and W sit above the channel axis.
    res.oh = od.dims[static_cast<size_t>(od.rank - 3)];
    res.ow = od.dims[static_cast<size_t>(od.rank - 2)];

    const auto seed = [&](uint64_t s) {
        return c.add_to ? make_host_random(od.dims, od.dtype, false, s, 0.0f, 0.5f,
                                           act_layout)
                        : make_host_zeroed(od.dims, od.dtype, false, act_layout);
    };
    HostTensor out_ref = seed(37);
    HostTensor out_dev = seed(37);

    std::vector<TensorView> cpu_ins = {act.view, wgt.view};
    if (c.use_bias) { cpu_ins.push_back(bias.view); }
    cpu_run(*op_cpu, out_ref.view, cpu_ins);

    auto op_dev = Conv2D::create(attrs, Backend::CUDA);
    DeviceTensor dact = to_device(act);
    DeviceTensor dwgt = to_device(wgt);
    DeviceTensor dbias;
    std::vector<TensorView> dev_ins = {dact.view, dwgt.view};
    if (c.use_bias) {
        dbias = to_device(bias);
        dev_ins.push_back(dbias.view);
    }
    DeviceTensor dout = alloc_like(out_dev.view, out_dev);
    cuda_run(*op_dev, dout.view, dev_ins);
    from_device(dout, out_dev);

    const float tol_abs = c.dtype == DataType::f16 ? kF16AccumTol : 1e-3f;
    const float tol_rel = c.dtype == DataType::f16 ? 2e-2f : 1e-4f;
    expect_same(out_dev, out_ref, tol_abs, tol_rel, what);

    res.tensor = std::move(out_ref);
    return res;
}

}  // namespace

NNOPS_TEST(cuda_conv2d_geometry) {
    {
        ConvCase c;
        c.ic = 8; c.oc = 16;         run_case(c, "cuda_conv2d_geometry");
    }
    {
        ConvCase c;   // stride 2
        c.ic = 8; c.oc = 16; c.sh = c.sw = 2;
        c.ph = c.pw = 0;
        run_case(c, "cuda_conv2d_geometry");
    }
    {
        ConvCase c;   // 1x1
        c.ic = 16; c.oc = 32; c.kh = c.kw = 1;
        c.ph = c.pw = 0;
        run_case(c, "cuda_conv2d_geometry");
    }
    {
        ConvCase c;   // non-square filter, dilation 2
        c.ic = 6; c.oc = 12; c.kh = 3; c.kw = 2;
        c.dh = c.dw = 2; c.ph = 2; c.pw = 0;
        run_case(c, "cuda_conv2d_geometry");
    }
    {
        ConvCase c;   // 7x7 plane, several batches
        c.n = 2; c.ic = 16; c.oc = 24;         c.ih = c.iw = 7;
        run_case(c, "cuda_conv2d_geometry");
    }
}

NNOPS_TEST(cuda_conv2d_groups) {
    ConvCase c;
    c.ic = 8; c.oc = 16; c.groups = 2;
    run_case(c, "cuda_conv2d_groups");

    ConvCase d;
    d.ic = 8; d.oc = 8; d.groups = 8;
    run_case(d, "cuda_conv2d_groups");
}

NNOPS_TEST(cuda_conv2d_f16_add_to) {
    Epilogue ep;
    ep.type = EpilogueActivateType::Relu;

    ConvCase c;
    c.ic = 8; c.oc = 16; c.dtype = DataType::f16;
    c.use_bias = true; c.add_to = true; c.epilogue = ep;
    run_case(c, "cuda_conv2d_f16_add_to");
}

NNOPS_TEST(cuda_conv2d_wide_channels) {
    // More than 64 output channels and more than 64 pixels: this is the shape
    // that exercises the second piece of a CTA tile wider than the 64-wide MMA
    // tile, which every smaller case here masks out completely.
    ConvCase c;
    c.ic = 64; c.oc = 128; c.ih = c.iw = 14; c.dtype = DataType::f16;
        c.use_bias = true;
    run_case(c, "cuda_conv2d_wide_channels");
}

NNOPS_TEST(cuda_conv2d_wide_planes) {
    // Planes of thousands of pixels rather than tens, so the grid is full and
    // the three things a small case masks all run at scale: the M edge and the
    // partial last tile, the split-K range, and the epilogue's per-row pixel
    // decode. 7x7 and 14x14 sit under one wave of the 64-pixel tile.
    struct Wide {
        int64_t n, ic, oc, side, kh, sh, groups;
        DataType dtype;
    };
    const Wide cases[] = {
        {1, 64, 64, 112, 3, 1, 1, DataType::f32},
        {1, 64, 64, 112, 3, 1, 1, DataType::f16},
        {1, 64, 128, 112, 3, 1, 1, DataType::f16},
        {1, 32, 32, 224, 1, 1, 1, DataType::f16},
        {1, 64, 64, 224, 3, 2, 1, DataType::f32},
        {2, 48, 64, 112, 3, 1, 1, DataType::f16},
        {1, 32, 32, 224, 3, 1, 2, DataType::f32},
        // Past the 65535-pixel wall the old guard stood at, and a stride-2 plane
        // that lands exactly on it: both are the fill's 32-bit offset span, and
        // the CPU oracle is what proves the span is the right bound.
        {1, 32, 32, 320, 3, 1, 1, DataType::f16},
        {1, 16, 32, 512, 3, 2, 1, DataType::f32},
    };
    for (const Wide& w : cases) {
        ConvCase c;
        c.n = w.n;
        c.ic = w.ic;
        c.oc = w.oc;
        c.ih = c.iw = w.side;
        c.kh = c.kw = w.kh;
        c.sh = c.sw = w.sh;
        c.ph = c.pw = w.kh / 2;
        c.groups = w.groups;
        c.dtype = w.dtype;
        c.use_bias = true;
        run_case(c, "cuda_conv2d_wide_planes");
    }
}

NNOPS_TEST(cuda_conv2d_i8) {
    // The quantized route on CUDA: a pre-quantized u8/s8 activation against an
    // s8 HWIO filter, compared bit for bit with the CPU route.
    // Integer destinations are exact by construction (the accumulate is int32);
    // the dequantizing ones use the fp tolerance.
    struct I8Conv {
        int64_t n, ic, ih, iw, oc, kh, kw, sh, sw, dh, dw, ph, pw, groups;
        DataType adt, odt;
        bool bias, relu;
        uint64_t seed;
        bool cl = true;    // NHWC activation + HWIO filter
        int32_t zp_x = 999;  // 999 keeps the default; else an explicit value
        bool sym_w = false;  // all-zero per-channel weight zero points
    };
    // Every (activation, destination) pair the quantized route resolves,
    // (NHWC activation + HWIO filter).
    const I8Conv cases[] = {
        // Isolation: s8 activation with a zero point of 0 and symmetric weights,
        // so the raw accumulate is the whole answer.
        {1, 16, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, DataType::s8, DataType::s8, false, false, 77, true, 0, true},
        // Only the -zp_a·R_b[n] half is live (the weight zero points are zero).
        {1, 16, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, DataType::u8, DataType::s8, false, false, 79, true, 128, true},
        // Only the -(R_a[m] - K·zp_a)·zp_b[n] half is live (the activation zero
        // point is zero).
        {1, 16, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, DataType::s8, DataType::s8, false, false, 78, true, 0, false},

        // IC/G a multiple of 16 so the vector A-fill runs.
        {1, 16, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, DataType::u8, DataType::s8, true, true, 111, true},
        {1, 16, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, DataType::s8, DataType::s8, true, false, 113, true},
        {1, 16, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, DataType::u8, DataType::u8, true, true, 115, true},
        {1, 16, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, DataType::s8, DataType::u8, true, true, 117, true},
        {1, 16, 8, 8, 8, 1, 1, 1, 1, 1, 1, 0, 0, 1, DataType::u8, DataType::f32, true, true, 119, true},
        {1, 16, 8, 8, 8, 1, 1, 1, 1, 1, 1, 0, 0, 1, DataType::s8, DataType::f32, false, false, 121, true},
        {1, 16, 8, 8, 8, 3, 3, 2, 1, 1, 1, 1, 1, 1, DataType::u8, DataType::f16, true, true, 123, true},
        {1, 16, 8, 8, 8, 3, 3, 1, 1, 2, 1, 1, 1, 1, DataType::s8, DataType::f16, true, true, 125, true},
        // Groups split ICG to 8, so this one takes the scalar fill.
        {1, 16, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 2, DataType::u8, DataType::s8, true, true, 127, true},
        // Planes wider than the 64-pixel tile, on both fills: the per-pixel
        // activation sums are shared memory indexed by tile row.
        {1, 16, 16, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, DataType::u8, DataType::s8, true, true, 133, true},
        {1, 16, 12, 12, 16, 3, 3, 1, 1, 1, 1, 1, 1, 2, DataType::u8, DataType::u8, true, true, 137, true},

        // And planes of thousands of pixels, for the same three reasons as the
        // fp wide-plane table.
        {1, 64, 112, 112, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, DataType::u8, DataType::s8, true, true, 211, true},
        {1, 32, 224, 224, 32, 1, 1, 1, 1, 1, 1, 0, 0, 1, DataType::s8, DataType::f32, false, false, 213, true},

        // Depthwise-shaped groups (IC/G = OC/G = 1) stay on the quantized GEMM.
        {1, 16, 8, 8, 16, 3, 3, 1, 1, 1, 1, 1, 1, 16, DataType::s8, DataType::u8, true, true, 131, true},

        // s8 -> s8 (symmetric quantization) beyond the first cases' 8x8 tile:
        // 128 channels a CTA is the wide tile, OC = 24 misses the 16 B copy-out
        // unit and falls to the 4 B one, and the grouped one splits ICG to 16.
        {1, 128, 16, 16, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, DataType::s8, DataType::s8, true, false, 301, true, 0, true},
        {1, 32, 32, 32, 24, 3, 3, 1, 1, 1, 1, 1, 1, 1, DataType::s8, DataType::s8, true, true, 303, true, -6, false},
        {1, 32, 16, 16, 32, 3, 3, 1, 1, 1, 1, 1, 1, 2, DataType::s8, DataType::s8, true, false, 305, true, 0, false},
        {1, 64, 56, 56, 128, 1, 1, 1, 1, 1, 1, 0, 0, 1, DataType::s8, DataType::s8, true, false, 307, true, 0, true},

        // Planes small enough for the k reduction to be cut, so the split
        // reduce's plane sums and the per-split activation sums are what
        // produce the answer; the second one leaves through the dequant path.
        {1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, DataType::s8, DataType::s8, true, true, 311, true, 0, false},
        {1, 128, 7, 7, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, DataType::s8, DataType::f32, false, false, 313, true, 0, false},
    };

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
        const I8Conv& c = cases[ci];
        const int64_t icg = c.ic / c.groups;
        test::XorShift128 rng(c.seed);
        const int64_t an = c.n * c.ic * c.ih * c.iw;
        std::vector<uint8_t> a(static_cast<size_t>(an));
        for (auto& e : a) { e = static_cast<uint8_t>(rng.next_u64() & 0xFFu); }
        const int64_t wn = c.oc * icg * c.kh * c.kw;
        std::vector<int8_t> w(static_cast<size_t>(wn));
        for (auto& e : w) {
            e = static_cast<int8_t>(int32_t(rng.next_u64() % 255) - 127);
        }

        std::vector<float> w_s(static_cast<size_t>(c.oc));
        std::vector<int32_t> w_zp(static_cast<size_t>(c.oc));
        for (int64_t i = 0; i < c.oc; ++i) {
            w_s[static_cast<size_t>(i)] = 0.008f + 0.001f * float(i % 5);
            w_zp[static_cast<size_t>(i)] = c.sym_w ? 0 : int32_t(i % 7) - 3;
        }

        QuantParams qa;
        qa.scale = 0.0078125f;
        qa.zero_point = c.zp_x != 999 ? c.zp_x : (c.adt == DataType::u8 ? 128 : -6);
        QuantParams qw;
        qw.granularity = QuantGranularity::PerChannel;
        qw.scale_data = w_s.data();
        qw.zero_point_data = w_zp.data();
        qw.num_scales = c.oc;
        qw.scale = 0.008f;
        qw.zero_point = 0;
        QuantParams qc;
        qc.scale = 0.03f;
        qc.zero_point = c.odt == DataType::u8 ? 110 : 0;

        const bool int_out = c.odt == DataType::s32 || c.odt == DataType::s8 ||
                             c.odt == DataType::u8;
        const DataType bdt = int_out ? DataType::s32 : c.odt;
        std::vector<uint8_t> bias_bytes;
        if (c.bias) {
            bias_bytes.resize(static_cast<size_t>(c.oc) *
                              size_t(data_type_size(bdt)));
            for (int64_t i = 0; i < c.oc; ++i) {
                if (bdt == DataType::s32) {
                    reinterpret_cast<int32_t*>(bias_bytes.data())[i] =
                        int32_t(rng.next_u64() % 2001) - 1000;
                } else if (bdt == DataType::f16) {
                    reinterpret_cast<nnops::backend::cpu::half*>(
                        bias_bytes.data())[i] =
                        nnops::backend::cpu::float_to_half(
                            rng.next_float(-0.5f, 0.5f));
                } else {
                    reinterpret_cast<float*>(bias_bytes.data())[i] =
                        rng.next_float(-0.5f, 0.5f);
                }
            }
        }

        Conv2DAttributes attrs;
        attrs.kernel_size = {c.kh, c.kw};
        attrs.stride = {c.sh, c.sw};
        attrs.dilation = {c.dh, c.dw};
        attrs.padding = {c.ph, c.pw};
        attrs.groups = c.groups;
        attrs.output_dtype = c.odt;
        if (c.relu) { attrs.epilogue.type = EpilogueActivateType::Relu; }

        const auto view = [](const std::vector<int64_t>& sh, DataType dt,
                             void* d, TensorLayout lay,
                             const QuantParams& qp = {}) {
            return TensorView(std::span<const int64_t>(sh.data(), sh.size()), dt,
                              d, lay, qp);
        };
        const TensorLayout alay = TensorLayout::NHWC;
        const TensorLayout wlay = TensorLayout::HWIO;
        const std::vector<int64_t> ashape = {c.n, c.ih, c.iw, c.ic};
        const std::vector<int64_t> wshape = {c.kh, c.kw, icg, c.oc};

        TensorView av = view(ashape, c.adt, a.data(), alay, qa);
        TensorView wv = view(wshape, DataType::s8, w.data(), wlay, qw);
        TensorView biasv;
        std::vector<TensorView> ins = {av, wv};
        std::vector<TensorDesc> descs = {av.desc(), wv.desc()};
        if (c.bias) {
            biasv = view({c.oc}, bdt, bias_bytes.data(), TensorLayout::Dense);
            ins.push_back(biasv);
            descs.push_back(biasv.desc());
        }

        auto op_cpu = Conv2D::create(attrs, Backend::CPU);
        const TensorDesc od = op_cpu->getOutputTensorDesc(descs).front();
        const std::vector<int64_t> odims(od.dims.begin(), od.dims.begin() + od.rank);
        int64_t on = 1;
        for (auto d : odims) { on *= d; }
        const size_t obytes = static_cast<size_t>(on) *
                              size_t(data_type_size(od.dtype));
        std::vector<uint8_t> out_cpu(obytes, 0);
        std::vector<uint8_t> out_dev(obytes, 0);
        TensorView cv = view(odims, od.dtype, out_cpu.data(), od.layout, qc);
        cpu_run(*op_cpu, cv, ins);

        CudaBuffer da(static_cast<size_t>(an)), dw(static_cast<size_t>(wn));
        da.upload(a.data(), a.size());
        dw.upload(w.data(), w.size());
        TensorView dav = view(ashape, c.adt, da.data(), alay, qa);
        TensorView dwv = view(wshape, DataType::s8, dw.data(), wlay, qw);
        CudaBuffer dbias;
        std::vector<TensorView> dins = {dav, dwv};
        if (c.bias) {
            dbias = CudaBuffer(bias_bytes.size());
            dbias.upload(bias_bytes.data(), bias_bytes.size());
            dins.push_back(view({c.oc}, bdt, dbias.data(), TensorLayout::Dense));
        }
        CudaBuffer dout(obytes);
        TensorView dcv = view(odims, od.dtype, dout.data(), od.layout, qc);
        auto op_dev = Conv2D::create(attrs, Backend::CUDA);
        // Twice: the automatic split-K rule, then a forced cut. The forced arm is
        // the one compared with the CPU below, and the two must agree with each
        // other on every shape -- not just the ones the heuristic picks.
        auto run_arm = [&](int forced) {
            nnops::detail::cuda_splitk.store(forced, std::memory_order_relaxed);
            dout.zero();
            cuda_run(*op_dev, dcv, dins);
            dout.download(out_dev.data(), obytes);
        };
        run_arm(-1);
        std::vector<uint8_t> out_auto(out_dev);
        run_arm(3);
        nnops::detail::cuda_splitk.store(-1, std::memory_order_relaxed);
        NNOPS_EXPECT_TRUE(out_auto == out_dev);

        if (int_out) {
            if (std::memcmp(out_cpu.data(), out_dev.data(), obytes) != 0) {
                const size_t esz = size_t(data_type_size(od.dtype));
                const size_t n = obytes / esz;
                size_t bad = 0;
                while (bad < n &&
                       std::memcmp(out_cpu.data() + bad * esz,
                                   out_dev.data() + bad * esz, esz) == 0) {
                    ++bad;
                }
                char msg[512];
                int off = std::snprintf(msg, sizeof(msg),
                              "conv2d_i8 case %zu (cl=%d act=%d out=%d): elem %zu; "
                              "cpu:", ci, int(c.cl), int(c.adt), int(c.odt), bad);
                for (size_t k = 0; k < 8 && k < n; ++k) {
                    off += std::snprintf(msg + off, sizeof(msg) - size_t(off), " %d",
                                         od.dtype == DataType::s8
                                             ? int(int8_t(out_cpu[k]))
                                             : int(out_cpu[k]));
                }
                off += std::snprintf(msg + off, sizeof(msg) - size_t(off), " | cuda:");
                for (size_t k = 0; k < 8 && k < n; ++k) {
                    off += std::snprintf(msg + off, sizeof(msg) - size_t(off), " %d",
                                         od.dtype == DataType::s8
                                             ? int(int8_t(out_dev[k]))
                                             : int(out_dev[k]));
                }
                throw std::runtime_error(msg);
            }
            continue;
        }
        const size_t n = obytes / size_t(data_type_size(od.dtype));
        for (size_t i = 0; i < n; ++i) {
            float x = 0.0f, y = 0.0f;
            if (od.dtype == DataType::f16) {
                x = nnops::backend::cpu::half_to_float(
                    reinterpret_cast<const nnops::backend::cpu::half*>(
                        out_dev.data())[i]);
                y = nnops::backend::cpu::half_to_float(
                    reinterpret_cast<const nnops::backend::cpu::half*>(
                        out_cpu.data())[i]);
            } else {
                x = reinterpret_cast<const float*>(out_dev.data())[i];
                y = reinterpret_cast<const float*>(out_cpu.data())[i];
            }
            NNOPS_EXPECT_TRUE(std::fabs(x - y) <=
                              1e-4f * std::max(1.0f, std::fabs(y)));
        }
    }

    // An s32 destination has no Conv2D route on either backend (MatMul keeps it).
    {
        const int64_t ashape[] = {1, 8, 8, 4};   // NHWC [N, H, W, C]
        const int64_t wshape[] = {3, 3, 4, 6};
        std::vector<uint8_t> a(1 * 4 * 8 * 8, 1);
        std::vector<int8_t> w(6 * 4 * 3 * 3, 1);
        QuantParams qa;
        qa.scale = 0.01f;
        qa.zero_point = 128;
        QuantParams qw;
        qw.scale = 0.01f;
        qw.zero_point = 0;
        TensorView av(ashape, DataType::u8, a.data(), TensorLayout::NHWC, qa);
        TensorView wv(wshape, DataType::s8, w.data(), TensorLayout::HWIO, qw);
        const int64_t cshape[] = {1, 6, 6, 6};
        std::vector<int32_t> out(1 * 6 * 6 * 6, 0);
        TensorView cv(cshape, DataType::s32, out.data(), TensorLayout::NHWC);

        Conv2DAttributes attrs;
        attrs.kernel_size = {3, 3};
        attrs.padding = {1, 1};
        attrs.output_dtype = DataType::s32;
        auto op = Conv2D::create(attrs, Backend::CUDA);
        const TensorView ins[] = {av, wv};
        NNOPS_EXPECT_EQ(op->compute({&cv, 1}, ins, cuda_context(), nullptr),
                        Status::unsupported);
    }
}

/// The prepacked filter is the same bytes the gather would have produced, so
/// the packed call must be bit-identical to the raw one — not merely close. The
/// cases walk the packer's three tails: an oc block the tile only half fills, an
/// icg that does not divide a 16-byte run, and a group count that makes every
/// group one row wide.
NNOPS_TEST(cuda_conv2d_i8_packed_filter) {
    struct PackedCase {
        int64_t ic, ih, iw, oc, kh, kw, groups;
    };
    const PackedCase cases[] = {
        {64, 32, 32, 64, 3, 3, 1},
        {16, 8, 8, 40, 3, 3, 1},
        {8, 8, 8, 16, 3, 3, 8},
        {64, 16, 16, 128, 1, 1, 1},
        {24, 8, 8, 64, 3, 3, 1},
    };
    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
        const PackedCase& c = cases[ci];
        const int64_t icg = c.ic / c.groups;
        test::XorShift128 rng(311 + ci);

        const int64_t an = c.ic * c.ih * c.iw;
        std::vector<uint8_t> a(static_cast<size_t>(an));
        for (auto& e : a) { e = static_cast<uint8_t>(rng.next_u64() & 0xFFu); }
        const int64_t wn = c.oc * icg * c.kh * c.kw;
        std::vector<int8_t> w(static_cast<size_t>(wn));
        for (auto& e : w) { e = static_cast<int8_t>(int32_t(rng.next_u64() % 255) - 127); }

        std::vector<float> w_s(static_cast<size_t>(c.oc));
        std::vector<int32_t> w_zp(static_cast<size_t>(c.oc));
        for (int64_t i = 0; i < c.oc; ++i) {
            w_s[static_cast<size_t>(i)] = 0.008f + 0.001f * float(i % 5);
            w_zp[static_cast<size_t>(i)] = int32_t(i % 7) - 3;
        }
        QuantParams qa;
        qa.scale = 0.0078125f;
        qa.zero_point = 128;
        QuantParams qw;
        qw.granularity = QuantGranularity::PerChannel;
        qw.scale_data = w_s.data();
        qw.zero_point_data = w_zp.data();
        qw.num_scales = c.oc;
        qw.scale = 0.008f;
        qw.zero_point = 0;
        QuantParams qc;
        qc.scale = 0.03f;
        qc.zero_point = 0;

        Conv2DAttributes attrs;
        attrs.kernel_size = {c.kh, c.kw};
        attrs.padding = {c.kh / 2, c.kw / 2};
        attrs.groups = c.groups;
        attrs.output_dtype = DataType::s8;

        const auto view = [](const std::vector<int64_t>& sh, DataType dt, void* d,
                             TensorLayout lay, const QuantParams& qp) {
            return TensorView(std::span<const int64_t>(sh.data(), sh.size()), dt, d, lay, qp);
        };
        const std::vector<int64_t> ashape = {1, c.ih, c.iw, c.ic};
        const std::vector<int64_t> wshape = {c.kh, c.kw, icg, c.oc};
        TensorView av = view(ashape, DataType::u8, nullptr, TensorLayout::NHWC, qa);
        TensorView wv = view(wshape, DataType::s8, nullptr, TensorLayout::HWIO, qw);
        const std::vector<TensorDesc> descs = {av.desc(), wv.desc()};
        const TensorDesc od =
            Conv2D::create(attrs, Backend::CPU)->getOutputTensorDesc(descs).front();
        const std::vector<int64_t> odims(od.dims.begin(), od.dims.begin() + od.rank);
        int64_t on = 1;
        for (auto d : odims) { on *= d; }

        CudaBuffer da(static_cast<size_t>(an)), dw(static_cast<size_t>(wn));
        da.upload(a.data(), a.size());
        dw.upload(w.data(), w.size());
        TensorView dav = view(ashape, DataType::u8, da.data(), TensorLayout::NHWC, qa);
        TensorView dwv = view(wshape, DataType::s8, dw.data(), TensorLayout::HWIO, qw);
        auto op = Conv2D::create(attrs, Backend::CUDA);

        // The raw arm, then the two-step pack and the same call over the blob.
        CudaBuffer draw(static_cast<size_t>(on));
        draw.zero();
        TensorView drawv = view(odims, DataType::s8, draw.data(), od.layout, qc);
        const TensorView raw_ins[] = {dav, dwv};
        cuda_run(*op, drawv, raw_ins);

        std::vector<TensorView> qin = {dwv};
        std::vector<TensorView> qout = {TensorView{}};
        op->prepackWeights(qin, qout, cuda_context());
        NNOPS_EXPECT_TRUE(qout[0].rank() == 2);
        NNOPS_EXPECT_TRUE(qout[0].layout() == TensorLayout::PackedHWIOCuda);
        NNOPS_EXPECT_EQ(qout[0].shape(0), c.oc);
        CudaBuffer dblob(static_cast<size_t>(qout[0].numel()));
        const std::vector<int64_t> bshape = {qout[0].shape(0), qout[0].shape(1)};
        TensorView bv = view(bshape, DataType::s8, dblob.data(), qout[0].layout(), qw);
        std::vector<TensorView> pin = {dwv};
        std::vector<TensorView> pout = {bv};
        op->prepackWeights(pin, pout, cuda_context());
        NNOPS_CUDA_CHECK(cudaDeviceSynchronize());

        CudaBuffer dpacked(static_cast<size_t>(on));
        dpacked.zero();
        TensorView dpackedv = view(odims, DataType::s8, dpacked.data(), od.layout, qc);
        const TensorView packed_ins[] = {dav, bv};
        cuda_run(*op, dpackedv, packed_ins);

        std::vector<int8_t> raw(static_cast<size_t>(on)), packed(static_cast<size_t>(on));
        draw.download(raw.data(), raw.size());
        dpacked.download(packed.data(), packed.size());
        if (std::memcmp(raw.data(), packed.data(), raw.size()) != 0) {
            size_t bad = 0;
            while (bad < raw.size() && raw[bad] == packed[bad]) { ++bad; }
            char msg[512];
            std::snprintf(msg, sizeof(msg),
                          "conv2d_i8 packed case %zu (ic=%lld oc=%lld g=%lld k%lldx%lld): "
                          "elem %zu of %zu; raw %d packed %d",
                          ci, (long long)c.ic, (long long)c.oc, (long long)c.groups,
                          (long long)c.kh, (long long)c.kw, bad, raw.size(), int(raw[bad]),
                          int(packed[bad]));
            throw std::runtime_error(msg);
        }
    }
}

#endif  // NNOPS_HAS_CUDA
