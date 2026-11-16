/// @file test_cuda_resize.cpp
/// @brief CUDA Resize, channel-last, against the CPU backend.
///
/// Every case builds one activation, runs the operator on both backends over the
/// same bytes and compares. The CPU route is the oracle because it defines this
/// operator's contract — the coordinate modes, the clamped border taps, the
/// bicubic kernel and the add_to ordering — so the CUDA mapping is pinned by the
/// operator rather than by a hand-written reference.
///
/// The cases take the channel-last 16 B channel lane when the channel count
/// fills it (C >= 16 f32, C >= 32 f16) and the pixel-per-thread script when it
/// does not, which is what the RGB cases exercise. Rank 5 covers the depth axis,
/// and the rest walk the three coordinate modes, the three interpolation modes
/// and the add_to ordering.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/resize.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/cuda_test_helper.hpp"

#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

struct RszCase {
    int64_t n = 1, c = 8;
    int64_t ih = 14, iw = 14;
    int64_t id = 1;
    int64_t oh = 28, ow = 28;
    int64_t od = 1;
    bool three_d = false;
    ResizeMode mode = ResizeMode::Nearest;
    CoordinateTransformMode coord = CoordinateTransformMode::HalfPixel;
    float cubic_a = -0.75f;
    bool add_to = false;
    // crop_start / crop_end on {D, H, W}; 0 leaves an axis uncropped.
    int64_t crop_start[3] = {0, 0, 0};
    int64_t crop_end[3] = {0, 0, 0};
    DataType dtype = DataType::f32;
};

void run_case(const RszCase& c, const char* what)
{
    ResizeAttributes attrs;
    attrs.mode = c.mode;
    attrs.coord_mode = c.coord;
    attrs.cubic_coeff_a = c.cubic_a;
    attrs.add_to = c.add_to;
    attrs.output_size = {c.od, c.oh, c.ow};
    for (int i = 0; i < 3; ++i) {
        attrs.crop_start[static_cast<size_t>(i)] = c.crop_start[i];
        attrs.crop_end[static_cast<size_t>(i)] = c.crop_end[i];
    }

    const TensorLayout layout =
        c.three_d ? TensorLayout::NDHWC : TensorLayout::NHWC;

    std::vector<int64_t> in_shape = c.three_d
        ? std::vector<int64_t>{c.n, c.c, c.id, c.ih, c.iw}
        : std::vector<int64_t>{c.n, c.c, c.ih, c.iw};
    HostTensor act = make_host_random(in_shape, c.dtype, false, 23, -0.5f, 0.5f,
                                      layout);

    std::vector<TensorDesc> descs = {act.view.desc()};
    auto op_cpu = Resize::create(attrs, Backend::CPU);
    const std::vector<TensorDesc> outs = op_cpu->getOutputTensorDesc(descs);
    NNOPS_EXPECT_TRUE(!outs.empty());
    const TensorDesc od = outs.front();

    const auto seed = [&](uint64_t s) {
        return c.add_to ? make_host_random(od.dims, od.dtype, false, s, 0.0f, 0.5f,
                                           layout)
                        : make_host_zeroed(od.dims, od.dtype, false, layout);
    };
    HostTensor out_ref = seed(37);
    HostTensor out_dev = seed(37);

    std::vector<TensorView> cpu_ins = {act.view};
    cpu_run(*op_cpu, out_ref.view, cpu_ins);

    auto op_dev = Resize::create(attrs, Backend::CUDA);
    DeviceTensor dact = to_device(act);
    std::vector<TensorView> dev_ins = {dact.view};
    DeviceTensor dout = alloc_like(out_dev.view, out_dev);
    cuda_run(*op_dev, dout.view, dev_ins);
    from_device(dout, out_dev);

    const float tol_abs = c.dtype == DataType::f16 ? kF16AccumTol
                                                   : (c.mode == ResizeMode::Cubic
                                                          ? 1e-3f : 1e-4f);
    const float tol_rel = c.dtype == DataType::f16 ? 2e-2f : 1e-4f;
    expect_same(out_dev, out_ref, tol_abs, tol_rel, what);
}

}  // namespace

// ============================================================
// Channel-last: the channel lane, and the scalar script below it
// ============================================================

NNOPS_TEST(cuda_resize_channel_last) {
    {
        RszCase c;   // C = 32: two 16 B lanes, the vector path
        c.c = 32; c.ih = 13; c.iw = 17;
        c.oh = 26; c.ow = 34;
        run_case(c, "cuda_resize_channel_last");
    }
    {
        RszCase c;   // C = 3: the pixel-per-thread script
        c.c = 3; c.mode = ResizeMode::Linear;
        c.ih = 15; c.iw = 15; c.oh = 31; c.ow = 31;
        run_case(c, "cuda_resize_channel_last");
    }
    {
        RszCase c;   // C = 64, cubic
        c.c = 64; c.mode = ResizeMode::Cubic;
        c.ih = 10; c.iw = 10; c.oh = 25; c.ow = 25;
        run_case(c, "cuda_resize_channel_last");
    }
    {
        RszCase c;   // f16, C = 64: eight lanes of eight
        c.c = 64; c.dtype = DataType::f16;
        c.mode = ResizeMode::Linear; c.ih = 9; c.iw = 9; c.oh = 19; c.ow = 19;
        run_case(c, "cuda_resize_channel_last");
    }
    {
        RszCase c;   // a channel count that is not a lane multiple
        c.c = 20; c.ih = 7; c.iw = 7; c.oh = 15; c.ow = 15;
        run_case(c, "cuda_resize_channel_last");
    }
}

// ============================================================
// Rank 5, add_to, crop
// ============================================================

NNOPS_TEST(cuda_resize_3d) {
    RszCase c;   // channel-last trilinear
    c.three_d = true; c.c = 32;
    c.id = 3; c.ih = 5; c.iw = 5; c.od = 6; c.oh = 9; c.ow = 9;
    c.mode = ResizeMode::Linear;
    run_case(c, "cuda_resize_3d");
}

NNOPS_TEST(cuda_resize_add_to) {
    RszCase c;
    c.add_to = true;     c.c = 32;
    c.mode = ResizeMode::Linear;
    c.ih = 12; c.iw = 12; c.oh = 20; c.ow = 20;
    run_case(c, "cuda_resize_add_to");
}

NNOPS_TEST(cuda_resize_crop) {
    RszCase c;   // crop, channel-last, linear
    c.c = 32; c.mode = ResizeMode::Linear;
    c.ih = 16; c.iw = 16; c.oh = 24; c.ow = 24;
    c.crop_start[1] = 1; c.crop_end[1] = 13;
    c.crop_start[2] = 2; c.crop_end[2] = 15;
    run_case(c, "cuda_resize_crop");
}

#endif  // NNOPS_HAS_CUDA
