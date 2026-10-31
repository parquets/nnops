/// @file test_cuda_depthwise_conv.cpp
/// @brief CUDA DepthwiseConv against the CPU backend on the same inputs.
///
/// Every case builds one random host operand set, runs the operator twice —
/// once with Backend::CPU and once with Backend::CUDA over device copies — and
/// compares the outputs. The CPU route is the oracle and both routes read the
/// identical views, so a planar case exercises the width lane and a channel-last
/// case the channel lane without any layout conversion in the comparison.
///
/// Cases cover the shapes the kernel branches on: the vector lane's exact fit
/// (C = 4 f32 / 8 f16), a short and a long scalar tail (C = 5 and 17), padded
/// and strided windows, dilation, the bias seed, the epilogue, add_to's ordering,
/// the 3D forms, and f16 throughout. Planar tensors carry a padded pitch, as a
/// graph hands them; channel-last ones stay dense because their strides never
/// consult the pitch.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/depthwise_conv.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/cuda_test_helper.hpp"

#include <array>
#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

struct DwCase {
    std::vector<int64_t> in_shape;   // logical [N, C, H, W] or [N, C, D, H, W]
    std::array<int64_t, 3> kernel = {1, 3, 3};
    std::array<int64_t, 3> stride = {1, 1, 1};
    std::array<int64_t, 3> dilation = {1, 1, 1};
    std::array<int64_t, 3> padding = {0, 0, 0};
    Epilogue epilogue{};
    DataType dtype = DataType::f32;
    bool channel_last = false;
    bool bias = false;
    bool add_to = false;
};

std::pair<float, float> tolerance(DataType dt)
{
    return dt == DataType::f16 ? std::pair<float, float>{kF16AccumTol, 2e-2f}
                               : std::pair<float, float>{1e-5f, 1e-5f};
}

void run_case(const DwCase& c, const char* what)
{
    DepthwiseConvAttributes attrs;
    attrs.kernel_size = c.kernel;
    attrs.stride = c.stride;
    attrs.dilation = c.dilation;
    attrs.padding = c.padding;
    attrs.epilogue = c.epilogue;
    attrs.add_to = c.add_to;

    const int64_t C = c.in_shape[1];
    const int64_t srank = static_cast<int64_t>(c.in_shape.size()) - 2;
    const bool three_d = srank == 3;
    const TensorLayout in_layout =
        c.channel_last ? (three_d ? TensorLayout::NDHWC : TensorLayout::NHWC)
                       : (three_d ? TensorLayout::NCDHW : TensorLayout::NCHW);
    const bool padded = !c.channel_last;

    HostTensor in = make_host_random(c.in_shape, c.dtype, padded, 17, -1.0f, 1.0f,
                                     in_layout);

    // Planar reads the raw [C, 1, (KD,) KH, KW]; channel-last the HWCM
    // [(KD,) KH, KW, C], whose every tap is a contiguous run of C.
    std::vector<int64_t> w_shape;
    TensorLayout w_layout = in_layout;
    if (c.channel_last) {
        if (three_d) { w_shape.push_back(c.kernel[0]); }
        w_shape.push_back(c.kernel[1]);
        w_shape.push_back(c.kernel[2]);
        w_shape.push_back(C);
        w_layout = TensorLayout::HWCM;
    } else {
        w_shape = {C, 1};
        if (three_d) { w_shape.push_back(c.kernel[0]); }
        w_shape.push_back(c.kernel[1]);
        w_shape.push_back(c.kernel[2]);
    }
    HostTensor w = make_host_random(w_shape, c.dtype, false, 23, -0.5f, 0.5f,
                                    w_layout);

    HostTensor b;
    if (c.bias) {
        b = make_host_random(std::vector<int64_t>{C}, c.dtype, false, 29, -0.25f,
                             0.25f);
    }

    std::vector<TensorDesc> descs = {in.view.desc(), w.view.desc()};
    if (c.bias) { descs.push_back(b.view.desc()); }

    auto op_cpu = DepthwiseConv::create(attrs, Backend::CPU);
    const std::vector<TensorDesc> outs = op_cpu->getOutputTensorDesc(descs);
    NNOPS_EXPECT_TRUE(!outs.empty());
    const TensorDesc od = outs.front();
    NNOPS_EXPECT_TRUE(od.layout == in_layout);

    // add_to reads the destination, so both runs must start from the same bytes.
    const auto seed = [&](uint64_t s) {
        return c.add_to ? make_host_random(od.dims, od.dtype, padded, s, -1.0f,
                                           1.0f, in_layout)
                        : make_host_zeroed(od.dims, od.dtype, padded, in_layout);
    };
    HostTensor out_ref = seed(41);
    HostTensor out_dev = seed(41);

    std::vector<TensorView> cpu_ins = {in.view, w.view};
    if (c.bias) { cpu_ins.push_back(b.view); }
    cpu_run(*op_cpu, out_ref.view, cpu_ins);

    auto op_dev = DepthwiseConv::create(attrs, Backend::CUDA);
    DeviceTensor din = to_device(in);
    DeviceTensor dw = to_device(w);
    DeviceTensor db;
    std::vector<TensorView> dev_ins = {din.view, dw.view};
    if (c.bias) {
        db = to_device(b);
        dev_ins.push_back(db.view);
    }
    DeviceTensor dout = alloc_like(out_dev.view, out_dev);
    cuda_run(*op_dev, dout.view, dev_ins);
    from_device(dout, out_dev);

    const auto [tol_abs, tol_rel] = tolerance(c.dtype);
    expect_same(out_dev, out_ref, tol_abs, tol_rel, what);
}

}  // namespace

// ============================================================
// Planar
// ============================================================

NNOPS_TEST(cuda_dwconv_planar_stride_padding) {
    for (const auto& k : {std::array<int64_t, 3>{1, 3, 3},
                          std::array<int64_t, 3>{1, 1, 1}}) {
        DwCase c;
        c.in_shape = {2, 4, 12, 15};
        c.kernel = k;
        c.stride = {1, 2, 2};
        c.padding = {0, 1, 1};
        run_case(c, "cuda_dwconv_planar_stride_padding");
    }
}

NNOPS_TEST(cuda_dwconv_planar_dilation_epilogue) {
    DwCase c;
    c.in_shape = {1, 5, 10, 9};
    c.dilation = {1, 2, 2};
    c.epilogue.type = EpilogueActivateType::Relu;
    run_case(c, "cuda_dwconv_planar_dilation_epilogue");

    c.in_shape = {1, 3, 7, 6};
    c.dilation = {1, 1, 1};
    c.padding = {0, 1, 1};
    c.epilogue.type = EpilogueActivateType::Relu6;
    c.bias = true;
    c.add_to = true;
    run_case(c, "cuda_dwconv_planar_dilation_epilogue");
}

NNOPS_TEST(cuda_dwconv_planar_3d) {
    DwCase c;
    c.in_shape = {1, 4, 4, 6, 9};
    c.kernel = {3, 3, 3};
    c.padding = {1, 1, 1};
    c.bias = true;
    run_case(c, "cuda_dwconv_planar_3d");
    c.in_shape = {1, 2, 3, 5, 5};
    c.kernel = {2, 2, 2};
    c.padding = {0, 0, 0};
    c.add_to = true;
    run_case(c, "cuda_dwconv_planar_3d");
}

NNOPS_TEST(cuda_dwconv_planar_f16) {
    DwCase c;
    c.in_shape = {1, 8, 9, 11};
    c.dtype = DataType::f16;
    c.padding = {0, 1, 1};
    c.bias = true;
    run_case(c, "cuda_dwconv_planar_f16");
}

// ============================================================
// Channel-last
// ============================================================

NNOPS_TEST(cuda_dwconv_last_vector_lane) {
    // C = 4 (f32) and 8 (f16) divide the 16 B vector exactly; 5 and 17 leave a
    // scalar tail, both a short one and one longer than a vector.
    for (int64_t C : {4, 5, 8, 17}) {
        for (DataType dt : {DataType::f32, DataType::f16}) {
            if (dt == DataType::f32 && (C == 8 || C == 17)) { continue; }
            DwCase c;
            c.in_shape = {2, C, 9, 11};
            c.dtype = dt;
            c.channel_last = true;
            run_case(c, "cuda_dwconv_last_vector_lane");
        }
    }
}

NNOPS_TEST(cuda_dwconv_last_window) {
    DwCase c;
    c.in_shape = {1, 16, 14, 13};
    c.channel_last = true;
    c.stride = {1, 2, 2};
    c.padding = {0, 1, 1};
    c.bias = true;
    run_case(c, "cuda_dwconv_last_window");

    c.in_shape = {1, 8, 10, 10};
    c.stride = {1, 1, 1};
    c.dilation = {1, 2, 2};
    c.padding = {0, 0, 0};
    c.epilogue.type = EpilogueActivateType::HardSwish;
    c.add_to = true;
    run_case(c, "cuda_dwconv_last_window");
}

NNOPS_TEST(cuda_dwconv_last_f16) {
    DwCase c;
    c.in_shape = {1, 8, 7, 7};
    c.dtype = DataType::f16;
    c.channel_last = true;
    c.kernel = {1, 1, 1};
    c.stride = {1, 2, 2};
    run_case(c, "cuda_dwconv_last_f16");
    c.kernel = {1, 3, 3};
    c.stride = {1, 1, 1};
    c.padding = {0, 1, 1};
    c.epilogue.type = EpilogueActivateType::Gelu;
    c.bias = true;
    run_case(c, "cuda_dwconv_last_f16");
}

NNOPS_TEST(cuda_dwconv_last_3d) {
    DwCase c;
    c.in_shape = {1, 8, 3, 5, 6};
    c.kernel = {2, 2, 2};
    c.stride = {1, 1, 1};
    c.channel_last = true;
    run_case(c, "cuda_dwconv_last_3d");
    c.kernel = {3, 3, 3};
    c.padding = {1, 1, 1};
    c.bias = true;
    c.add_to = true;
    run_case(c, "cuda_dwconv_last_3d");
}

#endif  // NNOPS_HAS_CUDA
