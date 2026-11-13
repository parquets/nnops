/// @file test_cuda_norm.cpp
/// @brief CUDA Norm against the CPU backend on the same inputs.
///
/// Every case builds one random host tensor and its auxiliary inputs, runs the
/// operator twice — once with Backend::CPU and once with Backend::CUDA over
/// device copies — and compares the outputs. The CPU route is the oracle: it
/// defines the epsilon placement, the scalar-scale rule, the slot order and the
/// add/residual ordering, and both routes read the identical views, so nothing
/// in the comparison depends on a layout conversion.
///
/// The cases split the way the CUDA file does: one block per row for
/// LayerNorm / RMSNorm / L2Norm (tail axis and an interior one), one block per
/// (sample, channel) plane for BatchNorm on both its plane sizes plus the flat
/// non-spatial map, and one block per (sample, group) for GroupNorm. Channel-
/// first carries the 32 B pitch a graph hands the op; channel-last stays dense.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/norm.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/cuda_test_helper.hpp"

#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

struct NormCase {
    NormType type = NormType::LayerNorm;
    std::vector<int64_t> shape;
    int64_t axis = -1;
    int64_t groups = 1;
    DataType dtype = DataType::f32;
    TensorLayout layout = TensorLayout::NCHW;
    bool bias = false;
    bool residual = false;
    bool add_to = false;
    bool scalar_scale = false;
    bool spatial = true;
    /// Force a dense channel-first view: the non-spatial BatchNorm map reads a
    /// flat range, which only means anything when the rows have no padding.
    bool dense = false;
};

/// Two f32 reductions of the same row (the CPU's and the block's) round
/// differently, and f16 has to survive both the accumulation and the store.
std::pair<float, float> tolerance(DataType dt)
{
    return dt == DataType::f16 ? std::pair<float, float>{kF16AccumTol, 3e-2f}
                               : std::pair<float, float>{2e-4f, 2e-4f};
}

void run_case(const NormCase& c, const char* what)
{
    NormAttributes attrs;
    attrs.type = c.type;
    attrs.axis = c.axis;
    attrs.num_groups = c.groups;
    attrs.spatial = c.spatial;
    attrs.add_to = c.add_to;
    attrs.add_residual = c.residual;

    const int64_t rank = static_cast<int64_t>(c.shape.size());
    const bool padded = !c.dense && !is_channel_last(c.layout);
    HostTensor x = make_host_random(c.shape, c.dtype, padded, 23, -1.0f, 1.0f,
                                    c.layout);

    const int64_t ax = c.axis < 0 ? c.axis + rank : c.axis;
    int64_t norm_size = 1;
    for (int64_t d = ax; d < rank; ++d) {
        norm_size *= c.shape[static_cast<size_t>(d)];
    }
    const int64_t C = (rank >= 2) ? c.shape[1] : 1;

    std::vector<HostTensor> own;
    const auto add = [&](std::vector<int64_t> sh, uint64_t seed, float lo, float hi,
                         bool pad, TensorLayout lay) {
        own.push_back(make_host_random(sh, c.dtype, pad, seed, lo, hi, lay));
    };

    // Slot order, per the operator's contract:
    //   BatchNorm [X, scale, bias, mean, var]
    //   LayerNorm [X, scale, bias?, residual?]
    //   RMSNorm   [X, scale, residual?]
    //   GroupNorm [X, scale, bias?]
    //   L2Norm    [X, residual?]
    switch (c.type) {
    case NormType::BatchNorm: {
        const std::vector<int64_t> sh = c.spatial
            ? std::vector<int64_t>{C} : c.shape;
        const bool pad = c.spatial ? false : padded;
        const TensorLayout lay = c.spatial ? TensorLayout::NCHW : c.layout;
        add(sh, 71, -1.0f, 1.0f, pad, lay);   // scale
        add(sh, 72, -1.0f, 1.0f, pad, lay);   // bias
        add(sh, 73, -1.0f, 1.0f, pad, lay);   // mean
        add(sh, 74, 0.05f, 1.5f, pad, lay);   // var — must stay positive
        break;
    }
    case NormType::LayerNorm: {
        const std::vector<int64_t> sh = {c.scalar_scale ? 1 : norm_size};
        add(sh, 81, -1.0f, 1.0f, false, TensorLayout::NCHW);
        if (c.bias) { add(sh, 82, -1.0f, 1.0f, false, TensorLayout::NCHW); }
        if (c.residual) { add(c.shape, 83, -1.0f, 1.0f, padded, c.layout); }
        break;
    }
    case NormType::RMSNorm: {
        const std::vector<int64_t> sh = {c.scalar_scale ? 1 : norm_size};
        add(sh, 91, -1.0f, 1.0f, false, TensorLayout::NCHW);
        if (c.residual) { add(c.shape, 92, -1.0f, 1.0f, padded, c.layout); }
        break;
    }
    case NormType::GroupNorm: {
        add({C}, 95, -1.0f, 1.0f, false, TensorLayout::NCHW);
        if (c.bias) { add({C}, 96, -1.0f, 1.0f, false, TensorLayout::NCHW); }
        break;
    }
    case NormType::L2Norm:
        if (c.residual) { add(c.shape, 97, -1.0f, 1.0f, padded, c.layout); }
        break;
    }

    // `ins` is the call's view list; `aux` is the owner behind each slot (null
    // for the empty bias slot), which the device upload needs.
    std::vector<TensorView> ins;
    std::vector<const HostTensor*> aux;
    ins.push_back(x.view);
    const auto slot = [&](size_t i) {
        ins.push_back(own[i].view);
        aux.push_back(&own[i]);
    };
    switch (c.type) {
    case NormType::BatchNorm:
        for (size_t i = 0; i < own.size(); ++i) { slot(i); }
        break;
    case NormType::LayerNorm:
        // A residual needs the bias slot filled: check_views rejects an empty
        // view, so the cases pair the two.
        NNOPS_EXPECT_TRUE(!c.residual || c.bias);
        slot(0);
        if (c.bias) { slot(1); }
        if (c.residual) { slot(2); }
        break;
    case NormType::RMSNorm:
        slot(0);
        if (c.residual) { slot(1); }
        break;
    case NormType::GroupNorm:
        slot(0);
        if (c.bias) { slot(1); }
        break;
    case NormType::L2Norm:
        if (c.residual) { slot(0); }
        break;
    }

    std::vector<TensorDesc> in_descs;
    in_descs.reserve(ins.size());
    for (const auto& v : ins) { in_descs.push_back(v.desc()); }

    auto op_cpu = Norm::create(attrs, Backend::CPU);
    const std::vector<TensorDesc> outs =
        op_cpu->getOutputTensorDesc(in_descs);
    NNOPS_EXPECT_TRUE(!outs.empty());
    const TensorDesc od = outs.front();
    NNOPS_EXPECT_TRUE(od.layout == c.layout);
    NNOPS_EXPECT_TRUE(od.dtype == c.dtype);

    // add_to reads the destination, so both runs must start from the same bytes.
    const auto seed = [&](uint64_t s) {
        return c.add_to ? make_host_random(od.dims, od.dtype, padded, s, -1.0f, 1.0f,
                                           c.layout)
                        : make_host_zeroed(od.dims, od.dtype, padded, c.layout);
    };
    HostTensor out_ref = seed(41);
    HostTensor out_dev = seed(41);

    cpu_run(*op_cpu, out_ref.view, ins);

    auto op_dev = Norm::create(attrs, Backend::CUDA);
    DeviceTensor din = to_device(x);
    std::vector<DeviceTensor> daux;
    daux.reserve(aux.size());
    for (const HostTensor* h : aux) { daux.push_back(to_device(*h)); }
    DeviceTensor dout = alloc_like(out_dev.view, out_dev);

    std::vector<TensorView> dev_ins;
    dev_ins.push_back(din.view);
    for (auto& d : daux) { dev_ins.push_back(d.view); }
    cuda_run(*op_dev, dout.view, dev_ins);
    from_device(dout, out_dev);

    const auto [tol_abs, tol_rel] = tolerance(c.dtype);
    expect_same(out_dev, out_ref, tol_abs, tol_rel, what);
}

}  // namespace

// ============================================================
// LayerNorm
// ============================================================

NNOPS_TEST(cuda_layer_norm_tail) {
    NormCase c;
    c.shape = {2, 3, 16};
    run_case(c, "cuda_layer_norm_tail");
    c.bias = true;
    run_case(c, "cuda_layer_norm_tail_bias");
    c.bias = false;
    c.scalar_scale = true;
    run_case(c, "cuda_layer_norm_tail_scalar");
    c.scalar_scale = false;
    c.add_to = true;
    run_case(c, "cuda_layer_norm_tail_add_to");
}

NNOPS_TEST(cuda_layer_norm_inner_axis) {
    // An axis with dims on both sides: the pre group decodes a mixed radix.
    NormCase c;
    c.shape = {2, 5, 4, 16};
    c.dense = true;
    c.axis = 1;
    run_case(c, "cuda_layer_norm_inner_axis");
    c.axis = 0;
    run_case(c, "cuda_layer_norm_inner_axis_outer");
    c.axis = 2;
    run_case(c, "cuda_layer_norm_inner_axis_mid");
}

NNOPS_TEST(cuda_layer_norm_residual) {
    // The residual slot trails the bias, so every residual case carries one.
    // Channel-last is left out on purpose: its tail is strided, which the CPU
    // route's fused `norm(x + r)` does not define (it asserts on it and, in a
    // release build, drops the residual).
    NormCase c;
    c.shape = {2, 3, 16};
    c.residual = true;
    c.bias = true;
    run_case(c, "cuda_layer_norm_residual");
    c.add_to = true;
    run_case(c, "cuda_layer_norm_residual_add_to");
}

NNOPS_TEST(cuda_layer_norm_channel_last) {
    NormCase c;
    c.shape = {2, 6, 5};
    c.layout = TensorLayout::NHWC;
    run_case(c, "cuda_layer_norm_channel_last");
    c.bias = true;
    run_case(c, "cuda_layer_norm_channel_last_bias");
    // Over the spatial dims as well as the channels.
    c.bias = false;
    c.axis = 1;
    run_case(c, "cuda_layer_norm_channel_last_inner");
}

NNOPS_TEST(cuda_layer_norm_f16) {
    NormCase c;
    c.shape = {2, 3, 16};
    c.dtype = DataType::f16;
    c.bias = true;
    run_case(c, "cuda_layer_norm_f16");
    c.layout = TensorLayout::NHWC;
    c.shape = {2, 4, 8};
    run_case(c, "cuda_layer_norm_f16_channel_last");
    // A one-element scale tensor: the row kernel broadcasts it, and the CPU's
    // SIMD path has to as well instead of loading a lane past its end.
    c.layout = TensorLayout::NCHW;
    c.bias = false;
    c.scalar_scale = true;
    run_case(c, "cuda_layer_norm_f16_scalar");
}

// ============================================================
// RMSNorm
// ============================================================

NNOPS_TEST(cuda_rms_norm_tail) {
    NormCase c;
    c.type = NormType::RMSNorm;
    c.shape = {3, 4, 24};
    run_case(c, "cuda_rms_norm_tail");
    c.scalar_scale = true;
    run_case(c, "cuda_rms_norm_tail_scalar");
    c.scalar_scale = false;
    c.add_to = true;
    run_case(c, "cuda_rms_norm_tail_add_to");
}

NNOPS_TEST(cuda_rms_norm_inner_axis) {
    NormCase c;
    c.type = NormType::RMSNorm;
    c.shape = {2, 6, 5};
    c.dense = true;
    c.axis = 1;
    run_case(c, "cuda_rms_norm_inner_axis");
}

NNOPS_TEST(cuda_rms_norm_residual) {
    NormCase c;
    c.type = NormType::RMSNorm;
    c.shape = {2, 4, 12};
    c.residual = true;
    run_case(c, "cuda_rms_norm_residual");
}

NNOPS_TEST(cuda_rms_norm_channel_last_and_f16) {
    NormCase c;
    c.type = NormType::RMSNorm;
    c.shape = {2, 7, 8};
    c.layout = TensorLayout::NHWC;
    run_case(c, "cuda_rms_norm_channel_last");
    c.dtype = DataType::f16;
    run_case(c, "cuda_rms_norm_f16");
}

// ============================================================
// L2Norm
// ============================================================

NNOPS_TEST(cuda_l2_norm) {
    NormCase c;
    c.type = NormType::L2Norm;
    c.shape = {3, 4, 24};
    run_case(c, "cuda_l2_norm");
    c.residual = true;
    run_case(c, "cuda_l2_norm_residual");
    c.residual = false;
    c.add_to = true;
    run_case(c, "cuda_l2_norm_add_to");
}

NNOPS_TEST(cuda_l2_norm_inner_axis) {
    NormCase c;
    c.type = NormType::L2Norm;
    c.shape = {2, 6, 5};
    c.dense = true;
    c.axis = 1;
    run_case(c, "cuda_l2_norm_inner_axis");
    c.layout = TensorLayout::NHWC;
    c.axis = 2;
    run_case(c, "cuda_l2_norm_inner_axis_channel_last");
}

// ============================================================
// GroupNorm
// ============================================================

NNOPS_TEST(cuda_group_norm_channel_first) {
    NormCase c;
    c.type = NormType::GroupNorm;
    c.shape = {2, 8, 5, 6};
    c.groups = 4;
    run_case(c, "cuda_group_norm_channel_first");
    c.bias = true;
    run_case(c, "cuda_group_norm_channel_first_bias");
    c.bias = false;
    c.add_to = true;
    run_case(c, "cuda_group_norm_channel_first_add_to");
}

NNOPS_TEST(cuda_group_norm_group_shapes) {
    NormCase c;
    c.type = NormType::GroupNorm;
    // One channel per group is instance norm; one group over all of them is the
    // layer-norm-like extreme.
    c.shape = {2, 4, 3, 5};
    c.groups = 4;
    run_case(c, "cuda_group_norm_instance");
    c.shape = {2, 6, 2, 3};
    c.groups = 1;
    run_case(c, "cuda_group_norm_single");
    c.shape = {3, 6, 7};
    c.groups = 3;
    run_case(c, "cuda_group_norm_rank3");
}

NNOPS_TEST(cuda_group_norm_channel_last_and_f16) {
    NormCase c;
    c.type = NormType::GroupNorm;
    c.shape = {2, 8, 5, 6};   // dims stay [N][C][spatial] in either layout
    c.layout = TensorLayout::NHWC;
    c.groups = 4;
    run_case(c, "cuda_group_norm_channel_last");
    c.dtype = DataType::f16;
    c.bias = true;
    run_case(c, "cuda_group_norm_f16");
}

// ============================================================
// BatchNorm
// ============================================================

NNOPS_TEST(cuda_batch_norm_spatial) {
    NormCase c;
    c.type = NormType::BatchNorm;
    c.shape = {2, 3, 7, 9};   // plane 63: the block-per-plane route
    run_case(c, "cuda_batch_norm_spatial");
    c.add_to = true;
    run_case(c, "cuda_batch_norm_spatial_add_to");
}

NNOPS_TEST(cuda_batch_norm_small_plane) {
    NormCase c;
    c.type = NormType::BatchNorm;
    c.shape = {2, 4, 2, 3};   // plane 6: a thread per plane
    run_case(c, "cuda_batch_norm_small_plane");
    c.shape = {3, 5};         // rank two: the channel is the row
    run_case(c, "cuda_batch_norm_rank2");
    c.shape = {6};            // rank one: one implicit channel
    run_case(c, "cuda_batch_norm_rank1");
}

NNOPS_TEST(cuda_batch_norm_channel_last) {
    NormCase c;
    c.type = NormType::BatchNorm;
    c.shape = {2, 5, 4, 3};
    c.layout = TensorLayout::NHWC;
    run_case(c, "cuda_batch_norm_channel_last");
}

NNOPS_TEST(cuda_batch_norm_nonspatial) {
    NormCase c;
    c.type = NormType::BatchNorm;
    c.shape = {2, 3, 4, 5};
    c.spatial = false;
    c.dense = true;
    run_case(c, "cuda_batch_norm_nonspatial");
    c.add_to = true;
    run_case(c, "cuda_batch_norm_nonspatial_add_to");
}

NNOPS_TEST(cuda_batch_norm_f16) {
    NormCase c;
    c.type = NormType::BatchNorm;
    c.shape = {2, 3, 6, 5};
    c.dtype = DataType::f16;
    run_case(c, "cuda_batch_norm_f16");
    c.shape = {2, 8, 3};
    c.layout = TensorLayout::NHWC;
    run_case(c, "cuda_batch_norm_f16_channel_last");
}

#endif  // NNOPS_HAS_CUDA
