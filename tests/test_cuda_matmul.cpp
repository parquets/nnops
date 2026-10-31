/// @file test_cuda_matmul.cpp
/// @brief CUDA MatMul against the CPU backend on the same inputs.
///
/// Every case builds one random host operand pair, runs the operator twice —
/// once with Backend::CPU and once with Backend::CUDA over device copies — and
/// compares the outputs. The oracle is the CPU route rather than a hand-written
/// reference so the epilogue contract (alpha on the product alone, the bias
/// between product and activation, clamp and activation on the finished sum,
/// beta times the old C) is tested exactly as the operator defines it.
///
/// Cases cover the shapes the kernel branches on: M/N edges (a 5x3x7 product
/// inside one 64x64 tile), a grid wide enough to select the 64x128 f32 tile,
/// a K that is not a multiple of the 16-element MMA
/// step, both transpose flags (each picks a different copy atom and LDSM
/// operand), f32 and f16, batched broadcasting, and padded row pitches — the
/// padded case is where a stray 16 B vector load of a row tail would corrupt
/// the result.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/matmul.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/cuda_test_helper.hpp"

#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

struct MatmulCase {
    std::vector<int64_t> a_shape;
    std::vector<int64_t> b_shape;
    bool transpose_a = false;
    bool transpose_b = false;
    float alpha = 1.0f;
    float beta = 0.0f;
    DataType dtype = DataType::f32;
    bool use_bias = false;
    bool padded = true;
    Epilogue epilogue{};
};

/// Tolerance pair for the dtype: f16 accumulates through tensor cores here and
/// through the CPU's widening FMA there, so it gets the shared f16 budget.
std::pair<float, float> tolerance(DataType dt)
{
    return dt == DataType::f16 ? std::pair<float, float>{kF16AccumTol, 2e-2f}
                               : std::pair<float, float>{1e-4f, 1e-4f};
}

void run_case(const MatmulCase& c, const char* what)
{
    MatMulAttributes attrs;
    attrs.transpose_a = c.transpose_a;
    attrs.transpose_b = c.transpose_b;
    attrs.alpha = c.alpha;
    attrs.beta = c.beta;
    attrs.epilogue = c.epilogue;

    HostTensor a = make_host_random(c.a_shape, c.dtype, c.padded, 17);
    HostTensor b = make_host_random(c.b_shape, c.dtype, c.padded, 29);
    HostTensor bias;
    if (c.use_bias) {
        const int64_t n = c.transpose_b ? c.b_shape[c.b_shape.size() - 2]
                                        : c.b_shape.back();
        bias = make_host_random(std::vector<int64_t>{n}, c.dtype, false, 31, -0.25f, 0.25f);
    }

    std::vector<TensorDesc> descs = {a.view.desc(), b.view.desc()};
    if (c.use_bias) { descs.push_back(bias.view.desc()); }

    auto op_cpu = MatMul::create(attrs, Backend::CPU);
    const std::vector<TensorDesc> outs = op_cpu->getOutputTensorDesc(descs);
    NNOPS_EXPECT_TRUE(!outs.empty());
    const TensorDesc od = outs.front();

    // beta != 0 reads the destination, so both runs must start from the same
    // bytes; identical seeds give identical initial content.
    const auto seed = [&](uint64_t s) {
        return c.beta != 0.0f
                   ? make_host_random(od.dims, od.dtype, c.padded, s)
                   : make_host_zeroed(od.dims, od.dtype, c.padded);
    };
    HostTensor out_ref = seed(41);
    HostTensor out_dev = seed(41);

    std::vector<TensorView> cpu_ins = {a.view, b.view};
    if (c.use_bias) { cpu_ins.push_back(bias.view); }
    cpu_run(*op_cpu, out_ref.view, cpu_ins);

    auto op_dev = MatMul::create(attrs, Backend::CUDA);
    DeviceTensor da = to_device(a);
    DeviceTensor db = to_device(b);
    DeviceTensor dbias;
    std::vector<TensorView> dev_ins = {da.view, db.view};
    if (c.use_bias) {
        dbias = to_device(bias);
        dev_ins.push_back(dbias.view);
    }
    DeviceTensor dout = alloc_like(out_dev.view, out_dev);
    cuda_run(*op_dev, dout.view, dev_ins);
    from_device(dout, out_dev);

    const auto [tol_abs, tol_rel] = tolerance(c.dtype);
    expect_same(out_dev, out_ref, tol_abs, tol_rel, what);
}

}  // namespace

NNOPS_TEST(cuda_matmul_f32_shapes) {
    const int64_t shapes[][2] = {{64, 64}, {65, 63}, {1, 1000}, {1000, 1},
                                 {5, 7},    {129, 250}, {33, 17}};
    for (const auto& s : shapes) {
        MatmulCase c;
        c.a_shape = {s[0], 96};
        c.b_shape = {96, s[1]};
        run_case(c, "cuda_matmul_f32_shapes");
    }
}

NNOPS_TEST(cuda_matmul_f32_wide_tile) {
    // Enough CTAs for the launcher to pick the 64x128 f32 tile; one geometry
    // that divides both tile axes, one where both overhang. Each transpose flag
    // pairs it with a different shared-memory layout and copy atom.
    const int64_t shapes[][2] = {{1024, 2048}, {1400, 1200}};
    for (const auto& s : shapes) {
        for (int flags = 0; flags < 4; ++flags) {
            MatmulCase c;
            c.transpose_a = (flags & 1) != 0;
            c.transpose_b = (flags & 2) != 0;
            c.a_shape = c.transpose_a ? std::vector<int64_t>{96, s[0]}
                                      : std::vector<int64_t>{s[0], 96};
            c.b_shape = c.transpose_b ? std::vector<int64_t>{s[1], 96}
                                      : std::vector<int64_t>{96, s[1]};
            run_case(c, "cuda_matmul_f32_wide_tile");
        }
    }
}

NNOPS_TEST(cuda_matmul_f32_ragged_k) {
    // K shorter than one 64-element k tile, and a k tile plus a partial one.
    for (int64_t k : {1, 7, 16, 17, 63, 64, 65, 130}) {
        MatmulCase c;
        c.a_shape = {40, k};
        c.b_shape = {k, 40};
        run_case(c, "cuda_matmul_f32_ragged_k");
    }
}

NNOPS_TEST(cuda_matmul_f16_shapes) {
    const int64_t shapes[][2] = {{64, 64}, {65, 63}, {5, 7}, {129, 250}};
    for (const auto& s : shapes) {
        MatmulCase c;
        c.a_shape = {s[0], 96};
        c.b_shape = {96, s[1]};
        c.dtype = DataType::f16;
        run_case(c, "cuda_matmul_f16_shapes");
    }
}

NNOPS_TEST(cuda_matmul_transposes) {
    // A is [M,K] or the transposed [K,M]; B is [K,N] or the transposed [N,K].
    for (int flags = 0; flags < 4; ++flags) {
        MatmulCase c;
        c.transpose_a = (flags & 1) != 0;
        c.transpose_b = (flags & 2) != 0;
        c.a_shape = c.transpose_a ? std::vector<int64_t>{48, 37}
                                  : std::vector<int64_t>{37, 48};
        c.b_shape = c.transpose_b ? std::vector<int64_t>{91, 48}
                                  : std::vector<int64_t>{48, 91};
        run_case(c, "cuda_matmul_transposes");
    }
}

NNOPS_TEST(cuda_matmul_transposes_f16) {
    for (int flags = 0; flags < 4; ++flags) {
        MatmulCase c;
        c.transpose_a = (flags & 1) != 0;
        c.transpose_b = (flags & 2) != 0;
        c.dtype = DataType::f16;
        c.a_shape = c.transpose_a ? std::vector<int64_t>{48, 37}
                                  : std::vector<int64_t>{37, 48};
        c.b_shape = c.transpose_b ? std::vector<int64_t>{91, 48}
                                  : std::vector<int64_t>{48, 91};
        run_case(c, "cuda_matmul_transposes_f16");
    }
}

NNOPS_TEST(cuda_matmul_batched_broadcast) {
    // A: [2, 3, 5, 40], B: [2, 1, 40, 7] — batch dims align from the right, so
    // B's middle 1 broadcasts over A's innermost batch axis.
    {
        MatmulCase c;
        c.a_shape = {2, 3, 5, 40};
        c.b_shape = {2, 1, 40, 7};
        run_case(c, "cuda_matmul_batched_broadcast");
    }
    // B broadcast over the batch: [2, 5, 40, 7] against [40, 7].
    {
        MatmulCase c;
        c.a_shape = {2, 5, 40};
        c.b_shape = {40, 7};
        run_case(c, "cuda_matmul_batched_broadcast");
    }
    // Both operands batched.
    {
        MatmulCase c;
        c.a_shape = {4, 2, 3, 40};
        c.b_shape = {4, 1, 40, 9};
        run_case(c, "cuda_matmul_batched_broadcast");
    }
}

NNOPS_TEST(cuda_matmul_epilogue) {
    auto make = [](float alpha, float beta, bool bias, Epilogue ep) {
        MatmulCase c;
        c.a_shape = {37, 65};
        c.b_shape = {65, 43};
        c.alpha = alpha;
        c.beta = beta;
        c.use_bias = bias;
        c.epilogue = ep;
        return c;
    };

    run_case(make(0.5f, 0.0f, true, Epilogue{}), "cuda_matmul_epilogue");
    run_case(make(1.0f, 1.0f, true, Epilogue{}), "cuda_matmul_epilogue");

    Epilogue relu6;
    relu6.type = EpilogueActivateType::Relu6;
    run_case(make(1.0f, 0.0f, true, relu6), "cuda_matmul_epilogue");

    Epilogue clamp;
    clamp.min_clip = -0.05f;
    clamp.max_clip = 0.08f;
    run_case(make(2.0f, 0.5f, true, clamp), "cuda_matmul_epilogue");

    Epilogue gelu;
    gelu.type = EpilogueActivateType::Gelu;
    run_case(make(1.0f, 0.0f, false, gelu), "cuda_matmul_epilogue");

    Epilogue act16;
    act16.type = EpilogueActivateType::Relu;
    MatmulCase c = make(1.0f, 0.0f, true, act16);
    c.dtype = DataType::f16;
    run_case(c, "cuda_matmul_epilogue");
}

NNOPS_TEST(cuda_matmul_dense_strides) {
    // Same geometry with tight rows: exercises the unpadded ldc/lda/ldb path.
    MatmulCase c;
    c.a_shape = {37, 65};
    c.b_shape = {65, 43};
    c.padded = false;
    c.use_bias = true;
    run_case(c, "cuda_matmul_dense_strides");
}

NNOPS_TEST(cuda_matmul_rejects_unimplemented) {
    // The integer pairs are accepted by the operator but not by the CUDA route.
    const int64_t ashape[] = {4, 8};
    const int64_t bshape[] = {8, 4};
    std::vector<int8_t> a(32, 1);
    std::vector<int8_t> b(32, 1);
    TensorView av(ashape, DataType::s8, a.data());
    TensorView bv(bshape, DataType::s8, b.data());
    const int64_t cshape[] = {4, 4};
    std::vector<int32_t> out(16, 0);
    TensorView cv(cshape, DataType::s32, out.data());

    auto op = MatMul::create(MatMulAttributes{}, Backend::CUDA);
    const TensorView ins[] = {av, bv};
    const Status st = op->compute({&cv, 1}, ins, cuda_context(), nullptr);
    NNOPS_EXPECT_EQ(st, Status::unsupported);
}

#endif  // NNOPS_HAS_CUDA
