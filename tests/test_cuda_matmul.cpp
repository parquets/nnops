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
/// operand), f32 and f16, a grid deep enough for the wide f16 tile, batched
/// broadcasting, padded row pitches — the
/// padded case is where a stray 16 B vector load of a row tail would corrupt
/// the result — and the two front-end ceilings, rank six and a flattened batch
/// that fits blockIdx.z.
///
/// The int8 block (u8|s8 × s8) shares that CPU-oracle shape: the accumulate is
/// integer, so every integer destination is compared bit for bit, per-token and
/// per-channel zero points included; the dequantizing destinations use the fp
/// tolerance. Its QuantParams buffers are host arrays, which the backend stages
/// for the device — the same contract the CPU route reads.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/matmul.hpp"
#include "detail/bench_knobs.hpp"
#include "nnops/detail/half.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/cuda_test_helper.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
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

NNOPS_TEST(cuda_matmul_f16_paired_tile) {
    // The f16 launcher pairs two 64-wide tiles into one 256-thread block only on
    // a grid of at least one full wave *and* a long k loop, so this is the
    // smallest pair of geometries that reaches it: one that divides both tile
    // axes, one where both overhang (the N tail is the paired CTA's own risk —
    // its odd group has no tile at all, and the second B panel is masked). The
    // oracle is the CPU route, which is why these two cases cost seconds, not
    // milliseconds.
    const int64_t shapes[][2] = {{2048, 4096}, {2057, 4103}};
    bool transpose_b = false;
    for (const auto& s : shapes) {
        MatmulCase c;
        c.dtype = DataType::f16;
        c.transpose_b = transpose_b;
        c.a_shape = {s[0], 4096};
        c.b_shape = c.transpose_b ? std::vector<int64_t>{s[1], 4096}
                                  : std::vector<int64_t>{4096, s[1]};
        run_case(c, "cuda_matmul_f16_paired_tile");
        transpose_b = true;  // the second B layout, on the tail geometry
    }
}

NNOPS_TEST(cuda_matmul_splitk) {
    // Grids one or two tiles deep against a long reduction: the launcher cuts k
    // and reduces the planes, so these are the shapes that reach that pass.
    // The M and N tails matter here as much as anywhere — each split writes only
    // its in-bounds tile — and both dtypes take the cut.
    const int64_t shapes[][2] = {{1, 512}, {16, 256}, {33, 17}, {65, 130}};
    for (const auto& s : shapes) {
        for (DataType dt : {DataType::f32, DataType::f16}) {
            MatmulCase c;
            c.dtype = dt;
            c.a_shape = {s[0], 2048};
            c.b_shape = {2048, s[1]};
            run_case(c, "cuda_matmul_splitk");
        }
    }

    // The epilogue runs in the reduce pass on the summed planes, with beta
    // reading the destination there rather than in the split blocks.
    MatmulCase c;
    c.a_shape = {8, 4096};
    c.b_shape = {4096, 128};
    c.beta = 0.5f;
    c.use_bias = true;
    c.epilogue.type = EpilogueActivateType::Relu;
    run_case(c, "cuda_matmul_splitk_epilogue");

    // The batch axis multiplies the grid while the split axis rides on it:
    // blockIdx.z carries batch·splits + split, so this pins that arithmetic.
    MatmulCase b;
    b.a_shape = {2, 8, 2048};
    b.b_shape = {2, 2048, 128};
    run_case(b, "cuda_matmul_splitk_batched");
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

NNOPS_TEST(cuda_matmul_f32_simt_poses) {
    // All four transpose poses through the hand-written f32 tile: each pose is
    // its own panel order and its own FMA nest.
    for (int flags = 0; flags < 4; ++flags) {
        for (const auto& s : {std::pair<int64_t, int64_t>{256, 384},
                             std::pair<int64_t, int64_t>{517, 263}}) {
            MatmulCase c;
            c.transpose_a = (flags & 1) != 0;
            c.transpose_b = (flags & 2) != 0;
            c.a_shape = c.transpose_a ? std::vector<int64_t>{96, s.first}
                                      : std::vector<int64_t>{s.first, 96};
            c.b_shape = c.transpose_b ? std::vector<int64_t>{s.second, 96}
                                      : std::vector<int64_t>{96, s.second};
            run_case(c, "cuda_matmul_f32_simt_poses");
        }
    }
}

NNOPS_TEST(cuda_matmul_epilogue_forced) {
    // The operator epilogue (alpha, bias, beta, clamp, activation) runs inside
    // the hand-written kernels for both dtypes, at shapes the grid gate takes.
    // The narrow cases below put it on the same kernels from the other side.
    // bf16 has no CPU route on x86-64, so the oracle would be missing;
    // its epilogue path is covered by the bf16 suite's own arms.
    for (DataType dt : {DataType::f32, DataType::f16}) {
        MatmulCase c;
        c.dtype = dt;
        c.a_shape = {512, 96};
        c.b_shape = {96, 640};
        c.alpha = 0.5f;
        c.beta = 0.25f;
        c.use_bias = true;
        c.epilogue.type = EpilogueActivateType::Relu;
        run_case(c, "cuda_matmul_epilogue_forced");

        // The k-major pose takes the other epilogue path in the f16 kernel
        // (straight out of the accumulators), so it gets its own case.
        MatmulCase t;
        t.dtype = dt;
        t.transpose_b = true;
        t.a_shape = {512, 96};
        t.b_shape = {640, 96};
        t.alpha = 0.5f;
        t.beta = 0.25f;
        t.use_bias = true;
        t.epilogue.type = EpilogueActivateType::Relu;
        run_case(t, "cuda_matmul_epilogue_forced_tn");

        MatmulCase g;
        g.dtype = dt;
        g.a_shape = {517, 263};
        g.b_shape = {263, 129};
        g.use_bias = true;
        g.epilogue.type = EpilogueActivateType::Gelu;
        run_case(g, "cuda_matmul_epilogue_forced_gelu");
    }
}

NNOPS_TEST(cuda_matmul_f16_poses_forced) {
    // All four transpose poses through the f16 PTX kernel. The A-transposed pair
    // keeps A in its `[k][m]` arrival order and reads it with ldmatrix.trans, so
    // each pose is its own panel order and fragment shuffle.
    for (int flags = 0; flags < 4; ++flags) {
        for (const auto& s : {std::pair<int64_t, int64_t>{256, 384},
                             std::pair<int64_t, int64_t>{517, 263}}) {
            MatmulCase c;
            c.dtype = DataType::f16;
            c.transpose_a = (flags & 1) != 0;
            c.transpose_b = (flags & 2) != 0;
            c.a_shape = c.transpose_a ? std::vector<int64_t>{96, s.first}
                                      : std::vector<int64_t>{s.first, 96};
            c.b_shape = c.transpose_b ? std::vector<int64_t>{s.second, 96}
                                      : std::vector<int64_t>{96, s.second};
            run_case(c, "cuda_matmul_f16_poses_forced");
        }
    }
}

NNOPS_TEST(cuda_matmul_batch_forced) {
    // f16/bf16 batches fold into blockIdx.z beside the k splits, so a batched
    // product is the same kernel.
    for (DataType dt : {DataType::f32, DataType::f16}) {
        {
            MatmulCase c;
            c.dtype = dt;
            c.a_shape = {3, 640, 96};
            c.b_shape = {96, 320};
            run_case(c, "cuda_matmul_batch_forced_broadcast");
        }
        {
            MatmulCase c;
            c.dtype = dt;
            c.a_shape = {3, 512, 96};
            c.b_shape = {3, 96, 320};
            run_case(c, "cuda_matmul_batch_forced_both");
        }
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

NNOPS_TEST(cuda_matmul_wide_n) {
    // N past the once-shipped 65535 cap. The f32 tile walks n on blockIdx.y, so
    // the real bound is the tile count: 70000 columns is 547 tiles.
    MatmulCase c;
    c.a_shape = {3, 8};
    c.b_shape = {8, 70000};
    run_case(c, "cuda_matmul_wide_n");
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

NNOPS_TEST(cuda_matmul_i8_rejects_unsupported) {
    // The fp-activation × s8 pair (quantize-on-the-fly) has no CUDA route: B's
    // dtype and A's disagree, and the fp engine refuses the mismatch.
    const int64_t ashape[] = {4, 8};
    const int64_t bshape[] = {8, 4};
    std::vector<float> a(32, 1.0f);
    std::vector<int8_t> b(32, 1);
    std::vector<float> a_s(4, 0.01f);
    std::vector<int32_t> a_zp(4, 0);
    QuantParams qa;
    qa.scale = 0.01f;
    qa.zero_point = 3;
    qa.granularity = QuantGranularity::PerToken;
    qa.scale_data = a_s.data();
    qa.zero_point_data = a_zp.data();
    qa.num_scales = 4;

    TensorView av(ashape, DataType::f32, a.data(), TensorLayout::Dense, qa);
    TensorView bv(bshape, DataType::s8, b.data());
    const int64_t cshape[] = {4, 4};
    std::vector<float> out(16, 0.0f);
    TensorView cv(cshape, DataType::f32, out.data());

    auto op = MatMul::create(MatMulAttributes{}, Backend::CUDA);
    const TensorView ins[] = {av, bv};
    NNOPS_EXPECT_EQ(op->compute({&cv, 1}, ins, cuda_context(), nullptr),
                    Status::unsupported);
}

// ============================================================
// int8 — MatMulInteger semantics (u8|s8 × s8)
// ============================================================

namespace {

int64_t numel_of(const std::vector<int64_t>& s)
{
    int64_t n = 1;
    for (auto d : s) { n *= d; }
    return n;
}

TensorView view_of(const std::vector<int64_t>& shape, DataType dt, void* data,
                   const QuantParams& qp = {})
{
    return TensorView(std::span<const int64_t>(shape.data(), shape.size()), dt,
                      data, TensorLayout::Dense, qp);
}

struct I8Case {
    std::vector<int64_t> ashape, bshape;
    DataType adt = DataType::s8;
    DataType odt = DataType::s32;
    bool ta = false, tb = false;
    bool bias = false;
    bool relu = false;
    bool per_token = false;
    bool per_channel = false;
    uint64_t seed = 3;
};

/// One int8 call through both backends. Integer destinations are exact by
/// construction (the accumulate is int32), so they compare byte for byte; the
/// dequantizing destinations get the fp tolerance.
void run_i8_case(const I8Case& c, const char* what)
{
    const int64_t ar = static_cast<int64_t>(c.ashape.size());
    const int64_t br = static_cast<int64_t>(c.bshape.size());
    const int64_t M = c.ta ? c.ashape[ar - 1] : c.ashape[ar - 2];
    const int64_t K = c.ta ? c.ashape[ar - 2] : c.ashape[ar - 1];
    const int64_t N = c.tb ? c.bshape[br - 2] : c.bshape[br - 1];
    NNOPS_EXPECT_EQ(K, c.tb ? c.bshape[br - 1] : c.bshape[br - 2]);

    test::XorShift128 rng(c.seed);
    std::vector<uint8_t> a(static_cast<size_t>(numel_of(c.ashape)));
    for (auto& e : a) { e = static_cast<uint8_t>(rng.next_u64() & 0xFFu); }
    std::vector<int8_t> b(static_cast<size_t>(numel_of(c.bshape)));
    for (auto& e : b) {
        e = static_cast<int8_t>(int32_t(rng.next_u64() % 255) - 127);
    }

    std::vector<float> a_s(static_cast<size_t>(M));
    std::vector<int32_t> a_zp(static_cast<size_t>(M));
    std::vector<float> b_s(static_cast<size_t>(N));
    std::vector<int32_t> b_zp(static_cast<size_t>(N));
    for (int64_t i = 0; i < M; ++i) {
        a_s[static_cast<size_t>(i)] = 0.01f + 0.002f * float(i % 7);
        a_zp[static_cast<size_t>(i)] = int32_t(i % 9) - 4;
    }
    for (int64_t i = 0; i < N; ++i) {
        b_s[static_cast<size_t>(i)] = 0.004f + 0.001f * float(i % 5);
        b_zp[static_cast<size_t>(i)] = int32_t(i % 7) - 3;
    }

    QuantParams qa;
    qa.scale = 0.01f;
    qa.zero_point = c.adt == DataType::u8 ? 128 : -5;
    if (c.per_token) {
        qa.granularity = QuantGranularity::PerToken;
        qa.scale_data = a_s.data();
        qa.zero_point_data = a_zp.data();
        qa.num_scales = M;
    }
    QuantParams qb;
    qb.scale = 0.004f;
    qb.zero_point = 3;
    if (c.per_channel) {
        qb.granularity = QuantGranularity::PerChannel;
        qb.scale_data = b_s.data();
        qb.zero_point_data = b_zp.data();
        qb.num_scales = N;
    }
    QuantParams qc;
    qc.scale = 0.02f;
    qc.zero_point = c.odt == DataType::u8 ? 120 : 0;

    const bool int_out = c.odt == DataType::s32 || c.odt == DataType::s8 ||
                         c.odt == DataType::u8;
    const DataType bdt = int_out ? DataType::s32 : c.odt;
    std::vector<uint8_t> bias_bytes;
    if (c.bias) {
        bias_bytes.resize(static_cast<size_t>(N) * size_t(data_type_size(bdt)));
        for (int64_t i = 0; i < N; ++i) {
            if (bdt == DataType::s32) {
                reinterpret_cast<int32_t*>(bias_bytes.data())[i] =
                    int32_t(rng.next_u64() % 3001) - 1500;
            } else if (bdt == DataType::f16) {
                reinterpret_cast<nnops::backend::cpu::half*>(bias_bytes.data())[i] =
                    nnops::backend::cpu::float_to_half(
                        rng.next_float(-0.5f, 0.5f));
            } else {
                reinterpret_cast<float*>(bias_bytes.data())[i] =
                    rng.next_float(-0.5f, 0.5f);
            }
        }
    }

    MatMulAttributes attrs;
    attrs.transpose_a = c.ta;
    attrs.transpose_b = c.tb;
    attrs.output_dtype = c.odt;
    if (c.relu) { attrs.epilogue.type = EpilogueActivateType::Relu; }

    TensorView av = view_of(c.ashape, c.adt, a.data(), qa);
    TensorView bv = view_of(c.bshape, DataType::s8, b.data(), qb);
    TensorView biasv;
    std::vector<TensorView> ins = {av, bv};
    std::vector<TensorDesc> descs = {av.desc(), bv.desc()};
    if (c.bias) {
        biasv = view_of({N}, bdt, bias_bytes.data());
        ins.push_back(biasv);
        descs.push_back(biasv.desc());
    }

    auto op_cpu = MatMul::create(attrs, Backend::CPU);
    const TensorDesc od = op_cpu->getOutputTensorDesc(descs).front();
    const std::vector<int64_t> odims(od.dims.begin(),
                                     od.dims.begin() + od.rank);
    const size_t obytes = static_cast<size_t>(numel_of(odims)) *
                          size_t(data_type_size(od.dtype));

    std::vector<uint8_t> out_cpu(obytes, 0);
    std::vector<uint8_t> out_dev(obytes, 0);
    TensorView cv = view_of(odims, od.dtype, out_cpu.data(), qc);
    cpu_run(*op_cpu, cv, ins);

    CudaBuffer da(static_cast<size_t>(a.size()));
    CudaBuffer db(static_cast<size_t>(b.size()));
    da.upload(a.data(), a.size());
    db.upload(b.data(), b.size());
    TensorView dav = view_of(c.ashape, c.adt, da.data(), qa);
    TensorView dbv = view_of(c.bshape, DataType::s8, db.data(), qb);

    CudaBuffer dbias;
    std::vector<TensorView> dins = {dav, dbv};
    if (c.bias) {
        dbias = CudaBuffer(bias_bytes.size());
        dbias.upload(bias_bytes.data(), bias_bytes.size());
        dins.push_back(view_of({N}, bdt, dbias.data()));
    }

    CudaBuffer dout(obytes);
    dout.zero();
    TensorView dcv = view_of(odims, od.dtype, dout.data(), qc);
    auto op_dev = MatMul::create(attrs, Backend::CUDA);
    cuda_run(*op_dev, dcv, dins);
    dout.download(out_dev.data(), obytes);

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
            char msg[256];
            auto val = [&](const std::vector<uint8_t>& buf) {
                if (od.dtype == DataType::s8) {
                    return int(reinterpret_cast<const int8_t*>(buf.data())[bad]);
                }
                if (od.dtype == DataType::u8) {
                    return int(buf[bad]);
                }
                return int(reinterpret_cast<const int32_t*>(buf.data())[bad]);
            };
            std::snprintf(msg, sizeof(msg),
                          "%s: a=(%lld,%lld) ta=%d tb=%d out=%d elem %zu (of %zu) "
                          "cpu=%d cuda=%d",
                          what, (long long)c.ashape[0],
                          (long long)c.ashape.back(), int(c.ta), int(c.tb),
                          int(c.odt), bad, n, val(out_cpu), val(out_dev));
            throw std::runtime_error(msg);
        }
        return;
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

}  // namespace

NNOPS_TEST(cuda_matmul_i8_s32) {
    // The raw MatMulInteger destination: tails on both axes, both activation
    // storages, the bias and the Relu stage.
    { I8Case c; c.ashape = {37, 65}; c.bshape = {65, 43};
      c.adt = DataType::u8; c.bias = true; c.relu = true; c.seed = 11;
      run_i8_case(c, "cuda_matmul_i8_s32"); }
    { I8Case c; c.ashape = {37, 65}; c.bshape = {43, 65}; c.tb = true;
      c.adt = DataType::s8; c.bias = true; c.per_token = true;
      c.per_channel = true; c.seed = 13;
      run_i8_case(c, "cuda_matmul_i8_s32"); }
    // K inside one tile and past it; the vectorized path and the scalar gather
    // (B stored [K, N]) both run.
    { I8Case c; c.ashape = {16, 33}; c.bshape = {33, 9};
      c.adt = DataType::u8; c.seed = 17;
      run_i8_case(c, "cuda_matmul_i8_s32"); }
    { I8Case c; c.ashape = {64, 64}; c.bshape = {64, 64};
      c.adt = DataType::s8; c.seed = 19;
      run_i8_case(c, "cuda_matmul_i8_s32"); }
    // transpose_a stages A through its own fill.
    { I8Case c; c.ashape = {65, 37}; c.bshape = {65, 43}; c.ta = true;
      c.adt = DataType::u8; c.bias = true; c.seed = 23;
      run_i8_case(c, "cuda_matmul_i8_s32"); }
    // The 4-byte transpose fills with both axes ragged: B [K, N] whose row pitch
    // divides by four, and a transposed A whose row pitch does.
    { I8Case c; c.ashape = {70, 96}; c.bshape = {96, 40};
      c.adt = DataType::u8; c.seed = 29;
      run_i8_case(c, "cuda_matmul_i8_s32"); }
    { I8Case c; c.ashape = {96, 72}; c.bshape = {96, 41}; c.ta = true;
      c.adt = DataType::s8; c.bias = true; c.seed = 31;
      run_i8_case(c, "cuda_matmul_i8_s32"); }
}

NNOPS_TEST(cuda_matmul_i8_ptx_forced) {
    // The hand-written int8 tile (NN and NT: A row-major, B either k-major or
    // n-major). A [K, N] B is staged through the 4x4 byte transpose, which
    // needs a word-addressable pitch -- the odd-N cases below are there to
    // prove the route still answers when it is not.
    // Interior tiles: both fills take their 16 B vector path and the k loop
    // never sees a tail.
    { I8Case c; c.ashape = {128, 128}; c.bshape = {128, 128};
      c.adt = DataType::s8; c.seed = 301;
      run_i8_case(c, "cuda_matmul_i8_ptx_forced_nn"); }
    { I8Case c; c.ashape = {128, 128}; c.bshape = {128, 128}; c.tb = true;
      c.adt = DataType::u8; c.bias = true; c.relu = true; c.seed = 303;
      run_i8_case(c, "cuda_matmul_i8_ptx_forced_nt"); }
    // Per-token and per-channel zero points ride the same compensation.
    { I8Case c; c.ashape = {128, 192}; c.bshape = {128, 192}; c.tb = true;
      c.adt = DataType::s8; c.per_token = true; c.per_channel = true;
      c.bias = true; c.seed = 305;
      run_i8_case(c, "cuda_matmul_i8_ptx_forced_nt_per"); }
    // Ragged m and n tails past a whole tile, k across two tiles.
    { I8Case c; c.ashape = {70, 96}; c.bshape = {96, 40};
      c.adt = DataType::u8; c.bias = true; c.relu = true; c.seed = 307;
      run_i8_case(c, "cuda_matmul_i8_ptx_forced_nn_tail"); }
    { I8Case c; c.ashape = {100, 96}; c.bshape = {44, 96}; c.tb = true;
      c.adt = DataType::s8; c.per_channel = true; c.seed = 309;
      run_i8_case(c, "cuda_matmul_i8_ptx_forced_nt_tail"); }
    // K inside one tile, and a k tail that ends inside the 32-byte MMA step.
    { I8Case c; c.ashape = {37, 33}; c.bshape = {33, 40};
      c.adt = DataType::s8; c.bias = true; c.seed = 311;
      run_i8_case(c, "cuda_matmul_i8_ptx_forced_k_tail"); }
    { I8Case c; c.ashape = {64, 100}; c.bshape = {64, 100}; c.tb = true;
      c.adt = DataType::u8; c.relu = true; c.seed = 313;
      run_i8_case(c, "cuda_matmul_i8_ptx_forced_k_tail_nt"); }
    // TN and TT: A is stored [K, M], so its fill goes through the same byte 4x4
    // transpose the NN pose uses for B. The panel is k-major in every pose, so
    // the fragment map and the MMA do not change -- only this fill does.
    { I8Case c; c.ashape = {96, 100}; c.bshape = {44, 96}; c.ta = true;
      c.tb = true; c.adt = DataType::u8; c.bias = true; c.relu = true;
      c.seed = 317;
      run_i8_case(c, "cuda_matmul_i8_ptx_forced_tn"); }
    { I8Case c; c.ashape = {96, 64}; c.bshape = {96, 40}; c.ta = true;
      c.adt = DataType::s8; c.per_channel = true; c.seed = 319;
      run_i8_case(c, "cuda_matmul_i8_ptx_forced_tt"); }
    // Both axes ragged past a whole tile, k across two of them.
    { I8Case c; c.ashape = {130, 72}; c.bshape = {72, 130}; c.ta = true;
      c.tb = true; c.adt = DataType::s8; c.bias = true; c.seed = 321;
      run_i8_case(c, "cuda_matmul_i8_ptx_forced_tn_tail"); }
    // K inside one tile with A transposed: the fill zeroes its k tail.
    { I8Case c; c.ashape = {40, 64}; c.bshape = {40, 40}; c.ta = true;
      c.adt = DataType::u8; c.seed = 323;
      run_i8_case(c, "cuda_matmul_i8_ptx_forced_tt_ktail"); }

    // All four poses at sizes that leave a tail on both axes and a k that
    // crosses two tiles.
    { I8Case c; c.ashape = {150, 130}; c.bshape = {130, 68};
      c.adt = DataType::u8; c.bias = true; c.seed = 331;
      run_i8_case(c, "cuda_matmul_i8_ptx_rows_nn"); }
    { I8Case c; c.ashape = {150, 130}; c.bshape = {70, 130}; c.tb = true;
      c.adt = DataType::s8; c.per_channel = true; c.seed = 333;
      run_i8_case(c, "cuda_matmul_i8_ptx_rows_nt"); }
    { I8Case c; c.ashape = {130, 152}; c.bshape = {130, 40}; c.ta = true;
      c.adt = DataType::u8; c.bias = true; c.seed = 335;
      run_i8_case(c, "cuda_matmul_i8_ptx_rows_tn"); }
    { I8Case c; c.ashape = {130, 152}; c.bshape = {44, 130}; c.ta = true;
      c.tb = true; c.adt = DataType::s8; c.relu = true; c.seed = 337;
      run_i8_case(c, "cuda_matmul_i8_ptx_rows_tt"); }

    // The other four destinations, which were the CuTe engine's until now: the
    // s8/u8 requant and the f32/f16 dequantization, each with the bias its
    // format carries, over both B layouts.
    const DataType odts[] = {DataType::s8, DataType::u8, DataType::f32,
                             DataType::f16};
    uint64_t seed = 341;
    for (DataType odt : odts) {
        { I8Case c; c.ashape = {150, 130}; c.bshape = {130, 68}; c.odt = odt;
          c.adt = DataType::u8; c.bias = true; c.relu = true;
          c.per_token = true; c.per_channel = true; c.seed = seed++;
          run_i8_case(c, "cuda_matmul_i8_ptx_dst_nn"); }
        { I8Case c; c.ashape = {150, 130}; c.bshape = {70, 130}; c.odt = odt;
          c.tb = true; c.adt = DataType::s8; c.bias = true; c.seed = seed++;
          run_i8_case(c, "cuda_matmul_i8_ptx_dst_nt"); }
        // Tails on both axes, no bias, and the fp activation on the dequantized
        // path only (an integer destination applies Relu and nothing else).
        { I8Case c; c.ashape = {71, 96}; c.bshape = {96, 40}; c.odt = odt;
          c.seed = seed++;
          run_i8_case(c, "cuda_matmul_i8_ptx_dst_nn_tail"); }
        // An odd column count: the pitch refuses every pair store, so the scalar
        // fallback of each destination has to agree with the CPU too.
        { I8Case c; c.ashape = {150, 130}; c.bshape = {41, 130}; c.odt = odt;
          c.tb = true; c.adt = DataType::u8; c.bias = true; c.seed = seed++;
          run_i8_case(c, "cuda_matmul_i8_ptx_dst_nt_odd"); }
    }
}

NNOPS_TEST(cuda_matmul_i8_requant) {
    // s8/u8 destinations: per-token and per-channel scales feed the CPU's exact
    // round-away-from-zero requant, so these stay bit-exact.
    { I8Case c; c.ashape = {33, 17}; c.bshape = {17, 29};
      c.adt = DataType::s8; c.odt = DataType::u8; c.bias = true;
      c.per_token = true; c.per_channel = true; c.relu = true; c.seed = 31;
      run_i8_case(c, "cuda_matmul_i8_requant"); }
    { I8Case c; c.ashape = {3, 8, 40}; c.bshape = {40, 7};
      c.adt = DataType::u8; c.odt = DataType::s8; c.per_channel = true;
      c.seed = 37;
      run_i8_case(c, "cuda_matmul_i8_requant"); }
}

NNOPS_TEST(cuda_matmul_i8_decode) {
    // The narrow-m (decode) path: at most sixteen rows take the column-parallel
    // dp4a kernel, whose split-K planes are reduced with the same epilogue.
    // Both operand orientations and both activation storages.
    { I8Case c; c.ashape = {16, 1024}; c.bshape = {1024, 512};
      c.adt = DataType::u8; c.seed = 131;
      run_i8_case(c, "cuda_matmul_i8_decode"); }
    { I8Case c; c.ashape = {16, 2048}; c.bshape = {512, 2048}; c.tb = true;
      c.adt = DataType::s8; c.bias = true; c.seed = 137;
      run_i8_case(c, "cuda_matmul_i8_decode"); }
    { I8Case c; c.ashape = {1, 4096}; c.bshape = {4096, 128};
      c.adt = DataType::u8; c.seed = 139;
      run_i8_case(c, "cuda_matmul_i8_decode"); }
    { I8Case c; c.ashape = {16, 4096}; c.bshape = {128, 4096}; c.tb = true;
      c.adt = DataType::u8; c.odt = DataType::f32; c.per_token = true;
      c.per_channel = true; c.seed = 149;
      run_i8_case(c, "cuda_matmul_i8_decode"); }
    { I8Case c; c.ashape = {5, 1000}; c.bshape = {1000, 77};
      c.adt = DataType::s8; c.odt = DataType::s8; c.bias = true; c.relu = true;
      c.seed = 151;
      run_i8_case(c, "cuda_matmul_i8_decode"); }
}

NNOPS_TEST(cuda_matmul_i8_all_combos) {
    // The whole supported matrix: A u8|s8 × B s8, into every destination the
    // operator resolves (s32, s8, u8, f32, f16), with bias + Relu on each and
    // then again through a transposed, per-channel weight.
    const DataType adts[] = {DataType::u8, DataType::s8};
    const DataType odts[] = {DataType::s32, DataType::s8, DataType::u8,
                             DataType::f32, DataType::f16};
    uint64_t seed = 200;
    for (DataType adt : adts) {
        for (DataType odt : odts) {
            I8Case c;
            c.ashape = {23, 41};
            c.bshape = {41, 19};
            c.adt = adt;
            c.odt = odt;
            c.bias = true;
            c.relu = true;
            c.per_token = true;
            c.seed = seed++;
            run_i8_case(c, "cuda_matmul_i8_all_combos");

            I8Case t;
            t.ashape = {23, 41};
            t.bshape = {19, 41};
            t.tb = true;
            t.adt = adt;
            t.odt = odt;
            t.per_channel = true;
            t.seed = seed++;
            run_i8_case(t, "cuda_matmul_i8_all_combos");
        }
    }
}

NNOPS_TEST(cuda_matmul_i8_dequant) {
    // fp destinations dequantize with s_a·s_b, then clip and activate.
    { I8Case c; c.ashape = {21, 19}; c.bshape = {19, 33};
      c.adt = DataType::s8; c.odt = DataType::f32; c.bias = true;
      c.per_token = true; c.per_channel = true; c.relu = true; c.seed = 41;
      run_i8_case(c, "cuda_matmul_i8_dequant"); }
    { I8Case c; c.ashape = {2, 3, 5, 16}; c.bshape = {16, 11};
      c.adt = DataType::u8; c.odt = DataType::f16; c.seed = 43;
      run_i8_case(c, "cuda_matmul_i8_dequant"); }
    { I8Case c; c.ashape = {40, 32}; c.bshape = {48, 32}; c.tb = true;
      c.adt = DataType::u8; c.odt = DataType::f32; c.per_channel = true;
      c.seed = 47;
      run_i8_case(c, "cuda_matmul_i8_dequant"); }
}

/// w4a16 against the CPU route: an f16 activation and a llama.cpp block
/// weight, in both physical B layouts. The CPU packed route decodes the same
/// bytes, so it is the oracle for the kernel's own decode, the batch offset
/// (which is counted in blocks, not elements) and the epilogue.
void run_q4_case(const std::vector<int64_t>& ashape, const std::vector<int64_t>& bshape,
                 bool transpose_a, bool transpose_b, DataType fmt, float alpha, float beta,
                 Epilogue ep, bool bias, const char* what)
{
    MatMulAttributes attrs;
    attrs.transpose_a = transpose_a;
    attrs.transpose_b = transpose_b;
    attrs.alpha = alpha;
    attrs.beta = beta;
    attrs.epilogue = ep;

    HostTensor a = make_host_random(ashape, DataType::f16, false, 17);
    HostTensor b = make_host_q4(bshape, fmt, 29);
    HostTensor bias_t;
    if (bias) {
        // The bias is indexed by the column: N is B's row count when
        // transposed, its last extent otherwise.
        const int64_t n = transpose_b ? bshape[bshape.size() - 2] : bshape.back();
        bias_t = make_host_random(std::vector<int64_t>{n}, DataType::f16, false, 31, -0.25f, 0.25f);
    }

    std::vector<TensorDesc> descs = {a.view.desc(), b.view.desc()};
    if (bias) {
        descs.push_back(bias_t.view.desc());
    }

    auto op_cpu = MatMul::create(attrs, Backend::CPU);
    const TensorDesc od = op_cpu->getOutputTensorDesc(descs).front();
    const auto seed = [&](uint64_t s) {
        return beta != 0.0f ? make_host_random(od.dims, od.dtype, false, s, 0.0f, 0.5f) :
                              make_host_zeroed(od.dims, od.dtype, false);
    };
    HostTensor out_ref = seed(37);
    HostTensor out_dev = seed(37);

    std::vector<TensorView> cpu_ins = {a.view, b.view};
    if (bias) {
        cpu_ins.push_back(bias_t.view);
    }
    cpu_run(*op_cpu, out_ref.view, cpu_ins);

    auto op_dev = MatMul::create(attrs, Backend::CUDA);
    DeviceTensor da = to_device(a);
    DeviceTensor db = to_device(b);
    DeviceTensor dbias;
    std::vector<TensorView> dev_ins = {da.view, db.view};
    if (bias) {
        dbias = to_device(bias_t);
        dev_ins.push_back(dbias.view);
    }
    DeviceTensor dout = alloc_like(out_dev.view, out_dev);
    cuda_run(*op_dev, dout.view, dev_ins);
    from_device(dout, out_dev);

    expect_same(out_dev, out_ref, kF16AccumTol, 2e-2f, what);
}

NNOPS_TEST(cuda_matmul_q4) {
    Epilogue relu;
    relu.type = EpilogueActivateType::Relu;

    for (DataType fmt : {DataType::q4_0, DataType::q4_1, DataType::q8_0}) {
        // [M, K] x [N, K] with transpose_b, the Linear pose: one row, a k tail,
        // and a full tile against a wider N.
        run_q4_case({1, 96}, {64, 96}, false, true, fmt, 1.0f, 0.0f, Epilogue{}, false,
                    "cuda_matmul_q4 k-major");
        run_q4_case({5, 65}, {64, 65}, false, true, fmt, 1.0f, 0.0f, Epilogue{}, true,
                    "cuda_matmul_q4 k tail");
        run_q4_case({64, 128}, {96, 128}, false, true, fmt, 0.5f, 0.5f, relu, true,
                    "cuda_matmul_q4 alpha beta relu");
        // [M, K] x [K, N] without transpose_b: the pose whose block is picked by
        // the column instead of the row.
        run_q4_case({5, 65}, {65, 64}, false, false, fmt, 1.0f, 0.0f, Epilogue{}, true,
                    "cuda_matmul_q4 n-major");
        run_q4_case({64, 128}, {128, 96}, false, false, fmt, 1.0f, 0.0f, relu, false,
                    "cuda_matmul_q4 n-major tile");
    }

    // The narrow-m loop (a Linear pose at most sixteen rows deep), taken here
    // through MatMul's own front end.
    run_q4_case({4, 2048}, {80, 2048}, false, true, DataType::q4_1, 1.0f, 0.0f, Epilogue{},
                true, "cuda_matmul_q4 narrow");

    // transpose_a goes through the A panel's own copy atom.
    run_q4_case({65, 8}, {64, 65}, true, true, DataType::q4_1, 1.0f, 0.0f, Epilogue{}, false,
                "cuda_matmul_q4 transpose_a");

    // A batched B: each slice's block offset is its row count times the pitch,
    // counted in blocks.
    run_q4_case({2, 7, 64}, {2, 33, 64}, false, true, DataType::q4_1, 1.0f, 0.0f, Epilogue{},
                true, "cuda_matmul_q4 batched B");
    run_q4_case({3, 5, 64}, {64, 64}, false, true, DataType::q4_0, 1.0f, 0.0f, Epilogue{}, false,
                "cuda_matmul_q4 broadcast B");
}

NNOPS_TEST(cuda_matmul_rejects_rank_above_six) {
    // Four batch axes are all the kernels carry, so a rank-seven call is refused
    // by validate() instead of reaching the launch and failing there.
    const int64_t ashape[] = {2, 1, 1, 1, 1, 4, 8};
    const int64_t bshape[] = {8, 3};
    std::vector<float> a(2 * 4 * 8, 1.0f);
    std::vector<float> b(8 * 3, 1.0f);
    TensorView av(ashape, DataType::f32, a.data());
    TensorView bv(bshape, DataType::f32, b.data());
    const int64_t cshape[] = {2, 1, 1, 1, 1, 4, 3};
    std::vector<float> out(2 * 4 * 3, 0.0f);
    TensorView cv(cshape, DataType::f32, out.data());

    auto op = MatMul::create(MatMulAttributes{}, Backend::CUDA);
    const TensorView ins[] = {av, bv};
    NNOPS_EXPECT_EQ(op->compute({&cv, 1}, ins, cuda_context(), nullptr),
                    Status::unsupported);
}

NNOPS_TEST(cuda_matmul_rejects_oversized_batch) {
    // 256x256 batch slices: each axis is legal on its own, but the flattened
    // batch overflows gridDim.z, so the call has to come back unsupported
    // rather than as a launch error surfaced as internal.
    const int64_t ashape[] = {256, 256, 4, 8};
    const int64_t bshape[] = {8, 3};
    std::vector<float> a(256 * 256 * 4 * 8, 1.0f);
    std::vector<float> b(8 * 3, 1.0f);
    TensorView av(ashape, DataType::f32, a.data());
    TensorView bv(bshape, DataType::f32, b.data());
    const int64_t cshape[] = {256, 256, 4, 3};
    std::vector<float> out(256 * 256 * 4 * 3, 0.0f);
    TensorView cv(cshape, DataType::f32, out.data());

    auto op = MatMul::create(MatMulAttributes{}, Backend::CUDA);
    const TensorView ins[] = {av, bv};
    NNOPS_EXPECT_EQ(op->compute({&cv, 1}, ins, cuda_context(), nullptr),
                    Status::unsupported);
}

#endif  // NNOPS_HAS_CUDA
