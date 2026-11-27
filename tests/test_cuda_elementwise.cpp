/// @file test_cuda_elementwise.cpp
/// @brief CUDA Activation and Softmax against the CPU backend.
///
/// Both routes read the same bytes — the host tensor is uploaded once and the
/// CPU oracle runs on the host copy — so the comparison holds the device kernel
/// to the CPU's definition of the op, not to a re-derived one. Activation runs
/// the CPU's own SIMD approximations of the transcendentals, and softmax's block
/// summation tree cannot match a sequential row sum in the last bit, so both take
/// a tolerance rather than the byte compare the movement ops use.
///
/// The shapes are chosen for the 128-bit lanes the kernels use. An f32 row of 37
/// makes the row bases alternate off the 16 B bound, so half of them take the
/// scalar fallback; an f16 row of 37 with a 32 B pitch keeps the bases aligned
/// and exercises the tail instead. A 20 k-element row is past the shared-memory
/// budget, so softmax re-reads it from global memory. Softmax runs on NHWC.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/activation.hpp"
#include "nnops/ops/softmax.hpp"
#include "nnops/ops/eltwise.hpp"
#include "nnops/ops/unary.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/cuda_test_helper.hpp"

#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

constexpr float kF32Abs = 2e-4f;
constexpr float kF32Rel = 2e-4f;
constexpr float kF16Abs = 2e-2f;
constexpr float kF16Rel = 4e-2f;

void run_activation(ActivationType type, std::vector<int64_t> shape, DataType dt,
                    bool padded, TensorLayout layout = TensorLayout::NHWC,
                    bool add_to = false, const char* what = "")
{
    ActivationAttributes attrs;
    attrs.type = type;
    attrs.add_to = add_to;

    HostTensor in = make_host_random(shape, dt, padded, 17, -1.0f, 1.0f, layout);
    const HostTensor ins[] = {in};
    // The CPU route evaluates the transcendental ones with its own SIMD
    // approximation, so this is a tolerance compare, not a byte compare.
    const float tol_abs = dt == DataType::f16 ? kF16Abs : kF32Abs;
    const float tol_rel = dt == DataType::f16 ? kF16Rel : kF32Rel;
    run_op_close([&](Backend be) { return Activation::create(attrs, be); }, ins,
                 tol_abs, tol_rel, what);
}

void run_softmax(std::vector<int64_t> shape, int64_t axis, DataType dt,
                 bool padded, TensorLayout layout = TensorLayout::NHWC,
                 bool log_softmax = false, const char* what = "",
                 float tol_abs = 0.0f, float tol_rel = 0.0f,
                 float temperature = 1.0f)
{
    SoftmaxAttributes attrs;
    attrs.axis = axis;
    attrs.log_softmax = log_softmax;
    attrs.temperature = temperature;

    // NHWC views must stay dense: their strides never consult the pitch.
    const bool pad = padded && !is_channel_last(layout);
    HostTensor in = make_host_random(shape, dt, pad, 19, -4.0f, 4.0f, layout);
    const HostTensor ins[] = {in};
    run_op_close([&](Backend be) { return Softmax::create(attrs, be); }, ins,
                 tol_abs, tol_rel, what);
}

}  // namespace

NNOPS_TEST(cuda_activation_lanes_and_tail) {
    // f32 of 37: row bases 148 B apart, so odd rows are off the 16 B bound and
    // fall back to the scalar loop.
    run_activation(ActivationType::Relu, {64, 37}, DataType::f32, false, TensorLayout::NHWC,
                   false, "cuda_activation_relu_37");
    // The same width with the graph's 32 B pitch: every base is aligned and only
    // the tail (37 = 4·9 + 1) takes the scalar loop.
    run_activation(ActivationType::Silu, {64, 37}, DataType::f32, true, TensorLayout::NHWC,
                   false, "cuda_activation_silu_37_pitched");
    run_activation(ActivationType::Gelu, {8, 1024}, DataType::f32, true, TensorLayout::NHWC,
                   false, "cuda_activation_gelu");
    run_activation(ActivationType::Sigmoid, {8, 1024}, DataType::f32, true,
                   TensorLayout::NHWC, false, "cuda_activation_sigmoid");
    run_activation(ActivationType::HardSwish, {8, 1024}, DataType::f32, true,
                   TensorLayout::NHWC, false, "cuda_activation_hardswish");
    // A rank-1 tensor takes the flat kernel, whose grid is sized in lanes.
    run_activation(ActivationType::Relu, {100003}, DataType::f32, false, TensorLayout::Dense,
                   false, "cuda_activation_flat_odd");
    run_activation(ActivationType::Silu, {100003}, DataType::f16, false, TensorLayout::Dense,
                   false, "cuda_activation_flat_odd_f16");
}

NNOPS_TEST(cuda_activation_f16_and_add_to) {
    // f16 lanes are 8 wide, so a 37-wide row leaves 5 elements on the tail.
    run_activation(ActivationType::Elu, {64, 37}, DataType::f16, true, TensorLayout::NHWC,
                   false, "cuda_activation_elu_37_pitched");
    run_activation(ActivationType::LeakyRelu, {8, 1024}, DataType::f16, true,
                   TensorLayout::NHWC, false, "cuda_activation_leaky_f16");
    run_activation(ActivationType::Silu, {8, 1024}, DataType::f16, true, TensorLayout::NHWC,
                   true, "cuda_activation_silu_add_to");
    // NHWC rows are one spatial line of W·C, which is the last dim the
    // kernel is handed.
    run_activation(ActivationType::Relu, {2, 7, 5, 8}, DataType::f32, false,
                   TensorLayout::NHWC, false, "cuda_activation");
}

NNOPS_TEST(cuda_softmax_contiguous) {
    // A float row of 4 KB is held in shared memory for the later passes.
    run_softmax({128, 1024}, -1, DataType::f32, true, TensorLayout::NHWC, false,
                "cuda_softmax_1024", kF32Abs, kF32Rel);
    // 37 = 4·9 + 1: an unaligned row on half the pitches, a tail on all of them.
    run_softmax({64, 37}, -1, DataType::f32, false, TensorLayout::NHWC, false,
                "cuda_softmax_37_dense", kF32Abs, kF32Rel);
    run_softmax({64, 37}, -1, DataType::f32, true, TensorLayout::NHWC, true,
                "cuda_softmax_log_37_pitched", kF32Abs, kF32Rel);
    run_softmax({8, 8192}, -1, DataType::f32, true, TensorLayout::NHWC, false,
                "cuda_softmax_8192", kF32Abs, kF32Rel);
    // 20000 floats is 80 KB: past the budget, so the row is read again instead
    // of being kept.
    run_softmax({4, 20000}, -1, DataType::f32, false, TensorLayout::NHWC, false,
                "cuda_softmax_20000", kF32Abs, kF32Rel);
    run_softmax({4, 20000}, -1, DataType::f32, false, TensorLayout::NHWC, true,
                "cuda_softmax_log_20000", kF32Abs, kF32Rel);
}

NNOPS_TEST(cuda_softmax_and_f16) {
    // NHWC with the axis on C (the innermost dim of [N, H, W, C]): the groups
    // are the layout's own rows.
    run_softmax({1, 16, 16, 8}, 3, DataType::f32, false, TensorLayout::NHWC,
                false, "cuda_softmax", kF32Abs, kF32Rel);
    run_softmax({128, 1024}, -1, DataType::f16, true, TensorLayout::NHWC, false,
                "cuda_softmax_f16", kF16Abs, kF16Rel);
    run_softmax({64, 37}, -1, DataType::f16, true, TensorLayout::NHWC, true,
                "cuda_softmax_log_f16_37", kF16Abs, kF16Rel);
}

NNOPS_TEST(cuda_softmax_temperature) {
    // The device route divides the shifted logits by T exactly where the CPU does;
    // until this was pinned it ignored the attribute and returned T = 1 results.
    run_softmax({64, 1000}, -1, DataType::f32, true, TensorLayout::NHWC, false,
                "cuda_softmax_t_half", kF32Abs, kF32Rel, 0.5f);
    run_softmax({64, 1000}, -1, DataType::f32, true, TensorLayout::NHWC, true,
                "cuda_softmax_t_two_log", kF32Abs, kF32Rel, 2.0f);
}

NNOPS_TEST(cuda_eltwise_broadcast) {
    // The operator broadcasts under the NumPy rules (`Eltwise::validate` says
    // so); the CUDA kernels used to assume both operands had the output's
    // shape, so a bias add read past the short operand.
    const auto run = [](EltwiseType type, std::vector<int64_t> as, std::vector<int64_t> bs,
                        bool add_to, const char* what, bool padded_out = false) {
        EltwiseAttributes attrs;
        attrs.type = type;
        attrs.add_to = add_to;
        HostTensor a = make_host_random(as, DataType::f32, false, 31, -1.0f, 1.0f);
        HostTensor b = make_host_random(bs, DataType::f32, false, 32, -1.0f, 1.0f);
        const HostTensor ins[] = {a, b};
        run_op_close([&](Backend be) { return Eltwise::create(attrs, be); }, ins, 1e-5f, 1e-5f,
                     what, padded_out);
    };
    // Trailing-axis broadcast: the bias add a graph produces.
    run(EltwiseType::Add, {64, 128}, {128}, false, "eltwise broadcast row");
    // A leading broadcast axis, and the same axis twice.
    run(EltwiseType::Mul, {4, 8, 16}, {4, 1, 16}, false, "eltwise broadcast mid");
    run(EltwiseType::Sub, {1, 8, 16}, {6, 8, 16}, false, "eltwise broadcast lead");
    // The same run accumulating into the old output.
    run(EltwiseType::Add, {8, 32}, {32}, true, "eltwise broadcast add_to");
    // A row that is not a whole 32 B unit, so the output is pitched: the row
    // walk has to read the destination through its own stride, not the flat
    // index.
    run(EltwiseType::Add, {2, 3, 5, 13}, {13}, false, "eltwise broadcast short row", true);
    run(EltwiseType::Mul, {2, 3, 5, 13}, {2, 1, 5, 13}, true, "eltwise broadcast short row add_to",
        true);
    // f16 takes the same walk.
    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Add;
    HostTensor a = make_host_random(std::vector<int64_t>{16, 24}, DataType::f16, false, 33,
                                    -1.0f, 1.0f);
    HostTensor b = make_host_random(std::vector<int64_t>{24}, DataType::f16, false, 34,
                                    -1.0f, 1.0f);
    const HostTensor ins[] = {a, b};
    run_op_close([&](Backend be) { return Eltwise::create(attrs, be); }, ins, 2e-3f, 2e-3f,
                 "eltwise broadcast f16");
}

// ============================================================
// Quantized routes — dequantize, op, requantize in one pass
// ============================================================
//
// The views carry QuantParams on both backends. A per-token pair is compared
// within one integer step: the CPU evaluates the transcendentals with its own
// SIMD approximations, and its dequantize falls to a non-FMA scalar tail when a
// row is not a whole vector multiple.

namespace {

QuantParams per_tensor(float scale, int32_t zp)
{
    QuantParams qp;
    qp.scale = scale;
    qp.zero_point = zp;
    return qp;
}

/// Random per-row parameters over `rows`; the caller binds them to a view.
void fill_per_token(int64_t rows, uint64_t seed, std::vector<float>& scale,
                    std::vector<int32_t>& zp, QuantParams& qp)
{
    scale.resize(static_cast<size_t>(rows));
    zp.resize(static_cast<size_t>(rows));
    XorShift128 rng(seed);
    for (int64_t r = 0; r < rows; ++r) {
        scale[static_cast<size_t>(r)] = rng.next_float(0.01f, 0.05f);
        zp[static_cast<size_t>(r)] = static_cast<int32_t>(rng.next_u64() % 9) - 4;
    }
    qp.granularity = QuantGranularity::PerToken;
    qp.scale_data = scale.data();
    qp.zero_point_data = zp.data();
    qp.num_scales = rows;
}

}  // namespace

NNOPS_TEST(cuda_activation_quant_per_tensor) {
    const std::vector<int64_t> shape = {3, 7, 32};
    HostTensor x = make_host_int(std::span<const int64_t>(shape), DataType::s8, 311, -128, 127);
    const QuantParams xq = per_tensor(0.02f, -4);
    const QuantParams yq = per_tensor(0.05f, 10);
    const TensorView xv = with_quant_params(x.view, xq);
    const TensorView ins[] = {xv};
    const HostTensor* owners[] = {&x};

    for (ActivationType t : {ActivationType::Relu, ActivationType::Sigmoid,
                             ActivationType::LeakyRelu}) {
        ActivationAttributes attrs;
        attrs.type = t;
        run_quant_op([&](Backend b) { return Activation::create(attrs, b); },
                     std::span<const TensorView>(ins, 1),
                     std::span<const HostTensor* const>(owners, 1), yq, 1.0f,
                     "activation quant per-tensor");
    }
}

NNOPS_TEST(cuda_activation_quant_pairs_and_tokens) {
    // u8 both sides, a zero point in the middle of the range.
    {
        HostTensor x = make_host_int(std::vector<int64_t>{2, 5, 64}, DataType::u8, 313, 0, 255);
        const QuantParams xq = per_tensor(0.01f, 128);
        const QuantParams yq = per_tensor(0.02f, 120);
        const TensorView xv = with_quant_params(x.view, xq);
        const TensorView ins[] = {xv};
        const HostTensor* owners[] = {&x};
        ActivationAttributes attrs;
        attrs.type = ActivationType::HardSwish;
        run_quant_op([&](Backend b) { return Activation::create(attrs, b); },
                     std::span<const TensorView>(ins, 1),
                     std::span<const HostTensor* const>(owners, 1), yq, 1.0f,
                     "activation quant u8");
    }
    // s8 in, u8 out — the pair the fused route carries.
    {
        HostTensor x = make_host_int(std::vector<int64_t>{16}, DataType::s8, 317, -128, 127);
        const QuantParams xq = per_tensor(0.03f, 0);
        const QuantParams yq = per_tensor(0.04f, 100);
        const TensorView xv = with_quant_params(x.view, xq);
        const TensorView ins[] = {xv};
        const HostTensor* owners[] = {&x};
        ActivationAttributes attrs;
        attrs.type = ActivationType::Elu;
        run_quant_op([&](Backend b) { return Activation::create(attrs, b); },
                     std::span<const TensorView>(ins, 1),
                     std::span<const HostTensor* const>(owners, 1), yq, 1.0f,
                     "activation quant s8 to u8");
    }
    // Per-token: one scale pair per row, on a row that leaves a scalar tail.
    {
        const int64_t rows = 5, cols = 40;
        HostTensor x = make_host_int(std::vector<int64_t>{rows, cols}, DataType::s8, 319, -128, 127);
        std::vector<float> xs, ys;
        std::vector<int32_t> xz, yz;
        QuantParams xq, yq;
        fill_per_token(rows, 321, xs, xz, xq);
        fill_per_token(rows, 323, ys, yz, yq);
        const TensorView xv = with_quant_params(x.view, xq);
        const TensorView ins[] = {xv};
        const HostTensor* owners[] = {&x};
        ActivationAttributes attrs;
        attrs.type = ActivationType::Gelu;
        run_quant_op([&](Backend b) { return Activation::create(attrs, b); },
                     std::span<const TensorView>(ins, 1),
                     std::span<const HostTensor* const>(owners, 1), yq, 1.0f,
                     "activation quant per-token");
    }
}

NNOPS_TEST(cuda_unary_quant) {
    HostTensor x = make_host_int(std::vector<int64_t>{4, 48}, DataType::s8, 331, -128, 127);
    const QuantParams xq = per_tensor(0.02f, 3);
    const QuantParams yq = per_tensor(0.01f, -5);
    const TensorView xv = with_quant_params(x.view, xq);
    const TensorView ins[] = {xv};
    const HostTensor* owners[] = {&x};

    for (UnaryType t : {UnaryType::Exp, UnaryType::Tanh, UnaryType::Abs, UnaryType::Sign}) {
        UnaryAttributes attrs;
        attrs.type = t;
        run_quant_op([&](Backend b) { return Unary::create(attrs, b); },
                     std::span<const TensorView>(ins, 1),
                     std::span<const HostTensor* const>(owners, 1), yq, 1.0f,
                     "unary quant per-tensor");
    }
}

NNOPS_TEST(cuda_unary_quant_per_token_u8) {
    const int64_t rows = 3, cols = 32;
    HostTensor x = make_host_int(std::vector<int64_t>{rows, cols}, DataType::u8, 337, 0, 255);
    std::vector<float> xs, ys;
    std::vector<int32_t> xz, yz;
    QuantParams xq, yq;
    fill_per_token(rows, 339, xs, xz, xq);
    fill_per_token(rows, 341, ys, yz, yq);
    // A u8 zero point instead of the signed spread.
    for (int64_t r = 0; r < rows; ++r) {
        xz[static_cast<size_t>(r)] = 100 + static_cast<int32_t>(r);
        yz[static_cast<size_t>(r)] = 120 - static_cast<int32_t>(r);
    }
    const TensorView xv = with_quant_params(x.view, xq);
    const TensorView ins[] = {xv};
    const HostTensor* owners[] = {&x};
    UnaryAttributes attrs;
    attrs.type = UnaryType::Sin;
    run_quant_op([&](Backend b) { return Unary::create(attrs, b); },
                 std::span<const TensorView>(ins, 1),
                 std::span<const HostTensor* const>(owners, 1), yq, 1.0f,
                 "unary quant per-token u8");
}

NNOPS_TEST(cuda_eltwise_quant) {
    HostTensor a = make_host_int(std::vector<int64_t>{2, 8, 16}, DataType::s8, 343, -128, 127);
    HostTensor b = make_host_int(std::vector<int64_t>{2, 8, 16}, DataType::s8, 347, -128, 127);
    const QuantParams aq = per_tensor(0.02f, -1);
    const QuantParams bq = per_tensor(0.03f, 2);
    const QuantParams yq = per_tensor(0.04f, -7);
    const TensorView av = with_quant_params(a.view, aq);
    const TensorView bv = with_quant_params(b.view, bq);
    const TensorView ins[] = {av, bv};
    const HostTensor* owners[] = {&a, &b};

    for (EltwiseType t : {EltwiseType::Add, EltwiseType::Mul, EltwiseType::Max,
                          EltwiseType::Sub}) {
        EltwiseAttributes attrs;
        attrs.type = t;
        run_quant_op([&](Backend be) { return Eltwise::create(attrs, be); },
                     std::span<const TensorView>(ins, 2),
                     std::span<const HostTensor* const>(owners, 2), yq, 1.0f,
                     "eltwise quant per-tensor");
    }
}

#endif  // NNOPS_HAS_CUDA
