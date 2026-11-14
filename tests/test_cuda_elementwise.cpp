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
/// budget, so softmax re-reads it from global memory, and the same split covers
/// the strided general-axis path on a row either side of that budget.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/activation.hpp"
#include "nnops/ops/softmax.hpp"
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
                    bool padded, TensorLayout layout = TensorLayout::NCHW,
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
                 bool padded, TensorLayout layout = TensorLayout::NCHW,
                 bool log_softmax = false, const char* what = "",
                 float tol_abs = 0.0f, float tol_rel = 0.0f,
                 float temperature = 1.0f)
{
    SoftmaxAttributes attrs;
    attrs.axis = axis;
    attrs.log_softmax = log_softmax;
    attrs.temperature = temperature;

    HostTensor in = make_host_random(shape, dt, padded, 19, -4.0f, 4.0f, layout);
    const HostTensor ins[] = {in};
    run_op_close([&](Backend be) { return Softmax::create(attrs, be); }, ins,
                 tol_abs, tol_rel, what);
}

}  // namespace

NNOPS_TEST(cuda_activation_lanes_and_tail) {
    // Dense f32 of 37: row bases 148 B apart, so odd rows are off the 16 B
    // bound and fall back to the scalar loop.
    run_activation(ActivationType::Relu, {64, 37}, DataType::f32, false, {},
                   false, "cuda_activation_relu_37");
    // The same width with the graph's 32 B pitch: every base is aligned and only
    // the tail (37 = 4·9 + 1) takes the scalar loop.
    run_activation(ActivationType::Silu, {64, 37}, DataType::f32, true, {},
                   false, "cuda_activation_silu_37_pitched");
    run_activation(ActivationType::Gelu, {8, 1024}, DataType::f32, true, {},
                   false, "cuda_activation_gelu");
    run_activation(ActivationType::Sigmoid, {8, 1024}, DataType::f32, true, {},
                   false, "cuda_activation_sigmoid");
    run_activation(ActivationType::HardSwish, {8, 1024}, DataType::f32, true, {},
                   false, "cuda_activation_hardswish");
    // A rank-1 tensor takes the flat kernel, whose grid is sized in lanes.
    run_activation(ActivationType::Relu, {100003}, DataType::f32, false, {}, false,
                   "cuda_activation_flat_odd");
    run_activation(ActivationType::Silu, {100003}, DataType::f16, false, {}, false,
                   "cuda_activation_flat_odd_f16");
}

NNOPS_TEST(cuda_activation_f16_and_add_to) {
    // f16 lanes are 8 wide, so a 37-wide row leaves 5 elements on the tail.
    run_activation(ActivationType::Elu, {64, 37}, DataType::f16, true, {},
                   false, "cuda_activation_elu_37_pitched");
    run_activation(ActivationType::LeakyRelu, {8, 1024}, DataType::f16, true, {},
                   false, "cuda_activation_leaky_f16");
    run_activation(ActivationType::Silu, {8, 1024}, DataType::f16, true, {},
                   true, "cuda_activation_silu_add_to");
    // Channel-last rows are one spatial line of W·C, which is the last dim the
    // kernel is handed.
    run_activation(ActivationType::Relu, {2, 8, 7, 5}, DataType::f32, false,
                   TensorLayout::NHWC, false, "cuda_activation_channel_last");
}

NNOPS_TEST(cuda_softmax_contiguous) {
    // A float row of 4 KB is held in shared memory for the later passes.
    run_softmax({128, 1024}, -1, DataType::f32, true, {}, false,
                "cuda_softmax_1024", kF32Abs, kF32Rel);
    // 37 = 4·9 + 1: an unaligned row on half the pitches, a tail on all of them.
    run_softmax({64, 37}, -1, DataType::f32, false, {}, false,
                "cuda_softmax_37_dense", kF32Abs, kF32Rel);
    run_softmax({64, 37}, -1, DataType::f32, true, {}, true,
                "cuda_softmax_log_37_pitched", kF32Abs, kF32Rel);
    run_softmax({8, 8192}, -1, DataType::f32, true, {}, false,
                "cuda_softmax_8192", kF32Abs, kF32Rel);
    // 20000 floats is 80 KB: past the budget, so the row is read again instead
    // of being kept.
    run_softmax({4, 20000}, -1, DataType::f32, false, {}, false,
                "cuda_softmax_20000", kF32Abs, kF32Rel);
    run_softmax({4, 20000}, -1, DataType::f32, false, {}, true,
                "cuda_softmax_log_20000", kF32Abs, kF32Rel);
}

NNOPS_TEST(cuda_softmax_channel_last_and_f16) {
    // Channel-last with the axis on C: the groups are the layout's own rows.
    run_softmax({1, 8, 16, 16}, 1, DataType::f32, false, TensorLayout::NHWC,
                false, "cuda_softmax_channel_last", kF32Abs, kF32Rel);
    run_softmax({128, 1024}, -1, DataType::f16, true, {}, false,
                "cuda_softmax_f16", kF16Abs, kF16Rel);
    run_softmax({64, 37}, -1, DataType::f16, true, {}, true,
                "cuda_softmax_log_f16_37", kF16Abs, kF16Rel);
}

NNOPS_TEST(cuda_softmax_temperature) {
    // The device route divides the shifted logits by T exactly where the CPU does;
    // until this was pinned it ignored the attribute and returned T = 1 results.
    run_softmax({64, 1000}, -1, DataType::f32, true, {}, false,
                "cuda_softmax_t_half", kF32Abs, kF32Rel, 0.5f);
    run_softmax({64, 1000}, -1, DataType::f32, true, {}, true,
                "cuda_softmax_t_two_log", kF32Abs, kF32Rel, 2.0f);
    run_softmax({3, 5, 7}, 1, DataType::f32, false, {}, false,
                "cuda_softmax_t_half_axis1", kF32Abs, kF32Rel, 0.5f);
}

NNOPS_TEST(cuda_softmax_strided_axis) {
    // Axis 1 of [3, 5, 7]: the general kernel walks D elements at a stride, and
    // five of them fit the residency budget.
    run_softmax({3, 5, 7}, 1, DataType::f32, false, {}, false,
                "cuda_softmax_axis1", kF32Abs, kF32Rel);
    run_softmax({2, 600, 3}, 1, DataType::f32, false, {}, false,
                "cuda_softmax_axis1_600", kF32Abs, kF32Rel);
    run_softmax({2, 600, 3}, 1, DataType::f32, false, {}, true,
                "cuda_softmax_axis1_600_log", kF32Abs, kF32Rel);
    // 15000 floats is 60 KB, past the budget: the general path re-reads the row.
    run_softmax({2, 15000, 3}, 1, DataType::f32, false, {}, false,
                "cuda_softmax_axis1_15000", kF32Abs, kF32Rel);
}

#endif  // NNOPS_HAS_CUDA
