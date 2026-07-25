/// @file test_unary_vulkan.cpp
/// @brief Unit tests for Unary operator (Vulkan GPU backend).
///
/// Each test:
///   1. Computes the expected result on CPU (reference)
///   2. Runs the same computation via the Vulkan backend
///   3. Reads back GPU results and compares against CPU reference

#ifdef NNOPS_HAS_VULKAN

#include "nnops/ops/unary.hpp"
#include "nnops/detail/half.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/vulkan_test_helper.hpp"

#include <vector>
#include <cmath>
#include <memory>
#include <cstring>

using namespace nnops;

// ============================================================
// Helper: run unary on Vulkan backend
// ============================================================

static std::vector<float> run_unary_vulkan(
    const float* in_data, int64_t numel, UnaryType op_type, bool add_to = false,
    const float* init_out = nullptr)
{
    test::VulkanTestEnv env;

    size_t buf_size = numel * sizeof(float);

    // Create device buffers
    auto buf_in = env.create_buffer(buf_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto buf_out = env.create_buffer(buf_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // Copy input data
    env.copy_to_device(buf_in, in_data, buf_size);
    if (init_out) {
        env.copy_to_device(buf_out, init_out, buf_size);
    } else {
        std::vector<float> zeros(numel, 0.0f);
        env.copy_to_device(buf_out, zeros.data(), buf_size);
    }

    // Build Vulkan ComputeContext
    ComputeContext ctx;
    ctx.expected_backend = Backend::Vulkan;
    ctx.vulkan_device = env.device();
    ctx.vulkan_descriptor_pool = env.descriptor_pool();

    const void* bufs[] = { (const void*)(uintptr_t)buf_in.buffer,
                           (const void*)(uintptr_t)buf_out.buffer };
    ctx.vulkan_buffers = bufs;
    ctx.vulkan_buffers_count = 2;

    // Create operator
    UnaryAttributes attrs;
    attrs.type = op_type;
    attrs.add_to = add_to;
    auto op = Unary::create(attrs, Backend::Vulkan);

    // Record and submit
    VkCommandBuffer cmd = env.begin_cmd();
    ctx.vulkan_cmd_buffer = cmd;

    const int64_t shape[] = {numel};
    TensorView in_view(shape, DataType::f32, const_cast<float*>(in_data));
    TensorView out_view(shape, DataType::f32, nullptr);  // data ptr unused

    const TensorView ins[] = {in_view};
    op->compute(out_view, ins, ctx, nullptr);

    env.submit_and_wait();

    // Read back result
    std::vector<float> result(numel);
    env.copy_from_device(buf_out, result.data(), buf_size);

    // Cleanup
    env.destroy_buffer(buf_in);
    env.destroy_buffer(buf_out);

    return result;
}

// ============================================================
// Tests: Each unary op type
// ============================================================

NNOPS_TEST(unary_vulkan_exp) {
    const int N = 256;
    float in[256];
    for (int i = 0; i < N; ++i) in[i] = (i - 128) * 0.05f;

    auto result = run_unary_vulkan(in, N, UnaryType::Exp);

    for (int i = 0; i < N; ++i) {
        // GPU exp() has slightly different precision than std::exp();
        // 2e-4 relative tolerance accounts for this.
        float expected = std::exp(in[i]);
        float tol = 1e-5f + 2e-4f * std::abs(expected);
        NNOPS_EXPECT_NEAR(result[i], expected, tol);
    }
}

NNOPS_TEST(unary_vulkan_log) {
    const int N = 128;
    float in[128];
    for (int i = 0; i < N; ++i) in[i] = 1.0f + i * 0.1f;

    auto result = run_unary_vulkan(in, N, UnaryType::Log);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], std::log(in[i]), 1e-4f);
    }
}

NNOPS_TEST(unary_vulkan_sin) {
    const int N = 100;
    float in[100];
    for (int i = 0; i < N; ++i) in[i] = i * 0.1f;

    auto result = run_unary_vulkan(in, N, UnaryType::Sin);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], std::sin(in[i]), 1e-4f);
    }
}

NNOPS_TEST(unary_vulkan_cos) {
    const int N = 100;
    float in[100];
    for (int i = 0; i < N; ++i) in[i] = i * 0.1f;

    auto result = run_unary_vulkan(in, N, UnaryType::Cos);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], std::cos(in[i]), 1e-4f);
    }
}

NNOPS_TEST(unary_vulkan_tan) {
    const int N = 50;
    float in[50];
    for (int i = 0; i < N; ++i) in[i] = (i - 25) * 0.04f;

    auto result = run_unary_vulkan(in, N, UnaryType::Tan);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], std::tan(in[i]), 1e-3f);
    }
}

NNOPS_TEST(unary_vulkan_tanh) {
    const int N = 128;
    float in[128];
    for (int i = 0; i < N; ++i) in[i] = (i - 64) * 0.05f;

    auto result = run_unary_vulkan(in, N, UnaryType::Tanh);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], std::tanh(in[i]), 1e-4f);
    }
}

NNOPS_TEST(unary_vulkan_abs) {
    const int N = 256;
    float in[256];
    for (int i = 0; i < N; ++i) in[i] = (i - 128) * 0.5f;

    auto result = run_unary_vulkan(in, N, UnaryType::Abs);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], std::abs(in[i]), 1e-5f);
    }
}

NNOPS_TEST(unary_vulkan_neg) {
    const int N = 128;
    float in[128];
    for (int i = 0; i < N; ++i) in[i] = i * 0.5f - 32.0f;

    auto result = run_unary_vulkan(in, N, UnaryType::Neg);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], -in[i], 1e-5f);
    }
}

NNOPS_TEST(unary_vulkan_sqrt) {
    const int N = 256;
    float in[256];
    for (int i = 0; i < N; ++i) in[i] = i * 0.1f;

    auto result = run_unary_vulkan(in, N, UnaryType::Sqrt);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], std::sqrt(in[i]), 1e-4f);
    }
}

NNOPS_TEST(unary_vulkan_add_to) {
    const int N = 64;
    float in[64], init[64];
    for (int i = 0; i < N; ++i) { in[i] = i * 0.1f; init[i] = 5.0f; }

    auto result = run_unary_vulkan(in, N, UnaryType::Exp, true, init);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], init[i] + std::exp(in[i]), 1e-4f);
    }
}

NNOPS_TEST(unary_vulkan_large) {
    const int N = 10000;
    auto [in_vec, in_view] = test::make_random_tensor({N}, 0.1f, 4.0f, 333);

    auto result = run_unary_vulkan(in_vec.data(), N, UnaryType::Sqrt);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], std::sqrt(in_vec[i]), 1e-4f);
    }
}

NNOPS_TEST(unary_vulkan_many_elements) {
    // Test with >256 elements to verify multi-workgroup dispatch
    const int N = 9876;
    auto [in_vec, in_view] = test::make_random_tensor({N}, -1.0f, 1.0f, 555);

    auto result = run_unary_vulkan(in_vec.data(), N, UnaryType::Tanh);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], std::tanh(in_vec[i]), 1e-4f);
    }
}

// ============================================================
// fp16 (half-precision) tests
// ============================================================

using nnops::backend::cpu::half;
using nnops::backend::cpu::half_to_float;
using nnops::backend::cpu::float_to_half;

/// Helper: convert float array to fp16 uint16_t array
static std::vector<uint16_t> floats_to_f16(const float* src, int64_t n) {
    std::vector<uint16_t> result(n);
    for (int64_t i = 0; i < n; ++i) {
        result[i] = float_to_half(src[i]).bits;
    }
    return result;
}

/// Helper: convert fp16 uint16_t array to float array
static std::vector<float> f16_to_floats(const uint16_t* src, int64_t n) {
    std::vector<float> result(n);
    for (int64_t i = 0; i < n; ++i) {
        result[i] = half_to_float(half{src[i]});
    }
    return result;
}

/// Helper: run unary on Vulkan backend with f16 data
static std::vector<float> run_unary_vulkan_f16(
    const float* in_data, int64_t numel, UnaryType op_type, bool add_to = false,
    const float* init_out = nullptr)
{
    test::VulkanTestEnv env;

    auto in_f16 = floats_to_f16(in_data, numel);
    std::vector<uint16_t> out_f16(numel, 0);
    if (init_out) {
        out_f16 = floats_to_f16(init_out, numel);
    }

    size_t buf_size = numel * sizeof(uint16_t);

    auto buf_in = env.create_buffer(buf_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto buf_out = env.create_buffer(buf_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    env.copy_to_device(buf_in, in_f16.data(), buf_size);
    env.copy_to_device(buf_out, out_f16.data(), buf_size);

    ComputeContext ctx;
    ctx.expected_backend = Backend::Vulkan;
    ctx.vulkan_device = env.device();
    ctx.vulkan_descriptor_pool = env.descriptor_pool();

    const void* bufs[] = { (const void*)(uintptr_t)buf_in.buffer,
                           (const void*)(uintptr_t)buf_out.buffer };
    ctx.vulkan_buffers = bufs;
    ctx.vulkan_buffers_count = 2;

    UnaryAttributes attrs;
    attrs.type = op_type;
    attrs.add_to = add_to;
    auto op = Unary::create(attrs, Backend::Vulkan);

    VkCommandBuffer cmd = env.begin_cmd();
    ctx.vulkan_cmd_buffer = cmd;

    const int64_t shape[] = {numel};
    uint16_t dummy = 0;
    TensorView in_view(shape, DataType::f16, &dummy);
    TensorView out_view(shape, DataType::f16, nullptr);

    const TensorView ins[] = {in_view};
    op->compute(out_view, ins, ctx, nullptr);

    env.submit_and_wait();

    std::vector<uint16_t> result_f16(numel);
    env.copy_from_device(buf_out, result_f16.data(), buf_size);

    env.destroy_buffer(buf_in);
    env.destroy_buffer(buf_out);

    return f16_to_floats(result_f16.data(), numel);
}

/// Helper: compute fp16 unary reference (round-trip through fp16)
static std::vector<float> compute_unary_f16_ref(
    const float* in, int64_t n, UnaryType op_type, bool add_to = false,
    const float* init = nullptr)
{
    std::vector<float> result(n);
    for (int64_t i = 0; i < n; ++i) {
        float x = half_to_float(float_to_half(in[i]));
        float r;
        switch (op_type) {
        case UnaryType::Exp:  r = std::exp(x);  break;
        case UnaryType::Log:  r = std::log(x);  break;
        case UnaryType::Sin:  r = std::sin(x);  break;
        case UnaryType::Cos:  r = std::cos(x);  break;
        case UnaryType::Tan:  r = std::tan(x);  break;
        case UnaryType::Tanh: r = std::tanh(x); break;
        case UnaryType::Abs:  r = std::abs(x);  break;
        case UnaryType::Neg:  r = -x;           break;
        case UnaryType::Sqrt: r = std::sqrt(x); break;
        default: r = 0;
        }
        if (add_to && init) {
            float init_f16 = half_to_float(float_to_half(init[i]));
            r = init_f16 + r;
        }
        result[i] = half_to_float(float_to_half(r));
    }
    return result;
}

NNOPS_TEST(unary_vulkan_f16_exp) {
    const int N = 256;
    float in[256];
    for (int i = 0; i < N; ++i) in[i] = (i - 128) * 0.02f;  // safe range for exp

    auto result = run_unary_vulkan_f16(in, N, UnaryType::Exp);
    auto expected = compute_unary_f16_ref(in, N, UnaryType::Exp);

    for (int i = 0; i < N; ++i) {
        float tol = 1e-3f + 5e-3f * std::abs(expected[i]);
        NNOPS_EXPECT_NEAR(result[i], expected[i], tol);
    }
}

NNOPS_TEST(unary_vulkan_f16_log) {
    const int N = 128;
    float in[128];
    for (int i = 0; i < N; ++i) in[i] = 0.5f + i * 0.05f;

    auto result = run_unary_vulkan_f16(in, N, UnaryType::Log);
    auto expected = compute_unary_f16_ref(in, N, UnaryType::Log);

    for (int i = 0; i < N; ++i) {
        float tol = 1e-3f + 5e-3f * std::abs(expected[i]);
        NNOPS_EXPECT_NEAR(result[i], expected[i], tol);
    }
}

NNOPS_TEST(unary_vulkan_f16_sin) {
    const int N = 100;
    float in[100];
    for (int i = 0; i < N; ++i) in[i] = i * 0.1f;

    auto result = run_unary_vulkan_f16(in, N, UnaryType::Sin);
    auto expected = compute_unary_f16_ref(in, N, UnaryType::Sin);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], expected[i], 1e-3f);
    }
}

NNOPS_TEST(unary_vulkan_f16_cos) {
    const int N = 100;
    float in[100];
    for (int i = 0; i < N; ++i) in[i] = i * 0.1f;

    auto result = run_unary_vulkan_f16(in, N, UnaryType::Cos);
    auto expected = compute_unary_f16_ref(in, N, UnaryType::Cos);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], expected[i], 1e-3f);
    }
}

NNOPS_TEST(unary_vulkan_f16_tanh) {
    const int N = 128;
    float in[128];
    for (int i = 0; i < N; ++i) in[i] = (i - 64) * 0.05f;

    auto result = run_unary_vulkan_f16(in, N, UnaryType::Tanh);
    auto expected = compute_unary_f16_ref(in, N, UnaryType::Tanh);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], expected[i], 1e-3f);
    }
}

NNOPS_TEST(unary_vulkan_f16_abs) {
    const int N = 256;
    float in[256];
    for (int i = 0; i < N; ++i) in[i] = (i - 128) * 0.5f;

    auto result = run_unary_vulkan_f16(in, N, UnaryType::Abs);
    auto expected = compute_unary_f16_ref(in, N, UnaryType::Abs);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], expected[i], 1e-5f);
    }
}

NNOPS_TEST(unary_vulkan_f16_neg) {
    const int N = 128;
    float in[128];
    for (int i = 0; i < N; ++i) in[i] = i * 0.5f - 32.0f;

    auto result = run_unary_vulkan_f16(in, N, UnaryType::Neg);
    auto expected = compute_unary_f16_ref(in, N, UnaryType::Neg);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], expected[i], 1e-5f);
    }
}

NNOPS_TEST(unary_vulkan_f16_sqrt) {
    const int N = 256;
    float in[256];
    for (int i = 0; i < N; ++i) in[i] = i * 0.1f + 0.01f;

    auto result = run_unary_vulkan_f16(in, N, UnaryType::Sqrt);
    auto expected = compute_unary_f16_ref(in, N, UnaryType::Sqrt);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], expected[i], 1e-3f);
    }
}

NNOPS_TEST(unary_vulkan_f16_add_to) {
    const int N = 64;
    float in[64], init[64];
    for (int i = 0; i < N; ++i) { in[i] = 0.1f * i; init[i] = 5.0f; }

    auto result = run_unary_vulkan_f16(in, N, UnaryType::Exp, true, init);
    auto expected = compute_unary_f16_ref(in, N, UnaryType::Exp, true, init);

    for (int i = 0; i < N; ++i) {
        float tol = 1e-3f + 5e-3f * std::abs(expected[i]);
        NNOPS_EXPECT_NEAR(result[i], expected[i], tol);
    }
}

NNOPS_TEST(unary_vulkan_f16_large) {
    const int N = 10000;
    auto [in_vec, in_view] = test::make_random_tensor({N}, 0.01f, 4.0f, 1234);

    auto result = run_unary_vulkan_f16(in_vec.data(), N, UnaryType::Sqrt);
    auto expected = compute_unary_f16_ref(in_vec.data(), N, UnaryType::Sqrt);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], expected[i], 1e-3f);
    }
}

#endif  // NNOPS_HAS_VULKAN
