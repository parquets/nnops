/// @file test_eltwise_vulkan.cpp
/// @brief Unit tests for Eltwise operator (Vulkan GPU backend).
///
/// Each test:
///   1. Computes the expected result on CPU (reference)
///   2. Runs the same computation via the Vulkan backend
///   3. Reads back GPU results and compares against CPU reference

#ifdef NNOPS_HAS_VULKAN

#include "nnops/ops/eltwise.hpp"
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
// Helper: run eltwise on Vulkan backend
// ============================================================

static std::vector<float> run_eltwise_vulkan(
    const float* a_data, const float* b_data,
    int64_t numel, EltwiseType op_type, bool add_to = false,
    const float* init_out = nullptr)
{
    test::VulkanTestEnv env;

    size_t buf_size = numel * sizeof(float);

    // Create device buffers
    auto buf_a = env.create_buffer(buf_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto buf_b = env.create_buffer(buf_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto buf_o = env.create_buffer(buf_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // Copy input data to device
    env.copy_to_device(buf_a, a_data, buf_size);
    env.copy_to_device(buf_b, b_data, buf_size);
    if (init_out) {
        env.copy_to_device(buf_o, init_out, buf_size);
    } else {
        std::vector<float> zeros(numel, 0.0f);
        env.copy_to_device(buf_o, zeros.data(), buf_size);
    }

    // Build Vulkan ComputeContext
    ComputeContext ctx;
    ctx.expected_backend = Backend::Vulkan;
    ctx.vulkan_device = env.device();
    ctx.vulkan_descriptor_pool = env.descriptor_pool();

    const void* bufs[] = { (const void*)(uintptr_t)buf_a.buffer,
                           (const void*)(uintptr_t)buf_b.buffer,
                           (const void*)(uintptr_t)buf_o.buffer };
    ctx.vulkan_buffers = bufs;
    ctx.vulkan_buffers_count = 3;

    // Create operator
    EltwiseAttributes attrs;
    attrs.type = op_type;
    attrs.add_to = add_to;
    auto op = Eltwise::create(attrs, Backend::Vulkan);

    // Record and submit
    VkCommandBuffer cmd = env.begin_cmd();
    ctx.vulkan_cmd_buffer = cmd;

    const int64_t shape[] = {numel};
    // We need TensorViews for the operator API — but Vulkan backend doesn't
    // read from the data pointer, it reads from VkBuffers. We still need
    // valid TensorViews for shape/numel metadata.
    // Use dummy pointers — the Vulkan backend ignores the data pointer.
    TensorView a_view(shape, DataType::f32, const_cast<float*>(a_data));
    TensorView b_view(shape, DataType::f32, const_cast<float*>(b_data));
    TensorView o_view(shape, DataType::f32, nullptr);  // data ptr unused

    const TensorView ins[] = {a_view, b_view};
    op->compute(o_view, ins, ctx, nullptr);

    env.submit_and_wait();

    // Read back result
    std::vector<float> result(numel);
    env.copy_from_device(buf_o, result.data(), buf_size);

    // Cleanup
    env.destroy_buffer(buf_a);
    env.destroy_buffer(buf_b);
    env.destroy_buffer(buf_o);

    return result;
}

// ============================================================
// Tests: Vulkan vs CPU reference
// ============================================================

NNOPS_TEST(eltwise_vulkan_add_1d) {
    const int N = 256;
    float a[256], b[256];
    for (int i = 0; i < N; ++i) { a[i] = i * 1.0f; b[i] = i * 2.0f; }

    auto result = run_eltwise_vulkan(a, b, N, EltwiseType::Add);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], a[i] + b[i], 1e-4f);
    }
}

NNOPS_TEST(eltwise_vulkan_sub_1d) {
    const int N = 100;
    float a[100], b[100];
    for (int i = 0; i < N; ++i) { a[i] = 100.0f; b[i] = i * 1.0f; }

    auto result = run_eltwise_vulkan(a, b, N, EltwiseType::Sub);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], a[i] - b[i], 1e-4f);
    }
}

NNOPS_TEST(eltwise_vulkan_mul_1d) {
    const int N = 128;
    float a[128], b[128];
    for (int i = 0; i < N; ++i) { a[i] = i + 1.0f; b[i] = 3.0f; }

    auto result = run_eltwise_vulkan(a, b, N, EltwiseType::Mul);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], a[i] * b[i], 1e-4f);
    }
}

NNOPS_TEST(eltwise_vulkan_div_1d) {
    const int N = 64;
    float a[64], b[64];
    for (int i = 0; i < N; ++i) { a[i] = 100.0f + i; b[i] = 2.0f; }

    auto result = run_eltwise_vulkan(a, b, N, EltwiseType::Div);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], a[i] / b[i], 1e-4f);
    }
}

NNOPS_TEST(eltwise_vulkan_add_to) {
    const int N = 32;
    float a[32], b[32], init[32];
    for (int i = 0; i < N; ++i) { a[i] = 1.0f; b[i] = 2.0f; init[i] = 10.0f; }

    auto result = run_eltwise_vulkan(a, b, N, EltwiseType::Add, true, init);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], init[i] + a[i] + b[i], 1e-4f);
    }
}

NNOPS_TEST(eltwise_vulkan_large_add) {
    const int N = 10000;
    auto [a_vec, a_view] = test::make_random_tensor({N}, -10.0f, 10.0f, 123);
    auto [b_vec, b_view] = test::make_random_tensor({N}, -10.0f, 10.0f, 456);

    auto result = run_eltwise_vulkan(a_vec.data(), b_vec.data(), N, EltwiseType::Add);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], a_vec[i] + b_vec[i], 1e-4f);
    }
}

NNOPS_TEST(eltwise_vulkan_many_elements) {
    // Test with >256 elements to verify multi-workgroup dispatch
    const int N = 12345;
    auto [a_vec, a_view] = test::make_random_tensor({N}, -5.0f, 5.0f, 789);
    auto [b_vec, b_view] = test::make_random_tensor({N}, -5.0f, 5.0f, 101);

    auto result = run_eltwise_vulkan(a_vec.data(), b_vec.data(), N, EltwiseType::Mul);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], a_vec[i] * b_vec[i], 1e-3f);
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

/// Helper: run eltwise on Vulkan backend with f16 data
/// Returns fp16 results converted to float for comparison.
static std::vector<float> run_eltwise_vulkan_f16(
    const float* a_data, const float* b_data,
    int64_t numel, EltwiseType op_type, bool add_to = false,
    const float* init_out = nullptr)
{
    test::VulkanTestEnv env;

    // Convert float inputs to fp16
    auto a_f16 = floats_to_f16(a_data, numel);
    auto b_f16 = floats_to_f16(b_data, numel);
    std::vector<uint16_t> o_f16(numel, 0);
    if (init_out) {
        o_f16 = floats_to_f16(init_out, numel);
    }

    size_t buf_size = numel * sizeof(uint16_t);

    auto buf_a = env.create_buffer(buf_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto buf_b = env.create_buffer(buf_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto buf_o = env.create_buffer(buf_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    env.copy_to_device(buf_a, a_f16.data(), buf_size);
    env.copy_to_device(buf_b, b_f16.data(), buf_size);
    env.copy_to_device(buf_o, o_f16.data(), buf_size);

    ComputeContext ctx;
    ctx.expected_backend = Backend::Vulkan;
    ctx.vulkan_device = env.device();
    ctx.vulkan_descriptor_pool = env.descriptor_pool();

    const void* bufs[] = { (const void*)(uintptr_t)buf_a.buffer,
                           (const void*)(uintptr_t)buf_b.buffer,
                           (const void*)(uintptr_t)buf_o.buffer };
    ctx.vulkan_buffers = bufs;
    ctx.vulkan_buffers_count = 3;

    EltwiseAttributes attrs;
    attrs.type = op_type;
    attrs.add_to = add_to;
    auto op = Eltwise::create(attrs, Backend::Vulkan);

    VkCommandBuffer cmd = env.begin_cmd();
    ctx.vulkan_cmd_buffer = cmd;

    const int64_t shape[] = {numel};
    // Use a dummy uint16_t pointer for TensorView; Vulkan backend ignores data ptr
    uint16_t dummy = 0;
    TensorView a_view(shape, DataType::f16, &dummy);
    TensorView b_view(shape, DataType::f16, &dummy);
    TensorView o_view(shape, DataType::f16, nullptr);

    const TensorView ins[] = {a_view, b_view};
    op->compute(o_view, ins, ctx, nullptr);

    env.submit_and_wait();

    // Read back fp16 result
    std::vector<uint16_t> result_f16(numel);
    env.copy_from_device(buf_o, result_f16.data(), buf_size);

    env.destroy_buffer(buf_a);
    env.destroy_buffer(buf_b);
    env.destroy_buffer(buf_o);

    return f16_to_floats(result_f16.data(), numel);
}

/// Helper: compute fp16 reference result.
/// Converts inputs to fp16, computes in float, converts back to fp16.
static std::vector<float> compute_eltwise_f16_ref(
    const float* a, const float* b, int64_t n,
    EltwiseType op_type, bool add_to = false,
    const float* init = nullptr)
{
    std::vector<float> result(n);
    for (int64_t i = 0; i < n; ++i) {
        float va = half_to_float(float_to_half(a[i]));
        float vb = half_to_float(float_to_half(b[i]));
        float r;
        switch (op_type) {
        case EltwiseType::Add: r = va + vb; break;
        case EltwiseType::Sub: r = va - vb; break;
        case EltwiseType::Mul: r = va * vb; break;
        case EltwiseType::Div: r = va / vb; break;
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

NNOPS_TEST(eltwise_vulkan_f16_add_1d) {
    const int N = 256;
    float a[256], b[256];
    for (int i = 0; i < N; ++i) { a[i] = i * 0.5f; b[i] = i * 1.0f; }

    auto result = run_eltwise_vulkan_f16(a, b, N, EltwiseType::Add);
    auto expected = compute_eltwise_f16_ref(a, b, N, EltwiseType::Add);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], expected[i], 1e-3f);
    }
}

NNOPS_TEST(eltwise_vulkan_f16_mul_1d) {
    const int N = 128;
    float a[128], b[128];
    for (int i = 0; i < N; ++i) { a[i] = i + 1.0f; b[i] = 2.0f; }

    auto result = run_eltwise_vulkan_f16(a, b, N, EltwiseType::Mul);
    auto expected = compute_eltwise_f16_ref(a, b, N, EltwiseType::Mul);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], expected[i], 1e-3f);
    }
}

NNOPS_TEST(eltwise_vulkan_f16_add_to) {
    const int N = 64;
    float a[64], b[64], init[64];
    for (int i = 0; i < N; ++i) { a[i] = 0.5f; b[i] = 1.5f; init[i] = 10.0f; }

    auto result = run_eltwise_vulkan_f16(a, b, N, EltwiseType::Add, true, init);
    auto expected = compute_eltwise_f16_ref(a, b, N, EltwiseType::Add, true, init);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], expected[i], 1e-3f);
    }
}

NNOPS_TEST(eltwise_vulkan_f16_large) {
    const int N = 10000;
    auto [a_vec, a_view] = test::make_random_tensor({N}, -5.0f, 5.0f, 777);
    auto [b_vec, b_view] = test::make_random_tensor({N}, -5.0f, 5.0f, 888);

    auto result = run_eltwise_vulkan_f16(a_vec.data(), b_vec.data(), N, EltwiseType::Add);
    auto expected = compute_eltwise_f16_ref(a_vec.data(), b_vec.data(), N, EltwiseType::Add);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], expected[i], 1e-3f);
    }
}

NNOPS_TEST(eltwise_vulkan_f16_many_elements) {
    const int N = 12345;
    auto [a_vec, a_view] = test::make_random_tensor({N}, -8.0f, 8.0f, 999);
    auto [b_vec, b_view] = test::make_random_tensor({N}, -8.0f, 8.0f, 111);

    auto result = run_eltwise_vulkan_f16(a_vec.data(), b_vec.data(), N, EltwiseType::Mul);
    auto expected = compute_eltwise_f16_ref(a_vec.data(), b_vec.data(), N, EltwiseType::Mul);

    for (int i = 0; i < N; ++i) {
        NNOPS_EXPECT_NEAR(result[i], expected[i], 1e-3f);
    }
}

#endif  // NNOPS_HAS_VULKAN
