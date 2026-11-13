/// @file test_cuda_gather_scatter.cpp
/// @brief CUDA Gather / GatherElements / GatherND / ScatterElements / ScatterND
///        against the CPU backend.
///
/// The oracle is the CPU route over the same bytes, as elsewhere, and the
/// comparison is byte-exact: both operators only move values, and where they add
/// or multiply, every index set here is duplicate-free, so each output element
/// sees at most one update and the arithmetic is a single rounding-identical
/// operation on both sides. The one accumulating case that *does* repeat an
/// index runs on s32, where integer addition order cannot matter.
///
/// Indices are s64 and s32, with negatives wrapping from the end of the axis.
/// Out-of-range indices are not tested: the CPU rejects them during validation
/// while the CUDA kernel clamps (there is no host pass to report from), so the
/// two backends deliberately differ there.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/gather.hpp"
#include "nnops/ops/scatter.hpp"

#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/cuda_test_helper.hpp"

#include <span>
#include <utility>
#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

/// Random f32/f16 data (the builder the other CUDA tests use).
HostTensor rnd(std::vector<int64_t> shape, DataType dt = DataType::f32,
               bool padded = false, uint64_t seed = 17)
{
    return make_host_random(shape, dt, padded, seed, -0.5f, 0.5f);
}

/// Arbitrary bytes for the integer dtypes the random builder does not fill.
HostTensor raw(std::vector<int64_t> shape, DataType dt)
{
    HostTensor t;
    const TensorView probe(shape, dt, nullptr);
    t.bytes.resize(probe.storage_bytes());
    for (size_t i = 0; i < t.bytes.size(); ++i) {
        t.bytes[i] = std::byte(static_cast<unsigned char>(i * 37u + 11u));
    }
    t.view = TensorView(shape, dt, t.bytes.data());
    return t;
}

/// An index tensor holding the given values, in the requested index dtype.
HostTensor idxs(std::vector<int64_t> shape, DataType dt,
                std::vector<int64_t> values)
{
    HostTensor t;
    const TensorView probe(shape, dt, nullptr);
    t.bytes.assign(probe.storage_bytes(), std::byte{0});
    const size_t n = std::min(values.size(), static_cast<size_t>(probe.numel()));
    if (dt == DataType::s64) {
        auto* p = reinterpret_cast<int64_t*>(t.bytes.data());
        for (size_t i = 0; i < n; ++i) { p[i] = values[i]; }
    } else {
        auto* p = reinterpret_cast<int32_t*>(t.bytes.data());
        for (size_t i = 0; i < n; ++i) { p[i] = static_cast<int32_t>(values[i]); }
    }
    t.view = TensorView(shape, dt, t.bytes.data());
    return t;
}

}  // namespace

// ============================================================
// Gather
// ============================================================

NNOPS_TEST(cuda_gather_axis0) {
    HostTensor d = rnd({5, 3}, DataType::f32, true);
    HostTensor i = idxs({2}, DataType::s64, {1, -1});   // -1 wraps to row 4
    const HostTensor in[] = {std::move(d), std::move(i)};
    run_op_both([](Backend be) { return Gather::create(GatherAttributes{0}, be); },
                std::span<const HostTensor>(in), "cuda_gather_axis0");
}

NNOPS_TEST(cuda_gather_axis1_2d_indices) {
    HostTensor d = rnd({2, 3}, DataType::f32);
    HostTensor i = idxs({2, 2}, DataType::s32, {2, 0, -1, 1});
    const HostTensor in[] = {std::move(d), std::move(i)};
    run_op_both([](Backend be) { return Gather::create(GatherAttributes{1}, be); },
                std::span<const HostTensor>(in), "cuda_gather_axis1");
}

NNOPS_TEST(cuda_gather_3d_mid_axis) {
    HostTensor d = rnd({2, 4, 3}, DataType::f16);
    HostTensor i = idxs({2, 2, 2}, DataType::s64, {0, 3, 2, 1, 3, 0, 1, 2});
    const HostTensor in[] = {std::move(d), std::move(i)};
    run_op_both([](Backend be) { return Gather::create(GatherAttributes{1}, be); },
                std::span<const HostTensor>(in), "cuda_gather_3d_mid");
}

NNOPS_TEST(cuda_gather_rank1_data) {
    // The gathered axis is the last one, so the whole output is elements rather
    // than rows of them.
    HostTensor d = raw({6}, DataType::s32);
    HostTensor i = idxs({4}, DataType::s32, {5, 0, 3, 2});
    const HostTensor in[] = {std::move(d), std::move(i)};
    run_op_both([](Backend be) { return Gather::create(GatherAttributes{0}, be); },
                std::span<const HostTensor>(in), "cuda_gather_rank1");
}

NNOPS_TEST(cuda_gather_int_data) {
    HostTensor d = raw({3, 4}, DataType::s64);
    HostTensor i = idxs({2, 2}, DataType::s64, {0, 2, 1, 0});
    const HostTensor in[] = {std::move(d), std::move(i)};
    run_op_both([](Backend be) { return Gather::create(GatherAttributes{1}, be); },
                std::span<const HostTensor>(in), "cuda_gather_s64");
}

// ============================================================
// GatherElements
// ============================================================

NNOPS_TEST(cuda_gather_elements_axis0) {
    HostTensor d = rnd({3, 4}, DataType::f32);
    HostTensor i = idxs({3, 4}, DataType::s64,
                        {0, 1, 2, 0, 2, 0, 1, 2, 1, 2, 0, -1});
    const HostTensor in[] = {std::move(d), std::move(i)};
    run_op_both([](Backend be) {
        return GatherElements::create(GatherElementsAttributes{0}, be);
    }, std::span<const HostTensor>(in), "cuda_gather_elements_axis0");
}

NNOPS_TEST(cuda_gather_elements_axis1) {
    HostTensor d = rnd({2, 3, 4}, DataType::f16);
    HostTensor i = idxs({2, 3, 4}, DataType::s32,
                        {0, 1, 2, 3, 3, 2, 1, 0, -1, 0, 1, 2,
                         3, 0, 2, 1, 1, 1, 1, 1, 0, 3, 2, 1});
    const HostTensor in[] = {std::move(d), std::move(i)};
    run_op_both([](Backend be) {
        return GatherElements::create(GatherElementsAttributes{2}, be);
    }, std::span<const HostTensor>(in), "cuda_gather_elements_axis2");
}

// ============================================================
// GatherND
// ============================================================

NNOPS_TEST(cuda_gather_nd_slice) {
    // K = 2 into a rank-3 tensor: each tuple picks a row, the output keeps the
    // trailing axis whole.
    HostTensor d = rnd({2, 3, 4}, DataType::f32, true);
    HostTensor i = idxs({2, 2}, DataType::s64, {0, 1, 1, -1});
    const HostTensor in[] = {std::move(d), std::move(i)};
    run_op_both([](Backend be) {
        return GatherND::create(GatherNDAttributes{0}, be);
    }, std::span<const HostTensor>(in), "cuda_gather_nd_slice");
}

NNOPS_TEST(cuda_gather_nd_scalar) {
    // K = the data rank: one element per tuple, so the output is the index shape
    // without its last axis.
    HostTensor d = raw({3, 2, 4}, DataType::s8);
    HostTensor i = idxs({2, 3}, DataType::s32, {2, 1, -1, 0, 0, 3});
    const HostTensor in[] = {std::move(d), std::move(i)};
    run_op_both([](Backend be) {
        return GatherND::create(GatherNDAttributes{0}, be);
    }, std::span<const HostTensor>(in), "cuda_gather_nd_scalar");
}

NNOPS_TEST(cuda_gather_nd_batch) {
    HostTensor d = rnd({2, 3, 4}, DataType::f16);
    HostTensor i = idxs({2, 1, 2}, DataType::s64, {2, 3, 0, 1});
    const HostTensor in[] = {std::move(d), std::move(i)};
    run_op_both([](Backend be) {
        return GatherND::create(GatherNDAttributes{1}, be);
    }, std::span<const HostTensor>(in), "cuda_gather_nd_batch");
}

// ============================================================
// ScatterElements
// ============================================================

NNOPS_TEST(cuda_scatter_elements_none) {
    for (DataType dt : {DataType::f32, DataType::f16}) {
        HostTensor d = rnd({2, 3}, dt, true);
        HostTensor i = idxs({1, 3}, DataType::s64, {1, 0, -1});
        HostTensor u = rnd({1, 3}, dt, false, 41);
        const HostTensor in[] = {std::move(d), std::move(i), std::move(u)};
        run_op_both([](Backend be) {
            return ScatterElements::create(ScatterElementsAttributes{0}, be);
        }, std::span<const HostTensor>(in), "cuda_scatter_elements_none");
    }
}

NNOPS_TEST(cuda_scatter_elements_axis1) {
    HostTensor d = rnd({2, 4}, DataType::f32);
    HostTensor i = idxs({2, 2}, DataType::s32, {3, 0, 1, 2});
    HostTensor u = rnd({2, 2}, DataType::f32, false, 43);
    const HostTensor in[] = {std::move(d), std::move(i), std::move(u)};
    run_op_both([](Backend be) {
        return ScatterElements::create(ScatterElementsAttributes{1}, be);
    }, std::span<const HostTensor>(in), "cuda_scatter_elements_axis1");
}

NNOPS_TEST(cuda_scatter_elements_add) {
    // One update per destination: accumulations are one add either side. The
    // repeated-index case runs on s32 below, where the order cannot matter.
    HostTensor d = rnd({3, 4}, DataType::f32);
    HostTensor i = idxs({3, 2}, DataType::s64, {0, 1, 2, 3, 1, 0});
    HostTensor u = rnd({3, 2}, DataType::f32, false, 47);
    const HostTensor in[] = {std::move(d), std::move(i), std::move(u)};
    run_op_both([](Backend be) {
        return ScatterElements::create(
            ScatterElementsAttributes{1, ScatterReduction::Add}, be);
    }, std::span<const HostTensor>(in), "cuda_scatter_elements_add");
}

NNOPS_TEST(cuda_scatter_elements_int_add_repeats) {
    HostTensor d = raw({3, 2}, DataType::s32);
    HostTensor i = idxs({3, 2}, DataType::s32, {0, 0, 1, 1, 0, 0});
    HostTensor u = raw({3, 2}, DataType::s32);
    const HostTensor in[] = {std::move(d), std::move(i), std::move(u)};
    run_op_both([](Backend be) {
        return ScatterElements::create(
            ScatterElementsAttributes{1, ScatterReduction::Add}, be);
    }, std::span<const HostTensor>(in), "cuda_scatter_elements_int_add");
}

NNOPS_TEST(cuda_scatter_elements_mul) {
    HostTensor d = rnd({2, 5}, DataType::f32);
    HostTensor i = idxs({2, 2}, DataType::s64, {4, 0, 2, 3});
    HostTensor u = rnd({2, 2}, DataType::f32, false, 53);
    const HostTensor in[] = {std::move(d), std::move(i), std::move(u)};
    run_op_both([](Backend be) {
        return ScatterElements::create(
            ScatterElementsAttributes{1, ScatterReduction::Mul}, be);
    }, std::span<const HostTensor>(in), "cuda_scatter_elements_mul");
}

// ============================================================
// ScatterND
// ============================================================

NNOPS_TEST(cuda_scatter_nd_none) {
    // Each tuple writes a whole [k:] slice.
    HostTensor d = rnd({3, 2, 4}, DataType::f32, true);
    HostTensor i = idxs({2, 2}, DataType::s64, {0, 1, -1, 0});
    HostTensor u = rnd({2, 4}, DataType::f32, false, 59);
    const HostTensor in[] = {std::move(d), std::move(i), std::move(u)};
    run_op_both([](Backend be) {
        return ScatterND::create(ScatterNDAttributes{ScatterReduction::None}, be);
    }, std::span<const HostTensor>(in), "cuda_scatter_nd_none");
}

NNOPS_TEST(cuda_scatter_nd_add_and_f16) {
    {
        HostTensor d = rnd({4, 3}, DataType::f32);
        HostTensor i = idxs({2, 1}, DataType::s32, {1, 3});
        HostTensor u = rnd({2, 3}, DataType::f32, false, 61);
        const HostTensor in[] = {std::move(d), std::move(i), std::move(u)};
        run_op_both([](Backend be) {
            return ScatterND::create(ScatterNDAttributes{ScatterReduction::Add}, be);
        }, std::span<const HostTensor>(in), "cuda_scatter_nd_add");
    }
    {
        HostTensor d = rnd({2, 3, 2}, DataType::f16);
        HostTensor i = idxs({2, 2}, DataType::s64, {0, 2, 1, 0});
        HostTensor u = rnd({2, 2}, DataType::f16, false, 67);
        const HostTensor in[] = {std::move(d), std::move(i), std::move(u)};
        run_op_both([](Backend be) {
            return ScatterND::create(ScatterNDAttributes{ScatterReduction::None}, be);
        }, std::span<const HostTensor>(in), "cuda_scatter_nd_f16");
    }
}


#endif  // NNOPS_HAS_CUDA
