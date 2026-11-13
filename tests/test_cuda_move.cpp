/// @file test_cuda_move.cpp
/// @brief CUDA Permute / Slice / Tile / Expand / Concat / Split / Pad against
///        the CPU backend.
///
/// Every case builds its inputs, runs the operator on both backends through the
/// same descriptors (run_op_both), and compares each output byte for byte. Byte
/// equality is the right test here and a stronger one than the other CUDA suites
/// use: a movement operator never touches a value, so the two backends have to
/// produce identical bytes, not merely close ones — and that covers f16 and the
/// integer dtypes without a widening pass.
///
/// Inputs are channel-first and, where the operator accepts them, pitched: a
/// padded row is the case a stride-folded walk gets wrong, so several cases hand
/// the kernels a 32 B pitch on the source.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/concat.hpp"
#include "nnops/ops/pad.hpp"
#include "nnops/ops/permute.hpp"
#include "nnops/ops/slice.hpp"
#include "nnops/ops/split.hpp"
#include "nnops/ops/tile.hpp"

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

/// Arbitrary bytes for the integer dtypes the random builder does not fill: a
/// movement operator only copies them, so the values need no meaning.
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

}  // namespace

// ============================================================
// Permute
// ============================================================

NNOPS_TEST(cuda_permute_last_two) {
    // The one case with a kernel of its own (the tiled transpose), on a padded
    // source so the row pitch is part of the address.
    for (DataType dt : {DataType::f32, DataType::f16}) {
        HostTensor a = rnd({2, 3, 4, 5}, dt, true);
        const HostTensor in[] = {std::move(a)};
        run_op_both([](Backend be) { return Permute::create(PermuteAttributes{{0, 1, 3, 2}}, be); },
                 std::span<const HostTensor>(in), "cuda_permute_last_two");
    }
}

NNOPS_TEST(cuda_permute_general) {
    {
        HostTensor a = rnd({2, 3, 4, 5});
        const HostTensor in[] = {std::move(a)};
        run_op_both([](Backend be) { return Permute::create(PermuteAttributes{{2, 0, 3, 1}}, be); },
                 std::span<const HostTensor>(in), "cuda_permute_general");
    }
    {
        HostTensor a = rnd({3, 5, 7}, DataType::f16);
        const HostTensor in[] = {std::move(a)};
        run_op_both([](Backend be) { return Permute::create(PermuteAttributes{{2, 1, 0}}, be); },
                 std::span<const HostTensor>(in), "cuda_permute_general");
    }
    {
        // A reversed permutation: every axis moves.
        HostTensor a = rnd({2, 3, 4, 5});
        const HostTensor in[] = {std::move(a)};
        run_op_both([](Backend be) { return Permute::create(PermuteAttributes{{3, 2, 1, 0}}, be); },
                 std::span<const HostTensor>(in), "cuda_permute_general");
    }
}

NNOPS_TEST(cuda_permute_rank2_and_1) {
    {
        HostTensor a = rnd({3, 5}, DataType::f32, true);
        const HostTensor in[] = {std::move(a)};
        run_op_both([](Backend be) { return Permute::create(PermuteAttributes{{1, 0}}, be); },
                 std::span<const HostTensor>(in), "cuda_permute_rank2");
    }
    {
        HostTensor a = rnd({7}, DataType::f16);
        const HostTensor in[] = {std::move(a)};
        run_op_both([](Backend be) { return Permute::create(PermuteAttributes{{0}}, be); },
                 std::span<const HostTensor>(in), "cuda_permute_rank1");
    }
}

// ============================================================
// Slice
// ============================================================

NNOPS_TEST(cuda_slice_basic) {
    HostTensor a = rnd({4, 6, 8});
    const HostTensor in[] = {std::move(a)};
    run_op_both([](Backend be) {
        return Slice::create(SliceAttributes{{1, 2}, {3, 7}, {1, 2}, {}}, be);
    }, std::span<const HostTensor>(in), "cuda_slice_basic");
}

NNOPS_TEST(cuda_slice_step) {
    HostTensor a = rnd({3, 8, 10}, DataType::f16);
    const HostTensor in[] = {std::move(a)};
    run_op_both([](Backend be) {
        return Slice::create(SliceAttributes{{0}, {10}, {2}, {2}}, be);
    }, std::span<const HostTensor>(in), "cuda_slice_step");
}

NNOPS_TEST(cuda_slice_negative_and_clamp) {
    HostTensor a = rnd({2, 9}, DataType::f32, true);
    const HostTensor in[] = {std::move(a)};
    // -3 and -1 wrap; the end past the axis clamps.
    run_op_both([](Backend be) {
        return Slice::create(SliceAttributes{{-3}, {-1}, {1}, {}}, be);
    }, std::span<const HostTensor>(in), "cuda_slice_negative");
    run_op_both([](Backend be) {
        return Slice::create(SliceAttributes{{0, 1}, {50, 4}, {0, 1}, {}}, be);
    }, std::span<const HostTensor>(in), "cuda_slice_clamp");
}

// ============================================================
// Tile and Expand
// ============================================================

NNOPS_TEST(cuda_tile_basic) {
    for (DataType dt : {DataType::f32, DataType::f16}) {
        HostTensor a = rnd({2, 3}, dt);
        const HostTensor in[] = {std::move(a)};
        run_op_both([](Backend be) { return Tile::create(TileAttributes{{2, 3}}, be); },
                 std::span<const HostTensor>(in), "cuda_tile_basic");
    }
    {
        HostTensor a = rnd({1, 2, 3}, DataType::f32, true);
        const HostTensor in[] = {std::move(a)};
        run_op_both([](Backend be) { return Tile::create(TileAttributes{{2, 1, 2}}, be); },
                 std::span<const HostTensor>(in), "cuda_tile_basic");
    }
}

NNOPS_TEST(cuda_tile_last_axis_only) {
    HostTensor a = rnd({3, 4}, DataType::f32);
    const HostTensor in[] = {std::move(a)};
    run_op_both([](Backend be) { return Tile::create(TileAttributes{{1, 3}}, be); },
             std::span<const HostTensor>(in), "cuda_tile_last");
    {
        HostTensor b = rnd({5}, DataType::f16);
        const HostTensor inb[] = {std::move(b)};
        run_op_both([](Backend be) { return Tile::create(TileAttributes{{4}}, be); },
                 std::span<const HostTensor>(inb), "cuda_tile_rank1");
    }
}

NNOPS_TEST(cuda_tile_int_dtypes) {
    HostTensor a = raw({2, 3, 4}, DataType::s32);
    const HostTensor in[] = {std::move(a)};
    run_op_both([](Backend be) { return Tile::create(TileAttributes{{1, 2, 3}}, be); },
             std::span<const HostTensor>(in), "cuda_tile_s32");
}

NNOPS_TEST(cuda_expand_broadcast) {
    HostTensor a = rnd({1, 3, 1}, DataType::f32);
    const HostTensor in[] = {std::move(a)};
    run_op_both([](Backend be) { return Expand::create(ExpandAttributes{{2, 3, 4}}, be); },
             std::span<const HostTensor>(in), "cuda_expand_broadcast");
}

NNOPS_TEST(cuda_expand_rank_increase) {
    {
        HostTensor a = rnd({3}, DataType::f16);
        const HostTensor in[] = {std::move(a)};
        run_op_both([](Backend be) { return Expand::create(ExpandAttributes{{2, 2, 3}}, be); },
                 std::span<const HostTensor>(in), "cuda_expand_rank");
    }
    {
        // -1 keeps the input's extent on that axis.
        HostTensor a = rnd({2, 4}, DataType::f32);
        const HostTensor in[] = {std::move(a)};
        run_op_both([](Backend be) { return Expand::create(ExpandAttributes{{3, 2, -1}}, be); },
                 std::span<const HostTensor>(in), "cuda_expand_minus_one");
    }
    {
        HostTensor a = rnd({4, 1, 5}, DataType::f32, true);
        const HostTensor in[] = {std::move(a)};
        run_op_both([](Backend be) { return Expand::create(ExpandAttributes{{4, 3, 5}}, be); },
                 std::span<const HostTensor>(in), "cuda_expand_mid_axis");
    }
}

// ============================================================
// Concat and Split
// ============================================================

NNOPS_TEST(cuda_concat_axis0) {
    for (DataType dt : {DataType::f32, DataType::f16}) {
        HostTensor a = rnd({2, 3}, dt, true);
        HostTensor b = rnd({3, 3}, dt);
        const HostTensor in[] = {std::move(a), std::move(b)};
        run_op_both([](Backend be) { return Concat::create(ConcatAttributes{0}, be); },
                 std::span<const HostTensor>(in), "cuda_concat_axis0");
    }
}

NNOPS_TEST(cuda_concat_axis2) {
    HostTensor a = rnd({2, 3, 4});
    HostTensor b = rnd({2, 3, 2});
    HostTensor c = rnd({2, 3, 1});
    const HostTensor in[] = {std::move(a), std::move(b), std::move(c)};
    run_op_both([](Backend be) { return Concat::create(ConcatAttributes{2}, be); },
             std::span<const HostTensor>(in), "cuda_concat_axis2");
}

NNOPS_TEST(cuda_concat_negative_axis_and_rank1) {
    {
        HostTensor a = rnd({2, 3}, DataType::f16);
        HostTensor b = rnd({2, 4}, DataType::f16);
        const HostTensor in[] = {std::move(a), std::move(b)};
        run_op_both([](Backend be) { return Concat::create(ConcatAttributes{-1}, be); },
                 std::span<const HostTensor>(in), "cuda_concat_neg_axis");
    }
    {
        // Rank 1 is the plain-copy path: no axis to band.
        HostTensor a = rnd({3}, DataType::f32);
        HostTensor b = rnd({2}, DataType::f32);
        const HostTensor in[] = {std::move(a), std::move(b)};
        run_op_both([](Backend be) { return Concat::create(ConcatAttributes{0}, be); },
                 std::span<const HostTensor>(in), "cuda_concat_rank1");
    }
}

NNOPS_TEST(cuda_split_axis0) {
    HostTensor a = rnd({5, 4});
    const HostTensor in[] = {std::move(a)};
    run_op_both([](Backend be) {
        SplitAttributes attrs;
        attrs.axis = 0;
        attrs.split = {2, 3};
        return Split::create(attrs, be);
    }, std::span<const HostTensor>(in), "cuda_split_axis0");
}

NNOPS_TEST(cuda_split_equal_and_negative_axis) {
    HostTensor a = rnd({2, 6}, DataType::f16);
    const HostTensor in[] = {std::move(a)};
    run_op_both([](Backend be) {
        SplitAttributes attrs;
        attrs.axis = 1;
        attrs.split = {1, 2, 3};
        return Split::create(attrs, be);
    }, std::span<const HostTensor>(in), "cuda_split_axis1");

    run_op_both([](Backend be) {
        SplitAttributes attrs;
        attrs.axis = -1;
        attrs.num_outputs = 2;
        return Split::create(attrs, be);
    }, std::span<const HostTensor>(in), "cuda_split_equal");
}

// ============================================================
// Pad
// ============================================================

NNOPS_TEST(cuda_pad_constant) {
    for (DataType dt : {DataType::f32, DataType::f16}) {
        HostTensor a = rnd({2, 3}, dt);
        const HostTensor in[] = {std::move(a)};
        run_op_both([](Backend be) {
            PadAttributes attrs;
            attrs.pads = {1, 0, 1, 2};
            attrs.constant_value = -1.5f;
            return Pad::create(attrs, be);
        }, std::span<const HostTensor>(in), "cuda_pad_constant");
    }
}

NNOPS_TEST(cuda_pad_edge) {
    HostTensor a = rnd({2, 3, 4}, DataType::f32);
    const HostTensor in[] = {std::move(a)};
    run_op_both([](Backend be) {
        PadAttributes attrs;
        attrs.pads = {0, 2, 1, 0, 1, 3};
        attrs.mode = PadAttributes::Mode::Edge;
        return Pad::create(attrs, be);
    }, std::span<const HostTensor>(in), "cuda_pad_edge");
}

NNOPS_TEST(cuda_pad_reflect) {
    HostTensor a = rnd({2, 4, 5}, DataType::f16);
    const HostTensor in[] = {std::move(a)};
    run_op_both([](Backend be) {
        PadAttributes attrs;
        attrs.pads = {0, 2, 1, 0, 3, 2};
        attrs.mode = PadAttributes::Mode::Reflect;
        return Pad::create(attrs, be);
    }, std::span<const HostTensor>(in), "cuda_pad_reflect");
}

NNOPS_TEST(cuda_pad_wrap) {
    HostTensor a = rnd({3, 4}, DataType::f32);
    const HostTensor in[] = {std::move(a)};
    run_op_both([](Backend be) {
        PadAttributes attrs;
        attrs.pads = {5, 1, 2, 6};
        attrs.mode = PadAttributes::Mode::Wrap;
        return Pad::create(attrs, be);
    }, std::span<const HostTensor>(in), "cuda_pad_wrap");
}

NNOPS_TEST(cuda_pad_crop_and_rank1) {
    {
        HostTensor a = rnd({2, 6, 6}, DataType::f32, true);
        const HostTensor in[] = {std::move(a)};
        run_op_both([](Backend be) {
            PadAttributes attrs;
            attrs.pads = {0, -1, -2, 0, 2, 1};
            attrs.constant_value = 0.25f;
            return Pad::create(attrs, be);
        }, std::span<const HostTensor>(in), "cuda_pad_crop");
    }
    {
        HostTensor a = raw({5}, DataType::s8);
        const HostTensor in[] = {std::move(a)};
        run_op_both([](Backend be) {
            PadAttributes attrs;
            attrs.pads = {2, 3};
            attrs.constant_value = 7.0f;
            return Pad::create(attrs, be);
        }, std::span<const HostTensor>(in), "cuda_pad_rank1_s8");
    }
}

NNOPS_TEST(cuda_pad_int_dtypes) {
    for (DataType dt : {DataType::s32, DataType::u8, DataType::s64}) {
        HostTensor a = raw({2, 3}, dt);
        const HostTensor in[] = {std::move(a)};
        run_op_both([](Backend be) {
            PadAttributes attrs;
            attrs.pads = {1, 1, 1, 1};
            attrs.mode = PadAttributes::Mode::Edge;
            return Pad::create(attrs, be);
        }, std::span<const HostTensor>(in), "cuda_pad_int");
    }
}

#endif  // NNOPS_HAS_CUDA
