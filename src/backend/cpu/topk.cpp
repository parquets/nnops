/// @file topk.cpp
/// @brief CPU TopK — a k-sized heap per row, with a thread-split axis for a
///        single short-and-wide row.
///
/// Selection keeps the best k seen so far in a binary heap whose root is the
/// *worst* kept candidate, so a new element is compared against one value and
/// replaces it in O(log k). The old form re-sorted all k candidates on every
/// improvement — O(n·k log k) with a sort call per element.
///
/// The order is (value, index): a tie keeps the earlier index, which makes the
/// result deterministic regardless of the scan (ONNX leaves ties unspecified).
///
/// The pool splits over the outer axes. When there are fewer outer positions
/// than workers — top-k on one row of a vocabulary logit — the row itself is
/// split: each worker keeps the best k of its slice, and the slices merge into
/// the final k. Both phases use the same order, so the split does not change
/// which elements win.
///
/// Supports dense and spatial 2D/3D layouts. Not for packed layouts. The input
/// is read through its own strides, so a padded pitch is the same walk with a
/// different offset group (see axis_walk.hpp); the outputs are dense, which is
/// what their descriptors specify.

#include "nnops/ops/topk.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/axis_walk.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "common/dtype_dispatch.hpp"
#include "common/memory_pool.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace nnops::backend::cpu {

using nnops::simd::s_load;
using nnops::simd::s_store;

namespace {

struct Pair {
    float value;
    int64_t index;
};

/// Largest k the axis split merges on the stack.
constexpr int64_t kMaxMergeK = 64;

/// Strict "a is the better candidate" order. The heap keeps its *maximum* at
/// the root, and under this order the maximum is the worst kept candidate — the
/// one a new element has to beat.
struct Better {
    bool is_max;
    bool operator()(const Pair& a, const Pair& b) const
    {
        if (a.value != b.value) {
            return is_max ? (a.value > b.value) : (a.value < b.value);
        }
        return a.index < b.index; // a tie keeps the earlier index
    }
};

/// The best `k` of `count` elements at `row + j * axis_stride`, into `out`.
/// `index_base` makes the recorded indices absolute for a sliced row.
template <typename T>
void select_best(const T* row, int64_t axis_stride, int64_t count, int64_t k,
                 int64_t index_base, const Better& better, Pair* out)
{
    int64_t j = 0;
    for (; j < k; ++j) {
        out[j] = {s_load(row + j * axis_stride), index_base + j};
    }
    std::make_heap(out, out + k, better);

    if (k == 1) {
        // The root is the only candidate: one compare and no heap traffic.
        for (; j < count; ++j) {
            const Pair cand{s_load(row + j * axis_stride), index_base + j};
            if (better(cand, out[0])) {
                out[0] = cand;
            }
        }
        return;
    }

    for (; j < count; ++j) {
        const Pair cand{s_load(row + j * axis_stride), index_base + j};
        if (better(cand, out[0])) {
            std::pop_heap(out, out + k, better);
            out[k - 1] = cand;
            std::push_heap(out, out + k, better);
        }
    }
}

} // anonymous namespace

template <typename T>
void topk_impl(const TopKAttributes& attrs, TensorView& values_out, TensorView& indices_out,
               std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& in = inputs[0];
    const int64_t rank = in.rank();
    const int64_t ax = (attrs.axis < 0) ? attrs.axis + rank : attrs.axis;
    const int64_t axis_dim = in.shape(ax);
    const int64_t k = attrs.k;

    NNOPS_ASSERT(k >= 1);
    NNOPS_ASSERT(k <= axis_dim);
    NNOPS_ASSERT(ax >= 0 && ax < rank);

    const bool is_max = (attrs.type == TopKType::Max);
    const bool sorted_opt = attrs.sorted;
    const Better better{is_max};

    int64_t outer_dims = 1;
    for (int64_t d = 0; d < ax; ++d) {
        outer_dims *= in.shape(d);
    }
    int64_t inner_size = 1;
    for (int64_t d = ax + 1; d < rank; ++d) {
        inner_size *= in.shape(d);
    }

    const auto* i_ptr = in.ptr<T>();
    auto* v_ptr = values_out.ptr<T>();
    auto* idx_ptr = indices_out.ptr<int64_t>();

    const int64_t axis_stride = in.stride_elems(ax);

    // The dims on either side of the axis, so a pitched input addresses its
    // rows and its inner tail instead of flattening them. The two outputs get
    // the same treatment: a caller may hand either one a row pad, and the k
    // entries along the axis are contiguous within the row.
    detail::AxisGroup pre;
    detail::AxisGroup post;
    if (ax > 0) {
        detail::axis_fill_group(pre, in, 0, ax, in.stride_elems(ax - 1));
    } else {
        detail::axis_fill_group(pre, in, 0, 0, 0);
    }
    detail::axis_fill_group_unit(post, in, ax + 1, rank);

    detail::AxisGroup vpre, vpost, ipre, ipost;
    if (ax > 0) {
        detail::axis_fill_group(vpre, values_out, 0, ax, values_out.stride_elems(ax - 1));
        detail::axis_fill_group(ipre, indices_out, 0, ax, indices_out.stride_elems(ax - 1));
    } else {
        detail::axis_fill_group(vpre, values_out, 0, 0, 0);
        detail::axis_fill_group(ipre, indices_out, 0, 0, 0);
    }
    detail::axis_fill_group_unit(vpost, values_out, ax + 1, rank);
    detail::axis_fill_group_unit(ipost, indices_out, ax + 1, rank);
    // The k entries are consecutive only when the axis is the last one; else
    // they step by the output's stride on that axis.
    const int64_t v_k_stride = values_out.stride_elems(ax);
    const int64_t i_k_stride = indices_out.stride_elems(ax);

    const auto out_base = [&](int64_t outer, int64_t inner) {
        return std::pair<int64_t, int64_t>{
            detail::axis_group_offset(outer, vpre) + detail::axis_group_offset(inner, vpost),
            detail::axis_group_offset(outer, ipre) + detail::axis_group_offset(inner, ipost)};
    };

    const auto write_out = [&](int64_t v_base, int64_t i_base, Pair* best) {
        if (sorted_opt) {
            std::sort(best, best + k, better);
        }
        for (int64_t j = 0; j < k; ++j) {
            s_store(v_ptr + v_base + j * v_k_stride, best[j].value);
            idx_ptr[i_base + j * i_k_stride] = best[j].index;
        }
    };

    // The row-serial route: one task per outer position, `inner_size` rows each.
    const int64_t units = outer_dims * inner_size;

    int threads = 1;
    if (ctx.cpu.num_threads) {
        threads = std::max(1, ctx.cpu.num_threads());
    }

    // Split the axis when the outer space cannot fill the pool. Every chunk must
    // be at least k long, since a chunk's local pass seeds its heap from k
    // elements.
    const int64_t min_chunk = std::max<int64_t>(4096, k);
    const int64_t want = (units >= threads) ? 1 : std::min<int64_t>(threads / units, 64);
    const int64_t chunks = std::max<int64_t>(1, std::min(want, axis_dim / min_chunk));

    if (chunks > 1 && k <= kMaxMergeK && units * chunks * k <= (int64_t{1} << 20)) {
        PoolArray<Pair> partial(static_cast<size_t>(units * chunks * k));
        // Floor division keeps the last chunk at least as long as the others.
        const int64_t chunk_len = axis_dim / chunks;

        ctx.cpu.run(0, units * chunks, [&](int64_t t) {
            const int64_t unit = t / chunks;
            const int64_t chunk = t % chunks;
            const int64_t m0 = chunk * chunk_len;
            const int64_t m1 = (chunk + 1 == chunks) ? axis_dim : (m0 + chunk_len);

            const int64_t base = detail::axis_group_offset(unit / inner_size, pre) +
                                 detail::axis_group_offset(unit % inner_size, post) +
                                 m0 * axis_stride;
            select_best(i_ptr + base, axis_stride, m1 - m0, k, m0, better,
                        partial.data() + (t)*k);
        });

        ctx.cpu.run(0, units, [&](int64_t unit) {
            Pair heap[kMaxMergeK];
            for (int64_t c = 0; c < chunks; ++c) {
                const Pair* src = partial.data() + (unit * chunks + c) * k;
                if (c == 0) {
                    for (int64_t j = 0; j < k; ++j) {
                        heap[j] = src[j];
                    }
                    std::make_heap(heap, heap + k, better);
                    continue;
                }
                for (int64_t j = 0; j < k; ++j) {
                    if (better(src[j], heap[0])) {
                        std::pop_heap(heap, heap + k, better);
                        heap[k - 1] = src[j];
                        std::push_heap(heap, heap + k, better);
                    }
                }
            }
            const auto [v_base, i_base] = out_base(unit / inner_size, unit % inner_size);
            write_out(v_base, i_base, heap);
        });
        return;
    }

    auto process_outer = [&](int64_t outer) {
        // Per-thread heap to avoid allocation in hot loop
        PoolArray<Pair> heap(static_cast<size_t>(k));

        for (int64_t inner = 0; inner < inner_size; ++inner) {
            const int64_t in_base = (outer > 0 ? detail::axis_group_offset(outer, pre) : 0) +
                                    (inner > 0 ? detail::axis_group_offset(inner, post) : 0);

            select_best<T>(i_ptr + in_base, axis_stride, axis_dim, k, 0, better, heap.data());

            const auto [v_base, i_base] = out_base(outer, inner);
            write_out(v_base, i_base, heap.data());
        }
    };

    ctx.cpu.run(0, outer_dims, process_outer);
}

Status topk_cpu(const TopKAttributes& attrs, TensorView& values, TensorView& indices,
                std::span<const TensorView> inputs, const ComputeContext& ctx, void* /*workspace*/)
{
    if (!dispatch_f32_f16(inputs[0].data_type(), "topk_cpu", [&](auto tag) {
            using T = typename decltype(tag)::type;
            topk_impl<T>(attrs, values, indices, inputs, ctx);
        })) {
        return Status::unsupported;
    }
    return Status::ok;
}

} // namespace nnops::backend::cpu
