/// @file reduce.cpp
/// @brief SIMD-optimized CPU implementation of the Reduce operator.
///
/// f32 and f16; channel-first and channel-last layouts; one axis or many.
/// The reduction is driven by the logical axes; stride_elems() turns them into
/// physical offsets, so the same axis means the same thing in either layout.
///
/// A multi-axis set is split into maximal runs of adjacent axes that are dense
/// in memory (stride(d) == shape(d+1) * stride(d+1)); each run becomes one
/// (dim = product, stride = stride_elems(last axis)) pair, and a flat index k
/// over the run maps to base + k * stride.
///
/// Single-axis dispatch (unchanged), in order:
///   1. axis==rank-1            -> kernel::reduce_process_contiguous_row
///   2. inner-contiguous        -> kernel::reduce_process_inner_contiguous_block
///   3. general                 -> scalar strided walk over (outer x inner)
/// Channel-last: axis==1 (C) is the contiguous case and runs one row reduction
/// per spatial position; every other axis walks the strided reduce axis.
///
/// Multi-axis dispatch:
///   1. one dense run + no kept dims trailing the run -> row kernel per outer
///      block
///   2. one dense run + a contiguous kept tail       -> inner block kernel per
///      (outer, lane chunk)
///   3. otherwise                                    -> an odometer over the
///      runs, vectorized across the last kept axis when it steps by one

#include "nnops/ops/reduce.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "simd_kernel/simd_reduce.hpp"
#include "simd_kernel/simd_reduce_primitive.hpp"
#include "common/index.hpp"
#include "common/dtype_dispatch.hpp"

#include <algorithm>
#include <cfloat>
#include <vector>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

constexpr int64_t kLaneChunk = 256;

/// One reduced run: `dim` elements spaced `stride` apart.
struct ReduceRun {
    int64_t dim;
    int64_t stride;
};

/// Collapse the sorted axes into maximal dense runs.
int64_t build_reduced_runs(const TensorView& input,
                           const int64_t* axes,
                           int64_t n_axes,
                           ReduceRun* runs)
{
    int64_t n = 0;
    int64_t dim = input.shape(axes[0]);
    int64_t run_stride = input.stride_elems(axes[0]);
    for (int64_t i = 1; i < n_axes; ++i) {
        const int64_t d = axes[i];
        if (d == axes[i - 1] + 1 &&
            run_stride == input.shape(d) * input.stride_elems(d)) {
            dim *= input.shape(d);
            run_stride = input.stride_elems(d);
            continue;
        }
        runs[n++] = {dim, run_stride};
        dim = input.shape(d);
        run_stride = input.stride_elems(d);
    }
    runs[n++] = {dim, run_stride};
    return n;
}

template <typename T>
void reduce_channel_last(const ReduceAttributes& attrs,
                         TensorView& output,
                         const TensorView& input,
                         int64_t axis,
                         const ComputeContext& ctx)
{
    const int64_t rank = input.rank();
    const int64_t rank_out = output.rank();
    const int64_t D = input.shape(axis);
    const int64_t axis_stride = input.stride_elems(axis);
    const bool keepdims = attrs.keepdims;

    int64_t num_positions = 1;
    for (int64_t d = 0; d < rank; ++d) {
        if (d != axis) { num_positions *= input.shape(d); }
    }

    const T* x_ptr = input.ptr<T>();
    T* y_ptr = output.ptr<T>();
    const float inv_D = 1.0f / static_cast<float>(D);

    // axis==1 is C: contiguous in every channel-last tensor.
    if (axis == 1) {
        const auto process_pos = [&](int64_t s) {
            const int64_t in_off =
                decompose_flat_offset(s, input, 0, rank - 1, /*skip_dim=*/1);
            const int64_t out_off = decompose_flat_offset(
                s, output, 0, rank_out - 1, keepdims ? 1 : -1);
            float result = 0.0f;
            kernel::reduce_process_contiguous_row<T>(
                x_ptr + in_off, &result, D, attrs.type);
            s_store(&y_ptr[out_off], result);
        };
        ctx.cpu.run(0, num_positions, process_pos);
        return;
    }

    const auto process_pos = [&](int64_t s) {
        const int64_t in_off =
            decompose_flat_offset(s, input, 0, rank - 1, axis);
        const int64_t out_off = decompose_flat_offset(
            s, output, 0, rank_out - 1, keepdims ? axis : -1);

        switch (attrs.type) {
        case ReduceType::Sum:
        case ReduceType::Mean: {
            float sum = 0.0f;
            for (int64_t k = 0; k < D; ++k) {
                sum += s_load(&x_ptr[in_off + k * axis_stride]);
            }
            const float result =
                (attrs.type == ReduceType::Mean) ? sum * inv_D : sum;
            s_store(&y_ptr[out_off], result);
            break;
        }
        case ReduceType::Max: {
            float best = -std::numeric_limits<float>::infinity();
            for (int64_t k = 0; k < D; ++k) {
                const float xv = s_load(&x_ptr[in_off + k * axis_stride]);
                if (xv > best) { best = xv; }
            }
            s_store(&y_ptr[out_off], best);
            break;
        }
        case ReduceType::Min: {
            float best = std::numeric_limits<float>::infinity();
            for (int64_t k = 0; k < D; ++k) {
                const float xv = s_load(&x_ptr[in_off + k * axis_stride]);
                if (xv < best) { best = xv; }
            }
            s_store(&y_ptr[out_off], best);
            break;
        }
        }
    };
    ctx.cpu.run(0, num_positions, process_pos);
}

template <typename T>
void reduce_contiguous_simd(const ReduceAttributes& attrs,
                            TensorView& output,
                            const TensorView& input,
                            int64_t axis,
                            const ComputeContext& ctx)
{
    const int64_t rank = input.rank();
    NNOPS_ASSERT(axis == rank - 1);

    const int64_t num_rows = [&]() {
        int64_t n = 1;
        for (int64_t d = 0; d < axis; ++d) { n *= input.shape(d); }
        return n;
    }();
    const int64_t norm_size = input.shape(axis);
    const int64_t row_stride = input.row_stride_elems();

    const T* x_ptr = input.ptr<T>();
    T* y_ptr = output.ptr<T>();

    auto process_row = [&](int64_t row) {
        float result = 0.0f;
        kernel::reduce_process_contiguous_row<T>(
            x_ptr + row * row_stride, &result, norm_size, attrs.type);
        s_store(&y_ptr[row], result);
    };

    ctx.cpu.run(0, num_rows, process_row);
}

template <typename T>
void reduce_inner_contiguous_simd(const ReduceAttributes& attrs,
                                   TensorView& output,
                                   const TensorView& input,
                                   int64_t axis,
                                   int64_t num_outer,
                                   int64_t reduce_size,
                                   int64_t num_inner,
                                   const ComputeContext& ctx)
{
    const int64_t axis_stride = input.stride_elems(axis);
    const int64_t x_outer_stride = (axis > 0) ? input.stride_elems(axis - 1) : 0;

    const int64_t y_outer_stride = (axis > 0)
        ? output.stride_elems(axis - 1) : num_inner;

    const T* x_ptr = input.ptr<T>();
    T* y_ptr = output.ptr<T>();
    constexpr int L = simd_lane_for<T>;
    const float inv_reduce = 1.0f / static_cast<float>(reduce_size);

    auto process_outer = [&](int64_t outer) {
        const T* x_base = x_ptr + outer * x_outer_stride;
        T* y_base = y_ptr + outer * y_outer_stride;

        int64_t inner = 0;

        kernel::reduce_process_inner_contiguous_block<T>(
            x_base, y_base, 0, num_inner, reduce_size, axis_stride,
            attrs.type, inv_reduce);

        // Scalar tail past the SIMD body.
        for (inner = (num_inner / L) * L; inner < num_inner; ++inner) {
            switch (attrs.type) {
            case ReduceType::Sum:
            case ReduceType::Mean: {
                float sum = 0.0f;
                for (int64_t k = 0; k < reduce_size; ++k) {
                    sum += s_load(&x_base[k * axis_stride + inner]);
                }
                float result = (attrs.type == ReduceType::Mean)
                    ? sum * inv_reduce : sum;
                s_store(&y_base[inner], result);
                break;
            }
            case ReduceType::Max: {
                float best = -std::numeric_limits<float>::infinity();
                for (int64_t k = 0; k < reduce_size; ++k) {
                    float xv = s_load(&x_base[k * axis_stride + inner]);
                    if (xv > best) { best = xv; }
                }
                s_store(&y_base[inner], best);
                break;
            }
            case ReduceType::Min: {
                float best = std::numeric_limits<float>::infinity();
                for (int64_t k = 0; k < reduce_size; ++k) {
                    float xv = s_load(&x_base[k * axis_stride + inner]);
                    if (xv < best) { best = xv; }
                }
                s_store(&y_base[inner], best);
                break;
            }
            }
        }
    };

    ctx.cpu.run(0, num_outer, process_outer);
}

template <typename T>
void reduce_general_scalar(const ReduceAttributes& attrs,
                           TensorView& output,
                           const TensorView& input,
                           int64_t axis,
                           int64_t num_outer,
                           int64_t reduce_size,
                           int64_t num_inner,
                           const ComputeContext& ctx)
{
    const int64_t rank = input.rank();
    const int64_t rank_out = output.rank();

    const PoolArray<int64_t> inner_offsets =
        build_offsets(input, axis + 1, rank - 1, num_inner);

    const int64_t y_inner_start = attrs.keepdims ? axis + 1 : axis;
    const PoolArray<int64_t> y_inner_offsets =
        build_offsets(output, y_inner_start, rank_out - 1, num_inner);

    const int64_t x_outer_stride = (axis > 0) ? input.stride_elems(axis - 1) : 0;
    const int64_t y_outer_stride = (axis > 0) ? output.stride_elems(axis - 1) : 0;
    const int64_t axis_stride = input.stride_elems(axis);

    const T* x_ptr = input.ptr<T>();
    T* y_ptr = output.ptr<T>();
    const float inv_reduce = 1.0f / static_cast<float>(reduce_size);

    auto process_outer = [&](int64_t outer) {
        const int64_t x_row_base = outer * x_outer_stride;
        const int64_t y_row_base = outer * y_outer_stride;

        for (int64_t inner = 0; inner < num_inner; ++inner) {
            const int64_t in_off = x_row_base + inner_offsets[static_cast<size_t>(inner)];
            const int64_t out_off = y_row_base + y_inner_offsets[static_cast<size_t>(inner)];

            switch (attrs.type) {
            case ReduceType::Sum:
            case ReduceType::Mean: {
                float sum = 0.0f;
                for (int64_t k = 0; k < reduce_size; ++k) {
                    sum += s_load(&x_ptr[in_off + k * axis_stride]);
                }
                float result = (attrs.type == ReduceType::Mean)
                    ? sum * inv_reduce : sum;
                s_store(&y_ptr[out_off], result);
                break;
            }
            case ReduceType::Max: {
                float best = -std::numeric_limits<float>::infinity();
                for (int64_t k = 0; k < reduce_size; ++k) {
                    float xv = s_load(&x_ptr[in_off + k * axis_stride]);
                    if (xv > best) { best = xv; }
                }
                s_store(&y_ptr[out_off], best);
                break;
            }
            case ReduceType::Min: {
                float best = std::numeric_limits<float>::infinity();
                for (int64_t k = 0; k < reduce_size; ++k) {
                    float xv = s_load(&x_ptr[in_off + k * axis_stride]);
                    if (xv < best) { best = xv; }
                }
                s_store(&y_ptr[out_off], best);
                break;
            }
            }
        }
    };

    ctx.cpu.run(0, num_outer, process_outer);
}

/// One dense reduced run with no kept tail: one row reduction per outer block.
template <typename T>
void reduce_multi_fast_row(const ReduceAttributes& attrs,
                           TensorView& output,
                           const TensorView& input,
                           int64_t num_outer,
                           int64_t outer_stride,
                           int64_t reduce_size,
                           const ComputeContext& ctx)
{
    const int64_t rank_out = output.rank();
    const T* x_ptr = input.ptr<T>();
    T* y_ptr = output.ptr<T>();

    ctx.cpu.run(0, num_outer, [&](int64_t outer) {
        const int64_t out_off =
            decompose_flat_offset(outer, output, 0, rank_out - 1);
        float result = 0.0f;
        kernel::reduce_process_contiguous_row<T>(
            x_ptr + outer * outer_stride, &result, reduce_size, attrs.type);
        s_store(&y_ptr[out_off], result);
    });
}

/// One dense reduced run followed by a contiguous kept tail: the reduce stride
/// equals the lane count, so the inner block kernel works lane by lane.
template <typename T>
void reduce_multi_fast_lanes(const ReduceAttributes& attrs,
                             TensorView& output,
                             const TensorView& input,
                             int64_t num_outer,
                             int64_t outer_stride,
                             int64_t reduce_size,
                             int64_t num_inner,
                             const ComputeContext& ctx)
{
    const T* x_ptr = input.ptr<T>();
    T* y_ptr = output.ptr<T>();
    constexpr int L = simd_lane_for<T>;
    const float inv_reduce = 1.0f / static_cast<float>(reduce_size);
    const int64_t n_chunks = (num_inner + kLaneChunk - 1) / kLaneChunk;

    ctx.cpu.run(0, num_outer * n_chunks, [&](int64_t task) {
        const int64_t outer = task / n_chunks;
        const int64_t chunk = task % n_chunks;
        const int64_t lane0 = chunk * kLaneChunk;
        const int64_t lane1 = std::min(lane0 + kLaneChunk, num_inner);

        const T* x_base = x_ptr + outer * outer_stride;
        T* y_base = y_ptr + outer * num_inner;

        kernel::reduce_process_inner_contiguous_block<T>(
            x_base, y_base, lane0, lane1, reduce_size, num_inner,
            attrs.type, inv_reduce);

        const int64_t vec_end = lane0 + ((lane1 - lane0) / L) * L;
        for (int64_t inner = vec_end; inner < lane1; ++inner) {
            switch (attrs.type) {
            case ReduceType::Sum:
            case ReduceType::Mean: {
                float sum = 0.0f;
                for (int64_t k = 0; k < reduce_size; ++k) {
                    sum += s_load(&x_base[k * num_inner + inner]);
                }
                float result = (attrs.type == ReduceType::Mean)
                    ? sum * inv_reduce : sum;
                s_store(&y_base[inner], result);
                break;
            }
            case ReduceType::Max: {
                float best = -std::numeric_limits<float>::infinity();
                for (int64_t k = 0; k < reduce_size; ++k) {
                    float xv = s_load(&x_base[k * num_inner + inner]);
                    if (xv > best) { best = xv; }
                }
                s_store(&y_base[inner], best);
                break;
            }
            case ReduceType::Min: {
                float best = std::numeric_limits<float>::infinity();
                for (int64_t k = 0; k < reduce_size; ++k) {
                    float xv = s_load(&x_base[k * num_inner + inner]);
                    if (xv < best) { best = xv; }
                }
                s_store(&y_base[inner], best);
                break;
            }
            }
        }
    });
}

/// General multi-axis path: an odometer over the runs, vectorized across the
/// last kept axis when its stride is one on both sides.
template <typename T, typename Op>
void reduce_multi_general(const ReduceAttributes& attrs,
                          TensorView& output,
                          const TensorView& input,
                          const ReduceRun* runs,
                          int64_t n_runs,
                          const bool* kept,
                          float inv_count,
                          const ComputeContext& ctx)
{
    const int64_t rank = input.rank();
    const bool keepdims = attrs.keepdims;

    int64_t out_pos[TensorView::kMaxRank] = {};
    int64_t n_kept = 0;
    for (int64_t d = 0, p = 0; d < rank; ++d) {
        if (kept[d]) {
            out_pos[d] = keepdims ? d : p;
            ++p;
            ++n_kept;
        }
    }

    int64_t lane_axis = -1;
    if (n_kept > 0) {
        int64_t last_kept = -1;
        for (int64_t d = rank - 1; d >= 0; --d) {
            if (kept[d]) { last_kept = d; break; }
        }
        if (input.stride_elems(last_kept) == 1 &&
            output.stride_elems(out_pos[last_kept]) == 1) {
            lane_axis = last_kept;
        }
    }
    const int64_t lanes = (lane_axis >= 0) ? input.shape(lane_axis) : 1;

    int64_t unit_axes[TensorView::kMaxRank] = {};
    int64_t n_unit_axes = 0;
    int64_t units = 1;
    for (int64_t d = 0; d < rank; ++d) {
        if (kept[d] && d != lane_axis) {
            unit_axes[n_unit_axes++] = d;
            units *= input.shape(d);
        }
    }

    const T* x_ptr = input.ptr<T>();
    T* y_ptr = output.ptr<T>();
    constexpr int L = simd_lane_for<T>;
    const bool is_mean = attrs.type == ReduceType::Mean;
    const int64_t n_chunks = (lanes + kLaneChunk - 1) / kLaneChunk;

    ctx.cpu.run(0, units * n_chunks, [&](int64_t task) {
        const int64_t unit = task / n_chunks;
        const int64_t chunk = task % n_chunks;
        const int64_t lane0 = chunk * kLaneChunk;
        const int64_t lane1 = std::min(lane0 + kLaneChunk, lanes);

        int64_t in_base = 0;
        int64_t out_off = 0;
        int64_t rem = unit;
        for (int64_t i = n_unit_axes - 1; i >= 0; --i) {
            const int64_t ax = unit_axes[i];
            const int64_t idx = rem % input.shape(ax);
            rem /= input.shape(ax);
            in_base += idx * input.stride_elems(ax);
            out_off += idx * output.stride_elems(out_pos[ax]);
        }

        int64_t idx[TensorView::kMaxRank] = {};
        auto step = [&](int64_t& delta) -> bool {
            for (int64_t g = n_runs - 1; g >= 0; --g) {
                delta += runs[g].stride;
                if (++idx[g] < runs[g].dim) { return true; }
                idx[g] = 0;
                delta -= runs[g].dim * runs[g].stride;
            }
            return false;
        };

        int64_t lane = lane0;
        for (; lane + L <= lane1; lane += L) {
            auto acc = Op::vec_identity(x_ptr);
            int64_t delta = 0;
            do {
                acc = Op::combine(acc, v_load(x_ptr + in_base + delta + lane));
            } while (step(delta));
            if (is_mean) { acc = v_mul(acc, v_set1(x_ptr, inv_count)); }
            v_store(y_ptr + out_off + lane, acc);
        }
        for (; lane < lane1; ++lane) {
            float acc = Op::scalar_identity();
            int64_t delta = 0;
            do {
                acc = Op::combine_scalar(
                    acc, s_load(&x_ptr[in_base + delta + lane]));
            } while (step(delta));
            if (is_mean) { acc *= inv_count; }
            s_store(&y_ptr[out_off + lane], acc);
        }
    });
}

template <typename T>
void reduce_copy(TensorView& output,
                 const TensorView& input,
                 const ComputeContext& ctx)
{
    const int64_t rank = input.rank();
    const int64_t n = input.numel();
    if (rank < 2) {
        std::copy_n(input.ptr<T>(), static_cast<size_t>(n), output.ptr<T>());
        return;
    }

    // total_rows()/numel() gives the logical elements in one physical row for
    // every layout: W planar, C*W channel-last, the packed block width packed.
    const int64_t rows = input.total_rows();
    if (rows <= 0) { return; }
    const int64_t cols = n / rows;
    const int64_t in_row = input.row_stride_elems();
    const int64_t out_row = output.row_stride_elems();
    const T* x_ptr = input.ptr<T>();
    T* y_ptr = output.ptr<T>();

    ctx.cpu.run(0, rows, [&](int64_t row) {
        std::copy_n(x_ptr + row * in_row, static_cast<size_t>(cols),
                    y_ptr + row * out_row);
    });
}

/// The single-axis dispatch, unchanged: channel-last, last axis, inner
/// contiguous, general.
template <typename T>
void reduce_single_axis(const ReduceAttributes& attrs,
                        TensorView& output,
                        const TensorView& input,
                        int64_t axis,
                        const ComputeContext& ctx)
{
    const int64_t rank = input.rank();

    if (is_channel_last(input.layout())) {
        reduce_channel_last<T>(attrs, output, input, axis, ctx);
        return;
    }

    if (axis == rank - 1) {
        reduce_contiguous_simd<T>(attrs, output, input, axis, ctx);
        return;
    }

    int64_t num_outer = 1;
    for (int64_t d = 0; d < axis; ++d) { num_outer *= input.shape(d); }

    const int64_t reduce_size = input.shape(axis);

    int64_t num_inner = 1;
    for (int64_t d = axis + 1; d < rank; ++d) { num_inner *= input.shape(d); }

    const bool input_inner_contiguous = (input.stride_elems(axis) == num_inner);
    const bool output_inner_contiguous = (output.rank() <= 1) ||
        (output.row_stride_elems() == output.shape(output.rank() - 1));

    if (input_inner_contiguous && output_inner_contiguous) {
        reduce_inner_contiguous_simd<T>(attrs, output, input, axis,
                                         num_outer, reduce_size, num_inner, ctx);
        return;
    }

    reduce_general_scalar<T>(attrs, output, input, axis,
                              num_outer, reduce_size, num_inner, ctx);
}

}  // anonymous namespace

template <typename T>
void reduce_impl(const ReduceAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();

    int64_t axes[TensorView::kMaxRank] = {};
    int64_t n_axes = 0;
    if (attrs.axes.empty()) {
        if (attrs.noop_with_empty_axes) {
            reduce_copy<T>(output, input, ctx);
            return;
        }
        for (int64_t d = 0; d < rank; ++d) { axes[n_axes++] = d; }
    } else {
        for (const int64_t a : attrs.axes) {
            const int64_t ax = a < 0 ? a + rank : a;
            NNOPS_ASSERT(ax >= 0 && ax < rank);
            axes[n_axes++] = ax;
        }
        std::sort(axes, axes + n_axes);
    }

    if (n_axes == 1) {
        reduce_single_axis<T>(attrs, output, input, axes[0], ctx);
        return;
    }

    bool kept[TensorView::kMaxRank] = {};
    for (int64_t d = 0; d < rank; ++d) { kept[d] = true; }
    for (int64_t i = 0; i < n_axes; ++i) { kept[axes[i]] = false; }

    int64_t count = 1;
    for (int64_t i = 0; i < n_axes; ++i) { count *= input.shape(axes[i]); }
    const float inv_count = 1.0f / static_cast<float>(count);

    ReduceRun runs[TensorView::kMaxRank];
    const int64_t n_runs = build_reduced_runs(input, axes, n_axes, runs);

    if (n_runs == 1) {
        const int64_t first = axes[0];
        const int64_t last = axes[n_axes - 1];

        int64_t num_outer = 1;
        for (int64_t d = 0; d < first; ++d) { num_outer *= input.shape(d); }

        int64_t num_inner = 1;
        for (int64_t d = last + 1; d < rank; ++d) {
            num_inner *= input.shape(d);
        }

        bool outer_dense = true;
        for (int64_t d = 0; d + 1 < first; ++d) {
            if (input.stride_elems(d) !=
                input.shape(d + 1) * input.stride_elems(d + 1)) {
                outer_dense = false;
                break;
            }
        }
        const int64_t outer_stride =
            (first > 0) ? input.stride_elems(first - 1) : 0;

        if (outer_dense && num_inner == 1 && runs[0].stride == 1) {
            reduce_multi_fast_row<T>(attrs, output, input, num_outer,
                                     outer_stride, runs[0].dim, ctx);
            return;
        }

        const bool output_dense = output.rank() < 2 ||
            output.row_stride_elems() == output.shape(output.rank() - 1);
        if (outer_dense && num_inner > 1 &&
            runs[0].stride == num_inner &&
            input.stride_elems(rank - 1) == 1 && output_dense) {
            reduce_multi_fast_lanes<T>(attrs, output, input, num_outer,
                                       outer_stride, runs[0].dim, num_inner,
                                       ctx);
            return;
        }
    }

    switch (attrs.type) {
    case ReduceType::Sum:
    case ReduceType::Mean:
        reduce_multi_general<T, kernel::ReduceAdd>(
            attrs, output, input, runs, n_runs, kept, inv_count, ctx);
        break;
    case ReduceType::Max:
        reduce_multi_general<T, kernel::ReduceMax>(
            attrs, output, input, runs, n_runs, kept, inv_count, ctx);
        break;
    case ReduceType::Min:
        reduce_multi_general<T, kernel::ReduceMin>(
            attrs, output, input, runs, n_runs, kept, inv_count, ctx);
        break;
    }
}

Status reduce_cpu(const ReduceAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* /*workspace*/)
{
    dispatch_f32_f16(inputs[0].data_type(), "reduce_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        reduce_impl<T>(attrs, output, inputs, ctx);
    });
    return Status::ok;
}

}  // namespace nnops::backend::cpu
