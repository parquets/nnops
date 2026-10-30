/// @file reduce.cpp
/// @brief SIMD-optimized CPU implementation of the Reduce operator.
///
/// f32 and f16; planar (NCHW/NCDHW) and channel-packed (NCHWC8/NCDHWC8)
/// layouts. Pitch-aware via stride_elems() / row_stride_elems().
///
/// Dispatch, in order:
///   1.  axis==rank-1, pack>1   -> kernel::reduce_process_packed_row
///   1b. axis==1,      pack>1   -> kernel::reduce_process_packed_channel
///   1c. other axis,   pack>1   -> kernel::reduce_process_packed_col
///   2.  axis==rank-1, pack==1  -> kernel::reduce_process_contiguous_row
///   3.  inner-contiguous, pack==1 -> kernel::reduce_process_inner_contiguous_block
///   4.  general                   -> scalar strided walk over (outer x inner)

#include "nnops/ops/reduce.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "simd_kernel/simd_reduce.hpp"
#include "common/index.hpp"
#include "common/dtype_dispatch.hpp"

#include <cfloat>
#include <vector>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

template <typename T>
void reduce_packed_simd(const ReduceAttributes& attrs,
                         TensorView& output,
                         const TensorView& input,
                         int64_t axis,
                         const ComputeContext& ctx)
{
    const int64_t rank = input.rank();
    NNOPS_ASSERT(axis == rank - 1);

    const int64_t pack = input.channel_pack_size();
    const int64_t D = input.shape(axis);
    const int64_t num_rows = input.total_rows();
    const int64_t x_rs = input.row_stride_elems();

    const T* x_ptr = input.ptr<T>();
    T* y_ptr = output.ptr<T>();
    const float inv_D = 1.0f / static_cast<float>(D);

    const auto process_row = [&](int64_t r) {
        kernel::reduce_process_packed_row<T>(
            x_ptr + r * x_rs, y_ptr + r * pack, D, pack, attrs.type, inv_D);
    };

    ctx.cpu.run(0, num_rows, process_row);
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

    const std::vector<int64_t> inner_offsets =
        build_offsets(input, axis + 1, rank - 1, num_inner);

    const int64_t y_inner_start = attrs.keepdims ? axis + 1 : axis;
    const std::vector<int64_t> y_inner_offsets =
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

}  // anonymous namespace

template <typename T>
void reduce_impl(const ReduceAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();

    int64_t axis = attrs.axis;
    if (axis < 0) { axis += rank; }
    NNOPS_ASSERT(axis >= 0 && axis < rank);

    const int64_t pack = input.channel_pack_size();

    if (axis == rank - 1 && pack > 1) {
        reduce_packed_simd<T>(attrs, output, input, axis, ctx);
        return;
    }

    if (axis == 1 && pack > 1) {
        const int64_t C = input.shape(1);                // logical channel count
        const int64_t C8 = input.num_channel_blocks();   // ceil(C / pack)
        const int64_t chan_stride = input.stride_elems(1);
        const int64_t valid_lanes = (C % pack == 0) ? pack : (C % pack);
        const float inv_total = 1.0f / static_cast<float>(C);

        const T* x_ptr = input.ptr<T>();
        T* y_ptr = output.ptr<T>();

        int64_t num_spatial = 1;
        for (int64_t d = 0; d < rank; ++d) {
            if (d != 1) {
                num_spatial *= input.shape(d);
            }
        }

        const int64_t rank_out = output.rank();
        const bool keepdims = attrs.keepdims;

        const auto process_pos = [&](int64_t s) {
            const int64_t in_off = decompose_flat_offset(s, input, 0, rank - 1, /*skip_dim=*/1);
            const int64_t out_off = decompose_flat_offset(s, output, 0, rank_out - 1,
                                                          keepdims ? 1 : -1);
            kernel::reduce_process_packed_channel<T>(
                x_ptr + in_off, y_ptr + out_off,
                chan_stride, C8, pack, valid_lanes, attrs.type, inv_total);
        };

        ctx.cpu.run(0, num_spatial, process_pos);

        return;
    }

    if (pack > 1) {
        const int64_t D = input.shape(axis);
        const int64_t axis_stride = input.stride_elems(axis);
        const float inv_D = 1.0f / static_cast<float>(D);

        const T* x_ptr = input.ptr<T>();
        T* y_ptr = output.ptr<T>();

        int64_t num_positions = 1;
        for (int64_t d = 0; d < rank; ++d) {
            if (d != axis) {
                num_positions *= input.shape(d);
            }
        }

        const int64_t rank_out = output.rank();
        const bool keepdims = attrs.keepdims;

        const auto process_pos = [&](int64_t s) {
            const int64_t in_off = decompose_flat_offset(s, input, 0, rank - 1, /*skip_dim=*/axis);
            const int64_t out_off = decompose_flat_offset(s, output, 0, rank_out - 1,
                                                          keepdims ? axis : -1);
            kernel::reduce_process_packed_col<T>(
                x_ptr + in_off, y_ptr + out_off,
                axis_stride, D, pack, attrs.type, inv_D);
        };

        ctx.cpu.run(0, num_positions, process_pos);

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

    const bool input_inner_contiguous = (pack == 1) &&
        (input.stride_elems(axis) == num_inner);
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
