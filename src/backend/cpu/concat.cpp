/// @file concat.cpp
/// @brief Copy-based CPU implementation of Concat (f32/f16).
///
/// Driven by the logical axis, so channel-first and channel-last
/// (channel-last) share one walk. Every tensor is split into *lines* — the run
/// that is physically contiguous and perpendicular to the concat axis:
///
///   Channel-first  line = one (n, c, spatial…) row of W elements
///   Channel-last   line = one pixel's C channels
///
/// The axis coordinate selects the source input; the remaining coordinates give
/// the source and destination offsets through stride_elems(), so pitch padding
/// needs no special case. When the concat axis *is* the line axis the output
/// line is filled input by input; otherwise each line comes wholly from one
/// input.

#include "nnops/ops/concat.hpp"
#include "common/dtype_dispatch.hpp"

#include <cstring>
#include <vector>

namespace nnops::backend::cpu {

template <typename T>
void concat_impl(const ConcatAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs)
{
    const int64_t n_in = static_cast<int64_t>(inputs.size());
    if (n_in == 0) {
        return;
    }

    const int64_t rank = inputs[0].rank();
    const int64_t ax = (attrs.axis < 0) ? attrs.axis + rank : attrs.axis;

    auto* o_ptr = output.ptr<T>();

    // rank < 2 is a flat 1-D concat.
    if (rank < 2) {
        int64_t col = 0;
        for (const auto& in : inputs) {
            const int64_t n = in.numel();
            std::memcpy(o_ptr + col, in.ptr<T>(), static_cast<size_t>(n) * sizeof(T));
            col += n;
        }
        return;
    }

    std::vector<int64_t> axis_offset(static_cast<size_t>(n_in + 1), 0);
    for (int64_t i = 0; i < n_in; ++i) {
        axis_offset[static_cast<size_t>(i + 1)] =
            axis_offset[static_cast<size_t>(i)] +
            inputs[static_cast<size_t>(i)].shape(ax);
    }

    const bool channel_last = is_channel_last(output.layout());
    const int64_t line_axis = channel_last ? 1 : rank - 1;
    const int64_t line_len  = output.shape(line_axis);

    // Every dimension except the line axis is walked as one output line.
    std::vector<int64_t> walk_dims;
    walk_dims.reserve(static_cast<size_t>(rank - 1));
    for (int64_t d = 0; d < rank; ++d) {
        if (d != line_axis) {
            walk_dims.push_back(d);
        }
    }

    int64_t num_lines = 1;
    for (int64_t d : walk_dims) {
        num_lines *= output.shape(d);
    }
    if (num_lines == 0 || line_len == 0) {
        return;
    }

    const bool axis_is_line = (ax == line_axis);
    int64_t ax_pos = -1;  // position of the concat axis inside walk_dims
    for (size_t i = 0; i < walk_dims.size(); ++i) {
        if (walk_dims[i] == ax) { ax_pos = static_cast<int64_t>(i); }
    }

    std::vector<int64_t> out_stride(walk_dims.size());
    for (size_t i = 0; i < walk_dims.size(); ++i) {
        out_stride[i] = output.stride_elems(walk_dims[i]);
    }

    std::vector<int64_t> idx(walk_dims.size(), 0);

    for (int64_t line = 0; line < num_lines; ++line) {
        int64_t o_off = 0;
        for (size_t i = 0; i < idx.size(); ++i) {
            o_off += idx[i] * out_stride[i];
        }

        if (axis_is_line) {
            int64_t o_col = 0;
            for (int64_t n = 0; n < n_in; ++n) {
                const auto& in = inputs[static_cast<size_t>(n)];
                int64_t i_off = 0;
                for (size_t i = 0; i < idx.size(); ++i) {
                    i_off += idx[i] * in.stride_elems(walk_dims[i]);
                }
                const int64_t len = in.shape(ax);
                std::memcpy(o_ptr + o_off + o_col, in.ptr<T>() + i_off,
                            static_cast<size_t>(len) * sizeof(T));
                o_col += len;
            }
        } else {
            const int64_t o_ax = idx[static_cast<size_t>(ax_pos)];
            int64_t n = 0;
            while (n + 1 < n_in && o_ax >= axis_offset[static_cast<size_t>(n + 1)]) {
                ++n;
            }
            const auto& in = inputs[static_cast<size_t>(n)];
            const int64_t i_ax = o_ax - axis_offset[static_cast<size_t>(n)];

            int64_t i_off = 0;
            for (size_t i = 0; i < idx.size(); ++i) {
                const int64_t d = walk_dims[i];
                i_off += ((d == ax) ? i_ax : idx[i]) * in.stride_elems(d);
            }
            std::memcpy(o_ptr + o_off, in.ptr<T>() + i_off,
                        static_cast<size_t>(line_len) * sizeof(T));
        }

        // Odometer over the walked dimensions.
        int64_t d = static_cast<int64_t>(idx.size()) - 1;
        while (d >= 0 && ++idx[static_cast<size_t>(d)] ==
                                output.shape(walk_dims[static_cast<size_t>(d)])) {
            idx[static_cast<size_t>(d--)] = 0;
        }
        if (d < 0) {
            break;
        }
    }
}

Status concat_cpu(const ConcatAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& /*ctx*/,
                 void* /*workspace*/)
{
    dispatch_f32_f16(inputs[0].data_type(), "concat_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        concat_impl<T>(attrs, output, inputs);
    });
    return Status::ok;
}

}  // namespace nnops::backend::cpu
