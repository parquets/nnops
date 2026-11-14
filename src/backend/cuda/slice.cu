/// @file slice.cu
/// @brief CUDA Slice — the layout walk with a start folded into the base and a
///        step folded into the source stride.
///
/// `x[s:e:st]` on a listed axis moves the row base by `s` and multiplies that
/// axis's source stride by `st`; the output extent comes from the descriptor,
/// which already carries the ONNX clamp. Everything else — the other axes, the
/// pitch of a padded input, the vectorized row copy — is `move_common.cuh`.
///
/// A strided slice (st > 1) simply leaves the source run non-unit, so the copy
/// falls to the scalar path; the destination stays contiguous.
///
/// f32, f16 and bf16 in, out at the same dtype.

#include "move_common.cuh"
#include "nnops/ops/slice.hpp"

namespace nnops::backend::cuda {
namespace {

template <class T>
Status slice_entry(const SliceAttributes& attrs, TensorView& output,
                   const TensorView& input, cudaStream_t stream)
{
    const int64_t rank = input.rank();
    const int64_t total = output.numel();
    if (total == 0) { return Status::ok; }

    int64_t dims[8], src_strides[8], dst_strides[8];
    fill_output_dims(output, dims);
    fill_view_strides(input, src_strides);
    fill_view_strides(output, dst_strides);

    int64_t base = 0;
    for (size_t a = 0; a < attrs.axes.size(); ++a) {
        int64_t ax = attrs.axes[a] < 0 ? attrs.axes[a] + rank : attrs.axes[a];
        const int64_t n = input.shape(ax);
        int64_t s = attrs.starts[a];
        if (s < 0) { s += n; }
        s = std::clamp<int64_t>(s, 0, n);
        const int64_t step = (a < attrs.steps.size()) ? attrs.steps[a] : 1;
        base += s * input.stride_elems(ax);
        src_strides[ax] = input.stride_elems(ax) * step;
    }

    MovePlan p;
    p.rank = rank;
    p.src = make_move_layout(dims, src_strides);
    p.dst = make_move_layout(dims, dst_strides);
    p.src_base = base;
    p.cols = dims[rank - 1];
    p.rows = total / p.cols;
    p.src_step = src_strides[rank - 1];
    p.dst_step = dst_strides[rank - 1];
    p.dst_compact = view_is_compact(output);
    for (int d = 0; d < 8; ++d) {
        p.dims[d] = dims[d];
        p.src_strides[d] = src_strides[d];
    }
    return launch_move(input.ptr<T>(), output.ptr<T>(), p, stream);
}

}  // anonymous namespace

Status slice_cuda(const SliceAttributes& attrs, TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx, void* /*workspace*/)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    switch (inputs[0].data_type()) {
    case DataType::f32:
        return slice_entry<float>(attrs, output, inputs[0], stream);
    case DataType::f16:
        return slice_entry<__half>(attrs, output, inputs[0], stream);
    case DataType::bf16:
        return slice_entry<__nv_bfloat16>(attrs, output, inputs[0], stream);
    default:
        return Status::unsupported;
    }
}

}  // namespace nnops::backend::cuda
