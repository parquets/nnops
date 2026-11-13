/// @file split.cu
/// @brief CUDA Split — one layout walk per output, each reading its own band of
///        the input.
///
/// Output j covers the axis range [off, off + len), so its source base is
/// `off * in.stride(axis)` and the extents come from the output's own
/// descriptor. The inputs and outputs are separate tensors here, which is the
/// only structural difference from Concat.
///
/// f32 and f16 in, out at the same dtype.

#include "move_common.cuh"
#include "nnops/ops/split.hpp"

namespace nnops::backend::cuda {
namespace {

template <class T>
Status split_entry(const SplitAttributes& attrs, std::span<TensorView> outputs,
                   const TensorView& input, cudaStream_t stream)
{
    const int64_t rank = input.rank();
    const int64_t axis = attrs.axis < 0 ? attrs.axis + rank : attrs.axis;
    const int64_t axis_stride = input.stride_elems(axis);

    int64_t src_strides[8];
    fill_view_strides(input, src_strides);

    int64_t off = 0;
    for (TensorView& out : outputs) {
        const int64_t total = out.numel();
        const int64_t len = out.shape(axis);
        if (total != 0) {
            int64_t dims[8], dst_strides[8];
            fill_output_dims(out, dims);
            fill_view_strides(out, dst_strides);

            MovePlan p;
            p.rank = rank;
            p.src = make_move_layout(dims, src_strides);
            p.dst = make_move_layout(dims, dst_strides);
            p.src_base = off * axis_stride;
            p.cols = dims[rank - 1];
            p.rows = total / p.cols;
            p.src_step = src_strides[rank - 1];
            p.dst_step = dst_strides[rank - 1];
            p.dst_compact = view_is_compact(out);
            for (int d = 0; d < 8; ++d) {
                p.dims[d] = dims[d];
                p.src_strides[d] = src_strides[d];
            }
            launch_move(input.ptr<T>(), out.ptr<T>(), p, stream);
        }
        off += len;
    }
    return Status::ok;
}

}  // anonymous namespace

Status split_cuda(const SplitAttributes& attrs, std::span<TensorView> outputs,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx, void* /*workspace*/)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    switch (inputs[0].data_type()) {
    case DataType::f32:
        return split_entry<float>(attrs, outputs, inputs[0], stream);
    case DataType::f16:
        return split_entry<__half>(attrs, outputs, inputs[0], stream);
    default:
        return Status::unsupported;
    }
}

}  // namespace nnops::backend::cuda
