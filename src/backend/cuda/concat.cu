/// @file concat.cu
/// @brief CUDA Concat — one layout walk per input, each writing its own band of
///        the output.
///
/// The concat axis is the only place the inputs disagree: input i owns the
/// output range [offset, offset + dim_i) on that axis, which is a base offset
/// of `offset * out.stride(axis)` and nothing else. So the call is one
/// `move_common.cuh` walk per input, over that input's own extents and strides
/// with the output's strides — pitch on either side needs no special case, and
/// a rank-1 concat is a flat copy of each input into the running position.
///
/// f32 and f16 in, out at the same dtype.

#include "move_common.cuh"
#include "nnops/ops/concat.hpp"

namespace nnops::backend::cuda {
namespace {

template <class T>
Status concat_entry(const ConcatAttributes& attrs, TensorView& output,
                    std::span<const TensorView> inputs, cudaStream_t stream)
{
    const int64_t rank = output.rank();
    T* dst = output.ptr<T>();

    if (rank < 2) {
        int64_t off = 0;
        for (const auto& in : inputs) {
            const size_t bytes = static_cast<size_t>(in.numel()) * sizeof(T);
            if (bytes != 0) {
                CUDA_CHECK(cudaMemcpyAsync(dst + off, in.ptr<T>(), bytes,
                                           cudaMemcpyDeviceToDevice, stream));
            }
            off += in.numel();
        }
        return Status::ok;
    }

    const int64_t ax = attrs.axis < 0 ? attrs.axis + rank : attrs.axis;
    int64_t dst_strides[8];
    fill_view_strides(output, dst_strides);

    int64_t band = 0;
    for (const auto& in : inputs) {
        const int64_t total = in.numel();
        if (total != 0) {
            int64_t dims[8], src_strides[8];
            for (int64_t d = 0; d < rank; ++d) { dims[d] = in.shape(d); }
            for (int64_t d = rank; d < 8; ++d) { dims[d] = 1; }
            fill_view_strides(in, src_strides);

            MovePlan p;
            p.rank = rank;
            p.src = make_move_layout(dims, src_strides);
            p.dst = make_move_layout(dims, dst_strides);
            p.dst_base = band * output.stride_elems(ax);
            p.cols = dims[rank - 1];
            p.rows = total / p.cols;
            p.src_step = src_strides[rank - 1];
            p.dst_step = dst_strides[rank - 1];
            p.dst_compact = false;
            for (int d = 0; d < 8; ++d) {
                p.dims[d] = dims[d];
                p.src_strides[d] = src_strides[d];
            }
            launch_move(in.ptr<T>(), dst, p, stream);
        }
        band += in.shape(ax);
    }
    return Status::ok;
}

}  // anonymous namespace

Status concat_cuda(const ConcatAttributes& attrs, TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx, void* /*workspace*/)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    switch (inputs[0].data_type()) {
    case DataType::f32:
        return concat_entry<float>(attrs, output, inputs, stream);
    case DataType::f16:
        return concat_entry<__half>(attrs, output, inputs, stream);
    default:
        return Status::unsupported;
    }
}

}  // namespace nnops::backend::cuda
