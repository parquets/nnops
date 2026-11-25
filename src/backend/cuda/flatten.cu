/// @file flatten.cu
/// @brief CUDA Flatten — the input's logical order copied into dense storage.
///
/// This is the movement engine with the identity axis mapping: the source
/// layout is the input's own stride array (so an NHWC view read as dense is
/// already compact and the plan collapses to one `cudaMemcpyAsync`), the
/// destination is the output's stride array, which is usually compact. Every
/// whole-element dtype is carried — it is a byte copy, not arithmetic.
///
/// Like the CPU route, an input that is compact against a compact output takes
/// the copy path, and anything else walks one row per run.

#include "move_common.cuh"
#include "nnops/ops/flatten.hpp"

namespace nnops::backend::cuda {

namespace {

template <class T>
Status flatten_entry(const TensorView& input, TensorView& output, cudaStream_t stream)
{
    const int64_t rank = input.rank();
    if (rank <= 0) {
        return Status::ok;
    }
    if (rank == 1) {
        const size_t bytes = static_cast<size_t>(input.numel()) * sizeof(T);
        if (bytes != 0) {
            CUDA_CHECK(cudaMemcpyAsync(output.ptr<T>(), input.ptr<T>(), bytes,
                                       cudaMemcpyDeviceToDevice, stream));
        }
        return Status::ok;
    }

    int64_t dims[8], src_strides[8], dst_strides[8];
    fill_output_dims(output, dims);
    const int64_t total = output.numel();
    if (total == 0) {
        return Status::ok;
    }
    for (int64_t d = 0; d < rank; ++d) {
        src_strides[d] = input.stride_elems(d);
        dst_strides[d] = output.stride_elems(d);
    }
    for (int64_t d = rank; d < 8; ++d) {
        src_strides[d] = 0;
        dst_strides[d] = 0;
    }

    MovePlan p;
    p.rank = rank;
    p.src = make_move_layout(dims, src_strides);
    p.dst = make_move_layout(dims, dst_strides);
    p.cols = dims[rank - 1];
    p.rows = p.cols > 0 ? total / p.cols : 0;
    p.src_step = src_strides[rank - 1];
    p.dst_step = dst_strides[rank - 1];
    p.dst_compact = view_is_compact(output) &&
                    output.row_stride_elems() == dims[rank - 1];
    for (int d = 0; d < 8; ++d) {
        p.dims[d] = dims[d];
        p.src_strides[d] = src_strides[d];
    }
    return launch_move(input.ptr<T>(), output.ptr<T>(), p, stream);
}

} // anonymous namespace

Status flatten_cuda(const FlattenAttributes&, TensorView& output,
                    std::span<const TensorView> inputs, const ComputeContext& ctx, void*)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    switch (inputs[0].data_type()) {
    case DataType::f32:
        return flatten_entry<float>(inputs[0], output, stream);
    case DataType::f16:
        return flatten_entry<__half>(inputs[0], output, stream);
    case DataType::bf16:
        return flatten_entry<__nv_bfloat16>(inputs[0], output, stream);
    case DataType::s8:
        return flatten_entry<int8_t>(inputs[0], output, stream);
    case DataType::u8:
        return flatten_entry<uint8_t>(inputs[0], output, stream);
    case DataType::s32:
        return flatten_entry<int32_t>(inputs[0], output, stream);
    case DataType::s64:
        return flatten_entry<int64_t>(inputs[0], output, stream);
    default:
        return Status::unsupported;
    }
}

} // namespace nnops::backend::cuda
