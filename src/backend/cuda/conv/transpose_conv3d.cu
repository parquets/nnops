/// @file transpose_conv3d.cu
/// @brief CUDA TransposeConv3D — the 3-D implicit GEMM of conv3d_common.cuh,
///        posed as the backward gather.
///
/// The output voxel o is written from the input voxels a scaled kernel names:
/// i = (o + pad − k·dil) / stride, and a tap that does not divide is not in that
/// output's reduction. The operator's own weight order is
/// [IC][OC/G][KD][KH][KW], so the kernel walks its five strides; the group
/// offset is on the IC axis, the opposite of Conv3D's.
///
/// The M axis is the output voxel, so a stride larger than one leaves the
/// reduction with fewer contributing taps rather than fewer output voxels —
/// that is what makes this the same kernel with a predicate.

#include "conv3d_common.cuh"
#include "nnops/ops/transpose_conv3d.hpp"

#include <climits>
#include <span>

namespace nnops::backend::cuda {
namespace {

using namespace conv3d;

template <class Elem>
Status tconv3d_impl(const C3Params& p, dim3 grid, cudaStream_t stream)
{
    return c3_launch<Elem, true>(p, grid, stream);
}

} // anonymous namespace

Status transpose_conv3d_cuda(const TransposeConv3DAttributes& attrs, TensorView& output,
                             std::span<const TensorView> inputs, const ComputeContext& ctx,
                             void*)
{
    const TensorView& act = inputs[0];
    const TensorView& wgt = inputs[1];
    const TensorView* bias = inputs.size() > 2 ? &inputs[2] : nullptr;

    // The operator's own contract is f32/f16; the weight is
    // [IC][OC/G][KD][KH][KW].
    const DataType dt = act.data_type();
    if ((dt != DataType::f32 && dt != DataType::f16) || wgt.data_type() != dt ||
        output.data_type() != dt || wgt.rank() != 5) {
        return Status::unsupported;
    }
    if (output.numel() == 0) {
        return Status::ok;
    }

    C3Params p;
    p.id = static_cast<int>(act.shape(1));
    p.ih = static_cast<int>(act.shape(2));
    p.iw = static_cast<int>(act.shape(3));
    p.ic = static_cast<int>(act.shape(4));
    p.od = static_cast<int>(output.shape(1));
    p.oh = static_cast<int>(output.shape(2));
    p.ow = static_cast<int>(output.shape(3));
    p.oc = static_cast<int>(output.shape(4));

    p.kd = static_cast<int>(attrs.kernel_size[0]);
    p.kh = static_cast<int>(attrs.kernel_size[1]);
    p.kw = static_cast<int>(attrs.kernel_size[2]);
    p.sd = static_cast<int>(attrs.stride[0]);
    p.sh = static_cast<int>(attrs.stride[1]);
    p.sw = static_cast<int>(attrs.stride[2]);
    p.dd = static_cast<int>(attrs.dilation[0]);
    p.dh = static_cast<int>(attrs.dilation[1]);
    p.dw = static_cast<int>(attrs.dilation[2]);
    p.pd = static_cast<int>(attrs.padding[0]);
    p.ph = static_cast<int>(attrs.padding[1]);
    p.pw = static_cast<int>(attrs.padding[2]);

    p.groups = static_cast<int>(attrs.groups);
    p.ocg = static_cast<int>(wgt.shape(1));
    p.icg = p.ic / p.groups;
    p.m = int64_t(p.od) * p.oh * p.ow;

    p.in_batch = act.stride_elems(0);
    p.in_d = act.stride_elems(1);
    p.in_h = act.stride_elems(2);
    p.in_w = act.stride_elems(3);
    p.in_c = act.stride_elems(4);
    p.out_batch = output.stride_elems(0);
    p.out_d = output.stride_elems(1);
    p.out_h = output.stride_elems(2);
    p.out_w = output.stride_elems(3);
    p.out_c = output.stride_elems(4);

    // [IC][OC/G][KD][KH][KW].
    p.w_ic = wgt.stride_elems(0);
    p.w_oc = wgt.stride_elems(1);
    p.w_kd = wgt.stride_elems(2);
    p.w_kh = wgt.stride_elems(3);
    p.w_kw = wgt.stride_elems(4);
    // A group steps along the filter's IC axis.
    p.w_group = p.icg * p.w_ic;

    // 16 B copies: a tap's channel run is contiguous and aligned. The filter's
    // output channel axis here has the tap axes inside it, so its run is a
    // stride and the fill walks elements instead.
    const int64_t ch = 16 / int64_t(data_type_size(dt));
    p.fill_vec = p.in_c == 1 && p.icg % ch == 0 && p.in_w % ch == 0 &&
                 p.in_batch % ch == 0 &&
                 reinterpret_cast<uintptr_t>(p.in) % 16 == 0;
    p.w_vec = 0;

    p.in = act.ptr<void>();
    p.w = wgt.ptr<void>();
    p.out = output.ptr<void>();
    p.bias = bias != nullptr ? bias->ptr<void>() : nullptr;

    p.ep = attrs.epilogue.type;
    p.ep_alpha = attrs.epilogue.alpha;
    p.ep_beta = attrs.epilogue.beta;
    p.min_clip = attrs.epilogue.min_clip;
    p.max_clip = attrs.epilogue.max_clip;
    p.add_to = attrs.add_to ? 1 : 0;
    p.pre_act = (attrs.add_to && attrs.residual_pre_activation) ? 1 : 0;

    const int64_t k = int64_t(p.icg) * p.kd * p.kh * p.kw;
    if (p.icg <= 0 || p.ocg <= 0 || p.m <= 0 || k <= 0 || k > INT_MAX || p.groups <= 0 ||
        p.kd <= 0 || p.kh <= 0 || p.kw <= 0 || p.sd <= 0 || p.sh <= 0 || p.sw <= 0 ||
        p.dd <= 0 || p.dh <= 0 || p.dw <= 0) {
        return Status::unsupported;
    }
    if (int64_t(act.shape(0)) * p.groups > 65535) {
        return Status::unsupported;
    }
    // The gather's offsets are 32-bit: a slice whose span leaves the int range
    // would wrap, so it is refused rather than run.
    if (int64_t(p.id) * p.in_d > (int64_t(1) << 30) ||
        int64_t(p.ih) * p.in_h > (int64_t(1) << 30) ||
        int64_t(p.ic) * p.in_c > (int64_t(1) << 30) ||
        int64_t(p.od) * p.out_d > (int64_t(1) << 30) ||
        int64_t(p.oc) * p.out_c > (int64_t(1) << 30)) {
        return Status::unsupported;
    }

    const bool fp32 = dt == DataType::f32;
    const int64_t bm = fp32 ? int64_t(c3_bm<float, true>()) : int64_t(c3_bm<__half, true>());
    const int64_t bn = fp32 ? int64_t(c3_bn<float, true>()) : int64_t(c3_bn<__half, true>());
    const dim3 grid(static_cast<unsigned>(ceil_div(p.m, bm)),
                    static_cast<unsigned>(ceil_div(int64_t(p.ocg), bn)),
                    static_cast<unsigned>(int64_t(act.shape(0)) * p.groups));
    const cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    return dt == DataType::f32 ? tconv3d_impl<float>(p, grid, stream)
                               : tconv3d_impl<__half>(p, grid, stream);
}

} // namespace nnops::backend::cuda
