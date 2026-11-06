/// @file lowering.cpp
/// @brief Graph node → the nnops operator that runs it.
///
/// This is where the framework's generic attribute list becomes a typed one.
/// The importer normalised every ONNX attribute into the graph's own names and
/// filled in defaults; here those values are read back out into the operator's
/// attribute struct and handed to its create(). Nothing is validated beyond
/// what the struct's types force — an operator's own constructor and validate()
/// are the authority on what it accepts, and duplicating those rules here would
/// only let the two drift.
///
/// The one thing added on the way through is the fused epilogue: a pass may have
/// written `fused_activation` onto a node whose operator carries an Epilogue,
/// and that is folded into the typed attributes here, not by the pass, so the
/// pass never needs to know which operator a node lowers to.

#include "internal.hpp"

#include "nnops/ops/activation.hpp"
#include "nnops/ops/argminmax.hpp"
#include "nnops/ops/clamp.hpp"
#include "nnops/ops/concat.hpp"
#include "nnops/ops/conv2d.hpp"
#include "nnops/ops/depthwise_conv.hpp"
#include "nnops/ops/eltwise.hpp"
#include "nnops/ops/gather.hpp"
#include "nnops/ops/linear.hpp"
#include "nnops/ops/matmul.hpp"
#include "nnops/ops/norm.hpp"
#include "nnops/ops/pad.hpp"
#include "nnops/ops/permute.hpp"
#include "nnops/ops/pooling.hpp"
#include "nnops/ops/reduce.hpp"
#include "nnops/ops/resize.hpp"
#include "nnops/ops/slice.hpp"
#include "nnops/ops/softmax.hpp"
#include "nnops/ops/split.hpp"
#include "nnops/ops/tile.hpp"
#include "nnops/ops/topk.hpp"
#include "nnops/ops/transpose_conv2d.hpp"
#include "nnops/ops/unary.hpp"
#include "nnops/ops/where.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace nnops::framework::detail {

namespace {

Epilogue epilogue_for(const Node& node)
{
    Epilogue ep;
    ep.type = node.fused_activation;
    ep.alpha = node.fused_alpha;
    ep.beta = node.fused_beta;
    return ep;
}

/// Copy a 2-entry attribute into a fixed array, defaulting a missing entry.
template <size_t N>
std::array<int64_t, N> as_array(std::span<const int64_t> values)
{
    std::array<int64_t, N> out{};
    for (size_t i = 0; i < N; ++i) {
        out[i] = i < values.size() ? values[i] : 0;
    }
    return out;
}

/// A 2D spatial attribute stored in the operators' 3-slot window arrays: the
/// last two slots carry H/W and the first stays neutral.
std::array<int64_t, 3> spatial3(std::span<const int64_t> values, int64_t fill)
{
    std::array<int64_t, 3> out = {fill, fill, fill};
    if (values.size() >= 3) {
        out = {values[0], values[1], values[2]};
    } else if (values.size() == 2) {
        out = {fill, values[0], values[1]};
    }
    return out;
}

template <typename Small>
void assign_small(Small& out, std::span<const int64_t> values)
{
    out = values;
}

template <typename Op, typename Attrs>
Status make(const Attrs& attrs, Backend backend, std::unique_ptr<OpBase>& impl)
{
    impl = Op::create(attrs, backend);
    return impl != nullptr ? Status::ok : Status::out_of_memory;
}

}  // namespace

Conv2DAttributes conv2d_attributes(const Node& node)
{
    const Attributes& a = node.attrs;
    Conv2DAttributes attrs;
    attrs.kernel_size = as_array<2>(a.get_ints("kernel"));
    attrs.stride = as_array<2>(a.get_ints("stride"));
    attrs.dilation = as_array<2>(a.get_ints("dilation"));
    attrs.padding = as_array<2>(a.get_ints("padding"));
    attrs.groups = a.get_int("groups", 1);
    attrs.auto_pad = static_cast<Conv2DAttributes::AutoPad>(a.get_int("auto_pad", 0));
    return attrs;
}

Status lower_node(const Node& node, Backend backend,
                  std::unique_ptr<OpBase>& impl, std::string& error)
{
    impl.reset();
    if (node.kind != NodeKind::Op) {
        return Status::ok;  // Shape and Cast run in the executor, not a kernel
    }

    const Attributes& a = node.attrs;
    switch (node.op) {
    case OpType::Activation: {
        ActivationAttributes attrs;
        attrs.type = static_cast<ActivationType>(a.get_int("type"));
        attrs.alpha = a.get_float("alpha", 0.0f);
        attrs.beta = a.get_float("beta", 1.0f);
        return make<Activation>(attrs, backend, impl);
    }
    case OpType::Unary: {
        UnaryAttributes attrs;
        attrs.type = static_cast<UnaryType>(a.get_int("type"));
        return make<Unary>(attrs, backend, impl);
    }
    case OpType::Eltwise: {
        EltwiseAttributes attrs;
        attrs.type = static_cast<EltwiseType>(a.get_int("type"));
        return make<Eltwise>(attrs, backend, impl);
    }
    case OpType::Conv2D: {
        Conv2DAttributes attrs = conv2d_attributes(node);
        attrs.epilogue = epilogue_for(node);
        return make<Conv2D>(attrs, backend, impl);
    }
    case OpType::DepthwiseConv: {
        // The depthwise operator's spatial arrays carry a third slot for 3D;
        // on a 2D input its first entry is inert, so the 2-slot kernel shape
        // fills the last two.
        DepthwiseConvAttributes attrs;
        attrs.kernel_size = spatial3(a.get_ints("kernel"), 1);
        attrs.stride = spatial3(a.get_ints("stride"), 1);
        attrs.dilation = spatial3(a.get_ints("dilation"), 1);
        attrs.padding = spatial3(a.get_ints("padding"), 0);
        attrs.epilogue = epilogue_for(node);
        return make<DepthwiseConv>(attrs, backend, impl);
    }
    case OpType::TransposeConv2D: {
        TransposeConv2DAttributes attrs;
        attrs.kernel_size = as_array<2>(a.get_ints("kernel"));
        attrs.stride = as_array<2>(a.get_ints("stride"));
        attrs.dilation = as_array<2>(a.get_ints("dilation"));
        attrs.padding = as_array<2>(a.get_ints("padding"));
        attrs.output_padding = as_array<2>(a.get_ints("output_padding"));
        attrs.groups = a.get_int("groups", 1);
        attrs.epilogue = epilogue_for(node);
        return make<TransposeConv2D>(attrs, backend, impl);
    }
    case OpType::MatMul: {
        MatMulAttributes attrs;
        attrs.transpose_a = a.get_int("transpose_a", 0) != 0;
        attrs.transpose_b = a.get_int("transpose_b", 0) != 0;
        attrs.epilogue = epilogue_for(node);
        return make<MatMul>(attrs, backend, impl);
    }
    case OpType::Linear: {
        LinearAttributes attrs;
        attrs.epilogue = epilogue_for(node);
        return make<Linear>(attrs, backend, impl);
    }
    case OpType::Pooling: {
        PoolingAttributes attrs;
        attrs.type = static_cast<PoolingType>(a.get_int("type"));
        attrs.global = a.get_int("global", 0) != 0;
        attrs.kernel_shape = spatial3(a.get_ints("kernel_shape"), 1);
        attrs.stride = spatial3(a.get_ints("stride"), 1);
        attrs.padding = spatial3(a.get_ints("padding"), 0);
        attrs.dilation = spatial3(a.get_ints("dilation"), 1);
        attrs.exclude_pad = a.get_int("exclude_pad", 0) != 0;
        attrs.auto_pad = static_cast<PoolingAttributes::AutoPad>(
            a.get_int("auto_pad", 0));
        return make<Pooling>(attrs, backend, impl);
    }
    case OpType::Norm: {
        NormAttributes attrs;
        attrs.type = static_cast<NormType>(a.get_int("type"));
        attrs.axis = a.get_int("axis", -1);
        attrs.epsilon = a.get_float("epsilon", 1e-5f);
        return make<Norm>(attrs, backend, impl);
    }
    case OpType::Softmax: {
        SoftmaxAttributes attrs;
        attrs.axis = a.get_int("axis", -1);
        attrs.log_softmax = a.get_int("log_softmax", 0) != 0;
        return make<Softmax>(attrs, backend, impl);
    }
    case OpType::Reduce: {
        ReduceAttributes attrs;
        attrs.type = static_cast<ReduceType>(a.get_int("type"));
        attrs.axis = a.get_int("axis", 0);
        attrs.keepdims = a.get_int("keepdims", 0) != 0;
        return make<Reduce>(attrs, backend, impl);
    }
    case OpType::Resize: {
        ResizeAttributes attrs;
        attrs.mode = static_cast<ResizeMode>(a.get_int("mode", 0));
        attrs.coord_mode = static_cast<CoordinateTransformMode>(
            a.get_int("coord_mode", 0));
        attrs.output_size = spatial3(a.get_ints("output_size"), 0);
        return make<Resize>(attrs, backend, impl);
    }
    case OpType::Clamp: {
        ClampAttributes attrs;
        attrs.min_val = a.get_float("min", -INFINITY);
        attrs.max_val = a.get_float("max", INFINITY);
        return make<Clamp>(attrs, backend, impl);
    }
    case OpType::Concat: {
        ConcatAttributes attrs;
        attrs.axis = a.get_int("axis", -1);
        return make<Concat>(attrs, backend, impl);
    }
    case OpType::Permute: {
        PermuteAttributes attrs;
        assign_small(attrs.perm, a.get_ints("perm"));
        return make<Permute>(attrs, backend, impl);
    }
    case OpType::Slice: {
        SliceAttributes attrs;
        assign_small(attrs.starts, a.get_ints("starts"));
        assign_small(attrs.ends, a.get_ints("ends"));
        assign_small(attrs.axes, a.get_ints("axes"));
        assign_small(attrs.steps, a.get_ints("steps"));
        return make<Slice>(attrs, backend, impl);
    }
    case OpType::Pad: {
        PadAttributes attrs;
        const auto pads = a.get_ints("pads");
        attrs.pads.assign(pads.begin(), pads.end());
        attrs.mode = static_cast<PadAttributes::Mode>(a.get_int("mode", 0));
        attrs.constant_value = a.get_float("constant_value", 0.0f);
        return make<Pad>(attrs, backend, impl);
    }
    case OpType::Gather: {
        GatherAttributes attrs;
        attrs.axis = a.get_int("axis", 0);
        return make<Gather>(attrs, backend, impl);
    }
    case OpType::Where: {
        return make<Where>(WhereAttributes{}, backend, impl);
    }
    case OpType::Tile: {
        TileAttributes attrs;
        const auto repeats = a.get_ints("repeats");
        attrs.repeats.assign(repeats.begin(), repeats.end());
        return make<Tile>(attrs, backend, impl);
    }
    case OpType::Expand: {
        ExpandAttributes attrs;
        const auto shape = a.get_ints("shape");
        attrs.shape.assign(shape.begin(), shape.end());
        return make<Expand>(attrs, backend, impl);
    }
    case OpType::Split: {
        SplitAttributes attrs;
        attrs.axis = a.get_int("axis", 0);
        const auto split = a.get_ints("split");
        attrs.split.assign(split.begin(), split.end());
        attrs.num_outputs = a.get_int("num_outputs", 0);
        return make<Split>(attrs, backend, impl);
    }
    case OpType::TopK: {
        TopKAttributes attrs;
        attrs.type = static_cast<TopKType>(a.get_int("type", 0));
        attrs.axis = a.get_int("axis", -1);
        attrs.k = a.get_int("k", 1);
        return make<TopK>(attrs, backend, impl);
    }
    case OpType::ArgMax: {
        ArgMinMaxAttributes attrs;
        attrs.type = ArgMinMaxType::Max;
        attrs.axis = a.get_int("axis", 0);
        attrs.keepdims = a.get_int("keepdims", 0) != 0;
        return make<ArgMax>(attrs, backend, impl);
    }
    case OpType::ArgMin: {
        ArgMinMaxAttributes attrs;
        attrs.type = ArgMinMaxType::Min;
        attrs.axis = a.get_int("axis", 0);
        attrs.keepdims = a.get_int("keepdims", 0) != 0;
        return make<ArgMin>(attrs, backend, impl);
    }
    default:
        break;
    }

    error = "node '" + node.name + "' (" + node.op_type +
            "): no lowering to an nnops operator";
    return Status::unsupported;
}

}  // namespace nnops::framework::detail
