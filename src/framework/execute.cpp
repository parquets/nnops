/// @file execute.cpp
/// @brief The per-node half of the runtime: shape inference, dispatch and the
///        framework's own dtype conversion.
///
/// Two of these three are deliberately thin. Shape inference forwards to the
/// operator's getOutputTensorDesc() and execution to its compute(), so a graph
/// can never disagree with a hand-written call about what a Conv2D produces —
/// the framing the operator author tested is the framing the graph uses.
///
/// What the framework does own is the two metadata kinds. A Shape node
/// re-derives its output descriptor here (Reshape semantics, Flatten's two-way
/// split, Squeeze/Unsqueeze) without touching a byte — the planner has already
/// pointed the output at the input's storage, so execution is a no-op. Cast is
/// the one conversion no nnops operator covers, and it is a plain elementwise
/// walk through f32.

#include "internal.hpp"

#include "nnops/detail/bf16.hpp"
#include "nnops/detail/half.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace nnops::framework::detail {

namespace {

/// Product of dims [0, axis); the other side is numel() / this.
int64_t dims_prefix(const TensorDesc& desc, int64_t axis) noexcept
{
    int64_t n = 1;
    for (int64_t i = 0; i < axis; ++i) { n *= desc.dims[static_cast<size_t>(i)]; }
    return n;
}

/// ONNX axes wrap from the end; this resolves them against `rank`.
std::vector<int64_t> resolve_axes(std::span<const int64_t> axes, int64_t rank)
{
    std::vector<int64_t> out;
    out.reserve(axes.size());
    for (int64_t axis : axes) {
        out.push_back(axis < 0 ? axis + rank : axis);
    }
    std::sort(out.begin(), out.end());
    return out;
}

Status shape_of_shape_node(const Node& node, const TensorDesc& in,
                           TensorDesc& out, std::string& error);

}  // namespace

Status infer_node(const Node& node, const OpBase* impl,
                  std::span<const TensorDesc> inputs,
                  std::vector<TensorDesc>& outputs, std::string& error)
{
    outputs.clear();
    if (inputs.empty() && node.kind != NodeKind::Op) {
        error = "node '" + node.name + "' has no input";
        return Status::invalid_argument;
    }

    switch (node.kind) {
    case NodeKind::Cast: {
        TensorDesc desc = inputs[0];
        desc.dtype = node.target_dtype;
        desc.row_stride_elems = 0;
        outputs.push_back(desc);
        return Status::ok;
    }
    case NodeKind::Shape: {
        TensorDesc desc;
        const Status st = shape_of_shape_node(node, inputs[0], desc, error);
        if (is_error(st)) { return st; }
        outputs.push_back(desc);
        return Status::ok;
    }
    case NodeKind::Op:
        break;
    }

    if (impl == nullptr) {
        error = "node '" + node.name + "' has no operator to infer its shape";
        return Status::not_initialized;
    }
    outputs = impl->getOutputTensorDesc(inputs);
    if (outputs.size() != node.outputs.size()) {
        error = "node '" + node.name + "' (" + node.op_type + ") infers " +
                std::to_string(outputs.size()) + " outputs for " +
                std::to_string(node.outputs.size()) + " slots";
        return Status::invalid_argument;
    }
    for (const TensorDesc& desc : outputs) {
        for (int64_t i = 0; i < desc.rank; ++i) {
            if (desc.dims[static_cast<size_t>(i)] < 0) {
                error = "node '" + node.name + "' (" + node.op_type +
                        ") could not resolve a dynamic extent";
                return Status::invalid_argument;
            }
        }
    }
    return Status::ok;
}

Status run_node(const Node& node, OpBase* impl, std::span<TensorView> outputs,
                std::span<const TensorView> inputs, const ComputeContext& ctx,
                void* workspace)
{
    switch (node.kind) {
    case NodeKind::Shape:
        // The planner aliased the output onto the input's storage; there is
        // nothing to move and no bytes to convert.
        return Status::ok;
    case NodeKind::Cast:
        return cast_tensor(inputs[0], outputs[0]);
    case NodeKind::Op:
        break;
    }
    if (impl == nullptr) { return Status::not_initialized; }
    return impl->compute(outputs, inputs, ctx, workspace);
}

namespace {

Status shape_of_shape_node(const Node& node, const TensorDesc& in,
                           TensorDesc& out, std::string& error)
{
    const std::string& op = node.op_type;
    out = TensorDesc{};
    out.dtype = in.dtype;
    out.layout = in.layout;

    if (op == "Identity" || op == "Dropout") {
        out = in;
        out.row_stride_elems = 0;
        return Status::ok;
    }

    if (op == "Flatten") {
        if (in.rank < 0) {
            error = "Flatten on an unranked input";
            return Status::invalid_argument;
        }
        int64_t axis = node.attrs.get_int("axis", 1);
        if (axis < 0) { axis += in.rank; }
        axis = std::clamp<int64_t>(axis, 0, in.rank);
        const int64_t outer = dims_prefix(in, axis);
        out.rank = 2;
        out.dims.push_back(outer);
        out.dims.push_back(outer == 0 ? 0 : in.numel() / outer);
        return Status::ok;
    }

    if (op == "Reshape") {
        const auto shape = node.attrs.get_ints("shape");
        if (shape.empty()) {
            error = "Reshape without a target shape";
            return Status::invalid_argument;
        }
        const bool allowzero = node.attrs.get_int("allowzero", 0) != 0;
        std::vector<int64_t> target(shape.begin(), shape.end());

        int64_t known = 1;
        int64_t inferred = -1;
        for (size_t i = 0; i < target.size(); ++i) {
            int64_t& dim = target[i];
            if (dim == 0 && !allowzero && i < static_cast<size_t>(in.rank)) {
                dim = in.dims[i];
            }
            if (dim == -1) {
                if (inferred >= 0) {
                    error = "Reshape names -1 twice";
                    return Status::invalid_argument;
                }
                inferred = static_cast<int64_t>(i);
            } else {
                known *= dim;
            }
        }
        if (inferred >= 0) {
            if (known == 0 || in.numel() % known != 0) {
                error = "Reshape's target shape does not divide its element count";
                return Status::invalid_argument;
            }
            target[static_cast<size_t>(inferred)] = in.numel() / known;
        } else if (known != in.numel()) {
            error = "Reshape's target shape holds " + std::to_string(known) +
                    " elements, the input " + std::to_string(in.numel());
            return Status::invalid_argument;
        }
        out.rank = static_cast<int64_t>(target.size());
        out.dims = std::span<const int64_t>(target);
        return Status::ok;
    }

    if (op == "Squeeze") {
        std::vector<int64_t> axes = resolve_axes(node.attrs.get_ints("axes"), in.rank);
        std::vector<int64_t> dims;
        for (int64_t i = 0; i < in.rank; ++i) {
            const bool drop = axes.empty()
                ? in.dims[static_cast<size_t>(i)] == 1
                : std::find(axes.begin(), axes.end(), i) != axes.end();
            if (!drop) { dims.push_back(in.dims[static_cast<size_t>(i)]); }
        }
        out.rank = static_cast<int64_t>(dims.size());
        out.dims = std::span<const int64_t>(dims);
        return Status::ok;
    }

    if (op == "Unsqueeze") {
        const auto axes = resolve_axes(node.attrs.get_ints("axes"), in.rank + 1);
        if (axes.empty()) {
            error = "Unsqueeze without axes";
            return Status::invalid_argument;
        }
        std::vector<int64_t> dims;
        size_t next = 0;
        const int64_t out_rank = in.rank + static_cast<int64_t>(axes.size());
        for (int64_t i = 0; i < out_rank; ++i) {
            if (next < axes.size() && axes[next] == i) {
                dims.push_back(1);
                ++next;
            } else if (i - static_cast<int64_t>(next) < in.rank) {
                dims.push_back(in.dims[static_cast<size_t>(i - static_cast<int64_t>(next))]);
            } else {
                dims.push_back(1);
            }
        }
        out.rank = out_rank;
        out.dims = std::span<const int64_t>(dims);
        return Status::ok;
    }

    error = "shape node '" + node.name + "' (" + op + ") has no inference rule";
    return Status::unsupported;
}

// ---- Cast ------------------------------------------------------------------

float load_element(const void* data, size_t index, DataType dtype) noexcept
{
    switch (dtype) {
    case DataType::f32:  return static_cast<const float*>(data)[index];
    case DataType::f16:  return backend::cpu::half_to_float(backend::cpu::half_from_bits(
                                static_cast<const uint16_t*>(data)[index]));
    case DataType::bf16: return bf16_to_float(static_cast<const uint16_t*>(data)[index]);
    case DataType::s8:   return static_cast<float>(static_cast<const int8_t*>(data)[index]);
    case DataType::u8:   return static_cast<float>(static_cast<const uint8_t*>(data)[index]);
    case DataType::s32:  return static_cast<float>(static_cast<const int32_t*>(data)[index]);
    case DataType::s64:  return static_cast<float>(static_cast<const int64_t*>(data)[index]);
    default:             return 0.0f;
    }
}

void store_element(void* data, size_t index, DataType dtype, float value) noexcept
{
    switch (dtype) {
    case DataType::f32:
        static_cast<float*>(data)[index] = value;
        break;
    case DataType::f16:
        static_cast<uint16_t*>(data)[index] = backend::cpu::half_to_bits(
            backend::cpu::float_to_half(value));
        break;
    case DataType::bf16:
        static_cast<uint16_t*>(data)[index] = float_to_bf16(value);
        break;
    case DataType::s8:
        static_cast<int8_t*>(data)[index] = static_cast<int8_t>(std::lround(value));
        break;
    case DataType::u8:
        static_cast<uint8_t*>(data)[index] = static_cast<uint8_t>(std::lround(value));
        break;
    case DataType::s32:
        static_cast<int32_t*>(data)[index] = static_cast<int32_t>(std::lround(value));
        break;
    case DataType::s64:
        static_cast<int64_t*>(data)[index] = static_cast<int64_t>(std::llround(value));
        break;
    default:
        break;
    }
}

}  // namespace

Status cast_tensor(const TensorView& src, TensorView& dst)
{
    const int64_t count = src.numel();
    if (dst.numel() != count) { return Status::invalid_argument; }
    if (src.is_empty() || dst.is_empty() || count == 0) { return Status::ok; }

    const size_t src_size = data_type_size(src.data_type());
    const size_t dst_size = data_type_size(dst.data_type());
    if (src_size == 0 || dst_size == 0) { return Status::unsupported; }

    // Both views are dense here, but the row pitch is honoured anyway: the
    // helper is the natural place a caller outside the planner would reach for.
    const int64_t rank = src.rank();
    const int64_t last = rank > 0 ? src.shape(rank - 1) : count;
    const int64_t rows = last > 0 ? count / last : 0;
    const int64_t src_pitch = src.pitch();
    const int64_t dst_pitch = dst.pitch();
    const bool flat = (src_pitch == last * static_cast<int64_t>(src_size) &&
                       dst_pitch == last * static_cast<int64_t>(dst_size));

    if (flat) {
        auto* src_bytes = src.ptr<std::byte>();
        auto* dst_bytes = dst.ptr<std::byte>();
        for (int64_t i = 0; i < count; ++i) {
            store_element(dst_bytes, static_cast<size_t>(i), dst.data_type(),
                          load_element(src_bytes, static_cast<size_t>(i), src.data_type()));
        }
        return Status::ok;
    }

    for (int64_t row = 0; row < rows; ++row) {
        const auto* src_row = src.ptr<std::byte>(static_cast<size_t>(row));
        auto* dst_row = dst.ptr<std::byte>(static_cast<size_t>(row));
        for (int64_t i = 0; i < last; ++i) {
            store_element(dst_row, static_cast<size_t>(i), dst.data_type(),
                          load_element(src_row, static_cast<size_t>(i), src.data_type()));
        }
    }
    return Status::ok;
}

}  // namespace nnops::framework::detail
