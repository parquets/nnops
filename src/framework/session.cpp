/// @file session.cpp
/// @brief Session — shape resolution, buffer placement and execution.
///
/// The planning problem is small but has one twist that decides the whole
/// structure: a Shape node produces no bytes of its own. Its output *is* its
/// input's storage under another descriptor — that is what makes Reshape,
/// Flatten, Squeeze and Identity free. So placement runs over groups of
/// aliased values (a union-find over the alias edges), not over values: a
/// group gets one buffer, and every member writes its own view over it.
///
/// Groups are then placed by liveness. Nodes run in dependency order, each
/// value's live range is the span between the node that writes it and the last
/// node that reads it, and a group is put into any pooled buffer whose current
/// occupant died before the group was born. Graph inputs and constants are not
/// part of that pool at all: their storage is the caller's tensor and the
/// graph's own bytes respectively, which is what makes feeding a model
/// zero-copy.
///
/// Shapes are resolved per call, so one session takes any batch size. The pool
/// itself persists, so a steady shape allocates nothing after the first run.

#include "internal.hpp"

#include "nnops/framework/session.hpp"

#include <algorithm>
#include <climits>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace nnops::framework {

namespace {

/// A view over `data` with `desc`'s shape, dtype and layout.
TensorView view_of(void* data, const TensorDesc& desc)
{
    return TensorView(std::span<const int64_t>(desc.dims.data(),
                                               static_cast<size_t>(desc.rank)),
                      desc.dtype, data, desc.layout);
}

bool same_shape(const TensorDesc& a, const TensorDesc& b) noexcept
{
    if (a.rank != b.rank || a.dtype != b.dtype) { return false; }
    for (int64_t i = 0; i < a.rank; ++i) {
        if (a.dims[static_cast<size_t>(i)] != b.dims[static_cast<size_t>(i)]) {
            return false;
        }
    }
    return true;
}

const char* dtype_name(DataType dtype) noexcept
{
    switch (dtype) {
    case DataType::f32:  return "f32";
    case DataType::f16:  return "f16";
    case DataType::bf16: return "bf16";
    case DataType::s8:   return "s8";
    case DataType::u8:   return "u8";
    case DataType::s32:  return "s32";
    case DataType::s64:  return "s64";
    default:             return "other";
    }
}

}  // namespace

struct Session::Impl {
    const Graph* graph = nullptr;
    SessionOptions opts;

    std::vector<std::unique_ptr<OpBase>> impls;  ///< parallel to graph->nodes()
    std::vector<int> order;                      ///< node ids, dependency order
    std::vector<ValueInfo> in_info;
    std::vector<ValueInfo> out_info;

    std::string error;
    bool ok = false;
    size_t peak = 0;

    /// Per node, the filter prepacked for the geometry it last ran at, and the
    /// geometry that blob was built for. Empty means "hand compute() the raw
    /// filter", which is what every non-conv node and every filter the operator
    /// declined to pack gets.
    std::vector<Tensor> packed_weight;
    std::vector<std::vector<int64_t>> packed_key;

    /// Persistent buffer pool; slot_bytes is its current capacity.
    std::vector<Tensor> slots;
    std::vector<size_t> slot_bytes;
    Tensor scratch;
    size_t scratch_bytes = 0;

    // ---- per-run state ----
    std::vector<TensorDesc> descs;  ///< per value id
    std::vector<int> group;         ///< per value id -> group root
    std::vector<int> source;        ///< per group root -> providing value, or -1
    std::vector<int> slot;          ///< per group root -> slot, or -1
    std::vector<int> input_index;   ///< per value id -> caller input index, or -1

    /// Prepack a constant convolution filter for the geometry this call runs at
    /// — the Winograd U where the operator's policy says the transform pays, and
    /// nothing otherwise. Cached on the plane and the channel split, which are
    /// what the prepack reads.
    void prepack_conv(int node_id, const Node& node,
                      std::span<const TensorDesc> in_descs);

    Status prepare(std::span<const Tensor> inputs, std::span<Tensor>& outputs,
                   std::vector<Tensor>* owned);
    Status execute(std::span<const Tensor> inputs, std::span<Tensor> outputs);

    void* value_ptr(int value, std::span<const Tensor> inputs);
};

void* Session::Impl::value_ptr(int value, std::span<const Tensor> inputs)
{
    const int root = group[static_cast<size_t>(value)];
    const int src = source[static_cast<size_t>(root)];
    if (src >= 0) {
        if (graph->values()[static_cast<size_t>(src)].constant) {
            auto& bytes = const_cast<std::vector<std::byte>&>(graph->constant_data(src));
            return bytes.data();
        }
        const Tensor& input =
            inputs[static_cast<size_t>(input_index[static_cast<size_t>(src)])];
        return const_cast<void*>(input.data());
    }
    Tensor& buffer = slots[static_cast<size_t>(slot[static_cast<size_t>(root)])];
    return buffer.data();
}

void Session::Impl::prepack_conv(int node_id, const Node& node,
                                 std::span<const TensorDesc> in_descs)
{
    Tensor& blob = packed_weight[static_cast<size_t>(node_id)];
    std::vector<int64_t>& key = packed_key[static_cast<size_t>(node_id)];

    if (node.kind != NodeKind::Op || node.op != OpType::Conv2D ||
        node.inputs.size() < 2 || node.inputs[1] < 0 ||
        in_descs.size() < 2 || in_descs[0].rank != 4 || in_descs[1].rank != 4) {
        return;
    }

    const ValueInfo& info = graph->values()[static_cast<size_t>(node.inputs[1])];
    if (!info.constant || (info.desc.dtype != DataType::f32 &&
                           info.desc.dtype != DataType::f16)) {
        return;  // a caller's filter, or one with no fp transform
    }

    // The transform only pays where the geometry is a k3s1 the plane can
    // amortize it over, and prepacking is not always a win on the im2col route
    // — measured, it costs ResNet-18 4.5 ms — so elsewhere compute() takes the
    // raw filter and packs per call, which this route does not notice.
    const Conv2DAttributes attrs = detail::conv2d_attributes(node);
    const int64_t group = attrs.groups > 0 ? attrs.groups : 1;
    const int64_t ic_per_group = in_descs[0].dims[1] / group;
    if (!winograd_applicable(attrs, ic_per_group, in_descs[0].dims[2],
                             in_descs[0].dims[3])) {
        blob = Tensor{};
        key.clear();
        return;
    }

    // The Winograd order is baked into U and chosen from the plane and the
    // team, so a different plane or pool is a different blob.
    std::vector<int64_t> want = {in_descs[0].dims[2], in_descs[0].dims[3], ic_per_group,
                                 static_cast<int64_t>(opts.ctx.cpu.thread_slot_count())};
    if (key == want) { return; }

    OpBase* impl = impls[static_cast<size_t>(node_id)].get();
    if (impl == nullptr) { return; }

    auto& bytes = const_cast<std::vector<std::byte>&>(graph->constant_data(node.inputs[1]));
    const TensorView filter = view_of(bytes.data(), info.desc);
    const TensorView plane = view_of(nullptr, in_descs[0]);

    // Two-step prepack: query the blob's descriptor, allocate it, fill it.
    const TensorView in[] = {filter, plane};
    TensorView query;
    TensorView q[] = {query};
    impl->prepackWeights(in, q, opts.ctx);
    if (q[0].rank() == 0) { return; }  // the operator has no packed form for it

    Tensor packed;
    const std::vector<int64_t> shape(q[0].shape(), q[0].shape() + q[0].rank());
    if (is_error(packed.reset(shape, q[0].data_type(), q[0].layout()))) {
        key.clear();
        return;
    }
    TensorView fill[] = {packed.view()};
    impl->prepackWeights(in, fill, opts.ctx);

    blob = std::move(packed);
    key = std::move(want);
}

Status Session::Impl::prepare(std::span<const Tensor> inputs,
                              std::span<Tensor>& outputs,
                              std::vector<Tensor>* owned)
{
    const auto& values = graph->values();
    const auto& nodes = graph->nodes();
    const size_t value_count = values.size();
    const int node_count = static_cast<int>(order.size());

    if (inputs.size() != graph->inputs().size()) {
        error = "the model takes " + std::to_string(graph->inputs().size()) +
                " inputs, " + std::to_string(inputs.size()) + " were given";
        return Status::invalid_argument;
    }

    descs.assign(value_count, TensorDesc{});
    input_index.assign(value_count, -1);
    std::vector<char> known(value_count, 0);

    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i].constant) {
            descs[i] = values[i].desc;
            known[i] = 1;
        }
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        const int v = graph->inputs()[i];
        const TensorDesc& given = inputs[i].desc();
        const ValueInfo& declared = values[static_cast<size_t>(v)];
        input_index[static_cast<size_t>(v)] = static_cast<int>(i);

        if (declared.desc.rank > 0 && given.rank != declared.desc.rank) {
            error = "input '" + declared.name + "' expects rank " +
                    std::to_string(declared.desc.rank) + ", got " +
                    std::to_string(given.rank);
            return Status::invalid_argument;
        }
        if (given.dtype != declared.desc.dtype) {
            error = "input '" + declared.name + "' expects dtype " +
                    dtype_name(declared.desc.dtype) + ", got " + dtype_name(given.dtype);
            return Status::invalid_argument;
        }
        for (int64_t d = 0; d < declared.desc.rank && d < given.rank; ++d) {
            const int64_t want = declared.desc.dims[static_cast<size_t>(d)];
            if (want >= 0 && want != given.dims[static_cast<size_t>(d)]) {
                error = "input '" + declared.name + "' dim " + std::to_string(d) +
                        " expects " + std::to_string(want) + ", got " +
                        std::to_string(given.dims[static_cast<size_t>(d)]);
                return Status::invalid_argument;
            }
        }
        descs[static_cast<size_t>(v)] = given;
        known[static_cast<size_t>(v)] = 1;
    }

    // ---- shapes, in dependency order ----
    std::vector<TensorDesc> in_descs;
    std::vector<TensorDesc> out_descs;
    std::vector<size_t> ws(static_cast<size_t>(node_count), 0);

    for (int k = 0; k < node_count; ++k) {
        const Node& node = nodes[static_cast<size_t>(order[static_cast<size_t>(k)])];
        in_descs.clear();
        for (int v : node.inputs) {
            if (v < 0) { continue; }
            if (known[static_cast<size_t>(v)] == 0) {
                error = "value '" + values[static_cast<size_t>(v)].name +
                        "' feeds node '" + node.name + "' but nothing produced it";
                return Status::invalid_argument;
            }
            in_descs.push_back(descs[static_cast<size_t>(v)]);
        }

        out_descs.clear();
        const Status st = detail::infer_node(node, impls[static_cast<size_t>(order[static_cast<size_t>(k)])].get(),
                                     in_descs, out_descs, error);
        if (is_error(st)) { return st; }

        size_t o = 0;
        for (int v : node.outputs) {
            if (v < 0) { continue; }
            descs[static_cast<size_t>(v)] = out_descs[o++];
            known[static_cast<size_t>(v)] = 1;
        }

        if (OpBase* impl = impls[static_cast<size_t>(order[static_cast<size_t>(k)])].get()) {
            const Status valid = impl->validate(in_descs, out_descs);
            if (is_error(valid)) {
                error = "node '" + node.name + "' (" + node.op_type + ") rejected its call: " +
                        std::string(to_string(valid));
                return valid;
            }
            ws[static_cast<size_t>(k)] = impl->getWorkspaceSize(in_descs, out_descs);
        }
        prepack_conv(order[static_cast<size_t>(k)], node, in_descs);
    }

    // A graph output nothing produced has no bytes at all — it is not a
    // zero-extent tensor but a dangling name, and it would otherwise surface as
    // a failed copy-out far from the model that caused it.
    for (int v : graph->outputs()) {
        if (v < 0 || known[static_cast<size_t>(v)] != 0) { continue; }
        error = "graph output '" + values[static_cast<size_t>(v)].name +
                "' is produced by no node";
        return Status::invalid_argument;
    }

    // ---- aliasing ----
    std::vector<int> parent(value_count);
    for (size_t i = 0; i < value_count; ++i) { parent[i] = static_cast<int>(i); }
    auto find = [&parent](int v) {
        while (parent[static_cast<size_t>(v)] != v) {
            parent[static_cast<size_t>(v)] = parent[static_cast<size_t>(parent[static_cast<size_t>(v)])];
            v = parent[static_cast<size_t>(v)];
        }
        return v;
    };
    for (const Node& node : nodes) {
        if (node.kind != NodeKind::Shape || node.alias_of < 0) { continue; }
        for (int v : node.outputs) {
            if (v < 0) { continue; }
            const int a = find(v);
            const int b = find(node.alias_of);
            if (a != b) { parent[static_cast<size_t>(a)] = b; }
        }
    }

    group.assign(value_count, -1);
    source.assign(value_count, -1);
    std::vector<size_t> group_bytes(value_count, 0);
    for (size_t i = 0; i < value_count; ++i) {
        group[i] = find(static_cast<int>(i));
        const size_t bytes = descs[i].nbytes();
        size_t& target = group_bytes[static_cast<size_t>(group[i])];
        target = std::max(target, bytes);
    }
    for (size_t i = 0; i < value_count; ++i) {
        if (!values[i].constant && input_index[i] < 0) { continue; }
        source[static_cast<size_t>(group[i])] = static_cast<int>(i);
    }

    // ---- liveness ----
    std::vector<int> first(value_count, INT_MAX);
    std::vector<int> last(value_count, -1);
    for (int k = 0; k < node_count; ++k) {
        const Node& node = nodes[static_cast<size_t>(order[static_cast<size_t>(k)])];
        for (int v : node.inputs) {
            if (v < 0) { continue; }
            first[static_cast<size_t>(v)] = std::min(first[static_cast<size_t>(v)], k);
            last[static_cast<size_t>(v)] = std::max(last[static_cast<size_t>(v)], k);
        }
        for (int v : node.outputs) {
            if (v < 0) { continue; }
            first[static_cast<size_t>(v)] = std::min(first[static_cast<size_t>(v)], k);
            last[static_cast<size_t>(v)] = std::max(last[static_cast<size_t>(v)], k);
        }
    }
    for (int v : graph->outputs()) {
        if (v < 0) { continue; }
        first[static_cast<size_t>(v)] = std::min(first[static_cast<size_t>(v)], 0);
        last[static_cast<size_t>(v)] = node_count;
    }

    std::vector<int> group_first(value_count, INT_MAX);
    std::vector<int> group_last(value_count, -1);
    for (size_t i = 0; i < value_count; ++i) {
        const size_t root = static_cast<size_t>(group[i]);
        if (first[i] == INT_MAX) { continue; }
        group_first[root] = std::min(group_first[root], first[i]);
        group_last[root] = std::max(group_last[root], last[i]);
    }

    // ---- placement ----
    std::vector<int> candidates;
    for (size_t i = 0; i < value_count; ++i) {
        if (group_first[i] == INT_MAX || source[i] >= 0) { continue; }
        if (group_bytes[i] == 0) { continue; }
        candidates.push_back(static_cast<int>(i));
    }
    std::sort(candidates.begin(), candidates.end(), [&group_first](int a, int b) {
        return group_first[static_cast<size_t>(a)] < group_first[static_cast<size_t>(b)];
    });

    slot.assign(value_count, -1);
    std::vector<int> slot_last(slots.size(), -1);
    std::vector<char> slot_used(slots.size(), 0);
    for (int root : candidates) {
        const size_t bytes = group_bytes[static_cast<size_t>(root)];
        int chosen = -1;
        for (size_t j = 0; j < slots.size(); ++j) {
            if (slot_used[j] != 0) { continue; }
            if (slot_last[j] >= group_first[static_cast<size_t>(root)]) { continue; }
            if (slot_bytes[j] < bytes) { continue; }
            chosen = static_cast<int>(j);
            break;
        }
        if (chosen < 0) {
            chosen = static_cast<int>(slots.size());
            slots.emplace_back();
            slot_bytes.push_back(0);
            slot_last.push_back(-1);
            slot_used.push_back(0);
        }
        if (slot_bytes[static_cast<size_t>(chosen)] < bytes) {
            const int64_t shape[1] = {static_cast<int64_t>(bytes)};
            const Status st = slots[static_cast<size_t>(chosen)].reset(shape, DataType::u8);
            if (is_error(st)) {
                error = "out of memory planning a " + std::to_string(bytes) + "-byte buffer";
                return st;
            }
            slot_bytes[static_cast<size_t>(chosen)] = bytes;
        }
        slot_used[static_cast<size_t>(chosen)] = 1;
        slot_last[static_cast<size_t>(chosen)] = group_last[static_cast<size_t>(root)];
        slot[static_cast<size_t>(root)] = chosen;
    }

    size_t pool = 0;
    for (size_t j = 0; j < slots.size(); ++j) {
        if (slot_used[j] != 0) { pool += slot_bytes[j]; }
    }
    peak = std::max(peak, pool);

    // ---- workspace ----
    size_t max_ws = 0;
    for (size_t bytes : ws) { max_ws = std::max(max_ws, bytes); }
    if (max_ws > scratch_bytes) {
        const int64_t shape[1] = {static_cast<int64_t>(max_ws)};
        const Status st = scratch.reset(shape, DataType::u8);
        if (is_error(st)) {
            error = "out of memory planning a " + std::to_string(max_ws) + "-byte workspace";
            return st;
        }
        scratch_bytes = max_ws;
    }

    // ---- outputs ----
    const size_t out_count = graph->outputs().size();
    if (owned != nullptr) {
        owned->resize(out_count);
        for (size_t i = 0; i < out_count; ++i) {
            const TensorDesc& want = descs[static_cast<size_t>(graph->outputs()[i])];
            Tensor& slot_out = (*owned)[i];
            if (!same_shape(slot_out.desc(), want)) {
                const Status st = slot_out.reset(want);
                if (is_error(st)) { return st; }
            }
        }
        outputs = std::span<Tensor>(owned->data(), owned->size());
    }

    if (outputs.size() != out_count) {
        error = "the model returns " + std::to_string(out_count) + " outputs, " +
                std::to_string(outputs.size()) + " slots were given";
        return Status::invalid_argument;
    }
    for (size_t i = 0; i < out_count; ++i) {
        const ValueInfo& info = values[static_cast<size_t>(graph->outputs()[i])];
        if (!same_shape(outputs[i].desc(), descs[static_cast<size_t>(graph->outputs()[i])])) {
            error = "output '" + info.name + "' does not match its buffer's shape or dtype";
            return Status::invalid_argument;
        }
    }
    return Status::ok;
}

Status Session::Impl::execute(std::span<const Tensor> inputs,
                              std::span<Tensor> outputs)
{
    const auto& nodes = graph->nodes();
    std::vector<TensorView> in_views;
    std::vector<TensorView> out_views;

    for (int node_id : order) {
        const Node& node = nodes[static_cast<size_t>(node_id)];
        OpBase* impl = impls[static_cast<size_t>(node_id)].get();

        // A conv whose filter the planner prepacked reads the blob in place of
        // the graph's constant — the same operand slot, a transformed value.
        const Tensor& packed = packed_weight[static_cast<size_t>(node_id)];
        in_views.clear();
        for (int v : node.inputs) {
            if (v < 0) { continue; }
            if (in_views.size() == 1 && !packed.is_empty()) {
                in_views.push_back(packed.view());
            } else {
                in_views.push_back(view_of(value_ptr(v, inputs), descs[static_cast<size_t>(v)]));
            }
        }
        out_views.clear();
        for (int v : node.outputs) {
            if (v < 0) { continue; }
            out_views.push_back(view_of(value_ptr(v, inputs), descs[static_cast<size_t>(v)]));
        }

        const Status st = detail::run_node(node, impl, out_views, in_views, opts.ctx,
                                   scratch_bytes > 0 ? scratch.data() : nullptr);
        if (is_error(st)) {
            error = "node '" + (node.name.empty() ? node.op_type : node.name) +
                    "' failed: " + std::string(to_string(st));
            return st;
        }
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
        const int v = graph->outputs()[i];
        const Status st = outputs[i].copy_from(value_ptr(v, inputs),
                                               descs[static_cast<size_t>(v)].nbytes());
        if (is_error(st)) {
            error = "output '" + graph->values()[static_cast<size_t>(v)].name +
                    "' could not be copied out";
            return st;
        }
    }
    return Status::ok;
}

// ---- Session ---------------------------------------------------------------

Session::Session(const Model& model, const SessionOptions& options)
    : impl_(std::make_unique<Impl>())
{
    impl_->graph = &model.graph();
    impl_->opts = options;

    if (!model.valid()) {
        impl_->error = "the model did not load";
        return;
    }

    const Graph& graph = model.graph();
    impl_->order = graph.topological_order();
    if (impl_->order.size() != graph.nodes().size()) {
        impl_->error = "the graph has a cycle";
        return;
    }

    impl_->impls.resize(graph.nodes().size());
    impl_->packed_weight.resize(graph.nodes().size());
    impl_->packed_key.resize(graph.nodes().size());
    for (size_t i = 0; i < graph.nodes().size(); ++i) {
        std::string error;
        const Status st = detail::lower_node(graph.nodes()[i], options.backend,
                                             impl_->impls[i], error);
        if (is_error(st)) {
            impl_->error = error;
            return;
        }
    }

    for (int v : graph.inputs()) {
        impl_->in_info.push_back(graph.values()[static_cast<size_t>(v)]);
    }
    for (int v : graph.outputs()) {
        impl_->out_info.push_back(graph.values()[static_cast<size_t>(v)]);
    }
    impl_->ok = true;
}

Session::~Session() = default;
Session::Session(Session&&) noexcept = default;
Session& Session::operator=(Session&&) noexcept = default;

Status Session::run(std::span<const Tensor> inputs, std::vector<Tensor>& outputs)
{
    std::span<Tensor> view;
    const Status st = impl_->prepare(inputs, view, &outputs);
    if (is_error(st)) { return st; }
    return impl_->execute(inputs, view);
}

Status Session::run(std::span<const Tensor> inputs, std::span<Tensor> outputs)
{
    std::span<Tensor> view = outputs;
    const Status st = impl_->prepare(inputs, view, nullptr);
    if (is_error(st)) { return st; }
    return impl_->execute(inputs, view);
}

const std::vector<ValueInfo>& Session::input_info() const noexcept
{
    return impl_->in_info;
}

const std::vector<ValueInfo>& Session::output_info() const noexcept
{
    return impl_->out_info;
}

const std::string& Session::last_error() const noexcept
{
    return impl_->error;
}

bool Session::ok() const noexcept
{
    return impl_->ok;
}

size_t Session::peak_buffer_bytes() const noexcept
{
    return impl_->peak;
}

}  // namespace nnops::framework
