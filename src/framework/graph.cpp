/// @file graph.cpp
/// @brief Attribute storage, value bookkeeping and the graph's two batch
///        operations (topological order, dead-node elimination).

#include "nnops/framework/graph.hpp"

#include <algorithm>
#include <cstring>

namespace nnops::framework {

// ---- Attributes ------------------------------------------------------------

Attribute* Attributes::find(std::string_view name) noexcept
{
    for (auto& item : items_) {
        if (item.first == name) { return &item.second; }
    }
    return nullptr;
}

const Attribute* Attributes::find(std::string_view name) const noexcept
{
    for (const auto& item : items_) {
        if (item.first == name) { return &item.second; }
    }
    return nullptr;
}

Attribute& Attributes::obtain(std::string_view name)
{
    if (Attribute* existing = find(name)) { return *existing; }
    items_.emplace_back(std::string(name), Attribute{});
    return items_.back().second;
}

bool Attributes::has(std::string_view name) const noexcept
{
    return find(name) != nullptr;
}

int64_t Attributes::get_int(std::string_view name, int64_t fallback) const noexcept
{
    const Attribute* a = find(name);
    return (a != nullptr && a->type == Attribute::Type::Int) ? a->i : fallback;
}

float Attributes::get_float(std::string_view name, float fallback) const noexcept
{
    const Attribute* a = find(name);
    return (a != nullptr && a->type == Attribute::Type::Float) ? a->f : fallback;
}

std::string_view Attributes::get_string(std::string_view name) const noexcept
{
    const Attribute* a = find(name);
    return (a != nullptr && a->type == Attribute::Type::String)
        ? std::string_view(a->s) : std::string_view();
}

std::span<const int64_t> Attributes::get_ints(std::string_view name) const noexcept
{
    const Attribute* a = find(name);
    return (a != nullptr && a->type == Attribute::Type::Ints)
        ? std::span<const int64_t>(a->ints) : std::span<const int64_t>();
}

std::span<const float> Attributes::get_floats(std::string_view name) const noexcept
{
    const Attribute* a = find(name);
    return (a != nullptr && a->type == Attribute::Type::Floats)
        ? std::span<const float>(a->floats) : std::span<const float>();
}

void Attributes::set_int(std::string_view name, int64_t value)
{
    Attribute& a = obtain(name);
    a = Attribute{};
    a.type = Attribute::Type::Int;
    a.i = value;
}

void Attributes::set_float(std::string_view name, float value)
{
    Attribute& a = obtain(name);
    a = Attribute{};
    a.type = Attribute::Type::Float;
    a.f = value;
}

void Attributes::set_string(std::string_view name, std::string value)
{
    Attribute& a = obtain(name);
    a = Attribute{};
    a.type = Attribute::Type::String;
    a.s = std::move(value);
}

void Attributes::set_ints(std::string_view name, std::vector<int64_t> value)
{
    Attribute& a = obtain(name);
    a = Attribute{};
    a.type = Attribute::Type::Ints;
    a.ints = std::move(value);
}

void Attributes::set_floats(std::string_view name, std::vector<float> value)
{
    Attribute& a = obtain(name);
    a = Attribute{};
    a.type = Attribute::Type::Floats;
    a.floats = std::move(value);
}

void Attributes::erase(std::string_view name)
{
    for (auto it = items_.begin(); it != items_.end(); ++it) {
        if (it->first == name) {
            items_.erase(it);
            return;
        }
    }
}

// ---- Graph -----------------------------------------------------------------

int Graph::add_value(ValueInfo info)
{
    values_.push_back(std::move(info));
    constant_data_.emplace_back();
    return static_cast<int>(values_.size()) - 1;
}

int Graph::find_value(std::string_view name) const noexcept
{
    for (size_t i = 0; i < values_.size(); ++i) {
        if (values_[i].name == name) { return static_cast<int>(i); }
    }
    return -1;
}

std::vector<int> Graph::topological_order() const
{
    const size_t n = nodes_.size();
    std::vector<int> producer(values_.size(), -1);
    for (size_t i = 0; i < n; ++i) {
        for (int v : nodes_[i].outputs) {
            if (v >= 0) { producer[static_cast<size_t>(v)] = static_cast<int>(i); }
        }
    }

    std::vector<std::vector<int>> consumers(n);
    std::vector<int> indegree(n, 0);
    for (size_t i = 0; i < n; ++i) {
        for (int v : nodes_[i].inputs) {
            if (v < 0) { continue; }
            const int p = producer[static_cast<size_t>(v)];
            // A self-reference is a malformed graph, not an ordering problem;
            // letting it through here surfaces as a cycle below.
            if (p >= 0) {
                consumers[static_cast<size_t>(p)].push_back(static_cast<int>(i));
                ++indegree[i];
            }
        }
    }

    std::vector<int> order;
    order.reserve(n);
    std::vector<int> ready;
    for (size_t i = 0; i < n; ++i) {
        if (indegree[i] == 0) { ready.push_back(static_cast<int>(i)); }
    }
    while (!ready.empty()) {
        const int i = ready.back();
        ready.pop_back();
        order.push_back(i);
        for (int j : consumers[static_cast<size_t>(i)]) {
            if (--indegree[static_cast<size_t>(j)] == 0) { ready.push_back(j); }
        }
    }

    if (order.size() != n) { return {}; }  // cycle
    return order;
}

void Graph::remove_dead_nodes()
{
    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<char> used(values_.size(), 0);
        for (int v : outputs_) {
            if (v >= 0) { used[static_cast<size_t>(v)] = 1; }
        }
        for (const auto& node : nodes_) {
            for (int v : node.inputs) {
                if (v >= 0) { used[static_cast<size_t>(v)] = 1; }
            }
        }

        std::vector<Node> kept;
        kept.reserve(nodes_.size());
        for (auto& node : nodes_) {
            bool keep = false;
            for (int v : node.outputs) {
                if (v >= 0 && used[static_cast<size_t>(v)] != 0) {
                    keep = true;
                    break;
                }
            }
            if (keep) {
                kept.push_back(std::move(node));
            } else {
                changed = true;
            }
        }
        nodes_ = std::move(kept);
    }
}

}  // namespace nnops::framework
