#pragma once
/// @file model_prof.hpp
/// @brief Wall-clock accounting for the model runners: one table per layer
///        (named like the checkpoints) and one per operator.
///
/// Both tables stay off until the runner starts the pass it means to report,
/// so the untimed warm-up — the pass that pays the Winograd weight transform
/// and the first-touch allocation — never lands in the numbers.

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace model {

using Clock = std::chrono::steady_clock;

/// Which table a scope charges its time to. They are separate on purpose: an
/// operator runs inside a layer, so a nested scope is counted once in each
/// table and never twice in the same one.
enum class ProfTable : unsigned char { Layer, Op };

struct ProfRow {
    std::string name;
    double ms = 0.0;
    long calls = 0;
};

namespace detail {

/// Rows keep call order — the network's, not the alphabet's (a sorted map
/// would print features.10 before features.2).
struct ProfRows {
    std::vector<ProfRow> rows;
    std::map<std::string, std::size_t> index;

    void add(const char* name, double ms)
    {
        const auto [it, fresh] = index.try_emplace(name, rows.size());
        if (fresh) { rows.push_back(ProfRow{name, 0.0, 0}); }
        rows[it->second].ms += ms;
        ++rows[it->second].calls;
    }
};

inline ProfRows& rows(ProfTable t)
{
    static ProfRows layers;
    static ProfRows ops;
    return t == ProfTable::Layer ? layers : ops;
}

inline bool& enabled()
{
    static bool on = false;
    return on;
}

}  // namespace detail

/// Adds this scope's wall clock to @p name when it goes out of scope. The
/// default table is the operator one, which is what the op wrappers use.
class ProfScope {
public:
    explicit ProfScope(const char* name, ProfTable table = ProfTable::Op)
        : name_(name), table_(table)
    {
        if (detail::enabled()) { t0_ = Clock::now(); }
    }

    ~ProfScope()
    {
        if (!detail::enabled()) { return; }
        const double ms =
            std::chrono::duration<double, std::milli>(Clock::now() - t0_).count();
        detail::rows(table_).add(name_, ms);
    }

    ProfScope(const ProfScope&) = delete;
    ProfScope& operator=(const ProfScope&) = delete;

private:
    const char* name_;
    ProfTable table_;
    Clock::time_point t0_{};
};

/// Run @p fn and charge its wall clock to layer @p name.
template <class Fn>
auto layer(const char* name, Fn&& fn) -> decltype(fn())
{
    ProfScope scope(name, ProfTable::Layer);
    return fn();
}

/// Turn the tables on for the pass about to run, dropping anything collected
/// before it (the warm-up never collects, but a second run in one process
/// would otherwise accumulate).
inline void prof_reset()
{
    detail::rows(ProfTable::Layer) = detail::ProfRows{};
    detail::rows(ProfTable::Op) = detail::ProfRows{};
    detail::enabled() = true;
}

/// Print both tables against @p total_ms, the runner's own forward timing.
/// The layer table's last row is what the layers do not explain, which after
/// the activation free list is essentially the per-checkpoint comparison.
inline void prof_report(const char* title, double total_ms)
{
    detail::enabled() = false;

    auto share = [total_ms](double ms) {
        return total_ms > 0.0 ? 100.0 * ms / total_ms : 0.0;
    };

    for (ProfTable t : {ProfTable::Layer, ProfTable::Op}) {
        const auto& set = detail::rows(t);
        if (set.rows.empty()) { continue; }
        std::printf("%s: %s\n", title,
                    t == ProfTable::Layer ? "per layer" : "per op");
        double sum = 0.0;
        for (const auto& r : set.rows) {
            std::printf("  %-18s %8.2f ms %6.1f%%  x%ld\n", r.name.c_str(), r.ms,
                        share(r.ms), r.calls);
            sum += r.ms;
        }
        if (t == ProfTable::Layer) {
            std::printf("  %-18s %8.2f ms %6.1f%%\n", "(other)",
                        total_ms - sum, share(total_ms - sum));
        }
    }
}

}  // namespace model
