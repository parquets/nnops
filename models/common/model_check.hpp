#pragma once
/// @file model_check.hpp
/// @brief Compare a computed activation against the PyTorch reference recorded
///        in the artifact, and report the per-checkpoint error.
///
/// The tolerance is the usual scaled one — element i passes when
///   |actual - expected| <= atol + rtol * |expected|
/// so a checkpoint whose values are all near zero is held to @p atol, and one
/// with large values is held to a relative bound. Errors are reported both
/// absolute and relative to the reference's peak magnitude, which is the number
/// worth quoting for a whole-network comparison.

#include "model_io.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace model {

namespace detail {

/// One sweep of the comparison, over @p n elements eight at a time.
struct Diff {
    float max_abs = 0.0f;
    float ref_scale = 0.0f;
    float margin = 0.0f;  ///< max over i of |a-e| - (atol + rtol·|e|)
};

/// The two maxima the report needs, plus the worst tolerance margin.
///
/// Reporting the margin instead of counting failures is what keeps the loop
/// branch-free: a checkpoint that passes — nearly all of them, on every run —
/// is settled in one pass over the data, and only a failing one pays for the
/// second pass that counts the offending elements.
///
/// The arithmetic is f32, which is the check's whole cost: the checkpoints
/// differ by ~1e-6 relative and the tolerance is 1e-3, so a comparison that
/// rounds in its last bit cannot flip a verdict. (A checkpoint within one ulp
/// of its threshold is not a verdict worth trusting in either precision.)
inline Diff sweep(const float* a, const float* e, int64_t n, float atol,
                  float rtol)
{
    using namespace nnops::simd;
    const v_f32x8 vatol = v_set1_f32x8(atol);
    const v_f32x8 vrtol = v_set1_f32x8(rtol);
    v_f32x8 vmax_err = v_zero_f32x8();
    v_f32x8 vref_scale = v_zero_f32x8();
    v_f32x8 vmargin = v_set1_f32x8(-3.0e38f);

    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const v_f32x8 va = v_load_f32x8(a + i);
        const v_f32x8 ve = v_load_f32x8(e + i);
        const v_f32x8 err = v_abs(v_sub(va, ve));
        const v_f32x8 want = v_abs(ve);
        vmax_err = v_max(vmax_err, err);
        vref_scale = v_max(vref_scale, want);
        vmargin = v_max(vmargin, v_sub(err, v_add(vatol, v_mul(vrtol, want))));
    }

    // Horizontal reduction, scalar: eight lanes, then the ragged tail.
    float lanes[8];
    float max_err = 0.0f;
    v_store(lanes, vmax_err);
    for (int l = 0; l < 8; ++l) { max_err = std::max(max_err, lanes[l]); }
    float ref_scale = 0.0f;
    v_store(lanes, vref_scale);
    for (int l = 0; l < 8; ++l) { ref_scale = std::max(ref_scale, lanes[l]); }
    float margin = -3.0e38f;
    v_store(lanes, vmargin);
    for (int l = 0; l < 8; ++l) { margin = std::max(margin, lanes[l]); }
    for (; i < n; ++i) {
        const float err = std::fabs(a[i] - e[i]);
        const float want = std::fabs(e[i]);
        max_err = std::max(max_err, err);
        ref_scale = std::max(ref_scale, want);
        margin = std::max(margin, err - (atol + rtol * want));
    }
    return Diff{max_err, ref_scale, margin};
}

}  // namespace detail

struct CheckRow {
    std::string name;
    double max_abs = 0.0;   ///< max |actual - expected|
    double ref_scale = 0.0; ///< max |expected| — the denominator of max_rel
    int64_t bad = 0;        ///< elements outside the tolerance
    bool shape_ok = true;

    bool pass() const { return shape_ok && bad == 0; }
    double max_rel() const { return ref_scale > 0.0 ? max_abs / ref_scale : max_abs; }
};

class ModelCheck {
public:
    /// @p verbose false keeps the per-checkpoint lines off, for the untimed
    /// warm-up pass a runner does to pay the model-load costs (weight
    /// transforms, first-touch allocation) before it starts its clock.
    ModelCheck(double atol, double rtol, bool verbose = true)
        : atol_(atol), rtol_(rtol), verbose_(verbose) {}

    /// Turn the comparison off. A timed pass wants the model's own time and
    /// nothing else, so a runner that has already verified its checkpoints on
    /// an untimed pass runs the clocked one with this set — the sweep over
    /// every activation is 8-11% of a pass, which torch and ORT do not pay.
    void set_enabled(bool on) { enabled_ = on; }
    bool enabled() const { return enabled_; }

    /// Compare @p actual against the artifact tensor @p ref_name.
    void compare(const std::string& name, const nnops::TensorView& actual,
                 const nnops::TensorView& expected)
    {
        if (!enabled_) { return; }
        CheckRow row;
        row.name = name;

        if (!same_shape(actual, expected)) {
            row.shape_ok = false;
            if (verbose_) {
                std::printf("  %-24s SHAPE MISMATCH\n", name.c_str());
            }
            rows_.push_back(row);
            return;
        }
        if (!dense_planar(actual) || !dense_planar(expected)) {
            row.shape_ok = false;
            if (verbose_) {
                std::printf("  %-24s LAYOUT: expected dense planar, got packed "
                            "or pitch-padded\n", name.c_str());
            }
            rows_.push_back(row);
            return;
        }
        if (actual.data_type() != nnops::DataType::f32) {
            row.shape_ok = false;
            if (verbose_) {
                std::printf("  %-24s DTYPE: only f32 is compared\n", name.c_str());
            }
            rows_.push_back(row);
            return;
        }

        const auto* a = actual.ptr<float>();
        const auto* e = expected.ptr<float>();
        const int64_t n = actual.numel();
        const auto atol = static_cast<float>(atol_);
        const auto rtol = static_cast<float>(rtol_);
        const detail::Diff d = detail::sweep(a, e, n, atol, rtol);
        row.max_abs = d.max_abs;
        row.ref_scale = d.ref_scale;
        if (d.margin > 0.0f) {
            // Some element is outside the tolerance; count them. Only a failing
            // checkpoint gets here, and it is the one that wants the count.
            for (int64_t i = 0; i < n; ++i) {
                if (std::fabs(a[i] - e[i]) > atol + rtol * std::fabs(e[i])) {
                    ++row.bad;
                }
            }
        }

        if (verbose_) {
            std::printf("  %-24s %-4s max|d|=%.3e  rel=%.3e  bad=%lld/%lld\n",
                        name.c_str(), row.pass() ? "ok" : "FAIL", row.max_abs,
                        row.max_rel(), static_cast<long long>(row.bad),
                        static_cast<long long>(n));
        }
        rows_.push_back(row);
    }

    /// Print the summary line; returns the number of failed checkpoints.
    int report(const char* model) const
    {
        int failed = 0;
        double worst_rel = 0.0;
        double worst_abs = 0.0;
        for (const auto& row : rows_) {
            if (!row.pass()) { ++failed; }
            worst_rel = std::max(worst_rel, row.max_rel());
            worst_abs = std::max(worst_abs, row.max_abs);
        }
        std::printf("%s: %zu checkpoints, %d failed; worst max|d|=%.3e "
                    "rel=%.3e (atol=%g rtol=%g)\n",
                    model, rows_.size(), failed, worst_abs, worst_rel, atol_,
                    rtol_);
        return failed;
    }

    size_t size() const { return rows_.size(); }

private:
    static bool same_shape(const nnops::TensorView& a,
                           const nnops::TensorView& b)
    {
        if (a.rank() != b.rank()) { return false; }
        for (int64_t d = 0; d < a.rank(); ++d) {
            if (a.shape(d) != b.shape(d)) { return false; }
        }
        return true;
    }

    static bool dense_planar(const nnops::TensorView& v)
    {
        if (v.rank() < 2) { return true; }
        const int64_t elem = static_cast<int64_t>(
            nnops::data_type_size(v.data_type()));
        return v.layout() == nnops::TensorLayout::NCHW &&
               v.pitch() == v.shape(v.rank() - 1) * elem;
    }

    double atol_;
    double rtol_;
    bool verbose_;
    bool enabled_ = true;
    std::vector<CheckRow> rows_;
};

}  // namespace model
