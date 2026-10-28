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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace model {

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

    /// Compare @p actual against the artifact tensor @p ref_name.
    void compare(const std::string& name, const nnops::TensorView& actual,
                 const nnops::TensorView& expected)
    {
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
        for (int64_t i = 0; i < n; ++i) {
            const double want = e[i];
            const double err = std::fabs(static_cast<double>(a[i]) - want);
            row.max_abs = std::max(row.max_abs, err);
            row.ref_scale = std::max(row.ref_scale, std::fabs(want));
            if (err > atol_ + rtol_ * std::fabs(want)) { ++row.bad; }
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
    std::vector<CheckRow> rows_;
};

}  // namespace model
