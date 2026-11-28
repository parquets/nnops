#pragma once
/// @file rope_plan.hpp
/// @brief The host-side RoPE pair plan — the per-pair frequency ratio and the
///        long-context scaling blend — shared by the CPU and CUDA routes.
///
/// The angle of pair p at position s is a function of the attributes alone plus
/// s, so every constant that does not depend on the position is computed once
/// per call and carried into the table build as four values a pair. The table
/// used to re-evaluate `base^(2p/dim)` and the scaling ramps for every
/// (position, pair), which at S=2048, half=64 was 131k `pow` calls per RoPE call
/// and dominated the op once the rotation pass became parallel.
///
/// Both backends build the same plan on the host: the CPU from it computes
/// cos/sin rows, the CUDA route uploads it and its table kernel reads it. The
/// scaling formulas follow HuggingFace's rope_scaling (and llama.cpp where the
/// two spell a bound differently):
///
///   Dynamic  base grows once the run's end position passes
///            original_max_position.
///   YaRN     a ramp over [beta_fast, beta_slow] blends the interpolated and
///            extrapolated angles, and cos/sin carry a magnitude.
///   Llama3   pairs are split on wavelength: the short end keeps the original
///            angle, the long end divides by the factor, the band blends.

#include "nnops/ops/rope.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace nnops::backend {

inline constexpr float kRopeTwoPi = 6.28318530717958647692528677f;

/// One mRoPE section after the scaling type rewrote its base and, for YaRN, its
/// ramp bounds. `pairs` is dim / 2.
struct RopeSection {
    int64_t dim = 0;
    int64_t pairs = 0;
    float base = 10000.0f;
    float low = 0.0f;
    float high = 0.0f;
};

/// What one pair costs per position, split into the part that does not depend
/// on the position: the wavelength `ratio` and the scaling blend.
struct RopePairPlan {
    float ratio = 1.0f; ///< base^(2p/section_dim): the angle is pos / ratio
    float ramp = 0.0f;  ///< YaRN: 0 keeps the extrapolated angle, 1 the interpolated one
    float smooth = 0.0f; ///< Llama3 band blend, -1 when the pair sits outside the band
    bool long_wl = false; ///< Llama3: wavelength past the short end, so it is divided
};

/// Inverse of the "which dim rotates this many times per context window"
/// relation: a pair rotating `rotations` times over the pretraining window sits
/// at this dim. Both HF and llama.cpp spell it this way.
inline float rope_correction_dim(int64_t dim, float rotations, float base, int64_t orig)
{
    return static_cast<float>(dim) * std::log(static_cast<float>(orig) / (rotations * kRopeTwoPi)) /
           (2.0f * std::log(base));
}

/// Split @p rotary_dim into sections (mRoPE's dims/bases, or one section) and
/// apply the scaling type's base rewrite. Dynamic NTK pivots on the run's end
/// position — `max_start` + `seq_len`, which is the seq_len HF derives from
/// max(position_ids) + 1.
inline std::vector<RopeSection> plan_rope_sections(const RoPEAttributes& attrs, int64_t rotary_dim,
                                                   int64_t seq_len, int64_t max_start)
{
    std::vector<RopeSection> out;
    std::vector<std::pair<int64_t, float>> raw; // (dim, base)
    if (attrs.mrope_section_dims.empty()) {
        raw.push_back({rotary_dim, attrs.base});
    } else {
        for (size_t i = 0; i < attrs.mrope_section_dims.size(); ++i) {
            const float b =
                i < attrs.mrope_section_bases.size() ? attrs.mrope_section_bases[i] : attrs.base;
            raw.push_back({attrs.mrope_section_dims[i], b});
        }
    }

    const double end_pos = static_cast<double>(max_start) + static_cast<double>(seq_len);
    const double orig = static_cast<double>(attrs.original_max_position);
    out.reserve(raw.size());
    for (const auto& [sec_dim, sec_base] : raw) {
        RopeSection s;
        s.dim = sec_dim;
        s.pairs = sec_dim / 2;
        s.base = sec_base;
        if (attrs.scaling_type == RoPEScalingType::Dynamic) {
            const double len = std::max(end_pos, orig);
            const double exp = sec_dim > 2 ? static_cast<double>(sec_dim) /
                                                 static_cast<double>(sec_dim - 2)
                                           : 1.0;
            const double mult = attrs.scaling_factor * len / orig - (attrs.scaling_factor - 1.0);
            s.base = static_cast<float>(sec_base * std::pow(mult, exp));
        } else if (attrs.scaling_type == RoPEScalingType::YaRN) {
            // Same truncation llama.cpp always applies: floor the fast bound,
            // ceil the slow one, then clamp into the dimension range.
            s.low = std::max(0.0f, std::floor(rope_correction_dim(
                                       sec_dim, attrs.yarn_beta_fast, sec_base,
                                       attrs.original_max_position)));
            s.high = std::min(static_cast<float>(sec_dim - 1),
                              std::ceil(rope_correction_dim(sec_dim, attrs.yarn_beta_slow, sec_base,
                                                            attrs.original_max_position)));
        }
        out.push_back(s);
    }
    return out;
}

/// The per-pair plan, one entry per rotary pair (half = rotary_dim / 2).
inline std::vector<RopePairPlan> build_rope_pair_plans(const std::vector<RopeSection>& sections,
                                                       const RoPEAttributes& attrs, int64_t half)
{
    std::vector<RopePairPlan> pairs(static_cast<size_t>(half));
    const bool yarn = attrs.scaling_type == RoPEScalingType::YaRN;
    const bool llama3 = attrs.scaling_type == RoPEScalingType::Llama3;

    // Llama3's two wavelength bands, both divisors of the pretraining window.
    const float orig = static_cast<float>(attrs.original_max_position);
    const float low_wavelen = orig / attrs.llama3_low_freq_factor;
    const float high_wavelen = orig / attrs.llama3_high_freq_factor;
    const float band = attrs.llama3_high_freq_factor - attrs.llama3_low_freq_factor;

    size_t idx = 0;
    for (const RopeSection& sec : sections) {
        const float dim_f = static_cast<float>(sec.dim);
        for (int64_t p = 0; p < sec.pairs && idx < pairs.size(); ++p, ++idx) {
            RopePairPlan pp;
            pp.ratio = std::pow(sec.base, 2.0f * static_cast<float>(p) / dim_f);
            if (yarn) {
                pp.ramp = std::clamp((static_cast<float>(p) - sec.low) /
                                         std::max(0.001f, sec.high - sec.low),
                                     0.0f, 1.0f);
            } else if (llama3) {
                const float wavelen = kRopeTwoPi * pp.ratio;
                pp.long_wl = wavelen > low_wavelen;
                if (!pp.long_wl && wavelen >= high_wavelen) {
                    pp.smooth = std::clamp(
                        (orig / wavelen - attrs.llama3_low_freq_factor) / band, 0.0f, 1.0f);
                } else {
                    pp.smooth = -1.0f;
                }
            }
            pairs[idx] = pp;
        }
    }
    return pairs;
}

/// YaRN's cos/sin magnitude: the explicit attention_factor, else
/// 1 + 0.1·ln(scaling_factor) (1 when the factor does not grow the context).
inline float yarn_attention_scale(const RoPEAttributes& attrs)
{
    if (attrs.attention_factor > 0.0f) {
        return attrs.attention_factor;
    }
    return attrs.scaling_factor > 1.0f ? 1.0f + 0.1f * std::log(attrs.scaling_factor) : 1.0f;
}

} // namespace nnops::backend
