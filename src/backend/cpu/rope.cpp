/// @file rope.cpp
/// @brief SIMD-optimized CPU implementation of RoPE (Rotary Position Embedding).
///
/// Builds cos/sin tables once for all positions (shared across heads and
/// batches), then rotates row by row with the kernels in
/// simd_kernel/simd_rope.hpp. f32 and f16 via dtype dispatch.
///
/// Table layout: table[pos * half + pair_idx], half = head_dim/2. Interleaved:
/// pair_idx = d/2, the pair is (2p, 2p+1). Split-half: pair_idx = d, the pair
/// is (d, d+half).
///
/// mRoPE: when mrope_section_dims is non-empty, head_dim is partitioned into
/// sections, each with its own base frequency.
///
/// Position offset: row s is at absolute position position_offset + s.
///
/// Scaling: Linear divides the angle by the factor; Dynamic NTK grows the
/// section's base with the run's end position past original_max_position; YaRN
/// mixes interpolated and extrapolated angles over a correction ramp and
/// multiplies cos/sin by the attention magnitude; Llama3 splits on the pair's
/// wavelength, scaling only the long end and blending the band between. All
/// four follow the HuggingFace rope_scaling formulas.

#include "nnops/ops/rope.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "simd_kernel/simd_rope.hpp"
#include "common/dtype_dispatch.hpp"
#include "common/memory_pool.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

struct RoPESection {
    int64_t dim_start;   // inclusive
    int64_t dim_end;     // exclusive (must be even)
    float   base;
};

std::vector<RoPESection> build_sections(const RoPEAttributes& attrs, int64_t head_dim) {
    std::vector<RoPESection> secs;
    if (attrs.mrope_section_dims.empty()) {
        secs.push_back({0, head_dim, attrs.base});
    } else {
        int64_t off = 0;
        for (size_t i = 0; i < attrs.mrope_section_dims.size(); ++i) {
            float b = (i < attrs.mrope_section_bases.size())
                ? attrs.mrope_section_bases[i] : attrs.base;
            secs.push_back({off, off + attrs.mrope_section_dims[i], b});
            off += attrs.mrope_section_dims[i];
        }
    }
    return secs;
}

/// Per-section constants the angle loop needs: the base already adjusted for
/// Dynamic NTK, and the YaRN ramp bounds in dim units.
struct SectionPlan {
    int64_t dim;
    int64_t pairs;
    float   base;
    float   low;
    float   high;
};

constexpr float kTwoPi = 6.28318530717958647692528677f;

/// Inverse of the "which dim rotates this many times per context window"
/// relation: a pair rotating `rotations` times over the pretraining window
/// sits at this dim. Both HF and llama.cpp spell it this way.
float correction_dim(int64_t dim, float rotations, float base, int64_t orig)
{
    return static_cast<float>(dim)
         * std::log(static_cast<float>(orig) / (rotations * kTwoPi))
         / (2.0f * std::log(base));
}

std::vector<SectionPlan> plan_sections(const RoPEAttributes& attrs,
                                       const std::vector<RoPESection>& sections,
                                       int64_t seq_len)
{
    std::vector<SectionPlan> plans;
    plans.reserve(sections.size());

    // Dynamic NTK pivots on the run's end position: with an offset the chunk
    // ends at offset + seq_len, which is the seq_len HF derives from
    // max(position_ids) + 1.
    const double end_pos = static_cast<double>(attrs.position_offset)
                         + static_cast<double>(seq_len);
    const double orig = static_cast<double>(attrs.original_max_position);

    for (const auto& sec : sections) {
        const int64_t sec_len = sec.dim_end - sec.dim_start;
        SectionPlan p{sec_len, sec_len / 2, sec.base, 0.0f, 0.0f};

        if (attrs.scaling_type == RoPEScalingType::Dynamic) {
            const double len = std::max(end_pos, orig);
            const double exp = sec_len > 2
                ? static_cast<double>(sec_len) / static_cast<double>(sec_len - 2)
                : 1.0;
            const double mult = attrs.scaling_factor * len / orig
                              - (attrs.scaling_factor - 1.0);
            p.base = static_cast<float>(sec.base * std::pow(mult, exp));
        } else if (attrs.scaling_type == RoPEScalingType::YaRN) {
            // Same truncation llama.cpp always applies: floor the fast bound,
            // ceil the slow one, then clamp into the dimension range.
            p.low = std::max(0.0f,
                std::floor(correction_dim(sec_len, attrs.yarn_beta_fast,
                                          sec.base, attrs.original_max_position)));
            p.high = std::min(static_cast<float>(sec_len - 1),
                std::ceil(correction_dim(sec_len, attrs.yarn_beta_slow,
                                         sec.base, attrs.original_max_position)));
        }
        plans.push_back(p);
    }
    return plans;
}

float yarn_attention_scale(const RoPEAttributes& attrs)
{
    if (attrs.attention_factor > 0.0f) { return attrs.attention_factor; }
    return attrs.scaling_factor > 1.0f
        ? 1.0f + 0.1f * std::log(attrs.scaling_factor)
        : 1.0f;
}

void compute_cos_sin_tables(
    PoolArray<float>& cos_table,
    PoolArray<float>& sin_table,
    int64_t seq_len,
    int64_t head_dim,
    const std::vector<RoPESection>& sections,
    const RoPEAttributes& attrs)
{
    const int64_t half = head_dim / 2;
    const size_t  total = static_cast<size_t>(seq_len) * static_cast<size_t>(half);
    cos_table.reset(total);
    sin_table.reset(total);

    const auto  plans      = plan_sections(attrs, sections, seq_len);
    const bool  linear     = attrs.scaling_type == RoPEScalingType::Linear;
    const bool  yarn       = attrs.scaling_type == RoPEScalingType::YaRN;
    const bool  llama3     = attrs.scaling_type == RoPEScalingType::Llama3;
    const float factor     = attrs.scaling_factor;
    const float magnitude  = yarn ? yarn_attention_scale(attrs) : 1.0f;

    // Llama3's two wavelength bands, both divisors of the pretraining window.
    const float orig       = static_cast<float>(attrs.original_max_position);
    const float low_wavelen  = orig / attrs.llama3_low_freq_factor;
    const float high_wavelen = orig / attrs.llama3_high_freq_factor;
    const float band = attrs.llama3_high_freq_factor - attrs.llama3_low_freq_factor;

    for (int64_t i = 0; i < seq_len; ++i) {
        const float pos_f = static_cast<float>(attrs.position_offset + i);
        float* cos_row = cos_table.data() + i * half;
        float* sin_row = sin_table.data() + i * half;

        int64_t pair_idx = 0;
        for (const auto& plan : plans) {
            const float sec_dim_f = static_cast<float>(plan.dim);

            for (int64_t p = 0; p < plan.pairs; ++p) {
                const float ratio = std::pow(plan.base,
                    2.0f * static_cast<float>(p) / sec_dim_f);
                const float extrap = pos_f / ratio;

                float theta = extrap;
                if (linear) {
                    theta = extrap / factor;
                } else if (yarn) {
                    // Ramp 0 keeps the extrapolated angle at the fast-rotating
                    // dims, 1 takes the interpolated one at the slow dims.
                    const float interp = extrap / factor;
                    const float ramp = std::clamp(
                        (static_cast<float>(p) - plan.low)
                            / std::max(0.001f, plan.high - plan.low),
                        0.0f, 1.0f);
                    theta = interp * ramp + extrap * (1.0f - ramp);
                } else if (llama3) {
                    // Short wavelengths are untouched, long ones divide by
                    // the factor, the band between them blends the two.
                    const float wavelen = kTwoPi * ratio;
                    if (wavelen > low_wavelen) {
                        theta = extrap / factor;
                    } else if (wavelen >= high_wavelen) {
                        const float smooth = std::clamp(
                            (orig / wavelen - attrs.llama3_low_freq_factor) / band,
                            0.0f, 1.0f);
                        theta = extrap * ((1.0f - smooth) / factor + smooth);
                    }
                }

                cos_row[pair_idx] = std::cos(theta) * magnitude;
                sin_row[pair_idx] = std::sin(theta) * magnitude;
                ++pair_idx;
            }
        }
    }
}

}  // anonymous namespace

template <typename T>
void rope_impl(const RoPEAttributes& attrs,
               TensorView& output,
               std::span<const TensorView> inputs,
               const ComputeContext& ctx)
{
    const auto& X = inputs[0];
    const int64_t rank = X.rank();
    NNOPS_ASSERT(rank >= 2);

    const int64_t seq_len = X.shape(rank - 2);
    const int64_t head_dim = X.shape(rank - 1);
    NNOPS_ASSERT(head_dim % 2 == 0);

    const int64_t half = head_dim / 2;

    int64_t outer_count = 1;
    for (int64_t i = 0; i < rank - 2; ++i) {
        outer_count *= X.shape(i);
    }

    auto sections = build_sections(attrs, head_dim);
    PoolArray<float> cos_table;
    PoolArray<float> sin_table;
    compute_cos_sin_tables(cos_table, sin_table, seq_len, head_dim, sections, attrs);

    const T* x_ptr   = X.ptr<T>();
    T*       y_ptr   = output.ptr<T>();
    const int64_t row_stride  = X.row_stride_elems();
    const int64_t out_row_stride = output.row_stride_elems();
    const bool    interleaved = attrs.interleaved;

    const auto process_outer = [&](int64_t outer) {
        const T* x_outer = x_ptr + outer * seq_len * row_stride;
        T*       y_outer = y_ptr + outer * seq_len * out_row_stride;

        for (int64_t pos = 0; pos < seq_len; ++pos) {
            const T* x_row = x_outer + pos * row_stride;
            T*       y_row = y_outer + pos * out_row_stride;
            const float* cos_row = cos_table.data() + pos * half;
            const float* sin_row = sin_table.data() + pos * half;

            kernel::rope_process_row<T>(
                y_row, x_row, head_dim, cos_row, sin_row,
                interleaved);
        }
    };

    ctx.cpu.run(0, outer_count, process_outer);
}

Status rope_cpu(const RoPEAttributes& attrs,
              TensorView& output,
              std::span<const TensorView> inputs,
              const ComputeContext& ctx,
              void* /*workspace*/)
{
    dispatch_f32_f16(inputs[0].data_type(), "rope_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        rope_impl<T>(attrs, output, inputs, ctx);
    });
    return Status::ok;
}

}  // namespace nnops::backend::cpu
