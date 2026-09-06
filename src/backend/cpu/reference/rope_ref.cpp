/// @file rope_ref.cpp
/// @brief Naive CPU reference implementation of RoPE (Rotary Position Embedding).
///
/// Pure scalar computation for correctness baseline. Supports both interleaved
/// (LLaMA) and split-half (GPT-NeoX) pairing conventions, as well as mRoPE
/// with per-section base frequencies.

#include "nnops/ops/rope.hpp"
#include "nnops/core/parallel_for.hpp"

#include <cmath>
#include <vector>

namespace nnops::backend::cpu::reference {

namespace {

struct RoPESection {
    int64_t dim_start;
    int64_t dim_end;
    float   base;
};

/// Build section table from attributes. Returns a single section covering
/// [0, head_dim) for standard RoPE, or one section per mrope entry.
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

/// Find which section a given dimension index belongs to, and return the
/// local_pair index within that section (0-based, counts by pair).
struct SectionLookup {
    const RoPESection* sec;
    int64_t local_pair;  // pair index within this section
};

SectionLookup find_section(int64_t d, const std::vector<RoPESection>& sections) {
    for (const auto& sec : sections) {
        if (d >= sec.dim_start && d < sec.dim_end) {
            return {&sec, (d - sec.dim_start) / 2};
        }
    }
    // Fallback (should not happen with valid input)
    return {&sections[0], d / 2};
}

}  // anonymous namespace

void rope_ref(const RoPEAttributes& attrs,
              TensorView& output,
              std::span<const TensorView> inputs,
              const ComputeContext& ctx,
              void* /*workspace*/)
{
    const auto& X = inputs[0];
    const int64_t rank = X.rank();
    NNOPS_ASSERT(rank >= 2);

    const int64_t seq_len = X.shape(rank - 2);
    const int64_t head_dim = X.shape(rank - 1);
    NNOPS_ASSERT(head_dim % 2 == 0);

    // Flatten leading dims → single outer loop
    int64_t outer_count = 1;
    for (int64_t i = 0; i < rank - 2; ++i) {
        outer_count *= X.shape(i);
    }

    const int64_t row_stride = X.row_stride_elems();
    const auto* x_ptr = X.ptr<float>();
    auto* y_ptr = output.ptr<float>();

    // Build mRoPE section table
    auto sections = build_sections(attrs, head_dim);

    const bool interleaved = attrs.interleaved;
    const int64_t half     = head_dim / 2;

    const auto process_outer = [&](int64_t outer) {
        const float* x_outer = x_ptr + outer * seq_len * row_stride;
        float*       y_outer = y_ptr + outer * seq_len * row_stride;

        for (int64_t pos = 0; pos < seq_len; ++pos) {
            const float* x_row = x_outer + pos * row_stride;
            float*       y_row = y_outer + pos * row_stride;
            const float  pos_f = static_cast<float>(pos);

            if (interleaved) {
                for (int64_t d = 0; d < head_dim; d += 2) {
                    auto [sec, local_pair] = find_section(d, sections);
                    float sec_dim = static_cast<float>(sec->dim_end - sec->dim_start);
                    float theta = pos_f / std::pow(sec->base,
                        2.0f * static_cast<float>(local_pair) / sec_dim);
                    float cv = std::cos(theta);
                    float sv = std::sin(theta);

                    float x0 = x_row[d];
                    float x1 = x_row[d + 1];
                    float y0 = x0 * cv - x1 * sv;
                    float y1 = x1 * cv + x0 * sv;

                    y_row[d]     = y0;
                    y_row[d + 1] = y1;
                }
            } else {
                for (int64_t d = 0; d < half; ++d) {
                    auto [sec, local_pair] = find_section(d, sections);
                    float sec_dim = static_cast<float>(sec->dim_end - sec->dim_start);
                    float theta = pos_f / std::pow(sec->base,
                        2.0f * static_cast<float>(local_pair) / sec_dim);
                    float cv = std::cos(theta);
                    float sv = std::sin(theta);

                    float x0 = x_row[d];
                    float x1 = x_row[d + half];
                    float y0 = x0 * cv - x1 * sv;
                    float y1 = x1 * cv + x0 * sv;

                    y_row[d]        = y0;
                    y_row[d + half] = y1;
                }
            }
        }
    };

    ctx.cpu.run(0, outer_count, process_outer);
}

}  // namespace nnops::backend::cpu::reference
