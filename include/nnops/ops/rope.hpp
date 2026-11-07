#pragma once
/// @file rope.hpp
/// @brief RoPE (Rotary Position Embedding) with mRoPE, position offset and
///        long-context scaling.
///
/// Rotates pairs of the last dim, head_dim (even), by position-dependent
/// angles: theta_i = pos / base^(2*i/rotary_dim), then
///   x'[i0] = x[i0]*cos(theta_i) - x[i1]*sin(theta_i)
///   x'[i1] = x[i1]*cos(theta_i) + x[i0]*sin(theta_i)
/// mRoPE splits rotary_dim into sections, each with its own base. Input and
/// output are [..., S, D].
///
/// Partial rotary: only the first rotary_dim lanes of each head rotate; the
/// rest are copied through. The frequency exponents keep rotary_dim in the
/// denominator, as in HuggingFace's `inv_freq` built over `rotary_dim`.
///
/// Optional fused Q/K normalization: when input [1] is present the row is
/// RMS-normalized over the whole head_dim before the rotation, with a per-lane
/// scale [1] and optional bias [2] — the q_norm / k_norm step Qwen-family
/// models run between the projection and RoPE. Fusing it here removes a full
/// tensor pass and the intermediate write.
///
/// Row s sits at absolute position position_offset + s, so a decode step can
/// rotate its new token where it really belongs instead of at 0.
///
/// Long-context scaling follows the HuggingFace `rope_scaling` conventions and
/// rewrites the same angle:
///   Linear   every inverse frequency is divided by scaling_factor.
///   Dynamic  NTK: the base grows once the run's end position passes
///            original_max_position; scaling_factor is the context multiplier
///            the model was trained for.
///   YaRN     NTK-by-parts: high-frequency pairs keep the original angle,
///            low-frequency pairs are interpolated, the ramp between
///            yarn_beta_fast and yarn_beta_slow deciding where, and cos/sin
///            carry a magnitude of 1 + 0.1*ln(scaling_factor) (or the explicit
///            attention_factor).
///   Llama3   wavelength bands: pairs whose wavelength stays under
///            original_max_position / high_freq_factor keep the original
///            angle, pairs beyond / low_freq_factor are scaled by the factor,
///            and the band between the two is interpolated smoothly.
/// With mRoPE sections each section scales independently.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace nnops {

enum class RoPEScalingType : uint8_t {
    None = 0,     ///< theta = pos / base^(2*i/head_dim)
    Linear = 1,   ///< inverse frequencies divided by the factor
    Dynamic = 2,  ///< NTK: base grows past original_max_position
    YaRN = 3,     ///< NTK-by-parts interpolation + attention magnitude
    Llama3 = 4,   ///< wavelength bands (Llama 3.1)
};

struct RoPEAttributes {
    /// Rotary base frequency (default 10000.0, standard for most LLMs).
    float base = 10000.0f;

    /// true: adjacent pairs (0,1), (2,3), ... (LLaMA); false: split-half pairs
    /// (0, R/2), (1, R/2+1), ... (GPT-NeoX), R = rotary_dim.
    bool interleaved = true;

    /// Lanes of each head that rotate; 0 means the whole head_dim. Must be
    /// even and no larger than head_dim; the lanes past it are copied through.
    int64_t rotary_dim = 0;

    /// mRoPE: per-section dim counts; the sum must equal rotary_dim.
    std::vector<int64_t> mrope_section_dims;

    /// mRoPE: per-section base frequencies, same size as mrope_section_dims.
    /// Empty means every section uses `base`.
    std::vector<float> mrope_section_bases;

    /// Absolute position of row 0; rows are consecutive.
    int64_t position_offset = 0;

    /// Long-context scaling; see the file header.
    RoPEScalingType scaling_type = RoPEScalingType::None;

    /// Context-extension factor (HF rope_scaling["factor"]); > 0, 1 = no-op.
    float scaling_factor = 1.0f;

    /// Pretraining context length, the pivot for Dynamic and YaRN; must be
    /// positive for those two.
    int64_t original_max_position = 0;

    /// YaRN ramp bounds in rotations (the paper's 32 and 1).
    float yarn_beta_fast = 32.0f;
    float yarn_beta_slow = 1.0f;

    /// YaRN cos/sin magnitude; 0 derives it as 1 + 0.1*ln(scaling_factor).
    float attention_factor = 0.0f;

    /// Llama3 band bounds as divisors of original_max_position; the paper's
    /// 1 and 4. high_freq_factor must exceed low_freq_factor.
    float llama3_low_freq_factor = 1.0f;
    float llama3_high_freq_factor = 4.0f;

    /// Epsilon of the fused RMSNorm (only read when input [1] is present).
    float norm_epsilon = 1e-5f;
};

class RoPE : public OpBase {
public:
    static std::unique_ptr<RoPE> create(const RoPEAttributes& attrs = {},
                                         Backend backend = Backend::CPU);

    static std::unique_ptr<RoPE> create(Backend backend) {
        return create(RoPEAttributes{}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    Status validate(std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs) const override;

    using OpBase::compute;

    Status compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::RoPE; }
    Backend getBackend() const override { return backend_; }

    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::ChannelFirstOnly;
    }

    const RoPEAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in rope.cpp
    ~RoPE();

private:
    RoPE(const RoPEAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    RoPEAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
