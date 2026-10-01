#pragma once
/// @file rope.hpp
/// @brief RoPE (Rotary Position Embedding) operator with mRoPE (multi-modal) support.
///
/// Applies rotary position encoding to query/key tensors. Supports two common
/// dimension pairing conventions and multi-modal frequency scheduling (mRoPE).

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace nnops {

/// Attributes for the RoPE operator.
///
/// For standard RoPE (single base frequency):
///   theta_i = pos / base^(2*i/head_dim)
///   x'[i0] = x[i0] * cos(theta_i) - x[i1] * sin(theta_i)
///   x'[i1] = x[i1] * cos(theta_i) + x[i0] * sin(theta_i)
///
/// For mRoPE (multi-modal, when mrope_section_dims is non-empty):
///   Head_dim is partitioned into sections, each with its own base frequency.
///   Sections use their individual base value for frequency computation.
///
/// Input:  X [..., S, D]   (S = sequence length, D = head_dim, D % 2 == 0)
/// Output: Y [..., S, D]   (same shape and dtype as input)
struct RoPEAttributes {
    /// Rotary base frequency (default 10000.0, standard for most LLMs).
    float base = 10000.0f;

    /// If true, adjacent elements form rotation pairs: (0,1), (2,3), ...
    /// (LLaMA convention). If false, the first half pairs with the second
    /// half: (0, D/2), (1, D/2+1), ... (GPT-NeoX convention).
    bool interleaved = true;

    /// mRoPE: dimension counts for each section. When non-empty, head_dim is
    /// partitioned into sections, and each section uses a potentially different
    /// base frequency. The sum of all entries must equal head_dim.
    ///
    /// Example: sections = {64, 32, 32} means dims [0,64), [64,96), [96,128).
    std::vector<int64_t> mrope_section_dims;

    /// mRoPE: per-section base frequencies. Must have the same size as
    /// mrope_section_dims when non-empty. If mrope_section_dims is non-empty
    /// but this is empty, all sections use `base`.
    std::vector<float> mrope_section_bases;
};

/// Rotary Position Embedding operator.
///
/// Applies 2D rotation to the last dimension (head_dim) of the input tensor,
/// using position-dependent angles. The rotation angle for each dimension
/// pair is determined by the position index (0, 1, ..., seq_len-1) along
/// the second-to-last dimension.
class RoPE : public OpBase {
public:
    /// Create a RoPE operator for the specified backend.
    static std::unique_ptr<RoPE> create(const RoPEAttributes& attrs = {},
                                         Backend backend = Backend::CPU);

    /// Create with defaults.
    static std::unique_ptr<RoPE> create(Backend backend) {
        return create(RoPEAttributes{}, backend);
    }

    // ---- OpBase interface ----
    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    /// inputs[0] = X [..., S, D]
    /// outputs[0] = Y [..., S, D]
    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::RoPE; }
    Backend getBackend() const override { return backend_; }

    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::PlanarOnly;
    }

    const RoPEAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in rope.cpp (Pimpl pattern)
    ~RoPE();

private:
    RoPE(const RoPEAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    RoPEAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
