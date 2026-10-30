#pragma once
/// @file rope.hpp
/// @brief RoPE (Rotary Position Embedding) with mRoPE (multi-modal) support.
///
/// Rotates pairs of the last dim, head_dim (even), by position-dependent
/// angles: theta_i = pos / base^(2*i/head_dim), then
///   x'[i0] = x[i0]*cos(theta_i) - x[i1]*sin(theta_i)
///   x'[i1] = x[i1]*cos(theta_i) + x[i0]*sin(theta_i)
/// mRoPE splits head_dim into sections, each with its own base. Input and
/// output are [..., S, D].

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace nnops {

struct RoPEAttributes {
    /// Rotary base frequency (default 10000.0, standard for most LLMs).
    float base = 10000.0f;

    /// true: adjacent pairs (0,1), (2,3), ... (LLaMA); false: split-half pairs
    /// (0, D/2), (1, D/2+1), ... (GPT-NeoX).
    bool interleaved = true;

    /// mRoPE: per-section dim counts; the sum must equal head_dim.
    std::vector<int64_t> mrope_section_dims;

    /// mRoPE: per-section base frequencies, same size as mrope_section_dims.
    /// Empty means every section uses `base`.
    std::vector<float> mrope_section_bases;
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
        return LayoutSupport::PlanarOnly;
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
