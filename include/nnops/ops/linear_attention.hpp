#pragma once
/// @file linear_attention.hpp
/// @brief LinearAttention operator — gated linear attention with recurrent state.
///
/// Implements O(n) linear attention (Gated Delta Net / KDA) as an alternative
/// to O(n²) softmax attention. Instead of an explicit KV-cache, a fixed-size
/// recurrent hidden state matrix [head_dim, head_dim] per head is maintained
/// and updated at each step.
///
/// The operator is stateless: all recurrent state memory is owned by the caller
/// and passed as input/output tensors.
///
/// Reference: llama.cpp ggml_gated_linear_attn (ggml.h:2523)

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes for the LinearAttention (gated linear attention) operator.
///
/// Static configuration set at operator creation time. Dynamic per-step
/// values (token count, sequence count) are derived from input shapes.
struct LinearAttentionAttributes {
    /// Per-head state dimension (S). Same dimension for Q, K, V, and Gate.
    /// This is the key dimension that determines the recurrent state size.
    int64_t head_dim = 64;

    /// Number of query attention heads (H).
    int64_t num_heads = 8;

    /// Number of key/value heads for GQA (Grouped Query Attention).
    /// When 0 (default), equals num_heads (standard, no grouping).
    /// When < num_heads, each KV head is shared by num_heads/num_kv_heads query heads.
    int64_t num_kv_heads = 0;

    /// Scaling factor. If 0, auto-computed as 1/sqrt(head_dim).
    float scale = 0.0f;

    /// Short causal convolution kernel size applied to Q, K, V before the
    /// recurrent update. Set to 0 to disable.
    int64_t conv_kernel_size = 4;
};

/// Gated Linear Attention with recurrent state (class-based API).
///
/// Computes one step of O(n) recurrent linear attention:
///   1. Optionally applies causal conv1d to Q, K, V
///   2. Computes gated recurrence: S_new = S * exp(gate) + K^T * (V - K*S) * beta
///   3. Reads output: O = Q * S_new
///
/// Unlike CausalAttention which stores all past K/V in a growing cache,
/// LinearAttention maintains a fixed-size recurrent state matrix [D, D] per head.
/// State is placed in outputs (like K_cache / V_cache in CausalAttention):
/// the caller initializes it once (to zeros or a checkpoint), then passes the
/// same buffer every step. The kernel reads the old state and overwrites it
/// with the new state in-place.
///
/// Inputs (4):
///   inputs[0] = Q     [B, H,    Sq, D]    query
///   inputs[1] = K     [B, H_kv, Sk, D]    key
///   inputs[2] = V     [B, H_kv, Sk, D]    value
///   inputs[3] = Gate  [B, H,    Sq, D]    decay/forget gate
///
/// Outputs (2):
///   outputs[0] = Output [B, H,    Sq, D]   attention result
///   outputs[1] = State  [B, H_kv, D,  D]   recurrent state (read then overwritten)
class LinearAttention : public OpBase {
public:
    /// Create a LinearAttention operator for the specified backend.
    static std::unique_ptr<LinearAttention> create(
        const LinearAttentionAttributes& attrs,
        Backend backend = Backend::CPU);

    /// Create with defaults.
    static std::unique_ptr<LinearAttention> create(Backend backend = Backend::CPU) {
        return create(LinearAttentionAttributes{}, backend);
    }

    // ---- OpBase interface ----
    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::LinearAttention; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::PlanarOnly;
    }

    const LinearAttentionAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in linear_attention.cpp (Pimpl pattern)
    ~LinearAttention();

private:
    LinearAttention(const LinearAttentionAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    LinearAttentionAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
